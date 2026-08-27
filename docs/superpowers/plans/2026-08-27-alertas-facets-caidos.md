# Alertas de facets caídos — plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** que un facet caído en la Mesa web se detecte y avise por
Telegram en minutos, en vez de tres días o nunca.

**Architecture:** una tabla `facet_health_event` es la fuente única de
verdad; el chokepoint `_invoke_facet` la escribe con un `outcome` tipado;
una sonda horaria (más una por rebinding) garantiza que haya eventos aun
sin usuarios; el reaper de `las_manos` es el único lector, calcula estado
por facet y alerta en transiciones.

**Tech Stack:** Python 3.12 (`jax`) / 3.14 (`jax-platform`), FastAPI,
aiomysql, MariaDB 11.8 (`jax_memory`), pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-08-27-alertas-facets-caidos-design.md`
(mergeado en `master` como `0de626b`). El plan argumenta desde el spec;
los ejecutores leen los dos.

---

## Global Constraints

Aplican a **todas** las tasks. No se repiten en cada brief.

- **Ningún test puede disparar una llamada real a un proveedor.** Ver
  §"Riesgo de costo" abajo. Es la restricción más importante del plan.
- **`ts` es epoch `DOUBLE`**, nunca `TIMESTAMP`. Comparaciones siempre
  contra epoch, jamás contra un string de fecha.
- **Fail-closed:** ningún `except` nuevo puede convertir un error en
  "seguí igual". Todo `except` fail-soft lleva el comentario marcador
  `# fail-soft: <motivo>`, o el scanner P10 del CI lo rechaza.
- **`detail` es diagnóstico interno**, nunca se muestra en UI: no entra
  en i18n. Ningún string de esta feature es visible al usuario final.
- **Sin hardcoding:** la lista de facets sale de
  `config["personalities"]`, nunca de un literal en código.
- **Backup antes de modificar** cualquier archivo existente (regla del
  carpintero).
- `py_compile` / suite verde antes de cerrar cada task.
- **Un "Expected" del brief que no se cumple → PARAR y reportar.** Nunca
  ajustar el código, ni el test, ni el Expected, para que dé.

### Riesgo de costo — precedente real, no hipotético

La sonda hace **llamadas pagas a proveedores reales** (openai, gemini,
deepseek, zhipu). Ya pasó exactamente el accidente que hay que prevenir:

`las_manos/_motor_v02_test.py` (hoy cuarentenado como
`scripts/manual_motor_v02_integration.py`) no era un test: era un script
manual de integración cuyo nombre matcheaba el patrón de descubrimiento
de pytest (`*_test.py`), **con todo el código a nivel de módulo**, sin
`if __name__ == "__main__":`. Cualquier `pytest` corrido sobre
`las_manos/` lo importaba y disparaba un dispatch real a Kimi contra el
LAS MANOS de **producción**, con polling de hasta 120 s, y —si llegaba a
completar— activaba el kill switch `/etc/jax/PAUSE` vía sudo.
**11 dispatches reales confirmados por `journalctl`**, disparados por
correr la suite durante el propio diagnóstico (2026-08-24).

El mecanismo fue: *nombre descubrible por pytest* + *código a nivel de
módulo*. Reglas que salen de ahí, obligatorias en este plan:

1. **Todo test de la sonda parchea el I/O.** `_invoke_facet` se
   monkeypatchea; nunca se llama la de verdad en un test.
2. **Ningún código a nivel de módulo** en archivos descubribles por
   pytest ejecuta una sonda, arranca un loop, o toca la red.
3. **El loop de la sonda nunca arranca bajo pytest.** Task 4 le pone un
   guard explícito y un test que lo verifica.
4. Si alguna vez hace falta un test con llamada real, va **fuera** del
   patrón de descubrimiento (`scripts/manual_*.py`), marcado en el
   nombre, y **nunca** se agrega al CI. Hoy no hace falta ninguno.

### Orden de deploy — INVERSO al de la ronda anterior

**Verificado, no heredado.** La ronda de `_HTTP_FACETS` (2026-08-27)
exigía `jax-las-manos` **primero** porque `jax-platform` llamaba un
endpoint nuevo de `las_manos`. **Acá la dependencia va al revés:**

- `run_migrations()` corre en el `lifespan` de `jax-platform`
  (`backend/main.py:84`) — o sea, las tablas `facet_health_event` y
  `facet_health_alert` **se crean al reiniciar `jax-platform`**.
- El lector vive en el reaper de `las_manos`, que **lee** esas tablas.

Por lo tanto: **`jax-platform` PRIMERO, `jax-las-manos` DESPUÉS.**

Si se copiara el orden de la ronda anterior, el reaper arrancaría
consultando una tabla inexistente: error en cada barrido, y —peor— cero
eventos leídos se interpretan como `unknown`, así que la primera acción
del sistema nuevo sería una alerta falsa de "la sonda no está corriendo".
**Rollback:** el inverso (`jax-las-manos` primero, `jax-platform`
después), por la misma razón.

Esto está escrito acá, antes de la Task 8, a propósito: se decide en el
plan, no se descubre en el deploy.

---

## File Structure

**`jax-platform` (escritor + sonda):**

| Archivo | Responsabilidad |
|---|---|
| `backend/db/migrations.py` *(modificar)* | DDL de las dos tablas nuevas |
| `backend/facet_health.py` *(crear)* | `record_facet_health()` — único escritor |
| `backend/api/chat.py` *(modificar)* | partición `_invoke_facet_dispatch` + envoltorio instrumentado |
| `backend/jax_engine/facet_canary.py` *(crear)* | sonda periódica + sonda por rebinding |
| `backend/main.py` *(modificar)* | arrancar el loop de la sonda |
| `backend/api/admin/models.py` *(modificar)* | hook de rebinding (escritor 1) |
| `backend/api/admin/facet_bindings.py` *(modificar)* | hook de rebinding (escritor 2) |
| `backend/tests/test_facet_health_writer.py` *(crear)* | tests del escritor |
| `backend/tests/test_facet_canary.py` *(crear)* | tests de la sonda |

**`jax` (lector + alerta):**

| Archivo | Responsabilidad |
|---|---|
| `jacobs/facet_health.py` *(crear)* | máquina de estados pura + lectura + alerta |
| `jacobs/reaper.py` *(modificar)* | invocar el chequeo en cada barrido |
| `jacobs/_facet_health_test.py` *(crear)* | tests de la máquina de estados |
| `.github/workflows/policy.yml` *(modificar)* | job `facet-health-unit` |

---

## Task 1 — Migración: las dos tablas

**Files:**
- Modify: `jax-platform/backend/db/migrations.py` (constantes DDL + `_TABLES`)

**Interfaces:**
- Consumes: nada.
- Produces: tablas `facet_health_event` y `facet_health_alert` en
  `jax_memory`, creadas por `run_migrations()`.

- [ ] **Step 1: Backup**

```bash
cd /home/fruiz/jax-platform && cp backend/db/migrations.py backend/db/migrations.py.backup-pre-facethealth-$(date +%Y%m%d-%H%M%S)
```

- [ ] **Step 2: Agregar las constantes DDL**

Junto a las otras constantes `CREATE_*` de `migrations.py`:

```python
CREATE_FACET_HEALTH_EVENT = """
CREATE TABLE facet_health_event (
    id      BIGINT AUTO_INCREMENT PRIMARY KEY,
    facet   VARCHAR(50) NOT NULL,
    outcome ENUM('ok','provider_error','gate_denied','gate_unreachable',
                 'unbound','unsupported_transport','probe_error') NOT NULL,
    source  ENUM('chat','canary_periodic','canary_rebind') NOT NULL,
    detail  VARCHAR(255) NULL,
    ts      DOUBLE NOT NULL,
    KEY idx_facet_ts (facet, ts),
    KEY idx_ts (ts)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

# `ts` es epoch DOUBLE, no TIMESTAMP: misma decision y misma razon que
# jacobs_events.ts -- inmune a la timezone de sesion. La limpieza de
# axioma_usage del 2026-08-21 perdio 90 de 106 filas comparando un
# TIMESTAMP contra un string de fecha.

CREATE_FACET_HEALTH_ALERT = """
CREATE TABLE facet_health_alert (
    facet         VARCHAR(50) PRIMARY KEY,
    state         ENUM('ok','down','unknown') NOT NULL,
    first_seen_ts DOUBLE NOT NULL,
    notified_ts   DOUBLE NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""

# facet_health_alert NO es una segunda fuente de verdad de salud: es el
# registro de que ya se aviso. La salud se calcula exclusivamente desde
# facet_health_event. La clave centinela '__system__' guarda el estado de
# la alerta agregada (sonda entera caida) para que pase por la MISMA
# supresion que las de facet -- sin eso, una sonda muerta un viernes
# produce un mensaje por barrido, 288 el sabado.
```

- [ ] **Step 3: Registrarlas en `_TABLES`**

Al final de la lista `_TABLES` (`migrations.py:427-448`). No tienen FK,
así que el orden relativo no importa:

```python
    ("facet_health_event", CREATE_FACET_HEALTH_EVENT),
    ("facet_health_alert", CREATE_FACET_HEALTH_ALERT),
```

- [ ] **Step 4: Verificar que compila**

Run: `cd /home/fruiz/jax-platform/backend && python -m py_compile db/migrations.py`
Expected: sin salida, exit 0.

- [ ] **Step 5: Verificar idempotencia sin tocar producción**

`run_migrations()` usa `_table_exists()` antes de cada `CREATE`, así que
correrlo dos veces es seguro. Verificarlo leyendo `migrations.py:1409-1411`
y confirmando que las dos entradas nuevas pasan por ese mismo bucle.

Expected: las dos entradas están dentro del `for table_name, ddl in _TABLES`
guardado por `_table_exists`. **Si no lo están, PARAR y reportar.**

- [ ] **Step 6: Commit**

```bash
cd /home/fruiz/jax-platform
git add backend/db/migrations.py
git commit -m "feat(health): tablas facet_health_event y facet_health_alert"
```

---

## Task 2 — Escritor: `record_facet_health()`

**Files:**
- Create: `jax-platform/backend/facet_health.py`
- Test: `jax-platform/backend/tests/test_facet_health_writer.py`

**Interfaces:**
- Consumes: `db.connection.get_pool` (ya existe, lo usa `model_catalog.py`).
- Produces:
  ```python
  OUTCOMES: frozenset[str]   # los 7 valores del ENUM
  SOURCES: frozenset[str]    # {'chat','canary_periodic','canary_rebind'}
  async def record_facet_health(
      facet: str, outcome: str, source: str, detail: str | None = None
  ) -> bool
  ```
  Devuelve `True` si escribió, `False` si falló (fail-soft). Task 3 y
  Task 4 la consumen.

- [ ] **Step 1: Escribir el test que falla**

`backend/tests/test_facet_health_writer.py`:

```python
"""Escritor de facet_health_event. TODO el I/O esta parcheado: este
archivo NUNCA toca la DB real ni la red. Ver "Riesgo de costo" en el plan
(incidente _motor_v02_test.py, 2026-08-24: 11 dispatches reales a
produccion disparados por correr pytest)."""
import pytest
import facet_health


class _FakeCursor:
    def __init__(self, sink): self.sink = sink
    async def execute(self, sql, params=None): self.sink.append((sql, params))
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False


class _FakeConn:
    def __init__(self, sink): self.sink = sink
    def cursor(self): return _FakeCursor(self.sink)
    async def commit(self): self.sink.append(("COMMIT", None))
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False


class _FakePool:
    def __init__(self, sink): self.sink = sink
    def acquire(self): return _FakeConn(self.sink)


def test_record_escribe_una_fila_con_epoch(monkeypatch):
    sink = []
    async def fake_pool(): return _FakePool(sink)
    monkeypatch.setattr(facet_health, "get_pool", fake_pool)

    import asyncio
    ok = asyncio.run(facet_health.record_facet_health(
        "thot", "provider_error", "chat", "HTTPStatusError: 502"))

    assert ok is True
    inserts = [s for s in sink if s[0] != "COMMIT"]
    assert len(inserts) == 1
    sql, params = inserts[0]
    assert "INSERT INTO facet_health_event" in sql
    assert params[0] == "thot"
    assert params[1] == "provider_error"
    assert params[2] == "chat"
    assert isinstance(params[4], float)   # ts epoch, NO string de fecha


def test_record_rechaza_outcome_invalido(monkeypatch):
    sink = []
    async def fake_pool(): return _FakePool(sink)
    monkeypatch.setattr(facet_health, "get_pool", fake_pool)

    import asyncio
    with pytest.raises(ValueError):
        asyncio.run(facet_health.record_facet_health("thot", "inventado", "chat"))
    assert sink == []   # no escribio nada


def test_record_es_fail_soft_ante_error_de_db(monkeypatch, caplog):
    async def fake_pool(): raise RuntimeError("db caida")
    monkeypatch.setattr(facet_health, "get_pool", fake_pool)

    import asyncio
    ok = asyncio.run(facet_health.record_facet_health("thot", "ok", "chat"))

    assert ok is False              # no revienta el turno de chat
    assert caplog.records           # pero NO es silencioso
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `cd /home/fruiz/jax-platform/backend && python -m pytest tests/test_facet_health_writer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'facet_health'`.

- [ ] **Step 3: Implementar**

`backend/facet_health.py`:

```python
"""Escritor de facet_health_event -- unico lugar donde se registra el
resultado de una invocacion de facet. La SALUD se calcula en otro lado
(jax/jacobs/facet_health.py, en el reaper): aca solo se escriben hechos.
Ver docs/superpowers/specs/2026-08-27-alertas-facets-caidos-design.md."""
import logging
import time

from db.connection import get_pool

logger = logging.getLogger(__name__)

OUTCOMES = frozenset({
    "ok", "provider_error", "gate_denied", "gate_unreachable",
    "unbound", "unsupported_transport", "probe_error",
})
SOURCES = frozenset({"chat", "canary_periodic", "canary_rebind"})

_DETAIL_MAX = 255


async def record_facet_health(
    facet: str, outcome: str, source: str, detail: str | None = None
) -> bool:
    """Escribe UNA fila. Devuelve True si escribio, False si fallo.

    Fail-soft ante error de DB a proposito: un fallo registrando salud no
    puede tumbar un turno de chat que ya respondio. Pero NUNCA silencioso
    -- loguea, y la ausencia de la fila se convierte rio abajo en
    `unknown`, jamas en `ok` (ver el lector en jax/jacobs/facet_health.py).

    Un `outcome` o `source` invalido SI lanza: es un bug del llamador, no
    una condicion de runtime, y enmascararlo dejaria filas que el ENUM
    rechazaria en silencio."""
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome invalido: {outcome!r}")
    if source not in SOURCES:
        raise ValueError(f"source invalido: {source!r}")

    if detail is not None:
        detail = detail[:_DETAIL_MAX]

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO facet_health_event "
                    "(facet, outcome, source, detail, ts) VALUES (%s,%s,%s,%s,%s)",
                    (facet, outcome, source, detail, time.time()),
                )
            await conn.commit()
        return True
    except Exception:  # fail-soft: registrar salud no puede tumbar un turno de chat ya respondido; la ausencia de fila se lee como `unknown` rio abajo, nunca como `ok`
        logger.warning(
            "facet_health: no se pudo registrar facet=%s outcome=%s source=%s",
            facet, outcome, source, exc_info=True,
        )
        return False
```

- [ ] **Step 4: Correr los tests y verificar que pasan**

Run: `cd /home/fruiz/jax-platform/backend && python -m pytest tests/test_facet_health_writer.py -v`
Expected: 3 passed.

- [ ] **Step 5: Verificar que el scanner P10 acepta el `except`**

Run: `cd /home/fruiz/jax-platform/backend && python -m pytest tests/test_no_fail_open_except.py -v`
Expected: PASS. **Si falla, el marcador `# fail-soft:` está mal escrito —
PARAR y reportar, no relajar el scanner.**

- [ ] **Step 6: Commit**

```bash
cd /home/fruiz/jax-platform
git add backend/facet_health.py backend/tests/test_facet_health_writer.py
git commit -m "feat(health): record_facet_health, escritor unico de eventos de salud"
```

---

## Task 3 — Instrumentar `_invoke_facet`

**Files:**
- Modify: `jax-platform/backend/api/chat.py:781-875`
- Test: `jax-platform/backend/tests/test_facet_health_outcomes.py` *(crear)*

**Interfaces:**
- Consumes: `facet_health.record_facet_health` (Task 2).
- Produces: `_invoke_facet(..., *, source: str = "chat")` con la firma
  pública **intacta**; `_invoke_facet_dispatch(...) -> tuple[str, UsageInfo | None, str]`.

**CRITERIO DE CIERRE NO NEGOCIABLE:** los dos tests que ya llaman a
`_invoke_facet` directo — `tests/test_chat_facet_validation.py:333` y
`tests/test_chat_resolved_version_capture.py:126` — deben seguir pasando
**sin tocarlos**. Si hay que modificar alguno, la firma pública cambió y
eso no es lo acordado: **PARAR y reportar**. Un test editado para
acomodar un refactor deja de ser evidencia de que el refactor no rompió
nada.

- [ ] **Step 1: Backup**

```bash
cd /home/fruiz/jax-platform && cp backend/api/chat.py backend/api/chat.py.backup-pre-facethealth-$(date +%Y%m%d-%H%M%S)
```

- [ ] **Step 2: Capturar la línea base de los dos tests intocables**

Run:
```bash
cd /home/fruiz/jax-platform/backend && python -m pytest \
  tests/test_chat_facet_validation.py tests/test_chat_resolved_version_capture.py -v
```
Expected: todos pasan. Anotar el número exacto de passed — es la línea
base contra la que se compara en el Step 7.

- [ ] **Step 3: Escribir el test que falla**

`backend/tests/test_facet_health_outcomes.py`:

```python
"""Mapeo de los 6 caminos de salida de _invoke_facet a `outcome` tipado.

TODO el I/O esta parcheado. Ningun test de este archivo llama a un
proveedor real -- ver "Riesgo de costo" en el plan."""
import asyncio
import pytest
from api import chat as chat_mod


def _capture(monkeypatch):
    """Reemplaza el escritor por un sink en memoria."""
    got = []
    async def fake_record(facet, outcome, source, detail=None):
        got.append({"facet": facet, "outcome": outcome,
                    "source": source, "detail": detail})
        return True
    monkeypatch.setattr(chat_mod, "record_facet_health", fake_record)
    return got


def _config():
    return {"personalities": {"jax_local": {"system_prompt": "x"},
                              "thot": {"system_prompt": "x"}}}


def test_unbound_se_registra_como_unbound(monkeypatch):
    got = _capture(monkeypatch)

    async def boom(facet): raise chat_mod.FacetUnavailableError("sin binding")
    monkeypatch.setattr(chat_mod, "resolve_facet", boom)

    texto, usage = asyncio.run(
        chat_mod._invoke_facet("thot", _config(), "u1", "hola"))

    assert usage is None
    assert [g["outcome"] for g in got] == ["unbound"]


def test_provider_error_se_registra_y_la_excepcion_SUBE(monkeypatch):
    got = _capture(monkeypatch)

    class _F:
        transport = "ollama"; model = "m"; provider_id = "p"
    async def ok_resolve(facet): return _F()
    async def boom(*a, **k): raise RuntimeError("proveedor caido")
    monkeypatch.setattr(chat_mod, "resolve_facet", ok_resolve)
    monkeypatch.setattr(chat_mod, "_call_ollama", boom)

    with pytest.raises(RuntimeError):
        asyncio.run(chat_mod._invoke_facet("jax_local", _config(), "u1", "hola"))

    assert [g["outcome"] for g in got] == ["provider_error"]


def test_source_por_defecto_es_chat(monkeypatch):
    got = _capture(monkeypatch)
    async def boom(facet): raise chat_mod.FacetUnavailableError("x")
    monkeypatch.setattr(chat_mod, "resolve_facet", boom)

    asyncio.run(chat_mod._invoke_facet("thot", _config(), "u1", "hola"))
    assert got[0]["source"] == "chat"


def test_source_se_puede_pasar_como_keyword(monkeypatch):
    got = _capture(monkeypatch)
    async def boom(facet): raise chat_mod.FacetUnavailableError("x")
    monkeypatch.setattr(chat_mod, "resolve_facet", boom)

    asyncio.run(chat_mod._invoke_facet(
        "thot", _config(), "u1", "hola", source="canary_periodic"))
    assert got[0]["source"] == "canary_periodic"


# --- EL test de esta ronda -------------------------------------------------
# Los estados "gate deniega bien" y "gate deniega por error" son los UNICOS
# que hoy son indistinguibles, y ocurren DENTRO del gate. Si un refactor
# futuro los vuelve a colapsar, el sistema regresa exactamente al problema
# que esta ronda vino a cerrar -- y regresa en silencio.
#
# La asercion es sobre el outcome EXACTO, no sobre "no fue ok": un test que
# solo verifica `!= "ok"` pasaria igual si los dos estados colapsaran entre
# si, que es precisamente la regresion que hay que impedir.

class _Governed:
    transport = "http_openai_compat"
    model = "m"; provider_id = "p"; credential = "c"; base_url = "u"
    max_tokens_param = "max_tokens"; max_output_tokens = 100


def _gobernado(monkeypatch):
    async def resolve(facet): return _Governed()
    monkeypatch.setattr(chat_mod, "resolve_facet", resolve)


def test_gate_que_responde_NO_se_registra_como_gate_denied(monkeypatch):
    got = _capture(monkeypatch)
    _gobernado(monkeypatch)

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"allowed": False, "reason": "no autorizado"}
    class _Client:
        async def post(self, *a, **k): return _Resp()
    async def fake_client(): return _Client()
    monkeypatch.setattr(chat_mod, "get_http_client", fake_client)

    texto, usage = asyncio.run(
        chat_mod._invoke_facet("thot", _config(), "u1", "hola"))

    assert usage is None
    assert [g["outcome"] for g in got] == ["gate_denied"]


def test_gate_inalcanzable_se_registra_como_gate_unreachable(monkeypatch):
    got = _capture(monkeypatch)
    _gobernado(monkeypatch)

    async def fake_client(): raise RuntimeError("las_manos inalcanzable")
    monkeypatch.setattr(chat_mod, "get_http_client", fake_client)

    texto, usage = asyncio.run(
        chat_mod._invoke_facet("thot", _config(), "u1", "hola"))

    assert usage is None
    assert [g["outcome"] for g in got] == ["gate_unreachable"]


def test_los_dos_estados_del_gate_NO_colapsan_entre_si(monkeypatch):
    """El texto que ve el usuario es IDENTICO en los dos casos -- por eso
    hoy son indistinguibles. Lo que tiene que diferir es el outcome."""
    got = _capture(monkeypatch)
    _gobernado(monkeypatch)

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"allowed": False, "reason": "x"}
    class _Client:
        async def post(self, *a, **k): return _Resp()
    async def responde(): return _Client()
    monkeypatch.setattr(chat_mod, "get_http_client", responde)
    texto_a, _ = asyncio.run(chat_mod._invoke_facet("thot", _config(), "u1", "h"))

    async def no_responde(): raise RuntimeError("caido")
    monkeypatch.setattr(chat_mod, "get_http_client", no_responde)
    texto_b, _ = asyncio.run(chat_mod._invoke_facet("thot", _config(), "u1", "h"))

    assert texto_a == texto_b                    # el usuario ve lo mismo
    assert got[0]["outcome"] != got[1]["outcome"]  # el operador NO
    assert [g["outcome"] for g in got] == ["gate_denied", "gate_unreachable"]
```

- [ ] **Step 4: Correr el test y verificar que falla**

Run: `cd /home/fruiz/jax-platform/backend && python -m pytest tests/test_facet_health_outcomes.py -v`
Expected: FAIL — `record_facet_health` no existe en `api.chat`, y
`_invoke_facet` no acepta `source`.

- [ ] **Step 5: Implementar la partición**

En `api/chat.py`, agregar el import junto a los otros
(`from db.connection import get_pool` está en la línea 26):

```python
from facet_health import record_facet_health
```

Renombrar la función existente `async def _invoke_facet(...)` a
`async def _invoke_facet_dispatch(...)`, cambiar su anotación de retorno a
`tuple[str, UsageInfo | None, str]`, y hacer que **cada** `return`
devuelva su `outcome` como tercer elemento — literal tipado, **nunca**
deducido del texto (esos strings son texto de usuario en español;
matchearlos violaría i18n además de ser frágil):

| `return` actual | outcome |
|---|---|
| `"⚠️ … sin binding activo configurado."` | `"unbound"` |
| `"⚠️ … acceso no autorizado."` | `"gate_denied"` o `"gate_unreachable"` |
| `_model_identity_reply(...)` | `"ok"` |
| ollama / gemini / openai_compat con `UsageInfo` | `"ok"` |
| `"⚠️ … transporte '…' no soportado…"` | `"unsupported_transport"` |

Para separar `gate_denied` de `gate_unreachable`, el bloque del gate
(hoy `chat.py:805-840`) ya tiene los dos caminos en `except`/`else`
distintos con sus dos `logger.warning`. Llevar una variable local:

```python
        gate_outcome = "gate_denied"      # las_manos respondio "no"
        ...
        except Exception as e:
            allowed = False
            gate_outcome = "gate_unreachable"   # las_manos no respondio
            logger.warning(...)
        if not allowed:
            return f"⚠️ {facet} no está disponible: acceso no autorizado.", None, gate_outcome
```

Y agregar el envoltorio, que conserva la firma pública:

```python
async def _invoke_facet(
    facet: str, config: dict, user_id: str, message: str,
    semantic_context: list[dict] | None = None,
    *, source: str = "chat",
) -> tuple[str, UsageInfo | None]:
    """Envoltorio instrumentado. La particion existe para que el
    `outcome` sea un literal tipado en cada punto de retorno de
    _invoke_facet_dispatch, en vez de deducirse del texto de la respuesta.

    La firma publica es la de antes mas `source` keyword-only con default:
    los llamadores y tests existentes no cambian.

    Instrumentado ACA y no en un envoltorio que el llamador tenga que
    acordarse de usar: asi el chat real y la sonda quedan cubiertos por
    construccion, sin una segunda ruta que pueda divergir."""
    try:
        texto, usage, outcome = await _invoke_facet_dispatch(
            facet, config, user_id, message, semantic_context)
    except Exception as e:
        await record_facet_health(
            facet, "provider_error", source, f"{type(e).__name__}: {e}")
        raise            # SIEMPRE re-lanza: no puede volverse fail-open
    await record_facet_health(facet, outcome, source)
    return texto, usage
```

- [ ] **Step 6: Correr los tests nuevos**

Run: `cd /home/fruiz/jax-platform/backend && python -m pytest tests/test_facet_health_outcomes.py -v`
Expected: 7 passed.

- [ ] **Step 7: Verificar el criterio de cierre — los dos tests intocables**

Run:
```bash
cd /home/fruiz/jax-platform/backend && git diff --stat -- \
  tests/test_chat_facet_validation.py tests/test_chat_resolved_version_capture.py
python -m pytest tests/test_chat_facet_validation.py tests/test_chat_resolved_version_capture.py -v
```
Expected: `git diff --stat` **vacío** (cero cambios en esos dos archivos)
y el mismo número de passed que el Step 2.
**Si el diff no está vacío, o si baja el número de passed: PARAR y
reportar. No editar los tests.**

- [ ] **Step 8: Suite completa + scanner P10**

Run: `cd /home/fruiz/jax-platform/backend && JAX_CI_NO_DB=1 python -m pytest -q`
Expected: sin regresiones respecto de la línea base del repo.

Run: `python -m pytest tests/test_no_fail_open_except.py -v`
Expected: PASS (el `except` del envoltorio re-lanza, así que no es
fail-open).

- [ ] **Step 9: Commit**

```bash
cd /home/fruiz/jax-platform
git add backend/api/chat.py backend/tests/test_facet_health_outcomes.py
git commit -m "feat(health): instrumentar _invoke_facet con outcome tipado por camino"
```

---

## Task 4 — La sonda

**Files:**
- Create: `jax-platform/backend/jax_engine/facet_canary.py`
- Modify: `jax-platform/backend/main.py` (arrancar el loop)
- Test: `jax-platform/backend/tests/test_facet_canary.py`

**Interfaces:**
- Consumes: `api.chat._invoke_facet` (Task 3), `api.chat._load_config`.
- Produces:
  ```python
  CANARY_INTERVAL_SECONDS: int      # 3600
  CANARY_USER_ID: str
  CANARY_MESSAGE: str
  def canary_facets(config: dict) -> list[str]
  async def probe_facet(facet: str, config: dict, source: str) -> str | None
  async def probe_all(source: str = "canary_periodic") -> list[str | None]
  async def start_facet_canary() -> None
  ```
  Task 5 consume `probe_facet`.

**`probe_facet` devuelve `None` cuando logró invocar, y `"probe_error"`
cuando no pudo.** Nunca devuelve `"ok"`: el resultado real de la
invocación ya lo registró `_invoke_facet` en la tabla, y si la sonda
además dijera "ok" por su cuenta habría **dos lugares decidiendo qué es
sano** — `motor.model_ref` otra vez. Peor: una denegación del gate
retorna *normalmente* (con el string ⚠️), así que un `return "ok"` sobre
"no lanzó excepción" reportaría verde exactamente sobre el fallo que esta
ronda cierra. La sonda solo garantiza tráfico y registra sus propias
fallas.

- [ ] **Step 1: Escribir el test que falla — incluye el test del hueco #1**

`backend/tests/test_facet_canary.py`:

```python
"""Sonda de facets.

NINGUN test de este archivo hace una llamada real a un proveedor:
_invoke_facet esta parcheado en todos. Ver "Riesgo de costo" en el plan
-- el 2026-08-24, correr pytest disparo 11 dispatches reales a produccion
por un archivo con codigo a nivel de modulo y nombre descubrible."""
import asyncio
import pytest
from jax_engine import facet_canary
from api import chat as chat_mod


def _config():
    return {"personalities": {
        "jax_local": {}, "hyde": {}, "jekyll": {},
        "hipatia": {}, "thot": {}, "ada": {}, "kimi": {}}}


def test_canary_facets_excluye_hyde_y_no_filtra_por_transporte():
    facets = facet_canary.canary_facets(_config())
    assert "hyde" not in facets          # chat() lo corta antes del dispatch
    assert "kimi" in facets              # DEBE sondearse: reporta
                                         # unsupported_transport, no invisible
    assert set(facets) == {"jax_local", "jekyll", "hipatia",
                           "thot", "ada", "kimi"}


def test_canary_message_no_dispara_el_cortocircuito_de_identidad():
    """Trampa real: _is_model_identity_question() cortocircuitea ANTES del
    dispatch y devuelve una respuesta enlatada. Si CANARY_MESSAGE pareciera
    una pregunta de identidad, la sonda reportaria `ok` SIN haber tocado al
    proveedor -- el detector mintiendo en verde."""
    assert chat_mod._is_model_identity_question(facet_canary.CANARY_MESSAGE) is False


def test_la_sonda_pasa_POR_el_gate_y_no_lo_saltea(monkeypatch):
    """La sonda invoca _invoke_facet -- la MISMA funcion del chat real --
    y por lo tanto pasa por el gate. Si en vez de eso resolviera el facet
    por su cuenta y llamara al proveedor directo, reportaria sano mientras
    el gate deniega a todos los usuarios reales.

    (El outcome exacto de una denegacion se verifica en Task 3, donde vive
    la instrumentacion. Aca se verifica el camino.)"""
    llamadas = []
    async def espia(facet, config, user_id, message,
                    semantic_context=None, *, source="chat"):
        llamadas.append((facet, source))
        return "⚠️ acceso no autorizado", None
    monkeypatch.setattr(facet_canary, "_invoke_facet", espia)

    out = asyncio.run(facet_canary.probe_facet("thot", _config(), "canary_periodic"))

    assert llamadas == [("thot", "canary_periodic")]
    assert out is None      # invoco; el outcome real lo registro _invoke_facet


def test_probe_facet_NUNCA_devuelve_ok(monkeypatch):
    """Un `return "ok"` sobre "no lanzo excepcion" seria un SEGUNDO lugar
    decidiendo que es sano -- y una denegacion del gate retorna
    normalmente, asi que reportaria verde sobre el fallo que esta ronda
    cierra."""
    async def denegado(*a, **k): return "⚠️ acceso no autorizado", None
    monkeypatch.setattr(facet_canary, "_invoke_facet", denegado)

    out = asyncio.run(facet_canary.probe_facet("thot", _config(), "canary_periodic"))
    assert out != "ok"
    assert out is None


def test_probe_facet_registra_probe_error_si_falla_antes_de_invocar(monkeypatch):
    async def boom(*a, **k): raise RuntimeError("no pude leer config")
    monkeypatch.setattr(facet_canary, "_invoke_facet", boom)
    recorded = []
    async def fake_record(facet, outcome, source, detail=None):
        recorded.append((facet, outcome)); return True
    monkeypatch.setattr(facet_canary, "record_facet_health", fake_record)

    out = asyncio.run(facet_canary.probe_facet("thot", _config(), "canary_periodic"))
    assert out == "probe_error"
    assert recorded == [("thot", "probe_error")]


def test_el_loop_NO_EJECUTA_NINGUNA_SONDA_bajo_pytest(monkeypatch):
    """Regla 3 del "Riesgo de costo".

    Prueba el EFECTO, no el detector: verificar que
    _running_under_pytest() devuelve True probaria que la funcion sabe
    donde esta, no que el loop se abstiene. Lo que puede costar plata es
    que probe_all corra -- eso es lo que se asserta.

    Si esta proteccion se rompe, correr la suite dispara sondas pagas
    contra los 4 facets: exactamente el accidente del 2026-08-24."""
    llamadas = []
    async def espia(source="canary_periodic"):
        llamadas.append(source)
        return []
    monkeypatch.setattr(facet_canary, "probe_all", espia)

    # Si el guard fallara, start_facet_canary() entraria en `while True` y
    # este test colgaria en vez de fallar. El timeout lo convierte en un
    # fallo legible.
    async def _run():
        await asyncio.wait_for(facet_canary.start_facet_canary(), timeout=5)

    asyncio.run(_run())

    assert llamadas == [], (
        "start_facet_canary() ejecuto sondas bajo pytest -- son llamadas "
        "PAGAS a proveedores reales")


def test_running_under_pytest_detecta_el_entorno():
    """Complemento del anterior: el detector en si. Por separado, para que
    quede claro cual de los dos prueba que -- si este pasa y el otro falla,
    el guard existe pero no se esta aplicando."""
    assert facet_canary._running_under_pytest() is True
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `cd /home/fruiz/jax-platform/backend && python -m pytest tests/test_facet_canary.py -v`
Expected: FAIL — `ModuleNotFoundError: jax_engine.facet_canary`.

- [ ] **Step 3: Implementar**

`backend/jax_engine/facet_canary.py`:

```python
"""Sonda activa de facets.

POR QUE ACTIVA Y NO PASIVA: del 2026-08-20 al 08-26 inclusive hubo CERO
turnos de chat, y thot quedo rebindeado a gpt-5.6-terra el 08-24 11:08:01.
Durante los tres dias que estuvo roto nadie lo llamo. Un detector derivado
del trafico real no habria detectado nada -- ver §1.3 del spec.

COSTO: cada sonda es una llamada PAGA a un proveedor real. Ningun test
puede ejecutarla; el loop no arranca bajo pytest (ver
_running_under_pytest). Precedente: 2026-08-24, correr pytest disparo 11
dispatches reales a produccion."""
import asyncio
import logging
import os
import sys

from api.chat import _invoke_facet, _load_config
from facet_health import record_facet_health

logger = logging.getLogger(__name__)

CANARY_INTERVAL_SECONDS = 3600
CANARY_USER_ID = "__canary__"
# NO puede parecer una pregunta de identidad de modelo: _is_model_identity_question()
# cortocircuitea antes del dispatch y devolveria una respuesta enlatada, o
# sea `ok` sin haber tocado al proveedor. Hay un test que lo verifica.
CANARY_MESSAGE = "Respondé únicamente con la palabra: listo."

# hyde no se sondea: chat() lo corta antes del dispatch con una respuesta
# enlatada, no hay nada que medir.
_NOT_DISPATCHED = frozenset({"hyde"})


def _running_under_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules


def canary_facets(config: dict) -> list[str]:
    """El MISMO conjunto contra el que chat() valida req.facet
    (api/chat.py:902), o sea exactamente lo que un usuario puede elegir.

    NO se filtra por transporte a proposito: si se filtrara a "transportes
    despachables", kimi (transport=motor_registry) quedaria fuera y su
    caida seria invisible por diseno -- que es justo la clase de falla que
    esta feature existe para detectar."""
    return sorted(set(config["personalities"]) - _NOT_DISPATCHED)


async def probe_facet(facet: str, config: dict, source: str) -> str | None:
    """Sondea UN facet. Devuelve None si logro invocar, 'probe_error' si no.

    NUNCA devuelve 'ok'. El resultado real de la invocacion ya lo registro
    _invoke_facet en la tabla (Task 3); si la sonda ademas dijera 'ok' por
    su cuenta habria DOS lugares decidiendo que es sano. Y una denegacion
    del gate retorna NORMALMENTE (con el string de degradacion), asi que un
    'ok' basado en "no lanzo excepcion" reportaria verde justo sobre el
    fallo que esta ronda cierra.

    La sonda pasa por el gate igual que el chat real, porque invoca la
    MISMA funcion: los estados gate_denied y gate_unreachable solo ocurren
    DENTRO del gate. Una sonda que resolviera el facet por su cuenta y
    llamara al proveedor directo seria ciega a los dos."""
    try:
        await _invoke_facet(facet, config, CANARY_USER_ID, CANARY_MESSAGE,
                            source=source)
        return None
    except Exception as e:
        # La sonda no llego a completar la invocacion. Un detector que falla
        # produce un evento, no un silencio.
        await record_facet_health(
            facet, "probe_error", source, f"{type(e).__name__}: {e}")
        return "probe_error"


async def probe_all(source: str = "canary_periodic") -> list[str | None]:
    config = _load_config()
    return [await probe_facet(f, config, source) for f in canary_facets(config)]


async def start_facet_canary() -> None:
    if _running_under_pytest():
        logger.warning("facet_canary: no arranca bajo pytest (llamadas pagas)")
        return
    while True:
        try:
            await probe_all("canary_periodic")
        except Exception:  # fail-soft: loop en background, mismo patron que owner_cleanup.py -- nunca debe tumbar el proceso, el proximo ciclo reintenta
            logger.warning("facet_canary: barrido fallo", exc_info=True)
        await asyncio.sleep(CANARY_INTERVAL_SECONDS)
```

- [ ] **Step 4: Correr los tests**

Run: `cd /home/fruiz/jax-platform/backend && python -m pytest tests/test_facet_canary.py -v`
Expected: 7 passed.

- [ ] **Step 5: Arrancar el loop en `main.py`**

Backup primero:
```bash
cd /home/fruiz/jax-platform && cp backend/main.py backend/main.py.backup-pre-facethealth-$(date +%Y%m%d-%H%M%S)
```

Junto a `asyncio.create_task(start_owner_file_cleanup())` (`main.py:87`):

```python
from jax_engine.facet_canary import start_facet_canary
...
    asyncio.create_task(start_facet_canary())
```

- [ ] **Step 6: Verificar que la suite sigue verde y que NO se disparó ninguna sonda**

Run: `cd /home/fruiz/jax-platform/backend && JAX_CI_NO_DB=1 python -m pytest -q`
Expected: sin regresiones.

Run: `journalctl -u jax-platform --since "5 min ago" | grep -c "credential_resolution"`
Expected: **0 llamadas nuevas** atribuibles a la suite. Es la verificación
directa del precedente del 24-ago. **Si aparecen llamadas: PARAR y
reportar — el guard no funciona.**

- [ ] **Step 7: Commit**

```bash
cd /home/fruiz/jax-platform
git add backend/jax_engine/facet_canary.py backend/main.py backend/tests/test_facet_canary.py
git commit -m "feat(health): sonda activa horaria, con guard anti-pytest"
```

---

## Task 5 — Sonda por rebinding, en los DOS escritores

**Files:**
- Modify: `jax-platform/backend/api/admin/models.py:148-190` (`approve_proposal`)
- Modify: `jax-platform/backend/api/admin/facet_bindings.py:87-125` (`update_facet_binding`)
- Test: `jax-platform/backend/tests/test_facet_canary_rebind.py` *(crear)*

**Interfaces:**
- Consumes: `jax_engine.facet_canary.probe_facet` (Task 4).
- Produces: `probe_after_rebind(facet_key: str) -> None`, encolable con
  `BackgroundTasks`.

**Por qué en los dos:** `facet_binding` tiene **dos** escritores. El
docstring de `approve_proposal` registra que esa misma duplicación ya
causó un incidente el 2026-08-24: *"volvio a fallar 5 dias despues porque
PUT /api/admin/facet-bindings/{key} escribia facet_binding sin pasar por
aca"*. Colgar la sonda solo de `approve` reproduce ese bug exactamente.

- [ ] **Step 1: Escribir el test que falla**

`backend/tests/test_facet_canary_rebind.py`:

```python
"""La sonda por rebinding cuelga de los DOS escritores de facet_binding.
I/O parcheado; no se llama a ningun proveedor."""
import asyncio
import inspect
from api.admin import models as models_mod
from api.admin import facet_bindings as fb_mod
from jax_engine import facet_canary


def test_los_dos_escritores_reciben_background_tasks():
    """Si un escritor no acepta BackgroundTasks, no puede encolar la sonda
    -- y ese es exactamente el bug que ya paso con motor.model_ref."""
    for fn in (models_mod.approve_proposal, fb_mod.update_facet_binding):
        params = inspect.signature(fn).parameters
        assert any("BackgroundTasks" in str(p.annotation) for p in params.values()), \
            f"{fn.__name__} no recibe BackgroundTasks"


def test_probe_after_rebind_usa_source_canary_rebind(monkeypatch):
    got = {}
    async def fake_probe(facet, config, source):
        got["facet"], got["source"] = facet, source
        return None
    monkeypatch.setattr(facet_canary, "probe_facet", fake_probe)
    monkeypatch.setattr(facet_canary, "_load_config",
                        lambda: {"personalities": {"thot": {}}})

    asyncio.run(facet_canary.probe_after_rebind("thot"))
    assert got == {"facet": "thot", "source": "canary_rebind"}
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `cd /home/fruiz/jax-platform/backend && python -m pytest tests/test_facet_canary_rebind.py -v`
Expected: FAIL — `probe_after_rebind` no existe; los escritores no reciben
`BackgroundTasks`.

- [ ] **Step 3: Agregar `probe_after_rebind` a `facet_canary.py`**

```python
async def probe_after_rebind(facet_key: str) -> str:
    """Sonda disparada por un cambio de binding.

    Se encola con BackgroundTasks DESPUES del conn.commit() del escritor:
    FastAPI corre las background tasks despues de emitir la respuesta, asi
    que "primero aprobado, despues sondeado" queda garantizado por
    construccion, no por convencion.

    Su resultado se alerta en el barrido siguiente del reaper (<=300s), no
    en la corrida horaria: por eso el lector evalua en CADA barrido."""
    config = _load_config()
    return await probe_facet(facet_key, config, "canary_rebind")
```

- [ ] **Step 4: Enganchar los dos escritores**

Backups:
```bash
cd /home/fruiz/jax-platform
cp backend/api/admin/models.py backend/api/admin/models.py.backup-pre-facethealth-$(date +%Y%m%d-%H%M%S)
cp backend/api/admin/facet_bindings.py backend/api/admin/facet_bindings.py.backup-pre-facethealth-$(date +%Y%m%d-%H%M%S)
```

En **ambos** endpoints: agregar `background_tasks: BackgroundTasks` a la
firma, importar `from fastapi import BackgroundTasks` y
`from jax_engine.facet_canary import probe_after_rebind`, y **después del
`await conn.commit()`**, antes del `return`:

```python
    # DESPUES del commit a proposito: la sonda solo tiene sentido sobre un
    # binding ya aprobado. Encolada, no await inline -- un await colgaria
    # la request del admin de una llamada a un proveedor externo.
    background_tasks.add_task(probe_after_rebind, facet_key)
```

En `approve_proposal` la variable es `facet_key` (viene de
`_fetch_proposal`); en `update_facet_binding` es el parámetro `facet_key`.

- [ ] **Step 5: Correr los tests**

Run: `cd /home/fruiz/jax-platform/backend && python -m pytest tests/test_facet_canary_rebind.py -v`
Expected: 2 passed.

- [ ] **Step 6: Suite completa**

Run: `cd /home/fruiz/jax-platform/backend && JAX_CI_NO_DB=1 python -m pytest -q`
Expected: sin regresiones. **Los tests existentes de admin que llamen a
estos endpoints pueden necesitar pasar `BackgroundTasks` — si alguno
falla, PARAR y reportar antes de editarlo.**

- [ ] **Step 7: Commit**

```bash
cd /home/fruiz/jax-platform
git add backend/jax_engine/facet_canary.py backend/api/admin/models.py \
        backend/api/admin/facet_bindings.py backend/tests/test_facet_canary_rebind.py
git commit -m "feat(health): sonda por rebinding en los dos escritores de facet_binding"
```

---

## Task 6 — Lector y máquina de estados (repo `jax`)

**Files:**
- Create: `jax/jacobs/facet_health.py`
- Test: `jax/jacobs/_facet_health_test.py`

**Interfaces:**
- Consumes: `jacobs.store.get_conn`, `jacobs.reaper.send_telegram_alert`.
- Produces:
  ```python
  HEALTH_WINDOW_SECONDS: int          # 7200
  ALERT_REPEAT_SECONDS: int           # 21600
  HEALTH_EVENT_RETENTION_DAYS: int    # 30
  SYSTEM_KEY: str                     # '__system__'
  def evaluate_states(events_by_facet: dict[str, tuple[float, str]],
                      known_facets: list[str], now: float) -> dict[str, str]
  def transitions_to_notify(current: dict[str, str],
                            ledger: dict[str, tuple[str, float | None]],
                            now: float) -> list[tuple[str, str]]
  async def check_facet_health() -> dict
  ```
  Task 7 consume `check_facet_health`.

- [ ] **Step 1: Escribir el test que falla**

`jax/jacobs/_facet_health_test.py`:

```python
"""Maquina de estados de salud de facets. Funciones PURAS -- sin I/O, sin
red, sin DB. Ningun test de este archivo puede disparar una llamada paga."""
from jacobs import facet_health as fh

NOW = 1_000_000.0
W = fh.HEALTH_WINDOW_SECONDS


def test_cero_eventos_es_unknown_NUNCA_ok():
    """Ausencia de datos NO es salud. El chequeo es
    `if total_eventos == 0: unknown`, nunca `if fallos == 0: ok` -- esa
    segunda forma es el bug escrito como codigo."""
    got = fh.evaluate_states({}, ["thot", "ada"], NOW)
    assert got == {"thot": "unknown", "ada": "unknown"}
    assert "ok" not in got.values()


def test_evento_viejo_fuera_de_ventana_es_unknown():
    got = fh.evaluate_states({"thot": (NOW - W - 1, "ok")}, ["thot"], NOW)
    assert got["thot"] == "unknown"


def test_ultimo_evento_ok_es_ok():
    got = fh.evaluate_states({"thot": (NOW - 10, "ok")}, ["thot"], NOW)
    assert got["thot"] == "ok"


def test_ultimo_evento_de_falla_es_down():
    for bad in ("provider_error", "gate_denied", "gate_unreachable",
                "unbound", "unsupported_transport", "probe_error"):
        got = fh.evaluate_states({"thot": (NOW - 10, bad)}, ["thot"], NOW)
        assert got["thot"] == "down", bad


def test_ningun_facet_con_eventos_produce_una_sola_alerta_de_sistema():
    got = fh.evaluate_states({}, ["thot", "ada", "jekyll", "kimi"], NOW)
    notify = fh.transitions_to_notify(got, ledger={}, now=NOW)
    keys = [k for k, _ in notify]
    assert keys == [fh.SYSTEM_KEY]      # una sola, no cuatro


def test_la_alerta_de_sistema_respeta_la_repeticion_de_6h():
    """Si la sonda muere un viernes, no queremos 288 mensajes el sabado."""
    states = fh.evaluate_states({}, ["thot", "ada"], NOW)
    ledger = {fh.SYSTEM_KEY: ("unknown", NOW - 60)}   # ya avisado hace 1 min
    assert fh.transitions_to_notify(states, ledger, NOW) == []

    ledger = {fh.SYSTEM_KEY: ("unknown", NOW - fh.ALERT_REPEAT_SECONDS - 1)}
    assert [k for k, _ in fh.transitions_to_notify(states, ledger, NOW)] == [fh.SYSTEM_KEY]


def test_facet_caido_no_re_alerta_en_cada_barrido():
    states = {"thot": "down"}
    ledger = {"thot": ("down", NOW - 60)}
    assert fh.transitions_to_notify(states, ledger, NOW) == []


def test_recuperacion_se_notifica():
    states = {"thot": "ok"}
    ledger = {"thot": ("down", NOW - 60)}
    assert fh.transitions_to_notify(states, ledger, NOW) == [("thot", "ok")]


def test_system_key_no_puede_colisionar_con_un_facet_real():
    """Los nombres de facet vienen de facet.key; ninguno empieza con
    guion bajo."""
    assert fh.SYSTEM_KEY.startswith("__")
    for real in ("thot", "ada", "hipatia", "jekyll", "kimi", "jax_local", "hyde"):
        assert real != fh.SYSTEM_KEY
        assert not real.startswith("_")
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `cd /home/fruiz/jax && python -m pytest jacobs/_facet_health_test.py -v`
Expected: FAIL — `ModuleNotFoundError: jacobs.facet_health`.

- [ ] **Step 3: Implementar**

`jax/jacobs/facet_health.py`:

```python
"""Lector UNICO de salud de facets. La salud se calcula EXCLUSIVAMENTE
aca, desde facet_health_event. Quien escribe esa tabla es
jax-platform/backend/facet_health.py; quien la lee es solo este modulo.

facet_health_alert NO es una segunda fuente de verdad: es el registro de
que ya se aviso -- la distincion entre un valor y su acuse de recibo.

Ver docs/superpowers/specs/2026-08-27-alertas-facets-caidos-design.md."""
import logging
import time

from jacobs import store

logger = logging.getLogger("jacobs.facet_health")

HEALTH_WINDOW_SECONDS = 7200        # 2 * CANARY_INTERVAL_SECONDS
ALERT_REPEAT_SECONDS = 21600        # 6h
HEALTH_EVENT_RETENTION_DAYS = 30
SYSTEM_KEY = "__system__"

_OK = "ok"


def evaluate_states(events_by_facet, known_facets, now):
    """PURA. events_by_facet: {facet: (ts, outcome)} del evento MAS
    RECIENTE de cada facet. Devuelve {facet: 'ok'|'down'|'unknown'}.

    AUSENCIA DE DATOS NO ES SALUD: cero eventos -> 'unknown', que alerta
    igual que 'down'. El chequeo es sobre el total de eventos, NUNCA
    `if fallos == 0: ok` -- esa forma es el bug que produjo
    "OK -- 0/0 dispatches (0.0% gap)" en el reaper y "1 passed" sobre cero
    archivos escaneados en el scanner P10."""
    out = {}
    cutoff = now - HEALTH_WINDOW_SECONDS
    for facet in known_facets:
        entry = events_by_facet.get(facet)
        if entry is None or entry[0] < cutoff:
            out[facet] = "unknown"
        elif entry[1] == _OK:
            out[facet] = _OK
        else:
            out[facet] = "down"
    return out


def transitions_to_notify(current, ledger, now):
    """PURA. Devuelve [(clave, estado_nuevo)] a notificar.

    Si TODOS los facets estan en unknown, se emite UNA sola alerta bajo
    SYSTEM_KEY ("la sonda no esta corriendo") en vez de una por facet: el
    diagnostico es distinto y el mensaje debe decirlo. Esa alerta agregada
    pasa por el MISMO ledger y la MISMA supresion -- sin eso, una sonda
    muerta el viernes produce 288 mensajes el sabado."""
    if current and all(s == "unknown" for s in current.values()):
        return _maybe(SYSTEM_KEY, "unknown", ledger, now)

    notify = []
    for facet, state in sorted(current.items()):
        notify.extend(_maybe(facet, state, ledger, now))
    return notify


def _maybe(key, state, ledger, now):
    prev_state, prev_notified = ledger.get(key, (None, None))
    if prev_state != state:
        return [(key, state)]                 # transicion: siempre avisa
    if state == _OK:
        return []                             # sigue bien: nada que decir
    if prev_notified is None or now - prev_notified >= ALERT_REPEAT_SECONDS:
        return [(key, state)]                 # sigue mal: recordatorio 6h
    return []


async def check_facet_health() -> dict:
    """Un chequeo completo: lee, evalua, notifica, actualiza el ledger y
    poda. Se llama desde el reaper en CADA barrido (300s) -- leer es una
    consulta; lo caro (la llamada al LLM) sigue siendo horario y vive del
    otro lado."""
    from jacobs.reaper import send_telegram_alert   # import diferido: evita ciclo

    now = time.time()
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT e.facet, e.ts, e.outcome FROM facet_health_event e "
                "JOIN (SELECT facet, MAX(ts) mt FROM facet_health_event "
                "      WHERE ts >= %s GROUP BY facet) m "
                "  ON m.facet = e.facet AND m.mt = e.ts",
                (now - HEALTH_WINDOW_SECONDS,),
            )
            latest = {r[0]: (r[1], r[2]) for r in await cur.fetchall()}

            await cur.execute("SELECT `key` FROM facet")
            known = [r[0] for r in await cur.fetchall()]

            await cur.execute(
                "SELECT facet, state, notified_ts FROM facet_health_alert")
            ledger = {r[0]: (r[1], r[2]) for r in await cur.fetchall()}

            states = evaluate_states(latest, known, now)
            notify = transitions_to_notify(states, ledger, now)

            for key, state in notify:
                msg = (f"JAX -- la sonda de facets no esta corriendo "
                       f"(cero eventos en {HEALTH_WINDOW_SECONDS // 3600}h)"
                       if key == SYSTEM_KEY else
                       f"JAX -- facet '{key}': {state}")
                await send_telegram_alert(msg)
                await cur.execute(
                    "INSERT INTO facet_health_alert (facet, state, first_seen_ts, notified_ts) "
                    "VALUES (%s,%s,%s,%s) ON DUPLICATE KEY UPDATE "
                    "state=VALUES(state), notified_ts=VALUES(notified_ts)",
                    (key, state, now, now),
                )

            # Retencion: sin esto la tabla crece para siempre -- el error
            # que motivo owner_cleanup.py ("Had no cleanup, so the
            # directory grew forever"). 30 dias >> la ventana de 2h, asi
            # que la poda no puede fabricar un `unknown`.
            await cur.execute(
                "DELETE FROM facet_health_event WHERE ts < %s",
                (now - HEALTH_EVENT_RETENTION_DAYS * 86400,),
            )
        await conn.commit()
    finally:
        conn.close()

    return {"states": states, "notified": [k for k, _ in notify]}
```

- [ ] **Step 4: Correr los tests**

Run: `cd /home/fruiz/jax && python -m pytest jacobs/_facet_health_test.py -v`
Expected: 9 passed.

- [ ] **Step 5: Scanner P10**

Run: `cd /home/fruiz/jax && python -m pytest policy/tests/test_no_fail_open_except.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/fruiz/jax
git add jacobs/facet_health.py jacobs/_facet_health_test.py
git commit -m "feat(health): lector unico de salud de facets, ausencia de datos = unknown"
```

---

## Task 7 — Enganchar al reaper + job de CI

**Files:**
- Modify: `jax/jacobs/reaper.py` (`start_reaper_loop`, ~línea 447-459)
- Modify: `jax/.github/workflows/policy.yml`

**Interfaces:**
- Consumes: `jacobs.facet_health.check_facet_health` (Task 6).
- Produces: chequeo de salud en cada barrido de 300 s.

- [ ] **Step 1: Backup**

```bash
cd /home/fruiz/jax && cp jacobs/reaper.py jacobs/reaper.py.backup-pre-facethealth-$(date +%Y%m%d-%H%M%S)
```

- [ ] **Step 2: Enganchar en `start_reaper_loop`**

`check_facet_health()` va en **cada** barrido, no cada
`RECONCILIATION_CHECK_EVERY_N_SWEEPS`: una sonda por rebinding tiene que
alertarse en ≤5 min, no en la corrida horaria siguiente.

```python
        try:
            await check_facet_health()
        except Exception:  # fail-soft: loop de fondo, mismo patron que el barrido de arriba -- nunca debe tumbar el proceso, el proximo ciclo (300s) reintenta
            logger.error(
                "Reaper: chequeo de salud de facets FALLO -- el detector no "
                "corrio en este barrido; no hay salud calculada, no 'todo ok'",
                exc_info=True,
            )
        sweep_count += 1
```

Con `from jacobs.facet_health import check_facet_health` arriba.

**`logger.error`, no `logger.warning`, y con ese texto a propósito.** Un
chequeo que no corre no es un chequeo que dio bien. Ésta es la única parte
del sistema donde un fallo del detector queda solo en el journal: si el
esquema está incompleto o la DB no responde, este `except` se traga el
error cada 300 s y **nadie se entera**. Por eso el gate de esquema completo
de la Task 8 Step 3 es una parada dura y no una advertencia — es la
mitigación real de este hueco. **Queda declarado como limitación conocida
de la v1**, en la misma línea que el §4 del spec: el detector detecta
facets caídos, pero su propia caída total solo se ve en el journal.

- [ ] **Step 3: Verificar que compila y que la suite de `jax` sigue verde**

Run: `cd /home/fruiz/jax && python -m py_compile jacobs/reaper.py && python -m pytest jacobs/_facet_health_test.py policy/tests/ -v`
Expected: todo PASS.

- [ ] **Step 4: Agregar el job de CI**

En `.github/workflows/policy.yml`, siguiendo el patrón de los 4 jobs
existentes (uno por archivo de test):

```yaml
  facet-health-unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install pytest
      - run: python -m pytest jacobs/_facet_health_test.py -v
```

- [ ] **Step 5: Commit**

```bash
cd /home/fruiz/jax
git add jacobs/reaper.py .github/workflows/policy.yml
git commit -m "feat(health): chequeo de salud en cada barrido del reaper + job de CI"
```

---

## Task 8 — Deploy y verificación rompiendo a propósito

**PRE-REQUISITO: leer "Orden de deploy" en Global Constraints.**
`jax-platform` **PRIMERO**, `jax-las-manos` **DESPUÉS**. Es el **inverso**
del de la ronda anterior, y está verificado: `run_migrations()` corre en
el `lifespan` de `jax-platform` (`main.py:84`), y el lector vive en
`las_manos`.

- [ ] **Step 1: Mergear ambos PRs y actualizar los checkouts que sirven a producción**

`jax-platform` sirve desde `/home/fruiz/jax-platform`; `las_manos` desde
`/home/fruiz/jax` (**no** un worktree — mismo error que se cazó la ronda
pasada).

- [ ] **Step 2: Reiniciar `jax-platform` PRIMERO**

```bash
sudo systemctl restart jax-platform
systemctl is-active jax-platform
journalctl -u jax-platform -n 40 --no-pager | grep -iE "error|traceback" || echo "arranque limpio"
```
Expected: `active`, sin excepciones.

- [ ] **Step 3: Gate de ESQUEMA COMPLETO — no de "existe la tabla"**

**Por qué el gate es sobre el esquema completo y no sobre la existencia:**
el DDL en MariaDB **no es transaccional** — cada `CREATE TABLE`
auto-commitea. `run_migrations()` recorre `_TABLES` en un bucle y commitea
recién al final (`migrations.py:1409-1445`), así que **un fallo a mitad
deja tablas parciales en disco**, no revierte nada.

El modo de falla es el peor posible: si se creó `facet_health_event` pero
no `facet_health_alert`, el lector del reaper consulta la primera bien y
**revienta en la segunda**; su `except` fail-soft lo captura, loguea, y el
barrido sigue. Resultado: **el detector no corre nunca y no avisa nunca**,
un `WARNING` cada 300 s en un journal que nadie mira. Es el patrón que
esta ronda existe para eliminar, reproducido por el propio deploy.

```bash
set -a && . /etc/jax/.env && set +a
Q() { mysql -h "$JAX_DB_HOST" -P "$JAX_DB_PORT" -u "$JAX_DB_USER" \
        -p"$JAX_DB_PASSWORD" "$JAX_DB_NAME" -N -B -e "$1"; }

echo "-- tablas --"
Q "SELECT COUNT(*) FROM information_schema.TABLES
   WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME IN
   ('facet_health_event','facet_health_alert');"

echo "-- columnas de facet_health_event --"
Q "SELECT GROUP_CONCAT(COLUMN_NAME ORDER BY ORDINAL_POSITION) FROM information_schema.COLUMNS
   WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='facet_health_event';"

echo "-- columnas de facet_health_alert --"
Q "SELECT GROUP_CONCAT(COLUMN_NAME ORDER BY ORDINAL_POSITION) FROM information_schema.COLUMNS
   WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='facet_health_alert';"

echo "-- los 7 valores del ENUM outcome --"
Q "SELECT COLUMN_TYPE FROM information_schema.COLUMNS
   WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='facet_health_event'
     AND COLUMN_NAME='outcome';"

echo "-- ts es DOUBLE, no TIMESTAMP --"
Q "SELECT DATA_TYPE FROM information_schema.COLUMNS
   WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='facet_health_event'
     AND COLUMN_NAME='ts';"
```

Expected, los cinco:
1. `2`
2. `id,facet,outcome,source,detail,ts`
3. `facet,state,first_seen_ts,notified_ts`
4. el `enum(...)` con los **7** valores: `ok`, `provider_error`,
   `gate_denied`, `gate_unreachable`, `unbound`, `unsupported_transport`,
   `probe_error`
5. `double`

**Cualquiera de los cinco que no coincida: PARAR. No reiniciar
`jax-las-manos`.** Con esquema incompleto, arrancar el lector produce un
detector que falla en silencio, que es peor que no tener detector — al
menos hoy sabemos que no hay ninguno.

- [ ] **Step 4: Reiniciar `jax-las-manos` DESPUÉS**

```bash
sudo systemctl restart jax-las-manos
systemctl is-active jax-las-manos
```
Expected: `active`.

- [ ] **Step 5: Verificar que la sonda escribe de verdad**

Esperar un ciclo y consultar:
```sql
SELECT facet, outcome, source, FROM_UNIXTIME(ts) FROM facet_health_event ORDER BY ts DESC LIMIT 20;
```
Expected: filas con `source='canary_periodic'` para los 6 facets.
**Expected específico: `kimi` con `outcome='unsupported_transport'`.**
Es el criterio de aceptación del §5 del spec: si `kimi` **no** aparece
caído, la v1 tiene un hueco. **PARAR y reportar.**

- [ ] **Step 6: Romper el gate a propósito — `gate_denied`**

Guardar el valor original **antes** de tocarlo, para poder revertir con el
valor exacto y no con uno reconstruido de memoria:

```bash
set -a && . /etc/jax/.env && set +a
Q() { mysql -h "$JAX_DB_HOST" -P "$JAX_DB_PORT" -u "$JAX_DB_USER" \
        -p"$JAX_DB_PASSWORD" "$JAX_DB_NAME" -N -B -e "$1"; }

Q "SELECT allowed_callers FROM facet WHERE \`key\`='ada';"   # ANOTAR la salida
Q "UPDATE facet SET allowed_callers='[]' WHERE \`key\`='ada';"
```

Disparar una sonda, verificar `outcome='gate_denied'`, y **revertir con el
valor anotado**:

```bash
Q "UPDATE facet SET allowed_callers='[\"jacobs\", \"jax_platform_chat\"]' WHERE \`key\`='ada';"
Q "SELECT allowed_callers FROM facet WHERE \`key\`='ada';"   # confirmar que volvio
```
Expected: `gate_denied`. Si sale `ok`, la sonda se está salteando el gate
— **PARAR**, es el hueco #1.

- [ ] **Step 7: Romper la alcanzabilidad — `gate_unreachable`**

Parar `jax-las-manos` brevemente, disparar una sonda, verificar
`outcome='gate_unreachable'`, reiniciar.
Expected: `gate_unreachable`, **distinto** del Step 6. **Ésta es la
prueba de que los estados 3 y 4 quedaron separados** — el objetivo
central del ítem.

- [ ] **Step 8: Probar `unknown`**

Parar la sonda con eventos frescos en la tabla y esperar a que la ventana
de 2 h los deje afuera (o insertar eventos con `ts` viejo en un entorno de
prueba). Expected: estado `unknown` y alerta de Telegram — **no** `ok`.

- [ ] **Step 9: Confirmar entrega real de Telegram**

Expected: mensaje recibido. `send_telegram_alert` devuelve
`{ok, message_id, error}`; verificar `ok=True` en el log, no asumirlo.

---

## Self-Review

**Cobertura del spec:** §2.1 → Task 1. §2.2 → Task 3. §2.3 → Task 4.
§2.4 → Task 5. §2.5 → Tasks 6 y 7. §3 (romper a propósito) → Task 8,
Steps 6-8; §3 (CI) → Tasks 2/3/4/6 + job en Task 7. §4 (fuera de alcance)
→ no genera tasks, por definición. §5 (`kimi`) → criterio de aceptación en
Task 8 Step 5. §6 (6 turnos) → va a `DEUDA.md` al cerrar la ronda.

**Consistencia de tipos:** `record_facet_health(facet, outcome, source,
detail)` es la misma firma en Tasks 2, 3 y 4. `probe_facet(facet, config,
source)` es la misma en Tasks 4 y 5. `evaluate_states` /
`transitions_to_notify` / `check_facet_health` coinciden entre Task 6 y
Task 7. Los 7 valores de `outcome` del DDL (Task 1) son los mismos de
`OUTCOMES` (Task 2), la tabla de mapeo (Task 3) y los tests (Task 6).

**Dos defectos encontrados y corregidos en esta revisión, no declarados
como huecos:**

1. `probe_facet` devolvía `"ok"` cuando `_invoke_facet` retornaba sin
   excepción — pero una denegación del gate **retorna normalmente**, con
   el string de degradación. La sonda habría reportado verde justo sobre
   el fallo que la ronda cierra, y habría sido un segundo lugar decidiendo
   qué es sano. Ahora devuelve `None` (invocó) o `"probe_error"` (no
   pudo), y hay un test que verifica que **nunca** devuelve `"ok"`.
2. El `UPDATE` de la Task 8 Step 6 tenía `WHERE` antes de `SET`. Corregido,
   y con el valor original guardado antes de tocarlo para que la reversión
   use el valor exacto y no uno reconstruido de memoria.

**Dónde vive el test del hueco #1:** en la Task 3, donde vive la
instrumentación, con aserción sobre el `outcome` exacto y un tercer test
que verifica que el texto al usuario es idéntico en los dos casos mientras
los `outcome` difieren. La Task 4 verifica lo complementario: que la sonda
pasa **por** `_invoke_facet` y no alrededor.

---

En memoria de Jairo Urbina.
