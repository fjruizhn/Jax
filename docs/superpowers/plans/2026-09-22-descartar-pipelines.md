# Descartar pipelines detenidos: plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que un pipeline detenido se pueda descartar, recuperar, ocultar y restaurar, sin borrar nunca una fila.

**Architecture:** Jacobs (repo **jax**) suma dos estados terminales, `discarded` y `hidden`, y cuatro rutas de transición escritas con compare-and-set sobre la época. jax-platform hace de proxy de esas rutas, decide QUIÉN puede pedir cada una (dueño, quien descartó o superadmin), saca los estados nuevos de las listas y agrega las vistas "Descartados" y "Pipelines ocultos".

**Tech Stack:** Python 3.12, FastAPI, aiomysql, MariaDB 12.3.3 y pytest (jax y jax-platform backend); React 19, Zustand, vitest y testing-library (jax-platform frontend).

**Spec:** `docs/superpowers/specs/2026-09-22-descartar-pipelines-design.md` (repo jax, master `52e2599`).

## Global Constraints

- Ninguna fila de `jacobs_pipelines`, `jacobs_steps`, `jacobs_events` ni `axioma_usage` se borra. "Borrar" en la interfaz es `hidden`.
- Toda transición es compare-and-set: `WHERE pipeline_id=%s AND run_epoch=%s AND status IN (...)`. Si no escribe, 409.
- Transiciones permitidas, y ninguna otra:
  - `aborted|expired → discarded`
  - `discarded → <status_previo>`
  - `discarded → hidden`
  - `hidden → discarded`
- Permisos, en jax-platform:
  - Descartar: el dueño (`_require_pipeline_owner`).
  - Recuperar: `descartado_por == user.user_id`, o superadmin.
  - Ocultar y restaurar: solo superadmin.
- `/continue` y `/cancel` responden 409 sobre `discarded` y `hidden`.
- `discarded` y `hidden` están en `ESTADOS_SIN_CUPO`. El reaper no los toca.
- Frontend:
  - Ningún texto visible hardcodeado: todo va en `frontend/src/i18n/es.js` y `en.js`.
  - Colores solo por tokens. Tema claro y oscuro.
  - Nada de `confirm(`, `alert(` ni `prompt(`, ni con prefijo ni desnudos. Descartar usa `Dialogo`; ocultar usa `ConfirmacionSuma`.
- Barrera de producción: `/etc/jax/.env` apunta a PRODUCCIÓN. Ningún test ni comando lo carga. Los tests con base usan `base_de_test.fijar_base_de_test()`.
- Nunca `git stash`: la pila es compartida entre worktrees. Para apartar cambios, un commit WIP.
- LAS CUATRO:
  - `EXPLAIN` sin `Using filesort` en cada lista nueva.
  - Medición de carga de `GET /pipelines` y de la lista de descartados, antes del GO.
- Commits en español, terminando con `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- Pisos exactos de tests en `.github/workflows/policy.yml` de cada repo: si cambia el conteo, se actualiza con su nota. YAML validado con `yaml.safe_load` y sin U+2028/U+2029.
- **Orden de despliegue:** jax primero (Jacobs conoce los estados nuevos) y jax-platform después.

---

## File Structure

**jax** (worktree `~/worktrees/jax-descartar`, rama `feat/descartar-pipelines`)

| Archivo | Responsabilidad |
|---|---|
| `jacobs/models.py` | `PipelineStatus.discarded`, `PipelineStatus.hidden` |
| `jacobs/policy.py` | los dos estados en `ESTADOS_SIN_CUPO` |
| `jacobs/store.py` | columnas `status_previo`, `descartado_por`, `descartado_at`; dos índices; `pipeline_transicion_descarte()` |
| `jacobs/descarte.py` (nuevo) | reglas de transición puras: `TRANSICIONES`, `destino_de()` |
| `jacobs/routes.py` | `POST /jacobs/pipeline/{id}/{discard,recover,hide,restore}`; `/cancel` rechaza los estados nuevos |
| `tests/test_jacobs_descarte.py` (nuevo) | reglas puras + rutas con store simulado (sin DB) |
| `tests/test_jacobs_descarte_db.py` (nuevo) | CAS real, columnas, índices y `EXPLAIN` (con DB) |

**jax-platform** (worktree `~/worktrees/jax-platform-descartar`, rama `feat/descartar-pipelines`)

| Archivo | Responsabilidad |
|---|---|
| `backend/api/pipelines.py` | 4 proxies con guardia, filtro de la lista, `?estado=discarded`, 404 a un `hidden` |
| `backend/api/admin/pipelines_ocultos.py` (nuevo) | `GET /api/admin/pipelines/ocultos` (superadmin) |
| `backend/main.py` (o donde se registran routers) | registrar el router nuevo |
| `backend/tests/test_pipelines_descarte.py` (nuevo) | guardias, filtro, 404, proxy |
| `frontend/src/components/RightPanel/RightPanel.jsx` | botón Descartar + `Dialogo` |
| `frontend/src/components/historial/HistorialContenido.jsx` | pestaña Descartados + Recuperar + Borrar |
| `frontend/src/pages/admin/AdminPipelinesOcultos.jsx` (nuevo) | vista de ocultos + Restaurar |
| `frontend/src/pages/Admin.jsx`, `frontend/src/components/BarraUsuario.jsx` | ruta y enlace de la vista de ocultos |
| `frontend/src/i18n/es.js`, `en.js` | textos |

---

### Task 1: Estados nuevos, columnas e índices (jax)

**Files:**
- Modify: `jacobs/models.py` (enum `PipelineStatus`, ~línea 27)
- Modify: `jacobs/policy.py:47` (`ESTADOS_SIN_CUPO`)
- Modify: `jacobs/store.py:940-951` (`_INDICES`) y `:1037` (lista de `ALTER TABLE ... ADD COLUMN`)
- Test: `tests/test_jacobs_descarte.py` (sin DB), `tests/test_jacobs_descarte_db.py` (con DB)

**Interfaces:**
- Produces:
  - `PipelineStatus.discarded == "discarded"`, `PipelineStatus.hidden == "hidden"`.
  - Columnas `jacobs_pipelines.status_previo VARCHAR(20) NULL`, `descartado_por VARCHAR(50) NULL`, `descartado_at DOUBLE NULL`.
  - Índices `idx_pipelines_descartados (user_id, tenant_id, status, descartado_at)` e `idx_pipelines_ocultos (status, descartado_at)`.

- [ ] **Step 1: Test que falla (sin DB)**

```python
# tests/test_jacobs_descarte.py
"""Descartar pipelines detenidos (spec 2026-09-22-descartar-pipelines-design). Sin DB."""
from __future__ import annotations

from base_de_test import fijar_base_de_test  # noqa: E402

fijar_base_de_test()

from jacobs.models import PipelineStatus  # noqa: E402
from jacobs import policy  # noqa: E402


def test_los_estados_nuevos_existen():
    assert PipelineStatus("discarded") is PipelineStatus.discarded
    assert PipelineStatus("hidden") is PipelineStatus.hidden


def test_descartado_y_oculto_no_ocupan_cupo():
    assert PipelineStatus.discarded in policy.ESTADOS_SIN_CUPO
    assert PipelineStatus.hidden in policy.ESTADOS_SIN_CUPO
    assert PipelineStatus.discarded not in policy.ESTADOS_QUE_OCUPAN_CUPO
    assert PipelineStatus.hidden not in policy.ESTADOS_QUE_OCUPAN_CUPO
```

- [ ] **Step 2: Correr y ver el rojo**

Run: `python -m pytest tests/test_jacobs_descarte.py -v`
Expected: FAIL con `ValueError: 'discarded' is not a valid PipelineStatus`.

- [ ] **Step 3: Implementar el enum y el cupo**

En `jacobs/models.py`, dentro de `class PipelineStatus(str, Enum)`, después de `disputed`:

```python
    # 2026-09-22 (spec descartar-pipelines): terminales y SIN cupo. `discarded`
    # = el dueño lo sacó de "Detenidos" (recuperable); `hidden` = el
    # superadmin lo ocultó (restaurable). Ninguno borra filas.
    discarded   = "discarded"
    hidden      = "hidden"
```

En `jacobs/policy.py`, en `ESTADOS_SIN_CUPO`, después de `PipelineStatus.disputed,`:

```python
    # 2026-09-22 (spec descartar-pipelines): descartado y oculto son terminales.
    PipelineStatus.discarded, PipelineStatus.hidden,
```

- [ ] **Step 4: Correr y ver el verde**

Run: `python -m pytest tests/test_jacobs_descarte.py -v`
Expected: 2 passed.

- [ ] **Step 5: Test con DB de columnas e índices**

```python
# tests/test_jacobs_descarte_db.py
"""Descartar pipelines: columnas, índices y CAS contra la base de TEST."""
from __future__ import annotations

from base_de_test import fijar_base_de_test  # noqa: E402

fijar_base_de_test()

import pytest  # noqa: E402

from jacobs import store  # noqa: E402

pytestmark = pytest.mark.asyncio


async def _columnas():
    pool = await store.get_pool()
    async with pool.acquire() as conn, conn.cursor() as cur:
        await cur.execute("SHOW COLUMNS FROM jacobs_pipelines")
        return {fila[0]: fila[1] for fila in await cur.fetchall()}


async def test_columnas_del_descarte_existen():
    await store.init_schema()
    cols = await _columnas()
    assert cols["status_previo"].lower() == "varchar(20)"
    assert cols["descartado_por"].lower() == "varchar(50)"
    assert cols["descartado_at"].lower() == "double"


async def test_indices_del_descarte_existen():
    await store.init_schema()
    pool = await store.get_pool()
    async with pool.acquire() as conn, conn.cursor() as cur:
        await cur.execute("SHOW INDEX FROM jacobs_pipelines")
        filas = await cur.fetchall()
    por_indice: dict[str, list[str]] = {}
    for f in filas:
        por_indice.setdefault(f[2], []).append((f[3], f[4]))
    cols = {k: [c for _, c in sorted(v)] for k, v in por_indice.items()}
    assert cols["idx_pipelines_descartados"] == ["user_id", "tenant_id", "status", "descartado_at"]
    assert cols["idx_pipelines_ocultos"] == ["status", "descartado_at"]
```

> Si `store.get_pool` / `store.init_schema` se llaman distinto, usa los nombres reales de `jacobs/store.py`, los mismos que usa `tests/test_jacobs_reaper_cas_db.py`, y ajusta el test. No inventes un helper.

- [ ] **Step 6: Rojo, implementar, verde**

Run: `python -m pytest tests/test_jacobs_descarte_db.py -v` → FAIL (`KeyError: 'status_previo'`).

En `jacobs/store.py`, en la lista `for col, ddl in [...]` de `jacobs_pipelines` (~línea 1037), al final:

```python
                # 2026-09-22 (spec descartar-pipelines §3): a qué vuelve al
                # recuperar, quién descartó (decide quién puede recuperar) y
                # cuándo (orden de la vista). INSTANT: nunca copiar la tabla.
                ("status_previo", "ALTER TABLE jacobs_pipelines ADD COLUMN "
                    "status_previo VARCHAR(20) NULL, ALGORITHM=INSTANT"),
                ("descartado_por", "ALTER TABLE jacobs_pipelines ADD COLUMN "
                    "descartado_por VARCHAR(50) NULL, ALGORITHM=INSTANT"),
                ("descartado_at", "ALTER TABLE jacobs_pipelines ADD COLUMN "
                    "descartado_at DOUBLE NULL, ALGORITHM=INSTANT"),
```

En `_INDICES` (~línea 951), después de `idx_jacobs_pipelines_duenio`:

```python
    # 2026-09-22 (spec descartar-pipelines §6): la vista "Descartados" filtra
    # por dueño + status y ordena por descartado_at; la de ocultos (todos los
    # usuarios) por status + descartado_at. Sin estos, EXPLAIN da filesort.
    ("jacobs_pipelines", "idx_pipelines_descartados",
     "CREATE INDEX idx_pipelines_descartados ON jacobs_pipelines "
     "(user_id, tenant_id, status, descartado_at) ALGORITHM=INPLACE LOCK=NONE", True),
    ("jacobs_pipelines", "idx_pipelines_ocultos",
     "CREATE INDEX idx_pipelines_ocultos ON jacobs_pipelines "
     "(status, descartado_at) ALGORITHM=INPLACE LOCK=NONE", True),
```

Run otra vez → 2 passed.

- [ ] **Step 7: El reaper no toca los estados nuevos**

Agrega a `tests/test_jacobs_descarte.py`:

```python
import inspect  # noqa: E402

from jacobs import reaper  # noqa: E402


def test_el_reaper_solo_cosecha_no_terminales():
    fuente = inspect.getsource(reaper)
    assert "[PipelineStatus.pending, PipelineStatus.running, PipelineStatus.interrupted]" in fuente
    assert "PipelineStatus.discarded" not in fuente
    assert "PipelineStatus.hidden" not in fuente
```

Run: pasa sin tocar el reaper. Es una baranda contra regresiones, y así va declarada en la nota del piso de CI.

- [ ] **Step 8: Commit**

```bash
git add jacobs/models.py jacobs/policy.py jacobs/store.py tests/test_jacobs_descarte.py tests/test_jacobs_descarte_db.py
git commit -m "feat(jacobs): estados discarded/hidden, columnas e índices del descarte"
```

---

### Task 2: Transición compare-and-set del descarte (jax)

**Files:**
- Create: `jacobs/descarte.py`
- Modify: `jacobs/store.py` (función nueva después de `pipeline_update_status_si_epoca`, ~línea 1514)
- Test: `tests/test_jacobs_descarte.py`, `tests/test_jacobs_descarte_db.py`

**Interfaces:**
- Consumes: estados y columnas de la Task 1.
- Produces:
  - `jacobs.descarte.TRANSICIONES: dict[str, frozenset[PipelineStatus]]`, con las claves `"discard"`, `"recover"`, `"hide"` y `"restore"`.
  - `jacobs.descarte.destino_de(accion: str, status_previo: str | None) -> PipelineStatus`
  - `jacobs.descarte.validar_transicion(accion: str, desde: PipelineStatus, a: PipelineStatus) -> None` (fix round 1,
    Ruling 7): levanta `descarte.TransicionDescarteInvalida` (fail-closed) si `desde` no está en
    `TRANSICIONES[accion]`, o si `a` no es el destino correcto. Para `discard`/`hide`/`restore`, `a`
    tiene que ser exactamente `destino_de(accion, None)`. Para `recover`, `a` sólo se valida contra el
    conjunto general de estados previos posibles (`TRANSICIONES["discard"]`) -- que coincida con el
    `status_previo` REAL de esa fila lo garantiza el propio `WHERE` del compare-and-set
    (`status_previo=%s` con `a.value`), no esta función.
  - `store.pipeline_transicion_descarte(pipeline_id: str, epoca: int, accion: str, *, desde: PipelineStatus, a: PipelineStatus, user_id: str, evento_tipo: str, evento_payload: dict) -> bool`
    (fix round 1, Ruling 7 y Ruling 9 -- firma actualizada en la Task 3, fix round 1, 2026-09-22):
    llama a `descarte.validar_transicion` ANTES de tocar la base -- una transición inválida (p.ej.
    `discard` desde `running`) levanta `TransicionDescarteInvalida` sin escribir nada. En `recover`,
    además de `desde=discarded` y `run_epoch`, el `WHERE` exige `status_previo=a.value`: un `recover`
    con un `a` que no coincide con el `status_previo` guardado devuelve `False` (no escribe) en vez de
    levantar, porque `a` sí era válido en general, sólo no coincidía con ESTA fila. `user_id` sólo se
    persiste en `discard` (columna `descartado_por`); en `recover`/`hide`/`restore` no se escribe en
    ninguna columna. **Ruling 9 (Task 3, fix round 1):** `evento_tipo`/`evento_payload` -- el CAS y el
    INSERT en `jacobs_events` van en la MISMA transacción (`conexion_dedicada(found_rows=True)` +
    `transaccion()`, reutilizados del store). Si el UPDATE no escribe (rowcount != 1), no se inserta
    el evento y devuelve `False`. Si el INSERT falla, la transacción se descarta entera (la conexión
    se cierra, no se manda un `ROLLBACK` explícito -- mismo patrón que el resto del store) y la
    excepción sube: el estado nunca cambia sin su evento. La ruta (Task 3) ya NO llama a
    `store.event_append` por su cuenta.

- [ ] **Step 1: Test de las reglas puras (falla)**

```python
from jacobs import descarte  # noqa: E402


def test_transiciones_permitidas():
    assert descarte.TRANSICIONES["discard"] == frozenset({PipelineStatus.aborted, PipelineStatus.expired})
    assert descarte.TRANSICIONES["recover"] == frozenset({PipelineStatus.discarded})
    assert descarte.TRANSICIONES["hide"] == frozenset({PipelineStatus.discarded})
    assert descarte.TRANSICIONES["restore"] == frozenset({PipelineStatus.hidden})


def test_recuperar_vuelve_al_estado_exacto_previo():
    assert descarte.destino_de("recover", "expired") is PipelineStatus.expired
    assert descarte.destino_de("recover", "aborted") is PipelineStatus.aborted


def test_recuperar_sin_estado_previo_valido_es_error():
    import pytest
    for malo in (None, "", "running", "discarded"):
        with pytest.raises(descarte.EstadoPrevioInvalido):
            descarte.destino_de("recover", malo)


def test_destinos_fijos():
    assert descarte.destino_de("discard", None) is PipelineStatus.discarded
    assert descarte.destino_de("hide", None) is PipelineStatus.hidden
    assert descarte.destino_de("restore", None) is PipelineStatus.discarded
```

Run: `python -m pytest tests/test_jacobs_descarte.py -v` → FAIL (`ModuleNotFoundError: jacobs.descarte`).

- [ ] **Step 2: Implementar `jacobs/descarte.py`**

```python
"""Reglas del descarte de pipelines detenidos (spec 2026-09-22-descartar-pipelines).

Puras, sin I/O: qué estado puede pasar a cuál. La escritura (compare-and-set)
vive en store.pipeline_transicion_descarte; quién puede pedirla, en
jax-platform. Ninguna transición borra filas.
"""
from __future__ import annotations

from jacobs.models import PipelineStatus

TRANSICIONES: dict[str, frozenset[PipelineStatus]] = {
    "discard": frozenset({PipelineStatus.aborted, PipelineStatus.expired}),
    "recover": frozenset({PipelineStatus.discarded}),
    "hide": frozenset({PipelineStatus.discarded}),
    "restore": frozenset({PipelineStatus.hidden}),
}

#: A qué puede volver un recuperado: exactamente lo que se podía descartar.
_PREVIOS_VALIDOS = TRANSICIONES["discard"]


class EstadoPrevioInvalido(ValueError):
    """`status_previo` no es un estado del que se pueda haber descartado.
    Fail-closed: no se adivina a qué volver."""


def destino_de(accion: str, status_previo: str | None) -> PipelineStatus:
    if accion == "discard":
        return PipelineStatus.discarded
    if accion == "hide":
        return PipelineStatus.hidden
    if accion == "restore":
        return PipelineStatus.discarded
    if accion == "recover":
        try:
            previo = PipelineStatus(status_previo)
        except ValueError:
            raise EstadoPrevioInvalido(repr(status_previo)) from None
        if previo not in _PREVIOS_VALIDOS:
            raise EstadoPrevioInvalido(repr(status_previo))
        return previo
    raise KeyError(accion)
```

Run → los 4 tests nuevos pasan.

- [ ] **Step 3: Test de CAS con DB (falla)**

Agrega a `tests/test_jacobs_descarte_db.py` un helper que inserte un pipeline con el mismo `pipeline_create` que usa `tests/test_jacobs_reaper_cas_db.py` (mismo `Pipeline(...)`, con `user_id="u1"`, `tenant_id="1"`, `run_epoch=3`) y estos tests:

```python
async def test_descartar_escribe_estado_y_columnas_en_la_misma_escritura(pipeline_abortado):
    pid = pipeline_abortado  # status aborted, run_epoch 3
    ok = await store.pipeline_transicion_descarte(
        pid, 3, "discard", desde=PipelineStatus.aborted, a=PipelineStatus.discarded, user_id="u1")
    assert ok is True
    fila = await _fila(pid)  # SELECT status, status_previo, descartado_por, descartado_at
    assert fila == ("discarded", "aborted", "u1", pytest.approx(fila[3]))
    assert fila[3] is not None


async def test_descartar_con_epoca_vieja_no_escribe(pipeline_abortado):
    ok = await store.pipeline_transicion_descarte(
        pipeline_abortado, 2, "discard", desde=PipelineStatus.aborted,
        a=PipelineStatus.discarded, user_id="u1")
    assert ok is False
    assert (await _fila(pipeline_abortado))[0] == "aborted"


async def test_descartar_desde_otro_estado_no_escribe(pipeline_abortado):
    ok = await store.pipeline_transicion_descarte(
        pipeline_abortado, 3, "discard", desde=PipelineStatus.expired,
        a=PipelineStatus.discarded, user_id="u1")
    assert ok is False


async def test_recuperar_limpia_las_tres_columnas(pipeline_abortado):
    await store.pipeline_transicion_descarte(
        pipeline_abortado, 3, "discard", desde=PipelineStatus.aborted,
        a=PipelineStatus.discarded, user_id="u1")
    ok = await store.pipeline_transicion_descarte(
        pipeline_abortado, 3, "recover", desde=PipelineStatus.discarded,
        a=PipelineStatus.aborted, user_id="u1")
    assert ok is True
    assert await _fila(pipeline_abortado) == ("aborted", None, None, None)


async def test_ocultar_y_restaurar_conservan_las_columnas(pipeline_abortado):
    await store.pipeline_transicion_descarte(
        pipeline_abortado, 3, "discard", desde=PipelineStatus.aborted,
        a=PipelineStatus.discarded, user_id="u1")
    antes = await _fila(pipeline_abortado)
    assert await store.pipeline_transicion_descarte(
        pipeline_abortado, 3, "hide", desde=PipelineStatus.discarded,
        a=PipelineStatus.hidden, user_id="admin")
    assert (await _fila(pipeline_abortado))[1:] == antes[1:]
    assert await store.pipeline_transicion_descarte(
        pipeline_abortado, 3, "restore", desde=PipelineStatus.hidden,
        a=PipelineStatus.discarded, user_id="admin")
    assert await _fila(pipeline_abortado) == ("discarded",) + antes[1:]
```

Run → FAIL (`AttributeError: pipeline_transicion_descarte`).

- [ ] **Step 4: Implementar en `jacobs/store.py`**

Después de `pipeline_update_status_si_epoca`:

```python
#: SET por acción del descarte (spec 2026-09-22 §3). Una sola sentencia por
#: transición: estado y columnas se escriben JUNTOS o no se escribe nada.
#: discard recibe `status_previo` como PARÁMETRO (el `desde` leído) y no como
#: `status_previo=status`: así no depende del orden en que MariaDB evalúa
#: las asignaciones del SET.
_SETS_DESCARTE = {
    "discard": "status=%s, updated_at=%s, status_previo=%s, "
               "descartado_por=%s, descartado_at=%s",
    "recover": "status=%s, updated_at=%s, status_previo=NULL, "
               "descartado_por=NULL, descartado_at=NULL",
    "hide": "status=%s, updated_at=%s",
    "restore": "status=%s, updated_at=%s",
}


async def pipeline_transicion_descarte(
    pipeline_id: str,
    epoca: int,
    accion: str,
    *,
    desde: PipelineStatus,
    a: PipelineStatus,
    user_id: str,
) -> bool:
    """Compare-and-set de una transición del descarte. True si escribió
    (el pipeline estaba en `epoca` y en `desde`)."""
    ahora = time.time()
    sets = _SETS_DESCARTE[accion]  # KeyError si la acción no existe: error de contrato
    params: list = [a.value, ahora]
    if accion == "discard":
        params += [desde.value, user_id, ahora]
    params += [pipeline_id, epoca, desde.value]
    sql = (f"UPDATE jacobs_pipelines SET {sets} "
           "WHERE pipeline_id=%s AND run_epoch=%s AND status=%s")
    return await _ejecutar_condicional(sql, params) == 1
```

Run → los 5 tests pasan.

- [ ] **Step 5: Mutaciones (una por arreglo, se revierte cada una)**

1. Quitar `AND run_epoch=%s` (y su parámetro): `test_descartar_con_epoca_vieja_no_escribe` tiene que caer.
2. En recover, no limpiar `descartado_por`: `test_recuperar_limpia_las_tres_columnas` tiene que caer.
3. En hide, limpiar `status_previo`: `test_ocultar_y_restaurar_conservan_las_columnas` tiene que caer.

Anota cuáles cayeron en el mensaje del commit.

- [ ] **Step 6: Commit**

```bash
git add jacobs/descarte.py jacobs/store.py tests/test_jacobs_descarte.py tests/test_jacobs_descarte_db.py
git commit -m "feat(jacobs): transición compare-and-set del descarte, sin borrar filas"
```

---

### Task 3: Rutas de Jacobs, eventos y rechazos de cancel/continue (jax)

**Files:**
- Modify: `jacobs/routes.py` (después de `cancel_pipeline`, ~línea 785; y la tupla de estados de `cancel_pipeline`, ~línea 760)
- Test: `tests/test_jacobs_descarte.py`

**Interfaces:**
- Consumes: `descarte.TRANSICIONES`, `descarte.destino_de`, `descarte.EstadoPrevioInvalido`, `store.pipeline_transicion_descarte`, `store.pipeline_get`, `store.pipeline_status_previo`.
  **Fix round 1 (Ruling 9):** la ruta ya NO consume `store.event_append` directo -- le pasa
  `evento_tipo`/`evento_payload` a `store.pipeline_transicion_descarte`, que los escribe en la MISMA
  transacción que el CAS (ver Interfaces de la Task 2, arriba).
- Produces:
  - Ruling 1 (controlador): CUATRO rutas LITERALES, no la genérica `/{accion}` del brief --
    `POST /jacobs/pipeline/{id}/discard`, `/recover`, `/hide`, `/restore`, cada una llamando a la
    función común `transicion_descarte(pipeline_id, accion, req)`. Cuerpo `{"user_id": str}` con
    `Field(min_length=1)` (fix round 1, M4: un `user_id` vacío es 422 de Pydantic, antes de que el
    handler toque el store).
  - Respuesta 200: `{"pipeline_id", "status"}`.
  - 404 `{"code": "pipeline_no_encontrado"}` si no existe.
  - 409 `{"code": "transicion_no_permitida", "status": <actual>}`, o `{"code": "cambio_concurrente"}`
    (Ruling 8: sólo se atrapa `descarte.EstadoPrevioInvalido`; `TransicionDescarteInvalida` y un
    `ValueError` genérico no se atrapan -- son bugs de contrato, 500).
  - 422 `{"code": "estado_previo_invalido"}`.
  - Eventos `PIPELINE_DISCARDED`, `PIPELINE_RECOVERED`, `PIPELINE_HIDDEN` y `PIPELINE_RESTORED`, con
    payload `{"user_id", "desde", "a"}` -- escritos por `store.pipeline_transicion_descarte`, no por
    esta ruta (Ruling 9).

- [ ] **Step 1: Tests de rutas con store simulado (fallan)**

Usa el mismo patrón que `tests/test_jacobs_cancel_reaper_epoca.py` (`patch.object(routes.store, ...)` con `AsyncMock`):

```python
import asyncio  # noqa: E402
import time  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from jacobs import routes  # noqa: E402
from jacobs.models import Pipeline  # noqa: E402


def _p(status, previo=None):
    ahora = time.time()
    p = Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma", mode="autonomous",
                 status=status, run_epoch=3, created_at=ahora, updated_at=ahora)
    return p, previo


def _llamar(accion, status, cas=True, previo=None):
    pipeline, _ = _p(status)
    trans, eventos = AsyncMock(return_value=cas), AsyncMock()
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=pipeline)), \
         patch.object(routes.store, "pipeline_status_previo", AsyncMock(return_value=previo)), \
         patch.object(routes.store, "pipeline_transicion_descarte", trans), \
         patch.object(routes.store, "event_append", eventos):
        r = asyncio.run(routes.transicion_descarte("p1", accion, routes.DescarteRequest(user_id="u1")))
    return r, trans, eventos


def test_descartar_un_abortado():
    r, trans, eventos = _llamar("discard", PipelineStatus.aborted)
    assert r == {"pipeline_id": "p1", "status": "discarded"}
    trans.assert_awaited_once_with("p1", 3, "discard", desde=PipelineStatus.aborted,
                                   a=PipelineStatus.discarded, user_id="u1")
    eventos.assert_awaited_once_with("p1", "PIPELINE_DISCARDED",
                                     {"user_id": "u1", "desde": "aborted", "a": "discarded"})


@pytest.mark.parametrize("accion,status", [
    ("discard", PipelineStatus.running), ("discard", PipelineStatus.completed),
    ("recover", PipelineStatus.aborted), ("hide", PipelineStatus.aborted),
    ("restore", PipelineStatus.discarded),
])
def test_transicion_no_permitida_es_409(accion, status):
    with pytest.raises(HTTPException) as e:
        _llamar(accion, status)
    assert e.value.status_code == 409
    assert e.value.detail["code"] == "transicion_no_permitida"


def test_carrera_perdida_es_409_y_no_emite_evento():
    with pytest.raises(HTTPException) as e:
        _llamar("discard", PipelineStatus.aborted, cas=False)
    assert e.value.detail["code"] == "cambio_concurrente"


def test_recuperar_vuelve_a_expired():
    r, trans, _ = _llamar("recover", PipelineStatus.discarded, previo="expired")
    assert r["status"] == "expired"
    assert trans.await_args.kwargs["a"] is PipelineStatus.expired


def test_recuperar_con_previo_corrupto_es_422():
    with pytest.raises(HTTPException) as e:
        _llamar("recover", PipelineStatus.discarded, previo="running")
    assert e.value.status_code == 422


@pytest.mark.parametrize("status", [PipelineStatus.discarded, PipelineStatus.hidden])
def test_cancel_rechaza_descartados_y_ocultos(status):
    pipeline, _ = _p(status)
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=pipeline)):
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.cancel_pipeline("p1"))
    assert e.value.status_code == 409


def test_continue_no_acepta_descartados():
    from jacobs.continuar import ESTADOS_CONTINUABLES
    assert PipelineStatus.discarded not in ESTADOS_CONTINUABLES
    assert PipelineStatus.hidden not in ESTADOS_CONTINUABLES
```

Run → FAIL (`AttributeError: transicion_descarte`).

- [ ] **Step 2: `store.pipeline_status_previo`**

En `jacobs/store.py`:

```python
async def pipeline_status_previo(pipeline_id: str) -> str | None:
    """`status_previo` de un pipeline descartado (spec descartar §3)."""
    async with _conexion_o_pool(None) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT status_previo FROM jacobs_pipelines WHERE pipeline_id=%s", (pipeline_id,))
            fila = await cur.fetchone()
    return fila[0] if fila else None
```

- [ ] **Step 3: La ruta, en `jacobs/routes.py`**

Import arriba: `from jacobs import descarte`. Después de `cancel_pipeline`:

```python
# ----------------------------------------------------------------
#  POST /jacobs/pipeline/{id}/{discard,recover,hide,restore}
#  (spec 2026-09-22-descartar-pipelines). Quién puede pedirla lo decide
#  jax-platform; acá solo se valida la transición y se escribe con CAS.
# ----------------------------------------------------------------

class DescarteRequest(BaseModel):
    user_id: str


_EVENTO_DE = {
    "discard": "PIPELINE_DISCARDED", "recover": "PIPELINE_RECOVERED",
    "hide": "PIPELINE_HIDDEN", "restore": "PIPELINE_RESTORED",
}


@router.post("/pipeline/{pipeline_id}/{accion}")
async def transicion_descarte(pipeline_id: str, accion: str, req: DescarteRequest) -> dict:
    if accion not in descarte.TRANSICIONES:
        raise HTTPException(status_code=404, detail={"code": "accion_desconocida"})
    pipeline = await store.pipeline_get(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail={"code": "pipeline_no_encontrado"})
    if pipeline.status not in descarte.TRANSICIONES[accion]:
        raise HTTPException(status_code=409, detail={
            "code": "transicion_no_permitida", "status": pipeline.status.value})
    previo = await store.pipeline_status_previo(pipeline_id) if accion == "recover" else None
    try:
        destino = descarte.destino_de(accion, previo)
    except descarte.EstadoPrevioInvalido:
        raise HTTPException(status_code=422, detail={"code": "estado_previo_invalido"}) from None
    if not await store.pipeline_transicion_descarte(
        pipeline_id, pipeline.run_epoch, accion,
        desde=pipeline.status, a=destino, user_id=req.user_id,
    ):
        raise HTTPException(status_code=409, detail={"code": "cambio_concurrente"})
    await store.event_append(pipeline_id, _EVENTO_DE[accion], {
        "user_id": req.user_id, "desde": pipeline.status.value, "a": destino.value})
    return {"pipeline_id": pipeline_id, "status": destino.value}
```

⚠️ **Orden de rutas.** `/pipeline/{pipeline_id}/{accion}` es genérica y FastAPI toma la primera que coincide. Esta función va DESPUÉS de todas las rutas `POST /pipeline/{id}/<literal>` (cancel, resume, continue, approve-step y las demás). Agrega un test que haga `POST /jacobs/pipeline/p1/cancel` con `TestClient` y verifique que llega a `cancel_pipeline` (store simulado), no a `transicion_descarte`. Si el orden resulta frágil, usa cuatro rutas literales que llamen a una función común. Es más explícito y, en ese caso, preferible: decídelo y dilo.

En `cancel_pipeline`, agrega a la tupla de estados que dan 409:

```python
        # 2026-09-22 (spec descartar-pipelines): descartado y oculto son terminales.
        PipelineStatus.discarded, PipelineStatus.hidden,
```

- [ ] **Step 4: Verde y mutaciones**

Run: `python -m pytest tests/test_jacobs_descarte.py -v` → todo en verde.

Mutaciones:
- Quitar `discarded, hidden` de la tupla de cancel: su test tiene que caer.
- Emitir el evento antes del CAS: `test_carrera_perdida...` tiene que caer, porque el test también verifica que `eventos` NO fue llamado. Agrega `eventos.assert_not_awaited()` en ese test.

- [ ] **Step 5: Autenticación de LAS MANOS**

`las_manos/auth_servicio.py` ya admite `POST /jacobs/.+` para la identidad `plataforma`, así que no hay que tocarlo. Verifícalo con un test en `tests/test_jacobs_descarte.py`:

```python
def test_plataforma_puede_llamar_las_rutas_del_descarte():
    import auth_servicio
    permiso = auth_servicio.PERMISOS[auth_servicio.IDENTIDAD_PLATAFORMA]
    for accion in ("discard", "recover", "hide", "restore"):
        assert permiso.admite_ruta("POST", f"/jacobs/pipeline/p1/{accion}")
```

La identidad `jacobs` NO debe poder: comprueba si su `Permiso` admite esas rutas. Si las admite, restríngelas y agrega el test del caso negativo.

- [ ] **Step 6: Suite, piso y commit**

Run: la suite completa sin DB (`JAX_CI_NO_DB=1 python -m pytest -q`) y con DB, igual que en `.github/workflows/policy.yml`. Actualiza los pisos con su nota: di qué tests prueban el defecto y cuáles son barandas.

```bash
git add jacobs/routes.py jacobs/store.py tests/test_jacobs_descarte.py .github/workflows/policy.yml
git commit -m "feat(jacobs): rutas discard/recover/hide/restore con CAS y eventos; cancel rechaza terminales nuevos"
```

- [ ] **Step 7: PR de jax**

Push y PR `feat(jacobs): descartar, recuperar, ocultar y restaurar pipelines detenidos`. En la descripción incluye el plan y el spec. Revisión adversarial (escalón 3) antes del merge. Este PR se despliega ANTES que el de jax-platform.

---

### Task 4: API de jax-platform (proxies, guardias, filtro, ocultos)

**Files:**
- Modify: `backend/api/pipelines.py` (`SQL_PIPELINES_DEL_USUARIO` ~694, `list_pipelines` ~812, `_require_pipeline_owner` ~661; proxies después de `cancel_pipeline` ~1010)
- Create: `backend/api/admin/pipelines_ocultos.py`
- Modify: el archivo donde se registran los routers de admin (busca `include_router` de `admin.memoria`)
- Test: `backend/tests/test_pipelines_descarte.py`

**Interfaces:**
- Consumes: rutas de Jacobs de la Task 3.
- Produces:
  - `POST /api/pipelines/{id}/discard|recover|hide|restore`, que responden con el JSON de Jacobs o con su mismo código de error.
  - `GET /api/pipelines?estado=discarded&limite&offset`, cuyas filas traen `descartado_at`.
  - `GET /api/admin/pipelines/ocultos?limite&offset`, cuyas filas traen `user_id` y `descartado_por`.

- [ ] **Step 1: Tests (fallan)**

Usa los fixtures que usan los tests existentes de `pipelines.py` (busca `client_superadmin` y el cliente de usuario normal en `backend/tests/conftest.py`), y el cliente HTTP a Jacobs simulado como en los tests de `cancel`. Casos, cada uno en su test:

1. `POST /discard` de un pipeline ajeno → 404 `pipeline_no_encontrado`, y no se llama a Jacobs.
2. `POST /discard` del dueño → se llama a `{JACOBS_URL}/pipeline/{id}/discard` con `json={"user_id": user.user_id}` y la cabecera de `encabezados_las_manos()`.
3. `POST /recover` de un usuario distinto de `descartado_por`, que no es superadmin → 403 `recuperar_no_permitido`.
4. `POST /recover` del superadmin sobre el descartado de otro → 200.
5. `POST /hide` y `/restore` de un no superadmin → 403.
6. `GET /api/pipelines` excluye `discarded` y `hidden` (siembra uno de cada uno en la base de test).
7. `GET /api/pipelines?estado=discarded` devuelve solo los descartados del usuario, ordenados por `descartado_at DESC`.
8. `GET /api/pipelines/{id}` de un `hidden` para su dueño no superadmin → 404.
9. `GET /api/admin/pipelines/ocultos` de un no superadmin → 403; del superadmin → los `hidden` de TODOS los usuarios.
10. `EXPLAIN` de las tres consultas nuevas o cambiadas: `key` es el índice esperado (`idx_jacobs_pipelines_duenio`, `idx_pipelines_descartados`, `idx_pipelines_ocultos`) y `Extra` no contiene `Using filesort`. Mismo patrón que los `EXPLAIN` de `test_historial_pipelines.py`.
11. `axioma_usage` intacto: siembra dos filas de uso para un pipeline, ocúltalo por la API con Jacobs simulado y con la transición aplicada directo en la base de test. La suma de `costo` del mes es la misma antes y después.

- [ ] **Step 2: SQL de las listas**

```python
SQL_PIPELINES_DEL_USUARIO = (
    "SELECT pipeline_id, name, status, created_at, updated_at FROM jacobs_pipelines "
    "WHERE user_id=%s AND tenant_id=%s AND owner_ack_at IS NOT NULL "
    "AND status NOT IN ('discarded','hidden') "
    "ORDER BY created_at DESC LIMIT %s OFFSET %s"
)
# 2026-09-22 (spec descartar-pipelines §4): la vista "Descartados". Va por
# idx_pipelines_descartados (user_id, tenant_id, status, descartado_at).
SQL_DESCARTADOS_DEL_USUARIO = (
    "SELECT pipeline_id, name, status, created_at, updated_at, descartado_at "
    "FROM jacobs_pipelines "
    "WHERE user_id=%s AND tenant_id=%s AND owner_ack_at IS NOT NULL AND status='discarded' "
    "ORDER BY descartado_at DESC LIMIT %s OFFSET %s"
)
```

En `list_pipelines`, agrega `estado: str | None = Query(None, pattern="^discarded$")`. Si `estado == "discarded"`, usa `SQL_DESCARTADOS_DEL_USUARIO` y sus filas de 6 columnas: la desestructuración de filas cambia, así que respétala en el código que arma la respuesta y en el cálculo de `detenidos`, que para esta vista va vacío.

- [ ] **Step 3: El 404 del oculto**

En `_require_pipeline_owner`, agrega `status` al SELECT y rechaza con 404 si es `hidden` y el usuario no es superadmin. Usa el mismo criterio de rol que `require_superadmin` en `backend/auth/middleware.py:113`: lee cómo lo decide y reutilízalo, no lo copies. Y ajusta a todos los que desestructuran esa fila.

- [ ] **Step 4: Los proxies**

```python
async def _proxy_descarte(pipeline_id: str, accion: str, user: AuthUser) -> dict:
    client = await get_http_client()
    try:
        r = await client.post(f"{JACOBS_URL}/pipeline/{pipeline_id}/{accion}", timeout=10.0,
                              json={"user_id": str(user.user_id)},
                              headers=encabezados_las_manos())
    except Exception as e:  # noqa: BLE001 -- 502 con motivo, mismo contrato que cancel
        raise HTTPException(status_code=502, detail={
            "code": "jacobs_no_responde", "motivo": recortar_redactado(str(e), 200)}) from None
    return _json_de_jacobs(r)


@router.post("/{pipeline_id}/discard")
async def discard_pipeline(pipeline_id: str, user: AuthUser = Depends(get_current_user)):
    await _require_pipeline_owner(pipeline_id, user)
    return await _proxy_descarte(pipeline_id, "discard", user)


@router.post("/{pipeline_id}/recover")
async def recover_pipeline(pipeline_id: str, user: AuthUser = Depends(get_current_user)):
    await _require_pipeline_owner(pipeline_id, user)
    if not _es_superadmin(user) and await _descartado_por(pipeline_id) != str(user.user_id):
        raise HTTPException(status_code=403, detail="recuperar_no_permitido")
    return await _proxy_descarte(pipeline_id, "recover", user)


@router.post("/{pipeline_id}/hide")
async def hide_pipeline(pipeline_id: str, user: AuthUser = Depends(require_superadmin)):
    return await _proxy_descarte(pipeline_id, "hide", user)


@router.post("/{pipeline_id}/restore")
async def restore_pipeline(pipeline_id: str, user: AuthUser = Depends(require_superadmin)):
    return await _proxy_descarte(pipeline_id, "restore", user)
```

Detalles a resolver al implementar:
- **`_es_superadmin(user)`:** es el mismo criterio que `require_superadmin`. Si en `auth/middleware.py` ya existe un predicado, úsalo.
- **`_descartado_por(pipeline_id)`:** `SELECT descartado_por FROM jacobs_pipelines WHERE pipeline_id=%s`.
- **Un superadmin que no es dueño** no pasa `_require_pipeline_owner` en recover. Ramifica: si es superadmin, solo exige que el pipeline exista.
- **`_json_de_jacobs`:** verifica que propague el `status_code` y el `detail` de Jacobs (409/422). Si no, ajústalo con un test.
- **El `except Exception`** tiene que pasar el detector `no-fail-open-except`: relanza, igual que `cancel`.

- [ ] **Step 5: `backend/api/admin/pipelines_ocultos.py`**

```python
"""Pipelines ocultos (spec 2026-09-22-descartar-pipelines §4-5): la vista del
superadmin. Todos los usuarios, paginada, por idx_pipelines_ocultos."""
from fastapi import APIRouter, Depends, Query

from auth.middleware import require_superadmin
from auth.models import AuthUser
from db.connection import get_pool

router = APIRouter(prefix="/api/admin/pipelines")

LIMITE_MAX = 50

SQL_OCULTOS = (
    "SELECT pipeline_id, name, user_id, tenant_id, descartado_por, descartado_at, created_at "
    "FROM jacobs_pipelines WHERE status='hidden' "
    "ORDER BY descartado_at DESC LIMIT %s OFFSET %s"
)


@router.get("/ocultos")
async def listar_ocultos(
    limite: int = Query(LIMITE_MAX, ge=1, le=LIMITE_MAX),
    offset: int = Query(0, ge=0),
    user: AuthUser = Depends(require_superadmin),
):
    pool = await get_pool()
    async with pool.acquire() as conn, conn.cursor() as cur:
        await cur.execute(SQL_OCULTOS, (limite + 1, offset))
        filas = await cur.fetchall()
    campos = ("pipeline_id", "name", "user_id", "tenant_id", "descartado_por",
              "descartado_at", "created_at")
    return {"pipelines": [dict(zip(campos, f)) for f in filas[:limite]],
            "hay_mas": len(filas) > limite}
```

Regístralo donde se registran los routers de admin (`include_router`).

- [ ] **Step 6: Verde, mutaciones, piso y commit**

Mutaciones:
- Quitar el `status NOT IN (...)` del listado: el test 6 tiene que caer.
- Quitar la guardia de recover: el test 3 tiene que caer.
- Quitar el 404 del oculto: el test 8 tiene que caer.

Después, las suites con y sin DB, y los pisos de `policy.yml` con su nota.

```bash
git add backend/api/pipelines.py backend/api/admin/pipelines_ocultos.py backend/tests/test_pipelines_descarte.py <archivo-de-routers> .github/workflows/policy.yml
git commit -m "feat(pipelines): descartar/recuperar/ocultar/restaurar con permisos, y listas sin los descartados"
```

---

### Task 5: Botón Descartar en "Detenidos" (frontend)

**Files:**
- Modify: `frontend/src/components/RightPanel/RightPanel.jsx:234-250`
- Modify: `frontend/src/i18n/es.js`, `en.js`
- Test: `frontend/src/components/RightPanel/RightPanel.descartar.test.jsx` (nuevo; mismo estilo que los tests existentes de RightPanel)

**Interfaces:**
- Consumes: `POST /pipelines/{id}/discard` (con `api` de `frontend/src/api`).
- Produces: las claves i18n `descartarPipeline`, `descartarTitulo`, `descartarMensaje(nombre)`, `descartarConfirmar`, `descartarError` y `cancelar` (si `cancelar` ya existe, reutilízala).

- [ ] **Step 1: Test (falla)**

1. Cada tarjeta de Detenidos tiene un botón con el texto `t.descartarPipeline`.
2. Al pulsarlo se abre un `Dialogo` con `role="dialog"` que muestra el nombre del pipeline. No se llama `api.post` todavía.
3. Al confirmar se llama `api.post('/pipelines/p1/discard')`, la tarjeta desaparece y se vuelve a pedir `/pipelines`.
4. Si falla, aparece el error traducido (`mensajeDeError`) y la tarjeta sigue.
5. Escape cierra el diálogo sin llamar a la API.

- [ ] **Step 2: Implementar**

Debajo del botón Continuar de cada tarjeta:

```jsx
<button type="button" onClick={() => setADescartar(p)}
  className="mt-1 w-full py-1.5 rounded-lg border border-borde text-texto-suave hover:text-texto text-xs font-semibold transition-colors">
  {t.descartarPipeline}
</button>
```

Y al final, junto a `ContinuarPipelineModal`:

```jsx
{aDescartar && (
  <Dialogo idTitulo="titulo-descartar" titulo={t.descartarTitulo}
    onCerrar={() => setADescartar(null)}>
    <p className="text-sm text-texto mb-4">{t.descartarMensaje(aDescartar.name)}</p>
    {errorDescartar && <AlertaError className="mb-3 text-xs">{errorDescartar}</AlertaError>}
    <div className="flex justify-end gap-2">
      <button type="button" onClick={() => setADescartar(null)}
        className="px-3 py-1.5 rounded-lg border border-borde text-sm">{t.cancelar}</button>
      <button type="button" onClick={confirmarDescartar} disabled={descartando}
        className="px-3 py-1.5 rounded-lg bg-accion hover:bg-accion-hover text-sobre-color text-sm font-semibold">
        {t.descartarConfirmar}
      </button>
    </div>
  </Dialogo>
)}
```

`confirmarDescartar` hace `api.post(...)`. Si sale bien, quita la tarjeta de `detenidos` y cierra. Si sale mal, `setErrorDescartar(mensajeDeError(t, err))`. Verifica que el tamaño de toque de los botones pasa los detectores de #141 (`frontend/src/politica/tamanoDeToque.test.js`), y usa las constantes de `frontend/src/tema/botones.js` si corresponde.

Textos:
- es: `descartarPipeline: 'Descartar'`, `descartarTitulo: 'Descartar pipeline'`, `descartarMensaje: (n) => \`"${n}" sale de Detenidos. Lo puedes recuperar desde Historial → Descartados.\``, `descartarConfirmar: 'Descartar'`.
- en: los equivalentes.

- [ ] **Step 3: Verde, vitest completo, piso y commit**

```bash
git commit -m "feat(panel): descartar un pipeline detenido, con confirmación en ventana propia"
```

---

### Task 6: Pestaña "Descartados" en el historial (frontend)

**Files:**
- Modify: `frontend/src/components/historial/HistorialContenido.jsx`
- Modify: `frontend/src/i18n/es.js`, `en.js`
- Test: `frontend/src/components/historial/HistorialContenido.descartados.test.jsx` (nuevo)

**Interfaces:**
- Consumes: `GET /pipelines?estado=discarded&limite&offset`, `POST /pipelines/{id}/recover`, `POST /pipelines/{id}/hide`, y `useJaxStore((s) => s.user?.role === 'superadmin')`.
- Produces: las claves `pestanaTodos`, `pestanaDescartados`, `recuperarPipeline`, `borrarPipeline`, `borrarTitulo`, `borrarMensaje(nombre)`, `sinDescartados` y `cargarMas` (reutilízala si existe).

- [ ] **Step 1: Test (falla)**

1. Hay dos pestañas, y "Todos" es la inicial.
2. "Descartados" pide `GET /pipelines?estado=discarded&limite=50&offset=0` y lista los pipelines con su fecha de descarte.
3. Si `hay_mas`, "Cargar más" pide `offset=50`.
4. **Recuperar** llama `POST /pipelines/{id}/recover` y quita la fila.
5. **Borrar** aparece solo para el superadmin. Abre `ConfirmacionSuma` y, al confirmar, llama `POST /pipelines/{id}/hide`.
6. Un usuario que no es superadmin NO ve "Borrar".
7. La lista vacía muestra `t.sinDescartados`.

- [ ] **Step 2: Implementar**

La pestaña Descartados tiene su propio estado local: lista, `hayMas`, `cargando`, `error` y `offset`. No toca el store `historial`, porque los descartados no son parte del historial principal. Usa la tabla existente como modelo de marcado.

`ConfirmacionSuma` se usa con `titulo`, `mensaje`, `textoConfirmar`, `onConfirmar`, `onCancelar` y `numeros={numerosAlAzar()}`, igual que en `frontend/src/pages/Memoria.jsx`.

- [ ] **Step 3: Verde, vitest, piso y commit**

```bash
git commit -m "feat(historial): pestaña Descartados con recuperar y, para el superadmin, borrar (ocultar)"
```

---

### Task 7: Administración → Pipelines ocultos (frontend)

**Files:**
- Create: `frontend/src/pages/admin/AdminPipelinesOcultos.jsx`
- Modify: `frontend/src/pages/Admin.jsx:21-33` (ruta `pipelines-ocultos`), `frontend/src/components/BarraUsuario.jsx:~214` (enlace junto al de Memoria, solo si `esSuperadmin`)
- Modify: i18n
- Test: `frontend/src/pages/admin/AdminPipelinesOcultos.test.jsx`

**Interfaces:**
- Consumes: `GET /admin/pipelines/ocultos`, `POST /pipelines/{id}/restore`.
- Produces: las claves `pipelinesOcultosTitulo`, `restaurarPipeline`, `sinOcultos` y `ocultoDe(usuario)`.

- [ ] **Step 1: Test (falla)**

1. Lista paginada con nombre, dueño (`user_id`) y fecha.
2. **Restaurar** llama `POST /pipelines/{id}/restore` y quita la fila. No lleva confirmación, porque es reversible.
3. Si la API responde 403, se muestra el error traducido.
4. El enlace en `BarraUsuario` aparece solo para el superadmin.

- [ ] **Step 2: Implementar y registrar**

En `Admin.jsx`: `<Route path="pipelines-ocultos" element={<AdminPipelinesOcultos />} />`. En `BarraUsuario.jsx`, un enlace a `/admin/pipelines-ocultos` con el mismo marcado que el de `/admin/memoria`.

- [ ] **Step 3: Verde, vitest, piso y commit**

```bash
git commit -m "feat(admin): vista de pipelines ocultos con restaurar"
```

---

### Task 8: Carga, escaneo final, PR y despliegue (jax-platform)

**Files:**
- Modify: `.github/workflows/policy.yml` (pisos)
- Modify: `docs/` de jax-platform o `DEUDA.md` de jax, donde se registran las mediciones (sigue el patrón del repo)

- [ ] **Step 1: Carga (LAS CUATRO #4)**

Mide `GET /api/pipelines` y `GET /api/pipelines?estado=discarded` contra la base de TEST, con el usuario de más pipelines sembrado a escala: como mínimo 5.000 pipelines del usuario, 20 % descartados. Usa el mismo método que las mediciones previas del repo (busca `loadtest/`). Registra p50 y p95 y la concurrencia en la que empieza a degradarse, antes y después del cambio. Sin número no hay GO.

- [ ] **Step 2: Escaneo de diálogos**

```bash
git grep -nE "(^|[^.A-Za-z_])(confirm|alert|prompt)\(" -- frontend/src | grep -v test
```

Expected: ninguna coincidencia nueva.

- [ ] **Step 3: PR**

Push y PR `feat: descartar, recuperar, ocultar y restaurar pipelines detenidos`. Revisión adversarial (escalón 3) antes del merge.

- [ ] **Step 4: Despliegue (GO de Fernando)**

1. Se despliega jax (PR de las Tasks 1–3) en `/srv/jax-prod/jax` y se reinicia `jax-las-manos` con 0 pipelines en vuelo. `init_schema` crea las columnas y los índices. Verificación: `SHOW COLUMNS` y `SHOW INDEX` en producción, solo lectura.
2. Se despliega el backend de jax-platform en `/srv/jax-prod/jax-platform` y se reinicia `jax-platform`.
3. Se despliega el frontend: build de master, respaldo de `atem-ai:/www/wwwroot/axioma-ia.io/`, rsync con `--exclude .user.ini` en los dos saltos, y verificación desde fuera (hash del bundle).
4. Fernando lo verifica en vivo: descarta el pipeline `prueba-freno-en-vuelo` de su captura, lo ve en Historial → Descartados, lo recupera y lo vuelve a descartar.
