# Ejecutor SP1 · Plan 3 · C4 freno en vuelo para el proxy y la cuenta del Ejecutor

> **Para agentes ejecutores:** SUB-SKILL REQUERIDA: usar `superpowers:subagent-driven-development`
> (recomendado) o `superpowers:executing-plans`. Los pasos usan casillas (`- [ ]`).

**Objetivo:** que al poner el interruptor de JAX mueran, en menos de un segundo y sin volver a nacer, el cerebro

> **Enmienda 2026-09-17 (plan 4 ejecutado):** C5 no escribe el interruptor global sino la pausa PROPIA del Ejecutor
> (`jax/ejecutor/contratos/pausa.py`, `JAX_EJECUTOR_PAUSA`). El freno root y el proxy de este plan tienen que actuar
> con **cualquiera de las dos** puestas (el interruptor de JAX o la pausa del Ejecutor), con los mismos tests. El
> proxy ya responde 423 con la pausa del Ejecutor o sin latido del vigía (`tests/test_ejecutor_proxy_pausa.py`).
del Ejecutor (su proxy corta el stream en curso y rechaza los nuevos) y **todos** los procesos de la cuenta
`axioma` —los desprendidos con `setsid`/`nohup` y los que corren al otro lado de un `ssh -tt`—, verificado dos
veces con segundos de por medio.

> **EJECUTADO 2026-09-17 (Mr. Hyde, subagente) — lo que manda es la rama `feat/ejecutor-c4`.** Correcciones, cada una
> con test o prueba real vista en rojo:
> 1. **Dos frenos:** proxy y freno root actúan con el interruptor de JAX **o** la pausa del Ejecutor (C5). La prueba real
>    usa la pausa del Ejecutor (`probar_c4.py --freno ejecutor`, por omisión), no el interruptor global.
> 2. **Proxy:** el 423 del freno va DESPUÉS de anotar los resultados (C3) y el vigía en vuelo arranca recién ahí; en
>    vuelo, 423 legible si no salieron cabeceras y stream truncado si ya salieron.
> 3. **Remotas habilitadas a propósito** (`JAX_EJECUTOR_FRENO_REMOTOS`); `remotos_cargados` sólo con todas. La política
>    se carga con `politica.cargar`. El freno nunca mata root, uid < 1000 ni al administrador. Latido 0644.
> 4. **`ejecutor-freno-remoto`** salva también a `$PPID` (el `sshd-session` de la cuenta), y da tres vueltas.
> 5. **INSTALABLES** suma `pausa.py`; la unidad entra en el manifiesto; `--rama-aprobada` explícito para instalar fuera
>    de master.
> 6. **`probar_c4.py`:** freno puesto recién con el escenario corriendo; zombis no cuentan; `--remoto` (plan 5 Task 7);
>    `--lecturas`. Ensayo remoto completo en `probar_freno_remoto_en_contenedor.sh`.
> 7. **C6:** `revertir_en_maquina.sh` con varias líneas de known_hosts, y `esperar_sshd` tras el reload.

**Arquitectura:** **no se toca la base del kill switch**: la hace el frente B (`feat/kill-switch-real`,
`jax/core/interruptor.py`: `JAX_KILL_SWITCH_PATH`, `interruptor_activo`, `INTERVALO_DE_SONDEO`, `escribir_pausa`,
`borrar_pausa`; producción `/etc/jax/interruptor/PAUSE`; la ruta heredada `/etc/jax/PAUSE` sigue frenando). Este plan
agrega lo que esa base no cubre: (1) el proxy vigila el interruptor por petición; (2) un servicio **root**
`ejecutor-freno` que, mientras el interruptor esté puesto, escribe `1` en `cgroup.kill` de `user-<uid>.slice` y
barre `/proc` por uid cada 250 ms, y lanza el barrido remoto con una llave de comando forzado; (3) el comando forzado
`ejecutor-freno-remoto`; (4) cron/at/linger cerrados para la cuenta, que sacarían procesos de su cgroup.

**Tech stack:** Python 3.12 (CI) / 3.14 (root en hall9000, `/usr/bin/python3 -I`), sólo biblioteca estándar en lo
instalado; cgroup v2 (`cgroup.kill`, kernel 7.0 medido); systemd; `sh`.

**Spec:** `docs/superpowers/specs/2026-09-15-ejecutor-design.md` §4 C4. Decisión D-SP1-3 del índice
`docs/superpowers/plans/2026-09-17-ejecutor-contratos-0-indice.md`. Base: plan del frente B
`/home/fruiz/worktrees/jax-platform-hallazgos-docs/docs/superpowers/plans/2026-09-16-frente-b-kill-switch.md`.

## DEPENDENCIA DURA

**Este plan no empieza hasta que `feat/kill-switch-real` (frente B) esté mergeado en `master` de jax y de
jax-platform y desplegado.** Primer paso de la Task 1: verificarlo con evidencia, no suponerlo:
`git -C /home/fruiz/jax log --oneline master -- jax/core/interruptor.py | head -1` no vacío, y
`grep -c '^def interruptor_activo\|^async def correr_con_interruptor\|^INTERVALO_DE_SONDEO' /home/fruiz/jax/jax/core/interruptor.py`
→ `3`. Si las firmas cambiaron respecto de las que usa este plan (Interfaces), se para y se adapta el plan. **No se
editan archivos del frente B** (worktrees `jax-frente-b`, `jax-platform-frente-b` y lo que ya mergearon).

## Global Constraints

- **Nunca cambiar de rama en `/home/fruiz/jax`.** Worktree `/home/fruiz/worktrees/jax-sp1-c4`, rama `feat/ejecutor-c4`
  desde `master` con el frente B y los planes 1 y 2 mergeados. Siempre `git -C <ruta absoluta>`.
- **`/etc/jax/.env` apunta a PRODUCCIÓN.** Ningún test la carga. El `conftest.py` de la raíz (frente B) ya fuerza
  `JAX_KILL_SWITCH_PATH` a un temporal; los tests de este plan usan su propio `tmp_path` con `monkeypatch.setenv`.
- **P10:** todo `except` amplio sin `raise`, o con cuerpo `pass`, lleva `# fail-soft: <razón específica>` en la
  MISMA línea. `python3 policy/tests/test_no_fail_open_except.py` verde.
- **Piso de `tests-puros`:** tests nuevos en **las dos** listas; el `grep` del piso con el número **del runner**.
- **Mutaciones:** `PYTHONDONTWRITEBYTECODE=1`; backup `.mut-bak` y `cmp`; nunca `git checkout` para restaurar.
- **Tests con procesos:** `multiprocessing.get_context("fork")` explícito cuando el objetivo es una función; para
  procesos del sistema, `subprocess.Popen(..., start_new_session=True)`; vida con `ps -o pid= -p <pid>` o
  `/proc/<pid>/stat` (estado `Z` = muerto sin cosechar). **Nunca `pgrep -f`.** **Ningún test mata procesos por uid
  real**: el barrido por uid se prueba con un `/proc` falso, y el comando forzado con un `ps` falso — un test que
  corriera `ejecutor-freno-remoto` de verdad como `fruiz` mataría la sesión de Fernando.
- **Cero strings visibles hardcodeados:** códigos. **Sin hardcoding:** cuenta, llave, rutas desde el entorno;
  `/opt/ejecutor/lib` se hornea en la unidad al instalar (render), como el gancho del plan 1.
- **Lo instalado en `/opt/ejecutor/lib` es sólo biblioteca estándar** (el test del plan 1 lo impone; este plan
  amplía `INSTALABLES` con `jax/core/__init__.py`, `jax/core/interruptor.py` —medido: sólo importa `asyncio`,
  `logging`, `os`, `tempfile`, `pathlib`— y `jax/ejecutor/contratos/freno.py`).
- **Nada destructivo sobre .10/.11/.20.** El barrido remoto real se instala y prueba en el plan 5 Parte B.
- **La prueba real pone el freno GLOBAL:** la Mesa responde 423 mientras dura (~10 s por ronda). Sólo con la Mesa
  sin uso real en los 5 minutos previos (journal de `jax-platform`); si aparece uso, se aborta y se reintenta.
- Commits en castellano, terminados con `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

## Mapa de archivos

| Archivo | Acción | Responsabilidad |
|---|---|---|
| `jax/ejecutor/proxy_carril.py` | Modificar | 423 con el freno puesto; corte en vuelo; no arranca sin ruta del freno |
| `jax/ejecutor/contratos/freno.py` | Crear | `cgroup.kill` + barrido de `/proc` por uid + barrido remoto; latido |
| `ops/ejecutor/ejecutor-freno-remoto` | Crear | comando forzado: mata lo de la cuenta menos su propia sesión |
| `ops/ejecutor/ejecutor-freno.service.plantilla` | Crear | unidad root |
| `jax/ejecutor/contratos/instalacion.py` | Modificar | `INSTALABLES` + `renderizar_unidad_freno` |
| `ops/ejecutor/instalar_freno.sh` | Crear | instalación root, cron/at/linger cerrados |
| `scripts/ejecutor_contratos/probar_c4.py` | Crear | simulacro real: control, dos rondas, dos lecturas por ronda |
| `tests/test_ejecutor_proxy_freno.py`, `tests/test_ejecutor_contratos_freno.py`, `tests/test_ejecutor_freno_remoto.py` | Crear | tests puros |
| `tests/test_ejecutor_contratos_instalacion.py` | Modificar | permitir `jax.core.interruptor` si está en `INSTALABLES`; render de la unidad |
| `.github/workflows/policy.yml` | Modificar | listas y piso |

## Interfaces

Consume (frente B, `jax/core/interruptor.py`, verificado en su rama el 2026-09-17):
```python
VARIABLE_RUTA = "JAX_KILL_SWITCH_PATH"
class InterruptorSinConfigurar(RuntimeError)
def ruta_del_interruptor() -> Path
def interruptor_activo(ruta: Path | str | None = None) -> bool   # también mira la ruta heredada
def escribir_pausa(ruta: Path, contenido: str) -> bool
def borrar_pausa(ruta: Path) -> bool
INTERVALO_DE_SONDEO = 0.25
```
Consume (planes 1 y 2): `cuenta_axioma.Cuenta`, `correr_en_la_cuenta`, `politica.validar`, `Fallo`, `formato.campos`.

Produce:
```python
# jax/ejecutor/proxy_carril.py
KILL_SWITCH_ACTIVO = "kill_switch_activo"

# jax/ejecutor/contratos/freno.py
REPASOS_S = (0.0, 2.0, 10.0)
@dataclass(frozen=True) class Remoto: nombre: str; ip: str; puerto: int
@dataclass(frozen=True) class ConfigFreno: uid: int | None; cuenta: str; cgroup: Path; proc: Path; llave: Path; known_hosts: Path; remotos: tuple; estado: Path
def ruta_cgroup_kill(cgroup: Path, uid: int) -> Path
def matar_cgroup(cgroup: Path, uid: int) -> bool
def pids_de(proc: Path, uid: int) -> list[int]
def matar_pids(pids, matar=os.kill) -> int
def argv_barrido(r: Remoto, cfg: ConfigFreno) -> list[str]
def barrer(r: Remoto, cfg: ConfigFreno, correr=subprocess.run) -> str        # "ok" | "fallo" | "tiempo_agotado"
class Freno:
    def __init__(self, cfg, *, activo=None, reloj=time.monotonic, lanzar_barrido=None, matar=os.kill)
    def paso(self) -> dict
def config_desde_entorno(env) -> tuple[ConfigFreno, bool]   # (config, remotos_cargados)
def principal() -> int
# latido en cfg.estado (JSON): {"momento": epoch, "activo": bool, "remotos_cargados": bool, "uid_resuelto": bool}

# jax/ejecutor/contratos/instalacion.py
def renderizar_unidad_freno(lib: str) -> str
```

---

### Task 1: el proxy frena

**Files:**
- Modify: `/home/fruiz/worktrees/jax-sp1-c4/jax/ejecutor/proxy_carril.py` (import, constante, `atender`, comienzo de `_reenviar` después de anotar resultados, `arrancar`)
- Test: `/home/fruiz/worktrees/jax-sp1-c4/tests/test_ejecutor_proxy_freno.py`

**Interfaces:** Consume `interruptor.interruptor_activo`, `ruta_del_interruptor`, `InterruptorSinConfigurar`,
`INTERVALO_DE_SONDEO`. Produce `KILL_SWITCH_ACTIVO`.

**Por qué un vigía propio y no `correr_con_interruptor`:** con el freno ya puesto, `correr_con_interruptor` lanza
antes de correr la corrutina, y el arnés vería una conexión cortada en vez de un 423 legible con
`x-should-retry: false` (sin esa cabecera reintenta 10 veces, medido en §6.1 del spec de Fase 2). Se reusa lo que
decide —`interruptor_activo` y el intervalo— y no la forma de cortar.

- [ ] **Step 1: Verificar la dependencia (ver «DEPENDENCIA DURA»)**

Run: `git -C /home/fruiz/jax log --oneline master -- jax/core/interruptor.py | head -1 && grep -c '^def interruptor_activo\|^async def correr_con_interruptor\|^INTERVALO_DE_SONDEO' /home/fruiz/jax/jax/core/interruptor.py`
Expected: un commit y `3`. Si no, **STOP**.

- [ ] **Step 2: Escribir los tests que fallan**

```python
# tests/test_ejecutor_proxy_freno.py
"""C4 en el proxy: con el interruptor puesto, el cerebro del Ejecutor no responde
(423 sin tocar el upstream) y un stream en curso se corta en menos de un segundo."""
import asyncio
import json
import time

import httpx
import pytest

from jax.core import interruptor
from jax.ejecutor import proxy_carril
from tests.test_ejecutor_proxy_carril import Proxy, Upstream, _correr


@pytest.fixture
def freno(tmp_path, monkeypatch):
    ruta = tmp_path / "interruptor" / "PAUSE"
    ruta.parent.mkdir()
    monkeypatch.setenv(interruptor.VARIABLE_RUTA, str(ruta))
    return ruta


def test_con_el_freno_puesto_423_sin_tocar_el_upstream(tmp_path, freno):
    freno.write_text("{}")

    async def escenario():
        async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 2) as px, httpx.AsyncClient() as cli:
            r = await cli.post(px.url + "/v1/messages", content=b"{}")
            return r.status_code, r.headers.get("x-should-retry"), r.json(), len(up.recibidas)

    assert _correr(escenario()) == (423, "false", {"type": "error", "error": {"type": "kill_switch_activo"}}, 0)


def test_el_freno_corta_el_stream_en_vuelo_y_suelta_el_carril(tmp_path, freno):
    async def escenario():
        async with Upstream(n_trozos=3) as up, Proxy(up.url, tmp_path, 5) as px, httpx.AsyncClient() as cli:
            cortado, demora = False, None
            try:
                async with cli.stream("POST", px.url + "/v1/messages", content=b"{}") as r:
                    trozos = r.aiter_raw()
                    await asyncio.wait_for(anext(trozos), 3)
                    inicio = time.monotonic()
                    freno.write_text("{}")
                    async for _ in trozos:
                        pass
            except httpx.RemoteProtocolError:
                cortado, demora = True, time.monotonic() - inicio
            freno.unlink()
            up.avanzar.set()
            up.n_trozos = 1  # la siguiente respuesta no espera a que el test avance
            siguiente = await cli.post(px.url + "/v1/messages", content=b"{}")
            return cortado, demora, siguiente.status_code

    cortado, demora, siguiente = _correr(escenario())
    assert cortado is True
    assert demora < 1.0, f"el corte tardó {demora:.2f} s"
    assert siguiente == 200, "el carril quedó tomado después del freno"


def test_sin_ruta_del_freno_el_proxy_no_arranca(tmp_path, monkeypatch):
    monkeypatch.delenv(interruptor.VARIABLE_RUTA, raising=False)
    cfg = proxy_carril.Config(upstream="http://127.0.0.1:9", raiz=tmp_path, tope_s=1, host="127.0.0.1", puerto=0,
                              registro=tmp_path / "registro.jsonl")
    with pytest.raises(interruptor.InterruptorSinConfigurar):
        _correr(proxy_carril.arrancar(cfg))
```

- [ ] **Step 3: Correr y ver que fallan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c4 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_proxy_freno.py -v`
Expected: FAIL: el primero da 200, el segundo no corta (vence por tiempo), el tercero no lanza.

- [ ] **Step 4: Implementar**

Import y constante en `proxy_carril.py`:

```python
from jax.core import interruptor

KILL_SWITCH_ACTIVO = "kill_switch_activo"
```

Función de módulo, junto a `_esperar_cierre`:

```python
def _freno_puesto() -> bool:
    try:
        return interruptor.interruptor_activo()
    except interruptor.InterruptorSinConfigurar:
        return True  # sin saber dónde está el freno, se frena (arrancar ya lo exige; esto cubre que se borre después)


async def _esperar_freno() -> None:
    while not _freno_puesto():
        await asyncio.sleep(interruptor.INTERVALO_DE_SONDEO)
```

`atender` completo:

```python
    async def atender(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        conn = h11.Connection(h11.SERVER)
        try:
            try:
                peticion, cuerpo = await _leer_peticion(conn, reader)
            except h11.RemoteProtocolError:
                await _responder_error(conn, writer, 400, Motivo(PETICION_INVALIDA))
                return
            if peticion is None:
                return
            # Con el freno YA puesto no se vigila: _reenviar contesta 423 legible. Si se
            # vigilara, el vigía ganaría la carrera y el arnés vería un corte, no un 423.
            ya_frenado = _freno_puesto()
            trabajo = asyncio.create_task(self._reenviar(conn, writer, peticion, cuerpo))
            vigia = asyncio.create_task(_esperar_cierre(reader))
            freno = asyncio.create_task(asyncio.sleep(0) if ya_frenado else _esperar_freno())
            await asyncio.wait({trabajo, vigia} if ya_frenado else {trabajo, vigia, freno},
                               return_when=asyncio.FIRST_COMPLETED)
            frenado = not ya_frenado and freno.done() and not trabajo.done()
            if not trabajo.done():
                # Cortó el cliente o se puso el freno: cancelar suelta el carril (espera o
                # stream) y cierra la respuesta del upstream.
                log.info("proxy_carril %s metodo=%s ruta=%s", KILL_SWITCH_ACTIVO if frenado else "corte_cliente",
                         peticion.method.decode("latin-1"), _ruta_sin_query(peticion.target))
                trabajo.cancel()
                await asyncio.wait({trabajo})
                if frenado:
                    # Sin EndOfMessage: el arnés ve un stream truncado, nunca uno completo.
                    writer.transport.abort()
            for t in (vigia, freno):
                t.cancel()
            await asyncio.wait({vigia, freno})
            if not trabajo.cancelled() and trabajo.exception() is not None:
                exc = trabajo.exception()
                if not isinstance(exc, ConnectionError):
                    raise exc
                log.info("proxy_carril corte_cliente tipo=%s", type(exc).__name__)
        finally:
            await _cerrar(writer)
```

En `_reenviar`, inmediatamente **después** del bloque que anota los resultados (plan 2) y **antes** de
`try: async with carril_ejecutor_async(...)` — lo que ya corrió se anota aunque el freno esté puesto:

```python
        if _freno_puesto():
            log.warning("proxy_carril %s metodo=%s ruta=%s", KILL_SWITCH_ACTIVO, metodo, ruta)
            await _responder_error(conn, writer, 423, Motivo(KILL_SWITCH_ACTIVO),
                                   extra=((b"x-should-retry", b"false"),))
            return
```

En `arrancar`, primera línea:

```python
    interruptor.ruta_del_interruptor()  # InterruptorSinConfigurar: sin saber dónde está el freno, no hay cerebro
```

- [ ] **Step 5: Correr (nuevos y todos los del proxy)**

Run: `cd /home/fruiz/worktrees/jax-sp1-c4 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_proxy_freno.py tests/test_ejecutor_proxy_carril.py tests/test_ejecutor_proxy_registro.py -v && PYTHONPATH=.:las_manos python -m pytest tests/test_no_blocking_in_async.py -q && python3 policy/tests/test_no_fail_open_except.py`
Expected: todos PASS. Si `test_no_blocking_in_async.py` marca el `os.stat` de `interruptor_activo` dentro de un
`async def`, **no se le agrega excepción**: se mueve la llamada a `await asyncio.to_thread(_freno_puesto)` en los dos
sitios y se vuelve a medir el corte (< 1 s).

- [ ] **Step 6: Mutación**

Con backup y `cmp`: en `atender`, `if frenado:` → `if frenado and False:`. Expected: FAIL
`test_el_freno_corta_el_stream_en_vuelo_y_suelta_el_carril` (sin abort el cliente no ve el corte). Restaurar.

- [ ] **Step 7: Commit**

```bash
git -C /home/fruiz/worktrees/jax-sp1-c4 add jax/ejecutor/proxy_carril.py tests/test_ejecutor_proxy_freno.py
git -C /home/fruiz/worktrees/jax-sp1-c4 commit -m "feat(ejecutor): el proxy del Ejecutor obedece al interruptor, antes y en vuelo

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: el freno root — `cgroup.kill`, `/proc` por uid, barrido remoto y latido

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-c4/jax/ejecutor/contratos/freno.py`
- Test: `/home/fruiz/worktrees/jax-sp1-c4/tests/test_ejecutor_contratos_freno.py`

**Interfaces:** Consume `interruptor` (frente B), `politica.validar` (plan 1). Produce todo `freno.py` (ver Interfaces).

**Por qué dos formas de matar local:** `cgroup.kill` es atómico y alcanza lo que nace mientras se mata, pero sólo
dentro de `user-<uid>.slice` (lo que entra por ssh con `pam_systemd`). Un proceso de la cuenta fuera de su slice
(p. ej. uno lanzado por `cron`, cuyo PAM en Ubuntu no pasa por `pam_systemd`) no está ahí. El barrido de `/proc` por
uid real o efectivo lo alcanza. Y cron/at se cierran para la cuenta en la Task 4.

**Por qué el latido:** un freno que se cayó no avisa. `ejecutor-freno` escribe su estado cada vuelta; el arranque
(plan 6) exige un latido de menos de 2 s y `remotos_cargados`. Si la política no se puede leer, el freno **sigue
matando local** (no depende de la política) y lo dice en el latido.

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_ejecutor_contratos_freno.py
"""Freno root del Ejecutor (C4) sin root: cgroup y /proc falsos, barrido remoto
falso. Nada de este archivo manda una señal a un proceso real que no haya creado."""
import json
import signal
import subprocess
from pathlib import Path

import pytest

from jax.ejecutor.contratos import freno as F


def _cfg(tmp_path, remotos=(), uid=1001):
    return F.ConfigFreno(uid=uid, cuenta="axioma", cgroup=tmp_path / "cgroup", proc=tmp_path / "proc",
                         llave=Path("/etc/jax-ejecutor/freno/id_freno"),
                         known_hosts=Path("/etc/jax-ejecutor/freno/known_hosts"), remotos=tuple(remotos),
                         estado=tmp_path / "estado.json")


def _proc(tmp_path, procesos):
    for pid, (real, efectivo) in procesos.items():
        d = tmp_path / "proc" / str(pid)
        d.mkdir(parents=True)
        (d / "status").write_text(f"Name:\tx\nUid:\t{real}\t{efectivo}\t{efectivo}\t{efectivo}\n")
    (tmp_path / "proc" / "self").mkdir(parents=True, exist_ok=True)


def test_cgroup_kill_escribe_1_si_hay_slice(tmp_path):
    ruta = F.ruta_cgroup_kill(tmp_path / "cgroup", 1001)
    assert ruta == tmp_path / "cgroup" / "user.slice" / "user-1001.slice" / "cgroup.kill"
    assert F.matar_cgroup(tmp_path / "cgroup", 1001) is False
    ruta.parent.mkdir(parents=True)
    ruta.write_text("")
    assert F.matar_cgroup(tmp_path / "cgroup", 1001) is True
    assert ruta.read_text() == "1"


def test_pids_por_uid_real_o_efectivo(tmp_path):
    _proc(tmp_path, {10: (1001, 1001), 11: (0, 1001), 12: (1001, 0), 13: (1000, 1000)})
    assert sorted(F.pids_de(tmp_path / "proc", 1001)) == [10, 11, 12]


def test_matar_pids_tolera_los_que_ya_murieron():
    vistos = []

    def matar(pid, sig):
        vistos.append((pid, sig))
        if pid == 2:
            raise ProcessLookupError

    assert F.matar_pids([1, 2, 3], matar) == 2
    assert vistos == [(1, signal.SIGKILL), (2, signal.SIGKILL), (3, signal.SIGKILL)]


def test_argv_del_barrido_sin_comando_y_con_llave_propia(tmp_path):
    argv = F.argv_barrido(F.Remoto("bridge", "192.0.2.20", 58291), _cfg(tmp_path))
    assert argv[0] == "ssh" and argv[-1] == "axioma@192.0.2.20"
    assert ["-i", "/etc/jax-ejecutor/freno/id_freno"] == argv[1:3]
    assert "IdentitiesOnly=yes" in argv and "StrictHostKeyChecking=yes" in argv and "BatchMode=yes" in argv


@pytest.mark.parametrize("rc, salida, esperado", [
    (0, b"freno_remoto=ok quedan=0\n", "ok"),
    (0, b"freno_remoto=ok quedan=2\n", "fallo"),
    (255, b"", "fallo"),
])
def test_barrer_exige_cero_procesos(tmp_path, rc, salida, esperado):
    def correr(argv, **k):
        return subprocess.CompletedProcess(argv, rc, salida, b"")
    assert F.barrer(F.Remoto("bridge", "192.0.2.20", 58291), _cfg(tmp_path), correr) == esperado


def test_barrer_con_tiempo_agotado(tmp_path):
    def correr(argv, **k):
        raise subprocess.TimeoutExpired(argv, 15)
    assert F.barrer(F.Remoto("bridge", "192.0.2.20", 58291), _cfg(tmp_path), correr) == "tiempo_agotado"


class Reloj:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_paso_suelto_no_mata_nada(tmp_path):
    _proc(tmp_path, {10: (1001, 1001)})
    muertos = []
    fr = F.Freno(_cfg(tmp_path), activo=lambda: False, matar=lambda p, s: muertos.append(p),
                 lanzar_barrido=lambda r: pytest.fail("barrió suelto"))
    assert fr.paso() == {"activo": False}
    assert muertos == []
    assert json.loads((tmp_path / "estado.json").read_text())["activo"] is False


def test_paso_puesto_mata_local_cada_vuelta_y_barre_en_sus_repasos(tmp_path):
    _proc(tmp_path, {10: (1001, 1001), 13: (1000, 1000)})
    remotos = (F.Remoto("bridge", "192.0.2.20", 58291), F.Remoto("prod", "192.0.2.10", 58291))
    reloj, muertos, barridos = Reloj(), [], []
    fr = F.Freno(_cfg(tmp_path, remotos), activo=lambda: True, reloj=reloj, matar=lambda p, s: muertos.append(p),
                 lanzar_barrido=lambda rs: barridos.append((reloj.t, tuple(r.nombre for r in rs))))
    for delta in (0.0, 0.25, 1.9, 2.1, 5.0, 10.5, 11.0):
        reloj.t = 100.0 + delta
        fr.paso()
    assert muertos == [10] * 7, "el barrido local corre en CADA vuelta"
    assert barridos == [(100.0, ("bridge", "prod")), (102.1, ("bridge", "prod")), (110.5, ("bridge", "prod"))]


def test_al_soltar_y_volver_a_poner_repasa_desde_cero(tmp_path):
    _proc(tmp_path, {})
    reloj, estado, barridos = Reloj(), {"puesto": True}, []
    fr = F.Freno(_cfg(tmp_path, (F.Remoto("bridge", "192.0.2.20", 58291),)), activo=lambda: estado["puesto"],
                 reloj=reloj, matar=lambda p, s: None, lanzar_barrido=lambda rs: barridos.append(reloj.t))
    fr.paso()
    estado["puesto"] = False
    reloj.t = 101.0
    fr.paso()
    estado["puesto"] = True
    reloj.t = 102.0
    fr.paso()
    assert barridos == [100.0, 102.0]


def test_interruptor_sin_configurar_cuenta_como_puesto(tmp_path, monkeypatch):
    monkeypatch.delenv("JAX_KILL_SWITCH_PATH", raising=False)
    assert F.Freno._interruptor() is True


def test_config_sin_politica_legible_sigue_matando_local(tmp_path, monkeypatch):
    env = {"JAX_EJECUTOR_CUENTA": "no-existe-esta-cuenta", "JAX_EJECUTOR_FRENO_LLAVE": "/k",
           "JAX_EJECUTOR_FRENO_KNOWN_HOSTS": "/kh", "JAX_EJECUTOR_POLITICA": str(tmp_path / "no-hay.json"),
           "JAX_EJECUTOR_FRENO_ESTADO": str(tmp_path / "estado.json")}
    cfg, remotos_ok = F.config_desde_entorno(env)
    assert cfg.remotos == () and remotos_ok is False and cfg.uid is None
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c4 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_freno.py -v`
Expected: FAIL (`cannot import name 'freno'`).

- [ ] **Step 3: Implementar**

```python
# jax/ejecutor/contratos/freno.py
"""Freno en vuelo de la cuenta del Ejecutor (C4). Corre como ROOT: ejecutor-freno.service.

Mientras el interruptor de JAX esté puesto (jax/core/interruptor.py, frente B), en
cada vuelta de INTERVALO_DE_SONDEO:
1. escribe "1" en cgroup.kill de user-<uid>.slice: todo lo que entró por ssh, incluido
   lo desprendido con setsid/nohup y lo que nazca mientras se mata (cgroup v2);
2. barre /proc y mata todo proceso con uid real o efectivo de la cuenta, esté donde
   esté (p. ej. fuera de su slice);
3. en los repasos REPASOS_S desde que se puso (0, 2 y 10 s) lanza —sin esperarlo— el
   barrido de cada máquina remota con la llave de comando forzado
   (ejecutor-freno-remoto). Un repaso tardío atrapa lo que arrancó justo al frenar.

FAIL-CLOSED: interruptor sin configurar = puesto. Política ilegible o cuenta
inexistente: se sigue matando lo que se pueda y se declara en el latido; el arranque
del Ejecutor (plan 6) exige latido fresco con remotos cargados.

Sólo biblioteca estándar (se instala en /opt/ejecutor/lib y corre con python3 -I).
"""
from __future__ import annotations

import json
import logging
import os
import pwd
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from jax.core import interruptor
from jax.ejecutor.contratos import politica

REPASOS_S = (0.0, 2.0, 10.0)
_TOPE_BARRIDO_S = 15
log = logging.getLogger("ejecutor.freno")


@dataclass(frozen=True)
class Remoto:
    nombre: str
    ip: str
    puerto: int


@dataclass(frozen=True)
class ConfigFreno:
    uid: int | None
    cuenta: str
    cgroup: Path
    proc: Path
    llave: Path
    known_hosts: Path
    remotos: tuple
    estado: Path


def ruta_cgroup_kill(cgroup: Path, uid: int) -> Path:
    return cgroup / "user.slice" / f"user-{uid}.slice" / "cgroup.kill"


def matar_cgroup(cgroup: Path, uid: int) -> bool:
    try:
        ruta_cgroup_kill(cgroup, uid).write_text("1")
    except FileNotFoundError:
        return False  # sin sesiones de la cuenta no existe su slice: no hay nada que matar ahí
    return True


def pids_de(proc: Path, uid: int) -> list[int]:
    pids = []
    for entrada in proc.iterdir():
        if not entrada.name.isdigit():
            continue
        try:
            for linea in (entrada / "status").read_text().splitlines():
                if linea.startswith("Uid:"):
                    real, efectivo = (int(x) for x in linea.split()[1:3])
                    if uid in (real, efectivo):
                        pids.append(int(entrada.name))
                    break
        except (FileNotFoundError, ProcessLookupError):
            continue  # el proceso terminó entre listar y leer
    return pids


def matar_pids(pids, matar=os.kill) -> int:
    muertos = 0
    for pid in pids:
        try:
            matar(pid, signal.SIGKILL)
            muertos += 1
        except ProcessLookupError:
            continue
    return muertos


def argv_barrido(r: Remoto, cfg: ConfigFreno) -> list[str]:
    return ["ssh", "-i", str(cfg.llave), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={cfg.known_hosts}",
            "-o", "ConnectTimeout=5", "-p", str(r.puerto), f"{cfg.cuenta}@{r.ip}"]


def barrer(r: Remoto, cfg: ConfigFreno, correr=subprocess.run) -> str:
    try:
        hecho = correr(argv_barrido(r, cfg), capture_output=True, timeout=_TOPE_BARRIDO_S, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return "tiempo_agotado"
    return "ok" if hecho.returncode == 0 and b"freno_remoto=ok quedan=0" in hecho.stdout else "fallo"


def _escribir_estado(ruta: Path, estado: dict) -> None:
    fd, temporal = tempfile.mkstemp(dir=ruta.parent, prefix=".estado-")
    with os.fdopen(fd, "w") as f:
        json.dump(estado, f)
    os.replace(temporal, ruta)


class Freno:
    def __init__(self, cfg: ConfigFreno, *, activo=None, reloj=time.monotonic, lanzar_barrido=None,
                 matar=os.kill, remotos_cargados: bool = True):
        self.cfg = cfg
        self._activo = activo or self._interruptor
        self._reloj = reloj
        self._matar = matar
        self._remotos_cargados = remotos_cargados
        self._pool = ThreadPoolExecutor(max_workers=max(1, len(cfg.remotos)))
        self._lanzar = lanzar_barrido or self._barrer_en_fondo
        self._desde = None
        self._repasos = set()

    @staticmethod
    def _interruptor() -> bool:
        try:
            return interruptor.interruptor_activo()
        except interruptor.InterruptorSinConfigurar:
            return True

    def _barrer_en_fondo(self, remotos) -> None:
        for r in remotos:
            futuro = self._pool.submit(barrer, r, self.cfg)
            futuro.add_done_callback(lambda f, n=r.nombre: log.warning("freno barrido=%s resultado=%s", n, f.result()))

    def paso(self) -> dict:
        activo = self._activo()
        salida = {"activo": activo}
        if not activo:
            self._desde, self._repasos = None, set()
        else:
            ahora = self._reloj()
            if self._desde is None:
                self._desde = ahora
            if self.cfg.uid is not None:
                salida["cgroup"] = matar_cgroup(self.cfg.cgroup, self.cfg.uid)
                salida["muertos"] = matar_pids(pids_de(self.cfg.proc, self.cfg.uid), self._matar)
            for i, repaso in enumerate(REPASOS_S):
                if i not in self._repasos and ahora - self._desde >= repaso:
                    self._repasos.add(i)
                    if self.cfg.remotos:
                        self._lanzar(self.cfg.remotos)
                    break
        _escribir_estado(self.cfg.estado, {"momento": time.time(), "activo": activo,
                                           "remotos_cargados": self._remotos_cargados,
                                           "uid_resuelto": self.cfg.uid is not None})
        return salida if activo else {"activo": False}


def config_desde_entorno(env) -> tuple[ConfigFreno, bool]:
    cuenta = env["JAX_EJECUTOR_CUENTA"]
    try:
        uid = pwd.getpwnam(cuenta).pw_uid
    except KeyError:
        uid = None
    remotos, cargados = (), True
    try:
        p = politica.validar(json.loads(Path(env["JAX_EJECUTOR_POLITICA"]).read_bytes()))
        remotos = tuple(Remoto(h.nombre, h.ip, h.puerto) for h in p.hosts if not h.es_local)
    except (OSError, ValueError):  # fail-soft: sin política se sigue matando local; el latido dice remotos_cargados=false y el arranque se niega
        cargados = False
    cfg = ConfigFreno(uid=uid, cuenta=cuenta, cgroup=Path("/sys/fs/cgroup"), proc=Path("/proc"),
                      llave=Path(env["JAX_EJECUTOR_FRENO_LLAVE"]), known_hosts=Path(env["JAX_EJECUTOR_FRENO_KNOWN_HOSTS"]),
                      remotos=remotos, estado=Path(env["JAX_EJECUTOR_FRENO_ESTADO"]))
    return cfg, cargados


def principal() -> int:
    logging.basicConfig(level=logging.INFO)
    cfg, cargados = config_desde_entorno(os.environ)
    freno = Freno(cfg, remotos_cargados=cargados)
    while True:
        estado = freno.paso()
        if estado["activo"]:
            log.warning("freno activo cgroup=%s muertos=%s", estado.get("cgroup"), estado.get("muertos"))
        time.sleep(interruptor.INTERVALO_DE_SONDEO)


if __name__ == "__main__":
    sys.exit(principal())
```

Nota: `politica.PoliticaIlegible` hereda de `ValueError`: queda en el `except (OSError, ValueError)`.
`test_paso_puesto_mata_local_cada_vuelta_y_barre_en_sus_repasos` espera un solo repaso por vuelta (el `break`): en
`delta=10.5` se lanza el repaso de 10 s aunque el de 2 s ya se hizo en `2.1`.

- [ ] **Step 4: Correr y ver que pasan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c4 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_freno.py -v && python3 policy/tests/test_no_fail_open_except.py && python3 policy/tests/test_claude_subprocess_solo_via_sandbox.py`
Expected: 14 PASS; guardias verdes.

- [ ] **Step 5: Mutaciones (taxonomía)**

Con backup y `cmp`, una por vez:
1. **Suprimir** el barrido de `/proc`: `salida["muertos"] = matar_pids(...)` → `salida["muertos"] = 0`.
   Expected: FAIL `test_paso_puesto_mata_local_cada_vuelta_y_barre_en_sus_repasos`.
2. **Neutralizar el fail-closed:** en `_interruptor`, `return True` → `return False`.
   Expected: FAIL `test_interruptor_sin_configurar_cuenta_como_puesto`.
3. **Barrido que da por bueno cualquier salida:** en `barrer`, `and b"freno_remoto=ok quedan=0" in hecho.stdout` →
   eliminarlo. Expected: FAIL `test_barrer_exige_cero_procesos[0-freno_remoto=ok quedan=2\n-fallo]`.

- [ ] **Step 6: Commit**

```bash
git -C /home/fruiz/worktrees/jax-sp1-c4 add jax/ejecutor/contratos/freno.py tests/test_ejecutor_contratos_freno.py
git -C /home/fruiz/worktrees/jax-sp1-c4 commit -m "feat(ejecutor): freno root de C4 — cgroup.kill, barrido por uid, repasos remotos y latido

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: el comando forzado remoto

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-c4/ops/ejecutor/ejecutor-freno-remoto`
- Test: `/home/fruiz/worktrees/jax-sp1-c4/tests/test_ejecutor_freno_remoto.py`

**Interfaces:** Produce el guion que el plan 5 instala en `/usr/local/sbin/ejecutor-freno-remoto` de cada máquina y
ata a la llave del freno con `command="…",restrict`. Salida: exactamente `freno_remoto=ok quedan=<n>`.

**Por qué salva su propia sesión:** corre como la cuenta; un `kill -KILL -1` o `pkill -u` se mataría a sí mismo antes
de poder decir cuántos quedaron, y el freno no sabría si funcionó.

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_ejecutor_freno_remoto.py
"""Comando forzado del freno (C4): mata lo de la cuenta salvo su sesión y dice
cuántos quedan. Se prueba con un `ps` FALSO que sólo lista procesos creados por
el test: correrlo de verdad como el usuario del test mataría su sesión entera."""
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
GUION = RAIZ / "ops" / "ejecutor" / "ejecutor-freno-remoto"

_PS_FALSO = """#!{python}
import os, sys
if "-p" in sys.argv:
    print(" 1")
    sys.exit(0)
for linea in open({lista!r}):
    pid, sid = linea.split()
    try:
        estado = open(f"/proc/{{pid}}/stat").read().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        continue
    if estado != "Z":
        print(f" {{pid}} {{sid}}")
"""


def _entorno(tmp_path, victimas, de_su_sesion=()):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    lista = tmp_path / "pids"
    lista.write_text("".join(f"{p.pid} 999\n" for p in victimas) + "".join(f"{p.pid} 1\n" for p in de_su_sesion))
    (bin_ / "ps").write_text(_PS_FALSO.format(python=sys.executable, lista=str(lista)))
    (bin_ / "id").write_text("#!/bin/sh\necho cuenta-de-prueba\n")
    for f in ("ps", "id"):
        (bin_ / f).chmod(0o755)
    return {"PATH": f"{bin_}:/usr/bin:/bin"}


def _vivo(p):
    try:
        estado = Path(f"/proc/{p.pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return estado != "Z"


def test_mata_lo_de_otras_sesiones_y_dice_cero(tmp_path):
    victimas = [subprocess.Popen(["sleep", "60"], start_new_session=True) for _ in range(2)]
    try:
        r = subprocess.run([str(GUION)], env=_entorno(tmp_path, victimas), capture_output=True, timeout=20)
        assert r.stdout == b"freno_remoto=ok quedan=0\n", r
        assert [_vivo(p) for p in victimas] == [False, False]
    finally:
        for p in victimas:
            p.kill()
            p.wait(5)


def test_no_mata_su_propia_sesion(tmp_path):
    propia = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        r = subprocess.run([str(GUION)], env=_entorno(tmp_path, [], de_su_sesion=[propia]), capture_output=True,
                           timeout=20)
        assert r.stdout == b"freno_remoto=ok quedan=0\n"
        assert _vivo(propia) is True
    finally:
        propia.kill()
        propia.wait(5)


def test_el_guion_es_ejecutable_y_sh():
    assert os.access(GUION, os.X_OK)
    assert GUION.read_text().startswith("#!/bin/sh\n")
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c4 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_freno_remoto.py -v`
Expected: FAIL (el guion no existe).

- [ ] **Step 3: Implementar**

```sh
#!/bin/sh
# ops/ejecutor/ejecutor-freno-remoto — comando forzado de la llave del freno (C4).
# Se instala root 0755 en /usr/local/sbin de cada máquina del inventario (plan 5) y la
# llave del freno lo ata con command="/usr/local/sbin/ejecutor-freno-remoto",restrict.
# Corre COMO la cuenta del Ejecutor: mata todos sus procesos salvo esta misma sesión
# (si se matara a sí mismo, el freno no sabría cuántos quedaron) y lo dice.
# Límite declarado: un proceso lanzado con sudo corre como root y no es de la cuenta.
# Hoy la cuenta no tiene sudo en ninguna máquina; antes de dárselo (Fase 3) se amplía esto.
set -u
yo=$(ps -o sid= -p $$ | tr -d ' ')
cuenta=$(id -un)
for pid in $(ps -u "$cuenta" -o pid=,sid= | awk -v s="$yo" '$2 != s {print $1}'); do
  kill -KILL "$pid" 2>/dev/null
done
sleep 0.2
quedan=$(ps -u "$cuenta" -o pid=,sid= | awk -v s="$yo" '$2 != s' | wc -l | tr -d ' ')
echo "freno_remoto=ok quedan=$quedan"
```

`git -C /home/fruiz/worktrees/jax-sp1-c4 update-index --chmod=+x ops/ejecutor/ejecutor-freno-remoto` además de `chmod +x`.

- [ ] **Step 4: Correr y ver que pasan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c4 && chmod +x ops/ejecutor/ejecutor-freno-remoto && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_freno_remoto.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Mutación**

Con backup y `cmp`: `awk -v s="$yo" '$2 != s {print $1}'` → `awk '{print $1}'` (en el loop de `kill`).
Expected: FAIL `test_no_mata_su_propia_sesion`.

- [ ] **Step 6: Commit**

```bash
git -C /home/fruiz/worktrees/jax-sp1-c4 add ops/ejecutor/ejecutor-freno-remoto tests/test_ejecutor_freno_remoto.py
git -C /home/fruiz/worktrees/jax-sp1-c4 commit -m "feat(ejecutor): comando forzado del freno remoto, salva su sesión y cuenta lo que queda

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: instalación del freno en hall9000

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-c4/ops/ejecutor/ejecutor-freno.service.plantilla`
- Modify: `/home/fruiz/worktrees/jax-sp1-c4/jax/ejecutor/contratos/instalacion.py` (`INSTALABLES`, `renderizar_unidad_freno`, `principal` escribe `ejecutor-freno.service`)
- Modify: `/home/fruiz/worktrees/jax-sp1-c4/tests/test_ejecutor_contratos_instalacion.py`
- Create: `/home/fruiz/worktrees/jax-sp1-c4/ops/ejecutor/instalar_freno.sh`

- [ ] **Step 1: Tests que fallan**

En `tests/test_ejecutor_contratos_instalacion.py`, `test_lo_instalable_es_solo_biblioteca_estandar` completo (acepta
`jax.core.*` y `from jax.core import …` sólo si lo importado también se instala):

```python
_PAQUETES_PROPIOS = {"jax.ejecutor.contratos": "jax/ejecutor/contratos", "jax.core": "jax/core"}


@pytest.mark.parametrize("rel", instalacion.INSTALABLES)
def test_lo_instalable_es_solo_biblioteca_estandar(rel):
    arbol = ast.parse((RAIZ / rel).read_text(encoding="utf-8"))
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.ImportFrom) and nodo.module in _PAQUETES_PROPIOS:
            for alias in nodo.names:
                assert f"{_PAQUETES_PROPIOS[nodo.module]}/{alias.name}.py" in instalacion.INSTALABLES, (rel, alias.name)
            continue
        if isinstance(nodo, ast.Import):
            modulos = [a.name for a in nodo.names]
        elif isinstance(nodo, ast.ImportFrom):
            modulos = [nodo.module or ""]
        else:
            continue
        for m in modulos:
            paquete, _, hoja = m.rpartition(".")
            if paquete in _PAQUETES_PROPIOS:
                assert f"{_PAQUETES_PROPIOS[paquete]}/{hoja}.py" in instalacion.INSTALABLES, (rel, m)
            else:
                assert m == "__future__" or m.split(".")[0] in sys.stdlib_module_names, (rel, m)
```

Tests nuevos:

```python
def test_el_freno_y_el_interruptor_se_instalan():
    for rel in ("jax/core/__init__.py", "jax/core/interruptor.py", "jax/ejecutor/contratos/freno.py"):
        assert rel in instalacion.INSTALABLES


def test_render_de_la_unidad_del_freno():
    texto = instalacion.renderizar_unidad_freno("/opt/ejecutor/lib")
    assert "User=root" in texto and "Restart=always" in texto and "EnvironmentFile=/etc/jax/.env" in texto
    assert "sys.path.insert(0, '/opt/ejecutor/lib')" in texto and "/usr/bin/python3 -I" in texto
    assert "@LIB@" not in texto
```

- [ ] **Step 2: Correr y ver que fallan**

Run: `cd /home/fruiz/worktrees/jax-sp1-c4 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_instalacion.py -v`
Expected: FAIL en los dos nuevos.

- [ ] **Step 3: Implementar**

```ini
# ops/ejecutor/ejecutor-freno.service.plantilla
[Unit]
Description=Ejecutor · freno en vuelo (C4): mata la cuenta del Ejecutor mientras el interruptor esté puesto
After=local-fs.target

[Service]
Type=simple
User=root
EnvironmentFile=/etc/jax/.env
RuntimeDirectory=ejecutor-freno
RuntimeDirectoryMode=0755
ExecStart=/usr/bin/python3 -I -c "import sys; sys.path.insert(0, '@LIB@'); from jax.ejecutor.contratos.freno import principal; sys.exit(principal())"
Restart=always
RestartSec=1

[Install]
WantedBy=multi-user.target
```

En `instalacion.py`:

```python
PLANTILLA_FRENO = RAIZ / "ops" / "ejecutor" / "ejecutor-freno.service.plantilla"

INSTALABLES = (
    "jax/__init__.py",
    "jax/core/__init__.py",
    "jax/core/interruptor.py",
    "jax/ejecutor/__init__.py",
    "jax/ejecutor/contratos/__init__.py",
    "jax/ejecutor/contratos/formato.py",
    "jax/ejecutor/contratos/destinos.py",
    "jax/ejecutor/contratos/politica.py",
    "jax/ejecutor/contratos/gancho.py",
    "jax/ejecutor/contratos/freno.py",
)


def renderizar_unidad_freno(lib: str) -> str:
    return PLANTILLA_FRENO.read_text(encoding="utf-8").replace("@LIB@", _ruta(lib))
```

y en `principal`, dentro de `renderizados`: `"ejecutor-freno.service": renderizar_unidad_freno(lib).encode(),`.

```bash
#!/usr/bin/env bash
# ops/ejecutor/instalar_freno.sh — C4 en hall9000. Corre como fruiz; sudo para lo de root.
set -euo pipefail
: "${JAX_EJECUTOR_LIB:?}" "${JAX_EJECUTOR_CUENTA:?}" "${JAX_EJECUTOR_FRENO_LLAVE:?}"
: "${JAX_EJECUTOR_FRENO_KNOWN_HOSTS:?}" "${JAX_EJECUTOR_FRENO_ESTADO:?}" "${JAX_KILL_SWITCH_PATH:?}"
REPO="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
test "$(git -C "$REPO" branch --show-current)" = master

# La biblioteca (gancho, freno, interruptor) la instala el instalador del plan 1, con INSTALABLES ya ampliado.
"$REPO/ops/ejecutor/instalar_contratos.sh"
ETAPA="$(mktemp -d)"; trap 'rm -rf "$ETAPA"' EXIT
( cd "$REPO" && PYTHONDONTWRITEBYTECODE=1 python3 -m jax.ejecutor.contratos.instalacion "$ETAPA" )

# Llave del freno: root, nunca en la cuenta. La pública se instala en cada máquina en el plan 5.
sudo install -d -o root -g root -m 0700 "$(dirname "$JAX_EJECUTOR_FRENO_LLAVE")"
sudo test -e "$JAX_EJECUTOR_FRENO_LLAVE" || sudo ssh-keygen -q -t ed25519 -N '' -C ejecutor-freno -f "$JAX_EJECUTOR_FRENO_LLAVE"
sudo touch "$JAX_EJECUTOR_FRENO_KNOWN_HOSTS"

# Lo que sacaría procesos de la cuenta de su cgroup: cron, at y linger.
for f in /etc/cron.deny /etc/at.deny; do
  sudo touch "$f"
  sudo grep -qx "$JAX_EJECUTOR_CUENTA" "$f" || echo "$JAX_EJECUTOR_CUENTA" | sudo tee -a "$f" >/dev/null
done
test ! -e /etc/cron.allow || { echo "cron.allow existe: cron.deny no rige" >&2; exit 1; }
sudo loginctl disable-linger "$JAX_EJECUTOR_CUENTA"

sudo install -o root -g root -m 0644 "$ETAPA/ejecutor-freno.service" /etc/systemd/system/ejecutor-freno.service
sudo systemctl daemon-reload
sudo systemctl enable --now ejecutor-freno.service
sleep 2
python3 - "$JAX_EJECUTOR_FRENO_ESTADO" <<'PY'
import json, sys, time
e = json.load(open(sys.argv[1]))
assert time.time() - e["momento"] < 2 and e["uid_resuelto"] is True, e
print("freno_latido=ok remotos_cargados=" + str(e["remotos_cargados"]).lower())
PY
```

- [ ] **Step 4: Correr tests y commit**

Run: `cd /home/fruiz/worktrees/jax-sp1-c4 && PYTHONPATH=.:las_manos python -m pytest tests/test_ejecutor_contratos_instalacion.py -v`
Expected: PASS (incluido el de biblioteca estándar sobre `jax/core/interruptor.py`).

```bash
git -C /home/fruiz/worktrees/jax-sp1-c4 add ops/ejecutor/ejecutor-freno.service.plantilla ops/ejecutor/instalar_freno.sh jax/ejecutor/contratos/instalacion.py tests/test_ejecutor_contratos_instalacion.py
git -C /home/fruiz/worktrees/jax-sp1-c4 commit -m "ops(ejecutor): unidad root del freno, llave propia y cron/at/linger cerrados para la cuenta

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: CI — listas, piso y canario rojo

**Files:** Modify `/home/fruiz/worktrees/jax-sp1-c4/.github/workflows/policy.yml`

- [ ] **Step 1:** Agregar en **las dos** listas de `tests-puros`: `tests/test_ejecutor_proxy_freno.py`,
  `tests/test_ejecutor_contratos_freno.py`, `tests/test_ejecutor_freno_remoto.py`. Verificar igualdad de listas
  (script del plan 1, Task 8 Step 2).
- [ ] **Step 2:** Push; piso con el número del runner y línea
  `# N -> M el 2026-09-17 (Ejecutor SP1 plan 3): C4 — proxy frena antes y en vuelo, freno root (cgroup, /proc, repasos remotos, latido), comando forzado remoto. CONFIRMADO POR EL RUNNER (3.12, PR #<número del PR>).`
  **Comprobar en el log del runner que `test_ejecutor_freno_remoto.py` CORRE** (no `skipped`): usa `/proc`, `sleep` y
  `awk`, que el runner tiene; si se salteara, el piso lo delata.
- [ ] **Step 3:** Canario rojo por API: commit con la mutación 1 de la Task 2 → `tests-puros` = `failure` sobre ese
  sha; revert → `success`.
- [ ] **Step 4:** Commit `ci(ejecutor): C4 en tests-puros con piso del runner` con el trailer.

---

### Task 6: hall9000 · simulacro real — control, dos rondas, dos lecturas por ronda; y VERLO FALLAR

> Pone el freno GLOBAL ~10 s por ronda. Sólo con la Mesa sin uso real en los 5 minutos previos. Requiere planes 1–2
> instalados, el frente B desplegado (`JAX_KILL_SWITCH_PATH` en `/etc/jax/.env`, directorio `/etc/jax/interruptor`
> `2770 root:fruiz`) y la Task 4 instalada.

**Files:**
- Create: `/home/fruiz/worktrees/jax-sp1-c4/scripts/ejecutor_contratos/probar_c4.py`

**Qué se prueba y por qué así:** dos procesos de la cuenta que, si viven 5 s, crean una marca: uno desprendido
(`setsid nohup`) y uno al otro lado de un `ssh -tt` a la propia cuenta (127.0.0.1 está en el inventario y en el
cerco). **Primero una ronda de control sin freno**: las dos marcas TIENEN que aparecer — si no, la prueba con freno
«pasaría» porque el escenario no corría. Después, con freno: a +1 s y a +6 s, ningún proceso de la cuenta
(`ps -u`, desde `fruiz`), y ninguna marca después de soltar. Dos rondas con 10 s de por medio. El `ssh -tt` remoto
real contra otra máquina se prueba en el plan 5 Parte B.

- [ ] **Step 1: El script**

```python
#!/usr/bin/env python3
# scripts/ejecutor_contratos/probar_c4.py
"""Simulacro real de C4 en hall9000, re-ejecutable por un tercero.

Uso (con la Mesa sin uso real):
  set -a; . /etc/jax/.env; set +a
  PYTHONPATH=.:las_manos python3 scripts/ejecutor_contratos/probar_c4.py
Pone el interruptor GLOBAL de JAX durante ~8 s por ronda y lo quita. Lee /etc/jax/.env (producción) para
JAX_EJECUTOR_* y JAX_KILL_SWITCH_PATH; no toca la DB.
"""
import asyncio
import json
import os
import secrets
import subprocess
import sys

from jax.core import interruptor
from jax.ejecutor.contratos import cuenta_axioma, formato

ESPERA_CONTROL_S = 7
LECTURAS_S = (1, 6)
ENTRE_RONDAS_S = 10


def _procesos(cuenta: str) -> list[str]:
    r = subprocess.run(["ps", "-u", cuenta, "-o", "pid="], capture_output=True, text=True)
    return r.stdout.split()


def _escenario(c, nonce: str) -> str:
    return (f'setsid nohup sh -c \'sleep 5 && touch "$HOME/.c4-desprendido-{nonce}"\' >/dev/null 2>&1 </dev/null & '
            f'ssh -tt -o BatchMode=yes -p {c.puerto} {c.nombre}@127.0.0.1 \'sleep 5 && touch "$HOME/.c4-anidado-{nonce}"\'')


async def _marcas(c, nonce: str) -> list[str]:
    rc, salida, _ = await cuenta_axioma.correr_en_la_cuenta(
        c, f'ls -1 "$HOME"/.c4-*-{nonce} 2>/dev/null; rm -f "$HOME"/.c4-*-{nonce}', tope_s=30)
    return [l.rsplit("/", 1)[1] for l in salida.decode().split()]


async def ronda(c, fallos: list, *, con_freno: bool) -> None:
    nonce = secrets.token_hex(6)
    lanzado = asyncio.create_task(cuenta_axioma.correr_en_la_cuenta(c, _escenario(c, nonce), tope_s=60))
    await asyncio.sleep(1)
    if not con_freno:
        await asyncio.sleep(ESPERA_CONTROL_S)
        await asyncio.wait_for(lanzado, 60)
        marcas = await _marcas(c, nonce)
        if len(marcas) != 2:
            fallos.append(("control_fallido", (("marcas", marcas),)))
        return
    ruta = interruptor.ruta_del_interruptor()
    if not interruptor.escribir_pausa(ruta, json.dumps({"origen": "probar_c4", "nonce": nonce})):
        fallos.append(("interruptor_ya_puesto", ()))
        return
    try:
        transcurrido = 0
        for lectura in LECTURAS_S:
            await asyncio.sleep(lectura - transcurrido)
            transcurrido = lectura
            vivos = _procesos(c.nombre)
            if vivos:
                fallos.append(("procesos_vivos", (("a_los_s", lectura), ("pids", vivos))))
        await asyncio.wait_for(lanzado, 10)
    finally:
        interruptor.borrar_pausa(ruta)
    await asyncio.sleep(2)
    marcas = await _marcas(c, nonce)
    if marcas:
        fallos.append(("marca_creada", (("marcas", marcas),)))


async def principal() -> int:
    c = cuenta_axioma.cuenta_desde_entorno()
    fallos: list = []
    activo = subprocess.run(["systemctl", "is-active", "ejecutor-freno.service"], capture_output=True, text=True)
    if activo.stdout.strip() != "active":
        fallos.append(("freno_inactivo", ()))
    if interruptor.interruptor_activo() or _procesos(c.nombre):
        print(formato.campos((("c4_vivo", False), ("codigo", "estado_inicial_no_limpio"))))
        return 2
    await ronda(c, fallos, con_freno=False)
    if not any(f[0] == "control_fallido" for f in fallos):
        for i in range(2):
            if i:
                await asyncio.sleep(ENTRE_RONDAS_S)
            await ronda(c, fallos, con_freno=True)
    for codigo, datos in fallos:
        print(formato.campos((("contrato", "c4"), ("codigo", codigo)) + tuple(datos)))
    print(formato.campos((("c4_vivo", not fallos),)))
    return 0 if not fallos else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(principal()))
```

Nota: el script **no** está en `_AISLADO_POR_CUENTA` y no lo necesita: no menciona el arnés en ningún literal. Si la
guardia lo marcara, se revisan sus literales, no se agrega excepción.

- [ ] **Step 2: Verlo vivo**

Mesa sin uso (journal `jax-platform` sin `/api/chat` en 5 min). Run:
`cd /home/fruiz/jax && set -a && . /etc/jax/.env && set +a && PYTHONPATH=.:las_manos python3 scripts/ejecutor_contratos/probar_c4.py`
Expected: `c4_vivo=true`, exit 0. Si sale `control_fallido`: el escenario no corre (p. ej. la llave de la cuenta
no está autorizada para `axioma@127.0.0.1`): se arregla eso **antes** de mirar el freno; nunca se da por vivo un
C4 con el control fallido.

- [ ] **Step 3: VERLO FALLAR**

`sudo systemctl stop ejecutor-freno.service` → correr el script → Expected: `freno_inactivo`,
`procesos_vivos a_los_s=1`, `procesos_vivos a_los_s=6` (en las dos rondas) y `marca_creada` con las dos marcas;
exit 1. Esto prueba que la detección funciona y que **sin el freno root nada mata a la cuenta** (el watcher de LAS
MANOS no puede). `sudo systemctl start ejecutor-freno.service` → correr otra vez → `c4_vivo=true`.

- [ ] **Step 4: Latencia del corte (medición)**

Una ronda extra con lecturas a +0,3 s y +0,6 s (constante `LECTURAS_S` cambiada localmente, sin commitear) y
registrar en qué lectura la cuenta ya está vacía. **Criterio pre-registrado:** vacía a +0,6 s (dos vueltas de 250 ms
más margen). Si no, se mide cuánto tarda `cgroup.kill` antes de tocar nada.

- [ ] **Step 5: Commit y Biblioteca**

```bash
git -C /home/fruiz/worktrees/jax-sp1-c4 add scripts/ejecutor_contratos/probar_c4.py
git -C /home/fruiz/worktrees/jax-sp1-c4 commit -m "ops(ejecutor): simulacro real de C4 con ronda de control y dos lecturas por ronda

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

`CONTEXT.md` §9: la salida literal de los Steps 2 y 3, el número del Step 4, cuántas veces se puso el freno global y
por cuánto, y la lección: **«un freno que vive dentro del proceso que frena, o en una cuenta que no puede matar al
frenado, no es freno»**.

---

## Auto-revisión (hecha al escribir)

- **Spec §4 C4:** scope de cgroup propio ✓ (`user-<uid>.slice` de la cuenta dedicada, Task 2); `PAUSE` mata la jaula
  entera ✓ (Tasks 2 y 6) y lo remoto ✓ (`ssh -tt` → SIGHUP y barrido con comando forzado; prueba real remota en el
  plan 5 Parte B); comando remoto largo + PAUSE → muerto, **verificado dos veces con segundos de por medio** ✓
  (dos lecturas por ronda, +1 y +6 s, y dos rondas con 10 s entre ellas); el proxy también ✓ (Task 1).
- **Frente B:** sólo se consumen sus funciones; ningún archivo suyo se edita; el plan no arranca sin su merge.
- **Placeholders:** ninguno. **Tipos:** `ConfigFreno` (con `estado`), `Freno.paso() -> dict`,
  `config_desde_entorno(env) -> (ConfigFreno, bool)` iguales en Interfaces, código y tests.
