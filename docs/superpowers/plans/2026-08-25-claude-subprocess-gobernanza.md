# Gobernanza de Sub-agentes Claude Code Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Los 2 call sites reales que lanzan `claude` como subproceso (`jacobs/executor.py::_invoke_hyde` y `jax/muscles/subprocess_muscle.py::SubprocessMuscle._call`) ya están sandboxeados con bwrap, pero cada uno reimplementa el lanzamiento por separado (uno con semáforo, el otro sin ninguno). Centralizar el lanzamiento en un único punto de entrada aprobado (`hyde_sandbox.py::run_sandboxed_claude`) y agregar un guard de CI que falle si aparece un futuro call site de `claude` fuera de ese punto — cerrando el gap prospectivo que `DEUDA.md` describe ("cualquier OTRO músculo/automatización que dispare `claude`").

**Architecture:** `hyde_sandbox.py` gana una función async (`run_sandboxed_claude`) que posee el ciclo de vida completo del subproceso (armar el bwrap, adquirir un lock cross-proceso vía `flock(2)` sobre un archivo dentro de `workspace_dir`, lanzar, comunicar con timeout, matar y cosechar en timeout/cancelación) — devuelve el `Process` ya esperado más stdout/stderr crudos, dejando la interpretación de exit-code/stderr a cada llamador (que hoy difiere: `RuntimeError` en Jacobs, `MuscleInvocationError`/`MuscleTimeoutError` en el REPL — este plan preserva esa diferencia, no la unifica).

**Corrección aplicada en ejecución (2026-08-25), antes de dispatchear Task 1:** el diseño original de este plan usaba `asyncio.Semaphore(1)` para el lock "compartido". Verificado por enumeración real de imports (no supuesto) que `jacobs/executor.py` corre DENTRO del proceso de `las_manos` (systemd `jax-las-manos`, uvicorn `server:app`) mientras que `SubprocessMuscle` tiene un único importador real, `jax/core/main.py` — el REPL, un proceso de SO SEPARADO. Un `asyncio.Semaphore` de módulo vive en la memoria de un único intérprete y no cruza esa frontera entre procesos: el diseño original no cerraba el gap de concurrencia que el propio plan describe ("dos Hydes concurrentes via callers distintos no se coordinaban entre si"). Se reemplazó por un `flock(2)` sobre `workspace_dir/.claude_subprocess.lock` — visible por cualquier proceso que abra el mismo path, ya que `workspace_dir` es `JAX_WORKSPACE_DIR` resuelto por cada llamador desde la misma fuente única (`/etc/jax/.env`) — corrido en `asyncio.to_thread` (flock es bloqueante, nunca en el event loop) con timeout explícito y fail-closed (`TimeoutError` ruidoso si no se puede adquirir, nunca se sigue sin lock en silencio).

Un scanner AST nuevo (mismo patrón que `policy/tests/test_no_fail_open_except.py`) falla el CI si algún archivo fuera de `hyde_sandbox.py` llama `asyncio.create_subprocess_exec`/`create_subprocess_shell` Y menciona la palabra "claude" — verificado contra el código real de ambos repos: cero falsos positivos hoy (ningún otro call site de subprocess async menciona "claude").

**Tech Stack:** Python 3.12, `asyncio`, `ast` (scanner estático), `unittest.IsolatedAsyncioTestCase` + `unittest.mock.patch`.

**Spec:** `DEUDA.md` (raíz de `/home/fruiz/jax`), sección "Bloquea trabajo", ítem "Sub-agentes de Claude Code sin gobernanza real".

## Global Constraints

- **Preservar exactamente** el contrato de excepciones de cada llamador existente — `_invoke_hyde` debe seguir dejando propagar `asyncio.CancelledError` SIN envolver (Jacobs cancela `_dispatch_step` desde afuera con su propio `wait_for`; envolver la cancelación en otro tipo de excepción rompe la propagación real de cancelación de asyncio). `SubprocessMuscle._call` debe seguir lanzando `MuscleTimeoutError`/`MuscleInvocationError` (contrato de la interfaz `Muscle`).
- No se cambia el comportamiento observable de Hyde (mismo bwrap, mismo timeout, mismo criterio de error) — solo se centraliza dónde vive el código.
- Verificado en el código real (2026-08-25): ningún otro archivo de `jax` o `jax-platform` que llama `asyncio.create_subprocess_exec`/`create_subprocess_shell` (`las_manos/workers/ssh_worker.py`, `jax/voice/tts.py`, `jax/voice/ears.py`, `jax-platform/backend/api/command.py`) menciona la palabra "claude" en su código fuente — el scanner de Task 3 no tiene falsos positivos contra el estado actual.

---

### Task 1: `run_sandboxed_claude()` — punto de entrada único, con test unitario

**Files:**
- Modify: `hyde_sandbox.py` (agregar `import asyncio`, `CLAUDE_SUBPROCESS_SEMAPHORE`, `run_sandboxed_claude`)
- Create: `_hyde_sandbox_test.py` (raíz de `/home/fruiz/jax`, mismo patrón de ubicación que el propio `hyde_sandbox.py`)

**Interfaces:**
- Produce: `async def run_sandboxed_claude(cmd: list[str], workspace_dir: str, prompt: str, timeout: float) -> tuple[asyncio.subprocess.Process, bytes, bytes]` — devuelve `(proc, stdout_bytes, stderr_bytes)`. En timeout (de la corrida real O de la espera del lock cross-proceso) / cancelación: mata el proceso si llegó a lanzarse, cosecha con `wait()`, y re-lanza la excepción SIN envolver (`raise` desnudo) — `asyncio.TimeoutError` es alias de `TimeoutError` desde Python 3.11, así que el mismo `except` de cada llamador distingue ambos casos sin cambios. Fail-closed sin atrapar: `SandboxUnavailable` (de `wrap_hyde_command`) sube tal cual.
- Produce (internas, prefijo `_`): `_acquire_cross_process_lock(workspace_dir: str, timeout: float)` / `_release_cross_process_lock(fh)` — `flock(2)` real sobre `workspace_dir/.claude_subprocess.lock`, BLOQUEANTES, se llaman solo vía `asyncio.to_thread`.

- [ ] **Step 1: Escribir los tests (fallan hoy — nada de esto existe todavía)**

```python
#!/usr/bin/env python3
"""hyde_sandbox.run_sandboxed_claude() -- unico punto de entrada aprobado
para lanzar `claude` como subproceso sandboxeado. Antes de este modulo,
jacobs/executor.py y jax/muscles/subprocess_muscle.py reimplementaban el
lanzamiento cada uno por su lado -- uno con HYDE_SEMAPHORE (un
asyncio.Semaphore, valido solo DENTRO de un proceso), el otro SIN NINGUN
lock. Jacobs corre dentro del proceso de las_manos (systemd
jax-las-manos); SubprocessMuscle solo lo importa jax/core/main.py, el
REPL -- un proceso de SO SEPARADO (confirmado por enumeracion real de
imports, 2026-08-25). Un asyncio.Semaphore de modulo no cruza esa
frontera -- se usa flock(2) sobre un archivo dentro de workspace_dir en
su lugar, visible por cualquier proceso que abra el mismo path.

La mayoria de estos tests mockean asyncio.create_subprocess_exec y
wrap_hyde_command -- no requieren bwrap real. ClaudeSubprocessLockFailClosedTest
y ClaudeSubprocessLockRealCrossProcessTest usan flock(2) DE VERDAD (sin
mock): dos corrutinas del mismo proceso pasarian igual con el
asyncio.Semaphore viejo, asi que no prueban nada sobre el problema real --
ClaudeSubprocessLockRealCrossProcessTest lanza dos procesos de Python DE
VERDAD (subprocess.Popen, no dos tasks de asyncio) para confirmar que el
lock serializa entre procesos de SO distintos.

Corre con:
  cd /home/fruiz/jax && .venv/bin/python -m pytest _hyde_sandbox_test.py -v

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import hyde_sandbox


class _FakeProc:
    def __init__(self, communicate_coro, returncode: int = 0):
        self._communicate_coro = communicate_coro
        self.returncode = returncode
        self.killed = False
        self.waited = False

    async def communicate(self, input=None):
        return await self._communicate_coro()

    def kill(self):
        self.killed = True

    async def wait(self):
        self.waited = True
        return self.returncode


class RunSandboxedClaudeWrappingTest(unittest.IsolatedAsyncioTestCase):
    async def test_aplica_wrap_hyde_command_antes_de_lanzar(self):
        captured_argv = {}

        async def fake_communicate():
            return b"hola", b""

        async def fake_create_subprocess_exec(*argv, **kwargs):
            captured_argv["argv"] = argv
            return _FakeProc(fake_communicate)

        with tempfile.TemporaryDirectory() as ws:
            with patch.object(hyde_sandbox, "wrap_hyde_command", return_value=["BWRAP_MARKER", "claude"]) as fake_wrap, \
                 patch("asyncio.create_subprocess_exec", fake_create_subprocess_exec):
                proc, stdout, stderr = await hyde_sandbox.run_sandboxed_claude(
                    ["claude", "--print"], ws, "prompt", timeout=5,
                )

            fake_wrap.assert_called_once_with(["claude", "--print"], ws)
            self.assertEqual(captured_argv["argv"], ("BWRAP_MARKER", "claude"))
            self.assertEqual(stdout, b"hola")
            self.assertEqual(proc.returncode, 0)


class RunSandboxedClaudeConcurrencyTest(unittest.IsolatedAsyncioTestCase):
    async def test_serializa_dos_invocaciones_concurrentes_mismo_proceso(self):
        # Prueba de humo del flujo mockeado -- NO es la prueba de la
        # propiedad cross-proceso real (ver ClaudeSubprocessLockRealCrossProcessTest).
        events = []

        def make_communicate(tag):
            async def _communicate():
                events.append(("start", tag, time.monotonic()))
                await asyncio.sleep(0.05)
                events.append(("end", tag, time.monotonic()))
                return b"out", b""
            return _communicate

        counter = iter([1, 2])

        async def fake_create_subprocess_exec(*argv, **kwargs):
            tag = next(counter)
            return _FakeProc(make_communicate(tag))

        with tempfile.TemporaryDirectory() as ws:
            with patch.object(hyde_sandbox, "wrap_hyde_command", side_effect=lambda cmd, w: cmd), \
                 patch("asyncio.create_subprocess_exec", fake_create_subprocess_exec):
                await asyncio.gather(
                    hyde_sandbox.run_sandboxed_claude(["claude"], ws, "p1", timeout=5),
                    hyde_sandbox.run_sandboxed_claude(["claude"], ws, "p2", timeout=5),
                )

        # Serializado de verdad: el segundo "start" debe ocurrir DESPUES del
        # primer "end" -- si corrieran en paralelo, ambos "start" saldrian
        # antes que cualquier "end".
        starts = [e for e in events if e[0] == "start"]
        ends = [e for e in events if e[0] == "end"]
        self.assertEqual(len(starts), 2)
        self.assertEqual(len(ends), 2)
        self.assertLess(ends[0][2], starts[1][2], f"no se serializó: {events}")


class RunSandboxedClaudeTimeoutTest(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_mata_proceso_y_propaga_timeouterror_sin_envolver(self):
        async def hangs_forever():
            await asyncio.sleep(999)
            return b"", b""

        fake_proc = _FakeProc(hangs_forever)

        async def fake_create_subprocess_exec(*argv, **kwargs):
            return fake_proc

        with tempfile.TemporaryDirectory() as ws:
            with patch.object(hyde_sandbox, "wrap_hyde_command", side_effect=lambda cmd, w: cmd), \
                 patch("asyncio.create_subprocess_exec", fake_create_subprocess_exec):
                with self.assertRaises(asyncio.TimeoutError):
                    await hyde_sandbox.run_sandboxed_claude(["claude"], ws, "p", timeout=0.01)

            self.assertTrue(fake_proc.killed)
            self.assertTrue(fake_proc.waited)


class ClaudeSubprocessLockFailClosedTest(unittest.TestCase):
    """flock(2) real, sin mock -- prueba el camino fail-closed: si el lock
    ya esta tomado, _acquire_cross_process_lock NO se cuelga para siempre,
    falla con TimeoutError explicito despues del timeout pedido. flock()
    bloquea entre dos open() distintos del MISMO proceso igual que entre
    procesos distintos (el lock es del open-file-description, no del
    proceso) -- valido para probar el timeout sin necesitar un segundo
    proceso de SO."""

    def test_timeout_explicito_si_el_lock_ya_esta_tomado(self):
        with tempfile.TemporaryDirectory() as ws:
            holder_fh = hyde_sandbox._acquire_cross_process_lock(ws, timeout=5)
            try:
                start = time.monotonic()
                with self.assertRaises(TimeoutError) as ctx:
                    hyde_sandbox._acquire_cross_process_lock(ws, timeout=0.2)
                elapsed = time.monotonic() - start
                self.assertGreaterEqual(elapsed, 0.2)
                self.assertIn("lock cross-proceso", str(ctx.exception))
            finally:
                hyde_sandbox._release_cross_process_lock(holder_fh)


_CROSS_PROCESS_WORKER = """
import asyncio, json, sys, time
sys.path.insert(0, {repo_root!r})
import hyde_sandbox

async def main():
    workspace_dir, tag, hold_seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
    events = []
    fh = await asyncio.to_thread(hyde_sandbox._acquire_cross_process_lock, workspace_dir, 10.0)
    events.append(("start", tag, time.monotonic()))
    await asyncio.sleep(hold_seconds)
    events.append(("end", tag, time.monotonic()))
    await asyncio.to_thread(hyde_sandbox._release_cross_process_lock, fh)
    print(json.dumps(events))

asyncio.run(main())
"""


class ClaudeSubprocessLockRealCrossProcessTest(unittest.TestCase):
    """Dos procesos de Python DE VERDAD (subprocess.Popen), no dos tasks
    de asyncio del mismo proceso -- eso es exactamente lo que un
    asyncio.Semaphore de modulo pasaria igual, sin probar nada sobre el
    problema real (Jacobs en el proceso de las_manos vs SubprocessMuscle
    en el proceso del REPL). time.monotonic() es CLOCK_MONOTONIC, un
    reloj de todo el sistema (no por-proceso) en Linux -- comparable entre
    los dos procesos hijos."""

    def test_flock_serializa_entre_dos_procesos_de_so_reales(self):
        repo_root = str(Path(hyde_sandbox.__file__).resolve().parent)
        with tempfile.TemporaryDirectory() as ws:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as script_f:
                script_f.write(_CROSS_PROCESS_WORKER.format(repo_root=repo_root))
                script_path = script_f.name

            try:
                p1 = subprocess.Popen(
                    [sys.executable, script_path, ws, "A", "0.3"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                p2 = subprocess.Popen(
                    [sys.executable, script_path, ws, "B", "0.3"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                out1, err1 = p1.communicate(timeout=15)
                out2, err2 = p2.communicate(timeout=15)
            finally:
                Path(script_path).unlink(missing_ok=True)

            self.assertEqual(p1.returncode, 0, err1)
            self.assertEqual(p2.returncode, 0, err2)

            events = json.loads(out1) + json.loads(out2)
            starts = sorted(e[2] for e in events if e[0] == "start")
            ends = sorted(e[2] for e in events if e[0] == "end")
            self.assertEqual(len(starts), 2)
            self.assertEqual(len(ends), 2)
            # Serializado de verdad, entre procesos de SO reales: el
            # segundo "start" ocurre DESPUES del primer "end".
            self.assertLess(ends[0], starts[1], f"no se serializó entre procesos: {events}")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Correr y confirmar que fallan (nada de esto existe todavía)**

Run: `cd /home/fruiz/jax && .venv/bin/python -m pytest _hyde_sandbox_test.py -v`
Expected: `AttributeError: module 'hyde_sandbox' has no attribute 'run_sandboxed_claude'` (o `'_acquire_cross_process_lock'`/`'_release_cross_process_lock'`) en las 5 pruebas.

- [ ] **Step 3: Implementar `run_sandboxed_claude` + el lock cross-proceso en `hyde_sandbox.py`**

Agregar a los imports (línea 58, junto a `json`/`os`/`shutil`):
```python
import asyncio
import fcntl
import json
import os
import shutil
import time
from pathlib import Path
```

Agregar al final del archivo (después de `wrap_hyde_command`, línea 185):
```python

# Serializa TODAS las invocaciones de `claude` sandboxeado entre si, sin
# importar el llamador NI el proceso de SO -- antes cada uno tenia su
# propio mecanismo (Jacobs: HYDE_SEMAPHORE en jacobs/executor.py, un
# asyncio.Semaphore que solo sirve DENTRO de un proceso; REPL: ninguno).
# Jacobs corre DENTRO del proceso de las_manos (systemd jax-las-manos);
# el REPL (jax/core/main.py) es un proceso de SO SEPARADO -- confirmado
# por enumeracion real de imports, 2026-08-25. Un asyncio.Semaphore de
# modulo no cruza esa frontera. Se usa flock(2) sobre un archivo dentro
# de workspace_dir en su lugar -- un lock a nivel de kernel, visible por
# CUALQUIER proceso que abra el mismo path. workspace_dir ya es
# JAX_WORKSPACE_DIR resuelto por cada llamador (fuente unica en
# /etc/jax/.env, ver jax-workspace-relocation-fix) -- el lock hereda esa
# misma fuente de verdad sin leer la env var de nuevo aca.
_CLAUDE_SUBPROCESS_LOCK_NAME = ".claude_subprocess.lock"
_CLAUDE_SUBPROCESS_LOCK_TIMEOUT_S = 30.0
_CLAUDE_SUBPROCESS_LOCK_POLL_S = 0.05


def _acquire_cross_process_lock(workspace_dir: str, timeout: float):
    """BLOQUEANTE -- llamar SOLO via asyncio.to_thread, nunca en el event
    loop (flock(2) no tiene equivalente async). Sondea con LOCK_NB en vez
    de bloquear en LOCK_EX puro para poder fail-closed con un timeout
    explicito: si otro proceso (REPL o las_manos) tiene el lock mas de
    `timeout` segundos, lanza TimeoutError con mensaje explicito en vez de
    colgar el thread para siempre.

    Devuelve el file handle abierto -- el llamador debe pasarlo a
    _release_cross_process_lock (tambien via to_thread) cuando termine,
    en un finally."""
    lock_path = Path(workspace_dir) / _CLAUDE_SUBPROCESS_LOCK_NAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except BlockingIOError:
            if time.monotonic() >= deadline:
                fh.close()
                raise TimeoutError(
                    f"no se pudo adquirir el lock cross-proceso de subprocess "
                    f"'claude' en {timeout}s ({lock_path}) -- otro proceso "
                    "(REPL o las_manos) sigue teniendo un claude corriendo. "
                    "Fail-closed: no se lanza sin exclusion mutua real."
                )
            time.sleep(_CLAUDE_SUBPROCESS_LOCK_POLL_S)


def _release_cross_process_lock(fh) -> None:
    """BLOQUEANTE (aunque en la practica instantaneo) -- llamar via
    asyncio.to_thread por simetria con _acquire_cross_process_lock."""
    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    fh.close()


async def run_sandboxed_claude(
    cmd: list[str], workspace_dir: str, prompt: str, timeout: float,
) -> tuple["asyncio.subprocess.Process", bytes, bytes]:
    """Unico punto de entrada aprobado para lanzar `claude` como subproceso
    -- ver policy/tests/test_claude_subprocess_solo_via_sandbox.py, que
    falla el CI si aparece un create_subprocess_exec/create_subprocess_shell
    de un comando que mencione "claude" fuera de este modulo.

    Aplica wrap_hyde_command (sandbox de bwrap, fail-closed via
    SandboxUnavailable si no hay bwrap -- NO se atrapa acá) y serializa
    con un flock(2) cross-proceso sobre workspace_dir (ver
    _acquire_cross_process_lock) -- no un asyncio.Semaphore, que no cruza
    la frontera real entre el proceso de las_manos (Jacobs) y el proceso
    del REPL.

    Devuelve (proc, stdout, stderr) crudos -- la interpretacion de exit
    code / contenido de stderr queda en el llamador, cada uno con su
    propio contrato de excepciones (RuntimeError en Jacobs,
    MuscleInvocationError en el REPL viejo -- no se unifican acá).

    En timeout (de la corrida real O de la espera del lock) o
    cancelacion: mata el proceso si llego a lanzarse, cosecha el zombie
    con wait(), y RE-LANZA la excepcion SIN envolver -- CancelledError
    debe seguir siendo CancelledError (Jacobs cancela _dispatch_step desde
    afuera con su propio wait_for; envolverla rompe la propagacion real de
    cancelacion de asyncio). TimeoutError del lock y TimeoutError del
    wait_for son la misma clase (asyncio.TimeoutError es alias de
    TimeoutError desde Python 3.11) -- ambos llamadores ya distinguen por
    esa clase, no hace falta un tipo nuevo."""
    sandboxed_cmd = wrap_hyde_command(cmd, workspace_dir)

    lock_fh = await asyncio.to_thread(
        _acquire_cross_process_lock, workspace_dir, _CLAUDE_SUBPROCESS_LOCK_TIMEOUT_S,
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *sandboxed_cmd,
            cwd=workspace_dir,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
                timeout=timeout,
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            proc.kill()
            await proc.wait()
            raise
    finally:
        await asyncio.to_thread(_release_cross_process_lock, lock_fh)

    return proc, stdout, stderr
```

- [ ] **Step 4: Correr los tests, confirmar que pasan**

Run: `cd /home/fruiz/jax && .venv/bin/python -m pytest _hyde_sandbox_test.py -v`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
cd /home/fruiz/jax
git add hyde_sandbox.py _hyde_sandbox_test.py
git commit -m "feat(hyde_sandbox): agrega run_sandboxed_claude, punto de entrada unico con flock cross-proceso"
```

---

### Task 2: Migrar los 2 call sites a `run_sandboxed_claude`

**Files:**
- Modify: `jacobs/executor.py:91-98,398-488` (quitar `HYDE_SEMAPHORE`, usar `run_sandboxed_claude`)
- Modify: `jax/muscles/subprocess_muscle.py:128-148` (usar `run_sandboxed_claude`)

**Interfaces:**
- Consume: `hyde_sandbox.run_sandboxed_claude` (Task 1).

- [ ] **Step 1: `jacobs/executor.py` — quitar `HYDE_SEMAPHORE` (líneas 91-98) y el import viejo**

Reemplazar:
```python
from hyde_sandbox import wrap_hyde_command
```
Por:
```python
from hyde_sandbox import run_sandboxed_claude
```

Borrar el bloque completo (líneas 91-98):
```python
# Semáforo específico de Hyde: el DAG de Jacobs puede programar dos steps
# `hyde` en la MISMA ola paralela (asyncio.gather), pero el mecanismo que
# estamos portando (subprocess_muscle.py, CLI viejo) siempre corrió secuencial
# — nunca hubo dos `claude` escribiendo a la vez en HYDE_WORKSPACE_DIR. Este
# semáforo serializa solo las invocaciones de Hyde entre sí, sin bloquear a
# las demás facetas de la misma ola (mismo patrón que GPU_SEMAPHORE en
# jax/muscles/ollama_muscle.py).
HYDE_SEMAPHORE = asyncio.Semaphore(1)
```

- [ ] **Step 2: `jacobs/executor.py::_invoke_hyde` — reemplazar el bloque de lanzamiento (líneas 435-471)**

Reemplazar:
```python
    # Sandbox de bubblewrap (2026-08-22, fix de fondo del P0 -- ver
    # hyde_sandbox.py y jax-hyde-bash-sin-jail-p0 en memoria). Confina a
    # nivel de namespace de montaje, no de heuristica de --allowedTools.
    # SandboxUnavailable NO se atrapa acá -- fail-closed (P10): sin bwrap,
    # el step falla con motivo explícito (_run_one_step ya lo hace vía su
    # except Exception genérico), nunca corre Hyde sin confinamiento.
    sandboxed_cmd = wrap_hyde_command(cmd, HYDE_WORKSPACE_DIR)

    # Un solo `claude` corriendo a la vez entre steps hyde (ver HYDE_SEMAPHORE).
    async with HYDE_SEMAPHORE:
        proc = await asyncio.create_subprocess_exec(
            *sandboxed_cmd,
            cwd=HYDE_WORKSPACE_DIR,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=safe_prompt.encode("utf-8")),
                timeout=timeout,
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            # kill() = SIGKILL en asyncio. Necesario en AMBOS casos: si vence
            # nuestro propio wait_for, o si _run_one_step nos cancela desde
            # afuera (envuelve _dispatch_step en su PROPIO asyncio.wait_for con
            # el MISMO timeout — con duraciones iguales, esa cancelación externa
            # casi siempre llega antes que nuestro TimeoutError interno, como
            # CancelledError, no TimeoutError). Sin cubrir los dos casos, el
            # proceso `claude` queda huérfano.
            proc.kill()
            await proc.wait()
            raise

        stdout_str = stdout.decode("utf-8", errors="replace")
        stderr_str = stderr.decode("utf-8", errors="replace")
```

Por:
```python
    # Sandbox de bubblewrap + semaforo compartido (ver
    # hyde_sandbox.py::run_sandboxed_claude -- unico punto de entrada
    # aprobado para lanzar `claude`, DEUDA.md "gobernanza de sub-agentes").
    # SandboxUnavailable NO se atrapa acá -- fail-closed (P10): sin bwrap,
    # el step falla con motivo explícito (_run_one_step ya lo hace vía su
    # except Exception genérico), nunca corre Hyde sin confinamiento.
    # TimeoutError/CancelledError tampoco se atrapan acá -- run_sandboxed_claude
    # ya mató y cosechó el proceso, y necesitamos que la excepción de
    # asyncio se propague SIN envolver (ver docstring de esa función).
    proc, stdout, stderr = await run_sandboxed_claude(
        cmd, HYDE_WORKSPACE_DIR, safe_prompt, timeout,
    )
    stdout_str = stdout.decode("utf-8", errors="replace")
    stderr_str = stderr.decode("utf-8", errors="replace")
```

- [ ] **Step 3: `jax/muscles/subprocess_muscle.py` — reemplazar el bloque de lanzamiento (líneas 122-148)**

Reemplazar:
```python
from jax.muscles.base import Muscle, MuscleInvocationError, MuscleTimeoutError
from hyde_sandbox import wrap_hyde_command
```
Por:
```python
from jax.muscles.base import Muscle, MuscleInvocationError, MuscleTimeoutError
from hyde_sandbox import run_sandboxed_claude
```

Reemplazar (líneas 122-148):
```python
        # Sandbox de bubblewrap (2026-08-22, fix de fondo del P0 -- ver
        # hyde_sandbox.py). SandboxUnavailable NO se atrapa acá -- fail-closed
        # (P10): sin bwrap, la llamada falla con motivo explícito, nunca
        # corre Hyde sin confinamiento.
        sandboxed_cmd = wrap_hyde_command(cmd, self.workspace_dir)

        proc = await asyncio.create_subprocess_exec(
            *sandboxed_cmd,
            cwd=self.workspace_dir,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=safe_prompt.encode("utf-8")),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            # kill() = SIGKILL en asyncio. Matamos el proceso colgado y
            # cosechamos el zombie con wait() antes de propagar el error.
            proc.kill()
            await proc.wait()
            raise MuscleTimeoutError(
                f"[{self.name}] sin respuesta en {self.timeout}s"
            )

        stdout_str = stdout.decode("utf-8", errors="replace")
        stderr_str = stderr.decode("utf-8", errors="replace")
```

Por:
```python
        # Sandbox de bubblewrap + semaforo compartido con Jacobs (ver
        # hyde_sandbox.py::run_sandboxed_claude -- unico punto de entrada
        # aprobado, DEUDA.md "gobernanza de sub-agentes"). SandboxUnavailable
        # NO se atrapa acá -- fail-closed (P10): sin bwrap, la llamada falla
        # con motivo explícito, nunca corre Hyde sin confinamiento.
        try:
            proc, stdout, stderr = await run_sandboxed_claude(
                cmd, self.workspace_dir, safe_prompt, self.timeout,
            )
        except asyncio.TimeoutError:
            raise MuscleTimeoutError(
                f"[{self.name}] sin respuesta en {self.timeout}s"
            )

        stdout_str = stdout.decode("utf-8", errors="replace")
        stderr_str = stderr.decode("utf-8", errors="replace")
```

Nota: `import asyncio` ya está presente en este archivo (línea 31) — sigue haciendo falta para `asyncio.TimeoutError` en el `except`.

- [ ] **Step 4: Correr los tests existentes que cubren estos dos call sites, confirmar cero regresión**

Run: `cd /home/fruiz/jax && PYTHONPATH=/home/fruiz/jax:/home/fruiz/jax/las_manos .venv/bin/python -m pytest jacobs/_direct_usage_test.py _hyde_sandbox_test.py -v`
Expected: todos en verde — `_direct_usage_test.py::test_dispatch_step_no_registra_usage_para_hyde_subprocess` sigue pasando (mockea `_invoke_hyde` completo, no le importa el cambio interno).

- [ ] **Step 5: Commit**

```bash
cd /home/fruiz/jax
git add jacobs/executor.py jax/muscles/subprocess_muscle.py
git commit -m "refactor(hyde): migra los 2 call sites de claude a run_sandboxed_claude, elimina HYDE_SEMAPHORE duplicado"
```

---

### Task 3: Guard de CI — ningún subprocess de `claude` fuera de `hyde_sandbox.py`

**Files:**
- Create: `policy/tests/test_claude_subprocess_solo_via_sandbox.py`
- Modify: `.github/workflows/policy.yml`

**Interfaces:**
- Produce: `test_no_naked_claude_subprocess()` (descubierto por pytest), `main()` (CLI).

- [ ] **Step 1: Escribir el scanner (mismo patrón que `test_no_fail_open_except.py`)**

```python
#!/usr/bin/env python3
"""Gobernanza de sub-agentes de Claude Code (DEUDA.md, "Sub-agentes de
Claude Code sin gobernanza real"): el UNICO punto de entrada aprobado
para lanzar `claude` como subproceso es
hyde_sandbox.py::run_sandboxed_claude() -- aplica el sandbox de bwrap y
el semaforo compartido (ver Task 1/2 del plan que agregó este scanner).

Enforcement mecanico y acotado (mismo espiritu que
test_no_fail_open_except.py): un archivo que (a) llama a
asyncio.create_subprocess_exec o asyncio.create_subprocess_shell, Y (b)
contiene la palabra "claude" en algun lugar de su codigo fuente, DEBE ser
hyde_sandbox.py -- cualquier otro archivo con esa combinacion esta
lanzando `claude` (o algo que lo referencia) por fuera del sandbox
compartido. Verificado 2026-08-25 contra el codigo real de ambos repos:
cero falsos positivos (ningun otro call site de subprocess async
-- las_manos/workers/ssh_worker.py, jax/voice/tts.py, jax/voice/ears.py,
jax-platform/backend/api/command.py -- menciona "claude").

No detecta un lanzamiento de `claude` disfrazado (sin la palabra literal
"claude" en el archivo, ej. leida de una variable de entorno con otro
nombre) -- eso queda como residuo conocido, mismo criterio que P10 con
las formas mas sutiles del patron fail-open.

Corre con:
  python3 policy/tests/test_claude_subprocess_solo_via_sandbox.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

_THIS_REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_roots() -> list[Path]:
    roots = [_THIS_REPO_ROOT]
    env_root = os.environ.get("JAX_PLATFORM_REPO_ROOT")
    roots.append(Path(env_root) if env_root else _THIS_REPO_ROOT.parent / "jax-platform")
    return roots


REPO_ROOTS = _repo_roots()

EXCLUDE_DIR_NAMES = {
    ".venv", "venv", "node_modules", ".git", ".worktrees", "worktrees",
    "__pycache__", "dist", "build",
}

ALLOWED_FILENAME = "hyde_sandbox.py"

_SUBPROCESS_CALL_NAMES = {"create_subprocess_exec", "create_subprocess_shell"}


def _iter_python_files():
    for root in REPO_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
                continue
            yield path


def _calls_create_subprocess(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else None
        )
        if name in _SUBPROCESS_CALL_NAMES:
            return True
    return False


def find_naked_claude_subprocess_files() -> list[str]:
    violations = []
    for path in _iter_python_files():
        if path.name == ALLOWED_FILENAME:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "claude" not in source.lower():
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        if _calls_create_subprocess(tree):
            violations.append(str(path))
    return violations


def test_no_naked_claude_subprocess() -> None:
    violations = find_naked_claude_subprocess_files()
    assert not violations, (
        f"{len(violations)} archivo(s) lanzan un subprocess async y "
        "mencionan 'claude', fuera de hyde_sandbox.py::run_sandboxed_claude() "
        "-- cualquier invocacion de `claude` como subproceso debe pasar por "
        "ese wrapper (sandbox + semaforo compartido):\n" + "\n".join(violations)
    )


def main() -> int:
    violations = find_naked_claude_subprocess_files()
    if violations:
        print(f"FAIL — {len(violations)} archivo(s):")
        for v in violations:
            print(f"  {v}")
        return 1
    print("OK — ningun subprocess de 'claude' fuera de hyde_sandbox.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Correr contra el estado actual del repo (post Task 2), confirmar cero violaciones**

Run: `python3 /home/fruiz/jax/policy/tests/test_claude_subprocess_solo_via_sandbox.py`
Expected: `OK — ningun subprocess de 'claude' fuera de hyde_sandbox.py` — si Task 2 se aplicó bien, `jacobs/executor.py` y `subprocess_muscle.py` ya no llaman `create_subprocess_exec` directamente.

- [ ] **Step 3: Prueba de que el scanner SÍ detecta una violación real — agregar temporalmente un call site falso, confirmar que falla, y revertir**

Run:
```bash
cd /home/fruiz/jax
cat > /tmp/_violation_probe.py << 'EOF'
import asyncio
CLAUDE_BIN = "claude"
async def bad():
    return await asyncio.create_subprocess_exec(CLAUDE_BIN)
EOF
cp /tmp/_violation_probe.py scripts/_violation_probe.py
python3 policy/tests/test_claude_subprocess_solo_via_sandbox.py
rm scripts/_violation_probe.py
```
Expected: `FAIL — 1 archivo(s):` listando `scripts/_violation_probe.py` — confirma que el scanner detecta el patrón real antes de confiar en que "pasa" solo porque no encontró nada.

- [ ] **Step 4: Wirear a CI, agregando un job nuevo junto al de P10**

En `.github/workflows/policy.yml`, agregar un segundo job (después de `no-fail-open-except`):
```yaml
  no-naked-claude-subprocess:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install pytest
      - run: python -m pytest policy/tests/test_claude_subprocess_solo_via_sandbox.py -v
```

- [ ] **Step 5: Commit**

```bash
cd /home/fruiz/jax
git add policy/tests/test_claude_subprocess_solo_via_sandbox.py .github/workflows/policy.yml
git commit -m "feat(policy): agrega scanner de CI -- ningun subprocess de claude fuera de hyde_sandbox.py"
```

---

### Task 4 — ELIMINADO (era puramente edición de `DEUDA.md`)

Política explícita del usuario (2026-08-25): nada va a `DEUDA.md` en este plan. El único ítem de `DEUDA.md` que se toca en toda esta ronda es la red de Hyde, en el plan `hyde-semaforo-y-deuda-precision`. Este plan termina en Task 3 — el ítem de gobernanza de sub-agentes queda cerrado en código y en CI (scanner real, probado con caso negativo), pero `DEUDA.md` no se actualiza; el usuario decide si y cuándo lo refleja ahí.

---

## Self-Review

- **Cobertura del spec:** el ítem pedía cerrar el gap prospectivo (futuro musculo sin gobernanza) — Task 1/2 centralizan el único mecanismo real, Task 3 lo convierte en invariante verificable por CI, Task 4 documenta el cierre con precisión.
- **Consistencia de tipos:** `run_sandboxed_claude` devuelve `(proc, stdout, stderr)` en los 3 usos (Task 1 test, Task 2 Step 2 y Step 3) — mismo orden, mismos tipos (`bytes` crudos, no decodificados, decisión consciente para no imponer un único manejo de encoding a ambos llamadores).
- **Corrección de diseño (2026-08-25, antes de ejecutar Task 1):** el borrador original de Task 1 usaba `asyncio.Semaphore(1)` como "semáforo compartido". Enumeración real de imports confirmó que `jacobs/executor.py` (dentro del proceso `las_manos`) y `SubprocessMuscle` (único importador real: `jax/core/main.py`, el REPL) corren en procesos de SO distintos — un `asyncio.Semaphore` de módulo no cruza esa frontera, así que el diseño original no cerraba la concurrencia cross-proceso que el propio plan dice resolver. Reemplazado por `flock(2)` sobre `workspace_dir/.claude_subprocess.lock` vía `asyncio.to_thread`, con timeout explícito fail-closed y un test de exclusión con dos procesos de SO reales (`subprocess.Popen`, no dos tasks de asyncio).
- **Sin placeholders:** todo el código de los 3 tasks es el diff/archivo completo a aplicar.
- **Dependencia hacia adelante:** el plan `2026-08-25-hyde-semaforo-y-deuda-precision.md` depende de que este plan (Task 1-2) ya esté aplicado — usa `_hyde_sandbox_test.py` como el mismo archivo a extender, no lo recrea.
