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
