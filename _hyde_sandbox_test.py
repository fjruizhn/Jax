#!/usr/bin/env python3
"""hyde_sandbox: el lock cross-proceso del confinamiento de Hyde (flock(2)).
Antes de este modulo, el lanzamiento de `claude` se reimplementaba por su lado
-- uno con HYDE_SEMAPHORE (un asyncio.Semaphore, valido solo DENTRO de un
proceso), el otro SIN NINGUN lock. Un asyncio.Semaphore de modulo no cruza la
frontera entre procesos de SO -- se usa flock(2) en su lugar, visible por
cualquier proceso que abra el mismo path. (T16, 2026-10-02: `run_sandboxed_claude`,
que usaba este lock, y sus pruebas se retiraron junto con el REPL y la ruta directa
de Jacobs; el lock y wrap_hyde_command siguen, los usa cli_sandbox.) El archivo del lock vive en /run/jax-locks/hyde del HOST
(derivado de workspace_dir por hash), NUNCA dentro de workspace_dir: ese
directorio se bindea read-write dentro del sandbox y el `claude` confinado
podia borrar el archivo, lo que dejaba al siguiente acquire crear un inodo
nuevo y correr en paralelo con el holder -- ver
ClaudeSubprocessLockPathOutsideSandboxTest.

No requieren bwrap real. ClaudeSubprocessLockFailClosedTest
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
import contextlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import grp
import shutil

import cli_sandbox
import hyde_sandbox

# El lock de Hyde es COMPARTIDO (MAJOR-14): directorio y grupo son constantes del
# codigo y el directorio/archivo se verifican (dueno root, grupo del lock, sin
# escritura ajena). Los tests no corren como root ni tienen el grupo real: se
# parchean las constantes a un directorio efimero propio y al grupo del que corre,
# y el dueno se inyecta por `uid_esperado`. Los procesos hijos
# (ClaudeSubprocessLockRealCrossProcessTest) reciben ambos por argv.
_LOCKS_DE_PRUEBA = None
_PATCHES = []
_GRUPO_DE_PRUEBA = grp.getgrgid(os.getgid()).gr_name


def setUpModule():
    global _LOCKS_DE_PRUEBA
    _LOCKS_DE_PRUEBA = tempfile.mkdtemp(prefix="hyde-locks-test-")
    d = os.path.join(_LOCKS_DE_PRUEBA, "locks")
    os.mkdir(d, 0o750)
    os.chmod(d, 0o750)
    for p in (
        patch.multiple(hyde_sandbox, HYDE_LOCK_DIR=d, HYDE_LOCK_GROUP=_GRUPO_DE_PRUEBA),
        patch.object(hyde_sandbox, "_acquire_cross_process_lock", _adquirir_como_tmpfiles),
    ):
        p.start()
        _PATCHES.append(p)


def tearDownModule():
    for p in _PATCHES:
        p.stop()
    _PATCHES.clear()
    shutil.rmtree(_LOCKS_DE_PRUEBA, ignore_errors=True)


def _sembrar_lock(workspace_dir: str) -> None:
    """Lo que hace tmpfiles.d en el host: crea el archivo del lock del workspace."""
    ruta = hyde_sandbox._lock_path_for_workspace(workspace_dir)
    # atomico y ya con el modo final: dos hilos sembrando a la vez no se ven a medias
    with contextlib.suppress(FileExistsError):
        os.close(os.open(ruta, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640))


_ADQUIRIR_REAL = hyde_sandbox._acquire_cross_process_lock


def _adquirir_como_tmpfiles(workspace_dir: str, timeout: float, **kw):
    """Reemplaza a la funcion real DURANTE ESTE MODULO: hace de tmpfiles.d (siembra el
    archivo del lock si falta) e inyecta el dueno del que corre el test; todo lo
    demas -- directorio, grupo, modos, flock -- es la verificacion real. Asi tambien
    el codigo que no pasa esos parametros corre contra el lock real."""
    if not hyde_sandbox._lock_path_for_workspace(workspace_dir).exists():
        _sembrar_lock(workspace_dir)
    kw.setdefault("uid_esperado", os.getuid())
    return _ADQUIRIR_REAL(workspace_dir, timeout, **kw)


def _adquirir(workspace_dir: str, timeout: float):
    return hyde_sandbox._acquire_cross_process_lock(workspace_dir, timeout)


class LlamadaAlLockCompartidoTest(unittest.TestCase):
    """MINOR-25 (reauditoria 2026-10-02): el modulo entero parchea
    `_acquire_cross_process_lock` con un envoltorio que INYECTA el dueno del que corre el
    test, asi que un `uid_esperado` distinto de 0 (o un gid fijo) en produccion pasaba todos
    los tests. Aqui se mira la firma de la funcion REAL: dueno root y grupo del sistema."""

    def test_los_defaults_de_la_funcion_real_son_root_y_grupo_real(self):
        p = inspect.signature(_ADQUIRIR_REAL).parameters
        self.assertEqual(p["uid_esperado"].default, 0)
        self.assertIsNone(p["gid_esperado"].default)
        self.assertEqual(p["uid_esperado"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(p["gid_esperado"].kind, inspect.Parameter.KEYWORD_ONLY)


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
            holder_fh = _adquirir(ws, 5)
            try:
                start = time.monotonic()
                with self.assertRaises(TimeoutError) as ctx:
                    hyde_sandbox._acquire_cross_process_lock(ws, timeout=0.2)
                elapsed = time.monotonic() - start
                self.assertGreaterEqual(elapsed, 0.2)
                self.assertIn("lock cross-proceso", str(ctx.exception))
            finally:
                hyde_sandbox._release_cross_process_lock(holder_fh)


class ClaudeSubprocessLockPathOutsideSandboxTest(unittest.TestCase):
    """Regresion del hallazgo CRITICO de la review final: el archivo del
    lock vivia DENTRO de workspace_dir, que wrap_hyde_command bindea
    READ-WRITE dentro del sandbox. El `claude` confinado (o cualquier
    limpieza del workspace) podia borrarlo; flock(2) es del inodo, asi que
    el siguiente open(path, "w") creaba un inodo NUEVO y tomaba su lock al
    instante -- dos claude en paralelo, sin error y sin log. El fix es
    estructural: el lock vive en /run/jax-locks/hyde del HOST, que el sandbox nunca
    ve (no monta /run; recibe su propio --tmpfs /tmp privado). Si el path del lock no esta
    dentro de workspace_dir, el proceso confinado no puede tocarlo -- no
    hay nada que re-simular."""

    def test_el_lock_no_vive_dentro_del_workspace(self):
        with tempfile.TemporaryDirectory() as ws:
            lock_path = hyde_sandbox._lock_path_for_workspace(ws)
            self.assertFalse(
                lock_path.resolve().is_relative_to(Path(ws).resolve()),
                f"el lock {lock_path} esta dentro del workspace bindeado RW {ws}",
            )

    def test_el_handle_devuelto_apunta_fuera_del_workspace(self):
        # No solo el helper: el archivo REALMENTE abierto por
        # _acquire_cross_process_lock tiene que estar fuera del workspace.
        with tempfile.TemporaryDirectory() as ws:
            fh = _adquirir(ws, 5)
            try:
                real_path = Path(os.readlink(f"/proc/self/fd/{fh.fileno()}")).resolve()
                self.assertFalse(
                    real_path.is_relative_to(Path(ws).resolve()),
                    f"el lock abierto {real_path} esta dentro del workspace {ws}",
                )
                # Y el workspace no queda con ningun archivo de lock dentro.
                self.assertEqual(
                    sorted(p.name for p in Path(ws).iterdir()), [],
                    "quedo un archivo de lock dentro del workspace bindeado RW",
                )
            finally:
                hyde_sandbox._release_cross_process_lock(fh)

    def test_workspaces_distintos_tienen_locks_independientes(self):
        with tempfile.TemporaryDirectory() as ws_a, tempfile.TemporaryDirectory() as ws_b:
            self.assertNotEqual(
                hyde_sandbox._lock_path_for_workspace(ws_a),
                hyde_sandbox._lock_path_for_workspace(ws_b),
            )
            # Y el lock de uno no bloquea al otro.
            fh_a = _adquirir(ws_a, 5)
            try:
                fh_b = _adquirir(ws_b, 1)
                hyde_sandbox._release_cross_process_lock(fh_b)
            finally:
                hyde_sandbox._release_cross_process_lock(fh_a)


_CROSS_PROCESS_WORKER = """
import asyncio, json, sys, time
sys.path.insert(0, {repo_root!r})
import os
import hyde_sandbox

async def main():
    workspace_dir, tag, hold_seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
    hyde_sandbox.HYDE_LOCK_DIR, hyde_sandbox.HYDE_LOCK_GROUP = sys.argv[4], sys.argv[5]
    events = []
    fh = await asyncio.to_thread(
        lambda: hyde_sandbox._acquire_cross_process_lock(workspace_dir, 10.0, uid_esperado=os.getuid()))
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
    problema real (dos procesos de SO distintos que toman el mismo lock). time.monotonic() es CLOCK_MONOTONIC, un
    reloj de todo el sistema (no por-proceso) en Linux -- comparable entre
    los dos procesos hijos."""

    def test_flock_serializa_entre_dos_procesos_de_so_reales(self):
        repo_root = str(Path(hyde_sandbox.__file__).resolve().parent)
        with tempfile.TemporaryDirectory() as ws:
            _sembrar_lock(ws)
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as script_f:
                script_f.write(_CROSS_PROCESS_WORKER.format(repo_root=repo_root))
                script_path = script_f.name

            try:
                p1 = subprocess.Popen(
                    [sys.executable, script_path, ws, "A", "0.3", hyde_sandbox.HYDE_LOCK_DIR, _GRUPO_DE_PRUEBA],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                p2 = subprocess.Popen(
                    [sys.executable, script_path, ws, "B", "0.3", hyde_sandbox.HYDE_LOCK_DIR, _GRUPO_DE_PRUEBA],
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
