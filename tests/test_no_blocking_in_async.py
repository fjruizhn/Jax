#!/usr/bin/env python3
"""Ninguna llamada BLOQUEANTE dentro de un `async def` — politica 3 de LAS
CUATRO DEL RENDIMIENTO, verificada sobre el arbol entero y no sobre la memoria
de quien revisa.

QUE ATRAPA Y POR QUE IMPORTA. Un `subprocess.run`, un `requests.get` o un
`time.sleep` dentro de una corrutina no bloquea "esa peticion": bloquea el
EVENT LOOP entero. Mientras corre, el proceso no atiende a nadie mas -- ni al
health check, ni al WebSocket de otro usuario, ni al turno de chat que ya
estaba a la mitad. Es la falla mas barata de escribir y la mas cara de
diagnosticar, porque se ve como "el servidor va lento a veces".

Es un test de CLASE, no de casos: recorre todos los .py del arbol y falla ante
CUALQUIER llamada bloqueante nueva dentro de una corrutina. Un test que
enumerara los sitios conocidos no serviria de nada -- el proximo se escribe
manana y ningun caso lo nombra.

Medido 2026-09-11: dos hallazgos, los dos en motor_registry/worker.py
(`git show` por archivo escrito y `git reset --hard`, timeout 10-15 s).

La salida correcta NO es sacar el subprocess: es `asyncio.to_thread(...)`, que
lo corre en un hilo y devuelve el control al loop mientras tanto.
"""
from __future__ import annotations

import ast
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# Directorios que no son codigo de servicio (no corren dentro de un loop vivo).
EXCLUIDOS = {".git", ".venv", "node_modules", "__pycache__", "docs", "workspace"}

# Archivos que NO son modulos Python aunque terminen en .py. Se declaran con
# motivo, igual que en test_aiomysql_connect_timeout_tripwire.py. Uno que no
# parsee y NO este aqui ES un hallazgo: ver
# test_un_archivo_ilegible_no_declarado_es_hallazgo.
# Vacía desde el 2026-09-16 (E-01): el único no-módulo real del árbol,
# _director_patch/routes_block.py, se retiró. El mecanismo se conserva y se
# ejercita con un .py roto declarado a propósito dentro del test.
_NO_PARSEA: dict[str, str] = {}

# Nombre COMPLETO de la llamada tal como aparece escrita. Se compara por texto
# del atributo y no resolviendo el import: un `from time import sleep` seguido
# de `sleep(5)` no se atrapa aca, y esta bien -- este test cierra la forma
# comun, no pretende ser un analizador de flujo. Lo que no cierra, lo cierra la
# revision.
BLOQUEANTES = {
    "time.sleep",
    "requests.get", "requests.post", "requests.put", "requests.delete",
    "requests.head", "requests.patch", "requests.request",
    "subprocess.run", "subprocess.call", "subprocess.check_call",
    "subprocess.check_output", "subprocess.Popen",
    "urllib.request.urlopen",
    "socket.create_connection",
}


def _archivos_py(raiz: Path = RAIZ):
    for p in raiz.rglob("*.py"):
        if any(parte in EXCLUIDOS for parte in p.parts):
            continue
        # Los tests y los scripts de diagnostico corren fuera de un servicio.
        nombre = p.name
        if nombre.startswith("_") and nombre.endswith("_test.py"):
            continue
        if nombre.startswith("test_") or p.parent.name == "tests":
            continue
        if p.parent.name == "scripts":
            continue
        if ".backup-" in nombre or ".old-" in str(p):
            continue
        yield p


def _hallazgos_en(path: Path, raiz: Path = RAIZ):
    try:
        arbol = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        # ARREGLADO 2026-09-16: antes `return []` -- un archivo que el analizador
        # no puede leer quedaba SIN REVISAR y el test pasaba igual, que es
        # exactamente el fail-open que esta familia de controles persigue. Un
        # `.py` ilegible o es un no-modulo declarado, o es un hallazgo.
        rel = path.relative_to(raiz).as_posix()
        if rel in _NO_PARSEA:
            return []
        return [f"{rel}: no se pudo analizar (SyntaxError) -- sin revisar"]
    encontrados = []

    class V(ast.NodeVisitor):
        def visit_AsyncFunctionDef(self, nodo):
            for hijo in ast.walk(nodo):
                # Una funcion sincrona ANIDADA dentro de la corrutina esta bien:
                # es exactamente lo que se le pasa a asyncio.to_thread.
                if isinstance(hijo, ast.FunctionDef) and hijo is not nodo:
                    continue
                if isinstance(hijo, ast.Call):
                    try:
                        nombre = ast.unparse(hijo.func)
                    except Exception as exc:  # fail-closed: se reporta como no analizable, nunca se salta
                        # ARREGLADO 2026-09-16: antes `continue` -- si el nombre
                        # de la llamada no se podia reconstruir, esa llamada
                        # quedaba SIN REVISAR y una bloqueante podia colarse sin
                        # que nadie lo supiera. Ahora se reporta.
                        encontrados.append(
                            f"{path.relative_to(raiz)}:{hijo.lineno}: llamada no "
                            f"analizable dentro de `async def {nodo.name}` ({exc}) "
                            f"-- revisar a mano"
                        )
                        continue
                    if nombre in BLOQUEANTES:
                        encontrados.append(
                            f"{path.relative_to(raiz)}:{hijo.lineno}: {nombre}() "
                            f"dentro de `async def {nodo.name}`"
                        )
            self.generic_visit(nodo)

    V().visit(arbol)
    return encontrados


@contextmanager
def _temporal_fuera_del_checkout():
    """Un .py sintético para los controles, dentro de un directorio temporal
    FUERA del checkout: una corrida matada o dos en paralelo no dejan un
    archivo roto a la vista del escaneo del árbol completo. Se escanea con
    `_hallazgos_en(ruta, ruta.parent)`."""
    with tempfile.TemporaryDirectory(prefix="jax-tripwire-") as directorio:
        with tempfile.NamedTemporaryFile("w", suffix=".py", dir=directorio, delete=True) as fh:
            yield fh


class NoBlockingInAsyncTest(unittest.TestCase):
    def test_ninguna_llamada_bloqueante_dentro_de_una_corrutina(self):
        hallazgos = []
        for p in _archivos_py():
            hallazgos.extend(_hallazgos_en(p))
        self.assertEqual(
            hallazgos, [],
            "llamadas bloqueantes dentro de `async def` (bloquean el event loop "
            "ENTERO, no solo esa peticion):\n  " + "\n  ".join(hallazgos) +
            "\nEnvolverlas en `await asyncio.to_thread(...)`.",
        )

    def test_el_detector_reconoce_un_bloqueo_de_verdad(self):
        """El freno se ejercita: si el detector no atrapa un caso obvio, el
        verde del test de arriba no significa nada (un control que no falla no
        valida nada)."""
        codigo = (
            "import subprocess\n"
            "async def f():\n"
            "    subprocess.run(['ls'])\n"
        )
        with _temporal_fuera_del_checkout() as fh:
            fh.write(codigo)
            fh.flush()
            hallazgos = _hallazgos_en(Path(fh.name), Path(fh.name).parent)
        self.assertEqual(len(hallazgos), 1, f"el detector no vio el bloqueo: {hallazgos}")


    def test_un_archivo_ilegible_no_declarado_es_hallazgo(self):
        """Antes se devolvia [] y el archivo no se revisaba nunca. El control
        que se salta lo que no entiende no es un control."""
        with _temporal_fuera_del_checkout() as fh:
            fh.write("def f(:\n    pass\n")
            fh.flush()
            hallazgos = _hallazgos_en(Path(fh.name), Path(fh.name).parent)
        self.assertEqual(len(hallazgos), 1, "un archivo ilegible se saltó en silencio")
        self.assertIn("no se pudo analizar", hallazgos[0])

    def test_una_llamada_no_analizable_es_hallazgo(self):
        """Si no se puede reconstruir el nombre de la llamada, se reporta para
        revisar a mano; NUNCA se salta, porque podria ser la bloqueante."""
        codigo = "async def f():\n    subprocess.run(['ls'])\n"
        original = ast.unparse

        def unparse_roto(nodo):
            raise ValueError("no se puede reconstruir")

        with _temporal_fuera_del_checkout() as fh:
            fh.write(codigo)
            fh.flush()
            ast.unparse = unparse_roto
            try:
                hallazgos = _hallazgos_en(Path(fh.name), Path(fh.name).parent)
            finally:
                ast.unparse = original
        self.assertEqual(len(hallazgos), 1, f"la llamada se saltó en silencio: {hallazgos}")
        self.assertIn("no analizable", hallazgos[0])

    def test_un_no_modulo_declarado_no_ensucia(self):
        """Control de la excepción: lo declarado no cuenta, y si algún día
        parsea, la entrada sobra. Con la tabla real vacía desde E-01
        (2026-09-16), la declaración se ejercita con un .py roto de mentira."""
        from unittest.mock import patch
        with _temporal_fuera_del_checkout() as fh:
            fh.write("def f(:\n    pass\n")
            fh.flush()
            ruta = Path(fh.name)
            rel = ruta.relative_to(ruta.parent).as_posix()
            with patch.dict(_NO_PARSEA, {rel: "roto a propósito para este test"}):
                self.assertEqual(_hallazgos_en(ruta, ruta.parent), [])
        for rel in _NO_PARSEA:
            ruta = RAIZ / rel
            self.assertTrue(ruta.exists(), f"{rel} ya no existe: retirar de _NO_PARSEA")
            self.assertEqual(_hallazgos_en(ruta), [], f"{rel} esta declarado y no deberia dar hallazgo")


if __name__ == "__main__":
    unittest.main()


class ControlesFueraDelCheckoutTest(unittest.TestCase):
    """Revisión final del frente E: los controles escribían su .py sintético
    (a veces roto a propósito) en la raíz del checkout. Una corrida matada a
    mitad de camino, o dos corridas en paralelo, dejaban ese archivo a la vista
    del escaneo del árbol completo y todos los tripwires se ponían rojos. El
    escaneo recibe la raíz inyectada y los controles escriben en un directorio
    temporal fuera del checkout."""

    def test_los_controles_no_escriben_en_el_checkout(self):
        arbol = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        ofensores = [
            n.value.lineno for n in ast.walk(arbol)
            if isinstance(n, ast.keyword) and n.arg == "dir"
            and isinstance(n.value, ast.Name) and n.value.id == "RAIZ"
        ]
        self.assertEqual(ofensores, [], f"un control escribe dentro del checkout (dir=RAIZ) en las líneas {ofensores}")

    def test_la_raiz_del_escaneo_se_inyecta(self):
        with tempfile.TemporaryDirectory(prefix="jax-tripwire-") as d:
            ruta = Path(d) / "roto.py"
            ruta.write_text("def f(:\n    pass\n", encoding="utf-8")
            hallazgos = _hallazgos_en(ruta, Path(d))
        self.assertEqual(len(hallazgos), 1, hallazgos)
        self.assertTrue(hallazgos[0].startswith("roto.py"), hallazgos)
