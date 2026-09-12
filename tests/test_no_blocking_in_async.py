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
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# Directorios que no son codigo de servicio (no corren dentro de un loop vivo).
EXCLUIDOS = {".git", ".venv", "node_modules", "__pycache__", "docs", "workspace"}

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


def _archivos_py():
    for p in RAIZ.rglob("*.py"):
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


def _hallazgos_en(path: Path):
    try:
        arbol = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
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
                    except Exception:
                        continue
                    if nombre in BLOQUEANTES:
                        encontrados.append(
                            f"{path.relative_to(RAIZ)}:{hijo.lineno}: {nombre}() "
                            f"dentro de `async def {nodo.name}`"
                        )
            self.generic_visit(nodo)

    V().visit(arbol)
    return encontrados


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
        import tempfile
        codigo = (
            "import subprocess\n"
            "async def f():\n"
            "    subprocess.run(['ls'])\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", dir=RAIZ, delete=True) as fh:
            fh.write(codigo)
            fh.flush()
            hallazgos = _hallazgos_en(Path(fh.name))
        self.assertEqual(len(hallazgos), 1, f"el detector no vio el bloqueo: {hallazgos}")


if __name__ == "__main__":
    unittest.main()
