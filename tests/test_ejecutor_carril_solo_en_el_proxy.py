# tests/test_ejecutor_carril_solo_en_el_proxy.py
"""El carril del Ejecutor sólo lo toma el proxy (cuenta fruiz). Si otro módulo lo usa,
los locks pueden nacer con el umask de otra cuenta (hallazgo 7, LEDGER 2026-09-17)."""
import ast
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PERMITIDOS = {"jax/ejecutor/proxy_carril.py", "jax/ejecutor/prioridad.py"}
NOMBRES = {"carril_ejecutor", "carril_ejecutor_async"}


def test_solo_el_proxy_usa_el_carril_del_ejecutor():
    usos = []
    for ruta in RAIZ.rglob("*.py"):
        rel = ruta.relative_to(RAIZ).as_posix()
        if rel.startswith(("tests/", ".venv/", "las_manos/.venv/")) or "/site-packages/" in rel or rel in PERMITIDOS:
            continue
        try:
            arbol = ast.parse(ruta.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # fail-soft: archivos que no parsean (p. ej. _director_patch) no pueden importar nada al correr
            continue
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Name) and nodo.id in NOMBRES or isinstance(nodo, ast.Attribute) and nodo.attr in NOMBRES \
                    or isinstance(nodo, ast.alias) and nodo.name in NOMBRES:
                usos.append(rel)
    assert usos == [], usos
