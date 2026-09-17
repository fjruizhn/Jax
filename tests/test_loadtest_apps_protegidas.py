"""Las apps de carga miden el camino REAL de LAS MANOS, y el camino real exige
credencial de servicio (las_manos/auth_servicio.py, 2026-09-17). Una app de
carga que monta los routers sin `proteger(app)` mide un endpoint que en
producción no existe así, y deja en el árbol una forma de servirlos abiertos.

Se lee por AST: cualquier `loadtest/*_app.py` que construya `FastAPI(...)`
tiene que llamar `proteger(app)` en el nivel del módulo.
"""
from __future__ import annotations

import ast
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
APPS = sorted((RAIZ / "loadtest").glob("*_app.py"))


def _construye_fastapi(arbol: ast.Module) -> bool:
    return any(isinstance(n, ast.Call) and getattr(n.func, "id", None) == "FastAPI"
               for n in ast.walk(arbol))


def _llama_proteger(arbol: ast.Module) -> bool:
    for n in arbol.body:
        if (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and getattr(n.value.func, "id", None) == "proteger"
                and n.value.args and getattr(n.value.args[0], "id", None) == "app"):
            return True
    return False


def test_hay_apps_de_carga_que_revisar():
    # Si el glob no encuentra nada, el test de abajo pasa sin mirar nada.
    assert len(APPS) >= 2


def test_toda_app_de_carga_monta_la_credencial_de_servicio():
    sin_proteger = [p.name for p in APPS
                    if _construye_fastapi(ast.parse(p.read_text()))
                    and not _llama_proteger(ast.parse(p.read_text()))]
    assert sin_proteger == []
