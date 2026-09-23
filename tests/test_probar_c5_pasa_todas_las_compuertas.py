#!/usr/bin/env python3
"""Todo llamador de `validar_eleccion` pasa TODOS sus parámetros, sin depender
de un default.

POR QUE EXISTE (2026-09-23). La compuerta `admite_mismo_proveedor` entró el
2026-09-20 con default False ("un llamador olvidadizo obtiene el
comportamiento estricto"). arranque.py la pasó; scripts/ejecutor_contratos/
probar_c5.py no. La prueba real de C5, la que un tercero corre para saber si
C5 está vivo, contestó `c5_vivo=false` con la compuerta ABIERTA durante tres
días mientras producción arrancaba bien: la herramienta de verificación
contradecía al sistema. El default estricto está bien para el código nuevo;
en los llamadores conocidos, depender de él es este defecto.

Suite pura: lee el AST, no importa nada del Ejecutor.
"""
from __future__ import annotations

import ast
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DEFINICION = RAIZ / "jax" / "ejecutor" / "contratos" / "eleccion_c5.py"
LLAMADORES = (
    RAIZ / "scripts" / "ejecutor_contratos" / "probar_c5.py",
    RAIZ / "jax" / "ejecutor" / "contratos" / "arranque.py",
    RAIZ / "jax" / "ejecutor" / "contratos" / "eleccion_c5.py",
)


def _parametros() -> set[str]:
    arbol = ast.parse(DEFINICION.read_text(encoding="utf-8"))
    [fn] = [n for n in arbol.body if isinstance(n, ast.AsyncFunctionDef | ast.FunctionDef)
            and n.name == "validar_eleccion"]
    return {a.arg for a in fn.args.kwonlyargs + fn.args.args}


def _llamadas(ruta: Path):
    for nodo in ast.walk(ast.parse(ruta.read_text(encoding="utf-8"))):
        if isinstance(nodo, ast.Call):
            f = nodo.func
            nombre = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if nombre == "validar_eleccion":
                yield nodo


def test_hay_llamadores_que_revisar():
    # Si un refactor los renombra, este test tiene que enterarse, no quedar vacío.
    assert sum(1 for r in LLAMADORES for _ in _llamadas(r)) >= 3


def test_cada_llamador_pasa_todos_los_parametros():
    esperados = _parametros()
    assert "admite_mismo_proveedor" in esperados
    faltan = {}
    for ruta in LLAMADORES:
        for llamada in _llamadas(ruta):
            pasados = {k.arg for k in llamada.keywords}
            if esperados - pasados:
                faltan[f"{ruta.relative_to(RAIZ)}:{llamada.lineno}"] = sorted(esperados - pasados)
    assert faltan == {}, f"llamadores que dependen de un default de validar_eleccion: {faltan}"
