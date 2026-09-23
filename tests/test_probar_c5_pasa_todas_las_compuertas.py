#!/usr/bin/env python3
"""La prueba real de C5 hace lo MISMO que el Ejecutor real: mismas compuertas,
mismo auditor.

POR QUE EXISTE (2026-09-23). La compuerta `admite_mismo_proveedor` entró el
2026-09-20 con default False ("un llamador olvidadizo obtiene el
comportamiento estricto"). arranque.py la pasó; scripts/ejecutor_contratos/
probar_c5.py no. La prueba que corre un tercero para saber si C5 está vivo
contestó `c5_vivo=false` con la compuerta ABIERTA durante tres días, mientras
producción arrancaba bien. Y elegía el auditor con `cfg.auditor_faceta`
(`thot`, nube) mientras una misión real en hall9000 usa
`auditor_faceta_local` (`el_juez`). La herramienta de verificación medía otro
sistema.

Suite pura: lee el AST, no importa nada del Ejecutor.
"""
from __future__ import annotations

import ast
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DEFINICION = RAIZ / "jax" / "ejecutor" / "contratos" / "eleccion_c5.py"
PROBAR_C5 = RAIZ / "scripts" / "ejecutor_contratos" / "probar_c5.py"
FUNCIONES = ("validar_eleccion", "validar_proveedores")
FUERA = {"tests", ".venv", "venv", "node_modules", ".git", "__pycache__", "workspace", "repo", "missions"}


def _parametros(funcion: str) -> set[str]:
    arbol = ast.parse(DEFINICION.read_text(encoding="utf-8"))
    [fn] = [n for n in arbol.body if isinstance(n, ast.AsyncFunctionDef | ast.FunctionDef) and n.name == funcion]
    return {a.arg for a in fn.args.kwonlyargs + fn.args.args}


def _es_test(ruta: Path) -> bool:
    # Las dos convenciones de tests del repo: `test_*.py` y `_*_test.py`
    # (jacobs/_arbitro_test.py). NO `*_test.py` a secas: eso sacaba código que no es
    # test (scripts/load_test.py, base_de_test.py) y lo dejaba sin revisar.
    return ruta.name.startswith("test_") or (ruta.name.startswith("_") and ruta.name.endswith("_test.py"))


def _archivos():
    """TODO .py del repo fuera de tests: un llamador nuevo en otro archivo entra solo.
    Un archivo que no se puede leer o no parsea NO se salta: rompe el test (un control
    que ignora lo que no entiende da verde sin haber mirado)."""
    for ruta in RAIZ.rglob("*.py"):
        if set(ruta.relative_to(RAIZ).parts) & FUERA or _es_test(ruta):
            continue
        yield ruta, ast.parse(ruta.read_text(encoding="utf-8"), filename=str(ruta))


def _nombre(llamada: ast.Call) -> str | None:
    f = llamada.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)


def _llamadas():
    for ruta, arbol in _archivos():
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Call) and _nombre(nodo) in FUNCIONES:
                yield ruta, nodo


def test_hay_llamadores_que_revisar():
    # Si un refactor los renombra, este test tiene que enterarse, no quedar vacío.
    encontrados = {(r.name, _nombre(n)) for r, n in _llamadas()}
    assert ("probar_c5.py", "validar_eleccion") in encontrados
    assert ("arranque.py", "validar_eleccion") in encontrados
    assert ("arranque.py", "validar_proveedores") in encontrados


def test_cada_llamador_pasa_todos_los_parametros():
    faltan = {}
    for ruta, llamada in _llamadas():
        esperados = _parametros(_nombre(llamada))
        pasados = {k.arg for k in llamada.keywords}  # **kwargs da arg None: cuenta como faltante
        if esperados - pasados:
            faltan[f"{ruta.relative_to(RAIZ)}:{llamada.lineno}"] = sorted(esperados - pasados)
    assert faltan == {}, f"llamadores que dependen de un default: {faltan}"


def _viene_de_la_config(valor: ast.expr) -> bool:
    """`cfg.admite_mismo_proveedor` (la ConfigC5 leída de axioma_config) o el parámetro
    homónimo que la trae de más arriba. Se exige la forma buena en vez de prohibir las
    malas: `False`, `not True`, `bool(0)` o `args.admite_mismo_proveedor` (un flag con
    default propio) quedan todos afuera."""
    return (isinstance(valor, ast.Attribute) and valor.attr == "admite_mismo_proveedor"
            and isinstance(valor.value, ast.Name) and valor.value.id == "cfg") or \
           (isinstance(valor, ast.Name) and valor.id == "admite_mismo_proveedor")


def test_la_compuerta_no_se_escribe_a_mano():
    # Un valor escrito a mano pasaría el test de arriba y reproduciría el defecto.
    a_mano = [f"{r.relative_to(RAIZ)}:{n.lineno}" for r, n in _llamadas()
              for k in n.keywords if k.arg == "admite_mismo_proveedor" and not _viene_de_la_config(k.value)]
    assert a_mano == []


def test_probar_c5_elige_el_auditor_como_el_ejecutor_real():
    arbol = ast.parse(PROBAR_C5.read_text(encoding="utf-8"))
    nombres = {_nombre(n) for n in ast.walk(arbol) if isinstance(n, ast.Call)}
    assert "elegir_y_resolver_auditor" in nombres
    # Y no vuelve a elegir por su cuenta con la clave que no corresponde a hall9000.
    atributos = {n.attr for n in ast.walk(arbol) if isinstance(n, ast.Attribute)}
    cadenas = {n.value for n in ast.walk(arbol) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "auditor_faceta" not in atributos | cadenas  # ni cfg.auditor_faceta ni getattr(cfg, "auditor_faceta")
