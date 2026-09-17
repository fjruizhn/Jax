# policy/tests/test_ejecutor_lanza_solo_con_contratos.py
"""Nadie lanza el cerebro del Ejecutor, ni abre el proxy con un vigía, sin exigir los
contratos (spec 2026-09-15 §4; Principio IX; Ejecutor SP1 plan 6).

Guardia de CI por AST:
- un módulo que usa `remoto_claude` (el lanzamiento del arnés dentro de la jaula) sin
  `exigir_contratos` es violación;
- un módulo que usa `vigilar` (el vigía de C5: su latido es lo que hace servir al proxy)
  sin `exigir_contratos` es violación.
Los archivos declarados como aislados por cuenta en la guardia de subprocesos también
tienen que exigirlos o ser el lanzador.

Permitidos, con su razón (cada uno tiene que existir: test_los_permitidos_existen):
- cuenta_axioma.py: define el lanzador;
- canario_c1.py: ES el canario de C1, corre dentro de exigir_contratos, contra un upstream falso;
- scripts/ejecutor_contratos/probar_c3_corte.py: prueba de C3 contra un upstream falso, sin modelo;
- vigia.py: define el vigía;
- scripts/ejecutor_contratos/probar_c5.py: vigía con pausa y latido en un directorio temporal,
  sobre un registro grabado: el proxy de producción no lo ve.
Corre con: python -m pytest policy/tests/test_ejecutor_lanza_solo_con_contratos.py -v
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
LANZADORES = ("remoto_claude", "vigilar")
EXIGIR = "exigir_contratos"
PERMITIDOS = {
    "jax/ejecutor/contratos/cuenta_axioma.py": "define el lanzador",
    "jax/ejecutor/contratos/canario_c1.py": "es el canario de C1: corre dentro de exigir_contratos",
    "scripts/ejecutor_contratos/probar_c3_corte.py": "prueba de C3 contra un upstream falso, sin modelo",
    "jax/ejecutor/contratos/vigia.py": "define el vigía",
    "scripts/ejecutor_contratos/probar_c5.py": "vigía con pausa y latido en un directorio temporal",
}


def _nombres(arbol: ast.AST) -> set[str]:
    vistos = set()
    for n in ast.walk(arbol):
        if isinstance(n, ast.Name):
            vistos.add(n.id)
        elif isinstance(n, ast.Attribute):
            vistos.add(n.attr)
        elif isinstance(n, ast.alias):
            vistos.add(n.name.rsplit(".", 1)[-1])
    return vistos


def viola(fuente: str) -> bool:
    nombres = _nombres(ast.parse(fuente))
    return any(l in nombres for l in LANZADORES) and EXIGIR not in nombres


def _aislados_por_cuenta() -> dict:
    ruta = RAIZ / "policy" / "tests" / "test_claude_subprocess_solo_via_sandbox.py"
    spec = importlib.util.spec_from_file_location("guardia_subprocesos", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo._AISLADO_POR_CUENTA


def _archivos():
    for ruta in RAIZ.rglob("*.py"):
        rel = ruta.relative_to(RAIZ).as_posix()
        if (rel.startswith(("tests/", "policy/", ".venv/", "las_manos/.venv/", ".git/"))
                or "/site-packages/" in rel or "/node_modules/" in rel):
            continue
        yield rel, ruta


def test_nadie_lanza_sin_exigir():
    violaciones = []
    for rel, ruta in _archivos():
        if rel in PERMITIDOS:
            continue
        try:
            fuente = ruta.read_text(encoding="utf-8")
            if any(l in fuente for l in LANZADORES) and viola(fuente):
                violaciones.append(rel)
        except (SyntaxError, UnicodeDecodeError):  # fail-soft: un archivo que no parsea no puede importarse ni lanzar nada
            continue
    assert violaciones == [], violaciones


def test_los_aislados_por_cuenta_exigen_o_son_el_lanzador():
    for rel in _aislados_por_cuenta():
        if rel in PERMITIDOS:
            continue
        assert EXIGIR in _nombres(ast.parse((RAIZ / rel).read_text(encoding="utf-8"))), rel


def test_detecta_el_uso_directo_y_por_alias():
    assert viola("from jax.ejecutor.contratos.cuenta_axioma import remoto_claude\nremoto_claude(c)\n")
    assert viola("from jax.ejecutor.contratos import cuenta_axioma as ca\nca.remoto_claude(c)\n")
    assert viola("from jax.ejecutor.contratos import vigia\nawait vigia.vigilar(cfg, a, fin)\n")


def test_acepta_el_que_exige():
    assert not viola("from jax.ejecutor.contratos import arranque, cuenta_axioma\n"
                     "async def f(ctx):\n    await arranque.exigir_contratos(ctx)\n    cuenta_axioma.remoto_claude(ctx.cuenta)\n")


def test_los_permitidos_existen():
    for rel in PERMITIDOS:
        assert (RAIZ / rel).is_file(), rel
