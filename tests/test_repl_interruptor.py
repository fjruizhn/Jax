"""El REPL y `jax --task` bajo el freno (plan 2026-09-16-frente-b-kill-switch,
Task 5). Se lee el código fuente de jax/core/main.py, igual que
test_degradaciones_declaradas.py: importar main arrastra voz y oído.

Antes: la ruta salía de config/config.toml y se miraba una sola vez antes
de invocar; `jax --task` corría a Hyde entero con el freno puesto después.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
FUENTE = (RAIZ / "jax" / "core" / "main.py").read_text(encoding="utf-8")
ARBOL = ast.parse(FUENTE)


def _funcion(nombre):
    return next(n for n in ast.walk(ARBOL)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == nombre)


def _es_llamada_a(nodo, nombre):
    return isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name) and nodo.func.id == nombre


@pytest.mark.parametrize("nombre", ["run_task", "main"])
def test_cada_invoke_corre_bajo_el_interruptor(nombre):
    fn = _funcion(nombre)
    invokes = [n for n in ast.walk(fn)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "invoke"]
    assert invokes, "no hay invoke: el test no probaría nada"
    envueltas = {id(n.args[0]) for n in ast.walk(fn) if _es_llamada_a(n, "correr_con_interruptor") and n.args}
    assert [n.lineno for n in invokes if id(n) not in envueltas] == []


@pytest.mark.parametrize("nombre", ["run_task", "main"])
def test_la_ruta_sale_de_la_variable_y_sin_ella_se_sale_con_1(nombre):
    fn = _funcion(nombre)
    assert len([n for n in ast.walk(fn) if _es_llamada_a(n, "ruta_del_interruptor")]) == 1
    manejadores = [h for n in ast.walk(fn) if isinstance(n, ast.Try) for h in n.handlers
                   if isinstance(h.type, ast.Name) and h.type.id == "InterruptorSinConfigurar"]
    assert manejadores, f"{nombre} no falla cerrado sin JAX_KILL_SWITCH_PATH"
    assert any(ast.unparse(s) == "sys.exit(1)" for h in manejadores for s in h.body)


def test_main_ya_no_lee_la_ruta_de_la_config():
    assert "kill_switch_path" not in FUENTE
    assert "def kill_switch_active" not in FUENTE


def test_la_config_del_repl_no_declara_la_ruta():
    with open(RAIZ / "config" / "config.toml", "rb") as f:
        assert "kill_switch_path" not in tomllib.load(f)["jax"]
