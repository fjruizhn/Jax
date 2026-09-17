"""El relanzador CLI es un cliente de jacobs/continuar.py (spec 2026-09-17 §5.4):
sin escrituras propias y sin leer /etc/jax/.env al importarse.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import importlib.util
import os
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import pytest  # noqa: E402

from jacobs import continuar  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus  # noqa: E402

RUTA = Path(__file__).resolve().parents[1] / "tools" / "jacobs_relaunch.py"


def _modulo():
    spec = importlib.util.spec_from_file_location("jacobs_relaunch_bajo_test", RUTA)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_llama_a_continuar_y_corre_la_corrida_que_devuelve():
    cli = _modulo()
    pipeline = Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma", mode="autonomous", run_epoch=2)
    final = pipeline.model_copy(update={"status": PipelineStatus.completed})
    respuesta = {"run_epoch": 2, "pasos_reusados": [0], "pasos_a_correr": [1], "costo_max_usd": "0.100000"}
    with patch("jacobs.continuar.continuar", AsyncMock(return_value=(respuesta, pipeline))) as servicio, \
         patch("jacobs.executor.run_pipeline", AsyncMock()) as corrida, \
         patch("jacobs.store.pipeline_get", AsyncMock(return_value=final)), \
         patch("jacobs.store.steps_by_pipeline", AsyncMock(return_value=[])):
        rc = asyncio.run(cli.relanzar("p1", {"1": "ada"}, Decimal("0.5")))
    assert rc == 0
    servicio.assert_awaited_once_with("p1", "plataforma", reasignar={"1": "ada"},
                                      costo_max_aceptado_usd=Decimal("0.5"))
    corrida.assert_awaited_once_with(pipeline)


def test_un_rechazo_sale_con_1_sin_correr():
    cli = _modulo()
    rechazo = continuar.ContinuarRechazado(409, "estado_no_continuable", {"status": "completed", "mensaje": "m"})
    with patch("jacobs.continuar.continuar", AsyncMock(side_effect=rechazo)), \
         patch("jacobs.executor.run_pipeline", AsyncMock()) as corrida:
        rc = asyncio.run(cli.relanzar("p1", {}, None))
    assert rc == 1
    corrida.assert_not_awaited()


def test_un_error_inesperado_sale_con_2_sin_correr_y_redactado(capsys):
    """DB caída, faceta o credencial no disponible durante el análisis/pre-vuelo
    (desvío 20 del plan): no es un rechazo de `continuar()` (no hay
    ContinuarRechazado), así que sale distinto de 1 y no corre nada. El
    mensaje se imprime redactado (mismo criterio que
    jacobs/routes.py::_no_disponible)."""
    cli = _modulo()
    con_secreto = RuntimeError('conexión falló: api_key="sk-super-secreta-123"')
    with patch("jacobs.continuar.continuar", AsyncMock(side_effect=con_secreto)), \
         patch("jacobs.executor.run_pipeline", AsyncMock()) as corrida:
        rc = asyncio.run(cli.relanzar("p1", {}, None))
    assert rc == 2
    corrida.assert_not_awaited()
    salida = capsys.readouterr().out
    assert "sk-super-secreta-123" not in salida
    assert "RuntimeError" in salida


def test_parsear_reasignar_valido():
    assert _modulo().parsear_reasignar(["4=ada", " 5 = thot "]) == {"4": "ada", "5": "thot"}


def test_parsear_reasignar_invalido_lanza():
    cli = _modulo()
    for malo in ("ada", "x=ada", "4="):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parsear_reasignar([malo])


def test_no_escribe_pasos_ni_estado_por_su_cuenta():
    arbol = ast.parse(RUTA.read_text(encoding="utf-8"))
    llamados = {n.func.attr for n in ast.walk(arbol)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    prohibidos = {"step_upsert", "step_upsert_si_epoca", "pipeline_update_status",
                  "pipeline_update_status_si_epoca", "event_append", "continuar_transaccion"}
    assert not (llamados & prohibidos), llamados & prohibidos


def test_el_env_de_produccion_solo_se_lee_desde_una_funcion():
    arbol = ast.parse(RUTA.read_text(encoding="utf-8"))

    def permitido(nodo):
        if isinstance(nodo, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign)):
            return True
        if isinstance(nodo, ast.Expr) and isinstance(nodo.value, ast.Constant):
            return True  # docstring
        return (isinstance(nodo, ast.If) and isinstance(nodo.test, ast.Compare)
                and isinstance(nodo.test.left, ast.Name) and nodo.test.left.id == "__name__")

    sueltos = [ast.dump(n)[:60] for n in arbol.body if not permitido(n)]
    assert sueltos == []
