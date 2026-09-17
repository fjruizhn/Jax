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


def test_un_error_inesperado_sale_con_4_sin_correr_y_redactado(capsys):
    """DB caída, faceta o credencial no disponible durante el análisis/pre-vuelo
    (desvío 20 del plan): no es un rechazo de `continuar()` (no hay
    ContinuarRechazado), así que sale distinto de 1 y no corre nada. El
    mensaje se imprime redactado (mismo criterio que
    jacobs/routes.py::_no_disponible). Código 4, no 2: 2 es el que usa
    argparse para sus propios errores de uso (fix round 1, hallazgo 1) --
    reusarlo mezclaría "invocación mal formada" con "el servicio falló"."""
    cli = _modulo()
    con_secreto = RuntimeError('conexión falló: api_key="sk-super-secreta-123"')
    with patch("jacobs.continuar.continuar", AsyncMock(side_effect=con_secreto)), \
         patch("jacobs.executor.run_pipeline", AsyncMock()) as corrida:
        rc = asyncio.run(cli.relanzar("p1", {}, None))
    assert rc == 4
    corrida.assert_not_awaited()
    salida = capsys.readouterr().out
    assert "sk-super-secreta-123" not in salida
    assert "RuntimeError" in salida


def test_error_de_uso_sale_2_y_error_de_servicio_sale_4_distintos():
    """argparse sale con 2 en sus propios errores (parser.error(), disparado
    acá por --reasignar mal formado); un error inesperado del servicio sale
    con 4 (test de arriba). Los dos tienen que quedar DISTINGUIBLES: antes
    del fix round 1 los dos eran 2 (fix round 1, hallazgo 1)."""
    cli = _modulo()
    with pytest.raises(SystemExit) as salida_uso:
        cli.main(["p1", "--reasignar", "bad"])
    assert salida_uso.value.code == 2

    con_falla = RuntimeError("boom")
    with patch("jacobs.continuar.continuar", AsyncMock(side_effect=con_falla)), \
         patch("jacobs.executor.run_pipeline", AsyncMock()) as corrida:
        rc = asyncio.run(cli.relanzar("p1", {}, None))
    assert rc == 4
    corrida.assert_not_awaited()
    assert rc != salida_uso.value.code


def test_parsear_reasignar_valido():
    assert _modulo().parsear_reasignar(["4=ada", " 5 = thot "]) == {"4": "ada", "5": "thot"}


def test_parsear_reasignar_invalido_lanza():
    cli = _modulo()
    for malo in ("ada", "x=ada", "4="):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parsear_reasignar([malo])


def test_no_escribe_pasos_ni_estado_por_su_cuenta():
    """Fix round 1, hallazgo 3: además de `store.step_upsert(...)` (ast.Attribute),
    también se marca una llamada SIN prefijo tras un `from jacobs.store import
    step_upsert` (ast.Name) -- la versión vieja de este guardia solo miraba
    Attribute y dejaba pasar esa forma sin verla."""
    arbol = ast.parse(RUTA.read_text(encoding="utf-8"))
    llamados = set()
    for n in ast.walk(arbol):
        if not isinstance(n, ast.Call):
            continue
        if isinstance(n.func, ast.Attribute):
            llamados.add(n.func.attr)
        elif isinstance(n.func, ast.Name):
            llamados.add(n.func.id)
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
