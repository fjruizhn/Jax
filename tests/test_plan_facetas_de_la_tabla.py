"""E-03 / E-17 / E-23 (2026-09-16): las facetas del planner salen de la tabla
`facet`, y una que no está activa RECHAZA el plan.

Antes jacobs/plan.py:799 reemplazaba una faceta desconocida del LLM por
jax_local sin log ni evento, contra una lista fija (VALID_FACETS, duplicada en
models.py). El plan corría con una faceta que nadie pidió. Ahora el rechazo es
PlanRejected -> 422 + PLAN_REJECTED en jacobs_events, por los dos caminos
(spec y LLM), porque la validación vive en build().

La gobernanza se parchea: se prueba la POLÍTICA. La lectura real de la tabla
la prueba tests/test_facetas_de_gobernanza_db.py contra MariaDB.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock

import pytest

os.environ["JAX_DB_NAME"] = "jax_memory_test"

from jacobs import models, routes  # noqa: E402
from jacobs import plan as plan_mod  # noqa: E402

GOBERNANZA = {
    "capabilities": {
        "research": {"allowed_motors": [], "max_execution_minutes": 5},
        "analysis": {"allowed_motors": ["kimi"], "max_execution_minutes": 5},
    },
    "motors": {"kimi": True, "jax_local": True},
    "facets": frozenset({"hipatia", "jekyll", "kimi", "jax_local"}),
}


@pytest.fixture(autouse=True)
def gobernanza(monkeypatch):
    from jacobs import store
    monkeypatch.setattr(store, "get_motor_governance", AsyncMock(return_value=GOBERNANZA))


def _build(steps_spec=None, llm=None):
    async def correr():
        b = plan_mod.PlanBuilder()
        if llm is not None:
            b._llm_plan = llm
            b._ada_plan = llm
        return await b.build(pipeline_id="p-facetas", objective="algo trivial", max_steps=3, steps_spec=steps_spec)
    return asyncio.run(correr())


def test_valid_facets_ya_no_existe_en_ningun_modulo():
    assert not hasattr(plan_mod, "VALID_FACETS")
    assert not hasattr(models, "VALID_FACETS")


def test_parse_plan_json_no_cambia_una_faceta_desconocida():
    specs = asyncio.run(plan_mod.PlanBuilder._parse_plan_json(
        '[{"facet": "inventada", "capability": "research", "prompt": "x"}]', 3))
    assert specs[0]["facet"] == "inventada"


def test_build_rechaza_una_faceta_del_spec_que_no_esta_en_la_tabla():
    with pytest.raises(plan_mod.PlanRejected) as e:
        _build(steps_spec=[{"facet": "inventada", "capability": "research", "prompt": "x"}])
    [violacion] = e.value.violations
    assert violacion.facet == "inventada"
    assert "tabla `facet`" in violacion.reason


def test_build_rechaza_la_faceta_inventada_por_el_llm():
    async def llm(objective, max_steps, capability_hint):
        return await plan_mod.PlanBuilder._parse_plan_json(
            '[{"facet": "inventada", "capability": "research", "prompt": "x"}]', max_steps)
    with pytest.raises(plan_mod.PlanRejected) as e:
        _build(llm=llm)
    assert [v.facet for v in e.value.violations] == ["inventada"]


def test_build_acepta_las_facetas_activas_de_la_tabla():
    steps = _build(steps_spec=[
        {"facet": "hipatia", "capability": "research", "prompt": "x"},
        {"facet": "jekyll", "capability": "research", "prompt": "y", "depends_on": [0]},
    ])
    assert [s.facet for s in steps] == ["hipatia", "jekyll"]


def test_un_spec_sin_faceta_se_rechaza_en_vez_de_caer_a_jax_local():
    with pytest.raises(plan_mod.PlanRejected) as e:
        _build(steps_spec=[{"capability": "research", "prompt": "x"}])
    assert e.value.violations[0].facet == ""


def test_el_rechazo_sale_como_422_con_evento_PLAN_REJECTED(monkeypatch):
    from fastapi import HTTPException
    from jacobs import store
    evento = AsyncMock()
    monkeypatch.setattr(store, "event_append", evento)
    with pytest.raises(HTTPException) as e:
        asyncio.run(routes._build_plan_or_reject(
            "p-422", "o", 3, [{"facet": "inventada", "capability": "research", "prompt": "x"}]))
    assert e.value.status_code == 422
    pipeline_id, tipo, payload = evento.await_args.args
    assert (pipeline_id, tipo) == ("p-422", "PLAN_REJECTED")
    assert payload["violations"][0]["facet"] == "inventada"
