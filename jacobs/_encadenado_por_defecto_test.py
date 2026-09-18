import asyncio
import json

from jacobs.models import StepSpec
from jacobs.plan import PlanBuilder

GOBIERNO = {"capabilities": {"text_generation": {}}}


def _parsear(items):
    return asyncio.run(
        PlanBuilder._parse_plan_json(json.dumps(items), max_steps=10, governance=GOBIERNO)
    )


def test_el_plan_del_llm_sin_depends_on_se_encadena():
    """b8f80733: seis pasos con depends_on vacio arrancaron en el mismo
    instante (09:55:25.67) y ninguno leyo al anterior."""
    pasos = _parsear([
        {"facet": "jax_local", "capability": "text_generation", "prompt": "uno"},
        {"facet": "ada", "capability": "text_generation", "prompt": "dos"},
        {"facet": "jekyll", "capability": "text_generation", "prompt": "tres"},
    ])
    assert [p["depends_on"] for p in pasos] == [[], [0], [1]]


def test_el_paralelo_declarado_se_respeta():
    """Ausencia != lista vacia declarada. Un plan que PIDE paralelo lo tiene."""
    pasos = _parsear([
        {"facet": "jax_local", "capability": "text_generation", "prompt": "uno"},
        {"facet": "ada", "capability": "text_generation", "prompt": "dos",
         "depends_on": []},
    ])
    assert pasos[1]["depends_on"] == []


def test_el_default_de_pydantic_no_finge_una_decision_del_caller():
    """models.py:200-210 ya conto esta historia con timeout_seconds:
    model_dump() SIEMPRE incluye la clave, asi que un default de lista vacia
    se lee como 'el caller pidio paralelo' aunque nunca la haya tocado."""
    assert StepSpec(facet="ada", capability="text_generation").depends_on is None


def test_pasos_explicitos_sin_depends_on_se_encadenan():
    constructor = PlanBuilder.__new__(PlanBuilder)
    specs = [StepSpec(facet="ada", capability="text_generation", prompt="uno").model_dump(),
             StepSpec(facet="jekyll", capability="text_generation", prompt="dos").model_dump()]
    pasos = constructor._from_spec("p1", specs, {"text_generation": {}})
    assert [p.depends_on for p in pasos] == [[], [0]]
