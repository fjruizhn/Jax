"""Task 4 (2026-09-18) — El paso árbitro.

El informe encontró cinco hojas de ruta que no convergían (ef9b2d6e / b8f80733):
nadie decidía al final. El último paso de todo plan de 2+ pasos pasa a ser un
ÁRBITRO que recibe todo y produce una decisión citando el paso que la sostiene.

Dos rulings de Fernando por encima del brief original
(.superpowers/sdd/2026-09-18-historial-y-arreglos-de-pipeline/task-4-brief.md):

Ruling 1 — el árbitro se agrega en LOS DOS caminos de construcción del plan
(steps_spec explícito y el plan del LLM/fallback), no solo el que el brief
mostraba (`_parse_plan_json` -> list[dict]). Los dos caminos convergen en
`PlanBuilder._from_spec` (ver plan.py: build() lo llama directo para
steps_spec, y _from_objective() lo llama al final con los specs del LLM o del
plan de respaldo) -- ahí es donde se engancha `_con_arbitro`, UNA sola vez,
sobre la representación list[dict] que _from_spec ya recibe, cubriendo los
dos caminos sin duplicar la construcción de Step.

Ruling 2 — si la faceta árbitro no está activa (o no hay ninguna configurada),
el plan se RECHAZA con un motivo que empieza con "arbitro_no_disponible" (no
un rechazo genérico de _check_facets, que confundiría "el spec del caller
pidió una faceta inválida" con "el sistema no tiene quién arbitre"). Fail
closed: un pipeline sin quien decida es el defecto que este trabajo cierra.

La faceta árbitro sale de axioma_config (config_key='ejecutor.auditor_faceta',
verificado en la base de test: valor 'thot', mismo config que ya usa
jax/ejecutor/contratos/eleccion_c5.py para su propio auditor) -- no se
hardcodea acá. `PlanBuilder._con_arbitro` la recibe como parámetro
(`arbitro_faceta`), resuelta por build()/_from_objective() desde
`governance["arbitro_faceta"]` (agregado a store.get_motor_governance(), MISMA
foto que facets/capabilities/motors -- Ruling R43: una lectura de gobernanza
por build()).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from jacobs.models import Step
from jacobs.plan import PlanBuilder, PlanRejected


# ---------------------------------------------------------------------------
# _con_arbitro: la unidad de decisión, sobre list[dict] (la representación
# que _from_spec ya recibe de LOS DOS caminos).
# ---------------------------------------------------------------------------

def test_el_plan_termina_en_un_arbitro_configurado():
    pasos = PlanBuilder._con_arbitro(
        [
            {"facet": "jax_local", "capability": "text_generation", "prompt": "uno", "depends_on": []},
            {"facet": "ada", "capability": "text_generation", "prompt": "dos", "depends_on": [0]},
        ],
        facetas_activas=frozenset({"jax_local", "ada", "thot"}),
        arbitro_faceta="thot",
    )
    assert len(pasos) == 3
    assert pasos[-1]["facet"] == "thot"
    assert pasos[-1]["depends_on"] == [0, 1]
    assert pasos[-1]["capability"] == PlanBuilder.CAPABILITY_ARBITRO
    assert "cita" in pasos[-1]["prompt"].lower()


def test_thot_productor_rechaza_el_plan():
    """El que produce no arbitra. Es rechazo al crearse, no un aviso -- ni
    siquiera hace falta un segundo paso para que dispare (brief Step 1)."""
    with pytest.raises(PlanRejected) as exc:
        PlanBuilder._con_arbitro(
            [{"facet": "thot", "capability": "text_generation", "prompt": "uno", "depends_on": []}],
            facetas_activas=frozenset({"thot"}),
            arbitro_faceta="thot",
        )
    msg = str(exc.value).lower()
    assert "arbitro_no_disponible" not in msg, "es el rechazo de sala-limpia, no de disponibilidad"
    assert "arbitra" in msg or "sala limpia" in msg or "sala-limpia" in msg


def test_un_plan_de_un_solo_paso_no_gana_arbitro():
    """Con un solo productor no hay nada que arbitrar -- ni se consulta si el
    árbitro está disponible: no hace falta."""
    pasos = PlanBuilder._con_arbitro(
        [{"facet": "jax_local", "capability": "text_generation", "prompt": "uno", "depends_on": []}],
        facetas_activas=frozenset(),  # thot NI SIQUIERA está en el universo -- no importa
        arbitro_faceta=None,
    )
    assert len(pasos) == 1


def test_arbitro_no_disponible_si_la_faceta_configurada_no_esta_activa():
    """Ruling 2: thot configurado pero caído/inactivo en la tabla `facet` ->
    RECHAZO fail-closed, código propio, no un plan sin árbitro."""
    with pytest.raises(PlanRejected) as exc:
        PlanBuilder._con_arbitro(
            [
                {"facet": "jax_local", "capability": "reason", "prompt": "uno", "depends_on": []},
                {"facet": "ada", "capability": "design", "prompt": "dos", "depends_on": [0]},
            ],
            facetas_activas=frozenset({"jax_local", "ada"}),  # thot NO está acá
            arbitro_faceta="thot",
        )
    assert "arbitro_no_disponible" in str(exc.value).lower()


def test_arbitro_no_disponible_sin_configuracion():
    """Ruling 2, fail-closed: sin fila en axioma_config (o vacía), tampoco se
    entrega un plan sin árbitro."""
    with pytest.raises(PlanRejected) as exc:
        PlanBuilder._con_arbitro(
            [
                {"facet": "jax_local", "capability": "reason", "prompt": "uno", "depends_on": []},
                {"facet": "ada", "capability": "design", "prompt": "dos", "depends_on": [0]},
            ],
            facetas_activas=frozenset({"jax_local", "ada", "thot"}),
            arbitro_faceta=None,
        )
    assert "arbitro_no_disponible" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Ruling 1: LOS DOS CAMINOS de construcción llegan al árbitro.
#
# _from_spec produce list[Step] (no list[dict]) -- acá se prueba con la
# representación que corresponde a CADA camino: steps_spec (dicts que YA
# trae el caller, tal cual routes.py los arma) y specs del LLM/fallback
# (dicts que ya devolvió _parse_plan_json/_fallback_plan). Ambos pasan por
# _from_spec, que es donde vive el enganche -- por eso alcanza con probar
# _from_spec directamente para los dos casos, más un smoke test de build()
# end-to-end con steps_spec para confirmar que los gates existentes
# (_check_facets/_check_cleanroom/_validate_plan_capabilities) corren
# DESPUÉS y no lo saltean.
# ---------------------------------------------------------------------------

def test_from_spec_agrega_el_arbitro_como_step_real():
    """Camino A (steps_spec explícito, ya dict): _from_spec devuelve
    list[Step], no list[dict] -- FACETA_ARBITRO tiene que llegar convertida,
    con su propio step_index y timeout por capability."""
    builder = PlanBuilder.__new__(PlanBuilder)
    steps = builder._from_spec(
        "p-camino-a",
        [
            {"facet": "hipatia", "capability": "research", "prompt": "x"},
            {"facet": "jekyll", "capability": "research", "prompt": "y", "depends_on": [0]},
        ],
        {"research": {"max_execution_minutes": 5}, "critique": {"max_execution_minutes": 15}},
        facetas_activas=frozenset({"hipatia", "jekyll", "thot"}),
        arbitro_faceta="thot",
    )
    assert [s.facet for s in steps] == ["hipatia", "jekyll", "thot"]
    assert isinstance(steps[-1], Step)
    assert steps[-1].depends_on == [0, 1]
    assert steps[-1].capability == PlanBuilder.CAPABILITY_ARBITRO
    assert steps[-1].timeout_seconds == 900  # critique: 15 min, de `caps`


def test_from_spec_sin_facetas_activas_no_agrega_arbitro():
    """Retrocompatibilidad deliberada: los callers de bajo nivel que no pasan
    `facetas_activas` (tests existentes de timeout/encadenado) no ganan un
    árbitro que no pidieron -- solo build()/_from_objective() lo piden,
    pasando la gobernanza real."""
    builder = PlanBuilder.__new__(PlanBuilder)
    steps = builder._from_spec(
        "p-sin-gobernanza",
        [
            {"facet": "hipatia", "capability": "research", "prompt": "x"},
            {"facet": "jekyll", "capability": "research", "prompt": "y", "depends_on": [0]},
        ],
        {"research": {"max_execution_minutes": 5}},
    )
    assert [s.facet for s in steps] == ["hipatia", "jekyll"]


def test_build_agrega_el_arbitro_por_el_camino_de_steps_spec():
    """Camino A end-to-end: build() con steps_spec explícito. El árbitro pasa
    por los gates reales (_check_facets etc.) -- si algo en su construcción
    estuviera mal (faceta inactiva, capability inexistente), build() lo
    rechazaría igual que a cualquier otro step."""
    from jacobs import store

    governance = {
        "capabilities": {
            "research": {
                "allowed_motors": [], "allowed_callers": ["jacobs"], "risk_level": "low",
                "sandbox_only": True, "requires_human_gate": False, "max_execution_minutes": 5,
                "max_recursion_depth": 0, "output_schema": "", "fallback_motor": None,
                "fallback_mode": "manual_only", "forbidden_paths": [], "auditor_motor": None,
            },
            "critique": {
                "allowed_motors": [], "allowed_callers": ["jacobs", "hyde", "thot"], "risk_level": "low",
                "sandbox_only": True, "requires_human_gate": False, "max_execution_minutes": 15,
                "max_recursion_depth": 0, "output_schema": "", "fallback_motor": None,
                "fallback_mode": "manual_only", "forbidden_paths": [], "auditor_motor": None,
            },
        },
        "motors": {},
        "facets": frozenset({"hipatia", "jekyll", "thot"}),
        "arbitro_faceta": "thot",
    }

    async def correr():
        builder = PlanBuilder()
        original = store.get_motor_governance
        store.get_motor_governance = AsyncMock(return_value=governance)
        try:
            return await builder.build(
                pipeline_id="p-build-steps-spec",
                objective="objetivo de prueba",
                steps_spec=[
                    {"facet": "hipatia", "capability": "research", "prompt": "x"},
                    {"facet": "jekyll", "capability": "research", "prompt": "y", "depends_on": [0]},
                ],
            )
        finally:
            store.get_motor_governance = original

    steps = asyncio.run(correr())
    assert [s.facet for s in steps] == ["hipatia", "jekyll", "thot"]
    assert steps[-1].capability == PlanBuilder.CAPABILITY_ARBITRO


def test_build_agrega_el_arbitro_por_el_camino_del_llm():
    """Camino B end-to-end: build() con el plan que arma el cerebro (LLM),
    sin steps_spec. Es el camino que el brief SÍ mostraba -- confirmado acá
    con el mismo gate real de build(), no solo con el helper interno."""
    from jacobs import store

    governance = {
        "capabilities": {
            "research": {
                "allowed_motors": [], "allowed_callers": ["jacobs"], "risk_level": "low",
                "sandbox_only": True, "requires_human_gate": False, "max_execution_minutes": 5,
                "max_recursion_depth": 0, "output_schema": "", "fallback_motor": None,
                "fallback_mode": "manual_only", "forbidden_paths": [], "auditor_motor": None,
            },
            "critique": {
                "allowed_motors": [], "allowed_callers": ["jacobs", "hyde", "thot"], "risk_level": "low",
                "sandbox_only": True, "requires_human_gate": False, "max_execution_minutes": 15,
                "max_recursion_depth": 0, "output_schema": "", "fallback_motor": None,
                "fallback_mode": "manual_only", "forbidden_paths": [], "auditor_motor": None,
            },
        },
        "motors": {},
        "facets": frozenset({"hipatia", "jekyll", "thot", "jax_local"}),
        "arbitro_faceta": "thot",
    }

    async def cerebro_falso(objective, max_steps, capability_hint, *, facetas_activas, governance=None):
        return [
            {"facet": "hipatia", "capability": "research", "prompt": "investigá x"},
            {"facet": "jekyll", "capability": "research", "prompt": "analizá y", "depends_on": [0]},
        ]

    async def correr():
        builder = PlanBuilder()
        builder._llm_plan = cerebro_falso
        builder._ada_plan = cerebro_falso
        original = store.get_motor_governance
        store.get_motor_governance = AsyncMock(return_value=governance)
        try:
            return await builder.build(
                pipeline_id="p-build-llm", objective="algo trivial", max_steps=3,
            )
        finally:
            store.get_motor_governance = original

    steps = asyncio.run(correr())
    assert [s.facet for s in steps] == ["hipatia", "jekyll", "thot"]
    assert steps[-1].depends_on == [0, 1]
