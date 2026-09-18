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
import json
from contextlib import asynccontextmanager
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
    """El que produce no arbitra. Es rechazo al crearse, no un aviso -- con
    OTRO productor en el plan, así que sí hay un 'también' (spec §3.6:
    'si el plan pone a Thot TAMBIÉN como productor'). Ver MEDIA 7 (revisión
    final 2026-09-18) y test_un_plan_de_un_solo_paso_con_la_faceta_arbitro_NO_se_rechaza
    más abajo -- con UN solo paso no hay 'también', y antes de ese arreglo
    este mismo caso con un solo step (thot, sin otro productor) también
    rechazaba, lo cual contradecía el spec."""
    with pytest.raises(PlanRejected) as exc:
        PlanBuilder._con_arbitro(
            [
                {"facet": "jekyll", "capability": "analysis", "prompt": "uno", "depends_on": []},
                {"facet": "thot", "capability": "text_generation", "prompt": "dos", "depends_on": [0]},
            ],
            facetas_activas=frozenset({"jekyll", "thot"}),
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


def test_un_plan_de_un_solo_paso_con_la_faceta_arbitro_NO_se_rechaza():
    """MEDIA 7 (revisión final 2026-09-18): la sala limpia corría ANTES del
    caso 'menos de 2 pasos', así que un plan de UN solo step con la faceta
    árbitro como su único productor se rechazaba igual -- aunque build() NO
    fuera a agregar ningún árbitro (len(specs) < 2 corta antes). El spec
    §3.6 dice 'si el plan pone a Thot TAMBIÉN como productor': con un solo
    paso no hay 'también' -- no hay nada que thot esté arbitrando además de
    producir. El chequeo de sala limpia ahora corre DESPUÉS del caso de un
    solo paso, así que este plan se devuelve TAL CUAL, sin árbitro agregado
    y sin rechazo."""
    pasos = PlanBuilder._con_arbitro(
        [{"facet": "thot", "capability": "text_generation", "prompt": "uno", "depends_on": []}],
        facetas_activas=frozenset({"thot"}),
        arbitro_faceta="thot",
    )
    assert len(pasos) == 1
    assert pasos[0]["facet"] == "thot"


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


def test_build_rechaza_cuando_el_arbitro_pasa_el_tope_de_20_pasos():
    """MEDIA 8 (revisión final 2026-09-18): routes.py rechaza un steps_spec
    de MÁS de 20 pasos ANTES de llamar a build() -- pero build() agrega el
    árbitro DESPUÉS, y nada revalida el conteo final. Un steps_spec de
    EXACTAMENTE 20 pasos explícitos (que pasa el chequeo de routes.py, que
    usa '> 20') termina persistiendo 21 -- el tope duro
    (MAX_STEPS_PER_PIPELINE) se pasa por uno. build() tiene que rechazar
    esto: es el único punto donde convergen los dos caminos (steps_spec y
    LLM) después de que el árbitro ya se agregó."""
    from jacobs import store
    from jacobs.models import MAX_STEPS_PER_PIPELINE
    from jacobs.plan import PlanRejected as _PlanRejected

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
    assert MAX_STEPS_PER_PIPELINE == 20, "el test asume el tope real; si cambió, ajustar N"

    steps_spec = [
        {"facet": "hipatia" if i % 2 == 0 else "jekyll", "capability": "research",
         "prompt": f"paso {i}", "depends_on": [i - 1] if i else []}
        for i in range(MAX_STEPS_PER_PIPELINE)
    ]

    async def correr():
        builder = PlanBuilder()
        original = store.get_motor_governance
        store.get_motor_governance = AsyncMock(return_value=governance)
        try:
            return await builder.build(
                pipeline_id="p-tope-20", objective="objetivo de prueba", steps_spec=steps_spec,
            )
        finally:
            store.get_motor_governance = original

    with pytest.raises(_PlanRejected) as exc:
        asyncio.run(correr())
    assert "21" in str(exc.value) and "20" in str(exc.value), str(exc.value)


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


# ---------------------------------------------------------------------------
# Ronda de arreglo 1 (2026-09-18): el prompt modular de Ada (_PLAN_SYSTEM_MODULAR,
# plan.py:~174) pedía un step FIJO thot/validate_consistency como antepenúltimo
# de cualquier plan formal -- thot como PRODUCTOR a mitad de plan. Con el
# árbitro de Task 4 activo, ese step choca con la sala limpia (thot no puede
# producir Y ser el árbitro configurado del mismo plan): CUALQUIER plan modular
# real de Ada se rechazaba siempre, no un caso raro -- es el camino principal
# del patrón compilador. Verificado antes del arreglo simulando el step (dict
# a mano) contra _con_arbitro -- insuficiente como test: no ejercitaba el
# camino real (_ada_plan -> HTTP -> _parse_plan_json -> _from_spec -> build()),
# que es exactamente lo que nadie corría. El test de acá sí lo ejercita.
# ---------------------------------------------------------------------------

def _governance_patron_modular(arbitro_faceta: str = "thot"):
    entry = {"allowed_motors": [], "max_execution_minutes": 15}
    return {
        "capabilities": {"design": entry, "reconcile": entry, "assemble": entry,
                          "critique": entry, "validate_consistency": entry},
        "motors": {},
        "facets": frozenset({"ada", arbitro_faceta}),
        "arbitro_faceta": arbitro_faceta,
    }


def _correr_ada_plan_real(monkeypatch, plan_json: str, *, arbitro_faceta: str = "thot",
                           capturados: list | None = None):
    """Ejercita el camino real de Ada: _ada_plan arma el prompt, pega al HTTP
    (mockeado), _parse_plan_json interpreta la respuesta, _from_spec/_con_arbitro
    deciden, y build() aplica los gates -- nada de eso se hand-wirea.

    `capturados`, si se pasa, recibe los payloads (`json=`) que _ada_plan
    mandó al `.stream()` mockeado -- para inspeccionar el TEXTO real que le
    llegó a Ada, no solo el resultado final del plan (Ronda de arreglo 2:
    demostrar que el prompt nombra la faceta árbitro CONFIGURADA, no un
    literal)."""
    from jacobs import store
    from jacobs import plan as plan_mod
    from facet_resolver import ResolvedFacet

    class _StreamResp:
        status_code = 200

        async def aiter_lines(self):
            yield "data: " + json.dumps({"choices": [{"delta": {"content": plan_json}}]})
            yield "data: [DONE]"

        async def aread(self):
            return b""

    class _Cliente:
        def stream(self, method, url, json=None, **kw):
            if capturados is not None:
                capturados.append(json)

            @asynccontextmanager
            async def _cm():
                yield _StreamResp()
            return _cm()

    governance = _governance_patron_modular(arbitro_faceta)

    async def correr():
        builder = plan_mod.PlanBuilder()
        original_gov = store.get_motor_governance
        store.get_motor_governance = AsyncMock(return_value=governance)
        monkeypatch.setattr(plan_mod, "resolve_facet", AsyncMock(return_value=ResolvedFacet(
            key="ada", provider_id="p", base_url="http://ada.test/v1", model="m",
            credential="c", transport="http_openai_compat", persona=None, params=None)))
        monkeypatch.setattr(plan_mod, "limite_de_salida", AsyncMock(return_value={"max_tokens": 100}))
        monkeypatch.setattr(plan_mod, "obtener_cliente_http", lambda: _Cliente())
        try:
            # objective > 200 caracteres -> _classify_difficulty = "formal" -> Ada.
            return await builder.build(
                pipeline_id="p-ada-modular", objective="x" * 250, max_steps=6,
            )
        finally:
            store.get_motor_governance = original_gov

    return asyncio.run(correr())


def test_un_plan_modular_de_ada_con_el_step_viejo_de_thot_se_autorrechaza(monkeypatch):
    """Prueba de raíz: la FORMA que el prompt VIEJO exigía (thot/validate_consistency
    a mitad de plan) se sigue rechazando -- la sala limpia no se ablandó, lo que
    cambió es qué le pedimos a Ada que genere. Si este test alguna vez empezara a
    pasar en verde sin querer, sería porque la sala limpia se rompió, no porque el
    prompt mejoró."""
    plan_json_viejo = json.dumps([
        {"facet": "ada", "capability": "design", "prompt": "tipos comunes", "depends_on": []},
        {"facet": "ada", "capability": "design", "prompt": "modulo x", "depends_on": [0]},
        {"facet": "thot", "capability": "validate_consistency", "prompt": "valida",
         "depends_on": [0, 1]},
        {"facet": "ada", "capability": "reconcile", "prompt": "aplica parches",
         "depends_on": [2]},
        {"facet": "ada", "capability": "assemble", "prompt": "manifest",
         "depends_on": [0, 1, 2, 3]},
    ])
    with pytest.raises(PlanRejected) as exc:
        _correr_ada_plan_real(monkeypatch, plan_json_viejo)
    assert "sala limpia" in str(exc.value).lower() or "arbitra" in str(exc.value).lower()


def test_un_plan_modular_de_ada_con_el_prompt_actual_no_se_autorrechaza(monkeypatch):
    """LA regresión de la ronda de arreglo 1: la forma que el prompt ACTUAL le
    pide a Ada (sin thot/validate_consistency -- ver _PLAN_SYSTEM_MODULAR y el
    prompt de _ada_plan) llega a build() por el camino real y NO se rechaza.
    Antes de este arreglo, la única forma "realista" de un plan modular era la
    del prompt viejo (con thot) -- y esa SIEMPRE se rechazaba (test de arriba).
    """
    plan_json_actual = json.dumps([
        {"facet": "ada", "capability": "design", "prompt": "tipos comunes", "depends_on": []},
        {"facet": "ada", "capability": "design", "prompt": "modulo x", "depends_on": [0]},
        {"facet": "ada", "capability": "reconcile", "prompt": "revisa consistencia y aplica parches",
         "depends_on": [0, 1]},
        {"facet": "ada", "capability": "assemble", "prompt": "manifest", "depends_on": [0, 1, 2]},
    ])
    steps = _correr_ada_plan_real(monkeypatch, plan_json_actual)
    assert [s.facet for s in steps] == ["ada", "ada", "ada", "ada", "thot"]
    assert steps[-1].capability == PlanBuilder.CAPABILITY_ARBITRO
    assert steps[-1].depends_on == [0, 1, 2, 3]
    assert "thot" not in [s.facet for s in steps[:-1]], (
        "el patrón modular de Ada no debe producir ningun step con facet 'thot' -- "
        "ese facet lo reserva el árbitro"
    )


def test_el_prompt_modular_ya_no_pide_thot_como_productor():
    """Asserción directa sobre el texto del prompt (no solo el comportamiento):
    con la faceta árbitro de HOY ('thot'), ni la regla del sistema ni el resto
    del texto de _ada_plan deben mencionar a 'thot' como facet de un step, ni
    al antiguo step fijo de validación de consistencia -- confirma que el
    texto es consistente consigo mismo después de sacar ese paso (no quedó un
    '4. El ANTEPENÚLTIMO...' colgado)."""
    from jacobs.plan import _texto_plan_system_modular

    texto = _texto_plan_system_modular("thot")

    assert "validate_consistency" not in texto
    assert "ANTEPENÚLTIMO" not in texto
    assert "PENÚLTIMO" in texto and "ÚLTIMO" in texto
    # 'thot' SÍ aparece -- pero solo en la prohibición ("NUNCA 'thot'"), no
    # como facet de un step productor: no hay ejemplo JSON en este texto (el
    # ejemplo vive en el f-string de _ada_plan, cubierto por el test de abajo).
    assert "'thot'" in texto


# ---------------------------------------------------------------------------
# Ronda de arreglo 2 (2026-09-18): _con_arbitro ya leía governance["arbitro_faceta"]
# (dinámico, Ruling 2), pero el TEXTO que le llegaba a Ada seguía diciendo
# 'thot' como literal en _CLEANROOM_RULE, _PLAN_SYSTEM_MODULAR y el f-string
# de _ada_plan -- si ejecutor.auditor_faceta cambiara de valor, la LÓGICA
# exigiría la faceta nueva pero el PROMPT le seguiría prohibiendo a Ada la
# vieja (ya irrelevante) y nunca mencionaría la nueva: Ada volvería a
# producir la faceta árbitro real como productora, y todo plan modular real
# se autorrechazaría otra vez -- el mismo defecto de la ronda 1, reaparecido
# bajo otro nombre la primera vez que alguien toque la config.
#
# El test de abajo NO mira solo que el literal 'thot' desapareciera (eso lo
# probaría incluso si el código simplemente lo hubiera borrado sin reemplazo,
# dejando el prompt mudo sobre qué faceta evitar) -- ejercita el camino real
# con la faceta árbitro configurada a 'hipatia' (algo DISTINTO de 'thot') y
# lee el payload HTTP que _ada_plan realmente mandó, verificando que NOMBRA
# 'hipatia' como la faceta prohibida y que 'thot' no aparece en ningún lado.
# ---------------------------------------------------------------------------

def test_el_prompt_que_recibe_ada_nombra_la_faceta_arbitro_configurada(monkeypatch):
    """Con arbitro_faceta='hipatia' (no 'thot'), el prompt real que _ada_plan
    manda por HTTP prohíbe 'hipatia' -- no 'thot'. Ejercita _ada_plan de punta
    a punta (HTTP mockeado únicamente), no un string armado a mano."""
    plan_json = json.dumps([
        {"facet": "ada", "capability": "design", "prompt": "tipos comunes", "depends_on": []},
        {"facet": "ada", "capability": "design", "prompt": "modulo x", "depends_on": [0]},
        {"facet": "ada", "capability": "reconcile", "prompt": "revisa consistencia y aplica parches",
         "depends_on": [0, 1]},
        {"facet": "ada", "capability": "assemble", "prompt": "manifest", "depends_on": [0, 1, 2]},
    ])
    # 'zeta', no 'hipatia': una faceta que NO está en _MENU_DE_FACETAS, para
    # que "aparece en el texto" no pueda deberse a que además se ofrece en el
    # menú general de facetas (eso pasaría con cualquier faceta real activa,
    # y volvería ambigua la aserción -- con 'zeta' el único lugar posible
    # donde puede aparecer es la prohibición que arma el código).
    capturados: list = []
    steps = _correr_ada_plan_real(
        monkeypatch, plan_json, arbitro_faceta="zeta", capturados=capturados,
    )

    # El plan construido termina en la faceta árbitro CONFIGURADA, no en thot.
    assert steps[-1].facet == "zeta"
    assert steps[-1].capability == PlanBuilder.CAPABILITY_ARBITRO

    # El TEXTO que salió por HTTP -- lo que Ada de verdad recibió -- nombra
    # 'zeta' como la faceta reservada, y 'thot' no aparece en ningún lado (ni
    # en el system prompt ni en el prompt de usuario, los dos mensajes que
    # arma _ada_plan).
    assert len(capturados) == 1, "se esperaba una sola llamada HTTP a Ada"
    mensajes = capturados[0]["messages"]
    texto_completo = "\n".join(m["content"] for m in mensajes)
    assert "zeta" in texto_completo
    assert "thot" not in texto_completo.lower()
    # Y el motivo concreto: la prohibición explícita nombra a la faceta real,
    # entre comillas, como todas las menciones de facet en este prompt.
    assert "'zeta'" in texto_completo
