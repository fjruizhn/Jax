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

from base_de_test import fijar_base_de_test  # noqa: E402

fijar_base_de_test()

from jacobs import models, routes  # noqa: E402
from jacobs import plan as plan_mod  # noqa: E402

GOBERNANZA = {
    "capabilities": {
        "research": {"allowed_motors": [], "max_execution_minutes": 5},
        "analysis": {"allowed_motors": ["kimi"], "max_execution_minutes": 5},
        # Task 4 (2026-09-18): capability real del árbitro (PlanBuilder.CAPABILITY_ARBITRO).
        "critique": {"allowed_motors": [], "max_execution_minutes": 15},
    },
    "motors": {"kimi": True, "jax_local": True},
    # "thot" activo: sin él, todo plan de 2+ pasos de este archivo se
    # rechazaría por arbitro_no_disponible (Ruling 2, Task 4) -- salvo en
    # los tests que apagan "thot" a propósito para probar exactamente eso.
    "facets": frozenset({"hipatia", "jekyll", "kimi", "jax_local", "thot"}),
    # Mismo config real que store.get_motor_governance() lee de
    # axioma_config.ejecutor.auditor_faceta (verificado: 'thot').
    "arbitro_faceta": "thot",
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
    async def llm(objective, max_steps, capability_hint, *, facetas_activas, governance=None):
        return await plan_mod.PlanBuilder._parse_plan_json(
            '[{"facet": "inventada", "capability": "research", "prompt": "x"}]', max_steps)
    with pytest.raises(plan_mod.PlanRejected) as e:
        _build(llm=llm)
    assert [v.facet for v in e.value.violations] == ["inventada"]


def test_build_acepta_las_facetas_activas_de_la_tabla():
    """Task 4: un plan de 2+ pasos termina en el árbitro (thot, activo y
    configurado en GOBERNANZA) -- el 3er step no lo pidió el caller, lo
    agrega build()."""
    steps = _build(steps_spec=[
        {"facet": "hipatia", "capability": "research", "prompt": "x"},
        {"facet": "jekyll", "capability": "research", "prompt": "y", "depends_on": [0]},
    ])
    assert [s.facet for s in steps] == ["hipatia", "jekyll", "thot"]
    assert steps[-1].depends_on == [0, 1]


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


# ---------------------------------------------------------------------------
# Revisión final del frente E: el MENÚ de facetas que el planner le ofrece al
# LLM era una lista fija (hipatia, jekyll, thot, ada, kimi, hyde). Con E-17 una
# faceta que no está activa rechaza el plan con 422, así que ofrecerla es
# fabricar planes que se van a rechazar. El menú sale de governance["facets"].
# ---------------------------------------------------------------------------

import re  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from unittest.mock import patch  # noqa: E402

from facet_resolver import ResolvedFacet  # noqa: E402


class _ClienteQueCaptura:
    def __init__(self, plan_json: str):
        self.posts: list[dict] = []
        self.streams: list[dict] = []
        self._plan_json = plan_json

    async def post(self, url, json=None, **kw):
        self.posts.append(json)
        plan_json = self._plan_json

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return {"model": "q", "message": {"content": plan_json}}
        return _Resp()

    def stream(self, method, url, json=None, **kw):
        self.streams.append(json)
        plan_json = self._plan_json

        @asynccontextmanager
        async def _cm():
            class _Resp:
                status_code = 200

                async def aiter_lines(self):
                    chunk = {"choices": [{"delta": {"content": plan_json}}]}
                    yield "data: " + __import__("json").dumps(chunk)
                    yield "data: [DONE]"
            yield _Resp()
        return _cm()


def _facet(key, transport):
    return ResolvedFacet(key=key, provider_id="p", base_url="http://x.test/v1", model="m",
                         credential="c", transport=transport, persona=None, params=None)


def _gobernanza_con(facetas):
    return {**GOBERNANZA, "facets": frozenset(facetas)}


def _correr_con_cerebros(monkeypatch, facetas, objective, plan_json):
    from jacobs import store
    cliente = _ClienteQueCaptura(plan_json)
    resolver = AsyncMock(side_effect=lambda key: _facet(key, "ollama" if key == "jax_local" else "http_openai_compat"))
    evento = AsyncMock()
    monkeypatch.setattr(store, "get_motor_governance", AsyncMock(return_value=_gobernanza_con(facetas)))
    monkeypatch.setattr(store, "event_append", evento)
    monkeypatch.setattr(plan_mod, "resolve_facet", resolver)
    monkeypatch.setattr(plan_mod, "limite_de_salida",
                        AsyncMock(return_value={"options": {"num_predict": 100}, "max_tokens": 100}))
    monkeypatch.setattr(plan_mod, "record_resolved_version_safe", AsyncMock())
    monkeypatch.setattr(plan_mod, "obtener_cliente_http", lambda: cliente)

    async def correr():
        return await plan_mod.PlanBuilder().build(pipeline_id="p-menu", objective=objective, max_steps=3)
    return cliente, resolver, evento, correr


def _mensaje_de_usuario(payload):
    return next(m["content"] for m in payload["messages"] if m["role"] == "user")


def _nombra(texto, faceta):
    return re.search(rf"\b{faceta}\b", texto) is not None


def test_el_menu_de_qwen_solo_ofrece_las_facetas_activas(monkeypatch):
    cliente, _, _, correr = _correr_con_cerebros(
        monkeypatch, {"hipatia", "jekyll", "kimi", "jax_local"}, "algo trivial",
        '[{"facet": "hipatia", "capability": "research", "prompt": "x"}]')
    asyncio.run(correr())
    [payload] = cliente.posts
    texto = _mensaje_de_usuario(payload)
    for inactiva in ("thot", "ada", "hyde"):
        assert not _nombra(texto, inactiva), f"qwen ve '{inactiva}' en el menú sin estar activa:\n{texto}"
    for activa in ("hipatia", "jekyll", "kimi"):
        assert _nombra(texto, activa), f"falta '{activa}' en el menú:\n{texto}"


def test_el_menu_de_ada_solo_ofrece_las_facetas_activas(monkeypatch):
    cliente, _, _, correr = _correr_con_cerebros(
        monkeypatch, {"ada", "thot", "kimi", "jax_local"}, "x" * 250,
        '[{"facet": "kimi", "capability": "analysis", "prompt": "x"}]')
    asyncio.run(correr())
    [payload] = cliente.streams
    texto = _mensaje_de_usuario(payload) + payload["messages"][0]["content"]
    for inactiva in ("hipatia", "jekyll", "hyde"):
        assert not _nombra(texto, inactiva), f"Ada ve '{inactiva}' sin estar activa"
    assert _nombra(texto, "kimi")


def test_ada_no_se_consulta_si_el_patron_modular_pide_una_faceta_inactiva(monkeypatch):
    """El patrón compilador de Ada exige thot (validate_consistency) y ada
    (reconcile/assemble). Sin thot activa, el plan de Ada se rechazaría con
    certeza: no se gasta la llamada paga y se cae a qwen con el motivo."""
    cliente, resolver, evento, correr = _correr_con_cerebros(
        monkeypatch, {"ada", "kimi", "jax_local"}, "x" * 250,
        '[{"facet": "kimi", "capability": "analysis", "prompt": "x"}]')
    asyncio.run(correr())
    assert cliente.streams == [], "Ada se consultó con thot inactiva"
    assert "ada" not in [c.args[0] for c in resolver.await_args_list]
    pipeline_id, tipo, payload = evento.await_args_list[0].args
    assert (tipo, payload["de"], payload["a"]) == ("PLAN_CEREBRO_FALLBACK", "ada", "qwen")
    assert "thot" in payload["motivo"]


def test_qwen_no_se_consulta_sin_ninguna_faceta_del_menu_activa(monkeypatch):
    cliente, _, _, correr = _correr_con_cerebros(monkeypatch, {"jax_local"}, "algo trivial", "[]")
    with pytest.raises(plan_mod.PlanRejected):
        asyncio.run(correr())
    assert cliente.posts == [], "qwen se consultó con un menú vacío"


def test_el_plan_de_respaldo_con_una_faceta_inactiva_se_rechaza_nombrandola(monkeypatch):
    """El plan fijo (hipatia -> jekyll, Task 4: + thot como árbitro agregado
    por build()) no se reescribe: sin thot activa, build() lo rechaza con
    PlanRejected nombrandola -- antes por _check_facets (el 3er step fijo
    del plan), ahora por _con_arbitro (arbitro_no_disponible, porque el 3er
    step ya no es fijo: build() lo agrega y necesita que thot esté activa
    para poder agregarlo). Mismo facet en la violación, antes de persistir
    nada (422 en routes)."""
    _, _, _, correr = _correr_con_cerebros(
        monkeypatch, {"hipatia", "jekyll", "kimi", "jax_local"}, "algo trivial", "no es json")
    with pytest.raises(plan_mod.PlanRejected) as e:
        asyncio.run(correr())
    assert [v.facet for v in e.value.violations] == ["thot"]
    assert "arbitro_no_disponible" in e.value.violations[0].reason
