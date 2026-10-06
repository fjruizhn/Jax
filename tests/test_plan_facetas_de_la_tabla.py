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
    for inactiva in ("thot", "ada", "hyde", "kimi", "jax_local"):
        assert not _nombra(texto, inactiva), f"qwen ve '{inactiva}' no ejecutable en el menú:\n{texto}"
    for activa in ("hipatia", "jekyll"):
        assert _nombra(texto, activa), f"falta '{activa}' en el menú:\n{texto}"


def test_el_menu_de_ada_solo_ofrece_las_facetas_activas(monkeypatch):
    cliente, _, _, correr = _correr_con_cerebros(
        monkeypatch, {"ada", "thot", "kimi", "jax_local"}, "x" * 250,
        '[{"facet": "ada", "capability": "implementation", "prompt": "x"}]')
    asyncio.run(correr())
    [payload] = cliente.streams
    texto = _mensaje_de_usuario(payload) + payload["messages"][0]["content"]
    menu = texto.split("Facetas disponibles:", 1)[1].split(".\\n", 1)[0]
    for no_disponible in ("kimi", "jax_local", "hyde"):
        assert not _nombra(menu, no_disponible), f"Ada ofrece '{no_disponible}'"
    assert _nombra(menu, "ada")
    from jacobs.plan import _menu_de_facetas
    menu = _menu_de_facetas(frozenset({"ada", "thot", "kimi", "jax_local", "hyde"}), "thot")
    assert {fila[0] for fila in menu} == {"ada"}


def test_plan_y_preflight_rechazan_facetas_sin_dispatch_gobernado(monkeypatch):
    from fastapi import HTTPException
    from jacobs import store
    from jacobs.models import StepSpec

    monkeypatch.setattr(routes, "check_kill_switch", lambda: False)
    evento = AsyncMock()
    monkeypatch.setattr(store, "event_append", evento)
    sin_prevuelo = AsyncMock()
    monkeypatch.setattr(routes, "_prevuelo_o_503", sin_prevuelo)

    for facet in ("kimi", "jax_local", "hyde"):
        with pytest.raises(HTTPException) as rechazo_plan:
            asyncio.run(routes.plan_only(routes.PlanRequest(
                name="prueba", objective="x", invoked_by="plataforma", mode="dry_run",
                steps=[StepSpec(facet=facet, capability="analysis", prompt="x")],
            )))
        assert rechazo_plan.value.status_code == 422
        assert rechazo_plan.value.detail["code"] == "motor_gobernado_no_disponible"

        with pytest.raises(HTTPException) as rechazo_prevuelo:
            asyncio.run(routes.preflight(routes.PreflightRequest(
                invoked_by="plataforma", objective="x",
                steps=[StepSpec(facet=facet, capability="analysis", prompt="x")],
            )))
        assert rechazo_prevuelo.value.status_code == 422
        assert rechazo_prevuelo.value.detail["code"] == "motor_gobernado_no_disponible"
    sin_prevuelo.assert_not_awaited()


def test_ada_no_se_consulta_si_el_patron_modular_pide_una_faceta_inactiva(monkeypatch):
    """El patrón compilador de Ada exige thot (validate_consistency) y ada
    (reconcile/assemble). Sin thot activa, el plan de Ada se rechazaría con
    certeza: no se gasta la llamada paga y se cae a qwen con el motivo."""
    cliente, resolver, evento, correr = _correr_con_cerebros(
        monkeypatch, {"ada", "kimi", "jax_local"}, "x" * 250,
        '[{"facet": "kimi", "capability": "analysis", "prompt": "x"}]')
    with pytest.raises(plan_mod.MotorGobernadoNoDisponible):
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


# --- E2b-1a MINOR 1 (auditoria #362): lista blanca, no lista negra ---------

@pytest.mark.parametrize("faceta", sorted(models.HTTP_FACETS))
def test_el_predicado_acepta_solo_las_facetas_http_gobernadas(faceta):
    assert models.faceta_ejecutable_en_pipeline(faceta) is True


@pytest.mark.parametrize("faceta", ["kimi", "jax_local", "el_juez", "hyde", "faceta_nueva_subprocess", "", None])
def test_el_predicado_rechaza_toda_faceta_fuera_de_la_lista_blanca(faceta):
    """Una faceta nueva activada en la tabla (otro transporte, otro nombre) no
    entra por omision: sin estar en HTTP_FACETS, el plan no pasa -- si no, falla
    al ejecutar, despues de haber cobrado los pasos anteriores."""
    assert models.faceta_ejecutable_en_pipeline(faceta) is False


def test_validar_facetas_ejecutables_rechaza_una_faceta_nueva_no_listada():
    nuevo = models.Step(facet="faceta_nueva_subprocess", capability="analysis", step_index=0)
    ok = models.Step(facet="hipatia", capability="analysis", step_index=1)
    with pytest.raises(plan_mod.MotorGobernadoNoDisponible) as exc:
        plan_mod.validar_facetas_ejecutables([nuevo, ok])
    assert [v.facet for v in exc.value.violations] == ["faceta_nueva_subprocess"]


# --- El auditor local (el_juez) no es un paso de pipeline ------------------
# `ejecutor.auditor_faceta_local` (hoy `el_juez`) lo elige C5 para auditar
# misiones con datos de clientes. Con la lista blanca, el_juez no es ejecutable
# como PASO de pipeline (igual que jax_local, que corre el mismo modelo): eso es
# lo correcto, y NO rompe a C5 porque C5 no pasa por el predicado, el reroute,
# el menu ni el arbitro de Jacobs. Estas dos pruebas fijan ambas mitades.

def _modulos_py(*carpetas):
    from pathlib import Path
    raiz = Path(__file__).resolve().parents[1]
    for carpeta in carpetas:
        for p in sorted((raiz / carpeta).rglob("*.py")):
            if "__pycache__" not in p.parts and not p.name.endswith("_test.py") and not p.name.startswith("test_"):
                yield p


def test_c5_y_el_auditor_local_no_pasan_por_el_predicado_ni_por_el_planificador_de_jacobs():
    import ast
    prohibidos_modulo = {"jacobs.plan", "jacobs.executor", "jacobs.models", "jacobs.continuar", "jacobs.devolucion"}
    prohibidos_nombre = {"faceta_ejecutable_en_pipeline", "validar_facetas_ejecutables",
                         "HTTP_FACETS", "MOTOR_FACETS", "FACETAS_CERRADAS_A_PROPOSITO", "PlanBuilder"}
    hallazgos = []
    archivos = list(_modulos_py("jax/ejecutor", "scripts/ejecutor_contratos"))
    assert archivos, "no se encontro el codigo de C5"
    for p in archivos:
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.ImportFrom) and n.module:
                importados = {n.module} | {f"{n.module}.{a.name}" for a in n.names}
                if importados & prohibidos_modulo or any(a.name in prohibidos_nombre for a in n.names):
                    hallazgos.append(f"{p.name}:{n.lineno}")
            elif isinstance(n, ast.Import) and any(a.name in prohibidos_modulo for a in n.names):
                hallazgos.append(f"{p.name}:{n.lineno}")
            elif isinstance(n, ast.Name) and n.id in prohibidos_nombre:
                hallazgos.append(f"{p.name}:{n.lineno}")
    assert hallazgos == []


def test_jacobs_solo_lee_la_faceta_arbitro_http_y_nunca_la_local():
    """El arbitro del plan sale SOLO de `ejecutor.auditor_faceta` (hoy thot, HTTP);
    ninguna linea de jacobs/ lee `auditor_faceta_local`."""
    leen_local = [p.name for p in _modulos_py("jacobs") if "auditor_faceta_local" in p.read_text(encoding="utf-8")]
    assert leen_local == []
    from jax.ejecutor.contratos import eleccion_c5
    cfg = eleccion_c5.ConfigC5(**{**{c: None for c in eleccion_c5.ConfigC5.__dataclass_fields__},
                                  "auditor_faceta": "thot", "auditor_faceta_local": "el_juez"})
    assert eleccion_c5.elegir_auditor_faceta(cfg, hay_datos_de_clientes=True) == "el_juez"
    assert eleccion_c5.elegir_auditor_faceta(cfg, hay_datos_de_clientes=False) == "thot"
    assert models.faceta_ejecutable_en_pipeline("thot") is True
    assert models.faceta_ejecutable_en_pipeline("el_juez") is False
