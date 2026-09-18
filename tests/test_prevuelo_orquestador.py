"""Orquestador del pre-vuelo (spec 2026-09-17 §4; desvíos 1, 3, 4 y 16 del
plan). Catálogo y sonda mockeados; el armado del prompt es el REAL del ejecutor.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from decimal import Decimal
from unittest.mock import AsyncMock

# Base de tests de ESTA sesión: respeta JAX_TEST_DB_SUFIJO en vez de
# clavar el nombre (mismo override incondicional que antes).
from base_de_test import fijar_base_de_test  # noqa: E402

fijar_base_de_test()

import pytest  # noqa: E402

from jacobs import prevuelo as pv  # noqa: E402
from jacobs.executor import MAX_DEP_CONTEXT_CHARS, _build_context_input, _enrich_prompt  # noqa: E402
from jacobs.models import Pipeline, Step  # noqa: E402
from jacobs.prevuelo_catalogo import Catalogo, FilaFaceta, FilaModelo, MotorResuelto  # noqa: E402
from jacobs.sonda import ResultadoSonda  # noqa: E402

AHORA = 1_000_000.0


def _catalogo(salud=None, credencial=True):
    return Catalogo(
        facetas={
            "jekyll": FilaFaceta("jekyll", "http_openai_compat", None, "deepseek",
                                 "https://api.deepseek.example/v1", "deepseek-v4-flash"),
            "hipatia": FilaFaceta("hipatia", "http_gemini", None, "gemini",
                                  "https://g.example/v1beta", "gemini-x"),
        },
        modelos={
            ("deepseek", "deepseek-v4-flash"): FilaModelo("max_tokens", 8192, Decimal("0.27"), Decimal("1.10")),
            ("gemini", "gemini-x"): FilaModelo(None, 65536, Decimal("0.30"), Decimal("2.50")),
            ("moonshot", "kimi-k3"): FilaModelo("max_tokens", 131072, Decimal("0.60"), Decimal("2.50")),
        },
        min_output_tokens={},
        proveedores_con_credencial=frozenset({"deepseek", "gemini", "moonshot"}) if credencial else frozenset(),
        salud=salud if salud is not None else {
            "jekyll": (AHORA - 60, "ok"), "hipatia": (AHORA - 60, "ok"), "kimi": (AHORA - 60, "ok")},
    )


def _paso(i, facet, capability="research", deps=None, prompt="investigá"):
    return Step(pipeline_id="p", step_index=i, facet=facet, capability=capability,
                input={"prompt": prompt}, depends_on=deps or [])


@asynccontextmanager
async def _conexion_de_prueba():
    # Task 15b: prevuelo() toma UNA conexión del pool del store y la pasa a
    # los dos lectores, que acá están mockeados: no se abre nada real.
    yield object()


def _instalar(monkeypatch, catalogo, motores=None, sondear=None):
    monkeypatch.setattr(pv.store, "conexion_del_pool", _conexion_de_prueba)
    leer = AsyncMock(return_value=catalogo)
    monkeypatch.setattr(pv.prevuelo_catalogo, "leer_catalogo", leer)
    monkeypatch.setattr(pv.prevuelo_catalogo, "resolver_motores", AsyncMock(return_value=motores or {}))
    sonda = sondear or AsyncMock(return_value=ResultadoSonda(True, None))
    monkeypatch.setattr(pv.sonda, "sondear", sonda)
    monkeypatch.setattr(pv, "_ahora", lambda: AHORA)
    monkeypatch.setattr(pv, "_max_iteraciones", lambda: 5)
    monkeypatch.setenv("JAX_PREVUELO_CHARS_POR_TOKEN", "2")
    return leer, sonda


def _correr(pasos, contexto=None, **kw):
    return asyncio.run(pv.prevuelo(pasos, contexto or {"objective": "o"}, **kw))


def test_crear_evalua_todos_los_pasos_y_suma_el_costo(monkeypatch):
    _instalar(monkeypatch, _catalogo())
    v = _correr([_paso(0, "jekyll"), _paso(1, "hipatia", deps=[0])])
    assert v.ok
    assert [c.paso for c in v.pasos_costo] == [0, 1]
    assert v.costo_max_usd == sum(c.usd_max for c in v.pasos_costo)


def test_continuar_solo_evalua_los_pendientes(monkeypatch):
    _instalar(monkeypatch, _catalogo())
    v = _correr([_paso(0, "jekyll"), _paso(1, "hipatia", deps=[0])], pendientes={1})
    assert [c.paso for c in v.pasos_costo] == [1]


def test_salud_fresca_no_sondea(monkeypatch):
    _, sonda = _instalar(monkeypatch, _catalogo())
    v = _correr([_paso(0, "jekyll")])
    sonda.assert_not_awaited()
    assert v.sondeadas == ()


def test_sin_evento_sondea_y_si_responde_pasa(monkeypatch):
    _, sonda = _instalar(monkeypatch, _catalogo(salud={}))
    v = _correr([_paso(0, "jekyll")])
    assert sonda.await_args.args[0] == "jekyll"
    assert v.ok and v.sondeadas == ("jekyll",)


def test_sonda_que_no_responde_bloquea_con_faceta_caida(monkeypatch):
    caida = AsyncMock(return_value=ResultadoSonda(False, "timeout de sonda (20s)"))
    _instalar(monkeypatch, _catalogo(salud={}), sondear=caida)
    v = _correr([_paso(0, "jekyll")])
    assert not v.ok
    assert [(x.regla, x.detalle) for x in v.violaciones] == [("faceta_caida", "timeout de sonda (20s)")]


def test_una_sola_sonda_por_clave_aunque_haya_varios_pasos(monkeypatch):
    _, sonda = _instalar(monkeypatch, _catalogo(salud={}))
    _correr([_paso(0, "jekyll"), _paso(1, "jekyll", deps=[0])])
    assert sonda.await_count == 1


def test_las_sondas_corren_en_paralelo(monkeypatch):
    estado = {"en_curso": 0, "max": 0}

    async def sondear(clave, d, **kw):
        estado["en_curso"] += 1
        estado["max"] = max(estado["max"], estado["en_curso"])
        await asyncio.sleep(0.01)
        estado["en_curso"] -= 1
        return ResultadoSonda(True, None)

    _instalar(monkeypatch, _catalogo(salud={}), sondear=sondear)
    _correr([_paso(0, "jekyll"), _paso(1, "hipatia")])
    assert estado["max"] == 2


def test_dependencia_sin_ref_cuenta_el_peor_caso(monkeypatch):
    _instalar(monkeypatch, _catalogo())
    v = _correr([_paso(0, "jekyll"), _paso(1, "jekyll", deps=[0])])
    assert v.pasos_costo[1].tokens_in_max >= MAX_DEP_CONTEXT_CHARS // 2


def test_dependencia_con_ref_mide_lo_real(monkeypatch):
    _instalar(monkeypatch, _catalogo())
    contexto = {"objective": "o", "step_0_ref": "inline:" + json.dumps({"result": "corto"})}
    v = _correr([_paso(0, "jekyll"), _paso(1, "jekyll", deps=[0])], contexto, pendientes={1})
    assert v.pasos_costo[0].tokens_in_max < 2000


def test_solo_hyde_no_lee_el_catalogo(monkeypatch):
    leer, _ = _instalar(monkeypatch, _catalogo())
    v = _correr([_paso(0, "hyde", capability="execute")])
    leer.assert_not_awaited()
    assert v.ok and v.pasos_costo[0].motivo == "suscripcion"


def test_un_error_de_la_base_se_propaga(monkeypatch):
    _instalar(monkeypatch, _catalogo())
    monkeypatch.setattr(pv.prevuelo_catalogo, "leer_catalogo", AsyncMock(side_effect=OSError("base caída")))
    with pytest.raises(OSError):
        _correr([_paso(0, "jekyll")])


def test_paso_de_motor_usa_el_motor_resuelto(monkeypatch):
    motor = MotorResuelto("kimi", "http_openai_compat", "moonshot", "https://api.moonshot.example/v1",
                          "kimi-k3", 8000, False, "")
    _, sonda = _instalar(monkeypatch, _catalogo(salud={}), motores={0: motor})
    v = _correr([_paso(0, "kimi", capability="generate")])
    clave, despacho = sonda.await_args.args
    assert clave == "kimi" and despacho.via_motor
    assert v.pasos_costo[0].tokens_out_max == 8000


def test_paso_con_violacion_no_se_sondea(monkeypatch):
    _, sonda = _instalar(monkeypatch, _catalogo(salud={}, credencial=False))
    v = _correr([_paso(0, "jekyll")])
    sonda.assert_not_awaited()
    assert [x.regla for x in v.violaciones] == ["credencial_ausente"]


def test_sonda_que_revienta_propaga_sin_dejar_tareas_pendientes(monkeypatch):
    """Ninguna instrucción del brief pedía esto, pero Task 7 (post-brief)
    exige que una excepción de CUALQUIER sonda del `asyncio.gather` se
    propague (503 aguas arriba, §8) SIN dejar la otra sonda como tarea
    huérfana (warning "Task exception was never retrieved" / tarea nunca
    cancelada). jekyll (sondeada primero alfabéticamente, tarea "hipatia" en
    verdad -- ver nota abajo) duerme; hipatia revienta antes: si prevuelo()
    usara un asyncio.gather ingenuo, la tarea de jekyll seguiría pendiente
    cuando la excepción sale de prevuelo()."""
    from facet_resolver import FacetUnavailableError

    async def sondear(clave, d, **kw):
        if clave == "hipatia":
            raise FacetUnavailableError("DB caída al resolver la faceta")
        await asyncio.sleep(0.05)
        return ResultadoSonda(True, None)

    _instalar(monkeypatch, _catalogo(salud={}), sondear=sondear)

    async def correr():
        with pytest.raises(FacetUnavailableError):
            await pv.prevuelo([_paso(0, "jekyll"), _paso(1, "hipatia")], {"objective": "o"})
        pendientes = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        assert pendientes == []

    asyncio.run(correr())


# ---------------------------------------------------------------------
# Fix round 1 (revisión de Task 8, 2026-09-17)
# ---------------------------------------------------------------------

def test_relleno_de_dependencia_faltante_no_subestima_una_real_sobredimensionada(monkeypatch):
    """Hallazgo [Important] de la revisión: `_RELLENO_DE_DEPENDENCIA` medía
    exactamente MAX_DEP_CONTEXT_CHARS -- el mismo largo que `_build_context_input`
    deja pasar SIN marcarlo truncado (executor.py: `truncated = len(text) >
    MAX_DEP_CONTEXT_CHARS`, estrictamente mayor). Una dependencia real que
    de verdad excede el tope SÍ queda truncada, y `_enrich_prompt` le agrega
    la nota " [TRUNCADO -- dependencia excede el tope]" (40 chars) al
    armar el prompt -- el relleno del peor caso, sin esa nota, contaba 40
    chars menos que un caso real. "Sobreestima, nunca subestima" (spec
    §4.6) se rompía por code exacto en el borde.
    Con el relleno viejo ("x" * MAX_DEP_CONTEXT_CHARS) este test da rojo:
    el peor caso (sin ref) mide MENOS que una dependencia real de 2x el
    tope, porque a esta última SÍ se le agrega la nota de truncado."""
    _instalar(monkeypatch, _catalogo())
    v_falta = _correr([_paso(0, "jekyll"), _paso(1, "jekyll", deps=[0])])

    contexto = {"objective": "o",
                "step_0_ref": "inline:" + json.dumps({"result": "z" * (2 * MAX_DEP_CONTEXT_CHARS)})}
    v_real = _correr([_paso(0, "jekyll"), _paso(1, "jekyll", deps=[0])], contexto, pendientes={1})

    assert v_falta.pasos_costo[1].tokens_in_max >= v_real.pasos_costo[0].tokens_in_max


def test_motor_incluye_el_contexto_de_identidad_completo_con_separador(monkeypatch):
    """M1 de la revisión: si el bloque de `_chars_de_entrada` que suma
    `build_identity_context(...)` + el separador "\\n---\\n" se salteara para
    un paso de motor, este valor pinneado (calculado con las MISMAS
    funciones reales) dejaría de coincidir -- rojo bajo esa mutación."""
    from motor_registry.identity_context import build_identity_context
    from motor_registry.worker import _REFORMAS_V3_PREDICATES

    motor = MotorResuelto("kimi", "http_openai_compat", "moonshot",
                          "https://api.moonshot.example/v1", "kimi-k3", 8000, False, "")
    _instalar(monkeypatch, _catalogo(salud={}), motores={0: motor})
    paso = _paso(0, "kimi", capability="generate")
    v = _correr([paso])

    pipeline = Pipeline(name="prevuelo", invoked_by="plataforma", mode="autonomous",
                        plan=[paso], context={"objective": "o"})
    chars = len(_enrich_prompt(_build_context_input(paso, pipeline)))
    chars += len(build_identity_context(
        motor_name="kimi", capabilities=["generate"], catalog={},
        predicates=_REFORMAS_V3_PREDICATES, task_id=pv._TASK_ID_DE_MEDIDA,
    )) + len("\n---\n")
    esperado = -(-chars // 2)  # JAX_PREVUELO_CHARS_POR_TOKEN=2, fijado por _instalar

    assert v.pasos_costo[0].tokens_in_max == esperado


def test_persona_de_la_faceta_suma_exactamente_su_largo_al_costo(monkeypatch):
    """M2 de la revisión: si `+ len(d.persona or "")` se sacara de
    `_chars_de_entrada`, el paso con persona mediría lo mismo que el paso
    sin persona -- las dos igualdades pinneadas de abajo (y por lo tanto
    su diferencia exacta) dejarían de cumplirse."""
    persona = "Sos Jekyll: respondé con evidencia citada, sin inventar."
    base = _catalogo()

    _instalar(monkeypatch, base)
    paso = _paso(0, "jekyll")
    v_sin = _correr([paso])

    catalogo_con = Catalogo(
        facetas={"jekyll": FilaFaceta("jekyll", "http_openai_compat", persona, "deepseek",
                                      "https://api.deepseek.example/v1", "deepseek-v4-flash")},
        modelos=base.modelos, min_output_tokens={},
        proveedores_con_credencial=frozenset({"deepseek"}),
        salud={"jekyll": (AHORA - 60, "ok")},
    )
    _instalar(monkeypatch, catalogo_con)
    v_con = _correr([paso])

    pipeline = Pipeline(name="prevuelo", invoked_by="plataforma", mode="autonomous",
                        plan=[paso], context={"objective": "o"})
    chars_sin = len(_enrich_prompt(_build_context_input(paso, pipeline)))
    chars_con = chars_sin + len(persona)
    tokens_sin = -(-chars_sin // 2)
    tokens_con = -(-chars_con // 2)

    assert v_sin.pasos_costo[0].tokens_in_max == tokens_sin
    assert v_con.pasos_costo[0].tokens_in_max == tokens_con
    assert v_con.pasos_costo[0].tokens_in_max - v_sin.pasos_costo[0].tokens_in_max == tokens_con - tokens_sin


def test_tokens_in_max_coincide_con_el_armado_real_del_ejecutor(monkeypatch):
    """M5 de la revisión: si `_enrich_prompt`/`_build_context_input` se
    reemplazaran por otra cosa dentro de `_chars_de_entrada`, este valor
    -- calculado acá con las funciones REALES del ejecutor, no una
    aproximación -- dejaría de coincidir con `tokens_in_max`."""
    _instalar(monkeypatch, _catalogo())
    paso = _paso(0, "jekyll")
    v = _correr([paso])

    pipeline = Pipeline(name="prevuelo", invoked_by="plataforma", mode="autonomous",
                        plan=[paso], context={"objective": "o"})
    chars = len(_enrich_prompt(_build_context_input(paso, pipeline)))
    esperado = -(-chars // 2)

    assert v.pasos_costo[0].tokens_in_max == esperado


def test_medir_el_prompt_corre_por_asyncio_to_thread(monkeypatch):
    """M6 de la revisión: si el `await asyncio.to_thread(_chars_de_entrada,
    ...)` de `prevuelo()` se reemplazara por una llamada directa (bloqueante
    en el loop), este espía -- que envuelve el `asyncio.to_thread` REAL, así
    que el resultado no cambia -- dejaría de ver pasar `_chars_de_entrada`
    por él."""
    _instalar(monkeypatch, _catalogo())
    real_to_thread = asyncio.to_thread
    llamadas = []

    async def espia(fn, *args, **kwargs):
        llamadas.append(fn)
        return await real_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr(pv.asyncio, "to_thread", espia)
    v = _correr([_paso(0, "jekyll")])

    assert pv._chars_de_entrada in llamadas
    assert v.ok


# ---------------------------------------------------------------------
# F5 (ola final, 2026-09-17): estampida de sondas. N pre-vuelos concurrentes
# con la misma clave vencida lanzaban N sondas PAGAS. Ahora hay un solo vuelo
# por clave en el proceso: los demás esperan el mismo resultado; el registro
# se limpia al terminar (también con excepción o cancelación) y una
# excepción llega a todos los que esperan.
# ---------------------------------------------------------------------

def _sonda_lenta(llamadas, resultado=None, error=None, pausa=0.05):
    async def sondear(clave, d, **kw):
        llamadas.append(clave)
        await asyncio.sleep(pausa)
        if error is not None:
            raise error
        return resultado or ResultadoSonda(True, None)
    return sondear


def test_diez_prevuelos_concurrentes_lanzan_una_sola_sonda(monkeypatch):
    llamadas = []
    _instalar(monkeypatch, _catalogo(salud={}), sondear=_sonda_lenta(llamadas))

    async def correr():
        return await asyncio.gather(*[
            pv.prevuelo([_paso(0, "jekyll")], {"objective": "o"}) for _ in range(10)])

    veredictos = asyncio.run(correr())
    assert llamadas == ["jekyll"]
    assert all(v.ok and v.sondeadas == ("jekyll",) for v in veredictos)
    assert pv._sondas_en_vuelo == {}


def test_la_excepcion_del_vuelo_llega_a_todos_y_el_registro_se_limpia(monkeypatch):
    from facet_resolver import FacetUnavailableError

    llamadas = []
    _instalar(monkeypatch, _catalogo(salud={}),
              sondear=_sonda_lenta(llamadas, error=FacetUnavailableError("DB caída")))

    async def correr():
        return await asyncio.gather(*[
            pv.prevuelo([_paso(0, "jekyll")], {"objective": "o"}) for _ in range(5)],
            return_exceptions=True)

    salidas = asyncio.run(correr())
    assert llamadas == ["jekyll"]
    assert all(isinstance(s, FacetUnavailableError) for s in salidas)
    assert pv._sondas_en_vuelo == {}


def test_terminado_el_vuelo_el_siguiente_prevuelo_vuelve_a_sondear(monkeypatch):
    """No es una caché de resultados: sólo se comparte lo que está EN VUELO.
    El dato fresco lo guarda facet_health_event, no este registro."""
    llamadas = []
    _instalar(monkeypatch, _catalogo(salud={}), sondear=_sonda_lenta(llamadas, pausa=0))
    _correr([_paso(0, "jekyll")])
    _correr([_paso(0, "jekyll")])
    assert llamadas == ["jekyll", "jekyll"]


def test_cancelar_a_uno_no_corta_la_sonda_de_los_demas_y_cancelar_a_todos_si(monkeypatch):
    llamadas, terminadas = [], []

    async def sondear(clave, d, **kw):
        llamadas.append(clave)
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            terminadas.append("cancelada")
            raise
        terminadas.append("completa")
        return ResultadoSonda(True, None)

    _instalar(monkeypatch, _catalogo(salud={}), sondear=sondear)

    async def correr():
        a = asyncio.ensure_future(pv.prevuelo([_paso(0, "jekyll")], {"objective": "o"}))
        b = asyncio.ensure_future(pv.prevuelo([_paso(0, "jekyll")], {"objective": "o"}))
        await asyncio.sleep(0.02)
        a.cancel()
        veredicto_b = await b
        assert a.cancelled()
        assert veredicto_b.ok and terminadas == ["completa"]

        c = asyncio.ensure_future(pv.prevuelo([_paso(0, "jekyll")], {"objective": "o"}))
        d = asyncio.ensure_future(pv.prevuelo([_paso(0, "jekyll")], {"objective": "o"}))
        await asyncio.sleep(0.02)
        c.cancel()
        d.cancel()
        await asyncio.gather(c, d, return_exceptions=True)
        assert terminadas == ["completa", "cancelada"]
        assert pv._sondas_en_vuelo == {}
        pendientes = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        assert pendientes == []

    asyncio.run(correr())
    assert llamadas == ["jekyll", "jekyll"]


def test_un_vuelo_de_un_loop_cerrado_no_se_reusa_y_se_vuelve_a_sondear(monkeypatch):
    """Pasada final R34, 3: un event loop que se cierra con una sonda en
    vuelo (un arnés, un CLI que corta) deja su entrada en el registro. El
    próximo pre-vuelo corre en OTRO loop: sumarse a esa tarea daría
    "attached to a different loop" (o esperaría para siempre). Tiene que
    ignorarla y sondear de nuevo."""
    monkeypatch.setattr(pv, "_sondas_en_vuelo", {})
    llamadas = []

    async def sondear(clave, d, **kw):
        llamadas.append(clave)
        if len(llamadas) == 1:
            await asyncio.sleep(3600)  # queda en vuelo cuando se cierra el loop
        return ResultadoSonda(True, None)

    _instalar(monkeypatch, _catalogo(salud={}), sondear=sondear)

    viejo = asyncio.new_event_loop()
    try:
        viejo.create_task(pv.prevuelo([_paso(0, "jekyll")], {"objective": "o"}))
        viejo.run_until_complete(asyncio.sleep(0.05))
        assert llamadas == ["jekyll"] and "jekyll" in pv._sondas_en_vuelo
    finally:
        viejo.close()  # sin cancelar ni esperar: el peor caso

    v = asyncio.run(pv.prevuelo([_paso(0, "jekyll")], {"objective": "o"}))
    assert v.ok and v.sondeadas == ("jekyll",)
    assert llamadas == ["jekyll", "jekyll"]

    # Pasada R37, 2: las corrutinas que quedaron en el loop cerrado se
    # destruyen al recolectarlas y, al cerrarse, intentan cancelar tareas de
    # ese loop -> "Event loop is closed". Es el peor caso que este test arma a
    # propósito: se recolectan ACÁ, bajo un hook que las espera, en vez de que
    # el GC las suelte como PytestUnraisableExceptionWarning sobre otro test.
    import gc
    import sys

    capturadas = []
    hook_original = sys.unraisablehook
    sys.unraisablehook = capturadas.append
    try:
        del viejo
        monkeypatch.setattr(pv, "_sondas_en_vuelo", {})
        for _ in range(3):
            gc.collect()
    finally:
        sys.unraisablehook = hook_original
    inesperadas = [c for c in capturadas
                   if not (isinstance(c.exc_value, RuntimeError) and "Event loop is closed" in str(c.exc_value))]
    assert capturadas and inesperadas == [], [(type(c.exc_value), c.exc_value) for c in capturadas]
