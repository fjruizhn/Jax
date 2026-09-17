"""Orquestador del pre-vuelo (spec 2026-09-17 §4; desvíos 1, 3, 4 y 16 del
plan). Catálogo y sonda mockeados; el armado del prompt es el REAL del ejecutor.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import json
import os
from decimal import Decimal
from unittest.mock import AsyncMock

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import pytest  # noqa: E402

from jacobs import prevuelo as pv  # noqa: E402
from jacobs.executor import MAX_DEP_CONTEXT_CHARS  # noqa: E402
from jacobs.models import Step  # noqa: E402
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


def _instalar(monkeypatch, catalogo, motores=None, sondear=None):
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
