"""Un paso truncado por tope de longitud falla, no entrega texto a medias.

Hoy los tres transportes HTTP directos (openai-compat, Ollama, Gemini) ni leen
el campo de corte del proveedor: el texto a medias sigue viaje como resultado
bueno y el paso siguiente construye sobre una frase cortada. El camino de
motor (las_manos/motor_registry/worker.py:828) ya falla el job cuando
finish_reason == "length" -- esto es lo mismo para los transportes directos.

IMPORTANTE 3 (revisión final 2026-09-18, tanda historial-y-arreglos-de-pipeline):
el paso que corta por longitud es la llamada MÁS CARA posible (gastó todo el
tope de salida) y su costo NUNCA se registraba -- `PasoTruncado` se lanzaba
dentro de `_invoke_*`, y `record_direct_usage` corría después de que
`invoke()` retornara normalmente, que acá nunca pasa. El arreglo: el consumo
viaja EN la excepción (`tokens_in`/`tokens_out`), y `invoke()` lo registra
antes de propagar el fallo. Los tests de abajo cubren, en orden: que
`_texto_o_truncado` adjunta el consumo real de cada transporte a la
excepción, y que `invoke()` (el nivel que arma el resultado por transporte)
llama a `record_direct_usage` con ESE consumo antes de dejar que el fallo
suba.
"""
from __future__ import annotations

import pytest

from jacobs.executor import PasoTruncado, _texto_o_truncado


def test_openai_compat_truncado_falla():
    """Hoy el texto a medias sigue viaje y el paso siguiente construye sobre
    una frase cortada."""
    data = {"choices": [{"message": {"content": "a medi"}, "finish_reason": "length"}]}
    with pytest.raises(PasoTruncado):
        _texto_o_truncado(data, "openai_compat")


def test_openai_compat_completo_pasa():
    data = {"choices": [{"message": {"content": "entero"}, "finish_reason": "stop"}]}
    assert _texto_o_truncado(data, "openai_compat") == "entero"


def test_openai_compat_truncado_lleva_el_consumo_en_la_excepcion():
    data = {"choices": [{"message": {"content": "a medi"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 111, "completion_tokens": 222}}
    with pytest.raises(PasoTruncado) as exc:
        _texto_o_truncado(data, "openai_compat")
    assert exc.value.tokens_in == 111
    assert exc.value.tokens_out == 222


def test_ollama_truncado_falla():
    data = {"message": {"content": "a medi"}, "done_reason": "length"}
    with pytest.raises(PasoTruncado):
        _texto_o_truncado(data, "ollama")


def test_ollama_completo_pasa():
    data = {"message": {"content": "entero"}, "done_reason": "stop"}
    assert _texto_o_truncado(data, "ollama") == "entero"


def test_ollama_truncado_lleva_el_consumo_en_la_excepcion():
    data = {"message": {"content": "a medi"}, "done_reason": "length",
            "prompt_eval_count": 333, "eval_count": 444}
    with pytest.raises(PasoTruncado) as exc:
        _texto_o_truncado(data, "ollama")
    assert exc.value.tokens_in == 333
    assert exc.value.tokens_out == 444


def test_gemini_truncado_falla():
    data = {"candidates": [{"finishReason": "MAX_TOKENS",
                            "content": {"parts": [{"text": "a medi"}]}}]}
    with pytest.raises(PasoTruncado):
        _texto_o_truncado(data, "gemini")


def test_gemini_completo_pasa():
    data = {"candidates": [{"finishReason": "STOP",
                            "content": {"parts": [{"text": "entero"}]}}]}
    assert _texto_o_truncado(data, "gemini") == "entero"


def test_gemini_truncado_lleva_el_consumo_en_la_excepcion():
    data = {
        "candidates": [{"finishReason": "MAX_TOKENS",
                         "content": {"parts": [{"text": "a medi"}]}}],
        "usageMetadata": {"promptTokenCount": 555, "candidatesTokenCount": 666},
    }
    with pytest.raises(PasoTruncado) as exc:
        _texto_o_truncado(data, "gemini")
    assert exc.value.tokens_in == 555
    assert exc.value.tokens_out == 666


def test_truncado_sin_datos_de_consumo_no_explota():
    """Un proveedor que corta pero no manda `usage` (o lo manda vacío) no
    puede romper -- el consumo por defecto es 0, no un KeyError."""
    data = {"choices": [{"message": {"content": "a"}, "finish_reason": "length"}]}
    with pytest.raises(PasoTruncado) as exc:
        _texto_o_truncado(data, "openai_compat")
    assert exc.value.tokens_in == 0
    assert exc.value.tokens_out == 0


# ---------------------------------------------------------------------------
# `_despachar_transporte_directo`: el nivel que arma `result` por transporte
# HTTP directo y llama a `record_direct_usage` -- acá se prueba que el
# registro de uso corre TAMBIÉN cuando el paso falla por truncado, con el
# consumo real que trae la excepción (no con ceros).
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402

from unittest.mock import AsyncMock  # noqa: E402

from jacobs import executor  # noqa: E402
from jacobs.models import Pipeline, Step  # noqa: E402


class _FacetFalso:
    transport = "http_openai_compat"
    provider_id = "prov-1"
    model = "modelo-de-mentira"
    base_url = "http://facet.invalid/v1"
    credential = "token-de-mentira"
    persona = None


def _paso_y_pipeline():
    step = Step(pipeline_id="p1", step_index=0, facet="thot", capability="critique",
                input={"prompt": "hola"}, depends_on=[])
    pipeline = Pipeline(pipeline_id="p1", name="n", invoked_by="plataforma", mode="autonomous",
                        plan=[step], context={"objective": "o"}, run_epoch=0,
                        user_id="u1", tenant_id="t1")
    return step, pipeline


def test_el_registro_de_uso_corre_igual_cuando_el_paso_trunca(monkeypatch):
    """El escenario del hallazgo: el paso MÁS CARO (gastó todo el tope de
    salida) es justo el que antes NO se registraba. Acá se verifica que
    `record_direct_usage` se llama con el consumo real de la excepción, y
    DESPUÉS el fallo sigue subiendo (el step tiene que seguir fallando)."""
    step, pipeline = _paso_y_pipeline()
    f = _FacetFalso()

    async def _invoke_que_trunca(f_, prompt, timeout):
        raise executor.PasoTruncado("cortado", tokens_in=1000, tokens_out=2000)

    monkeypatch.setattr(executor, "_invoke_http_openai_compat", _invoke_que_trunca)
    llamadas = []

    async def _registro_falso(*args, **kwargs):
        llamadas.append((args, kwargs))

    monkeypatch.setattr(executor, "record_direct_usage", _registro_falso)

    async def correr():
        with pytest.raises(executor.PasoTruncado):
            await executor._despachar_transporte_directo(step, pipeline, f, "prompt", 30)

    asyncio.run(correr())

    assert len(llamadas) == 1, "record_direct_usage tiene que llamarse una vez, no cero"
    args, kwargs = llamadas[0]
    assert args[:5] == ("u1", "t1", "thot", "prov-1", "modelo-de-mentira")
    assert args[5:7] == (1000, 2000), f"se esperaban los tokens de la excepción, no ceros: {args}"
    assert kwargs.get("pipeline_id") == "p1"


# ---------------------------------------------------------------------------
# MENOR (revisión final 2026-09-18): `PasoTruncado.codigo` existía y nadie lo
# leía -- `_run_one_step` guardaba `str(exc)` a secas, así que la Mesa no
# podía distinguir un fallo por truncado (justo el caso donde Continuar tiene
# remedio) de cualquier otro error. Ahora el código viaja como PREFIJO del
# error que se guarda, para cualquier excepción que declare `.codigo` -- no
# sólo PasoTruncado.
# ---------------------------------------------------------------------------

class _TiendaFalsaMinima:
    """Lo mínimo que `_run_one_step` necesita: escritura condicional que
    siempre gana, y un registro de los eventos que se le mandan."""

    def __init__(self):
        self.eventos: list[tuple[str, dict]] = []

    async def step_upsert_si_epoca(self, s, epoca):
        return True

    async def event_append(self, pipeline_id, event_type, payload=None, step_id=None):
        self.eventos.append((event_type, payload or {}))


def test_un_paso_truncado_guarda_el_codigo_como_prefijo_del_error(monkeypatch):
    tienda = _TiendaFalsaMinima()
    monkeypatch.setattr(executor, "store", tienda)
    step, pipeline = _paso_y_pipeline()
    pipeline.plan = [step]

    async def _despacho_que_trunca(paso, pipe):
        raise executor.PasoTruncado("openai_compat corto la salida por longitud (length)")

    monkeypatch.setattr(executor, "correr_con_interruptor", lambda coro: coro)
    monkeypatch.setattr(executor, "_dispatch_step", _despacho_que_trunca)

    asyncio.run(executor._run_one_step(step, 0, pipeline))

    fallidos = [p for t, p in tienda.eventos if t == "STEP_FAILED"]
    assert len(fallidos) == 1
    assert fallidos[0]["error"].startswith("[paso_truncado]"), fallidos[0]["error"]


def test_un_error_comun_no_gana_prefijo(monkeypatch):
    """Un error sin `.codigo` (el caso general) sigue guardándose tal cual --
    el prefijo es sólo para excepciones que declaran un código propio."""
    tienda = _TiendaFalsaMinima()
    monkeypatch.setattr(executor, "store", tienda)
    step, pipeline = _paso_y_pipeline()
    pipeline.plan = [step]

    async def _despacho_que_rompe(paso, pipe):
        raise RuntimeError("se cayó el proveedor")

    monkeypatch.setattr(executor, "correr_con_interruptor", lambda coro: coro)
    monkeypatch.setattr(executor, "_dispatch_step", _despacho_que_rompe)

    asyncio.run(executor._run_one_step(step, 0, pipeline))

    fallidos = [p for t, p in tienda.eventos if t == "STEP_FAILED"]
    assert len(fallidos) == 1
    assert fallidos[0]["error"] == "se cayó el proveedor"


def test_el_registro_de_uso_corre_una_sola_vez_en_el_camino_feliz(monkeypatch):
    """No duplicar el registro cuando el paso SÍ completa -- mismo
    comportamiento que antes de este arreglo."""
    step, pipeline = _paso_y_pipeline()
    f = _FacetFalso()

    async def _invoke_ok(f_, prompt, timeout):
        return {"success": True, "result": "listo", "tokens_in": 10, "tokens_out": 20}

    monkeypatch.setattr(executor, "_invoke_http_openai_compat", _invoke_ok)
    llamadas = []

    async def _registro_falso(*args, **kwargs):
        llamadas.append((args, kwargs))

    monkeypatch.setattr(executor, "record_direct_usage", _registro_falso)

    async def correr():
        return await executor._despachar_transporte_directo(step, pipeline, f, "prompt", 30)

    resultado = asyncio.run(correr())
    assert resultado["result"] == "listo"
    assert len(llamadas) == 1
    assert llamadas[0][0][5:7] == (10, 20)
