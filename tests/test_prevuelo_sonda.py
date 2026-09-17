"""La sonda del pre-vuelo (spec 2026-09-17 §4.5, §8). Sin red ni DB: httpx, el
resolver, el registro de salud y el de uso van mockeados.

«Responde» = 2xx del proveedor, aunque el contenido venga cortado: mide
disponibilidad, no calidad.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import json
import os
from decimal import Decimal
from unittest.mock import ANY, AsyncMock, patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import httpx  # noqa: E402

from facet_resolver import ResolvedFacet  # noqa: E402
from jacobs import sonda, usage_writer  # noqa: E402
from jacobs.prevuelo_reglas import Despacho  # noqa: E402


def _despacho(**cambios):
    base = dict(
        clave_salud="jekyll", via_motor=False, transporte="http_openai_compat",
        provider_id="deepseek", base_url="https://api.example/v1", modelo="deepseek-v4-flash",
        max_tokens_param="max_tokens", max_output_tokens=8192, motor_max_tokens=0,
        precio_in=Decimal("0.27"), precio_out=Decimal("1.10"),
        tiene_herramientas=False, schema_con_reintento=False, persona=None,
    )
    base.update(cambios)
    return Despacho(**base)


def _faceta(**cambios):
    base = dict(key="jekyll", provider_id="deepseek", base_url="https://api.example/v1",
                model="deepseek-v4-flash", credential="k-secreta", transport="http_openai_compat",
                persona=None, params=None)
    base.update(cambios)
    return ResolvedFacet(**base)


class _Resp:
    def __init__(self, status, cuerpo):
        self.status_code = status
        self._cuerpo = cuerpo
        self.text = cuerpo if isinstance(cuerpo, str) else json.dumps(cuerpo)

    def json(self):
        return self._cuerpo


_OK_OPENAI = {"choices": [{"message": {"content": "o"}, "finish_reason": "length"}],
              "usage": {"prompt_tokens": 9, "completion_tokens": 16}}


def _sondear(monkeypatch, d, respuesta, faceta=None, registrar=None, uso=None):
    capturado = {}

    async def post(self, url, headers=None, json=None, **kw):
        capturado.update(url=url, headers=headers, json=json)
        if isinstance(respuesta, Exception):
            raise respuesta
        return respuesta

    monkeypatch.setenv("JAX_PREVUELO_SONDA_MAX_TOKENS", "16")
    monkeypatch.setenv("JAX_PREVUELO_SONDA_TIMEOUT_S", "1")
    with patch("httpx.AsyncClient.post", post), \
         patch.object(sonda, "resolve_facet", AsyncMock(return_value=faceta or _faceta())), \
         patch.object(sonda.facet_health, "registrar_evento_de_sonda", registrar or AsyncMock()), \
         patch.object(sonda, "record_direct_usage", uso or AsyncMock()):
        r = asyncio.run(sonda.sondear(d.clave_salud, d))
    return r, capturado


def test_openai_compat_2xx_aunque_venga_cortada_responde(monkeypatch):
    r, c = _sondear(monkeypatch, _despacho(), _Resp(200, _OK_OPENAI))
    assert r.ok and r.detalle is None
    assert c["url"] == "https://api.example/v1/chat/completions"
    assert c["json"]["max_tokens"] == 16 and c["json"]["stream"] is False
    assert c["headers"]["Authorization"] == "Bearer k-secreta"


def test_el_limite_es_el_menor_entre_config_y_catalogo(monkeypatch):
    d = _despacho(max_tokens_param="max_completion_tokens", max_output_tokens=8)
    _, c = _sondear(monkeypatch, d, _Resp(200, _OK_OPENAI))
    assert c["json"]["max_completion_tokens"] == 8
    assert "max_tokens" not in c["json"]


def test_gemini_manda_cabecera_y_maxOutputTokens_sin_herramientas(monkeypatch):
    f = _faceta(key="hipatia", provider_id="gemini", base_url="https://g.example/v1beta",
                model="gemini-x", credential="AIza-secreta", transport="http_gemini")
    d = _despacho(clave_salud="hipatia", transporte="http_gemini", provider_id="gemini",
                  modelo="gemini-x", max_tokens_param=None, max_output_tokens=65536)
    cuerpo = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}],
              "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1}}
    r, c = _sondear(monkeypatch, d, _Resp(200, cuerpo), faceta=f)
    assert r.ok
    assert c["url"] == "https://g.example/v1beta/models/gemini-x:generateContent"
    assert c["headers"] == {"x-goog-api-key": "AIza-secreta"}
    assert c["json"]["generationConfig"] == {"maxOutputTokens": 16}
    assert "tools" not in c["json"]


def test_http_5xx_no_responde_y_el_detalle_va_redactado(monkeypatch):
    r, _ = _sondear(monkeypatch, _despacho(), _Resp(503, "caído, la key k-secreta no sirve"))
    assert not r.ok
    assert "503" in r.detalle and "k-secreta" not in r.detalle


def test_timeout_da_faceta_caida_con_los_segundos(monkeypatch):
    r, _ = _sondear(monkeypatch, _despacho(), httpx.ReadTimeout("lento"))
    assert not r.ok and r.detalle == "timeout de sonda (1s)"


def test_motor_usa_el_cliente_del_worker_con_su_credencial(monkeypatch):
    d = _despacho(clave_salud="kimi", via_motor=True, provider_id="moonshot", modelo="kimi-k3",
                  base_url="https://api.moonshot.example/v1", max_output_tokens=131072,
                  motor_max_tokens=8000)
    llamar = AsyncMock(return_value=_OK_OPENAI)
    monkeypatch.setenv("JAX_PREVUELO_SONDA_MAX_TOKENS", "16")
    with patch("motor_registry.worker._call_http_openai_compat", llamar), \
         patch.object(sonda, "resolve_credential_instrumented", AsyncMock(return_value="k-moon")), \
         patch.object(sonda.facet_health, "registrar_evento_de_sonda", AsyncMock()), \
         patch.object(sonda, "record_direct_usage", AsyncMock()):
        r = asyncio.run(sonda.sondear("kimi", d))
    assert r.ok
    kw = llamar.await_args.kwargs
    assert (kw["api_url"], kw["model"], kw["api_key"], kw["limite"]) == (
        "https://api.moonshot.example/v1", "kimi-k3", "k-moon", {"max_tokens": 16})


def test_motor_ollama_sin_credencial_y_con_max_tokens(monkeypatch):
    d = _despacho(clave_salud="jax_local", via_motor=True, transporte="ollama", provider_id="ollama",
                  modelo="qwen3-coder:30b", base_url="http://localhost:11434/v1",
                  max_tokens_param=None, precio_in=None, precio_out=None)
    llamar, credencial = AsyncMock(return_value=_OK_OPENAI), AsyncMock()
    monkeypatch.setenv("JAX_PREVUELO_SONDA_MAX_TOKENS", "16")
    with patch("motor_registry.worker._call_http_openai_compat", llamar), \
         patch.object(sonda, "resolve_credential_instrumented", credencial), \
         patch.object(sonda.facet_health, "registrar_evento_de_sonda", AsyncMock()), \
         patch.object(sonda, "record_direct_usage", AsyncMock()):
        asyncio.run(sonda.sondear("jax_local", d))
    credencial.assert_not_awaited()
    assert llamar.await_args.kwargs["api_key"] == ""
    assert llamar.await_args.kwargs["limite"] == {"max_tokens": 16}


def test_registra_el_evento_con_el_resultado(monkeypatch):
    registrar = AsyncMock()
    _sondear(monkeypatch, _despacho(), _Resp(200, _OK_OPENAI), registrar=registrar)
    registrar.assert_awaited_once_with("jekyll", "ok", None, ANY)
    registrar = AsyncMock()
    r, _ = _sondear(monkeypatch, _despacho(), _Resp(500, "error"), registrar=registrar)
    registrar.assert_awaited_once_with("jekyll", "provider_error", r.detalle, ANY)


def test_si_no_se_puede_registrar_el_veredicto_se_mantiene_y_se_cuenta(monkeypatch):
    antes = sonda.registros_perdidos()
    r, _ = _sondear(monkeypatch, _despacho(), _Resp(200, _OK_OPENAI),
                    registrar=AsyncMock(side_effect=RuntimeError("base caída")))
    assert r.ok
    assert sonda.registros_perdidos() == antes + 1


def test_el_uso_de_la_sonda_se_registra_como_preflight_probe(monkeypatch):
    uso = AsyncMock()
    _sondear(monkeypatch, _despacho(), _Resp(200, _OK_OPENAI), uso=uso)
    uso.assert_awaited_once_with(None, None, "jekyll", "deepseek", "deepseek-v4-flash", 9, 16,
                                 request_type="preflight_probe")


def test_record_direct_usage_encola_con_el_request_type_pedido(monkeypatch):
    # Host y puerto de mentira: sin ellos _db_cfg() lanza ANTES de conectar y
    # el test no ejercitaría el camino de la conexión caída.
    monkeypatch.setenv("JAX_DB_HOST", "127.0.0.1")
    monkeypatch.setenv("JAX_DB_PORT", "1")
    encolar = AsyncMock(return_value="spool-1")
    with patch.object(usage_writer.aiomysql, "connect", AsyncMock(side_effect=OSError("sin base"))), \
         patch.object(usage_writer, "encolar_uso", encolar):
        asyncio.run(usage_writer.record_direct_usage(
            "1", "1", "jekyll", "deepseek", "m", 1, 2, request_type="preflight_probe"))
    assert encolar.await_args.args[0]["request_type"] == "preflight_probe"


def test_record_direct_usage_inserta_con_el_request_type_pedido(monkeypatch):
    monkeypatch.setenv("JAX_DB_HOST", "127.0.0.1")
    monkeypatch.setenv("JAX_DB_PORT", "1")
    ejecutados = []

    class _Cur:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def execute(self, sql, params=None):
            ejecutados.append((sql, params))

        async def fetchone(self):
            return None

    class _Conn:
        def cursor(self):
            return _Cur()

        def close(self):
            pass

    with patch.object(usage_writer.aiomysql, "connect", AsyncMock(return_value=_Conn())):
        asyncio.run(usage_writer.record_direct_usage(
            "1", "1", "jekyll", "deepseek", "m", 1, 2, request_type="preflight_probe"))
        asyncio.run(usage_writer.record_direct_usage("1", "1", "jekyll", "deepseek", "m", 1, 2))
    inserts = [p for sql, p in ejecutados if sql.startswith("INSERT INTO axioma_usage")]
    assert [p[-1] for p in inserts] == ["preflight_probe", "pipeline"]
