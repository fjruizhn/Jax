"""La sonda del pre-vuelo (spec 2026-09-17 §4.5, §8). Sin red ni DB: httpx, el
resolver, el registro de salud y el de uso van mockeados.

«Responde» = 2xx del proveedor, aunque el contenido venga cortado: mide
disponibilidad, no calidad.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import uuid
from decimal import Decimal
from unittest.mock import ANY, AsyncMock, patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import aiomysql  # noqa: E402
import httpx  # noqa: E402

from credential_resolver import CredentialUnavailableError  # noqa: E402
from facet_resolver import FacetUnavailableError, ResolvedFacet  # noqa: E402
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


class _RespuestaHTTPError:
    """Lo mínimo que `exc.response.status_code`/`exc.response.text` de
    httpx.HTTPStatusError necesitan -- no un httpx.Response real."""
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


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
    # max_tokens_param distinto del hardcode que había antes ("max_tokens"):
    # si el código volviera a fijar el nombre en vez de leer d.max_tokens_param,
    # este test lo vería (item 6, revisión Task 7 fix round 1).
    d = _despacho(clave_salud="kimi", via_motor=True, provider_id="moonshot", modelo="kimi-k3",
                  base_url="https://api.moonshot.example/v1", max_output_tokens=131072,
                  motor_max_tokens=20000, max_tokens_param="max_completion_tokens")
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
        "https://api.moonshot.example/v1", "kimi-k3", "k-moon", {"max_completion_tokens": 16})


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


def test_si_no_se_puede_registrar_el_veredicto_se_mantiene_y_se_cuenta(monkeypatch, caplog):
    antes = sonda.registros_perdidos()
    with caplog.at_level(logging.WARNING, logger="jacobs.sonda"):
        r, _ = _sondear(monkeypatch, _despacho(), _Resp(200, _OK_OPENAI),
                        registrar=AsyncMock(side_effect=RuntimeError("base caída")))
    assert r.ok
    assert sonda.registros_perdidos() == antes + 1
    assert any(
        "no se pudo registrar la sonda" in m and "jekyll" in m for m in caplog.messages
    ), caplog.messages


def test_el_uso_de_la_sonda_se_registra_como_preflight_probe(monkeypatch):
    uso = AsyncMock()
    _sondear(monkeypatch, _despacho(), _Resp(200, _OK_OPENAI), uso=uso)
    uso.assert_awaited_once_with(None, None, "jekyll", "deepseek", "deepseek-v4-flash", 9, 16,
                                 request_type="preflight_probe")


# ---------------------------------------------------------------------------
# Fix round 1 (revisión post-Task 7). Ruling R13: tres clases de resultado --
# (a) una caída real de la DB al resolver SE PROPAGA (nunca se convierte en
# veredicto); (b) una falla LOCAL de preparación (credencial ausente,
# transporte desconocido, contrato sin tope) -> config_error; (c) sólo la
# llamada al proveedor en sí -> provider_error. Ruling R14: una sonda que no
# midió tokens no registra uso. Ruling R15: el tope también entra por
# motor_max_tokens.
# ---------------------------------------------------------------------------

def test_error_de_resolucion_se_propaga_sin_convertirse_en_veredicto(monkeypatch):
    """R13a/R16: el camino DIRECTO usa resolve_facet(), que levanta
    FacetUnavailableError (facet_resolver.py:325) tanto si la DB está caída
    como si la faceta no tiene binding -- sin distinción posible (R16), así
    que SIEMPRE se propaga tal cual, para que el llamador responda 503
    prevuelo_no_disponible (§8), no faceta_caida. Antes esta prueba usaba
    aiomysql.OperationalError directo, que no es lo que resolve_facet() real
    levanta -- ver fix round 2, item 2. Ni el registro de salud ni el de uso
    se llaman: no hay veredicto que guardar."""
    registrar, uso = AsyncMock(), AsyncMock()
    monkeypatch.setenv("JAX_PREVUELO_SONDA_MAX_TOKENS", "16")
    monkeypatch.setenv("JAX_PREVUELO_SONDA_TIMEOUT_S", "1")
    falla_de_resolucion = FacetUnavailableError("jekyll")
    with patch.object(sonda, "resolve_facet", AsyncMock(side_effect=falla_de_resolucion)), \
         patch.object(sonda.facet_health, "registrar_evento_de_sonda", registrar), \
         patch.object(sonda, "record_direct_usage", uso):
        try:
            asyncio.run(sonda.sondear("jekyll", _despacho()))
        except FacetUnavailableError as exc:
            assert exc is falla_de_resolucion
        else:
            raise AssertionError("sondear() debía propagar FacetUnavailableError, no tragarlo")
    registrar.assert_not_awaited()
    uso.assert_not_awaited()


def test_credencial_ausente_en_motor_es_config_error_no_provider_error(monkeypatch):
    """R13b/R16: CredentialUnavailableError cuya CAUSA es OTRO
    CredentialUnavailableError (fila genuinamente ausente, ver el
    comentario en el sitio) es una falla LOCAL de preparación (ni siquiera
    se intentó llamar al proveedor) -- outcome 'config_error', no
    'provider_error'. El lector de salud (OUTCOMES_DE_PROVEEDOR) ignora
    config_error, así que el próximo pre-vuelo vuelve a sondear. Mock a
    nivel de sonda.resolve_credential_instrumented (la cadena REAL de
    credential_resolver.py se ejercita en
    test_credencial_sin_fila_via_cadena_real_es_config_error, abajo)."""
    d = _despacho(clave_salud="kimi", via_motor=True, provider_id="moonshot", modelo="kimi-k3",
                  base_url="https://api.moonshot.example/v1", max_output_tokens=131072)
    registrar = AsyncMock()
    llamar = AsyncMock()
    sin_fila = CredentialUnavailableError("moonshot")
    sin_fila.__cause__ = CredentialUnavailableError("no active credential for moonshot")
    monkeypatch.setenv("JAX_PREVUELO_SONDA_MAX_TOKENS", "16")
    with patch("motor_registry.worker._call_http_openai_compat", llamar), \
         patch.object(sonda, "resolve_credential_instrumented", AsyncMock(side_effect=sin_fila)), \
         patch.object(sonda.facet_health, "registrar_evento_de_sonda", registrar), \
         patch.object(sonda, "record_direct_usage", AsyncMock()):
        r = asyncio.run(sonda.sondear("kimi", d))
    assert not r.ok
    assert "la sonda no pudo preparar la llamada" in r.detalle
    llamar.assert_not_awaited()
    registrar.assert_awaited_once_with("kimi", "config_error", r.detalle, ANY)


def test_credencial_con_causa_de_db_se_propaga_en_el_motor(monkeypatch):
    """Item 1 (fix round 2, Ruling R16): sobre la cadena REAL de
    credential_resolver -- se mockea SOLO `_query_active_credential`, no
    `resolve_credential_instrumented` -- para que resolve_credential()
    (credential_resolver.py:108-128) envuelva un aiomysql.OperationalError
    REAL en CredentialUnavailableError con esa causa
    (credential_resolver.py:120,128). sondear() tiene que propagar esa
    excepción tal cual, NO convertirla en config_error -- ver el rojo contra
    3ba6f71 en el reporte de fix round 2: ese commit la atrapaba sin mirar
    la causa."""
    import credential_resolver as cr
    provider_id = f"zz-test-db-{uuid.uuid4().hex[:8]}"
    falla_db = aiomysql.OperationalError(2003, "Can't connect to MySQL server")

    async def query_falla(pid):
        raise falla_db

    d = _despacho(clave_salud="kimi", via_motor=True, provider_id=provider_id, modelo="kimi-k3",
                  base_url="https://api.moonshot.example/v1", max_output_tokens=131072)
    registrar, uso = AsyncMock(), AsyncMock()
    monkeypatch.setenv("JAX_PREVUELO_SONDA_MAX_TOKENS", "16")
    monkeypatch.setenv("JAX_PREVUELO_SONDA_TIMEOUT_S", "1")
    with patch.object(cr, "_query_active_credential", query_falla), \
         patch.object(sonda.facet_health, "registrar_evento_de_sonda", registrar), \
         patch.object(sonda, "record_direct_usage", uso):
        try:
            asyncio.run(sonda.sondear("kimi", d))
        except CredentialUnavailableError as exc:
            assert exc.__cause__ is falla_db
        else:
            raise AssertionError("sondear() debía propagar la caída de DB, no tragarla")
    registrar.assert_not_awaited()
    uso.assert_not_awaited()


def test_credencial_sin_fila_via_cadena_real_es_config_error(monkeypatch):
    """Item 1 (fix round 2, Ruling R16), contraparte del test anterior sobre
    la misma cadena real: `_query_active_credential` levanta
    CredentialUnavailableError DIRECTO (fila ausente, sin `from`,
    credential_resolver.py:104) -- la causa de lo que sale de
    resolve_credential() es OTRO CredentialUnavailableError, la señal que
    Ruling R16 usa para decidir 'config_error'."""
    import credential_resolver as cr
    provider_id = f"zz-test-sinfila-{uuid.uuid4().hex[:8]}"

    async def query_sin_fila(pid):
        raise cr.CredentialUnavailableError(f"no active credential for {pid}")

    d = _despacho(clave_salud="kimi", via_motor=True, provider_id=provider_id, modelo="kimi-k3",
                  base_url="https://api.moonshot.example/v1", max_output_tokens=131072)
    registrar = AsyncMock()
    llamar = AsyncMock()
    monkeypatch.setenv("JAX_PREVUELO_SONDA_MAX_TOKENS", "16")
    with patch.object(cr, "_query_active_credential", query_sin_fila), \
         patch("motor_registry.worker._call_http_openai_compat", llamar), \
         patch.object(sonda.facet_health, "registrar_evento_de_sonda", registrar), \
         patch.object(sonda, "record_direct_usage", AsyncMock()):
        r = asyncio.run(sonda.sondear("kimi", d))
    assert not r.ok
    assert "la sonda no pudo preparar la llamada" in r.detalle
    llamar.assert_not_awaited()
    registrar.assert_awaited_once_with("kimi", "config_error", r.detalle, ANY)


def test_transporte_desconocido_es_config_error(monkeypatch):
    """R13b: un transporte que la sonda no sabe hablar es un hueco de
    configuración del catálogo, no una respuesta del proveedor."""
    f = _faceta(transport="grpc")
    registrar = AsyncMock()
    with patch.object(sonda, "resolve_facet", AsyncMock(return_value=f)), \
         patch.object(sonda.facet_health, "registrar_evento_de_sonda", registrar), \
         patch.object(sonda, "record_direct_usage", AsyncMock()):
        r = asyncio.run(sonda.sondear("jekyll", _despacho()))
    assert not r.ok
    assert "grpc" in r.detalle and "la sonda no pudo preparar la llamada" in r.detalle
    registrar.assert_awaited_once_with("jekyll", "config_error", r.detalle, ANY)


def test_sin_max_output_tokens_es_config_error(monkeypatch):
    """R13b: contrato sin tope de salida (max_output_tokens=None) -- la
    sonda no puede armar un límite de salida, así que no puede ni preguntar."""
    d = _despacho(max_output_tokens=None)
    registrar = AsyncMock()
    with patch.object(sonda, "resolve_facet", AsyncMock(return_value=_faceta())), \
         patch.object(sonda.facet_health, "registrar_evento_de_sonda", registrar), \
         patch.object(sonda, "record_direct_usage", AsyncMock()):
        r = asyncio.run(sonda.sondear("jekyll", d))
    assert not r.ok
    assert "max_output_tokens" in r.detalle
    registrar.assert_awaited_once_with("jekyll", "config_error", r.detalle, ANY)


def test_http_401_no_responde_y_es_provider_error(monkeypatch):
    """Item 2: un mutante que cambiara el chequeo de 2xx por `>= 500` deja
    pasar un 4xx como si fuera sano -- este test lo agarra."""
    registrar = AsyncMock()
    r, _ = _sondear(monkeypatch, _despacho(), _Resp(401, "no autorizado"), registrar=registrar)
    assert not r.ok
    assert "401" in r.detalle
    registrar.assert_awaited_once_with("jekyll", "provider_error", r.detalle, ANY)


def test_motor_401_es_provider_error(monkeypatch):
    """Item 2, camino del motor: un 401 real llega como httpx.HTTPStatusError
    desde _call_http_openai_compat (raise_for_status) -- tiene que seguir
    siendo provider_error, no config_error ni una excepción que se propaga."""
    resp = _RespuestaHTTPError(401, "no autorizado")
    llamar = AsyncMock(side_effect=httpx.HTTPStatusError("401", request=None, response=resp))
    d = _despacho(clave_salud="kimi", via_motor=True, provider_id="moonshot", modelo="kimi-k3",
                  base_url="https://api.moonshot.example/v1", max_output_tokens=131072)
    registrar = AsyncMock()
    monkeypatch.setenv("JAX_PREVUELO_SONDA_MAX_TOKENS", "16")
    with patch("motor_registry.worker._call_http_openai_compat", llamar), \
         patch.object(sonda, "resolve_credential_instrumented", AsyncMock(return_value="k-moon")), \
         patch.object(sonda.facet_health, "registrar_evento_de_sonda", registrar), \
         patch.object(sonda, "record_direct_usage", AsyncMock()):
        r = asyncio.run(sonda.sondear("kimi", d))
    assert not r.ok
    assert "401" in r.detalle
    registrar.assert_awaited_once_with("kimi", "provider_error", r.detalle, ANY)


def test_motor_4xx_redacta_la_credencial_en_el_detalle(monkeypatch):
    """Item 5: la lista de redacción del camino del motor (la credencial
    resuelta, no la de la faceta) está untested -- un error que la eco en el
    cuerpo no puede sobrevivir en el detalle guardado."""
    resp = _RespuestaHTTPError(401, "no autorizado, la key k-moon-secreta no sirve")
    llamar = AsyncMock(side_effect=httpx.HTTPStatusError("401", request=None, response=resp))
    d = _despacho(clave_salud="kimi", via_motor=True, provider_id="moonshot", modelo="kimi-k3",
                  base_url="https://api.moonshot.example/v1", max_output_tokens=131072)
    monkeypatch.setenv("JAX_PREVUELO_SONDA_MAX_TOKENS", "16")
    with patch("motor_registry.worker._call_http_openai_compat", llamar), \
         patch.object(sonda, "resolve_credential_instrumented", AsyncMock(return_value="k-moon-secreta")), \
         patch.object(sonda.facet_health, "registrar_evento_de_sonda", AsyncMock()), \
         patch.object(sonda, "record_direct_usage", AsyncMock()):
        r = asyncio.run(sonda.sondear("kimi", d))
    assert not r.ok
    assert "k-moon-secreta" not in r.detalle


def test_sin_wait_for_el_timeout_no_corta_la_llamada_colgada(monkeypatch):
    """Item 3: si se sacara el asyncio.wait_for que acota la llamada al
    proveedor, este test se cuelga 5s en vez de cortar a 1s -- visto a mano
    sacando el wait_for (ver el reporte de arreglos), no se deja así de
    forma permanente porque colgaría la corrida entera de CI."""
    async def post(self, url, headers=None, json=None, **kw):
        await asyncio.sleep(5)
        return _Resp(200, _OK_OPENAI)

    monkeypatch.setenv("JAX_PREVUELO_SONDA_MAX_TOKENS", "16")
    monkeypatch.setenv("JAX_PREVUELO_SONDA_TIMEOUT_S", "1")
    with patch("httpx.AsyncClient.post", post), \
         patch.object(sonda, "resolve_facet", AsyncMock(return_value=_faceta())), \
         patch.object(sonda.facet_health, "registrar_evento_de_sonda", AsyncMock()), \
         patch.object(sonda, "record_direct_usage", AsyncMock()):
        r = asyncio.run(sonda.sondear("jekyll", _despacho()))
    assert not r.ok and r.detalle == "timeout de sonda (1s)"


def test_ollama_directo_manda_num_predict(monkeypatch):
    """Item 4: el camino ollama NO via_motor (faceta resuelta con
    transport='ollama') estaba sin ningún test."""
    f = _faceta(key="jax_local", provider_id="ollama_local", base_url=None,
                model="qwen3-coder:30b", credential="", transport="ollama")
    d = _despacho(clave_salud="jax_local", transporte="ollama", provider_id="ollama_local",
                  modelo="qwen3-coder:30b", max_tokens_param=None, precio_in=None, precio_out=None)
    cuerpo = {"message": {"content": "ok"}, "prompt_eval_count": 7, "eval_count": 3}
    r, c = _sondear(monkeypatch, d, _Resp(200, cuerpo), faceta=f)
    assert r.ok
    from jacobs.executor import OLLAMA_URL
    assert c["url"] == OLLAMA_URL
    assert c["json"]["options"] == {"num_predict": 16}
    assert c["headers"] == {}


def test_una_sonda_fallida_no_registra_uso(monkeypatch):
    """Item 8: sin tokens medidos (una sonda que no respondió) no hay nada
    que cobrar -- record_direct_usage no se llama."""
    uso = AsyncMock()
    # Pasada final R34, 2: un 5xx pudo procesarse y cobrarse (registra
    # estimado, ver abajo); el caso "fallida y NO cobrable" es un 4xx.
    r, _ = _sondear(monkeypatch, _despacho(), _Resp(429, "cuota agotada"), uso=uso)
    assert not r.ok
    uso.assert_not_awaited()


def test_el_tope_de_motor_tambien_entra_en_el_minimo(monkeypatch):
    """R15: min(config, max_output_tokens, motor_max_tokens si via_motor y
    >0) -- antes el código ignoraba motor_max_tokens y mandaba el tope de
    config/catálogo aunque el motor pidiera menos."""
    d = _despacho(clave_salud="kimi", via_motor=True, provider_id="moonshot", modelo="kimi-k3",
                  base_url="https://api.moonshot.example/v1", max_output_tokens=131072,
                  motor_max_tokens=10)
    llamar = AsyncMock(return_value=_OK_OPENAI)
    monkeypatch.setenv("JAX_PREVUELO_SONDA_MAX_TOKENS", "16")
    with patch("motor_registry.worker._call_http_openai_compat", llamar), \
         patch.object(sonda, "resolve_credential_instrumented", AsyncMock(return_value="k-moon")), \
         patch.object(sonda.facet_health, "registrar_evento_de_sonda", AsyncMock()), \
         patch.object(sonda, "record_direct_usage", AsyncMock()):
        asyncio.run(sonda.sondear("kimi", d))
    assert llamar.await_args.kwargs["limite"] == {"max_tokens": 10}


@asynccontextmanager
async def _pool_que_explota():
    # R38 (fix round 1): usage_writer escribe por el pool del store.
    raise OSError("sin base")
    yield


def test_record_direct_usage_encola_con_el_request_type_pedido(monkeypatch):
    # Host y puerto de mentira: sin ellos _db_cfg() lanza ANTES de conectar y
    # el test no ejercitaría el camino de la conexión caída.
    monkeypatch.setenv("JAX_DB_HOST", "127.0.0.1")
    monkeypatch.setenv("JAX_DB_PORT", "1")
    encolar = AsyncMock(return_value="spool-1")
    with patch.object(usage_writer.store, "conexion_del_pool", _pool_que_explota), \
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

    @asynccontextmanager
    async def pool_ok():
        yield _Conn()

    with patch.object(usage_writer.store, "conexion_del_pool", pool_ok):
        asyncio.run(usage_writer.record_direct_usage(
            "1", "1", "jekyll", "deepseek", "m", 1, 2, request_type="preflight_probe"))
        asyncio.run(usage_writer.record_direct_usage("1", "1", "jekyll", "deepseek", "m", 1, 2))
    inserts = [p for sql, p in ejecutados if sql.startswith("INSERT INTO axioma_usage")]
    assert [p[-1] for p in inserts] == ["preflight_probe", "pipeline"]


# ---------------------------------------------------------------------------
# F4 (ola final, Ruling R30 revierte R14): una sonda que VENCE o un 2xx SIN
# usage pudo cobrarse y no dejaba fila en axioma_usage. Ahora se registra con
# tokens ESTIMADOS -- entrada: ⌈len(MENSAJE_DE_SONDA) / chars_por_token⌉;
# salida: el tope que pidió la sonda -- y request_type='preflight_probe_est'
# (la marca de estimación: axioma_usage no tiene otra columna para eso). Una
# falla local (config_error), un error HTTP del proveedor o la base caída al
# resolver NO registran: no hubo llamada cobrable.
# ---------------------------------------------------------------------------

import math  # noqa: E402

import pytest  # noqa: E402


def _estimado(tope=16, chars_por_token=2):
    return (math.ceil(len(sonda.MENSAJE_DE_SONDA) / chars_por_token), tope)


def test_request_type_estimado_entra_en_la_columna():
    assert sonda.REQUEST_TYPE_SONDA_ESTIMADA == "preflight_probe_est"
    assert len(sonda.REQUEST_TYPE_SONDA_ESTIMADA) <= 20  # axioma_usage.request_type VARCHAR(20)


def test_sonda_que_vence_registra_uso_estimado(monkeypatch):
    uso = AsyncMock()
    r, _ = _sondear(monkeypatch, _despacho(), httpx.ReadTimeout("lento"), uso=uso)
    assert not r.ok
    uso.assert_awaited_once_with(None, None, "jekyll", "deepseek", "deepseek-v4-flash", *_estimado(),
                                 request_type="preflight_probe_est")


def test_sonda_cortada_por_wait_for_registra_uso_estimado(monkeypatch):
    async def post(self, url, headers=None, json=None, **kw):
        await asyncio.sleep(5)
        return _Resp(200, _OK_OPENAI)

    uso = AsyncMock()
    monkeypatch.setenv("JAX_PREVUELO_SONDA_MAX_TOKENS", "16")
    monkeypatch.setenv("JAX_PREVUELO_SONDA_TIMEOUT_S", "1")
    monkeypatch.setenv("JAX_PREVUELO_CHARS_POR_TOKEN", "3")
    with patch("httpx.AsyncClient.post", post), \
         patch.object(sonda, "resolve_facet", AsyncMock(return_value=_faceta())), \
         patch.object(sonda.facet_health, "registrar_evento_de_sonda", AsyncMock()), \
         patch.object(sonda, "record_direct_usage", uso):
        r = asyncio.run(sonda.sondear("jekyll", _despacho(max_output_tokens=8)))
    assert not r.ok
    uso.assert_awaited_once_with(None, None, "jekyll", "deepseek", "deepseek-v4-flash",
                                 *_estimado(tope=8, chars_por_token=3), request_type="preflight_probe_est")


def test_2xx_sin_usage_registra_uso_estimado_y_sigue_sana(monkeypatch):
    uso = AsyncMock()
    cuerpo = {"choices": [{"message": {"content": "ok"}}]}
    r, _ = _sondear(monkeypatch, _despacho(), _Resp(200, cuerpo), uso=uso)
    assert r.ok
    uso.assert_awaited_once_with(None, None, "jekyll", "deepseek", "deepseek-v4-flash", *_estimado(),
                                 request_type="preflight_probe_est")


def test_gemini_2xx_sin_usage_metadata_registra_uso_estimado(monkeypatch):
    f = _faceta(key="hipatia", provider_id="gemini", base_url="https://g.example/v1beta",
                model="gemini-x", credential="AIza-secreta", transport="http_gemini")
    d = _despacho(clave_salud="hipatia", transporte="http_gemini", provider_id="gemini",
                  modelo="gemini-x", max_tokens_param=None, max_output_tokens=65536)
    uso = AsyncMock()
    r, _ = _sondear(monkeypatch, d, _Resp(200, {"candidates": []}), faceta=f, uso=uso)
    assert r.ok
    uso.assert_awaited_once_with(None, None, "hipatia", "gemini", "gemini-x", *_estimado(),
                                 request_type="preflight_probe_est")


def test_motor_2xx_sin_usage_registra_uso_estimado(monkeypatch):
    d = _despacho(clave_salud="kimi", via_motor=True, provider_id="moonshot", modelo="kimi-k3",
                  base_url="https://api.moonshot.example/v1", max_output_tokens=131072)
    uso = AsyncMock()
    monkeypatch.setenv("JAX_PREVUELO_SONDA_MAX_TOKENS", "16")
    with patch("motor_registry.worker._call_http_openai_compat",
               AsyncMock(return_value={"choices": [{"message": {"content": "ok"}}]})), \
         patch.object(sonda, "resolve_credential_instrumented", AsyncMock(return_value="k-moon")), \
         patch.object(sonda.facet_health, "registrar_evento_de_sonda", AsyncMock()), \
         patch.object(sonda, "record_direct_usage", uso):
        r = asyncio.run(sonda.sondear("kimi", d, user_id="7", tenant_id="1"))
    assert r.ok
    uso.assert_awaited_once_with("7", "1", "kimi", "moonshot", "kimi-k3", *_estimado(),
                                 request_type="preflight_probe_est")


def test_2xx_con_usage_medido_sigue_como_preflight_probe(monkeypatch):
    """Control: con tokens medidos no se estima nada."""
    uso = AsyncMock()
    _sondear(monkeypatch, _despacho(), _Resp(200, _OK_OPENAI), uso=uso)
    assert uso.await_args.kwargs == {"request_type": "preflight_probe"}


def test_falla_local_no_registra_uso(monkeypatch):
    uso = AsyncMock()
    r, _ = _sondear(monkeypatch, _despacho(max_output_tokens=None), _Resp(200, _OK_OPENAI), uso=uso)
    assert not r.ok
    uso.assert_not_awaited()


def test_base_caida_al_resolver_no_registra_uso(monkeypatch):
    uso = AsyncMock()
    monkeypatch.setenv("JAX_PREVUELO_SONDA_TIMEOUT_S", "1")
    with patch.object(sonda, "resolve_facet", AsyncMock(side_effect=FacetUnavailableError("base caída"))), \
         patch.object(sonda.facet_health, "registrar_evento_de_sonda", AsyncMock()), \
         patch.object(sonda, "record_direct_usage", uso):
        with pytest.raises(FacetUnavailableError):
            asyncio.run(sonda.sondear("jekyll", _despacho()))
    uso.assert_not_awaited()


# ---------------------------------------------------------------------------
# Pasada final R34, 2: la regla de cobro de F4 se decide por si el pedido
# SALIÓ hacia el proveedor. No salió (nada que registrar): ConnectError y
# ConnectTimeout (no hubo conexión), PoolTimeout (nunca obtuvo conexión),
# UnsupportedProtocol (se rechaza antes de conectar) y un 4xx (el proveedor lo
# rechazó sin procesarlo). Salió y no hay uso medido (estimado): ReadTimeout y
# WriteTimeout, ReadError, RemoteProtocolError y cualquier 5xx, gateways
# 504/524 incluidos.
# ---------------------------------------------------------------------------

_NO_SALIO = [
    httpx.ConnectError("sin ruta"),
    httpx.ConnectTimeout("no conectó"),
    httpx.PoolTimeout("sin conexión libre"),
    httpx.UnsupportedProtocol("esquema raro"),
]
_SALIO = [
    httpx.ReadTimeout("lento"),
    httpx.WriteTimeout("lento al escribir"),
    httpx.ReadError("cortó a mitad"),
    httpx.RemoteProtocolError("respuesta rota"),
]


@pytest.mark.parametrize("error", _NO_SALIO, ids=lambda e: type(e).__name__)
def test_pedido_que_no_salio_no_registra_uso(monkeypatch, error):
    uso = AsyncMock()
    r, _ = _sondear(monkeypatch, _despacho(), error, uso=uso)
    assert not r.ok
    uso.assert_not_awaited()


@pytest.mark.parametrize("error", _SALIO, ids=lambda e: type(e).__name__)
def test_pedido_que_salio_sin_uso_registra_estimado(monkeypatch, error):
    uso = AsyncMock()
    r, _ = _sondear(monkeypatch, _despacho(), error, uso=uso)
    assert not r.ok
    uso.assert_awaited_once_with(None, None, "jekyll", "deepseek", "deepseek-v4-flash", *_estimado(),
                                 request_type="preflight_probe_est")


@pytest.mark.parametrize("status", [400, 401, 404, 429])
def test_4xx_no_registra_uso(monkeypatch, status):
    uso = AsyncMock()
    r, _ = _sondear(monkeypatch, _despacho(), _Resp(status, "rechazado"), uso=uso)
    assert not r.ok and str(status) in r.detalle
    uso.assert_not_awaited()


@pytest.mark.parametrize("status", [500, 502, 503, 504, 524])
def test_5xx_registra_uso_estimado(monkeypatch, status):
    uso = AsyncMock()
    r, _ = _sondear(monkeypatch, _despacho(), _Resp(status, "falla del lado del proveedor"), uso=uso)
    assert not r.ok and str(status) in r.detalle
    uso.assert_awaited_once_with(None, None, "jekyll", "deepseek", "deepseek-v4-flash", *_estimado(),
                                 request_type="preflight_probe_est")


@pytest.mark.parametrize("status,cobrable", [(401, False), (524, True)])
def test_motor_decide_el_cobro_por_el_status(monkeypatch, status, cobrable):
    d = _despacho(clave_salud="kimi", via_motor=True, provider_id="moonshot", modelo="kimi-k3",
                  base_url="https://api.moonshot.example/v1", max_output_tokens=131072)
    resp = _RespuestaHTTPError(status, "respuesta")
    uso = AsyncMock()
    monkeypatch.setenv("JAX_PREVUELO_SONDA_MAX_TOKENS", "16")
    with patch("motor_registry.worker._call_http_openai_compat",
               AsyncMock(side_effect=httpx.HTTPStatusError(str(status), request=None, response=resp))), \
         patch.object(sonda, "resolve_credential_instrumented", AsyncMock(return_value="k-moon")), \
         patch.object(sonda.facet_health, "registrar_evento_de_sonda", AsyncMock()), \
         patch.object(sonda, "record_direct_usage", uso):
        r = asyncio.run(sonda.sondear("kimi", d))
    assert not r.ok
    assert uso.await_count == (1 if cobrable else 0)


# ---------------------------------------------------------------------------
# Pasada R37, 1: wait_for y httpx usaban el MISMO timeout y el reloj de
# wait_for arranca antes, así que un connect colgado salía como TimeoutError
# (cobrado) y la clasificación ConnectTimeout (no cobrado) era código muerto.
# wait_for lleva ahora un margen declarado por encima del timeout de httpx.
# ---------------------------------------------------------------------------

def test_connect_colgado_sale_como_connecttimeout_y_no_se_cobra(monkeypatch):
    async def post(self, url, headers=None, json=None, **kw):
        # Lo que hace httpx con timeout=1: espera su timeout de connect (un
        # poco más, por el trabajo alrededor) y levanta ConnectTimeout.
        await asyncio.sleep(1.2)
        raise httpx.ConnectTimeout("connect colgado")

    uso = AsyncMock()
    monkeypatch.setenv("JAX_PREVUELO_SONDA_MAX_TOKENS", "16")
    monkeypatch.setenv("JAX_PREVUELO_SONDA_TIMEOUT_S", "1")
    with patch("httpx.AsyncClient.post", post), \
         patch.object(sonda, "resolve_facet", AsyncMock(return_value=_faceta())), \
         patch.object(sonda.facet_health, "registrar_evento_de_sonda", AsyncMock()), \
         patch.object(sonda, "record_direct_usage", uso):
        r = asyncio.run(sonda.sondear("jekyll", _despacho()))
    assert not r.ok and r.detalle == "timeout de sonda (1s)"
    uso.assert_not_awaited()
