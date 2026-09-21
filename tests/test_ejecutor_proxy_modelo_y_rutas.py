# tests/test_ejecutor_proxy_modelo_y_rutas.py
"""SP3 (2026-09-17): la jaula comparte la GPU con la Mesa. El proxy FIJA el modelo y el tope
de salida, y sólo deja pasar lo que el arnés manda de verdad.

- Otro `model` cargaría otro modelo en Ollama y desalojaría el de la Mesa (medido en SP3: una
  alternancia de contexto recarga ~3 s y la Mesa espera). Se corta con 403, sin tocar Ollama.
- `max_tokens` por encima del tope rompe la cuenta de la espera de la Mesa (§6.3 del spec de
  Fase 2): una petición en curso no se interrumpe, así que lo que genera la cuenta.
- GET/HEAD arbitrarios: el upstream es el Ollama de producción (`/api/tags`, `/api/ps`,
  `/api/show` dicen qué hay cargado y con qué). Medido con el arnés 2.1.273 por el proxy
  (g1_20260917/proxy_v3.jsonl, 43 peticiones): sólo `HEAD /api/hello` y `POST /v1/messages`.
"""
import json

import httpx
import pytest

from jax.ejecutor import proxy_carril
from tests.test_ejecutor_proxy_carril import MAX_SALIDA_TOKENS, MODELO_PERMITIDO, Proxy, Upstream, _correr


def _registro(tmp_path):
    return [json.loads(l)["evento"] for l in (tmp_path / "registro.jsonl").read_text().splitlines()]


def _pedir(tmp_path, metodo, ruta, cuerpo=None):
    async def escenario():
        async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 2) as px, httpx.AsyncClient() as cli:
            r = await cli.request(metodo, px.url + ruta, content=cuerpo)
            codigo = r.json()["error"]["type"] if r.status_code >= 400 and r.content else None
            return r.status_code, codigo, len(up.recibidas)

    return _correr(escenario())


def _mensaje(**cambios):
    base = {"model": MODELO_PERMITIDO, "max_tokens": MAX_SALIDA_TOKENS, "stream": True,
            "messages": [{"role": "user", "content": "hola"}]}
    base.update(cambios)
    return json.dumps({k: v for k, v in base.items() if v is not _FUERA}).encode()


_FUERA = object()


@pytest.mark.parametrize("modelo", [
    # Dos nombres de modelo REALES (2026-09-21): no importa CUÁLES sean --
    # lo único que este caso prueba es "cualquier cosa que no sea
    # MODELO_PERMITIDO se rechaza", así que no dependen del catálogo y no se
    # rompen si esos dos modelos cambian o se deprecan. Reales a propósito
    # (no un literal inventado tipo "otro-modelo"): un cliente mal
    # configurado va a pedir un nombre real, no uno de mentira.
    "qwen3.6:35b-a3b-q4_K_M", "claude-haiku-4-5",
    "modelo-permitido ", "MODELO-PERMITIDO",  # variantes de MODELO_PERMITIDO: espacio/mayúsculas NO cuentan como el mismo modelo
    "", None, 7, _FUERA])
def test_otro_modelo_no_llega_a_ollama(tmp_path, modelo):
    assert _pedir(tmp_path, "POST", "/v1/messages", _mensaje(model=modelo)) == (
        403, proxy_carril.MODELO_NO_PERMITIDO, 0)
    assert _registro(tmp_path) == ["registro_abierto"], "una petición rechazada no se anota como resultado"


@pytest.mark.parametrize("cuerpo", [b"", b"{", b"[]", b'"modelo-permitido"', b"\x1f\x8b\x08\x00"])
def test_cuerpo_que_no_se_puede_leer_no_llega_a_ollama(tmp_path, cuerpo):
    # Un cuerpo comprimido (CLAUDE_CODE_GZIP_REQUEST_BODIES existe en el arnés) tampoco: lo
    # que el proxy no puede leer, no sabe qué modelo pide.
    assert _pedir(tmp_path, "POST", "/v1/messages", cuerpo) == (403, proxy_carril.MODELO_NO_PERMITIDO, 0)


@pytest.mark.parametrize("max_tokens", [MAX_SALIDA_TOKENS + 1, 32000, 0, -1, True, 1024.0, "1024", None, _FUERA])
def test_salida_fuera_del_tope_no_llega_a_ollama(tmp_path, max_tokens):
    assert _pedir(tmp_path, "POST", "/v1/messages", _mensaje(max_tokens=max_tokens)) == (
        403, proxy_carril.SALIDA_NO_PERMITIDA, 0)


@pytest.mark.parametrize("max_tokens", [1, MAX_SALIDA_TOKENS])
def test_el_modelo_permitido_dentro_del_tope_si_llega(tmp_path, max_tokens):
    assert _pedir(tmp_path, "POST", "/v1/messages?beta=true", _mensaje(max_tokens=max_tokens)) == (200, None, 1)


@pytest.mark.parametrize("metodo, ruta", [
    ("GET", "/api/tags"), ("GET", "/api/ps"), ("GET", "/api/version"), ("GET", "/"),
    ("GET", "/api/hello"), ("GET", "/v1/models"), ("POST", "/v1/messages/count_tokens"),
    ("OPTIONS", "/v1/messages"), ("POST", "/api/show"),
])
def test_ningun_get_arbitrario_llega_a_ollama(tmp_path, metodo, ruta):
    assert _pedir(tmp_path, metodo, ruta) == (403, proxy_carril.RUTA_NO_PERMITIDA, 0)


@pytest.mark.parametrize("ruta", ["/api/tags", "/api/ps", "/api/hello/../tags", "/"])
def test_ningun_head_arbitrario_llega_a_ollama_y_el_error_sale_sin_cuerpo(tmp_path, caplog, ruta):
    # Una respuesta a HEAD no lleva cuerpo (RFC 9110 §9.3.2): mandarlo hacía que h11 lanzara
    # dentro del handler ("Unhandled exception in client_connected_cb") con las cabeceras ya
    # enviadas. Pasaba también con el 503 de espera agotada a un HEAD /api/hello (proxy_v3.log).
    assert _pedir(tmp_path, "HEAD", ruta) == (403, None, 0)
    assert "Unhandled exception" not in caplog.text


def test_head_api_hello_si_llega(tmp_path):
    estado, _, n = _pedir(tmp_path, "HEAD", "/api/hello")
    assert (estado, n) == (200, 1)
