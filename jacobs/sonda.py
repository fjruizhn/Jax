"""Jacobs — sonda de disponibilidad del pre-vuelo (spec 2026-09-17 §4.5, §8).

Una llamada MÍNIMA por el transporte real de la clave: mismo endpoint,
resolver y credencial que el ejecutor, pero payload propio (sin google_search
y con el menor entre JAX_PREVUELO_SONDA_MAX_TOKENS y el tope del catálogo):
las _invoke_* mandarían herramientas y el tope completo del modelo.

«Responde» = 2xx del proveedor aunque el contenido venga cortado: mide
disponibilidad, no calidad. Timeout propio (JAX_PREVUELO_SONDA_TIMEOUT_S).

Cada resultado se registra en facet_health_event (source='preflight', outcome
ok/provider_error) para que el próximo pre-vuelo dentro de la ventana no
vuelva a sondear, y su uso en axioma_usage (request_type='preflight_probe'):
se paga y se ve. Si el registro de salud falla, el veredicto se mantiene (el
dato es la respuesta del proveedor, no la fila), se loguea WARNING y se cuenta.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx
from credential_resolver import resolve_credential_instrumented
from facet_resolver import resolve_facet
from redaccion import recortar_redactado

from jacobs import facet_health, prevuelo_config
from jacobs.prevuelo_reglas import Despacho
from jacobs.usage_writer import record_direct_usage

logger = logging.getLogger("jacobs.sonda")

MENSAJE_DE_SONDA = "Respondé solamente: ok"
REQUEST_TYPE_SONDA = "preflight_probe"
_LARGO_DETALLE = 200


@dataclass(frozen=True)
class ResultadoSonda:
    ok: bool
    detalle: str | None
    tokens_in: int = 0
    tokens_out: int = 0


_registros_perdidos = 0


def registros_perdidos() -> int:
    """Eventos de sonda que no se pudieron escribir desde que arrancó el proceso."""
    return _registros_perdidos


async def _post(url: str, headers: dict, payload: dict, timeout: int, secretos: list[str]) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
    if not 200 <= resp.status_code < 300:
        raise RuntimeError(f"HTTP {resp.status_code}: {recortar_redactado(resp.text, _LARGO_DETALLE, secretos)}")
    return resp.json()


async def _llamar(clave: str, d: Despacho, timeout: int) -> ResultadoSonda:
    limite = min(prevuelo_config.sonda_max_tokens(), d.max_output_tokens)
    mensajes = [{"role": "user", "content": MENSAJE_DE_SONDA}]

    if d.via_motor:
        # El MISMO cliente que worker.py usa para despachar el motor.
        from motor_registry import worker
        api_key = "" if d.transporte == "ollama" else await resolve_credential_instrumented(d.provider_id)
        campo = "max_tokens" if d.transporte == "ollama" else d.max_tokens_param
        try:
            data = await worker._call_http_openai_compat(
                api_url=d.base_url, model=d.modelo, api_key=api_key,
                messages=mensajes, timeout=timeout, limite={campo: limite},
            )
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"HTTP {exc.response.status_code}: "
                f"{recortar_redactado(exc.response.text, _LARGO_DETALLE, [api_key])}"
            ) from exc
        uso = data.get("usage") or {}
        return ResultadoSonda(True, None, uso.get("prompt_tokens", 0), uso.get("completion_tokens", 0))

    f = await resolve_facet(clave)
    if f.transport == "http_gemini":
        data = await _post(
            f"{f.base_url}/models/{f.model}:generateContent",
            {"x-goog-api-key": f.credential},
            {"contents": [{"role": "user", "parts": [{"text": MENSAJE_DE_SONDA}]}],
             "generationConfig": {"maxOutputTokens": limite}},
            timeout, [f.credential],
        )
        uso = data.get("usageMetadata") or {}
        return ResultadoSonda(True, None, uso.get("promptTokenCount", 0), uso.get("candidatesTokenCount", 0))
    if f.transport == "http_openai_compat":
        data = await _post(
            f"{f.base_url}/chat/completions",
            {"Authorization": f"Bearer {f.credential}", "Content-Type": "application/json"},
            {"model": f.model, "messages": mensajes, "stream": False, d.max_tokens_param: limite},
            timeout, [f.credential],
        )
        uso = data.get("usage") or {}
        return ResultadoSonda(True, None, uso.get("prompt_tokens", 0), uso.get("completion_tokens", 0))
    if f.transport == "ollama":
        from jacobs.executor import OLLAMA_URL
        data = await _post(
            OLLAMA_URL, {},
            {"model": f.model, "messages": mensajes, "stream": False, "options": {"num_predict": limite}},
            timeout, [],
        )
        return ResultadoSonda(True, None, data.get("prompt_eval_count", 0), data.get("eval_count", 0))
    raise RuntimeError(f"transporte '{f.transport}' de '{clave}' no tiene sonda")


async def _registrar(clave: str, d: Despacho, r: ResultadoSonda,
                     user_id: str | None, tenant_id: str | None) -> None:
    global _registros_perdidos
    outcome = "ok" if r.ok else "provider_error"
    try:
        await facet_health.registrar_evento_de_sonda(clave, outcome, r.detalle, time.time())
    except Exception as exc:  # fail-soft: el veredicto sale de la respuesta del proveedor, no de esta fila; sin ella el próximo pre-vuelo vuelve a sondear, y la pérdida queda contada en registros_perdidos() y en el WARNING
        _registros_perdidos += 1
        logger.warning(
            "pre-vuelo: no se pudo registrar la sonda de '%s' en facet_health_event (%s); "
            "van %d registros perdidos",
            clave, recortar_redactado(f"{type(exc).__name__}: {exc}", _LARGO_DETALLE), _registros_perdidos,
        )
    if r.tokens_in or r.tokens_out:
        await record_direct_usage(
            user_id, tenant_id, clave, d.provider_id, d.modelo, r.tokens_in, r.tokens_out,
            request_type=REQUEST_TYPE_SONDA,
        )


async def sondear(clave: str, d: Despacho, *, user_id: str | None = None,
                  tenant_id: str | None = None) -> ResultadoSonda:
    timeout = prevuelo_config.sonda_timeout_s()
    try:
        resultado = await asyncio.wait_for(_llamar(clave, d, timeout), timeout=timeout)
    except (asyncio.TimeoutError, httpx.TimeoutException):
        resultado = ResultadoSonda(False, f"timeout de sonda ({timeout}s)")
    except Exception as exc:  # fail-closed: una sonda que no completó la llamada da faceta_caida, nunca sana; el motivo redactado viaja en el veredicto
        resultado = ResultadoSonda(False, recortar_redactado(f"{type(exc).__name__}: {exc}", _LARGO_DETALLE))
    await _registrar(clave, d, resultado, user_id, tenant_id)
    return resultado
