"""Jacobs — sonda de disponibilidad del pre-vuelo (spec 2026-09-17 §4.5, §8).

Una llamada MÍNIMA por el transporte real de la clave: mismo endpoint,
resolver y credencial que el ejecutor, pero payload propio (sin google_search
y con el menor entre JAX_PREVUELO_SONDA_MAX_TOKENS, el tope del catálogo y
-- via motor -- el presupuesto propio del motor, Ruling R15): las _invoke_*
mandarían herramientas y el tope completo del modelo.

«Responde» = 2xx del proveedor aunque el contenido venga cortado: mide
disponibilidad, no calidad. Timeout propio (JAX_PREVUELO_SONDA_TIMEOUT_S),
y acota SOLO la llamada al proveedor (Ruling R13): resolver la faceta o la
credencial tiene sus propios timeouts de conexión (tripwire aiomysql).

Tres clases de resultado (Ruling R13, fix round 1 de la revisión de Task 7):
  (a) una caída real de la DB al resolver (FacetUnavailableError -- que ya
      envuelve aiomysql/pymysql/OSError -- o cualquier excepción que no sea
      una de las dos siguientes) se PROPAGA fuera de sondear(): jacobs no
      pudo ni ver su propio catálogo, así que no tiene nada que decir sobre
      el proveedor -- el llamador responde 503 prevuelo_no_disponible (§8),
      nunca faceta_caida.
  (b) una falla LOCAL de preparación (transporte desconocido, contrato sin
      tope de salida, o CredentialUnavailableError del camino motor cuya
      CAUSA muestra que la fila de credencial genuinamente no existe --
      Ruling R16, ver el comentario en el sitio) -> ok=False, outcome
      'config_error': no dice nada del proveedor, así que el próximo
      pre-vuelo dentro de la ventana vuelve a sondear (el lector de salud,
      OUTCOMES_DE_PROVEEDOR, lo ignora). En el camino DIRECTO (resolve_facet)
      no hay nada que distinguir: FacetUnavailableError envuelve tanto DB
      caída como credencial ausente sin dejar señal aparte (R16), así que
      SIEMPRE se propaga -- cae en (a).
  (c) sólo la llamada al proveedor en sí (2xx fallido, error HTTP, timeout)
      -> outcome 'provider_error'.

Cada resultado se registra en facet_health_event (source='preflight', outcome
ok/provider_error/config_error) y, si midió tokens, su uso en axioma_usage
(request_type='preflight_probe'): se paga y se ve. Si el registro de salud
falla, el veredicto se mantiene (el dato es la respuesta del proveedor, no
la fila), se loguea WARNING y se cuenta.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx
from credential_resolver import CredentialUnavailableError, resolve_credential_instrumented
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


class _FallaDePreparacion(Exception):
    """Fallo LOCAL antes de tocar al proveedor: transporte desconocido,
    contrato sin tope de salida (max_output_tokens=None), o -- SOLO en el
    camino motor -- CredentialUnavailableError cuya CAUSA es otro
    CredentialUnavailableError (Ruling R16: _query_active_credential la
    levanta directo cuando la fila no existe, credential_resolver.py:104;
    ese es el único caso genuinamente local). Se registra con outcome
    'config_error'. Cualquier OTRA excepción durante la preparación (p.ej.
    FacetUnavailableError del camino directo, que envuelve TANTO una DB
    caída como una credencial ausente sin dejar señal para distinguirlas, o
    un CredentialUnavailableError del motor cuya causa es una caída real de
    DB) NO se atrapa acá: se propaga fuera de sondear()."""


def _tope(d: Despacho) -> int:
    """Ruling R15: el mismo menor-de-tres que worker._limite_del_motor --
    config, catálogo y (sólo via motor, y sólo si el motor declara uno
    propio) el presupuesto del motor."""
    if d.max_output_tokens is None:
        raise _FallaDePreparacion("el catálogo no declara max_output_tokens")
    limite = min(prevuelo_config.sonda_max_tokens(), d.max_output_tokens)
    if d.via_motor and d.motor_max_tokens:
        limite = min(limite, d.motor_max_tokens)
    return limite


async def _post(url: str, headers: dict, payload: dict, timeout: int, secretos: list[str]) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
    if not 200 <= resp.status_code < 300:
        raise RuntimeError(f"HTTP {resp.status_code}: {recortar_redactado(resp.text, _LARGO_DETALLE, secretos)}")
    return resp.json()


async def _preparar(clave: str, d: Despacho):
    """Resuelve todo lo que la llamada real necesita (faceta o credencial del
    motor) y devuelve un cerrado (closure) `async def (timeout) -> ResultadoSonda`
    que HACE la llamada. Nada de esto corre bajo el timeout de la sonda --
    ver Ruling R13 y el docstring del módulo. Levanta _FallaDePreparacion
    para las tres fallas LOCALES conocidas; cualquier otra excepción (p.ej.
    una caída real de la DB al resolver faceta o credencial) se propaga tal
    cual, sin envolver."""
    limite = _tope(d)
    mensajes = [{"role": "user", "content": MENSAJE_DE_SONDA}]

    if d.via_motor:
        # El MISMO cliente que worker.py usa para despachar el motor.
        from motor_registry import worker
        if d.transporte == "ollama":
            api_key, campo = "", "max_tokens"
        else:
            try:
                api_key = await resolve_credential_instrumented(d.provider_id)
            except CredentialUnavailableError as exc:
                # Ruling R16: resolve_credential() (las_manos/credential_resolver.py:108-128)
                # SIEMPRE envuelve en CredentialUnavailableError, tanto si la
                # fila no existe (_query_active_credential la levanta directo,
                # línea 104, SIN causa) como si la DB está caída (el `except
                # Exception as e` de la línea 120 atrapa cualquier otra cosa
                # -- un aiomysql.OperationalError real, por ejemplo -- y la
                # re-envuelve con `from e` en la línea 128).
                # resolve_credential_instrumented (líneas 131-146) reintenta
                # por .env y, si tampoco hay valor, relanza la MISMA excepción
                # con `raise` pelado -- la cadena de causa no se pierde.
                # La única señal que distingue los dos casos es esa causa: si
                # es OTRO CredentialUnavailableError, la fila genuinamente no
                # existe -> config_error. Cualquier otra causa (una caída real
                # de DB) tiene que propagarse -- 503, no faceta_caida.
                if isinstance(exc.__cause__, CredentialUnavailableError):
                    raise _FallaDePreparacion(f"sin credencial activa para '{d.provider_id}': {exc}") from exc
                raise
            campo = d.max_tokens_param

        async def _llamada(timeout: int) -> ResultadoSonda:
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

        return _llamada

    f = await resolve_facet(clave)

    if f.transport == "http_gemini":
        async def _llamada(timeout: int) -> ResultadoSonda:
            data = await _post(
                f"{f.base_url}/models/{f.model}:generateContent",
                {"x-goog-api-key": f.credential},
                {"contents": [{"role": "user", "parts": [{"text": MENSAJE_DE_SONDA}]}],
                 "generationConfig": {"maxOutputTokens": limite}},
                timeout, [f.credential],
            )
            uso = data.get("usageMetadata") or {}
            return ResultadoSonda(True, None, uso.get("promptTokenCount", 0), uso.get("candidatesTokenCount", 0))
        return _llamada

    if f.transport == "http_openai_compat":
        async def _llamada(timeout: int) -> ResultadoSonda:
            data = await _post(
                f"{f.base_url}/chat/completions",
                {"Authorization": f"Bearer {f.credential}", "Content-Type": "application/json"},
                {"model": f.model, "messages": mensajes, "stream": False, d.max_tokens_param: limite},
                timeout, [f.credential],
            )
            uso = data.get("usage") or {}
            return ResultadoSonda(True, None, uso.get("prompt_tokens", 0), uso.get("completion_tokens", 0))
        return _llamada

    if f.transport == "ollama":
        from jacobs.executor import OLLAMA_URL

        async def _llamada(timeout: int) -> ResultadoSonda:
            data = await _post(
                OLLAMA_URL, {},
                {"model": f.model, "messages": mensajes, "stream": False, "options": {"num_predict": limite}},
                timeout, [],
            )
            return ResultadoSonda(True, None, data.get("prompt_eval_count", 0), data.get("eval_count", 0))
        return _llamada

    raise _FallaDePreparacion(f"transporte '{f.transport}' de '{clave}' no tiene sonda")


async def _registrar(clave: str, d: Despacho, r: ResultadoSonda, outcome: str,
                     user_id: str | None, tenant_id: str | None) -> None:
    global _registros_perdidos
    try:
        await facet_health.registrar_evento_de_sonda(clave, outcome, r.detalle, time.time())
    except Exception as exc:  # fail-soft: el veredicto sale de la respuesta del proveedor, no de esta fila; sin ella el próximo pre-vuelo vuelve a sondear, y la pérdida queda contada en registros_perdidos() y en el WARNING
        _registros_perdidos += 1
        logger.warning(
            "pre-vuelo: no se pudo registrar la sonda de '%s' en facet_health_event (%s); "
            "van %d registros perdidos",
            clave, recortar_redactado(f"{type(exc).__name__}: {exc}", _LARGO_DETALLE), _registros_perdidos,
        )
    # Ruling R14 (2026-09-17, fix round 1 de la revisión de Task 7): un 2xx
    # SIN campo `usage` (proveedor que no lo declara) deja tokens_in=
    # tokens_out=0 y NO se registra en axioma_usage -- no hay tokens medidos
    # que cobrar, y una fila con ceros no es "se cobró $0", es "no se sabe".
    # Caso agregado a la entrada de DEUDA.md de la Task 14 ("sonda que vence
    # sin registrar uso"), misma fecha: mientras tanto ese 2xx queda sin fila.
    if r.tokens_in or r.tokens_out:
        await record_direct_usage(
            user_id, tenant_id, clave, d.provider_id, d.modelo, r.tokens_in, r.tokens_out,
            request_type=REQUEST_TYPE_SONDA,
        )


async def sondear(clave: str, d: Despacho, *, user_id: str | None = None,
                  tenant_id: str | None = None) -> ResultadoSonda:
    timeout = prevuelo_config.sonda_timeout_s()
    try:
        llamada = await _preparar(clave, d)
    except _FallaDePreparacion as exc:
        resultado = ResultadoSonda(False, recortar_redactado(
            f"la sonda no pudo preparar la llamada: {exc}", _LARGO_DETALLE))
        await _registrar(clave, d, resultado, "config_error", user_id, tenant_id)
        return resultado

    try:
        resultado = await asyncio.wait_for(llamada(timeout), timeout=timeout)
    except (asyncio.TimeoutError, httpx.TimeoutException):
        resultado = ResultadoSonda(False, f"timeout de sonda ({timeout}s)")
    except Exception as exc:  # fail-closed: una sonda que no completó la llamada al proveedor da faceta_caida, nunca sana; el motivo redactado viaja en el veredicto
        resultado = ResultadoSonda(False, recortar_redactado(f"{type(exc).__name__}: {exc}", _LARGO_DETALLE))
    await _registrar(clave, d, resultado, "ok" if resultado.ok else "provider_error", user_id, tenant_id)
    return resultado
