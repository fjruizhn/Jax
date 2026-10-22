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
ok/provider_error/config_error) y su uso en axioma_usage: se paga y se ve.
Con tokens medidos, request_type='preflight_probe'. Si la llamada salió pero
no hay tokens medidos -- la sonda VENCIÓ o el proveedor respondió 2xx SIN
usage -- el proveedor pudo cobrarla igual: se registra con tokens ESTIMADOS
(entrada ⌈len(MENSAJE_DE_SONDA) / JAX_PREVUELO_CHARS_POR_TOKEN⌉, salida el
tope que pidió la sonda) y request_type='preflight_probe_est', que es la
marca de estimación (axioma_usage no tiene otra columna donde ponerla sin
DDL; la columna es VARCHAR(20) y el valor tiene 19). Ola final F4, 2026-09-17,
revierte el Ruling R14.

Se decide por si el pedido SALIÓ hacia el proveedor (pasada final R34):
  - no salió, NO registra: httpx.ConnectError y ConnectTimeout (no hubo
    conexión), PoolTimeout (nunca obtuvo una), UnsupportedProtocol (se
    rechaza antes de conectar), y un 4xx (el proveedor lo rechazó sin
    procesarlo); tampoco una falla local (config_error) ni una base caída
    al resolver;
  - salió y no hay uso medido, registra ESTIMADO: 2xx sin usage, el timeout
    propio de la sonda (wait_for: no se sabe en qué fase cortó, se asume que
    salió), ReadTimeout/WriteTimeout, ReadError, RemoteProtocolError, un 5xx
    (gateways 504/524 incluidos: el modelo pudo haber corrido detrás) y
    cualquier otra excepción de la llamada -- ante la duda, el costo se ve. Si el registro de salud
falla, el veredicto se mantiene (el dato es la respuesta del proveedor, no
la fila), se loguea WARNING y se cuenta.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass

import httpx

from cliente_http_compartido import obtener_cliente_http
from credential_resolver import CredentialUnavailableError, resolve_credential
from facet_resolver import resolve_facet
from redaccion import recortar_redactado

from jacobs import facet_health, prevuelo_config
from jacobs.prevuelo_reglas import Despacho
from jacobs.usage_writer import record_direct_usage
from jacobs.office_runtime import (
    OfficeErrorPhase, OfficeExecutionError, assert_office_transport, ensure_office_http_log_privacy,
    map_office_exception, office_http_log_scope, office_http_status_error,
    parse_office_provider_response, require_office_runtime_policy, stable_office_error,
)
from jacobs.office_contract import OfficeRejectionCode

logger = logging.getLogger("jacobs.sonda")

MENSAJE_DE_SONDA = "Respondé solamente: ok"
REQUEST_TYPE_SONDA = "preflight_probe"
REQUEST_TYPE_SONDA_ESTIMADA = "preflight_probe_est"
_LARGO_DETALLE = 200
# Pasada R37: segundos que wait_for espera POR ENCIMA del timeout de httpx.
# Con el mismo valor, el reloj de wait_for (que arranca antes que el de
# httpx) ganaba siempre: un connect colgado salía como TimeoutError genérico
# -- cobrado -- en vez de httpx.ConnectTimeout -- no cobrado. El margen deja
# que la excepción CLASIFICADA de httpx llegue primero cuando aplica; wait_for
# queda como techo total (httpx acota cada fase, no la suma). Es estructural,
# no de configuración: sólo tiene que cubrir el trabajo alrededor del
# timeout de httpx dentro de la misma llamada.
MARGEN_DE_WAIT_FOR_S = 1


@dataclass(frozen=True)
class ResultadoSonda:
    ok: bool
    detalle: str | None
    tokens_in: int = 0
    tokens_out: int = 0
    # False cuando la respuesta 2xx no trae el campo de uso del proveedor
    # (F4): los ceros de arriba no son "costó 0", son "no se sabe".
    uso_medido: bool = True


_registros_perdidos = 0


def registros_perdidos() -> int:
    """Eventos de sonda que no se pudieron escribir desde que arrancó el proceso."""
    return _registros_perdidos


class _RespuestaNoExitosa(RuntimeError):
    """El proveedor respondió fuera de 2xx. `status` decide el cobro: un 4xx
    no se procesó, un 5xx pudo procesarse (pasada final R34)."""

    def __init__(self, status: int, mensaje: str):
        super().__init__(mensaje)
        self.status = status


# Excepciones de httpx que garantizan que el pedido NO llegó a salir.
_NO_SALIO = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.UnsupportedProtocol)


def _pudo_cobrarse(exc: BaseException) -> bool:
    if isinstance(exc, _NO_SALIO):
        return False
    if isinstance(exc, _RespuestaNoExitosa):
        return exc.status >= 500
    return True


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


def _office_health_detail(error: OfficeExecutionError) -> str:
    """Return the closed, content-free Office health classification."""
    if error.exception_classname == "HTTP_4XX":
        return "provider_http_4xx"
    if error.exception_classname == "HTTP_5XX":
        return "provider_http_5xx"
    if error.code == OfficeRejectionCode.OFFICE_PROVIDER_TIMEOUT:
        return "provider_timeout"
    if error.code == OfficeRejectionCode.OFFICE_PROVIDER_UNAVAILABLE:
        return "transport_error"
    return "config_error"


def _office_probe_result(data: object, transport: str) -> ResultadoSonda:
    """Validate all provider data before health or usage can observe it."""
    try:
        parsed = parse_office_provider_response(data, transport)
    except OfficeExecutionError as exc:
        # Parsing is after a successful HTTP response, so accounting must not
        # mistake a malformed payload for a request that never left Jacobs.
        raise stable_office_error(
            exc, provider_call_made=True, phase=OfficeErrorPhase.PREFLIGHT,
        ) from None
    # Probe quality/truncation is deliberately not a health failure: a valid
    # 2xx response establishes reachability.  Only its validated counters
    # cross this boundary.
    return ResultadoSonda(
        True, None, parsed.tokens_in, parsed.tokens_out,
        uso_medido=parsed.usage_measured,
    )


def _office_probe_failure(
    exc: BaseException, *, provider_call_made: bool, uso_medido: bool = True,
) -> ResultadoSonda:
    error = stable_office_error(
        map_office_exception(exc, provider_call_made=provider_call_made),
        provider_call_made=provider_call_made if not isinstance(exc, OfficeExecutionError) else None,
    )
    return ResultadoSonda(
        False, _office_health_detail(error), error.tokens_in, error.tokens_out,
        uso_medido=uso_medido and error.usage_measured,
    )


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


async def _post(url: str, headers: dict, payload: dict, timeout: int, secretos: list[str], *, runtime_policy=None) -> dict:
    # E-24 (merge 2026-09-17): el cliente HTTP COMPARTIDO del proceso, no uno
    # por sonda -- un AsyncClient por llamada deja un socket en TIME_WAIT cada
    # vez y es lo que el frente E cerró en todos los caminos de salida.
    if runtime_policy is not None:
        require_office_runtime_policy(runtime_policy)
        ensure_office_http_log_privacy()
        error = None
        sent = False
        resp = None
        try:
            post = obtener_cliente_http().post
            with office_http_log_scope():
                sent = True
                resp = await post(url, headers=headers, json=payload, timeout=timeout, follow_redirects=False)
                if not 200 <= resp.status_code < 300:
                    status_error = office_http_status_error(resp.status_code)
                    status_error.exception_classname = "HTTP_4XX" if resp.status_code < 500 else "HTTP_5XX"
                    raise status_error
                data = resp.json()
        except asyncio.CancelledError:
            raise
        except OfficeExecutionError as exc:
            error = stable_office_error(exc, provider_call_made=exc.provider_call_made or sent)
        except Exception as exc:
            error = map_office_exception(exc, provider_call_made=sent)
        if error is not None:
            resp = None
            raise error
        return data
    resp = await obtener_cliente_http().post(url, headers=headers, json=payload, timeout=timeout)
    if not 200 <= resp.status_code < 300:
        raise _RespuestaNoExitosa(
            resp.status_code,
            f"HTTP {resp.status_code}: {recortar_redactado(resp.text, _LARGO_DETALLE, secretos)}",
        )
    return resp.json()


async def _preparar(clave: str, d: Despacho, *, runtime_policy=None):
    """Resuelve todo lo que la llamada real necesita (faceta o credencial del
    motor) y devuelve un cerrado (closure) `async def (timeout) -> ResultadoSonda`
    que HACE la llamada. Nada de esto corre bajo el timeout de la sonda --
    ver Ruling R13 y el docstring del módulo. Levanta _FallaDePreparacion
    para las tres fallas LOCALES conocidas; cualquier otra excepción (p.ej.
    una caída real de la DB al resolver faceta o credencial) se propaga tal
    cual, sin envolver."""
    limite = _tope(d)
    mensajes = [{"role": "user", "content": MENSAJE_DE_SONDA}]

    if runtime_policy is not None:
        runtime_policy = require_office_runtime_policy(runtime_policy, phase=OfficeErrorPhase.PREFLIGHT)
        if d.via_motor:
            raise OfficeExecutionError(
                OfficeRejectionCode.OFFICE_TRANSPORT_NOT_ALLOWED,
                phase=OfficeErrorPhase.PREFLIGHT,
            )
        assert_office_transport(runtime_policy, d.transporte, phase=OfficeErrorPhase.PREFLIGHT)
    if d.via_motor:
        # El MISMO cliente que worker.py usa para despachar el motor.
        from motor_registry import worker
        if d.transporte == "ollama":
            api_key, campo = "", "max_tokens"
        else:
            try:
                api_key = await resolve_credential(d.provider_id)
            except CredentialUnavailableError as exc:
                # Ruling R16: resolve_credential() (las_manos/credential_resolver.py:108-128)
                # SIEMPRE envuelve en CredentialUnavailableError, tanto si la
                # fila no existe (_query_active_credential la levanta directo,
                # línea 104, SIN causa) como si la DB está caída (el `except
                # Exception as e` de la línea 120 atrapa cualquier otra cosa
                # -- un aiomysql.OperationalError real, por ejemplo -- y la
                # re-envuelve con `from e` en la línea 128).
                # B1.4 (master, 2026-09-17): la variante con respaldo por .env
                # ya no existe -- la base es la única fuente de credenciales, y
                # el guard de tests/test_credencial_sin_fallback_env.py prohíbe
                # hasta nombrarla. La distinción de abajo NO cambia: sigue
                # estando en la causa que `resolve_credential` encadena con
                # `from e`.
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
                raise _RespuestaNoExitosa(
                    exc.response.status_code,
                    f"HTTP {exc.response.status_code}: "
                    f"{recortar_redactado(exc.response.text, _LARGO_DETALLE, [api_key])}"
                ) from exc
            uso = data.get("usage") or {}
            return ResultadoSonda(True, None, uso.get("prompt_tokens", 0), uso.get("completion_tokens", 0),
                                  uso_medido=bool(uso))

        return _llamada

    f = await resolve_facet(clave)
    if runtime_policy is not None:
        assert_office_transport(runtime_policy, f.transport, phase=OfficeErrorPhase.PREFLIGHT)

    if f.transport == "http_gemini":
        async def _llamada(timeout: int) -> ResultadoSonda:
            data = await _post(
                f"{f.base_url}/models/{f.model}:generateContent",
                {"x-goog-api-key": f.credential},
                {"contents": [{"role": "user", "parts": [{"text": MENSAJE_DE_SONDA}]}],
                 "generationConfig": {"maxOutputTokens": limite}},
                timeout, [f.credential], runtime_policy=runtime_policy,
            )
            if runtime_policy is not None:
                return _office_probe_result(data, f.transport)
            uso = data.get("usageMetadata") or {}
            return ResultadoSonda(True, None, uso.get("promptTokenCount", 0), uso.get("candidatesTokenCount", 0),
                                  uso_medido=bool(uso))
        return _llamada

    if f.transport == "http_openai_compat":
        async def _llamada(timeout: int) -> ResultadoSonda:
            data = await _post(
                f"{f.base_url}/chat/completions",
                {"Authorization": f"Bearer {f.credential}", "Content-Type": "application/json"},
                {"model": f.model, "messages": mensajes, "stream": False, d.max_tokens_param: limite},
                timeout, [f.credential], runtime_policy=runtime_policy,
            )
            if runtime_policy is not None:
                return _office_probe_result(data, f.transport)
            uso = data.get("usage") or {}
            return ResultadoSonda(True, None, uso.get("prompt_tokens", 0), uso.get("completion_tokens", 0),
                                  uso_medido=bool(uso))
        return _llamada

    if f.transport == "ollama":
        from jacobs.executor import OLLAMA_URL

        async def _llamada(timeout: int) -> ResultadoSonda:
            data = await _post(
                OLLAMA_URL, {},
                {"model": f.model, "messages": mensajes, "stream": False, "options": {"num_predict": limite}},
                timeout, [], runtime_policy=runtime_policy,
            )
            if runtime_policy is not None:
                return _office_probe_result(data, f.transport)
            return ResultadoSonda(True, None, data.get("prompt_eval_count", 0), data.get("eval_count", 0),
                                  uso_medido="prompt_eval_count" in data or "eval_count" in data)
        return _llamada

    raise _FallaDePreparacion(f"transporte '{f.transport}' de '{clave}' no tiene sonda")


def _uso_estimado(d: Despacho) -> tuple[int, int]:
    """F4: tokens de una llamada que salió sin uso medido. Entrada con el
    mismo divisor conservador del costo máximo (§4.6); salida, el tope que
    pidió la sonda (lo máximo que el proveedor pudo facturar)."""
    return math.ceil(len(MENSAJE_DE_SONDA) / prevuelo_config.chars_por_token()), _tope(d)


async def _registrar(clave: str, d: Despacho, r: ResultadoSonda, outcome: str,
                     user_id: str | None, tenant_id: str | None, *, llamada_cobrable: bool,
                     runtime_policy=None) -> None:
    global _registros_perdidos
    try:
        await facet_health.registrar_evento_de_sonda(clave, outcome, r.detalle, time.time())
    except Exception as exc:  # fail-soft: el veredicto sale de la respuesta del proveedor, no de esta fila; sin ella el próximo pre-vuelo vuelve a sondear, y la pérdida queda contada en registros_perdidos() y en el WARNING
        _registros_perdidos += 1
        logger.warning(
            "pre-vuelo: no se pudo registrar la sonda de '%s' en facet_health_event (%s); "
            "van %d registros perdidos",
            clave,
            type(exc).__name__ if runtime_policy is not None else recortar_redactado(
                f"{type(exc).__name__}: {exc}", _LARGO_DETALLE),
            _registros_perdidos,
        )
    if not llamada_cobrable:
        return
    if r.uso_medido and (r.tokens_in or r.tokens_out):
        try:
            await record_direct_usage(
                user_id, tenant_id, clave, d.provider_id, d.modelo, r.tokens_in, r.tokens_out,
                request_type=REQUEST_TYPE_SONDA,
            )
        except Exception:
            if runtime_policy is None:
                raise
            logger.warning("pre-vuelo Office: no se pudo registrar uso (%s)", "usage_error")
    elif not r.uso_medido:
        estimated_in, estimated_out = _uso_estimado(d)
        if runtime_policy is None:
            # Classic has historically recorded the estimate pair exactly.
            tokens_in, tokens_out = estimated_in, estimated_out
        else:
            # A malformed peer counter makes the charge uncertain, but a
            # verified sibling remains a lower bound and must never be
            # replaced by a smaller estimate.
            tokens_in = max(r.tokens_in, estimated_in)
            tokens_out = max(r.tokens_out, estimated_out)
        try:
            await record_direct_usage(
                user_id, tenant_id, clave, d.provider_id, d.modelo, tokens_in, tokens_out,
                request_type=REQUEST_TYPE_SONDA_ESTIMADA,
            )
        except Exception:
            if runtime_policy is None:
                raise
            logger.warning("pre-vuelo Office: no se pudo registrar uso (%s)", "usage_error")


async def sondear(clave: str, d: Despacho, *, user_id: str | None = None,
                  tenant_id: str | None = None, runtime_policy=None) -> ResultadoSonda:
    timeout = prevuelo_config.sonda_timeout_s()
    try:
        if runtime_policy is not None:
            runtime_policy = require_office_runtime_policy(runtime_policy, phase=OfficeErrorPhase.PREFLIGHT)
        llamada = await _preparar(clave, d, runtime_policy=runtime_policy)
    except OfficeExecutionError as exc:
        resultado = _office_probe_failure(exc, provider_call_made=False)
        await _registrar(clave, d, resultado, "config_error", user_id, tenant_id,
                         llamada_cobrable=False, runtime_policy=runtime_policy)
        return resultado
    except _FallaDePreparacion as exc:
        if runtime_policy is not None:
            resultado = _office_probe_failure(exc, provider_call_made=False)
            await _registrar(clave, d, resultado, "config_error", user_id, tenant_id,
                             llamada_cobrable=False, runtime_policy=runtime_policy)
            return resultado
        resultado = ResultadoSonda(False, recortar_redactado(
            f"la sonda no pudo preparar la llamada: {exc}", _LARGO_DETALLE))
        await _registrar(clave, d, resultado, "config_error", user_id, tenant_id, llamada_cobrable=False)
        return resultado
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if runtime_policy is not None:
            resultado = _office_probe_failure(exc, provider_call_made=False)
            await _registrar(clave, d, resultado, "config_error", user_id, tenant_id,
                             llamada_cobrable=False, runtime_policy=runtime_policy)
            return resultado
        raise

    # F4 + pasada final R34: se registra uso estimado si el pedido pudo SALIR
    # (_pudo_cobrarse); ver el docstring del módulo.
    cobrable = True
    try:
        resultado = await asyncio.wait_for(llamada(timeout), timeout=timeout + MARGEN_DE_WAIT_FOR_S)
    except OfficeExecutionError as exc:
        resultado = _office_probe_failure(
            exc, provider_call_made=exc.provider_call_made,
            uso_medido=exc.usage_measured,
        )
        cobrable = exc.provider_call_made and exc.exception_classname != "HTTP_4XX"
    except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
        if runtime_policy is not None:
            resultado = _office_probe_failure(exc, provider_call_made=True, uso_medido=False)
            cobrable = True
        else:
            resultado = ResultadoSonda(False, f"timeout de sonda ({timeout}s)", uso_medido=False)
            cobrable = _pudo_cobrarse(exc)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # fail-closed: una sonda que no completó la llamada al proveedor da faceta_caida, nunca sana; el motivo redactado viaja en el veredicto
        if runtime_policy is not None:
            resultado = _office_probe_failure(exc, provider_call_made=True, uso_medido=False)
            cobrable = True
        else:
            resultado = ResultadoSonda(False, recortar_redactado(f"{type(exc).__name__}: {exc}", _LARGO_DETALLE),
                                       uso_medido=False)
            cobrable = _pudo_cobrarse(exc)
    await _registrar(clave, d, resultado, "ok" if resultado.ok else "provider_error", user_id, tenant_id,
                     llamada_cobrable=cobrable, runtime_policy=runtime_policy)
    return resultado
