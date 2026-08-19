"""
LAS MANOS — Motor Registry: worker, dispatch por transport (R4).

Ejecuta el job completo:
  1. Marca RUNNING
  2. Verifica kill switch (/etc/jax/PAUSE) antes de llamar
  3. Llama a la API del motor con httpx async, vía la función de
     `motor.transport` (ver `_TRANSPORT_DISPATCH`) — no un motor hardcodeado
  4. Comprueba kill switch cada 5s durante la ejecución
  5. Extrae content y reasoning_content de la respuesta
     — reasoning_content: guardado en JSONL como metadata, NO expuesto en la API
     — content: validado contra el output_schema de la capability
  6. Almacena resultado validado o raw en job_store
  7. Marca COMPLETED / FAILED según corresponda

Kill switch: si /etc/jax/PAUSE existe antes o durante → FAILED con error "killed_by_switch".

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any

import httpx

from motor_registry.catalog import MotorCatalog
from motor_registry.identity_context import build_identity_context
from credential_resolver import resolve_credential_instrumented, CredentialUnavailableError
from motor_registry.job_store import JobStore
from motor_registry.models import JobStatus
from motor_registry.output_validator import validate
from motor_registry.tools_catalog import TOOLS_CATALOG

logger = logging.getLogger(__name__)

_KILL_SWITCH_INTERVAL = 5.0  # segundos entre chequeos durante la ejecución

# Los ocho predicados cerrados de REFORMAS-v3.md §3.1.3. Hardcodeados acá
# (en vez de parsear el YAML en cada job) porque el codebase no tiene hoy
# ningún cargador YAML establecido — solo tomllib para config.toml — y
# sumar uno para un único archivo chico no ameritaba esa dependencia nueva
# en Fase 1. Fuente versionada y lista cerrada: ./policy/vocabulary/predicates.yaml
# (raíz del repo — NO las_manos/policy/...). Si ese archivo cambia, esta
# lista debe actualizarse a mano.
_REFORMAS_V3_PREDICATES = [
    "CAPABILITY_AVAILABLE",
    "FACET_EXISTS",
    "ENGINE_STATUS",
    "CONFIG_VALUE",
    "FILE_EXISTS",
    "AUDIT_EVENT_EXISTS",
    "JOB_STATUS",
    "MEMORY_ENTRY_EXISTS",
]


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

async def _watch_kill_switch(path: str) -> None:
    """Retorna en cuanto detecta el archivo PAUSE. Chequea cada 5s."""
    while True:
        if Path(path).exists():
            return
        await asyncio.sleep(_KILL_SWITCH_INTERVAL)


async def _call_http_openai_compat(
    *,
    api_url: str,
    model: str,
    api_key: str,
    prompt: str,
    timeout: float,
    max_tokens: int = 0,
    tools: list[dict] | None = None,
    reasoning_effort: str | None = None,
) -> dict:
    """Llama a un endpoint OpenAI-compatible. Usado tanto para
    transport='http_openai_compat' (Kimi/Ada/futuros con API key) como para
    transport='ollama' (Qwen local, api_key='') -- Ollama expone el mismo
    formato de request/response en /v1/chat/completions, verificado en vivo
    (2026-08-18): mismo choices[0].message.content/finish_reason/usage.

    max_tokens (2026-08-10): sin esto, un motor de razonamiento puede gastar
    todo el completion budget en reasoning_content y devolver `content`
    cortado a mitad de palabra — bug real reproducido en vivo contra la API
    de Moonshot. 0/falsy = no mandar el campo.

    tools (GAP2 Fase1, 2026-08-19): OPCIONAL -- si no se pasa, el payload
    sale idéntico a antes de este cambio, cero comportamiento nuevo para
    cualquier caller existente. Verificado en vivo contra Ollama+qwen3.6
    (/v1/chat/completions, el mismo endpoint que esta función usa, no el
    nativo /api/chat): acepta 'tools' forma OpenAI y devuelve
    choices[0].message.tool_calls con finish_reason='tool_calls'.

    reasoning_effort (T2, 2026-08-19): OPCIONAL -- solo el caller (run(),
    abajo) decide si lo manda, y solo lo hace para transport=='ollama'
    (unico camino verificado). Probado real contra las 3 APIs con
    reasoning_effort='none': Ollama lo aplica (content inmediato,
    finish_reason='stop'); Moonshot/Kimi devuelve 400 "only type=enabled is
    allowed for this model" (RECHAZA la llamada entera); Zhipu/Ada responde
    200 pero lo ignora (reasoning_content sigue poblado). Por eso esta
    función nunca decide sola mandarlo -- el caller ya filtró por transport
    antes de pasarlo."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if tools:
        payload["tools"] = tools
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{api_url}/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


# transport -> función de dispatch. Un motor nuevo elige un transporte
# EXISTENTE por nombre (dato en DB, Task 1/2) -- agregar un transporte
# nuevo (ej. http_gemini, subprocess) sí requiere código acá, a propósito
# (R4: los transportes son lógica, los motores son dato).
_TRANSPORT_DISPATCH = {
    "http_openai_compat": _call_http_openai_compat,
    "ollama": _call_http_openai_compat,
}


def _humanize_error(exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "Timeout al llamar a la API del motor — la respuesta tardó demasiado"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"Error HTTP {exc.response.status_code} de la API del motor"
    if isinstance(exc, httpx.ConnectError):
        return "No se pudo conectar a la API del motor — verificar red y URL"
    return f"Error inesperado al llamar al motor: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

async def run(
    *,
    job_id: str,
    motor: str,
    capability: str,
    prompt: str,
    context: dict[str, Any],
    store: JobStore,
    catalog: MotorCatalog,
    kill_switch_path: str,
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> None:
    store.update(job_id, status=JobStatus.RUNNING.value, started_at=time.time())

    # Kill switch: chequeo antes de llamar
    if Path(kill_switch_path).exists():
        store.update(
            job_id,
            status=JobStatus.FAILED.value,
            finished_at=time.time(),
            error="killed_by_switch — PAUSE detectado antes de iniciar",
        )
        return

    # Validar motor en catálogo
    motor_entry = catalog.get_motor(motor)
    if motor_entry is None:
        store.update(
            job_id,
            status=JobStatus.FAILED.value,
            finished_at=time.time(),
            error=f"Motor '{motor}' no encontrado en el catálogo",
        )
        return

    # Validar transporte soportado (R4 -- generalizado, ya no solo Kimi)
    call_fn = _TRANSPORT_DISPATCH.get(motor_entry.transport)
    if call_fn is None:
        store.update(
            job_id,
            status=JobStatus.FAILED.value,
            finished_at=time.time(),
            error=f"transport '{motor_entry.transport}' del motor '{motor}' no tiene dispatcher implementado",
        )
        return

    # Validar API key -- guard igual a facet_resolver.py:81-82: ollama/subprocess
    # no usan credencial de proveedor gestionada aca.
    provider_id = motor_entry.provider_id or motor
    api_key = ""
    if motor_entry.transport not in ("ollama", "subprocess"):
        try:
            api_key = await resolve_credential_instrumented(provider_id)
        except CredentialUnavailableError:
            store.update(
                job_id,
                status=JobStatus.FAILED.value,
                finished_at=time.time(),
                error=f"Sin credencial válida configurada para '{provider_id}'",
            )
            return

    # Output schema de la capability
    cap_entry = catalog.get_capability(capability)
    output_schema = cap_entry.output_schema if cap_entry else ""

    # REFORMAS-v3 R3.5 — identidad+capabilities+predicados
    # inyectados antes del prompt real (quién es, qué capability tiene en
    # esta tarea, qué motores existen y qué puede cada uno, predicados
    # emitibles, protocolo de rechazo tipado CAPABILITY_UNBOUND).
    #
    # Catálogo vacío {} por ahora: MotorCatalog indexa por motor individual
    # y por capability (catalog.py:47-92) pero no expone una vista pública
    # invertida {motor: {"allowed_motors_for": [...]}} (la forma que
    # build_identity_context espera — dict[str, dict], no dict[str, list]).
    # Construirla acá requeriría leer sus atributos internos
    # (_motors/_capabilities) o sumar un método nuevo a MotorCatalog —
    # ambos, un refactor mayor no pedido por Fase 1 (ver brief Task 5, nota
    # del ejecutor). {} es type-correct y produce cero "otros motores" — el
    # mismo efecto práctico que antes, sin el riesgo de que un futuro
    # catálogo con más de una entrada dispare un AttributeError dentro de
    # build_identity_context (que hace info.get(...) sobre cada valor).
    identity = build_identity_context(
        motor_name=motor,
        capabilities=[capability],
        catalog={},
        predicates=_REFORMAS_V3_PREDICATES,
        task_id=job_id,
    )
    prompt_with_identity = identity + "\n---\n" + prompt

    # GAP2 Fase1 (2026-08-19): SOLO jax_local, a propósito. executor.py:731-733
    # ya documenta que _HTTP_FACETS (ada/thot/kimi-via-http) no pasa por la
    # gobernanza del Motor Registry (allowed_callers/requires_human_gate/
    # sandbox_only) -- darles tool_calls antes de resolver eso despertaría
    # ese riesgo hoy dormido. Este `if` literal es el gate completo de Fase1
    # (cero mapeo tool->capability, eso es Fase2); reemplazarlo por algo
    # basado en capability es explícitamente trabajo de otra ronda.
    tools_for_call = TOOLS_CATALOG if motor == "jax_local" else None

    # T2 (2026-08-19): jacobs/plan.py::_llm_plan() ya aplica think:false
    # contra el endpoint NATIVO /api/chat de Ollama -- este es el mismo
    # lever para el camino GENERICO (/motor/dispatch, via este worker,
    # OpenAI-compat /v1/chat/completions). Config vive en el motor
    # (MotorEntry.disable_reasoning, DB motor.disable_reasoning), no en la
    # capability/facet: es una propiedad de qué API hay detrás de cada
    # motor, no una política pareja -- verificado que Moonshot/Kimi
    # RECHAZA el mismo parámetro con 400 y Zhipu/Ada lo ignora (ver
    # docstring de _call_http_openai_compat). Override por request: un
    # caller que sí necesita razonamiento activado para ESTE job puede
    # pasar context={"reasoning": true} sin tocar el default del motor.
    want_reasoning = bool(context.get("reasoning"))
    reasoning_effort = None
    if motor_entry.transport == "ollama" and motor_entry.disable_reasoning and not want_reasoning:
        reasoning_effort = "none"

    # Lanzar tarea HTTP y watcher de kill switch en paralelo
    api_task = asyncio.create_task(
        call_fn(
            api_url=motor_entry.api_url,
            model=motor_entry.model,
            api_key=api_key,
            prompt=prompt_with_identity,
            timeout=float(motor_entry.default_timeout_seconds),
            max_tokens=motor_entry.max_tokens,
            tools=tools_for_call,
            reasoning_effort=reasoning_effort,
        )
    )
    kill_task = asyncio.create_task(_watch_kill_switch(kill_switch_path))

    try:
        done, pending = await asyncio.wait(
            [api_task, kill_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
    except asyncio.CancelledError:
        api_task.cancel()
        kill_task.cancel()
        store.update(
            job_id,
            status=JobStatus.CANCELLED.value,
            finished_at=time.time(),
            error="Job cancelado externamente",
        )
        raise

    # Cancelar la tarea que no terminó
    for task in pending:
        task.cancel()

    # Kill switch ganó y la API todavía no terminó
    if kill_task in done and api_task not in done:
        store.update(
            job_id,
            status=JobStatus.FAILED.value,
            finished_at=time.time(),
            error="killed_by_switch — PAUSE detectado durante la ejecución",
        )
        return

    # Obtener resultado de la API (puede lanzar excepción)
    try:
        response_json = api_task.result()
    except Exception as exc:
        # Observabilidad: traceback completo en el log (sin filtrar la API key —
        # format_exc no vuelca variables locales ni headers).
        logger.error(
            "Error de API en job %s: %s\n%s",
            job_id, exc, traceback.format_exc(),
        )
        store.update(
            job_id,
            status=JobStatus.FAILED.value,
            finished_at=time.time(),
            error=_humanize_error(exc),
        )
        return

    # Extraer content y reasoning_content
    choices = response_json.get("choices", [])
    if not choices:
        store.update(
            job_id,
            status=JobStatus.FAILED.value,
            finished_at=time.time(),
            error="API retornó respuesta sin 'choices' — formato inesperado",
        )
        return

    message = choices[0].get("message", {})
    content: str = message.get("content") or ""
    # T3 (2026-08-19): "reasoning_content" es correcto para Moonshot/Kimi y
    # Zhipu/Ada (verificado real, ambos usan esa clave). Ollama (transport
    # dispatchable desde hoy mismo, commit 8daf273) usa "reasoning" -- clave
    # distinta, verificado real contra /v1/chat/completions. Sin este OR,
    # _reasoning_content quedaba null SIEMPRE para jax_local -- no una
    # deuda vieja, el bug nació el mismo día que Ollama empezó a pasar por
    # este código. Se extraen ambas por compatibilidad entre proveedores en
    # vez de una rama `if provider_id == "ollama"` -- ningún proveedor
    # conocido manda las dos a la vez, así que no hay ambigüedad real que
    # resolver eligiendo una sobre la otra.
    reasoning_content: str = message.get("reasoning_content") or message.get("reasoning") or ""
    finish_reason = choices[0].get("finish_reason")
    tool_calls = message.get("tool_calls") or []

    # GAP2 Fase1 (2026-08-19): el modelo pidió tools -- Fase1 es
    # observación pura, cero ejecución, cero segundo turno. NO marcar
    # COMPLETED (el trabajo no terminó, el modelo está esperando un
    # resultado que nunca le va a llegar en esta fase) -- ese sería
    # exactamente el fail-open de output_validator.py que dejamos
    # anotado como deuda: reportar éxito desde una rama que no lo es.
    if tool_calls:
        logger.info(
            "job %s pidió %d tool_call(s) (Fase1, sin ejecutar): %s",
            job_id, len(tool_calls),
            [
                {"name": tc.get("function", {}).get("name"),
                 "arguments": tc.get("function", {}).get("arguments")}
                for tc in tool_calls
            ],
        )
        store.update(
            job_id,
            status=JobStatus.TOOLS_REQUESTED.value,
            finished_at=time.time(),
            result_summary=f"Modelo pidió {len(tool_calls)} tool(s), no ejecutadas (Fase1)",
            _reasoning_content=reasoning_content[:2000] if reasoning_content else None,
            _finish_reason=finish_reason,
            _tool_calls=[
                {"name": tc.get("function", {}).get("name"),
                 "arguments": tc.get("function", {}).get("arguments")}
                for tc in tool_calls
            ],
        )
        return
    usage = response_json.get("usage")

    if reasoning_content:
        logger.debug(
            "reasoning_content de job %s (%d chars) — solo en log, no expuesto",
            job_id, len(reasoning_content),
        )

    # Observabilidad (2026-08-10): antes finish_reason/usage se descartaban
    # por completo — un corte real (finish_reason='length') era
    # indiagnosticable despues del hecho, solo quedaba `content` truncado
    # sin ninguna pista de por que.
    if finish_reason == "length":
        logger.warning(
            "job %s cortado por limite de tokens (finish_reason=length) — "
            "content=%d chars, usage=%s",
            job_id, len(content), usage,
        )

    # Validar output contra el schema de la capability
    validation = validate(content, output_schema)
    if validation.get("warning"):
        logger.warning("Validación output job %s: %s", job_id, validation["warning"])

    result_summary = content[:200] if content else "(sin contenido)"

    store.update(
        job_id,
        status=JobStatus.COMPLETED.value,
        finished_at=time.time(),
        result_summary=result_summary,
        # Campos internos — guardados en JSONL, no expuestos en MotorJobView
        _reasoning_content=reasoning_content[:2000] if reasoning_content else None,
        _finish_reason=finish_reason,
        _usage=usage,
        _validation_validated=validation["validated"],
        _validation_warning=validation.get("warning"),
        _validation_missing_fields=validation.get("missing_fields") or [],
    )

    # Usage tracking (2026-08-10): best-effort, nunca debe romper el job ya
    # marcado COMPLETED arriba. record_motor_usage es fail-soft por su
    # cuenta (sin user_id/tenant_id no escribe nada; error de DB solo loguea).
    # provider_id: reutiliza la MISMA variable ya resuelta arriba (linea ~141)
    # para la credencial -- antes se re-declaraba aca con un default DISTINTO
    # (.get(motor) sin fallback vs .get(motor, motor)), dos semanticas para
    # el mismo nombre dentro de la misma funcion, y para cualquier motor sin
    # entrada en el mapa esta segunda copia daba None en silencio (0 filas
    # de uso, sin log).
    from motor_registry.usage_writer import record_motor_usage
    if provider_id and usage:
        await record_motor_usage(
            user_id, tenant_id, motor, provider_id, motor_entry.model,
            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
        )
    elif provider_id:
        # Observabilidad (I2, 2026-08-10): sin este log, un provider que
        # responde sin bloque 'usage' hace que Costos subreporte sin ningun
        # rastro -- mismo hueco que finish_reason=='length' ya cierra arriba
        # para el caso de corte por limite de tokens.
        logger.warning(
            "job %s (motor=%s): respuesta sin 'usage' -- no se registra "
            "fila de costo (tokens desconocidos)",
            job_id, motor,
        )
