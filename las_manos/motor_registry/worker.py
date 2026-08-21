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
import json
import logging
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import httpx

from motor_registry.catalog import MotorCatalog
from motor_registry.identity_context import build_identity_context
from credential_resolver import resolve_credential_instrumented, CredentialUnavailableError
from motor_registry.job_store import JobStore
from motor_registry.tool_authority import authorize_and_execute_tool_call, get_workspace_head
from motor_registry.models import JobStatus
from motor_registry.output_validator import validate
from motor_registry.tools_catalog import TOOLS_CATALOG

logger = logging.getLogger(__name__)

_KILL_SWITCH_INTERVAL = 5.0  # segundos entre chequeos durante la ejecución

# GAP2 Fase3 (2026-08-19): cotas del bucle de tool-calling, configurables
# por env var (mismo patrón que CREDENTIAL_CACHE_TTL_SECONDS en
# credential_resolver.py) -- nunca hardcodeadas sin escape.
# MAX_TOOL_LOOP_ITERATIONS: tope duro de turnos modelo<->tools por job.
MAX_TOOL_LOOP_ITERATIONS = int(os.getenv("MOTOR_TOOL_LOOP_MAX_ITERATIONS", "5"))
# LOOP_DETECTION_THRESHOLD: la MISMA tool con los MISMOS argumentos (string
# crudo) repetida esta cantidad de veces SEGUIDAS corta el bucle antes de
# agotar las iteraciones completas -- señal más específica que "se acabaron
# los turnos". 3 = deja lugar a un reintento legítimo (ej. el modelo repite
# una vez por las dudas) sin tolerar un ciclo real.
LOOP_DETECTION_THRESHOLD = int(os.getenv("MOTOR_TOOL_LOOP_DETECTION_THRESHOLD", "3"))
# MAX_TOTAL_READ_BYTES: presupuesto ACUMULADO de bytes leídos por read_file
# en todo el bucle (no por archivo -- eso ya lo cubre tool_authority.
# MAX_READ_BYTES por-llamada). Defensa en profundidad contra un agente que
# no puede exfiltrar un archivo grande de una vez pero sí varios chicos a
# lo largo de varios turnos.
MAX_TOTAL_READ_BYTES = int(os.getenv("MOTOR_TOOL_LOOP_MAX_TOTAL_READ_BYTES", "500000"))
# GAP2 Fase4 (2026-08-19): mismo criterio que MAX_TOTAL_READ_BYTES pero para
# write_file -- acumulado ACROSS turnos, distinto del cap por-llamada
# (tool_authority.MAX_WRITE_BYTES). Un agente que no puede escribir un solo
# archivo gigante sí podría escribir muchos chicos; esto acota el volumen
# total de disco tocado por job, no solo por llamada.
MAX_TOTAL_WRITE_BYTES = int(os.getenv("MOTOR_TOOL_LOOP_MAX_TOTAL_WRITE_BYTES", "500000"))

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
    messages: list[dict],
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
    antes de pasarlo.

    messages (GAP2 Fase3, 2026-08-19): antes era `prompt: str` y esta
    función armaba el único mensaje user acá adentro -- ahora el caller
    (run(), que sostiene el historial del bucle de tool-calling) manda la
    lista completa. Formato de 'tool' verificado real contra Ollama (no
    inferido de OpenAI): {"role":"assistant","content":...,"tool_calls":[...]}
    seguido de {"role":"tool","tool_call_id":<mismo id>,"content":<string>}
    -- un mensaje 'tool' por cada tool_call de la respuesta anterior."""
    payload = {
        "model": model,
        "messages": messages,
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


def _partial_iteration_entry(iteration: int, tool_calls: list[dict], iteration_results: list[dict]) -> dict:
    """Entrada de _tool_loop_history para UNA iteración -- funciona tanto al
    completarla entera (len(iteration_results) == len(tool_calls)) como a
    mitad de camino (un corte de cota disparado antes de procesar todos los
    tool_calls de esa respuesta): tool_calls[:len(iteration_results)] recorta
    a lo que efectivamente ya tiene resultado, nunca más."""
    return {
        "iteration": iteration,
        "tool_calls": [
            {"name": tc.get("function", {}).get("name"), "arguments": tc.get("function", {}).get("arguments")}
            for tc in tool_calls[:len(iteration_results)]
        ],
        "results": iteration_results,
    }


# GAP2 Fase4 (T4): timeout propio de la auditoría -- otra llamada a LLM,
# no puede colgar el job (que ya terminó y quedó COMPLETED). Constante
# separada de motor.default_timeout_seconds del motor PRODUCTOR: el
# auditor puede ser un motor distinto con su propio timeout configurado,
# pero se acota igual con un techo duro acá por si ese configurado es
# generoso -- la auditoría es best-effort, no debe ser la parte lenta.
AUDIT_TIMEOUT_SECONDS = int(os.getenv("MOTOR_TOOL_LOOP_AUDIT_TIMEOUT_SECONDS", "60"))


def _resolve_auditor_motor(*, producing_motor: str, context: dict, catalog: MotorCatalog) -> tuple[str | None, str]:
    """Resuelve qué motor audita. Devuelve (motor_key|None, motivo).
    None = auditoría desactivada (explícito, ver motivo).

    Fuente: capability 'file_write' (auditor_motor) -- es propiedad de la
    operación auditada, no del motor productor ni de la capability del job
    (que puede ser 'generate' y no tener nada que ver). Override real por
    request: context={"auditor": "<motor>"} o context={"auditor": false}
    para desactivar explícito, mismo patrón que context={"reasoning": true}."""
    if "auditor" in context:
        override = context["auditor"]
        if not override:
            return None, "desactivado explícito por request (context.auditor)"
        auditor = str(override)
        source = "override por request"
    else:
        write_cap = catalog.get_capability("file_write")
        auditor = write_cap.auditor_motor if write_cap else None
        if not auditor:
            return None, "sin auditor_motor configurado en capability 'file_write' (default None)"
        source = "default de capability 'file_write'"

    if auditor == producing_motor:
        # Decisión explícita: auto-revisión no vale. No es un error del
        # job (ya corrió y quedó COMPLETED) -- se desactiva la auditoría
        # para ESTE job con motivo visible, no se bloquea nada retroactivo.
        return None, f"auditor resuelto ('{auditor}', {source}) es el mismo motor que produjo el trabajo -- auto-revisión rechazada"

    return auditor, source


async def _audit_and_notify(
    *, job_id: str, motor: str, prompt: str, catalog: MotorCatalog,
    context: dict, files_written: list[dict], job_start_sha: str | None,
) -> None:
    """Cierre de un job que escribió: audita (si hay auditor configurado) y
    notifica por Telegram. Nunca lanza, nunca toca el status del job (ya
    quedó COMPLETED antes de llamar acá) -- ver docstring de run()."""
    from jacobs.reaper import send_telegram_alert
    from motor_registry.tool_authority import WORKSPACE_ROOT

    total_bytes = sum(f["bytes"] for f in files_written)
    file_list = ", ".join(f"{f['path']} ({f['bytes']}b)" for f in files_written)

    auditor_motor, auditor_source = _resolve_auditor_motor(producing_motor=motor, context=context, catalog=catalog)
    verdict, verdict_reason = None, None

    if auditor_motor is not None:
        auditor_entry = catalog.get_motor(auditor_motor)
        if auditor_entry is None or not auditor_entry.enabled:
            # T4 (P10 ambiguo, decisión consciente): el auditor configurado
            # no existe/está deshabilitado en el catálogo -- fail-soft, NO
            # se revierte trabajo bueno por una configuración rota. Se
            # marca visible (log ERROR + verdict "unavailable"), no
            # silencioso.
            logger.error("job %s: auditor '%s' no disponible en el catálogo -- auditoría omitida", job_id, auditor_motor)
            verdict, verdict_reason = "unavailable", f"motor '{auditor_motor}' no encontrado o deshabilitado"
        else:
            try:
                diffs = []
                for f in files_written:
                    if not f.get("sha"):
                        continue
                    r = subprocess.run(
                        ["git", "-C", str(WORKSPACE_ROOT), "show", f["sha"]],
                        capture_output=True, text=True, timeout=10,
                    )
                    if r.returncode == 0:
                        diffs.append(r.stdout[:8000])  # cap defensivo -- no inflar el prompt del auditor sin límite
                diff_text = "\n\n---\n\n".join(diffs) or "(sin diff disponible)"

                audit_prompt = (
                    "Sos un auditor de código. Otro agente ejecutó este objetivo:\n"
                    f"{prompt}\n\nY produjo estos cambios (git show de cada commit):\n{diff_text}\n\n"
                    "Respondé SOLO con un JSON: "
                    '{"verdict": "pass"|"pass_with_observations"|"revert", "reason": "<una frase>"}. '
                    "\"revert\" solo si hay algo genuinamente peligroso o que no cumple el objetivo -- "
                    "no seas exigente con el estilo."
                )
                auditor_api_key = ""
                if auditor_entry.transport not in ("ollama", "subprocess"):
                    auditor_api_key = await resolve_credential_instrumented(auditor_entry.provider_id or auditor_motor)
                auditor_reasoning_effort = "none" if (auditor_entry.transport == "ollama" and auditor_entry.disable_reasoning) else None
                resp = await asyncio.wait_for(
                    _call_http_openai_compat(
                        api_url=auditor_entry.api_url, model=auditor_entry.model, api_key=auditor_api_key,
                        messages=[{"role": "user", "content": audit_prompt}],
                        timeout=float(AUDIT_TIMEOUT_SECONDS), max_tokens=auditor_entry.max_tokens,
                        reasoning_effort=auditor_reasoning_effort,
                    ),
                    timeout=AUDIT_TIMEOUT_SECONDS,
                )
                raw = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                start, end = raw.find("{"), raw.rfind("}")
                parsed = json.loads(raw[start:end + 1]) if start != -1 and end != -1 else {}
                verdict = parsed.get("verdict")
                verdict_reason = parsed.get("reason")
                if verdict not in ("pass", "pass_with_observations", "revert"):
                    # Ambigüedad del auditor -- P10 aplicado acá: NO se
                    # interpreta como "revert" (revertir trabajo bueno tiene
                    # costo real); se marca degradado, visible, sin castigar
                    # al productor por una salida del auditor mal formada.
                    logger.warning("job %s: auditor devolvió verdict no reconocible: %r", job_id, raw[:300])
                    verdict, verdict_reason = "pass_with_observations", f"auditor no dio un verdict parseable (raw: {raw[:200]!r})"
            except Exception as exc:  # fail-soft: mismo criterio que arriba -- timeout/red del auditor no revierte trabajo bueno
                logger.error("job %s: fallo llamando al auditor '%s': %s", job_id, auditor_motor, exc, exc_info=True)
                verdict, verdict_reason = "unavailable", f"{type(exc).__name__}: {exc}"

    reverted = False
    if verdict == "revert" and job_start_sha:
        r = subprocess.run(
            ["git", "-C", str(WORKSPACE_ROOT), "reset", "--hard", job_start_sha],
            capture_output=True, text=True, timeout=15,
        )
        reverted = r.returncode == 0
        logger.error("job %s: auditor pidió REVERTIR (%s) -- git reset --hard %s: %s", job_id, verdict_reason, job_start_sha, "OK" if reverted else r.stderr)
        try:
            from jacobs.store import event_append
            await event_append(job_id, "TOOL_WRITE_REVERTED", {"reason": verdict_reason, "sha_before": job_start_sha, "ok": reverted})
        except Exception:  # fail-soft: mismo criterio del resto del módulo
            logger.error("job %s: no se pudo registrar TOOL_WRITE_REVERTED", job_id, exc_info=True)

    # Severidad (T5): 🔴 el auditor pidió revertir (haya o no funcionado el
    # revert -- ambos casos son la señal fuerte). 🟡 se auditó pero con
    # observaciones, o no se pudo auditar (config rota / auditor caído).
    # 🟢 pass limpio, o sin auditor configurado (ninguna señal negativa que
    # dar -- informativo puro).
    if verdict == "revert":
        severity = "🔴"
    elif verdict in ("pass_with_observations", "unavailable"):
        severity = "🟡"
    else:
        severity = "🟢"
    last_sha = files_written[-1].get("sha") if files_written else None
    message = (
        f"{severity} JAX tool-calling: job {job_id} escribió {len(files_written)} archivo(s), {total_bytes} bytes.\n"
        f"Archivos: {file_list}\n"
        f"Auditor: {auditor_motor or 'ninguno'} ({auditor_source}) -- veredicto: {verdict or 'sin auditar'}"
        + (f" ({verdict_reason})" if verdict_reason else "") + "\n"
        f"sha para revertir: {last_sha or '(sin commit)'}"
        + (" -- YA REVERTIDO" if reverted else "")
    )
    result = await send_telegram_alert(message)
    if not result["ok"]:
        logger.error("job %s: notificación de escritura no se pudo entregar: %s", job_id, result["error"])


async def _notify_failed_with_writes(*, job_id: str, files_written: list[dict], reason: str) -> None:
    """T5: 'cota agotada con escrituras hechas' es alerta fuerte -- el job
    falló mid-loop pero YA escribió (y commiteó) algo antes del corte. Sin
    esto, esas escrituras quedan en el repo de workspace sin que nadie se
    entere hasta encontrarlas por accidente. No hay auditor acá (el job no
    llegó a un final coherente que auditar) -- solo la señal de que hay
    trabajo a medias commiteado, con el sha para revisarlo/revertirlo a mano."""
    if not files_written:
        return
    from jacobs.reaper import send_telegram_alert
    total_bytes = sum(f["bytes"] for f in files_written)
    file_list = ", ".join(f"{f['path']} ({f['bytes']}b)" for f in files_written)
    last_sha = files_written[-1].get("sha")
    message = (
        f"🔴 JAX tool-calling: job {job_id} FALLÓ ({reason}) pero ya había "
        f"escrito {len(files_written)} archivo(s), {total_bytes} bytes.\n"
        f"Archivos: {file_list}\nsha del último commit: {last_sha or '(sin commit)'} "
        "-- revisar a mano, sin auditoría (el job no llegó a un cierre coherente)."
    )
    result = await send_telegram_alert(message)
    if not result["ok"]:
        logger.error("job %s: notificación de fallo-con-escrituras no se pudo entregar: %s", job_id, result["error"])


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
    caller: str | None = None,
    timeout_seconds: int | None = None,
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

    # GAP2 Fase1 (2026-08-19): originalmente `motor == "jax_local"` literal,
    # a propósito -- executor.py:731-733 documenta que _HTTP_FACETS
    # (ada/thot/kimi-via-http) no pasa por la gobernanza del Motor Registry
    # (allowed_callers/requires_human_gate/sandbox_only), y darles tool_calls
    # antes de resolver eso despertaría ese riesgo hoy dormido.
    #
    # T1 (2026-08-21, diagnóstico pipeline 19ad2c42-cdf): el string
    # hardcodeado era una segunda fuente de verdad que nada podía consultar
    # -- el frontend pedía /motors/capabilities y no tenía de dónde leer
    # esta señal, así que armaba planes asignando "implementation" a kimi
    # sin saber que kimi no puede ejecutar tools. Reemplazado por
    # motor_entry.has_tool_access (columna DB, migrations.py, poblada hoy
    # solo para jax_local=TRUE -- mismo resultado que el `if` viejo, pero
    # consultable desde fuera en vez de vivir solo acá). Sigue siendo cero
    # mapeo tool->capability (eso es Fase2, sin cambios en esta ronda).
    tools_for_call = TOOLS_CATALOG if motor_entry.has_tool_access else None

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

    # GAP2 Fase3 (2026-08-19): bucle de tool-calling. Formato verificado
    # REAL contra Ollama (no inferido de OpenAI): un mensaje assistant con
    # tool_calls, seguido de UN mensaje {"role":"tool","tool_call_id":<id>,
    # "content":<string>} por CADA tool_call de esa respuesta -- confirmado
    # con un ciclo de 2 turnos real, el modelo usó el contenido del turno
    # anterior sin alucinar.
    #
    # Requisito no negociable: el gate de autoridad vive EN EL BUCLE, cada
    # iteración pasa por tool_authority.py de cero -- nunca se cachea una
    # autorización de un turno anterior (executor.py:731-733: _HTTP_FACETS
    # no pasa por Motor Registry, no asumir que "ya pasó por acá" cubre el
    # turno siguiente).
    messages: list[dict] = [{"role": "user", "content": prompt_with_identity}]
    # timeout_seconds=0 es un presupuesto real (agotado de inmediato), no
    # "sin presupuesto" -- `is not None`, nunca la verdad de Python (0 es
    # falsy y hubiera desactivado el chequeo por completo).
    loop_deadline = (time.time() + timeout_seconds) if timeout_seconds is not None else None
    last_call_signature: tuple[str, str] | None = None
    consecutive_same_calls = 0
    total_read_bytes = 0
    total_write_bytes = 0
    files_written: list[dict] = []  # {"path": rel, "bytes": n, "sha": sha} -- para T4/T5 al cierre
    job_start_sha: str | None = None  # HEAD del workspace ANTES de la 1ra escritura -- ancla del rollback del job entero
    cumulative_prompt_tokens = 0
    cumulative_completion_tokens = 0
    tool_loop_history: list[dict] = []
    iteration = 0
    validation_retried = False

    while True:
        iteration += 1
        if iteration > MAX_TOOL_LOOP_ITERATIONS:
            # T2/T3: nunca completed con salida parcial. El step de Jacobs
            # (executor.py::_invoke_motor) trata status=='failed' como
            # terminal-error de inmediato -- no espera a su propio timeout.
            store.update(
                job_id, status=JobStatus.FAILED.value, finished_at=time.time(),
                error=f"Bucle de tool-calling agotó {MAX_TOOL_LOOP_ITERATIONS} iteraciones sin resolver",
                _tool_loop_iterations=iteration - 1, _tool_loop_history=tool_loop_history,
                _files_written=files_written,
            )
            await _notify_failed_with_writes(job_id=job_id, files_written=files_written, reason=f"{MAX_TOOL_LOOP_ITERATIONS} iteraciones agotadas")
            return
        if loop_deadline is not None and time.time() >= loop_deadline:
            store.update(
                job_id, status=JobStatus.FAILED.value, finished_at=time.time(),
                error=f"Bucle de tool-calling agotó el presupuesto de tiempo ({timeout_seconds}s) en la iteración {iteration}",
                _tool_loop_iterations=iteration - 1, _tool_loop_history=tool_loop_history,
                _files_written=files_written,
            )
            await _notify_failed_with_writes(job_id=job_id, files_written=files_written, reason="presupuesto de tiempo agotado")
            return

        # --- un turno: dispatch HTTP + watcher de kill switch en paralelo ---
        api_task = asyncio.create_task(
            call_fn(
                api_url=motor_entry.api_url,
                model=motor_entry.model,
                api_key=api_key,
                messages=messages,
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

        for task in pending:
            task.cancel()

        if kill_task in done and api_task not in done:
            store.update(
                job_id,
                status=JobStatus.FAILED.value,
                finished_at=time.time(),
                error="killed_by_switch — PAUSE detectado durante la ejecución",
                _files_written=files_written,
            )
            await _notify_failed_with_writes(job_id=job_id, files_written=files_written, reason="killed_by_switch")
            return

        try:
            response_json = api_task.result()
        except Exception as exc:
            # Observabilidad: traceback completo en el log (sin filtrar la API key —
            # format_exc no vuelca variables locales ni headers).
            logger.error(
                "Error de API en job %s (iteración %d): %s\n%s",
                job_id, iteration, exc, traceback.format_exc(),
            )
            store.update(
                job_id,
                status=JobStatus.FAILED.value,
                finished_at=time.time(),
                error=_humanize_error(exc),
                _files_written=files_written,
            )
            await _notify_failed_with_writes(job_id=job_id, files_written=files_written, reason="error de API del motor")
            return

        choices = response_json.get("choices", [])
        if not choices:
            store.update(
                job_id,
                status=JobStatus.FAILED.value,
                finished_at=time.time(),
                error="API retornó respuesta sin 'choices' — formato inesperado",
                _files_written=files_written,
            )
            await _notify_failed_with_writes(job_id=job_id, files_written=files_written, reason="respuesta de API sin choices")
            return

        message = choices[0].get("message", {})
        content: str = message.get("content") or ""
        # T3 Fase2: "reasoning_content" es correcto para Moonshot/Kimi y
        # Zhipu/Ada; Ollama usa "reasoning" -- se extraen ambas.
        reasoning_content: str = message.get("reasoning_content") or message.get("reasoning") or ""
        finish_reason = choices[0].get("finish_reason")
        tool_calls = message.get("tool_calls") or []
        usage = response_json.get("usage") or {}
        cumulative_prompt_tokens += usage.get("prompt_tokens", 0)
        cumulative_completion_tokens += usage.get("completion_tokens", 0)

        if reasoning_content:
            logger.debug(
                "reasoning_content de job %s iteración %d (%d chars) — solo en log, no expuesto",
                job_id, iteration, len(reasoning_content),
            )
        if finish_reason == "length":
            logger.warning(
                "job %s iteración %d cortado por limite de tokens (finish_reason=length) — "
                "content=%d chars, usage=%s",
                job_id, iteration, len(content), usage,
            )

        if not tool_calls:
            # T2 (2026-08-19): antes, un content que no parseaba como JSON
            # (o no cumplía el schema) igual marcaba el job COMPLETED con
            # un warning -- fail-open en el lugar más visible del sistema.
            # Ahora: si hay schema_name, se valida ACÁ (antes de salir del
            # bucle) -- válido o sin schema, es la respuesta final (break).
            # Inválido y todavía no reintentamos: UN reintento (barato con
            # think:false, mismo criterio que el retry estricto único de
            # Hipatia/grounding) con el error explícito en el mensaje, para
            # que el modelo se corrija -- no un segundo intento ciego.
            # Inválido tras el reintento: FAILED explícito, nunca completed
            # con una salida que no se puede usar.
            validation = validate(content, output_schema)
            if not output_schema or validation["validated"] or validation["skipped"]:
                break
            if validation_retried:
                store.update(
                    job_id, status=JobStatus.FAILED.value, finished_at=time.time(),
                    error=f"Salida no cumple el schema '{output_schema}' tras reintento: {validation['warning']}",
                    _tool_loop_iterations=iteration, _tool_loop_history=tool_loop_history,
                    _files_written=files_written, _validation_warning=validation.get("warning"),
                )
                await _notify_failed_with_writes(job_id=job_id, files_written=files_written, reason="salida no cumple el schema tras reintento")
                return
            validation_retried = True
            logger.warning("job %s: salida no cumple schema '%s' (%s) -- reintentando una vez", job_id, output_schema, validation["warning"])
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    f"Tu respuesta anterior no es válida: {validation['warning']}. "
                    f"Respondé de nuevo, SOLO con el JSON pedido por el schema '{output_schema}', sin texto adicional."
                ),
            })
            continue

        logger.info(
            "job %s iteración %d pidió %d tool_call(s): %s",
            job_id, iteration, len(tool_calls),
            [
                {"name": tc.get("function", {}).get("name"),
                 "arguments": tc.get("function", {}).get("arguments")}
                for tc in tool_calls
            ],
        )

        # Historial: el mensaje assistant con tool_calls va PRIMERO, antes
        # de los resultados -- Ollama lo exige así (verificado real).
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", ""),
                    },
                }
                for tc in tool_calls
            ],
        })

        # T1: múltiples tool_calls en una sola respuesta -- se ejecutan
        # TODAS, en el orden que las devolvió el modelo, secuencial (no
        # asyncio.gather: cada una escribe su propio evento de auditoría vía
        # tool_authority.py, secuencial evita logs intercalados y no hay
        # necesidad real de paralelismo para un read_file). Un rechazo de
        # UNA no aborta las demás de la misma iteración: son autorizaciones
        # independientes (allowed_callers/forbidden_paths de cada tool_name
        # se resuelven por separado) -- rechazar la tool B porque la tool A
        # de la misma respuesta fue rechazada sería castigar una petición
        # que en sí misma puede ser legítima. El modelo ve el resultado real
        # de cada una (ejecutado, rechazado, o error) y decide qué hacer con
        # esa información en la iteración siguiente.
        iteration_results = []
        for tc in tool_calls:
            name = tc.get("function", {}).get("name", "")
            args_json = tc.get("function", {}).get("arguments", "")
            tc_id = tc.get("id", "")

            # Detección de loop: la MISMA tool con los MISMOS argumentos
            # (string crudo, sin normalizar -- si el modelo cambia aunque
            # sea un espacio ya no cuenta como "lo mismo", intencional: no
            # queremos falsos positivos por reformateo trivial) repetida
            # LOOP_DETECTION_THRESHOLD veces seguidas corta el bucle antes
            # de agotar las 5 iteraciones completas -- señal más específica
            # y diagnosticable que "se acabaron las iteraciones".
            signature = (name, args_json)
            if signature == last_call_signature:
                consecutive_same_calls += 1
            else:
                consecutive_same_calls = 1
                last_call_signature = signature

            if consecutive_same_calls >= LOOP_DETECTION_THRESHOLD:
                # Auditoría completa: incluye lo YA ejecutado de ESTA
                # iteración antes del corte (iteration_results), aunque el
                # for-loop no haya terminado -- sin esto, la última
                # iteración (la que disparó el corte) quedaba invisible en
                # _tool_loop_history, encontrado con evidencia (test real)
                # en esta misma sesión.
                store.update(
                    job_id, status=JobStatus.FAILED.value, finished_at=time.time(),
                    error=(
                        f"Bucle detectado: '{name}' con los mismos argumentos "
                        f"{LOOP_DETECTION_THRESHOLD} veces seguidas (iteración {iteration})"
                    ),
                    _tool_loop_iterations=iteration,
                    _tool_loop_history=tool_loop_history + [_partial_iteration_entry(iteration, tool_calls, iteration_results)],
                    _files_written=files_written,
                )
                await _notify_failed_with_writes(job_id=job_id, files_written=files_written, reason="bucle detectado")
                return

            # job_start_sha: HEAD del repo de workspace ANTES de la primera
            # escritura de ESTE job -- capturado antes de despachar, nunca
            # después (después ya incluiría el commit que se está por
            # hacer). Ancla el rollback de "todo lo que escribió este job",
            # ver T1/T7.
            if name == "write_file" and job_start_sha is None:
                job_start_sha = get_workspace_head()

            # T2 (requisito no negociable): autoridad se resuelve de CERO en
            # cada tool_call de cada iteración. Sin excepción, sin caché.
            result = await authorize_and_execute_tool_call(
                tool_name=name, arguments_json=args_json,
                caller=caller or "", job_id=job_id, catalog=catalog,
                tool_call_id=tc_id,
            )
            iteration_results.append(result)

            if result["decision"] == "executed":
                if name == "read_file" and result.get("content"):
                    total_read_bytes += len(result["content"].encode("utf-8"))
                elif name == "write_file":
                    # "content" acá es un mensaje de estado ("Escrito: X
                    # bytes"), NO el contenido del archivo -- el tamaño real
                    # viene de bytes_written, puesto ahí a propósito por
                    # tool_authority._write_file para no confundir los dos.
                    total_write_bytes += result.get("bytes_written", 0)
                    written_path = None
                    try:
                        written_path = json.loads(args_json).get("path")
                    except (json.JSONDecodeError, TypeError):  # fail-soft: written_path solo alimenta el texto de la notificacion (ver _notify_failed_with_writes); si falla el parseo queda None y se pierde el nombre en el mensaje, no la escritura real (ya ejecutada) ni sha/bytes/git_committed, que son los campos de los que depende la reversibilidad
                        pass
                    files_written.append({
                        "path": written_path,
                        "bytes": result.get("bytes_written", 0),
                        "sha": result.get("git_sha"),
                        "git_committed": result.get("git_committed"),
                    })

            # T3: rechazo por autoridad y error operativo reciben el MISMO
            # tratamiento de bucle (se informa al modelo y se continúa) --
            # la distinción TOOL_CALL_REJECTED/TOOL_CALL_EXECUTION_ERROR ya
            # quedó marcada en jacobs_events por tool_authority.py, que es
            # donde importa para la señal de seguridad. Acá solo se decide
            # si el bucle sigue, y en ambos casos la respuesta correcta es
            # "decile al modelo qué pasó y dejalo reaccionar" -- ver
            # justificación completa en el mensaje de esta sesión.
            tool_content = result["content"] if result["decision"] == "executed" else f"ERROR: {result['reason']}"
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": tool_content})

            if total_read_bytes > MAX_TOTAL_READ_BYTES:
                store.update(
                    job_id, status=JobStatus.FAILED.value, finished_at=time.time(),
                    error=(
                        f"Presupuesto acumulado de lectura excedido "
                        f"({total_read_bytes} > {MAX_TOTAL_READ_BYTES} bytes) en la iteración {iteration}"
                    ),
                    _tool_loop_iterations=iteration,
                    _tool_loop_history=tool_loop_history + [_partial_iteration_entry(iteration, tool_calls, iteration_results)],
                    _files_written=files_written,
                )
                await _notify_failed_with_writes(job_id=job_id, files_written=files_written, reason="presupuesto acumulado de lectura excedido")
                return

            if total_write_bytes > MAX_TOTAL_WRITE_BYTES:
                store.update(
                    job_id, status=JobStatus.FAILED.value, finished_at=time.time(),
                    error=(
                        f"Presupuesto acumulado de escritura excedido "
                        f"({total_write_bytes} > {MAX_TOTAL_WRITE_BYTES} bytes) en la iteración {iteration}"
                    ),
                    _tool_loop_iterations=iteration,
                    _tool_loop_history=tool_loop_history + [_partial_iteration_entry(iteration, tool_calls, iteration_results)],
                    _files_written=files_written,
                )
                await _notify_failed_with_writes(job_id=job_id, files_written=files_written, reason="presupuesto acumulado de escritura excedido")
                return

        tool_loop_history.append(_partial_iteration_entry(iteration, tool_calls, iteration_results))
        # sigue al siguiente turno del while

    # --- Bucle terminó con respuesta final (content sin tool_calls) ---
    # `validation` ya se calculó adentro del bucle (T2, arriba) para decidir
    # si esto era la respuesta final o si hacía falta reintentar -- se
    # reusa, no se recalcula.
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
        _usage={"prompt_tokens": cumulative_prompt_tokens, "completion_tokens": cumulative_completion_tokens},
        _validation_validated=validation["validated"],
        _validation_warning=validation.get("warning"),
        _validation_missing_fields=validation.get("missing_fields") or [],
        _tool_loop_iterations=iteration,
        _tool_loop_history=tool_loop_history,
        _files_written=files_written,
    )

    # Usage tracking (2026-08-10): best-effort, nunca debe romper el job ya
    # marcado COMPLETED arriba. record_motor_usage es fail-soft por su
    # cuenta (sin user_id/tenant_id no escribe nada; error de DB solo loguea).
    # Suma de TODOS los turnos del bucle (GAP2 Fase3) -- antes (single-shot)
    # era el usage de la única llamada; con múltiples turnos, reportar solo
    # el último hubiera subreportado el costo real de los turnos con tools.
    from motor_registry.usage_writer import record_motor_usage
    if provider_id and (cumulative_prompt_tokens or cumulative_completion_tokens):
        await record_motor_usage(
            user_id, tenant_id, motor, provider_id, motor_entry.model,
            cumulative_prompt_tokens, cumulative_completion_tokens,
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

    # GAP2 Fase4 (T4/T5): auditoría posterior + notificación -- solo si el
    # job escribió algo. El job YA quedó COMPLETED arriba: esto es un
    # revisor, no un gate. Nunca puede tumbar ni revertir el status del job.
    if files_written:
        await _audit_and_notify(
            job_id=job_id, motor=motor, prompt=prompt, catalog=catalog,
            context=context, files_written=files_written, job_start_sha=job_start_sha,
        )
