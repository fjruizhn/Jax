"""
Jacobs — StepExecutor v0.2.

Ejecuta cada step según su faceta. Respeta kill switch antes de cada step.
Propaga contexto (objective + previous_outputs) a cada invocación.
En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import httpx

from jacobs import store
from jacobs.artifacts import read_artifact, save_if_large
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus
from jacobs.policy import check_kill_switch

logger = logging.getLogger("jacobs.executor")

LAS_MANOS_BASE = "http://127.0.0.1:7777"
OLLAMA_URL     = "http://localhost:11434/api/chat"
MOTOR_POLL_INTERVAL = 5  # segundos entre polls de job

# Tope de seguridad para el output COMPLETO de cada dependencia declarada (~15K tokens).
# Si el ensamble de muchas deps roza la ventana, ajustar y re-verificar con el log de C1.
MAX_DEP_CONTEXT_CHARS = 60_000

# Facetas que van directo por HTTP a sus APIs
_HTTP_FACETS = frozenset({"hipatia", "jekyll", "thot", "ada"})

# Facetas que van vía Motor Registry de LAS MANOS
_MOTOR_FACETS = frozenset({"kimi"})


# ----------------------------------------------------------------
#  Helpers de referencia de artifacts
# ----------------------------------------------------------------

def _load_ref(ref: str) -> dict:
    """Carga un output desde su ref (inline o artifact)."""
    if not ref:
        return {}
    if ref.startswith("inline:"):
        try:
            return json.loads(ref[7:])
        except (json.JSONDecodeError, ValueError):
            return {}
    if ref.startswith("artifact://"):
        try:
            return read_artifact(ref)
        except Exception:  # noqa: BLE001
            return {}
    return {}


# Regla anti-simulación compartida — fuente única de verdad. Se inyecta en el
# prompt de CADA step del pipeline (todas las facets, incluido el motor Kimi),
# porque el path de pipeline NO usa los system_prompt de config.toml.
_EVIDENCE_RULE = (
    "REGLA DE EVIDENCIA (innegociable): Nunca simules, inventes ni asumas la "
    "salida de un comando, log, archivo o llamada a API. Si no lo ejecutaste y "
    "viste su salida real, NO lo reportes como hecho. Si no podés verificar un "
    "dato, declarálo como INCÓGNITA — no lo rellenes con suposiciones. Un reporte "
    "con resultados inventados es peor que no entregar nada. Pegá la evidencia "
    "cruda, no una descripción de lo que harías. \"El que supone se equivoca.\""
)


def _build_context_input(step: Step, pipeline: Pipeline) -> dict:
    """Construye el input enriquecido.

    Si el step declara depends_on, carga el output COMPLETO de esas dependencias
    (hasta MAX_DEP_CONTEXT_CHARS por dep). Si no, resumen 500 chars de los anteriores
    (comportamiento original — no rompe pipelines triviales).
    """
    objective = pipeline.context.get("objective", "")
    previous_outputs: list[dict] = []

    deps = getattr(step, "depends_on", []) or []
    if deps:
        indices = [j for j in deps if 0 <= j < step.step_index]
        full = True
    else:
        indices = list(range(step.step_index))
        full = False

    for j in indices:
        ref = pipeline.context.get(f"step_{j}_ref", "")
        if not ref:
            continue
        facet_name = pipeline.plan[j].facet if j < len(pipeline.plan) else "unknown"
        try:
            data = _load_ref(ref)
            result_text = data.get("result") or data.get("text") or json.dumps(data)
            text = str(result_text)
            if full:
                content = text[:MAX_DEP_CONTEXT_CHARS]
                truncated = len(text) > MAX_DEP_CONTEXT_CHARS
            else:
                content = text[:500]
                truncated = len(text) > 500
        except Exception:  # noqa: BLE001
            content = f"[ref: {ref}]"
            truncated = False
        previous_outputs.append({
            "step_index": j,
            "facet": facet_name,
            "summary": content,
            "truncated": truncated,
        })

    total_chars = sum(len(p["summary"]) for p in previous_outputs)
    logger.info(
        "Jacobs step %s deps=%s contexto=%d chars%s",
        step.step_index, deps, total_chars,
        " [ALGUNA DEP TRUNCADA]" if any(p.get("truncated") for p in previous_outputs) else "",
    )

    return {
        "objective": objective,
        "previous_outputs": previous_outputs,
        "prompt": step.input.get("prompt", ""),
    }


def _enrich_prompt(ctx_input: dict) -> str:
    """Construye el prompt final incluyendo contexto previo."""
    parts: list[str] = [_EVIDENCE_RULE]

    if ctx_input.get("objective"):
        parts.append(f"Objetivo del pipeline: {ctx_input['objective']}")

    prev = ctx_input.get("previous_outputs", [])
    if prev:
        parts.append("\nSalidas de las dependencias declaradas (usalas como fuente, no las reinventes):")
        for p in prev:
            nota = " [TRUNCADO — dependencia excede el tope]" if p.get("truncated") else ""
            parts.append(
                f"\n--- Dependencia: step {p['step_index']} ({p['facet']}){nota} ---\n{p['summary']}"
            )

    if ctx_input.get("prompt"):
        parts.append(f"\nTu tarea: {ctx_input['prompt']}")

    return "\n".join(parts)


# ----------------------------------------------------------------
#  Invocadores por faceta
# ----------------------------------------------------------------

async def _invoke_hipatia(prompt: str, timeout: int) -> dict:
    """Gemini 2.5 Flash con grounding required_web."""
    api_key = os.environ["GEMINI_API_KEY"]
    model   = "gemini-2.5-flash"
    url     = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:generateContent?key={api_key}"
    )
    contents = [{"role": "user", "parts": [{"text": prompt}]}]
    payload  = {
        "contents": contents,
        "tools": [{"google_search": {}}],
    }

    async def _call() -> dict:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text[:200]}")
            return resp.json()

    data = await _call()
    candidate = data.get("candidates", [{}])[0]
    parts_raw = candidate.get("content", {}).get("parts", []) or []
    texto = "".join(p.get("text", "") for p in parts_raw)
    meta  = candidate.get("groundingMetadata", {}) or {}
    chunks = meta.get("groundingChunks") or []
    queries = meta.get("webSearchQueries") or []

    # Retry si no hubo grounding
    if not chunks:
        retry_payload = dict(payload)
        retry_payload["contents"] = [
            {"role": "user", "parts": [{"text": (
                prompt + "\n\nDEBES usar búsqueda web (google_search) para responder. "
                "Si no puedes buscar, responde exactamente: NO_VERIFICADO"
            )}]}
        ]
        data2 = await _call()
        candidate2 = data2.get("candidates", [{}])[0]
        parts2  = candidate2.get("content", {}).get("parts", []) or []
        texto2  = "".join(p.get("text", "") for p in parts2)
        meta2   = candidate2.get("groundingMetadata", {}) or {}
        chunks2 = meta2.get("groundingChunks") or []
        if chunks2:
            texto  = texto2
            chunks = chunks2
            queries = meta2.get("webSearchQueries") or []

    sources = []
    seen: set = set()
    for ch in chunks:
        web = ch.get("web") or {}
        uri = web.get("uri")
        if uri and uri not in seen:
            seen.add(uri)
            sources.append({"title": web.get("title", uri), "url": uri})

    return {
        "success": True,
        "facet":   "hipatia",
        "model":   model,
        "result":  texto,
        "sources": sources,
        "queries": queries,
        "grounded": bool(chunks),
    }


async def _invoke_jekyll(prompt: str, timeout: int) -> dict:
    """DeepSeek V4 Flash — análisis humanista."""
    api_key = os.environ["DEEPSEEK_API_KEY"]
    model   = "deepseek-v4-flash"
    url     = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    messages = [
        {"role": "system", "content": (
            "Eres Jekyll, un analista con sensibilidad humanista. "
            "Reflexionas sobre las implicaciones humanas y sociales de los temas. "
            "Eres profundo, poético cuando es apropiado, pero siempre concreto."
        )},
        {"role": "user", "content": prompt},
    ]
    payload = {"model": model, "messages": messages, "stream": False}

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"DeepSeek HTTP {resp.status_code}: {resp.text[:200]}")
        data  = resp.json()
        texto = data["choices"][0]["message"].get("content", "")

    return {
        "success": True,
        "facet":   "jekyll",
        "model":   model,
        "result":  texto,
    }


async def _invoke_thot(prompt: str, timeout: int) -> dict:
    """OpenAI GPT-5.5 — crítica y cuestionamiento."""
    api_key = os.environ["OPENAI_API_KEY"]
    model   = "gpt-5.5"
    url     = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    messages = [
        {"role": "system", "content": (
            "Eres Thot, el crítico de JAX. Tu trabajo es cuestionar, "
            "identificar supuestos peligrosos, riesgos ocultos y fallas de razonamiento. "
            "Sé preciso, incisivo y honesto. No seas condescendiente."
        )},
        {"role": "user", "content": prompt},
    ]
    payload = {"model": model, "messages": messages, "stream": False}

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI HTTP {resp.status_code}: {resp.text[:200]}")
        data  = resp.json()
        texto = data["choices"][0]["message"].get("content", "")

    return {
        "success": True,
        "facet":   "thot",
        "model":   model,
        "result":  texto,
    }


async def _invoke_ada(prompt: str, timeout: int) -> dict:
    """Z.ai GLM-5.2 — diseño de arquitectura. Gracia si la key no está."""
    api_key = os.environ.get("ZAI_API_KEY") or os.environ.get("ZHIPU_API_KEY", "")
    if not api_key:
        return {
            "success": False,
            "facet":   "ada",
            "result":  "Ada no disponible — ZAI_API_KEY no configurada.",
            "disabled": True,
        }
    model   = "glm-5.2"
    url     = "https://api.z.ai/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    messages = [
        {"role": "system", "content": (
            "Eres Ada, arquitecta de sistemas. "
            "Diseñas soluciones técnicas elegantes con rigor matemático."
        )},
        {"role": "user", "content": prompt},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": 131072,
    }
    texto = ""
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"Z.ai HTTP {resp.status_code}: {body[:200]!r}")
            partes = []
            async for linea in resp.aiter_lines():
                if not linea or not linea.startswith("data:"):
                    continue
                payload_str = linea[5:].strip()
                if payload_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                pieza = delta.get("content")
                if pieza:
                    partes.append(pieza)
            texto = "".join(partes)

    return {
        "success": True,
        "facet":   "ada",
        "model":   model,
        "result":  texto,
    }


async def _invoke_jax_local(prompt: str, timeout: int) -> dict:
    """qwen3:14b via Ollama — razonamiento local."""
    payload = {
        "model":    "qwen3:14b",
        "messages": [{"role": "user", "content": prompt}],
        "stream":   False,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(OLLAMA_URL, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text[:200]}")
        data  = resp.json()
        texto = data.get("message", {}).get("content", "")

    return {
        "success": True,
        "facet":   "jax_local",
        "model":   "qwen3:14b",
        "result":  texto,
    }


_CAPABILITY_MAP = {
    "analysis":  "pipeline_analysis",
    "research":  "pipeline_analysis",
    "review":    "refactor",
    "code":      "refactor",
    "implement": "code_swarm",
}


async def _invoke_motor(step: Step, timeout: int) -> dict:
    """Kimi via Motor Registry de LAS MANOS. Polling hasta completar."""
    capability = _CAPABILITY_MAP.get(step.capability, step.capability)
    # Observabilidad: si la capability no estaba en el mapa de traducción, se
    # despacha cruda al Motor Registry. Antes esto fallaba en 0.0s en silencio.
    if step.capability not in _CAPABILITY_MAP:
        logger.warning(
            "Capability '%s' no está en _CAPABILITY_MAP; se despacha cruda al "
            "Motor Registry (debe existir como [capabilities.%s] en config.toml).",
            step.capability, capability,
        )
    payload = {
        "caller":     "jacobs",
        "capability": capability,
        "motor":      step.facet,
        "trace_id":   step.trace_id,
        "prompt":     _EVIDENCE_RULE + "\n\n" + step.input.get("prompt", json.dumps(step.input)),
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{LAS_MANOS_BASE}/motor/dispatch", json=payload)
        resp.raise_for_status()
        dispatch = resp.json()

    job_id = dispatch.get("job_id")
    if dispatch.get("status") == "rejected":
        reason = dispatch.get("rejected_reason", "sin razón")
        logger.error(
            "Motor Registry RECHAZÓ job (caller=jacobs, capability=%s, motor=%s): %s",
            capability, step.facet, reason,
        )
        raise RuntimeError(f"Motor Registry rechazó el job: {reason}")
    if not job_id:
        raise RuntimeError(f"Motor Registry no devolvió job_id: {dispatch}")

    # Polling hasta timeout
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await asyncio.sleep(MOTOR_POLL_INTERVAL)
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{LAS_MANOS_BASE}/motor/job/{job_id}")
            resp.raise_for_status()
            job = resp.json()

        status = job.get("status", "")
        if status == "completed":
            return {
                "success":        True,
                "facet":          step.facet,
                "job_id":         job_id,
                "result":         job.get("result_summary", ""),
                "result_full":    job,
            }
        if status in ("failed", "cancelled", "rejected"):
            raise RuntimeError(
                f"Motor job {job_id} terminó en estado '{status}': "
                f"{job.get('error', '')}"
            )

    raise asyncio.TimeoutError(
        f"Motor job {job_id} no completó en {timeout}s"
    )


# ----------------------------------------------------------------
#  Ensamble mecánico
# ----------------------------------------------------------------

def _assemble_mechanical(step: Step, pipeline: Pipeline) -> dict:
    """Ensamble MECÁNICO del paquete final. Sin LLM. Concatena los outputs de los
    módulos ya generados (steps de diseño), incluye el manifest que generó este step
    (si su prompt produjo uno) y los parches de reconciliación. No puede fallar por tamaño."""
    partes = []
    partes.append("# PAQUETE MODULAR ENSAMBLADO\n")
    partes.append(f"# Pipeline: {pipeline.pipeline_id}\n")
    partes.append(f"# Objetivo: {pipeline.context.get('objective', '')}\n")
    partes.append(f"# Generado por Jacobs (ensamble mecánico) — {len(pipeline.plan)} steps\n\n")

    skip_caps = {"validate_consistency", "critique", "reconcile", "assemble"}
    patches_text = ""
    for j in range(step.step_index):
        prev = pipeline.plan[j]
        ref = pipeline.context.get(f"step_{j}_ref", "")
        if not ref:
            continue
        data = _load_ref(ref)
        result = data.get("result") or data.get("text") or ""
        if prev.capability == "reconcile":
            patches_text = str(result)
            continue
        if prev.capability in skip_caps:
            continue
        partes.append(f"\n{'='*70}\n## MÓDULO (step {j}): {prev.capability}\n{'='*70}\n")
        partes.append(str(result))

    if patches_text:
        partes.append(f"\n{'='*70}\n## PARCHES DE RECONCILIACIÓN (correcciones del validador)\n{'='*70}\n")
        partes.append(patches_text)

    documento = "\n".join(partes)
    logger.info("Jacobs ensamble mecánico: %d chars de %d módulos", len(documento), step.step_index)
    return {
        "success": True,
        "facet": "ada",
        "model": "mechanical_assembler",
        "result": documento,
    }


# ----------------------------------------------------------------
#  Dispatcher principal
# ----------------------------------------------------------------

async def _dispatch_step(step: Step, pipeline: Pipeline) -> dict:
    """Selecciona el worker correcto según la faceta."""
    # Ensamble mecánico: NO pasa por ningún LLM. Concatena los módulos ya generados.
    if step.capability == "assemble":
        return _assemble_mechanical(step, pipeline)

    ctx_input = _build_context_input(step, pipeline)
    prompt    = _enrich_prompt(ctx_input)
    timeout   = step.timeout_seconds

    if step.facet == "hipatia":
        return await _invoke_hipatia(prompt, timeout)
    if step.facet == "jekyll":
        return await _invoke_jekyll(prompt, timeout)
    if step.facet == "thot":
        return await _invoke_thot(prompt, timeout)
    if step.facet == "ada":
        return await _invoke_ada(prompt, timeout)
    if step.facet == "jax_local":
        return await _invoke_jax_local(prompt, timeout)
    if step.facet in _MOTOR_FACETS:
        return await _invoke_motor(step, timeout)
    if step.facet == "hyde":
        # Llegamos aquí solo si Fernando aprobó vía /approve-step.
        # En v0.2 devuelve placeholder — conexión real pendiente para v0.3.
        return {
            "success":  True,
            "facet":    "hyde",
            "result":   (
                "[v0.2] Hyde aprobado por Fernando. "
                "Conexión a ejecución real pendiente para v0.3."
            ),
            "approved": True,
        }

    raise ValueError(f"Faceta desconocida: '{step.facet}'")


# ----------------------------------------------------------------
#  Cálculo de olas topológicas a partir del DAG (depends_on)
# ----------------------------------------------------------------

def _compute_waves(plan: list[Step], done: set[int]) -> list[list[int]]:
    """Particiona los step_index pendientes en olas topológicas.

    Una ola = todos los steps cuyas dependencias ya están satisfechas (en `done`
    o completadas en olas previas). Steps sin deps van en la ola 0.

    Respeta SOLO depends_on, no el orden del plan. Si hay ciclo o dep inexistente
    (plan.py valida 0 <= dep < idx, así que no debería), los steps irresolubles
    quedan fuera y se loguean — nunca se cuelga. "El que supone se equivoca."
    """
    pending = {s.step_index for s in plan if s.step_index not in done}
    deps_by_idx = {s.step_index: set(s.depends_on or []) for s in plan}

    waves: list[list[int]] = []
    satisfied = set(done)

    while pending:
        ready = sorted(
            idx for idx in pending
            if deps_by_idx.get(idx, set()) <= satisfied
        )
        if not ready:
            logger.error(
                "Jacobs: %d steps sin dependencias resolubles (posible ciclo): %s",
                len(pending), sorted(pending),
            )
            break
        waves.append(ready)
        for idx in ready:
            pending.discard(idx)
            satisfied.add(idx)

    return waves


# ----------------------------------------------------------------
#  Ejecución de UN step (cuerpo del antiguo try/except, extraído)
# ----------------------------------------------------------------

async def _run_one_step(step: Step, i: int, pipeline: Pipeline) -> bool:
    """Ejecuta un step individual. Devuelve True si completó, False si falló.

    Es el cuerpo del antiguo bloque try/except del loop secuencial, extraído sin
    cambios de lógica para poder lanzarlo en paralelo vía asyncio.gather.
    Artifacts, persistencia al repo y eventos: idénticos al original.
    """
    step.status     = StepStatus.running
    step.started_at = time.time()
    await store.step_upsert(step)
    await store.event_append(
        pipeline.pipeline_id, "STEP_STARTED",
        {"step_index": i, "facet": step.facet, "capability": step.capability},
        step.step_id,
    )

    try:
        raw_output = await asyncio.wait_for(
            _dispatch_step(step, pipeline),
            timeout=step.timeout_seconds,
        )

        ref, inline = save_if_large(pipeline.pipeline_id, step.step_id, raw_output)
        if ref:
            step.output_ref = ref
            pipeline.context[f"step_{i}_ref"] = ref
        else:
            inline_ref = f"inline:{json.dumps(inline, ensure_ascii=False)}"
            step.output_ref = inline_ref
            pipeline.context[f"step_{i}_ref"] = inline_ref

        step.status      = StepStatus.completed
        step.finished_at = time.time()
        await store.step_upsert(step)
        await store.event_append(
            pipeline.pipeline_id, "STEP_COMPLETED",
            {"step_index": i, "output_ref": step.output_ref},
            step.step_id,
        )
        try:
            await _persist_step_to_repo(
                pipeline_id=pipeline.pipeline_id,
                pipeline_name=pipeline.name,
                step_index=i,
                facet=step.facet,
                capability=step.capability,
                raw_output=raw_output,
            )
        except Exception as _persist_err:  # noqa: BLE001
            logger.warning("No se pudo persistir step %d al repo: %s", i, _persist_err)
        return True

    except asyncio.TimeoutError:
        await _fail_step(pipeline, step, i, f"Timeout ({step.timeout_seconds}s)")
        return False
    except Exception as exc:  # noqa: BLE001
        await _fail_step(pipeline, step, i, str(exc))
        return False


# ----------------------------------------------------------------
#  Pipeline runner — DIRECTOR DE ORQUESTA (ejecución por olas)
# ----------------------------------------------------------------

async def run_pipeline(pipeline: Pipeline) -> None:
    """
    Ejecuta el pipeline por OLAS topológicas. Dentro de cada ola, los steps
    corren EN PARALELO (asyncio.gather). El orden entre olas respeta depends_on.

    Modos:
      dry_run    — no ejecuta nada, completa inmediatamente.
      supervised — ejecuta UNA ola y pausa (status=interrupted); espera /resume.
                   La granularidad de aprobación es la OLA, no el step.
    Hyde: si un step de la ola es hyde sin aprobar, la ola NO se ejecuta y el
    pipeline se interrumpe hasta /approve-step.
    """
    pipeline_id = pipeline.pipeline_id

    if pipeline.mode == "dry_run":
        await store.pipeline_update_status(pipeline_id, PipelineStatus.completed)
        await store.event_append(pipeline_id, "DRY_RUN_COMPLETE", {"steps": len(pipeline.plan)})
        return

    await store.pipeline_update_status(
        pipeline_id, PipelineStatus.running, pipeline.current_step_index, pipeline.context
    )
    await store.event_append(pipeline_id, "PIPELINE_STARTED")

    # Estado derivado del DAG, no de un cursor lineal: un step está "hecho" si
    # tiene su ref en context (sobrevive a /resume y al relanzador).
    done = {
        i for i in range(len(pipeline.plan))
        if pipeline.context.get(f"step_{i}_ref")
    }

    waves = _compute_waves(pipeline.plan, done)
    logger.info(
        "Jacobs director: %d olas, tamaños=%s (ya completos: %s)",
        len(waves), [len(w) for w in waves], sorted(done),
    )

    for wave_num, wave in enumerate(waves):
        # ---- Kill switch: antes de cada ola ----
        if check_kill_switch():
            for i in wave:
                step = pipeline.plan[i]
                step.status = StepStatus.failed
                step.error  = "Kill switch activo"
                await store.step_upsert(step)
            await store.event_append(
                pipeline_id, "KILL_SWITCH_ABORTED", {"wave": wave_num, "steps": wave}
            )
            await store.pipeline_update_status(pipeline_id, PipelineStatus.aborted)
            return

        # ---- Hyde gate: si algún step de la ola es hyde sin aprobar, interrumpir ----
        hyde_pending = [
            i for i in wave
            if pipeline.plan[i].facet == "hyde"
            and not pipeline.context.get(f"hyde_approved_{pipeline.plan[i].step_id}")
        ]
        if hyde_pending:
            for i in hyde_pending:
                step = pipeline.plan[i]
                step.status = StepStatus.blocked_human_gate
                await store.step_upsert(step)
                await store.event_append(
                    pipeline_id, "STEP_BLOCKED_HUMAN_GATE",
                    {"step_index": i, "facet": "hyde"}, step.step_id,
                )
            await store.pipeline_update_status(
                pipeline_id, PipelineStatus.interrupted, wave[0], pipeline.context
            )
            await store.event_append(
                pipeline_id, "PIPELINE_INTERRUPTED",
                {"at_wave": wave_num, "hyde_steps": hyde_pending,
                 "reason": "hyde — requiere /approve-step"},
            )
            return

        # ---- EJECUTAR LA OLA EN PARALELO ----
        await store.event_append(
            pipeline_id, "WAVE_STARTED",
            {"wave": wave_num, "steps": wave, "parallel": len(wave)},
        )
        results = await asyncio.gather(*[
            _run_one_step(pipeline.plan[i], i, pipeline)
            for i in wave
        ])

        # Persistir avance del context tras la ola completa.
        # current_step_index = primer índice NO completado (informativo).
        next_idx = max(wave) + 1
        await store.pipeline_update_status(
            pipeline_id, PipelineStatus.running, next_idx, pipeline.context
        )

        # ---- ¿Algún step falló sin skip_on_fail? → abortar ----
        failed = [
            i for i, ok in zip(wave, results)
            if not ok and not pipeline.plan[i].skip_on_fail
        ]
        if failed:
            await store.pipeline_update_status(pipeline_id, PipelineStatus.aborted)
            await store.event_append(
                pipeline_id, "PIPELINE_ABORTED",
                {"at_wave": wave_num, "failed_steps": failed},
            )
            return

        await store.event_append(
            pipeline_id, "WAVE_COMPLETED", {"wave": wave_num, "steps": wave}
        )

        # ---- Supervised: pausar después de cada ola ----
        if pipeline.mode == "supervised":
            await store.pipeline_update_status(
                pipeline_id, PipelineStatus.interrupted, next_idx, pipeline.context
            )
            await store.event_append(
                pipeline_id, "PIPELINE_INTERRUPTED",
                {"after_wave": wave_num, "next_index": next_idx,
                 "reason": "supervised — awaiting /resume"},
            )
            return

    # ---- Todas las olas terminaron ----
    await store.pipeline_update_status(
        pipeline_id, PipelineStatus.completed, len(pipeline.plan), pipeline.context
    )
    await store.event_append(pipeline_id, "PIPELINE_COMPLETED")


async def _persist_step_to_repo(
    pipeline_id: str,
    pipeline_name: str,
    step_index: int,
    facet: str,
    capability: str,
    raw_output: dict,
) -> None:
    """Guarda el output de un step como .md en ~/jax/repo/documents/"""
    import aiofiles
    from pathlib import Path

    repo_dir = Path(os.path.expanduser("~/jax/repo/documents"))
    repo_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{pipeline_id[:8]}_{step_index:02d}_{facet}.md"
    filepath = repo_dir / filename

    result_text = raw_output.get("result", "")
    sources     = raw_output.get("sources", [])
    model       = raw_output.get("model", "desconocido")
    success     = raw_output.get("success", False)

    lines = [
        f"# {pipeline_name}",
        f"",
        f"| Campo | Valor |",
        f"|-------|-------|",
        f"| Pipeline | `{pipeline_id}` |",
        f"| Step | {step_index + 1} |",
        f"| Faceta | {facet} |",
        f"| Capability | {capability} |",
        f"| Modelo | {model} |",
        f"| Estado | {'✓ completado' if success else '✗ fallido'} |",
        f"",
        f"## Respuesta",
        f"",
        result_text,
    ]

    if sources:
        lines += ["", "## Fuentes", ""]
        for s in sources:
            title = s.get("title") or s.get("url", "")
            url   = s.get("url", "")
            lines.append(f"- [{title}]({url})")

    content = "\n".join(lines)

    async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
        await f.write(content)

    logger.info("Step output persistido: %s", filename)


async def _fail_step(
    pipeline: Pipeline, step: Step, step_index: int, error: str
) -> None:
    step.status      = StepStatus.failed
    step.error       = error
    step.finished_at = time.time()
    await store.step_upsert(step)
    await store.event_append(
        pipeline.pipeline_id, "STEP_FAILED",
        {"step_index": step_index, "error": error},
        step.step_id,
    )
    if not step.skip_on_fail:
        await store.pipeline_update_status(pipeline.pipeline_id, PipelineStatus.aborted)
        await store.event_append(
            pipeline.pipeline_id, "PIPELINE_ABORTED",
            {"at_step": step_index, "reason": error},
        )
