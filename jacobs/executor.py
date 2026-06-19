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


def _build_context_input(step: Step, pipeline: Pipeline) -> dict:
    """Construye el input enriquecido con contexto de steps anteriores."""
    objective = pipeline.context.get("objective", "")
    previous_outputs: list[dict] = []

    for j in range(step.step_index):
        ref = pipeline.context.get(f"step_{j}_ref", "")
        if not ref:
            continue
        facet_name = (
            pipeline.plan[j].facet
            if j < len(pipeline.plan) else "unknown"
        )
        try:
            data = _load_ref(ref)
            result_text = data.get("result") or data.get("text") or json.dumps(data)
            summary = str(result_text)[:500]
        except Exception:  # noqa: BLE001
            summary = f"[ref: {ref}]"
        previous_outputs.append({
            "step_index": j,
            "facet": facet_name,
            "summary": summary,
        })

    return {
        "objective": objective,
        "previous_outputs": previous_outputs,
        "prompt": step.input.get("prompt", ""),
    }


def _enrich_prompt(ctx_input: dict) -> str:
    """Construye el prompt final incluyendo contexto previo."""
    parts: list[str] = []

    if ctx_input.get("objective"):
        parts.append(f"Objetivo del pipeline: {ctx_input['objective']}")

    prev = ctx_input.get("previous_outputs", [])
    if prev:
        parts.append("\nContexto de pasos anteriores:")
        for p in prev:
            parts.append(
                f"  Paso {p['step_index'] + 1} ({p['facet']}): {p['summary']}"
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
    api_key = os.environ.get("ZHIPU_API_KEY", "")
    if not api_key:
        return {
            "success": False,
            "facet":   "ada",
            "result":  "Ada no disponible — ZHIPU_API_KEY no configurada.",
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
    payload = {"model": model, "messages": messages, "stream": False}

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Z.ai HTTP {resp.status_code}: {resp.text[:200]}")
        data  = resp.json()
        texto = data["choices"][0]["message"].get("content", "")

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


async def _invoke_motor(step: Step, timeout: int) -> dict:
    """Kimi via Motor Registry de LAS MANOS. Polling hasta completar."""
    payload = {
        "caller":     "jacobs",
        "capability": step.capability,
        "motor":      step.facet,
        "trace_id":   step.trace_id,
        "prompt":     step.input.get("prompt", json.dumps(step.input)),
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{LAS_MANOS_BASE}/motor/dispatch", json=payload)
        resp.raise_for_status()
        dispatch = resp.json()

    job_id = dispatch.get("job_id")
    if dispatch.get("status") == "rejected":
        raise RuntimeError(
            f"Motor Registry rechazó el job: {dispatch.get('rejected_reason', 'sin razón')}"
        )
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
#  Dispatcher principal
# ----------------------------------------------------------------

async def _dispatch_step(step: Step, pipeline: Pipeline) -> dict:
    """Selecciona el worker correcto según la faceta."""
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
#  Pipeline runner
# ----------------------------------------------------------------

async def run_pipeline(pipeline: Pipeline) -> None:
    """
    Ejecuta todos los steps del pipeline en secuencia.
    Modos:
      dry_run    — no ejecuta nada, completa inmediatamente.
      supervised — pausa con status=interrupted después de cada step;
                   espera reanudación explícita vía /resume.
    Hyde siempre se bloquea (blocked_human_gate) hasta /approve-step.
    """
    pipeline_id = pipeline.pipeline_id

    if pipeline.mode == "dry_run":
        await store.pipeline_update_status(pipeline_id, PipelineStatus.completed)
        await store.event_append(pipeline_id, "DRY_RUN_COMPLETE", {"steps": len(pipeline.plan)})
        return

    # Capturar el índice de inicio ANTES de actualizar el estado.
    # En supervised, solo el step en este índice ejecuta; los siguientes quedan bloqueados.
    start_index = pipeline.current_step_index

    await store.pipeline_update_status(
        pipeline_id, PipelineStatus.running, pipeline.current_step_index, pipeline.context
    )
    await store.event_append(pipeline_id, "PIPELINE_STARTED")

    for i, step in enumerate(pipeline.plan):
        if i < pipeline.current_step_index:
            continue

        # ---- Kill switch ----
        if check_kill_switch():
            step.status = StepStatus.failed
            step.error  = "Kill switch activo"
            await store.step_upsert(step)
            await store.event_append(
                pipeline_id, "KILL_SWITCH_ABORTED", {"step_index": i}, step.step_id
            )
            await store.pipeline_update_status(pipeline_id, PipelineStatus.aborted)
            return

        # ---- Hyde: bloquear hasta aprobación manual (marca en context) ----
        if step.facet == "hyde" and not pipeline.context.get(f"hyde_approved_{step.step_id}"):
            step.status = StepStatus.blocked_human_gate
            step.error  = "Hyde requiere aprobación manual de Fernando"
            await store.step_upsert(step)
            await store.event_append(
                pipeline_id, "STEP_BLOCKED_HUMAN_GATE",
                {"step_index": i, "facet": "hyde"},
                step.step_id,
            )
            await store.pipeline_update_status(
                pipeline_id, PipelineStatus.interrupted, i, pipeline.context
            )
            await store.event_append(
                pipeline_id, "PIPELINE_INTERRUPTED",
                {"at_step": i, "reason": "hyde — requiere /approve-step"},
            )
            return

        # ---- Supervised: ejecutar solo el step en start_index; bloquear los siguientes ----
        if pipeline.mode == "supervised" and i > start_index:
            step.status = StepStatus.blocked
            await store.step_upsert(step)
            await store.event_append(
                pipeline_id, "STEP_BLOCKED",
                {"step_index": i, "reason": "supervised — awaiting resume"},
                step.step_id,
            )
            await store.pipeline_update_status(
                pipeline_id, PipelineStatus.interrupted, i, pipeline.context
            )
            await store.event_append(
                pipeline_id, "PIPELINE_INTERRUPTED",
                {"at_step": i, "reason": "supervised — next step requires resume"},
            )
            return

        # ---- Ejecutar step ----
        step.status     = StepStatus.running
        step.started_at = time.time()
        await store.step_upsert(step)
        await store.event_append(
            pipeline_id, "STEP_STARTED",
            {"step_index": i, "facet": step.facet, "capability": step.capability},
            step.step_id,
        )

        try:
            raw_output = await asyncio.wait_for(
                _dispatch_step(step, pipeline),
                timeout=step.timeout_seconds,
            )

            # ---- Artifact handling ----
            ref, inline = save_if_large(pipeline_id, step.step_id, raw_output)
            if ref:
                step.output_ref             = ref
                pipeline.context[f"step_{i}_ref"] = ref
            else:
                inline_ref = f"inline:{json.dumps(inline, ensure_ascii=False)}"
                step.output_ref             = inline_ref
                pipeline.context[f"step_{i}_ref"] = inline_ref

            step.status      = StepStatus.completed
            step.finished_at = time.time()
            await store.step_upsert(step)
            await store.pipeline_update_status(
                pipeline_id, PipelineStatus.running, i + 1, pipeline.context
            )
            await store.event_append(
                pipeline_id, "STEP_COMPLETED",
                {"step_index": i, "output_ref": step.output_ref},
                step.step_id,
            )

        except asyncio.TimeoutError:
            await _fail_step(pipeline, step, i, f"Timeout ({step.timeout_seconds}s)")
            if not step.skip_on_fail:
                return

        except Exception as exc:  # noqa: BLE001
            await _fail_step(pipeline, step, i, str(exc))
            if not step.skip_on_fail:
                return

    # ---- Todos los steps terminaron ----
    await store.pipeline_update_status(
        pipeline_id, PipelineStatus.completed, len(pipeline.plan), pipeline.context
    )
    await store.event_append(pipeline_id, "PIPELINE_COMPLETED")


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
