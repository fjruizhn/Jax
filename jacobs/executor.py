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
import shutil
import time
import tomllib
from pathlib import Path
from facet_resolver import resolve_facet, ResolvedFacet
from contrato_dispatch import limite_de_salida
from model_catalog import record_resolved_version_safe

from jacobs import store
from jacobs import aviso
from jacobs.store import espera_de_turno_sin_plazo  # R38: sobrevive a los tests que reemplazan `store`
from jacobs.artifacts import read_artifact, save_if_large
# Vive en jax/core (capa base, compartido con el HttpMuscle del REPL); llega a
# este proceso por el symlink las_manos/grounding_sources.py, como facet_resolver.
from grounding_sources import build_sources, render_sources_block, resolve_redirects
# Ruling T6-6: jax/core/redaccion.py por symlink en las_manos/, como arriba.
from redaccion import recortar_redactado, redactar_secretos
# E-21: jax/core/config_entorno.py por symlink en las_manos/, como arriba.
from config_entorno import ruta_absoluta_requerida, url_requerida
from cliente_http_compartido import obtener_cliente_http
from auth_servicio import IDENTIDAD_JACOBS, encabezado_propio
from jacobs.models import HTTP_FACETS as _HTTP_FACETS
from jacobs.models import MOTOR_FACETS as _MOTOR_FACETS
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus
from jacobs.plan import CapabilityUnbound
from jacobs.policy import check_kill_switch
from jacobs.usage_writer import record_direct_usage
from hyde_sandbox import run_sandboxed_claude
from interruptor import correr_con_interruptor

logger = logging.getLogger("jacobs.executor")

# E-21 (2026-09-16): del entorno, validadas al importar. Sin ellas LAS MANOS no
# arranca (EntornoInvalido en el journal) en vez de apuntar a un host fijo.
LAS_MANOS_BASE = url_requerida("LAS_MANOS_URL")
OLLAMA_URL     = url_requerida("JAX_OLLAMA_URL") + "/api/chat"

# E-22 (2026-09-16): `documents/` dentro de JAX_REPO_BASE, la MISMA variable
# con la que jax-platform (api/admin/repository.py, REPO_BASE) lista y sirve
# estos .md. Validada al importar: sin ella LAS MANOS no arranca.
REPO_DOCUMENTS_DIR = ruta_absoluta_requerida("JAX_REPO_BASE") / "documents"

MOTOR_POLL_INTERVAL = 5  # segundos entre polls de job

# Tope de seguridad para el output COMPLETO de cada dependencia declarada (~15K tokens).
# Si el ensamble de muchas deps roza la ventana, ajustar y re-verificar con el log de C1.
MAX_DEP_CONTEXT_CHARS = 60_000

# T2 (2026-08-21): _HTTP_FACETS/_MOTOR_FACETS ahora viven en jacobs.models
# (import de arriba) -- plan.py los necesita para la validación pre-persist
# y no puede importar este módulo (circular: executor.py ya importa de
# plan.py). Un solo lugar define la partición, dos módulos la consumen.


# ----------------------------------------------------------------
#  Hyde (v0.3) — system prompt real desde el MISMO config.toml que usa el
#  CLI viejo (jax/core/main.py → SubprocessMuscle). No se reinventa un prompt
#  corto para Jacobs como con thot/ada/jekyll: la identidad de Hyde ya está
#  afinada (Fernando + DeepSeek + Claude) y probada en producción. Fail-open:
#  si config.toml no está o no tiene la sección, Hyde arranca con un prompt
#  mínimo en vez de tumbar el step.
# ----------------------------------------------------------------
_PERSONALITIES_PATH = Path(__file__).resolve().parent.parent / "config" / "config.toml"
try:
    with open(_PERSONALITIES_PATH, "rb") as _pf:
        _HYDE_CFG: dict = tomllib.load(_pf).get("personalities", {}).get("hyde", {})
    _HYDE_SYSTEM_PROMPT = (_HYDE_CFG.get("system_prompt") or "").strip()
    if not _HYDE_SYSTEM_PROMPT:
        raise ValueError("system_prompt vacío o ausente en [personalities.hyde]")
except Exception as _hyde_cfg_err:  # noqa: BLE001  # fail-soft: es la persona de Hyde (--append-system-prompt), no un control de autoridad -- el sandbox y la aprobación de steps siguen aplicando, el fallback conserva "nada destructivo sin confirmación" y _EVIDENCE_RULE se inyecta aparte en cada step
    logger.warning(
        "Jacobs no pudo leer [personalities.hyde] de %s: %s — Hyde arranca con "
        "prompt mínimo", _PERSONALITIES_PATH, _hyde_cfg_err,
    )
    _HYDE_SYSTEM_PROMPT = (
        "Sos Hyde, la faceta técnica de JAX. Sé directo, verificá antes de "
        "afirmar, nada destructivo sin confirmación explícita."
    )

# jax-las-manos.service corre bajo systemd con PATH mínimo (sin el bin de
# nvm) — "claude" a secas resuelve en shell interactivo pero NO en el
# servicio real. shutil.which cubre el caso interactivo/dev; el fallback
# absoluto (documentado como ruta canónica de Node en CLAUDE.md) cubre el
# servicio. Verificado con evidencia: systemctl show jax-las-manos -p
# Environment está vacío, y systemd sin PATH propio usa el default de
# /etc/environment, que no incluye ~/.nvm.
HYDE_CLI_PATH = (
    shutil.which("claude")
    or "/home/fruiz/.nvm/versions/node/v24.16.0/bin/claude"
)
HYDE_WORKSPACE_DIR   = os.getenv("JAX_WORKSPACE_DIR", "/home/fruiz/jax-workspace")
HYDE_MAX_PROMPT_CHARS = 32000


# ----------------------------------------------------------------
#  Helpers de referencia de artifacts
# ----------------------------------------------------------------

class RefIlegible(RuntimeError):
    """La ref EXISTE pero su contenido no se pudo leer.

    ARREGLADO 2026-09-16. Antes `_load_ref` devolvía `{}` en los tres casos:
    «no hay ref», «el JSON inline está roto» y «el artifact no se puede leer».
    Ese `{}` viajaba río abajo como si fuera la salida real de la dependencia:
    `_build_context_input` lo convertía en la cadena `'{}'` y se la daba al
    step siguiente como el output de aquello de lo que depende, y
    `_assemble_mechanical` concatenaba un módulo VACÍO al documento final,
    que igual se devolvía con `success: True`. Error tragado -> paso completado
    con producto incorrecto, que es exactamente lo que P10 prohíbe.

    Ahora la ausencia y el fallo son distinguibles, y cada consumidor decide:
    una dependencia declarada que no se puede leer TUMBA el step; un módulo
    que falta hace que el paquete NO sea exitoso.
    """


def _load_ref(ref: str) -> dict:
    """Carga un output desde su ref (inline o artifact).

    Devuelve `{}` SOLO cuando no hay ref. Si hay ref y no se puede leer,
    lanza `RefIlegible`: quien llama tiene que decidir, no heredar un vacío
    indistinguible de «no había nada».
    """
    if not ref:
        return {}
    if ref.startswith("inline:"):
        try:
            return json.loads(ref[7:])
        except (json.JSONDecodeError, ValueError) as exc:
            raise RefIlegible(f"inline ilegible: {exc}") from exc
    if ref.startswith("artifact://"):
        try:
            return read_artifact(ref)
        except Exception as exc:  # noqa: BLE001
            raise RefIlegible(f"artifact ilegible ({ref}): {exc}") from exc
    raise RefIlegible(f"ref con formato desconocido: {ref!r}")


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
        perdido = False
        try:
            data = _load_ref(ref)
            result_text = data.get("result") or data.get("text") or json.dumps(data)
            text = str(result_text)
            # Las fuentes viajan con el texto (2026-09-12): antes solo pasaba
            # `result` y el paso que audita no veía ninguna fuente.
            if data.get("sources"):
                text += (
                    "\n\nFuentes verificables (URL final + fragmento de la respuesta que respaldan):\n"
                    + render_sources_block(data["sources"])
                )
            if full:
                content = text[:MAX_DEP_CONTEXT_CHARS]
                truncated = len(text) > MAX_DEP_CONTEXT_CHARS
            else:
                content = text[:500]
                truncated = len(text) > 500
        except RefIlegible as exc:
            # Una dependencia DECLARADA (`depends_on`) que no se puede leer no
            # es contexto de cortesía: el step no puede hacer su trabajo sin
            # ella. Antes se sustituía por el literal "[ref: ...]" con
            # truncated=False —— una mentira doble: ni está truncado, ni está.
            if full:
                logger.error(
                    "Jacobs step %s: dependencia declarada step_%d ilegible (%s) —— se corta el step",
                    step.step_index, j, exc,
                )
                raise
            logger.warning(
                "Jacobs step %s: contexto opcional del step_%d ilegible (%s) —— se sigue sin él",
                step.step_index, j, exc,
            )
            content = f"[contexto no disponible: {exc}]"
            truncated = False
            perdido = True
        except Exception:  # noqa: BLE001  # fail-soft: armar el resumen no debe tumbar un step que no declaró esta dep
            if full:
                raise
            logger.warning("Jacobs step %s: no se pudo resumir el step_%d", step.step_index, j, exc_info=True)
            content = f"[contexto no disponible]"
            truncated = False
            perdido = True
        previous_outputs.append({
            "step_index": j,
            "facet": facet_name,
            "summary": content,
            "truncated": truncated,
            "perdido": perdido,
        })

    total_chars = sum(len(p["summary"]) for p in previous_outputs)
    logger.info(
        "Jacobs step %s deps=%s contexto=%d chars%s",
        step.step_index, deps, total_chars,
        (" [ALGUNA DEP TRUNCADA]" if any(p.get("truncated") for p in previous_outputs) else "")
        + (" [ALGUNA DEP PERDIDA]" if any(p.get("perdido") for p in previous_outputs) else ""),
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

class PasoTruncado(Exception):
    """El proveedor corto la salida por tope de longitud. Entregar el texto a
    medias es peor que fallar: el paso siguiente construye sobre una frase
    cortada y nadie se entera. Fallo cerrado, y el pipeline queda continuable
    (cae en el `except Exception` de `_run_one_step` -> `_fail_step`, el mismo
    camino que cualquier otro error de un step; nada nuevo que mantener ahí).

    Espejo del camino de motor: las_manos/motor_registry/worker.py:828 ya
    falla el job cuando `finish_reason == "length"`. Esto cubre los tres
    transportes HTTP directos, que ni leían ese campo."""
    codigo = "paso_truncado"


# Cada proveedor nombra el corte de longitud distinto -- no se adivina, se
# verifica contra la respuesta real de cada API. Un solo lugar que lo sepa.
_CORTE_POR_LONGITUD = {
    "openai_compat": ("length",),
    "ollama": ("length",),
    "gemini": ("MAX_TOKENS",),
}


def _texto_o_truncado(data: dict, transporte: str) -> str:
    """Lee el texto de la respuesta cruda de `transporte` y lo devuelve, o
    levanta `PasoTruncado` si el proveedor cortó por tope de longitud.

    Único lector de "¿este texto vino completo?" para los tres transportes
    HTTP directos -- evita que cada `_invoke_*` reimplemente (o se olvide de)
    el nombre del campo de corte, que es distinto en cada API."""
    if transporte == "openai_compat":
        eleccion = (data.get("choices") or [{}])[0]
        razon = eleccion.get("finish_reason")
        texto = (eleccion.get("message") or {}).get("content", "")
    elif transporte == "ollama":
        razon = data.get("done_reason")
        texto = (data.get("message") or {}).get("content", "")
    elif transporte == "gemini":
        candidato = (data.get("candidates") or [{}])[0]
        razon = candidato.get("finishReason")
        partes = (candidato.get("content") or {}).get("parts") or [{}]
        texto = "".join(p.get("text", "") for p in partes)
    else:
        raise ValueError(f"transporte sin lector de truncado: {transporte}")

    if razon in _CORTE_POR_LONGITUD.get(transporte, ()):
        raise PasoTruncado(
            f"{transporte} corto la salida por longitud ({razon}); "
            f"{len(texto)} caracteres entregados"
        )
    return texto


async def _invoke_http_gemini(f: "ResolvedFacet", prompt: str, timeout: int) -> dict:
    """Formato Gemini + grounding required_web. Transporte, no faceta —
    hoy solo hipatia lo usa, pero cualquier facet con transport=http_gemini
    entra aca sin codigo nuevo."""
    model = f.model
    # Ruling T6-6 (2026-09-15): la key va en la cabecera x-goog-api-key, NO en
    # `?key=`. httpx loguea la URL entera en INFO en cada pedido y la mete en
    # str(HTTPStatusError): con la key en la query quedaba en claro.
    url = f"{f.base_url}/models/{model}:generateContent"
    headers = {"x-goog-api-key": f.credential}
    contents = [{"role": "user", "parts": [{"text": prompt}]}]
    payload  = {
        "contents": contents,
        "tools": [{"google_search": {}}],
    }

    async def _call() -> dict:
        resp = await obtener_cliente_http().post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            # Google devuelve la key rechazada DENTRO del cuerpo del error:
            # redactar antes de recortar (y este texto termina en
            # jacobs_steps.error via _fail_step).
            cuerpo = recortar_redactado(resp.text, 200, [f.credential])
            raise RuntimeError(f"Gemini HTTP {resp.status_code}: {cuerpo}")
        return resp.json()

    data = await _call()
    final_data = data
    candidate = data.get("candidates", [{}])[0]
    # E-25: PasoTruncado si Gemini cortó por MAX_TOKENS -- antes de leer
    # groundingMetadata, que igual no importa sobre una respuesta a medias.
    texto = _texto_o_truncado(data, "gemini")
    meta  = candidate.get("groundingMetadata", {}) or {}
    chunks = meta.get("groundingChunks") or []
    supports = meta.get("groundingSupports") or []
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
        # Mismo control en el retry: el corte por longitud puede pasar en
        # cualquiera de los dos llamados, no solo en el primero.
        texto2  = _texto_o_truncado(data2, "gemini")
        meta2   = candidate2.get("groundingMetadata", {}) or {}
        chunks2 = meta2.get("groundingChunks") or []
        if chunks2:
            texto  = texto2
            chunks = chunks2
            queries = meta2.get("webSearchQueries") or []
            supports = meta2.get("groundingSupports") or []
            final_data = data2

    # Usage (scope expansion 2026-08-10, mismo campo que jax-platform/backend/
    # api/chat.py::_call_gemini): tokens de la respuesta que realmente aporto
    # el `texto` final -- si hubo retry por falta de grounding, es data2, no
    # el primer data (nota: si hubo retry, el primer llamado tambien consumio
    # tokens y no se contabilizan aca; limitacion conocida, ver reporte).
    gemini_usage = final_data.get("usageMetadata") or {}
    tokens_in  = gemini_usage.get("promptTokenCount", 0)
    tokens_out = gemini_usage.get("candidatesTokenCount", 0)

    # Fuentes verificables (2026-09-12): URL final (siguiendo la redirección
    # opaca de Google) + los fragmentos que cada una respalda. Antes eran
    # solo redirecciones con una etiqueta de dominio: la auditoría no podía
    # contrastar ninguna (E2E b2d87971). Ver jax/core/grounding_sources.py.
    sources = build_sources(chunks, supports)
    await resolve_redirects(sources)

    # D1.2 — 'modelVersion' es el campo real de Gemini (distinto de 'model'
    # que usan las APIs OpenAI-compatible abajo). Nota de incertidumbre:
    # heredado de jax-platform, nunca verificado contra una respuesta real
    # de Gemini con curl — ver CONTEXT.md.
    await record_resolved_version_safe(f.key, data.get("modelVersion"))

    return {
        "success": True,
        "facet":   f.key,
        "model":   model,
        "result":  texto,
        "sources": sources,
        "queries": queries,
        "grounded": bool(chunks),
        "tokens_in":  tokens_in,
        "tokens_out": tokens_out,
    }


async def _invoke_http_openai_compat(f: "ResolvedFacet", prompt: str, timeout: int) -> dict:
    """Formato chat/completions estilo OpenAI. Transporte, no faceta —
    jekyll/thot/ada convergen aca (antes eran 3 copias casi identicas con
    URL/modelo/persona hardcodeados). Sin streaming: jekyll/thot ya lo
    probaban sin streaming; ada pierde SSE a cambio de una sola funcion
    para las 3 (simplificacion deliberada, el contenido final es
    equivalente, solo cambia como se ensambla)."""
    url = f"{f.base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {f.credential}", "Content-Type": "application/json"}
    messages = []
    if f.persona:
        messages.append({"role": "system", "content": f.persona})
    messages.append({"role": "user", "content": prompt})
    # PR-K ronda 1: límite de salida con nombre y tope de la fila de `model`
    # de f.model (jax/core/contrato_dispatch.py). Antes no mandaba ninguno. Sin
    # contrato, ModelDispatchConfigError sube y el step falla con el UPDATE.
    payload = {"model": f.model, "messages": messages, "stream": False,
               **await limite_de_salida(f.transport, f.provider_id, f.model)}

    resp = await obtener_cliente_http().post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"[{f.key}] HTTP {resp.status_code}: {recortar_redactado(resp.text, 200, [f.credential])}")
    data  = resp.json()
    # E-25: PasoTruncado si el proveedor cortó por longitud (finish_reason
    # "length") en vez de entregar el texto a medias como resultado bueno.
    texto = _texto_o_truncado(data, "openai_compat")

    # D1.2 — best-effort, fuera del context manager del client: nunca debe
    # poder romper la respuesta al step (record_resolved_version_safe ya
    # atrapa sus propias excepciones).
    await record_resolved_version_safe(f.key, data.get("model"))

    # Usage (scope expansion 2026-08-10, mismo campo que jax-platform/backend/
    # api/chat.py::_call_openai_compat): jekyll/thot/ada convergen aca, asi
    # que este solo lugar cubre los 3.
    usage = data.get("usage") or {}
    tokens_in  = usage.get("prompt_tokens", 0)
    tokens_out = usage.get("completion_tokens", 0)

    return {
        "success": True,
        "facet":   f.key,
        "model":   f.model,
        "result":  texto,
        "tokens_in":  tokens_in,
        "tokens_out": tokens_out,
    }


async def _invoke_ollama(f: "ResolvedFacet", prompt: str, timeout: int) -> dict:
    """Ollama local — razonamiento local. Modelo desde facet_binding, ya no
    hardcodeado (ver plan.py::_llm_plan, mismo patron). f.base_url
    ("http://localhost:11434/v1") es para el path OpenAI-compat generico
    (_call_openai_compat) — el endpoint nativo /api/chat que este payload
    espera (respuesta en data["message"]["content"]) es siempre local y
    fijo. Solo el modelo viene del facet, nunca la URL — bug real hasta
    2026-08-19 (404 por concatenar
    f.base_url + "/api/chat"). jax_local es una faceta activa de la tabla facet (E-03),
    solo no aparece en la lista de facetas que _llm_plan le sugiere al LLM
    para auto-generar steps — un pipeline con step facet="jax_local" armado
    a mano (_from_spec) si lo hubiera disparado.
    OJO: GPU_SEMAPHORE (jax/muscles/ollama_muscle.py::GPU_SEMAPHORE -- por simbolo, no por linea: la referencia decia :37 y el simbolo ya se habia movido) es un
    asyncio.Semaphore de PROCESO del REPL de JAX -- esta llamada corre en el
    proceso de jax-las-manos y le pega a Ollama directo por httpx, sin pasar
    por ese semáforo. No hay exclusión mutua real entre el REPL y Jacobs
    para el acceso a la GPU (verificado 2026-08-19, sonda T0.a/T1 de
    latencia de _llm_plan).

    MEDIDO 2026-08-28: esa falta de exclusion mutua no produce contencion
    hoy porque Ollama serializa (OLLAMA_NUM_PARALLEL=1) -- la generacion se
    mantiene en ~76.7 tok/s con 1, 2 y 3 requests concurrentes y lo unico
    que crece es la espera en cola. Veredicto y que lo reabre:
    docs/superpowers/specs/2026-08-25-gpu-concurrency-resultado.md"""
    # PR-K ronda 2 (M3): límite de salida de la fila de `model`, como
    # options.num_predict de /api/chat (github.com/ollama/ollama docs/api.md).
    # Antes no mandaba ninguno. Sin contrato, el step falla con el UPDATE.
    payload = {
        "model":    f.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream":   False,
        **await limite_de_salida(f.transport, f.provider_id, f.model),
    }
    resp = await obtener_cliente_http().post(OLLAMA_URL, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {resp.status_code}: {recortar_redactado(resp.text, 200)}")
    data  = resp.json()
    # E-25: PasoTruncado si Ollama cortó por longitud (done_reason "length").
    texto = _texto_o_truncado(data, "ollama")

    # D1.2 — capturado por consistencia con los transportes HTTP; ver
    # CONTEXT.md para la limitacion real (tags de Ollama no son alias
    # moviles del proveedor, no detecta drift de pesos bajo el mismo tag).
    await record_resolved_version_safe(f.key, data.get("model"))

    # Usage (scope expansion 2026-08-10, mismo campo que jax-platform/backend/
    # api/chat.py::_call_ollama).
    tokens_in  = data.get("prompt_eval_count", 0)
    tokens_out = data.get("eval_count", 0)

    return {
        "success": True,
        "facet":   f.key,
        "model":   f.model,
        "result":  texto,
        "tokens_in":  tokens_in,
        "tokens_out": tokens_out,
    }


async def _invoke_hyde(f: "ResolvedFacet", prompt: str, timeout: int) -> dict:
    """Claude Code CLI (binario `claude`) como subproceso headless — mismo
    mecanismo de jax/muscles/subprocess_muscle.py, en producción hace meses
    en el CLI viejo. Adaptado a la firma de Jacobs: sin serialización de
    historial (Jacobs ya arma el contexto completo en `prompt` vía
    _enrich_prompt, antes de llegar acá — igual que para las demás facetas)."""
    model = f.model

    safe_prompt = prompt
    if len(safe_prompt) > HYDE_MAX_PROMPT_CHARS:
        safe_prompt = safe_prompt[:HYDE_MAX_PROMPT_CHARS] + "\n[...truncado por Jacobs...]"

    cmd = [
        HYDE_CLI_PATH,
        "--model", model,
        "--append-system-prompt", _HYDE_SYSTEM_PROMPT,
        "--print",
        "--output-format", "text",
        "--permission-mode", "acceptEdits",
        # Bash SIN acotar por patron (2026-08-22, configuracion definitiva
        # post-sandbox -- ver hyde_sandbox.py y jax-hyde-bash-sin-jail-p0 en
        # memoria). El PR#18 (pwd/ls) fue andamio TEMPORAL mientras no habia
        # confinamiento real: --allowedTools nunca fue una defensa de
        # filesystem que sirviera (Bash pelado no tenia jail; "Bash(<cmd> *)"
        # con parentesis solo cubria cat/redireccion, python3 -c
        # "open(path).read()" y `git diff --no-index` lo esquivaban igual,
        # confirmado). Ahora la defensa real es el namespace de montaje de
        # bwrap (ver hyde_sandbox.py::run_sandboxed_claude, abajo): lo que no
        # esta bind-mounteado no existe, sin importar el comando. Verificado
        # en T5 (13 casos adversariales, incluidos estos dos bypasses) CON
        # "Bash" pelado -- todo bloqueado por el sandbox, cero ayuda del
        # allowlist. Restringir el allowlist ahora solo volveria a
        # inutilizar a Hyde sin sumar seguridad real -- la capa que importa
        # es la de abajo.
        "--allowedTools", "Write,Edit,Read,Bash",
        "--add-dir", HYDE_WORKSPACE_DIR,
    ]

    # Sandbox de bubblewrap + lock cross-proceso via flock(2) (ver
    # hyde_sandbox.py::run_sandboxed_claude -- unico punto de entrada
    # aprobado para lanzar `claude`, DEUDA.md "gobernanza de sub-agentes").
    # SandboxUnavailable NO se atrapa acá -- fail-closed (P10): sin bwrap,
    # el step falla con motivo explícito (_run_one_step ya lo hace vía su
    # except Exception genérico), nunca corre Hyde sin confinamiento.
    # TimeoutError/CancelledError tampoco se atrapan acá -- run_sandboxed_claude
    # ya mató y cosechó el proceso, y necesitamos que la excepción de
    # asyncio se propague SIN envolver (ver docstring de esa función).
    proc, stdout, stderr = await run_sandboxed_claude(
        cmd, HYDE_WORKSPACE_DIR, safe_prompt, timeout,
    )
    stdout_str = stdout.decode("utf-8", errors="replace")
    stderr_str = stderr.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        raise RuntimeError(f"[hyde] claude exit {proc.returncode}: {stderr_str[:200]}")
    low = stderr_str.lower()
    if any(t in low for t in ("error", "fatal", "exception", "failed")):
        raise RuntimeError(f"[hyde] error en stderr: {stderr_str[:200]}")

    # D1.2 (Bloque D) — deliberadamente SIN captura de resolved_version:
    # --output-format text (arriba) no trae ningun campo de que version
    # corrio de verdad. Ver CONTEXT.md ("decision previa al wiring de
    # resolved_version en REPL/Jacobs").
    return {
        "success": True,
        "facet":   "hyde",
        "model":   model,
        "result":  stdout_str.strip(),
    }


async def _invoke_motor(step: Step, pipeline: Pipeline, timeout: int, prompt: str | None = None) -> dict:
    """Kimi/jax_local via Motor Registry de LAS MANOS. Polling hasta completar.

    Bloque 3 (2026-08-21): _CAPABILITY_MAP eliminado -- resolvía alias
    semánticos ("analysis"->"pipeline_analysis", etc.) a un nombre de
    catálogo, pero verificado contra capability_motor real + jacobs_steps
    histórico: ningún alias tuvo NUNCA una fila en capability_motor ni se
    usó jamás con un facet-motor (kimi/jax_local) -- los 3 que sí se usan
    (analysis/research/review) lo hacen exclusivamente con facets HTTP-directos,
    donde este mapa nunca se consultaba. Muerto, no reemplazado. step.capability
    llega acá ya validado por NIVEL A/B de validate_capability() (existe en
    `capability`, el motor está en su allowed_motors) -- se despacha tal cual,
    sin resolución intermedia."""
    payload = {
        "caller":     "jacobs",
        "capability": step.capability,
        "motor":      step.motor,  # None = MotorPolicy resuelve por competencia (R4)
        "trace_id":   step.trace_id,
        # El prompt ARMADO por _dispatch_step (regla de evidencia + objetivo +
        # salidas de las dependencias + tarea). Antes se reconstruía acá desde
        # step.input y el contexto de las dependencias se perdía: en la E2E de
        # la cadena (1bb0da78, 2026-09-12) kimi tenía que "producir con el
        # plan unificado" sin recibir el plan. El respaldo queda solo para
        # callers directos que no pasan prompt.
        "prompt":     prompt if prompt is not None
                      else _EVIDENCE_RULE + "\n\n" + step.input.get("prompt", json.dumps(step.input)),
        "user_id":    pipeline.user_id,
        "tenant_id":  pipeline.tenant_id,
        # GAP2 Fase3 (2026-08-19): mismo presupuesto que ya gobierna el
        # polling de abajo (deadline = timeout) -- el bucle de tool-calling
        # de worker.py lo consume como SU presupuesto de tiempo, no uno
        # nuevo. Ningun cambio para el polling mismo, que sigue intacto.
        "timeout_seconds": timeout,
    }
    resp = await obtener_cliente_http().post(f"{LAS_MANOS_BASE}/motor/dispatch", json=payload, timeout=30,
                                           headers=encabezado_propio(IDENTIDAD_JACOBS))
    resp.raise_for_status()
    dispatch = resp.json()

    job_id = dispatch.get("job_id")
    if dispatch.get("status") == "rejected":
        reason = dispatch.get("rejected_reason", "sin razón")
        logger.error(
            "Motor Registry RECHAZÓ job (caller=jacobs, capability=%s, motor=%s): %s",
            step.capability, step.motor or step.facet, reason,
        )
        raise RuntimeError(f"Motor Registry rechazó el job: {reason}")
    if not job_id:
        raise RuntimeError(f"Motor Registry no devolvió job_id: {dispatch}")

    # Polling hasta timeout
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(MOTOR_POLL_INTERVAL)
            resp = await obtener_cliente_http().get(f"{LAS_MANOS_BASE}/motor/job/{job_id}", timeout=15,
                                                      headers=encabezado_propio(IDENTIDAD_JACOBS))
            resp.raise_for_status()
            job = resp.json()

            status = job.get("status", "")
            if status == "completed":
                # Ronda de arreglo 1 de Task 1 (2026-09-18): con
                # MotorJobView.model expuesto (las_manos/motor_registry/
                # models.py + worker.py, esta misma ronda) el job trae el
                # model_id REAL que despachó -- ya no None por default.
                # kimi/jax_local son las facetas de la mayoría de los pasos
                # reales; sin esto, casi todo el historial decía "Modelo
                # desconocido".
                step.modelo_real = job.get("model")
                return {
                    "success":        True,
                    "facet":          step.facet,
                    "job_id":         job_id,
                    "result":         await _read_motor_result(job),
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
    except (asyncio.CancelledError, asyncio.TimeoutError):
        # Vencer el paso sin avisarle a LAS MANOS deja el job corriendo y
        # cobrando (pipeline b8f80733, 2026-09-12: kimi siguió 3 min después
        # del aborto). Dos caminos llegan acá: el wait_for de _run_step
        # (CancelledError) y el deadline de este polling (TimeoutError).
        await _cancel_motor_job(job_id)
        raise


async def _read_motor_result(job: dict) -> str:
    """Salida COMPLETA de un motor job. `result_summary` son 200 caracteres
    (worker.py); era lo único que llegaba a los pasos siguientes y al repo.
    Desde 2026-09-12 el worker guarda el texto entero en `result_path`.

    Job sin result_path (anterior al arreglo): el resumen es todo lo que
    existe. Job CON result_path ilegible: falla el paso -- pasar 200
    caracteres en silencio como si fueran el producto es exactamente el
    defecto que esto cierra."""
    path = job.get("result_path")
    if not path:
        return job.get("result_summary", "") or ""
    try:
        return await asyncio.to_thread(Path(path).read_text, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"Motor job {job.get('job_id')}: no se pudo leer su salida completa en {path}: {exc}"
        ) from exc


async def _cancel_motor_job(job_id: str) -> None:
    """Pide a LAS MANOS que corte el job. Lo mejor que se puede hacer, no
    una condición: el paso ya venció y eso es lo que se reporta. Si el aviso
    falla, queda en el log con el job_id -- nunca reemplaza la causa real.
    409 = el job ya había terminado solo, no hay nada que cortar."""
    try:
        resp = await obtener_cliente_http().post(f"{LAS_MANOS_BASE}/motor/job/{job_id}/cancel", timeout=5,
                                                   headers=encabezado_propio(IDENTIDAD_JACOBS))
        if resp.status_code not in (200, 409):
            logger.error(
                "No se pudo cancelar el motor job %s tras vencer su paso: HTTP %s",
                job_id, resp.status_code,
            )
    except Exception as exc:  # noqa: BLE001  # fail-soft: el paso ya venció y se reporta fallido por timeout -- cancelar es un aviso best-effort a LAS MANOS, no una condición; el job_id queda en el log de error para cortarlo a mano
        logger.error(
            "No se pudo cancelar el motor job %s tras vencer su paso: %s -- "
            "puede seguir corriendo y cobrando en LAS MANOS",
            job_id, exc,
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
    modulos_perdidos: list[str] = []
    for j in range(step.step_index):
        prev = pipeline.plan[j]
        ref = pipeline.context.get(f"step_{j}_ref", "")
        if not ref:
            continue
        try:
            data = _load_ref(ref)
        except RefIlegible as exc:
            # Antes el modulo entraba VACIO y el paquete salia con success:True.
            # Un documento al que le falta un modulo no es un documento exitoso.
            logger.error("Jacobs ensamble: modulo del step %d ilegible (%s)", j, exc)
            modulos_perdidos.append(f"step {j} ({prev.capability}): {exc}")
            partes.append(f"\n{'='*70}\n## MÓDULO (step {j}): {prev.capability} —— NO DISPONIBLE\n{'='*70}\n")
            partes.append(f"[este módulo no se pudo leer: {exc}]")
            continue
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
    logger.info(
        "Jacobs ensamble mecánico: %d chars de %d módulos%s",
        len(documento), step.step_index,
        f" —— {len(modulos_perdidos)} NO DISPONIBLES" if modulos_perdidos else "",
    )
    salida = {
        "success": not modulos_perdidos,
        "facet": "ada",
        "model": "mechanical_assembler",
        "result": documento,
    }
    if modulos_perdidos:
        salida["error"] = (
            f"{len(modulos_perdidos)} módulo(s) no se pudieron leer y faltan del "
            f"paquete: " + "; ".join(modulos_perdidos)
        )
    return salida


# ----------------------------------------------------------------
#  Dispatcher principal
# ----------------------------------------------------------------

async def validate_capability(step: Step) -> CapabilityUnbound | str | None:
    """Validación PRE-dispatch en DOS NIVELES (FASE A §3.4; Bloque 3
    2026-08-21: fuente única, DB real vía store.get_motor_governance() --
    antes NIVEL A leía jacobs/plan.py::VALID_CAPABILITIES (frozenset
    estático) y NIVEL B leía las_manos/config.toml + _CAPABILITY_MAP
    (ambos también estáticos, cargados una vez al importar este módulo).
    Las tres copias se desincronizaban de la DB sin aviso -- causa raíz de
    dos P0 reales (2026-08-22: VALID_CAPABILITIES sin file_read/file_write
    pese a existir en la DB desde días antes; config.toml con el mismo
    hueco). Ver DEUDA.md para el detalle y la evidencia de drift.

    Separa dos preguntas que antes estaban mezcladas. Devuelve CapabilityUnbound
    o un mensaje de error (str) si el step es inválido, o None si es válido.

    NIVEL A — existencia de vocabulario. Aplica a TODOS los facets, incluido
        hyde (verificado en vivo: jacobs_steps tiene un step real,
        2026-08-21, facet=hyde, capability='execute', rechazado acá con
        exactamente este mensaje -- caso de prueba de T4).
        ¿step.capability existe como fila en `capability`? Cierra la
        asimetría (un facet directo con capability inexistente se rechaza).
        NO mira allowed_motors, por eso NO rompe facets directos cuyo destino
        de catálogo sea kimi-only -- hipatia/research y jekyll/analysis
        (ambos con fila propia en `capability`, sin capability_motor --
        Bloque 3, T2: nunca se resolvían a través de un alias, ambos nombres
        se usan tal cual, sin traducción, desde siempre) PASAN porque
        'research'/'analysis' existen en el vocabulario real. Es el
        _fallback_plan, que no se puede romper.

    NIVEL B — contrato del motor. Aplica SOLO a _MOTOR_FACETS (hoy kimi, jax_local).
        Mismo contrato que policy.check valida en el Motor Registry, adelantado
        acá para fallar limpio antes del HTTP: la capability existe en la DB,
        (step.motor or step.facet) ∈ allowed_motors y el caller
        'jacobs' ∈ allowed_callers. Se valida step.motor cuando está seteado
        porque, desde Task 5, es lo que realmente despacha (_invoke_motor pasa
        step.motor al Motor Registry, no step.facet) — validar solo step.facet
        dejaría pasar un step con facet="kimi", motor="ada" a nombre de kimi
        mientras en realidad despacha ada. Los facets de API directa NO pasan
        por aquí (ignoran capability en el dispatch real).

    Fail-closed (P10): si la DB no responde, store.get_motor_governance()
    propaga la excepción sin capturarla acá -- _run_one_step ya envuelve
    _dispatch_step en un try/except general (línea ~936) que falla el step
    limpio con el motivo real. Sin gobernanza, no se despacha -- nunca un
    "pasa porque no pude verificar". Ningún caso de fail-soft identificado
    para esta función; no se marca ninguno.

    Devuelve CapabilityUnbound (tipado, REFORMAS-v3 R3.4) cuando el motivo
    de rechazo es un binding capability→motor ausente (NIVEL B) — el
    scheduler lo reenruta. Devuelve str para NIVEL A (vocabulario cerrado,
    no es un problema de binding, no tiene candidates que ofrecer).
    """
    cap = step.capability

    # 'assemble' es mecánico (se cortocircuita antes en _dispatch_step); válido.
    if cap == "assemble":
        return None

    governance = await store.get_motor_governance()
    caps = governance["capabilities"]

    # ---- NIVEL A: existencia real en la DB (TODOS los facets) ----
    entry = caps.get(cap)
    if entry is None:
        return f"capability desconocida: '{cap}' no está en la tabla `capability`"

    # ---- NIVEL C (2026-08-27): admisión para _HTTP_FACETS ----
    # Antes de esta ronda, hipatia/jekyll/thot/ada nunca pasaban por
    # ninguno de los checks de MotorPolicy (DEUDA.md, bullet _HTTP_FACETS
    # sin gobernanza). check_capability_admission() cubre allowed_callers/
    # requires_human_gate/recursion_depth/claves prohibidas -- import
    # directo, Jacobs corre en el mismo proceso que las_manos (docs/
    # superpowers/specs/2026-08-27-http-facets-motor-policy-governance-
    # design.md). NO incluye techo de timeout (decisión explícita, punto 2
    # del spec) ni resolución de motor (N/A -- facets HTTP no son motores).
    # Fail-closed: si MotorCatalog.from_db() no puede leer la DB, la
    # excepción se propaga sin capturarla acá (mismo criterio P10 que el
    # resto de esta función) -- _run_one_step falla el step limpio, nunca
    # despacha sin haber podido verificar.
    if step.facet in _HTTP_FACETS:
        from motor_registry.catalog import MotorCatalog
        from motor_registry.policy import MotorPolicy
        catalog = await MotorCatalog.from_db()
        policy = MotorPolicy(catalog)
        admission = policy.check_capability_admission(
            caller="jacobs", capability=cap,
            context_keys=list(step.input.keys()),
            recursion_depth=0, human_gate_token=None,
        )
        if not admission.allowed:
            return admission.reason

    # ---- NIVEL B: contrato de motor (SOLO facets-motor, hoy kimi/jax_local) ----
    if step.facet in _MOTOR_FACETS:
        if (step.motor or step.facet) not in entry["allowed_motors"]:
            return CapabilityUnbound(
                required=[cap],
                candidates=list(entry["allowed_motors"]),
                task_id=step.step_id,
            )
        if "jacobs" not in entry["allowed_callers"]:
            return CapabilityUnbound(
                required=[cap], candidates=[], task_id=step.step_id,
            )
    return None


async def _dispatch_step(step: Step, pipeline: Pipeline) -> dict:
    """Selecciona el worker correcto según la faceta."""
    # Ensamble mecánico: NO pasa por ningún LLM. Concatena los módulos ya generados.
    if step.capability == "assemble":
        return _assemble_mechanical(step, pipeline)

    # FASE A §3.4: validación uniforme del contrato ANTES de rutear por facet.
    # Si falla, el step falla limpio (lo captura _run_one_step → _fail_step) sin
    # haber tocado ninguna API ni el Motor Registry.
    #
    # REFORMAS-v3 R3.4 — CAPABILITY_UNBOUND se intercepta y reenruta a un
    # candidate antes de abortar el pipeline. NIVEL A (str) no tiene
    # candidatos — falla igual que antes. El usuario nunca ve el estado
    # intermedio: si el reroute encuentra un candidato válido, el pipeline
    # sigue como si el step hubiera sido asignado a ese facet desde el inicio.
    original_facet = step.facet
    tried_facets = {step.facet}
    cap_error = await validate_capability(step)
    while isinstance(cap_error, CapabilityUnbound):
        # El reroute SOLO puede apuntar a facets efectivamente despachables
        # (HTTP o Motor Registry). 'hyde' se excluye a propósito: tiene su
        # propio gate de aprobación humana en run_pipeline, que chequea
        # plan[i].facet == "hyde" ANTES de que este código corra — si el
        # reroute pudiera asignar step.facet = "hyde" después de ese
        # chequeo, el step ejecutaría con result["approved"] = True sin
        # aprobación humana real. No alcanzable hoy (ninguna capability
        # lista "hyde" en allowed_motors), pero a una fila de capability_motor
        # de distancia (Bloque 3: la DB es la única fuente ahora, un INSERT
        # directo lo cambiaría). 'jax_local' (R4: sumado a _MOTOR_FACETS, SÍ es un
        # conjunto de dispatch) tampoco aparece hoy como candidato de
        # reroute -- no porque esté excluido del dispatch, sino porque
        # ninguna fila de capability_motor lo lista en allowed_motors
        # todavía (mismo gap que "hyde": a una fila de capability_motor de
        # distancia). NOTA: reroute SÍ puede apuntar a _HTTP_FACETS
        # (ada/thot). Hasta 2026-08-26 eso era un agujero: esos facets no
        # pasaban por NINGÚN check del Motor Registry, así que un reroute
        # los usaba como puerta de atrás. Desde 2026-08-27 ya no: el
        # re-chequeo del final de este bucle vuelve a llamar a
        # validate_capability(), y su bloque NIVEL C aplica
        # check_capability_admission() (checks 1-5: capability existe,
        # allowed_callers, requires_human_gate, recursion_depth, claves
        # prohibidas) al facet NUEVO, no al original. Checks 6-7 (resolver
        # motor, motor.sandbox_only) son N/A — un facet HTTP no es un
        # motor. El techo de timeout YA NO esta diferido: se exige en
        # plan-time desde 2026-09-01 (_validate_plan_capabilities lo valida
        # contra `capability.max_execution_minutes` para TODOS los steps,
        # HTTP incluidos).
        #
        # Residuo real que queda, distinto del viejo: NIVEL C pasa
        # human_gate_token=None fijo, y una denegación de admisión devuelve
        # str, no CapabilityUnbound. O sea: si el reroute aterriza en un
        # facet HTTP con una capability que exige gate humano, el step
        # falla DURO acá abajo (`raise ValueError`) en vez de seguir
        # buscando candidatos. Es el comportamiento correcto (fail-closed),
        # pero es una falla sin reintento — ver DEUDA.md para las
        # capabilities con requires_human_gate=1 alcanzables por esta vía.
        untried = [
            c for c in cap_error.candidates
            if c not in tried_facets and c in (_HTTP_FACETS | _MOTOR_FACETS)
        ]
        if not untried:
            raise ValueError(
                f"Capability inválida (pre-dispatch, candidatos agotados): "
                f"{cap_error.to_dict()} (facet original: {original_facet})"
            )
        new_facet = untried[0]
        logger.warning(
            "Jacobs reroute: step %s capability='%s' facet '%s' -> '%s' "
            "(CAPABILITY_UNBOUND, candidatos=%s)",
            step.step_id, step.capability, step.facet, new_facet, cap_error.candidates,
        )
        await store.event_append(
            pipeline.pipeline_id, "STEP_REROUTED",
            {
                "original_facet": step.facet,
                "new_facet": new_facet,
                "capability": step.capability,
                "candidates": cap_error.candidates,
            },
            step.step_id,
        )
        step.facet = new_facet
        tried_facets.add(new_facet)
        cap_error = await validate_capability(step)
    if isinstance(cap_error, str):
        raise ValueError(f"Capability inválida (pre-dispatch): {cap_error}")

    ctx_input = _build_context_input(step, pipeline)
    prompt    = _enrich_prompt(ctx_input)
    timeout   = step.timeout_seconds

    # Despacho por TRANSPORTE (Bloque C — antes era if/elif por nombre de
    # faceta con modelo/URL hardcodeados en cada rama). resolve_facet() es
    # FAIL-CLOSED: sin binding activo, FacetUnavailableError sube y
    # _run_one_step la captura igual que cualquier otra excepcion — el step
    # falla con motivo explicito, nunca un default silencioso.
    if step.facet in _MOTOR_FACETS:
        # Task 1 (2026-09-18, historial-y-arreglos-de-pipeline): este camino
        # NO pasa por resolve_facet(). Hasta la ronda de arreglo 1 (mismo
        # día), MotorJobView no traía el model_id real -- worker.py SÍ lo
        # conocía (motor_entry.model, worker.py:672/725) pero nunca lo
        # exponía en el job (job_store.py filtraba cualquier campo fuera de
        # MotorJobView.model_fields al leer), así que escribir
        # job.get("motor") acá hubiera guardado el NOMBRE del motor
        # (kimi/jax_local, lo mismo que step.facet ya dice) disfrazado de
        # model_id: el mismo defecto que esta tarea cierra. Con
        # MotorJobView.model expuesto (las_manos/motor_registry/models.py +
        # worker.py), _invoke_motor ahora escribe step.modelo_real de verdad
        # al completar (ver abajo, status == "completed").
        return await _invoke_motor(step, pipeline, timeout, prompt)

    f = await resolve_facet(step.facet)
    step.modelo_real = f.model

    # Transportes HTTP directos (scope expansion 2026-08-10): la Mesa web ya
    # atribuye costo para estas mismas facetas via jax-platform/backend/api/
    # chat.py (Tasks 1-4); esto cubre el MISMO transporte cuando lo dispara un
    # pipeline de Jacobs en vez de un chat directo. record_direct_usage es
    # fail-soft por su cuenta (sin identidad no escribe; error de DB solo
    # loguea) -- nunca puede romper un step ya exitoso.
    if f.transport in ("http_gemini", "http_openai_compat", "ollama"):
        if f.transport == "http_gemini":
            result = await _invoke_http_gemini(f, prompt, timeout)
        elif f.transport == "http_openai_compat":
            result = await _invoke_http_openai_compat(f, prompt, timeout)
        else:
            result = await _invoke_ollama(f, prompt, timeout)
        await record_direct_usage(
            pipeline.user_id, pipeline.tenant_id, step.facet,
            f.provider_id, f.model,
            result.get("tokens_in", 0), result.get("tokens_out", 0),
        )
        return result
    if f.transport == "subprocess":
        # Llegamos aquí solo si Fernando aprobó vía /approve-step (gate de
        # aprobación intacto, no tocado en esta misión).
        result = await _invoke_hyde(f, prompt, timeout)
        result["approved"] = True
        return result

    raise ValueError(f"Transporte desconocido para facet '{step.facet}': '{f.transport}'")


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

class _SinEscritura:
    """Resultado de un paso que NO se pudo escribir con su escritura
    condicional (m2 de la re-revisión final, 2026-09-17). Antes era `False`,
    el mismo valor que "el paso falló": `step_upsert_si_epoca` devuelve 0
    filas tanto si la corrida perdió la época como si la FILA del paso cambió
    (borrada, otro step_id) con la corrida vigente, y en ese segundo caso el
    pipeline abortaba con `"errores": {"3": null}` -- un aborto sin motivo.
    run_pipeline los distingue releyendo la época."""

    def __repr__(self) -> str:  # pragma: no cover - sólo para diagnósticos
        return "<paso sin escribir>"


PASO_SIN_ESCRITURA = _SinEscritura()

_MOTIVO_SIN_ESCRITURA = (
    "no se pudo escribir el paso: su fila cambió (otro step_id o borrada) "
    "mientras la corrida seguía vigente"
)


async def _run_one_step(step: Step, i: int, pipeline: Pipeline) -> bool | _SinEscritura:
    """Ejecuta un step individual. Devuelve True si completó, False si falló y
    PASO_SIN_ESCRITURA si su escritura condicional no tocó ninguna fila
    (época perdida o fila del paso cambiada).

    Época (spec 2026-09-17 §5.3): cada escritura del paso es condicional a
    `pipeline.run_epoch` y a status='running'. Si la primera falla, el paso NO
    se despacha (no se gasta en una corrida superada). Si falla la de
    `completed`, el resultado tardío se descarta y su ref no entra al contexto.
    Lo despachado no se interrumpe: es el mismo límite que el kill switch
    entre olas. run_pipeline descubre la pérdida en su escritura de fin de
    ola y registra RUN_SUPERSEDED una sola vez.
    """
    epoca = pipeline.run_epoch
    step.status     = StepStatus.running
    step.started_at = time.time()
    if not await store.step_upsert_si_epoca(step, epoca):
        return PASO_SIN_ESCRITURA
    await store.event_append(
        pipeline.pipeline_id, "STEP_STARTED",
        {"step_index": i, "facet": step.facet, "capability": step.capability},
        step.step_id,
    )

    try:
        raw_output = await asyncio.wait_for(
            # El freno en vuelo (2026-09-16, frente B): antes un step ya lanzado
            # seguía hasta terminar la ola aunque el kill switch estuviera
            # puesto. Si el freno APARECE, se cancela en <= 250 ms:
            # run_sandboxed_claude mata a Hyde y _invoke_motor cancela el job
            # en LAS MANOS. InterruptorActivado cae en el except general de
            # abajo, así que queda _fail_step con "killed_by_switch". Un
            # timeout de este wait_for (o una cancelación externa de este
            # mismo step) también esperan esa misma limpieza interna antes de
            # propagar -- el finally de correr_con_interruptor la awaitea,
            # fix del 2026-09-17 (antes solo pedía tarea.cancel() sin
            # esperarla, y _fail_step podía correr con la limpieza a medias).
            correr_con_interruptor(_dispatch_step(step, pipeline)),
            timeout=step.timeout_seconds,
        )

        # F1 (ola final): la época va en la ruta -- este archivo se escribe
        # antes de la escritura condicional y una corrida superada no puede
        # pisar el de la vigente. En un hilo: es escritura a disco.
        ref, inline = await asyncio.to_thread(
            save_if_large, pipeline.pipeline_id, step.step_id, raw_output, epoca=epoca,
        )
        if ref:
            step.output_ref = ref
        else:
            step.output_ref = f"inline:{json.dumps(inline, ensure_ascii=False)}"

        step.status      = StepStatus.completed
        step.finished_at = time.time()
        if not await store.step_upsert_si_epoca(step, epoca):
            return PASO_SIN_ESCRITURA
        pipeline.context[f"step_{i}_ref"] = step.output_ref
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
        except Exception as _persist_err:  # noqa: BLE001  # fail-soft: es la copia .md de cortesía en REPO_DOCUMENTS_DIR que el admin de jax-platform lista (/api/admin/repo) -- el output canónico ya quedó en output_ref y en store.step_upsert_si_epoca antes de este try; si la copia falla queda el warning y el step sigue completado
            logger.warning("No se pudo persistir step %d al repo: %s", i, _persist_err)
        return True

    except asyncio.TimeoutError:
        await _fail_step(pipeline, step, i, f"Timeout ({step.timeout_seconds}s)")
        return False
    except Exception as exc:  # noqa: BLE001  # fail-soft: no traga nada -- convierte cualquier error del step en fallo EXPLÍCITO vía _fail_step (status=failed + STEP_FAILED + error) y devuelve False, que es lo que la ola usa para cortar el pipeline
        await _fail_step(pipeline, step, i, str(exc))
        return False


# ----------------------------------------------------------------
#  Pipeline runner — DIRECTOR DE ORQUESTA (ejecución por olas)
# ----------------------------------------------------------------

# Estados desde los que arranca una corrida: pending (recién creado), running
# (continue ya lo dejó así con la época nueva), interrupted (resume y
# approve-step no cambian el status, sólo la época).
_DESDE_ARRANQUE = (PipelineStatus.pending, PipelineStatus.running, PipelineStatus.interrupted)


async def _perdio_la_epoca(pipeline: Pipeline) -> None:
    """La corrida perdió su época o el pipeline dejó de estar `running`
    (cancelado, vencido por el reaper, continuado o reanudado por otro
    pedido). Se registra UNA vez y quien llama termina sin escribir nada más."""
    actual = await store.pipeline_epoca_y_status(pipeline.pipeline_id)
    epoca_actual = actual[0] if actual else None
    status_actual = actual[1].value if actual else None
    logger.warning(
        "Jacobs %s: corrida de la época %s superada (época actual %s, status %s) -- termina sin escribir",
        pipeline.pipeline_id, pipeline.run_epoch, epoca_actual, status_actual,
    )
    await store.event_append(pipeline.pipeline_id, "RUN_SUPERSEDED", {
        "epoca": pipeline.run_epoch,
        "epoca_actual": epoca_actual,
        "status_actual": status_actual,
    })


def _disparar_aviso_fin(pipeline: Pipeline, estado: PipelineStatus) -> None:
    """Task 6 (2026-09-18): avisa por Telegram que el pipeline terminó.

    Se llama SOLO desde los puntos donde `pipeline_update_status_si_epoca`
    ya devolvió True para un status terminal (completed/aborted) -- esa
    escritura es el mismo reclamo atómico condicional (WHERE pipeline_id=?
    AND run_epoch=? AND status IN (...), store.py:1436) que el resto del
    ejecutor usa para "una sola vez gana"; una corrida que perdió la época
    nunca llega hasta acá (se va por `_perdio_la_epoca`). No hace falta una
    tabla de deduplicación aparte -- ver jacobs/aviso.py.

    `aviso.avisar_fin_pipeline` ya es fire-and-forget (no espera el POST a
    Telegram) y fail-soft por dentro (nunca lanza). Este try/except es la
    última barrera, solo contra un fallo agendando la tarea en sí -- para
    que un pipeline YA completado/abortado (el status ya está escrito) jamás
    vea ese status revertido ni la excepción propagarse hacia arriba."""
    try:
        aviso.avisar_fin_pipeline(
            pipeline_id=pipeline.pipeline_id, nombre=pipeline.name, estado=estado.value,
        )
    except Exception:
        logger.error(
            "Pipeline %s: no se pudo agendar el aviso de Telegram de fin (status %s ya escrito)",
            pipeline.pipeline_id, estado.value, exc_info=True,
        )


async def run_pipeline(pipeline: Pipeline) -> None:
    """Corre el pipeline (ver _correr_pipeline). Ruling R38, fix round 1: la
    corrida es un trabajo de fondo -- sus escrituras esperan turno del pool
    sin plazo (store.espera_de_turno_sin_plazo) en vez de morir por cola con
    la base sana; una base caída sigue fallando al conectar."""
    with espera_de_turno_sin_plazo():
        await _correr_pipeline(pipeline)


async def _correr_pipeline(pipeline: Pipeline) -> None:
    """
    Ejecuta el pipeline por OLAS topológicas. Dentro de cada ola, los steps
    corren EN PARALELO (asyncio.gather). El orden entre olas respeta depends_on.

    Modos:
      dry_run    — no ejecuta nada, completa inmediatamente.
      supervised — ejecuta UNA ola y pausa (status=interrupted); espera /resume.
                   La granularidad de aprobación es la OLA, no el step.
    Hyde: si un step de la ola es hyde sin aprobar, la ola NO se ejecuta y el
    pipeline se interrumpe hasta /approve-step.

    Época (spec 2026-09-17 §5.3): `pipeline.run_epoch` es la época de ESTA
    corrida. Toda escritura de estado es condicional a ella y a
    status='running'; antes de cada ola se relee (una consulta por PK). Si no
    coincide, RUN_SUPERSEDED una vez y termina.
    """
    pipeline_id = pipeline.pipeline_id
    epoca = pipeline.run_epoch

    if pipeline.mode == "dry_run":
        if not await store.pipeline_update_status_si_epoca(
            pipeline_id, epoca, PipelineStatus.completed, desde=_DESDE_ARRANQUE,
        ):
            await _perdio_la_epoca(pipeline)
            return
        await store.event_append(pipeline_id, "DRY_RUN_COMPLETE", {"steps": len(pipeline.plan)})
        _disparar_aviso_fin(pipeline, PipelineStatus.completed)
        return

    if not await store.pipeline_update_status_si_epoca(
        pipeline_id, epoca, PipelineStatus.running,
        pipeline.current_step_index, pipeline.context, desde=_DESDE_ARRANQUE,
    ):
        await _perdio_la_epoca(pipeline)
        return
    await store.event_append(pipeline_id, "PIPELINE_STARTED", {"run_epoch": epoca})

    # Estado derivado del DAG, no de un cursor lineal: un step está "hecho" si
    # tiene su ref en context (sobrevive a /resume y a /continue).
    done = {
        i for i in range(len(pipeline.plan))
        if pipeline.context.get(f"step_{i}_ref")
    }

    waves = _compute_waves(pipeline.plan, done)
    logger.info(
        "Jacobs director: %d olas, tamaños=%s (ya completos: %s, época %s)",
        len(waves), [len(w) for w in waves], sorted(done), epoca,
    )

    for wave_num, wave in enumerate(waves):
        # ---- ¿Sigue siendo mi corrida? ----
        if await store.pipeline_epoca_y_status(pipeline_id) != (epoca, PipelineStatus.running):
            await _perdio_la_epoca(pipeline)
            return

        # ---- Kill switch: antes de cada ola ----
        if check_kill_switch():
            for i in wave:
                step = pipeline.plan[i]
                step.status = StepStatus.failed
                step.error  = "Kill switch activo"
                if not await store.step_upsert_si_epoca(step, epoca):
                    await _perdio_la_epoca(pipeline)
                    return
            if not await store.pipeline_update_status_si_epoca(pipeline_id, epoca, PipelineStatus.aborted):
                await _perdio_la_epoca(pipeline)
                return
            await store.event_append(
                pipeline_id, "KILL_SWITCH_ABORTED", {"wave": wave_num, "steps": wave}
            )
            _disparar_aviso_fin(pipeline, PipelineStatus.aborted)
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
                if not await store.step_upsert_si_epoca(step, epoca):
                    await _perdio_la_epoca(pipeline)
                    return
                await store.event_append(
                    pipeline_id, "STEP_BLOCKED_HUMAN_GATE",
                    {"step_index": i, "facet": "hyde"}, step.step_id,
                )
            if not await store.pipeline_update_status_si_epoca(
                pipeline_id, epoca, PipelineStatus.interrupted, wave[0], pipeline.context,
            ):
                await _perdio_la_epoca(pipeline)
                return
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

        # Persistir avance del context tras la ola completa, SOLO si sigue
        # siendo mi corrida: esta es la escritura que antes resucitaba un
        # pipeline cancelado a `running`.
        next_idx = max(wave) + 1
        if not await store.pipeline_update_status_si_epoca(
            pipeline_id, epoca, PipelineStatus.running, next_idx, pipeline.context,
        ):
            await _perdio_la_epoca(pipeline)
            return

        # ---- Pasos que no se pudieron escribir (m2) ----
        # `PASO_SIN_ESCRITURA` no distingue por sí solo entre "perdí la época"
        # y "la fila del paso cambió": se relee la época para saber cuál fue.
        # Si la corrida sigue siendo la vigente, es lo segundo y el paso lleva
        # su motivo al evento en vez de un error nulo. skip_on_fail NO aplica
        # acá: no se puede saltar un paso cuya fila no se pudo escribir.
        sin_escritura = [i for i, r in zip(wave, results) if r is PASO_SIN_ESCRITURA]
        if sin_escritura:
            if await store.pipeline_epoca_y_status(pipeline_id) != (epoca, PipelineStatus.running):
                await _perdio_la_epoca(pipeline)
                return
            for i in sin_escritura:
                if pipeline.plan[i].error is None:
                    pipeline.plan[i].error = _MOTIVO_SIN_ESCRITURA

        # ---- ¿Algún step falló sin skip_on_fail? → abortar (UN evento) ----
        failed = [
            i for i, ok in zip(wave, results)
            if ok is not True and (i in sin_escritura or not pipeline.plan[i].skip_on_fail)
        ]
        if failed:
            if not await store.pipeline_update_status_si_epoca(pipeline_id, epoca, PipelineStatus.aborted):
                await _perdio_la_epoca(pipeline)
                return
            await store.event_append(
                pipeline_id, "PIPELINE_ABORTED",
                {"at_wave": wave_num, "failed_steps": failed,
                 "errores": {str(i): pipeline.plan[i].error for i in failed}},
            )
            _disparar_aviso_fin(pipeline, PipelineStatus.aborted)
            return

        await store.event_append(
            pipeline_id, "WAVE_COMPLETED", {"wave": wave_num, "steps": wave}
        )

        # ---- Supervised: pausar después de cada ola ----
        if pipeline.mode == "supervised":
            if not await store.pipeline_update_status_si_epoca(
                pipeline_id, epoca, PipelineStatus.interrupted, next_idx, pipeline.context,
            ):
                await _perdio_la_epoca(pipeline)
                return
            await store.event_append(
                pipeline_id, "PIPELINE_INTERRUPTED",
                {"after_wave": wave_num, "next_index": next_idx,
                 "reason": "supervised — awaiting /resume"},
            )
            return

    # ---- Todas las olas terminaron ----
    if not await store.pipeline_update_status_si_epoca(
        pipeline_id, epoca, PipelineStatus.completed, len(pipeline.plan), pipeline.context,
    ):
        await _perdio_la_epoca(pipeline)
        return
    await store.event_append(pipeline_id, "PIPELINE_COMPLETED")
    _disparar_aviso_fin(pipeline, PipelineStatus.completed)


async def _persist_step_to_repo(
    pipeline_id: str,
    pipeline_name: str,
    step_index: int,
    facet: str,
    capability: str,
    raw_output: dict,
) -> None:
    """Guarda el output de un step como .md en REPO_DOCUMENTS_DIR: copia de
    cortesía que el admin de jax-platform lista en /api/admin/repo. La escritura
    (mkdir incluido) corre en un hilo: nada de disco dentro del event loop (E-12)."""
    filename = f"{pipeline_id[:8]}_{step_index:02d}_{facet}.md"
    directorio = REPO_DOCUMENTS_DIR
    filepath = directorio / filename

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
        lines += ["", "## Fuentes", "", render_sources_block(sources)]

    content = "\n".join(lines)

    def _escribir() -> None:
        directorio.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")

    await asyncio.to_thread(_escribir)

    logger.info("Step output persistido: %s", filename)


async def _fail_step(
    pipeline: Pipeline, step: Step, step_index: int, error: str
) -> None:
    # Ruling T6-6: aca se ESCRIBE el error de un paso (jacobs_steps.error y el
    # evento STEP_FAILED, que jax-platform muestra). Se redacta en el punto de
    # escritura para que ningun llamador -- ni el str(exc) generico de
    # _run_one_step -- pueda guardar un secreto en claro.
    #
    # 2026-09-17 (spec §5.3): ya NO escribe `aborted` ni PIPELINE_ABORTED. Lo
    # hace run_pipeline al cerrar la ola, una sola vez y con los errores de
    # todos los pasos caídos: antes salían DOS PIPELINE_ABORTED con payloads
    # distintos. Y la escritura del paso es condicional a la época.
    error = redactar_secretos(error)
    step.status      = StepStatus.failed
    step.error       = error
    step.finished_at = time.time()
    if not await store.step_upsert_si_epoca(step, pipeline.run_epoch):
        return
    await store.event_append(
        pipeline.pipeline_id, "STEP_FAILED",
        {"step_index": step_index, "error": error},
        step.step_id,
    )
