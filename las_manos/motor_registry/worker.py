"""
LAS MANOS — Motor Registry: worker con Kimi real.

Ejecuta el job completo:
  1. Marca RUNNING
  2. Verifica kill switch (/etc/jax/PAUSE) antes de llamar
  3. Llama a la API del motor (Kimi/moonshot) con httpx async
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
from credential_resolver import resolve_credential_instrumented, CredentialUnavailableError

# motor (catalogo local) -> provider_id (tabla `credential`). Hoy solo kimi
# es un motor real via este camino (_MOTOR_FACETS en jacobs/executor.py).
_MOTOR_PROVIDER_MAP = {"kimi": "moonshot"}
from motor_registry.job_store import JobStore
from motor_registry.models import JobStatus
from motor_registry.output_validator import validate

logger = logging.getLogger(__name__)

_KILL_SWITCH_INTERVAL = 5.0  # segundos entre chequeos durante la ejecución


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

async def _watch_kill_switch(path: str) -> None:
    """Retorna en cuanto detecta el archivo PAUSE. Chequea cada 5s."""
    while True:
        if Path(path).exists():
            return
        await asyncio.sleep(_KILL_SWITCH_INTERVAL)


async def _call_kimi(
    *,
    api_url: str,
    model: str,
    api_key: str,
    prompt: str,
    timeout: float,
    max_tokens: int = 0,
) -> dict:
    """Llama a la API de Kimi. Devuelve el dict JSON completo de la respuesta.

    max_tokens (2026-08-10): sin esto, un motor de razonamiento puede gastar
    todo el completion budget en reasoning_content y devolver `content`
    cortado a mitad de palabra — bug real reproducido en vivo contra la API
    de Moonshot. 0/falsy = no mandar el campo (motor sin este limite
    configurado)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


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

    # Validar API key
    provider_id = _MOTOR_PROVIDER_MAP.get(motor, motor)
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

    # Lanzar tarea HTTP y watcher de kill switch en paralelo
    api_task = asyncio.create_task(
        _call_kimi(
            api_url=motor_entry.api_url,
            model=motor_entry.model,
            api_key=api_key,
            prompt=prompt,
            timeout=float(motor_entry.default_timeout_seconds),
            max_tokens=motor_entry.max_tokens,
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
    reasoning_content: str = message.get("reasoning_content") or ""
    finish_reason = choices[0].get("finish_reason")
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
