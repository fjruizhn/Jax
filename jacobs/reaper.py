"""
Jacobs — Reaper de pipelines huérfanos.

Cosecha pipelines en status no-terminal (pending/running/interrupted)
que quedaron sin avance real -- el proceso murió a mitad de camino
(reinicio, timeout de cliente) o son la clase de huérfano descubierta
en T1.a de la sesión 2026-08-19: interrupted sin owner ack, donde el
cliente nunca confirmó recepción del pipeline_id y /resume queda
inalcanzable (_require_pipeline_owner en jax-platform devuelve 404 sin
owner_ack_at poblado).

Umbrales calibrados post-T1 de la misma sesión: think:false aplicado a
_llm_plan(), build() medido en 1.3-8.7s en 6 corridas reales (3
objetivos x think true/false) + 8.1s de verificación end-to-end vía el
endpoint completo con el objetivo real de Fernando. Antes de ese fix
build() tardaba 17-37s -- los umbrales de abajo asumen el piso nuevo.

- PENDING_MAX_AGE_SECONDS = 300 (5min): ~35x margen sobre el máximo de
  creación end-to-end observado (8.7s). Un pipeline pending más viejo
  que esto nunca llegó a correr background.add_task -- el proceso
  murió entre el INSERT y el arranque de esa tarea.
- RUNNING_STALE_SECONDS = 1800 (30min), medido contra updated_at (no
  created_at) -- un running sano actualiza updated_at en cada
  transición de step. 2x margen sobre el timeout de step más largo
  configurado hoy (jacobs/plan.py: _CAPABILITY_TIMEOUT_SECONDS,
  "reconcile"=900s).
- INTERRUPTED_NO_OWNER_MAX_AGE_SECONDS = 600 (10min): margen generoso
  sobre una escritura de owner file que en el camino sano es casi
  instantánea (<1ms) -- guarda contra el caso raro de disco lento, no
  contra operación normal.

RESUELTO (ronda 5, 2026-08-20, T1): el chequeo de owner file cruzaba de
repo -- leía ~/jax/pipelines/{id}_owner.json, que escribía jax-platform/
backend/api/pipelines.py, no Jacobs. Si jax-platform cambiaba
PIPELINES_DIR, o algún día corrían en hosts distintos, ese chequeo veía
"sin owner" en todo y cosechaba pipelines legítimos. Migrado a la
columna owner_ack_at en jacobs_pipelines (misma DB física jax_memory que
ya comparten ambos servicios) -- jax-platform ahora hace un UPDATE
directo a esa columna con el mismo pool de DB que ya usa para
capability/motor, sin pasar por Jacobs ni por filesystem. Verificado al
migrar: 0 owner files vivos en disco (directorio vacío) -- no hubo datos
que trasladar, solo código. El directorio ~/jax/pipelines/ y los owner
files viejos (si aparecen) quedan huérfanos en disco, sin lector desde
este cambio -- no se borran automáticamente (decisión: limpieza manual
si molestan, no vale la pena un cron para un directorio vacío).

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

import httpx

from jacobs import store
from jacobs.models import PipelineStatus

logger = logging.getLogger("jacobs.reaper")

PENDING_MAX_AGE_SECONDS = 300
RUNNING_STALE_SECONDS = 1800
INTERRUPTED_NO_OWNER_MAX_AGE_SECONDS = 600
SWEEP_INTERVAL_SECONDS = 300  # cada 5 min

# Si un solo barrido cosecha más que esto, no es ruido normal -- es el
# mismo patrón que produjo los 8 zombies de Bug A (T4 ronda 2026-08-19):
# el límite duro de pipelines concurrentes agotado por algo que sigue
# insertando filas fuera del camino normal. Mismo número que
# jacobs.policy.MAX_PARALLEL_PIPELINES -- si se cosechan más que el
# cupo total configurado en una sola corrida, hay una fuga activa.
TELEGRAM_ALERT_THRESHOLD = 3


async def send_telegram_alert(message: str) -> dict:
    """Manda la alerta y devuelve {ok, message_id, error} -- SIEMPRE, nunca
    None. El caller (reap_orphaned_pipelines) decide qué hacer con el
    resultado: la cosecha ya ocurrió antes de llegar acá, así que esta
    función jamás lanza (fail-soft real: un token rotado o un chat_id
    roto no debe tumbar el barrido) pero tampoco descarta el resultado en
    silencio -- T1 (2026-08-19): antes de este fix, `ok`/`message_id` de
    la respuesta de sendMessage se descartaban por completo; la alerta
    funcionaba "por suerte", sin verificación real de entrega.

    Promovida de _send_telegram_alert (privada) a send_telegram_alert
    (T5, GAP2 Fase4, 2026-08-19): worker.py la reusa para notificar al
    cierre de jobs que escribieron -- mismo patrón de captura, no
    duplicado. Único otro caller: reap_orphaned_pipelines, abajo."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("Reaper: alerta suprimida, TELEGRAM_BOT_TOKEN/CHAT_ID no configurados")
        return {"ok": False, "message_id": None, "error": "TELEGRAM_BOT_TOKEN/CHAT_ID no configurados"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat_id, "text": message},
            )
        body = resp.json()
    except Exception as exc:  # fail-soft: red caída/timeout -- no debe tumbar el reaper
        logger.error("Reaper: fallo de red enviando alerta a Telegram", exc_info=True)
        return {"ok": False, "message_id": None, "error": f"{type(exc).__name__}: {exc}"}

    if resp.status_code == 200 and body.get("ok"):
        message_id = body.get("result", {}).get("message_id")
        logger.info("Reaper: alerta entregada a Telegram, message_id=%s", message_id)
        return {"ok": True, "message_id": message_id, "error": None}

    # ok=false o HTTP no-200: Telegram respondió, pero rechazó el envío
    # (token rotado, chat_id inválido, bot bloqueado, etc). Esto es
    # exactamente el caso que el código viejo enmascaraba.
    error = f"HTTP {resp.status_code}: {body.get('description', body)}"
    logger.error("Reaper: Telegram rechazó la alerta -- %s", error)
    return {"ok": False, "message_id": None, "error": error}


async def reap_orphaned_pipelines() -> list[dict]:
    """Un barrido completo. Encuentra y cosecha pipelines huérfanos,
    transicionándolos a PipelineStatus.expired. Devuelve la lista de lo
    cosechado (para logging/alerta/tests). No lanza -- errores por
    pipeline individual no deben abortar el resto del barrido."""
    now = time.time()
    reaped: list[dict] = []

    candidates = await store.pipelines_by_status(
        [PipelineStatus.pending, PipelineStatus.running, PipelineStatus.interrupted]
    )
    for p in candidates:
        age = now - p.created_at
        stale_since = now - p.updated_at
        reason = None

        if p.status == PipelineStatus.pending and age > PENDING_MAX_AGE_SECONDS:
            reason = f"pending sin avance {age:.0f}s (umbral {PENDING_MAX_AGE_SECONDS}s)"
        elif p.status == PipelineStatus.running and stale_since > RUNNING_STALE_SECONDS:
            reason = f"running sin avance {stale_since:.0f}s (umbral {RUNNING_STALE_SECONDS}s)"
        elif (
            p.status == PipelineStatus.interrupted
            and age > INTERRUPTED_NO_OWNER_MAX_AGE_SECONDS
            and p.owner_ack_at is None
        ):
            reason = (
                f"interrupted sin owner ack, {age:.0f}s "
                f"(umbral {INTERRUPTED_NO_OWNER_MAX_AGE_SECONDS}s)"
            )

        if not reason:
            continue

        try:
            await store.pipeline_update_status(p.pipeline_id, PipelineStatus.expired)
            await store.event_append(p.pipeline_id, "REAPED", {
                "prev_status": p.status.value, "reason": reason,
            })
        except Exception:  # fail-soft: un fallo cosechando ESTE pipeline no debe abortar el resto del barrido; el proximo ciclo (SWEEP_INTERVAL_SECONDS) reintenta
            logger.warning("Reaper: fallo cosechando %s", p.pipeline_id, exc_info=True)
            continue

        entry = {"pipeline_id": p.pipeline_id, "name": p.name, "prev_status": p.status.value, "reason": reason}
        reaped.append(entry)
        logger.warning("Reaper cosechó %s (%s) [%s]: %s", p.pipeline_id, p.name, p.status.value, reason)

    if reaped:
        if len(reaped) > TELEGRAM_ALERT_THRESHOLD:
            result = await send_telegram_alert(
                f"⚠️ Reaper de Jacobs cosechó {len(reaped)} pipelines huérfanos en un "
                f"solo barrido (umbral {TELEGRAM_ALERT_THRESHOLD}) -- posible fuga activa, "
                f"no ruido normal. Ver logs de jax-las-manos."
            )
            # fail-soft: la cosecha YA ocurrió (transiciones + REAPED ya
            # persistidos arriba) -- no hay nada que abortar. "Visible"
            # aca significa: log a ERROR (ya lo hace send_telegram_alert)
            # + un evento propio en jacobs_events por cada pipeline
            # cosechado en este barrido, para que quede auditable sin
            # depender de journalctl ni de que alguien mire Telegram y
            # note el silencio. Sin esto, un token rotado deja la fuga
            # activa sin ninguna traza en la DB de que la alerta falló.
            if not result["ok"]:
                for entry in reaped:
                    try:
                        await store.event_append(entry["pipeline_id"], "REAPER_ALERT_FAILED", {
                            "batch_size": len(reaped),
                            "threshold": TELEGRAM_ALERT_THRESHOLD,
                            "error": result["error"],
                        })
                    except Exception:  # fail-soft: si ni la DB responde, ya quedó el log a ERROR de arriba
                        logger.error("Reaper: no se pudo registrar REAPER_ALERT_FAILED para %s", entry["pipeline_id"], exc_info=True)
        else:
            logger.info("Reaper: %d pipeline(s) cosechado(s), dentro de lo esperado", len(reaped))

    return reaped


async def start_reaper_loop() -> None:
    """Barrido periódico en background. Se llama desde el startup del
    server además de esto -- ver server.py::_jacobs_init."""
    while True:
        try:
            await reap_orphaned_pipelines()
        except Exception:  # fail-soft: loop de limpieza en background, mismo patron que jax-platform/jax_engine/owner_cleanup.py -- nunca debe tumbar el proceso, el proximo ciclo reintenta
            logger.warning("Reaper: barrido periódico falló", exc_info=True)
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
