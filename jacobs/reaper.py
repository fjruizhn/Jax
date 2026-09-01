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
  configurado hoy (`capability.max_execution_minutes` en la DB, fuente
  unica desde 2026-09-01; "reconcile"=15min=900s).
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
import json
import logging
import os
import time
from pathlib import Path

import httpx

from jacobs import store
from jacobs.facet_health import check_facet_health
from jacobs.models import HTTP_FACETS, PipelineStatus

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


# ----------------------------------------------------------------
#  T3 (2026-08-22, auditoria usage_writer) -- reconciliación
#  motor_jobs.jsonl vs axioma_usage. La única forma de saber que la
#  contabilidad estaba rota (0/9 dispatches reconciliados) fue que
#  alguien pidiera auditarla a mano -- esto la convierte en un chequeo
#  periódico, mismo mecanismo de alerta que reap_orphaned_pipelines.
# ----------------------------------------------------------------

MOTOR_JOBS_LOG_PATH = os.getenv(
    "MOTOR_JOBS_LOG_PATH",
    str(Path(__file__).resolve().parent.parent / "las_manos" / "logs" / "motor_jobs.jsonl"),
)
# Ventana de auditoría: jobs viejos (pre-fix de T1.b/c, sin job_id en la fila
# de axioma_usage aunque exista) nunca van a reconciliar -- arrastrarlos para
# siempre haría que el chequeo diera 100% de gap para siempre, sin señal.
# 24h cubre tráfico "bajo el régimen del fix" sin ese ruido histórico.
RECONCILIATION_WINDOW_SECONDS = 24 * 3600
# Umbral: el hallazgo real fue 100% (0/9). 20% deja margen para un fallo
# aislado de DB (2 reintentos agotados, T1.d) sin generar ruido por cada
# escritura perdida individual -- pero cualquier valor sostenido por encima
# de esto es la misma clase de problema que el hallazgo original, no ruido.
RECONCILIATION_ALERT_THRESHOLD_PCT = 20.0
# No cada barrido de 5min -- leer+parsear el JSONL completo cada vez es
# más caro que el barrido de pipelines huérfanos (una query indexada). Una
# vez por hora alcanza para detectar un problema sostenido sin sumar carga.
RECONCILIATION_CHECK_EVERY_N_SWEEPS = 12  # 12 * 300s = 3600s


def _load_terminal_motor_jobs(path: str, since: float) -> list[dict]:
    """Última línea por job_id (estado final) para jobs completed/failed/
    cancelled con created_at >= since. has_usage=True solo si el job llegó
    a acumular _usage (T1.a: los fallos pre-loop -- kill switch, motor/
    transport/credencial ausente -- genuinamente gastaron 0 tokens, no
    cuentan como 'esperados')."""
    jobs: dict[str, dict] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:  # fail-soft: línea JSONL corrupta -- mismo criterio que JobStore._load, se descarta, no aborta el chequeo
                    continue
                jobs[e["job_id"]] = e
    except FileNotFoundError:
        return []
    out = []
    for job_id, e in jobs.items():
        if e.get("status") not in ("completed", "failed", "cancelled"):
            continue
        if (e.get("created_at") or 0) < since:
            continue
        usage = e.get("_usage") or {}
        has_usage = bool(usage.get("prompt_tokens") or usage.get("completion_tokens"))
        out.append({"job_id": job_id, "status": e["status"], "has_usage": has_usage})
    return out


def _compute_reconciliation_gap(motor_jobs: list[dict], usage_job_ids: set) -> dict:
    """Pura, sin I/O -- job_ids que debían tener fila (has_usage=True) vs
    los que realmente la tienen en axioma_usage."""
    expected = [j["job_id"] for j in motor_jobs if j.get("has_usage")]
    missing = [jid for jid in expected if jid not in usage_job_ids]
    gap_pct = (len(missing) / len(expected) * 100.0) if expected else 0.0
    return {
        "expected": len(expected),
        "reconciled": len(expected) - len(missing),
        "missing": missing,
        "gap_pct": gap_pct,
    }


async def _fetch_reconciled_job_ids(job_ids: list[str]) -> set:
    if not job_ids:
        return set()
    conn = await store.get_conn()
    try:
        placeholders = ",".join(["%s"] * len(job_ids))
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT job_id FROM axioma_usage WHERE request_type='motor' "
                f"AND job_id IN ({placeholders})",
                tuple(job_ids),
            )
            return {row[0] for row in await cur.fetchall()}
    finally:
        conn.close()


async def _fetch_http_direct_expected(since: float) -> dict[str, int]:
    """Steps HTTP-directos (hipatia/jekyll/thot/ada) completados en la
    ventana, por facet. Fuente: jacobs_events -- join STEP_STARTED (trae
    el facet en el payload) con STEP_COMPLETED por step_id, dentro de la
    misma tabla. No hay job_id para este camino (record_direct_usage no
    lo escribe, ver jacobs/usage_writer.py), así que a diferencia de
    Motor Registry esto NO puede confirmar qué dispatch puntual falta --
    solo cuántos se esperaban por facet. `ts` es epoch (DOUBLE), inmune
    a timezone -- comparar directo contra `since` (también epoch)."""
    conn = await store.get_conn()
    try:
        placeholders = ",".join(["%s"] * len(HTTP_FACETS))
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT JSON_UNQUOTE(JSON_EXTRACT(s.payload, '$.facet')) AS facet, "
                "COUNT(DISTINCT s.step_id) AS n "
                "FROM jacobs_events s "
                "JOIN jacobs_events c ON c.step_id = s.step_id AND c.event_type = 'STEP_COMPLETED' "
                f"WHERE s.event_type = 'STEP_STARTED' "
                f"AND JSON_UNQUOTE(JSON_EXTRACT(s.payload, '$.facet')) IN ({placeholders}) "
                "AND s.ts >= %s "
                "GROUP BY facet",
                (*HTTP_FACETS, since),
            )
            return {row[0]: row[1] for row in await cur.fetchall()}
    finally:
        conn.close()


async def _fetch_http_direct_actual(since: float) -> dict[str, int]:
    """Filas reales en axioma_usage para esos mismos facets, misma
    ventana. request_type='pipeline' es el marcador que usa
    record_direct_usage -- no confundir con 'chat' (Mesa web) ni 'motor'
    (Motor Registry). `created_at` es TIMESTAMP (sensible a timezone de
    sesión) -- comparar via UNIX_TIMESTAMP(created_at), nunca con un
    string de fecha literal: el primer intento de esta misma auditoría
    (limpieza de axioma_usage, 2026-08-21) perdió 90/106 filas de un
    WHERE por string de fecha exactamente por este motivo, detectado
    solo porque se verificó el conteo real del resultado."""
    conn = await store.get_conn()
    try:
        placeholders = ",".join(["%s"] * len(HTTP_FACETS))
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT facet, COUNT(*) AS n FROM axioma_usage "
                f"WHERE request_type='pipeline' AND facet IN ({placeholders}) "
                "AND UNIX_TIMESTAMP(created_at) >= %s "
                "GROUP BY facet",
                (*HTTP_FACETS, since),
            )
            return {row[0]: row[1] for row in await cur.fetchall()}
    finally:
        conn.close()


def _compute_http_direct_gap(expected_by_facet: dict[str, int], actual_by_facet: dict[str, int]) -> dict:
    """Pura, sin I/O. Aproximada por diseño (aclarado arriba en
    _fetch_http_direct_expected): compara CONTEOS por facet en la
    ventana, no dispatches individuales por job_id/step_id como sí hace
    _compute_reconciliation_gap para Motor Registry. `reconciled` capea
    el match a min(esperado, real) POR FACET antes de sumar -- si no,
    un facet con filas de más (otra ventana, otro escritor) taparía un
    gap real en otro facet distinto al sumar todo junto primero."""
    facets = sorted(set(expected_by_facet) | set(actual_by_facet))
    total_expected = sum(expected_by_facet.values())
    reconciled = sum(min(expected_by_facet.get(f, 0), actual_by_facet.get(f, 0)) for f in facets)
    missing_by_facet = {
        f: expected_by_facet.get(f, 0) - actual_by_facet.get(f, 0)
        for f in facets
        if expected_by_facet.get(f, 0) > actual_by_facet.get(f, 0)
    }
    gap_pct = ((total_expected - reconciled) / total_expected * 100.0) if total_expected else 0.0
    return {
        "expected": total_expected,
        "reconciled": reconciled,
        "missing_by_facet": missing_by_facet,
        "gap_pct": gap_pct,
    }


async def check_usage_reconciliation() -> dict:
    """Un chequeo completo. No lanza -- mismo criterio que
    reap_orphaned_pipelines: un fallo acá no debe tumbar el reaper.

    Cubre DOS caminos de dispatch (T5, 2026-08-21 -- el chequeo original
    solo cubría Motor Registry; hipatia/jekyll/thot/ada no tenían ninguna
    alerta si record_direct_usage fallaba en silencio, el mismo tipo de
    punto ciego que motivó este chequeo en primer lugar para Motor
    Registry). Motor Registry reconcilia por job_id exacto (preciso);
    HTTP-directo reconcilia por conteo agregado por facet (aproximado,
    ver _fetch_http_direct_expected) -- cobertura real es mejor que cero
    cobertura, pero no reemplaza tener job_id/step_id en axioma_usage
    para ese camino."""
    since = time.time() - RECONCILIATION_WINDOW_SECONDS

    try:
        motor_jobs = _load_terminal_motor_jobs(MOTOR_JOBS_LOG_PATH, since)
        expected_ids = [j["job_id"] for j in motor_jobs if j["has_usage"]]
        reconciled_ids = await _fetch_reconciled_job_ids(expected_ids)
        motor_result = _compute_reconciliation_gap(motor_jobs, reconciled_ids)
    except Exception:  # fail-soft: el chequeo de reconciliación no debe tumbar el reaper -- el próximo ciclo reintenta
        logger.warning("Reaper: chequeo de reconciliación de usage (Motor Registry) falló", exc_info=True)
        motor_result = {"expected": 0, "reconciled": 0, "missing": [], "gap_pct": 0.0, "error": True}

    try:
        http_expected = await _fetch_http_direct_expected(since)
        http_actual = await _fetch_http_direct_actual(since)
        http_result = _compute_http_direct_gap(http_expected, http_actual)
    except Exception:
        logger.warning("Reaper: chequeo de reconciliación de usage (HTTP-directo) falló", exc_info=True)
        http_result = {"expected": 0, "reconciled": 0, "missing_by_facet": {}, "gap_pct": 0.0, "error": True}

    if motor_result["expected"] > 0 and motor_result["gap_pct"] > RECONCILIATION_ALERT_THRESHOLD_PCT:
        logger.error(
            "Reaper: gap de reconciliación de usage (Motor Registry) %.1f%% (%d/%d dispatches "
            "sin fila en axioma_usage, ventana %dh) -- umbral %.0f%%",
            motor_result["gap_pct"], len(motor_result["missing"]), motor_result["expected"],
            RECONCILIATION_WINDOW_SECONDS // 3600, RECONCILIATION_ALERT_THRESHOLD_PCT,
        )
        await send_telegram_alert(
            f"⚠️ Jacobs: {motor_result['gap_pct']:.0f}% de los dispatches de Motor Registry "
            f"({len(motor_result['missing'])}/{motor_result['expected']}) en las últimas "
            f"{RECONCILIATION_WINDOW_SECONDS // 3600}h no tienen fila de costo en "
            f"axioma_usage -- gastaron tokens reales sin contabilizar. Ver "
            f"jacobs.reaper.check_usage_reconciliation / motor_jobs.jsonl."
        )
    else:
        logger.info(
            "Reaper: reconciliación de usage (Motor Registry) OK -- %d/%d dispatches con fila (%.1f%% gap)",
            motor_result["reconciled"], motor_result["expected"], motor_result["gap_pct"],
        )

    if http_result["expected"] > 0 and http_result["gap_pct"] > RECONCILIATION_ALERT_THRESHOLD_PCT:
        logger.error(
            "Reaper: gap de reconciliación de usage (HTTP-directo) %.1f%% (%d/%d dispatches "
            "sin fila en axioma_usage, ventana %dh, faltantes por facet: %s) -- umbral %.0f%%",
            http_result["gap_pct"], http_result["expected"] - http_result["reconciled"],
            http_result["expected"], RECONCILIATION_WINDOW_SECONDS // 3600,
            http_result["missing_by_facet"], RECONCILIATION_ALERT_THRESHOLD_PCT,
        )
        await send_telegram_alert(
            f"⚠️ Jacobs: {http_result['gap_pct']:.0f}% de los dispatches HTTP-directos "
            f"(hipatia/jekyll/thot/ada) en las últimas "
            f"{RECONCILIATION_WINDOW_SECONDS // 3600}h no tienen fila de costo en "
            f"axioma_usage -- por facet: {http_result['missing_by_facet']}. Chequeo "
            f"aproximado (conteo por facet, sin job_id/step_id) -- ver "
            f"jacobs.reaper.check_usage_reconciliation / jacobs_events."
        )
    else:
        logger.info(
            "Reaper: reconciliación de usage (HTTP-directo) OK -- %d/%d dispatches con fila (%.1f%% gap)",
            http_result["reconciled"], http_result["expected"], http_result["gap_pct"],
        )

    return {"motor_registry": motor_result, "http_direct": http_result}


async def start_reaper_loop() -> None:
    """Barrido periódico en background. Se llama desde el startup del
    server además de esto -- ver server.py::_jacobs_init."""
    sweep_count = 0
    while True:
        try:
            await reap_orphaned_pipelines()
        except Exception:  # fail-soft: loop de limpieza en background, mismo patron que jax-platform/jax_engine/owner_cleanup.py -- nunca debe tumbar el proceso, el proximo ciclo reintenta
            logger.warning("Reaper: barrido periódico falló", exc_info=True)
        try:
            await check_facet_health()
        except Exception:  # fail-soft: loop de fondo, mismo patron que el barrido de arriba -- nunca debe tumbar el proceso, el proximo ciclo (300s) reintenta
            logger.error(
                "Reaper: chequeo de salud de facets FALLO -- el detector no "
                "corrio en este barrido; no hay salud calculada, no 'todo ok'",
                exc_info=True,
            )
        sweep_count += 1
        if sweep_count % RECONCILIATION_CHECK_EVERY_N_SWEEPS == 0:
            await check_usage_reconciliation()
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
