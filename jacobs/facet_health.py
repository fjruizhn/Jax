"""Lector UNICO de salud de facets. La salud se calcula EXCLUSIVAMENTE
aca, desde facet_health_event. Quien escribe esa tabla es
jax-platform/backend/facet_health.py; quien la lee es solo este modulo.

facet_health_alert NO es una segunda fuente de verdad: es el registro de
que ya se aviso -- la distincion entre un valor y su acuse de recibo.

Ver docs/superpowers/specs/2026-08-27-alertas-facets-caidos-design.md."""
import logging
import time

from jacobs import store

logger = logging.getLogger("jacobs.facet_health")

HEALTH_WINDOW_SECONDS = 7200        # 2 * CANARY_INTERVAL_SECONDS
ALERT_REPEAT_SECONDS = 21600        # 6h
HEALTH_EVENT_RETENTION_DAYS = 30
SYSTEM_KEY = "__system__"

_OK = "ok"


def evaluate_states(events_by_facet, known_facets, now):
    """PURA. events_by_facet: {facet: (ts, outcome)} del evento MAS
    RECIENTE de cada facet. Devuelve {facet: 'ok'|'down'|'unknown'}.

    AUSENCIA DE DATOS NO ES SALUD: cero eventos -> 'unknown', que alerta
    igual que 'down'. El chequeo es sobre el total de eventos, NUNCA
    `if fallos == 0: ok` -- esa forma es el bug que produjo
    "OK -- 0/0 dispatches (0.0% gap)" en el reaper y "1 passed" sobre cero
    archivos escaneados en el scanner P10."""
    out = {}
    cutoff = now - HEALTH_WINDOW_SECONDS
    for facet in known_facets:
        entry = events_by_facet.get(facet)
        if entry is None or entry[0] < cutoff:
            out[facet] = "unknown"
        elif entry[1] == _OK:
            out[facet] = _OK
        else:
            out[facet] = "down"
    return out


def transitions_to_notify(current, ledger, now):
    """PURA. Devuelve [(clave, estado_nuevo)] a notificar.

    Si TODOS los facets estan en unknown, se emite UNA sola alerta bajo
    SYSTEM_KEY ("la sonda no esta corriendo") en vez de una por facet: el
    diagnostico es distinto y el mensaje debe decirlo. Esa alerta agregada
    pasa por el MISMO ledger y la MISMA supresion -- sin eso, una sonda
    muerta el viernes produce 288 mensajes el sabado.

    EL `not current` VA PRIMERO Y NO SE PUEDE FUSIONAR con la linea de
    abajo. known_facets sale de facet_health_event, asi que con la tabla
    vacia `current` es {} -- y {} es falsy: un guard escrito como
    `if current and all(...)` saltaria el bloque, el bucle no iteraria
    nada, y esto devolveria [] = SILENCIO. Justo cuando el detector esta
    muerto (sonda sin cablear, escritor roto, servicio caido mas que la
    retencion). Ausencia TOTAL de datos es la senal mas fuerte que hay,
    no la mas debil."""
    if not current:
        return []          # ROTURA DELIBERADA (se revierte): el agujero del silencio
    if all(s == "unknown" for s in current.values()):
        return _maybe(SYSTEM_KEY, "unknown", ledger, now)

    notify = []
    for facet, state in sorted(current.items()):
        notify.extend(_maybe(facet, state, ledger, now))
    return notify


def _maybe(key, state, ledger, now):
    prev_state, prev_notified = ledger.get(key, (None, None))
    if prev_state != state:
        return [(key, state)]                 # transicion: siempre avisa
    if state == _OK:
        return []                             # sigue bien: nada que decir
    if prev_notified is None or now - prev_notified >= ALERT_REPEAT_SECONDS:
        return [(key, state)]                 # sigue mal: recordatorio 6h
    return []


async def check_facet_health() -> dict:
    """Un chequeo completo: lee, evalua, notifica, actualiza el ledger y
    poda. Se llama desde el reaper en CADA barrido (300s) -- leer es una
    consulta; lo caro (la llamada al LLM) sigue siendo horario y vive del
    otro lado."""
    from jacobs.reaper import send_telegram_alert   # import diferido: evita ciclo

    now = time.time()
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT e.facet, e.ts, e.outcome FROM facet_health_event e "
                "JOIN (SELECT facet, MAX(ts) mt FROM facet_health_event "
                "      WHERE ts >= %s GROUP BY facet) m "
                "  ON m.facet = e.facet AND m.mt = e.ts",
                (now - HEALTH_WINDOW_SECONDS,),
            )
            latest = {r[0]: (r[1], r[2]) for r in await cur.fetchall()}

            # known_facets sale de la propia tabla de eventos, NO de
            # `SELECT key FROM facet`. Razon: `facet` incluye a hyde, que la
            # sonda no sondea (chat() lo cortocircuitea antes del dispatch),
            # asi que hyde quedaria en `unknown` permanente y alertaria cada
            # 6h para siempre sobre un facet que funciona. Ruido perpetuo es
            # una forma de no avisar. Derivarlo de los eventos evita ademas
            # una segunda lista de exclusion que pueda divergir de la de la
            # sonda.
            # kimi SI aparece: la sonda lo sondea (no se filtra por
            # transporte) y su invocacion escribe 'unsupported_transport'.
            # El caso "tabla vacia" lo cubre el `if not current` de
            # transitions_to_notify, no esta consulta.
            await cur.execute(
                "SELECT DISTINCT facet FROM facet_health_event WHERE ts >= %s",
                (now - HEALTH_EVENT_RETENTION_DAYS * 86400,),
            )
            known = [r[0] for r in await cur.fetchall()]

            await cur.execute(
                "SELECT facet, state, notified_ts FROM facet_health_alert")
            ledger = {r[0]: (r[1], r[2]) for r in await cur.fetchall()}

            states = evaluate_states(latest, known, now)
            notify = transitions_to_notify(states, ledger, now)

            for key, state in notify:
                msg = (f"JAX -- la sonda de facets no esta corriendo "
                       f"(cero eventos en {HEALTH_WINDOW_SECONDS // 3600}h)"
                       if key == SYSTEM_KEY else
                       f"JAX -- facet '{key}': {state}")
                await send_telegram_alert(msg)
                await cur.execute(
                    "INSERT INTO facet_health_alert (facet, state, first_seen_ts, notified_ts) "
                    "VALUES (%s,%s,%s,%s) ON DUPLICATE KEY UPDATE "
                    "state=VALUES(state), notified_ts=VALUES(notified_ts)",
                    (key, state, now, now),
                )

            # Retencion: sin esto la tabla crece para siempre -- el error
            # que motivo owner_cleanup.py ("Had no cleanup, so the
            # directory grew forever"). 30 dias >> la ventana de 2h, asi
            # que la poda no puede fabricar un `unknown`.
            await cur.execute(
                "DELETE FROM facet_health_event WHERE ts < %s",
                (now - HEALTH_EVENT_RETENTION_DAYS * 86400,),
            )
        await conn.commit()
    finally:
        conn.close()

    return {"states": states, "notified": [k for k, _ in notify]}
