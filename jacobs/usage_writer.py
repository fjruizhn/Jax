# jax/jacobs/usage_writer.py
"""Escritura directa a axioma_usage (jax-platform) desde jacobs/executor.py,
para los 3 transportes HTTP directos (hipatia/gemini, jekyll+thot+ada/openai
compat, jax_local/ollama) cuando se invocan via un pipeline de Jacobs -- no
via la Mesa web (esa ruta ya escribe axioma_usage desde jax-platform/backend/
api/chat.py, Tasks 1-4).

Mismo patron de conexion que credential_resolver.py/store.py/
las_manos/motor_registry/usage_writer.py: cada repo se conecta a la misma DB
jax_memory con su propio conector minimo, sin paquete compartido. jacobs NO
importa motor_registry (no esta en su sys.path standalone, ver comentario en
jacobs/executor.py sobre el catalogo de capabilities) -- por eso este modulo
es una copia adaptada, no un import cruzado.

request_type='pipeline' (no 'chat'): distingue en /api/admin/usage estos
mismos transportes invocados DESDE un pipeline de Jacobs (posiblemente sin
supervision humana en el momento, corriendo en cadena con otros steps) de la
misma faceta invocada directo desde la Mesa web -- util para filtrar/atribuir
costo mas adelante sin tener que inferirlo de otra tabla.

COLA DURABLE (Task 7, 2026-09-15): si la base no esta, la fila ya no se pierde
-- se deposita en el respaldo de `cola_uso` con `origen='jacobs'`. **Este
proceso NO DRENA**: solo `jax-platform` lee el respaldo e inserta, porque es la
duena de `axioma_usage` y la unica con migraciones (la columna `spool_id` y su
UNIQUE, que es lo que hace idempotente al reintento). Si este modulo tambien
insertara, dos procesos borrarian el mismo archivo sin coordinacion y el cobro
se duplicaria en la ventana entre el INSERT y el borrado.

request_type='preflight_probe' (2026-09-17): la sonda del pre-vuelo de Jacobs
(jacobs/sonda.py) paga una llamada mínima; se registra aparte para que se vea.
request_type='preflight_probe_est' (ola final F4, 2026-09-17): la misma sonda
cuando venció o el proveedor respondió 2xx sin usage -- tokens ESTIMADOS."""
from __future__ import annotations

import logging
import os

import aiomysql

try:
    # LAS MANOS produccion (cwd=las_manos, uvicorn) y jobs con PYTHONPATH
    # incluyendo las_manos/: bare, resuelve a las_manos/db_connect_config.py
    # (symlink) o directo si jacobs corre con las_manos en su propio path.
    from db_connect_config import db_connect_timeout_seconds
except ImportError:
    # CI sin PYTHONPATH propio (p.ej. facet-health-io) y REPL: cwd=raiz del
    # repo, solo el paquete jax.core es importable.
    from jax.core.db_connect_config import db_connect_timeout_seconds

try:
    # Mismo doble import que db_connect_config, por la misma razon: en
    # produccion `jax.core` no es importable con cwd=las_manos, y
    # las_manos/cola_uso.py es un symlink a jax/core/cola_uso.py.
    from cola_uso import _ahora_iso, encolar as encolar_uso
except ImportError:
    from jax.core.cola_uso import _ahora_iso, encolar as encolar_uso

logger = logging.getLogger("jacobs.usage_writer")


def _entero_o_none(valor) -> int | None:
    """tenant_id/user_id son INT(11) en la tabla y llegan como string. El cast
    NO puede propagar: esto corre en el camino de recuperacion de una fila que
    ya se perdio una vez. Un valor no numerico entra como NULL -- exactamente
    lo mismo que hace hoy la columna cuando el INSERT lo rechaza, pero con la
    fila guardada en vez de tirada."""
    if valor is None:
        return None
    try:
        return int(valor)
    except (TypeError, ValueError):
        logger.warning("tenant_id/user_id no numerico (%r) -- se encola como NULL", valor)
        return None


def _db_cfg() -> dict:
    host = os.environ.get("JAX_DB_HOST")
    port = os.environ.get("JAX_DB_PORT")
    if not host or not port:
        raise RuntimeError(
            "JAX_DB_HOST/JAX_DB_PORT no están seteados -- sin default "
            "silencioso a localhost:3306 (esa instancia está muerta, ver "
            "memoria jax-dual-mariadb-instances). Sourceá /etc/jax/.env o "
            "exportalos a mano antes de conectar."
        )
    return {
        "host": host,
        "port": int(port),
        "user": os.getenv("JAX_DB_USER", ""),
        "password": os.getenv("JAX_DB_PASSWORD", ""),
        "db": os.getenv("JAX_DB_NAME", "jax_memory"),
        "charset": "utf8mb4",
        "autocommit": True,
    }


async def _lookup_model_price(conn, provider_id: str, model: str) -> tuple[float | None, float | None]:
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT price_input_per_1m_usd, price_output_per_1m_usd "
            "FROM model WHERE provider_id=%s AND model_id=%s",
            (provider_id, model),
        )
        row = await cur.fetchone()
    if not row:
        return None, None
    return row[0], row[1]


async def record_direct_usage(
    user_id: str | None,
    tenant_id: str | None,
    facet: str,
    provider_id: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    request_type: str = "pipeline",
) -> None:
    """Best-effort (usage tracking no debe romper un step ya completado),
    pero no silencioso.

    T1.c (2026-08-22, auditoria usage_writer): mismo bug que
    motor_registry/usage_writer.py::record_motor_usage -- sin user_id/
    tenant_id esto retornaba SIN loguear. Ahora escribe con tenant_id/
    user_id NULL (la columna lo permite) y loguea WARNING.

    T1.b (auditoria usage_writer, alcance de esta ronda): el escritor de
    Motor Registry tenía un bug real -- la llamada vivía solo en la rama de
    éxito, así que ningún job fallido contabilizaba, confirmado 7/9 jobs
    reales sin fila. Este escritor (transportes HTTP directos) reconcilió
    4/4 en la única corrida real disponible -- sin evidencia del mismo
    problema, NO se tocó el punto de llamada en executor.py::_dispatch_step
    esta ronda (tocar _invoke_http_gemini/_invoke_http_openai_compat/
    _invoke_ollama para capturar tokens parciales antes de una excepción
    sería un cambio no verificado). Deuda declarada, no una garantía.

    tenant_id/user_id: la columna es INT(11) -- el cast a int() se hace
    SOLO si no es None (None se inserta como NULL real, mismo motivo que
    record_motor_usage documenta)."""
    if not user_id or not tenant_id:
        logger.warning(
            f"record_direct_usage facet={facet} sin identidad "
            f"(user_id={user_id!r} tenant_id={tenant_id!r}) -- escribe con "
            f"tenant_id/user_id NULL, no se descarta"
        )
    # La hora del TURNO, leída ANTES del primer intento: si se tomara al
    # encolar, una caída de dos horas movería el costo al día siguiente.
    creado_en = _ahora_iso()
    cost = None
    try:
        # connect_timeout explícito (no en _db_cfg()): hallazgo de revisión,
        # Tarea 2b (tanda A, ronda de arreglo 1, 2026-09-14) -- sin esto,
        # aiomysql espera sin límite si la DB se cuelga.
        conn = await aiomysql.connect(**_db_cfg(), connect_timeout=db_connect_timeout_seconds())
        try:
            price_in, price_out = await _lookup_model_price(conn, provider_id, model)
            if price_in is not None and price_out is not None:
                cost = (tokens_in * float(price_in) + tokens_out * float(price_out)) / 1_000_000
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO axioma_usage (tenant_id, user_id, facet, model, tokens_in, tokens_out, cost_usd, request_type) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        int(tenant_id) if tenant_id is not None else None,
                        int(user_id) if user_id is not None else None,
                        facet, model, tokens_in, tokens_out, cost, request_type,
                    ),
                )
        finally:
            conn.close()
        return
    except Exception as e:  # fail-soft: la contabilidad no puede tumbar un step ya completado; la fila va al respaldo, no a la basura
        motivo = f"{type(e).__name__}: {e}"

    # T7 (2026-09-15): hasta hoy acá terminaba todo con un logger.error y la
    # fila se perdía para siempre. El turno ya se le cobró al proveedor: eso es
    # dinero real que el total de Admin -> Costos nunca volvía a ver. Ahora se
    # deposita en el respaldo. NO se drena desde acá (ver docstring del módulo).
    #
    # cost_usd va None cuando la caída fue antes del lookup de precios: sin base
    # no hay tabla `model` que consultar. La plataforma lo resuelve al insertar.
    spool_id = await encolar_uso({
        "created_at": creado_en,
        "tenant_id": _entero_o_none(tenant_id),
        "user_id": _entero_o_none(user_id),
        "facet": facet,
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost,
        "request_type": request_type,
        "origen": "jacobs",
        # explicitos aunque el contrato los admita ausentes: jacobs no invoca
        # trabajos del motor, y que se vean en None dice que es una decision y
        # no un campo que se olvido de mandar.
        "status": None,
        "job_id": None,
    })
    if spool_id:
        # INFO, no ERROR: encolada NO es perdida. Un ERROR acá entrena a
        # ignorar el log, y entonces el ERROR de abajo -- que sí es una pérdida
        # real -- no se distingue de nada.
        logger.info(
            f"record_direct_usage facet={facet} no pudo escribir en la base "
            f"({motivo}) -- fila encolada en el respaldo spool_id={spool_id}, "
            f"la inserta jax-platform"
        )
        return
    logger.error(
        f"record_direct_usage failed facet={facet} reason={motivo} -- TAMPOCO "
        f"se pudo encolar: fila PERDIDA (tokens_in={tokens_in} "
        f"tokens_out={tokens_out})"
    )
