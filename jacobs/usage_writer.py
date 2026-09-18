# jax/jacobs/usage_writer.py
"""Escritura directa a axioma_usage (jax-platform) desde jacobs/executor.py,
para los 3 transportes HTTP directos (hipatia/gemini, jekyll+thot+ada/openai
compat, jax_local/ollama) cuando se invocan via un pipeline de Jacobs -- no
via la Mesa web (esa ruta ya escribe axioma_usage desde jax-platform/backend/
api/chat.py, Tasks 1-4).

CONEXION (2026-09-17, Ruling R38 fix round 1): por el pool del store de Jacobs
(jacobs/store.py::conexion / conexion_del_pool), no una conexion suelta por
fila: esto corre tras cada step HTTP de un pipeline y tras cada sonda del
pre-vuelo. Una falla del pool (base caida, pool agotado) sube como excepcion y
toma el MISMO camino que antes tomaba un `aiomysql.connect` caido: la fila va a
la cola durable. Antes: cada repo se conecta a la misma DB jax_memory con su
propio conector minimo, sin paquete compartido. jacobs NO importa
motor_registry (no esta en su sys.path standalone, ver comentario en
jacobs/executor.py sobre el catalogo de capabilities) -- por eso este modulo es
una copia adaptada de las_manos/motor_registry/usage_writer.py, no un import
cruzado. OJO (m5 de la re-revision final, 2026-09-17):
la direccion CONTRARIA si existe desde R38 fix round 3: `las_manos/motor_registry/worker.py importa jacobs.store`
para correr el job bajo `espera_de_turno_sin_plazo`; esta declarada en el
docstring de ese archivo. O sea: motor_registry -> jacobs, si;
jacobs -> motor_registry, no.

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

from jacobs import store

try:
    # Doble import (mismo patron que db_connect_config en store.py): en
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
    pipeline_id: str | None = None,
) -> None:
    """Best-effort (usage tracking no debe romper un step ya completado),
    pero no silencioso.

    pipeline_id (Task 7b, 2026-09-18): Pipeline.pipeline_id (jacobs/models.py)
    de quien dispara este uso. executor.py lo manda siempre (un step SIEMPRE
    corre dentro de un pipeline); sonda.py (pre-vuelo) NO lo manda -- una
    sonda no es un paso de un pipeline, y None ahí es correcto, no un hueco.
    Sin esta columna, api/pipelines.py::list_pipelines() (jax-platform) no
    tiene con qué sumar el costo real de un pipeline (Principio VIII: hasta
    esta ronda costo_usd salía null siempre, documentado en el HALLAZGO de
    Task 7 de jax-platform).

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
        # R38, fix round 1 (2026-09-17): por el pool del store de Jacobs y no
        # por una conexión propia -- la sonda del pre-vuelo escribe acá en el
        # camino de /jacobs/preflight, crear y continue, y el ejecutor en cada
        # paso HTTP. El guard de JAX_DB_HOST/PORT, el connect_timeout y el
        # límite de espera por hueco viven en store. Un error del pool (turno
        # vencido en un pedido, base caída) cae al respaldo igual que antes
        # caía un connect fallido; el ejecutor espera turno sin plazo
        # (store.espera_de_turno_sin_plazo).
        async with store.conexion() as conn:
            price_in, price_out = await _lookup_model_price(conn, provider_id, model)
            if price_in is not None and price_out is not None:
                cost = (tokens_in * float(price_in) + tokens_out * float(price_out)) / 1_000_000
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO axioma_usage (tenant_id, user_id, facet, model, tokens_in, tokens_out, cost_usd, request_type, pipeline_id) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        int(tenant_id) if tenant_id is not None else None,
                        int(user_id) if user_id is not None else None,
                        facet, model, tokens_in, tokens_out, cost, request_type, pipeline_id,
                    ),
                )
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
        "pipeline_id": pipeline_id,
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
