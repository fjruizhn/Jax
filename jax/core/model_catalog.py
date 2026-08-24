"""
Catalogo de modelos — captura de resolved_version (Bloque D, D1.2). Espejo
minimo en jax-platform, jax/core, las_manos, mismo patron que
facet_resolver.py/credential_resolver.py: repos/venvs independientes, no
justifica un paquete compartido.

Alcance de ESTE espejo: solo record_resolved_version (D1.2). El sync de
3 capas (D1.3), la deprecacion (D1.4) y los endpoints de aprobacion viven
unicamente en jax-platform (unico repo con admin UI) — no se duplican aca.

Decision de que se captura por transporte (Hyde=NULL nunca llamado,
jax_local=capturado con limitacion documentada): ver CONTEXT.md, entrada
"decision previa al wiring de resolved_version en REPL/Jacobs".

Ver jax-platform/docs/fase2-facetas-diseno.md D1.2/D1.3.
"""
import logging
import os

import aiomysql

logger = logging.getLogger("model_catalog")


async def _db_conn() -> aiomysql.Connection:
    host = os.environ.get("JAX_DB_HOST")
    port = os.environ.get("JAX_DB_PORT")
    if not host or not port:
        raise RuntimeError(
            "JAX_DB_HOST/JAX_DB_PORT no están seteados -- sin default "
            "silencioso a localhost:3306 (esa instancia está muerta, ver "
            "memoria jax-dual-mariadb-instances). Sourceá /etc/jax/.env o "
            "exportalos a mano antes de conectar."
        )
    return await aiomysql.connect(
        host=host,
        port=int(port),
        user=os.getenv("JAX_DB_USER", ""),
        password=os.getenv("JAX_DB_PASSWORD", ""),
        db=os.getenv("JAX_DB_NAME", "jax_memory"),
        charset="utf8mb4",
        autocommit=True,
    )


async def record_resolved_version(facet_key: str, resolved_version: str) -> dict:
    """D1.2 — best-effort, fire-and-forget desde el llamador (el invocador
    por transporte atrapa la excepcion, nunca rompe la respuesta al
    usuario). Compara contra el ultimo valor observado; si cambio, crea un
    model_binding_proposal (la alerta ES la proposal pendiente — decision
    D1.1, sin tabla de log aparte). La primera observacion nunca es drift."""
    if not resolved_version:
        return {"drift": False, "proposal_id": None}

    conn = await _db_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT provider_id, model_ref, resolved_version FROM facet_binding "
                "WHERE facet_key=%s AND role='primary'",
                (facet_key,),
            )
            row = await cur.fetchone()
            if not row:
                return {"drift": False, "proposal_id": None}
            provider_id, current_model_ref, previous_resolved = row

            await cur.execute(
                "UPDATE facet_binding SET resolved_version=%s, resolved_version_checked_at=NOW() "
                "WHERE facet_key=%s AND role='primary'",
                (resolved_version, facet_key),
            )

            if previous_resolved is None or previous_resolved == resolved_version:
                return {"drift": False, "proposal_id": None}

            await cur.execute(
                "INSERT IGNORE INTO model (provider_id, model_id, is_alias, status, source, source_checked_at) "
                "VALUES (%s, %s, FALSE, 'available', 'observed', NOW())",
                (provider_id, resolved_version),
            )
            await cur.execute(
                "SELECT id FROM model WHERE provider_id=%s AND model_id=%s",
                (provider_id, resolved_version),
            )
            (proposed_model_ref,) = await cur.fetchone()

            await cur.execute(
                "INSERT INTO model_binding_proposal "
                "(facet_key, current_model_ref, proposed_model_ref, reason, detail) "
                "VALUES (%s, %s, %s, 'drift_detected', %s)",
                (
                    facet_key, current_model_ref, proposed_model_ref,
                    f"resolved_version cambio de '{previous_resolved}' a '{resolved_version}'",
                ),
            )
            proposal_id = cur.lastrowid
    finally:
        conn.close()

    logger.warning(
        f"model_catalog drift facet={facet_key} from={previous_resolved} to={resolved_version} proposal_id={proposal_id}"
    )
    return {"drift": True, "proposal_id": proposal_id}


async def record_resolved_version_safe(facet_key: str, resolved_version: str | None) -> None:
    """Envoltorio fire-and-forget real: atrapa CUALQUIER excepcion (DB
    caida, tabla ausente, lo que sea) y solo logea — nunca debe poder
    romper la respuesta al usuario. Es lo que los invocadores por
    transporte deben llamar, no record_resolved_version directo."""
    if not resolved_version:
        return
    try:
        await record_resolved_version(facet_key, resolved_version)
    except Exception as e:
        logger.warning(f"resolved_version capture failed facet={facet_key} reason={type(e).__name__}")
