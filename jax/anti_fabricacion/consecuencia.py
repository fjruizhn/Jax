"""Consecuencias del veredicto de estructura (anti-fabricación, capa 4).

`aplicar(veredicto, modo) -> Decision` es PURA: decide qué hacer, no lo hace. El
worker (las_manos/motor_registry/worker.py) ejecuta la decisión.

MODOS (axioma_config `anti_fabricacion.modo`):
  * `duro`   : un veredicto `marcado` termina el job FAILED con
               `error="estructura sin respaldo: ..."`, sin COMPLETED ni result_path.
  * `sombra` : el estado del job no cambia; solo queda el rastro.
En ambos modos un `marcado` deja `_unbacked_claims` y `_estructura` en el JSONL del
job, un evento en audit.jsonl y un aviso inmediato por Telegram. `no_verificable`
y `no_corrio` NO acusan (no hay estructura sin respaldo demostrada) pero quedan
visibles en `_estructura` y nunca figuran como `limpio`.

El modo ausente, ilegible, con la base caída o con un valor desconocido es
`duro`: fallar cerrado. Pasar a `duro` desde `sombra` lo decide Fernando.

Lectura del modo: una consulta por clave primaria (`axioma_config.config_key` es
PRI) por job. Sin caché a propósito: el camino del job ya hace una consulta
agregada para la huella y varios segundos de LLM; un caché del modo exigiría
invalidación entre procesos para que un cambio a `duro` valiera ya.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import Enum

from .estructura import MARCADO, Veredicto

try:
    # LAS MANOS / Jacobs (cwd las_manos/, jax.core no es importable ahí).
    from facet_resolver import _db_conn
except ImportError:  # pragma: no cover - según desde dónde se importe
    # Scripts y REPL (PYTHONPATH=. desde la raíz del repo).
    from jax.core.facet_resolver import _db_conn

logger = logging.getLogger(__name__)

CLAVE_MODO = "anti_fabricacion.modo"
SQL_MODO = "SELECT config_value FROM axioma_config WHERE config_key = %s"

#: Prefijo del error de un job marcado en modo duro (lo fija la spec).
PREFIJO_ERROR = "estructura sin respaldo: "
#: Prefijo del error cuando el detector no pudo derivar su whitelist.
ERROR_DETECTOR = "detector_no_disponible"
EVENTO_AUDITORIA = "ESTRUCTURA_SIN_RESPALDO"

_MAX_ERROR = 600
#: Cota de `_unbacked_claims` en el JSONL del job (el total real va en `_unbacked_claims_total`).
MAX_CLAIMS = 100


class Modo(str, Enum):
    DURO = "duro"
    SOMBRA = "sombra"


@dataclass(frozen=True)
class Decision:
    modo: Modo
    fallar: bool = False
    error: str | None = None
    avisar: bool = False
    #: campos privados del JSONL del job: `_unbacked_claims` y `_estructura`
    guardar: dict = field(default_factory=dict)


def aplicar(veredicto: Veredicto, modo: Modo) -> Decision:
    guardar = {
        "_unbacked_claims": [asdict(i) for i in veredicto.items[:MAX_CLAIMS]],
        "_unbacked_claims_total": len(veredicto.items),
        "_estructura": veredicto.como_dict(),
    }
    if veredicto.estado != MARCADO:
        return Decision(modo=modo, guardar=guardar)
    if modo == Modo.DURO:
        detalle = "; ".join(f"{i.candidato} [{i.tipo}, {i.clase}]" for i in veredicto.items)
        return Decision(modo=modo, fallar=True, error=(PREFIJO_ERROR + detalle)[:_MAX_ERROR],
                        avisar=True, guardar=guardar)
    return Decision(modo=modo, avisar=True, guardar=guardar)


async def leer_modo(conectar=None, *, motivos: list[str] | None = None) -> Modo:
    """Modo vigente. Cualquier cosa que no sea exactamente `sombra` o `duro` es `duro`.

    `motivos`: si se pasa una lista, se le agrega POR QUÉ el modo cayó al `duro` por defecto
    (clave ausente, valor inválido, base ilegible). Con la clave bien puesta queda vacía. El
    worker la usa para avisar una vez por proceso: un `duro` por omisión no es silencioso."""
    try:
        conn = await (conectar or _db_conn)()
        try:
            async with conn.cursor() as cur:
                await cur.execute(SQL_MODO, (CLAVE_MODO,))
                fila = await cur.fetchone()
        finally:
            conn.close()
        valor = str(fila[0]).strip().lower() if fila and fila[0] is not None else ""
        if valor == Modo.SOMBRA.value:
            return Modo.SOMBRA
        if valor != Modo.DURO.value:
            razon = (f"la clave {CLAVE_MODO} falta en axioma_config" if not valor
                     else f"{CLAVE_MODO}={valor!r} no es válido")
            logger.warning("anti_fabricacion: %s: se usa 'duro'", razon)
            if motivos is not None:
                motivos.append(razon)
        return Modo.DURO
    except Exception as exc:  # fail-closed: modo ilegible o base caída -> duro, nunca sombra por accidente
        logger.warning("anti_fabricacion: no se pudo leer %s (%s: %s): se usa 'duro'",
                       CLAVE_MODO, type(exc).__name__, exc)
        if motivos is not None:
            motivos.append(f"no se pudo leer {CLAVE_MODO} ({type(exc).__name__})")
        return Modo.DURO


def mensaje_aviso(*, job_id: str, motor: str, capability: str, veredicto: Veredicto, modo: Modo,
                  pipeline_id: str | None) -> str:
    """Texto del aviso inmediato de Telegram para un job marcado."""
    marca = "🔴" if modo == Modo.DURO else "🟡"
    consecuencia = "el job FALLÓ" if modo == Modo.DURO else "modo sombra: el job no cambió de estado"
    detalle = "\n".join(f"- {i.candidato} ({i.tipo}, {i.clase}): {i.motivo}" for i in veredicto.items[:10])
    return (
        f"{marca} JAX anti-fabricación: job {job_id} ({motor}/{capability}"
        f"{', pipeline ' + pipeline_id if pipeline_id else ''}) afirma estructura sin respaldo; {consecuencia}.\n"
        f"{detalle}\n"
        f"(escapes usados en esta salida: nuevo:={veredicto.usos_nuevo}, «?»={veredicto.usos_hipotesis})"
    )
