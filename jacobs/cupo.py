"""Jacobs — el cupo de pipelines concurrentes, hecho cumplir por LA BASE.

POR QUÉ EXISTE ESTE MÓDULO (2026-09-17, decisión de Fernando).
`routes._pipeline_create_lock` era un `asyncio.Lock()` global del proceso.
Envolvía leer el conteo de activos, consumir el token de sub-pipeline,
planificar (20-40 s de LLM) e insertar la fila, porque leer el conteo en una
consulta y decidir en otra es una lectura optimista: dos POST concurrentes
leían el mismo número y los dos pasaban. El candado tapaba esa carrera, pero
serializaba TODA la creación en un solo objeto compartido — la medición del
frente G (2026-09-17, hall9000, instancias aisladas) dio un techo de ~43
delegaciones/s, y la creación de pipelines de la Mesa esperaba 70-3200 ms (p95,
c=5 a c=25) detrás de las delegaciones de Ada.

CÓMO SE ARREGLA DE RAÍZ. El mismo patrón que ya hace cumplir el token de
sub-pipelines (`jacobs/subpipelines.py`): UNA sentencia que decide, en
autocommit, y la decisión sale de las FILAS AFECTADAS, no de una lectura
previa. `INSERT ... SELECT ... WHERE (SELECT COUNT(*) ...) < limite`:
  - 1 fila afectada  -> hay cupo y la reserva es tuya;
  - 0 filas          -> cupo agotado, el mismo rechazo explícito de antes.
Nadie puede colarse entre el conteo y el INSERT porque son la misma sentencia.

EL DEADLOCK ES PARTE DEL CONTRATO, NO UN DETALLE. El `INSERT ... SELECT` lee
la tabla en la que inserta, así que InnoDB toma candados compartidos sobre el
rango de `status` y después intenta insertar en ese mismo rango: dos reservas
simultáneas se traban y una muere con el error 1213. Es justo lo que hace
correcta a la sentencia (una lectura no bloqueante daría el conteo de un
snapshot y volvería la carrera), así que no se evita: se REINTENTA. Medido el
2026-09-17 en hall9000 contra `jax_memory_test`: sin reintento mueren 5 de 10,
22 de 25 y 21 de 50 intentos; con reintento, cero errores y p95 8,41 ms a c=50.

RESERVAR PRIMERO, PLANIFICAR DESPUÉS — y por qué ese orden importa.
La reserva se toma ANTES de consumir el token de Ada y ANTES de planificar.
Así se conservan los DOS invariantes que sostenía el candado, no sólo el del
conteo:
  - ningún token de sub-pipeline se quema sin que haya cupo (si el cupo se
    comprobara al final, un hijo de Ada gastaría su token y 20-40 s de LLM
    para recibir un 422);
  - la planificación sigue acotada: sólo puede haber tantas en vuelo como cupo
    hay (3), porque cada una tiene su reserva. Antes el candado la acotaba a 1.
La contrapartida es que existe una fila `pending` sin plan mientras se
planifica. No es un estado nuevo: `routes.create_pipeline` ya dejaba filas
`pending` antes de arrancar el background, y el reaper cosecha lo `pending` de
más de `PENDING_MAX_AGE_SECONDS` (300 s) — quince veces el peor plan medido.
Y el camino de fallo suelta la reserva explícitamente (`soltar_reserva`).

En honor al Prof. Raúl Jacobs. En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random

import aiomysql

from jacobs import store
from jacobs.models import Pipeline, PipelineStatus
from jacobs.policy import (
    ESTADOS_QUE_OCUPAN_CUPO,
    ESTADOS_SIN_CUPO,
    MAX_PARALLEL_PIPELINES,
    SQL_ESTADOS_VIVOS,
    CupoAgotado,
)

logger = logging.getLogger(__name__)

#: Error de MariaDB "Deadlock found when trying to get lock".
_DEADLOCK = 1213

#: Reintentos del 1213. Con 50 corrutinas nunca hizo falta más de uno, pero la
#: prueba de carga del 2026-09-17 (25 VUs, ~2.900 reservas/s contra el cupo
#: lleno) agotó CINCO reintentos 711 veces y las devolvió como 500. Fail-closed,
#: sí, pero un 500 lo ve el usuario. Doce con espera creciente hasta 50 ms cubre
#: ese peor caso medido; agotarlos sigue levantando el error, nunca devuelve
#: "reservado" sin fila.
MAX_REINTENTOS_DEADLOCK = 12

#: Tope de la espera entre reintentos. Doce reintentos con crecimiento x2 desde
#: ~3 ms suman ~0,4 s en el peor caso: menos que el timeout del pool y mucho
#: menos que lo que tarda planificar.
ESPERA_MAXIMA_SEGUNDOS = 0.05


#: LA sentencia que decide. Una sola, autocommit, y se juzga por `rowcount`.
#: `FROM DUAL` para que el SELECT no tenga tabla de origen: lo único que se lee
#: es el COUNT del cupo.
SQL_RESERVAR = f"""
INSERT INTO jacobs_pipelines
    (pipeline_id, name, invoked_by, mode, status, plan, current_step_index,
     max_steps, context_refs, created_at, updated_at, user_id, tenant_id,
     parent_pipeline_id, depth)
SELECT %s, %s, %s, %s, %s, '[]', 0, %s, %s, %s, %s, %s, %s, NULL, 0
FROM DUAL
WHERE (SELECT COUNT(*) FROM jacobs_pipelines WHERE status IN ({SQL_ESTADOS_VIVOS})) < %s
"""

#: Completa la reserva con lo que sólo se sabe después: el plan, y —para un hijo
#: de Ada— el padre, la profundidad y la identidad, que salen de la FILA del
#: token y nunca del cuerpo del pedido (I-3 del frente F).
SQL_COMPLETAR = """
UPDATE jacobs_pipelines
   SET plan = %s, context_refs = %s, parent_pipeline_id = %s, depth = %s,
       user_id = %s, tenant_id = %s, updated_at = %s
 WHERE pipeline_id = %s
"""

#: Suelta una reserva que nunca llegó a ser pipeline. El `status='pending'` no
#: es decoración: impide que un error de programación borre un pipeline que ya
#: arrancó.
SQL_SOLTAR = "DELETE FROM jacobs_pipelines WHERE pipeline_id = %s AND status = %s"


#: Re-exportados desde `jacobs/policy.py`, que es donde viven para que
#: `jacobs/store.py` pueda usarlos sin importar este módulo (y al revés).
#: `ESTADOS_QUE_OCUPAN_CUPO`, `ESTADOS_SIN_CUPO` y `CupoAgotado` se siguen
#: pidiendo por acá porque este es el módulo del cupo.
__all__ = [
    "CupoAgotado", "ESTADOS_QUE_OCUPAN_CUPO", "ESTADOS_SIN_CUPO",
    "reservar_cupo", "completar_reserva", "soltar_reserva", "activos",
    "SQL_RESERVAR", "parametros_de_reserva",
]


def parametros_de_reserva(p: Pipeline, limite: int) -> tuple:
    """Los parámetros de `SQL_RESERVAR`. Función aparte para que el test del
    EXPLAIN mida la sentencia REAL, con los mismos valores, y no una copia que
    se puede desincronizar del código que corre en producción."""
    return (
        p.pipeline_id, p.name, p.invoked_by, p.mode, PipelineStatus.pending.value,
        p.max_steps, json.dumps(p.context, ensure_ascii=False),
        p.created_at, p.updated_at,
        p.user_id, p.tenant_id,
        limite,
    )


async def _ejecutar_reserva(p: Pipeline, limite: int) -> int:
    """Una pasada de la sentencia. Devuelve las filas afectadas (0 o 1).

    Aislada del reintento a propósito: el test de "deadlock que no cede" la
    sustituye para comprobar que agotar los reintentos levanta el error en vez
    de devolver una reserva inventada.
    """
    async with store.conexion() as conn:
        async with conn.cursor() as cur:
            await cur.execute(SQL_RESERVAR, parametros_de_reserva(p, limite))
            return cur.rowcount


async def reservar_cupo(p: Pipeline, limite: int | None = None) -> bool:
    """Reserva un lugar para `p` y escribe su fila `pending`, o devuelve False.

    True  = la fila existe y el cupo es tuyo hasta que el pipeline termine o
            hasta que lo sueltes con `soltar_reserva`.
    False = cupo agotado. El llamador responde el mismo rechazo explícito de
            siempre (422 con el motivo).

    Fail-closed: cualquier error de base (pool lleno, socket caído, deadlock
    que no cede) SUBE. No hay rama que cree el pipeline "por las dudas": sin
    base no se sabe si hay cupo, y sin saberlo no se crea nada.
    """
    tope = MAX_PARALLEL_PIPELINES if limite is None else limite
    ultimo: BaseException | None = None
    for intento in range(MAX_REINTENTOS_DEADLOCK):
        try:
            return await _ejecutar_reserva(p, tope) == 1
        except aiomysql.OperationalError as exc:
            if exc.args[0] != _DEADLOCK:
                raise
            ultimo = exc
            # Espera con jitter y crecimiento: sin azar, las dos víctimas de un
            # deadlock reintentan a la vez y se vuelven a trabar; sin
            # crecimiento, una tormenta de reservas no se dispersa nunca.
            await asyncio.sleep(min(
                random.uniform(0.001, 0.005) * (2 ** intento),
                ESPERA_MAXIMA_SEGUNDOS,
            ))
    logger.error(
        "cupo: %d reintentos de deadlock agotados reservando %s",
        MAX_REINTENTOS_DEADLOCK, p.pipeline_id,
    )
    assert ultimo is not None
    raise ultimo


async def completar_reserva(p: Pipeline, conexion=None) -> None:
    """Escribe en la fila reservada lo que sólo se sabe después de planificar.

    `conexion`: la del bloque de la transacción de creación, para que el plan,
    los pasos y los eventos entren todo-o-nada. Sin ella, una del pool.
    """
    args = (
        json.dumps([s.model_dump() for s in p.plan], ensure_ascii=False),
        json.dumps(p.context, ensure_ascii=False),
        p.parent_pipeline_id, p.depth, p.user_id, p.tenant_id,
        p.updated_at, p.pipeline_id,
    )
    async with store._conexion_o_pool(conexion) as conn:
        async with conn.cursor() as cur:
            await cur.execute(SQL_COMPLETAR, args)


async def soltar_reserva(pipeline_id: str) -> int:
    """Devuelve el cupo de una reserva que no llegó a ser pipeline.

    Devuelve las filas borradas (0 si el pipeline ya no estaba `pending`).
    """
    async with store.conexion() as conn:
        async with conn.cursor() as cur:
            await cur.execute(SQL_SOLTAR, (pipeline_id, PipelineStatus.pending.value))
            return cur.rowcount


async def activos() -> int:
    """Cuántos pipelines ocupan cupo ahora mismo.

    NO decide nada: la decisión es `reservar_cupo`. Esto sirve para el MENSAJE
    del rechazo (cuántos había) y para los tests. Usarlo para decidir sería
    volver a la lectura optimista que este módulo existe para borrar.

    Delega en `store.pipeline_count_active()` a propósito: el criterio de "qué
    ocupa cupo" no puede tener dos copias que se desincronicen. La única otra
    aparición de esos estados es `SQL_RESERVAR`, que es la sentencia que decide
    y no puede delegar en nadie.
    """
    return await store.pipeline_count_active()
