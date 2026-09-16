"""Migraciones idempotentes del esquema de memoria.

POR QUE EXISTE. `jax_memory_schema.sql` DESCRIBE el esquema (lo usa el checker
de drift y los tests para crear tablas), pero no lo APLICA: hasta 2026-09-11 no
habia ningun migrador para `conversations`/`messages`/`facts`. Los cambios se
hacian con un `ALTER TABLE` a mano contra produccion, que es exactamente como
la columna `depends_on` de `jacobs_steps` termino existiendo en produccion y en
ninguna base nueva (ver DEUDA.md). Un ALTER manual no llega a un dev nuevo, ni
a CI, ni a un restore de desastre.

Mismo patron que `jacobs/store.py::init_tables()`: chequeo previo contra
`information_schema` y despues el DDL. Corre en CADA arranque de los tres
procesos, asi que tiene que ser barato (cuatro SELECT sobre catalogo) y no
puede fallar el arranque: si algo sale mal, se registra y JAX sigue -- una
memoria degradada es mejor que un servicio que no levanta.

NO crea tablas: si la base esta vacia, esto no la puebla. Es deliberado --
crear el esquema completo desde aca seria una SEGUNDA fuente de verdad frente
a `jax_memory_schema.sql`. Esto solo lleva un esquema existente hacia adelante.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# (tabla, columna, DDL) — aditivas y nullable a proposito: una columna nueva
# NOT NULL sobre una tabla con datos exige un default y bloquea la tabla.
_COLUMNAS = [
    # Scope desnormalizado desde `conversations`. Existe para que el WHERE del
    # scope y el ORDER BY vectorial vivan en la MISMA tabla que el indice
    # HNSW: el JOIN a `conversations` sacaba al optimizador del indice
    # (58,5 ms contra 0,4 ms medidos el 2026-09-11 sobre 1.149 filas).
    ("messages", "user_id", "ALTER TABLE messages ADD COLUMN user_id INT NULL"),
    ("messages", "project_id", "ALTER TABLE messages ADD COLUMN project_id INT NULL"),
]

# (tabla, indice, DDL). MariaDB no acepta CREATE INDEX IF NOT EXISTS.
_INDICES = [
    # Compuesto (project_id, user_id): cubre las dos ramas del scope
    # --proyecto compartido e individual-- con un solo indice. El orden importa:
    # project_id primero porque la rama individual filtra por
    # `project_id IS NULL AND user_id = %s`, y un IS NULL sobre la columna que
    # ENCABEZA el indice si lo usa.
    ("messages", "idx_msg_scope",
     "CREATE INDEX idx_msg_scope ON messages (project_id, user_id)"),
]

# Backfill de las filas anteriores a la columna. Idempotente por el WHERE: solo
# toca las que todavia no tienen la copia. Sin esto, toda la memoria historica
# queda invisible para la busqueda -- una migracion a medias que se ve como
# "JAX se olvido de todo".
_BACKFILL = [
    ("messages", "user_id",
     "UPDATE messages m JOIN conversations c ON m.conversation_id = c.id "
     "SET m.user_id = c.user_id, m.project_id = c.project_id "
     "WHERE m.user_id IS NULL AND m.project_id IS NULL"),
]


async def ensure_schema(pool) -> bool:
    """Aplica lo que falte. Devuelve True si el esquema quedo al dia.

    Fail-soft y RUIDOSO: un error se registra como ERROR y devuelve False, pero
    no lanza -- el arranque de JAX no depende de esto.
    """
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                for tabla, columna, ddl in _COLUMNAS:
                    await cur.execute(
                        "SELECT COUNT(*) FROM information_schema.COLUMNS "
                        "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND COLUMN_NAME=%s",
                        (tabla, columna),
                    )
                    if not (await cur.fetchone())[0]:
                        await cur.execute(ddl)
                        logger.info("migracion: %s.%s agregada", tabla, columna)

                for tabla, columna, ddl in _BACKFILL:
                    await cur.execute(ddl)
                    if cur.rowcount:
                        logger.info("migracion: backfill de %s.%s en %d filas",
                                    tabla, columna, cur.rowcount)

                for tabla, indice, ddl in _INDICES:
                    await cur.execute(
                        "SELECT COUNT(*) FROM information_schema.STATISTICS "
                        "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND INDEX_NAME=%s",
                        (tabla, indice),
                    )
                    if not (await cur.fetchone())[0]:
                        await cur.execute(ddl)
                        logger.info("migracion: indice %s creado", indice)
        return True
    except Exception as e:  # fail-soft: el False que devuelve SI se consume desde connect() (self.schema_ok) y health_check() lo reporta como base NO sana; migrar es best-effort, servir mintiendo sobre el esquema no
        logger.error("migracion del esquema de memoria fallo: %s -- se sigue con "
                     "el esquema que haya", e)
        return False
