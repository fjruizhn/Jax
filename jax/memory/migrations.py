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
import re

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
    # Autoria de la aprobacion (spec 2026-09-18-memoria-admin §2.2: "con quien y
    # cuando"). `verified_at` ya existia; el quien, no. NULL para las filas
    # viejas: no se inventa un aprobador retroactivo -- el unico hecho
    # verificado de antes de esta ronda queda con autor desconocido, y eso es
    # la verdad.
    #
    # `AFTER verified_at` / `AFTER verified_by` (agregado 2026-09-20,
    # auditoria adversarial): SIN el AFTER, un `ALTER TABLE ADD COLUMN` pone
    # la columna nueva al FINAL de la tabla, pero jax_memory_schema.sql (la
    # fuente de verdad del checker de deriva) la declara en el MEDIO, entre
    # `verified_at` y `expires_at`. Una base MIGRADA (no creada de cero desde
    # el .sql) terminaba con las columnas en otra posicion que una base
    # nueva -- mismas columnas, mismo tipo, y el checker de deriva
    # (scripts/check_memory_schema_drift.py) salia "DIFIERE" con el mismo
    # texto a los dos lados: el peor rojo posible, porque parece un bug del
    # propio checker y no del esquema. Medido 2026-09-20: reproducido contra
    # jax_memory_test en hall9000. Ver test_el_migrador_agrega_las_columnas_
    # en_la_MISMA_posicion_que_el_esquema (test_memoria_gobernanza.py), que
    # ata esta posicion a la del .sql.
    ("facts", "verified_by",
     "ALTER TABLE facts ADD COLUMN verified_by INT NULL AFTER verified_at"),
    # Quien CORRIGIO. Distinto de `superseded_by`, que es el id del HECHO que
    # reemplaza. Los dos hacen falta: uno reconstruye la cadena de versiones, el
    # otro dice de quien fue la decision.
    ("facts", "superseded_by_user",
     "ALTER TABLE facts ADD COLUMN superseded_by_user INT NULL AFTER verified_by"),
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
    # Sirve a `SQL_LISTAR` de la pantalla de Memoria (jax-platform,
    # backend/api/admin/memoria.py): filtra por `is_verified` y `expires_at`
    # y ordena por `expires_at DESC, created_at DESC` -- esos tres, EN ESE
    # ORDEN, son exactamente las columnas de este indice compuesto.
    # Verificado con EXPLAIN contra jax_memory_test: sin filesort.
    #
    # CORREGIDO 2026-09-20 (auditoria adversarial): este comentario decia
    # "para que no aparezca Using filesort" sobre LA PANTALLA en general, y
    # eso es falso para `MemoryDB.get_facts()` (db.py): esa consulta ordena
    # por `COALESCE(importance, 0) DESC, created_at DESC`, y NINGUNA B-Tree
    # puede servir un COALESCE -- filesort ahi es inevitable, y este indice
    # no lo evita ni esta pensado para hacerlo. El indice no sobra (medido:
    # SQL_LISTAR si lo usa limpio); lo que estaba mal era describir la
    # consulta equivocada.
    ("facts", "idx_facts_revision",
     "CREATE INDEX idx_facts_revision ON facts (is_verified, expires_at, created_at)"),
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

# Ejecutor: sudo real y machine_id del inventario (2026-09-22). `ejecutor_host`
# la CREA y la puebla jax-platform (backend/db/migrations.py,
# `_ejecutor_inventario_v1`, desde JAX_EJECUTOR_INVENTARIO) -- esto NO duplica
# esa migracion ni toca jax-platform. Es solo un DATO que faltaba sobre una
# tabla que YA EXISTE: la Fase 3 le dio sudo real a `axioma` en las cuatro
# maquinas con clientes (`~/ejecutor-producto/LEDGER.md`, entradas "2026-09-22
# · FASE 3 (sudo) APLICADA..." y "SUDO EN LAS CUATRO"), pero
# `ejecutor_host.sudo`/`.machine_id` quedaron en 0/NULL -- el propio ledger lo
# deja escrito: "solo informativos, sin lector en el codigo; se corrigen por
# migracion en PR (regla del ledger), no a mano". Corre ACA porque LAS MANOS,
# el memory worker y la sintesis ya llaman a `ensure_schema()` en cada
# `connect()` contra la MISMA base (`jax/memory/db.py`) -- no hace falta una
# segunda migracion en jax-platform para un dato que ya tiene tabla y columnas.
# `ejecutor-prueba` (VM desechable de la Fase 2) NO entra: nunca tuvo sudo real.
_EJECUTOR_HOST_MACHINE_ID = {
    "atemai": "95e56bf6da0d41f993a3e36869699af1",
    "bridge": "ee090efa28cd46a7a0bff22d34e57eb4",
    "prod": "da476dce01ea4c3e9e72a8078a3ffd48",
    "hall9000": "37ce158242c649fa80804a8c17b83ca4",
}


#: Extrae la columna de referencia de un DDL con `AFTER <columna>` (case
#: insensible -- MariaDB no distingue mayusculas en palabras clave). Entradas
#: de `_COLUMNAS` sin `AFTER` (la columna va al final, la posicion no
#: importa) no matchean, y eso es el comportamiento correcto: nada que
#: reposicionar.
_AFTER_RE = re.compile(r"\bAFTER\s+(\w+)\b", re.IGNORECASE)


async def _reposicionar_si_hace_falta(cur, tabla: str, columna: str, ddl: str) -> None:
    """MIGRACION COMPENSATORIA (2026-09-20, auditoria adversarial M2).

    La columna YA EXISTE (si no, el llamador ya la agrego con `ddl` tal
    cual). Pudo haber quedado en la posicion equivocada porque una version
    anterior de este migrador la agrego SIN `AFTER` -- un `ALTER TABLE ADD
    COLUMN` sin eso la deja al FINAL de la tabla, mientras
    `jax_memory_schema.sql` (la fuente de verdad del checker de deriva) la
    declara en el MEDIO. El resultado medido 2026-09-20 contra
    `jax_memory_test`: `verified_by` en la posicion 20 y
    `superseded_by_user` en la 21, cuando el .sql las pone en la 10 y la 11
    -- y como `conftest.py`/`base_de_test.py` CLONAN el esquema de esa base
    para cada base de sesion nueva, la deriva se propaga a toda base que
    nazca a partir de ahi. Produccion se salvaba solo por cronologia
    (todavia no tenia las columnas cuando se escribio esto).

    Idempotente: si la columna ya esta en su lugar, el `MODIFY COLUMN` no
    corre. Si la columna de referencia (`AFTER <esa>`) tampoco existe
    todavia (base mas vieja que ni siquiera tiene `verified_at`), no hay
    nada que reposicionar EN ESTA CORRIDA -- eso es lo mismo que documenta
    `_agregar_columnas_de_facts` sobre `verified_at`: la siguiente corrida
    de `ensure_schema`, una vez que la referencia exista, lo resuelve."""
    referencia = _AFTER_RE.search(ddl)
    if not referencia:
        return
    columna_referencia = referencia.group(1)
    await cur.execute(
        "SELECT ORDINAL_POSITION FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND COLUMN_NAME=%s",
        (tabla, columna),
    )
    fila = await cur.fetchone()
    pos_actual = fila[0] if fila else None
    await cur.execute(
        "SELECT ORDINAL_POSITION FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND COLUMN_NAME=%s",
        (tabla, columna_referencia),
    )
    fila_ref = await cur.fetchone()
    pos_referencia = fila_ref[0] if fila_ref else None
    if pos_actual is None or pos_referencia is None or pos_actual == pos_referencia + 1:
        return
    # El DDL de ADD ya trae el tipo/nullability + el mismo `AFTER`: el
    # MODIFY es literalmente el mismo texto con el verbo cambiado. Dos
    # fuentes de verdad para "que tipo tiene esta columna" se desincronizan
    # solas (Regla Absoluta) -- por eso no se repite el tipo a mano aca.
    reposicionar = ddl.replace("ADD COLUMN", "MODIFY COLUMN", 1)
    await cur.execute(reposicionar)
    logger.info(
        "migracion: %s.%s reposicionada AFTER %s (estaba en la posicion "
        "%d, no en la %d)",
        tabla, columna, columna_referencia, pos_actual, pos_referencia + 1,
    )


async def ensure_schema(pool) -> bool:
    """Aplica lo que falte. Devuelve True si el esquema quedo al dia (todos
    los pasos, en TODAS las tablas, sin un solo error).

    Fail-soft y RUIDOSO: un error se registra como ERROR y el paso que fallo
    hace que el resultado final sea False, pero no lanza -- el arranque de
    JAX no depende de esto.

    CADA PASO va en su propio try/except (arreglado 2026-09-20, auditoria
    adversarial m2): antes los tres bucles (columnas, backfill, indices)
    vivian bajo un UNICO try/except de la funcion entera. Si una base
    todavia no tenia `verified_at` (una base mas vieja que la columna que
    esta migracion agrego en una ronda anterior), el `ALTER TABLE ... ADD
    COLUMN verified_by ... AFTER verified_at` de MAS ARRIBA fallaba con
    ERROR 1054 (columna desconocida) -- y esa excepcion escapaba del bucle
    de columnas y abortaba TAMBIEN el backfill y la creacion de indices, que
    no tienen nada que ver con `verified_at`. Con cada paso aislado, un
    fallo puntual se registra y el resto de la migracion sigue -- la
    proxima corrida de `ensure_schema()` (el proximo arranque) reintenta
    el paso que fallo.
    """
    ok = True
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                for tabla, columna, ddl in _COLUMNAS:
                    try:
                        await cur.execute(
                            "SELECT COUNT(*) FROM information_schema.COLUMNS "
                            "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND COLUMN_NAME=%s",
                            (tabla, columna),
                        )
                        if not (await cur.fetchone())[0]:
                            await cur.execute(ddl)
                            logger.info("migracion: %s.%s agregada", tabla, columna)
                        else:
                            await _reposicionar_si_hace_falta(cur, tabla, columna, ddl)
                    except Exception as e:  # fail-soft: una columna que no se pudo agregar no tiene que tumbar el backfill ni los indices de las demas tablas -- se registra el error, ok pasa a False (el caller no queda creyendo que el esquema esta al dia) y el resto de la migracion sigue
                        logger.error(
                            "migracion: %s.%s fallo (%s: %s) -- se sigue con el "
                            "resto de la migracion", tabla, columna, type(e).__name__, e)
                        ok = False

                for tabla, columna, ddl in _BACKFILL:
                    try:
                        await cur.execute(ddl)
                        if cur.rowcount:
                            logger.info("migracion: backfill de %s.%s en %d filas",
                                        tabla, columna, cur.rowcount)
                    except Exception as e:  # fail-soft: un backfill fallido en una tabla no tiene que impedir el backfill de las demas ni la creacion de indices -- se registra, ok pasa a False, y el proximo arranque reintenta este paso puntual
                        logger.error(
                            "migracion: backfill de %s.%s fallo (%s: %s) -- se sigue "
                            "con el resto de la migracion",
                            tabla, columna, type(e).__name__, e)
                        ok = False

                for nombre, machine_id in _EJECUTOR_HOST_MACHINE_ID.items():
                    try:
                        await cur.execute(
                            "UPDATE ejecutor_host SET sudo=1, machine_id=%s "
                            "WHERE nombre=%s AND (sudo=0 OR machine_id IS NULL OR machine_id<>%s)",
                            (machine_id, nombre, machine_id),
                        )
                        if cur.rowcount:
                            logger.info(
                                "migracion: ejecutor_host.%s sudo=1 machine_id=%s", nombre, machine_id)
                    except Exception as e:  # fail-soft: ejecutor_host puede no existir todavia (base minima sin las migraciones de jax-platform) -- se registra, ok pasa a False, y el proximo arranque reintenta este host puntual
                        logger.error(
                            "migracion: ejecutor_host.%s fallo (%s: %s) -- se sigue con el "
                            "resto de la migracion", nombre, type(e).__name__, e)
                        ok = False

                for tabla, indice, ddl in _INDICES:
                    try:
                        await cur.execute(
                            "SELECT COUNT(*) FROM information_schema.STATISTICS "
                            "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND INDEX_NAME=%s",
                            (tabla, indice),
                        )
                        if not (await cur.fetchone())[0]:
                            await cur.execute(ddl)
                            logger.info("migracion: indice %s creado", indice)
                    except Exception as e:  # fail-soft: un indice que no se pudo crear en una tabla no tiene que impedir los indices de las demas -- se registra, ok pasa a False, y el arranque de JAX no depende de que este indice exista hoy mismo
                        logger.error(
                            "migracion: indice %s (%s) fallo (%s: %s) -- se sigue "
                            "con el resto de la migracion",
                            indice, tabla, type(e).__name__, e)
                        ok = False
        return ok
    except Exception as e:  # fail-soft: falla de CONEXION (acquire/cursor), no de un paso puntual -- nada de lo de arriba corrio
        logger.error("migracion del esquema de memoria fallo: %s -- se sigue con "
                     "el esquema que haya", e)
        return False
