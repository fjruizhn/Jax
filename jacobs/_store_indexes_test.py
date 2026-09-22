#!/usr/bin/env python3
# jax/jacobs/_store_indexes_test.py
"""Indices de las tablas de Jacobs: toda columna por la que se filtra tiene
indice, y lo crea `init_tables()` -- no un ALTER a mano.

POR QUE EXISTE. Medido el 2026-09-11: `jacobs_events`, `jacobs_steps` y
`jacobs_pipelines` tenian UN solo indice cada una (la PK), y el codigo las
consulta por `pipeline_id` (3 sitios) y por `status` (2). Con 611 filas no se
nota; el plan es un scan igual, y crece lineal. Es la politica 1 de LAS CUATRO
DEL RENDIMIENTO: la verificacion no es "hoy es rapido", es el plan.

POR QUE POR `init_tables()` Y NO POR UN `ALTER TABLE` A MANO. Este repo no
tiene carpeta `migrations/`: el mecanismo real es `init_tables()`, idempotente,
chequeando `information_schema` antes de cada DDL. Un ALTER manual sin
registrar es la deuda que ya se pago una vez con la columna `depends_on`, que
existia en produccion y en ninguna base nueva (ver DEUDA.md). Un indice puesto
a mano en `jax_memory` tendria exactamente la misma forma: invisible para un
dev nuevo, para CI y para un restore de desastre.

TAMBIEN cubre `ENGINE=InnoDB` de `jacobs_pipelines`/`jacobs_steps`/
`jacobs_events` (revision final de la rama Descartar, 2026-09-22; corregido
en la ronda 1 de PR#261): mismo mecanismo (`init_tables()`,
`information_schema`), mismo job de CI, asi que va aca en vez de en un
archivo nuevo que requeriria su propio wireado en policy.yml.
`pipeline_transicion_descarte` (Ruling 9) inserta el evento de auditoria del
descarte en la MISMA transaccion que el CAS de estado; eso depende de que las
dos tablas sean transaccionales, y sin la clausula EXPLICITA esa garantia
depende de `default_storage_engine` del server, no del codigo. A diferencia
de `StoreIndexesTest` (arriba), que SI necesita la base compartida real
(prueba el indice de `jacobs_pipelines`), `EngineInnoDBTest` no toca esa
tabla en absoluto: corre el DDL contra nombres de tabla DESCARTABLES propios
(`jacobs_engine_probe_*`) -- MAJOR-3 de la ronda 1 de PR#261: la version
anterior DROPeaba `jacobs_pipelines`/`jacobs_steps`/`jacobs_events` de la
base COMPARTIDA (`jax_memory_test` sin `JAX_TEST_DB_SUFIJO`) y las recreaba
solo con el DDL base, sin las columnas de los ALTER ni los indices de
`_INDICES` ni filas -- la clase de incidente R38: otra sesion en la misma
base pierde sus filas y su esquema a mitad de camino. Nombres propios evita
el problema de raiz, no lo repara despues: no hace falta ninguna base vacia
ni el permiso de destruir nada, alcanza con poder crear dos tablas chicas
descartables y dropearlas al terminar.

Corre con:
  PYTHONPATH=/home/fruiz/jax .venv/bin/python -m pytest jacobs/_store_indexes_test.py
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

# Forzado, no setdefault: si el proceso ya sourceo /etc/jax/.env, JAX_DB_NAME
# apunta a produccion y este test crearia indices ahi sin pasar por el camino
# real. Mismo blindaje que _pipeline_identity_test.py.
from base_de_test import VARIABLE_DEL_SUFIJO, fijar_base_de_test  # noqa: E402

fijar_base_de_test()

from jacobs import store

# (tabla, columna) por las que el codigo REALMENTE filtra. Cada entrada sale de
# un grep sobre las consultas, no de una intuicion sobre el modelo.
COLUMNAS_CONSULTADAS = [
    ("jacobs_events", "pipeline_id"),   # store.py: SELECT ... WHERE pipeline_id=%s
    ("jacobs_steps", "pipeline_id"),    # store.py: pasos de un pipeline
    ("jacobs_steps", "status"),         # reaper: pasos colgados por estado
    ("jacobs_pipelines", "status"),     # reaper/listados por estado
]


class StoreIndexesTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Pool por loop: cada test trae el suyo y lo cierra (jacobs/store.py).
        self.addAsyncCleanup(store.cerrar_pool)
        await store.init_tables()

    async def _indice_de(self, tabla: str, columna: str) -> int:
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                # SEQ_IN_INDEX=1: la columna tiene que ser la PRIMERA del
                # indice. Ser la segunda de un compuesto no sirve para filtrar
                # solo por ella -- un indice que no se puede usar cuenta como
                # no tenerlo, que es justo el error que esta ronda encontro en
                # el indice vectorial de `messages`.
                await cur.execute(
                    "SELECT COUNT(*) FROM information_schema.STATISTICS "
                    "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s "
                    "AND COLUMN_NAME=%s AND SEQ_IN_INDEX=1",
                    (tabla, columna),
                )
                (n,) = await cur.fetchone()
                return n

    async def test_toda_columna_consultada_tiene_indice_que_la_encabeza(self):
        faltantes = []
        for tabla, columna in COLUMNAS_CONSULTADAS:
            if await self._indice_de(tabla, columna) == 0:
                faltantes.append(f"{tabla}.{columna}")
        self.assertEqual(
            faltantes, [],
            f"sin indice que las encabece: {faltantes}. El codigo filtra por esas "
            "columnas, asi que el plan es un scan. Agregarlas a init_tables() "
            "-- no con un ALTER a mano, que no llega a una base nueva.",
        )

    async def _columnas_del_indice(self, tabla: str, indice: str) -> list[str]:
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COLUMN_NAME FROM information_schema.STATISTICS "
                    "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND INDEX_NAME=%s "
                    "ORDER BY SEQ_IN_INDEX",
                    (tabla, indice),
                )
                return [r[0] for r in await cur.fetchall()]

    async def test_indice_de_duenio_de_jacobs_pipelines(self):
        """Ruling T6-6 (2026-09-15): jax-platform lista los pipelines de un
        dueño filtrando por (user_id, tenant_id) y ordenando por created_at.
        Compuesto y en ESE orden: el filtro de igualdad primero y el ORDER BY
        al final es lo que evita el filesort.

        Se BORRA primero y se vuelve a correr init_tables(): medido el
        2026-09-15, la jax_memory_test local ya lo tenia (lo creo otro camino,
        no init_tables), y el test pasaba sin el cambio -- un control que no
        falla no valida. Solo toca la base forzada arriba (jax_memory_test)."""
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DROP INDEX IF EXISTS idx_jacobs_pipelines_duenio ON jacobs_pipelines")
        await store.init_tables()
        self.assertEqual(
            await self._columnas_del_indice("jacobs_pipelines", "idx_jacobs_pipelines_duenio"),
            ["user_id", "tenant_id", "created_at"],
            "falta idx_jacobs_pipelines_duenio (user_id, tenant_id, created_at) -- "
            "agregarlo a la lista idempotente de init_tables()",
        )

    async def test_init_tables_es_idempotente_para_los_indices(self):
        """Correrlo dos veces no duplica indices ni revienta.

        `init_tables()` corre en CADA arranque de LAS MANOS -- el unico
        proceso de produccion que lo llama (verificado 2026-09-22 contra el
        codigo; jax-platform y el Ejecutor NO lo hacen) -- y ademas de
        scripts de este repo o de la suite de tests contra la misma base: si
        crear un indice no fuera idempotente, el segundo arranque fallaria
        -- y fallaria en produccion, no aca.
        """
        antes = {(t, c): await self._indice_de(t, c) for t, c in COLUMNAS_CONSULTADAS}
        await store.init_tables()
        despues = {(t, c): await self._indice_de(t, c) for t, c in COLUMNAS_CONSULTADAS}
        self.assertEqual(antes, despues, "init_tables() duplico o perdio indices al repetirse")


#: Sufijo de ESTE proceso para los nombres de tabla descartables -- MINOR-2
#: de la ronda 2 de PR#261: los nombres eran constantes fijas, y dos
#: corridas concurrentes contra la MISMA `jax_memory_test` (compartida --
#: por ejemplo dentro de un job de CI que no define `JAX_TEST_DB_SUFIJO`, o
#: dos sesiones locales que por lo que sea resuelven a la misma base) se
#: pisarian sobre el MISMO nombre de tabla de prueba: un rojo espantoso
#: (`Table ... already exists` a mitad de un `CREATE`, o un `DROP` de la
#: sesion ajena) que no es un defecto del codigo bajo prueba, es una carrera
#: de este test consigo mismo. Usa el sufijo de la SESION
#: (`JAX_TEST_DB_SUFIJO`, ya fijado arriba por `fijar_base_de_test()` --
#: el MISMO mecanismo que aisla la base entera por sesion, ver
#: `base_de_test.py`) si esta puesto; si no (el caso de un job de CI, que
#: corre en su propio contenedor MariaDB efimero y por eso no lo necesita),
#: el PID de este proceso.
_SUFIJO_DE_PRUEBA = os.environ.get(VARIABLE_DEL_SUFIJO) or str(os.getpid())


def _con_nombre_de_prueba(ddl: str, tabla: str, tabla_prueba: str) -> str:
    """Devuelve `ddl` (el texto de `store._DDL_JACOBS_*`) con `tabla`
    renombrada a `tabla_prueba` en el header del `CREATE TABLE`. Revienta si
    el reemplazo NO tuvo efecto -- MINOR-1 de la ronda 2 de PR#261:
    `str.replace` es SILENCIOSO si el patron `f"EXISTS {tabla} ("` no
    aparece (alguien reformatea el DDL con backticks alrededor del nombre,
    o mete un salto de linea antes del parentesis). Sin este chequeo, el
    DDL correria con el nombre REAL: contra una base vacia crearia
    `jacobs_pipelines`/`jacobs_steps`/`jacobs_events` de verdad, con solo
    las columnas base del DDL (sin las de los ALTER ni los indices de
    `_INDICES`), y la limpieza -- que solo dropea los nombres DE PRUEBA --
    las dejaria asi para siempre."""
    marca = f"EXISTS {tabla} ("
    reemplazo = ddl.replace(marca, f"EXISTS {tabla_prueba} (", 1)
    if reemplazo == ddl:
        raise AssertionError(
            f"el DDL de {tabla!r} no contiene {marca!r} -- el reemplazo al "
            f"nombre de prueba ({tabla_prueba!r}) no tuvo efecto, y correr "
            "este DDL tal cual crearia la tabla REAL. Revisar el formato de "
            "la constante _DDL_* correspondiente en jacobs/store.py."
        )
    return reemplazo


#: (tabla real, constante de DDL, nombre DESCARTABLE bajo el que se prueba).
#: MAJOR-3 (revision 1, PR#261): antes se corria el DDL bajo el nombre REAL
#: (con un DROP TABLE de la tabla compartida antes) -- el nombre descartable
#: es lo que evita tocar `jacobs_pipelines`/`jacobs_steps`/`jacobs_events` de
#: la sesion. El DDL en si es el MISMO texto que corre `init_tables()`
#: (`store._DDL_JACOBS_*`); solo cambia el nombre de la tabla que declara.
#: El sufijo (`_SUFIJO_DE_PRUEBA`) es MINOR-2 de la ronda 2: sin el, dos
#: procesos concurrentes se pisarian sobre el mismo nombre.
_DDL_DE = (
    ("jacobs_pipelines", "_DDL_JACOBS_PIPELINES",
     f"jacobs_engine_probe_pipelines_{_SUFIJO_DE_PRUEBA}"),
    ("jacobs_steps", "_DDL_JACOBS_STEPS",
     f"jacobs_engine_probe_steps_{_SUFIJO_DE_PRUEBA}"),
    ("jacobs_events", "_DDL_JACOBS_EVENTS",
     f"jacobs_engine_probe_events_{_SUFIJO_DE_PRUEBA}"),
)


class EngineInnoDBTest(unittest.IsolatedAsyncioTestCase):
    """Ruling 9 del ledger de descartar-pipelines (revision final, 2026-09-22):
    `pipeline_transicion_descarte` escribe el CAS de estado y el evento de
    auditoria en la MISMA transaccion (`jacobs/store.py`). Eso depende de que
    `jacobs_pipelines` y `jacobs_events` sean transaccionales -- sin
    `ENGINE=InnoDB` EXPLICITO en el `CREATE TABLE`, la garantia la da el
    `default_storage_engine` del server, no el codigo: en un server que no
    lo traiga en InnoDB, un evento fallido ya no revertiria el CAS (o al
    reves), reabriendo en silencio el hueco que Ruling 9 cerro. `jacobs_steps`
    tiene el mismo hueco (mismo archivo, mismo patron) aunque nada la use
    todavia en una transaccion multi-tabla; se corrige junto para no dejarla
    de deuda a medias.

    MAJOR-3 (revision 1, PR#261): la version anterior de este test DROPeaba
    `jacobs_pipelines`/`jacobs_steps`/`jacobs_events` de la base -- que, sin
    `JAX_TEST_DB_SUFIJO`, es la `jax_memory_test` COMPARTIDA -- y las
    recreaba solo con el DDL base, sin las columnas de los ALTER (`user_id`,
    `tenant_id`, la `visible` GENERATED, las columnas del descarte,
    `run_epoch`, `owner_ack_at`...), sin ningun indice de `_INDICES` y sin
    filas. Otra sesion corriendo contra la misma base en ese momento perdia
    sus filas y podia pegar contra un 1054 (columna desconocida) en su
    proxima consulta -- la misma clase de incidente que R38, y el detector
    de DELETE de `tests/test_delete_de_tablas_compartidas.py` no cubre DROP.

    Se decidio por nombres de tabla DESCARTABLES (`jacobs_engine_probe_*`) en
    vez de restaurar despues (`self.addAsyncCleanup(store.init_tables)`)
    porque restaurar despues sigue dejando una VENTANA en la que la tabla
    compartida esta incompleta mientras el test corre, y si el proceso muere
    a mitad (timeout, OOM, Ctrl-C) el cleanup nunca corre y la base queda
    rota para todos. Con nombres propios, `jacobs_pipelines`/`jacobs_steps`/
    `jacobs_events` de la sesion NUNCA se tocan -- no hay ventana que cerrar
    porque no se abre ninguna. No hace falta una base vacia ni permiso para
    destruir nada: alcanza con poder crear y dropear tres tablas chicas
    propias, en cualquier base (compartida o no)."""

    async def asyncSetUp(self):
        self.addAsyncCleanup(store.cerrar_pool)
        self.addAsyncCleanup(self._dropear_tablas_de_prueba)

    async def _dropear_tablas_de_prueba(self) -> None:
        async with store.conexion(desechable=True) as conn:
            async with conn.cursor() as cur:
                for _, _, tabla_prueba in _DDL_DE:
                    await cur.execute(f"DROP TABLE IF EXISTS {tabla_prueba}")

    async def _engine_de(self, cur, tabla: str) -> str | None:
        await cur.execute(
            "SELECT ENGINE FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s",
            (tabla,),
        )
        fila = await cur.fetchone()
        return fila[0] if fila else None

    async def _crear_bajo_myisam_forzado(self) -> list[str]:
        """Ejecuta el DDL de cada tabla bajo una sesion cuyo
        `default_storage_engine` esta forzado a MyISAM, y devuelve las que
        NO dieron InnoDB.

        No alcanza con dropear y recrear bajo el `default_storage_engine`
        real de esta base: ya es InnoDB (medido contra la MariaDB real de
        esta sesion), asi que un `CREATE TABLE` sin `ENGINE=` explicito DA
        InnoDB igual -- ese control no falla contra el codigo viejo, y un
        control que no falla no valida nada. `jax_user` tampoco tiene SUPER
        para un `SET GLOBAL default_storage_engine=...` que lo simule de
        verdad (probado a mano: 1227 Access denied).

        En cambio `init_command` en la conexion SI cambia el
        `default_storage_engine` de la SESION sin privilegios especiales
        (probado a mano), y el pool de aiomysql (`create_pool`) fija los
        argumentos de conexion UNA vez, al crear el pool, y los reusa para
        cada conexion nueva que abre despues -- asi que el parche tiene que
        estar activo ANTES de la primera `conexion()` de este test.

        MAJOR-2 (revision 1, PR#261): eso significa que si el pool de ESTE
        loop YA existe cuando se llega aca -- por ejemplo porque alguien
        agrega `await store.init_tables()` a `asyncSetUp`, copiando el
        patron de `StoreIndexesTest`, o la suite pasa a un loop de sesion
        compartido en vez de uno nuevo por metodo -- `init_command` NUNCA
        corre, la sesion sigue en el `default_storage_engine` real del
        server (hoy InnoDB), y las tres tablas darian InnoDB aunque el
        codigo NO tuviera `ENGINE=InnoDB` -- el test pasaria sin haber
        probado nada. Por eso, antes de crear ninguna tabla, se AUTO-VERIFICA
        con un `SELECT` que la sesion realmente quedo en MyISAM; si no,
        revienta con un mensaje que dice por que, en vez de dejar pasar un
        test que no probo nada. `test_selfcheck_detecta_pool_creado_antes_del_parche`
        (abajo) es la mutacion que ejercita justo este camino.

        Corre el DDL de cada tabla bajo su nombre DESCARTABLE
        (`jacobs_engine_probe_*`, ver `_DDL_DE`), no el real: correr
        `init_tables()` completo bajo MyISAM ademas revienta ANTES de
        siquiera llegar a crear `jacobs_steps`/`jacobs_events`, en el loop de
        `ALTER TABLE ... ALGORITHM=INSTANT` sobre `jacobs_pipelines`
        (`visible` es GENERATED) -- probado a mano: bajo esta misma sesion en
        MyISAM, ese ALTER da `1845 ALGORITHM=INSTANT is not supported`, que
        solo InnoDB soporta. Correr el DDL nombrado aislado evita ese choque
        y deja probar el ENGINE de las tres."""
        cfg_original = store._db_cfg

        def _cfg_con_myisam() -> dict:
            cfg = cfg_original()
            cfg["init_command"] = "SET SESSION default_storage_engine='MyISAM'"
            return cfg

        faltantes: list[str] = []
        with mock.patch.object(store, "_db_cfg", _cfg_con_myisam):
            async with store.conexion(desechable=True) as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT @@SESSION.default_storage_engine")
                    (motor_de_sesion,) = await cur.fetchone()
                    if motor_de_sesion != "MyISAM":
                        raise AssertionError(
                            "la simulacion no aplico: "
                            f"@@SESSION.default_storage_engine={motor_de_sesion!r}, "
                            "esperaba 'MyISAM'. El pool de este loop ya existia con "
                            "la conexion REAL antes de este parche (por ejemplo, "
                            "otro asyncSetUp llamo a store.init_tables() primero): "
                            "init_command solo se aplica al CREAR una conexion "
                            "nueva del pool. Sin este chequeo, el test de "
                            "ENGINE=InnoDB pasaria igual aunque el codigo no lo diga."
                        )
                    for tabla, nombre_ddl, tabla_prueba in _DDL_DE:
                        ddl = _con_nombre_de_prueba(
                            getattr(store, nombre_ddl), tabla, tabla_prueba)
                        await cur.execute(f"DROP TABLE IF EXISTS {tabla_prueba}")
                        await cur.execute(ddl)
                    for tabla, _, tabla_prueba in _DDL_DE:
                        engine = await self._engine_de(cur, tabla_prueba)
                        if engine != "InnoDB":
                            faltantes.append(f"{tabla}={engine!r}")
        return faltantes

    async def test_jacobs_pipelines_steps_events_son_innodb_explicito(self):
        """El test SI falla contra el codigo viejo (medido: las tres dan
        MyISAM bajo esta sesion forzada) -- ver `_crear_bajo_myisam_forzado`
        para como se fuerza la simulacion."""
        faltantes = await self._crear_bajo_myisam_forzado()
        self.assertEqual(
            faltantes, [],
            f"sin ENGINE=InnoDB explicito: {faltantes} -- agregar ENGINE=InnoDB "
            "al CREATE TABLE correspondiente en jacobs/store.py (Ruling 9: "
            "pipeline_transicion_descarte depende de que sean transaccionales).",
        )

    async def test_selfcheck_detecta_pool_creado_antes_del_parche(self):
        """Mutacion pedida en la revision (MAJOR-2, PR#261 ronda 1): crea el
        pool de ESTE loop con la conexion REAL (la misma trampa que copiar
        `asyncSetUp` de `StoreIndexesTest`, que llama a `store.init_tables()`
        ahi) ANTES de que `_crear_bajo_myisam_forzado` parchee `_db_cfg`.
        `store.init_tables()` en si es seguro contra la base compartida --
        `CREATE TABLE IF NOT EXISTS` + ALTERs idempotentes, lo mismo que hace
        `StoreIndexesTest.asyncSetUp` en cada corrida -- lo que se prueba
        aca es que, con el pool ya creado, la simulacion de MyISAM NO aplica
        y el auto-chequeo lo detecta y revienta con un mensaje claro, en vez
        de dejar pasar un test que no probo nada."""
        await store.init_tables()  # crea el pool de este loop con la cfg REAL
        with self.assertRaisesRegex(AssertionError, "la simulacion no aplico"):
            await self._crear_bajo_myisam_forzado()


class NombreDePruebaTest(unittest.TestCase):
    """Puras, sin DB -- prueban `_con_nombre_de_prueba` y `_DDL_DE` como
    datos, mismo criterio que `FormaDelDDLTest` de
    `tests/test_store_indice_duenio.py`."""

    def test_revienta_si_el_patron_no_aparece(self):
        """MINOR-1 (ronda 2, PR#261): si el DDL no tiene el header
        `EXISTS <tabla> (` tal cual -- por ejemplo porque alguien le puso
        backticks al nombre, o un salto de linea antes del parentesis --
        `str.replace` no hace nada y devolveria el DDL SIN CAMBIAR. Este
        test prueba que en cambio revienta."""
        ddl_reformateado = (
            "CREATE TABLE IF NOT EXISTS `jacobs_pipelines` (\n"
            "    pipeline_id VARCHAR(36) PRIMARY KEY\n"
            ") ENGINE=InnoDB"
        )
        with self.assertRaises(AssertionError):
            _con_nombre_de_prueba(
                ddl_reformateado, "jacobs_pipelines", "jacobs_engine_probe_pipelines_x")

    def test_reemplaza_cuando_el_patron_aparece(self):
        ddl = "CREATE TABLE IF NOT EXISTS jacobs_pipelines (\n    x INT\n) ENGINE=InnoDB"
        nuevo = _con_nombre_de_prueba(ddl, "jacobs_pipelines", "jacobs_engine_probe_pipelines_x")
        self.assertIn("EXISTS jacobs_engine_probe_pipelines_x (", nuevo)
        self.assertNotIn("EXISTS jacobs_pipelines (", nuevo)

    def test_los_tres_nombres_de_prueba_llevan_el_sufijo_de_esta_sesion(self):
        """MINOR-2 (ronda 2, PR#261): si alguien vuelve a hardcodear un
        nombre sin sufijo (regresion), esto lo detecta sin necesitar DB. La
        limpieza (`EngineInnoDBTest._dropear_tablas_de_prueba`) itera este
        MISMO `_DDL_DE` -- garantizar que los tres nombres llevan el sufijo
        alcanza para garantizar que la limpieza dropea exactamente lo que
        se creo, con el mismo sufijo, en la misma corrida."""
        for _, _, tabla_prueba in _DDL_DE:
            self.assertTrue(
                tabla_prueba.endswith(_SUFIJO_DE_PRUEBA),
                f"{tabla_prueba!r} no lleva el sufijo de esta sesion "
                f"({_SUFIJO_DE_PRUEBA!r}) -- dos corridas concurrentes "
                "contra la misma base compartida se pisarian sobre el "
                "mismo nombre de tabla de prueba.",
            )


if __name__ == "__main__":
    unittest.main()
