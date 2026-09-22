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
`jacobs_events` (revision final de la rama Descartar, 2026-09-22): mismo
mecanismo (`init_tables()`, `information_schema`), mismo job de CI (necesita
una base VACIA -- ver mas abajo por que el DROP), asi que va aca en vez de en
un archivo nuevo que requeriria su propio wireado en policy.yml.
`pipeline_transicion_descarte` (Ruling 9) inserta el evento de auditoria del
descarte en la MISMA transaccion que el CAS de estado; eso depende de que las
dos tablas sean transaccionales, y sin la clausula EXPLICITA esa garantia
depende de `default_storage_engine` del server, no del codigo.

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
from base_de_test import fijar_base_de_test  # noqa: E402

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

        `init_tables()` corre en CADA arranque de los tres procesos: si crear un
        indice no fuera idempotente, el segundo arranque fallaria -- y fallaria
        en produccion, no aca.
        """
        antes = {(t, c): await self._indice_de(t, c) for t, c in COLUMNAS_CONSULTADAS}
        await store.init_tables()
        despues = {(t, c): await self._indice_de(t, c) for t, c in COLUMNAS_CONSULTADAS}
        self.assertEqual(antes, despues, "init_tables() duplico o perdio indices al repetirse")


#: (tabla, constante de DDL) -- misma correspondencia que `init_tables()`.
_DDL_DE = (
    ("jacobs_pipelines", "_DDL_JACOBS_PIPELINES"),
    ("jacobs_steps", "_DDL_JACOBS_STEPS"),
    ("jacobs_events", "_DDL_JACOBS_EVENTS"),
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
    de deuda a medias."""

    async def asyncSetUp(self):
        self.addAsyncCleanup(store.cerrar_pool)

    async def _engine_de(self, cur, tabla: str) -> str | None:
        await cur.execute(
            "SELECT ENGINE FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s",
            (tabla,),
        )
        fila = await cur.fetchone()
        return fila[0] if fila else None

    async def test_jacobs_pipelines_steps_events_son_innodb_explicito(self):
        """No alcanza con dropear y recrear en ESTA base: su
        `default_storage_engine` ya es InnoDB (medido contra la MariaDB real
        de esta sesion), asi que un `CREATE TABLE` sin `ENGINE=` explicito
        DA InnoDB igual -- ese control no falla contra el codigo viejo, y un
        control que no falla no valida nada. `jax_user` tampoco tiene SUPER
        para un `SET GLOBAL default_storage_engine=...` que lo simule de
        verdad (probado a mano: 1227 Access denied).

        En cambio `init_command` en la conexion SI cambia el
        `default_storage_engine` de la SESION sin privilegios especiales
        (probado a mano), y el pool de aiomysql (`create_pool`) fija los
        argumentos de conexion UNA vez, al crear el pool, y los reusa para
        cada conexion nueva que abre despues -- asi que el parche tiene que
        estar activo ANTES de la primera `conexion()` de este test (loop
        nuevo por metodo con `IsolatedAsyncioTestCase`: este pool no existe
        todavia).

        Ejecuta el DDL de cada tabla (`store._DDL_JACOBS_*`, el MISMO texto
        que corre `init_tables()`) directo, sin pasar por el resto de
        `init_tables()`: esa funcion sigue con un loop de `ALTER TABLE ...
        ALGORITHM=INSTANT` sobre columnas de `jacobs_pipelines` (`visible`
        es GENERATED) que solo InnoDB soporta -- probado a mano: bajo esta
        misma sesion en MyISAM, ese ALTER revienta con
        `1845 ALGORITHM=INSTANT is not supported`, ANTES de llegar siquiera
        a crear `jacobs_steps`/`jacobs_events`. Correr el DDL nombrado
        aislado evita ese choque y deja probar el ENGINE de las tres, y
        el test SI falla contra el codigo viejo (medido: las tres dan
        MyISAM bajo esta sesion)."""
        cfg_original = store._db_cfg

        def _cfg_con_myisam() -> dict:
            cfg = cfg_original()
            cfg["init_command"] = "SET SESSION default_storage_engine='MyISAM'"
            return cfg

        faltantes = []
        with mock.patch.object(store, "_db_cfg", _cfg_con_myisam):
            async with store.conexion(desechable=True) as conn:
                async with conn.cursor() as cur:
                    for tabla, nombre_ddl in _DDL_DE:
                        await cur.execute(f"DROP TABLE IF EXISTS {tabla}")
                        await cur.execute(getattr(store, nombre_ddl))
                    for tabla, _ in _DDL_DE:
                        engine = await self._engine_de(cur, tabla)
                        if engine != "InnoDB":
                            faltantes.append(f"{tabla}={engine!r}")
        self.assertEqual(
            faltantes, [],
            f"sin ENGINE=InnoDB explicito: {faltantes} -- agregar ENGINE=InnoDB "
            "al CREATE TABLE correspondiente en jacobs/store.py (Ruling 9: "
            "pipeline_transicion_descarte depende de que sean transaccionales).",
        )


if __name__ == "__main__":
    unittest.main()
