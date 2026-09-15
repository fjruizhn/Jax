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

Corre con:
  PYTHONPATH=/home/fruiz/jax .venv/bin/python -m pytest jacobs/_store_indexes_test.py
"""
from __future__ import annotations

import os
import unittest

# Forzado, no setdefault: si el proceso ya sourceo /etc/jax/.env, JAX_DB_NAME
# apunta a produccion y este test crearia indices ahi sin pasar por el camino
# real. Mismo blindaje que _pipeline_identity_test.py.
os.environ["JAX_DB_NAME"] = "jax_memory_test"

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
        await store.init_tables()

    async def _indice_de(self, tabla: str, columna: str) -> int:
        conn = await store.get_conn()
        try:
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
        finally:
            conn.close()

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
        conn = await store.get_conn()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COLUMN_NAME FROM information_schema.STATISTICS "
                    "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND INDEX_NAME=%s "
                    "ORDER BY SEQ_IN_INDEX",
                    (tabla, indice),
                )
                return [r[0] for r in await cur.fetchall()]
        finally:
            conn.close()

    async def test_indice_de_duenio_de_jacobs_pipelines(self):
        """Ruling T6-6 (2026-09-15): jax-platform lista los pipelines de un
        dueño filtrando por (user_id, tenant_id) y ordenando por created_at.
        Compuesto y en ESE orden: el filtro de igualdad primero y el ORDER BY
        al final es lo que evita el filesort.

        Se BORRA primero y se vuelve a correr init_tables(): medido el
        2026-09-15, la jax_memory_test local ya lo tenia (lo creo otro camino,
        no init_tables), y el test pasaba sin el cambio -- un control que no
        falla no valida. Solo toca la base forzada arriba (jax_memory_test)."""
        conn = await store.get_conn()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DROP INDEX IF EXISTS idx_jacobs_pipelines_duenio ON jacobs_pipelines")
        finally:
            conn.close()
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


if __name__ == "__main__":
    unittest.main()
