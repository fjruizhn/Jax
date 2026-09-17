"""Human gate de LAS MANOS contra una MariaDB REAL (2026-09-17): esquema,
emisión con solo el sha256, consumo de un solo uso, carrera y plan.

Seguridad: con JAX_DB_NAME apuntando a una base que no sea de tests
(típico después de sourcear /etc/jax/.env), el módulo se niega a importar.

Corre con:
  set -a; . /etc/jax/.env; set +a; export JAX_DB_NAME=jax_memory_test
  PYTHONPATH=.:las_manos python -m pytest -v las_manos/_human_gate_io_test.py
"""
from __future__ import annotations

import os

from base_de_test import exigir_base_de_test  # noqa: E402

exigir_base_de_test()

import asyncio  # noqa: E402
import time  # noqa: E402
import unittest  # noqa: E402

import aiomysql  # noqa: E402

from jacobs import store  # noqa: E402
import human_gate as hg  # noqa: E402

TABLA = "las_manos_human_gate_tokens"
COLUMNAS = ["token_hash", "emitido_por", "emitido_at", "vence_at", "usado_at", "usado_en"]


async def _una_fila(sql: str, args: tuple = ()) -> dict | None:
    async with store.conexion() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(sql, args)
            return await cur.fetchone()


async def _ejecutar(sql: str, args: tuple = ()) -> None:
    async with store.conexion() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, args)


@unittest.skipUnless(os.getenv("JAX_DB_HOST"), "necesita la MariaDB real (jax_memory_test)")
class _ConBase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await store.init_tables()
        self.hashes: list[str] = []

    async def asyncTearDown(self):
        for token_hash in self.hashes:
            await _ejecutar(
                f"DELETE FROM {TABLA} WHERE token_hash = %s",
                # marcador-propio: el hash de un token que emitió ESTE test
                # (self.hashes); no toca filas de nadie más.
                (token_hash,),
            )
        await store.cerrar_pool()

    async def emitir(self, ttl: int = 300) -> str:
        emitido = await hg.emitir_token_gate("test-io", ttl)
        self.hashes.append(hg.hash_token(emitido.token))
        return emitido.token


class EsquemaTest(_ConBase):
    async def test_init_tables_crea_la_tabla_sin_columna_para_el_token(self):
        # Se BORRA y se recrea: si ya existía, el test pasaría sin init_tables().
        await _ejecutar(f"DROP TABLE IF EXISTS {TABLA}")
        await store.init_tables()
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COLUMN_NAME FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() "
                    "AND TABLE_NAME=%s ORDER BY ORDINAL_POSITION", (TABLA,))
                self.assertEqual([r[0] for r in await cur.fetchall()], COLUMNAS)

    async def test_init_tables_es_idempotente(self):
        await store.init_tables()
        await store.init_tables()


class EmisionTest(_ConBase):
    async def test_guarda_solo_el_hash(self):
        token = await self.emitir()
        fila = await _una_fila(f"SELECT * FROM {TABLA} WHERE token_hash = %s", (hg.hash_token(token),))
        self.assertIsNotNone(fila)
        self.assertNotIn(token, [str(v) for v in fila.values()])
        self.assertIsNone(await _una_fila(f"SELECT 1 FROM {TABLA} WHERE token_hash = %s", (token,)))
        self.assertEqual(fila["emitido_por"], "test-io")
        self.assertIsNone(fila["usado_at"])


class ConsumoTest(_ConBase):
    async def test_un_solo_uso(self):
        token = await self.emitir()
        primero = await hg.consumir_token_gate(token, "execute:r1")
        segundo = await hg.consumir_token_gate(token, "execute:r2")
        self.assertEqual((primero.aceptado, primero.motivo), (True, None))
        self.assertEqual((segundo.aceptado, segundo.motivo), (False, hg.Motivo.TOKEN_USADO))
        fila = await _una_fila(f"SELECT usado_en FROM {TABLA} WHERE token_hash = %s",
                               (hg.hash_token(token),))
        self.assertEqual(fila["usado_en"], "execute:r1")

    async def test_token_inventado(self):
        r = await hg.consumir_token_gate("inventado-" + str(time.time()), "execute:r1")
        self.assertEqual((r.aceptado, r.motivo), (False, hg.Motivo.TOKEN_DESCONOCIDO))

    async def test_token_vencido(self):
        token = await self.emitir()
        await _ejecutar(f"UPDATE {TABLA} SET vence_at = %s WHERE token_hash = %s",
                        (time.time() - 1, hg.hash_token(token)))
        r = await hg.consumir_token_gate(token, "execute:r1")
        self.assertEqual((r.aceptado, r.motivo), (False, hg.Motivo.TOKEN_VENCIDO))

    async def test_veinte_consumos_a_la_vez_uno_solo_gana(self):
        token = await self.emitir()
        resultados = await asyncio.gather(
            *(hg.consumir_token_gate(token, f"execute:r{i}") for i in range(20)))
        self.assertEqual(sum(r.aceptado for r in resultados), 1)
        self.assertEqual({r.motivo for r in resultados if not r.aceptado}, {hg.Motivo.TOKEN_USADO})

    async def test_el_consumo_va_por_la_pk(self):
        token = await self.emitir()
        ahora = time.time()
        async with store.conexion() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # La SQL REAL (la constante que ejecuta el store), no una copia.
                await cur.execute("EXPLAIN " + store.SQL_CONSUMIR_TOKEN_GATE,
                                  (ahora, "explain", hg.hash_token(token), ahora))
                (plan,) = await cur.fetchall()
                self.assertEqual(plan["key"], "PRIMARY", plan)
                await cur.execute("EXPLAIN " + store.SQL_DIAGNOSTICO_TOKEN_GATE, (hg.hash_token(token),))
                (plan,) = await cur.fetchall()
                self.assertIn(plan["type"], {"const", "eq_ref"}, plan)


if __name__ == "__main__":
    unittest.main()
