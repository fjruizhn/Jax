#!/usr/bin/env python3
"""La memoria no puede dar por buena una escritura que fallo.

Tres defectos encontrados en la auditoria P10 del 2026-09-16, los tres de la
misma familia: **el llamador no miraba lo que la capa de datos devolvia**.

1. `connect()` llamaba a `ensure_schema()` y DESCARTABA el booleano. Como esa
   funcion aplica DDL + backfill sin transaccion, un fallo a mitad deja la base
   en un estado parcial -- y todo JAX arrancaba creyendo que el esquema estaba
   al dia, con una sola linea de log que nadie mira.

2. `_find_nearest_fact()` devolvia None tanto cuando NO HAY candidato como
   cuando la consulta FALLABA. Con `is_correction=True` eso hacia que nunca se
   entrara al bloque de correccion: el hecho erroneo seguia activo y el nuevo
   entraba al lado.

3. `save_fact()` llamaba a `supersede_fact()` **sin comprobar el retorno** y
   despues escribia "fact N corrige a fact M". Como supersede_fact esta
   decorado con db_error_handler, cualquier error devuelve None en silencio: el
   log afirmaba una correccion que no ocurrio.

Sin base de datos a proposito: son fallos de logica, y /etc/jax/.env apunta a
PRODUCCION fuera de pytest.
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jax.memory import db as dbmod  # noqa: E402


def _pool_falso(last_insert_id: int = 42):
    """Un pool que acepta el INSERT y devuelve un id."""
    cur = mock.AsyncMock()
    cur.fetchone = mock.AsyncMock(return_value=(last_insert_id,))
    cur.execute = mock.AsyncMock()
    ctx_cur = mock.MagicMock()
    ctx_cur.__aenter__ = mock.AsyncMock(return_value=cur)
    ctx_cur.__aexit__ = mock.AsyncMock(return_value=False)
    conn = mock.MagicMock()
    conn.cursor = mock.MagicMock(return_value=ctx_cur)
    ctx_conn = mock.MagicMock()
    ctx_conn.__aenter__ = mock.AsyncMock(return_value=conn)
    ctx_conn.__aexit__ = mock.AsyncMock(return_value=False)
    pool = mock.MagicMock()
    pool.acquire = mock.MagicMock(return_value=ctx_conn)
    return pool


class EsquemaTest(unittest.TestCase):
    def test_una_migracion_fallida_deja_la_base_como_NO_sana(self):
        m = dbmod.MemoryDB()
        m.pool = _pool_falso()
        m.schema_ok = False
        self.assertIs(asyncio.run(m.health_check()), False,
                      "una base con el esquema a medias se reporto como sana")

    def test_sin_fallo_de_migracion_el_health_mira_la_base(self):
        """Control del control: con el esquema al dia, health_check sigue
        consultando la base como siempre."""
        m = dbmod.MemoryDB()
        m.pool = _pool_falso()
        m.schema_ok = True
        self.assertIs(asyncio.run(m.health_check()), True)

    def test_el_atributo_existe_antes_de_conectar(self):
        self.assertIsNone(dbmod.MemoryDB().schema_ok)


class BusquedaDeFactTest(unittest.TestCase):
    def test_una_busqueda_fallida_lanza_en_vez_de_parecer_sin_candidato(self):
        m = dbmod.MemoryDB()
        m.pool = mock.MagicMock()
        m.pool.acquire = mock.MagicMock(side_effect=OSError("la base no responde"))
        with self.assertRaises(dbmod.BusquedaDeFactFallida):
            asyncio.run(m._find_nearest_fact([0.1, 0.2], 1, None))


class CorreccionTest(unittest.IsolatedAsyncioTestCase):
    async def test_si_no_se_puede_buscar_el_fact_a_corregir_NO_se_inserta(self):
        """Insertar sin poder supersedar deja dos hechos contradictorios."""
        m = dbmod.MemoryDB()
        m.pool = _pool_falso()
        m.get_embedding = mock.AsyncMock(return_value=[0.1, 0.2])
        m._find_nearest_fact = mock.AsyncMock(
            side_effect=dbmod.BusquedaDeFactFallida("la base no responde"))
        resultado = await m.save_fact.__wrapped__(
            m, "Fernando vive en Tegucigalpa", "user", is_correction=True)
        self.assertIsNone(resultado, "se inserto una correccion sin poder mirar que corregia")

    async def test_un_fact_normal_si_se_inserta_aunque_no_se_pueda_deduplicar(self):
        """Control: sin correccion, el unico riesgo es duplicar -- se sigue,
        pero el log lo dice."""
        m = dbmod.MemoryDB()
        m.pool = _pool_falso()
        m.get_embedding = mock.AsyncMock(return_value=[0.1, 0.2])
        m._find_nearest_fact = mock.AsyncMock(
            side_effect=dbmod.BusquedaDeFactFallida("la base no responde"))
        with self.assertLogs(dbmod.logger, level="WARNING"):
            resultado = await m.save_fact.__wrapped__(m, "Un hecho cualquiera", "user")
        self.assertTrue(resultado)

    async def test_si_supersede_falla_se_revierte_y_no_se_afirma_la_correccion(self):
        """user_id=990001 EXPLICITO: sin el, el candado de la Task 2 Step 4
        (feat/memoria-admin, "sin user_id no se supersede") ni siquiera
        llamaria a supersede_fact, y este test dejaria de probar lo que dice
        probar -- la falla TECNICA de supersede_fact, no la falta de autor."""
        m = dbmod.MemoryDB()
        m.pool = _pool_falso()
        m.get_embedding = mock.AsyncMock(return_value=[0.1, 0.2])
        m._find_nearest_fact = mock.AsyncMock(
            return_value={"id": 7, "fact_text": "viejo", "distancia": 0.05})
        m.supersede_fact = mock.AsyncMock(return_value=None)   # el decorador se lo trago
        m.delete_fact = mock.AsyncMock(return_value=True)
        with self.assertLogs(dbmod.logger, level="ERROR") as capturado:
            resultado = await m.save_fact.__wrapped__(
                m, "Fernando vive en San Pedro Sula", "user",
                is_correction=True, user_id=990001)
        self.assertIsNone(resultado)
        m.delete_fact.assert_awaited_once()
        logs = "\n".join(capturado.output)
        self.assertNotIn("corrige a", logs, "se afirmo una correccion que no ocurrio")
        self.assertIn("se revirtio", logs)

    async def test_si_tampoco_se_puede_revertir_se_grita(self):
        """user_id explicito, mismo motivo que el test anterior."""
        m = dbmod.MemoryDB()
        m.pool = _pool_falso()
        m.get_embedding = mock.AsyncMock(return_value=[0.1, 0.2])
        m._find_nearest_fact = mock.AsyncMock(
            return_value={"id": 7, "fact_text": "viejo", "distancia": 0.05})
        m.supersede_fact = mock.AsyncMock(return_value=None)
        m.delete_fact = mock.AsyncMock(return_value=None)
        with self.assertLogs(dbmod.logger, level="CRITICAL") as capturado:
            resultado = await m.save_fact.__wrapped__(
                m, "Fernando vive en San Pedro Sula", "user",
                is_correction=True, user_id=990001)
        self.assertIsNone(resultado)
        self.assertIn("dos hechos contradictorios activos", "\n".join(capturado.output))

    async def test_sin_user_id_no_se_supersede(self):
        """Task 2 Step 4 del plan 2026-09-20-memoria-admin: "si no se sabe
        quien (user_id None o 0), NO se supersede". Ni siquiera se INTENTA
        -- no es que supersede_fact falle, es que no se llama. El fact nuevo
        queda como fact nuevo (igual que cuando confirmado=False) y el viejo
        sigue activo: dos hechos sin resolver es mejor que un supersede con
        un autor inventado."""
        m = dbmod.MemoryDB()
        m.pool = _pool_falso()
        m.get_embedding = mock.AsyncMock(return_value=[0.1, 0.2])
        m._find_nearest_fact = mock.AsyncMock(
            return_value={"id": 7, "fact_text": "viejo", "distancia": 0.05})
        m.supersede_fact = mock.AsyncMock()
        with self.assertLogs(dbmod.logger, level="WARNING") as capturado:
            resultado = await m.save_fact.__wrapped__(
                m, "Fernando vive en San Pedro Sula", "user",
                is_correction=True, user_id=None)
        self.assertTrue(resultado, "sin user_id la insercion del fact nuevo deberia seguir")
        m.supersede_fact.assert_not_awaited()
        self.assertIn("no hay user_id", "\n".join(capturado.output))

    async def test_user_id_cero_tampoco_supersede(self):
        """0 no es un id de usuario valido: cuenta como 'no se sabe', igual
        que None. `if not user_id` en save_fact lo cubre a proposito."""
        m = dbmod.MemoryDB()
        m.pool = _pool_falso()
        m.get_embedding = mock.AsyncMock(return_value=[0.1, 0.2])
        m._find_nearest_fact = mock.AsyncMock(
            return_value={"id": 7, "fact_text": "viejo", "distancia": 0.05})
        m.supersede_fact = mock.AsyncMock()
        with self.assertLogs(dbmod.logger, level="WARNING"):
            resultado = await m.save_fact.__wrapped__(
                m, "Fernando vive en San Pedro Sula", "user",
                is_correction=True, user_id=0)
        self.assertTrue(resultado)
        m.supersede_fact.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
