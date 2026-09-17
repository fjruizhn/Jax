"""Contrato de sub-pipelines contra una MariaDB REAL (frente F, 2026-09-16):
esquema, emisión, consumo atómico, carreras y plan de la consulta.

Las tablas de Jacobs las crea ESTE repo (store.init_tables), así que el job de
CI no necesita el esquema de jax-platform.

Seguridad: la barrera de base vive en jacobs/_arnes_ada.py, que se importa
PRIMERO: con JAX_DB_NAME apuntando a otra base que no sea jax_memory_test, el
módulo entero levanta RuntimeError.

Corre con:
  bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -v jacobs/_subpipeline_contrato_io_test.py"
"""
from __future__ import annotations

from jacobs import _arnes_ada as ada  # primero: barrera de base de prueba

import asyncio  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import unittest  # noqa: E402
import uuid  # noqa: E402
from unittest.mock import patch  # noqa: E402

import aiomysql  # noqa: E402

from jacobs import store  # noqa: E402
from jacobs import subpipelines as sp  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus  # noqa: E402

COLUMNAS_TOKENS = [
    "token_hash", "parent_pipeline_id", "parent_step", "depth_hijo",
    "emitido_at", "vence_at", "usado_at", "hijo_pipeline_id",
]


@unittest.skipUnless(os.getenv("JAX_DB_HOST"), "necesita la MariaDB real (jax_memory_test)")
class _ConBase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await store.init_tables()
        freno = patch("jacobs.policy.check_kill_switch", return_value=False)
        freno.start()
        self.addCleanup(freno.stop)
        entorno = patch.dict(os.environ)
        entorno.start()
        self.addCleanup(entorno.stop)
        os.environ.pop(sp.ENV_TTL, None)
        os.environ.pop(sp.ENV_MAX_PROFUNDIDAD, None)
        self.padres: list[str] = []

    async def asyncTearDown(self):
        for pid in self.padres:
            await ada.cerrar(pid)

    async def padre(self, **kw) -> tuple[str, str]:
        pid, paso = await ada.padre_en_ejecucion(**kw)
        self.padres.append(pid)
        return pid, paso

    async def columnas(self, tabla: str) -> list[str]:
        conn = await store.get_conn()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s ORDER BY ORDINAL_POSITION",
                    (tabla,),
                )
                return [r[0] for r in await cur.fetchall()]
        finally:
            conn.close()


class EsquemaTest(_ConBase):
    async def test_init_tables_crea_la_tabla_de_tokens_sin_columna_para_el_token(self):
        # Se BORRA y se recrea: la jax_memory_test local puede tenerla de una
        # corrida anterior, y entonces el test pasaría sin init_tables().
        await ada.ejecutar("DROP TABLE IF EXISTS jacobs_subpipeline_tokens")
        await store.init_tables()
        self.assertEqual(await self.columnas("jacobs_subpipeline_tokens"), COLUMNAS_TOKENS)

    async def test_tokens_tienen_pk_por_hash_e_indice_por_padre(self):
        sql = (
            "SELECT GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) AS cols "
            "FROM information_schema.STATISTICS WHERE TABLE_SCHEMA=DATABASE() "
            "AND TABLE_NAME='jacobs_subpipeline_tokens' AND INDEX_NAME=%s"
        )
        self.assertEqual((await ada.una_fila(sql, ("PRIMARY",)))["cols"], "token_hash")
        self.assertEqual(
            (await ada.una_fila(sql, ("idx_subpipeline_tokens_padre",)))["cols"],
            "parent_pipeline_id",
        )

    async def test_jacobs_pipelines_gana_parent_y_depth_aun_si_la_tabla_ya_existia(self):
        await ada.ejecutar(
            "ALTER TABLE jacobs_pipelines DROP COLUMN IF EXISTS parent_pipeline_id, "
            "DROP COLUMN IF EXISTS depth"
        )
        await store.init_tables()
        columnas = await self.columnas("jacobs_pipelines")
        self.assertIn("parent_pipeline_id", columnas)
        self.assertIn("depth", columnas)

    async def test_init_tables_es_idempotente_con_el_contrato(self):
        await store.init_tables()
        await store.init_tables()
        self.assertEqual(await self.columnas("jacobs_subpipeline_tokens"), COLUMNAS_TOKENS)
        self.assertEqual((await self.columnas("jacobs_pipelines")).count("depth"), 1)

    async def test_pipeline_create_conserva_parent_y_depth(self):
        pid = str(uuid.uuid4())
        ahora = time.time()
        await store.pipeline_create(Pipeline(
            pipeline_id=pid, name="hijo", invoked_by="ada", mode="dry_run",
            status=PipelineStatus.completed, parent_pipeline_id="padre-x", depth=1,
            created_at=ahora, updated_at=ahora,
        ))
        leido = await store.pipeline_get(pid)
        self.assertEqual((leido.parent_pipeline_id, leido.depth), ("padre-x", 1))


class EmisionTest(_ConBase):
    async def test_guarda_solo_el_hash(self):
        padre, paso = await self.padre()
        token = await sp.emitir_token_subpipeline(padre, paso)
        self.assertGreaterEqual(len(token), 43)  # 32 bytes en base64url
        self.assertIsNotNone(await ada.fila_token(sp.hash_token(token)))
        self.assertIsNone(await ada.fila_token(token))
        fila = await ada.una_fila(
            "SELECT COUNT(*) AS n FROM jacobs_subpipeline_tokens "
            "WHERE parent_step=%s OR hijo_pipeline_id=%s OR parent_pipeline_id=%s",
            (token, token, token),
        )
        self.assertEqual(fila["n"], 0)

    async def test_depth_hijo_es_la_del_padre_mas_uno_y_vence_con_el_ttl(self):
        os.environ[sp.ENV_MAX_PROFUNDIDAD] = "3"
        os.environ[sp.ENV_TTL] = "120"
        padre, paso = await self.padre(depth=2)
        antes = time.time()
        token = await sp.emitir_token_subpipeline(padre, paso)
        fila = await ada.fila_token(sp.hash_token(token))
        self.assertEqual(fila["depth_hijo"], 3)
        self.assertEqual((fila["parent_pipeline_id"], fila["parent_step"]), (padre, paso))
        self.assertAlmostEqual(fila["vence_at"] - fila["emitido_at"], 120, delta=0.01)
        self.assertGreaterEqual(fila["emitido_at"], antes)
        self.assertIsNone(fila["usado_at"])
        self.assertIsNone(fila["hijo_pipeline_id"])

    async def test_deja_evento_emitido_sin_el_token(self):
        padre, paso = await self.padre()
        token = await sp.emitir_token_subpipeline(padre, paso)
        eventos = [e for e in await store.events_by_pipeline(padre)
                   if e["event_type"] == "SUBPIPELINE_TOKEN_EMITIDO"]
        self.assertEqual(len(eventos), 1)
        self.assertEqual(eventos[0]["step_id"], paso)
        self.assertEqual(eventos[0]["payload"]["token_ref"], sp.token_ref(sp.hash_token(token)))
        self.assertEqual(eventos[0]["payload"]["depth_hijo"], 1)
        self.assertNotIn(token, json.dumps(eventos[0]["payload"]))

    async def _rechazo(self, padre: str, paso: str) -> sp.Motivo:
        with self.assertRaises(sp.EmisionRechazada) as ctx:
            await sp.emitir_token_subpipeline(padre, paso)
        eventos = [e for e in await store.events_by_pipeline(padre)
                   if e["event_type"] == "SUBPIPELINE_RECHAZADO"]
        self.assertTrue(eventos, "un rechazo de emisión tiene que dejar evento")
        self.assertEqual(eventos[-1]["payload"]["fase"], "emision")
        self.assertEqual(eventos[-1]["payload"]["motivo"], ctx.exception.motivo.value)
        n = await ada.una_fila(
            "SELECT COUNT(*) AS n FROM jacobs_subpipeline_tokens WHERE parent_pipeline_id=%s",
            (padre,),
        )
        self.assertEqual(n["n"], 0, "un rechazo no puede dejar un token emitido")
        return ctx.exception.motivo

    async def test_rechaza_padre_que_no_corre(self):
        padre, paso = await self.padre()
        await ada.cerrar(padre)
        self.assertEqual(await self._rechazo(padre, paso), sp.Motivo.PADRE_INACTIVO)

    async def test_rechaza_paso_que_no_es_de_ada(self):
        padre, paso = await self.padre(facet_del_paso="jekyll")
        self.assertEqual(await self._rechazo(padre, paso), sp.Motivo.PASO_NO_ES_ADA)

    async def test_rechaza_profundidad_excedida(self):
        padre, paso = await self.padre(depth=3)  # con el máximo por defecto (3)
        self.assertEqual(await self._rechazo(padre, paso), sp.Motivo.PROFUNDIDAD_EXCEDIDA)

    async def test_rechaza_con_kill_switch_activo(self):
        padre, paso = await self.padre()
        with patch("jacobs.policy.check_kill_switch", return_value=True):
            motivo = await self._rechazo(padre, paso)
        self.assertEqual(motivo, sp.Motivo.KILL_SWITCH_ACTIVO)


class ConsumoTest(_ConBase):
    async def test_consumo_legitimo_marca_usado_y_ata_al_hijo(self):
        padre, paso = await self.padre()
        token = await sp.emitir_token_subpipeline(padre, paso)
        hijo = str(uuid.uuid4())
        resultado = await sp.consumir_token_subpipeline(token, padre, hijo)
        self.assertEqual(
            resultado, sp.TokenConsumido(
                parent_pipeline_id=padre, parent_step=paso, depth=1, user_id=None, tenant_id=None))
        fila = await ada.fila_token(sp.hash_token(token))
        self.assertEqual(fila["hijo_pipeline_id"], hijo)
        self.assertIsNotNone(fila["usado_at"])

    async def test_token_inexistente_es_desconocido_y_deja_evento_sin_el_token(self):
        hijo = str(uuid.uuid4())
        resultado = await sp.consumir_token_subpipeline("no-existe-este-token", "padre-x", hijo)
        self.assertEqual(resultado, sp.ConsumoRechazado(sp.Motivo.TOKEN_DESCONOCIDO))
        eventos = await store.events_by_pipeline(hijo)
        self.assertEqual([e["event_type"] for e in eventos], ["SUBPIPELINE_RECHAZADO"])
        self.assertEqual(eventos[0]["payload"]["fase"], "consumo")
        self.assertNotIn("no-existe-este-token", json.dumps(eventos[0]["payload"]))

    async def test_rechazo_por_otro_padre_no_quema_el_token(self):
        padre, paso = await self.padre()
        token = await sp.emitir_token_subpipeline(padre, paso)
        self.assertEqual(
            await sp.consumir_token_subpipeline(token, str(uuid.uuid4()), str(uuid.uuid4())),
            sp.ConsumoRechazado(sp.Motivo.PADRE_NO_COINCIDE),
        )
        # M-1: el rechazo también se ve desde el padre REAL del token (fila).
        rechazos_en_padre = [
            e for e in await store.events_by_pipeline(padre)
            if e["event_type"] == "SUBPIPELINE_RECHAZADO"
        ]
        self.assertEqual(
            [(e["payload"]["fase"], e["payload"]["motivo"]) for e in rechazos_en_padre],
            [("consumo", sp.Motivo.PADRE_NO_COINCIDE.value)],
        )
        self.assertNotIn(token, json.dumps(rechazos_en_padre[0]["payload"]))
        self.assertIsInstance(
            await sp.consumir_token_subpipeline(token, padre, str(uuid.uuid4())),
            sp.TokenConsumido,
        )

    async def test_paso_de_ada_terminado_con_padre_corriendo_emite_y_consume(self):
        """Enmienda de Fernando 2026-09-16: en el modo "plan de delegación" Jacobs
        emite los tokens DESPUÉS de que el step `delegate` terminó. Lo que habilita
        es el pipeline padre `running`; el estado del paso no cuenta."""
        padre, paso = await self.padre()
        await ada.ejecutar(
            "UPDATE jacobs_steps SET status = 'completed' WHERE step_id = %s", (paso,))
        token = await sp.emitir_token_subpipeline(padre, paso)
        token_previo = await sp.emitir_token_subpipeline(padre, paso)
        self.assertIsInstance(
            await sp.consumir_token_subpipeline(token, padre, str(uuid.uuid4())),
            sp.TokenConsumido,
        )
        await ada.cerrar(padre)
        with self.assertRaises(sp.EmisionRechazada) as ctx:
            await sp.emitir_token_subpipeline(padre, paso)
        self.assertEqual(ctx.exception.motivo, sp.Motivo.PADRE_INACTIVO)
        self.assertEqual(
            await sp.consumir_token_subpipeline(token_previo, padre, str(uuid.uuid4())),
            sp.ConsumoRechazado(sp.Motivo.PADRE_INACTIVO),
        )

    async def test_carrera_de_veinte_consumos_tiene_un_solo_ganador(self):
        padre, paso = await self.padre()
        for ronda in range(10):
            token = await sp.emitir_token_subpipeline(padre, paso)
            hijos = [str(uuid.uuid4()) for _ in range(20)]
            resultados = await asyncio.gather(*[
                sp.consumir_token_subpipeline(token, padre, hijo) for hijo in hijos
            ])
            ganadores = [r for r in resultados if isinstance(r, sp.TokenConsumido)]
            perdedores = [r for r in resultados if not isinstance(r, sp.TokenConsumido)]
            self.assertEqual(len(ganadores), 1, f"ronda {ronda}: {resultados}")
            self.assertEqual(
                set(perdedores), {sp.ConsumoRechazado(sp.Motivo.TOKEN_USADO)}, f"ronda {ronda}")
            hijo_ganador = hijos[resultados.index(ganadores[0])]
            fila = await ada.fila_token(sp.hash_token(token))
            self.assertEqual(
                fila["hijo_pipeline_id"], hijo_ganador,
                f"ronda {ronda}: la fila quedó atada a un hijo que no es el que ganó",
            )

    async def test_carrera_forzada_el_segundo_espera_el_candado_y_pierde(self):
        """Intercalado forzado con dos sesiones reales: A consume y NO confirma;
        B tiene que quedar esperando el candado de fila y, al confirmar A,
        releer y perder. Sin `usado_at IS NULL` en el WHERE, B también gana."""
        padre, paso = await self.padre()
        token = await sp.emitir_token_subpipeline(padre, paso)
        max_prof = sp.config_subpipelines().max_profundidad
        conn_a = await store.get_conn()
        try:
            await conn_a.autocommit(False)
            async with conn_a.cursor() as cur:
                ahora = time.time()
                await cur.execute(
                    store.SQL_CONSUMIR_TOKEN,
                    (ahora, "hijo-a", sp.hash_token(token), ahora, padre, max_prof),
                )
                self.assertEqual(cur.rowcount, 1)
            tarea_b = asyncio.create_task(sp.consumir_token_subpipeline(token, padre, "hijo-b"))
            await asyncio.sleep(1.0)
            self.assertFalse(
                tarea_b.done(),
                "B no esperó el candado de fila de A: el consumo no es atómico",
            )
            await conn_a.commit()
            resultado_b = await asyncio.wait_for(tarea_b, timeout=15)
        finally:
            conn_a.close()
            # Si una aserción cortó antes de esperar a B, no queda colgada.
            if "tarea_b" in locals() and not tarea_b.done():
                tarea_b.cancel()
        self.assertEqual(resultado_b, sp.ConsumoRechazado(sp.Motivo.TOKEN_USADO))
        self.assertEqual((await ada.fila_token(sp.hash_token(token)))["hijo_pipeline_id"], "hijo-a")

    async def test_nivel_de_aislamiento_declarado(self):
        fila = await ada.una_fila("SELECT @@SESSION.transaction_isolation AS nivel")
        self.assertIn(
            fila["nivel"], {"REPEATABLE-READ", "READ-COMMITTED"},
            "el argumento de atomicidad del consumo (UPDATE con candado de fila que "
            "relee la versión confirmada) está escrito para estos dos niveles",
        )

    async def test_explain_del_consumo_usa_claves_primarias(self):
        padre, paso = await self.padre()
        for _ in range(30):  # volumen: con tablas casi vacías el optimizador puede elegir ALL
            await sp.emitir_token_subpipeline(padre, paso)
        token = await sp.emitir_token_subpipeline(padre, paso)
        conn = await store.get_conn()
        try:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                ahora = time.time()
                # La SQL REAL (la constante que ejecuta el store), no una copia.
                await cur.execute(
                    "EXPLAIN " + store.SQL_CONSUMIR_TOKEN,
                    (ahora, "hijo-explain", sp.hash_token(token), ahora, padre,
                     sp.config_subpipelines().max_profundidad),
                )
                plan = await cur.fetchall()
        finally:
            conn.close()
        por_tabla = {f["table"]: f for f in plan}
        self.assertEqual(set(por_tabla), {"t", "p", "s"}, plan)
        for alias, fila in por_tabla.items():
            self.assertIn(fila["type"], {"const", "eq_ref"}, f"{alias}: {fila}")
            self.assertEqual(fila["key"], "PRIMARY", f"{alias}: {fila}")
        # Las dos lecturas de identidad del padre (revisión final, I-3): t y p por PK.
        for sql in (store.SQL_TOKEN_CONSUMIDO, store.SQL_IDENTIDAD_PADRE_DEL_TOKEN):
            conn = await store.get_conn()
            try:
                async with conn.cursor(aiomysql.DictCursor) as cur:
                    await cur.execute("EXPLAIN " + sql, (sp.hash_token(token),))
                    plan = await cur.fetchall()
            finally:
                conn.close()
            por_tabla = {f["table"]: f for f in plan}
            self.assertEqual(set(por_tabla), {"t", "p"}, plan)
            for alias, fila in por_tabla.items():
                self.assertIn(fila["type"], {"const", "eq_ref"}, f"{alias}: {fila}")
                self.assertEqual(fila["key"], "PRIMARY", f"{alias}: {fila}")


if __name__ == "__main__":
    unittest.main()
