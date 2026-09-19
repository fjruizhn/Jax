#!/usr/bin/env python3
"""El árbitro devuelve (spec 2026-09-18-arbitro-devuelve-design) -- lo que
`jacobs/_devolucion_test.py` NO puede probar con mocks: columnas reales,
migración idempotente, y el cupo contra una MariaDB real.

Necesita la gobernanza de jax-platform (axioma_config, facet, capability,
motor -- store.get_tope_devoluciones() y store.get_motor_governance() las
leen), así que corre en el mismo job que jacobs/_pipeline_identity_test.py y
jacobs/_step_motor_test.py (subpipeline-contrato-db, que clona jax-platform y
corre SUS migraciones antes de esto).

Corre con:
  JAX_TEST_DB_SUFIJO=<algo> PYTHONPATH=.:las_manos python -m pytest \
    jacobs/_devolucion_persistencia_test.py -v
(o sin el sufijo, contra la `jax_memory_test` compartida -- nunca
`jax_memory`: `base_de_test.fijar_base_de_test()` se niega a eso.)

En memoria de Jairo Urbina.
"""
from __future__ import annotations

from base_de_test import fijar_base_de_test  # noqa: E402

fijar_base_de_test()

import time  # noqa: E402
import unittest  # noqa: E402
import uuid  # noqa: E402
from decimal import Decimal  # noqa: E402

from jacobs import cupo, store  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus  # noqa: E402
from jacobs.policy import CupoAgotado  # noqa: E402

PREFIJO = "devolucion-persist-test-"


def _pipeline(**over) -> Pipeline:
    ahora = time.time()
    base = dict(
        pipeline_id=str(uuid.uuid4()), name=PREFIJO + "p", invoked_by="plataforma",
        mode="supervised", status=PipelineStatus.pending,
        created_at=ahora, updated_at=ahora,
    )
    base.update(over)
    return Pipeline(**base)


class _ConLimpieza(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.addAsyncCleanup(store.cerrar_pool)
        await store.init_tables()
        self._pipeline_ids: list[str] = []

    async def asyncTearDown(self):
        if not self._pipeline_ids:
            return
        conn = await store.conexion_dedicada()
        try:
            async with conn.cursor() as cur:
                for pid in self._pipeline_ids:
                    await cur.execute("DELETE FROM jacobs_steps WHERE pipeline_id=%s", (pid,))
                    await cur.execute("DELETE FROM jacobs_events WHERE pipeline_id=%s", (pid,))
                    await cur.execute("DELETE FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))
        finally:
            conn.close()

    def _rastrear(self, pipeline_id: str) -> None:
        self._pipeline_ids.append(pipeline_id)


# ---------------------------------------------------------------------------
# Migración idempotente (spec: "EN la lista de migración", mismo patrón que
# run_epoch/modelo_real).
# ---------------------------------------------------------------------------

class MigracionTest(_ConLimpieza):
    async def test_init_tables_es_idempotente_y_deja_las_columnas_nuevas(self):
        await store.init_tables()  # segunda vez: no debe fallar ni duplicar
        conn = await store.conexion_dedicada()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_DEFAULT "
                    "FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() "
                    "AND TABLE_NAME='jacobs_pipelines' AND COLUMN_NAME IN "
                    "('costo_max_aceptado_usd', 'devoluciones')"
                )
                filas = {r[0]: r for r in await cur.fetchall()}
        finally:
            conn.close()
        self.assertIn("costo_max_aceptado_usd", filas)
        self.assertIn("devoluciones", filas)
        self.assertEqual(filas["devoluciones"][2], "NO")  # NOT NULL
        self.assertEqual(str(filas["devoluciones"][3]), "0")  # DEFAULT 0


# ---------------------------------------------------------------------------
# costo_max_aceptado_usd persiste -- verificado en los DOS caminos de
# creación reales: store.pipeline_create() y cupo.reservar_cupo() +
# cupo.completar_reserva() (el que routes.py usa de verdad).
# ---------------------------------------------------------------------------

class PersistenciaDelTopeTest(_ConLimpieza):
    async def test_pipeline_create_persiste_el_tope_y_arranca_en_cero_devoluciones(self):
        p = _pipeline(costo_max_aceptado_usd=Decimal("12.5000"))
        self._rastrear(p.pipeline_id)
        await store.pipeline_create(p)
        leido = await store.pipeline_get(p.pipeline_id)
        self.assertEqual(leido.costo_max_aceptado_usd, Decimal("12.5000"))
        self.assertEqual(leido.devoluciones, 0)

    async def test_pipeline_create_sin_tope_persiste_none(self):
        p = _pipeline(costo_max_aceptado_usd=None)
        self._rastrear(p.pipeline_id)
        await store.pipeline_create(p)
        leido = await store.pipeline_get(p.pipeline_id)
        self.assertIsNone(leido.costo_max_aceptado_usd)

    async def test_el_camino_real_de_creacion_completar_reserva_persiste_el_tope(self):
        """routes.py NO llama pipeline_create(): reserva con cupo.reservar_cupo
        y completa con cupo.completar_reserva -- verificado contra el código
        antes de esta ronda: SQL_COMPLETAR no traía costo_max_aceptado_usd
        (spec §3.4, "verificado")."""
        p = _pipeline()
        self._rastrear(p.pipeline_id)
        self.assertTrue(await cupo.reservar_cupo(p, limite=1000))
        p.plan = [Step(step_id="s0", pipeline_id=p.pipeline_id, step_index=0,
                        facet="jax_local", capability="text_generation",
                        status=StepStatus.pending, input={"prompt": "x"})]
        p.costo_max_aceptado_usd = Decimal("3.4100")
        await cupo.completar_reserva(p)
        leido = await store.pipeline_get(p.pipeline_id)
        self.assertEqual(leido.costo_max_aceptado_usd, Decimal("3.4100"))
        self.assertEqual(len(leido.plan), 1)


# ---------------------------------------------------------------------------
# El tope de devoluciones sale de axioma_config, fail-closed sin fila.
# ---------------------------------------------------------------------------

class TopeDeDevolucionesTest(unittest.IsolatedAsyncioTestCase):
    CLAVE = "jacobs.tope_devoluciones"

    async def asyncSetUp(self):
        self.addAsyncCleanup(store.cerrar_pool)
        await store.init_tables()
        self.addAsyncCleanup(self._borrar_config)

    async def _borrar_config(self):
        conn = await store.conexion_dedicada()
        try:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM axioma_config WHERE config_key=%s", (self.CLAVE,))
        finally:
            conn.close()

    async def _set_config(self, valor: str) -> None:
        conn = await store.conexion_dedicada()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO axioma_config (config_key, config_value) VALUES (%s, %s) "
                    "ON DUPLICATE KEY UPDATE config_value=VALUES(config_value)",
                    (self.CLAVE, valor),
                )
        finally:
            conn.close()

    async def test_sin_fila_el_tope_es_cero_fail_closed(self):
        await self._borrar_config()
        self.assertEqual(await store.get_tope_devoluciones(), 0)

    async def test_con_fila_valida_el_tope_es_ese_valor(self):
        await self._set_config("2")
        self.assertEqual(await store.get_tope_devoluciones(), 2)

    async def test_valor_no_numerico_cae_a_cero_fail_closed(self):
        await self._set_config("dos")
        self.assertEqual(await store.get_tope_devoluciones(), 0)

    async def test_valor_negativo_cae_a_cero_fail_closed(self):
        await self._set_config("-1")
        self.assertEqual(await store.get_tope_devoluciones(), 0)


# ---------------------------------------------------------------------------
# El bug que aplicar_cupo=False evita: un pipeline `running` YA cuenta contra
# su propio cupo. Con aplicar_cupo=True (la forma de continuar/resume, que
# SÍ pide un lugar nuevo porque el pipeline dejó de estar vivo) y el cupo
# saturado, la devolución se negaría a sí misma sin ningún motivo real.
# ---------------------------------------------------------------------------

class CupoDeLaDevolucionTest(_ConLimpieza):
    async def _pipeline_corriendo_con_un_paso(self) -> Pipeline:
        p = _pipeline(status=PipelineStatus.running, run_epoch=1)
        self._rastrear(p.pipeline_id)
        paso = Step(step_id="s0", pipeline_id=p.pipeline_id, step_index=0,
                    facet="jax_local", capability="text_generation",
                    status=StepStatus.completed, input={"prompt": "x"})
        p.plan = [paso]
        p.costo_max_aceptado_usd = Decimal("9.9900")
        await store.pipeline_create(p)
        await store.step_upsert(paso)
        return p

    async def test_devolucion_no_se_frena_contra_su_propio_cupo(self):
        p = await self._pipeline_corriendo_con_un_paso()
        # Cupo saturado A PROPÓSITO: el límite es exactamente 1, y este
        # pipeline (running) ya lo ocupa -- así se ve el bug si volviera.
        activos = await store.pipeline_count_active()
        self.assertGreaterEqual(activos, 1)

        paso = p.plan[0]
        nueva = await store.continuar_transaccion(
            p.pipeline_id, p.run_epoch, p.status, [paso], p.plan, p.context, 0,
            evento_payload={"paso": 0, "motivo": "m", "cita": "[paso 0]"},
            evento_tipo="PIPELINE_DEVUELTO",
            aplicar_cupo=False, incrementar_devoluciones=True,
            cupo_maximo=activos,  # a propósito, IGUAL a los activos: saturado
        )
        self.assertIsNotNone(nueva, "la devolución se frenó contra su propio cupo")
        self.assertEqual(nueva, p.run_epoch + 1)

        leido = await store.pipeline_get(p.pipeline_id)
        self.assertEqual(leido.devoluciones, 1)
        self.assertEqual(leido.run_epoch, p.run_epoch + 1)
        self.assertEqual(leido.status, PipelineStatus.running)

    async def test_la_misma_situacion_CON_cupo_si_se_frena(self):
        """Control (Principio VII -- un freno sin prueba no es freno): con
        `aplicar_cupo=True` en la MISMA situación saturada, la escritura NO
        avanza (CupoAgotado) -- confirma que el test de arriba prueba algo
        real y no pasa porque el cupo nunca frena nada."""
        p = await self._pipeline_corriendo_con_un_paso()
        activos = await store.pipeline_count_active()

        paso = p.plan[0]
        with self.assertRaises(CupoAgotado):
            await store.continuar_transaccion(
                p.pipeline_id, p.run_epoch, p.status, [paso], p.plan, p.context, 0,
                evento_payload={"paso": 0, "motivo": "m", "cita": "[paso 0]"},
                evento_tipo="PIPELINE_DEVUELTO",
                aplicar_cupo=True, incrementar_devoluciones=True,
                cupo_maximo=activos,
            )


# ---------------------------------------------------------------------------
# ALTO (Ronda de arreglo 1): la crítica tiene que sobrevivir a los TRES
# caminos de recuperación (/continue, /resume, /approve-step) -- los tres
# reconstruyen su plan de trabajo desde `jacobs_steps` vía
# `store.steps_by_pipeline()` ("los pasos VIGENTES, no la foto de creación"),
# NO desde `jacobs_pipelines.plan`. `_inyectar_critica` (jacobs/devolucion.py)
# solo tocaba el objeto en memoria; sin que `_SQL_PASO_A_CORRER` persista
# `input_ref`, un abort después de la devolución perdía la crítica en el
# primer /continue -- se paga dos veces el mismo error, el defecto que esta
# ronda entera existe para cerrar.
#
# El test NO mira que el campo `input["prompt"]` exista (eso ya lo prueba
# jacobs/_devolucion_test.py, con mocks, sobre el payload en memoria): arma
# la fila real vía continuar_transaccion, la relee con steps_by_pipeline
# (exactamente lo que hacen continuar.analizar/resume/approve-step) y
# renderiza el prompt con jacobs.executor._build_context_input +
# _enrich_prompt -- lo que el modelo REALMENTE va a leer.
# ---------------------------------------------------------------------------

class CriticaSobreviveALaRecuperacionTest(_ConLimpieza):
    async def test_la_critica_sobrevive_steps_by_pipeline_tras_abortar_y_releer(self):
        from jacobs import executor
        from jacobs.devolucion import _inyectar_critica
        from jacobs.veredicto import VeredictoArbitro

        p = _pipeline(status=PipelineStatus.running, run_epoch=1)
        self._rastrear(p.pipeline_id)
        original = Step(
            step_id="s0", pipeline_id=p.pipeline_id, step_index=0, facet="ada",
            capability="architecture_review", status=StepStatus.completed,
            input={"prompt": "prompt original del paso 0"},
        )
        p.plan = [original]
        p.costo_max_aceptado_usd = Decimal("9.9900")
        await store.pipeline_create(p)
        await store.step_upsert(original)

        # La devolución: MISMO mecanismo que jacobs/devolucion.py usa --
        # copiar el Step, inyectar la crítica en su prompt, pasarlo como
        # `pasos_a_correr` de continuar_transaccion.
        devuelto = original.model_copy(deep=True)
        devuelto.status = StepStatus.pending
        veredicto = VeredictoArbitro(
            decision="devolver", paso=0,
            motivo="usaste BIGSERIAL y TIMESTAMPTZ; el destino es MariaDB",
            cita="[paso 0]",
        )
        _inyectar_critica(devuelto, veredicto, 1)
        nueva = await store.continuar_transaccion(
            p.pipeline_id, p.run_epoch, p.status, [devuelto], [devuelto], p.context, 0,
            evento_payload={"paso": 0, "motivo": veredicto.motivo, "cita": veredicto.cita},
            evento_tipo="PIPELINE_DEVUELTO",
            aplicar_cupo=False, incrementar_devoluciones=True,
        )
        self.assertIsNotNone(nueva)

        # "ADA rehace, timeout, aborted" -- el escenario exacto del hallazgo.
        await store.pipeline_update_status(p.pipeline_id, PipelineStatus.aborted)

        # "Fernando hace /continue": la reconstrucción real de
        # continuar.analizar()/resume/approve-step, no el objeto en memoria.
        releido = await store.steps_by_pipeline(p.pipeline_id)
        paso_releido = next(s for s in releido if s.step_index == 0)
        self.assertIn("BIGSERIAL", paso_releido.input.get("prompt", ""))
        self.assertIn("prompt original del paso 0", paso_releido.input.get("prompt", ""))

        # Y lo que el modelo REALMENTE va a leer, no solo el campo crudo.
        pipeline_releido = await store.pipeline_get(p.pipeline_id)
        pipeline_releido.plan = releido
        ctx_input = executor._build_context_input(paso_releido, pipeline_releido)
        prompt_renderizado = executor._enrich_prompt(ctx_input)
        self.assertIn("BIGSERIAL", prompt_renderizado)
        self.assertIn("MariaDB", prompt_renderizado)
        self.assertIn("prompt original del paso 0", prompt_renderizado)


if __name__ == "__main__":
    unittest.main(verbosity=2)
