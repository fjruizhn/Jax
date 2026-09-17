"""El cupo de pipelines lo hace cumplir LA BASE, no un candado del proceso.

POR QUÉ EXISTE (2026-09-17). `routes._pipeline_create_lock` era un
`asyncio.Lock()` global del proceso que envolvía leer el conteo de activos,
consumir el token de sub-pipeline, planificar (20-40 s de LLM) e insertar la
fila. Hacía cumplir `MAX_PARALLEL_PIPELINES`, sí, pero al precio de serializar
TODA la creación: la medición del frente G (2026-09-17) dio un techo de ~43
delegaciones/s y, peor, la creación de pipelines de la Mesa esperaba detrás de
las delegaciones de Ada porque el candado era el MISMO objeto.

QUÉ VIGILA ESTE ARCHIVO. Que el reemplazo —un `INSERT ... SELECT ... WHERE
(SELECT COUNT(*) ...) < limite`, que decide por filas afectadas— cumple lo
mismo SIN candado:

  1. Con N corrutinas reservando a la vez nunca se supera el cupo. Un control
     de concurrencia que pase también con el código viejo no prueba nada del
     arreglo (el viejo cumplía el límite, con candado); lo que prueba el
     arreglo es que se cumple CON EL CANDADO BORRADO, y eso es lo que corre acá.
  2. El `INSERT` condicionado toma candados de fila sobre `jacobs_pipelines` y
     dos reservas simultáneas pueden trabarse (error 1213). Medido el
     2026-09-17 en hall9000: sin reintento, 5/10, 22/25 y 21/50 intentos
     mueren de deadlock; con UN reintento, cero errores. Por eso el reintento
     es parte del contrato y se prueba, no un detalle de implementación.
  3. Soltar la reserva devuelve el cupo (el camino de fallo después de
     reservar).
  4. Fail-closed: si la base no responde, NO se reserva — nunca fail-open.
  5. El `EXPLAIN` de la sentencia REAL usa el índice de `status`.

Corre con:
  bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -v jacobs/_cupo_io_test.py"

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

from jacobs import _arnes_ada as ada  # primero: barrera de base de prueba

import asyncio  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import unittest  # noqa: E402
import uuid  # noqa: E402
from unittest.mock import patch  # noqa: E402

import aiomysql  # noqa: E402

from jacobs import cupo, store  # noqa: E402
from jacobs.models import Pipeline  # noqa: E402

PREFIJO = "cupo-io-test-"


def _pipeline(nombre: str) -> Pipeline:
    ahora = time.time()
    return Pipeline(
        pipeline_id=str(uuid.uuid4()), name=nombre, invoked_by="plataforma",
        mode="dry_run", created_at=ahora, updated_at=ahora,
    )


@unittest.skipUnless(os.getenv("JAX_DB_HOST"), "necesita la MariaDB real (jax_memory_test)")
class CupoEnLaBaseTest(unittest.IsolatedAsyncioTestCase):
    """Base COMPARTIDA con otras sesiones: todo DELETE lleva WHERE por el
    prefijo de este archivo, y el cupo se mide sobre las filas propias."""

    async def asyncSetUp(self):
        self.addAsyncCleanup(store.cerrar_pool)
        self.addAsyncCleanup(self._limpiar)
        await store.init_tables()
        await self._limpiar()
        # Ninguna fila viva ajena: si la hubiera, ocuparía cupo y el número
        # medido no sería el de este test. Se declara, no se supone.
        self.ajenos = await cupo.activos()
        self.assertEqual(
            self.ajenos, 0,
            "hay pipelines vivos ajenos en jax_memory_test: el cupo medido no sería el de este test",
        )

    async def _limpiar(self):
        await ada.ejecutar(
            "DELETE FROM jacobs_pipelines WHERE name LIKE %s", (PREFIJO + "%",)
        )

    async def _vivos(self) -> int:
        fila = await ada.una_fila(
            "SELECT COUNT(*) AS n FROM jacobs_pipelines "
            "WHERE status IN ('pending','running') AND name LIKE %s", (PREFIJO + "%",)
        )
        return int(fila["n"])

    # ---- 1. corrección sin candado, bajo concurrencia -------------------

    async def test_veinticinco_corrutinas_a_la_vez_no_superan_el_cupo(self):
        limite = 3
        pico = 0

        async def reservar(i: int) -> bool:
            ok = await cupo.reservar_cupo(_pipeline(f"{PREFIJO}{i}"), limite=limite)
            nonlocal pico
            pico = max(pico, await self._vivos())
            return ok

        resultados = await asyncio.gather(*[reservar(i) for i in range(25)])

        self.assertEqual(sum(resultados), limite, "se admitieron más (o menos) que el cupo")
        self.assertEqual(await self._vivos(), limite)
        self.assertLessEqual(pico, limite, "el cupo se superó en algún instante de la carrera")

    async def test_cincuenta_a_la_vez_sin_un_solo_error_de_deadlock(self):
        """El reintento del 1213 es contrato, no detalle: sin él esta misma
        carrera dejó 21 errores de 50 en la medición del 2026-09-17."""
        limite = 3
        resultados = await asyncio.gather(
            *[cupo.reservar_cupo(_pipeline(f"{PREFIJO}d{i}"), limite=limite) for i in range(50)],
            return_exceptions=True,
        )
        errores = [r for r in resultados if isinstance(r, BaseException)]
        self.assertEqual(errores, [], f"deadlocks sin reintentar: {errores[:3]}")
        self.assertEqual(sum(1 for r in resultados if r is True), limite)

    # ---- 2. el cupo vuelve cuando la reserva se suelta -------------------

    async def test_soltar_la_reserva_devuelve_el_cupo(self):
        limite = 2
        uno, dos = _pipeline(PREFIJO + "uno"), _pipeline(PREFIJO + "dos")
        self.assertTrue(await cupo.reservar_cupo(uno, limite=limite))
        self.assertTrue(await cupo.reservar_cupo(dos, limite=limite))
        self.assertFalse(await cupo.reservar_cupo(_pipeline(PREFIJO + "tres"), limite=limite))

        await cupo.soltar_reserva(uno.pipeline_id)
        self.assertEqual(await self._vivos(), 1)
        self.assertTrue(await cupo.reservar_cupo(_pipeline(PREFIJO + "cuatro"), limite=limite))

    async def test_soltar_no_toca_un_pipeline_que_ya_arranco(self):
        """`soltar_reserva` solo borra una reserva sin estrenar (`pending`).
        Un pipeline que ya corre no se borra ni por error de programación."""
        p = _pipeline(PREFIJO + "corriendo")
        self.assertTrue(await cupo.reservar_cupo(p, limite=3))
        await ada.ejecutar(
            "UPDATE jacobs_pipelines SET status='running' WHERE pipeline_id=%s", (p.pipeline_id,)
        )
        self.assertEqual(await cupo.soltar_reserva(p.pipeline_id), 0)
        self.assertIsNotNone(await store.pipeline_get(p.pipeline_id))

    # ---- 3. completar la reserva deja la fila REAL ----------------------

    async def test_completar_la_reserva_escribe_plan_identidad_y_padre(self):
        p = _pipeline(PREFIJO + "completar")
        self.assertTrue(await cupo.reservar_cupo(p, limite=3))
        reservado = await store.pipeline_get(p.pipeline_id)
        self.assertEqual(reservado.plan, [], "la reserva no debe traer plan todavía")

        padre, _paso = await ada.padre_en_ejecucion()
        try:
            p.parent_pipeline_id, p.depth = padre, 1
            p.user_id, p.tenant_id = "7", "t-7"
            p.plan = (await ada.plan_de_un_paso(p.pipeline_id, "o", 1, None))
            await cupo.completar_reserva(p)

            leido = await store.pipeline_get(p.pipeline_id)
            self.assertEqual(leido.parent_pipeline_id, padre)
            self.assertEqual(leido.depth, 1)
            self.assertEqual((leido.user_id, leido.tenant_id), ("7", "t-7"))
            self.assertEqual(len(leido.plan), 1)
        finally:
            await ada.cerrar(padre)

    # ---- 4. fail-closed --------------------------------------------------

    async def test_si_la_base_no_responde_no_se_reserva(self):
        with patch.object(store, "conexion", side_effect=TimeoutError("pool lleno")):
            with self.assertRaises(TimeoutError):
                await cupo.reservar_cupo(_pipeline(PREFIJO + "sin-base"), limite=3)
        self.assertEqual(await self._vivos(), 0)

    async def test_deadlock_que_no_cede_termina_en_error_no_en_reserva(self):
        """Fail-closed también cuando el reintento se agota: se levanta el
        error, jamás se devuelve `True` sin fila."""
        agotado = aiomysql.OperationalError(1213, "Deadlock found when trying to get lock")
        with patch.object(cupo, "_ejecutar_reserva", side_effect=agotado):
            with self.assertRaises(aiomysql.OperationalError):
                await cupo.reservar_cupo(_pipeline(PREFIJO + "trabado"), limite=3)
        self.assertEqual(await self._vivos(), 0)

    # ---- 5. un estado que el cupo no conoce es un cupo mal contado -------

    async def test_la_tabla_no_trae_un_estado_que_el_cupo_no_clasifica(self):
        """Contra la tabla REAL: si otro servicio u otra rama escribe un estado
        que `PipelineStatus` no conoce (el frente G trae `queued`,
        `awaiting_approval` y `waiting_children`), el cupo lo cuenta como "no
        ocupa" sin que nadie lo haya decidido. Esto se pone rojo antes."""
        from jacobs.models import PipelineStatus

        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT DISTINCT status FROM jacobs_pipelines")
                en_la_tabla = {fila[0] for fila in await cur.fetchall()}

        conocidos = {e.value for e in PipelineStatus}
        desconocidos = en_la_tabla - conocidos
        self.assertEqual(
            desconocidos, set(),
            f"jacobs_pipelines trae estados que jacobs/cupo.py no clasifica: "
            f"{sorted(desconocidos)}. Decidí si ocupan cupo antes de que lo decida el silencio.",
        )

    # ---- 6. el plan de la consulta REAL ---------------------------------

    async def test_explain_de_la_reserva_usa_el_indice_de_status(self):
        """EXPLAIN sobre la sentencia REAL (Principio I y política 1 de LAS
        CUATRO): un índice que existe no es un índice que se usa."""
        p = _pipeline(PREFIJO + "explain")
        async with store.conexion() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("EXPLAIN " + cupo.SQL_RESERVAR, cupo.parametros_de_reserva(p, 3))
                filas = await cur.fetchall()

        conteo = [f for f in filas if f["table"] == "jacobs_pipelines" and f["key"]]
        self.assertTrue(conteo, f"el COUNT del cupo no usa índice: {filas}")
        self.assertEqual(conteo[0]["key"], "idx_pipelines_status")
        for f in filas:
            extra = f.get("Extra") or ""
            self.assertNotIn("filesort", extra)
            self.assertNotIn("Using temporary", extra)


@unittest.skipUnless(os.getenv("JAX_DB_HOST"), "necesita la MariaDB real (jax_memory_test)")
class CreacionConcurrenteSinCandadoTest(unittest.IsolatedAsyncioTestCase):
    """El endpoint completo, con el candado global BORRADO: `POST
    /jacobs/pipeline` concurrente nunca supera `MAX_PARALLEL_PIPELINES`."""

    async def asyncSetUp(self):
        self.addAsyncCleanup(store.cerrar_pool)
        self.addAsyncCleanup(self._limpiar)
        await store.init_tables()
        await self._limpiar()
        self.assertEqual(await cupo.activos(), 0, "hay pipelines vivos ajenos en jax_memory_test")
        freno = patch("jacobs.policy.check_kill_switch", return_value=False)
        freno.start()
        self.addCleanup(freno.stop)

    async def _limpiar(self):
        await ada.ejecutar(
            "DELETE FROM jacobs_pipelines WHERE name LIKE %s", (PREFIJO + "%",)
        )

    async def test_diez_creaciones_a_la_vez_admiten_exactamente_el_cupo(self):
        from unittest.mock import AsyncMock

        from fastapi import BackgroundTasks, HTTPException

        from jacobs import routes
        from jacobs.models import PipelineCreateRequest
        from jacobs.policy import MAX_PARALLEL_PIPELINES

        async def plan_lento(pipeline_id, objective, max_steps, steps_spec):
            # El planificador REAL tarda 20-40 s. Acá 50 ms alcanzan para que
            # las diez creaciones se solapen: si quedara un candado global,
            # esto se serializaría y el test tardaría diez veces más.
            await asyncio.sleep(0.05)
            return await ada.plan_de_un_paso(pipeline_id, objective, max_steps, steps_spec)

        async def crear(i: int):
            req = PipelineCreateRequest.model_validate({
                "name": f"{PREFIJO}ruta-{i}", "objective": "o",
                "invoked_by": "plataforma", "mode": "autonomous",
            })
            try:
                return await routes.create_pipeline(req, BackgroundTasks())
            except HTTPException as exc:
                return exc

        with patch.object(routes, "_plan_builder") as builder, \
             patch.object(routes.store, "step_upsert", AsyncMock(return_value=None)):
            builder.build = AsyncMock(side_effect=plan_lento)
            comenzo = time.perf_counter()
            resultados = await asyncio.gather(*[crear(i) for i in range(10)])
            tardo = time.perf_counter() - comenzo

        creados = [r for r in resultados if isinstance(r, dict)]
        rechazos = [r for r in resultados if isinstance(r, HTTPException)]
        self.assertEqual(len(creados), MAX_PARALLEL_PIPELINES)
        self.assertEqual(len(rechazos), 10 - MAX_PARALLEL_PIPELINES)
        self.assertTrue(all(r.status_code == 422 for r in rechazos), rechazos)
        self.assertTrue(all("Límite duro" in str(r.detail) for r in rechazos), rechazos)
        self.assertEqual(await cupo.activos(), MAX_PARALLEL_PIPELINES)
        # Sin candado global las diez planificaciones se solapan: el total
        # tiene que parecerse a UNA, no a diez en fila (10 x 50 ms = 0,5 s).
        self.assertLess(tardo, 0.35, f"la creación sigue serializada: {tardo:.3f} s")

    async def test_un_plan_rechazado_devuelve_el_cupo(self):
        """Camino de fallo DESPUÉS de reservar: la reserva se suelta, el cupo
        no queda ocupado por un pipeline que nunca existió."""
        from unittest.mock import AsyncMock

        from fastapi import BackgroundTasks, HTTPException

        from jacobs import routes
        from jacobs.models import PipelineCreateRequest

        req = PipelineCreateRequest.model_validate({
            "name": PREFIJO + "plan-roto", "objective": "o",
            "invoked_by": "plataforma", "mode": "autonomous",
        })
        with patch.object(
            routes, "_build_plan_or_reject",
            AsyncMock(side_effect=HTTPException(status_code=422, detail="plan inejecutable")),
        ):
            with self.assertRaises(HTTPException):
                await routes.create_pipeline(req, BackgroundTasks())

        self.assertEqual(await cupo.activos(), 0, "la reserva quedó ocupando cupo")


if __name__ == "__main__":
    unittest.main()
