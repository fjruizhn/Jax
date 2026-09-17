"""Época de corrida (spec 2026-09-17 §5.3) contra MariaDB real: la columna, las
escrituras condicionales y el incremento atómico. Job jacobs-gobernanza-db.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid

_db = os.environ.get("JAX_DB_NAME")
if _db and _db != "jax_memory_test":
    raise RuntimeError(
        f"JAX_DB_NAME={_db!r}: este archivo escribe filas y solo corre contra jax_memory_test."
    )
os.environ.setdefault("JAX_DB_NAME", "jax_memory_test")

from jacobs import store  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus  # noqa: E402


async def _crear(status=PipelineStatus.running, epoca=0):
    await store.init_tables()
    pid = str(uuid.uuid4())
    paso = Step(pipeline_id=pid, step_index=0, facet="jekyll", capability="research",
                input={"prompt": "p"})
    ahora = time.time()
    pipeline = Pipeline(pipeline_id=pid, name="t-epoca", invoked_by="plataforma",
                        mode="autonomous", status=status, plan=[paso],
                        context={"objective": "o"}, created_at=ahora, updated_at=ahora,
                        run_epoch=epoca)
    await store.pipeline_create(pipeline)
    await store.step_upsert(paso)
    return pipeline, paso


async def _borrar(pid):
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM jacobs_steps WHERE pipeline_id=%s", (pid,))
            await cur.execute("DELETE FROM jacobs_events WHERE pipeline_id=%s", (pid,))
            await cur.execute("DELETE FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))
    finally:
        conn.close()


async def _explain(sql, params):
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("EXPLAIN " + sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in await cur.fetchall()]
    finally:
        conn.close()


def test_init_tables_crea_run_epoch_con_default_cero():
    async def cuerpo():
        await store.init_tables()
        conn = await store.get_conn()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COLUMN_DEFAULT, IS_NULLABLE, DATA_TYPE FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='jacobs_pipelines' "
                    "AND COLUMN_NAME='run_epoch'")
                return await cur.fetchone()
        finally:
            conn.close()
    fila = asyncio.run(cuerpo())
    assert fila is not None, "jacobs_pipelines no tiene run_epoch"
    assert (str(fila[0]), fila[1], fila[2]) == ("0", "NO", "int")


def test_update_condicional_respeta_epoca_y_status():
    async def cuerpo():
        pipeline, _ = await _crear(status=PipelineStatus.running, epoca=2)
        pid = pipeline.pipeline_id
        try:
            otra_epoca = await store.pipeline_update_status_si_epoca(pid, 1, PipelineStatus.aborted)
            otro_status = await store.pipeline_update_status_si_epoca(
                pid, 2, PipelineStatus.completed, desde=(PipelineStatus.pending,))
            vigente = await store.pipeline_update_status_si_epoca(pid, 2, PipelineStatus.aborted)
            releido = await store.pipeline_get(pid)
            return otra_epoca, otro_status, vigente, releido.status, releido.run_epoch
        finally:
            await _borrar(pid)
    assert asyncio.run(cuerpo()) == (False, False, True, PipelineStatus.aborted, 2)


def test_step_upsert_si_epoca_solo_con_la_epoca_vigente_y_running():
    async def cuerpo():
        pipeline, paso = await _crear(status=PipelineStatus.running, epoca=1)
        pid = pipeline.pipeline_id
        try:
            paso.status = StepStatus.completed
            ajena = await store.step_upsert_si_epoca(paso, 0)
            propia = await store.step_upsert_si_epoca(paso, 1)
            tras_propia = (await store.steps_by_pipeline(pid))[0].status
            await store.pipeline_update_status_si_epoca(pid, 1, PipelineStatus.aborted)
            paso.status = StepStatus.failed
            paso.error = "tarde"
            no_running = await store.step_upsert_si_epoca(paso, 1)
            final = (await store.steps_by_pipeline(pid))[0]
            return ajena, propia, tras_propia, no_running, final.status, final.error
        finally:
            await _borrar(pid)
    assert asyncio.run(cuerpo()) == (
        False, True, StepStatus.completed, False, StepStatus.completed, None)


def test_tomar_epoca_una_sola_vez():
    async def cuerpo():
        pipeline, _ = await _crear(status=PipelineStatus.interrupted, epoca=5)
        pid = pipeline.pipeline_id
        try:
            resultados = await asyncio.gather(
                store.pipeline_tomar_epoca(pid, 5, (PipelineStatus.interrupted,)),
                store.pipeline_tomar_epoca(pid, 5, (PipelineStatus.interrupted,)),
            )
            return sorted(resultados, key=lambda r: r is None), await store.pipeline_epoca_y_status(pid)
        finally:
            await _borrar(pid)
    resultados, actual = asyncio.run(cuerpo())
    assert resultados == [6, None]
    assert actual == (6, PipelineStatus.interrupted)


def test_step_upsert_actualiza_la_faceta():
    async def cuerpo():
        pipeline, paso = await _crear()
        try:
            paso.facet = "thot"
            await store.step_upsert(paso)
            return (await store.steps_by_pipeline(pipeline.pipeline_id))[0].facet
        finally:
            await _borrar(pipeline.pipeline_id)
    assert asyncio.run(cuerpo()) == "thot"


def test_explain_de_las_consultas_de_epoca_usa_la_clave_primaria():
    async def cuerpo():
        pipeline, paso = await _crear(status=PipelineStatus.running, epoca=0)
        pid = pipeline.pipeline_id
        try:
            lectura = await _explain(store._SQL_EPOCA_Y_STATUS, (pid,))
            update = await _explain(store._sql_update_si_epoca(False, False, 1),
                                    ("aborted", time.time(), pid, 0, "running"))
            pasos = await _explain(store._SQL_STEP_SI_EPOCA, (
                "completed", "jekyll", None, None, 300, None, None, None, paso.step_id, pid, 0))
            tomar = await _explain(store._sql_tomar_epoca(False, 1), (time.time(), pid, 0, "running"))
            return lectura, update, pasos, tomar
        finally:
            await _borrar(pid)
    for filas in asyncio.run(cuerpo()):
        assert filas, "EXPLAIN vacío"
        assert all(f["key"] == "PRIMARY" for f in filas), filas
        assert all("filesort" not in (f.get("Extra") or "") and
                   "temporary" not in (f.get("Extra") or "") for f in filas), filas
