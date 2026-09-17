"""Transacción de continuar y continue concurrente contra MariaDB real
(spec 2026-09-17 §5.2 regla 10, §8, §9). Job jacobs-gobernanza-db.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, patch

_db = os.environ.get("JAX_DB_NAME")
if _db and _db != "jax_memory_test":
    raise RuntimeError(f"JAX_DB_NAME={_db!r}: este archivo escribe filas y solo corre contra jax_memory_test.")
os.environ.setdefault("JAX_DB_NAME", "jax_memory_test")

import pytest  # noqa: E402

from jacobs import continuar, store  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus  # noqa: E402
from jacobs.prevuelo_reglas import Veredicto  # noqa: E402

_REF = 'inline:{"result": "hecho"}'


async def _abortado():
    await store.init_tables()
    pid = str(uuid.uuid4())
    pasos = [
        Step(pipeline_id=pid, step_index=i, facet="jekyll", capability="research",
             input={"prompt": f"p{i}"}, depends_on=[i - 1] if i else [],
             status=StepStatus.completed if i < 2 else StepStatus.failed,
             error=None if i < 2 else "cortado", output_ref=_REF if i < 2 else None)
        for i in range(3)
    ]
    ahora = time.time()
    pipeline = Pipeline(pipeline_id=pid, name="t-continuar", invoked_by="plataforma",
                        mode="autonomous", status=PipelineStatus.aborted, plan=pasos,
                        context={"objective": "o", "step_0_ref": _REF, "step_1_ref": _REF},
                        created_at=ahora, updated_at=ahora)
    await store.pipeline_create(pipeline)
    for paso in pasos:
        await store.step_upsert(paso)
    return pipeline, pasos


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


def test_la_transaccion_aplica_pasos_plan_contexto_y_epoca():
    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            pasos[2].facet = "thot"
            contexto = {"objective": "o", "step_0_ref": _REF, "step_1_ref": _REF}
            nueva = await store.continuar_transaccion(
                pid, 0, PipelineStatus.aborted, [pasos[2]], pasos, contexto, 2)
            p = await store.pipeline_get(pid)
            s2 = (await store.steps_by_pipeline(pid))[2]
            explains = [
                await _explain(store._SQL_BLOQUEAR_PIPELINE, (pid,)),
                await _explain(store._SQL_PASO_A_CORRER, ("thot", None, s2.step_id, pid)),
            ]
            return nueva, p, s2, explains
        finally:
            await _borrar(pid)
    nueva, p, s2, explains = asyncio.run(cuerpo())
    assert nueva == 1
    assert (p.status, p.run_epoch, p.current_step_index, p.plan[2].facet) == (
        PipelineStatus.running, 1, 2, "thot")
    assert (s2.status, s2.facet, s2.error, s2.output_ref) == (StepStatus.pending, "thot", None, None)
    for filas in explains:
        assert all(f["key"] == "PRIMARY" for f in filas), filas


def test_si_falla_a_mitad_no_cambia_nada(monkeypatch):
    monkeypatch.setattr(store, "_SQL_PIPELINE_CONTINUAR",
                        "UPDATE tabla_que_no_existe SET plan=%s, context_refs=%s, "
                        "current_step_index=%s, updated_at=%s WHERE pipeline_id=%s AND run_epoch=%s")

    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            pasos[2].facet = "thot"
            with pytest.raises(Exception):
                await store.continuar_transaccion(
                    pid, 0, PipelineStatus.aborted, [pasos[2]], pasos, pipeline.context, 2)
            return await store.pipeline_get(pid), (await store.steps_by_pipeline(pid))[2]
        finally:
            await _borrar(pid)
    p, s2 = asyncio.run(cuerpo())
    assert (p.status, p.run_epoch) == (PipelineStatus.aborted, 0)
    assert (s2.status, s2.facet, s2.error) == (StepStatus.failed, "jekyll", "cortado")


def test_dos_transacciones_con_la_misma_epoca_solo_una_gana():
    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            return await asyncio.gather(*(
                store.continuar_transaccion(pid, 0, PipelineStatus.aborted, [pasos[2]], pasos,
                                            pipeline.context, 2)
                for _ in range(2)
            ))
        finally:
            await _borrar(pid)
    resultados = asyncio.run(cuerpo())
    assert sorted(resultados, key=lambda r: r is None) == [1, None]


def test_continue_concurrente_sobre_el_mismo_pipeline_solo_uno_gana():
    ok = Veredicto(ok=True, violaciones=(), costo_max_usd=Decimal(0), pasos_costo=(), sondeadas=())

    async def cuerpo():
        pipeline, _ = await _abortado()
        pid = pipeline.pipeline_id
        try:
            with patch.object(continuar, "prevuelo", AsyncMock(return_value=ok)), \
                 patch.object(continuar, "check_kill_switch", return_value=False), \
                 patch.object(continuar.store, "pipeline_count_active", AsyncMock(return_value=0)):
                resultados = await asyncio.gather(
                    *(continuar.continuar(pid, "plataforma") for _ in range(10)),
                    return_exceptions=True)
            return resultados, await store.pipeline_epoca_y_status(pid)
        finally:
            await _borrar(pid)
    resultados, actual = asyncio.run(cuerpo())
    ganadores = [r for r in resultados if isinstance(r, tuple)]
    rechazos = [r for r in resultados if isinstance(r, continuar.ContinuarRechazado)]
    assert len(ganadores) == 1 and len(rechazos) == 9, resultados
    assert all(r.status_code == 409 for r in rechazos)
    assert actual == (1, PipelineStatus.running)
