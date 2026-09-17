"""Transacción de continuar y continue concurrente contra MariaDB real
(spec 2026-09-17 §5.2 regla 10, §8, §9). Job jacobs-gobernanza-db.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import aiomysql

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
    conn = await store.conexion_dedicada()
    try:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM jacobs_steps WHERE pipeline_id=%s", (pid,))
            await cur.execute("DELETE FROM jacobs_events WHERE pipeline_id=%s", (pid,))
            await cur.execute("DELETE FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))
    finally:
        conn.close()


async def _explain(sql, params):
    conn = await store.conexion_dedicada()
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
                pid, 0, PipelineStatus.aborted, [pasos[2]], pasos, contexto, 2,
                evento_payload=None)
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
        # type != index: en un UPDATE, un recorrido completo de PRIMARY
        # también dice key='PRIMARY' (ola final F7).
        assert all(f["key"] == "PRIMARY" and f["type"] not in ("ALL", "index") for f in filas), filas


def test_explain_del_update_final_de_continuar_usa_la_clave_primaria():
    """Ola final F7 (revisión final m8): la tercera consulta de la
    transacción, _SQL_PIPELINE_CONTINUAR, no tenía EXPLAIN al lado de las
    otras dos (LAS CUATRO #1). Se explica con parámetros reales."""
    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            return await _explain(store._SQL_PIPELINE_CONTINUAR, (
                json.dumps([s.model_dump() for s in pasos], ensure_ascii=False),
                json.dumps(pipeline.context, ensure_ascii=False), 2, time.time(), pid, 0,
            ))
        finally:
            await _borrar(pid)
    filas = asyncio.run(cuerpo())
    assert filas, "EXPLAIN vacío"
    for f in filas:
        # En un UPDATE, un recorrido COMPLETO de la clave primaria sale como
        # type='index' con key='PRIMARY' (medido: 5.810 filas con el WHERE
        # mutado a `name=%s`). Mirar sólo `key` no lo distingue.
        assert f["type"] not in ("ALL", "index") and f["key"] == "PRIMARY", filas
        assert "filesort" not in (f.get("Extra") or "") and "temporary" not in (f.get("Extra") or ""), filas


def test_si_falla_a_mitad_no_cambia_nada(monkeypatch):
    monkeypatch.setattr(store, "_SQL_PIPELINE_CONTINUAR",
                        "UPDATE tabla_que_no_existe SET plan=%s, context_refs=%s, "
                        "current_step_index=%s, updated_at=%s WHERE pipeline_id=%s AND run_epoch=%s")

    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            pasos[2].facet = "thot"
            evento = {"by": "plataforma", "from_status": "aborted", "run_epoch": 1,
                      "pasos_a_correr": [2], "pasos_reusados": [0, 1], "reasignados": {},
                      "costo_max_usd": "0.000000"}
            # ProgrammingError específico (1146, tabla inexistente), no
            # Exception genérico -- fix round 2: un catch-all también
            # atraparía un TypeError ajeno (p. ej. un argumento mal armado
            # en esta misma llamada) y el test daría un falso verde.
            with pytest.raises(aiomysql.ProgrammingError):
                await store.continuar_transaccion(
                    pid, 0, PipelineStatus.aborted, [pasos[2]], pasos, pipeline.context, 2,
                    evento_payload=evento)
            return (await store.pipeline_get(pid), (await store.steps_by_pipeline(pid))[2],
                    await store.events_by_pipeline(pid))
        finally:
            await _borrar(pid)
    p, s2, eventos = asyncio.run(cuerpo())
    assert (p.status, p.run_epoch) == (PipelineStatus.aborted, 0)
    assert (s2.status, s2.facet, s2.error) == (StepStatus.failed, "jekyll", "cortado")
    # Regla 10 / Ruling R22: la transacción caída no deja NINGÚN rastro, ni
    # siquiera el evento de auditoría -- si el evento se escribiera con otra
    # conexión (event_append() fuera de esta transacción) sobreviviría al
    # rollback y este assert lo vería.
    assert eventos == []


def test_el_evento_continued_se_escribe_con_la_misma_transaccion():
    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            pasos[2].facet = "thot"
            evento = {"by": "plataforma", "from_status": "aborted", "run_epoch": 1,
                      "pasos_a_correr": [2], "pasos_reusados": [0, 1], "reasignados": {},
                      "costo_max_usd": "0.000000"}
            nueva = await store.continuar_transaccion(
                pid, 0, PipelineStatus.aborted, [pasos[2]], pasos, pipeline.context, 2,
                evento_payload=evento)
            eventos = await store.events_by_pipeline(pid)
            return nueva, eventos
        finally:
            await _borrar(pid)
    nueva, eventos = asyncio.run(cuerpo())
    assert nueva == 1
    assert [e["event_type"] for e in eventos] == ["PIPELINE_CONTINUED"]
    assert eventos[0]["payload"] == {"by": "plataforma", "from_status": "aborted", "run_epoch": 1,
                                     "pasos_a_correr": [2], "pasos_reusados": [0, 1], "reasignados": {},
                                     "costo_max_usd": "0.000000"}


def test_si_el_evento_falla_no_queda_el_pipeline_a_medias(monkeypatch):
    # Ruling R22: si la escritura del evento fallara DESPUÉS del commit (como
    # hacía 45d60ec con event_append() por fuera), el pipeline y los pasos
    # quedarían escritos y el evento perdido -- un estado a medias. Con el
    # evento adentro de la MISMA transacción, una falla ahí también hace
    # ROLLBACK de todo: nada cambia, ni el pipeline, ni los pasos.
    monkeypatch.setattr(store, "_SQL_EVENTO_CONTINUED",
                        "INSERT INTO tabla_que_no_existe (pipeline_id, step_id, event_type, payload, ts) "
                        "VALUES (%s,%s,%s,%s,%s)")

    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            pasos[2].facet = "thot"
            evento = {"by": "plataforma", "from_status": "aborted", "run_epoch": 1,
                      "pasos_a_correr": [2], "pasos_reusados": [0, 1], "reasignados": {},
                      "costo_max_usd": "0.000000"}
            # ProgrammingError específico (1146, tabla inexistente) -- misma
            # razón que arriba (fix round 2): un Exception genérico no
            # discrimina.
            with pytest.raises(aiomysql.ProgrammingError):
                await store.continuar_transaccion(
                    pid, 0, PipelineStatus.aborted, [pasos[2]], pasos, pipeline.context, 2,
                    evento_payload=evento)
            return (await store.pipeline_get(pid), (await store.steps_by_pipeline(pid))[2],
                    await store.events_by_pipeline(pid))
        finally:
            await _borrar(pid)
    p, s2, eventos = asyncio.run(cuerpo())
    assert (p.status, p.run_epoch) == (PipelineStatus.aborted, 0)
    assert (s2.status, s2.facet, s2.error) == (StepStatus.failed, "jekyll", "cortado")
    assert eventos == []


def test_status_distinto_con_la_misma_epoca_no_gana():
    # Requisito (a) — mutación: si el chequeo sólo mirara la época y no el
    # status, esta llamada (época correcta, status equivocado a propósito)
    # ganaría la transacción igual.
    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            pasos[2].facet = "thot"
            nueva = await store.continuar_transaccion(
                pid, 0, PipelineStatus.expired, [pasos[2]], pasos, pipeline.context, 2,
                evento_payload=None)
            return nueva, await store.pipeline_get(pid), (await store.steps_by_pipeline(pid))[2]
        finally:
            await _borrar(pid)
    nueva, p, s2 = asyncio.run(cuerpo())
    assert nueva is None
    assert (p.status, p.run_epoch) == (PipelineStatus.aborted, 0)
    assert (s2.status, s2.facet) == (StepStatus.failed, "jekyll")


def test_si_el_update_final_no_toca_la_fila_no_queda_nada(monkeypatch):
    # Ruling R23 -- cinturón: aunque el SELECT...FOR UPDATE ya vio la fila
    # con la época correcta, mutamos el UPDATE final para que no toque
    # ninguna fila (condición imposible añadida) SIN lanzar excepción. Si el
    # código no revisara rowcount, devolvería la época nueva con nada escrito.
    monkeypatch.setattr(store, "_SQL_PIPELINE_CONTINUAR", store._SQL_PIPELINE_CONTINUAR + " AND 1=0")

    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            pasos[2].facet = "thot"
            nueva = await store.continuar_transaccion(
                pid, 0, PipelineStatus.aborted, [pasos[2]], pasos, pipeline.context, 2,
                evento_payload=None)
            return nueva, await store.pipeline_get(pid), (await store.steps_by_pipeline(pid))[2]
        finally:
            await _borrar(pid)
    nueva, p, s2 = asyncio.run(cuerpo())
    assert nueva is None
    assert (p.status, p.run_epoch) == (PipelineStatus.aborted, 0)
    # Rollback COMPLETO: ni siquiera el UPDATE de jacobs_steps quedó.
    assert (s2.status, s2.facet, s2.error) == (StepStatus.failed, "jekyll", "cortado")


def test_evento_payload_es_obligatorio():
    # Principio IX / fix round 2: el evento de auditoría no es opcional por
    # descuido. `evento_payload` no tiene default -- omitirlo es un TypeError
    # en el sitio de la llamada, no un pipeline continuado sin rastro. El
    # binding de argumentos de Python ocurre al invocar la corrutina, antes
    # de que corra una sola línea del cuerpo (y antes de tocar la DB), así
    # que ni hace falta awaitear ni armar un pipeline real para verlo.
    with pytest.raises(TypeError):
        store.continuar_transaccion(
            "cualquier-id", 0, PipelineStatus.aborted, [], [], {}, 0)


def test_dos_transacciones_con_la_misma_epoca_solo_una_gana():
    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            return await asyncio.gather(*(
                store.continuar_transaccion(pid, 0, PipelineStatus.aborted, [pasos[2]], pasos,
                                            pipeline.context, 2, evento_payload=None)
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
    assert len(ganadores) == 2 and len(rechazos) == 9, resultados
    assert all(r.status_code == 409 for r in rechazos)
    assert actual == (1, PipelineStatus.running)
