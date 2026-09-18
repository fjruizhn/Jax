"""Reaper contra MariaDB real (job jacobs-gobernanza-db).

- Pasada final R34, 5: no cosecha un running que avanzó entre su lectura y
  su escritura (CAS con updated_at).
- Ruling R36: el umbral de un running es POR PIPELINE,
  max(RUNNING_STALE_SECONDS, 2 x el mayor timeout_seconds de sus pasos en
  running), leído de jacobs_steps en la MISMA consulta del barrido. Un paso
  sano de 30 min no escribe mientras corre: con el umbral global de 1800 s
  la época + el CAS convertían ese falso "expired" en una corrida cortada.

El barrido corre la consulta REAL sobre toda la base (sólo lectura) y se
queda con las filas de ESTE test: la base es compartida y las filas ajenas
no se escriben. Filas propias con UUID, borradas en finally.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from unittest.mock import patch

_db = os.environ.get("JAX_DB_NAME")
if _db and _db != "jax_memory_test":
    raise RuntimeError(f"JAX_DB_NAME={_db!r}: este archivo escribe filas y solo corre contra jax_memory_test.")
os.environ.setdefault("JAX_DB_NAME", "jax_memory_test")

from jacobs import reaper, store  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus  # noqa: E402


async def _running(hace_s: float, timeouts_en_curso=(), timeouts_otros=()):
    """Un pipeline running con updated_at de hace `hace_s` segundos, pasos
    running con esos timeout_seconds y pasos completed con los otros."""
    await store.init_tables()
    pid = str(uuid.uuid4())
    viejo = time.time() - hace_s
    await store.pipeline_create(Pipeline(
        pipeline_id=pid, name="t-reaper-cas", invoked_by="plataforma", mode="autonomous",
        status=PipelineStatus.running, created_at=viejo, updated_at=viejo, run_epoch=3))
    i = 0
    for estado, timeouts in ((StepStatus.running, timeouts_en_curso), (StepStatus.completed, timeouts_otros)):
        for t in timeouts:
            await store.step_upsert(Step(pipeline_id=pid, step_index=i, facet="jekyll", capability="research",
                                         input={"prompt": "p"}, status=estado, timeout_seconds=t))
            i += 1
    return pid


async def _ejecutar(sql, params=()):
    conn = await store.conexion_dedicada()
    try:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            return await cur.fetchall()
    finally:
        conn.close()


async def _borrar(pid):
    await _ejecutar("DELETE FROM jacobs_events WHERE pipeline_id=%s", (pid,))
    await _ejecutar("DELETE FROM jacobs_steps WHERE pipeline_id=%s", (pid,))
    await _ejecutar("DELETE FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))


def _es_mia(fila, pid):
    pipeline, _max_timeout = fila
    return pipeline.pipeline_id == pid


async def _barrer(pid, *, avanza_entre_lectura_y_escritura=False):
    """Barrido real. La lectura es la consulta de producción
    (store.candidatos_del_reaper) filtrada a la fila propia. Si `avanza`, el
    ejecutor escribe su avance justo después."""
    real = store.candidatos_del_reaper

    async def leer(estados):
        filas = [f for f in await real(estados) if _es_mia(f, pid)]
        if avanza_entre_lectura_y_escritura:
            await _ejecutar("UPDATE jacobs_pipelines SET updated_at=%s WHERE pipeline_id=%s", (time.time(), pid))
        return filas

    with patch.object(reaper.store, "candidatos_del_reaper", leer):
        cosechados = await reaper.reap_orphaned_pipelines()
    fila = await store.pipeline_get(pid)
    eventos = await _ejecutar("SELECT event_type FROM jacobs_events WHERE pipeline_id=%s", (pid,))
    return cosechados, fila.status, [e[0] for e in eventos]


def _con_fila(hace_s, **kw):
    avanza = kw.pop("avanza", False)

    async def todo():
        pid = await _running(hace_s, **kw)
        try:
            return await _barrer(pid, avanza_entre_lectura_y_escritura=avanza)
        finally:
            await _borrar(pid)
    return asyncio.run(todo())


def test_paso_de_1800s_en_curso_y_1900s_sin_avance_no_vence():
    cosechados, status, eventos = _con_fila(1900, timeouts_en_curso=(300, 1800))
    assert cosechados == []
    assert status == PipelineStatus.running
    assert "REAPED" not in eventos


def test_paso_de_1800s_en_curso_y_3700s_sin_avance_vence():
    cosechados, status, eventos = _con_fila(3700, timeouts_en_curso=(1800,))
    assert len(cosechados) == 1 and "umbral 3600s" in cosechados[0]["reason"]
    assert status == PipelineStatus.expired
    assert eventos == ["REAPED"]


def test_sin_pasos_en_curso_el_umbral_sigue_en_1800():
    """Los timeouts de pasos que NO están running no cuentan: ya no pueden
    estar trabajando sin escribir."""
    cosechados, status, eventos = _con_fila(1900, timeouts_otros=(1800,))
    assert len(cosechados) == 1 and f"umbral {reaper.RUNNING_STALE_SECONDS}s" in cosechados[0]["reason"]
    assert status == PipelineStatus.expired


def test_running_que_avanzo_entre_lectura_y_escritura_no_se_cosecha():
    cosechados, status, eventos = _con_fila(3700, timeouts_en_curso=(1800,), avanza=True)
    assert cosechados == []
    assert status == PipelineStatus.running
    assert "REAPED" not in eventos


def test_explain_del_update_con_corte_de_avance_usa_la_clave_primaria():
    async def todo():
        pid = await _running(3700)
        try:
            sql = store._sql_update_si_epoca(False, False, 1, con_corte=True)
            conn = await store.conexion_dedicada()
            try:
                async with conn.cursor() as cur:
                    await cur.execute("EXPLAIN " + sql, (
                        "expired", time.time(), pid, 3, "running", time.time() - reaper.RUNNING_STALE_SECONDS))
                    cols = [d[0] for d in cur.description]
                    return sql, [dict(zip(cols, r)) for r in await cur.fetchall()]
            finally:
                conn.close()
        finally:
            await _borrar(pid)
    sql, filas = asyncio.run(todo())
    assert "updated_at < %s" in sql
    assert filas
    for f in filas:
        assert f["key"] == "PRIMARY" and f["type"] not in ("ALL", "index"), filas
        assert "filesort" not in (f.get("Extra") or "") and "temporary" not in (f.get("Extra") or ""), filas


def test_explain_de_la_consulta_del_barrido():
    """R36, LAS CUATRO #1: el barrido filtra jacobs_pipelines por status
    (idx_pipelines_status) y la subconsulta correlacionada lee los pasos de
    CADA candidato por idx_steps_pipeline (ref por pipeline_id; el filtro por
    status corre sobre a lo sumo 20 pasos, el tope duro de un plan)."""
    async def todo():
        sql = store._sql_candidatos_del_reaper(3)
        conn = await store.conexion_dedicada()
        try:
            async with conn.cursor() as cur:
                await cur.execute("EXPLAIN " + sql, ("pending", "running", "interrupted"))
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r)) for r in await cur.fetchall()]
        finally:
            conn.close()
    filas = asyncio.run(todo())
    por_tabla = {f["table"]: f for f in filas}
    assert set(por_tabla) == {"p", "s"}, filas
    assert por_tabla["p"]["key"] == "idx_pipelines_status" and por_tabla["p"]["type"] == "range", filas
    assert por_tabla["s"]["key"] in ("idx_steps_pipeline",) and por_tabla["s"]["type"] == "ref", filas
    for f in filas:
        assert "filesort" not in (f.get("Extra") or "") and "temporary" not in (f.get("Extra") or ""), filas
