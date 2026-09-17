"""El reaper no cosecha un running que avanzó entre su lectura y su escritura
(pasada final R34, 5). Contra MariaDB real; job jacobs-gobernanza-db.

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
from jacobs.models import Pipeline, PipelineStatus  # noqa: E402


async def _running_estancado():
    await store.init_tables()
    pid = str(uuid.uuid4())
    viejo = time.time() - reaper.RUNNING_STALE_SECONDS - 120
    await store.pipeline_create(Pipeline(
        pipeline_id=pid, name="t-reaper-cas", invoked_by="plataforma", mode="autonomous",
        status=PipelineStatus.running, created_at=viejo, updated_at=viejo, run_epoch=3))
    return pid


async def _ejecutar(sql, params=()):
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            return await cur.fetchall()
    finally:
        conn.close()


async def _borrar(pid):
    await _ejecutar("DELETE FROM jacobs_events WHERE pipeline_id=%s", (pid,))
    await _ejecutar("DELETE FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))


def _cosechar_solo(pid, *, avanza_entre_lectura_y_escritura):
    """El barrido real sobre UNA fila propia (pipelines_by_status devuelve sólo
    la de este test: la base es compartida). Si `avanza`, el ejecutor escribe
    su avance justo después de la lectura del reaper."""
    async def leer(_estados):
        p = await store.pipeline_get(pid)
        if avanza_entre_lectura_y_escritura:
            await _ejecutar("UPDATE jacobs_pipelines SET updated_at=%s WHERE pipeline_id=%s",
                            (time.time(), pid))
        return [p]

    async def cuerpo():
        with patch.object(reaper.store, "pipelines_by_status", leer):
            cosechados = await reaper.reap_orphaned_pipelines()
        fila = await store.pipeline_get(pid)
        eventos = await _ejecutar(
            "SELECT event_type FROM jacobs_events WHERE pipeline_id=%s", (pid,))
        return cosechados, fila.status, [e[0] for e in eventos]
    return cuerpo


def test_running_que_avanzo_entre_lectura_y_escritura_no_se_cosecha():
    async def todo():
        pid = await _running_estancado()
        try:
            return await _cosechar_solo(pid, avanza_entre_lectura_y_escritura=True)()
        finally:
            await _borrar(pid)
    cosechados, status, eventos = asyncio.run(todo())
    assert cosechados == []
    assert status == PipelineStatus.running
    assert "REAPED" not in eventos


def test_running_estancado_de_verdad_se_cosecha():
    async def todo():
        pid = await _running_estancado()
        try:
            return await _cosechar_solo(pid, avanza_entre_lectura_y_escritura=False)()
        finally:
            await _borrar(pid)
    cosechados, status, eventos = asyncio.run(todo())
    assert len(cosechados) == 1
    assert status == PipelineStatus.expired
    assert eventos == ["REAPED"]


def test_explain_del_update_con_corte_de_avance_usa_la_clave_primaria():
    async def todo():
        pid = await _running_estancado()
        try:
            sql = store._sql_update_si_epoca(False, False, 1, con_corte=True)
            conn = await store.get_conn()
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
