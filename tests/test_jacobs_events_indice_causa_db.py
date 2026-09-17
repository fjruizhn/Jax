"""Índice compuesto de causas en `jacobs_events` (Ruling R20, 2026-09-17, Task 12b).

Origen (LAS CUATRO #1): la Mesa (jax-platform `backend/api/pipelines.py`,
`sql_eventos_de_causa`/`EVENTOS_DE_CAUSA`) lee la causa de aborto de hasta 50
pipelines con UNA consulta:
    SELECT pipeline_id, id, event_type, payload FROM jacobs_events
    WHERE pipeline_id IN (...) AND event_type IN
        ('STEP_FAILED','PIPELINE_ABORTED','PIPELINE_CANCELLED',
         'KILL_SWITCH_ABORTED','REAPED')
Medido por el plan P en jax_memory_test (50 pipelines detenidos, 11.050
eventos): EXPLAIN range sobre idx_events_pipeline (solo pipeline_id), 11.050
filas examinadas para devolver 1.050, 8,1 ms; carga c=25 p95 147 ms.
Evidencia: jax-platform-prevuelo/.superpowers/sdd/
2026-09-17-prevuelo-y-continuar-mesa/task-12-report.md ("Fix round 1").

Este archivo NO importa de jax-platform (limites.md, "Trabaja SOLO en tu
worktree"): la forma de la consulta se copia acá, literal, como contrato
propio de jax con esa consulta.

Job jacobs-gobernanza-db. En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid

_db = os.environ.get("JAX_DB_NAME")
if _db and _db != "jax_memory_test":
    raise RuntimeError(
        f"JAX_DB_NAME={_db!r}: este archivo siembra eventos y solo corre contra jax_memory_test."
    )
os.environ.setdefault("JAX_DB_NAME", "jax_memory_test")

from jacobs import store  # noqa: E402

# Copia literal de la forma de jax-platform backend/api/pipelines.py
# (EVENTOS_DE_CAUSA / sql_eventos_de_causa), leída 2026-09-17.
_TIPOS_DE_CAUSA = (
    "STEP_FAILED", "PIPELINE_ABORTED", "PIPELINE_CANCELLED",
    "KILL_SWITCH_ABORTED", "REAPED",
)


def _sql_eventos_de_causa(n_ids: int) -> str:
    ids = ", ".join(["%s"] * n_ids)
    tipos = ", ".join(["%s"] * len(_TIPOS_DE_CAUSA))
    return (f"SELECT pipeline_id, id, event_type, payload FROM jacobs_events "
            f"WHERE pipeline_id IN ({ids}) AND event_type IN ({tipos})")


# Volumen (patrón del caso medido por el plan P: 50 pipelines / 11.050
# eventos): 20 pipelines marca-uuid, 200 eventos no-causa + 2 de causa cada
# uno (4.040 filas) -- selectivo a favor del compuesto sobre idx_events_pipeline,
# que sólo filtra por pipeline_id y deja los 200 no-causa por pipeline para
# examinar fila por fila.
N_PIPELINES = 20
EVENTOS_NO_CAUSA_POR_PIPELINE = 200
EVENTOS_CAUSA_POR_PIPELINE = 2


async def _sembrar() -> list[str]:
    ids = [str(uuid.uuid4()) for _ in range(N_PIPELINES)]
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            ahora = time.time()
            filas = []
            for pid in ids:
                for _ in range(EVENTOS_NO_CAUSA_POR_PIPELINE):
                    filas.append((pid, None, "STEP_STARTED", "{}", ahora))
                for _ in range(EVENTOS_CAUSA_POR_PIPELINE):
                    filas.append((pid, None, "STEP_FAILED", '{"step_index": 0}', ahora))
            await cur.executemany(
                "INSERT INTO jacobs_events (pipeline_id, step_id, event_type, payload, ts) "
                "VALUES (%s,%s,%s,%s,%s)",
                filas,
            )
    finally:
        conn.close()
    return ids


async def _borrar(ids: list[str]) -> None:
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            marcador = ", ".join(["%s"] * len(ids))
            await cur.execute(
                f"DELETE FROM jacobs_events WHERE pipeline_id IN ({marcador})",
                tuple(ids),
            )
    finally:
        conn.close()


async def _explain_json(sql: str, params: tuple) -> dict:
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("EXPLAIN FORMAT=JSON " + sql, params)
            (doc,) = await cur.fetchone()
            return json.loads(doc)
    finally:
        conn.close()


def test_init_tables_crea_idx_events_pipeline_tipo():
    async def cuerpo():
        await store.init_tables()
        conn = await store.get_conn()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COLUMN_NAME, SEQ_IN_INDEX FROM information_schema.STATISTICS "
                    "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='jacobs_events' "
                    "AND INDEX_NAME='idx_events_pipeline_tipo' ORDER BY SEQ_IN_INDEX"
                )
                return await cur.fetchall()
        finally:
            conn.close()
    filas = asyncio.run(cuerpo())
    assert list(filas) == [("pipeline_id", 1), ("event_type", 2)], filas


def test_explain_de_la_consulta_de_causas_usa_el_compuesto():
    async def cuerpo():
        await store.init_tables()
        ids = await _sembrar()
        try:
            sql = _sql_eventos_de_causa(len(ids))
            params = (*ids, *_TIPOS_DE_CAUSA)
            return await _explain_json(sql, params)
        finally:
            await _borrar(ids)
    plan = asyncio.run(cuerpo())
    # MariaDB envuelve incluso una consulta de una sola tabla en
    # nested_loop=[{table: {...}}] (verificado 2026-09-17 contra el código
    # viejo, sin el compuesto: key=idx_events_pipeline, used_key_parts sólo
    # pipeline_id, attached_condition con event_type sin usar el índice).
    tabla = plan["query_block"]["nested_loop"][0]["table"]
    assert tabla["access_type"] != "ALL", plan
    assert tabla.get("key") == "idx_events_pipeline_tipo", plan
    assert "event_type" in tabla.get("used_key_parts", []), plan
    # 4.040 filas sembradas, 40 de causa: el plan tiene que acercarse a las
    # devueltas, no a examinar los 200 no-causa por pipeline.
    assert tabla.get("rows", 10**9) < 400, plan
    extra = json.dumps(plan)
    assert "filesort" not in extra.lower(), plan


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
