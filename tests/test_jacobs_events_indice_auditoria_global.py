"""Índices de soporte para el feed admin de auditoría de descartes.

El feed consulta `jacobs_events` por tipo, ordena por `ts, id` y, cuando se
filtra por pipeline, usa también `pipeline_id`. El tenant se resuelve desde
`jacobs_pipelines` en la plataforma. Estos índices evitan filesort en ambos
caminos.
"""
from __future__ import annotations

import asyncio

from base_de_test import exigir_base_de_test

exigir_base_de_test()

from jacobs import store  # noqa: E402


async def _columnas_de_indice(nombre: str) -> list[str]:
    await store.init_tables()
    conn = await store.conexion_dedicada()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT COLUMN_NAME FROM information_schema.STATISTICS "
                "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='jacobs_events' "
                "AND INDEX_NAME=%s ORDER BY SEQ_IN_INDEX",
                (nombre,),
            )
            return [fila[0] for fila in await cur.fetchall()]
    finally:
        conn.close()


def test_init_tables_crea_indice_global_por_tipo_fecha_y_id():
    columnas = asyncio.run(_columnas_de_indice("idx_events_auditoria_fecha"))
    assert columnas == ["event_type", "ts", "id"], columnas


def test_init_tables_crea_indice_por_pipeline_tipo_fecha_y_id():
    columnas = asyncio.run(_columnas_de_indice("idx_events_pipeline_auditoria_fecha"))
    assert columnas == ["pipeline_id", "event_type", "ts", "id"], columnas
