"""E2a / T2: `las_manos.proyecto_activo.estado_del_proyecto` contra MariaDB real.

Reusa los ayudantes de `test_project_authority_mariadb.py` (mismo patron que
`test_project_documents_schema_mariadb.py`). La consulta pasa por el pool REAL
de LAS MANOS (`jacobs.store.conexion()`), apuntado a la base de pruebas.
"""
from __future__ import annotations

import uuid

import pytest

from jacobs import store as jacobs_store
from test_project_authority_mariadb import (  # noqa: F401  (el fixture se registra por importacion)
    _crear_proyecto_activo, _crear_scope, _crear_tenant, _esquema_de_proyectos,
    _sql, asincrono, requiere_db_de_prueba,
)

pytestmark = requiere_db_de_prueba


async def _proyecto_con_scope(status: str) -> str:
    tenant = await _crear_tenant("pa")
    project_id = await _crear_proyecto_activo(tenant, name="PA")
    await _crear_scope(project_id, tenant, status)
    fila = (await _sql("SELECT project_uuid FROM projects WHERE id=%s", (project_id,), fetch=True))[0]
    return fila["project_uuid"]


async def _estado(project_uuid: str):
    import proyecto_activo
    try:
        return await proyecto_activo.estado_del_proyecto(project_uuid)
    finally:
        await jacobs_store.cerrar_pool()


@asincrono
async def test_scope_active_da_active():
    assert await _estado(await _proyecto_con_scope("ACTIVE")) == "ACTIVE"


@asincrono
async def test_scope_archived_da_archived():
    assert await _estado(await _proyecto_con_scope("ARCHIVED")) == "ARCHIVED"


@asincrono
async def test_uuid_inexistente_da_none():
    assert await _estado(str(uuid.uuid4())) is None
