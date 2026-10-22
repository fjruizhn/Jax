"""¿Existe el proyecto y está ACTIVE? (E2a, adenda E2 §3.2). LAS MANOS NO verifica
membresía: solo la credencial de plataforma llama a /procesamiento y jax-platform
ya verificó el papel. Esto solo impide trabajar sobre un proyecto archivado,
oculto o inexistente. Usa `idx` único de projects.project_uuid y la PK de scope."""
from __future__ import annotations

from jacobs import store as jacobs_store
from processing_ownership import ProcessingOwnershipContext

_CONSULTA = (
    "SELECT s.status FROM projects p JOIN jax_project_scope s ON s.project_id = p.id "
    "WHERE p.project_uuid = %s"
)


async def estado_del_proyecto(project_uuid: str) -> str | None:
    async with jacobs_store.conexion() as conn:
        async with conn.cursor() as cur:
            await cur.execute(_CONSULTA, (project_uuid,))
            fila = await cur.fetchone()
    if fila is None:
        return None
    valor = fila["status"] if isinstance(fila, dict) else fila[0]
    return str(valor).upper()


async def identidad_activa_del_proyecto(project_uuid: str, ownership: ProcessingOwnershipContext) -> bool:
    query = (
        "SELECT p.id, s.tenant_id FROM projects p "
        "JOIN jax_project_scope s ON s.project_id=p.id "
        "WHERE p.project_uuid=%s AND s.status='ACTIVE'"
    )
    async with jacobs_store.conexion() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, (project_uuid,))
            row = await cur.fetchone()
    if row is None:
        return False
    project_id, tenant_id = (row["id"], row["tenant_id"]) if isinstance(row, dict) else row
    return (str(project_id), str(tenant_id)) == (ownership.project_id, ownership.tenant_id)
