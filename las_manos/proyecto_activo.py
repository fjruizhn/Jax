"""¿Existe el proyecto y está ACTIVE? (E2a, adenda E2 §3.2). LAS MANOS NO verifica
membresía: solo la credencial de plataforma llama a /procesamiento y jax-platform
ya verificó el papel. Esto solo impide trabajar sobre un proyecto archivado,
oculto o inexistente. Usa `idx` único de projects.project_uuid y la PK de scope."""
from __future__ import annotations

from jacobs import store as jacobs_store

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
