"""Consultas de SOLO LECTURA de proyectos para la plataforma (E1, 2026-10-02).

Nada aquí escribe ni toma candados. Cada función relee la identidad del
usuario (activo, del tenant) en la misma consulta: el token de la plataforma
dice quién pide, no qué puede ver. Las mutaciones viven en
`project_authority.ProjectAuthorityAdmin`.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from jax.memory.project_authority import ProjectNotVisible, ProjectRoleInsufficient
from jax.memory.scope_authority import TENANT_ADMIN_ROLES


class ProjectView(str, Enum):
    ACTIVOS = "activos"
    ARCHIVADOS = "archivados"
    OCULTOS = "ocultos"


#: DISABLED no tiene vista: no se lista para nadie (D3, igual que resolve_project_read).
_VIEW_STATUS = {ProjectView.ACTIVOS: "ACTIVE", ProjectView.ARCHIVADOS: "ARCHIVED", ProjectView.OCULTOS: "HIDDEN"}
_ADMIN_ROLES = tuple(sorted(TENANT_ADMIN_ROLES))
_ADMIN_PH = ",".join(["%s"] * len(_ADMIN_ROLES))

#: 101, no 100: la API pide `limite+1` para saber si hay página siguiente y
#: valida ella misma que `limite <= 100`.
_LIMITE_LISTA_MAX = 101

_SQL_LISTA = (
    "SELECT p.id AS project_id, p.project_uuid, p.name, p.description, s.status, m.project_role AS role "
    "FROM jax_project_membership m "
    "JOIN jax_project_scope s ON s.project_id=m.project_id AND s.tenant_id=m.tenant_id "
    "JOIN projects p ON p.id=m.project_id "
    "WHERE m.tenant_id=%s AND m.user_id=%s AND m.status='ACTIVE' AND s.status=%s "
    "AND (%s IS NULL OR m.project_id < %s) "
    "ORDER BY m.project_id DESC LIMIT %s"
)


def _v(row: Any, name: str, pos: int) -> Any:
    return row.get(name) if isinstance(row, dict) else row[pos]


async def _usuario(cur: Any, tenant_id: int, user_id: int) -> tuple[bool, bool]:
    """(activo_en_el_tenant, es_admin_del_tenant), leídos de la base."""
    await cur.execute("SELECT status, role FROM jax_users WHERE user_id=%s AND tenant_id=%s", (user_id, tenant_id))
    row = await cur.fetchone()
    if not row or str(_v(row, "status", 0)).lower() != "active":
        return False, False
    return True, str(_v(row, "role", 1) or "").lower() in TENANT_ADMIN_ROLES


def _fila(row: Any) -> dict:
    return {"project_id": int(_v(row, "project_id", 0)), "project_uuid": str(_v(row, "project_uuid", 1)),
            "name": _v(row, "name", 2), "description": _v(row, "description", 3),
            "status": str(_v(row, "status", 4)).upper(), "role": str(_v(row, "role", 5)).upper()}


async def list_projects_for_user(pool: Any, *, tenant_id: int, user_id: int, view: ProjectView,
                                 before_id: int | None, limit: int) -> list[dict]:
    limit = max(1, min(int(limit), _LIMITE_LISTA_MAX))
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            activo, es_admin = await _usuario(cur, tenant_id, user_id)
            if not activo:
                return []
            if view is ProjectView.OCULTOS and not es_admin:
                raise ProjectRoleInsufficient("hidden projects are visible only to a tenant administrator")
            await cur.execute(_SQL_LISTA, (tenant_id, user_id, _VIEW_STATUS[view], before_id, before_id, limit))
            return [_fila(r) for r in await cur.fetchall()]


async def _proyecto_visible(cur: Any, tenant_id: int, user_id: int, project_id: int) -> dict:
    activo, es_admin = await _usuario(cur, tenant_id, user_id)
    if not activo:
        raise ProjectNotVisible("actor is not an active tenant member")
    await cur.execute(
        "SELECT p.id AS project_id, p.project_uuid, p.name, p.description, s.status, m.project_role AS role "
        "FROM jax_project_membership m "
        "JOIN jax_project_scope s ON s.project_id=m.project_id AND s.tenant_id=m.tenant_id "
        "JOIN projects p ON p.id=m.project_id "
        "WHERE m.project_id=%s AND m.tenant_id=%s AND m.user_id=%s AND m.status='ACTIVE'",
        (project_id, tenant_id, user_id))
    row = await cur.fetchone()
    if not row:
        raise ProjectNotVisible("project not visible")
    fila = _fila(row)
    # Espejo, en SQL de solo lectura, de `ScopeAuthority.resolve_project_read`
    # (scope_authority.py, D3): ACTIVE/ARCHIVED para el miembro activo, HIDDEN
    # solo para el admin del tenant, DISABLED NUNCA (ni para el admin). No se
    # delega en ese resolvedor: este modulo trabaja con un cursor de la pool y
    # sin `ScopeContext`, y no debe cambiar firmas. Si la regla de allí cambia,
    # cambia aquí (hay pruebas de las dos mitades en test_project_queries_mariadb).
    if fila["status"] == "DISABLED" or (fila["status"] == "HIDDEN" and not es_admin):
        raise ProjectNotVisible("project not visible")
    return fila


async def get_project_for_user(pool: Any, *, tenant_id: int, user_id: int, project_id: int) -> dict:
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            return await _proyecto_visible(cur, tenant_id, user_id, project_id)


async def list_project_members(pool: Any, *, tenant_id: int, user_id: int, project_id: int) -> list[dict]:
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await _proyecto_visible(cur, tenant_id, user_id, project_id)
            await cur.execute(
                "SELECT m.user_id, u.email, m.project_role AS role, m.grant_origin "
                "FROM jax_project_membership m JOIN jax_users u ON u.user_id=m.user_id "
                "WHERE m.project_id=%s AND m.tenant_id=%s AND m.status='ACTIVE' ORDER BY u.email",
                (project_id, tenant_id))
            return [{"user_id": int(_v(r, "user_id", 0)), "email": _v(r, "email", 1),
                     "role": str(_v(r, "role", 2)).upper(), "grant_origin": str(_v(r, "grant_origin", 3))}
                    for r in await cur.fetchall()]


async def list_invite_candidates(pool: Any, *, tenant_id: int, user_id: int, project_id: int,
                                 query: str, limit: int) -> list[dict]:
    """Usuarios del tenant que el OWNER de un proyecto ACTIVE puede invitar.

    2026-10-02, decision de Fernando (Proyectos E1.1): la pestana Miembros muestra la
    lista de usuarios con casillas para marcar varios, y el buscador por email la
    filtra; ya no es una busqueda obligatoria. Por eso:
      - `query` vacio o solo espacios -> lista sin filtrar (rama de SQL sin LIKE).
      - `query` de 1 o mas caracteres -> filtro por prefijo de email (antes, menos de
        2 caracteres devolvia []).
      - `limit` se acota a 1..100 (antes 1..20).
    Sin cambios: excluye miembros ACTIVE, admins del tenant, inactivos y otros
    tenants; exige OWNER de un proyecto ACTIVE; escapa el LIKE; ordena por email.
    """
    query = (query or "").strip()
    limit = max(1, min(int(limit), 100))
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            fila = await _proyecto_visible(cur, tenant_id, user_id, project_id)
            if fila["role"] != "OWNER" or fila["status"] != "ACTIVE":
                raise ProjectRoleInsufficient("inviting requires OWNER on an ACTIVE project")
            base = (
                "SELECT u.user_id, u.email FROM jax_users u "
                "LEFT JOIN jax_project_membership m ON m.user_id=u.user_id AND m.project_id=%s AND m.status='ACTIVE' "
                f"WHERE u.tenant_id=%s AND u.status='active' AND LOWER(u.role) NOT IN ({_ADMIN_PH}) "
                "AND m.user_id IS NULL ")
            args: tuple = (project_id, tenant_id, *_ADMIN_ROLES)
            if query:
                patron = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
                await cur.execute(base + "AND u.email LIKE %s ORDER BY u.email LIMIT %s", (*args, patron, limit))
            else:
                await cur.execute(base + "ORDER BY u.email LIMIT %s", (*args, limit))
            return [{"user_id": int(_v(r, "user_id", 0)), "email": _v(r, "email", 1)} for r in await cur.fetchall()]
