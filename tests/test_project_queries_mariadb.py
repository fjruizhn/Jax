"""E1 / T2: consultas de lectura de proyectos contra MariaDB real.

Reusa los ayudantes de `test_project_authority_mariadb.py` (el repo ya importa
entre tests) y su fixture de esquema.
"""
from __future__ import annotations

import uuid

import pytest

from jax.memory.project_authority import ProjectNotVisible, ProjectRoleInsufficient
from jax.memory.project_queries import (
    _SQL_LISTA as _SQL_LISTA_PARA_EXPLAIN,
    ProjectView, get_project_for_user, list_invite_candidates, list_project_members, list_projects_for_user,
)
from test_project_authority_mariadb import (  # noqa: F401  (el fixture se registra por importacion)
    _crear_membresia, _crear_proyecto_activo, _crear_scope, _crear_tenant, _crear_usuario,
    _esquema_de_proyectos, _pool, _sql, asincrono, requiere_db_de_prueba,
)

pytestmark = requiere_db_de_prueba




@asincrono
async def test_lista_solo_proyectos_donde_es_miembro_activo_y_por_vista():
    t = await _crear_tenant("q1")
    u = await _crear_usuario(t)
    otro = await _crear_usuario(t)
    a = await _crear_proyecto_activo(t, name="A"); await _crear_scope(a, t)
    b = await _crear_proyecto_activo(t, name="B"); await _crear_scope(b, t, status="ARCHIVED")
    c = await _crear_proyecto_activo(t, name="C"); await _crear_scope(c, t)
    await _crear_membresia(a, t, u, role="VIEWER")
    await _crear_membresia(b, t, u, role="OWNER")
    await _crear_membresia(c, t, otro, role="OWNER")
    pool = await _pool()
    try:
        act = await list_projects_for_user(pool, tenant_id=t, user_id=u, view=ProjectView.ACTIVOS, before_id=None, limit=50)
        arc = await list_projects_for_user(pool, tenant_id=t, user_id=u, view=ProjectView.ARCHIVADOS, before_id=None, limit=50)
    finally:
        pool.close(); await pool.wait_closed()
    assert [p["project_id"] for p in act] == [a]
    assert act[0]["role"] == "VIEWER" and act[0]["status"] == "ACTIVE"
    assert [p["project_id"] for p in arc] == [b]


@asincrono
async def test_paginacion_por_cursor_descendente():
    t = await _crear_tenant("q2")
    u = await _crear_usuario(t)
    ids = []
    for i in range(5):
        p = await _crear_proyecto_activo(t, name=f"p{i}"); await _crear_scope(p, t)
        await _crear_membresia(p, t, u, role="OWNER"); ids.append(p)
    pool = await _pool()
    try:
        pag1 = await list_projects_for_user(pool, tenant_id=t, user_id=u, view=ProjectView.ACTIVOS, before_id=None, limit=2)
        pag2 = await list_projects_for_user(pool, tenant_id=t, user_id=u, view=ProjectView.ACTIVOS,
                                            before_id=pag1[-1]["project_id"], limit=2)
    finally:
        pool.close(); await pool.wait_closed()
    assert [p["project_id"] for p in pag1] == sorted(ids, reverse=True)[:2]
    assert [p["project_id"] for p in pag2] == sorted(ids, reverse=True)[2:4]


@asincrono
async def test_limit_se_acota_a_101_para_que_la_api_detecte_pagina_siguiente():
    t = await _crear_tenant("q2b")
    u = await _crear_usuario(t)
    for i in range(102):
        p = await _crear_proyecto_activo(t, name=f"p{i}"); await _crear_scope(p, t)
        await _crear_membresia(p, t, u, role="OWNER")
    pool = await _pool()
    try:
        r101 = await list_projects_for_user(pool, tenant_id=t, user_id=u, view=ProjectView.ACTIVOS, before_id=None, limit=101)
        r500 = await list_projects_for_user(pool, tenant_id=t, user_id=u, view=ProjectView.ACTIVOS, before_id=None, limit=500)
        r0 = await list_projects_for_user(pool, tenant_id=t, user_id=u, view=ProjectView.ACTIVOS, before_id=None, limit=0)
    finally:
        pool.close(); await pool.wait_closed()
    assert len(r101) == 101
    assert len(r500) == 101
    assert len(r0) == 1


@asincrono
async def test_ocultos_solo_para_admin_del_tenant():
    t = await _crear_tenant("q3")
    u = await _crear_usuario(t, role="operator")
    adm = await _crear_usuario(t, role="superadmin")
    h = await _crear_proyecto_activo(t, name="H"); await _crear_scope(h, t, status="HIDDEN")
    await _crear_membresia(h, t, u, role="OWNER"); await _crear_membresia(h, t, adm, role="OWNER")
    pool = await _pool()
    try:
        with pytest.raises(ProjectRoleInsufficient):
            await list_projects_for_user(pool, tenant_id=t, user_id=u, view=ProjectView.OCULTOS, before_id=None, limit=50)
        ocultos = await list_projects_for_user(pool, tenant_id=t, user_id=adm, view=ProjectView.OCULTOS, before_id=None, limit=50)
        with pytest.raises(ProjectNotVisible):
            await get_project_for_user(pool, tenant_id=t, user_id=u, project_id=h)
    finally:
        pool.close(); await pool.wait_closed()
    assert [p["project_id"] for p in ocultos] == [h]


@asincrono
async def test_get_y_miembros_niegan_a_no_miembro_y_a_otro_tenant():
    t1 = await _crear_tenant("q4a"); t2 = await _crear_tenant("q4b")
    u1 = await _crear_usuario(t1); u2 = await _crear_usuario(t2); fuera = await _crear_usuario(t1)
    p = await _crear_proyecto_activo(t1, name="P"); await _crear_scope(p, t1)
    await _crear_membresia(p, t1, u1, role="OWNER")
    pool = await _pool()
    try:
        for tenant, user in ((t1, fuera), (t2, u2)):
            with pytest.raises(ProjectNotVisible):
                await get_project_for_user(pool, tenant_id=tenant, user_id=user, project_id=p)
            with pytest.raises(ProjectNotVisible):
                await list_project_members(pool, tenant_id=tenant, user_id=user, project_id=p)
        miembros = await list_project_members(pool, tenant_id=t1, user_id=u1, project_id=p)
    finally:
        pool.close(); await pool.wait_closed()
    assert [m["user_id"] for m in miembros] == [u1]


@asincrono
async def test_candidatos_excluyen_miembros_admins_y_otros_tenants():
    t = await _crear_tenant("q5"); t2 = await _crear_tenant("q5b")
    owner = await _crear_usuario(t)
    libre = await _crear_usuario(t)
    miembro = await _crear_usuario(t)
    adm = await _crear_usuario(t, role="superadmin")
    ajeno = await _crear_usuario(t2)
    p = await _crear_proyecto_activo(t, name="P"); await _crear_scope(p, t)
    await _crear_membresia(p, t, owner, role="OWNER"); await _crear_membresia(p, t, miembro, role="VIEWER")
    filas = await _sql("SELECT user_id, email FROM jax_users WHERE user_id IN (%s,%s,%s,%s)",
                       (libre, miembro, adm, ajeno), fetch=True)
    emails = {f["user_id"]: f["email"] for f in filas}
    pool = await _pool()
    try:
        todos = []
        for uid in (libre, miembro, adm, ajeno):
            todos += await list_invite_candidates(pool, tenant_id=t, user_id=owner, project_id=p,
                                                  query=emails[uid][:6], limit=20)
        with pytest.raises(ProjectRoleInsufficient):
            await list_invite_candidates(pool, tenant_id=t, user_id=miembro, project_id=p, query="ab", limit=20)
        corto = await list_invite_candidates(pool, tenant_id=t, user_id=owner, project_id=p, query=emails[libre][:1], limit=20)
    finally:
        pool.close(); await pool.wait_closed()
    ids = {c["user_id"] for c in todos}
    assert libre in ids and not ids & {miembro, adm, ajeno, owner}
    # E1.1: con UN caracter ya filtra (antes []): el elegible SI aparece y los excluidos no.
    corto_ids = {c["user_id"] for c in corto}
    assert libre in corto_ids and not corto_ids & {miembro, adm, ajeno, owner}


@asincrono
async def test_candidatos_sin_query_mantienen_la_autoridad_de_owner_y_proyecto_activo():
    t = await _crear_tenant("q5g")
    owner = await _crear_usuario(t); miembro = await _crear_usuario(t); fuera = await _crear_usuario(t)
    p = await _crear_proyecto_activo(t, name="P"); await _crear_scope(p, t)
    await _crear_membresia(p, t, owner, role="OWNER"); await _crear_membresia(p, t, miembro, role="VIEWER")
    apagado = await _crear_proyecto_activo(t, name="X"); await _crear_scope(apagado, t, status="ARCHIVED")
    await _crear_membresia(apagado, t, owner, role="OWNER")
    pool = await _pool()
    try:
        for q in ("", "   ", "ab"):
            with pytest.raises(ProjectRoleInsufficient):
                await list_invite_candidates(pool, tenant_id=t, user_id=miembro, project_id=p, query=q, limit=20)
            with pytest.raises(ProjectNotVisible):
                await list_invite_candidates(pool, tenant_id=t, user_id=fuera, project_id=p, query=q, limit=20)
            with pytest.raises(ProjectRoleInsufficient):
                await list_invite_candidates(pool, tenant_id=t, user_id=owner, project_id=apagado, query=q, limit=20)
    finally:
        pool.close(); await pool.wait_closed()


@asincrono
async def test_candidatos_con_un_caracter_filtran_por_prefijo_de_email():
    t = await _crear_tenant("q5f")
    owner = await _crear_usuario(t)
    sufijo = uuid.uuid4().hex[:12]  # email es unico global y la base de pruebas persiste
    con_a = await _crear_usuario(t, email=f"ana-{sufijo}@test.invalid")
    con_b = await _crear_usuario(t, email=f"beto-{sufijo}@test.invalid")
    p = await _crear_proyecto_activo(t, name="P"); await _crear_scope(p, t)
    await _crear_membresia(p, t, owner, role="OWNER")
    pool = await _pool()
    try:
        a = await list_invite_candidates(pool, tenant_id=t, user_id=owner, project_id=p, query="a", limit=20)
        b = await list_invite_candidates(pool, tenant_id=t, user_id=owner, project_id=p, query=" B ", limit=20)
    finally:
        pool.close(); await pool.wait_closed()
    assert [c["user_id"] for c in a] == [con_a]
    assert [c["user_id"] for c in b] == [con_b]


@asincrono
async def test_candidatos_sin_query_listan_todos_los_elegibles_y_ningun_excluido():
    t = await _crear_tenant("q5c"); t2 = await _crear_tenant("q5d")
    owner = await _crear_usuario(t)
    libres = [await _crear_usuario(t) for _ in range(3)]
    miembro = await _crear_usuario(t)
    adm = await _crear_usuario(t, role="superadmin")
    inactivo = await _crear_usuario(t)
    await _sql("UPDATE jax_users SET status='inactive' WHERE user_id=%s", (inactivo,))
    ajeno = await _crear_usuario(t2)
    p = await _crear_proyecto_activo(t, name="P"); await _crear_scope(p, t)
    await _crear_membresia(p, t, owner, role="OWNER"); await _crear_membresia(p, t, miembro, role="VIEWER")
    pool = await _pool()
    try:
        for q in ("", "   "):
            lista = await list_invite_candidates(pool, tenant_id=t, user_id=owner, project_id=p, query=q, limit=100)
            ids = [c["user_id"] for c in lista]
            assert set(ids) == set(libres)
            assert not set(ids) & {owner, miembro, adm, inactivo, ajeno}
            emails = [c["email"] for c in lista]
            assert emails == sorted(emails)
    finally:
        pool.close(); await pool.wait_closed()


@asincrono
async def test_candidatos_limit_se_acota_a_100():
    t = await _crear_tenant("q5e")
    owner = await _crear_usuario(t)
    for _ in range(105):
        await _crear_usuario(t)
    p = await _crear_proyecto_activo(t, name="P"); await _crear_scope(p, t)
    await _crear_membresia(p, t, owner, role="OWNER")
    pool = await _pool()
    try:
        r100 = await list_invite_candidates(pool, tenant_id=t, user_id=owner, project_id=p, query="", limit=100)
        r500 = await list_invite_candidates(pool, tenant_id=t, user_id=owner, project_id=p, query="", limit=500)
        r30 = await list_invite_candidates(pool, tenant_id=t, user_id=owner, project_id=p, query="", limit=30)
    finally:
        pool.close(); await pool.wait_closed()
    assert len(r100) == 100 and len(r500) == 100 and len(r30) == 30


@asincrono
async def test_explain_lista_usa_indice_de_membresia_por_usuario():
    t = await _crear_tenant("q6"); u = await _crear_usuario(t)
    pool = await _pool()
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("EXPLAIN " + _SQL_LISTA_PARA_EXPLAIN, (t, u, "ACTIVE", None, None, 50))
                plan = await cur.fetchall()
    finally:
        pool.close(); await pool.wait_closed()
    tabla_m = [r for r in plan if (r["table"] if isinstance(r, dict) else r[2]) == "m"][0]
    key = tabla_m["key"] if isinstance(tabla_m, dict) else tabla_m[6]
    extra = (tabla_m["Extra"] if isinstance(tabla_m, dict) else tabla_m[9]) or ""
    assert key == "idx_jax_project_membership_user_list"
    assert "filesort" not in extra.lower()


# ---------------------------------------------------------------- M-2: la regla de lectura = resolve_project_read

@asincrono
async def test_membresia_revocada_no_ve_el_proyecto_ni_en_lista_ni_en_get():
    t = await _crear_tenant("q7")
    u = await _crear_usuario(t)
    p = await _crear_proyecto_activo(t, name="R"); await _crear_scope(p, t)
    await _crear_membresia(p, t, u, role="OWNER", status="REVOKED")
    pool = await _pool()
    try:
        lista = await list_projects_for_user(pool, tenant_id=t, user_id=u, view=ProjectView.ACTIVOS, before_id=None, limit=50)
        with pytest.raises(ProjectNotVisible):
            await get_project_for_user(pool, tenant_id=t, user_id=u, project_id=p)
    finally:
        pool.close(); await pool.wait_closed()
    assert lista == []


@asincrono
async def test_usuario_inactivo_lista_vacia_y_get_niega():
    t = await _crear_tenant("q8")
    u = await _crear_usuario(t, status="inactive")
    p = await _crear_proyecto_activo(t, name="I"); await _crear_scope(p, t)
    await _crear_membresia(p, t, u, role="OWNER")
    pool = await _pool()
    try:
        lista = await list_projects_for_user(pool, tenant_id=t, user_id=u, view=ProjectView.ACTIVOS, before_id=None, limit=50)
        with pytest.raises(ProjectNotVisible):
            await get_project_for_user(pool, tenant_id=t, user_id=u, project_id=p)
    finally:
        pool.close(); await pool.wait_closed()
    assert lista == []


@asincrono
async def test_admin_ve_hidden_con_get_pero_no_disabled_en_get_miembros_ni_candidatos():
    """Espejo de `resolve_project_read` (scope_authority.py): HIDDEN solo para el
    admin del tenant; DISABLED NUNCA, tampoco para el admin (D3)."""
    t = await _crear_tenant("q9")
    adm = await _crear_usuario(t, role="superadmin")
    oculto = await _crear_proyecto_activo(t, name="H"); await _crear_scope(oculto, t, status="HIDDEN")
    apagado = await _crear_proyecto_activo(t, name="D"); await _crear_scope(apagado, t, status="DISABLED")
    await _crear_membresia(oculto, t, adm, role="OWNER")
    await _crear_membresia(apagado, t, adm, role="OWNER")
    pool = await _pool()
    try:
        fila = await get_project_for_user(pool, tenant_id=t, user_id=adm, project_id=oculto)
        with pytest.raises(ProjectNotVisible):
            await get_project_for_user(pool, tenant_id=t, user_id=adm, project_id=apagado)
        with pytest.raises(ProjectNotVisible):
            await list_project_members(pool, tenant_id=t, user_id=adm, project_id=apagado)
        with pytest.raises(ProjectNotVisible):
            await list_invite_candidates(pool, tenant_id=t, user_id=adm, project_id=apagado, query="ab", limit=5)
    finally:
        pool.close(); await pool.wait_closed()
    assert fila["status"] == "HIDDEN"


@asincrono
async def test_candidatos_sobre_proyecto_archivado_exigen_proyecto_activo():
    t = await _crear_tenant("q10")
    owner = await _crear_usuario(t)
    p = await _crear_proyecto_activo(t, name="A"); await _crear_scope(p, t, status="ARCHIVED")
    await _crear_membresia(p, t, owner, role="OWNER")
    pool = await _pool()
    try:
        with pytest.raises(ProjectRoleInsufficient):
            await list_invite_candidates(pool, tenant_id=t, user_id=owner, project_id=p, query="ab", limit=5)
    finally:
        pool.close(); await pool.wait_closed()
