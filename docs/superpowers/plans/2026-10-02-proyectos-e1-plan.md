# Proyectos E1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que cualquier usuario cree proyectos, administre sus miembros y elija un proyecto en el chat de jax-platform, sobre la autoridad de B9 ya desplegada; con HAMURABI y los 241 huérfanos migrados.

**Architecture:** jax aporta lo que le falta a la autoridad de B9 (renombrar, consultas de lectura paginadas) y un guion de migración de datos. jax-platform expone `/api/proyectos` llamando a `ProjectAuthorityAdmin` y a las consultas de jax **en proceso** (mismo patrón que `backend/api/chat.py` y `backend/api/admin/users.py`), y agrega la pantalla `/proyectos` y el selector de proyecto en el chat. La plataforma no decide papeles: traduce errores de B9 a códigos HTTP estables.

**Tech Stack:** Python 3.12, aiomysql, MariaDB 12.3 (127.0.0.1:3308), FastAPI, pytest (base `jax_test`); React 19, Zustand, Vite, vitest, Tailwind con tokens.

**Spec:** `docs/superpowers/specs/2026-10-02-proyectos-e1-adenda-design.md` (Jax#314), que continúa `docs/superpowers/specs/2026-09-22-proyectos-y-selector-design.md`.

## Global Constraints

- Papeles ofrecidos: lector = `VIEWER`, editor = `CONTRIBUTOR`, dueño = `OWNER`. `REVIEWER` nunca se ofrece.
- Estados expuestos: `ACTIVE`, `ARCHIVED`, `HIDDEN`. `DISABLED` no se expone. Destruir está FUERA.
- 404 si el proyecto no existe, está oculto o no eres miembro (sin revelar cuál); 403 si eres miembro y te falta el papel.
- Toda mutación pasa por `ProjectAuthorityAdmin`; la plataforma no escribe en `jax_project_*` ni en `projects` directamente.
- Nombre de proyecto 1–255 caracteres (NFC, sin espacios a los lados); descripción ≤ 2000.
- i18n en `frontend/src/i18n/es.js` y `en.js`; colores solo con tokens; claro y oscuro; ningún `confirm(`, `alert(` ni `prompt(` (tampoco desnudos): toda confirmación en `components/Dialogo.jsx` o `ConfirmacionSuma`.
- MariaDB en el puerto **3308**. Las pruebas usan `jax_test` (credencial `~/.config/jax/test-db.env`); **nunca** `/etc/jax/.env` ni la base `jax_memory`.
- Subagentes: sin `sudo`, sin `git push`, sin `gh`; mensajes de commit con `git commit -F <archivo>`. Los controles negativos se corren en directorios de prueba, nunca contra rutas del sistema.
- jax usa la rama `master` (no `main`); jax-platform también.
- Las escrituras en producción (Tarea 11) solo con el GO de Fernando o su ventana.

## Review Focus

1. **Proyecto de otro tenant o inexistente con un id válido** → la API responde 404 con el mismo cuerpo que «no eres miembro»; nunca 403 ni 500. Prueba en la Tarea 5 (`test_otro_tenant_da_404_igual_que_inexistente`).
2. **Doble clic en «Crear proyecto»** → un solo proyecto (la `Idempotency-Key` la genera el cliente una vez por diálogo abierto). Pruebas en las Tareas 5 y 7.
3. **El último dueño intenta rebajarse o salir** → 409 `ultimo_dueno` con texto claro, nada cambia. Prueba en la Tarea 5.
4. **Proyecto archivado elegido en el selector del chat** (o archivado mientras estaba elegido) → el selector no lo ofrece y, si estaba elegido, vuelve a «Personal» con aviso; nunca manda un `project_id` archivado. Prueba en la Tarea 9.
5. **Lista con más de una página / cursor manipulado** → el cursor es un id entero; uno no numérico da 422, uno que apunta a un proyecto ajeno solo acota la página (no filtra datos). Prueba en las Tareas 2 y 5.

---

## Parte A — jax (rama `feat/proyectos-e1-jax`, worktree `~/worktrees/jax-proyectos-e1-impl`)

### Task 1: `rename_project` en la autoridad de B9

**Files:**
- Modify: `jax/memory/project_authority.py` (nuevo método después de `set_project_lifecycle`, ~línea 851)
- Test: `tests/test_project_authority_mariadb.py` (agregar al final)

**Interfaces:**
- Produces: `ProjectAuthorityAdmin.rename_project(request: MutationAuthorizationRequest, project_id: int, *, name: str, description: str | None) -> bool` — `True` si cambió, `False` si ya tenía ese nombre y descripción (no-op idempotente, sin evento). Operación `"RENAME_PROJECT"`, mínimo OWNER, solo proyecto ACTIVE. Evento `RENAME_PROJECT` en `jax_project_membership_event` con `target_user_id=None`, `old_status=new_status='ACTIVE'`.

- [ ] **Step 1: Escribir las pruebas que fallan**

Usar los ayudantes que ya tiene el archivo (`_crear_tenant`, `_crear_usuario`, `_crear_proyecto_activo`, `_crear_scope`, `_crear_membresia`, `_pool`, `_scope`, `_request`, `_admin`, `_sql`, `asincrono`).

```python
@asincrono
async def test_rename_project_cambia_nombre_y_deja_evento():
    t = await _crear_tenant("ren")
    u = await _crear_usuario(t)
    p = await _crear_proyecto_activo(t, name="viejo")
    await _crear_scope(p, t)
    await _crear_membresia(p, t, u, role="OWNER")
    async with _pool() as pool:
        cambio = await _admin(pool).rename_project(
            _request(_scope(u, t, p), "RENAME_PROJECT"), p, name="  Nuevo  ", description="d")
    assert cambio is True
    fila = await _sql("SELECT name, description FROM projects WHERE id=%s", (p,), fetch=True)
    assert fila[0] == ("Nuevo", "d")
    ev = await _sql("SELECT COUNT(*) FROM jax_project_membership_event WHERE project_id=%s AND operation='RENAME_PROJECT'",
                    (p,), fetch=True)
    assert ev[0][0] == 1


@asincrono
async def test_rename_project_igual_es_noop_sin_evento():
    t = await _crear_tenant("ren2")
    u = await _crear_usuario(t)
    p = await _crear_proyecto_activo(t, name="mismo")
    await _crear_scope(p, t)
    await _crear_membresia(p, t, u, role="OWNER")
    await _sql("UPDATE projects SET description=NULL WHERE id=%s", (p,))
    async with _pool() as pool:
        cambio = await _admin(pool).rename_project(
            _request(_scope(u, t, p), "RENAME_PROJECT"), p, name="mismo", description=None)
    assert cambio is False
    ev = await _sql("SELECT COUNT(*) FROM jax_project_membership_event WHERE project_id=%s AND operation='RENAME_PROJECT'",
                    (p,), fetch=True)
    assert ev[0][0] == 0


@asincrono
async def test_rename_project_exige_owner_y_activo():
    t = await _crear_tenant("ren3")
    owner = await _crear_usuario(t)
    editor = await _crear_usuario(t)
    p = await _crear_proyecto_activo(t, name="x")
    await _crear_scope(p, t)
    await _crear_membresia(p, t, owner, role="OWNER")
    await _crear_membresia(p, t, editor, role="CONTRIBUTOR")
    async with _pool() as pool:
        with pytest.raises(ProjectRoleInsufficient):
            await _admin(pool).rename_project(_request(_scope(editor, t, p), "RENAME_PROJECT"), p,
                                              name="y", description=None)
        await _sql("UPDATE jax_project_scope SET status='ARCHIVED' WHERE project_id=%s", (p,))
        with pytest.raises(ProjectStateConflict):
            await _admin(pool).rename_project(_request(_scope(owner, t, p), "RENAME_PROJECT"), p,
                                              name="y", description=None)


@asincrono
async def test_rename_project_valida_nombre():
    t = await _crear_tenant("ren4")
    u = await _crear_usuario(t)
    p = await _crear_proyecto_activo(t, name="x")
    await _crear_scope(p, t)
    await _crear_membresia(p, t, u, role="OWNER")
    async with _pool() as pool:
        for malo in ("", "   ", "x" * 256):
            with pytest.raises(AuthorizationDenied):
                await _admin(pool).rename_project(_request(_scope(u, t, p), "RENAME_PROJECT"), p,
                                                  name=malo, description=None)
        with pytest.raises(AuthorizationDenied):
            await _admin(pool).rename_project(_request(_scope(u, t, p), "RENAME_PROJECT"), p,
                                              name="ok", description="d" * 2001)
```

Si `ProjectRoleInsufficient`, `ProjectStateConflict` o `AuthorizationDenied` no están importados en el archivo de pruebas, agregarlos al `from jax.memory.project_authority import ...` / `from jax.memory.b9 import ...` existentes.

- [ ] **Step 2: Correr y ver que fallan**

Run: `cd ~/worktrees/jax-proyectos-e1-impl && set -a && . ~/.config/jax/test-db.env && set +a && .venv/bin/pytest tests/test_project_authority_mariadb.py -k rename_project -v`
Expected: FAIL con `AttributeError: 'ProjectAuthorityAdmin' object has no attribute 'rename_project'`.
(Si el worktree no tiene `.venv`, usar el de `~/jax/.venv/bin/pytest`.)

- [ ] **Step 3: Implementar**

El candado de `projects` va por `lock_target` para respetar el orden del módulo (`tenant -> actor -> destino/projects -> scope -> membership -> events`).

```python
    async def rename_project(self, request: MutationAuthorizationRequest, project_id: int, *,
                             name: str, description: str | None) -> bool:
        """Renombra un proyecto ACTIVE. Mínimo OWNER. Idempotente: si nombre y
        descripción ya son esos, no escribe ni deja evento (un reintento tras un
        resultado DESCONOCIDO no duplica). El candado de `projects` se toma vía
        `lock_target`, ANTES del alcance, como exige el orden del módulo."""
        normalized_name = unicodedata.normalize("NFC", name or "").strip()
        if not (1 <= len(normalized_name) <= 255):
            raise AuthorizationDenied("project name must be 1-255 characters")
        if description is not None and len(description) > 2000:
            raise AuthorizationDenied("project description must be <= 2000 characters")

        async def op(cur: Any) -> bool:
            try:
                async def _lock_project(cur: Any) -> Any:
                    await cur.execute("SELECT name,description FROM projects WHERE id=%s FOR UPDATE", (project_id,))
                    return await cur.fetchone()

                actor, current = await self._resolve_project_actor_cur(
                    cur, request, project_id, expected_operation="RENAME_PROJECT",
                    min_role=ProjectRole.OWNER, allowed_states=frozenset({ProjectLifecycle.ACTIVE}),
                    lock_target=_lock_project)
                if not current:
                    raise ProjectNotVisible("project does not exist")
                if (self._value(current, "name", 0) == normalized_name
                        and self._value(current, "description", 1) == description):
                    return False
                await cur.execute("UPDATE projects SET name=%s,description=%s WHERE id=%s",
                                  (normalized_name, description, project_id))
                await self._event(cur, request.scope, "RENAME_PROJECT", project_id, None, actor.tenant_id,
                                  None, None, "ACTIVE", "ACTIVE")
                return True
            except ProjectAuthorityError:
                raise
            except Exception as exc:
                raise _wrap_unexpected_db_error(exc) from exc

        return await self._store.mutation(op)
```

- [ ] **Step 4: Correr y ver que pasan**

Run: el mismo comando del Step 2. Expected: 4 passed. Luego toda la suite de proyectos: `.venv/bin/pytest tests/test_project_authority_mariadb.py tests/test_project_authority_input_validation.py tests/test_project_scope_authority.py -q` → sin fallos.

- [ ] **Step 5: Commit**

```bash
git add jax/memory/project_authority.py tests/test_project_authority_mariadb.py
git commit -F /ruta/al/mensaje.txt   # "feat(proyectos): rename_project en la autoridad de B9 (E1, T1)"
```

---

### Task 2: Consultas de lectura de proyectos (`project_queries.py`) + índice

**Files:**
- Create: `jax/memory/project_queries.py`
- Modify: `jax/memory/project_authority_migrations.py` (un índice nuevo, idempotente)
- Test: `tests/test_project_queries_mariadb.py` (nuevo; copiar del encabezado de `tests/test_project_authority_mariadb.py` los ayudantes `asincrono`, `_conn_params`, `_ensure_schema`, `_sql`, `_crear_tenant`, `_crear_usuario`, `_crear_proyecto_activo`, `_crear_scope`, `_crear_membresia`, `_pool` y el fixture de esquema — o importarlos de ese módulo si ya se importan entre tests en el repo; elegir lo que haga el repo hoy)

**Interfaces:**
- Produces (todas de solo lectura, reciben un pool de aiomysql que entrega cursores de mapeo — el `_B9MappingPool` de la plataforma o el de los tests):
  - `class ProjectView(str, Enum): ACTIVOS="activos"; ARCHIVADOS="archivados"; OCULTOS="ocultos"`
  - `async def list_projects_for_user(pool, *, tenant_id: int, user_id: int, view: ProjectView, before_id: int | None, limit: int) -> list[dict]` — cada dict: `{"project_id": int, "project_uuid": str, "name": str, "description": str | None, "status": "ACTIVE"|"ARCHIVED"|"HIDDEN", "role": "VIEWER"|"CONTRIBUTOR"|"OWNER"|"REVIEWER"}`, orden `project_id DESC`, `limit` en 1..100. `OCULTOS` para un no-admin del tenant → `ProjectRoleInsufficient`. Usuario inexistente/inactivo o de otro tenant → lista vacía.
  - `async def get_project_for_user(pool, *, tenant_id: int, user_id: int, project_id: int) -> dict` — mismo dict; `ProjectNotVisible` si no es miembro activo, el alcance no es de ese tenant, o está HIDDEN/DISABLED y no es admin.
  - `async def list_project_members(pool, *, tenant_id: int, user_id: int, project_id: int) -> list[dict]` — exige lo mismo que `get_project_for_user`; cada dict: `{"user_id": int, "email": str, "role": str, "grant_origin": str}`, solo membresías ACTIVE, orden por email.
  - `async def list_invite_candidates(pool, *, tenant_id: int, user_id: int, project_id: int, query: str, limit: int) -> list[dict]` — exige OWNER del proyecto ACTIVE (`ProjectRoleInsufficient` si no); usuarios `active` del mismo tenant cuyo email empieza por `query` (≥ 2 caracteres, si no lista vacía), que no son miembros ACTIVE ni admins del tenant; `{"user_id", "email"}`, máximo `limit` (1..20).
  - > **E1.1 (2026-10-02, Fernando).** `query` es opcional (vacío = lista sin filtrar; 1 o más caracteres = prefijo de email) y el tope de `limit` es 1..100; la UI de Miembros pasa a lista con casillas más filtro por email. Lo de arriba (≥ 2 caracteres, máximo 20) es la historia de E1: no se reescribe. Índice `idx_jax_users_tenant_email` en la migración 005i.

- [ ] **Step 1: Escribir las pruebas que fallan**

```python
from jax.memory.project_queries import (ProjectView, list_projects_for_user, get_project_for_user,
                                        list_project_members, list_invite_candidates)
from jax.memory.project_authority import ProjectNotVisible, ProjectRoleInsufficient


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
    async with _pool() as pool:
        act = await list_projects_for_user(pool, tenant_id=t, user_id=u, view=ProjectView.ACTIVOS, before_id=None, limit=50)
        arc = await list_projects_for_user(pool, tenant_id=t, user_id=u, view=ProjectView.ARCHIVADOS, before_id=None, limit=50)
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
    async with _pool() as pool:
        pag1 = await list_projects_for_user(pool, tenant_id=t, user_id=u, view=ProjectView.ACTIVOS, before_id=None, limit=2)
        pag2 = await list_projects_for_user(pool, tenant_id=t, user_id=u, view=ProjectView.ACTIVOS,
                                            before_id=pag1[-1]["project_id"], limit=2)
    assert [p["project_id"] for p in pag1] == sorted(ids, reverse=True)[:2]
    assert [p["project_id"] for p in pag2] == sorted(ids, reverse=True)[2:4]


@asincrono
async def test_ocultos_solo_para_admin_del_tenant():
    t = await _crear_tenant("q3")
    u = await _crear_usuario(t, role="operator")
    adm = await _crear_usuario(t, role="superadmin")
    h = await _crear_proyecto_activo(t, name="H"); await _crear_scope(h, t, status="HIDDEN")
    await _crear_membresia(h, t, u, role="OWNER"); await _crear_membresia(h, t, adm, role="OWNER")
    async with _pool() as pool:
        with pytest.raises(ProjectRoleInsufficient):
            await list_projects_for_user(pool, tenant_id=t, user_id=u, view=ProjectView.OCULTOS, before_id=None, limit=50)
        ocultos = await list_projects_for_user(pool, tenant_id=t, user_id=adm, view=ProjectView.OCULTOS, before_id=None, limit=50)
        with pytest.raises(ProjectNotVisible):
            await get_project_for_user(pool, tenant_id=t, user_id=u, project_id=h)
    assert [p["project_id"] for p in ocultos] == [h]


@asincrono
async def test_get_y_miembros_niegan_a_no_miembro_y_a_otro_tenant():
    t1 = await _crear_tenant("q4a"); t2 = await _crear_tenant("q4b")
    u1 = await _crear_usuario(t1); u2 = await _crear_usuario(t2); fuera = await _crear_usuario(t1)
    p = await _crear_proyecto_activo(t1, name="P"); await _crear_scope(p, t1)
    await _crear_membresia(p, t1, u1, role="OWNER")
    async with _pool() as pool:
        for tenant, user in ((t1, fuera), (t2, u2)):
            with pytest.raises(ProjectNotVisible):
                await get_project_for_user(pool, tenant_id=tenant, user_id=user, project_id=p)
            with pytest.raises(ProjectNotVisible):
                await list_project_members(pool, tenant_id=tenant, user_id=user, project_id=p)
        miembros = await list_project_members(pool, tenant_id=t1, user_id=u1, project_id=p)
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
    emails = dict(await _sql("SELECT user_id, email FROM jax_users WHERE user_id IN (%s,%s,%s,%s)",
                             (libre, miembro, adm, ajeno), fetch=True))
    async with _pool() as pool:
        todos = []
        for uid in (libre, miembro, adm, ajeno):
            todos += await list_invite_candidates(pool, tenant_id=t, user_id=owner, project_id=p,
                                                  query=emails[uid][:6], limit=20)
        with pytest.raises(ProjectRoleInsufficient):
            await list_invite_candidates(pool, tenant_id=t, user_id=miembro, project_id=p, query="ab", limit=20)
        corto = await list_invite_candidates(pool, tenant_id=t, user_id=owner, project_id=p, query="a", limit=20)
    ids = {c["user_id"] for c in todos}
    assert libre in ids and not ids & {miembro, adm, ajeno, owner}
    assert corto == []
```

Nota: si `_crear_usuario` genera emails con un prefijo común (p. ej. `u-<uuid>@test`), los primeros 6 caracteres pueden coincidir entre usuarios; la prueba solo afirma pertenencia, así que sigue siendo válida.

Prueba de plan (Las Cuatro, índice):

```python
@asincrono
async def test_explain_lista_usa_indice_de_membresia_por_usuario():
    t = await _crear_tenant("q6"); u = await _crear_usuario(t)
    async with _pool() as pool:
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("EXPLAIN " + _SQL_LISTA_PARA_EXPLAIN, (t, u, "ACTIVE", None, None, 50))
                plan = await cur.fetchall()
    tabla_m = [r for r in plan if (r["table"] if isinstance(r, dict) else r[2]) == "m"][0]
    key = tabla_m["key"] if isinstance(tabla_m, dict) else tabla_m[6]
    extra = (tabla_m["Extra"] if isinstance(tabla_m, dict) else tabla_m[9]) or ""
    assert key == "idx_jax_project_membership_user_status"
    assert "filesort" not in extra.lower()
```

con `from jax.memory.project_queries import _SQL_LISTA as _SQL_LISTA_PARA_EXPLAIN`.

- [ ] **Step 2: Correr y ver que fallan**

Run: `.venv/bin/pytest tests/test_project_queries_mariadb.py -v` (con `test-db.env` cargado como en la Tarea 1)
Expected: FAIL `ModuleNotFoundError: No module named 'jax.memory.project_queries'`.

- [ ] **Step 3: Índice nuevo**

En `jax/memory/project_authority_migrations.py`, agregar a la tupla de sentencias de migración (al final, antes del cierre) — idempotente:

```python
    """CREATE INDEX IF NOT EXISTS idx_jax_project_membership_user_status
       ON jax_project_membership (tenant_id, user_id, status, project_id)""",
```

Confirmar que el aplicador de esa tupla tolera sentencias `CREATE INDEX IF NOT EXISTS` (MariaDB las soporta). Si el archivo tiene una función de reversión que enumera objetos, agregar `DROP INDEX IF EXISTS idx_jax_project_membership_user_status ON jax_project_membership` en el lugar simétrico, y correr la prueba de migración existente (`test_migracion_apply_apply_revert_apply_y_falla_cerrado_con_fila_archivada`).

- [ ] **Step 4: Implementar `jax/memory/project_queries.py`**

```python
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


_VIEW_STATUS = {ProjectView.ACTIVOS: "ACTIVE", ProjectView.ARCHIVADOS: "ARCHIVED", ProjectView.OCULTOS: "HIDDEN"}
_ADMIN_ROLES = tuple(sorted(TENANT_ADMIN_ROLES))
_ADMIN_PH = ",".join(["%s"] * len(_ADMIN_ROLES))

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
    limit = max(1, min(int(limit), 100))
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
    if fila["status"] in ("HIDDEN", "DISABLED") and not es_admin:
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
    query = (query or "").strip()
    limit = max(1, min(int(limit), 20))
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            fila = await _proyecto_visible(cur, tenant_id, user_id, project_id)
            if fila["role"] != "OWNER" or fila["status"] != "ACTIVE":
                raise ProjectRoleInsufficient("inviting requires OWNER on an ACTIVE project")
            if len(query) < 2:
                return []
            patron = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            await cur.execute(
                "SELECT u.user_id, u.email FROM jax_users u "
                "LEFT JOIN jax_project_membership m ON m.user_id=u.user_id AND m.project_id=%s AND m.status='ACTIVE' "
                f"WHERE u.tenant_id=%s AND u.status='active' AND LOWER(u.role) NOT IN ({_ADMIN_PH}) "
                "AND m.user_id IS NULL AND u.email LIKE %s ORDER BY u.email LIMIT %s",
                (project_id, tenant_id, *_ADMIN_ROLES, patron, limit))
            return [{"user_id": int(_v(r, "user_id", 0)), "email": _v(r, "email", 1)} for r in await cur.fetchall()]
```

- [ ] **Step 5: Correr y ver que pasan**

Run: `.venv/bin/pytest tests/test_project_queries_mariadb.py -v` → 6 passed. Si la prueba de `EXPLAIN` elige otro índice, NO forzar con `FORCE INDEX` a ciegas: mirar el plan, ajustar el orden de columnas del índice y volver a medir; anotar el plan final en el mensaje de commit.

- [ ] **Step 6: Commit**

```bash
git add jax/memory/project_queries.py jax/memory/project_authority_migrations.py tests/test_project_queries_mariadb.py
git commit -F /ruta/al/mensaje.txt   # "feat(proyectos): consultas de lectura e índice por usuario (E1, T2)"
```

---

### Task 3: Guion de migración de datos E1 (`scripts/proyectos_e1_migrar.py`)

**Files:**
- Create: `scripts/proyectos_e1_migrar.py`
- Test: `tests/test_proyectos_e1_migrar_mariadb.py`

**Interfaces:**
- Consumes: `ProjectAuthorityAdmin.bootstrap_existing_project`, `.create_project`, `.set_project_lifecycle`; `MariaDBB9Store`.
- Produces: CLI `python scripts/proyectos_e1_migrar.py {--verificar|--aplicar} --actor-user-id N [--database NOMBRE]`. Funciones importables:
  - `async def medir(pool) -> dict` → `{"huerfanos_ids": [int], "filas_por_tabla": {"conversations": n, "messages": n, "facts": n, "decisions": n, "action_items": n}, "hamurabi_con_alcance": bool, "evaluacion_project_id": int | None}`
  - `async def aplicar(pool, *, actor_user_id: int) -> dict` → `{"antes": medir, "despues": medir, "evaluacion_project_id": int}`; idempotente (segunda corrida no cambia nada y devuelve el mismo id).

Reglas que implementa (spec §5.4):
1. HAMURABI = `projects.id = 1`. `bootstrap_existing_project(request(BOOTSTRAP_PROJECT, scope project_id=1), 1, owner_user_id=actor_user_id)`; si ya tiene alcance, `bootstrap` devuelve `False` (no-op) — aceptable.
2. Proyecto «Evaluación grounding SP3 · 2026-09-03» con `create_project(..., idempotency_key="e1-evaluacion-grounding-sp3-20260903")` y después `set_project_lifecycle(..., ARCHIVED)`. La llave fija hace la creación idempotente.
3. Huérfanos = `DISTINCT project_id` de las cinco tablas con `project_id IS NOT NULL AND project_id NOT IN (SELECT id FROM projects)`. Se reescriben con `UPDATE <tabla> SET project_id=%s WHERE project_id IN (...)` en **una** transacción para las cinco tablas. Antes y después: `filas_por_tabla` (filas con `project_id` en el conjunto huérfano, antes; con `project_id = evaluacion`, después) tienen que coincidir tabla por tabla; si no, `ROLLBACK` y error.
4. Nada de FKs aquí (Tarea 4). Nunca `DELETE`.
5. El scope de actor se arma como en la plataforma: `ScopeContext(actor_principal=f"user:{id}", actor_type="USER", subject_user_id=str(id), tenant_id="1", project_id=..., calling_component="proyectos-e1-migrar")`.
6. Credencial: lee la conexión de variables de entorno (`JAX_DB_HOST`, `JAX_DB_PORT`, `JAX_DB_USER`, `JAX_DB_PASSWORD`, `JAX_DB_NAME`: los nombres de `~/.config/jax/test-db.env`, verificados 2026-10-02), nunca abre `/etc/jax/.env` por su cuenta. En producción el runbook carga el entorno.

- [ ] **Step 1: Escribir las pruebas que fallan** (en `jax_test`, con datos sembrados por la prueba)

```python
@asincrono
async def test_aplicar_mueve_huerfanos_con_conteos_iguales_y_es_idempotente():
    t = 1  # LEGACY_PROJECT_TENANT_ID: el guion asume el tenant legado
    await _asegurar_tenant(t)
    actor = await _crear_usuario(t, role="superadmin")
    await _asegurar_proyecto_1()                       # projects.id=1 sin alcance (HAMURABI de prueba)
    await _sembrar_huerfanos([900001, 900002, 1400055])  # 1 conversación + 2 mensajes por id
    async with _pool() as pool:
        r1 = await aplicar(pool, actor_user_id=actor)
        r2 = await aplicar(pool, actor_user_id=actor)
    ev = r1["evaluacion_project_id"]
    assert r2["evaluacion_project_id"] == ev
    assert r1["despues"]["huerfanos_ids"] == []
    assert r1["antes"]["filas_por_tabla"]["conversations"] == 3
    assert r1["antes"]["filas_por_tabla"]["messages"] == 6
    assert await _contar("conversations", ev) == 3 and await _contar("messages", ev) == 6
    assert (await _status_alcance(ev)) == "ARCHIVED"
    assert r1["despues"]["hamurabi_con_alcance"] is True
    assert await _rol(1, actor) == "OWNER"


@asincrono
async def test_verificar_no_escribe():
    ...  # sembrar 1 huérfano; correr medir(); afirmar que el conteo de filas y de projects no cambió
```

Los ayudantes `_asegurar_tenant`, `_asegurar_proyecto_1`, `_sembrar_huerfanos`, `_contar`, `_status_alcance`, `_rol` se escriben en el archivo de prueba con `_sql`. **Ojo:** `jax_test` es compartida por la suite; `_asegurar_proyecto_1` hace `INSERT IGNORE INTO projects (id, project_uuid, name, status) VALUES (1, UUID(), 'HAMURABI', 'active')` y la prueba limpia al final solo lo que creó (filas por id), nunca `TRUNCATE`.

- [ ] **Step 2: Correr y ver que fallan** — `ModuleNotFoundError` / `ImportError`.

- [ ] **Step 3: Implementar** el guion según las reglas 1–6 (funciones `medir`, `aplicar`, `main` con `argparse`; `--verificar` imprime `medir()` como JSON; `--aplicar` imprime `aplicar()` como JSON). Importar el script desde la prueba con `importlib` por ruta (`scripts/` no es paquete), como ya hacen otras pruebas del repo con `scripts/` (buscar `spec_from_file_location` en `tests/`).

- [ ] **Step 4: Correr y ver que pasan.**

- [ ] **Step 5: Commit** — `"feat(proyectos): guion de migración E1 (HAMURABI y huérfanos) (E1, T3)"`.

---

### Task 4: Ensayo de las FKs de `project_id` y runbook de producción

**Files:**
- Create: `docs/runbooks/proyectos-e1-produccion.md`
- Create: `scripts/proyectos_e1_fks.py` (`--ensayar` sobre una copia / `--aplicar`)
- Modify: `DEUDA.md` (solo si alguna FK se separa)

**Interfaces:**
- Produces: `scripts/proyectos_e1_fks.py --ensayar --database <copia>` imprime, por tabla, `{"tabla", "filas", "algoritmo": "INPLACE"|"COPY", "segundos", "hnsw_intacto": bool}`; `--aplicar --database jax_memory` aplica solo las tablas cuyo ensayo dio `INPLACE` y `hnsw_intacto`.

- [ ] **Step 1: Ensayo en una copia** (lo hace la sesión principal, no un subagente: toca datos reales)
  - Volcado de `jax_memory` con el procedimiento de respaldo del runbook de B9 (`docs/runbooks/`), restaurado a `jax_memory_e1_ensayo` en el mismo MariaDB 3308.
  - En la copia, correr `scripts/proyectos_e1_migrar.py --aplicar` (los huérfanos tienen que desaparecer antes de las FKs).
  - Por cada tabla (`conversations`, `messages`, `facts`, `decisions`, `action_items`): `ALTER TABLE <t> ADD CONSTRAINT fk_<t>_project FOREIGN KEY (project_id) REFERENCES projects(id), ALGORITHM=INPLACE, LOCK=NONE`. Si MariaDB rechaza `INPLACE`, registrar `COPY` y **no** aplicarla en esa tabla.
  - Para `messages` (lleva el índice vectorial): después del `ALTER`, correr el detector de envenenamiento HNSW del runbook de jax#239 (MDEV-41227) sobre la copia y una búsqueda semántica de control; `hnsw_intacto=False` si cualquiera falla.
- [ ] **Step 2: Guion `scripts/proyectos_e1_fks.py`** que automatiza el Step 1 (sin crear la copia: recibe `--database`). Prueba en `jax_test` con tablas mínimas sembradas: `--aplicar` crea la FK y es idempotente (si ya existe, no falla).
- [ ] **Step 3: Runbook** `docs/runbooks/proyectos-e1-produccion.md`, en este orden:
  1. Ventana o GO de Fernando; `bin/ventana estado` si aplica.
  2. Respaldo verificado de `jax_memory` con restauración probada (mismo procedimiento de `~/respaldos-despliegue/2026-10-02-suscripcion-fase1`).
  3. Desplegar jax con T1–T3 (migración del índice de T2 incluida en el arranque de B9 o aplicada por su aplicador; documentar cuál).
  4. `proyectos_e1_migrar.py --verificar` → anotar números; `--aplicar --actor-user-id 1` → comparar.
  5. `proyectos_e1_fks.py --aplicar` solo con las tablas que pasaron el ensayo; las otras van a `DEUDA.md` con el motivo y la validación en la aplicación.
  6. Verificación por efecto: `/chat` con `project_id=1` como Fernando → 200; como otro usuario → 403.
  7. Cómo revertir cada paso (las FKs con `DROP FOREIGN KEY`; los huérfanos con el `UPDATE` inverso usando la lista guardada por `--verificar`; el bootstrap no se revierte: el alcance de HAMURABI queda).
- [ ] **Step 4: Commit** — `"docs(runbook): producción de Proyectos E1 y ensayo de FKs (E1, T4)"`.

**Cierre de la Parte A:** auditoría de escalón 3 (`arquitecto-adversarial`) sobre la rama completa; PR a `master` de jax; lo integra la sesión principal con CI verde y sin BLOCK/MAJOR abiertos (o Fernando). La Parte B consume `master` de jax en su CI, así que **A se integra antes de abrir el PR de B**.

---

## Parte B — jax-platform (rama `feat/proyectos-e1`, worktree `~/worktrees/jax-platform-proyectos-e1`)

### Task 5: API `/api/proyectos`

**Files:**
- Create: `backend/api/proyectos.py`
- Modify: `backend/main.py` (importar y registrar el router junto a los demás, ~línea 86–100 y el bucle de `include_router` ~línea 271)
- Test: `backend/tests/test_proyectos_api.py`

**Interfaces:**
- Consumes: `jax.memory.project_queries.*` (T2), `ProjectAuthorityAdmin.{create_project, rename_project, set_project_lifecycle, grant_member, change_project_role, revoke_member}`, `MariaDBB9Store`, `_B9MappingPool` (`backend/api/chat.py:280`; si sigue privado ahí, moverlo a un módulo compartido `backend/b9_pool.py` en este mismo task y que `chat.py` lo importe de ahí — sin cambiar su comportamiento), `get_current_user`/`AuthUser`, `get_pool`.
- Produces (JSON; prefijo `/api` como el resto):
  - `GET /proyectos?vista=activos|archivados|ocultos&antes_de=<int>&limite=<1..100>` → `{"proyectos": [ProyectoOut], "siguiente": int | null}`
  - `POST /proyectos` cabecera `Idempotency-Key` (requerida) cuerpo `{"nombre": str, "descripcion": str | null}` → 201 `ProyectoOut` (200 si fue repetición idempotente)
  - `GET /proyectos/{id}` → `ProyectoOut`
  - `PATCH /proyectos/{id}` `{"nombre", "descripcion"}` → `ProyectoOut`
  - `POST /proyectos/{id}/estado` `{"estado": "ACTIVE"|"ARCHIVED"|"HIDDEN"}` → `ProyectoOut`
  - `GET /proyectos/{id}/miembros` → `{"miembros": [{"user_id","email","papel","origen"}]}`
  - `POST /proyectos/{id}/miembros` `{"email", "papel": "VIEWER"|"CONTRIBUTOR"|"OWNER"}` → 201 `{"user_id"}`
  - `PATCH /proyectos/{id}/miembros/{user_id}` `{"papel"}` → 204
  - `DELETE /proyectos/{id}/miembros/{user_id}` → 204
  - `GET /proyectos/{id}/candidatos?q=` → `{"candidatos": [{"user_id","email"}]}`
  - `ProyectoOut = {"id", "uuid", "nombre", "descripcion", "estado", "papel"}`
  - Errores: `detail = {"code": <codigo>}` con esta tabla única (`_HTTP_DE_ERROR` en el módulo):

| Excepción de B9 | HTTP | code |
|---|---|---|
| `ProjectNotVisible`, `MemberNotFound` | 404 | `proyecto_no_encontrado` / `miembro_no_encontrado` |
| `ProjectRoleInsufficient` | 403 | `papel_insuficiente` |
| `ProjectStateConflict` | 409 | `estado_no_permite` |
| `LastOwnerRequired` | 409 | `ultimo_dueno` |
| `TenantAdminMembershipProtected` | 409 | `admin_protegido` |
| `AlreadyMember` | 409 | `ya_es_miembro` |
| `TargetUserNotEligible` | 422 | `usuario_no_elegible` |
| `IdempotencyKeyConflict` | 409 | `idempotencia_conflicto` |
| `InvalidIdempotencyKey` | 422 | `idempotencia_invalida` |
| `ReservedProjectIdRange` | 500 | `proyecto_id_reservado` (y `logger.error`) |
| `ProjectAuthorityRetryable` | 503 | `reintentar` |
| `AuthorizationDenied` (validación de nombre/papel) | 422 | `datos_invalidos` |

`ProjectNotVisible` hereda de `ScopeDenied`; el `except` de `ProjectNotVisible` va **antes** que cualquier `except ScopeDenied/AuthorizationDenied` genérico. Cualquier otra excepción → 500 `proyectos_error` con `logger.error(exc_info=True)`; el mensaje interno nunca va al cliente.

- [ ] **Step 1: Escribir las pruebas que fallan** — con el fixture `client` y las identidades reales de `backend/tests/identidades.py` (`crear_usuario`, `cabeceras`), en un tenant propio de la prueba. Casos mínimos (uno por función de prueba):
  - `test_crear_y_listar` — POST con `Idempotency-Key` → 201, aparece en `GET /proyectos` con `papel="OWNER"`, `estado="ACTIVE"`.
  - `test_crear_repetido_con_la_misma_llave_devuelve_el_mismo` — segunda llamada idéntica → 200 y mismo `id`; misma llave con otro nombre → 409 `idempotencia_conflicto`.
  - `test_crear_sin_llave_422`.
  - `test_otro_tenant_da_404_igual_que_inexistente` — `GET /proyectos/{id}` de un proyecto de otro tenant y `GET /proyectos/999999999` devuelven el mismo status y el mismo cuerpo.
  - `test_lector_no_invita_403` — VIEWER hace `POST /miembros` → 403 `papel_insuficiente`.
  - `test_ultimo_dueno_no_se_rebaja` — único OWNER (no admin) hace `PATCH /miembros/{su id}` a VIEWER → 409 `ultimo_dueno`, y sigue siendo OWNER.
  - `test_invitar_cambiar_quitar` — flujo completo con un usuario del mismo tenant.
  - `test_archivar_y_restaurar`, `test_ocultar_requiere_admin` (OWNER no admin → 403).
  - `test_vista_ocultos_403_para_no_admin`.
  - `test_cursor_no_numerico_422` (`antes_de=abc`) y `test_limite_fuera_de_rango_422` (`limite=0`, `limite=101`).
  - `test_renombrar_valida_nombre` (`""` → 422 `datos_invalidos`).
  - `test_candidatos_minimo_dos_letras`.
  - `test_ningun_500_filtra_mensaje_interno` — forzar con `monkeypatch` que `list_projects_for_user` lance `RuntimeError("secreto")` → 500 y `"secreto"` no aparece en la respuesta.

- [ ] **Step 2: Correr y ver que fallan** — `cd backend && set -a && . ~/.config/jax/test-db.env && set +a && ../.venv/bin/pytest tests/test_proyectos_api.py -v` (usar el comando con que la suite con DB corre hoy en `policy.yml`; copiarlo de ahí) → 404 en todas las rutas.

- [ ] **Step 3: Implementar `backend/api/proyectos.py`**

```python
"""Proyectos (E1, 2026-10-02): API sobre la autoridad de B9.

La plataforma NO decide papeles: toda mutación va a `ProjectAuthorityAdmin`
y toda lectura a `jax.memory.project_queries`, que releen la identidad en la
base. Aquí solo se arman el ScopeContext desde el usuario autenticado y se
traducen las excepciones de B9 a códigos HTTP estables (spec E1 §5.2).
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field

from auth.middleware import get_current_user
from auth.models import AuthUser
from db.connection import get_pool
from b9_pool import B9MappingPool                         # ver Interfaces: módulo compartido
from jax.memory.b9 import AuthorizationDenied, MutationAuthorizationRequest, ScopeContext, Visibility
from jax.memory.b9_mariadb import MariaDBB9Store
from jax.memory.project_authority import (
    AlreadyMember, IdempotencyKeyConflict, InvalidIdempotencyKey, LastOwnerRequired, MemberNotFound,
    ProjectAuthorityAdmin, ProjectAuthorityRetryable, ProjectNotVisible, ProjectRoleInsufficient,
    ProjectStateConflict, ReservedProjectIdRange, TargetUserNotEligible, TenantAdminMembershipProtected,
)
from jax.memory.project_queries import (ProjectView, get_project_for_user, list_invite_candidates,
                                        list_project_members, list_projects_for_user)
from jax.memory.scope_authority import ProjectLifecycle, ProjectRole

logger = logging.getLogger(__name__)
router = APIRouter()

# Orden importa: ProjectNotVisible hereda de ScopeDenied/AuthorizationDenied.
_HTTP_DE_ERROR: tuple[tuple[type[Exception], int, str], ...] = (
    (ProjectNotVisible, 404, "proyecto_no_encontrado"),
    (MemberNotFound, 404, "miembro_no_encontrado"),
    (ProjectRoleInsufficient, 403, "papel_insuficiente"),
    (ProjectStateConflict, 409, "estado_no_permite"),
    (LastOwnerRequired, 409, "ultimo_dueno"),
    (TenantAdminMembershipProtected, 409, "admin_protegido"),
    (AlreadyMember, 409, "ya_es_miembro"),
    (TargetUserNotEligible, 422, "usuario_no_elegible"),
    (IdempotencyKeyConflict, 409, "idempotencia_conflicto"),
    (InvalidIdempotencyKey, 422, "idempotencia_invalida"),
    (ProjectAuthorityRetryable, 503, "reintentar"),
    (ReservedProjectIdRange, 500, "proyecto_id_reservado"),
    (AuthorizationDenied, 422, "datos_invalidos"),
)

_PAPELES_ASIGNABLES = Literal["VIEWER", "CONTRIBUTOR", "OWNER"]
_ESTADOS = Literal["ACTIVE", "ARCHIVED", "HIDDEN"]


def _http(exc: Exception) -> HTTPException:
    for clase, status, code in _HTTP_DE_ERROR:
        if isinstance(exc, clase):
            if status >= 500:
                logger.error("proyectos: %s", code, exc_info=exc)
            return HTTPException(status_code=status, detail={"code": code})
    logger.error("proyectos: error inesperado", exc_info=exc)
    return HTTPException(status_code=500, detail={"code": "proyectos_error"})


def _ids(user: AuthUser) -> tuple[int, int]:
    if not user.tenant_id:
        raise HTTPException(status_code=403, detail={"code": "tenant_scope_required"})
    return int(user.tenant_id), int(user.user_id)


def _request(user: AuthUser, operation: str, project_id: int | None) -> MutationAuthorizationRequest:
    scope = ScopeContext(actor_principal=f"user:{user.user_id}", actor_type="USER",
                         subject_user_id=str(user.user_id), tenant_id=str(user.tenant_id),
                         project_id=str(project_id) if project_id is not None else None,
                         calling_component="jax-platform-proyectos")
    return MutationAuthorizationRequest(scope, operation, Visibility.PROJECT_SHARED)


async def _pool() -> B9MappingPool:
    return B9MappingPool(await get_pool())


async def _admin() -> ProjectAuthorityAdmin:
    return ProjectAuthorityAdmin(MariaDBB9Store(await _pool()))


def _out(p: dict) -> dict:
    return {"id": p["project_id"], "uuid": p["project_uuid"], "nombre": p["name"],
            "descripcion": p["description"], "estado": p["status"], "papel": p["role"]}


class ProyectoIn(BaseModel):
    nombre: str = Field(min_length=1, max_length=255)
    descripcion: str | None = Field(default=None, max_length=2000)


class EstadoIn(BaseModel):
    estado: _ESTADOS


class MiembroIn(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    papel: _PAPELES_ASIGNABLES


class PapelIn(BaseModel):
    papel: _PAPELES_ASIGNABLES


@router.get("/proyectos")
async def listar(vista: ProjectView = ProjectView.ACTIVOS, antes_de: int | None = Query(default=None, ge=1),
                 limite: int = Query(default=50, ge=1, le=100), user: AuthUser = Depends(get_current_user)):
    tenant_id, user_id = _ids(user)
    try:
        filas = await list_projects_for_user(await _pool(), tenant_id=tenant_id, user_id=user_id,
                                             view=vista, before_id=antes_de, limit=limite + 1)
    except Exception as exc:
        raise _http(exc) from exc
    siguiente = filas[limite - 1]["project_id"] if len(filas) > limite else None
    return {"proyectos": [_out(p) for p in filas[:limite]], "siguiente": siguiente}


async def _leer(user: AuthUser, project_id: int) -> dict:
    tenant_id, user_id = _ids(user)
    try:
        return _out(await get_project_for_user(await _pool(), tenant_id=tenant_id, user_id=user_id,
                                               project_id=project_id))
    except Exception as exc:
        raise _http(exc) from exc


@router.post("/proyectos", status_code=201)
async def crear(body: ProyectoIn, response: Response, user: AuthUser = Depends(get_current_user),
                idempotency_key: str = Header(alias="Idempotency-Key")):
    _ids(user)
    try:
        creado = await (await _admin()).create_project(_request(user, "CREATE_PROJECT", None),
                                                       name=body.nombre, description=body.descripcion,
                                                       idempotency_key=idempotency_key)
    except Exception as exc:
        raise _http(exc) from exc
    if not creado.created:
        response.status_code = 200
    return await _leer(user, creado.project_id)


@router.get("/proyectos/{project_id}")
async def ver(project_id: int, user: AuthUser = Depends(get_current_user)):
    return await _leer(user, project_id)


@router.patch("/proyectos/{project_id}")
async def renombrar(project_id: int, body: ProyectoIn, user: AuthUser = Depends(get_current_user)):
    _ids(user)
    try:
        await (await _admin()).rename_project(_request(user, "RENAME_PROJECT", project_id), project_id,
                                              name=body.nombre, description=body.descripcion)
    except Exception as exc:
        raise _http(exc) from exc
    return await _leer(user, project_id)


@router.post("/proyectos/{project_id}/estado")
async def cambiar_estado(project_id: int, body: EstadoIn, user: AuthUser = Depends(get_current_user)):
    _ids(user)
    try:
        await (await _admin()).set_project_lifecycle(_request(user, "SET_PROJECT_LIFECYCLE", project_id),
                                                     project_id, ProjectLifecycle(body.estado))
    except Exception as exc:
        raise _http(exc) from exc
    return await _leer(user, project_id)


@router.get("/proyectos/{project_id}/miembros")
async def miembros(project_id: int, user: AuthUser = Depends(get_current_user)):
    tenant_id, user_id = _ids(user)
    try:
        filas = await list_project_members(await _pool(), tenant_id=tenant_id, user_id=user_id,
                                           project_id=project_id)
    except Exception as exc:
        raise _http(exc) from exc
    return {"miembros": [{"user_id": m["user_id"], "email": m["email"], "papel": m["role"],
                          "origen": m["grant_origin"]} for m in filas]}


@router.post("/proyectos/{project_id}/miembros", status_code=201)
async def invitar(project_id: int, body: MiembroIn, user: AuthUser = Depends(get_current_user)):
    _ids(user)
    try:
        nuevo = await (await _admin()).grant_member(_request(user, "GRANT_MEMBER", project_id), project_id,
                                                    email=body.email, role=ProjectRole(body.papel))
    except Exception as exc:
        raise _http(exc) from exc
    return {"user_id": nuevo}


@router.patch("/proyectos/{project_id}/miembros/{user_id}", status_code=204)
async def cambiar_papel(project_id: int, user_id: int, body: PapelIn, user: AuthUser = Depends(get_current_user)):
    _ids(user)
    try:
        await (await _admin()).change_project_role(_request(user, "CHANGE_PROJECT_ROLE", project_id),
                                                   project_id, user_id, ProjectRole(body.papel))
    except Exception as exc:
        raise _http(exc) from exc
    return Response(status_code=204)


@router.delete("/proyectos/{project_id}/miembros/{user_id}", status_code=204)
async def quitar(project_id: int, user_id: int, user: AuthUser = Depends(get_current_user)):
    _ids(user)
    try:
        await (await _admin()).revoke_member(_request(user, "REVOKE_MEMBER", project_id), project_id, user_id)
    except Exception as exc:
        raise _http(exc) from exc
    return Response(status_code=204)


@router.get("/proyectos/{project_id}/candidatos")
async def candidatos(project_id: int, q: str = Query(default="", max_length=320),
                     user: AuthUser = Depends(get_current_user)):
    tenant_id, user_id = _ids(user)
    try:
        filas = await list_invite_candidates(await _pool(), tenant_id=tenant_id, user_id=user_id,
                                             project_id=project_id, query=q, limit=20)
    except Exception as exc:
        raise _http(exc) from exc
    return {"candidatos": filas}
```

Antes de escribirlo, **verificar contra el código** (no suponer): que los nombres de operación siguen siendo `GRANT_MEMBER`, `CHANGE_PROJECT_ROLE`, `REVOKE_MEMBER` y `SET_PROJECT_LIFECYCLE` (verificados 2026-10-02 en `project_authority.py:690,749,788,824`), y que los imports (`auth.middleware.get_current_user`, `auth.models.AuthUser`, `db.connection.get_pool`, `ProjectRole`/`ProjectLifecycle` de `jax.memory.scope_authority`) siguen donde estaban (verificados 2026-10-02). `ProjectRoleInsufficient` hereda de `AuthorizationDenied`: por eso va antes en `_HTTP_DE_ERROR`. Si `get_current_user` no corta a un usuario que debe cambiar su contraseña, usar la dependencia que use `chat.py` para lo mismo.

- [ ] **Step 4: Registrar el router** en `backend/main.py` igual que los otros (`from api.proyectos import router as proyectos_router` y agregarlo a la lista que recorre `include_router`).

- [ ] **Step 5: Correr y ver que pasan.** Luego la suite con DB completa: no puede bajar de 3071 passed; el piso se sube al número nuevo medido en el runner (Tarea 10).

- [ ] **Step 6: Commit** — `"feat(proyectos): API /proyectos sobre la autoridad de B9 (E1, T5)"`.

---

### Task 6: Cliente y textos (frontend)

**Files:**
- Create: `frontend/src/api/proyectos.js`, `frontend/src/api/proyectos.test.js`
- Modify: `frontend/src/i18n/es.js`, `frontend/src/i18n/en.js` (bloque `proyectos`)

**Interfaces:**
- Produces:
  - `listarProyectos({ vista = 'activos', antesDe = null, limite = 50 }) -> Promise<{ proyectos, siguiente }>`
  - `crearProyecto({ nombre, descripcion }, idempotencyKey) -> Promise<Proyecto>`
  - `verProyecto(id)`, `renombrarProyecto(id, { nombre, descripcion })`, `cambiarEstado(id, estado)`
  - `listarMiembros(id)`, `invitarMiembro(id, { email, papel })`, `cambiarPapel(id, userId, papel)`, `quitarMiembro(id, userId)`, `buscarCandidatos(id, q)`
  - i18n: `t.proyectos.{titulo, nuevo, vistas: {activos, archivados, ocultos}, papeles: {VIEWER, CONTRIBUTOR, OWNER}, estados: {ACTIVE, ARCHIVED, HIDDEN}, errores: {proyecto_no_encontrado, miembro_no_encontrado, papel_insuficiente, estado_no_permite, ultimo_dueno, admin_protegido, ya_es_miembro, usuario_no_elegible, idempotencia_conflicto, idempotencia_invalida, reintentar, datos_invalidos, generico}, ...}` (los textos de pantalla que usen las Tareas 7–9 se agregan aquí; la Tarea 7–9 no escribe ningún literal visible).

- [ ] **Step 1: Pruebas que fallan** (vitest, mockeando `api` como hacen `client.test.js`): cada función llama al método y ruta correctos; `crearProyecto` manda la cabecera `Idempotency-Key`; `listarProyectos` pasa `vista`, `antes_de`, `limite` como query; y una prueba que recorra `Object.keys(es.proyectos)` y exija la misma forma en `en.proyectos` (mismas claves, en profundidad).
- [ ] **Step 2: Verlas fallar.** `cd frontend && npx vitest run src/api/proyectos.test.js`
- [ ] **Step 3: Implementar** (`import api from './client'`; cada función `const { data } = await api.<metodo>(...); return data`).
- [ ] **Step 4: Verlas pasar.**
- [ ] **Step 5: Commit** — `"feat(proyectos): cliente y textos i18n (E1, T6)"`.

---

### Task 7: Pantalla `/proyectos` (lista, pestañas, crear)

**Files:**
- Create: `frontend/src/pages/Proyectos.jsx`, `frontend/src/pages/Proyectos.test.jsx`, `frontend/src/components/proyectos/CrearProyectoModal.jsx`
- Modify: `frontend/src/App.jsx` (ruta `/proyectos` y `/proyectos/:id` dentro de `RequireAuth`, como `/historial`), el menú/navegación donde está el enlace a Historial (buscar `to="/historial"` y agregar el de Proyectos al lado)

**Interfaces:**
- Consumes: Tarea 6.
- Produces: `<Proyectos />` (lista) y la ruta `/proyectos/:id` que monta `<ProyectoDetalle />` (Tarea 8).

Comportamiento:
- Pestañas Activos / Archivados / Ocultos; «Ocultos» solo si `user.role` es admin del tenant (mismo criterio que usa el resto del frontend para mostrar Admin).
- Lista paginada: botón «Cargar más» mientras `siguiente !== null`.
- «Nuevo proyecto» abre `CrearProyectoModal` (sobre `Dialogo`). La `Idempotency-Key` se genera con `crypto.randomUUID()` **al abrir** el modal y se reusa en reintentos; el botón Crear se deshabilita mientras la petición está en vuelo.
- Errores: `codigoDe(err)` → `t.proyectos.errores[code] ?? t.proyectos.errores.generico`.
- Tokens de color (`bg-superficie`, `text-texto`, etc., los mismos de `AdminUsers.jsx`), botones con `TAMANO_BOTON_ACCION`.

- [ ] **Step 1: Pruebas que fallan** (Testing Library, como `Historial.test.jsx`): renderiza la lista de `listarProyectos` mockeado; cambiar de pestaña pide `vista='archivados'`; un no-admin no ve «Ocultos»; «Cargar más» pide `antesDe=<siguiente>`; **doble clic en Crear manda una sola petición y con la misma llave**; un 409 muestra el texto traducido; no hay ningún `window.confirm/alert/prompt` (espiarlos y afirmar 0 llamadas).
- [ ] **Step 2–4:** fallar, implementar, pasar.
- [ ] **Step 5: Commit** — `"feat(proyectos): pantalla de proyectos y crear (E1, T7)"`.

---

### Task 8: Detalle del proyecto — Miembros y Ajustes

**Files:**
- Create: `frontend/src/pages/ProyectoDetalle.jsx`, `frontend/src/pages/ProyectoDetalle.test.jsx`, `frontend/src/components/proyectos/Miembros.jsx`, `frontend/src/components/proyectos/Ajustes.jsx`

**Interfaces:**
- Consumes: Tarea 6; `ConfirmacionSuma` para ocultar (acción de admin que saca el proyecto de la vista de todos).

Comportamiento:
- Cabecera con nombre, estado y tu papel. Pestañas **Miembros** y **Ajustes**.
- Miembros: lista; si eres OWNER y el proyecto está ACTIVE: buscador de candidatos (debounce 300 ms, mínimo 2 letras) + papel + Invitar; por fila, selector de papel (lector/editor/dueño) y Quitar (confirmación en `Dialogo`). Las filas con `origen === 'TENANT_ADMIN'` no muestran controles (B9 las protege; la UI no ofrece lo que va a fallar).
- Ajustes (OWNER): renombrar; Archivar (confirmación en `Dialogo`); si archivado: Restaurar; si admin y archivado: Ocultar (`ConfirmacionSuma`); si admin y oculto: Mostrar (vuelve a archivado).
- 404 → mensaje «no encontrado» y enlace a `/proyectos` (sin decir por qué).
- Tras cualquier mutación, recargar proyecto y miembros desde la API (no editar el estado local a mano).

- [ ] **Step 1: Pruebas que fallan**: un VIEWER no ve controles; un OWNER invita (llama `invitarMiembro` con email y papel); cambiar papel del último dueño muestra el texto de `ultimo_dueno`; filas TENANT_ADMIN sin controles; archivar pide confirmación y no llama a la API si se cancela; Ocultar usa `ConfirmacionSuma`; 404 muestra el aviso; sin diálogos del navegador.
- [ ] **Step 2–4.**
- [ ] **Step 5: Commit** — `"feat(proyectos): miembros y ajustes del proyecto (E1, T8)"`.

---

### Task 9: Selector de proyecto en el chat

**Files:**
- Create: `frontend/src/components/BottomBar/SelectorDeProyecto.jsx`, `.test.jsx`
- Modify: `frontend/src/store/useJaxStore.js` (estado `proyectoActivo: null | {id, nombre}` y `setProyectoActivo`), `frontend/src/components/BottomBar/BottomBar.jsx` (~línea 227: agregar `project_id: proyectoActivo?.id ?? null` a `chatBody` solo cuando hay proyecto), y su test existente.

**Interfaces:**
- Consumes: `listarProyectos({ vista: 'activos' })`.
- Produces: `proyectoActivo` en el store; `/api/chat` recibe `project_id` solo desde aquí.

Comportamiento:
- Opciones: «Personal» + proyectos ACTIVE donde eres miembro (primera página, 100). Se recarga al abrir el selector.
- Si el proyecto elegido ya no está en la lista al recargar (archivado, quitado), vuelve a «Personal» y muestra un aviso i18n; nunca manda ese `project_id`.
- Si `/api/chat` responde 403 `project_scope_denied`, vuelve a «Personal» con el mismo aviso y no reintenta solo.
- La conversación cambia con el proyecto: al cambiar de proyecto se limpia la conversación visible igual que hoy al cambiar de faceta (mirar qué hace `setActiveFacet` y replicar el criterio; si hoy no se limpia, NO inventarlo: dejar escrito en el PR qué pasa con el historial).

- [ ] **Step 1: Pruebas que fallan**: sin proyecto, `chatBody` no lleva `project_id`; con proyecto, lo lleva; un proyecto que desaparece de la lista se des-elige con aviso; un 403 `project_scope_denied` des-elige; el test existente de `BottomBar` sigue verde.
- [ ] **Step 2–4.**
- [ ] **Step 5: Commit** — `"feat(proyectos): selector de proyecto en el chat (E1, T9)"`.

---

### Task 10: Las Cuatro, pisos y cierre de la rama

**Files:**
- Create: `loadtest/proyectos_e1.py` (siguiendo el patrón de los guiones de `loadtest/` existentes: credenciales de PRUEBA, nunca `/etc/jax/.env`)
- Modify: `.github/workflows/policy.yml` (pisos re-medidos)
- Create: `docs/historia/2026-10-XX-proyectos-e1.md` en jax (Biblioteca), con la fecha real

- [ ] **Step 1: Barrido de política**: `grep -rnE "\b(window\.)?(confirm|alert|prompt)\(" frontend/src --include=*.jsx --include=*.js | grep -v test` → sin resultados nuevos; ningún literal visible fuera de i18n en los archivos nuevos; revisar en claro y oscuro con el navegador (chrome-devtools: captura de `/proyectos` y del detalle en los dos temas).
- [ ] **Step 2: Índices**: `EXPLAIN` de las consultas de `project_queries.py` contra la base de prueba con 1.000 proyectos sembrados y un usuario miembro de 300; pegar los planes en el PR. Sin `Using filesort`/`Using temporary` en la lista.
- [ ] **Step 3: Carga**: `loadtest/proyectos_e1.py` contra la plataforma local con base de prueba: 50 usuarios concurrentes listando proyectos y 20 enviando turnos de chat con proyecto (facetas falsas como en los loadtests existentes) durante 2 min; registrar rps, p95 y desde cuántos usuarios se degrada. El número va a la Biblioteca.
- [ ] **Step 4: Pisos**: correr la suite completa (con DB, sin DB y vitest) en el runner de CI y subir los pisos exactos en `policy.yml` al número medido allí (Regla 5 de SESIONES EN PARALELO: si `master` se movió, re-medir sobre `master`).
- [ ] **Step 5: Commit** y auditoría de escalón 3 de la rama completa antes del PR.

---

## Parte C — producción (sesión principal, con GO de Fernando)

### Task 11: Despliegue coordinado

- [ ] Integrar PR de jax (Parte A) y PR de jax-platform (Parte B) con CI verde y auditoría sin BLOCK/MAJOR.
- [ ] Avisar a la sesión que coordina el despliegue de jax-platform (pendiente de #170–#174) y desplegar **una sola vez** (incluye publicar el sitio de atem-ai).
- [ ] Correr el runbook de la Tarea 4 en producción, con GO de Fernando o su ventana: respaldo verificado → `proyectos_e1_migrar.py` → FKs que pasaron el ensayo.
- [ ] Verificación por efecto: Fernando abre `/proyectos`, ve HAMURABI y el proyecto archivado de evaluación; chatea con HAMURABI; un usuario de prueba no lo ve.
- [ ] `PENDIENTES.md`: nada que cerrar de E1 en sí (las líneas del Selector y del respaldo del chat siguen abiertas para E2/E3); anotar en la del Selector «E1 desplegado <fecha>, <SHA>».
