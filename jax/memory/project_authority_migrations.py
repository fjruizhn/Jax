"""Tracked, never-auto-applied DDL for JAX project authorization.

The tables are in the shared physical ``jax_memory`` schema but are an
identity/authorization namespace.  ``project_id`` deliberately matches the
deployed ``projects.id`` INT key; legacy projects gain no inferred scope.

Migration 005 (project lifecycle / D1-D5, section 3-bis) lives ONLY here, as a
Python hook guarded by ``information_schema`` checks -- deliberately not a
second ``.sql`` copy (003 already has that duplication, H2; the README next
to this module documents why 005 does not repeat it).
"""
from __future__ import annotations

import re
from typing import Any

PROJECT_AUTHORITY_DDL = (
    """CREATE TABLE IF NOT EXISTS jax_project_scope (
      project_id INT NOT NULL,
      tenant_id INT NOT NULL,
      authorization_namespace VARCHAR(32) NOT NULL DEFAULT 'CLASSIC',
      status VARCHAR(16) NOT NULL,
      created_at DATETIME(6) NOT NULL,
      created_by VARCHAR(128) NOT NULL,
      updated_at DATETIME(6) NOT NULL,
      version BIGINT NOT NULL DEFAULT 1,
      PRIMARY KEY (project_id),
      UNIQUE KEY uq_jax_project_scope_project_tenant (project_id, tenant_id),
      KEY idx_jax_project_scope_tenant_status (tenant_id, status),
      KEY idx_jax_project_scope_namespace_tenant_status_project (authorization_namespace, tenant_id, status, project_id),
      KEY idx_jax_project_scope_namespace_status_project (authorization_namespace, status, project_id),
      CONSTRAINT fk_jax_project_scope_project FOREIGN KEY (project_id) REFERENCES projects(id),
      CONSTRAINT fk_jax_project_scope_tenant FOREIGN KEY (tenant_id) REFERENCES jax_tenants(tenant_id),
      CONSTRAINT chk_jax_project_scope_status CHECK (status IN ('ACTIVE','DISABLED')),
      CONSTRAINT chk_jax_project_scope_namespace CHECK (authorization_namespace IN ('CLASSIC','OFFICE_CONFIDENTIAL'))
    ) ENGINE=InnoDB""",
    """CREATE TABLE IF NOT EXISTS jax_project_membership (
      membership_id CHAR(36) NOT NULL,
      project_id INT NOT NULL,
      tenant_id INT NOT NULL,
      user_id INT NOT NULL,
      project_role VARCHAR(16) NOT NULL,
      status VARCHAR(16) NOT NULL,
      created_at DATETIME(6) NOT NULL,
      created_by VARCHAR(128) NOT NULL,
      updated_at DATETIME(6) NOT NULL,
      version BIGINT NOT NULL DEFAULT 1,
      PRIMARY KEY (membership_id),
      UNIQUE KEY uq_jax_project_membership_subject (project_id, user_id),
      KEY idx_jax_project_membership_lookup (project_id, user_id, status),
      KEY idx_jax_project_membership_tenant_user (tenant_id, user_id),
      CONSTRAINT fk_jax_project_membership_scope FOREIGN KEY (project_id, tenant_id)
        REFERENCES jax_project_scope(project_id, tenant_id),
      CONSTRAINT fk_jax_project_membership_user FOREIGN KEY (user_id) REFERENCES jax_users(user_id),
      CONSTRAINT chk_jax_project_membership_role CHECK (project_role IN ('VIEWER','CONTRIBUTOR','REVIEWER','OWNER')),
      CONSTRAINT chk_jax_project_membership_status CHECK (status IN ('ACTIVE','REVOKED'))
    ) ENGINE=InnoDB""",
    """CREATE TABLE IF NOT EXISTS jax_project_membership_event (
      event_id CHAR(36) NOT NULL,
      operation VARCHAR(32) NOT NULL,
      actor_principal VARCHAR(255) NOT NULL,
      actor_type VARCHAR(64) NOT NULL,
      project_id INT NOT NULL,
      target_user_id INT NULL,
      tenant_id INT NOT NULL,
      old_project_role VARCHAR(16) NULL,
      new_project_role VARCHAR(16) NULL,
      old_status VARCHAR(16) NULL,
      new_status VARCHAR(16) NULL,
      occurred_at DATETIME(6) NOT NULL,
      request_id VARCHAR(255) NULL,
      trace_id VARCHAR(255) NULL,
      PRIMARY KEY (event_id),
      KEY idx_jax_project_membership_event_project (project_id, occurred_at),
      CONSTRAINT fk_jax_project_membership_event_scope FOREIGN KEY (project_id, tenant_id)
        REFERENCES jax_project_scope(project_id, tenant_id)
    ) ENGINE=InnoDB""",
    """CREATE TRIGGER IF NOT EXISTS no_update_jax_project_membership_event
       BEFORE UPDATE ON jax_project_membership_event FOR EACH ROW
       SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'jax_project_membership_event append-only'""",
    """CREATE TRIGGER IF NOT EXISTS no_delete_jax_project_membership_event
       BEFORE DELETE ON jax_project_membership_event FOR EACH ROW
       SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'jax_project_membership_event append-only'""",
)

#: Constraint/index names 005 adds or drops. Named module constants so the
#: guard checks and the revert function (which touches the very same names)
#: cannot drift from each other.
_CHK_SCOPE_STATUS_V1 = "chk_jax_project_scope_status"
_CHK_SCOPE_STATUS_V2 = "chk_jax_project_scope_status_v2"
_CHK_MEMBERSHIP_ORIGIN = "chk_jax_project_membership_origin"
_CHK_MEMBERSHIP_ORIGIN_V2 = "chk_jax_project_membership_origin_v2"
_CHK_MEMBERSHIP_PRE_ADMIN = "chk_jax_project_membership_pre_admin"
_CHK_NAMESPACE_MEMBERSHIP_EXECUTION = "chk_jax_namespace_membership_mutation_execution"
_CHK_NAMESPACE_MEMBERSHIP_RESULT = "chk_jax_namespace_membership_mutation_result"
_CHK_NAMESPACE_MEMBERSHIP_RESULT_V2 = "chk_jax_namespace_membership_mutation_result_v2"
_CHK_NAMESPACE_MEMBERSHIP_RESULT_V3 = "chk_jax_namespace_membership_mutation_result_v3"
_CHK_NAMESPACE_MEMBERSHIP_RESULT_V4 = "chk_jax_namespace_membership_mutation_result_v4"
_UQ_NAMESPACE_MEMBERSHIP_INVITATION = "uq_jax_namespace_membership_mutation_invitation"
_IDX_MEMBERSHIP_USER_LIST = "idx_jax_project_membership_user_list"
#: Ronda 4, MAJOR MJ-2: `jax_users` is platform-owned (jax-platform's own
#: migrations create the table), but the admin-set read this module needs
#: (`_read_tenant_admins`) has to hit an index by tenant, or MariaDB falls
#: back to a full table scan under REPEATABLE READ -- which next-key-locks
#: rows across every tenant, not just the one being read (worse than the
#: `FOR UPDATE` this round removes). Declared and guarded here rather than
#: in jax-platform's own migrations because this module is the only reader
#: that needs it, same reasoning as 005e for jax_project_membership.
_IDX_USERS_TENANT_ROLE_STATUS = "idx_jax_users_tenant_role_status"
#: E1.1 (2026-10-02, adenda §5.6.1): `list_invite_candidates` lists the tenant's
#: users ordered by email (with or without an email-prefix filter). Without
#: `(tenant_id, email)` MariaDB filesorts every tenant user on each call.
#: Lives here although `jax_users` is platform-owned, for the same reason as
#: `idx_jax_users_tenant_role_status`: this module is the only reader that
#: needs it, and it is guarded so a re-run is a no-op.
_IDX_USERS_TENANT_EMAIL = "idx_jax_users_tenant_email"
_CHK_SCOPE_NAMESPACE = "chk_jax_project_scope_namespace"
_IDX_SCOPE_NAMESPACE_TENANT_STATUS_PROJECT = "idx_jax_project_scope_namespace_tenant_status_project"
# The Office root list spans tenants.  Its predicates and cursor ordering are
# namespace/status/project_id, so the tenant-scoped index above cannot serve
# it without a gap between the equality predicates and the ordered cursor.
_IDX_SCOPE_NAMESPACE_STATUS_PROJECT = "idx_jax_project_scope_namespace_status_project"

_MANAGED_INDEXES: dict[tuple[str, str], tuple[tuple[str, ...], bool]] = {
    ("jax_project_scope", _IDX_SCOPE_NAMESPACE_TENANT_STATUS_PROJECT):
        (("authorization_namespace", "tenant_id", "status", "project_id"), False),
    ("jax_project_scope", _IDX_SCOPE_NAMESPACE_STATUS_PROJECT):
        (("authorization_namespace", "status", "project_id"), False),
    ("jax_project_membership", _IDX_MEMBERSHIP_USER_LIST):
        (("tenant_id", "user_id", "status", "project_id"), False),
    ("jax_namespace_membership_mutation_request", _UQ_NAMESPACE_MEMBERSHIP_INVITATION):
        (("namespace", "invitation_id"), True),
    ("jax_users", _IDX_USERS_TENANT_ROLE_STATUS): (("tenant_id", "role", "status"), False),
    ("jax_users", _IDX_USERS_TENANT_EMAIL): (("tenant_id", "email"), False),
}

_MANAGED_CHECKS = {
    _CHK_SCOPE_STATUS_V1: "status IN ('ACTIVE','DISABLED')",
    _CHK_SCOPE_STATUS_V2: "status IN ('ACTIVE','ARCHIVED','HIDDEN','DISABLED')",
    _CHK_SCOPE_NAMESPACE: "authorization_namespace IN ('CLASSIC','OFFICE_CONFIDENTIAL')",
    _CHK_MEMBERSHIP_ORIGIN: "grant_origin IN ('EXPLICIT','CREATOR','TENANT_ADMIN')",
    _CHK_MEMBERSHIP_ORIGIN_V2: "grant_origin IN ('EXPLICIT','CREATOR','TENANT_ADMIN','OFFICE')",
    _CHK_MEMBERSHIP_PRE_ADMIN: "(pre_admin_role IS NULL AND pre_admin_status IS NULL) OR (grant_origin='TENANT_ADMIN' AND pre_admin_role IN ('VIEWER','CONTRIBUTOR','REVIEWER','OWNER') AND pre_admin_status IN ('ACTIVE','REVOKED'))",
    _CHK_NAMESPACE_MEMBERSHIP_EXECUTION: "(execution_kind='INTERACTIVE_USER' AND invitation_id IS NULL) OR (execution_kind='INVITATION_ACCEPTANCE' AND invitation_id IS NOT NULL)",
    _CHK_NAMESPACE_MEMBERSHIP_RESULT: "project_role IN ('VIEWER','CONTRIBUTOR') AND status IN ('ACTIVE','REVOKED') AND changed=1",
    _CHK_NAMESPACE_MEMBERSHIP_RESULT_V2: "project_role IN ('VIEWER','CONTRIBUTOR','REVIEWER','OWNER') AND status IN ('ACTIVE','REVOKED') AND changed=1",
    _CHK_NAMESPACE_MEMBERSHIP_RESULT_V4: "((operation='GRANT_OFFICE_MEMBER' AND member_kind IS NOT NULL AND member_kind='INTERNAL' AND project_role='CONTRIBUTOR' AND status='ACTIVE') OR (operation='GRANT_OFFICE_MEMBER' AND member_kind IS NOT NULL AND member_kind='EXTERNAL_CLIENT' AND project_role='VIEWER' AND status='ACTIVE') OR (operation='REVOKE_OFFICE_MEMBER' AND member_kind IS NULL AND project_role IN ('VIEWER','CONTRIBUTOR','REVIEWER','OWNER') AND status='REVOKED')) AND changed=1",
}

_LEGACY_STATUS_ENUM = "ENUM('planning','active','paused','completed','archived')"
_LIFECYCLE_STATUS_ENUM = ("ENUM('planning','active','paused','completed','archived','hidden','disabled') "
                          "DEFAULT 'planning'")

#: E2a (2026-10-03, adenda E2 §3.1): un documento subido a un proyecto. Vive acá,
#: aunque la escribe jax-platform, por la misma razón que 005h/005i: este hook es el
#: que jax-platform corre en cada arranque (`db/migrations.py:124-142`), y la lee
#: también LAS MANOS en E2b. CREATE ... IF NOT EXISTS: re-correrlo no hace nada.
_DDL_PROJECT_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS project_documents (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  project_id INT(11) NOT NULL,
  sha256 CHAR(64) NOT NULL,
  nombre_original VARCHAR(1024) NOT NULL,
  ruta_entrada VARCHAR(1024) NULL,
  carpeta_procesado VARCHAR(255) NULL,
  bytes BIGINT UNSIGNED NOT NULL,
  tipo VARCHAR(16) NOT NULL,
  estado ENUM('en_cola','pendiente','procesando','listo','parcial','error','sin_extractor','cancelado')
    NOT NULL DEFAULT 'en_cola',
  error VARCHAR(1000) NULL,
  job_id VARCHAR(64) NULL,
  subido_por INT(11) NOT NULL,
  oculto_at DATETIME(6) NULL,
  oculto_por INT(11) NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  UNIQUE KEY uq_project_documents_sha (project_id, sha256),
  KEY idx_project_documents_lista (project_id, oculto_at, id),
  KEY idx_project_documents_despacho (estado, job_id, id),
  CONSTRAINT fk_project_documents_project FOREIGN KEY (project_id) REFERENCES projects (id),
  CONSTRAINT fk_project_documents_subido_por FOREIGN KEY (subido_por) REFERENCES jax_users (user_id),
  CONSTRAINT fk_project_documents_oculto_por FOREIGN KEY (oculto_por) REFERENCES jax_users (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""


def _scalar(row: Any) -> int:
    if row is None:
        return 0
    value = row.get("n") if isinstance(row, dict) else row[0]
    return int(value or 0)


async def _check_constraint_exists(cursor: Any, name: str) -> bool:
    await cursor.execute(
        "SELECT CHECK_CLAUSE FROM information_schema.CHECK_CONSTRAINTS "
        "WHERE CONSTRAINT_SCHEMA=DATABASE() AND CONSTRAINT_NAME=%s", (name,))
    row = await cursor.fetchone()
    if row is None:
        return False
    expected = _MANAGED_CHECKS.get(name)
    if expected is not None:
        actual = _field(row, "CHECK_CLAUSE", 0)
        # MariaDB renders string literals with a charset introducer in
        # information_schema (for example ``_utf8mb4'ACTIVE'``); it does not
        # change the CHECK semantics and must not make a fresh schema look
        # drifted.
        normalize = lambda value: re.sub(r"_UTF8MB4|[\s`]+", "", str(value).upper())
        if normalize(actual) != normalize(expected):
            raise RuntimeError(f"cannot migrate project authority: {name} has an unexpected CHECK expression")
    return True


async def _index_exists(cursor: Any, table: str, name: str) -> bool:
    expected = _MANAGED_INDEXES.get((table, name))
    if expected is None:
        await cursor.execute(
            "SELECT COUNT(*) AS n FROM information_schema.STATISTICS "
            "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND INDEX_NAME=%s", (table, name))
        return _scalar(await cursor.fetchone()) > 0
    columns, unique = expected
    shape = await _index_shape(cursor, table, name, columns, unique=unique)
    if shape == "drifted":
        raise RuntimeError(f"cannot migrate project authority: {name} has an unexpected definition")
    return shape == "exact"


def _field(row: Any, name: str, position: int) -> Any:
    return row.get(name) if isinstance(row, dict) else row[position]


async def _index_shape(cursor: Any, table: str, name: str,
                       columns: tuple[str, ...], *, unique: bool = False) -> str:
    """Return ``absent``, ``exact``, or ``drifted`` for a managed index.

    Index names are part of this migration's schema contract.  Treating a
    same-name index with another order, prefix length, type, uniqueness, or ignored state as
    installed would silently defeat a bounded Office list; replacing it would
    mutate an index this migration does not own.  MariaDB versions before the
    ``IGNORED`` metadata column are handled without selecting that column.
    """
    await cursor.execute(
        "SELECT COUNT(*) AS n FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA='information_schema' AND TABLE_NAME='STATISTICS' "
        "AND COLUMN_NAME='IGNORED'")
    has_ignored_metadata = _scalar(await cursor.fetchone()) > 0
    select = "COLUMN_NAME,SUB_PART,NON_UNIQUE,INDEX_TYPE"
    if has_ignored_metadata:
        select += ",IGNORED"
    await cursor.execute(
        "SELECT " + select + " FROM information_schema.STATISTICS "
        "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND INDEX_NAME=%s "
        "ORDER BY SEQ_IN_INDEX", (table, name))
    rows = await cursor.fetchall()
    if not rows:
        return "absent"
    actual_columns = tuple(str(_field(row, "COLUMN_NAME", 0)) for row in rows)
    btree_and_cardinality = all(
        _field(row, "SUB_PART", 1) is None
        and int(_field(row, "NON_UNIQUE", 2)) == (0 if unique else 1)
        and str(_field(row, "INDEX_TYPE", 3)).upper() == "BTREE"
        for row in rows)
    not_ignored = (not has_ignored_metadata or all(
        str(_field(row, "IGNORED", 4)).upper() in ("", "0", "NO", "FALSE")
        for row in rows))
    return "exact" if actual_columns == columns and btree_and_cardinality and not_ignored else "drifted"


async def _column_type(cursor: Any, table: str, column: str) -> str | None:
    await cursor.execute(
        "SELECT COLUMN_TYPE FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND COLUMN_NAME=%s", (table, column))
    row = await cursor.fetchone()
    if row is None:
        return None
    return row.get("COLUMN_TYPE") if isinstance(row, dict) else row[0]


async def apply_project_authority_migration(cursor: Any) -> None:
    """Execute reviewed idempotent DDL on an already-owned migration cursor.

    No application import, request path, or worker calls this function.
    """
    for statement in PROJECT_AUTHORITY_DDL:
        await cursor.execute(statement)
    await _apply_project_lifecycle_migration(cursor)


async def _apply_project_lifecycle_migration(cursor: Any) -> None:
    """005: ARCHIVED/HIDDEN/DISABLED lifecycle, membership provenance
    (grant_origin/pre_admin_*), the tenant-user-list index and
    ``jax_project_creation_request`` (idempotent CREATE_PROJECT). Every step
    is individually guarded so a second call is a no-op (`apply -> apply`
    idempotence, per the plan's migration test).
    """
    # A managed same-name index with a foreign definition is schema drift, not
    # an invitation to mutate this database.  Check before this function's
    # first ALTER so the failed upgrade remains observational.
    office_list_index = await _index_shape(
        cursor, "jax_project_scope", _IDX_SCOPE_NAMESPACE_STATUS_PROJECT,
        ("authorization_namespace", "status", "project_id"))
    if office_list_index == "drifted":
        raise RuntimeError(
            "cannot upgrade project authority: " + _IDX_SCOPE_NAMESPACE_STATUS_PROJECT
            + " has an unexpected definition")
    # Namespace comes before the lifecycle CHECK. That ordering matches a fresh
    # authority schema and keeps apply -> revert -> apply structurally stable.
    # Adding the NOT NULL column with its default backfills all legacy scopes as
    # CLASSIC without rewriting their identity or lifecycle data.
    if await _column_type(cursor, "jax_project_scope", "authorization_namespace") is None:
        await cursor.execute(
            "ALTER TABLE jax_project_scope ADD COLUMN authorization_namespace "
            "VARCHAR(32) NOT NULL DEFAULT 'CLASSIC' AFTER tenant_id")
    if not await _check_constraint_exists(cursor, _CHK_SCOPE_NAMESPACE):
        await cursor.execute(
            "ALTER TABLE jax_project_scope ADD CONSTRAINT " + _CHK_SCOPE_NAMESPACE + " "
            "CHECK (authorization_namespace IN ('CLASSIC','OFFICE_CONFIDENTIAL'))")
    if not await _index_exists(cursor, "jax_project_scope", _IDX_SCOPE_NAMESPACE_TENANT_STATUS_PROJECT):
        await cursor.execute(
            "CREATE INDEX " + _IDX_SCOPE_NAMESPACE_TENANT_STATUS_PROJECT + " "
            "ON jax_project_scope (authorization_namespace, tenant_id, status, project_id)")
    if office_list_index == "absent":
        await cursor.execute(
            "CREATE INDEX " + _IDX_SCOPE_NAMESPACE_STATUS_PROJECT + " "
            "ON jax_project_scope (authorization_namespace, status, project_id)")
    # 005a/005b: widen the CHECK, then drop the old one -- in that order, so
    # the table is never left without a status CHECK at all.
    if not await _check_constraint_exists(cursor, _CHK_SCOPE_STATUS_V2):
        await cursor.execute(
            "ALTER TABLE jax_project_scope ADD CONSTRAINT " + _CHK_SCOPE_STATUS_V2 + " "
            "CHECK (status IN ('ACTIVE','ARCHIVED','HIDDEN','DISABLED'))")
    if await _check_constraint_exists(cursor, _CHK_SCOPE_STATUS_V1):
        await cursor.execute("ALTER TABLE jax_project_scope DROP CONSTRAINT " + _CHK_SCOPE_STATUS_V1)
    # 005c
    if await _column_type(cursor, "jax_project_membership", "grant_origin") is None:
        await cursor.execute(
            "ALTER TABLE jax_project_membership ADD COLUMN grant_origin "
            "VARCHAR(16) NOT NULL DEFAULT 'EXPLICIT' AFTER status")
    if await _column_type(cursor, "jax_project_membership", "pre_admin_role") is None:
        await cursor.execute(
            "ALTER TABLE jax_project_membership ADD COLUMN pre_admin_role VARCHAR(16) NULL AFTER grant_origin")
    if await _column_type(cursor, "jax_project_membership", "pre_admin_status") is None:
        await cursor.execute(
            "ALTER TABLE jax_project_membership ADD COLUMN pre_admin_status VARCHAR(16) NULL AFTER pre_admin_role")
    # 005d
    # On an already-upgraded database only V2 exists.  Recreating V1 there
    # would reject valid OFFICE rows before we get a chance to inspect it.
    if (not await _check_constraint_exists(cursor, _CHK_MEMBERSHIP_ORIGIN)
            and not await _check_constraint_exists(cursor, _CHK_MEMBERSHIP_ORIGIN_V2)):
        await cursor.execute(
            "ALTER TABLE jax_project_membership ADD CONSTRAINT " + _CHK_MEMBERSHIP_ORIGIN + " "
            "CHECK (grant_origin IN ('EXPLICIT','CREATOR','TENANT_ADMIN'))")
    # Office memberships are a distinct provenance.  Add the widening CHECK
    # before dropping the legacy one so apply never leaves the table without
    # a provenance constraint and legacy rows need no rewrite.
    if not await _check_constraint_exists(cursor, _CHK_MEMBERSHIP_ORIGIN_V2):
        await cursor.execute(
            "ALTER TABLE jax_project_membership ADD CONSTRAINT " + _CHK_MEMBERSHIP_ORIGIN_V2 + " "
            "CHECK (grant_origin IN ('EXPLICIT','CREATOR','TENANT_ADMIN','OFFICE'))")
    if await _check_constraint_exists(cursor, _CHK_MEMBERSHIP_ORIGIN):
        await cursor.execute("ALTER TABLE jax_project_membership DROP CONSTRAINT " + _CHK_MEMBERSHIP_ORIGIN)
    if not await _check_constraint_exists(cursor, _CHK_MEMBERSHIP_PRE_ADMIN):
        await cursor.execute(
            "ALTER TABLE jax_project_membership ADD CONSTRAINT " + _CHK_MEMBERSHIP_PRE_ADMIN + " CHECK ("
            "(pre_admin_role IS NULL AND pre_admin_status IS NULL) OR "
            "(grant_origin='TENANT_ADMIN' AND pre_admin_role IN ('VIEWER','CONTRIBUTOR','REVIEWER','OWNER') "
            "AND pre_admin_status IN ('ACTIVE','REVOKED')))")
    # 005e
    if not await _index_exists(cursor, "jax_project_membership", _IDX_MEMBERSHIP_USER_LIST):
        await cursor.execute(
            "CREATE INDEX " + _IDX_MEMBERSHIP_USER_LIST + " "
            "ON jax_project_membership (tenant_id, user_id, status, project_id)")
    # 005f
    await cursor.execute(
        "CREATE TABLE IF NOT EXISTS jax_project_creation_request ("
        "idempotency_key CHAR(36) NOT NULL, tenant_id INT NOT NULL, user_id INT NOT NULL, "
        "request_digest CHAR(64) NOT NULL, project_id INT NOT NULL, created_at DATETIME(6) NOT NULL, "
        "PRIMARY KEY (idempotency_key), "
        "KEY idx_jax_project_creation_request_project (project_id), "
        "CONSTRAINT fk_jax_project_creation_request_scope FOREIGN KEY (project_id, tenant_id) "
        "REFERENCES jax_project_scope(project_id, tenant_id), "
        "CONSTRAINT fk_jax_project_creation_request_user FOREIGN KEY (user_id) REFERENCES jax_users(user_id)"
        ") ENGINE=InnoDB")
    await cursor.execute(
        "CREATE TABLE IF NOT EXISTS jax_namespace_project_creation_request ("
        "namespace VARCHAR(32) NOT NULL, idempotency_key CHAR(36) NOT NULL, actor_user_id INT NOT NULL, "
        "target_tenant_id INT NOT NULL, request_digest CHAR(64) NOT NULL, project_id INT NOT NULL, "
        "created_at DATETIME(6) NOT NULL, PRIMARY KEY (namespace,idempotency_key), "
        "KEY idx_jax_namespace_project_creation_request_tenant_actor_created "
        "(target_tenant_id,actor_user_id,created_at), "
        "CONSTRAINT fk_jax_namespace_project_creation_request_scope FOREIGN KEY (project_id,target_tenant_id) "
        "REFERENCES jax_project_scope(project_id,tenant_id), "
        "CONSTRAINT fk_jax_namespace_project_creation_request_actor FOREIGN KEY (actor_user_id) REFERENCES jax_users(user_id)"
        ") ENGINE=InnoDB")
    await cursor.execute(
        "CREATE TABLE IF NOT EXISTS jax_namespace_membership_mutation_request ("
        "namespace VARCHAR(32) NOT NULL, idempotency_key CHAR(36) NOT NULL, operation VARCHAR(32) NOT NULL, "
        "actor_user_id INT NOT NULL, target_tenant_id INT NOT NULL, project_id INT NOT NULL, target_user_id INT NOT NULL, "
        "member_kind VARCHAR(32) NULL, request_digest CHAR(64) NOT NULL, execution_kind VARCHAR(32) NOT NULL DEFAULT 'INTERACTIVE_USER', "
        "invitation_id CHAR(36) NULL, membership_id CHAR(36) NOT NULL, "
        "event_id CHAR(36) NOT NULL, project_role VARCHAR(16) NOT NULL, status VARCHAR(16) NOT NULL, changed TINYINT(1) NOT NULL, "
        "created_at DATETIME(6) NOT NULL, PRIMARY KEY (namespace,idempotency_key), "
        "KEY idx_jax_namespace_membership_mutation_target (target_tenant_id,project_id,target_user_id,created_at), "
        "CONSTRAINT fk_jax_namespace_membership_mutation_scope FOREIGN KEY (project_id,target_tenant_id) "
        "REFERENCES jax_project_scope(project_id,tenant_id), "
        "CONSTRAINT fk_jax_namespace_membership_mutation_actor FOREIGN KEY (actor_user_id) REFERENCES jax_users(user_id), "
        "CONSTRAINT fk_jax_namespace_membership_mutation_target_user FOREIGN KEY (target_user_id) REFERENCES jax_users(user_id), "
        "CONSTRAINT fk_jax_namespace_membership_mutation_membership FOREIGN KEY (membership_id) "
        "REFERENCES jax_project_membership(membership_id), "
        "CONSTRAINT fk_jax_namespace_membership_mutation_event FOREIGN KEY (event_id) "
        "REFERENCES jax_project_membership_event(event_id), "
        "CONSTRAINT chk_jax_namespace_membership_mutation_operation CHECK "
        "(operation IN ('GRANT_OFFICE_MEMBER','REVOKE_OFFICE_MEMBER')), "
        "CONSTRAINT chk_jax_namespace_membership_mutation_kind CHECK "
        "((operation='GRANT_OFFICE_MEMBER' AND member_kind IN ('INTERNAL','EXTERNAL_CLIENT')) OR "
        "(operation='REVOKE_OFFICE_MEMBER' AND member_kind IS NULL)), "
        "CONSTRAINT chk_jax_namespace_membership_mutation_execution CHECK "
        "((execution_kind='INTERACTIVE_USER' AND invitation_id IS NULL) OR "
        "(execution_kind='INVITATION_ACCEPTANCE' AND invitation_id IS NOT NULL)), "
        "CONSTRAINT chk_jax_namespace_membership_mutation_result_v4 CHECK ("
        "((operation='GRANT_OFFICE_MEMBER' AND member_kind IS NOT NULL AND member_kind='INTERNAL' AND project_role='CONTRIBUTOR' AND status='ACTIVE') OR "
        "(operation='GRANT_OFFICE_MEMBER' AND member_kind IS NOT NULL AND member_kind='EXTERNAL_CLIENT' AND project_role='VIEWER' AND status='ACTIVE') OR "
        "(operation='REVOKE_OFFICE_MEMBER' AND member_kind IS NULL AND project_role IN ('VIEWER','CONTRIBUTOR','REVIEWER','OWNER') AND status='REVOKED')) "
        "AND changed=1)"
        ") ENGINE=InnoDB")
    if await _column_type(cursor, "jax_namespace_membership_mutation_request", "execution_kind") is None:
        await cursor.execute(
            "ALTER TABLE jax_namespace_membership_mutation_request ADD COLUMN execution_kind "
            "VARCHAR(32) NOT NULL DEFAULT 'INTERACTIVE_USER' AFTER request_digest")
    if await _column_type(cursor, "jax_namespace_membership_mutation_request", "invitation_id") is None:
        await cursor.execute(
            "ALTER TABLE jax_namespace_membership_mutation_request ADD COLUMN invitation_id CHAR(36) NULL "
            "AFTER execution_kind")
    if not await _check_constraint_exists(cursor, _CHK_NAMESPACE_MEMBERSHIP_EXECUTION):
        await cursor.execute(
            "ALTER TABLE jax_namespace_membership_mutation_request ADD CONSTRAINT "
            + _CHK_NAMESPACE_MEMBERSHIP_EXECUTION + " CHECK "
            "((execution_kind='INTERACTIVE_USER' AND invitation_id IS NULL) OR "
            "(execution_kind='INVITATION_ACCEPTANCE' AND invitation_id IS NOT NULL))")
    # A grant's fixed kind/role/status tuple and a revoke's recorded historic
    # role are one linked invariant.  Existing impossible records indicate a
    # prior contract breach; stop before DDL rather than rewriting evidence.
    if not await _check_constraint_exists(cursor, _CHK_NAMESPACE_MEMBERSHIP_RESULT_V4):
        await cursor.execute(
            "SELECT 1 FROM jax_namespace_membership_mutation_request WHERE COALESCE(("
            "(operation='GRANT_OFFICE_MEMBER' AND member_kind IS NOT NULL AND member_kind='INTERNAL' AND project_role='CONTRIBUTOR' AND status='ACTIVE' AND changed=1) OR "
            "(operation='GRANT_OFFICE_MEMBER' AND member_kind IS NOT NULL AND member_kind='EXTERNAL_CLIENT' AND project_role='VIEWER' AND status='ACTIVE' AND changed=1) OR "
            "(operation='REVOKE_OFFICE_MEMBER' AND member_kind IS NULL AND project_role IN ('VIEWER','CONTRIBUTOR','REVIEWER','OWNER') AND status='REVOKED' AND changed=1)"
            "),FALSE)=FALSE LIMIT 1")
        if await cursor.fetchone() is not None:
            raise RuntimeError("cannot upgrade membership mutation requests: impossible operation result")
        await cursor.execute(
            "ALTER TABLE jax_namespace_membership_mutation_request ADD CONSTRAINT "
            + _CHK_NAMESPACE_MEMBERSHIP_RESULT_V4 + " CHECK ("
            "((operation='GRANT_OFFICE_MEMBER' AND member_kind IS NOT NULL AND member_kind='INTERNAL' AND project_role='CONTRIBUTOR' AND status='ACTIVE') OR "
            "(operation='GRANT_OFFICE_MEMBER' AND member_kind IS NOT NULL AND member_kind='EXTERNAL_CLIENT' AND project_role='VIEWER' AND status='ACTIVE') OR "
            "(operation='REVOKE_OFFICE_MEMBER' AND member_kind IS NULL AND project_role IN ('VIEWER','CONTRIBUTOR','REVIEWER','OWNER') AND status='REVOKED')) "
            "AND changed=1)")
    for name in (_CHK_NAMESPACE_MEMBERSHIP_RESULT, _CHK_NAMESPACE_MEMBERSHIP_RESULT_V2,
                 _CHK_NAMESPACE_MEMBERSHIP_RESULT_V3):
        if await _check_constraint_exists(cursor, name):
            await cursor.execute(
                "ALTER TABLE jax_namespace_membership_mutation_request DROP CONSTRAINT " + name)
    # One invitation capability has exactly one durable consumption.  Check
    # old data before installing the unique key: a conflicting upgrade stops
    # closed and preserves the historical rows for human repair.
    if not await _index_exists(cursor, "jax_namespace_membership_mutation_request", _UQ_NAMESPACE_MEMBERSHIP_INVITATION):
        await cursor.execute(
            "SELECT namespace,invitation_id,COUNT(*) AS n "
            "FROM jax_namespace_membership_mutation_request "
            "WHERE invitation_id IS NOT NULL GROUP BY namespace,invitation_id HAVING COUNT(*)>1")
        if await cursor.fetchone() is not None:
            raise RuntimeError("cannot upgrade membership mutation requests: duplicate invitation consumption")
        await cursor.execute(
            "CREATE UNIQUE INDEX " + _UQ_NAMESPACE_MEMBERSHIP_INVITATION + " "
            "ON jax_namespace_membership_mutation_request (namespace,invitation_id)")
    # 005g
    current_type = await _column_type(cursor, "projects", "status")
    if current_type is not None and "hidden" not in current_type:
        await cursor.execute("ALTER TABLE projects MODIFY COLUMN status " + _LIFECYCLE_STATUS_ENUM)
    # 005h (MAJOR MJ-2): guarded like every other step here, even though
    # `jax_users` is not this module's own table -- only created if missing.
    if not await _index_exists(cursor, "jax_users", _IDX_USERS_TENANT_ROLE_STATUS):
        await cursor.execute(
            "CREATE INDEX " + _IDX_USERS_TENANT_ROLE_STATUS + " ON jax_users (tenant_id, role, status)")
    # 005i (E1.1): candidate list ordered by email, see _IDX_USERS_TENANT_EMAIL.
    if not await _index_exists(cursor, "jax_users", _IDX_USERS_TENANT_EMAIL):
        await cursor.execute(
            "CREATE INDEX " + _IDX_USERS_TENANT_EMAIL + " ON jax_users (tenant_id, email)")
    # 006a (E2a): documentos del proyecto, ver _DDL_PROJECT_DOCUMENTS.
    await cursor.execute(_DDL_PROJECT_DOCUMENTS)


async def revert_project_lifecycle_migration(cursor: Any) -> None:
    """Bring 005 back down. Run by hand via
    ``scripts/b9_revertir_005.py --aplicar`` -- never automatically.

    Fails closed: refuses if any ``jax_project_scope`` row is ARCHIVED/HIDDEN
    or any ``projects`` row is hidden/disabled, since going back to the old
    two-state CHECK would otherwise silently strand that data in a state the
    old constraint rejects. Loses the TENANT_ADMIN/pre_admin_* provenance on
    every membership row -- there is no way back from that once run.
    """
    # Namespace data belongs to the Office authority contract. A legacy 005
    # rollback must never erase it or leave its request references dangling.
    # These checks run before every DDL statement below.
    if await _column_type(cursor, "jax_project_scope", "authorization_namespace") is not None:
        await cursor.execute(
            "SELECT COUNT(*) AS n FROM jax_project_scope WHERE authorization_namespace<>'CLASSIC'")
        if _scalar(await cursor.fetchone()) > 0:
            raise RuntimeError(
                "cannot revert migration 005: non-CLASSIC project scopes require the namespace contract")
    office_list_index = await _index_shape(
        cursor, "jax_project_scope", _IDX_SCOPE_NAMESPACE_STATUS_PROJECT,
        ("authorization_namespace", "status", "project_id"))
    if office_list_index == "drifted":
        raise RuntimeError(
            "cannot revert migration 005: " + _IDX_SCOPE_NAMESPACE_STATUS_PROJECT
            + " has an unexpected definition")
    await cursor.execute("SHOW TABLES LIKE 'jax_namespace_project_creation_request'")
    if await cursor.fetchone() is not None:
        await cursor.execute("SELECT COUNT(*) AS n FROM jax_namespace_project_creation_request")
        if _scalar(await cursor.fetchone()) > 0:
            raise RuntimeError(
                "cannot revert migration 005: namespace project creation requests require the namespace contract")
    await cursor.execute("SHOW TABLES LIKE 'jax_namespace_membership_mutation_request'")
    if await cursor.fetchone() is not None:
        await cursor.execute("SELECT COUNT(*) AS n FROM jax_namespace_membership_mutation_request WHERE namespace='OFFICE_CONFIDENTIAL'")
        if _scalar(await cursor.fetchone()) > 0:
            raise RuntimeError(
                "cannot revert migration 005: Office membership mutation requests require the namespace contract")
    await cursor.execute(
        "SELECT COUNT(*) AS n FROM jax_project_scope WHERE status IN ('ARCHIVED','HIDDEN')")
    if _scalar(await cursor.fetchone()) > 0:
        raise RuntimeError(
            "cannot revert migration 005: jax_project_scope has ARCHIVED/HIDDEN rows "
            "that the old ('ACTIVE','DISABLED') CHECK would reject")
    await cursor.execute("SELECT COUNT(*) AS n FROM projects WHERE status IN ('hidden','disabled')")
    if _scalar(await cursor.fetchone()) > 0:
        raise RuntimeError(
            "cannot revert migration 005: projects has hidden/disabled rows "
            "that the old status ENUM does not have")
    # 006a (E2a): la tabla mas nueva, referencia a `projects` y a `jax_users`.
    # Vacia se baja; con filas, fallo cerrado ANTES de tocar nada de 005.
    await cursor.execute("SHOW TABLES LIKE 'project_documents'")
    if await cursor.fetchone() is not None:
        await cursor.execute("SELECT COUNT(*) AS n FROM project_documents")
        if _scalar(await cursor.fetchone()) > 0:
            raise RuntimeError(
                "cannot revert migration 006a: project_documents has rows "
                "(uploaded project documents would be lost)")

    if not await _check_constraint_exists(cursor, _CHK_SCOPE_STATUS_V1):
        await cursor.execute(
            "ALTER TABLE jax_project_scope ADD CONSTRAINT " + _CHK_SCOPE_STATUS_V1 + " "
            "CHECK (status IN ('ACTIVE','DISABLED'))")
    if await _check_constraint_exists(cursor, _CHK_SCOPE_STATUS_V2):
        await cursor.execute("ALTER TABLE jax_project_scope DROP CONSTRAINT " + _CHK_SCOPE_STATUS_V2)
    await cursor.execute("DROP TABLE IF EXISTS project_documents")
    await cursor.execute("DROP TABLE IF EXISTS jax_namespace_project_creation_request")
    await cursor.execute("DROP TABLE IF EXISTS jax_namespace_membership_mutation_request")
    await cursor.execute("DROP TABLE IF EXISTS jax_project_creation_request")
    if await _index_exists(cursor, "jax_project_scope", _IDX_SCOPE_NAMESPACE_TENANT_STATUS_PROJECT):
        await cursor.execute(
            "DROP INDEX " + _IDX_SCOPE_NAMESPACE_TENANT_STATUS_PROJECT + " ON jax_project_scope")
    if office_list_index == "exact":
        await cursor.execute(
            "DROP INDEX " + _IDX_SCOPE_NAMESPACE_STATUS_PROJECT + " ON jax_project_scope")
    if await _check_constraint_exists(cursor, _CHK_SCOPE_NAMESPACE):
        await cursor.execute("ALTER TABLE jax_project_scope DROP CONSTRAINT " + _CHK_SCOPE_NAMESPACE)
    if await _column_type(cursor, "jax_project_scope", "authorization_namespace") is not None:
        await cursor.execute("ALTER TABLE jax_project_scope DROP COLUMN authorization_namespace")
    if await _index_exists(cursor, "jax_project_membership", _IDX_MEMBERSHIP_USER_LIST):
        await cursor.execute("DROP INDEX " + _IDX_MEMBERSHIP_USER_LIST + " ON jax_project_membership")
    if await _index_exists(cursor, "jax_users", _IDX_USERS_TENANT_ROLE_STATUS):
        await cursor.execute("DROP INDEX " + _IDX_USERS_TENANT_ROLE_STATUS + " ON jax_users")
    if await _index_exists(cursor, "jax_users", _IDX_USERS_TENANT_EMAIL):
        await cursor.execute("DROP INDEX " + _IDX_USERS_TENANT_EMAIL + " ON jax_users")
    # Restore the old provenance CHECK before dropping the Office-aware one.
    # The early Office-scope/request guards above ensure no OFFICE membership
    # survives this rollback.
    if not await _check_constraint_exists(cursor, _CHK_MEMBERSHIP_ORIGIN):
        await cursor.execute(
            "ALTER TABLE jax_project_membership ADD CONSTRAINT " + _CHK_MEMBERSHIP_ORIGIN + " "
            "CHECK (grant_origin IN ('EXPLICIT','CREATOR','TENANT_ADMIN'))")
    for name in (_CHK_MEMBERSHIP_ORIGIN_V2, _CHK_MEMBERSHIP_PRE_ADMIN):
        if await _check_constraint_exists(cursor, name):
            await cursor.execute("ALTER TABLE jax_project_membership DROP CONSTRAINT " + name)
    await cursor.execute(
        "ALTER TABLE jax_project_membership "
        "DROP COLUMN IF EXISTS pre_admin_status, "
        "DROP COLUMN IF EXISTS pre_admin_role, "
        "DROP COLUMN IF EXISTS grant_origin")
    await cursor.execute("ALTER TABLE projects MODIFY COLUMN status " + _LEGACY_STATUS_ENUM + " DEFAULT 'planning'")
