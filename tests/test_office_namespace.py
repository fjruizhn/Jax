import uuid

import pytest

from jax.memory.b9 import ScopeContext
from jax.memory.b9 import AuthorizationDenied
from jax.memory.project_authority import (
    OFFICE_NAMESPACE_CONTRACT_VERSION,
    NamespacedMemberKind,
    NamespaceMembershipExecution,
    NamespaceMembershipExecutionKind,
    NamespaceMembershipOperation,
    ProjectAuthorityAdmin,
    ProjectAuthorityError,
    ProjectAuthorityRetryable,
    UpdatedNamespacedProjectMetadata,
)
from jax.memory.scope_authority import MariaDBScopeAuthorityResolver, ProjectNamespace


class Cursor:
    def __init__(self, rows):
        self.rows = list(rows)
        self.calls = []

    async def execute(self, sql, args=()):
        self.calls.append((sql, args))

    async def fetchone(self):
        return self.rows.pop(0) if self.rows else None


class Store:
    def __init__(self, rows):
        self.cursor = Cursor(rows)
        self.mutations = 0

    async def mutation(self, operation):
        self.mutations += 1
        return await operation(self.cursor)

    def acquire(self):
        return _Acquire(self)


class FreshTransactionStore:
    """One cursor per mutation, so a retry cannot reuse transaction state."""
    def __init__(self, row_sets):
        self.cursors = [Cursor(rows) for rows in row_sets]
        self.mutations = 0

    async def mutation(self, operation):
        cursor = self.cursors[self.mutations]
        self.mutations += 1
        return await operation(cursor)


class _Acquire:
    def __init__(self, store): self.store = store
    async def __aenter__(self): return self
    async def __aexit__(self, *_): pass
    def cursor(self): return _CursorContext(self.store.cursor)


class _CursorContext:
    def __init__(self, cursor): self.cursor = cursor
    async def __aenter__(self): return self.cursor
    async def __aexit__(self, *_): pass


class Authorizer:
    def __init__(self):
        self.calls = []

    async def authorize_in_transaction(self, cursor, *, actor_user_id, target_tenant_id,
                                       namespace, operation):
        self.calls.append((cursor, actor_user_id, target_tenant_id, namespace, operation))


class _DuplicateKeyError(Exception):
    pass


_DuplicateKeyError.__module__ = "pymysql.err"


class _CheckReadError(Exception):
    pass


_CheckReadError.__module__ = "pymysql.err"


class _Application1020Error(Exception):
    pass


class DuplicateStore:
    def __init__(self): self.mutations = 0
    async def mutation(self, _operation):
        self.mutations += 1
        raise _DuplicateKeyError(1062, "unrelated unique constraint")


class CheckReadStore:
    def __init__(self, rows, *, failures=1):
        self.cursor = Cursor(rows)
        self.failures = failures
        self.mutations = 0

    async def mutation(self, operation):
        self.mutations += 1
        if self.mutations <= self.failures:
            raise _CheckReadError(1020, "driver record changed")
        return await operation(self.cursor)


class Application1020Store:
    def __init__(self):
        self.mutations = 0

    async def mutation(self, _operation):
        self.mutations += 1
        raise _Application1020Error(1020, "application record changed")


def test_office_namespace_contract_marker_identifies_the_reviewed_resolver_contract():
    """Catches JXP enabling Office features against a different resolver contract."""
    assert OFFICE_NAMESPACE_CONTRACT_VERSION == "office-namespace.2"


class MembershipAuthorizer:
    def __init__(self):
        self.calls = []

    async def authorize_membership_in_transaction(self, cursor, **kwargs):
        self.calls.append((cursor, kwargs))


@pytest.mark.asyncio
async def test_office_metadata_update_locks_and_reauthorizes_in_one_transaction():
    cursor = Cursor([
        {"tenant_id": 4},
        {"tenant_id": 4, "status": "ACTIVE"},
        {"tenant_id": 4, "status": "ACTIVE"},
        {"tenant_id": 4, "status": "ACTIVE", "authorization_namespace": "OFFICE_CONFIDENTIAL"},
        {"name": "before", "description": None},
    ])
    store = Store([])
    store.cursor = cursor
    authorizer = Authorizer()

    result = await ProjectAuthorityAdmin(store).update_namespaced_project_metadata(
        actor_user_id=7, target_tenant_id=4, project_id=17,
        namespace=ProjectNamespace.OFFICE_CONFIDENTIAL,
        name="  Cafe\u0301  ", description="descripcio\u0301n", authorizer=authorizer)

    assert result == UpdatedNamespacedProjectMetadata(
        17, 4, ProjectNamespace.OFFICE_CONFIDENTIAL, "Café", "descripción", True)
    assert store.mutations == 1
    assert authorizer.calls == [(cursor, 7, 4, ProjectNamespace.OFFICE_CONFIDENTIAL,
                                 "UPDATE_OFFICE_PROJECT_METADATA")]
    statements = [sql for sql, _ in cursor.calls]
    assert statements.index("SELECT tenant_id,status FROM jax_tenants WHERE tenant_id=%s FOR UPDATE") < \
        statements.index("SELECT tenant_id,status FROM jax_users WHERE user_id=%s FOR UPDATE") < \
        statements.index("SELECT tenant_id,status,authorization_namespace FROM jax_project_scope WHERE project_id=%s FOR UPDATE") < \
        statements.index("SELECT name,description FROM projects WHERE id=%s FOR UPDATE")
    assert any("UPDATE projects SET name=%s,description=%s" in sql for sql in statements)
    assert any("UPDATE_OFFICE_PROJECT_METADATA" in args for sql, args in cursor.calls
               if "jax_project_membership_event" in sql)


@pytest.mark.asyncio
async def test_office_metadata_update_noop_is_idempotent_without_event():
    cursor = Cursor([
        {"tenant_id": 4},
        {"tenant_id": 4, "status": "ACTIVE"},
        {"tenant_id": 4, "status": "ACTIVE"},
        {"tenant_id": 4, "status": "ACTIVE", "authorization_namespace": "OFFICE_CONFIDENTIAL"},
        {"name": "Office", "description": "private"},
    ])
    store = Store([]); store.cursor = cursor
    authorizer = Authorizer()
    result = await ProjectAuthorityAdmin(store).update_namespaced_project_metadata(
        actor_user_id=7, target_tenant_id=4, project_id=17,
        namespace=ProjectNamespace.OFFICE_CONFIDENTIAL,
        name="Office", description="private", authorizer=authorizer)
    assert result.changed is False
    assert len(authorizer.calls) == 1
    assert not any("UPDATE projects" in sql or "jax_project_membership_event" in sql
                   for sql, _ in cursor.calls)


@pytest.mark.asyncio
async def test_office_metadata_update_denies_inactive_tenant_before_authorizer_or_project():
    cursor = Cursor([{"tenant_id": 4}, {"tenant_id": 4, "status": "DISABLED"}])
    store = Store([]); store.cursor = cursor
    authorizer = Authorizer()
    with pytest.raises(ProjectAuthorityError, match="tenant is not active"):
        await ProjectAuthorityAdmin(store).update_namespaced_project_metadata(
            actor_user_id=7, target_tenant_id=4, project_id=17,
            namespace=ProjectNamespace.OFFICE_CONFIDENTIAL,
            name="Office", description=None, authorizer=authorizer)
    assert not authorizer.calls
    assert not any("FROM projects" in sql for sql, _ in cursor.calls)


@pytest.mark.asyncio
async def test_office_metadata_update_rejects_classic_namespace_before_transaction():
    store = Store([])
    with pytest.raises(AuthorizationDenied, match="OFFICE_CONFIDENTIAL"):
        await ProjectAuthorityAdmin(store).update_namespaced_project_metadata(
            actor_user_id=7, target_tenant_id=4, project_id=17,
            namespace=ProjectNamespace.CLASSIC,
            name="Office", description=None, authorizer=Authorizer())
    assert store.mutations == 0


@pytest.mark.asyncio
async def test_office_metadata_update_retries_checkread_in_a_fresh_transaction_without_event_duplication():
    class FailingUpdateCursor(Cursor):
        def __init__(self, rows):
            super().__init__(rows)
            self.failed = False

        async def execute(self, sql, args=()):
            if "UPDATE projects SET name=%s,description=%s" in sql and not self.failed:
                self.failed = True
                raise _CheckReadError(1020, "record changed")
            await super().execute(sql, args)

    rows = [
        {"tenant_id": 4},
        {"tenant_id": 4, "status": "ACTIVE"},
        {"tenant_id": 4, "status": "ACTIVE"},
        {"tenant_id": 4, "status": "ACTIVE", "authorization_namespace": "OFFICE_CONFIDENTIAL"},
        {"name": "before", "description": None},
    ]
    first, second = FailingUpdateCursor(rows), Cursor(rows)

    class RetryStore:
        def __init__(self):
            self.cursors = [first, second]
            self.mutations = 0

        async def mutation(self, operation):
            cursor = self.cursors[self.mutations]
            self.mutations += 1
            return await operation(cursor)

    authorizer = Authorizer()
    result = await ProjectAuthorityAdmin(RetryStore()).update_namespaced_project_metadata(
        actor_user_id=7, target_tenant_id=4, project_id=17,
        namespace=ProjectNamespace.OFFICE_CONFIDENTIAL,
        name="after", description=None, authorizer=authorizer)
    assert result.changed
    assert len(authorizer.calls) == 2
    assert not any("jax_project_membership_event" in sql for sql, _ in first.calls)
    assert sum("jax_project_membership_event" in sql for sql, _ in second.calls) == 1


@pytest.mark.asyncio
async def test_office_metadata_update_revoke_has_no_effect_or_event():
    class Revoked(Authorizer):
        async def authorize_in_transaction(self, *args, **kwargs):
            await super().authorize_in_transaction(*args, **kwargs)
            raise PermissionError("revoked")

    cursor = Cursor([
        {"tenant_id": 4},
        {"tenant_id": 4, "status": "ACTIVE"},
        {"tenant_id": 4, "status": "ACTIVE"},
        {"tenant_id": 4, "status": "ACTIVE", "authorization_namespace": "OFFICE_CONFIDENTIAL"},
    ])
    store = Store([]); store.cursor = cursor
    authorizer = Revoked()
    with pytest.raises(PermissionError, match="revoked"):
        await ProjectAuthorityAdmin(store).update_namespaced_project_metadata(
            actor_user_id=7, target_tenant_id=4, project_id=17,
            namespace=ProjectNamespace.OFFICE_CONFIDENTIAL,
            name="Office", description=None, authorizer=authorizer)
    assert len(authorizer.calls) == 1
    assert not any("FROM projects" in sql or "UPDATE projects" in sql
                   or "jax_project_membership_event" in sql for sql, _ in cursor.calls)


@pytest.mark.asyncio
async def test_office_membership_uses_the_callers_cursor_and_records_a_typed_grant():
    """The adapter must be same-transaction: it cannot call store.mutation,
    nor manufacture a classic role from caller input."""
    cursor = Cursor([
        {"tenant_id": 4, "status": "ACTIVE"},
        {"tenant_id": 8, "status": "ACTIVE", "account_class": "OFFICE_INTERNAL"},
        {"tenant_id": 4, "status": "ACTIVE", "account_class": "EXTERNAL_CLIENT"},
        {"tenant_id": 4, "status": "ACTIVE", "authorization_namespace": "OFFICE_CONFIDENTIAL"},
        None, None,
    ])
    authorizer = MembershipAuthorizer()
    result = await ProjectAuthorityAdmin(object()).grant_namespaced_member_in_transaction(
        cursor, actor_user_id=8, target_tenant_id=4, project_id=17, target_user_id=9,
        namespace=ProjectNamespace.OFFICE_CONFIDENTIAL,
        member_kind=NamespacedMemberKind.EXTERNAL_CLIENT, idempotency_key=str(uuid.uuid4()),
        authorizer=authorizer, request_id="r", trace_id="t")

    assert result.changed is True and result.replayed is False
    assert result.project_role.value == "VIEWER" and result.status == "ACTIVE"
    assert authorizer.calls[0][0] is cursor
    assert authorizer.calls[0][1]["execution"] == NamespaceMembershipExecution(
        NamespaceMembershipExecutionKind.INTERACTIVE_USER)
    assert authorizer.calls[0][1]["idempotency_key"]
    assert any("grant_origin='OFFICE'" in sql or "'OFFICE'" in sql for sql, _ in cursor.calls)
    assert not any("BEGIN" in sql.upper() or "COMMIT" in sql.upper() or "ROLLBACK" in sql.upper()
                   for sql, _ in cursor.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("execution", [
    False,
    0,
    "",
    [],
    NamespaceMembershipExecution(NamespaceMembershipExecutionKind.INTERACTIVE_USER, str(uuid.uuid4())),
    NamespaceMembershipExecution(NamespaceMembershipExecutionKind.INVITATION_ACCEPTANCE),
    NamespaceMembershipExecution(NamespaceMembershipExecutionKind.INVITATION_ACCEPTANCE, "not-a-uuid"),
])
async def test_office_membership_rejects_invalid_execution_before_sql(execution):
    with pytest.raises(AuthorizationDenied):
        await ProjectAuthorityAdmin(object()).grant_namespaced_member_in_transaction(
            Cursor([]), actor_user_id=8, target_tenant_id=4, project_id=17, target_user_id=9,
            namespace=ProjectNamespace.OFFICE_CONFIDENTIAL,
            member_kind=NamespacedMemberKind.EXTERNAL_CLIENT, idempotency_key=str(uuid.uuid4()),
            authorizer=MembershipAuthorizer(), execution=execution)


@pytest.mark.asyncio
async def test_capability_membership_binds_canonical_key_and_truthful_created_by():
    invitation_id = str(uuid.uuid4()).upper()
    key = str(uuid.uuid4()).upper()
    cursor = Cursor([
        {"tenant_id": 4, "status": "ACTIVE"},
        {"tenant_id": 8, "status": "ACTIVE", "account_class": "OFFICE_INTERNAL"},
        {"tenant_id": 4, "status": "ACTIVE", "account_class": "EXTERNAL_CLIENT"},
        {"tenant_id": 4, "status": "ACTIVE", "authorization_namespace": "OFFICE_CONFIDENTIAL"},
        None, None, None,
    ])
    authorizer = MembershipAuthorizer()
    await ProjectAuthorityAdmin(object()).grant_namespaced_member_in_transaction(
        cursor, actor_user_id=8, target_tenant_id=4, project_id=17, target_user_id=9,
        namespace=ProjectNamespace.OFFICE_CONFIDENTIAL,
        member_kind=NamespacedMemberKind.EXTERNAL_CLIENT, idempotency_key=key,
        authorizer=authorizer,
        execution=NamespaceMembershipExecution(NamespaceMembershipExecutionKind.INVITATION_ACCEPTANCE, invitation_id))
    assert authorizer.calls[0][1]["idempotency_key"] == key.lower()
    insert_args = next(args for sql, args in cursor.calls if "INSERT INTO jax_project_membership" in sql)
    assert insert_args[-1] == f"office-invitation:{invitation_id.lower()}"


def _new_store():
    # actor hint, tenant lock, canonical actor lock, idempotency lookup, LAST_INSERT_ID.
    return Store([{"tenant_id": 4}, {"tenant_id": 4}, {"tenant_id": 4}, None, {"id": 17}])


@pytest.mark.asyncio
async def test_office_creation_authorizes_and_inserts_confidential_scope_in_one_transaction():
    """Catches a creation path that authorizes outside the write transaction
    or creates its scope as CLASSIC."""
    store = _new_store()
    authorizer = Authorizer()

    created = await ProjectAuthorityAdmin(store).create_namespaced_project(
        actor_user_id=7, target_tenant_id=4, namespace=ProjectNamespace.OFFICE_CONFIDENTIAL,
        name=" Office ", description="private", idempotency_key=str(uuid.uuid4()), authorizer=authorizer)

    assert created.project_id == 17 and created.created is True
    assert store.mutations == 1
    assert authorizer.calls == [(store.cursor, 7, 4, ProjectNamespace.OFFICE_CONFIDENTIAL,
                                 "CREATE_OFFICE_PROJECT")]
    scope = next((args for sql, args in store.cursor.calls if "INSERT INTO jax_project_scope" in sql), None)
    assert scope is not None and scope[1] == 4 and scope[2] == "OFFICE_CONFIDENTIAL"
    assert not any("jax_project_membership" in sql for sql, _ in store.cursor.calls)
    assert [args for sql, args in store.cursor.calls if "FROM jax_tenants" in sql] == [(4,)]


@pytest.mark.asyncio
async def test_office_replay_reauthorizes_and_rejects_actor_or_payload_change_without_project_leak():
    """Catches replay that returns a project before current authority or
    accepts a key bound to a different actor/payload."""
    key = str(uuid.uuid4())
    authorizer = Authorizer()
    matching = Store([
        {"tenant_id": 4}, {"tenant_id": 4}, {"tenant_id": 4},
        {"actor_user_id": 7, "target_tenant_id": 4, "request_digest": "wrong", "project_id": 17},
    ])
    with pytest.raises(Exception):
        await ProjectAuthorityAdmin(matching).create_namespaced_project(
            actor_user_id=7, target_tenant_id=4, namespace=ProjectNamespace.OFFICE_CONFIDENTIAL,
            name="Office", description=None, idempotency_key=key, authorizer=authorizer)
    assert authorizer.calls[0][1:3] == (7, 4)
    assert not any("FROM projects" in sql for sql, _ in matching.cursor.calls)


@pytest.mark.asyncio
async def test_classic_scope_resolver_denies_office_project_before_membership_roles_are_read():
    """Catches classic resolution that lets an OFFICE row inherit an OWNER
    membership or tenant administration."""
    store = Store([
        {"tenant_id": 4, "status": "ACTIVE", "role": "admin"},
        {"tenant_id": 4, "status": "ACTIVE", "authorization_namespace": "OFFICE_CONFIDENTIAL"},
    ])
    requested = ScopeContext("user:7", "USER", "7", "4", "17")
    with pytest.raises(Exception, match="namespace"):
        await MariaDBScopeAuthorityResolver(store).resolve_project_scope(requested)
    assert not any("jax_project_membership" in sql for sql, _ in store.cursor.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("namespace", [None, "FUTURE_NAMESPACE"])
async def test_classic_scope_resolver_fails_closed_for_missing_or_unknown_namespace(namespace):
    """Catches treating incomplete or future scope rows as classic."""
    project = {"tenant_id": 4, "status": "ACTIVE"}
    if namespace is not None:
        project["authorization_namespace"] = namespace
    store = Store([
        {"tenant_id": 4, "status": "ACTIVE", "role": "admin"}, project,
    ])
    with pytest.raises(Exception, match="namespace"):
        await MariaDBScopeAuthorityResolver(store).resolve_project_scope(
            ScopeContext("user:7", "USER", "7", "4", "17"))
    assert not any("jax_project_membership" in sql for sql, _ in store.cursor.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [
    {"actor_user_id": True}, {"target_tenant_id": True}, {"name": 7}, {"description": 7},
])
async def test_namespaced_creation_rejects_malformed_public_values_before_sql(kwargs):
    """Catches bool-as-id and non-text inputs escaping into DB/Unicode APIs."""
    values = dict(actor_user_id=7, target_tenant_id=4,
                  namespace=ProjectNamespace.OFFICE_CONFIDENTIAL, name="Office", description=None,
                  idempotency_key=str(uuid.uuid4()), authorizer=Authorizer())
    values.update(kwargs)
    store = Store([])
    with pytest.raises(AuthorizationDenied):
        await ProjectAuthorityAdmin(store).create_namespaced_project(**values)
    assert store.mutations == 0


@pytest.mark.asyncio
async def test_namespaced_creation_bounds_a_stable_duplicate_constraint():
    """Catches infinite retry recursion on a stable duplicate constraint."""
    store = DuplicateStore()
    with pytest.raises(ProjectAuthorityError, match="duplicate conflict") as error:
        await ProjectAuthorityAdmin(store).create_namespaced_project(
            actor_user_id=7, target_tenant_id=4, namespace=ProjectNamespace.OFFICE_CONFIDENTIAL,
            name="Office", description=None, idempotency_key=str(uuid.uuid4()), authorizer=Authorizer())
    assert error.value.code == "namespace_creation_conflict"
    assert store.mutations == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(("actor_tenant", "target_tenant", "expected_locks"), [
    (4, 9, [(4,), (9,)]),
    (9, 4, [(4,), (9,)]),
    (4, 4, [(4,)]),
])
async def test_namespaced_creation_locks_actor_and_target_tenants_in_canonical_order(
    actor_tenant, target_tenant, expected_locks,
):
    store = Store([
        {"tenant_id": actor_tenant}, *[{"tenant_id": tenant} for tenant, in expected_locks],
        {"tenant_id": actor_tenant},
        None, {"id": 17},
    ])
    await ProjectAuthorityAdmin(store).create_namespaced_project(
        actor_user_id=7, target_tenant_id=target_tenant, namespace=ProjectNamespace.OFFICE_CONFIDENTIAL,
        name="Office", description=None, idempotency_key=str(uuid.uuid4()), authorizer=Authorizer())
    tenant_locks = [args for sql, args in store.cursor.calls if "FROM jax_tenants" in sql]
    assert tenant_locks == expected_locks
    actor_selects = [sql for sql, _ in store.cursor.calls if "FROM jax_users WHERE user_id=%s" in sql]
    assert actor_selects == [
        "SELECT tenant_id FROM jax_users WHERE user_id=%s",
        "SELECT tenant_id FROM jax_users WHERE user_id=%s FOR UPDATE",
    ]


@pytest.mark.asyncio
async def test_namespaced_creation_missing_actor_denies_before_tenant_lock_or_authorizer():
    store = Store([None])
    authorizer = Authorizer()
    with pytest.raises(Exception, match="actor does not exist"):
        await ProjectAuthorityAdmin(store).create_namespaced_project(
            actor_user_id=7, target_tenant_id=4, namespace=ProjectNamespace.OFFICE_CONFIDENTIAL,
            name="Office", description=None, idempotency_key=str(uuid.uuid4()), authorizer=authorizer)
    assert not authorizer.calls
    assert not any("jax_tenants" in sql for sql, _ in store.cursor.calls)


@pytest.mark.asyncio
async def test_namespaced_creation_actor_tenant_drift_retries_in_a_fresh_transaction_before_authorizing():
    store = FreshTransactionStore([
        [{"tenant_id": 4}, {"tenant_id": 4}, {"tenant_id": 9}, {"tenant_id": 9}],
        [{"tenant_id": 9}, {"tenant_id": 4}, {"tenant_id": 9}, {"tenant_id": 9}, None, {"id": 17}],
    ])
    authorizer = Authorizer()
    created = await ProjectAuthorityAdmin(store).create_namespaced_project(
        actor_user_id=7, target_tenant_id=4, namespace=ProjectNamespace.OFFICE_CONFIDENTIAL,
        name="Office", description=None, idempotency_key=str(uuid.uuid4()), authorizer=authorizer)
    assert created.created and store.mutations == 2
    assert len(authorizer.calls) == 1 and authorizer.calls[0][0] is store.cursors[1]
    first_locks = [args for sql, args in store.cursors[0].calls if "FROM jax_tenants" in sql]
    second_locks = [args for sql, args in store.cursors[1].calls if "FROM jax_tenants" in sql]
    assert first_locks == [(4,)] and second_locks == [(4,), (9,)]


@pytest.mark.asyncio
async def test_namespaced_creation_actor_tenant_drift_exhaustion_is_stable_retryable():
    row_sets = [[{"tenant_id": 4}, {"tenant_id": 4}, {"tenant_id": 9}]] * 5
    store = FreshTransactionStore(row_sets)
    with pytest.raises(ProjectAuthorityRetryable) as failure:
        await ProjectAuthorityAdmin(store).create_namespaced_project(
            actor_user_id=7, target_tenant_id=4, namespace=ProjectNamespace.OFFICE_CONFIDENTIAL,
            name="Office", description=None, idempotency_key=str(uuid.uuid4()), authorizer=Authorizer())
    assert failure.value.code == "reintentar"
    assert str(failure.value) == "namespace creation actor tenant changed; retry"
    assert store.mutations == 5


@pytest.mark.asyncio
async def test_namespaced_creation_checkread_retries_with_a_fresh_transaction():
    store = CheckReadStore([{"tenant_id": 4}, {"tenant_id": 4}, {"tenant_id": 4}, None, {"id": 17}])
    authorizer = Authorizer()
    result = await ProjectAuthorityAdmin(store).create_namespaced_project(
        actor_user_id=7, target_tenant_id=4, namespace=ProjectNamespace.OFFICE_CONFIDENTIAL,
        name="Office", description=None, idempotency_key=str(uuid.uuid4()), authorizer=authorizer)
    assert result.created and store.mutations == 2
    assert len(authorizer.calls) == 1 and authorizer.calls[0][0] is store.cursor


@pytest.mark.asyncio
async def test_namespaced_creation_checkread_exhaustion_is_static_retryable_without_driver_text():
    store = CheckReadStore([], failures=5)
    with pytest.raises(ProjectAuthorityRetryable) as failure:
        await ProjectAuthorityAdmin(store).create_namespaced_project(
            actor_user_id=7, target_tenant_id=4, namespace=ProjectNamespace.OFFICE_CONFIDENTIAL,
            name="Office", description=None, idempotency_key=str(uuid.uuid4()), authorizer=Authorizer())
    assert store.mutations == 5 and failure.value.code == "reintentar"
    assert str(failure.value) == "namespace project creation contention, retry"
    assert "driver record changed" not in str(failure.value)


@pytest.mark.asyncio
async def test_namespaced_creation_does_not_retry_an_application_exception_with_1020_args():
    store = Application1020Store()
    with pytest.raises(_Application1020Error, match="application record changed"):
        await ProjectAuthorityAdmin(store).create_namespaced_project(
            actor_user_id=7, target_tenant_id=4, namespace=ProjectNamespace.OFFICE_CONFIDENTIAL,
            name="Office", description=None, idempotency_key=str(uuid.uuid4()), authorizer=Authorizer())
    assert store.mutations == 1

