import uuid

import pytest

from jax.memory.b9 import ScopeContext
from jax.memory.b9 import AuthorizationDenied
from jax.memory.project_authority import ProjectAuthorityAdmin, ProjectAuthorityError
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


class DuplicateStore:
    def __init__(self): self.mutations = 0
    async def mutation(self, _operation):
        self.mutations += 1
        raise _DuplicateKeyError(1062, "unrelated unique constraint")


def _new_store():
    # tenant lock, idempotency lookup, LAST_INSERT_ID.
    return Store([{"tenant_id": 4}, None, {"id": 17}])


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


@pytest.mark.asyncio
async def test_office_replay_reauthorizes_and_rejects_actor_or_payload_change_without_project_leak():
    """Catches replay that returns a project before current authority or
    accepts a key bound to a different actor/payload."""
    key = str(uuid.uuid4())
    authorizer = Authorizer()
    matching = Store([{"tenant_id": 4}, {"actor_user_id": 7, "target_tenant_id": 4, "request_digest": "wrong", "project_id": 17}])
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

