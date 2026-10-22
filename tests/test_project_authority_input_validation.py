import pytest

from jax.memory.b9 import AuthorizationDenied, MutationAuthorizationRequest, ScopeContext, Visibility
from jax.memory.project_authority import ProjectAuthorityAdmin
from jax.memory.scope_authority import ProjectRole


class _StoreThatMustNotMutate:
    def __init__(self):
        self.calls = 0

    async def mutation(self, _operation):
        self.calls += 1
        raise AssertionError("REVIEWER role must be rejected before opening a transaction")


@pytest.mark.asyncio
async def test_change_project_role_rejects_reviewer_before_transaction_or_sql():
    store = _StoreThatMustNotMutate()
    request = MutationAuthorizationRequest(
        ScopeContext("user:7", "USER", "7", "1", "9"),
        "CHANGE_PROJECT_ROLE", Visibility.PROJECT_SHARED)

    with pytest.raises(AuthorizationDenied, match="REVIEWER"):
        await ProjectAuthorityAdmin(store).change_project_role(request, 9, 8, ProjectRole.REVIEWER)

    assert store.calls == 0
