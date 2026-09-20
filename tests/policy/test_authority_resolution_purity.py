from test_authority_resolution_context import empty_view, context
from policy.authority_resolution import resolve_static_authority

def test_resolver_does_not_mutate_context_or_view():
    view, ctx = empty_view(), context()
    before = (view, ctx)
    resolve_static_authority(view, ctx)
    assert (view, ctx) == before
