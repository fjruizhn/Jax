from test_authority_resolution_context import empty_view, context
from policy.authority_resolution import resolve_static_authority
from policy.authority_resolution.serialization import canonical_resolution_bytes

def test_canonical_bytes_repeat():
    view, ctx = empty_view(), context()
    assert canonical_resolution_bytes(resolve_static_authority(view, ctx)) == canonical_resolution_bytes(resolve_static_authority(view, ctx))
