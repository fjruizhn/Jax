from test_authority_resolution_context import empty_view, context
from policy.authority_resolution import resolve_static_authority

def test_empty_result_contract():
    result = resolve_static_authority(empty_view(), context())
    assert result.kind == "JAX_STATIC_AUTHORITY_RESOLUTION"
    assert result.resolver_identity == "JAX-AUTHORITY-RESOLVER/1"
    assert result.controlling_rule_id is None
