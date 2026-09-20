from test_authority_resolution_context import empty_view, context
from policy.authority_resolution import resolve_static_authority

def test_candidate_result_never_grants_execution():
    result = resolve_static_authority(empty_view(), context())
    assert all(value not in {"ACTIVE", "RATIFIED", "AUTHORIZED"} for value in result.__dict__.values())
