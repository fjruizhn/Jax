from pathlib import Path
from policy.authority_resolution import load_validated_candidate, to_static_policy_view, resolve_static_authority
from test_authority_resolution_context import context

def test_empty_view_has_no_precedence_winner():
    view = to_static_policy_view(load_validated_candidate(Path(__file__).resolve().parents[2]))
    result = resolve_static_authority(view, context())
    assert result.winning_precedence_layer is None
