from test_authority_resolution_context import context

def test_context_has_one_subject_and_one_action():
    assert context().subject == "ALICE"
    assert context().action == "READ"
