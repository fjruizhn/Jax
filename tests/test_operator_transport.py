import pytest
from policy.governance.governed_renderer import GovernedRenderer,RenderContext
from policy.governance.output_lifecycle import mint_governed_transport_unit,OutputLifecycleError
from policy.governance.operator_transport import *
from policy.governance.response import (
    ContentBlock,
    ContentBlockKind,
    ContractState,
    GovernedResponseCandidate,
    _seal_candidate_for_server,
)
from test_governed_renderer import NOW,scope,receipt


def base():
    response_scope = scope()
    candidate = GovernedResponseCandidate(
        "f2-c.1", "operator-wire", response_scope.request_id,
        response_scope.trace_id, response_scope, "web-chat", (),
        (ContentBlock(ContentBlockKind.TOOL_DATA, {"message": "hello"}),), (), (),
    )
    envelope = _seal_candidate_for_server(
        candidate, contract_state=ContractState.VALID, governance_receipt=receipt(),
    )
    context = RenderContext(None, now=lambda: NOW)
    rendered = GovernedRenderer().render_text(envelope, context)
    unit = mint_governed_transport_unit(
        envelope, rendered, context, transport_kind="operator", idempotency_key="x", now=NOW,
    )
    return unit, rendered


def test_wire_bytes():
    unit, rendered = base()
    wire = mint_operator_wire(
        unit, rendered,
        OperatorTransportMetadata("EXTERNAL_ALERT_V1", "telegram-json-v1", "cfg-secret", "42"),
        now=NOW,
    )
    assert wire.wire_bytes == b'{"chat_id":"42","text":"{&quot;message&quot;: &quot;hello&quot;}"}'
    projection = wire.durable_projection()
    assert "cfg-secret" not in repr(projection)
    assert '"42"' not in repr(projection)


def test_cli_durable():
    unit, rendered = base()
    assert mint_operator_wire(unit, rendered, OperatorTransportMetadata("OPERATOR_CLI_TEXT_V1", "cli-utf8-v1", "cfg"), now=NOW).wire_bytes == b'{&quot;message&quot;: &quot;hello&quot;}\n'
    assert mint_operator_wire(unit, rendered, OperatorTransportMetadata("DURABLE_JOB_RESULT_V1", "durable-text-v1", "cfg"), now=NOW).wire_bytes == b'{&quot;message&quot;: &quot;hello&quot;}'


def test_mutation_rejected():
    unit, rendered = base()
    wire = mint_operator_wire(unit, rendered, OperatorTransportMetadata("EXTERNAL_ALERT_V1", "telegram-json-v1", "cfg", "42"), now=NOW)
    def forged(**changes):
        value = object.__new__(OperatorWireUnit)
        for field in ("unit", "metadata", "wire_bytes", "wire_digest", "bound_digest"):
            object.__setattr__(value, field, changes.get(field, getattr(wire, field)))
        return value

    for changed in (
        forged(wire_bytes=b"x"),
        forged(wire_digest="sha256:x"),
        forged(bound_digest="sha256:x"),
        forged(metadata=OperatorTransportMetadata("EXTERNAL_ALERT_V1", "telegram-json-v1", "cfg", "43")),
    ):
        with pytest.raises(OutputLifecycleError):
            revalidate_operator_wire(changed, now=NOW)


def test_closed_and_private():
    with pytest.raises(OutputLifecycleError):
        OperatorTransportMetadata("OPERATOR_CLI_TEXT_V1", "telegram-json-v1", "cfg", "42")
    with pytest.raises(OutputLifecycleError):
        OperatorTransportMetadata("EXTERNAL_ALERT_V1", "telegram-json-v2", "cfg", "42")
    with pytest.raises(OutputLifecycleError):
        OperatorWireUnit()
