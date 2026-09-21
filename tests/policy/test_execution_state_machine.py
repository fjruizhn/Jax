from policy.execution_control.models import ExecutionState
from policy.execution_control.state_machine import transition
def test_ready_dispatch_transition(): assert transition(ExecutionState.READY_TO_DISPATCH, ExecutionState.DISPATCHED) is ExecutionState.DISPATCHED

def test_terminal_cannot_transition():
    import pytest
    with pytest.raises(Exception): transition(ExecutionState.SUCCEEDED, ExecutionState.RUNNING)
