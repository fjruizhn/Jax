"""Closed, append-only governed execution lifecycle."""
from .errors import InvalidExecutionStateTransitionError
from .models import ExecutionState

_TERMINAL = frozenset({ExecutionState.SUCCEEDED, ExecutionState.FAILED,
    ExecutionState.CANCELLED, ExecutionState.TIMED_OUT, ExecutionState.EXPIRED})
_LEGAL = {
    ExecutionState.WAITING_HUMAN_APPROVAL: {ExecutionState.READY_FOR_DRY_RUN, ExecutionState.READY_TO_DISPATCH, ExecutionState.CANCELLED, ExecutionState.EXPIRED},
    ExecutionState.READY_FOR_DRY_RUN: {ExecutionState.READY_TO_DISPATCH, ExecutionState.FAILED, ExecutionState.CANCELLED, ExecutionState.EXPIRED},
    ExecutionState.READY_TO_DISPATCH: {ExecutionState.DISPATCHED, ExecutionState.CANCELLED, ExecutionState.EXPIRED},
    ExecutionState.DISPATCHED: {ExecutionState.RUNNING, ExecutionState.FAILED, ExecutionState.CANCELLED, ExecutionState.TIMED_OUT},
    ExecutionState.RUNNING: {ExecutionState.SUCCEEDED, ExecutionState.FAILED, ExecutionState.CANCELLED, ExecutionState.TIMED_OUT},
}

def initial_state(*, requires_human_approval: bool, requires_dry_run: bool) -> ExecutionState:
    if requires_human_approval:
        return ExecutionState.WAITING_HUMAN_APPROVAL
    return ExecutionState.READY_FOR_DRY_RUN if requires_dry_run else ExecutionState.READY_TO_DISPATCH

def transition(current: ExecutionState, target: ExecutionState) -> ExecutionState:
    if current in _TERMINAL or target not in _LEGAL.get(current, frozenset()):
        raise InvalidExecutionStateTransitionError(f"{current.value} -> {target.value} inválida")
    return target
