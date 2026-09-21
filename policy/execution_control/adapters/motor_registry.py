"""Narrow adapter: a governed record must be dispatched before a Motor job."""
from __future__ import annotations
from ..errors import GovernedExecutionRequiredError

def governed_motor_payload(record, authorization, *, trace_id: str) -> dict:
    if record.execution_id is None or record.decision_id != authorization.decision_id:
        raise GovernedExecutionRequiredError("binding de ejecución requerido")
    request = authorization.execution_request
    return {"execution_id":record.execution_id,"decision_id":record.decision_id,
      "execution_request_hash":record.execution_request_hash,"authorization_id":authorization.authorization_id,
      "execution_authorization_hash":record.execution_authorization_hash,"caller":request.authenticated_caller_id,
      "capability":request.capability,"motor":request.motor,"prompt":request.prompt,"context":request.context,
      "timeout_seconds":request.timeout_seconds,"sandbox":True,"trace_id":trace_id,
      "tenant_id":request.tenant_id,"user_id":request.user_id}
