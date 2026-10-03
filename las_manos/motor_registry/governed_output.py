"""Motor Registry's fixed F2-E HTTP DTO contracts.

The route passes a canonical JobStore only to recover immutable owner scope;
it does not select a resolver or receipt.  Legacy/ownerless jobs therefore
fall back to the server-owned unavailable response.
"""
from __future__ import annotations

import os
from typing import Mapping

from jax.external_output import (
    GovernedJSONResponse,
    JaxExternalOutputChannel,
    StructuredRouteContract,
    prepare_registered_http_output,
    response_scope,
)
from policy.governance.runtime_output_composition import RuntimeClaimRequest, RuntimeSlotContract
from policy.governance.structured_output import (
    OriginBinding, OutputOrigin, StructJSONNumberBinding,
)


_COMPONENT_ID = "las-manos.motor-registry"
_MOTOR_JOB_NUMBER_BINDING_CONTRACT_ID = "motor-job-view-v1"

# Imported by LAS MANOS composition; routes cannot provide or replace these.
MOTOR_RUNTIME_SLOT_CONTRACTS = {
    ("JOB_STATUS", "/status"): RuntimeSlotContract(
        "JOB_STATUS", "/status", {"job_id": "/job_id", "status": "/status"}),
}
MOTOR_JOB_NUMBER_BINDINGS = (
    StructJSONNumberBinding("/created_at"),
    StructJSONNumberBinding("/started_at", nullable=True),
    StructJSONNumberBinding("/finished_at", nullable=True),
    StructJSONNumberBinding("/status_updated_at", nullable=True),
)

_DISPATCH_CONTRACT = StructuredRouteContract(
    "motor-dispatch-response", "1", _COMPONENT_ID, JaxExternalOutputChannel.LAS_MANOS_HTTP,
    (
        OriginBinding("/job_id", OutputOrigin.SYSTEM, "runtime-receipt:0"),
        OriginBinding("/status", OutputOrigin.SYSTEM, "runtime-receipt:0"),
        OriginBinding("/motor", OutputOrigin.TOOL),
        OriginBinding("/capability", OutputOrigin.TOOL),
        OriginBinding("/trace_id", OutputOrigin.TOOL),
        OriginBinding("/rejected_reason", OutputOrigin.TOOL),
    ), {}, 202,
)

_JOB_CONTRACT = StructuredRouteContract(
    "motor-job-view", "1", _COMPONENT_ID, JaxExternalOutputChannel.LAS_MANOS_HTTP,
    tuple(
        OriginBinding(pointer, OutputOrigin.SYSTEM, "runtime-receipt:0")
        if pointer in {"/job_id", "/status"} else OriginBinding(pointer, OutputOrigin.TOOL)
        for pointer in (
            "/job_id", "/status", "/motor", "/capability", "/caller", "/trace_id",
            "/created_at", "/started_at", "/finished_at", "/status_updated_at", "/error",
            "/result_summary", "/result_path", "/model", "/pipeline_id", "/tenant_id",
            "/user_id", "/project_id",
        )
    ), {}, 200, MOTOR_JOB_NUMBER_BINDINGS, _MOTOR_JOB_NUMBER_BINDING_CONTRACT_ID,
)


async def governed_motor_job_response(*, store, value: object, status_code: int = 200):
    """Return an exact governed HTTP response for a Motor job DTO."""
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    if not isinstance(payload, Mapping) or not isinstance(payload.get("job_id"), str) or not isinstance(payload.get("status"), str):
        return GovernedJSONResponse()
    try:
        snapshot = store.authoritative_snapshot(payload["job_id"])
        canonical = snapshot.view if snapshot is not None else None
        tenant_id = getattr(canonical, "tenant_id", None)
        user_id = getattr(canonical, "user_id", None)
        if not isinstance(tenant_id, str) or not tenant_id or not isinstance(user_id, str) or not user_id:
            raise ValueError("ownerless job")
        scope = response_scope(
            environment=os.getenv("JAX_GOVERNANCE_ENVIRONMENT", "production"),
            tenant_id=tenant_id, subject_id=user_id, component_id=_COMPONENT_ID,
        )
        contract = _DISPATCH_CONTRACT if set(payload) == {
            "job_id", "status", "motor", "capability", "trace_id", "rejected_reason"
        } else _JOB_CONTRACT
        if status_code != contract.status_code:
            contract = StructuredRouteContract(contract.layout_id, contract.layout_version,
                contract.component_id, contract.channel, contract.origins,
                contract.presentation_maps, status_code, contract.number_bindings,
                contract.number_binding_contract_id)
        claim = RuntimeClaimRequest("JOB_STATUS", {
            "job_id": payload["job_id"], "status": payload["status"],
        }, "/status")
        return await prepare_registered_http_output(contract=contract, scope=scope,
            payload=payload, claims=(claim,))
    except Exception:
        # No dynamic candidate field crosses the safe failure path.
        return GovernedJSONResponse()
