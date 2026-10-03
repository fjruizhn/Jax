"""Jacobs' fixed F2-E contracts for canonical pipeline status responses."""
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
    OriginBinding,
    OutputOrigin,
    StructJSONNumberBinding,
    StructJSONNumberPattern,
    StructuredNumberBindingContract,
)


_COMPONENT_ID = "jacobs.service"

JACOBS_RUNTIME_SLOT_CONTRACTS = {
    ("PIPELINE_STATUS", "/status"): RuntimeSlotContract(
        "PIPELINE_STATUS", "/status", {"pipeline_id": "/pipeline_id", "status": "/status"}),
    ("PIPELINE_STATUS", "/pipeline/status"): RuntimeSlotContract(
        "PIPELINE_STATUS", "/pipeline/status", {
            "pipeline_id": "/pipeline/pipeline_id", "status": "/pipeline/status",
        }),
}

_PIPELINE_DETAIL_NUMBER_BINDING_CONTRACT_ID = "jacobs-pipeline-detail-v1"
JACOBS_PIPELINE_DETAIL_NUMBER_BINDINGS = StructuredNumberBindingContract(
    bindings=(
        StructJSONNumberBinding("/pipeline/created_at"),
        StructJSONNumberBinding("/pipeline/updated_at"),
        StructJSONNumberBinding("/pipeline/owner_ack_at", nullable=True),
    ),
    patterns=(
        StructJSONNumberPattern("/steps/*/started_at", nullable=True),
        StructJSONNumberPattern("/steps/*/finished_at", nullable=True),
    ),
)

_PIPELINE_STATUS_CONTRACT = StructuredRouteContract(
    "jacobs-pipeline-status", "1", _COMPONENT_ID, JaxExternalOutputChannel.JACOBS_HTTP,
    (
        OriginBinding("/pipeline_id", OutputOrigin.SYSTEM, "runtime-receipt:0"),
        OriginBinding("/status", OutputOrigin.SYSTEM, "runtime-receipt:0"),
    ), {}, 200,
)

_PIPELINE_DETAIL_CONTRACT = StructuredRouteContract(
    "jacobs-pipeline-detail", "1", _COMPONENT_ID, JaxExternalOutputChannel.JACOBS_HTTP,
    (
        OriginBinding("", OutputOrigin.TOOL),
        OriginBinding("/pipeline/pipeline_id", OutputOrigin.SYSTEM, "runtime-receipt:0"),
        OriginBinding("/pipeline/status", OutputOrigin.SYSTEM, "runtime-receipt:0"),
    ), {}, 200, number_binding_contract_id=_PIPELINE_DETAIL_NUMBER_BINDING_CONTRACT_ID,
)


async def governed_pipeline_status_response(*, payload: Mapping[str, object], status_code: int = 200):
    """Prepare the exact two-field canonical pipeline-status DTO.

    Rich pipeline/detail/result DTOs have their own layouts.  They cannot be
    silently squeezed through this narrow status layout because extra fields
    are rejected by F2-C origin coverage.
    """
    if set(payload) != {"pipeline_id", "status"}:
        return GovernedJSONResponse()
    pipeline_id, status = payload.get("pipeline_id"), payload.get("status")
    if not isinstance(pipeline_id, str) or not isinstance(status, str):
        return GovernedJSONResponse()
    try:
        from jacobs import store
        pipeline = await store.pipeline_get(pipeline_id)
        tenant_id, user_id = getattr(pipeline, "tenant_id", None), getattr(pipeline, "user_id", None)
        if not isinstance(tenant_id, str) or not tenant_id or not isinstance(user_id, str) or not user_id:
            return GovernedJSONResponse()
        scope = response_scope(environment=os.getenv("JAX_GOVERNANCE_ENVIRONMENT", "production"),
            tenant_id=tenant_id, subject_id=user_id, component_id=_COMPONENT_ID)
        contract = _PIPELINE_STATUS_CONTRACT if status_code == 200 else StructuredRouteContract(
            _PIPELINE_STATUS_CONTRACT.layout_id, _PIPELINE_STATUS_CONTRACT.layout_version,
            _PIPELINE_STATUS_CONTRACT.component_id, _PIPELINE_STATUS_CONTRACT.channel,
            _PIPELINE_STATUS_CONTRACT.origins, _PIPELINE_STATUS_CONTRACT.presentation_maps, status_code)
        claim = RuntimeClaimRequest("PIPELINE_STATUS", {
            "pipeline_id": pipeline_id, "status": status,
        }, "/status")
        return await prepare_registered_http_output(contract=contract, scope=scope,
            payload=payload, claims=(claim,))
    except Exception:
        return GovernedJSONResponse()


async def governed_pipeline_detail_response(*, payload: Mapping[str, object], status_code: int = 200):
    """Render the complete canonical pipeline and step DTO through F2-E."""
    pipeline_data = payload.get("pipeline")
    steps = payload.get("steps")
    if not isinstance(pipeline_data, Mapping) or not isinstance(steps, list):
        return GovernedJSONResponse()
    pipeline_id, status = pipeline_data.get("pipeline_id"), pipeline_data.get("status")
    if not isinstance(pipeline_id, str) or not isinstance(status, str):
        return GovernedJSONResponse()
    try:
        from jacobs import store
        pipeline = await store.pipeline_get(pipeline_id)
        tenant_id, user_id = getattr(pipeline, "tenant_id", None), getattr(pipeline, "user_id", None)
        if not isinstance(tenant_id, str) or not tenant_id or not isinstance(user_id, str) or not user_id:
            return GovernedJSONResponse()
        scope = response_scope(environment=os.getenv("JAX_GOVERNANCE_ENVIRONMENT", "production"),
            tenant_id=tenant_id, subject_id=user_id, component_id=_COMPONENT_ID)
        contract = _PIPELINE_DETAIL_CONTRACT if status_code == 200 else StructuredRouteContract(
            _PIPELINE_DETAIL_CONTRACT.layout_id, _PIPELINE_DETAIL_CONTRACT.layout_version,
            _PIPELINE_DETAIL_CONTRACT.component_id, _PIPELINE_DETAIL_CONTRACT.channel,
            _PIPELINE_DETAIL_CONTRACT.origins, _PIPELINE_DETAIL_CONTRACT.presentation_maps, status_code,
            _PIPELINE_DETAIL_CONTRACT.number_bindings,
            _PIPELINE_DETAIL_CONTRACT.number_binding_contract_id,
        )
        claim = RuntimeClaimRequest("PIPELINE_STATUS", {
            "pipeline_id": pipeline_id, "status": status,
        }, "/pipeline/status")
        return await prepare_registered_http_output(contract=contract, scope=scope,
            payload=payload, claims=(claim,))
    except Exception:
        return GovernedJSONResponse()
