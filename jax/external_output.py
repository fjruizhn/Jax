"""JAX route composition for immutable governed structured output.

This module has no HTTP route registrations and no authority resolver choices.
Each route supplies one server-owned contract from this module, while the
shared F2-E runtime composition resolves the only accredited status claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import uuid
from typing import Mapping

from starlette.responses import Response

from policy.governance.external_output import (
    CHANNELS,
    ExternalOutputChannelId,
    GovernedExternalOutputAdapter,
    GovernedExternalOutputSubmission,
)
from policy.governance.output_lifecycle import (
    StructuredGovernedTransportUnit,
    StructuredTransportMetadata,
    revalidate_structured_for_transport,
)
from policy.governance.response import ContractState, GovernanceContractError, ResponseScope
from policy.governance.runtime_output_composition import (
    RuntimeClaimRequest,
    RuntimeOutputComposition,
)
from policy.governance.structured_output import (
    OriginBinding,
    OutputOrigin,
    StructJSONNumberBinding,
    StructuredLayout,
    StructuredLayoutRegistry,
)


class JaxExternalOutputError(GovernanceContractError):
    """A JAX route cannot produce a governed external output."""


class JaxExternalOutputChannel(str, Enum):
    JACOBS_HTTP = "JACOBS_HTTP"
    LAS_MANOS_HTTP = "LAS_MANOS_HTTP"


_CHANNELS = {
    JaxExternalOutputChannel.JACOBS_HTTP: ExternalOutputChannelId.JACOBS_SERVICE_HTTP_JSON_V1,
    JaxExternalOutputChannel.LAS_MANOS_HTTP: ExternalOutputChannelId.LAS_MANOS_SERVICE_HTTP_JSON_V1,
}
_UNAVAILABLE_JSON = b'{"error":"governed_output_unavailable"}'
_HTTP_ADAPTER: "GovernedJaxHTTPAdapter | None" = None


@dataclass(frozen=True)
class StructuredRouteContract:
    """Closed route-owned schema and provenance contract.

    ``origins`` covers every possible leaf of the native DTO.  Route handlers
    cannot append an unregistered field after F2-C because renderer coverage
    will reject it before F2-D preparation.
    """

    layout_id: str
    layout_version: str
    component_id: str
    channel: JaxExternalOutputChannel
    origins: tuple[OriginBinding, ...]
    presentation_maps: Mapping[str, Mapping[str, object]]
    status_code: int
    number_bindings: tuple[StructJSONNumberBinding, ...] = ()
    number_binding_contract_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.layout_id, str) or not self.layout_id:
            raise JaxExternalOutputError("route layout id is required")
        if not isinstance(self.layout_version, str) or not self.layout_version:
            raise JaxExternalOutputError("route layout version is required")
        if not isinstance(self.component_id, str) or not self.component_id:
            raise JaxExternalOutputError("route component id is required")
        if not isinstance(self.channel, JaxExternalOutputChannel):
            raise JaxExternalOutputError("route channel must be closed")
        if not isinstance(self.origins, tuple) or not self.origins:
            raise JaxExternalOutputError("route origins must be a nonempty tuple")
        if not all(isinstance(item, OriginBinding) for item in self.origins):
            raise JaxExternalOutputError("route origins must be typed")
        if not isinstance(self.presentation_maps, Mapping):
            raise JaxExternalOutputError("route presentation maps must be mapping")
        if not isinstance(self.status_code, int) or not 100 <= self.status_code <= 599:
            raise JaxExternalOutputError("route HTTP status is invalid")
        if (not isinstance(self.number_bindings, tuple)
                or not all(isinstance(binding, StructJSONNumberBinding)
                           for binding in self.number_bindings)):
            raise JaxExternalOutputError("route numeric bindings must be typed")
        if self.number_binding_contract_id is not None and (
                not isinstance(self.number_binding_contract_id, str)
                or not self.number_binding_contract_id):
            raise JaxExternalOutputError("route numeric binding contract must be server registered")

    @property
    def channel_id(self) -> ExternalOutputChannelId:
        return _CHANNELS[self.channel]


@dataclass(frozen=True)
class GovernedHTTPResponse:
    """Exact bytes ready for an HTTP adapter's transport-commit boundary."""

    status_code: int
    canonical_bytes: bytes
    transport_unit: StructuredGovernedTransportUnit

    def bytes_for_transport_commit(self, *, now: datetime | None = None) -> bytes:
        rendered = revalidate_structured_for_transport(
            self.transport_unit, now or datetime.now(timezone.utc))
        if rendered.canonical_bytes != self.canonical_bytes:
            raise JaxExternalOutputError("governed output changed before HTTP commitment")
        return rendered.canonical_bytes


class GovernedJSONResponse(Response):
    """ASGI response that revalidates the exact F2-D unit at commitment.

    The response intentionally does not manufacture a delivery acknowledgement:
    a successful ASGI send is only ``OUTPUT_COMMITTED_TO_TRANSPORT``.
    """

    media_type = "application/json"

    def __init__(self, prepared: GovernedHTTPResponse | None = None, *,
                 status_code: int = 503) -> None:
        self._prepared = prepared
        super().__init__(content=b"", status_code=(prepared.status_code if prepared else status_code),
                         media_type=self.media_type)

    async def __call__(self, scope, receive, send) -> None:
        if self._prepared is None:
            response = Response(_UNAVAILABLE_JSON, status_code=self.status_code,
                                media_type=self.media_type)
            await response(scope, receive, send)
            return
        try:
            body = self._prepared.bytes_for_transport_commit()
            response = Response(body, status_code=self._prepared.status_code,
                                media_type=self.media_type)
        except JaxExternalOutputError:
            response = Response(_UNAVAILABLE_JSON, status_code=503,
                                media_type=self.media_type)
        await response(scope, receive, send)


def runtime_status_templates() -> Mapping[tuple[str, str, str], str]:
    """Load the checked-in, hash-verified F2-E runtime templates.

    Template selection stays closed to the four accredited runtime predicates;
    a route cannot use this helper to render an unaccredited predicate.
    """
    from policy.governance.loaders import load_templates
    from policy.governance.runtime_status import RUNTIME_STATUS_API_VERSION

    loaded = load_templates()
    predicates = ("JOB_STATUS", "PIPELINE_STATUS", "FACET_RUNTIME_STATUS", "ENGINE_STATUS")
    templates: dict[tuple[str, str, str], str] = {}
    for predicate in predicates:
        spec = loaded.get(predicate)
        if spec is None or spec.status != "definida" or not isinstance(spec.template, str):
            raise JaxExternalOutputError("approved runtime template is unavailable")
        templates[(predicate, RUNTIME_STATUS_API_VERSION, "es")] = spec.template
    return templates


def response_scope(*, environment: str, tenant_id: str, subject_id: str | None,
                   component_id: str, actor_id: str = "service:jax",
                   audience: str = "service") -> ResponseScope:
    """Create a route scope only from server/canonical owner metadata."""
    return ResponseScope(environment, tenant_id, None, subject_id, actor_id,
                         audience, component_id, "f2e:" + str(uuid.uuid4()),
                         "f2e:" + str(uuid.uuid4()))


class GovernedJaxHTTPAdapter:
    """Shared JAX HTTP boundary above F2-A/B/C/D.

    The injected composition is process-owned.  No endpoint request, model,
    tool result, or producer can replace its resolver registry, receipt key,
    renderer context, layout identity, or F2-D metadata.
    """

    def __init__(self, composition: RuntimeOutputComposition) -> None:
        if not isinstance(composition, RuntimeOutputComposition):
            raise JaxExternalOutputError("JAX external output requires server composition")
        self._composition = composition

    async def prepare(self, *, contract: StructuredRouteContract,
                      scope: ResponseScope, payload: Mapping[str, object] | tuple[object, ...],
                      claims: tuple[RuntimeClaimRequest, ...] = ()) -> GovernedHTTPResponse:
        if not isinstance(contract, StructuredRouteContract) or not isinstance(scope, ResponseScope):
            raise JaxExternalOutputError("route contract and scope are required")
        if scope.component_id != contract.component_id:
            raise JaxExternalOutputError("route contract does not own response scope")
        if not isinstance(payload, (Mapping, tuple)) or not isinstance(claims, tuple):
            raise JaxExternalOutputError("structured payload and claims must be typed")
        try:
            composed = await self._composition.compose(
                scope=scope, response_id="f2e:" + str(uuid.uuid4()),
                producer=contract.component_id, tool_data=payload, requests=claims,
                number_binding_contract_id=contract.number_binding_contract_id,
            )
            layout = StructuredLayout(
                contract.layout_id, contract.layout_version, contract.channel_id.value, 0,
                composed.slots, contract.origins, contract.presentation_maps,
                number_bindings=composed.number_bindings,
                number_patterns=composed.number_patterns,
                number_binding_manifest=composed.number_binding_manifest,
            )
            registry = StructuredLayoutRegistry({layout.layout_id: layout})
            metadata = StructuredTransportMetadata(
                contract.channel_id.value, "application/json", "http-response",
                http_status=contract.status_code,
            )
            unit = GovernedExternalOutputAdapter(composed.context, registry,
                CHANNELS[contract.channel_id]).prepare_structured(
                    GovernedExternalOutputSubmission(composed.envelope, scope,
                        OutputOrigin.SYSTEM, contract.channel_id),
                    layout=layout, metadata=metadata,
                    idempotency_key="f2e-http:" + str(uuid.uuid4()),
                )
            if unit.rendered.contract_state is not ContractState.VALID:
                return GovernedHTTPResponse(503, _UNAVAILABLE_JSON, unit)
            return GovernedHTTPResponse(contract.status_code, unit.rendered.canonical_bytes, unit)
        except GovernanceContractError as exc:
            raise JaxExternalOutputError("external output governance unavailable") from exc


def configure_jax_external_http_adapter(adapter: GovernedJaxHTTPAdapter) -> None:
    """Install the one server-owned JAX HTTP adapter during composition."""
    global _HTTP_ADAPTER
    if not isinstance(adapter, GovernedJaxHTTPAdapter):
        raise JaxExternalOutputError("JAX external output requires typed adapter")
    if _HTTP_ADAPTER is not None:
        raise JaxExternalOutputError("JAX external output adapter is already configured")
    _HTTP_ADAPTER = adapter


def reset_jax_external_http_adapter_for_testing() -> None:
    """Test-only reset; production cannot replace route composition at runtime."""
    global _HTTP_ADAPTER
    _HTTP_ADAPTER = None


async def prepare_registered_http_output(*, contract: StructuredRouteContract,
                                         scope: ResponseScope,
                                         payload: Mapping[str, object] | tuple[object, ...],
                                         claims: tuple[RuntimeClaimRequest, ...] = ()) -> GovernedJSONResponse:
    """Prepare one registered route response, or fail closed to static JSON."""
    if _HTTP_ADAPTER is None:
        return GovernedJSONResponse()
    try:
        prepared = await _HTTP_ADAPTER.prepare(contract=contract, scope=scope,
                                                payload=payload, claims=claims)
    except JaxExternalOutputError:
        return GovernedJSONResponse()
    return GovernedJSONResponse(prepared)
