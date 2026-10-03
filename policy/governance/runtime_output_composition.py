"""Server-owned F2-B composition for structured runtime-status output.

Routes supply only their already canonical claim requests and typed platform
bridge.  This module owns the registry, per-process receipt authenticator,
receipt references, render context, and F2-A sealing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import secrets
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from .governed_renderer import GovernedDomainRegistry, RenderContext
from .resolution import (
    AdapterKind, ReceiptAuthenticator, ReferenceLookupRecord, ResolutionStatus,
    RuntimeStatusEvidence, ResolverRegistry,
)
from .response import (
    ClaimDisposition, ClaimRecord, ContentBlock, ContentBlockKind, ContractState,
    EpistemicStatus, GovernedResponseCandidate, GovernedResponseEnvelope,
    GovernanceContractError, GovernanceReceipt, ReferenceRef, ReferenceType,
    ResponseScope, SourceClass, TemplateContract, TemporalClass, ExistenceState,
    _freeze, _plain, _seal_candidate_for_server, _text,
)
from .runtime_status import (
    JacobsPipelineStatusResolver, MotorJobStatusResolver, PlatformRuntimeStatusSnapshot,
    build_runtime_status_registry, platform_runtime_status_evidence,
)
from .structured_output import (
    ExpandedNumberBindingManifest, SlotProjectionMode, StructJSONNumberBinding,
    StructJSONNumberPattern, StructuredClaimSlot, StructuredNumberBindingContract,
    expand_structured_number_patterns, pack_structured_float64_carriers,
)


def runtime_governance_receipt(registry: ResolverRegistry) -> GovernanceReceipt:
    """Derive envelope identity from the exact registry and checked-in policy files."""
    if not isinstance(registry, ResolverRegistry):
        raise RuntimeOutputCompositionError("runtime receipt requires exact resolver registry")
    root = Path(__file__).resolve().parents[2]
    try:
        policy_bytes = (root / "policy" / "vocabulary" / "predicates.yaml").read_bytes()
        vocabulary_bytes = (root / "policy" / "vocabulary" / "closed_vocabulary.yaml").read_bytes()
    except OSError as exc:
        raise RuntimeOutputCompositionError("runtime governance identity files unavailable") from exc
    digest = lambda value: "sha256:" + hashlib.sha256(value).hexdigest()
    from .runtime_status import RUNTIME_STATUS_API_VERSION
    from .structured_output import STRUCTURED_RENDERER_API_VERSION, STRUCTURED_OUTPUT_SCHEMA_VERSION
    return GovernanceReceipt(digest(policy_bytes), digest(vocabulary_bytes), registry.snapshot_digest,
        RUNTIME_STATUS_API_VERSION, STRUCTURED_RENDERER_API_VERSION + ":" + STRUCTURED_OUTPUT_SCHEMA_VERSION)


_RUNTIME_PREDICATES = frozenset({"JOB_STATUS", "PIPELINE_STATUS", "FACET_RUNTIME_STATUS", "ENGINE_STATUS"})
_PLATFORM_PREDICATES = frozenset({"FACET_RUNTIME_STATUS", "ENGINE_STATUS"})


class RuntimeOutputCompositionError(GovernanceContractError):
    pass


@dataclass(frozen=True)
class RuntimeClaimRequest:
    """A route-contract request, never a model/request payload schema."""
    predicate: str
    canonical_arguments: Mapping[str, object]
    json_pointer: str
    presentation_map_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicate", _text(self.predicate, "runtime predicate"))
        if self.predicate not in _RUNTIME_PREDICATES:
            raise RuntimeOutputCompositionError("runtime output predicate is not accredited")
        if not isinstance(self.canonical_arguments, Mapping):
            raise RuntimeOutputCompositionError("runtime claim arguments must be mapping")
        frozen = _freeze(self.canonical_arguments, "runtime claim arguments")
        expected = {"job_id", "status"} if self.predicate == "JOB_STATUS" else (
            {"pipeline_id", "status"} if self.predicate == "PIPELINE_STATUS" else {"name", "status"})
        if set(frozen) != expected or not all(isinstance(value, str) and value for value in frozen.values()):
            raise RuntimeOutputCompositionError("runtime claim arguments are not canonical")
        object.__setattr__(self, "canonical_arguments", frozen)
        if not isinstance(self.json_pointer, str) or not self.json_pointer.startswith("/"):
            raise RuntimeOutputCompositionError("runtime JSON pointer must be non-root pointer")
        if self.presentation_map_id is not None:
            object.__setattr__(self, "presentation_map_id", _text(self.presentation_map_id, "presentation map id"))


@dataclass(frozen=True)
class RuntimeSlotContract:
    """Registered route layout binding; never model/request metadata."""
    predicate: str
    json_pointer: str
    argument_pointers: Mapping[str, str]
    server_constants: Mapping[str, object] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicate", _text(self.predicate, "slot predicate"))
        if self.predicate not in _RUNTIME_PREDICATES or not isinstance(self.json_pointer, str) or not self.json_pointer.startswith("/"):
            raise RuntimeOutputCompositionError("runtime slot contract is invalid")
        expected = {"job_id", "status"} if self.predicate == "JOB_STATUS" else (
            {"pipeline_id", "status"} if self.predicate == "PIPELINE_STATUS" else {"name", "status"})
        if not isinstance(self.argument_pointers, Mapping) or "status" not in self.argument_pointers:
            raise RuntimeOutputCompositionError("runtime slot contract needs status pointer")
        pointers = dict(self.argument_pointers)
        constants = dict(self.server_constants or {})
        if (pointers["status"] != self.json_pointer or set(pointers) & set(constants)
                or set(pointers) | set(constants) != expected):
            raise RuntimeOutputCompositionError("runtime slot contract must partition canonical arguments")
        if not all(isinstance(key, str) and isinstance(value, str) and value.startswith("/") for key, value in pointers.items()):
            raise RuntimeOutputCompositionError("runtime slot contract pointer invalid")
        object.__setattr__(self, "argument_pointers", MappingProxyType(dict(sorted(pointers.items()))))
        object.__setattr__(self, "server_constants", _freeze(constants, "runtime slot constants"))


@dataclass(frozen=True)
class RuntimeComposedOutput:
    envelope: GovernedResponseEnvelope
    context: RenderContext
    slots: tuple[StructuredClaimSlot, ...]
    number_bindings: tuple[StructJSONNumberBinding, ...] = ()
    number_binding_contract_id: str | None = None
    number_patterns: tuple[StructJSONNumberPattern, ...] = ()
    number_binding_manifest: ExpandedNumberBindingManifest = field(
        default_factory=lambda: ExpandedNumberBindingManifest((), {}))


PlatformEvidenceBridge = Callable[[RuntimeClaimRequest, ResponseScope], RuntimeStatusEvidence]
RuntimeRegistryFactory = Callable[[ResponseScope, ReceiptAuthenticator], ResolverRegistry]


def _pointer_parent(pointer: str) -> str:
    parent, _, _tail = pointer.rpartition("/")
    return parent


def _template_contract(value: str | None) -> TemplateContract:
    if not isinstance(value, str) or "@" not in value or ":" not in value:
        raise RuntimeOutputCompositionError("runtime registry has no template contract")
    template_id, rest = value.split("@", 1)
    version, locale = rest.rsplit(":", 1)
    return TemplateContract(template_id, version, locale)


class RuntimeOutputComposition:
    """Long-lived trusted server composition; never serialized or request-built."""
    def __init__(self, *, platform_source_configuration: Mapping[str, Mapping[str, object]] | None,
                 governance_receipt: GovernanceReceipt | None, templates: Mapping[tuple[str, str, str], str],
                 notices: Mapping[str, str] | None = None,
                 platform_evidence_bridge: PlatformEvidenceBridge | None = None,
                 registry_factory: RuntimeRegistryFactory | None = None,
                 slot_contracts: Mapping[tuple[str, str], RuntimeSlotContract] | None = None,
                 number_bindings: tuple[StructJSONNumberBinding, ...] = (),
                 number_binding_contracts: Mapping[str, StructuredNumberBindingContract | tuple[StructJSONNumberBinding, ...]] | None = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 authenticator: ReceiptAuthenticator | None = None) -> None:
        if governance_receipt is not None and not isinstance(governance_receipt, GovernanceReceipt):
            raise RuntimeOutputCompositionError("runtime output receipt must be server owned")
        if not isinstance(templates, Mapping) or not callable(clock):
            raise RuntimeOutputCompositionError("runtime output composition requires trusted configuration")
        if platform_evidence_bridge is not None and not callable(platform_evidence_bridge):
            raise RuntimeOutputCompositionError("platform evidence bridge must be server callable")
        if registry_factory is not None and not callable(registry_factory):
            raise RuntimeOutputCompositionError("runtime registry factory must be server callable")
        if platform_source_configuration is None and registry_factory is None:
            raise RuntimeOutputCompositionError("runtime composition needs explicit source configuration or registry factory")
        # Production callers use a unique key, never the test convenience key.
        if authenticator is None:
            authenticator = ReceiptAuthenticator(secrets.token_bytes(32),
                key_id="runtime-output-process:" + secrets.token_hex(16))
        if not isinstance(authenticator, ReceiptAuthenticator) or authenticator.key_id == "test-ephemeral":
            raise RuntimeOutputCompositionError("runtime output requires non-test receipt authenticator")
        self._platform_source_configuration = (None if platform_source_configuration is None
            else _freeze(platform_source_configuration, "platform runtime source configuration"))
        self._governance_receipt = governance_receipt
        self._templates = MappingProxyType(dict(templates))
        self._notices = MappingProxyType(dict(notices or {}))
        self._platform_evidence_bridge = platform_evidence_bridge
        self._registry_factory = registry_factory
        contracts = dict(slot_contracts or {})
        if not all(isinstance(key, tuple) and len(key) == 2 and isinstance(value, RuntimeSlotContract)
                   and key == (value.predicate, value.json_pointer) for key, value in contracts.items()):
            raise RuntimeOutputCompositionError("runtime slot contracts must be server registered")
        self._slot_contracts = MappingProxyType(contracts)
        if (not isinstance(number_bindings, tuple)
                or not all(isinstance(binding, StructJSONNumberBinding) for binding in number_bindings)):
            raise RuntimeOutputCompositionError("runtime numeric bindings must be server registered")
        if number_binding_contracts is not None and number_bindings:
            raise RuntimeOutputCompositionError("runtime numeric bindings need one registration mode")
        self._number_bindings = number_bindings
        contracts_by_id: dict[str, StructuredNumberBindingContract] = {}
        if number_binding_contracts is not None:
            if not isinstance(number_binding_contracts, Mapping):
                raise RuntimeOutputCompositionError("runtime numeric binding contracts must be mapping")
            for contract_id, contract in number_binding_contracts.items():
                contract_id = _text(contract_id, "runtime numeric binding contract id")
                # Tuple is retained for the existing fixed-binding registrations;
                # it is normalized immediately into the one route contract type.
                if isinstance(contract, tuple):
                    contract = StructuredNumberBindingContract(contract)
                if not isinstance(contract, StructuredNumberBindingContract):
                    raise RuntimeOutputCompositionError("runtime numeric binding contract invalid")
                contracts_by_id[contract_id] = contract
        self._number_binding_contracts = MappingProxyType(dict(sorted(contracts_by_id.items())))
        self._clock = clock
        self._authenticator = authenticator

    async def compose(self, *, scope: ResponseScope, response_id: str, producer: str,
                      tool_data: Mapping[str, object] | tuple[object, ...],
                      requests: tuple[RuntimeClaimRequest, ...],
                      number_binding_contract_id: str | None = None) -> RuntimeComposedOutput:
        if not isinstance(scope, ResponseScope) or producer != scope.component_id:
            raise RuntimeOutputCompositionError("runtime output producer must equal trusted scope component")
        if not isinstance(response_id, str) or not response_id or not isinstance(requests, tuple):
            raise RuntimeOutputCompositionError("runtime output needs response id and typed requests")
        if not all(isinstance(request, RuntimeClaimRequest) for request in requests):
            raise RuntimeOutputCompositionError("runtime output requests must be typed")
        if len({request.json_pointer for request in requests}) != len(requests):
            raise RuntimeOutputCompositionError("runtime output cannot bind duplicate JSON pointers")
        number_contract = self._number_contract_for(number_binding_contract_id)
        expanded, manifest = expand_structured_number_patterns(tool_data, number_contract.patterns)
        bindings = number_contract.bindings + expanded
        # A no-claim output still has a governed envelope: its receipt and
        # context bind the real server registry for this exact trusted scope.
        registry = self._registry(scope)
        claims: list[ClaimRecord] = []
        references: list[ReferenceRef] = []
        receipts = {}
        lookups = {}
        slots: list[StructuredClaimSlot] = []
        for index, request in enumerate(requests):
            evidence = await self._evidence(request, scope)
            now = self._clock()
            if not isinstance(now, datetime) or now.tzinfo is None:
                raise RuntimeOutputCompositionError("runtime output clock must be timezone-aware")
            receipt = registry.resolve(request.predicate, request.canonical_arguments, scope,
                validation_time=now, runtime_status_evidence=evidence)
            if receipt.status is not ResolutionStatus.RESOLVED:
                raise RuntimeOutputCompositionError("runtime status source is unavailable or scope/configuration mismatched")
            reference_id = f"runtime-receipt:{index}"
            reference = ReferenceRef(reference_id, ReferenceType.RESOLUTION_RECEIPT,
                "axioma://governance/runtime-receipt/" + receipt.receipt_id,
                "runtime-receipt:" + receipt.receipt_id, receipt.receipt_id, scope.scope_digest,
                TemporalClass.CURRENT, ExistenceState.PRESENT)
            entry = registry._entries.get(request.predicate) if registry is not None else None
            if entry is None:
                raise RuntimeOutputCompositionError("runtime registry entry disappeared")
            claim_id = f"runtime-claim:{index}"
            claims.append(ClaimRecord(claim_id, request.predicate, request.canonical_arguments, scope,
                SourceClass.CURRENT_SOURCE, EpistemicStatus.CURRENT_OBSERVATION,
                resolution_receipt_ref=reference_id, disposition=ClaimDisposition.ASSERTABLE,
                template_contract=_template_contract(entry.template_contract_ref)))
            references.append(reference)
            receipts[reference_id] = receipt
            lookups[reference_id] = ReferenceLookupRecord(reference.ref_id, reference.ref_type,
                reference.canonical_locator, reference.immutable_identity, reference.revision_or_digest,
                reference.scope_digest, reference.temporal_class, reference.existence_state, True)
            slot_contract = self._slot_contracts.get((request.predicate, request.json_pointer))
            if slot_contract is None:
                raise RuntimeOutputCompositionError("runtime output lacks server slot contract")
            if any(claims[-1].typed_arguments[name] != value for name, value in slot_contract.server_constants.items()):
                raise RuntimeOutputCompositionError("runtime server slot constant mismatches accredited claim")
            slots.append(StructuredClaimSlot(request.predicate, _pointer_parent(request.json_pointer), claim_id,
                slot_contract.argument_pointers,
                projection_mode=SlotProjectionMode.PRESENTATION_MAP if request.presentation_map_id else SlotProjectionMode.EXACT,
                presentation_argument="status" if request.presentation_map_id else None,
                presentation_map_id=request.presentation_map_id,
                server_constants=slot_contract.server_constants))
        try:
            packed_tool_data = pack_structured_float64_carriers(tool_data, bindings,
                tuple(slots))
        except GovernanceContractError as exc:
            raise RuntimeOutputCompositionError("runtime structured numeric carrier invalid") from exc
        blocks = (ContentBlock(ContentBlockKind.TOOL_DATA, packed_tool_data),)
        if claims:
            blocks += (ContentBlock(ContentBlockKind.CLAIM_REF_BLOCK,
                claim_refs=tuple(claim.claim_id for claim in claims)),)
        candidate = GovernedResponseCandidate("f2-c.1", response_id, scope.request_id, scope.trace_id,
            scope, producer, (), blocks, tuple(claims), tuple(references))
        receipt_identity = self._governance_receipt or runtime_governance_receipt(registry)
        envelope = _seal_candidate_for_server(candidate, contract_state=ContractState.VALID,
            governance_receipt=receipt_identity)

        def valid_reference(reference: object, reference_scope: ResponseScope) -> bool:
            record = lookups.get(getattr(reference, "ref_id", ""))
            return bool(record and reference_scope.scope_digest == scope.scope_digest
                and record.ref_id == getattr(reference, "ref_id", None)
                and record.revision_or_digest == getattr(reference, "revision_or_digest", None)
                and record.accessible)

        context = RenderContext(registry, receipts, self._templates, self._notices,
            GovernedDomainRegistry(), valid_reference, self._clock,
            receipt_reference_resolver=lambda reference, reference_scope:
                lookups.get(getattr(reference, "ref_id", "")) if valid_reference(reference, reference_scope) else None)
        return RuntimeComposedOutput(envelope, context, tuple(slots), bindings, number_binding_contract_id,
            number_contract.patterns, manifest)

    def _number_contract_for(self, contract_id: str | None) -> StructuredNumberBindingContract:
        """Select one closed server registration; raw payloads never supply bindings."""
        if contract_id is None:
            return StructuredNumberBindingContract(self._number_bindings)
        if not isinstance(contract_id, str) or contract_id not in self._number_binding_contracts:
            raise RuntimeOutputCompositionError("runtime numeric binding contract is unregistered")
        return self._number_binding_contracts[contract_id]

    def _registry(self, scope: ResponseScope) -> ResolverRegistry:
        if self._registry_factory is not None:
            registry = self._registry_factory(scope, self._authenticator)
            if not isinstance(registry, ResolverRegistry):
                raise RuntimeOutputCompositionError("runtime registry factory returned invalid registry")
            return registry
        if self._platform_source_configuration is None:
            raise RuntimeOutputCompositionError("runtime source configuration is unavailable")
        return build_runtime_status_registry(scope, authenticator=self._authenticator,
            platform_source_configuration=self._platform_source_configuration)

    async def _evidence(self, request: RuntimeClaimRequest, scope: ResponseScope) -> RuntimeStatusEvidence:
        if request.predicate == "JOB_STATUS":
            return MotorJobStatusResolver().evidence(request.canonical_arguments, scope)
        if request.predicate == "PIPELINE_STATUS":
            return await JacobsPipelineStatusResolver().evidence(request.canonical_arguments, scope)
        if self._platform_evidence_bridge is None:
            raise RuntimeOutputCompositionError("platform runtime evidence bridge is unavailable")
        evidence = self._platform_evidence_bridge(request, scope)
        if not isinstance(evidence, RuntimeStatusEvidence):
            raise RuntimeOutputCompositionError("platform runtime bridge returned untyped evidence")
        expected = AdapterKind.FACET_RUNTIME_STATUS if request.predicate == "FACET_RUNTIME_STATUS" else AdapterKind.ENGINE_STATUS
        if evidence.adapter_kind is not expected:
            raise RuntimeOutputCompositionError("platform runtime bridge returned wrong source kind")
        return evidence
