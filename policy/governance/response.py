"""F2-A structural output-governance contracts.

This module deliberately contains no resolver, renderer, transport, or I/O
behaviour.  It establishes the sealed value objects that later F2 blocks use
to prevent authority, evidence, memory, and current observations from being
silently conflated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Mapping


class GovernanceContractError(ValueError):
    """A response contract would fail closed at a future output boundary."""


class SourceClass(str, Enum):
    CURRENT_SOURCE = "CURRENT_SOURCE"
    EVIDENCE = "EVIDENCE"
    TOOL_RESULT = "TOOL_RESULT"
    MEMORY = "MEMORY"
    USER_INPUT = "USER_INPUT"
    MODEL_KNOWLEDGE = "MODEL_KNOWLEDGE"
    INFERENCE = "INFERENCE"
    UNKNOWN = "UNKNOWN"


class EpistemicStatus(str, Enum):
    CURRENT_OBSERVATION = "CURRENT_OBSERVATION"
    EVIDENCE_BOUND = "EVIDENCE_BOUND"
    USER_ASSERTED = "USER_ASSERTED"
    MEMORY_DERIVED = "MEMORY_DERIVED"
    MODEL_KNOWLEDGE = "MODEL_KNOWLEDGE"
    INFERRED = "INFERRED"
    UNVERIFIED = "UNVERIFIED"
    UNAVAILABLE = "UNAVAILABLE"


class AuthorityOrigin(str, Enum):
    HUMAN = "HUMAN"
    POLICY = "POLICY"
    SERVICE = "SERVICE"
    DELEGATED = "DELEGATED"
    SYSTEM = "SYSTEM"


class ClaimDisposition(str, Enum):
    ASSERTABLE = "ASSERTABLE"
    WITHHELD = "WITHHELD"
    UNAVAILABLE = "UNAVAILABLE"


class ContractState(str, Enum):
    VALID = "VALID"
    DEGRADED_STRUCTURED = "DEGRADED_STRUCTURED"
    BLOCKED_SYSTEM_CLAIM = "BLOCKED_SYSTEM_CLAIM"
    UNAVAILABLE = "UNAVAILABLE"


class ReferenceType(str, Enum):
    AUTHORITY = "AUTHORITY"
    EVIDENCE = "EVIDENCE"
    MEMORY = "MEMORY"
    TOOL_RESULT = "TOOL_RESULT"
    CURRENT_SOURCE = "CURRENT_SOURCE"
    USER_ASSERTION = "USER_ASSERTION"
    MODEL_KNOWLEDGE = "MODEL_KNOWLEDGE"
    INFERENCE = "INFERENCE"
    RESOLUTION_RECEIPT = "RESOLUTION_RECEIPT"
    ARTIFACT = "ARTIFACT"


class TemporalClass(str, Enum):
    CURRENT = "CURRENT"
    HISTORICAL = "HISTORICAL"
    ATEMPORAL = "ATEMPORAL"


class ExistenceState(str, Enum):
    PRESENT = "PRESENT"
    TOMBSTONED = "TOMBSTONED"
    UNKNOWN = "UNKNOWN"


class ContentBlockKind(str, Enum):
    NARRATIVE_TEXT = "NARRATIVE_TEXT"
    CLAIM_REF_BLOCK = "CLAIM_REF_BLOCK"
    ATTRIBUTED_QUOTE = "ATTRIBUTED_QUOTE"
    TOOL_DATA = "TOOL_DATA"
    SAFE_STATIC_NOTICE = "SAFE_STATIC_NOTICE"
    ERROR_NOTICE = "ERROR_NOTICE"


_TRUSTED_PAYLOAD_KEYS = frozenset({
    "authority", "authority_refs", "authority_origin", "source", "source_class",
    "verification", "verified", "epistemic_status", "trust", "trusted_label",
    "current_observation", "citation", "citations", "memory_label",
})


def _nonempty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GovernanceContractError(f"{field_name} must be a non-empty string")
    return value


def _unique(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if any(not isinstance(value, str) or not value for value in values):
        raise GovernanceContractError(f"{field_name} contains an invalid identifier")
    if len(values) != len(set(values)):
        raise GovernanceContractError(f"{field_name} contains duplicate identifiers")
    return tuple(sorted(values))


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _plain(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _contains_trusted_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(str(key).lower() in _TRUSTED_PAYLOAD_KEYS or _contains_trusted_key(item)
                   for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return any(_contains_trusted_key(item) for item in value)
    return False


def _canonical_response_bytes(value: Any) -> bytes:
    """Stable JSON for the F2-A sealed value object.

    The policy-wide Unicode-normalizing canonicalizer intentionally fails
    closed outside its pinned Unicode runtime.  Governance CI currently runs
    Python 3.12 with Unicode 15 while that package pins Unicode 16, so using
    it would make this pure structural contract unconstructable in its
    supported CI environment.  F2-A instead uses the established B9-style
    compact, sorted JSON representation and accepts only JSON-native values.
    """
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class ResponseScope:
    environment: str
    tenant_id: str
    project_id: str | None
    subject_id: str | None
    actor_id: str | None
    audience: str
    component_id: str
    request_id: str
    trace_id: str

    def __post_init__(self) -> None:
        for name in ("environment", "tenant_id", "audience", "component_id", "request_id", "trace_id"):
            _nonempty(getattr(self, name), name)
        if self.actor_id is not None:
            _nonempty(self.actor_id, "actor_id")
        if self.subject_id is not None:
            _nonempty(self.subject_id, "subject_id")
        if self.project_id is not None:
            _nonempty(self.project_id, "project_id")

    @property
    def scope_digest(self) -> str:
        return "sha256:" + hashlib.sha256(_canonical_response_bytes(self.semantic_projection())).hexdigest()

    def semantic_projection(self) -> dict[str, Any]:
        return {
            "environment": self.environment, "tenant_id": self.tenant_id,
            "project_id": self.project_id, "subject_id": self.subject_id,
            "actor_id": self.actor_id, "audience": self.audience,
            "component_id": self.component_id, "request_id": self.request_id,
            "trace_id": self.trace_id,
        }


@dataclass(frozen=True)
class ReferenceRef:
    ref_id: str
    ref_type: ReferenceType
    canonical_locator: str
    immutable_identity: str
    revision_or_digest: str | None
    scope_digest: str
    temporal_class: TemporalClass
    existence_state: ExistenceState

    def __post_init__(self) -> None:
        for name in ("ref_id", "canonical_locator", "immutable_identity", "scope_digest"):
            _nonempty(getattr(self, name), name)
        if self.existence_state is not ExistenceState.PRESENT:
            raise GovernanceContractError("a response reference must be structurally present")
        if self.ref_type in {ReferenceType.EVIDENCE, ReferenceType.MEMORY,
                             ReferenceType.AUTHORITY, ReferenceType.RESOLUTION_RECEIPT} and not self.revision_or_digest:
            raise GovernanceContractError("immutable governance references require revision_or_digest")


@dataclass(frozen=True)
class TemplateContract:
    template_id: str
    template_version: str
    locale: str

    def __post_init__(self) -> None:
        for name in ("template_id", "template_version", "locale"):
            _nonempty(getattr(self, name), name)


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    predicate: str
    typed_arguments: Mapping[str, Any]
    claim_scope: ResponseScope
    source_class: SourceClass
    epistemic_status: EpistemicStatus
    authority_refs: tuple[str, ...] = ()
    basis_refs: tuple[str, ...] = ()
    resolution_receipt_ref: str | None = None
    disposition: ClaimDisposition = ClaimDisposition.WITHHELD
    reason_codes: tuple[str, ...] = ()
    template_contract: TemplateContract | None = None

    def __post_init__(self) -> None:
        _nonempty(self.claim_id, "claim_id")
        _nonempty(self.predicate, "predicate")
        if not isinstance(self.typed_arguments, Mapping):
            raise GovernanceContractError("typed_arguments must be a mapping")
        object.__setattr__(self, "authority_refs", _unique(self.authority_refs, "authority_refs"))
        object.__setattr__(self, "basis_refs", _unique(self.basis_refs, "basis_refs"))
        object.__setattr__(self, "reason_codes", _unique(self.reason_codes, "reason_codes"))
        if self.resolution_receipt_ref is not None:
            _nonempty(self.resolution_receipt_ref, "resolution_receipt_ref")
        expected = {
            EpistemicStatus.CURRENT_OBSERVATION: SourceClass.CURRENT_SOURCE,
            EpistemicStatus.EVIDENCE_BOUND: SourceClass.EVIDENCE,
            EpistemicStatus.USER_ASSERTED: SourceClass.USER_INPUT,
            EpistemicStatus.MEMORY_DERIVED: SourceClass.MEMORY,
            EpistemicStatus.MODEL_KNOWLEDGE: SourceClass.MODEL_KNOWLEDGE,
            EpistemicStatus.INFERRED: SourceClass.INFERENCE,
        }.get(self.epistemic_status)
        if expected is not None and self.source_class is not expected:
            raise GovernanceContractError("source_class and epistemic_status are not a permitted pairing")
        if self.epistemic_status is EpistemicStatus.CURRENT_OBSERVATION:
            if not self.resolution_receipt_ref:
                raise GovernanceContractError("CURRENT_OBSERVATION requires resolution_receipt_ref")
            if self.disposition is not ClaimDisposition.ASSERTABLE:
                raise GovernanceContractError("CURRENT_OBSERVATION must be ASSERTABLE or omitted")
        if self.epistemic_status is EpistemicStatus.USER_ASSERTED and not self.basis_refs:
            raise GovernanceContractError("USER_ASSERTED requires an attributed source reference")
        if self.disposition is ClaimDisposition.ASSERTABLE and self.template_contract is None:
            raise GovernanceContractError("ASSERTABLE claims require a template_contract")


@dataclass(frozen=True)
class ContentBlock:
    kind: ContentBlockKind
    payload: Any = None
    claim_refs: tuple[str, ...] = ()
    attribution_ref: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_refs", _unique(self.claim_refs, "claim_refs"))
        if _contains_trusted_key(self.payload):
            raise GovernanceContractError("untrusted payload may not define trusted metadata")
        if self.kind is ContentBlockKind.NARRATIVE_TEXT:
            if not isinstance(self.payload, str) or self.claim_refs or self.attribution_ref:
                raise GovernanceContractError("NARRATIVE_TEXT cannot own claims or attribution")
        elif self.kind is ContentBlockKind.CLAIM_REF_BLOCK:
            if self.payload is not None or not self.claim_refs or self.attribution_ref:
                raise GovernanceContractError("CLAIM_REF_BLOCK contains only claim identities")
        elif self.kind is ContentBlockKind.ATTRIBUTED_QUOTE:
            if not isinstance(self.payload, str) or not self.attribution_ref or self.claim_refs:
                raise GovernanceContractError("ATTRIBUTED_QUOTE requires payload and attribution")
        elif self.kind is ContentBlockKind.SAFE_STATIC_NOTICE:
            if not isinstance(self.payload, str) or self.claim_refs or self.attribution_ref:
                raise GovernanceContractError("SAFE_STATIC_NOTICE cannot carry dynamic trust metadata")
        elif self.kind is ContentBlockKind.ERROR_NOTICE:
            if not isinstance(self.payload, str) or self.claim_refs:
                raise GovernanceContractError("ERROR_NOTICE cannot own claims")
        elif self.kind is ContentBlockKind.TOOL_DATA:
            if self.claim_refs or self.attribution_ref:
                raise GovernanceContractError("TOOL_DATA cannot own claims or attribution")


@dataclass(frozen=True)
class GovernanceReceipt:
    policy_version: str
    registry_snapshot_digest: str
    validator_version: str

    def __post_init__(self) -> None:
        for name in ("policy_version", "registry_snapshot_digest", "validator_version"):
            _nonempty(getattr(self, name), name)


@dataclass(frozen=True)
class GovernedResponseEnvelope:
    schema_version: str
    response_id: str
    request_id: str
    trace_id: str
    response_scope: ResponseScope
    producer: str
    issuance_authority_refs: tuple[str, ...]
    content_blocks: tuple[ContentBlock, ...]
    claims: tuple[ClaimRecord, ...]
    references: tuple[ReferenceRef, ...]
    contract_state: ContractState
    governance_receipt: GovernanceReceipt
    envelope_digest: str | None = field(default=None)

    def __post_init__(self) -> None:
        for name in ("schema_version", "response_id", "request_id", "trace_id", "producer"):
            _nonempty(getattr(self, name), name)
        if self.request_id != self.response_scope.request_id or self.trace_id != self.response_scope.trace_id:
            raise GovernanceContractError("envelope identity must match response scope")
        object.__setattr__(self, "issuance_authority_refs", _unique(self.issuance_authority_refs, "issuance_authority_refs"))
        claim_ids = tuple(claim.claim_id for claim in self.claims)
        if len(claim_ids) != len(set(claim_ids)):
            raise GovernanceContractError("duplicate claim_id")
        refs = {ref.ref_id: ref for ref in self.references}
        if len(refs) != len(self.references):
            raise GovernanceContractError("duplicate ref_id")
        self._validate_references(refs)
        claim_by_id = {claim.claim_id: claim for claim in self.claims}
        for block in self.content_blocks:
            if block.kind is ContentBlockKind.CLAIM_REF_BLOCK:
                for claim_id in block.claim_refs:
                    claim = claim_by_id.get(claim_id)
                    if claim is None or claim.disposition is not ClaimDisposition.ASSERTABLE:
                        raise GovernanceContractError("CLAIM_REF_BLOCK may reference only ASSERTABLE claims")
            if block.kind is ContentBlockKind.ATTRIBUTED_QUOTE:
                ref = refs.get(block.attribution_ref or "")
                if ref is None or ref.ref_type is not ReferenceType.USER_ASSERTION:
                    raise GovernanceContractError("ATTRIBUTED_QUOTE requires USER_ASSERTION reference")
                if ref.scope_digest != self.response_scope.scope_digest:
                    raise GovernanceContractError("ATTRIBUTED_QUOTE reference scope mismatch")
        expected_digest = self.compute_digest()
        if self.envelope_digest is not None and self.envelope_digest != expected_digest:
            raise GovernanceContractError("envelope_digest does not match canonical envelope")
        object.__setattr__(self, "envelope_digest", expected_digest)

    def _validate_references(self, refs: Mapping[str, ReferenceRef]) -> None:
        for ref_id in self.issuance_authority_refs:
            ref = refs.get(ref_id)
            if ref is None or ref.ref_type is not ReferenceType.AUTHORITY:
                raise GovernanceContractError("issuance authority must reference AUTHORITY")
            if ref.scope_digest != self.response_scope.scope_digest:
                raise GovernanceContractError("issuance authority reference scope mismatch")
        for claim in self.claims:
            if claim.claim_scope.scope_digest != self.response_scope.scope_digest:
                raise GovernanceContractError("claim scope must equal response scope in F2-A")
            for ref_id in claim.authority_refs:
                ref = refs.get(ref_id)
                if ref is None or ref.ref_type is not ReferenceType.AUTHORITY:
                    raise GovernanceContractError("claim authority must reference AUTHORITY")
                if ref.scope_digest != self.response_scope.scope_digest:
                    raise GovernanceContractError("claim authority reference scope mismatch")
            for ref_id in claim.basis_refs:
                ref = refs.get(ref_id)
                if ref is None:
                    raise GovernanceContractError("claim basis reference is missing")
                if ref.scope_digest != self.response_scope.scope_digest:
                    raise GovernanceContractError("claim basis reference scope mismatch")
            if claim.resolution_receipt_ref is not None:
                ref = refs.get(claim.resolution_receipt_ref)
                if ref is None or ref.ref_type is not ReferenceType.RESOLUTION_RECEIPT:
                    raise GovernanceContractError("resolution receipt must reference RESOLUTION_RECEIPT")
            if claim.epistemic_status is EpistemicStatus.USER_ASSERTED:
                if not any(refs[ref_id].ref_type is ReferenceType.USER_ASSERTION for ref_id in claim.basis_refs):
                    raise GovernanceContractError("USER_ASSERTED requires USER_ASSERTION basis")
            if claim.epistemic_status is EpistemicStatus.MEMORY_DERIVED:
                if not any(refs[ref_id].ref_type is ReferenceType.MEMORY for ref_id in claim.basis_refs):
                    raise GovernanceContractError("MEMORY_DERIVED requires MEMORY basis")
            if claim.epistemic_status is EpistemicStatus.EVIDENCE_BOUND:
                if not any(refs[ref_id].ref_type is ReferenceType.EVIDENCE for ref_id in claim.basis_refs):
                    raise GovernanceContractError("EVIDENCE_BOUND requires EVIDENCE basis")
            if claim.epistemic_status is EpistemicStatus.CURRENT_OBSERVATION:
                receipt = refs[claim.resolution_receipt_ref or ""]
                if receipt.scope_digest != self.response_scope.scope_digest:
                    raise GovernanceContractError("resolution receipt scope mismatch")

    def canonical_projection(self) -> dict[str, Any]:
        value = _plain(self)
        value.pop("envelope_digest", None)
        return value

    def compute_digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.to_canonical_bytes()).hexdigest()

    def to_canonical_bytes(self) -> bytes:
        """Stable F2-A serialization used for sealing and storage boundaries."""
        return _canonical_response_bytes(self.canonical_projection())

    @classmethod
    def from_canonical_projection(cls, value: Mapping[str, Any]) -> "GovernedResponseEnvelope":
        """Rebuild a sealed F2-A value without accepting a caller-supplied digest."""
        scope_value = value["response_scope"]
        response_scope = ResponseScope(**scope_value)
        references = tuple(ReferenceRef(
            ref_id=item["ref_id"], ref_type=ReferenceType(item["ref_type"]),
            canonical_locator=item["canonical_locator"], immutable_identity=item["immutable_identity"],
            revision_or_digest=item["revision_or_digest"], scope_digest=item["scope_digest"],
            temporal_class=TemporalClass(item["temporal_class"]),
            existence_state=ExistenceState(item["existence_state"]),
        ) for item in value["references"])
        claims = tuple(ClaimRecord(
            claim_id=item["claim_id"], predicate=item["predicate"],
            typed_arguments=item["typed_arguments"], claim_scope=ResponseScope(**item["claim_scope"]),
            source_class=SourceClass(item["source_class"]),
            epistemic_status=EpistemicStatus(item["epistemic_status"]),
            authority_refs=tuple(item["authority_refs"]), basis_refs=tuple(item["basis_refs"]),
            resolution_receipt_ref=item["resolution_receipt_ref"],
            disposition=ClaimDisposition(item["disposition"]), reason_codes=tuple(item["reason_codes"]),
            template_contract=(TemplateContract(**item["template_contract"])
                               if item["template_contract"] is not None else None),
        ) for item in value["claims"])
        blocks = tuple(ContentBlock(
            kind=ContentBlockKind(item["kind"]), payload=item["payload"],
            claim_refs=tuple(item["claim_refs"]), attribution_ref=item["attribution_ref"],
        ) for item in value["content_blocks"])
        receipt = GovernanceReceipt(**value["governance_receipt"])
        return cls(
            schema_version=value["schema_version"], response_id=value["response_id"],
            request_id=value["request_id"], trace_id=value["trace_id"],
            response_scope=response_scope, producer=value["producer"],
            issuance_authority_refs=tuple(value["issuance_authority_refs"]), content_blocks=blocks,
            claims=claims, references=references, contract_state=ContractState(value["contract_state"]),
            governance_receipt=receipt,
        )
