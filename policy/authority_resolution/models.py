"""Frozen value objects used by the pure authority resolver."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
import unicodedata
from typing import Mapping

from .errors import InvalidEvaluationContextError, InvalidValidatedCorpusError, ResolverContractError

_ID = re.compile(r"[A-Z][A-Z0-9_]*\Z")
_DOC_ID = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")

class ApplicabilityState(str, Enum):
    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INDETERMINATE = "INDETERMINATE"

class SelectionState(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    SUPPRESSED = "SUPPRESSED"
    LOWER_PRECEDENCE = "LOWER_PRECEDENCE"
    CONTROLLING = "CONTROLLING"
    CONFLICTING = "CONFLICTING"
    IRRELEVANT = "IRRELEVANT"

_LAYERS = ("CONSTITUTIONAL_CORE", "PRODUCT_POLICY", "SUBORDINATE_POLICY")
_RANK = {name: 3 - i for i, name in enumerate(_LAYERS)}

def _nfc(value: object, label: str) -> str:
    if not isinstance(value, str) or unicodedata.normalize("NFC", value) != value:
        raise InvalidEvaluationContextError(f"{label}: string NFC inválido")
    return value

def _token(value: object, label: str) -> str:
    value = _nfc(value, label)
    if not _ID.fullmatch(value):
        raise InvalidEvaluationContextError(f"{label}: ID inválido")
    return value

def _set_tokens(values: object, label: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list, frozenset, set)):
        raise InvalidEvaluationContextError(f"{label}: colección inválida")
    result = tuple(_token(item, f"{label}[{i}]") for i, item in enumerate(values))
    if len(result) != len(set(result)):
        raise InvalidEvaluationContextError(f"{label}: duplicado")
    return tuple(sorted(result))

@dataclass(frozen=True)
class ConditionResult:
    id: str
    value: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _token(self.id, "condition.id"))
        if type(self.value) is not bool:
            raise InvalidEvaluationContextError("condition.value debe ser booleano JSON")

@dataclass(frozen=True)
class EvaluationContext:
    schema_version: str
    kind: str
    jurisdiction: str
    subject: str
    action: str
    conditions: tuple[ConditionResult, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "1.0" or self.kind != "JAX_AUTHORITY_EVALUATION_CONTEXT":
            raise InvalidEvaluationContextError("contexto con schema o kind inválido")
        for field, value in (("jurisdiction", self.jurisdiction), ("subject", self.subject), ("action", self.action)):
            if field == "jurisdiction":
                if value != "JAX":
                    raise InvalidEvaluationContextError("jurisdiction debe ser JAX")
            else:
                _token(value, f"context.{field}")
        if not isinstance(self.conditions, tuple):
            raise InvalidEvaluationContextError("conditions debe ser tuple")
        ids = [item.id for item in self.conditions]
        if any(not isinstance(item, ConditionResult) for item in self.conditions) or len(ids) != len(set(ids)):
            raise InvalidEvaluationContextError("conditions inválidas o duplicadas")
        object.__setattr__(self, "conditions", tuple(sorted(self.conditions, key=lambda x: x.id)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "EvaluationContext":
        if not isinstance(value, Mapping) or set(value) != {"schema_version", "kind", "jurisdiction", "subject", "action", "conditions"}:
            raise InvalidEvaluationContextError("EvaluationContext cerrado inválido")
        raw = value["conditions"]
        if not isinstance(raw, (list, tuple)):
            raise InvalidEvaluationContextError("conditions debe ser array")
        records = []
        for item in raw:
            if not isinstance(item, Mapping) or set(item) != {"id", "value"}:
                raise InvalidEvaluationContextError("ConditionResult cerrado inválido")
            records.append(ConditionResult(item["id"], item["value"]))
        return cls(value["schema_version"], value["kind"], value["jurisdiction"], value["subject"], value["action"], tuple(records))

@dataclass(frozen=True)
class FrozenScope:
    jurisdiction: str
    subjects: tuple[str, ...]
    actions: tuple[str, ...]
    conditions_all: tuple[str, ...]

@dataclass(frozen=True)
class FrozenRelationships:
    supersedes: tuple[str, ...]
    superseded_by: tuple[str, ...]

@dataclass(frozen=True)
class FrozenNormativeDocument:
    id: str
    document_class: str
    normative_layer: str
    normative_effect: str
    scope: FrozenScope
    relationships: FrozenRelationships

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _DOC_ID.fullmatch(self.id) or unicodedata.normalize("NFC", self.id) != self.id:
            raise ResolverContractError("document id inválido")
        if self.document_class not in _LAYERS or self.normative_layer != self.document_class or self.normative_effect != "ACTIVE_WHEN_CORPUS_ACTIVE":
            raise ResolverContractError("document class/layer/effect inválidos")
        if self.scope.jurisdiction != "JAX":
            raise ResolverContractError("document jurisdiction inválida")
        for collection in (self.scope.subjects, self.scope.actions, self.scope.conditions_all, self.relationships.supersedes, self.relationships.superseded_by):
            if tuple(collection) != tuple(sorted(collection)) or len(collection) != len(set(collection)):
                raise ResolverContractError("colección de documento no es SET canónico")

@dataclass(frozen=True)
class ProtectedConstraint:
    id: str
    statement: str
    non_waivable: bool = True

@dataclass(frozen=True)
class FrozenPrecedence:
    authority_meta_contract: str
    ordered_document_layers: tuple[str, ...]
    overlay_position: str
    external_constraints: str
    equal_rank_conflict: str
    unresolved_conflict: str

@dataclass(frozen=True)
class FrozenNormativeSources:
    permitted_document_classes: tuple[str, ...]
    manifest_classification_required: bool
    document_self_classification_authoritative: bool
    legacy_status_is_normative_force: bool
    unlisted_documents_have_normative_force: bool

@dataclass(frozen=True)
class FrozenExternalConstraints:
    jax_normative: bool
    effect: str
    may_grant_authority: bool
    provenance_required_at_evaluation: bool

@dataclass(frozen=True)
class FrozenFailClosed:
    default: str
    missing_or_invalid_root_pair: str
    invalid_membership: str
    unknown_document_class: str
    use_of_non_active_candidate: str
    invalid_or_expired_overlay: str
    unresolved_precedence: str

@dataclass(frozen=True)
class FrozenAuthorityMetaContract:
    scope_jurisdiction: str
    precedence: FrozenPrecedence
    normative_sources: FrozenNormativeSources
    protected_metanorms: tuple[ProtectedConstraint, ...]
    external_constraints: FrozenExternalConstraints
    fail_closed: FrozenFailClosed

    def __post_init__(self) -> None:
        if self.scope_jurisdiction != "JAX":
            raise ResolverContractError("Meta-Contract fuera de JAX")
        if self.precedence.authority_meta_contract != "ROOT_META_LEVEL":
            raise ResolverContractError("Meta-Contract no es root")
        if tuple(self.precedence.ordered_document_layers) != ("PROTECTED_METANORM", *_LAYERS):
            raise ResolverContractError("precedencia inválida")
        if tuple(self.normative_sources.permitted_document_classes) != _LAYERS:
            raise ResolverContractError("clases normativas inválidas")
        ids = tuple(x.id for x in self.protected_metanorms)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)) or any(x.non_waivable is not True for x in self.protected_metanorms):
            raise ResolverContractError("protected metanorms inválidas")
        if self.external_constraints != FrozenExternalConstraints(False, "CEILING_ONLY", False, True):
            raise ResolverContractError("external constraint declaration inválida")

@dataclass(frozen=True)
class ValidatedMember:
    id: str
    path: str
    document_class: str
    normative_layer: str
    normative_effect: str

@dataclass(frozen=True)
class ValidatedManifestBinding:
    manifest_id: str
    authority_id: str
    members: tuple[ValidatedMember, ...]

@dataclass(frozen=True)
class ValidatedCandidateCorpus:
    policy_corpus_hash: str
    canonicalizer_identity: str
    bootstrap_bundle_id: str
    authority: FrozenAuthorityMetaContract
    manifest: ValidatedManifestBinding
    normative_documents: tuple[FrozenNormativeDocument, ...]

    def __post_init__(self) -> None:
        if not _HASH.fullmatch(self.policy_corpus_hash) or not _HASH.fullmatch(self.bootstrap_bundle_id):
            raise InvalidValidatedCorpusError("hash inválido")
        if self.canonicalizer_identity != "JAX-POLICY-C14N/3":
            raise InvalidValidatedCorpusError("canonicalizer inválido")
        if not isinstance(self.authority, FrozenAuthorityMetaContract) or not isinstance(self.manifest, ValidatedManifestBinding):
            raise InvalidValidatedCorpusError("corpus no atómico")
        ids = tuple(d.id for d in self.normative_documents)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise InvalidValidatedCorpusError("documentos no ordenados o duplicados")

@dataclass(frozen=True)
class ValidatedStaticPolicyView:
    schema_version: str
    kind: str
    policy_corpus_hash: str
    canonicalizer_identity: str
    bootstrap_bundle_id: str
    authority_meta_contract: FrozenAuthorityMetaContract
    ordinary_documents: tuple[FrozenNormativeDocument, ...]

    def __post_init__(self) -> None:
        if (self.schema_version, self.kind) != ("1.0", "JAX_VALIDATED_STATIC_POLICY_VIEW"):
            raise ResolverContractError("StaticPolicyView inválida")
        if not isinstance(self.authority_meta_contract, FrozenAuthorityMetaContract):
            raise ResolverContractError("Meta-Contract no congelado")
        if tuple(d.id for d in self.ordinary_documents) != tuple(sorted(d.id for d in self.ordinary_documents)):
            raise ResolverContractError("documentos no ordenados")

@dataclass(frozen=True)
class RuleResolutionRecord:
    rule_id: str
    document_class: str
    normative_layer: str
    applicability_state: ApplicabilityState
    selection_state: SelectionState
    superseded_by: tuple[str, ...] = ()
    suppression_sources: tuple[str, ...] = ()
    indeterminacy_relevant: bool = False

@dataclass(frozen=True)
class ConflictRecord:
    kind: str
    precedence_layer: str
    rule_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind != "UNRESOLVED_SAME_RANK_MULTIPLE_RULES" or self.precedence_layer not in _LAYERS:
            raise ResolverContractError("ConflictRecord inválido")
        if len(self.rule_ids) < 2 or tuple(self.rule_ids) != tuple(sorted(self.rule_ids)) or len(set(self.rule_ids)) != len(self.rule_ids):
            raise ResolverContractError("ConflictRecord rule_ids inválidos")

@dataclass(frozen=True)
class StaticAuthorityResolution:
    schema_version: str
    kind: str
    resolver_identity: str
    resolver_version: str
    policy_corpus_hash: str
    context: EvaluationContext
    rule_results: tuple[RuleResolutionRecord, ...]
    controlling_rule_id: str | None
    winning_precedence_layer: str | None
    conflicts: tuple[ConflictRecord, ...]
    relevant_indeterminate_rule_ids: tuple[str, ...]
    protected_constraints: tuple[ProtectedConstraint, ...]
    external_constraint_mode: str
    resolution_status: str
