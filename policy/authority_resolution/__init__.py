"""Pure, static authority-rule selection for the C14N/3 candidate corpus."""

from .adapter import to_static_policy_view
from .candidate_loader import load_validated_candidate
from .models import (
    ApplicabilityState,
    ConditionResult,
    EvaluationContext,
    SelectionState,
    StaticAuthorityResolution,
    ValidatedCandidateCorpus,
    ValidatedStaticPolicyView,
)
from .resolver import resolve_static_authority
from .serialization import canonical_resolution_bytes, canonical_resolution_dict

__all__ = [
    "ApplicabilityState",
    "ConditionResult",
    "EvaluationContext",
    "SelectionState",
    "StaticAuthorityResolution",
    "ValidatedCandidateCorpus",
    "ValidatedStaticPolicyView",
    "load_validated_candidate",
    "to_static_policy_view",
    "resolve_static_authority",
    "canonical_resolution_dict",
    "canonical_resolution_bytes",
]
