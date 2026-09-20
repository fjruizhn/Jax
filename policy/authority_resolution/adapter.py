"""Freeze the atomic loader result into the lifecycle-neutral resolver view."""
from __future__ import annotations

from .errors import ResolverContractError
from .models import ValidatedCandidateCorpus, ValidatedStaticPolicyView

def to_static_policy_view(corpus: ValidatedCandidateCorpus) -> ValidatedStaticPolicyView:
    if not isinstance(corpus, ValidatedCandidateCorpus):
        raise ResolverContractError("adapter sólo acepta ValidatedCandidateCorpus")
    docs = tuple(sorted(corpus.normative_documents, key=lambda d: d.id))
    return ValidatedStaticPolicyView(
        "1.0", "JAX_VALIDATED_STATIC_POLICY_VIEW", corpus.policy_corpus_hash,
        corpus.canonicalizer_identity, corpus.bootstrap_bundle_id,
        corpus.authority, docs,
    )
