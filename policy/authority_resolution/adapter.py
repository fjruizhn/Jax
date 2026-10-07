"""Freeze the atomic loader result into the lifecycle-neutral resolver view."""
from __future__ import annotations

from .errors import ResolverContractError
from .models import ValidatedCandidateCorpus, ValidatedStaticPolicyView

def to_static_policy_view(corpus: ValidatedCandidateCorpus) -> ValidatedStaticPolicyView:
    # Tipo EXACTO y método de la clase sin enlazar: ni una subclase ni un atributo
    # de instancia pueden responder por sí mismos que están validados.
    if type(corpus) is not ValidatedCandidateCorpus or not ValidatedCandidateCorpus._was_loader_validated(corpus):
        raise ResolverContractError("adapter sólo acepta ValidatedCandidateCorpus")
    docs = tuple(sorted(corpus.normative_documents, key=lambda d: d.id))
    return ValidatedStaticPolicyView(
        "1.0", "JAX_VALIDATED_STATIC_POLICY_VIEW", corpus.policy_corpus_hash,
        corpus.canonicalizer_identity, corpus.bootstrap_bundle_id,
        corpus.authority, docs,
    )
