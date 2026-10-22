"""Freeze the atomic loader result into the lifecycle-neutral resolver view."""
from __future__ import annotations

from .errors import ResolverContractError
from .models import (
    ValidatedCandidateCorpus,
    ValidatedStaticPolicyView,
    _VALIDATED_CANDIDATE_SEAL,
    _digest_contenido,
)

def to_static_policy_view(corpus: ValidatedCandidateCorpus) -> ValidatedStaticPolicyView:
    # Tomar el snapshot una sola vez: la validación del binding y la vista que
    # consume Block 4 comparten exactamente estos objetos, sin una segunda
    # lectura del candidate entre ambos pasos.
    if type(corpus) is not ValidatedCandidateCorpus:
        raise ResolverContractError("adapter sólo acepta ValidatedCandidateCorpus")
    policy_corpus_hash = corpus.policy_corpus_hash
    canonicalizer_identity = corpus.canonicalizer_identity
    bootstrap_bundle_id = corpus.bootstrap_bundle_id
    authority = corpus.authority
    manifest = corpus.manifest
    docs = tuple(corpus.normative_documents)
    binding = corpus._content_binding
    if (
        corpus._loader_seal is not _VALIDATED_CANDIDATE_SEAL
        or binding != _digest_contenido(
            ValidatedCandidateCorpus._contenido_canonico_desde_campos(
                policy_corpus_hash, canonicalizer_identity, bootstrap_bundle_id,
                authority, manifest, docs,
            )
        )
    ):
        raise ResolverContractError("adapter sólo acepta ValidatedCandidateCorpus")
    return ValidatedStaticPolicyView(
        "1.0", "JAX_VALIDATED_STATIC_POLICY_VIEW", policy_corpus_hash,
        canonicalizer_identity, bootstrap_bundle_id,
        authority, tuple(sorted(docs, key=lambda d: d.id)),
    )
