"""Ataques de falsificación del sello del loader de Block 3 (patrón de #377).

Hallazgo del auditor de #377: ``ValidatedCandidateCorpus._loader_seal`` era
``init=True`` y ``dataclasses.replace`` lo transportaba — un corpus alterado
pasaba por validado. Cierre: sello ``init=False`` (solo lo estampa el loader
en ``_from_validated_snapshot``), contenido congelado profundo (las
colecciones entran como tuple/list y se guardan SIEMPRE como tuple) y un
``_content_binding`` — digest del contenido congelado estampado por el loader
que todo consumidor re-deriva y compara antes de confiar. Cada test mata un
mutante del arreglo (tabla en la entrega).
"""
import dataclasses

import pytest

from policy.authority_ledger.errors import AuthorityStateError
from policy.authority_ledger.service import ratification_intent_from_candidate
from policy.authority_resolution import load_validated_candidate, to_static_policy_view
from policy.authority_resolution.errors import ResolverContractError
from policy.authority_resolution.models import ValidatedCandidateCorpus
from test_authority_resolution_context import ROOT


def test_replace_no_transporta_el_sello_ni_el_binding():
    """Ataque 1: replace() sobre un corpus legítimo conservando el sello.

    En master el sello viajaba en el constructor y replace lo copiaba: un hash
    de corpus arbitrario pasaba por validado. Con el sello init=False, el
    corpus reconstruido llega sin sello Y sin binding. (Mutante «init=True»:
    el sello viaja y la primera aserción explota.)
    """
    corpus = load_validated_candidate(ROOT)
    forged = dataclasses.replace(corpus, policy_corpus_hash="sha256:" + "ab" * 32)
    assert forged._loader_seal is None
    assert forged._content_binding is None
    with pytest.raises(ResolverContractError):
        to_static_policy_view(forged)
    with pytest.raises(AuthorityStateError):
        ratification_intent_from_candidate(forged)


def test_replace_con_contenido_alterado_tampoco_valida():
    """Variante del ataque 1 alterando el contenido y conservando el hash."""
    corpus = load_validated_candidate(ROOT)
    metanormas = corpus.authority.protected_metanorms
    assert metanormas
    ultima = metanormas[-1]
    altered = dataclasses.replace(ultima, statement=ultima.statement + " (alterado)")
    authority = dataclasses.replace(corpus.authority, protected_metanorms=(*metanormas[:-1], altered))
    forged = dataclasses.replace(corpus, authority=authority)
    assert forged._loader_seal is None
    with pytest.raises(ResolverContractError):
        to_static_policy_view(forged)


def test_construccion_directa_no_es_corpus_valido():
    """Ataque 3: construir el corpus a mano no abre ninguna puerta — el objeto
    existe, pero ningún consumidor lo acepta."""
    corpus = load_validated_candidate(ROOT)
    unsealed = ValidatedCandidateCorpus(
        corpus.policy_corpus_hash, corpus.canonicalizer_identity,
        corpus.bootstrap_bundle_id, corpus.authority, corpus.manifest,
        corpus.normative_documents,
    )
    assert unsealed._loader_seal is None
    assert not unsealed._was_loader_validated()
    with pytest.raises(ResolverContractError):
        to_static_policy_view(unsealed)
    with pytest.raises(AuthorityStateError):
        ratification_intent_from_candidate(unsealed)


def _documento_sintetico():
    from policy.authority_resolution.models import FrozenNormativeDocument, FrozenRelationships, FrozenScope
    return FrozenNormativeDocument(
        "doc-alpha", "PRODUCT_POLICY", "PRODUCT_POLICY", "ACTIVE_WHEN_CORPUS_ACTIVE",
        FrozenScope("JAX", ["ALICE"], ("READ",), []),
        FrozenRelationships((), ()),
    )


def test_el_contenido_queda_congelado_profundo():
    """Ataque 2: mutar el contenido tras sellar.

    Las colecciones que entran como list se decuelven como tuple y quedan
    DESACOPLADAS de la lista original: mutar la fuente no cambia el corpus y
    el árbol guardado no tiene nada mutable. (Mutante «sin congelar»: el
    corpus guarda la lista viva y estas aserciones explotan.)
    """
    corpus = load_validated_candidate(ROOT)
    doc = _documento_sintetico()
    docs = [doc]
    members = list(corpus.manifest.members)
    unsealed = ValidatedCandidateCorpus(
        corpus.policy_corpus_hash, corpus.canonicalizer_identity,
        corpus.bootstrap_bundle_id, corpus.authority,
        dataclasses.replace(corpus.manifest, members=members),
        docs,
    )
    assert isinstance(unsealed.normative_documents, tuple)
    assert isinstance(unsealed.manifest.members, tuple)
    assert isinstance(unsealed.normative_documents[0].scope.subjects, tuple)
    docs.clear()
    members.clear()
    assert len(unsealed.normative_documents) == 1
    assert unsealed.manifest.members == corpus.manifest.members
    with pytest.raises(AttributeError):
        unsealed.normative_documents.append(doc)
    with pytest.raises(AttributeError):
        unsealed.manifest.members.append(corpus.manifest.members[0])
    with pytest.raises(AttributeError):
        unsealed.normative_documents[0].id = "otro"


def test_el_consumidor_recalcula_el_digest_del_contenido():
    """Drift del contenido tras sellar: el digest estampado por el loader ya no
    coincide con el re-derivado y todo consumidor niega. (Mutante «sin comparar
    hash»: _was_loader_validated vuelve a mirar solo el sello y explota.)"""
    corpus = load_validated_candidate(ROOT)
    assert corpus._was_loader_validated()
    metanormas = corpus.authority.protected_metanorms
    assert metanormas
    ultima = metanormas[-1]
    drifted = dataclasses.replace(ultima, statement=ultima.statement + " (alterado)")
    authority = dataclasses.replace(corpus.authority, protected_metanorms=(*metanormas[:-1], drifted))
    object.__setattr__(corpus, "authority", authority)
    assert not corpus._was_loader_validated()
    with pytest.raises(ResolverContractError):
        to_static_policy_view(corpus)
    with pytest.raises(AuthorityStateError):
        ratification_intent_from_candidate(corpus)


def test_el_loader_sella_y_liga_hash_y_contenido_del_corpus_real():
    corpus = load_validated_candidate(ROOT)
    assert corpus._was_loader_validated()
    # El binding es sensible al contenido: otro corpus, otro digest.
    otro = dataclasses.replace(corpus, policy_corpus_hash="sha256:" + "cd" * 32)
    assert otro._content_binding is None or otro._content_binding != corpus._content_binding
