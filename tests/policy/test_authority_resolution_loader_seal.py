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
import copy
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


# --- Ronda 2 (auditoría de c3d471ed): subclases y binding campo por campo ----


def _sin_init_subclass(monkeypatch):
    """Quita el freno ``__init_subclass__`` solo para poder fabricar la subclase
    atacante y comprobar que los CONSUMIDORES niegan por sí solos (defensa en
    profundidad: cada capa se prueba sin la otra)."""
    monkeypatch.setattr(
        ValidatedCandidateCorpus, "__init_subclass__", classmethod(lambda cls, **kw: None)
    )


def _expect_deny(corpus):
    with pytest.raises(ResolverContractError):
        to_static_policy_view(corpus)
    with pytest.raises(AuthorityStateError):
        ratification_intent_from_candidate(corpus)


def test_subclasear_el_corpus_falla_en_la_definicion():
    """MAJOR del auditor: ``class X(ValidatedCandidateCorpus)`` no existe.
    (Mutante «sin __init_subclass__»: la definición pasa y la prueba explota.)"""
    with pytest.raises(TypeError):
        class Falso(ValidatedCandidateCorpus):  # noqa: F841
            def _was_loader_validated(self):
                return True


def test_subclase_que_afirma_estar_validada_es_negada_por_los_consumidores(monkeypatch):
    """Ataque del MAJOR: subclase con ``_was_loader_validated`` -> True y
    contenido/hash que el loader nunca validó. Los consumidores niegan por
    ``type(corpus) is ...`` y por la llamada sin enlazar. (Mutante
    «isinstance»: la subclase pasa y la aserción de negación explota.)"""
    _sin_init_subclass(monkeypatch)

    class Falso(ValidatedCandidateCorpus):
        def _was_loader_validated(self):
            return True

    corpus = load_validated_candidate(ROOT)
    falso = Falso(
        "sha256:" + "ef" * 32, corpus.canonicalizer_identity, corpus.bootstrap_bundle_id,
        corpus.authority, corpus.manifest, corpus.normative_documents,
    )
    assert falso._was_loader_validated() is True
    _expect_deny(falso)


def test_subclase_con_sello_y_binding_legitimos_tambien_es_negada(monkeypatch):
    """Variante: la instancia lleva sello y binding válidos (copia de un corpus
    real con ``__class__`` cambiado) y SOLO el tipo la delata. Con
    ``isinstance`` + llamada sin enlazar pasaría; con ``type(...) is`` no.
    (Mata el mutante «isinstance» aun si la llamada fuera sin enlazar.)"""
    _sin_init_subclass(monkeypatch)

    class Falso(ValidatedCandidateCorpus):
        pass

    corpus = load_validated_candidate(ROOT)
    falso = copy.copy(corpus)
    object.__setattr__(falso, "__class__", Falso)
    assert isinstance(falso, ValidatedCandidateCorpus)
    assert ValidatedCandidateCorpus._was_loader_validated(falso) is True
    _expect_deny(falso)


def test_metodo_sombreado_en_la_instancia_no_engana_a_los_consumidores():
    """La instancia lleva un ``_was_loader_validated`` propio que devuelve True
    sobre contenido alterado: el consumidor llama el método DE LA CLASE sin
    enlazar. (Mutante «método enlazado» ``corpus._was_loader_validated()``: el
    atributo de instancia gana, pasa, y la aserción explota.)"""
    corpus = load_validated_candidate(ROOT)
    object.__setattr__(corpus, "policy_corpus_hash", "sha256:" + "ab" * 32)
    object.__setattr__(corpus, "_was_loader_validated", lambda: True)
    assert corpus._was_loader_validated() is True
    _expect_deny(corpus)


def _alterar_normative_documents(corpus):
    nuevo = (_documento_sintetico(),)
    assert nuevo != corpus.normative_documents
    object.__setattr__(corpus, "normative_documents", nuevo)


def _alterar_policy_corpus_hash(corpus):
    nuevo = "sha256:" + "12" * 32
    assert nuevo != corpus.policy_corpus_hash
    object.__setattr__(corpus, "policy_corpus_hash", nuevo)


def _alterar_manifest(corpus):
    nuevo = dataclasses.replace(corpus.manifest, manifest_id=corpus.manifest.manifest_id + "-x")
    assert nuevo != corpus.manifest
    object.__setattr__(corpus, "manifest", nuevo)


def _alterar_authority(corpus):
    metanormas = corpus.authority.protected_metanorms
    assert metanormas
    ultima = metanormas[-1]
    alterada = dataclasses.replace(ultima, statement=ultima.statement + " (alterado)")
    object.__setattr__(
        corpus, "authority",
        dataclasses.replace(corpus.authority, protected_metanorms=(*metanormas[:-1], alterada)),
    )


def _alterar_canonicalizer_identity(corpus):
    nuevo = "JAX-POLICY-C14N/2"
    assert nuevo != corpus.canonicalizer_identity
    object.__setattr__(corpus, "canonicalizer_identity", nuevo)


def _alterar_bootstrap_bundle_id(corpus):
    # Otro hash con la MISMA forma valida de _HASH: el valor sigue pasando la
    # validacion de forma del __post_init__ original; solo el binding lo delata.
    nuevo = "sha256:" + "34" * 32
    assert nuevo != corpus.bootstrap_bundle_id
    object.__setattr__(corpus, "bootstrap_bundle_id", nuevo)


@pytest.mark.parametrize(
    "alterar",
    [
        _alterar_normative_documents,
        _alterar_policy_corpus_hash,
        _alterar_manifest,
        _alterar_authority,
        _alterar_canonicalizer_identity,
        _alterar_bootstrap_bundle_id,
    ],
    ids=[
        "normative_documents",
        "policy_corpus_hash",
        "manifest",
        "authority",
        "canonicalizer_identity",
        "bootstrap_bundle_id",
    ],
)
def test_cada_campo_del_binding_esta_fijado(alterar):
    """Alterar UN campo tras sellar (``object.__setattr__``) rompe el binding y
    ambos consumidores niegan. Mata los mutantes que quitan esa entrada de
    ``_contenido_canonico``: M4 (normative_documents), M5 (policy_corpus_hash),
    M6 (manifest), el de authority, y los de ``canonicalizer_identity`` y
    ``bootstrap_bundle_id`` (fijados el 2026-10-07, chore/pisos-doc-binding:
    las seis entradas del dict quedan ejercitadas una por una)."""
    corpus = load_validated_candidate(ROOT)
    assert corpus._was_loader_validated()
    alterar(corpus)
    assert not corpus._was_loader_validated()
    _expect_deny(corpus)
