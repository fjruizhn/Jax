"""Ataques contra intents caller-created y proyecciones congeladas.

`RATIFICATION_GRANTED` sólo firma por
`append_ratification_from_candidate`: el append genérico niega aun un intent
reconstruido por `dataclasses.replace`. La proyección que llega a serialización
sigue congelada en profundidad. Cada test fija ese contrato.
"""
import dataclasses
from collections.abc import Mapping

import pytest

from policy.authority_ledger.canonical import plain
from policy.authority_ledger.errors import AuthorityStateError
from policy.authority_ledger.models import (AuthorityEventIntent, AuthorityEventType,
                                            RuleRatificationGrantPayload)
from tests.policy.test_authority_ledger_events import append_authority_event
from tests.policy.test_authority_ledger_events import ratification_intent, setup_ledger
from tests.policy._sellos_de_prueba import rule_grant_intent
from tests.policy.test_authority_ledger_rule_ratifications import sample_grant


def test_generic_append_rejects_a_replaced_ratification_intent():
    """Una reconstrucción de caller no puede usar la frontera genérica."""
    store, root, key = setup_ledger()
    base = ratification_intent()
    forged_hash = "sha256:" + "ab" * 32
    forged = dataclasses.replace(
        base,
        policy_corpus_hash=forged_hash,
        static_policy_view_projection={"policy_corpus_hash": forged_hash},
    )
    assert forged._ratification_snapshot_seal is None
    with pytest.raises(AuthorityStateError, match="append_ratification_from_candidate"):
        append_authority_event(store, root, key, forged)
    assert store.events() == ()


def test_rule_grant_replace_attack_cannot_reuse_the_faro_seal():
    """El mismo ataque contra el sello Faro individual de #369."""
    store, root, key = setup_ledger()
    base = rule_grant_intent(sample_grant())
    other = RuleRatificationGrantPayload(**(sample_grant().__dict__ | {"rule_id": "other-rule"}))
    forged = dataclasses.replace(base, rule_ratification=other)
    assert forged._rule_ratification_snapshot_seal is None
    with pytest.raises(AuthorityStateError, match="sealed Faro snapshot"):
        append_authority_event(
            store, root, key, forged, event_id="018cc251-f400-7000-8000-000000000040",
        )
    assert store.events() == ()


def test_projection_is_frozen_deep_when_the_intent_is_built():
    """La proyección no puede mutarse después de construir el intent.

    En master la proyección era un dict mutable. Ahora es MappingProxyType
    hasta el fondo: ni la clave del hash ni los dicts anidados se dejan
    escribir. Las aserciones sobre `protected_metanorms[0]` y sobre un corpus
    con documentos son incondicionales. (Mutantes «dict mutable», «congelado
    superficial» y «congelar sin recursión»: alguna asignación deja de lanzar
    TypeError o un tipo deja de ser Mapping/tuple, y este test explota.)
    """
    base = ratification_intent()
    with pytest.raises(TypeError):
        base.static_policy_view_projection["policy_corpus_hash"] = "sha256:" + "cd" * 32
    authority = base.static_policy_view_projection["authority_meta_contract"]
    with pytest.raises(TypeError):
        authority["scope_jurisdiction"] = "OTRO"
    metanorms = authority["protected_metanorms"]
    assert isinstance(metanorms, tuple) and metanorms
    assert isinstance(metanorms[0], Mapping) and not isinstance(metanorms[0], dict)
    with pytest.raises(TypeError):
        metanorms[0]["non_waivable"] = False
    assert isinstance(base.static_policy_view_projection["ordinary_documents"], tuple)

    # Corpus de prueba CON documentos: el corpus real no los tiene, así que
    # sin esto la profundidad de `ordinary_documents` no se ejercita.
    plain_projection = plain(base.static_policy_view_projection)
    plain_projection["ordinary_documents"] = [
        {"id": "doc-a", "relationships": {"supersedes": ["doc-0"]}},
    ]
    with_docs = AuthorityEventIntent(
        AuthorityEventType.RATIFICATION_GRANTED, "human:fernando", (),
        base.policy_corpus_hash, plain_projection,
    )
    documents = with_docs.static_policy_view_projection["ordinary_documents"]
    assert isinstance(documents, tuple) and len(documents) == 1
    assert not isinstance(documents[0], dict)
    with pytest.raises(TypeError):
        documents[0]["id"] = "otro"
    with pytest.raises(TypeError):
        documents[0]["relationships"]["supersedes"] = ()
    assert isinstance(documents[0]["relationships"]["supersedes"], tuple)


def test_generic_append_rejects_hash_drift_in_caller_intent():
    """Ni siquiera un intent congelado y luego alterado llega a firma."""
    store, root, key = setup_ledger()
    base = ratification_intent()
    drifted = "sha256:" + "cd" * 32
    assert drifted != base.policy_corpus_hash
    object.__setattr__(base, "policy_corpus_hash", drifted)
    with pytest.raises(AuthorityStateError, match="append_ratification_from_candidate"):
        append_authority_event(store, root, key, base)
    assert store.events() == ()
