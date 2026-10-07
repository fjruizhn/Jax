"""Ataques de falsificación del sello y la proyección congelada (falla 1 del auditor r3).

El sello de ratificación debe ser inalcanzable por las vías públicas de
construcción: `dataclasses.replace` sobre un intent legítimo NO puede
transportarlo, y la proyección congelada no puede mutarse después de sellada.
La frontera que firma (`append_authority_event`) vuelve a derivar la
coherencia hash/proyección antes de firmar. Cada test aquí mata un mutante
del arreglo (ver entrega para la tabla mutante→prueba→aserción).
"""
import dataclasses
from collections.abc import Mapping

import pytest

from policy.authority_ledger.canonical import plain
from policy.authority_ledger.errors import AuthorityStateError
from policy.authority_ledger.models import AuthorityEventIntent, RuleRatificationGrantPayload
from tests.policy.test_authority_ledger_events import append_authority_event
from tests.policy.test_authority_ledger_events import ratification_intent, setup_ledger
from tests.policy._sellos_de_prueba import rule_grant_intent
from tests.policy.test_authority_ledger_rule_ratifications import sample_grant


def test_replace_attack_cannot_reuse_the_snapshot_seal():
    """Ataque 1: replace() sobre un intent legítimo conservando el sello.

    En master el sello viajaba en el constructor y `dataclasses.replace`
    lo copiaba tal cual: un corpus arbitrario quedaba firmado. Con el sello
    `init=False`, el intent reconstruido llega sin sello y la frontera que
    firma lo rechaza. (Mutante «volver a init=True»: el replace transporta
    el sello, el append firma y este test explota.)
    """
    store, root, key = setup_ledger()
    base = ratification_intent()
    forged_hash = "sha256:" + "ab" * 32
    forged = dataclasses.replace(
        base,
        policy_corpus_hash=forged_hash,
        static_policy_view_projection={"policy_corpus_hash": forged_hash},
    )
    assert forged._ratification_snapshot_seal is None
    with pytest.raises(AuthorityStateError, match="snapshot sellado"):
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


def test_projection_is_frozen_deep_after_sealing():
    """Ataque 2: mutar `static_policy_view_projection` tras sellar.

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
    with_docs = AuthorityEventIntent._from_validated_snapshot(
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


def test_hash_drift_after_sealing_is_recomputed_and_rejected_at_append():
    """Ataque 3: hash que no coincide con la proyección congelada.

    El constructor ya rechaza pares incoherentes, así que la única vía de
    producir la divergencia es mutar el campo después de sellar. La frontera
    que firma recalcula el hash desde la proyección y lo compara antes de
    firmar. (Mutante «quitar el recálculo»: el append firma y este test
    explota.)
    """
    store, root, key = setup_ledger()
    base = ratification_intent()
    drifted = "sha256:" + "cd" * 32
    assert drifted != base.policy_corpus_hash
    object.__setattr__(base, "policy_corpus_hash", drifted)
    with pytest.raises(AuthorityStateError, match="no coincide con la proyección"):
        append_authority_event(store, root, key, base)
    assert store.events() == ()
