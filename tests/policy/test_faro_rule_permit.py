from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from policy.rule_authority.errors import RuleAuthorityError
from policy.authority_ledger.errors import AuthorityEventValidationError
from policy.rule_authority.permit import RulePermit, RulePermitDraft, _trusted_permit


def _draft(**changes):
    now = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
    values = {
        "permit_id": "0199f8a1-8c00-7000-8000-000000000201",
        "request_id": "0199f8a1-8c00-7000-8000-000000000202",
        "request_hash": "sha256:" + "1" * 64,
        "rule_id": "rule-one",
        "rule_path": "policy/faro/rule-one.yaml",
        "rule_blob_oid": "a" * 40,
        "rule_content_hash": "sha256:" + "2" * 64,
        "policy_revision": "b" * 40,
        "policy_tree_oid": "c" * 40,
        "policy_snapshot_hash": "sha256:" + "3" * 64,
        "ratification_event_id": "0199f8a1-8c00-7000-8000-000000000203",
        "authority_ledger_checkpoint": {"sequence": 5, "head_event_hash": "sha256:" + "4" * 64},
        "stop_checkpoint": {"version": 7, "fingerprint": "stop:7"},
        "capability_id": "mail.send",
        "capability_version": "1",
        "capability_class": "OBLIGATING",
        "capability_limits": {"form": "CANTIDAD", "unit": "mensajes", "max": 3},
        "issued_at_utc": now,
        "expires_at_utc": now + timedelta(minutes=2),
    }
    values.update(changes)
    return RulePermitDraft(**values)


def test_permit_persistido_tiene_proyeccion_canónica_y_hash_de_dominio():
    permit = _trusted_permit(_draft())

    projection = permit.projection()
    assert projection["permit_id"] == "0199f8a1-8c00-7000-8000-000000000201"
    assert projection["authority_ledger_checkpoint"]["sequence"] == 5
    assert projection["permit_hash"].startswith("sha256:")
    assert len(projection["permit_hash"]) == 71
    with pytest.raises(TypeError):
        projection["capability_limits"]["max"] = 999


def test_caller_no_puede_fabricar_ni_mutar_un_permit_confiable():
    draft = _draft()
    with pytest.raises(RuleAuthorityError, match="store"):
        RulePermit(draft)

    trusted = _trusted_permit(draft)
    with pytest.raises((AttributeError, RuleAuthorityError)):
        trusted.request_hash = "sha256:" + "f" * 64


@pytest.mark.parametrize("change", [
    {"rule_path": "policy/faro/../secret.yaml"},
    {"rule_blob_oid": "g" * 40},
    {"policy_revision": "d" * 64},
    {"expires_at_utc": datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)},
    {"capability_limits": {"max": 1.5}},
])
def test_permit_rechaza_proyecciones_invalidas(change):
    with pytest.raises((RuleAuthorityError, AuthorityEventValidationError)):
        _draft(**change)


def test_roundtrip_conserva_hash_y_rechaza_hash_alterado():
    trusted = _trusted_permit(_draft())
    clone = _trusted_permit(_draft())
    assert clone.projection() == trusted.projection()

    tampered = dict(trusted.projection())
    tampered["capability_id"] = "shell.exec"
    with pytest.raises(RuleAuthorityError, match="permit_hash"):
        _trusted_permit(tampered)

    tampered = dict(trusted.projection())
    tampered["request_hash"] = "sha256:" + "f" * 64
    with pytest.raises(RuleAuthorityError, match="permit_hash"):
        _trusted_permit(tampered)
