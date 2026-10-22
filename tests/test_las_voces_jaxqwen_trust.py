"""Synthetic adversarial coverage for the host-only jaxqwen trust broker."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("jaxqwen_trust_test", REPO / "projects/las-voces/authority/jaxqwen_trust.py")
assert spec and spec.loader
trust = importlib.util.module_from_spec(spec); sys.modules[spec.name] = trust; spec.loader.exec_module(trust)


def claims():
    return {"service_identity": "jaxqwen", "capability_id": "las_voces.builder.qwen.execute", "mission_id": "a" * 64, "project_id": "las-voces", "task_id": "LV-001", "dispatcher_ack_id": "b" * 64, "correlation_id": "lease-1", "lease_id": "lease-1", "project_hash": "c" * 64, "repository": "/host/repo", "branch": "las-voces/lv-001", "worktree": "/host/missions/lv-001", "allowed_tools": ["write_file"], "expiry": 0, "nonce": "d" * 64, "issuer": trust.ISSUER, "issuer_version": trust.VERSION}


def test_credential_is_bound_consumed_and_secret_free_audit(tmp_path):
    secret = b"s" * 32; broker = trust.JaxQwenTrustBroker(secret, tmp_path / "host-state", current_state=lambda _: True, clock=lambda: 100)
    body = claims(); token = broker.issue(body, ttl_seconds=10)
    expected = {k: v for k, v in body.items() if k in trust.BOUND}
    assert broker.verify_and_consume(token, expected)["mission_id"] == body["mission_id"]
    with pytest.raises(trust.TrustError, match="replay"):
        broker.verify_and_consume(token, expected)
    assert secret.decode() not in (tmp_path / "host-state/trust.ndjson").read_text()
    assert "\"mac\"" not in (tmp_path / "host-state/trust.ndjson").read_text()


@pytest.mark.parametrize("field,value", [("service_identity", "plataforma"), ("capability_id", "shell"), ("mission_id", "e" * 64), ("task_id", "LV-999"), ("dispatcher_ack_id", "f" * 64), ("correlation_id", "other"), ("lease_id", "swap"), ("project_hash", "0" * 64), ("repository", "/other"), ("worktree", "/other/worktree")])
def test_bound_field_swap_is_rejected(tmp_path, field, value):
    broker = trust.JaxQwenTrustBroker(b"s" * 32, tmp_path / "state", current_state=lambda _: True, clock=lambda: 100); body = claims(); token = broker.issue(body)
    expected = {k: v for k, v in claims().items() if k in trust.BOUND}; expected[field] = value
    with pytest.raises(trust.TrustError, match="binding mismatch"):
        broker.verify_and_consume(token, expected)


def test_forgery_expiry_stale_and_revocation_fail_closed(tmp_path):
    now = [100]; current = [True]; broker = trust.JaxQwenTrustBroker(b"s" * 32, tmp_path / "state", current_state=lambda _: current[0], clock=lambda: now[0]); body = claims(); token = broker.issue(body, ttl_seconds=1)
    forged = json.loads(token); forged["body"]["service_identity"] = "jaxqwen"; forged["body"]["task_id"] = "LV-999"
    with pytest.raises(trust.TrustError, match="forged"):
        broker.verify_and_consume(json.dumps(forged), {k: v for k, v in body.items() if k in trust.BOUND})
    now[0] = 102
    with pytest.raises(trust.TrustError, match="expired"):
        broker.verify_and_consume(token, {k: v for k, v in body.items() if k in trust.BOUND})
    now[0] = 100; current[0] = False
    with pytest.raises(trust.TrustError, match="stale"):
        broker.issue({**body, "nonce": "f" * 64})
    current[0] = True; token = broker.issue({**body, "nonce": "0" * 64}); current[0] = False
    with pytest.raises(trust.TrustError, match="stale"):
        broker.verify_and_consume(token, {k: v for k, v in body.items() if k in trust.BOUND})
    current[0] = True; token = broker.issue({**body, "nonce": "e" * 64}); broker.revoke_mission(body["mission_id"], task_id=body["task_id"], dispatcher_ack_id=body["dispatcher_ack_id"], lease_id=body["lease_id"])
    with pytest.raises(trust.TrustError, match="revoked"):
        broker.verify_and_consume(token, {k: v for k, v in body.items() if k in trust.BOUND})


def test_wrong_issuer_and_identity_cannot_issue(tmp_path):
    broker = trust.JaxQwenTrustBroker(b"s" * 32, tmp_path / "state", current_state=lambda _: True)
    with pytest.raises(trust.TrustError): broker.issue({**claims(), "issuer": "attacker"})
    with pytest.raises(trust.TrustError): broker.issue({**claims(), "service_identity": "jacobs"})


def test_expiry_rechecked_after_slow_current_state_and_exact_boundary(tmp_path):
    now = [100]; advance = [False]
    def slow(_):
        if advance[0]: now[0] = 101
        return True
    broker = trust.JaxQwenTrustBroker(b"s" * 32, tmp_path / "state", current_state=slow, clock=lambda: now[0])
    body = claims(); body["nonce"] = "f" * 64; token = broker.issue(body, ttl_seconds=1)
    advance[0] = True
    with pytest.raises(trust.TrustError, match="expired"):
        broker.verify_and_consume(token, {k: v for k, v in body.items() if k in trust.BOUND})
    audit = (tmp_path / "state/trust.ndjson").read_text()
    assert '"event":"EXPIRED_REJECTED"' in audit


def test_broker_state_symlink_permissions_and_deletion_fail_closed(tmp_path):
    real = tmp_path / "real"; real.mkdir(mode=0o700)
    link = tmp_path / "link"; link.symlink_to(real, target_is_directory=True)
    with pytest.raises(trust.TrustError, match="symlink"):
        trust.JaxQwenTrustBroker(b"s" * 32, link, current_state=lambda _: True)
    broker = trust.JaxQwenTrustBroker(b"s" * 32, real, current_state=lambda _: True)
    token = broker.issue(claims()); expected = {k: v for k, v in claims().items() if k in trust.BOUND}
    (real / "trust.ndjson").unlink()
    with pytest.raises(trust.TrustError, match="missing"):
        trust.JaxQwenTrustBroker(b"s" * 32, real, current_state=lambda _: True)


def test_restarted_broker_preserves_nonce_replay_state(tmp_path):
    state = tmp_path / "state"
    body = claims()
    first = trust.JaxQwenTrustBroker(b"s" * 32, state, current_state=lambda _: True, clock=lambda: 100)
    token = first.issue(body)
    expected = {key: value for key, value in body.items() if key in trust.BOUND}
    first.verify_and_consume(token, expected)
    restarted = trust.JaxQwenTrustBroker(b"s" * 32, state, current_state=lambda _: True, clock=lambda: 100)
    with pytest.raises(trust.TrustError, match="replay"):
        restarted.verify_and_consume(token, expected)


def test_nonce_reuse_and_service_revocation_block_issuance(tmp_path):
    broker = trust.JaxQwenTrustBroker(b"s" * 32, tmp_path / "state", current_state=lambda _: True)
    body = claims(); token = broker.issue(body)
    with pytest.raises(trust.TrustError, match="nonce"):
        broker.issue(body)
    broker.revoke_service()
    with pytest.raises(trust.TrustError, match="revoked"):
        broker.issue({**body, "nonce": "f" * 64})
    audit = (tmp_path / "state/trust.ndjson").read_text()
    assert '"event":"REPLAY_REJECTED"' in audit and '"event":"CREDENTIAL_REJECTED"' in audit
    assert token not in audit and "mac" not in audit and "secret" not in audit
