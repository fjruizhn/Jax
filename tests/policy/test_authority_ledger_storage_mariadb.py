"""Block 4 storage integration against an isolated MariaDB 12.3.3 instance."""
from __future__ import annotations

import os
from pathlib import Path
import secrets
import shlex
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone

import pymysql

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from policy.authority_ledger.canonical import canonical_bytes
from policy.authority_ledger.models import (
    AuthorityEventIntent, AuthorityEventType, AuthorityLedgerGenesis,
    OverlayPayload, OverlayScope, OverlayType, RuleRatificationGrantPayload,
    _RULE_RATIFICATION_SNAPSHOT_SEAL,
)
from policy.authority_ledger.replay import event_hash, genesis_hash, verify_authority_ledger
from policy.authority_ledger.service import append_authority_event, ratification_intent_from_candidate
from policy.authority_ledger.signatures import encode_public_key, public_key_bytes, public_key_fingerprint
from policy.authority_ledger.storage import MariaDBAuthorityLedgerStore
from policy.authority_ledger.trusted_root import TrustedAuthorityRoot
from policy.authority_resolution.candidate_loader import load_validated_candidate


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "policy/authority_ledger/migrations/001_authority_ledger.sql"
IMAGE = os.environ.get("JAX_AUTHORITY_LEDGER_TEST_MARIADB_IMAGE", "mariadb:12.3.3")


def _run(docker: list[str], *args: str, **kwargs):
    return subprocess.run([*docker, *args], check=True, capture_output=True, text=True, **kwargs)


def _apply_migration(connection):
    sql = "\n".join(
        line for line in MIGRATION.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )
    before_triggers, rest = sql.split("DELIMITER //", 1)
    triggers, _ = rest.split("DELIMITER ;", 1)
    with connection.cursor() as cursor:
        for statement in before_triggers.split(";"):
            if statement.strip():
                cursor.execute(statement)
        for statement in triggers.split("//"):
            if statement.strip():
                cursor.execute(statement)
        cursor.execute(
            "INSERT INTO jax_authority.authority_ledger_head "
            "(singleton,sequence,head_event_id,head_event_hash) VALUES (1,0,NULL,NULL)"
        )
    connection.commit()


def test_sign_insert_read_and_replay_preserve_authority_event():
    docker = shlex.split(os.environ.get("JAX_AUTHORITY_LEDGER_DOCKER_CMD", "docker"))
    password = secrets.token_urlsafe(24)
    container = f"jax-ledger-test-{uuid.uuid4().hex[:10]}"
    with tempfile.TemporaryDirectory(prefix="jax-ledger-socket-") as socket_dir:
        _run(
            docker, "run", "-d", "--rm", "--network", "none", "--name", container,
            "--mount", f"type=bind,source={socket_dir},target=/run/mysqld",
            "-e", f"MARIADB_ROOT_PASSWORD={password}", IMAGE,
        )
        try:
            socket_path = Path(socket_dir) / "mysqld.sock"
            socket_ready = False
            for _ in range(90):
                probe = subprocess.run(
                    [*docker, "exec", container, "test", "-S", "/run/mysqld/mysqld.sock"],
                    capture_output=True, text=True, check=False,
                )
                if probe.returncode == 0:
                    socket_ready = True
                    break
                time.sleep(1)
            assert socket_ready, "MariaDB no creó su socket Unix"
            _run(docker, "exec", "--user=root", container, "chmod", "0755", "/run/mysqld")
            connection = None
            last_error = None
            for _ in range(90):
                try:
                    connection = pymysql.connect(
                        unix_socket=str(socket_path), user="root", password=password,
                        autocommit=False, charset="utf8mb4",
                    )
                    break
                except (OSError, pymysql.MySQLError) as exc:
                    last_error = exc
                    time.sleep(1)
            assert connection is not None, f"MariaDB socket no disponible: {last_error}"
            with connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT VERSION()")
                    assert cursor.fetchone()[0].startswith("12.3.3-")
                _apply_migration(connection)

            def connect():
                return pymysql.connect(
                    unix_socket=str(socket_path), user="root", password=password,
                    database="jax_authority", autocommit=False, charset="utf8mb4",
                )

            key = Ed25519PrivateKey.generate()
            public_key = key.public_key()
            genesis = AuthorityLedgerGenesis(
                "1.0", "JAX_AUTHORITY_LEDGER_GENESIS", "JAX-AUTHORITY-LEDGER/1",
                "human:fernando", "f1.1-test-key", encode_public_key(public_key),
            )
            root = TrustedAuthorityRoot(
                "1.0", "JAX_TRUSTED_AUTHORITY_ROOT", genesis.ledger_identity,
                genesis_hash(genesis), genesis.constitutional_key_id,
                public_key_fingerprint(public_key_bytes(public_key)),
            )
            with connect() as db:
                with db.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO jax_authority.authority_ledger_genesis "
                        "(singleton,canonical_genesis,genesis_hash) VALUES (1,%s,%s)",
                        (canonical_bytes(genesis.projection()), genesis_hash(genesis)),
                    )
                db.commit()

            store = MariaDBAuthorityLedgerStore(connect)
            now = datetime(2026, 10, 6, tzinfo=timezone.utc)
            corpus_intent = ratification_intent_from_candidate(load_validated_candidate(ROOT))
            corpus_event_id = "018cc251-f400-7000-8000-000000000001"
            overlay = OverlayPayload(
                "test-exception", OverlayType.EXCEPTION, corpus_intent.policy_corpus_hash,
                OverlayScope(("ALICE",), ("READ",)), now, None,
                target_rule_ids=("send-receipt",), exception_code="TEST_ONLY",
            )
            grant = RuleRatificationGrantPayload(
                "send-receipt", "policy/faro/send-receipt.yaml", "a" * 40,
                "sha256:" + "b" * 64, "c" * 40, "d" * 40,
                "sha256:" + "e" * 64, now, None,
            )
            intents = (
                corpus_intent,
                AuthorityEventIntent(AuthorityEventType.ACTIVATION_GRANTED, "human:fernando", ratification_event_id=corpus_event_id),
                AuthorityEventIntent(AuthorityEventType.RATIFICATION_REVOKED, "human:fernando", ratification_event_id=corpus_event_id),
                AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"),
                AuthorityEventIntent(AuthorityEventType.OVERLAY_ISSUED, "human:fernando", overlay=overlay),
                AuthorityEventIntent(AuthorityEventType.OVERLAY_REVOKED, "human:fernando", overlay_id="test-exception"),
                AuthorityEventIntent(
                    AuthorityEventType.RULE_RATIFICATION_GRANTED,
                    "human:fernando", rule_ratification=grant,
                    _rule_ratification_snapshot_seal=_RULE_RATIFICATION_SNAPSHOT_SEAL,
                ),
                AuthorityEventIntent(
                    AuthorityEventType.RULE_RATIFICATION_REVOKED, "human:fernando",
                    rule_ratification_event_id="018cc251-f400-7000-8000-000000000007",
                ),
            )
            events = tuple(
                append_authority_event(
                    store, root, key, intent,
                    event_id=f"018cc251-f400-7000-8000-{index:012d}",
                    recorded_at_utc=now,
                )
                for index, intent in enumerate(intents, 1)
            )
            restored = store.events()
            state = verify_authority_ledger(store.get_genesis(), restored, root)

            assert len(restored) == 8
            assert restored == events
            assert all(item.event_hash == event_hash(item) for item in restored)
            assert state.checkpoint.sequence == 8
            assert state.latest_rule_ratifications["send-receipt"].event_id == events[6].event_id
            assert events[6].event_id in state.revoked_rule_ratifications
            with connect() as db:
                with db.cursor() as cursor:
                    cursor.execute(
                        "SELECT canonical_intent,canonical_event,evidence_refs "
                        "FROM jax_authority.authority_events ORDER BY sequence"
                    )
                    rows = cursor.fetchall()
            assert len(rows) == 8
            for stored, event in zip(rows, events, strict=True):
                intent_bytes, event_bytes, evidence_bytes = stored
                assert intent_bytes == canonical_bytes(event.intent.canonical_projection())
                assert event_bytes == canonical_bytes(event.unsigned_projection() | {
                    "signature": event.signature,
                    "event_hash": event.event_hash,
                })
                assert evidence_bytes == canonical_bytes(event.intent.evidence_refs)
        finally:
            subprocess.run([*docker, "stop", container], capture_output=True, text=True, check=False)
            subprocess.run(
                ["sudo", "-n", "chown", "-R", f"{os.getuid()}:{os.getgid()}", socket_dir],
                capture_output=True, text=True, check=True,
            )
