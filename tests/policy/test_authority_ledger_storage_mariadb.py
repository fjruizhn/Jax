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
from policy.authority_ledger.models import AuthorityEventIntent, AuthorityEventType, AuthorityLedgerGenesis
from policy.authority_ledger.replay import event_hash, genesis_hash, verify_authority_ledger
from policy.authority_ledger.service import append_authority_event
from policy.authority_ledger.signatures import encode_public_key, public_key_bytes, public_key_fingerprint
from policy.authority_ledger.storage import MariaDBAuthorityLedgerStore
from policy.authority_ledger.trusted_root import TrustedAuthorityRoot


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
            event = append_authority_event(
                store, root, key,
                AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"),
                event_id="018cc251-f400-7000-8000-000000000001",
                recorded_at_utc=datetime(2026, 10, 6, tzinfo=timezone.utc),
            )
            restored = store.events()
            state = verify_authority_ledger(store.get_genesis(), restored, root)

            assert len(restored) == 1
            assert restored[0] == event
            assert restored[0].event_hash == event_hash(restored[0])
            assert state.checkpoint.sequence == 1
            with connect() as db:
                with db.cursor() as cursor:
                    cursor.execute(
                        "SELECT canonical_intent,canonical_event,evidence_refs "
                        "FROM jax_authority.authority_events WHERE sequence=1"
                    )
                    intent_bytes, event_bytes, evidence_bytes = cursor.fetchone()
            assert intent_bytes == canonical_bytes(event.intent.canonical_projection())
            assert event_bytes == canonical_bytes(restored[0].unsigned_projection() | {
                "signature": restored[0].signature,
                "event_hash": restored[0].event_hash,
            })
            assert evidence_bytes == canonical_bytes(event.intent.evidence_refs)
        finally:
            subprocess.run([*docker, "stop", container], capture_output=True, text=True, check=False)
            subprocess.run(
                ["sudo", "-n", "chown", "-R", f"{os.getuid()}:{os.getgid()}", socket_dir],
                capture_output=True, text=True, check=True,
            )
