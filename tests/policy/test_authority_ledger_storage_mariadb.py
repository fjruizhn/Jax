"""Block 4 storage integration against an isolated MariaDB 12.3.3 instance."""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone

import pymysql
import pytest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from policy.authority_ledger.canonical import canonical_bytes
from policy.authority_ledger.models import (
    AuthorityEventIntent, AuthorityEventType, AuthorityLedgerGenesis,
    OverlayPayload, OverlayScope, OverlayType, RuleRatificationGrantPayload,
)
from policy.authority_ledger.replay import event_hash, genesis_hash, verify_authority_ledger
from policy.authority_ledger.service import append_authority_event, ratification_intent_from_candidate
from policy.authority_ledger.signatures import encode_public_key, public_key_bytes, public_key_fingerprint
from policy.authority_ledger.storage import MariaDBAuthorityLedgerStore
from policy.authority_ledger.trusted_root import TrustedAuthorityRoot
from tests.policy._sellos_de_prueba import rule_grant_intent
from policy.authority_resolution.candidate_loader import load_validated_candidate


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "policy/authority_ledger/migrations/001_authority_ledger.sql"
UPGRADE_MIGRATION = ROOT / "policy/authority_ledger/migrations/002_canonical_intent_not_null.sql"
IMAGE = os.environ.get("JAX_AUTHORITY_LEDGER_TEST_MARIADB_IMAGE", "mariadb:12.3.3")


def _run(docker: list[str], *args: str, **kwargs):
    return subprocess.run([*docker, *args], check=True, capture_output=True, text=True, **kwargs)


def _apply_migration(connection, path):
    sql = "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )
    with connection.cursor() as cursor:
        if "DELIMITER //" in sql:
            before_triggers, rest = sql.split("DELIMITER //", 1)
            triggers, _ = rest.split("DELIMITER ;", 1)
            for statement in before_triggers.split(";"):
                if statement.strip():
                    cursor.execute(statement)
            for statement in triggers.split("//"):
                if statement.strip():
                    cursor.execute(statement)
        else:
            for statement in sql.split(";"):
                if statement.strip():
                    cursor.execute(statement)
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
                _apply_migration(connection, MIGRATION)
                # Exercise the upgrade path from the former nullable shape.
                with connection.cursor() as cursor:
                    cursor.execute(
                        "ALTER TABLE jax_authority.authority_events "
                        "MODIFY canonical_intent LONGBLOB NULL"
                    )
                connection.commit()
                _apply_migration(connection, UPGRADE_MIGRATION)
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SHOW COLUMNS FROM jax_authority.authority_events "
                        "LIKE 'canonical_intent'"
                    )
                    assert cursor.fetchone()[2] == "NO"
                    cursor.execute("SELECT sequence,head_event_id,head_event_hash FROM jax_authority.authority_ledger_head WHERE singleton=1")
                    assert cursor.fetchone() == (0, None, None)

            def connect():
                return pymysql.connect(
                    unix_socket=str(socket_path), user="root", password=password,
                    database="jax_authority", autocommit=False, charset="utf8mb4",
                )

            app_password = secrets.token_urlsafe(24)
            from policy.authority_ledger.provisioning import provision_application_account
            with connect() as admin:
                provision_application_account(admin, "jax_authority_app_test", app_password)

            trigger_password = secrets.token_urlsafe(24)
            with connect() as admin:
                with admin.cursor() as cursor:
                    cursor.execute(
                        "CREATE USER 'jax_authority_trigger_test'@'localhost' IDENTIFIED BY %s",
                        (trigger_password,),
                    )
                    cursor.execute(
                        "GRANT SELECT,UPDATE,DELETE ON jax_authority.authority_events "
                        "TO 'jax_authority_trigger_test'@'localhost'"
                    )
                admin.commit()

            def app_connect():
                return pymysql.connect(
                    unix_socket=str(socket_path), user="jax_authority_app_test", password=app_password,
                    database="jax_authority", autocommit=False, charset="utf8mb4",
                )

            def trigger_connect():
                return pymysql.connect(
                    unix_socket=str(socket_path), user="jax_authority_trigger_test",
                    password=trigger_password, database="jax_authority",
                    autocommit=False, charset="utf8mb4",
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

            store = MariaDBAuthorityLedgerStore(app_connect)
            with pytest.raises(pymysql.err.OperationalError):
                with app_connect() as app:
                    with app.cursor() as cursor:
                        cursor.execute("UPDATE jax_authority.authority_events SET actor_id='actor:tamper' WHERE sequence=1")
                    app.commit()
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
                rule_grant_intent(grant),
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
            assert state.latest_unrevoked_rule_ratification("send-receipt") is None
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
            with trigger_connect() as privileged:
                with privileged.cursor() as cursor:
                    with pytest.raises(pymysql.MySQLError):
                        cursor.execute(
                            "UPDATE authority_events SET actor_id='actor:tamper' WHERE sequence=1"
                        )
                    privileged.rollback()
                    with pytest.raises(pymysql.MySQLError):
                        cursor.execute("DELETE FROM authority_events WHERE sequence=1")
                    privileged.rollback()
            assert store.events() == events
        finally:
            subprocess.run([*docker, "stop", container], capture_output=True, text=True, check=False)
            chown = subprocess.run(
                ["sudo", "-n", "chown", "-R", f"{os.getuid()}:{os.getgid()}", socket_dir],
                capture_output=True, text=True, check=False,
            )
            if chown.returncode and __import__("sys").exc_info()[0] is None:
                raise RuntimeError(f"falló la limpieza de permisos del socket temporal: {chown.stderr.strip()}")


@contextlib.contextmanager
def _ephemeral_mariadb():
    """Disposable MariaDB 12.3.3 (--network none, Unix socket); yields (connect, root_password)."""
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
            socket_path = str(Path(socket_dir) / "mysqld.sock")

            def connect(**kwargs):
                kwargs.setdefault("user", "root")
                kwargs.setdefault("password", password)
                return pymysql.connect(
                    unix_socket=socket_path, autocommit=False, charset="utf8mb4", **kwargs
                )

            last_error = None
            for _ in range(90):
                try:
                    connect().close()
                    break
                except (OSError, pymysql.MySQLError) as exc:
                    last_error = exc
                    time.sleep(1)
            else:
                raise AssertionError(f"MariaDB socket no disponible: {last_error}")
            yield connect
        finally:
            subprocess.run([*docker, "stop", container], capture_output=True, text=True, check=False)
            chown = subprocess.run(
                ["sudo", "-n", "chown", "-R", f"{os.getuid()}:{os.getgid()}", socket_dir],
                capture_output=True, text=True, check=False,
            )
            if chown.returncode and sys.exc_info()[0] is None:
                raise RuntimeError(f"falló la limpieza de permisos del socket temporal: {chown.stderr.strip()}")


def test_provisioning_revokes_preexisting_privileges_and_grants_exact_contract():
    user = "jax_authority_prov_test"
    with _ephemeral_mariadb() as connect:
        with connect() as admin:
            _apply_migration(admin, MIGRATION)
            with admin.cursor() as cursor:
                cursor.execute(f"CREATE USER '{user}'@'localhost' IDENTIFIED BY 'old-password'")
                cursor.execute(f"GRANT ALL PRIVILEGES ON *.* TO '{user}'@'localhost' WITH GRANT OPTION")
                cursor.execute(f"GRANT ALL PRIVILEGES ON jax_authority.* TO '{user}'@'localhost' WITH GRANT OPTION")
            admin.commit()
        with connect(database="jax_authority") as admin:
            from policy.authority_ledger.provisioning import provision_application_account
            provision_application_account(admin, user, secrets.token_urlsafe(24))
        with connect() as admin:
            with admin.cursor() as cursor:
                cursor.execute(f"SHOW GRANTS FOR '{user}'@'localhost'")
                grants = [row[0] for row in cursor.fetchall()]
    usage = [g for g in grants if g.startswith("GRANT USAGE ON *.* TO ")]
    assert len(usage) == 1 and re.fullmatch(
        rf"GRANT USAGE ON \*\.\* TO `{user}`@`localhost`( IDENTIFIED .*)?", usage[0]
    ), usage
    assert "GRANT OPTION" not in usage[0]
    assert sorted(g for g in grants if g not in usage) == sorted([
        f"GRANT SELECT ON `jax_authority`.`authority_ledger_genesis` TO `{user}`@`localhost`",
        f"GRANT SELECT, INSERT ON `jax_authority`.`authority_events` TO `{user}`@`localhost`",
        f"GRANT SELECT, UPDATE ON `jax_authority`.`authority_ledger_head` TO `{user}`@`localhost`",
    ])


def _insert_event(cursor, sequence, intent):
    cursor.execute(
        "INSERT INTO jax_authority.authority_events (sequence,event_id,event_type,actor_id,"
        "canonical_intent,canonical_event,evidence_refs,previous_event_hash,event_hash,signature,"
        "recorded_at_utc) VALUES (%s,%s,'t','a',%s,'e','r',NULL,%s,'s','2026-10-07 00:00:00')",
        (sequence, f"00000000-0000-0000-0000-{sequence:012d}", intent, f"sha256:{sequence:064d}"),
    )


def test_upgrade_migration_fails_closed_on_null_intent_even_with_permissive_sql_mode():
    with _ephemeral_mariadb() as connect:
        with connect() as db:
            _apply_migration(db, MIGRATION)
            with db.cursor() as cursor:
                cursor.execute("ALTER TABLE jax_authority.authority_events MODIFY canonical_intent LONGBLOB NULL")
                _insert_event(cursor, 1, None)
                cursor.execute("SET SESSION sql_mode=''")
                cursor.execute("SELECT @@SESSION.sql_mode")
                assert cursor.fetchone()[0] == ""
            db.commit()
            with pytest.raises(pymysql.MySQLError, match="integrity review required"):
                _apply_migration(db, UPGRADE_MIGRATION)
            db.rollback()
            with db.cursor() as cursor:
                cursor.execute("SELECT sequence, canonical_intent FROM jax_authority.authority_events")
                assert cursor.fetchall() == ((1, None),)
                cursor.execute("SHOW COLUMNS FROM jax_authority.authority_events LIKE 'canonical_intent'")
                assert cursor.fetchone()[2] == "YES"


def test_upgrade_migration_passes_without_null_intent_and_keeps_rows():
    with _ephemeral_mariadb() as connect:
        with connect() as db:
            _apply_migration(db, MIGRATION)
            with db.cursor() as cursor:
                cursor.execute("ALTER TABLE jax_authority.authority_events MODIFY canonical_intent LONGBLOB NULL")
                _insert_event(cursor, 1, b"intent")
                cursor.execute("SET SESSION sql_mode=''")
            db.commit()
            _apply_migration(db, UPGRADE_MIGRATION)
            with db.cursor() as cursor:
                cursor.execute("SELECT canonical_intent FROM jax_authority.authority_events")
                assert cursor.fetchall() == ((b"intent",),)
                cursor.execute("SHOW COLUMNS FROM jax_authority.authority_events LIKE 'canonical_intent'")
                assert cursor.fetchone()[2] == "NO"
