"""Block 4 storage integration against an isolated MariaDB 12.3.3 instance."""
from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import pymysql
import pytest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from policy.authority_ledger.canonical import canonical_bytes
from policy.authority_ledger.models import (
    AuthorityEventIntent, AuthorityEventType, AuthorityLedgerGenesis,
    OverlayPayload, OverlayScope, OverlayType, RuleRatificationGrantPayload,
)
from policy.authority_ledger.replay import event_hash, genesis_hash, verify_authority_ledger
from tests.policy.test_authority_ledger_events import append_authority_event
from policy.authority_ledger.service import (append_ratification_from_candidate,
                                             initialize_authority_ledger,
                                             ratification_intent_from_candidate)
from policy.authority_ledger.trusted_checkpoint import TrustedCheckpointStore
from policy.rule_authority.models import (
    RuleDecision, RuleDecisionStatus, RuleEvaluationRequest,
)
from policy.rule_authority.permit import (
    RulePermitConsumptionDraft, RulePermitDraft, _permit_after_kernel_evaluation,
)
from policy.rule_authority.provisioning import provision_application_account as provision_rule_authority_account
from policy.rule_authority.errors import CheckpointInvalido
from policy.rule_authority.trusted_checkpoint import RuleAuditCheckpointStore
from policy.rule_authority.storage import (
    MariaDBRuleDecisionStore, RuleAuthorityCommittedCheckpointError,
    RuleAuthorityCommittedCleanupError,
    RuleAuthorityStorageError,
)
from policy.authority_ledger.signatures import encode_public_key, public_key_bytes, public_key_fingerprint
from policy.authority_ledger.storage import (
    MariaDBAuthorityLedgerStore, MariaDBAuthorityLedgerTransactionOwner,
)
from policy.authority_ledger.trusted_root import TrustedAuthorityRoot
from policy.authority_ledger.errors import AuthorityStateError, TrustedRootMismatchError
from tests.policy._sellos_de_prueba import rule_grant_intent
from policy.authority_resolution.candidate_loader import load_validated_candidate
from tests.policy.catalogo_pin import catalogo_del_pin


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "policy/authority_ledger/migrations/001_authority_ledger.sql"
UPGRADE_MIGRATION = ROOT / "policy/authority_ledger/migrations/002_canonical_intent_not_null.sql"
RULE_AUTHORITY_MIGRATION = ROOT / "policy/rule_authority/migrations/001_rule_authority_kernel.sql"
RULE_AUTHORITY_STEP6_MIGRATION = ROOT / "policy/rule_authority/migrations/002_enable_atomic_permits.sql"
IMAGE = os.environ.get("JAX_AUTHORITY_LEDGER_TEST_MARIADB_IMAGE", "mariadb:12.3.3")


class _AutocommitConnection:
    def __init__(self, enabled=True, server_status=0):
        self.enabled = enabled
        self.server_status = server_status
        self.closed = False
        self.began = False

    def get_autocommit(self):
        return self.enabled

    def begin(self):
        self.began = True

    def close(self):
        self.closed = True


class _RecordingTransactionCursor:
    """Small DB-API double for the pure lock-order contract test."""

    def __init__(self):
        self.statements = []
        self.closed = False
        self.row = None

    def execute(self, statement, params=None):
        self.statements.append((statement, params))

    def fetchone(self):
        return self.row if self.row is not None else ("permit-id", "sha256:" + "a" * 64)

    def close(self):
        self.closed = True


class _RecordingTransactionConnection:
    def __init__(self):
        self.server_status = 0
        self.cursor_instance = _RecordingTransactionCursor()
        self.closed = False
        self.rolled_back = False
        self.committed = False

    def get_autocommit(self):
        return False

    def begin(self):
        self.server_status = 1

    def cursor(self):
        return self.cursor_instance

    def rollback(self):
        self.rolled_back = True
        self.server_status = 0

    def commit(self):
        self.committed = True
        self.server_status = 0

    def close(self):
        self.closed = True


class _CloseFailingTransactionConnection(_RecordingTransactionConnection):
    """DB commit succeeds, but local resource cleanup reports a failure."""

    def close(self):
        self.closed = True
        raise OSError("cierre de conexión inyectado")


def _bootstrapped_rule_authority_checkpoint_store(tmp_path):
    store = RuleAuditCheckpointStore(tmp_path / "rule-authority-checkpoints.jsonl")
    store.bootstrap()
    return store


def _rule_authority_decision_store(checkpoint_store, connect):
    return MariaDBRuleDecisionStore(connect, checkpoint_store=checkpoint_store)


def _wait_for_rule_authority_flock(lock_path, started, acquired):
    """Run in a separate process so success proves the OS flock, not its RLock."""
    with Path(lock_path).open("a+b") as handle:
        started.set()
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            acquired.set()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def test_transaction_owner_rejects_autocommit_connection_before_begin(tmp_path):
    from tests.policy.test_authority_ledger_events import setup_ledger

    ledger, _root, _key = setup_ledger()
    connection = _AutocommitConnection(enabled=True)
    store = MariaDBAuthorityLedgerStore(lambda: pytest.fail("checkpoint_fence no abre conexiones"))
    decision_store = _rule_authority_decision_store(
        _bootstrapped_rule_authority_checkpoint_store(tmp_path), lambda: connection
    )

    with store.checkpoint_fence(ledger._checkpoint_store) as fence:
        with pytest.raises(AuthorityStateError, match="autocommit desactivado"):
            with fence.transaction_owner(decision_store):
                pytest.fail("no debe abrirse una transacción autocommit")

    assert not connection.began
    assert connection.closed


def test_transaction_owner_rejects_preexisting_transaction_before_begin(tmp_path):
    from tests.policy.test_authority_ledger_events import setup_ledger

    ledger, _root, _key = setup_ledger()
    connection = _AutocommitConnection(enabled=False, server_status=1)
    store = MariaDBAuthorityLedgerStore(lambda: pytest.fail("checkpoint_fence no abre conexiones"))
    decision_store = _rule_authority_decision_store(
        _bootstrapped_rule_authority_checkpoint_store(tmp_path), lambda: connection
    )

    with store.checkpoint_fence(ledger._checkpoint_store) as fence:
        with pytest.raises(AuthorityStateError, match="transacción previa"):
            with fence.transaction_owner(decision_store):
                pytest.fail("no debe anidar una transacción")

    assert not connection.began
    assert connection.closed


def test_checkpoint_fence_exposes_verify_for_update_and_rejects_unsealed_owner():
    from tests.policy.test_authority_ledger_events import setup_ledger

    ledger, root, _key = setup_ledger()
    store = MariaDBAuthorityLedgerStore(lambda: pytest.fail("checkpoint_fence no abre conexiones"))

    with store.checkpoint_fence(ledger._checkpoint_store) as fence:
        with pytest.raises(AuthorityStateError, match="owner MariaDB activo y sellado"):
            fence.verify_for_update(object(), root)


def test_transaction_owner_rejects_fake_rule_authority_lock_before_connecting():
    from tests.policy.test_authority_ledger_events import setup_ledger

    ledger, _root, _key = setup_ledger()

    class FakeRuleAuthorityCheckpointStore:
        @contextlib.contextmanager
        def locked(self):
            yield self

    store = MariaDBAuthorityLedgerStore(lambda: pytest.fail("checkpoint_fence no abre conexiones"))
    with store.checkpoint_fence(ledger._checkpoint_store) as fence:
        with pytest.raises(AuthorityStateError, match="decision-store MariaDB"):
            with fence.transaction_owner(FakeRuleAuthorityCheckpointStore()):
                pytest.fail("un lock falso no puede crear transaction owner")


def test_transaction_owner_rejects_separate_concrete_checkpoint_from_decision_wiring(tmp_path):
    from tests.policy.test_authority_ledger_events import setup_ledger

    ledger, _root, _key = setup_ledger()
    canonical = _bootstrapped_rule_authority_checkpoint_store(tmp_path)
    separate = RuleAuditCheckpointStore(tmp_path / "separate-checkpoint.jsonl")
    separate.bootstrap()
    store = MariaDBAuthorityLedgerStore(lambda: pytest.fail("fence no abre conexiones"))

    with store.checkpoint_fence(ledger._checkpoint_store) as fence:
        with pytest.raises(AuthorityStateError, match="decision-store MariaDB"):
            with fence.transaction_owner(separate):
                pytest.fail("un checkpoint separado no puede crear transaction owner")


def test_transaction_creation_is_nested_inside_both_sealed_fences(tmp_path):
    from tests.policy.test_authority_ledger_events import setup_ledger

    ledger, _root, _key = setup_ledger()
    connection = _AutocommitConnection(enabled=True)
    store = MariaDBAuthorityLedgerStore(lambda: pytest.fail("fence no debe abrir conexiones"))
    assert not hasattr(MariaDBAuthorityLedgerTransactionOwner, "begin")
    rule_authority_checkpoint_store = _bootstrapped_rule_authority_checkpoint_store(tmp_path)
    decision_store = _rule_authority_decision_store(rule_authority_checkpoint_store, lambda: connection)

    with store.checkpoint_fence(ledger._checkpoint_store) as fence:
        with pytest.raises(AuthorityStateError, match="autocommit desactivado"):
            with fence.transaction_owner(decision_store):
                pytest.fail("BEGIN no debe aceptar esta conexión")
        assert rule_authority_checkpoint_store._depth == 0
        with pytest.raises(AuthorityStateError, match="decision-store MariaDB"):
            with fence.transaction_owner(object()):
                pytest.fail("falta el lock de Rule Authority")


def test_consumption_locks_permit_before_block4_without_exposing_cursor(tmp_path, monkeypatch):
    """The only pre-verification DB operation is the required permit row lock.

    Rule Authority's process flock is acquired before BEGIN for checkpoint
    durability.  Within the shared MariaDB transaction, consumption must lock
    the permit row before Block 4's verifier locks authority_ledger_head.
    """
    from tests.policy.test_authority_ledger_events import setup_ledger

    ledger, _root, _key = setup_ledger()
    connection = _RecordingTransactionConnection()
    store = MariaDBAuthorityLedgerStore(lambda: pytest.fail("fence no abre conexiones"))
    rule_authority_checkpoint_store = _bootstrapped_rule_authority_checkpoint_store(tmp_path)
    decision_store = _rule_authority_decision_store(rule_authority_checkpoint_store, lambda: connection)
    request = _cross_schema_request("0199f8a1-8c00-7000-8000-0000000009a1")
    _decision, permit_draft = _cross_schema_permit(request)
    permit_id = permit_draft.permit_id
    request_hash = permit_draft.request_hash
    from policy.rule_authority.permit import RulePermit, _trusted_permit
    sealed_permit = _trusted_permit(permit_draft)
    monkeypatch.setattr(
        MariaDBRuleDecisionStore, "_decode_permit_row", staticmethod(lambda _row: sealed_permit)
    )

    with store.checkpoint_fence(ledger._checkpoint_store) as fence:
        with fence.transaction_owner(decision_store) as owner:
            assert not hasattr(owner, "cursor")
            with pytest.raises(AuthorityStateError, match="Block 4 previa"):
                owner.lock_rule_authority_audit_head_for_update()

            locked_permit = owner.lock_permit_for_consumption(permit_id, request_hash)
            assert type(locked_permit) is RulePermit
            assert locked_permit._is_store_sealed() is True
            assert locked_permit.projection() == sealed_permit.projection()
            assert connection.cursor_instance.statements == [(
                "SELECT permit_id,request_id,request_hash,decision_status,rule_id,rule_path,"
                "rule_blob_oid,rule_content_hash,policy_revision,policy_tree_oid,"
                "policy_snapshot_hash,ratification_event_id,authority_ledger_checkpoint,"
                "stop_checkpoint,capability_id,capability_version,capability_class,"
                "capability_limits,issued_at_utc,expires_at_utc,permit_hash,canonical_payload,"
                "previous_record_hash,record_hash,audit_sequence "
                "FROM jax_rule_authority.rule_permits "
                "WHERE permit_id=%s AND request_hash=%s FOR UPDATE",
                (permit_id, request_hash),
            )]
            with pytest.raises(AuthorityStateError, match="solo permite un permit"):
                owner.lock_permit_for_consumption(permit_id, request_hash)
            owner.rollback()


def test_transaction_owner_rejects_commit_after_permit_lock_without_atomic_consumption(tmp_path, monkeypatch):
    """A locked permit cannot commit unless the RA store appended its audit record."""
    from tests.policy.test_authority_ledger_events import setup_ledger

    ledger, _root, _key = setup_ledger()
    connection = _RecordingTransactionConnection()
    store = MariaDBAuthorityLedgerStore(lambda: pytest.fail("fence no abre conexiones"))
    checkpoint = _bootstrapped_rule_authority_checkpoint_store(tmp_path)
    decision_store = _rule_authority_decision_store(checkpoint, lambda: connection)
    request = _cross_schema_request("0199f8a1-8c00-7000-8000-0000000009a2")
    _decision, permit_draft = _cross_schema_permit(request)
    from policy.rule_authority.permit import _trusted_permit
    decoded = _trusted_permit(permit_draft)
    monkeypatch.setattr(
        MariaDBRuleDecisionStore, "_decode_permit_row", staticmethod(lambda _row: decoded)
    )

    with store.checkpoint_fence(ledger._checkpoint_store) as fence:
        with fence.transaction_owner(decision_store) as owner:
            owner.lock_permit_for_consumption(permit_draft.permit_id, permit_draft.request_hash)
            with pytest.raises(AuthorityStateError, match="consume_permit"):
                owner.commit()

    assert connection.rolled_back
    assert not connection.committed
    assert connection.closed


def test_locked_permit_is_fully_decoded_and_store_sealed(tmp_path):
    """The transaction owner retains a decoded closed value, never a row/draft."""
    from tests.policy.test_authority_ledger_events import setup_ledger
    from policy.rule_authority.permit import RulePermit, _trusted_permit
    from policy.rule_authority.storage import _permit_record_hash

    ledger, _root, _key = setup_ledger()
    request = _cross_schema_request("0199f8a1-8c00-7000-8000-0000000009a3")
    _decision, permit_draft = _cross_schema_permit(request)
    sealed = _trusted_permit(permit_draft)
    payload = dict(sealed.projection())
    previous_hash = "sha256:" + "d" * 64
    sequence = 2
    connection = _RecordingTransactionConnection()
    connection.cursor_instance.row = (
        sealed.permit_id, sealed.request_id, sealed.request_hash,
        RuleDecisionStatus.PERMIT.value, sealed.rule_id, sealed.rule_path,
        sealed.rule_blob_oid, sealed.rule_content_hash, sealed.policy_revision,
        sealed.policy_tree_oid, sealed.policy_snapshot_hash,
        sealed.ratification_event_id,
        canonical_bytes(sealed.authority_ledger_checkpoint),
        canonical_bytes(sealed.stop_checkpoint), sealed.capability_id,
        sealed.capability_version, sealed.capability_class,
        canonical_bytes(sealed.capability_limits), sealed.issued_at_utc,
        sealed.expires_at_utc, sealed.permit_hash, canonical_bytes(payload),
        previous_hash,
        _permit_record_hash(sequence, sealed.permit_id, previous_hash, payload),
        sequence,
    )
    store = MariaDBAuthorityLedgerStore(lambda: pytest.fail("fence no abre conexiones"))
    checkpoint = _bootstrapped_rule_authority_checkpoint_store(tmp_path)
    decision_store = _rule_authority_decision_store(checkpoint, lambda: connection)

    with store.checkpoint_fence(ledger._checkpoint_store) as fence:
        with fence.transaction_owner(decision_store) as owner:
            locked = owner.lock_permit_for_consumption(
                sealed.permit_id, sealed.request_hash
            )
            assert type(locked) is RulePermit
            assert locked._is_store_sealed() is True
            assert locked.projection() == sealed.projection()
            assert owner._locked_permit is locked
            owner.rollback()


def test_commit_success_with_close_failure_is_recorded_without_losing_commit(tmp_path):
    """Cleanup cannot recast a confirmed DB commit as a rollback outcome."""
    from tests.policy.test_authority_ledger_events import setup_ledger

    ledger, _root, _key = setup_ledger()
    connection = _CloseFailingTransactionConnection()
    store = MariaDBAuthorityLedgerStore(lambda: pytest.fail("fence no abre conexiones"))
    checkpoint = _bootstrapped_rule_authority_checkpoint_store(tmp_path)
    decision_store = _rule_authority_decision_store(checkpoint, lambda: connection)

    with store.checkpoint_fence(ledger._checkpoint_store) as fence:
        with fence.transaction_owner(decision_store) as owner:
            owner.commit()
            assert connection.committed is True
            assert connection.closed is True
            assert owner.active is False
            assert isinstance(owner._post_commit_close_error, OSError)


def test_rule_authority_lock_token_rejects_cross_thread_and_stale_epochs(tmp_path):
    from policy.rule_authority.errors import CheckpointInvalido
    from policy.rule_authority.trusted_checkpoint import require_active_authority_ledger_lock_token

    store = _bootstrapped_rule_authority_checkpoint_store(tmp_path)
    fence = object()
    with store.locked():
        token = store.issue_authority_ledger_lock_token(fence)
        failures = []

        def cross_thread_attempt():
            try:
                store.issue_authority_ledger_lock_token(fence)
            except Exception as exc:  # fail-soft: capturar en el hilo padre permite afirmar que ambas capacidades rechazaron el cruce de hilo
                failures.append(exc)
            try:
                require_active_authority_ledger_lock_token(token, fence)
            except Exception as exc:  # fail-soft: capturar en el hilo padre permite afirmar que ambas capacidades rechazaron el cruce de hilo
                failures.append(exc)

        thread = threading.Thread(target=cross_thread_attempt)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert len(failures) == 2 and all(
            isinstance(failure, CheckpointInvalido) for failure in failures
        )

    with pytest.raises(CheckpointInvalido):
        require_active_authority_ledger_lock_token(token, fence)
    with store.locked():
        with pytest.raises(CheckpointInvalido):
            require_active_authority_ledger_lock_token(token, fence)


def _run(docker: list[str], *args: str, **kwargs):
    return subprocess.run([*docker, *args], check=True, capture_output=True, text=True, **kwargs)


_TRANSIENT_MARIADB_STARTUP_CODES = frozenset((2002, 2003, 2006, 2013))
_TRANSIENT_SOCKET_ERRNOS = frozenset((errno.EAGAIN, errno.ECONNREFUSED, errno.EINTR, errno.ENOENT))


def _is_transient_mariadb_startup_error(error: BaseException) -> bool:
    """Return true only for a socket/server transition during startup."""
    if isinstance(error, OSError):
        return error.errno in _TRANSIENT_SOCKET_ERRNOS
    return (
        isinstance(error, pymysql.err.OperationalError)
        and bool(error.args)
        and type(error.args[0]) is int
        and error.args[0] in _TRANSIENT_MARIADB_STARTUP_CODES
    )


def _connect_after_transient_mariadb_startup_error(connect_once, *, timeout: float, retry_delay: float):
    """Retry only transient socket/server startup failures, bounded by ``timeout``."""
    deadline = time.monotonic() + timeout
    last_error = None
    while True:
        try:
            return connect_once()
        except (OSError, pymysql.MySQLError) as error:
            if not _is_transient_mariadb_startup_error(error):
                raise
            last_error = error
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"MariaDB siguió inaccesible tras {timeout:.0f} s: {last_error}"
            ) from last_error
        time.sleep(retry_delay)


def _container_logs(docker: list[str], container: str, *, tail: int) -> str:
    logs = subprocess.run(
        [*docker, "logs", "--tail", str(tail), container],
        capture_output=True,
        text=True,
        check=False,
    )
    return logs.stdout + logs.stderr


def _final_mariadb_server_ready(logs: str) -> bool:
    """Recognize completed init or the final TCP listener, never port-0 init."""
    if "init process done" in logs:
        return True
    return re.search(
        r"ready for connections\.\s*Version: [^\n]*\bport:\s*'?3306'?(?=\s|$)",
        logs,
    ) is not None


def _wait_until_ready(
    docker: list[str], container: str, socket_dir: str, password: str, timeout: float = 60.0
) -> Path:
    """Wait for the final MariaDB server, after entrypoint initialization has finished."""
    socket_path = Path(socket_dir) / "mysqld.sock"
    deadline = time.monotonic() + timeout
    last_error = None
    initialized = False
    while time.monotonic() < deadline:
        if not initialized:
            initialized = _final_mariadb_server_ready(
                _container_logs(docker, container, tail=200)
            )
        if initialized:
            _run(docker, "exec", "--user=root", container, "chmod", "0755", "/run/mysqld")
            try:
                def connect_once():
                    return pymysql.connect(
                        unix_socket=str(socket_path), user="root", password=password,
                        autocommit=False, charset="utf8mb4", connect_timeout=5,
                    )

                with _connect_after_transient_mariadb_startup_error(
                    connect_once, timeout=min(5.0, max(0.0, deadline - time.monotonic())),
                    retry_delay=0.25,
                ) as probe:
                    with probe.cursor() as cursor:
                        cursor.execute("SELECT 1")
                        if cursor.fetchone() == (1,):
                            return socket_path
            except (OSError, pymysql.MySQLError, AssertionError) as error:
                if not _is_transient_mariadb_startup_error(error.__cause__ or error):
                    raise
                last_error = error
        time.sleep(0.25)
    raise AssertionError(
        f"MariaDB no quedó lista en {timeout:.0f} s (init terminado: {initialized}; "
        f"último error: {last_error}).\n--- docker logs ---\n"
        f"{_container_logs(docker, container, tail=200)[-4000:]}"
    )


def test_transient_mariadb_startup_errors_are_narrowly_classified():
    assert _is_transient_mariadb_startup_error(OSError(errno.ENOENT, "socket absent"))
    assert _is_transient_mariadb_startup_error(
        pymysql.err.OperationalError(2003, "Can't connect to MySQL server")
    )
    assert not _is_transient_mariadb_startup_error(
        pymysql.err.OperationalError(1045, "Access denied")
    )
    assert not _is_transient_mariadb_startup_error(ValueError("bad test setup"))


def test_connect_retries_only_transient_mariadb_startup_error(monkeypatch):
    attempts = 0
    connection = object()

    def connect_once():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise pymysql.err.OperationalError(2003, "socket temporarily unavailable")
        return connection

    monkeypatch.setattr(time, "sleep", lambda _: None)
    assert _connect_after_transient_mariadb_startup_error(
        connect_once, timeout=1, retry_delay=0
    ) is connection
    assert attempts == 2


def test_connect_preserves_permanent_mariadb_error_without_retry(monkeypatch):
    attempts = 0
    failure = pymysql.err.OperationalError(1045, "Access denied")

    def connect_once():
        nonlocal attempts
        attempts += 1
        raise failure

    monkeypatch.setattr(time, "sleep", lambda _: None)
    with pytest.raises(pymysql.err.OperationalError, match="Access denied"):
        _connect_after_transient_mariadb_startup_error(
            connect_once, timeout=1, retry_delay=0
        )
    assert attempts == 1


def test_final_server_detection_ignores_temporary_port_zero_startup() -> None:
    temporary = """mariadbd: ready for connections.
Version: '12.3.3-MariaDB' socket: '/run/mysqld/mysqld.sock' port: 0 mariadb.org
"""
    final_server = """mariadbd: ready for connections.
Version: '12.3.3-MariaDB' socket: '/run/mysqld/mysqld.sock' port: 3306 mariadb.org
"""

    assert not _final_mariadb_server_ready(temporary)
    assert _final_mariadb_server_ready(final_server)
    assert _final_mariadb_server_ready("init process done")


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
            socket_path = _wait_until_ready(docker, container, socket_dir, password)
            connection = _connect_after_transient_mariadb_startup_error(
                lambda: pymysql.connect(
                    unix_socket=str(socket_path), user="root", password=password,
                    autocommit=False, charset="utf8mb4", connect_timeout=5,
                ),
                timeout=10,
                retry_delay=0.25,
            )
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
            checkpoint_directory = tempfile.TemporaryDirectory(prefix="jax-ledger-checkpoint-")
            checkpoint_store = TrustedCheckpointStore(
                Path(checkpoint_directory.name) / "checkpoints.log",
                bootstrap_receipt_path=Path(checkpoint_directory.name) / "bootstrap-receipt.json",
            )
            initialize_authority_ledger(store, genesis, root, checkpoint_store)
            store._checkpoint_store = checkpoint_store
            with pytest.raises(pymysql.err.OperationalError):
                with app_connect() as app:
                    with app.cursor() as cursor:
                        cursor.execute("UPDATE jax_authority.authority_events SET actor_id='actor:tamper' WHERE sequence=1")
                    app.commit()
            now = datetime(2026, 10, 6, tzinfo=timezone.utc)
            candidate = load_validated_candidate(ROOT)
            corpus_intent = ratification_intent_from_candidate(candidate)
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
            # El overlay se emite ANTES de revocar la ratificación: desde el
            # hallazgo del auditor de #377, OVERLAY_ISSUED exige ratificación
            # vigente del corpus objetivo en ese punto del stream.
            intents = (
                AuthorityEventIntent(AuthorityEventType.ACTIVATION_GRANTED, "human:fernando", ratification_event_id=corpus_event_id),
                AuthorityEventIntent(AuthorityEventType.OVERLAY_ISSUED, "human:fernando", overlay=overlay),
                AuthorityEventIntent(AuthorityEventType.OVERLAY_REVOKED, "human:fernando", overlay_id="test-exception"),
                AuthorityEventIntent(AuthorityEventType.RATIFICATION_REVOKED, "human:fernando", ratification_event_id=corpus_event_id),
                AuthorityEventIntent(AuthorityEventType.ACTIVATION_DEACTIVATED, "human:fernando"),
                rule_grant_intent(grant),
                AuthorityEventIntent(
                    AuthorityEventType.RULE_RATIFICATION_REVOKED, "human:fernando",
                    rule_ratification_event_id="018cc251-f400-7000-8000-000000000007",
                ),
            )
            first = append_ratification_from_candidate(
                store, root, key, candidate,
                event_id=corpus_event_id,
                recorded_at_utc=now,
                checkpoint_store=checkpoint_store,
            )
            events = (first,) + tuple(
                append_authority_event(
                    store, root, key, intent,
                    event_id=f"018cc251-f400-7000-8000-{index:012d}",
                    recorded_at_utc=now,
                )
                for index, intent in enumerate(intents, 2)
            )
            restored = store.events()
            state = verify_authority_ledger(
                store.get_genesis(), restored, root, checkpoint_store
            )

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
            socket_path = str(_wait_until_ready(docker, container, socket_dir, password))

            def connect(**kwargs):
                kwargs.setdefault("user", "root")
                kwargs.setdefault("password", password)
                kwargs.setdefault("connect_timeout", 5)
                kwargs.setdefault("autocommit", False)
                return _connect_after_transient_mariadb_startup_error(
                    lambda: pymysql.connect(
                        unix_socket=socket_path, charset="utf8mb4", **kwargs
                    ),
                    timeout=10,
                    retry_delay=0.25,
                )

            yield connect
        finally:
            subprocess.run([*docker, "stop", container], capture_output=True, text=True, check=False)
            chown = subprocess.run(
                ["sudo", "-n", "chown", "-R", f"{os.getuid()}:{os.getgid()}", socket_dir],
                capture_output=True, text=True, check=False,
            )
            if chown.returncode and sys.exc_info()[0] is None:
                raise RuntimeError(f"falló la limpieza de permisos del socket temporal: {chown.stderr.strip()}")


@contextlib.contextmanager
def _initialized_authority_ledger():
    """Yield a real disposable MariaDB ledger and its external checkpoint anchor."""
    with _ephemeral_mariadb() as connect:
        with connect() as admin:
            _apply_migration(admin, MIGRATION)
            _apply_migration(admin, UPGRADE_MIGRATION)

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
        with connect() as admin:
            with admin.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO jax_authority.authority_ledger_genesis "
                    "(singleton,canonical_genesis,genesis_hash) VALUES (1,%s,%s)",
                    (canonical_bytes(genesis.projection()), genesis_hash(genesis)),
                )
            admin.commit()

        def app_connect():
            return connect(database="jax_authority")

        store = MariaDBAuthorityLedgerStore(app_connect)
        with tempfile.TemporaryDirectory(prefix="jax-ledger-checkpoint-") as directory:
            checkpoint_store = TrustedCheckpointStore(
                Path(directory) / "checkpoints.log",
                bootstrap_receipt_path=Path(directory) / "bootstrap-receipt.json",
            )
            initialize_authority_ledger(store, genesis, root, checkpoint_store)
            yield connect, store, root, key, checkpoint_store


def _rule_authority_checkpoint_store(checkpoint_store):
    path = checkpoint_store.path.with_name("rule-authority-checkpoints.log")
    store = RuleAuditCheckpointStore(path)
    if not path.exists():
        store.bootstrap()
    return store


@contextlib.contextmanager
def _mariadb_transaction_owner(connect, fence, checkpoint_store):
    decision_store = _rule_authority_decision_store(
        _rule_authority_checkpoint_store(checkpoint_store),
        lambda **kwargs: connect(database="jax_authority", **kwargs),
    )
    with fence.transaction_owner(decision_store) as owner:
        yield owner


def _cross_schema_request(request_id: str) -> RuleEvaluationRequest:
    return RuleEvaluationRequest(
        request_id=request_id,
        rule_id="rule-one",
        subject="actor:fernando",
        capability="mail.send",
        objective="notify-client",
        resource_id="message:invoice-42",
        arguments={"recipient": "client@example.test", "body": "Invoice ready"},
        catalogo=catalogo_del_pin(),
        quantity=1,
        quantity_unit="mensajes",
    )


def _cross_schema_permit(request: RuleEvaluationRequest) -> tuple[RuleDecision, RulePermitDraft]:
    decision = RuleDecision(
        request_id=request.request_id,
        request_hash=request.request_hash,
        status=RuleDecisionStatus.PERMIT,
        required_rule_id=request.rule_id,
        reason_code=None,
        decided_at_utc=datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc),
    )
    return decision, RulePermitDraft(
        permit_id="0199f8a1-8c00-7000-8000-000000000901",
        request_id=request.request_id,
        request_hash=request.request_hash,
        rule_id=request.rule_id,
        rule_path="policy/faro/rule-one.yaml",
        rule_blob_oid="a" * 40,
        rule_content_hash="sha256:" + "2" * 64,
        policy_revision="b" * 40,
        policy_tree_oid="c" * 40,
        policy_snapshot_hash="sha256:" + "3" * 64,
        ratification_event_id="0199f8a1-8c00-7000-8000-000000000902",
        authority_ledger_checkpoint={"sequence": 0, "head_event_hash": None},
        stop_checkpoint={"version": 0, "fingerprint": "stop:0"},
        capability_id=request.capability,
        capability_version="1",
        capability_class="OBLIGATING",
        capability_limits={"form": "CANTIDAD", "unit": "mensajes", "max": 2},
        issued_at_utc=decision.decided_at_utc,
        expires_at_utc=decision.decided_at_utc + timedelta(minutes=2),
    )


@contextlib.contextmanager
def _initialized_cross_schema_consumption():
    """Both schemas plus the provisioned Rule Authority application principal."""
    with _initialized_authority_ledger() as (connect, ledger_store, root, _key, checkpoint_store):
        app_user = "jax_rule_authority_consume_test"
        app_password = secrets.token_urlsafe(24)
        with connect() as admin:
            _apply_migration(admin, RULE_AUTHORITY_MIGRATION)
            _apply_migration(admin, RULE_AUTHORITY_STEP6_MIGRATION)
            provision_rule_authority_account(admin, app_user, app_password)

        def app_connect():
            return connect(
                user=app_user,
                password=app_password,
                database="jax_rule_authority",
                autocommit=False,
            )

        audit_checkpoint = _rule_authority_checkpoint_store(checkpoint_store)
        yield connect, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect


def _consume_under_verified_fence(ledger_store, checkpoint_store, root, decision_store, draft):
    with ledger_store.checkpoint_fence(checkpoint_store) as fence:
        with fence.transaction_owner(decision_store) as owner:
            owner.lock_permit_for_consumption(draft.permit_id, draft.request_hash)
            assert fence.verify_for_update(owner, root).checkpoint.sequence == 0
            return decision_store.consume_permit(owner, draft)


def test_cross_schema_consumption_uses_provisioned_principal_and_retries_exact_permit():
    with _initialized_cross_schema_consumption() as (
        connect, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
    ):
        decision_store = MariaDBRuleDecisionStore(app_connect, checkpoint_store=audit_checkpoint)
        request = _cross_schema_request("0199f8a1-8c00-7000-8000-000000000903")
        decision, permit_draft = _cross_schema_permit(request)
        permit = decision_store.record_permit(
            request, decision, _permit_after_kernel_evaluation(permit_draft)
        )
        permit_head = audit_checkpoint.head_actual()
        draft = RulePermitConsumptionDraft(
            permit_id=permit.permit_id,
            request_hash=request.request_hash,
            consumed_at_utc=datetime(2026, 10, 10, 12, 1, tzinfo=timezone.utc),
        )

        consumed = _consume_under_verified_fence(
            ledger_store, checkpoint_store, root, decision_store, draft
        )
        retry_draft = RulePermitConsumptionDraft(
            permit_id=permit.permit_id,
            request_hash=request.request_hash,
            consumed_at_utc=datetime(2026, 10, 10, 12, 11, tzinfo=timezone.utc),
        )
        retried = _consume_under_verified_fence(
            ledger_store, checkpoint_store, root, decision_store, retry_draft
        )

        assert retried.projection() == consumed.projection()
        assert audit_checkpoint.confirmar(audit_checkpoint.head_actual()) is True
        with connect(database="jax_rule_authority") as admin:
            with admin.cursor() as cursor:
                cursor.execute(
                    "SELECT audit_sequence,head_hash,head_kind,head_key "
                    "FROM rule_authority_audit_head WHERE singleton=1"
                )
                sequence, head, kind, key = cursor.fetchone()
                assert (sequence, head, kind, key) == (
                    3, audit_checkpoint.head_actual(), "CONSUMPTION", permit.permit_id,
                )
                cursor.execute(
                    "SELECT record_hash,audit_sequence FROM rule_permit_consumptions "
                    "WHERE permit_id=%s",
                    (permit.permit_id,),
                )
                assert cursor.fetchall() == ((audit_checkpoint.head_actual(), 3),)

        with app_connect() as application_connection:
            with application_connection.cursor() as cursor:
                with pytest.raises(pymysql.err.OperationalError, match="UPDATE command denied"):
                    cursor.execute(
                        "UPDATE jax_authority.authority_ledger_head SET sequence=sequence "
                        "WHERE singleton=1"
                    )


def test_cross_schema_double_consumer_is_one_durable_consumption():
    with _initialized_cross_schema_consumption() as (
        connect, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
    ):
        decision_store = MariaDBRuleDecisionStore(app_connect, checkpoint_store=audit_checkpoint)
        request = _cross_schema_request("0199f8a1-8c00-7000-8000-000000000904")
        decision, permit_draft = _cross_schema_permit(request)
        permit = decision_store.record_permit(
            request, decision, _permit_after_kernel_evaluation(permit_draft)
        )
        draft = RulePermitConsumptionDraft(
            permit_id=permit.permit_id,
            request_hash=request.request_hash,
            consumed_at_utc=datetime(2026, 10, 10, 12, 2, tzinfo=timezone.utc),
        )

        with ThreadPoolExecutor(max_workers=2) as pool:
            consumptions = list(pool.map(
                lambda _ignored: _consume_under_verified_fence(
                    ledger_store, checkpoint_store, root, decision_store, draft
                ),
                range(2),
            ))

        assert consumptions[0].projection() == consumptions[1].projection()
        with connect(database="jax_rule_authority") as admin:
            with admin.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM rule_permit_consumptions")
                assert cursor.fetchone()[0] == 1
                cursor.execute("SELECT audit_sequence FROM rule_authority_audit_head WHERE singleton=1")
                assert cursor.fetchone()[0] == 3


def test_cross_schema_consumption_insert_failure_rolls_back_permit_leaf_and_audit_head():
    with _initialized_cross_schema_consumption() as (
        connect, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
    ):
        decision_store = MariaDBRuleDecisionStore(app_connect, checkpoint_store=audit_checkpoint)
        request = _cross_schema_request("0199f8a1-8c00-7000-8000-000000000905")
        decision, permit_draft = _cross_schema_permit(request)
        permit = decision_store.record_permit(
            request, decision, _permit_after_kernel_evaluation(permit_draft)
        )
        permit_head = audit_checkpoint.head_actual()
        with connect(database="jax_rule_authority") as admin:
            with admin.cursor() as cursor:
                cursor.execute(
                    "CREATE TRIGGER fail_cross_schema_consumption BEFORE INSERT ON "
                    "rule_permit_consumptions FOR EACH ROW SIGNAL SQLSTATE '45000' "
                    "SET MESSAGE_TEXT='injected consumption failure'"
                )
            admin.commit()
        draft = RulePermitConsumptionDraft(
            permit_id=permit.permit_id,
            request_hash=request.request_hash,
            consumed_at_utc=datetime(2026, 10, 10, 12, 3, tzinfo=timezone.utc),
        )

        with pytest.raises(RuleAuthorityStorageError, match="consumo RulePermit"):
            _consume_under_verified_fence(
                ledger_store, checkpoint_store, root, decision_store, draft
            )

        with connect(database="jax_rule_authority") as admin:
            with admin.cursor() as cursor:
                cursor.execute("DROP TRIGGER fail_cross_schema_consumption")
                cursor.execute("SELECT COUNT(*) FROM rule_permit_consumptions")
                assert cursor.fetchone()[0] == 0
                cursor.execute("SELECT audit_sequence,head_hash FROM rule_authority_audit_head WHERE singleton=1")
                assert cursor.fetchone() == (2, permit_head)
        assert audit_checkpoint.head_actual() == permit_head


def test_cross_schema_consumption_checkpoint_failure_reports_committed_state(monkeypatch):
    with _initialized_cross_schema_consumption() as (
        connect, ledger_store, root, checkpoint_store, _audit_checkpoint, app_connect,
    ):
        checkpoint = RuleAuditCheckpointStore(
            checkpoint_store.path.with_name("rule-authority-publication-failure.log")
        )
        checkpoint.bootstrap()
        decision_store = MariaDBRuleDecisionStore(app_connect, checkpoint_store=checkpoint)
        request = _cross_schema_request("0199f8a1-8c00-7000-8000-000000000906")
        decision, permit_draft = _cross_schema_permit(request)
        permit = decision_store.record_permit(
            request, decision, _permit_after_kernel_evaluation(permit_draft)
        )
        permit_head = checkpoint.head_actual()
        def fail_publication(_head: str, *, anterior: str) -> None:
            raise CheckpointInvalido("resultado de publicación desconocido")

        # The production fence requires this exact concrete type.  Patch only
        # this instance's publication seam after its normal bootstrap/PERMIT
        # publication, preserving the production type guard.
        monkeypatch.setattr(checkpoint, "publicar", fail_publication)
        draft = RulePermitConsumptionDraft(
            permit_id=permit.permit_id,
            request_hash=request.request_hash,
            consumed_at_utc=datetime(2026, 10, 10, 12, 4, tzinfo=timezone.utc),
        )

        with pytest.raises(RuleAuthorityCommittedCheckpointError, match="confirmado en DB"):
            _consume_under_verified_fence(
                ledger_store, checkpoint_store, root, decision_store, draft
            )

        with connect(database="jax_rule_authority") as admin:
            with admin.cursor() as cursor:
                cursor.execute("SELECT audit_sequence,head_kind,head_key FROM rule_authority_audit_head WHERE singleton=1")
                assert cursor.fetchone() == (3, "CONSUMPTION", permit.permit_id)
                cursor.execute("SELECT COUNT(*) FROM rule_permit_consumptions")
                assert cursor.fetchone()[0] == 1
        assert checkpoint.head_actual() == permit_head


def test_cross_schema_consumption_close_failure_reports_anchored_committed_state():
    """A close error after COMMIT is surfaced only after the anchor advances."""
    class CloseFailsAfterCommit:
        def __init__(self, connection):
            self._connection = connection

        def __getattr__(self, name):
            return getattr(self._connection, name)

        def close(self):
            self._connection.close()
            raise OSError("cierre de conexión inyectado después de COMMIT")

    with _initialized_cross_schema_consumption() as (
        connect, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
    ):
        fail_close = False

        def consumption_connect():
            connection = app_connect()
            return CloseFailsAfterCommit(connection) if fail_close else connection

        decision_store = MariaDBRuleDecisionStore(
            consumption_connect, checkpoint_store=audit_checkpoint
        )
        request = _cross_schema_request("0199f8a1-8c00-7000-8000-000000000907")
        decision, permit_draft = _cross_schema_permit(request)
        permit = decision_store.record_permit(
            request, decision, _permit_after_kernel_evaluation(permit_draft)
        )
        permit_head = audit_checkpoint.head_actual()
        fail_close = True
        draft = RulePermitConsumptionDraft(
            permit_id=permit.permit_id,
            request_hash=request.request_hash,
            consumed_at_utc=datetime(2026, 10, 10, 12, 5, tzinfo=timezone.utc),
        )

        with pytest.raises(RuleAuthorityCommittedCleanupError, match="confirmado y anclado"):
            _consume_under_verified_fence(
                ledger_store, checkpoint_store, root, decision_store, draft
            )

        assert audit_checkpoint.head_actual() != permit_head
        assert audit_checkpoint.confirmar(audit_checkpoint.head_actual()) is True
        with connect(database="jax_rule_authority") as admin:
            with admin.cursor() as cursor:
                cursor.execute("SELECT record_hash FROM rule_permit_consumptions WHERE permit_id=%s", (permit.permit_id,))
                assert cursor.fetchone() == (audit_checkpoint.head_actual(),)


def _wait_for_innodb_lock_wait(connect, timeout=5.0):
    """Wait for MariaDB to report the competing transaction in its lock table."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with connect() as observer:
            with observer.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM information_schema.INNODB_LOCK_WAITS")
                waiting = cursor.fetchone()[0]
        if waiting:
            return
        time.sleep(0.05)
    raise AssertionError("MariaDB no reportó el escritor esperando el lock de head")


@pytest.mark.parametrize("completion", ["commit", "rollback"])
def test_transaction_fence_holds_mariadb_head_lock_until_completion(completion):
    with _initialized_authority_ledger() as (connect, store, root, _key, checkpoint_store):
        writer_started = threading.Event()
        writer_acquired = threading.Event()
        writer_errors = []
        checkpoint_lock_started = threading.Event()
        checkpoint_lock_acquired = threading.Event()

        def competing_block4_writer():
            connection = connect(database="jax_authority")
            try:
                with connection.cursor() as cursor:
                    writer_started.set()
                    cursor.execute(
                        "SELECT sequence FROM jax_authority.authority_ledger_head "
                        "WHERE singleton=1 FOR UPDATE"
                    )
                    writer_acquired.set()
                connection.rollback()
            except Exception as exc:  # fail-soft: guardar el error del worker permite fallar la aserción en el hilo de pytest
                writer_errors.append(exc)
            finally:
                connection.close()

        def competing_checkpoint_reader():
            checkpoint_lock_started.set()
            with checkpoint_store.locked():
                checkpoint_lock_acquired.set()

        thread = threading.Thread(target=competing_block4_writer, daemon=True)
        checkpoint_thread = threading.Thread(target=competing_checkpoint_reader, daemon=True)
        rule_authority_checkpoint_store = _rule_authority_checkpoint_store(checkpoint_store)
        flock_context = multiprocessing.get_context("spawn")
        rule_authority_lock_started = flock_context.Event()
        rule_authority_lock_acquired = flock_context.Event()
        rule_authority_process = flock_context.Process(
            target=_wait_for_rule_authority_flock,
            args=(
                str(rule_authority_checkpoint_store.lock_path),
                rule_authority_lock_started,
                rule_authority_lock_acquired,
            ),
        )
        with store.checkpoint_fence(checkpoint_store) as fence:
            decision_store = _rule_authority_decision_store(
                rule_authority_checkpoint_store,
                lambda **kwargs: connect(database="jax_authority", **kwargs),
            )
            with fence.transaction_owner(decision_store) as owner:
                state = fence.verify_for_update(owner, root)
                assert state.checkpoint.sequence == 0
                assert writer_errors == []
                thread.start()
                checkpoint_thread.start()
                rule_authority_process.start()
                assert writer_started.wait(5), "el escritor MariaDB no inició"
                assert checkpoint_lock_started.wait(5), "el lector del checkpoint no inició"
                assert rule_authority_lock_started.wait(5), "el lector Rule Authority no inició"
                _wait_for_innodb_lock_wait(connect)
                assert not checkpoint_lock_acquired.is_set(), "el checkpoint lock se perdió dentro del contexto"
                assert not rule_authority_lock_acquired.is_set(), "el flock Rule Authority se perdió dentro del contexto"
                if completion == "commit":
                    owner.commit()
                else:
                    owner.rollback()
                assert not checkpoint_lock_acquired.is_set(), "commit/rollback liberó el checkpoint antes del contexto"
                assert not rule_authority_lock_acquired.is_set(), "commit/rollback liberó Rule Authority antes del contexto"
                assert checkpoint_store._lock_depth == 1, "el fence perdió la propiedad del checkpoint tras finalizar DB"
                assert rule_authority_checkpoint_store._depth == 1, "el lock RA se liberó antes del contexto"

        assert writer_acquired.wait(5), "el escritor siguió bloqueado tras commit/rollback"
        assert checkpoint_lock_acquired.wait(5), "el checkpoint lock siguió retenido tras cerrar el contexto"
        assert rule_authority_lock_acquired.wait(5), "el flock Rule Authority siguió retenido tras cerrar el contexto"
        thread.join(timeout=5)
        checkpoint_thread.join(timeout=5)
        rule_authority_process.join(timeout=5)
        assert not thread.is_alive()
        assert not checkpoint_thread.is_alive()
        assert not rule_authority_process.is_alive()
        assert rule_authority_process.exitcode == 0
        assert writer_errors == []


def test_transaction_fence_rejects_trusted_root_mismatch_on_real_database():
    with _initialized_authority_ledger() as (connect, store, root, _key, checkpoint_store):
        wrong_root = TrustedAuthorityRoot(
            root.schema_version, root.kind, root.ledger_identity, root.genesis_hash,
            "different-key-id", root.constitutional_public_key_fingerprint,
        )
        checkpoint_store.path.write_bytes(b"invalid checkpoint log\n")
        with store.checkpoint_fence(checkpoint_store) as fence:
            with _mariadb_transaction_owner(connect, fence, checkpoint_store) as owner:
                with pytest.raises(TrustedRootMismatchError):
                    fence.verify_for_update(owner, wrong_root)


def test_checkpoint_fence_reuses_owner_connection_and_rejects_real_autocommit():
    with _initialized_authority_ledger() as (connect, store, root, _key, checkpoint_store):
        opened = []

        def counted_connect(**kwargs):
            opened.append(True)
            return connect(**kwargs)

        store = MariaDBAuthorityLedgerStore(counted_connect)
        with store.checkpoint_fence(checkpoint_store) as fence:
            with _mariadb_transaction_owner(counted_connect, fence, checkpoint_store) as owner:
                with pytest.raises(AuthorityStateError, match="cursor requiere"):
                    owner.cursor
                with pytest.raises(AuthorityStateError, match="owner MariaDB activo y sellado"):
                    fence.verify_for_update(object(), root)
                assert fence.verify_for_update(owner, root).checkpoint.sequence == 0
                assert len(opened) == 1, "verifier abrió una segunda conexión"
                owner.rollback()

        with store.checkpoint_fence(checkpoint_store) as fence:
            with pytest.raises(AuthorityStateError, match="autocommit desactivado"):
                decision_store = _rule_authority_decision_store(
                    _rule_authority_checkpoint_store(checkpoint_store),
                    lambda: connect(database="jax_authority", autocommit=True),
                )
                with fence.transaction_owner(decision_store):
                    pytest.fail("no debe aceptarse autocommit")


def test_checkpoint_fence_rolls_back_unfinalized_owner_before_releasing_flock():
    with _initialized_authority_ledger() as (connect, store, root, _key, checkpoint_store):
        owner = None
        try:
            with pytest.raises(AuthorityStateError, match="antes de salir del lock Rule Authority"):
                with store.checkpoint_fence(checkpoint_store) as fence:
                    decision_store = _rule_authority_decision_store(
                        _rule_authority_checkpoint_store(checkpoint_store),
                        lambda: connect(database="jax_authority"),
                    )
                    with fence.transaction_owner(decision_store) as owner:
                        assert fence.verify_for_update(owner, root).checkpoint.sequence == 0
            assert owner is not None and not owner.active
        finally:
            if owner is not None:
                owner.close()


def test_transaction_fence_rejects_db_head_not_matching_replay_on_real_database():
    with _initialized_authority_ledger() as (connect, store, root, _key, checkpoint_store):
        with connect(database="jax_authority") as admin:
            with admin.cursor() as cursor:
                cursor.execute(
                    "UPDATE jax_authority.authority_ledger_head "
                    "SET sequence=1,head_event_id=%s,head_event_hash=%s WHERE singleton=1",
                    ("00000000-0000-7000-8000-000000000001", "sha256:" + "a" * 64),
                )
            admin.commit()
        with store.checkpoint_fence(checkpoint_store) as fence:
            with _mariadb_transaction_owner(connect, fence, checkpoint_store) as owner:
                with pytest.raises(AuthorityStateError, match="head no coincide"):
                    fence.verify_for_update(owner, root)


def test_checkpoint_fence_rejects_noncanonical_genesis_bytes():
    with _initialized_authority_ledger() as (connect, store, root, _key, checkpoint_store):
        with connect() as admin:
            with admin.cursor() as cursor:
                cursor.execute("DROP TRIGGER jax_authority.no_update_genesis")
                cursor.execute("SELECT canonical_genesis FROM jax_authority.authority_ledger_genesis WHERE singleton=1")
                canonical_genesis = cursor.fetchone()[0]
                noncanonical = json.dumps(json.loads(canonical_genesis), indent=2).encode("utf-8")
                cursor.execute(
                    "UPDATE jax_authority.authority_ledger_genesis SET canonical_genesis=%s WHERE singleton=1",
                    (noncanonical,),
                )
            admin.commit()
        with store.checkpoint_fence(checkpoint_store) as fence:
            with _mariadb_transaction_owner(connect, fence, checkpoint_store) as owner:
                with pytest.raises(AuthorityStateError, match="canonical_genesis"):
                    fence.verify_for_update(owner, root)


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
