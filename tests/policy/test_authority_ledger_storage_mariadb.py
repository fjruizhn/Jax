"""Block 4 storage integration against an isolated MariaDB 12.3.3 instance."""
from __future__ import annotations

import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor
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
from dataclasses import replace
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
from policy.rule_authority.kernel import RuleAuthorityKernel
from policy.rule_authority.providers import (
    ClaseCapability, ContratoCapability, FormaLimites,
)
from policy.rule_authority.permit import (
    RulePermitConsumptionDraft, RulePermitDraft, _consumption_hash,
    _permit_after_kernel_evaluation,
)
from policy.rule_authority.provisioning import provision_application_account as provision_rule_authority_account
from policy.rule_authority.errors import CheckpointInvalido
from policy.rule_authority.trusted_checkpoint import RuleAuditCheckpointStore
from policy.rule_authority.storage import (
    MariaDBRuleDecisionStore, RuleAuthorityCommittedCheckpointError,
    RuleAuthorityCommittedCleanupError,
    RuleAuthorityDurabilityUnknown, RuleAuthorityStorageError,
    _consumption_record_hash,
)
from policy.authority_ledger.signatures import encode_public_key, public_key_bytes, public_key_fingerprint
from policy.authority_ledger.storage import (
    AuthorityLedgerTransactionPurpose, MariaDBAuthorityLedgerStore,
    MariaDBAuthorityLedgerTransactionOwner,
    _AuthorityLedgerCheckpointFence,
)
from policy.authority_ledger.trusted_root import TrustedAuthorityRoot
from policy.authority_ledger.errors import AuthorityStateError, TrustedRootMismatchError
from policy.authority_ledger.bootstrap import (
    BootstrapInspectionStatus, initialize_staged_authority_ledger,
    inspect_authority_bootstrap, stage_authority_genesis,
)
from tests.policy._sellos_de_prueba import rule_grant_intent
from policy.authority_resolution.candidate_loader import load_validated_candidate
from tests.policy.catalogo_pin import catalogo_del_pin
from tests.policy.proveedores_dobles import (
    ClasificacionFija, PinFijo, RelojDeterminista, StopFijo,
)
from policy.rule_authority.snapshot import TrustedPolicyPin, load_trusted_policy_snapshot


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "policy/authority_ledger/migrations/001_authority_ledger.sql"
UPGRADE_MIGRATION = ROOT / "policy/authority_ledger/migrations/002_canonical_intent_not_null.sql"
RULE_AUTHORITY_MIGRATION = ROOT / "policy/rule_authority/migrations/001_rule_authority_kernel.sql"
RULE_AUTHORITY_STEP6_MIGRATION = ROOT / "policy/rule_authority/migrations/002_enable_atomic_permits.sql"
RULE_AUTHORITY_STEP6_RETRY_MIGRATION = ROOT / "policy/rule_authority/migrations/003_durable_consumption_request_id.sql"
RULE_AUTHORITY_BINDINGS_MIGRATION = ROOT / "policy/rule_authority/migrations/004_permit_policy_capability_identity.sql"
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
                "stop_checkpoint,capability_id,policy_repository_id,capability_contract_id,"
                "capability_version,capability_class,"
                "capability_limits,issued_at_utc,expires_at_utc,permit_hash,canonical_payload,"
                "previous_record_hash,record_hash,audit_sequence "
                "FROM jax_rule_authority.rule_permits "
                "WHERE permit_id=%s AND request_hash=%s FOR UPDATE",
                (permit_id, request_hash),
            )]
            with pytest.raises(AuthorityStateError, match="solo permite un permit"):
                owner.lock_permit_for_consumption(permit_id, request_hash)
            owner.rollback()


def test_evaluation_owner_forbids_permit_lock_and_allows_audit_head_after_block4(
    tmp_path, monkeypatch,
):
    """An evaluation has no permit row, but still serializes on the RA audit head."""
    from tests.policy.test_authority_ledger_events import setup_ledger

    ledger, _root, _key = setup_ledger()
    connection = _RecordingTransactionConnection()
    connection.cursor_instance.row = (0, None)
    store = MariaDBAuthorityLedgerStore(lambda: pytest.fail("fence no abre conexiones"))
    checkpoint = _bootstrapped_rule_authority_checkpoint_store(tmp_path)
    decision_store = _rule_authority_decision_store(checkpoint, lambda: connection)

    with store.checkpoint_fence(ledger._checkpoint_store) as fence:
        with fence.transaction_owner(
            decision_store, purpose=AuthorityLedgerTransactionPurpose.EVALUATION
        ) as owner:
            with pytest.raises(AuthorityStateError, match="EVALUATION no permite lock de permit"):
                owner.lock_permit_for_consumption("permit-id", "sha256:" + "a" * 64)
            monkeypatch.setattr(owner, "_verify_for_update", lambda _store, _root: object())
            assert fence.verify_for_update(owner, object()) is not None
            assert owner.lock_rule_authority_audit_head_for_update() == (0, None)
            owner.rollback()


def test_evaluation_owner_cannot_commit_before_block4_verification(tmp_path):
    """A purpose label cannot turn an unverified DB transaction into an evaluation."""
    from tests.policy.test_authority_ledger_events import setup_ledger

    ledger, _root, _key = setup_ledger()
    connection = _RecordingTransactionConnection()
    store = MariaDBAuthorityLedgerStore(lambda: pytest.fail("fence no abre conexiones"))
    checkpoint = _bootstrapped_rule_authority_checkpoint_store(tmp_path)
    decision_store = _rule_authority_decision_store(checkpoint, lambda: connection)

    with store.checkpoint_fence(ledger._checkpoint_store) as fence:
        with fence.transaction_owner(
            decision_store, purpose=AuthorityLedgerTransactionPurpose.EVALUATION
        ) as owner:
            with pytest.raises(AuthorityStateError, match="EVALUATION solo finaliza mediante record_evaluation_in_transaction"):
                owner.commit()

    assert connection.rolled_back is True
    assert connection.committed is False


def test_consumption_owner_requires_permit_before_block4_then_allows_audit_head(
    tmp_path, monkeypatch,
):
    """Consumption fixes the permit → Block 4 head → RA audit-head row order."""
    from tests.policy.test_authority_ledger_events import setup_ledger
    from policy.rule_authority.permit import _trusted_permit

    ledger, _root, _key = setup_ledger()
    connection = _RecordingTransactionConnection()
    store = MariaDBAuthorityLedgerStore(lambda: pytest.fail("fence no abre conexiones"))
    checkpoint = _bootstrapped_rule_authority_checkpoint_store(tmp_path)
    decision_store = _rule_authority_decision_store(checkpoint, lambda: connection)
    request = _cross_schema_request("0199f8a1-8c00-7000-8000-0000000009a4")
    _decision, permit_draft = _cross_schema_permit(request)
    sealed_permit = _trusted_permit(permit_draft)
    monkeypatch.setattr(
        MariaDBRuleDecisionStore, "_decode_permit_row", staticmethod(lambda _row: sealed_permit)
    )

    with store.checkpoint_fence(ledger._checkpoint_store) as fence:
        with fence.transaction_owner(
            decision_store, purpose=AuthorityLedgerTransactionPurpose.CONSUMPTION
        ) as owner:
            with pytest.raises(AuthorityStateError, match="CONSUMPTION requiere lock de permit"):
                fence.verify_for_update(owner, object())
            owner.lock_permit_for_consumption(permit_draft.permit_id, permit_draft.request_hash)
            monkeypatch.setattr(owner, "_verify_for_update", lambda _store, _root: object())
            assert fence.verify_for_update(owner, object()) is not None
            connection.cursor_instance.row = (0, None)
            assert owner.lock_rule_authority_audit_head_for_update() == (0, None)
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
            sealed.policy_repository_id, sealed.capability_contract_id,
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
                provision_application_account(
                    admin, "jax_authority_app_test", app_password, host="localhost", role="writer",
                )

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

            # Extra facts for narrow tests that prove the account over TCP
            # inside this network-none container. Existing callers still get
            # the same one-argument connection factory.
            connect._docker = docker
            connect._container = container
            connect._socket_dir = socket_dir
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
    with fence.transaction_owner(
        decision_store, purpose=AuthorityLedgerTransactionPurpose.EVALUATION
    ) as owner:
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
        policy_repository_id="jax",
        capability_contract_id="mail.send.contract.v1",
        capability_version="1",
        capability_class="OBLIGATING",
        capability_limits={"form": "CANTIDAD", "unit": "mensajes", "max": 2},
        issued_at_utc=decision.decided_at_utc,
        expires_at_utc=decision.decided_at_utc + timedelta(minutes=2),
    )


@contextlib.contextmanager
def _initialized_cross_schema_consumption(*, include_retry_migration=True):
    """Both schemas plus the provisioned Rule Authority application principal."""
    with _initialized_authority_ledger() as (connect, ledger_store, root, _key, checkpoint_store):
        # Test-only signing material for rule grants emitted through the real
        # Block-4 append boundary by kernel integration tests below.
        ledger_store._test_rule_grant_key = _key
        app_user = "jax_rule_authority_consume_test"
        app_password = secrets.token_urlsafe(24)
        with connect() as admin:
            _apply_migration(admin, RULE_AUTHORITY_MIGRATION)
            _apply_migration(admin, RULE_AUTHORITY_STEP6_MIGRATION)
            if include_retry_migration:
                _apply_migration(admin, RULE_AUTHORITY_STEP6_RETRY_MIGRATION)
            _apply_migration(admin, RULE_AUTHORITY_BINDINGS_MIGRATION)
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


def _kernel_permit_fixture(tmp_path, ledger_store, root, checkpoint_store,
                           audit_checkpoint, app_connect, *, request_id, event_id):
    """Build a real pinned policy repo and its matching Block-4 rule grant.

    The MariaDB race tests must not manufacture a snapshot or authority state:
    the temporary repository is committed, loaded through the production
    snapshot loader, then the signed grant is derived from its sealed rule.
    """
    repo = tmp_path / "pinned-policy"
    faro = repo / "policy" / "faro"
    faro.mkdir(parents=True)
    (faro / "catalogo-topes.json").write_bytes(
        (ROOT / "policy/faro/catalogo-topes.json").read_bytes()
    )
    (faro / "ejemplo.yaml").write_bytes(
        (ROOT / "tests/policy/fixtures/faro_rules/regla-ejemplo.yaml").read_bytes()
    )

    def git(*args):
        completed = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout.strip()

    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "faro fixture")
    commit = git("rev-parse", "HEAD")
    policy_tree_oid = git("rev-parse", "HEAD:policy")
    pin_value = TrustedPolicyPin("jax", commit, policy_tree_oid, "refs/heads/main")
    snapshot = load_trusted_policy_snapshot(repo, pin_value)
    sealed = next(entry for entry in snapshot.reglas if entry.regla.rule_id == "ejemplo-regla")
    now = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
    grant = RuleRatificationGrantPayload(
        sealed.regla.rule_id, sealed.ruta, sealed.blob_oid, sealed.content_hash,
        snapshot.commit, snapshot.policy_tree_oid, snapshot.snapshot_hash, now, None,
    )
    grant_event = append_authority_event(
        ledger_store, root, ledger_store._test_rule_grant_key, rule_grant_intent(grant), event_id=event_id,
        recorded_at_utc=now, checkpoint_store=checkpoint_store,
    )
    request = RuleEvaluationRequest(
        request_id=request_id, rule_id="ejemplo-regla", subject="actor:ejemplo",
        capability="CAPABILITY_ID", objective="objetivo-ejemplo", resource_id="resource:1",
        arguments={}, catalogo=snapshot.catalogo,
    )
    kernel = RuleAuthorityKernel(
        active_policy_pin_provider=PinFijo(
            pin_value, "refs/heads/main", repository_path=repo.resolve(),
        ),
        authority_ledger_store=ledger_store, trusted_root=root,
        trusted_checkpoint_store=checkpoint_store,
        stop_provider=StopFijo(activo=False, huella="sha256:stop-clear"),
        trusted_clock=RelojDeterminista(now),
        capability_classification_provider=ClasificacionFija({
            "CAPABILITY_ID": ContratoCapability(
                identidad="test.reversible.v1", version="1",
                clase=ClaseCapability.REVERSIBLE, unidad=None, moneda=None,
                forma_limites=FormaLimites.NINGUNA,
            ),
        }),
        rule_authority_store=MariaDBRuleDecisionStore(
            app_connect, checkpoint_store=audit_checkpoint,
        ),
    )
    return kernel, request, grant_event


def _kernel_after_checkpoint_restart(kernel, checkpoint_store, app_connect):
    """Recreate process-owned RA stores/kernel while retaining DB and providers."""
    return RuleAuthorityKernel(
        active_policy_pin_provider=kernel._pin_provider,
        authority_ledger_store=kernel._ledger_store,
        trusted_root=kernel._trusted_root,
        trusted_checkpoint_store=kernel._trusted_checkpoint_store,
        stop_provider=kernel._stop_provider,
        trusted_clock=kernel._clock,
        capability_classification_provider=kernel._classification_provider,
        rule_authority_store=MariaDBRuleDecisionStore(
            app_connect, checkpoint_store=checkpoint_store,
        ),
        ttl_max_seconds=kernel._ttl_max_seconds,
    )


@contextlib.contextmanager
def _fail_checkpoint_directory_fsync_after_replace(monkeypatch, checkpoint):
    """Inject a persistent post-replace directory fsync failure for one log."""
    replaced = False
    original_replace = os.replace
    original_fsync_directory = RuleAuditCheckpointStore._fsync_directory

    def replace(source, destination):
        nonlocal replaced
        original_replace(source, destination)
        if Path(destination) == checkpoint.path:
            replaced = True

    def fsync_directory(path):
        if replaced and Path(path) == checkpoint.path.parent:
            raise OSError("injected persistent post-replace directory fsync failure")
        original_fsync_directory(path)

    with monkeypatch.context() as injected:
        injected.setattr(os, "replace", replace)
        injected.setattr(RuleAuditCheckpointStore, "_fsync_directory", fsync_directory)
        yield


def _consume_under_verified_fence(ledger_store, checkpoint_store, root, decision_store, draft):
    with ledger_store.checkpoint_fence(checkpoint_store) as fence:
        with fence.transaction_owner(decision_store) as owner:
            owner.lock_permit_for_consumption(draft.permit_id, draft.request_hash)
            assert fence.verify_for_update(owner, root).checkpoint.sequence == 0
            return decision_store.consume_permit(owner, draft)


def _record_evaluation_permit(ledger_store, checkpoint_store, root, decision_store,
                              request, decision, draft):
    """Persist test permits only through the same verified Block-4 owner as production."""
    with ledger_store.checkpoint_fence(checkpoint_store) as fence:
        with fence.transaction_owner(
            decision_store, purpose=AuthorityLedgerTransactionPurpose.EVALUATION
        ) as owner:
            fence.verify_for_update(owner, root)
            return decision_store.record_evaluation_in_transaction(
                owner, request, decision, _permit_after_kernel_evaluation(draft)
            )


def test_evaluation_persists_permit_with_block4_verification_in_one_owner(monkeypatch):
    """A permit cannot use the legacy standalone decision-store transaction.

    The same owner that replays Block 4 must append both the decision and the
    permit, so a revocation cannot interleave between verification and commit.
    """
    with _initialized_cross_schema_consumption() as (
        _connect, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
    ):
        decision_store = MariaDBRuleDecisionStore(app_connect, checkpoint_store=audit_checkpoint)
        request = _cross_schema_request("0199f8a1-8c00-7000-8000-000000000921")
        decision, draft = _cross_schema_permit(request)

        with ledger_store.checkpoint_fence(checkpoint_store) as fence:
            with fence.transaction_owner(
                decision_store, purpose=AuthorityLedgerTransactionPurpose.EVALUATION
            ) as owner:
                assert fence.verify_for_update(owner, root).checkpoint.sequence == 0
                persisted_decision, permit = decision_store.record_evaluation_in_transaction(
                    owner, request, decision, _permit_after_kernel_evaluation(draft)
                )

        assert persisted_decision == decision
        assert permit is not None
        assert permit._is_store_sealed() is True

        # The final permit row is the externally anchored head; its previous
        # hash and sequence must bind the decision row. Recovery rejects a
        # broken but self-consistent chain even if the final head is anchored.
        from policy.rule_authority.storage import _permit_record_hash
        for corruption in ("previous", "sequence"):
            with ledger_store.checkpoint_fence(checkpoint_store) as fence:
                with fence.transaction_owner(
                    decision_store, purpose=AuthorityLedgerTransactionPurpose.EVALUATION
                ) as owner:
                    fence.verify_for_update(owner, root)
                    cursor = owner._cursor
                    decision_row = decision_store._select_request(cursor, request.request_id)
                    permit_row = decision_store._select_permit(cursor, request.request_id)
                    corrupted = list(permit_row)
                    if corruption == "previous":
                        corrupted[-3] = "sha256:" + "f" * 64
                    else:
                        corrupted[-1] = decision_row[9] + 2
                    corrupted[-2] = _permit_record_hash(
                        corrupted[-1], permit.permit_id, corrupted[-3], permit.projection(),
                    )
                    with monkeypatch.context() as patch:
                        patch.setattr(
                            decision_store, "_select_permit",
                            lambda _cursor, _request_id: tuple(corrupted),
                        )
                        patch.setattr(
                            decision_store._checkpoint_store, "confirmar", lambda _head: True,
                        )
                        with pytest.raises(RuleAuthorityStorageError, match="enlaza de forma consecutiva"):
                            decision_store.recover_evaluation_in_transaction(owner, request)
                    owner.rollback()

        with ledger_store.checkpoint_fence(checkpoint_store) as fence:
            with fence.transaction_owner(
                decision_store, purpose=AuthorityLedgerTransactionPurpose.EVALUATION
            ) as owner:
                fence.verify_for_update(owner, root)
                decision_row = decision_store._select_request(owner._cursor, request.request_id)
                with monkeypatch.context() as patch:
                    patch.setattr(decision_store._checkpoint_store, "confirmar", lambda _head: False)
                    with pytest.raises(RuleAuthorityStorageError, match="head final de la evaluación"):
                        decision_store._recover_anchored_evaluation(
                            owner._cursor, request.request_id, decision_row, request.catalogo
                        )
                owner.rollback()

        # Recovery and the same-owner idempotent append both accept the exact
        # durable pair whose permit head anchors the decision predecessor.
        with ledger_store.checkpoint_fence(checkpoint_store) as fence:
            with fence.transaction_owner(
                decision_store, purpose=AuthorityLedgerTransactionPurpose.EVALUATION
            ) as owner:
                fence.verify_for_update(owner, root)
                recovered = decision_store.recover_evaluation_in_transaction(owner, request)
                assert recovered[0] == decision
                assert recovered[1] is not None
                assert dict(recovered[1].projection()) == dict(permit.projection())
                repeated = decision_store.record_evaluation_in_transaction(
                    owner, request, *recovered
                )
                assert repeated[0] == recovered[0]
                assert repeated[1] is not None
                assert dict(repeated[1].projection()) == dict(recovered[1].projection())

        with ledger_store.checkpoint_fence(checkpoint_store) as fence:
            with fence.transaction_owner(
                decision_store, purpose=AuthorityLedgerTransactionPurpose.CONSUMPTION
            ) as owner:
                owner.lock_permit_for_consumption(permit.permit_id, request.request_hash)
                fence.verify_for_update(owner, root)
                projection = decision_store.load_consumption_request_in_transaction(
                    owner, request.catalogo
                )
                assert projection == request.canonical_projection()
                owner.rollback()


def test_evaluation_persists_and_recovers_deny_through_shared_owner():
    with _initialized_cross_schema_consumption() as (
        _connect, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
    ):
        decision_store = MariaDBRuleDecisionStore(app_connect, checkpoint_store=audit_checkpoint)
        request = _cross_schema_request("0199f8a1-8c00-7000-8000-000000000931")
        decision = RuleDecision(
            request.request_id, request.request_hash, RuleDecisionStatus.DENY,
            request.rule_id, "STOP_ACTIVE", datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc),
        )

        with ledger_store.checkpoint_fence(checkpoint_store) as fence:
            with fence.transaction_owner(
                decision_store, purpose=AuthorityLedgerTransactionPurpose.EVALUATION
            ) as owner:
                fence.verify_for_update(owner, root)
                stored, permit = decision_store.record_evaluation_in_transaction(
                    owner, request, decision, None
                )

        assert stored == decision
        assert permit is None
        assert decision_store.recover_durable_evaluation(request) == (decision, None)


def test_evaluation_insert_failure_rolls_back_decision_and_audit_head():
    with _initialized_cross_schema_consumption() as (
        connect, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
    ):
        with connect(database="jax_rule_authority") as admin:
            with admin.cursor() as cursor:
                cursor.execute(
                    "CREATE TRIGGER fail_rule_decision_insert "
                    "BEFORE INSERT ON jax_rule_authority.rule_decisions FOR EACH ROW "
                    "SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='injected evaluation failure'"
                )
            admin.commit()
        decision_store = MariaDBRuleDecisionStore(app_connect, checkpoint_store=audit_checkpoint)
        request = _cross_schema_request("0199f8a1-8c00-7000-8000-000000000932")
        decision = RuleDecision(
            request.request_id, request.request_hash, RuleDecisionStatus.DENY,
            request.rule_id, "STOP_ACTIVE", datetime(2026, 10, 10, 12, 1, tzinfo=timezone.utc),
        )

        with pytest.raises(RuleAuthorityStorageError, match="persistir evaluación"):
            with ledger_store.checkpoint_fence(checkpoint_store) as fence:
                with fence.transaction_owner(
                    decision_store, purpose=AuthorityLedgerTransactionPurpose.EVALUATION
                ) as owner:
                    fence.verify_for_update(owner, root)
                    decision_store.record_evaluation_in_transaction(
                        owner, request, decision, None
                    )

        with connect() as admin:
            with admin.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM jax_rule_authority.rule_decisions WHERE request_id=%s",
                    (request.request_id,),
                )
                assert cursor.fetchone()[0] == 0
                cursor.execute(
                    "SELECT audit_sequence,head_hash FROM jax_rule_authority.rule_authority_audit_head "
                    "WHERE singleton=1"
                )
                assert cursor.fetchone() == (0, None)
        from policy.rule_authority.trusted_checkpoint import RULE_AUDIT_GENESIS_HEAD
        assert audit_checkpoint.head_actual() == RULE_AUDIT_GENESIS_HEAD
        assert audit_checkpoint.confirmar(RULE_AUDIT_GENESIS_HEAD) is True
        assert decision_store.get(request) is None


def test_cross_schema_consumption_uses_provisioned_principal_and_retries_exact_permit():
    with _initialized_cross_schema_consumption() as (
        connect, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
    ):
        decision_store = MariaDBRuleDecisionStore(app_connect, checkpoint_store=audit_checkpoint)
        request = _cross_schema_request("0199f8a1-8c00-7000-8000-000000000903")
        decision, permit_draft = _cross_schema_permit(request)
        _persisted_decision, permit = _record_evaluation_permit(
            ledger_store, checkpoint_store, root, decision_store,
            request, decision, permit_draft,
        )
        permit_head = audit_checkpoint.head_actual()
        draft = RulePermitConsumptionDraft(
            permit_id=permit.permit_id,
            consumption_request_id="0199f8a1-8c00-7000-8000-000000000905",
            request_hash=request.request_hash,
            consumed_at_utc=datetime(2026, 10, 10, 12, 1, tzinfo=timezone.utc),
        )

        consumed = _consume_under_verified_fence(
            ledger_store, checkpoint_store, root, decision_store, draft
        )
        with ledger_store.checkpoint_fence(checkpoint_store) as fence:
            with fence.transaction_owner(
                decision_store, purpose=AuthorityLedgerTransactionPurpose.CONSUMPTION
            ) as owner:
                owner.lock_permit_for_consumption(permit.permit_id, request.request_hash)
                fence.verify_for_update(owner, root)
                recovered_after_lock = decision_store.recover_consumption_in_transaction(
                    owner, request.request_hash, draft.consumption_request_id
                )
                assert recovered_after_lock is not None
                assert recovered_after_lock.projection() == consumed.projection()
                owner.rollback()
        retry_draft = RulePermitConsumptionDraft(
            permit_id=permit.permit_id,
            consumption_request_id=draft.consumption_request_id,
            request_hash=request.request_hash,
            consumed_at_utc=datetime(2026, 10, 10, 12, 11, tzinfo=timezone.utc),
        )
        retried = _consume_under_verified_fence(
            ledger_store, checkpoint_store, root, decision_store, retry_draft
        )

        assert retried.projection() == consumed.projection()
        different_request = RulePermitConsumptionDraft(
            permit_id=permit.permit_id,
            consumption_request_id="0199f8a1-8c00-7000-8000-000000000908",
            request_hash=request.request_hash,
            consumed_at_utc=datetime(2026, 10, 10, 12, 12, tzinfo=timezone.utc),
        )
        with pytest.raises(AuthorityStateError, match="consumption_request_id"):
            _consume_under_verified_fence(
                ledger_store, checkpoint_store, root, decision_store, different_request
            )
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


def test_retry_migration_preserves_legacy_consumption_and_rejects_new_identity():
    with _initialized_cross_schema_consumption(include_retry_migration=False) as (
        connect, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
    ):
        decision_store = MariaDBRuleDecisionStore(app_connect, checkpoint_store=audit_checkpoint)
        request = _cross_schema_request("0199f8a1-8c00-7000-8000-000000000920")
        decision, permit_draft = _cross_schema_permit(request)
        _persisted_decision, permit = _record_evaluation_permit(
            ledger_store, checkpoint_store, root, decision_store,
            request, decision, permit_draft,
        )

        # Seed a valid pre-003 row without a retry identity. Its old canonical
        # projection and chain link must remain byte-for-byte unchanged.
        consumed_at = datetime(2026, 10, 10, 12, 20, tzinfo=timezone.utc)
        legacy_projection = {
            "permit_id": permit.permit_id,
            "request_hash": request.request_hash,
            "consumed_at_utc": consumed_at.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        }
        legacy_projection["consumption_hash"] = _consumption_hash(legacy_projection)
        legacy_bytes = canonical_bytes(legacy_projection)
        previous_hash = audit_checkpoint.head_actual()
        record_hash = _consumption_record_hash(
            3, permit.permit_id, previous_hash, legacy_projection,
        )
        with connect(database="jax_rule_authority") as admin:
            with admin.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO rule_permit_consumptions "
                    "(permit_id,request_hash,consumed_at_utc,canonical_payload,consumption_hash,"
                    "previous_record_hash,record_hash,audit_sequence) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,3)",
                    (permit.permit_id, request.request_hash, consumed_at.replace(tzinfo=None),
                     legacy_bytes, legacy_projection["consumption_hash"], previous_hash,
                     record_hash),
                )
                cursor.execute(
                    "UPDATE rule_authority_audit_head SET audit_sequence=3,head_hash=%s,"
                    "head_kind='CONSUMPTION',head_key=%s WHERE singleton=1 AND audit_sequence=2",
                    (record_hash, permit.permit_id),
                )
                assert cursor.rowcount == 1
            admin.commit()
        audit_checkpoint.publicar(record_hash, anterior=previous_hash)
        assert audit_checkpoint.confirmar(record_hash) is True

        # Apply the additive migration after the historical row already exists.
        with connect(database="jax_rule_authority") as admin:
            _apply_migration(admin, RULE_AUTHORITY_STEP6_RETRY_MIGRATION)
        with connect(database="jax_rule_authority") as admin:
            with admin.cursor() as cursor:
                cursor.execute(
                    "SELECT permit_id,request_hash,consumed_at_utc,canonical_payload,"
                    "consumption_hash,previous_record_hash,record_hash,audit_sequence,"
                    "consumption_request_id FROM rule_permit_consumptions WHERE permit_id=%s",
                    (permit.permit_id,),
                )
                row = cursor.fetchone()
        assert row[3] == legacy_bytes
        assert row[4:] == (
            legacy_projection["consumption_hash"], previous_hash, record_hash, 3, None,
        )
        decoded, decoded_record_hash, decoded_sequence = MariaDBRuleDecisionStore._decode_consumption_row(row)
        assert decoded.consumption_request_id is None
        assert canonical_bytes(dict(decoded.projection())) == legacy_bytes
        assert (decoded_record_hash, decoded_sequence) == (record_hash, 3)

        head_before_retry = audit_checkpoint.head_actual()
        retry_with_new_id = RulePermitConsumptionDraft(
            permit_id=permit.permit_id,
            consumption_request_id="0199f8a1-8c00-7000-8000-000000000921",
            request_hash=request.request_hash,
            consumed_at_utc=datetime(2026, 10, 10, 12, 21, tzinfo=timezone.utc),
        )
        with pytest.raises(AuthorityStateError, match="consumption_request_id"):
            _consume_under_verified_fence(
                ledger_store, checkpoint_store, root, decision_store, retry_with_new_id
            )
        assert audit_checkpoint.head_actual() == head_before_retry
        assert audit_checkpoint.confirmar(head_before_retry) is True
        with connect(database="jax_rule_authority") as admin:
            with admin.cursor() as cursor:
                cursor.execute(
                    "SELECT audit_sequence,head_hash,head_kind,head_key "
                    "FROM rule_authority_audit_head WHERE singleton=1"
                )
                assert cursor.fetchone() == (3, record_hash, "CONSUMPTION", permit.permit_id)


def test_cross_schema_double_consumer_is_one_durable_consumption():
    with _initialized_cross_schema_consumption() as (
        connect, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
    ):
        decision_store = MariaDBRuleDecisionStore(app_connect, checkpoint_store=audit_checkpoint)
        request = _cross_schema_request("0199f8a1-8c00-7000-8000-000000000904")
        decision, permit_draft = _cross_schema_permit(request)
        _persisted_decision, permit = _record_evaluation_permit(
            ledger_store, checkpoint_store, root, decision_store,
            request, decision, permit_draft,
        )
        draft = RulePermitConsumptionDraft(
            permit_id=permit.permit_id,
            consumption_request_id="0199f8a1-8c00-7000-8000-000000000906",
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
        _persisted_decision, permit = _record_evaluation_permit(
            ledger_store, checkpoint_store, root, decision_store,
            request, decision, permit_draft,
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
            consumption_request_id="0199f8a1-8c00-7000-8000-000000000909",
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
        _persisted_decision, permit = _record_evaluation_permit(
            ledger_store, checkpoint_store, root, decision_store,
            request, decision, permit_draft,
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
            consumption_request_id="0199f8a1-8c00-7000-8000-000000000910",
            request_hash=request.request_hash,
            consumed_at_utc=datetime(2026, 10, 10, 12, 4, tzinfo=timezone.utc),
        )

        with pytest.raises(RuleAuthorityCommittedCheckpointError, match="confirmado en DB") as failure:
            _consume_under_verified_fence(
                ledger_store, checkpoint_store, root, decision_store, draft
            )
        assert not isinstance(failure.value, RuleAuthorityDurabilityUnknown)

        with connect(database="jax_rule_authority") as admin:
            with admin.cursor() as cursor:
                cursor.execute("SELECT audit_sequence,head_kind,head_key FROM rule_authority_audit_head WHERE singleton=1")
                assert cursor.fetchone() == (3, "CONSUMPTION", permit.permit_id)
                cursor.execute("SELECT COUNT(*) FROM rule_permit_consumptions")
                assert cursor.fetchone()[0] == 1
        assert checkpoint.head_actual() == permit_head

        # A retry can acquire this permit after a competing commit that failed
        # its external publication.  It must reject that current head mismatch
        # before returning the historical consumption row.
        with ledger_store.checkpoint_fence(checkpoint_store) as fence:
            with fence.transaction_owner(
                decision_store, purpose=AuthorityLedgerTransactionPurpose.CONSUMPTION
            ) as owner:
                owner.lock_permit_for_consumption(permit.permit_id, request.request_hash)
                fence.verify_for_update(owner, root)
                with pytest.raises(RuleAuthorityStorageError, match="audit head DB"):
                    decision_store.recover_consumption_in_transaction(
                        owner, request.request_hash, draft.consumption_request_id
                    )
                owner.rollback()


def test_evaluation_checkpoint_failure_keeps_db_row_and_blocks_durable_recovery(monkeypatch):
    with _initialized_cross_schema_consumption() as (
        connect, ledger_store, root, checkpoint_store, _audit_checkpoint, app_connect,
    ):
        checkpoint = RuleAuditCheckpointStore(
            checkpoint_store.path.with_name("rule-authority-evaluation-publication-failure.log")
        )
        checkpoint.bootstrap()
        decision_store = MariaDBRuleDecisionStore(app_connect, checkpoint_store=checkpoint)
        request = _cross_schema_request("0199f8a1-8c00-7000-8000-000000000936")
        decision = RuleDecision(
            request.request_id, request.request_hash, RuleDecisionStatus.DENY,
            request.rule_id, "STOP_ACTIVE", datetime(2026, 10, 10, 12, 6, tzinfo=timezone.utc),
        )

        monkeypatch.setattr(
            checkpoint, "publicar",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                CheckpointInvalido("publicación de evaluación inyectada")
            ),
        )
        with pytest.raises(RuleAuthorityCommittedCheckpointError, match="evaluación confirmada en DB") as failure:
            with ledger_store.checkpoint_fence(checkpoint_store) as fence:
                with fence.transaction_owner(
                    decision_store, purpose=AuthorityLedgerTransactionPurpose.EVALUATION
                ) as owner:
                    fence.verify_for_update(owner, root)
                    decision_store.record_evaluation_in_transaction(owner, request, decision, None)
        assert not isinstance(failure.value, RuleAuthorityDurabilityUnknown)

        with connect(database="jax_rule_authority") as admin:
            with admin.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM rule_decisions WHERE request_id=%s", (request.request_id,)
                )
                assert cursor.fetchone()[0] == 1
                cursor.execute("SELECT audit_sequence FROM rule_authority_audit_head WHERE singleton=1")
                assert cursor.fetchone()[0] == 1
        with pytest.raises(RuleAuthorityStorageError, match="checkpoint externo no coincide"):
            decision_store.recover_durable_evaluation(request)


def test_post_replace_fsync_unknown_fails_closed_across_kernel_restart_and_exact_retry(
        monkeypatch, tmp_path):
    """Committed PERMIT/consumption keys survive; restart cannot recover before fsync."""
    with _initialized_cross_schema_consumption() as (
        connect, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
    ):
        kernel, request, _grant_event = _kernel_permit_fixture(
            tmp_path, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
            request_id="0199f8a1-8c00-7000-8000-000000000970",
            event_id="0199f8a1-8c00-7000-8000-000000000971",
        )

        # Evaluation commits its idempotency key and permit to MariaDB, then
        # os.replace makes the new external head visible. A persistent parent
        # directory fsync failure means this call must expose no PERMIT.
        with _fail_checkpoint_directory_fsync_after_replace(monkeypatch, audit_checkpoint):
            with pytest.raises(RuleAuthorityCommittedCheckpointError) as evaluation_failure:
                kernel.evaluate(request)
            assert isinstance(evaluation_failure.value, RuleAuthorityDurabilityUnknown)

            restarted_checkpoint = RuleAuditCheckpointStore(audit_checkpoint.path)
            restarted_kernel = _kernel_after_checkpoint_restart(
                kernel, restarted_checkpoint, app_connect,
            )
            with pytest.raises(RuleAuthorityDurabilityUnknown, match="DURABILITY_UNKNOWN"):
                restarted_kernel.evaluate(request)

            with connect(database="jax_rule_authority") as admin:
                with admin.cursor() as cursor:
                    cursor.execute(
                        "SELECT COUNT(*) FROM rule_decisions WHERE request_id=%s",
                        (request.request_id,),
                    )
                    assert cursor.fetchone()[0] == 1
                    cursor.execute(
                        "SELECT COUNT(*) FROM rule_permits WHERE request_id=%s",
                        (request.request_id,),
                    )
                    assert cursor.fetchone()[0] == 1

        # The exact retry is recoverable only after the new process fsyncs the
        # checkpoint file and directory, then rereads the unchanged hash chain.
        outcome = restarted_kernel.evaluate(request)
        assert outcome.decision.status is RuleDecisionStatus.PERMIT
        assert outcome.permit is not None

        consumption_request_id = "0199f8a1-8c00-7000-8000-000000000972"
        with _fail_checkpoint_directory_fsync_after_replace(monkeypatch, restarted_checkpoint):
            with pytest.raises(RuleAuthorityCommittedCheckpointError) as consumption_failure:
                restarted_kernel.consume(
                    outcome.permit.permit_id, request.request_hash,
                    consumption_request_id,
                )
            assert isinstance(consumption_failure.value, RuleAuthorityDurabilityUnknown)

            consumption_restart_checkpoint = RuleAuditCheckpointStore(
                restarted_checkpoint.path
            )
            consumption_restart_kernel = _kernel_after_checkpoint_restart(
                restarted_kernel, consumption_restart_checkpoint, app_connect,
            )
            with pytest.raises(RuleAuthorityDurabilityUnknown, match="DURABILITY_UNKNOWN"):
                consumption_restart_kernel.consume(
                    outcome.permit.permit_id, request.request_hash,
                    consumption_request_id,
                )
            with pytest.raises(RuleAuthorityDurabilityUnknown, match="DURABILITY_UNKNOWN"):
                consumption_restart_kernel.consume(
                    outcome.permit.permit_id, request.request_hash,
                    "0199f8a1-8c00-7000-8000-000000000973",
                )

            with connect(database="jax_rule_authority") as admin:
                with admin.cursor() as cursor:
                    cursor.execute(
                        "SELECT COUNT(*),MAX(consumption_request_id) "
                        "FROM rule_permit_consumptions WHERE permit_id=%s",
                        (outcome.permit.permit_id,),
                    )
                    assert cursor.fetchone() == (1, consumption_request_id)

        recovered_consumption = consumption_restart_kernel.consume(
            outcome.permit.permit_id, request.request_hash, consumption_request_id,
        )
        assert recovered_consumption.permit_id == outcome.permit.permit_id
        assert recovered_consumption.consumption_request_id == consumption_request_id
        with pytest.raises(AuthorityStateError, match="consumption_request_id distinto"):
            consumption_restart_kernel.consume(
                outcome.permit.permit_id, request.request_hash,
                "0199f8a1-8c00-7000-8000-000000000973",
            )

        # Model the other crash outcome: the rename was visible before power
        # loss but its directory entry was not durable, so restart sees the
        # previously confirmed file. DB's committed head is ahead; recovery
        # must retain the request key/rows and deny instead of reconstructing
        # or issuing a second PERMIT from the stale external anchor.
        lost_request = replace(
            request, request_id="0199f8a1-8c00-7000-8000-000000000974",
            request_hash=None,
        )
        previously_durable_payload = consumption_restart_checkpoint.path.read_bytes()
        with _fail_checkpoint_directory_fsync_after_replace(
                monkeypatch, consumption_restart_checkpoint):
            with pytest.raises(RuleAuthorityCommittedCheckpointError):
                consumption_restart_kernel.evaluate(lost_request)

        consumption_restart_checkpoint.path.write_bytes(previously_durable_payload)
        crash_restarted_checkpoint = RuleAuditCheckpointStore(
            consumption_restart_checkpoint.path
        )
        crash_restarted_kernel = _kernel_after_checkpoint_restart(
            consumption_restart_kernel, crash_restarted_checkpoint, app_connect,
        )
        with pytest.raises(RuleAuthorityStorageError, match="checkpoint externo no coincide"):
            crash_restarted_kernel.evaluate(lost_request)
        with connect(database="jax_rule_authority") as admin:
            with admin.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM rule_decisions WHERE request_id=%s",
                    (lost_request.request_id,),
                )
                assert cursor.fetchone()[0] == 1
                cursor.execute(
                    "SELECT COUNT(*) FROM rule_permits WHERE request_id=%s",
                    (lost_request.request_id,),
                )
                assert cursor.fetchone()[0] == 1


def test_cancelacion_post_replace_no_emite_permit_y_retry_reconcilia_fsync(
        monkeypatch, tmp_path):
    """Cancellation after rename is typed; exact retry waits for durable reconciliation."""
    with _initialized_cross_schema_consumption() as (
        connect, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
    ):
        kernel, request, _grant_event = _kernel_permit_fixture(
            tmp_path, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
            request_id="0199f8a1-8c00-7000-8000-000000000978",
            event_id="0199f8a1-8c00-7000-8000-000000000979",
        )
        original_replace = os.replace

        def replace_then_cancel(source, destination):
            original_replace(source, destination)
            if Path(destination) == audit_checkpoint.path:
                raise asyncio.CancelledError("injected cancellation after checkpoint replace")

        with monkeypatch.context() as cancelled:
            cancelled.setattr(os, "replace", replace_then_cancel)
            with pytest.raises(RuleAuthorityDurabilityUnknown, match="DURABILITY_UNKNOWN"):
                kernel.evaluate(request)

        with connect(database="jax_rule_authority") as admin:
            with admin.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM rule_decisions WHERE request_id=%s",
                    (request.request_id,),
                )
                assert cursor.fetchone()[0] == 1
                cursor.execute(
                    "SELECT COUNT(*) FROM rule_permits WHERE request_id=%s",
                    (request.request_id,),
                )
                assert cursor.fetchone()[0] == 1

        original_fsync_directory = RuleAuditCheckpointStore._fsync_directory

        def fail_directory_fsync(path):
            if Path(path) == audit_checkpoint.path.parent:
                raise OSError("injected persistent retry directory fsync failure")
            original_fsync_directory(path)

        with monkeypatch.context() as failed_retry_fsync:
            failed_retry_fsync.setattr(
                RuleAuditCheckpointStore, "_fsync_directory", fail_directory_fsync,
            )
            with pytest.raises(RuleAuthorityDurabilityUnknown, match="DURABILITY_UNKNOWN"):
                kernel.evaluate(request)

        # The exact committed result becomes visible only after retry performs
        # file and parent-directory fsync and validates the unchanged log.
        outcome = kernel.evaluate(request)
        assert outcome.decision.status is RuleDecisionStatus.PERMIT
        assert outcome.permit is not None


def test_checkpoint_lost_after_consume_crash_retains_original_attempt_and_denies_retries(
        monkeypatch, tmp_path):
    """If restart sees the prior head, neither retry ID can invent a recovery."""
    with _initialized_cross_schema_consumption() as (
        connect, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
    ):
        kernel, request, _grant_event = _kernel_permit_fixture(
            tmp_path, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
            request_id="0199f8a1-8c00-7000-8000-000000000975",
            event_id="0199f8a1-8c00-7000-8000-000000000976",
        )
        outcome = kernel.evaluate(request)
        assert outcome.decision.status is RuleDecisionStatus.PERMIT
        assert outcome.permit is not None
        consumption_request_id = "0199f8a1-8c00-7000-8000-000000000977"
        previous_payload = audit_checkpoint.path.read_bytes()

        with _fail_checkpoint_directory_fsync_after_replace(monkeypatch, audit_checkpoint):
            with pytest.raises(RuleAuthorityCommittedCheckpointError) as failure:
                kernel.consume(
                    outcome.permit.permit_id, request.request_hash,
                    consumption_request_id,
                )
            assert isinstance(failure.value, RuleAuthorityDurabilityUnknown)

        # Simulate power loss discarding the unconfirmed directory entry.
        audit_checkpoint.path.write_bytes(previous_payload)
        restarted_checkpoint = RuleAuditCheckpointStore(audit_checkpoint.path)
        restarted_kernel = _kernel_after_checkpoint_restart(
            kernel, restarted_checkpoint, app_connect,
        )
        for retry_id in (
            consumption_request_id,
            "0199f8a1-8c00-7000-8000-000000000978",
        ):
            with pytest.raises(RuleAuthorityStorageError, match="checkpoint externo no coincide"):
                restarted_kernel.consume(
                    outcome.permit.permit_id, request.request_hash, retry_id,
                )

        with connect(database="jax_rule_authority") as admin:
            with admin.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*),MAX(consumption_request_id) "
                    "FROM rule_permit_consumptions WHERE permit_id=%s",
                    (outcome.permit.permit_id,),
                )
                assert cursor.fetchone() == (1, consumption_request_id)


def test_kernel_evaluate_stop_deny_persists_and_exact_retry_recovers(monkeypatch, tmp_path):
    with _initialized_cross_schema_consumption() as (
        _connect, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
    ):
        request = _cross_schema_request("0199f8a1-8c00-7000-8000-000000000937")
        now = datetime(2026, 10, 10, 12, 7, tzinfo=timezone.utc)
        pin = PinFijo(
            TrustedPolicyPin("jax", "a" * 40, "b" * 40, "refs/heads/main"),
            "refs/heads/main", repository_path=tmp_path,
        )
        snapshot = object()
        snapshot_loads = []

        def load_snapshot(*args):
            snapshot_loads.append(args)
            return snapshot

        monkeypatch.setattr("policy.rule_authority.kernel.load_trusted_policy_snapshot", load_snapshot)
        kernel = RuleAuthorityKernel(
            active_policy_pin_provider=pin,
            authority_ledger_store=ledger_store,
            trusted_root=root,
            trusted_checkpoint_store=checkpoint_store,
            stop_provider=StopFijo(activo=True, huella="sha256:stop-active"),
            trusted_clock=RelojDeterminista(now),
            capability_classification_provider=ClasificacionFija({}),
            rule_authority_store=MariaDBRuleDecisionStore(
                app_connect, checkpoint_store=audit_checkpoint
            ),
        )

        first = kernel.evaluate(request)
        retry = kernel.evaluate(request)

        assert first.decision.status is RuleDecisionStatus.DENY
        assert first.decision.reason_code == "STOP_ACTIVE"
        assert first.permit is None
        assert retry == first
        assert len(snapshot_loads) == 1


def _revoke_rule_grant(ledger_store, root, checkpoint_store, grant_event, *, event_id):
    return append_authority_event(
        ledger_store, root, ledger_store._test_rule_grant_key,
        AuthorityEventIntent(
            AuthorityEventType.RULE_RATIFICATION_REVOKED, "human:fernando",
            rule_ratification_event_id=grant_event.event_id,
        ),
        event_id=event_id,
        recorded_at_utc=datetime(2026, 10, 10, 12, 1, tzinfo=timezone.utc),
        checkpoint_store=checkpoint_store,
    )


def test_kernel_evaluate_serializes_rule_revocation_in_both_orders(monkeypatch, tmp_path):
    """The real Block-4 fence orders a permit and a signed rule revocation."""
    with _initialized_cross_schema_consumption() as (
        _connect, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
    ):
        kernel, request, grant_event = _kernel_permit_fixture(
            tmp_path, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
            request_id="0199f8a1-8c00-7000-8000-000000000941",
            event_id="0199f8a1-8c00-7000-8000-000000000942",
        )
        # Force two identical requests to miss the fast recovery path before
        # either can enter the shared Block-4 owner. The loser must recover the
        # first committed decision+permit from that locked transaction.
        recovery_barrier = threading.Barrier(2)
        original_recover = kernel._store.recover_durable_evaluation

        def both_initial_misses(candidate):
            recovered = original_recover(candidate)
            if recovered is None:
                recovery_barrier.wait(timeout=10)
            return recovered

        monkeypatch.setattr(kernel._store, "recover_durable_evaluation", both_initial_misses)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(kernel.evaluate, request)
            second = pool.submit(kernel.evaluate, request)
            first_outcome = first.result(timeout=20)
            second_outcome = second.result(timeout=20)
        assert first_outcome.decision == second_outcome.decision
        assert first_outcome.decision.status is RuleDecisionStatus.PERMIT
        assert first_outcome.permit.projection() == second_outcome.permit.projection()
        monkeypatch.setattr(kernel._store, "recover_durable_evaluation", original_recover)
        with _connect(database="jax_rule_authority") as admin:
            with admin.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM rule_decisions WHERE request_id=%s", (request.request_id,))
                assert cursor.fetchone()[0] == 1
                cursor.execute("SELECT COUNT(*) FROM rule_permits WHERE request_id=%s", (request.request_id,))
                assert cursor.fetchone()[0] == 1

        request = replace(
            request, request_id="0199f8a1-8c00-7000-8000-00000000094a", request_hash=None,
        )
        verified = threading.Event()
        release = threading.Event()
        original_verify = _AuthorityLedgerCheckpointFence.verify_for_update

        def pause_after_verified(fence, owner, trusted_root):
            state = original_verify(fence, owner, trusted_root)
            if not verified.is_set():
                verified.set()
                assert release.wait(timeout=5), "la prueba no liberó Block 4"
            return state

        monkeypatch.setattr(_AuthorityLedgerCheckpointFence, "verify_for_update", pause_after_verified)
        revoke_started = threading.Event()
        revoke_lock_attempted = threading.Event()
        original_locked = TrustedCheckpointStore.locked

        @contextlib.contextmanager
        def signal_revoke_lock_attempt(store):
            if store is checkpoint_store and revoke_started.is_set():
                revoke_lock_attempted.set()
            with original_locked(store):
                yield

        monkeypatch.setattr(TrustedCheckpointStore, "locked", signal_revoke_lock_attempt)

        def revoke():
            revoke_started.set()
            return _revoke_rule_grant(
                ledger_store, root, checkpoint_store, grant_event,
                event_id="0199f8a1-8c00-7000-8000-000000000943",
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            evaluation = pool.submit(kernel.evaluate, request)
            assert verified.wait(timeout=5)
            revocation = pool.submit(revoke)
            assert revoke_started.wait(timeout=5)
            assert revoke_lock_attempted.wait(timeout=5)
            assert not revocation.done(), "la revocación entró durante el lock Block 4"
            release.set()
            outcome = evaluation.result(timeout=5)
            revocation.result(timeout=5)

        assert outcome.decision.status is RuleDecisionStatus.PERMIT
        assert outcome.permit is not None
        # Reverse order: once the revoke owns Block 4 first, a new evaluation
        # sees no current grant and fails closed.
        denied = kernel.evaluate(replace(
            request, request_id="0199f8a1-8c00-7000-8000-000000000944", request_hash=None,
        ))
        assert denied.decision.status is RuleDecisionStatus.DENY
        assert denied.decision.reason_code == "AUTHORITY_INVALID"
        assert denied.permit is None


def test_kernel_evaluate_waits_for_revocation_that_already_holds_block4_flock(
        monkeypatch, tmp_path):
    """Revocation-first is concurrent: evaluation reaches, then waits on, Block 4."""
    with _initialized_cross_schema_consumption() as (
        _connect, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
    ):
        kernel, request, grant_event = _kernel_permit_fixture(
            tmp_path, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
            request_id="0199f8a1-8c00-7000-8000-000000000959",
            event_id="0199f8a1-8c00-7000-8000-000000000960",
        )
        revoker_thread = []
        revoke_lock_held = threading.Event()
        evaluation_lock_attempted = threading.Event()
        release_revocation = threading.Event()
        original_locked = TrustedCheckpointStore.locked

        @contextlib.contextmanager
        def hold_revocation_lock(store):
            is_revoker = threading.get_ident() in revoker_thread
            if store is checkpoint_store and is_revoker:
                with original_locked(store):
                    revoke_lock_held.set()
                    assert release_revocation.wait(timeout=5), "la prueba no liberó revocación"
                    yield
            else:
                if store is checkpoint_store and revoke_lock_held.is_set():
                    evaluation_lock_attempted.set()
                with original_locked(store):
                    yield

        monkeypatch.setattr(TrustedCheckpointStore, "locked", hold_revocation_lock)

        def revoke():
            revoker_thread.append(threading.get_ident())
            return _revoke_rule_grant(
                ledger_store, root, checkpoint_store, grant_event,
                event_id="0199f8a1-8c00-7000-8000-000000000961",
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            revocation = pool.submit(revoke)
            assert revoke_lock_held.wait(timeout=5)
            evaluation = pool.submit(kernel.evaluate, request)
            assert evaluation_lock_attempted.wait(timeout=5)
            assert not evaluation.done(), "evaluate no esperó la revocación que ya tenía Block 4"
            release_revocation.set()
            revocation.result(timeout=5)
            outcome = evaluation.result(timeout=5)

        assert outcome.decision.status is RuleDecisionStatus.DENY
        assert outcome.decision.reason_code == "AUTHORITY_INVALID"
        assert outcome.permit is None


def test_kernel_consume_serializes_rule_revocation_in_both_orders(monkeypatch, tmp_path):
    """A real permit lock + Block-4 owner cannot be crossed by revocation."""
    with _initialized_cross_schema_consumption() as (
        _connect, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
    ):
        kernel, request, grant_event = _kernel_permit_fixture(
            tmp_path / "first", ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
            request_id="0199f8a1-8c00-7000-8000-000000000951",
            event_id="0199f8a1-8c00-7000-8000-000000000952",
        )
        permit = kernel.evaluate(request).permit
        assert permit is not None
        validated = threading.Event()
        release = threading.Event()
        original_validate = kernel._validate_consumption_context

        def pause_after_block4(**kwargs):
            validated.set()
            assert release.wait(timeout=5), "la prueba no liberó consumo verificado"
            return original_validate(**kwargs)

        monkeypatch.setattr(kernel, "_validate_consumption_context", pause_after_block4)
        revoke_started = threading.Event()
        revoke_lock_attempted = threading.Event()
        original_locked = TrustedCheckpointStore.locked

        @contextlib.contextmanager
        def signal_revoke_lock_attempt(store):
            if store is checkpoint_store and revoke_started.is_set():
                revoke_lock_attempted.set()
            with original_locked(store):
                yield

        monkeypatch.setattr(TrustedCheckpointStore, "locked", signal_revoke_lock_attempt)

        def revoke_first_grant():
            revoke_started.set()
            return _revoke_rule_grant(
                ledger_store, root, checkpoint_store, grant_event,
                event_id="0199f8a1-8c00-7000-8000-000000000953",
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            consumption = pool.submit(
                kernel.consume, permit.permit_id, request.request_hash,
                "0199f8a1-8c00-7000-8000-000000000954",
            )
            assert validated.wait(timeout=5)
            revocation = pool.submit(revoke_first_grant)
            assert revoke_started.wait(timeout=5)
            assert revoke_lock_attempted.wait(timeout=5)
            assert not revocation.done(), "la revocación entró durante consumo Block 4"
            release.set()
            consumed = consumption.result(timeout=5)
            revocation.result(timeout=5)

        assert consumed.permit_id == permit.permit_id

        # Reverse order on a fresh, still-unconsumed permit: revoke first,
        # then consume must replay Block 4 and reject its withdrawn grant.
        kernel2, request2, grant_event2 = _kernel_permit_fixture(
            tmp_path / "second", ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
            request_id="0199f8a1-8c00-7000-8000-000000000955",
            event_id="0199f8a1-8c00-7000-8000-000000000956",
        )
        permit2 = kernel2.evaluate(request2).permit
        assert permit2 is not None
        _revoke_rule_grant(
            ledger_store, root, checkpoint_store, grant_event2,
            event_id="0199f8a1-8c00-7000-8000-000000000957",
        )
        with pytest.raises(AuthorityStateError, match="ratificación.*vigente"):
            kernel2.consume(
                permit2.permit_id, request2.request_hash,
                "0199f8a1-8c00-7000-8000-000000000958",
            )


def test_kernel_consume_waits_for_revocation_that_already_holds_block4_flock(
        monkeypatch, tmp_path):
    """A preexisting revocation fence wins before the permit can be consumed."""
    with _initialized_cross_schema_consumption() as (
        _connect, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
    ):
        kernel, request, grant_event = _kernel_permit_fixture(
            tmp_path, ledger_store, root, checkpoint_store, audit_checkpoint, app_connect,
            request_id="0199f8a1-8c00-7000-8000-000000000962",
            event_id="0199f8a1-8c00-7000-8000-000000000963",
        )
        permit = kernel.evaluate(request).permit
        assert permit is not None
        revoker_thread = []
        revoke_lock_held = threading.Event()
        consumption_lock_attempted = threading.Event()
        release_revocation = threading.Event()
        original_locked = TrustedCheckpointStore.locked

        @contextlib.contextmanager
        def hold_revocation_lock(store):
            is_revoker = threading.get_ident() in revoker_thread
            if store is checkpoint_store and is_revoker:
                with original_locked(store):
                    revoke_lock_held.set()
                    assert release_revocation.wait(timeout=5), "la prueba no liberó revocación"
                    yield
            else:
                if store is checkpoint_store and revoke_lock_held.is_set():
                    consumption_lock_attempted.set()
                with original_locked(store):
                    yield

        monkeypatch.setattr(TrustedCheckpointStore, "locked", hold_revocation_lock)

        def revoke():
            revoker_thread.append(threading.get_ident())
            return _revoke_rule_grant(
                ledger_store, root, checkpoint_store, grant_event,
                event_id="0199f8a1-8c00-7000-8000-000000000964",
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            revocation = pool.submit(revoke)
            assert revoke_lock_held.wait(timeout=5)
            consumption = pool.submit(
                kernel.consume, permit.permit_id, request.request_hash,
                "0199f8a1-8c00-7000-8000-000000000965",
            )
            assert consumption_lock_attempted.wait(timeout=5)
            assert not consumption.done(), "consume no esperó la revocación que ya tenía Block 4"
            release_revocation.set()
            revocation.result(timeout=5)
            with pytest.raises(AuthorityStateError, match="ratificación.*vigente"):
                consumption.result(timeout=5)




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
        _persisted_decision, permit = _record_evaluation_permit(
            ledger_store, checkpoint_store, root, decision_store,
            request, decision, permit_draft,
        )
        permit_head = audit_checkpoint.head_actual()
        fail_close = True
        draft = RulePermitConsumptionDraft(
            permit_id=permit.permit_id,
            consumption_request_id="0199f8a1-8c00-7000-8000-000000000911",
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


def test_evaluation_fence_holds_mariadb_head_lock_until_rollback():
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
            with fence.transaction_owner(
                decision_store, purpose=AuthorityLedgerTransactionPurpose.EVALUATION
            ) as owner:
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
                owner.rollback()
                assert not checkpoint_lock_acquired.is_set(), "rollback liberó el checkpoint antes del contexto"
                assert not rule_authority_lock_acquired.is_set(), "rollback liberó Rule Authority antes del contexto"
                assert checkpoint_store._lock_depth == 1, "el fence perdió la propiedad del checkpoint tras rollback DB"
                assert rule_authority_checkpoint_store._depth == 1, "el lock RA se liberó antes del contexto"

        assert writer_acquired.wait(5), "el escritor siguió bloqueado tras rollback"
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
                owner.rollback()


def test_checkpoint_fence_reuses_owner_connection_and_rejects_real_autocommit():
    with _initialized_authority_ledger() as (connect, store, root, _key, checkpoint_store):
        opened = []

        def counted_connect(**kwargs):
            opened.append(True)
            return connect(**kwargs)

        store = MariaDBAuthorityLedgerStore(counted_connect)
        with store.checkpoint_fence(checkpoint_store) as fence:
            with _mariadb_transaction_owner(counted_connect, fence, checkpoint_store) as owner:
                assert not hasattr(owner, "cursor"), "el owner no expone cursor SQL crudo"
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
                    with fence.transaction_owner(
                        decision_store, purpose=AuthorityLedgerTransactionPurpose.EVALUATION
                    ) as owner:
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
                owner.rollback()


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
                owner.rollback()


def test_provisioning_rejects_preexisting_principal_without_adopting_privileges():
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
            with pytest.raises(ValueError, match="principal nuevo"):
                provision_application_account(
                    admin, user, secrets.token_urlsafe(24), host="localhost", role="writer",
                )
        with connect() as admin:
            with admin.cursor() as cursor:
                cursor.execute(f"SHOW GRANTS FOR '{user}'@'localhost'")
                grants = [row[0] for row in cursor.fetchall()]
    assert any("ALL PRIVILEGES" in grant for grant in grants)


def test_provisioning_rejects_public_identity_before_creating_target():
    with _ephemeral_mariadb() as connect:
        with connect() as admin:
            _apply_migration(admin, MIGRATION)
            with admin.cursor() as cursor:
                cursor.execute("CREATE USER 'PUBLIC'@'%' IDENTIFIED BY 'ephemeral-test-only'")
            admin.commit()
        with connect(database="jax_authority") as admin:
            from policy.authority_ledger.provisioning import provision_application_account
            with pytest.raises(ValueError, match="PUBLIC"):
                provision_application_account(
                    admin, "brand_new_authority", secrets.token_urlsafe(24),
                    host="127.0.0.1", role="reader",
                )
        with connect() as admin:
            with admin.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM mysql.global_priv WHERE User='brand_new_authority'")
                assert cursor.fetchone() == (0,)


def test_provisioning_rejects_homonymous_other_host_and_role_before_dcl():
    user = "authority_homonym"
    with _ephemeral_mariadb() as connect:
        with connect() as admin:
            _apply_migration(admin, MIGRATION)
            with admin.cursor() as cursor:
                cursor.execute(f"CREATE USER '{user}'@'localhost' IDENTIFIED BY 'ephemeral-test-only'")
                cursor.execute("CREATE ROLE authority_test_role")
                cursor.execute(f"GRANT authority_test_role TO '{user}'@'localhost'")
            admin.commit()
        with connect(database="jax_authority") as admin:
            from policy.authority_ledger.provisioning import provision_application_account
            with pytest.raises(ValueError, match="principal nuevo"):
                provision_application_account(
                    admin, user, secrets.token_urlsafe(24), host="127.0.0.1", role="reader",
                )
        with connect() as admin:
            with admin.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM mysql.global_priv WHERE User=%s AND Host='127.0.0.1'", (user,))
                assert cursor.fetchone() == (0,)


def test_provisioning_unlock_response_loss_is_unknown_and_fresh_admin_observes_state():
    """The server applies UNLOCK but the client loses its response: never claim locked."""
    user = "authority_unlock_loss"
    with _ephemeral_mariadb() as connect:
        with connect() as admin:
            _apply_migration(admin, MIGRATION)

        class CursorLoss:
            def __init__(self, cursor):
                self.cursor = cursor
            def execute(self, statement, params=None):
                result = self.cursor.execute(statement, params)
                if statement.startswith("ALTER USER") and "ACCOUNT UNLOCK" in statement:
                    raise pymysql.err.OperationalError(2013, "simulated response loss after server DCL")
                return result
            def __getattr__(self, name):
                return getattr(self.cursor, name)

        class ConnectionLoss:
            def __init__(self, connection):
                self.connection = connection
            def cursor(self):
                return CursorLoss(self.connection.cursor())
            def __getattr__(self, name):
                return getattr(self.connection, name)

        raw = connect(database="jax_authority")
        try:
            from policy.authority_ledger.provisioning import provision_application_account
            with pytest.raises(RuntimeError, match="resultado desconocido"):
                provision_application_account(ConnectionLoss(raw), user, secrets.token_urlsafe(24),
                                              host="127.0.0.1", role="reader")
        finally:
            raw.close()
        with connect() as fresh_admin:
            with fresh_admin.cursor() as cursor:
                cursor.execute("SELECT JSON_VALUE(Priv, '$.account_locked') FROM mysql.global_priv WHERE User=%s AND Host=%s",
                               (user, "127.0.0.1"))
                assert str(cursor.fetchone()[0]).lower() in {"0", "false"}


def test_bootstrap_first_init_exact_retry_and_different_stage_fail_closed():
    """Exercise B4's actual MariaDB insert + external anchors in an isolated DB."""
    with _ephemeral_mariadb() as connect:
        with connect() as admin:
            _apply_migration(admin, MIGRATION)
        staged = stage_authority_genesis(
            encode_public_key(Ed25519PrivateKey.generate().public_key()), "bootstrap-test-1",
        )
        with tempfile.TemporaryDirectory(prefix="jax-b4-bootstrap-") as directory:
            directory_path = Path(directory)
            checkpoints = TrustedCheckpointStore(
                directory_path / "checkpoint.log",
                bootstrap_receipt_path=directory_path / "receipt.json",
            )
            store = MariaDBAuthorityLedgerStore(lambda: connect(database="jax_authority"))
            root_path = directory_path / "trusted-root.json"
            initialize_staged_authority_ledger(
                store, lambda: connect(database="jax_authority"), staged, staged.trusted_root,
                checkpoints, trusted_root_path=root_path, trusted_root_owner_uid=os.getuid(),
            )
            assert store.get_genesis() == staged.genesis
            # A completed exact attempt verifies; it never reinitializes.
            initialize_staged_authority_ledger(
                store, lambda: connect(database="jax_authority"), staged, staged.trusted_root,
                checkpoints, trusted_root_path=root_path, trusted_root_owner_uid=os.getuid(),
            )
            different = stage_authority_genesis(
                encode_public_key(Ed25519PrivateKey.generate().public_key()), "bootstrap-test-2",
            )
            with pytest.raises(ValueError, match="genesis existente difiere"):
                initialize_staged_authority_ledger(
                    store, lambda: connect(database="jax_authority"), different, different.trusted_root,
                    checkpoints, trusted_root_path=root_path, trusted_root_owner_uid=os.getuid(),
                )


def test_bootstrap_existing_genesis_without_external_anchors_requires_recovery():
    with _ephemeral_mariadb() as connect:
        with connect() as admin:
            _apply_migration(admin, MIGRATION)
        staged = stage_authority_genesis(
            encode_public_key(Ed25519PrivateKey.generate().public_key()), "bootstrap-test-partial",
        )
        with connect() as admin:
            with admin.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO jax_authority.authority_ledger_genesis "
                    "(singleton,canonical_genesis,genesis_hash) VALUES (1,%s,%s)",
                    (staged.canonical_genesis, staged.genesis_hash),
                )
            admin.commit()
        with tempfile.TemporaryDirectory(prefix="jax-b4-bootstrap-partial-") as directory:
            path = Path(directory)
            checkpoints = TrustedCheckpointStore(path / "checkpoint.log", bootstrap_receipt_path=path / "receipt.json")
            store = MariaDBAuthorityLedgerStore(lambda: connect(database="jax_authority"))
            with pytest.raises(ValueError, match="parcial"):
                initialize_staged_authority_ledger(
                    store, lambda: connect(database="jax_authority"), staged, staged.trusted_root,
                    checkpoints, trusted_root_path=path / "trusted-root.json", trusted_root_owner_uid=os.getuid(),
                )
            assert not checkpoints.path.exists()


def test_bootstrap_rejects_nonempty_head_history_without_external_anchors():
    with _ephemeral_mariadb() as connect:
        with connect() as admin:
            _apply_migration(admin, MIGRATION)
            with admin.cursor() as cursor:
                cursor.execute(
                    "UPDATE jax_authority.authority_ledger_head SET sequence=1,head_event_id=%s,head_event_hash=%s WHERE singleton=1",
                    ("event-existing", "sha256:" + "a" * 64),
                )
            admin.commit()
        staged = stage_authority_genesis(
            encode_public_key(Ed25519PrivateKey.generate().public_key()), "bootstrap-test-history",
        )
        with tempfile.TemporaryDirectory(prefix="jax-b4-bootstrap-history-") as directory:
            path = Path(directory)
            checkpoints = TrustedCheckpointStore(path / "checkpoint.log", bootstrap_receipt_path=path / "receipt.json")
            store = MariaDBAuthorityLedgerStore(lambda: connect(database="jax_authority"))
            with pytest.raises(ValueError, match="no está vacío"):
                initialize_staged_authority_ledger(
                    store, lambda: connect(database="jax_authority"), staged, staged.trusted_root,
                    checkpoints, trusted_root_path=path / "trusted-root.json", trusted_root_owner_uid=os.getuid(),
                )
            assert not checkpoints.path.exists()


def test_storage_rejects_corrupt_stored_genesis_hash():
    with _ephemeral_mariadb() as connect:
        with connect() as admin:
            _apply_migration(admin, MIGRATION)
        staged = stage_authority_genesis(
            encode_public_key(Ed25519PrivateKey.generate().public_key()), "bootstrap-test-corrupt",
        )
        with connect() as admin:
            with admin.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO jax_authority.authority_ledger_genesis "
                    "(singleton,canonical_genesis,genesis_hash) VALUES (1,%s,%s)",
                    (staged.canonical_genesis, staged.genesis_hash),
                )
                cursor.execute("DROP TRIGGER jax_authority.no_update_genesis")
                cursor.execute(
                    "UPDATE jax_authority.authority_ledger_genesis SET genesis_hash=%s WHERE singleton=1",
                    ("sha256:" + "0" * 64,),
                )
            admin.commit()
        store = MariaDBAuthorityLedgerStore(lambda: connect(database="jax_authority"))
        with pytest.raises(AuthorityStateError, match="hash almacenado"):
            store.get_genesis()


def test_bootstrap_inspection_real_absent_partial_and_limited_metadata():
    with _ephemeral_mariadb() as connect:
        with tempfile.TemporaryDirectory(prefix="jax-b4-inspect-") as directory:
            root = Path(directory)
            with connect() as admin:
                absent = inspect_authority_bootstrap(
                    admin, trusted_root_path=root / "trusted-root", checkpoint_path=root / "checkpoint",
                    receipt_path=root / "receipt", trusted_owner_uid=os.getuid(),
                )
                assert absent.status is BootstrapInspectionStatus.CANDIDATE
                with admin.cursor() as cursor:
                    cursor.execute("CREATE SCHEMA jax_authority")
                admin.commit()
                partial = inspect_authority_bootstrap(
                    admin, trusted_root_path=root / "trusted-root", checkpoint_path=root / "checkpoint",
                    receipt_path=root / "receipt", trusted_owner_uid=os.getuid(),
                )
                assert partial.status is BootstrapInspectionStatus.RECOVERY
                password = secrets.token_urlsafe(24)
                cursor = admin.cursor()
                cursor.execute("CREATE USER 'limited_inspect'@'localhost' IDENTIFIED BY %s", (password,))
                cursor.close()
                admin.commit()
            limited = pymysql.connect(
                unix_socket=connect._socket_dir + "/mysqld.sock", user="limited_inspect", password=password,
                autocommit=False, charset="utf8mb4",
            )
            try:
                result = inspect_authority_bootstrap(
                    limited, trusted_root_path=root / "trusted-root", checkpoint_path=root / "checkpoint",
                    receipt_path=root / "receipt", trusted_owner_uid=os.getuid(),
                )
            finally:
                limited.close()
            assert result.status is BootstrapInspectionStatus.INSPECTION_UNAVAILABLE


def test_bootstrap_inspection_rejects_effective_schema_missing_required_trigger():
    with _ephemeral_mariadb() as connect:
        with connect() as admin:
            _apply_migration(admin, MIGRATION)
            with tempfile.TemporaryDirectory(prefix="jax-b4-schema-") as directory:
                root = Path(directory)
                candidate = inspect_authority_bootstrap(
                    admin, trusted_root_path=root / "trusted-root", checkpoint_path=root / "checkpoint",
                    receipt_path=root / "receipt", trusted_owner_uid=os.getuid(),
                )
                assert candidate.status is BootstrapInspectionStatus.CANDIDATE
                with admin.cursor() as cursor:
                    cursor.execute("DROP TRIGGER jax_authority.no_delete_authority_events")
                admin.commit()
                result = inspect_authority_bootstrap(
                    admin, trusted_root_path=root / "trusted-root", checkpoint_path=root / "checkpoint",
                    receipt_path=root / "receipt", trusted_owner_uid=os.getuid(),
                )
                assert result.status is BootstrapInspectionStatus.RECOVERY


def test_provisioning_reader_tcp_is_exact_and_cannot_write():
    user = "jax_authority_tcp_reader"
    password = secrets.token_urlsafe(24)
    with _ephemeral_mariadb() as connect:
        with connect() as admin:
            _apply_migration(admin, MIGRATION)
            from policy.authority_ledger.provisioning import provision_application_account
            provision_application_account(
                admin, user, password, host="127.0.0.1", role="reader",
            )
        # The password is held only in a 0600 client configuration inside the
        # disposable socket mount, never in docker's command line or output.
        subprocess.run(
            ["sudo", "-n", "chown", f"{os.getuid()}:{os.getgid()}", connect._socket_dir],
            check=True, capture_output=True, text=True,
        )
        client_config = Path(connect._socket_dir) / "reader.cnf"
        client_config.write_text(
            f"[client]\nuser={user}\npassword={password}\ndatabase=jax_authority\n",
            encoding="utf-8",
        )
        client_config.chmod(0o600)
        result = subprocess.run(
            [*connect._docker, "exec", connect._container, "mariadb",
             "--defaults-extra-file=/run/mysqld/reader.cnf", "--protocol=TCP",
             "-h", "127.0.0.1", "-N", "-e", "SELECT CURRENT_USER(), @@port"],
            check=True, capture_output=True, text=True,
        )
        assert result.stdout.strip() == f"{user}@127.0.0.1\t3306"
        denied = subprocess.run(
            [*connect._docker, "exec", connect._container, "mariadb",
             "--defaults-extra-file=/run/mysqld/reader.cnf", "--protocol=TCP",
             "-h", "127.0.0.1", "-e",
             "INSERT INTO authority_events (sequence,event_id,event_type,actor_id,canonical_intent,canonical_event,evidence_refs,event_hash,signature,recorded_at_utc) VALUES (1,'x','x','x','x','x','x','x','x',NOW())"],
            check=False, capture_output=True, text=True,
        )
        assert denied.returncode != 0
        assert "INSERT command denied" in denied.stderr


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
