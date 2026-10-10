"""MariaDB integration for Faro F1.1's durable decision store."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import errno
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

import pymysql
import pytest

from tests.policy.catalogo_pin import catalogo_del_pin
from policy.authority_ledger.canonical import canonical_bytes
from policy.authority_ledger.errors import AuthorityStateError
from policy.rule_authority.errors import RuleAuthorityError
from policy.rule_authority.models import (
    RuleDecision,
    RuleDecisionStatus,
    RuleEvaluationRequest,
)
from policy.rule_authority.provisioning import provision_application_account
from policy.rule_authority.storage import (
    MariaDBRuleDecisionStore,
    RuleAuthorityStorageError,
    _catalog_hash,
    _catalog_projection,
    _decision_projection,
    _record_hash,
)


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "policy/rule_authority/migrations/001_rule_authority_kernel.sql"
IMAGE = os.environ.get("JAX_RULE_AUTHORITY_TEST_MARIADB_IMAGE", "mariadb:12.3.3")
CATALOG = catalogo_del_pin()
_CATALOG_BYTES = (ROOT / "policy/faro/catalogo-topes.json").read_bytes()
ALTERNATE_CATALOG = catalogo_del_pin(_CATALOG_BYTES.replace(b'"pagos"]', b'"pagos", "ordenes"]'))


def _run(docker: list[str], *args: str):
    return subprocess.run([*docker, *args], check=True, capture_output=True, text=True)


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
        [*docker, "logs", "--timestamps", "--tail", str(tail), container],
        capture_output=True,
        text=True,
        check=False,
    )
    return logs.stdout + logs.stderr


def _container_diagnostics(docker: list[str], container: str) -> str:
    state = subprocess.run(
        [*docker, "inspect", "--format", "{{json .State}}", container],
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        "--- docker inspect .State ---\n"
        f"{state.stdout}{state.stderr}\n"
        "--- docker logs --timestamps ---\n"
        f"{_container_logs(docker, container, tail=200)[-4000:]}"
    )


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
                        if cursor.fetchone() != (1,):
                            raise AssertionError("MariaDB respondió una sonda SELECT 1 inválida")
                        cursor.execute("SELECT @@port")
                        if cursor.fetchone() == (3306,):
                            return socket_path
                        last_error = AssertionError("MariaDB informó un puerto final distinto de 3306")
            except (OSError, pymysql.MySQLError, AssertionError) as error:
                if not _is_transient_mariadb_startup_error(error.__cause__ or error):
                    raise
                last_error = error
        time.sleep(0.25)
    raise AssertionError(
        f"MariaDB no quedó lista en {timeout:.0f} s (init terminado: {initialized}; "
        f"último error: {last_error})."
    )


def _apply_migration(connection) -> None:
    delimiter = ";"
    statement = []
    statement_ordinal = 0
    sql = MIGRATION.read_text(encoding="utf-8")
    with connection.cursor() as cursor:
        def execute_migration_statement(statement_text: str) -> None:
            nonlocal statement_ordinal
            statement_ordinal += 1
            try:
                cursor.execute(statement_text)
            except pymysql.MySQLError as error:
                raise AssertionError(
                    f"falló la sentencia {statement_ordinal} de la migración"
                ) from error

        for raw_line in sql.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("--"):
                continue
            if line.upper().startswith("DELIMITER "):
                delimiter = line.split(None, 1)[1]
                continue
            statement.append(raw_line)
            joined = "\n".join(statement).rstrip()
            if joined.endswith(delimiter):
                execute_migration_statement(joined[:-len(delimiter)].strip())
                statement = []
    assert not statement, "migración dejó SQL sin terminar"
    connection.commit()


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


def test_readiness_rejects_a_socket_server_reporting_a_nonfinal_port(monkeypatch, tmp_path) -> None:
    class Cursor:
        def __init__(self):
            self.executed = []
            self._result = None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def execute(self, statement):
            self.executed.append(statement)
            self._result = (1,) if statement == "SELECT 1" else (0,)

        def fetchone(self):
            return self._result

    class Connection:
        def __init__(self):
            self.cursor_instance = Cursor()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def cursor(self):
            return self.cursor_instance

    connection = Connection()
    monotonic_values = iter((0.0, 0.0, 0.0, 0.0, 1.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(
        sys.modules[__name__], "_container_logs",
        lambda *_args, **_kwargs: "init process done",
    )
    monkeypatch.setattr(sys.modules[__name__], "_run", lambda *_args: None)
    monkeypatch.setattr(pymysql, "connect", lambda **_kwargs: connection)

    with pytest.raises(AssertionError, match="puerto final distinto de 3306"):
        _wait_until_ready(["docker"], "test-container", str(tmp_path), "test-password", timeout=0.5)

    assert connection.cursor_instance.executed == ["SELECT 1", "SELECT @@port"]


def test_migration_failure_identifies_statement_ordinal_without_retry(monkeypatch, tmp_path) -> None:
    class Cursor:
        def __init__(self):
            self.executed = []

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def execute(self, statement):
            self.executed.append(statement)
            if statement == "TWO":
                raise pymysql.err.OperationalError(2013, "connection lost")

    class Connection:
        def __init__(self):
            self.cursor_instance = Cursor()
            self.committed = False

        def cursor(self):
            return self.cursor_instance

        def commit(self):
            self.committed = True

    migration = tmp_path / "migration.sql"
    migration.write_text("ONE;\nTWO;\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "MIGRATION", migration)
    connection = Connection()

    with pytest.raises(AssertionError, match="sentencia 2"):
        _apply_migration(connection)

    assert connection.cursor_instance.executed == ["ONE", "TWO"]
    assert not connection.committed


def _request(
    request_id: str,
    recipient: str = "client@example.test",
    catalog=CATALOG,
) -> RuleEvaluationRequest:
    return RuleEvaluationRequest(
        request_id=request_id,
        rule_id="rule-one",
        subject="actor:fernando",
        capability="mail.send",
        objective="notify-client",
        resource_id="message:invoice-42",
        arguments={"recipient": recipient, "body": "Invoice ready"},
        catalogo=catalog,
        quantity=1,
        quantity_unit="mensajes",
    )


def _decision(
    request: RuleEvaluationRequest,
    *,
    status=RuleDecisionStatus.DENY,
    at=1,
) -> RuleDecision:
    return RuleDecision(
        request_id=request.request_id,
        request_hash=request.request_hash,
        status=status,
        required_rule_id=request.rule_id,
        reason_code=(
            "STOP_ACTIVE" if status is RuleDecisionStatus.DENY
            else "RULE_NOT_FOUND" if status is RuleDecisionStatus.MISSING_RULE
            else None
        ),
        decided_at_utc=datetime(2026, 10, 7, 12, 0, at, tzinfo=timezone.utc),
    )


ZERO_HASH = "sha256:" + "0" * 64
APP_USER = "jax_rule_authority_app_test"
TRIGGER_USER = "jax_rule_authority_trigger_test"
IMMUTABLE_TABLES = ("rule_decisions", "rule_permits", "rule_permit_consumptions")


def _append_only(table: str) -> str:
    """Código 1644 (SIGNAL de un trigger) y el mensaje exacto del trigger."""
    return rf"\(1644, '{table} are append-only'\)"


class _Db:
    def __init__(self, docker, socket_path, root_password):
        self.docker = docker
        self.socket_path = socket_path
        self.root_password = root_password

    def connect_as(self, username: str, password: str, *, autocommit: bool, database="jax_rule_authority"):
        return pymysql.connect(
            unix_socket=str(self.socket_path), user=username, password=password,
            database=database, autocommit=autocommit, charset="utf8mb4",
        )

    def root(self, *, autocommit: bool = False):
        return self.connect_as("root", self.root_password, autocommit=autocommit)


@pytest.fixture
def db():
    """MariaDB 12.3.3 efímera, sin red, con la migración aplicada. Una por prueba."""
    assert MIGRATION.is_file(), "falta la migración MariaDB de Rule Authority"
    docker = shlex.split(os.environ.get("JAX_RULE_AUTHORITY_DOCKER_CMD", "docker"))
    root_password = secrets.token_urlsafe(24)
    container = f"jax-rule-authority-{uuid.uuid4().hex[:10]}"
    with tempfile.TemporaryDirectory(prefix="jax-rule-authority-socket-") as socket_dir:
        _run(
            docker, "run", "-d", "--rm", "--network", "none", "--name", container,
            "--mount", f"type=bind,source={socket_dir},target=/run/mysqld",
            "-e", f"MARIADB_ROOT_PASSWORD={root_password}", IMAGE,
        )
        try:
            try:
                socket_path = _wait_until_ready(docker, container, socket_dir, root_password)
                admin = _connect_after_transient_mariadb_startup_error(
                    lambda: pymysql.connect(
                        unix_socket=str(socket_path), user="root", password=root_password,
                        autocommit=False, charset="utf8mb4", connect_timeout=5,
                    ),
                    timeout=10,
                    retry_delay=0.25,
                )
                with admin:
                    with admin.cursor() as cursor:
                        cursor.execute("SELECT VERSION()")
                        assert cursor.fetchone()[0].startswith("12.3.3-")
                    _apply_migration(admin)
            except BaseException as error:
                raise AssertionError(
                    f"falló la preparación de MariaDB aislada:\n"
                    f"{_container_diagnostics(docker, container)}"
                ) from error
            yield _Db(docker, socket_path, root_password)
        finally:
            subprocess.run(
                [*docker, "stop", container], capture_output=True, text=True, check=False
            )
            chown = subprocess.run(
                ["sudo", "-n", "chown", "-R", f"{os.getuid()}:{os.getgid()}", socket_dir],
                capture_output=True, text=True, check=False,
            )
            if chown.returncode and sys.exc_info()[0] is None:
                raise RuntimeError(
                    f"falló la limpieza del socket MariaDB temporal: {chown.stderr.strip()}"
                )


@pytest.fixture
def app(db):
    """Cuenta de aplicación provisionada por el código versionado."""
    password = secrets.token_urlsafe(24)
    with db.root() as admin:
        provision_application_account(admin, APP_USER, password)

    def connect():
        # Explicit BEGIN in the adapter is required: autocommit makes the
        # partial-write assertion kill the no-transaction mutant.
        return db.connect_as(APP_USER, password, autocommit=True)

    return connect


@pytest.fixture
def trigger_account(db):
    """Cuenta que SÍ tiene GRANT de UPDATE y DELETE: solo los triggers frenan."""
    password = secrets.token_urlsafe(24)
    with db.root() as admin:
        with admin.cursor() as cursor:
            cursor.execute(
                f"CREATE USER '{TRIGGER_USER}'@'localhost' IDENTIFIED BY %s", (password,)
            )
            for table in IMMUTABLE_TABLES:
                cursor.execute(
                    f"GRANT SELECT,UPDATE,DELETE ON jax_rule_authority.{table} "
                    f"TO '{TRIGGER_USER}'@'localhost'"
                )
        admin.commit()
    return lambda: db.connect_as(TRIGGER_USER, password, autocommit=False)


def _enable_permit_seeding(db) -> None:
    """Quita el trigger que cierra PERMIT hasta el paso 6, solo en esta base efímera."""
    with db.root() as admin:
        with admin.cursor() as cursor:
            cursor.execute("DROP TRIGGER trg_rule_decisions_no_permit_until_step6")
        admin.commit()


def _seed_permit_decision(cursor, request: RuleEvaluationRequest, sequence: int) -> None:
    decision = RuleDecision(
        request_id=request.request_id,
        request_hash=request.request_hash,
        status=RuleDecisionStatus.PERMIT,
        required_rule_id=request.rule_id,
        reason_code=None,
        decided_at_utc=datetime(2026, 10, 7, 12, 1, tzinfo=timezone.utc),
    )
    projection = _decision_projection(decision, _catalog_hash(request.catalogo))
    cursor.execute(
        "INSERT INTO rule_decisions "
        "(request_id,request_hash,request_catalog_hash,required_rule_id,status,"
        "reason_code,decided_at_utc,canonical_decision,previous_record_hash,"
        "record_hash,audit_sequence) "
        "VALUES (%s,%s,%s,%s,'PERMIT',NULL,%s,%s,NULL,%s,%s)",
        (request.request_id, request.request_hash, _catalog_hash(request.catalogo),
         request.rule_id, decision.decided_at_utc.replace(tzinfo=None),
         canonical_bytes(projection),
         _record_hash(sequence, request.request_id, None, projection), sequence),
    )


PERMIT_INSERT = (
    "INSERT INTO rule_permits "
    "(permit_id,request_id,request_hash,rule_id,rule_path,rule_blob_oid,"
    "rule_content_hash,policy_revision,policy_tree_oid,policy_snapshot_hash,"
    "ratification_event_id,authority_ledger_checkpoint,stop_checkpoint,"
    "capability_id,"
    "capability_version,capability_class,capability_limits,issued_at_utc,"
    "expires_at_utc,permit_hash,canonical_payload,previous_record_hash,"
    "record_hash,audit_sequence) VALUES "
    "(%s,%s,%s,'rule-one','policy/faro/rule-one.yaml',%s,%s,%s,%s,%s,%s,"
    "%s,%s,'mail.send','1','external',%s,UTC_TIMESTAMP(6),"
    "DATE_ADD(UTC_TIMESTAMP(6),INTERVAL 5 MINUTE),%s,%s,%s,%s,%s)"
)


def _permit_values(permit_id: str, request: RuleEvaluationRequest, number: int, sequence: int):
    return (
        permit_id, request.request_id, request.request_hash, "a" * 64,
        ZERO_HASH, "b" * 64, "c" * 64, ZERO_HASH,
        "0199f8a1-8c00-7000-8000-000000000112", b"{}", b"{}", b"{}",
        "sha256:" + f"{number:064x}", b"{}", ZERO_HASH, ZERO_HASH, sequence,
    )


CONSUMPTION_INSERT = (
    "INSERT INTO rule_permit_consumptions "
    "(permit_id,request_hash,consumed_at_utc,canonical_payload,"
    "consumption_hash,previous_record_hash,record_hash,audit_sequence) "
    "VALUES (%s,%s,UTC_TIMESTAMP(6),%s,%s,%s,%s,%s)"
)


def _record_deny(store, request_id: str):
    request = _request(request_id)
    decision = _decision(request)
    assert store.record(request, decision) == decision
    return request, decision


def test_mariadb_store_rolls_back_partial_writes_and_fails_closed(db, app):
    store = MariaDBRuleDecisionStore(app)

    # Fail after INSERT but before the audit-head update can commit. With
    # autocommit enabled in `app`, this proves the adapter owns one transaction.
    with db.root() as admin:
        with admin.cursor() as cursor:
            cursor.execute("""
                CREATE TRIGGER fail_rule_authority_head_update
                BEFORE UPDATE ON rule_authority_audit_head FOR EACH ROW
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='injected failure'
            """)
        admin.commit()
    failed_request = _request("0199f8a1-8c00-7000-8000-000000000102")
    with pytest.raises(RuleAuthorityError):
        store.record(failed_request, _decision(failed_request))
    with db.root(autocommit=True) as admin:
        with admin.cursor() as cursor:
            cursor.execute("DROP TRIGGER fail_rule_authority_head_update")
            cursor.execute(
                "SELECT COUNT(*) FROM rule_decisions WHERE request_id=%s",
                (failed_request.request_id,),
            )
            assert cursor.fetchone()[0] == 0
    assert store.get(failed_request) is None

    # PERMIT is closed until step 6 and nothing is left behind.
    permit_request = _request("0199f8a1-8c00-7000-8000-000000000103")
    with pytest.raises(RuleAuthorityError):
        store.record(permit_request, _decision(permit_request, status=RuleDecisionStatus.PERMIT))
    assert store.get(permit_request) is None

    # An unavailable database is an error, never a returned PERMIT.
    unavailable_calls = 0

    def unavailable():
        nonlocal unavailable_calls
        unavailable_calls += 1
        raise pymysql.err.OperationalError(2003, "database unavailable")

    other = _request("0199f8a1-8c00-7000-8000-000000000104")
    with pytest.raises(RuleAuthorityError):
        MariaDBRuleDecisionStore(unavailable).record(other, _decision(other))
    assert unavailable_calls == 1

    # Distinct new UUID7 requests serialize on the audit head without
    # deadlocking while holding missing-key gap locks.
    concurrent_requests = [
        _request(f"0199f8a1-8c00-7000-8000-{index:012d}") for index in range(200, 208)
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda item: store.record(item, _decision(item)), concurrent_requests
        ))
    assert results == [_decision(item) for item in concurrent_requests]


def test_mariadb_store_rollback_failure_is_logged_and_original_error_raised(db, app, caplog):
    class RollbackBroken:
        """Conexión real cuyo rollback() falla: el error original no se pierde."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def rollback(self):
            raise pymysql.err.OperationalError(2013, "lost connection on rollback")

    with db.root() as admin:
        with admin.cursor() as cursor:
            cursor.execute("""
                CREATE TRIGGER fail_rule_authority_head_update
                BEFORE UPDATE ON rule_authority_audit_head FOR EACH ROW
                SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='injected failure'
            """)
        admin.commit()
    request = _request("0199f8a1-8c00-7000-8000-000000000106")
    store = MariaDBRuleDecisionStore(lambda: RollbackBroken(app()))
    with caplog.at_level("ERROR", logger="policy.rule_authority.storage"):
        with pytest.raises(RuleAuthorityStorageError) as raised:
            store.record(request, _decision(request))
    assert "injected failure" in str(raised.value.__cause__)
    assert any("rollback" in record.getMessage() for record in caplog.records)
    assert any(record.exc_info for record in caplog.records)


def test_mariadb_store_rollback_failure_after_state_error_is_typed_and_logged(db, app, caplog):
    class RollbackBroken:
        """Conexión real cuyo rollback() falla tras un AuthorityStateError."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def rollback(self):
            raise pymysql.err.OperationalError(2013, "lost connection on rollback")

    request_id = "0199f8a1-8c00-7000-8000-000000000107"
    first = _request(request_id)
    assert MariaDBRuleDecisionStore(app).record(first, _decision(first)) == _decision(first)
    other = _request(request_id, recipient="other@example.test")
    store = MariaDBRuleDecisionStore(lambda: RollbackBroken(app()))
    with caplog.at_level("ERROR", logger="policy.rule_authority.storage"):
        with pytest.raises(RuleAuthorityStorageError) as raised:
            store.record(other, _decision(other))
    assert not isinstance(raised.value, AuthorityStateError)
    assert "reutilizado con otro hash" in str(raised.value.__cause__)
    assert any("rollback" in record.getMessage() for record in caplog.records)
    assert any(record.exc_info for record in caplog.records)


def test_mariadb_store_is_idempotent_durable_and_bound_to_request_and_catalog(db, app):
    store = MariaDBRuleDecisionStore(app)
    request = _request("0199f8a1-8c00-7000-8000-000000000101")
    assert store.get(request) is None
    assert store.get(_request(request.request_id, "other@example.test")) is None
    other_pin_request = _request(request.request_id, catalog=ALTERNATE_CATALOG)
    # El pin con otro catálogo cambia el request_hash (lleva el OID del pin).
    assert other_pin_request.request_hash != request.request_hash
    assert store.get(other_pin_request) is None

    decision = _decision(request)
    assert store.record(request, decision) == decision

    # Con la fila YA existente, el mismo request_id bajo otro catálogo no la lee.
    # El None sale de `row[0] != request.request_hash` (el request_hash lleva el OID
    # del pin y por eso difiere); el chequeo `row[1] != _catalog_hash(...)` de get()
    # es defensa en profundidad y esta aserción no lo aísla.
    assert store.get(request) == decision
    assert store.get(other_pin_request) is None

    # El catálogo del request queda persistido y ligado a la fila canónica.
    with db.root(autocommit=True) as admin:
        with admin.cursor() as cursor:
            cursor.execute(
                "SELECT request_catalog_hash FROM rule_decisions WHERE request_id=%s",
                (request.request_id,),
            )
            assert cursor.fetchone()[0] == _catalog_hash(CATALOG)
    assert _catalog_hash(CATALOG) != _catalog_hash(ALTERNATE_CATALOG)

    # A separate connection sees the row immediately after return: durable
    # persistence is part of the synchronous return contract.
    with app() as independent:
        with independent.cursor() as cursor:
            cursor.execute(
                "SELECT request_hash,status,reason_code FROM rule_decisions "
                "WHERE request_id=%s",
                (request.request_id,),
            )
            assert cursor.fetchone() == (request.request_hash, "DENY", "STOP_ACTIVE")

    replay = _decision(request, at=2)
    assert store.record(request, replay) == decision
    assert store.get(request) == decision
    with app() as independent:
        with independent.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM rule_decisions WHERE request_id=%s",
                (request.request_id,),
            )
            assert cursor.fetchone()[0] == 1
    changed = _request(request.request_id, "other@example.test")
    assert store.get(changed) is None
    with pytest.raises(AuthorityStateError):
        store.record(changed, _decision(changed))
    with pytest.raises(AuthorityStateError):
        store.record(other_pin_request, _decision(other_pin_request))

    missing_request = _request("0199f8a1-8c00-7000-8000-000000000107")
    missing = _decision(missing_request, status=RuleDecisionStatus.MISSING_RULE)
    assert store.record(missing_request, missing) == missing
    assert store.get(missing_request) == missing
    with app() as independent:
        with independent.cursor() as cursor:
            cursor.execute(
                "SELECT status,reason_code FROM rule_decisions WHERE request_id=%s",
                (missing_request.request_id,),
            )
            assert cursor.fetchone() == ("MISSING_RULE", "RULE_NOT_FOUND")


@pytest.mark.parametrize("no_catalogo", [
    pytest.param({"oid_pin": "0" * 40, "clases": {}}, id="dict"),
    pytest.param(dict(CATALOG), id="copia-de-dict"),
    pytest.param(None, id="none"),
])
def test_catalog_projection_rejects_anything_that_is_not_a_sealed_catalog(no_catalogo):
    with pytest.raises(RuleAuthorityStorageError, match="catalogo de solicitud"):
        _catalog_projection(no_catalogo)


def test_mariadb_append_only_triggers_fire_for_every_update_and_delete(db, app, trigger_account):
    """Cada trigger se ejercita con una fila SIN hijos, para que la FK no lo tape."""
    store = MariaDBRuleDecisionStore(app)
    deny_request, _ = _record_deny(store, "0199f8a1-8c00-7000-8000-000000000301")
    _enable_permit_seeding(db)

    consumed_request = _request("0199f8a1-8c00-7000-8000-000000000302")
    free_request = _request("0199f8a1-8c00-7000-8000-000000000303")
    consumed_permit = "0199f8a1-8c00-7000-8000-000000000311"
    free_permit = "0199f8a1-8c00-7000-8000-000000000312"
    with db.root() as admin:
        with admin.cursor() as cursor:
            _seed_permit_decision(cursor, consumed_request, 100)
            _seed_permit_decision(cursor, free_request, 101)
            cursor.execute(
                PERMIT_INSERT, _permit_values(consumed_permit, consumed_request, 1, 100)
            )
            cursor.execute(PERMIT_INSERT, _permit_values(free_permit, free_request, 2, 101))
            cursor.execute(
                CONSUMPTION_INSERT,
                (consumed_permit, consumed_request.request_hash, b"{}",
                 ZERO_HASH, ZERO_HASH, ZERO_HASH, 102),
            )
        admin.commit()

    # (tabla, columna clave, valor): todas sin filas hijas que referencien la fila
    # salvo `consumed_permit`, que solo se usa para UPDATE.
    targets = (
        ("rule_decisions", "request_id", deny_request.request_id),          # DENY sin hijos
        ("rule_permits", "permit_id", free_permit),                         # permiso sin consumo
        ("rule_permit_consumptions", "permit_id", consumed_permit),         # consumo (hoja)
    )
    trigger = trigger_account()
    with trigger:
        for table, key, value in targets:
            for verb, sql in (
                ("UPDATE", f"UPDATE {table} SET audit_sequence=audit_sequence+1 WHERE {key}=%s"),
                ("DELETE", f"DELETE FROM {table} WHERE {key}=%s"),
            ):
                with trigger.cursor() as cursor:
                    with pytest.raises(pymysql.MySQLError, match=_append_only(table)):
                        cursor.execute(sql, (value,))
                    trigger.rollback()
        # UPDATE sobre filas con hijos también lo frena el trigger.
        for table, key, value in (
            ("rule_decisions", "request_id", consumed_request.request_id),
            ("rule_permits", "permit_id", consumed_permit),
        ):
            with trigger.cursor() as cursor:
                with pytest.raises(pymysql.MySQLError, match=_append_only(table)):
                    cursor.execute(
                        f"UPDATE {table} SET audit_sequence=audit_sequence+1 WHERE {key}=%s",
                        (value,),
                    )
                trigger.rollback()

    # Nada cambió: las tres filas objetivo siguen ahí.
    with db.root(autocommit=True) as admin:
        with admin.cursor() as cursor:
            for table, key, value in targets:
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {key}=%s", (value,))
                assert cursor.fetchone()[0] == 1


def test_mariadb_constraints_permit_gate_oids_and_indexed_plans(db, app):
    store = MariaDBRuleDecisionStore(app)
    request, _ = _record_deny(store, "0199f8a1-8c00-7000-8000-000000000401")
    permit_request = _request("0199f8a1-8c00-7000-8000-000000000402")
    permit_id = "0199f8a1-8c00-7000-8000-000000000411"
    values = _permit_values(permit_id, permit_request, 1, 100)

    # Mientras el paso 6 no exista, ni la cuenta de aplicación puede insertar PERMIT.
    with app() as connection:
        with connection.cursor() as cursor:
            with pytest.raises(
                pymysql.MySQLError, match=r"\(1644, 'PERMIT persistence requires step 6"
            ):
                cursor.execute(
                    "INSERT INTO rule_decisions "
                    "(request_id,request_hash,request_catalog_hash,required_rule_id,status,"
                    "reason_code,decided_at_utc,canonical_decision,record_hash,audit_sequence) "
                    "VALUES (%s,%s,%s,%s,'PERMIT',NULL,UTC_TIMESTAMP(6),%s,%s,999)",
                    (permit_request.request_id, permit_request.request_hash,
                     _catalog_hash(permit_request.catalogo), permit_request.rule_id,
                     b"{}", ZERO_HASH),
                )
    _enable_permit_seeding(db)

    with db.root() as admin:
        with admin.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM information_schema.KEY_COLUMN_USAGE "
                "WHERE TABLE_SCHEMA='jax_rule_authority' AND TABLE_NAME='rule_permits' "
                "AND CONSTRAINT_NAME='fk_rule_permits_decision'"
            )
            assert cursor.fetchone()[0] == 4
            cursor.execute(
                "SELECT COLUMN_NAME, REFERENCED_COLUMN_NAME FROM "
                "information_schema.KEY_COLUMN_USAGE WHERE TABLE_SCHEMA='jax_rule_authority' "
                "AND TABLE_NAME='rule_permits' AND CONSTRAINT_NAME='fk_rule_permits_decision' "
                "ORDER BY ORDINAL_POSITION"
            )
            assert cursor.fetchall() == (
                ("request_id", "request_id"),
                ("request_hash", "request_hash"),
                ("decision_status", "status"),
                ("rule_id", "required_rule_id"),
            )
            cursor.execute("SELECT @@FOREIGN_KEY_CHECKS")
            assert cursor.fetchone()[0] == 1
            _seed_permit_decision(cursor, permit_request, 100)
            with pytest.raises(pymysql.MySQLError):
                cursor.execute(PERMIT_INSERT, (*values[:3], "a" * 41, *values[4:]))
            with pytest.raises(pymysql.MySQLError):
                cursor.execute(PERMIT_INSERT, (*values[:3], "a" * 40 + "\n", *values[4:]))
            cursor.execute("SAVEPOINT oid40_probe")
            cursor.execute(
                PERMIT_INSERT,
                (*values[:3], "a" * 40, values[4], "b" * 40, "c" * 40, *values[7:]),
            )
            cursor.execute("ROLLBACK TO SAVEPOINT oid40_probe")
            with pytest.raises(pymysql.MySQLError):  # permiso sobre una decisión DENY
                cursor.execute(
                    PERMIT_INSERT, (permit_id, request.request_id, request.request_hash, *values[3:])
                )
            with pytest.raises(pymysql.MySQLError):  # hash ajeno
                cursor.execute(PERMIT_INSERT, (permit_id, permit_request.request_id, ZERO_HASH, *values[3:]))
            with pytest.raises(pymysql.MySQLError):  # regla ajena
                cursor.execute(PERMIT_INSERT.replace("'rule-one'", "'other-rule'", 1), values)
            cursor.execute(PERMIT_INSERT, values)
            with pytest.raises(pymysql.MySQLError):  # consumo con hash ajeno
                cursor.execute(
                    CONSUMPTION_INSERT,
                    (permit_id, "sha256:" + "f" * 64, b"{}", ZERO_HASH, ZERO_HASH, ZERO_HASH, 101),
                )
            cursor.execute(
                CONSUMPTION_INSERT,
                (permit_id, permit_request.request_hash, b"{}", ZERO_HASH, ZERO_HASH, ZERO_HASH, 101),
            )
        admin.commit()
    with pytest.raises(RuleAuthorityStorageError):
        store.get(permit_request)

    with db.root(autocommit=True) as admin:
        with admin.cursor() as cursor:
            queries = (
                ("SELECT request_id FROM rule_decisions WHERE request_id=%s", (request.request_id,)),
                (
                    "SELECT request_id FROM rule_decisions WHERE required_rule_id=%s AND status=%s",
                    (request.rule_id, "DENY"),
                ),
                ("SELECT permit_id FROM rule_permits WHERE expires_at_utc=%s", (datetime.now(timezone.utc),)),
                ("SELECT permit_id FROM rule_permit_consumptions WHERE permit_id=%s", (permit_id,)),
            )
            for query, params in queries:
                cursor.execute("EXPLAIN " + query, params)
                plan = cursor.fetchone()
                assert plan[5] is not None, f"EXPLAIN sin índice: {query}"
                extra = plan[9] or ""
                assert "Using filesort" not in extra and "Using temporary" not in extra


def _grants(admin, username: str, host: str) -> str:
    with admin.cursor() as cursor:
        cursor.execute(f"SHOW GRANTS FOR '{username}'@'{host}'")
        return "\n".join(row[0] for row in cursor.fetchall()).upper()


def _assert_least_privilege(grants: str) -> None:
    assert "GRANT OPTION" not in grants
    assert "ALL PRIVILEGES" not in grants
    assert "TRIGGER" not in grants and "DELETE" not in grants
    for table in IMMUTABLE_TABLES:
        assert f"GRANT SELECT, INSERT ON `JAX_RULE_AUTHORITY`.`{table.upper()}`" in grants
    assert "GRANT SELECT, UPDATE ON `JAX_RULE_AUTHORITY`.`RULE_AUTHORITY_AUDIT_HEAD`" in grants


def _broad_account(admin, username: str, host: str, password: str) -> None:
    with admin.cursor() as cursor:
        cursor.execute(
            f"CREATE USER '{username}'@'{host.replace('%', '%%')}' IDENTIFIED BY %s", (password,)
        )
        cursor.execute(
            f"GRANT ALL PRIVILEGES ON jax_rule_authority.* TO '{username}'@'{host}' "
            "WITH GRANT OPTION"
        )
    admin.commit()


def test_mariadb_provisioning_revokes_prior_grants_for_localhost_and_wildcard_account(db):
    password = secrets.token_urlsafe(24)
    with db.root() as admin:
        _broad_account(admin, APP_USER, "localhost", password)
        _broad_account(admin, APP_USER, "%", password)
        assert "ALL PRIVILEGES" in _grants(admin, APP_USER, "localhost")
        assert "ALL PRIVILEGES" in _grants(admin, APP_USER, "%")
        provision_application_account(admin, APP_USER, password)
        _assert_least_privilege(_grants(admin, APP_USER, "localhost"))
        _assert_least_privilege(_grants(admin, APP_USER, "%"))
        # Repetible: una segunda provisión converge al mismo mínimo.
        provision_application_account(admin, APP_USER, password)
        _assert_least_privilege(_grants(admin, APP_USER, "localhost"))
        _assert_least_privilege(_grants(admin, APP_USER, "%"))


def test_mariadb_provisioning_revokes_every_host_entry_of_the_account(db):
    password = secrets.token_urlsafe(24)
    with db.root() as admin:
        _broad_account(admin, APP_USER, "localhost", password)
        _broad_account(admin, APP_USER, "127.0.0.1", password)
        assert "ALL PRIVILEGES" in _grants(admin, APP_USER, "127.0.0.1")
        provision_application_account(admin, APP_USER, password)
        _assert_least_privilege(_grants(admin, APP_USER, "localhost"))
        _assert_least_privilege(_grants(admin, APP_USER, "127.0.0.1"))
        with admin.cursor() as cursor:
            cursor.execute("SELECT host FROM mysql.user WHERE user=%s ORDER BY host", (APP_USER,))
            assert [row[0] for row in cursor.fetchall()] == ["127.0.0.1", "localhost"]


def test_mariadb_provisioning_without_wildcard_account_creates_none_and_main_runs(db, monkeypatch, capsys):
    from policy.rule_authority import provisioning

    password = secrets.token_urlsafe(24)
    with db.root() as admin:
        _broad_account(admin, APP_USER, "localhost", password)

    monkeypatch.setenv("JAX_RULE_AUTHORITY_ADMIN_UNIX_SOCKET", str(db.socket_path))
    monkeypatch.setenv("JAX_RULE_AUTHORITY_ADMIN_USER", "root")
    monkeypatch.setenv("JAX_RULE_AUTHORITY_ADMIN_PASSWORD", db.root_password)
    monkeypatch.setenv("JAX_RULE_AUTHORITY_APP_USERNAME", APP_USER)
    monkeypatch.setenv("JAX_RULE_AUTHORITY_APP_PASSWORD", password)
    assert provisioning.main() == 0
    assert "provisioned" in capsys.readouterr().out
    with db.root() as admin:
        _assert_least_privilege(_grants(admin, APP_USER, "localhost"))
        with admin.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM mysql.user WHERE User=%s AND Host='%%'", (APP_USER,))
            assert cursor.fetchone()[0] == 0

    monkeypatch.delenv("JAX_RULE_AUTHORITY_APP_PASSWORD")
    with pytest.raises(SystemExit, match="JAX_RULE_AUTHORITY_APP_PASSWORD"):
        provisioning.main()
