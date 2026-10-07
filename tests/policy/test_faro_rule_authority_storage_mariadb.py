"""MariaDB integration for Faro F1.1's durable decision store."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import secrets
import shlex
import subprocess
import tempfile
import time
import uuid

import pymysql
import pytest

from jax.faro.catalogo_topes import cargar_catalogo_bytes
from policy.authority_ledger.errors import AuthorityStateError
from policy.rule_authority.errors import RuleAuthorityError
from policy.rule_authority.models import (
    RuleDecision,
    RuleDecisionStatus,
    RuleEvaluationRequest,
)


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "policy/rule_authority/migrations/001_rule_authority_kernel.sql"
IMAGE = os.environ.get("JAX_RULE_AUTHORITY_TEST_MARIADB_IMAGE", "mariadb:12.3.3")
CATALOG = cargar_catalogo_bytes((ROOT / "policy/faro/catalogo-topes.json").read_bytes())


def _run(docker: list[str], *args: str):
    return subprocess.run([*docker, *args], check=True, capture_output=True, text=True)


def _apply_migration(connection) -> None:
    delimiter = ";"
    statement = []
    sql = MIGRATION.read_text(encoding="utf-8")
    with connection.cursor() as cursor:
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
                cursor.execute(joined[:-len(delimiter)].strip())
                statement = []
    assert not statement, "migración dejó SQL sin terminar"
    connection.commit()


def _request(request_id: str, recipient: str = "client@example.test") -> RuleEvaluationRequest:
    return RuleEvaluationRequest(
        request_id=request_id,
        rule_id="rule-one",
        subject="actor:fernando",
        capability="mail.send",
        objective="notify-client",
        resource_id="message:invoice-42",
        arguments={"recipient": recipient, "body": "Invoice ready"},
        catalogo=CATALOG,
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


def test_mariadb_decision_store_is_atomic_idempotent_and_request_bound():
    assert MIGRATION.is_file(), "falta la migración MariaDB de Rule Authority"
    from policy.rule_authority.provisioning import provision_application_account
    from policy.rule_authority.storage import MariaDBRuleDecisionStore, RuleAuthorityStorageError

    docker = shlex.split(os.environ.get("JAX_RULE_AUTHORITY_DOCKER_CMD", "docker"))
    root_password = secrets.token_urlsafe(24)
    app_password = secrets.token_urlsafe(24)
    trigger_password = secrets.token_urlsafe(24)
    container = f"jax-rule-authority-{uuid.uuid4().hex[:10]}"

    with tempfile.TemporaryDirectory(prefix="jax-rule-authority-socket-") as socket_dir:
        _run(
            docker, "run", "-d", "--rm", "--network", "none", "--name", container,
            "--mount", f"type=bind,source={socket_dir},target=/run/mysqld",
            "-e", f"MARIADB_ROOT_PASSWORD={root_password}", IMAGE,
        )
        try:
            socket_path = Path(socket_dir) / "mysqld.sock"
            for _ in range(90):
                probe = subprocess.run(
                    [*docker, "exec", container, "test", "-S", "/run/mysqld/mysqld.sock"],
                    capture_output=True, text=True, check=False,
                )
                if probe.returncode == 0:
                    break
                time.sleep(1)
            else:
                pytest.fail("MariaDB no creó su socket Unix")
            _run(docker, "exec", "--user=root", container, "chmod", "0755", "/run/mysqld")

            admin = None
            last_error = None
            for _ in range(90):
                try:
                    admin = pymysql.connect(
                        unix_socket=str(socket_path), user="root", password=root_password,
                        autocommit=False, charset="utf8mb4",
                    )
                    break
                except (OSError, pymysql.MySQLError) as exc:
                    last_error = exc
                    time.sleep(1)
            assert admin is not None, f"MariaDB no aceptó conexiones: {last_error}"

            with admin:
                with admin.cursor() as cursor:
                    cursor.execute("SELECT VERSION()")
                    assert cursor.fetchone()[0].startswith("12.3.3-")
                _apply_migration(admin)
                provision_application_account(admin, "jax_rule_authority_app_test", app_password)
                with admin.cursor() as cursor:
                    cursor.execute(
                        "CREATE USER 'jax_rule_authority_trigger_test'@'localhost' "
                        "IDENTIFIED BY %s",
                        (trigger_password,),
                    )
                    for table in ("rule_decisions", "rule_permits", "rule_permit_consumptions"):
                        cursor.execute(
                            f"GRANT SELECT,UPDATE,DELETE ON jax_rule_authority.{table} "
                            "TO 'jax_rule_authority_trigger_test'@'localhost'"
                        )
                admin.commit()

            def connect_as(username: str, password: str, *, autocommit: bool):
                return pymysql.connect(
                    unix_socket=str(socket_path), user=username, password=password,
                    database="jax_rule_authority", autocommit=autocommit,
                    charset="utf8mb4",
                )

            def app_connect():
                # Explicit BEGIN in the adapter is required: autocommit makes the
                # partial-write assertion kill the no-transaction mutant.
                return connect_as("jax_rule_authority_app_test", app_password, autocommit=True)

            request = _request("0199f8a1-8c00-7000-8000-000000000101")
            store = MariaDBRuleDecisionStore(app_connect)
            assert store.get(request) is None
            assert store.get(_request(request.request_id, "other@example.test")) is None

            decision = _decision(request)
            assert store.record(request, decision) == decision

            # A separate connection sees the row immediately after return: durable
            # persistence is part of the synchronous return contract.
            with app_connect() as independent:
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
            changed = _request(request.request_id, "other@example.test")
            assert store.get(changed) is None
            with pytest.raises(AuthorityStateError):
                store.record(changed, _decision(changed))

            missing_request = _request("0199f8a1-8c00-7000-8000-000000000104")
            missing = _decision(missing_request, status=RuleDecisionStatus.MISSING_RULE)
            assert store.record(missing_request, missing) == missing
            assert store.get(missing_request) == missing
            with app_connect() as independent:
                with independent.cursor() as cursor:
                    cursor.execute(
                        "SELECT status,reason_code FROM rule_decisions WHERE request_id=%s",
                        (missing_request.request_id,),
                    )
                    assert cursor.fetchone() == ("MISSING_RULE", "RULE_NOT_FOUND")

            # Fail after INSERT but before the audit-head update can commit. With
            # autocommit enabled above, this proves the adapter owns one transaction.
            with connect_as("root", root_password, autocommit=False) as admin:
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
            with connect_as("root", root_password, autocommit=True) as admin:
                with admin.cursor() as cursor:
                    cursor.execute("DROP TRIGGER fail_rule_authority_head_update")
                    cursor.execute(
                        "SELECT COUNT(*) FROM rule_decisions WHERE request_id=%s",
                        (failed_request.request_id,),
                    )
                    assert cursor.fetchone()[0] == 0

            permit_request = _request("0199f8a1-8c00-7000-8000-000000000103")
            permit = _decision(permit_request, status=RuleDecisionStatus.PERMIT)
            with pytest.raises(RuleAuthorityError):
                store.record(permit_request, permit)
            assert store.get(permit_request) is None

            # An unavailable database is an error, never a returned PERMIT.
            def unavailable():
                raise pymysql.err.OperationalError(2003, "database unavailable")

            with pytest.raises(RuleAuthorityError):
                MariaDBRuleDecisionStore(unavailable).record(permit_request, permit)

            # Seed placeholder records for every immutable table, then exercise the
            # triggers with a principal that does have UPDATE and DELETE grants.
            zero_hash = "sha256:" + "0" * 64
            permit_id = "0199f8a1-8c00-7000-8000-000000000111"
            with connect_as("root", root_password, autocommit=False) as admin:
                with admin.cursor() as cursor:
                    cursor.execute(
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
                        "DATE_ADD(UTC_TIMESTAMP(6),INTERVAL 5 MINUTE),%s,%s,%s,%s,100)",
                        (permit_id, request.request_id, request.request_hash, "a" * 40,
                         zero_hash, "b" * 40, "c" * 40, zero_hash,
                         "0199f8a1-8c00-7000-8000-000000000112", b"{}", b"{}", b"{}",
                         zero_hash, b"{}", zero_hash, zero_hash),
                    )
                    cursor.execute(
                        "INSERT INTO rule_permit_consumptions "
                        "(permit_id,request_hash,consumed_at_utc,canonical_payload,"
                        "consumption_hash,previous_record_hash,record_hash,audit_sequence) "
                        "VALUES (%s,%s,UTC_TIMESTAMP(6),%s,%s,%s,%s,101)",
                        (permit_id, request.request_hash, b"{}", zero_hash, zero_hash, zero_hash),
                    )
                admin.commit()

            trigger = connect_as(
                "jax_rule_authority_trigger_test", trigger_password, autocommit=False
            )
            with trigger:
                for table, key, value in (
                    ("rule_decisions", "request_id", request.request_id),
                    ("rule_permits", "permit_id", permit_id),
                    ("rule_permit_consumptions", "permit_id", permit_id),
                ):
                    with trigger.cursor() as cursor:
                        with pytest.raises(pymysql.MySQLError):
                            cursor.execute(
                                f"UPDATE {table} SET audit_sequence=audit_sequence+1 "
                                f"WHERE {key}=%s",
                                (value,),
                            )
                        trigger.rollback()
                        with pytest.raises(pymysql.MySQLError):
                            cursor.execute(f"DELETE FROM {table} WHERE {key}=%s", (value,))
                        trigger.rollback()

            with connect_as("root", root_password, autocommit=True) as admin:
                with admin.cursor() as cursor:
                    queries = (
                        (
                            "SELECT request_id FROM rule_decisions WHERE request_id=%s",
                            (request.request_id,),
                        ),
                        (
                            "SELECT request_id FROM rule_decisions "
                            "WHERE required_rule_id=%s AND status=%s",
                            (request.rule_id, "DENY"),
                        ),
                        (
                            "SELECT permit_id FROM rule_permits WHERE expires_at_utc=%s",
                            (datetime.now(timezone.utc),),
                        ),
                        (
                            "SELECT permit_id FROM rule_permit_consumptions WHERE permit_id=%s",
                            (permit_id,),
                        ),
                    )
                    for query, params in queries:
                        cursor.execute("EXPLAIN " + query, params)
                        plan = cursor.fetchone()
                        assert plan[5] is not None, f"EXPLAIN sin índice: {query}"
                        extra = plan[9] or ""
                        assert "Using filesort" not in extra and "Using temporary" not in extra
        finally:
            subprocess.run(
                [*docker, "stop", container], capture_output=True, text=True, check=False
            )
            chown = subprocess.run(
                ["sudo", "-n", "chown", "-R", f"{os.getuid()}:{os.getgid()}", socket_dir],
                capture_output=True, text=True, check=False,
            )
            if chown.returncode and __import__("sys").exc_info()[0] is None:
                raise RuntimeError(
                    f"falló la limpieza del socket MariaDB temporal: {chown.stderr.strip()}"
                )
