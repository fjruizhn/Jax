import pymysql
import pytest

import jaxctl.runtime as runtime
from policy.authority_ledger.provisioning import provision_application_account


def test_block4_factory_uses_only_dedicated_closed_environment(monkeypatch):
    called = {}
    monkeypatch.setattr(pymysql, "connect", lambda **kwargs: called.update(kwargs) or object())
    monkeypatch.setenv("JAX_AUTHORITY_DB_HOST", "172.30.5.1")
    monkeypatch.setenv("JAX_AUTHORITY_DB_PORT", "3308")
    monkeypatch.setenv("JAX_AUTHORITY_DB_USER", "authority-reader")
    monkeypatch.setenv("JAX_AUTHORITY_DB_PASSWORD", "test-password")
    monkeypatch.setenv("JAX_DB_HOST", "wrong-general-db")
    runtime._authority_connection_factory()()
    assert called == {
        "host": "172.30.5.1", "port": 3308, "user": "authority-reader",
        "password": "test-password", "database": "jax_authority", "charset": "utf8mb4",
        "autocommit": False, "connect_timeout": 5,
    }


@pytest.mark.parametrize("name, value", [
    ("JAX_AUTHORITY_DB_HOST", "db.example"),
    ("JAX_AUTHORITY_DB_PORT", "0"),
    ("JAX_AUTHORITY_DB_USER", ""),
    ("JAX_AUTHORITY_DB_PASSWORD", ""),
])
def test_block4_factory_fails_closed_without_valid_dedicated_values(monkeypatch, name, value):
    for key, configured in {
        "JAX_AUTHORITY_DB_HOST": "127.0.0.1",
        "JAX_AUTHORITY_DB_PORT": "3308",
        "JAX_AUTHORITY_DB_USER": "authority-reader",
        "JAX_AUTHORITY_DB_PASSWORD": "test-password",
    }.items():
        monkeypatch.setenv(key, configured)
    monkeypatch.setenv(name, value)
    with pytest.raises(runtime.UnavailableSource, match="B4 composition unavailable"):
        runtime._authority_connection_factory()


class _Cursor:
    def __init__(self):
        self.calls = []
        self.closed = False

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def fetchone(self):
        return (0,)

    def fetchall(self):
        if self.calls and "FROM mysql.global_priv WHERE User=%s" in self.calls[-1][0]:
            return []
        return [
            ("GRANT SELECT ON `jax_authority`.`authority_ledger_genesis` TO `authority_reader`@`172.30.5.1`",),
            ("GRANT SELECT ON `jax_authority`.`authority_events` TO `authority_reader`@`172.30.5.1`",),
            ("GRANT SELECT ON `jax_authority`.`authority_ledger_head` TO `authority_reader`@`172.30.5.1`",),
        ]

    def close(self):
        self.closed = True


class _Connection:
    def __init__(self):
        self.cursor_value = _Cursor()
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self.cursor_value

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def test_provision_reader_has_select_only_exact_tcp_principal():
    connection = _Connection()
    provision_application_account(connection, "authority_reader", "password", host="172.30.5.1", role="reader")
    statements = [sql for sql, _ in connection.cursor_value.calls]
    # MariaDB DCL commits implicitly; this path must not pretend a rollback
    # provides security after grants have been issued.
    assert not connection.rolled_back and connection.cursor_value.closed
    principal_statements = [sql for sql in statements if "`authority_reader`@" in sql]
    assert principal_statements and all("172.30.5.1" in sql for sql in principal_statements)
    assert any("GRANT SELECT ON jax_authority.authority_events" in sql for sql in statements)
    assert not any("INSERT ON jax_authority.authority_events" in sql for sql in statements)
    assert not any("UPDATE ON jax_authority.authority_ledger_head" in sql for sql in statements)
    assert all("password" not in sql for sql in statements)


def test_provision_keeps_account_locked_when_effective_grants_are_unsafe():
    class UnsafeCursor(_Cursor):
        def fetchall(self):
            if self.calls and "FROM mysql.global_priv WHERE User=%s" in self.calls[-1][0]:
                return []
            return [("GRANT ALL PRIVILEGES ON *.* TO `authority_reader`@`172.30.5.1`",)]
    class UnsafeConnection(_Connection):
        def __init__(self):
            self.cursor_value = UnsafeCursor()
            self.committed = self.rolled_back = False
    connection = UnsafeConnection()
    with pytest.raises(ValueError, match="privilegios efectivos"):
        provision_application_account(connection, "authority_reader", "password", host="172.30.5.1", role="reader")
    statements = [sql for sql, _ in connection.cursor_value.calls]
    assert any("ACCOUNT LOCK" in sql for sql in statements)
    assert not any("ACCOUNT UNLOCK" in sql for sql in statements)


@pytest.mark.parametrize("host", ["%", "127.0.0.%", "localhost.example", "localhost;drop"])
def test_provision_rejects_nonexact_tcp_hosts(host):
    with pytest.raises(ValueError, match="host"):
        provision_application_account(_Connection(), "authority_writer", "password", host=host, role="writer")


def test_provision_rejects_unknown_role_before_connection_use():
    connection = _Connection()
    with pytest.raises(ValueError, match="role"):
        provision_application_account(connection, "authority_writer", "password", host="127.0.0.1", role="admin")
    assert connection.cursor_value.calls == []


def test_provision_rejects_existing_or_homonymous_principal_before_dcl():
    class ExistingCursor(_Cursor):
        def fetchall(self):
            return [("localhost", "authority_reader")]
    class ExistingConnection(_Connection):
        def __init__(self):
            self.cursor_value = ExistingCursor()
            self.committed = self.rolled_back = False
    connection = ExistingConnection()
    with pytest.raises(ValueError, match="principal nuevo"):
        provision_application_account(connection, "authority_reader", "password", host="172.30.5.1", role="reader")
    assert not any("CREATE USER" in sql for sql, _ in connection.cursor_value.calls)
