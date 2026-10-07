"""Provision the least-privilege MariaDB principal for Rule Authority."""
from __future__ import annotations

import os
import re


_USERNAME = re.compile(r"[A-Za-z0-9_]{1,32}\Z")
_TABLES_APPEND_ONLY = ("rule_decisions", "rule_permits", "rule_permit_consumptions")
# El principal canónico es user@localhost (creado si falta). Si además existe la
# cuenta homónima user@'%', se le aplica el mismo REVOKE/GRANT: una cuenta con el
# mismo nombre y otro host no puede conservar privilegios previos. No se crea.
_HOMONYM_HOST = "%"


def _principal(username: str, host: str) -> str:
    return f"`{username}`@`{host}`"


def _apply_least_privilege(cursor, principal: str) -> None:
    cursor.execute(f"REVOKE ALL PRIVILEGES, GRANT OPTION FROM {principal}")
    for table in _TABLES_APPEND_ONLY:
        cursor.execute(f"GRANT SELECT,INSERT ON jax_rule_authority.{table} TO {principal}")
    cursor.execute(
        f"GRANT SELECT,UPDATE ON jax_rule_authority.rule_authority_audit_head TO {principal}"
    )


def provision_application_account(connection, username: str, password: str) -> None:
    """Create/update the application principal with append-only table grants."""
    if not isinstance(username, str) or not _USERNAME.fullmatch(username):
        raise ValueError("Rule Authority app username inválido")
    if not isinstance(password, str) or not password:
        raise ValueError("Rule Authority app password requerido")

    principal = _principal(username, "localhost")
    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute(f"CREATE USER IF NOT EXISTS {principal} IDENTIFIED BY %s", (password,))
        cursor.execute(f"ALTER USER {principal} IDENTIFIED BY %s", (password,))
        _apply_least_privilege(cursor, principal)
        cursor.execute(
            "SELECT COUNT(*) FROM mysql.user WHERE User=%s AND Host=%s",
            (username, _HOMONYM_HOST),
        )
        if cursor.fetchone()[0]:
            _apply_least_privilege(cursor, _principal(username, _HOMONYM_HOST))
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        if cursor is not None:
            cursor.close()


def main() -> int:
    """Run from an administrative host with all connection data in environment."""
    required = (
        "JAX_RULE_AUTHORITY_ADMIN_UNIX_SOCKET", "JAX_RULE_AUTHORITY_ADMIN_USER",
        "JAX_RULE_AUTHORITY_APP_USERNAME", "JAX_RULE_AUTHORITY_APP_PASSWORD",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit("faltan variables: " + ", ".join(missing))

    import pymysql

    with pymysql.connect(
        unix_socket=os.environ["JAX_RULE_AUTHORITY_ADMIN_UNIX_SOCKET"],
        user=os.environ["JAX_RULE_AUTHORITY_ADMIN_USER"],
        password=os.environ.get("JAX_RULE_AUTHORITY_ADMIN_PASSWORD", ""),
        charset="utf8mb4",
        autocommit=False,
    ) as connection:
        provision_application_account(
            connection,
            os.environ["JAX_RULE_AUTHORITY_APP_USERNAME"],
            os.environ["JAX_RULE_AUTHORITY_APP_PASSWORD"],
        )
    print("Rule Authority application account provisioned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
