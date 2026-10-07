"""Provision the least-privilege MariaDB principal for Rule Authority."""
from __future__ import annotations

import re


_USERNAME = re.compile(r"[A-Za-z0-9_]{1,32}\Z")


def provision_application_account(connection, username: str, password: str) -> None:
    """Create/update the application principal with append-only table grants."""
    if not isinstance(username, str) or not _USERNAME.fullmatch(username):
        raise ValueError("Rule Authority app username inválido")
    if not isinstance(password, str) or not password:
        raise ValueError("Rule Authority app password requerido")

    principal = f"`{username}`@`localhost`"
    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute(f"CREATE USER IF NOT EXISTS {principal} IDENTIFIED BY %s", (password,))
        cursor.execute(f"ALTER USER {principal} IDENTIFIED BY %s", (password,))
        cursor.execute(f"REVOKE ALL PRIVILEGES, GRANT OPTION FROM {principal}")
        for table in ("rule_decisions", "rule_permits", "rule_permit_consumptions"):
            cursor.execute(f"GRANT SELECT,INSERT ON jax_rule_authority.{table} TO {principal}")
        cursor.execute(
            f"GRANT SELECT,UPDATE ON jax_rule_authority.rule_authority_audit_head TO {principal}"
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        if cursor is not None:
            cursor.close()
