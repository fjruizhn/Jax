"""Provision the least-privilege MariaDB principal for the authority ledger."""
from __future__ import annotations

import os
import re


_USERNAME = re.compile(r"[A-Za-z0-9_]{1,32}\Z")


def provision_application_account(connection, username: str, password: str) -> None:
    """Create/update an application account with only ledger append privileges.

    The username is constrained before being interpolated as an SQL identifier;
    the password is always sent as a bound value and never logged.
    """
    if not isinstance(username, str) or not _USERNAME.fullmatch(username):
        raise ValueError("JAX Authority app username inválido")
    if not isinstance(password, str) or not password:
        raise ValueError("JAX Authority app password requerido")

    principal = f"`{username}`@`localhost`"
    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute(f"CREATE USER IF NOT EXISTS {principal} IDENTIFIED BY %s", (password,))
        cursor.execute(f"ALTER USER {principal} IDENTIFIED BY %s", (password,))
        # Converge to the contract: an account that pre-existed with broader
        # privileges (or GRANT OPTION) must not keep them.
        cursor.execute(f"REVOKE ALL PRIVILEGES, GRANT OPTION FROM {principal}")
        cursor.execute(f"GRANT SELECT ON jax_authority.authority_ledger_genesis TO {principal}")
        cursor.execute(f"GRANT SELECT,INSERT ON jax_authority.authority_events TO {principal}")
        cursor.execute(f"GRANT SELECT,UPDATE ON jax_authority.authority_ledger_head TO {principal}")
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
        "JAX_AUTHORITY_ADMIN_UNIX_SOCKET", "JAX_AUTHORITY_ADMIN_USER",
        "JAX_AUTHORITY_APP_USERNAME", "JAX_AUTHORITY_APP_PASSWORD",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit("faltan variables: " + ", ".join(missing))

    import pymysql

    with pymysql.connect(
        unix_socket=os.environ["JAX_AUTHORITY_ADMIN_UNIX_SOCKET"],
        user=os.environ["JAX_AUTHORITY_ADMIN_USER"],
        password=os.environ.get("JAX_AUTHORITY_ADMIN_PASSWORD", ""),
        charset="utf8mb4",
        autocommit=False,
    ) as connection:
        provision_application_account(
            connection,
            os.environ["JAX_AUTHORITY_APP_USERNAME"],
            os.environ["JAX_AUTHORITY_APP_PASSWORD"],
        )
    print("JAX Authority application account provisioned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
