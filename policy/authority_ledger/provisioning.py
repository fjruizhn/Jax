"""Provision the least-privilege MariaDB principal for the authority ledger."""
from __future__ import annotations

import os
import re
import ipaddress


_USERNAME = re.compile(r"[A-Za-z0-9_]{1,32}\Z")
_ROLES = frozenset(("reader", "writer"))


def _validated_host(host: str) -> str:
    """Accept localhost compatibility or one exact TCP address, never a pattern."""
    if host == "localhost":
        return host
    if not isinstance(host, str) or not host or any(token in host for token in ("%", "_", "'", "`")):
        raise ValueError("JAX Authority app host inválido")
    try:
        return str(ipaddress.ip_address(host))
    except ValueError as exc:
        raise ValueError("JAX Authority app host debe ser una IP TCP exacta") from exc


def provision_application_account(
    connection, username: str, password: str, *, host: str, role: str,
) -> None:
    """Create/update one Block 4 principal with the exact assigned role.

    The username is constrained before being interpolated as an SQL identifier;
    the host is either legacy ``localhost`` or an exact TCP IP; the password is
    always sent as a bound value and never logged.
    """
    if not isinstance(username, str) or not _USERNAME.fullmatch(username):
        raise ValueError("JAX Authority app username inválido")
    if not isinstance(password, str) or not password:
        raise ValueError("JAX Authority app password requerido")
    if role not in _ROLES:
        raise ValueError("JAX Authority app role inválido")

    validated_host = _validated_host(host)
    principal = f"`{username}`@`{validated_host}`"
    cursor = None
    try:
        cursor = connection.cursor()
        # Bootstrap is intentionally creation-only. An existing identity,
        # including the same user on another host, may have live sessions or
        # inherited/default roles that DCL cannot safely adopt here.
        cursor.execute("SELECT Host,User FROM mysql.global_priv WHERE User=%s", (username,))
        if cursor.fetchall():
            raise ValueError("primer provisionamiento exige principal nuevo y no ambiguo")
        # SHOW GRANTS FOR a user does not include effective PUBLIC privileges.
        # A PUBLIC identity is therefore an unsafe unknown for this narrow
        # bootstrap contract, even if the target account itself is new.
        cursor.execute("SELECT COUNT(*) FROM mysql.global_priv WHERE User='PUBLIC'")
        if cursor.fetchone()[0]:
            raise ValueError("privilegios PUBLIC impiden provisionamiento Block 4")
        # DCL commits implicitly. Lock first so any later failure leaves the
        # principal unable to connect; rollback would not undo CREATE/GRANT.
        cursor.execute(f"CREATE USER {principal} IDENTIFIED BY %s ACCOUNT LOCK", (password,))
        cursor.execute(f"ALTER USER {principal} ACCOUNT LOCK")
        cursor.execute(
            "SELECT COUNT(*) FROM mysql.roles_mapping WHERE User=%s AND Host=%s",
            (username, validated_host),
        )
        if cursor.fetchone()[0]:
            raise ValueError("JAX Authority app tiene roles heredados; requiere reconciliación auditada")
        cursor.execute(f"REVOKE ALL PRIVILEGES, GRANT OPTION FROM {principal}")
        cursor.execute(f"ALTER USER {principal} IDENTIFIED BY %s", (password,))
        cursor.execute(f"GRANT SELECT ON jax_authority.authority_ledger_genesis TO {principal}")
        if role == "writer":
            cursor.execute(f"GRANT SELECT,INSERT ON jax_authority.authority_events TO {principal}")
            cursor.execute(f"GRANT SELECT,UPDATE ON jax_authority.authority_ledger_head TO {principal}")
        else:
            cursor.execute(f"GRANT SELECT ON jax_authority.authority_events TO {principal}")
            cursor.execute(f"GRANT SELECT ON jax_authority.authority_ledger_head TO {principal}")
        cursor.execute(f"SHOW GRANTS FOR {principal}")
        # MariaDB renders comma spacing differently across releases; compare
        # the effective privilege tokens after normalizing that presentation.
        grants = "\n".join(row[0].upper() for row in cursor.fetchall()).replace(", ", ",")
        required = ("GRANT SELECT ON `JAX_AUTHORITY`.`AUTHORITY_LEDGER_GENESIS`",)
        if role == "reader":
            required += (
                "GRANT SELECT ON `JAX_AUTHORITY`.`AUTHORITY_EVENTS`",
                "GRANT SELECT ON `JAX_AUTHORITY`.`AUTHORITY_LEDGER_HEAD`",
            )
        else:
            required += (
                "GRANT SELECT,INSERT ON `JAX_AUTHORITY`.`AUTHORITY_EVENTS`",
                "GRANT SELECT,UPDATE ON `JAX_AUTHORITY`.`AUTHORITY_LEDGER_HEAD`",
            )
        if any(item not in grants for item in required) or "ALL PRIVILEGES" in grants or "GRANT OPTION" in grants:
            raise ValueError("privilegios efectivos Block 4 no cumplen el contrato")
        if role == "reader" and ("INSERT" in grants or "UPDATE" in grants or "DELETE" in grants):
            raise ValueError("reader Block 4 recibió privilegios de escritura")
        if role == "writer" and (
            "GRANT SELECT,INSERT ON `JAX_AUTHORITY`.`AUTHORITY_EVENTS`" not in grants
            or "GRANT SELECT,UPDATE ON `JAX_AUTHORITY`.`AUTHORITY_LEDGER_HEAD`" not in grants
            or "DELETE" in grants
        ):
            raise ValueError("writer Block 4 no cumple privilegios efectivos")
        try:
            cursor.execute(f"ALTER USER {principal} ACCOUNT UNLOCK")
        except Exception as exc:
            # DCL outcome is unknown after an unlock transport/server error.
            # Do not claim that rollback restored the prior lock; an operator
            # must use a new administrative connection before enabling service.
            raise RuntimeError("desbloqueo con resultado desconocido; verificar con nueva conexión admin") from exc
    except Exception:
        # DCL commits implicitly; leave the account locked on every failure.
        # A rollback is only best effort for drivers that buffered non-DCL work.
        try:
            connection.rollback()
        except Exception:  # fail-soft: DCL outcome is unknown; re-raise original failure
            pass
        raise
    finally:
        if cursor is not None:
            cursor.close()


def main() -> int:
    """Run from an administrative host with all connection data in environment."""
    required = ("JAX_AUTHORITY_APP_USERNAME", "JAX_AUTHORITY_APP_PASSWORD",
                "JAX_AUTHORITY_APP_HOST", "JAX_AUTHORITY_APP_ROLE")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit("faltan variables: " + ", ".join(missing))
    if os.environ["JAX_AUTHORITY_APP_HOST"] == "localhost":
        raise SystemExit("JAX_AUTHORITY_APP_HOST debe ser la IP TCP observada")

    import pymysql

    socket = os.environ.get("JAX_AUTHORITY_ADMIN_UNIX_SOCKET")
    if socket:
        admin = {"unix_socket": socket, "user": os.environ.get("JAX_AUTHORITY_ADMIN_USER", ""),
                 "password": os.environ.get("JAX_AUTHORITY_ADMIN_PASSWORD", "")}
    else:
        tcp_required = ("JAX_AUTHORITY_ADMIN_HOST", "JAX_AUTHORITY_ADMIN_PORT",
                        "JAX_AUTHORITY_ADMIN_USER", "JAX_AUTHORITY_ADMIN_PASSWORD")
        tcp_missing = [name for name in tcp_required if not os.environ.get(name)]
        if tcp_missing:
            raise SystemExit("faltan variables admin: " + ", ".join(tcp_missing))
        try:
            admin = {"host": str(ipaddress.ip_address(os.environ["JAX_AUTHORITY_ADMIN_HOST"])),
                     "port": int(os.environ["JAX_AUTHORITY_ADMIN_PORT"]),
                     "user": os.environ["JAX_AUTHORITY_ADMIN_USER"],
                     "password": os.environ["JAX_AUTHORITY_ADMIN_PASSWORD"]}
        except ValueError as exc:
            raise SystemExit("JAX_AUTHORITY_ADMIN_HOST/PORT inválidos") from exc
        if not 1 <= admin["port"] <= 65535:
            raise SystemExit("JAX_AUTHORITY_ADMIN_HOST/PORT inválidos")
    if not admin["user"]:
        raise SystemExit("JAX_AUTHORITY_ADMIN_USER requerido")
    with pymysql.connect(**admin, charset="utf8mb4", autocommit=False) as connection:
        provision_application_account(
            connection,
            os.environ["JAX_AUTHORITY_APP_USERNAME"],
            os.environ["JAX_AUTHORITY_APP_PASSWORD"],
            host=os.environ["JAX_AUTHORITY_APP_HOST"],
            role=os.environ["JAX_AUTHORITY_APP_ROLE"],
        )
    print("JAX Authority application account provisioned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
