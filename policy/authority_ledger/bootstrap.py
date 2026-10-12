"""Read-only inspection and offline staging for Block 4 first bootstrap.

These helpers deliberately do not initialize MariaDB, create trusted files, or
sign anything.  An operator supplies public material, inspects the live state,
and then uses the existing service boundary only when the runbook permits it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import os
import stat
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes
from .models import AuthorityLedgerGenesis
from .replay import genesis_hash
from .signatures import decode_public_key, public_key_bytes, public_key_fingerprint
from .trusted_checkpoint import (
    DEFAULT_BOOTSTRAP_RECEIPT_PATH,
    DEFAULT_TRUSTED_CHECKPOINT_PATH,
)
from .trusted_root import DEFAULT_TRUSTED_ROOT_PATH, TrustedAuthorityRoot


class BootstrapInspectionStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    STAGED = "STAGED"
    MATERIAL_PRESENT_UNVERIFIED = "MATERIAL_PRESENT_UNVERIFIED"
    RECOVERY = "RECOVERY"
    LEGACY = "LEGACY"
    INCONSISTENT = "INCONSISTENT"
    INSPECTION_UNAVAILABLE = "INSPECTION_UNAVAILABLE"


@dataclass(frozen=True)
class BootstrapInspection:
    status: BootstrapInspectionStatus
    database: dict[str, Any]
    external_artifacts: dict[str, dict[str, Any]]
    detail: str


@dataclass(frozen=True)
class StagedAuthorityBootstrap:
    """Offline public values for a human-reviewed, future initialization."""
    classification: str
    genesis: AuthorityLedgerGenesis
    trusted_root: TrustedAuthorityRoot
    canonical_genesis: bytes
    genesis_hash: str
    constitutional_public_key_fingerprint: str


_TABLES = (
    "authority_ledger_genesis",
    "authority_events",
    "authority_ledger_head",
)


def _artifact(path: Path, *, owner_uid: int | None) -> dict[str, Any]:
    try:
        result = os.lstat(path)
    except FileNotFoundError:
        return {"path": str(path), "state": "ABSENT"}
    except OSError as exc:
        return {"path": str(path), "state": "UNAVAILABLE", "error": type(exc).__name__}
    if not stat.S_ISREG(result.st_mode):
        return {"path": str(path), "state": "NOT_REGULAR", "mode": result.st_mode & 0o777}
    if owner_uid is not None and result.st_uid != owner_uid:
        return {"path": str(path), "state": "UNSAFE", "mode": result.st_mode & 0o777}
    if result.st_mode & 0o022:
        return {"path": str(path), "state": "UNSAFE", "mode": result.st_mode & 0o777}
    return {"path": str(path), "state": "PRESENT", "mode": result.st_mode & 0o777, "size": result.st_size}


def stage_authority_genesis(
    constitutional_public_key: str, constitutional_key_id: str,
) -> StagedAuthorityBootstrap:
    """Derive public bootstrap values without writing or granting authority."""
    public = decode_public_key(constitutional_public_key)
    genesis = AuthorityLedgerGenesis(
        "1.0", "JAX_AUTHORITY_LEDGER_GENESIS", "JAX-AUTHORITY-LEDGER/1",
        "human:fernando", constitutional_key_id, constitutional_public_key,
    )
    calculated_hash = genesis_hash(genesis)
    fingerprint = public_key_fingerprint(public_key_bytes(public))
    root = TrustedAuthorityRoot(
        "1.0", "JAX_TRUSTED_AUTHORITY_ROOT", genesis.ledger_identity,
        calculated_hash, constitutional_key_id, fingerprint,
    )
    return StagedAuthorityBootstrap(
        "NOT_AUTHORITY", genesis, root, canonical_bytes(genesis.projection()),
        calculated_hash, fingerprint,
    )


def _root_projection(root: TrustedAuthorityRoot) -> dict[str, str]:
    return {
        "schema_version": root.schema_version, "kind": root.kind,
        "ledger_identity": root.ledger_identity, "genesis_hash": root.genesis_hash,
        "constitutional_key_id": root.constitutional_key_id,
        "constitutional_public_key_fingerprint": root.constitutional_public_key_fingerprint,
    }


def _open_safe_root_directory(path: Path, owner_uid: int) -> int:
    """Open an existing, caller-owned root directory without following links."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path.parent, flags)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != owner_uid:
            raise ValueError("directorio trusted root no pertenece al owner configurado")
        if metadata.st_mode & 0o022:
            raise ValueError("directorio trusted root permite escritura de grupo/mundo")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _write_root_exclusively(path: Path, root: TrustedAuthorityRoot, *, owner_uid: int) -> None:
    """Publish a root once; never replace existing authority material."""
    payload = canonical_bytes(_root_projection(root)) + b"\n"
    directory_fd = _open_safe_root_directory(path, owner_uid)
    fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600,
                 dir_fd=directory_fd)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(directory_fd)
    except BaseException:
        # Never remove a possibly durable root after a failed publication.
        raise
    finally:
        os.close(directory_fd)


def _load_existing_root(path: Path, *, owner_uid: int) -> TrustedAuthorityRoot:
    directory_fd = _open_safe_root_directory(path, owner_uid)
    metadata = os.lstat(path)
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o022
            or metadata.st_uid != owner_uid):
        os.close(directory_fd)
        raise ValueError("trusted root existente no es archivo seguro; requiere recovery")
    fd = os.open(path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
    try:
        with os.fdopen(fd, "rb") as handle:
            data = json.loads(handle.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("trusted root existente ilegible; requiere recovery") from exc
    finally:
        os.close(directory_fd)
    if not isinstance(data, dict):
        raise ValueError("trusted root existente inválido; requiere recovery")
    return TrustedAuthorityRoot(**data)


def initialize_staged_authority_ledger(
    store, connection_factory, staged: StagedAuthorityBootstrap,
    approved_root: TrustedAuthorityRoot, checkpoint_store, *,
    trusted_root_path: Path = DEFAULT_TRUSTED_ROOT_PATH,
    trusted_root_owner_uid: int,
) -> None:
    """Perform the narrow, recoverable first-init boundary for an exact stage.

    This boundary accepts an already-approved external root.  It does not
    generate keys, signatures, approvals, or receipts; failures after a
    durable write intentionally leave material for recovery inspection.
    """
    if staged.classification != "NOT_AUTHORITY":
        raise ValueError("stage Block 4 no es material de staging")
    recomputed = stage_authority_genesis(
        staged.genesis.constitutional_public_key, staged.genesis.constitutional_key_id,
    )
    if staged != recomputed or approved_root != recomputed.trusted_root:
        raise ValueError("stage/root Block 4 no coinciden exactamente")
    expected = recomputed.canonical_genesis
    connection = connection_factory()
    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT canonical_genesis,genesis_hash FROM jax_authority.authority_ledger_genesis WHERE singleton=1 FOR UPDATE")
        row = cursor.fetchone()
        cursor.execute("SELECT COUNT(*) FROM jax_authority.authority_events")
        events = cursor.fetchone()[0]
        cursor.execute("SELECT sequence,head_event_id,head_event_hash FROM jax_authority.authority_ledger_head WHERE singleton=1 FOR UPDATE")
        head = cursor.fetchone()
        if events or head != (0, None, None):
            raise ValueError("Block 4 no está vacío; requiere recovery")
        if row is None:
            cursor.execute(
                "INSERT INTO jax_authority.authority_ledger_genesis (singleton,canonical_genesis,genesis_hash) VALUES (1,%s,%s)",
                (expected, recomputed.genesis_hash),
            )
            connection.commit()
        elif row != (expected, recomputed.genesis_hash):
            raise ValueError("genesis existente difiere; requiere recovery")
        else:
            connection.rollback()
    except Exception:
        try:
            connection.rollback()
        except Exception:
            pass
        raise
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()
    root_path = Path(trusted_root_path)
    if root_path.exists():
        if _load_existing_root(root_path, owner_uid=trusted_root_owner_uid) != approved_root:
            raise ValueError("trusted root existente difiere; requiere recovery")
    else:
        _write_root_exclusively(root_path, approved_root, owner_uid=trusted_root_owner_uid)
    from .service import initialize_authority_ledger
    from .replay import verify_authority_ledger
    # A complete prior attempt is only retried through the existing verifier.
    # Missing only one external artifact remains a recovery case in that boundary.
    if checkpoint_store.path.exists() and checkpoint_store.bootstrap_receipt_path.exists():
        verify_authority_ledger(store.get_genesis(), store.events(), approved_root, checkpoint_store)
    else:
        initialize_authority_ledger(store, recomputed.genesis, approved_root, checkpoint_store)
        verify_authority_ledger(store.get_genesis(), store.events(), approved_root, checkpoint_store)


def inspect_authority_bootstrap(
    connection, *, trusted_root_path: Path = DEFAULT_TRUSTED_ROOT_PATH,
    checkpoint_path: Path = DEFAULT_TRUSTED_CHECKPOINT_PATH,
    receipt_path: Path = DEFAULT_BOOTSTRAP_RECEIPT_PATH,
    trusted_owner_uid: int | None = None,
) -> BootstrapInspection:
    """Inspect B4 state conservatively; permissions failures are never emptiness."""
    artifacts = {
        "trusted_root": _artifact(Path(trusted_root_path), owner_uid=trusted_owner_uid),
        "checkpoint": _artifact(Path(checkpoint_path), owner_uid=trusted_owner_uid),
        "bootstrap_receipt": _artifact(Path(receipt_path), owner_uid=trusted_owner_uid),
    }
    database: dict[str, Any] = {"schema": "jax_authority", "tables": {}}
    cursor = None
    metadata_visible = False
    found: set[str] = set()
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema=%s AND table_name IN (%s,%s,%s)",
            ("jax_authority", *_TABLES),
        )
        found = {row[0] for row in cursor.fetchall()}
        metadata_visible = True
        database["tables"] = {name: name in found for name in _TABLES}
        # information_schema may hide tables from a limited account. A direct,
        # read-only probe distinguishes visibility denial from an absent schema.
        for table in _TABLES:
            cursor.execute(f"SELECT 1 FROM jax_authority.{table} WHERE 1=0")
        if found != set(_TABLES):
            all_absent = not found
            if all_absent and {value["state"] for value in artifacts.values()} == {"ABSENT"}:
                return BootstrapInspection(BootstrapInspectionStatus.CANDIDATE, database, artifacts,
                                           "schema Block 4 todavía no existe")
            return BootstrapInspection(BootstrapInspectionStatus.RECOVERY, database, artifacts,
                                       "schema Block 4 parcial o material externo preexistente")
        cursor.execute("SELECT canonical_genesis,genesis_hash FROM jax_authority.authority_ledger_genesis WHERE singleton=1")
        genesis = cursor.fetchone()
        cursor.execute("SELECT sequence,head_event_id,head_event_hash FROM jax_authority.authority_ledger_head WHERE singleton=1")
        head = cursor.fetchone()
        cursor.execute("SELECT COUNT(*) FROM jax_authority.authority_events")
        event_count = cursor.fetchone()[0]
        database.update({"genesis_present": genesis is not None, "head": head, "event_count": event_count})
    except Exception as exc:
        code = getattr(exc, "args", (None,))[0]
        # 1049 (unknown database) and 1146 (missing table) mean absence only
        # after a successful metadata query.  1045/1142 and every unknown
        # exception are never allowed to masquerade as an empty ledger.
        if metadata_visible and not found and code in {1049, 1146}:
            if {value["state"] for value in artifacts.values()} == {"ABSENT"}:
                return BootstrapInspection(BootstrapInspectionStatus.CANDIDATE, database, artifacts,
                                           "schema Block 4 ausente y visibilidad administrativa confirmada")
            return BootstrapInspection(BootstrapInspectionStatus.RECOVERY, database, artifacts,
                                       "schema ausente con material externo preexistente")
        database["error"] = type(exc).__name__
        return BootstrapInspection(BootstrapInspectionStatus.INSPECTION_UNAVAILABLE, database, artifacts,
                                   "no se pudo inspeccionar MariaDB Block 4")
    finally:
        if cursor is not None:
            cursor.close()

    external_states = {value["state"] for value in artifacts.values()}
    if head is None or (event_count == 0 and head != (0, None, None)):
        return BootstrapInspection(BootstrapInspectionStatus.INCONSISTENT, database, artifacts,
                                   "head Block 4 no coincide con eventos")
    if event_count and "PRESENT" not in external_states:
        return BootstrapInspection(BootstrapInspectionStatus.LEGACY, database, artifacts,
                                   "eventos sin ancla externa completa")
    if any(state in {"UNAVAILABLE", "NOT_REGULAR", "UNSAFE"} for state in external_states):
        return BootstrapInspection(BootstrapInspectionStatus.RECOVERY, database, artifacts,
                                   "artefacto externo no es verificable")
    if genesis is None and event_count == 0 and external_states == {"ABSENT"}:
        return BootstrapInspection(BootstrapInspectionStatus.CANDIDATE, database, artifacts,
                                   "ledger vacío sin material externo")
    if genesis is not None and event_count == 0 and external_states == {"ABSENT"}:
        return BootstrapInspection(BootstrapInspectionStatus.STAGED, database, artifacts,
                                   "genesis almacenado; faltan anclas externas")
    if genesis is not None and external_states == {"PRESENT"}:
        return BootstrapInspection(BootstrapInspectionStatus.MATERIAL_PRESENT_UNVERIFIED, database, artifacts,
                                   "material presente sin verify completo; autoridad no establecida")
    return BootstrapInspection(BootstrapInspectionStatus.RECOVERY, database, artifacts,
                               "estado parcial; no inicializar ni reparar automáticamente")
