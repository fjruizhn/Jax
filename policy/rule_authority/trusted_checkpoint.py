"""Durable external anchor for the Rule Authority audit chain.

The MariaDB audit head is mutable transactional state. This file is the separate,
append-only monotonic witness that lets the kernel distinguish a current DB head
from a restored or truncated database copy. It is host-local; flock is not a
distributed lock.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile
import threading

from policy.authority_ledger.canonical import canonical_bytes, domain_hash

from .errors import CheckpointInvalido


_DOMAIN = "JAX-FARO-RULE-AUTHORITY-AUDIT-CHECKPOINT"
_VERSION = "1.0"
_FIELDS = frozenset({
    "schema_version", "kind", "sequence", "head", "previous_head",
    "previous_checkpoint_hash", "checkpoint_hash",
})
_GENESIS_HEAD = domain_hash(_DOMAIN, _VERSION, {"kind": "GENESIS", "sequence": 0})
RULE_AUDIT_GENESIS_HEAD = _GENESIS_HEAD


class RuleAuditCheckpointStore:
    """Hash-chained checkpoint log with CAS publication and durable reread.

    ``head`` is the Rule Authority audit record hash. The first publication uses
    the empty string as its predecessor; normal operation must compare this
    external head to the locked MariaDB audit head before every mutation.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._thread_lock = threading.RLock()
        self._depth = 0
        self._lock_handle = None

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(f".{self.path.name}.lock")

    @contextmanager
    def locked(self):
        """Serialize checkpoint and DB mutations across threads and processes."""
        with self._thread_lock:
            if self._depth == 0:
                self._ensure_durable_directory(self.path.parent)
                self._lock_handle = self.lock_path.open("a+b")
                try:
                    os.fchmod(self._lock_handle.fileno(), 0o600)
                    fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX)
                except Exception:
                    self._lock_handle.close()
                    self._lock_handle = None
                    raise
            self._depth += 1
            try:
                yield self
            finally:
                self._depth -= 1
                if self._depth == 0:
                    try:
                        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
                    finally:
                        self._lock_handle.close()
                        self._lock_handle = None

    def head_actual(self) -> str:
        with self.locked():
            checkpoints = self._read_all()
            return checkpoints[-1]["head"] if checkpoints else ""

    def bootstrap(self) -> None:
        """Create the explicit empty-DB anchor once; never infer it from MariaDB."""
        with self.locked():
            if self.path.exists():
                raise CheckpointInvalido("bootstrap rechazado: checkpoint externo ya existe")
            row = {
                "schema_version": _VERSION,
                "kind": "JAX_RULE_AUTHORITY_AUDIT_CHECKPOINT",
                "sequence": 0,
                "head": _GENESIS_HEAD,
                "previous_head": None,
                "previous_checkpoint_hash": None,
            }
            row["checkpoint_hash"] = _checkpoint_hash(row)
            self._replace_log((row,))
            if not self.confirmar(_GENESIS_HEAD):
                raise CheckpointInvalido("bootstrap no quedó en el log durable")

    def publicar(self, head: str, *, anterior: str) -> None:
        if not _valid_hash(head) or (anterior != "" and not _valid_hash(anterior)):
            raise CheckpointInvalido("head/anterior de checkpoint inválido")
        with self.locked():
            checkpoints = self._read_all()
            if not checkpoints or checkpoints[0]["sequence"] != 0:
                raise CheckpointInvalido("publicación exige bootstrap externo explícito")
            current = checkpoints[-1]
            if head == current["head"]:
                if anterior == (current["previous_head"] or ""):
                    return
                raise CheckpointInvalido("head repetido con anterior distinto")
            if any(row["head"] == head for row in checkpoints):
                raise CheckpointInvalido("checkpoint no permite retroceso")
            if anterior != current["head"]:
                raise CheckpointInvalido("head conflictivo: anterior no coincide con head actual")
            sequence = current["sequence"] + 1
            previous_checkpoint_hash = current["checkpoint_hash"]

            row = {
                "schema_version": _VERSION,
                "kind": "JAX_RULE_AUTHORITY_AUDIT_CHECKPOINT",
                "sequence": sequence,
                "head": head,
                "previous_head": anterior or None,
                "previous_checkpoint_hash": previous_checkpoint_hash,
            }
            row["checkpoint_hash"] = _checkpoint_hash(row)
            self._replace_log((*checkpoints, row))
            if not self.confirmar(head):
                raise CheckpointInvalido("checkpoint publicado no aparece en el log durable")

    def confirmar(self, head: str) -> bool:
        if not _valid_hash(head):
            return False
        with self.locked():
            return any(row["head"] == head for row in self._read_all())

    def checkpoints(self) -> tuple[dict[str, object], ...]:
        with self.locked():
            return tuple(dict(row) for row in self._read_all())

    def _read_all(self) -> tuple[dict[str, object], ...]:
        if not self.path.exists():
            return ()
        try:
            payload = self.path.read_bytes()
        except OSError as exc:
            raise CheckpointInvalido("checkpoint externo ilegible") from exc
        if not payload or not payload.endswith(b"\n"):
            raise CheckpointInvalido("checkpoint externo vacío o termina en fila parcial")
        rows: list[dict[str, object]] = []
        seen_heads: set[str] = set()
        for raw in payload[:-1].split(b"\n"):
            if not raw:
                raise CheckpointInvalido("checkpoint externo contiene fila vacía")
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CheckpointInvalido("checkpoint externo inválido") from exc
            if (type(value) is not dict or frozenset(value) != _FIELDS
                    or canonical_bytes(value) != raw):
                raise CheckpointInvalido("checkpoint externo no es un objeto canónico cerrado")
            expected_sequence = len(rows)
            expected_previous_hash = rows[-1]["checkpoint_hash"] if rows else None
            expected_previous_head = rows[-1]["head"] if rows else None
            if (value["schema_version"] != _VERSION
                    or value["kind"] != "JAX_RULE_AUTHORITY_AUDIT_CHECKPOINT"
                    or type(value["sequence"]) is not int
                    or value["sequence"] != expected_sequence
                    or not _valid_hash(value["head"])
                    or value["previous_head"] != expected_previous_head
                    or value["previous_checkpoint_hash"] != expected_previous_hash
                    or not _valid_hash(value["checkpoint_hash"])
                    or _checkpoint_hash(value) != value["checkpoint_hash"]):
                raise CheckpointInvalido("checkpoint externo rompe secuencia o hash chain")
            if not rows and (value["sequence"] != 0 or value["head"] != _GENESIS_HEAD):
                raise CheckpointInvalido("primera fila debe ser genesis explícito")
            if value["head"] in seen_heads:
                raise CheckpointInvalido("checkpoint externo repite un head histórico")
            seen_heads.add(value["head"])
            rows.append(value)
        return tuple(rows)

    def _replace_log(self, rows: tuple[dict[str, object], ...]) -> None:
        self._ensure_durable_directory(self.path.parent)
        payload = b"".join(canonical_bytes(row) + b"\n" for row in rows)
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            self._fsync_directory_chain(self.path.parent)
        except OSError as exc:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise CheckpointInvalido("resultado de publicación durable desconocido") from exc

    @classmethod
    def _ensure_durable_directory(cls, path: Path) -> None:
        missing = []
        current = path
        while not current.exists():
            missing.append(current)
            parent = current.parent
            if parent == current:
                raise CheckpointInvalido("no existe ancestro para crear directorio de checkpoint")
            current = parent
        if not current.is_dir():
            raise CheckpointInvalido("ruta de checkpoint no es un directorio")
        for directory in reversed(missing):
            try:
                directory.mkdir(mode=0o700)
            except FileExistsError:
                if not directory.is_dir():
                    raise CheckpointInvalido("directorio de checkpoint fue reemplazado")
            cls._fsync_directory(directory.parent)
        if not path.is_dir():
            raise CheckpointInvalido("directorio de checkpoint ausente o reemplazado")

    @classmethod
    def _fsync_directory_chain(cls, path: Path) -> None:
        current = path
        while True:
            if not current.is_dir():
                raise CheckpointInvalido("directorio de checkpoint ausente durante fsync")
            cls._fsync_directory(current)
            parent = current.parent
            if parent == current:
                return
            current = parent

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _valid_hash(value: object) -> bool:
    return type(value) is str and len(value) == 71 and value.startswith("sha256:") and all(
        char in "0123456789abcdef" for char in value[7:]
    )


def _checkpoint_hash(row: dict[str, object]) -> str:
    projection = {key: value for key, value in row.items() if key != "checkpoint_hash"}
    return domain_hash(_DOMAIN, _VERSION, projection)
