"""External, append-only monotonic checkpoint anchor (never MariaDB)."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile

from .canonical import canonical_bytes
from .errors import (CheckpointPublicationOutcomeUnknownError,
                     LedgerIntegrityError, LedgerRollbackError)
from .models import AuthorityLedgerCheckpoint

DEFAULT_TRUSTED_CHECKPOINT_PATH = Path("/var/lib/jax/authority/trusted-checkpoints.log")
DEFAULT_BOOTSTRAP_RECEIPT_PATH = Path("/etc/jax/authority/trusted-checkpoint-bootstrap-receipt.json")


class TrustedCheckpointStore:
    """Append-only checkpoint anchor for one host and its local filesystem.

    All writers that share ``path`` coordinate through the stable sidecar lock.
    This assumes a single host/local filesystem; ``flock`` is not a distributed
    locking protocol.
    """
    def __init__(self, path: Path = DEFAULT_TRUSTED_CHECKPOINT_PATH, *, bootstrap_receipt_path: Path = DEFAULT_BOOTSTRAP_RECEIPT_PATH) -> None:
        self.path = Path(path)
        self.bootstrap_receipt_path = Path(bootstrap_receipt_path)

    def bootstrap(self, checkpoint: AuthorityLedgerCheckpoint, receipt: dict) -> None:
        """Create sequence-zero anchor and its separate one-time receipt exclusively."""
        if checkpoint.sequence != 0 or checkpoint.head_event_id is not None or checkpoint.head_event_hash is not None:
            raise LedgerIntegrityError("bootstrap exige checkpoint genesis de secuencia cero")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.bootstrap_receipt_path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() or self.bootstrap_receipt_path.exists():
            raise LedgerIntegrityError("bootstrap rechazado: checkpoint o recibo ya existe")
        # Create checkpoint without replacement, then durable receipt. A crash
        # between them is intentionally fail-closed and requires artifact recovery.
        checkpoint_created = False
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            checkpoint_created = True
            with os.fdopen(fd, "wb") as handle:
                handle.write(canonical_bytes(checkpoint.projection()) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._fsync_directory(self.path.parent)
            if self.latest().projection() != checkpoint.projection():
                raise LedgerIntegrityError("checkpoint genesis publicado no coincide")
            receipt_fd = os.open(self.bootstrap_receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(receipt_fd, "wb") as handle:
                handle.write(canonical_bytes(receipt) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._fsync_directory(self.bootstrap_receipt_path.parent)
        except Exception as exc:
            # Never delete/overwrite a possibly durable checkpoint or receipt.
            # Once the seq-0 path was created, even a failure while publishing
            # its separate receipt has an outcome that requires matched-backup
            # recovery; do not report it as a clean bootstrap rejection.
            if checkpoint_created and not isinstance(exc, CheckpointPublicationOutcomeUnknownError):
                raise CheckpointPublicationOutcomeUnknownError(
                    "bootstrap publicó checkpoint cero, pero la durabilidad/recibo quedó desconocido; restaurar artifacts emparejados"
                ) from exc
            raise

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(f".{self.path.name}.lock")

    @contextmanager
    def locked(self):
        """Hold the checkpoint writer lock across a complete ledger transaction."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield self
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def append(self, checkpoint: AuthorityLedgerCheckpoint) -> None:
        with self.locked():
            self._append_locked(checkpoint)

    def _append_locked(self, checkpoint: AuthorityLedgerCheckpoint) -> None:
        """Publish while ``locked()`` is held, then reread the exact record."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        checkpoints = self._read_all() if self.path.exists() else ()
        current = checkpoints[-1] if checkpoints else None
        if current is not None and checkpoint.sequence <= current.sequence:
            raise LedgerRollbackError("checkpoint externo no permite retroceso ni secuencia repetida")
        self._validate_checkpoint(checkpoint, current)
        prior = self.path.read_bytes() if self.path.exists() else b""
        payload = prior + canonical_bytes(checkpoint.projection()) + b"\n"
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        try:
            self._fsync_parent_directory()
            published = self.latest()
            if published.projection() != checkpoint.projection():
                raise LedgerIntegrityError("checkpoint externo publicado no coincide con el checkpoint solicitado")
        except Exception as exc:
            raise CheckpointPublicationOutcomeUnknownError(
                "resultado de publicación del checkpoint desconocido después de os.replace"
            ) from exc

    def latest(self) -> AuthorityLedgerCheckpoint:
        return self._read_all()[-1]

    def checkpoints(self) -> tuple[AuthorityLedgerCheckpoint, ...]:
        return self._read_all()

    def validate_bootstrap_receipt(self, expected: dict, has_genesis_checkpoint: bool) -> None:
        """Require the exact one-time receipt iff the external log has seq 0."""
        if not has_genesis_checkpoint:
            if self.bootstrap_receipt_path.exists():
                raise LedgerIntegrityError("recibo bootstrap presente sin checkpoint genesis")
            return
        try:
            raw = self.bootstrap_receipt_path.read_bytes()
        except OSError as exc:
            raise LedgerIntegrityError("recibo bootstrap requerido y ausente/ilegible") from exc
        if not raw.endswith(b"\n"):
            raise LedgerIntegrityError("recibo bootstrap incompleto")
        try:
            value = json.loads(raw[:-1].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LedgerIntegrityError("recibo bootstrap inválido") from exc
        if not isinstance(value, dict) or canonical_bytes(value) + b"\n" != raw or value != expected:
            raise LedgerIntegrityError("recibo bootstrap no coincide con genesis/root/checkpoint")

    def _read_all(self) -> tuple[AuthorityLedgerCheckpoint, ...]:
        if not self.path.exists():
            raise LedgerIntegrityError("checkpoint externo ausente")
        payload = self.path.read_bytes()
        if not payload:
            raise LedgerIntegrityError("checkpoint externo vacío")
        if not payload.endswith(b"\n"):
            raise LedgerIntegrityError("checkpoint externo termina en línea parcial")
        checkpoints = []
        for row in payload[:-1].split(b"\n"):
            if not row:
                raise LedgerIntegrityError("checkpoint externo contiene una fila vacía")
            try:
                data = json.loads(row.decode("utf-8"))
                if not isinstance(data, dict):
                    raise TypeError("fila no es objeto")
                checkpoint = AuthorityLedgerCheckpoint(**data)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
                raise LedgerIntegrityError("checkpoint externo inválido") from exc
            if canonical_bytes(checkpoint.projection()) != row:
                raise LedgerIntegrityError("checkpoint externo no es canónico")
            self._validate_checkpoint(checkpoint, checkpoints[-1] if checkpoints else None)
            checkpoints.append(checkpoint)
        if not checkpoints:
            raise LedgerIntegrityError("checkpoint externo vacío")
        return tuple(checkpoints)

    @staticmethod
    def _validate_checkpoint(checkpoint: AuthorityLedgerCheckpoint, previous: AuthorityLedgerCheckpoint | None) -> None:
        if (checkpoint.schema_version != "1.0"
                or checkpoint.kind != "JAX_AUTHORITY_LEDGER_CHECKPOINT"
                or not isinstance(checkpoint.ledger_identity, str) or not checkpoint.ledger_identity
                or type(checkpoint.sequence) is not int or checkpoint.sequence < 0
                or (checkpoint.sequence == 0 and (checkpoint.head_event_id is not None or checkpoint.head_event_hash is not None))
                or (checkpoint.sequence > 0 and (not isinstance(checkpoint.head_event_id, str) or not checkpoint.head_event_id
                or not isinstance(checkpoint.head_event_hash, str) or not checkpoint.head_event_hash))):
            raise LedgerIntegrityError("checkpoint externo inválido")
        if previous is not None:
            if checkpoint.ledger_identity != previous.ledger_identity:
                raise LedgerIntegrityError("checkpoint externo cambia ledger_identity")
            if checkpoint.sequence <= previous.sequence:
                raise LedgerIntegrityError("checkpoint externo no aumenta estrictamente")

    def _fsync_parent_directory(self) -> None:
        self._fsync_directory(self.path.parent)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
