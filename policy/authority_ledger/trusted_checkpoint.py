"""External, append-only monotonic checkpoint anchor (never MariaDB)."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile
import threading

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
        self._thread_lock = threading.RLock()
        self._lock_depth = 0
        self._lock_handle = None

    def bootstrap(self, checkpoint: AuthorityLedgerCheckpoint, receipt: dict) -> None:
        """Create sequence-zero anchor and its separate one-time receipt exclusively."""
        if checkpoint.sequence != 0 or checkpoint.head_event_id is not None or checkpoint.head_event_hash is not None:
            raise LedgerIntegrityError("bootstrap exige checkpoint genesis de secuencia cero")
        self._ensure_durable_directory(self.path.parent)
        self._ensure_durable_directory(self.bootstrap_receipt_path.parent)
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
            self._fsync_directory_chain(self.path.parent)
            if self.latest().projection() != checkpoint.projection():
                raise LedgerIntegrityError("checkpoint genesis publicado no coincide")
            receipt_fd = os.open(self.bootstrap_receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(receipt_fd, "wb") as handle:
                handle.write(canonical_bytes(receipt) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._fsync_directory_chain(self.bootstrap_receipt_path.parent)
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
        """Hold the checkpoint lock across a transaction; nesting is thread-reentrant."""
        with self._thread_lock:
            if self._lock_depth == 0:
                self._ensure_durable_directory(self.path.parent)
                self._lock_handle = self.lock_path.open("a+b")
                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX)
            self._lock_depth += 1
            try:
                yield self
            finally:
                self._lock_depth -= 1
                if self._lock_depth == 0:
                    try:
                        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
                    finally:
                        self._lock_handle.close()
                        self._lock_handle = None

    def sync_for_verification(self) -> None:
        """Re-establish file and directory durability before issuing authority."""
        if not self.path.is_file():
            raise LedgerIntegrityError("checkpoint externo ausente o ilegible")
        if not self.bootstrap_receipt_path.is_file():
            raise LedgerIntegrityError("recibo bootstrap requerido y ausente/ilegible")
        try:
            self._fsync_file(self.path)
            self._fsync_directory_chain(self.path.parent)
            self._fsync_file(self.bootstrap_receipt_path)
            self._fsync_directory_chain(self.bootstrap_receipt_path.parent)
        except OSError as exc:
            raise LedgerIntegrityError(
                "no se pudo confirmar durabilidad del checkpoint/recibo; autoridad no verificable"
            ) from exc

    def append(self, checkpoint: AuthorityLedgerCheckpoint) -> None:
        with self.locked():
            self._append_locked(checkpoint)

    def _append_locked(self, checkpoint: AuthorityLedgerCheckpoint) -> None:
        """Publish while ``locked()`` is held, then reread the exact record."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Only bootstrap() may create the zero anchor. A normal append must
        # never reconstruct missing external evidence from the DB head.
        checkpoints = self._read_all()
        current = checkpoints[-1]
        if checkpoint.sequence <= current.sequence:
            raise LedgerRollbackError("checkpoint externo no permite retroceso ni secuencia repetida")
        self._validate_checkpoint(checkpoint, current)
        prior = self.path.read_bytes()
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

    def validate_bootstrap_receipt(self, expected: dict) -> None:
        """Require the exact external genesis receipt for every current log format."""
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
            if not checkpoints and checkpoint.sequence != 0:
                raise LedgerIntegrityError(
                    "primera fila del checkpoint debe ser genesis cero; logs legacy requieren adopción explícita"
                )
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
        self._fsync_directory_chain(self.path.parent)

    def _ensure_durable_directory(self, path: Path) -> None:
        """Create missing directory components and persist every parent entry."""
        missing = []
        current = path
        while not current.exists():
            missing.append(current)
            parent = current.parent
            if parent == current:
                raise LedgerIntegrityError("no existe ancestro para crear directorio confiable")
            current = parent
        if not current.is_dir():
            raise LedgerIntegrityError("ruta de directorio confiable no es un directorio")
        for directory in reversed(missing):
            try:
                directory.mkdir()
            except FileExistsError:
                if not directory.is_dir():
                    raise LedgerIntegrityError("componente de directorio confiable fue reemplazado")
            # Persist the new child entry in its parent. Then continue upward
            # so a newly created ancestor itself is durable as well.
            self._fsync_directory(directory.parent)
        if not path.is_dir():
            raise LedgerIntegrityError("directorio confiable ausente o reemplazado")

    def _fsync_directory_chain(self, path: Path) -> None:
        """Persist a directory and every ancestor entry up to the filesystem root."""
        current = path
        while True:
            if not current.is_dir():
                raise LedgerIntegrityError("directorio confiable ausente durante fsync")
            self._fsync_directory(current)
            parent = current.parent
            if parent == current:
                break
            current = parent

    @staticmethod
    def _fsync_file(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def require_trusted_checkpoint_store(value) -> TrustedCheckpointStore:
    """Reject partial adapters before they can represent current authority."""
    required = ("locked", "latest", "checkpoints", "validate_bootstrap_receipt",
                "append", "bootstrap")
    if (not isinstance(value, TrustedCheckpointStore)
            or any(not callable(getattr(value, name, None)) for name in required)):
        raise LedgerIntegrityError(
            "autoridad actual requiere TrustedCheckpointStore completo con checkpoint y recibo externos"
        )
    return value
