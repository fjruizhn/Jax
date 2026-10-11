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
import secrets
import stat
import threading

from policy.authority_ledger.canonical import canonical_bytes, domain_hash

from .errors import CheckpointDurabilityUnknown, CheckpointInvalido


_DOMAIN = "JAX-FARO-RULE-AUTHORITY-AUDIT-CHECKPOINT"
_VERSION = "1.0"
_FIELDS = frozenset({
    "schema_version", "kind", "sequence", "head", "previous_head",
    "previous_checkpoint_hash", "checkpoint_hash",
})
_GENESIS_HEAD = domain_hash(_DOMAIN, _VERSION, {"kind": "GENESIS", "sequence": 0})
RULE_AUDIT_GENESIS_HEAD = _GENESIS_HEAD
_AUTHORITY_LEDGER_LOCK_TOKEN_SEAL = object()


class _AuthorityLedgerLockToken:
    """Private proof that this store still holds its real Rule Authority flock."""

    def __init__(self, store, authority_ledger_fence, owner_thread_id, epoch, seal) -> None:
        if seal is not _AUTHORITY_LEDGER_LOCK_TOKEN_SEAL:
            raise CheckpointInvalido("token Rule Authority solo puede emitirlo el checkpoint store")
        self._store = store
        self._authority_ledger_fence = authority_ledger_fence
        self._owner_thread_id = owner_thread_id
        self._epoch = epoch
        self._active = True

    def _matches(self, authority_ledger_fence) -> bool:
        return (
            self._active
            and self._authority_ledger_fence is authority_ledger_fence
            and self._store._depth > 0
            and self._store._lock_thread_id == threading.get_ident()
            and self._owner_thread_id == threading.get_ident()
            and self._epoch == self._store._lock_epoch
        )

    def _deactivate(self) -> None:
        self._active = False


class RuleAuditCheckpointStore:
    """Hash-chained checkpoint log with CAS publication and durable reread.

    ``head`` is the Rule Authority audit record hash. Bootstrap explicitly
    anchors genesis as sequence zero; the first audit publication uses that
    genesis head as its predecessor. Normal operation compares this external
    head to the locked MariaDB audit head before every mutation.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._thread_lock = threading.RLock()
        self._depth = 0
        self._lock_handle = None
        self._lock_thread_id = None
        self._lock_epoch = 0
        # A new process cannot know whether the previous process exited after
        # replace but before directory fsync. Require a durable confirmation
        # before exposing or extending the existing log.
        self._durability_unknown = True
        self._durability_unknown_head: str | None = None

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(f".{self.path.name}.lock")

    @contextmanager
    def locked(self):
        """Serialize checkpoint and DB mutations across threads and processes."""
        with self._thread_lock:
            if self._depth == 0:
                self._ensure_durable_directory(self.path.parent)
                flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
                self._lock_handle = os.fdopen(os.open(self.lock_path, flags, 0o600), "r+b")
                try:
                    os.fchmod(self._lock_handle.fileno(), 0o600)
                    fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX)
                    self._cleanup_abandoned_temporaries()
                    self._lock_thread_id = threading.get_ident()
                    self._lock_epoch += 1
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
                        self._lock_thread_id = None

    def issue_authority_ledger_lock_token(self, authority_ledger_fence):
        """Issue a private Block 4 capability only while this real flock is held."""
        if self._depth <= 0 or self._lock_thread_id != threading.get_ident():
            raise CheckpointInvalido("token Block 4 exige flock Rule Authority activo")
        return _AuthorityLedgerLockToken(
            self, authority_ledger_fence, threading.get_ident(), self._lock_epoch,
            _AUTHORITY_LEDGER_LOCK_TOKEN_SEAL,
        )

    def head_actual(self) -> str:
        with self.locked():
            unknown_head = self._durability_unknown_head
            checkpoints = self._read_all()
            candidate = unknown_head or (
                checkpoints[-1]["head"] if self._durability_unknown and checkpoints else None
            )
            if candidate is not None and not self.confirmar(candidate):
                raise CheckpointDurabilityUnknown(
                    "DURABILITY_UNKNOWN: checkpoint no reconciliado"
                )
            if not checkpoints:
                self._durability_unknown = False
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
            try:
                self._replace_log((row,))
            except CheckpointDurabilityUnknown:
                self._durability_unknown_head = _GENESIS_HEAD
                self._durability_unknown = True
                raise
            if self._read_all() != (row,):
                self._durability_unknown_head = _GENESIS_HEAD
                self._durability_unknown = True
                raise CheckpointDurabilityUnknown(
                    "bootstrap no confirmó la relectura exacta durable del log"
                )
            self._durability_unknown = False
            self._durability_unknown_head = None

    def publicar(self, head: str, *, anterior: str) -> None:
        if not _valid_hash(head) or (anterior != "" and not _valid_hash(anterior)):
            raise CheckpointInvalido("head/anterior de checkpoint inválido")
        with self.locked():
            unknown_head = self._durability_unknown_head
            checkpoints_before = self._read_all()
            candidate = unknown_head or (
                checkpoints_before[-1]["head"]
                if self._durability_unknown and checkpoints_before else None
            )
            if candidate is not None and not self.confirmar(candidate):
                raise CheckpointDurabilityUnknown(
                    "DURABILITY_UNKNOWN: publicación bloqueada hasta reconciliar checkpoint"
                )
            if not checkpoints_before:
                self._durability_unknown = False
            checkpoints = checkpoints_before
            if not checkpoints or checkpoints[0]["sequence"] != 0:
                raise CheckpointInvalido("publicación exige bootstrap externo explícito")
            current = checkpoints[-1]
            if head == current["head"]:
                if anterior == (current["previous_head"] or ""):
                    if not self.confirmar(head):
                        raise CheckpointInvalido(
                            "reintento no pudo confirmar la durabilidad del checkpoint"
                        )
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
            expected_checkpoints = (*checkpoints, row)
            try:
                self._replace_log(expected_checkpoints)
            except CheckpointDurabilityUnknown:
                self._durability_unknown_head = head
                self._durability_unknown = True
                raise
            try:
                observed_after_replace = self._read_all()
            except CheckpointInvalido as exc:
                self._durability_unknown_head = head
                self._durability_unknown = True
                raise CheckpointDurabilityUnknown(
                    "publicación no pudo validar el checkpoint tras replace"
                ) from exc
            if observed_after_replace != expected_checkpoints:
                self._durability_unknown_head = head
                self._durability_unknown = True
                raise CheckpointDurabilityUnknown(
                    "publicación no confirmó la relectura exacta tras replace"
                )

    def confirmar(self, head: str) -> bool:
        if not _valid_hash(head):
            return False
        with self.locked():
            was_unknown = self._durability_unknown_head == head
            try:
                identity_observed = self._checkpoint_identity()
                observed = self._read_all()
            except (OSError, CheckpointInvalido) as exc:
                if was_unknown:
                    self._durability_unknown = True
                    raise CheckpointDurabilityUnknown(
                        "DURABILITY_UNKNOWN: no se pudo releer checkpoint incierto"
                    ) from exc
                raise
            if not any(row["head"] == head for row in observed):
                return False
            try:
                identity_before = self._fsync_checkpoint_file()
                self._fsync_directory_chain(self.path.parent)
                reconciled = self._read_all()
                identity_after = self._checkpoint_identity()
                if (identity_observed != identity_before or identity_before != identity_after
                        or reconciled != observed
                        or not any(row["head"] == head for row in reconciled)):
                    raise CheckpointDurabilityUnknown(
                        "identidad/hash del checkpoint cambió durante reconciliación durable"
                    )
            except CheckpointDurabilityUnknown:
                self._durability_unknown_head = head
                self._durability_unknown = True
                raise
            except (OSError, CheckpointInvalido) as exc:
                self._durability_unknown_head = head
                self._durability_unknown = True
                raise CheckpointDurabilityUnknown(
                    "DURABILITY_UNKNOWN: no se pudo confirmar la durabilidad del checkpoint"
                ) from exc
            if self._durability_unknown_head == head:
                self._durability_unknown_head = None
            if reconciled and reconciled[-1]["head"] == head:
                self._durability_unknown = False
            return True

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
        temporary_path = self.path.with_name(
            f".{self.path.name}.tmp-{secrets.token_hex(16)}"
        )
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
        replace_started = False
        try:
            fd = os.open(temporary_path, flags, 0o600)
            with os.fdopen(fd, "wb") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            # An error returned by replace can be ambiguous: the filesystem
            # operation may have changed the name before reporting failure.
            replace_started = True
            os.replace(temporary_path, self.path)
            self._fsync_directory_chain(self.path.parent)
        except (OSError, CheckpointInvalido) as exc:
            try:
                temporary_path.unlink()
            except FileNotFoundError:  # fail-soft: otro limpiador ya retiró el temporal y no queda estado que recuperar
                pass
            except OSError:  # fail-soft: el temporal privado se limpia en la siguiente adquisición del flock
                # A leftover private temp is cleaned on the next flock. It does
                # not change which checkpoint head was durably published.
                pass
            if replace_started:
                raise CheckpointDurabilityUnknown(
                    "resultado de publicación durable desconocido tras replace"
                ) from exc
            raise CheckpointInvalido(
                "no se pudo preparar checkpoint antes de replace"
            ) from exc

    def _fsync_checkpoint_file(self) -> tuple[int, int]:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self.path, flags)
        except OSError as exc:
            raise CheckpointInvalido("archivo de checkpoint no disponible para fsync") from exc
        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise CheckpointInvalido("checkpoint no es un archivo regular durante fsync")
            os.fsync(fd)
            return file_stat.st_dev, file_stat.st_ino
        finally:
            os.close(fd)

    def _checkpoint_identity(self) -> tuple[int, int]:
        try:
            file_stat = os.stat(self.path, follow_symlinks=False)
        except OSError as exc:
            raise CheckpointInvalido("identidad del checkpoint no disponible") from exc
        if not stat.S_ISREG(file_stat.st_mode):
            raise CheckpointInvalido("checkpoint dejó de ser un archivo regular")
        return file_stat.st_dev, file_stat.st_ino

    def _cleanup_abandoned_temporaries(self) -> None:
        """Remove only this store's private, regular publication temp files.

        The caller holds the checkpoint flock, so no cooperating publisher can
        still own a temp file. Reserved 128-bit names and metadata checks avoid
        treating unrelated files or symlinks as abandoned publications.
        """
        prefix = f".{self.path.name}.tmp-"
        for candidate in self.path.parent.iterdir():
            if not candidate.name.startswith(prefix):
                continue
            suffix = candidate.name[len(prefix):]
            if len(suffix) != 32 or any(char not in "0123456789abcdef" for char in suffix):
                continue
            try:
                metadata = candidate.lstat()
                if (not stat.S_ISREG(metadata.st_mode)
                        or stat.S_IMODE(metadata.st_mode) != 0o600
                        or metadata.st_uid != os.getuid()
                        or metadata.st_nlink != 1):
                    continue
                candidate.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise CheckpointInvalido("temporal abandonado de checkpoint no se pudo limpiar") from exc

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


def require_active_authority_ledger_lock_token(token, authority_ledger_fence) -> None:
    """Reject every value except a live capability emitted under this store's flock."""
    if (not isinstance(token, _AuthorityLedgerLockToken)
            or not token._matches(authority_ledger_fence)):
        raise CheckpointInvalido("token Rule Authority inválido o fuera de su flock")
