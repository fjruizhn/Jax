"""External, append-only monotonic checkpoint anchor (never MariaDB)."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

from .canonical import canonical_bytes
from .errors import LedgerIntegrityError, LedgerRollbackError
from .models import AuthorityLedgerCheckpoint

DEFAULT_TRUSTED_CHECKPOINT_PATH = Path("/var/lib/jax/authority/trusted-checkpoints.log")


class TrustedCheckpointStore:
    """Append-only JSONL-like canonical checkpoint store; tests may inject a path."""
    def __init__(self, path: Path = DEFAULT_TRUSTED_CHECKPOINT_PATH) -> None:
        self.path = Path(path)

    def append(self, checkpoint: AuthorityLedgerCheckpoint) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        current = self.latest() if self.path.exists() else None
        if current is not None and checkpoint.sequence <= current.sequence:
            raise LedgerRollbackError("checkpoint externo no permite retroceso ni secuencia repetida")
        prior = self.path.read_bytes() if self.path.exists() else b""
        if prior and not prior.endswith(b"\n"):
            raise LedgerIntegrityError("checkpoint externo termina en línea parcial")
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

    def latest(self) -> AuthorityLedgerCheckpoint:
        if not self.path.exists():
            raise LedgerIntegrityError("checkpoint externo ausente")
        import json
        rows = [line for line in self.path.read_bytes().splitlines() if line]
        if not rows:
            raise LedgerIntegrityError("checkpoint externo vacío")
        try:
            data = json.loads(rows[-1].decode("utf-8"))
            return AuthorityLedgerCheckpoint(**data)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise LedgerIntegrityError("checkpoint externo inválido") from exc
