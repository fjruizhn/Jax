"""External, append-only monotonic checkpoint anchor (never MariaDB)."""
from __future__ import annotations

import os
from pathlib import Path

from .canonical import canonical_bytes
from .errors import LedgerIntegrityError
from .models import AuthorityLedgerCheckpoint

DEFAULT_TRUSTED_CHECKPOINT_PATH = Path("/var/lib/jax/authority/trusted-checkpoints.log")


class TrustedCheckpointStore:
    """Append-only JSONL-like canonical checkpoint store; tests may inject a path."""
    def __init__(self, path: Path = DEFAULT_TRUSTED_CHECKPOINT_PATH) -> None:
        self.path = Path(path)

    def append(self, checkpoint: AuthorityLedgerCheckpoint) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(canonical_bytes(checkpoint.projection()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())

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
