"""External root of trust; never sourced from the authority ledger DB."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .errors import TrustedRootMismatchError
from .ids import sha256_id

DEFAULT_TRUSTED_ROOT_PATH = Path("/etc/jax/authority/trusted-root.json")


@dataclass(frozen=True)
class TrustedAuthorityRoot:
    schema_version: str
    kind: str
    ledger_identity: str
    genesis_hash: str
    constitutional_key_id: str
    constitutional_public_key_fingerprint: str

    def __post_init__(self) -> None:
        if (self.schema_version, self.kind, self.ledger_identity) != ("1.0", "JAX_TRUSTED_AUTHORITY_ROOT", "JAX-AUTHORITY-LEDGER/1"):
            raise TrustedRootMismatchError("trusted root inválido")
        sha256_id(self.genesis_hash, "genesis_hash")
        sha256_id(self.constitutional_public_key_fingerprint, "constitutional_public_key_fingerprint")
        if not isinstance(self.constitutional_key_id, str) or not self.constitutional_key_id:
            raise TrustedRootMismatchError("constitutional_key_id inválido")

    @classmethod
    def load(cls, path: Path = DEFAULT_TRUSTED_ROOT_PATH) -> "TrustedAuthorityRoot":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TrustedRootMismatchError("trusted root ilegible") from exc
        if not isinstance(data, dict) or set(data) != {"schema_version", "kind", "ledger_identity", "genesis_hash", "constitutional_key_id", "constitutional_public_key_fingerprint"}:
            raise TrustedRootMismatchError("trusted root cerrado inválido")
        return cls(**data)
