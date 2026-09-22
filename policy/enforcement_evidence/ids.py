"""Domain separated identifiers used only by Block 7."""
from __future__ import annotations
import hashlib, re
from typing import Any
from policy.canonicalization.canonical_json import canonical_json_bytes
from .errors import EvidenceBlobHashMismatchError

_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
def sha256_bytes(data: bytes) -> str:
    if not isinstance(data, bytes): raise TypeError("bytes requeridos")
    return "sha256:" + hashlib.sha256(data).hexdigest()
def require_hash(value: object, field: str="hash") -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise EvidenceBlobHashMismatchError(f"{field} inválido")
    return value
def domain_hash(domain: str, value: Any) -> str:
    if not isinstance(domain, str) or not domain.endswith("/1"): raise ValueError("dominio inválido")
    return sha256_bytes(domain.encode("ascii") + b"\0" + canonical_json_bytes(value))
