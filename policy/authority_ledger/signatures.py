"""Ed25519 authentication used for constitutional human events."""
from __future__ import annotations

import base64
import hashlib

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

from .errors import AuthorityAuthenticationError, AuthorityEventValidationError


def public_key_fingerprint(public_key_bytes: bytes) -> str:
    return "sha256:" + hashlib.sha256(public_key_bytes).hexdigest()


def public_key_bytes(key: Ed25519PublicKey) -> bytes:
    return key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def decode_public_key(encoded: str) -> Ed25519PublicKey:
    try:
        return Ed25519PublicKey.from_public_bytes(base64.b64decode(encoded, validate=True))
    except (ValueError, TypeError) as exc:
        raise AuthorityEventValidationError("constitutional_public_key inválida") from exc


def encode_public_key(key: Ed25519PublicKey) -> str:
    return base64.b64encode(public_key_bytes(key)).decode("ascii")


def sign(private_key: Ed25519PrivateKey, message: bytes) -> str:
    return base64.b64encode(private_key.sign(message)).decode("ascii")


def verify(public_key: Ed25519PublicKey, message: bytes, signature: str) -> None:
    try:
        public_key.verify(base64.b64decode(signature, validate=True), message)
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise AuthorityAuthenticationError("firma Ed25519 inválida") from exc
