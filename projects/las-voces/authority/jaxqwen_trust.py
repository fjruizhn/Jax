"""Host-only, mission-bound credentials for the jaxqwen capability.

The secret and the append-only state directory are host provisioned.  This
module intentionally has no environment fallback, network listener, or
repository-relative default: callers must explicitly provide both.
"""
from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import stat
import time
from pathlib import Path
from typing import Any, Callable

SERVICE_IDENTITY = "jaxqwen"
ISSUER = "las-voces.jaxqwen.host-broker"
VERSION = 1
REQUIRED = frozenset({"service_identity", "capability_id", "mission_id", "project_id", "task_id", "dispatcher_ack_id", "correlation_id", "lease_id", "project_hash", "repository", "branch", "worktree", "allowed_tools", "expiry", "nonce", "issuer", "issuer_version"})
BOUND = REQUIRED - {"expiry", "nonce", "issuer", "issuer_version"}


class TrustError(ValueError):
    pass


def _wire(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


class JaxQwenTrustBroker:
    """A local issuer/verifier.  State survives restart and is never Git state."""
    def __init__(self, secret: bytes, state_dir: Path, *, clock: Callable[[], float] = time.time):
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise TrustError("broker secret must be at least 32 host-provisioned bytes")
        self._secret, self.state_dir, self._clock = secret, state_dir.resolve(), clock
        if self.state_dir.is_symlink():
            raise TrustError("broker state cannot be a symlink")
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not stat.S_ISDIR(self.state_dir.stat().st_mode):
            raise TrustError("broker state is not a directory")
        self._state, self._lock = self.state_dir / "trust.ndjson", self.state_dir / "trust.lock"

    def _locked(self):
        fd = os.open(self._lock, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd); raise TrustError("unsafe broker lock")
        return os.fdopen(fd, "r+")

    def _events(self) -> list[dict[str, Any]]:
        if not self._state.exists(): return []
        if self._state.is_symlink() or not self._state.is_file(): raise TrustError("unsafe broker state")
        try: return [json.loads(line) for line in self._state.read_text(encoding="utf-8").splitlines() if line]
        except json.JSONDecodeError as exc: raise TrustError("malformed broker state") from exc

    def _event(self, kind: str, **fields: Any) -> None:
        row = {"event": kind, **fields}
        # Tokens and MACs are deliberately never accepted in audit fields.
        if {"token", "mac", "secret"} & row.keys(): raise TrustError("secret audit field")
        fd = os.open(self._state, os.O_CREAT | os.O_APPEND | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode): raise TrustError("unsafe broker state")
            os.write(fd, _wire(row) + b"\n"); os.fsync(fd)
        finally: os.close(fd)

    def issue(self, claims: dict[str, Any], *, ttl_seconds: int = 300) -> str:
        if not isinstance(claims, dict) or REQUIRED - set(claims): raise TrustError("incomplete credential claims")
        if claims.get("service_identity") != SERVICE_IDENTITY or claims.get("issuer") != ISSUER or claims.get("issuer_version") != VERSION: raise TrustError("invalid issuer claims")
        if not isinstance(ttl_seconds, int) or not 1 <= ttl_seconds <= 300: raise TrustError("invalid credential ttl")
        body = dict(claims); body["issued_at"] = int(self._clock()); body["expiry"] = body["issued_at"] + ttl_seconds
        mac = hmac.new(self._secret, _wire(body), hashlib.sha256).hexdigest()
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX); self._event("CREDENTIAL_ISSUED", mission_id=body["mission_id"], task_id=body["task_id"], dispatcher_ack_id=body["dispatcher_ack_id"], lease_id=body["lease_id"], nonce=body["nonce"])
        return json.dumps({"body": body, "mac": mac}, sort_keys=True, separators=(",", ":"))

    def verify_and_consume(self, token: Any, expected: dict[str, Any], *, current: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        try:
            packed = json.loads(token) if isinstance(token, str) else None; body = packed["body"]; mac = packed["mac"]
            if not isinstance(body, dict) or not isinstance(mac, str) or set(packed) != {"body", "mac"}: raise TrustError("malformed credential")
        except (TypeError, KeyError, json.JSONDecodeError) as exc: raise TrustError("malformed credential") from exc
        want = hmac.new(self._secret, _wire(body), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(want, mac): raise TrustError("forged credential")
        if REQUIRED - set(body) or body.get("issuer") != ISSUER or body.get("issuer_version") != VERSION or body.get("service_identity") != SERVICE_IDENTITY: raise TrustError("invalid credential identity")
        if not isinstance(body.get("expiry"), int) or body["expiry"] < int(self._clock()): raise TrustError("expired credential")
        if any(body.get(k) != v for k, v in expected.items()): raise TrustError("credential binding mismatch")
        if not current(body): raise TrustError("credential is stale against canonical state")
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX); events = self._events()
            if any(e.get("event") == "SERVICE_REVOKED" and e.get("service_identity") == SERVICE_IDENTITY for e in events): raise TrustError("service identity revoked")
            if any(e.get("event") == "MISSION_REVOKED" and e.get("mission_id") == body["mission_id"] for e in events): raise TrustError("mission credential revoked")
            if any(e.get("event") == "CREDENTIAL_CONSUMED" and e.get("nonce") == body["nonce"] for e in events): raise TrustError("credential replay")
            self._event("CREDENTIAL_ACCEPTED", mission_id=body["mission_id"], task_id=body["task_id"], dispatcher_ack_id=body["dispatcher_ack_id"], lease_id=body["lease_id"], nonce=body["nonce"])
            self._event("CREDENTIAL_CONSUMED", mission_id=body["mission_id"], nonce=body["nonce"])
        return body

    def revoke_mission(self, mission_id: str) -> None:
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX); self._event("MISSION_REVOKED", mission_id=mission_id)

    def revoke_service(self) -> None:
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX); self._event("SERVICE_REVOKED", service_identity=SERVICE_IDENTITY)
