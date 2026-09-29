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
    def __init__(self, secret: bytes, state_dir: Path, *, current_state: Callable[[dict[str, Any]], bool], clock: Callable[[], float] = time.time):
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise TrustError("broker secret must be at least 32 host-provisioned bytes")
        if not callable(current_state): raise TrustError("host current-state verifier is required")
        supplied = Path(os.path.abspath(state_dir))
        if any(part.is_symlink() for part in (supplied, *supplied.parents)):
            raise TrustError("broker state path cannot traverse a symlink")
        self._secret, self.state_dir, self._clock = secret, supplied, clock
        self._current_state = current_state
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not stat.S_ISDIR(self.state_dir.stat().st_mode):
            raise TrustError("broker state is not a directory")
        st = self.state_dir.stat()
        if st.st_uid != os.geteuid() or st.st_mode & 0o077:
            raise TrustError("broker state must be owner-only host state")
        self._state, self._lock = self.state_dir / "trust.ndjson", self.state_dir / "trust.lock"
        marker = self.state_dir / "broker.initialized"
        if marker.exists():
            self._check_regular(marker)
            if not self._state.exists(): raise TrustError("broker replay state is missing; operator reconciliation required")
            self._check_regular(self._state)
        else:
            if self._state.exists(): raise TrustError("broker initialization marker is missing; operator reconciliation required")
            self._create_private(self._state, b"")
            self._create_private(marker, b"jaxqwen-broker-v1\n")

    @staticmethod
    def _check_regular(path: Path) -> None:
        try:
            st = path.lstat()
        except OSError as exc: raise TrustError("broker state is unavailable") from exc
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o077:
            raise TrustError("unsafe broker state file")

    @staticmethod
    def _create_private(path: Path, data: bytes) -> None:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode): raise TrustError("unsafe broker state file")
            if data: os.write(fd, data)
            os.fsync(fd)
        finally: os.close(fd)

    def _locked(self):
        fd = os.open(self._lock, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o077:
            os.close(fd); raise TrustError("unsafe broker lock")
        return os.fdopen(fd, "r+")

    def _events(self) -> list[dict[str, Any]]:
        if not self._state.exists(): raise TrustError("broker replay state is missing; operator reconciliation required")
        try:
            fd = os.open(self._state, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                st = os.fstat(fd)
                if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o077:
                    raise TrustError("unsafe broker state")
                chunks = []
                while part := os.read(fd, 65536): chunks.append(part)
            finally: os.close(fd)
            return [json.loads(line) for line in b"".join(chunks).decode("utf-8").splitlines() if line]
        except json.JSONDecodeError as exc: raise TrustError("malformed broker state") from exc

    def _event(self, kind: str, **fields: Any) -> None:
        row = {"event": kind, "observed_at": int(self._clock()), **fields}
        # Tokens and MACs are deliberately never accepted in audit fields.
        if {"token", "mac", "secret"} & row.keys(): raise TrustError("secret audit field")
        fd = os.open(self._state, os.O_APPEND | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o077: raise TrustError("unsafe broker state")
            os.write(fd, _wire(row) + b"\n"); os.fsync(fd)
        finally: os.close(fd)

    def issue(self, claims: dict[str, Any], *, ttl_seconds: int = 300) -> str:
        if not isinstance(claims, dict) or set(claims) != REQUIRED: raise TrustError("credential claims must have the exact schema")
        if claims.get("service_identity") != SERVICE_IDENTITY or claims.get("issuer") != ISSUER or claims.get("issuer_version") != VERSION: raise TrustError("invalid issuer claims")
        if not isinstance(ttl_seconds, int) or not 1 <= ttl_seconds <= 300: raise TrustError("invalid credential ttl")
        body = dict(claims); body["issued_at"] = int(self._clock()); body["expiry"] = body["issued_at"] + ttl_seconds
        mac = hmac.new(self._secret, _wire(body), hashlib.sha256).hexdigest()
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX); events = self._events()
            if any(e.get("event") == "SERVICE_REVOKED" and e.get("service_identity") == SERVICE_IDENTITY for e in events):
                self._event("CREDENTIAL_REJECTED", reason="service_revoked", mission_id=body["mission_id"], task_id=body["task_id"], dispatcher_ack_id=body["dispatcher_ack_id"], lease_id=body["lease_id"])
                raise TrustError("service identity revoked")
            if any(e.get("event") == "CREDENTIAL_ISSUED" and e.get("nonce") == body["nonce"] for e in events):
                self._event("REPLAY_REJECTED", reason="nonce_reuse", mission_id=body["mission_id"], task_id=body["task_id"], dispatcher_ack_id=body["dispatcher_ack_id"], lease_id=body["lease_id"])
                raise TrustError("nonce reuse")
            if not self._current_state(body):
                self._event("CREDENTIAL_REJECTED", reason="stale_at_issue", mission_id=body["mission_id"], task_id=body["task_id"], dispatcher_ack_id=body["dispatcher_ack_id"], lease_id=body["lease_id"])
                raise TrustError("credential is stale against canonical state")
            self._event("CREDENTIAL_ISSUED", mission_id=body["mission_id"], task_id=body["task_id"], dispatcher_ack_id=body["dispatcher_ack_id"], lease_id=body["lease_id"], nonce=body["nonce"])
        return json.dumps({"body": body, "mac": mac}, sort_keys=True, separators=(",", ":"))

    def verify_and_consume(self, token: Any, expected: dict[str, Any]) -> dict[str, Any]:
        def reject(reason: str, message: str, body: dict[str, Any] | None = None):
            identity = body if isinstance(body, dict) else expected
            event = "REPLAY_REJECTED" if reason == "replay" else "EXPIRED_REJECTED" if reason == "expired" else "CREDENTIAL_REJECTED"
            fields = {k: identity[k] for k in ("mission_id", "task_id", "dispatcher_ack_id", "lease_id") if isinstance(identity.get(k), str)}
            fields.update(reason=reason, credential_digest=hashlib.sha256(token.encode("utf-8", "replace") if isinstance(token, str) else b"non-text-credential").hexdigest())
            with self._locked() as lock:
                fcntl.flock(lock, fcntl.LOCK_EX); self._event(event, **fields)
            raise TrustError(message)
        try:
            packed = json.loads(token) if isinstance(token, str) else None; body = packed["body"]; mac = packed["mac"]
            if not isinstance(body, dict) or not isinstance(mac, str) or set(packed) != {"body", "mac"}: raise TrustError("malformed credential")
        except (TypeError, KeyError, json.JSONDecodeError) as exc: reject("malformed", "malformed credential")
        if set(expected) != BOUND: reject("invalid_verifier_binding", "host verifier binding schema is invalid", body)
        if set(body) != (REQUIRED | {"issued_at"}): reject("invalid_schema", "invalid credential schema", body)
        want = hmac.new(self._secret, _wire(body), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(want, mac): reject("forged", "forged credential", expected)
        if body.get("issuer") != ISSUER or body.get("issuer_version") != VERSION or body.get("service_identity") != SERVICE_IDENTITY: reject("identity", "invalid credential identity", body)
        if not isinstance(body.get("expiry"), int) or not isinstance(body.get("issued_at"), int) or body["expiry"] <= int(self._clock()): reject("expired", "expired credential", body)
        if any(body.get(k) != v for k, v in expected.items()): reject("binding", "credential binding mismatch", body)
        if not self._current_state(body): reject("stale", "credential is stale against canonical state", body)
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX); events = self._events()
            bound = {k: body[k] for k in ("mission_id", "task_id", "dispatcher_ack_id", "lease_id")}
            if body["issued_at"] > int(self._clock()) or body["expiry"] <= int(self._clock()):
                self._event("EXPIRED_REJECTED", **bound, nonce=body["nonce"]); raise TrustError("expired credential")
            if not self._current_state(body):
                self._event("CREDENTIAL_REJECTED", **bound, reason="stale_at_consume", nonce=body["nonce"]); raise TrustError("credential is stale against canonical state")
            if any(e.get("event") == "SERVICE_REVOKED" and e.get("service_identity") == SERVICE_IDENTITY for e in events):
                self._event("CREDENTIAL_REJECTED", **bound, reason="service_revoked", nonce=body["nonce"]); raise TrustError("service identity revoked")
            if any(e.get("event") == "MISSION_REVOKED" and e.get("mission_id") == body["mission_id"] for e in events):
                self._event("CREDENTIAL_REJECTED", **bound, reason="mission_revoked", nonce=body["nonce"]); raise TrustError("mission credential revoked")
            if any(e.get("event") == "CREDENTIAL_CONSUMED" and e.get("nonce") == body["nonce"] for e in events):
                self._event("REPLAY_REJECTED", **bound, reason="replay", nonce=body["nonce"])
                raise TrustError("credential replay")
            self._event("CREDENTIAL_ACCEPTED", **bound, nonce=body["nonce"])
            self._event("CREDENTIAL_CONSUMED", **bound, nonce=body["nonce"])
        return body

    def revoke_mission(self, mission_id: str, *, task_id: str, dispatcher_ack_id: str, lease_id: str) -> None:
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX); self._event("MISSION_REVOKED", mission_id=mission_id, task_id=task_id, dispatcher_ack_id=dispatcher_ack_id, lease_id=lease_id)

    def revoke_service(self) -> None:
        with self._locked() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX); self._event("SERVICE_REVOKED", service_identity=SERVICE_IDENTITY)
