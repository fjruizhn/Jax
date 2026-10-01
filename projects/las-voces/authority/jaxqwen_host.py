"""Privileged host composition and AF_UNIX M2M boundary for jaxqwen.

The dispatcher can submit only an existing ACK idempotency key over a local
socket. Kernel peer credentials identify the dedicated dispatcher account;
the capability independently validates the unique DISPATCHED ACK, live lease,
canonical hash, repository, branch and workspace before issuing a short-lived
single-use credential. No request supplies identity, claims, verifier, model,
tools, command, cwd or environment.
"""
from __future__ import annotations

import asyncio
import fcntl
import importlib.util
import json
import logging
import os
import pwd
import socket
import stat
import struct
import sys
from pathlib import Path
from typing import Any


_MAX_FRAME = 8192
_SYSTEMD_CREDENTIALS_DIRECTORY = "/run/credentials/jaxqwen.service"
_TRUST_CREDENTIAL_NAME = "jaxqwen-trust.key"
_TRUST_CREDENTIAL_MODE_ROOT = 0o440
_TRUST_CREDENTIAL_MODE_SERVICE = 0o400
_MAX_TRUST_KEY_BYTES = 4096
log = logging.getLogger("las_voces.jaxqwen_host")


def _validate_credential_directory(st) -> None:
    """Require a root-controlled systemd credential directory."""
    if (not stat.S_ISDIR(st.st_mode) or st.st_uid != 0
            or stat.S_IMODE(st.st_mode) & 0o022):
        raise ValueError("untrusted systemd credential directory")


def _validate_trust_credential(st, *, service_uid: int | None = None,
                               credential_mount_read_only: bool = False) -> None:
    """Accept only the observed root copy or systemd's per-service copy."""
    mode = stat.S_IMODE(st.st_mode)
    root_owned = (st.st_uid == 0 and st.st_gid == 0
                  and mode == _TRUST_CREDENTIAL_MODE_ROOT)
    service_owned = (credential_mount_read_only and service_uid is not None and st.st_uid == service_uid
                     and mode == _TRUST_CREDENTIAL_MODE_SERVICE)
    if not stat.S_ISREG(st.st_mode) or not (root_owned or service_owned) or st.st_nlink != 1:
        raise ValueError("untrusted systemd trust credential")


def _fstat_trust_credential(fd: int):
    return os.fstat(fd)


def _read_trust_credential(directory_fd: int) -> bytes:
    """Read the fixed credential name relative to an already-open trusted dir."""
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    fd = os.open(_TRUST_CREDENTIAL_NAME, flags, dir_fd=directory_fd)
    try:
        before = _fstat_trust_credential(fd)
        mount_flags = os.fstatvfs(fd).f_flag
        credential_mount_read_only = bool(mount_flags & os.ST_RDONLY)
        _validate_trust_credential(before, service_uid=os.geteuid(),
                                   credential_mount_read_only=credential_mount_read_only)
        if not 32 <= before.st_size <= _MAX_TRUST_KEY_BYTES:
            raise ValueError("invalid systemd trust credential length")
        chunks = []
        remaining = _MAX_TRUST_KEY_BYTES + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        secret = b"".join(chunks)
        after = _fstat_trust_credential(fd)
        stable = ("st_dev", "st_ino", "st_uid", "st_gid", "st_mode", "st_nlink", "st_size")
        if any(getattr(before, key) != getattr(after, key) for key in stable):
            raise ValueError("systemd trust credential changed while reading")
        if len(secret) != before.st_size or not 32 <= len(secret) <= _MAX_TRUST_KEY_BYTES:
            raise ValueError("invalid systemd trust credential length")
        return secret
    finally:
        os.close(fd)


def _open_systemd_credential_directory(root_fd: int) -> int:
    """Walk the fixed system-unit path without following links or trusting parents."""
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = os.dup(root_fd)
    try:
        _validate_credential_directory(os.fstat(fd))
        for component in ("run", "credentials", "jaxqwen.service"):
            child = os.open(component, directory_flags, dir_fd=fd)
            os.close(fd)
            fd = child
            _validate_credential_directory(os.fstat(fd))
        return fd
    except BaseException:
        os.close(fd)
        raise


def _systemd_trust_key() -> bytes:
    """Read only the fixed LoadCredential entry for this systemd unit.

    CREDENTIALS_DIRECTORY is systemd's credential locator, not a secret or a
    caller-supplied file path. The fixed unit directory and fixed basename
    prevent selecting an arbitrary readable file.
    """
    if os.environ.get("CREDENTIALS_DIRECTORY") != _SYSTEMD_CREDENTIALS_DIRECTORY:
        raise ValueError("systemd trust credential directory is unavailable")

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    root_fd = os.open("/", directory_flags)
    fd = None
    try:
        fd = _open_systemd_credential_directory(root_fd)
        return _read_trust_credential(fd)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(root_fd)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader: raise RuntimeError("governed jaxqwen module unavailable")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally: sys.dont_write_bytecode = old
    return module


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out: raise ValueError("duplicate request key")
        out[key] = value
    return out


class JaxQwenHost:
    """Dedicated local service. Construction binds every verifier/config value."""
    def __init__(self, *, root: Path, canonical_root: Path, handoff_state_dir: Path, source_worktree_root: Path, workspace_root: Path,
                 mission_state_dir: Path, trust_state_dir: Path,
                 dispatch_socket: Path, dispatcher_uid: int, dispatch_gid: int,
                 model_socket: Path, model_socket_uid: int, model_socket_gid: int,
                 model: str, max_output_tokens: int = 4096,
                 dispatch_enabled: bool = False, service_identity_verifier=None):
        self.root, self.canonical_root, self.handoff_state_dir = root.resolve(), canonical_root.resolve(), handoff_state_dir.resolve()
        self.source_worktree_root, self.workspace_root, self.mission_state_dir = source_worktree_root, workspace_root, mission_state_dir
        self.dispatch_socket, self.dispatcher_uid, self.dispatch_gid = dispatch_socket, dispatcher_uid, dispatch_gid
        self.model_socket, self.model_socket_uid, self.model_socket_gid = model_socket, model_socket_uid, model_socket_gid
        self.model, self.max_output_tokens = model, max_output_tokens
        identity = service_identity_verifier() if service_identity_verifier else pwd.getpwuid(os.geteuid()).pw_name
        if identity != "jaxqwen": raise ValueError("host must run as the dedicated jaxqwen service identity")
        self.dispatch_enabled = dispatch_enabled
        if dispatcher_uid in {0, os.geteuid()} or dispatch_gid < 1:
            raise ValueError("dispatcher must be a separate dedicated service identity")
        if not dispatch_socket.is_absolute() or not model_socket.is_absolute(): raise ValueError("socket paths must be absolute")
        secret = _systemd_trust_key()
        # Execute only the installed/read-only canonical host implementation;
        # the dispatcher checkout is data and ACK state, never service code.
        authority = self.canonical_root / "projects/las-voces/authority"
        self.capability = _load("jaxqwen_host_capability", authority / "jaxqwen_capability.py")
        self.trust = _load("jaxqwen_host_trust", authority / "jaxqwen_trust.py")
        self.transport_module = _load("jaxqwen_host_transport", authority / "jaxqwen_transport.py")
        self._holder = {}
        self._broker = self.trust.JaxQwenTrustBroker(secret, trust_state_dir,
                         current_state=lambda claims: self._holder["capability"]._broker_claims_current(claims))
        self._operator_token = object()
        self._holder["capability"] = self.capability.JaxQwenCapability(
            self.root, self.handoff_state_dir, workspace_root, canonical_root=self.canonical_root, trust_broker=self._broker,
            source_worktree_root=source_worktree_root, host_state_dir=mission_state_dir,
            operator_verifier=lambda proof: proof is self._operator_token)
        self._server = None
        self._active: dict[str, Any] = {}
        self._active_lock = asyncio.Lock()

    @staticmethod
    def _peer_credentials(writer):
        sock = writer.get_extra_info("socket")
        raw = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        return struct.unpack("3i", raw)

    async def _reply(self, writer, value: dict[str, Any]):
        writer.write(json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        await writer.drain()

    async def _serve_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer_uid = None
        try:
            _pid, peer_uid, _gid = self._peer_credentials(writer)
            line = await asyncio.wait_for(reader.readline(), timeout=3.0)
            if not line.endswith(b"\n") or len(line) > _MAX_FRAME: raise ValueError("invalid request frame")
            request = json.loads(line.decode("utf-8"), object_pairs_hook=_pairs)
            if not isinstance(request, dict) or "action" not in request: raise ValueError("invalid request")
            if peer_uid == self.dispatcher_uid:
                if set(request) != {"action", "idempotency_key"} or request["action"] != "dispatch": raise ValueError("dispatcher request schema rejected")
                if not self.dispatch_enabled:
                    await self._reply(writer, {"decision": "HUMAN_REQUIRED", "reason": "first live dispatch gate is disabled"})
                    return
                key = request["idempotency_key"]
                transport = self.transport_module.JaxQwenUnixModelTransport(
                    self.model_socket, expected_proxy_uid=self.model_socket_uid,
                    expected_proxy_gid=self.model_socket_gid, model=self.model,
                    max_output_tokens=self.max_output_tokens)
                result = self._holder["capability"].start(idempotency_key=key)
                if result.decision == "ACCEPTED" and result.mission_id:
                    async with self._active_lock:
                        if result.mission_id in self._active: raise ValueError("mission execution already active")
                        task = asyncio.create_task(self._run_mission(result.mission_id, transport))
                        self._active[result.mission_id] = (transport, task)
                await self._reply(writer, {"decision": result.decision, "reason": result.reason, "mission_id": result.mission_id})
                return
            if peer_uid == 0:
                if request.get("action") == "cancel" and set(request) == {"action", "mission_id"}:
                    result = self._holder["capability"].cancel(request["mission_id"], operator_authorization=self._operator_token)
                    active = self._active.get(request["mission_id"])
                    if active: active[0].cancel()
                    await self._reply(writer, {"decision": result.decision, "reason": result.reason})
                    return
                if request == {"action": "revoke_service"}:
                    self._broker.revoke_service()
                    cancel_errors = []
                    for mission_id, (transport, _task) in tuple(self._active.items()):
                        try:
                            self._holder["capability"].cancel(mission_id, operator_authorization=self._operator_token)
                        except Exception as exc:  # fail-soft: still interrupt every model transport after durable service revoke
                            cancel_errors.append(type(exc).__name__)
                        finally:
                            transport.cancel()
                    if cancel_errors: raise RuntimeError("service revoked; mission audit reconciliation required")
                    await self._reply(writer, {"decision": "REVOKED", "service_identity": "jaxqwen"})
                    return
            await self._reply(writer, {"decision": "REJECTED", "reason": "authenticated peer required"})
        except (ValueError, OSError, asyncio.TimeoutError, json.JSONDecodeError, KeyError) as exc:
            log.warning("jaxqwen_transport_rejected peer_uid=%s reason=%s", peer_uid, type(exc).__name__)
            try: await self._reply(writer, {"decision": "REJECTED", "reason": "invalid or unauthenticated request"})
            except (ConnectionError, OSError): pass  # fail-soft: rejected peer may have closed before denial reply
        finally:
            writer.close()
            try: await writer.wait_closed()
            except (ConnectionError, OSError): pass  # fail-soft: peer disconnect does not alter authorization result

    async def _run_mission(self, mission_id: str, transport):
        try:
            await asyncio.to_thread(self._holder["capability"].run_tool_loop, mission_id, transport,
                                    credential_provider=lambda mission: self._holder["capability"].provision_credential(mission.mission_id))
        except Exception as exc:  # fail-soft: stop this mission and record only the exception class
            log.error("jaxqwen_mission_stopped mission=%s error=%s", mission_id, type(exc).__name__)
        finally:
            self._active.pop(mission_id, None)

    async def start(self):
        path = self.dispatch_socket
        parent = path.parent
        if parent.is_symlink() or not parent.is_dir() or parent.stat().st_mode & 0o022:
            raise RuntimeError("dispatcher socket directory is not protected")
        if path.exists() or path.is_symlink(): raise RuntimeError("dispatcher socket path already exists")
        if len(os.fsencode(path)) >= 104: raise RuntimeError("dispatcher socket path is too long")
        try:
            self._server = await asyncio.start_unix_server(self._serve_client, path=str(path), limit=_MAX_FRAME)
            os.chmod(path, 0o660, follow_symlinks=False); os.chown(path, -1, self.dispatch_gid, follow_symlinks=False)
        except BaseException:
            if self._server:
                self._server.close(); await self._server.wait_closed(); self._server = None
            if path.exists() and not path.is_symlink(): path.unlink()
            raise
        return path

    async def close(self):
        if self._server:
            self._server.close(); await self._server.wait_closed()
        active = tuple(self._active.values())
        for transport, _task in active: transport.cancel()
        if active: await asyncio.gather(*(task for _transport, task in active), return_exceptions=True)
        if self.dispatch_socket.exists() and not self.dispatch_socket.is_symlink(): self.dispatch_socket.unlink()
