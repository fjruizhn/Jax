"""Single-purpose client for the dedicated jaxqwen dispatch socket.

Only the governed dispatcher composition should call this after it has
recorded a DISPATCHED ACK. It sends the ACK idempotency key and nothing else.
Both Unix peer identities are checked; no username/body identity is trusted.
"""
from __future__ import annotations

import json
import socket
import stat
import struct
from pathlib import Path
from typing import Any


class DispatchTransportError(ValueError):
    pass


def request_jaxqwen(socket_path: Path, idempotency_key: str, *, expected_service_uid: int,
                    timeout_seconds: float = 5.0) -> dict[str, Any]:
    if not isinstance(idempotency_key, str) or len(idempotency_key) != 64 or any(c not in "0123456789abcdef" for c in idempotency_key):
        raise DispatchTransportError("invalid ACK idempotency key")
    socket_path = Path(socket_path)
    if not socket_path.is_absolute() or socket_path.is_symlink(): raise DispatchTransportError("unsafe jaxqwen socket path")
    st = socket_path.lstat()
    if not stat.S_ISSOCK(st.st_mode) or st.st_mode & 0o007 or st.st_uid != expected_service_uid: raise DispatchTransportError("jaxqwen socket is not protected")
    parent = socket_path.parent.stat()
    if socket_path.parent.is_symlink() or parent.st_mode & 0o022 or parent.st_uid != expected_service_uid:
        raise DispatchTransportError("jaxqwen socket directory is not protected")
    request = json.dumps({"action": "dispatch", "idempotency_key": idempotency_key}, separators=(",", ":")).encode() + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout_seconds)
        client.connect(str(socket_path))
        raw = client.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        _pid, uid, _gid = struct.unpack("3i", raw)
        if uid != expected_service_uid: raise DispatchTransportError("unexpected jaxqwen service peer")
        client.sendall(request)
        response = bytearray()
        while len(response) <= 8192:
            chunk = client.recv(1024)
            if not chunk: break
            response.extend(chunk)
            if b"\n" in response: break
        if len(response) > 8192 or not response.endswith(b"\n"): raise DispatchTransportError("invalid jaxqwen response frame")
    try: value = json.loads(response.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise DispatchTransportError("malformed jaxqwen response") from exc
    if not isinstance(value, dict) or value.get("decision") not in {"ACCEPTED", "NOOP", "REJECTED", "HUMAN_REQUIRED"}:
        raise DispatchTransportError("invalid jaxqwen response")
    return value
