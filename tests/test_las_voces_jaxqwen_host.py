"""Host IPC admission tests use real AF_UNIX peer credentials."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
AUTHORITY = REPO / "projects/las-voces/authority"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


host_module = load("jaxqwen_host_test", AUTHORITY / "jaxqwen_host.py")


async def request(host, parent: Path, payload: dict):
    socket_path = parent / "dispatch.sock"
    server = await asyncio.start_unix_server(host._serve_client, path=str(socket_path))
    os.chmod(socket_path, 0o660)
    try:
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        writer.write(json.dumps(payload).encode() + b"\n")
        await writer.drain()
        response = json.loads(await asyncio.wait_for(reader.readline(), 2))
        writer.close()
        await writer.wait_closed()
        return response
    finally:
        server.close()
        await server.wait_closed()


def test_host_dispatch_requires_peer_uid_then_passes_only_ack_key(tmp_path):
    async def scenario():
        host = object.__new__(host_module.JaxQwenHost)
        host.dispatcher_uid = os.getuid()
        host.dispatch_enabled = True
        host.model_socket = tmp_path / "unused.sock"
        host.model_socket_uid, host.model_socket_gid = 10001, 10002
        host.model, host.max_output_tokens = "fixed-qwen", 32
        host.transport_module = SimpleNamespace(JaxQwenUnixModelTransport=lambda *a, **k: object())
        seen = []
        host._holder = {"capability": SimpleNamespace(start=lambda **kwargs: (seen.append(kwargs) or SimpleNamespace(decision="REJECTED", reason="synthetic fail closed", mission_id=None)))}
        response = await request(host, tmp_path, {"action": "dispatch", "idempotency_key": "a" * 64})
        assert response["decision"] == "REJECTED"
        assert seen == [{"idempotency_key": "a" * 64}]

    asyncio.run(scenario())


def test_host_does_not_accept_body_identity_or_uid_without_configured_dispatcher(tmp_path):
    async def scenario():
        host = object.__new__(host_module.JaxQwenHost)
        host.dispatcher_uid = os.getuid() + 1
        host.dispatch_enabled = True
        called = []
        host._holder = {"capability": SimpleNamespace(start=lambda **kwargs: called.append(kwargs))}
        response = await request(host, tmp_path, {"action": "dispatch", "idempotency_key": "b" * 64,
                                                  "service_identity": "jaxqwen", "actor": "jaxqwen"})
        assert response["decision"] == "REJECTED"
        assert called == []

    asyncio.run(scenario())


def test_disabled_first_dispatch_gate_never_calls_capability(tmp_path):
    async def scenario():
        host = object.__new__(host_module.JaxQwenHost)
        host.dispatcher_uid = os.getuid()
        host.dispatch_enabled = False
        called = []
        host._holder = {"capability": SimpleNamespace(start=lambda **kwargs: called.append(kwargs))}
        response = await request(host, tmp_path, {"action": "dispatch", "idempotency_key": "c" * 64})
        assert response["decision"] == "HUMAN_REQUIRED"
        assert called == []

    asyncio.run(scenario())
