"""Adversarial tests for fixed, peer-bound Qwen model transport."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("jaxqwen_transport_test", REPO / "projects/las-voces/authority/jaxqwen_transport.py")
assert spec and spec.loader
jaxqwen_transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jaxqwen_transport)


def test_proxy_peer_must_match_host_config_and_model_gets_only_structured_tools(tmp_path):
    async def scenario():
        parent = tmp_path / "proxy"
        parent.mkdir(mode=0o700)
        path = parent / "model.sock"
        requests = []

        async def handle(reader, writer):
            headers = await reader.readuntil(b"\r\n\r\n")
            content_length = next(int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n") if line.lower().startswith(b"content-length:"))
            payload = json.loads(await reader.readexactly(content_length))
            requests.append(payload)
            response = json.dumps({"content": [{"type": "tool_use", "id": "tool-1", "name": "write_file", "input": {"path": "note.txt", "content": "bounded"}}]}).encode()
            writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: " + str(len(response)).encode() + b"\r\nconnection: close\r\n\r\n" + response)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_unix_server(handle, path=str(path))
        os.chmod(path, 0o660)
        os.chown(path, os.getuid(), os.getgid())
        mission = SimpleNamespace(mission_id="m", allowed_tools=("write_file",), task_id="LV-001",
                                  acceptance_criteria=("write note",), evidence_requirements=())
        transport = jaxqwen_transport.JaxQwenUnixModelTransport(
            path, expected_proxy_uid=os.getuid(), expected_proxy_gid=os.getgid(), model="qwen-fixed")
        try:
            calls = await asyncio.to_thread(
                transport.request_tools, mission, ({"role": "system", "content": "assigned"},))
        finally:
            server.close()
            await server.wait_closed()
        assert calls == [{"name": "write_file", "arguments": {"path": "note.txt", "content": "bounded"}}]
        assert requests[0]["model"] == "qwen-fixed"
        assert [tool["name"] for tool in requests[0]["tools"]] == ["write_file"]
        assert "shell" not in json.dumps(requests[0]["tools"])

    asyncio.run(scenario())


def test_wrong_proxy_peer_and_replaced_socket_are_denied(tmp_path):
    async def scenario():
        parent = tmp_path / "proxy"
        parent.mkdir(mode=0o700)
        path = parent / "model.sock"
        accepted = asyncio.Event()

        async def handle(reader, writer):
            accepted.set()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_unix_server(handle, path=str(path))
        os.chmod(path, 0o660)
        os.chown(path, os.getuid(), os.getgid())
        mission = SimpleNamespace(mission_id="m", allowed_tools=("write_file",), task_id="LV-001",
                                  acceptance_criteria=(), evidence_requirements=())
        transport = jaxqwen_transport.JaxQwenUnixModelTransport(
            path, expected_proxy_uid=os.getuid() + 1, expected_proxy_gid=os.getgid(), model="qwen-fixed")
        try:
            with pytest.raises(jaxqwen_transport.TransportError, match="model socket"):
                transport.request_tools(mission, ())
            assert not accepted.is_set()
            path.unlink()
            path.write_text("replacement", encoding="utf-8")
            with pytest.raises(jaxqwen_transport.TransportError, match="model socket"):
                transport.request_tools(mission, ())
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())
