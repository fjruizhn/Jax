"""LV-001B: isolated MCP smoke and a fail-closed read boundary."""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from types import MappingProxyType

from jax.faro.paquete import PaqueteCargado

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "projects/las-voces/authority/faro_readonly_mcp.py"
SPEC = importlib.util.spec_from_file_location("lv_faro_readonly", MODULE)
assert SPEC and SPEC.loader
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


def package() -> PaqueteCargado:
    return PaqueteCargado(
        sha="a" * 40,
        constitucion="# Constitución de prueba\n",
        skills=MappingProxyType({"alfa": MappingProxyType({"SKILL.md": b"---\ndescription: mide\n---\nlee"})}),
        agentes=MappingProxyType({"explorador": b"---\ndescription: descubre\nmodel: haiku\n---\nsecret-body"}),
    )


def req(method: str, params: dict | None = None, ident: int = 1) -> dict:
    return {"jsonrpc": "2.0", "id": ident, "method": method, "params": params or {}}


def test_isolated_stdio_smoke_reads_verified_package_only():
    incoming = [req("initialize"), req("tools/list", ident=2),
                req("tools/call", {"name": "skills.buscar", "arguments": {"consulta": "mide"}}, 3),
                req("tools/call", {"name": "skills.leer", "arguments": {"nombre": "alfa"}}, 4),
                req("tools/call", {"name": "agentes.listar", "arguments": {}}, 5),
                req("resources/read", {"uri": "ecosistema://constitucion"}, 6)]
    output = io.StringIO()
    assert bridge.serve(package(), io.StringIO("".join(json.dumps(x) + "\n" for x in incoming)), output) == 0
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert rows[0]["result"]["serverInfo"]["name"] == "las-voces-faro-readonly"
    assert [x["name"] for x in rows[1]["result"]["tools"]] == list(bridge.READ_TOOLS)
    assert "alfa" in rows[2]["result"]["content"][0]["text"]
    assert "lee" in rows[3]["result"]["content"][0]["text"]
    assert "secret-body" not in rows[4]["result"]["content"][0]["text"]
    assert "Constitución" in rows[5]["result"]["contents"][0]["text"]


def test_readonly_boundary_denies_unknown_methods_tools_and_paths():
    p = package()
    denied = [
        req("control/crear"), req("missions/launch"), req("resources/write", {"uri": "skill://alfa"}),
        req("tools/call", {"name": "faro.crear", "arguments": {}}),
        req("tools/call", {"name": "skills.leer", "arguments": {"nombre": "../secreto"}}),
        req("resources/read", {"uri": "file:///etc/passwd"}),
        req("tools/call", {"name": "skills.leer", "arguments": {"nombre": "alfa", "archivo": "../token"}}),
        req("tools/call", {"name": "skills.buscar", "arguments": {"limite": True}}),
        req("tools/call", {"name": "agentes.listar", "arguments": {"execute": True}}),
    ]
    assert all("error" in bridge.handle(p, request) for request in denied)
    assert bridge.handle(p, {"jsonrpc": "2.0", "method": "tools/call",
                             "params": {"name": "faro.crear", "arguments": {}}}) is None


def test_fails_closed_without_verified_package(monkeypatch):
    monkeypatch.delenv("JAX_FARO_REPO", raising=False)
    monkeypatch.delenv("JAX_FARO_SHA", raising=False)
    monkeypatch.delenv("JAX_FARO_ECOSISTEMA_DIR", raising=False)
    assert bridge.main() == 2


def test_oversized_or_unterminated_request_closes_without_response():
    for data in ("x" * (bridge.MAX_REQUEST_BYTES + 1), json.dumps(req("tools/list"))):
        out = io.StringIO()
        assert bridge.serve(package(), io.StringIO(data), out) == 2
        assert out.getvalue() == ""
