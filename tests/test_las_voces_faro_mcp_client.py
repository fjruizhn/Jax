"""Real MCP SDK client against the LV-001B read-only stdio server."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.shared.exceptions import MCPError

ROOT = Path(__file__).resolve().parents[1]


def test_real_client_lists_reads_and_cannot_mutate():
    async def smoke():
        params = StdioServerParameters(command=sys.executable,
                                      args=[str(ROOT / "tests/_las_voces_readonly_fixture.py")],
                                      env={**os.environ, "PYTHONPATH": str(ROOT)}, cwd=str(ROOT))
        async with Client(params) as client:
            tools = await client.list_tools()
            assert [tool.name for tool in tools.tools] == ["skills.buscar", "skills.leer", "agentes.listar"]
            found = await client.call_tool("skills.buscar", {"consulta": "mide"})
            assert "alfa" in str(found)
            read = await client.call_tool("skills.leer", {"nombre": "alfa"})
            assert "lee" in str(read)
            agents = await client.call_tool("agentes.listar", {})
            assert "explorador" in str(agents)
            assert "secret-body" not in str(agents)
            try:
                await client.call_tool("faro.crear", {})
            except MCPError as exc:
                assert exc.code == -32602
            else:
                raise AssertionError("Faro mutation was accepted")
            assert (await client.list_tools()).tools  # denial did not corrupt the channel
    asyncio.run(smoke())
