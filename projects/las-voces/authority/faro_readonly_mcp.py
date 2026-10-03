"""MCP read-only view of a verified Faro package for local Qwen Code.

This process does not connect to Faro control or the per-run Puerto. It loads the
same immutable package the Puerto serves, then exposes a fixed read allowlist.
It is a compatibility view, not a Faro execution or authority channel.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TextIO

# Qwen starts the server in the project directory; Python otherwise puts only
# authority/ on sys.path, which would make the repository's Faro module invisible.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from jax.faro.config import ConfigFaro
from jax.faro.paquete import NoExiste, PaqueteCargado, cargar_paquete

MAX_REQUEST_BYTES = 1024 * 1024
PROTOCOL_VERSION = "2025-03-26"
READ_TOOLS = ("skills.buscar", "skills.leer", "agentes.listar")


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"name": name, "description": description,
            "inputSchema": {"type": "object", "properties": properties,
                            "required": required, "additionalProperties": False},
            "annotations": {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}}


TOOLS = [
    _tool("skills.buscar", "Busca skills en el paquete verificado de El Faro.",
          {"consulta": {"type": "string"}, "limite": {"type": "integer", "minimum": 1, "maximum": 100}}, []),
    _tool("skills.leer", "Lee un archivo de una skill del paquete verificado.",
          {"nombre": {"type": "string"}, "archivo": {"type": "string"}}, ["nombre"]),
    _tool("agentes.listar", "Lista el catalogo de agentes sin lanzarlos.", {}, []),
]


def _params(value: object, allowed: set[str], required: set[str] = frozenset()) -> dict:
    if not isinstance(value, dict) or set(value) - allowed or not required <= set(value):
        raise ValueError("argumentos invalidos")
    return value


def _call(package: PaqueteCargado, name: str, arguments: object) -> object:
    if name == "skills.buscar":
        p = _params(arguments, {"consulta", "limite"})
        q, limit = p.get("consulta", ""), p.get("limite", 20)
        if not isinstance(q, str) or len(q) > 4096 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("consulta o limite invalido")
        return package.buscar(q, limit)
    if name == "skills.leer":
        p = _params(arguments, {"nombre", "archivo"}, {"nombre"})
        name_arg, file_arg = p["nombre"], p.get("archivo", "SKILL.md")
        if not isinstance(name_arg, str) or not isinstance(file_arg, str) or len(name_arg) > 4096 or len(file_arg) > 4096:
            raise ValueError("nombre o archivo invalido")
        return package.leer(name_arg, file_arg)
    if name == "agentes.listar":
        _params(arguments, set())
        return package.catalogo_agentes()
    raise ValueError("herramienta no permitida")


def handle(package: PaqueteCargado, request: object) -> dict | None:
    if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
    request_id = request.get("id")
    method = request.get("method")
    if not isinstance(method, str):
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32600, "message": "Invalid Request"}}
    if "id" not in request:  # MCP notifications cannot invoke a tool.
        return None
    try:
        if method == "initialize":
            result = {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}, "resources": {}},
                      "serverInfo": {"name": "las-voces-faro-readonly", "version": "1"},
                      "instructions": "Catalogo verificado de El Faro: solo lectura; sin ejecucion ni autoridad."}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            p = _params(request.get("params"), {"name", "arguments", "_meta"}, {"name"})
            value = _call(package, p["name"], p.get("arguments", {}))
            result = {"content": [{"type": "text", "text": value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}]}
        elif method == "resources/list":
            result = {"resources": [
                {"uri": "ecosistema://constitucion", "name": "constitucion", "mimeType": "text/markdown"},
                {"uri": "ecosistema://agentes", "name": "agentes", "mimeType": "application/json"}]}
        elif method == "resources/templates/list":
            result = {"resourceTemplates": [{"uriTemplate": "skill://{nombre}", "name": "skill", "mimeType": "text/markdown"}]}
        elif method == "resources/read":
            p = _params(request.get("params"), {"uri", "_meta"}, {"uri"})
            uri = p["uri"]
            if uri == "ecosistema://constitucion":
                value, mime = package.constitucion, "text/markdown"
            elif uri == "ecosistema://agentes":
                value, mime = json.dumps(package.catalogo_agentes(), ensure_ascii=False), "application/json"
            elif isinstance(uri, str) and uri.startswith("skill://"):
                value, mime = package.leer(uri[len("skill://"):]), "text/markdown"
            else:
                raise ValueError("recurso no permitido")
            result = {"contents": [{"uri": uri, "mimeType": mime, "text": value}]}
        else:
            return {"jsonrpc": "2.0", "id": request_id,
                    "error": {"code": -32601, "message": "Method not found"}}
    except (ValueError, KeyError, NoExiste) as exc:
        return {"jsonrpc": "2.0", "id": request_id,
                "error": {"code": -32602, "message": str(exc)}}
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def serve(package: PaqueteCargado, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    while line := stdin.readline(MAX_REQUEST_BYTES + 1):
        if len(line.encode("utf-8")) > MAX_REQUEST_BYTES or not line.endswith("\n"):
            return 2
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            return 2
        response = handle(package, request)
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
            stdout.flush()
    return 0


def serve_from_config(config: ConfigFaro, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    """Load the verified package before making any MCP response available."""
    return serve(cargar_paquete(config), stdin, stdout)


def main() -> int:
    try:
        config = ConfigFaro.desde_entorno(os.environ)
        return serve_from_config(config)
    except Exception as exc:  # fail-closed: no verified package means no MCP server is opened
        print(f"faro-readonly: paquete no disponible o no verificado: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
