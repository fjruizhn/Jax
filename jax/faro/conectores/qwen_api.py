"""Traduccion Qwen: tools/list (MCP) a function calling (spec §0.8/P6).

Qwen CLI habla function calling nativo; el Faro le presenta las herramientas del
Puerto traduciendo cada entrada de `tools/list` al shape de OpenAI:

    {"type": "function", "function": {"name", "description", "parameters"}}

donde `parameters` es el `inputSchema` JSON Schema tal cual. Es una funcion PURA:
traduce, no resuelve ni ejecuta nada. Falta `name` o schema invalido -> falla
cerrada (ValueError), nunca se inventa una firma.
"""
from __future__ import annotations


def tools_a_function_calling(herramientas: list) -> list:
    """Traduce la lista de herramientas MCP a entradas de function calling.

    `herramientas` es el campo `tools` de una respuesta `tools/list`: objetos con
    `name` (obligatorio), `description` (opcional) e `inputSchema` (opcional)."""
    if not isinstance(herramientas, (list, tuple)):
        raise TypeError(f"herramientas: se esperaba list, llego {type(herramientas).__name__}")
    traducidas = []
    for i, tool in enumerate(herramientas):
        if not isinstance(tool, dict):
            raise TypeError(f"herramientas[{i}]: se esperaba dict, llego {type(tool).__name__}")
        nombre = tool.get("name")
        if not isinstance(nombre, str) or not nombre:
            raise ValueError(f"herramientas[{i}]: 'name' obligatorio y no vacio")
        for ch in nombre:
            if ord(ch) < 0x20 or ord(ch) == 0x7F:
                raise ValueError(f"herramientas[{i}]: nombre con caracter de control")
        descripcion = tool.get("description", "")
        if not isinstance(descripcion, str):
            raise TypeError(f"herramientas[{i}]: 'description' debe ser str")
        esquema = tool.get("inputSchema", {})
        if not isinstance(esquema, dict):
            raise TypeError(f"herramientas[{i}]: 'inputSchema' debe ser objeto JSON Schema")
        traducidas.append(
            {
                "type": "function",
                "function": {
                    "name": nombre,
                    "description": descripcion,
                    "parameters": esquema,
                },
            }
        )
    return traducidas
