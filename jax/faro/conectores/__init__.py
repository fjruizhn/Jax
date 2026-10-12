"""Conectores del Faro: generadores puros de configuracion por motor (spec §0.8/P6).

Cada motor (Codex, Kimi, Claude, Qwen) recibe una config que apunta su MCP y su
API al puente loopback del Faro, con credenciales de marcador ficticio que el
provisionador real reemplaza en el hogar efimero. Ver los modulos individuales.
"""
from __future__ import annotations

from .claude import ConectorClaude, generar as generar_claude
from .codex import ConectorCodex, generar as generar_codex
from .kimi import ConectorKimi, generar as generar_kimi
from .qwen_api import tools_a_function_calling

__all__ = [
    "ConectorClaude",
    "ConectorCodex",
    "ConectorKimi",
    "generar_claude",
    "generar_codex",
    "generar_kimi",
    "tools_a_function_calling",
]
