"""
LAS MANOS — Motor Registry: catálogo de tools (GAP 2, Fase 1).

Declaraciones de schema JSON (forma OpenAI `tools`) — SOLO forma, sin
implementación ejecutable. Fase 1 expone estas declaraciones al modelo,
parsea lo que pida y lo loguea; no ejecuta nada (ver worker.py::run,
rama TOOLS_REQUESTED).

Por qué acá y no en DB ni hardcodeado en worker.py: mismo criterio que
_REFORMAS_V3_PREDICATES en worker.py -- "vocabulario cerrado, versionado
en git", sin sumar una tabla nueva (capability/capability_motor de R4
son para AUTORIDAD de ejecución, fase 2 -- acá todavía no hay nada que
autorizar, solo declarar). Una tabla DB para esto sería prematura: fase 1
es deliberadamente de solo-observación, cero ejecución, cero mapeo a
capability.

Elegidas read_file y write_file: son las dos que el objetivo real de
Fernando ("has un html de una calculadora científica y financiera")
necesita para completarse de punta a punta -- generar contenido y
guardarlo en disco, que es exactamente lo que Qwen respondió no poder
hacer cuando Fernando lo probó por primera vez (origen de esta
investigación completa).

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

TOOLS_CATALOG: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lee el contenido de un archivo de texto del workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Ruta del archivo a leer, relativa al workspace.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Escribe (crea o sobrescribe) un archivo de texto en el workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Ruta del archivo a escribir, relativa al workspace.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Contenido completo a escribir en el archivo.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
]
