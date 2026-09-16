#!/usr/bin/env python3
"""Fase 0 · U7: API compatible con Anthropic de Moonshot (Kimi) y Z.ai (GLM),
con NUESTRAS llaves (resolve_facet, en vivo). Ningún comando se ejecuta: la
herramienta es ficticia; solo se mira si el modelo la pide.
Correr con el prefijo de entorno del plan (lee la DB, no escribe).
"""
import asyncio
import json
import pathlib
import sys
import tomllib
import urllib.error
import urllib.request

from facet_resolver import resolve_facet

AQUI = pathlib.Path(__file__).resolve().parent
HERRAMIENTA = {"name": "run_shell", "description": "Ejecuta un comando de shell y devuelve su salida",
               "input_schema": {"type": "object", "properties": {"command": {"type": "string"}},
                                "required": ["command"]}}


def _mensajes(base, llave, modelo):
    cuerpo = {"model": modelo, "max_tokens": 400, "tools": [HERRAMIENTA],
              "messages": [{"role": "user", "content": "¿Cuánto disco libre queda en el servidor? Usa la herramienta."}]}
    req = urllib.request.Request(base.rstrip("/") + "/v1/messages", data=json.dumps(cuerpo).encode(),
                                 headers={"content-type": "application/json", "x-api-key": llave,
                                          "authorization": f"Bearer {llave}", "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


async def main(salida):
    cerebros = tomllib.loads((AQUI / "cerebros.toml").read_text())
    fallos = 0
    for nombre in ("kimi", "glm"):
        c = cerebros[nombre]
        f = await resolve_facet(c["faceta"])
        fila = {"cerebro": nombre, "modelo": f.model}
        try:
            r = _mensajes(c["anthropic_base_url"], f.credential, f.model)
            usos = [b for b in r.get("content", []) if b.get("type") == "tool_use"]
            fila.update(ok=r.get("stop_reason") == "tool_use" and any(b.get("name") == "run_shell" for b in usos),
                        stop_reason=r.get("stop_reason"), tool_use=usos, usage=r.get("usage"))
        except urllib.error.HTTPError as e:  # fail-soft: el error HTTP ES el resultado de U7 y queda registrado; el exit code final lo refleja
            fila.update(ok=False, http=e.code, cuerpo=e.read().decode()[:500])
        with salida.open("a") as fh:
            fh.write(json.dumps(fila, ensure_ascii=False) + "\n")
        print(json.dumps(fila, ensure_ascii=False))
        fallos += not fila["ok"]
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(pathlib.Path(sys.argv[1]).expanduser())))
