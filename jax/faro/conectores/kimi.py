"""Conector Kimi Code CLI del Faro (spec §0.8/P6): config.toml + mcp.json efimero.

Verificado contra la documentacion oficial de Kimi Code CLI (kimi.com, docs de
config-files y mcp) y contra el CLI local 2.1.1:

- `KIMI_CODE_HOME` reubica el data dir: la config vive en
  `$KIMI_CODE_HOME/config.toml` y los servers MCP en `$KIMI_CODE_HOME/mcp.json`.
- `[providers."<id>"]`: type (kimi|anthropic|openai|openai_responses|google-genai|
  vertexai), base_url, api_key (texto plano; api_key_env es alternativa
  mutuamente excluyente; si falta cualquier credencial, el arranque FALLA — por
  eso el marcador ficticio va explicito y el provisionador real lo reemplaza).
- `[models."<alias>"]`: provider, model, max_context_size (obligatorio, >= 1).
- `default_model` top-level; `telemetry = false` top-level apaga la telemetria.
- mcp.json: {"mcpServers": {"faro": {"command": ..., "args": [...]}}}

El mcp.json es EFIMERO: se genera por ejecucion en el hogar (cierra F-3: ningun
server MCP del Faro persiste en la config global del usuario).
"""
from __future__ import annotations

from dataclasses import dataclass

from ._comun import (
    argv_rele,
    credencial_ficticia,
    json_ordenado,
    modelo_sano,
    nombre_sano,
    ruta_absoluta,
    toml_texto,
    url_loopback,
)

PROVIDER_ID = "faro-loopback"


@dataclass(frozen=True)
class ConectorKimi:
    """Salida pura y determinista del conector Kimi."""

    config_toml: str
    mcp_json: str
    ruta_config: str  # $KIMI_CODE_HOME/config.toml
    ruta_mcp: str  # $KIMI_CODE_HOME/mcp.json
    entorno: dict  # str -> str: KIMI_CODE_HOME + variables del proceso


def generar(
    *,
    nombre: str,
    modelo: str,
    base_url: str,
    socket: str,
    token_file: str,
    kimi_code_home: str,
    python: str = "/usr/bin/python3",
    tipo: str = "openai",
    max_contexto: int = 262144,
) -> ConectorKimi:
    """Genera la configuracion de Kimi apuntando al puente loopback del Faro.

    `kimi_code_home` es el hogar efimero de la ejecucion (ruta absoluta): alli
    viven config.toml y el mcp.json efimero. Falla cerrada con dato invalido."""
    nombre = nombre_sano(nombre)
    modelo = modelo_sano(modelo)
    base_url = url_loopback(base_url)
    hogar = ruta_absoluta(kimi_code_home, "kimi_code_home")
    argv = argv_rele(python, socket, token_file)
    tipo = nombre_sano(tipo, "tipo")
    if tipo not in ("kimi", "anthropic", "openai", "openai_responses", "google-genai", "vertexai"):
        raise ValueError(f"tipo: proveedor Kimi no soportado: {tipo!r}")
    if not isinstance(max_contexto, int) or isinstance(max_contexto, bool):
        raise TypeError(f"max_contexto: se esperaba int, llego {type(max_contexto).__name__}")
    if max_contexto < 1:
        raise ValueError("max_contexto: entero >= 1")

    alias = f"faro/{modelo}"
    lineas = [
        f"default_model = {toml_texto(alias)}",
        "telemetry = false",
        "",
        f'[providers."{PROVIDER_ID}"]',
        f"type = {toml_texto(tipo)}",
        f"base_url = {toml_texto(base_url)}",
        f"api_key = {toml_texto(credencial_ficticia(nombre))}",
        "",
        f'[models."{alias}"]',
        f"provider = {toml_texto(PROVIDER_ID)}",
        f"model = {toml_texto(modelo)}",
        f"max_context_size = {max_contexto}",
        "",
    ]
    config_toml = "\n".join(lineas)

    mcp_json = json_ordenado({"mcpServers": {"faro": {"command": argv[0], "args": argv[1:]}}})

    return ConectorKimi(
        config_toml=config_toml,
        mcp_json=mcp_json,
        ruta_config=f"{hogar}/config.toml",
        ruta_mcp=f"{hogar}/mcp.json",
        entorno={"KIMI_CODE_HOME": hogar},
    )
