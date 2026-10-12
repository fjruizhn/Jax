"""Conector Codex CLI del Faro (spec §0.8/P6): config.toml + requirements.toml.

Verificado contra la documentacion oficial (developers.openai.com/codex/config-reference
y config-advanced) y contra strings del binario codex-cli 0.162.1
(core/src/config/requirements.rs):

- `model_providers.<id>`: name, base_url, wire_api (solo "responses"),
  requires_openai_auth (bool), supports_websockets (bool), env_key (variable de
  entorno que provee la API key; NO existe bearer_token directo).
- `requirements.toml` (config gestionada): exige `default_permissions` salvo que
  `allowed_permission_profiles` permita :workspace y :read-only a la vez; el
  namespace `mcp_servers.` es gestionado por requirements ("is managed by
  requirements and cannot be changed"), asi que el server Faro se declara ahi.
- OTel off: [otel] exporter/metrics_exporter/trace_exporter = "none".
  Analytics off: [analytics] enabled = false.

La API key NO se escribe en el TOML: `env_key` nombra la variable, y el mapa de
entorno que acompana al proceso lleva el marcador ficticio (el provisionador real
lo reemplaza en el hogar efimero).
"""
from __future__ import annotations

from dataclasses import dataclass

from ._comun import (
    argv_rele,
    credencial_ficticia,
    modelo_sano,
    nombre_sano,
    ruta_absoluta,
    toml_arreglo_textos,
    toml_texto,
    url_loopback,
)

ENV_KEY = "FARO_LOOPBACK_API_KEY"
PROVIDER_ID = "faro-loopback"
PERMISOS = "workspace-write"
PERFILES = [":workspace", ":read-only"]


@dataclass(frozen=True)
class ConectorCodex:
    """Salida pura y determinista del conector Codex."""

    config_toml: str
    requirements_toml: str
    entorno: dict  # str -> str: variables que el lanzador debe poner al proceso
    argv_rele: list  # argv del rele MCP (python -m jax.faro.relay ...)


def generar(
    *,
    nombre: str,
    modelo: str,
    base_url: str,
    socket: str,
    token_file: str,
    python: str = "/usr/bin/python3",
) -> ConectorCodex:
    """Genera la configuracion de Codex apuntando al puente loopback del Faro.

    `nombre` identifica la ejecucion (sano, [a-z0-9._-]); `modelo` es el nombre
    publico del modelo detras del puente. Todo dato invalido levanta
    TypeError/ValueError (falla cerrada)."""
    nombre = nombre_sano(nombre)
    modelo = modelo_sano(modelo)
    base_url = url_loopback(base_url)
    argv = argv_rele(python, socket, token_file)
    env_key_valor = credencial_ficticia(nombre)

    lineas = [
        f"model = {toml_texto(modelo)}",
        f"model_provider = {toml_texto(PROVIDER_ID)}",
        "",
        f"[model_providers.{PROVIDER_ID}]",
        f"name = {toml_texto('Faro loopback')}",
        f"base_url = {toml_texto(base_url)}",
        'wire_api = "responses"',
        "requires_openai_auth = false",
        "supports_websockets = false",
        f"env_key = {toml_texto(ENV_KEY)}",
        "",
        "[otel]",
        'exporter = "none"',
        'metrics_exporter = "none"',
        'trace_exporter = "none"',
        "",
        "[analytics]",
        "enabled = false",
        "",
        "[mcp_servers.faro]",
        f"command = {toml_texto(argv[0])}",
        f"args = {toml_arreglo_textos(argv[1:])}",
        "",
    ]
    config_toml = "\n".join(lineas)

    lineas_req = [
        f"default_permissions = {toml_texto(PERMISOS)}",
        f"allowed_permission_profiles = {toml_arreglo_textos(PERFILES)}",
        "",
        "[mcp_servers.faro]",
        f"command = {toml_texto(argv[0])}",
        f"args = {toml_arreglo_textos(argv[1:])}",
        "",
    ]
    requirements_toml = "\n".join(lineas_req)

    return ConectorCodex(
        config_toml=config_toml,
        requirements_toml=requirements_toml,
        entorno={ENV_KEY: env_key_valor},
        argv_rele=argv,
    )


def ruta_config(hogar: str) -> str:
    """Ruta del config.toml dentro del hogar efimero de Codex."""
    hogar = ruta_absoluta(hogar, "hogar")
    return f"{hogar}/config.toml"


def ruta_requirements(hogar: str) -> str:
    """Ruta del requirements.toml gestionado dentro del hogar efimero."""
    hogar = ruta_absoluta(hogar, "hogar")
    return f"{hogar}/requirements.toml"
