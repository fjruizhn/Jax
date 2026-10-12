"""Conector Claude Code del Faro (spec §0.8/P6): settings.json + --mcp-config.

Verificado contra el binario local claude v2.1.296 (`--help`) y contra el guion de
fijado de navegadores `claude-skills/bin/mcp-navegadores-pin.sh`:

- `--mcp-config <configs...>` acepta archivos de servers MCP; NO se usa
  `--strict-mcp-config` (el usuario podria tener servers propios; el Faro solo
  añade el suyo).
- Los MCP de navegador van fijados por npm (sin `npx -y ...@latest`): binarios en
  `<mcp_dir>/chrome-devtools/node_modules/.bin/chrome-devtools-mcp` y el equivalente
  de playwright. Banderas base no negociables (constitucion, seccion MCP de
  navegador): chrome-devtools lleva `--headless --isolated --no-usage-statistics
  --no-performance-crux --redactNetworkHeaders`; playwright `--headless --isolated`.
- Trafico no esencial off: `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` en el entorno.

Variantes por maquina (`--executablePath`, `--browser chromium`) quedan como
parametros del llamador, no hardcodeadas aqui (ver TRASPASO.md).
"""
from __future__ import annotations

from dataclasses import dataclass

from ._comun import (
    argv_rele,
    credencial_ficticia,
    json_ordenado,
    nombre_sano,
    ruta_absoluta,
    url_loopback,
)

FLAGS_CHROME_DEVTOOLS = [
    "--headless",
    "--isolated",
    "--no-usage-statistics",
    "--no-performance-crux",
    "--redactNetworkHeaders",
]
FLAGS_PLAYWRIGHT = ["--headless", "--isolated"]


@dataclass(frozen=True)
class ConectorClaude:
    """Salida pura y determinista del conector Claude."""

    settings_json: str
    mcp_config_json: str
    argv: list  # ["claude", "--mcp-config", <ruta>]
    ruta_settings: str  # hogar efimero/settings.json
    ruta_mcp_config: str  # hogar efimero/mcp-config.json
    entorno: dict  # str -> str


def generar(
    *,
    nombre: str,
    base_url: str,
    socket: str,
    token_file: str,
    hogar: str,
    bin_chrome_devtools: str,
    bin_playwright: str,
    python: str = "/usr/bin/python3",
    flags_chrome_devtools: tuple = tuple(FLAGS_CHROME_DEVTOOLS),
    flags_playwright: tuple = tuple(FLAGS_PLAYWRIGHT),
) -> ConectorClaude:
    """Genera la configuracion de Claude apuntando al puente loopback del Faro.

    `hogar` es el hogar efimero de la ejecucion (ruta absoluta). Los binarios de
    navegador deben ser rutas absolutas a los binarios fijados por npm. Falla
    cerrada con dato invalido."""
    nombre = nombre_sano(nombre)
    base_url = url_loopback(base_url)
    hogar = ruta_absoluta(hogar, "hogar")
    bin_chrome_devtools = ruta_absoluta(bin_chrome_devtools, "bin_chrome_devtools")
    bin_playwright = ruta_absoluta(bin_playwright, "bin_playwright")
    argv_rele_faro = argv_rele(python, socket, token_file)
    flags_cd = [str(f) for f in flags_chrome_devtools]
    flags_pw = [str(f) for f in flags_playwright]
    for f in flags_cd + flags_pw:
        if not f.startswith("--") or any(ord(c) < 0x20 for c in f):
            raise ValueError(f"flags: bandera invalida {f!r}")

    settings = {
        "env": {
            "ANTHROPIC_AUTH_TOKEN": credencial_ficticia(nombre),
            "ANTHROPIC_BASE_URL": base_url,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
    }
    mcp_config = {
        "mcpServers": {
            "chrome-devtools": {"command": bin_chrome_devtools, "args": flags_cd},
            "faro": {"command": argv_rele_faro[0], "args": argv_rele_faro[1:]},
            "playwright": {"command": bin_playwright, "args": flags_pw},
        }
    }
    ruta_mcp_config = f"{hogar}/mcp-config.json"
    return ConectorClaude(
        settings_json=json_ordenado(settings),
        mcp_config_json=json_ordenado(mcp_config),
        argv=["claude", "--mcp-config", ruta_mcp_config],
        ruta_settings=f"{hogar}/settings.json",
        ruta_mcp_config=ruta_mcp_config,
        entorno=dict(settings["env"]),
    )
