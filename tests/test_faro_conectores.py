"""El Faro 0.8/P6: los conectores son generadores puros y deterministicos.

Cubre el contrato de `jax/faro/conectores/`: salida determinista, JSON/TOML
parseables, sin home del usuario en la salida, credencial siempre ficticia y
explicita, token SOLO por `--token-file`, rechazo cerrado de URLs externas /
rutas relativas / caracteres de control, y banderas exactas de navegador.
"""
from __future__ import annotations

import json
import tomllib

import pytest

from jax.faro.conectores import (
    generar_claude,
    generar_codex,
    generar_kimi,
    tools_a_function_calling,
)
from jax.faro.conectores._comun import MARCA_FICTICIA
from jax.faro.conectores.claude import FLAGS_CHROME_DEVTOOLS, FLAGS_PLAYWRIGHT

URL = "http://127.0.0.1:8377"
SOCKET = "/run/faro/abc123/puerto.sock"
TOKEN = "/run/faro/abc123/token"
HOGAR = "/tmp/faro-hogar-abc123"
PYTHON = "/usr/bin/python3"
BIN_CD = "/opt/mcp/chrome-devtools/node_modules/.bin/chrome-devtools-mcp"
BIN_PW = "/opt/mcp/playwright/node_modules/.bin/playwright-mcp"


def _codex(**kw):
    base = dict(nombre="ejec-1", modelo="modelo-x", base_url=URL, socket=SOCKET, token_file=TOKEN)
    base.update(kw)
    return generar_codex(**base)


def _kimi(**kw):
    base = dict(
        nombre="ejec-1",
        modelo="modelo-x",
        base_url=URL,
        socket=SOCKET,
        token_file=TOKEN,
        kimi_code_home=HOGAR,
    )
    base.update(kw)
    return generar_kimi(**base)


def _claude(**kw):
    base = dict(
        nombre="ejec-1",
        base_url=URL,
        socket=SOCKET,
        token_file=TOKEN,
        hogar=HOGAR,
        bin_chrome_devtools=BIN_CD,
        bin_playwright=BIN_PW,
    )
    base.update(kw)
    return generar_claude(**base)


def _sin_home(*salidas: str) -> None:
    for salida in salidas:
        for prohibido in ("~/.codex", "~/.claude", "~/.kimi-code", "/home/", "/root/"):
            assert prohibido not in salida, f"home del usuario filtrado a la salida: {prohibido}"


# --- determinismo y parseo -------------------------------------------------


def test_codex_es_determinista_y_parseable():
    a, b = _codex(), _codex()
    assert a.config_toml == b.config_toml
    assert a.requirements_toml == b.requirements_toml
    assert tomllib.loads(a.config_toml)
    assert tomllib.loads(a.requirements_toml)


def test_kimi_es_determinista_y_parseable():
    a, b = _kimi(), _kimi()
    assert a.config_toml == b.config_toml
    assert a.mcp_json == b.mcp_json
    assert tomllib.loads(a.config_toml)
    assert json.loads(a.mcp_json)


def test_claude_es_determinista_y_parseable():
    a, b = _claude(), _claude()
    assert a.settings_json == b.settings_json
    assert a.mcp_config_json == b.mcp_config_json
    assert json.loads(a.settings_json)
    assert json.loads(a.mcp_config_json)


# --- sin home del usuario, credencial ficticia explicita --------------------


def test_la_salida_no_fuga_el_home_del_usuario():
    c, k, cl = _codex(), _kimi(), _claude()
    _sin_home(c.config_toml, c.requirements_toml, k.config_toml, k.mcp_json, cl.settings_json, cl.mcp_config_json)


def test_la_credencial_es_ficticia_y_explicita():
    c, k, cl = _codex(), _kimi(), _claude()
    assert MARCA_FICTICIA in k.config_toml
    assert MARCA_FICTICIA in cl.settings_json
    assert all(MARCA_FICTICIA in v for v in c.entorno.values())
    assert all(MARCA_FICTICIA not in v for v in k.entorno.values())  # Kimi: home, no credencial
    assert MARCA_FICTICIA in cl.entorno["ANTHROPIC_AUTH_TOKEN"]
    # y nunca un valor que parezca una key real (sk-,Bearer, etc.)
    for salida in (k.config_toml, cl.settings_json, json.dumps(c.entorno)):
        for parecido in ("sk-", "Bearer ", "ghp_", "AIza"):
            assert parecido not in salida


# --- el token SOLO via --token-file ------------------------------------------


def test_el_token_nunca_viaja_por_argv_ni_env():
    c, k, cl = _codex(), _kimi(), _claude()
    argv_reles = [c.argv_rele, json.loads(k.mcp_json)["mcpServers"]["faro"]["args"],
                  json.loads(cl.mcp_config_json)["mcpServers"]["faro"]["args"]]
    for argv in argv_reles:
        assert argv[argv.index("--token-file") + 1] == TOKEN
        assert "--token-file" in argv and "--socket" in argv
        assert TOKEN not in " ".join(argv[: argv.index("--token-file")])  # no antes del flag
        assert TOKEN not in " ".join(argv[argv.index("--token-file") + 2 :])  # no despues
    for env in (c.entorno, k.entorno, cl.entorno):
        assert TOKEN not in env.values()


# --- contenido puntual por motor ----------------------------------------------


def test_codex_apunta_al_loopback_y_apaga_telemetria():
    c = _codex()
    cfg = tomllib.loads(c.config_toml)
    assert cfg["model"] == "modelo-x"
    prov = cfg["model_providers"]["faro-loopback"]
    assert prov["base_url"] == URL
    assert prov["wire_api"] == "responses"
    assert prov["requires_openai_auth"] is False
    assert prov["supports_websockets"] is False
    assert cfg["otel"] == {"exporter": "none", "metrics_exporter": "none", "trace_exporter": "none"}
    assert cfg["analytics"] == {"enabled": False}
    req = tomllib.loads(c.requirements_toml)
    assert ":workspace" in req["allowed_permission_profiles"]
    assert ":read-only" in req["allowed_permission_profiles"]
    assert "default_permissions" in req
    assert req["mcp_servers"]["faro"]["args"] == c.argv_rele[1:]


def test_kimi_usa_home_efimero_y_mcp_efimero():
    k = _kimi()
    cfg = tomllib.loads(k.config_toml)
    assert cfg["default_model"] == "faro/modelo-x"
    assert cfg["telemetry"] is False
    assert cfg["providers"]["faro-loopback"]["base_url"] == URL
    modelo = cfg["models"]["faro/modelo-x"]
    assert modelo["max_context_size"] >= 1
    assert k.ruta_config == f"{HOGAR}/config.toml"
    assert k.ruta_mcp == f"{HOGAR}/mcp.json"
    assert k.entorno == {"KIMI_CODE_HOME": HOGAR}
    mcp = json.loads(k.mcp_json)
    assert list(mcp["mcpServers"]) == ["faro"]


def test_claude_monta_faro_y_navegadores_fijados_sin_strict():
    cl = _claude()
    mcp = json.loads(cl.mcp_config_json)["mcpServers"]
    assert set(mcp) == {"faro", "chrome-devtools", "playwright"}
    assert mcp["chrome-devtools"]["command"] == BIN_CD
    assert mcp["chrome-devtools"]["args"] == FLAGS_CHROME_DEVTOOLS
    assert mcp["playwright"]["command"] == BIN_PW
    assert mcp["playwright"]["args"] == FLAGS_PLAYWRIGHT
    assert "--headless" in mcp["chrome-devtools"]["args"]
    assert "--redactNetworkHeaders" in mcp["chrome-devtools"]["args"]
    assert "--strict-mcp-config" not in cl.argv
    assert cl.argv[:2] == ["claude", "--mcp-config"]
    ajustes = json.loads(cl.settings_json)
    assert ajustes["env"]["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert ajustes["env"]["ANTHROPIC_BASE_URL"] == URL


# --- traduccion Qwen -----------------------------------------------------------


def test_qwen_traduce_tools_list_a_function_calling():
    tools = [
        {"name": "leer", "description": "lee", "inputSchema": {"type": "object"}},
        {"name": "escribir"},
    ]
    fc = tools_a_function_calling(tools)
    assert fc[0] == {
        "type": "function",
        "function": {"name": "leer", "description": "lee", "parameters": {"type": "object"}},
    }
    assert fc[1]["function"]["parameters"] == {}
    assert tools_a_function_calling([]) == []


# --- fallo cerrado --------------------------------------------------------------


@pytest.mark.parametrize("mala", ["http://example.com", "https://127.0.0.1:1/x", "http://user@127.0.0.1",
                                  "http://127.0.0.1/?q=1", "http://127.0.0.1/#f", "http://[::1]:80"])
def test_rechaza_url_externa_o_con_credenciales(mala):
    with pytest.raises(ValueError):
        _codex(base_url=mala)
    with pytest.raises(ValueError):
        _kimi(base_url=mala)
    with pytest.raises(ValueError):
        _claude(base_url=mala)


@pytest.mark.parametrize("mala", ["socket", "./socket", "", "/run/../etc/x", "/run/x\ny"])
def test_rechaza_ruta_relativa_o_con_control(mala):
    with pytest.raises((ValueError, TypeError)):
        _codex(socket=mala)
    with pytest.raises((ValueError, TypeError)):
        _kimi(token_file=mala)
    with pytest.raises((ValueError, TypeError)):
        _claude(hogar=mala)


@pytest.mark.parametrize("mala", ["ejec 1", "EJEC", "a;b", "a|b", "..", ".oculto", "a\nb"])
def test_rechaza_nombre_con_inyeccion(mala):
    with pytest.raises((ValueError, TypeError)):
        _codex(nombre=mala)
    with pytest.raises((ValueError, TypeError)):
        _kimi(nombre=mala)
    with pytest.raises((ValueError, TypeError)):
        _claude(nombre=mala)


def test_falla_cerrada_con_dato_faltante_o_de_otro_tipo():
    with pytest.raises(TypeError):
        _codex(modelo=None)
    with pytest.raises(TypeError):
        _kimi(max_contexto="262144")
    with pytest.raises(ValueError):
        _kimi(max_contexto=0)
    with pytest.raises(ValueError):
        _kimi(tipo="glm")
    with pytest.raises(TypeError):
        tools_a_function_calling("no-lista")
    with pytest.raises(ValueError):
        tools_a_function_calling([{"description": "sin nombre"}])
    with pytest.raises(ValueError):
        tools_a_function_calling([{"name": "a\nb"}])
