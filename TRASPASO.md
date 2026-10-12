# TRASPASO · Faro 0.8/P6 — Conectores (Kimi, Tier 2)

## Base y estado

- Worktree: `/home/fruiz/jax/.worktrees/faro-conectores`, rama `codex/faro-conectores`.
- Base: `ecfe07f3c2760ed43f53e7d1e858dad8e864e7e1` ("Merge pull request #402"), status limpio.
- Alcance implementado al 100% en este worktree; sin push (lo publica Codex CEO).
- Ownership respetado: solo se tocaron los archivos de abajo. Proxy y R2/B4 viven en otros worktrees.

## Archivos

- `jax/faro/conectores/__init__.py` — re-exports.
- `jax/faro/conectores/_comun.py` — validadores (`texto_plano`, `nombre_sano`, `modelo_sano`,
  `ruta_absoluta`, `url_loopback`), `MARCA_FICTICIA`, `credencial_ficticia`,
  `argv_rele`, `json_ordenado`, escape TOML. Solo stdlib.
- `jax/faro/conectores/codex.py` — `generar_codex()` -> `ConectorCodex(config_toml,
  requirements_toml, entorno, argv_rele)` + helpers de ruta.
- `jax/faro/conectores/kimi.py` — `generar_kimi()` -> `ConectorKimi(config_toml, mcp_json,
  ruta_config, ruta_mcp, entorno)`. mcp.json EFIMERO en `$KIMI_CODE_HOME` (cierra F-3).
- `jax/faro/conectores/claude.py` — `generar_claude()` -> `ConectorClaude(settings_json,
  mcp_config_json, argv, ruta_settings, ruta_mcp_config, entorno)`.
- `jax/faro/conectores/qwen_api.py` — `tools_a_function_calling(tools)` (traduccion pura).
- `tests/test_faro_conectores.py` — 29 pruebas, todas verdes.

## Decisiones (con fuente verificada)

1. **Codex**: `model_providers.faro-loopback` con `wire_api="responses"`,
   `requires_openai_auth=false`, `supports_websockets=false`, `env_key` (NO bearer_token
   directo). requirements.toml lleva `default_permissions`, `allowed_permission_profiles`
   con `:workspace` + `:read-only`, y `[mcp_servers.faro]` (namespace gestionado por
   requirements: "is managed by requirements and cannot be changed"). OTel off
   (`[otel]` tres exporter "none") y `[analytics] enabled=false`.
   Fuente: developers.openai.com/codex/config-reference y config-advanced; strings del
   binario codex-cli 0.162.1 (`core/src/config/requirements.rs`). Los fetches a
   raw.githubusercontent.com/openai/codex docs de requirements dieron 404; el contrato
   sale de strings del binario — es el hueco documentado mas abajo.
2. **Kimi**: `KIMI_CODE_HOME` reubica data dir (config.toml + mcp.json alli).
   `[providers."faro-loopback"]` type/base_url/api_key; `[models."faro/<modelo>"]`
   provider/model/max_context_size (default 262144); `default_model` y `telemetry=false`
   top-level. Fuente: docs oficiales kimi.com (config-files, mcp) + CLI local 2.1.1.
3. **Claude**: `--mcp-config <ruta>` (sin `--strict-mcp-config`); mcp-config con faro +
   chrome-devtools + playwright; binarios fijados por npm (parametro
   `bin_chrome_devtools`/`bin_playwright`, default del llamador); banderas base chrome-devtools
   `--headless --isolated --no-usage-statistics --no-performance-crux --redactNetworkHeaders`
   y playwright `--headless --isolated` (constitucion, seccion MCP de navegador;
   guion `claude-skills/bin/mcp-navegadores-pin.sh`). `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`,
   `ANTHROPIC_BASE_URL` loopback, token ficticio. Fuente: binario claude v2.1.296 (`--help`).
4. **Token**: siempre por `--token-file` hacia `python -m jax.faro.relay` (relay.py:10);
   jamas argv/env. Credenciales de config llevan `FARO-CREDENCIAL-FICTICIA-<nombre>`;
   provisionador real las reemplaza en el hogar efimero.
5. **Falla cerrada**: TypeError (tipo) / ValueError (valor); rechaza URL externa,
   userinfo, query, fragment, `https`, rutas relativas, `..`, caracteres de control,
   nombres con inyeccion (`;`, `|`, mayusculas, espacios). Sin GLM/ZCode.

## Pruebas

Comando: `cd /home/fruiz/jax/.worktrees/faro-conectores && /tmp/jax-faro-r2-venv/bin/python -m pytest tests/test_faro_conectores.py -q`

Salida: `29 passed in 0.05s` (Python 3.14.4, pytest 9.1.1).

## Huecos / pendientes (NO cerrados aqui)

- **Contrato requirements.toml de Codex** verificado solo por strings del binario 0.162.1;
  la doc publica de managed config no se pudo obtener (404). Aceptar riesgo o re-verificar
  contra release notes antes de produccion.
- **Variantes de navegador por maquina** (`--executablePath`, `--browser chromium` en stxtra)
  quedan como parametros del llamador (`flags_*`), no hardcodeadas; falta decision por host.
- **S1 real**: los conectores no resuelven proveedor real; el modelo detras del loopback
  lo define el Puerto (S1).
- **Composicion con `Servicio`/lanzador**: estos generadores producen bytes; quien escribe
  los archivos en el hogar efimero y lanza el proceso es el modulo de composicion (otro agente).
- **CI/pisos**: no se tocaron workflows ni pisos (prohibido en este alcance).
- **Auditoria y despliegue**: pendientes (Codex CEO verifica/publica; auditoria Sol).

## Continuacion sugerida

1. CEO Codex revisa y empuja la rama; auditoria Tier 3 sobre el diff.
2. El agente de composicion (`Servicio`) importa estos generadores y monta el hogar efimero.
3. Subir piso de pruebas de CI con `tests/test_faro_conectores.py` en el job correspondiente.
