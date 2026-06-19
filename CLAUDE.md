# JAX — Contexto para Claude Code (Hyde)

## LEER PRIMERO: ~/.claude/CLAUDE.md (políticas globales)

## ESTE PROYECTO
JAX es el ecosistema de orquestación multi-AI personal de Fernando Ruiz.
Honra la memoria de Jairo Urbina, pionero del software libre en Honduras.
En honor al Prof. Raúl Jacobs — El Director.

## ESTRUCTURA CRÍTICA
~/jax/
  config/config.toml    Personalidades/facetas — NO hardcodear aquí
  jax/                  Núcleo Python
    core/main.py        REPL + orquestador
    core/router.py      Router híbrido (keywords + clasificador LLM)
    muscles/base.py     Contrato de músculos — providers: deepseek/gemini/openai/kimi/zhipu
    muscles/ollama_muscle.py   JAX Local (GPU local)
    muscles/subprocess_muscle.py  Hyde (Claude Code)
  las_manos/            Sistema nervioso inhibitorio :7777
    motor_registry/     Motor Registry — despacha Kimi/Ada
  jacobs/               Director ejecutivo de pipelines
  jax-platform/         → Ver ~/jax-platform/CLAUDE.md

## REGLAS CRÍTICAS JAX
1. GPU_SEMAPHORE=1 — UNA sola inferencia Ollama a la vez
2. Kill switch: /etc/jax/PAUSE — si existe, JAX no invoca músculos
3. Credenciales SOLO en /etc/jax/.env — NUNCA en código
4. Backup obligatorio antes de modificar cualquier archivo (*backup-pre-<cambio>*)
5. py_compile en TODOS los archivos Python modificados
6. El router conoce: jax_local, jekyll, hyde, hipatia, thot, kimi + LABELS/ICONS/ALIASES/VALID_FACETAS
7. Modo autonomous de Jacobs: habilitado desde 18-jun-2026
8. Node.js: /home/fruiz/.nvm/versions/node/v24.16.0/bin/ — NUNCA NodeSource

## FACETAS ACTIVAS (7)
- jax_local: Qwen3:14b, Ollama local, GPU
- jekyll: DeepSeek V4 Flash, API
- hyde: Claude Code, subprocess
- hipatia: Gemini 2.5 Flash, API + grounding required_web
- thot: GPT-5.5, OpenAI API
- kimi: K2.7 Code, Moonshot API
- ada: GLM-5.2, Z.ai API (ZHIPU_API_KEY pendiente semana 22-jun)

## INFRAESTRUCTURA JAX
- Servicios systemd: jax-las-manos (:7777), jax-platform (:8080), jax-platform-frontend (:5173)
- MariaDB: jax_memory (tablas: conversations, messages, facts, jacobs_pipelines, jacobs_steps)
- Restic backup: Cloudflare R2, cron 3AM
