faceta: hipatia

# JAX 3.0 — Documento de Arquitectura Completa
**Tarea:** Investigación + Propuesta de arquitectura  
**Entrega:** ~/jax/prompts/jax3_arquitectura_result.md

---

## CONTEXTO DEL ECOSISTEMA ACTUAL

**Hardware:**
- hall9000: Ryzen 5 8500G, RX 9060 XT 16GB RDNA4, 32GB RAM, Ubuntu 24.04.4
- Red: 172.16.20.5, VMs en 172.16.20.10 (prod) y 172.16.20.12 (dev)

**JAX 2.0 actual:**
- REPL terminal con voz (Kokoro TTS + Whisper STT)
- 4 facetas: jax_local (qwen3:14b/Ollama), jekyll (DeepSeek V4-Flash), hipatia (Gemini 2.5 Flash + grounding), hyde (Claude via Claude Code)
- Memoria persistente en MariaDB (~/jax/jax/memory/db.py)
- Modo --task autónomo recién implementado
- Router con clasificador local

**Productos que JAX debe servir:**
- AteneaERP: Laravel 13 + React 19, SaaS multi-tenant LATAM PyMEs
- HAMMURABI: Banking SaaS LATAM, orchestrator JP
- Black Diamond AI: módulo de IA embebido en AteneaERP

---

## LO QUE SE QUIERE CONSTRUIR: JAX 3.0

La visión es un ecosistema unificado donde JAX opera como sistema nervioso central con DOS interfaces que funcionan como una sola:

**Interface 1 — TUI (terminal, hall9000):**
- Voz siempre activa (sin /escucha ni /voz — conversación natural)
- Colores/tema que cambian según la personalidad activa
- Puede mostrar imágenes inline
- Copy/paste normal
- Acceso completo al servidor

**Interface 2 — WebUI (browser, red local):**
- Equivalente a claude.ai pero self-hosted
- Todas las facetas disponibles como modelos seleccionables
- Voz integrada sin comandos
- Plugins/tools que ejecutan bash en hall9000
- Imágenes, video, documentos
- Historial compartido con TUI

**Memoria unificada:**
- Una sola base de datos contextual
- Toda interacción (TUI + WebUI + AteneaERP + HAMMURABI) alimenta la misma memoria
- JAX aprende de todo lo que pasa en el ecosistema

**Agentes autónomos:**
- JAX puede recibir una tarea y ejecutarla solo, sin intervención humana
- Puede manejar el servidor: archivos, servicios, git, docker, etc.
- Puede investigar, escribir código, probarlo y reportar resultado
- No necesita que Fernando esté sentado en el teclado

---

## PREGUNTAS QUE DEBE RESPONDER LA INVESTIGACIÓN

### 1. Open WebUI como base de JAX WebUI
- ¿Open WebUI puede conectarse a Ollama + DeepSeek + Gemini + Claude simultáneamente?
- ¿Sus Tools/Functions permiten ejecutar bash arbitrario en el host?
- ¿Soporta MariaDB como backend de memoria (no SQLite)?
- ¿Tiene TTS/STT comparable a Kokoro + Whisper que ya tenemos?
- ¿Puede compartir sesión/memoria con el TUI terminal?
- ¿Cuál es el comando exacto de instalación Docker para hall9000 con MariaDB?

### 2. TUI moderno para JAX
- ¿Existe un framework TUI Python que soporte imágenes inline, colores dinámicos, y copy/paste normal? (Textual, Rich, Urwid, etc.)
- ¿Puede un TUI moderno mostrar imágenes reales (no ASCII art)?
- ¿Cómo se implementa voz siempre activa (VAD — Voice Activity Detection) sin comando manual?

### 3. Memoria unificada
- ¿Cómo diseñar un schema MariaDB único que sirva a JAX TUI + Open WebUI + AteneaERP + HAMMURABI?
- ¿Existe un servicio de memoria tipo mem0, Zep, o similar que se pueda self-hostear y conectar a todos?

### 4. Agencia real (LAS MANOS)
- ¿Cómo implementar un executor de herramientas (bash, filesystem, git, docker, systemctl) que todas las facetas puedan invocar?
- ¿MCP local embebido en JAX o servidor MCP separado? ¿Cuál es más estable en 2026?
- ¿Existe una librería Python madura para esto (smolagents, langchain tools, etc.)?

### 5. Integración AteneaERP + HAMMURABI
- ¿Cómo exponer las facetas de JAX como API REST para que AteneaERP las consuma?
- ¿Ollama ya expone una API compatible con OpenAI que Laravel puede consumir directamente?

---

## FORMATO DE ENTREGA

Documento Markdown con:
- Resumen ejecutivo (máximo 1 párrafo)
- Tabla de decisiones: para cada pregunta, respuesta verificada + fuente + recomendación
- Diagrama de arquitectura en texto (ASCII o Mermaid)
- Plan de implementación en fases ordenadas por impacto/esfuerzo
- Lista de dependencias a instalar con comandos exactos
- Riesgos identificados

Usar solo fuentes verificadas 2026. No suposiciones.
