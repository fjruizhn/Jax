# LAS VOCES — Project Charter

**Programa:** Axioma 3.0
**Estado:** PLANNING
**Versión:** 0.1.0
**Objetivo RC:** 30 de noviembre de 2026 *(objetivo operativo, no compromiso contractual)*

## North Star
> Hablar con JAX debe sentirse como presencia, no como usar un asistente de voz.

LAS VOCES es el canal humano de JAX: voz natural, gobernada, soberana y ubicua. No es un “TTS feature”. Incluye ASR, VoiceIdentity, Speech Renderer, streaming, barge-in, turn-taking, acceso móvil 24/7, smart glasses y telefonía UCM6300.

## Regla de producto
**Si suena como un GPS de los 2010, no está terminado.**

## Operating model
- Fernando: Human Authority / Product Owner.
- Ariadna (ACTIVE_GOVERNED): PM runtime gobernado; coordina pero no hace merge/deploy ni tiene autoridad humana.
- Qwen: primary builder en worktree aislado.
- Hyde/Claude: integrador / segundo carril.
- Thot/Codex: revisión adversarial, diagnóstico y gates.
- Jacobs: coordinador runtime, no project manager.

## Fuente de verdad
`axioma/projects/las-voces/`

El dashboard debe renderizar datos de `project.json`. Las configuraciones de Codex, Claude Code y Qwen son proyecciones sincronizadas, no fuentes independientes.
