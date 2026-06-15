# MISIÓN DE INVESTIGACIÓN — Hipatia
**Para:** Hipatia (faceta investigadora de JAX)
**Objetivo:** Recolectar datos reales con fuentes para diseñar la sala de control (TUI) y el sistema de skills/plugins por faceta del Hubermech.
**Regla:** Fuentes reales con URLs. Nada inventado. Si algo no existe para un motor, decirlo con honestidad — eso también es un dato.

---

## CONTEXTO

Acabamos de terminar **LAS MANOS**, el sistema de capacidades del equipo JAX:
- API REST local en hall9000 (127.0.0.1:7777)
- Flujo: intención → planner → policy engine → human gate → dry-run → worker → audit forense
- Kill switch en vuelo (watcher async + ssh -tt), probado en fuego real
- Permisos por faceta: Hipatia solo lee, Jekyll escribe en staging, Thot audita, Hyde ejecuta con human gate en prod

Ahora vamos a construir, en este orden (validado por Thot):
1. La **sala de control TUI** ("antes de darle manos al equipo, dale ojos al operador")
2. El **sistema de skills persistente** por faceta
3. La **conexión de cada faceta a su ecosistema de plugins comunitario** (el Hubermech)

---

## INVESTIGACIÓN 1 — Sala de control TUI

Busca mejores prácticas y proyectos de referencia para interfaces de terminal (TUI) tipo "sala de control operacional" o "mission control". Específicamente:

- Proyectos construidos con **Textual** (Python) que sean dashboards de monitoreo o control en vivo
- Patrones de diseño para mostrar: cola de tareas pendientes, aprobación humana, kill switch siempre visible, timeline de auditoría en streaming
- Cómo manejan TUIs como **lazygit**, **k9s**, **btop** la actualización en vivo y la jerarquía visual
- Widgets de Textual relevantes: tablas vivas, paneles fijos, streams de log, modales de confirmación
- Cita tus fuentes con URLs

---

## INVESTIGACIÓN 2 — OpenClaw reverse engineering

Investiga el proyecto open source **OpenClaw**. 
- Qué es y para qué sirve
- Cómo está arquitecturado
- Qué patrones podríamos aprender de él para nuestra sala de control o para LAS MANOS
- URLs del repo y documentación, detalles concretos

---

## INVESTIGACIÓN 3 — Skills persistentes por agente

Investiga cómo los frameworks de agentes IA modernos implementan "skills" o "tools" persistentes y reutilizables:

- Cómo **Claude (Anthropic)** maneja skills y el formato `SKILL.md`
- Cómo **OpenAI** maneja function calling y tools persistentes
- Cómo **Gemini** (vos misma) maneja function declarations y tools
- Si existe un formato común o un patrón para que un agente guarde una capacidad aprendida y la recargue después
- Patrones de "skill library" o "tool registry" en proyectos open source de agentes (**LangChain, CrewAI, AutoGPT**, etc.)

---

## INVESTIGACIÓN 4 — Ecosistemas de plugins por motor (el corazón del Hubermech)

La visión: cada faceta conectada al marketplace/repo de plugins que su propia comunidad ya construyó, de forma ordenada y reproducible.

### A) Claude Code (para Hyde)
- Cómo funciona el sistema de plugins de Claude Code
- El marketplace de plugins de Anthropic: cómo se instalan, dónde viven, cómo se versionan
- El ecosistema **MCP (Model Context Protocol)**: servers disponibles, cómo se agregan
- Cómo "superpowers" y plugins similares se distribuyen e instalan
- Formato de un plugin: estructura de carpetas, manifiesto, etc.

### B) OpenAI / Codex (para Thot)
- Sistema de tools y function calling persistente
- Codex CLI: extensiones, configuración, qué se puede agregar
- Repos comunitarios de tools para GPT
- GPT Actions / custom tools: cómo se empaquetan y reutilizan

### C) Gemini (para vos, Hipatia)
- Gemini extensions y cómo se instalan
- Function declarations reutilizables
- Gemini CLI si existe, y su ecosistema
- Repos comunitarios de tools para Gemini

### D) Ollama (para JAX local)
- Cómo se distribuyen y cargan modelos y herramientas
- Modelfiles, y si hay ecosistema de "tools" para modelos locales

---

## INVESTIGACIÓN 5 — Patrón de organización limpia

Busca cómo proyectos serios organizan plugins de múltiples fuentes de forma ordenada y reproducible:

- Estructura de repos para plugins propios (¿un repo por faceta?)
- Cómo versionar, instalar, actualizar, listar plugins de forma limpia
- Patrones tipo "package manager" para skills/plugins de agentes
- Cómo evitar el caos de dependencias cuando hay múltiples fuentes

---

## ENTREGABLE

Para cada investigación: fuentes reales con URLs, formatos concretos, ejemplos de estructura de carpetas. Organizá los hallazgos para que el equipo pueda decidir arquitectura con datos, no con suposiciones.

Es muy probable que los ecosistemas estén desparejos (Claude Code maduro, otros menos). Eso NO es un problema — es información valiosa. Nos dice dónde beber de la comunidad directamente y dónde construir el puente nosotros.

**El que supone se equivoca. Traé datos, no teoría.**
