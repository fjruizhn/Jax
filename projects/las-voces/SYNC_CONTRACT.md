# LAS VOCES — Sync Contract v0.1 (historical import)

This imported LV-000 document records the intended target families. The
implemented LV-002 contract is `projections/CONTRACT.md`; where they differ,
the LV-002 fail-closed rules govern. In particular, it does not claim ownership
of existing home-level monolithic files and Qwen is not projected until its
runtime format is verified.

## Canonical source
`axioma/projects/las-voces/`

Canónico:
- project.json
- PROJECT_CHARTER.md
- decisions/
- risks.json
- activity.ndjson
- agents/
- skills/

## Proyecciones
Shared:
- AGENTS.md

Codex:
- .codex/skills/

Claude Code:
- CLAUDE.md
- .claude/skills/
- .claude/agents/

Qwen Code:
- QWEN.md
- .qwen/skills/
- .qwen/agents/

## Regla
Canonical first. Las copias específicas de harness son generadas. `axioma sync las-voces` debe detectar drift por hash y negarse a sobrescribir silenciosamente modificaciones manuales.

## Agent Bus
Toda conversación/handoff relevante entre agentes debe producir un envelope auditable:
message_id, project_id, task_id, sender_agent, recipient_agent, intent, evidence_refs, authority_context, correlation_id, created_at, status.

Ariadna consume eventos verificados y actualiza `project.json`; nunca inventa estado.
