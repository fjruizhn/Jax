# LAS VOCES — Sync Contract v0.1

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
Codex:
- `projects/las-voces/AGENTS.md` (nested instructions discovered by Codex from
  the project working directory). This environment has no repository-local
  Codex skill path; no `.codex/skills/` projection is invented.

Claude Code:
- `projects/las-voces/CLAUDE.md`

Qwen Code:
- `projects/las-voces/QWEN.md`
- `projects/las-voces/.qwen/skills/`
- `projects/las-voces/.qwen/agents/`

## Regla
Canonical first. Las copias específicas de harness son generadas. `axioma sync las-voces` debe detectar drift por hash y negarse a sobrescribir silenciosamente modificaciones manuales.

## Reconciliation
`python3 scripts/axioma_sync.py las-voces --check` is non-mutating and fails
closed on stale or manually edited projections. `python3 scripts/axioma_sync.py
las-voces` is the explicit reconciliation mode; it stages all files, atomically
replaces them, restores prior files if a replacement fails, writes
`sync/manifest.json`, and verifies the result.

## Agent Bus
Toda conversación/handoff relevante entre agentes debe producir un envelope auditable:
message_id, project_id, task_id, sender_agent, recipient_agent, intent, evidence_refs, authority_context, correlation_id, created_at, status.

Ariadna consume eventos verificados y actualiza `project.json`; nunca inventa estado.
