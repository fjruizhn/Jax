# LAS VOCES — Sync Contract v0.1

## Canonical source
`projects/las-voces/`

**LV-003 documentation correction (2026-09-28):** the original LV-000 import
used `axioma/projects/las-voces/` as its intended repository-relative target.
The LV-000 commit `9a63721` created the actual canonical tree at
`projects/las-voces/`, and LV-002's `scripts/axioma_sync.py` reads that same
path. This corrects documentation drift only; it does not rewrite historical
events or alter canonical content.

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

## 2026-10-03 · revisión de la proyección Qwen (Jax#320)

**DECISIÓN — Fernando, comentario del PR #320:** el constructor Qwen declara
alcance completo de herramientas en `project.json` (`tools: ["*"]`,
`disallowedTools: []`); el generador falla si faltan esos campos. El `*` es el
comodín explícito que Qwen Code 0.24.7 expande al catálogo de herramientas
disponible, incluidos los MCP registrados. La proyección usa `approvalMode:
bubble`, que el parser de esa versión acepta.

**HECHO — Codex, bundle instalado de Qwen Code 0.24.7:**
`chunks/chunk-AN36BHDM.js:562` expande `tools: ["*"]`; `:949` acepta
`bubble`; `:954` lee `tools`, `disallowedTools` y `approvalMode`; `:780`
resuelve `bubble` según el modo de la sesión que invoca:

| Padre | Subagente con `bubble` |
| --- | --- |
| `default` | `default` |
| `auto-edit` | `auto_edit` |
| `auto` | `auto` |
| `yolo` | `yolo` |
| `plan` | `default` |

**DECISIÓN — Fernando, 2026-10-03, vía jax-14:** conservar `bubble` y declarar
la única excepción, `plan → default`. El objetivo es evitar una escalada de
autonomía sin humano; en `default`, cada edición requiere confirmación humana.
La prueba con el parser y resolvedor reales de Qwen Code 0.24.7 fija esta
tabla para detectar cualquier cambio de comportamiento tras actualizar Qwen.

**HECHO — Codex:** `--check` enumera los archivos de `.qwen/agents/` y
`.qwen/skills/`, compara el manifiesto completo y exige un SHA de commit real
y ancestro. La CI ejecuta el check sobre el checkout versionado y usa el parser
real de Qwen 0.24.7. Una edición manual de `approvalMode` a `yolo` produjo
rc=1 y `DRIFT DETECTED` localmente. Las pruebas adversariales cubren
separadores, controles, sustitutos sueltos y nombres de skill inseguros.

**Alternativa descartada — Codex:** enumerar nombres de herramientas del bundle
congelaría el catálogo y omitiría herramientas MCP registradas después. El
comodín explícito conserva el alcance completo decidido por Fernando.

**Alternativas descartadas — Fernando, 2026-10-03:** fijar `plan` dejaría al
constructor sin poder construir; bloquear la invocación desde `plan` no es
posible desde esta proyección.

## Agent Bus
Toda conversación/handoff relevante entre agentes debe producir un envelope auditable:
message_id, project_id, task_id, sender_agent, recipient_agent, intent, evidence_refs, authority_context, correlation_id, created_at, status.

Ariadna consume eventos verificados y actualiza `project.json`; nunca inventa estado.
