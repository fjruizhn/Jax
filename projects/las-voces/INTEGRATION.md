# LV-000 integration mapping

## Canonical state

`project.json` is the single canonical operational-state document for LAS
VOCES in this directory. `activity.ndjson` is its append-only factual history.
The charter, sync contract, Ariadna specification, and dashboard are imported
design/supporting artifacts; they do not grant authority or supersede JAX
policy. Risks and task state remain fields of `project.json` until an approved
canonical data-contract change deliberately separates them.

The project is repository-managed under `projects/las-voces/`, because current
JAX has no established project-control directory while it is the Axioma
repository that already owns the root agent adapter and project-related
governance code. This registry is deliberately separate from `jacobs/`, which
is runtime pipeline orchestration.

## Dashboard

`index.html` fetches sibling `project.json` with `cache: no-store` when served.
Its `EMBEDDED` object is an offline fallback only. No UI route or static-asset
registry exists in current JAX for project dashboards; wiring this renderer into
the separate `jax-platform` frontend is deferred. The future integration point
is a project-data API that serves this canonical document plus a frontend route
that renders `index.html` (or its successor) from that API, without copying
state into the frontend.

## Ariadna

`agents/ARIADNA.md` is an imported proposal, not an active autonomous-agent
registration. Its stated prohibitions remain binding: no merge, deploy,
production mutation, capability grant, inferred human authority, or DONE claim
without acceptance evidence.

## LV-002 harness mapping (factual as of import)

| Canonical LAS VOCES artifact | Codex | Claude Code | Qwen |
| --- | --- | --- | --- |
| Shared instructions | `AGENTS.md` (repository root) | root `AGENTS.md` is referenced by `docs/agents/claude.md` | expected `QWEN.md` (not present) |
| Provider instruction shim | no tracked Codex instruction shim; `.Codex/napkin.md` exists | `docs/agents/claude.md`; `.claude/settings.json` | expected `QWEN.md` (not present) |
| Skills | no tracked `.codex/skills/` | no tracked `.claude/skills/` | expected `.qwen/skills/` (not present) |
| Agent definitions | `docs/agents/` is the existing tracked location | `docs/agents/claude.md`; no tracked `.claude/agents/` | expected `.qwen/agents/` (not present) |
| MCP/tools | no repository-local registry found | `.claude/settings.json` is configuration, not a LAS VOCES registry | expected `.qwen/` configuration (not present) |

LV-002 should first define an approved canonical schema in this project
directory, then generate provider projections with hashes and fail on manual
drift. It must not overwrite the existing JAX root adapter or Claude shim.
