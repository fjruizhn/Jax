# LV-002 canonical projection contract

`agents/*.json` is the only projection input. It is a project registry, not
JAX normative authority: `policy/**` remains authoritative. `project.json` is
the LAS VOCES operational-state document; its agent table is contextual
summary only and is never read by this generator. `agents/ARIADNA.md` remains
the LV-000 imported evidence; `agents/ariadna.json` is the sole canonical
agent representation.

The source direction is only `canonical → projection`. This tool has no
import, reverse-sync, or target-to-canonical write operation.

## Schema and lifecycle

`schemas/agent.schema.json` is strict (`additionalProperties: false`). The
fields exist because runtimes need identity/description/instructions
(`id`, `display_name`, `role`, `purpose`, `instructions`), lifecycle routing
(`status`, `runtime_targets`, `version`), safe dependency and execution
boundaries (`skills`, `capabilities`, `tools`, `constraints`, `authority`),
and auditability (`schema_version`, `provenance`). No field grants JAX
authority. The closed lifecycle vocabulary is `PROPOSED_NOT_ACTIVE`, `ACTIVE`,
`DISABLED`, `RETIRED`; only explicit `ACTIVE` agents emit artifacts.

Canonical SHA-256 is UTF-8 JSON serialized with sorted keys and compact
separators. Every generated artifact carries a comment marker with canonical
agent id/source/hash/version, target, schema/generator versions and SHA-256 of
the payload. Timestamps are intentionally omitted so bytes are deterministic.

## Runtime mapping

| Field | Codex agent TOML | Claude agent Markdown | Qwen |
| --- | --- | --- | --- |
| `id` | SUPPORTED: filename/name | SUPPORTED: filename | UNSUPPORTED pending runtime evidence |
| display/role/purpose/instructions | TRANSFORMED to TOML description/instructions | TRANSFORMED to heading/body | UNSUPPORTED |
| status/runtime_targets | TRANSFORMED to emission gate | TRANSFORMED to emission gate | UNSUPPORTED |
| skills | SUPPORTED only when a canonical skill artifact exists; otherwise fail | same | UNSUPPORTED |
| capabilities/tools | never a runtime grant; `NONE` only in LV-002 | never a runtime grant; `NONE` only | UNSUPPORTED |
| constraints/authority | TRANSFORMED to restrictive prompt text, not enforcement | same | UNSUPPORTED |
| provenance/version | SUPPORTED managed marker | SUPPORTED managed marker | UNSUPPORTED |

The implemented paths are isolated agent definitions:
`<sandbox>/.codex/agents/<id>.toml` and
`<sandbox>/.claude/agents/<id>.md` (the CLI output root is the corresponding
runtime directory). `~/.codex/AGENTS.md`, `~/.claude/CLAUDE.md`, their skills
trees, `QWEN.md`, and `.qwen/skills` are not owned in LV-002. Existing
monolithic files are therefore never silently merged or overwritten.

Qwen paths requested by the project (`QWEN.md`, `.qwen/agents/`,
`.qwen/skills/`) are recognized but the adapter returns
`HUMAN_DECISION_REQUIRED` because neither an installation nor a file format
was evidenced. This is fail-closed, not a projection.

## Drift and write policy

`inspect` is read-only and returns per expected path: `IN_SYNC`, `MISSING`,
`UNMANAGED`, `INVALID`, `DRIFT`, or `STALE`. A payload-hash mismatch is
`DRIFT`; an intact payload generated from an older canonical hash is `STALE`.
Malformed markers are `INVALID`. Only exact computed managed paths are
inspected; unrelated personal files are out of scope.

`generate` validates all canonical inputs before planning. It creates MISSING
files and updates only integrity-verified STALE files. `DRIFT`, `UNMANAGED`,
or `INVALID` aborts before any write; there is no force mode. `--dry-run`
prints planned paths and writes nothing. Tests use fixtures/sandboxes only;
development does not touch home configurations.

## Qwen/Infra resolution

**HUMAN_DECISION_REQUIRED.** Evidence establishes task owner `Qwen/Infra`,
primary-builder label `Qwen local`, and expected target paths, but not an
installed runtime, runtime name, or whether `/Infra` is a lane or identity.

1. Recommended pending approval: canonical id `qwen`, runtime name `Qwen
   Code`, role `Primary Builder — LAS VOCES`, target Qwen paths, non-active
   until LV-001 verifies format/runtime.
2. Create distinct `qwen` and named infra identities after a human authority
   decision.
3. Composite `qwen-infra` is not selected because it conflates identities.
