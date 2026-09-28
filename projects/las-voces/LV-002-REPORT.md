# LAS VOCES — LV-002 CANONICAL PROJECTION REPORT

## Baseline

- Branch: `project/las-voces-lv002`
- HEAD: `805bae76396a8762fa3b117f3527cd27971b79ca`
- `origin/master`: same baseline commit.
- Working tree: clean before LV-002 edits.

## Contract

`agents/*.json` is the sole canonical projection input, with strict schema in
`schemas/agent.schema.json`. It is deliberately distinct from JAX normative
authority and from `project.json` operational state. Canonical data is
projected one-way only; there is no target import or reverse sync.

Codex mapping is an isolated `agents/<id>.toml`; Claude mapping is isolated
`agents/<id>.md`. Existing home-level `AGENTS.md`/`CLAUDE.md` and skills are
unmanaged to avoid replacing independent monolithic adapters. Qwen mapping is
not implemented because its runtime format is unverified and fails closed.

The detector is read-only: `IN_SYNC`, `MISSING`, `UNMANAGED`, `INVALID`,
`DRIFT`, and `STALE`. A modified artifact is never overwritten. Only a
payload-integrity-verified `STALE` artifact may be regenerated from newer
canonical data; `--dry-run` never writes.

## Qwen/Infra

**HUMAN_DECISION_REQUIRED.** Evidence supports `Qwen/Infra` as LV-001 task
owner and Qwen as a primary-builder label, but not a runtime identity,
installation, or adapter syntax. No identity is invented or activated.

## Evidence

- Ariadna is canonicalized as `PROPOSED_NOT_ACTIVE` with no runtime targets.
- Tests: `python3 -m unittest discover -s projects/las-voces/projections/tests -v`
- Adversarial cases include manual drift, old projection, unmanaged file,
  malformed canonical fields, absent dependency, unsupported target, Qwen
  fail-closed, secret-named field, dry run, and no reverse sync.
- `git diff --check` passes.

## Known limitations / human decision

The initial implementation intentionally does not compose into personal global
instruction or skill files. A Qwen adapter requires a human-approved runtime
identity and verified format. Neither limitation creates an alternate source
of authority.

## Status

**COMPLETE for the LV-002 contract:** Codex and Claude isolated projections,
deterministic provenance, validation, read-only drift detection, overwrite
gate, and tests are implemented. Qwen/Infra remains explicitly
`HUMAN_DECISION_REQUIRED`; no Qwen projection has been activated.
