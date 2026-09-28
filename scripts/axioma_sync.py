#!/usr/bin/env python3
"""Deterministic, fail-closed LAS VOCES harness projection sync.

Usage:
    python3 scripts/axioma_sync.py las-voces [--check]

Canonical definitions live below ``projects/las-voces``.  Generated files are
never an input: a check compares their bytes with a freshly rendered projection
and reports drift without modifying anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


GENERATOR_VERSION = "1.0"
PROJECT_ID = "las-voces"
REQUIRED_DEFINITION_FIELDS = {
    "id", "version", "purpose", "scope", "inputs", "outputs", "authority",
    "allowed_actions", "forbidden_actions", "required_evidence", "handoff_contract",
}


class SyncError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SyncError(f"invalid canonical JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SyncError(f"invalid canonical JSON object: {path}")
    return value


def _source_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"


def _canonical(root: Path) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    project = root / "projects" / PROJECT_ID
    project_json = _read_json(project / "project.json")
    if project_json.get("project", {}).get("id") != PROJECT_ID:
        raise SyncError("missing or invalid LAS VOCES project identity")
    agent = _read_json(project / "agents" / "ariadna.json")
    skill = _read_json(project / "skills" / "las-voces-governance.json")
    envelope = _read_json(project / "sync" / "message-envelope.schema.json")
    for name, definition in (("agent", agent), ("skill", skill)):
        missing = REQUIRED_DEFINITION_FIELDS - set(definition)
        if missing:
            raise SyncError(f"invalid canonical {name}; missing fields: {', '.join(sorted(missing))}")
    required_envelope = {"message_id", "project_id", "task_id", "sender_agent", "recipient_agent", "intent", "evidence_refs", "authority_context", "correlation_id", "created_at", "status"}
    if set(envelope.get("required", [])) != required_envelope:
        raise SyncError("invalid MessageEnvelope contract")
    return project, project_json, agent, skill, envelope


def _source_hash(project: Path) -> str:
    files = [
        project / "project.json", project / "PROJECT_CHARTER.md", project / "SYNC_CONTRACT.md",
        project / "agents" / "ariadna.json", project / "skills" / "las-voces-governance.json",
        project / "sync" / "message-envelope.schema.json",
    ]
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file():
            raise SyncError(f"missing canonical source: {path}")
        digest.update(path.relative_to(project).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _notice() -> str:
    return "<!-- GENERATED FROM AXIOMA CANONICAL SOURCE. DO NOT EDIT DIRECTLY. -->\n"


def _governance(agent: dict[str, Any], skill: dict[str, Any]) -> str:
    return f"""Project source of truth: `projects/las-voces/`.

Human Authority is Fernando. Task ownership follows `project.json`; do not take
an unassigned task. Ariadna is PROPOSED / NOT ACTIVE. Qwen is the PRIMARY
BUILDER — LAS VOCES and works only in a worktree/sandbox.

No merge, deployment, production mutation, capability grant, or frozen-contract
change without explicit Human Authority. Evidence precedes DONE: include tests,
commit/PR and acceptance evidence in every handoff. Canonical definitions flow
only CANONICAL → GENERATED PROJECTIONS; never hand-maintain harness copies.

Canonical agent: {agent['id']} v{agent['version']} — {agent['purpose']}
Canonical skill: {skill['id']} v{skill['version']} — {skill['purpose']}
"""


def _render(project_json: dict[str, Any], agent: dict[str, Any], skill: dict[str, Any]) -> dict[str, bytes]:
    common = _governance(agent, skill)
    qwen = common + """
Qwen may read the project, implement an assigned task, write/run tests, and
prepare a commit/PR. Qwen may not merge, deploy, modify production, grant
capabilities, change frozen contracts, or claim DONE without evidence.
"""
    skill_text = _notice() + "# LAS VOCES governance\n\n" + common
    agent_text = _notice() + "# Qwen primary builder — LAS VOCES\n\n" + qwen
    return {
        "AGENTS.md": (_notice() + "# LAS VOCES — Codex instructions\n\n" + common).encode(),
        "CLAUDE.md": (_notice() + "# LAS VOCES — Claude Code instructions\n\n" + common).encode(),
        "QWEN.md": (_notice() + "# LAS VOCES — Qwen Code instructions\n\n" + qwen).encode(),
        ".qwen/skills/las-voces-governance/SKILL.md": skill_text.encode(),
        ".qwen/agents/primary-builder.md": agent_text.encode(),
    }


def _manifest(project: Path, projections: dict[str, bytes], source_hash: str, repo: Path) -> bytes:
    return _json_bytes({
        "schema_version": "1.0",
        "project_id": PROJECT_ID,
        "generator_version": GENERATOR_VERSION,
        "source_hash": source_hash,
        "source_commit": _source_commit(repo),
        "generated_at": "content-addressed",
        "projections": [
            {"canonical_source": "projects/las-voces", "target_harness": harness,
             "target_path": path, "generator_template_version": GENERATOR_VERSION,
             "source_hash": source_hash, "generated_hash": _sha256(data),
             "generation_timestamp": "content-addressed", "source_commit": _source_commit(repo)}
            for path, data in sorted(projections.items())
            for harness in [("codex" if path == "AGENTS.md" else "claude-code" if path == "CLAUDE.md" else "qwen-code")]
        ],
    })


def _expected(root: Path) -> tuple[Path, dict[str, bytes]]:
    project, project_json, agent, skill, _ = _canonical(root)
    projections = _render(project_json, agent, skill)
    projections["sync/manifest.json"] = _manifest(project, projections, _source_hash(project), root)
    return project, projections


def check(root: Path) -> int:
    project, expected = _expected(root)
    failures = []
    expected.pop("sync/manifest.json")
    for relative, data in expected.items():
        target = project / relative
        if not target.is_file():
            failures.append(f"SYNC REQUIRED: missing projection {relative}")
        elif target.read_bytes() != data:
            failures.append(f"DRIFT DETECTED: {relative}")
    manifest_path = project / "sync/manifest.json"
    try:
        manifest = _read_json(manifest_path)
        rendered_hashes = {path: _sha256(data) for path, data in expected.items()}
        entries = {entry.get("target_path"): entry for entry in manifest.get("projections", [])}
        if (
            manifest.get("project_id") != PROJECT_ID
            or manifest.get("generator_version") != GENERATOR_VERSION
            or manifest.get("source_hash") != _source_hash(project)
            or set(entries) != set(rendered_hashes)
            or any(entries[path].get("generated_hash") != digest for path, digest in rendered_hashes.items())
            or not manifest.get("source_commit")
        ):
            failures.append("DRIFT DETECTED: sync/manifest.json")
    except SyncError:
        failures.append("DRIFT DETECTED: sync/manifest.json")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("PASS: LAS VOCES canonical projections match")
    return 0


def _atomic_batch(project: Path, expected: dict[str, bytes]) -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix=".axioma-sync-", dir=project))
    backups: dict[Path, bytes | None] = {}
    try:
        staged: dict[Path, Path] = {}
        for relative, data in expected.items():
            staged_path = temp_dir / relative
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            staged_path.write_bytes(data)
            staged[project / relative] = staged_path
        for target, staged_path in staged.items():
            backups[target] = target.read_bytes() if target.exists() else None
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_path, target)
    except Exception:
        for target, old in backups.items():
            if old is None:
                target.unlink(missing_ok=True)
            else:
                restore = temp_dir / ("restore-" + hashlib.sha256(str(target).encode()).hexdigest())
                restore.write_bytes(old)
                os.replace(restore, target)
        raise
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def generate(root: Path) -> int:
    project, expected = _expected(root)
    _atomic_batch(project, expected)
    return check(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.project != PROJECT_ID:
        raise SyncError(f"unsupported project: {args.project}")
    root = Path(__file__).resolve().parents[1]
    return check(root) if args.check else generate(root)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SyncError as exc:
        print(f"SYNC FAILED CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
