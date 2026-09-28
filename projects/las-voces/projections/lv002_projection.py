#!/usr/bin/env python3
"""Canonical-only LAS VOCES agent projection and read-only drift inspection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

SCHEMA_VERSION = "1.0"
GENERATOR_VERSION = "1.0.0"
TARGETS = {"codex", "claude", "qwen"}
PROJECTABLE_TARGETS = {"codex", "claude"}
STATUSES = {"PROPOSED_NOT_ACTIVE", "ACTIVE", "DISABLED", "RETIRED"}
ID_RE = re.compile(r"^[a-z][a-z0-9-]{1,62}$")
SECRET_KEY_RE = re.compile(r"(?:password|secret|api[_-]?key|token|credential)", re.I)


class ContractError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def read_agents(root: Path) -> list[dict[str, Any]]:
    agents = []
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"invalid canonical JSON: {path}: {exc}") from exc
        data["_path"] = path
        validate_agent(data, root)
        agents.append(data)
    if not agents:
        raise ContractError("no canonical agents found")
    ids = [a["id"] for a in agents]
    if len(ids) != len(set(ids)):
        raise ContractError("identity conflict: duplicate canonical agent id")
    return agents


def validate_agent(agent: dict[str, Any], root: Path) -> None:
    required = {"schema_version", "id", "display_name", "status", "role", "purpose", "instructions", "runtime_targets", "skills", "capabilities", "tools", "constraints", "authority", "provenance", "version", "_path"}
    extra = set(agent) - required
    missing = required - set(agent)
    def reject_secrets(value: Any, key: str = "") -> None:
        if SECRET_KEY_RE.search(key):
            raise ContractError("secrets are forbidden in canonical agent data")
        if isinstance(value, dict):
            for child_key, child in value.items():
                reject_secrets(child, child_key)
        elif isinstance(value, list):
            for child in value:
                reject_secrets(child, key)
    reject_secrets({k: v for k, v in agent.items() if k != "_path"})
    if missing or extra:
        raise ContractError(f"schema fields missing={sorted(missing)} extra={sorted(extra)}")
    if not isinstance(agent["id"], str) or not ID_RE.fullmatch(agent["id"]):
        raise ContractError("invalid id")
    if agent["schema_version"] != SCHEMA_VERSION or agent["status"] not in STATUSES or agent["role"] not in {"PROJECT_MANAGER", "PRIMARY_BUILDER", "INTEGRATOR", "ADVERSARIAL_REVIEWER", "RUNTIME_COORDINATOR"}:
        raise ContractError("unsupported status or role")
    if not all(isinstance(agent[k], list) for k in ("instructions", "runtime_targets", "skills", "capabilities", "tools", "constraints")):
        raise ContractError("agent list field has wrong type")
    if not agent["instructions"] or not agent["constraints"]:
        raise ContractError("instructions and constraints are required")
    if len(agent["runtime_targets"]) != len(set(agent["runtime_targets"])) or not set(agent["runtime_targets"]) <= TARGETS:
        raise ContractError("invalid runtime target")
    if len(agent["skills"]) != len(set(agent["skills"])) or not all(isinstance(x, str) and ID_RE.fullmatch(x) for x in agent["skills"]):
        raise ContractError("invalid skill reference")
    for skill in agent["skills"]:
        if not (root.parent / "skills" / f"{skill}.json").is_file():
            raise ContractError(f"missing canonical skill reference: {skill}")
    if set(agent["capabilities"]) - {"PROJECT_COORDINATION"} or len(agent["capabilities"]) != len(set(agent["capabilities"])):
        raise ContractError("unsupported capability")
    if agent["tools"] != ["NONE"]:
        raise ContractError("tools cannot be projected safely")
    authority = agent["authority"]
    forbidden = {"MERGE", "DEPLOY", "PRODUCTION_MUTATION", "CAPABILITY_GRANT", "INFER_HUMAN_AUTHORITY", "DECLARE_DONE_WITHOUT_EVIDENCE"}
    if set(authority) != {"can", "cannot", "human_approval_required"} or set(authority["can"]) - {"PROJECT_COORDINATION"} or not set(authority["cannot"]) <= forbidden or not authority["cannot"]:
        raise ContractError("invalid authority")
    source = agent["provenance"]
    expected_source = f"agents/{agent['id']}.json"
    if set(source) != {"canonical_source", "evidence_refs"} or source["canonical_source"] != expected_source or not source["evidence_refs"]:
        raise ContractError("invalid provenance")
    if agent["_path"].resolve() != (root / f"{agent['id']}.json").resolve():
        raise ContractError("canonical identity/source conflict")
    if not re.fullmatch(r"\d+\.\d+\.\d+", agent["version"]):
        raise ContractError("invalid version")


def metadata(agent: dict[str, Any], target: str, payload: str) -> dict[str, str]:
    clean = {k: v for k, v in agent.items() if k != "_path"}
    return {"canonical_agent_id": agent["id"], "canonical_source": agent["provenance"]["canonical_source"], "canonical_sha256": digest(clean), "canonical_version": agent["version"], "projection_target": target, "schema_version": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION, "payload_sha256": hashlib.sha256(payload.encode()).hexdigest()}


def envelope(agent: dict[str, Any], target: str, payload: str, comment: str) -> str:
    closing = " -->" if comment == "<!--" else ""
    return f"{comment} LV-002-PROVENANCE {json.dumps(metadata(agent, target, payload), sort_keys=True, separators=(',', ':'))}{closing}\n{payload}"


def render_agent(agent: dict[str, Any], target: str) -> tuple[str, str]:
    constraints = "\n".join(f"- {item}" for item in agent["constraints"])
    instructions = "\n".join(f"- {item}" for item in agent["instructions"])
    if target == "codex":
        payload = "\n".join([f'name = "{agent["id"]}"', f'description = "{agent["purpose"]}"', 'developer_instructions = """', "# " + agent["display_name"], "## Instructions", instructions, "## Constraints", constraints, '"""', ""])
        return f"agents/{agent['id']}.toml", envelope(agent, target, payload, "#")
    payload = f"# {agent['display_name']}\n\n## Purpose\n\n{agent['purpose']}\n\n## Instructions\n\n{instructions}\n\n## Constraints\n\n{constraints}\n"
    return f"agents/{agent['id']}.md", envelope(agent, target, payload, "<!--")


def expected_artifacts(agents: list[dict[str, Any]], target: str) -> dict[str, str]:
    if target not in TARGETS:
        raise ContractError(f"unsupported target: {target}")
    if target not in PROJECTABLE_TARGETS:
        raise ContractError("HUMAN_DECISION_REQUIRED: Qwen adapter format is unverified")
    active = [a for a in agents if target in a["runtime_targets"]]
    for agent in active:
        if agent["status"] != "ACTIVE":
            raise ContractError(f"refusing projection of non-active agent: {agent['id']}")
        if agent["skills"]:
            raise ContractError(f"skill projection is unsupported without canonical skill content: {agent['id']}")
    return dict(render_agent(agent, target) for agent in active)


def inspect(agents: list[dict[str, Any]], target: str, output: Path) -> dict[str, str]:
    expected = expected_artifacts(agents, target)
    result: dict[str, str] = {}
    for rel, wanted in expected.items():
        actual = output / rel
        if not actual.exists():
            result[rel] = "MISSING"
            continue
        text = actual.read_text(encoding="utf-8")
        if "LV-002-PROVENANCE " not in text:
            result[rel] = "UNMANAGED"
            continue
        if text == wanted:
            result[rel] = "IN_SYNC"
            continue
        line = text.splitlines()[0]
        try:
            raw = line.split("LV-002-PROVENANCE ", 1)[1].removesuffix(" -->")
            meta = json.loads(raw)
            payload = "\n".join(text.splitlines()[1:]) + ("\n" if text.endswith("\n") else "")
            good_payload = meta["payload_sha256"] == hashlib.sha256(payload.encode()).hexdigest()
            wanted_meta = json.loads(wanted.splitlines()[0].split("LV-002-PROVENANCE ", 1)[1].removesuffix(" -->"))
        except (IndexError, KeyError, json.JSONDecodeError):
            result[rel] = "INVALID"
            continue
        if not good_payload:
            result[rel] = "DRIFT"
        elif meta.get("canonical_sha256") != wanted_meta["canonical_sha256"]:
            result[rel] = "STALE"
        else:
            result[rel] = "DRIFT"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["validate", "inspect", "generate"])
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--target", choices=sorted(TARGETS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        agents = read_agents(args.canonical)
        if args.command == "validate":
            print("VALID")
            return 0
        if not args.target or not args.output:
            raise ContractError("--target and --output are required")
        if args.command == "inspect":
            report = inspect(agents, args.target, args.output)
            print(json.dumps(report, sort_keys=True))
            return 0 if all(v == "IN_SYNC" for v in report.values()) else 1
        expected = expected_artifacts(agents, args.target)
        current = inspect(agents, args.target, args.output)
        blockers = {p: s for p, s in current.items() if s in {"DRIFT", "UNMANAGED", "INVALID"}}
        if blockers:
            raise ContractError(f"refusing overwrite due to drift: {blockers}")
        if args.dry_run:
            print(json.dumps({"would_write": sorted(p for p, s in current.items() if s != "IN_SYNC")}, sort_keys=True))
            return 0
        for rel, content in expected.items():
            path = args.output / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        print(json.dumps({"written": sorted(expected)}, sort_keys=True))
        return 0
    except ContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
