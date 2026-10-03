"""Local, isolated Qwen Code 0.24.7 smoke for LV-001B.

Uses a temporary MCP config with a Faro package built from a toy Git repo.
Nothing connects to a Faro service or uses production credentials.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects/las-voces"
EXPECTED = ("skills_buscar", "skills_leer", "agentes_listar")
BARE_BUILTINS = ("read_file", "notebook_edit", "run_shell_command", "get_goal", "update_goal", "edit")


def _local_url(raw: str) -> str:
    url = urlsplit(raw)
    if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.username or url.password or url.query or url.fragment):
        raise argparse.ArgumentTypeError("--base-url must be a local HTTP endpoint without credentials")
    return raw


def _tool_events(lines: str) -> tuple[list[tuple[str, str]], dict[str, str], dict]:
    calls: list[tuple[str, str]] = []
    results: dict[str, str] = {}
    init: dict = {}
    for line in lines.splitlines():
        item = json.loads(line)
        if item.get("type") == "system" and item.get("subtype") == "init":
            init = item
        for part in item.get("message", {}).get("content", []):
            if part.get("type") == "tool_use":
                name = part.get("name", "")
                if name == "tool_search":
                    continue
                if name == "tool_call":
                    name = part.get("input", {}).get("name", "")
                calls.append((part.get("id", ""), name))
            elif part.get("type") == "tool_result":
                if part.get("is_error"):
                    raise RuntimeError(f"Qwen tool error: {str(part.get('content'))[:200]}")
                results[part.get("tool_use_id", "")] = str(part.get("content", ""))
    return calls, results, init


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True, type=_local_url)
    ap.add_argument("--qwen", default="qwen")
    args = ap.parse_args()
    version = subprocess.run([args.qwen, "--version"], check=True, capture_output=True, text=True).stdout.strip()
    if version != "0.24.7":
        raise RuntimeError(f"Qwen Code 0.24.7 required, got {version}")
    qwen_path = args.qwen if "/" in args.qwen else shutil.which(args.qwen)
    if not qwen_path:
        raise RuntimeError("qwen executable not found")
    bundle = Path(qwen_path).resolve()
    bundle = bundle.parent
    context = subprocess.run(["node", str(ROOT / "tests/las_voces_qwen_context.mjs")], cwd=ROOT,
                             env={**os.environ, "QWEN_BUNDLE_DIR": str(bundle),
                                  "LV001B_QWEN_MODEL": args.model},
                             check=True, capture_output=True, text=True)
    context_record = json.loads(context.stdout.strip())
    with tempfile.TemporaryDirectory(prefix="lv001b-qwen-") as temp:
        config = Path(temp) / "mcp.json"
        config.write_text(json.dumps({"mcpServers": {"faro-readonly": {
            "command": sys.executable,
            "args": [str(ROOT / "tests/_las_voces_readonly_fixture.py")],
            "cwd": str(ROOT),
            "includeTools": ["skills.buscar", "skills.leer", "agentes.listar"],
            "trust": True,  # only the fixed toy package in this temporary test config
        }}}))
        prompt = ("Usa exactamente las tres herramientas MCP de faro-readonly, en este orden: "
                  "skills.buscar con consulta mide, skills.leer con nombre alfa, agentes.listar. "
                  "Da los nombres obtenidos. No ejecutes herramientas fuera de ese servidor.")
        run = subprocess.run([
            args.qwen, "--bare", "--mcp-config", str(config),
            "--allowed-mcp-server-names", "faro-readonly",
            "--auth-type", "openai", "--openai-base-url", args.base_url,
            "--openai-api-key", "ollama", "--model", args.model, "--approval-mode", "auto",
            *(part for name in BARE_BUILTINS for part in ("--exclude-tools", name)),
            "--output-format", "stream-json", "--max-wall-time", "120s", "--max-tool-calls", "3", prompt,
        ], cwd=PROJECT, capture_output=True, text=True, timeout=135)
    if run.returncode:
        raise RuntimeError(f"Qwen smoke rc={run.returncode}: {run.stderr[-600:]}")
    calls, results, init = _tool_events(run.stdout)
    if not any(x.get("name") == "faro-readonly" and x.get("status") == "connected"
               for x in init.get("mcp_servers", [])):
        raise RuntimeError("Qwen did not connect to faro-readonly")
    exposed = init.get("tools", [])
    if len(exposed) != 3 or any(not re.fullmatch(
            r"mcp__faro-readonly__(skills_buscar|skills_leer|agentes_listar)_[a-z0-9]+", name)
            for name in exposed):
        raise RuntimeError(f"smoke session exposes non-Faro tools: {exposed}")
    observed = []
    for call_id, name in calls:
        match = re.fullmatch(r"mcp__faro-readonly__(skills_buscar|skills_leer|agentes_listar)_[a-z0-9]+", name)
        if not match or call_id not in results:
            raise RuntimeError(f"unexpected or unanswered tool call: {name}")
        observed.append(match.group(1))
    if tuple(observed) != EXPECTED:
        raise RuntimeError(f"expected exactly {EXPECTED}, got {observed}")
    for (call_id, _), marker in zip(calls, ("alfa", "cuerpo alfa", "explorador"), strict=True):
        if marker not in results[call_id]:
            raise RuntimeError(f"tool result missing expected marker {marker}")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout.strip()
    print(json.dumps({"task": "LV-001", "sha": sha, "qwen": version, "model": args.model,
                      "projectEffectiveContextWindowSize": context_record["effectiveContextWindowSize"],
                      "smokeMode": "bare with only Faro read tools exposed",
                      "mcpServer": "faro-readonly", "calls": observed, "results": "success",
                      "package": "toy git repo built and verified by Faro"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
