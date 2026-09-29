#!/usr/bin/env python3
"""Fixed systemd entrypoint for the dedicated LAS VOCES jaxqwen broker."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "projects/las-voces/authority"
if str(AUTHORITY) not in sys.path: sys.path.insert(0, str(AUTHORITY))
from jaxqwen_host import JaxQwenHost

_FIELDS = {"root", "handoff_state_dir", "source_worktree_root", "workspace_root", "mission_state_dir", "trust_state_dir",
           "dispatch_socket", "dispatcher_uid", "dispatch_gid", "model_socket", "model",
           "model_socket_uid", "model_socket_gid",
           "max_output_tokens", "dispatch_enabled"}


def _config(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("unreadable jaxqwen host config") from exc
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError("jaxqwen host config schema mismatch")
    paths = ("root", "handoff_state_dir", "source_worktree_root", "workspace_root", "mission_state_dir", "trust_state_dir", "dispatch_socket", "model_socket")
    for name in paths:
        if not isinstance(value[name], str) or not Path(value[name]).is_absolute(): raise ValueError("jaxqwen host paths must be absolute")
        value[name] = Path(value[name])
    if (type(value["dispatcher_uid"]) is not int or value["dispatcher_uid"] < 1
            or type(value["dispatch_gid"]) is not int or value["dispatch_gid"] < 1
            or type(value["model_socket_uid"]) is not int or value["model_socket_uid"] < 1
            or type(value["model_socket_gid"]) is not int or value["model_socket_gid"] < 1
            or type(value["max_output_tokens"]) is not int or not 1 <= value["max_output_tokens"] <= 32768
            or type(value["dispatch_enabled"]) is not bool
            or not isinstance(value["model"], str) or not value["model"]):
        raise ValueError("invalid jaxqwen host config value")
    return value


async def _run(config: Path):
    value = _config(config)
    credentials_dir = os.environ.get("CREDENTIALS_DIRECTORY")
    if not credentials_dir or not Path(credentials_dir).is_absolute(): raise RuntimeError("systemd trust credential is unavailable")
    host = JaxQwenHost(**value, trust_key_file=Path(credentials_dir) / "jaxqwen-trust.key")
    path = await host.start()
    logging.getLogger(__name__).info("jaxqwen host ready socket=%s dispatch_enabled=%s", path, value["dispatch_enabled"])
    try:
        await asyncio.Future()
    finally:
        await host.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try: asyncio.run(_run(args.config))
    except (OSError, ValueError, RuntimeError) as exc:
        logging.getLogger(__name__).error("jaxqwen host refused startup (%s)", type(exc).__name__)
        return 2
    return 0


if __name__ == "__main__": raise SystemExit(main())
