#!/usr/bin/env python3
"""Repository entrypoint for the governed Ariadna host; never installs a service."""
from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import sys
from pathlib import Path


def _load(source: Path):
    spec = importlib.util.spec_from_file_location("ariadna_host_entrypoint", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--health", action="store_true")
    parser.add_argument("--run-forever", action="store_true")
    args = parser.parse_args(argv)
    host_module = _load(args.root / "projects/las-voces/authority/ariadna_host.py")
    host = host_module.AriadnaHost(args.root, host_module.HostConfig.from_file(args.config))
    if args.health:
        state = host._health_path
        print(state.read_text(encoding="utf-8") if state.is_file() else json.dumps(host.health(), sort_keys=True)); return 0
    if args.run_forever:
        signal.signal(signal.SIGTERM, lambda _signum, _frame: host.request_stop())
        signal.signal(signal.SIGINT, lambda _signum, _frame: host.request_stop())
        host.run_forever(); return 0
    parser.error("select --health or --run-forever")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
