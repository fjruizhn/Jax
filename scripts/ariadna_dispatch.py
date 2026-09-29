#!/usr/bin/env python3
"""Explicit operator entrypoint for the governed builder handoff consumer.

It creates a worktree and ACK only.  It never starts Qwen or another builder.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path


def load(root: Path):
    source = root / "projects/las-voces/authority/ariadna_dispatcher.py"
    spec = importlib.util.spec_from_file_location("ariadna_dispatcher_cli", source)
    if spec is None or spec.loader is None: raise RuntimeError("cannot load dispatcher")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module); return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="development checkout used for canonical validation")
    parser.add_argument("--canonical-root", type=Path, help="trusted producer checkout containing current canonical LAS VOCES state")
    parser.add_argument("--handoff-state-dir", type=Path, required=True, help="read-only Ariadna producer control directory")
    parser.add_argument("--worktree-root", type=Path, required=True, help="approved development worktree parent")
    parser.add_argument("--idempotency-key", required=True)
    parser.add_argument("--consume", action="store_true", help="required before a worktree may be created")
    args = parser.parse_args()
    if not args.consume:
        print(json.dumps({"decision": "NOOP", "reason": "--consume is required; no handoff was processed"}, sort_keys=True)); return 0
    result = load(args.root).GovernedHandoffConsumer(args.root, args.handoff_state_dir, args.worktree_root, canonical_root=args.canonical_root).consume(args.idempotency_key)
    print(json.dumps(asdict(result), sort_keys=True)); return 0 if result.decision in {"DISPATCHED", "NOOP"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
