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
    parser.add_argument("--request-jaxqwen", action="store_true",
                        help="explicitly request the fixed jaxqwen capability after a new DISPATCHED ACK")
    parser.add_argument("--jaxqwen-socket", type=Path, help="provisioned local jaxqwen dispatcher socket")
    parser.add_argument("--jaxqwen-service-uid", type=int, help="numeric UID of the dedicated jaxqwen host")
    args = parser.parse_args()
    if args.request_jaxqwen != (args.jaxqwen_socket is not None and args.jaxqwen_service_uid is not None):
        parser.error("--request-jaxqwen requires both --jaxqwen-socket and --jaxqwen-service-uid")
    if not args.request_jaxqwen and (args.jaxqwen_socket is not None or args.jaxqwen_service_uid is not None):
        parser.error("jaxqwen transport options require --request-jaxqwen")
    if not args.consume:
        print(json.dumps({"decision": "NOOP", "reason": "--consume is required; no handoff was processed"}, sort_keys=True)); return 0
    module = load(args.root)
    consumer = module.GovernedHandoffConsumer(args.root, args.handoff_state_dir, args.worktree_root,
                                               canonical_root=args.canonical_root,
                                               jaxqwen_socket=args.jaxqwen_socket if args.request_jaxqwen else None,
                                               jaxqwen_service_uid=args.jaxqwen_service_uid if args.request_jaxqwen else None)
    result = consumer.consume(args.idempotency_key)
    output = asdict(result)
    if args.request_jaxqwen and result.decision in {"DISPATCHED", "NOOP"}:
        try: output["jaxqwen"] = consumer.dispatch_jaxqwen(result)
        except (OSError, ValueError) as exc:
            output["jaxqwen"] = {"decision": "HUMAN_REQUIRED", "reason": type(exc).__name__}
    print(json.dumps(output, sort_keys=True))
    qwen_decision = output.get("jaxqwen", {}).get("decision") if isinstance(output.get("jaxqwen"), dict) else None
    return 0 if result.decision in {"DISPATCHED", "NOOP"} and qwen_decision not in {"REJECTED", "HUMAN_REQUIRED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
