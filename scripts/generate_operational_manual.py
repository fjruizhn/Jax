#!/usr/bin/env python3
"""Deterministically render the non-authoritative printable operational manual."""
from __future__ import annotations
import argparse, hashlib, html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/operations/operational-manual.md'; OUTPUT=ROOT/'docs/operations/generated/jax-operational-docs.html'
def identity(path):
    return "sha256:"+hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--revision", help="optional presentation metadata; not used for reproducibility")
    args=parser.parse_args()
    text=html.escape(SOURCE.read_text(encoding='utf-8'))
    body=''.join(f'<p>{line}</p>' for line in text.splitlines() if line.strip())
    rendered=f'''<!doctype html><html><head><meta charset="utf-8"><style>@page {{ size: Letter; margin: 0.6in; }} body {{ font-family: serif; }} .notice {{ border: 2px solid #900; padding: .4em; font-weight:bold; }}</style></head><body><p class="notice">NON-AUTHORITATIVE GENERATED PRESENTATION</p><p>Source: docs/operations/operational-manual.md<br>Source revision: {identity(SOURCE)}<br>Generator: scripts/generate_operational_manual.py<br>Generator revision: {identity(Path(__file__))}</p><p>Query jaxctl and docs/operations/query-guide.md for current runtime state.</p>{body}</body></html>'''
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != rendered:
            raise SystemExit("generated operational manual is stale")
        return
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(rendered,encoding='utf-8')
if __name__=='__main__': main()
