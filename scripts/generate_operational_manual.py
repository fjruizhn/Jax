#!/usr/bin/env python3
"""Deterministically render the non-authoritative printable operational manual."""
from __future__ import annotations
import argparse, html, re, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/operations/operational-manual.md'; OUTPUT=ROOT/'docs/operations/generated/jax-operational-docs.html'
def revision():
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--revision")
    args=parser.parse_args()
    if args.revision is None:
        if args.check and OUTPUT.is_file():
            match=re.search(r"Revision: ([^<]+)", OUTPUT.read_text(encoding="utf-8"))
            if not match: raise SystemExit("generated manual lacks declared revision")
            args.revision=match.group(1)
        else:
            args.revision=revision()
    text=html.escape(SOURCE.read_text(encoding='utf-8'))
    body=''.join(f'<p>{line}</p>' for line in text.splitlines() if line.strip())
    rendered=f'''<!doctype html><html><head><meta charset="utf-8"><style>@page {{ size: Letter; margin: 0.6in; }} body {{ font-family: serif; }} .notice {{ border: 2px solid #900; padding: .4em; font-weight:bold; }}</style></head><body><p class="notice">NON-AUTHORITATIVE GENERATED PRESENTATION</p><p>Source: docs/operations/operational-manual.md<br>Generator: scripts/generate_operational_manual.py<br>Revision: {args.revision}</p><p>Query jaxctl and docs/operations/query-guide.md for current runtime state.</p>{body}</body></html>'''
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != rendered:
            raise SystemExit("generated operational manual is stale")
        return
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(rendered,encoding='utf-8')
if __name__=='__main__': main()
