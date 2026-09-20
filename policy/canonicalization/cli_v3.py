"""C14N/3 read-only candidate hash command."""
from __future__ import annotations
import json, sys
from pathlib import Path
from .corpus_v3 import validate_candidate_corpus
from .errors import CanonicalizationError
def main(argv=None):
 argv=sys.argv[1:] if argv is None else argv
 if not argv or argv[0] != "candidate-hash" or len(argv)>2:
  print("usage: jax-policy-c14n-v3 candidate-hash [repo-root]",file=sys.stderr); return 64
 try: report=validate_candidate_corpus(Path(argv[1]) if len(argv)==2 else Path.cwd())
 except CanonicalizationError as exc:
  print(f"INVALID: {exc}",file=sys.stderr); return 2
 print(json.dumps(report,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
