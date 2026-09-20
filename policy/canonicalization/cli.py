"""Inspección read-only del mecanismo shadow; nunca decide GO/NO-GO."""

from __future__ import annotations

import json
from pathlib import Path

from .bootstrap import verified_bootstrap
from .corpus import hash_policy_corpus, load_shadow_manifest
from .legacy import compare_legacy_identities


def _shadow_report(repo_root: Path | str) -> dict:
    root = Path(repo_root)
    bootstrap = verified_bootstrap()
    manifest = load_shadow_manifest(root)
    computed = hash_policy_corpus(root)
    legacy = compare_legacy_identities(root / "policy")
    return {
        "mode": "SHADOW",
        "authorizes": False,
        "blocks_production": False,
        "replaces_legacy_identity": False,
        "writes_policy": False,
        "bootstrap_bundle_id": bootstrap.bundle_id,
        "shadow_policy_corpus_hash": computed,
        "manifest_hash_matches": manifest["shadow_policy_corpus_hash"] == computed,
        "legacy_identities": legacy.identities,
        "legacy_divergent": legacy.divergent,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(_shadow_report(root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
