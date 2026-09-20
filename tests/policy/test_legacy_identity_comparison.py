from __future__ import annotations

from pathlib import Path
from hashlib import sha256

from policy.canonicalization.legacy import compare_legacy_identities


ROOT = Path(__file__).resolve().parents[2]


def test_compara_las_tres_identidades_legacy_sin_corregirlas() -> None:
    version_before = (ROOT / "policy/VERSION").read_bytes()
    generated_before = (ROOT / "policy/generated/CORPUS.md").read_bytes()

    comparison = compare_legacy_identities(ROOT / "policy")

    assert set(comparison.identities) == {
        "version_declared_rules_sha256",
        "computed_raw_rules_sha256",
        "generated_corpus_embedded_sha256",
    }
    assert comparison.divergent is True
    assert comparison.identities == {
        "version_declared_rules_sha256":
            "040f32ea14ae2d21bcb53df3d03a7897151e745c388532fd3f603c867c963d60",
        "computed_raw_rules_sha256":
            "c30424f98e78aa514144d22d48eb6f1203f8e0328fcbd88e356ef7ac5927af46",
        "generated_corpus_embedded_sha256":
            "d641add34e2f4e616d6840fc80c17be705919521d0538200a2a49ceab01ed031",
    }

    ordered = sorted(
        (path.stem.split("-", 1)[0], path)
        for path in (ROOT / "policy/rules").glob("*.yaml")
    )
    digest = sha256()
    for _, path in ordered:
        digest.update(path.read_bytes())
    assert digest.hexdigest() == comparison.identities["computed_raw_rules_sha256"]
    assert (ROOT / "policy/VERSION").read_bytes() == version_before
    assert (ROOT / "policy/generated/CORPUS.md").read_bytes() == generated_before
