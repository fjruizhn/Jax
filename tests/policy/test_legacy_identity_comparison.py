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
    # FOTO ACTUALIZADA el 2026-09-23 (decisión de Fernando, jax#270). Antes:
    # reglas c30424f9…, corpus embebido d641add3…. Cambiaron porque P04 citaba
    # `las_manos/policy.py` (borrado en jax#262) y MA01-03 citaban por ruta
    # absoluta `missions/bridge-migration.md` (sacado de HEAD en B1.4), y porque
    # CORPUS.md estaba desfasado de sus reglas desde el 2026-08-25. Al
    # regenerarlo, reglas y corpus embebido pasan a COINCIDIR (606e6687…): la
    # divergencia que queda, y que esta foto sigue documentando, es la de
    # VERSION (040f32…), que no se toca.
    assert comparison.identities == {
        "version_declared_rules_sha256":
            "040f32ea14ae2d21bcb53df3d03a7897151e745c388532fd3f603c867c963d60",
        "computed_raw_rules_sha256":
            "606e668774781ab831b17e36dc0300c89e7c79c02376eb73baf4bf1db2105d09",
        "generated_corpus_embedded_sha256":
            "606e668774781ab831b17e36dc0300c89e7c79c02376eb73baf4bf1db2105d09",
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
