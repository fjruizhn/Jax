"""
policy/governance — Barrido léxico contra el vocabulario cerrado
(REFORMAS-v3.md §3.1.5).

Puro: recibe el vocabulario ya cargado y su mapeo de categorías
(loaders.load_vocabulary().term_categories), nunca abre un archivo.
Decidir qué hacer con los términos encontrados es responsabilidad del
llamador — sub-proyecto 2.
"""
from __future__ import annotations

import re


def sweep(
    text: str, term_categories: dict[str, frozenset[str]]
) -> list[tuple[str, frozenset[str]]]:
    if not term_categories:
        return []
    terms = sorted((t for t in term_categories if t), key=len, reverse=True)
    if not terms:
        return []
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(t) for t in terms) + r")(?!\w)",
        re.IGNORECASE,
    )
    found = {m.group(0).lower() for m in pattern.finditer(text)}
    lowered_to_original = {t.lower(): t for t in term_categories if t}
    matched = [lowered_to_original[f] for f in found if f in lowered_to_original]
    return sorted((t, term_categories[t]) for t in matched)
