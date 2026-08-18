"""
policy/governance — Barrido léxico contra el vocabulario cerrado
(REFORMAS-v3.md §3.1.5).

Puro: recibe el vocabulario ya cargado (loaders.load_vocabulary().flattened),
nunca abre un archivo. Decidir qué hacer con los términos encontrados
(reformular como claim, rechazar el bloque) es responsabilidad del
llamador — sub-proyecto 2, fuera de alcance acá.
"""
from __future__ import annotations

import re


def sweep(text: str, vocabulary: frozenset[str]) -> list[str]:
    if not vocabulary:
        return []
    terms = sorted((t for t in vocabulary if t), key=len, reverse=True)
    if not terms:
        return []
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(t) for t in terms) + r")(?!\w)",
        re.IGNORECASE,
    )
    found = {m.group(0).lower() for m in pattern.finditer(text)}
    # Map back to the original-cased vocabulary entries whose lowercase form matched.
    lowered_to_original = {t.lower(): t for t in vocabulary if t}
    return sorted(lowered_to_original[f] for f in found if f in lowered_to_original)
