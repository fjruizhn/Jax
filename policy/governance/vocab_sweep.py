"""
policy/governance — Barrido léxico contra el vocabulario cerrado
(REFORMAS-v3.md §3.1.5).

Puro: recibe el vocabulario ya cargado (loaders.load_vocabulary().flattened),
nunca abre un archivo. Decidir qué hacer con los términos encontrados
(reformular como claim, rechazar el bloque) es responsabilidad del
llamador — sub-proyecto 2, fuera de alcance acá.
"""
from __future__ import annotations


def sweep(text: str, vocabulary: frozenset[str]) -> list[str]:
    return sorted(term for term in vocabulary if term and term in text)
