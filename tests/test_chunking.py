#!/usr/bin/env python3
"""
Test de regresion — chunking de conversaciones largas (item #5).

Suite pytest. Verifica _chunk_conversation() (jax.memory.worker) SIN tocar
DB, red, ni el extractor — es una funcion pura sobre un string.

Uso: python -m pytest tests/test_chunking.py -v
"""
from __future__ import annotations

from jax.memory.worker import _chunk_conversation, MAX_CHARS_PER_EXTRACTION


def test_short_conversation_is_one_chunk():
    conv = "user: hola\njax_local: hola Fernando"
    assert _chunk_conversation(conv) == [conv]


def test_long_conversation_splits_into_multiple_chunks():
    lineas = [f"user: mensaje numero {i} con algo de relleno de texto" for i in range(500)]
    conv = "\n".join(lineas)
    assert len(conv) > MAX_CHARS_PER_EXTRACTION

    chunks = _chunk_conversation(conv)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= MAX_CHARS_PER_EXTRACTION


def test_chunks_reconstruct_original_lines():
    lineas = [f"user: mensaje numero {i} con algo de relleno de texto" for i in range(500)]
    conv = "\n".join(lineas)
    chunks = _chunk_conversation(conv)
    reconstruido = "\n".join(chunks)
    assert reconstruido == conv
