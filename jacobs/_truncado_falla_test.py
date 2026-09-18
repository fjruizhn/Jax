"""Un paso truncado por tope de longitud falla, no entrega texto a medias.

Hoy los tres transportes HTTP directos (openai-compat, Ollama, Gemini) ni leen
el campo de corte del proveedor: el texto a medias sigue viaje como resultado
bueno y el paso siguiente construye sobre una frase cortada. El camino de
motor (las_manos/motor_registry/worker.py:828) ya falla el job cuando
finish_reason == "length" -- esto es lo mismo para los transportes directos.
"""
from __future__ import annotations

import pytest

from jacobs.executor import PasoTruncado, _texto_o_truncado


def test_openai_compat_truncado_falla():
    """Hoy el texto a medias sigue viaje y el paso siguiente construye sobre
    una frase cortada."""
    data = {"choices": [{"message": {"content": "a medi"}, "finish_reason": "length"}]}
    with pytest.raises(PasoTruncado):
        _texto_o_truncado(data, "openai_compat")


def test_openai_compat_completo_pasa():
    data = {"choices": [{"message": {"content": "entero"}, "finish_reason": "stop"}]}
    assert _texto_o_truncado(data, "openai_compat") == "entero"


def test_ollama_truncado_falla():
    data = {"message": {"content": "a medi"}, "done_reason": "length"}
    with pytest.raises(PasoTruncado):
        _texto_o_truncado(data, "ollama")


def test_ollama_completo_pasa():
    data = {"message": {"content": "entero"}, "done_reason": "stop"}
    assert _texto_o_truncado(data, "ollama") == "entero"


def test_gemini_truncado_falla():
    data = {"candidates": [{"finishReason": "MAX_TOKENS",
                            "content": {"parts": [{"text": "a medi"}]}}]}
    with pytest.raises(PasoTruncado):
        _texto_o_truncado(data, "gemini")


def test_gemini_completo_pasa():
    data = {"candidates": [{"finishReason": "STOP",
                            "content": {"parts": [{"text": "entero"}]}}]}
    assert _texto_o_truncado(data, "gemini") == "entero"
