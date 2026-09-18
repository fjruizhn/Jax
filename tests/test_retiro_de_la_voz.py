"""Retiro de la voz (2026-09-17): Kokoro TTS y Whisper quedan retirados.

Decisión de Fernando. Motivo medido el mismo día: no existía el venv
(`~/kokoro-test`), no había modelos en `/srv/jax-data/`, `import kokoro` daba
ModuleNotFoundError y `JAX_KOKORO_PYTHON` no estaba en `/etc/jax/.env` — pero
`jax.core.main.main()` llamaba a `_python_de_kokoro()` al arrancar, así que la
puerta de una funcionalidad inexistente impedía arrancar el REPL entero.

Estos tests fallan contra el árbol anterior al retiro. Ver DEUDA.md §
"Retiro de la voz" para cómo traerla de vuelta.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

# Base de tests de ESTA sesión: respeta JAX_TEST_DB_SUFIJO y, si ya venía
# una base que NO es de tests, corta en vez de seguir. barrera: acá no se abre conexión
from base_de_test import exigir_base_de_test  # noqa: E402

exigir_base_de_test()

RAIZ = Path(__file__).resolve().parents[1]


def test_el_paquete_de_la_voz_ya_no_esta_en_el_arbol():
    assert not (RAIZ / "jax" / "voice").exists()


def test_el_arranque_del_repl_no_exige_la_variable_de_la_voz():
    """La puerta era `_python_de_kokoro()` dentro de `main()`. `url_requerida
    ("JAX_OLLAMA_URL")` se queda: esa variable sí está viva."""
    fuente = (RAIZ / "jax" / "core" / "main.py").read_text(encoding="utf-8")
    assert "JAX_KOKORO_PYTHON" not in fuente
    assert "_python_de_kokoro" not in fuente
    assert 'url_requerida("JAX_OLLAMA_URL")' in fuente


def test_ningun_modulo_de_servicio_importa_la_voz():
    for arbol in ("jax", "jacobs", "las_manos", "policy"):
        for ruta in (RAIZ / arbol).rglob("*.py"):
            if "__pycache__" in ruta.parts:
                continue
            nodo_raiz = ast.parse(ruta.read_text(encoding="utf-8"))
            for nodo in ast.walk(nodo_raiz):
                modulos: list[str] = []
                if isinstance(nodo, ast.Import):
                    modulos = [a.name for a in nodo.names]
                elif isinstance(nodo, ast.ImportFrom):
                    modulos = [nodo.module or ""]
                for modulo in modulos:
                    assert not modulo.startswith("jax.voice"), f"{ruta}: {modulo}"
                    assert modulo.split(".")[0] not in ("kokoro", "faster_whisper", "soundfile"), \
                        f"{ruta}: {modulo}"
