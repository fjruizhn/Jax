"""E-21 (2026-09-16): el extractor y el sintetizador de la memoria toman la URL
de DeepSeek del catálogo (provider.base_url), no de un default escrito en
jax/muscles/base.py.

Antes: build_extractor() y build_synthesizer() armaban HttpMuscle sin api_url y
despachaban a la URL fija de base.py. Al quitar los defaults de proveedor, los
dos timers de producción (jax-memory-worker cada 20 min, jax-memory-synthesis
diario) habrían fallado en cada corrida. Sin fila o sin base_url en el
catálogo: MuscleInvocationError, sin URL de respaldo.

Puro: la lectura del catálogo se parchea (_db_conn de registro_facetas).
"""
from __future__ import annotations

import asyncio
import os

import pytest

os.environ["JAX_DB_NAME"] = "jax_memory_test"


class _Cursor:
    def __init__(self, fila, consultas):
        self._fila = fila
        self._consultas = consultas

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, sql, params):
        self._consultas.append((sql, params))

    async def fetchone(self):
        return self._fila


class _Conexion:
    def __init__(self, fila, consultas):
        self._fila = fila
        self._consultas = consultas
        self.cerrada = False

    def cursor(self):
        return _Cursor(self._fila, self._consultas)

    def close(self):
        self.cerrada = True


def _catalogo(monkeypatch, fila):
    from jax.core import registro_facetas
    consultas: list = []
    conexiones: list = []

    async def conectar():
        conexion = _Conexion(fila, consultas)
        conexiones.append(conexion)
        return conexion

    monkeypatch.setattr(registro_facetas, "_db_conn", conectar)
    return consultas, conexiones


@pytest.mark.parametrize("modulo, constructor", [
    ("jax.memory.worker", "build_extractor"),
    ("jax.memory.synthesis_worker", "build_synthesizer"),
])
def test_el_worker_de_memoria_toma_la_url_de_deepseek_del_catalogo(monkeypatch, modulo, constructor):
    import importlib
    consultas, conexiones = _catalogo(monkeypatch, ("https://deepseek.catalogo.test/",))
    construir = getattr(importlib.import_module(modulo), constructor)
    musculo = asyncio.run(construir())
    assert musculo.api_url == "https://deepseek.catalogo.test/chat/completions"
    assert musculo._url_del_catalogo() == "https://deepseek.catalogo.test/chat/completions"
    assert consultas == [("SELECT base_url FROM provider WHERE id = %s", ("deepseek",))]
    assert all(c.cerrada for c in conexiones)


@pytest.mark.parametrize("fila", [None, (None,), ("",)])
@pytest.mark.parametrize("modulo, constructor", [
    ("jax.memory.worker", "build_extractor"),
    ("jax.memory.synthesis_worker", "build_synthesizer"),
])
def test_sin_base_url_en_el_catalogo_el_worker_no_arranca(monkeypatch, modulo, constructor, fila):
    import importlib
    from jax.muscles.base import MuscleInvocationError
    _, conexiones = _catalogo(monkeypatch, fila)
    construir = getattr(importlib.import_module(modulo), constructor)
    with pytest.raises(MuscleInvocationError) as e:
        asyncio.run(construir())
    assert "sin URL del proveedor" in str(e.value)
    assert all(c.cerrada for c in conexiones)


def test_gemini_usa_la_base_sin_sufijo(monkeypatch):
    from jax.core.registro_facetas import url_del_proveedor
    consultas, _ = _catalogo(monkeypatch, ("https://gemini.catalogo.test/v1beta",))
    assert asyncio.run(url_del_proveedor("gemini")) == "https://gemini.catalogo.test/v1beta"
    assert consultas[0][1] == ("gemini",)


def test_clave_de_proveedor_desconocida_no_consulta(monkeypatch):
    from jax.core.registro_facetas import url_del_proveedor
    from jax.muscles.base import MuscleInvocationError
    consultas, _ = _catalogo(monkeypatch, ("https://x.test",))
    with pytest.raises(MuscleInvocationError):
        asyncio.run(url_del_proveedor("inexistente"))
    assert consultas == []
