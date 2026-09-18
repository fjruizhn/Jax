"""Configuración del pre-vuelo (spec 2026-09-17 §4.5, §4.6): se lee en cada
llamada y un valor inválido FALLA con el nombre de la variable, nunca cae a un
default silencioso (mismo trato que JAX_DB_CONNECT_TIMEOUT_SECONDS).

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import os
import pathlib

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import pytest  # noqa: E402

from jacobs import prevuelo_config as pc  # noqa: E402


def test_timeout_de_sonda_por_defecto(monkeypatch):
    monkeypatch.delenv(pc.SONDA_TIMEOUT_S, raising=False)
    assert pc.sonda_timeout_s() == 20


def test_chars_por_token_por_defecto_es_2(monkeypatch):
    monkeypatch.delenv(pc.CHARS_POR_TOKEN, raising=False)
    assert pc.chars_por_token() == 2


def test_max_tokens_de_sonda_por_defecto(monkeypatch):
    monkeypatch.delenv(pc.SONDA_MAX_TOKENS, raising=False)
    assert pc.sonda_max_tokens() == 16


def test_valor_valido_del_entorno_manda(monkeypatch):
    monkeypatch.setenv(pc.CHARS_POR_TOKEN, "3")
    assert pc.chars_por_token() == 3


def test_no_numerico_lanza_con_el_nombre(monkeypatch):
    monkeypatch.setenv(pc.SONDA_TIMEOUT_S, "veinte")
    with pytest.raises(RuntimeError, match="JAX_PREVUELO_SONDA_TIMEOUT_S"):
        pc.sonda_timeout_s()


def test_cero_lanza(monkeypatch):
    monkeypatch.setenv(pc.SONDA_MAX_TOKENS, "0")
    with pytest.raises(RuntimeError, match="mayor que cero"):
        pc.sonda_max_tokens()


def test_negativo_lanza(monkeypatch):
    monkeypatch.setenv(pc.CHARS_POR_TOKEN, "-2")
    with pytest.raises(RuntimeError, match="JAX_PREVUELO_CHARS_POR_TOKEN"):
        pc.chars_por_token()


# Task 15b (2026-09-17): tamaño del pool; R38 lo pasa al store entero
# (JAX_DB_POOL_MAX, default derivado en jacobs/store.py::db_pool_max).
def test_pool_max_por_defecto_es_10(monkeypatch):
    from jacobs import store

    monkeypatch.delenv(store.DB_POOL_MAX, raising=False)
    monkeypatch.setenv("JAX_PREVUELO_DB_POOL_MAX", "3")  # el nombre viejo no lo lee nadie
    assert store.db_pool_max() == 10


def test_pool_max_invalido_lanza_con_el_nombre(monkeypatch):
    from jacobs import store

    monkeypatch.setenv(store.DB_POOL_MAX, "cinco")
    with pytest.raises(RuntimeError, match="JAX_DB_POOL_MAX"):
        store.db_pool_max()


def test_la_variable_del_candado_se_retiro_con_el_candado():
    """2026-09-17: se fue el GET_LOCK del cupo, así que se fue su timeout. Una
    variable de entorno que no lee nadie es peor que ninguna: el día que
    alguien la ponga en /etc/jax/.env va a creer que cambió algo."""
    assert not hasattr(pc, "candado_timeout_s")
    assert not hasattr(pc, "CANDADO_TIMEOUT_S")
    fuente = pathlib.Path(pc.__file__).read_text(encoding="utf-8")
    assert "candado_timeout_s" not in fuente


