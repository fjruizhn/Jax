"""
Hallazgo de revisión, Tarea 2b (tanda A, 2026-09-14, PR-B): `MotorCatalog.
from_db()` abría `aiomysql.connect(...)` sin `connect_timeout` -- medido
contra el venv de jax-platform (aiomysql 0.3.2): el default es
`connect_timeout=None`, y aiomysql lo pasa tal cual a
`asyncio.wait_for(timeout=None)` al abrir el socket (aiomysql/connection.py)
-- "sin límite". Si la DB se cuelga en vez de rechazar, `from_db()` (y con
ella LAS MANOS / jax-platform, que recargan el catálogo por acá) esperan
para siempre.

Test puro: se mockea `aiomysql.connect` con un AsyncMock que registra sus
kwargs y lanza para cortar la ejecución ahí mismo -- nada toca la DB.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from las_manos.motor_registry import catalog as catalog_mod
from las_manos.motor_registry.catalog import MotorCatalog


class _CorteDeliberado(Exception):
    """Señal para cortar from_db() justo después de aiomysql.connect()."""


def _mock_connect(monkeypatch) -> AsyncMock:
    mock = AsyncMock(side_effect=_CorteDeliberado())
    monkeypatch.setattr(catalog_mod.aiomysql, "connect", mock)
    return mock


def _env_db_basico(monkeypatch):
    monkeypatch.setenv("JAX_DB_HOST", "127.0.0.1")
    monkeypatch.setenv("JAX_DB_PORT", "3308")


def test_from_db_pasa_connect_timeout_default(monkeypatch):
    _env_db_basico(monkeypatch)
    monkeypatch.delenv("JAX_DB_CONNECT_TIMEOUT_SECONDS", raising=False)
    mock = _mock_connect(monkeypatch)

    with pytest.raises(_CorteDeliberado):
        asyncio.run(MotorCatalog.from_db())

    assert mock.await_args.kwargs["connect_timeout"] == 10


def test_from_db_pasa_connect_timeout_de_la_variable(monkeypatch):
    _env_db_basico(monkeypatch)
    monkeypatch.setenv("JAX_DB_CONNECT_TIMEOUT_SECONDS", "3")
    mock = _mock_connect(monkeypatch)

    with pytest.raises(_CorteDeliberado):
        asyncio.run(MotorCatalog.from_db())

    assert mock.await_args.kwargs["connect_timeout"] == 3


@pytest.mark.parametrize("valor_invalido", ["0", "-5", "no-es-numero", ""])
def test_from_db_rechaza_connect_timeout_invalido(monkeypatch, valor_invalido):
    _env_db_basico(monkeypatch)
    monkeypatch.setenv("JAX_DB_CONNECT_TIMEOUT_SECONDS", valor_invalido)
    mock = _mock_connect(monkeypatch)

    with pytest.raises(RuntimeError, match="JAX_DB_CONNECT_TIMEOUT_SECONDS"):
        asyncio.run(MotorCatalog.from_db())

    mock.assert_not_awaited()
