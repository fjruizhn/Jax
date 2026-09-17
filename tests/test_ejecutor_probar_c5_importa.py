# tests/test_ejecutor_probar_c5_importa.py
"""Regresión: el frente F retiró `jacobs.store.get_conn` y `scripts/ejecutor_contratos/probar_c5.py`
lo seguía importando: la prueba real de C5 reventaba con ImportError antes de leer nada (07:14
del 2026-09-17 hubo que correrla desde una copia). Sin DB ni config: el módulo importa y
`principal()` pide la conexión por `jacobs.store.conexion` (la misma que usa el exportador)."""
import argparse
import asyncio
import contextlib
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ejecutor_contratos" / "probar_c5.py"


def _cargar():
    spec = importlib.util.spec_from_file_location("probar_c5_regresion", _SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_el_script_importa_sin_config():
    import jacobs.store

    modulo = _cargar()
    assert modulo.conexion is jacobs.store.conexion


class _Alcanzada(Exception):
    pass


def test_principal_lee_la_config_por_la_conexion_del_pool(monkeypatch):
    modulo = _cargar()
    usadas = []

    @contextlib.asynccontextmanager
    async def conexion_falsa(desechable=False):
        conn = object()
        usadas.append((conn, desechable))
        yield conn

    async def leer_config(conn):
        assert conn is usadas[-1][0]
        raise _Alcanzada

    monkeypatch.setattr(modulo, "conexion", conexion_falsa)
    monkeypatch.setattr(modulo.eleccion_c5, "leer_config", leer_config)
    args = argparse.Namespace(corridas=0, cerebro=None, auditor=None, instrucciones=None, url_auditor=None)
    with pytest.raises(_Alcanzada):
        asyncio.run(modulo.principal(args))
    assert [d for _, d in usadas] == [True]
