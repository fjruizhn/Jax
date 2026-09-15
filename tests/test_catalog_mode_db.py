"""
MotorCatalog.from_db() trae capability.mode (tanda A v2). Corre en el job
jacobs-gobernanza-db, cuya base la crean las migraciones de jax-platform
(clonado de master): requiere PR-A (capability.mode) mergeado. Solo lee.
"""
from __future__ import annotations

import asyncio

from las_manos.motor_registry.catalog import MotorCatalog


def test_from_db_trae_el_modo_sembrado_de_cada_capability():
    modos = {c.name: c.mode for c in asyncio.run(MotorCatalog.from_db()).capabilities()}
    assert modos["file_write"] == "mutating"
    assert modos["generate"] == "read_only"
    assert {n for n, m in modos.items() if m == "mutating"} == {"file_write"}
