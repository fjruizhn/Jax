"""El freno que ESCRIBE jax-platform es el que LEE LAS MANOS (plan
2026-09-16-frente-b-kill-switch, Task 3): el peor caso del motor en vuelo con
el escritor real de la plataforma, cargado desde su checkout. En CI lo corre
`mirror-sync` con JAX_PLATFORM_REPO_ROOT y piso de 0 skipped.

Corre con:
  JAX_PLATFORM_REPO_ROOT=<checkout> PYTHONPATH=.:las_manos python -m pytest tests/test_interruptor_escritor_de_la_plataforma.py -v
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path

import pytest

from motor_registry import worker
from motor_registry.models import JobStatus
from tests.test_interruptor_lectores import _job, _transporte_lento

RAIZ = Path(__file__).resolve().parents[1]
RAIZ_PLATAFORMA = Path(os.environ.get("JAX_PLATFORM_REPO_ROOT", Path.home() / "jax-platform"))
COPIA = RAIZ_PLATAFORMA / "backend" / "interruptor.py"

pytestmark = pytest.mark.skipif(
    not COPIA.is_file(),
    reason=f"sin {COPIA} (seteá JAX_PLATFORM_REPO_ROOT); en CI lo cubre el job mirror-sync")


def _escritor_de_la_plataforma():
    spec = importlib.util.spec_from_file_location("interruptor_de_la_plataforma", COPIA)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_la_copia_es_identica_al_canonico():
    sys.path.insert(0, str(RAIZ / "scripts"))
    from check_mirror_sync import FAMILIAS, revisar  # noqa: E402

    familia = next(f for f in FAMILIAS if f.nombre == "interruptor")
    drift, _declaradas, faltantes = revisar(familia)
    assert (drift, faltantes) == ([], [])


def test_el_freno_de_la_plataforma_mata_el_motor_en_vuelo(tmp_path, monkeypatch):
    plataforma = _escritor_de_la_plataforma()
    ruta = tmp_path / "PAUSE"
    monkeypatch.setattr(worker, "_KILL_SWITCH_INTERVAL", 0.05)
    cancelado = []

    async def escenario():
        en_vuelo = asyncio.Event()

        async def durante(tarea):
            await asyncio.wait_for(en_vuelo.wait(), 2.0)
            assert plataforma.escribir_pausa(ruta, '{"user_id": "plataforma"}') is True

        return await _job(tmp_path, ruta, _transporte_lento(en_vuelo, cancelado), durante)

    estado = asyncio.run(escenario())
    assert estado["status"] == JobStatus.FAILED.value
    assert "killed_by_switch" in estado["error"]
    assert cancelado == [True]
