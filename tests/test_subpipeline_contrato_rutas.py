"""Contrato de sub-pipelines de punta a punta por la ruta real (frente F, 2026-09-16).

El arnés (`jacobs/_arnes_ada.py`) hace de Ada: arma un padre `running` con un
paso de Ada, emite tokens con la función de servidor y pide hijos por
`routes.create_pipeline`. Token, base y eventos son los reales.

ROJO CONTRA MASTER (984ce46): `test_ada_con_token_inventado_no_crea_pipeline`
falló con "DID NOT RAISE": `validate_create` aceptaba cualquier string no vacío
como token y creaba el pipeline.

Corre con:
  bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -v tests/test_subpipeline_contrato_rutas.py"
"""
from __future__ import annotations

import os

from jacobs import _arnes_ada as ada  # primero: barrera de base de prueba

import asyncio  # noqa: E402

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.getenv("JAX_DB_HOST"),
    reason="necesita la MariaDB real con JAX_DB_NAME=jax_memory_test",
)

TOKEN_INVENTADO = "token-inventado-por-el-atacante"


def test_ada_con_token_inventado_no_crea_pipeline():
    async def escenario():
        padre, _paso = await ada.padre_en_ejecucion()
        try:
            with pytest.raises(HTTPException) as rechazo:
                await ada.pedir_hijo(TOKEN_INVENTADO, padre)
            return rechazo.value
        finally:
            await ada.cerrar(padre)

    rechazo = asyncio.run(escenario())
    assert rechazo.status_code == 403
    assert "token_desconocido" in rechazo.detail
    assert TOKEN_INVENTADO not in rechazo.detail
