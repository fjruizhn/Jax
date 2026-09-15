"""
PR-K (2026-09-14): `limite_de_salida()` contra una MariaDB real, con el esquema
y las semillas de las migraciones de jax-platform (job jacobs-gobernanza-db).
SOLO LEE: no inserta ni borra filas.

Filas usadas, las dos las crea la migración en una base vacía
(_seed_models_and_backfill deriva el catálogo de los bindings sembrados):
  - deepseek / deepseek-v4-flash: sembrada con ('max_tokens', 131072).
  - zhipu / glm-5.2: sin contrato (NULL/NULL) -- la semilla solo
    cubre glm-5.3. Es justamente el caso de fallo cerrado.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os

import pytest

_db = os.environ.get("JAX_DB_NAME", "")
if not _db.endswith("_test"):
    raise RuntimeError(f"JAX_DB_NAME={_db!r}: este test solo corre contra una base *_test.")

import contrato_dispatch as cd  # noqa: E402  (camino de Jacobs: symlink en las_manos/)
from facet_resolver import _db_conn  # noqa: E402


def test_lee_nombre_y_tope_de_la_fila_real():
    assert asyncio.run(cd.limite_de_salida("http_openai_compat", "deepseek", "deepseek-v4-flash")) == {"max_tokens": 131072}


def test_una_fila_sin_contrato_falla_con_los_dos_updates():
    # glm-5.2: la crea la migración desde el binding sembrado, sin contrato
    # (en producción, id 7, también NULL/NULL -- medido por el controller).
    with pytest.raises(cd.ModelDispatchConfigError) as exc:
        asyncio.run(cd.limite_de_salida("http_openai_compat", "zhipu", "glm-5.2"))
    assert "UPDATE model SET max_tokens_param" in str(exc.value)
    assert "UPDATE model SET max_output_tokens" in str(exc.value)


def test_fila_inexistente_falla():
    with pytest.raises(cd.ModelDispatchConfigError, match="no está en el catálogo"):
        asyncio.run(cd.limite_de_salida("http_openai_compat", "deepseek", "no-existe-pr-k"))


def test_la_consulta_usa_la_clave_unica():
    """Las cuatro del rendimiento: EXPLAIN sobre la consulta REAL."""
    async def _plan():
        conn = await _db_conn()
        try:
            async with conn.cursor() as cur:
                await cur.execute("EXPLAIN " + cd._SQL_CONTRATO, ("deepseek", "deepseek-v4-flash"))
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r)) for r in await cur.fetchall()]
        finally:
            conn.close()
    filas = asyncio.run(_plan())
    assert len(filas) == 1, filas
    assert filas[0]["key"] == "uk_provider_model", filas
    extra = filas[0].get("Extra") or ""
    assert "filesort" not in extra and "temporary" not in extra, filas


def test_cargar_registro_trae_transporte_proveedor_y_url_del_modelo():
    """PR-K ronda 2 (I1): la 2da consulta de cargar_registro contra el esquema
    y la semilla reales. jekyll se siembra con deepseek-v4-flash."""
    from jax.core.registro_facetas import cargar_registro
    registro = asyncio.run(cargar_registro())
    jekyll = registro["jekyll"]
    assert jekyll["transport"] == "http_openai_compat", jekyll
    assert jekyll["provider_modelo"] == "deepseek", jekyll
    assert jekyll["base_url_modelo"] == "https://api.deepseek.com/v1", jekyll
