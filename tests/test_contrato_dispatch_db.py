"""
PR-K (2026-09-14): `limite_de_salida()` contra una MariaDB real, con el esquema
de las migraciones de jax-platform (job jacobs-gobernanza-db).

Ronda 4: CADA test crea SU fila de `model` (provider existente de la semilla
núcleo, `deepseek`; model_id sintético único) y la borra en `finally`. Antes
dependían de filas sembradas (zhipu/glm-5.2 sin contrato, deepseek-v4-flash
con 131072) y la semilla de PR-L las cambió: el job quedó rojo en Jax#158. Una
semilla es de otro PR; un test de contrato no puede depender de ella.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from base_de_test import es_base_de_test  # noqa: E402

_db = os.environ.get("JAX_DB_NAME", "")
if not es_base_de_test(_db):
    raise RuntimeError(f"JAX_DB_NAME={_db!r}: este test solo corre contra una base de tests.")

import contrato_dispatch as cd  # noqa: E402  (camino de Jacobs: symlink en las_manos/)
from facet_resolver import _db_conn  # noqa: E402

# Proveedor de la semilla núcleo de jax-platform (_PROVIDER_SEED): la FK de
# model.provider_id lo exige, y es el que menos cambia.
_PROVIDER = "deepseek"


async def _con_fila(max_tokens_param, max_output_tokens, cuerpo):
    """Crea una fila de `model` propia, corre `cuerpo(model_id)` y la borra."""
    model_id = f"zz-pr-k-{uuid.uuid4().hex[:12]}"
    conn = await _db_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO model (provider_id, model_id, is_alias, status, source, "
                "source_checked_at, max_tokens_param, max_output_tokens) "
                "VALUES (%s, %s, FALSE, 'available', 'manual', NOW(), %s, %s)",
                (_PROVIDER, model_id, max_tokens_param, max_output_tokens),
            )
        return await cuerpo(model_id)
    finally:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM model WHERE provider_id=%s AND model_id=%s", (_PROVIDER, model_id))
        conn.close()


def test_lee_nombre_y_tope_de_la_fila_real():
    async def cuerpo(model_id):
        return await cd.limite_de_salida("http_openai_compat", _PROVIDER, model_id)
    assert asyncio.run(_con_fila("max_completion_tokens", 4242, cuerpo)) == {"max_completion_tokens": 4242}


def test_una_fila_sin_contrato_falla_con_los_dos_updates():
    async def cuerpo(model_id):
        with pytest.raises(cd.ModelDispatchConfigError) as exc:
            await cd.limite_de_salida("http_openai_compat", _PROVIDER, model_id)
        return str(exc.value)
    texto = asyncio.run(_con_fila(None, None, cuerpo))
    assert "UPDATE model SET max_tokens_param" in texto
    assert "UPDATE model SET max_output_tokens" in texto


def test_fila_inexistente_falla():
    with pytest.raises(cd.ModelDispatchConfigError, match="no está en el catálogo"):
        asyncio.run(cd.limite_de_salida("http_openai_compat", _PROVIDER, f"zz-no-existe-{uuid.uuid4().hex}"))


def test_la_consulta_usa_la_clave_unica():
    """Las cuatro del rendimiento: EXPLAIN sobre la consulta REAL, con una fila
    que existe (sin fila, MariaDB responde "Impossible WHERE" y no muestra key)."""
    async def cuerpo(model_id):
        conn = await _db_conn()
        try:
            async with conn.cursor() as cur:
                await cur.execute("EXPLAIN " + cd._SQL_CONTRATO, (_PROVIDER, model_id))
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r)) for r in await cur.fetchall()]
        finally:
            conn.close()
    filas = asyncio.run(_con_fila("max_tokens", 1, cuerpo))
    assert len(filas) == 1, filas
    assert filas[0]["key"] == "uk_provider_model", filas
    extra = filas[0].get("Extra") or ""
    assert "filesort" not in extra and "temporary" not in extra, filas


def test_cargar_registro_trae_transporte_proveedor_y_url_del_modelo():
    """PR-K ronda 2 (I1): la 2da consulta de cargar_registro contra el esquema
    real. Ronda 4: sin fijar qué modelo siembra la semilla -- se compara contra
    lo que dicen facet, model y provider para el binding de jekyll."""
    from jax.core.registro_facetas import cargar_registro

    async def esperado():
        conn = await _db_conn()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT f.transport, m.provider_id, p.base_url FROM facet_binding b "
                    "JOIN facet f ON f.`key` = b.facet_key JOIN model m ON m.id = b.model_ref "
                    "JOIN provider p ON p.id = m.provider_id "
                    "WHERE b.facet_key = 'jekyll' AND b.role = 'primary'"
                )
                return await cur.fetchone()
        finally:
            conn.close()

    transporte, provider_id, base_url = asyncio.run(esperado())
    jekyll = asyncio.run(cargar_registro())["jekyll"]
    assert (jekyll["transport"], jekyll["provider_modelo"], jekyll["base_url_modelo"]) == \
        (transporte, provider_id, base_url), jekyll
    assert base_url, "la semilla de provider tiene que traer la base_url de jekyll"
