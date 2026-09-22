#!/usr/bin/env python3
"""MotorCatalog.from_db() — lee motor/capability/capability_motor de la DB
compartida (jax_memory) en vez de config.toml (R4 — motor desacoplado de
faceta). Corre contra la DB real de desarrollo -- mismo criterio que
credential_resolver.py, sin mock de DB (aiomysql no tiene un modo in-memory
liviano establecido en este repo).

Corre desde /home/fruiz/jax/las_manos con:
  PYTHONPATH=/home/fruiz/jax/las_manos \
  /home/fruiz/jax/las_manos/.venv/bin/python \
  /home/fruiz/jax/las_manos/_catalog_from_db_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import unittest

from facet_resolver import _db_conn
from motor_registry.catalog import MotorCatalog


async def _binding_vigente(facet_key: str) -> tuple[str, str]:
    """(provider_id, model_id) que `facet_binding` declara HOY para
    `facet_key`, en la base a la que este proceso está conectado (test o
    producción, según JAX_DB_NAME) -- leído en vivo, nunca un literal.
    Hallazgo 2026-09-21 (mismo pedido que el catálogo de modelos de
    jax_memory_test): un literal como `kimi.model == 'kimi-k3'` se rompe
    apenas alguien cambia el binding real, y ya pasó (jax#248, con la CLAVE
    de faceta en vez del modelo, mismo patrón). Principio IV: el código es
    lógica, no datos -- el dato vigente se LEE, no se copia a mano."""
    conn = await _db_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT provider_id, model_id FROM facet_binding "
                "WHERE facet_key=%s AND role='primary'",
                (facet_key,),
            )
            fila = await cur.fetchone()
    finally:
        conn.close()
    assert fila is not None, f"'{facet_key}' no tiene binding primario en esta base"
    return fila


class CatalogFromDbTest(unittest.IsolatedAsyncioTestCase):
    async def test_from_db_carga_kimi_y_ada(self):
        catalog = await MotorCatalog.from_db()
        kimi = catalog.get_motor("kimi")
        assert kimi is not None, "kimi no cargó desde DB"
        assert kimi.transport == "http_openai_compat", kimi.transport
        provider_id, model_id = await _binding_vigente("kimi")
        assert kimi.provider_id == provider_id, (kimi.provider_id, provider_id)
        # model reusa el campo existente (no model_id nuevo) -- ver
        # motor_resolved (facet_binding, no motor.model_ref) más abajo.
        assert kimi.model == model_id, (kimi.model, model_id)
        # max_tokens == 0, NO 8000 (arreglo-ci-3, 2026-09-18): este test nunca
        # había corrido en CI -- el detector de cobertura recién lo enganchó
        # esta noche, y afirmaba el valor VIEJO de antes de D1. D1 de Fernando
        # (spec 2026-09-17 §1 y §7 A, jax-platform/backend/db/migrations.py::
        # _motor_max_tokens_al_catalogo_v1 + MOTORES_AL_TOPE_DEL_CATALOGO):
        # kimi y ada pasan a motor.max_tokens=0 a propósito -- 0 = "sin
        # presupuesto propio, usá model.max_output_tokens del catálogo" (ver
        # las_manos/motor_registry/worker.py::_limite_del_motor y
        # motor_registry/catalog.py::MotorEntry.max_tokens). El 8000 fijo fue
        # el que cortó el pipeline ef9b2d6e el 2026-09-16 -- por eso se
        # corrigió. Verificado en vivo 2026-09-18 con SELECT directo contra
        # jax_memory Y jax_memory_test (127.0.0.1:3308): las dos tienen
        # kimi.max_tokens=0 y ada.max_tokens=0 hoy. En CI (jacobs-gobernanza-db,
        # DB efímera con las migraciones de jax-platform), _MOTOR_SEED siembra
        # kimi/ada con max_tokens=0 directo -- no depende del UPDATE de
        # corrección. Si esto vuelve a fallar en 8000, alguien deshizo D1 sin
        # querer.
        assert kimi.max_tokens == 0, kimi.max_tokens
        assert kimi.enabled is True

    async def test_from_db_carga_capability_con_allowed_motors_en_orden(self):
        catalog = await MotorCatalog.from_db()
        cap = catalog.get_capability("generate")
        assert cap is not None, "capability 'generate' no cargó desde DB"
        # Task 4 agregó jax_local a 'generate' (priority 2, detrás de kimi/ada).
        assert cap.allowed_motors == ["kimi", "ada", "jax_local"], cap.allowed_motors

    async def test_from_db_capability_critique_incluye_thot_en_orden(self):
        catalog = await MotorCatalog.from_db()
        cap = catalog.get_capability("critique")
        assert cap is not None
        # Task 8 agregó thot a 'critique' (priority 0, delante de ada).
        assert "thot" in cap.allowed_motors, cap.allowed_motors
        assert cap.allowed_motors == ["thot", "ada"], cap.allowed_motors

    async def test_from_db_has_tool_access_jax_local_kimi_ada_true_thot_false(self):
        """T1 (diagnóstico pipeline 19ad2c42-cdf): has_tool_access vivía solo
        como `if motor == "jax_local"` en worker.py:488, sin fuente
        consultable. Ahora es columna en `motor` -- este test confirma que
        MotorCatalog la lee (no la vuelve a hardcodear en otro lado).

        Task 5 (2026-09-18, historial-y-arreglos-de-pipeline): la
        contradicción original (kimi con fila capability_motor pero
        has_tool_access=False) se cerró -- `jacobs` ya tenía 'jacobs' en
        `capability.allowed_callers` de file_read/file_write desde GAP2
        Fase2 (2026-08-19); el hueco real era este, no el del caller.
        Medido con una llamada real por faceta contra su proveedor real:
        ada (glm-5.3, zhipu) y kimi (kimi-k3, moonshot) responden HTTP 200
        y llaman a read_file; thot (gpt-6-astra, openai) responde HTTP 400
        ("Function tools with reasoning_effort are not supported for
        gpt-6-astra in /v1/chat/completions"). thot se queda en False a
        propósito, dos razones: su proveedor la rechaza, y es la faceta
        árbitro (juzga lo que otros steps produjeron, no necesita leer el
        workspace). Ver migrations.py::_seed_ada_kimi_has_tool_access."""
        catalog = await MotorCatalog.from_db()
        jax_local = catalog.get_motor("jax_local")
        assert jax_local is not None
        assert jax_local.has_tool_access is True, jax_local
        kimi = catalog.get_motor("kimi")
        assert kimi is not None
        assert kimi.has_tool_access is True, kimi
        ada = catalog.get_motor("ada")
        assert ada is not None
        assert ada.has_tool_access is True, ada
        thot = catalog.get_motor("thot")
        assert thot is not None
        assert thot.has_tool_access is False, thot


if __name__ == "__main__":
    unittest.main(verbosity=2)
