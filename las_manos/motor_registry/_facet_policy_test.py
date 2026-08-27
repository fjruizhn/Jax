#!/usr/bin/env python3
"""check_facet_admission() -- gobernanza de nivel FACET para callers que
NO tienen concepto de capability (Mesa web -- ver docs/superpowers/specs/
2026-08-27-http-facets-motor-policy-governance-design.md, seccion 3, la
correccion sobre por que esto NO reusa check_capability_admission()).

Corre contra una DB real (jax_memory_test), mismo criterio que
_catalog_from_db_test.py -- sin mock de aiomysql para los casos de datos.
Requiere que la Task 3 (migracion de facet.allowed_callers) ya haya
corrido contra esa DB.

Corre desde la raiz del checkout de `jax` con:
  PYTHONPATH=<checkout>/las_manos <checkout>/las_manos/.venv/bin/python \
    -m unittest motor_registry._facet_policy_test

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import json
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Guard duro, mismo patron que jacobs/_http_facet_admission_test.py:18-24.
# Antes esto era `os.environ.setdefault("JAX_DB_NAME", os.environ.get(
# "JAX_DB_NAME", "jax_memory"))` -- un no-op que resolvia a PRODUCCION por
# default. Este test solo hace SELECT, asi que nada estuvo en riesgo, pero
# la inconsistencia con el test hermano es exactamente la trampa que ya
# causo un incidente real en este proyecto (un test mal nombrado disparando
# dispatches contra prod). Un JAX_DB_NAME distinto ABORTA, no se silencia.
_existing_db_name = os.environ.get("JAX_DB_NAME")
if _existing_db_name and _existing_db_name != "jax_memory_test":
    raise RuntimeError(
        f"JAX_DB_NAME={_existing_db_name!r} ya está seteado -- este test "
        f"corre contra jax_memory_test, no contra esa DB."
    )
os.environ.setdefault("JAX_DB_NAME", "jax_memory_test")

from motor_registry.facet_policy import check_facet_admission


class CheckFacetAdmissionTest(unittest.IsolatedAsyncioTestCase):
    async def test_caller_autorizado_pasa(self):
        # hipatia se siembra con allowed_callers incluyendo "jacobs" (Task 3).
        allowed, reason = await check_facet_admission("jacobs", "hipatia")
        self.assertTrue(allowed, reason)

    async def test_caller_no_autorizado_rechaza(self):
        allowed, reason = await check_facet_admission("caller_fantasma", "hipatia")
        self.assertFalse(allowed)
        self.assertIn("no autorizado", reason)

    async def test_facet_sin_allowed_callers_configurado_rechaza(self):
        # jax_local/kimi/hyde quedan NULL a propósito (fuera de alcance,
        # spec seccion 3) -- NULL debe denegar, no "lista vacia = todos".
        allowed, reason = await check_facet_admission("jacobs", "jax_local")
        self.assertFalse(allowed)
        self.assertIn("no configurado", reason)

    async def test_facet_inexistente_rechaza(self):
        allowed, reason = await check_facet_admission("jacobs", "no_existe")
        self.assertFalse(allowed)


def _conn_returning(raw_value: str | None):
    """Fake de aiomysql que devuelve una sola fila con ese allowed_callers.

    Se mockea acá, y solo acá, porque el caso bajo prueba es un valor que
    la DB real NO tiene (ni debe tener) -- sembrarlo para probarlo seria
    mutar datos compartidos. El resto de la suite sigue contra la DB real.
    """
    cur = MagicMock()
    cur.execute = AsyncMock(return_value=None)
    cur.fetchone = AsyncMock(return_value=(raw_value,))
    cursor_ctx = MagicMock()
    cursor_ctx.__aenter__ = AsyncMock(return_value=cur)
    cursor_ctx.__aexit__ = AsyncMock(return_value=False)
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor_ctx)
    conn.close = MagicMock(return_value=None)
    return conn


class AllowedCallersTipoInesperadoTest(unittest.IsolatedAsyncioTestCase):
    """El CHECK de la DB garantiza json_valid(), NO que sea una lista.

    Sin el guard de isinstance, `caller not in allowed_callers` sobre un
    string JSON degenera en matching de SUBSTRING -- el fail-open mas
    peligroso posible, dentro de la unica funcion cuyo contrato entero es
    fail-closed.
    """

    async def _check(self, raw: str):
        with patch(
            "motor_registry.facet_policy._db_conn",
            new=AsyncMock(return_value=_conn_returning(raw)),
        ):
            return await check_facet_admission("jax_platform_chat", "hipatia")

    async def test_string_json_que_contiene_al_caller_como_substring_rechaza(self):
        # "jax_platform_chat" ES substring de "jax_platform_chatX".
        # Con `in` sobre un str, esto pasaria. Debe denegar.
        allowed, reason = await self._check(json.dumps("jax_platform_chatX"))
        self.assertFalse(allowed, reason)
        self.assertIn("no es una lista", reason)

    async def test_dict_json_rechaza(self):
        # Membership sobre un dict prueba las CLAVES -- igual de silencioso.
        allowed, reason = await self._check(json.dumps({"jax_platform_chat": True}))
        self.assertFalse(allowed, reason)
        self.assertIn("no es una lista", reason)

    async def test_lista_json_valida_sigue_pasando(self):
        # Control: el guard nuevo no rompe el camino bueno.
        allowed, reason = await self._check(json.dumps(["jax_platform_chat"]))
        self.assertTrue(allowed, reason)


if __name__ == "__main__":
    unittest.main()
