#!/usr/bin/env python3
"""NIVEL C de validate_capability(): admisión para facets HTTP-directos.
Contra la DB real de test, sin mockear (salvo el fallo de DB, que se
simula). Ver Task 6 del plan de gobernanza de _HTTP_FACETS.

Lo que ESTE archivo cubre, y nada más:
  - camino feliz: hipatia/research con caller 'jacobs' pasa
  - requires_human_gate=1 sin token -> denegado
  - MotorCatalog.from_db() que explota -> la excepción PROPAGA (fail-closed:
    _run_one_step la captura y falla el step; nunca despacha sin verificar)

NO cubre el rechazo por allowed_callers (check 2). Se escribió y se
descartó a propósito: hoy TODA capability sembrada lista "jacobs" en
allowed_callers, así que el caso solo es alcanzable mockeando el catálogo
-- y entonces el test prueba el mock, no el sistema. La cobertura real de
ese check vive en las_manos/motor_registry/_policy_test.py
(CheckCapabilityAdmissionTest::test_caller_no_autorizado_rechaza), que lo
ejercita directo sobre check_capability_admission() con un catálogo
armado a mano. Si algún día una capability excluye a 'jacobs', este
archivo es el lugar para el test de integración.

Corre desde /home/fruiz/jax con:
  PYTHONPATH=/home/fruiz/jax .venv/bin/python -m unittest jacobs._http_facet_admission_test

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, patch

_existing_db_name = os.environ.get("JAX_DB_NAME")
if _existing_db_name and _existing_db_name != "jax_memory_test":
    raise RuntimeError(
        f"JAX_DB_NAME={_existing_db_name!r} ya está seteado -- este test "
        f"corre contra jax_memory_test, no contra esa DB."
    )
os.environ.setdefault("JAX_DB_NAME", "jax_memory_test")

from jacobs.executor import validate_capability
from jacobs.models import Step


class HttpFacetAdmissionTest(unittest.IsolatedAsyncioTestCase):
    async def test_research_con_caller_jacobs_pasa(self):
        # 'research' (hipatia) tiene allowed_callers=["jacobs"] sembrado
        # -- jacobs SIEMPRE fue el caller real de este camino.
        step = Step(facet="hipatia", capability="research", input={"prompt": "x"})
        result = await validate_capability(step)
        self.assertIsNone(result)


    async def test_capability_con_human_gate_es_denegada(self):
        """NIVEL C pasa human_gate_token=None fijo -- no hay hoy ningun
        mecanismo para que Jacobs consiga un token real por este camino.

        No es hipotetico: `bug_hunt` y `code_swarm` tienen
        requires_human_gate=1 y listan "jacobs" en allowed_callers
        (verificado contra jax_memory y jax_memory_test, 2026-08-27), y
        NADA impide que el planner se las asigne a un facet HTTP --
        _validate_plan_capabilities (jacobs/plan.py:294-308) solo inspecciona
        MOTOR_FACETS, y el cierre de vocabulario del planner (plan.py:639)
        las deja pasar porque SI existen en la tabla `capability`.

        Este test documenta el comportamiento REAL que se shippeo: el step
        falla, y falla como str (no CapabilityUnbound), asi que _dispatch_step
        NO reintenta ni reenruta -- es un fallo duro. Correcto (fail-closed),
        pero conviene que este escrito y no que se descubra en produccion.
        """
        step = Step(facet="ada", capability="bug_hunt", input={"prompt": "x"})
        result = await validate_capability(step)
        self.assertIsInstance(result, str)
        self.assertIn("human_gate_token", result)


class HttpFacetAdmissionFailClosedTest(unittest.IsolatedAsyncioTestCase):
    async def test_db_caida_al_leer_catalogo_no_deja_pasar_el_step(self):
        """Simula MotorCatalog.from_db() fallando (DB caida/timeout real,
        no un mock que devuelve un error prolijo) -- confirma que
        validate_capability() PROPAGA la excepcion en vez de tragarla, que
        es lo que _run_one_step necesita para fallar el step cerrado."""
        step = Step(facet="hipatia", capability="research", input={"prompt": "x"})
        with patch(
            "motor_registry.catalog.MotorCatalog.from_db",
            new=AsyncMock(side_effect=RuntimeError("DB no responde (simulado)")),
        ):
            with self.assertRaises(RuntimeError):
                await validate_capability(step)


if __name__ == "__main__":
    unittest.main()
