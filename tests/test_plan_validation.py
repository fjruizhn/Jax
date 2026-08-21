"""
T2/T3 (2026-08-21, diagnóstico pipeline 19ad2c42-cdf) — validación bloqueante
de PlanBuilder.build() ANTES de que routes.py persista el plan en
jacobs_steps.

Corre contra la DB real (jax_memory) -- mismo criterio que
las_manos/_catalog_from_db_test.py: la validación consulta capability_motor
y motor.has_tool_access reales, no un mock. Sin pytest-asyncio instalado en
este venv -- unittest.IsolatedAsyncioTestCase, mismo patrón que
_catalog_from_db_test.py.

Uso (necesita JAX_DB_PORT=3308, ver /etc/jax/.env):
  set -a; source <(sudo grep JAX_DB_ /etc/jax/.env); set +a
  PYTHONPATH=/home/fruiz/jax:/home/fruiz/jax/las_manos \
    /home/fruiz/jax/.venv/bin/python -m unittest tests.test_plan_validation -v

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import unittest

from jacobs.models import Step
from jacobs.plan import PlanBuilder, PlanRejected, _validate_plan_capabilities


def _step(step_index, facet, capability, motor=None, depends_on=None):
    return Step(
        pipeline_id="test-pipeline",
        step_index=step_index,
        facet=facet,
        motor=motor,
        capability=capability,
        depends_on=depends_on or [],
    )


class PlanValidationTest(unittest.IsolatedAsyncioTestCase):
    async def test_rechaza_capability_de_tool_contra_motor_sin_has_tool_access(self):
        """kimi tiene fila capability_motor para file_write (ronda 7) pero
        has_tool_access=False (T1) -- la contradicción diagnosticada."""
        steps = [_step(0, facet="kimi", capability="file_write")]
        with self.assertRaises(PlanRejected) as ctx:
            await _validate_plan_capabilities(steps)
        assert "kimi" in str(ctx.exception)
        assert "has_tool_access" in str(ctx.exception)

    async def test_acepta_capability_de_tool_contra_jax_local(self):
        steps = [_step(0, facet="jax_local", capability="file_write")]
        await _validate_plan_capabilities(steps)  # no debe lanzar

    async def test_rechaza_capability_ausente_de_capability_motor(self):
        steps = [_step(0, facet="kimi", capability="capability-que-no-existe")]
        with self.assertRaises(PlanRejected) as ctx:
            await _validate_plan_capabilities(steps)
        assert "capability-que-no-existe" in str(ctx.exception)
        assert "capability_motor" in str(ctx.exception)

    async def test_ignora_facets_http_directos_sin_gobernanza_de_motor_registry(self):
        """hipatia/jekyll/thot/ada no pasan por Motor Registry -- la
        validación de capability_motor/has_tool_access no les aplica
        (executor.py los dispatcha por HTTP directo, sin worker.py de por
        medio)."""
        steps = [_step(0, facet="hipatia", capability="cualquier-cosa-inventada")]
        await _validate_plan_capabilities(steps)  # no debe lanzar

    async def test_build_con_steps_spec_propaga_el_rechazo_antes_de_devolver_el_plan(self):
        builder = PlanBuilder()
        with self.assertRaises(PlanRejected):
            await builder.build(
                pipeline_id="test-pipeline-2",
                objective="objetivo de prueba",
                steps_spec=[{"facet": "kimi", "capability": "file_write", "prompt": "x"}],
            )

    async def test_cleanroom_ahora_bloquea_en_vez_de_solo_advertir(self):
        """T3: antes _check_cleanroom solo emitía logger.warning. Un plan
        donde un facet audita su propio trabajo debe rechazarse igual que un
        plan con capability/motor inejecutable."""
        builder = PlanBuilder()
        with self.assertRaises(PlanRejected) as ctx:
            await builder.build(
                pipeline_id="test-pipeline-3",
                objective="objetivo de prueba",
                steps_spec=[
                    {"facet": "kimi", "capability": "implementation", "prompt": "x"},
                    {"facet": "kimi", "capability": "critique", "prompt": "y", "depends_on": [0]},
                ],
            )
        msg = str(ctx.exception).lower()
        assert "mismo facet" in msg or "cleanroom" in msg


if __name__ == "__main__":
    unittest.main(verbosity=2)
