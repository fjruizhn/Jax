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
        """Task 5 (2026-09-18, historial-y-arreglos-de-pipeline): el caso
        real que motivó este test (kimi con fila capability_motor para
        file_write pero has_tool_access=False, T1) ya NO existe en la DB
        real -- `ada`/`kimi` subieron a has_tool_access=1 (medido: los dos
        llaman a read_file de verdad contra su proveedor). El único motor
        que sigue en 0 (`thot`) no tiene fila capability_motor para
        file_write, así que con él la validación rechaza un turno ANTES
        (nivel `capability_motor`, otro test ya lo cubre) y nunca llega a
        este branch. La única forma de seguir probando el mecanismo mismo
        (no el dato de hoy, que puede volver a cambiar) es una `governance`
        sintética -- mismo parámetro que `build()` ya acepta para la foto
        de Ruling R43, no un mock del método bajo prueba."""
        governance = {
            "motors": {"kimi": False},
            "capabilities": {
                "file_write": {"allowed_motors": ["kimi"], "requires_human_gate": False},
            },
            "facets": frozenset(),
            "arbitro_faceta": None,
        }
        steps = [_step(0, facet="kimi", capability="file_write")]
        with self.assertRaises(PlanRejected) as ctx:
            await _validate_plan_capabilities(steps, governance)
        assert "kimi" in str(ctx.exception)
        assert "has_tool_access" in str(ctx.exception)

    async def test_rechaza_file_write_real_contra_todos_los_motores_sembrados(self):
        """Complemento del test sintético de arriba, contra la DB real de
        hoy (Task 5): de los motores que SÍ están en capability_motor para
        file_write (jax_local, kimi -- verificado con SELECT, `ada` NO
        tiene fila ahí), ninguno debe rechazarse por has_tool_access -- los
        dos lo tienen en 1 (ver migrations.py::_seed_ada_kimi_has_tool_access).
        `ada` deliberadamente NO entra en este loop: no está en
        `MOTOR_FACETS` (jacobs/models.py -- solo {kimi, jax_local}), así que
        _validate_plan_capabilities ni siquiera la mira (mismo bypass que
        `test_ignora_facets_http_directos_sin_gobernanza_de_motor_registry`
        ejercita para hipatia). Su `has_tool_access=1` es habilitación de
        datos para cuando ese camino se gobierne -- hoy no cambia el
        dispatch HTTP-directo de executor.py, que no lee esta columna.
        `thot` tampoco entra: sigue sin fila capability_motor y su
        proveedor rechaza function tools con reasoning_effort (evidencia en
        el docstring de la migración)."""
        for motor_key in ("jax_local", "kimi"):
            steps = [_step(0, facet=motor_key, capability="file_write")]
            await _validate_plan_capabilities(steps)  # no debe lanzar

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
        """Task 5 (2026-09-18): el step original (kimi/file_write) ya no
        sirve para este test -- kimi ganó has_tool_access, así que ese plan
        AHORA es válido (justo el comportamiento que Task 5 buscaba). Se
        usa una capability inexistente para seguir probando lo que este
        test realmente verifica: que build() propaga el rechazo antes de
        persistir, sin importar cuál sea la violación."""
        builder = PlanBuilder()
        with self.assertRaises(PlanRejected):
            await builder.build(
                pipeline_id="test-pipeline-2",
                objective="objetivo de prueba",
                steps_spec=[{"facet": "kimi", "capability": "capability-que-no-existe", "prompt": "x"}],
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
