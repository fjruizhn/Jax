"""
Jacobs — un rechazo del Motor Registry llega con su motivo real.

E2E de la cadena, 2026-09-12 (pipeline 93fcd81f): el Motor Registry rechazó
el paso de kimi con "timeout_seconds=900 excede el techo de 'generate' (5
min = 300s)" -- su catálogo en memoria era anterior a la migración 5 -> 15.
El step falló con `name 'capability' is not defined`: la línea que loguea el
rechazo nombraba una variable que no existe desde el 2026-06-29. El motivo
real no llegó ni al log ni al step; hubo que buscarlo en motor_jobs.jsonl.

Solo se mockea httpx; no toca la DB.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock, patch

from base_de_test import exigir_base_de_test  # noqa: E402

exigir_base_de_test()

from jacobs.executor import _invoke_motor  # noqa: E402
from policy.execution_control.errors import GovernedExecutionRequiredError  # noqa: E402
from jacobs.models import Pipeline, Step  # noqa: E402

class InvokeMotorRechazoTest(unittest.IsolatedAsyncioTestCase):
    async def test_motor_sin_ejecucion_gobernada_no_hace_http(self):
        step = Step(facet="kimi", capability="generate", motor="kimi")
        pipeline = Pipeline(name="t", invoked_by="t", user_id="1", tenant_id="1", mode="dry_run")
        cliente = Mock()
        cliente.post = AsyncMock(side_effect=AssertionError("no debe despachar al Motor Registry"))
        with patch("jacobs.executor.obtener_cliente_http", return_value=cliente) as obtener_cliente:
            with self.assertRaises(GovernedExecutionRequiredError):
                await _invoke_motor(step, pipeline, timeout=900)
        obtener_cliente.assert_not_called()
        cliente.post.assert_not_awaited()


class CodigoMuertoDelDespachoLegacyTest(unittest.TestCase):
    """E2b-1a MINOR 3 (auditoria #362): el despacho directo a LAS MANOS esta
    cerrado; el codigo que lo sostenia (polling, lectura de resultado,
    cancelacion, `_rechazado`) quedo inalcanzable y se borro. Mismo riesgo que
    advierte `_invoke_hyde`: quien lo resucite sin que nada lo note reabre un
    camino sin gobernar. Si hace falta de nuevo, va con ejecucion gobernada."""

    def test_executor_ya_no_tiene_el_polling_ni_la_cancelacion_de_motor_jobs(self):
        from jacobs import executor
        for nombre in ("_read_motor_result", "_cancel_motor_job", "MOTOR_POLL_INTERVAL"):
            self.assertFalse(hasattr(executor, nombre), nombre)

    def test_las_manos_dispatch_ya_no_tiene_el_cuerpo_legacy(self):
        from motor_registry import routes
        self.assertFalse(hasattr(routes, "_rechazado"))
        self.assertFalse(hasattr(routes, "human_gate"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
