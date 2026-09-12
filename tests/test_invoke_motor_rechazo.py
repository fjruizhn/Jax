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

import os
import unittest
from unittest.mock import patch

_existing_db_name = os.environ.get("JAX_DB_NAME")
if _existing_db_name and _existing_db_name != "jax_memory_test":
    raise RuntimeError(
        f"JAX_DB_NAME={_existing_db_name!r} ya está seteado -- unset antes de correr este test."
    )
os.environ.setdefault("JAX_DB_NAME", "jax_memory_test")

from jacobs.executor import _invoke_motor  # noqa: E402
from jacobs.models import Pipeline, Step  # noqa: E402

_REASON = "timeout_seconds=900 excede el techo de 'generate' (5 min = 300s)"


class _Resp:
    status_code = 202

    def json(self):
        return {"job_id": "j1", "status": "rejected", "rejected_reason": _REASON}

    def raise_for_status(self):
        pass


class InvokeMotorRechazoTest(unittest.IsolatedAsyncioTestCase):
    async def test_el_rechazo_falla_con_su_motivo_real(self):
        async def fake_post(client_self, url, **kw):
            return _Resp()

        step = Step(facet="kimi", capability="generate", motor="kimi")
        pipeline = Pipeline(name="t", invoked_by="t", user_id="1", tenant_id="1", mode="dry_run")
        with patch("httpx.AsyncClient.post", fake_post):
            with self.assertRaises(RuntimeError) as ctx:
                await _invoke_motor(step, pipeline, timeout=900)

        assert _REASON in str(ctx.exception), str(ctx.exception)


if __name__ == "__main__":
    unittest.main(verbosity=2)
