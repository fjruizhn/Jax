"""
Jacobs — un paso de motor que vence cancela su job en LAS MANOS.

Pipeline b8f80733 (2026-09-12): el paso 04 venció a los 300 s,
`asyncio.wait_for` canceló a `_invoke_motor` y el pipeline abortó... pero
nadie le avisó a LAS MANOS. El job de kimi siguió tres minutos más, se
cobró y su salida no la leyó nadie.

Dos caminos llegan a "venció": el `wait_for` de afuera (executor.py
envuelve cada step) y el deadline propio del polling. Los dos cancelan.

Solo se mockea httpx; no toca la DB.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import patch

# Mismo guard que jacobs/_step_motor_test.py: importar jacobs.store con
# /etc/jax/.env cargado apuntaría a la DB real.
from base_de_test import exigir_base_de_test  # noqa: E402

exigir_base_de_test()

from jacobs import executor  # noqa: E402
from jacobs.executor import _invoke_motor  # noqa: E402
from jacobs.models import Pipeline, Step  # noqa: E402


class _Resp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class _FakeLasManos:
    """Registra cada POST; el job queda en `job_status` para siempre."""

    def __init__(self, job_status="running", cancel_raises=False):
        self.job_status = job_status
        self.cancel_raises = cancel_raises
        self.posts: list[str] = []

    async def post(self, client_self, url, json=None, **kw):
        self.posts.append(url)
        if url.endswith("/cancel"):
            if self.cancel_raises:
                raise RuntimeError("LAS MANOS no responde")
            return _Resp({"status": "cancelled"})
        return _Resp({"job_id": "j1", "status": "pending"})

    async def get(self, client_self, url, **kw):
        return _Resp({"status": self.job_status, "result_summary": "ok"})

    @property
    def cancel_posts(self):
        return [u for u in self.posts if u.endswith("/motor/job/j1/cancel")]


def _step():
    return Step(facet="kimi", capability="generate", motor="kimi")


def _pipeline():
    return Pipeline(name="test", invoked_by="test", user_id="1", tenant_id="1", mode="dry_run")


class InvokeMotorCancelTest(unittest.IsolatedAsyncioTestCase):
    def _patched(self, fake):
        async def post(client_self, url, **kw):
            return await fake.post(client_self, url, **kw)

        async def get(client_self, url, **kw):
            return await fake.get(client_self, url, **kw)

        return (
            patch("httpx.AsyncClient.post", post),
            patch("httpx.AsyncClient.get", get),
            patch.object(executor, "MOTOR_POLL_INTERVAL", 0.01),
        )

    async def test_wait_for_externo_cancela_el_job(self):
        """El camino del incidente: el wait_for de executor.py vence."""
        fake = _FakeLasManos()
        p1, p2, p3 = self._patched(fake)
        with p1, p2, p3:
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(_invoke_motor(_step(), _pipeline(), timeout=60), timeout=0.2)
        assert len(fake.cancel_posts) == 1, fake.posts

    async def test_deadline_propio_del_polling_cancela_el_job(self):
        fake = _FakeLasManos()
        p1, p2, p3 = self._patched(fake)
        with p1, p2, p3:
            with self.assertRaises(asyncio.TimeoutError):
                await _invoke_motor(_step(), _pipeline(), timeout=0.05)
        assert len(fake.cancel_posts) == 1, fake.posts

    async def test_job_completado_no_se_cancela(self):
        fake = _FakeLasManos(job_status="completed")
        p1, p2, p3 = self._patched(fake)
        with p1, p2, p3:
            result = await _invoke_motor(_step(), _pipeline(), timeout=5)
        assert result["success"] is True
        assert fake.cancel_posts == [], fake.posts

    async def test_si_la_cancelacion_falla_se_conserva_el_timeout_original(self):
        """Avisar a LAS MANOS es lo mejor que se puede hacer, no una
        condición: si falla, el paso igual venció y eso es lo que se
        reporta -- no un RuntimeError de la limpieza que tape la causa."""
        fake = _FakeLasManos(cancel_raises=True)
        p1, p2, p3 = self._patched(fake)
        with p1, p2, p3:
            with self.assertRaises(asyncio.TimeoutError):
                await _invoke_motor(_step(), _pipeline(), timeout=0.05)
        assert len(fake.cancel_posts) == 1, fake.posts


if __name__ == "__main__":
    unittest.main(verbosity=2)
