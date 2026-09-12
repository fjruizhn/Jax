"""
Motor Registry — el catálogo en memoria se entera de cambios en la DB.

Regresión del 2026-09-12 (13:03 a 13:29): `routes.py` cargaba
`MotorCatalog.from_db()` UNA vez, al arrancar LAS MANOS. El deploy reinició
jax-las-manos y jax-platform a la vez; la migración `generate` 5 -> 15 corre
al arrancar jax-platform, así que LAS MANOS ya tenía el 5. Jacobs leía 900 s
de la DB y el Motor Registry rechazaba contra su copia vieja: todo paso de
`generate` por kimi/jax_local, rechazado durante 26 minutos.

Se reusa el sello de `facet_resolver` (un archivo por instalación; los
escritores le tocan el mtime): el catálogo se recarga cuando el sello quedó
más nuevo que el instante en que se cargó. Un stat por dispatch; la consulta
a la DB solo cuando cambió.

Solo se mockea `MotorCatalog.from_db` y el worker (nada sale a la red ni a la
DB); el sello es un archivo real en un directorio temporal.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

import facet_resolver
from motor_registry import routes
from motor_registry.catalog import MotorCatalog
from motor_registry.job_store import JobStore
from motor_registry.models import JobStatus, MotorDispatchRequest


def _catalog(generate_minutes: int) -> MotorCatalog:
    return MotorCatalog({
        "motors": {"kimi": {
            "enabled": True, "provider": "kimi", "transport": "http_openai_compat",
            "provider_id": "moonshot", "api_key_env": "", "api_url": "http://x",
            "model": "kimi-k3", "max_context_tokens": 0, "sandbox_only": True,
            "default_timeout_seconds": 600, "supports_reasoning": True,
            "reasoning_default_visibility": "audit_only", "max_tokens": 8000,
        }},
        "capabilities": {"generate": {
            "allowed_motors": ["kimi"], "allowed_callers": ["jacobs"],
            "risk_level": "low", "sandbox_only": True, "requires_human_gate": False,
            "max_execution_minutes": generate_minutes, "max_recursion_depth": 0,
            "output_schema": "generate.v1",
        }},
    })


def _req(timeout_seconds: int = 900) -> MotorDispatchRequest:
    return MotorDispatchRequest(caller="jacobs", capability="generate", motor="kimi",
                                prompt="x", timeout_seconds=timeout_seconds)


class CatalogoSelladoTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.seal = str(Path(self._tmp.name) / "facet-cache-seal")
        self.store = JobStore(str(Path(self._tmp.name) / "jobs.jsonl"))
        self.from_db = AsyncMock(return_value=_catalog(5))
        self._patches = [
            patch.object(facet_resolver, "FACET_SEAL_PATH", self.seal),
            patch.object(routes, "_STORE", self.store),
            patch.object(routes.MotorCatalog, "from_db", self.from_db),
            patch.object(routes.motor_worker, "run", AsyncMock()),
            patch.object(routes, "_CATALOG", None),
            patch.object(routes, "_POLICY", None),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def _stamp(self):
        # Mismo escritor que usan jax-platform y los rebinds de facets.
        assert facet_resolver._tocar_sello()

    async def test_el_incidente_900s_se_acepta_sin_reiniciar(self):
        """Techo 5 min cargado -> la DB pasa a 15 y se estampa el sello -> el
        siguiente dispatch usa 15, sin reiniciar LAS MANOS."""
        await routes.init_motor_catalog()
        first = await routes.dispatch(_req(900))
        assert first.status == JobStatus.REJECTED, first
        assert "5 min" in (first.rejected_reason or "")

        self.from_db.return_value = _catalog(15)
        self._stamp()
        second = await routes.dispatch(_req(900))

        assert second.status != JobStatus.REJECTED, second.rejected_reason
        assert self.from_db.await_count == 2

    async def test_sin_sello_nuevo_no_se_consulta_la_db(self):
        self._stamp()
        time.sleep(0.01)  # el sello queda estrictamente ANTES de la carga
        await routes.init_motor_catalog()
        for _ in range(3):
            await routes.dispatch(_req(300))
        assert self.from_db.await_count == 1, "recargó sin que el sello cambiara"

    async def test_sello_ausente_no_invalida(self):
        """None = sin señal, nunca 'invalidar' (mismo contrato que facets)."""
        await routes.init_motor_catalog()
        assert not os.path.exists(self.seal)
        await routes.dispatch(_req(300))
        assert self.from_db.await_count == 1

    async def test_recargas_concurrentes_hacen_una_sola_consulta(self):
        await routes.init_motor_catalog()
        self._stamp()

        async def slow():
            await asyncio.sleep(0.05)
            return _catalog(15)

        self.from_db.side_effect = slow
        await asyncio.gather(*(routes._ensure_catalog_fresh() for _ in range(5)))
        assert self.from_db.await_count == 2, f"{self.from_db.await_count - 1} recargas en paralelo"

    async def test_recarga_fallida_falla_cerrado_y_reintenta(self):
        """Un catálogo viejo puede seguir permitiendo lo que se revocó: servirlo
        tras un sello nuevo sería un gate fail-open. 503 y el próximo dispatch
        vuelve a intentar."""
        await routes.init_motor_catalog()
        viejo = routes._CATALOG
        self._stamp()
        self.from_db.side_effect = OSError("DB caída")

        with self.assertRaises(HTTPException) as ctx:
            await routes.dispatch(_req(300))
        assert ctx.exception.status_code == 503
        assert routes._CATALOG is viejo, "una recarga fallida no puede dejar el catálogo a medias"

        self.from_db.side_effect = None
        self.from_db.return_value = _catalog(15)
        ok = await routes.dispatch(_req(900))
        assert ok.status != JobStatus.REJECTED, ok.rejected_reason

    async def test_arranque_sin_catalogo_se_recupera_solo(self):
        """Si la DB no respondió al arrancar, el primer dispatch vuelve a cargar
        en vez de rechazar para siempre hasta un reinicio."""
        self.from_db.side_effect = OSError("DB caída al arrancar")
        with self.assertRaises(OSError):
            await routes.init_motor_catalog()
        assert routes._CATALOG is None

        self.from_db.side_effect = None
        self.from_db.return_value = _catalog(15)
        ok = await routes.dispatch(_req(900))
        assert ok.status != JobStatus.REJECTED, ok.rejected_reason


if __name__ == "__main__":
    unittest.main(verbosity=2)
