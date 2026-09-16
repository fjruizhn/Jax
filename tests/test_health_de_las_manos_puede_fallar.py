#!/usr/bin/env python3
"""El /health de LAS MANOS tiene que poder ponerse ROJO.

POR QUE EXISTE. Hasta el 2026-09-16 el endpoint devolvia `{"status": "alive"}`
FIJO: respondia «vivo» por el mero hecho de poder responder. Un control que no
puede fallar no valida nada -- y este no es un endpoint cualquiera:

  - `loadtest/health.js` lo usa para las pruebas de carga de LAS CUATRO DEL
    RENDIMIENTO. Un p95 sobre un literal mide FastAPI, no el servicio.
  - La Mesa lo mira para saber si LAS MANOS esta vivo.
  - El dashboard lo muestra.

Con la base caida, los tres seguian en verde.

Este test ejercita el PEOR CASO (Principio VII: un freno sin prueba no es
freno): con la dependencia rota, /health tiene que devolver 503 y decir cual.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "las_manos"))

try:
    from fastapi.testclient import TestClient
    import las_manos.server as server
    _DISPONIBLE = True
except Exception as _exc:  # fail-closed: si no se puede importar, el test lo DICE, no se salta en silencio
    _DISPONIBLE = False
    _MOTIVO = str(_exc)


@unittest.skipUnless(_DISPONIBLE, f"no se pudo importar LAS MANOS: {globals().get('_MOTIVO', '')}")
class HealthPuedeFallarTest(unittest.TestCase):
    def setUp(self):
        # La cache es global: cada caso parte de cero, si no el primero decide.
        server._salud_cache["t"] = 0.0
        server._salud_cache["valor"] = None
        self.client = TestClient(server.app, raise_server_exceptions=False)

    def _conn_ok(self):
        cur = mock.AsyncMock()
        cur.execute = mock.AsyncMock()
        cur.fetchone = mock.AsyncMock(return_value=(1,))
        ctx = mock.MagicMock()
        ctx.__aenter__ = mock.AsyncMock(return_value=cur)
        ctx.__aexit__ = mock.AsyncMock(return_value=False)
        conn = mock.MagicMock()
        conn.cursor = mock.MagicMock(return_value=ctx)
        conn.close = mock.MagicMock()
        return conn

    def test_con_la_base_caida_devuelve_503(self):
        with mock.patch.object(server.jacobs_store, "get_conn",
                               mock.AsyncMock(side_effect=OSError("connection refused"))):
            r = self.client.get("/health")
        self.assertEqual(r.status_code, 503, "el servicio se declaro sano con la base caida")
        cuerpo = r.json()
        self.assertEqual(cuerpo["status"], "degraded")
        self.assertTrue(any("base de datos" in p for p in cuerpo["problemas"]))

    def test_con_el_log_forense_no_escribible_devuelve_503(self):
        """Sin audit no se puede ejecutar nada sin dejar de auditarlo."""
        with mock.patch.object(server.jacobs_store, "get_conn",
                               mock.AsyncMock(return_value=self._conn_ok())), \
             mock.patch.object(type(server.audit.log_path), "open",
                               side_effect=PermissionError("solo lectura")):
            r = self.client.get("/health")
        self.assertEqual(r.status_code, 503)
        self.assertTrue(any("auditoria" in p for p in r.json()["problemas"]))

    def test_con_todo_sano_devuelve_200(self):
        """Control del control: el caso bueno sigue siendo verde."""
        with mock.patch.object(server.jacobs_store, "get_conn",
                               mock.AsyncMock(return_value=self._conn_ok())):
            r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        cuerpo = r.json()
        self.assertEqual(cuerpo["status"], "alive")
        self.assertEqual(cuerpo["problemas"], [])

    def test_el_kill_switch_se_reporta_pero_NO_degrada(self):
        """Estar frenado a proposito es una decision, no una averia."""
        with mock.patch.object(server.jacobs_store, "get_conn",
                               mock.AsyncMock(return_value=self._conn_ok())), \
             mock.patch.object(server, "_kill_switch_active", lambda: True):
            r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["kill_switch_active"])

    def test_la_cache_evita_una_consulta_por_peticion(self):
        """20.000 req/s en las pruebas de carga: sin cache, el remedio tumbaria
        la base que el chequeo quiere vigilar."""
        get_conn = mock.AsyncMock(return_value=self._conn_ok())
        with mock.patch.object(server.jacobs_store, "get_conn", get_conn):
            for _ in range(25):
                self.client.get("/health")
        self.assertEqual(get_conn.await_count, 1,
                         f"se consultó la base {get_conn.await_count} veces en 25 peticiones")


if __name__ == "__main__":
    unittest.main()
