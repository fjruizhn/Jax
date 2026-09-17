#!/usr/bin/env python3
"""POST /motor/authorize-facet -- endpoint test end-to-end (FastAPI
TestClient real, sin mockear check_facet_admission: si la DB no responde,
este test lo va a mostrar, que es exactamente lo que queremos saber).

ACTUALIZADO 2026-09-17 (ronda pre-vuelo y continuar): este test estaba VIEJO y
daba 401 en los dos casos. Desde "LAS MANOS autentica a sus llamadores"
(`las_manos/auth_servicio.py`, master) la ruta es deny-by-default y exige la
credencial de servicio, y la identidad declarada en el cuerpo tiene que ser la
que la credencial puede declarar: `/motor/authorize-facet` sólo la alcanza la
credencial `plataforma`, que sólo puede declarar `caller="jax_platform_chat"`.
Nunca se vio el fallo porque el archivo no está en ningún job de CI.

Por eso el caso "caller no autorizado" ya no se puede armar por HTTP: un
`caller` ajeno lo corta el middleware con 403 ANTES de la ruta (eso lo cubre
`tests/test_las_manos_auth_servicio.py::test_declarar_otra_identidad_se_rechaza`,
que usa exactamente `{"caller": "jacobs"}` contra esta ruta). Lo que le queda a
este test es lo suyo: que la ruta devuelve 200 con el veredicto REAL de
`check_facet_admission` contra la DB, en los dos sentidos -- permitido y no
permitido --, y que sin credencial no se llega.

Corre con:
  bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -v \
    las_manos/motor_registry/_authorize_facet_endpoint_test.py"

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import os
import unittest

from fastapi.testclient import TestClient


@unittest.skipUnless(
    os.getenv("JAX_DB_HOST") and os.getenv("JAX_LAS_MANOS_CREDENCIAL_PLATAFORMA"),
    "necesita la MariaDB real (jax_memory_test) y la credencial de servicio",
)
class AuthorizeFacetEndpointTest(unittest.TestCase):
    """`caller` fijo en `jax_platform_chat`: es el ÚNICO que la credencial
    `plataforma` puede declarar en esta ruta. El veredicto lo decide la
    faceta."""

    CALLER = "jax_platform_chat"

    def setUp(self):
        from auth_servicio import ENCABEZADO, IDENTIDAD_PLATAFORMA, cargar_credenciales
        from server import app

        self.client = TestClient(app)
        # La credencial sale del entorno, como la del proceso real: no se
        # escribe ningún secreto en el test.
        self.headers = {ENCABEZADO: cargar_credenciales()[IDENTIDAD_PLATAFORMA].decode()}

    def test_faceta_admitida_devuelve_allowed_true(self):
        resp = self.client.post(
            "/motor/authorize-facet",
            json={"caller": self.CALLER, "facet": "hipatia"},
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertTrue(resp.json()["allowed"], resp.text)

    def test_faceta_desconocida_devuelve_allowed_false(self):
        resp = self.client.post(
            "/motor/authorize-facet",
            json={"caller": self.CALLER, "facet": "faceta_fantasma"},
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertFalse(resp.json()["allowed"], resp.text)

    def test_sin_credencial_no_se_llega_a_la_ruta(self):
        """Deny by default: la ruta no se alcanza sin credencial."""
        resp = self.client.post(
            "/motor/authorize-facet",
            json={"caller": self.CALLER, "facet": "hipatia"},
        )
        self.assertEqual(resp.status_code, 401, resp.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
