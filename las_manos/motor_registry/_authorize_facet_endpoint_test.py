#!/usr/bin/env python3
"""POST /motor/authorize-facet -- endpoint test end-to-end (FastAPI
TestClient real, sin mockear check_facet_admission: si la DB no responde,
este test lo va a mostrar, que es exactamente lo que queremos saber).

Corre desde /home/fruiz/jax/las_manos con:
  PYTHONPATH=/home/fruiz/jax/las_manos \
  /home/fruiz/jax/las_manos/.venv/bin/python \
  /home/fruiz/jax/las_manos/motor_registry/_authorize_facet_endpoint_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import unittest

from fastapi.testclient import TestClient
from server import app


class AuthorizeFacetEndpointTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_caller_autorizado_devuelve_allowed_true(self):
        resp = self.client.post(
            "/motor/authorize-facet",
            json={"caller": "jacobs", "facet": "hipatia"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["allowed"])

    def test_caller_no_autorizado_devuelve_allowed_false(self):
        resp = self.client.post(
            "/motor/authorize-facet",
            json={"caller": "caller_fantasma", "facet": "hipatia"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["allowed"])


if __name__ == "__main__":
    unittest.main()
