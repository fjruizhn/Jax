#!/usr/bin/env python3
"""output_validator._KNOWN_UNIMPLEMENTED_SCHEMAS es una segunda fuente de
verdad -- un snapshot codeado a mano de capability.output_schema tomado
2026-08-25. Sin este test, un cambio SOLO en la DB (agregar una capability
con un output_schema nuevo, o bumpear 'critique.v1' a 'critique.v2') hace
que ese schema empiece a fallar cerrado en producción sin que nadie lo haya
decidido a propósito -- mismo bug de dos-fuentes-de-verdad que
motor/facet_binding (cerrado 2026-08-24), en otra tabla.

Corre contra la DB real, mismo patrón que _catalog_from_db_test.py -- sin
mock, sin fixture local. NO wireado a CI (no hay servicio de DB en el
runner) -- correr a mano antes de cualquier cambio a capability.output_schema
en la DB de producción.

Corre desde /home/fruiz/jax/las_manos con:
  PYTHONPATH=/home/fruiz/jax/las_manos \
  /home/fruiz/jax/las_manos/.venv/bin/python \
  /home/fruiz/jax/las_manos/_output_validator_db_drift_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import unittest

from motor_registry.catalog import MotorCatalog
from motor_registry.output_validator import SCHEMAS, _KNOWN_UNIMPLEMENTED_SCHEMAS


class OutputValidatorDbDriftTest(unittest.IsolatedAsyncioTestCase):
    async def test_todos_los_output_schema_de_capability_estan_cubiertos(self):
        catalog = await MotorCatalog.from_db()
        db_schemas = {
            cap.output_schema
            for cap in catalog._capabilities.values()
            if cap.output_schema
        }
        covered = set(SCHEMAS.keys()) | _KNOWN_UNIMPLEMENTED_SCHEMAS
        uncovered = db_schemas - covered
        self.assertEqual(
            uncovered, set(),
            f"Schemas en capability.output_schema sin cobertura en SCHEMAS "
            f"ni _KNOWN_UNIMPLEMENTED_SCHEMAS: {uncovered} -- actualizar "
            "output_validator.py antes de que un job real falle en "
            "producción sin que nadie lo haya decidido a propósito"
        )


if __name__ == "__main__":
    unittest.main()
