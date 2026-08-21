"""
Regresión P0 (2026-08-22, original): file_write/file_read faltaban en
VALID_CAPABILITIES (jacobs/plan.py) -- GAP2 Fase2 las agregó a la DB
(capability, capability_motor) pero nunca a esa lista estática. NIVEL A de
executor.py::validate_capability() rechazaba cualquier step con esas
capabilities ANTES del dispatch, incluso cuando T2 (_validate_plan_capabilities,
capability_motor real) las aprobaba -- dos fuentes de verdad en desacuerdo,
la más vieja ganaba en producción.

Bloque 3 (2026-08-21): VALID_CAPABILITIES eliminado -- NIVEL A ahora
consulta la DB real (store.get_motor_governance()) directamente, no una
copia. Esto reescribe el test para que siga siendo la regresión real: ya
no puede volver a desincronizarse de la DB porque ES la DB, pero re-verifica
contra jax_memory real que file_read/file_write existen como filas de
`capability` -- mismo caso de prueba, mecanismo distinto.

Corre contra la DB real (mismo criterio que tests/test_plan_validation.py).
"""
from __future__ import annotations

import unittest

from jacobs import store


class ValidCapabilitiesFileToolsTest(unittest.IsolatedAsyncioTestCase):
    async def test_file_write_existe_en_capability_real(self):
        governance = await store.get_motor_governance()
        assert "file_write" in governance["capabilities"]

    async def test_file_read_existe_en_capability_real(self):
        governance = await store.get_motor_governance()
        assert "file_read" in governance["capabilities"]
