#!/usr/bin/env python3
"""output_validator.validate() -- distingue "schema declarado pero sin
validacion de campos implementada" (7 nombres reales en uso hoy en la DB
de produccion, ver Global Constraints del plan) de "schema genuinamente
desconocido" (typo, capability mal configurada) -- P10 (DEUDA.md).

Antes de este fix, AMBOS casos devolvian validated=True con un warning
-- el caller (las_manos/motor_registry/worker.py) nunca se enteraba de
que una capability mal configurada estaba devolviendo basura sin
verificar nada. Ahora solo el primer caso (declarado, pendiente) sigue
fail-open; el segundo (no declarado en absoluto) falla cerrado.

Corre desde /home/fruiz/jax/las_manos con:
  PYTHONPATH=/home/fruiz/jax/las_manos \
  /home/fruiz/jax/las_manos/.venv/bin/python \
  /home/fruiz/jax/las_manos/_output_validator_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import unittest

from motor_registry.output_validator import validate


class OutputValidatorSchemaImplementadoTest(unittest.TestCase):
    """Comportamiento existente, NO debe cambiar."""

    def test_schema_implementado_con_todos_los_campos_valida(self):
        content = '{"diff": "...", "files_modified": ["a.py"], "description": "x"}'
        result = validate(content, "code_patch.v1")
        self.assertTrue(result["validated"])
        self.assertEqual(result["missing_fields"], [])

    def test_schema_implementado_con_campos_faltantes_no_valida(self):
        content = '{"diff": "..."}'
        result = validate(content, "code_patch.v1")
        self.assertFalse(result["validated"])
        self.assertIn("files_modified", result["missing_fields"])
        self.assertIn("description", result["missing_fields"])

    def test_sin_schema_name_valida_solo_por_ser_json(self):
        result = validate('{"cualquier_cosa": 1}', "")
        self.assertTrue(result["validated"])

    def test_has_tool_calls_se_saltea_la_validacion(self):
        result = validate("", "code_patch.v1", has_tool_calls=True)
        self.assertTrue(result["skipped"])
        self.assertFalse(result["validated"])

    def test_json_invalido_no_valida(self):
        result = validate("esto no es json", "code_patch.v1")
        self.assertFalse(result["validated"])
        self.assertIn("texto libre", result["warning"])


class OutputValidatorSchemaPendienteTest(unittest.TestCase):
    """Los 7 schemas declarados hoy en la DB de producción
    (capability.output_schema) sin validación de campos implementada --
    deben SEGUIR fail-open (romperían 7 capabilities reales si no)."""

    def test_schema_declarado_pendiente_sigue_fail_open(self):
        for schema_name in (
            "critique.v1", "design.v1", "generate.v1", "analysis.v1",
            "reason.v1", "reconcile.v1", "validation.v1",
        ):
            with self.subTest(schema=schema_name):
                result = validate('{"cualquier_cosa": 1}', schema_name)
                self.assertTrue(result["validated"], f"{schema_name} debe seguir fail-open")
                self.assertIn("declarado", result["warning"].lower())


class OutputValidatorSchemaDesconocidoTest(unittest.TestCase):
    """Comportamiento NUEVO -- schema que no es ni implementado ni
    declarado-pendiente ahora falla cerrado (P10)."""

    def test_schema_realmente_desconocido_falla_cerrado(self):
        result = validate('{"cualquier_cosa": 1}', "typo_que_nadie_declaro.v1")
        self.assertFalse(result["validated"])
        self.assertIn("no reconocido", result["warning"].lower())


if __name__ == "__main__":
    unittest.main()
