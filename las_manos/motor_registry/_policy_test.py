#!/usr/bin/env python3
"""MotorPolicy.check() -- caracterizacion ANTES de partirlo en
check_capability_admission() + wrapper (docs/superpowers/specs/
2026-08-27-http-facets-motor-policy-governance-design.md). No existia
ningun test dedicado a check() -- esta suite es la base real contra la
que se mide "cero cambio de comportamiento" en la Task 2. Debe seguir
pasando SIN MODIFICAR despues del split.

Catalogo armado a mano (MotorCatalog(dict), constructor dict-shaped que
el modulo conserva para tests -- ver catalog.py:83-135), sin DB real:
check() es "modulo puro, sin I/O" por diseno.

Corre desde /home/fruiz/jax/las_manos con:
  PYTHONPATH=/home/fruiz/jax/las_manos \
  /home/fruiz/jax/las_manos/.venv/bin/python \
  /home/fruiz/jax/las_manos/motor_registry/_policy_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import unittest

from motor_registry.catalog import MotorCatalog
from motor_registry.policy import MotorPolicy


def _catalog() -> MotorCatalog:
    return MotorCatalog({
        "motors": {
            "kimi": {
                "enabled": True, "sandbox_only": True,
                "transport": "http_openai_compat",
            },
        },
        "capabilities": {
            "implementation": {
                "allowed_motors": ["kimi"],
                "allowed_callers": ["jacobs", "hyde"],
                "risk_level": "medium",
                "sandbox_only": True,
                "requires_human_gate": False,
                "max_execution_minutes": 5,
                "max_recursion_depth": 0,
                "output_schema": "code_patch.v1",
            },
        },
    })


class MotorPolicyCheckTest(unittest.TestCase):
    def setUp(self):
        self.policy = MotorPolicy(_catalog())

    def test_caller_autorizado_pasa(self):
        result = self.policy.check(
            caller="jacobs", capability="implementation", motor=None,
            context_keys=[], recursion_depth=0, human_gate_token=None,
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.resolved_motor, "kimi")

    def test_caller_no_autorizado_rechaza(self):
        result = self.policy.check(
            caller="caller_fantasma", capability="implementation", motor=None,
            context_keys=[], recursion_depth=0, human_gate_token=None,
        )
        self.assertFalse(result.allowed)
        self.assertIn("no autorizado", result.reason)

    def test_capability_desconocida_rechaza(self):
        result = self.policy.check(
            caller="jacobs", capability="no_existe", motor=None,
            context_keys=[], recursion_depth=0, human_gate_token=None,
        )
        self.assertFalse(result.allowed)
        self.assertIn("desconocida", result.reason)

    def test_recursion_depth_excede_limite_rechaza(self):
        result = self.policy.check(
            caller="jacobs", capability="implementation", motor=None,
            context_keys=[], recursion_depth=1, human_gate_token=None,
        )
        self.assertFalse(result.allowed)
        self.assertIn("recursion_depth", result.reason)

    def test_clave_prohibida_en_context_rechaza(self):
        result = self.policy.check(
            caller="jacobs", capability="implementation", motor=None,
            context_keys=["prompt", "api_key"], recursion_depth=0,
            human_gate_token=None,
        )
        self.assertFalse(result.allowed)
        self.assertIn("prohibidas", result.reason)

    def test_timeout_excede_techo_rechaza(self):
        result = self.policy.check(
            caller="jacobs", capability="implementation", motor=None,
            context_keys=[], recursion_depth=0, human_gate_token=None,
            timeout_seconds=301,
        )
        self.assertFalse(result.allowed)
        self.assertIn("excede el techo", result.reason)

    def test_motor_no_sandbox_only_rechaza(self):
        catalog = MotorCatalog({
            "motors": {
                "kimi": {"enabled": True, "sandbox_only": False},
            },
            "capabilities": {
                "implementation": {
                    "allowed_motors": ["kimi"], "allowed_callers": ["jacobs"],
                    "risk_level": "medium", "sandbox_only": True,
                    "requires_human_gate": False, "max_execution_minutes": 5,
                    "max_recursion_depth": 0, "output_schema": "",
                },
            },
        })
        result = MotorPolicy(catalog).check(
            caller="jacobs", capability="implementation", motor=None,
            context_keys=[], recursion_depth=0, human_gate_token=None,
        )
        self.assertFalse(result.allowed)
        self.assertIn("sandbox_only", result.reason)


class CheckCapabilityAdmissionTest(unittest.TestCase):
    """check_capability_admission() -- subconjunto de check() (checks 1-5,
    SIN motor resuelto ni techo de timeout). No requiere que exista un
    motor para la capability -- a diferencia de check(), que fallaria en
    el check 6 (resolver motor) para una capability sin allowed_motors."""

    def setUp(self):
        self.catalog = MotorCatalog({
            "motors": {},
            "capabilities": {
                "research": {
                    "allowed_motors": [],  # HTTP-directo: sin motor, a proposito
                    "allowed_callers": ["jacobs"],
                    "risk_level": "low", "sandbox_only": True,
                    "requires_human_gate": False, "max_execution_minutes": 5,
                    "max_recursion_depth": 0, "output_schema": "",
                },
            },
        })
        self.policy = MotorPolicy(self.catalog)

    def test_caller_autorizado_pasa_sin_necesitar_motor(self):
        result = self.policy.check_capability_admission(
            caller="jacobs", capability="research",
            context_keys=[], recursion_depth=0, human_gate_token=None,
        )
        self.assertTrue(result.allowed)

    def test_caller_no_autorizado_rechaza(self):
        result = self.policy.check_capability_admission(
            caller="jax_platform_chat", capability="research",
            context_keys=[], recursion_depth=0, human_gate_token=None,
        )
        self.assertFalse(result.allowed)
        self.assertIn("no autorizado", result.reason)

    def test_capability_desconocida_rechaza(self):
        result = self.policy.check_capability_admission(
            caller="jacobs", capability="no_existe",
            context_keys=[], recursion_depth=0, human_gate_token=None,
        )
        self.assertFalse(result.allowed)
        self.assertIn("desconocida", result.reason)

    def test_no_tiene_parametro_timeout_seconds(self):
        # El check 8 (techo) queda fuera de esta función a propósito
        # (decisión del spec, punto 2) -- confirmamos que la firma no lo
        # acepta, para que nadie lo reintroduzca sin querer.
        import inspect
        sig = inspect.signature(MotorPolicy.check_capability_admission)
        self.assertNotIn("timeout_seconds", sig.parameters)
        self.assertNotIn("motor", sig.parameters)


if __name__ == "__main__":
    unittest.main()
