"""T4 (2026-08-22, planner LLM) -- _build_capability_hint construye el
fragmento dinámico del prompt del planner a partir de get_motor_governance()
real. Gobernanza FABRICADA acá (no pega contra la DB) -- pura por diseño:
lo único que hace es formatear el dict que get_motor_governance() ya
devuelve. test_plan_validation.py cubre el camino que sí valida contra
jax_memory real.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import unittest

from jacobs.plan import _build_capability_hint
from jacobs import store as _store


class CapabilityHintTest(unittest.TestCase):
    def test_regla_fija_nombra_al_unico_motor_con_tool_access(self):
        governance = {
            "jax_local": {
                "allowed_capabilities": {"file_read", "file_write", "generate"},
                "has_tool_access": True,
            },
            "kimi": {
                "allowed_capabilities": {"file_read", "file_write", "implementation"},
                "has_tool_access": False,
            },
        }
        hint = _build_capability_hint(governance)
        regla, _, resto = hint.partition("Capabilities reales")
        assert "motor en: jax_local" in regla
        assert "kimi" not in regla

    def test_regla_fija_lista_todos_los_motores_con_tool_access_si_hay_mas_de_uno(self):
        governance = {
            "jax_local": {"allowed_capabilities": {"file_write"}, "has_tool_access": True},
            "kimi": {"allowed_capabilities": {"file_write"}, "has_tool_access": True},
        }
        hint = _build_capability_hint(governance)
        assert "motor en: jax_local, kimi" in hint

    def test_sin_ningun_motor_con_tool_access_advierte_no_planificar_file_ops(self):
        governance = {"kimi": {"allowed_capabilities": {"generate"}, "has_tool_access": False}}
        hint = _build_capability_hint(governance)
        assert "ningún motor tiene acceso a herramientas" in hint.lower()
        assert "no planifiques" in hint.lower()

    def test_mapa_dinamico_excluye_file_read_write_y_lista_capacidades_reales(self):
        governance = {
            "kimi": {
                "allowed_capabilities": {"file_read", "file_write", "critique", "bug_hunt"},
                "has_tool_access": False,
            },
        }
        hint = _build_capability_hint(governance)
        linea_kimi = next(l for l in hint.splitlines() if l.startswith("- kimi:"))
        assert linea_kimi == "- kimi: bug_hunt, critique"

    def test_motor_sin_capacidades_no_file_no_aparece_en_el_mapa_dinamico(self):
        governance = {
            "jax_local": {
                "allowed_capabilities": {"file_read", "file_write"},
                "has_tool_access": True,
            },
        }
        hint = _build_capability_hint(governance)
        assert "Capabilities reales" not in hint


class CapabilityHintContraDBRealTest(unittest.IsolatedAsyncioTestCase):
    """Confirma que _build_capability_hint interpreta correctamente la forma
    REAL que devuelve get_motor_governance() -- no solo el dict fabricado de
    arriba. Corre contra jax_memory real (mismo criterio que
    test_plan_validation.py), sin mutar nada (solo SELECT)."""

    async def test_hint_contra_governance_real_nombra_jax_local_como_unico_motor_con_tools(self):
        governance = await _store.get_motor_governance()
        hint = _build_capability_hint(governance)
        regla, _, _ = hint.partition("Capabilities reales")
        assert "motor en: jax_local" in regla
        assert "kimi" not in regla
        assert "ada" not in regla
        assert "thot" not in regla


if __name__ == "__main__":
    unittest.main(verbosity=2)
