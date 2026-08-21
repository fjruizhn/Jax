"""
Regresión P0 (2026-08-22): file_write/file_read faltaban en VALID_CAPABILITIES
(jacobs/plan.py) -- GAP2 Fase2 las agregó a la DB (capability, capability_motor)
pero nunca a esta lista estática. NIVEL A de executor.py::validate_capability()
rechaza cualquier step con esas capabilities ANTES del dispatch, incluso
cuando T2 (_validate_plan_capabilities, capability_motor real) las aprueba --
dos fuentes de verdad en desacuerdo, la más vieja gana en producción.

Encontrado en vivo: un pipeline con step {facet: jax_local, capability:
file_write, motor: jax_local} -- exactamente lo que PipelineModal.jsx arma
hoy tras el fix de T5 -- falló con "capability desconocida: 'file_write' no
está en VALID_CAPABILITIES", pese a que jax_local tiene has_tool_access=True
y capability_motor lista file_write para jax_local.
"""
from __future__ import annotations

from jacobs.plan import VALID_CAPABILITIES


def test_file_write_esta_en_valid_capabilities():
    assert "file_write" in VALID_CAPABILITIES


def test_file_read_esta_en_valid_capabilities():
    assert "file_read" in VALID_CAPABILITIES
