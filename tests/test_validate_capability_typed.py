"""
Réplica aislada de la lógica NIVEL B de validate_capability (jacobs/executor.py),
mismo patrón que tests/test_jacobs_director.py — sin importar jacobs.executor
directo (evita dependencias de red/DB en el import).
"""
from __future__ import annotations

from dataclasses import dataclass

from jacobs.plan import CapabilityUnbound


@dataclass
class _FakeStep:
    facet: str
    capability: str


def _validate_nivel_b(step, catalog_caps: dict, motor_facets: frozenset, task_id: str):
    """Réplica exacta de jacobs/executor.py:718-733, retornando CapabilityUnbound
    tipado en vez de string — esto es lo que Task 3 implementa en el original."""
    if step.facet not in motor_facets:
        return None
    entry = catalog_caps.get(step.capability)
    if entry is None:
        return CapabilityUnbound(
            required=[step.capability], candidates=[], task_id=task_id,
        )
    if step.facet not in entry.get("allowed_motors", []):
        return CapabilityUnbound(
            required=[step.capability],
            candidates=list(entry.get("allowed_motors", [])),
            task_id=task_id,
        )
    return None


def test_motor_no_autorizado_devuelve_capability_unbound_tipado():
    step = _FakeStep(facet="ada", capability="code_swarm")
    catalog = {"code_swarm": {"allowed_motors": ["kimi"], "allowed_callers": ["jacobs"]}}
    result = _validate_nivel_b(step, catalog, frozenset({"kimi", "ada"}), "task-1")
    assert result is not None
    assert result.to_dict() == {
        "status": "CAPABILITY_UNBOUND",
        "required": ["code_swarm"],
        "candidates": ["kimi"],
        "task_id": "task-1",
    }


def test_motor_autorizado_devuelve_none():
    step = _FakeStep(facet="kimi", capability="code_swarm")
    catalog = {"code_swarm": {"allowed_motors": ["kimi"], "allowed_callers": ["jacobs"]}}
    result = _validate_nivel_b(step, catalog, frozenset({"kimi"}), "task-1")
    assert result is None
