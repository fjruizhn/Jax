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


def _validate_nivel_b(
    step,
    catalog_caps: dict,
    motor_facets: frozenset,
    capability_map: dict,
    task_id: str,
):
    """Réplica exacta de jacobs/executor.py:723-743 (bloque NIVEL B de
    validate_capability), retornando CapabilityUnbound tipado en vez de
    string — esto es lo que Task 3 implementa en el original. Cubre las 3
    ramas que devuelven CapabilityUnbound (catálogo sin la capability, motor
    no autorizado, caller no autorizado) más el fail-open de catálogo vacío
    y la resolución de nombre vía capability_map (~ _CAPABILITY_MAP)."""
    if step.facet not in motor_facets:
        return None
    # Fail-open si el catálogo no cargó: net secundario, no SPOF.
    if not catalog_caps:
        return None
    resolved = capability_map.get(step.capability, step.capability)
    entry = catalog_caps.get(resolved)
    if entry is None:
        return CapabilityUnbound(
            required=[resolved], candidates=[], task_id=task_id,
        )
    if step.facet not in entry.get("allowed_motors", []):
        return CapabilityUnbound(
            required=[resolved],
            candidates=list(entry.get("allowed_motors", [])),
            task_id=task_id,
        )
    if "jacobs" not in entry.get("allowed_callers", []):
        return CapabilityUnbound(
            required=[resolved], candidates=[], task_id=task_id,
        )
    return None


def test_motor_no_autorizado_devuelve_capability_unbound_tipado():
    step = _FakeStep(facet="ada", capability="code_swarm")
    catalog = {"code_swarm": {"allowed_motors": ["kimi"], "allowed_callers": ["jacobs"]}}
    result = _validate_nivel_b(step, catalog, frozenset({"kimi", "ada"}), {}, "task-1")
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
    result = _validate_nivel_b(step, catalog, frozenset({"kimi"}), {}, "task-1")
    assert result is None


def test_capability_ausente_del_catalogo_devuelve_capability_unbound_sin_candidatos():
    step = _FakeStep(facet="kimi", capability="misterio")
    catalog = {"code_swarm": {"allowed_motors": ["kimi"], "allowed_callers": ["jacobs"]}}
    result = _validate_nivel_b(step, catalog, frozenset({"kimi"}), {}, "task-2")
    assert result is not None
    assert result.to_dict() == {
        "status": "CAPABILITY_UNBOUND",
        "required": ["misterio"],
        "candidates": [],
        "task_id": "task-2",
    }


def test_caller_no_autorizado_devuelve_capability_unbound_sin_candidatos():
    step = _FakeStep(facet="kimi", capability="code_swarm")
    catalog = {"code_swarm": {"allowed_motors": ["kimi"], "allowed_callers": ["otro_caller"]}}
    result = _validate_nivel_b(step, catalog, frozenset({"kimi"}), {}, "task-3")
    assert result is not None
    assert result.to_dict() == {
        "status": "CAPABILITY_UNBOUND",
        "required": ["code_swarm"],
        "candidates": [],
        "task_id": "task-3",
    }


def test_catalogo_vacio_es_fail_open_devuelve_none():
    step = _FakeStep(facet="kimi", capability="code_swarm")
    result = _validate_nivel_b(step, {}, frozenset({"kimi"}), {}, "task-4")
    assert result is None


def test_facet_no_motor_ignora_capability_devuelve_none():
    step = _FakeStep(facet="hipatia", capability="misterio")
    catalog = {"code_swarm": {"allowed_motors": ["kimi"], "allowed_callers": ["jacobs"]}}
    result = _validate_nivel_b(step, catalog, frozenset({"kimi"}), {}, "task-5")
    assert result is None


def test_capability_map_resuelve_alias_antes_de_buscar_en_catalogo():
    """Mismo comportamiento que _CAPABILITY_MAP en jacobs/executor.py: 'implement'
    resuelve a 'code_swarm' antes de mirar el catálogo — required refleja el
    nombre resuelto, no el alias original."""
    step = _FakeStep(facet="ada", capability="implement")
    catalog = {"code_swarm": {"allowed_motors": ["kimi"], "allowed_callers": ["jacobs"]}}
    capability_map = {"implement": "code_swarm"}
    result = _validate_nivel_b(step, catalog, frozenset({"kimi", "ada"}), capability_map, "task-6")
    assert result is not None
    assert result.to_dict() == {
        "status": "CAPABILITY_UNBOUND",
        "required": ["code_swarm"],
        "candidates": ["kimi"],
        "task_id": "task-6",
    }
