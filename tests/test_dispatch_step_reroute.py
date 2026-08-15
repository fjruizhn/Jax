"""
Réplica aislada de la lógica de reroute que Task 4 agrega al inicio de
jacobs/executor.py:_dispatch_step — mismo patrón que test_jacobs_director.py.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from jacobs.plan import CapabilityUnbound


@dataclass
class _FakeStep:
    facet: str
    capability: str


def _dispatch_with_reroute(step, validate_fn, max_attempts: int = 3):
    """Réplica de la lógica que Task 4 agrega antes del `raise ValueError`
    actual en _dispatch_step: reintenta con candidatos no probados."""
    tried = {step.facet}
    current = step
    for _ in range(max_attempts):
        result = validate_fn(current)
        if result is None:
            return current, None  # listo para dispatch real
        if isinstance(result, str):
            return current, result  # NIVEL A, no reenruta
        untried = [c for c in result.candidates if c not in tried]
        if not untried:
            return current, result  # candidatos agotados, falla como CapabilityUnbound
        tried.add(untried[0])
        current = replace(current, facet=untried[0])
    return current, result


def test_reroute_a_primer_candidato_no_probado():
    step = _FakeStep(facet="ada", capability="code_swarm")

    def fake_validate(s):
        if s.facet == "ada":
            return CapabilityUnbound(required=["code_swarm"], candidates=["kimi"], task_id="t1")
        return None  # kimi sí está autorizado

    final_step, error = _dispatch_with_reroute(step, fake_validate)
    assert error is None
    assert final_step.facet == "kimi"


def test_candidatos_agotados_falla_con_capability_unbound():
    step = _FakeStep(facet="ada", capability="code_swarm")

    def fake_validate(s):
        return CapabilityUnbound(required=["code_swarm"], candidates=["kimi"], task_id="t1")

    final_step, error = _dispatch_with_reroute(step, fake_validate, max_attempts=2)
    assert isinstance(error, CapabilityUnbound)


def test_nivel_a_no_reenruta():
    """Vocabulario desconocido (str, no CapabilityUnbound) no tiene candidatos
    — no se reenruta, falla directo como hoy."""
    step = _FakeStep(facet="ada", capability="capability-inventada")

    def fake_validate(s):
        return "capability desconocida: 'capability-inventada' no está en VALID_CAPABILITIES"

    final_step, error = _dispatch_with_reroute(step, fake_validate)
    assert error == "capability desconocida: 'capability-inventada' no está en VALID_CAPABILITIES"
    assert final_step.facet == "ada"  # no cambió
