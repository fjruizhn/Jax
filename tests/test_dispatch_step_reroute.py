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


# --- E2b-1a MINOR 2 (auditoria #362): el reroute REAL usa el predicado ------
# Las pruebas de arriba son una replica de la logica. Estas ejercitan
# jacobs.executor._dispatch_step de verdad (validate_capability y el store
# reemplazados; nada toca la DB ni la red).

import asyncio  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402

import pytest  # noqa: E402

from base_de_test import exigir_base_de_test  # noqa: E402


def _despachar_con_candidatos(candidatos, *, facet_original="ada"):
    exigir_base_de_test()
    from jacobs import executor
    from jacobs.models import Step

    step = Step(facet=facet_original, capability="code_swarm", step_index=0)
    evento = AsyncMock()

    async def validar(s):
        if s.facet == facet_original:
            return CapabilityUnbound(required=["code_swarm"], candidates=list(candidatos), task_id="t1")
        return None

    async def correr():
        with patch.object(executor, "validate_capability", validar), \
             patch.object(executor.store, "event_append", evento), \
             patch.object(executor, "_build_context_input", side_effect=RuntimeError("llego al dispatch")):
            return await executor._dispatch_step(step, executor.Pipeline(
                name="t", invoked_by="t", user_id="1", tenant_id="1", mode="dry_run"))

    return step, evento, correr


@pytest.mark.parametrize("candidato", ["kimi", "jax_local", "hyde", "faceta_nueva_subprocess"])
def test_el_reroute_real_no_aterriza_en_una_faceta_no_ejecutable(candidato):
    """Un candidato que el predicado rechaza (motor, Hyde, faceta nueva) no es
    destino de reroute: candidatos agotados, sin evento STEP_REROUTED."""
    step, evento, correr = _despachar_con_candidatos([candidato])
    with pytest.raises(ValueError, match="candidatos agotados"):
        asyncio.run(correr())
    assert step.facet == "ada"
    evento.assert_not_awaited()


def test_el_reroute_real_salta_los_no_ejecutables_y_elige_el_primer_http():
    step, evento, correr = _despachar_con_candidatos(["kimi", "hyde", "thot"])
    with pytest.raises(RuntimeError, match="llego al dispatch"):
        asyncio.run(correr())
    assert step.facet == "thot"
    [llamada] = evento.await_args_list
    assert llamada.args[1] == "STEP_REROUTED" and llamada.args[2]["new_facet"] == "thot"
