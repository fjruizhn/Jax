"""
El techo de ejecucion declarado en la DB es un techo de verdad.

EL DEFECTO. `capability.max_execution_minutes` decia ser el limite de ejecucion
de una capability, pero nada lo hacia cumplir en plan-time: un
`timeout_seconds` explicito en el spec de un step pisaba el default y llegaba
intacto a `asyncio.wait_for`. El techo declarado no era un techo -- era una
columna que alguien podia editar en un panel admin creyendo que cambiaba algo.

`_validate_plan_capabilities` existia y no lo miraba: validaba capability<->motor
y `has_tool_access`, nada mas. Y solo recorria los steps de `MOTOR_FACETS`.

QUE SE DECIDIO, Y POR QUE (las dos cosas con datos, 2026-09-01):

1. **Aplica a TODOS los steps, no solo a los de MOTOR_FACETS.** El techo es
   propiedad de la CAPABILITY, no del motor, y `executor.py` envuelve CADA step
   en `asyncio.wait_for(..., timeout=step.timeout_seconds)` -- motor o facet
   HTTP, da igual. Un techo que no cubre a la mitad de los steps no es un techo.

2. **Rechaza, no recorta.** Recortar en silencio convierte un error de
   configuracion en comportamiento sordo: el plan corre con un timeout que
   nadie pidio y nadie ve.

3. **Capability sin techo declarado -> 300 s, no "cualquier cosa".** Es el
   unico caso ambiguo y se decidio con datos, no por gusto: `assemble` esta
   exenta A PROPOSITO en el planner y no tiene fila en `capability`. Aceptar
   cualquier timeout ahi seria fail-open; rechazar de plano rompería
   `assemble`. Se le da `_DEFAULT_TIMEOUT_SECONDS`, que es el mismo valor que
   el codigo ya le asigna. Medido en `jacobs_steps`: 6 steps `assemble` reales,
   TODOS con timeout_seconds=300 -- este techo no rechaza ninguno.

Los tests no tocan la DB: `get_motor_governance` se parchea. Lo que se prueba
es la POLITICA, no el I/O.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest

from jacobs import plan as plan_mod
from jacobs.models import Step

# Espeja la forma real de get_motor_governance()["capabilities"], medida contra
# la DB el 2026-09-01: design/reason/reconcile a 15 min, el resto a 5.
GOBERNANZA = {
    "capabilities": {
        "analysis": {"allowed_motors": ["kimi"], "max_execution_minutes": 5},
        "design": {"allowed_motors": ["kimi"], "max_execution_minutes": 15},
        "research": {"allowed_motors": [], "max_execution_minutes": 5},
    },
    "motors": {"kimi": True, "jax_local": True},
}


@pytest.fixture(autouse=True)
def gobernanza(monkeypatch):
    async def _fake():
        return GOBERNANZA

    from jacobs import store as _store

    monkeypatch.setattr(_store, "get_motor_governance", _fake)


def _step(capability="analysis", timeout=300, facet="hipatia", motor=None, indice=0):
    return Step(
        step_id=str(uuid.uuid4()),
        pipeline_id="p1",
        step_index=indice,
        facet=facet,
        motor=motor,
        capability=capability,
        input={},
        depends_on=[],
        timeout_seconds=timeout,
        skip_on_fail=False,
    )


def _validar(steps):
    return asyncio.run(plan_mod._validate_plan_capabilities(steps))


def _rechazo(steps) -> plan_mod.PlanRejected:
    with pytest.raises(plan_mod.PlanRejected) as e:
        _validar(steps)
    return e.value


# ---------------------------------------------------------------------------
# (a) por encima del techo -> rechaza
# ---------------------------------------------------------------------------

def test_timeout_por_encima_del_techo_rechaza_el_plan():
    # analysis: techo 5 min = 300 s
    exc = _rechazo([_step(capability="analysis", timeout=301)])
    assert exc.violations, "tiene que haber al menos una violacion"


def test_el_mensaje_nombra_el_valor_el_techo_y_la_capability():
    """Un rechazo que no dice cuanto se pidio ni contra que se lo midio obliga
    a leer el codigo para entenderlo."""
    exc = _rechazo([_step(capability="analysis", timeout=1800)])
    mensaje = " ".join(v.reason for v in exc.violations)
    assert "1800" in mensaje, mensaje
    assert "300" in mensaje, mensaje
    assert "analysis" in mensaje, mensaje


def test_el_default_NUNCA_puede_exceder_el_techo():
    """INVARIANTE NUEVO, habilitado por la deduplicacion (2026-09-01).

    Antes habia dos causas posibles para un timeout por encima del techo: "lo
    pidio el spec" y "el default por-capability del codigo diverge de la DB".
    La segunda YA NO PUEDE PASAR: el default sale de `_techo_segundos()`, la
    MISMA funcion y la MISMA fila que produce el techo. Este test lo fija --
    para toda capability, con fila o sin ella, el default es exactamente el
    techo, asi que la comparacion `timeout > techo` nunca puede dispararse por
    un default.
    """
    from jacobs.plan import _techo_segundos, _DEFAULT_TIMEOUT_SECONDS

    caps = GOBERNANZA["capabilities"]
    for capability in list(caps) + ["assemble", "inventada"]:
        techo, _ = _techo_segundos(caps, capability)
        default, _ = _techo_segundos(caps, capability)
        assert default == techo, capability
        assert default <= techo, capability
    # y la capability sin fila cae al piso conservador, no a "sin limite"
    assert _techo_segundos(caps, "assemble")[0] == _DEFAULT_TIMEOUT_SECONDS


def test_el_mensaje_ya_no_ofrece_dos_causas():
    """Corolario: el mensaje de rechazo dice UNA sola causa, porque solo queda
    una. Ofrecer dos cuando una es imposible manda a investigar un camino que
    no existe."""
    mensaje = " ".join(v.reason for v in _rechazo(
        [_step(capability="analysis", timeout=1800)]).violations)
    assert "spec" in mensaje, mensaje
    assert "check_timeout_consistency" not in mensaje, (
        "el script ya no existe: se borro al quedar una sola fuente")


# ---------------------------------------------------------------------------
# (b) por debajo -> pasa   (c) sin timeout explicito -> pasa
# ---------------------------------------------------------------------------

def test_timeout_por_debajo_del_techo_pasa():
    _validar([_step(capability="analysis", timeout=120)])


def test_timeout_exactamente_en_el_techo_pasa():
    """El techo es inclusivo: 300 s con techo de 300 s no es una violacion. Si
    fuera exclusivo, el default por-capability de TODA capability de 5 min
    quedaria rechazado -- o sea, ningun plan pasaria."""
    _validar([_step(capability="analysis", timeout=300)])


def test_el_default_por_capability_pasa():
    """El camino normal: sin timeout explicito, _from_spec pone el default por
    capability. design=900 contra un techo de 15 min=900 -> pasa."""
    _validar([_step(capability="design", timeout=900)])


# ---------------------------------------------------------------------------
# (d) capability sin techo declarado -- la decision, fijada
# ---------------------------------------------------------------------------

def test_capability_sin_fila_no_recibe_techo_infinito():
    """FAIL-OPEN EVITADO: 'assemble' no tiene fila en `capability`. Sin esta
    regla, cualquier timeout pasaria por no estar declarado."""
    exc = _rechazo([_step(capability="assemble", timeout=3600)])
    mensaje = " ".join(v.reason for v in exc.violations)
    assert "sin fila" in mensaje, mensaje


def test_assemble_con_su_timeout_real_sigue_pasando():
    """La contracara, medida: los 6 steps `assemble` reales de la DB usan 300 s.
    La regla nueva no puede romperlos."""
    _validar([_step(capability="assemble", timeout=300)])


def test_max_execution_minutes_vacio_cae_al_techo_conservador():
    """Hoy inalcanzable (la columna es NOT NULL), pero la rama existe para que
    un cambio de esquema futuro caiga hacia 300 y no hacia 'sin limite'."""
    gob = {"capabilities": {"rara": {"allowed_motors": [], "max_execution_minutes": 0}},
           "motors": {}}
    import jacobs.store as _store
    original = _store.get_motor_governance

    async def _fake():
        return gob

    _store.get_motor_governance = _fake
    try:
        exc = _rechazo([_step(capability="rara", timeout=900)])
    finally:
        _store.get_motor_governance = original
    assert "300" in " ".join(v.reason for v in exc.violations)


# ---------------------------------------------------------------------------
# El alcance: TODOS los steps, no solo los de MOTOR_FACETS
# ---------------------------------------------------------------------------

def test_tambien_cubre_los_facets_HTTP():
    """El hueco que el item describia: `_validate_plan_capabilities` solo
    recorria MOTOR_FACETS, y `hipatia` no es uno. executor.py igual lo envuelve
    en asyncio.wait_for con su timeout."""
    exc = _rechazo([_step(facet="hipatia", capability="research", timeout=9999)])
    assert any(v.facet == "hipatia" for v in exc.violations), exc.violations


def test_tambien_cubre_los_facets_de_motor():
    exc = _rechazo([_step(facet="kimi", capability="analysis", timeout=9999)])
    assert any(v.facet == "kimi" for v in exc.violations), exc.violations


def test_un_plan_sin_steps_de_motor_ya_no_sale_por_la_puerta_de_atras():
    """Antes habia un `if not relevant: return` que salteaba TODA la validacion
    cuando ningun step era de motor. Ese early-return se fue: con el, un plan
    100% HTTP nunca veia el techo."""
    exc = _rechazo([
        _step(facet="hipatia", capability="research", timeout=100),
        _step(facet="jekyll", capability="analysis", timeout=99999, indice=1),
    ])
    assert len(exc.violations) == 1
    assert exc.violations[0].step_index == 1


def test_varios_steps_pasados_reportan_todas_las_violaciones():
    """Reportar solo la primera obliga a arreglar de a una."""
    exc = _rechazo([
        _step(capability="analysis", timeout=999, indice=0),
        _step(capability="analysis", timeout=888, indice=1),
    ])
    assert len(exc.violations) == 2, exc.violations


def test_un_plan_valido_no_lanza():
    _validar([
        _step(facet="hipatia", capability="research", timeout=200, indice=0),
        _step(facet="kimi", capability="analysis", timeout=300, indice=1),
    ])
