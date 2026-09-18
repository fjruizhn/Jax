"""BLOQUEANTE 1 (revisión final 2026-09-18, tanda historial-y-arreglos-de-pipeline):
el menú de facetas le ofrece el árbitro al planificador, y después lo rechaza.

EL DEFECTO. `_llm_plan` (el camino de qwen/jax_local) arma su prompt con
`_menu_de_facetas(facetas_activas)` y `_PLAN_SYSTEM` -- ninguno de los dos
sabe que existe una faceta reservada para el árbitro. `_menu_de_facetas`
sólo filtra por facetas ACTIVAS (E-17), así que si la faceta árbitro
configurada (`governance["arbitro_faceta"]`, hoy 'thot') está activa -- que
es la condición NORMAL, no un caso raro -- queda en el menú con su
descripción REAL ("criticar/critique"). qwen la lee, arma un step con esa
faceta como PRODUCTOR, y `_con_arbitro` (sala limpia, Task 4) rechaza el
plan DESPUÉS de haber pagado la llamada de planificación completa.

"Quién puede ser productor" vivía en tres lugares: la prosa del prompt de
Ada, `_con_arbitro`, y por omisión (nunca se excluía) el menú. Arreglo de
raíz: `_menu_de_facetas()` también excluye `arbitro_faceta` -- un solo lugar
decide qué facetas se OFRECEN como productoras.

Los tests de acá ejercitan el camino REAL de `_llm_plan` (HTTP mockeado
únicamente, mismo patrón que
`_arbitro_test.py::test_el_prompt_que_recibe_ada_nombra_la_faceta_arbitro_configurada`)
y comprueban que el payload que de verdad sale por HTTP hacia Ollama NO
ofrece la faceta árbitro como opción.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio

import pytest

from jacobs import plan as plan_mod
from jacobs.plan import _menu_de_facetas, _MENU_DE_FACETAS


# ---------------------------------------------------------------------------
# Unidad: _menu_de_facetas por sí sola.
# ---------------------------------------------------------------------------

def test_menu_de_facetas_excluye_al_arbitro_configurado():
    """thot está activa Y es la faceta árbitro configurada: no debe aparecer
    en el menú que se le ofrece al planificador como opción de productor."""
    activas = frozenset({"hipatia", "jekyll", "thot", "ada", "kimi"})
    menu = _menu_de_facetas(activas, arbitro_faceta="thot")
    assert "thot" not in [fila[0] for fila in menu]
    # El resto de las facetas activas sigue ofreciéndose -- no es un filtro
    # que vacíe el menú entero.
    assert {"hipatia", "jekyll", "ada", "kimi"} <= {fila[0] for fila in menu}


def test_menu_de_facetas_sin_arbitro_configurado_no_filtra_nada():
    """Retrocompatibilidad: los callers que no pasan `arbitro_faceta` (o
    pasan None) siguen viendo el menú completo -- mismo comportamiento que
    antes de este arreglo."""
    activas = frozenset({"hipatia", "thot"})
    assert _menu_de_facetas(activas) == _menu_de_facetas(activas, arbitro_faceta=None)
    assert "thot" in [fila[0] for fila in _menu_de_facetas(activas)]


def test_menu_de_facetas_arbitro_inactivo_no_afecta_el_menu():
    """Si la faceta árbitro configurada NO está entre las activas, no hay
    nada que excluir del menú -- ya estaba afuera por no estar activa."""
    activas = frozenset({"hipatia", "jekyll"})
    menu = _menu_de_facetas(activas, arbitro_faceta="thot")
    assert menu == _menu_de_facetas(activas)


# ---------------------------------------------------------------------------
# Camino real de _llm_plan (qwen/jax_local): el payload HTTP que de verdad
# sale no ofrece la faceta árbitro.
# ---------------------------------------------------------------------------

class _RespuestaFalsa:
    def __init__(self, data):
        self._data = data
        self.status_code = 200

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


class _ClienteFalso:
    def __init__(self, capturados):
        self._capturados = capturados

    async def post(self, url, json=None, timeout=None, **kw):
        self._capturados.append(json)
        return _RespuestaFalsa({
            "model": "qwen-de-mentira",
            "message": {"content": "[]"},
        })


def _governance(arbitro_faceta="thot"):
    return {
        "capabilities": {},
        "motors": {},
        "facets": frozenset({"hipatia", "jekyll", "thot", "jax_local"}),
        "arbitro_faceta": arbitro_faceta,
    }


def test_llm_plan_no_ofrece_la_faceta_arbitro_en_el_prompt_real(monkeypatch):
    """Ejercita `_llm_plan` de punta a punta (HTTP mockeado): el texto real
    que sale hacia Ollama no debe nombrar a 'thot' como opción de faceta --
    ni en la lista de "Facetas disponibles", ni en el ejemplo JSON."""
    from facet_resolver import ResolvedFacet

    capturados: list = []

    async def _resolver_falso(key):
        assert key == "jax_local"
        return ResolvedFacet(
            key="jax_local", provider_id="p", base_url=None, model="qwen-de-mentira",
            credential=None, transport="ollama", persona=None, params=None,
        )

    monkeypatch.setattr(plan_mod, "resolve_facet", _resolver_falso)
    monkeypatch.setattr(
        plan_mod, "limite_de_salida",
        lambda *a, **kw: asyncio.sleep(0, result={"options": {"num_predict": 100}}),
    )
    monkeypatch.setattr(plan_mod, "obtener_cliente_http", lambda: _ClienteFalso(capturados))
    monkeypatch.setattr(plan_mod, "record_resolved_version_safe",
                         lambda *a, **kw: asyncio.sleep(0))

    builder = plan_mod.PlanBuilder()
    governance = _governance("thot")

    async def correr():
        return await builder._llm_plan(
            "un objetivo cualquiera", 5, "", governance=governance,
        )

    asyncio.run(correr())

    assert len(capturados) == 1, "se esperaba una sola llamada HTTP a Ollama"
    payload = capturados[0]
    texto_completo = "\n".join(m["content"] for m in payload["messages"])
    assert "'thot'" not in texto_completo
    # thot NO debe figurar como faceta ofrecida en la lista de "Facetas
    # disponibles:" -- las demás activas sí.
    assert "thot (criticar/critique)" not in texto_completo
    assert "hipatia (investigar/research)" in texto_completo


def test_llm_plan_rechaza_si_solo_queda_el_arbitro_en_el_menu(monkeypatch):
    """Caso límite: si la ÚNICA faceta activa es la reservada para el
    árbitro, el menú queda vacío después de excluirla -- `_llm_plan` tiene
    que fallar cerrado con `CerebroNoDisponible`, ANTES de despachar nada
    (nunca llega a resolver la faceta ni a pegarle a Ollama): no se ofrece
    nada, así que no hay nada que planificar."""
    async def _resolver_que_no_deberia_llamarse(key):
        raise AssertionError(
            "no debería resolverse ninguna faceta: el menú vacío tiene que "
            "cortar ANTES, no llegar a fallar por otra razón (resolve_facet "
            "real explotando sin config) que daría un verde por el motivo "
            "equivocado"
        )

    monkeypatch.setattr(plan_mod, "resolve_facet", _resolver_que_no_deberia_llamarse)

    builder = plan_mod.PlanBuilder()
    governance = {
        "capabilities": {}, "motors": {},
        "facets": frozenset({"thot"}),
        "arbitro_faceta": "thot",
    }

    async def correr():
        with pytest.raises(plan_mod.CerebroNoDisponible) as exc:
            await builder._llm_plan("objetivo", 5, "", governance=governance)
        assert "ninguna faceta del menú está activa" in str(exc.value)

    asyncio.run(correr())
