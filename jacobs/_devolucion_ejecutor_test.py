#!/usr/bin/env python3
"""jacobs/executor.py::_correr_pipeline -- lo que hace con cada resultado de
`jacobs.devolucion.evaluar_y_devolver` (spec 2026-09-18-arbitro-devuelve-design).

Arnés propio (mismo patrón de tienda falsa que jacobs/_aviso_executor_test.py,
sin DB ni red) para no competir por el mismo archivo con esa tarea. Acá se
mockea `jacobs.devolucion.evaluar_y_devolver` directamente -- su lógica
interna (parseo del veredicto, tope, presupuesto) ya está cubierta en
jacobs/_devolucion_test.py y jacobs/_devolucion_persistencia_test.py; esto
prueba SOLO lo que `_correr_pipeline` hace con cada resultado.

Ronda de arreglo 2 (revisión 2026-09-18, hallazgo MEDIO -- "tope alcanzado
deja de verse igual que aprobado"):
  - RESULTADO_TOPE -> status PROPIO (`disputed`), evento PIPELINE_DISPUTED,
    aviso con estado 'disputed' -- NUNCA 'completed'.
  - RESULTADO_COMPLETAR -> el camino de siempre (`completed`).
  - RESULTADO_DEVUELTO -> se recorre _correr_pipeline de nuevo (recursión
    corta), sin agendar un aviso en esta pasada.
  - Una excepción de evaluar_y_devolver no tumba el pipeline: se completa
    igual (hallazgo MEDIO de la ronda anterior, cubierto acá con el mismo
    arnés).

Corre con:
  PYTHONPATH=.:las_manos python -m pytest -v jacobs/_devolucion_ejecutor_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio

from base_de_test import fijar_base_de_test  # noqa: E402

fijar_base_de_test()

from unittest.mock import AsyncMock, patch  # noqa: E402

from jacobs import aviso, devolucion, executor  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus, Step  # noqa: E402


class _TiendaFalsa:
    def __init__(self, status="pending", epoca=0):
        self.status = status
        self.epoca = epoca
        self.contexto: dict = {}
        self.eventos: list[tuple[str, dict]] = []

    async def pipeline_update_status_si_epoca(self, pipeline_id, epoca, status,
                                                current_step_index=None, context=None, *,
                                                desde=(PipelineStatus.running,)):
        if epoca != self.epoca or self.status not in {d.value for d in desde}:
            return False
        self.status = status.value
        if context is not None:
            self.contexto = dict(context)
        return True

    async def pipeline_epoca_y_status(self, pipeline_id):
        return self.epoca, PipelineStatus(self.status)

    async def step_upsert_si_epoca(self, s, epoca):
        if epoca != self.epoca or self.status != "running":
            return False
        return True

    async def event_append(self, pipeline_id, event_type, payload=None, step_id=None):
        self.eventos.append((event_type, payload or {}))

    def tipos(self):
        return [t for t, _ in self.eventos]


def _pipeline_con_arbitro(epoca=0, devoluciones=0):
    """2 pasos: un productor + un "árbitro" (alcanza con que
    _correr_pipeline llegue al final de todas las olas -- la forma real del
    veredicto no importa: evaluar_y_devolver está mockeado)."""
    pasos = [
        Step(pipeline_id="p1", step_index=0, facet="jekyll", capability="research",
             input={"prompt": "paso 0"}, depends_on=[]),
        Step(pipeline_id="p1", step_index=1, facet="thot", capability="critique",
             input={"prompt": "arbitra"}, depends_on=[0]),
    ]
    return Pipeline(pipeline_id="p1", name="mi-pipeline", invoked_by="plataforma", mode="autonomous",
                    plan=pasos, context={"objective": "o"}, run_epoch=epoca, devoluciones=devoluciones)


def _correr(monkeypatch, tienda, pipeline, despachar, *, kill_switch=False):
    monkeypatch.setattr(executor, "store", tienda)
    monkeypatch.setattr(executor, "check_kill_switch", lambda: kill_switch)
    monkeypatch.setattr(executor, "save_if_large", lambda pid, sid, raw, **kw: (None, raw))
    monkeypatch.setattr(executor, "_persist_step_to_repo", AsyncMock())
    monkeypatch.setattr(executor, "_dispatch_step", despachar)
    asyncio.run(executor.run_pipeline(pipeline))


async def _despachar_ok(step, pipeline):
    return {"success": True, "result": "ok"}


def test_resultado_tope_termina_disputed_no_completed(monkeypatch):
    """El hallazgo central de esta ronda: agotar el tope con una objeción
    sin resolver NO puede verse igual que un pipeline aprobado."""
    llamadas = []
    monkeypatch.setattr(aviso, "avisar_fin_pipeline", lambda **kw: llamadas.append(kw))
    tienda = _TiendaFalsa()

    with patch(
        "jacobs.devolucion.evaluar_y_devolver",
        AsyncMock(return_value=(devolucion.RESULTADO_TOPE, {"veredicto": None})),
    ):
        _correr(monkeypatch, tienda, _pipeline_con_arbitro(devoluciones=2), _despachar_ok)

    assert tienda.status == "disputed"
    assert tienda.status != "completed"
    assert "PIPELINE_DISPUTED" in tienda.tipos()
    assert "PIPELINE_COMPLETED" not in tienda.tipos()
    assert len(llamadas) == 1
    assert llamadas[0]["estado"] == "disputed"


def test_resultado_completar_sigue_terminando_completed(monkeypatch):
    """Control (Principio VII): el camino de siempre no cambió."""
    llamadas = []
    monkeypatch.setattr(aviso, "avisar_fin_pipeline", lambda **kw: llamadas.append(kw))
    tienda = _TiendaFalsa()

    with patch(
        "jacobs.devolucion.evaluar_y_devolver",
        AsyncMock(return_value=(devolucion.RESULTADO_COMPLETAR, {})),
    ):
        _correr(monkeypatch, tienda, _pipeline_con_arbitro(), _despachar_ok)

    assert tienda.status == "completed"
    assert "PIPELINE_COMPLETED" in tienda.tipos()
    assert "PIPELINE_DISPUTED" not in tienda.tipos()
    assert llamadas[0]["estado"] == "completed"


def test_resultado_devuelto_no_marca_ningun_status_terminal_en_esta_pasada(monkeypatch):
    """RESULTADO_DEVUELTO recorre _correr_pipeline de nuevo -- en la
    llamada ORIGINAL no se escribe completed/disputed ni se avisa; lo que
    pasa después lo decide la corrida recursiva (con evaluar_y_devolver ya
    sin mock, real, pidiendo aprobar para no recursar sin fin)."""
    llamadas = []
    monkeypatch.setattr(aviso, "avisar_fin_pipeline", lambda **kw: llamadas.append(kw))
    tienda = _TiendaFalsa()
    pipeline = _pipeline_con_arbitro()

    primera_vez = {"listo": False}

    async def _evaluar(pl):
        if not primera_vez["listo"]:
            primera_vez["listo"] = True
            # Payload mínimo válido: mismo plan (sin invalidar nada extra),
            # época +1 -- alcanza para que executor.py reasigne y recorra.
            # La época de la TIENDA se adelanta acá a propósito: en la
            # devolución real, continuar_transaccion (mockeada afuera de
            # este test) es quien escribe esa época nueva en la base ANTES
            # de que evaluar_y_devolver devuelva -- este mock reproduce ese
            # efecto para que la corrida recursiva encuentre una fila
            # consistente, igual que la real.
            tienda.epoca = pl.run_epoch + 1
            return devolucion.RESULTADO_DEVUELTO, {
                "plan": pl.plan, "context": pl.context,
                "run_epoch": pl.run_epoch + 1, "afectados": [0, 1],
            }
        return devolucion.RESULTADO_COMPLETAR, {}

    with patch("jacobs.devolucion.evaluar_y_devolver", _evaluar):
        _correr(monkeypatch, tienda, pipeline, _despachar_ok)

    # La corrida recursiva SÍ tiene que terminar completed (la segunda
    # pasada por evaluar_y_devolver dio RESULTADO_COMPLETAR) -- la prueba
    # de que RESULTADO_DEVUELTO no es un callejón sin salida.
    assert tienda.status == "completed"
    assert len(llamadas) == 1  # un solo aviso, de la pasada que sí terminó


def test_una_excepcion_evaluando_el_veredicto_completa_en_vez_de_colgar(monkeypatch):
    """Hallazgo MEDIO (ronda de arreglo 1): un parpadeo de la base evaluando
    el veredicto no puede dejar el pipeline `running` para siempre -- se
    completa como si el árbitro hubiera aprobado."""
    llamadas = []
    monkeypatch.setattr(aviso, "avisar_fin_pipeline", lambda **kw: llamadas.append(kw))
    tienda = _TiendaFalsa()

    async def _explota(pl):
        raise RuntimeError("MariaDB se cayó justo acá")

    with patch("jacobs.devolucion.evaluar_y_devolver", _explota):
        _correr(monkeypatch, tienda, _pipeline_con_arbitro(), _despachar_ok)

    assert tienda.status == "completed"
    assert "PIPELINE_COMPLETED" in tienda.tipos()
    assert len(llamadas) == 1
    assert llamadas[0]["estado"] == "completed"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
