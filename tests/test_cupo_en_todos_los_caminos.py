"""TODO camino que pone un pipeline a correr respeta `MAX_PARALLEL_PIPELINES`.

POR QUÉ EXISTE (2026-09-17). Buscando qué invariantes sostenía el candado del
cupo apareció un agujero que no era de ninguna de las dos ramas, sino anterior:
**`resume` y `approve-step` movían un pipeline de `interrupted` a correr sin
mirar el cupo.** Ni el `asyncio.Lock` de proceso ni el `GET_LOCK` con nombre los
tomaban -- los tomaban sólo `crear` y `continue`. Y `interrupted` no cuenta como
activo, así que con tres pipelines interrumpidos y tres `resume` se pasaba el
límite sin que nada lo impidiera.

Un límite que dice que existe y no existe es peor que no tenerlo: alguien
dimensionó la GPU, el pool y los techos de costo creyendo que como mucho hay
tres pipelines a la vez.

Lo que vigila este archivo es el CONTRATO, no una implementación: por cada
camino que ocupa cupo, la escritura que lo ocupa lleva la condición adentro. Hoy
son cuatro: crear (INSERT), continue, resume y approve-step (UPDATE). Si mañana
aparece un quinto, el test de la partición de estados y este par de controles
son los que tienen que ponerse rojos.

Rojo contra master (`b450248`) verificado: `resume` y `approve-step` no le
pasaban `cupo_maximo` a `pipeline_tomar_epoca`, y el UPDATE de la época no
llevaba ninguna condición de cupo.
"""
from __future__ import annotations

import inspect

import pytest

from jacobs import continuar as servicio_continuar
from jacobs import cupo, routes, store
from jacobs.policy import MAX_PARALLEL_PIPELINES, CupoAgotado


# ---------------------------------------------------------------------------
#  El contrato: las cuatro escrituras que ocupan cupo lo comprueban
# ---------------------------------------------------------------------------

def test_crear_ocupa_cupo_con_un_insert_condicionado():
    assert "COUNT(*)" in cupo.SQL_RESERVAR
    assert "< %s" in cupo.SQL_RESERVAR


def test_continue_ocupa_cupo_con_un_update_condicionado():
    """`continue` NO inserta: revive una fila que ya existe. El INSERT
    condicionado de crear no lo cubre, y por eso hizo falta la misma regla en
    forma de UPDATE -- si no, retirar el candado dejaba este camino sin nada."""
    assert "cupo_x.c < %s" in store._SQL_PIPELINE_CONTINUAR
    assert "cupo_maximo" in inspect.signature(store.continuar_transaccion).parameters


@pytest.mark.parametrize("endpoint", [routes.resume_pipeline, routes.approve_step])
def test_resume_y_approve_step_le_pasan_el_tope_a_la_escritura(endpoint):
    """EL HALLAZGO. Contra master, `cupo_maximo` no aparece en el cuerpo de
    ninguno de los dos: reanudar y aprobar ponían un pipeline a correr sin
    comprobar nada."""
    fuente = inspect.getsource(endpoint)
    assert "cupo_maximo=MAX_PARALLEL_PIPELINES" in fuente, (
        f"{endpoint.__name__} pone un pipeline a correr sin comprobar el cupo"
    )
    assert "CupoAgotado" in fuente, (
        f"{endpoint.__name__} no traduce el cupo lleno a un rechazo explícito"
    )


def test_el_update_de_la_epoca_lleva_la_condicion_cuando_se_le_pide_tope():
    con = store._sql_tomar_epoca(False, 1, True)
    sin = store._sql_tomar_epoca(False, 1, False)
    assert "cupo_x.c < %s" in con
    # Sin tope queda como estaba: el llamador DECLARA que su escritura ocupa
    # cupo. Un default silencioso haría que un camino que NO ocupa cupo (si
    # algún día lo hay) lo comprobara igual, y nadie lo notaría.
    assert "cupo_x" not in sin


# ---------------------------------------------------------------------------
#  El comportamiento: el cupo lleno se rechaza, no se ignora
# ---------------------------------------------------------------------------

def test_el_tope_llega_a_la_sentencia_y_el_cupo_lleno_levanta_cupo_agotado():
    """Sobre la función real de `store`, con la base sustituida: si el UPDATE
    no toca la fila y el recuento está en el tope, el motivo es el cupo -- y se
    dice, en vez de devolver None y que el endpoint hable de "otro pedido ganó",
    que mandaría a buscar un problema que no existe."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    from jacobs.models import PipelineStatus

    with patch.object(store, "_ejecutar_condicional", AsyncMock(return_value=0)), \
         patch.object(store, "pipeline_count_active",
                      AsyncMock(return_value=MAX_PARALLEL_PIPELINES)):
        with pytest.raises(CupoAgotado) as exc:
            asyncio.run(store.pipeline_tomar_epoca(
                "p1", 3, (PipelineStatus.interrupted,),
                cupo_maximo=MAX_PARALLEL_PIPELINES))
    assert exc.value.activos == MAX_PARALLEL_PIPELINES


def test_si_no_es_el_cupo_sigue_siendo_otro_pedido_gano():
    """El otro motivo de 0 filas no se convierte en un 429 mentiroso."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    from jacobs.models import PipelineStatus

    with patch.object(store, "_ejecutar_condicional", AsyncMock(return_value=0)), \
         patch.object(store, "pipeline_count_active", AsyncMock(return_value=0)):
        assert asyncio.run(store.pipeline_tomar_epoca(
            "p1", 3, (PipelineStatus.interrupted,),
            cupo_maximo=MAX_PARALLEL_PIPELINES)) is None


def test_continue_traduce_el_cupo_de_la_escritura_a_429():
    fuente = inspect.getsource(servicio_continuar.continuar)
    assert "except CupoAgotado" in fuente
    assert "limite_de_activos" in fuente
