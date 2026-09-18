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
from jacobs.policy import MAX_PARALLEL_PIPELINES, ContencionAlReservar, CupoAgotado


# ---------------------------------------------------------------------------
#  El contrato: las cuatro escrituras que ocupan cupo lo comprueban
# ---------------------------------------------------------------------------

def test_crear_ocupa_cupo_con_un_insert_condicionado():
    assert not "COUNT(*)" in cupo.SQL_RESERVAR
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


# ---------------------------------------------------------------------------
#  El cupo se mira ANTES de gastar plata
# ---------------------------------------------------------------------------
# Pedido del autor del mecanismo retirado (2026-09-17): el pre-vuelo SONDEA
# FACETAS, y sondear es una llamada paga. Si el cupo se revisara después, un
# resume rechazado por falta de lugar ya habría gastado dinero.

@pytest.mark.parametrize("endpoint", [routes.resume_pipeline, routes.approve_step])
def test_el_cupo_se_mira_antes_del_prevuelo(endpoint):
    """Sobre el CÓDIGO y por ORDEN, no por presencia: invertir las dos líneas
    deja el test rojo, que es justo lo que tiene que pasar."""
    fuente = inspect.getsource(endpoint)
    assert "_cupo_o_429()" in fuente, f"{endpoint.__name__} no mira el cupo antes de sondear"
    assert "_prevuelo_de_reanudacion" in fuente, endpoint.__name__
    assert fuente.index("_cupo_o_429()") < fuente.index("_prevuelo_de_reanudacion"), (
        f"{endpoint.__name__} sondea (paga) ANTES de mirar el cupo: un rechazo por "
        "cupo ya gastó dinero"
    )


def test_la_compuerta_barata_no_es_la_que_decide():
    """`_cupo_o_429` ahorra el gasto, no hace cumplir el límite. Si alguien la
    tomara por el control, una lectura vieja dejaría pasar de más -- por eso la
    decisión sigue viajando dentro del UPDATE."""
    fuente = inspect.getsource(routes._cupo_o_429)
    assert "NO decide" in fuente
    for endpoint in (routes.resume_pipeline, routes.approve_step):
        assert "cupo_maximo=MAX_PARALLEL_PIPELINES" in inspect.getsource(endpoint)


def test_resume_no_sondea_con_el_cupo_lleno():
    """Comportamiento, no sólo orden: con el cupo lleno, el pre-vuelo NO se
    llama. Es el test que se pone rojo si alguien mueve la compuerta."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    from fastapi import BackgroundTasks, HTTPException

    from jacobs.models import Pipeline, PipelineStatus

    interrumpido = Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma",
                            mode="autonomous", status=PipelineStatus.interrupted,
                            created_at=0.0, updated_at=0.0)
    prevuelo = AsyncMock()
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=interrumpido)), \
         patch.object(routes.store, "steps_by_pipeline", AsyncMock(return_value=[])), \
         patch.object(routes, "check_kill_switch", return_value=False), \
         patch.object(routes, "_prevuelo_de_reanudacion", prevuelo), \
         patch.object(routes.cupo, "activos", AsyncMock(return_value=MAX_PARALLEL_PIPELINES)):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(routes.resume_pipeline(
                "p1", routes.ResumeRequest(invoked_by="plataforma"), BackgroundTasks()))
    assert exc.value.status_code == 429
    assert exc.value.detail["code"] == "limite_de_activos"
    prevuelo.assert_not_awaited()


# ---------------------------------------------------------------------------
#  La contención es un 503, no un 500 — y su espera tiene techo declarado
# ---------------------------------------------------------------------------

def test_el_techo_de_espera_acumulada_esta_declarado_y_se_cumple():
    """El número que pidió el coordinador, calculado con las MISMAS constantes
    del código: sin presupuesto, 24 intentos con espera hasta 100 ms dan 2,06 s
    en el peor caso. Subir los reintentos sin mirar el techo pone esto rojo."""
    peor_sin_presupuesto = sum(
        min(0.005 * (2 ** n), cupo.ESPERA_MAXIMA_SEGUNDOS)
        for n in range(cupo.MAX_REINTENTOS_DEADLOCK)
    )
    assert peor_sin_presupuesto > 2.0, peor_sin_presupuesto

    # Con el presupuesto puesto, la espera acumulada NUNCA lo pasa, sea cual sea
    # el jitter. Se comprueba con las dos puntas del azar (0,001 y 0,005), que
    # son las que acotan todas las corridas posibles.
    for base in (0.001, 0.005):
        acumulado, intentos = 0.0, 0
        for n in range(cupo.MAX_REINTENTOS_DEADLOCK):
            espera = min(base * (2 ** n), cupo.ESPERA_MAXIMA_SEGUNDOS)
            if acumulado + espera > cupo.PRESUPUESTO_DE_ESPERA_SEGUNDOS:
                break
            acumulado += espera
            intentos = n + 1
        assert acumulado <= cupo.PRESUPUESTO_DE_ESPERA_SEGUNDOS, (base, acumulado)
        # Y sigue habiendo reintentos de sobra antes de rendirse: el presupuesto
        # acota el tiempo, no convierte esto en "un intento y chau".
        assert intentos >= 10, (base, intentos)
    assert cupo.PRESUPUESTO_DE_ESPERA_SEGUNDOS <= 1.0


def test_la_contencion_agotada_sale_como_contencion_no_como_error_de_base():
    import asyncio
    from unittest.mock import AsyncMock, patch

    import aiomysql

    from jacobs.models import Pipeline

    p = Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma", mode="dry_run",
                 created_at=0.0, updated_at=0.0)
    trabado = aiomysql.OperationalError(1213, "Deadlock found when trying to get lock")
    with patch.object(cupo, "_ejecutar_reserva", AsyncMock(side_effect=trabado)), \
         patch.object(cupo.asyncio, "sleep", AsyncMock()):
        with pytest.raises(ContencionAlReservar) as exc:
            asyncio.run(cupo.reservar_cupo(p, limite=3))
    assert exc.value.intentos >= 1


@pytest.mark.parametrize("endpoint", ["create_pipeline", "resume_pipeline", "approve_step"])
def test_los_endpoints_traducen_la_contencion_a_503(endpoint):
    fuente = inspect.getsource(getattr(routes, endpoint))
    assert "_contencion_503" in fuente, endpoint


def test_el_503_de_contencion_no_se_confunde_con_el_422_del_cupo():
    """422 = "tu pedido no es válido". La contención no tiene nada de inválido:
    es "volvé a intentar", y por eso lleva Retry-After."""
    respuesta = routes._contencion_503(ContencionAlReservar(14, 0.955))
    assert respuesta.status_code == 503
    assert respuesta.detail["code"] == "contencion_al_reservar"
    assert respuesta.headers["Retry-After"] == "1"
