"""Barrido de fuente: la creación de pipelines no vuelve a un candado global.

POR QUÉ (2026-09-17). `jacobs/routes.py` tenía `_pipeline_create_lock =
asyncio.Lock()`, un candado de PROCESO que serializaba toda la creación —
incluidos los 20-40 s del planificador— para que `MAX_PARALLEL_PIPELINES`
fuera un límite real. Hacía cumplir el cupo y, de paso, ponía a la Mesa a
esperar detrás de las delegaciones de Ada (medido en el frente G, 2026-09-17:
techo de ~43 delegaciones/s y 70-3200 ms de espera para el chat de Fernando).

El cupo ahora lo hace cumplir la base (`jacobs/cupo.py`, `INSERT` condicionado
que decide por filas afectadas). Este barrido es el freno que impide que el
candado vuelva por comodidad en el próximo cambio: un módulo con un `Lock()`
global en el camino de crear un pipeline reintroduce exactamente el cuello que
se pagó por sacar.

Es un test de FUENTE a propósito: un test de comportamiento vería el límite
cumplirse igual con candado y sin candado (por eso el control de concurrencia
vive en `jacobs/_cupo_io_test.py` y mide además el tiempo).
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from jacobs import cupo, routes

RUTAS = Path(routes.__file__)


def _arbol() -> ast.Module:
    return ast.parse(RUTAS.read_text(encoding="utf-8"))


def test_routes_no_tiene_un_candado_de_modulo():
    globales_con_lock = []
    for nodo in _arbol().body:
        if not isinstance(nodo, ast.Assign):
            continue
        llamada = nodo.value
        if not isinstance(llamada, ast.Call):
            continue
        nombre = ast.unparse(llamada.func)
        if nombre.endswith(("Lock", "Semaphore", "BoundedSemaphore")):
            globales_con_lock += [ast.unparse(t) for t in nodo.targets]

    assert globales_con_lock == [], (
        "jacobs/routes.py volvió a tener un candado global en el camino de creación: "
        f"{globales_con_lock}. El cupo lo hace cumplir la base (jacobs/cupo.py)."
    )


def test_el_nombre_del_candado_viejo_no_reaparece():
    """Sobre el CÓDIGO, no sobre el texto: el comentario que cuenta por qué se
    fue nombra al candado a propósito, y una historia escrita no es un candado."""
    codigo = ast.unparse(_arbol())
    assert "_pipeline_create_lock" not in codigo


def test_create_pipeline_no_toma_ningun_candado():
    fuente = inspect.getsource(routes.create_pipeline)
    assert "async with _" not in fuente or "lock" not in fuente.lower(), fuente[:400]
    assert ".acquire()" not in fuente


def test_el_cupo_lo_decide_la_base_por_filas_afectadas():
    """La sentencia que decide es UNA sola: si alguien la parte en un SELECT
    y después un INSERT, vuelve la lectura optimista que el candado tapaba."""
    sql = " ".join(cupo.SQL_RESERVAR.split()).upper()
    assert sql.startswith("INSERT INTO JACOBS_PIPELINES")
    assert "SELECT COUNT(*)" in sql
    assert "WHERE" in sql


def test_policy_ya_no_decide_el_cupo_con_un_conteo_recibido():
    """`validate_create` no vuelve a recibir `active_count`: el conteo que se
    lee en una consulta y se decide en otra es justo la carrera que el candado
    existía para tapar."""
    from jacobs import policy

    assert "active_count" not in inspect.signature(policy.validate_create).parameters


@pytest.mark.parametrize("nombre", ["reservar_cupo", "soltar_reserva", "completar_reserva"])
def test_cupo_expone_el_contrato_completo(nombre):
    assert callable(getattr(cupo, nombre))


# ---------------------------------------------------------------------------
#  Qué estados ocupan cupo: el contrato que tiene que ponerse rojo solo
# ---------------------------------------------------------------------------
# Pedido del coordinador (2026-09-17): que el próximo que agregue un estado se
# entere por un rojo, no por un cupo mal contado en producción. El frente G ya
# trae `queued`, `awaiting_approval` y `waiting_children`.

def test_todo_estado_de_pipeline_esta_clasificado():
    """Partición EXHAUSTIVA de `PipelineStatus`. Un estado nuevo sin clasificar
    es un cupo mal contado: si es un estado vivo y queda afuera, entran más
    pipelines de los permitidos y nadie se entera."""
    from jacobs.models import PipelineStatus

    clasificados = set(cupo.ESTADOS_QUE_OCUPAN_CUPO) | set(cupo.ESTADOS_SIN_CUPO)
    sin_clasificar = set(PipelineStatus) - clasificados
    assert sin_clasificar == set(), (
        f"estados de pipeline sin clasificar en jacobs/cupo.py: "
        f"{sorted(e.value for e in sin_clasificar)}. Decidí si ocupan cupo y "
        "agregalos a ESTADOS_QUE_OCUPAN_CUPO o a ESTADOS_SIN_CUPO."
    )
    assert not (set(cupo.ESTADOS_QUE_OCUPAN_CUPO) & set(cupo.ESTADOS_SIN_CUPO))


def test_la_sentencia_cuenta_exactamente_los_estados_declarados():
    """La lista de la sentencia no puede escribirse a mano aparte del contrato:
    se genera de él, y esto lo comprueba sobre el SQL que de verdad corre."""
    sql = cupo.SQL_RESERVAR
    for estado in cupo.ESTADOS_QUE_OCUPAN_CUPO:
        assert f"'{estado.value}'" in sql, estado
    for estado in cupo.ESTADOS_SIN_CUPO:
        assert f"'{estado.value}'" not in sql, estado


def test_no_hay_una_segunda_copia_del_criterio():
    """El recuento de `store` y las tres escrituras condicionadas tienen que
    contar los MISMOS estados. Se comprueba sobre el SQL que de verdad corre, no
    sobre la intención: si alguien vuelve a escribir la lista a mano en uno de
    los cuatro lugares, esto se pone rojo."""
    from jacobs import store

    esperado = {e.value for e in cupo.ESTADOS_QUE_OCUPAN_CUPO}
    sentencias = {
        "contar (store)": store._SQL_CONTAR_ACTIVOS,
        "reservar (cupo)": cupo.SQL_RESERVAR,
        "continuar (store)": store._SQL_PIPELINE_CONTINUAR,
        "tomar época con cupo (store)": store._sql_tomar_epoca(False, 1, True),
    }
    for nombre, sql in sentencias.items():
        presentes = {e.value for e in cupo.ESTADOS_QUE_OCUPAN_CUPO if f"'{e.value}'" in sql}
        assert presentes == esperado, f"{nombre} no cuenta {esperado - presentes}"
        for fuera in cupo.ESTADOS_SIN_CUPO:
            # `aborted`/`expired` aparecen en el WHERE de continuar como estados
            # de ORIGEN, no como parte del recuento: se mira sólo el paréntesis
            # del COUNT.
            conteo = sql[sql.index("COUNT(*)"):sql.index(")", sql.index("status IN (", sql.index("COUNT(*)")))]
            assert f"'{fuera.value}'" not in conteo, f"{nombre} cuenta {fuera.value}"


def test_las_tres_escrituras_que_ocupan_cupo_llevan_la_condicion():
    """Crear INSERTA, continuar/resume/approve REVIVEN. Las dos formas tienen
    que llevar la condición: el INSERT condicionado solo no cubre a las que
    reviven una fila que ya existe, y ese fue el hallazgo que obligó a ampliar
    el alcance (2026-09-17)."""
    from jacobs import store

    assert "COUNT(*)" in cupo.SQL_RESERVAR
    assert "cupo_x.c <" in store._SQL_PIPELINE_CONTINUAR
    assert "cupo_x.c <" in store._sql_tomar_epoca(False, 1, True)
    # …y sin pedir cupo, la sentencia de la época queda como estaba: el
    # llamador DECLARA que su escritura ocupa cupo.
    assert "cupo_x" not in store._sql_tomar_epoca(False, 1, False)
