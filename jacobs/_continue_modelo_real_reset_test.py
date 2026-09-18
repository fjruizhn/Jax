"""ANTES DE MERGEAR 6 (revisión final 2026-09-18, tanda historial-y-arreglos-de-pipeline):
`continue` deja ver el `modelo_real` de la corrida anterior.

EL DEFECTO. `_SQL_PASO_A_CORRER` (jacobs/store.py, usada por
`continuar_transaccion` para revivir los pasos que `/continue` vuelve a
poner en `pending`) resetea `facet`, `motor`, `status`, `output_ref`,
`started_at`, `finished_at` y `error` -- pero NO `modelo_real`. Un step
`pending` recién continuado sigue mostrando el modelo de SU corrida
anterior, como si fuera un dato definitivo, hasta que el próximo despacho
lo pisa. Es el único punto donde la función que esta misma ronda entregó
(Task 1: "el paso guarda qué modelo lo ejecutó de verdad") muestra un dato
FALSO con cara de verdadero -- el resto de las columnas de estado ya se
resetean; a ésta se la había dejado afuera, documentada como diferida.

Test puro de texto de SQL (sin DB): el defecto ES que falta una cláusula en
la sentencia -- inspeccionar el SQL es la verificación precisa para este
caso, mismo criterio que ya usa el repo cuando el bug está en la FORMA de
la consulta, no en su resultado (ver
`feedback-explain-igual-no-valida-el-cambio` en la memoria del proyecto:
"el test tiene que mirar el SQL").

Corre con:
  cd /home/fruiz/worktrees/jax-historial && PYTHONPATH=.:las_manos python -m pytest -v jacobs/_continue_modelo_real_reset_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import re

from jacobs import store


def test_continuar_resetea_modelo_real():
    """La UPDATE que revive un step al continuar tiene que resetear
    `modelo_real` igual que el resto de las columnas de estado -- si no, un
    step 'pending' recién continuado muestra el modelo de la corrida
    anterior como si fuera de la corrida actual."""
    assert re.search(r"\bmodelo_real\s*=\s*NULL\b", store._SQL_PASO_A_CORRER, re.IGNORECASE), (
        f"_SQL_PASO_A_CORRER no resetea modelo_real: {store._SQL_PASO_A_CORRER!r}"
    )


def test_continuar_sigue_reseteando_el_resto_de_las_columnas_de_estado():
    """Guarda de no-regresión: el arreglo de modelo_real no puede pisar (ni
    accidentalmente sacar) ninguna de las columnas que YA se reseteaban."""
    for columna in ("status='pending'", "output_ref=NULL", "started_at=NULL",
                     "finished_at=NULL", "error=NULL"):
        assert columna in store._SQL_PASO_A_CORRER, store._SQL_PASO_A_CORRER


def test_continuar_sigue_actualizando_facet_y_motor_por_parametro():
    """`facet=%s, motor=%s` siguen viniendo de los parámetros (el reroute de
    STEP_REROUTED los cambia) -- el reset de modelo_real no puede convertir
    esto en un valor fijo."""
    assert "facet=%s" in store._SQL_PASO_A_CORRER
    assert "motor=%s" in store._SQL_PASO_A_CORRER


def test_el_where_sigue_acotado_a_un_solo_step_del_pipeline():
    """Guarda de no-regresión: agregar la cláusula nueva no puede ensanchar
    (ni angostar) el WHERE -- sigue siendo un step puntual de un pipeline
    puntual."""
    assert "WHERE step_id=%s AND pipeline_id=%s" in store._SQL_PASO_A_CORRER
