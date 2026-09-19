#!/usr/bin/env python3
"""El veredicto del árbitro (spec 2026-09-18-arbitro-devuelve-design §3.1).

El caso real que motiva esta ronda: el pipeline `e570ac1c-8cae-4423-b397-
d354f30b4328` corrió con árbitro por primera vez. El paso 6 (`thot`) escribió
15 KB de prosa diciendo, en sustancia, "no apruebes los pasos 4-5" -- y ahí se
acabó: la prosa no es algo que el código pueda accionar. Verificado contra la
base real (`jacobs_steps.output_ref` de ese pipeline, 2026-09-18): el texto
de hoy NO trae ningún bloque ```veredicto```, así que `parsear_veredicto`
sobre ESE texto real tiene que dar None -- es el primer caso rojo de esta
ronda, antes de que exista el módulo.

Fallo cerrado (spec §5.2, criterio 2): sin bloque, con JSON roto, con una
clave faltante o con una cita que no coincide con el paso declarado,
`parsear_veredicto` devuelve None -- nunca lanza, nunca adivina a qué paso
se refería el árbitro.

Corre con:
  PYTHONPATH=. python -m pytest jacobs/_veredicto_test.py -v
(o, dentro del REPL/CI de LAS MANOS, con PYTHONPATH=las_manos -- no toca la
base ni la red, mismo criterio que jacobs/_arbitro_test.py.)

En memoria de Jairo Urbina.
"""
from __future__ import annotations

from jacobs.veredicto import (
    DECISION_APROBAR,
    DECISION_DEVOLVER,
    VeredictoArbitro,
    parsear_veredicto,
)

# El pipeline tiene 7 pasos (0..6); el árbitro es el step_index 6. Un
# veredicto de devolución sólo puede apuntar a 0..5.
N_PASOS = 7


def _bloque(cuerpo: str) -> str:
    return f"```veredicto\n{cuerpo}\n```"


# ---------------------------------------------------------------------------
# El caso real, primero (spec §5, criterio 5: rojo antes que nada más).
# ---------------------------------------------------------------------------

def test_la_prosa_real_del_pipeline_e570ac1c_no_trae_veredicto():
    """Recorte real de jacobs_steps.output_ref (`result`) del step 6 (thot) de
    e570ac1c-8cae-4423-b397-d354f30b4328, leído de la base el 2026-09-18. Sin
    bloque ```veredicto```: fallo cerrado, None -- el pipeline de hoy termina
    'completado' con la crítica escrita y nadie la accionó, que es exactamente
    el defecto que esta ronda cierra."""
    prosa_real = (
        "## Decisión única\n\n**Continuar con una evolución incremental de "
        "AteneaERP, tomando el brief del paso 0 como contrato de destino, "
        "pero no aprobar todavía el DDL ni la arquitectura de los pasos "
        "4-5 para implementación.** [paso 0] [paso 1] [paso 4] [paso 5]\n\n"
        "**Motivo:** las propuestas posteriores se construyeron sin recibir "
        "insumos que ahora sí están presentes y contienen incompatibilidades "
        "explícitas con el brief."
    )
    assert parsear_veredicto(prosa_real, N_PASOS) is None


# ---------------------------------------------------------------------------
# Camino feliz.
# ---------------------------------------------------------------------------

def test_aprobar_sin_paso_ni_motivo():
    texto = "Todo en orden. " + _bloque('{"decision": "aprobar"}')
    v = parsear_veredicto(texto, N_PASOS)
    assert v == VeredictoArbitro(decision=DECISION_APROBAR)


def test_devolver_valido_con_paso_motivo_y_cita():
    texto = (
        "Prosa de siempre citando [paso 4]. " + _bloque(
            '{"decision": "devolver", "paso": 4, '
            '"motivo": "usaste BIGSERIAL y TIMESTAMPTZ; el destino es MariaDB", '
            '"cita": "[paso 4]"}'
        )
    )
    v = parsear_veredicto(texto, N_PASOS)
    assert v.decision == DECISION_DEVOLVER
    assert v.paso == 4
    assert v.motivo == "usaste BIGSERIAL y TIMESTAMPTZ; el destino es MariaDB"
    assert v.cita == "[paso 4]"


def test_toma_el_ultimo_bloque_si_hay_mas_de_uno():
    """Si el árbitro explica el formato antes de usarlo (dos bloques), gana
    el ÚLTIMO -- su decisión real, no el ejemplo de la explicación."""
    texto = (
        _bloque('{"decision": "aprobar"}')
        + "\n\nEn realidad, revisando de nuevo:\n\n"
        + _bloque('{"decision": "devolver", "paso": 1, "motivo": "m", "cita": "[paso 1]"}')
    )
    v = parsear_veredicto(texto, N_PASOS)
    assert v.decision == DECISION_DEVOLVER
    assert v.paso == 1


# ---------------------------------------------------------------------------
# Fallo cerrado: cada forma de "no alcanza" da None, nunca una excepción.
# ---------------------------------------------------------------------------

def test_sin_bloque_da_none():
    assert parsear_veredicto("Prosa sin ningún bloque de código.", N_PASOS) is None


def test_texto_vacio_da_none():
    assert parsear_veredicto("", N_PASOS) is None


def test_json_invalido_da_none():
    texto = _bloque('{"decision": "devolver", "paso": 4,')  # truncado
    assert parsear_veredicto(texto, N_PASOS) is None


def test_bloque_no_es_un_objeto_json_da_none():
    texto = _bloque('["devolver", 4]')
    assert parsear_veredicto(texto, N_PASOS) is None


def test_decision_ausente_da_none():
    texto = _bloque('{"paso": 4, "motivo": "m", "cita": "[paso 4]"}')
    assert parsear_veredicto(texto, N_PASOS) is None


def test_decision_desconocida_da_none():
    texto = _bloque('{"decision": "reescribir", "paso": 4}')
    assert parsear_veredicto(texto, N_PASOS) is None


def test_paso_ausente_da_none():
    texto = _bloque('{"decision": "devolver", "motivo": "m", "cita": "[paso 4]"}')
    assert parsear_veredicto(texto, N_PASOS) is None


def test_paso_no_entero_da_none():
    texto = _bloque('{"decision": "devolver", "paso": "4", "motivo": "m", "cita": "[paso 4]"}')
    assert parsear_veredicto(texto, N_PASOS) is None


def test_paso_booleano_no_cuenta_como_entero():
    """bool es subclase de int en Python -- True/False no son un índice de
    paso válido, y aceptarlos silenciosamente sería un defecto sutil."""
    texto = _bloque('{"decision": "devolver", "paso": true, "motivo": "m", "cita": "[paso 4]"}')
    assert parsear_veredicto(texto, N_PASOS) is None


def test_paso_negativo_da_none():
    texto = _bloque('{"decision": "devolver", "paso": -1, "motivo": "m", "cita": "[paso -1]"}')
    assert parsear_veredicto(texto, N_PASOS) is None


def test_paso_apuntando_al_arbitro_mismo_da_none():
    """El árbitro es SIEMPRE el último paso (PlanBuilder._con_arbitro): no
    puede devolverse a sí mismo. N_PASOS=7 -> índices de productor 0..5;
    6 es el árbitro."""
    texto = _bloque('{"decision": "devolver", "paso": 6, "motivo": "m", "cita": "[paso 6]"}')
    assert parsear_veredicto(texto, N_PASOS) is None


def test_paso_fuera_de_rango_da_none():
    texto = _bloque('{"decision": "devolver", "paso": 99, "motivo": "m", "cita": "[paso 99]"}')
    assert parsear_veredicto(texto, N_PASOS) is None


def test_motivo_ausente_da_none():
    texto = _bloque('{"decision": "devolver", "paso": 4, "cita": "[paso 4]"}')
    assert parsear_veredicto(texto, N_PASOS) is None


def test_motivo_vacio_da_none():
    texto = _bloque('{"decision": "devolver", "paso": 4, "motivo": "   ", "cita": "[paso 4]"}')
    assert parsear_veredicto(texto, N_PASOS) is None


def test_cita_ausente_da_none():
    texto = _bloque('{"decision": "devolver", "paso": 4, "motivo": "m"}')
    assert parsear_veredicto(texto, N_PASOS) is None


def test_cita_que_no_coincide_con_el_paso_da_none():
    """La cita tiene que sostener EXACTAMENTE el paso que dice -- el mismo
    criterio de citación que el resto de las afirmaciones del árbitro
    (spec §3.1: 'la cita del paso que lo sostiene, igual que el resto de
    sus afirmaciones'). Citar otro paso no vale como evidencia de este."""
    texto = _bloque('{"decision": "devolver", "paso": 4, "motivo": "m", "cita": "[paso 1]"}')
    assert parsear_veredicto(texto, N_PASOS) is None


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
