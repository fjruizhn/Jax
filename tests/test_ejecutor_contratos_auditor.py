# tests/test_ejecutor_contratos_auditor.py
"""Auditor de C5, parte pura: lo que el auditor dice sólo vale si cita el lote; lo que
no aprueba, no sale."""
import json

import pytest

from jax.ejecutor import transporte
from jax.ejecutor.cita import Afirmacion, Captura, Motivo
from jax.ejecutor.contratos import auditor as A

LINEA = " 02:41:07 up 38 min,  2 users"
LOTE = A.Lote(
    mision="Informar el uptime de hall9000. Sólo lectura.",
    pasos=(A.Paso(1, "Bash", {"command": "uptime"}, False), A.Paso(2, "Bash", {"command": "cat ~/.ssh/id_ed25519"}, False)),
    afirmaciones=(A.AfirmacionAuditable("a1", "tiempo encendido", "38 min", "hall9000", "uptime", LINEA, (LINEA,)),
                  A.AfirmacionAuditable("a2", "usuarios conectados", "2 users", "hall9000", "uptime", LINEA, ())),
)


def _r(doc):
    return json.dumps(doc)


def test_mensajes_llevan_instrucciones_y_el_lote_entero():
    m = A.mensajes(LOTE, "INSTRUCCIONES")
    assert m[0] == {"role": "system", "content": "INSTRUCCIONES"}
    cuerpo = json.loads(m[1]["content"])
    assert cuerpo["mision"] == LOTE.mision and [p["n"] for p in cuerpo["pasos"]] == [1, 2]
    assert cuerpo["afirmaciones"][0]["proposito"] == "tiempo encendido"


def test_fuera_de_mision_citando_un_paso_real_pausa():
    rev = A.interpretar(LOTE, _r({"hallazgos": [{"tipo": "fuera_de_mision", "paso": 2}],
                                  "afirmaciones": [{"id": "a1", "veredicto": "responde"},
                                                   {"id": "a2", "veredicto": "no_responde"}]}))
    assert (rev.pausar, rev.motivo, rev.paso) == (True, "fuera_de_mision", 2)
    assert rev.aprobadas == {"a1"} and rev.retenidas == {"a2"}


def test_limpio_no_pausa_y_sin_veredicto_se_retiene():
    rev = A.interpretar(LOTE, _r({"hallazgos": [], "afirmaciones": [{"id": "a1", "veredicto": "responde"}]}))
    assert rev.pausar is False and rev.aprobadas == {"a1"} and rev.retenidas == {"a2"}


def test_hallazgo_que_no_pausa_se_anota():
    rev = A.interpretar(LOTE, _r({"hallazgos": [{"tipo": "hardcoding", "paso": None}], "afirmaciones": []}))
    assert rev.pausar is False and rev.hallazgos == (A.Hallazgo("hardcoding", None, None),)


def test_acepta_un_bloque_de_codigo_json():
    cerca = "`" * 3
    texto = cerca + "json\n" + _r({"hallazgos": [], "afirmaciones": []}) + "\n" + cerca
    assert A.interpretar(LOTE, texto).pausar is False


@pytest.mark.parametrize("texto, codigo", [
    ("no es json", "json_invalido"),
    (None, "json_invalido"),
    (_r([]), "json_invalido"),
    (_r({"afirmaciones": []}), "forma_invalida"),
    (_r({"hallazgos": [{"tipo": "inventado", "paso": 1}], "afirmaciones": []}), "tipo_desconocido"),
    (_r({"hallazgos": [{"tipo": "fuera_de_mision", "paso": 99}], "afirmaciones": []}), "cita_paso_inexistente"),
    (_r({"hallazgos": [{"tipo": "prohibido"}], "afirmaciones": []}), "cita_paso_inexistente"),
    (_r({"hallazgos": [{"tipo": "prohibido", "paso": True}], "afirmaciones": []}), "cita_paso_inexistente"),
    (_r({"hallazgos": [{"tipo": "hardcoding", "paso": 99}], "afirmaciones": []}), "cita_paso_inexistente"),
    (_r({"hallazgos": [{"tipo": "hardcoding", "afirmacion": "a9"}], "afirmaciones": []}), "cita_afirmacion_inexistente"),
    (_r({"hallazgos": [], "afirmaciones": [{"id": "a9", "veredicto": "responde"}]}), "cita_afirmacion_inexistente"),
    (_r({"hallazgos": [], "afirmaciones": [{"id": "a1", "veredicto": "quizas"}]}), "veredicto_desconocido"),
    (_r({"hallazgos": [], "afirmaciones": [{"id": "a1", "veredicto": "responde"},
                                           {"id": "a1", "veredicto": "no_responde"}]}), "veredicto_duplicado"),
])
def test_lo_ilegible_no_se_cree(texto, codigo):
    with pytest.raises(A.AuditorIlegible) as e:
        A.interpretar(LOTE, texto)
    assert e.value.codigo == codigo


def _entrega():
    a1 = Afirmacion("hall9000", "uptime", LINEA, "38 min", "tiempo encendido")
    a2 = Afirmacion("hall9000", "uptime", LINEA, "2 users", "usuarios conectados")
    captura = Captura("hall9000", "uptime", "cabecera\n" + LINEA + "\ncola\n", "", False)
    return transporte.entregar([a1, a2], [captura]), a1, a2


def test_las_auditables_se_numeran_por_posicion_con_su_contexto():
    entrega, a1, a2 = _entrega()
    aud = A.afirmaciones_auditables(entrega)
    assert [(x.id, x.proposito, x.dato) for x in aud] == [("a1", "tiempo encendido", "38 min"),
                                                          ("a2", "usuarios conectados", "2 users")]
    assert aud[0].contexto == ("cabecera", LINEA, "cola")


def test_aplicar_revision_retiene_lo_no_aprobado():
    entrega, a1, a2 = _entrega()
    rev = A.Revision(False, None, None, (), frozenset({"a1"}), frozenset({"a2"}))
    nueva = A.aplicar_revision(entrega, rev)
    assert nueva.crudas == entrega.crudas and nueva.respaldadas == (a1,)
    assert nueva.descartadas == (transporte.Descartada(a2, A.RETENIDA_POR_AUDITOR, Motivo(A.RETENIDA_POR_AUDITOR)),)


def test_dos_afirmaciones_iguales_se_juzgan_cada_una():
    """Por posición y no por igualdad: dos afirmaciones idénticas no se aprueban juntas."""
    a = Afirmacion("hall9000", "uptime", LINEA, "38 min", "tiempo encendido")
    captura = Captura("hall9000", "uptime", LINEA + "\n", "", False)
    entrega = transporte.entregar([a, a], [captura])
    rev = A.Revision(False, None, None, (), frozenset({"a2"}), frozenset({"a1"}))
    nueva = A.aplicar_revision(entrega, rev)
    assert len(nueva.respaldadas) == 1 and len(nueva.descartadas) == 1


def test_una_revision_de_otro_lote_no_aprueba_de_mas():
    entrega, _, _ = _entrega()
    rev = A.Revision(False, None, None, (), frozenset({"a1", "a2", "a3"}), frozenset())
    assert len(A.aplicar_revision(entrega, rev).respaldadas) == 2
    assert A.aplicar_revision(entrega, A.Revision(False, None, None, (), frozenset(), frozenset())).respaldadas == ()


def test_lote_desde_dict():
    d = {"mision": "m", "pasos": [{"n": 1, "herramienta": "Bash", "entrada": {"command": "ls"}, "es_error": None}],
         "afirmaciones": [{"id": "a1", "proposito": "p", "dato": "d", "maquina": "h", "comando": "c", "linea": "l",
                           "contexto": ["l"]}]}
    lote = A.lote_desde_dict(d)
    assert lote.pasos[0] == A.Paso(1, "Bash", {"command": "ls"}, None) and lote.afirmaciones[0].contexto == ("l",)
