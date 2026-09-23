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
    maquinas=(A.Maquina("hall9000", "192.0.2.5", 58291),),
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
                           "contexto": ["l"]}],
         "maquinas": [{"nombre": "h", "ip": "192.0.2.5", "puerto": 58291}]}
    lote = A.lote_desde_dict(d)
    assert lote.pasos[0] == A.Paso(1, "Bash", {"command": "ls"}, None) and lote.afirmaciones[0].contexto == ("l",)


def test_el_lote_lleva_las_maquinas_de_la_mision_con_su_direccion():
    """Visto en real (2026-09-17 11:23, misión de Fernando «deime el hostname de esta maquina»): el auditor
    recibía sólo el texto, vio `ssh … axioma@192.168.122.50 hostname` y lo juzgó «tocar otra máquina»:
    pausa `fuera_de_mision` sobre el único paso legítimo. Sin saber qué máquinas eligió la misión, «esta
    máquina» no se puede juzgar."""
    maquinas = (A.Maquina("ejecutor-prueba", "192.168.122.50", 58291),)
    lote = A.Lote(LOTE.mision, LOTE.pasos, LOTE.afirmaciones, maquinas)
    cuerpo = json.loads(A.mensajes(lote, "I")[1]["content"])
    assert cuerpo["maquinas_de_la_mision"] == [{"nombre": "ejecutor-prueba", "ip": "192.168.122.50", "puerto": 58291}]


def test_maquinas_de_filtra_el_inventario_por_las_de_la_mision_en_orden_estable():
    from jax.ejecutor.contratos.destinos import Host
    inventario = (Host("hall9000", "172.16.20.5", 58291, "hypervisor", True),
                  Host("ejecutor-prueba", "192.168.122.50", 58291, "desarrollo", False),
                  Host("atemai", "172.16.20.11", 58291, "desarrollo", False))
    assert A.maquinas_de(inventario, frozenset({"atemai", "ejecutor-prueba"})) == (
        A.Maquina("atemai", "172.16.20.11", 58291), A.Maquina("ejecutor-prueba", "192.168.122.50", 58291))


def test_una_maquina_de_la_mision_que_no_esta_en_el_inventario_no_se_inventa():
    with pytest.raises(ValueError, match="maquina_fuera_del_inventario"):
        A.maquinas_de((), frozenset({"ejecutor-prueba"}))


def test_lote_desde_dict_lee_las_maquinas():
    d = {"mision": "m", "pasos": [], "afirmaciones": [],
         "maquinas": [{"nombre": "ejecutor-prueba", "ip": "192.168.122.50", "puerto": 58291}]}
    assert A.lote_desde_dict(d).maquinas == (A.Maquina("ejecutor-prueba", "192.168.122.50", 58291),)


def test_las_instrucciones_dicen_que_la_maquina_de_la_mision_es_la_elegida():
    from pathlib import Path
    texto = (Path(A.__file__).parent / "auditor_instrucciones.md").read_text()
    assert "maquinas_de_la_mision" in texto


def test_las_instrucciones_mencionan_que_un_paso_puede_ser_una_skill():
    """Minor (auditoría adversarial 2026-09-22): mision.py cuenta las llamadas a Skill
    como pasos (cerebros.toml `skills`) -- el auditor tiene que saber que se va a
    encontrar con ellas, no sólo con comandos de Bash."""
    from pathlib import Path
    texto = (Path(A.__file__).parent / "auditor_instrucciones.md").read_text()
    assert "Skill" in texto


@pytest.mark.parametrize("maquinas", [(), ("hall9000",)])
def test_un_lote_sin_maquinas_no_se_construye(maquinas):
    """Con la lista vacía, las instrucciones hacen que TODO paso parezca salirse (visto en los canarios
    de C5 al agregar el campo: c5_vivo=false por falsos positivos). Fail-closed al construir."""
    with pytest.raises(ValueError, match="lote_sin_maquinas"):
        A.Lote("m", (), (), maquinas)
