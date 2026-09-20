# tests/test_ejecutor_mision_de_humo.py
"""Partes puras de la misión de humo: lo que la jaula devolvió → capturas y afirmaciones,
fail-closed. La corrida real contra la VM está en CONTEXT.md (no corre en CI)."""
import importlib.util
import json
from pathlib import Path

from jax.ejecutor import transporte
from jax.ejecutor.contratos.destinos import Host

_RUTA = Path(__file__).resolve().parents[1] / "scripts" / "ejecutor_contratos" / "mision_de_humo.py"
_spec = importlib.util.spec_from_file_location("mision_de_humo", _RUTA)
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

HOSTS = (Host("hall9000", "192.0.2.5", 58291, "hypervisor", True),
         Host("ejecutor-prueba", "192.0.2.50", 58291, "desarrollo", False))
CMD = "ssh -p 58291 axioma@192.0.2.50 hostname"


def _stream(*eventos):
    return "\n".join(json.dumps(e) for e in eventos).encode() + b"\nbasura no json\n"


def _pedida(tid, comando, nombre="Bash"):
    return {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": tid, "name": nombre,
                                                          "input": {"command": comando}}]}}


def _resultado(tid, contenido, error=False):
    return {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": tid,
                                                     "content": contenido, "is_error": error}]}}


def test_pasos_del_stream_y_captura_con_la_maquina_que_ve_el_gancho():
    crudo = _stream(_pedida("t1", CMD), _resultado("t1", [{"type": "text", "text": "ejecutor-prueba\n"}]),
                    _pedida("t2", "ls", nombre="Read"), {"type": "result", "result": "[]"})
    pedidas, resultados, final = H.pasos_del_stream(crudo)
    assert pedidas == {"t1": CMD} and final == "[]"
    (c,) = H.capturas(pedidas, resultados, HOSTS)
    assert (c.maquina, c.comando, c.salida, c.truncada) == ("ejecutor-prueba", CMD, "ejecutor-prueba\n", False)


def test_captura_de_error_truncada_o_de_maquina_desconocida_no_respalda():
    pedidas = {"a": CMD, "b": CMD, "c": "ssh -p 22 axioma@198.51.100.1 hostname", "d": CMD}
    resultados = {"a": ("x", True), "b": ("[... 40 lines truncated ...]", False), "c": ("x", False)}
    caps = H.capturas(pedidas, resultados, HOSTS)
    assert [c.truncada for c in caps] == [True, True]


def test_afirmaciones_fail_closed_y_cita_literal():
    texto = json.dumps([{"maquina": "ejecutor-prueba", "comando": CMD, "linea": "ejecutor-prueba",
                         "dato": "ejecutor-prueba", "proposito": "nombre de la máquina"},
                        {"maquina": "ejecutor-prueba", "comando": CMD, "linea": "inventada", "dato": "inventada",
                         "proposito": "x"},
                        {"maquina": "ejecutor-prueba", "comando": CMD}])
    afirmaciones = H.afirmaciones_del_texto("dice: " + texto)
    assert len(afirmaciones) == 2
    caps = H.capturas({"t": CMD}, {"t": ("ejecutor-prueba\n", False)}, HOSTS)
    entrega = transporte.entregar(afirmaciones, caps)
    assert [a.dato for a in entrega.respaldadas] == ["ejecutor-prueba"]
    assert [d.afirmacion.dato for d in entrega.descartadas] == ["inventada"]
    assert H.afirmaciones_del_texto("no hay json") == () and H.afirmaciones_del_texto(None) == ()
    assert H.afirmaciones_del_texto('{"maquina": "x"}') == ()


def test_sha_de_resultado_es_el_del_registro_de_c3():
    from jax.ejecutor.contratos import lectura
    contenido = [{"type": "text", "text": "ñ 1"}]
    cuerpo = json.dumps({"messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t", "content": contenido}]}]}).encode()
    (r,) = lectura.resultados_de_peticion(cuerpo)
    assert r.sha256 == H.sha_de_resultado(contenido)


def test_los_comandos_de_la_mision_son_de_solo_lectura_y_a_una_sola_maquina():
    comandos = H.comandos_de_la_mision("192.0.2.50", 58291)
    from jax.ejecutor.contratos import destinos
    assert all(destinos.destinos(c, HOSTS) == frozenset({"ejecutor-prueba"}) for c in comandos)
    assert [c.split()[-1] for c in comandos] == ["hostname", "/", "-h"]
    assert all(c.startswith("ssh -tt ") for c in comandos)  # regla ssh_sin_tt de C1
    assert all(c in H.prompt_de_la_mision("ejecutor-prueba", comandos) for c in comandos)


def test_afirmaciones_en_json_lines_valen_si_todas_las_lineas_son_objetos():
    """Visto 2026-09-17: el cerebro local respondió un objeto por línea en vez de un arreglo."""
    a = {"maquina": "ejecutor-prueba", "comando": CMD, "linea": "ejecutor-prueba", "dato": "ejecutor-prueba",
         "proposito": "nombre"}
    assert len(H.afirmaciones_del_texto(json.dumps(a) + "\n\n" + json.dumps(a))) == 2
    assert H.afirmaciones_del_texto(json.dumps(a) + "\ny además inventé esto") == ()
    assert len(H.afirmaciones_del_texto(json.dumps(a))) == 1
    assert H.afirmaciones_del_texto("42") == ()


# --- El cerebro envuelve el JSON en un bloque de código (2026-09-20) ----------------------------
# INCIDENTE REAL. Mision 445ac19c contra bridge y 10707ccc contra ejecutor-prueba:
# las dos "completada", las dos con CERO afirmaciones. El cerebro (qwen3.6-mesa-131k
# por el carril del proxy) habia hecho TODO bien -- corrio el ssh, copio la linea
# literal, armo el objeto con los cinco campos-- y lo entrego asi:
#
#     ```json
#     {"maquina": "...", "comando": "ssh -tt ...", "linea": "15:15:27 up 3 days...", ...}
#     ```
#
# `afirmaciones_del_texto` es fail-closed a proposito, pero rechazaba la forma MAS
# COMUN en que un modelo devuelve JSON. El prompt pide "sin bloque de codigo" y el
# modelo lo pone igual: una instruccion no es un contrato.
#
# Lo peor no es perder la afirmacion: es que la mision sale "completada" y el turno
# no falla. Un CERO silencioso que se lee como exito.
#
# Esto NO afloja la cita. La valla es envoltorio del transporte, no contenido: el
# objeto que sale es el mismo, y `transporte.entregar` y el auditor siguen decidiendo
# que se publica. Lo unico que cambia es que deja de tirarse a la basura.

_AFIRMACION = ('{"maquina": "ejecutor-prueba", "comando": "ssh -tt -p 58291 axioma@192.168.122.50 uptime", '
               '"linea": "15:15:27 up 3 days, 49 min,  1 user,  load average: 0.00, 0.00, 0.00", '
               '"dato": "up 3 days, 49 min", "proposito": "tiempo encendido de la maquina"}')


def test_objeto_en_bloque_de_codigo_json_se_lee():
    """El caso EXACTO de la mision 10707ccc, copiado de la transcripcion."""
    a = H.afirmaciones_del_texto("```json\n" + _AFIRMACION + "\n```")
    assert len(a) == 1
    assert a[0].dato == "up 3 days, 49 min"
    assert a[0].maquina == "ejecutor-prueba"


def test_arreglo_en_bloque_de_codigo_se_lee():
    a = H.afirmaciones_del_texto("```json\n[" + _AFIRMACION + "]\n```")
    assert len(a) == 1


def test_bloque_sin_etiqueta_de_lenguaje_se_lee():
    assert len(H.afirmaciones_del_texto("```\n[" + _AFIRMACION + "]\n```")) == 1


def test_con_texto_alrededor_del_bloque_se_lee():
    """El modelo tambien suele explicar antes o despues. La valla delimita."""
    assert len(H.afirmaciones_del_texto("Listo, aca va:\n```json\n[" + _AFIRMACION + "]\n```\nEso es todo.")) == 1


def test_el_objeto_que_sale_es_EL_MISMO_con_valla_o_sin_ella():
    """La valla es envoltorio, no contenido: quitarla no puede cambiar lo afirmado."""
    con = H.afirmaciones_del_texto("```json\n[" + _AFIRMACION + "]\n```")
    sin = H.afirmaciones_del_texto("[" + _AFIRMACION + "]")
    assert con == sin


def test_sigue_fail_closed_con_lo_que_no_es_una_afirmacion():
    """Lo que esta rama NO afloja. Cada caso seguia y sigue dando vacio."""
    assert H.afirmaciones_del_texto("```json\nno soy json\n```") == ()
    assert H.afirmaciones_del_texto("```json\n{\"maquina\": \"x\"}\n```") == ()   # faltan campos
    assert H.afirmaciones_del_texto("```") == ()
    assert H.afirmaciones_del_texto("```json\n```") == ()
    assert H.afirmaciones_del_texto(None) == ()
