# tests/test_ejecutor_mision.py
"""SP2 del Ejecutor: un turno de misión por el camino gobernado, lanzado desde la plataforma.

`jax.ejecutor.mision` generaliza la misión de humo (scripts/ejecutor_contratos/mision_de_humo.py)
a un objetivo libre con sesión del arnés que se retoma por id. Acá se prueba el orquestador con
dependencias falsas: la corrida real contra la VM desechable está en CONTEXT.md."""
import asyncio
import hashlib
import json
import uuid

import pytest

from jax.ejecutor import mision as M
from jax.ejecutor.contratos.arranque import ContratosNoVerificados
from jax.ejecutor.contratos.auditor import Revision
from jax.ejecutor.contratos.destinos import Host
from jax.ejecutor.contratos.fallo import Fallo

HOSTS = (Host("hall9000", "192.0.2.5", 58291, "hypervisor", True),
         Host("ejecutor-prueba", "192.0.2.50", 58291, "desarrollo", False))
CMD = "ssh -tt -p 58291 axioma@192.0.2.50 free -h"
LINEA = "Mem:           1.9Gi       180Mi       1.5Gi"
MISION = str(uuid.uuid4())
SESION = str(uuid.uuid4())


def _turno(**cambios):
    doc = {"mision_id": MISION, "n": 1, "sesion": SESION, "objetivo": "memoria de la VM",
           "instruccion": "memoria de la VM", "hosts": ["ejecutor-prueba"]}
    doc.update(cambios)
    return json.dumps(doc).encode()


# --- el pedido del turno (lo escribe la plataforma por stdin) -------------------------------

def test_turno_legible():
    t = M.turno_desde_json(_turno())
    assert (t.mision_id, t.n, t.sesion, t.reanudar, t.hosts) == (MISION, 1, SESION, False, frozenset({"ejecutor-prueba"}))
    assert M.turno_desde_json(_turno(n=2)).reanudar is True


@pytest.mark.parametrize("datos, codigo", [
    (b"no json", "turno_no_es_json"), (b"[]", "turno_no_es_objeto"),
    (_turno(mision_id="x"), "turno_mision_invalida"), (_turno(n=0), "turno_numero_invalido"),
    (_turno(n=True), "turno_numero_invalido"), (_turno(sesion="../../etc"), "turno_sesion_invalida"),
    (_turno(sesion=SESION.upper()), "turno_sesion_invalida"), (_turno(objetivo="  "), "turno_sin_objetivo"),
    (_turno(instruccion=""), "turno_sin_instruccion"), (_turno(hosts=[]), "turno_sin_maquinas"),
    (_turno(hosts=["a", 3]), "turno_sin_maquinas"),
])
def test_turno_ilegible_con_codigo(datos, codigo):
    with pytest.raises(M.TurnoIlegible) as exc:
        M.turno_desde_json(datos)
    assert exc.value.args[0] == codigo


def test_el_prompt_nombra_solo_las_maquinas_de_la_mision_con_ssh_tt_y_pide_pares():
    p = M.prompt_del_turno("memoria de la VM", HOSTS, frozenset({"ejecutor-prueba"}))
    assert "ssh -tt -p 58291 axioma@192.0.2.50" in p and "192.0.2.5 " not in p and "hall9000" not in p
    assert all(c in p for c in ('"maquina"', '"comando"', '"linea"', '"dato"', '"proposito"'))
    assert "memoria de la VM" in p
    # Visto en la primera misión real desde la plataforma (2026-09-17): el cerebro citó `hostname` en vez del
    # comando que corrió (`ssh -tt … hostname`) y las tres afirmaciones cayeron por comando_no_corrido. El
    # verificador tenía razón; lo que se ajusta es el pedido (riesgo 1 del spec de Fase 2), no la cita.
    assert "COMPLETO tal como lo pasaste a Bash" in p


def test_en_un_turno_retomado_el_prompt_exige_correr_de_nuevo():
    """Visto en real (2026-09-17, turno 3 retomado): el cerebro respondió con el uptime del turno 2 sin
    correr nada. La cita lo tiró (comando_no_corrido): un dato viejo presentado como actual. El pedido
    tiene que decirlo; la regla la sigue haciendo cumplir la cita."""
    nuevo = M.prompt_del_turno("uptime", HOSTS, frozenset({"ejecutor-prueba"}))
    retomado = M.prompt_del_turno("uptime", HOSTS, frozenset({"ejecutor-prueba"}), reanudar=True)
    assert "turnos anteriores" not in nuevo
    assert "turnos anteriores ya no valen" in retomado and "EN ESTE TURNO" in retomado


# --- lo que sale hacia la plataforma --------------------------------------------------------

def test_evento_es_una_linea_json_con_codigo_turno_y_datos():
    linea = M.evento("paso", 2, comando="ls", en_registro=True)
    assert json.loads(linea) == {"evento": "paso", "turno": 2, "datos": {"comando": "ls", "en_registro": True}}
    assert "\n" not in linea


# --- el orquestador, con dependencias falsas ------------------------------------------------

def _stream(*eventos):
    return "\n".join(json.dumps(e) for e in eventos).encode()


def _crudo(final, *, sesion=SESION, comando=CMD, salida=LINEA + "\n"):
    return _stream({"type": "system", "subtype": "init", "session_id": sesion},
                   {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Bash",
                                                                  "input": {"command": comando}}]}},
                   {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1",
                                                             "content": salida, "is_error": False}]}},
                   {"type": "result", "result": final})


AFIRMACION = {"maquina": "ejecutor-prueba", "comando": CMD, "linea": LINEA, "dato": "1.9Gi",
              "proposito": "memoria total"}


def _sha(contenido):
    return hashlib.sha256(json.dumps(contenido, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


class Falsas:
    def __init__(self):
        self.fallos = ()
        self.latido = True
        self.vigia_vive = True
        self.vigia_cierre = (0, "arranco=true cerrada=true", "")
        self.cerebro = (0, _crudo(json.dumps([AFIRMACION])))
        self.registro = None  # por defecto: todo anotado y con sha que cuadra
        self.revision = None
        self.auditor_revienta = False
        self.cadena = True
        self.pausa_leida = None
        self.llamadas = []

    async def contexto(self):
        return "ctx"

    async def hosts(self, ctx):
        return HOSTS

    async def exigir(self, ctx):
        self.llamadas.append("exigir")
        if self.fallos:
            raise ContratosNoVerificados(self.fallos)

    async def tamano_registro(self, ctx):
        return 10

    async def abrir_vigia(self, ctx, id_vigia, texto, hosts):
        self.llamadas.append(("vigia", id_vigia, hosts, texto))
        falsas = self

        class V:
            def vive(self):
                return falsas.vigia_vive

            async def cerrar(self):
                falsas.llamadas.append("cerrar_vigia")
                return falsas.vigia_cierre
        return V()

    async def latido_fresco(self, ctx):
        return self.latido

    async def correr_cerebro(self, ctx, prompt, sesion, reanudar):
        self.llamadas.append(("cerebro", sesion, reanudar))
        return self.cerebro

    async def eventos_desde(self, ctx, desde):
        if self.registro is not None:
            return self.registro
        return [{"evento": "herramienta_pedida", "tool_use_id": "t1"},
                {"evento": "resultado_devuelto", "tool_use_id": "t1", "sha256": _sha(LINEA + "\n")}]

    async def auditar(self, texto, entrega, maquinas):
        self.maquinas_auditadas = maquinas
        if self.auditor_revienta:
            raise ValueError("json_invalido")
        if self.revision is not None:
            return self.revision
        ids = frozenset(f"a{i + 1}" for i in range(len(entrega.respaldadas)))
        return Revision(False, None, None, (), ids, frozenset())

    async def cadena_ok(self, ctx):
        return self.cadena

    async def leer_pausa(self, ctx):
        return self.pausa_leida

    def deps(self):
        return M.Dependencias(contexto=self.contexto, hosts=self.hosts, exigir=self.exigir,
                              tamano_registro=self.tamano_registro, abrir_vigia=self.abrir_vigia,
                              latido_fresco=self.latido_fresco, correr_cerebro=self.correr_cerebro,
                              eventos_desde=self.eventos_desde, auditar=self.auditar, cadena_ok=self.cadena_ok,
                              leer_pausa=self.leer_pausa, espera_latido_s=0.2, paso_espera_s=0.01)


def _correr(falsas, datos=None):
    eventos = []
    resultado = asyncio.run(M.correr_turno(M.turno_desde_json(datos or _turno()), falsas.deps(),
                                           lambda linea: eventos.append(json.loads(linea))))
    return resultado, eventos


def _codigos(eventos):
    return [e["evento"] for e in eventos]


def test_turno_completo_entrega_el_par_con_la_linea_literal_y_las_crudas():
    f = Falsas()
    r, eventos = _correr(f)
    assert r["estado"] == "completado" and r["codigo"] is None and r["sesion_iniciada"] is True
    assert r["afirmaciones"] == [{**AFIRMACION, "linea": LINEA}]
    assert r["crudas"] == [{"maquina": "ejecutor-prueba", "comando": CMD, "salida": LINEA + "\n", "truncada": False}]
    assert r["verificacion"] == {"registro_cuadra": True, "cadena_ok": True, "pausa_puesta": False,
                                 "auditor_pauso": False, "auditor_legible": True}
    assert _codigos(eventos) == ["turno_lanzado", "arranque_verificado", "vigia_late", "cerebro_termino", "paso",
                                 "afirmacion_entregada", "vigia_cerrado", "turno_completado"]
    assert ("cerebro", SESION, False) in f.llamadas and f.llamadas[-1] == "cerrar_vigia"
    assert ("vigia", f"{MISION}-t1", frozenset({"ejecutor-prueba"}), "memoria de la VM") in f.llamadas


def test_el_turno_siguiente_retoma_la_sesion():
    f = Falsas()
    r, _ = _correr(f, _turno(n=3, instruccion="y el disco"))
    assert ("cerebro", SESION, True) in f.llamadas and r["estado"] == "completado"


def test_el_vigia_y_el_auditor_juzgan_el_turno_con_su_instruccion():
    """Visto en la misión real desde la plataforma (2026-09-17): el turno 2 pidió uptime y kernel y el
    vigía recibió sólo el objetivo del turno 1 (hostname, df, free): C5 marcó `fuera_de_mision`, puso la
    pausa del Ejecutor y el auditor retuvo las dos afirmaciones. El contrato hizo lo correcto con lo que
    le dieron; lo que estaba mal era lo que le dieron."""
    f = Falsas()
    textos = []
    auditar = f.auditar

    async def anota(texto, entrega, maquinas):
        textos.append(texto)
        return await auditar(texto, entrega, maquinas)
    f.auditar = anota
    _correr(f, _turno(n=2, objetivo="memoria de la VM", instruccion="ahora el kernel"))
    (vigia,) = [x for x in f.llamadas if isinstance(x, tuple) and x[0] == "vigia"]
    assert "memoria de la VM" in vigia[3] and "ahora el kernel" in vigia[3]
    assert textos == [vigia[3]]
    f = Falsas()
    _correr(f, _turno(n=1))
    (vigia,) = [x for x in f.llamadas if isinstance(x, tuple) and x[0] == "vigia"]
    assert vigia[3] == "memoria de la VM"


def test_arranque_rechazado_devuelve_los_codigos_y_no_abre_nada():
    f = Falsas()
    f.fallos = (Fallo("c5", "auditor_no_admite_datos_de_clientes", (("host", "atemai"),)),
                Fallo("arranque", "maquina_sin_contratos_remotos", (("host", "atemai"),)))
    r, eventos = _correr(f)
    assert r["estado"] == "rechazado" and r["codigo"] == "arranque_rechazado"
    assert r["rechazo"] == [{"contrato": "c5", "codigo": "auditor_no_admite_datos_de_clientes",
                             "datos": {"host": "atemai"}},
                            {"contrato": "arranque", "codigo": "maquina_sin_contratos_remotos",
                             "datos": {"host": "atemai"}}]
    assert _codigos(eventos) == ["turno_lanzado", "arranque_rechazado", "turno_rechazado"]
    assert not any(isinstance(x, tuple) for x in f.llamadas)  # ni vigía ni cerebro
    assert r["crudas"] == [] and r["afirmaciones"] == []


def test_vigia_que_no_late_falla_sin_lanzar_el_cerebro_y_lo_cierra():
    f = Falsas()
    f.latido = False
    r, eventos = _correr(f)
    assert (r["estado"], r["codigo"]) == ("fallido", "vigia_no_latio")
    assert not any(isinstance(x, tuple) and x[0] == "cerebro" for x in f.llamadas)
    assert "cerrar_vigia" in f.llamadas and "vigia_no_latio" in _codigos(eventos)


def test_vigia_muerto_antes_de_latir_no_espera_el_tope():
    f = Falsas()
    f.latido, f.vigia_vive = False, False
    r, _ = _correr(f)
    assert r["codigo"] == "vigia_no_latio"


def test_pausa_puesta_al_final_es_fallo_con_el_motivo_en_la_bitacora():
    f = Falsas()
    f.pausa_leida = {"puesta": True, "legible": True, "origen": "c5", "motivo": "fuera_de_mision", "paso": 3}
    r, eventos = _correr(f)
    assert (r["estado"], r["codigo"]) == ("fallido", "pausa_puesta")
    (p,) = [e for e in eventos if e["evento"] == "pausa_detectada"]
    assert p["datos"] == {"origen": "c5", "motivo": "fuera_de_mision", "paso": 3, "legible": True}
    assert r["crudas"]  # el piso: las crudas salen igual


def test_cerebro_con_error_es_fallo_pero_entrega_crudas():
    f = Falsas()
    f.cerebro = (1, _crudo("nada"))
    r, _ = _correr(f)
    assert (r["estado"], r["codigo"]) == ("fallido", "cerebro_fallo") and r["crudas"]


def test_resultado_que_no_cuadra_con_el_registro_de_c3_es_fallo():
    f = Falsas()
    f.registro = [{"evento": "herramienta_pedida", "tool_use_id": "t1"},
                  {"evento": "resultado_devuelto", "tool_use_id": "t1", "sha256": "0" * 64}]
    r, eventos = _correr(f)
    assert (r["estado"], r["codigo"]) == ("fallido", "registro_no_cuadra")
    (paso,) = [e for e in eventos if e["evento"] == "paso"]
    assert paso["datos"] == {"comando": CMD, "en_registro": True, "cuadra": False}


def test_paso_no_anotado_por_el_proxy_es_fallo():
    f = Falsas()
    f.registro = []
    r, _ = _correr(f)
    assert r["codigo"] == "registro_no_cuadra"


def test_cadena_rota_y_vigia_que_no_cierra_son_fallos():
    f = Falsas()
    f.cadena = False
    assert _correr(f)[0]["codigo"] == "cadena_rota"
    f = Falsas()
    f.vigia_cierre = (1, "arranco=false", "Traceback: el vigia reventó")
    assert _correr(f)[0]["codigo"] == "vigia_no_cerro"


def test_auditor_ilegible_retiene_todo_fail_closed():
    f = Falsas()
    f.auditor_revienta = True
    r, _ = _correr(f)
    assert (r["estado"], r["codigo"]) == ("fallido", "auditor_ilegible")
    assert r["afirmaciones"] == [] and r["descartadas"][0]["estado"] == "retenida_por_auditor"
    assert r["descartadas"][0]["codigo"] == "auditor_ilegible" and r["crudas"]


def test_afirmacion_retenida_por_el_auditor_no_sale():
    f = Falsas()
    f.revision = Revision(False, None, None, (), frozenset(), frozenset({"a1"}))
    r, _ = _correr(f)
    assert r["estado"] == "completado" and r["afirmaciones"] == []
    assert r["descartadas"] == [{"estado": "retenida_por_auditor", "codigo": "retenida_por_auditor", "datos": {},
                                 **AFIRMACION}]


def test_auditor_que_pausa_es_fallo():
    f = Falsas()
    f.revision = Revision(True, "prohibido", 1, (), frozenset({"a1"}), frozenset())
    r, _ = _correr(f)
    assert (r["estado"], r["codigo"]) == ("fallido", "auditor_pauso") and r["verificacion"]["auditor_pauso"]


def test_afirmacion_inventada_se_descarta_con_su_codigo():
    f = Falsas()
    f.cerebro = (0, _crudo(json.dumps([{**AFIRMACION, "linea": "Mem: 91 GB", "dato": "91 GB"}])))
    r, eventos = _correr(f)
    assert r["afirmaciones"] == [] and r["descartadas"][0]["estado"] == "sin_respaldo"
    assert "afirmacion_descartada" in _codigos(eventos)


def test_sesion_no_iniciada_si_el_arnes_no_la_anuncia():
    f = Falsas()
    f.cerebro = (1, b"")
    r, _ = _correr(f)
    assert r["sesion_iniciada"] is False and r["codigo"] == "cerebro_fallo"


def test_tope_del_cerebro_vencido_es_fallo_y_cierra_el_vigia():
    f = Falsas()

    async def tarda(*a):
        raise asyncio.TimeoutError()
    f.correr_cerebro = tarda
    r, eventos = _correr(f)
    assert (r["estado"], r["codigo"], r["crudas"]) == ("fallido", "cerebro_tope_vencido", [])
    assert "cerrar_vigia" in f.llamadas and _codigos(eventos)[-1] == "turno_fallido"


def test_una_excepcion_inesperada_cierra_el_vigia_y_se_propaga():
    f = Falsas()

    async def revienta(*a):
        raise RuntimeError("x")
    f.correr_cerebro = revienta
    with pytest.raises(RuntimeError):
        _correr(f)
    assert "cerrar_vigia" in f.llamadas


def test_el_auditor_final_recibe_las_maquinas_de_la_mision_con_su_direccion():
    """Misión real 2026-09-17 11:23: sin las máquinas, el auditor juzgó «otra máquina» el ssh a la VM elegida."""
    f = Falsas()
    _correr(f)
    assert [(m.nombre, m.ip) for m in f.maquinas_auditadas] == [("ejecutor-prueba", "192.0.2.50")]



def test_el_prompt_exige_el_dato_literal_y_una_afirmacion_por_valor():
    """Misión real en atemai (2026-09-17 13:0x, primera con la compuerta abierta): el modelo mandó
    `dato` en prosa («51G disponible de 94G total (44% usado)») y la cita lo tiró con `dato_no_entero`.
    El turno terminó bien y sin afirmaciones: fail-closed correcto, resultado inútil. El prompt tiene
    que decir que el dato es UN valor que aparece tal cual en la línea, y que dos preguntas son dos
    afirmaciones."""
    p = M.prompt_del_turno("espacio libre y versión", HOSTS, frozenset({"ejecutor-prueba"}))
    assert "tal cual" in p and "una afirmación por cada" in p
    assert "no juntes" in p.lower() and "no calcules" in p.lower()


def test_el_prompt_manda_las_tuberias_dentro_del_comando_remoto():
    """Misión real en atemai (2026-09-17 13:12): el modelo escribió
    `ssh -tt … "df -h /" 2>&1 | tail -n 3`. La tubería corre en hall9000, así que el comando toca DOS
    máquinas y `capturas` no lo respalda (correcto: la máquina la decide el gancho, no el modelo).
    Resultado: cero capturas y cero afirmaciones. El prompt tiene que pedir la tubería DENTRO de las
    comillas del comando remoto."""
    p = M.prompt_del_turno("espacio libre", HOSTS, frozenset({"ejecutor-prueba"}))
    assert "DENTRO de las comillas" in p and "tubería" in p


def test_capturas_ignora_el_comando_que_toca_dos_maquinas():
    """Control del defecto de arriba, con el mismo comando que mandó el modelo."""
    con_tuberia = f'ssh -tt -p 58291 axioma@192.0.2.50 "df -h /" 2>&1 | tail -n 3'
    pedidas, resultados = {"t1": con_tuberia}, {"t1": ("/dev/sda1 20G 1G 19G 5% /", False)}
    assert M.capturas(pedidas, resultados, HOSTS) == ()
    adentro = 'ssh -tt -p 58291 axioma@192.0.2.50 "df -h / | tail -n 3"'
    (captura,) = M.capturas({"t1": adentro}, resultados, HOSTS)
    assert captura.maquina == "ejecutor-prueba"


# --- Un turno que no entrega NINGUNA afirmacion no es "completado" (2026-09-20) ----------------
# INCIDENTE. Misiones 445ac19c y 10707ccc salieron `estado="completada"`, `codigo=None`,
# indistinguibles de un turno que si afirmo. La causa de fondo era otra (el cerebro
# entregaba el JSON dentro de un bloque ```json y el parser lo tiraba, jax#229), pero lo
# que convirtio un defecto de parseo en un FALSO EXITO fue esto: nadie mira una mision
# que dice "completada".
#
# El Ejecutor existe para producir afirmaciones RESPALDADAS. Cero entregadas es cero
# trabajo entregado, y tiene que leerse asi.
#
# Ubicacion en la lista de codigos: AL FINAL, a proposito. Si el auditor pauso, si el
# registro no cuadra, si la cadena se rompio o si el vigia no cerro, ESE es el motivo del
# cero y es el que hay que leer. `sin_afirmaciones` es el caso residual: todo lo demas
# salio bien y aun asi no salio nada.
#
# NO distingue "el cerebro no afirmo nada" de "el auditor las descarto todas": las dos
# entregan cero. Cual fue se lee en `descartadas`, que viaja en el mismo resultado.

def test_turno_sin_ninguna_afirmacion_es_fallo_con_codigo_propio():
    f = Falsas()
    f.cerebro = (0, _crudo("no devolvi ninguna afirmacion"))
    r, eventos = _correr(f)
    assert (r["estado"], r["codigo"]) == ("fallido", "sin_afirmaciones"), r
    assert r["afirmaciones"] == []
    assert "turno_fallido" in _codigos(eventos)


def test_el_cero_no_tapa_el_motivo_real_cuando_lo_hay():
    """Si el registro no cuadra, el codigo tiene que seguir siendo `registro_no_cuadra`:
    es la causa, y `sin_afirmaciones` seria la consecuencia. Leer la consecuencia en vez
    de la causa manda a depurar al lugar equivocado."""
    f = Falsas()
    f.cerebro = (0, _crudo("no devolvi ninguna afirmacion"))
    f.registro = []
    r, _ = _correr(f)
    assert r["codigo"] == "registro_no_cuadra", r


def test_un_turno_que_SI_afirma_sigue_completado():
    """El control que impide que esto rompa el camino bueno."""
    r, _ = _correr(Falsas())
    assert (r["estado"], r["codigo"]) == ("completado", None), r
    assert len(r["afirmaciones"]) == 1


def test_todas_descartadas_por_el_auditor_SIGUE_completado():
    """El limite de este cambio, y es deliberado. Si el auditor RETUVO la afirmacion, el
    turno sigue "completado": el sistema hizo exactamente su trabajo y el cero SE VE en
    `descartadas`, que la pantalla muestra. Esa decision es anterior (ver
    test_afirmacion_retenida_por_el_auditor_no_sale) y esta rama NO la pisa.

    Lo que el incidente destapo no es "cero entregadas": es el cero INVISIBLE -- nada
    propuesto y nada descartado, un turno que no deja rastro de que no hizo nada."""
    f = Falsas()
    f.revision = Revision(False, None, None, (), frozenset(), frozenset({"a1"}))
    r, _ = _correr(f)
    assert (r["estado"], r["codigo"]) == ("completado", None), r
    assert r["afirmaciones"] == [] and r["descartadas"], "el cero se ve en descartadas"


def test_el_vigia_que_no_late_conserva_SU_codigo_no_el_del_cero():
    """Regresion de la primera version de este cambio: con el vigia caido el cerebro ni
    corre, `entrega` queda en None, y `not entrega.respaldadas` reventaba con
    AttributeError. El codigo del vigia ya explica el cero."""
    f = Falsas()
    f.latido = False
    r, _ = _correr(f)
    assert (r["estado"], r["codigo"]) == ("fallido", "vigia_no_latio"), r


def test_el_stderr_del_vigia_llega_a_la_bitacora_cuando_NO_cierra():
    """Lo que faltaba el 2026-09-20: una mision fallo con `vigia_no_latio` y no habia
    una sola linea para investigar. El diagnostico salio corriendo el vigia a mano."""
    f = Falsas()
    f.vigia_cierre = (1, "arranco=false", "Traceback: el vigia reventó")
    _, eventos = _correr(f)
    (cerrado,) = [e for e in eventos if e["evento"] == "vigia_cerrado"]
    assert "Traceback" in cerrado["datos"].get("stderr", ""), cerrado


def test_el_stderr_NO_ensucia_la_bitacora_cuando_el_vigia_cierra_bien():
    """En el camino feliz el stderr son lineas de INFO de httpx que no dicen nada.
    Un log que siempre grita es un log que nadie lee."""
    f = Falsas()
    f.vigia_cierre = (0, "arranco=true cerrada=true", "INFO:httpx:HTTP Request: POST ... 200 OK")
    _, eventos = _correr(f)
    (cerrado,) = [e for e in eventos if e["evento"] == "vigia_cerrado"]
    assert "stderr" not in cerrado["datos"], cerrado


# --- `vigia_no_latio` tiene que decir CUANTO espero (2026-09-20) --------------------------------
# Una mision fallo asi y el evento solo decia `vivo=false`. Para saber si el vigia
# estaba muerto o simplemente lento hubo que medir a mano, contra la base, la distancia
# entre `turno_lanzado` y `vigia_late` de las misiones que SI latieron. Resultado: con
# `el_juez` se tarda 125-199 s contra un presupuesto de 180 -- pegado al techo. Con
# `thot` eran 16 s: el tope estaba calibrado para el auditor de nube.
#
# Si el evento hubiera dicho "espere 180 s", eso se sabia en diez segundos.

def test_vigia_no_latio_dice_cuanto_espero_y_si_el_proceso_vivia():
    f = Falsas()
    f.latido = False
    _, eventos = _correr(f)
    (e,) = [x for x in eventos if x["evento"] == "vigia_no_latio"]
    assert "espera_s" in e["datos"], e
    assert e["datos"]["espera_s"] == pytest.approx(0.2)
    # `vivo` distingue el caso que importa: un vigia MUERTO es un fallo del vigia; uno
    # VIVO que no llego a latir es un presupuesto corto, que es otro problema.
    assert "vivo" in e["datos"]
