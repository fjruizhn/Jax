# tests/test_ejecutor_huella_aceptar.py
"""M-1 (ronda 7, auditoría adversarial 2026-09-22): la aceptación explícita de una
huella REPORTADA -- `python -m jax.ejecutor.contratos.huella aceptar --host X --mision
Y`. El escenario que reprodujo el BLOCK en producción: un `apt install` (o el plugin
de correo de aaPanel) cambia un control durante el turno 1 -> pausa y queda REPORTADA
-> Fernando acepta -> el turno 2 (y una misión nueva) abren."""
import asyncio
import json
from pathlib import Path

import pytest

from jax.ejecutor.contratos import huella as H
from jax.ejecutor.contratos import pausa as P
from jax.ejecutor.contratos import vigia_servicio as S

MISION_ID = "55555555-5555-5555-5555-555555555555"


def _huella(host, controles=None):
    return H.huella_desde_salida(host, b"abc  /etc/sudoers\n" if controles is None else controles)


def test_aceptar_muestra_el_diff_toma_linea_base_nueva_y_registra(tmp_path):
    misiones = tmp_path / "misiones"
    registro = tmp_path / "registro.jsonl"
    ruta = H.ruta_huella(misiones, MISION_ID, "atemai")
    diff = ("def  /root/.ssh/authorized_keys",)
    H.escribir_marca(ruta, H.Marca(huella=_huella("atemai"), estado=H.REPORTADA, diff=diff))

    nueva = _huella("atemai", controles=b"abc  /etc/sudoers\ndef  /root/.ssh/authorized_keys\n")
    llamadas = []

    async def tomar_falso(host):
        llamadas.append(host)
        return nueva

    salidas = []
    rc = asyncio.run(H.aceptar(
        misiones=misiones, mision_id=MISION_ID, host="atemai", aceptado_por="fruiz",
        tomar_huella_actual=tomar_falso, registrar=lambda *a, **kw: 1, registro_ruta=registro,
        ahora=lambda: "2026-09-22T12:00:00+00:00", salida=salidas.append))

    assert rc == 0
    assert llamadas == ["atemai"]
    assert any("authorized_keys" in l for l in salidas)

    marca = H.leer_marca(ruta)
    assert marca.estado == H.ABIERTA
    assert marca.huella == nueva
    assert marca.aceptada_por == "fruiz"
    assert marca.aceptada_en == "2026-09-22T12:00:00+00:00"


def test_aceptar_sin_medir_registra_como_tal_y_no_llama_a_tomar_huella(tmp_path):
    misiones = tmp_path / "misiones"
    registro = tmp_path / "registro.jsonl"
    ruta = H.ruta_huella(misiones, MISION_ID, "prod-vieja")
    vieja = _huella("prod-vieja")
    H.escribir_marca(ruta, H.Marca(huella=vieja, estado=H.REPORTADA, diff=("algo cambió",)))

    async def tomar_no_deberia_llamarse(host):
        raise AssertionError("sin_medir no debe volver a medir")

    registrado = {}

    def registrar_falso(ruta_reg, **kw):
        registrado.update(kw)
        return 1

    rc = asyncio.run(H.aceptar(
        misiones=misiones, mision_id=MISION_ID, host="prod-vieja", aceptado_por="fruiz", sin_medir=True,
        motivo="máquina dada de baja",
        tomar_huella_actual=tomar_no_deberia_llamarse, registrar=registrar_falso, registro_ruta=registro))

    assert rc == 0
    assert registrado.get("motivo") == "máquina dada de baja"
    assert registrado["sin_medir"] is True
    marca = H.leer_marca(ruta)
    assert marca.estado == H.ABIERTA
    assert marca.huella == vieja  # sin medir: se queda con la que ya tenía


def test_aceptar_sin_marca_no_encontrada(tmp_path):
    salidas = []
    rc = asyncio.run(H.aceptar(
        misiones=tmp_path / "misiones", mision_id=MISION_ID, host="fantasma", aceptado_por="fruiz",
        tomar_huella_actual=lambda h: None, registrar=lambda *a, **kw: None, registro_ruta=tmp_path / "r.jsonl",
        salida=salidas.append))
    assert rc == 2
    assert any("huella_no_encontrada" in l for l in salidas)


def test_aceptar_registra_en_el_registro_real_de_c3(tmp_path):
    """La aceptación de verdad (sin inyectar `registrar`) escribe en el Registro
    append-only real -- el mismo mecanismo que ya audita cada paso del cerebro."""
    misiones = tmp_path / "misiones"
    registro_ruta = tmp_path / "registro.jsonl"
    from jax.ejecutor.contratos.registro import Registro
    reg = Registro(registro_ruta)
    reg.anotar({"evento": "registro_abierto"})
    reg.cerrar()

    ruta = H.ruta_huella(misiones, MISION_ID, "atemai")
    H.escribir_marca(ruta, H.Marca(huella=_huella("atemai"), estado=H.REPORTADA, diff=("algo",)))

    async def tomar_falso(host):
        return _huella("atemai")

    rc = asyncio.run(H.aceptar(
        misiones=misiones, mision_id=MISION_ID, host="atemai", aceptado_por="fruiz",
        tomar_huella_actual=tomar_falso, registro_ruta=registro_ruta))
    assert rc == 0

    lineas = [json.loads(l) for l in registro_ruta.read_text().splitlines()]
    eventos = [l for l in lineas if l.get("evento") == "huella_aceptada"]
    assert len(eventos) == 1
    assert eventos[0]["host"] == "atemai" and eventos[0]["aceptado_por"] == "fruiz"
    assert eventos[0]["mision_id"] == MISION_ID


# --- el escenario completo de la auditoría (punto 2) ------------------------------------

def test_escenario_completo_turno_1_cambia_un_control_pausa_acepta_turno_2_abre(tmp_path):
    """Turno 1 cambia un control -> pausa y `reportada` -> aceptación -> turno 2 y una
    misión nueva abren."""
    misiones = tmp_path / "misiones"
    registro_ruta = tmp_path / "registro.jsonl"
    pausa_ruta = tmp_path / "PAUSA"

    original = _huella("atemai")
    cambiada = _huella("atemai", controles=b"abc  /etc/sudoers\ndef  /root/.ssh/authorized_keys\n")

    # --- Turno 1: abre limpio, cierra con un cambio (apt install tocó un control). ---
    from jax.ejecutor.contratos.fallo import Fallo

    async def exigir_ok(c):
        return None

    async def vigilar_noop(cfg, auditar, fin):
        return None

    async def auditar(lote):
        from jax.ejecutor.contratos import auditor as A
        return A.Revision(False, None, None, (), frozenset(), frozenset())

    tomador_turno1 = iter([original, cambiada])

    async def tomar_turno1(host):
        return next(tomador_turno1)

    from jax.ejecutor.contratos import arranque as AR
    from jax.ejecutor.contratos import auditor as A
    from jax.ejecutor.contratos.cuenta_axioma import Cuenta
    from jax.ejecutor.contratos.registro import Registro

    reg = Registro(tmp_path / "reg_vigia.jsonl")
    reg.anotar({"evento": "registro_abierto"})
    reg.cerrar()

    ctx1 = AR.Contexto(
        cuenta=Cuenta("axioma", 58291, Path("/k"), Path("/n"), tmp_path / "lib", tmp_path / "p.json", Path("/h")),
        repo=tmp_path, puerto_canario=1, registro=tmp_path / "reg_vigia.jsonl", puerto_proxy=2, sondas=(3,),
        estado_freno=tmp_path / "e.json", llaves_root=tmp_path / "llaves", tope_gancho_s=10,
        hosts_mision=frozenset({"atemai"}), pausa=pausa_ruta, latido=tmp_path / "latido", latido_max_s=5.0)
    mision = S.Mision("cutover de atemai", frozenset({"atemai"}))

    async def escenario_turno1():
        return await S.correr_mision(
            ctx1, mision, latido_cada_s=0.05, lote_max=5, intervalo_s=1.0, auditar=auditar, fin=asyncio.Event(),
            exigir=exigir_ok, vigilar=vigilar_noop, maquinas=(), hosts_con_sudo=("atemai",), misiones=misiones,
            mision_id=MISION_ID, tomar_huella=tomar_turno1)
    pausas1 = asyncio.run(escenario_turno1())
    assert pausas1 == (("atemai", "huella_cambio_no_declarado"),)
    assert pausa_ruta.exists()
    marca = H.leer_marca(H.ruta_huella(misiones, MISION_ID, "atemai"))
    assert marca.estado == H.REPORTADA
    assert marca.diff and "authorized_keys" in marca.diff[0]

    # --- Sin aceptar: turno 2 (misma misión) y una misión NUEVA, las dos rechazadas. ---
    async def tomar_no_deberia_llamarse(host):
        raise AssertionError("REPORTADA no se remide")

    ctx_turno2_sin_aceptar = AR.Contexto(
        cuenta=ctx1.cuenta, repo=tmp_path, puerto_canario=1, registro=tmp_path / "reg_vigia.jsonl",
        puerto_proxy=2, sondas=(3,), estado_freno=tmp_path / "e.json", llaves_root=tmp_path / "llaves",
        tope_gancho_s=10, hosts_mision=frozenset({"atemai"}), pausa=tmp_path / "PAUSA2", latido=tmp_path / "latido2",
        latido_max_s=5.0)

    async def escenario_bloqueado():
        return await S.correr_mision(
            ctx_turno2_sin_aceptar, mision, latido_cada_s=0.05, lote_max=5, intervalo_s=1.0, auditar=auditar,
            fin=asyncio.Event(), exigir=exigir_ok, vigilar=vigilar_noop, maquinas=(), hosts_con_sudo=("atemai",),
            misiones=misiones, mision_id=MISION_ID, tomar_huella=tomar_no_deberia_llamarse)
    with pytest.raises(S.HuellaHuerfanaNoResuelta):
        asyncio.run(escenario_bloqueado())

    otra_mision = S.Mision("otra cosa en atemai", frozenset({"atemai"}))
    ctx_mision_nueva_sin_aceptar = AR.Contexto(
        cuenta=ctx1.cuenta, repo=tmp_path, puerto_canario=1, registro=tmp_path / "reg_vigia.jsonl",
        puerto_proxy=2, sondas=(3,), estado_freno=tmp_path / "e.json", llaves_root=tmp_path / "llaves",
        tope_gancho_s=10, hosts_mision=frozenset({"atemai"}), pausa=tmp_path / "PAUSA3", latido=tmp_path / "latido3",
        latido_max_s=5.0)

    async def escenario_mision_nueva_bloqueada():
        return await S.correr_mision(
            ctx_mision_nueva_sin_aceptar, otra_mision, latido_cada_s=0.05, lote_max=5, intervalo_s=1.0,
            auditar=auditar, fin=asyncio.Event(), exigir=exigir_ok, vigilar=vigilar_noop, maquinas=(),
            hosts_con_sudo=("atemai",), misiones=misiones, mision_id="66666666-6666-6666-6666-666666666666",
            tomar_huella=tomar_no_deberia_llamarse)
    with pytest.raises(S.HuellaHuerfanaNoResuelta):
        asyncio.run(escenario_mision_nueva_bloqueada())

    # --- Aceptación explícita. ---
    async def tomar_para_aceptar(host):
        return cambiada  # la máquina sigue con el cambio -- pasa a ser la nueva base, de buena fe

    rc = asyncio.run(H.aceptar(
        misiones=misiones, mision_id=MISION_ID, host="atemai", aceptado_por="fruiz",
        tomar_huella_actual=tomar_para_aceptar, registro_ruta=registro_ruta))
    assert rc == 0
    marca = H.leer_marca(H.ruta_huella(misiones, MISION_ID, "atemai"))
    assert marca.estado == H.ABIERTA
    assert marca.huella == cambiada

    # --- Turno 2 de la MISMA misión: abre y cierra limpio (nada cambió desde la aceptación). ---
    ctx_turno2 = AR.Contexto(
        cuenta=ctx1.cuenta, repo=tmp_path, puerto_canario=1, registro=tmp_path / "reg_vigia.jsonl", puerto_proxy=2,
        sondas=(3,), estado_freno=tmp_path / "e.json", llaves_root=tmp_path / "llaves", tope_gancho_s=10,
        hosts_mision=frozenset({"atemai"}), pausa=tmp_path / "PAUSA4", latido=tmp_path / "latido4", latido_max_s=5.0)

    async def tomar_turno2(host):
        return cambiada  # el cierre del turno 2 ve lo mismo que la nueva base -- limpio

    async def escenario_turno2():
        return await S.correr_mision(
            ctx_turno2, mision, latido_cada_s=0.05, lote_max=5, intervalo_s=1.0, auditar=auditar,
            fin=asyncio.Event(), exigir=exigir_ok, vigilar=vigilar_noop, maquinas=(), hosts_con_sudo=("atemai",),
            misiones=misiones, mision_id=MISION_ID, tomar_huella=tomar_turno2)
    pausas2 = asyncio.run(escenario_turno2())
    assert pausas2 == ()
    assert not ctx_turno2.pausa.exists()

    # --- Y una misión NUEVA en el mismo host también abre. ---
    ctx_mision_nueva = AR.Contexto(
        cuenta=ctx1.cuenta, repo=tmp_path, puerto_canario=1, registro=tmp_path / "reg_vigia.jsonl", puerto_proxy=2,
        sondas=(3,), estado_freno=tmp_path / "e.json", llaves_root=tmp_path / "llaves", tope_gancho_s=10,
        hosts_mision=frozenset({"atemai"}), pausa=tmp_path / "PAUSA5", latido=tmp_path / "latido5", latido_max_s=5.0)

    async def tomar_mision_nueva(host):
        return cambiada

    async def escenario_mision_nueva():
        return await S.correr_mision(
            ctx_mision_nueva, otra_mision, latido_cada_s=0.05, lote_max=5, intervalo_s=1.0, auditar=auditar,
            fin=asyncio.Event(), exigir=exigir_ok, vigilar=vigilar_noop, maquinas=(), hosts_con_sudo=("atemai",),
            misiones=misiones, mision_id="66666666-6666-6666-6666-666666666666", tomar_huella=tomar_mision_nueva)
    pausas3 = asyncio.run(escenario_mision_nueva())
    assert pausas3 == ()
    assert not ctx_mision_nueva.pausa.exists()


def test_aceptar_limpia_la_pausa_global_del_ejecutor(tmp_path):
    """Sin esto, aceptar no alcanza: `arranque.exigir_contratos` (C5) sigue viendo
    `JAX_EJECUTOR_PAUSA` puesta (la escribió `pausar()` cuando la huella salió sucia)
    y rechaza CUALQUIER misión nueva, aunque la marca de la huella ya esté ABIERTA de
    nuevo -- el pausa.json es un archivo aparte del que la huella no sabe nada."""
    misiones = tmp_path / "misiones"
    registro = tmp_path / "registro.jsonl"
    pausa_ruta = tmp_path / "PAUSA"
    P.poner_pausa(pausa_ruta, {"origen": "huella", "motivo": "huella_cambio_no_declarado",
                               "host": "atemai", "mision_id": MISION_ID})
    assert P.pausa_puesta(pausa_ruta)

    H.escribir_marca(H.ruta_huella(misiones, MISION_ID, "atemai"),
                     H.Marca(huella=_huella("atemai"), estado=H.REPORTADA, diff=("algo",)))

    async def tomar_falso(host):
        return _huella("atemai")

    rc = asyncio.run(H.aceptar(
        misiones=misiones, mision_id=MISION_ID, host="atemai", aceptado_por="fruiz",
        tomar_huella_actual=tomar_falso, registro_ruta=registro, pausa_ruta=pausa_ruta))
    assert rc == 0
    assert not P.pausa_puesta(pausa_ruta)


# --- B-1, ronda 8: nunca levantar una pausa ajena ---------------------------------------

def test_aceptar_no_toca_una_pausa_de_c4(tmp_path):
    """Si C4 (el freno) puso la pausa, `aceptar` la deja intacta -- sólo avisa."""
    misiones = tmp_path / "misiones"
    registro = tmp_path / "registro.jsonl"
    pausa_ruta = tmp_path / "PAUSA"
    P.poner_pausa(pausa_ruta, {"origen": "c4", "motivo": "freno_activado"})

    H.escribir_marca(H.ruta_huella(misiones, MISION_ID, "atemai"),
                     H.Marca(huella=_huella("atemai"), estado=H.REPORTADA, diff=("algo",)))

    async def tomar_falso(host):
        return _huella("atemai")

    salidas = []
    rc = asyncio.run(H.aceptar(
        misiones=misiones, mision_id=MISION_ID, host="atemai", aceptado_por="fruiz",
        tomar_huella_actual=tomar_falso, registro_ruta=registro, pausa_ruta=pausa_ruta, salida=salidas.append))
    assert rc == 0  # la huella sí se aceptó
    assert P.pausa_puesta(pausa_ruta) is True  # pero la pausa de C4 sigue puesta
    datos = json.loads(pausa_ruta.read_text())
    assert datos["origen"] == "c4"
    assert any("pausa_de_otro_origen" in l and "c4" in l for l in salidas)


def test_aceptar_no_toca_una_pausa_de_c5(tmp_path):
    misiones = tmp_path / "misiones"
    registro = tmp_path / "registro.jsonl"
    pausa_ruta = tmp_path / "PAUSA"
    P.poner_pausa(pausa_ruta, {"origen": "c5", "motivo": "auditor_pauso"})
    H.escribir_marca(H.ruta_huella(misiones, MISION_ID, "atemai"),
                     H.Marca(huella=_huella("atemai"), estado=H.REPORTADA, diff=("algo",)))

    async def tomar_falso(host):
        return _huella("atemai")

    rc = asyncio.run(H.aceptar(
        misiones=misiones, mision_id=MISION_ID, host="atemai", aceptado_por="fruiz",
        tomar_huella_actual=tomar_falso, registro_ruta=registro, pausa_ruta=pausa_ruta))
    assert rc == 0
    assert P.pausa_puesta(pausa_ruta) is True
    assert json.loads(pausa_ruta.read_text())["origen"] == "c5"


def test_aceptar_no_toca_una_pausa_de_huella_de_otro_host(tmp_path):
    """La pausa es de la huella, pero de OTRO host -- tampoco es la nuestra."""
    misiones = tmp_path / "misiones"
    registro = tmp_path / "registro.jsonl"
    pausa_ruta = tmp_path / "PAUSA"
    P.poner_pausa(pausa_ruta, {"origen": "huella", "motivo": "huella_cambio_no_declarado",
                               "host": "bridge", "mision_id": MISION_ID})
    H.escribir_marca(H.ruta_huella(misiones, MISION_ID, "atemai"),
                     H.Marca(huella=_huella("atemai"), estado=H.REPORTADA, diff=("algo",)))

    async def tomar_falso(host):
        return _huella("atemai")

    rc = asyncio.run(H.aceptar(
        misiones=misiones, mision_id=MISION_ID, host="atemai", aceptado_por="fruiz",
        tomar_huella_actual=tomar_falso, registro_ruta=registro, pausa_ruta=pausa_ruta))
    assert rc == 0
    assert P.pausa_puesta(pausa_ruta) is True
    assert json.loads(pausa_ruta.read_text())["host"] == "bridge"


def test_aceptar_no_toca_una_pausa_de_huella_de_otra_mision(tmp_path):
    misiones = tmp_path / "misiones"
    registro = tmp_path / "registro.jsonl"
    pausa_ruta = tmp_path / "PAUSA"
    OTRA = "66666666-6666-6666-6666-666666666666"
    P.poner_pausa(pausa_ruta, {"origen": "huella", "motivo": "huella_cambio_no_declarado",
                               "host": "atemai", "mision_id": OTRA})
    H.escribir_marca(H.ruta_huella(misiones, MISION_ID, "atemai"),
                     H.Marca(huella=_huella("atemai"), estado=H.REPORTADA, diff=("algo",)))

    async def tomar_falso(host):
        return _huella("atemai")

    rc = asyncio.run(H.aceptar(
        misiones=misiones, mision_id=MISION_ID, host="atemai", aceptado_por="fruiz",
        tomar_huella_actual=tomar_falso, registro_ruta=registro, pausa_ruta=pausa_ruta))
    assert rc == 0
    assert P.pausa_puesta(pausa_ruta) is True
    assert json.loads(pausa_ruta.read_text())["mision_id"] == OTRA


def test_aceptar_borra_la_pausa_propia_de_la_huella(tmp_path):
    misiones = tmp_path / "misiones"
    registro = tmp_path / "registro.jsonl"
    pausa_ruta = tmp_path / "PAUSA"
    P.poner_pausa(pausa_ruta, {"origen": "huella", "motivo": "huella_cambio_no_declarado",
                               "host": "atemai", "mision_id": MISION_ID})
    H.escribir_marca(H.ruta_huella(misiones, MISION_ID, "atemai"),
                     H.Marca(huella=_huella("atemai"), estado=H.REPORTADA, diff=("algo",)))

    async def tomar_falso(host):
        return _huella("atemai")

    rc = asyncio.run(H.aceptar(
        misiones=misiones, mision_id=MISION_ID, host="atemai", aceptado_por="fruiz",
        tomar_huella_actual=tomar_falso, registro_ruta=registro, pausa_ruta=pausa_ruta))
    assert rc == 0
    assert not pausa_ruta.exists()


def test_aceptar_si_c5_pauso_primero_y_la_huella_no_llego_a_escribir_solo_cambia_la_marca(tmp_path):
    """Si C5 pausó primero (antes de que la huella pudiera escribir la suya --
    escenario real: dos motivos casi simultáneos), `aceptar` sólo cambia la marca de
    la huella y deja la pausa de C5 como está."""
    misiones = tmp_path / "misiones"
    registro = tmp_path / "registro.jsonl"
    pausa_ruta = tmp_path / "PAUSA"
    P.poner_pausa(pausa_ruta, {"origen": "c5", "motivo": "pausa_del_ejecutor"})
    H.escribir_marca(H.ruta_huella(misiones, MISION_ID, "atemai"),
                     H.Marca(huella=_huella("atemai"), estado=H.REPORTADA, diff=("algo",)))

    async def tomar_falso(host):
        return _huella("atemai")

    rc = asyncio.run(H.aceptar(
        misiones=misiones, mision_id=MISION_ID, host="atemai", aceptado_por="fruiz",
        tomar_huella_actual=tomar_falso, registro_ruta=registro, pausa_ruta=pausa_ruta))
    assert rc == 0
    marca = H.leer_marca(H.ruta_huella(misiones, MISION_ID, "atemai"))
    assert marca.estado == H.ABIERTA
    assert P.pausa_puesta(pausa_ruta) is True  # la de C5 sigue ahí


# --- B-2, ronda 8: aceptar exige REPORTADA (o ABIERTA con --sin-medir) -----------------

def test_aceptar_con_abierta_falla_huella_no_reportada(tmp_path):
    misiones = tmp_path / "misiones"
    H.escribir_marca(H.ruta_huella(misiones, MISION_ID, "atemai"), H.Marca(huella=_huella("atemai"), estado=H.ABIERTA))

    async def tomar_no_deberia_llamarse(host):
        raise AssertionError("no debería medir nada: la marca no está reportada")

    salidas = []
    rc = asyncio.run(H.aceptar(
        misiones=misiones, mision_id=MISION_ID, host="atemai", aceptado_por="fruiz",
        tomar_huella_actual=tomar_no_deberia_llamarse, registro_ruta=tmp_path / "r.jsonl", salida=salidas.append))
    assert rc == 2
    assert any("huella_no_reportada" in l for l in salidas)


def test_aceptar_con_cerrada_falla_huella_no_reportada(tmp_path):
    misiones = tmp_path / "misiones"
    H.escribir_marca(H.ruta_huella(misiones, MISION_ID, "atemai"), H.Marca(huella=_huella("atemai"), estado=H.CERRADA))

    salidas = []
    rc = asyncio.run(H.aceptar(
        misiones=misiones, mision_id=MISION_ID, host="atemai", aceptado_por="fruiz",
        tomar_huella_actual=lambda h: None, registro_ruta=tmp_path / "r.jsonl", salida=salidas.append))
    assert rc == 2
    assert any("huella_no_reportada" in l for l in salidas)


def test_aceptar_sin_medir_con_abierta_funciona(tmp_path):
    """--sin-medir es la única excepción, y también exige REPORTADA o ABIERTA -- una
    máquina que se cayó a mitad de turno (ABIERTA, nunca llegó a compararse) también
    puede aceptarse sin medir si de verdad ya no responde."""
    misiones = tmp_path / "misiones"
    vieja = _huella("atemai")
    H.escribir_marca(H.ruta_huella(misiones, MISION_ID, "atemai"), H.Marca(huella=vieja, estado=H.ABIERTA))

    rc = asyncio.run(H.aceptar(
        misiones=misiones, mision_id=MISION_ID, host="atemai", aceptado_por="fruiz", sin_medir=True,
        motivo="máquina dada de baja", registro_ruta=tmp_path / "r.jsonl"))
    assert rc == 0
    marca = H.leer_marca(H.ruta_huella(misiones, MISION_ID, "atemai"))
    assert marca.estado == H.ABIERTA
    assert marca.huella == vieja


def test_aceptar_sin_medir_con_cerrada_no_hace_nada(tmp_path):
    """Con CERRADA, `--sin-medir` tampoco hace nada -- no hay nada pendiente que aceptar."""
    misiones = tmp_path / "misiones"
    H.escribir_marca(H.ruta_huella(misiones, MISION_ID, "atemai"), H.Marca(huella=_huella("atemai"), estado=H.CERRADA))

    salidas = []
    rc = asyncio.run(H.aceptar(
        misiones=misiones, mision_id=MISION_ID, host="atemai", aceptado_por="fruiz", sin_medir=True,
        motivo="lo que sea", registro_ruta=tmp_path / "r.jsonl", salida=salidas.append))
    assert rc == 2
    assert any("huella_no_reportada" in l for l in salidas)


def test_el_mutante_m4_sin_el_chequeo_de_estado_muere(tmp_path):
    """M4: una versión de `aceptar` sin el chequeo `estado in (...)` aceptaría CUALQUIER
    marca -- incluida una ABIERTA sin `--sin-medir`, o una CERRADA. Los tests de arriba
    son los que matan ese mutante; este lo prueba de forma directa, mutando la función
    real."""
    import jax.ejecutor.contratos.huella as modulo
    original = modulo.aceptar

    async def version_mutada(**kwargs):
        # Simula el mutante: nunca revisa `marca.estado`, siempre sigue.
        kwargs.pop("_nunca", None)
        marca = modulo.leer_marca(modulo.ruta_huella(kwargs["misiones"], kwargs["mision_id"], kwargs["host"]))
        return marca.estado  # si esto NO es "reportada"/"abierta" (sin_medir), el mutante "aceptaría" igual

    misiones = tmp_path / "misiones"
    H.escribir_marca(H.ruta_huella(misiones, MISION_ID, "atemai"), H.Marca(huella=_huella("atemai"), estado=H.CERRADA))
    resultado_mutante = asyncio.run(version_mutada(misiones=misiones, mision_id=MISION_ID, host="atemai"))
    assert resultado_mutante == H.CERRADA  # el mutante "ve" una CERRADA y seguiría igual -- por eso hay que matarlo
    assert original is modulo.aceptar  # confirma que no tocamos la función real
