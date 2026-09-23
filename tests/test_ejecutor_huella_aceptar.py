# tests/test_ejecutor_huella_aceptar.py
"""M-1 (ronda 7, auditoría adversarial 2026-09-22): la aceptación explícita de una
huella REPORTADA -- `python -m jax.ejecutor.contratos.huella aceptar --host X --mision
Y`. El escenario que reprodujo el BLOCK en producción: un `apt install` (o el plugin
de correo de aaPanel) cambia un control durante el turno 1 -> pausa y queda REPORTADA
-> Fernando acepta -> el turno 2 (y una misión nueva) abren."""
import asyncio
import json
import multiprocessing
import os
import pwd
from pathlib import Path

import pytest

from jax.ejecutor.contratos import huella as H
from jax.ejecutor.contratos import pausa as P
from jax.ejecutor.contratos import vigia_servicio as S

#: FIX CI (ronda 8, auditoría adversarial 2026-09-22): BLOCK-J pasaba "fruiz" a
#: `ruta_authorized_keys_admin()` -- pwd.getpwnam real, sólo resuelve en hall9000.
#: En el runner de CI esa cuenta no existe (medido en jax#263, 3 tests rotos). La
#: cuenta que de verdad corre el proceso existe en cualquier máquina, por definición.
_ADMIN_REAL = pwd.getpwuid(os.getuid()).pw_name

MISION_ID = "55555555-5555-5555-5555-555555555555"

# Formas de hash REALES (MAJOR-6, ronda 2: huella_valida() exige 64 hex) -- no "abc"/"def".
_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64


def _base_completa() -> bytes:
    """MINOR (ronda 3)/RONDA 4: `huella_valida()`, llamada por `vigia_servicio.py` con
    `rutas=huella.RUTAS_DECLARADAS_POR_DEFAULT (+ rutas_extra)`, exige que CADA ruta
    declarada tenga un estado hash/D/A -- nunca E ni ausencia total. Los TIPOS acá
    reflejan lo medido de verdad en hall9000/atemai/prod: `/etc/ssh/sshd_config` es un
    ARCHIVO (hash, no `D`) -- escribir `D` para un archivo era ficción.

    MAJOR-G (ronda 5, auditoría adversarial 2026-09-22): esta nota decía antes que
    `/root/.ssh/authorized_keys` "NO EXISTE en ninguna de las tres" -- ERA FALSO.
    Medido el 2026-09-22: SÍ existe (vacío, 0600, root) en hall9000, atemai y prod;
    sólo falta en `bridge`. El `A` de acá representa el estado SANO de `bridge`; lo
    que se corrige es a qué máquina describía la nota, no el fixture.

    BLOCK-E, punto 2 (ronda 5): `RUTAS_DECLARADAS_POR_DEFAULT` exige TAMBIÉN el glob
    de sbin representado."""
    lineas_por_ruta = {
        "/etc/sudoers": f"{_HASH_A}  /etc/sudoers",
        "/etc/sudoers.d": "D /etc/sudoers.d",
        "/etc/ssh/sshd_config": f"{_HASH_B}  /etc/ssh/sshd_config",
        "/etc/ssh/sshd_config.d": "D /etc/ssh/sshd_config.d",
        "/etc/ssh/authorized_keys.d": "D /etc/ssh/authorized_keys.d",
        "/root/.ssh/authorized_keys": "A /root/.ssh/authorized_keys",
        "/etc/ejecutor-huella": "D /etc/ejecutor-huella",
        H.RUTA_GLOB_SBIN_EJECUTOR: f"D {H.RUTA_GLOB_SBIN_EJECUTOR}",
    }
    assert set(lineas_por_ruta) == set(H.RUTAS_DECLARADAS_POR_DEFAULT), \
        "RUTAS_DECLARADAS_POR_DEFAULT cambió -- actualizar este fixture"
    lineas = [lineas_por_ruta[r] for r in H.RUTAS_DECLARADAS_POR_DEFAULT]
    return ("\n".join(lineas) + "\n").encode()


def _base_completa_con_cambio() -> bytes:
    """`/root/.ssh/authorized_keys` pasa de `A` (ausente, confirmado) a un HASH real
    -- el caso que la ronda 4 señala: "que la ruta pase de A a existir es un CAMBIO"."""
    base = _base_completa().decode()
    sin_root = base.replace("A /root/.ssh/authorized_keys\n", "")
    return (sin_root + f"{_HASH_C}  /root/.ssh/authorized_keys\n").encode()


def _huella(host, controles=None):
    return H.huella_desde_salida(host, _base_completa() if controles is None else controles)


def test_aceptar_muestra_el_diff_toma_linea_base_nueva_y_registra(tmp_path):
    misiones = tmp_path / "misiones"
    registro = tmp_path / "registro.jsonl"
    ruta = H.ruta_huella(misiones, MISION_ID, "atemai")
    diff = ("def  /root/.ssh/authorized_keys",)
    H.escribir_marca(ruta, H.Marca(huella=_huella("atemai"), estado=H.REPORTADA, diff=diff))

    nueva = _huella("atemai", controles=_base_completa_con_cambio())
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


def test_aceptar_rechaza_una_medicion_invalida_no_la_toma_como_linea_base_major_i(tmp_path):
    """MAJOR-I (ronda 6, auditoría adversarial 2026-09-22): `aceptar()` NO llamaba a
    `huella_valida()` sobre la medición NUEVA -- una remedición rota (una ruta con
    `E ... find_fallo`, por ejemplo un `find` que falló a mitad de camino) se aceptaba
    igual como línea base, `estado=ABIERTA`, y la pausa se borraba -- rc=0, sin que
    nada de eso fuera cierto. Ahora tiene que rechazarla ANTES de escribir nada: no
    toca la marca, no registra en C3, no borra la pausa, y lo dice con su propio
    código."""
    misiones = tmp_path / "misiones"
    registro = tmp_path / "registro.jsonl"
    pausa_ruta = tmp_path / "PAUSA"
    ruta = H.ruta_huella(misiones, MISION_ID, "atemai")
    vieja = _huella("atemai")
    H.escribir_marca(ruta, H.Marca(huella=vieja, estado=H.REPORTADA, diff=("algo cambió",)))
    P.poner_pausa(pausa_ruta, {"origen": "huella", "motivo": "huella_cambio_no_declarado",
                               "host": "atemai", "mision_id": MISION_ID, "detalle": []})

    rota = _huella("atemai", controles=_base_completa().replace(
        b"D /etc/sudoers.d", b"E /etc/sudoers.d find_fallo"))
    llamadas = []

    async def tomar_falso(host):
        llamadas.append(host)
        return rota

    registrado = {"n": 0}

    def registrar_falso(*a, **kw):
        registrado["n"] += 1
        return 1

    salidas = []
    rc = asyncio.run(H.aceptar(
        misiones=misiones, mision_id=MISION_ID, host="atemai", aceptado_por="fruiz",
        tomar_huella_actual=tomar_falso, registrar=registrar_falso, registro_ruta=registro,
        pausa_ruta=pausa_ruta, ahora=lambda: "2026-09-22T12:00:00+00:00", salida=salidas.append))

    assert rc == 2
    assert llamadas == ["atemai"]  # sí llegó a medir -- el rechazo es DESPUÉS de medir
    assert registrado["n"] == 0  # nunca se registró en C3: no llegó a "aceptar" de verdad
    assert any("medicion_no_valida" in l for l in salidas), salidas

    marca = H.leer_marca(ruta)
    assert marca.estado == H.REPORTADA  # SIGUE reportada -- no se sobreescribió con la rota
    assert marca.huella == vieja
    assert pausa_ruta.exists()  # la pausa NO se borró


# --- BLOCK-J (ronda 7, auditoría adversarial 2026-09-22): ningún test de MAJOR-I pasaba
# `admin_usuario` -- la rama de `aceptar()` que agrega el `authorized_keys` del
# ADMINISTRADOR a `rutas_exigidas` (justo donde vive la llave del servicio) estaba MUERTA
# en la suite: un mutante que la borrara dejaba los tests en verde igual. Acá la
# remedición es COMPLETA para `RUTAS_DECLARADAS_POR_DEFAULT`, y le falla ÚNICAMENTE la
# ruta del administrador. -----------------------------------------------------------

def test_aceptar_con_admin_usuario_rechaza_si_falla_su_propia_ruta_block_j(tmp_path):
    admin = _ADMIN_REAL
    ruta_admin = H.ruta_authorized_keys_admin(admin)
    misiones = tmp_path / "misiones"
    registro = tmp_path / "registro.jsonl"
    pausa_ruta = tmp_path / "PAUSA"
    ruta = H.ruta_huella(misiones, MISION_ID, "atemai")
    vieja = _huella("atemai")
    H.escribir_marca(ruta, H.Marca(huella=vieja, estado=H.REPORTADA, diff=("algo cambió",)))
    P.poner_pausa(pausa_ruta, {"origen": "huella", "motivo": "huella_cambio_no_declarado",
                               "host": "atemai", "mision_id": MISION_ID, "detalle": []})

    # COMPLETA para RUTAS_DECLARADAS_POR_DEFAULT -- lo único roto es la ruta del admin.
    rota = _huella("atemai", controles=_base_completa() + f"E {ruta_admin} find_fallo\n".encode())

    async def tomar_falso(host):
        return rota

    registrado = {"n": 0}

    def registrar_falso(*a, **kw):
        registrado["n"] += 1
        return 1

    salidas = []
    rc = asyncio.run(H.aceptar(
        misiones=misiones, mision_id=MISION_ID, host="atemai", aceptado_por="fruiz",
        admin_usuario=admin, tomar_huella_actual=tomar_falso, registrar=registrar_falso,
        registro_ruta=registro, pausa_ruta=pausa_ruta, salida=salidas.append))

    assert rc == 2
    assert registrado["n"] == 0
    assert any("medicion_no_valida" in l for l in salidas), salidas
    marca = H.leer_marca(ruta)
    assert marca.estado == H.REPORTADA
    assert marca.huella == vieja
    assert pausa_ruta.exists()


def test_aceptar_con_admin_usuario_y_medicion_completa_acepta_block_j(tmp_path):
    """Contraparte de la de arriba: con la ruta del administrador TAMBIÉN medida y
    sana, `aceptar()` con `admin_usuario` tiene que aceptar igual que sin él -- pasar
    `admin_usuario` no es, por sí solo, motivo de rechazo."""
    admin = _ADMIN_REAL
    ruta_admin = H.ruta_authorized_keys_admin(admin)
    misiones = tmp_path / "misiones"
    registro = tmp_path / "registro.jsonl"
    pausa_ruta = tmp_path / "PAUSA"
    ruta = H.ruta_huella(misiones, MISION_ID, "atemai")
    vieja = _huella("atemai")
    H.escribir_marca(ruta, H.Marca(huella=vieja, estado=H.REPORTADA, diff=("algo cambió",)))
    P.poner_pausa(pausa_ruta, {"origen": "huella", "motivo": "huella_cambio_no_declarado",
                               "host": "atemai", "mision_id": MISION_ID, "detalle": []})

    completa = _huella("atemai", controles=_base_completa() + f"A {ruta_admin}\n".encode())

    async def tomar_falso(host):
        return completa

    salidas = []
    rc = asyncio.run(H.aceptar(
        misiones=misiones, mision_id=MISION_ID, host="atemai", aceptado_por="fruiz",
        admin_usuario=admin, tomar_huella_actual=tomar_falso, registrar=lambda *a, **kw: 1,
        registro_ruta=registro, pausa_ruta=pausa_ruta, salida=salidas.append))

    assert rc == 0
    marca = H.leer_marca(ruta)
    assert marca.estado == H.ABIERTA
    assert marca.huella == completa
    assert not pausa_ruta.exists()


def test_el_mutante_que_ignora_admin_usuario_en_aceptar_muere_block_j():
    """El mutante EXACTO que pide la auditoría: quitar el `if admin_usuario is not
    None: rutas_exigidas = ...` de `aceptar()`, dejando `rutas_exigidas` fija en
    `RUTAS_DECLARADAS_POR_DEFAULT` sin importar qué se pase. Con una medición rota
    ÚNICAMENTE en la ruta del administrador, el código real la rechaza; el mutante
    la deja pasar."""
    admin = _ADMIN_REAL
    ruta_admin = H.ruta_authorized_keys_admin(admin)
    rota = H.huella_desde_salida("atemai", _base_completa() + f"E {ruta_admin} find_fallo\n".encode())

    rutas_reales = H.RUTAS_DECLARADAS_POR_DEFAULT + (H.ruta_authorized_keys_admin(admin),)
    rutas_mutadas = H.RUTAS_DECLARADAS_POR_DEFAULT  # el mutante: ignora admin_usuario

    assert H.huella_valida(rota, rutas=rutas_reales) is False  # el código real: rechaza
    assert H.huella_valida(rota, rutas=rutas_mutadas) is True  # el mutante "logra" pasar


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
    cambiada = _huella("atemai", controles=_base_completa_con_cambio())

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


def test_aceptar_barre_temporales_huerfanos_de_la_pausa_al_arrancar_ronda9(tmp_path):
    """Ronda 9: `aceptar` limpia los `.PAUSA.quitar-tmp-*` huérfanos que un kill previo
    de `quitar_pausa_si` pueda haber dejado -- ANTES de hacer cualquier otra cosa."""
    misiones = tmp_path / "misiones"
    pausa_ruta = tmp_path / "PAUSA"
    P.poner_pausa(pausa_ruta, {"origen": "huella", "host": "atemai", "mision_id": MISION_ID})
    huerfano = tmp_path / ".PAUSA.quitar-tmp-9999-cccc"
    huerfano.write_text(pausa_ruta.read_text())
    H.escribir_marca(H.ruta_huella(misiones, MISION_ID, "atemai"),
                     H.Marca(huella=_huella("atemai"), estado=H.REPORTADA, diff=("algo",)))

    registrado = {}

    def registrar_falso(ruta_reg, **kw):
        registrado.update(kw)
        return 1

    rc = asyncio.run(H.aceptar(
        misiones=misiones, mision_id=MISION_ID, host="atemai", aceptado_por="fruiz", sin_medir=True,
        motivo="prueba barrido",
        tomar_huella_actual=lambda h: (_ for _ in ()).throw(AssertionError("no debería medir")),
        registrar=registrar_falso, registro_ruta=tmp_path / "registro.jsonl", pausa_ruta=pausa_ruta))

    assert rc == 0
    assert not huerfano.exists()
    assert not pausa_ruta.exists()  # la propia sí se borró -- coincide origen/host/mision_id


# --- ronda 10, MINOR: nunca un huella_aceptada=true mudo sobre el estado de la pausa --

def test_aceptar_dice_sin_pausa_que_quitar_cuando_no_habia_ninguna(tmp_path):
    """MINOR (ronda 10): sin pausa puesta (`vista is None`), antes `aceptar` no decía
    NADA sobre eso -- ahora tiene que avisar explícitamente, nunca quedarse mudo antes
    de `huella_aceptada=true`."""
    misiones = tmp_path / "misiones"
    registro = tmp_path / "registro.jsonl"
    pausa_ruta = tmp_path / "PAUSA"  # no existe -- nadie pausó nada

    H.escribir_marca(H.ruta_huella(misiones, MISION_ID, "atemai"),
                     H.Marca(huella=_huella("atemai"), estado=H.REPORTADA, diff=("algo",)))

    async def tomar_falso(host):
        return _huella("atemai")

    salidas = []
    rc = asyncio.run(H.aceptar(
        misiones=misiones, mision_id=MISION_ID, host="atemai", aceptado_por="fruiz",
        tomar_huella_actual=tomar_falso, registro_ruta=registro, pausa_ruta=pausa_ruta, salida=salidas.append))
    assert rc == 0
    assert any("sin_pausa_que_quitar" in l for l in salidas)


def test_aceptar_dice_pausa_propia_borrada_cuando_la_borra(tmp_path):
    """MINOR (ronda 10): cuando SÍ borra su propia pausa, lo dice explícitamente (antes
    sólo había mensaje en los casos de "otro origen" o silencio)."""
    misiones = tmp_path / "misiones"
    registro = tmp_path / "registro.jsonl"
    pausa_ruta = tmp_path / "PAUSA"
    P.poner_pausa(pausa_ruta, {"origen": "huella", "host": "atemai", "mision_id": MISION_ID})

    H.escribir_marca(H.ruta_huella(misiones, MISION_ID, "atemai"),
                     H.Marca(huella=_huella("atemai"), estado=H.REPORTADA, diff=("algo",)))

    async def tomar_falso(host):
        return _huella("atemai")

    salidas = []
    rc = asyncio.run(H.aceptar(
        misiones=misiones, mision_id=MISION_ID, host="atemai", aceptado_por="fruiz",
        tomar_huella_actual=tomar_falso, registro_ruta=registro, pausa_ruta=pausa_ruta, salida=salidas.append))
    assert rc == 0
    assert any("pausa_propia_borrada" in l for l in salidas)


def test_aceptar_no_crashea_si_quitar_pausa_si_explota_con_la_marca_ya_escrita(tmp_path):
    """MINOR (ronda 10): la marca y el registro YA se escribieron para cuando se llega
    a tocar la pausa -- si `quitar_pausa_si` revienta con algo inesperado, `aceptar` no
    puede dejar escapar un traceback crudo (la aceptación de la huella en sí YA es
    válida y ya quedó persistida; lo único que falló es un efecto colateral). Se
    reporta con un código claro y sigue."""
    misiones = tmp_path / "misiones"
    registro = tmp_path / "registro.jsonl"
    pausa_ruta = tmp_path / "PAUSA"
    P.poner_pausa(pausa_ruta, {"origen": "huella", "host": "atemai", "mision_id": MISION_ID})

    ruta_marca = H.ruta_huella(misiones, MISION_ID, "atemai")
    H.escribir_marca(ruta_marca, H.Marca(huella=_huella("atemai"), estado=H.REPORTADA, diff=("algo",)))

    async def tomar_falso(host):
        return _huella("atemai")

    def quitar_pausa_si_explota(ruta, *, coincide):
        raise RuntimeError("boom -- algo inesperado del sistema de archivos")

    import jax.ejecutor.contratos.huella as modulo
    original = modulo.pausa.quitar_pausa_si
    modulo.pausa.quitar_pausa_si = quitar_pausa_si_explota
    try:
        salidas = []
        rc = asyncio.run(H.aceptar(
            misiones=misiones, mision_id=MISION_ID, host="atemai", aceptado_por="fruiz",
            tomar_huella_actual=tomar_falso, registro_ruta=registro, pausa_ruta=pausa_ruta,
            salida=salidas.append))
    finally:
        modulo.pausa.quitar_pausa_si = original

    assert rc == 0  # la huella SÍ se aceptó -- no revienta
    assert any("pausa_no_verificable" in l for l in salidas)
    marca = H.leer_marca(ruta_marca)
    assert marca.estado == H.ABIERTA  # la marca quedó escrita igual
    assert any("huella_aceptada=true" in l for l in salidas)  # y llega hasta el final


# --- ronda 10, MAJOR: candado real entre procesos, reproduce el escenario de la
# auditoría 8 -- "B pasa el chequeo de inode, A borra, C5 pone su pausa y B hace
# unlink". Con el candado, la pausa de C5 sobrevive. -----------------------------------

def _tarea_aceptar_inmediata(misiones_str, mision_id, host, registro_str, pausa_str, resultado_dict):
    """Proceso A: acepta sin demora -- si la pausa original sigue en su lugar cuando
    le toca el turno, la borra de inmediato."""
    import asyncio
    from pathlib import Path
    from jax.ejecutor.contratos import huella as _H

    async def tomar_falso(h):
        return _H.huella_desde_salida(h, f"{_HASH_A}  /etc/sudoers\n".encode())

    rc = asyncio.run(_H.aceptar(
        misiones=Path(misiones_str), mision_id=mision_id, host=host, aceptado_por="fruiz",
        sin_medir=True, motivo="proceso A",
        tomar_huella_actual=tomar_falso, registro_ruta=Path(registro_str), pausa_ruta=Path(pausa_str)))
    resultado_dict["a_rc"] = rc


def _tarea_aceptar_con_demora_antes_del_unlink(misiones_str, mision_id, host, registro_str, pausa_str,
                                                b_reviso_evt, puede_borrar_evt, resultado_dict):
    """Proceso B: llega a pasar su propio chequeo de inodo (dentro de
    `quitar_pausa_si`, vía `aceptar`) y se queda esperando justo ANTES de llamar
    `unlink(ruta)` -- reproduce la ventana que describe la auditoría 8."""
    import asyncio
    import os as _os
    from pathlib import Path
    from jax.ejecutor.contratos import huella as _H

    ruta_pausa = Path(pausa_str)
    real_unlink = _os.unlink

    def unlink_con_demora(path, *a, **kw):
        if str(path) == str(ruta_pausa):
            b_reviso_evt.set()
            puede_borrar_evt.wait(timeout=10)
        return real_unlink(path, *a, **kw)

    _os.unlink = unlink_con_demora

    async def tomar_falso(h):
        return _H.huella_desde_salida(h, f"{_HASH_A}  /etc/sudoers\n".encode())

    rc = asyncio.run(_H.aceptar(
        misiones=Path(misiones_str), mision_id=mision_id, host=host, aceptado_por="fruiz",
        sin_medir=True, motivo="proceso B",
        tomar_huella_actual=tomar_falso, registro_ruta=Path(registro_str), pausa_ruta=Path(pausa_str)))
    resultado_dict["b_rc"] = rc


def test_race_real_c5_pone_pausa_mientras_dos_aceptar_compiten_sobrevive_con_candado(tmp_path):
    """MAJOR (ronda 10, auditoría 8): reproduce con DOS PROCESOS REALES el escenario
    exacto -- B pasa el chequeo de inodo, A borra la pausa original, C5 pone una pausa
    NUEVA, y B (que ya había pasado su chequeo) intenta hacer unlink. Con el candado
    (ya aplicado dentro de `aceptar()`, ronda 10), B no puede ni EMPEZAR su chequeo
    hasta que A termine del todo -- para cuando B actúa, ve el estado real, y para
    cuando C5 consigue pausar de nuevo, tanto A como B ya terminaron: su pausa nunca
    corre riesgo."""
    import multiprocessing as mp
    from jax.ejecutor.contratos import pausa as P

    misiones = tmp_path / "misiones"
    pausa_ruta = tmp_path / "PAUSA"
    registro_a = tmp_path / "registro-a.jsonl"
    registro_b = tmp_path / "registro-b.jsonl"
    host = "atemai"

    P.poner_pausa(pausa_ruta, {"origen": "huella", "host": host, "mision_id": MISION_ID})
    H.escribir_marca(H.ruta_huella(misiones, MISION_ID, host),
                     H.Marca(huella=_huella(host), estado=H.REPORTADA, diff=("algo cambió",)))

    ctx = mp.get_context("fork")
    manager = ctx.Manager()
    resultado = manager.dict()
    b_reviso = ctx.Event()
    puede_borrar = ctx.Event()

    b = ctx.Process(target=_tarea_aceptar_con_demora_antes_del_unlink,
                    args=(str(misiones), MISION_ID, host, str(registro_b), str(pausa_ruta),
                          b_reviso, puede_borrar, resultado))
    b.start()
    try:
        assert b_reviso.wait(timeout=10), "B no llegó a su chequeo de inodo"

        # Mientras B sostiene el candado (esperando adentro), A tiene que quedar
        # BLOQUEADO tratando de adquirirlo -- no puede ni empezar su propio chequeo.
        a = ctx.Process(target=_tarea_aceptar_inmediata,
                        args=(str(misiones), MISION_ID, host, str(registro_a), str(pausa_ruta), resultado))
        a.start()
        try:
            a.join(timeout=1)
            assert a.is_alive(), "A no debería poder avanzar mientras B sostiene el candado"

            # C5 tampoco puede pausar todavía: la pausa original SIGUE ahí (B no la
            # borró -- está esperando), y poner_pausa nunca pisa una existente.
            assert P.poner_pausa(pausa_ruta, {"origen": "c5", "motivo": "demasiado_pronto"}) is False

            # Se libera a B: borra la pausa ORIGINAL (la que de verdad vio), termina,
            # suelta el candado.
            puede_borrar.set()
            b.join(timeout=10)
            assert b.exitcode == 0
            assert resultado.get("b_rc") == 0

            # Ahora A, que estaba bloqueado, puede avanzar -- su propio chequeo ve que
            # ya no hay nada que borrar.
            a.join(timeout=10)
            assert a.exitcode == 0
            assert resultado.get("a_rc") == 0
        finally:
            if a.is_alive():
                a.terminate()
                a.join(timeout=5)

        # Recién AHORA, con A y B los dos terminados, C5 pausa de verdad.
        assert P.poner_pausa(pausa_ruta, {"origen": "c5", "motivo": "ya_libre"}) is True
        contenido = json.loads(pausa_ruta.read_text())
        assert contenido["origen"] == "c5"
        assert contenido["motivo"] == "ya_libre"
    finally:
        puede_borrar.set()
        if b.is_alive():
            b.terminate()
            b.join(timeout=5)


# --- MAJOR-K (ronda 7, auditoría adversarial 2026-09-22): la marca se leía AFUERA del
# candado que protege `_cuerpo()` -- dos `aceptar()` concurrentes sobre la MISMA marca
# REPORTADA podían leer los DOS "REPORTADA" antes de que ninguno tuviera ESE candado (el
# de `_cuerpo()`); el que llegaba SEGUNDO ahí seguía adelante con esa lectura vieja,
# escribiendo una SEGUNDA línea base y un segundo evento de C3 -- absorbiendo en
# silencio cualquier cambio ocurrido entre medio. Reproducido con DOS PROCESOS REALES
# (`flock` es por archivo/proceso, no por hilo -- un solo proceso con dos hilos NO
# probaría nada de esto).
#
# OJO -- `pausa.barrer_temporales_huerfanos()` (que `aceptar()` llama SIEMPRE primero)
# usa el MISMO archivo de candado que `_cuerpo()`, así que una pausa insertada DESPUÉS
# de esa llamada (ej. adentro de `tomar()`) queda re-serializada por el candado del
# barrido de todos modos, y el bug NO se reproduce -- hay que pausar exactamente en el
# momento de la LECTURA de la marca (vía un monkeypatch de `leer_marca`), que es lo que
# de verdad se movió de lugar en esta ronda. -------------------------------------------

def _tarea_primero_en_leer_major_k(misiones_str, mision_id, host, registro_str, pausa_str,
                                   huella_ok_bytes, llego_evt, puede_seguir_evt, resultado_dict):
    """Este proceso LEE la marca primero, pero queda pausado justo ahí -- ANTES de
    ronda 7, esa lectura vivía AFUERA del candado de `_cuerpo()`; después de ronda 7,
    vive DENTRO. El monkeypatch de `leer_marca` marca el momento exacto, sin importar
    de qué lado del candado esté esta vez."""
    import asyncio
    from pathlib import Path
    from jax.ejecutor.contratos import huella as _H

    real_leer_marca = _H.leer_marca

    def leer_marca_con_pausa(ruta):
        m = real_leer_marca(ruta)
        llego_evt.set()
        puede_seguir_evt.wait(timeout=10)
        return m

    _H.leer_marca = leer_marca_con_pausa

    async def tomar(h):
        return _H.huella_desde_salida(h, huella_ok_bytes)

    rc = asyncio.run(_H.aceptar(
        misiones=Path(misiones_str), mision_id=mision_id, host=host, aceptado_por="primero",
        tomar_huella_actual=tomar, registro_ruta=Path(registro_str), pausa_ruta=Path(pausa_str)))
    resultado_dict["primero_rc"] = rc


def _tarea_segundo_en_llegar_major_k(misiones_str, mision_id, host, registro_str, pausa_str,
                                     huella_ok_bytes, resultado_dict, salidas_list):
    """Arranca DESPUÉS de que el primero ya leyó y quedó pausado -- sin ronda 7, esto
    alcanza a leer, aceptar y escribir ANTES de que el primero se libere (por eso
    "segundo en llegar" puede terminar PRIMERO)."""
    import asyncio
    from pathlib import Path
    from jax.ejecutor.contratos import huella as _H

    async def tomar(h):
        return _H.huella_desde_salida(h, huella_ok_bytes)

    salidas = []
    rc = asyncio.run(_H.aceptar(
        misiones=Path(misiones_str), mision_id=mision_id, host=host, aceptado_por="segundo",
        tomar_huella_actual=tomar, registro_ruta=Path(registro_str), pausa_ruta=Path(pausa_str),
        salida=salidas.append))
    resultado_dict["segundo_rc"] = rc
    salidas_list.extend(salidas)


def test_dos_procesos_reales_solo_uno_acepta_la_reportada_major_k(tmp_path):
    """MAJOR-K: con la marca leída DENTRO del candado de `_cuerpo()`, dos `aceptar()`
    concurrentes sobre la MISMA `REPORTADA` dan UNA sola aceptación. `primero` lee la
    marca y queda pausado ahí (dentro del candado, desde ronda 7); `segundo` arranca
    recién entonces, no puede ni empezar (bloqueado en el candado del propio barrido
    de temporales, que comparte archivo con el de `_cuerpo()`), y sólo avanza cuando
    `primero` termina y suelta -- para entonces relee la marca YA `ABIERTA` y
    rechaza. Los dos procesos comparten el MISMO `registro_ruta`: si el bug
    reapareciera, aparecerían DOS eventos `huella_aceptada` ahí -- no uno."""
    ctx = multiprocessing.get_context("fork")
    misiones = tmp_path / "misiones"
    pausa_ruta = tmp_path / "PAUSA"
    registro = tmp_path / "registro.jsonl"
    host = "atemai"

    P.poner_pausa(pausa_ruta, {"origen": "huella", "motivo": "huella_cambio_no_declarado",
                               "host": host, "mision_id": MISION_ID, "detalle": []})
    H.escribir_marca(H.ruta_huella(misiones, MISION_ID, host),
                     H.Marca(huella=_huella(host), estado=H.REPORTADA, diff=("algo cambió",)))

    huella_ok_bytes = _base_completa_con_cambio()

    manager = ctx.Manager()
    resultado = manager.dict()
    salidas_segundo = manager.list()
    llego = ctx.Event()
    puede_seguir = ctx.Event()

    primero = ctx.Process(target=_tarea_primero_en_leer_major_k, args=(
        str(misiones), MISION_ID, host, str(registro), str(pausa_ruta), huella_ok_bytes,
        llego, puede_seguir, resultado))
    segundo = ctx.Process(target=_tarea_segundo_en_llegar_major_k, args=(
        str(misiones), MISION_ID, host, str(registro), str(pausa_ruta), huella_ok_bytes,
        resultado, salidas_segundo))
    primero.start()
    try:
        assert llego.wait(timeout=10), "el primer proceso no llegó a leer la marca"

        segundo.start()
        try:
            segundo.join(timeout=1)
            assert segundo.is_alive(), "el segundo no debería poder terminar mientras el primero sigue pausado"

            puede_seguir.set()  # se libera al primero: valida, acepta, escribe, suelta
            primero.join(timeout=10)
            assert primero.exitcode == 0

            segundo.join(timeout=10)
            assert segundo.exitcode == 0
        finally:
            if segundo.is_alive():
                segundo.terminate()
                segundo.join(timeout=5)
    finally:
        puede_seguir.set()
        if primero.is_alive():
            primero.terminate()
            primero.join(timeout=5)

    assert resultado.get("primero_rc") == 0  # el que leyó primero, acepta
    assert resultado.get("segundo_rc") == 2  # el que llegó segundo, NO acepta
    assert any("huella_no_reportada" in l for l in salidas_segundo), list(salidas_segundo)

    marca = H.leer_marca(H.ruta_huella(misiones, MISION_ID, host))
    assert marca.estado == H.ABIERTA
    assert marca.huella == _huella(host, controles=huella_ok_bytes)
    assert marca.aceptada_por == "primero"  # UNA sola transición -- "segundo" nunca escribió

    eventos = [json.loads(l) for l in registro.read_text().splitlines()]
    aceptaciones = [e for e in eventos if e.get("evento") == "huella_aceptada"]
    assert len(aceptaciones) == 1  # un solo evento de C3 -- no dos
    assert aceptaciones[0]["aceptado_por"] == "primero"
