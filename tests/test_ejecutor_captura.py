import pytest
from jax.ejecutor.captura import correr, a_captura


def test_captura_la_salida_y_el_codigo():
    c = correr("echo hola", maquina="local")
    assert c.codigo == 0
    assert "hola" in c.salida
    assert c.truncada is False
    assert c.maquina == "local"


def test_un_comando_que_falla_no_lanza_y_queda_registrado():
    c = correr("exit 3", maquina="local")
    assert c.codigo == 3
    assert c.truncada is False


def test_pasado_el_tope_la_captura_queda_MARCADA_como_truncada():
    """El truncado se marca, no se esconde: es el dato del que depende la
    regla de §2.4 del spec."""
    c = correr("seq 1 100000", maquina="local", tope_bytes=1024)
    assert c.truncada is True
    assert c.bytes_totales > 1024
    assert len(c.salida.encode()) <= 1024


def test_a_captura_conserva_la_marca_de_truncado():
    c = correr("seq 1 100000", maquina="local", tope_bytes=1024)
    assert a_captura(c).truncada is True


# --- Agregados 2026-09-16 al implementar la Task 2 (no estaban en el plan) ---

import subprocess
import time
import uuid

from jax.ejecutor.captura import TRUNCADO_POR_TIMEOUT, TRUNCADO_POR_TOPE
from jax.ejecutor.cita import Afirmacion, FUENTE_TRUNCADA, verificar


def test_una_salida_que_no_es_utf8_se_captura_igual_y_sin_inventar_bytes():
    """Hueco del plan: con `text=True` un solo byte inválido hacía LANZAR a
    `correr` (UnicodeDecodeError) y se perdía la captura entera, que es el
    piso de §2.3. Y decodificar con `errors="ignore"` tampoco sirve: borra el
    byte y pega lo de los lados -- `12\\xff4` pasaría a ser `124`, un número
    que la máquina nunca imprimió y que después sería citable."""
    c = correr(r"printf 'ok\n12\3774\n'", maquina="local")
    assert c.codigo == 0
    assert c.salida.splitlines()[0] == "ok"
    assert "124" not in c.salida
    assert c.bytes_totales == len(b"ok\n12\xff4\n")


def test_la_salida_es_literal_y_los_bytes_son_los_reales():
    """Hueco del plan: `text=True` traduce `\\r\\n` a `\\n`, así que la
    salida ya no era literal y `bytes_totales` mentía (4 en vez de 6)."""
    c = correr(r"printf 'a\r\nb\r\n'", maquina="local")
    assert c.salida == "a\r\nb\r\n"
    assert c.bytes_totales == 6


def test_un_comando_colgado_vuelve_al_vencer_el_plazo_marcado_como_truncado():
    """Hueco del plan: `correr` no tenía plazo. Un comando que no termina
    (un apt esperando el lock, un ssh pidiendo clave) colgaba al Ejecutor
    para siempre. Al vencer, la salida está incompleta: es truncada."""
    t0 = time.monotonic()
    c = correr("echo antes; sleep 30; echo despues", maquina="local", timeout_s=0.5)
    assert time.monotonic() - t0 < 5
    assert c.truncada is True
    assert TRUNCADO_POR_TIMEOUT in c.motivos_truncado
    assert c.codigo != 0
    # Lo que salió antes de vencer se entrega (piso de §2.3)...
    assert c.salida.splitlines() == ["antes"]
    # ...pero no respalda nada, porque la salida no está completa.
    a = Afirmacion(maquina=c.maquina, comando=c.comando, linea="antes", dato="antes")
    assert verificar(a, [a_captura(c)]).estado == FUENTE_TRUNCADA


def test_al_vencer_el_plazo_no_quedan_procesos_huerfanos():
    """`subprocess.run(timeout=...)` con `shell=True` mata al shell y deja
    vivo al nieto (medido 2026-09-16: `sleep` seguía corriendo). Un apt
    huérfano sigue con el lock de dpkg tomado.

    La marca es única por corrida: con una fija, un huérfano de una corrida
    anterior (o de otra suite en paralelo) pone este test en rojo sin que el
    código de ahora tenga la culpa -- pasó al ejercitarlo por mutación."""
    marca = f"sleep 31.{uuid.uuid4().int % 10**8:08d}"
    correr(f"{marca}; echo fin", maquina="local", timeout_s=0.3)
    time.sleep(0.3)
    vivos = subprocess.run(["pgrep", "-f", marca], capture_output=True, text=True).stdout
    assert vivos == ""


def test_la_captura_dice_POR_QUE_se_trunco():
    """Spec §3.2: «si se truncó y por qué». El plan no tenía el porqué."""
    completa = correr("echo hola", maquina="local")
    assert completa.motivos_truncado == ()
    por_tope = correr("seq 1 100000", maquina="local", tope_bytes=1024)
    assert por_tope.motivos_truncado == (TRUNCADO_POR_TOPE,)


def test_un_stderr_que_pasa_el_tope_tambien_trunca_la_captura():
    """Si stderr se corta, la captura no está completa: se marca. Del lado
    seguro, aunque stdout haya llegado entero."""
    c = correr("seq 1 100000 1>&2; echo ok", maquina="local", tope_bytes=1024)
    assert c.salida == "ok\n"
    assert c.truncada is True
    assert c.motivos_truncado == (TRUNCADO_POR_TOPE,)
    assert c.bytes_totales_stderr > 1024
    assert len(c.stderr.encode()) <= 1024


# --- Contrato 2026-09-16: stderr citable y máquina en la procedencia ---

from jax.ejecutor.cita import FUENTE_INEXISTENTE, RESPALDADA


def test_a_captura_conserva_la_maquina_y_el_stderr():
    c = correr("echo afuera; echo 'sudo: a password is required' 1>&2", maquina="hall9000")
    corta = a_captura(c)
    assert corta.maquina == "hall9000"
    assert corta.stderr == "sudo: a password is required\n"
    a = Afirmacion(maquina="hall9000", comando=c.comando,
                   linea="sudo: a password is required", dato="a password is required")
    assert verificar(a, [corta]).estado == RESPALDADA
    otra = Afirmacion(maquina="atemai", comando=c.comando,
                      linea="sudo: a password is required", dato="a password is required")
    assert verificar(otra, [corta]).estado == FUENTE_INEXISTENTE


def test_un_stderr_truncado_no_deja_citar_ni_lo_que_llegó_entero_por_stdout():
    """La captura está incompleta si CUALQUIERA de los dos flujos se cortó:
    se rechaza antes de mirar el contenido, aunque la línea esté en stdout."""
    c = correr("seq 1 100000 1>&2; echo ok", maquina="local", tope_bytes=1024)
    a = Afirmacion(maquina="local", comando=c.comando, linea="ok", dato="ok")
    assert verificar(a, [a_captura(c)]).estado == FUENTE_TRUNCADA
