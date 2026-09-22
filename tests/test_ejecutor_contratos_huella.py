# tests/test_ejecutor_contratos_huella.py
"""Integridad de los propios controles del Ejecutor, por ESTADO (M-1/M-2/B-1, ronda 4,
auditoría adversarial 2026-09-22): "El log de sudo se juzga por ser solo-añadido, no por
el tamaño"; huella en DOS niveles (CONTROLES pausa, PERSISTENCIA sólo informa); lo
"declarado" se compara por RUTA EXACTA, no por cualquier fragmento de la línea."""
import pytest

from jax.ejecutor.contratos import huella as H


# --- comandos: dos niveles, listas LITERALES (M-3: no contra sí mismas) ------------------

def test_rutas_controles_es_exactamente_la_lista_esperada():
    # Lista LITERAL -- si alguien quita /etc/sudoers.d de RUTAS_CONTROLES esto revienta,
    # a diferencia de comparar la tupla contra sí misma.
    assert H.RUTAS_CONTROLES == (
        "/etc/sudoers",
        "/etc/sudoers.d",
        "/etc/ssh/sshd_config",
        "/etc/ssh/sshd_config.d",
        "/etc/ssh/authorized_keys.d",
        "/root/.ssh/authorized_keys",
        "/usr/local/sbin/ejecutor-freno-remoto",
        "/usr/local/sbin/ejecutor-revocar",
        "/var/lib/systemd/linger",
        "/etc/cron.deny",
        "/etc/at.deny",
        "/etc/crontab",
        "/etc/passwd",
        "/etc/group",
        "/etc/shadow",
    )


def test_rutas_persistencia_es_exactamente_la_lista_esperada():
    assert H.RUTAS_PERSISTENCIA == (
        "/etc/systemd/system",
        "/etc/cron.d",
        "/var/spool/cron/crontabs",
    )


def test_quitar_sudoers_d_de_controles_rompe_el_test_literal():
    # Prueba que la aserción de arriba SÍ mata ese mutante (no es tautológica).
    mutado = tuple(r for r in H.RUTAS_CONTROLES if r != "/etc/sudoers.d")
    assert mutado != H.RUTAS_CONTROLES
    assert "/etc/sudoers.d" not in mutado


def test_comando_apertura_menciona_las_rutas_de_los_dos_niveles():
    cmd = H.comando_apertura("axioma")
    for ruta in H.RUTAS_CONTROLES + H.RUTAS_PERSISTENCIA:
        assert ruta in cmd, ruta
    assert "/var/log/sudo-axioma.log" in cmd
    assert "sudo-io" in cmd
    assert "atq" in cmd


def test_comando_cierre_pide_los_primeros_n_bytes_del_log():
    cmd = H.comando_cierre("axioma", tamano_apertura_log=12345)
    assert "12345" in cmd
    assert "head -c" in cmd


@pytest.mark.parametrize("cuenta", ["axioma; rm -rf /", "../etc", "con espacio", ""])
def test_comando_rechaza_una_cuenta_ilegible(cuenta):
    with pytest.raises(ValueError):
        H.comando_apertura(cuenta)
    with pytest.raises(ValueError):
        H.comando_cierre(cuenta, tamano_apertura_log=1)


# --- B-1: el log de sudo se juzga por ser solo-añadido, no por el tamaño ------------------

def _salida_apertura(inode="100", tamano="500", sha_completo="a" * 64, extra=b""):
    return (f"LOG /var/log/sudo-axioma.log {inode} {tamano} {sha_completo}\n".encode()
            + b"===CONTROLES===\n" + extra + b"===PERSISTENCIA===\n===FIN===\n")


def _salida_cierre(inode="100", tamano="500", sha_primeros_n="a" * 64, extra=b""):
    return (f"LOG /var/log/sudo-axioma.log {inode} {tamano} {sha_primeros_n}\n".encode()
            + b"===CONTROLES===\n" + extra + b"===PERSISTENCIA===\n===FIN===\n")


def test_log_crece_mismo_inode_y_prefijo_igual_es_legitimo():
    apertura = H.huella_de_apertura_desde_salida("atemai", _salida_apertura(tamano="500", sha_completo="f" * 64))
    cierre = H.huella_de_cierre_desde_salida("atemai", _salida_cierre(tamano="900", sha_primeros_n="f" * 64))
    assert H.log_intacto(apertura.log, cierre.log) is True


def test_log_truncado_mismo_tamano_pero_prefijo_distinto_no_es_legitimo():
    apertura = H.huella_de_apertura_desde_salida("atemai", _salida_apertura(tamano="500", sha_completo="f" * 64))
    cierre = H.huella_de_cierre_desde_salida("atemai", _salida_cierre(tamano="500", sha_primeros_n="0" * 64))
    assert H.log_intacto(apertura.log, cierre.log) is False


def test_log_con_inode_distinto_no_es_legitimo_aunque_el_prefijo_cuadre():
    apertura = H.huella_de_apertura_desde_salida("atemai", _salida_apertura(inode="100", sha_completo="f" * 64))
    cierre = H.huella_de_cierre_desde_salida("atemai", _salida_cierre(inode="200", sha_primeros_n="f" * 64))
    assert H.log_intacto(apertura.log, cierre.log) is False


def test_log_que_encoge_no_es_legitimo():
    apertura = H.huella_de_apertura_desde_salida("atemai", _salida_apertura(tamano="500", sha_completo="f" * 64))
    cierre = H.huella_de_cierre_desde_salida("atemai", _salida_cierre(tamano="10", sha_primeros_n="f" * 64))
    assert H.log_intacto(apertura.log, cierre.log) is False


def test_log_ausente_en_la_apertura_o_el_cierre_no_es_medible():
    apertura = H.huella_de_apertura_desde_salida("atemai", b"===CONTROLES===\n===PERSISTENCIA===\n===FIN===\n")
    assert apertura.log is None
    cierre = H.huella_de_cierre_desde_salida("atemai", b"===CONTROLES===\n===PERSISTENCIA===\n===FIN===\n")
    assert cierre.log is None


# --- M-2: rutas exactas, no fragmentos; symlinks por destino Y contenido resuelto --------

def test_declarado_exige_la_ruta_exacta_no_cualquier_fragmento_de_la_linea():
    # La línea tiene DOS rutas (el symlink y su destino). La misión menciona el DESTINO,
    # pero lo que hay que declarar es la ruta que CAMBIÓ (el symlink), no su destino.
    antes = H.huella_de_apertura_desde_salida(
        "atemai", b"===CONTROLES===\n===PERSISTENCIA===\n===FIN===\n")
    despues = H.huella_de_apertura_desde_salida(
        "atemai", b"===CONTROLES===\nL /root/.ssh/authorized_keys -> /tmp/otra-cosa\n===PERSISTENCIA===\n===FIN===\n")
    hallazgos = H.hallazgos_controles(antes, despues, "Cambié /tmp/otra-cosa nada más")
    assert hallazgos and "/root/.ssh/authorized_keys" in hallazgos[0]


def test_declarado_con_la_ruta_exacta_si_blanquea():
    log = b"LOG /var/log/sudo-axioma.log 1 1 " + b"a" * 64 + b"\n"
    antes = H.huella_de_apertura_desde_salida("atemai", log + b"===CONTROLES===\n===PERSISTENCIA===\n===FIN===\n")
    despues = H.huella_de_apertura_desde_salida(
        "atemai", log + b"===CONTROLES===\nabc  /etc/crontab\n===PERSISTENCIA===\n===FIN===\n")
    hallazgos = H.hallazgos_controles(antes, despues, "Voy a tocar /etc/crontab para el cliente")
    assert hallazgos == ()


# --- persistencia informa, NUNCA pausa -----------------------------------------------------

def test_persistencia_cambiada_no_declarada_se_reporta_pero_nunca_pausa():
    log = b"LOG /var/log/sudo-axioma.log 1 1 " + b"a" * 64 + b"\n"
    antes = H.huella_de_apertura_desde_salida("atemai", log + b"===CONTROLES===\n===PERSISTENCIA===\n===FIN===\n")
    despues = H.huella_de_apertura_desde_salida(
        "atemai", log + b"===CONTROLES===\n===PERSISTENCIA===\nabc  /etc/systemd/system/cliente.service\n===FIN===\n")
    assert H.hallazgos_controles(antes, despues, "") == ()
    informe = H.hallazgos_persistencia(antes, despues, "")
    assert informe and "cliente.service" in informe[0]


# --- M-3: try/except que traga la apertura tiene que fallar -------------------------------

def test_apertura_que_traga_una_excepcion_es_un_mutante_que_debe_morir(tmp_path):
    """Si `huella_de_apertura_de_la_mision` estuviera envuelta en un try/except que
    devuelve {}, una máquina sin huella real pasaría como "sin cambios" -- eso es
    exactamente lo que NO puede pasar."""
    import asyncio

    from jax.ejecutor.contratos import vigia_servicio as V

    async def tomar_que_revienta(host):
        raise RuntimeError("ssh caído")

    with pytest.raises(RuntimeError):
        asyncio.run(V.huella_de_apertura_de_la_mision(
            misiones=tmp_path, mision_id="11111111-1111-1111-1111-111111111111", host="atemai",
            tomar_huella=tomar_que_revienta))
