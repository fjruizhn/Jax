# tests/test_ejecutor_contratos_huella.py
"""M-1/M-2 (ronda 3, auditoría adversarial 2026-09-22): integridad de los propios
controles del Ejecutor por ESTADO (huella sha256+listado+atributos), no por texto de
comando. Reemplaza la parte léxica de M-1/M-2 que la ronda 2 intentaba con más
regex -- "detectar, no adivinar"."""
import pytest

from jax.ejecutor.contratos import huella as H


def test_comando_huella_incluye_las_rutas_declaradas():
    cmd = H.comando_huella("axioma")
    for ruta in H.RUTAS_A_VIGILAR:
        assert ruta in cmd, ruta
    assert "/var/log/sudo-axioma.log" in cmd
    assert "lsattr" in cmd


@pytest.mark.parametrize("cuenta", ["axioma; rm -rf /", "../etc", "con espacio", ""])
def test_comando_huella_rechaza_una_cuenta_ilegible(cuenta):
    with pytest.raises(ValueError):
        H.comando_huella(cuenta)


def test_huella_desde_salida_es_deterministica():
    salida = b"abc123  /etc/sudoers.d/50-ejecutor-axioma-registro\n"
    h1 = H.huella_desde_salida("atemai", salida)
    h2 = H.huella_desde_salida("atemai", salida)
    assert h1 == h2
    assert h1.host == "atemai"


def test_cambio_detecta_una_linea_de_mas():
    antes = H.huella_desde_salida("atemai", b"abc123  /etc/sudoers.d/50-ejecutor-axioma-registro\n")
    despues = H.huella_desde_salida(
        "atemai", b"abc123  /etc/sudoers.d/50-ejecutor-axioma-registro\n"
                  b"def456  /etc/sudoers.d/99-otra-cosa\n")
    assert H.cambio(antes, despues) is True


def test_sin_cambio_da_false():
    salida = b"abc123  /etc/sudoers.d/50-ejecutor-axioma-registro\n"
    antes = H.huella_desde_salida("atemai", salida)
    despues = H.huella_desde_salida("atemai", salida)
    assert H.cambio(antes, despues) is False


def test_cambio_exige_la_misma_maquina():
    antes = H.huella_desde_salida("atemai", b"x")
    despues = H.huella_desde_salida("bridge", b"x")
    with pytest.raises(ValueError):
        H.cambio(antes, despues)


def test_lineas_agregadas_o_quitadas():
    antes = H.huella_desde_salida("atemai", b"linea-a\nlinea-b\n")
    despues = H.huella_desde_salida("atemai", b"linea-a\nlinea-c\n")
    assert H.lineas_agregadas_o_quitadas(antes, despues) == ("linea-b", "linea-c")


def test_cambio_declarado_por_mencion_literal_en_la_mision():
    linea = "abc123  /etc/systemd/system/mi-servicio-nuevo.service"
    assert H.cambio_declarado(linea, "Crear /etc/systemd/system/mi-servicio-nuevo.service en atemai") is True
    assert H.cambio_declarado(linea, "Revisar el estado de nginx") is False
    assert H.cambio_declarado(linea, "") is False


def test_hallazgos_vacio_sin_cambios():
    salida = b"abc123  /etc/sudoers.d/50-ejecutor-axioma-registro\n"
    antes = H.huella_desde_salida("atemai", salida)
    despues = H.huella_desde_salida("atemai", salida)
    assert H.hallazgos(antes, despues, "cualquier misión") == ()


def test_hallazgos_reporta_lo_no_declarado():
    antes = H.huella_desde_salida("atemai", b"abc123  /etc/sudoers.d/50-ejecutor-axioma-registro\n")
    despues = H.huella_desde_salida(
        "atemai", b"abc123  /etc/sudoers.d/50-ejecutor-axioma-registro\n"
                  b"def456  /root/.ssh/authorized_keys\n")
    hallazgos = H.hallazgos(antes, despues, "Instalar una skill nueva")
    assert hallazgos == ("def456  /root/.ssh/authorized_keys",)


def test_hallazgos_no_reporta_lo_que_la_mision_declaro():
    antes = H.huella_desde_salida("atemai", b"abc123  /etc/sudoers.d/50-ejecutor-axioma-registro\n")
    despues = H.huella_desde_salida(
        "atemai", b"abc123  /etc/sudoers.d/50-ejecutor-axioma-registro\n"
                  b"def456  /etc/systemd/system/mi-servicio.service\n")
    mision = "Crear la unidad /etc/systemd/system/mi-servicio.service para el cliente en atemai"
    assert H.hallazgos(antes, despues, mision) == ()


def test_hallazgos_mixto_solo_reporta_lo_no_declarado():
    antes = H.huella_desde_salida("atemai", b"")
    despues = H.huella_desde_salida(
        "atemai", b"def456  /etc/systemd/system/mi-servicio.service\n"
                  b"ghi789  /etc/sudoers.d/50-ejecutor-axioma-registro\n")
    mision = "Crear la unidad /etc/systemd/system/mi-servicio.service para el cliente"
    hallazgos = H.hallazgos(antes, despues, mision)
    assert hallazgos == ("ghi789  /etc/sudoers.d/50-ejecutor-axioma-registro",)
