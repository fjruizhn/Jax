# tests/test_ejecutor_contratos_huella.py
"""Integridad de los propios controles del Ejecutor, por ESTADO (ronda 6/7, auditoría
adversarial 2026-09-22 -- «la huella solo mide lo que NO debe cambiar nunca durante una
misión»): sudoers, sshd_config, authorized_keys.d/root authorized_keys, los binarios
`ejecutor-*`. Sin declarado: cualquier cambio pausa.

Ronda 7 (BLOCK reproducido): `/etc/passwd`/`group`/`shadow` SALEN de la huella --
`apt install` y el plugin de correo de aaPanel crean cuentas de sistema, y eso es
administración LEGÍTIMA del cutover, no un ataque a los controles del Ejecutor. Una
cuenta nueva con sudo real pasa igual por `sudoers.d`, que sí se mide."""
import pytest

from jax.ejecutor.contratos import huella as H


def test_rutas_controles_es_exactamente_la_lista_esperada():
    assert H.RUTAS_CONTROLES == (
        "/etc/sudoers",
        "/etc/sudoers.d",
        "/etc/ssh/sshd_config",
        "/etc/ssh/sshd_config.d",
        "/etc/ssh/authorized_keys.d",
        "/root/.ssh/authorized_keys",
    )


def test_no_hay_rutas_sha_solo_passwd_group_shadow_salieron_ronda7():
    """Ronda 7: crear cuentas de sistema (apt install, el plugin de correo de aaPanel)
    es administración legítima -- ya no está en la huella."""
    assert not hasattr(H, "RUTAS_SHA_SOLO")


def test_quitar_sudoers_d_rompe_el_test_literal():
    mutado = tuple(r for r in H.RUTAS_CONTROLES if r != "/etc/sudoers.d")
    assert mutado != H.RUTAS_CONTROLES


def test_no_hay_mas_nivel_persistencia():
    assert not hasattr(H, "RUTAS_PERSISTENCIA")
    assert not hasattr(H, "hallazgos_persistencia")


def test_comando_huella_menciona_las_rutas_y_el_glob_de_ejecutor():
    cmd = H.comando_huella()
    for ruta in H.RUTAS_CONTROLES:
        assert ruta in cmd, ruta
    assert "ejecutor-*" in cmd
    assert "/usr/local/sbin" in cmd


def test_comando_huella_no_mide_passwd_group_shadow_ronda7():
    cmd = H.comando_huella()
    for ruta in ("/etc/passwd", "/etc/group", "/etc/shadow"):
        assert ruta not in cmd, ruta


def test_comando_huella_usa_rutas_absolutas_no_el_path():
    """LÍMITE (ronda 6): sin `secure_path`, un root podría poner un `sha256sum` falso
    antes en el PATH. El comando usa binarios por ruta absoluta -- no ata la seguridad
    del contrato a que nadie haya tocado el PATH del shell remoto. `find -printf %l` da
    el destino de un symlink sin un `readlink` aparte."""
    cmd = H.comando_huella()
    for binario in ("/usr/bin/find", "/usr/bin/sha256sum", "/usr/bin/sort"):
        assert binario in cmd, binario
    import re
    for nombre in ("find", "sha256sum", "sort"):
        for m in re.finditer(rf"(?<![\w/.]){nombre}\b", cmd):
            assert cmd[:m.start()].endswith("/usr/bin/"), \
                f"{nombre!r} suelto en el comando (sin ruta absoluta): ...{cmd[max(0, m.start()-20):m.start()+20]}..."


def test_huella_no_tiene_log_de_sudo_ni_atq_ni_sudo_io():
    cmd = H.comando_huella()
    assert "sudo-io" not in cmd
    assert "sudo-" not in cmd  # ni /var/log/sudo-<cuenta>.log
    assert "atq" not in cmd
    assert "systemd" not in cmd
    assert "cron" not in cmd


def test_comando_huella_no_pide_una_cuenta():
    """El comando ya no depende del nombre de la cuenta (B-1 se fue: no hay log de
    sudo que mirar) -- ronda 6."""
    import inspect
    firma = inspect.signature(H.comando_huella)
    assert list(firma.parameters) == []


# --- estado, comparación: cualquier cambio es hallazgo, sin declarado -------------------

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


def test_hallazgos_sin_declarado_cualquier_cambio_es_hallazgo():
    antes = H.huella_desde_salida("atemai", b"abc  /etc/sudoers.d/50-ejecutor-axioma-registro\n")
    despues = H.huella_desde_salida(
        "atemai", b"abc  /etc/sudoers.d/50-ejecutor-axioma-registro\n"
                  b"def  /root/.ssh/authorized_keys\n")
    hallazgos = H.hallazgos(antes, despues)
    assert hallazgos == ("def  /root/.ssh/authorized_keys",)


def test_hallazgos_no_toma_un_texto_de_mision():
    """La firma ya no acepta `texto_mision`: no hay declarado (B-2 se fue, ronda 6)."""
    import inspect
    firma = inspect.signature(H.hallazgos)
    assert list(firma.parameters) == ["antes", "despues"]


def test_hallazgos_vacio_sin_cambios():
    salida = b"abc  /etc/sudoers.d/50-ejecutor-axioma-registro\n"
    antes = H.huella_desde_salida("atemai", salida)
    despues = H.huella_desde_salida("atemai", salida)
    assert H.hallazgos(antes, despues) == ()


# --- MINOR: huella vacía o sin parsear es fallo (ronda 6) --------------------------------

def test_huella_vacia_no_es_valida():
    assert H.huella_valida(H.huella_desde_salida("atemai", b"")) is False


def test_huella_con_contenido_es_valida():
    assert H.huella_valida(H.huella_desde_salida("atemai", b"abc  /etc/sudoers\n")) is True


# --- M-1, ronda 7: estados de la marca (abierta / reportada / cerrada) -----------------

def test_estados_son_los_tres_esperados():
    assert H.ABIERTA == "abierta"
    assert H.REPORTADA == "reportada"
    assert H.CERRADA == "cerrada"


def test_ruta_huella_valida_el_mision_id():
    with pytest.raises(H.MisionIdInvalido):
        H.ruta_huella("/tmp/misiones", "no-es-un-uuid", "atemai")


def test_ruta_huella_valida_el_host():
    uuid = "11111111-1111-1111-1111-111111111111"
    with pytest.raises(ValueError):
        H.ruta_huella("/tmp/misiones", uuid, "../etc")


def test_escribir_y_leer_marca_redondea(tmp_path):
    h = H.huella_desde_salida("atemai", b"abc  /etc/sudoers\n")
    m = H.Marca(huella=h, estado=H.ABIERTA)
    ruta = tmp_path / "m.json"
    H.escribir_marca(ruta, m)
    leida = H.leer_marca(ruta)
    assert leida == m


def test_marca_reportada_lleva_el_diff():
    h = H.huella_desde_salida("atemai", b"abc  /etc/sudoers\n")
    m = H.Marca(huella=h, estado=H.REPORTADA, diff=("def  /root/.ssh/authorized_keys",))
    assert m.diff == ("def  /root/.ssh/authorized_keys",)


def test_marca_aceptada_registra_quien_y_cuando(tmp_path):
    h = H.huella_desde_salida("atemai", b"abc  /etc/sudoers\n")
    m = H.Marca(huella=h, estado=H.ABIERTA, aceptada_por="fruiz", aceptada_en="2026-09-22T10:00:00+00:00")
    ruta = tmp_path / "m.json"
    H.escribir_marca(ruta, m)
    leida = H.leer_marca(ruta)
    assert leida.aceptada_por == "fruiz"
    assert leida.aceptada_en == "2026-09-22T10:00:00+00:00"


# --- la CLI: `python -m jax.ejecutor.contratos.huella aceptar ...` ---------------------

def test_principal_sin_variables_de_entorno_falla_con_codigo(monkeypatch, capsys):
    monkeypatch.delenv("JAX_EJECUTOR_MISIONES", raising=False)
    rc = H.principal(["aceptar", "--host", "atemai", "--mision", "11111111-1111-1111-1111-111111111111"])
    assert rc == 2
    assert "sin_configurar" in capsys.readouterr().err


def test_principal_sin_marca_no_toca_la_red(tmp_path, monkeypatch, capsys):
    """Sin marca que aceptar, `principal()` no debería siquiera intentar ssh --
    `aceptar()` corta apenas no encuentra el archivo."""
    monkeypatch.setenv("JAX_EJECUTOR_MISIONES", str(tmp_path / "misiones"))
    monkeypatch.setenv("JAX_EJECUTOR_ADMIN_USUARIO", "fruiz")
    monkeypatch.setenv("JAX_EJECUTOR_REGISTRO", str(tmp_path / "registro.jsonl"))
    monkeypatch.setenv("JAX_EJECUTOR_POLITICA", str(tmp_path / "politica.json"))
    monkeypatch.setenv("JAX_EJECUTOR_PAUSA", str(tmp_path / "PAUSA"))
    rc = H.principal(["aceptar", "--host", "fantasma", "--mision", "11111111-1111-1111-1111-111111111111"])
    assert rc == 2
    assert "huella_no_encontrada" in capsys.readouterr().out


def test_principal_exige_host_y_mision():
    with pytest.raises(SystemExit):
        H.principal(["aceptar"])


def test_principal_sin_medir_es_una_bandera_reconocida(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JAX_EJECUTOR_MISIONES", str(tmp_path / "misiones"))
    monkeypatch.setenv("JAX_EJECUTOR_ADMIN_USUARIO", "fruiz")
    monkeypatch.setenv("JAX_EJECUTOR_REGISTRO", str(tmp_path / "registro.jsonl"))
    monkeypatch.setenv("JAX_EJECUTOR_POLITICA", str(tmp_path / "politica.json"))
    monkeypatch.setenv("JAX_EJECUTOR_PAUSA", str(tmp_path / "PAUSA"))
    rc = H.principal(["aceptar", "--host", "fantasma", "--mision", "11111111-1111-1111-1111-111111111111",
                      "--sin-medir"])
    assert rc == 2  # sigue sin marca -- pero no revienta por la bandera
