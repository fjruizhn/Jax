# tests/test_ejecutor_contratos_huella.py
"""Integridad de los propios controles del Ejecutor, por ESTADO (ronda 6/7, auditoría
adversarial 2026-09-22 -- «la huella solo mide lo que NO debe cambiar nunca durante una
misión»): sudoers, sshd_config, authorized_keys.d/root authorized_keys, los binarios
`ejecutor-*`. Sin declarado: cualquier cambio pausa.

Ronda 7 (BLOCK reproducido): `/etc/passwd`/`group`/`shadow` SALEN de la huella --
`apt install` y el plugin de correo de aaPanel crean cuentas de sistema, y eso es
administración LEGÍTIMA del cutover, no un ataque a los controles del Ejecutor. Una
cuenta nueva con sudo real pasa igual por `sudoers.d`, que sí se mide."""
import json
import os
import pwd

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
    cmd = H.comando_huella("fruiz")
    for ruta in H.RUTAS_CONTROLES:
        assert ruta in cmd, ruta
    assert "ejecutor-*" in cmd
    assert "/usr/local/sbin" in cmd


def test_comando_huella_menciona_el_authorized_keys_del_administrador():
    """LÍMITE 9 (ronda 2): el authorized_keys del administrador -- donde vive el acceso
    privilegiado real, y donde este mismo arreglo pone la llave del servicio -- entra
    a la huella. Con OTRO admin_usuario, la ruta medida cambia con él."""
    assert "/home/fruiz/.ssh/authorized_keys" in H.comando_huella("fruiz")
    assert "/home/otro-admin/.ssh/authorized_keys" in H.comando_huella("otro-admin")
    assert "/home/fruiz/.ssh/authorized_keys" not in H.comando_huella("otro-admin")


def test_comando_huella_no_mide_passwd_group_shadow_ronda7():
    cmd = H.comando_huella("fruiz")
    for ruta in ("/etc/passwd", "/etc/group", "/etc/shadow"):
        assert ruta not in cmd, ruta


def test_comando_huella_usa_rutas_absolutas_no_el_path():
    """LÍMITE (ronda 6): sin `secure_path`, un root podría poner un `sha256sum` falso
    antes en el PATH. El comando usa binarios por ruta absoluta -- no ata la seguridad
    del contrato a que nadie haya tocado el PATH del shell remoto. `find -printf %l` da
    el destino de un symlink sin un `readlink` aparte."""
    cmd = H.comando_huella("fruiz")
    for binario in ("/usr/bin/find", "/usr/bin/sha256sum", "/usr/bin/sort"):
        assert binario in cmd, binario
    import re
    for nombre in ("find", "sha256sum", "sort"):
        for m in re.finditer(rf"(?<![\w/.]){nombre}\b", cmd):
            assert cmd[:m.start()].endswith("/usr/bin/"), \
                f"{nombre!r} suelto en el comando (sin ruta absoluta): ...{cmd[max(0, m.start()-20):m.start()+20]}..."


def test_huella_no_tiene_log_de_sudo_ni_atq_ni_sudo_io():
    cmd = H.comando_huella("fruiz")
    assert "sudo-io" not in cmd
    assert "sudo-" not in cmd  # ni /var/log/sudo-<cuenta>.log
    assert "atq" not in cmd
    assert "systemd" not in cmd
    assert "cron" not in cmd


def test_comando_huella_ahora_pide_el_admin_usuario_ronda2():
    """Ronda 2 (LÍMITE 9): esto REEMPLAZA a `test_comando_huella_no_pide_una_cuenta`
    (ronda 6) -- esa premisa dejó de ser cierta el día que la huella tuvo que empezar a
    vigilar SU PROPIA llave de acceso, que vive en el `authorized_keys` de una cuenta
    concreta (`ruta_authorized_keys_admin`). Sin `admin_usuario`, `comando_huella()` no
    sabría qué ruta agregar."""
    import inspect
    firma = inspect.signature(H.comando_huella)
    assert list(firma.parameters) == ["admin_usuario"]
    with pytest.raises(TypeError):
        H.comando_huella()


def test_ruta_authorized_keys_admin_valida_el_nombre():
    assert H.ruta_authorized_keys_admin("fruiz") == "/home/fruiz/.ssh/authorized_keys"
    with pytest.raises(ValueError):
        H.ruta_authorized_keys_admin("../etc")
    with pytest.raises(ValueError):
        H.ruta_authorized_keys_admin("")


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


def test_huella_con_un_hash_real_es_valida():
    hash64 = "a" * 64
    salida = f"{hash64}  /etc/sudoers\n".encode()
    assert H.huella_valida(H.huella_desde_salida("atemai", salida)) is True


def test_huella_solo_con_lineas_d_o_l_no_es_valida_ronda2_major6():
    """MAJOR-6 (ronda 2, auditoría adversarial 2026-09-22): si `sha256sum` faltara en
    la remota, `find -exec sha256sum` falla en silencio pero los OTROS `find` del
    mismo tramo (listado de directorios/symlinks) NO dependen de `sha256sum` y siguen
    produciendo líneas -- "no vacía" no bastaba. Sin NINGÚN hash real, es lo mismo que
    no medible."""
    salida = b"D /etc/sudoers.d\nD /etc/ssh/sshd_config.d\nL /algo -> /otro\n"
    assert H.huella_valida(H.huella_desde_salida("atemai", salida)) is False


def test_huella_con_hash_de_menos_de_64_no_es_valida():
    """Un hash truncado/corrupto (por ejemplo `sha256sum` reemplazado por algo que no
    calcula sha256 de verdad) tampoco cuenta como medición real."""
    salida = b"abc123  /etc/sudoers\n"
    assert H.huella_valida(H.huella_desde_salida("atemai", salida)) is False


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
    monkeypatch.setenv("JAX_EJECUTOR_CUENTA", "axioma-de-prueba-que-no-existe")
    monkeypatch.setenv("JAX_EJECUTOR_HUELLA_LLAVE", str(tmp_path / "id_ejecutor_huella"))
    monkeypatch.setenv("JAX_EJECUTOR_HUELLA_KNOWN_HOSTS", str(tmp_path / "known_hosts_huella"))
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
    monkeypatch.setenv("JAX_EJECUTOR_CUENTA", "axioma-de-prueba-que-no-existe")
    monkeypatch.setenv("JAX_EJECUTOR_HUELLA_LLAVE", str(tmp_path / "id_ejecutor_huella"))
    monkeypatch.setenv("JAX_EJECUTOR_HUELLA_KNOWN_HOSTS", str(tmp_path / "known_hosts_huella"))
    rc = H.principal(["aceptar", "--host", "fantasma", "--mision", "11111111-1111-1111-1111-111111111111",
                      "--sin-medir", "--motivo", "prueba de la bandera"])
    assert rc == 2  # sigue sin marca -- pero no revienta por la bandera ni por el motivo


# --- M-2 (ronda 8, corregido ronda 9): identidad DECLARADA por SUDO_UID + pwd -- no
# "verificada". El rechazo de axioma queda como freno del error accidental, no como
# barrera anti-suplantación real (ver docstrings de `_resolver_identidad_invocante` y
# `principal`).

def test_resolver_identidad_invocante_usa_sudo_uid_declarada_por_sudo(monkeypatch):
    monkeypatch.setenv("SUDO_UID", str(os.getuid()))
    monkeypatch.setenv("SUDO_USER", "mentira-no-deberia-usarse")
    uid, nombre, declarada_por = H._resolver_identidad_invocante(os.environ)
    assert uid == os.getuid()
    assert nombre == pwd.getpwuid(os.getuid()).pw_name
    assert nombre != "mentira-no-deberia-usarse"
    assert declarada_por == "sudo"  # ronda 9: "declarado por sudo", no "identidad verificada"


def test_resolver_identidad_invocante_ignora_sudo_uid_basura_y_cae_a_getuid(monkeypatch):
    monkeypatch.setenv("SUDO_UID", "no-es-un-numero")
    uid, nombre, declarada_por = H._resolver_identidad_invocante(os.environ)
    assert uid == os.getuid()
    assert nombre == pwd.getpwuid(os.getuid()).pw_name
    assert declarada_por == "proceso"


def test_resolver_identidad_invocante_ignora_sudo_uid_inexistente_y_cae_a_getuid(monkeypatch):
    monkeypatch.setenv("SUDO_UID", "999999")  # uid que casi seguro no existe en el sistema
    uid, nombre, declarada_por = H._resolver_identidad_invocante(os.environ)
    assert uid == os.getuid()
    assert declarada_por == "proceso"


def test_resolver_identidad_invocante_sin_sudo_uid_usa_getuid_y_nunca_user(monkeypatch):
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.setenv("USER", "mentira-no-deberia-usarse")
    uid, nombre, declarada_por = H._resolver_identidad_invocante(os.environ)
    assert uid == os.getuid()
    assert nombre != "mentira-no-deberia-usarse"
    assert declarada_por == "proceso"


def test_principal_registra_aceptado_por_desde_sudo_uid_no_desde_sudo_user(tmp_path, monkeypatch):
    """B-2/M-2 combinados: con --sin-medir + --motivo (no toca la red), la marca queda
    ABIERTA con `aceptada_por` == el nombre resuelto por SUDO_UID -- nunca la mentira de
    SUDO_USER/USER."""
    misiones = tmp_path / "misiones"
    mision_id = "22222222-2222-2222-2222-222222222222"
    ruta = H.ruta_huella(misiones, mision_id, "atemai")
    H.escribir_marca(ruta, H.Marca(huella=H.huella_desde_salida("atemai", b"abc  /etc/sudoers\n"),
                                    estado=H.REPORTADA, diff=("algo cambió",)))

    monkeypatch.setenv("JAX_EJECUTOR_MISIONES", str(misiones))
    monkeypatch.setenv("JAX_EJECUTOR_ADMIN_USUARIO", "fruiz")
    monkeypatch.setenv("JAX_EJECUTOR_REGISTRO", str(tmp_path / "registro.jsonl"))
    monkeypatch.setenv("JAX_EJECUTOR_POLITICA", str(tmp_path / "politica.json"))
    monkeypatch.setenv("JAX_EJECUTOR_PAUSA", str(tmp_path / "PAUSA"))
    monkeypatch.setenv("JAX_EJECUTOR_CUENTA", "axioma-de-prueba-que-no-existe")
    monkeypatch.setenv("JAX_EJECUTOR_HUELLA_LLAVE", str(tmp_path / "id_ejecutor_huella"))
    monkeypatch.setenv("JAX_EJECUTOR_HUELLA_KNOWN_HOSTS", str(tmp_path / "known_hosts_huella"))
    monkeypatch.setenv("SUDO_UID", str(os.getuid()))
    monkeypatch.setenv("SUDO_USER", "mentira-no-deberia-usarse")
    monkeypatch.setenv("USER", "otra-mentira")

    rc = H.principal(["aceptar", "--host", "atemai", "--mision", mision_id, "--sin-medir",
                      "--motivo", "prueba M-2"])
    assert rc == 0
    marca = H.leer_marca(ruta)
    nombre_esperado = pwd.getpwuid(os.getuid()).pw_name
    assert marca.aceptada_por == nombre_esperado
    assert marca.aceptada_por not in ("mentira-no-deberia-usarse", "otra-mentira")
    evento = json.loads((tmp_path / "registro.jsonl").read_text().splitlines()[-1])
    assert evento["identidad_declarada_por"] == "sudo"  # ronda 9: declarada, no "verificada"


def test_principal_rechaza_cuando_el_invocante_es_la_propia_cuenta_axioma(tmp_path, monkeypatch, capsys):
    """axioma no puede aceptar su propia huella -- `JAX_EJECUTOR_CUENTA` apunta al
    MISMO usuario que está invocando la CLI (simulado: sin SUDO_UID, `os.getuid()` es
    el del proceso de test, y `JAX_EJECUTOR_CUENTA` se pone con el nombre de ESE mismo
    usuario)."""
    misiones = tmp_path / "misiones"
    mision_id = "33333333-3333-3333-3333-333333333333"
    ruta = H.ruta_huella(misiones, mision_id, "atemai")
    H.escribir_marca(ruta, H.Marca(huella=H.huella_desde_salida("atemai", b"abc  /etc/sudoers\n"),
                                    estado=H.REPORTADA, diff=("algo cambió",)))

    monkeypatch.setenv("JAX_EJECUTOR_MISIONES", str(misiones))
    monkeypatch.setenv("JAX_EJECUTOR_ADMIN_USUARIO", "fruiz")
    monkeypatch.setenv("JAX_EJECUTOR_REGISTRO", str(tmp_path / "registro.jsonl"))
    monkeypatch.setenv("JAX_EJECUTOR_POLITICA", str(tmp_path / "politica.json"))
    monkeypatch.setenv("JAX_EJECUTOR_PAUSA", str(tmp_path / "PAUSA"))
    monkeypatch.delenv("SUDO_UID", raising=False)
    monkeypatch.setenv("JAX_EJECUTOR_CUENTA", pwd.getpwuid(os.getuid()).pw_name)

    rc = H.principal(["aceptar", "--host", "atemai", "--mision", mision_id])
    assert rc == 2
    assert "axioma_no_puede_aceptar_su_propia_huella" in capsys.readouterr().err
    marca = H.leer_marca(ruta)
    assert marca.estado == H.REPORTADA  # NO se tocó: el rechazo es antes de aceptar nada


def test_el_mutante_m5_sin_el_rechazo_de_axioma_muere(tmp_path, monkeypatch):
    """M5: una versión de `principal` sin la comparación uid_invocante == uid_axioma
    dejaría que axioma acepte su propia huella. Se mata mutando `_resolver_identidad_invocante`
    para que el proceso invocante y `JAX_EJECUTOR_CUENTA` resuelvan al mismo uid, y
    verificando -- contra el código REAL de `principal` -- que rechaza."""
    import jax.ejecutor.contratos.huella as modulo

    original_resolver = modulo._resolver_identidad_invocante
    try:
        modulo._resolver_identidad_invocante = lambda env: (12345, "axioma", "sudo")

        misiones = tmp_path / "misiones"
        mision_id = "44444444-4444-4444-4444-444444444444"
        ruta = H.ruta_huella(misiones, mision_id, "atemai")
        H.escribir_marca(ruta, H.Marca(huella=H.huella_desde_salida("atemai", b"abc  /etc/sudoers\n"),
                                        estado=H.REPORTADA, diff=("algo cambió",)))

        monkeypatch.setenv("JAX_EJECUTOR_MISIONES", str(misiones))
        monkeypatch.setenv("JAX_EJECUTOR_ADMIN_USUARIO", "fruiz")
        monkeypatch.setenv("JAX_EJECUTOR_REGISTRO", str(tmp_path / "registro.jsonl"))
        monkeypatch.setenv("JAX_EJECUTOR_POLITICA", str(tmp_path / "politica.json"))
        monkeypatch.setenv("JAX_EJECUTOR_PAUSA", str(tmp_path / "PAUSA"))

        class _PwdFalso:
            pw_uid = 12345

        monkeypatch.setattr(modulo.pwd, "getpwnam", lambda nombre: _PwdFalso())
        monkeypatch.setenv("JAX_EJECUTOR_CUENTA", "axioma")

        rc = H.principal(["aceptar", "--host", "atemai", "--mision", mision_id])
        assert rc == 2
    finally:
        modulo._resolver_identidad_invocante = original_resolver


# --- arreglo del bug de producción (jax#260, 2026-09-22): la llave PROPIA del servicio,
# NUNCA la personal del administrador -- `argv_huella_servicio` reemplaza
# `revocacion.argv_admin` en el camino en vivo. Medido en producción: `jaxsvc` (dueño
# real de `jax-platform.service` desde el 2026-09-17) no puede leer `~fruiz/.ssh/*`, y
# el camino viejo (sin `-i`, resolución de identidad por default de ssh) daba
# `vigia_no_latio=true rc=2`. -----------------------------------------------------------

from jax.ejecutor.contratos.destinos import Host as _Host  # noqa: E402

_HOST_DE_PRUEBA = _Host(nombre="atemai", ip="172.16.20.11", puerto=58291, rol="desarrollo", es_local=False)


def test_argv_huella_servicio_usa_la_llave_del_servicio_con_identities_only():
    argv = H.argv_huella_servicio(_HOST_DE_PRUEBA, llave=H.Path("/etc/jax/controlador/id_ejecutor_huella"),
                                  known_hosts=H.Path("/etc/jax/controlador/known_hosts_huella"),
                                  admin_usuario="fruiz", tope_s=30)
    assert argv[0] == "ssh"
    assert "-i" in argv
    assert argv[argv.index("-i") + 1] == "/etc/jax/controlador/id_ejecutor_huella"
    assert "IdentitiesOnly=yes" in argv
    assert "BatchMode=yes" in argv
    assert "StrictHostKeyChecking=yes" in argv
    assert any(o == "UserKnownHostsFile=/etc/jax/controlador/known_hosts_huella" for o in argv)
    assert "fruiz@172.16.20.11" in argv
    assert "-p" in argv and argv[argv.index("-p") + 1] == "58291"
    # MINOR (ronda 2): -F /dev/null -- ningún ssh_config del proceso invocante puede
    # pisar -i/IdentitiesOnly en silencio.
    assert "-F" in argv and argv[argv.index("-F") + 1] == "/dev/null"


def test_argv_huella_servicio_no_toma_parametros_extra_de_texto_de_mision():
    """La firma es explícita (host, llave, known_hosts, admin_usuario, tope_s) -- no
    hay un `texto_mision`/`declarado` que colarse acá (mismo criterio que
    `hallazgos()`, ronda 6: sin declarado)."""
    import inspect
    firma = inspect.signature(H.argv_huella_servicio)
    assert list(firma.parameters) == ["h", "llave", "known_hosts", "admin_usuario", "tope_s"]


def test_tomar_huella_actual_usa_argv_huella_servicio_no_argv_admin():
    """La comprobación explícita que pide el encargo: `huella.py` NO arma el ssh con la
    llave personal del administrador -- ni `revocacion.argv_admin` (que resuelve la
    identidad de ssh por DEFAULT, la llave de quien invoca) ni `comando_huella()`
    aparecen en el cuerpo de `_tomar_huella_actual`; la única forma de llegar a la red
    es `argv_huella_servicio`, con la llave del servicio."""
    import inspect
    fuente = inspect.getsource(H._tomar_huella_actual)
    assert "argv_admin" not in fuente
    assert "revocacion" not in fuente
    assert "comando_huella" not in fuente
    assert "argv_huella_servicio" in fuente


def test_tomar_huella_actual_host_desconocido_no_llega_a_la_red(tmp_path, monkeypatch):
    """`_tomar_huella_actual` contra un host que la política no conoce revienta ANTES
    de tocar la red -- prueba que la resolución de host es lo primero, sin depender de
    un valor por default silencioso."""
    import asyncio

    import jax.ejecutor.contratos.politica as P

    monkeypatch.setattr(P, "validar", lambda doc: type("_P", (), {"hosts": ()})())
    (tmp_path / "politica.json").write_text("{}")

    async def escenario():
        return await H._tomar_huella_actual(
            "fantasma", politica_ruta=tmp_path / "politica.json", admin_usuario="fruiz",
            huella_llave=tmp_path / "id_ejecutor_huella", huella_known_hosts=tmp_path / "known_hosts_huella")

    with pytest.raises(ValueError, match="host_desconocido"):
        asyncio.run(escenario())


def test_tomar_huella_actual_arma_el_mismo_argv_que_argv_huella_servicio(tmp_path, monkeypatch):
    """Con un host CONOCIDO, `_tomar_huella_actual` le pasa a `correr_huella_por_ssh`
    EXACTAMENTE el argv que arma `argv_huella_servicio` -- ni un comando alternativo, ni
    uno con la identidad por default."""
    import asyncio

    import jax.ejecutor.contratos.politica as P
    import jax.ejecutor.contratos.vigia_servicio as V

    monkeypatch.setattr(P, "validar", lambda doc: type("_P", (), {"hosts": (_HOST_DE_PRUEBA,)})())
    (tmp_path / "politica.json").write_text("{}")

    capturado = {}

    async def correr_falso(argv, host, *, tope_s):
        capturado["argv"], capturado["host"] = argv, host
        return H.huella_desde_salida(host, b"abc  /etc/sudoers\n")

    monkeypatch.setattr(V, "correr_huella_por_ssh", correr_falso)

    llave, known_hosts = tmp_path / "id_ejecutor_huella", tmp_path / "known_hosts_huella"
    resultado = asyncio.run(H._tomar_huella_actual(
        "atemai", politica_ruta=tmp_path / "politica.json", admin_usuario="fruiz",
        huella_llave=llave, huella_known_hosts=known_hosts, tope_s=30))

    assert resultado.host == "atemai"
    esperado = H.argv_huella_servicio(_HOST_DE_PRUEBA, llave=llave, known_hosts=known_hosts,
                                      admin_usuario="fruiz", tope_s=30)
    assert capturado["argv"] == esperado
    assert capturado["host"] == "atemai"


def test_el_mutante_sin_llave_propia_del_servicio_reproduce_el_bug_de_produccion():
    """El mutante EXACTO que causó el defecto medido: `argv_huella_servicio` sin `-i
    llave` deja que ssh resuelva la identidad por DEFAULT -- la llave personal del
    administrador, que `jaxsvc` no puede leer (`vigia_no_latio=true rc=2` en
    producción). Se reconstruye esa versión mutada y se confirma que el chequeo real
    de arriba (`test_argv_huella_servicio_usa_la_llave_del_servicio_con_identities_only`)
    no la habría dejado pasar."""
    def mutado_sin_llave_propia(h, *, llave, known_hosts, admin_usuario, tope_s):
        return ["ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
                "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={known_hosts}",
                "-o", f"ConnectTimeout={int(tope_s)}", "-p", str(h.puerto), f"{admin_usuario}@{h.ip}",
                "ejecutor-huella"]

    argv_real = H.argv_huella_servicio(_HOST_DE_PRUEBA, llave=H.Path("/etc/jax/controlador/id_ejecutor_huella"),
                                       known_hosts=H.Path("/x"), admin_usuario="fruiz", tope_s=5)
    argv_mutado = mutado_sin_llave_propia(_HOST_DE_PRUEBA, llave=H.Path("/etc/jax/controlador/id_ejecutor_huella"),
                                          known_hosts=H.Path("/x"), admin_usuario="fruiz", tope_s=5)
    assert "-i" in argv_real
    assert "-i" not in argv_mutado  # el mutante "logra" pasar -- reproduce el bug de producción


def test_el_mutante_sin_identities_only_muere():
    """El segundo mutante: sin `IdentitiesOnly=yes`, ssh puede caer a OTRA llave (agente,
    u otra de la personal del administrador) si la del servicio no funcionara -- el
    mismo tipo de fuga que este arreglo cierra, un paso más sutil que "sin -i" porque
    la llave del servicio SÍ se ofrece, pero no en exclusiva."""
    def mutado_sin_identities_only(h, *, llave, known_hosts, admin_usuario, tope_s):
        return ["ssh", "-i", str(llave), "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={known_hosts}",
                "-o", f"ConnectTimeout={int(tope_s)}", "-p", str(h.puerto), f"{admin_usuario}@{h.ip}",
                "ejecutor-huella"]

    argv_real = H.argv_huella_servicio(_HOST_DE_PRUEBA, llave=H.Path("/x"), known_hosts=H.Path("/y"),
                                       admin_usuario="fruiz", tope_s=5)
    argv_mutado = mutado_sin_identities_only(_HOST_DE_PRUEBA, llave=H.Path("/x"), known_hosts=H.Path("/y"),
                                             admin_usuario="fruiz", tope_s=5)
    assert "IdentitiesOnly=yes" in argv_real
    assert "IdentitiesOnly=yes" not in argv_mutado  # el mutante "logra" pasar sin el freno


# --- fallo cerrado: script ausente en la remota / la llave no sirve --------------------

def test_correr_huella_por_ssh_con_el_camino_del_servicio_permission_denied_revienta(tmp_path):
    """La llave del servicio no está autorizada todavía (o el script no está instalado
    en la remota): ssh sale con Permission denied, rc=255 -- `correr_huella_por_ssh`
    (compartida por el camino viejo y el nuevo) tiene que reventar igual que con
    cualquier otro rc != 0, nunca devolver una huella vacía o parcial como si fuera
    válida."""
    import asyncio

    import jax.ejecutor.contratos.vigia_servicio as V

    fake_ssh = tmp_path / "ssh"
    fake_ssh.write_text("#!/bin/sh\necho 'Permission denied (publickey).' >&2\nexit 255\n")
    fake_ssh.chmod(0o755)

    argv = H.argv_huella_servicio(_HOST_DE_PRUEBA, llave=tmp_path / "id_ejecutor_huella",
                                  known_hosts=tmp_path / "known_hosts_huella", admin_usuario="fruiz", tope_s=5)
    argv[0] = str(fake_ssh)  # mismo argv que produce el código real, ssh real sustituido por el falso

    async def escenario():
        return await V.correr_huella_por_ssh(argv, "atemai", tope_s=5)

    with pytest.raises(RuntimeError, match="huella_rc_255"):
        asyncio.run(escenario())


# --- BLOCK-2/MAJOR-3 (ronda 2, auditoría adversarial 2026-09-22): la línea de
# authorized_keys del administrador remoto -- generada por PYTHON, testeable, y
# CONVERGENTE (si ya hay una marcada, la reemplaza en vez de duplicarla). ----------------

_TIPO = "ssh-ed25519"
_CLAVE = "AAAAC3NzaC1lZDI1NTE5AAAAIGVzdG8tZXMtdW5hLWNsYXZlLWRlLXBydWViYQ"


def test_linea_authorized_keys_servicio_trae_command_restrict_y_from():
    linea = H.linea_authorized_keys_servicio(_TIPO, _CLAVE, origen_ip="172.16.20.5")
    assert 'command="sudo -n /usr/local/sbin/ejecutor-huella"' in linea
    assert ",restrict," in linea
    assert 'from="172.16.20.5"' in linea
    assert linea.endswith(f"{_TIPO} {_CLAVE} {H.MARCA_HUELLA_SERVICIO}")


def test_linea_authorized_keys_servicio_rechaza_llave_o_ip_con_comillas():
    with pytest.raises(ValueError):
        H.linea_authorized_keys_servicio(_TIPO, 'clave"con-comillas', origen_ip="1.2.3.4")
    with pytest.raises(ValueError):
        H.linea_authorized_keys_servicio(_TIPO, _CLAVE, origen_ip='1.2.3.4"; rm -rf /')


def test_el_mutante_que_borra_command_restrict_muere():
    """BLOCK-2, el mutante que pide matar: una línea SIN `command=...,restrict,` sería
    una llave de acceso COMPLETO (shell interactiva) en vez de una atada a un único
    comando -- justo lo que este arreglo existe para evitar. Se reconstruye la versión
    mutada y se confirma que el chequeo real de arriba no la habría dejado pasar."""
    def mutado_sin_command_restrict(tipo, clave, *, origen_ip):
        return f'from="{origen_ip}" {tipo} {clave} {H.MARCA_HUELLA_SERVICIO}'

    linea_real = H.linea_authorized_keys_servicio(_TIPO, _CLAVE, origen_ip="172.16.20.5")
    linea_mutada = mutado_sin_command_restrict(_TIPO, _CLAVE, origen_ip="172.16.20.5")
    assert 'command="sudo -n /usr/local/sbin/ejecutor-huella",restrict,' in linea_real
    assert 'command="sudo -n /usr/local/sbin/ejecutor-huella",restrict,' not in linea_mutada  # el mutante "pasaría"


def test_actualizar_authorized_keys_admin_agrega_si_no_hay_marca():
    actuales = "ssh-ed25519 AAAAotra otra-llave-del-administrador\n"
    salida = H.actualizar_authorized_keys_admin(actuales, tipo=_TIPO, clave=_CLAVE, origen_ip="172.16.20.5")
    assert "otra-llave-del-administrador" in salida  # el resto del archivo queda intacto
    assert salida.count(H.MARCA_HUELLA_SERVICIO) == 1
    assert 'from="172.16.20.5"' in salida


def test_actualizar_authorized_keys_admin_es_idempotente():
    actuales = "ssh-ed25519 AAAAotra otra-llave\n"
    primera = H.actualizar_authorized_keys_admin(actuales, tipo=_TIPO, clave=_CLAVE, origen_ip="172.16.20.5")
    segunda = H.actualizar_authorized_keys_admin(primera, tipo=_TIPO, clave=_CLAVE, origen_ip="172.16.20.5")
    assert primera == segunda
    assert segunda.count(H.MARCA_HUELLA_SERVICIO) == 1


def test_actualizar_authorized_keys_admin_converge_major3():
    """MAJOR-3: si `origen_ip` cambió (por ejemplo, hall9000 cambió de IP), la
    re-corrida REEMPLAZA la línea vieja -- nunca queda una segunda entrada compitiendo
    por el mismo comando forzado."""
    actuales = "ssh-ed25519 AAAAotra otra-llave\n"
    con_ip_vieja = H.actualizar_authorized_keys_admin(actuales, tipo=_TIPO, clave=_CLAVE, origen_ip="172.16.20.5")
    con_ip_nueva = H.actualizar_authorized_keys_admin(con_ip_vieja, tipo=_TIPO, clave=_CLAVE, origen_ip="172.16.20.99")

    assert con_ip_nueva.count(H.MARCA_HUELLA_SERVICIO) == 1  # UNA sola entrada, no dos
    assert 'from="172.16.20.5"' not in con_ip_nueva  # la vieja se fue
    assert 'from="172.16.20.99"' in con_ip_nueva  # quedó la nueva
    assert "otra-llave" in con_ip_nueva  # las demás líneas del administrador, intactas


def test_actualizar_authorized_keys_admin_no_toca_otras_llaves_del_administrador():
    actuales = "ssh-ed25519 AAAA1 llave-personal-1\nssh-rsa AAAA2 llave-personal-2\n"
    salida = H.actualizar_authorized_keys_admin(actuales, tipo=_TIPO, clave=_CLAVE, origen_ip="172.16.20.5")
    assert "llave-personal-1" in salida
    assert "llave-personal-2" in salida
    assert salida.count("\n") == 3  # 2 líneas originales + 1 nueva
