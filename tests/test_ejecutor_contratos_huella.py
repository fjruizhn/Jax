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
        "/etc/ejecutor-huella",
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
    a la huella. Con OTRO admin_usuario, la ruta medida cambia con él.

    MAJOR-5 (ronda 3): `ruta_authorized_keys_admin` ahora sale de `pwd.getpwnam` real
    -- se usan DOS cuentas reales de esta máquina (`fruiz`, `axioma`), no un nombre
    inventado que ya no resolvería."""
    assert "/home/fruiz/.ssh/authorized_keys" in H.comando_huella("fruiz")
    assert "/home/axioma/.ssh/authorized_keys" in H.comando_huella("axioma")
    assert "/home/fruiz/.ssh/authorized_keys" not in H.comando_huella("axioma")


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


# --- MAJOR-6 (ronda 3, auditoría adversarial 2026-09-22): `tipo` sin validar permitía
# colar una SEGUNDA línea de authorized_keys (sin command=/restrict) con un salto de
# línea; `origen_ip="*"` volvía el `from=` inútil (acepta cualquier origen). ------------

def test_linea_authorized_keys_servicio_rechaza_tipo_con_salto_de_linea():
    """El caso concreto que pide la auditoría: un `tipo` con `\\n` cuela una SEGUNDA
    línea de authorized_keys sin `command=`/`restrict` delante -- una llave de acceso
    COMPLETO disfrazada de este arreglo."""
    tipo_hostil = "ssh-ed25519\nssh-ed25519 AAAAotra-clave-hostil shell-completa"
    with pytest.raises(ValueError):
        H.linea_authorized_keys_servicio(tipo_hostil, _CLAVE, origen_ip="172.16.20.5")


def test_linea_authorized_keys_servicio_rechaza_tipo_desconocido():
    for tipo_malo in ("", "ssh-ed25519 extra", "no-es-un-tipo", "ssh-ed25519;rm -rf /"):
        with pytest.raises(ValueError):
            H.linea_authorized_keys_servicio(tipo_malo, _CLAVE, origen_ip="172.16.20.5")


def test_linea_authorized_keys_servicio_acepta_los_tipos_reales():
    for tipo_bueno in ("ssh-ed25519", "ssh-rsa", "ecdsa-sha2-nistp256",
                       "sk-ssh-ed25519@openssh.com"):
        linea = H.linea_authorized_keys_servicio(tipo_bueno, _CLAVE, origen_ip="172.16.20.5")
        assert linea.endswith(f" {tipo_bueno} {_CLAVE} {H.MARCA_HUELLA_SERVICIO}")


def test_linea_authorized_keys_servicio_rechaza_clave_con_salto_de_linea():
    clave_hostil = f"{_CLAVE}\nssh-ed25519 AAAAotra-clave-hostil shell-completa"
    with pytest.raises(ValueError):
        H.linea_authorized_keys_servicio(_TIPO, clave_hostil, origen_ip="172.16.20.5")


def test_linea_authorized_keys_servicio_rechaza_origen_ip_con_comodin():
    """`from="*"` en ssh acepta CUALQUIER origen -- exactamente lo que `from=` existe
    para impedir."""
    for comodin in ("*", "?", "172.16.20.*", "0.0.0.0/0", "172.16.20.5,*"):
        with pytest.raises(ValueError):
            H.linea_authorized_keys_servicio(_TIPO, _CLAVE, origen_ip=comodin)


def test_linea_authorized_keys_servicio_acepta_ip_literal_v4_y_v6():
    for ip in ("172.16.20.5", "::1", "2001:db8::5"):
        linea = H.linea_authorized_keys_servicio(_TIPO, _CLAVE, origen_ip=ip)
        assert f'from="{ip}"' in linea


def test_el_mutante_sin_validar_tipo_muere():
    """El mutante que reproduce el defecto real: `linea_authorized_keys_servicio` SIN
    validar `tipo` -- un `tipo` con `\\n` pasaría directo a la línea final."""
    def mutado_sin_validar_tipo(tipo, clave, *, origen_ip):
        if not clave or any(c.isspace() for c in clave) or "'" in clave or '"' in clave:
            raise ValueError("llave_invalida")
        ip = H._validar_ip_literal(origen_ip)
        return f'command="{H._COMANDO_FORZADO_HUELLA}",restrict,from="{ip}" {tipo} {clave} {H.MARCA_HUELLA_SERVICIO}'

    tipo_hostil = "ssh-ed25519\nssh-ed25519 AAAAhostil shell-completa"
    with pytest.raises(ValueError):
        H.linea_authorized_keys_servicio(tipo_hostil, _CLAVE, origen_ip="172.16.20.5")
    linea_mutada = mutado_sin_validar_tipo(tipo_hostil, _CLAVE, origen_ip="172.16.20.5")
    assert "\n" in linea_mutada  # el mutante "logra" colar una segunda línea


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


# --- RONDA 4 (auditoría adversarial 2026-09-22): el guion declara SIEMPRE una línea
# por ruta -- hash/D/A/E. `A` (medida, confirmada AUSENTE) es válido; `E` (no se pudo
# medir) es inválido. Antes, "ausente" y "no medible" daban lo MISMO (cero líneas) y
# `huella_valida(rutas=...)` los trataba igual -- bloqueaba el Ejecutor en la máquina
# donde esa ruta de verdad está ausente (MAJOR-G, ronda 5: medido el 2026-09-22 --
# `/root/.ssh/authorized_keys` SÍ existe, vacío, en hall9000, atemai y prod; sólo
# falta en `bridge`. El texto de esta sección decía antes "no existe en NINGUNA de
# las tres" -- ERA FALSO, corregido acá y en huella.py/los docs de runbook). ------------

def _huella_de_texto(texto: str):
    return H.huella_desde_salida("atemai", texto.encode())


def test_huella_valida_estado_hash_es_valido():
    hash64 = "a" * 64
    h = _huella_de_texto(f"{hash64}  /etc/sudoers\n")
    assert H.huella_valida(h, rutas=("/etc/sudoers",)) is True


def test_huella_valida_estado_d_es_valido():
    hash64 = "a" * 64
    h = _huella_de_texto(f"{hash64}  /usr/local/sbin/ejecutor-huella\nD /etc/sudoers.d\n")
    assert H.huella_valida(h, rutas=("/etc/sudoers.d",)) is True


def test_huella_valida_estado_ausente_es_valido():
    """El punto central de la ronda 4: una ruta declarada puede estar genuinamente
    ausente (medido por Fernando, ronda 5: es el caso de `bridge` -- hall9000, atemai
    y prod SÍ tienen `/root/.ssh/authorized_keys`) -- eso es sano, `A` tiene que
    contar como medido y válido, no como "no se pudo medir"."""
    hash64 = "a" * 64
    h = _huella_de_texto(f"{hash64}  /etc/sudoers\nA /root/.ssh/authorized_keys\n")
    assert H.huella_valida(h, rutas=("/root/.ssh/authorized_keys",)) is True


def test_huella_valida_estado_error_es_invalido():
    hash64 = "a" * 64
    h = _huella_de_texto(f"{hash64}  /etc/sudoers\nE /root/.ssh/authorized_keys permiso_denegado\n")
    assert H.huella_valida(h, rutas=("/root/.ssh/authorized_keys",)) is False


def test_huella_valida_ruta_sin_ninguna_linea_es_invalida():
    hash64 = "a" * 64
    h = _huella_de_texto(f"{hash64}  /etc/sudoers\n")
    assert H.huella_valida(h, rutas=("/root/.ssh/authorized_keys",)) is False


def test_huella_valida_a_ausente_pasando_a_existir_es_un_cambio():
    """"Que la ruta pase de A a existir (o al revés) es un CAMBIO y pausa: eso es
    justo lo que queremos detectar" -- `cambio()`/`hallazgos()` lo ven como cualquier
    otra línea que cambia de texto, sin lógica especial."""
    hash64 = "a" * 64
    antes = _huella_de_texto(f"{hash64}  /etc/sudoers\nA /root/.ssh/authorized_keys\n")
    despues = _huella_de_texto(f"{hash64}  /etc/sudoers\n{hash64}  /root/.ssh/authorized_keys\n")
    assert H.cambio(antes, despues) is True
    assert H.hallazgos(antes, despues) != ()


def test_el_mutante_all_a_any_en_huella_valida_muere():
    """El mutante EXACTO que pide la auditoría: `all` -> `any` en `huella_valida`.
    Con una ruta válida (hash) y otra que ni siquiera aparece (ni un E), el código
    real (`all`) rechaza; el mutante (`any`) la dejaría pasar porque la PRIMERA sí es
    válida."""
    def huella_valida_mutada(h, *, rutas=None):
        texto = h.texto.strip()
        if not texto:
            return False
        lineas = texto.splitlines()
        if not any(H._LINEA_CON_HASH.match(l) for l in lineas):
            return False
        if rutas is None:
            return True
        return any(H._estado_de_ruta_declarada(lineas, r) in (H._HASH, H._D, H._AUSENTE) for r in rutas)

    hash64 = "a" * 64
    h = _huella_de_texto(f"{hash64}  /etc/sudoers\n")  # falta /etc/sudoers.d por completo
    rutas = ("/etc/sudoers", "/etc/sudoers.d")
    assert H.huella_valida(h, rutas=rutas) is False  # el código real: exige LAS DOS
    assert huella_valida_mutada(h, rutas=rutas) is True  # el mutante "logra" pasar


def test_el_mutante_que_trata_ausente_como_invalido_muere():
    """El mutante que reproduce el BLOCK real de la ronda 3: tratar `A` igual que "no
    medible" -- rechazaría una máquina SANA (`bridge`, donde esa ruta de verdad no
    existe)."""
    def valida_mutada_sin_ausente(h, *, rutas=None):
        texto = h.texto.strip()
        if not texto:
            return False
        lineas = texto.splitlines()
        if not any(H._LINEA_CON_HASH.match(l) for l in lineas):
            return False
        if rutas is None:
            return True
        return all(H._estado_de_ruta_declarada(lineas, r) in (H._HASH, H._D) for r in rutas)  # sin _AUSENTE

    hash64 = "a" * 64
    h = _huella_de_texto(f"{hash64}  /etc/sudoers\nA /root/.ssh/authorized_keys\n")
    rutas = ("/etc/sudoers", "/root/.ssh/authorized_keys")
    assert H.huella_valida(h, rutas=rutas) is True  # el código real: A es válido
    assert valida_mutada_sin_ausente(h, rutas=rutas) is False  # el mutante bloquea una máquina sana


# --- BLOCK-E, punto 2 (ronda 5, auditoría adversarial 2026-09-22): el glob de sbin
# tenía el MISMO defecto que el caso directorio (un `find` que falla pasaba por
# bueno) y además NO estaba representado en `huella_valida()` en absoluto -- "su
# desaparición total no invalida nada", por diseño de ronda 4. Ahora
# `RUTAS_DECLARADAS_POR_DEFAULT` lo exige como cualquier otra ruta. --------------------

def test_rutas_declaradas_por_default_incluye_el_glob_de_sbin():
    assert H.RUTA_GLOB_SBIN_EJECUTOR in H.RUTAS_DECLARADAS_POR_DEFAULT
    assert H.RUTAS_DECLARADAS_POR_DEFAULT == H.RUTAS_CONTROLES + (H.RUTA_GLOB_SBIN_EJECUTOR,)


def test_huella_valida_exige_que_el_glob_de_sbin_este_representado():
    """Sin ninguna línea (ni D ni E) para el glob de sbin, una huella que por lo
    demás está completa se rechaza -- antes esta ruta quedaba fuera del chequeo."""
    hash64 = "a" * 64
    h = _huella_de_texto(f"{hash64}  /etc/sudoers\n")
    assert H.huella_valida(h, rutas=(H.RUTA_GLOB_SBIN_EJECUTOR,)) is False


def test_huella_valida_con_glob_de_sbin_representado_pasa():
    hash64 = "a" * 64
    texto = f"{hash64}  /etc/sudoers\nD {H.RUTA_GLOB_SBIN_EJECUTOR}\n"
    h = _huella_de_texto(texto)
    assert H.huella_valida(h, rutas=("/etc/sudoers", H.RUTA_GLOB_SBIN_EJECUTOR)) is True


def test_huella_valida_con_glob_de_sbin_roto_es_invalida():
    hash64 = "a" * 64
    texto = f"{hash64}  /etc/sudoers\nE {H.RUTA_GLOB_SBIN_EJECUTOR} find_fallo\n"
    h = _huella_de_texto(texto)
    assert H.huella_valida(h, rutas=("/etc/sudoers", H.RUTA_GLOB_SBIN_EJECUTOR)) is False


# --- MAJOR-H (ronda 5): un destino de symlink (o un nombre de archivo) puede traer un
# salto de línea de VERDAD -- `find -printf` lo imprime tal cual, partiendo una línea en
# dos y forjando una SEGUNDA línea que dice ser la MISMA ruta declarada que la línea
# genuina. Quedarse con la PRIMERA por orden (rondas 2-4) deja que la forjada gane si
# ordena antes que la real. Dos o más líneas para la MISMA ruta declarada, sin importar
# sus estados, NUNCA es sano. --------------------------------------------------------

def test_huella_valida_dos_lineas_para_la_misma_ruta_es_invalida_major_h():
    """Un hash genuino de OTRA ruta hace que el chequeo "al menos un hash en algún
    lado" pase -- así se aísla la colisión de `/etc/sudoers.d` (una `D` genuina y una
    `E` forjada, o viceversa: da igual cuál "gane" por orden, las dos juntas ya son
    inválidas)."""
    hash64 = "a" * 64
    texto = f"{hash64}  /etc/ssh/sshd_config\nD /etc/sudoers.d\nE /etc/sudoers.d find_fallo\n"
    h = _huella_de_texto(texto)
    assert H.huella_valida(h, rutas=("/etc/sudoers.d",)) is False


def test_huella_valida_dos_lineas_para_la_misma_ruta_es_invalida_aunque_ambas_serian_validas_solas():
    """Ni siquiera dos estados "válidos" (D y A) para la MISMA ruta se toleran -- una
    colisión es, por definición, una medición que no se puede confiar por sí sola."""
    hash64 = "a" * 64
    texto = f"{hash64}  /etc/ssh/sshd_config\nD /etc/sudoers.d\nA /etc/sudoers.d\n"
    h = _huella_de_texto(texto)
    assert H.huella_valida(h, rutas=("/etc/sudoers.d",)) is False


def test_el_mutante_que_vuelve_al_primer_match_por_orden_muere_major_h():
    """El mutante EXACTO que pide la auditoría: `_estado_de_ruta_declarada` quedándose
    con la PRIMERA línea que dice ser `ruta`, en vez de detectar la colisión."""
    def estado_mutado_primer_match(lineas, ruta):
        con_contenido_debajo = False
        for linea in lineas:
            clasificada = H._clasificar_linea(linea)
            if clasificada is None:
                continue
            r, estado = clasificada
            if r == ruta:
                return estado  # el mutante: se queda con la PRIMERA, sin ver si hay otra
            if r.startswith(ruta + "/"):
                con_contenido_debajo = True
        return H._D if con_contenido_debajo else None

    hash64 = "a" * 64
    # "A" ordena antes que "D" (byte a byte, LC_ALL=C) -- la forjada "gana" con el
    # criterio viejo si apareciera primero en la lista que se le pasa a la función.
    lineas = [f"{hash64}  /etc/ssh/sshd_config", "A /etc/sudoers.d", "D /etc/sudoers.d"]
    assert H._estado_de_ruta_declarada(lineas, "/etc/sudoers.d") == H._ERROR  # el código real: colisión
    assert estado_mutado_primer_match(lineas, "/etc/sudoers.d") == H._AUSENTE  # el mutante "logra" pasar
