# tests/test_ejecutor_huella_sh.py
"""ejecutor-huella (arreglo del bug de producción jax#260, 2026-09-22): el comando
forzado de la llave PROPIA del servicio -- reemplaza el camino viejo
(`revocacion.argv_admin` + `huella.comando_huella()` armado en Python y mandado por
ssh como el administrador), que `jaxsvc` no puede satisfacer porque no puede leer la
llave personal de `fruiz`.

Mismo patrón de instalación que `ejecutor-freno-remoto`/`ejecutor-revocar`
(`ops/ejecutor/instalar_en_maquina.sh`): un script de root, ESTÁTICO, sin parámetros.
`comando_huella()`/`RUTAS_CONTROLES` (jax/ejecutor/contratos/huella.py) siguen siendo la
definición en Python de QUÉ se mide; este archivo mantiene sincronizado el script real
contra esa definición -- una sola fuente de verdad, dos formas (Python armaba un `sh -c`
para mandarlo por ssh, el script ahora corre ESTÁTICO en la remota, pero mide las
MISMAS rutas)."""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from jax.ejecutor.contratos import huella as H

RAIZ = Path(__file__).resolve().parents[1]
GUION = RAIZ / "ops" / "ejecutor" / "ejecutor-huella"
TEXTO = GUION.read_text()


def test_el_script_pasa_sh_menos_n():
    assert subprocess.run(["sh", "-n", str(GUION)]).returncode == 0


def test_el_script_es_ejecutable():
    assert GUION.stat().st_mode & 0o111 == 0o111


def test_el_script_no_lee_argv_del_script_ignora_lo_que_le_manden():
    """No hay superficie de ataque: lo que mide es fijo, nunca lo que alguien mande por
    ssh -- el `command=` forzado tampoco se lo dejaría pedir, pero el script no depende
    de eso para ser seguro. `tramo()` usa `$1` como parámetro de SU PROPIA función (las
    seis rutas fijas que le pasa el cuerpo del script, nunca argv del script) -- por eso
    esto se prueba corriéndolo con basura en argv y comparando la salida, no con un grep
    de `$1` (que daría un falso positivo contra `tramo()`)."""
    normal = subprocess.run([str(GUION)], capture_output=True, timeout=30)
    con_argv_hostil = subprocess.run(
        [str(GUION), "/etc/shadow", "--", "; rm -rf /"], capture_output=True, timeout=30)
    assert con_argv_hostil.returncode == normal.returncode
    assert con_argv_hostil.stdout == normal.stdout


# --- sincronización con jax.ejecutor.contratos.huella (una sola fuente de verdad) -------

def test_el_script_mide_exactamente_las_mismas_rutas_que_rutas_controles():
    for ruta in H.RUTAS_CONTROLES:
        assert TEXTO.count(f"tramo {ruta}") >= 1, ruta
    # Y nada de más entre las rutas FIJAS de nivel superior: cada línea `tramo <ruta>`
    # DIRECTA del cuerpo del script (no la del helper `tramo_admin`, paramétrica --
    # ver el test de abajo) está en RUTAS_CONTROLES.
    rutas_del_script = re.findall(r"^\s{2}tramo (\S+)$", TEXTO, flags=re.MULTILINE)
    fijas = set(rutas_del_script) - {'"$admin_home/.ssh/authorized_keys"'}  # dentro de tramo_admin(), paramétrica
    assert fijas == set(H.RUTAS_CONTROLES)


def test_el_script_mide_el_authorized_keys_del_administrador():
    """LÍMITE 9 (ronda 2, auditoría adversarial 2026-09-22): el `authorized_keys` del
    administrador es donde vive el acceso privilegiado real -- y donde este mismo
    commit pone la llave del servicio; no medirlo dejaría el propio cambio invisible a
    la huella.

    MAJOR-4/MAJOR-5 (ronda 3): ya NO hay un `ADMIN_USUARIO=fruiz` hardcodeado ni un
    `/home/$ADMIN_USUARIO` armado a mano -- `tramo_admin()` lee la cuenta de
    `ADMIN_USUARIO_ARCHIVO` (que el instalador escribe) y su home de `getent passwd`."""
    assert "tramo_admin" in TEXTO
    assert "ADMIN_USUARIO_ARCHIVO=/etc/ejecutor-huella/admin_usuario" in TEXTO
    assert "ADMIN_USUARIO=fruiz" not in TEXTO  # MAJOR-4: nunca hardcodeado
    assert '"/home/' not in TEXTO  # MAJOR-5: nunca /home/<usuario> armado a mano
    assert "getent" in TEXTO


def test_el_script_usa_lc_all_c():
    assert "export LC_ALL=C" in TEXTO


def test_el_script_mide_el_glob_de_ejecutor_en_usr_local_sbin():
    assert "/usr/local/sbin" in TEXTO
    assert "ejecutor-*" in TEXTO


def test_el_script_no_mide_passwd_group_shadow():
    """Mismo criterio que `comando_huella()` (ronda 7): crear cuentas de sistema es
    administración legítima."""
    for ruta in ("/etc/passwd", "/etc/group", "/etc/shadow"):
        assert ruta not in TEXTO, ruta


def test_el_script_usa_binarios_por_ruta_absoluta():
    """Mismo LÍMITE documentado en huella.py: sin `secure_path`, un root remoto podría
    reemplazar `sha256sum`/`find`/`sort` en el PATH."""
    for binario in ("/usr/bin/find", "/usr/bin/sha256sum", "/usr/bin/sort"):
        assert binario in TEXTO, binario


def test_el_script_termina_con_sort():
    """Mismo formato que `comando_huella()`: todo se ordena al final -- la comparación
    `cambio()`/`hallazgos()` no debe depender del orden en que el remoto haya listado
    los archivos."""
    assert TEXTO.rstrip().splitlines()[-1].strip() == '} | "$SORT"'


# --- comportamiento: correr el script REAL, sin privilegios (mismo criterio que
# test_ejecutor_revocar_sh.py) -- lo que SÍ se puede leer sin root (/etc/ssh/sshd_config)
# tiene que aparecer; lo que no se puede leer sin root se salta en silencio (2>/dev/null,
# igual que documenta huella.py -- una ruta ausente/no legible cuenta como "no existe",
# no como error) y el script de todos modos sale con éxito. -----------------------------

def test_correr_sin_privilegios_sale_con_exito_y_es_deterministico():
    """El orden que da `sort(1)` depende de LC_COLLATE de la máquina -- no tiene que
    coincidir con el de Python (huella_valida()/cambio() sólo exigen que sea el MISMO
    entre dos corridas del MISMO proceso, no un orden en particular)."""
    r1 = subprocess.run([str(GUION)], capture_output=True, timeout=30)
    r2 = subprocess.run([str(GUION)], capture_output=True, timeout=30)
    assert r1.returncode == 0
    assert r1.stdout  # /etc/ssh/sshd_config es legible por cualquiera: nunca queda vacío
    assert r1.stdout == r2.stdout


def test_correr_sin_privilegios_incluye_sshd_config_legible():
    r = subprocess.run([str(GUION)], capture_output=True, timeout=30)
    assert any(linea.endswith(" /etc/ssh/sshd_config") for linea in r.stdout.decode().splitlines())


def test_correr_sin_privilegios_no_revienta_por_rutas_no_legibles():
    """`/etc/sudoers` (0440 root) no es legible por el usuario del test -- el script NO
    debe fallar por eso: el `2>/dev/null` de cada `tramo()` lo trata como "no existe",
    igual que documenta `huella.py` (fail-closed vive en `huella_valida()`, del lado de
    Python, que trata una huella VACÍA como no medible -- un script que revienta por un
    `find` sin permiso sería peor: ni siquiera se sabría que no midió nada)."""
    r = subprocess.run([str(GUION)], capture_output=True, timeout=30)
    assert r.returncode == 0
    assert r.stderr == b""


# --- MAJOR-7 (ronda 2, auditoría adversarial 2026-09-22): la sincronía compara SALIDAS
# sobre el MISMO árbol de prueba, no listas de rutas -- bwrap bind-monta contenido de
# prueba encima de las rutas absolutas reales (mismo mecanismo que ya usa
# jax/ejecutor/contratos/cuenta_axioma.py para la jaula del Ejecutor), sin tocar el
# filesystem del host, y corre el script real Y `comando_huella()` (el texto que antes
# viajaba por ssh) contra ese mismo árbol. -----------------------------------------------

REQUIERE_BWRAP = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap no disponible")
#: MAJOR-4 (ronda 3): DOS usuarios reales y distintos de este mismo host (`getent
#: passwd` los tiene que poder resolver de verdad dentro del bwrap) -- nunca "fruiz"
#: fijo. `axioma` es una cuenta real en hall9000 con home propio; se usa acá sólo como
#: SEGUNDO admin de prueba, no porque axioma vaya a ser admin_usuario en producción.
_ADMIN_DE_PRUEBA = "fruiz"
_OTRO_ADMIN_DE_PRUEBA = "axioma"


def _arbol_de_prueba(tmp_path: Path, *, admin_usuario: str = _ADMIN_DE_PRUEBA, con_sha256sum: bool = True) -> dict:
    """Arma un árbol de prueba completo (las RUTAS_CONTROLES + el authorized_keys del
    administrador + /usr/local/sbin con ejecutor-huella instalado de verdad +
    /etc/ejecutor-huella/admin_usuario, MAJOR-4) y devuelve el mapeo
    {ruta_real: ruta_de_prueba} para bwrap. `admin_usuario` resuelve su HOME con
    `pwd.getpwnam` LOCAL (MAJOR-5) -- el mismo passwd que `getent` va a ver dentro del
    bwrap, porque es el MISMO host. `con_sha256sum=False` (MAJOR-6): el `sha256sum`
    que bwrap expone es un binario que SIEMPRE falla -- simula que el binario real
    está roto/ausente sin necesitar borrar nada del host."""
    import pwd as _pwd

    admin_home = _pwd.getpwnam(admin_usuario).pw_dir
    sudoers = tmp_path / "sudoers"; sudoers.write_text("root ALL=(ALL) ALL\n")
    sudoers_d = tmp_path / "sudoers.d"; sudoers_d.mkdir()
    (sudoers_d / "50-x").write_text("fruiz ALL=(ALL) NOPASSWD: ALL\n")
    sshd_config = tmp_path / "sshd_config"; sshd_config.write_text("Port 58291\n")
    sshd_config_d = tmp_path / "sshd_config.d"; sshd_config_d.mkdir()
    authorized_keys_d = tmp_path / "authorized_keys.d"; authorized_keys_d.mkdir()
    (authorized_keys_d / "axioma").write_text("ssh-ed25519 AAAAaxioma ejecutor-axioma\n")
    root_authorized_keys = tmp_path / "root_authorized_keys"
    root_authorized_keys.write_text("ssh-ed25519 AAAAroot root@hall9000\n")
    admin_authorized_keys = tmp_path / "admin_authorized_keys"
    admin_authorized_keys.write_text(f"ssh-ed25519 AAAAadmin {admin_usuario}@hall9000\n")
    ejecutor_huella_dir = tmp_path / "ejecutor-huella-config"; ejecutor_huella_dir.mkdir()
    (ejecutor_huella_dir / "admin_usuario").write_text(f"{admin_usuario}\n")
    sbin = tmp_path / "sbin"; sbin.mkdir()
    (sbin / "ejecutor-huella").write_bytes(GUION.read_bytes())
    (sbin / "ejecutor-huella").chmod(0o755)
    (sbin / "ejecutor-freno-remoto").write_text("#!/bin/sh\necho freno\n")
    (sbin / "ejecutor-freno-remoto").chmod(0o755)

    bin_falso = None
    if not con_sha256sum:
        bin_falso = tmp_path / "bin-falso"; bin_falso.mkdir()
        (bin_falso / "sha256sum").write_text("#!/bin/sh\nexit 127\n")
        (bin_falso / "sha256sum").chmod(0o755)

    binds = {
        "/etc/sudoers": sudoers,
        "/etc/sudoers.d": sudoers_d,
        "/etc/ssh/sshd_config": sshd_config,
        "/etc/ssh/sshd_config.d": sshd_config_d,
        "/etc/ssh/authorized_keys.d": authorized_keys_d,
        "/root/.ssh/authorized_keys": root_authorized_keys,
        "/etc/ejecutor-huella": ejecutor_huella_dir,
        f"{admin_home}/.ssh/authorized_keys": admin_authorized_keys,
        "/usr/local/sbin": sbin,
    }
    if bin_falso is not None:
        binds["/usr/bin/sha256sum"] = bin_falso / "sha256sum"
    return binds


def _argv_bwrap(binds: dict) -> list:
    """`--tmpfs /root`: `/root/.ssh` no existe en este host, y bwrap necesita poder
    crear el punto de montaje. Cualquier `.../authorized_keys` (`/root/...` o el HOME
    de cualquier admin de prueba, MAJOR-4/5: `axioma` tiene `.ssh` 0700 -- fruiz no
    puede ni entrar ahí para que bwrap monte encima) recibe el mismo tratamiento: un
    tmpfs PROPIO sobre su `.ssh`, aislado a este sandbox, que no toca el host real."""
    argv = ["bwrap", "--dev-bind", "/", "/", "--die-with-parent", "--tmpfs", "/root"]
    for real in binds:
        if not real.endswith("/authorized_keys"):
            continue
        home = str(Path(real).parent.parent)
        if home in ("/root", os.path.expanduser("~")):
            continue  # /root ya tiene su --tmpfs; el HOME del usuario del test (fruiz)
            # ya es plenamente suyo -- un --tmpfs de más ahí TAPARÍA el propio checkout
            # (visto: rompía /home/fruiz/worktrees/... y el guion dejaba de existir).
        # El HOME entero, no sólo `.ssh`: `axioma` (MAJOR-4/5) tiene `/home/axioma` en
        # 0750 dueño axioma -- el usuario del test no puede ENTRAR ahí, pero MONTAR un
        # tmpfs ENCIMA de ese punto sólo exige poder resolver el padre (`/home`, 0755) --
        # no entrar al directorio que se está reemplazando. Verificado sin sudo: alcanza.
        argv += ["--tmpfs", home]
    for real, prueba in binds.items():
        argv += ["--bind", str(prueba), real]
    return argv


@REQUIERE_BWRAP
@pytest.mark.parametrize("admin_usuario", [_ADMIN_DE_PRUEBA, _OTRO_ADMIN_DE_PRUEBA])
def test_el_guion_y_comando_huella_dan_la_misma_salida_sobre_el_mismo_arbol(tmp_path, admin_usuario):
    """MAJOR-4 (ronda 3): corre con DOS usuarios administradores distintos -- si el
    guion tuviera `fruiz` hardcodeado en vez de leerlo del archivo que escribe el
    instalador, la corrida con `axioma` fallaría (el script seguiría midiendo el
    `authorized_keys` de fruiz, `comando_huella("axioma")` mediría el de axioma, y las
    salidas NO coincidirían)."""
    binds = _arbol_de_prueba(tmp_path, admin_usuario=admin_usuario)
    base = _argv_bwrap(binds)

    salida_guion = subprocess.run(base + ["--", str(GUION)], capture_output=True, timeout=30)
    comando = H.comando_huella(admin_usuario)
    salida_python = subprocess.run(base + ["--", "/bin/sh", "-c", comando], capture_output=True, timeout=30)

    assert salida_guion.returncode == 0, salida_guion.stderr
    assert salida_python.returncode == 0, salida_python.stderr
    assert salida_guion.stdout  # si saliera vacío, la comparación de abajo no probaría nada
    assert salida_guion.stdout == salida_python.stdout


@REQUIERE_BWRAP
def test_el_mutante_xtype_f_a_type_f_rompe_la_sincronia(tmp_path):
    """MAJOR-7, el mutante que pide matar: `-xtype f` (sigue symlinks al hashear) vs
    `-type f` (NO los sigue -- hashea el symlink como archivo, `sha256sum` sobre un
    link simbólico revienta con ENOENT/ELOOP según el caso) dan salidas DISTINTAS en
    cuanto hay un symlink de por medio. Se arma un symlink real en el árbol de prueba y
    se confirma que el guion (con `-xtype f`) y una variante mutada (con `-type f`)
    YA NO COINCIDEN -- la comparación de salidas de arriba SÍ lo habría atrapado."""
    binds = _arbol_de_prueba(tmp_path)
    # Symlink real: sudoers.d/50-x -> un archivo por fuera, para que -xtype/-type difieran.
    destino = tmp_path / "afuera.txt"; destino.write_text("contenido\n")
    enlace = tmp_path / "sudoers.d" / "50-enlace"
    enlace.symlink_to(destino)

    guion_mutado = tmp_path / "ejecutor-huella-mutado"
    guion_mutado.write_text(GUION.read_text().replace("-xtype f", "-type f"))
    guion_mutado.chmod(0o755)

    base = _argv_bwrap(binds)
    salida_real = subprocess.run(base + ["--", str(GUION)], capture_output=True, timeout=30)
    salida_mutada = subprocess.run(base + ["--", str(guion_mutado)], capture_output=True, timeout=30)

    assert salida_real.returncode == 0 and salida_mutada.returncode == 0
    assert salida_real.stdout != salida_mutada.stdout  # el mutante SÍ cambia la salida -- se lo detecta


# --- MAJOR-6 (ronda 2): si falta el binario que hashea, la huella tiene que salir
# INVÁLIDA -- no "vacía" (huella_valida() ya lo cazaba) sino "con líneas, pero sin
# NINGÚN hash real". -----------------------------------------------------------------

@REQUIERE_BWRAP
def test_sin_sha256sum_la_huella_sale_invalida(tmp_path):
    binds = _arbol_de_prueba(tmp_path, con_sha256sum=False)
    base = _argv_bwrap(binds)
    r = subprocess.run(base + ["--", str(GUION)], capture_output=True, timeout=30)
    assert r.returncode == 0  # el script no revienta -- rc=0 es justo el peligro que MAJOR-6 señala
    h = H.huella_desde_salida("prueba", r.stdout)
    assert H.huella_valida(h) is False, r.stdout


# --- RONDA 4 (auditoría adversarial 2026-09-22, BLOCK reproducido en atemai y prod):
# el guion tiene que declarar SIEMPRE una de hash/D/A/E para cada ruta -- probado
# contra el guion REAL, con bwrap armando cada escenario sin tocar el host. ------------

@REQUIERE_BWRAP
def test_ruta_ausente_de_verdad_da_a_no_e(tmp_path):
    """El caso central de la ronda 4: si `/root/.ssh/authorized_keys` NO EXISTE (el
    estado real en hall9000/atemai/prod), el guion tiene que decir `A`, no quedarse
    mudo ni decir `E`."""
    binds = _arbol_de_prueba(tmp_path)
    del binds["/root/.ssh/authorized_keys"]  # no se bindea nada -- la ruta real
    # tampoco existe en este host de pruebas (confirmado: hall9000 no tiene /root/.ssh).
    base = _argv_bwrap(binds)
    r = subprocess.run(base + ["--", str(GUION)], capture_output=True, timeout=30)
    assert r.returncode == 0
    lineas = r.stdout.decode().splitlines()
    assert "A /root/.ssh/authorized_keys" in lineas


@REQUIERE_BWRAP
def test_tramo_admin_con_archivo_de_config_ausente_da_error_no_silencio(tmp_path):
    """MAJOR-C (ronda 4): si `/etc/ejecutor-huella/admin_usuario` no existe (o está
    vacío), `tramo_admin()` NUNCA se queda mudo (`return 0` sin imprimir nada) --
    tiene que decir `E`, aunque no tenga una ruta de archivo real que reportar."""
    binds = _arbol_de_prueba(tmp_path)
    (binds["/etc/ejecutor-huella"] / "admin_usuario").unlink()
    base = _argv_bwrap(binds)
    r = subprocess.run(base + ["--", str(GUION)], capture_output=True, timeout=30)
    assert r.returncode == 0
    lineas = r.stdout.decode().splitlines()
    assert any(l.startswith("E (admin_usuario) archivo_ausente_o_vacio") for l in lineas), lineas


@REQUIERE_BWRAP
def test_tramo_admin_con_cuenta_inexistente_da_error_no_silencio(tmp_path):
    """MAJOR-C: si `getent passwd` no resuelve la cuenta que dice el archivo de
    config (cuenta borrada, nombre mal escrito), es `E`, no silencio."""
    binds = _arbol_de_prueba(tmp_path)
    (binds["/etc/ejecutor-huella"] / "admin_usuario").write_text("cuenta-que-no-existe-de-verdad\n")
    base = _argv_bwrap(binds)
    r = subprocess.run(base + ["--", str(GUION)], capture_output=True, timeout=30)
    assert r.returncode == 0
    lineas = r.stdout.decode().splitlines()
    assert any(l.startswith("E (cuenta-que-no-existe-de-verdad) getent_no_resuelve") for l in lineas), lineas


@REQUIERE_BWRAP
def test_tramo_admin_incluye_la_ruta_resuelta_major_c(tmp_path):
    """MAJOR-C: la línea del admin lleva la RUTA RESUELTA (el HOME real de la cuenta,
    vía `getent passwd`) -- si el home cambiara en el passwd, la ruta que aparece acá
    cambia, y eso es justo lo que se quiere ver como cambio."""
    binds = _arbol_de_prueba(tmp_path, admin_usuario=_OTRO_ADMIN_DE_PRUEBA)
    base = _argv_bwrap(binds)
    r = subprocess.run(base + ["--", str(GUION)], capture_output=True, timeout=30)
    assert r.returncode == 0
    salida = r.stdout.decode()
    assert "/home/axioma/.ssh/authorized_keys" in salida
    assert "/home/fruiz/.ssh/authorized_keys" not in salida


@REQUIERE_BWRAP
def test_tramo_admin_con_authorized_keys_ausente_da_a_con_ruta_resuelta(tmp_path):
    """MAJOR-C: si la cuenta SÍ resuelve pero su `authorized_keys` no existe, es `A
    <ruta resuelta>` -- medido y confirmado ausente, no un error. Se usa `axioma` (no
    `fruiz`, el usuario del test) y se arma el argv de bwrap A MANO -- `_argv_bwrap`
    decide poner `--tmpfs` sobre un HOME mirando qué claves `/authorized_keys` hay en
    `binds`, y acá se necesita el `--tmpfs` (para que el archivo esté REALMENTE
    ausente, no el de axioma en el host de pruebas) sin bindear ningún archivo
    encima."""
    binds = _arbol_de_prueba(tmp_path, admin_usuario=_OTRO_ADMIN_DE_PRUEBA)
    admin_home = __import__("pwd").getpwnam(_OTRO_ADMIN_DE_PRUEBA).pw_dir
    del binds[f"{admin_home}/.ssh/authorized_keys"]
    base = ["bwrap", "--dev-bind", "/", "/", "--die-with-parent", "--tmpfs", "/root", "--tmpfs", admin_home]
    for real, prueba in binds.items():
        base += ["--bind", str(prueba), real]
    r = subprocess.run(base + ["--", str(GUION)], capture_output=True, timeout=30)
    assert r.returncode == 0
    lineas = r.stdout.decode().splitlines()
    assert f"A {admin_home}/.ssh/authorized_keys" in lineas
