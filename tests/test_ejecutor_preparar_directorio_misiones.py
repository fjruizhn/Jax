# tests/test_ejecutor_preparar_directorio_misiones.py
"""B-2 (ronda 4, auditoría adversarial 2026-09-22): `JAX_EJECUTOR_MISIONES` con una
ACL EXPLÍCITA -- el admin (fruiz) conserva rwx, la cuenta del Ejecutor (axioma) recibe
SÓLO travesía (--x, sin +r), el resto de "otros" no tiene nada. Reproduce -- no
inventa -- la ACL real de producción medida el 2026-09-22 (`getfacl
/var/lib/jax-ejecutor-misiones`: `other::---`, `user:fruiz:rwx` con sus defaults).

Corre el script REAL con un `sudo` de mentira en el PATH (mismo truco que
`test_ejecutor_reemplazar_directorio_atomico.py`): como el test no corre como root, el
`sudo` falso saca `-o`/`-g` de `install` y ejecuta todo lo demás (`setfacl`) tal cual --
setfacl NO necesita ser root para tocar la ACL de un directorio del que ya se es dueño."""
import shutil
import subprocess
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
SCRIPT = RAIZ / "ops" / "ejecutor" / "preparar_directorio_misiones.sh"

_SUDO_FALSO = """#!/bin/bash
prog="$1"; shift
if [ "$prog" = "install" ]; then
  args=()
  while [ $# -gt 0 ]; do
    case "$1" in
      -o|-g) shift 2 ;;
      *) args+=("$1"); shift ;;
    esac
  done
  exec install "${args[@]}"
elif [ "$prog" = "chown" ]; then
  exit 0
else
  exec "$prog" "$@"
fi
"""

requiere_setfacl = pytest.mark.skipif(shutil.which("setfacl") is None or shutil.which("getfacl") is None,
                                      reason="setfacl/getfacl no están instalados")


def _bin_con_sudo_falso(tmp_path: Path, *, log: Path | None = None) -> str:
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    contenido = _SUDO_FALSO
    if log is not None:
        # MINOR (ronda 8): además de ejecutar, deja constancia de QUÉ programa se
        # invocó -- para probar el MECANISMO (install -d al crear, chmod al ya
        # existente), no sólo el resultado final (que ya cubren los tests de arriba).
        contenido = contenido.replace('prog="$1"; shift\n', f'prog="$1"; shift\necho "$prog" >> "{log}"\n')
    (bin_ / "sudo").write_text(contenido)
    (bin_ / "sudo").chmod(0o755)
    return f"{bin_}:/usr/bin:/bin"


def _correr(tmp_path, destino, admin, cuenta, *, log: Path | None = None):
    return subprocess.run([str(SCRIPT), str(destino), admin, cuenta], capture_output=True, text=True,
                          env={"PATH": _bin_con_sudo_falso(tmp_path, log=log)}, timeout=30)


def _replicar_acl_real_de_produccion(destino: Path, dueno: str) -> None:
    """El ESTADO INICIAL: la ACL medida en `/var/lib/jax-ejecutor-misiones` el
    2026-09-22 -- `other::---`, `user:<admin>:rwx` (con sus defaults) ya puesta por una
    instalación anterior. No se inventa: es la que reportó `getfacl` en producción."""
    destino.mkdir(parents=True)
    destino.chmod(0o770)
    subprocess.run(["setfacl", "-m", f"u:{dueno}:rwx", "-m", f"d:u:{dueno}:rwx", str(destino)], check=True)


def test_el_script_pasa_bash_menos_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


@requiere_setfacl
def test_exige_los_tres_argumentos():
    assert subprocess.run([str(SCRIPT)], capture_output=True).returncode != 0


@requiere_setfacl
def test_admin_conserva_rwx_y_la_cuenta_recibe_solo_travesia(tmp_path):
    destino = tmp_path / "produccion" / "misiones"
    admin = subprocess.run(["id", "-un"], capture_output=True, text=True, check=True).stdout.strip()
    cuenta = "nobody"
    _replicar_acl_real_de_produccion(destino, admin)

    r = _correr(tmp_path, destino, admin, cuenta)
    assert r.returncode == 0, r.stderr

    acl = subprocess.run(["getfacl", "-p", str(destino)], capture_output=True, text=True, check=True).stdout
    assert f"user:{admin}:rwx" in acl
    assert f"user:{cuenta}:--x" in acl
    assert f"default:user:{admin}:rwx" in acl
    assert f"default:user:{cuenta}:--x" in acl
    assert "other::---" in acl


@requiere_setfacl
def test_directorio_nuevo_queda_en_0700_no_0750(tmp_path):
    """MINOR (ronda 6): con 0750, `group::r-x` le daría LISTAR a cualquier miembro del
    grupo jaxsvc -- no sólo a admin/cuenta, que son las dos únicas cuentas pensadas.
    El acceso de las dos sale ENTERO de sus entradas ACL, no del modo. `group::` es lo
    que hay que mirar -- una vez que hay una ACL, `stat()`/el modo "de en medio"
    reflejan el MASK (que sí tiene que dar rwx, para que a `admin` no se le recorte),
    no la entrada `group::` de base."""
    destino = tmp_path / "misiones"
    admin = subprocess.run(["id", "-un"], capture_output=True, text=True, check=True).stdout.strip()
    r = _correr(tmp_path, destino, admin, "nobody")
    assert r.returncode == 0, r.stderr
    acl = subprocess.run(["getfacl", "-p", str(destino)], capture_output=True, text=True, check=True).stdout
    assert "group::---" in acl, acl


@requiere_setfacl
def test_en_una_corrida_idempotente_SI_reconverge_el_modo_de_un_directorio_existente(tmp_path):
    """Ronda 7, punto 4 (reemplaza el criterio de la ronda 6): el modo/ACL se aplican
    TAMBIÉN cuando el directorio ya existe -- si alguien (u otra instalación vieja) lo
    dejó con permisos de más (`0770`, `group::rwx`), la corrida siguiente lo corrige,
    no lo deja así "porque ya existía". Sin ventana: la corrección es UN solo
    `setfacl -m` con todas las entradas juntas (base + nombradas + default), no un
    `chmod` separado seguido de un `setfacl` -- no hay un estado intermedio distinto
    del inicial o del final."""
    destino = tmp_path / "misiones"
    admin = subprocess.run(["id", "-un"], capture_output=True, text=True, check=True).stdout.strip()
    r1 = _correr(tmp_path, destino, admin, "nobody")
    assert r1.returncode == 0, r1.stderr

    # Alguien deja el directorio con permisos "de más" a mano, para simular el estado
    # real que preexistiría en una corrida idempotente sobre producción.
    destino.chmod(0o770)
    r2 = _correr(tmp_path, destino, admin, "nobody")
    assert r2.returncode == 0, r2.stderr
    acl2 = subprocess.run(["getfacl", "-p", str(destino)], capture_output=True, text=True, check=True).stdout
    assert "group::---" in acl2, acl2
    assert f"user:{admin}:rwx" in acl2
    assert "user:nobody:--x" in acl2


@requiere_setfacl
def test_es_idempotente(tmp_path):
    destino = tmp_path / "misiones"
    admin = subprocess.run(["id", "-un"], capture_output=True, text=True, check=True).stdout.strip()
    _correr(tmp_path, destino, admin, "nobody")
    r2 = _correr(tmp_path, destino, admin, "nobody")
    assert r2.returncode == 0, r2.stderr


@requiere_setfacl
def test_la_cuenta_puede_atravesar_pero_no_listar_ronda4_b2(tmp_path):
    """"Su equivalente comprobable" a montarlo con bwrap real como axioma: NO se puede
    cambiar de UID sin privilegios (un `--unshare-user` de bwrap remapea el UID hacia
    ADENTRO del namespace, pero el kernel sigue comprobando el UID REAL contra la ACL
    de un bind del host) -- así que lo que SÍ se puede probar, real, sin inventar nada,
    es la propiedad misma que hace funcionar el diseño: con SOLO `--x` (sin `+r`) se
    puede entrar a un hijo conocido por nombre, pero NO se puede listar el padre. Se
    verifica reduciendo los permisos DE BASE del propio usuario del test a `--x` (lo
    que la ACL le da a `nobody` de verdad) y comprobando el mismo comportamiento que
    tendría axioma."""
    destino = tmp_path / "misiones"
    admin = subprocess.run(["id", "-un"], capture_output=True, text=True, check=True).stdout.strip()
    _correr(tmp_path, destino, admin, "nobody")
    hijo = destino / "11111111-1111-1111-1111-111111111111"
    hijo.mkdir(mode=0o700)
    (hijo / "archivo").write_text("dato")

    # Simula "sólo --x, sin +r" sobre el padre (lo que la ACL real le da a `nobody`):
    destino.chmod(0o110)
    try:
        # Entrar a un hijo CONOCIDO por su nombre exacto SÍ funciona con sólo --x:
        listado_del_hijo = subprocess.run(["ls", str(hijo)], capture_output=True, text=True)
        assert listado_del_hijo.returncode == 0 and "archivo" in listado_del_hijo.stdout
        # Pero LISTAR el padre (sin +r) falla:
        listado_del_padre = subprocess.run(["ls", str(destino)], capture_output=True, text=True)
        assert listado_del_padre.returncode != 0
    finally:
        destino.chmod(0o770)


@requiere_setfacl
def test_directorio_existente_con_acl_vieja_de_produccion_se_corrige_ronda7(tmp_path):
    """Ronda 7, punto 4: `apt install`/aaPanel demostraron el BLOCK -- y el directorio
    de producción real, medido el 2026-09-22, tenía `drwxrwx---` con `group::rwx` (una
    instalación de ANTES de que este script existiera). La ronda 6 sólo corregía el
    modo/ACL al CREAR el directorio -- si ya existía con la ACL vieja, se quedaba así
    para siempre. Este test replica exactamente esa ACL vieja y comprueba que el
    script la corrige, no que la deja intacta."""
    destino = tmp_path / "misiones"
    admin = subprocess.run(["id", "-un"], capture_output=True, text=True, check=True).stdout.strip()
    # ACL real de producción, previa a esta ronda: drwxrwx---, group::rwx.
    destino.mkdir(parents=True)
    destino.chmod(0o770)

    r = _correr(tmp_path, destino, admin, "nobody")
    assert r.returncode == 0, r.stderr

    acl = subprocess.run(["getfacl", "-p", str(destino)], capture_output=True, text=True, check=True).stdout
    assert "group::---" in acl, acl
    assert f"user:{admin}:rwx" in acl
    assert "user:nobody:--x" in acl
    assert "other::---" in acl


@requiere_setfacl
def test_directorio_existente_con_acl_vieja_de_produccion_es_idempotente_ronda7(tmp_path):
    """La corrección de una ACL vieja tiene que ser tan idempotente como la corrida
    normal: correrlo dos veces sobre el mismo estado inicial "malo" no falla."""
    destino = tmp_path / "misiones"
    admin = subprocess.run(["id", "-un"], capture_output=True, text=True, check=True).stdout.strip()
    destino.mkdir(parents=True)
    destino.chmod(0o770)

    r1 = _correr(tmp_path, destino, admin, "nobody")
    assert r1.returncode == 0, r1.stderr
    r2 = _correr(tmp_path, destino, admin, "nobody")
    assert r2.returncode == 0, r2.stderr
    acl = subprocess.run(["getfacl", "-p", str(destino)], capture_output=True, text=True, check=True).stdout
    assert "group::---" in acl


@requiere_setfacl
def test_directorio_nuevo_usa_install_d_no_mkdir_ronda8_minor(tmp_path):
    """MINOR (ronda 8): para CREAR, vuelve `install -d -m 0700` -- una sola invocación
    que fija el modo restrictivo sin ventana, en vez de `mkdir -p` (que no fija modo) +
    un `setfacl` posterior como único punto de corrección."""
    destino = tmp_path / "misiones"
    admin = subprocess.run(["id", "-un"], capture_output=True, text=True, check=True).stdout.strip()
    log = tmp_path / "invocaciones.log"

    r = _correr(tmp_path, destino, admin, "nobody", log=log)
    assert r.returncode == 0, r.stderr
    invocados = log.read_text().split()
    assert "install" in invocados, invocados
    assert "mkdir" not in invocados, invocados


@requiere_setfacl
def test_directorio_existente_usa_chmod_no_install_ronda8_minor(tmp_path):
    """MINOR (ronda 8): para el directorio que YA EXISTE, `chmod` + el `setfacl`
    combinado (ronda 7) -- no `install -d`, que resetea el modo incluso sobre un
    directorio preexistente (comprobado: 700→755 sin `-m`)."""
    destino = tmp_path / "misiones"
    admin = subprocess.run(["id", "-un"], capture_output=True, text=True, check=True).stdout.strip()
    destino.mkdir(parents=True)
    destino.chmod(0o770)
    log = tmp_path / "invocaciones.log"

    r = _correr(tmp_path, destino, admin, "nobody", log=log)
    assert r.returncode == 0, r.stderr
    invocados = log.read_text().split()
    assert "chmod" in invocados, invocados
    assert "install" not in invocados, invocados
