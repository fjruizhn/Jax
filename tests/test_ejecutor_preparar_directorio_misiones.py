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
else
  exec "$prog" "$@"
fi
"""

requiere_setfacl = pytest.mark.skipif(shutil.which("setfacl") is None or shutil.which("getfacl") is None,
                                      reason="setfacl/getfacl no están instalados")


def _bin_con_sudo_falso(tmp_path: Path) -> str:
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    (bin_ / "sudo").write_text(_SUDO_FALSO)
    (bin_ / "sudo").chmod(0o755)
    return f"{bin_}:/usr/bin:/bin"


def _correr(tmp_path, destino, admin, cuenta):
    return subprocess.run([str(SCRIPT), str(destino), admin, cuenta], capture_output=True, text=True,
                          env={"PATH": _bin_con_sudo_falso(tmp_path)}, timeout=30)


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
