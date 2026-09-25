# tests/test_permisos_proyectos.py
"""ops/permisos-proyectos.sh -- spec docs/superpowers/specs/2026-09-22-proyectos-y-selector-design.md
§5: RAIZ/proyectos/ tiene que ser escribible por jaxsvc (LAS MANOS, jax-platform corren como
esa cuenta) Y por fruiz (`scripts/procesar_archivos.py`, corrido a mano), con herencia para
todo lo que se cree después. Medido en hall9000 el 2026-09-25 antes de tocar nada:
`/home/fruiz/jax-workspace/proyectos` es fruiz:fruiz 775 sin ACL, y
`sudo -u jaxsvc test -w proyectos` da NO.

Dos partes, mismo criterio que tests/test_arranque_instalado.py:

(a) SIEMPRE corre, también en un runner sin ningún host de producción cerca: arma un árbol
    temporal propio, corre el guion real (`--aplicar`) contra esa RAIZ temporal, y comprueba
    la mecánica completa -- que un archivo nuevo, creado con umask 022, herede el grupo y
    quede escribible por grupo pese al umask (esto es LO QUE HACE que la ACL por defecto
    sirva: el umask se ignora cuando hay ACL por defecto, verificado empíricamente en
    hall9000 el 2026-09-25, no supuesto); que `--verificar` detecte un directorio al que se
    le retira la ACL por defecto (control negativo -- un detector que no se ejercita a sí
    mismo no se sabe si detecta); y que rechace RAIZ inválidas (`/`, vacía, una ruta sin
    `proyectos/`).

    El guion hardcodea los nombres `fruiz`/`jaxsvc` (así lo fija el diseño del spec: dueño
    fruiz, grupo jaxsvc). Para que (a) corra de verdad en CUALQUIER runner -- no sólo en
    hall9000, donde esas dos identidades ya existen -- el fixture `_identidades` las crea
    con `sudo` si hacen falta (idempotente: sólo crea lo que no existe ya; en hall9000 no
    toca nada). Si no hay `sudo -n` disponible para eso, o si el filesystem temporal no
    soporta ACL POSIX, el test se salta con el motivo explícito -- nunca falla en silencio
    ni pasa sin haber mirado nada.

(b) SÓLO en el host de producción de jax (existe `/home/fruiz/jax-workspace/proyectos` Y
    `sudo -n -u jaxsvc true` funciona): `sudo -n -u jaxsvc` crea y borra un archivo real
    dentro de `proyectos/` -- exige éxito. Ese test falla HOY contra el estado real (ver la
    corrida "ANTES DE APLICAR" en la Biblioteca de esta ronda, 2026-09-25): `proyectos/` no
    es escribible por jaxsvc todavía. Fuera del host de producción, se salta con el motivo.

Ninguna de las dos partes corre `--aplicar` contra `/home/fruiz/jax-workspace` real -- ver
`ops/permisos-proyectos.sh`, que lo documenta como decisión operativa (la aplicación en
producción la hace la sesión principal tras auditoría, no un test).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

RAIZ_REPO = Path(__file__).resolve().parents[1]
SCRIPT = RAIZ_REPO / "ops" / "permisos-proyectos.sh"

RAIZ_PRODUCCION = Path("/home/fruiz/jax-workspace")
PROYECTOS_PRODUCCION = RAIZ_PRODUCCION / "proyectos"


def test_el_guion_existe_y_es_ejecutable():
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK), "ops/permisos-proyectos.sh necesita el bit ejecutable"


def _sudo_n_disponible() -> bool:
    return subprocess.run(
        ["sudo", "-n", "true"], capture_output=True, timeout=10
    ).returncode == 0


def _acl_disponible_en(directorio: Path) -> bool:
    """Empírico, no supuesto: crea un archivo real y le intenta poner una ACL de acceso."""
    if shutil.which("setfacl") is None or shutil.which("getfacl") is None:
        return False
    prueba = directorio / ".prueba-acl"
    prueba.touch()
    try:
        resultado = subprocess.run(
            ["setfacl", "-m", f"u:{os.getlogin() if hasattr(os, 'getlogin') else os.environ.get('USER', 'root')}:rwx", str(prueba)],
            capture_output=True,
        )
        return resultado.returncode == 0
    finally:
        prueba.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def _identidades():
    """Asegura que existan las identidades que el guion hardcodea (fruiz/jaxsvc) -- ya
    existen en hall9000; en un runner efímero (CI) las crea, idempotente, y no las borra al
    salir (un runner efímero no necesita limpieza; en un host persistente no las creamos
    porque ya existen de entrada -- ver la comprobación previa de cada rama)."""
    if not _sudo_n_disponible():
        pytest.skip("sudo -n no disponible -- no se pueden garantizar las identidades fruiz/jaxsvc")

    tiene_grupo = subprocess.run(["getent", "group", "jaxsvc"], capture_output=True).returncode == 0
    if not tiene_grupo:
        creado = subprocess.run(["sudo", "-n", "groupadd", "-f", "jaxsvc"], capture_output=True)
        if creado.returncode != 0:
            pytest.skip(f"no se pudo crear el grupo jaxsvc: {creado.stderr.decode(errors='replace')}")

    tiene_usuario = subprocess.run(["getent", "passwd", "fruiz"], capture_output=True).returncode == 0
    if not tiene_usuario:
        creado = subprocess.run(
            ["sudo", "-n", "useradd", "--system", "--no-create-home", "fruiz"], capture_output=True
        )
        if creado.returncode != 0:
            pytest.skip(f"no se pudo crear el usuario fruiz: {creado.stderr.decode(errors='replace')}")

    return None


@pytest.fixture()
def arbol_temporal(tmp_path, _identidades):
    """RAIZ/proyectos/ temporal, con un archivo y un subdirectorio ya existentes (para
    ejercitar --aplicar sobre un árbol preexistente, como el real). tmp_path vive bajo el
    directorio base de pytest; en hall9000 eso resultó ser tmpfs con `nosuid` -- que un
    proceso sin privilegio no pueda dejar puesto el setgid ahí es justo lo que este guion
    sortea corriendo el chmod g+s como root y al final (ver el comentario del propio
    guion) -- así que no hace falta evitar tmpfs, sólo confirmar con el smoke test de abajo
    que el detector empírico de ACL ve lo que hace falta."""
    raiz = tmp_path / "raiz"
    proyectos = raiz / "proyectos"
    (proyectos / "un-proyecto" / "sub").mkdir(parents=True)
    (proyectos / "un-proyecto" / "archivo.txt").write_text("contenido\n")

    if not _acl_disponible_en(proyectos):
        pytest.skip("el filesystem temporal no soporta ACL POSIX (setfacl falló contra un archivo real)")

    # El directorio de arriba de "raiz" tiene que ser transitable por cualquiera (jaxsvc
    # incluido) para que sudo -u jaxsvc pueda siquiera llegar a proyectos/ -- mktemp/tmp_path
    # crean con 0700 por defecto.
    os.chmod(tmp_path, 0o755)
    os.chmod(raiz, 0o755)

    return raiz


def _correr(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(SCRIPT), *args], capture_output=True, text=True, timeout=60
    )


def test_verificar_por_defecto_y_rechaza_raiz_invalidas(arbol_temporal):
    # Por defecto (sin flag) es --verificar, y sale distinto de 0 contra un árbol sin tocar.
    sin_flag = _correr(str(arbol_temporal))
    assert sin_flag.returncode == 1
    assert "NO CUMPLE" in sin_flag.stdout

    raiz_slash = _correr("--verificar", "/")
    assert raiz_slash.returncode == 2

    raiz_vacia = _correr("--verificar", "")
    assert raiz_vacia.returncode == 2

    sin_proyectos = arbol_temporal.parent / "sin-proyectos"
    sin_proyectos.mkdir()
    resultado = _correr("--verificar", str(sin_proyectos))
    assert resultado.returncode == 2
    assert "proyectos" in (resultado.stdout + resultado.stderr)


def test_aplicar_deja_el_arbol_escribible_por_jaxsvc_con_herencia_pese_al_umask(arbol_temporal):
    aplicado = _correr("--aplicar", str(arbol_temporal))
    assert aplicado.returncode == 0, aplicado.stdout + aplicado.stderr

    proyectos = arbol_temporal / "proyectos"

    # --verificar sobre el árbol YA aplicado da 0 -- el propio --aplicar también lo corre al
    # final y lo exige, esto lo confirma desde afuera, como una corrida independiente.
    verificado = _correr("--verificar", str(arbol_temporal))
    assert verificado.returncode == 0, verificado.stdout + verificado.stderr

    # Respaldo: --aplicar tiene que haber dejado un .acl restaurable en ~/respaldos-permisos.
    respaldos = sorted((Path.home() / "respaldos-permisos").glob("proyectos-*.acl"), key=lambda p: p.stat().st_mtime)
    assert respaldos, "no se encontró ningún respaldo en ~/respaldos-permisos"
    contenido_respaldo = respaldos[-1].read_text()
    assert str(proyectos) in contenido_respaldo

    # El punto central del diseño: un archivo NUEVO, creado con umask 022 (que normalmente
    # dejaría al grupo sin escritura), tiene que heredar el grupo del directorio (setgid) y
    # quedar escribible por grupo de todas formas (ACL por defecto, que ignora el umask --
    # verificado empíricamente en hall9000 el 2026-09-25, no supuesto).
    umask_previo = os.umask(0o022)
    try:
        nuevo = proyectos / "un-proyecto" / "sub" / "nuevo.txt"
        fd = os.open(nuevo, os.O_CREAT | os.O_WRONLY, 0o666)
        os.close(fd)
    finally:
        os.umask(umask_previo)

    import grp
    grupo_del_nuevo = grp.getgrgid(nuevo.stat().st_gid).gr_name
    assert grupo_del_nuevo == "jaxsvc", f"el archivo nuevo heredó el grupo {grupo_del_nuevo!r}, no jaxsvc"

    acl = subprocess.run(["getfacl", "-p", str(nuevo)], capture_output=True, text=True, check=True).stdout
    assert any(
        linea.startswith("group:jaxsvc:rw") for linea in acl.splitlines()
    ), f"el archivo nuevo no quedó con ACL de grupo jaxsvc escribible pese al umask 022:\n{acl}"

    # Idempotencia: aplicar de nuevo no rompe nada y sigue en 0.
    reaplicado = _correr("--aplicar", str(arbol_temporal))
    assert reaplicado.returncode == 0, reaplicado.stdout + reaplicado.stderr


def test_verificar_detecta_un_directorio_sin_acl_por_defecto(arbol_temporal):
    """Control negativo (Principio VII -- un freno sin prueba no es freno): un detector
    que nunca se vio fallar no se sabe si detecta o si sólo dice que sí."""
    aplicado = _correr("--aplicar", str(arbol_temporal))
    assert aplicado.returncode == 0, aplicado.stdout + aplicado.stderr

    sub = arbol_temporal / "proyectos" / "un-proyecto" / "sub"
    # -k retira SÓLO la ACL por defecto de ese directorio -- el resto del árbol sigue en
    # regla, así que si el detector encuentra ESTA ruta y ninguna otra, es que de verdad
    # está mirando la ACL por defecto de cada directorio, no un chequeo global.
    subprocess.run(["setfacl", "-k", str(sub)], check=True)

    resultado = _correr("--verificar", str(arbol_temporal))
    assert resultado.returncode == 1
    assert str(sub) in resultado.stdout
    assert "ACL por defecto" in resultado.stdout


# --- (b) sólo en el host de producción real ------------------------------------------------

def _motivo_de_skip_fuera_de_produccion() -> str | None:
    if not PROYECTOS_PRODUCCION.is_dir():
        return f"esta máquina no tiene {PROYECTOS_PRODUCCION} -- no es el host de producción de jax"
    prueba = subprocess.run(
        ["sudo", "-n", "-u", "jaxsvc", "true"], capture_output=True
    )
    if prueba.returncode != 0:
        return "sudo -n -u jaxsvc no funciona en esta máquina"
    return None


def test_jaxsvc_puede_crear_y_borrar_un_archivo_real_en_proyectos():
    """Ejercita la escritura corriendo COMO jaxsvc contra la RAIZ de producción real --
    spec §5: 'un test que ejercita la escritura corriendo como jaxsvc... y que falla contra
    el estado de hoy'. No corre --aplicar: sólo comprueba el estado actual (o el que deje la
    sesión principal después de aplicar, con GO de Fernando)."""
    motivo = _motivo_de_skip_fuera_de_produccion()
    if motivo:
        pytest.skip(motivo)

    archivo_prueba = PROYECTOS_PRODUCCION / f".permisos-proyectos-selftest-{os.getpid()}"
    try:
        creado = subprocess.run(
            ["sudo", "-n", "-u", "jaxsvc", "touch", str(archivo_prueba)], capture_output=True, text=True
        )
        assert creado.returncode == 0, (
            f"jaxsvc no pudo crear un archivo en {PROYECTOS_PRODUCCION}: "
            f"{creado.stdout}{creado.stderr}"
        )

        borrado = subprocess.run(
            ["sudo", "-n", "-u", "jaxsvc", "rm", "-f", str(archivo_prueba)], capture_output=True, text=True
        )
        assert borrado.returncode == 0, (
            f"jaxsvc no pudo borrar el archivo que acababa de crear en {PROYECTOS_PRODUCCION}: "
            f"{borrado.stdout}{borrado.stderr}"
        )
    finally:
        # Red de seguridad, no parte del camino que se está exigiendo: si algo de arriba
        # falló y dejó el archivo de prueba puesto, lo limpia como root -- nunca antes de
        # haber ejercitado el borrado real como jaxsvc, o esa parte del test daría verde
        # sin haber probado nada (el archivo ya no estaría para cuando se intenta borrar).
        if archivo_prueba.exists():
            subprocess.run(["sudo", "-n", "rm", "-f", str(archivo_prueba)], capture_output=True)
