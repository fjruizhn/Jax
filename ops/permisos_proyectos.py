#!/usr/bin/env python3
# ops/permisos_proyectos.py [--verificar|--aplicar] [RAIZ] — spec
# docs/superpowers/specs/2026-09-22-proyectos-y-selector-design.md §5 (aprobado por
# Fernando, corrección de brief 2026-09-25): RAIZ/proyectos/ y todo lo de abajo tiene que
# quedar DUEÑO jaxsvc, GRUPO fruiz con escritura, setgid en directorios, y ACL POSIX de
# acceso Y por defecto `u:jaxsvc:rwX,g:fruiz:rwX,m::rwx` (la ACL por defecto es la que hace
# que un archivo nuevo quede escribible pese al umask -- verificado empíricamente que el
# umask se ignora cuando hay ACL por defecto, no supuesto).
#
# Reemplaza a ops/permisos-proyectos.sh (retirado el 2026-09-25 tras 3 BLOCK de la
# auditoría de escalón 3 sobre esa primera versión): el núcleo pasa a Python porque el
# guion de shell no podía defenderse de un TOCTOU de symlinks (B3, ver abajo) ni computar
# el permiso ACL EFECTIVO en vez del texto pedido (B1, ver abajo) con una herramienta que
# se pueda probar, no suponer.
#
# --- Amenaza (B3): root sigue symlinks, jaxsvc puede reemplazar un directorio -------------
# `jaxsvc` tiene escritura de grupo sobre `$RAIZ` (fruiz:jaxsvc 775) -- un proceso de
# jaxsvc con un bug, o comprometido, puede en cualquier momento BORRAR `proyectos` y
# ponerle en su lugar un symlink a, por ejemplo, `/etc`. Si el recorrido recursivo de este
# guion (que corre como ROOT vía sudo, porque el dueño/grupo/ACL nuevos ya no son los de
# quien lo invoca) siguiera ese symlink, terminaría haciendo `chown`/`chmod`/`setfacl` como
# root sobre archivos arbitrarios del sistema -- una escalada de privilegios real, no
# teórica. La defensa: CADA apertura de un componente del árbol usa `O_NOFOLLOW` relativo
# al descriptor del padre (`dir_fd`), nunca una ruta de texto vuelta a resolver desde la
# raíz. Si un nombre se convirtió en symlink -- incluso a mitad de la corrida, después de
# haber sido listado como directorio -- el `open(..., O_NOFOLLOW)` sobre ese nombre falla
# con `ELOOP` en vez de seguirlo: la ventana de carrera se cierra en el propio `open()`,
# no en un chequeo previo que dejaría una ventana entre mirar y abrir. Ver
# `_abrir_sin_symlink` y `_caminar`. Las mutaciones (`fchown`, `fchmod`, `setfacl`) actúan
# sobre EL DESCRIPTOR ya abierto (o, para `setfacl`, sobre `/proc/self/fd/N`, que apunta al
# inodo del descriptor, no a un nombre que se pueda haber vuelto a sustituir) -- nunca
# sobre una ruta de texto. Probado empíricamente en hall9000 (2026-09-25): un directorio
# reemplazado por un symlink a otro directorio real DESPUÉS de abrirlo con O_NOFOLLOW no
# contamina al objetivo del symlink cuando se aplica el ACL vía `/proc/self/fd`. Los dos
# tests de este guion (`test_symlink_en_el_punto_de_partida_se_rechaza`,
# `test_symlink_sustituido_a_mitad_de_la_corrida_no_contamina_el_objetivo`) automatizan esa
# prueba.
#
# --- Amenaza (B1): el texto de la ACL no es el permiso efectivo -------------------------
# Un archivo creado con `tempfile.mkstemp()` (modo 0600 explícito, como hacía
# `las_manos/motor_registry/tool_authority.py::_write_file` antes de este cambio) recibe,
# bajo un padre con ACL por defecto, entradas de ACL que SIGUEN MOSTRANDO "rwx" en el
# texto pero cuyo permiso EFECTIVO (entrada ∧ máscara) es "---": el algoritmo de creación
# de ACL interseca el modo pedido con la ACL por defecto, y 0600 pide grupo=0, así que la
# máscara resultante también es 0 -- verificado empíricamente en hall9000 el 2026-09-25.
# `--verificar` en su primera versión miraba el TEXTO ("rwx") y daba "cumple" con el
# archivo en realidad inaccesible para el otro usuario. La corrección: `_permiso_efectivo`
# calcula la intersección bit a bit, nunca confía en el texto pedido. El escritor real que
# lo disparaba (`tool_authority.py::_write_file`) se corrige en el mismo PR (ver el diff de
# ese archivo): `os.fchmod(fd, 0o664)` antes de `os.replace`, con test que reproduce el
# 0600-bajo-ACL-por-defecto y confirma que con el fchmod el efectivo pasa a ser rw.
from __future__ import annotations

import errno
import grp
import json
import os
import pwd
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

USUARIO = "jaxsvc"  # dueño nuevo, y entrada ACL nombrada de usuario
GRUPO = "fruiz"  # grupo nuevo, y entrada ACL nombrada de grupo

_FLAGS_DIR_SEGURO = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_FLAGS_ARCHIVO_SEGURO = os.O_RDONLY | os.O_NOFOLLOW

_ESPECIALES = stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX


class ErrorPermisosProyectos(Exception):
    """Cualquier fallo que tiene que abortar sin aplicar nada más."""


# --- Resolución de RAIZ / PROYECTOS -------------------------------------------------------

def _raiz_por_defecto() -> str:
    try:
        salida = subprocess.run(
            ["sudo", "-n", "grep", "^JAX_WORKSPACE_DIR=", "/etc/jax/.env"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        salida = None
    if salida is not None and salida.returncode == 0 and salida.stdout.strip():
        return salida.stdout.strip().split("=", 1)[1]
    return "/home/fruiz/jax-workspace"


def _resolver_raiz(raiz_arg: str | None) -> str:
    return raiz_arg if raiz_arg is not None else _raiz_por_defecto()


def _validar_y_obtener_proyectos(raiz: str) -> Path:
    """Rechaza RAIZ vacía, `/`, una ruta sin `proyectos/`, o cuyo `proyectos/` sea (o
    cuelgue de) un symlink -- ver la nota de amenaza B3 arriba del archivo."""
    if not raiz or raiz == "/":
        raise ErrorPermisosProyectos(f"RAIZ inválida: {raiz!r}")

    proyectos = Path(raiz) / "proyectos"
    if not proyectos.is_dir():
        raise ErrorPermisosProyectos(f"RAIZ inválida: no existe {proyectos} (una ruta sin proyectos/)")

    # B3: ni el propio proyectos/ ni ningún componente de su ruta puede ser un symlink.
    # os.path.realpath SIGUE symlinks; si el resultado no coincide con la ruta absoluta tal
    # cual se pidió, hay un symlink en algún punto del camino.
    absoluta = os.path.abspath(str(proyectos))
    real = os.path.realpath(absoluta)
    if real != absoluta or os.path.islink(absoluta):
        raise ErrorPermisosProyectos(
            f"RAIZ inválida: {proyectos} es (o cuelga de) un symlink -- no se sigue nunca"
        )

    return proyectos


# --- Recorrido seguro (O_NOFOLLOW + dir_fd, nunca por ruta de texto) ----------------------

def _abrir_dir_seguro(nombre: str, dir_fd: int):
    """`None` si `nombre` es (o se volvió, en el instante de este open) un symlink --
    ELOOP con O_NOFOLLOW es la propia carrera resuelta, no un chequeo previo con ventana."""
    try:
        return os.open(nombre, _FLAGS_DIR_SEGURO, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            return None
        raise


def _abrir_archivo_seguro(nombre: str, dir_fd: int):
    try:
        return os.open(nombre, _FLAGS_ARCHIVO_SEGURO, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            return None
        raise


class Resultado:
    def __init__(self):
        self.no_cumple: list[str] = []  # (--verificar) o fallos de mutación (--aplicar)
        self.symlinks_saltados: list[str] = []
        self.hardlinks_rechazados: list[str] = []
        self.bits_espurios_quitados: list[str] = []
        self.dirs_procesados = 0
        self.archivos_procesados = 0
        self.dot_saltados: list[str] = []


def _es_nombre_excluido(nombre: str) -> bool:
    """m0 (corrección de brief): nombres que empiezan con '.' -- estado propio de
    herramientas (p.ej. `.claude-flow`, típicamente 700) -- no reciben nada y no se
    reportan como incumplimiento; tampoco se recorre dentro de ellos."""
    return nombre.startswith(".")


def _caminar(dir_fd: int, ruta: str, *, mutar: bool, resultado: Resultado,
             hook_de_prueba=None) -> None:
    """Recorre `ruta` (ya abierta como `dir_fd`) de forma segura contra symlinks.
    `mutar=False` -> solo lee y compara (--verificar). `mutar=True` -> corrige
    (--aplicar, siempre corriendo como root). `hook_de_prueba(ruta)` se llama justo
    ANTES de listar cada directorio -- únicamente para que los tests puedan sustituir un
    subdirectorio por un symlink A MITAD de la corrida y comprobar que el objetivo
    sustituido no se contamina."""
    if hook_de_prueba is not None:
        hook_de_prueba(ruta)

    with os.scandir(dir_fd) as it:
        entradas = list(it)

    for entrada in entradas:
        nombre = entrada.name
        ruta_hija = f"{ruta}/{nombre}"

        if _es_nombre_excluido(nombre):
            resultado.dot_saltados.append(ruta_hija)
            continue

        try:
            st = os.stat(nombre, dir_fd=dir_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue

        if stat.S_ISLNK(st.st_mode):
            resultado.symlinks_saltados.append(ruta_hija)
            continue

        if stat.S_ISDIR(st.st_mode):
            fd_hija = _abrir_dir_seguro(nombre, dir_fd)
            if fd_hija is None:  # se volvió symlink justo entre el stat y el open: la
                resultado.symlinks_saltados.append(ruta_hija)  # carrera la resuelve ELOOP.
                continue
            try:
                _revisar_o_mutar_directorio(fd_hija, ruta_hija, mutar=mutar, resultado=resultado)
                resultado.dirs_procesados += 1
                _caminar(fd_hija, ruta_hija, mutar=mutar, resultado=resultado, hook_de_prueba=hook_de_prueba)
            finally:
                os.close(fd_hija)
            continue

        if stat.S_ISREG(st.st_mode):
            fd_hija = _abrir_archivo_seguro(nombre, dir_fd)
            if fd_hija is None:
                resultado.symlinks_saltados.append(ruta_hija)
                continue
            try:
                _revisar_o_mutar_archivo(fd_hija, ruta_hija, st, mutar=mutar, resultado=resultado)
                resultado.archivos_procesados += 1
            finally:
                os.close(fd_hija)
            continue

        # ni symlink, ni directorio, ni archivo regular (socket, fifo, device): no es
        # parte de lo que este guion gobierna, se ignora sin reportar.


# --- ACL: parseo y permiso EFECTIVO (B1) --------------------------------------------------

_RE_ENTRADA = re.compile(
    r"^(default:)?(user|group|mask|other)(?::([^:]*))?:([rwx-]{3})"
)


def _permisos_de(texto: str, prefijo_default: bool, tipo: str, calificador: str | None) -> str | None:
    for linea in texto.splitlines():
        m = _RE_ENTRADA.match(linea)
        if not m:
            continue
        es_default, t, cal, perm = m.groups()
        if bool(es_default) != prefijo_default:
            continue
        if t != tipo:
            continue
        if (cal or "") != (calificador or ""):
            continue
        return perm
    return None


def _bits(perm: str) -> int:
    v = 0
    if "r" in perm:
        v |= 4
    if "w" in perm:
        v |= 2
    if "x" in perm:
        v |= 1
    return v


def _permiso_efectivo(texto_acl: str, *, default: bool, tipo: str, calificador: str) -> int:
    """Permiso EFECTIVO = entrada ∧ máscara -- nunca el texto pedido a secas (B1). Si no
    hay entrada de máscara (ACL mínima, sin entradas nombradas), el efectivo es la propia
    entrada -- ese es el comportamiento POSIX cuando no hay ACL extendida."""
    entrada = _permisos_de(texto_acl, default, tipo, calificador)
    if entrada is None:
        return 0
    mascara = _permisos_de(texto_acl, default, "mask", None)
    if mascara is None:
        return _bits(entrada)
    return _bits(entrada) & _bits(mascara)


def _getfacl(fd: int) -> str:
    # pass_fds es obligatorio: subprocess.run cierra todo descriptor heredado que no esté
    # ahí (close_fds=True es el default desde Python 3.2) -- set_inheritable solo no
    # alcanza, el subprocess lo cerraría antes de que getfacl pudiera abrir
    # /proc/self/fd/N. "self" en el hijo es el propio getfacl, así que el número de
    # descriptor tiene que sobrevivir el exec sin cerrarse: por eso pass_fds.
    os.set_inheritable(fd, True)
    r = subprocess.run(
        ["getfacl", "-p", f"/proc/self/fd/{fd}"],
        pass_fds=(fd,), capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise ErrorPermisosProyectos(f"getfacl falló sobre el descriptor {fd}: {r.stderr}")
    return r.stdout


# --- Chequeo/corrección de un directorio --------------------------------------------------

def _revisar_o_mutar_directorio(fd: int, ruta: str, *, mutar: bool, resultado: Resultado) -> None:
    st = os.fstat(fd)

    if mutar:
        _mutar_directorio(fd, st, ruta, resultado)
        st = os.fstat(fd)  # releer tras mutar, para el chequeo final abajo

    faltas = []

    try:
        grupo_real = grp.getgrgid(st.st_gid).gr_name
    except KeyError:
        grupo_real = str(st.st_gid)
    if grupo_real != GRUPO:
        faltas.append(f"grupo={grupo_real}, esperado={GRUPO}")

    if not (st.st_mode & stat.S_ISGID):
        faltas.append("falta setgid")
    if st.st_mode & stat.S_ISUID:
        faltas.append("setuid espurio")
    if st.st_mode & stat.S_ISVTX:
        faltas.append("sticky espurio")

    texto_acl = _getfacl(fd)
    if _permiso_efectivo(texto_acl, default=False, tipo="user", calificador=USUARIO) & 0o7 != 0o7:
        faltas.append(f"ACL de acceso efectiva insuficiente para u:{USUARIO}")
    if _permiso_efectivo(texto_acl, default=False, tipo="group", calificador=GRUPO) & 0o7 != 0o7:
        faltas.append(f"ACL de acceso efectiva insuficiente para g:{GRUPO}")
    if _permiso_efectivo(texto_acl, default=True, tipo="user", calificador=USUARIO) & 0o7 != 0o7:
        faltas.append(f"ACL por defecto efectiva insuficiente para u:{USUARIO}")
    if _permiso_efectivo(texto_acl, default=True, tipo="group", calificador=GRUPO) & 0o7 != 0o7:
        faltas.append(f"ACL por defecto efectiva insuficiente para g:{GRUPO}")

    if faltas:
        resultado.no_cumple.append(f"{ruta}: {'; '.join(faltas)}")


def _mutar_directorio(fd: int, st: os.stat_result, ruta: str, resultado: Resultado) -> None:
    uid = pwd.getpwnam(USUARIO).pw_uid
    gid = grp.getgrnam(GRUPO).gr_gid
    os.fchown(fd, uid, gid)

    os.set_inheritable(fd, True)
    proc_fd = f"/proc/self/fd/{fd}"
    entrada = f"u:{USUARIO}:rwX,g:{GRUPO}:rwX,m::rwx"
    for extra in ([], ["-d"]):
        r = subprocess.run(
            ["setfacl", *extra, "-m", entrada, proc_fd],
            pass_fds=(fd,), capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise ErrorPermisosProyectos(f"setfacl falló sobre {ruta}: {r.stderr}")

    st_ahora = os.fstat(fd)
    tenia_espurios = bool(st_ahora.st_mode & (stat.S_ISUID | stat.S_ISVTX))
    nuevo_modo = (st_ahora.st_mode & ~(stat.S_ISUID | stat.S_ISVTX)) | stat.S_ISGID
    if nuevo_modo != st_ahora.st_mode:
        os.fchmod(fd, stat.S_IMODE(nuevo_modo))
        if tenia_espurios:
            resultado.bits_espurios_quitados.append(ruta)


# --- Chequeo/corrección de un archivo -----------------------------------------------------

def _revisar_o_mutar_archivo(fd: int, ruta: str, st: os.stat_result, *, mutar: bool,
                              resultado: Resultado) -> None:
    if st.st_nlink > 1:
        # m3: un hardlink comparte inodo con quién sabe qué otra ruta -- cambiarle
        # dueño/permiso desde acá cambiaría también esa otra ruta, que puede estar fuera
        # de proyectos/ por completo. Nunca se muta; se reporta como fallo.
        resultado.hardlinks_rechazados.append(f"{ruta} (nlink={st.st_nlink})")
        return

    if mutar:
        _mutar_archivo(fd, ruta, resultado)
        st = os.fstat(fd)

    faltas = []
    try:
        grupo_real = grp.getgrgid(st.st_gid).gr_name
    except KeyError:
        grupo_real = str(st.st_gid)
    if grupo_real != GRUPO:
        faltas.append(f"grupo={grupo_real}, esperado={GRUPO}")

    if st.st_mode & _ESPECIALES:
        faltas.append("bit especial (setuid/setgid/sticky) presente en archivo")

    texto_acl = _getfacl(fd)
    if _permiso_efectivo(texto_acl, default=False, tipo="user", calificador=USUARIO) & 0o6 != 0o6:
        faltas.append(f"ACL de acceso efectiva insuficiente para u:{USUARIO}")
    if _permiso_efectivo(texto_acl, default=False, tipo="group", calificador=GRUPO) & 0o6 != 0o6:
        faltas.append(f"ACL de acceso efectiva insuficiente para g:{GRUPO}")

    if faltas:
        resultado.no_cumple.append(f"{ruta}: {'; '.join(faltas)}")


def _mutar_archivo(fd: int, ruta: str, resultado: Resultado) -> None:
    uid = pwd.getpwnam(USUARIO).pw_uid
    gid = grp.getgrnam(GRUPO).gr_gid
    os.fchown(fd, uid, gid)

    os.set_inheritable(fd, True)
    proc_fd = f"/proc/self/fd/{fd}"
    r = subprocess.run(
        ["setfacl", "-m", f"u:{USUARIO}:rwX,g:{GRUPO}:rwX,m::rwx", proc_fd],
        pass_fds=(fd,), capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise ErrorPermisosProyectos(f"setfacl falló sobre {ruta}: {r.stderr}")

    st_ahora = os.fstat(fd)
    if st_ahora.st_mode & _ESPECIALES:
        os.fchmod(fd, stat.S_IMODE(st_ahora.st_mode & ~_ESPECIALES))
        resultado.bits_espurios_quitados.append(ruta)


# --- Orquestación de un recorrido completo ------------------------------------------------

def _recorrer(proyectos: Path, *, mutar: bool, hook_de_prueba=None) -> Resultado:
    resultado = Resultado()
    fd_raiz = os.open(str(proyectos.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        fd_proyectos = _abrir_dir_seguro(proyectos.name, fd_raiz)
        if fd_proyectos is None:
            raise ErrorPermisosProyectos(f"{proyectos} se volvió un symlink justo antes de abrirlo")
        try:
            _revisar_o_mutar_directorio(fd_proyectos, str(proyectos), mutar=mutar, resultado=resultado)
            resultado.dirs_procesados += 1
            _caminar(fd_proyectos, str(proyectos), mutar=mutar, resultado=resultado, hook_de_prueba=hook_de_prueba)
        finally:
            os.close(fd_proyectos)
    finally:
        os.close(fd_raiz)
    return resultado


def _reporte_legible(resultado: Resultado) -> tuple[str, bool]:
    lineas = []
    for r in resultado.no_cumple:
        lineas.append(f"NO CUMPLE: {r}")
    for r in resultado.hardlinks_rechazados:
        lineas.append(f"HARDLINK RECHAZADO (no se muta, nlink>1): {r}")
    for r in resultado.symlinks_saltados:
        lineas.append(f"SYMLINK saltado (no se sigue, no se reporta como falta): {r}")
    ok = not resultado.no_cumple and not resultado.hardlinks_rechazados
    return "\n".join(lineas), ok


# --- CLI ------------------------------------------------------------------------------

def _cmd_verificar(raiz: str) -> int:
    proyectos = _validar_y_obtener_proyectos(raiz)
    resultado = _recorrer(proyectos, mutar=False)
    texto, ok = _reporte_legible(resultado)
    if texto:
        print(texto)
    if ok:
        print(f"OK: {proyectos} cumple (dueño {USUARIO}, grupo {GRUPO}, setgid, sin bits "
              f"espurios, ACL de acceso y por defecto efectivas, sin hardlinks).")
        return 0
    return 1


def _respaldo_dir() -> Path:
    return Path(pwd.getpwnam("fruiz").pw_dir) / "respaldos-permisos"


def _hacer_respaldo(proyectos: Path) -> Path:
    """B2: mktemp/O_EXCL (nombre único, sin colisión posible) + `sudo -n getfacl -R -p`
    ANTES de tocar nada; exige rc==0 y tamaño>0, o aborta sin ningún cambio."""
    _respaldo_dir().mkdir(parents=True, exist_ok=True)
    fd, ruta_txt = tempfile.mkstemp(
        dir=str(_respaldo_dir()), prefix="proyectos-", suffix=".acl"
    )
    ruta = Path(ruta_txt)
    try:
        with os.fdopen(fd, "wb") as f:
            r = subprocess.run(
                ["sudo", "-n", "getfacl", "-R", "-p", str(proyectos)],
                stdout=f, stderr=subprocess.PIPE,
            )
        if r.returncode != 0:
            raise ErrorPermisosProyectos(
                f"el respaldo falló (getfacl rc={r.returncode}): {r.stderr.decode(errors='replace')} "
                f"-- no se aplicó ningún cambio."
            )
        if ruta.stat().st_size == 0:
            raise ErrorPermisosProyectos(
                f"el respaldo quedó vacío ({ruta}) -- no se aplicó ningún cambio."
            )
    except Exception:
        raise
    return ruta.resolve()


def _cmd_aplicar(raiz: str) -> int:
    proyectos = _validar_y_obtener_proyectos(raiz)

    if shutil.which("sudo") is None:
        raise ErrorPermisosProyectos("sudo no está disponible -- no se aplicó ningún cambio.")

    ruta_respaldo = _hacer_respaldo(proyectos)
    # M3: ruta ABSOLUTA impresa por el guion -- nunca "~", que bajo sudo no se expande
    # igual (HOME puede cambiar).
    print(f"Respaldo: {ruta_respaldo} (revertir con: sudo setfacl --restore=\"{ruta_respaldo}\")")

    r = subprocess.run(
        ["sudo", "-n", sys.executable, os.path.abspath(__file__),
         "--nucleo-privilegiado", str(proyectos)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise ErrorPermisosProyectos(
            f"el núcleo privilegiado (sudo -n python3 ... --nucleo-privilegiado) falló "
            f"rc={r.returncode}:\nstdout={r.stdout}\nstderr={r.stderr}"
        )
    try:
        datos = json.loads(r.stdout)
    except json.JSONDecodeError as exc:
        raise ErrorPermisosProyectos(f"el núcleo privilegiado no devolvió JSON válido: {exc}\n{r.stdout}")

    for ruta in datos["bits_espurios_quitados"]:
        print(f"bits espurios quitados: {ruta}")
    for ruta in datos["hardlinks_rechazados"]:
        print(f"HARDLINK RECHAZADO (no mutado, nlink>1): {ruta}")
    for ruta in datos["symlinks_saltados"]:
        print(f"SYMLINK saltado: {ruta}")
    print(f"Procesados: {datos['dirs_procesados']} directorios, {datos['archivos_procesados']} archivos.")

    if datos["hardlinks_rechazados"]:
        print("--aplicar encontró hardlinks -- no se mutaron, y el resultado es un fallo.", file=sys.stderr)
        return 1

    # Confirmar con una lectura independiente, sin privilegio, del estado ya aplicado.
    codigo = _cmd_verificar(str(proyectos.parent))
    if codigo != 0:
        print("--aplicar terminó pero --verificar final encontró fallos (ver arriba).", file=sys.stderr)
    else:
        print(f"OK: {proyectos} aplicado y verificado.")
    return codigo


def _cmd_nucleo_privilegiado(proyectos_str: str) -> int:
    """Corre como root (invocado por _cmd_aplicar vía `sudo -n`). Hace la mutación real y
    devuelve un resumen JSON por stdout -- el proceso sin privilegio lo lee e informa."""
    if os.geteuid() != 0:
        print("el núcleo privilegiado tiene que correr como root (sudo -n)", file=sys.stderr)
        return 2
    proyectos = Path(proyectos_str)
    # Re-validar acá también (defensa en profundidad: no confiar ciegamente en que el
    # padre ya validó -- este proceso es el que de verdad tiene el privilegio).
    absoluta = os.path.abspath(str(proyectos))
    real = os.path.realpath(absoluta)
    if real != absoluta or os.path.islink(absoluta):
        print(f"{proyectos} es (o cuelga de) un symlink", file=sys.stderr)
        return 2

    resultado = _recorrer(proyectos, mutar=True)
    print(json.dumps({
        "dirs_procesados": resultado.dirs_procesados,
        "archivos_procesados": resultado.archivos_procesados,
        "symlinks_saltados": resultado.symlinks_saltados,
        "hardlinks_rechazados": resultado.hardlinks_rechazados,
        "bits_espurios_quitados": resultado.bits_espurios_quitados,
    }))
    return 0


def main(argv: list[str]) -> int:
    modo = "--verificar"
    raiz_arg = None
    nucleo_arg = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--verificar", "--aplicar"):
            modo = a
        elif a == "--nucleo-privilegiado":
            i += 1
            nucleo_arg = argv[i]
            modo = "--nucleo-privilegiado"
        elif a.startswith("--"):
            print(f"opción desconocida: {a}", file=sys.stderr)
            return 2
        else:
            raiz_arg = a
        i += 1

    try:
        if modo == "--nucleo-privilegiado":
            return _cmd_nucleo_privilegiado(nucleo_arg)
        raiz = _resolver_raiz(raiz_arg)
        if modo == "--verificar":
            return _cmd_verificar(raiz)
        return _cmd_aplicar(raiz)
    except ErrorPermisosProyectos as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
