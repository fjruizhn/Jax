#!/usr/bin/env python3
# ops/permisos_proyectos.py [--verificar [RAIZ] | --aplicar | --deshacer] — spec
# docs/superpowers/specs/2026-09-22-proyectos-y-selector-design.md §5: RAIZ/proyectos/ y
# todo lo de abajo queda DUEÑO jaxsvc, GRUPO fruiz con escritura, setgid en directorios,
# ACL POSIX de acceso y por defecto `u:jaxsvc:rwX,g:fruiz:rwX,m::rwx`.
#
# TERCERA RONDA (2026-09-25) -- RECHAZADO de nuevo, esta vez con 2 BLOCK sobre el DISEÑO
# de la reversión de la ronda 2 (json.py/-I y MAJOR-1/2/3 de esa ronda quedaron cerrados
# por construcción, no hubo que tocarlos de nuevo). Los dos:
#
# BLOCK-1 (reconstruir el árbol desde un respaldo de texto, aunque el recorrido en sí
#   fuera seguro, tenía cuatro fallas de DISEÑO: (a) el respaldo real de producción no
#   trae "default:" en sus bloques de directorio salvo que YA tenga ACL por defecto -- el
#   parser decidía "es directorio" mirando eso, así que un directorio SIN ACL todavía
#   (el estado de HOY) se leía como archivo; (b) no restauraba setgid ni los bits
#   especiales, ni el resto de la máscara; (c) un nombre de archivo con un espacio al
#   final se recortaba al desescapar; (d) `--nucleo-revertir <respaldo> <ruta>` aceptaba
#   CUALQUIER respaldo Y cualquier ruta -- alguien podía fabricar un respaldo con
#   entradas para archivos arbitrarios y pedirle al núcleo root que les hiciera chown).
#   SOLUCIÓN, no parche: se elimina reconstruir-desde-respaldo por completo.
#   `--deshacer`/`--nucleo-deshacer` es un modo nuevo, DETERMINISTA: no lee ningún
#   archivo de estado -- lleva el árbol al único estado que production tiene hoy medido
#   con `stat` real (dueño fruiz:fruiz, sin ninguna ACL, sin bits especiales; el modo se
#   deriva del rwx que el DUEÑO ya tiene en cada objeto, que en el árbol ya aplicado da
#   exactamente 0775/0664 -- los dos casos reales medidos en hall9000 el 2026-09-25,
#   fuera de `.claude-flow`, que este guion nunca toca). El respaldo de `getfacl -R -p`
#   se conserva como REGISTRO FORENSE únicamente -- nunca se usa para reconstruir nada.
#
# BLOCK-2 (con --revertir eliminado, sus tres puntos de entrada al núcleo tenían un
#   problema compartido: aceptaban una ruta arbitraria como argumento -- incluso
#   validada contra symlinks, seguía siendo una ruta que el LLAMADOR elegía). Las TRES
#   entradas del núcleo (`--nucleo-privilegiado`, `--nucleo-respaldo`,
#   `--nucleo-deshacer`) ya NO aceptan ningún argumento de ruta -- cada una resuelve
#   `PROYECTOS` de forma independiente, siempre desde `/etc/jax/.env`, siempre la misma
#   función. Un argumento de más en cualquiera de las tres (`--nucleo-respaldo /etc/ssh`,
#   `--nucleo-deshacer /var/tmp/x`) se rechaza sin tocar nada. Modos repetidos
#   (`--aplicar --verificar`) también se rechazan.
#
# Además: MAJOR-1 de esta ronda -- `getfacl -R -p ... > archivo` puede dar rc==0 aunque
# la escritura haya fallado (verificado empíricamente: `getfacl ... > /dev/full` da
# rc==0). El respaldo ahora hace `fsync`, se relee, se parsea, y se compara la cantidad
# de entradas contra un recorrido independiente del árbol real -- si no coinciden, el
# respaldo se descarta y `--aplicar` aborta sin mutar nada. m1: sin `/etc/jax/.env`
# legible, todo lo que necesita `PROYECTOS` falla cerrado -- ya no hay ningún valor de
# reserva. m2: el sha256 del núcleo instalado se compara contra el CONTENIDO COMMITEADO
# en HEAD (`git show HEAD:ops/permisos_proyectos.py`), no contra el working tree; la
# cadena de directorios padre se revisa con `lstat` (nunca sigue un symlink). m4: la
# exclusión de `NOMBRES_EXCLUIDOS` sólo aplica al primer nivel de cada proyecto
# (`proyectos/<proyecto>/.claude-flow`), nunca en `proyectos/` mismo ni más profundo, y
# se reporta.
from __future__ import annotations

import grp
import hashlib
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

USUARIO = "jaxsvc"  # dueño tras --aplicar, y entrada ACL nombrada de usuario
GRUPO = "fruiz"  # grupo tras --aplicar, y entrada ACL nombrada de grupo
DUENO_ORIGINAL = "fruiz"  # a quien --deshacer devuelve el dueño (mismo nombre que GRUPO
# -- son dos roles distintos que hoy coinciden en la misma cuenta del sistema)

# m4 (ronda 3): sólo en el primer nivel de cada proyecto, nunca en proyectos/ mismo ni
# más profundo -- ver _caminar, que sólo aplica esto cuando profundidad == 2.
NOMBRES_EXCLUIDOS = frozenset({".claude-flow"})

RUTA_INSTALADA = Path("/usr/local/sbin/jax-permisos-proyectos")
RUTA_PYTHON = Path("/usr/bin/python3")
RUTA_RESPALDOS = Path("/var/backups/jax-permisos")
RUTA_ENV = Path("/etc/jax/.env")

_ESPECIALES = stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX


class ErrorPermisosProyectos(Exception):
    """Cualquier fallo que tiene que abortar sin aplicar nada más."""


# ============================================================================
# Resolución de RAIZ / PROYECTOS
# ============================================================================

def _leer_workspace_dir_directo() -> str | None:
    """Sólo para el núcleo privilegiado (ya es root): lee /etc/jax/.env directo, sin
    sudo -n -- root puede leer un archivo root:jaxsvc 640 sin ayuda. None si no se pudo
    leer o la variable no está -- m1, sin valor de reserva: el llamador decide fallar."""
    try:
        contenido = RUTA_ENV.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for linea in contenido.splitlines():
        if linea.startswith("JAX_WORKSPACE_DIR="):
            valor = linea.split("=", 1)[1].strip()
            return valor or None
    return None


def _raiz_configurada_privilegiada() -> Path:
    """Sólo la usan las tres entradas del núcleo (ya son root). m1: sin
    JAX_WORKSPACE_DIR legible, esto FALLA -- nunca un valor fijo por adivinanza."""
    workspace = _leer_workspace_dir_directo()
    if not workspace:
        raise ErrorPermisosProyectos(
            f"no se pudo leer JAX_WORKSPACE_DIR de {RUTA_ENV} -- sin valor de reserva."
        )
    proyectos_str = os.path.join(workspace, "proyectos")
    absoluta = os.path.abspath(proyectos_str)
    if os.path.islink(absoluta) or os.path.realpath(absoluta) != absoluta:
        raise ErrorPermisosProyectos(f"{absoluta} es (o cuelga de) un symlink -- no se sigue nunca")
    if not os.path.isdir(absoluta):
        raise ErrorPermisosProyectos(f"{absoluta} no existe")
    return Path(absoluta)


def _sudo_n_funciona() -> bool:
    """m5 (ronda 4): chequeo AISLADO de si `sudo -n` en sí funciona, separado de si
    /etc/jax/.env tiene o no la variable -- para que un llamador pueda distinguir "sudo
    no funciona" (abortar) de "sudo funciona pero no hay nada que comparar" (seguir)."""
    try:
        return subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _raiz_por_defecto() -> str:
    """m1/m-f (rondas 2 y 3): si `sudo -n` no puede leer /etc/jax/.env (root:jaxsvc 640)
    y no se dio una RAIZ explícita, esto FALLA -- nunca un valor fijo por adivinanza.
    Cadena vacía = "no se pudo resolver"; el llamador decide."""
    try:
        salida = subprocess.run(
            ["sudo", "-n", "grep", "^JAX_WORKSPACE_DIR=", str(RUTA_ENV)],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if salida.returncode == 0 and salida.stdout.strip():
        return salida.stdout.strip().split("=", 1)[1]
    return ""


def _resolver_raiz(raiz_arg: str | None) -> str:
    if raiz_arg is not None:
        return raiz_arg
    resuelta = _raiz_por_defecto()
    if not resuelta:
        raise ErrorPermisosProyectos(
            "no se pudo resolver RAIZ por defecto (sudo -n no leyó JAX_WORKSPACE_DIR de "
            f"{RUTA_ENV}) -- dala explícita: permisos_proyectos.py --verificar RAIZ"
        )
    return resuelta


def _validar_y_obtener_proyectos(raiz: str) -> Path:
    if not raiz or raiz == "/":
        raise ErrorPermisosProyectos(f"RAIZ inválida: {raiz!r}")

    proyectos = Path(raiz) / "proyectos"
    if not proyectos.is_dir():
        raise ErrorPermisosProyectos(f"RAIZ inválida: no existe {proyectos} (una ruta sin proyectos/)")

    absoluta = os.path.abspath(str(proyectos))
    real = os.path.realpath(absoluta)
    if real != absoluta or os.path.islink(absoluta):
        raise ErrorPermisosProyectos(
            f"RAIZ inválida: {proyectos} es (o cuelga de) un symlink -- no se sigue nunca"
        )

    return proyectos


# ============================================================================
# Apertura segura: SIEMPRE O_PATH, SIEMPRE por descriptor
# ============================================================================
#
# O_PATH no requiere NINGÚN permiso sobre el objetivo -- sólo travesía sobre los
# directorios padre, que ya se tiene por haber llegado hasta acá. Verificado
# empíricamente en hall9000: fstat()/getfacl()/setfacl() vía /proc/self/fd/N de un
# descriptor O_PATH funcionan siempre (incluso contra un archivo 0600 ajeno); sobre una
# FIFO NO bloquea; sobre un symlink NO falla -- da un descriptor que fstat() identifica
# como symlink, sin tocar el objetivo. fchown/fchmod necesitan un descriptor "real" --
# se reabre vía /proc/self/fd/N, que corriendo como root nunca falla por permisos.
# Clasificar y actuar son la MISMA apertura: no hay ventana entre "mirar qué es" y
# "usarlo" en la que algo se pueda haber sustituido.

def _abrir_o_path(nombre: str, dir_fd: int) -> int | None:
    """None si la entrada desapareció entre el listado y esta apertura (ENOENT)."""
    try:
        return os.open(nombre, os.O_PATH | os.O_NOFOLLOW, dir_fd=dir_fd)
    except FileNotFoundError:
        return None


def _reabrir_real(fd_path: int, flags: int) -> int:
    return os.open(f"/proc/self/fd/{fd_path}", flags)


def _getfacl(fd_path: int) -> str:
    os.set_inheritable(fd_path, True)
    r = subprocess.run(
        ["getfacl", "-p", f"/proc/self/fd/{fd_path}"],
        pass_fds=(fd_path,), capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise ErrorPermisosProyectos(f"getfacl falló sobre el descriptor {fd_path}: {r.stderr}")
    return r.stdout


def _setfacl(fd_path: int, entrada: str, *, default: bool = False) -> None:
    os.set_inheritable(fd_path, True)
    args = ["setfacl"] + (["-d"] if default else []) + ["-m", entrada, f"/proc/self/fd/{fd_path}"]
    r = subprocess.run(args, pass_fds=(fd_path,), capture_output=True, text=True)
    if r.returncode != 0:
        raise ErrorPermisosProyectos(f"setfacl falló sobre el descriptor {fd_path}: {r.stderr}")


def _limpiar_acl(fd_path: int, ruta: str) -> None:
    """--deshacer: quita la ACL de acceso Y por defecto por completo (verificado que
    `-b -k` no falla sobre un archivo, aunque -k no tenga nada que hacer ahí)."""
    os.set_inheritable(fd_path, True)
    r = subprocess.run(
        ["setfacl", "-b", "-k", f"/proc/self/fd/{fd_path}"],
        pass_fds=(fd_path,), capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise ErrorPermisosProyectos(f"setfacl -b -k falló sobre {ruta}: {r.stderr}")


# ============================================================================
# ACL: parseo y permiso EFECTIVO
# ============================================================================

_RE_ENTRADA = re.compile(r"^(default:)?(user|group|mask|other)(?::([^:]*))?:([rwx-]{3})")


def _permisos_de(texto: str, prefijo_default: bool, tipo: str, calificador: str | None) -> str | None:
    for linea in texto.splitlines():
        m = _RE_ENTRADA.match(linea)
        if not m:
            continue
        es_default, t, cal, perm = m.groups()
        if bool(es_default) != prefijo_default or t != tipo or (cal or "") != (calificador or ""):
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
    entrada = _permisos_de(texto_acl, default, tipo, calificador)
    if entrada is None:
        return 0
    mascara = _permisos_de(texto_acl, default, "mask", None)
    if mascara is None:
        return _bits(entrada)
    return _bits(entrada) & _bits(mascara)


# ============================================================================
# Resultado / exclusión por nombre
# ============================================================================

class Resultado:
    def __init__(self):
        self.no_cumple: list[str] = []
        self.symlinks_saltados: list[str] = []
        self.hardlinks_rechazados: list[str] = []
        self.bits_espurios_quitados: list[str] = []
        self.excluidos: list[str] = []
        self.dirs_procesados = 0
        self.archivos_procesados = 0


def _es_nombre_excluido(nombre: str) -> bool:
    return nombre in NOMBRES_EXCLUIDOS


# ============================================================================
# Recorrido seguro (compartido por verificar, aplicar y deshacer)
# ============================================================================

def _procesar_directorio(fd_path: int, ruta: str, st: os.stat_result, *, accion: str,
                          resultado: Resultado) -> None:
    if accion == "deshacer":
        _deshacer_objeto(fd_path, ruta, resultado)
        return
    _revisar_o_mutar_directorio(fd_path, ruta, st, mutar=(accion == "aplicar"), resultado=resultado)


def _procesar_archivo(fd_path: int, ruta: str, st: os.stat_result, *, accion: str,
                       resultado: Resultado) -> None:
    if st.st_nlink > 1:
        resultado.hardlinks_rechazados.append(f"{ruta} (nlink={st.st_nlink})")
        return
    if accion == "deshacer":
        _deshacer_objeto(fd_path, ruta, resultado)
        return
    _revisar_o_mutar_archivo(fd_path, ruta, st, mutar=(accion == "aplicar"), resultado=resultado)


def _caminar(dir_fd: int, ruta: str, profundidad: int, *, accion: str, resultado: Resultado,
             hook_de_prueba=None) -> None:
    """`profundidad` es la profundidad de las ENTRADAS que se listan en esta llamada,
    relativa a `proyectos/` (sus hijos directos son profundidad 1). m4: la exclusión de
    NOMBRES_EXCLUIDOS sólo aplica en profundidad 2 -- proyectos/<proyecto>/.claude-flow,
    nunca en proyectos/ mismo (profundidad 1) ni más abajo."""
    if hook_de_prueba is not None:
        hook_de_prueba(ruta)

    with os.scandir(dir_fd) as it:
        entradas = list(it)

    for entrada in entradas:
        nombre = entrada.name
        ruta_hija = f"{ruta}/{nombre}"

        if profundidad == 2 and _es_nombre_excluido(nombre):
            resultado.excluidos.append(ruta_hija)
            continue

        fd_path = _abrir_o_path(nombre, dir_fd)
        if fd_path is None:
            continue

        try:
            try:
                st = os.fstat(fd_path)
            except OSError as exc:
                raise ErrorPermisosProyectos(f"fstat inesperado sobre {ruta_hija}: {exc}") from exc

            if stat.S_ISLNK(st.st_mode):
                resultado.symlinks_saltados.append(ruta_hija)
                continue

            if stat.S_ISDIR(st.st_mode):
                _procesar_directorio(fd_path, ruta_hija, st, accion=accion, resultado=resultado)
                resultado.dirs_procesados += 1
                try:
                    fd_listable = _reabrir_real(fd_path, os.O_RDONLY | os.O_DIRECTORY)
                except PermissionError:
                    resultado.no_cumple.append(f"{ruta_hija}: sin permiso para listar el contenido (EACCES)")
                    continue
                try:
                    _caminar(fd_listable, ruta_hija, profundidad + 1, accion=accion,
                             resultado=resultado, hook_de_prueba=hook_de_prueba)
                finally:
                    os.close(fd_listable)
                continue

            if stat.S_ISREG(st.st_mode):
                _procesar_archivo(fd_path, ruta_hija, st, accion=accion, resultado=resultado)
                resultado.archivos_procesados += 1
                continue

            # FIFO, socket, device, etc: no gobernado, se ignora sin reportar.
        finally:
            os.close(fd_path)


def _recorrer(proyectos: Path, *, accion: str, hook_de_prueba=None) -> Resultado:
    resultado = Resultado()
    fd_raiz = os.open(str(proyectos.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        fd_proyectos = _abrir_o_path(proyectos.name, fd_raiz)
        if fd_proyectos is None:
            raise ErrorPermisosProyectos(f"{proyectos} desapareció justo antes de abrirlo")
        try:
            st = os.fstat(fd_proyectos)
            if stat.S_ISLNK(st.st_mode):
                raise ErrorPermisosProyectos(f"{proyectos} se volvió un symlink justo antes de abrirlo")
            _procesar_directorio(fd_proyectos, str(proyectos), st, accion=accion, resultado=resultado)
            resultado.dirs_procesados += 1
            fd_listable = _reabrir_real(fd_proyectos, os.O_RDONLY | os.O_DIRECTORY)
            try:
                _caminar(fd_listable, str(proyectos), 1, accion=accion, resultado=resultado,
                         hook_de_prueba=hook_de_prueba)
            finally:
                os.close(fd_listable)
        finally:
            os.close(fd_proyectos)
    finally:
        os.close(fd_raiz)
    return resultado


# ============================================================================
# --verificar / --aplicar: chequeo/corrección de un directorio o archivo
# ============================================================================

def _revisar_o_mutar_directorio(fd_path: int, ruta: str, st: os.stat_result, *, mutar: bool,
                                 resultado: Resultado) -> None:
    if mutar:
        _mutar_directorio(fd_path, ruta, resultado)
        st = os.fstat(fd_path)

    faltas = []
    try:
        dueno_real = pwd.getpwuid(st.st_uid).pw_name
    except KeyError:
        dueno_real = str(st.st_uid)
    if dueno_real != USUARIO:
        faltas.append(f"dueño={dueno_real}, esperado={USUARIO}")

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

    try:
        texto_acl = _getfacl(fd_path)
    except ErrorPermisosProyectos as exc:
        faltas.append(f"no se pudo leer la ACL: {exc}")
        texto_acl = ""

    if texto_acl:
        if _permiso_efectivo(texto_acl, default=False, tipo="user", calificador=USUARIO) & 0o7 != 0o7:
            faltas.append(f"ACL de acceso efectiva insuficiente para u:{USUARIO}")
        if _permiso_efectivo(texto_acl, default=False, tipo="group", calificador=GRUPO) & 0o7 != 0o7:
            faltas.append(f"ACL de acceso efectiva insuficiente para g:{GRUPO}")
        # m6 (ronda 4): la entrada de grupo DUEÑO (sin nombre) tiene que coincidir --
        # si no, getfacl muestra dos entradas de grupo con valores distintos para el
        # mismo grupo, y --deshacer puede terminar pisado por la vieja (ver ronda 3).
        if _permiso_efectivo(texto_acl, default=False, tipo="group", calificador="") & 0o7 != 0o7:
            faltas.append("ACL de acceso efectiva insuficiente para group:: (grupo dueño)")
        if _permiso_efectivo(texto_acl, default=True, tipo="user", calificador=USUARIO) & 0o7 != 0o7:
            faltas.append(f"ACL por defecto efectiva insuficiente para u:{USUARIO}")
        if _permiso_efectivo(texto_acl, default=True, tipo="group", calificador=GRUPO) & 0o7 != 0o7:
            faltas.append(f"ACL por defecto efectiva insuficiente para g:{GRUPO}")
        if _permiso_efectivo(texto_acl, default=True, tipo="group", calificador="") & 0o7 != 0o7:
            faltas.append("ACL por defecto efectiva insuficiente para group:: (grupo dueño)")

    if faltas:
        resultado.no_cumple.append(f"{ruta}: {'; '.join(faltas)}")


def _mutar_directorio(fd_path: int, ruta: str, resultado: Resultado) -> None:
    uid = pwd.getpwnam(USUARIO).pw_uid
    gid = grp.getgrnam(GRUPO).gr_gid
    fd_real = _reabrir_real(fd_path, os.O_RDONLY)
    try:
        os.fchown(fd_real, uid, gid)
    finally:
        os.close(fd_real)

    # m6 (ronda 4): g::rwX (la entrada de grupo DUEÑO, sin nombre) además de
    # g:{GRUPO}:rwX (la entrada NOMBRADA) -- antes sólo se tocaba la nombrada, y como
    # $GRUPO es también el grupo dueño de estos objetos, `getfacl` mostraba dos
    # entradas para el mismo grupo con valores que podían DIVERGIR (la nombrada
    # correcta, la dueño con lo que tuviera de antes de --aplicar) -- confuso para
    # cualquiera que lea la ACL, y root-cause real del defecto de --deshacer
    # encontrado en la ronda 3 (ver el docstring de _deshacer_objeto).
    entrada = f"u:{USUARIO}:rwX,g:{GRUPO}:rwX,g::rwX,m::rwx"
    _setfacl(fd_path, entrada)
    _setfacl(fd_path, entrada, default=True)

    fd_real = _reabrir_real(fd_path, os.O_RDONLY)
    try:
        st_ahora = os.fstat(fd_real)
        tenia_espurios = bool(st_ahora.st_mode & (stat.S_ISUID | stat.S_ISVTX))
        nuevo_modo = (st_ahora.st_mode & ~(stat.S_ISUID | stat.S_ISVTX)) | stat.S_ISGID
        if nuevo_modo != st_ahora.st_mode:
            os.fchmod(fd_real, stat.S_IMODE(nuevo_modo))
            if tenia_espurios:
                resultado.bits_espurios_quitados.append(ruta)
    finally:
        os.close(fd_real)


def _revisar_o_mutar_archivo(fd_path: int, ruta: str, st: os.stat_result, *, mutar: bool,
                              resultado: Resultado) -> None:
    if mutar:
        _mutar_archivo(fd_path, ruta, resultado)
        st = os.fstat(fd_path)

    faltas = []
    try:
        dueno_real = pwd.getpwuid(st.st_uid).pw_name
    except KeyError:
        dueno_real = str(st.st_uid)
    if dueno_real != USUARIO:
        faltas.append(f"dueño={dueno_real}, esperado={USUARIO}")

    try:
        grupo_real = grp.getgrgid(st.st_gid).gr_name
    except KeyError:
        grupo_real = str(st.st_gid)
    if grupo_real != GRUPO:
        faltas.append(f"grupo={grupo_real}, esperado={GRUPO}")

    if st.st_mode & _ESPECIALES:
        faltas.append("bit especial (setuid/setgid/sticky) presente en archivo")

    try:
        texto_acl = _getfacl(fd_path)
    except ErrorPermisosProyectos as exc:
        faltas.append(f"no se pudo leer la ACL: {exc}")
        texto_acl = ""

    if texto_acl:
        if _permiso_efectivo(texto_acl, default=False, tipo="user", calificador=USUARIO) & 0o6 != 0o6:
            faltas.append(f"ACL de acceso efectiva insuficiente para u:{USUARIO}")
        if _permiso_efectivo(texto_acl, default=False, tipo="group", calificador=GRUPO) & 0o6 != 0o6:
            faltas.append(f"ACL de acceso efectiva insuficiente para g:{GRUPO}")
        # m6 (ronda 4): ver la nota equivalente en _revisar_o_mutar_directorio.
        if _permiso_efectivo(texto_acl, default=False, tipo="group", calificador="") & 0o6 != 0o6:
            faltas.append("ACL de acceso efectiva insuficiente para group:: (grupo dueño)")

    if faltas:
        resultado.no_cumple.append(f"{ruta}: {'; '.join(faltas)}")


def _mutar_archivo(fd_path: int, ruta: str, resultado: Resultado) -> None:
    uid = pwd.getpwnam(USUARIO).pw_uid
    gid = grp.getgrnam(GRUPO).gr_gid
    fd_real = _reabrir_real(fd_path, os.O_RDONLY)
    try:
        os.fchown(fd_real, uid, gid)
    finally:
        os.close(fd_real)

    _setfacl(fd_path, f"u:{USUARIO}:rwX,g:{GRUPO}:rwX,g::rwX,m::rwx")

    fd_real = _reabrir_real(fd_path, os.O_RDONLY)
    try:
        st_ahora = os.fstat(fd_real)
        if st_ahora.st_mode & _ESPECIALES:
            os.fchmod(fd_real, stat.S_IMODE(st_ahora.st_mode & ~_ESPECIALES))
            resultado.bits_espurios_quitados.append(ruta)
    finally:
        os.close(fd_real)


# ============================================================================
# --deshacer: vuelve al estado conocido de hoy (BLOCK-1, ronda 3)
# ============================================================================
#
# Medido con `stat` real en hall9000 el 2026-09-25, fuera de `.claude-flow` (que este
# guion nunca toca): TODO `proyectos/` es hoy fruiz:fruiz, sin ninguna ACL, sin ningún
# bit especial -- 108 directorios en 0775, 249 archivos en 0664 (find ... -printf,
# ver el runbook). Las únicas excepciones medidas (2 directorios en 0700, 1 archivo en
# 0600) viven DENTRO de `.claude-flow`, que --deshacer -- igual que --verificar y
# --aplicar -- nunca alcanza. En vez de fijar 0775/0664 a fuego, el modo se DERIVA del
# rwx que el dueño YA tiene en cada objeto (lo que en el árbol ya aplicado da
# exactamente esos dos números) -- así, si algún día un objeto real dentro del alcance
# de este guion resultara ser una excepción legítima, --deshacer no lo fuerza a
# 0775/0664, preserva lo que su dueño ya podía hacer.

def _modo_deshecho(modo_actual: int) -> int:
    propietario = (modo_actual >> 6) & 0o7
    return (propietario << 6) | (propietario << 3) | (propietario & 0o5)


def _deshacer_objeto(fd_path: int, ruta: str, resultado: Resultado) -> None:
    """Orden verificado empíricamente (hall9000 y en un contenedor limpio, 2026-09-25):
    cuando hay ACL extendida, el bit de GRUPO que se ve por `stat` plano es la MÁSCARA,
    no la entrada `group::` real -- `_mutar_directorio`/`_mutar_archivo` sólo tocan la
    entrada NOMBRADA `g:fruiz:`, nunca `group::` (la entrada "grupo dueño" tradicional),
    así que ésta puede seguir teniendo el valor que tenía al crearse el archivo (0644 si
    quien lo creó tenía umask 022, como jaxsvc en producción). Si se calcula/aplica el
    modo ANTES de quitar la ACL, `setfacl -b` (que corre después) revela esa entrada
    `group::` vieja y pisa el modo recién puesto. Por eso acá la ACL se quita PRIMERO, y
    el modo se calcula y aplica DESPUÉS, como paso final -- el chmod explícito manda
    sobre lo que hubiera antes, sin nadie corriendo después que lo pueda revertir."""
    uid = pwd.getpwnam(DUENO_ORIGINAL).pw_uid
    gid = grp.getgrnam(GRUPO).gr_gid
    fd_real = _reabrir_real(fd_path, os.O_RDONLY)
    try:
        os.fchown(fd_real, uid, gid)
    finally:
        os.close(fd_real)

    _limpiar_acl(fd_path, ruta)

    fd_real = _reabrir_real(fd_path, os.O_RDONLY)
    try:
        st_ahora = os.fstat(fd_real)
        nuevo_modo = _modo_deshecho(st_ahora.st_mode)
        if nuevo_modo != stat.S_IMODE(st_ahora.st_mode):
            os.fchmod(fd_real, nuevo_modo)
    finally:
        os.close(fd_real)


# ============================================================================
# Instalación del núcleo privilegiado
# ============================================================================

def _sha256_de(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def _cadena_es_de_root_sin_escritura_de_grupo_u_otros(ruta: Path) -> str | None:
    """None si toda la cadena (la propia ruta y cada padre hasta la raíz) es de root, sin
    escritura de grupo/otros, y NINGUNO es un symlink -- m2 (ronda 3): lstat en cada
    nivel, nunca stat (que seguiría un symlink intermedio sin decirlo)."""
    actual = ruta
    while True:
        try:
            st = os.lstat(actual)
        except OSError as exc:
            return f"{actual} no se pudo leer: {exc}"
        if stat.S_ISLNK(st.st_mode):
            return f"{actual} es un symlink -- no se sigue nunca"
        if st.st_uid != 0:
            return f"{actual} no es de root (uid={st.st_uid})"
        if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            return f"{actual} es escribible por grupo u otros ({oct(stat.S_IMODE(st.st_mode))})"
        if actual == actual.parent:
            return None
        actual = actual.parent


def _sha256_del_head_committeado() -> tuple[str | None, str]:
    """m2 (ronda 3): el contenido COMMITEADO en HEAD, no el del working tree (que puede
    tener cambios sin commitear que nadie más ve). None + motivo si no se pudo obtener."""
    repo_script = Path(os.path.abspath(__file__))
    try:
        raiz_repo = subprocess.run(
            ["git", "-C", str(repo_script.parent), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=10,
        )
        if raiz_repo.returncode != 0:
            return None, f"no es un checkout de git: {raiz_repo.stderr.strip()}"
        ruta_relativa = repo_script.relative_to(raiz_repo.stdout.strip())
        contenido = subprocess.run(
            ["git", "-C", raiz_repo.stdout.strip(), "show", f"HEAD:{ruta_relativa}"],
            capture_output=True, timeout=10,
        )
        if contenido.returncode != 0:
            return None, f"git show HEAD:{ruta_relativa} falló: {contenido.stderr.decode(errors='replace')}"
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        return None, f"no se pudo consultar git: {exc}"
    return _sha256_de(contenido.stdout), ""


def _verificar_instalacion() -> str | None:
    """None si el núcleo instalado existe, su cadena de directorios es segura (lstat,
    root, sin escritura de grupo/otros), y su sha256 coincide con el HEAD commiteado."""
    if not RUTA_INSTALADA.is_file():
        return f"{RUTA_INSTALADA} no está instalado"
    motivo_cadena = _cadena_es_de_root_sin_escritura_de_grupo_u_otros(RUTA_INSTALADA)
    if motivo_cadena:
        return motivo_cadena
    sha_head, motivo_git = _sha256_del_head_committeado()
    if sha_head is None:
        return f"no se pudo obtener el sha256 de HEAD para comparar: {motivo_git}"
    if _sha256_de(RUTA_INSTALADA.read_bytes()) != sha_head:
        return f"{RUTA_INSTALADA} no coincide con HEAD:ops/permisos_proyectos.py (sha256 distinto -- reinstalar)"
    return None


def _invocar_nucleo(*args: str) -> subprocess.CompletedProcess:
    motivo = _verificar_instalacion()
    if motivo:
        raise ErrorPermisosProyectos(
            f"el núcleo privilegiado no está instalado de forma segura ({motivo}) -- "
            f"instalar con: sudo install -o root -g root -m 0755 "
            f"{os.path.abspath(__file__)} {RUTA_INSTALADA}"
        )
    if shutil.which("sudo") is None:
        raise ErrorPermisosProyectos("sudo no está disponible -- no se aplicó ningún cambio.")
    return subprocess.run(
        ["sudo", "-n", str(RUTA_PYTHON), "-I", str(RUTA_INSTALADA), *args],
        capture_output=True, text=True,
    )


# ============================================================================
# Respaldo: REGISTRO FORENSE únicamente (ronda 3 -- ya no se usa para revertir nada)
# ============================================================================

_RE_ESCAPE = re.compile(r"\\\\|\\[0-7]{3}")


def _desescapar_getfacl(texto: str) -> str:
    """getfacl -p escapa '\\' como '\\\\' y cualquier byte problemático como '\\OOO' (3
    dígitos octales) en la línea '# file: ...'. Un espacio al FINAL del nombre también
    se escapa (verificado: si no, getfacl -R -p perdería el espacio al re-parsear su
    propia salida) -- por eso este desescape opera sobre la línea completa, sin recortar
    espacios antes de aplicarlo (ver _parsear_respaldo)."""
    def _uno(m: re.Match) -> str:
        s = m.group(0)
        if s == "\\\\":
            return "\\"
        return chr(int(s[1:], 8))
    return _RE_ESCAPE.sub(_uno, texto)


_MARCADOR_FIN_RESPALDO = "# JAX_PERMISOS_PROYECTOS_RESPALDO_COMPLETO\n"


def _parsear_respaldo(contenido: str) -> list[str]:
    """Devuelve la lista de rutas que el respaldo registra -- sólo para el conteo
    forense (MAJOR-1). Ya no se usa para restaurar nada: no hace falta distinguir
    directorio de archivo ni guardar owner/ACL (ver el BLOCK-1 de la ronda 3, que
    retiró reconstruir-desde-respaldo por completo).

    m2 (ronda 4): un respaldo cortado a mitad de escritura -- incluso justo después de
    "# file: /a/b" en el ÚLTIMO bloque, sin nada más -- pasaba antes como válido, porque
    esta función sólo miraba si la primera línea de cada bloque empezaba con "# file:".
    Ahora exige que cada bloque traiga owner, group, y las tres entradas base
    (user::/group::/other::) -- lo que `getfacl -p` SIEMPRE escribe para un bloque
    completo, verificado empíricamente. Un bloque incompleto hace que TODO el respaldo
    se rechace (no sólo ese bloque): un respaldo parcialmente confiable no es un
    respaldo confiable. `_cmd_nucleo_respaldo`, además, exige el marcador de fin
    (`_MARCADOR_FIN_RESPALDO`) que él mismo escribe después de que `getfacl` termina --
    eso atrapa un corte que caiga EXACTO en un borde de bloque (un caso que la
    validación por bloque, sola, no vería mal)."""
    rutas = []
    bloques = contenido.split("\n\n")
    for bloque in bloques:
        lineas = [l for l in bloque.splitlines() if l.strip()]
        if not lineas or not lineas[0].startswith("# file:"):
            continue
        # NO se hace .strip() sobre el resto -- un espacio final es parte del nombre
        # real (ver el docstring de _desescapar_getfacl).
        ruta = _desescapar_getfacl(lineas[0][len("# file:"):].lstrip(" ").rstrip("\r\n"))

        tiene_owner = any(l.startswith("# owner:") for l in lineas)
        tiene_group = any(l.startswith("# group:") for l in lineas)
        tiene_triada = (
            any(l.startswith("user::") for l in lineas)
            and any(l.startswith("group::") for l in lineas)
            and any(l.startswith("other::") for l in lineas)
        )
        if not (tiene_owner and tiene_group and tiene_triada):
            raise ErrorPermisosProyectos(
                f"el respaldo está truncado o corrupto: el bloque de {ruta!r} no trae "
                f"owner/group/user::/group::/other:: completos -- no es confiable, se descarta"
            )
        rutas.append(ruta)
    return rutas


def _contar_objetos_reales(proyectos: Path) -> tuple[int, list[str]]:
    """Recorrido de sólo lectura, SIN aplicar NOMBRES_EXCLUIDOS (getfacl -R tampoco lo
    sabe), que cuenta cada directorio y archivo regular real bajo proyectos/ (incluido
    proyectos/ mismo), symlinks excluidos. Devuelve `(cantidad, no_gobernados)` --
    `no_gobernados` son rutas que no son symlink, directorio, ni archivo regular (FIFO,
    socket, device...). `getfacl -R -p` SÍ las enumera como su propio bloque "# file:"
    (verificado empíricamente: una FIFO aparece igual que cualquier archivo), pero este
    guion las ignora en todos sus recorridos (igual que `_caminar`) -- así que si el
    árbol tiene alguna, el conteo NUNCA podría coincidir con el del respaldo por más que
    todo lo demás esté bien. `_cmd_nucleo_respaldo` revisa `no_gobernados` ANTES de
    comparar cantidades, para poder nombrar la ruta exacta en vez de dejar un "no
    coincide" genérico que no dice dónde está el problema."""
    contador = 0
    no_gobernados: list[str] = []

    def _contar_hijos(dir_fd: int, ruta: str) -> None:
        nonlocal contador
        with os.scandir(dir_fd) as it:
            entradas = list(it)
        for entrada in entradas:
            ruta_hija = f"{ruta}/{entrada.name}"
            fd_path = _abrir_o_path(entrada.name, dir_fd)
            if fd_path is None:
                continue
            try:
                st = os.fstat(fd_path)
                if stat.S_ISLNK(st.st_mode):
                    continue
                if stat.S_ISDIR(st.st_mode):
                    contador += 1
                    try:
                        fd_listable = _reabrir_real(fd_path, os.O_RDONLY | os.O_DIRECTORY)
                    except PermissionError:
                        continue
                    try:
                        _contar_hijos(fd_listable, ruta_hija)
                    finally:
                        os.close(fd_listable)
                elif stat.S_ISREG(st.st_mode):
                    contador += 1
                else:
                    no_gobernados.append(ruta_hija)
            finally:
                os.close(fd_path)

    fd_raiz = os.open(str(proyectos.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        fd_proyectos = _abrir_o_path(proyectos.name, fd_raiz)
        if fd_proyectos is None:
            return 0, no_gobernados
        try:
            st = os.fstat(fd_proyectos)
            if stat.S_ISLNK(st.st_mode):
                return 0, no_gobernados
            contador += 1
            fd_listable = _reabrir_real(fd_proyectos, os.O_RDONLY | os.O_DIRECTORY)
            try:
                _contar_hijos(fd_listable, str(proyectos))
            finally:
                os.close(fd_listable)
        finally:
            os.close(fd_proyectos)
    finally:
        os.close(fd_raiz)
    return contador, no_gobernados


def _hacer_respaldo() -> Path:
    """Registro forense únicamente -- ya no se usa para revertir nada (ver --deshacer)."""
    r = _invocar_nucleo("--nucleo-respaldo")
    if r.returncode != 0:
        raise ErrorPermisosProyectos(f"el respaldo falló: {r.stdout}{r.stderr} -- no se aplicó ningún cambio.")
    ruta_texto = r.stdout.strip()
    if not ruta_texto:
        raise ErrorPermisosProyectos("el respaldo no devolvió ninguna ruta -- no se aplicó ningún cambio.")
    return Path(ruta_texto)


def _generar_respaldo_validado(proyectos: Path) -> Path:
    """El núcleo real de --nucleo-respaldo, separado de la fijación de RAIZ que hace
    _cmd_nucleo_respaldo() -- así los tests pueden ejercitar esta lógica (m1, m2, ronda
    4) contra un árbol de prueba arbitrario, sin pasar por la RAIZ configurada. Corre
    como root (mkstemp en RUTA_RESPALDOS, que es root:root 0700). Levanta
    ErrorPermisosProyectos con el motivo si el respaldo no es confiable -- nunca deja un
    archivo parcial: se borra en el propio `finally`."""
    RUTA_RESPALDOS.mkdir(parents=True, exist_ok=True)
    os.chmod(RUTA_RESPALDOS, 0o700)
    fd, ruta_txt = tempfile.mkstemp(dir=str(RUTA_RESPALDOS), prefix="proyectos-", suffix=".acl")
    ruta = Path(ruta_txt)
    ok = False
    try:
        with os.fdopen(fd, "wb") as f:
            r = subprocess.run(["getfacl", "-R", "-p", str(proyectos)], stdout=f, stderr=subprocess.PIPE)
            if r.returncode == 0:
                # m2 (ronda 4): marcador de fin escrito por ESTE proceso, como paso
                # aparte de lo que getfacl escribió -- atrapa un corte que caiga justo
                # en un borde de bloque, que la validación por bloque de
                # _parsear_respaldo, sola, no vería mal.
                f.write(_MARCADOR_FIN_RESPALDO.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        if r.returncode != 0:
            raise ErrorPermisosProyectos(f"getfacl rc={r.returncode}: {r.stderr.decode(errors='replace')}")

        # MAJOR-1 (ronda 3): rc==0 no es evidencia suficiente (verificado empíricamente:
        # getfacl -R -p ... > /dev/full también da rc==0). Se relee lo escrito, se exige
        # el marcador de fin, y se compara la cantidad de entradas contra un recorrido
        # independiente del árbol real.
        contenido = ruta.read_text(encoding="utf-8", errors="surrogateescape")
        if not contenido.strip():
            raise ErrorPermisosProyectos("el respaldo quedó vacío")
        if not contenido.endswith(_MARCADOR_FIN_RESPALDO):
            raise ErrorPermisosProyectos(
                "el respaldo parece truncado (falta el marcador de fin) -- no es confiable"
            )
        contenido_sin_marcador = contenido[: -len(_MARCADOR_FIN_RESPALDO)]

        n_respaldo = len(_parsear_respaldo(contenido_sin_marcador))

        # m1 (ronda 4): una FIFO/socket/device en el árbol hace que getfacl -R la
        # enumere (verificado empíricamente) pero _contar_objetos_reales la ignore
        # (igual que --verificar/--aplicar) -- eso nunca podría dar el mismo número por
        # más que todo lo demás esté bien. Se detecta y se nombra la ruta ANTES de
        # comparar cantidades, en vez de dejar un "no coincide" genérico.
        n_real, no_gobernados = _contar_objetos_reales(proyectos)
        if no_gobernados:
            raise ErrorPermisosProyectos(
                "el árbol tiene objetos que no son symlink, directorio ni archivo "
                "regular (FIFO/socket/device) -- getfacl los cuenta, este guion no, así "
                "que el respaldo nunca sería confiable mientras estén: "
                + ", ".join(no_gobernados)
            )
        if n_respaldo != n_real:
            raise ErrorPermisosProyectos(
                f"el respaldo tiene {n_respaldo} entradas pero el recorrido real ve "
                f"{n_real} -- no es confiable"
            )

        ok = True
        return ruta
    finally:
        if not ok:
            ruta.unlink(missing_ok=True)


def _cmd_nucleo_respaldo() -> int:
    if os.geteuid() != 0:
        print("el núcleo de respaldo tiene que correr como root", file=sys.stderr)
        return 2
    try:
        proyectos = _raiz_configurada_privilegiada()
    except ErrorPermisosProyectos as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        ruta = _generar_respaldo_validado(proyectos)
    except ErrorPermisosProyectos as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(str(ruta))
    return 0


# ============================================================================
# CLI
# ============================================================================

def _cmd_verificar(raiz: str) -> int:
    proyectos = _validar_y_obtener_proyectos(raiz)
    resultado = _recorrer(proyectos, accion="verificar")

    lineas = []
    for r in resultado.no_cumple:
        lineas.append(f"NO CUMPLE: {r}")
    for r in resultado.hardlinks_rechazados:
        lineas.append(f"HARDLINK RECHAZADO (no se muta, nlink>1): {r}")
    for r in resultado.symlinks_saltados:
        lineas.append(f"SYMLINK saltado (no se sigue, no se reporta como falta): {r}")
    for r in resultado.excluidos:
        lineas.append(f"excluido por nombre (sin tocar): {r}")
    if lineas:
        print("\n".join(lineas))

    motivo_instalacion = _verificar_instalacion()
    if motivo_instalacion:
        print(f"núcleo privilegiado NO instalado de forma segura: {motivo_instalacion}")

    ok = not resultado.no_cumple and not resultado.hardlinks_rechazados
    if ok:
        print(f"OK: {proyectos} cumple (dueño {USUARIO}, grupo {GRUPO}, setgid, sin bits "
              f"espurios, ACL de acceso y por defecto efectivas, sin hardlinks).")
        return 0
    return 1


def _cmd_aplicar(raiz: str) -> int:
    proyectos = _validar_y_obtener_proyectos(raiz)

    # La integridad del núcleo instalado se revisa ANTES que cualquier otra cosa -- si
    # no es de confianza, ninguna otra comprobación (ni siquiera cuál RAIZ se pidió)
    # importa todavía. _invocar_nucleo() la vuelve a exigir de todas formas cuando de
    # verdad se invoca; esto sólo adelanta el mismo motivo con un mensaje más claro.
    motivo_instalacion = _verificar_instalacion()
    if motivo_instalacion:
        raise ErrorPermisosProyectos(
            f"el núcleo privilegiado no está instalado de forma segura ({motivo_instalacion}) -- "
            f"instalar con: sudo install -o root -g root -m 0755 "
            f"{os.path.abspath(__file__)} {RUTA_INSTALADA}"
        )

    # Chequeo de cortesía: la RAIZ pedida tiene que ser la configurada -- el núcleo la
    # vuelve a comprobar de forma independiente y es la aplicación real de esto, pero
    # fallar acá antes de intentar el respaldo evita un mensaje confuso.
    #
    # m5 (ronda 4): si `sudo -n` en sí no puede correr (no hay ticket, sudo no está,
    # timeout), _raiz_por_defecto() devolvía "" -- lo mismo que devuelve cuando
    # /etc/jax/.env es legible pero simplemente no tiene la variable -- y este chequeo
    # trataba los dos casos igual: "no hay nada que comparar, seguir". Eso es fallar
    # ABIERTO ante un problema del propio chequeo, no ante una ausencia legítima. Ahora
    # se prueba `sudo -n true` aparte primero: si ESO falla, se aborta con un motivo
    # explícito; sólo si sudo -n funciona pero la variable no está, se sigue de largo
    # (ahí sí no hay nada que comparar).
    if not _sudo_n_funciona():
        raise ErrorPermisosProyectos(
            "no se pudo confirmar la RAIZ configurada (sudo -n no funciona) -- "
            "--aplicar no sigue sin poder hacer esa comparación."
        )
    configurada = _raiz_por_defecto()
    if configurada:
        absoluta_pedida = os.path.realpath(str(proyectos))
        absoluta_configurada = os.path.realpath(os.path.join(configurada, "proyectos"))
        if absoluta_pedida != absoluta_configurada:
            raise ErrorPermisosProyectos(
                f"{absoluta_pedida} no es la RAIZ configurada ({absoluta_configurada}) -- "
                f"--aplicar sólo actúa sobre la RAIZ configurada en {RUTA_ENV}."
            )

    ruta_respaldo = _hacer_respaldo()
    print(f"Respaldo (registro forense -- NO se usa para revertir, ver --deshacer): {ruta_respaldo}")

    r = _invocar_nucleo("--nucleo-privilegiado")
    if r.returncode != 0:
        raise ErrorPermisosProyectos(
            f"el núcleo privilegiado falló rc={r.returncode}:\nstdout={r.stdout}\nstderr={r.stderr}"
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
    for ruta in datos.get("excluidos", []):
        print(f"excluido por nombre (sin tocar): {ruta}")
    print(f"Procesados: {datos['dirs_procesados']} directorios, {datos['archivos_procesados']} archivos.")

    if datos["hardlinks_rechazados"]:
        print("--aplicar encontró hardlinks -- no se mutaron, y el resultado es un fallo.", file=sys.stderr)
        return 1

    codigo = _cmd_verificar(str(proyectos.parent))
    if codigo != 0:
        print("--aplicar terminó pero --verificar final encontró fallos (ver arriba).", file=sys.stderr)
    else:
        print(f"OK: {proyectos} aplicado y verificado.")
    return codigo


def _cmd_deshacer() -> int:
    """Sin argumento de RAIZ -- siempre la configurada (ver el docstring del módulo,
    BLOCK-2 de esta ronda)."""
    r = _invocar_nucleo("--nucleo-deshacer")
    if r.returncode != 0:
        raise ErrorPermisosProyectos(
            f"el núcleo de deshacer falló rc={r.returncode}:\nstdout={r.stdout}\nstderr={r.stderr}"
        )
    try:
        datos = json.loads(r.stdout)
    except json.JSONDecodeError as exc:
        raise ErrorPermisosProyectos(f"el núcleo de deshacer no devolvió JSON válido: {exc}\n{r.stdout}")

    for ruta in datos["hardlinks_rechazados"]:
        print(f"HARDLINK, NO deshecho: {ruta}")
    for ruta in datos["symlinks_saltados"]:
        print(f"SYMLINK saltado, NO deshecho: {ruta}")
    for ruta in datos.get("excluidos", []):
        print(f"excluido por nombre (sin tocar): {ruta}")
    print(f"Deshechos: {datos['dirs_procesados']} directorios, {datos['archivos_procesados']} archivos.")

    if datos["hardlinks_rechazados"]:
        print("--deshacer encontró hardlinks -- no se tocaron, y el resultado es un fallo.", file=sys.stderr)
        return 1
    print("OK: deshecho.")
    return 0


def _cmd_nucleo_privilegiado() -> int:
    if os.geteuid() != 0:
        print("el núcleo privilegiado tiene que correr como root (sudo -n)", file=sys.stderr)
        return 2
    try:
        proyectos = _raiz_configurada_privilegiada()
    except ErrorPermisosProyectos as exc:
        print(str(exc), file=sys.stderr)
        return 2

    resultado = _recorrer(proyectos, accion="aplicar")
    print(json.dumps({
        "dirs_procesados": resultado.dirs_procesados,
        "archivos_procesados": resultado.archivos_procesados,
        "symlinks_saltados": resultado.symlinks_saltados,
        "hardlinks_rechazados": resultado.hardlinks_rechazados,
        "bits_espurios_quitados": resultado.bits_espurios_quitados,
        "excluidos": resultado.excluidos,
    }))
    return 0


def _cmd_nucleo_deshacer() -> int:
    if os.geteuid() != 0:
        print("el núcleo de deshacer tiene que correr como root (sudo -n)", file=sys.stderr)
        return 2
    try:
        proyectos = _raiz_configurada_privilegiada()
    except ErrorPermisosProyectos as exc:
        print(str(exc), file=sys.stderr)
        return 2

    resultado = _recorrer(proyectos, accion="deshacer")
    print(json.dumps({
        "dirs_procesados": resultado.dirs_procesados,
        "archivos_procesados": resultado.archivos_procesados,
        "symlinks_saltados": resultado.symlinks_saltados,
        "hardlinks_rechazados": resultado.hardlinks_rechazados,
        "excluidos": resultado.excluidos,
    }))
    return 0


_MODOS = ("--verificar", "--aplicar", "--deshacer",
          "--nucleo-privilegiado", "--nucleo-respaldo", "--nucleo-deshacer")
_MODOS_SIN_ARGUMENTOS = ("--deshacer", "--nucleo-privilegiado", "--nucleo-respaldo", "--nucleo-deshacer")

_AYUDA = f"""\
uso: permisos_proyectos.py [--verificar [RAIZ] | --aplicar | --deshacer]

  --verificar [RAIZ]  Solo lectura. Sin RAIZ, usa la configurada en {RUTA_ENV}.
                       Es el modo por defecto si no se da ningún flag.
  --aplicar            Dueño {USUARIO}, grupo {GRUPO}, setgid, ACL de acceso y por
                       defecto. Sólo actúa sobre la RAIZ configurada (nunca acepta
                       una RAIZ distinta) -- necesita el GO de Fernando en producción.
  --deshacer           DETERMINISTA, sin argumentos: lleva el árbol a
                       {DUENO_ORIGINAL}:{GRUPO} 0775 (directorios) / 0664 (archivos),
                       sin ninguna ACL -- el estado medido con `stat` real en
                       producción el 2026-09-25. Esa tabla es una VERDAD
                       OPERACIONAL, no una constante: sólo es exacta mientras el
                       árbol siga homogéneo desde esa fecha -- volver a medir antes
                       de usarlo si pasó tiempo (ver
                       docs/runbooks/workspace-proyectos.md).

Sin ningún flag, equivale a --verificar.
"""


def main(argv: list[str]) -> int:
    if any(a in ("-h", "--help") for a in argv):
        print(_AYUDA, end="")
        return 0

    modo = None
    raiz_arg = None
    modos_vistos = 0
    raices_vistas = 0

    for a in argv:
        if a in _MODOS:
            modo = a
            modos_vistos += 1
        elif a.startswith("--"):
            print(f"opción desconocida: {a}", file=sys.stderr)
            return 2
        else:
            raiz_arg = a
            raices_vistas += 1

    if modos_vistos > 1:
        print("modos repetidos -- se pasó más de un flag de modo", file=sys.stderr)
        return 2
    # m5 (ronda 4): antes, una segunda RAIZ posicional pisaba en silencio a la primera
    # (quedaba la ÚLTIMA, sin avisar) -- ahora es un rechazo explícito.
    if raices_vistas > 1:
        print(f"más de una RAIZ -- se dieron {raices_vistas} argumentos posicionales", file=sys.stderr)
        return 2
    if modo is None:
        modo = "--verificar"

    # BLOCK-2 (ronda 3): --deshacer y las tres entradas del núcleo NO aceptan ningún
    # argumento de ruta -- siempre actúan sobre la RAIZ configurada.
    if modo in _MODOS_SIN_ARGUMENTOS and raiz_arg is not None:
        print(f"{modo} no acepta argumentos -- se dio {raiz_arg!r}", file=sys.stderr)
        return 2

    try:
        if modo == "--nucleo-privilegiado":
            return _cmd_nucleo_privilegiado()
        if modo == "--nucleo-respaldo":
            return _cmd_nucleo_respaldo()
        if modo == "--nucleo-deshacer":
            return _cmd_nucleo_deshacer()
        if modo == "--deshacer":
            return _cmd_deshacer()
        raiz = _resolver_raiz(raiz_arg)
        if modo == "--verificar":
            return _cmd_verificar(raiz)
        return _cmd_aplicar(raiz)
    except ErrorPermisosProyectos as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
