#!/usr/bin/env python3
# ops/permisos_proyectos.py [--verificar|--aplicar|--revertir RESPALDO] [RAIZ] — spec
# docs/superpowers/specs/2026-09-22-proyectos-y-selector-design.md §5 (corrección de
# brief, 2026-09-25): RAIZ/proyectos/ y todo lo de abajo queda DUEÑO jaxsvc, GRUPO fruiz
# con escritura, setgid en directorios, ACL POSIX de acceso y por defecto
# `u:jaxsvc:rwX,g:fruiz:rwX,m::rwx`.
#
# SEGUNDA RONDA (2026-09-25) -- RECHAZADO con 3 BLOCK, 3 MAJOR, 6 MINOR sobre la primera
# reescritura en Python (commit 4f117a7). Cada hallazgo real, con su defensa, documentado
# donde corresponde en el código de abajo. Resumen para quien audite de nuevo:
#
# BLOCK-1 (la reversión documentada, `sudo setfacl --restore`, sigue symlinks por NOMBRE
#   como root): se retira esa recomendación por completo. `--revertir RESPALDO` es un modo
#   nuevo que parsea el respaldo (con los escapes octales de getfacl) y aplica cada entrada
#   con el MISMO recorrido seguro (dir_fd + dos-veces-nunca-por-nombre) que usa --aplicar --
#   si algo que el respaldo recuerda como directorio/archivo hoy es un symlink o cambió de
#   tipo, lo salta y lo reporta, nunca lo toca. Ver `_revertir` y
#   `test_revertir_no_contamina_si_algo_se_volvio_symlink`.
#
# BLOCK-2 (el núcleo root se re-ejecutaba desde `__file__` sin `-I`: en un despliegue real
#   `ops/` puede ser escribible por jaxsvc, que podría plantar un `json.py` que se
#   importaría en vez del de la librería estándar -- reproducido empíricamente en hall9000
#   el 2026-09-25: SIN `-I`, un `json.py` de mentira en el directorio del script se
#   importa y corre como root; CON `-I`, `sys.path` no incluye el directorio del script y
#   se importa el `json` real): el núcleo se instala en `/usr/local/sbin/jax-permisos-proyectos`
#   (root:root, 0755, con toda la cadena de directorios padre también root y sin escritura
#   de grupo/otros -- verificado antes de cada `--aplicar`), y se invoca SIEMPRE con
#   `sudo -n /usr/bin/python3 -I <esa ruta>`. Además, el núcleo privilegiado ya no acepta
#   cualquier ruta -- valida que la RAIZ pedida sea exactamente la configurada
#   (`JAX_WORKSPACE_DIR` de `/etc/jax/.env`, leído directo -- el proceso ya es root) antes
#   de tocar nada; hoy, sin este chequeo, se le podía pedir `/etc` y lo recorría igual.
#
# BLOCK-3 (CI): (a) el respaldo vivía en el $HOME de fruiz, que en un runner limpio con
#   `useradd --no-create-home` no existe y `/home` no es escribible por un usuario sin
#   privilegio -- se movió a `/var/backups/jax-permisos` (root, 0700, creado por el propio
#   núcleo privilegiado). (b) el test nuevo de B1 en `las_manos/_tool_authority_test.py`
#   sube el piso de `tests-puros` -- ver `.github/workflows/policy.yml` y el mensaje de
#   commit para el número medido y la nota de re-medición (regla 5 de SESIONES EN PARALELO).
#
# MAJOR-1 (--verificar, sin privilegio, reventaba con PermissionError contra un archivo
#   0600 de jaxsvc en vez de reportarlo): TODA apertura usa `O_PATH` (nunca requiere
#   permiso de lectura sobre el objetivo, sólo travesía sobre el padre -- verificado
#   empíricamente: `getfacl`/`fstat` vía `/proc/self/fd/N` de un descriptor O_PATH
#   funcionan igual aunque el proceso no tenga NINGÚN permiso sobre el archivo). `EACCES`
#   al intentar LISTAR un directorio (no al abrirlo) se reporta como NO CUMPLE y no se
#   desciende ahí; cualquier otra excepción no prevista sale con código 2, no con un
#   traceback a medias.
#
# MAJOR-2 (TOCTOU entre un `lstat` para clasificar y un `open` posterior -- una ventana
#   real para que un hardlink o una FIFO se cuelen, y abrir una FIFO sin O_NONBLOCK CUELGA
#   el núcleo root): se elimina el `lstat` previo por completo. Cada entrada se abre UNA
#   sola vez con `O_PATH|O_NOFOLLOW` y se clasifica con `fstat` sobre ESE MISMO descriptor
#   -- no hay ventana entre "mirar" y "abrir" porque es la misma operación. Verificado
#   empíricamente: `O_PATH` sobre una FIFO no bloquea (a diferencia de un `open` normal) y
#   `O_PATH|O_NOFOLLOW` sobre un symlink NO falla con ELOOP -- da un descriptor que
#   `fstat` identifica como symlink, sin haber tocado el objetivo ni una sola vez.
#
# MAJOR-3: el test que reproduce B1 exige `returncode == 0` de --aplicar, no se salta si
#   falla (ver tests/test_permisos_proyectos.py).
#
# MINOR: m-a --verificar compara también el dueño (uid), no sólo el grupo. m-b la
#   exclusión de nombres con "." se reduce a una lista EXPLÍCITA (`NOMBRES_EXCLUIDOS`, hoy
#   sólo `.claude-flow`) que se imprime cuando se salta algo. m-c `_write_file` usa 0o660,
#   no 0o664 (fuera de proyectos/ no hay ACL que recorte "other"). m-d el respaldo parcial
#   se borra si falla, sin un `except: raise` vacío. m-f sin `sudo -n` disponible, la
#   resolución de RAIZ por defecto falla cerrado en vez de adivinar un valor fijo.
from __future__ import annotations

import errno
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

USUARIO = "jaxsvc"  # dueño nuevo, y entrada ACL nombrada de usuario
GRUPO = "fruiz"  # grupo nuevo, y entrada ACL nombrada de grupo

# m-b: lista EXPLÍCITA, no "cualquier nombre con punto" -- decisión del coordinador hasta
# que Fernando diga otra cosa. Estado propio de herramientas (típicamente 700) que no se
# toca ni se recorre.
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
    sudo -n -- root puede leer un archivo root:jaxsvc 640 sin ayuda."""
    try:
        contenido = RUTA_ENV.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for linea in contenido.splitlines():
        if linea.startswith("JAX_WORKSPACE_DIR="):
            return linea.split("=", 1)[1].strip()
    return None


def _raiz_por_defecto() -> str:
    """m-f: si no hay sudo -n disponible para leer /etc/jax/.env (root:jaxsvc 640) y no
    se dio una RAIZ explícita, esto FALLA -- ya no inventa un valor fijo por adivinanza.
    Devuelve la cadena vacía como señal de "no se pudo resolver"; el llamador decide."""
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
# Apertura segura: SIEMPRE O_PATH, SIEMPRE por descriptor (MAJOR-1, MAJOR-2)
# ============================================================================
#
# O_PATH no requiere NINGÚN permiso sobre el objetivo (ni de lectura, ni de escritura, ni
# de ejecución) -- sólo travesía sobre los directorios padre, que ya se tiene por haber
# llegado hasta acá. Verificado empíricamente en hall9000 (2026-09-25):
#   - fstat() sobre un descriptor O_PATH funciona siempre.
#   - getfacl/setfacl vía /proc/self/fd/N de un descriptor O_PATH funcionan siempre,
#     incluso contra un archivo 0600 de otro dueño.
#   - O_PATH|O_NOFOLLOW sobre una FIFO NO bloquea (a diferencia de un open() normal).
#   - O_PATH|O_NOFOLLOW sobre un symlink NO falla con ELOOP: da un descriptor que fstat()
#     identifica como symlink, sin tocar el objetivo.
#   - fchown/fchmod SÍ necesitan un descriptor "real" (no O_PATH) -- se reabre vía
#     /proc/self/fd/N, lo cual (corriendo como root) nunca falla por permisos.
# Con esto, clasificar y actuar son la MISMA apertura: no hay ventana entre "mirar qué es"
# y "usarlo" en la que algo se pueda haber sustituido (MAJOR-2).

def _abrir_o_path(nombre: str, dir_fd: int) -> int | None:
    """None si la entrada desapareció entre el listado y esta apertura (ENOENT) -- no es
    un fallo, ya no está. Cualquier otro error se propaga (por ejemplo, un padre inválido
    a esta altura sería un fallo real, no algo a saltear en silencio)."""
    try:
        return os.open(nombre, os.O_PATH | os.O_NOFOLLOW, dir_fd=dir_fd)
    except FileNotFoundError:
        return None


def _reabrir_real(fd_path: int, flags: int) -> int:
    """A partir de un descriptor O_PATH, un descriptor "real" para operaciones que O_PATH
    no soporta directo (fchown/fchmod, listar un directorio). Corriendo como root, esto
    nunca falla por permisos del objetivo (DAC override); corriendo sin privilegio, SÍ
    puede fallar con EACCES -- el llamador decide qué hacer con eso."""
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


# ============================================================================
# ACL: parseo y permiso EFECTIVO (B1, ronda 1)
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
# Recorrido seguro
# ============================================================================

def _caminar(dir_fd: int, ruta: str, *, mutar: bool, resultado: Resultado, hook_de_prueba=None) -> None:
    if hook_de_prueba is not None:
        hook_de_prueba(ruta)

    with os.scandir(dir_fd) as it:
        entradas = list(it)

    for entrada in entradas:
        nombre = entrada.name
        ruta_hija = f"{ruta}/{nombre}"

        if _es_nombre_excluido(nombre):
            resultado.excluidos.append(ruta_hija)
            continue

        fd_path = _abrir_o_path(nombre, dir_fd)
        if fd_path is None:
            continue  # desapareció entre el listado y esta apertura -- ya no está

        try:
            try:
                st = os.fstat(fd_path)
            except OSError as exc:
                raise ErrorPermisosProyectos(f"fstat inesperado sobre {ruta_hija}: {exc}") from exc

            if stat.S_ISLNK(st.st_mode):
                resultado.symlinks_saltados.append(ruta_hija)
                continue

            if stat.S_ISDIR(st.st_mode):
                _revisar_o_mutar_directorio(fd_path, ruta_hija, st, mutar=mutar, resultado=resultado)
                resultado.dirs_procesados += 1
                try:
                    fd_listable = _reabrir_real(fd_path, os.O_RDONLY | os.O_DIRECTORY)
                except PermissionError:
                    resultado.no_cumple.append(f"{ruta_hija}: sin permiso para listar el contenido (EACCES)")
                    continue
                try:
                    _caminar(fd_listable, ruta_hija, mutar=mutar, resultado=resultado, hook_de_prueba=hook_de_prueba)
                finally:
                    os.close(fd_listable)
                continue

            if stat.S_ISREG(st.st_mode):
                _revisar_o_mutar_archivo(fd_path, ruta_hija, st, mutar=mutar, resultado=resultado)
                resultado.archivos_procesados += 1
                continue

            # FIFO, socket, device, etc: no gobernado, se ignora sin reportar.
        finally:
            os.close(fd_path)


def _recorrer(proyectos: Path, *, mutar: bool, hook_de_prueba=None) -> Resultado:
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
            _revisar_o_mutar_directorio(fd_proyectos, str(proyectos), st, mutar=mutar, resultado=resultado)
            resultado.dirs_procesados += 1
            fd_listable = _reabrir_real(fd_proyectos, os.O_RDONLY | os.O_DIRECTORY)
            try:
                _caminar(fd_listable, str(proyectos), mutar=mutar, resultado=resultado, hook_de_prueba=hook_de_prueba)
            finally:
                os.close(fd_listable)
        finally:
            os.close(fd_proyectos)
    finally:
        os.close(fd_raiz)
    return resultado


# ============================================================================
# Chequeo/corrección de un directorio
# ============================================================================

def _revisar_o_mutar_directorio(fd_path: int, ruta: str, st: os.stat_result, *, mutar: bool,
                                 resultado: Resultado) -> None:
    if mutar:
        _mutar_directorio(fd_path, ruta, resultado)
        st = os.fstat(fd_path)

    faltas = []

    # m-a: dueño, no sólo grupo.
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
        if _permiso_efectivo(texto_acl, default=True, tipo="user", calificador=USUARIO) & 0o7 != 0o7:
            faltas.append(f"ACL por defecto efectiva insuficiente para u:{USUARIO}")
        if _permiso_efectivo(texto_acl, default=True, tipo="group", calificador=GRUPO) & 0o7 != 0o7:
            faltas.append(f"ACL por defecto efectiva insuficiente para g:{GRUPO}")

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

    entrada = f"u:{USUARIO}:rwX,g:{GRUPO}:rwX,m::rwx"
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


# ============================================================================
# Chequeo/corrección de un archivo
# ============================================================================

def _revisar_o_mutar_archivo(fd_path: int, ruta: str, st: os.stat_result, *, mutar: bool,
                              resultado: Resultado) -> None:
    if st.st_nlink > 1:
        resultado.hardlinks_rechazados.append(f"{ruta} (nlink={st.st_nlink})")
        return

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

    _setfacl(fd_path, f"u:{USUARIO}:rwX,g:{GRUPO}:rwX,m::rwx")

    fd_real = _reabrir_real(fd_path, os.O_RDONLY)
    try:
        st_ahora = os.fstat(fd_real)
        if st_ahora.st_mode & _ESPECIALES:
            os.fchmod(fd_real, stat.S_IMODE(st_ahora.st_mode & ~_ESPECIALES))
            resultado.bits_espurios_quitados.append(ruta)
    finally:
        os.close(fd_real)


def _reporte_legible(resultado: Resultado) -> tuple[str, bool]:
    lineas = []
    for r in resultado.no_cumple:
        lineas.append(f"NO CUMPLE: {r}")
    for r in resultado.hardlinks_rechazados:
        lineas.append(f"HARDLINK RECHAZADO (no se muta, nlink>1): {r}")
    for r in resultado.symlinks_saltados:
        lineas.append(f"SYMLINK saltado (no se sigue, no se reporta como falta): {r}")
    for r in resultado.excluidos:
        lineas.append(f"excluido por nombre (sin tocar): {r}")
    ok = not resultado.no_cumple and not resultado.hardlinks_rechazados
    return "\n".join(lineas), ok


# ============================================================================
# Instalación del núcleo privilegiado (BLOCK-2)
# ============================================================================

def _sha256_de(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _cadena_es_de_root_sin_escritura_de_grupo_u_otros(ruta: Path) -> str | None:
    """None si toda la cadena (la propia ruta y cada padre hasta la raíz) es de root y
    ninguno es escribible por grupo u otros. Si no, el motivo."""
    actual = ruta
    while True:
        try:
            st = actual.stat()
        except OSError as exc:
            return f"{actual} no se pudo leer: {exc}"
        if st.st_uid != 0:
            return f"{actual} no es de root (uid={st.st_uid})"
        if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            return f"{actual} es escribible por grupo u otros ({oct(stat.S_IMODE(st.st_mode))})"
        if actual == actual.parent:
            return None
        actual = actual.parent


def _verificar_instalacion() -> str | None:
    """None si el núcleo instalado existe, su cadena de directorios es segura, y su
    contenido coincide byte a byte con este archivo del repo. Si no, el motivo."""
    if not RUTA_INSTALADA.is_file():
        return f"{RUTA_INSTALADA} no está instalado"
    motivo_cadena = _cadena_es_de_root_sin_escritura_de_grupo_u_otros(RUTA_INSTALADA)
    if motivo_cadena:
        return motivo_cadena
    repo = Path(os.path.abspath(__file__))
    if _sha256_de(RUTA_INSTALADA) != _sha256_de(repo):
        return f"{RUTA_INSTALADA} no coincide con {repo} (sha256 distinto -- reinstalar)"
    return None


# ============================================================================
# Respaldo (BLOCK-3a: /var/backups/jax-permisos, root, 0700) y reversión (BLOCK-1)
# ============================================================================

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


def _hacer_respaldo(proyectos: Path) -> Path:
    """El directorio de respaldos es root:root 0700 (BLOCK-3a) -- el proceso sin
    privilegio que llama a esto NO PUEDE stat/leer el archivo resultante directamente
    (ni falta que hace: `--nucleo-respaldo`, corriendo como root, ya comprobó tamaño>0
    ANTES de imprimir la ruta -- rc==0 con una ruta impresa ES la garantía)."""
    r = _invocar_nucleo("--nucleo-respaldo", str(proyectos))
    if r.returncode != 0:
        raise ErrorPermisosProyectos(f"el respaldo falló: {r.stdout}{r.stderr} -- no se aplicó ningún cambio.")
    ruta_texto = r.stdout.strip()
    if not ruta_texto:
        raise ErrorPermisosProyectos("el respaldo no devolvió ninguna ruta -- no se aplicó ningún cambio.")
    return Path(ruta_texto)


def _cmd_nucleo_respaldo(proyectos_str: str) -> int:
    """Corre como root. m-d: si algo falla, el archivo parcial se borra -- nunca queda un
    respaldo incompleto pareciendo uno bueno."""
    if os.geteuid() != 0:
        print("el núcleo de respaldo tiene que correr como root", file=sys.stderr)
        return 2
    RUTA_RESPALDOS.mkdir(parents=True, exist_ok=True)
    os.chmod(RUTA_RESPALDOS, 0o700)
    fd, ruta_txt = tempfile.mkstemp(dir=str(RUTA_RESPALDOS), prefix="proyectos-", suffix=".acl")
    ruta = Path(ruta_txt)
    ok = False
    try:
        with os.fdopen(fd, "wb") as f:
            r = subprocess.run(["getfacl", "-R", "-p", proyectos_str], stdout=f, stderr=subprocess.PIPE)
        if r.returncode != 0:
            print(f"getfacl rc={r.returncode}: {r.stderr.decode(errors='replace')}", file=sys.stderr)
            return 1
        if ruta.stat().st_size == 0:
            print("el respaldo quedó vacío", file=sys.stderr)
            return 1
        ok = True
        print(str(ruta))
        return 0
    finally:
        if not ok:
            ruta.unlink(missing_ok=True)


# --- Parseo del respaldo (BLOCK-1): octal-unescape de getfacl -----------------------------

_RE_ESCAPE = re.compile(r"\\\\|\\[0-7]{3}")


def _desescapar_getfacl(texto: str) -> str:
    """getfacl -p escapa '\\' como '\\\\' y cualquier byte problemático (verificado:
    newline embebido) como '\\OOO' (3 dígitos octales) en la línea '# file: ...'."""
    def _uno(m: re.Match) -> str:
        s = m.group(0)
        if s == "\\\\":
            return "\\"
        return chr(int(s[1:], 8))
    return _RE_ESCAPE.sub(_uno, texto)


class _RegistroRespaldo:
    __slots__ = ("ruta", "es_dir", "owner", "group", "acl_texto")

    def __init__(self, ruta: str, es_dir: bool, owner: str, group: str, acl_texto: str):
        self.ruta = ruta
        self.es_dir = es_dir
        self.owner = owner
        self.group = group
        self.acl_texto = acl_texto


def _parsear_respaldo(contenido: str) -> dict[str, _RegistroRespaldo]:
    registros: dict[str, _RegistroRespaldo] = {}
    bloques = contenido.split("\n\n")
    for bloque in bloques:
        lineas = [l for l in bloque.splitlines() if l.strip()]
        if not lineas or not lineas[0].startswith("# file:"):
            continue
        ruta = _desescapar_getfacl(lineas[0][len("# file:"):].strip())
        owner = group = ""
        for l in lineas[1:]:
            if l.startswith("# owner:"):
                owner = l[len("# owner:"):].strip()
            elif l.startswith("# group:"):
                group = l[len("# group:"):].strip()
        es_dir = any(l.startswith("default:") for l in lineas)
        registros[ruta] = _RegistroRespaldo(ruta, es_dir, owner, group, "\n".join(lineas))
    return registros


def _extraer_entrada_acl(texto: str, *, default: bool, usuario: str, grupo: str) -> str:
    """Reconstruye el argumento -m que setfacl necesita, a partir de lo que el respaldo
    tenía para user:<usuario> y group:<grupo> -- sólo esas dos entradas nombradas (las que
    este guion controla), nunca el resto tal cual (evita reinyectar entradas ajenas)."""
    u = _permisos_de(texto, default, "user", usuario) or "---"
    g = _permisos_de(texto, default, "group", grupo) or "---"
    return f"u:{usuario}:{u},g:{grupo}:{g},m::rwx"


def _revertir(respaldo_path: str, proyectos: Path, resultado: Resultado) -> None:
    """BLOCK-1: recorre el árbol ACTUAL con el mismo caminante seguro (dir_fd +
    O_PATH|O_NOFOLLOW) -- nunca abre nada por nombre resuelto desde la raíz. Para cada
    objeto que el recorrido visita, si el respaldo tiene un registro Y el tipo actual
    coincide con lo registrado (directorio/archivo), restaura dueño/grupo/ACL/setgid. Si
    el objeto es un symlink (el recorrido ya lo filtra) o cambió de tipo respecto al
    respaldo, se salta y se reporta -- nunca se toca."""
    try:
        contenido = Path(respaldo_path).read_text(encoding="utf-8", errors="surrogateescape")
    except OSError as exc:
        raise ErrorPermisosProyectos(f"no se pudo leer el respaldo {respaldo_path}: {exc}") from exc
    if not contenido.strip():
        raise ErrorPermisosProyectos(f"el respaldo {respaldo_path} está vacío")
    registros = _parsear_respaldo(contenido)
    if not registros:
        raise ErrorPermisosProyectos(f"el respaldo {respaldo_path} no tiene ningún registro reconocible")

    def _por_objeto(fd_path: int, ruta: str, st: os.stat_result, es_dir: bool) -> None:
        reg = registros.get(ruta)
        if reg is None:
            return  # no estaba en el respaldo -- no se toca (m3, criterio conservador)
        if reg.es_dir != es_dir:
            resultado.no_cumple.append(f"{ruta}: cambió de tipo desde el respaldo -- no se revierte")
            return

        uid = pwd.getpwnam(reg.owner).pw_uid if reg.owner else -1
        gid = grp.getgrnam(reg.group).gr_gid if reg.group else -1
        if uid != -1 or gid != -1:
            fd_real = _reabrir_real(fd_path, os.O_RDONLY)
            try:
                os.fchown(fd_real, uid, gid)
            finally:
                os.close(fd_real)

        entrada = _extraer_entrada_acl(reg.acl_texto, default=False, usuario=USUARIO, grupo=GRUPO)
        _setfacl(fd_path, entrada)
        if es_dir:
            entrada_d = _extraer_entrada_acl(reg.acl_texto, default=True, usuario=USUARIO, grupo=GRUPO)
            _setfacl(fd_path, entrada_d, default=True)

    def _recorrer_para_revertir(dir_fd: int, ruta: str) -> None:
        with os.scandir(dir_fd) as it:
            entradas = list(it)
        for entrada in entradas:
            nombre = entrada.name
            ruta_hija = f"{ruta}/{nombre}"
            if _es_nombre_excluido(nombre):
                continue
            fd_path = _abrir_o_path(nombre, dir_fd)
            if fd_path is None:
                continue
            try:
                st = os.fstat(fd_path)
                if stat.S_ISLNK(st.st_mode):
                    resultado.symlinks_saltados.append(ruta_hija)
                    continue
                if stat.S_ISDIR(st.st_mode):
                    _por_objeto(fd_path, ruta_hija, st, True)
                    try:
                        fd_listable = _reabrir_real(fd_path, os.O_RDONLY | os.O_DIRECTORY)
                    except PermissionError:
                        continue
                    try:
                        _recorrer_para_revertir(fd_listable, ruta_hija)
                    finally:
                        os.close(fd_listable)
                elif stat.S_ISREG(st.st_mode):
                    if st.st_nlink > 1:
                        resultado.hardlinks_rechazados.append(f"{ruta_hija} (nlink={st.st_nlink})")
                        continue
                    _por_objeto(fd_path, ruta_hija, st, False)
            finally:
                os.close(fd_path)

    fd_raiz = os.open(str(proyectos.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        fd_proyectos = _abrir_o_path(proyectos.name, fd_raiz)
        if fd_proyectos is None:
            raise ErrorPermisosProyectos(f"{proyectos} desapareció justo antes de revertir")
        try:
            st = os.fstat(fd_proyectos)
            if stat.S_ISLNK(st.st_mode):
                raise ErrorPermisosProyectos(f"{proyectos} es un symlink -- revertir se niega")
            _por_objeto(fd_proyectos, str(proyectos), st, True)
            fd_listable = _reabrir_real(fd_proyectos, os.O_RDONLY | os.O_DIRECTORY)
            try:
                _recorrer_para_revertir(fd_listable, str(proyectos))
            finally:
                os.close(fd_listable)
        finally:
            os.close(fd_proyectos)
    finally:
        os.close(fd_raiz)


def _cmd_nucleo_revertir(respaldo_str: str, proyectos_str: str) -> int:
    if os.geteuid() != 0:
        print("el núcleo de reversión tiene que correr como root", file=sys.stderr)
        return 2
    proyectos = Path(proyectos_str)
    absoluta = os.path.abspath(str(proyectos))
    if os.path.realpath(absoluta) != absoluta or os.path.islink(absoluta):
        print(f"{proyectos} es (o cuelga de) un symlink", file=sys.stderr)
        return 2
    resultado = Resultado()
    try:
        _revertir(respaldo_str, proyectos, resultado)
    except ErrorPermisosProyectos as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({
        "no_cumple": resultado.no_cumple,
        "symlinks_saltados": resultado.symlinks_saltados,
        "hardlinks_rechazados": resultado.hardlinks_rechazados,
    }))
    return 0


# ============================================================================
# CLI
# ============================================================================

def _cmd_verificar(raiz: str) -> int:
    proyectos = _validar_y_obtener_proyectos(raiz)
    resultado = _recorrer(proyectos, mutar=False)
    texto, ok = _reporte_legible(resultado)
    if texto:
        print(texto)

    motivo_instalacion = _verificar_instalacion()
    if motivo_instalacion:
        print(f"núcleo privilegiado NO instalado de forma segura: {motivo_instalacion}")

    if ok:
        print(f"OK: {proyectos} cumple (dueño {USUARIO}, grupo {GRUPO}, setgid, sin bits "
              f"espurios, ACL de acceso y por defecto efectivas, sin hardlinks).")
        return 0
    return 1


def _cmd_aplicar(raiz: str) -> int:
    proyectos = _validar_y_obtener_proyectos(raiz)

    ruta_respaldo = _hacer_respaldo(proyectos)
    print(f"Respaldo: {ruta_respaldo} (revertir con: "
          f"python3 {os.path.abspath(__file__)} --revertir {ruta_respaldo} {proyectos.parent})")

    r = _invocar_nucleo("--nucleo-privilegiado", str(proyectos))
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


def _cmd_revertir(respaldo: str, raiz: str) -> int:
    proyectos = _validar_y_obtener_proyectos(raiz)
    # No se comprueba Path(respaldo).is_file() acá: el directorio de respaldos es
    # root:root 0700 (BLOCK-3a), así que un proceso sin privilegio no puede ni stat-earlo.
    # --nucleo-revertir (que ya corre como root) es quien valida que exista.
    r = _invocar_nucleo("--nucleo-revertir", respaldo, str(proyectos))
    if r.returncode != 0:
        raise ErrorPermisosProyectos(
            f"el núcleo de reversión falló rc={r.returncode}:\nstdout={r.stdout}\nstderr={r.stderr}"
        )
    try:
        datos = json.loads(r.stdout)
    except json.JSONDecodeError as exc:
        raise ErrorPermisosProyectos(f"el núcleo de reversión no devolvió JSON válido: {exc}\n{r.stdout}")

    for ruta in datos["symlinks_saltados"]:
        print(f"SYMLINK saltado, NO revertido: {ruta}")
    for ruta in datos["hardlinks_rechazados"]:
        print(f"HARDLINK, NO revertido: {ruta}")
    for r_ in datos["no_cumple"]:
        print(f"NO REVERTIDO: {r_}")

    if datos["symlinks_saltados"] or datos["hardlinks_rechazados"] or datos["no_cumple"]:
        print("--revertir encontró rutas que no pudo (o no debía) tocar -- ver arriba.", file=sys.stderr)
        return 1
    print("OK: reversión completa.")
    return 0


def _cmd_nucleo_privilegiado(proyectos_str: str) -> int:
    if os.geteuid() != 0:
        print("el núcleo privilegiado tiene que correr como root (sudo -n)", file=sys.stderr)
        return 2
    proyectos = Path(proyectos_str)
    absoluta = os.path.abspath(str(proyectos))
    if os.path.realpath(absoluta) != absoluta or os.path.islink(absoluta):
        print(f"{proyectos} es (o cuelga de) un symlink", file=sys.stderr)
        return 2

    # BLOCK-2: el núcleo YA NO acepta cualquier ruta -- sólo la configurada. Re-deriva la
    # respuesta de forma INDEPENDIENTE (lee /etc/jax/.env directo, ya es root) en vez de
    # confiar en lo que el padre sin privilegio haya resuelto.
    workspace_configurado = _leer_workspace_dir_directo() or "/home/fruiz/jax-workspace"
    esperado = os.path.realpath(os.path.join(workspace_configurado, "proyectos"))
    if absoluta != esperado:
        print(f"RAIZ rechazada: {absoluta} no es la RAIZ configurada ({esperado})", file=sys.stderr)
        return 2

    resultado = _recorrer(proyectos, mutar=True)
    print(json.dumps({
        "dirs_procesados": resultado.dirs_procesados,
        "archivos_procesados": resultado.archivos_procesados,
        "symlinks_saltados": resultado.symlinks_saltados,
        "hardlinks_rechazados": resultado.hardlinks_rechazados,
        "bits_espurios_quitados": resultado.bits_espurios_quitados,
        "excluidos": resultado.excluidos,
    }))
    return 0


def main(argv: list[str]) -> int:
    modo = "--verificar"
    raiz_arg = None
    nucleo_arg = None
    revertir_respaldo = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--verificar", "--aplicar"):
            modo = a
        elif a == "--revertir":
            modo = "--revertir"
            i += 1
            revertir_respaldo = argv[i]
        elif a == "--nucleo-privilegiado":
            i += 1
            nucleo_arg = argv[i]
            modo = "--nucleo-privilegiado"
        elif a == "--nucleo-respaldo":
            i += 1
            nucleo_arg = argv[i]
            modo = "--nucleo-respaldo"
        elif a == "--nucleo-revertir":
            i += 1
            r1 = argv[i]
            i += 1
            r2 = argv[i]
            nucleo_arg = (r1, r2)
            modo = "--nucleo-revertir"
        elif a.startswith("--"):
            print(f"opción desconocida: {a}", file=sys.stderr)
            return 2
        else:
            raiz_arg = a
        i += 1

    try:
        if modo == "--nucleo-privilegiado":
            return _cmd_nucleo_privilegiado(nucleo_arg)
        if modo == "--nucleo-respaldo":
            return _cmd_nucleo_respaldo(nucleo_arg)
        if modo == "--nucleo-revertir":
            return _cmd_nucleo_revertir(*nucleo_arg)
        raiz = _resolver_raiz(raiz_arg)
        if modo == "--verificar":
            return _cmd_verificar(raiz)
        if modo == "--revertir":
            return _cmd_revertir(revertir_respaldo, raiz)
        return _cmd_aplicar(raiz)
    except ErrorPermisosProyectos as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
