"""ACL POSIX por ejecucion, puestas por `faro` SIN root (auditoria de 0.3bc, el ABIERTO del uid de la jaula).

PARA QUE. Un bwrap que corre con el uid de la jaula tiene que poder ABRIR las dos fuentes de sus binds dentro del
directorio de sockets (0700 de `faro`), sin poder LISTARLO y sin tocar lo de otra ejecucion. El dueño de un
archivo puede dar permisos a un uid con nombre sin ser root:

    directorio de sockets   0700 + user:<uid>:--x     atravesar, no listar
    <run_id>.sock           0600 + user:<uid>:rw-
    <run_id>.token          0400 + user:<uid>:r--

La mascara aparece en los bits de grupo (0710, 0660, 0440). Se quitan al cerrar la ejecucion.

COMO. Con el xattr `system.posix_acl_access` por `os.setxattr`/`os.getxattr` (formato del kernel: version 2 y
entradas `tag u16, perm u16, id u32`, little endian): sin el binario `setfacl` ni libacl (no hay `posix1e` en el
Python del servicio y un argv de un binario externo seria otra dependencia). Verificado contra `getfacl`. Si el
sistema de archivos no admite ACL, FALLA CERRADO (`ConfigFaroInvalida`): sin ACL la jaula no llegaria a sus
archivos, y la alternativa seria abrir el directorio.

QUE ACEPTA EL PUERTO (reauditoria R-1: tampoco un directorio con ACL POR DEFECTO, `system.posix_acl_default`,
cuya herencia daria entradas con nombre a los archivos nuevos). `validar_privado`: ni un bit para «otros», grupo propietario sin permisos, ningun grupo con
nombre, y solo entradas de usuario con nombre de uids de jaulas VIVAS y con a lo sumo `--x`; la mascara en los bits
de grupo se explica por esas entradas, no por un `chmod g+x` a secas.
"""
from __future__ import annotations

import errno
import os
import stat
import struct
from collections.abc import Iterable

from .config import ConfigFaroInvalida

XATTR = "system.posix_acl_access"
XATTR_DEFECTO = "system.posix_acl_default"
X, W, R = 1, 2, 4
TAG_USER_OBJ, TAG_USER, TAG_GROUP_OBJ, TAG_GROUP, TAG_MASK, TAG_OTHER = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20
_VERSION = 2
_SIN_ID = 0xFFFFFFFF
_ENTRADA = struct.Struct("<HHI")
_UID_MAX = 2 ** 32 - 2
_SIN_SOPORTE = (errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS)


def _no_soportado(exc: OSError, path) -> ConfigFaroInvalida:
    return ConfigFaroInvalida(f"el sistema de archivos de {path} no admite ACL ({exc.strerror}): sin ellas la jaula no llegaria a sus archivos")


def _decodificar(crudo: bytes) -> list[tuple[int, int, int]]:
    if len(crudo) < 4 or struct.unpack_from("<I", crudo)[0] != _VERSION or (len(crudo) - 4) % _ENTRADA.size:
        raise ConfigFaroInvalida("ACL con un formato que no se reconoce")
    return [_ENTRADA.unpack_from(crudo, 4 + i * _ENTRADA.size) for i in range((len(crudo) - 4) // _ENTRADA.size)]


def _codificar(entradas: Iterable[tuple[int, int, int]]) -> bytes:
    return struct.pack("<I", _VERSION) + b"".join(_ENTRADA.pack(*e) for e in entradas)


def _normalizar(entradas: Iterable[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Ordena como espera el kernel y recalcula la mascara (union de grupo propietario y entradas con nombre)."""
    sin_mascara = [e for e in entradas if e[0] != TAG_MASK]
    con_nombre = [e for e in sin_mascara if e[0] in (TAG_USER, TAG_GROUP)]
    if con_nombre:
        grupo = next((p for t, p, _ in sin_mascara if t == TAG_GROUP_OBJ), 0)
        sin_mascara.append((TAG_MASK, grupo | _union(p for _, p, _ in con_nombre), _SIN_ID))
    return sorted(sin_mascara, key=lambda e: (e[0], e[2]))


def _union(valores: Iterable[int]) -> int:
    r = 0
    for v in valores:
        r |= v
    return r


def _comprobar_no_enlace(path) -> None:
    if stat.S_ISLNK(os.lstat(path).st_mode):
        raise ConfigFaroInvalida(f"{path} es un enlace: no se tocan ACL a traves de enlaces")


def _leer_o_modo(path) -> list[tuple[int, int, int]]:
    """Las entradas de la ACL de `path`; si no tiene ACL con nombre, las tres que da el modo."""
    _comprobar_no_enlace(path)
    try:
        return _decodificar(os.getxattr(path, XATTR, follow_symlinks=False))
    except OSError as exc:
        if exc.errno in _SIN_SOPORTE:
            raise _no_soportado(exc, path) from None
        if exc.errno != errno.ENODATA:
            raise
    m = stat.S_IMODE(os.lstat(path).st_mode)
    return [(TAG_USER_OBJ, (m >> 6) & 7, _SIN_ID), (TAG_GROUP_OBJ, (m >> 3) & 7, _SIN_ID), (TAG_OTHER, m & 7, _SIN_ID)]


def _escribir(path, entradas: Iterable[tuple[int, int, int]]) -> None:
    try:
        os.setxattr(path, XATTR, _codificar(_normalizar(entradas)), follow_symlinks=False)
    except OSError as exc:
        if exc.errno in _SIN_SOPORTE:
            raise _no_soportado(exc, path) from None
        raise


def _uid(uid: object) -> int:
    if isinstance(uid, bool) or not isinstance(uid, int) or not 0 <= uid <= _UID_MAX:
        raise ValueError("uid invalido")
    return uid


def conceder(path, uid: int, perm: int, *, exclusivo: bool = False) -> None:
    """`user:<uid>:<perm>` en `path` (sustituye la que hubiera para ese uid). `perm` es una mascara de R, W, X.

    `exclusivo=True` (el socket y el token, que son de UNA jaula): si el archivo ya trae entradas con nombre que no son
    de ese uid (p. ej. heredadas de una ACL por defecto del directorio) FALLA CERRADO sin tocarlo: la mascara que se
    recalcula es la union de todas las entradas con nombre y una heredada con efecto `---` pasaria a `rwx`."""
    uid = _uid(uid)
    if isinstance(perm, bool) or not isinstance(perm, int) or not 1 <= perm <= 7:
        raise ValueError("permiso invalido")
    actuales = _leer_o_modo(path)
    if exclusivo and any(t == TAG_GROUP or (t == TAG_USER and i != uid) for t, _, i in actuales):
        raise ConfigFaroInvalida(f"{path} ya trae entradas ACL con nombre ajenas al uid {uid} (herencia de una ACL por defecto?): no se toca")
    entradas = [e for e in actuales if not (e[0] == TAG_USER and e[2] == uid)]
    entradas.append((TAG_USER, perm, uid))
    _escribir(path, entradas)


def retirar(path, uid: int) -> None:
    """Quita `user:<uid>` de `path`; si no estaba, no hace nada. Sin entradas con nombre, el modo vuelve a ser el de antes."""
    uid = _uid(uid)
    entradas = _leer_o_modo(path)
    resto = [e for e in entradas if not (e[0] == TAG_USER and e[2] == uid)]
    if len(resto) != len(entradas):
        _escribir(path, resto)


def _tiene_acl_por_defecto(path) -> bool:
    try:
        os.getxattr(path, XATTR_DEFECTO, follow_symlinks=False)
        return True
    except OSError as exc:
        if exc.errno in (errno.ENODATA, *_SIN_SOPORTE):
            return False
        raise


def usuarios(path) -> dict[int, int]:
    """Las entradas de usuario con nombre de `path`: `{uid: perm}`."""
    try:
        entradas = _leer_o_modo(path)
    except ConfigFaroInvalida:
        return {}
    return {i: p for t, p, i in entradas if t == TAG_USER}


def validar_privado(path, uids_permitidos: Iterable[int], *, perm_max: int = X) -> None:
    """El directorio es privado: 0700, o 0700 mas entradas de usuario con nombre de `uids_permitidos` con a lo sumo
    `perm_max`. Cualquier otra cosa (otros, grupo, un grupo con nombre, un uid que no es jaula viva) lo rechaza."""
    permitidos = set(uids_permitidos)
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise ConfigFaroInvalida(f"{path} no es un directorio")
    privado = f"{path} tiene que ser privado (0700)"
    if st.st_mode & 0o007:
        raise ConfigFaroInvalida(f"{privado}: ningun permiso para otros (tiene {stat.S_IMODE(st.st_mode):04o})")
    if _tiene_acl_por_defecto(path):
        raise ConfigFaroInvalida(f"{privado}: tiene una ACL por defecto (herencia): los archivos nuevos heredarian entradas con nombre")
    entradas = _leer_o_modo(path)
    if all(t in (TAG_USER_OBJ, TAG_GROUP_OBJ, TAG_OTHER) for t, _, _ in entradas):
        if st.st_mode & 0o070:
            raise ConfigFaroInvalida(f"{privado}: ningun permiso de grupo ni de otros (tiene {stat.S_IMODE(st.st_mode):04o})")
        return
    for tag, perm, ident in entradas:
        if tag in (TAG_GROUP_OBJ, TAG_OTHER) and perm:
            raise ConfigFaroInvalida(f"{privado}: el grupo propietario y otros no pueden tener permisos (ACL)")
        if tag == TAG_GROUP:
            raise ConfigFaroInvalida(f"{privado}: ningun grupo con nombre en la ACL (grupo {ident})")
        if tag == TAG_USER and (ident not in permitidos or perm & ~perm_max):
            raise ConfigFaroInvalida(f"{privado}: la ACL da paso al uid {ident} y no es una jaula viva, o con mas que {perm_max:o}")
        if tag == TAG_MASK and perm & ~perm_max:
            raise ConfigFaroInvalida(f"{privado}: la mascara de la ACL da mas que {perm_max:o}")
