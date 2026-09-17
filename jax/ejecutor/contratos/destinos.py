# jax/ejecutor/contratos/destinos.py
"""¿A qué máquinas del inventario toca una llamada de herramienta? (C1 y C2)

Spec 2026-09-15 §4, «verdad incómoda»: C1 y C2 operan por patrones y atajan
ERRORES HONESTOS. Este módulo lee lo escrito a la vista —ssh, scp, rsync, sftp—
y nada más: `python3 -c` con un socket no se ve, `bash -c 'ssh …'` tampoco.

Lo que SÍ garantiza, para fallar cerrado:
- destino escrito que no está en el inventario → HostDesconocido;
- comando que no se puede partir, ssh sin destino, anidamiento > 3 → ComandoIlegible;
- inventario sin exactamente UNA máquina local → HostDesconocido.

Segmentos: se parte por operadores de shell. Un segmento cuyo programa efectivo
es `ssh` toca SÓLO su destino, y su comando remoto se lee recursivo con ese
destino como «local». Cualquier otro segmento toca la local; scp/rsync/sftp tocan
la local y cada remoto que nombran.

Sólo biblioteca estándar: lo corre `axioma` desde /opt/ejecutor/lib.
"""
from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass

PROFUNDIDAD_MAX = 3

_PUNTUACION = ";&|()"
_ASIGNACION = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SSH_CON_VALOR = frozenset("BbcDEeFIiJLlmOoPpQRSWw")
_PREFIJOS = {  # programa -> letras de opción que consumen el token siguiente
    "sudo": frozenset("ugphCDRTU"), "nohup": frozenset(), "exec": frozenset(), "time": frozenset(),
    "command": frozenset(), "nice": frozenset("n"), "ionice": frozenset("cnp"), "stdbuf": frozenset("ioe"),
    "env": frozenset("uCS"),
}
_REMOTO = re.compile(r"^(?:[^@/:\s]+@)?(\[[^\]]+\]|[^/:@\s\[\]]+):")
_USUARIO = re.compile(r"^[^@]*@")


@dataclass(frozen=True)
class Host:
    nombre: str
    ip: str
    puerto: int
    rol: str
    es_local: bool


class ComandoIlegible(ValueError):
    """`args[0]` es un código estable."""


class HostDesconocido(ValueError):
    """`args[0]` es el destino tal como está escrito."""


def _palabras(comando: str) -> list[str]:
    lexer = shlex.shlex(comando, posix=True, punctuation_chars=_PUNTUACION)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        raise ComandoIlegible("comillas_sin_cerrar") from None


def _segmentos(palabras):
    actual: list[str] = []
    for p in palabras:
        if p and set(p) <= set(_PUNTUACION):
            if actual:
                yield actual
            actual = []
        else:
            actual.append(p)
    if actual:
        yield actual


def _saltar_opciones(segmento, i, con_valor):
    while i < len(segmento) and segmento[i].startswith("-") and segmento[i] != "-":
        opcion = segmento[i]
        i += 1
        if len(opcion) == 2 and opcion[1] in con_valor:
            i += 1
    return i


def _programa(segmento):
    i = 0
    while i < len(segmento):
        palabra = segmento[i]
        base = os.path.basename(palabra)
        if _ASIGNACION.match(palabra):
            i += 1
        elif base == "timeout":
            i = _saltar_opciones(segmento, i + 1, frozenset("sk")) + 1
        elif base in _PREFIJOS:
            i = _saltar_opciones(segmento, i + 1, _PREFIJOS[base])
        else:
            return base, segmento[i + 1:]
    return "", []


def _destino_ssh(args):
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            i += 1
            break
        if len(a) > 1 and a.startswith("-"):
            for j in range(1, len(a)):
                if a[j] in _SSH_CON_VALOR:
                    if j == len(a) - 1:
                        i += 1
                    break
            i += 1
            continue
        break
    if i >= len(args):
        raise ComandoIlegible("ssh_sin_destino")
    return args[i], args[i + 1:]


def _host_de_destino(destino: str) -> str:
    if destino.startswith("ssh://"):
        destino = _USUARIO.sub("", destino[len("ssh://"):], count=1)
        if destino.count(":") == 1:
            destino = destino.split(":", 1)[0]
        return destino
    return _USUARIO.sub("", destino, count=1)


def _resolver(escrito: str, hosts) -> str:
    limpio = escrito.strip("[]")
    for h in hosts:
        if limpio in (h.nombre, h.ip):
            return h.nombre
    raise HostDesconocido(escrito)


def _destinos(comando, hosts, local, profundidad):
    if profundidad > PROFUNDIDAD_MAX:
        raise ComandoIlegible("anidamiento_excesivo")
    tocados: set[str] = set()
    for segmento in _segmentos(_palabras(comando)):
        programa, args = _programa(segmento)
        if programa == "ssh":
            destino, remoto = _destino_ssh(args)
            host = _resolver(_host_de_destino(destino), hosts)
            tocados.add(host)
            if remoto:
                tocados |= _destinos(" ".join(remoto), hosts, host, profundidad + 1)
        elif programa == "sftp":
            tocados.add(local)
            destino, _ = _destino_ssh(args)
            tocados.add(_resolver(_host_de_destino(destino.split(":", 1)[0]), hosts))
        elif programa in ("scp", "rsync"):
            tocados.add(local)
            for a in args:
                m = None if a.startswith("-") else _REMOTO.match(a)
                if m:
                    tocados.add(_resolver(m.group(1), hosts))
        else:
            tocados.add(local)
    return tocados or {local}


def destinos(comando: str, hosts) -> frozenset[str]:
    hosts = tuple(hosts)
    locales = [h.nombre for h in hosts if h.es_local]
    if len(locales) != 1:
        raise HostDesconocido("local")
    return frozenset(_destinos(comando, hosts, locales[0], 0))
