"""Whitelist derivada por job (esqueleto de la interfaz; la implementación viene en el commit siguiente)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable


class DetectorNoDisponible(RuntimeError):
    """Una fuente OBLIGATORIA no respondió: no hay whitelist, nunca una vacía."""


@dataclass(frozen=True)
class EsquemaDB:
    columnas: tuple[tuple[str, str], ...] = ()
    funciones: frozenset[str] = frozenset()
    palabras: frozenset[str] = frozenset()
    motores: frozenset[str] = frozenset()
    charsets: frozenset[str] = frozenset()


@dataclass(frozen=True)
class EstadoGit:
    head: str = ""
    status_hash: str = ""
    entradas: tuple[tuple[str, str], ...] = ()  # (sha del blob, ruta)


@dataclass(frozen=True)
class Fuentes:
    huella: Callable[[], Awaitable[str]]
    esquema: Callable[[], Awaitable[EsquemaDB]]
    git: Callable[[], Awaitable[EstadoGit]]
    simbolos: Callable[[EstadoGit], Awaitable[frozenset[str]]] | None = None
    rutas_app: Callable[[], Awaitable[frozenset[str]]] | None = None
    claves: Callable[[], Awaitable[frozenset[str]]] | None = None
    modelos: Callable[[], Awaitable[frozenset[str]]] | None = None


@dataclass(frozen=True)
class Whitelist:
    tablas: frozenset[str] = frozenset()
    columnas: frozenset[str] = frozenset()
    pares: frozenset[str] = frozenset()
    archivos: frozenset[str] = frozenset()
    directorios: frozenset[str] = frozenset()
    simbolos: frozenset[str] = frozenset()
    sql: frozenset[str] = frozenset()
    rutas_app: frozenset[str] = frozenset()
    claves: frozenset[str] = frozenset()
    modelos: frozenset[str] = frozenset()
    no_disponibles: frozenset[str] = field(default_factory=frozenset)
    clave_cache: tuple[str, str, str] = ("", "", "")


async def derivar(fuentes: Fuentes) -> Whitelist:
    return Whitelist()


def fuentes_por_defecto(raiz_repo, rutas_app=None) -> Fuentes:
    raise NotImplementedError
