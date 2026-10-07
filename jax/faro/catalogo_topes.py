"""Cargador del catalogo cerrado de recursos que admiten tope (R-4, M-7 + B-3).

La DECISION es de Fernando (2026-10-06) y sus DATOS viven en
``policy/faro/catalogo-topes.json`` — ``policy/**`` es reservado. Desde la ronda
5 (B-3), este modulo NO lee disco: recibe BYTES y valida. Quien carga el
catalogo es el SNAPSHOT desde el arbol ya verificado del pin (nunca el working
tree), y el runtime de topes lo recibe por parametro desde ese snapshot: sin
catalogo no se topea NADA.

Forma estricta y fail-closed: claves cerradas exactas (las de mas y las
duplicadas niegan — `object_pairs_hook`, igual que el YAML de las reglas),
version exacta, decision no vacia, clases no vacias con subids ASCII sin
duplicados. El resultado es INMUTABLE (``MappingProxyType``): mutarlo en el
proceso lanza.
"""
from __future__ import annotations

import json
import re
from types import MappingProxyType
from typing import Mapping

VERSION_CATALOGO = 1
_CLAVES_CERRADAS = frozenset({"version", "decision", "clases"})
_RE_SUBID = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


class CatalogoTopesInvalido(RuntimeError):
    """El catalogo falta o no tiene la forma cerrada: fail-closed, nada se topea."""


def _sin_claves_duplicadas(pares: list[tuple[str, object]]) -> dict:
    resultado: dict = {}
    for clave, valor in pares:
        if clave in resultado:
            raise CatalogoTopesInvalido(f"clave duplicada en el catalogo: {clave!r}")
        resultado[clave] = valor
    return resultado


def cargar_catalogo_bytes(crudo: bytes) -> MappingProxyType:
    """Valida los BYTES del catalogo y devuelve las clases inmutables.
    M-8: la clave duplicada no gana — niega. Falla cerrado ante cualquier vicio."""
    if not isinstance(crudo, bytes):
        raise CatalogoTopesInvalido("el catalogo se recibe como bytes, no como texto")
    try:
        datos = json.loads(crudo, object_pairs_hook=_sin_claves_duplicadas)
    except CatalogoTopesInvalido:
        raise
    except (UnicodeDecodeError, ValueError) as exc:
        raise CatalogoTopesInvalido(f"catalogo ilegible: {type(exc).__name__}") from exc
    if not isinstance(datos, dict) or frozenset(datos) != _CLAVES_CERRADAS:
        raise CatalogoTopesInvalido("el catalogo exige exactamente version, decision y clases")
    version = datos["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != VERSION_CATALOGO:
        raise CatalogoTopesInvalido(f"version del catalogo debe ser exactamente {VERSION_CATALOGO}")
    decision = datos["decision"]
    if not isinstance(decision, str) or not decision.strip():
        raise CatalogoTopesInvalido("el catalogo exige la decision que lo respalda")
    clases = datos["clases"]
    if not isinstance(clases, dict) or not clases:
        raise CatalogoTopesInvalido("clases: objeto no vacio")
    cerradas: dict[str, tuple[str, ...]] = {}
    for clase, subids in clases.items():
        if not isinstance(clase, str) or not _RE_SUBID.fullmatch(clase):
            raise CatalogoTopesInvalido(f"clase invalida: {clase!r}")
        if not isinstance(subids, list) or not subids:          # N9: lista vacia niega
            raise CatalogoTopesInvalido(f"clase {clase}: lista no vacia")
        vistos: list[str] = []          # se conserva el orden de la decision
        for subid in subids:
            if not isinstance(subid, str) or not _RE_SUBID.fullmatch(subid):
                raise CatalogoTopesInvalido(f"clase {clase}: subid invalido {subid!r}")
            if subid in vistos:
                raise CatalogoTopesInvalido(f"clase {clase}: subid duplicado {subid!r}")
            vistos.append(subid)
        cerradas[clase] = tuple(vistos)
    return MappingProxyType(cerradas)


def es_de_catalogo(recurso: object, catalogo: Mapping) -> bool:
    """True solo si ``recurso`` es ``<clase>.<subid>`` con la clase declarada y el
    subid en el catalogo DE ESA clase. Todo lo demas queda fuera."""
    if not isinstance(catalogo, Mapping):
        raise CatalogoTopesInvalido("el catalogo se recibe cargado (MappingProxyType)")
    if not isinstance(recurso, str):
        return False
    clase, sep, subid = recurso.partition(".")
    if not sep or subid.count("."):
        return False
    return subid in catalogo.get(clase, ())


def recursos_del_catalogo(catalogo: Mapping) -> tuple[str, ...]:
    """La lista plana ``clase.subid`` (para el espejo JSON y las pruebas de
    identidad entre datos, JSON, Python y runtime)."""
    return tuple(sorted(f"{clase}.{subid}" for clase, subids in catalogo.items()
                        for subid in subids))
