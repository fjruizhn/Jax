"""Cargador del catalogo cerrado de recursos que admiten tope (R-4, M-7).

La DECISION es de Fernando (2026-10-06) y por eso sus DATOS viven en
``policy/faro/catalogo-topes.json`` — ``policy/**`` es reservado y lo integran
Fernando o Hyde con ventana. Este modulo es solo el CARGADOR (hoja, stdlib):

  - lee ese archivo por ruta fija relativa a la raiz del repo;
  - valida su forma estricta (claves cerradas, listas de str ASCII sin
    duplicados, version exacta) y FALLA CERRADO si falta o es invalido: sin
    catalogo valido no se puede topear NADA;
  - expone el catalogo INMUTABLE en el proceso (``MappingProxyType`` y tuplas:
    mutarlo en runtime lanza, no se puede colar una clase por la puerta de atras).

Una lista negra por palabras nunca cierra; un catalogo cerrado si. Fuera del
catalogo no hay tope — en schema y en runtime, por igual.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import MappingProxyType

RUTA_DATOS = Path(__file__).resolve().parents[2] / "policy" / "faro" / "catalogo-topes.json"

VERSION_CATALOGO = 1
_CLAVES_CERRADAS = frozenset({"version", "decision", "clases"})
_RE_SUBID = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


class CatalogoTopesInvalido(RuntimeError):
    """El catalogo falta o no tiene la forma cerrada: fail-closed, nada se topea."""


def cargar_catalogo(ruta: Path = RUTA_DATOS) -> tuple[MappingProxyType, str]:
    """Valida y devuelve ``(clases, decision)`` inmutables. Falla cerrado."""
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CatalogoTopesInvalido(f"falta el catalogo de topes: {ruta}") from exc
    except (OSError, ValueError) as exc:
        raise CatalogoTopesInvalido(f"catalogo de topes ilegible: {type(exc).__name__}") from exc
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
        if not isinstance(subids, list) or not subids:
            raise CatalogoTopesInvalido(f"clase {clase}: lista no vacia")
        vistos: list[str] = []          # se conserva el orden del archivo: es el
        for subid in subids:            # dato de la decision, no un set cualquiera
            if not isinstance(subid, str) or not _RE_SUBID.fullmatch(subid):
                raise CatalogoTopesInvalido(f"clase {clase}: subid invalido {subid!r}")
            if subid in vistos:
                raise CatalogoTopesInvalido(f"clase {clase}: subid duplicado {subid!r}")
            vistos.append(subid)
        cerradas[clase] = tuple(vistos)
    return MappingProxyType(cerradas), decision


CLASES, DECISION_CATALOGO = cargar_catalogo()

# Inmutables expuestos: mutarlos en el proceso lanza (TypeError), no cambia nada.
CATALOGO_TOPES: MappingProxyType = CLASES
CLASES_RECURSO_CON_TOPE: tuple[str, ...] = tuple(sorted(CATALOGO_TOPES))


def es_recurso_de_catalogo(recurso: object) -> bool:
    """True solo si ``recurso`` es ``<clase>.<subid>`` con la clase declarada y el
    subid en el catalogo de ESA clase. Todo lo demas —infraestructura, agentes,
    variantes de idioma, nombres parecidos, subids de otra clase— queda fuera."""
    if not isinstance(recurso, str):
        return False
    clase, sep, subid = recurso.partition(".")
    if not sep or subid.count("."):
        return False
    return subid in CATALOGO_TOPES.get(clase, ())


def recursos_del_catalogo() -> tuple[str, ...]:
    """La lista plana ``clase.subid`` (para el espejo JSON y las pruebas de
    identidad entre datos, JSON, Python y runtime)."""
    return tuple(sorted(f"{clase}.{subid}" for clase, subids in CATALOGO_TOPES.items()
                        for subid in subids))
