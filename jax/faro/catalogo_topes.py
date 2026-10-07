"""Cargador del catalogo cerrado de recursos que admiten tope (R-4, M-7 + B-3).

La DECISION es de Fernando (2026-10-06) y sus DATOS viven en
``policy/faro/catalogo-topes.json`` — ``policy/**`` es reservado. Desde la ronda
5 (B-3), este modulo NO lee disco: recibe BYTES y valida. Quien carga el
catalogo es el SNAPSHOT desde el arbol ya verificado del pin (nunca el working
tree), y el runtime de topes lo recibe por parametro desde ese snapshot: sin
catalogo no se topea NADA.

Desde la ronda 6:
- los bytes son UTF-8 ESTRICTO y solo ``bytes`` de tipo exacto (UTF-16/32
  niegan en el decode; el BOM UTF-8 decodifica y lo niega ``json.loads``);
- las CLASES son piso de CODIGO (D-4): exactamente las cinco de la decision de
  Fernando, ni mas ni menos. Un catalogo ratificado con una sexta clase
  (``conexiones``...) se NIEGA: la ratificacion cubre ESTAS cinco. Los SUBID
  si son dato del pin — pero las seis palabras que D-4 excluye
  (conexiones/concurrencia/workers/hilos/procesos/agentes) tampoco valen como
  subid, en ninguna grafia (r7, MAJOR-2);
- el resultado es un ``CatalogoTopes`` SELLADO que SOLO emite el snapshot del
  pin (r7, MAJOR-1): la validacion es ``_cargar_catalogo_bytes`` (privada) y
  la construccion exige el testigo que el snapshot registro al importarse.
  El runtime exige ese tipo — con el OID del catalogo en el pin — y nunca un
  ``Mapping`` cualquiera.

Forma estricta y fail-closed: claves cerradas exactas (las de mas y las
duplicadas niegan — `object_pairs_hook`, igual que el YAML de las reglas),
version exacta, decision no vacia, clases no vacias con subids ASCII (NFC)
sin duplicados.
"""
from __future__ import annotations

import json
import re
import unicodedata
from types import MappingProxyType
from typing import Mapping

VERSION_CATALOGO = 1
_CLAVES_CERRADAS = frozenset({"version", "decision", "clases"})
_RE_SUBID = re.compile(r"[a-z][a-z0-9_]{0,63}\Z", re.ASCII)
_RE_OID = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z", re.ASCII)
# D-4 (r6): las clases topeables son las cinco que decidio Fernando — piso de
# CODIGO, no dato del pin. Cambiarlas es cambiar la decision, y eso pasa por
# codigo revisado, no por un JSON ratificado a espaldas del cargador.
CLASES_TOPEABLES = frozenset({"monto_dinero", "actos_externos", "frecuencia",
                              "duracion", "tokens_costo"})
# D-4 (r7, MAJOR-2): las seis palabras excluidas tampoco valen como SUBID —
# «nunca conexiones, concurrencia, workers, hilos, procesos ni agentes» no
# admite colarse por debajo de una clase admitida.
PALABRAS_PROHIBIDAS_D4 = frozenset({"conexiones", "concurrencia", "workers",
                                    "hilos", "procesos", "agentes"})
# r8: comparar la palabra exacta no alcanza (``conexiones_db``, ``num_hilos``,
# ``subprocesos``, ``multiagente`` cargaban). Se niega la palabra como SEGMENTO
# (separadores ``_``, ``-``, ``.``) y como SUBCADENA —de la palabra y de su
# raiz en singular— para las formas compuestas.
_RAICES_PROHIBIDAS_D4 = frozenset({"conexion", "concurrencia", "worker",
                                   "hilo", "proceso", "agente"})
_RE_SEPARADORES = re.compile(r"[_.\-]")


def _viola_d4(subid: str) -> bool:
    """True si el subid (tras NFC) es una palabra de D-4, la lleva como
    segmento o la contiene fundida en una forma compuesta."""
    normal = unicodedata.normalize("NFC", subid)
    if any(seg in PALABRAS_PROHIBIDAS_D4 for seg in _RE_SEPARADORES.split(normal)):
        return True
    return any(raiz in normal for raiz in _RAICES_PROHIBIDAS_D4)

# Emisor unico (r7, MAJOR-1): el snapshot registra AQUI su testigo al
# importarse; la lista se llena una sola vez. La clase compara identidad — el
# valor del testigo nunca sale de policy.rule_authority.snapshot.
_EMISOR: list = []


def _registrar_emisor(testigo: object) -> None:
    """Solo lo llama ``policy.rule_authority.snapshot`` con SU testigo, una
    vez. Con eso, CatalogoTopes solo puede construirlo el snapshot del pin."""
    if _EMISOR:
        raise CatalogoTopesInvalido("el emisor del catalogo ya esta registrado")
    _EMISOR.append(testigo)


class CatalogoTopes:
    """El catalogo SELLADO. Solo lo emite el snapshot del pin (r7, MAJOR-1):
    la construccion exige el testigo que el snapshot registro, y lleva el OID
    del catálogo dentro del pin. Inmutable; el runtime de topes exige ESTE
    tipo — la unica puerta al conteo con tope es el pin verificado."""

    __slots__ = ("_clases", "_oid_pin")

    def __init__(self, clases: Mapping[str, tuple], *, oid_pin: str,
                 _testigo: object = None) -> None:
        if not _EMISOR or _testigo is not _EMISOR[0]:
            raise CatalogoTopesInvalido(
                "CatalogoTopes solo lo emite el snapshot del pin (r7, MAJOR-1)")
        if not isinstance(oid_pin, str) or not _RE_OID.fullmatch(oid_pin):
            raise CatalogoTopesInvalido("CatalogoTopes exige el OID del catalogo en el pin")
        # copia defensiva + proxy: ni quien llama ni quien recibe mutan el interior
        object.__setattr__(self, "_clases", MappingProxyType(dict(clases)))
        object.__setattr__(self, "_oid_pin", oid_pin)

    @property
    def oid_pin(self) -> str:
        """El OID git del catalogo dentro del pin: de ahi salio este sellado."""
        return self._oid_pin

    def __setattr__(self, *_args) -> None:
        raise CatalogoTopesInvalido("CatalogoTopes es inmutable")

    def __delattr__(self, _name: str) -> None:
        raise CatalogoTopesInvalido("CatalogoTopes es inmutable")

    # Mapping de solo lectura sobre las clases cerradas (clase -> subids).
    def __getitem__(self, clase: str) -> tuple[str, ...]:
        return self._clases[clase]

    def __contains__(self, clase: object) -> bool:
        return clase in self._clases

    def __iter__(self):
        return iter(self._clases)

    def __len__(self) -> int:
        return len(self._clases)

    def get(self, clase: object, default=None):
        return self._clases.get(clase, default)

    def keys(self):
        return self._clases.keys()

    def items(self):
        return self._clases.items()

    def values(self):
        return self._clases.values()

    def __eq__(self, otro: object) -> bool:
        return isinstance(otro, CatalogoTopes) and self._clave() == otro._clave()

    def __hash__(self) -> int:
        return hash(self._clave())

    def _clave(self) -> tuple:
        return (tuple(sorted(self._clases.items())), self._oid_pin)

    def __repr__(self) -> str:
        return f"CatalogoTopes({dict(self._clases)!r}, oid_pin={self._oid_pin[:12]}…)"


class CatalogoTopesInvalido(RuntimeError):
    """El catalogo falta o no tiene la forma cerrada: fail-closed, nada se topea."""


def _sin_claves_duplicadas(pares: list[tuple[str, object]]) -> dict:
    resultado: dict = {}
    for clave, valor in pares:
        if clave in resultado:
            raise CatalogoTopesInvalido(f"clave duplicada en el catalogo: {clave!r}")
        resultado[clave] = valor
    return resultado


def _cargar_catalogo_bytes(crudo: bytes) -> dict[str, tuple[str, ...]]:
    """Valida los BYTES del catalogo y devuelve las clases CERRADAS (dict).
    PRIVADA (r7, MAJOR-1): quien sella es el snapshot, con estos datos y SU
    testigo — aqui no nace ningun CatalogoTopes.
    M-8: la clave duplicada no gana — niega. Falla cerrado ante cualquier vicio.
    r6/r7: solo ``bytes`` de tipo EXACTO en UTF-8 estricto, texto normalizado a
    NFC (UTF-16/32 niegan en el decode; el BOM UTF-8, json.loads) y las clases
    exactas de D-4, sin sus seis palabras como subid (ni como segmento ni fundidas en otra)."""
    if type(crudo) is not bytes:
        raise CatalogoTopesInvalido("el catalogo se recibe como bytes exactos, no como texto")
    try:
        texto = crudo.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CatalogoTopesInvalido("catalogo ilegible: no es UTF-8 estricto") from exc
    texto = unicodedata.normalize("NFC", texto)
    try:
        datos = json.loads(texto, object_pairs_hook=_sin_claves_duplicadas)
    except CatalogoTopesInvalido:
        raise
    except ValueError as exc:
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
            if _viola_d4(subid):            # r7 MAJOR-2 / r8: ni por debajo, ni compuesta
                raise CatalogoTopesInvalido(
                    f"clase {clase}: subid prohibido por D-4: {subid!r}")
            if subid in vistos:
                raise CatalogoTopesInvalido(f"clase {clase}: subid duplicado {subid!r}")
            vistos.append(subid)
        cerradas[clase] = tuple(vistos)
    # D-4 (r6): las cinco clases de la decision de Fernando son piso de codigo.
    # Una sexta clase ratificada (K2: «conexiones») niega igual; que falte una, tambien.
    if frozenset(cerradas) != CLASES_TOPEABLES:
        raise CatalogoTopesInvalido(
            f"clases del catalogo: exactamente {sorted(CLASES_TOPEABLES)} (D-4, "
            f"decision de Fernando); el pin trae {sorted(cerradas)}")
    return cerradas


def es_de_catalogo(recurso: object, catalogo: CatalogoTopes) -> bool:
    """True solo si ``recurso`` es ``<clase>.<subid>`` con la clase declarada y el
    subid en el catalogo DE ESA clase. Todo lo demas queda fuera. El catalogo
    tiene que ser el SELLADO (r6): un Mapping suelto niega."""
    if not isinstance(catalogo, CatalogoTopes):
        raise CatalogoTopesInvalido(
            "el catalogo se recibe SELLADO (CatalogoTopes), no un Mapping suelto")
    if not isinstance(recurso, str):
        return False
    clase, sep, subid = recurso.partition(".")
    if not sep or subid.count("."):
        return False
    return subid in catalogo.get(clase, ())


def recursos_del_catalogo(catalogo: CatalogoTopes) -> tuple[str, ...]:
    """La lista plana ``clase.subid`` (para el espejo JSON y las pruebas de
    identidad entre datos, JSON, Python y runtime)."""
    if not isinstance(catalogo, CatalogoTopes):
        raise CatalogoTopesInvalido(
            "el catalogo se recibe SELLADO (CatalogoTopes), no un Mapping suelto")
    return tuple(sorted(f"{clase}.{subid}" for clase, subids in catalogo.items()
                        for subid in subids))
