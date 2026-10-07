"""Proveedores confiables del Rule Authority Kernel (F1.1 §9, paso 7 r2).

SOLO interfaces: Protocolos, tipos de valor y el guard. Los dobles viven en
``tests/policy/proveedores_dobles.py`` — un kernel no puede armarse con dobles
porque aqui no hay ninguno. Sin wiring operativo.

Contratos que este modulo fija (decision de Hyde, r2):

  - ``VersionMonotonica``: entero >= 0 validado. El ORDEN lo garantiza quien
    EMITE versiones (el proveedor), no el tipo: dos instancias con el mismo
    numero existen y comparan iguales; la monotonia es obligacion del emisor,
    declarada en cada Protocolo. (Sin promesas vagas de «ABA inexpresable».)
  - **Leases con valor DENTRO**: ``lease_compartido()``/``lease_exclusivo()``
    entregan una ``VistaLease`` — valor y version DEL MISMO INSTANTE — y quien
    implementa el Protocolo garantiza exclusion real escritor-lector. El
    compartido entrega valor+version coherentes; el exclusivo excluye a todos.
  - **Reloj**: UTC consciente; monotonico por proceso — el retroceso se detecta
    contra la ULTIMA LECTURA, no solo contra la emision, y ``marcar_emision``
    no puede bajar el piso.
  - **Clasificacion**: ``ClaseCapability`` es un enum cerrado; ausente,
    dudoso o desconocido es error tipado y el kernel lo OBLIGA (trata como
    OBLIGATING, nunca rebaja); una clase nunca baja respecto de la version
    anterior del contrato de la misma capability.
  - **Checkpoint**: monotónico con compare-and-swap ATOMICA (anterior exacto),
    idempotencia SOLO con el mismo anterior y el mismo head, y relectura que
    confirma que el log durable contiene el head exacto (§10 paso 15, §11).
    La implementacion real persiste y hace fsync; la de pruebas no.
  - **Pin**: la procedencia la FIJA el proveedor sellado — nunca un nombre
    movil (``refs/…``) ni vacio.

Inmutabilidad: los tipos de valor usan ``__slots__`` sin setters;
``object.__setattr__`` NO es una frontera de seguridad (el que puede ejecutar
codigo en el proceso ya perdio) — es orden, no aislamiento.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from contextlib import AbstractContextManager
from typing import NamedTuple, Protocol, runtime_checkable

from .errors import (
    ClasificacionDesconocida,
    ProveedorInvalido,
    RelojInvalido,
    RuleAuthorityError,
    StopDesconocido,
)
from .snapshot import TrustedPolicyPin

# §9: orden global de adquisicion. Invertirlo invierte locks entre evaluacion,
# consumo y mutaciones.
ORDEN_ADQUISICION = ("pin_activo", "clasificacion", "stop", "permit_db",
                     "authority_ledger_head", "audit_head")

# Interfaz con plazos: FUERA DE ALCANCE en este paso. Los Protocolos son
# sincronos y sin timeout propio; el plazo lo pone el llamador del kernel
# (envolviendo la llamada), nunca la interfaz. Declarado, no diferido.


class VersionMonotonica:
    """Entero >= 0 validado. La monotonia la garantiza el EMISOR (ver Protocolos);
    el tipo solo valida forma y da orden total por numero."""

    __slots__ = ("_numero",)

    def __init__(self, numero: int) -> None:
        if isinstance(numero, bool) or not isinstance(numero, int) or numero < 0:
            raise ProveedorInvalido(f"version invalida: {numero!r}")
        object.__setattr__(self, "_numero", numero)

    def __setattr__(self, *_args) -> None:
        raise RuleAuthorityError("VersionMonotonica es inmutable (slots, sin setters)")

    @property
    def numero(self) -> int:
        return self._numero

    def es_posterior_a(self, otra: "VersionMonotonica") -> bool:
        return self._numero > otra._numero

    def __eq__(self, otra: object) -> bool:
        return isinstance(otra, VersionMonotonica) and self._numero == otra._numero

    def __lt__(self, otra: "VersionMonotonica") -> bool:
        return self._numero < otra._numero

    def __hash__(self) -> int:
        return hash(("VersionMonotonica", self._numero))


class ClaseCapability(str, Enum):
    """Enum cerrado. Fuera de estos dos valores no hay clase; duda = OBLIGA."""
    REVERSIBLE = "REVERSIBLE"
    OBLIGATING = "OBLIGATING"


class ContratoCapability:
    """Contrato inmutable de una capability; ``clase`` SOLO del enum cerrado."""

    __slots__ = ("identidad", "version", "clase", "unidad", "moneda")

    def __init__(self, *, identidad: str, version: str, clase: ClaseCapability,
                 unidad: str | None, moneda: str | None) -> None:
        if not isinstance(identidad, str) or not identidad.strip():
            raise ProveedorInvalido("ContratoCapability: identidad no vacia")
        if not isinstance(version, str) or not version.strip():
            raise ProveedorInvalido("ContratoCapability: version no vacia")
        if not isinstance(clase, ClaseCapability):
            raise ProveedorInvalido("ContratoCapability: clase fuera del enum cerrado")
        for campo, valor in (("unidad", unidad), ("moneda", moneda)):
            if valor is not None and (not isinstance(valor, str) or not valor.strip()):
                raise ProveedorInvalido(f"ContratoCapability: {campo} invalida")
        object.__setattr__(self, "identidad", identidad)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "clase", clase)
        object.__setattr__(self, "unidad", unidad)
        object.__setattr__(self, "moneda", moneda)

    def __setattr__(self, *_args) -> None:
        raise RuleAuthorityError("ContratoCapability es inmutable")

    def _clave(self) -> tuple:
        return (self.identidad, self.version, self.clase, self.unidad, self.moneda)

    def __eq__(self, otro: object) -> bool:
        return isinstance(otro, ContratoCapability) and self._clave() == otro._clave()

    def __hash__(self) -> int:
        return hash(self._clave())


class EstadoStop(NamedTuple):
    activo: bool
    version: VersionMonotonica
    huella: str


class VistaLease(NamedTuple):
    """Valor y version DEL MISMO instante: lo que un lease entrega."""
    valor: object
    version: VersionMonotonica


@runtime_checkable
class _ProveedorConLeases(Protocol):
    """Contrato de leases: exclusion real escritor-lector; la vista entrega
    valor+version coherentes; el exclusivo excluye a lectores y a otros
    escritores. El EMISOR garantiza versiones estrictamente crecientes."""

    def lease_compartido(self) -> AbstractContextManager[VistaLease]: ...

    def lease_exclusivo(self) -> AbstractContextManager[VistaLease]: ...


@runtime_checkable
class ProveedorPinActivo(_ProveedorConLeases, Protocol):
    """El pin activo con SU procedencia sellada — nunca nombre movil ni vacia."""

    def pin_activo(self) -> tuple[TrustedPolicyPin, str]: ...


@runtime_checkable
class ProveedorStop(_ProveedorConLeases, Protocol):
    """La fuente unica de STOP. Ilegible o sin configurar: StopDesconocido
    (tipado; el kernel lo traduce en DENY — fail-closed)."""

    def estado(self) -> EstadoStop: ...


@runtime_checkable
class RelojConfiable(Protocol):
    """UTC consciente, monotonico por proceso: el retroceso se detecta contra
    la ultima lectura; marcar_emision nunca baja el piso."""

    def ahora(self) -> datetime: ...

    def marcar_emision(self, momento: datetime) -> None: ...


@runtime_checkable
class ProveedorClasificacion(_ProveedorConLeases, Protocol):
    """Clase, unidad/moneda y forma de limites por capability. Ausente o
    desconocido: ClasificacionDesconocida (el kernel OBLIGA, no rebaja). Una
    clase nunca baja respecto de la version anterior del contrato."""

    def contrato_de(self, capability: str) -> ContratoCapability: ...


@runtime_checkable
class AlmacenCheckpoints(Protocol):
    """Checkpoint externo monotónico y OBLIGATORIO. CAS atomica con anterior
    exacto; idempotente SOLO con el mismo anterior Y el mismo head; confirmar
    relee y exige que el log durable contenga el head exacto (§10.15, §11).
    La implementacion REAL persiste y hace fsync antes de responder; la de
    pruebas vive en memoria y no sustituye a esa."""

    def head_actual(self) -> str: ...

    def publicar(self, head: str, *, anterior: str) -> None: ...

    def confirmar(self, head: str) -> bool: ...


# ------------------------------------------------- el guard de emision/consumo

def _vista_valida(vista: object) -> bool:
    return isinstance(vista, VistaLease) and isinstance(vista.version, VersionMonotonica)


def _exigir_lease_con_vista(nombre: str, proveedor: object) -> None:
    try:
        with proveedor.lease_compartido() as vista:          # type: ignore[attr-defined]
            if not _vista_valida(vista):
                raise ProveedorInvalido(f"{nombre}: el lease no entrega VistaLease valida")
    except ProveedorInvalido:
        raise
    except StopDesconocido:
        raise                                            # estado fail-closed legitimo: el kernel NEGARA
    except Exception as exc:
        raise ProveedorInvalido(f"{nombre}: lease_compartido no sirve: {type(exc).__name__}") from exc


def exigir_contrato_de_emision(*, pin: object, checkpoint: object, stop: object,
                               reloj: object, clasificacion: object) -> None:
    """Lo que el kernel llama ANTES de evaluar o consumir (paso 7 lo provee).

    - todos los proveedores presentes y ``isinstance`` de SU Protocolo;
    - una llamada de prueba valida los TIPOS devueltos (un Mock o un None se
      niegan aqui, no en produccion);
    - checkpoint OBLIGATORIO y ya publicado: vacio, niega (§11).
    """
    esperados = (("pin_activo", ProveedorPinActivo), ("stop", ProveedorStop),
                 ("clasificacion", ProveedorClasificacion),
                 ("checkpoint", AlmacenCheckpoints), ("reloj", RelojConfiable))
    disponibles = {"pin_activo": pin, "stop": stop, "clasificacion": clasificacion,
                   "checkpoint": checkpoint, "reloj": reloj}
    for nombre, protocolo in esperados:
        proveedor = disponibles[nombre]
        if proveedor is None:
            raise ProveedorInvalido(f"falta el proveedor {nombre}")
        if not isinstance(proveedor, protocolo):
            raise ProveedorInvalido(f"{nombre}: no implementa {protocolo.__name__}")

    _exigir_lease_con_vista("pin_activo", pin)
    _exigir_lease_con_vista("stop", stop)
    _exigir_lease_con_vista("clasificacion", clasificacion)

    try:
        valor_pin, procedencia = pin.pin_activo()           # type: ignore[attr-defined]
    except Exception as exc:
        raise ProveedorInvalido(f"pin_activo: no sirve: {type(exc).__name__}") from exc
    if not isinstance(valor_pin, TrustedPolicyPin) or not isinstance(procedencia, str) \
            or not procedencia.strip() or procedencia.startswith("refs/"):
        raise ProveedorInvalido("pin_activo: pin o procedencia invalida (sellada, nunca refs/)")

    try:
        estado = stop.estado()                              # type: ignore[attr-defined]
    except StopDesconocido:
        pass                                                # estado fail-closed valido: kernel NEGARA
    else:
        if not isinstance(estado, EstadoStop) or isinstance(estado.activo, bool) is False \
                or not isinstance(estado.version, VersionMonotonica) \
                or not isinstance(estado.huella, str):
            raise ProveedorInvalido("stop: EstadoStop mal formado (activo bool, version, huella)")

    if not isinstance(reloj.ahora(), datetime):             # type: ignore[attr-defined]
        raise ProveedorInvalido("reloj: ahora() no entrega datetime")
    if reloj.ahora().tzinfo is None:                        # type: ignore[attr-defined]
        raise ProveedorInvalido("reloj: hora sin timezone")
    if not callable(getattr(reloj, "marcar_emision", None)):
        raise ProveedorInvalido("reloj: falta marcar_emision")

    try:
        head = checkpoint.head_actual()                     # type: ignore[attr-defined]
    except Exception as exc:
        raise ProveedorInvalido(f"checkpoint: no legible: {type(exc).__name__}") from exc
    if not isinstance(head, str) or not head:
        raise ProveedorInvalido("checkpoint: nunca publicado — sin head anclado no hay emision (§11)")
    for metodo in ("publicar", "confirmar"):
        if not callable(getattr(checkpoint, metodo, None)):
            raise ProveedorInvalido(f"checkpoint: falta {metodo}")


__all__ = [
    "ORDEN_ADQUISICION", "VersionMonotonica", "ClaseCapability", "ContratoCapability",
    "EstadoStop", "VistaLease", "ProveedorPinActivo", "ProveedorStop", "RelojConfiable",
    "ProveedorClasificacion", "AlmacenCheckpoints", "exigir_contrato_de_emision",
    "ClasificacionDesconocida", "ProveedorInvalido", "RelojInvalido", "RuleAuthorityError",
    "StopDesconocido",
]
