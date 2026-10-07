"""Proveedores confiables del Rule Authority Kernel (F1.1 §9, paso 7 r3).

SOLO interfaces: Protocolos, tipos de valor y el guard. Sin wiring operativo.
Los dobles viven en ``tests/policy/proveedores_dobles.py`` y este modulo no
importa nada de ``tests/``. Ojo con lo que eso NO promete: nada aqui impide que
un llamador escriba (o importe) un doble y lo pase al guard; el guard solo
verifica CONTRATOS (tipos, procedencia, STOP, reloj, checkpoint). Que un doble
no llegue a produccion es trabajo del cableado, fuera de alcance de este paso.

Contratos que este modulo fija:

  - ``VersionMonotonica``: entero >= 0 validado. El ORDEN lo garantiza quien
    EMITE versiones (el proveedor), no el tipo: dos instancias con el mismo
    numero existen y comparan iguales; la monotonia es obligacion del emisor.
  - **Toda lectura es DENTRO de un lease** (r3, MAJOR-4): los Protocolos de pin,
    STOP y clasificacion NO tienen lecturas sueltas. ``lease_compartido()`` y
    ``lease_exclusivo()`` entregan una ``VistaLease`` — valor y version del
    MISMO instante — con la MISMA forma por las dos vias. El valor es del tipo
    que fija cada Protocolo (``PinActivo``, ``EstadoStop``,
    ``CatalogoClasificacion``) y el guard lo valida por tipo.
  - **Reloj**: UTC consciente (``tzinfo`` presente y ``utcoffset() == 0``);
    monotonico por proceso — el retroceso se detecta contra la ULTIMA LECTURA
    y ``marcar_emision`` no puede bajar el piso.
  - **Clasificacion** (§13): capability desconocida, sin clasificacion o
    rebajada => el kernel NIEGA. El guard sondea una capability centinela y
    exige ``ClasificacionDesconocida``; cualquier otra respuesta niega. Una
    clase nunca baja respecto de la version anterior del contrato.
  - **Checkpoint**: monotonico, CAS atomica con anterior exacto, idempotencia
    SOLO con el mismo anterior y el mismo head. ``confirmar(head)`` es «el log
    durable contiene el head exacto» (§11) — no «es el head actual»: otra
    operacion pudo agregar un descendiente. La implementacion real persiste y
    hace fsync; la de pruebas no.
  - **Pin**: la procedencia tiene forma CERRADA (``refs/heads/<rama>`` o
    ``refs/tags/<tag>``, anclada, sensible a mayusculas, sin espacios) y debe
    ser igual a la que declara el propio ``TrustedPolicyPin``. El pin sigue
    siendo un SHA: la procedencia es etiqueta de auditoria, no raiz de confianza.
  - **STOP ilegible**: el guard NIEGA (no delega ni lo da por buen estado).

Inmutabilidad: los tipos de valor usan ``__slots__`` sin setters ni deleters;
``object.__setattr__`` NO es una frontera de seguridad (el que puede ejecutar
codigo en el proceso ya perdio) — es orden, no aislamiento.
"""
from __future__ import annotations

import re
import secrets
from datetime import datetime, timedelta
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

    def __delattr__(self, _nombre: str) -> None:
        raise RuleAuthorityError("VersionMonotonica es inmutable (sin deleters)")

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


class FormaLimites(str, Enum):
    """Forma de los limites que acepta la capability (§10.7): el permiso la
    guarda y el consumo revalida que no cambio. Enum cerrado."""
    NINGUNA = "NINGUNA"
    CANTIDAD = "CANTIDAD"
    MONTO = "MONTO"
    CANTIDAD_Y_MONTO = "CANTIDAD_Y_MONTO"


class ContratoCapability:
    """Contrato inmutable de una capability; ``clase`` y ``forma_limites``
    SOLO de sus enums cerrados."""

    __slots__ = ("identidad", "version", "clase", "unidad", "moneda", "forma_limites")

    def __init__(self, *, identidad: str, version: str, clase: ClaseCapability,
                 unidad: str | None, moneda: str | None,
                 forma_limites: FormaLimites) -> None:
        if not isinstance(identidad, str) or not identidad.strip():
            raise ProveedorInvalido("ContratoCapability: identidad no vacia")
        if not isinstance(version, str) or not version.strip():
            raise ProveedorInvalido("ContratoCapability: version no vacia")
        if not isinstance(clase, ClaseCapability):
            raise ProveedorInvalido("ContratoCapability: clase fuera del enum cerrado")
        if not isinstance(forma_limites, FormaLimites):
            raise ProveedorInvalido("ContratoCapability: forma_limites fuera del enum cerrado")
        for campo, valor in (("unidad", unidad), ("moneda", moneda)):
            if valor is not None and (not isinstance(valor, str) or not valor.strip()):
                raise ProveedorInvalido(f"ContratoCapability: {campo} invalida")
        object.__setattr__(self, "identidad", identidad)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "clase", clase)
        object.__setattr__(self, "unidad", unidad)
        object.__setattr__(self, "moneda", moneda)
        object.__setattr__(self, "forma_limites", forma_limites)

    def __setattr__(self, *_args) -> None:
        raise RuleAuthorityError("ContratoCapability es inmutable")

    def __delattr__(self, _nombre: str) -> None:
        raise RuleAuthorityError("ContratoCapability es inmutable (sin deleters)")

    def _clave(self) -> tuple:
        return (self.identidad, self.version, self.clase, self.unidad, self.moneda,
                self.forma_limites)

    def __eq__(self, otro: object) -> bool:
        return isinstance(otro, ContratoCapability) and self._clave() == otro._clave()

    def __hash__(self) -> int:
        return hash(self._clave())


class PinActivo(NamedTuple):
    """Valor del lease del pin: el pin y la procedencia SELLADA por el proveedor
    (el guard exige que coincida con la que declara el propio pin)."""
    pin: TrustedPolicyPin
    procedencia: str


class EstadoStop(NamedTuple):
    """Valor del lease de STOP. La version del estado es la de la ``VistaLease``
    (una sola fuente: nunca dos versiones para un mismo estado)."""
    activo: bool
    huella: str


@runtime_checkable
class CatalogoClasificacion(Protocol):
    """Valor del lease de clasificacion. ``contrato_de`` solo existe DENTRO del
    lease. Capability ausente o sin clasificar: ``ClasificacionDesconocida``."""

    def contrato_de(self, capability: str) -> ContratoCapability: ...


class VistaLease(NamedTuple):
    """Valor y version DEL MISMO instante: lo que un lease entrega."""
    valor: object
    version: VersionMonotonica


@runtime_checkable
class _ProveedorConLeases(Protocol):
    """Contrato de leases: exclusion real escritor-lector y entre escritores; la
    vista entrega valor+version coherentes y con la MISMA forma por las dos
    vias; el EMISOR garantiza versiones estrictamente crecientes. Un compartido
    dentro del exclusivo del mismo hilo (y al reves) es un error, no un bloqueo."""

    def lease_compartido(self) -> AbstractContextManager[VistaLease]: ...

    def lease_exclusivo(self) -> AbstractContextManager[VistaLease]: ...


@runtime_checkable
class ProveedorPinActivo(_ProveedorConLeases, Protocol):
    """Valor del lease: ``PinActivo``. Sin lectura fuera del lease."""


@runtime_checkable
class ProveedorStop(_ProveedorConLeases, Protocol):
    """La fuente unica de STOP; valor del lease: ``EstadoStop``. Ilegible o sin
    configurar: el lease lanza ``StopDesconocido`` y el guard NIEGA."""


@runtime_checkable
class ProveedorClasificacion(_ProveedorConLeases, Protocol):
    """Valor del lease: ``CatalogoClasificacion``. Ausente o desconocido:
    ClasificacionDesconocida (el kernel NIEGA, §13). Una clase nunca baja
    respecto de la version anterior del contrato."""


@runtime_checkable
class RelojConfiable(Protocol):
    """UTC consciente, monotonico por proceso: el retroceso se detecta contra
    la ultima lectura; marcar_emision nunca baja el piso."""

    def ahora(self) -> datetime: ...

    def marcar_emision(self, momento: datetime) -> None: ...


@runtime_checkable
class AlmacenCheckpoints(Protocol):
    """Checkpoint externo monotonico y OBLIGATORIO. CAS atomica con anterior
    exacto; idempotente SOLO con el mismo anterior Y el mismo head;
    ``confirmar(head)`` relee y responde True solo si el log durable CONTIENE
    el head exacto (§10.15, §11). La implementacion REAL persiste y hace fsync
    antes de responder; la de pruebas vive en memoria y no sustituye a esa."""

    def head_actual(self) -> str: ...

    def publicar(self, head: str, *, anterior: str) -> None: ...

    def confirmar(self, head: str) -> bool: ...


# ------------------------------------------------- el guard de emision/consumo

# Prefijo reservado de las capabilities centinela. Cada sondeo usa un nombre
# ALEATORIO (prefijo + token): un catalogo no puede tratar «la centinela» como
# caso especial y responder REVERSIBLE a todo lo demas. El prefijo no es API.
_PREFIJO_CENTINELA = "__centinela_sin_clasificar__"


def _nueva_centinela() -> str:
    return f"{_PREFIJO_CENTINELA}{secrets.token_hex(16)}"


_RE_SEGMENTO = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9_-])?"        # no termina en «.»
_RE_PROCEDENCIA = re.compile(rf"refs/(?:heads|tags)/{_RE_SEGMENTO}(?:/{_RE_SEGMENTO})*", re.ASCII)


def _procedencia_cerrada(procedencia: object) -> bool:
    """Forma cerrada (git check-ref-format, subconjunto): sin «..», sin segmento
    que termine en «.lock» ni en «.», anclada y sensible a mayusculas."""
    if type(procedencia) is not str or ".." in procedencia:
        return False
    if any(seg.endswith(".lock") for seg in procedencia.split("/")):
        return False
    return _RE_PROCEDENCIA.fullmatch(procedencia) is not None


def _dentro_de_lease(nombre: str, proveedor: object, validar) -> None:
    """Abre el lease COMPARTIDO y valida la vista Y su valor sin salir de el."""
    try:
        with proveedor.lease_compartido() as vista:          # type: ignore[attr-defined]
            if not isinstance(vista, VistaLease) \
                    or not isinstance(vista.version, VersionMonotonica):
                raise ProveedorInvalido(f"{nombre}: el lease no entrega una VistaLease valida")
            validar(vista.valor)
    except ProveedorInvalido:
        raise
    except Exception as exc:
        raise ProveedorInvalido(f"{nombre}: lease/valor no sirve: {type(exc).__name__}") from exc


def _validar_pin(valor: object) -> None:
    if type(valor) is not PinActivo:
        raise ProveedorInvalido("pin_activo: el valor del lease no es PinActivo")
    if type(valor.pin) is not TrustedPolicyPin:
        raise ProveedorInvalido("pin_activo: el pin debe ser exactamente TrustedPolicyPin")
    if not _procedencia_cerrada(valor.procedencia):
        raise ProveedorInvalido("pin_activo: procedencia fuera de la forma cerrada "
                                "(refs/heads/<rama> o refs/tags/<tag>)")
    if valor.procedencia != valor.pin.procedencia:
        raise ProveedorInvalido("pin_activo: la procedencia sellada difiere de la del pin")


def _validar_stop(valor: object) -> None:
    if type(valor) is not EstadoStop:
        raise ProveedorInvalido("stop: el valor del lease no es EstadoStop")
    if type(valor.activo) is not bool or type(valor.huella) is not str or not valor.huella:
        raise ProveedorInvalido("stop: EstadoStop mal formado (activo bool, huella no vacia)")


def _validar_clasificacion(valor: object) -> None:
    if not isinstance(valor, CatalogoClasificacion):
        raise ProveedorInvalido("clasificacion: el valor del lease no implementa CatalogoClasificacion")
    _exigir_centinela_desconocida(valor)


def _exigir_centinela_desconocida(catalogo: CatalogoClasificacion) -> None:
    """§13: una capability desconocida NIEGA. Responder un contrato (REVERSIBLE
    incluido) o None, o fallar con otra cosa, es un catalogo inservible."""
    try:
        respuesta = catalogo.contrato_de(_nueva_centinela())
    except ClasificacionDesconocida:
        return
    except Exception as exc:
        raise ProveedorInvalido(
            f"clasificacion: la centinela fallo con {type(exc).__name__}, no con "
            "ClasificacionDesconocida") from exc
    raise ProveedorInvalido(
        f"clasificacion: respondio {respuesta!r} a una capability desconocida (§13: debe negar)")


def _validar_reloj(reloj: object) -> None:
    try:
        momento = reloj.ahora()                              # type: ignore[attr-defined]
    except Exception as exc:
        raise ProveedorInvalido(f"reloj: ahora() no sirve: {type(exc).__name__}") from exc
    if not isinstance(momento, datetime):
        raise ProveedorInvalido("reloj: ahora() no entrega datetime")
    if momento.tzinfo is None or momento.utcoffset() != timedelta(0):
        raise ProveedorInvalido("reloj: la hora debe ser UTC consciente (utcoffset == 0)")


def _validar_checkpoint(checkpoint: object) -> None:
    try:
        head = checkpoint.head_actual()                      # type: ignore[attr-defined]
    except Exception as exc:
        raise ProveedorInvalido(f"checkpoint: no legible: {type(exc).__name__}") from exc
    if not isinstance(head, str) or not head:
        raise ProveedorInvalido("checkpoint: nunca publicado — sin head anclado no hay emision (§11)")
    try:
        contiene = checkpoint.confirmar(head)                # type: ignore[attr-defined]
    except Exception as exc:
        raise ProveedorInvalido(f"checkpoint: confirmar no sirve: {type(exc).__name__}") from exc
    if contiene is not True:
        raise ProveedorInvalido("checkpoint: el log no contiene el head exacto (§11)")


def exigir_contrato_de_emision(*, pin: object, checkpoint: object, stop: object,
                               reloj: object, clasificacion: object) -> None:
    """Lo que el kernel llama ANTES de evaluar o consumir.

    - todos los proveedores presentes y ``isinstance`` de SU Protocolo;
    - el valor de cada lease se valida POR TIPO y DENTRO del lease; un Mock, un
      None o un STOP ilegible se niegan aqui;
    - pin: ``type(pin) is TrustedPolicyPin``, procedencia de forma cerrada e
      igual a la del pin; reloj UTC consciente; clasificacion que niega la
      capability centinela; checkpoint publicado y que ``confirmar`` su head.
    """
    esperados = (("pin_activo", pin, ProveedorPinActivo), ("stop", stop, ProveedorStop),
                 ("clasificacion", clasificacion, ProveedorClasificacion),
                 ("checkpoint", checkpoint, AlmacenCheckpoints), ("reloj", reloj, RelojConfiable))
    for nombre, proveedor, protocolo in esperados:
        if proveedor is None:
            raise ProveedorInvalido(f"falta el proveedor {nombre}")
        if not isinstance(proveedor, protocolo):
            raise ProveedorInvalido(f"{nombre}: no implementa {protocolo.__name__}")

    _dentro_de_lease("pin_activo", pin, _validar_pin)
    _dentro_de_lease("stop", stop, _validar_stop)
    _dentro_de_lease("clasificacion", clasificacion, _validar_clasificacion)
    _validar_reloj(reloj)
    _validar_checkpoint(checkpoint)


__all__ = [
    "ORDEN_ADQUISICION", "VersionMonotonica", "ClaseCapability",
    "FormaLimites", "ContratoCapability", "PinActivo", "EstadoStop", "CatalogoClasificacion",
    "VistaLease", "ProveedorPinActivo", "ProveedorStop", "RelojConfiable",
    "ProveedorClasificacion", "AlmacenCheckpoints", "exigir_contrato_de_emision",
    "ClasificacionDesconocida", "ProveedorInvalido", "RelojInvalido", "RuleAuthorityError",
    "StopDesconocido",
]
