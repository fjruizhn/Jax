"""Proveedores confiables del Rule Authority Kernel (F1.1 §9, paso 7).

Interfaces + dobles deterministas. SIN wiring operativo: aqui no se instala
ningun proveedor real — eso llega con el despliegue, con sus propias auditorias.

El contrato que este modulo fija (y que el kernel de evaluacion DEBE exigir via
``exigir_contrato_de_emision`` antes de evaluar o consumir):

  - **Versiones monotonicas SIN ABA** (``VersionMonotonica``): un valor repetido
    o retrocedido no se puede ni construir. Un proveedor que solo permita
    releer un valor NO satisface el contrato y no puede emitir ni consumir.
  - **Leases** (§9): evaluacion y consumo toman leases COMPARTIDOS de pin,
    clasificacion y STOP y los sostienen hasta despues del commit de DB; quien
    cambia el pin, un contrato de capability o activa STOP necesita el lease
    EXCLUSIVO. El lease entrega una version estable mientras vive.
  - **Orden global de adquisicion** (§9): ``ORDEN_ADQUISICION``; todos los
    dobles y adapters del kernel siguen ese orden para no invertir locks.
  - **Fallo cerrado tipado**: STOP desconocido, reloj ingenuo o retrocedido,
    clasificacion ausente y checkpoint retrocedido son errores tipados que el
    kernel traduce en DENY — ninguno se convierte en permiso.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, NamedTuple, Protocol

from .errors import (
    CheckpointInvalido,
    ClasificacionDesconocida,
    ProveedorInvalido,
    RelojInvalido,
    RelojRetrocedio,
    RuleAuthorityError,
    StopDesconocido,
    VersionRetrocedio,
)
from .snapshot import TrustedPolicyPin

# §9: orden global de adquisicion. Invertirlo invierte locks entre evaluacion,
# consumo y mutaciones.
ORDEN_ADQUISICION = ("pin_activo", "clasificacion", "stop", "permit_db",
                     "authority_ledger_head", "audit_head")


# ------------------------------------------------- versiones monotonicas sin ABA

class VersionMonotonica:
    """Version estrictamente creciente. Construir una igual o anterior a la
    previa lanza: el ABA no se puede expresar."""

    __slots__ = ("_numero",)

    def __init__(self, numero: int, *, _previa: "VersionMonotonica | None" = None) -> None:
        if isinstance(numero, bool) or not isinstance(numero, int) or numero < 0:
            raise ProveedorInvalido(f"version invalida: {numero!r}")
        if _previa is not None and numero <= _previa._numero:
            raise VersionRetrocedio(
                f"version {numero} no es posterior a {_previa._numero}: sin ABA, sin repeticion")
        object.__setattr__(self, "_numero", numero)

    def __setattr__(self, *_args) -> None:
        raise RuleAuthorityError("VersionMonotonica es inmutable")

    @property
    def numero(self) -> int:
        return self._numero

    def es_posterior_a(self, otra: "VersionMonotonica") -> bool:
        return self._numero > otra._numero

    def __eq__(self, otra: object) -> bool:
        return isinstance(otra, VersionMonotonica) and self._numero == otra._numero

    def __hash__(self) -> int:
        return hash(("VersionMonotonica", self._numero))


class RelojDeVersiones:
    """Fabrica determinista de versiones estrictamente crecientes (para dobles)."""

    def __init__(self) -> None:
        self._ultima: VersionMonotonica | None = None

    def siguiente(self) -> VersionMonotonica:
        nueva = VersionMonotonica(0 if self._ultima is None else self._ultima._numero + 1,
                                  _previa=self._ultima)
        self._ultima = nueva
        return nueva


# ------------------------------------------------------------------- leases

class _Lease:
    """Un lease con una version estable mientras vive. El exclusivo exige que no
    haya compartidos vivos (los dobles lo verifican; los providers reales
    coordinan con sus writers)."""

    __slots__ = ("_version", "_exclusivo", "_liberado")

    def __init__(self, version: VersionMonotonica, *, exclusivo: bool) -> None:
        object.__setattr__(self, "_version", version)
        object.__setattr__(self, "_exclusivo", exclusivo)
        object.__setattr__(self, "_liberado", False)

    def __setattr__(self, *_args) -> None:
        raise RuleAuthorityError("un lease no se muta")

    @property
    def version_estable(self) -> VersionMonotonica:
        if self._liberado:
            raise RuleAuthorityError("el lease ya esta liberado")
        return self._version

    def liberar(self) -> None:
        object.__setattr__(self, "_liberado", True)


class _ConLeases:
    """Base de dobles con leases: compartidos reentrantes, exclusivo exige
    exclusividad real (falla si hay compartidos vivos — el contrato)."""

    def __init__(self) -> None:
        self._versiones = RelojDeVersiones()
        self._base = self._versiones.siguiente()
        self._compartidos = 0

    def version_actual(self) -> VersionMonotonica:
        return self._base

    @contextmanager
    def lease_compartido(self) -> Iterator[_Lease]:
        self._compartidos += 1
        lease = _Lease(self._base, exclusivo=False)
        try:
            yield lease
        finally:
            self._compartidos -= 1
            lease.liberar()

    @contextmanager
    def lease_exclusivo(self) -> Iterator[_Lease]:
        if self._compartidos:
            raise RuleAuthorityError(
                "lease exclusivo con compartidos vivos: el writer espera (orden §9)")
        anterior, self._base = self._base, self._versiones.siguiente()
        lease = _Lease(self._base, exclusivo=True)
        try:
            yield lease
        finally:
            assert self._base.es_posterior_a(anterior)     # monotonia estructural
            lease.liberar()


# ------------------------------------------------- las cinco interfaces (§9)

class ProveedorPinActivo(Protocol):
    """Entrega el pin activo y su procedencia EXTERNA. Con leases."""

    def lease_compartido(self): ...
    def lease_exclusivo(self): ...
    def version_actual(self) -> VersionMonotonica: ...

    def pin_activo(self) -> tuple[TrustedPolicyPin, str]:
        """(pin, procedencia externa del pin). Nunca un nombre movil.
        Sincrono a proposito (paso 7, sin wiring): el kernel lo envuelve si
        su fuente real es de I/O."""


class EstadoStop(NamedTuple):
    activo: bool
    version: VersionMonotonica
    huella: str                      # version o huella de la fuente unica


class ProveedorStop(Protocol):
    """Lee la fuente unica de STOP. Desconocido -> StopDesconocido (DENY)."""

    def lease_compartido(self): ...
    def lease_exclusivo(self): ...
    def version_actual(self) -> VersionMonotonica: ...

    def estado(self) -> EstadoStop: ...


class RelojConfiable(Protocol):
    """UTC consciente; detecta retroceso respecto de la emision."""

    def ahora(self) -> datetime: ...
    def marcar_emision(self, momento: datetime) -> None: ...


class ContratoCapability(NamedTuple):
    """Clasificacion confiable: identidad y version inmutables del contrato."""
    identidad: str
    version: str
    clase: str                       # REVERSIBLE | OBLIGATING (nunca rebajada por YAML)
    unidad: str | None
    moneda: str | None


class ProveedorClasificacion(Protocol):
    """Fija clase, unidad o moneda y forma de limites. Con leases."""

    def lease_compartido(self): ...
    def lease_exclusivo(self): ...
    def version_actual(self) -> VersionMonotonica: ...

    def contrato_de(self, capability: str) -> ContratoCapability: ...


class AlmacenCheckpoints(Protocol):
    """Checkpoint externo, monotónico y OBLIGATORIO (§11): sin el, no hay
    emision ni consumo. Rechaza retrocesos y heads conflictivos."""

    def head_actual(self) -> str: ...
    def publicar(self, head: str, *, anterior: str) -> None: ...


# ------------------------------------------------------------- dobles (§9)

class PinFijo(_ConLeases):
    """Doble determinista: un pin fijo con su procedencia externa."""

    def __init__(self, pin: TrustedPolicyPin, procedencia_externa: str) -> None:
        super().__init__()
        self._pin = pin
        self._procedencia = procedencia_externa

    def pin_activo(self) -> tuple[TrustedPolicyPin, str]:
        return self._pin, self._procedencia


class StopFijo(_ConLeases):
    """Doble determinista: STOP conocido (activo o inactivo, con huella)."""

    def __init__(self, *, activo: bool, huella: str) -> None:
        super().__init__()
        self._activo = activo
        self._huella = huella

    def estado(self) -> EstadoStop:
        return EstadoStop(activo=self._activo, version=self.version_actual(),
                          huella=self._huella)


class ProveedorSinLeases:
    """Doble NEGATIVO (el anti-contrato): solo relee un valor fijo, sin leases
    ni version monotona. ``exigir_contrato_de_emision`` debe rechazarlo — asi
    se prueba que releer no alcanza para emitir ni consumir (§9)."""

    def __init__(self, valor: object = None) -> None:
        self._valor = valor

    def pin_activo(self):                           # type: ignore[empty-body]
        return self._valor

    def estado(self):                               # type: ignore[empty-body]
        return self._valor

    def contrato_de(self, capability: str):         # type: ignore[empty-body]
        return self._valor


class StopIlegible:
    """Doble que representa la fuente que no se puede leer: fail-closed."""

    def estado(self) -> EstadoStop:
        raise StopDesconocido("STOP ilegible o sin configurar: DENY")


class RelojDeterminista:
    """Doble determinista: hora fijable, UTC obligado, retroceso detectado."""

    def __init__(self, momento: datetime) -> None:
        self._momento = momento
        self._emision: datetime | None = None

    def fijar(self, momento: datetime) -> None:
        self._momento = momento

    @staticmethod
    def _consciente(momento: datetime) -> datetime:
        if momento.tzinfo is None or momento.utcoffset() is None:
            raise RelojInvalido("hora sin timezone: DENY")
        return momento.astimezone(timezone.utc)

    def ahora(self) -> datetime:
        momento = self._consciente(self._momento)
        if self._emision is not None and momento < self._emision:
            raise RelojRetrocedio(
                f"reloj retrocedio respecto de la emision ({momento} < {self._emision})")
        return momento

    def marcar_emision(self, momento: datetime) -> None:
        self._emision = self._consciente(momento)


class ClasificacionFija(_ConLeases):
    """Doble determinista: catalogo fijo de contratos por capability."""

    def __init__(self, contratos: dict[str, ContratoCapability]) -> None:
        super().__init__()
        self._contratos = dict(contratos)

    def contrato_de(self, capability: str) -> ContratoCapability:
        try:
            return self._contratos[capability]
        except KeyError:
            raise ClasificacionDesconocida(
                f"capability sin clasificacion confiable: {capability!r} (DENY, sin rebaja)"
            ) from None


class CheckpointsEnMemoria:
    """Doble determinista del checkpoint externo: log monotónico en memoria.
    Republicar el MISMO head con el MISMO anterior es idempotente; cualquier
    retroceso o salto conflictivo niega."""

    def __init__(self) -> None:
        self._head = ""
        self._historial: set[str] = set()

    def head_actual(self) -> str:
        return self._head

    def publicar(self, head: str, *, anterior: str) -> None:
        if not isinstance(head, str) or not head:
            raise CheckpointInvalido("head vacio")
        if self._head == "":                     # primera publicacion
            if anterior != "":
                raise CheckpointInvalido("primera publicacion con anterior no vacio")
        elif head == self._head:
            if anterior not in ("", self._head):
                raise CheckpointInvalido("head conflictivo: el anterior no encadena")
        else:
            if head in self._historial:
                raise CheckpointInvalido(
                    f"retroceso: {head!r} ya fue superado por {self._head!r}")
            if anterior != self._head:
                raise CheckpointInvalido(
                    f"head conflictivo: esperaba anterior {self._head!r}, llego {anterior!r}")
        self._historial.add(head)
        self._head = head


# ----------------------------------------------- el guard de emision/consumo

def _exigir_leases(nombre: str, proveedor: object) -> None:
    for metodo in ("lease_compartido", "lease_exclusivo", "version_actual"):
        if not callable(getattr(proveedor, metodo, None)):
            raise ProveedorInvalido(
                f"{nombre}: solo releer valores no satisface el contrato — "
                "sin leases (compartido/exclusivo) y version monotonica no se "
                "emite ni se consume (§9)")
    try:
        proveedor.version_actual()                       # type: ignore[attr-defined]
    except Exception as exc:
        raise ProveedorInvalido(f"{nombre}: sin version monotonica disponible") from exc


def exigir_contrato_de_emision(*, pin: object, checkpoint: object, stop: object,
                               reloj: object, clasificacion: object) -> None:
    """Lo que el kernel llama ANTES de evaluar o consumir (paso 7 lo provee;
    el evaluador lo invoca). Sin esto satisfecho, no hay emision ni consumo.

    - checkpoint es OBLIGATORIO: ``None`` jamas habilita emision (§9);
    - pin, STOP y clasificacion implementan leases con version monotonica:
      quien solo relee, niega;
    - reloj confiable con UTC consciente.
    """
    for nombre, proveedor in (("pin_activo", pin), ("stop", stop),
                              ("clasificacion", clasificacion)):
        if proveedor is None:
            raise ProveedorInvalido(f"falta el proveedor {nombre}")
        _exigir_leases(nombre, proveedor)
    if checkpoint is None:
        raise ProveedorInvalido("sin checkpoint externo no hay emision ni consumo (§11)")
    for metodo in ("head_actual", "publicar"):
        if not callable(getattr(checkpoint, metodo, None)):
            raise ProveedorInvalido("checkpoint: no implementa el contrato monotónico")
    if reloj is None:
        raise ProveedorInvalido("falta el reloj confiable")
    for metodo in ("ahora", "marcar_emision"):
        if not callable(getattr(reloj, metodo, None)):
            raise ProveedorInvalido("reloj: no implementa el contrato (UTC consciente)")
