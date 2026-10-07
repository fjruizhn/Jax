"""Dobles deterministas de los proveedores (F1.1 paso 7 r2).

Viven SOLO en pruebas (decision de Hyde): el modulo de produccion no tiene
dobles. Eso no impide que alguien importe estos o escriba otros: lo que los
frena en produccion es el cableado (fuera de alcance), no el modulo. Estos son REALES donde el
contrato lo exige: RW-lock con condicion (lectores excluyen escritor y al
reves), CAS atomica bajo lock en el checkpoint, reloj monotónico por proceso.
La implementacion real de checkpoint persiste y hace fsync; esta no (memoria).
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from policy.rule_authority.errors import (
    CheckpointInvalido,
    ClasificacionDesconocida,
    ProveedorInvalido,
    RelojInvalido,
    RelojRetrocedio,
    RuleAuthorityError,
    StopDesconocido,
)
from policy.rule_authority.providers import (
    ClaseCapability,
    ContratoCapability,
    EstadoStop,
    PinActivo,
    VersionMonotonica,
    VistaLease,
)
from policy.rule_authority.snapshot import TrustedPolicyPin


class RWLock:
    """Lector-escritor real: el compartido espera si hay exclusivo vivo; el
    exclusivo espera a que no haya compartidos ni otro exclusivo. Contadores
    bajo lock (threading.Condition). Reentrada y promocion del MISMO hilo son
    errores tipados, no bloqueos eternos (declarado en el Protocolo):
    compartido dentro de exclusivo, exclusivo dentro de exclusivo y exclusivo
    dentro de compartido."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._lectores = 0
        self._ids_lectores: dict[int, int] = {}
        self._escritor = False
        self._dueno: int | None = None
        self._esperando = 0              # hilos bloqueados en wait(): observable por las pruebas

    def esperando(self) -> int:
        with self._cond:
            return self._esperando

    def _esperar(self) -> None:
        """Con ``self._cond`` tomado."""
        self._esperando += 1
        try:
            self._cond.wait()
        finally:
            self._esperando -= 1

    @contextmanager
    def leer(self) -> Iterator[None]:
        yo = threading.get_ident()
        with self._cond:
            if self._dueno == yo:
                raise RuleAuthorityError("compartido dentro de exclusivo: reentrada no soportada")
            while self._escritor:
                self._esperar()
            self._lectores += 1
            self._ids_lectores[yo] = self._ids_lectores.get(yo, 0) + 1
        try:
            yield
        finally:
            with self._cond:
                self._lectores -= 1
                self._ids_lectores[yo] -= 1
                if not self._ids_lectores[yo]:
                    del self._ids_lectores[yo]
                self._cond.notify_all()

    @contextmanager
    def escribir(self) -> Iterator[None]:
        yo = threading.get_ident()
        with self._cond:
            if self._dueno == yo:
                raise RuleAuthorityError("exclusivo dentro de exclusivo: reentrada no soportada")
            if yo in self._ids_lectores:
                raise RuleAuthorityError("exclusivo dentro de compartido: promocion no soportada")
            while self._escritor or self._lectores:
                self._esperar()
            self._escritor = True
            self._dueno = yo
        try:
            yield
        finally:
            with self._cond:
                self._dueno = None
                self._escritor = False
                self._cond.notify_all()


class _BaseConLeases:
    """Base de dobles: valor+version del mismo instante DENTRO del lease; el
    exclusivo entrega la version nueva y excluye a todos; versiones estrictamente
    crecientes (obligacion del emisor, §9)."""

    def __init__(self, valor_inicial: object) -> None:
        self._lock = RWLock()
        self._numero = 0
        self._valor = valor_inicial

    def _version(self) -> VersionMonotonica:
        return VersionMonotonica(self._numero)

    def esperando(self) -> int:
        """Hilos bloqueados esperando un lease (para pruebas deterministas)."""
        return self._lock.esperando()

    @contextmanager
    def lease_compartido(self) -> Iterator[VistaLease]:
        with self._lock.leer():
            yield VistaLease(self._valor, self._version())     # mismo instante

    @contextmanager
    def lease_exclusivo(self) -> Iterator[VistaLease]:
        with self._lock.escribir():
            # Valor y version del MISMO instante: entrar NO avanza la version;
            # avanza al ESCRIBIR (_reemplazar), y los compartidos posteriores ven
            # el valor nuevo con la version nueva.
            yield VistaLease(self._valor, self._version())

    def _reemplazar(self, valor: object) -> None:
        """Solo con el exclusivo tomado POR ESTE HILO: valor y version cambian juntos."""
        if self._lock._dueno != threading.get_ident():
            raise RuleAuthorityError("escribir exige el lease exclusivo del hilo")
        self._valor = valor
        self._numero += 1


class PinFijo(_BaseConLeases):
    """Doble: el pin con la procedencia que se le diga. NO valida: la forma de
    la procedencia es del GUARD (las pruebas necesitan un lease valido con un
    pin malo para demostrar que el guard lo niega)."""

    def __init__(self, pin: TrustedPolicyPin, procedencia: str) -> None:
        super().__init__(PinActivo(pin, procedencia))


class StopFijo(_BaseConLeases):
    """Doble: STOP con lo que se le diga. NO valida (lo valida el guard)."""

    def __init__(self, *, activo: bool, huella: str) -> None:
        super().__init__(EstadoStop(activo=activo, huella=huella))


class StopIlegible:
    """Doble: la fuente que no se puede leer — el lease lanza el error tipado
    (fail-closed). Implementa el Protocolo: el guard lo niega por ILEGIBLE."""

    @contextmanager
    def lease_compartido(self) -> Iterator[VistaLease]:
        raise StopDesconocido("STOP ilegible o sin configurar: DENY")
        yield                                            # pragma: no cover

    def lease_exclusivo(self):
        return self.lease_compartido()


class RelojDeterminista:
    """Doble: hora fijable, UTC obligado, MONOTONICO POR PROCESO — el retroceso
    se detecta contra la ultima lectura; marcar_emision nunca baja el piso."""

    def __init__(self, momento: datetime) -> None:
        # El constructor NO valida: la hora ingenua se niega al USARLA (ahora /
        # marcar_emision), tipada — igual que un reloj real que no puede leerse.
        self._momento = momento
        self._piso: datetime | None = None
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
        if self._piso is not None and momento < self._piso:
            raise RelojRetrocedio(f"reloj retrocedio: {momento} < {self._piso}")
        self._piso = momento                                   # la ultima lectura es el piso
        return momento

    def marcar_emision(self, momento: datetime) -> None:
        momento = self._consciente(momento)
        if self._piso is not None and momento < self._piso:
            raise RelojRetrocedio("marcar_emision no puede bajar el piso del reloj")
        self._emision = momento
        self._piso = max(self._piso or momento, momento)


class CatalogoFijo:
    """Valor del lease de clasificacion: capability ausente => ClasificacionDesconocida."""

    def __init__(self, contratos: dict) -> None:
        self._contratos = dict(contratos)

    def contrato_de(self, capability: str) -> ContratoCapability:
        try:
            return self._contratos[capability]
        except KeyError:
            raise ClasificacionDesconocida(
                f"capability sin clasificacion confiable: {capability!r} (el kernel NIEGA)") from None

    def con(self, capability: str, contrato: ContratoCapability) -> "CatalogoFijo":
        return CatalogoFijo({**self._contratos, capability: contrato})


class ClasificacionFija(_BaseConLeases):
    """Doble: catalogo fijo. Clase del enum cerrado o niega; contrato mal formado
    niega; republicar una capability con clase MAS BAJA niega (sin rebaja)."""

    _JERARQUIA = {ClaseCapability.REVERSIBLE: 0, ClaseCapability.OBLIGATING: 1}

    def __init__(self, contratos: dict) -> None:
        for capability, contrato in contratos.items():
            if not isinstance(contrato, ContratoCapability):
                raise ProveedorInvalido(f"{capability}: contrato no es ContratoCapability")
            if not isinstance(contrato.clase, ClaseCapability):
                raise ProveedorInvalido(f"{capability}: clase fuera del enum cerrado")
        super().__init__(CatalogoFijo(contratos))

    def republicar(self, capability: str, contrato: ContratoCapability) -> None:
        """Con el EXCLUSIVO tomado; la clase nunca baja (sin rebaja)."""
        with self.lease_exclusivo() as vista:
            try:
                actual = vista.valor.contrato_de(capability)       # type: ignore[attr-defined]
            except ClasificacionDesconocida:
                actual = None
            if actual is not None and self._JERARQUIA[contrato.clase] < self._JERARQUIA[actual.clase]:
                raise ProveedorInvalido(
                    f"{capability}: sin rebaja — {contrato.clase.value} baja de {actual.clase.value}")
            self._reemplazar(vista.valor.con(capability, contrato))  # type: ignore[attr-defined]


class CheckpointsEnMemoria:
    """Doble del checkpoint externo: CAS ATOMICA bajo lock, retroceso y head
    conflictivo niegan, idempotencia SOLO con el mismo anterior Y el mismo head,
    y confirmar relee el log. En memoria: la implementacion real persiste y
    hace fsync antes de responder (esto no)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._head = ""
        self._log: list[tuple[str, str]] = []                  # (anterior, head)

    def head_actual(self) -> str:
        with self._lock:
            return self._head

    def publicar(self, head: str, *, anterior: str) -> None:
        if not isinstance(head, str) or not head:
            raise CheckpointInvalido("head vacio")
        with self._lock:                                       # CAS: anterior exacto bajo lock
            if self._head == "":
                if anterior != "":
                    raise CheckpointInvalido("primera publicacion con anterior no vacio")
            elif head == self._head and anterior == (self._log[-1][0] if self._log else ""):
                return                                         # idempotencia exacta
            elif head == self._head:
                raise CheckpointInvalido(
                    f"head repetido con anterior distinto ({anterior!r} != {self._log[-1][0]!r})")
            else:
                if any(head == h for _, h in self._log):
                    raise CheckpointInvalido(f"retroceso: {head!r} ya fue superado")
                if anterior != self._head:
                    raise CheckpointInvalido(
                        f"head conflictivo: esperaba anterior {self._head!r}, llego {anterior!r}")
            self._log.append((anterior, head))
            self._head = head

    def confirmar(self, head: str) -> bool:
        """Relee: el log CONTIENE el head exacto (§10 paso 15, §11). No exige
        que sea el actual: una operacion posterior pudo agregar un descendiente."""
        with self._lock:
            return any(h == head for _, h in self._log)


class ProveedorSinLeases:
    """Doble NEGATIVO: solo relee valores, sin leases (el anti-contrato). El guard niega."""

    def __init__(self, valor: object = None) -> None:
        self._valor = valor

    def pin_activo(self):
        return self._valor

    def estado(self):
        return self._valor

    def contrato_de(self, capability: str):
        return self._valor
