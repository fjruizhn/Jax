"""Perfiles inmutables de salida para el experimento S-1.

La identidad del motor se decide antes de abrir el socket. Ninguna parte del
pedido HTTP puede elegir proveedor, destino o credencial.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class PerfilNoDisponible(ValueError):
    """El motor no tiene un adaptador S-1 aprobado."""


@dataclass(frozen=True)
class PerfilCredencial:
    motor: str
    upstream: str
    ruta: str
    credencial: str = field(repr=False)
    metodos: frozenset[str] = frozenset({"POST"})
    modelos: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.motor not in {"claude", "kimi"}:
            raise PerfilNoDisponible(f"{self.motor}: NO_S1")
        if not self.upstream.startswith("https://") and not self.upstream.startswith("http://127.0.0.1:"):
            raise ValueError("el upstream del perfil debe ser HTTPS o el upstream local de prueba")
        if not self.ruta.startswith("/") or "?" in self.ruta or "#" in self.ruta:
            raise ValueError("la ruta del perfil debe ser absoluta y sin query")
        if not self.credencial:
            raise ValueError("el perfil requiere una credencial")
        if not self.metodos or not self.modelos:
            raise ValueError("el perfil requiere metodos y modelos explicitamente permitidos")


def perfil_para_motor(motor: str, upstream: str) -> PerfilCredencial:
    """Solo expone el cierre explícito mientras no exista adaptador aprobado.

    La construcción con credencial real corresponde al resolvedor de
    configuración de producción, fuera de este helper puro.
    """
    raise PerfilNoDisponible(f"{motor}: NO_S1")
