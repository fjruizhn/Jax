"""Lo que devuelve todo extractor. Las reglas de fallo cerrado viven acá,
en el tipo, para que ningún extractor pueda saltárselas por descuido."""
from __future__ import annotations

from dataclasses import dataclass, field

from procesamiento.ficha import ESTADOS


@dataclass(frozen=True)
class Resultado:
    estado: str
    salidas: dict[str, str]
    detalle: dict
    extractor: str
    version: str

    def __post_init__(self) -> None:
        if self.estado not in ESTADOS:
            raise ValueError(f"estado inválido: {self.estado!r}")
        if self.estado in {"error", "sin_extractor"} and self.salidas:
            raise ValueError(
                f"un resultado '{self.estado}' no puede traer salidas: "
                "un extracto entregado tras un fallo hace que el modelo invente"
            )
        if self.estado in {"ok", "parcial"} and not self.salidas:
            raise ValueError(f"un resultado '{self.estado}' sin salidas no es válido")

    @property
    def hubo_extracto(self) -> bool:
        return bool(self.salidas)
