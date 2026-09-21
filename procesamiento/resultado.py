"""Lo que devuelve todo extractor. Las reglas de fallo cerrado viven acá,
en el tipo, para que ningún extractor pueda saltárselas por descuido.

Ronda de arreglo 1 (2026-09-20, auditoría task-1-hallazgos.md):
- C-1/C-2: `salidas` y `detalle` se congelan con `MappingProxyType` tras
  copiarlos -- `frozen=True` sólo impide reasignar el nombre del campo, no
  mutar el diccionario que ese nombre apunta.
- C-3: la regla de fallo cerrado cubre explícitamente `sin_extractor`
  (antes sólo `error` tenía un test), el catálogo de `ESTADOS` se ejercita
  con un estado inventado, y `hubo_extracto` se prueba también en `False`.
- I-1: un `ok`/`parcial` con TODAS las salidas de contenido vacío (tras
  `.strip()`) se rechaza -- exigir que TODAS tengan contenido rompería el
  caso legítimo de una hoja en blanco entre seis; la regla es "al menos
  una con contenido".
- I-2: `salidas` tiene que ser un `Mapping` -- una lista o un string se
  rechazan en el origen, no revientan lejos de acá.
- I-3: `extractor` y `version` no pueden quedar vacíos (tras `.strip()`).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

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
            raise ValueError(f"estado inválido: {self.estado!r}. Válidos: {sorted(ESTADOS)}")

        if not isinstance(self.salidas, Mapping):
            raise ValueError(
                "salidas tiene que ser un mapa (dict), no "
                f"{type(self.salidas).__name__}"
            )

        salidas_copia = dict(self.salidas)
        detalle_copia = dict(self.detalle)

        if self.estado in {"error", "sin_extractor"} and salidas_copia:
            raise ValueError(
                f"un resultado '{self.estado}' no puede traer salidas: "
                "un extracto entregado tras un fallo hace que el modelo invente"
            )
        if self.estado in {"ok", "parcial"}:
            if not salidas_copia:
                raise ValueError(f"un resultado '{self.estado}' sin salidas no es válido")
            if not any(str(v).strip() for v in salidas_copia.values()):
                raise ValueError(
                    f"un resultado '{self.estado}' con todas las salidas vacías no "
                    "es válido: al menos una tiene que tener contenido"
                )

        if not self.extractor.strip():
            raise ValueError("extractor no puede estar vacío")
        if not self.version.strip():
            raise ValueError("version no puede estar vacía")

        object.__setattr__(self, "salidas", MappingProxyType(salidas_copia))
        object.__setattr__(self, "detalle", MappingProxyType(detalle_copia))

    @property
    def hubo_extracto(self) -> bool:
        return bool(self.salidas)
