"""La ficha de un extracto: quién lo hizo, con qué versión y si salió bien.

`extractor` y `extractor_version` no son adorno: son lo que permite reprocesar
todo el día que cambie una herramienta, sin releer los originales a ciegas.
Es lo que faltó en la migración a bge-m3 (2026-09-12).

Ronda de arreglo 1 (2026-09-20, auditoría task-1-hallazgos.md):
- C-1/C-2/I-4: `detalle` se congela con `MappingProxyType` tras copiarlo, así
  que mutar el dict devuelto (o el que se pasó al constructor) no cambia la
  Ficha ya construida.
- I-3: `sha256`, `origen`, `extractor`, `extractor_version` y `fecha` no
  pueden quedar vacíos (tras `.strip()`) -- una `extractor_version=""` no
  permite decidir si hay que reprocesar, que es la única razón de ese campo.
- I-5: `desde_json` valida que el JSON sea un objeto con exactamente las
  claves esperadas, y levanta `ValueError` con el detalle -- no el
  `TypeError` crudo de un constructor mal llamado.
- I-6: `detalle` se valida en la construcción con un viaje real a JSON y
  vuelta (`json.loads(json.dumps(...))`); si no sobrevive igual (tuplas,
  claves no-str, bytes), `ValueError` ahí mismo, no tres capas después.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType

ESTADOS = frozenset({"ok", "parcial", "error", "sin_extractor"})

_CAMPOS_NO_VACIOS = ("sha256", "origen", "extractor", "extractor_version", "fecha")


def sha256_de(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for bloque in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


@dataclass(frozen=True)
class Ficha:
    sha256: str
    origen: str
    extractor: str
    extractor_version: str
    fecha: str
    estado: str
    detalle: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for nombre in _CAMPOS_NO_VACIOS:
            valor = getattr(self, nombre)
            if not isinstance(valor, str) or not valor.strip():
                raise ValueError(f"{nombre} no puede estar vacío")

        if self.estado not in ESTADOS:
            raise ValueError(
                f"estado inválido: {self.estado!r}. Válidos: {sorted(ESTADOS)}"
            )

        detalle_copia = dict(self.detalle)
        try:
            ida_vuelta = json.loads(json.dumps(detalle_copia))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"detalle no es serializable a JSON sin pérdida: {exc}"
            ) from exc
        if ida_vuelta != detalle_copia:
            raise ValueError(
                "detalle no sobrevive el viaje a JSON y vuelta sin cambios "
                "(¿tuplas, claves que no son str, bytes?)"
            )

        # Frozen no congela el diccionario, sólo el nombre -- MappingProxyType
        # cierra ese hueco. La dataclass deja de ser hasheable; es esperado.
        object.__setattr__(self, "detalle", MappingProxyType(detalle_copia))

    def a_json(self) -> str:
        payload = {
            "sha256": self.sha256,
            "origen": self.origen,
            "extractor": self.extractor,
            "extractor_version": self.extractor_version,
            "fecha": self.fecha,
            "estado": self.estado,
            "detalle": dict(self.detalle),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)

    @classmethod
    def desde_json(cls, s: str) -> "Ficha":
        try:
            datos = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError(f"no es JSON válido: {exc}") from exc

        if not isinstance(datos, dict):
            raise ValueError(
                f"la ficha tiene que ser un objeto JSON, no {type(datos).__name__}"
            )

        esperadas = {f.name for f in fields(cls)}
        recibidas = set(datos)
        if recibidas != esperadas:
            partes = []
            de_mas = recibidas - esperadas
            de_menos = esperadas - recibidas
            if de_mas:
                partes.append(f"de más: {sorted(de_mas)}")
            if de_menos:
                partes.append(f"de menos: {sorted(de_menos)}")
            raise ValueError(
                "la ficha no tiene las claves esperadas (" + "; ".join(partes) + ")"
            )

        return cls(**datos)
