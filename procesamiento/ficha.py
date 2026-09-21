"""La ficha de un extracto: quién lo hizo, con qué versión y si salió bien.

`extractor` y `extractor_version` no son adorno: son lo que permite reprocesar
todo el día que cambie una herramienta, sin releer los originales a ciegas.
Es lo que faltó en la migración a bge-m3 (2026-09-12).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

ESTADOS = frozenset({"ok", "parcial", "error", "sin_extractor"})


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
        if self.estado not in ESTADOS:
            raise ValueError(
                f"estado inválido: {self.estado!r}. Válidos: {sorted(ESTADOS)}"
            )

    def a_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, indent=2)

    @classmethod
    def desde_json(cls, s: str) -> "Ficha":
        return cls(**json.loads(s))
