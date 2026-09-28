"""El diff de una misión de código, como estructura (spec 2026-09-28 §3.3.1).
C1 de la entrega decide sobre ESTO, no sobre texto plano con regex."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cambio:
    ruta: str
    estado: str
    ruta_anterior: str | None
    agregadas: tuple[str, ...]
    quitadas: tuple[str, ...]


def _cerrar(actual: dict | None, salida: list) -> None:
    if actual is not None:
        salida.append(Cambio(actual["ruta"], actual["estado"], actual["anterior"],
                             tuple(actual["mas"]), tuple(actual["menos"])))


def parsear(texto_diff: str) -> tuple[Cambio, ...]:
    salida: list[Cambio] = []
    actual: dict | None = None
    for linea in texto_diff.splitlines():
        if linea.startswith("diff --git "):
            _cerrar(actual, salida)
            b = linea.split(" b/", 1)[1]
            actual = {"ruta": b, "estado": "M", "anterior": None, "mas": [], "menos": []}
        elif actual is None:
            continue
        elif linea.startswith("new file mode"):
            actual["estado"] = "A"
        elif linea.startswith("deleted file mode"):
            actual["estado"] = "D"
        elif linea.startswith("rename from "):
            actual["estado"], actual["anterior"] = "R", linea[len("rename from "):]
        elif linea.startswith("rename to "):
            actual["ruta"] = linea[len("rename to "):]
        elif linea.startswith("+++ ") or linea.startswith("--- "):
            continue
        elif linea.startswith("+"):
            actual["mas"].append(linea[1:])
        elif linea.startswith("-"):
            actual["menos"].append(linea[1:])
    _cerrar(actual, salida)
    return tuple(salida)

