#!/usr/bin/env python3
"""Lee los pisos de CI de `ci/pisos.json`. Falla CERRADO: nunca hay un piso por defecto.

Uso (desde un paso de `.github/workflows/policy.yml`):

    python3 ci/piso.py verificar <clave> <archivo-con-la-salida-de-pytest>
    python3 ci/piso.py minimo <clave>

Códigos de salida:
    0  el piso se cumple (verificar) o se imprimió el mínimo (minimo)
    1  el piso NO se cumple: se imprime el mensaje de la clave
    2  el uso, el archivo de pisos o el archivo de salida están mal: ningún piso se
       compara, y el paso del workflow tiene que salir con error

Solo biblioteca estándar: corre en cualquier job, haya instalado pyyaml o no.
La historia de cada piso está en `docs/ci/pisos.md`.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RUTA_PISOS = Path(__file__).resolve().parent / "pisos.json"
VERSION = 1


class PisoError(Exception):
    """Cualquier problema que impida conocer un piso. Nunca se resuelve con un valor por defecto."""


def cargar(ruta: Path = RUTA_PISOS) -> dict:
    try:
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    except OSError as e:
        raise PisoError(f"no se puede leer {ruta}: {e}") from e
    except ValueError as e:
        raise PisoError(f"{ruta} no es JSON válido: {e}") from e
    if not isinstance(datos, dict) or datos.get("version") != VERSION:
        raise PisoError(f"{ruta}: falta 'version': {VERSION}")
    for seccion in ("pisos", "minimos"):
        if not isinstance(datos.get(seccion), dict):
            raise PisoError(f"{ruta}: falta la sección '{seccion}'")
    return datos


def _entrada(datos: dict, seccion: str, clave: str) -> object:
    if clave not in datos[seccion]:
        raise PisoError(f"la clave '{clave}' no está en la sección '{seccion}' de pisos.json")
    return datos[seccion][clave]


def piso(datos: dict, clave: str) -> tuple[str, str]:
    """(patron, mensaje) de un piso de tests corridos."""
    e = _entrada(datos, "pisos", clave)
    patron = e.get("patron") if isinstance(e, dict) else None
    mensaje = e.get("mensaje") if isinstance(e, dict) else None
    if not isinstance(patron, str) or not patron or not isinstance(mensaje, str) or not mensaje:
        raise PisoError(f"el piso '{clave}' necesita 'patron' y 'mensaje' como texto no vacío")
    try:
        re.compile(patron)
    except re.error as e:
        raise PisoError(f"el patrón del piso '{clave}' no compila: {e}") from e
    return patron, mensaje


def minimo(datos: dict, clave: str) -> int:
    e = _entrada(datos, "minimos", clave)
    if isinstance(e, bool) or not isinstance(e, int) or e < 1:
        raise PisoError(f"el mínimo '{clave}' tiene que ser un entero positivo")
    return e


def verificar(datos: dict, clave: str, archivo: str) -> bool:
    """True si el patrón aparece al inicio de alguna línea de `archivo` (como `grep -qE`)."""
    patron, mensaje = piso(datos, clave)
    try:
        texto = Path(archivo).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise PisoError(f"no se puede leer la salida de pytest {archivo}: {e}") from e
    if re.search(patron, texto, re.MULTILINE):
        return True
    print(mensaje)
    return False


def main(argv: list[str], ruta: Path = RUTA_PISOS) -> int:
    try:
        if len(argv) == 3 and argv[0] == "verificar":
            return 0 if verificar(cargar(ruta), argv[1], argv[2]) else 1
        if len(argv) == 2 and argv[0] == "minimo":
            print(minimo(cargar(ruta), argv[1]))
            return 0
        raise PisoError("uso: piso.py verificar <clave> <archivo> | piso.py minimo <clave>")
    except PisoError as e:
        print(f"ci/piso.py: ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
