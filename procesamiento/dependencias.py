"""Freno de dependencias de extraccion (jax-14, 2026-10-03).

El 2026-10-03 el venv de LAS MANOS en produccion no tenia `pdfplumber`,
`openpyxl` ni `python-docx` (declaradas en `requirements-archivos.txt`) y el
procesamiento dejo los PDF/DOCX/XLSX en `sin_extractor` en silencio. Este
modulo responde UNA pregunta: ¿se pueden importar las dependencias declaradas?

La lista de paquetes sale de `requirements-archivos.txt`, no se duplica a
mano; solo el MAPA paquete -> modulo es explicito (`python-docx` se importa
como `docx`). Un paquete declarado sin mapa cuenta como FALTANTE (falla
cerrado) y la prueba `_dependencias_test.py` lo atrapa antes.

Sin cache a proposito: cuando todo esta instalado cada comprobacion es una
busqueda en `sys.modules`; cuando falta algo, comprobar de nuevo es lo que
permite ver el arreglo sin reiniciar nada.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements-archivos.txt"

#: paquete (nombre en requirements, normalizado) -> modulo que se importa.
MODULO_POR_PAQUETE: dict[str, str] = {
    "openpyxl": "openpyxl",
    "pdfplumber": "pdfplumber",
    "python-docx": "docx",
}

_NOMBRE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalizar(nombre: str) -> str:
    return re.sub(r"[-_.]+", "-", nombre).lower()


def paquetes_declarados(requirements: Path = REQUIREMENTS) -> list[str]:
    """Nombres de paquete de cada linea del requirements (sin version ni extras)."""
    paquetes: list[str] = []
    for linea in Path(requirements).read_text(encoding="utf-8").splitlines():
        linea = linea.split("#", 1)[0].strip()
        if not linea or linea.startswith("-"):
            continue
        coincide = _NOMBRE.match(linea)
        if coincide:
            paquetes.append(_normalizar(coincide.group(1)))
    return paquetes


def faltantes(requirements: Path = REQUIREMENTS) -> list[str]:
    """Paquetes declarados que NO se pueden importar, ordenados por nombre."""
    faltan: list[str] = []
    for paquete in paquetes_declarados(requirements):
        modulo = MODULO_POR_PAQUETE.get(paquete)
        if modulo is None:
            faltan.append(paquete)  # sin mapa: no se puede afirmar que este
            continue
        try:
            importlib.import_module(modulo)
        except ImportError:
            faltan.append(paquete)
    return sorted(faltan)


def estado(requirements: Path = REQUIREMENTS) -> dict:
    """`{"ok": bool, "faltan": [...]}`; si el requirements no se puede leer, falla cerrado."""
    try:
        faltan = faltantes(requirements)
    except OSError:
        return {"ok": False, "faltan": [Path(requirements).name]}
    return {"ok": not faltan, "faltan": faltan}
