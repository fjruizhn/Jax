"""Excel → un CSV por hoja.

Una hoja de cálculo NO es texto: aplanarla a prosa destruye justo lo que la
hacía un balance (qué número en qué fila y en qué columna). Por eso el extracto
es CSV por hoja, no un .txt.

openpyxl y no LibreOffice porque LibreOffice convierte una sola hoja
(medido 2026-09-20 sobre un archivo real de 6 hojas: sacó 1, sin error).

`Resultado` (endurecido en a7ef8c9) exige que todo 'ok'/'parcial' tenga AL
MENOS UNA salida con contenido no vacío tras `.strip()`. Un libro cuyas
hojas están TODAS en blanco produciría puros CSV vacíos -- eso violaría esa
regla y `Resultado(...)` levantaría `ValueError`. Por eso ese caso se
detecta ACÁ, antes de construir el Resultado, y se devuelve 'error' con una
razón -- nunca se deja que la excepción se escape de este módulo. El caso
legítimo (una hoja en blanco ENTRE otras con datos) no cae en esta regla
porque sólo mira si TODAS están vacías.
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

from procesamiento.resultado import Resultado

EXTRACTOR = "openpyxl"


def _version() -> str:
    import openpyxl

    return openpyxl.__version__


def _slug(texto: str) -> str:
    s = re.sub(r"[^\w\-]+", "-", texto.strip(), flags=re.UNICODE).strip("-").lower()
    return s or "hoja"


def _hoja_a_csv(hoja) -> str:
    buffer = io.StringIO()
    escritor = csv.writer(buffer, lineterminator="\n")
    for fila in hoja.iter_rows(values_only=True):
        escritor.writerow(["" if c is None else c for c in fila])
    return buffer.getvalue()


def extraer(origen: Path) -> Resultado:
    import openpyxl

    try:
        libro = openpyxl.load_workbook(origen, data_only=True, read_only=True)
    except Exception as exc:  # archivo corrupto, no-zip, protegido
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR, version=_version(),
            detalle={"razon": f"no se pudo abrir: {type(exc).__name__}: {exc}"},
        )

    total = len(libro.worksheets)
    salidas: dict[str, str] = {}
    fallidas: list[str] = []
    for indice, hoja in enumerate(libro.worksheets, start=1):
        try:
            salidas[f"{indice:02d}-{_slug(hoja.title)}.csv"] = _hoja_a_csv(hoja)
        except Exception as exc:
            fallidas.append(f"{hoja.title}: {type(exc).__name__}: {exc}")
    libro.close()

    if not salidas:
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR, version=_version(),
            detalle={"razon": "ninguna hoja pudo extraerse", "fallidas": fallidas},
        )

    if not any(v.strip() for v in salidas.values()):
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR, version=_version(),
            detalle={
                "razon": "todas las hojas estan vacias",
                "hojas": total,
            },
        )

    estado = "ok" if len(salidas) == total else "parcial"
    return Resultado(
        estado=estado, salidas=salidas, extractor=EXTRACTOR, version=_version(),
        detalle={"hojas": total, "hojas_extraidas": len(salidas), "fallidas": fallidas},
    )
