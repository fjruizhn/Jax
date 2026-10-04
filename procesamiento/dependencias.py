"""Freno de dependencias de extraccion (jax-14, 2026-10-03).

El 2026-10-03 el venv de LAS MANOS en produccion no tenia `pdfplumber`,
`openpyxl` ni `python-docx` (declaradas en `requirements-archivos.txt`) y el
procesamiento dejo los PDF/DOCX/XLSX en `sin_extractor` en silencio. Este
modulo responde: ¿se pueden importar las dependencias declaradas?, y ¿que
tipos de archivo dependen de las que faltan?

La lista de paquetes sale de `requirements-archivos.txt`, no se duplica a
mano; solo los mapas paquete -> modulo y paquete -> extensiones son
explicitos. Todo lo que no se puede evaluar FALLA CERRADO:

- un paquete declarado sin mapa cuenta como faltante y frena todos los tipos;
- lineas `-r`, `-e`, otras opciones, URL o ruta local no se pueden comprobar:
  salen como `linea-no-reconocida:<n>` (el numero, NUNCA la linea cruda, que
  puede traer credenciales y el nombre va a log y a Telegram);
- un requirements vacio, ilegible o que no es UTF-8 da `ok=False`;
- los marcadores `;` se evaluan con `packaging.markers` si se puede importar;
  si no, o si el marcador no se puede evaluar, la linea se EXIGE igual.

`estado()` NUNCA lanza. Sin cache a proposito: con todo instalado cada
comprobacion es una busqueda en `sys.modules`, y no cachear permite ver el
arreglo sin reiniciar.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Callable

from procesamiento.tipos_imagen import EXTENSIONES_IMAGEN, leer_cabecera, tipo_por_contenido

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements-archivos.txt"

#: paquete (nombre en requirements, normalizado) -> modulo que se importa.
MODULO_POR_PAQUETE: dict[str, str] = {
    "openpyxl": "openpyxl",
    "pdfplumber": "pdfplumber",
    "python-docx": "docx",
    # `PIL.Image` y no `PIL`: `import PIL` no carga el binario `_imaging`, pero
    # en un proceso NUEVO `import PIL.Image` falla sin el (Image.py relanza el
    # error) y tambien si el binario es de otra version. OJO: si `PIL.Image` ya
    # estaba importado en el mismo proceso, un fallo posterior de `_imaging`
    # queda como `DeferredError` y esta comprobacion no lo ve.
    "pillow": "PIL.Image",
}

#: paquete -> extensiones que dejan de poder procesarse si falta (freno por tipo).
EXTENSIONES_POR_PAQUETE: dict[str, set[str]] = {
    "openpyxl": {".xlsx", ".xlsm"},
    "pdfplumber": {".pdf"},
    "python-docx": {".docx"},
    # el OCR de imagenes decodifica con Pillow antes de llamar a tesseract. La
    # MISMA lista que la compuerta (Jax#338 ronda 17: faltaba `.gif`).
    "pillow": set(EXTENSIONES_IMAGEN),
}

#: paquete -> tipo por CONTENIDO (`tipos_imagen.tipo_por_contenido`) que deja de
#: poder procesarse si falta. Jax#338 ronda 18: la compuerta enruta por la firma
#: (un PNG llamado `foto` va al OCR) y el freno decide igual.
CONTENIDO_POR_PAQUETE: dict[str, str] = {
    "pdfplumber": "pdf",
    "pillow": "imagen",
}

_NOMBRE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(?:$|[=<>!~;\s])")
_VACIO = "requirements vacío"


def _normalizar(nombre: str) -> str:
    return re.sub(r"[-_.]+", "-", nombre).lower()


def _marcador_excluye(marcador: str) -> bool:
    """True solo si `packaging` esta y evalua el marcador como FALSO."""
    try:
        from packaging.markers import Marker

        return not Marker(marcador).evaluate()
    except Exception:  # fail-soft: sin packaging, o marcador no evaluable, la linea se exige igual (falla cerrado); no hay nada que propagar
        return False


def _lineas_utiles(requirements: Path) -> list[tuple[int, str]]:
    texto = Path(requirements).read_text(encoding="utf-8")
    utiles: list[tuple[int, str]] = []
    for n, linea in enumerate(texto.splitlines(), start=1):
        linea = linea.split(" #", 1)[0].strip() if not linea.lstrip().startswith("#") else ""
        if linea:
            utiles.append((n, linea))
    return utiles


def _paquete_de(n: int, linea: str) -> str | None:
    """Nombre normalizado, `linea-no-reconocida:<n>` si no se puede comprobar,
    o None si un marcador falso excluye la linea."""
    especificacion, _, marcador = linea.partition(";")
    especificacion = especificacion.strip()
    if marcador.strip() and _marcador_excluye(marcador.strip()):
        return None
    coincide = _NOMBRE.match(especificacion)
    es_opcion_url_o_ruta = (
        especificacion.startswith(("-", ".", "/", "~"))
        or "://" in especificacion or "/" in especificacion or "@" in especificacion
    )
    if not coincide or es_opcion_url_o_ruta:
        return f"linea-no-reconocida:{n}"
    return _normalizar(coincide.group(1))


def paquetes_declarados(requirements: Path = REQUIREMENTS) -> list[str]:
    """Paquetes de cada linea util del requirements (sin version ni extras)."""
    paquetes = [_paquete_de(n, linea) for n, linea in _lineas_utiles(requirements)]
    return [p for p in paquetes if p is not None]


def estado(requirements: Path = REQUIREMENTS,
           importador: Callable[[str], object] | None = None) -> dict:
    """`{"ok": bool, "faltan": [...], "motivos": {paquete: tipo_de_error}}`. NUNCA lanza."""
    requirements = Path(requirements)
    importador = importador or importlib.import_module  # resuelto en la llamada, no al definir: asi se puede sustituir
    try:
        paquetes = paquetes_declarados(requirements)
    except (OSError, ValueError) as exc:  # fail-soft: requirements ilegible o no UTF-8 (UnicodeDecodeError es ValueError); se informa ok=False con el motivo en vez de propagar, el freno cierra
        return {"ok": False, "faltan": [requirements.name], "motivos": {requirements.name: type(exc).__name__}}
    if not paquetes:
        return {"ok": False, "faltan": [requirements.name], "motivos": {requirements.name: _VACIO}}
    faltan: dict[str, str] = {}
    for paquete in paquetes:
        modulo = MODULO_POR_PAQUETE.get(paquete)
        if modulo is None:
            faltan[paquete] = "sin mapa"
            continue
        try:
            importador(modulo)
        except Exception as exc:  # fail-soft: cualquier fallo al importar (ModuleNotFoundError, AttributeError, TypeError...) cuenta como faltante con su tipo de error; el freno no puede caerse por un modulo roto
            faltan[paquete] = type(exc).__name__
    lista = sorted(faltan)
    return {"ok": not lista, "faltan": lista, "motivos": {p: faltan[p] for p in lista}}


def faltantes(requirements: Path = REQUIREMENTS) -> list[str]:
    return estado(requirements)["faltan"]


def tipos_de(faltan: list[str]) -> list[str]:
    """Extensiones que dependen de los paquetes faltantes; `["*"]` si alguno no tiene mapa."""
    tipos: set[str] = set()
    for paquete in faltan:
        if paquete not in EXTENSIONES_POR_PAQUETE:
            return ["*"]
        tipos |= EXTENSIONES_POR_PAQUETE[paquete]
    return sorted(tipos)


def _tipo_de_contenido(ruta, abrir: Callable[[str], Path | None] | None) -> str | None:
    """El tipo por contenido de `ruta` (como la compuerta), o None si no se
    puede leer (sin `abrir`, ruta fuera del jail, archivo ausente) o si el
    contenido no decide."""
    if abrir is None:
        return None
    try:
        destino = abrir(ruta)
        # Ronda 19: sin bloquear; un FIFO (u otro no regular) no se lee
        cabecera = leer_cabecera(destino) if destino is not None else None
    except Exception:  # fail-soft: una ruta que no se puede resolver no da tipo por contenido y decide la extension (el respaldo de siempre); el freno nunca se cae por una ruta rota
        return None
    if cabecera is None:
        return None
    return tipo_por_contenido(cabecera, Path(str(ruta)).suffix)


def lote_afectado(faltan: list[str], rutas: list[str],
                  abrir: Callable[[str], Path | None] | None = None) -> bool:
    """True si el lote trae algo que dependa de un paquete faltante. Un faltante
    sin mapa afecta a todo lote no vacio (falla cerrado).

    Jax#338 ronda 18: decide como la compuerta. Con `abrir` (ruta -> archivo
    legible, o None), lee la cabecera: si el contenido es decisivo (`pdf`,
    `imagen`, `ole2`) manda el contenido; si no, o si no se puede leer, la
    extension, como siempre."""
    if not faltan or not rutas:
        return False
    tipos = tipos_de(faltan)
    if tipos == ["*"]:
        return True
    contenidos = {CONTENIDO_POR_PAQUETE[p] for p in faltan if p in CONTENIDO_POR_PAQUETE}
    for ruta in rutas:
        tipo = _tipo_de_contenido(ruta, abrir)
        if tipo is not None:
            if tipo in contenidos:
                return True
        elif Path(str(ruta)).suffix.lower() in tipos:
            return True
    return False


def codigo_de_import(exc: BaseException) -> str:
    """`dependencia_no_instalada` solo para ModuleNotFoundError; otro ImportError
    (presente pero rota) es `dependencia_rota`."""
    return "dependencia_no_instalada" if isinstance(exc, ModuleNotFoundError) else "dependencia_rota"
