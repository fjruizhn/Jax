#!/usr/bin/env python3
"""El camino de entrada más simple: `procesar(proyecto, rutas) -> dict`, y su
línea de comandos. `procesamiento/` ya está terminado y auditado (jax#252,
252 tests, en verde) -- este guion es lo único que hoy lo pone a trabajar.

Estructura de salida -- el trabajo es el PROYECTO:

    $JAX_WORKSPACE_DIR/proyectos/<slug-del-proyecto>/
        fuente/      <- los originales (subcarpeta de origen conservada)
        procesado/   <- lo que ya produce `procesamiento.ingesta`

Dos reglas de recorrido:

  1. Se conserva la subcarpeta de origen dentro de `fuente/`. Que un
     documento venga de "Estados Financieros" y no de "Documentación
     Legal" es información. Cada componente de carpeta se slugifica (sin
     acentos, minúsculas, separado por guiones) -- el NOMBRE del archivo
     queda intacto, y la subruta se valida contra el mismo jail que el
     resto de `procesamiento.ingesta` (ver `_resolver_subruta_de_fuente`
     en ingesta.py).
  2. Las carpetas que empiezan con `_` se ignoran -- no se descienden, sus
     archivos no se listan ni se ingieren. Son corridas viejas de este
     mismo sistema (`_extractos/`, `_extractos_ocr/`), no material fuente:
     ingerirlas sería procesar nuestros propios extractos.

Lo que este guion NO hace, a propósito (fuera del encargo, no se agrega de
paso): no escribe `proyecto.json`, no toca permisos, no vigila carpetas, no
habla con Nextcloud, no tiene interfaz. Es el camino de entrada, nada más.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

from motor_registry import tool_authority

from procesamiento import ingesta

_NOMBRE_FICHA = "ficha.json"


def _slug(texto: str) -> str:
    """Minúsculas, sin acentos, sin caracteres que no sean `[a-z0-9]`
    (reemplazados por un único `-`), sin guiones al borde. Se usa TANTO
    para el nombre del proyecto (`proyectos/<slug>/`) como para cada
    componente de subcarpeta dentro de `fuente/` -- mismo criterio en los
    dos lugares, una sola función."""
    sin_acentos = (
        unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    )
    normalizado = re.sub(r"[^a-z0-9]+", "-", sin_acentos.lower()).strip("-")
    # Un componente que se queda vacío tras slugificar (ej. un nombre hecho
    # enteramente de símbolos) no puede desaparecer -- perdería el lugar en
    # la jerarquía. "carpeta" es genérico a propósito: no se inventa un
    # nombre más específico que no está en el dato de origen.
    return normalizado or "carpeta"


def _archivos_bajo(raiz: Path) -> list[tuple[Path, str]]:
    """Recorre `raiz` (archivo o carpeta) y devuelve pares `(archivo,
    subruta)`. `subruta` es la subcarpeta de origen, slugificada componente
    por componente ("Estados Financieros/EEFF.pdf" -> archivo=EEFF.pdf,
    subruta="estados-financieros") -- "" cuando `raiz` es un archivo suelto
    o el archivo está en la raíz de la carpeta pasada.

    Carpetas que empiezan con `_` (en CUALQUIER nivel) nunca se descienden
    -- se podan del recorrido, no aparecen ni en el reporte ni en
    `fuente/`."""
    raiz = Path(raiz)
    if raiz.is_file():
        return [(raiz, "")]

    encontrados: list[tuple[Path, str]] = []
    for actual, subcarpetas, nombres in os.walk(raiz):
        subcarpetas[:] = sorted(d for d in subcarpetas if not d.startswith("_"))
        actual_path = Path(actual)
        relativo = actual_path.relative_to(raiz)
        componentes = [c for c in relativo.parts]
        subruta = "/".join(_slug(c) for c in componentes)
        for nombre in sorted(nombres):
            encontrados.append((actual_path / nombre, subruta))
    return encontrados


def _tamano_extracto(carpeta: Path) -> int:
    """Suma el tamaño de las salidas reales en `procesado/<huella>/`,
    excluyendo `ficha.json` (metadato, no contenido). `0` si la carpeta no
    existe -- `error`/`sin_extractor` no dejan nada ahí. Mismo cálculo que
    `scripts/medir_utilidad_extractos.py::_tamano_extracto`."""
    if not carpeta.is_dir():
        return 0
    return sum(
        p.stat().st_size
        for p in carpeta.iterdir()
        if p.is_file() and p.name != _NOMBRE_FICHA
    )


def _cabe_en_tope(estado: str, original_bytes: int, extracto_bytes: int, max_bytes: int) -> bool:
    """¿Lo que queda para leer vía `file_read` cabe en
    `tool_authority.MAX_READ_BYTES`? Con extracto (`ok`/`parcial`) es el
    tamaño del extracto -- es lo que un `file_read` sobre `procesado/`
    entrega. Sin extracto (`error`/`sin_extractor`) lo único legible es el
    ORIGINAL en `fuente/`, así que se mide contra ESE tamaño."""
    referencia = extracto_bytes if estado in {"ok", "parcial"} else original_bytes
    return referencia <= max_bytes


def _medir_documento(archivo: Path, ficha, carpeta: Path, tiempo_s: float, max_bytes: int) -> dict:
    original_bytes = archivo.stat().st_size
    hay_extracto = ficha.estado in {"ok", "parcial"}
    extracto_bytes = _tamano_extracto(carpeta) if hay_extracto else 0
    return {
        "archivo": archivo.name,
        "estado": ficha.estado,
        "extractor": ficha.extractor,
        "original_bytes": original_bytes,
        "extracto_bytes": extracto_bytes,
        "tiempo_s": round(tiempo_s, 3),
        "cabe_en_tope": _cabe_en_tope(ficha.estado, original_bytes, extracto_bytes, max_bytes),
    }


def _resumen(documentos: list[dict], tiempo_total_s: float) -> dict:
    return {
        "total_documentos": len(documentos),
        "por_estado": dict(Counter(d["estado"] for d in documentos)),
        "caben_en_tope": sum(1 for d in documentos if d["cabe_en_tope"]),
        "tiempo_total_s": round(tiempo_total_s, 3),
    }


def procesar(proyecto: str, rutas: list[Path]) -> dict:
    """Ingiere `rutas` (archivos y/o carpetas) contra el proyecto
    `proyecto`, conservando subcarpetas y saltando las que empiezan con
    `_`. Devuelve el informe completo -- documento por documento y el
    resumen. No lanza por un documento individual que falle: la ficha de
    `ingesta.ingerir` YA declara `error`/`sin_extractor` sin excepción
    (fallo cerrado del propio núcleo, ver `procesamiento/resultado.py`);
    lo que sí puede propagar es un error de jail (ruta fuera del
    workspace), que es una condición real del llamador, no de un documento."""
    slug = _slug(proyecto)
    trabajo = tool_authority.WORKSPACE_ROOT / "proyectos" / slug

    documentos: list[dict] = []
    t_inicio = time.perf_counter()
    for ruta in rutas:
        for archivo, subruta in _archivos_bajo(Path(ruta)):
            t0 = time.perf_counter()
            ficha = ingesta.ingerir(archivo, trabajo, subruta=subruta or None)
            tiempo_s = time.perf_counter() - t0
            carpeta = ingesta.ruta_procesado(trabajo, ficha.sha256)
            documentos.append(
                _medir_documento(archivo, ficha, carpeta, tiempo_s, tool_authority.MAX_READ_BYTES)
            )
    tiempo_total_s = time.perf_counter() - t_inicio

    return {
        "proyecto": proyecto,
        "slug": slug,
        "trabajo": str(trabajo),
        "documentos": documentos,
        "resumen": _resumen(documentos, tiempo_total_s),
    }


def _miles(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _imprimir_informe(r: dict) -> None:
    print("=" * 100)
    print(f"PROCESAMIENTO -- proyecto '{r['proyecto']}' (slug: {r['slug']})")
    print(f"Trabajo: {r['trabajo']}")
    print("=" * 100)
    encabezado = (
        f"{'archivo':<40} {'estado':<13} {'extractor':<12} "
        f"{'original':>12} {'extracto':>12} {'seg':>8} {'cabe':>5}"
    )
    print(encabezado)
    print("-" * len(encabezado))
    for d in r["documentos"]:
        print(
            f"{d['archivo'][:40]:<40} {d['estado']:<13} {d['extractor']:<12} "
            f"{_miles(d['original_bytes']):>12} {_miles(d['extracto_bytes']):>12} "
            f"{d['tiempo_s']:>7.2f}s {'sí' if d['cabe_en_tope'] else 'no':>5}"
        )
    print("-" * len(encabezado))

    s = r["resumen"]
    print(f"Total documentos:     {s['total_documentos']}")
    print(f"Por estado:           {s['por_estado']}")
    print(
        f"Caben en el tope:     {s['caben_en_tope']} / {s['total_documentos']} "
        f"(tope: {_miles(tool_authority.MAX_READ_BYTES)} bytes, file_read)"
    )
    print(f"Tiempo total:         {s['tiempo_total_s']}s")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("proyecto", help="nombre del proyecto -- se slugifica para el nombre de carpeta")
    p.add_argument("rutas", nargs="+", type=Path, help="archivo(s) o carpeta(s) a ingerir")
    a = p.parse_args()

    r = procesar(a.proyecto, a.rutas)
    _imprimir_informe(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
