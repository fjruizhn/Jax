"""Ingesta: el original entra a `fuente/` y su extracto queda en `procesado/`.

Dos invariantes:
  1. `fuente/` es inmutable. El original entra y no se toca mas. Dos
     archivos DISTINTOS con el mismo nombre -- un `balance.xlsx` de un
     cliente y otro `balance.xlsx` de otro -- no pueden pisarse: el
     segundo tiene que sobrevivir SIN destruir al primero.
  2. `procesado/` es DESECHABLE: se borra entero y se reconstruye. Nada que
     no sea reconstruible vive ahi. Un cache del que no te podes fiar para
     borrarlo no es un cache: es una segunda base de datos.

El cache se indexa por `sha256` del original. Si el original cambia, la
huella cambia y el extracto se regenera -- nunca se sirve uno viejo en
silencio.

Ronda de arreglo (2026-09-21, sobre el draft de task-8-brief.md, hallazgo de
Fernando ANTES de escribir código): el draft resolvia el nombre en
`fuente/` con `destino = fuente / origen.name` sin chequear colision --
dos archivos de contenido distinto con el mismo nombre hacian que el
segundo PISARA al primero en silencio, perdiendo el original de alguien.
Ruling: ante colision de nombre con contenido DISTINTO, se guarda como
`<nombre>-<primeros 8 del sha256><extension>`, y la ficha registra el
nombre REAL almacenado (`destino`), no el que traia el archivo. Si el
archivo ya esta en `fuente/` con la MISMA huella, no hay colision real y
no se copia de nuevo -- eso ya estaba bien en el draft.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from procesamiento import compuerta
from procesamiento.ficha import Ficha, sha256_de


def ruta_procesado(trabajo: Path, huella: str) -> Path:
    return Path(trabajo) / "procesado" / huella


def _ahora() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _destino_en_fuente(fuente: Path, origen: Path, huella: str) -> Path:
    """Elige el nombre REAL bajo el que `origen` vive en `fuente/`.

    Sin colision (el nombre esta libre, o ya esta ocupado por el MISMO
    contenido): el nombre original. Con colision de nombre y contenido
    DISTINTO: `<nombre>-<primeros 8 del sha256><extension>` -- y si ESE
    nombre alterno tambien colisiona por contenido distinto (colision de
    prefijo de hash, astronomicamente improbable pero no imposible), se
    usa la huella completa, que es unica por definicion.
    """
    candidato = fuente / origen.name
    if not candidato.exists() or sha256_de(candidato) == huella:
        return candidato

    alterno = fuente / f"{origen.stem}-{huella[:8]}{origen.suffix}"
    if not alterno.exists() or sha256_de(alterno) == huella:
        return alterno

    return fuente / f"{origen.stem}-{huella}{origen.suffix}"


def ingerir(origen: Path, trabajo: Path) -> Ficha:
    origen = Path(origen)
    trabajo = Path(trabajo)
    fuente = trabajo / "fuente"
    fuente.mkdir(parents=True, exist_ok=True)

    huella = sha256_de(origen)
    destino = _destino_en_fuente(fuente, origen, huella)
    if not destino.exists():
        shutil.copy2(origen, destino)

    carpeta = ruta_procesado(trabajo, huella)
    ficha_json = carpeta / "ficha.json"
    if ficha_json.is_file():
        # Cache vivo: misma huella, mismo extracto. Cero trabajo.
        return Ficha.desde_json(ficha_json.read_text(encoding="utf8"))

    resultado = compuerta.extraer(destino)

    # Escritura atomica por carpeta: primero temporal, despues rename. Si
    # algo se cae a la mitad, no queda un `procesado/` a medias que
    # parezca completo -- `carpeta` (el nombre final) solo aparece de un
    # solo golpe, via `rename()`.
    temporal = carpeta.with_name(carpeta.name + ".parcial")
    if temporal.exists():
        shutil.rmtree(temporal)
    temporal.mkdir(parents=True)

    for nombre, contenido in resultado.salidas.items():
        (temporal / nombre).write_text(contenido, encoding="utf8")

    ficha = Ficha(
        sha256=huella,
        origen=str(destino.relative_to(trabajo)),
        extractor=resultado.extractor,
        extractor_version=resultado.version,
        fecha=_ahora(),
        estado=resultado.estado,
        detalle=resultado.detalle,
    )
    (temporal / "ficha.json").write_text(ficha.a_json(), encoding="utf8")

    if carpeta.exists():
        shutil.rmtree(carpeta)
    temporal.rename(carpeta)
    return ficha
