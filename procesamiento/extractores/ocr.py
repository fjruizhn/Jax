"""OCR para lo que no tiene capa de texto: PDF escaneado e imagen.

tesseract por subproceso, nunca por binding. `spa` porque los documentos son
en español (verificado 2026-09-20: lee tildes y guion largo exacto).

Ronda de arreglo 1 (2026-09-21, task-5-6-hallazgos.md — NO ratificado, 3
críticos + 3 importantes sobre este archivo), tres lecciones que quedan acá
porque son las que se van a querer deshacer la próxima vez que un test se
ponga rojo:

- C-3: tesseract 5.5.0 **no lee PDF** (`Error in pixReadStream: Pdf reading
  is not supported`) -- defecto del brief original, no verificado contra la
  herramienta real. El módulo prometía "PDF escaneado e imagen" y sólo
  hacía lo segundo. Se rasteriza con `pdftoppm` (poppler, ya instalado --
  la misma suite que usa `pdftotext`) página por página, y se reporta por
  página igual que `pdf.py`: `paginas`, `paginas_sin_texto`.
- C-1: el PROMEDIO de confianza diluye -- seis renglones reales más cuatro
  de ruido promedian 77,73 (por encima de cualquier umbral razonable) y el
  extracto real termina en basura. Peor: con una foto movida, las palabras
  de baja confianza son EXACTAMENTE los números (`1,234,567` a 22 de
  confianza) mientras los rótulos van al 96 % -- el promedio tapa justo lo
  que importa. La defensa real es un piso POR PALABRA
  (`CONFIANZA_MINIMA_PALABRA`): las palabras por debajo se cuentan y se
  NOMBRAN en `detalle["palabras_dudosas"]` (nunca se borran del texto). Con
  más de la mitad de las palabras dudosas (`PROPORCION_MAXIMA_PALABRAS_DUDOSAS`)
  una IMAGEN sale `parcial` con `imagen_texto_dudoso`, CONSERVANDO el texto
  leído (Jax#338, decisión de Fernando; antes era `error`), y esa página de un
  PDF cuenta en `paginas_sin_texto`; con alguna pero no la mayoría es
  `parcial`. El promedio se sigue registrando SIEMPRE, como señal
  secundaria (I-1: antes desaparecía de `detalle` en el camino de "texto
  corto", justo la franja que hacía falta para calibrar el umbral).
- C-2: una página casi en blanco con sólo un membrete legible (pocas
  palabras, alta confianza) pasaba como `ok` -- la misma familia del
  defecto de los "cuarenta pies de página" de `pdf.py`. `MINIMO_PALABRAS`
  fuerza `parcial` (nunca `error`: un recibo legítimo puede tener pocas
  palabras) con la razón y las dimensiones de la imagen en `detalle`.

Los cuatro números (`MINIMO_CARACTERES`, `CONFIANZA_MINIMA_PALABRA`,
`MINIMO_PALABRAS`, `PROPORCION_MAXIMA_PALABRAS_DUDOSAS`) son las perillas
que un futuro ajuste va a querer mover -- cada una tiene su propio test que
la fija (I-3: antes se podían correr en una banda ancha sin que CI dijera
nada), y cada una se muta por separado.
"""
from __future__ import annotations

import array
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

from procesamiento.resultado import Resultado

EXTRACTOR = "tesseract"

# I-7 (final-hallazgos.md, ronda de cierre): mismo valor que
# `motor_registry.tool_authority.WORKSPACE_ROOT`,
# `jacobs/executor.py::HYDE_WORKSPACE_DIR` (y, hasta T16,
# `jax/muscles/subprocess_muscle.py`, retirado con el REPL) -- los call sites leen la MISMA env
# var en vez de importarse el módulo pesado de `tool_authority` unos de
# otros. El rasterizado de un PDF para OCR va acá, no al default de
# `tempfile` (`/tmp`), que en hall9000 es tmpfs -- RAM, en un hipervisor
# con dos VMs; un PDF patológico puede dejar decenas de PNG enormes
# simultáneamente en memoria.
#
# `.resolve()` (adenda MAJOR 2, final-hallazgos.md, ronda de cierre):
# `tool_authority.py:62` YA resuelve `WORKSPACE_ROOT` así -- acá faltaba.
# Si `JAX_WORKSPACE_DIR` trae un symlink, el jail (que compara contra la
# forma canónica) y este rasterizado terminan mirando dos rutas
# DISTINTAS -- una resuelta, una no -- aunque el valor de la env var sea
# el mismo en los dos.


def _resolver_workspace_dir() -> Path:
    """Separada de la constante de módulo para poder probar la resolución
    de symlinks sin recargar el módulo entero ni tocar el `_WORKSPACE_DIR`
    real que usa `extraer()`."""
    return Path(os.getenv("JAX_WORKSPACE_DIR", "/home/fruiz/jax-workspace")).resolve()


_WORKSPACE_DIR = _resolver_workspace_dir()

# Caracteres MÍNIMOS (tras `.strip()`) para que el modo texto plano cuente
# como "algo se leyó". Defensa barata contra el caso vacío (imagen en
# blanco, página pura imagen). Ante la duda, sube, nunca baja para que pase
# un fixture.
MINIMO_CARACTERES = 8

# Confianza mínima (0-100, la que reporta tesseract POR PALABRA en modo
# `tsv`) para que una palabra individual NO se cuente como dudosa. Medido a
# mano 2026-09-21: en una foto movida real, los NÚMEROS caían a confianza
# 22-52 mientras los rótulos seguían en 96 -- 60 separa cómodo ese caso sin
# castigar texto limpio.
CONFIANZA_MINIMA_PALABRA = 60.0

# Palabras mínimas reconocidas para que una página/imagen cuente como
# "cobertura suficiente". Por debajo: `parcial` (nunca `error` -- un recorte
# chico o un recibo legítimo puede tener pocas palabras, ver C-2).
MINIMO_PALABRAS = 10

# Proporción de palabras dudosas (confianza < CONFIANZA_MINIMA_PALABRA) que,
# superada, hace que la página entera se trate como de "baja confianza": una
# IMAGEN sale `parcial` con `imagen_texto_dudoso` (conserva el texto leído; con
# menos de MINIMO_CARACTERES es `imagen_sin_texto`, o `imagen_pagina_sin_texto`
# si es del tamaño de una página), y esa página de un PDF cuenta en
# `paginas_sin_texto` (si todas lo son, el PDF es `error`).
PROPORCION_MAXIMA_PALABRAS_DUDOSAS = 0.5

TIMEOUT_SEGUNDOS = 300

# DPI de rasterizado para OCR sobre PDF. 300 es el mínimo usual recomendado
# para OCR de documentos escaneados (por debajo, tesseract pierde exactitud
# en fuentes pequeñas de recibos/facturas).
DPI_RASTERIZADO = 300

_FIRMA_PDF = b"%PDF"

# Decision de Fernando (2026-10-03, regla de Jax#338): una IMAGEN que el OCR
# procesa sin texto confiable NO es un `error` -- `error` queda SOLO para un
# archivo danado o que no se puede abrir. Codigos estables en
# `detalle["codigo"]`, para quien triagea sin parsear la `razon`:
#
#   (A) menos de MINIMO_CARACTERES           -> ok      imagen_sin_texto
#   (B) mayoria de palabras dudosas          -> parcial imagen_texto_dudoso
#       (se CONSERVA el texto leido; B tiene prioridad sobre D)
#   (D) caso A con tamano de PAGINA          -> parcial imagen_pagina_sin_texto
#       (posible escaneo guardado como imagen: un documento sin texto SI es
#       un problema)
#   danada (firma invalida, no decodifica)  -> error   archivo_ilegible
#   sana pero no la leemos (modo F/I, GIF/WebP animado, leptonica la rechaza)
#                                            -> error   formato_no_soportado
#   demasiados pixeles / paginas             -> error   imagen_demasiado_grande
CODIGO_IMAGEN_SIN_TEXTO = "imagen_sin_texto"
CODIGO_IMAGEN_TEXTO_DUDOSO = "imagen_texto_dudoso"
CODIGO_IMAGEN_PAGINA_SIN_TEXTO = "imagen_pagina_sin_texto"
CODIGO_ARCHIVO_ILEGIBLE = "archivo_ilegible"
CODIGO_OCR_TIEMPO_EXCEDIDO = "ocr_tiempo_excedido"
CODIGO_FORMATO_NO_SOPORTADO = "formato_no_soportado"   # sano, pero no lo leemos
CODIGO_IMAGEN_DEMASIADO_GRANDE = "imagen_demasiado_grande"
CODIGO_OCR_SIN_MEMORIA = "ocr_sin_memoria"   # recursos, no archivo danado

# Version de la LOGICA de clasificacion de este extractor. `extractor_version`
# es la de tesseract y NO cambia cuando cambia una regla: esta cadena se guarda
# en la ficha (`detalle["_version_logica"]`) y `ingesta` la compara -- una
# ficha escrita con otra logica ni se reusa de cache ni cuenta como intento
# previo del tope D-2. SUBIRLA cada vez que cambie la regla.
# La version DEPENDE DEL CAMINO (`version_logica(camino)`): solo cambio la
# regla de las IMAGENES (A/B/D, codigos nuevos, TIFF multipagina) -- la del PDF
# escaneado no, asi que su cache sigue valiendo (no se re-OCR-ean los PDF).
# "2": regla de Fernando de la ronda 1 de Jax#338.
VERSION_LOGICA_IMAGEN = "2"


def version_logica(camino: str) -> str | None:
    """Version de la logica para un camino (`"imagen"` o `"pdf"`, el que decide
    el CONTENIDO del archivo, no su extension); `None` (sin marca) para el
    PDF."""
    return VERSION_LOGICA_IMAGEN if camino == "imagen" else None


# `Resultado` exige al menos una salida con contenido para `ok`: la unica
# salida del caso A es este aviso (nunca texto inventado). Regla (C) de
# Fernando: dice SOLO que el OCR no encontro texto.
AVISO_IMAGEN_SIN_TEXTO = "<!-- el OCR no encontró texto -->"
NOTA_TEXTO_DUDOSO = "<!-- texto de baja confianza -->"
NOTA_PAGINA_SIN_TEXTO = (
    "<!-- posible documento escaneado sin texto: revisar o reescanear -->"
)

# Un fallo de tesseract (no de timeout/I-O) que huele a archivo que no se
# pudo decodificar; solo se usa para el codigo, NUNCA se copia el stderr (trae
# rutas y contenido del archivo: vector de prompt injection).
_MARCAS_ARCHIVO_ILEGIBLE = (
    "pix not read", "pixReadStream", "findFileFormatStream",
    "Unsupported image type", "cannot be read", "image file not found",
    "pixReadMem",   # por stdin leptonica escribe `pixReadMem…`, no `pixReadStream`
    "pixRead",      # cualquier `pixRead…`, p. ej. `pixReadFromTiffStream`
    "is not uint",  # TIFF de coma flotante: `sample format = 3 is not uint`
)

# S-1: tesseract interpreta una entrada que no es imagen como LISTA DE RUTAS.
# Antes de llamarlo se exigen los bytes magicos de un formato de imagen
# (compuerta rutea por contenido, asi que vale cualquiera de ellos, no solo el
# de la extension), y la imagen viaja por STDIN, nunca por ruta.
_FIRMAS_IMAGEN = (
    b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"II*\x00", b"MM\x00*", b"BM",
    b"GIF87a", b"GIF89a",
)

# Tope de pixeles POR FOTOGRAMA, comprobado ANTES de decodificar. (El default
# de `Image.MAX_IMAGE_PIXELS` de Pillow es ~89 M y solo avisa; el error salta
# al doble, ~179 M: un tope propio mas bajo es el que realmente se alcanza.
# Pillow se deja con su default y su `DecompressionBombError` tambien se
# traduce a `demasiados_pixeles`, sin mutar el global.) La foto real mas
# grande de LACTOVI mide 13630x3826 (52 M).
MAX_PIXELES = 100_000_000

# Tope POR FOTOGRAMA mas bajo para los modos de 16 bits (I;16*). MEDIDO
# 2026-10-03 con el codigo ACTUAL (re-codificacion por rango real, sin
# normalizar; ru_maxrss del proceso que corre `extraer`, linea base 26 MB), un
# TIFF I;16 de 4000x4000 (16 Mpx) con texto:
#   una pagina, rango completo (x200): 184 MB; una pagina, valores 0-255: 184 MB;
#   dos paginas: 214 MB.
# (Las cifras de 603 MB a 25 Mpx y 398 MB a 16 Mpx de una ronda anterior eran de
# la normalizacion con percentiles, ya borrada: son una cota vieja y segura. El
# tope se deja en 16 Mpx; con ~184 MB habria margen para subirlo, pero eso es
# otra decision.)
MAX_PIXELES_NUMERICO = 16_000_000

# Tope por fotograma para los demas modos fuera de {1, L, LA, P, RGB, RGBA}
# (CMYK, LAB, YCbCr...): dos paginas de 10000x10000 en CMYK llegaron a 800 MB.
# MEDIDO 2026-10-03 (ru_maxrss, linea base 26 MB), con texto: CMYK de 25 Mpx
# (5000x5000) una pagina 222 MB, dos paginas (camino que convierte a RGB) 414 MB
# -> por debajo de ~600 MB, el tope queda en 25 Mpx.
MAX_PIXELES_OTROS_MODOS = 25_000_000

# Solo un TIFF es multipagina PARA TESSERACT (procesa todas sus paginas). Un
# GIF o WebP animado se RECHAZA (`animacion_no_soportada`: tesseract no lee
# animaciones, y leptonica decodifica todos los fotogramas antes de
# rechazarlas -- 8 de 15000x15000 llevaron a tesseract a 1,79 GB); un MPO
# (fotos de iPhone) se lee por su primer fotograma. Tope de paginas de un TIFF:
MAX_PAGINAS = 50

# Tope de pixeles de un TIFF sumando TODAS sus paginas, comprobado por el
# tamano de cada pagina ANTES de decodificar ninguna.
MAX_PIXELES_TOTAL = 300_000_000

# Plazo TOTAL del OCR de todas las paginas de un TIFF (ademas del timeout por
# llamada, que se acorta al tiempo que queda). `_reloj` se puede sustituir.
PLAZO_TOTAL_SEGUNDOS = 900
_reloj = time.monotonic

# (D) Tamano de pagina. NO se usa el metadato de DPI (el de un celular miente:
# "300 DPI" en una foto). Se mira la proporcion lado largo / lado corto, a
# +-TOLERANCIA_PROPORCION de A4 (297/210 = 1,4142) o carta (11/8,5 = 1,2941),
# y el DPI IMPLICITO (lado largo en px / lado largo de la pagina en pulgadas)
# entre DPI_MINIMO_PAGINA y DPI_MAXIMO_PAGINA. Oficio/legal (1,647) no se
# incluye: mas falsos positivos que beneficio. Medido sobre LACTOVI: las fotos
# de celular (1,333; a 0,039 de carta) y 5500x3830 (1,436; a 0,022 de A4 y
# ~470 DPI implicitos) quedan fuera; 2480x3508 (A4 a 300) cae.
PAGINAS_DE_REFERENCIA = (
    ("A4", 297 / 210, 11.69),
    ("carta", 11 / 8.5, 11.0),
)
TOLERANCIA_PROPORCION = 0.02
DPI_MINIMO_PAGINA = 150
DPI_MAXIMO_PAGINA = 400


def _version() -> str | None:
    """`None` cuando no se pudo determinar la versión -- NUNCA la cadena
    "desconocida" (Menor 10, final-hallazgos.md, ronda de cierre): esa
    cadena compara IGUAL A SÍ MISMA en dos fallos consecutivos, y
    `_version_vigente` la reenvía tal cual para invalidar el caché (I-2) --
    un extractor persistentemente incapaz de reportar su versión parecía
    "la misma versión de siempre" y el acierto de caché quedaba VÁLIDO
    justo cuando menos se podía confiar en él. `None` es el sentinel que
    `_ficha_de_cache_valida` ya trata como fallo de caché SIEMPRE. Los
    llamadores que necesitan un `str` no vacío para `Resultado.version`
    usan `_version() or "desconocida"`."""
    try:
        salida = subprocess.run(
            ["tesseract", "--version"], capture_output=True, text=True, timeout=30
        )
        return salida.stdout.splitlines()[0].split()[-1]
    except Exception:  # fail-soft: "tesseract --version" puede fallar (no instalado, timeout); se devuelve None (no determinable) en vez de propagar
        return None


def _como_texto(salida) -> str:
    """La salida de tesseract viaja en bytes (la imagen entra por stdin sin
    modo texto); se decodifica aqui, tolerando `str`/`None`."""
    if isinstance(salida, bytes):
        return salida.decode("utf8", errors="replace")
    return salida or ""


def tipo_por_cabecera(cabecera: bytes, sufijo: str = "") -> str | None:
    """UNICA fuente de verdad del tipo por CONTENIDO (la usan `ocr`,
    `compuerta` e `ingesta`): `"imagen"` si hay una firma de imagen valida
    (manda: un PNG con metadata `%PDF` es una imagen); `"pdf"` si el contenido
    EMPIEZA con `%PDF`, o si `%PDF` aparece desplazado (hasta 1024 bytes: el
    estandar tolera basura antes) Y la extension es `.pdf` -- un .txt, .csv o
    .eml que solo menciona `%PDF` no es un PDF; `None` si el contenido no
    decide. Un ZIP (`PK\\x03\\x04`: xlsx/docx) nunca es PDF."""
    if _tiene_firma_de_imagen(cabecera[:16]):
        return "imagen"
    if cabecera.startswith(b"PK\x03\x04"):
        return None
    if cabecera.startswith(_FIRMA_PDF):
        return "pdf"
    if sufijo.lower() == ".pdf" and _FIRMA_PDF in cabecera[:1024]:
        return "pdf"
    return None


def camino_de_tipo(tipo: str | None, sufijo: str) -> str:
    """`"pdf"` o `"imagen"` a partir del tipo por contenido; cuando el
    contenido no decide manda la EXTENSION (`.pdf` -> pdf), como la compuerta
    siempre hizo. DECISION DOCUMENTADA: un `Escanear 1.pdf` con 2000 bytes de
    basura antes de `%PDF` va por el camino pdf en las tres piezas."""
    if tipo == "pdf" or (tipo is None and sufijo.lower() == ".pdf"):
        return "pdf"
    return "imagen"


def camino_de(origen: Path, sufijo: str | None = None) -> str:
    """El camino (`"pdf"`/`"imagen"`) que toma el OCR para este archivo: la
    MISMA decision para `ocr.extraer`, la compuerta y la ingesta."""
    origen = Path(origen)
    try:
        with open(origen, "rb") as fh:
            cabecera = fh.read(1024)
    except OSError:
        cabecera = b""
    sufijo = sufijo if sufijo is not None else origen.suffix
    return camino_de_tipo(tipo_por_cabecera(cabecera, sufijo), sufijo)


def _tiene_firma_de_imagen(cabecera: bytes) -> bool:
    return cabecera.startswith(_FIRMAS_IMAGEN) or (
        cabecera[:4] == b"RIFF" and cabecera[8:12] == b"WEBP"
    )


def _gif_cuadros(datos: bytes, hasta: int = 2) -> int | None:
    """Cuenta los fotogramas de un GIF leyendo SOLO su estructura de bloques
    (sin decodificar nada: Pillow decodifica los fotogramas al contarlos y un
    GIF de 376 bytes llevo a 1,3 GB). Se detiene al llegar a `hasta`. `None`
    si la estructura esta rota."""
    n = len(datos)
    if n < 13:
        return None
    i = 13
    flags = datos[10]
    if flags & 0x80:
        i += 3 * (2 << (flags & 7))
    cuadros = 0
    while i < n:
        bloque = datos[i]
        i += 1
        if bloque == 0x3B:      # trailer
            return cuadros
        if bloque == 0x21:      # extension: etiqueta + sub-bloques
            i += 1
        elif bloque == 0x2C:    # descriptor de imagen
            if i + 9 > n:
                return None
            flags_img = datos[i + 8]
            i += 9
            if flags_img & 0x80:
                i += 3 * (2 << (flags_img & 7))
            i += 1              # tamano minimo de codigo LZW
            cuadros += 1
            if cuadros >= hasta:
                return cuadros
        else:
            return None
        while i < n and datos[i] != 0:      # sub-bloques hasta el terminador
            i += datos[i] + 1
        i += 1
    return None


def _webp_animado(datos: bytes) -> bool:
    """Bandera de animacion del chunk VP8X (bit 1 del byte de flags)."""
    return datos[12:16] == b"VP8X" and len(datos) > 20 and bool(datos[20] & 0x02)


def _validar_imagen(datos: bytes) -> tuple[str, list[tuple[int, int]]] | str:
    """Antes de OCR, sobre los BYTES (se leen una sola vez, MINOR-N2: nunca se
    relee la ruta, asi lo validado es lo que se procesa): firma magica (S-1),
    animaciones rechazadas leyendo solo su estructura (N8), y tamano de cada
    fotograma de un TIFF antes de decodificar ninguno (N13). Devuelve
    `(tipo, dimensiones)`: `"una"` (un fotograma, ya decodificado entero con
    Pillow -- MAJOR-2: un TIFF truncado da rc=0 y vacio en tesseract -- y los
    bytes originales son los que iran a tesseract) o `"tiff"` (varias paginas:
    se decodifican de a una en `_ocr_tiff`). O la CAUSA (`firma_invalida`,
    `no_decodifica`, `demasiados_pixeles`, `demasiadas_paginas`,
    `animacion_no_soportada`, `sin_pillow`): un codigo, nunca texto de la
    excepcion (puede traer rutas)."""
    if not _tiene_firma_de_imagen(datos[:16]):
        return "firma_invalida"
    if datos[:6] in (b"GIF87a", b"GIF89a"):
        cuadros = _gif_cuadros(datos)
        if cuadros is None:
            return "no_decodifica"
        if cuadros > 1:
            return "animacion_no_soportada"
    elif datos[:4] == b"RIFF" and _webp_animado(datos):
        return "animacion_no_soportada"
    try:
        from io import BytesIO

        from PIL import Image
    except ImportError:
        return "sin_pillow"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(BytesIO(datos)) as img:
                # Solo un TIFF es multipagina PARA TESSERACT. Un MPO (fotos de
                # iPhone: la foto + una vista previa) se lee por su primer
                # fotograma: tratarlo como paginas re-codificaria la foto
                # (medido en IMG_2353: otro resultado) y sumaria una vista
                # previa que tesseract nunca lee.
                n = getattr(img, "n_frames", 1) if img.format == "TIFF" else 1
                if n > MAX_PAGINAS:
                    return "demasiadas_paginas"
                dimensiones: list[tuple[int, int]] = []
                total = 0
                for i in range(n):
                    img.seek(i)
                    ancho, alto = img.size
                    total += ancho * alto
                    if img.mode in _MODOS_NO_SOPORTADOS:
                        return "modo_no_soportado"
                    if img.mode in _MODOS_16_BITS:
                        limite = MAX_PIXELES_NUMERICO
                    elif img.mode not in _MODOS_PNG:
                        limite = MAX_PIXELES_OTROS_MODOS
                    else:
                        limite = MAX_PIXELES
                    if ancho * alto > limite or total > MAX_PIXELES_TOTAL:
                        return "demasiados_pixeles"
                    dimensiones.append((ancho, alto))
                if n == 1:
                    img.load()
                    if img.mode in _MODOS_16_BITS:
                        # leptonica toma el byte alto: se re-codifica segun el
                        # rango real (`_a_modo_legible`), no van los originales
                        return "tiff", dimensiones
                    return "una", dimensiones
    except Image.DecompressionBombError:
        return "demasiados_pixeles"
    except MemoryError:
        return "sin_memoria"   # recursos, no archivo danado
    except Exception:  # fail-soft: Pillow lanza OSError/ValueError/SyntaxError/EOFError segun el formato roto; todas significan "no decodifica" y salen como Resultado(error, archivo_ilegible)
        return "no_decodifica"
    return "tiff", dimensiones


_MODOS_PNG = frozenset({"1", "L", "LA", "P", "RGB", "RGBA"})

# Rasters de coma flotante (F) o de 32 bits (I): no son documentos de
# expediente. `modo_no_soportado`, sin llamar a tesseract (leptonica sale con 0
# y un error en stderr, y cualquier escalado a 8 bits es una apuesta: un pixel
# nan o 1e30 o un poco de ruido lo aplastaba -- se borro).
_MODOS_NO_SOPORTADOS = frozenset({"F", "I"})

# 16 bits: leptonica los lee de forma NATIVA (medido), asi que van sin
# normalizar ni escalar; al re-codificar a PNG (camino multipagina) se guardan
# como PNG de 16 bits. PNG solo admite I;16 e I;16B: I;16L e I;16N se llevan a
# I;16B de forma EXPLICITA por sus bytes (`_a_modo_legible`). NO se convierten
# con `convert("I")`: Pillow convierte mal I;16N (extremos 0-255 sobre datos de
# 0-51000) y la pagina salia negra -> imagen_sin_texto en silencio.
_MODOS_16_BITS = frozenset({"I;16", "I;16B", "I;16L", "I;16N"})


def _a_modo_legible(img):
    """Fotograma decodificado -> uno que se pueda guardar como PNG para
    tesseract: los modos de `_MODOS_PNG` tal cual; el resto (CMYK, YCbCr, LAB,
    HSV...) a `RGB`; y los de 16 bits segun su RANGO REAL, sin escalar:

    - maximo <= 255 (lo que produce `convert("I").convert("I;16")` de una imagen
      de 8 bits): leptonica toma el byte ALTO de un PNG de 16 bits y la pagina
      sale NEGRA (ok/imagen_sin_texto en silencio). Se arma un `L` con los
      bytes BAJOS: los valores ya estan en 0-255. NO se usa `convert("L")`
      (Pillow escala y recorta) ni `>> 8` (tambien la dejaria negra).
    - maximo > 255 (12 bits, rango completo...): PNG de 16 bits, el camino
      nativo. I;16L e I;16N se llevan a I;16B por sus bytes (intercambio little
      -> big endian; I;16N es nativo y solo se intercambia en un host
      little-endian). NO se convierten con `convert("I")`: Pillow convierte
      mal I;16N.
    """
    if img.mode in _MODOS_16_BITS:
        from PIL import Image

        fuente_little = img.mode in ("I;16", "I;16L") or (
            img.mode == "I;16N" and sys.byteorder == "little")
        valores = array.array("H", img.tobytes())          # nativo
        if fuente_little != (sys.byteorder == "little"):
            valores.byteswap()                              # ahora son valores nativos
        if max(valores, default=0) <= 255:
            crudo = valores.tobytes()
            bajos = crudo[0::2] if sys.byteorder == "little" else crudo[1::2]
            return Image.frombytes("L", img.size, bajos)
        if img.mode in ("I;16", "I;16B"):
            return img
        if sys.byteorder == "little":
            valores.byteswap()                              # a big-endian
        return Image.frombytes("I;16B", img.size, valores.tobytes())
    if img.mode in _MODOS_PNG:
        return img
    return img.convert("RGB")


def _a_png(cuadro) -> bytes:
    from io import BytesIO

    salida = BytesIO()
    cuadro.save(salida, format="PNG")
    return salida.getvalue()


def _implica_pagina(ancho: int, alto: int) -> str | None:
    """Nombre de la pagina de referencia (A4/carta) si la imagen tiene
    tamano de pagina por proporcion y DPI implicito; `None` si no. Pura."""
    largo, corto = max(ancho, alto), min(ancho, alto)
    if corto <= 0:
        return None
    proporcion = largo / corto
    for nombre, referencia, pulgadas in PAGINAS_DE_REFERENCIA:
        if abs(proporcion - referencia) <= TOLERANCIA_PROPORCION:
            dpi = largo / pulgadas
            if DPI_MINIMO_PAGINA <= dpi <= DPI_MAXIMO_PAGINA:
                return nombre
    return None


def _analizar_tsv(salida_tsv: str) -> dict:
    """Función PURA (sin subprocesos) sobre la salida `tsv` de tesseract --
    misma pasada de reconocimiento que el modo texto plano, sólo en otro
    formato de salida (no es una segunda cuenta que pueda divergir, el
    defecto C-2 de `pdf.py`). Devuelve cuántas palabras reconoció, la
    confianza promedio sobre TODAS ellas (nunca una muestra), cuáles están
    por debajo de `CONFIANZA_MINIMA_PALABRA` (nombradas, no descartadas), y
    el ancho/alto de la página (fila de nivel 1 del tsv).

    Separada de `_ocr_una_imagen` (que sí corre el subproceso) justo para
    poder probarse con un `tsv` armado a mano, sin depender de que
    tesseract reconozca algo en particular -- mismo patrón que
    `pdf._tabla_a_bloque`, una función pura con sus propios tests directos.
    """
    ancho = alto = 0
    confianzas: list[float] = []
    dudosas: list[dict] = []
    lineas = salida_tsv.splitlines()
    for linea in lineas[1:]:  # lineas[0] es el encabezado de columnas
        campos = linea.split("\t")
        if len(campos) < 12:
            continue
        if campos[0] == "1":  # fila de nivel "página": trae ancho/alto
            try:
                ancho, alto = int(campos[8]), int(campos[9])
            except ValueError:  # fail-soft: las columnas de ancho/alto del tsv pueden venir no numericas; se dejan en 0 (solo informativas) y sigue el analisis del resto de la fila
                pass
            continue
        conf_str, texto_palabra = campos[10], campos[11]
        if not texto_palabra.strip():
            continue
        try:
            conf = float(conf_str)
        except ValueError:
            continue
        if conf < 0:  # -1: fila estructural (bloque/párrafo/línea), no palabra
            continue
        confianzas.append(conf)
        if conf < CONFIANZA_MINIMA_PALABRA:
            dudosas.append({"palabra": texto_palabra, "confianza": round(conf, 2)})

    n_palabras = len(confianzas)
    confianza_promedio = sum(confianzas) / n_palabras if n_palabras else 0.0
    return {
        "n_palabras": n_palabras,
        "confianza_promedio": round(confianza_promedio, 2),
        "palabras_dudosas": dudosas,
        "ancho": ancho,
        "alto": alto,
    }


def _clasificar(caracteres: int, analisis: dict) -> str:
    """Una de 'sin_texto' / 'con_dudas' / 'ok' -- la MISMA función para una
    imagen suelta y para cada página de un PDF rasterizado (evita repetir
    la regla en dos lugares que puedan divergir). 'sin_texto' cubre menos de
    MINIMO_CARACTERES Y mayoria de palabras dudosas; `_resolver_imagen` las
    separa (A / D / B) y el PDF cuenta ambas como pagina sin texto."""
    if caracteres < MINIMO_CARACTERES:
        return "sin_texto"
    n = analisis["n_palabras"]
    dudosas = analisis["palabras_dudosas"]
    if n and len(dudosas) / n > PROPORCION_MAXIMA_PALABRAS_DUDOSAS:
        return "sin_texto"
    if n < MINIMO_PALABRAS or dudosas:
        return "con_dudas"
    return "ok"


class _Presupuesto:
    """Un solo presupuesto de tiempo para todo el OCR de una imagen: se
    descuenta en CADA llamada a tesseract, leyendo `_reloj` justo antes."""

    def __init__(self, total: float):
        self.total = total
        self.inicio = _reloj()

    def restante(self) -> float:
        return self.total - (_reloj() - self.inicio)


def _ilegible_dict(causa: str) -> dict:
    return {"clasificacion": "ilegible", "causa": causa}


def _correr_tesseract(cmd: list, datos: bytes, presupuesto: _Presupuesto | None):
    """Una llamada a tesseract. Devuelve el proceso, o `None` si no pudo correr,
    o un dict `ilegible` si se agoto el tiempo: `tiempo_excedido` (el
    presupuesto total ya se consumio) o `tiempo_por_llamada` (el timeout de la
    llamada salto con presupuesto todavia disponible). Sin presupuesto (paginas
    de PDF) el timeout por llamada de siempre y `None` si falla."""
    if presupuesto is None:
        timeout = TIMEOUT_SEGUNDOS
    else:
        restante = presupuesto.restante()
        if restante <= 0:
            return _ilegible_dict("tiempo_excedido")
        timeout = min(TIMEOUT_SEGUNDOS, restante)
    try:
        return subprocess.run(cmd, input=datos, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        if presupuesto is None:
            return None
        return _ilegible_dict(
            "tiempo_excedido" if presupuesto.restante() <= 0 else "tiempo_por_llamada"
        )
    except Exception:  # fail-soft: el subproceso de tesseract puede fallar (I/O); se devuelve None y el llamador lo convierte en Resultado(estado="error")
        return None


def _ocr_bytes(datos: bytes, idioma: str, presupuesto: _Presupuesto | None = None) -> dict | None:
    """Corre tesseract DOS veces sobre la MISMA imagen -- texto plano (para
    el extracto exacto, tildes y guion largo incluidos) y `tsv` (para la
    confianza por palabra, que el modo texto plano no expone). `None` si el
    propio subproceso de tesseract no pudo correr sobre esta imagen
    (I/O, `returncode` distinto de cero) -- fallo cerrado del llamador, nunca
    una excepción escapando de acá. Si tesseract no pudo DECODIFICAR el
    archivo (imagen danada) o se agoto el tiempo, devuelve un dict con
    `clasificacion="ilegible"` y su `causa` en vez de `None`, para que
    `_resolver_imagen` pueda distinguirlo con su codigo."""
    proceso = _correr_tesseract(["tesseract", "-", "stdout", "-l", idioma], datos, presupuesto)
    if proceso is None or isinstance(proceso, dict):
        return proceso
    if proceso.returncode != 0:
        stderr = _como_texto(proceso.stderr)
        if any(marca in stderr for marca in _MARCAS_ARCHIVO_ILEGIBLE):
            return _ilegible_dict("tesseract_no_lee")
        return None
    # rc=0 NO basta: con un TIFF de coma flotante leptonica escribe el error en
    # stderr, sale con 0 y no entrega texto -- un archivo que nadie leyo.
    if any(marca in _como_texto(proceso.stderr) for marca in _MARCAS_ARCHIVO_ILEGIBLE):
        return _ilegible_dict("tesseract_no_lee")
    texto = _como_texto(proceso.stdout).strip()

    proceso_tsv = _correr_tesseract(
        ["tesseract", "-", "stdout", "-l", idioma, "tsv"], datos, presupuesto
    )
    if proceso_tsv is None or isinstance(proceso_tsv, dict):
        return proceso_tsv
    analisis = _analizar_tsv(_como_texto(proceso_tsv.stdout))

    return {
        "texto": texto,
        "caracteres": len(texto),
        **analisis,
        "clasificacion": _clasificar(len(texto), analisis),
    }


def _ocr_una_imagen(ruta: Path, idioma: str) -> dict | None:
    """Camino de las paginas rasterizadas de un PDF (PNG propios, en un
    directorio temporal nuestro): lee el archivo y lo manda por stdin."""
    try:
        datos = Path(ruta).read_bytes()
    except OSError:
        return None
    return _ocr_bytes(datos, idioma)


def _ocr_imagen(datos: bytes, tipo: str, dimensiones: list, idioma: str) -> dict | None:
    """OCR de una imagen ya validada. `"una"`: los bytes originales por stdin.
    `"tiff"`: pagina por pagina (decodificar -> PNG -> tesseract -> descartar:
    nunca se guarda la lista de PNG), con el plazo total `PLAZO_TOTAL_SEGUNDOS`
    y las metricas de la regla A/B sobre el TOTAL (caracteres y palabras
    sumados, confianza ponderada por palabras)."""
    presupuesto = _Presupuesto(PLAZO_TOTAL_SEGUNDOS)
    if tipo == "una":
        return _ocr_bytes(datos, idioma, presupuesto)

    from io import BytesIO

    from PIL import Image

    partes, dudosas = [], []
    caracteres = palabras = 0
    suma_conf = 0.0
    primero = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            img = Image.open(BytesIO(datos))
        except Exception:  # fail-soft: ya se abrio una vez en la validacion; si ahora falla, no decodifica
            return {"clasificacion": "ilegible", "causa": "no_decodifica"}
        with img:
            for numero in range(1, len(dimensiones) + 1):
                if presupuesto.restante() <= 0:
                    return _ilegible_dict("tiempo_excedido")
                try:
                    img.seek(numero - 1)
                    img.load()
                    cuadro = _a_modo_legible(img)
                    png = _a_png(cuadro)
                except MemoryError:
                    return _ilegible_dict("sin_memoria")   # recursos, no archivo danado
                except Exception:  # fail-soft: pagina truncada o corrupta = archivo que no decodifica
                    return _ilegible_dict("no_decodifica")
                r = _ocr_bytes(png, idioma, presupuesto)
                del png, cuadro
                if r is None:
                    return None
                if r["clasificacion"] == "ilegible":
                    return r
                primero = primero or r
                if r["texto"]:
                    partes.append(f"<!-- página {numero} -->\n{r['texto']}")
                caracteres += r["caracteres"]
                palabras += r["n_palabras"]
                suma_conf += r["confianza_promedio"] * r["n_palabras"]
                dudosas += [{"pagina": numero, **d} for d in r["palabras_dudosas"]]
    analisis = {
        "n_palabras": palabras,
        "confianza_promedio": round(suma_conf / palabras, 2) if palabras else 0.0,
        "palabras_dudosas": dudosas,
        "ancho": primero["ancho"], "alto": primero["alto"],
    }
    return {
        "texto": "\n\n".join(partes),
        "caracteres": caracteres,
        **analisis,
        "clasificacion": _clasificar(caracteres, analisis),
    }


def _rasterizar_pdf(origen: Path, destino: Path) -> list[Path] | None:
    """`pdftoppm` (poppler, ya instalado -- misma suite que `pdftotext`) una
    página por PNG. `None` si `pdftoppm` no corrió (no instalado, PDF
    inválido, timeout)."""
    prefijo = destino / "pagina"
    try:
        proceso = subprocess.run(
            ["pdftoppm", "-png", "-r", str(DPI_RASTERIZADO), str(origen), str(prefijo)],
            capture_output=True, text=True, timeout=TIMEOUT_SEGUNDOS,
        )
    except Exception:  # fail-soft: pdftoppm puede fallar rasterizando el PDF (timeout, PDF invalido); se devuelve None y el llamador lo convierte en Resultado(estado="error")
        return None
    if proceso.returncode != 0:
        return None
    # pdftoppm rellena con ceros segun la cantidad total de paginas (2
    # digitos hasta 99, 3 hasta 999, ...) -- el orden lexicografico del glob
    # ya es correcto para ese rango, pero se ordena por el numero real
    # extraido del nombre para no depender de esa convencion.
    paginas = list(destino.glob("pagina-*.png"))
    return sorted(paginas, key=lambda p: int("".join(c for c in p.stem if c.isdigit())))


def _detalle_comun(idioma: str, r: dict) -> dict:
    detalle = {
        "idioma": idioma,
        "caracteres": r["caracteres"],
        "palabras_totales": r["n_palabras"],
        "confianza_promedio": r["confianza_promedio"],
        "ancho": r["ancho"],
        "alto": r["alto"],
    }
    if r["palabras_dudosas"]:
        detalle["palabras_dudosas"] = r["palabras_dudosas"]
    return detalle


# Que codigo lleva cada causa. `archivo_ilegible` ("danado") SOLO si el archivo
# esta roto: firma que no es de imagen o que Pillow no decodifica (incluye los
# truncados). Un archivo SANO de un formato que no leemos es
# `formato_no_soportado`; esto incluye `tesseract_no_lee`: Pillow ya lo
# decodifico en la validacion, asi que si leptonica lo rechaza (BMP RGB565...)
# el archivo no esta danado. Los topes de tamano son `imagen_demasiado_grande`.
_DETALLE_FORMATO_NO_SOPORTADO = {
    "modo_no_soportado": "modo de color F o I de 32 bits",
    "animacion_no_soportada": "GIF o WebP animado",
    "tesseract_no_lee": "leptonica no lo lee",
}
_DETALLE_DEMASIADO_GRANDE = {
    "demasiados_pixeles": "demasiados pixeles",
    "demasiadas_paginas": "demasiadas paginas",
}


def _ilegible(causa: str, idioma: str) -> Resultado:
    if causa in ("tiempo_excedido", "tiempo_por_llamada"):
        razon, codigo = "se excedio el tiempo de OCR de la imagen", CODIGO_OCR_TIEMPO_EXCEDIDO
    elif causa == "sin_memoria":
        razon, codigo = "memoria insuficiente para procesar la imagen", CODIGO_OCR_SIN_MEMORIA
    elif causa in _DETALLE_FORMATO_NO_SOPORTADO:
        razon = f"formato de imagen no soportado ({_DETALLE_FORMATO_NO_SOPORTADO[causa]})"
        codigo = CODIGO_FORMATO_NO_SOPORTADO
    elif causa in _DETALLE_DEMASIADO_GRANDE:
        razon = f"imagen demasiado grande ({_DETALLE_DEMASIADO_GRANDE[causa]})"
        codigo = CODIGO_IMAGEN_DEMASIADO_GRANDE
    else:
        razon, codigo = (
            "no se pudo abrir ni decodificar la imagen (archivo danado)",
            CODIGO_ARCHIVO_ILEGIBLE,
        )
    return Resultado(
        estado="error", salidas={}, extractor=EXTRACTOR,
        version=_version() or "desconocida",
        detalle={
            "razon": razon, "codigo": codigo, "causa": causa, "idioma": idioma,
            "_camino": "imagen",
        },
    )


def _resolver_imagen(
    r: dict, idioma: str, paginas: list[tuple[int, int]] | None = None
) -> Resultado:
    if r["clasificacion"] == "ilegible":
        return _ilegible(r["causa"], idioma)

    detalle = _detalle_comun(idioma, r)
    detalle["_camino"] = "imagen"
    if paginas:
        detalle["ancho"], detalle["alto"] = paginas[0]
        if len(paginas) > 1:
            detalle["paginas"] = len(paginas)

    if r["clasificacion"] == "sin_texto":
        # DECISION DEL CONTROLADOR (Jax#338 ronda 2; Fernando puede cambiarla):
        # cuando A y B coinciden (menos de MINIMO_CARACTERES con mayoria de
        # palabras dudosas) se aplica A -- o D si es una pagina --. B exige al
        # menos MINIMO_CARACTERES: con una o dos palabras no hay texto que
        # conservar.
        if r["caracteres"] >= MINIMO_CARACTERES:
            # (B) mayoria de palabras dudosas: el texto leido SE CONSERVA.
            detalle["razon"] = (
                "texto de baja confianza: mas de la mitad de las palabras "
                "reconocidas tienen confianza baja (probable ruido o "
                "desenfoque) -- ver palabras_dudosas"
            )
            detalle["codigo"] = CODIGO_IMAGEN_TEXTO_DUDOSO
            return Resultado(
                estado="parcial",
                salidas={"texto.txt": f"{NOTA_TEXTO_DUDOSO}\n{r['texto']}"},
                extractor=EXTRACTOR, version=_version() or "desconocida",
                detalle=detalle,
            )
        detalle["razon"] = "el OCR no devolvio texto util"
        # D usa las dimensiones del fotograma 0 y solo aplica si TODOS los
        # fotogramas tienen tamano de pagina (un TIFF con una foto dentro no
        # es un escaneo).
        pagina = None
        if paginas:
            nombres = [_implica_pagina(ancho, alto) for ancho, alto in paginas]
            if all(nombres):
                pagina = nombres[0]
        if pagina is not None:
            # (D) tamano de pagina y sin texto: posible escaneo guardado como
            # imagen -- un documento sin texto SI es un problema.
            detalle["codigo"] = CODIGO_IMAGEN_PAGINA_SIN_TEXTO
            detalle["pagina_de_referencia"] = pagina
            detalle["razon"] = (
                "posible documento escaneado sin texto: revisar o reescanear"
            )
            return Resultado(
                estado="parcial",
                salidas={"texto.txt": f"{AVISO_IMAGEN_SIN_TEXTO}\n{NOTA_PAGINA_SIN_TEXTO}"},
                extractor=EXTRACTOR, version=_version() or "desconocida",
                detalle=detalle,
            )
        detalle["codigo"] = CODIGO_IMAGEN_SIN_TEXTO
        return Resultado(
            estado="ok", salidas={"texto.txt": AVISO_IMAGEN_SIN_TEXTO},
            extractor=EXTRACTOR, version=_version() or "desconocida",
            detalle=detalle,
        )

    if r["clasificacion"] == "con_dudas":
        razones = []
        if r["n_palabras"] < MINIMO_PALABRAS:
            razones.append(f"cobertura insuficiente (menos de {MINIMO_PALABRAS} palabras)")
        if r["palabras_dudosas"]:
            razones.append("hay palabras con confianza baja -- ver palabras_dudosas")
        detalle["razon"] = "; ".join(razones)
        return Resultado(
            estado="parcial", salidas={"texto.txt": r["texto"]},
            extractor=EXTRACTOR, version=_version() or "desconocida", detalle=detalle,
        )

    return Resultado(
        estado="ok", salidas={"texto.txt": r["texto"]},
        extractor=EXTRACTOR, version=_version() or "desconocida", detalle=detalle,
    )


def _resolver_pdf(resultados: list[dict | None], idioma: str) -> Resultado:
    total = len(resultados)
    paginas_sin_texto: list[int] = []
    paginas_con_dudas: list[int] = []
    palabras_dudosas: list[dict] = []
    partes_texto: list[str] = []
    # I-3 (final-hallazgos.md, ronda de cierre): la confianza medida se
    # registra SIEMPRE (spec §9-bis), pase o no pase -- justo para poder
    # calibrar el umbral. `_resolver_imagen`/`_detalle_comun` ya lo hacían
    # para una imagen suelta; este camino (PDF escaneado, el 100% de la
    # muestra `parcial` medida) armaba su propio `detalle` a mano y NUNCA
    # la incluía. Ponderada por palabras (no un promedio de promedios de
    # página): una página con 200 palabras pesa más que una de 3.
    # Claves STRING, no int: `Ficha` exige que `detalle` sobreviva un viaje
    # real a JSON y vuelta sin cambios (ficha.py, I-6 de su propia ronda) --
    # JSON no admite claves que no sean string, así que un dict con claves
    # `int` acá haría que CUALQUIER ingesta de un PDF escaneado con más de
    # una página reviente `Ficha(...)` con un `ValueError` en cuanto
    # `ingesta.ingerir()` intentara construir la ficha. Detectado corriendo
    # el mismo viaje a mano contra este cambio, no supuesto.
    confianza_por_pagina: dict[str, float] = {}
    suma_confianza_ponderada = 0.0
    palabras_totales = 0

    for numero, r in enumerate(resultados, start=1):
        if r is None or r["clasificacion"] in {"sin_texto", "ilegible"}:
            paginas_sin_texto.append(numero)
            continue
        partes_texto.append(f"<!-- página {numero} -->\n{r['texto']}")
        if r["clasificacion"] == "con_dudas":
            paginas_con_dudas.append(numero)
        for dudosa in r["palabras_dudosas"]:
            palabras_dudosas.append({"pagina": numero, **dudosa})
        confianza_por_pagina[str(numero)] = r["confianza_promedio"]
        suma_confianza_ponderada += r["confianza_promedio"] * r["n_palabras"]
        palabras_totales += r["n_palabras"]

    if len(paginas_sin_texto) == total:
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR,
            version=_version() or "desconocida",
            detalle={
                "razon": "ninguna pagina del PDF dio texto util via OCR",
                "idioma": idioma,
                "paginas": total,
                "_camino": "pdf",
            },
        )

    contenido = "\n\n".join(partes_texto).strip()
    confianza_promedio = (
        round(suma_confianza_ponderada / palabras_totales, 2) if palabras_totales else 0.0
    )
    detalle = {
        "idioma": idioma, "paginas": total, "confianza_promedio": confianza_promedio,
        "_camino": "pdf",
    }
    if confianza_por_pagina:
        detalle["confianza_por_pagina"] = confianza_por_pagina
    if paginas_sin_texto:
        detalle["paginas_sin_texto"] = paginas_sin_texto
    if paginas_con_dudas:
        detalle["paginas_con_dudas"] = paginas_con_dudas
    if palabras_dudosas:
        detalle["palabras_dudosas"] = palabras_dudosas

    estado = "parcial" if (paginas_sin_texto or paginas_con_dudas or palabras_dudosas) else "ok"
    return Resultado(
        estado=estado, salidas={"texto.txt": contenido},
        extractor=EXTRACTOR, version=_version() or "desconocida", detalle=detalle,
    )


def extraer(origen: Path, idioma: str = "spa", camino: str | None = None) -> Resultado:
    if shutil.which("tesseract") is None:
        # I-8 (final-hallazgos.md, ronda de cierre): antes era 'error',
        # indistinguible de "este documento es ilegible" -- el spec §8
        # mapea "sin extractor disponible" a 'sin_extractor', igual que
        # `openpyxl`/`python-docx`/`pdfplumber` ausentes. Sin esto, una
        # máquina sin tesseract marca TODOS sus escaneos como 'error' y
        # quien triagea por estado confunde un problema de despliegue con
        # documentos malos.
        return Resultado(
            estado="sin_extractor", salidas={}, extractor=EXTRACTOR, version="ausente",
            detalle={"razon": "tesseract no esta instalado"},
        )

    # MAJOR 2 (final-hallazgos.md, adenda 2026-09-21, ruling de Fernando):
    # el cuerpo entero corre protegido -- I-7 agregó `mkdir()` y
    # `TemporaryDirectory(dir=...)` SIN guarda, en una función que no
    # tenía `try` propio (los tres hermanos sí: D-1 en `word.py`, I-5 en
    # `excel.py`, el `try` de `extraer()` en `pdf.py`). Con el workspace
    # no escribible (disco lleno, remontado sólo-lectura, cuota excedida
    # -- los tres disparadores reales del camino degradado) salía un
    # `PermissionError`/`OSError` CRUDO de `ocr.extraer`, de
    # `compuerta.extraer` -- cuyo contrato dice lo contrario -- y de
    # `ingerir()`, que no lo atrapa. Cuarta aparición de la familia I-5,
    # introducida por la MISMA ronda que vino a cerrarla: un arreglo es
    # código nuevo, y el código nuevo entra con los mismos defectos que el
    # viejo si no se lo revisa igual.
    try:
        origen = Path(origen)
        if not origen.is_file():
            return Resultado(
                estado="error", salidas={}, extractor=EXTRACTOR,
                version=_version() or "desconocida",
                detalle={"razon": f"no existe el archivo: {origen}"},
            )

        # El camino lo decide UNA funcion (`camino_de`); la compuerta pasa el suyo.
        if camino is None:
            camino = camino_de(origen)
        if camino == "pdf":
            if shutil.which("pdftoppm") is None:
                return Resultado(
                    estado="error", salidas={}, extractor=EXTRACTOR,
                    version=_version() or "desconocida",
                    detalle={
                        "razon": "pdftoppm no esta instalado (poppler-utils); "
                        "no se puede rasterizar el PDF para OCR",
                    },
                )
            # I-7 (final-hallazgos.md, ronda de cierre) -- "ahora" de la
            # reserva a medias a propósito: el directorio del rasterizado
            # va BAJO JAX_WORKSPACE_DIR, no al default de `tempfile`
            # (`/tmp`, que en hall9000 es tmpfs -- RAM -- en un
            # hipervisor con dos VMs). Se lee la env var directo (mismo
            # patrón que los otros call sites de esta variable:
            # motor_registry/tool_authority.py, jacobs/executor.py) en vez de importar el
            # módulo pesado de tool_authority sólo para esto -- es el
            # cambio de una línea que pide el ruling, no una migración de
            # dependencias. Diferido a fase 2, con su razón escrita:
            # rasterizar página por página con timeout por página, para
            # que un vencimiento deje 'parcial' con lo que alcanzó en vez
            # de perder TODO callando el parcial (hoy: un solo
            # `pdftoppm` para el documento entero) -- es un cambio de
            # estructura, y ésta es una ronda de cierre, no de rediseño.
            _WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="ocr-pdf-", dir=_WORKSPACE_DIR) as tmp:
                paginas = _rasterizar_pdf(origen, Path(tmp))
                if not paginas:
                    return Resultado(
                        estado="error", salidas={}, extractor=EXTRACTOR,
                        version=_version() or "desconocida",
                        detalle={"razon": "no se pudo rasterizar el PDF con pdftoppm"},
                    )
                resultados = [_ocr_una_imagen(pagina, idioma) for pagina in paginas]
            return _resolver_pdf(resultados, idioma)

        # MINOR-N2: UNA sola lectura; lo validado es lo que se procesa.
        datos = origen.read_bytes()
        validacion = _validar_imagen(datos)
        if validacion == "sin_pillow":
            return Resultado(
                estado="sin_extractor", salidas={}, extractor=EXTRACTOR, version="ausente",
                detalle={"razon": "Pillow no esta instalado; no se puede validar la imagen"},
            )
        if isinstance(validacion, str):
            return _ilegible(validacion, idioma)
        tipo, dimensiones = validacion
        resultado_img = _ocr_imagen(datos, tipo, dimensiones, idioma)
        if resultado_img is None:
            return Resultado(
                estado="error", salidas={}, extractor=EXTRACTOR,
                version=_version() or "desconocida",
                detalle={"razon": "no se pudo correr tesseract sobre la imagen"},
            )
        return _resolver_imagen(resultado_img, idioma, dimensiones)
    except MemoryError:
        return _ilegible("sin_memoria", idioma)   # recursos, no archivo danado
    except Exception as exc:  # fail-soft: cualquier fallo inesperado (permisos, disco lleno, workspace remontado solo-lectura) sale como Resultado(estado="error"), nunca una excepcion cruda -- mismo tratamiento que D-1/I-5 en los hermanos
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR,
            version=_version() or "desconocida",
            detalle={"razon": f"fallo inesperado en OCR: {type(exc).__name__}"},
        )
