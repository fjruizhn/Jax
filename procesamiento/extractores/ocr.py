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
  `parcial`. Si TODAS las páginas de un PDF quedan sin texto útil, el PDF se
  trata igual que una imagen (decisión de Fernando, 2026-10-04; antes era
  `error`): `parcial` con `imagen_texto_dudoso` (conserva el texto de las
  páginas B), con `imagen_pagina_sin_texto` (todas A, de tamaño de página) o
  `ok` + `imagen_sin_texto` (todas A, sin él); `error` si alguna página es
  ilegible (`None`, `ilegible` o un PNG que no se puede medir). El promedio se sigue registrando SIEMPRE, como señal
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

import shutil
import struct
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

from procesamiento.resultado import Resultado
# Jax#338 ronda 18: las firmas y el tipo por contenido viven en un modulo puro
# (sin Pillow ni JAX_WORKSPACE_DIR) para que el freno de dependencias decida con
# la MISMA funcion; los nombres de siempre siguen disponibles en `ocr`.
from procesamiento.tipos_imagen import tiene_firma_de_imagen as _tiene_firma_de_imagen
from procesamiento.tipos_imagen import leer_cabecera, tipo_por_cabecera

try:
    # Bare primero, por consistencia con los otros symlinks de las_manos/ (via
    # las_manos/workspace_dir.py). En produccion jax.core TAMBIEN es importable
    # (drop-in z-pythonpath.conf, PYTHONPATH=/srv/jax-prod/jax); el calificado
    # cubre el REPL/CI con solo la raiz del repo en sys.path.
    from workspace_dir import workspace_dir
except ImportError:
    from jax.core.workspace_dir import workspace_dir

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
    return workspace_dir()


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
# `paginas_sin_texto` (si todas lo son, el PDF se trata como una imagen: `parcial`
# con `imagen_texto_dudoso` o `imagen_pagina_sin_texto`, ver `_resolver_pdf`).
PROPORCION_MAXIMA_PALABRAS_DUDOSAS = 0.5

TIMEOUT_SEGUNDOS = 300

# DPI de rasterizado para OCR sobre PDF. 300 es el mínimo usual recomendado
# para OCR de documentos escaneados (por debajo, tesseract pierde exactitud
# en fuentes pequeñas de recibos/facturas).
DPI_RASTERIZADO = 300

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
#   formato que no leemos (detectado ANTES de tesseract; `detalle.formato`:
#   gif_animado, webp_animado, png_animado, gris_16_bits, coma_flotante, entero_32_bits,
#   bmp_16_bits)                           -> error   formato_no_soportado
#   Pillow la decodifica y leptonica no     -> error   archivo_no_procesable
#   demasiados pixeles / paginas             -> error   imagen_demasiado_grande
CODIGO_IMAGEN_SIN_TEXTO = "imagen_sin_texto"
CODIGO_IMAGEN_TEXTO_DUDOSO = "imagen_texto_dudoso"
CODIGO_IMAGEN_PAGINA_SIN_TEXTO = "imagen_pagina_sin_texto"
CODIGO_ARCHIVO_ILEGIBLE = "archivo_ilegible"
CODIGO_OCR_TIEMPO_EXCEDIDO = "ocr_tiempo_excedido"
CODIGO_FORMATO_NO_SOPORTADO = "formato_no_soportado"   # detectado antes de tesseract; lleva `formato`
CODIGO_ARCHIVO_NO_PROCESABLE = "archivo_no_procesable"   # Pillow lo decodifica y leptonica no
CODIGO_IMAGEN_DEMASIADO_GRANDE = "imagen_demasiado_grande"
CODIGO_OCR_SIN_MEMORIA = "ocr_sin_memoria"   # recursos, no archivo danado

# Version de la LOGICA de clasificacion de este extractor. `extractor_version`
# es la de tesseract y NO cambia cuando cambia una regla: esta cadena se guarda
# en la ficha (`detalle["_version_logica"]`) y `ingesta` la compara -- una
# ficha escrita con otra logica ni se reusa de cache ni cuenta como intento
# previo del tope D-2. SUBIRLA cada vez que cambie la regla.
# La version DEPENDE DEL CAMINO (`version_logica(camino)`): la de las IMAGENES
# (A/B/D, codigos nuevos, TIFF multipagina) y la del PDF escaneado, que desde
# 2026-10-04 (VERSION_LOGICA_PDF "1") trata un PDF sin texto util como una imagen
# -- una ficha de PDF escrita antes (sin marca) ni se reusa ni cuenta para D-2.
# "2": regla de Fernando de la ronda 1 de Jax#338.
# "3": Jax#338 ronda 13 -- la transparencia real se aplana sobre blanco Y sobre
# negro y el resultado es la UNION de las dos pasadas (con duplicado por texto
# y caja, y el tope de texto dudoso); una ficha escrita con "2" (la seleccion,
# que podia perder texto) no se reusa.
# "4": Jax#338 ronda 17 -- las invariantes de la ronda 16 (una imagen con
# transparencia real nunca sale `ok`; ninguna linea del texto plano se pierde)
# cambian la clasificacion de las fichas escritas con "3".
# "5": Jax#338 ronda 18 -- un PNG/APNG de mas de un cuadro es
# formato_no_soportado:png_animado (una ficha "4" lo daba ok/imagen_sin_texto).
# "6": Jax#338 ronda 19 -- un MPO de varios cuadros se lee cuadro por cuadro
# (una ficha "5" solo tenia el texto del primer cuadro).
VERSION_LOGICA_IMAGEN = "6"
# "1": decision de Fernando 2026-10-04 -- un PDF escaneado sin texto util deja de
# ser `error` y pasa a `parcial` (imagen_texto_dudoso / imagen_pagina_sin_texto).
# Una ficha de PDF sin marca (None != "1") es la logica vieja.
VERSION_LOGICA_PDF = "1"


def version_logica(camino: str) -> str | None:
    """Version de la logica para un camino (`"imagen"` o `"pdf"`, el que decide
    el CONTENIDO del archivo, no su extension)."""
    return VERSION_LOGICA_IMAGEN if camino == "imagen" else VERSION_LOGICA_PDF


# `Resultado` exige al menos una salida con contenido para `ok`: la unica
# salida del caso A es este aviso (nunca texto inventado). Regla (C) de
# Fernando: dice SOLO que el OCR no encontro texto.
AVISO_IMAGEN_SIN_TEXTO = "<!-- el OCR no encontró texto -->"
NOTA_TEXTO_DUDOSO = "<!-- texto de baja confianza -->"
RAZON_IMAGEN_TRANSPARENTE = (
    "imagen con transparencia: texto combinado de dos lecturas (fondo blanco "
    "y negro); revisar"
)
RAZON_MAYORIA_DUDOSA = (
    "texto de baja confianza: mas de la mitad de las palabras reconocidas "
    "tienen confianza baja (probable ruido o desenfoque) -- ver palabras_dudosas"
)
RAZON_TEXTO_SIN_POSICION = (
    "el OCR devolvió texto sin datos de posición; se conserva sin verificar"
)
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
# (`tipos_imagen.FIRMAS_IMAGEN`), y la imagen viaja por STDIN, nunca por ruta.

# Tope de pixeles POR FOTOGRAMA, comprobado ANTES de decodificar. (El default
# de `Image.MAX_IMAGE_PIXELS` de Pillow es ~89 M y solo avisa; el error salta
# al doble, ~179 M: un tope propio mas bajo es el que realmente se alcanza.
# Pillow se deja con su default y su `DecompressionBombError` tambien se
# traduce a `demasiados_pixeles`, sin mutar el global.) La foto real mas
# grande de LACTOVI mide 13630x3826 (52 M).
MAX_PIXELES = 100_000_000

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
    # Jax#338 ronda 19: sin bloquear (un `open()` de un FIFO espera un escritor);
    # lo que no es un archivo regular no da tipo por contenido y decide la extension.
    cabecera = leer_cabecera(origen) or b""
    sufijo = sufijo if sufijo is not None else origen.suffix
    return camino_de_tipo(tipo_por_cabecera(cabecera, sufijo), sufijo)


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
            return _Rechazo("animacion_no_soportada", "gif_animado")
    elif datos[:4] == b"RIFF" and _webp_animado(datos):
        return _Rechazo("animacion_no_soportada", "webp_animado")
    elif _bmp_16_bits_con_mascaras(datos):
        return _Rechazo("bmp_no_soportado", "bmp_16_bits")
    try:
        from io import BytesIO

        from PIL import Image
    except ImportError:
        return "sin_pillow"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(BytesIO(datos)) as img:
                # Jax#338 ronda 18: un PNG/APNG de mas de un cuadro se rechaza
                # como un GIF o WebP animado. Leido como una sola imagen,
                # tesseract solo ve el cuadro por defecto (un primer cuadro en
                # blanco daba ok/imagen_sin_texto con el texto en el segundo).
                # Pillow cuenta los cuadros con el `acTL`, sin decodificarlos.
                if img.format == "PNG" and getattr(img, "n_frames", 1) > 1:
                    return _Rechazo("animacion_no_soportada", "png_animado")
                # Multicuadro PARA TESSERACT: un TIFF (sus paginas) y un MPO
                # (fotos de telefono: la foto y otros cuadros). Jax#338 ronda 19:
                # un MPO de varios cuadros ya no se lee por su primer cuadro solo
                # (un cuadro 1 en blanco perdia el texto del cuadro 2): cada cuadro
                # va con SUS bytes JPEG originales (`"mpo"`, sin re-codificar; el
                # primero es el comienzo del archivo y se lee igual que antes).
                # GIF, WebP y PNG de varios cuadros son animaciones y ya se
                # rechazaron arriba.
                n = getattr(img, "n_frames", 1) if img.format in ("TIFF", "MPO") else 1
                if n > MAX_PAGINAS:
                    return "demasiadas_paginas"
                dimensiones: list[tuple[int, int]] = []
                total = 0
                for i in range(n):
                    img.seek(i)
                    ancho, alto = img.size
                    total += ancho * alto
                    if _formato_de_imagen(img):
                        return _Rechazo("modo_no_soportado", _formato_de_imagen(img))
                    if img.mode not in _MODOS_PNG:
                        limite = MAX_PIXELES_OTROS_MODOS
                    else:
                        limite = MAX_PIXELES
                    if ancho * alto > limite or total > MAX_PIXELES_TOTAL:
                        return "demasiados_pixeles"
                    dimensiones.append((ancho, alto))
                if n == 1:
                    img.load()
                    mascara = _mascara_alfa(img)
                    transparente = mascara is not None
                    del mascara
                    if transparente and ancho * alto > MAX_PIXELES_OTROS_MODOS:
                        return "demasiados_pixeles"   # el tope reducido de todo lo que se aplana
                    if transparente or _bmp_con_mascaras(datos) is not None:
                        # transparencia REAL (o un BMP con mascaras): no se le
                        # pasan los bytes originales a leptonica; va por Pillow
                        # -> PNG. Un RGBA opaco sigue con los bytes originales.
                        return "tiff", dimensiones
                    return "una", dimensiones
                if img.format == "MPO":
                    return "mpo", dimensiones   # JPEG: sin alfa, no aplica N37
                # N37 (Jax#338 ronda 11): el tope del aplanado se comprueba en
                # TODAS las paginas de un TIFF ANTES del OCR de ninguna (si no,
                # se gasta el OCR de las primeras y el archivo falla entero en
                # una posterior). Solo hace falta decodificar una pagina que
                # supera el tope Y tiene un modo con alfa o un `transparency`:
                # sin eso no puede haber transparencia real. De a una pagina, y
                # la mascara se libera antes de pasar a la siguiente.
                for indice, (ancho, alto) in enumerate(dimensiones):
                    if ancho * alto <= MAX_PIXELES_OTROS_MODOS:
                        continue
                    img.seek(indice)
                    if img.mode not in _MODOS_CON_ALFA and "transparency" not in img.info:
                        continue
                    img.load()
                    if _mascara_alfa(img) is not None:
                        return "demasiados_pixeles"
    except Image.DecompressionBombError:
        return "demasiados_pixeles"
    except MemoryError:
        return "sin_memoria"   # recursos, no archivo danado
    except Exception:  # fail-soft: Pillow lanza OSError/ValueError/SyntaxError/EOFError segun el formato roto; todas significan "no decodifica" y salen como Resultado(error, archivo_ilegible)
        return "no_decodifica"
    return "tiff", dimensiones


_MODOS_PNG = frozenset({"1", "L", "LA", "P", "RGB", "RGBA"})

# Modos que NO se soportan, sin llamar a tesseract: F (coma flotante), I (32
# bits) e I;16* (gris de 16 bits). Ninguna imagen real de LACTOVI lo es. Cada
# intento de leerlos abrio un defecto nuevo (escalar F por extremos o
# percentiles, el byte alto de un PNG de 16 bits que deja negra la pagina por
# un solo valor >= 256): se rechazan, con un error VISIBLE que nombra el formato.
_FORMATO_POR_MODO = {
    "F": "coma_flotante",
    "I": "entero_32_bits",
    "I;16": "gris_16_bits", "I;16B": "gris_16_bits",
    "I;16L": "gris_16_bits", "I;16N": "gris_16_bits",
}


def _formato_de_modo(modo: str) -> str | None:
    """Valor estable de `detalle["formato"]` para un modo no soportado, o
    `None` si el modo se soporta."""
    # (Un PNG de 16 bits en COLOR o con alfa, con valores 0-255, lo informa
    # Pillow como RGB/RGBA y no como I;16: no se cambia, porque la lectura
    # estandar es fiel al archivo.)
    return _FORMATO_POR_MODO.get(modo)


def _formato_de_imagen(img) -> str | None:
    """Como `_formato_de_modo`, pero un TIFF de UN canal de 16 bits CON SIGNO
    (BitsPerSample=16, SampleFormat=2) que Pillow informa como `I` es
    `gris_16_bits`, no `entero_32_bits`."""
    if img.mode == "I" and getattr(img, "format", None) == "TIFF":
        try:
            bits = img.tag_v2.get(258)
        except Exception:  # fail-soft: sin tag_v2 legible se cae al mapa por modo (entero_32_bits), que tambien rechaza
            bits = None
        if bits in (16, (16,)):
            return "gris_16_bits"
    return _formato_de_modo(img.mode)


class _Rechazo(str):
    """Causa de rechazo de la validacion (un `str`: `modo_no_soportado`...)
    con el `formato` estable cuando es `formato_no_soportado`."""

    formato: str | None = None

    def __new__(cls, causa: str, formato: str | None = None):
        obj = super().__new__(cls, causa)
        obj.formato = formato
        return obj


def _bmp_con_mascaras(datos: bytes) -> int | None:
    """Los bpp de un BMP con mascaras de bits (compresion BI_BITFIELDS o
    BI_ALPHABITFIELDS, 3 o 6), o `None` si no lo es. Leptonica no lee ningun BMP
    comprimido (`cannot read compressed BMP files`); Pillow si."""
    if datos[:2] != b"BM" or len(datos) < 34:
        return None
    tamano_header, = struct.unpack_from("<I", datos, 14)
    if tamano_header < 40:
        return None
    bpp, = struct.unpack_from("<H", datos, 28)
    compresion, = struct.unpack_from("<I", datos, 30)
    return bpp if compresion in (3, 6) else None


def _bmp_16_bits_con_mascaras(datos: bytes) -> bool:
    """BMP de 16 bpp con mascaras (RGB565): formato NO soportado
    (`bmp_16_bits`), detectado por el header ANTES de tesseract. Un 16 bpp sin
    compresion (555) no. Los demas BMP con mascaras (p. ej. 32 bpp, como los
    exportan GIMP y Photoshop con alfa) NO se rechazan: pasan por Pillow y se
    mandan como PNG (`_a_modo_legible` / camino `tiff`)."""
    return _bmp_con_mascaras(datos) == 16


_MODOS_CON_ALFA = frozenset({"RGBA", "LA", "PA", "RGBa", "La"})


def _tabla_alfa(transparencia, opaco_por_defecto: int = 255) -> bytes:
    """Tabla de 256 entradas indice/gris -> alfa a partir de `info["transparency"]`:
    un entero (ese indice es transparente) o bytes (alfa por indice, como el
    `tRNS` de un PNG paletizado)."""
    if isinstance(transparencia, int):
        return bytes(0 if i == transparencia else opaco_por_defecto for i in range(256))
    alfas = bytes(transparencia)
    return bytes(alfas[i] if i < len(alfas) else opaco_por_defecto for i in range(256))


def _mascara_alfa(img):
    """Mascara `L` (255 = opaco) si la imagen tiene transparencia REAL, o `None`.
    Real = algun pixel con alfa < 255, o un `transparency` (indice, gris o color
    de un `tRNS`) que de verdad coincide con algun pixel. Un RGBA con el alfa en
    255 en toda la imagen NO es transparente: va por el camino de antes. Sin una
    RGBA intermedia (menos copias que `convert("RGBA")`)."""
    from PIL import Image, ImageChops

    modo = img.mode
    if modo in ("RGBA", "LA", "PA"):
        mascara = img.getchannel("A")
    elif modo in _MODOS_CON_ALFA:                      # RGBa / La, premultiplicados
        mascara = img.convert("RGBA").getchannel("A")
    elif "transparency" not in img.info:
        return None
    elif modo in ("P", "L"):
        tabla = _tabla_alfa(img.info["transparency"])
        mascara = Image.frombytes("L", img.size, img.tobytes().translate(tabla))
    elif modo == "1":
        tabla = _tabla_alfa(255 if img.info["transparency"] else 0)
        mascara = Image.frombytes("L", img.size, img.convert("L").tobytes().translate(tabla))
    elif modo == "RGB" and isinstance(img.info["transparency"], tuple):
        coincide = None
        for banda, valor in zip(img.split(), img.info["transparency"]):
            igual = banda.point(lambda v, c=valor: 255 if v == c else 0)
            coincide = igual if coincide is None else ImageChops.darker(coincide, igual)
        mascara = ImageChops.invert(coincide)
    else:
        return None
    return mascara if mascara.getextrema()[0] < 255 else None


def _aplanar(img, mascara, fondo):
    """RGB sin alfa, sobre `fondo`. Leptonica descarta el alfa de WebP, TIFF y
    GIF y el fondo transparente queda NEGRO: el texto oscuro queda negro sobre
    negro (ok/imagen_sin_texto en silencio). Un unico lienzo RGB y `paste` con
    mascara (sin RGBA intermedia)."""
    from PIL import Image

    lienzo = Image.new("RGB", img.size, fondo)
    lienzo.paste(img, mask=mascara)
    return lienzo


def _a_modo_legible(img):
    """Fotograma decodificado SIN transparencia real (`_mascara_alfa` dio
    `None`) -> uno que se pueda guardar como PNG para tesseract: un alfa opaco
    se descarta sin perder nada; los modos de `_MODOS_PNG` tal cual; el resto
    (CMYK, YCbCr, LAB, HSV...) a `RGB`. (F, I e I;16* ya se rechazaron en la
    validacion; la transparencia real la aplana `_ocr_cuadro`.)"""
    if img.mode in ("RGBA", "PA", "RGBa"):
        return img.convert("RGB")
    if img.mode in ("LA", "La"):
        return img.convert("L")
    if img.mode in _MODOS_PNG:
        return img
    return img.convert("RGB")


def _a_png(cuadro, dpi=None) -> bytes:
    """PNG en memoria. `dpi` (el `info["dpi"]` del original) se copia: sin el,
    tesseract estima la resolucion y lee distinto (una captura de 190 dpi pasaba
    de 1910 a 1852 caracteres)."""
    from io import BytesIO

    salida = BytesIO()
    if dpi:
        cuadro.save(salida, format="PNG", dpi=dpi)
    else:
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


def _clasificar_imagen(caracteres: int, analisis: dict, transparente: bool) -> str:
    """`_clasificar`, con la invariante 1 de Jax#338 ronda 16: una imagen con
    transparencia REAL (el texto combina dos lecturas, sobre fondo blanco y
    sobre fondo negro, que nadie verifico) NUNCA sale `ok`: lo que no es la
    regla (A) -- menos de MINIMO_CARACTERES, que con tamano de pagina es la
    (D) -- es `transparente` (parcial + imagen_texto_dudoso,
    `_resolver_imagen`)."""
    clasificacion = _clasificar(caracteres, analisis)
    if not transparente:
        return clasificacion
    if clasificacion == "sin_texto" and caracteres < MINIMO_CARACTERES:
        return clasificacion
    return "transparente"


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


def _ocr_bytes(
    datos: bytes, idioma: str, presupuesto: _Presupuesto | None = None, con_tsv: bool = False
) -> dict | None:
    """Corre tesseract DOS veces sobre la MISMA imagen -- texto plano (para
    el extracto exacto, tildes y guion largo incluidos) y `tsv` (para la
    confianza por palabra, que el modo texto plano no expone). `None` si el
    propio subproceso de tesseract no pudo correr sobre esta imagen
    (I/O, `returncode` distinto de cero) -- fallo cerrado del llamador, nunca
    una excepción escapando de acá. Si tesseract no pudo DECODIFICAR el
    archivo (imagen danada) o se agoto el tiempo, devuelve un dict con
    `clasificacion="ilegible"` y su `causa` en vez de `None`, para que
    `_resolver_imagen` pueda distinguirlo con su codigo. `con_tsv`: el dict
    lleva ademas la salida `tsv` cruda (`"tsv"`), para `_unir_pasadas`."""
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
    salida_tsv = _como_texto(proceso_tsv.stdout)
    analisis = _analizar_tsv(salida_tsv)

    resultado = {
        "texto": texto,
        "caracteres": len(texto),
        **analisis,
        "clasificacion": _clasificar(len(texto), analisis),
    }
    if con_tsv:
        resultado["tsv"] = salida_tsv
    return resultado


def _ocr_una_imagen(ruta: Path, idioma: str) -> dict | None:
    """Camino de las paginas rasterizadas de un PDF (PNG propios, en un
    directorio temporal nuestro): lee el archivo y lo manda por stdin."""
    try:
        datos = Path(ruta).read_bytes()
    except OSError:
        return None
    return _ocr_bytes(datos, idioma)


# Fondos sobre los que se aplana un fotograma con transparencia REAL, en este
# orden: el texto de la pasada blanca va primero en la union (Jax#338 ronda 12).
_FONDOS_DEL_APLANADO = ((255, 255, 255), (0, 0, 0))


# Un renglon de la pasada negra es DUPLICADO de uno de la blanca solo si tienen
# el mismo texto (sin espacios de mas) Y sus cajas se superponen: interseccion
# sobre la menor de las dos areas >= este valor (Jax#338 ronda 13). Dos `TOTAL`
# en lugares distintos de la imagen son dos renglones.
SUPERPOSICION_MINIMA_DUPLICADO = 0.5


def _normalizar_linea(linea: str) -> str:
    """Un renglon sin espacios de mas, para comparar renglones entre pasadas."""
    return " ".join(linea.split())


def _renglones_tsv(salida_tsv: str) -> list[dict]:
    """Renglones (pagina, bloque, parrafo, linea) de las filas de palabra (nivel
    5) de un `tsv`, en su orden y con texto: `{"texto": normalizado, "caja":
    (x0, y0, x1, y1) o None, "filas": [...]}`. La caja del renglon es la que
    encierra las de sus palabras (`left`, `top`, `width`, `height`); `None` si
    alguna no es numerica, y entonces el renglon nunca es duplicado."""
    grupos: dict[tuple, list[list[str]]] = {}
    for fila in salida_tsv.splitlines()[1:]:
        campos = fila.split("\t")
        if len(campos) >= 12 and campos[0] == "5":
            grupos.setdefault(tuple(campos[1:5]), []).append(campos)
    renglones = []
    for filas in grupos.values():
        texto = _normalizar_linea(" ".join(c[11] for c in filas))
        if not texto:
            continue
        try:
            cajas = [(int(c[6]), int(c[7]), int(c[6]) + int(c[8]), int(c[7]) + int(c[9])) for c in filas]
            caja = (min(c[0] for c in cajas), min(c[1] for c in cajas),
                    max(c[2] for c in cajas), max(c[3] for c in cajas))
        except ValueError:  # fail-soft: una caja no numerica deja el renglon sin caja, y un renglon sin caja nunca es duplicado (se conserva)
            caja = None
        renglones.append({"texto": texto, "caja": caja, "filas": ["\t".join(c) for c in filas]})
    return renglones


def _superposicion(a, b) -> float:
    """Interseccion de dos cajas `(x0, y0, x1, y1)` sobre la menor de sus areas;
    0 si alguna falta o tiene area nula."""
    if a is None or b is None:
        return 0.0
    ancho = min(a[2], b[2]) - max(a[0], b[0])
    alto = min(a[3], b[3]) - max(a[1], b[1])
    menor = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    if ancho <= 0 or alto <= 0 or menor <= 0:
        return 0.0
    return ancho * alto / menor


def _unir_pasadas(blanca: dict, negra: dict) -> dict:
    """UNION de las dos pasadas de un fotograma con transparencia real (Jax#338
    ronda 12): no se elige una y se descarta la otra, porque cada criterio de
    eleccion perdio texto real (TOTAL en una pasada y L500 en la otra daban
    ok/imagen_sin_texto).

    El `tsv` es la fuente principal (Jax#338 ronda 14): la deduplicacion es
    solo entre renglones del `tsv` (el texto plano divergia y la deduplicacion
    de uno borraba renglones reales del otro). De las filas de palabra de las dos
    pasadas salen, juntos, la deduplicacion, el texto y las metricas:
    - renglones = palabras agrupadas por (pagina, bloque, parrafo, renglon) en
      el orden del `tsv`, unidas por espacio (`_renglones_tsv`);
    - un renglon de la negra es DUPLICADO si tiene el mismo texto y su caja se
      superpone con la de un renglon de la blanca (`SUPERPOSICION_MINIMA_DUPLICADO`;
      cada renglon de la blanca empareja a lo sumo uno de la negra);
    - texto: los renglones de la blanca y despues los propios de la negra,
      separados por salto de linea;
    - metricas: `_analizar_tsv` sobre esas mismas filas; clasificacion:
      `_clasificar`.
    Ademas (ronda 16, invariante 2), al final se agrega toda linea del texto
    plano de cualquiera de las dos pasadas que no este ya en el texto, para no
    perder texto legible cuando el TSV es parcial; marca `sin_posicion`."""
    blancos = _renglones_tsv(blanca["tsv"])
    negros = _renglones_tsv(negra["tsv"])
    sin_pareja = list(blancos)
    propios_negros = []
    for renglon in negros:
        pareja = next((
            blanco for blanco in sin_pareja
            if blanco["texto"] == renglon["texto"]
            and _superposicion(blanco["caja"], renglon["caja"]) >= SUPERPOSICION_MINIMA_DUPLICADO
        ), None)
        if pareja is None:
            propios_negros.append(renglon)
        else:
            sin_pareja.remove(pareja)
    renglones = blancos + propios_negros
    lineas = [renglon["texto"] for renglon in renglones]
    # Invariante 2 (ronda 16): ninguna linea legible se pierde aunque el TSV sea
    # parcial. Toda linea del texto plano de CUALQUIERA de las dos pasadas cuyo
    # texto normalizado no esta ya en el texto final se agrega al final (de
    # esta pagina). Marca `sin_posicion`: ese texto no tiene caja ni metricas.
    presentes = {_normalizar_linea(linea) for linea in lineas}
    sin_posicion = False
    for pasada in (blanca, negra):
        for linea in pasada["texto"].splitlines():
            normalizada = _normalizar_linea(linea)
            if normalizada and normalizada not in presentes:
                lineas.append(linea.strip())
                presentes.add(normalizada)
                sin_posicion = True
    texto = "\n".join(lineas)
    # La fila de nivel 1 (pagina) solo da ancho y alto, no aporta palabras.
    paginas = [
        fila for salida in (blanca["tsv"], negra["tsv"]) for fila in salida.splitlines()[1:]
        if fila.split("\t", 1)[0] == "1"
    ][:1]
    analisis = _analizar_tsv("\n".join(
        ["encabezado", *paginas, *(fila for renglon in renglones for fila in renglon["filas"])]
    ))
    return {
        "texto": texto,
        "caracteres": len(texto),
        **analisis,
        "sin_posicion": sin_posicion,
        "transparente": True,
        "clasificacion": _clasificar_imagen(len(texto), analisis, True),
    }


def _ocr_cuadro(img, idioma: str, presupuesto: _Presupuesto) -> dict | None:
    """OCR de un fotograma ya decodificado. Sin transparencia real: una sola
    pasada (`_a_modo_legible` -> PNG). Con transparencia REAL (`_mascara_alfa`):
    DOS pasadas, aplanado sobre blanco y sobre negro, y el resultado es su
    UNION (`_unir_pasadas`). Ningun numero sobre toda la imagen decide el
    fondo: la luminancia media la decide la figura mas grande (un logo con
    emblema claro y texto oscuro) y no el texto. Un lienzo a la vez: aplanar
    -> PNG -> OCR -> liberar, y repetir.
    Un fallo de OCR (`None` o `ilegible`) en cualquiera de las dos pasadas es
    el resultado."""
    dpi = img.info.get("dpi")
    try:
        mascara = _mascara_alfa(img)
        png = _a_png(_a_modo_legible(img), dpi) if mascara is None else None
    except MemoryError:
        return _ilegible_dict("sin_memoria")   # recursos, no archivo danado
    except Exception:  # fail-soft: pagina truncada o corrupta = archivo que no decodifica
        return _ilegible_dict("no_decodifica")
    if mascara is None:
        return _ocr_bytes(png, idioma, presupuesto)
    if img.size[0] * img.size[1] > MAX_PIXELES_OTROS_MODOS:
        # tope del aplanado; `_validar_imagen` ya lo comprobo en todas las
        # paginas antes del OCR (N37): esto es la defensa si se llega igual
        return _ilegible_dict("demasiados_pixeles")
    pasadas = []
    for fondo in _FONDOS_DEL_APLANADO:
        try:
            png = _a_png(_aplanar(img, mascara, fondo), dpi)   # el lienzo muere aca
        except MemoryError:
            return _ilegible_dict("sin_memoria")
        except Exception:  # fail-soft: igual que arriba, un fotograma que no se puede convertir no decodifica
            return _ilegible_dict("no_decodifica")
        r = _ocr_bytes(png, idioma, presupuesto, con_tsv=True)
        del png
        if r is None or r["clasificacion"] == "ilegible":
            return r
        pasadas.append(r)
    return _unir_pasadas(*pasadas)


def _bytes_del_cuadro_mpo(img, datos: bytes, indice: int) -> bytes:
    """Los bytes JPEG ORIGINALES del cuadro `indice` de un MPO (ya posicionado
    con `img.seek(indice)`): desde `img.offset` (donde Pillow ubica el cuadro)
    y del tamano que declara su entrada MP. Un cuadro que no empieza con SOI o
    que queda cortado lanza `ValueError` (archivo que no decodifica)."""
    inicio = img.offset
    tamano = img.mpinfo[0xB002][indice]["Size"]
    trozo = datos[inicio:inicio + tamano]
    if len(trozo) != tamano or not trozo.startswith(b"\xff\xd8"):
        raise ValueError("cuadro MPO fuera del archivo")
    return trozo


def _ocr_imagen(datos: bytes, tipo: str, dimensiones: list, idioma: str) -> dict | None:
    """OCR de una imagen ya validada. `"una"`: los bytes originales por stdin.
    `"mpo"` (ronda 19): cuadro por cuadro, cada uno decodificado con Pillow para
    validarlo y enviado con sus bytes JPEG originales, con la marca
    `<!-- cuadro N -->`.
    `"tiff"`: pagina por pagina (decodificar -> `_ocr_cuadro` -> descartar:
    nunca se guarda la lista de PNG; con transparencia real, la union de las
    dos pasadas es POR PAGINA), con el plazo total `PLAZO_TOTAL_SEGUNDOS`
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
    transparente = sin_posicion = False   # alguna pagina con transparencia real / texto sin posicion
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
                    cuadro_mpo = _bytes_del_cuadro_mpo(img, datos, numero - 1) if tipo == "mpo" else None
                    img.load()
                except MemoryError:
                    return _ilegible_dict("sin_memoria")   # recursos, no archivo danado
                except Exception:  # fail-soft: pagina truncada o corrupta = archivo que no decodifica
                    return _ilegible_dict("no_decodifica")
                if cuadro_mpo is not None:
                    r = _ocr_bytes(cuadro_mpo, idioma, presupuesto)
                    del cuadro_mpo
                else:
                    r = _ocr_cuadro(img, idioma, presupuesto)
                if r is None:
                    return None
                if r["clasificacion"] == "ilegible":
                    return r
                primero = primero or r
                transparente = transparente or r.get("transparente", False)
                sin_posicion = sin_posicion or r.get("sin_posicion", False)
                if r["texto"] and len(dimensiones) == 1:
                    partes.append(r["texto"])   # un solo fotograma: sin marca de pagina
                elif r["texto"]:
                    marca = "cuadro" if tipo == "mpo" else "página"
                    partes.append(f"<!-- {marca} {numero} -->\n{r['texto']}")
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
        "sin_posicion": sin_posicion,
        "clasificacion": _clasificar_imagen(caracteres, analisis, transparente),
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


# Que codigo lleva cada causa:
# - `archivo_ilegible` ("danado"): SOLO firma que no es de imagen o que Pillow no
#   decodifica (incluye los truncados).
# - `formato_no_soportado`: la deteccion (SIEMPRE antes de tesseract) nombra el
#   formato en `detalle["formato"]` (gif_animado, webp_animado, png_animado, gris_16_bits,
#   coma_flotante, entero_32_bits, bmp_16_bits) y la accion.
# - `archivo_no_procesable` (`tesseract_no_lee`): Pillow lo decodifico y
#   leptonica lo rechaza (JPEG con basura en los datos, PNG cortado antes de
#   IEND, BMP comprimido...): puede estar danado O ser un formato que no
#   leemos, y no se afirma ninguna de las dos.
# - `imagen_demasiado_grande`: los topes de pixeles y paginas.
_DETALLE_DEMASIADO_GRANDE = {
    "demasiados_pixeles": "demasiados pixeles",
    "demasiadas_paginas": "demasiadas paginas",
}
_CAUSAS_FORMATO_NO_SOPORTADO = frozenset(
    {"modo_no_soportado", "animacion_no_soportada", "bmp_no_soportado"})
ACCION_FORMATO_NO_SOPORTADO = "conviértelo a JPEG o PNG y vuelve a subirlo"


def _ilegible(causa: str, idioma: str, formato: str | None = None) -> Resultado:
    formato = formato or getattr(causa, "formato", None)
    extra: dict = {}
    if causa in ("tiempo_excedido", "tiempo_por_llamada"):
        razon, codigo = "se excedio el tiempo de OCR de la imagen", CODIGO_OCR_TIEMPO_EXCEDIDO
    elif causa == "sin_memoria":
        razon, codigo = "memoria insuficiente para procesar la imagen", CODIGO_OCR_SIN_MEMORIA
    elif causa in _CAUSAS_FORMATO_NO_SOPORTADO and formato:
        razon = f"formato de imagen no soportado ({formato}): {ACCION_FORMATO_NO_SOPORTADO}"
        codigo = CODIGO_FORMATO_NO_SOPORTADO
        extra["formato"] = formato
    elif causa == "tesseract_no_lee":
        razon = "el OCR no pudo leer la imagen (dañada o en un formato no soportado)"
        codigo = CODIGO_ARCHIVO_NO_PROCESABLE
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
            "razon": razon, "codigo": codigo, "causa": str(causa), "idioma": idioma,
            "_camino": "imagen", **extra,
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
            detalle["razon"] = RAZON_MAYORIA_DUDOSA
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

    if r["clasificacion"] == "transparente":
        # Invariante 1 (Jax#338 ronda 16): imagen con transparencia real, como
        # mucho parcial; se CONSERVA todo el texto.
        razones = [RAZON_IMAGEN_TRANSPARENTE]
        n, dudosas = r["n_palabras"], r["palabras_dudosas"]
        if n and len(dudosas) / n > PROPORCION_MAXIMA_PALABRAS_DUDOSAS:
            razones.append(RAZON_MAYORIA_DUDOSA)
        if r.get("sin_posicion"):
            razones.append(RAZON_TEXTO_SIN_POSICION)
        detalle["razon"] = "; ".join(razones)
        detalle["codigo"] = CODIGO_IMAGEN_TEXTO_DUDOSO
        return Resultado(
            estado="parcial",
            salidas={"texto.txt": f"{NOTA_TEXTO_DUDOSO}\n{r['texto']}"},
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


def _dimensiones_de_png(ruta: Path) -> tuple[int, int] | None:
    """Ancho y alto del PNG RASTERIZADO de una pagina, leidos del archivo (no del
    TSV, que puede no traer la fila de pagina); `None` si no se pueden leer."""
    from PIL import Image   # su ausencia NO es una pagina ilegible: `extraer` la avisa antes de rasterizar

    try:
        with Image.open(ruta) as imagen:
            ancho, alto = imagen.size
    except Exception:  # fail-soft: un PNG que no decodifica cuenta como pagina ilegible
        return None
    return (ancho, alto) if ancho > 0 and alto > 0 else None


def _resolver_pdf_sin_texto(
    resultados: list[dict | None], idioma: str,
    dimensiones: list[tuple[int, int] | None] | None = None,
) -> Resultado:
    """Ninguna pagina del PDF dio texto util. Decision de Fernando (2026-10-04):
    se trata IGUAL que una imagen (`_resolver_imagen`), pero SOLO si todas las
    paginas son legibles:
    - alguna pagina `None`, `ilegible` o cuyo PNG rasterizado no se pudo medir:
      `error` (nada de parcial: una pagina que nadie leyo no se esconde), con
      `paginas_ilegibles` y el codigo de la causa si la hay;
    - alguna con mayoria de palabras dudosas (B, con al menos MINIMO_CARACTERES):
      `parcial` + `imagen_texto_dudoso`, CONSERVANDO su texto (B antes que D);
    - si no, todas son A (menos de MINIMO_CARACTERES), como la imagen: con alguna
      pagina de tamano de pagina (`_implica_pagina` sobre las dimensiones del PNG
      RASTERIZADO) -> `parcial` + `imagen_pagina_sin_texto` (D); si ninguna lo
      es -> `ok` + `imagen_sin_texto` (A). Siempre el aviso, nunca texto
      inventado."""
    total = len(resultados)
    dimensiones = list(dimensiones) if dimensiones is not None else [None] * total
    ilegibles = [
        numero for numero, r in enumerate(resultados, start=1)
        if r is None or r["clasificacion"] == "ilegible" or dimensiones[numero - 1] is None
    ]
    if ilegibles:
        detalle = {
            "razon": "ninguna pagina del PDF dio texto util via OCR",
            "idioma": idioma,
            "paginas": total,
            "paginas_ilegibles": ilegibles,
            "_camino": "pdf",
        }
        causas = [r["causa"] for r in resultados if r is not None and r["clasificacion"] == "ilegible"]
        if not causas and any(
            r is not None and dimensiones[n - 1] is None for n, r in enumerate(resultados, start=1)
        ):
            causas = ["pagina_sin_dimensiones"]   # cae en archivo_ilegible
        if causas:
            # el codigo que ya corresponde a esa causa en las imagenes
            detalle["codigo"] = _ilegible(causas[0], idioma).detalle["codigo"]
            detalle["causa"] = str(causas[0])
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR,
            version=_version() or "desconocida", detalle=detalle,
        )

    n_palabras = sum(r["n_palabras"] for r in resultados)
    detalle = {
        "idioma": idioma, "paginas": total, "_camino": "pdf",
        "paginas_sin_texto": list(range(1, total + 1)),
        "confianza_promedio": (
            round(sum(r["confianza_promedio"] * r["n_palabras"] for r in resultados) / n_palabras, 2)
            if n_palabras else 0.0
        ),
    }
    dudosas = [(numero, r) for numero, r in enumerate(resultados, start=1)
               if r["caracteres"] >= MINIMO_CARACTERES]
    if dudosas:
        # (B) el texto leido SE CONSERVA, igual que en una imagen.
        palabras = [
            {"pagina": numero, **palabra} for numero, r in dudosas for palabra in r["palabras_dudosas"]
        ]
        detalle["paginas_texto_dudoso"] = [numero for numero, _ in dudosas]
        if palabras:
            detalle["palabras_dudosas"] = palabras
        detalle["razon"] = RAZON_MAYORIA_DUDOSA
        detalle["codigo"] = CODIGO_IMAGEN_TEXTO_DUDOSO
        cuerpo = "\n\n".join(f"<!-- página {numero} -->\n{r['texto']}" for numero, r in dudosas)
        return Resultado(
            estado="parcial", salidas={"texto.txt": f"{NOTA_TEXTO_DUDOSO}\n{cuerpo}"},
            extractor=EXTRACTOR, version=_version() or "desconocida", detalle=detalle,
        )
    # Todas A: con tamano de PAGINA es la (D); si no, la (A).
    for ancho, alto in dimensiones:
        pagina = _implica_pagina(ancho, alto)
        if pagina is not None:
            detalle["ancho"], detalle["alto"] = ancho, alto
            detalle["pagina_de_referencia"] = pagina
            detalle["razon"] = "posible documento escaneado sin texto: revisar o reescanear"
            detalle["codigo"] = CODIGO_IMAGEN_PAGINA_SIN_TEXTO
            return Resultado(
                estado="parcial",
                salidas={"texto.txt": f"{AVISO_IMAGEN_SIN_TEXTO}\n{NOTA_PAGINA_SIN_TEXTO}"},
                extractor=EXTRACTOR, version=_version() or "desconocida", detalle=detalle,
            )
    detalle["razon"] = "el OCR no devolvio texto util"
    detalle["codigo"] = CODIGO_IMAGEN_SIN_TEXTO
    return Resultado(
        estado="ok", salidas={"texto.txt": AVISO_IMAGEN_SIN_TEXTO},
        extractor=EXTRACTOR, version=_version() or "desconocida", detalle=detalle,
    )


def _resolver_pdf(
    resultados: list[dict | None], idioma: str,
    dimensiones: list[tuple[int, int] | None] | None = None,
) -> Resultado:
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
        return _resolver_pdf_sin_texto(resultados, idioma, dimensiones)

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
            try:
                import PIL.Image  # noqa: F401  (D mide el PNG rasterizado con Pillow)
            except ImportError:
                return Resultado(
                    estado="error", salidas={}, extractor=EXTRACTOR,
                    version=_version() or "desconocida",
                    detalle={
                        "razon": "Pillow no esta instalado; no se puede medir la pagina rasterizada",
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
                dimensiones = [_dimensiones_de_png(pagina) for pagina in paginas]
            return _resolver_pdf(resultados, idioma, dimensiones)

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
