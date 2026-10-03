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
  NOMBRAN en `detalle["palabras_dudosas"]` (nunca se borran del texto), con
  más de la mitad de las palabras dudosas el resultado es `error`
  (`PROPORCION_MAXIMA_PALABRAS_DUDOSAS`), con alguna pero no la mayoría es
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

import os
import shutil
import subprocess
import tempfile
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
# superada, hace que la página entera se trate como "sin texto útil"
# (`error`, o esa página cuenta en `paginas_sin_texto` de un PDF) en vez de
# `parcial`: más de la mitad de las palabras en duda es, en la práctica, la
# misma situación que no haber leído nada -- no se puede confiar en el resto.
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
#   danada / ilegible                        -> error   archivo_ilegible
CODIGO_IMAGEN_SIN_TEXTO = "imagen_sin_texto"
CODIGO_IMAGEN_TEXTO_DUDOSO = "imagen_texto_dudoso"
CODIGO_IMAGEN_PAGINA_SIN_TEXTO = "imagen_pagina_sin_texto"
CODIGO_ARCHIVO_ILEGIBLE = "archivo_ilegible"

# Version de la LOGICA de clasificacion de este extractor. `extractor_version`
# es la de tesseract y NO cambia cuando cambia una regla: esta cadena se guarda
# en la ficha (`detalle["_version_logica"]`) y `ingesta` la compara -- una
# ficha escrita con otra logica ni se reusa de cache ni cuenta como intento
# previo del tope D-2. SUBIRLA cada vez que cambie la regla.
# La version DEPENDE DEL CAMINO (`version_logica(extension)`): solo cambio la
# regla de las IMAGENES (A/B/D, codigos nuevos, TIFF multipagina) -- la del PDF
# escaneado no, asi que su cache sigue valiendo (no se re-OCR-ean los PDF).
# "2": regla de Fernando de la ronda 1 de Jax#338.
VERSION_LOGICA_IMAGEN = "2"
EXTENSIONES_IMAGEN = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
)


def version_logica(extension: str) -> str | None:
    """Version de la logica para una ficha ingerida con esa extension; `None`
    (sin marca) para el camino PDF."""
    return VERSION_LOGICA_IMAGEN if extension.lower() in EXTENSIONES_IMAGEN else None


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

# Tope de fotogramas de un TIFF/GIF/WebP multipagina (tesseract procesa
# TODOS; cada uno se decodifica y se mide antes).
MAX_PAGINAS = 50

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


def _es_pdf(origen: Path) -> bool:
    """MINOR-4: la firma `%PDF` puede venir tras unos bytes de basura (BOM,
    saltos de linea); se busca en los primeros 1024. MINOR-N3: pero UNA FIRMA
    DE IMAGEN VALIDA MANDA -- un PNG con metadata `%PDF` es una imagen."""
    try:
        with open(origen, "rb") as fh:
            cabecera = fh.read(1024)
    except OSError:
        return False
    if _tiene_firma_de_imagen(cabecera[:16]):
        return False
    return _FIRMA_PDF in cabecera


def _tiene_firma_de_imagen(cabecera: bytes) -> bool:
    return cabecera.startswith(_FIRMAS_IMAGEN) or (
        cabecera[:4] == b"RIFF" and cabecera[8:12] == b"WEBP"
    )


def _validar_imagen(datos: bytes) -> tuple[list[bytes], list[tuple[int, int]]] | str:
    """Antes de OCR, sobre los BYTES (se leen una sola vez, MINOR-N2: nunca se
    relee la ruta, asi lo validado es lo que se procesa): firma magica (S-1) y
    decodificacion ENTERA de TODOS los fotogramas con Pillow (MAJOR-2/N1: un
    TIFF truncado da rc=0 y vacio en tesseract, y `load()`/`size` solo miran
    el fotograma 0). Devuelve `(paginas, dimensiones)` -- `paginas` son los
    bytes que iran por stdin: los originales si hay UN fotograma, un PNG
    re-codificado por fotograma ya decodificado si hay varios -- o la CAUSA
    (`firma_invalida`, `no_decodifica`, `demasiados_pixeles`,
    `demasiadas_paginas`, `sin_pillow`): un codigo, nunca texto de la
    excepcion (puede traer rutas)."""
    if not _tiene_firma_de_imagen(datos[:16]):
        return "firma_invalida"
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
                # iPhone: la foto + una vista previa) o un GIF/WebP animado se
                # leen por su primer fotograma: tratarlos como paginas
                # re-codificaria la foto (medido en IMG_2353: otro resultado)
                # y sumaria una vista previa que tesseract nunca lee.
                n = getattr(img, "n_frames", 1) if img.format == "TIFF" else 1
                if n > MAX_PAGINAS:
                    return "demasiadas_paginas"
                paginas: list[bytes] = []
                dimensiones: list[tuple[int, int]] = []
                for i in range(n):
                    img.seek(i)
                    ancho, alto = img.size
                    if ancho * alto > MAX_PIXELES:
                        return "demasiados_pixeles"
                    img.load()
                    dimensiones.append((ancho, alto))
                    if n > 1:
                        cuadro = img if img.mode in _MODOS_PNG else img.convert("RGB")
                        salida = BytesIO()
                        cuadro.save(salida, format="PNG")
                        paginas.append(salida.getvalue())
                if n == 1:
                    paginas = [datos]
    except Image.DecompressionBombError:
        return "demasiados_pixeles"
    except Exception:  # fail-soft: Pillow lanza OSError/ValueError/SyntaxError/EOFError segun el formato roto; todas significan "no decodifica" y salen como Resultado(error, archivo_ilegible)
        return "no_decodifica"
    return paginas, dimensiones


_MODOS_PNG = frozenset({"1", "L", "LA", "P", "RGB", "RGBA", "I;16"})


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
    la regla en dos lugares que puedan divergir)."""
    if caracteres < MINIMO_CARACTERES:
        return "sin_texto"
    n = analisis["n_palabras"]
    dudosas = analisis["palabras_dudosas"]
    if n and len(dudosas) / n > PROPORCION_MAXIMA_PALABRAS_DUDOSAS:
        return "sin_texto"
    if n < MINIMO_PALABRAS or dudosas:
        return "con_dudas"
    return "ok"


def _ocr_bytes(datos: bytes, idioma: str) -> dict | None:
    """Corre tesseract DOS veces sobre la MISMA imagen -- texto plano (para
    el extracto exacto, tildes y guion largo incluidos) y `tsv` (para la
    confianza por palabra, que el modo texto plano no expone). `None` si el
    propio subproceso de tesseract no pudo correr sobre esta imagen
    (timeout, I/O, `returncode` distinto de cero) -- fallo cerrado del
    llamador, nunca una excepción escapando de acá. Si tesseract no pudo
    DECODIFICAR el archivo (imagen danada), devuelve un dict con
    `clasificacion="ilegible"` en vez de `None`, para que `_resolver_imagen`
    pueda distinguirlo con `archivo_ilegible`."""
    try:
        proceso = subprocess.run(
            ["tesseract", "-", "stdout", "-l", idioma],
            input=datos, capture_output=True, timeout=TIMEOUT_SEGUNDOS,
        )
    except Exception:  # fail-soft: el subproceso de tesseract (modo texto) puede fallar (timeout, I/O); se devuelve None y el llamador lo convierte en Resultado(estado="error")
        return None
    if proceso.returncode != 0:
        stderr = _como_texto(proceso.stderr)
        if any(marca in stderr for marca in _MARCAS_ARCHIVO_ILEGIBLE):
            return {"clasificacion": "ilegible", "causa": "tesseract_no_lee"}
        return None
    texto = _como_texto(proceso.stdout).strip()

    try:
        proceso_tsv = subprocess.run(
            ["tesseract", "-", "stdout", "-l", idioma, "tsv"],
            input=datos, capture_output=True, timeout=TIMEOUT_SEGUNDOS,
        )
    except Exception:  # fail-soft: el subproceso de tesseract (modo tsv, confianza por palabra) puede fallar igual que el de texto plano; se devuelve None y el llamador lo convierte en Resultado(estado="error")
        return None
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


def _ocr_paginas(paginas: list[bytes], idioma: str) -> dict | None:
    """OCR de una imagen de UNO o varios fotogramas. Con varios, una llamada
    a tesseract por pagina y el texto se junta con un separador de pagina; las
    metricas de la regla A/B se calculan sobre el TOTAL (caracteres y palabras
    sumados, confianza ponderada por palabras)."""
    resultados = [_ocr_bytes(p, idioma) for p in paginas]
    if any(r is None for r in resultados):
        return None
    for r in resultados:
        if r["clasificacion"] == "ilegible":
            return r
    if len(resultados) == 1:
        return resultados[0]
    partes, dudosas = [], []
    caracteres = palabras = 0
    suma_conf = 0.0
    for numero, r in enumerate(resultados, start=1):
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
        "ancho": resultados[0]["ancho"], "alto": resultados[0]["alto"],
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


def _ilegible(causa: str, idioma: str) -> Resultado:
    return Resultado(
        estado="error", salidas={}, extractor=EXTRACTOR,
        version=_version() or "desconocida",
        detalle={
            "razon": "no se pudo abrir ni decodificar la imagen (archivo danado)",
            "codigo": CODIGO_ARCHIVO_ILEGIBLE,
            "causa": causa,
            "idioma": idioma,
        },
    )


def _resolver_imagen(
    r: dict, idioma: str, paginas: list[tuple[int, int]] | None = None
) -> Resultado:
    if r["clasificacion"] == "ilegible":
        return _ilegible(r["causa"], idioma)

    detalle = _detalle_comun(idioma, r)
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
            },
        )

    contenido = "\n\n".join(partes_texto).strip()
    confianza_promedio = (
        round(suma_confianza_ponderada / palabras_totales, 2) if palabras_totales else 0.0
    )
    detalle = {
        "idioma": idioma, "paginas": total, "confianza_promedio": confianza_promedio,
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


def extraer(origen: Path, idioma: str = "spa") -> Resultado:
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

        if _es_pdf(origen):
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
        paginas_bytes, dimensiones = validacion
        resultado_img = _ocr_paginas(paginas_bytes, idioma)
        if resultado_img is None:
            return Resultado(
                estado="error", salidas={}, extractor=EXTRACTOR,
                version=_version() or "desconocida",
                detalle={"razon": "no se pudo correr tesseract sobre la imagen"},
            )
        return _resolver_imagen(resultado_img, idioma, dimensiones)
    except Exception as exc:  # fail-soft: cualquier fallo inesperado (permisos, disco lleno, workspace remontado solo-lectura) sale como Resultado(estado="error"), nunca una excepcion cruda -- mismo tratamiento que D-1/I-5 en los hermanos
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR,
            version=_version() or "desconocida",
            detalle={"razon": f"fallo inesperado en OCR: {type(exc).__name__}"},
        )
