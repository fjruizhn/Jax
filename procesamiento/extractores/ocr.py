"""OCR para lo que no tiene capa de texto: PDF escaneado e imagen.

tesseract por subproceso, nunca por binding: un binding suma dependencia
nativa y no aporta nada acá. Se pasa por `spa` porque los documentos son en
español (verificado 2026-09-20: lee tildes y guion largo exacto).

El defecto propio de este extractor: un umbral de CARACTERES solo no basta
para distinguir texto real de ruido -- tesseract puede "leer" glifos sueltos
al azar y devolver más de `MINIMO_CARACTERES` de puro ruido (medido a mano,
2026-09-21: una imagen con 30 glifos sueltos produjo ~35 caracteres). Lo que
sí distingue ese caso es la confianza POR PALABRA que tesseract ya calcula
internamente (modo `tsv`, no expuesto en el modo texto plano): ~96 % en
texto real, ~55 % en el ruido medido. Se prefirió esto sobre una heurística
de "proporción de letras" porque esa heurística castigaría contenido
numérico legítimo -- un renglón como "INGRESOS 1,234,567.89 USD" tiene sólo
48 % de caracteres alfabéticos pero ~94 % de confianza de tesseract, y en un
extracto financiero los números SON el contenido.

La confianza se lee de una segunda invocación a tesseract (mismo archivo,
mismo idioma, modo `tsv`) -- no es una cuenta paralela que pueda divergir de
la que decide "hay texto o no" (el defecto C-2 de pdf.py, dos algoritmos
distintos para la misma pregunta): es la MISMA pasada de reconocimiento de
tesseract, sólo en otro formato de salida.

El número elegido para cada umbral, y por qué falla del lado barato: "no leí
nada" (falso negativo, se manda a revisar de más) es mucho más barato que
"leí ruido y lo di por bueno" (falso positivo, el modelo consumidor lo
inventa todo con la confianza de que es un extracto real). Ante la duda, los
dos umbrales suben, nunca bajan para que pase un fixture.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from procesamiento.resultado import Resultado

EXTRACTOR = "tesseract"

# Caracteres MÍNIMOS (tras `.strip()`) para que el modo texto plano cuente
# como "algo se leyó". Defensa barata contra el caso vacío/casi vacío (imagen
# en blanco, mancha, ruido puro que tesseract ni intenta leer). NO alcanza
# solo -- ver UMBRAL_CONFIANZA_PROMEDIO para el caso de ruido "leíble".
MINIMO_CARACTERES = 8

# Confianza PROMEDIO mínima (0-100, la que reporta tesseract por palabra en
# modo `tsv`) para aceptar el texto como real. Medido a mano 2026-09-21
# contra tesseract 5.5.0: texto real en español ~96 %, texto numérico real
# ~94 %, ruido de glifos sueltos ~55 %. 70 queda cómodo entre los dos casos
# reales medidos y el ruido medido -- ante la duda, sube, no baja para que
# pase un fixture.
UMBRAL_CONFIANZA_PROMEDIO = 70.0

TIMEOUT_SEGUNDOS = 300


def _version() -> str:
    try:
        salida = subprocess.run(
            ["tesseract", "--version"], capture_output=True, text=True, timeout=30
        )
        return salida.stdout.splitlines()[0].split()[-1]
    except Exception:
        return "desconocida"


def _confianza_promedio(origen: Path, idioma: str) -> tuple[float, int]:
    """Confianza promedio (0-100) de las palabras que tesseract reconoció, y
    cuántas palabras entraron en el promedio. Sin palabras reconocidas,
    devuelve (0.0, 0) -- mismo criterio de "no hay nada" que el texto vacío."""
    proceso = subprocess.run(
        ["tesseract", str(origen), "stdout", "-l", idioma, "tsv"],
        capture_output=True, text=True, timeout=TIMEOUT_SEGUNDOS,
    )
    confianzas: list[float] = []
    lineas = (proceso.stdout or "").splitlines()
    for linea in lineas[1:]:  # lineas[0] es el encabezado de columnas
        campos = linea.split("\t")
        if len(campos) < 12:
            continue
        conf_str, texto_palabra = campos[10], campos[11]
        if not texto_palabra.strip():
            continue
        try:
            conf = float(conf_str)
        except ValueError:
            continue
        if conf < 0:  # -1: fila estructural (página/bloque/línea), no palabra
            continue
        confianzas.append(conf)
    if not confianzas:
        return 0.0, 0
    return sum(confianzas) / len(confianzas), len(confianzas)


def extraer(origen: Path, idioma: str = "spa") -> Resultado:
    if shutil.which("tesseract") is None:
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR, version="ausente",
            detalle={"razon": "tesseract no esta instalado"},
        )
    if not Path(origen).is_file():
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR, version=_version(),
            detalle={"razon": f"no existe el archivo: {origen}"},
        )

    try:
        proceso = subprocess.run(
            ["tesseract", str(origen), "stdout", "-l", idioma],
            capture_output=True, text=True, timeout=TIMEOUT_SEGUNDOS,
        )
    except subprocess.TimeoutExpired:
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR, version=_version(),
            detalle={"razon": f"tesseract excedio {TIMEOUT_SEGUNDOS}s"},
        )
    except Exception as exc:
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR, version=_version(),
            detalle={"razon": f"no se pudo correr tesseract: {type(exc).__name__}: {exc}"},
        )

    texto = (proceso.stdout or "").strip()
    if proceso.returncode != 0 or len(texto) < MINIMO_CARACTERES:
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR, version=_version(),
            detalle={
                "razon": "el OCR no devolvio texto util",
                "returncode": proceso.returncode,
                "caracteres": len(texto),
                "stderr": (proceso.stderr or "")[:500],
            },
        )

    try:
        confianza, n_palabras = _confianza_promedio(origen, idioma)
    except subprocess.TimeoutExpired:
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR, version=_version(),
            detalle={"razon": f"tesseract (calculo de confianza) excedio {TIMEOUT_SEGUNDOS}s"},
        )

    if confianza < UMBRAL_CONFIANZA_PROMEDIO:
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR, version=_version(),
            detalle={
                "razon": "el OCR devolvio texto de baja confianza (probable ruido)",
                "confianza_promedio": round(confianza, 2),
                "palabras": n_palabras,
                "caracteres": len(texto),
            },
        )

    return Resultado(
        estado="ok", salidas={"texto.txt": texto},
        extractor=EXTRACTOR, version=_version(),
        detalle={
            "idioma": idioma,
            "caracteres": len(texto),
            "confianza_promedio": round(confianza, 2),
            "palabras": n_palabras,
        },
    )
