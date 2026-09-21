"""OCR para lo que no tiene capa de texto: PDF escaneado e imagen.

tesseract por subproceso. `spa` porque los documentos son en español
(verificado 2026-09-20: lee tildes y guion largo exacto).

El defecto propio de este extractor (que Excel y PDF no tienen): un umbral de
CARACTERES sólo no alcanza. Medido a mano antes de escribir el código
(2026-09-21, `tesseract 5.5.0`): una imagen con glifos sueltos al azar
("¡00*H*1|!%00$ ol aso 4 a0%%l % HoA", 35 caracteres) SUPERA cualquier umbral
razonable de longitud, y tesseract la "lee" igual -- lo que la delata es la
confianza por palabra que el propio tesseract calcula (modo `tsv`): ~96 % en
texto real, ~55 % en este ruido. Un documento financiero con números
("INGRESOS 1,234,567.89 USD") sigue en ~94 % de confianza -- así que el
umbral de confianza NO castiga contenido numérico legítimo, a diferencia de
una heurística por proporción de letras (que sí lo haría: esa misma frase
numérica tiene sólo 48 % de caracteres alfabéticos).

La confianza se lee de una SEGUNDA invocación a tesseract (modo `tsv`, sobre
el MISMO archivo e idioma) -- no es una segunda cuenta que pueda divergir de
la primera (el defecto C-2 de pdf.py): es la MISMA pasada de reconocimiento
de tesseract, sólo en otro formato de salida, leyendo un número que el modo
texto plano no expone.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from procesamiento.extractores import ocr

pytestmark = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract no instalado"
)
Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")


def _fuente(tamano: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", tamano
        )
    except OSError:
        return None


def _imagen_con_texto(destino: Path, texto: str) -> Path:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (900, 120), "white")
    d = ImageDraw.Draw(img)
    d.text((20, 40), texto, fill="black", font=_fuente(34))
    img.save(destino)
    return destino


def _imagen_con_ruido_glifos(destino: Path) -> Path:
    """Glifos sueltos al azar (no palabras) -- reproduce EXACTO el caso medido
    a mano: tesseract "lee" ~35 caracteres de esto, más que MINIMO_CARACTERES,
    pero con confianza promedio baja (~55 %, medido). Semilla fija: el
    fixture tiene que ser determinista, no una tirada de dados distinta en
    cada corrida de CI."""
    import random

    from PIL import Image, ImageDraw

    random.seed(3)
    img = Image.new("RGB", (900, 150), "white")
    d = ImageDraw.Draw(img)
    fuente = _fuente(34)
    caracteres = "aAo!I1l0O#$%&*"
    x = 20
    for _ in range(30):
        c = random.choice(caracteres)
        d.text((x, 40), c, fill="black", font=fuente)
        x += random.randint(15, 40)
    img.save(destino)
    return destino


def test_lee_texto_en_espanol_con_tildes(tmp_path: Path):
    origen = _imagen_con_texto(tmp_path / "a.png", "Estado de Situación Financiera")
    r = ocr.extraer(origen)
    assert r.estado == "ok"
    assert "Situación" in r.salidas["texto.txt"]


def test_lee_tildes_y_guion_largo_exacto(tmp_path: Path):
    """El caso EXACTO verificado el 2026-09-20 contra tesseract 5.5.0 real:
    tildes y guion largo (—), no sólo una palabra con acento."""
    texto = "Estado de Situación Financiera — año 2026"
    origen = _imagen_con_texto(tmp_path / "exacto.png", texto)
    r = ocr.extraer(origen)
    assert r.estado == "ok"
    assert texto in r.salidas["texto.txt"]


def test_una_imagen_en_blanco_no_se_declara_ok(tmp_path: Path):
    """Fallo cerrado: si el OCR no leyó nada, NO hay extracto."""
    from PIL import Image

    blanco = tmp_path / "blanco.png"
    Image.new("RGB", (400, 200), "white").save(blanco)
    r = ocr.extraer(blanco)
    assert r.estado == "error"
    assert r.salidas == {}


def test_ruido_ilegible_no_se_declara_ok(tmp_path: Path):
    """El defecto propio de OCR: tesseract puede devolver más de
    MINIMO_CARACTERES de puro ruido (glifos sueltos, no palabras) -- un
    umbral de longitud SOLO no lo detiene. Medido: esta imagen produce ~35
    caracteres (por encima de cualquier umbral razonable de longitud) con
    confianza promedio ~55 %, muy por debajo de un texto real (~96 %)."""
    origen = _imagen_con_ruido_glifos(tmp_path / "ruido.png")
    texto_crudo = subprocess.run(
        ["tesseract", str(origen), "stdout", "-l", "spa"],
        capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    assert len(texto_crudo) >= ocr.MINIMO_CARACTERES, (
        "el fixture de ruido tiene que superar el umbral de longitud por sí "
        f"solo para que el test pruebe la defensa de CONFIANZA, no la de "
        f"longitud -- tesseract devolvió {len(texto_crudo)!r} caracteres: "
        f"{texto_crudo!r}"
    )

    r = ocr.extraer(origen)
    assert r.estado == "error"
    assert r.salidas == {}


def test_texto_corto_por_debajo_del_minimo_no_se_declara_ok(tmp_path: Path):
    """Aísla MINIMO_CARACTERES de UMBRAL_CONFIANZA_PROMEDIO: "2026" mide
    confianza ~96 % (medido) -- pasaría el umbral de confianza sin problema
    -- pero son 4 caracteres, por debajo de MINIMO_CARACTERES (8). Sin este
    test, una mutación que borre sólo el umbral de longitud no la detecta
    ningún otro caso (el resto de los fixtures ya superan los 8 caracteres,
    y el de ruido superaba los 8 a propósito para probar la OTRA defensa)."""
    origen = _imagen_con_texto(tmp_path / "corto.png", "2026")
    r = ocr.extraer(origen)
    assert r.estado == "error"
    assert r.salidas == {}


def test_un_archivo_inexistente_da_error(tmp_path: Path):
    """El chequeo explícito de existencia es, en la práctica, redundante con
    el propio fallo de tesseract sobre un archivo que no existe (también da
    'error' sin él) -- lo que el chequeo aporta es una razón CLARA en vez
    del stderr crudo de tesseract, así que la aserción tiene que mirar la
    razón, no sólo el estado, para que una mutación que borre el chequeo
    caiga en este test."""
    r = ocr.extraer(tmp_path / "no-existe.png")
    assert r.estado == "error"
    assert r.salidas == {}
    assert "no existe el archivo" in r.detalle["razon"]


def test_tesseract_ausente_da_error_sin_excepcion(tmp_path: Path, monkeypatch):
    """tesseract no instalado (o no en PATH): 'error' con razón clara, nunca
    una excepción cruda escapando de este módulo."""
    origen = _imagen_con_texto(tmp_path / "a.png", "Estado de Situación Financiera")
    monkeypatch.setattr(ocr.shutil, "which", lambda _: None)
    r = ocr.extraer(origen)
    assert r.estado == "error"
    assert r.salidas == {}
    assert "razon" in r.detalle
