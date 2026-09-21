"""OCR para lo que no tiene capa de texto: PDF escaneado e imagen.

Ronda de arreglo 1 (2026-09-21, task-5-6-hallazgos.md — NO ratificado, 3
críticos + 3 importantes sobre este archivo):

- C-3: tesseract NO lee PDF -- se rasteriza con `pdftoppm` (poppler) y se
  reporta por página, igual que `pdf.py`.
- C-1: el promedio de confianza diluye -- un piso POR PALABRA
  (`CONFIANZA_MINIMA_PALABRA`) nombra las palabras dudosas en
  `detalle["palabras_dudosas"]` en vez de esconderlas detrás de un promedio.
- C-2: una página casi en blanco con sólo un membrete (pocas palabras, alta
  confianza) fuerza `parcial` (`MINIMO_PALABRAS`), nunca `error`.
- I-1: la confianza promedio se registra SIEMPRE en `detalle`, incluso
  cuando el texto es demasiado corto para pasar `MINIMO_CARACTERES`.
- I-3: cada umbral (`MINIMO_CARACTERES`, `CONFIANZA_MINIMA_PALABRA`,
  `MINIMO_PALABRAS`, `PROPORCION_MAXIMA_PALABRAS_DUDOSAS`) tiene sus propios
  tests que lo fijan -- la mayoría, contra `_clasificar()` y
  `_analizar_tsv()` directamente (funciones PURAS, sin invocar a tesseract:
  mismo patrón que `pdf._tabla_a_bloque`), porque un valor de confianza
  exacto no se puede forzar de forma confiable renderizando una imagen.
"""
import json
import shutil
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


def _imagen_una_linea(destino: Path, texto: str, size=(900, 120)) -> Path:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    d.text((20, 40), texto, fill="black", font=_fuente(34))
    img.save(destino)
    return destino


def _imagen_multilinea(destino: Path, lineas: list[str], size=(1100, 450)) -> Path:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    fuente = _fuente(30)
    y = 30
    for linea in lineas:
        d.text((20, y), linea, fill="black", font=fuente)
        y += 50
    img.save(destino)
    return destino


def _imagen_mixta_real_y_ruido(
    destino: Path, lineas_reales: list[str], n_lineas_ruido: int,
    size=(1100, 500), seed: int = 3,
) -> Path:
    """C-1: renglones financieros reales intercalados con líneas de glifos
    sueltos al azar (no palabras) -- mismo fixture de ruido que ya probó no
    ser detenido por un umbral de longitud solo."""
    import random

    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    fuente = _fuente(30)
    y = 20
    for linea in lineas_reales:
        d.text((20, y), linea, fill="black", font=fuente)
        y += 45
    random.seed(seed)
    caracteres = "aAo!I1l0O#$%&*"
    for _ in range(n_lineas_ruido):
        x = 20
        for _ in range(6):
            c = random.choice(caracteres)
            d.text((x, y), c, fill="black", font=fuente)
            x += random.randint(15, 40)
        y += 45
    img.save(destino)
    return destino


# ---------------------------------------------------------------------------
# Integración: imagen suelta (con tesseract real)
# ---------------------------------------------------------------------------


def test_lee_texto_en_espanol_con_tildes_y_guion_largo(tmp_path: Path):
    """Caso feliz: suficientes palabras (>= MINIMO_PALABRAS), sin ninguna
    dudosa -- tiene que ser 'ok', y el texto exacto (tildes, guion largo)
    sobrevive verbatim."""
    lineas = [
        "Estado de Situación Financiera — año 2026",
        "Activos totales 1,234,567.89 USD",
        "Pasivos totales 987,654.32 USD",
        "Patrimonio neto 246,913.57 USD",
    ]
    r = ocr.extraer(_imagen_multilinea(tmp_path / "tildes.png", lineas))
    assert r.estado == "ok"
    assert "Estado de Situación Financiera — año 2026" in r.salidas["texto.txt"]
    assert "palabras_dudosas" not in r.detalle


def test_una_imagen_en_blanco_no_se_declara_ok(tmp_path: Path):
    """Fallo cerrado: si el OCR no leyó nada, NO hay extracto."""
    from PIL import Image

    blanco = tmp_path / "blanco.png"
    Image.new("RGB", (400, 200), "white").save(blanco)
    r = ocr.extraer(blanco)
    assert r.estado == "error"
    assert r.salidas == {}


def test_ruido_con_mayoria_de_palabras_dudosas_da_error(tmp_path: Path):
    """30 glifos sueltos al azar (no palabras): más de la mitad de las
    palabras que tesseract "reconoce" caen por debajo de
    CONFIANZA_MINIMA_PALABRA -- eso es 'sin texto útil' aunque el texto
    plano tenga más caracteres que MINIMO_CARACTERES."""
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
    origen = tmp_path / "ruido.png"
    img.save(origen)

    r = ocr.extraer(origen)
    assert r.estado == "error"
    assert r.salidas == {}
    assert "palabras_dudosas" in r.detalle
    assert "confianza baja" in r.detalle["razon"]


def test_texto_real_mezclado_con_ruido_da_parcial_con_palabras_dudosas_nombradas(
    tmp_path: Path,
):
    """C-1, el caso EXACTO del hallazgo: 6 renglones financieros reales +
    4 líneas de ruido. Con el PROMEDIO viejo esto pasaba como 'ok' (medido
    por el auditor: 77,73 de confianza promedio, por encima del umbral
    viejo de 70). Con el piso por palabra, las palabras de ruido se nombran
    en `detalle["palabras_dudosas"]` SIN borrarlas del texto -- y como no
    son mayoría, el resultado es 'parcial', no 'error': el documento SÍ
    tiene contenido real rescatable."""
    reales = [
        "Estado de Situación Financiera",
        "Activos totales 1,234,567.89 USD",
        "Pasivos totales 987,654.32 USD",
        "Patrimonio neto 246,913.57 USD",
        "Periodo fiscal 2026",
        "Auditado por Contadores Asociados",
    ]
    origen = _imagen_mixta_real_y_ruido(tmp_path / "mixta.png", reales, 4)

    r = ocr.extraer(origen)

    assert r.estado == "parcial"
    assert "palabras_dudosas" in r.detalle
    assert len(r.detalle["palabras_dudosas"]) >= 1
    # las palabras dudosas se NOMBRAN -- cada una trae su propio texto y su
    # propia confianza, no un simple conteo.
    for dudosa in r.detalle["palabras_dudosas"]:
        assert "palabra" in dudosa and "confianza" in dudosa
        assert dudosa["confianza"] < ocr.CONFIANZA_MINIMA_PALABRA
    # el contenido real SIGUE presente -- 'parcial' no tira el dato.
    assert "Estado de Situación Financiera" in r.salidas["texto.txt"]
    assert "Patrimonio neto 246,913.57 USD" in r.salidas["texto.txt"]
    # y el propio JSON round-trip de Resultado ya lo verifica, pero lo
    # confirmamos explícito acá porque es justo lo que exige la evidencia.
    json.dumps(dict(r.detalle))


def test_membrete_con_pocas_palabras_da_parcial_con_dimensiones(tmp_path: Path):
    """C-2, el caso EXACTO del hallazgo: una imagen grande (recorte de
    página) con sólo un membrete corto legible -- alta confianza, pocas
    palabras. 'parcial', nunca 'error' (un recibo legítimo puede tener
    pocas palabras), con la razón y las dimensiones de la imagen."""
    origen = _imagen_una_linea(
        tmp_path / "membrete.png", "CONTADORES ASOCIADOS S.A.", size=(1200, 1600)
    )
    r = ocr.extraer(origen)
    assert r.estado == "parcial"
    assert "cobertura insuficiente" in r.detalle["razon"]
    assert r.detalle["ancho"] == 1200
    assert r.detalle["alto"] == 1600


def test_texto_corto_registra_confianza_promedio_igual(tmp_path: Path):
    """I-1: `BALANCE` (7 caracteres) cae por MINIMO_CARACTERES y sale
    'error' -- pero tesseract SÍ corrió y SÍ midió una confianza (~96 %), y
    esa franja de "texto corto" es justo la que hace falta para calibrar
    los umbrales. La confianza tiene que registrarse pase o no pase."""
    origen = _imagen_una_linea(tmp_path / "balance.png", "BALANCE")
    r = ocr.extraer(origen)
    assert r.estado == "error"
    assert "confianza_promedio" in r.detalle
    assert r.detalle["confianza_promedio"] > 0


def test_texto_por_debajo_del_minimo_de_caracteres_no_se_declara_ok(tmp_path: Path):
    """Pin de MINIMO_CARACTERES por el lado bajo: "Vencido" son 7
    caracteres (por debajo de 8) con alta confianza y NO forma mayoría
    dudosa -- si esto no fuera 'error', la única explicación sería que
    MINIMO_CARACTERES bajó."""
    origen = _imagen_una_linea(tmp_path / "vencido.png", "Vencido")
    r = ocr.extraer(origen)
    assert r.estado == "error"
    assert r.detalle["caracteres"] == 7


def test_un_archivo_inexistente_da_error(tmp_path: Path):
    r = ocr.extraer(tmp_path / "no-existe.png")
    assert r.estado == "error"
    assert r.salidas == {}
    assert "no existe el archivo" in r.detalle["razon"]


def test_tesseract_ausente_da_error_sin_excepcion(tmp_path: Path, monkeypatch):
    origen = _imagen_una_linea(tmp_path / "a.png", "Estado de Situación Financiera")
    monkeypatch.setattr(ocr.shutil, "which", lambda _: None)
    r = ocr.extraer(origen)
    assert r.estado == "error"
    assert r.salidas == {}
    assert "razon" in r.detalle


# ---------------------------------------------------------------------------
# Integración: PDF escaneado, vía pdftoppm (C-3)
# ---------------------------------------------------------------------------


def _pdf_de_imagenes(destino: Path, paginas: list) -> Path:
    paginas[0].save(destino, save_all=True, append_images=paginas[1:])
    return destino


def test_pdf_escaneado_multipagina_se_lee_de_punta_a_punta(tmp_path: Path):
    """C-3, evidencia exigida: un PDF escaneado real (rasterizado con
    `pdftoppm`, no un binding) de dos páginas, cada una con contenido
    financiero real y suficientes palabras -- 'ok', con las dos páginas
    presentes y en orden."""
    p1 = _imagen_multilinea(
        tmp_path / "p1.png",
        [
            "Estado de Situación Financiera",
            "Activos totales 1,234,567.89 USD",
            "Pasivos totales 987,654.32 USD",
            "Patrimonio neto 246,913.57 USD",
        ],
    )
    p2 = _imagen_multilinea(
        tmp_path / "p2.png",
        [
            "Estado de Resultados",
            "Ingresos totales 500,000.00 USD",
            "Costos totales 300,000.00 USD",
            "Utilidad neta 200,000.00 USD",
        ],
    )
    from PIL import Image

    origen = _pdf_de_imagenes(
        tmp_path / "escaneado.pdf", [Image.open(p1), Image.open(p2)]
    )

    r = ocr.extraer(origen)

    assert r.estado == "ok"
    assert r.detalle["paginas"] == 2
    md = r.salidas["texto.txt"]
    assert "<!-- página 1 -->" in md
    assert "<!-- página 2 -->" in md
    assert md.index("Situación Financiera") < md.index("Estado de Resultados")
    assert "Utilidad neta 200,000.00 USD" in md


def test_pdf_con_una_pagina_en_blanco_da_parcial(tmp_path: Path):
    """C-3: una página con contenido real, otra pura imagen en blanco (un
    escaneo con una hoja de más) -- 'parcial', con la página vacía nombrada,
    igual que hace `pdf.py` con `paginas_sin_texto`."""
    from PIL import Image

    p1 = _imagen_multilinea(
        tmp_path / "p1.png",
        [
            "Estado de Situación Financiera",
            "Activos totales 1,234,567.89 USD",
            "Pasivos totales 987,654.32 USD",
            "Patrimonio neto 246,913.57 USD",
        ],
    )
    blanco = Image.new("RGB", (1100, 450), "white")
    origen = _pdf_de_imagenes(tmp_path / "hibrido.pdf", [Image.open(p1), blanco])

    r = ocr.extraer(origen)

    assert r.estado == "parcial"
    assert r.detalle["paginas"] == 2
    assert r.detalle["paginas_sin_texto"] == [2]
    assert "Estado de Situación Financiera" in r.salidas["texto.txt"]


def test_pdf_pagina_con_membrete_corto_es_parcial_con_paginas_con_dudas(tmp_path: Path):
    """Hueco de cobertura (task-7, 2026-09-21): desactivar la propagación de
    `con_dudas` de una página al resultado agregado del PDF (en
    `_resolver_pdf`, la línea `paginas_con_dudas.append(numero)`) dejaba la
    suite ENTERA en verde -- ningún test la ejercitaba. Primera página con
    contenido completo (>= MINIMO_PALABRAS, sin dudosas) -> 'ok' por sí sola.
    Segunda página, sólo un membrete corto de alta confianza (< MINIMO_
    PALABRAS, mismo caso EXACTO de C-2 / `test_membrete_con_pocas_
    palabras_da_parcial_con_dimensiones`, pero acá dentro de un PDF) ->
    'con_dudas' por sí sola. El agregado del PDF tiene que ser 'parcial' con
    `detalle['paginas_con_dudas'] == [2]` -- sin la propagación, ninguna de
    las dos páginas cuenta como dudosa ni sin texto, y el agregado sale
    (incorrectamente) 'ok'."""
    p1 = _imagen_multilinea(
        tmp_path / "p1.png",
        [
            "Estado de Situación Financiera",
            "Activos totales 1,234,567.89 USD",
            "Pasivos totales 987,654.32 USD",
            "Patrimonio neto 246,913.57 USD",
        ],
    )
    p2 = _imagen_una_linea(
        tmp_path / "p2.png", "CONTADORES ASOCIADOS S.A.", size=(1200, 1600)
    )
    from PIL import Image

    origen = _pdf_de_imagenes(
        tmp_path / "membrete.pdf", [Image.open(p1), Image.open(p2)]
    )

    r = ocr.extraer(origen)

    assert r.estado == "parcial"
    assert r.detalle["paginas"] == 2
    assert r.detalle["paginas_con_dudas"] == [2]
    assert "paginas_sin_texto" not in r.detalle
    assert "Estado de Situación Financiera" in r.salidas["texto.txt"]


def test_pdf_donde_ninguna_pagina_da_texto_es_error(tmp_path: Path):
    """C-3: escaneo puro imagen, sin ninguna página legible -- 'error', no
    'ok' con un extracto vacío."""
    from PIL import Image

    b1 = Image.new("RGB", (1100, 450), "white")
    b2 = Image.new("RGB", (1100, 450), "white")
    origen = _pdf_de_imagenes(tmp_path / "todo-blanco.pdf", [b1, b2])

    r = ocr.extraer(origen)

    assert r.estado == "error"
    assert r.salidas == {}
    assert r.detalle["paginas"] == 2


def test_pdftoppm_ausente_da_error_sin_excepcion(tmp_path: Path, monkeypatch):
    origen = tmp_path / "algo.pdf"
    origen.write_bytes(b"%PDF-1.4\n%%EOF")
    real_which = ocr.shutil.which
    monkeypatch.setattr(
        ocr.shutil, "which", lambda cmd: None if cmd == "pdftoppm" else real_which(cmd)
    )
    r = ocr.extraer(origen)
    assert r.estado == "error"
    assert r.salidas == {}
    assert "pdftoppm" in r.detalle["razon"]


# ---------------------------------------------------------------------------
# Unidad: `_clasificar()` -- función PURA, sin tesseract. Fija cada umbral
# con un valor hardcodeado (no leído de `ocr.CONSTANTE`) para que mover la
# constante SÍ cambie el resultado del test (I-3).
# ---------------------------------------------------------------------------


def test_clasificar_bajo_el_minimo_de_caracteres_es_sin_texto():
    assert ocr._clasificar(7, {"n_palabras": 20, "palabras_dudosas": []}) == "sin_texto"


def test_clasificar_en_el_minimo_de_caracteres_no_es_sin_texto():
    assert ocr._clasificar(8, {"n_palabras": 20, "palabras_dudosas": []}) != "sin_texto"


def test_clasificar_mas_de_la_mitad_de_palabras_dudosas_es_sin_texto():
    dudosas = [{"palabra": "x", "confianza": 1.0}] * 6  # 6 de 10 = 60 %
    assert ocr._clasificar(100, {"n_palabras": 10, "palabras_dudosas": dudosas}) == "sin_texto"


def test_clasificar_exactamente_la_mitad_de_palabras_dudosas_no_es_sin_texto():
    dudosas = [{"palabra": "x", "confianza": 1.0}] * 5  # 5 de 10 = 50 % exacto
    assert ocr._clasificar(100, {"n_palabras": 10, "palabras_dudosas": dudosas}) != "sin_texto"


def test_clasificar_bajo_el_minimo_de_palabras_es_con_dudas():
    assert ocr._clasificar(100, {"n_palabras": 9, "palabras_dudosas": []}) == "con_dudas"


def test_clasificar_en_el_minimo_de_palabras_sin_dudas_es_ok():
    assert ocr._clasificar(100, {"n_palabras": 10, "palabras_dudosas": []}) == "ok"


def test_clasificar_con_alguna_palabra_dudosa_sin_ser_mayoria_es_con_dudas():
    dudosas = [{"palabra": "x", "confianza": 1.0}]  # 1 de 20 = 5 %
    assert ocr._clasificar(100, {"n_palabras": 20, "palabras_dudosas": dudosas}) == "con_dudas"


# ---------------------------------------------------------------------------
# Unidad: `_analizar_tsv()` -- función PURA sobre texto `tsv` armado a
# mano, sin invocar a tesseract. Mismo patrón que `pdf._tabla_a_bloque`.
# ---------------------------------------------------------------------------


def _tsv(filas_palabra: list[tuple], ancho: int = 500, alto: int = 300) -> str:
    """Arma un `tsv` sintético: encabezado + una fila de nivel 1 (página,
    con ancho/alto) + una fila de nivel 5 (palabra) por cada
    `(confianza, texto)` de `filas_palabra`."""
    lineas = [
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
        f"1\t1\t0\t0\t0\t0\t0\t0\t{ancho}\t{alto}\t-1\t",
    ]
    for i, (conf, texto) in enumerate(filas_palabra, start=1):
        lineas.append(f"5\t1\t1\t1\t1\t{i}\t0\t0\t10\t10\t{conf}\t{texto}")
    return "\n".join(lineas)


def test_analizar_tsv_lee_ancho_y_alto_de_la_fila_de_nivel_1():
    analisis = ocr._analizar_tsv(_tsv([(90.0, "hola")], ancho=1200, alto=1600))
    assert analisis["ancho"] == 1200
    assert analisis["alto"] == 1600


def test_analizar_tsv_confianza_exacta_en_el_piso_no_es_dudosa():
    analisis = ocr._analizar_tsv(_tsv([(60.0, "bien")]))
    assert analisis["palabras_dudosas"] == []


def test_analizar_tsv_confianza_justo_debajo_del_piso_es_dudosa():
    analisis = ocr._analizar_tsv(_tsv([(59.99, "mal")]))
    assert len(analisis["palabras_dudosas"]) == 1
    assert analisis["palabras_dudosas"][0]["palabra"] == "mal"


def test_analizar_tsv_promedio_es_sobre_TODAS_las_palabras_no_las_primeras_cinco():
    """Mutación específica que el auditor pidió matar: promediar sólo las
    primeras 5 palabras. Acá hay 7: cinco a 100 y dos a 0 -- el promedio
    real es 500/7 = 71,43; el de "sólo las 5 primeras" sería 100,0. Tienen
    que diferir para que el test pueda fallar."""
    filas = [(100.0, f"buena{i}") for i in range(5)] + [(0.0, "mala1"), (0.0, "mala2")]
    analisis = ocr._analizar_tsv(_tsv(filas))
    assert analisis["n_palabras"] == 7
    assert analisis["confianza_promedio"] == round(500 / 7, 2)


def test_analizar_tsv_fila_con_confianza_negativa_no_cuenta_como_palabra():
    """Las filas estructurales (bloque/párrafo/línea) traen `conf=-1` y
    texto vacío en tesseract real -- pero el guard tiene que sostenerse
    aunque una fila con conf=-1 traiga texto (fila corrupta/anómala)."""
    tsv = _tsv([(60.0, "real")]) + "\n5\t1\t1\t1\t1\t2\t0\t0\t10\t10\t-1\tfantasma"
    analisis = ocr._analizar_tsv(tsv)
    assert analisis["n_palabras"] == 1
    assert all(d["palabra"] != "fantasma" for d in analisis["palabras_dudosas"])


def test_analizar_tsv_linea_mal_formada_no_revienta_ni_se_cuenta():
    tsv = _tsv([(60.0, "real")]) + "\ncampo_incompleto\tsin\tsuficientes\tcolumnas"
    analisis = ocr._analizar_tsv(tsv)  # no debe lanzar excepción
    assert analisis["n_palabras"] == 1


def test_analizar_tsv_palabra_de_solo_espacios_no_cuenta():
    tsv = _tsv([(60.0, "real")]) + "\n5\t1\t1\t1\t1\t2\t0\t0\t10\t10\t45.0\t   "
    analisis = ocr._analizar_tsv(tsv)
    assert analisis["n_palabras"] == 1


# ---------------------------------------------------------------------------
# Unidad: `_ocr_una_imagen()` -- guard de returncode, con subprocess mockeado
# ---------------------------------------------------------------------------


def test_returncode_distinto_de_cero_no_se_toma_como_texto_valido(tmp_path, monkeypatch):
    """Aunque el stdout tenga contenido (más que MINIMO_CARACTERES), un
    `returncode` distinto de cero tiene que hacer fallar la lectura -- sin
    este guard, tesseract imprimiendo una advertencia y saliendo con error
    se tomaría igual como un extracto real."""

    class Falso:
        returncode = 1
        stdout = "esto no es un extracto de verdad, tiene mas de ocho caracteres"
        stderr = "algo salio mal"

    def fake_run(cmd, **kwargs):
        return Falso()

    monkeypatch.setattr(ocr.subprocess, "run", fake_run)
    origen = tmp_path / "cualquiera.png"
    origen.write_bytes(b"no importa el contenido, run esta mockeado")

    resultado = ocr._ocr_una_imagen(origen, "spa")

    assert resultado is None
