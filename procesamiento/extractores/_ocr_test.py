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


@pytest.fixture(autouse=True)
def _workspace_dir_es_tmp(tmp_path, monkeypatch):
    """Hallazgo PROPIO destapado verificando MAJOR 2 en Docker SIN root
    (la verificación anterior corría como root, que bypasea permisos de
    filesystem y tapaba esto): el default de `_WORKSPACE_DIR`
    (`/home/fruiz/jax-workspace`, I-7) es específico de ESTA máquina. En
    un runner de CI real (usuario `runner`, sin `/home/fruiz` y sin
    permiso para crearlo bajo `/home`, que no es escribible por un
    usuario sin privilegios) el `mkdir(parents=True)` de `extraer()`
    revienta con `PermissionError` -- que esta misma ronda ya atrapa
    (MAJOR 2), pero eso vuelve `'error'` CUALQUIER test de OCR sobre PDF
    que no controle el workspace, sin que tesseract/pdftoppm lleguen a
    correr. Se parchea a un tempdir por test -- mismo patrón que
    `_ingesta_test.py` con `tool_authority.WORKSPACE_ROOT`. Los dos tests
    que necesitan control fino (`test_workspace_no_escribible_...`,
    `test_rasterizado_usa_workspace_dir_no_tmp`) lo sobreescriben en su
    propio cuerpo, que corre DESPUÉS de este fixture."""
    monkeypatch.setattr(ocr, "_WORKSPACE_DIR", tmp_path / "workspace")


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


def test_una_imagen_en_blanco_es_un_documento_valido_sin_texto(tmp_path: Path):
    """Decision de Fernando (2026-10-03): una IMAGEN sin texto util (foto,
    plano, pasto) es un documento valido, no un error. `ok` con el codigo
    estable `imagen_sin_texto`, y la unica salida es un aviso explicito --
    nunca texto inventado ni la basura del OCR."""
    from PIL import Image

    blanco = tmp_path / "blanco.png"
    Image.new("RGB", (400, 200), "white").save(blanco)
    r = ocr.extraer(blanco)
    assert r.estado == "ok"
    assert r.detalle["codigo"] == "imagen_sin_texto"
    assert r.detalle["razon"] == "el OCR no devolvio texto util"
    # regla (C): el aviso dice SOLO que el OCR no encontro texto
    assert r.salidas["texto.txt"] == "<!-- el OCR no encontró texto -->"
    assert "foto" not in r.salidas["texto.txt"]


def test_una_imagen_con_texto_claro_no_lleva_codigo_imagen_sin_texto(tmp_path: Path):
    r = ocr.extraer(_imagen_multilinea(tmp_path / "claro.png", [
        "Estado de Situación Financiera — año 2026",
        "Activos totales 1,234,567.89 USD",
        "Pasivos totales 987,654.32 USD",
        "Patrimonio neto 246,913.57 USD",
    ]))
    assert r.estado == "ok"
    assert "codigo" not in r.detalle
    assert "no encontró texto" not in r.salidas["texto.txt"]


def test_un_jpeg_truncado_es_archivo_ilegible_y_error(tmp_path: Path):
    """`error` queda SOLO para un archivo danado o que no se puede abrir."""
    import random

    from PIL import Image

    random.seed(1)
    img = Image.new("L", (600, 400))
    img.putdata([random.randint(0, 255) for _ in range(600 * 400)])
    completo = tmp_path / "completo.jpg"
    img.save(completo, quality=90)
    datos = completo.read_bytes()
    roto = tmp_path / "roto.jpg"
    roto.write_bytes(datos[: len(datos) // 3])

    r = ocr.extraer(roto)

    assert r.estado == "error"
    assert r.salidas == {}
    assert r.detalle["codigo"] == "archivo_ilegible"
    assert r.detalle["razon"]


def test_bytes_que_no_son_una_imagen_son_archivo_ilegible(tmp_path: Path):
    falso = tmp_path / "falso.png"
    falso.write_bytes(b"esto no es un png")
    r = ocr.extraer(falso)
    assert r.estado == "error"
    assert r.detalle["codigo"] == "archivo_ilegible"


def test_un_pdf_escaneado_sin_texto_sigue_en_error_sin_codigo_de_imagen(tmp_path: Path):
    """La regla de `imagen_sin_texto` es SOLO de imagenes: un PDF sin texto
    es un problema y conserva su `error` y su razon."""
    from PIL import Image

    origen = _pdf_de_imagenes(
        tmp_path / "vacio.pdf", [Image.new("RGB", (800, 400), "white")]
    )
    r = ocr.extraer(origen)
    assert r.estado == "error"
    assert r.salidas == {}
    assert r.detalle["razon"] == "ninguna pagina del PDF dio texto util via OCR"
    assert "codigo" not in r.detalle


def test_la_cache_reusa_una_imagen_sin_texto(tmp_path: Path, monkeypatch):
    """Una imagen sin texto es un resultado VALIDO (`ok`), asi que la
    ingesta lo cachea: la segunda pasada no vuelve a llamar al extractor."""
    from motor_registry import tool_authority

    from procesamiento import compuerta, ingesta

    raiz = tmp_path.resolve()
    monkeypatch.setattr(tool_authority, "WORKSPACE_ROOT", raiz)
    from PIL import Image

    origen = tmp_path / "foto.png"
    Image.new("RGB", (400, 200), "white").save(origen)
    trabajo = raiz / "trabajo"

    f1 = ingesta.ingerir(origen, trabajo)
    assert f1.estado == "ok"
    assert f1.detalle["codigo"] == "imagen_sin_texto"

    llamadas = []
    original = compuerta.extraer
    monkeypatch.setattr(
        compuerta, "extraer", lambda *a, **k: llamadas.append(a) or original(*a, **k)
    )
    f2 = ingesta.ingerir(origen, trabajo)
    assert llamadas == []
    assert f2.estado == "ok"
    assert f2.detalle["codigo"] == "imagen_sin_texto"


def test_ruido_con_mayoria_de_palabras_dudosas_es_parcial_texto_dudoso(tmp_path: Path):
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
    # regla (B): mayoria de dudosas -> parcial, CONSERVANDO el texto leido
    assert r.estado == "parcial"
    assert r.detalle["codigo"] == "imagen_texto_dudoso"
    assert "texto de baja confianza" in r.salidas["texto.txt"]
    assert len(r.salidas["texto.txt"].splitlines()) >= 2  # nota + texto leido
    assert "palabras_dudosas" in r.detalle
    assert "texto de baja confianza" in r.detalle["razon"]


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
    assert r.estado == "ok"
    assert r.detalle["codigo"] == "imagen_sin_texto"
    assert "confianza_promedio" in r.detalle
    assert r.detalle["confianza_promedio"] > 0


def test_texto_por_debajo_del_minimo_de_caracteres_no_se_declara_ok(tmp_path: Path):
    """Pin de MINIMO_CARACTERES por el lado bajo: "Vencido" son 7
    caracteres (por debajo de 8) con alta confianza y NO forma mayoría
    dudosa -- si esto no fuera `imagen_sin_texto`, la única explicación sería que
    MINIMO_CARACTERES bajó."""
    origen = _imagen_una_linea(tmp_path / "vencido.png", "Vencido")
    r = ocr.extraer(origen)
    assert r.estado == "ok"
    assert r.detalle["codigo"] == "imagen_sin_texto"
    assert r.detalle["caracteres"] == 7


def test_un_archivo_inexistente_da_error(tmp_path: Path):
    r = ocr.extraer(tmp_path / "no-existe.png")
    assert r.estado == "error"
    assert r.salidas == {}
    assert "no existe el archivo" in r.detalle["razon"]


def test_tesseract_ausente_da_sin_extractor(tmp_path: Path, monkeypatch):
    """I-8 (final-hallazgos.md, ronda de cierre): antes era 'error',
    indistinguible de "este documento es ilegible". El spec §8 mapea "sin
    extractor disponible" a 'sin_extractor', igual que `openpyxl`/
    `python-docx`/`pdfplumber` ausentes -- una máquina sin tesseract no
    puede marcar TODOS sus escaneos como si el documento fuera el
    problema."""
    origen = _imagen_una_linea(tmp_path / "a.png", "Estado de Situación Financiera")
    monkeypatch.setattr(ocr.shutil, "which", lambda _: None)
    r = ocr.extraer(origen)
    assert r.estado == "sin_extractor"
    assert r.salidas == {}
    assert "razon" in r.detalle


def test_version_no_revienta_si_tesseract_falla(tmp_path: Path, monkeypatch):
    """Menor 10 (final-hallazgos.md, ronda de cierre): el sentinel de "no
    se pudo determinar la version" es `None`, NUNCA la cadena "desconocida"
    -- esa cadena compara IGUAL A SÍ MISMA en dos fallos consecutivos, y
    `ingesta._version_vigente` la reenvía tal cual para decidir si el
    caché sigue siendo válido (I-2)."""

    def _rota(cmd, **kwargs):
        raise OSError("tesseract --version fallo")

    monkeypatch.setattr(ocr.subprocess, "run", _rota)
    assert ocr._version() is None


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


def test_pdf_registra_confianza_promedio_del_documento_y_por_pagina(tmp_path: Path):
    """I-3 (final-hallazgos.md, ronda de cierre): `_resolver_imagen` YA
    registraba `confianza_promedio` (I-1 de la ronda 1); `_resolver_pdf`
    armaba su propio `detalle` a mano y NUNCA la incluía -- el PDF
    escaneado es el caso de uso real (7 de 23 documentos medidos, los 7
    'parcial'), y el pendiente de calibrar el umbral del OCR se quedaba
    sin el dato del único camino que necesita calibrarse."""
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
    assert "confianza_promedio" in r.detalle
    assert r.detalle["confianza_promedio"] > 0
    assert set(r.detalle["confianza_por_pagina"]) == {"1", "2"}, (
        "las claves tienen que ser STRING -- Ficha exige que detalle "
        "sobreviva un viaje a JSON, que no admite claves no-str"
    )
    for confianza in r.detalle["confianza_por_pagina"].values():
        assert confianza > 0


def test_pdf_detalle_sobrevive_construccion_de_ficha(tmp_path: Path):
    """Regresión propia de la ronda de cierre (no está en
    final-hallazgos.md -- la destapó verificar I-3 contra `Ficha`, no
    contra `Resultado` solo): `confianza_por_pagina` con claves `int`
    rompía `Ficha(...)` con un `ValueError` -- `Ficha` exige que `detalle`
    sobreviva un viaje REAL a JSON y vuelta sin cambios (`ficha.py`, I-6 de
    su propia ronda), y JSON no admite claves que no sean string. Con
    claves `int`, CUALQUIER ingesta de un PDF escaneado de más de una
    página hubiera reventado en cuanto `ingesta.ingerir()` intentara
    construir la ficha -- un defecto que ningún test contra `Resultado`
    solo (como el de arriba) podía ver."""
    from procesamiento.ficha import Ficha

    p1 = _imagen_multilinea(
        tmp_path / "p1.png",
        ["Estado de Situación Financiera", "Activos totales 1,234,567.89 USD"],
    )
    p2 = _imagen_multilinea(
        tmp_path / "p2.png",
        ["Estado de Resultados", "Ingresos totales 500,000.00 USD"],
    )
    from PIL import Image

    origen = _pdf_de_imagenes(
        tmp_path / "escaneado.pdf", [Image.open(p1), Image.open(p2)]
    )

    r = ocr.extraer(origen)

    Ficha(
        sha256="a" * 64, origen="fuente/escaneado.pdf",
        extractor=r.extractor, extractor_version=r.version,
        fecha="2026-09-21T00:00:00-06:00", estado=r.estado,
        detalle=dict(r.detalle),
    )  # no debe lanzar ValueError


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


def test_rasterizado_usa_workspace_dir_no_tmp(tmp_path: Path, monkeypatch):
    """I-7 (final-hallazgos.md, ronda de cierre): el directorio temporal
    del rasterizado va bajo `JAX_WORKSPACE_DIR`, no al default de
    `tempfile` (`/tmp`, que en hall9000 es tmpfs -- RAM, en un hipervisor
    con dos VMs). Se verifica interceptando `_rasterizar_pdf` (que recibe
    el `Path` del directorio temporal ya creado) y comprobando que cuelga
    de `_WORKSPACE_DIR`, no de `tempfile.gettempdir()`."""
    workspace = tmp_path / "workspace-ocr"
    monkeypatch.setattr(ocr, "_WORKSPACE_DIR", workspace)

    rutas_vistas: list[Path] = []
    original = ocr._rasterizar_pdf

    def _espia(origen, destino):
        rutas_vistas.append(destino)
        return original(origen, destino)

    monkeypatch.setattr(ocr, "_rasterizar_pdf", _espia)

    from PIL import Image

    p1 = _imagen_una_linea(tmp_path / "p1.png", "CONTADORES ASOCIADOS S.A.")
    origen = _pdf_de_imagenes(tmp_path / "escaneado.pdf", [Image.open(p1)])
    ocr.extraer(origen)

    assert rutas_vistas, "no se llamo a _rasterizar_pdf"
    assert workspace in rutas_vistas[0].parents, (
        f"el directorio temporal ({rutas_vistas[0]}) no cuelga de "
        f"_WORKSPACE_DIR ({workspace}) -- sigue yendo a /tmp"
    )
    assert workspace.is_dir(), "_WORKSPACE_DIR no se creo antes de rasterizar"


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


# ---------------------------------------------------------------------------
# MAJOR 2 (final-hallazgos.md, adenda 2026-09-21, ruling de Fernando): I-7
# agregó `mkdir()`/`TemporaryDirectory(dir=...)` SIN guarda en una función
# que no tenía `try` propio -- cuarta aparición de la familia I-5,
# introducida por la MISMA ronda que vino a cerrarla.
# ---------------------------------------------------------------------------


def test_workspace_no_escribible_da_resultado_de_error_no_excepcion_cruda(
    tmp_path: Path, monkeypatch
):
    """Disparador REAL (no mockeado): un directorio con permisos 000 --
    mismo efecto que disco lleno, workspace remontado sólo-lectura o cuota
    excedida, los tres disparadores reales del camino degradado que
    Fernando nombró. Antes de este arreglo, esto dejaba escapar un
    `PermissionError` crudo de `ocr.extraer` (y de `compuerta.extraer`,
    cuyo contrato dice lo contrario)."""
    import os

    raiz_sin_permiso = tmp_path / "sin-permiso"
    raiz_sin_permiso.mkdir()
    os.chmod(raiz_sin_permiso, 0o000)
    try:
        monkeypatch.setattr(ocr, "_WORKSPACE_DIR", raiz_sin_permiso / "workspace")

        origen = tmp_path / "algo.pdf"
        origen.write_bytes(b"%PDF-1.4\n%%EOF")

        r = ocr.extraer(origen)

        assert r.estado == "error"
        assert r.salidas == {}
        assert "razon" in r.detalle
    finally:
        os.chmod(raiz_sin_permiso, 0o755)  # permite que tmp_path se limpie despues


def test_workspace_dir_resuelve_symlinks(tmp_path: Path, monkeypatch):
    """`.resolve()` (adenda MAJOR 2): `tool_authority.py:62` YA resuelve
    `WORKSPACE_ROOT` así -- acá faltaba. Si `JAX_WORKSPACE_DIR` trae un
    symlink, el jail (que compara contra la forma canónica) y este
    rasterizado terminan mirando rutas DISTINTAS aunque el valor de la
    env var sea el mismo en los dos."""
    real = tmp_path / "real-workspace"
    real.mkdir()
    enlace = tmp_path / "enlace-workspace"
    enlace.symlink_to(real)

    monkeypatch.setenv("JAX_WORKSPACE_DIR", str(enlace))

    resuelto = ocr._resolver_workspace_dir()

    assert resuelto == real.resolve()
    assert resuelto != enlace, (
        "sin .resolve(), esto hubiera quedado apuntando al symlink, no al "
        "destino real"
    )


# ---------------------------------------------------------------------------
# Jax#338 ronda 1: S-1 (lista de rutas), MAJOR-2 (decodificar entero),
# MINOR-1/3/4/6
# ---------------------------------------------------------------------------


def _ruidosa(destino: Path, size=(600, 400), seed=1) -> Path:
    import random

    from PIL import Image

    rnd = random.Random(seed)
    img = Image.new("L", size)
    img.putdata([rnd.randint(0, 255) for _ in range(size[0] * size[1])])
    img.save(destino)
    return destino


def _sin_llamar_a_tesseract(monkeypatch) -> list:
    llamadas: list = []
    real = ocr.subprocess.run

    def espia(cmd, *a, **k):
        llamadas.append(list(cmd))
        return real(cmd, *a, **k)

    monkeypatch.setattr(ocr.subprocess, "run", espia)
    return llamadas


def test_s1_un_png_cuyo_contenido_es_la_ruta_de_otra_imagen_no_se_lee(
    tmp_path: Path, monkeypatch
):
    """S-1: tesseract interpreta una entrada que no es imagen como LISTA DE
    RUTAS. Un `foto.png` con la ruta de otra imagen real del host la leeria.
    Por la razon correcta: los bytes magicos no coinciden, ni se llama a
    tesseract."""
    secreta = _imagen_multilinea(tmp_path / "secreta.png", [
        "Estado de Situación Financiera — año 2026",
        "Activos totales 1,234,567.89 USD",
        "Pasivos totales 987,654.32 USD",
        "Patrimonio neto 246,913.57 USD",
    ])
    señuelo = tmp_path / "foto.png"
    señuelo.write_text(str(secreta) + "\n", encoding="utf8")
    llamadas = _sin_llamar_a_tesseract(monkeypatch)

    r = ocr.extraer(señuelo)

    assert r.estado == "error"
    assert r.salidas == {}
    assert r.detalle["codigo"] == "archivo_ilegible"
    assert r.detalle["causa"] == "firma_invalida"
    # (`tesseract --version` de `_version()` no lee ninguna imagen)
    assert not any(c and c[0] == "tesseract" and "--version" not in c for c in llamadas)


def test_s1_bytes_aleatorios_son_archivo_ilegible_sin_reventar(tmp_path: Path):
    import random

    azar = tmp_path / "azar.jpg"
    azar.write_bytes(random.Random(5).randbytes(4096))
    r = ocr.extraer(azar)
    assert r.estado == "error"
    assert r.detalle["codigo"] == "archivo_ilegible"


def test_s1_la_imagen_va_a_tesseract_por_stdin_nunca_por_ruta(
    tmp_path: Path, monkeypatch
):
    origen = _imagen_una_linea(tmp_path / "linea.png", "Activos totales 1,234 USD")
    visto: list = []
    real = ocr.subprocess.run

    def espia(cmd, *a, **k):
        if cmd and cmd[0] == "tesseract" and "--version" not in cmd:
            visto.append((list(cmd), k.get("input")))
        return real(cmd, *a, **k)

    monkeypatch.setattr(ocr.subprocess, "run", espia)
    ocr.extraer(origen)

    assert visto, "tesseract no se invoco"
    for cmd, entrada in visto:
        assert cmd[1] == "-"
        assert str(origen) not in cmd
        assert isinstance(entrada, bytes) and entrada == origen.read_bytes()


@pytest.mark.parametrize("formato", ["png", "jpg", "tif", "bmp", "webp"])
def test_cada_formato_de_imagen_pasa_por_stdin_y_se_lee(tmp_path: Path, formato: str):
    from PIL import Image

    base = _imagen_multilinea(tmp_path / "base.png", [
        "Estado de Situación Financiera", "Activos totales 1,234,567.89 USD",
        "Pasivos totales 987,654.32 USD", "Patrimonio neto 246,913.57 USD",
    ])
    destino = tmp_path / f"doc.{formato}"
    Image.open(base).convert("RGB").save(destino)
    r = ocr.extraer(destino)
    assert r.estado in {"ok", "parcial"}
    assert "Activos totales" in r.salidas["texto.txt"]


@pytest.mark.parametrize("formato", ["tif", "png", "jpg"])
def test_major2_imagen_truncada_es_archivo_ilegible(tmp_path: Path, formato: str):
    """MAJOR-2: un TIFF truncado da rc=0 y vacio en tesseract (quedaba `ok`).
    Se decodifica ENTERA con Pillow antes de OCR."""
    completa = _ruidosa(tmp_path / f"c.{formato}")
    datos = completa.read_bytes()
    rota = tmp_path / f"rota.{formato}"
    rota.write_bytes(datos[: len(datos) // 2])
    r = ocr.extraer(rota)
    assert r.estado == "error"
    assert r.detalle["codigo"] == "archivo_ilegible"
    assert r.detalle["causa"] == "no_decodifica"


def test_major2_jpeg_con_el_cuerpo_corrupto_es_archivo_ilegible(tmp_path: Path):
    import random

    completa = _ruidosa(tmp_path / "c.jpg")
    d = completa.read_bytes()
    i = d.index(b"\xff\xda") + 14
    malo = tmp_path / "malo.jpg"
    malo.write_bytes(d[:i] + random.Random(7).randbytes(len(d) - i - 2) + b"\xff\xd9")
    r = ocr.extraer(malo)
    assert r.estado == "error"
    assert r.detalle["codigo"] == "archivo_ilegible"


def test_major2_imagen_con_demasiados_pixeles_es_ilegible(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ocr, "MAX_PIXELES", 1000)
    r = ocr.extraer(_ruidosa(tmp_path / "grande.png", size=(100, 100)))
    assert r.estado == "error"
    assert r.detalle["codigo"] == "archivo_ilegible"
    assert r.detalle["causa"] == "demasiados_pixeles"


def test_minor3_la_ficha_no_copia_stderr_ni_rutas(tmp_path: Path):
    señuelo = tmp_path / "secreto-ruta.png"
    señuelo.write_text("/etc/passwd\n", encoding="utf8")
    r = ocr.extraer(señuelo)
    volcado = json.dumps(dict(r.detalle))
    assert "stderr" not in r.detalle
    assert str(tmp_path) not in volcado and "/etc/passwd" not in volcado


def test_minor4_es_pdf_busca_la_firma_en_los_primeros_1024_bytes(tmp_path: Path):
    con_basura = tmp_path / "a.pdf"
    con_basura.write_bytes(b"\x00\xef\xbb\xbf  basura\n" + b"%PDF-1.4\n")
    assert ocr._es_pdf(con_basura) is True
    lejos = tmp_path / "b.pdf"
    lejos.write_bytes(b"x" * 2000 + b"%PDF-1.4")
    assert ocr._es_pdf(lejos) is False
    assert ocr._es_pdf(tmp_path / "no-existe.pdf") is False


def test_minor6_foto_de_4032x3024_con_ruido_no_es_un_error(tmp_path: Path):
    """Forma del dato real: una foto de telefono (12 Mpx), no un fixture de
    400x200."""
    origen = _ruidosa(tmp_path / "foto.jpg", size=(4032, 3024), seed=2)
    r = ocr.extraer(origen)
    assert r.estado in {"ok", "parcial"}
    assert r.detalle["ancho"] == 4032 and r.detalle["alto"] == 3024


def _documento_fotografiado(destino: Path) -> Path:
    """Texto real en baja resolucion + ruido + desenfoque: el tipo de foto
    de documento que da mayoria de palabras dudosas."""
    import random

    from PIL import Image, ImageDraw, ImageFilter

    # Parametros medidos 2026-10-03 (tesseract 5.5.0): 171 palabras, 134
    # dudosas. Hay un acantilado -- un poco mas de ruido o de desenfoque y
    # tesseract no reconoce NINGUNA palabra -- por eso el test comprueba la
    # forma del fixture antes de usarlo.
    img = Image.new("L", (1000, 700), 215)
    d = ImageDraw.Draw(img)
    fuente = _fuente(8)
    for i in range(22):
        d.text(
            (20, 20 + i * 30),
            f"Gerente {i} area operaciones planta Choluteca turno {i * 7}",
            fill=70, font=fuente,
        )
    img = img.filter(ImageFilter.GaussianBlur(0.5))
    rnd = random.Random(11)
    px = img.load()
    for x in range(img.width):
        for y in range(img.height):
            px[x, y] = max(0, min(255, px[x, y] + rnd.randint(-16, 16)))
    img.save(destino)
    return destino


def test_minor6_documento_fotografiado_sintetico_no_es_un_error(tmp_path: Path):
    origen = _documento_fotografiado(tmp_path / "doc-foto.jpg")
    analisis = ocr._ocr_una_imagen(origen, "spa")
    assert analisis is not None and analisis["n_palabras"] >= 10
    # la forma del caso: la mayoria de las palabras reconocidas son dudosas
    assert len(analisis["palabras_dudosas"]) / analisis["n_palabras"] > 0.5
    r = ocr.extraer(origen)
    assert r.estado == "parcial"
    assert r.detalle["codigo"] == "imagen_texto_dudoso"
    assert r.salidas["texto.txt"].count("\n") > 10  # el texto leido se conserva


def _ingerir_imagen_sin_texto(tmp_path, monkeypatch):
    from motor_registry import tool_authority
    from PIL import Image

    from procesamiento import ingesta

    monkeypatch.setattr(tool_authority, "WORKSPACE_ROOT", tmp_path.resolve())
    origen = tmp_path / "foto.png"
    Image.new("RGB", (400, 200), "white").save(origen)
    trabajo = tmp_path.resolve() / "trabajo"
    ficha = ingesta.ingerir(origen, trabajo)
    return ingesta, origen, trabajo, ficha


def test_minor1_la_ficha_lleva_la_version_de_la_logica_del_ocr(tmp_path, monkeypatch):
    _, _, _, ficha = _ingerir_imagen_sin_texto(tmp_path, monkeypatch)
    assert ficha.detalle["_version_logica"] == ocr.VERSION_LOGICA


def test_minor1_cambiar_la_logica_invalida_la_cache(tmp_path, monkeypatch):
    from procesamiento import compuerta

    ingesta, origen, trabajo, _ = _ingerir_imagen_sin_texto(tmp_path, monkeypatch)
    monkeypatch.setattr(ocr, "VERSION_LOGICA_IMAGEN", "otra-regla")
    llamadas = []
    original = compuerta.extraer
    monkeypatch.setattr(
        compuerta, "extraer", lambda *a, **k: llamadas.append(a) or original(*a, **k)
    )
    f2 = ingesta.ingerir(origen, trabajo)
    assert len(llamadas) == 1
    assert f2.detalle["_version_logica"] == "otra-regla"


def test_minor1_un_error_viejo_con_tres_intentos_se_reintenta_si_cambio_la_logica(
    tmp_path, monkeypatch
):
    """El tope D-2 (3 intentos) congelaba el `error` viejo; una ficha escrita
    con otra logica no cuenta como intento previo."""
    from procesamiento import compuerta
    from procesamiento.ficha import Ficha

    ingesta, origen, trabajo, ficha = _ingerir_imagen_sin_texto(tmp_path, monkeypatch)
    carpeta = ingesta.ruta_procesado(trabajo, ficha.sha256)
    viejo = Ficha(
        sha256=ficha.sha256, origen=ficha.origen, extractor=ficha.extractor,
        extractor_version=ficha.extractor_version, fecha=ficha.fecha, estado="error",
        detalle={"razon": "vieja", "_extension_ingesta": ".png", "_intentos": 3},
    )
    (carpeta / "ficha.json").write_text(viejo.a_json(), encoding="utf8")
    llamadas = []
    original = compuerta.extraer
    monkeypatch.setattr(
        compuerta, "extraer", lambda *a, **k: llamadas.append(a) or original(*a, **k)
    )
    f2 = ingesta.ingerir(origen, trabajo)
    assert len(llamadas) == 1
    assert f2.estado == "ok"


# ---------------------------------------------------------------------------
# Jax#338 ronda 1: regla de Fernando (A/B/D) -- tamano de pagina
# ---------------------------------------------------------------------------


def _blanca(destino: Path, ancho: int, alto: int) -> Path:
    from PIL import Image

    Image.new("L", (ancho, alto), 255).save(destino)
    return destino


@pytest.mark.parametrize("ancho,alto", [
    (2480, 3508),   # A4 a 300 DPI
    (3508, 2480),   # apaisada
    (2550, 3300),   # carta a 300
    (1240, 1754),   # A4 a 150 (el borde bajo del DPI implicito)
    (2480, 3482),   # proporcion 1,404: dentro de +-0,02 de A4
])
def test_d_imagen_del_tamano_de_una_pagina_sin_texto_es_parcial(
    tmp_path: Path, ancho: int, alto: int
):
    r = ocr.extraer(_blanca(tmp_path / "p.png", ancho, alto))
    assert r.estado == "parcial"
    assert r.detalle["codigo"] == "imagen_pagina_sin_texto"
    assert "posible documento escaneado sin texto: revisar o reescanear" in r.salidas["texto.txt"]
    assert r.detalle["ancho"] == ancho and r.detalle["alto"] == alto


@pytest.mark.parametrize("ancho,alto", [
    (4032, 3024), (3024, 4032), (5500, 3830), (3288, 3740), (1500, 780),
    (2480, 3457),   # proporcion 1,394: justo fuera de +-0,02 de A4
    (2000, 2870),   # proporcion 1,435: justo fuera
    (1157, 1637),   # A4 a 140 DPI: bajo el minimo
    (3391, 4793),   # A4 a 410 DPI: sobre el maximo
])
def test_a_una_imagen_que_no_es_pagina_sin_texto_es_ok(tmp_path: Path, ancho, alto):
    r = ocr.extraer(_blanca(tmp_path / "f.png", ancho, alto))
    assert r.estado == "ok"
    assert r.detalle["codigo"] == "imagen_sin_texto"


def test_implica_pagina_es_pura_y_nombra_la_referencia():
    assert ocr._implica_pagina(2480, 3508) == "A4"
    assert ocr._implica_pagina(2550, 3300) == "carta"
    assert ocr._implica_pagina(4032, 3024) is None
    assert ocr._implica_pagina(0, 0) is None


def test_b_tiene_prioridad_sobre_d_aunque_la_imagen_sea_una_pagina():
    r = {
        "texto": "algo leido con baja confianza", "caracteres": 29, "n_palabras": 6,
        "confianza_promedio": 30.0, "palabras_dudosas": [{"palabra": "x", "confianza": 10.0}],
        "ancho": 0, "alto": 0, "clasificacion": "sin_texto",
    }
    res = ocr._resolver_imagen(r, "spa", [(2480, 3508)])
    assert res.estado == "parcial"
    assert res.detalle["codigo"] == "imagen_texto_dudoso"
    assert "algo leido con baja confianza" in res.salidas["texto.txt"]


_REAL = Path(
    "/srv/jax-data/jax-workspace/proyectos/01a1029d-3078-7707-aab7-2000d235137f"
    "/fuente/documenatcion-legal/socios/RTN Angel Molina.jpeg"
)


@pytest.mark.skipif(not _REAL.is_file(), reason="imagen real de produccion ausente (solo hall9000)")
def test_real_rtn_angel_molina_es_pagina_sin_texto():
    """Solo lectura sobre la imagen real de produccion (2480x3508)."""
    r = ocr.extraer(_REAL)
    assert r.estado == "parcial"
    assert r.detalle["codigo"] == "imagen_pagina_sin_texto"


_REAL_ORG = _REAL.parents[2] / "fotos" / "IMG_2353.jpeg"


@pytest.mark.skipif(not _REAL_ORG.is_file(), reason="IMG_2353.jpeg real ausente (solo hall9000)")
def test_real_img_2353_organigrama_es_parcial_y_legible():
    r = ocr.extraer(_REAL_ORG)
    assert r.estado == "parcial"
    assert r.detalle["codigo"] == "imagen_texto_dudoso"
    assert len(r.salidas["texto.txt"]) > 200


# ---------------------------------------------------------------------------
# Jax#338 ronda 2: TIFF multipagina (N1), una sola lectura (N2), firma de
# imagen antes de %PDF (N3), Pillow (N4), logica por camino (N5), tope (N6)
# ---------------------------------------------------------------------------


def _tiff_multipagina(destino: Path, paginas: list) -> Path:
    paginas[0].save(destino, save_all=True, append_images=paginas[1:])
    return destino


def test_n1_tiff_con_la_segunda_pagina_truncada_es_archivo_ilegible(tmp_path: Path):
    import random

    from PIL import Image

    rnd = random.Random(3)
    paginas = []
    for _ in range(2):
        im = Image.new("L", (300, 200))
        im.putdata([rnd.randint(0, 255) for _ in range(300 * 200)])
        paginas.append(im)
    completo = _tiff_multipagina(tmp_path / "ok.tif", paginas)
    datos = completo.read_bytes()
    roto = tmp_path / "multi_trunc_cut.tif"
    roto.write_bytes(datos[: int(len(datos) * 0.75)])

    r = ocr.extraer(roto)

    assert r.estado == "error"
    assert r.detalle["codigo"] == "archivo_ilegible"
    assert r.detalle["causa"] == "no_decodifica"


def test_n1_tiff_con_una_pagina_bomba_se_rechaza_sin_decodificarla(tmp_path: Path):
    """Pagina 2 de 20000x10000 (200 Mpx): antes pasaba la validacion porque
    solo se miraba el fotograma 0. Sin monkeypatch del tope."""
    from PIL import Image

    pequena = Image.new("1", (300, 200), 1)
    bomba = Image.new("1", (20000, 10000), 1)
    origen = tmp_path / "bomba.tif"
    pequena.save(origen, save_all=True, append_images=[bomba], compression="group4")

    r = ocr.extraer(origen)

    assert r.estado == "error"
    assert r.detalle["codigo"] == "archivo_ilegible"
    assert r.detalle["causa"] == "demasiados_pixeles"


def test_n1_tiff_con_demasiadas_paginas_se_rechaza(tmp_path: Path, monkeypatch):
    from PIL import Image

    monkeypatch.setattr(ocr, "MAX_PAGINAS", 3)
    paginas = [Image.new("L", (50, 50), 255) for _ in range(4)]
    origen = _tiff_multipagina(tmp_path / "largo.tif", paginas)
    r = ocr.extraer(origen)
    assert r.estado == "error"
    assert r.detalle["causa"] == "demasiadas_paginas"


def test_n1_tiff_de_dos_paginas_con_texto_en_las_dos_sale_completo(tmp_path: Path):
    from PIL import Image

    p1 = _imagen_multilinea(tmp_path / "p1.png", [
        "Primera pagina del contrato", "Activos totales 1,234,567.89 USD",
        "Pasivos totales 987,654.32 USD", "Patrimonio neto 246,913.57 USD",
    ])
    p2 = _imagen_multilinea(tmp_path / "p2.png", [
        "Segunda pagina de anexos", "Garantia hipotecaria sobre inmueble",
        "Avaluo comercial 5,000,000.00 USD", "Firmado ante notario publico",
    ])
    tif = _tiff_multipagina(
        tmp_path / "dos.tif", [Image.open(p1).convert("RGB"), Image.open(p2).convert("RGB")]
    )

    r = ocr.extraer(tif)

    assert r.estado in {"ok", "parcial"}
    texto = r.salidas["texto.txt"]
    assert "Primera pagina del contrato" in texto
    assert "Segunda pagina de anexos" in texto
    assert "Avaluo comercial" in texto
    assert r.detalle["paginas"] == 2


def test_n1_cada_pagina_va_por_stdin_como_png_no_el_tiff_original(tmp_path: Path, monkeypatch):
    from PIL import Image

    p1 = _imagen_una_linea(tmp_path / "a.png", "Activos totales 1,234 USD")
    tif = _tiff_multipagina(
        tmp_path / "dos.tif",
        [Image.open(p1).convert("RGB"), Image.open(p1).convert("RGB")],
    )
    entradas: list = []
    real = ocr.subprocess.run

    def espia(cmd, *a, **k):
        if cmd and cmd[0] == "tesseract" and "--version" not in cmd and "tsv" not in cmd:
            entradas.append(k.get("input"))
        return real(cmd, *a, **k)

    monkeypatch.setattr(ocr.subprocess, "run", espia)
    ocr.extraer(tif)
    assert len(entradas) == 2
    assert all(e.startswith(b"\x89PNG") for e in entradas)


def test_n2_los_bytes_se_leen_una_vez_y_son_los_que_van_a_tesseract(
    tmp_path: Path, monkeypatch
):
    """Carrera validar/leer: si el archivo cambia DESPUES de validarlo, lo que
    se procesa sigue siendo lo validado."""
    origen = _imagen_multilinea(tmp_path / "doc.png", [
        "Estado de Situación Financiera", "Activos totales 1,234,567.89 USD",
        "Pasivos totales 987,654.32 USD", "Patrimonio neto 246,913.57 USD",
    ])
    original = origen.read_bytes()
    real_validar = ocr._validar_imagen

    def validar_y_cambiar(datos, *a, **k):
        resultado = real_validar(datos, *a, **k)
        origen.write_text("/etc/hostname\n")  # el archivo cambia tras validar
        return resultado

    monkeypatch.setattr(ocr, "_validar_imagen", validar_y_cambiar)
    entradas: list = []
    real = ocr.subprocess.run

    def espia(cmd, *a, **k):
        if cmd and cmd[0] == "tesseract" and "--version" not in cmd:
            entradas.append(k.get("input"))
        return real(cmd, *a, **k)

    monkeypatch.setattr(ocr.subprocess, "run", espia)
    r = ocr.extraer(origen)
    assert r.estado == "ok"
    assert "Activos totales" in r.salidas["texto.txt"]
    assert entradas and all(e == original for e in entradas)


def test_n3_un_png_con_metadata_pdf_es_una_imagen(tmp_path: Path):
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    base = _imagen_multilinea(tmp_path / "b.png", [
        "Estado de Situación Financiera", "Activos totales 1,234,567.89 USD",
        "Pasivos totales 987,654.32 USD", "Patrimonio neto 246,913.57 USD",
    ])
    meta = PngInfo()
    meta.add_text("Software", "exportado de %PDF-1.7")
    destino = tmp_path / "con-meta.png"
    Image.open(base).save(destino, pnginfo=meta)
    assert b"%PDF" in destino.read_bytes()[:1024]

    assert ocr._es_pdf(destino) is False
    r = ocr.extraer(destino)
    assert r.estado == "ok"
    assert "Activos totales" in r.salidas["texto.txt"]


def test_n5_la_version_de_la_logica_depende_del_camino():
    assert ocr.version_logica(".png") == ocr.VERSION_LOGICA_IMAGEN
    assert ocr.version_logica(".JPEG") == ocr.VERSION_LOGICA_IMAGEN
    assert ocr.version_logica(".pdf") is None


def test_n5_una_ficha_vieja_de_pdf_escaneado_se_reusa_y_una_de_imagen_no(
    tmp_path, monkeypatch
):
    import json

    from motor_registry import tool_authority
    from PIL import Image

    from procesamiento import compuerta, ingesta

    monkeypatch.setattr(tool_authority, "WORKSPACE_ROOT", tmp_path.resolve())
    trabajo = tmp_path.resolve() / "trabajo"
    lineas = [
        "Estado de Situación Financiera", "Activos totales 1,234,567.89 USD",
        "Pasivos totales 987,654.32 USD", "Patrimonio neto 246,913.57 USD",
    ]
    pdf = _pdf_de_imagenes(
        tmp_path / "escaneo.pdf",
        [Image.open(_imagen_multilinea(tmp_path / "pg.png", lineas)).convert("RGB")],
    )
    img = _imagen_multilinea(tmp_path / "foto.png", lineas)

    fichas = {}
    for origen in (pdf, img):
        f = ingesta.ingerir(origen, trabajo)
        assert f.estado == "ok"
        ruta = ingesta.ruta_procesado(trabajo, f.sha256) / "ficha.json"
        datos = json.loads(ruta.read_text(encoding="utf8"))
        datos["detalle"].pop("_version_logica", None)  # ficha escrita ANTES
        ruta.write_text(json.dumps(datos), encoding="utf8")
        fichas[origen.name] = f

    llamadas: list = []
    original = compuerta.extraer
    monkeypatch.setattr(
        compuerta, "extraer", lambda *a, **k: llamadas.append(a) or original(*a, **k)
    )
    ingesta.ingerir(pdf, trabajo)
    assert llamadas == []          # el PDF NO cambio de logica: su cache sigue valiendo
    ingesta.ingerir(img, trabajo)
    assert len(llamadas) == 1      # la imagen SI: se reextrae


def test_n6_una_imagen_de_200_millones_de_pixeles_es_demasiados_pixeles(tmp_path: Path):
    from PIL import Image

    origen = tmp_path / "enorme.tif"
    Image.new("1", (20000, 10000), 1).save(origen, compression="group4")
    r = ocr.extraer(origen)
    assert r.estado == "error"
    assert r.detalle["causa"] == "demasiados_pixeles"


def test_n6_el_tope_de_pixeles_deja_pasar_la_foto_real_mas_grande():
    assert ocr.MAX_PIXELES == 100_000_000
    assert 13630 * 3826 < ocr.MAX_PIXELES


def test_minor3_un_fallo_inesperado_no_copia_el_mensaje_de_la_excepcion(
    tmp_path: Path, monkeypatch
):
    def revienta(*a, **k):
        raise RuntimeError("/ruta/secreta/con/contenido")

    monkeypatch.setattr(ocr, "_validar_imagen", revienta)
    r = ocr.extraer(_imagen_una_linea(tmp_path / "x.png", "Activos totales 1,234 USD"))
    assert r.estado == "error"
    assert "RuntimeError" in r.detalle["razon"]
    assert "secreta" not in json.dumps(dict(r.detalle))


def test_decision_a_y_b_coinciden_se_aplica_a_b_exige_ocho_caracteres():
    """Decision del controlador (Fernando puede cambiarla): menos de 8
    caracteres CON mayoria de palabras dudosas es A (o D si es pagina), no B:
    con una o dos palabras no hay texto que conservar."""
    r = {
        "texto": "ab c", "caracteres": 4, "n_palabras": 2, "confianza_promedio": 20.0,
        "palabras_dudosas": [{"palabra": "ab", "confianza": 10.0}, {"palabra": "c", "confianza": 9.0}],
        "ancho": 0, "alto": 0, "clasificacion": "sin_texto",
    }
    a = ocr._resolver_imagen(r, "spa", [(4032, 3024)])
    assert a.estado == "ok" and a.detalle["codigo"] == "imagen_sin_texto"
    d = ocr._resolver_imagen(r, "spa", [(2480, 3508)])
    assert d.estado == "parcial" and d.detalle["codigo"] == "imagen_pagina_sin_texto"


def test_d_en_un_tiff_multipagina_exige_que_todas_las_paginas_sean_de_pagina():
    r = {
        "texto": "", "caracteres": 0, "n_palabras": 0, "confianza_promedio": 0.0,
        "palabras_dudosas": [], "ancho": 0, "alto": 0, "clasificacion": "sin_texto",
    }
    todas = ocr._resolver_imagen(r, "spa", [(2480, 3508), (2480, 3508)])
    assert todas.detalle["codigo"] == "imagen_pagina_sin_texto"
    mezcla = ocr._resolver_imagen(r, "spa", [(2480, 3508), (4032, 3024)])
    assert mezcla.detalle["codigo"] == "imagen_sin_texto"
