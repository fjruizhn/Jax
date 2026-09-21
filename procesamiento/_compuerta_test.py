"""La compuerta: elige extractor por tipo, y NUNCA paga el caro si el barato
alcanzó.

Task 7 (2026-09-20/21, procesamiento-archivos-nucleo). La regla es una sola
idea: un PDF CON capa de texto no paga OCR, y un PDF SIN capa de texto no
pasa por el extractor de texto. Los tests lo verifican con monkeypatch,
comprobando que el camino caro NO se invocó -- aseverar qué NO se llamó es
tan importante como qué sí.
"""
from pathlib import Path

import pytest

from procesamiento import compuerta


def test_un_tipo_desconocido_dice_sin_extractor_y_no_inventa(tmp_path: Path):
    raro = tmp_path / "algo.xyz"
    raro.write_bytes(b"contenido cualquiera")
    r = compuerta.extraer(raro)
    assert r.estado == "sin_extractor"
    assert r.salidas == {}


def test_un_pdf_sin_capa_de_texto_cae_en_ocr_y_no_en_pdfplumber(
    tmp_path: Path, monkeypatch
):
    """El PDF escaneado NO se resuelve con el extractor de texto: pasa a OCR.
    Y el OCR sólo se invoca si el camino barato ya dijo que no."""
    from procesamiento.extractores import ocr, pdf
    from procesamiento.resultado import Resultado

    llamadas: list[str] = []

    def falso_tiene_texto(origen):
        llamadas.append("deteccion")
        return False

    def falso_ocr(origen, idioma="spa"):
        llamadas.append("ocr")
        return Resultado(
            estado="ok", salidas={"texto.txt": "leido por ocr"},
            detalle={}, extractor="tesseract", version="5.5.0",
        )

    def no_debe_llamarse(origen):
        raise AssertionError("pdfplumber no debe correr sobre un PDF sin texto")

    monkeypatch.setattr(pdf, "tiene_capa_de_texto", falso_tiene_texto)
    monkeypatch.setattr(pdf, "extraer", no_debe_llamarse)
    monkeypatch.setattr(ocr, "extraer", falso_ocr)

    archivo = tmp_path / "escaneado.pdf"
    archivo.write_bytes(b"%PDF-1.4 lo que sea")
    r = compuerta.extraer(archivo)

    assert r.estado == "ok"
    assert llamadas == ["deteccion", "ocr"]


def test_un_pdf_con_texto_no_paga_ocr(tmp_path: Path, monkeypatch):
    from procesamiento.extractores import ocr, pdf
    from procesamiento.resultado import Resultado

    monkeypatch.setattr(pdf, "tiene_capa_de_texto", lambda origen: True)
    monkeypatch.setattr(
        pdf, "extraer",
        lambda origen: Resultado(
            estado="ok", salidas={"texto.md": "hola"}, detalle={},
            extractor="pdfplumber", version="0.11.10",
        ),
    )

    def no_debe_llamarse(origen, idioma="spa"):
        raise AssertionError("no se paga OCR si el PDF ya tenia texto")

    monkeypatch.setattr(ocr, "extraer", no_debe_llamarse)

    archivo = tmp_path / "nativo.pdf"
    archivo.write_bytes(b"%PDF-1.4 lo que sea")
    assert compuerta.extraer(archivo).estado == "ok"


def test_un_xlsx_no_paga_ocr(tmp_path: Path, monkeypatch):
    """El ruteo por extensión es justo donde un error manda el archivo al
    extractor equivocado -- un .xlsx tiene que resolver con excel.extraer,
    nunca con OCR."""
    from procesamiento.extractores import excel, ocr
    from procesamiento.resultado import Resultado

    monkeypatch.setattr(
        excel, "extraer",
        lambda origen: Resultado(
            estado="ok", salidas={"hoja1.csv": "a,b"}, detalle={},
            extractor="openpyxl", version="3.1.5",
        ),
    )

    def no_debe_llamarse(origen, idioma="spa"):
        raise AssertionError("un xlsx no debe pasar por OCR")

    monkeypatch.setattr(ocr, "extraer", no_debe_llamarse)

    archivo = tmp_path / "balance.xlsx"
    archivo.write_bytes(b"PK\x03\x04 lo que sea")
    assert compuerta.extraer(archivo).estado == "ok"


def test_un_docx_no_pasa_por_el_extractor_de_excel(tmp_path: Path, monkeypatch):
    """Un .docx tiene que resolver con word.extraer, nunca con el de Excel."""
    from procesamiento.extractores import excel, word
    from procesamiento.resultado import Resultado

    monkeypatch.setattr(
        word, "extraer",
        lambda origen: Resultado(
            estado="ok", salidas={"texto.md": "hola"}, detalle={},
            extractor="python-docx", version="1.2.0",
        ),
    )

    def no_debe_llamarse(origen):
        raise AssertionError("un docx no debe pasar por el extractor de Excel")

    monkeypatch.setattr(excel, "extraer", no_debe_llamarse)

    archivo = tmp_path / "informe.docx"
    archivo.write_bytes(b"PK\x03\x04 lo que sea")
    assert compuerta.extraer(archivo).estado == "ok"


def test_import_compuerta_no_explota_sin_python_docx(tmp_path: Path, monkeypatch):
    """I-1 (ronda de arreglo, task-7-hallazgos.md): antes de este arreglo,
    `word.py` importaba `docx.oxml.ns`/`docx.table`/`docx.text.paragraph` A
    NIVEL DE MÓDULO -- al revés que `excel.py`/`pdf.py`/`ocr.py`, que
    importan perezosamente. Con python-docx ausente, `import
    procesamiento.compuerta` explotaba ENTERO -- Excel, PDF y OCR se caían
    con él, aunque sus propias dependencias estuvieran sanas. Se fuerza un
    import FRESCO (no el ya cacheado por la colección de tests, que corrió
    con python-docx presente) bloqueando 'docx' en `sys.modules` -- mismo
    patrón que usan los propios extractores para probar su guard de
    dependencia ausente."""
    import importlib
    import sys

    # Purgar TODO el namespace "docx.*" Y "procesamiento.*" ya cacheados
    # por la colección de tests (que corrió con python-docx real
    # presente). Las dos purgas hacen falta -- probado a mano: borrar
    # sólo "docx.oxml.ns"/"procesamiento.extractores.word" de
    # sys.modules NO alcanza, porque el paquete "procesamiento.extractores"
    # sigue teniendo el atributo `word` apuntando al módulo viejo YA
    # cargado (el mecanismo `from paquete import submodulo` de Python usa
    # `hasattr(paquete, submodulo)` antes de intentar un import fresco), y
    # el test pasaba con este falso negativo incluso contra el código
    # viejo (imports de `docx` a nivel de módulo) -- confirmado
    # reproduciendo el caso a mano antes de fijar este enfoque.
    for nombre in list(sys.modules):
        if (
            nombre == "docx" or nombre.startswith("docx.")
            or nombre == "procesamiento" or nombre.startswith("procesamiento.")
        ):
            monkeypatch.delitem(sys.modules, nombre, raising=False)
    monkeypatch.setitem(sys.modules, "docx", None)

    modulo = importlib.import_module("procesamiento.compuerta")

    archivo = tmp_path / "informe.docx"
    archivo.write_bytes(b"PK\x03\x04 lo que sea")
    r = modulo.extraer(archivo)
    assert r.estado == "sin_extractor"
    assert r.salidas == {}


def test_una_imagen_no_pasa_por_pdfplumber(tmp_path: Path, monkeypatch):
    """Una imagen suelta tiene que resolver con ocr.extraer, nunca con el
    extractor de PDF nativo (pdfplumber)."""
    from procesamiento.extractores import ocr, pdf
    from procesamiento.resultado import Resultado

    monkeypatch.setattr(
        ocr, "extraer",
        lambda origen, idioma="spa": Resultado(
            estado="ok", salidas={"texto.txt": "leido"}, detalle={},
            extractor="tesseract", version="5.5.0",
        ),
    )

    def no_debe_llamarse(origen):
        raise AssertionError("una imagen no debe pasar por pdfplumber")

    monkeypatch.setattr(pdf, "extraer", no_debe_llamarse)

    archivo = tmp_path / "recibo.png"
    archivo.write_bytes(b"\x89PNG lo que sea")
    assert compuerta.extraer(archivo).estado == "ok"


# ---------------------------------------------------------------------------
# I-2 (ronda de arreglo, task-7-hallazgos.md): IMAGENES tenía cobertura para
# ".png" y nada más (1 de 7) -- borrar jpg/jpeg/tif/tiff/bmp/webp del
# conjunto dejaba la suite en "6 passed", el número exacto del piso de CI,
# sin que nada lo notara. Mismo defecto en EXCEL (1 de 2, sólo ".xlsx").
#
# Contenido GENÉRICO (sin ninguna firma reconocida por
# `_tipo_por_contenido`) para aislar la ruta de RESPALDO por extensión del
# enrutado por contenido de I-3: si el contenido fuera una imagen real, con
# I-3 ya aplicado el enrutado la resolvería sola y remover la extensión del
# conjunto no rompería el ruteo (sólo dejaría de sumar el mismo resultado
# vía otro camino) -- el mutante quedaría vivo por una razón distinta a la
# que este test tiene que probar.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extension", sorted(compuerta.IMAGENES))
def test_cada_extension_de_imagenes_rutea_a_ocr(tmp_path: Path, monkeypatch, extension):
    from procesamiento.extractores import ocr
    from procesamiento.resultado import Resultado

    monkeypatch.setattr(
        ocr, "extraer",
        lambda origen, idioma="spa": Resultado(
            estado="ok", salidas={"texto.txt": "leido"}, detalle={},
            extractor="tesseract", version="5.5.0",
        ),
    )
    archivo = tmp_path / f"recibo{extension}"
    archivo.write_bytes(b"contenido generico sin firma reconocida")
    r = compuerta.extraer(archivo)
    assert r.estado == "ok"
    assert "extension_enganosa" not in r.detalle


@pytest.mark.parametrize("extension", sorted(compuerta.EXCEL))
def test_cada_extension_de_excel_rutea_a_excel(tmp_path: Path, monkeypatch, extension):
    from procesamiento.extractores import excel
    from procesamiento.resultado import Resultado

    monkeypatch.setattr(
        excel, "extraer",
        lambda origen: Resultado(
            estado="ok", salidas={"hoja1.csv": "a,b"}, detalle={},
            extractor="openpyxl", version="3.1.5",
        ),
    )
    archivo = tmp_path / f"balance{extension}"
    archivo.write_bytes(b"contenido generico sin firma reconocida")
    r = compuerta.extraer(archivo)
    assert r.estado == "ok"
    assert "extension_enganosa" not in r.detalle


# ---------------------------------------------------------------------------
# I-3 (ronda de arreglo, task-7-hallazgos.md): un PDF nativo de verdad
# renombrado ".png" saltaba la compuerta entera -- iba derecho a OCR sin
# pasar nunca por `tiene_capa_de_texto`. Se rutea por CONTENIDO cuando es
# decisivo (PDF/OLE2/imagen); la extensión queda de respaldo cuando el
# contenido no dice nada (zip, ambiguo entre .xlsx y .docx).
# ---------------------------------------------------------------------------


def _pdf_nativo(destino: Path) -> Path:
    """PDF 1.4 con capa de texto REAL, escrito a mano byte a byte (mismo
    fixture que `_pdf_test.py`, que ya probó parsear sin problema con
    pdfplumber -- sin depender de LibreOffice como dependencia de CI)."""
    stream = (
        b"BT /F1 12 Tf 20 180 Td (ACTIVOS TOTALES 1234) Tj ET\n"
        b"BT /F1 10 Tf 20 160 Td "
        b"(Estado de Situacion Financiera al cierre del periodo,) Tj ET\n"
        b"BT /F1 10 Tf 20 145 Td "
        b"(con el detalle completo de las cuentas patrimoniales y del) Tj ET\n"
        b"BT /F1 10 Tf 20 130 Td "
        b"(resultado del ejercicio fiscal correspondiente al periodo.) Tj ET\n"
    )
    partes = [
        b"%PDF-1.4\n",
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n",
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n",
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 220]"
        b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n",
        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n",
        b"5 0 obj<</Length " + str(len(stream)).encode() + b">>stream\n",
        stream,
        b"endstream endobj\n",
        b"trailer<</Root 1 0 R>>\n",
    ]
    destino.write_bytes(b"".join(partes))
    return destino


def test_pdf_nativo_renombrado_png_va_por_el_camino_del_pdf(tmp_path: Path, monkeypatch):
    """El caso EXACTO del hallazgo: un PDF nativo real, con extensión
    ``.png`` puesta encima -- tiene que ir por el camino del PDF (pagando
    pdfplumber, que es gratis comparado con OCR), invocar
    `tiene_capa_de_texto` de verdad (no mockeada -- se envuelve la real
    para contar, sin cambiar su resultado), y declarar en
    `detalle["extension_enganosa"]` que el nombre mentía."""
    from procesamiento.extractores import ocr, pdf
    pytest.importorskip("pdfplumber")

    llamadas = {"tiene_capa_de_texto": 0}
    original = pdf.tiene_capa_de_texto

    def contada(origen):
        llamadas["tiene_capa_de_texto"] += 1
        return original(origen)

    monkeypatch.setattr(pdf, "tiene_capa_de_texto", contada)

    def no_debe_llamarse(origen, idioma="spa"):
        raise AssertionError("un PDF nativo con texto no debe pagar OCR")

    monkeypatch.setattr(ocr, "extraer", no_debe_llamarse)

    archivo = _pdf_nativo(tmp_path / "escaneo.png")
    r = compuerta.extraer(archivo)

    assert llamadas["tiene_capa_de_texto"] == 1
    assert r.estado == "ok"
    assert "ACTIVOS TOTALES 1234" in r.salidas["texto.md"]
    assert r.detalle["extension_enganosa"] == {"nombre": ".png", "contenido": "pdf"}


def test_pdf_nativo_con_extension_correcta_no_marca_extension_enganosa(tmp_path: Path):
    """Contraparte del test anterior: cuando el nombre SÍ coincide con el
    contenido, no hay nada que anotar -- si esto marcara
    `extension_enganosa` con la extensión correcta, la señal dejaría de
    servir (ruido en todos lados en vez de en los casos que importan)."""
    pytest.importorskip("pdfplumber")
    archivo = _pdf_nativo(tmp_path / "estado.pdf")
    r = compuerta.extraer(archivo)
    assert r.estado == "ok"
    assert "extension_enganosa" not in r.detalle


def test_ole2_da_sin_extractor_sin_importar_la_extension(tmp_path: Path):
    """Un binario de Office viejo (OLE2) es `sin_extractor` sin importar
    qué extensión traiga encima -- ni siquiera si el nombre dice ``.xlsx``
    (mismo caso que I-3, ahora con la tercera fila de la tabla del
    ruling)."""
    archivo = tmp_path / "balance-viejo.xlsx"
    archivo.write_bytes(bytes.fromhex("D0CF11E0A1B11AE1") + b"resto del binario OLE2")
    r = compuerta.extraer(archivo)
    assert r.estado == "sin_extractor"
    assert r.salidas == {}
    assert r.detalle["extension_enganosa"] == {"nombre": ".xlsx", "contenido": "ole2"}


def test_imagen_real_renombrada_va_por_ocr_y_marca_extension_enganosa(
    tmp_path: Path, monkeypatch
):
    """Misma idea que el PDF de I-3, ahora del lado de la imagen: una PNG
    real (firma `\\x89PNG` de verdad) con extensión `.docx` encima tiene
    que resolver igual por OCR (el contenido manda) y declarar el
    engaño -- sin este caso, mutar la tupla `_FIRMAS_IMAGEN` (por ejemplo
    sacando la firma de PNG) queda absorbido en silencio por el respaldo
    de extensión cuando la extensión SÍ es una de `IMAGENES` (como en
    `test_una_imagen_no_pasa_por_pdfplumber`, que usa `.png` real): acá la
    extensión NO es de imagen, así que sólo el reconocimiento de la firma
    puede rutear esto a OCR."""
    from procesamiento.extractores import excel, ocr
    from procesamiento.resultado import Resultado

    monkeypatch.setattr(
        ocr, "extraer",
        lambda origen, idioma="spa": Resultado(
            estado="ok", salidas={"texto.txt": "leido"}, detalle={},
            extractor="tesseract", version="5.5.0",
        ),
    )

    def no_debe_llamarse(origen):
        raise AssertionError("una imagen real no debe pasar por el extractor de Excel")

    monkeypatch.setattr(excel, "extraer", no_debe_llamarse)

    archivo = tmp_path / "recibo.docx"
    archivo.write_bytes(b"\x89PNG\r\n\x1a\n resto de bytes de una PNG real")
    r = compuerta.extraer(archivo)

    assert r.estado == "ok"
    assert r.detalle["extension_enganosa"] == {"nombre": ".docx", "contenido": "imagen"}


def test_webp_real_renombrado_va_por_ocr_y_marca_extension_enganosa(
    tmp_path: Path, monkeypatch
):
    """Cobertura que faltaba (task-8, tarea extra que arrastraba la
    auditoría): la rama RIFF/WEBP de `_tipo_por_contenido`
    (compuerta.py:59-61) no tenía ningún caso propio -- mutar esa firma
    (por ejemplo, exigir `cabecera[:4] == b"RIFF"` sin el chequeo de
    `cabecera[8:12] == b"WEBP"`, o borrar la rama entera) dejaba la suite
    en verde, porque el respaldo por extensión (`.webp` está en
    `IMAGENES`) absorbía el caso feliz en silencio -- mismo patrón que
    `test_imagen_real_renombrada_va_por_ocr_y_marca_extension_enganosa`,
    ahora con la extensión puesta AL REVÉS (una firma WEBP real, renombrada
    con una extensión que NO es de imagen) para que sólo el reconocimiento
    de la firma pueda rutear esto a OCR."""
    from procesamiento.extractores import ocr, word
    from procesamiento.resultado import Resultado

    monkeypatch.setattr(
        ocr, "extraer",
        lambda origen, idioma="spa": Resultado(
            estado="ok", salidas={"texto.txt": "leido"}, detalle={},
            extractor="tesseract", version="5.5.0",
        ),
    )

    def no_debe_llamarse(origen):
        raise AssertionError("un WEBP real no debe pasar por el extractor de Word")

    monkeypatch.setattr(word, "extraer", no_debe_llamarse)

    # Firma RIFF/WEBP real: "RIFF" + 4 bytes de tamaño + "WEBP".
    firma = b"RIFF" + (100).to_bytes(4, "little") + b"WEBP" + b"resto de bytes"
    archivo = tmp_path / "foto.docx"
    archivo.write_bytes(firma)
    r = compuerta.extraer(archivo)

    assert r.estado == "ok"
    assert r.detalle["extension_enganosa"] == {"nombre": ".docx", "contenido": "imagen"}


def test_pdf_nativo_sin_pdfplumber_da_sin_extractor_no_ocr(tmp_path: Path, monkeypatch):
    """C-2 (final-hallazgos.md, ronda de cierre): sin `pdfplumber`
    instalado, un PDF NATIVO tiene que salir `sin_extractor` -- NUNCA
    rutear a OCR (que devolvería `estado='ok'` con las cifras DESTRUIDAS,
    medido: 0 de 8 cifras). Mismo tratamiento que un `.docx` sin
    `python-docx` (`test_import_compuerta_no_explota_sin_python_docx`)."""
    import sys

    from procesamiento.extractores import ocr

    def no_debe_llamarse(origen, idioma="spa"):
        raise AssertionError(
            "sin pdfplumber, un PDF nativo NO debe pagar OCR -- OCR produce "
            "'ok' con las cifras destruidas, no una alternativa honesta"
        )

    monkeypatch.setattr(ocr, "extraer", no_debe_llamarse)
    monkeypatch.setitem(sys.modules, "pdfplumber", None)

    archivo = _pdf_nativo(tmp_path / "estado.pdf")
    r = compuerta.extraer(archivo)

    assert r.estado == "sin_extractor"
    assert r.salidas == {}


def test_pdf_con_extension_pero_contenido_no_decisivo_usa_respaldo_por_extension(
    tmp_path: Path, monkeypatch
):
    """Hueco de cobertura encontrado con la propia mutación de esta ronda
    (no lo pedía ningún hallazgo, lo destapó el barrido): un archivo
    `.pdf` cuyo contenido NO empieza con `%PDF` (cabecera truncada/
    corrupta -- `_tipo_por_contenido` da `None`, no decisivo) tiene que
    seguir cayendo en la rama de RESPALDO por extensión (la del final de
    `extraer()`, distinta de la rama de contenido). Antes de este test,
    invertir la condición de `tiene_capa_de_texto` en ESA rama específica
    sobrevivía 20/20 -- ninguno de los otros tests la alcanza, porque
    todos los `.pdf` de la suite usan contenido que sí empieza con
    `%PDF`."""
    from procesamiento.extractores import ocr, pdf
    from procesamiento.resultado import Resultado

    monkeypatch.setattr(pdf, "tiene_capa_de_texto", lambda origen: True)
    monkeypatch.setattr(
        pdf, "extraer",
        lambda origen: Resultado(
            estado="ok", salidas={"texto.md": "hola"}, detalle={},
            extractor="pdfplumber", version="0.11.10",
        ),
    )

    def no_debe_llamarse(origen, idioma="spa"):
        raise AssertionError("con tiene_capa_de_texto=True no se paga OCR")

    monkeypatch.setattr(ocr, "extraer", no_debe_llamarse)

    archivo = tmp_path / "cabecera-rota.pdf"
    archivo.write_bytes(b"contenido generico sin firma reconocida")
    assert compuerta.extraer(archivo).estado == "ok"
