"""Word (.docx) → Markdown.

Ronda de arreglo 1 (2026-09-21, task-5-6-hallazgos.md — NO ratificado, 4
críticos + 3 importantes sobre este archivo). Cada Crítico era el mismo
modo de fallo: un extracto con `ok` que no es el documento.
"""
from pathlib import Path

import pytest

docx = pytest.importorskip("docx")

from procesamiento.extractores import word

_FIRMA_OLE2 = bytes.fromhex("D0CF11E0A1B11AE1")


def _documento_brief(destino: Path) -> Path:
    from docx import Document

    d = Document()
    d.add_heading("Estado de Resultados", level=1)
    d.add_paragraph("Periodo 2026")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text = "INGRESOS"
    t.cell(0, 1).text = "1000"
    t.cell(1, 0).text = "COSTOS"
    t.cell(1, 1).text = "400"
    d.save(destino)
    return destino


# ---------------------------------------------------------------------------
# Básicos (ya cubiertos antes de esta ronda, con el formato de tabla nuevo)
# ---------------------------------------------------------------------------


def test_conserva_titulo_parrafo_y_TODAS_las_filas_de_la_tabla(tmp_path: Path):
    """I-7: el único test de tabla del brief sólo aseveraba la primera
    fila -- una mutación que trunca `tabla.rows` a la primera quedaba en
    verde. Acá se asevera la tabla completa (encabezado y las DOS filas de
    datos), con el formato de bloque cercado (I-6: sin fila separadora,
    ninguna fila promovida)."""
    r = word.extraer(_documento_brief(tmp_path / "d.docx"))
    assert r.estado == "ok"
    md = r.salidas["texto.md"]
    assert "# Estado de Resultados" in md
    assert "Periodo 2026" in md
    assert "```tabla" in md
    assert "INGRESOS | 1000" in md
    assert "COSTOS | 400" in md


def test_un_docx_roto_da_error_sin_extracto(tmp_path: Path):
    malo = tmp_path / "roto.docx"
    malo.write_bytes(b"no soy un docx")
    r = word.extraer(malo)
    assert r.estado == "error"
    assert r.salidas == {}


def test_un_doc_viejo_da_sin_extractor_no_error_criptico(tmp_path: Path):
    """I-3 previa (no confundir con I-3 de OCR): la razón NO afirma que es
    Word -- un .doc, .xls o .ppt viejo dan la misma firma OLE2."""
    viejo = tmp_path / "balance-2020.doc"
    viejo.write_bytes(_FIRMA_OLE2 + b"\x00" * 200)
    r = word.extraer(viejo)
    assert r.estado == "sin_extractor"
    assert r.salidas == {}
    assert "Word" not in r.detalle["razon"]


def test_un_documento_sin_contenido_da_error_sin_extracto(tmp_path: Path):
    from docx import Document

    vacio = tmp_path / "vacio.docx"
    Document().save(vacio)
    r = word.extraer(vacio)
    assert r.estado == "error"
    assert r.salidas == {}


def test_una_celda_con_pipe_no_desalinea_la_tabla(tmp_path: Path):
    from docx import Document

    d = Document()
    t = d.add_table(rows=1, cols=2)
    t.cell(0, 0).text = "Ingresos|Egresos"
    t.cell(0, 1).text = "100"
    origen = tmp_path / "con-pipe.docx"
    d.save(origen)

    r = word.extraer(origen)

    assert r.estado == "ok"
    assert "Ingresos\\|Egresos | 100" in r.salidas["texto.md"]


def test_una_celda_con_salto_de_linea_no_parte_la_fila(tmp_path: Path):
    from docx import Document

    d = Document()
    t = d.add_table(rows=1, cols=2)
    celda = t.cell(0, 0)
    celda.text = "Cuentas por"
    celda.add_paragraph("cobrar diversas")
    t.cell(0, 1).text = "450"
    origen = tmp_path / "multilinea.docx"
    d.save(origen)

    r = word.extraer(origen)

    assert r.estado == "ok"
    md = r.salidas["texto.md"]
    assert "Cuentas por cobrar diversas | 450" in md
    assert md.count("Cuentas por") == 1


def test_extraer_sin_python_docx_instalado_da_sin_extractor(tmp_path: Path, monkeypatch):
    import sys

    origen = _documento_brief(tmp_path / "d.docx")
    monkeypatch.setitem(sys.modules, "docx", None)
    r = word.extraer(origen)
    assert r.estado == "sin_extractor"
    assert r.salidas == {}


# ---------------------------------------------------------------------------
# C-4: el orden real del documento se preserva
# ---------------------------------------------------------------------------


def test_orden_se_preserva_entre_titulos_y_tablas_de_distintos_anios(tmp_path: Path):
    """El caso EXACTO del hallazgo: "Balance 2025" + tabla, "Balance 2026"
    + tabla. Con el defecto viejo (todos los párrafos, después todas las
    tablas) el modelo atribuye las cifras al año equivocado. Acá se pide
    el orden exacto: 2025 y su cifra ANTES que 2026 y la suya."""
    from docx import Document

    d = Document()
    d.add_heading("Balance 2025", level=1)
    t1 = d.add_table(rows=1, cols=2)
    t1.cell(0, 0).text = "ACTIVO"
    t1.cell(0, 1).text = "1000"
    d.add_heading("Balance 2026", level=1)
    t2 = d.add_table(rows=1, cols=2)
    t2.cell(0, 0).text = "ACTIVO"
    t2.cell(0, 1).text = "9999"
    origen = tmp_path / "dos-anios.docx"
    d.save(origen)

    r = word.extraer(origen)

    assert r.estado == "ok"
    md = r.salidas["texto.md"]
    assert md.index("Balance 2025") < md.index("1000")
    assert md.index("1000") < md.index("Balance 2026")
    assert md.index("Balance 2026") < md.index("9999")


# ---------------------------------------------------------------------------
# C-5: tablas anidadas
# ---------------------------------------------------------------------------


def test_tabla_anidada_dentro_de_una_celda_no_desaparece(tmp_path: Path):
    """El caso EXACTO del hallazgo: una celda "DESGLOSE" con una tabla
    adentro (Ventas 5000, Otros 300). Los dos importes tienen que estar
    presentes -- 5.300 USD que antes se perdían enteros."""
    from docx import Document

    d = Document()
    outer = d.add_table(rows=1, cols=1)
    cell = outer.cell(0, 0)
    cell.text = "DESGLOSE"
    inner = cell.add_table(rows=2, cols=2)
    inner.cell(0, 0).text = "Ventas"
    inner.cell(0, 1).text = "5000"
    inner.cell(1, 0).text = "Otros"
    inner.cell(1, 1).text = "300"
    origen = tmp_path / "anidada.docx"
    d.save(origen)

    r = word.extraer(origen)

    assert r.estado == "ok"
    md = r.salidas["texto.md"]
    assert "DESGLOSE" in md
    assert "5000" in md
    assert "300" in md
    assert r.detalle["tablas"] == 1  # sigue siendo UNA tabla de nivel superior


# ---------------------------------------------------------------------------
# C-6: control de cambios
# ---------------------------------------------------------------------------


def test_control_de_cambios_conserva_el_valor_insertado_y_lo_declara(tmp_path: Path):
    """El caso EXACTO del hallazgo: "Utilidad neta: " con <w:del>100</w:del>
    <w:ins>9000</w:ins>. El valor VIGENTE (insertado) tiene que aparecer, y
    el documento tiene que declarar que trae cambios pendientes."""
    from docx import Document
    from docx.oxml.ns import qn
    from lxml import etree

    d = Document()
    p = d.add_paragraph()
    p.add_run("Utilidad neta: ")
    p_elem = p._p
    w_del = etree.SubElement(p_elem, qn("w:del"))
    w_del.set(qn("w:id"), "1")
    r_del = etree.SubElement(w_del, qn("w:r"))
    delText = etree.SubElement(r_del, qn("w:delText"))
    delText.text = "100"
    w_ins = etree.SubElement(p_elem, qn("w:ins"))
    w_ins.set(qn("w:id"), "2")
    r_ins = etree.SubElement(w_ins, qn("w:r"))
    t_ins = etree.SubElement(r_ins, qn("w:t"))
    t_ins.text = "9000"
    origen = tmp_path / "cambios.docx"
    d.save(origen)

    r = word.extraer(origen)

    assert r.estado == "ok"
    md = r.salidas["texto.md"]
    assert "9000" in md
    assert "100" not in md
    assert r.detalle["control_de_cambios"] is True


# ---------------------------------------------------------------------------
# C-7: encabezado, pie, w:sdt, cuadros de texto, notas al pie
# ---------------------------------------------------------------------------


def test_encabezado_y_pie_se_extraen(tmp_path: Path):
    from docx import Document

    d = Document()
    d.add_paragraph("Cuerpo del documento")
    sec = d.sections[0]
    sec.header.paragraphs[0].text = "Encabezado del reporte"
    sec.footer.paragraphs[0].text = "Total general: 1,234,567.89 USD"
    origen = tmp_path / "headfoot.docx"
    d.save(origen)

    r = word.extraer(origen)

    assert r.estado == "ok"
    md = r.salidas["texto.md"]
    assert "Encabezado del reporte" in md
    assert "Total general: 1,234,567.89 USD" in md


def test_control_de_contenido_sdt_extrae_parrafo_y_tabla(tmp_path: Path):
    """El caso EXACTO del hallazgo: un `w:sdt` (control de contenido -- la
    estructura de cualquier plantilla de formulario) con un párrafo y una
    tabla adentro. El párrafo Y la tabla tienen que sobrevivir, y
    `detalle["parrafos"]` no puede mentir contándolos en cero (I-7 de
    detalle: `documento.paragraphs` de python-docx NO ve los párrafos
    dentro de un `w:sdt`)."""
    from docx import Document
    from docx.oxml.ns import qn
    from lxml import etree

    d = Document()
    d.add_paragraph("antes")
    body = d.element.body
    sdt = etree.SubElement(body, qn("w:sdt"))
    etree.SubElement(sdt, qn("w:sdtPr"))
    sdt_content = etree.SubElement(sdt, qn("w:sdtContent"))
    p = etree.SubElement(sdt_content, qn("w:p"))
    run = etree.SubElement(p, qn("w:r"))
    texto = etree.SubElement(run, qn("w:t"))
    texto.text = "Formulario de cierre"
    tbl = etree.SubElement(sdt_content, qn("w:tbl"))
    tr = etree.SubElement(tbl, qn("w:tr"))
    tc1 = etree.SubElement(tr, qn("w:tc"))
    p1 = etree.SubElement(tc1, qn("w:p"))
    r1 = etree.SubElement(p1, qn("w:r"))
    t1 = etree.SubElement(r1, qn("w:t"))
    t1.text = "Provision"
    tc2 = etree.SubElement(tr, qn("w:tc"))
    p2 = etree.SubElement(tc2, qn("w:p"))
    r2 = etree.SubElement(p2, qn("w:r"))
    t2 = etree.SubElement(r2, qn("w:t"))
    t2.text = "4800000"
    sect_pr = body.find(qn("w:sectPr"))
    if sect_pr is not None:
        body.remove(sect_pr)
        body.append(sect_pr)
    origen = tmp_path / "sdt.docx"
    d.save(origen)

    r = word.extraer(origen)

    assert r.estado == "ok"
    md = r.salidas["texto.md"]
    assert "Formulario de cierre" in md
    assert "Provision | 4800000" in md
    # el detalle NO puede contar cero párrafos habiendo dos reales.
    assert r.detalle["parrafos"] >= 2


def test_cuadro_de_texto_se_cuenta_y_declara_con_parcial(tmp_path: Path):
    from docx import Document
    from docx.oxml.ns import qn
    from lxml import etree

    d = Document()
    p = d.add_paragraph("Cuerpo con contenido suficiente para no quedar vacio")
    run = p.add_run()
    drawing = etree.SubElement(run._r, qn("w:drawing"))
    wps_ns = "http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
    w_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    txbx = etree.SubElement(drawing, f"{{{wps_ns}}}txbx")
    txbx_content = etree.SubElement(txbx, f"{{{w_ns}}}txbxContent")
    pp = etree.SubElement(txbx_content, qn("w:p"))
    rr = etree.SubElement(pp, qn("w:r"))
    tt = etree.SubElement(rr, qn("w:t"))
    tt.text = "Provision por litigio: 750,000 USD"
    origen = tmp_path / "textbox.docx"
    d.save(origen)

    r = word.extraer(origen)

    assert r.estado == "parcial"
    assert r.detalle["cuadros_de_texto_omitidos"] == 1
    # NO se extrae -- se cuenta y se declara, no se inventa su contenido.
    assert "750,000" not in r.salidas["texto.md"]


def test_nota_al_pie_se_cuenta_y_declara_con_parcial(tmp_path: Path):
    from docx import Document
    from docx.oxml.ns import qn
    from lxml import etree

    d = Document()
    p = d.add_paragraph("Cuerpo con nota al pie referenciada")
    run = p.add_run()
    ref = etree.SubElement(run._r, qn("w:footnoteReference"))
    ref.set(qn("w:id"), "1")
    origen = tmp_path / "nota.docx"
    d.save(origen)

    r = word.extraer(origen)

    assert r.estado == "parcial"
    assert r.detalle["notas_al_pie_omitidas"] == 1


# ---------------------------------------------------------------------------
# I-4: celdas combinadas
# ---------------------------------------------------------------------------


def test_celda_combinada_horizontalmente_no_se_duplica(tmp_path: Path):
    """El caso EXACTO del hallazgo: "Total" fusionada sobre dos columnas.
    `fila.cells` la trae dos veces (mismo `_tc`) -- tiene que aparecer UNA
    sola vez en la fila de salida."""
    from docx import Document

    d = Document()
    t = d.add_table(rows=2, cols=2)
    a = t.cell(0, 0)
    b = t.cell(0, 1)
    fusionada = a.merge(b)
    fusionada.text = "Total"
    t.cell(1, 0).text = "Sub1"
    t.cell(1, 1).text = "Sub2"
    origen = tmp_path / "merge.docx"
    d.save(origen)

    r = word.extraer(origen)

    assert r.estado == "ok"
    md = r.salidas["texto.md"]
    lineas_con_total = [linea for linea in md.splitlines() if "Total" in linea]
    assert len(lineas_con_total) == 1
    assert lineas_con_total[0].count("Total") == 1


# ---------------------------------------------------------------------------
# I-5: un .xlsx con extensión .docx
# ---------------------------------------------------------------------------


def test_xlsx_con_extension_docx_da_sin_extractor(tmp_path: Path):
    openpyxl = pytest.importorskip("openpyxl")

    wb = openpyxl.Workbook()
    wb.active["A1"] = "hola"
    origen = tmp_path / "es-un-excel.docx"
    wb.save(origen)

    r = word.extraer(origen)

    assert r.estado == "sin_extractor"
    assert r.salidas == {}
    assert "spreadsheetml" in r.detalle["razon"]


# ---------------------------------------------------------------------------
# I-6: ninguna fila se promueve a encabezado, y no hay fila separadora
# ---------------------------------------------------------------------------


def test_ninguna_tabla_trae_fila_separadora_de_encabezado(tmp_path: Path):
    """I-6: el formato viejo (`| --- | --- |`) promovía la primera fila a
    encabezado -- con el propio fixture del brief, "1000" quedaba de
    nombre de columna. El formato nuevo (bloque cercado, igual que
    `pdf.py`) NUNCA emite una fila separadora."""
    r = word.extraer(_documento_brief(tmp_path / "d.docx"))
    assert "---" not in r.salidas["texto.md"]


# ---------------------------------------------------------------------------
# D-1 (task-9-brief.md, 2026-09-21): `objeto.style` es `None` en documentos
# reales -- medido: 3 de cada 4 .docx reales del lote de 23 documentos
# financieros. `(objeto.style.name or "").lower()` levantaba AttributeError
# CRUDO, escapando del módulo -- contradice el contrato de los cuatro
# extractores (siempre Resultado, nunca excepción).
# ---------------------------------------------------------------------------


def test_parrafo_con_style_none_no_revienta_y_se_trata_como_normal(
    tmp_path: Path, monkeypatch
):
    """El caso EXACTO del hallazgo: `Paragraph.style` es `None` (python-docx
    devuelve `None` cuando el documento no define un estilo por defecto para
    párrafo -- "no común" según el propio docstring de python-docx, pero
    medido en 3 de 4 documentos reales). Antes: `AttributeError: 'NoneType'
    object has no attribute 'name'`, crudo, fuera de `Resultado`. Ahora: se
    trata como "sin estilo" -- párrafo normal, sin encabezado -- y el
    extracto sigue."""
    from docx.text.paragraph import Paragraph

    # El documento se construye y se guarda ANTES de parchear -- `python-docx`
    # necesita el setter real de `style` para armar el `.docx` (add_heading
    # asigna `paragraph.style = "Heading 1"`). El parche sólo cubre la
    # LECTURA, que es donde vive el defecto de D-1.
    origen = _documento_brief(tmp_path / "sin-estilo.docx")
    monkeypatch.setattr(Paragraph, "style", property(lambda self: None))
    r = word.extraer(origen)

    assert r.estado == "ok"
    md = r.salidas["texto.md"]
    # Con estilo None, "Estado de Resultados" ya NO puede salir como
    # encabezado (no hay forma de saber que era "Heading 1") -- pero el
    # texto tiene que sobrevivir como párrafo normal, no perderse.
    assert "Estado de Resultados" in md
    assert "# Estado de Resultados" not in md
    assert "Periodo 2026" in md


def test_excepcion_inesperada_en_el_cuerpo_da_resultado_de_error(
    tmp_path: Path, monkeypatch
):
    """No sólo `style=None`: CUALQUIER excepción inesperada leyendo el
    cuerpo tiene que salir como `Resultado(estado="error", ...)`, nunca
    escapar cruda -- igual que `pdf.py` protege su bucle de páginas (spec
    de la tarea D-1)."""
    from docx.text.paragraph import Paragraph

    def explota(self):
        raise RuntimeError("fallo inesperado simulado leyendo el estilo")

    origen = _documento_brief(tmp_path / "explota.docx")
    monkeypatch.setattr(Paragraph, "style", property(explota))
    r = word.extraer(origen)

    assert r.estado == "error"
    assert r.salidas == {}
    assert "RuntimeError" in r.detalle["razon"]


def test_niveles_de_encabezado_no_colapsan_a_uno(tmp_path: Path):
    """Un "Heading 2" tiene que salir como "## ", no "# " -- una mutación
    que colapsa todos los niveles a 1 no se nota si sólo se prueba un
    documento con Heading 1."""
    from docx import Document

    d = Document()
    d.add_heading("Titulo principal", level=1)
    d.add_heading("Subtitulo", level=2)
    origen = tmp_path / "niveles.docx"
    d.save(origen)

    r = word.extraer(origen)

    md = r.salidas["texto.md"]
    assert "# Titulo principal" in md
    assert "## Subtitulo" in md
    assert "## Titulo principal" not in md
