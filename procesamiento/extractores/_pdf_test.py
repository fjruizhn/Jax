"""PDF nativo -> Markdown, con las tablas conservadas -- y la compuerta que
decide si un PDF necesita OCR.

pdftotext NO sirve: destruye las tablas, y en un expediente financiero la
tabla ES el documento. pdfplumber conserva filas y columnas.

Los fixtures son PDF 1.4 escritos a mano, byte a byte (sin dependencias de
generación). El ruling del controlador (task-4-brief.md) preveía que
pdfminer podía rechazar un PDF sin tabla `xref` y pedía recurrir a
LibreOffice en ese caso -- verificado contra pdfplumber==0.11.10 el
2026-09-21: todos los fixtures de abajo (incluido uno de 40 páginas)
parsean sin problema con el modo de recuperación de pdfminer.six, así que
no hace falta LibreOffice acá -- una dependencia externa menos en CI.

Ronda 2 (2026-09-21, task-4-hallazgos.md -- NO-GO del auditor, 3 críticos):
- C-1: el umbral por página NO cierra la familia del defecto por sí solo
  ("Escaneado con CamScanner" tiene 24 caracteres exactos, pasaba el
  umbral viejo de 24). El arreglo real es descontar el texto que se repite
  en la mayoría de las páginas -- `test_un_pdf_escaneado_con_sello_repetido_no_se_declara_ok`
  reproduce el caso EXACTO del hallazgo.
- C-2: `tiene_capa_de_texto()` y `extraer()` compartían la regla en teoría
  pero no en código -- divergieron. `test_tiene_capa_de_texto_usa_el_mismo_descuento_que_extraer`
  reproduce el caso exacto del hallazgo (10 páginas de "CONFIDENCIAL - COPIA").
- C-3: el piso por página no tenía NINGÚN test que lo matara solo (con un
  fixture de una sola página, el umbral global lo salvaba por casualidad).
  `test_el_piso_por_pagina_solo_lo_mata_un_fixture_multipagina` usa un
  fixture de 3 páginas donde el umbral por página es la ÚNICA razón por la
  que el resultado no es 'ok'.
- I-3: `test_el_importe_de_una_tabla_no_sale_duplicado` cuenta cuántas
  veces aparece un importe en el extracto -- tiene que ser UNA.
"""
from pathlib import Path

import pytest

pdfplumber = pytest.importorskip("pdfplumber")

from procesamiento.extractores import pdf


def _pdf_una_pagina(destino: Path, stream: bytes, mediabox: str = "0 0 200 200") -> Path:
    """Arma un PDF de UNA página con el content stream dado."""
    partes = [
        b"%PDF-1.4\n",
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n",
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n",
        f"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[{mediabox}]".encode()
        + b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n",
        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n",
        b"5 0 obj<</Length " + str(len(stream)).encode() + b">>stream\n",
        stream,
        b"endstream endobj\n",
        b"trailer<</Root 1 0 R>>\n",
    ]
    destino.write_bytes(b"".join(partes))
    return destino


def _pdf_multi_pagina(destino: Path, textos: list[str]) -> Path:
    """Arma un PDF de N páginas, cada una con UNA línea de texto (sin
    tabla `xref`, igual que el resto de los fixtures de este archivo)."""
    n = len(textos)
    kids = " ".join(f"{3 + i} 0 R" for i in range(n))
    font_obj = 3 + n
    partes = [
        b"%PDF-1.4\n",
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n",
        f"2 0 obj<</Type/Pages/Kids[{kids}]/Count {n}>>endobj\n".encode(),
    ]
    for i in range(n):
        pagina_num = 3 + i
        contenido_num = font_obj + 1 + i
        partes.append(
            (
                f"{pagina_num} 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]"
                f"/Resources<</Font<</F1 {font_obj} 0 R>>>>"
                f"/Contents {contenido_num} 0 R>>endobj\n"
            ).encode()
        )
    partes.append(f"{font_obj} 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n".encode())
    for i, texto in enumerate(textos):
        contenido_num = font_obj + 1 + i
        stream = f"BT /F1 10 Tf 20 100 Td ({texto}) Tj ET\n".encode()
        partes.append(f"{contenido_num} 0 obj<</Length {len(stream)}>>stream\n".encode())
        partes.append(stream)
        partes.append(b"endstream endobj\n")
    partes.append(b"trailer<</Root 1 0 R>>\n")
    destino.write_bytes(b"".join(partes))
    return destino


def _pdf_con_texto(destino: Path) -> Path:
    """PDF con capa de texto REAL, escrito a mano (sin dependencias): un
    párrafo, no cuatro palabras -- el umbral es la defensa contra un PDF
    escaneado con un poco de texto suelto encima; el fixture tiene que
    representar contenido de verdad, no acomodarse al umbral."""
    stream = (
        b"BT /F1 12 Tf 20 180 Td (ACTIVOS TOTALES 1234) Tj ET\n"
        b"BT /F1 10 Tf 20 160 Td "
        b"(Estado de Situacion Financiera al cierre del periodo,) Tj ET\n"
        b"BT /F1 10 Tf 20 145 Td "
        b"(con el detalle completo de las cuentas patrimoniales y del) Tj ET\n"
        b"BT /F1 10 Tf 20 130 Td "
        b"(resultado del ejercicio fiscal correspondiente al periodo.) Tj ET\n"
    )
    return _pdf_una_pagina(destino, stream, mediabox="0 0 300 220")


def _pdf_sin_texto(destino: Path) -> Path:
    """PDF de una página sin ningún content stream -- lo que produce un
    escaneo puesto en un PDF sin pasar por OCR: sólo la imagen, cero texto."""
    partes = [
        b"%PDF-1.4\n",
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n",
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n",
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n",
        b"trailer<</Root 1 0 R>>\n",
    ]
    destino.write_bytes(b"".join(partes))
    return destino


def _pdf_con_tabla(destino: Path) -> Path:
    """PDF de una página con una tabla REAL de 6 filas x 2 columnas --
    líneas vectoriales formando la grilla, más texto dentro de cada celda.
    Seis filas (no dos) para que el texto CRUDO de la página -- lo que
    clasifica si la página "tiene texto", independiente del bloque de la
    tabla -- por sí solo supere MINIMO_CARACTERES_PAGINA (80)."""
    filas = [
        ("Cuenta", "2025"),
        ("Efectivo y equivalentes", "1000"),
        ("Cuentas por cobrar", "2500"),
        ("Inventarios", "3200"),
        ("Propiedad planta y equipo", "8700"),
        ("Patrimonio neto", "9500"),
    ]
    alto_fila = 22
    x_izq, x_div, x_der = 20, 220, 320
    y_top = 40 + alto_fila * len(filas)
    comandos = [b"1 w\n"]
    for i in range(len(filas) + 1):
        y = y_top - i * alto_fila
        comandos.append(f"{x_izq} {y} m {x_der} {y} l S\n".encode())
    y_bottom = y_top - alto_fila * len(filas)
    for x in (x_izq, x_div, x_der):
        comandos.append(f"{x} {y_top} m {x} {y_bottom} l S\n".encode())
    for i, (izq, der) in enumerate(filas):
        y_texto = y_top - i * alto_fila - int(alto_fila * 0.65)
        comandos.append(f"BT /F1 9 Tf {x_izq + 5} {y_texto} Td ({izq}) Tj ET\n".encode())
        comandos.append(f"BT /F1 9 Tf {x_div + 5} {y_texto} Td ({der}) Tj ET\n".encode())
    stream = b"".join(comandos)
    return _pdf_una_pagina(destino, stream, mediabox=f"0 0 340 {y_top + 40}")


def _pdf_con_tabla_multilinea(destino: Path) -> Path:
    """Tabla de 3x2 donde la celda superior-izquierda tiene DOS líneas de
    texto (un rótulo que envuelve, I-1) -- pdfplumber devuelve esa celda
    como `"Cuentas por\\ncobrar diversas"`. Tres filas (no dos) para que el
    texto crudo de la página por sí solo supere MINIMO_CARACTERES_PAGINA."""
    stream = (
        b"1 w\n"
        b"20 200 m 180 200 l S\n"
        b"20 170 m 180 170 l S\n"
        b"20 140 m 180 140 l S\n"
        b"20 110 m 180 110 l S\n"
        b"20 200 m 20 110 l S\n"
        b"100 200 m 100 110 l S\n"
        b"180 200 m 180 110 l S\n"
        b"BT /F1 9 Tf 25 188 Td (Cuentas por) Tj ET\n"
        b"BT /F1 9 Tf 25 178 Td (cobrar diversas) Tj ET\n"
        b"BT /F1 9 Tf 110 183 Td (450) Tj ET\n"
        b"BT /F1 9 Tf 25 150 Td (Caja y equivalentes de efectivo) Tj ET\n"
        b"BT /F1 9 Tf 110 150 Td (990) Tj ET\n"
        b"BT /F1 9 Tf 25 120 Td (Inventarios de mercaderia disponible) Tj ET\n"
        b"BT /F1 9 Tf 110 120 Td (3200) Tj ET\n"
    )
    return _pdf_una_pagina(destino, stream, mediabox="0 0 200 220")


def _pdf_hibrido(destino: Path) -> Path:
    """Dos páginas: la primera con texto real (suficiente para superar
    MINIMO_CARACTERES_PAGINA), la segunda sin ningún content stream --
    una página que es pura imagen mezclada con una que sí es nativa."""
    stream1 = (
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
        b"2 0 obj<</Type/Pages/Kids[3 0 R 6 0 R]/Count 2>>endobj\n",
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 220]"
        b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n",
        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n",
        b"5 0 obj<</Length " + str(len(stream1)).encode() + b">>stream\n",
        stream1,
        b"endstream endobj\n",
        # página 2: sin /Contents -- ninguna capa de texto, como una imagen pura.
        b"6 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 220]>>endobj\n",
        b"trailer<</Root 1 0 R>>\n",
    ]
    destino.write_bytes(b"".join(partes))
    return destino


def _pdf_con_marca_de_agua(destino: Path) -> Path:
    """Una página escaneada con un sello/marca de agua corto estampado por
    el escáner -- hay ALGO de texto (no es el caso vacío de
    `_pdf_sin_texto`), pero es corto y no es contenido real del documento."""
    stream = b"BT /F1 10 Tf 20 100 Td (CONFIDENCIAL - COPIA) Tj ET\n"
    return _pdf_una_pagina(destino, stream)


def test_detecta_que_un_pdf_nativo_tiene_texto(tmp_path: Path):
    assert pdf.tiene_capa_de_texto(_pdf_con_texto(tmp_path / "n.pdf")) is True


def test_tiene_capa_de_texto_no_traga_modulenotfounderror(tmp_path: Path, monkeypatch):
    """C-2 (final-hallazgos.md, ronda de cierre): antes, CUALQUIER
    excepción (incluido `ModuleNotFoundError` por `pdfplumber` ausente) se
    trataba acá como "sin texto" -- la compuerta ruteaba entonces a OCR,
    que devolvía `estado='ok'` con las cifras DESTRUIDAS (medido: 8/8
    cifras con pdfplumber, 0/8 sin él, y `detalle` sin decir una palabra
    de la dependencia faltante). Ahora `ModuleNotFoundError` se deja
    propagar -- es `compuerta.extraer` quien decide qué hacer con eso."""
    import sys

    monkeypatch.setitem(sys.modules, "pdfplumber", None)
    with pytest.raises(ModuleNotFoundError):
        pdf.tiene_capa_de_texto(_pdf_con_texto(tmp_path / "n.pdf"))


def test_extrae_el_texto_del_pdf_nativo(tmp_path: Path):
    r = pdf.extraer(_pdf_con_texto(tmp_path / "n.pdf"))
    assert r.estado == "ok"
    assert "ACTIVOS TOTALES 1234" in r.salidas["texto.md"]
    assert r.detalle["paginas"] == 1


def test_un_pdf_roto_da_error_sin_extracto(tmp_path: Path):
    malo = tmp_path / "roto.pdf"
    malo.write_bytes(b"no soy un pdf")
    r = pdf.extraer(malo)
    assert r.estado == "error"
    assert r.salidas == {}


def test_un_pdf_sin_capa_de_texto_no_se_declara_ok(tmp_path: Path):
    """Un PDF de imagen pura NO se resuelve acá: se manda a OCR. Lo que NO
    puede pasar es que devuelva 'ok' con un extracto vacío."""
    vacio = _pdf_sin_texto(tmp_path / "escaneado.pdf")
    assert pdf.tiene_capa_de_texto(vacio) is False
    r = pdf.extraer(vacio)
    assert r.estado == "error"
    assert r.salidas == {}


def test_una_tabla_real_conserva_filas_y_columnas(tmp_path: Path):
    """Aseverá el CONTENIDO de la tabla, no sólo que extraer() no explote:
    esto es lo que pdftotext destruiría. Formato nuevo (I-4): bloque
    cercado, sin fila de encabezado."""
    r = pdf.extraer(_pdf_con_tabla(tmp_path / "tabla.pdf"))
    assert r.estado == "ok"
    md = r.salidas["texto.md"]
    assert "```tabla" in md
    assert "Cuenta | 2025" in md
    assert "Efectivo y equivalentes | 1000" in md
    assert "Patrimonio neto | 9500" in md
    assert r.detalle["tablas"] == 1


def test_el_importe_de_una_tabla_no_sale_duplicado(tmp_path: Path):
    """I-3: `extract_text()` de la página completa ya trae el texto de las
    celdas; si la tabla se emite ADEMÁS como bloque, cada importe sale dos
    veces. Un modelo que sume, o una conciliación por `grep`, cuenta doble."""
    r = pdf.extraer(_pdf_con_tabla(tmp_path / "tabla.pdf"))
    md = r.salidas["texto.md"]
    assert md.count("8700") == 1
    assert md.count("Cuentas por cobrar") == 1


def test_una_celda_con_salto_de_linea_no_parte_la_fila(tmp_path: Path):
    """I-1: un rótulo que envuelve (dos líneas dentro de la MISMA celda)
    no puede partir la fila y dejar el importe huérfano en otra línea."""
    r = pdf.extraer(_pdf_con_tabla_multilinea(tmp_path / "multilinea.pdf"))
    assert r.estado == "ok"
    md = r.salidas["texto.md"]
    assert "Cuentas por cobrar diversas | 450" in md


def test_un_pdf_hibrido_da_parcial_y_lo_dice_en_el_detalle(tmp_path: Path):
    """Página 1 con texto real, página 2 sin ninguna capa de texto (imagen
    pura). Esto NO es 'ok' -- salió MENOS de lo que el documento tiene, así
    que tiene que ser 'parcial', y el detalle tiene que decir CUÁL página
    quedó sin resolver, no callarlo."""
    r = pdf.extraer(_pdf_hibrido(tmp_path / "hibrido.pdf"))
    assert r.estado == "parcial"
    assert "ACTIVOS TOTALES 1234" in r.salidas["texto.md"]
    assert r.detalle["paginas"] == 2
    assert r.detalle["paginas_sin_texto"] == [2]
    # M12: la atribución de página no puede mentir -- es el dato con el
    # que se cruza `paginas_sin_texto`.
    assert "<!-- página 1 -->" in r.salidas["texto.md"]
    assert "<!-- página 2 -->" in r.salidas["texto.md"]


def test_una_marca_de_agua_corta_no_se_declara_ok(tmp_path: Path):
    """Una página escaneada cuyo único 'texto' es un sello corto NO puede
    salir 'ok' con eso como extracto -- sería declarar resuelto un
    documento que en realidad es una imagen con un sello encima, y que
    nadie manda a OCR después porque el sistema ya dijo que estaba bien."""
    r = pdf.extraer(_pdf_con_marca_de_agua(tmp_path / "sello.pdf"))
    assert r.estado == "error"
    assert r.salidas == {}


def test_un_pdf_escaneado_con_sello_repetido_no_se_declara_ok(tmp_path: Path):
    """C-1, el caso EXACTO del hallazgo: 40 páginas, cada una con SÓLO
    "Escaneado con CamScanner" (24 caracteres EXACTOS -- el umbral viejo
    de 24 con `< 24` lo dejaba pasar). Subir el número no alcanza: lo que
    cierra la familia entera del defecto es descontar el texto que se
    repite en la mayoría de las páginas. Acá se repite en el 100 %."""
    origen = _pdf_multi_pagina(
        tmp_path / "camscanner.pdf", ["Escaneado con CamScanner"] * 40
    )
    r = pdf.extraer(origen)
    assert r.estado != "ok"
    assert r.salidas == {}
    assert r.detalle["paginas"] == 40


def test_tiene_capa_de_texto_usa_el_mismo_descuento_que_extraer(tmp_path: Path):
    """C-2, el caso EXACTO del hallazgo: 10 páginas con SÓLO
    "CONFIDENCIAL - COPIA" (20 caracteres, por debajo de ambos umbrales
    incluso sin descuento). `tiene_capa_de_texto()` -- LA compuerta que
    decide si se paga OCR -- tiene que usar la MISMA cuenta por página que
    `extraer()`, no una copia separada que sume el documento entero."""
    origen = _pdf_multi_pagina(
        tmp_path / "confidencial10.pdf", ["CONFIDENCIAL - COPIA"] * 10
    )
    assert pdf.tiene_capa_de_texto(origen) is False


def test_el_piso_por_pagina_solo_lo_mata_un_fixture_multipagina(tmp_path: Path):
    """C-3: el piso por página (MINIMO_CARACTERES_PAGINA) necesita un test
    donde sea la ÚNICA razón por la que el resultado no es 'ok' -- un
    fixture de una sola página lo salva el umbral global "por casualidad".
    Acá: 3 páginas, cada una con el MISMO texto (así no dispara el
    descuento por repetición -- se repite en el 100 %, ver el test de
    arriba) -- en cambio, este texto es DISTINTO en cada página pero cada
    uno por separado queda BAJO los 80 caracteres del piso por página, y
    ninguno se repite lo suficiente entre sí como para activar el
    descuento (cada línea aparece en 1 de 3 páginas, 33 % < 60 %)."""
    origen = _pdf_multi_pagina(
        tmp_path / "corto3.pdf",
        [
            "Recibido en ventanilla numero uno del banco",
            "Procesado por el area de creditos comerciales",
            "Archivado en el expediente fisico correspondiente",
        ],
    )
    r = pdf.extraer(origen)
    assert r.estado == "error"
    assert r.salidas == {}
    assert r.detalle["paginas"] == 3


def test_extraer_sin_pdfplumber_instalado_da_sin_extractor(tmp_path: Path, monkeypatch):
    """Menor 3: un `ModuleNotFoundError` crudo no es un resultado -- existe
    el estado 'sin_extractor' justo para esto."""
    import sys

    monkeypatch.setitem(sys.modules, "pdfplumber", None)
    r = pdf.extraer(_pdf_con_texto(tmp_path / "n.pdf"))
    assert r.estado == "sin_extractor"
    assert r.salidas == {}


def test_version_no_revienta_si_pdfplumber_no_esta_instalado(monkeypatch):
    """Ronda P10 (2026-09-21), segundo arreglo del ruling del coordinador:
    `_version()` se llama desde `ingesta._version_vigente` FUERA de
    `extraer()` -- ahí no hay ningún `except ModuleNotFoundError` que
    convierta el fallo en 'sin_extractor'. Antes de este arreglo, `import
    pdfplumber` roto acá dejaba escapar un `ImportError` crudo (blindaje que
    `word.py`/`ocr.py` ya tenían y `pdf.py` no). Ahora, igual que ellos,
    nunca revienta.

    Menor 10 (final-hallazgos.md, ronda de cierre): el sentinel de "no se
    pudo determinar" es `None`, NUNCA la cadena "desconocida" -- esa cadena
    compara IGUAL A SÍ MISMA en dos fallos consecutivos, y
    `ingesta._version_vigente` la reenvía tal cual para decidir si el
    caché sigue siendo válido (I-2). Contra el código viejo esto falla:
    `_version()` devolvía la cadena "desconocida", no `None`."""
    import sys

    monkeypatch.setitem(sys.modules, "pdfplumber", None)
    version = pdf._version()
    assert version is None


def test_tabla_a_bloque_escapa_el_pipe_de_una_celda():
    """M9: si no se escapa, un '|' dentro de una celda se confunde con el
    separador de columnas y desalinea el resto de la fila."""
    bloque = pdf._tabla_a_bloque([["Ingresos|Egresos", "100"]])
    assert "Ingresos\\|Egresos | 100" in bloque


def test_tabla_a_bloque_reemplaza_salto_de_linea_de_celda_por_espacio():
    """I-1 sobre la función pura: un '\\n' dentro de una celda no puede
    generar una línea de más en el bloque -- sólo debe haber una línea por
    fila de la tabla."""
    bloque = pdf._tabla_a_bloque([["Cuentas por\ncobrar diversas", "450"]])
    assert "Cuentas por cobrar diversas | 450" in bloque
    # una sola fila de contenido -> 3 lineas en total: apertura, fila, cierre.
    assert bloque.count("\n") == 2


def test_tabla_a_bloque_rellena_filas_cortas_para_conservar_columnas():
    """M11: una fila con menos celdas que las demás tiene que rellenarse
    para conservar el número de columnas de la tabla."""
    bloque = pdf._tabla_a_bloque([["A", "B", "C"], ["D"]])
    lineas = bloque.splitlines()
    assert lineas[0] == "```tabla"
    assert lineas[1] == "A | B | C"
    assert lineas[2] == "D |  | "
    assert lineas[3] == "```"


def test_tabla_a_bloque_no_promueve_ninguna_fila_a_encabezado():
    """I-4: no hay fila de separador '---' ni fila promovida a título de
    columna -- una tabla que continúa de la página anterior no puede
    perder su primera fila de datos."""
    bloque = pdf._tabla_a_bloque([["Caja", "450"], ["Banco", "990"]])
    assert "---" not in bloque
    assert bloque == "```tabla\nCaja | 450\nBanco | 990\n```"
