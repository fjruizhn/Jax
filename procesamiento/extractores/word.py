"""Word (.docx) → Markdown. Un .docx es un ZIP con XML: el texto, los
títulos y las tablas ya están ahí, estructurados. No hace falta OCR ni
modelo.

Ronda de arreglo 1 (2026-09-21, task-5-6-hallazgos.md — NO ratificado, 4
críticos + 3 importantes sobre este archivo), lecciones que quedan acá
porque son las que se van a querer deshacer la próxima vez que un test se
ponga rojo:

- C-4: emitir TODOS los párrafos y DESPUÉS todas las tablas pierde el
  orden real del documento -- con dos secciones "Balance 2025" / tabla /
  "Balance 2026" / tabla, el extracto viejo entregaba los dos títulos
  juntos y las dos tablas después: el modelo atribuye las cifras al año
  equivocado, en verde. Se recorre `document.element.body` en el orden
  real del XML (`_iterar_bloques`), igual que `pdf.py` intercala prosa y
  tablas por página -- esto no es cosmético, es corrección. El recorrido
  entra también dentro de `w:sdt` (control de contenido): es la estructura
  de cualquier plantilla de formulario, así que quedarse solo en el nivel
  de `body` pierde justo lo más frecuente de lo que nos van a mandar.
- C-5: una tabla anidada dentro de una celda (`cell.tables`, no
  `cell.text`) desaparecía entera. Se resuelve con recursión
  (`_celda_texto` llama a `_tabla_anidada_texto`, que vuelve a llamar a
  `_celda_texto` por cada celda de la tabla anidada) -- cualquier
  profundidad de anidamiento, no sólo un nivel.
- C-6: `Paragraph.text` de python-docx sólo mira los `w:r` que son hijos
  DIRECTOS del párrafo -- un control de cambios envuelve los suyos en
  `w:ins`/`w:del`, así que un rótulo con una cifra en revisión salía SIN
  NINGÚN número. `<w:t>` (el texto de un run) nunca aparece dentro de
  `w:del` (que usa `w:delText`, otra etiqueta) -- iterar todos los `w:t`
  del párrafo (`_texto_parrafo`) da, sin necesidad de tratar `w:ins`/`w:del`
  aparte, el texto VIGENTE (el insertado, no el borrado). Se declara en
  `detalle["control_de_cambios"]` que el documento tiene cambios
  pendientes -- perder el valor viejo está bien si se dice.
- C-7: encabezado, pie, cuadros de texto, notas al pie y `w:sdt` se perdían
  en silencio, todos con `estado="ok"`. `w:sdt` se extrae (ver C-4).
  Encabezado y pie se extraen (`section.header`/`.footer`, ya los da
  python-docx). Cuadros de texto y notas al pie NO se extraen en esta fase
  (más trabajo del que corresponde) pero se CUENTAN y se declaran con
  `estado="parcial"` -- omitir declarándolo es honesto, omitirlo callando
  no.
- I-4: una celda combinada horizontalmente aparece UNA VEZ POR COLUMNA que
  ocupa en `fila.cells`, pero con el MISMO `_tc` subyacente -- sin
  deduplicar, "Total" fusionada sobre dos columnas salía
  `| Total | Total | 1000 |`. Se colapsan celdas consecutivas con
  identidad de `_tc` repetida.
- I-5: python-docx YA identifica el tipo real de un archivo mal nombrado
  (`... content type is '...spreadsheetml.sheet.main+xml'`) -- un Excel con
  extensión `.docx` ahora es `sin_extractor` (herramienta equivocada), no
  `error` (documento dañado).
- I-6: las tablas se emiten con el MISMO bloque cercado que `pdf.py`
  (` ```tabla `, una fila por línea, sin promover ninguna a encabezado) --
  la razón real no es "no hay continuación entre páginas" (como decía la
  ronda anterior): es que la PRIMERA FILA PUEDE SER DATO, y con el
  fixture del propio brief lo era (`1000` quedaba de nombre de columna).

Ronda de arreglo de la compuerta (2026-09-21, task-7-hallazgos.md, I-1):
`docx.oxml.ns.qn`, `docx.table.Table` y `docx.text.paragraph.Paragraph`
importaban A NIVEL DE MÓDULO -- al revés que `excel.py`, `pdf.py` y
`ocr.py`, que importan sus dependencias perezosamente DENTRO de cada
función. Con eso, sin `python-docx` instalado, `import
procesamiento.compuerta` (que importa los cuatro extractores) explotaba
ENTERO -- Excel, PDF y OCR se caían con él, con sus propias dependencias
sanas. El `try/except ModuleNotFoundError` de `extraer()` (abajo) era
código MUERTO: nunca se alcanzaba, porque el import de módulo ya había
reventado antes de que Python llegara a ejecutar una sola línea de
`extraer()`. Los tres imports pasan a ser locales, uno por función que
los usa (mismo patrón que los hermanos).
"""
from __future__ import annotations

from pathlib import Path

from procesamiento.resultado import Resultado

EXTRACTOR = "python-docx"

# Firma de las cabeceras OLE2 Compound File Binary Format -- el contenedor
# binario que usan .doc/.xls/.ppt "viejos" (pre-2007). Los primeros 8 bytes
# de CUALQUIER archivo así son EXACTAMENTE estos, sin excepción.
_FIRMA_OLE2 = bytes.fromhex("D0CF11E0A1B11AE1")


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("python-docx")
    except Exception:
        return "desconocida"


def _es_binario_ole2(origen: Path) -> bool:
    try:
        with open(origen, "rb") as fh:
            return fh.read(len(_FIRMA_OLE2)) == _FIRMA_OLE2
    except OSError:
        return False


def _texto_parrafo(p_elem) -> str:
    """Texto VIGENTE del párrafo -- incluye el texto insertado por control
    de cambios (`w:ins` envuelve un `w:r` normal, con `w:t`) y EXCLUYE el
    borrado (`w:del` envuelve un `w:r` con `w:delText`, una etiqueta
    distinta que este recorrido nunca visita). No hace falta tratar
    `w:ins`/`w:del` como casos aparte: `<w:t>` ya filtra los dos por
    construcción del propio formato.

    Además EXCLUYE cualquier `w:t` que esté dentro de un `txbxContent`
    (cuadro de texto, C-7): un cuadro de texto vive dentro del `w:drawing`
    de un run del párrafo que lo ancla, así que un recorrido ciego de
    `.iter(w:t)` lo mete en el texto principal -- justo lo que el ruling
    pide NO extraer (se cuenta y se declara aparte, no se cuela callado)."""
    return "".join(
        t.text or ""
        for t in p_elem.xpath('.//w:t[not(ancestor::*[local-name()="txbxContent"])]')
    )


def _tiene_cambios(elem) -> bool:
    from docx.oxml.ns import qn

    return (
        elem.find(".//" + qn("w:ins")) is not None
        or elem.find(".//" + qn("w:del")) is not None
    )


def _celdas_sin_fusion(fila) -> list:
    """`fila.cells` trae una entrada POR COLUMNA que ocupa la celda -- una
    combinada horizontalmente sobre dos columnas aparece dos veces, con el
    MISMO `_tc` subyacente. Se colapsan los tramos consecutivos que
    comparten identidad de `_tc` (I-4): una celda por tramo real, sin
    importar cuántas columnas de grilla ocupe."""
    vistas = []
    anterior = None
    for celda in fila.cells:
        if anterior is not None and celda._tc is anterior._tc:
            continue
        vistas.append(celda)
        anterior = celda
    return vistas


def _celda_texto(celda) -> str:
    """Texto de una celda: sus propios párrafos MÁS cualquier tabla anidada
    (C-5) -- una tabla dentro de una celda no vive en `celda.text` (que
    sólo concatena párrafos), vive en `celda.tables`."""
    partes = [
        t for t in (_texto_parrafo(p._p).strip() for p in celda.paragraphs) if t
    ]
    texto = " ".join(partes)
    for anidada in celda.tables:
        bloque = _tabla_anidada_texto(anidada)
        texto = f"{texto} {bloque}".strip() if texto else bloque
    return texto


def _tabla_anidada_texto(tabla) -> str:
    """Representación COMPACTA (una sola línea, sin '|' ni '\\n' propios)
    de una tabla anidada, para poder embeberla dentro de UNA celda de la
    tabla exterior sin romper el formato de fila de Markdown. Recursiva:
    `_celda_texto` vuelve a llamar acá por cada tabla anidada que encuentre
    dentro de sus propias celdas -- cualquier profundidad de anidamiento."""
    filas_repr = []
    for fila in tabla.rows:
        columnas = [_celda_texto(c) for c in _celdas_sin_fusion(fila)]
        filas_repr.append(" / ".join(columnas))
    return "[tabla anidada: " + "; ".join(filas_repr) + "]"


def _tabla_a_lineas(tabla) -> list[str]:
    """Bloque cercado (` ```tabla `), una fila por línea, celdas separadas
    por '|', SIN promover ninguna fila a encabezado (I-6, mismo formato que
    `pdf._tabla_a_bloque`): la primera fila puede ser dato, no título de
    columna -- con el fixture del propio brief de esta tarea, lo era."""
    filas = [
        [
            _celda_texto(celda).replace("\n", " ").replace("|", "\\|")
            for celda in _celdas_sin_fusion(fila)
        ]
        for fila in tabla.rows
    ]
    if not filas:
        return []
    ancho = max(len(f) for f in filas)
    filas = [f + [""] * (ancho - len(f)) for f in filas]
    cuerpo = "\n".join(" | ".join(f) for f in filas)
    return ["", "```tabla", cuerpo, "```"]


def _iterar_bloques(contenedor_xml, documento):
    """Genera `('parrafo', Paragraph)` o `('tabla', Table)` en el orden
    REAL del documento (C-4) -- entra también dentro de `w:sdt` (control de
    contenido, C-7): es la estructura de cualquier plantilla de formulario,
    y quedarse sólo en el nivel de `body` la perdería entera."""
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in contenedor_xml.iterchildren():
        if child.tag == qn("w:p"):
            yield ("parrafo", Paragraph(child, documento))
        elif child.tag == qn("w:tbl"):
            yield ("tabla", Table(child, documento))
        elif child.tag == qn("w:sdt"):
            contenido = child.find(qn("w:sdtContent"))
            if contenido is not None:
                yield from _iterar_bloques(contenido, documento)
        # otros hijos de body (sectPr, bookmarkStart, ...) se ignoran.


def _encabezado_y_pie(documento) -> list[tuple[str, str]]:
    """Encabezado y pie de cada sección (C-7) -- python-docx ya los da por
    `section.header`/`.footer`. `is_linked_to_previous` evita repetir el
    mismo encabezado/pie por defecto de cada sección adicional."""
    fragmentos: list[tuple[str, str]] = []
    vistos: set[str] = set()
    for section in documento.sections:
        for parte, etiqueta in ((section.header, "encabezado"), (section.footer, "pie")):
            if parte.is_linked_to_previous:
                continue
            texto = " ".join(
                t for t in (_texto_parrafo(p._p).strip() for p in parte.paragraphs) if t
            )
            if texto and texto not in vistos:
                vistos.add(texto)
                fragmentos.append((etiqueta, texto))
    return fragmentos


def extraer(origen: Path) -> Resultado:
    try:
        from docx import Document
        from docx.oxml.ns import qn
    except ModuleNotFoundError as exc:
        return Resultado(
            estado="sin_extractor", salidas={}, extractor=EXTRACTOR,
            version="sin instalar",
            detalle={"razon": f"python-docx no esta instalado: {exc}"},
        )

    origen = Path(origen)

    if _es_binario_ole2(origen):
        return Resultado(
            estado="sin_extractor", salidas={}, extractor=EXTRACTOR,
            version=_version(),
            detalle={
                "razon": (
                    "el archivo es un binario de Office antiguo (formato "
                    "OLE2); esta herramienta no lee ese formato -- hace "
                    "falta LibreOffice, fuera de alcance de esta fase"
                ),
            },
        )

    try:
        documento = Document(str(origen))
    except Exception as exc:
        mensaje = str(exc)
        # I-5: python-docx YA identifica el tipo real de un archivo mal
        # nombrado ("... is not a Word file, content type is '...'"). Un
        # Excel o PowerPoint con extensión .docx es la herramienta
        # equivocada, no un documento dañado.
        if "is not a Word file" in mensaje:
            return Resultado(
                estado="sin_extractor", salidas={}, extractor=EXTRACTOR,
                version=_version(),
                detalle={"razon": f"el archivo no es un Word: {mensaje}"},
            )
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR, version=_version(),
            detalle={"razon": f"no se pudo abrir: {type(exc).__name__}: {exc}"},
        )

    lineas: list[str] = []
    parrafos_con_contenido = 0
    tablas_con_contenido = 0
    hay_cambios = False

    for tipo, objeto in _iterar_bloques(documento.element.body, documento):
        if tipo == "parrafo":
            texto = _texto_parrafo(objeto._p).strip()
            if _tiene_cambios(objeto._p):
                hay_cambios = True
            if not texto:
                continue
            parrafos_con_contenido += 1
            estilo = (objeto.style.name or "").lower()
            if estilo.startswith("heading"):
                nivel = "".join(c for c in estilo if c.isdigit()) or "1"
                lineas.append("#" * min(int(nivel), 6) + " " + texto)
            else:
                lineas.append(texto)
        else:  # tabla
            if _tiene_cambios(objeto._tbl):
                hay_cambios = True
            bloque = _tabla_a_lineas(objeto)
            if bloque:
                tablas_con_contenido += 1
                lineas.extend(bloque)

    fragmentos_header_footer = _encabezado_y_pie(documento)
    lineas_header_footer = [f"[{etiqueta}] {texto}" for etiqueta, texto in fragmentos_header_footer]

    contenido = "\n".join(lineas_header_footer + lineas).strip()
    if not contenido:
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR, version=_version(),
            detalle={"razon": "el documento no tiene texto"},
        )

    # C-7: cuadros de texto y notas al pie no se extraen en esta fase (más
    # trabajo del que corresponde) pero NO se pierden en silencio -- se
    # cuentan y se declaran, con 'parcial'.
    cajas_texto = len(documento.element.body.xpath("//*[local-name()='txbxContent']"))
    notas_pie = len(documento.element.body.findall(".//" + qn("w:footnoteReference")))

    detalle: dict = {
        "parrafos": parrafos_con_contenido,
        "tablas": tablas_con_contenido,
    }
    if hay_cambios:
        detalle["control_de_cambios"] = True
    if fragmentos_header_footer:
        detalle["encabezado_o_pie"] = True
    if cajas_texto:
        detalle["cuadros_de_texto_omitidos"] = cajas_texto
    if notas_pie:
        detalle["notas_al_pie_omitidas"] = notas_pie

    estado = "parcial" if (cajas_texto or notas_pie) else "ok"
    if estado == "parcial":
        detalle["razon"] = (
            "el documento tiene cuadros de texto y/o notas al pie que esta "
            "fase no extrae -- contenido omitido, no perdido en silencio"
        )

    return Resultado(
        estado=estado, salidas={"texto.md": contenido},
        extractor=EXTRACTOR, version=_version(), detalle=detalle,
    )
