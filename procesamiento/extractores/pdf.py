"""PDF nativo -> Markdown, con las tablas conservadas.

pdftotext NO sirve acá: destruye las tablas, y en un expediente financiero la
tabla ES el documento. pdfplumber conserva filas y columnas porque detecta
las líneas vectoriales que las dibujan, no adivina por espacios.

La detección de capa de texto es la compuerta entre este extractor y el de
OCR: determinista, barata, sin modelos. Un PDF puede ser:

- nativo: ninguna página queda clasificada "sin texto" -> 'ok'.
- escaneado: TODAS las páginas quedan sin texto (es pura imagen, o cada
  página trae sólo un sello/encabezado corto o repetido) -> no se resuelve
  acá, corresponde OCR. Nunca 'ok' con un extracto vacío o de puro ruido --
  'error', sin salidas, con la razón en el detalle.
- híbrido: algunas páginas quedan sin texto y otras no. Sale MENOS de lo
  que el documento tiene, así que es 'parcial' -- nunca 'ok' -- y el
  detalle dice explícitamente qué páginas quedaron sin resolver
  (`paginas_sin_texto`).

Ronda 2 de arreglo (2026-09-21, task-4-hallazgos.md — NO-GO del auditor,
3 críticos), tres lecciones que quedan documentadas porque son las que se
van a querer deshacer la próxima vez que un test se ponga rojo:

- C-1: un umbral absoluto de caracteres NUNCA cierra la familia del
  defecto -- siempre hay un sello más largo que el número elegido (medido:
  "Escaneado con CamScanner", 24 caracteres EXACTOS, pasaba el umbral
  viejo de 24 con `< 24`). El número sube (a 80), pero lo que de verdad
  cierra el problema es DESCONTAR el texto que se repite en la mayoría de
  las páginas (`_lineas_repetidas`) -- un sello no deja de serlo por tener
  muchas letras.
- C-2: `tiene_capa_de_texto()` (la compuerta) y `extraer()` tenían CADA UNA
  su propia cuenta de "esta página tiene texto" -- divergieron, y la
  compuerta se quedó sumando el documento entero cuando `extraer()` ya
  filtraba por página. Ahora las dos llaman a `_paginas_sin_texto()`, la
  única implementación de esa regla -- arreglarlo copiando el código a la
  otra función es exactamente lo que causó la divergencia.
- I-3: el texto de una tabla salía DOS veces -- una como prosa (de
  `extract_text()` de la página completa) y otra como tabla. La prosa se
  recorta AFUERA del área de cada tabla (`Page.outside_bbox`); las tablas
  se emiten una sola vez, vía `Page.find_tables()` + `Table.extract()`.

Ronda 3 de arreglo (2026-09-21, expediente real de 99 documentos -- 2
fallaron con el MISMO defecto, uno de ellos un avalúo de maquinaria):
- Defecto de pdfplumber ya conocido (coordenada negativa u otra caja fuera
  de los límites de la página): una tabla detectada puede traer un `bbox`
  que se sale del `bbox` de la página. Verificado sobre los dos documentos
  reales (2026-09-21): el `ValueError` NO sale de `Table.extract()` --
  `t.extract()` funciona perfecto aun con ese `bbox` fuera de rango. Sale
  de `Page.outside_bbox(t.bbox)`, que es el recorte que arma la PROSA
  alrededor de la tabla (I-3, arriba). Antes, esa excepción escapaba del
  bucle entero y `extraer()` la atrapaba en el `except` de apertura del
  documento -- 'error', cero páginas, el expediente entero perdido por una
  sola tabla mal delimitada en una de treinta páginas.
- Ahora el recorte de esa tabla puntual se ataja POR TABLA: si
  `outside_bbox()` (o `Table.extract()`) revienta para una tabla, esa
  tabla se cuenta en `detalle["tablas_fallidas"]` y se sigue con el resto.
  Como el recorte no avanzó para esa tabla, su área queda SIN cortar de la
  prosa -- se pierde la dedup de I-3 sólo para esa tabla puntual (su
  contenido sale una vez, como prosa cruda, no como bloque estructurado),
  pero el texto de la página no se pierde. El estado pasa a 'parcial'
  -- nunca 'ok' callando que una tabla no se pudo extraer.
"""
from __future__ import annotations

from pathlib import Path

from procesamiento.resultado import Resultado

EXTRACTOR = "pdfplumber"

# Piso de caracteres ÚTILES por página (después de descontar el texto
# repetido, ver `_lineas_repetidas`) para que esa página cuente como "con
# texto". Subir el número es la parte SECUNDARIA del arreglo de C-1 -- lo
# que de verdad distingue un sello de contenido es el descuento por
# repetición, no este valor. Aun así, la dirección importa: ante la duda,
# el umbral sube, no baja. Una página de contenido real que cae en
# 'parcial' por accidente falla barato (se manda a revisar de más). Una
# página de puro sello que se cuela como 'ok' falla caro (el documento se
# pierde en silencio, porque nadie lo manda a OCR después).
MINIMO_CARACTERES_PAGINA = 80

# Una línea (recortada de espacios) que aparece en esta proporción o más de
# las páginas del documento es encabezado, pie de página o sello de
# escaneo -- nunca contenido -- y no cuenta para el mínimo de ninguna
# página. Esto es lo que cierra la familia ENTERA del defecto de C-1: no
# importa cuántos caracteres tenga el sello, si se repite en la mayoría de
# las páginas, se descuenta entero. Con menos de 2 páginas no hay
# "mayoría" que comparar -- ver `_lineas_repetidas`.
UMBRAL_TEXTO_REPETIDO = 0.6


def _version() -> str | None:
    """Blindado (ronda P10, 2026-09-21): esta función se llama también desde
    `ingesta._version_vigente`, FUERA de `extraer()` -- ahí no hay ningún
    `except ModuleNotFoundError` que convierta el fallo en 'sin_extractor'.
    Nunca puede dejar escapar una excepción, igual que `word._version()` y
    `ocr._version()`.

    `None` cuando no se pudo determinar la versión -- NUNCA la cadena
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
        import pdfplumber

        return pdfplumber.__version__
    except Exception:  # fail-soft: el import o el atributo __version__ pueden fallar (paquete no instalado o roto); se devuelve None (no determinable) en vez de propagar
        return None


def _textos_por_pagina(origen: Path) -> list[str]:
    import pdfplumber

    with pdfplumber.open(origen) as doc:
        return [pagina.extract_text() or "" for pagina in doc.pages]


def _lineas_repetidas(textos: list[str]) -> set[str]:
    """Líneas que aparecen en >= UMBRAL_TEXTO_REPETIDO de las páginas --
    encabezado, pie o sello de escaneo, no contenido. Con menos de 2
    páginas no hay manera de distinguir "esto se repite" de "es la única
    página que hay", así que no se descuenta nada."""
    total = len(textos)
    if total < 2:
        return set()
    conteo: dict[str, int] = {}
    for texto in textos:
        for linea in {l.strip() for l in texto.splitlines() if l.strip()}:
            conteo[linea] = conteo.get(linea, 0) + 1
    minimo = total * UMBRAL_TEXTO_REPETIDO
    return {linea for linea, n in conteo.items() if n >= minimo}


def _longitud_efectiva(texto: str, repetidas: set[str]) -> int:
    """Caracteres de contenido de la página, SIN las líneas repetidas."""
    utiles = [
        l.strip() for l in texto.splitlines() if l.strip() and l.strip() not in repetidas
    ]
    return len(" ".join(utiles))


def _paginas_sin_texto(textos: list[str]) -> list[int]:
    """La ÚNICA implementación de "esta página tiene texto útil o no".
    `tiene_capa_de_texto()` y `extraer()` llaman a ESTA función -- antes
    cada una tenía su propia cuenta y divergieron (C-2, ronda 2026-09-21):
    la compuerta se quedó sumando el documento entero contra un umbral
    global mientras `extraer()` ya filtraba por página. Arreglar esto
    copiando el código a la otra función es lo que causó el problema la
    primera vez."""
    repetidas = _lineas_repetidas(textos)
    return [
        numero
        for numero, texto in enumerate(textos, start=1)
        if _longitud_efectiva(texto, repetidas) < MINIMO_CARACTERES_PAGINA
    ]


def tiene_capa_de_texto(origen: Path) -> bool:
    """C-2 (final-hallazgos.md, ronda de cierre): `ModuleNotFoundError` NO
    se traga acá -- antes CUALQUIER excepción (incluida la de pdfplumber
    ausente) se trataba como "sin texto" y la compuerta ruteaba el PDF a
    OCR, que devolvía `estado="ok"` con las cifras DESTRUIDAS (medido: 8/8
    con pdfplumber, 0/8 sin él, y `detalle` no decía nada de la
    dependencia faltante). Un fallo de despliegue produciendo cifras
    falsas etiquetadas 'ok' es el Principio VIII en su peor forma. Se deja
    propagar `ModuleNotFoundError` -- es `compuerta.extraer` quien decide
    qué hacer con eso (delegar en `pdf.extraer`, que ya sabe convertirlo en
    'sin_extractor', igual que un `.docx` sin `python-docx`). El resto de
    las excepciones (PDF corrupto, lectura rota) SIGUE tratándose como
    "sin texto", que es correcto: ahí sí corresponde OCR."""
    try:
        textos = _textos_por_pagina(origen)
    except ModuleNotFoundError:
        raise
    except Exception:  # fail-soft: la deteccion barata de capa de texto puede fallar leyendo el PDF (corrupto, formato raro); se trata como "sin texto" y la compuerta rutea a OCR en vez de abortar
        return False
    if not textos:
        return False
    return len(_paginas_sin_texto(textos)) < len(textos)


def _tabla_a_bloque(tabla: list[list]) -> str:
    """Tabla como bloque cercado, una fila por línea, celdas separadas por
    '|', SIN promover ninguna fila a encabezado (I-4): una tabla que
    continúa de la página anterior no pierde su primera fila de datos por
    tratarla como título de columna -- el consumidor es un modelo, no un
    lector humano, y la estructura importa más que la decoración. El '\\n'
    dentro de una celda (un rótulo que envuelve, I-1) se reemplaza por un
    espacio -- si no, parte la fila y el importe de esa fila queda
    huérfano. El '|' dentro de una celda se escapa -- si no, se confunde
    con el separador de columnas y desalinea el resto de la fila."""
    filas = [
        [
            "" if c is None else str(c).replace("\n", " ").replace("|", "\\|")
            for c in f
        ]
        for f in tabla
    ]
    if not filas:
        return ""
    ancho = max(len(f) for f in filas)
    filas = [f + [""] * (ancho - len(f)) for f in filas]
    cuerpo = "\n".join(" | ".join(f) for f in filas)
    return "```tabla\n" + cuerpo + "\n```"


def extraer(origen: Path) -> Resultado:
    try:
        import pdfplumber
    except ModuleNotFoundError as exc:
        return Resultado(
            estado="sin_extractor", salidas={}, extractor=EXTRACTOR,
            version="sin instalar",
            detalle={"razon": f"pdfplumber no esta instalado: {exc}"},
        )

    try:
        partes: list[str] = []
        textos: list[str] = []
        tablas = 0
        tablas_fallidas: list[int] = []
        with pdfplumber.open(origen) as doc:
            for numero, pagina in enumerate(doc.pages, start=1):
                # texto COMPLETO de la página (con tablas incluidas) -- es
                # lo que clasifica si la página "tiene texto" (duda 2 de la
                # ronda anterior, confirmada por el coordinador: no separar
                # "prosa" de "tabla" para esta cuenta).
                texto_completo = pagina.extract_text() or ""
                textos.append(texto_completo)

                # I-3: la prosa que se EMITE sí se recorta afuera de cada
                # tabla, para no repetir el contenido de las celdas. Ronda 3
                # (ver docstring del módulo): una tabla puntual con un bbox
                # fuera de la página (defecto conocido de pdfplumber, verificado
                # ahí como el origen REAL del ValueError -- no `Table.extract()`,
                # sino `outside_bbox()`) no puede tumbar la página entera. Se
                # ataja POR TABLA: si el recorte o la extracción de ESA tabla
                # revientan, se cuenta en `tablas_fallidas` y se sigue con el
                # resto. Como `recorte` no avanza para la tabla que falló, su
                # área queda SIN cortar de la prosa -- pierde la dedup de I-3
                # sólo para esa tabla puntual, no el contenido de la página.
                tablas_pagina = pagina.find_tables()
                recorte = pagina
                bloques_pagina: list[str] = []
                for t in tablas_pagina:
                    try:
                        recorte = recorte.outside_bbox(t.bbox)
                        bloque = _tabla_a_bloque(t.extract())
                    except Exception:  # fail-soft: bbox de esta tabla puntual fuera de la página (u otro fallo de ESTA tabla); se cuenta en tablas_fallidas y se sigue con el resto, sin tumbar la página ni el documento
                        tablas_fallidas.append(numero)
                        continue
                    if bloque:
                        tablas += 1
                        bloques_pagina.append(bloque)

                if tablas_pagina:
                    prosa = recorte.extract_text() or ""
                else:
                    prosa = texto_completo

                partes.append(f"<!-- página {numero} -->")
                if prosa.strip():
                    partes.append(prosa)
                partes.extend(bloques_pagina)
    except Exception as exc:  # fail-soft: apertura o lectura del PDF con pdfplumber puede fallar; se devuelve Resultado(estado="error") con el detalle en vez de propagar
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR,
            version=_version() or "desconocida",
            detalle={"razon": f"no se pudo leer: {type(exc).__name__}: {exc}"},
        )

    contenido = "\n\n".join(partes).strip()
    paginas = len(textos)
    paginas_sin_texto = _paginas_sin_texto(textos)

    # I-2: si TODAS las páginas quedaron sin texto, no hay nada rescatado
    # -- es 'error' (corresponde OCR), no 'parcial'. 'parcial' implica que
    # algo se rescató; entregar puro ruido (diez marcas de agua) bajo ese
    # nombre hace que un consumidor lo ingiera como si fuera contenido.
    if len(paginas_sin_texto) == paginas:
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR,
            version=_version() or "desconocida",
            detalle={
                "razon": "sin capa de texto util; corresponde OCR",
                "paginas": paginas,
            },
        )

    # Ronda 3: 'parcial' (nunca 'ok' callándolo) también cuando una o más
    # tablas no se pudieron extraer -- ver docstring del módulo y el bucle
    # de arriba. Las dos razones son independientes y pueden darse juntas
    # (un híbrido con además una tabla de bbox roto en su única página con
    # texto), así que se combinan en vez de que una tape a la otra.
    if paginas_sin_texto or tablas_fallidas:
        razones = []
        detalle: dict = {"paginas": paginas, "tablas": tablas}
        if paginas_sin_texto:
            detalle["paginas_sin_texto"] = paginas_sin_texto
            razones.append(
                "algunas paginas no tienen capa de texto util (probable "
                "imagen o solo sello); no se resuelven aca, corresponde OCR"
            )
        if tablas_fallidas:
            detalle["tablas_fallidas"] = {
                "cantidad": len(tablas_fallidas),
                "paginas": sorted(set(tablas_fallidas)),
            }
            razones.append(
                "una o mas tablas no se pudieron extraer (bbox fuera de la "
                "pagina, defecto conocido de pdfplumber); el texto de esas "
                "paginas se conserva igual, sin la tabla estructurada"
            )
        detalle["razon"] = " / ".join(razones)
        return Resultado(
            estado="parcial", salidas={"texto.md": contenido},
            extractor=EXTRACTOR, version=_version() or "desconocida",
            detalle=detalle,
        )

    return Resultado(
        estado="ok", salidas={"texto.md": contenido},
        extractor=EXTRACTOR, version=_version() or "desconocida",
        detalle={"paginas": paginas, "tablas": tablas},
    )
