"""PDF nativo -> Markdown, con las tablas conservadas.

pdftotext NO sirve acá: destruye las tablas, y en un expediente financiero la
tabla ES el documento. pdfplumber conserva filas y columnas porque detecta
las líneas vectoriales que las dibujan, no adivina por espacios.

La detección de capa de texto es la compuerta entre este extractor y el de
OCR: determinista, barata, sin modelos. Un PDF puede ser:

- nativo: tiene texto de verdad en TODAS sus páginas -> 'ok'.
- escaneado: ninguna página tiene texto (es pura imagen) -> no se resuelve
  acá, corresponde OCR. Nunca 'ok' con un extracto vacío -- 'error', sin
  salidas, con la razón en el detalle.
- híbrido: algunas páginas tienen texto y otras no (una hoja escaneada
  mezclada en un expediente nativo, o una página cuyo único "texto" es una
  marca de agua/sello corto). Sale MENOS de lo que el documento tiene, así
  que es 'parcial' -- nunca 'ok' -- y el detalle dice explícitamente qué
  páginas quedaron sin resolver (`paginas_sin_texto`), para que quien
  consuma el resultado sepa que ahí falta mandar a OCR.
"""
from __future__ import annotations

from pathlib import Path

from procesamiento.resultado import Resultado

EXTRACTOR = "pdfplumber"
# Menos que esto en TODO el documento = no hay capa de texto util (corresponde
# OCR). Ronda anterior (2026-09-21) lo bajó a 16 porque el fixture de prueba
# no llegaba a 32 -- eso era mover la medida para que encaje con la prueba,
# el mismo defecto que relajar una aserción. Se corrigió al revés: se agrandó
# el fixture, el umbral vuelve a 32.
#
# La dirección del umbral NO es simétrica, y por eso se elige del lado caro
# de sospechar de más:
#   - umbral DEMASIADO ALTO -> un PDF nativo con poco texto se manda a OCR
#     de más. Cuesta CPU. El contenido igual sale (por el camino de OCR).
#   - umbral DEMASIADO BAJO -> un PDF ESCANEADO que trae un poco de texto
#     suelto (numero de pagina, marca de agua, encabezado que estampa el
#     escaner) se toma por nativo, se extraen unos pocos caracteres y se
#     declara 'ok'. El documento entero se pierde EN SILENCIO -- nadie manda
#     eso a OCR porque el sistema ya dijo que estaba resuelto.
# Falla barato > falla caro: ante la duda, el umbral sube, no baja.
MINIMO_CARACTERES = 32

# Mismo criterio, por PÁGINA: una página individual con menos de esto se
# cuenta como 'sin texto' aunque el documento entero, sumando las demás
# páginas, supere MINIMO_CARACTERES. Sin este piso, un PDF escaneado de
# varias páginas donde cada una trae sólo un encabezado o numero de página
# corto (por debajo de esto, pero cuya SUMA cruza el umbral del documento)
# se clasificaría entero como con texto -- exactamente el mismo error de
# fondo que el umbral global, pero evadiéndolo por acumulación entre
# páginas. Una página con 20 caracteres sueltos es una marca de agua, no
# contenido -- se cuenta como página sin texto, dispara la rama de
# 'parcial' (o 'error' si es la única página), y NUNCA se declara 'ok' con
# eso como extracto.
MINIMO_CARACTERES_PAGINA = 24


def _version() -> str:
    import pdfplumber

    return pdfplumber.__version__


def _texto_crudo(origen: Path) -> str:
    import pdfplumber

    partes: list[str] = []
    with pdfplumber.open(origen) as doc:
        for pagina in doc.pages:
            partes.append(pagina.extract_text() or "")
    return "\n".join(partes)


def tiene_capa_de_texto(origen: Path) -> bool:
    try:
        return len(_texto_crudo(origen).strip()) >= MINIMO_CARACTERES
    except Exception:
        return False


def _tabla_a_markdown(tabla: list[list]) -> str:
    filas = [["" if c is None else str(c).replace("|", "\\|") for c in f] for f in tabla]
    if not filas:
        return ""
    ancho = max(len(f) for f in filas)
    filas = [f + [""] * (ancho - len(f)) for f in filas]
    lineas = [
        "| " + " | ".join(filas[0]) + " |",
        "| " + " | ".join(["---"] * ancho) + " |",
    ]
    lineas += ["| " + " | ".join(f) + " |" for f in filas[1:]]
    return "\n".join(lineas)


def extraer(origen: Path) -> Resultado:
    import pdfplumber

    try:
        partes: list[str] = []
        tablas = 0
        paginas_sin_texto: list[int] = []
        with pdfplumber.open(origen) as doc:
            paginas = len(doc.pages)
            for numero, pagina in enumerate(doc.pages, start=1):
                texto = pagina.extract_text() or ""
                if len(texto.strip()) < MINIMO_CARACTERES_PAGINA:
                    paginas_sin_texto.append(numero)
                partes.append(f"<!-- página {numero} -->")
                if texto.strip():
                    partes.append(texto)
                for tabla in pagina.extract_tables() or []:
                    md = _tabla_a_markdown(tabla)
                    if md:
                        tablas += 1
                        partes.append(md)
    except Exception as exc:
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR, version=_version(),
            detalle={"razon": f"no se pudo leer: {type(exc).__name__}: {exc}"},
        )

    contenido = "\n\n".join(partes).strip()
    util = "".join(p for p in partes if not p.startswith("<!--")).strip()

    if len(util) < MINIMO_CARACTERES:
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR, version=_version(),
            detalle={
                "razon": "sin capa de texto util; corresponde OCR",
                "paginas": paginas,
            },
        )

    if paginas_sin_texto:
        return Resultado(
            estado="parcial", salidas={"texto.md": contenido},
            extractor=EXTRACTOR, version=_version(),
            detalle={
                "razon": "algunas paginas no tienen capa de texto (probable "
                "imagen); no se resuelven aca, corresponde OCR",
                "paginas": paginas,
                "tablas": tablas,
                "paginas_sin_texto": paginas_sin_texto,
            },
        )

    return Resultado(
        estado="ok", salidas={"texto.md": contenido},
        extractor=EXTRACTOR, version=_version(),
        detalle={"paginas": paginas, "tablas": tablas},
    )
