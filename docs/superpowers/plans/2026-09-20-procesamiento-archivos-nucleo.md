# Procesamiento de archivos — Núcleo (Plan 1 de 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convertir archivos a extractos cacheados de forma determinista, una sola vez, y
medir si eso ahorra al menos 30 % de tokens de entrada.

**Architecture:** Paquete `procesamiento/` con una compuerta que elige extractor por tipo,
cada extractor devuelve un `Resultado` uniforme, y una capa de caché indexada por
`sha256 + extractor + versión`. No toca Jacobs, no toca la GPU, no cuesta tokens.

**Tech Stack:** Python 3.14 · openpyxl · pdfplumber · python-docx · tesseract (subproceso) ·
LibreOffice headless (subproceso, sólo para recalcular) · pytest

**Spec:** `docs/superpowers/specs/2026-09-20-procesamiento-archivos-design.md`

## Global Constraints

- **Todo archivo de test nuevo se engancha a un job de `.github/workflows/policy.yml` en el
  MISMO commit**, o el detector `policy/tests/test_archivos_de_test_wireados_en_ci.py` falla.
  Los tests de este plan son puros (sin DB) → job `tests-puros`.
- **Ningún archivo de cliente entra al repositorio.** Los tests construyen sus propios
  archivos de prueba en `tmp_path`.
- **Fallo cerrado siempre.** Un extractor que no pudo NO escribe extracto. Estados válidos:
  `ok` · `parcial` · `error` · `sin_extractor`. Nunca un extracto vacío dado por bueno.
- **`pandas` NO se instala.** Peso sin beneficio para extraer.
- Dependencias con versión FIJADA en `requirements-archivos.txt` (igual que `requirements.txt`).
- Comandos: `python -m pytest <ruta> -v` desde la raíz del worktree.
- Los extractos viven bajo `JAX_WORKSPACE_DIR`, nunca fuera: reusar el jail existente.

---

## File Structure

| Archivo | Responsabilidad |
|---|---|
| `procesamiento/__init__.py` | paquete |
| `procesamiento/ficha.py` | la ficha del extracto (`Ficha`), serialización, `sha256_de` |
| `procesamiento/resultado.py` | `Resultado` que devuelve todo extractor |
| `procesamiento/extractores/excel.py` | `.xlsx`/`.xls` → un CSV por hoja |
| `procesamiento/extractores/pdf.py` | detección de capa de texto + PDF nativo → Markdown |
| `procesamiento/extractores/ocr.py` | PDF escaneado e imagen → texto |
| `procesamiento/extractores/word.py` | `.docx` → Markdown |
| `procesamiento/compuerta.py` | elige extractor por tipo; `sin_extractor` si no hay |
| `procesamiento/cache.py` | decide extraer o reusar; regenera si cambió la huella |
| `procesamiento/ingesta.py` | copia a `fuente/`, dispara extracción, escribe `procesado/` |
| `scripts/medir_ahorro_extractos.py` | línea base vs con caché (criterio B del spec) |

---

### Task 1: La ficha y la huella

**Files:**
- Create: `procesamiento/__init__.py`, `procesamiento/ficha.py`
- Test: `procesamiento/_ficha_test.py`
- Modify: `.github/workflows/policy.yml` (job `tests-puros`)

**Interfaces:**
- Produces: `sha256_de(path: Path) -> str`; `Ficha` (dataclass) con campos
  `sha256, origen, extractor, extractor_version, fecha, estado, detalle: dict`,
  métodos `a_json() -> str` y `Ficha.desde_json(s: str) -> Ficha`.

- [ ] **Step 1: Write the failing test**

```python
# procesamiento/_ficha_test.py
from pathlib import Path
from procesamiento.ficha import Ficha, sha256_de


def test_sha256_de_un_archivo_conocido(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"hola")
    # sha256("hola")
    assert sha256_de(f) == (
        "b221d9dbb083a7f33428d7c2a3c3198ae925614d70210e28716ccaa7cd4ddb79"
    )


def test_la_ficha_ida_y_vuelta_no_pierde_nada(tmp_path: Path):
    f = Ficha(
        sha256="abc",
        origen="fuente/x.xlsx",
        extractor="openpyxl",
        extractor_version="3.1.5",
        fecha="2026-09-20T23:14:00-06:00",
        estado="ok",
        detalle={"hojas": 6, "hojas_extraidas": 6},
    )
    assert Ficha.desde_json(f.a_json()) == f


def test_un_estado_inventado_se_rechaza():
    import pytest

    with pytest.raises(ValueError, match="estado inválido"):
        Ficha(
            sha256="a", origen="b", extractor="c", extractor_version="d",
            fecha="e", estado="casi_bien", detalle={},
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest procesamiento/_ficha_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'procesamiento'`

- [ ] **Step 3: Write minimal implementation**

```python
# procesamiento/ficha.py
"""La ficha de un extracto: quién lo hizo, con qué versión y si salió bien.

`extractor` y `extractor_version` no son adorno: son lo que permite reprocesar
todo el día que cambie una herramienta, sin releer los originales a ciegas.
Es lo que faltó en la migración a bge-m3 (2026-09-12).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

ESTADOS = frozenset({"ok", "parcial", "error", "sin_extractor"})


def sha256_de(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for bloque in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


@dataclass(frozen=True)
class Ficha:
    sha256: str
    origen: str
    extractor: str
    extractor_version: str
    fecha: str
    estado: str
    detalle: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.estado not in ESTADOS:
            raise ValueError(
                f"estado inválido: {self.estado!r}. Válidos: {sorted(ESTADOS)}"
            )

    def a_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, indent=2)

    @classmethod
    def desde_json(cls, s: str) -> "Ficha":
        return cls(**json.loads(s))
```

Y `procesamiento/__init__.py` vacío.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest procesamiento/_ficha_test.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Enganchar el test a CI**

En `.github/workflows/policy.yml`, dentro del job `tests-puros`, agregar una línea
`run:` igual a las que ya están:

```yaml
      - run: python -m pytest procesamiento/_ficha_test.py -v
```

- [ ] **Step 6: Verificar que el detector de cobertura queda verde**

Run: `python -m pytest policy/tests/test_archivos_de_test_wireados_en_ci.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add procesamiento/__init__.py procesamiento/ficha.py procesamiento/_ficha_test.py .github/workflows/policy.yml
git commit -m "feat(procesamiento): ficha del extracto con procedencia y huella sha256"
```

---

### Task 2: El resultado uniforme de un extractor

**Files:**
- Create: `procesamiento/resultado.py`
- Test: `procesamiento/_resultado_test.py`
- Modify: `.github/workflows/policy.yml`

**Interfaces:**
- Consumes: `procesamiento.ficha.ESTADOS`
- Produces: `Resultado` (dataclass) con `estado: str`, `salidas: dict[str, str]`
  (nombre de archivo → contenido), `detalle: dict`, `extractor: str`, `version: str`.
  Propiedad `hubo_extracto: bool`.

- [ ] **Step 1: Write the failing test**

```python
# procesamiento/_resultado_test.py
import pytest
from procesamiento.resultado import Resultado


def test_un_error_no_puede_traer_salidas():
    """Fallo cerrado: si falló, NO hay extracto. Un extracto vacío dado por
    bueno hace que el modelo invente lo que no pudo leer (Principio VIII)."""
    with pytest.raises(ValueError, match="no puede traer salidas"):
        Resultado(
            estado="error", salidas={"a.csv": "x"},
            detalle={"razon": "roto"}, extractor="t", version="1",
        )


def test_ok_sin_salidas_tampoco_se_permite():
    with pytest.raises(ValueError, match="sin salidas"):
        Resultado(estado="ok", salidas={}, detalle={}, extractor="t", version="1")


def test_parcial_lleva_salidas_y_avisa():
    r = Resultado(
        estado="parcial", salidas={"h1.csv": "a,b"},
        detalle={"hojas": 6, "hojas_extraidas": 1}, extractor="t", version="1",
    )
    assert r.hubo_extracto is True
    assert r.detalle["hojas_extraidas"] < r.detalle["hojas"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest procesamiento/_resultado_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'procesamiento.resultado'`

- [ ] **Step 3: Write minimal implementation**

> ⚠️ **ESTE BLOQUE QUEDÓ SUPERADO — no lo copies.** La auditoría adversarial del
> 2026-09-20 demostró que NO cumple el spec: `@dataclass(frozen=True)` congela el nombre
> del atributo, no el diccionario, así que `r.salidas["x"] = ...` después de construir
> rompía el fallo cerrado. Además 5 de 8 mutaciones al código dejaban los tests en verde.
> La implementación real y verificada está en `procesamiento/resultado.py` (commit
> `a7ef8c9`): copia defensiva + `MappingProxyType`, validación de mapa, campos
> obligatorios no vacíos, "al menos una salida con contenido", y `detalle` validado
> contra un ida y vuelta de JSON. Se deja este bloque como registro de lo que falló.

```python
# procesamiento/resultado.py
"""Lo que devuelve todo extractor. Las reglas de fallo cerrado viven acá,
en el tipo, para que ningún extractor pueda saltárselas por descuido."""
from __future__ import annotations

from dataclasses import dataclass, field

from procesamiento.ficha import ESTADOS


@dataclass(frozen=True)
class Resultado:
    estado: str
    salidas: dict[str, str]
    detalle: dict
    extractor: str
    version: str

    def __post_init__(self) -> None:
        if self.estado not in ESTADOS:
            raise ValueError(f"estado inválido: {self.estado!r}")
        if self.estado in {"error", "sin_extractor"} and self.salidas:
            raise ValueError(
                f"un resultado '{self.estado}' no puede traer salidas: "
                "un extracto entregado tras un fallo hace que el modelo invente"
            )
        if self.estado in {"ok", "parcial"} and not self.salidas:
            raise ValueError(f"un resultado '{self.estado}' sin salidas no es válido")

    @property
    def hubo_extracto(self) -> bool:
        return bool(self.salidas)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest procesamiento/_resultado_test.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Enganchar a CI**

```yaml
      - run: python -m pytest procesamiento/_resultado_test.py -v
```

- [ ] **Step 6: Commit**

```bash
git add procesamiento/resultado.py procesamiento/_resultado_test.py .github/workflows/policy.yml
git commit -m "feat(procesamiento): Resultado con fallo cerrado en el tipo"
```

---

### Task 3: Extractor de Excel — TODAS las hojas

**Files:**
- Create: `procesamiento/extractores/__init__.py`, `procesamiento/extractores/excel.py`,
  `requirements-archivos.txt`
- Test: `procesamiento/extractores/_excel_test.py`
- Modify: `.github/workflows/policy.yml`

**Interfaces:**
- Consumes: `procesamiento.resultado.Resultado`
- Produces: `extraer(origen: Path) -> Resultado` en `procesamiento.extractores.excel`.
  Una salida por hoja, nombre `<indice>-<titulo-slug>.csv`.

**Contexto medido (2026-09-20):** sobre `EEFF Cierre 2022 A 06-2025 NETEADOS.xlsx`
(6 hojas, real), `soffice --headless --convert-to csv` produjo **UN solo CSV**, sin error
y sin aviso. Este task existe para que eso no pueda pasar.

- [ ] **Step 1: Write the failing test**

```python
# procesamiento/extractores/_excel_test.py
"""El defecto que este extractor existe para evitar, reproducido con un archivo
sintético: LibreOffice convierte UNA hoja de seis, calladito."""
from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

from procesamiento.extractores import excel


def _libro_de_seis_hojas(destino: Path) -> Path:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for i, nombre in enumerate(
        ["2025 JUNIO", "2024", "2023", "2022", "consolidado Bac", "CONSOLIDADO"]
    ):
        ws = wb.create_sheet(title=nombre)
        ws["A1"] = "ACTIVOS"
        ws["B1"] = 100 * (i + 1)
        ws["A2"] = "PASIVOS"
        ws["B2"] = 40 * (i + 1)
    wb.save(destino)
    return destino


def test_un_libro_de_seis_hojas_produce_SEIS_extractos(tmp_path: Path):
    origen = _libro_de_seis_hojas(tmp_path / "eeff.xlsx")
    r = excel.extraer(origen)
    assert r.estado == "ok"
    assert len(r.salidas) == 6, (
        f"se perdieron hojas: {sorted(r.salidas)}. "
        "Este es exactamente el defecto de LibreOffice medido el 2026-09-20."
    )
    assert r.detalle["hojas"] == 6
    assert r.detalle["hojas_extraidas"] == 6


def test_el_contenido_conserva_filas_y_columnas(tmp_path: Path):
    origen = _libro_de_seis_hojas(tmp_path / "eeff.xlsx")
    r = excel.extraer(origen)
    primera = r.salidas[sorted(r.salidas)[0]]
    assert "ACTIVOS,100" in primera.replace("\r\n", "\n")
    assert "PASIVOS,40" in primera.replace("\r\n", "\n")


def test_un_archivo_que_no_es_excel_da_error_sin_extracto(tmp_path: Path):
    malo = tmp_path / "no-es.xlsx"
    malo.write_bytes(b"esto no es un zip")
    r = excel.extraer(malo)
    assert r.estado == "error"
    assert r.salidas == {}
    assert "razon" in r.detalle
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest procesamiento/extractores/_excel_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'procesamiento.extractores'`

- [ ] **Step 3: Instalar la dependencia fijada**

```bash
printf 'openpyxl==3.1.5\npdfplumber==0.11.10\npython-docx==1.2.0\n' > requirements-archivos.txt
python -m pip install --user -r requirements-archivos.txt
```

**`--user` NO es opcional y no es cosmético.** El spec exige que estas dependencias no
puedan tocar los servicios de producción. Medido el 2026-09-20: `jax-platform` y
`jax-las-manos` corren como **`jaxsvc`**, así que una instalación en
`/home/fruiz/.local/lib/python3.14/site-packages` les es invisible. Una instalación al
sistema (`sudo pip`) SÍ los alcanzaría.

En CI da igual (el runner es efímero y se descarta), por eso el paso de CI usa
`pip install -r requirements-archivos.txt` sin `--user`.

Las tres versiones están verificadas contra el índice de PyPI el 2026-09-20. Si alguna
fallara, fijar la última estable publicada y anotarlo; NO usar rangos.

- [ ] **Step 4: Write minimal implementation**

```python
# procesamiento/extractores/excel.py
"""Excel → un CSV por hoja.

Una hoja de cálculo NO es texto: aplanarla a prosa destruye justo lo que la
hacía un balance (qué número en qué fila y en qué columna). Por eso el extracto
es CSV por hoja, no un .txt.

openpyxl y no LibreOffice porque LibreOffice convierte una sola hoja
(medido 2026-09-20 sobre un archivo real de 6 hojas: sacó 1, sin error).
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

from procesamiento.resultado import Resultado

EXTRACTOR = "openpyxl"


def _version() -> str:
    import openpyxl

    return openpyxl.__version__


def _slug(texto: str) -> str:
    s = re.sub(r"[^\w\-]+", "-", texto.strip(), flags=re.UNICODE).strip("-").lower()
    return s or "hoja"


def _hoja_a_csv(hoja) -> str:
    buffer = io.StringIO()
    escritor = csv.writer(buffer, lineterminator="\n")
    for fila in hoja.iter_rows(values_only=True):
        escritor.writerow(["" if c is None else c for c in fila])
    return buffer.getvalue()


def extraer(origen: Path) -> Resultado:
    import openpyxl

    try:
        libro = openpyxl.load_workbook(origen, data_only=True, read_only=True)
    except Exception as exc:  # archivo corrupto, no-zip, protegido
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR, version=_version(),
            detalle={"razon": f"no se pudo abrir: {type(exc).__name__}: {exc}"},
        )

    total = len(libro.worksheets)
    salidas: dict[str, str] = {}
    fallidas: list[str] = []
    for indice, hoja in enumerate(libro.worksheets, start=1):
        try:
            salidas[f"{indice:02d}-{_slug(hoja.title)}.csv"] = _hoja_a_csv(hoja)
        except Exception as exc:
            fallidas.append(f"{hoja.title}: {type(exc).__name__}: {exc}")
    libro.close()

    if not salidas:
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR, version=_version(),
            detalle={"razon": "ninguna hoja pudo extraerse", "fallidas": fallidas},
        )

    estado = "ok" if len(salidas) == total else "parcial"
    return Resultado(
        estado=estado, salidas=salidas, extractor=EXTRACTOR, version=_version(),
        detalle={"hojas": total, "hojas_extraidas": len(salidas), "fallidas": fallidas},
    )
```

Y `procesamiento/extractores/__init__.py` vacío.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest procesamiento/extractores/_excel_test.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Enganchar a CI, con la instalación de dependencias**

En el job `tests-puros` de `policy.yml`, antes de los `run: python -m pytest`, asegurar:

```yaml
      - run: pip install -r requirements-archivos.txt
      - run: python -m pytest procesamiento/extractores/_excel_test.py -v
```

- [ ] **Step 7: Commit**

```bash
git add procesamiento/extractores/ requirements-archivos.txt .github/workflows/policy.yml
git commit -m "feat(procesamiento): Excel a un CSV por hoja -- las SEIS, no una"
```

---

### Task 4: PDF nativo — detección de capa de texto y tablas

**Files:**
- Create: `procesamiento/extractores/pdf.py`
- Test: `procesamiento/extractores/_pdf_test.py`
- Modify: `.github/workflows/policy.yml`

**Interfaces:**
- Consumes: `procesamiento.resultado.Resultado`
- Produces: `tiene_capa_de_texto(origen: Path) -> bool` y `extraer(origen: Path) -> Resultado`
  en `procesamiento.extractores.pdf`. Salida única `texto.md`.

**Por qué la detección es lo primero:** un PDF puede ser nativo (tiene texto), escaneado
(es una imagen adentro) o híbrido. Distinguirlos es determinista y barato: se pide el texto
y si vuelve casi vacío, es escaneado. **Nadie gasta OCR sin que el camino barato diga que no.**

- [ ] **Step 1: Write the failing test**

```python
# procesamiento/extractores/_pdf_test.py
from pathlib import Path

import pytest

pdfplumber = pytest.importorskip("pdfplumber")

from procesamiento.extractores import pdf


def _pdf_con_texto(destino: Path) -> Path:
    """PDF mínimo con capa de texto, escrito a mano (sin dependencias)."""
    contenido = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]"
        b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n"
        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"5 0 obj<</Length 52>>stream\n"
        b"BT /F1 12 Tf 20 100 Td (ACTIVOS TOTALES 1234) Tj ET\n"
        b"endstream endobj\n"
        b"trailer<</Root 1 0 R>>\n"
    )
    destino.write_bytes(contenido)
    return destino


def test_detecta_que_un_pdf_nativo_tiene_texto(tmp_path: Path):
    assert pdf.tiene_capa_de_texto(_pdf_con_texto(tmp_path / "n.pdf")) is True


def test_extrae_el_texto_del_pdf_nativo(tmp_path: Path):
    r = pdf.extraer(_pdf_con_texto(tmp_path / "n.pdf"))
    assert r.estado == "ok"
    assert "ACTIVOS TOTALES 1234" in r.salidas["texto.md"]


def test_un_pdf_roto_da_error_sin_extracto(tmp_path: Path):
    malo = tmp_path / "roto.pdf"
    malo.write_bytes(b"no soy un pdf")
    r = pdf.extraer(malo)
    assert r.estado == "error"
    assert r.salidas == {}


def test_un_pdf_sin_capa_de_texto_no_se_declara_ok(tmp_path: Path):
    """Un PDF de imagen pura NO se resuelve acá: se manda a OCR. Lo que NO
    puede pasar es que devuelva 'ok' con un extracto vacío."""
    vacio = tmp_path / "escaneado.pdf"
    vacio.write_bytes(
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
        b"trailer<</Root 1 0 R>>\n"
    )
    assert pdf.tiene_capa_de_texto(vacio) is False
    r = pdf.extraer(vacio)
    assert r.estado != "ok"
    assert r.salidas == {}
```

> **Ruling del controlador (barrido previo):** el PDF escrito a mano en el test no
> tiene tabla `xref` y pdfminer puede rechazarlo. Si `_pdf_con_texto` no parsea,
> **generar el fixture con LibreOffice** (instalado, 26.2.5.2) en vez de a mano:
> escribir un `.txt` y correr
> `soffice --headless --convert-to pdf --outdir <tmp> <txt>`.
> Lo que NO se permite es relajar la asercion para que pase: el test debe seguir
> exigiendo que el texto salga.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest procesamiento/extractores/_pdf_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'procesamiento.extractores.pdf'`

- [ ] **Step 3: Write minimal implementation**

```python
# procesamiento/extractores/pdf.py
"""PDF nativo → Markdown, con las tablas conservadas.

pdftotext NO sirve acá: destruye las tablas, y en un expediente financiero la
tabla ES el documento. pdfplumber conserva filas y columnas.

La detección de capa de texto es la compuerta entre este extractor y el de OCR:
determinista, barata, sin modelos.
"""
from __future__ import annotations

from pathlib import Path

from procesamiento.resultado import Resultado

EXTRACTOR = "pdfplumber"
# Menos de esto en TODO el documento = no hay capa de texto util.
MINIMO_CARACTERES = 32


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
    lineas = ["| " + " | ".join(filas[0]) + " |",
              "| " + " | ".join(["---"] * ancho) + " |"]
    lineas += ["| " + " | ".join(f) + " |" for f in filas[1:]]
    return "\n".join(lineas)


def extraer(origen: Path) -> Resultado:
    import pdfplumber

    try:
        partes: list[str] = []
        tablas = 0
        with pdfplumber.open(origen) as doc:
            paginas = len(doc.pages)
            for numero, pagina in enumerate(doc.pages, start=1):
                partes.append(f"<!-- página {numero} -->")
                texto = pagina.extract_text() or ""
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
            detalle={"razon": "sin capa de texto util; corresponde OCR",
                     "paginas": paginas},
        )

    return Resultado(
        estado="ok", salidas={"texto.md": contenido},
        extractor=EXTRACTOR, version=_version(),
        detalle={"paginas": paginas, "tablas": tablas},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest procesamiento/extractores/_pdf_test.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Enganchar a CI**

```yaml
      - run: python -m pytest procesamiento/extractores/_pdf_test.py -v
```

- [ ] **Step 6: Commit**

```bash
git add procesamiento/extractores/pdf.py procesamiento/extractores/_pdf_test.py .github/workflows/policy.yml
git commit -m "feat(procesamiento): PDF nativo con tablas, y deteccion de capa de texto"
```

---

### Task 5: OCR — PDF escaneado e imagen

**Files:**
- Create: `procesamiento/extractores/ocr.py`
- Test: `procesamiento/extractores/_ocr_test.py`
- Modify: `.github/workflows/policy.yml`

**Interfaces:**
- Consumes: `procesamiento.resultado.Resultado`
- Produces: `extraer(origen: Path, idioma: str = "spa") -> Resultado` en
  `procesamiento.extractores.ocr`. Salida `texto.txt`.

**Verificado 2026-09-20:** `tesseract 5.5.0` con `spa` leyó
`"Estado de Situación Financiera — año 2026"` exacto, con tildes y guion largo.

- [ ] **Step 1: Write the failing test**

```python
# procesamiento/extractores/_ocr_test.py
import shutil
from pathlib import Path

import pytest

from procesamiento.extractores import ocr

pytestmark = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract no instalado"
)
Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")


def _imagen_con_texto(destino: Path, texto: str) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (900, 120), "white")
    d = ImageDraw.Draw(img)
    try:
        fuente = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34
        )
    except OSError:
        fuente = None
    d.text((20, 40), texto, fill="black", font=fuente)
    img.save(destino)
    return destino


def test_lee_texto_en_espanol_con_tildes(tmp_path: Path):
    origen = _imagen_con_texto(tmp_path / "a.png", "Estado de Situación Financiera")
    r = ocr.extraer(origen)
    assert r.estado == "ok"
    assert "Situación" in r.salidas["texto.txt"]


def test_una_imagen_en_blanco_no_se_declara_ok(tmp_path: Path):
    """Fallo cerrado: si el OCR no leyó nada, NO hay extracto."""
    from PIL import Image

    blanco = tmp_path / "blanco.png"
    Image.new("RGB", (400, 200), "white").save(blanco)
    r = ocr.extraer(blanco)
    assert r.estado == "error"
    assert r.salidas == {}


def test_un_archivo_inexistente_da_error(tmp_path: Path):
    r = ocr.extraer(tmp_path / "no-existe.png")
    assert r.estado == "error"
    assert r.salidas == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest procesamiento/extractores/_ocr_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'procesamiento.extractores.ocr'`

- [ ] **Step 3: Write minimal implementation**

```python
# procesamiento/extractores/ocr.py
"""OCR para lo que no tiene capa de texto: PDF escaneado e imagen.

tesseract por subproceso, nunca por binding: un binding suma dependencia
nativa y no aporta nada acá. Se pasa por `spa` porque los documentos son en
español (verificado 2026-09-20: lee tildes y guion largo exacto).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from procesamiento.resultado import Resultado

EXTRACTOR = "tesseract"
MINIMO_CARACTERES = 8
TIMEOUT_SEGUNDOS = 300


def _version() -> str:
    try:
        salida = subprocess.run(
            ["tesseract", "--version"], capture_output=True, text=True, timeout=30
        )
        return salida.stdout.splitlines()[0].split()[-1]
    except Exception:
        return "desconocida"


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

    return Resultado(
        estado="ok", salidas={"texto.txt": texto},
        extractor=EXTRACTOR, version=_version(),
        detalle={"idioma": idioma, "caracteres": len(texto)},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest procesamiento/extractores/_ocr_test.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Enganchar a CI**

```yaml
      - run: sudo apt-get install -y tesseract-ocr tesseract-ocr-spa
      - run: python -m pytest procesamiento/extractores/_ocr_test.py -v
```

- [ ] **Step 6: Commit**

```bash
git add procesamiento/extractores/ocr.py procesamiento/extractores/_ocr_test.py .github/workflows/policy.yml
git commit -m "feat(procesamiento): OCR en espanol, con fallo cerrado si no leyo nada"
```

---

### Task 6: Word

**Files:**
- Create: `procesamiento/extractores/word.py`
- Test: `procesamiento/extractores/_word_test.py`
- Modify: `.github/workflows/policy.yml`

**Interfaces:**
- Produces: `extraer(origen: Path) -> Resultado` en `procesamiento.extractores.word`.
  Salida `texto.md`.

- [ ] **Step 1: Write the failing test**

```python
# procesamiento/extractores/_word_test.py
from pathlib import Path

import pytest

docx = pytest.importorskip("docx")

from procesamiento.extractores import word


def _documento(destino: Path) -> Path:
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


def test_conserva_titulo_parrafo_y_tabla(tmp_path: Path):
    r = word.extraer(_documento(tmp_path / "d.docx"))
    assert r.estado == "ok"
    md = r.salidas["texto.md"]
    assert "# Estado de Resultados" in md
    assert "Periodo 2026" in md
    assert "| INGRESOS | 1000 |" in md


def test_un_docx_roto_da_error_sin_extracto(tmp_path: Path):
    malo = tmp_path / "roto.docx"
    malo.write_bytes(b"no soy un docx")
    r = word.extraer(malo)
    assert r.estado == "error"
    assert r.salidas == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest procesamiento/extractores/_word_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'procesamiento.extractores.word'`

- [ ] **Step 3: Write minimal implementation**

```python
# procesamiento/extractores/word.py
"""Word → Markdown. Un .docx es un ZIP con XML: el texto, los titulos y las
tablas ya estan ahi, estructurados. No hace falta OCR ni modelo.

Los .doc viejos NO los lee python-docx: esos van por LibreOffice (fuera del
alcance de la fase 1; hoy caen como `sin_extractor`, que es honesto).
"""
from __future__ import annotations

from pathlib import Path

from procesamiento.resultado import Resultado

EXTRACTOR = "python-docx"


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("python-docx")
    except Exception:
        return "desconocida"


def extraer(origen: Path) -> Resultado:
    from docx import Document

    try:
        documento = Document(str(origen))
    except Exception as exc:
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR, version=_version(),
            detalle={"razon": f"no se pudo abrir: {type(exc).__name__}: {exc}"},
        )

    lineas: list[str] = []
    for parrafo in documento.paragraphs:
        texto = parrafo.text.strip()
        if not texto:
            continue
        estilo = (parrafo.style.name or "").lower()
        if estilo.startswith("heading"):
            nivel = "".join(c for c in estilo if c.isdigit()) or "1"
            lineas.append("#" * min(int(nivel), 6) + " " + texto)
        else:
            lineas.append(texto)

    for tabla in documento.tables:
        filas = [[celda.text.strip().replace("|", "\\|") for celda in fila.cells]
                 for fila in tabla.rows]
        if not filas:
            continue
        lineas.append("")
        lineas.append("| " + " | ".join(filas[0]) + " |")
        lineas.append("| " + " | ".join(["---"] * len(filas[0])) + " |")
        for fila in filas[1:]:
            lineas.append("| " + " | ".join(fila) + " |")

    contenido = "\n".join(lineas).strip()
    if not contenido:
        return Resultado(
            estado="error", salidas={}, extractor=EXTRACTOR, version=_version(),
            detalle={"razon": "el documento no tiene texto"},
        )

    return Resultado(
        estado="ok", salidas={"texto.md": contenido},
        extractor=EXTRACTOR, version=_version(),
        detalle={"parrafos": len(documento.paragraphs), "tablas": len(documento.tables)},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest procesamiento/extractores/_word_test.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Enganchar a CI y commit**

```yaml
      - run: python -m pytest procesamiento/extractores/_word_test.py -v
```

```bash
git add procesamiento/extractores/word.py procesamiento/extractores/_word_test.py .github/workflows/policy.yml
git commit -m "feat(procesamiento): Word a Markdown conservando titulos y tablas"
```

---

### Task 7: La compuerta — elegir extractor sin gastar de más

**Files:**
- Create: `procesamiento/compuerta.py`
- Test: `procesamiento/_compuerta_test.py`
- Modify: `.github/workflows/policy.yml`

**Interfaces:**
- Consumes: `procesamiento.extractores.{excel,pdf,ocr,word}.extraer`
- Produces: `extraer(origen: Path) -> Resultado` en `procesamiento.compuerta`.
  Un tipo sin extractor devuelve `estado="sin_extractor"`, nunca un extracto vacío.

**La regla:** cada capa corre SOLO si la anterior no alcanzó. Un PDF va primero a
`pdf.extraer`; sólo si no tiene capa de texto se paga el OCR.

- [ ] **Step 1: Write the failing test**

```python
# procesamiento/_compuerta_test.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest procesamiento/_compuerta_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'procesamiento.compuerta'`

- [ ] **Step 3: Write minimal implementation**

```python
# procesamiento/compuerta.py
"""Elige el extractor por tipo, y NUNCA paga el caro si el barato alcanzo.

Un tipo sin extractor devuelve `sin_extractor`: el archivo se guarda igual, pero
nadie finge haberlo leido. Es la diferencia entre un sistema honesto y uno que
hace inventar al modelo (Principio VIII).
"""
from __future__ import annotations

from pathlib import Path

from procesamiento.extractores import excel, ocr, pdf, word
from procesamiento.resultado import Resultado

IMAGENES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
EXCEL = {".xlsx", ".xlsm"}
WORD = {".docx"}


def extraer(origen: Path) -> Resultado:
    sufijo = Path(origen).suffix.lower()

    if sufijo in EXCEL:
        return excel.extraer(origen)
    if sufijo in WORD:
        return word.extraer(origen)
    if sufijo in IMAGENES:
        return ocr.extraer(origen)
    if sufijo == ".pdf":
        # Compuerta: el camino barato primero. OCR solo si no hay capa de texto.
        if pdf.tiene_capa_de_texto(origen):
            return pdf.extraer(origen)
        return ocr.extraer(origen)

    return Resultado(
        estado="sin_extractor", salidas={}, extractor="ninguno", version="-",
        detalle={"razon": f"no hay extractor para '{sufijo or 'sin extension'}'"},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest procesamiento/_compuerta_test.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Enganchar a CI y commit**

```yaml
      - run: python -m pytest procesamiento/_compuerta_test.py -v
```

```bash
git add procesamiento/compuerta.py procesamiento/_compuerta_test.py .github/workflows/policy.yml
git commit -m "feat(procesamiento): compuerta -- el caro solo si el barato no alcanzo"
```

---

### Task 8: Ingesta y caché — el ahorro de verdad

**Files:**
- Create: `procesamiento/ingesta.py`
- Test: `procesamiento/_ingesta_test.py`
- Modify: `.github/workflows/policy.yml`

**Interfaces:**
- Consumes: `procesamiento.compuerta.extraer`, `procesamiento.ficha.{Ficha, sha256_de}`
- Produces: en `procesamiento.ingesta`:
  - `ingerir(origen: Path, trabajo: Path) -> Ficha` — copia a `fuente/`, extrae si hace
    falta, escribe `procesado/<sha256>/`. Devuelve la ficha.
  - `ruta_procesado(trabajo: Path, huella: str) -> Path`

**Esta es la tarea que produce el ahorro.** Todo lo anterior es maquinaria.

- [ ] **Step 1: Write the failing test**

```python
# procesamiento/_ingesta_test.py
from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

from procesamiento import ingesta
from procesamiento.ficha import sha256_de


def _libro(destino: Path, valor: int = 100) -> Path:
    wb = openpyxl.Workbook()
    wb.active["A1"] = "ACTIVOS"
    wb.active["B1"] = valor
    wb.save(destino)
    return destino


def test_ingerir_copia_el_original_y_escribe_ficha(tmp_path: Path):
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")
    f = ingesta.ingerir(origen, trabajo)

    assert (trabajo / "fuente" / "e.xlsx").is_file()
    assert f.sha256 == sha256_de(origen)
    assert f.estado == "ok"
    assert (ingesta.ruta_procesado(trabajo, f.sha256) / "ficha.json").is_file()


def test_el_segundo_pase_NO_vuelve_a_extraer(tmp_path: Path, monkeypatch):
    """El corazon del ahorro: mismo archivo, cero trabajo la segunda vez."""
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")
    ingesta.ingerir(origen, trabajo)

    from procesamiento import compuerta

    def no_debe_llamarse(_):
        raise AssertionError("se volvio a extraer un archivo ya cacheado")

    monkeypatch.setattr(compuerta, "extraer", no_debe_llamarse)
    f = ingesta.ingerir(origen, trabajo)
    assert f.estado == "ok"


def test_si_el_original_cambia_se_REGENERA(tmp_path: Path):
    trabajo = tmp_path / "trabajo"
    origen = tmp_path / "e.xlsx"
    _libro(origen, valor=100)
    primera = ingesta.ingerir(origen, trabajo)

    _libro(origen, valor=999)
    segunda = ingesta.ingerir(origen, trabajo)

    assert segunda.sha256 != primera.sha256
    csv = next(
        (ingesta.ruta_procesado(trabajo, segunda.sha256)).glob("*.csv")
    ).read_text()
    assert "999" in csv


def test_borrar_procesado_entero_lo_reconstruye(tmp_path: Path):
    """Invariante central: `procesado/` es desechable."""
    import shutil

    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")
    primera = ingesta.ingerir(origen, trabajo)
    antes = (ingesta.ruta_procesado(trabajo, primera.sha256) / "ficha.json").read_text()

    shutil.rmtree(trabajo / "procesado")
    segunda = ingesta.ingerir(origen, trabajo)
    despues = (
        ingesta.ruta_procesado(trabajo, segunda.sha256) / "ficha.json"
    ).read_text()

    import json

    assert json.loads(antes)["sha256"] == json.loads(despues)["sha256"]


def test_un_tipo_sin_extractor_guarda_el_original_y_lo_dice(tmp_path: Path):
    trabajo = tmp_path / "trabajo"
    raro = tmp_path / "algo.xyz"
    raro.write_bytes(b"contenido")
    f = ingesta.ingerir(raro, trabajo)

    assert f.estado == "sin_extractor"
    assert (trabajo / "fuente" / "algo.xyz").is_file()
    carpeta = ingesta.ruta_procesado(trabajo, f.sha256)
    assert list(carpeta.glob("*.csv")) == []
    assert list(carpeta.glob("*.md")) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest procesamiento/_ingesta_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'procesamiento.ingesta'`

- [ ] **Step 3: Write minimal implementation**

```python
# procesamiento/ingesta.py
"""Ingesta: el original entra a `fuente/` y su extracto queda en `procesado/`.

Dos invariantes:
  1. `fuente/` es inmutable. El original entra y no se toca mas.
  2. `procesado/` es DESECHABLE: se borra entero y se reconstruye. Nada que no
     sea reconstruible vive ahi. Un cache del que no te podes fiar para borrarlo
     no es un cache: es una segunda base de datos.

El cache se indexa por `sha256` del original. Si el original cambia, la huella
cambia y el extracto se regenera -- nunca se sirve uno viejo en silencio.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from procesamiento import compuerta
from procesamiento.ficha import Ficha, sha256_de


def ruta_procesado(trabajo: Path, huella: str) -> Path:
    return Path(trabajo) / "procesado" / huella


def _ahora() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ingerir(origen: Path, trabajo: Path) -> Ficha:
    origen = Path(origen)
    trabajo = Path(trabajo)
    fuente = trabajo / "fuente"
    fuente.mkdir(parents=True, exist_ok=True)

    huella = sha256_de(origen)
    destino = fuente / origen.name
    if not destino.exists() or sha256_de(destino) != huella:
        shutil.copy2(origen, destino)

    carpeta = ruta_procesado(trabajo, huella)
    ficha_json = carpeta / "ficha.json"
    if ficha_json.is_file():
        # Cache vivo: misma huella, mismo extracto. Cero trabajo.
        return Ficha.desde_json(ficha_json.read_text(encoding="utf8"))

    resultado = compuerta.extraer(destino)

    # Escritura atomica por carpeta: primero temporal, despues rename. Si algo
    # se cae a la mitad, no queda un `procesado/` a medias que parezca completo.
    temporal = carpeta.with_name(carpeta.name + ".parcial")
    if temporal.exists():
        shutil.rmtree(temporal)
    temporal.mkdir(parents=True)

    for nombre, contenido in resultado.salidas.items():
        (temporal / nombre).write_text(contenido, encoding="utf8")

    ficha = Ficha(
        sha256=huella,
        origen=str(destino.relative_to(trabajo)),
        extractor=resultado.extractor,
        extractor_version=resultado.version,
        fecha=_ahora(),
        estado=resultado.estado,
        detalle=resultado.detalle,
    )
    (temporal / "ficha.json").write_text(ficha.a_json(), encoding="utf8")

    if carpeta.exists():
        shutil.rmtree(carpeta)
    temporal.rename(carpeta)
    return ficha
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest procesamiento/_ingesta_test.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Enganchar a CI y commit**

```yaml
      - run: python -m pytest procesamiento/_ingesta_test.py -v
```

```bash
git add procesamiento/ingesta.py procesamiento/_ingesta_test.py .github/workflows/policy.yml
git commit -m "feat(procesamiento): ingesta con cache por huella y procesado desechable"
```

---

### Task 9: Medir el ahorro — el criterio que puede matar el proyecto

**Files:**
- Create: `scripts/medir_ahorro_extractos.py`
- Test: `scripts/_medir_ahorro_extractos_test.py`
- Modify: `.github/workflows/policy.yml`

**Interfaces:**
- Consumes: `procesamiento.ingesta.ingerir`
- Produces: `medir(archivos: list[Path], trabajo: Path, pasos: int) -> dict` con claves
  `caracteres_sin_cache`, `caracteres_con_cache`, `ahorro_porcentual`,
  `extracciones` (cuántas veces se extrajo de verdad).

**Criterio del spec §7.B:** el segundo pipeline sobre el mismo archivo paga **$0 de
extracción** (binario), y los caracteres de entrada bajan **al menos 30 %**. Si no llega,
**el proyecto se detiene**. El número se escribió antes de medir para que no se acomode.

- [ ] **Step 1: Write the failing test**

```python
# scripts/_medir_ahorro_extractos_test.py
from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.medir_ahorro_extractos import medir


def _libro(destino: Path) -> Path:
    wb = openpyxl.Workbook()
    for fila in range(1, 200):
        wb.active[f"A{fila}"] = f"CUENTA CONTABLE NUMERO {fila}"
        wb.active[f"B{fila}"] = fila * 1000
    wb.save(destino)
    return destino


def test_la_medicion_cuenta_una_sola_extraccion_para_varios_pasos(tmp_path: Path):
    archivo = _libro(tmp_path / "e.xlsx")
    r = medir([archivo], tmp_path / "trabajo", pasos=6)
    # Seis pasos piden el mismo archivo. Sin cache serian 6 extracciones.
    assert r["extracciones"] == 1, (
        f"se extrajo {r['extracciones']} veces con 6 pasos: el cache no sirve"
    )


def test_la_medicion_reporta_ahorro_porcentual(tmp_path: Path):
    archivo = _libro(tmp_path / "e.xlsx")
    r = medir([archivo], tmp_path / "trabajo", pasos=6)
    assert r["caracteres_con_cache"] < r["caracteres_sin_cache"]
    assert 0 <= r["ahorro_porcentual"] <= 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/_medir_ahorro_extractos_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.medir_ahorro_extractos'`

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""Mide el ahorro real de caracteres de entrada del cache de extractos.

Linea base: cada paso del pipeline reenvia el archivo entero (hasta 60.000
caracteres por paso y por corrida -- ese es el gasto que se quiere evitar).
Con cache: el archivo se extrae UNA vez y cada paso reenvia solo el extracto.

Criterio del spec §7.B: si el ahorro queda bajo 30%, el proyecto se detiene.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from procesamiento import compuerta, ingesta

TOPE_POR_PASO = 60_000


def medir(archivos: list[Path], trabajo: Path, pasos: int) -> dict:
    extracciones = 0
    original_real = 0
    extracto_real = 0

    llamadas = {"n": 0}
    extraer_verdadero = compuerta.extraer

    def contando(origen):
        llamadas["n"] += 1
        return extraer_verdadero(origen)

    compuerta.extraer = contando  # type: ignore[assignment]
    try:
        for archivo in archivos:
            original_real += min(Path(archivo).stat().st_size, TOPE_POR_PASO)
            # Cada paso del pipeline PIDE el archivo. Sin cache eso serian
            # `pasos` extracciones; con cache tiene que ser exactamente 1.
            for _ in range(pasos):
                ficha = ingesta.ingerir(Path(archivo), Path(trabajo))
            carpeta = ingesta.ruta_procesado(Path(trabajo), ficha.sha256)
            extracto_real += sum(
                p.stat().st_size for p in carpeta.iterdir() if p.name != "ficha.json"
            )
    finally:
        compuerta.extraer = extraer_verdadero  # type: ignore[assignment]
        extracciones = llamadas["n"]

    sin_cache = original_real * pasos
    con_cache = extracto_real * pasos
    ahorro = 0.0 if sin_cache == 0 else (1 - con_cache / sin_cache) * 100
    return {
        "archivos": len(archivos),
        "pasos": pasos,
        "extracciones": extracciones,
        "caracteres_sin_cache": sin_cache,
        "caracteres_con_cache": con_cache,
        "ahorro_porcentual": round(max(ahorro, 0.0), 2),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archivos", nargs="+", type=Path)
    p.add_argument("--trabajo", type=Path, required=True)
    p.add_argument("--pasos", type=int, default=6)
    p.add_argument("--umbral", type=float, default=30.0)
    a = p.parse_args()

    r = medir(a.archivos, a.trabajo, a.pasos)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    if r["ahorro_porcentual"] < a.umbral:
        print(
            f"\nNO PASA: {r['ahorro_porcentual']}% < umbral {a.umbral}%. "
            "Segun el spec §7.B, el proyecto se detiene."
        )
        return 1
    print(f"\nPASA: {r['ahorro_porcentual']}% >= umbral {a.umbral}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest scripts/_medir_ahorro_extractos_test.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Correr la medición de verdad, con los archivos del ERP**

```bash
python scripts/medir_ahorro_extractos.py \
  ~/jax-workspace/erp-2026-09-18/brief-multistore.md \
  ~/jax-workspace/erp-2026-09-18/ateneaerp-esquema.sql \
  ~/jax-workspace/erp-2026-09-18/ateneaerp-estructura.md \
  --trabajo /tmp/medicion-ahorro --pasos 6
```

Anotar el número en `DEUDA.md` con fecha. **Si sale menos de 30 %, se para y se le
reporta a Fernando** — el spec lo dice y el número se fijó antes de medir.

- [ ] **Step 6: Enganchar a CI y commit**

```yaml
      - run: python -m pytest scripts/_medir_ahorro_extractos_test.py -v
```

```bash
git add scripts/medir_ahorro_extractos.py scripts/_medir_ahorro_extractos_test.py .github/workflows/policy.yml
git commit -m "feat(procesamiento): medidor de ahorro con umbral que puede detener el proyecto"
```

---

---

### Task 10: Verificación con archivos REALES y rendimiento

Cierra los criterios §7.A.2 (las 10 cifras) y §7.C (rendimiento) del spec, que ningún
task anterior cubre porque **no se pueden automatizar en CI**: dependen de archivos de
clientes, que no entran al repositorio.

**Files:**
- Create: `scripts/verificar_archivos_reales.py`
- No hay test de CI para este script: va a `EXCEPCIONES` del detector de cobertura con
  motivo escrito, porque necesita archivos que el repositorio no puede contener.

**Interfaces:**
- Consumes: `procesamiento.ingesta.ingerir`
- Produces: `verificar(directorio: Path, trabajo: Path) -> dict` con `por_estado`,
  `tiempos` (segundos por archivo) y `peor_caso`.

- [ ] **Step 1: Escribir el script**

```python
#!/usr/bin/env python3
"""Corre la ingesta sobre archivos REALES y reporta estados y tiempos.

No vive en CI: necesita documentos de clientes, que no entran al repositorio
(ver EXCEPCIONES en policy/tests/test_archivos_de_test_wireados_en_ci.py).

Cubre §7.A.2 y §7.C del spec. Los numeros se anotan en DEUDA.md con fecha:
una prueba de rendimiento vieja es una VERDAD OPERACIONAL caducada.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from procesamiento import ingesta


def verificar(directorio: Path, trabajo: Path) -> dict:
    estados: Counter[str] = Counter()
    tiempos: list[tuple[str, float]] = []
    for archivo in sorted(Path(directorio).rglob("*")):
        if not archivo.is_file():
            continue
        comienzo = time.monotonic()
        ficha = ingesta.ingerir(archivo, Path(trabajo))
        tiempos.append((archivo.name, round(time.monotonic() - comienzo, 3)))
        estados[ficha.estado] += 1
    peor = max(tiempos, key=lambda t: t[1], default=("-", 0.0))
    return {
        "archivos": len(tiempos),
        "por_estado": dict(estados),
        "peor_caso": {"archivo": peor[0], "segundos": peor[1]},
        "tiempos": tiempos,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directorio", type=Path)
    p.add_argument("--trabajo", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(verificar(a.directorio, a.trabajo), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Declararlo en EXCEPCIONES del detector de cobertura**

En `policy/tests/test_archivos_de_test_wireados_en_ci.py`, agregar a `EXCEPCIONES`:

```python
    "scripts/verificar_archivos_reales.py": (
        "No es un archivo de test: es un verificador manual que necesita "
        "documentos de clientes, que no entran al repositorio. Cubre §7.A.2 "
        "y §7.C del spec de procesamiento de archivos."
    ),
```

Sólo si el detector lo reclama (su criterio es por nombre: puede que no lo tome como test).
Verificar con: `python -m pytest policy/tests/test_archivos_de_test_wireados_en_ci.py -v`

- [ ] **Step 3: Correr sobre archivos reales**

Traer una muestra desde Nextcloud a un directorio FUERA del repositorio, con al menos:
un `.xlsx` de estados financieros con varias hojas, un PDF nativo con tablas, y un PDF
escaneado con el app del teléfono.

```bash
python scripts/verificar_archivos_reales.py ~/pruebas-procesamiento   --trabajo /tmp/verificacion-real
```

- [ ] **Step 4: Las 10 cifras — verificación de Fernando**

Del estado financiero escaneado, tomar **10 cifras elegidas por Fernando** y compararlas
contra el extracto. **Esto no lo decide Hyde**: es el criterio §7.A.2 y lo firma él.

Si tesseract falla estas 10 cifras, **recién ahí** se evalúa PaddleOCR (spec §6). No antes.

- [ ] **Step 5: Anotar los números en DEUDA.md, con fecha**

Peor caso en segundos, estados por archivo, y el resultado de las 10 cifras.
Sin número medido no hay GO (Principio I y regla 4 del rendimiento).

- [ ] **Step 6: Commit**

```bash
git add scripts/verificar_archivos_reales.py policy/tests/test_archivos_de_test_wireados_en_ci.py DEUDA.md
git commit -m "feat(procesamiento): verificador con archivos reales y rendimiento medido"
```

## Verificación final (antes de pedir revisión)

- [ ] `python -m pytest procesamiento scripts/_medir_ahorro_extractos_test.py -v` → todo verde
- [ ] `python -m pytest policy/tests/test_archivos_de_test_wireados_en_ci.py -v` → verde
- [ ] `python -m pytest policy/tests/ -v` → sin regresiones
- [ ] `grep -rn "pandas" procesamiento/ requirements-archivos.txt` → sin resultados
- [ ] Los 4 servicios siguen `active` (esto no los toca, pero se confirma)
- [ ] El número de ahorro, anotado con fecha
