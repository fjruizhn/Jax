"""Ingesta y caché -- la tarea que produce el ahorro. Todo lo anterior es
maquinaria para esto.

=====================================================================
Ronda de arreglo 0 (defecto encontrado por Fernando ANTES del primer commit,
sobre el draft de task-8-brief.md): `destino = fuente / origen.name` hacía
MUTABLE a `fuente/` -- dos archivos distintos con el mismo nombre y el
segundo pisaba al primero, en silencio.
=====================================================================

Ronda de arreglo 1 (2026-09-21, task-8-hallazgos.md -- 3 Críticos, 8
Importantes, veredicto spec NO, sin ratificar nada). Ver el docstring de
`procesamiento/ingesta.py` para el detalle completo de cada hallazgo; acá
sólo un mapa hallazgo -> test:

  C-1 (symlink roto en fuente/ escribe afuera) ->
      test_C1_symlink_roto_en_fuente_no_escribe_afuera
  C-2 (temporal compartido bajo concurrencia corrompe procesado/) ->
      test_C2_concurrencia_real_procesado_queda_consistente
  C-3 (ventana de colisión bajo concurrencia pierde un original) ->
      test_C3_concurrencia_real_fuente_no_se_corrompe
  I-1 (fallos cacheados para siempre) ->
      test_I1_un_fallo_se_cura_solo_cuando_se_arregla_la_causa
  I-2 (no invalida por versión del extractor) ->
      test_I2_extractor_desactualizado_es_fallo_de_cache
  I-3 (clave ignora la extensión de ruteo) ->
      test_I3_mismos_bytes_distinta_extension_no_comparten_cache
  I-4 (ficha corrupta explota para siempre) ->
      test_I4_ficha_corrupta_regenera_sin_explotar
  I-5 (completitud del cache inverificable) ->
      test_I5_borrar_una_salida_la_regenera
  I-6 (nombre largo + colisión -> OSError crudo) ->
      test_I6_nombre_largo_con_colision_no_revienta
  I-7 (trabajo sin jail) -> test_I7_trabajo_fuera_del_workspace_se_rechaza
"""
import json
import threading
import time
from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

from motor_registry import tool_authority

from procesamiento import compuerta, ingesta
from procesamiento.extractores import excel
from procesamiento.ficha import Ficha, sha256_de
from procesamiento.resultado import Resultado


@pytest.fixture(autouse=True)
def _workspace_root_es_tmp(tmp_path, monkeypatch):
    """`ingesta.py` ahora jailea todo contra `tool_authority.WORKSPACE_ROOT`
    (spec §4, C-1) -- mismo patrón que `las_manos/_tool_authority_test.py`:
    se parchea a un tempdir por test, nunca al workspace real."""
    monkeypatch.setattr(tool_authority, "WORKSPACE_ROOT", tmp_path.resolve())


def _libro(destino: Path, valor: int = 100) -> Path:
    wb = openpyxl.Workbook()
    wb.active["A1"] = "ACTIVOS"
    wb.active["B1"] = valor
    wb.save(destino)
    return destino


# ---------------------------------------------------------------------------
# El contrato original del brief (5 tests), adaptado al jail nuevo.
# ---------------------------------------------------------------------------


def test_ingerir_copia_el_original_y_escribe_ficha(tmp_path: Path):
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")
    f = ingesta.ingerir(origen, trabajo)

    assert (trabajo / "fuente" / "e.xlsx").is_file()
    assert f.sha256 == sha256_de(origen)
    assert f.estado == "ok"
    assert (ingesta.ruta_procesado(trabajo, f.sha256) / "ficha.json").is_file()
    # M-2 (mutación "ficha.origen absoluto"): la ficha guarda una ruta
    # RELATIVA a `trabajo`, nunca absoluta.
    assert not Path(f.origen).is_absolute()


def test_el_segundo_pase_NO_vuelve_a_extraer(tmp_path: Path, monkeypatch):
    """El corazon del ahorro: mismo archivo, cero trabajo la segunda vez."""
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")
    ingesta.ingerir(origen, trabajo)

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


# ---------------------------------------------------------------------------
# Ronda de arreglo 0: el defecto que Fernando encontró en el draft.
# ---------------------------------------------------------------------------


def test_colision_de_nombre_con_contenido_distinto_no_pisa_el_original(tmp_path: Path):
    """Dos archivos distintos con el mismo nombre. Los dos tienen que
    sobrevivir en `fuente/`, con sus dos extractos, y el primero tiene que
    seguir siendo legible DESPUES de ingerir el segundo."""
    trabajo = tmp_path / "trabajo"
    carpeta_a = tmp_path / "cliente-a"
    carpeta_b = tmp_path / "cliente-b"
    carpeta_a.mkdir()
    carpeta_b.mkdir()
    origen_a = _libro(carpeta_a / "balance.xlsx", valor=100)
    origen_b = _libro(carpeta_b / "balance.xlsx", valor=999)

    primera = ingesta.ingerir(origen_a, trabajo)

    ruta_primera = trabajo / "fuente" / Path(primera.origen).name
    huella_primera_antes = sha256_de(ruta_primera)

    segunda = ingesta.ingerir(origen_b, trabajo)

    assert primera.sha256 != segunda.sha256
    assert ruta_primera.is_file()
    assert sha256_de(ruta_primera) == huella_primera_antes == primera.sha256

    ruta_segunda = trabajo / "fuente" / Path(segunda.origen).name
    assert ruta_segunda.is_file()
    assert sha256_de(ruta_segunda) == segunda.sha256
    assert ruta_primera != ruta_segunda

    nombres_en_fuente = sorted(p.name for p in (trabajo / "fuente").iterdir())
    assert len(nombres_en_fuente) == 2

    csv_primera = next(
        ingesta.ruta_procesado(trabajo, primera.sha256).glob("*.csv")
    ).read_text()
    csv_segunda = next(
        ingesta.ruta_procesado(trabajo, segunda.sha256).glob("*.csv")
    ).read_text()
    assert "100" in csv_primera
    assert "999" in csv_segunda


def test_mismo_nombre_y_mismo_contenido_no_duplica_en_fuente(tmp_path: Path):
    """Contraparte: si el archivo que ya está en `fuente/` tiene la MISMA
    huella, no hay colisión real -- no se copia de nuevo ni se inventa un
    nombre alterno."""
    trabajo = tmp_path / "trabajo"
    carpeta_a = tmp_path / "cliente-a"
    carpeta_b = tmp_path / "cliente-b"
    carpeta_a.mkdir()
    carpeta_b.mkdir()
    origen_a = _libro(carpeta_a / "balance.xlsx", valor=100)
    origen_b = _libro(carpeta_b / "balance.xlsx", valor=100)

    primera = ingesta.ingerir(origen_a, trabajo)
    segunda = ingesta.ingerir(origen_b, trabajo)

    assert primera.sha256 == segunda.sha256
    nombres_en_fuente = sorted(p.name for p in (trabajo / "fuente").iterdir())
    assert len(nombres_en_fuente) == 1


def test_si_la_escritura_se_cae_a_la_mitad_no_queda_procesado_incompleto(tmp_path: Path):
    """Ojo: NO se usa el fixture `monkeypatch` para el parche de
    `Path.write_text` -- `monkeypatch.undo()` revertiría TAMBIÉN el jail de
    `WORKSPACE_ROOT` del fixture autouse (comparten la misma instancia).
    Se usa `unittest.mock.patch.object` como context manager, scopeado
    sólo a la llamada que tiene que fallar."""
    import unittest.mock

    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")
    huella = sha256_de(origen)

    original_write_text = Path.write_text

    def falla_al_escribir_la_ficha(self, *args, **kwargs):
        if self.name == "ficha.json":
            raise OSError("disco lleno, simulado")
        return original_write_text(self, *args, **kwargs)

    with unittest.mock.patch.object(Path, "write_text", falla_al_escribir_la_ficha):
        with pytest.raises(OSError):
            ingesta.ingerir(origen, trabajo)

    assert not ingesta.ruta_procesado(trabajo, huella).exists()

    ficha = ingesta.ingerir(origen, trabajo)
    assert ficha.estado == "ok"
    assert (ingesta.ruta_procesado(trabajo, huella) / "ficha.json").is_file()


# ---------------------------------------------------------------------------
# C-1 -- spec §4, incumplido en el draft: un symlink roto en `fuente/`
# escribía A TRAVÉS de él, afuera del workspace.
# ---------------------------------------------------------------------------


def test_C1_symlink_roto_en_fuente_no_escribe_afuera(tmp_path: Path, tmp_path_factory):
    """Caso exacto de C-1: `fuente/informe.xlsx` es un symlink ROTO que
    apunta afuera del workspace. La ingesta tiene que: (1) NUNCA escribir
    a través de él -- el archivo afuera nunca se crea; (2) NUNCA tocarlo
    -- sigue siendo el mismo symlink roto después; (3) igual completar la
    ingesta, con un nombre alterno."""
    afuera = tmp_path_factory.mktemp("afuera-del-workspace")
    objetivo_prohibido = afuera / "escrito.dat"
    assert not objetivo_prohibido.exists()

    trabajo = tmp_path / "trabajo"
    fuente_dir = trabajo / "fuente"
    fuente_dir.mkdir(parents=True)

    origen = _libro(tmp_path / "informe.xlsx")
    symlink_roto = fuente_dir / "informe.xlsx"
    symlink_roto.symlink_to(objetivo_prohibido)
    assert symlink_roto.is_symlink()
    assert not symlink_roto.exists()  # roto: el objetivo no existe

    ficha = ingesta.ingerir(origen, trabajo)

    # (1) EVIDENCIA CENTRAL: nunca se escribió afuera del workspace.
    assert not objetivo_prohibido.exists(), (
        "C-1 REABIERTO: se escribió a través del symlink, afuera del workspace"
    )
    # (2) el symlink roto sigue intacto, nunca se tocó.
    assert symlink_roto.is_symlink()
    assert not symlink_roto.exists()

    # (3) la ingesta igual se completó, con un nombre alterno.
    assert ficha.estado == "ok"
    real = trabajo / "fuente" / Path(ficha.origen).name
    assert real.is_file()
    assert not real.is_symlink()
    assert real.name != "informe.xlsx"


def test_I7_trabajo_fuera_del_workspace_se_rechaza(tmp_path: Path):
    """I-7, cerrado por el ruling de C-1: `trabajo` con `..` que se saldría
    del workspace se rechaza con `ValueError`, ANTES de crear ninguna
    carpeta."""
    origen = _libro(tmp_path / "e.xlsx")
    trabajo_malicioso = tmp_path / ".." / "afuera-del-jail"

    with pytest.raises(ValueError, match="escapa del workspace"):
        ingesta.ingerir(origen, trabajo_malicioso)

    assert not (tmp_path.parent / "afuera-del-jail").exists()


# ---------------------------------------------------------------------------
# C-2 y C-3 -- concurrencia real. El entrelazado EXACTO que usó el auditor
# para C-2 (dos escritores turnándose sobre el MISMO `<huella>.parcial`)
# ya no es reproducible tal cual: el arreglo hace que cada llamada use un
# `.parcial` con un uuid propio, así que A y B nunca podrían tocar el mismo
# directorio para empezar -- lo digo explícitamente en vez de simular un
# entrelazado que la estructura nueva ya no permite. En su lugar, pruebo
# con HILOS REALES (no simulados) sobre el mismo archivo, con un delay
# artificial en `compuerta.extraer` para ensanchar la ventana de la
# extracción real, y verifico que el resultado final es consistente.
# ---------------------------------------------------------------------------


def test_C2_concurrencia_real_procesado_queda_consistente(tmp_path: Path, monkeypatch):
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")
    huella = sha256_de(origen)

    original_extraer = compuerta.extraer
    barrera = threading.Barrier(2)

    def extraer_con_ventana_ancha(destino):
        barrera.wait(timeout=5)
        time.sleep(0.05)  # ensancha la ventana en la que A y B están "en vuelo"
        return original_extraer(destino)

    monkeypatch.setattr(compuerta, "extraer", extraer_con_ventana_ancha)

    resultados: list[Ficha] = []
    errores: list[BaseException] = []

    def correr():
        try:
            resultados.append(ingesta.ingerir(origen, trabajo))
        except BaseException as exc:  # fail-soft: la excepcion de un hilo no propaga al hilo principal; se recolecta en errores y se afirma vacia despues del join
            errores.append(exc)

    hilos = [threading.Thread(target=correr) for _ in range(2)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join(timeout=10)

    assert not errores, f"hilos con excepción bajo concurrencia real: {errores}"
    assert len(resultados) == 2

    carpeta = ingesta.ruta_procesado(trabajo, huella)
    ficha_en_disco = Ficha.desde_json((carpeta / "ficha.json").read_text())
    assert ficha_en_disco.estado == "ok"
    # I-5 aplicado como verificación: TODAS las salidas que la ficha dice
    # tener, están de verdad -- nunca "hojas: 6" con un solo .csv en disco
    # (el defecto exacto de C-2).
    salidas_listadas = ficha_en_disco.detalle["_salidas_ingesta"]
    assert salidas_listadas, "la ficha final no lista ninguna salida"
    for nombre in salidas_listadas:
        assert (carpeta / nombre).is_file(), (
            f"C-2 REABIERTO: la ficha dice que '{nombre}' existe, y no está"
        )


def test_C3_concurrencia_real_fuente_no_se_corrompe(tmp_path: Path):
    """Dos hilos REALES ingiriendo, al mismo tiempo, dos archivos DISTINTOS
    con el MISMO nombre. `O_CREAT|O_EXCL` es atómico a nivel de kernel --
    la garantía no depende de cómo el scheduler entrelace los hilos."""
    trabajo = tmp_path / "trabajo"
    carpeta_a = tmp_path / "cliente-a"
    carpeta_b = tmp_path / "cliente-b"
    carpeta_a.mkdir()
    carpeta_b.mkdir()
    origen_a = _libro(carpeta_a / "balance.xlsx", valor=111)
    origen_b = _libro(carpeta_b / "balance.xlsx", valor=222)

    barrera = threading.Barrier(2)
    resultados: dict[str, Ficha] = {}
    errores: list[BaseException] = []

    def correr(etiqueta: str, origen: Path):
        try:
            barrera.wait(timeout=5)
            resultados[etiqueta] = ingesta.ingerir(origen, trabajo)
        except BaseException as exc:  # fail-soft: la excepcion de un hilo no propaga al hilo principal; se recolecta en errores y se afirma vacia despues del join
            errores.append(exc)

    hilos = [
        threading.Thread(target=correr, args=("a", origen_a)),
        threading.Thread(target=correr, args=("b", origen_b)),
    ]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join(timeout=10)

    assert not errores, f"hilos con excepción bajo concurrencia real: {errores}"
    assert len(resultados) == 2

    ficha_a, ficha_b = resultados["a"], resultados["b"]
    assert ficha_a.sha256 != ficha_b.sha256

    ruta_a = trabajo / "fuente" / Path(ficha_a.origen).name
    ruta_b = trabajo / "fuente" / Path(ficha_b.origen).name
    assert ruta_a != ruta_b, "C-3 REABIERTO: los dos originales colapsaron al mismo nombre"
    assert ruta_a.is_file() and ruta_b.is_file()
    assert sha256_de(ruta_a) == ficha_a.sha256
    assert sha256_de(ruta_b) == ficha_b.sha256

    csv_a = next(ingesta.ruta_procesado(trabajo, ficha_a.sha256).glob("*.csv")).read_text()
    csv_b = next(ingesta.ruta_procesado(trabajo, ficha_b.sha256).glob("*.csv")).read_text()
    assert "111" in csv_a
    assert "222" in csv_b


# ---------------------------------------------------------------------------
# I-1 -- el hallazgo más revelador: los fallos quedaban condenados para
# siempre, y nada lo había decidido.
# ---------------------------------------------------------------------------


def test_I1_un_fallo_se_cura_solo_cuando_se_arregla_la_causa(tmp_path: Path, monkeypatch):
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")

    def falla_transitoriamente(destino):
        return Resultado(
            estado="error", salidas={}, detalle={"razon": "tesseract no instalado"},
            extractor="tesseract", version="5.5.0",
        )

    monkeypatch.setattr(compuerta, "extraer", falla_transitoriamente)
    primera = ingesta.ingerir(origen, trabajo)
    assert primera.estado == "error"

    llamadas: list[Path] = []

    def ahora_funciona(destino):
        llamadas.append(destino)
        return Resultado(
            estado="ok", salidas={"texto.txt": "leido"}, detalle={},
            extractor="tesseract", version="5.5.1",
        )

    monkeypatch.setattr(compuerta, "extraer", ahora_funciona)
    segunda = ingesta.ingerir(origen, trabajo)

    assert llamadas, "I-1 REABIERTO: no se reintentó la extracción tras arreglar la causa"
    assert segunda.estado == "ok"


# ---------------------------------------------------------------------------
# I-2 -- el caché no invalidaba cuando cambiaba el extractor o su versión.
# ---------------------------------------------------------------------------


def test_I2_extractor_desactualizado_es_fallo_de_cache(tmp_path: Path, monkeypatch):
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")
    ingesta.ingerir(origen, trabajo)  # extracción real, extractor="openpyxl"

    # "Cambió el extractor a 99.9.9" -- la versión VIGENTE ya no coincide
    # con la que quedó grabada en la ficha.
    monkeypatch.setattr(excel, "_version", lambda: "99.9.9")

    llamadas: list[Path] = []
    original_extraer = compuerta.extraer

    def rastreada(destino):
        llamadas.append(destino)
        return original_extraer(destino)

    monkeypatch.setattr(compuerta, "extraer", rastreada)
    ingesta.ingerir(origen, trabajo)

    assert llamadas, "I-2 REABIERTO: se sirvió una ficha vieja pese a que la versión del extractor cambió"


# ---------------------------------------------------------------------------
# I-3 -- la clave ignoraba la extensión, que es lo que la compuerta usa
# para decidir el extractor.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "primera_extension, segunda_extension",
    [(".xlsx", ".docx"), (".docx", ".xlsx")],
    ids=["xlsx-luego-docx", "docx-luego-xlsx"],
)
def test_I3_mismos_bytes_distinta_extension_no_comparten_cache(
    tmp_path: Path, monkeypatch, primera_extension, segunda_extension
):
    """A-4 (task-8-hallazgos-r2.md): el agujero estaba en el test, no en
    el código -- la versión original sólo cubría el orden `.xlsx` ->
    `.docx`. Al revés (`.docx` primero) el mismo defecto reaparece: el
    `.xlsx` recibía el extracto que se hizo para el `.docx`. Parametrizado
    en los DOS órdenes -- mutar `"_extension_ingesta": extension_actual`
    a un valor constante ya no sobrevive sin importar cuál va primero."""
    origen_1 = tmp_path / f"e{primera_extension}"
    origen_2 = tmp_path / f"e{segunda_extension}"
    contenido = b"contenido identico byte a byte"
    origen_1.write_bytes(contenido)
    origen_2.write_bytes(contenido)
    assert sha256_de(origen_1) == sha256_de(origen_2)  # mismos bytes, a propósito

    llamadas: list[str] = []

    def falsa(destino):
        llamadas.append(destino.suffix)
        return Resultado(
            estado="ok", salidas={"x.txt": "hola"}, detalle={},
            extractor=excel.EXTRACTOR, version=excel._version(),  # I-2 no confunde este test
        )

    monkeypatch.setattr(compuerta, "extraer", falsa)

    trabajo = tmp_path / "trabajo"
    primera = ingesta.ingerir(origen_1, trabajo)
    segunda = ingesta.ingerir(origen_2, trabajo)

    assert primera.sha256 == segunda.sha256
    assert llamadas == [primera_extension, segunda_extension], (
        f"I-3/A-4 REABIERTO: la segunda ingesta usó el caché de la primera pese a la "
        f"extensión distinta (llamadas reales a compuerta.extraer: {llamadas})"
    )


# ---------------------------------------------------------------------------
# I-4 -- una ficha corrupta explotaba en cada pase, para siempre.
# ---------------------------------------------------------------------------


def test_I4_ficha_corrupta_regenera_sin_explotar(tmp_path: Path):
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")
    primera = ingesta.ingerir(origen, trabajo)
    carpeta = ingesta.ruta_procesado(trabajo, primera.sha256)

    (carpeta / "ficha.json").write_text("")  # corte de luz a mitad de escritura

    segunda = ingesta.ingerir(origen, trabajo)  # NO debe lanzar ValueError
    assert segunda.estado == "ok"
    assert json.loads((carpeta / "ficha.json").read_text())["estado"] == "ok"


# ---------------------------------------------------------------------------
# I-5 -- la ficha no enumeraba sus salidas: la completitud del caché era
# inverificable POR DISEÑO.
# ---------------------------------------------------------------------------


def test_I5_borrar_una_salida_la_regenera(tmp_path: Path):
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")
    primera = ingesta.ingerir(origen, trabajo)
    carpeta = ingesta.ruta_procesado(trabajo, primera.sha256)

    algun_csv = next(carpeta.glob("*.csv"))
    algun_csv.unlink()
    assert not algun_csv.exists()

    segunda = ingesta.ingerir(origen, trabajo)

    assert segunda.estado == "ok"
    assert algun_csv.is_file(), "I-5 REABIERTO: faltaba una salida y no se regeneró"


# ---------------------------------------------------------------------------
# I-6 -- nombre largo + colisión reventaba con OSError crudo.
# ---------------------------------------------------------------------------


def test_I6_nombre_largo_con_colision_no_revienta(tmp_path: Path):
    """El nombre de ORIGEN tiene que poder existir de verdad en el
    filesystem (<=255), pero lo bastante largo como para que, al agregarle
    el sufijo de huella (colisión -> candidato #2, `-XXXXXXXX`), el
    resultado SÍ se pasaría de 255 sin el recorte de I-6. Con tronco de
    245 caracteres + '.xlsx' (5) = 250 (cabe); + sufijo de 9 (`-` + 8 hex)
    = 259 (NO cabe sin recortar)."""
    trabajo = tmp_path / "trabajo"
    nombre_largo = "x" * 245 + ".xlsx"

    carpeta_a = tmp_path / "cliente-a"
    carpeta_b = tmp_path / "cliente-b"
    carpeta_a.mkdir()
    carpeta_b.mkdir()
    origen_a = _libro(carpeta_a / nombre_largo, valor=1)
    origen_b = _libro(carpeta_b / nombre_largo, valor=2)  # mismo nombre, fuerza colisión

    primera = ingesta.ingerir(origen_a, trabajo)  # no debe lanzar OSError (ENAMETOOLONG)
    segunda = ingesta.ingerir(origen_b, trabajo)  # tampoco -- necesita el sufijo alterno

    assert primera.sha256 != segunda.sha256
    nombres = [p.name for p in (trabajo / "fuente").iterdir()]
    assert len(nombres) == 2
    for nombre in nombres:
        assert len(nombre) <= 255, f"I-6 REABIERTO: nombre de {len(nombre)} caracteres en fuente/"


# ---------------------------------------------------------------------------
# A-1/A-2 (task-8-hallazgos-r2.md) -- MISMO modelo de amenaza que C-1: algo
# plantado en `fuente/`, ahora por el camino de comparar huellas en vez de
# escribir. Un FIFO cuelga `sha256_de` para siempre (sin timeout); un
# directorio revienta con `IsADirectoryError`, determinista, en cada pase.
# ---------------------------------------------------------------------------


def test_A1_fifo_en_fuente_no_cuelga_la_ingesta(tmp_path: Path):
    """El repro exacto del hallazgo: un FIFO con el nombre del candidato
    primario. `sha256_de` abriría el FIFO y bloquearía sin timeout -- acá
    se corre en un hilo daemon con `join(timeout=...)` para que el test
    NUNCA pueda colgar la suite entera aunque el arreglo tuviera un
    agujero; la evidencia con `timeout` de shell REAL (reproduciendo el
    cuelgue contra el código mutado y confirmando que NO cuelga contra el
    arreglado) está pegada en task-8-report.md."""
    import os

    trabajo = tmp_path / "trabajo"
    fuente_dir = trabajo / "fuente"
    fuente_dir.mkdir(parents=True)

    origen = _libro(tmp_path / "informe.xlsx")
    fifo = fuente_dir / "informe.xlsx"
    os.mkfifo(fifo)

    resultado: dict[str, Ficha] = {}
    errores: list[BaseException] = []

    def correr():
        try:
            resultado["ficha"] = ingesta.ingerir(origen, trabajo)
        except BaseException as exc:  # fail-soft: la excepcion de un hilo no propaga al hilo principal; se recolecta para afirmar que no se disparo -- pragma: no cover, solo si el arreglo se rompe
            errores.append(exc)

    hilo = threading.Thread(target=correr, daemon=True)
    hilo.start()
    hilo.join(timeout=5)

    assert not hilo.is_alive(), (
        "A-1 REABIERTO: la ingesta quedó colgada más de 5s con un FIFO en fuente/"
    )
    assert not errores, f"errores inesperados: {errores}"

    ficha = resultado["ficha"]
    assert ficha.estado == "ok"
    real = trabajo / "fuente" / Path(ficha.origen).name
    assert real.is_file()
    assert real.name != "informe.xlsx"  # tuvo que usar el nombre alterno

    # El FIFO nunca se tocó: sigue siendo un FIFO, en el mismo lugar.
    import stat

    assert stat.S_ISFIFO(os.stat(fifo, follow_symlinks=False).st_mode)


def test_A2_directorio_colisionante_cae_al_nombre_alterno(tmp_path: Path):
    trabajo = tmp_path / "trabajo"
    fuente_dir = trabajo / "fuente"
    fuente_dir.mkdir(parents=True)

    origen = _libro(tmp_path / "informe.xlsx")
    directorio_colisionante = fuente_dir / "informe.xlsx"
    directorio_colisionante.mkdir()  # directorio con el nombre del candidato primario

    ficha = ingesta.ingerir(origen, trabajo)  # NO debe lanzar IsADirectoryError

    assert ficha.estado == "ok"
    real = trabajo / "fuente" / Path(ficha.origen).name
    assert real.is_file()
    assert real.name != "informe.xlsx"

    # El directorio colisionante sigue ahí, intacto, vacío.
    assert directorio_colisionante.is_dir()
    assert list(directorio_colisionante.iterdir()) == []


# ---------------------------------------------------------------------------
# A-5 (task-8-hallazgos-r2.md) -- seis mutaciones menores que sobrevivían.
# Tres de las seis (quitar O_NOFOLLOW; quitar el unlink de limpieza en la
# escritura fallida) dejaron de aplicar de raíz: la reescritura de
# `_asegurar_en_fuente` para el hallazgo propio de A-3 (temporal + hardlink
# en vez de escritura directa) las volvió estructuralmente imposibles de
# alcanzar -- el detalle completo, con evidencia, está en task-8-report.md.
# Acá las tres que sí se atacan con un test propio.
# ---------------------------------------------------------------------------


def test_A5_temporal_huerfano_se_limpia_en_finally(tmp_path: Path):
    """Quitar el `finally`/`rmtree` de la escritura de `procesado/` deja un
    directorio `<huella>.<pid>-<uuid>.parcial` huérfano cuando la escritura
    se cae a la mitad -- I-8 (ronda 1) quedó, en los hechos, sin test
    propio: el test de atomicidad sólo miraba que la carpeta FINAL no
    existiera, nunca que el temporal se limpiara."""
    import unittest.mock

    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")

    original_write_text = Path.write_text

    def falla_al_escribir_la_ficha(self, *args, **kwargs):
        if self.name == "ficha.json":
            raise OSError("disco lleno, simulado")
        return original_write_text(self, *args, **kwargs)

    with unittest.mock.patch.object(Path, "write_text", falla_al_escribir_la_ficha):
        with pytest.raises(OSError):
            ingesta.ingerir(origen, trabajo)

    procesado_dir = trabajo / "procesado"
    huerfanos = list(procesado_dir.glob("*.parcial")) if procesado_dir.exists() else []
    assert huerfanos == [], f"A-5 REABIERTO (I-8): quedó un temporal huérfano: {huerfanos}"


def test_A5_ficha_con_huella_distinta_a_la_carpeta_es_fallo_de_cache(tmp_path: Path):
    """Defensivo: la ficha guardada en `procesado/<huella>/` debería tener
    SIEMPRE `sha256 == huella` (así se indexa la carpeta) -- pero si algo
    la corrompe con una huella distinta, confiar en ella ciegamente sería
    servir el extracto de un archivo por otro."""
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")
    primera = ingesta.ingerir(origen, trabajo)
    carpeta = ingesta.ruta_procesado(trabajo, primera.sha256)

    datos = json.loads((carpeta / "ficha.json").read_text())
    datos["sha256"] = "0" * 64  # huella corrupta, distinta a la de la carpeta
    (carpeta / "ficha.json").write_text(json.dumps(datos))

    segunda = ingesta.ingerir(origen, trabajo)

    assert segunda.sha256 == primera.sha256, (
        "A-5 REABIERTO: se sirvió una ficha con huella distinta a la de la carpeta"
    )
    assert json.loads((carpeta / "ficha.json").read_text())["sha256"] == primera.sha256


def test_A5_extension_en_mayusculas_no_rompe_el_cache(tmp_path: Path, monkeypatch):
    """El `.lower()` sobre la extensión evita que 'E.XLSX' y 'e.xlsx'
    (mismos bytes) se traten como una colisión de extensión distinta
    (I-3) y disparen una reextracción innecesaria."""
    trabajo = tmp_path / "trabajo"
    base = _libro(tmp_path / "base.xlsx")
    contenido = base.read_bytes()

    origen_mayusculas = tmp_path / "E.XLSX"
    origen_mayusculas.write_bytes(contenido)
    origen_minusculas = tmp_path / "e.xlsx"
    origen_minusculas.write_bytes(contenido)
    assert sha256_de(origen_mayusculas) == sha256_de(origen_minusculas)

    llamadas: list[Path] = []
    original_extraer = compuerta.extraer

    def rastreada(destino):
        llamadas.append(destino)
        return original_extraer(destino)

    monkeypatch.setattr(compuerta, "extraer", rastreada)

    ingesta.ingerir(origen_mayusculas, trabajo)
    ingesta.ingerir(origen_minusculas, trabajo)

    assert len(llamadas) == 1, (
        f"A-5 REABIERTO: 'E.XLSX' y 'e.xlsx' (mismos bytes) no compartieron "
        f"caché -- se reextrajo {len(llamadas)} veces"
    )


# ---------------------------------------------------------------------------
# M-2: mutación "fecha constante" -- probada por separado de I-4/I-5 para
# que tenga un test dedicado que la mate.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# D-2 (task-9-brief.md, 2026-09-21): un documento que SIEMPRE falla costaba
# 300s en CADA intento -- los `error` no se cachean por diseño (I-1, sigue
# siendo correcto: un fallo transitorio tiene que poder curarse), así que un
# fallo PERMANENTE no tenía techo. Ruling: se conserva la curación y se le
# pone techo -- tres intentos, no uno; el tercero es el último; un cambio de
# `extractor_version` reinicia la cuenta (es la señal de que la causa pudo
# arreglarse); `ok` no se ve afectado por nada de esto.
# ---------------------------------------------------------------------------


def test_D2_el_tercer_intento_es_el_ultimo(tmp_path: Path, monkeypatch):
    """Un archivo que SIEMPRE falla: los primeros TRES `ingerir()` tienen
    que reintentar la extracción de verdad (la causa podría curarse en el
    segundo o el tercero, I-1). Del cuarto en adelante, cero llamadas
    nuevas a `compuerta.extraer` -- se sirve la ficha de error guardada."""
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")

    llamadas: list[Path] = []

    def falla_siempre(destino):
        llamadas.append(destino)
        # La versión se pide DINÁMICAMENTE (`excel._version()`, real) en
        # vez de hardcodear un string -- si no coincide con lo que
        # `_version_vigente` (I-2) lee del módulo instalado, D-2 nunca
        # bloquea (lo trataría como "cambió de versión" en cada llamada) y
        # este test daría un falso verde por casualidad de versión.
        return Resultado(
            estado="error", salidas={}, detalle={"razon": "PDF patologico, timeout"},
            extractor=excel.EXTRACTOR, version=excel._version(),
        )

    monkeypatch.setattr(compuerta, "extraer", falla_siempre)

    for _ in range(3):
        f = ingesta.ingerir(origen, trabajo)
        assert f.estado == "error"
    assert len(llamadas) == 3, (
        f"D-2 REABIERTO: se esperaban 3 llamadas reales a compuerta.extraer "
        f"tras 3 ingestas, hubo {len(llamadas)}"
    )

    # Cuarta, quinta, sexta ingesta: CERO trabajo nuevo -- el tope ya se
    # alcanzó. Ésta es la evidencia central de D-2: sin esto, cada llamada
    # vuelve a pagar el costo completo del extractor (300s en producción).
    for _ in range(3):
        f = ingesta.ingerir(origen, trabajo)
        assert f.estado == "error"
    assert len(llamadas) == 3, (
        f"D-2 REABIERTO: se siguió invocando compuerta.extraer más allá del "
        f"tope de 3 intentos -- {len(llamadas)} llamadas totales"
    )


def test_D2_cambio_de_version_del_extractor_reinicia_la_cuenta(tmp_path: Path, monkeypatch):
    """El tope se alcanzó (3 intentos agotados, extractor en su versión
    VIGENTE de hoy). El extractor se actualiza (I-2: se monkeypatchea
    `excel._version`, igual que `test_I2_...` -- `_version_vigente` lee la
    versión REAL del módulo, no lo que devuelva `compuerta.extraer`, así
    que simular "cambió de versión" es simular ESO, no el resultado) --
    "ese es el momento en que la causa pudo haberse arreglado". La
    siguiente ingesta tiene que volver a intentar de verdad, no servir la
    ficha vieja agotada."""
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")

    llamadas: list[Path] = []

    def falla_siempre(destino):
        llamadas.append(destino)
        return Resultado(
            estado="error", salidas={}, detalle={"razon": "algo raro paso"},
            extractor=excel.EXTRACTOR, version=excel._version(),
        )

    monkeypatch.setattr(compuerta, "extraer", falla_siempre)
    for _ in range(3):
        ingesta.ingerir(origen, trabajo)
    assert len(llamadas) == 3

    # El tope está agotado bajo la versión vigente -- una ingesta más NO
    # debe reintentar todavía.
    ingesta.ingerir(origen, trabajo)
    assert len(llamadas) == 3, "el tope no se respetó antes del cambio de versión"

    # "El extractor cambió de versión" -- la consulta barata que
    # `_version_vigente` hace (I-2) ahora devuelve otra cosa.
    monkeypatch.setattr(excel, "_version", lambda: "99.9.9")
    ingesta.ingerir(origen, trabajo)

    assert len(llamadas) == 4, (
        "D-2 REABIERTO: el cambio de extractor_version no reinició la cuenta "
        f"de intentos -- se siguió sirviendo la ficha agotada de la versión "
        f"vieja (llamadas reales: {len(llamadas)}, se esperaban 4)"
    )


def test_D2_version_desconocida_no_reinicia_la_cuenta_del_tope(tmp_path: Path, monkeypatch):
    """Corrección del hallazgo de la ronda P10 (2026-09-21): `None` de
    `_version_vigente` significa "no se pudo determinar la versión", NO
    "la versión cambió". Son cosas distintas -- confundirlas reabre D-2 en
    silencio: un extractor cuya consulta de versión falla de forma
    persistente reiniciaría la cuenta en CADA ingesta y el tope de 3 dejaría
    de existir, exactamente el defecto de los 300s por llamada que D-2 vino
    a cerrar. El tope se agota bajo la versión vigente de hoy; después se
    rompe `excel._version()` (simulando que la consulta de versión falla) y
    una ingesta más NO tiene que reintentar -- la cuenta sigue en 3."""
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")

    llamadas: list[Path] = []
    # Capturada ANTES de romper `excel._version` -- el mock necesita seguir
    # escribiendo la MISMA version en cada ficha (como haría la extracción
    # real), separado de lo que `_version_vigente` vaya a leer al chequear
    # el cache. Si el mock llamara a `excel._version()` en vivo después de
    # romperla, el RuntimeError saldría de acá, no del camino bajo prueba.
    version_real = excel._version()

    def falla_siempre(destino):
        llamadas.append(destino)
        return Resultado(
            estado="error", salidas={}, detalle={"razon": "algo raro paso"},
            extractor=excel.EXTRACTOR, version=version_real,
        )

    monkeypatch.setattr(compuerta, "extraer", falla_siempre)
    for _ in range(3):
        ingesta.ingerir(origen, trabajo)
    assert len(llamadas) == 3

    # "No se pudo determinar la versión vigente" -- `excel._version()` ahora
    # revienta (dependencia rota, timeout, lo que sea). `_version_vigente`
    # (ingesta.py) lo captura y devuelve `None`: un "no sé", no un "cambió".
    def version_rota():
        raise RuntimeError("no se pudo leer la version instalada")

    monkeypatch.setattr(excel, "_version", version_rota)
    ingesta.ingerir(origen, trabajo)

    assert len(llamadas) == 3, (
        "D-2 REABIERTO: una version VIGENTE desconocida (None) reinició la "
        f"cuenta de intentos como si hubiera 'cambiado' -- se volvió a "
        f"invocar compuerta.extraer ({len(llamadas)} llamadas, se esperaban "
        "3). None es 'no sé', no es 'cambió'."
    )


def test_D2_sin_extractor_se_comporta_igual_que_error(tmp_path: Path, monkeypatch):
    """El ruling es explícito: `sin_extractor` se comporta IGUAL que
    `error` para el tope de reintentos -- mismo mecanismo, mismo número."""
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")

    llamadas: list[Path] = []

    def sin_herramienta(destino):
        llamadas.append(destino)
        return Resultado(
            estado="sin_extractor", salidas={}, detalle={"razon": "no instalado"},
            extractor=excel.EXTRACTOR, version=excel._version(),
        )

    monkeypatch.setattr(compuerta, "extraer", sin_herramienta)

    for _ in range(3):
        f = ingesta.ingerir(origen, trabajo)
        assert f.estado == "sin_extractor"
    assert len(llamadas) == 3

    ingesta.ingerir(origen, trabajo)
    assert len(llamadas) == 3, (
        "D-2 REABIERTO: 'sin_extractor' no respetó el mismo tope de 3 "
        "intentos que 'error'"
    )


def test_D2_un_ok_no_se_ve_afectado_por_el_tope_de_reintentos(tmp_path: Path, monkeypatch):
    """Un `ok` se cachea de la forma de siempre (I-1) -- el mecanismo nuevo
    del tope de reintentos es exclusivo de `error`/`sin_extractor` y no le
    agrega ni le saca nada a este camino."""
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")

    llamadas: list[Path] = []
    original_extraer = compuerta.extraer

    def rastreada(destino):
        llamadas.append(destino)
        return original_extraer(destino)

    monkeypatch.setattr(compuerta, "extraer", rastreada)

    for _ in range(5):
        f = ingesta.ingerir(origen, trabajo)
        assert f.estado == "ok"

    # Cache de siempre: UNA sola extracción real, las otras cuatro cero
    # trabajo -- el tope de reintentos de D-2 no interfiere con esto.
    assert len(llamadas) == 1, (
        f"D-2 REABIERTO: un 'ok' se vio afectado por el mecanismo de tope de "
        f"reintentos -- {len(llamadas)} llamadas reales en vez de 1"
    )


def test_M2_fecha_cambia_entre_regeneraciones(tmp_path: Path):
    """A propósito NO se monkeypatchea `ingesta._ahora`: la mutación que
    este test tiene que matar reemplaza el CUERPO de `_ahora` por un
    string fijo, así que parchear la misma función (o `datetime.now`, que
    la versión mutada ya ni llama) taparía el defecto en vez de
    detectarlo. Se deja pasar tiempo real de reloj -- `_ahora()` trunca a
    segundos (`timespec="seconds"`), así que basta con más de 1s entre las
    dos extracciones reales."""
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")

    primera = ingesta.ingerir(origen, trabajo)
    carpeta = ingesta.ruta_procesado(trabajo, primera.sha256)
    (carpeta / "ficha.json").write_text("")  # fuerza una regeneración (I-4)

    time.sleep(1.1)
    segunda = ingesta.ingerir(origen, trabajo)

    assert segunda.fecha != primera.fecha, (
        "M-2 REABIERTO: la fecha no cambió entre dos extracciones reales separadas en el tiempo"
    )


# ---------------------------------------------------------------------------
# Reserva arquitectónica (final-hallazgos.md, ronda de cierre): "Toda la
# honestidad de este sistema vive en ficha.json, y nadie la lee". Un extracto
# `parcial` tiene que llevar el aviso DENTRO del propio archivo que el
# modelo lee -- la ficha no es lo que un `file_read` sobre `procesado/`
# entrega.
# ---------------------------------------------------------------------------


def test_reserva_extracto_parcial_lleva_encabezado_de_aviso(tmp_path: Path, monkeypatch):
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")

    def falso_parcial(destino):
        return Resultado(
            estado="parcial",
            salidas={"texto.md": "ACTIVOS TOTALES 1000"},
            detalle={"paginas": 3, "paginas_sin_texto": [2]},
            extractor=excel.EXTRACTOR, version=excel._version() or "desconocida",
        )

    monkeypatch.setattr(compuerta, "extraer", falso_parcial)

    ficha = ingesta.ingerir(origen, trabajo)
    carpeta = ingesta.ruta_procesado(trabajo, ficha.sha256)
    contenido = (carpeta / "texto.md").read_text(encoding="utf8")

    assert contenido.startswith("<!-- EXTRACTO PARCIAL"), (
        "RESERVA REABIERTA: un extracto 'parcial' llegó al archivo que el "
        f"modelo lee sin ningún aviso -- contenido: {contenido[:80]!r}"
    )
    assert "ficha.json" in contenido
    assert "ACTIVOS TOTALES 1000" in contenido, (
        "el aviso no puede reemplazar el contenido real, sólo anteponerse"
    )


def test_reserva_extracto_ok_no_lleva_encabezado(tmp_path: Path):
    """Contraparte: el caso feliz no paga ruido -- un 'ok' no lleva ningún
    encabezado antepuesto."""
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")

    ficha = ingesta.ingerir(origen, trabajo)
    assert ficha.estado == "ok"
    carpeta = ingesta.ruta_procesado(trabajo, ficha.sha256)
    salida = ficha.detalle["_salidas_ingesta"][0]
    contenido = (carpeta / salida).read_text(encoding="utf8")

    assert "EXTRACTO PARCIAL" not in contenido


def test_reserva_encabezado_resume_lo_que_falta_sin_repetir_las_cifras(
    tmp_path: Path, monkeypatch
):
    """"Que sea corto y útil: qué falta y dónde está el detalle, no las 141
    cifras" -- el encabezado no vuelca `palabras_dudosas` completo (cada
    entrada trae su propio texto y confianza), sólo la CUENTA."""
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")

    palabras_dudosas = [
        {"pagina": 1, "palabra": f"cifra{i}", "confianza": 10.0} for i in range(141)
    ]

    def falso_parcial(destino):
        return Resultado(
            estado="parcial",
            salidas={"texto.txt": "contenido real del OCR"},
            detalle={"idioma": "spa", "paginas": 1, "palabras_dudosas": palabras_dudosas},
            extractor=excel.EXTRACTOR, version=excel._version() or "desconocida",
        )

    monkeypatch.setattr(compuerta, "extraer", falso_parcial)

    ficha = ingesta.ingerir(origen, trabajo)
    carpeta = ingesta.ruta_procesado(trabajo, ficha.sha256)
    primera_linea = (carpeta / "texto.txt").read_text(encoding="utf8").splitlines()[0]

    assert "141" in primera_linea
    assert "cifra0" not in primera_linea, (
        "el encabezado no puede volcar las 141 cifras, sólo la cuenta"
    )


# ---------------------------------------------------------------------------
# I-6 (final-hallazgos.md, ronda de cierre): un extracto que en total supera
# `tool_authority.MAX_READ_BYTES` es ilegible por `file_read` aunque esté
# COMPLETO y sea CORRECTO -- el criterio §7.B.2 no lo ve si sólo evalúa
# documentos cuyo ORIGINAL no cabía (xlsx-04: 159.077 B -> 17.861.532 B,
# 'ok', 89x el tope). La ficha lo declara para que deje de ser invisible.
# ---------------------------------------------------------------------------


def test_I6_extracto_que_excede_el_tope_se_declara_en_la_ficha(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(tool_authority, "MAX_READ_BYTES", 100)
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")

    def falso_grande(destino):
        return Resultado(
            estado="ok", salidas={"hoja1.csv": "x" * 500},  # 500 B > tope de 100
            detalle={}, extractor=excel.EXTRACTOR, version=excel._version() or "desconocida",
        )

    monkeypatch.setattr(compuerta, "extraer", falso_grande)

    ficha = ingesta.ingerir(origen, trabajo)

    assert ficha.estado == "ok", "el estado NO cambia -- el extracto esta completo y es correcto"
    assert ficha.detalle["excede_tope_lectura"] is True, (
        "I-6 REABIERTO: un extracto mas grande que el tope de lectura quedo "
        "invisible en la ficha"
    )
    assert ficha.detalle["excede_tope_lectura_bytes"]["extracto"] == 500
    assert ficha.detalle["excede_tope_lectura_bytes"]["original"] > 0
    assert ficha.detalle["excede_tope_lectura_bytes"]["tope"] == 100


def test_I6_extracto_que_cabe_no_declara_el_flag(tmp_path: Path):
    """Contraparte: un extracto que cabe en el tope no lleva ningún flag --
    ni `False` explícito, para no ensuciar `detalle` en el caso normal."""
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")

    ficha = ingesta.ingerir(origen, trabajo)

    assert "excede_tope_lectura" not in ficha.detalle


def test_I6_excede_tope_lectura_se_cuenta_DESPUES_de_decorar(tmp_path: Path, monkeypatch):
    """MINOR (final-hallazgos.md, adenda 2026-09-21): antes se sumaban los
    bytes de `resultado.salidas.values()` SIN transformar -- subestimando
    el tamaño real cuando el encabezado de la reserva arquitectónica se
    antepone. Acá el contenido crudo (sin encabezado) queda POR DEBAJO del
    tope, y sólo con el encabezado antepuesto lo supera -- si el conteo
    fuera pre-decoración, `excede_tope_lectura` no se declararía."""
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")

    contenido_crudo = "x" * 90  # crudo: 90 B, cabe bajo cualquier tope >= 90

    def falso_parcial(destino):
        return Resultado(
            estado="parcial",
            salidas={"texto.md": contenido_crudo},
            detalle={"razon": "algo quedo afuera"},
            extractor=excel.EXTRACTOR, version=excel._version() or "desconocida",
        )

    monkeypatch.setattr(compuerta, "extraer", falso_parcial)
    # El encabezado real (con el prefijo "<!-- EXTRACTO PARCIAL ..." más la
    # ruta de la ficha) suma bastante más de 10 B -- el tope se fija entre
    # el tamaño crudo (90) y lo que da crudo+encabezado, así que SÓLO se
    # supera si el conteo incluye el encabezado.
    monkeypatch.setattr(tool_authority, "MAX_READ_BYTES", 95)

    ficha = ingesta.ingerir(origen, trabajo)

    assert ficha.estado == "parcial"
    assert ficha.detalle.get("excede_tope_lectura") is True, (
        "MINOR REABIERTO: el conteo pre-decoracion no vio que el encabezado "
        "empuja el archivo real por encima del tope"
    )
    assert ficha.detalle["excede_tope_lectura_bytes"]["extracto"] > len(
        contenido_crudo.encode("utf8")
    )


# ---------------------------------------------------------------------------
# MAJOR 1 (final-hallazgos.md, adenda 2026-09-21, ruling de Fernando sobre
# su propio ruling anterior): el encabezado de la reserva arquitectónica se
# antepone a TODA salida sin mirar el formato -- un comentario HTML dentro
# de un CSV no es un comentario, es UNA FILA MÁS. `csv.DictReader` la toma
# como nombre de columna y manda todo lo demás a la clave `None`.
#
# Por formato: `.md`/`.txt` siguen con el aviso DENTRO (ya probado arriba).
# Un `.csv` queda INTACTO y el aviso va a un archivo hermano `AVISO.txt`.
# ---------------------------------------------------------------------------


def test_csv_parcial_queda_intacto_y_el_aviso_va_a_AVISO_txt(tmp_path: Path, monkeypatch):
    """El caso EXACTO del hallazgo: un `.xlsx` con una fórmula sin caché
    (condición de 'parcial' frecuentísima en libros financieros) -- el CSV
    tiene que quedar byte a byte igual al que produjo el extractor, y
    `csv.DictReader` sobre él tiene que dar las columnas REALES, no el
    encabezado como nombre de columna."""
    import csv
    import io

    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")

    csv_real = "ACTIVOS,PASIVOS\n1000,400\n2000,800\n"

    def falso_parcial(destino):
        return Resultado(
            estado="parcial",
            salidas={"01-base.csv": csv_real},
            detalle={
                "hojas": 2, "hojas_extraidas": 2,
                "formulas_sin_valor": {"total": 1, "hojas": ["totales"]},
            },
            extractor=excel.EXTRACTOR, version=excel._version() or "desconocida",
        )

    monkeypatch.setattr(compuerta, "extraer", falso_parcial)

    ficha = ingesta.ingerir(origen, trabajo)
    carpeta = ingesta.ruta_procesado(trabajo, ficha.sha256)

    # El CSV es BYTE A BYTE el que produjo el extractor -- sin decorar.
    csv_en_disco = (carpeta / "01-base.csv").read_text(encoding="utf8")
    assert csv_en_disco == csv_real, (
        "MAJOR 1 REABIERTO: el CSV salió decorado -- csv.DictReader lo "
        "leería mal"
    )

    # Evidencia con la MISMA herramienta que reprodujo el hallazgo:
    # csv.DictReader sobre el archivo real en disco da las columnas
    # correctas, no el encabezado como nombre de columna.
    filas = list(csv.DictReader(io.StringIO(csv_en_disco)))
    assert set(filas[0]) == {"ACTIVOS", "PASIVOS"}, (
        f"csv.DictReader tomó columnas equivocadas: {set(filas[0])}"
    )
    assert filas[0]["ACTIVOS"] == "1000"
    assert None not in filas[0], (
        "csv.DictReader mandó datos reales a la clave None -- el "
        "encabezado se coló como fila"
    )

    # El aviso SÍ existe, en un archivo hermano.
    aviso = (carpeta / "AVISO.txt").read_text(encoding="utf8")
    assert aviso.startswith("<!-- EXTRACTO PARCIAL")
    assert "ficha.json" in aviso

    # Y queda listado -- si se borra, I-5 lo nota y se regenera.
    assert "AVISO.txt" in ficha.detalle["_salidas_ingesta"]


def test_md_parcial_sigue_con_el_aviso_DENTRO_no_en_AVISO_txt(tmp_path: Path, monkeypatch):
    """Contraparte: un `.md`/`.txt` NO produce `AVISO.txt` -- el aviso
    sigue yendo dentro del archivo, que es donde sirve para ese formato."""
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")

    def falso_parcial(destino):
        return Resultado(
            estado="parcial",
            salidas={"texto.md": "ACTIVOS TOTALES 1000"},
            detalle={"razon": "algo quedo afuera"},
            extractor=excel.EXTRACTOR, version=excel._version() or "desconocida",
        )

    monkeypatch.setattr(compuerta, "extraer", falso_parcial)

    ficha = ingesta.ingerir(origen, trabajo)
    carpeta = ingesta.ruta_procesado(trabajo, ficha.sha256)

    assert not (carpeta / "AVISO.txt").exists()
    assert "AVISO.txt" not in ficha.detalle["_salidas_ingesta"]
    assert (carpeta / "texto.md").read_text(encoding="utf8").startswith(
        "<!-- EXTRACTO PARCIAL"
    )


# ---------------------------------------------------------------------------
# MINOR 3 (final-hallazgos.md, adenda 2026-09-21): la pérdida MÁS GRAVE que
# puede tener un libro -- una hoja entera que no se pudo extraer -- era
# justo la única que `_resumen_parcial` no nombraba.
# ---------------------------------------------------------------------------


def test_resumen_parcial_nombra_las_hojas_perdidas_por_error(tmp_path: Path, monkeypatch):
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")

    def falso_parcial(destino):
        return Resultado(
            estado="parcial",
            salidas={"01-buena.csv": "A,B\r\n1,2\r\n"},
            detalle={
                "hojas": 2, "hojas_extraidas": 1,
                "fallidas": ["MALA: ValueError: boom"],
            },
            extractor=excel.EXTRACTOR, version=excel._version() or "desconocida",
        )

    monkeypatch.setattr(compuerta, "extraer", falso_parcial)

    ficha = ingesta.ingerir(origen, trabajo)
    carpeta = ingesta.ruta_procesado(trabajo, ficha.sha256)
    aviso = (carpeta / "AVISO.txt").read_text(encoding="utf8")

    assert "MALA" in aviso, (
        "MINOR 3 REABIERTO: la hoja perdida por error no aparece en el "
        f"aviso -- {aviso!r}"
    )
    assert "extracto parcial -- ver ficha.json" not in aviso, (
        "cayó al genérico en vez de nombrar la hoja perdida"
    )
