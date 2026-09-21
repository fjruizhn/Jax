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
        except BaseException as exc:  # se recolecta, se afirma después
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
        except BaseException as exc:
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


def test_I3_mismos_bytes_distinta_extension_no_comparten_cache(tmp_path: Path, monkeypatch):
    origen_xlsx = tmp_path / "e.xlsx"
    origen_docx = tmp_path / "e.docx"
    contenido = b"contenido identico byte a byte"
    origen_xlsx.write_bytes(contenido)
    origen_docx.write_bytes(contenido)
    assert sha256_de(origen_xlsx) == sha256_de(origen_docx)  # mismos bytes, a propósito

    llamadas: list[str] = []

    def falsa(destino):
        llamadas.append(destino.suffix)
        return Resultado(
            estado="ok", salidas={"x.txt": "hola"}, detalle={},
            extractor=excel.EXTRACTOR, version=excel._version(),  # I-2 no confunde este test
        )

    monkeypatch.setattr(compuerta, "extraer", falsa)

    trabajo = tmp_path / "trabajo"
    primera = ingesta.ingerir(origen_xlsx, trabajo)
    segunda = ingesta.ingerir(origen_docx, trabajo)

    assert primera.sha256 == segunda.sha256
    assert llamadas == [".xlsx", ".docx"], (
        f"I-3 REABIERTO: la segunda ingesta usó el caché de la primera pese a la "
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
# M-2: mutación "fecha constante" -- probada por separado de I-4/I-5 para
# que tenga un test dedicado que la mate.
# ---------------------------------------------------------------------------


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
