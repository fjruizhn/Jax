"""Ingesta y caché -- la tarea que produce el ahorro. Todo lo anterior es
maquinaria para esto.

Ronda de arreglo (2026-09-21, sobre el draft del brief task-8-brief.md):
Fernando encontró el defecto ANTES de que este archivo existiera --
`destino = fuente / origen.name` hacía MUTABLE a `fuente/`, que tenía que
ser inmutable: dos archivos distintos con el mismo nombre (un
`balance.xlsx` de un cliente y otro `balance.xlsx` de otro) y el segundo
pisaba al primero, en silencio -- el peor desenlace posible de todo este
proyecto (perder el original de alguien).

Ruling: ante colisión de nombre con contenido distinto, se guarda como
`<nombre>-<primeros 8 del sha256><extension>`, y la ficha registra el
nombre REAL almacenado, no el que traía. Si el archivo ya está con la
misma huella, no se copia de nuevo (eso sí estaba bien en el draft).
`test_colision_de_nombre_con_contenido_distinto_no_pisa_el_original`
reproduce el caso exacto del ruling.
"""
import json
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
# El defecto que Fernando encontró en el draft: `fuente/` tenía que ser
# inmutable y no lo era. Ver docstring del módulo.
# ---------------------------------------------------------------------------


def test_colision_de_nombre_con_contenido_distinto_no_pisa_el_original(tmp_path: Path):
    """Dos archivos distintos con el mismo nombre -- un `balance.xlsx` de
    un cliente y otro `balance.xlsx` de otro. Los dos tienen que sobrevivir
    en `fuente/`, con sus dos extractos, y el primero tiene que seguir
    siendo legible DESPUES de ingerir el segundo."""
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

    # El primero sigue vivo, en el mismo lugar, con el mismo contenido --
    # el segundo NO lo piso.
    assert ruta_primera.is_file()
    assert sha256_de(ruta_primera) == huella_primera_antes == primera.sha256

    # El segundo vive bajo un nombre DISTINTO (misma carpeta `fuente/`).
    ruta_segunda = trabajo / "fuente" / Path(segunda.origen).name
    assert ruta_segunda.is_file()
    assert sha256_de(ruta_segunda) == segunda.sha256
    assert ruta_primera != ruta_segunda

    # Los dos originales sobreviven, literalmente, en el directorio.
    nombres_en_fuente = sorted(p.name for p in (trabajo / "fuente").iterdir())
    assert len(nombres_en_fuente) == 2

    # Y cada uno con su propio extracto, no mezclado con el del otro.
    csv_primera = next(
        ingesta.ruta_procesado(trabajo, primera.sha256).glob("*.csv")
    ).read_text()
    csv_segunda = next(
        ingesta.ruta_procesado(trabajo, segunda.sha256).glob("*.csv")
    ).read_text()
    assert "100" in csv_primera
    assert "999" in csv_segunda


def test_mismo_nombre_y_mismo_contenido_no_duplica_en_fuente(tmp_path: Path):
    """Contraparte del test de colisión: si el archivo que ya está en
    `fuente/` tiene la MISMA huella, no hay colisión real -- no se copia
    de nuevo ni se inventa un nombre alterno."""
    trabajo = tmp_path / "trabajo"
    carpeta_a = tmp_path / "cliente-a"
    carpeta_b = tmp_path / "cliente-b"
    carpeta_a.mkdir()
    carpeta_b.mkdir()
    origen_a = _libro(carpeta_a / "balance.xlsx", valor=100)
    origen_b = _libro(carpeta_b / "balance.xlsx", valor=100)  # mismo contenido

    primera = ingesta.ingerir(origen_a, trabajo)
    segunda = ingesta.ingerir(origen_b, trabajo)

    assert primera.sha256 == segunda.sha256
    nombres_en_fuente = sorted(p.name for p in (trabajo / "fuente").iterdir())
    assert len(nombres_en_fuente) == 1


# ---------------------------------------------------------------------------
# La escritura de `procesado/<huella>/` es atómica: si algo se cae a la
# mitad, no puede quedar una carpeta incompleta que PAREZCA completa.
# ---------------------------------------------------------------------------


def test_si_la_escritura_se_cae_a_la_mitad_no_queda_procesado_incompleto(
    tmp_path: Path, monkeypatch
):
    trabajo = tmp_path / "trabajo"
    origen = _libro(tmp_path / "e.xlsx")
    huella = sha256_de(origen)

    original_write_text = Path.write_text

    def falla_al_escribir_la_ficha(self, *args, **kwargs):
        if self.name == "ficha.json":
            raise OSError("disco lleno, simulado")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", falla_al_escribir_la_ficha)

    with pytest.raises(OSError):
        ingesta.ingerir(origen, trabajo)

    # Nada renombrado a `procesado/<huella>/`: no queda una carpeta a
    # medias que un lector desprevenido confunda con un cache valido.
    assert not ingesta.ruta_procesado(trabajo, huella).exists()

    monkeypatch.undo()

    # Reintentar, ya sin el fallo, tiene que curarse solo.
    ficha = ingesta.ingerir(origen, trabajo)
    assert ficha.estado == "ok"
    assert (ingesta.ruta_procesado(trabajo, huella) / "ficha.json").is_file()
