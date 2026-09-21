"""El camino de entrada más simple: `procesar(proyecto, rutas) -> dict` y su
línea de comandos. `procesamiento/` ya está terminado y auditado (jax#252) --
esto es sólo lo que lo pone a trabajar contra un proyecto real.

Tres requisitos del encargo, uno a uno:
  1. se conserva la subcarpeta de origen dentro de `fuente/` (slugificada
     componente por componente -- el nombre del archivo queda intacto) ->
     test_procesar_conserva_subcarpetas_y_arma_informe
  2. las carpetas que empiezan con `_` se ignoran (no se descienden) ->
     test_ignora_carpetas_que_empiezan_con_guion_bajo
  3. informe legible: por archivo (estado/extractor/tamaños/segundos) y un
     resumen (reparto por estado, cuántos caben en el tope de
     `tool_authority.MAX_READ_BYTES`, tiempo total) ->
     test_resumen_reparto_por_estado_y_tope_de_lectura

Más: el slug del proyecto arma la carpeta de trabajo esperada
(`$JAX_WORKSPACE_DIR/proyectos/<slug>/`), y la CLI (`main()`) llama a
`procesar()` con lo que parseó de `sys.argv`.

`openpyxl` es opcional acá (fixture `_libro`): la mayoría de los tests usan
`.txt` plano, que no necesita ninguna dependencia -- la compuerta lo
resuelve a `sin_extractor` (ningún extractor para `.txt`), que alcanza para
probar el camino de entrada sin pagar el costo de un extractor real.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# `scripts/` no es un paquete (sin `__init__.py`) -- mismo patrón que
# `scripts/_medir_utilidad_extractos_test.py`: se agrega la raíz del repo a
# `sys.path` para que el namespace package implícito resuelva, sin depender
# de que PYTHONPATH ya lo traiga.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motor_registry import tool_authority

from procesamiento.extractores import excel
from scripts import procesar_archivos


@pytest.fixture(autouse=True)
def _workspace_root_es_tmp(tmp_path, monkeypatch):
    """Mismo patrón que `procesamiento/_ingesta_test.py`: el jail se
    parchea a un tempdir por test, nunca al workspace real."""
    monkeypatch.setattr(tool_authority, "WORKSPACE_ROOT", tmp_path.resolve())


def _libro(destino: Path, valor: int = 100) -> Path:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.active["A1"] = "ACTIVOS"
    wb.active["B1"] = valor
    wb.save(destino)
    return destino


def test_procesar_conserva_subcarpetas_y_arma_informe(tmp_path: Path):
    raiz = tmp_path / "origen"
    carpeta_ef = raiz / "Estados Financieros"
    carpeta_ef.mkdir(parents=True)
    _libro(carpeta_ef / "EEFF.xlsx")
    carpeta_legal = raiz / "Documentación Legal"
    carpeta_legal.mkdir(parents=True)
    (carpeta_legal / "contrato.txt").write_text("hola mundo", encoding="utf8")

    r = procesar_archivos.procesar("Proyecto Demo", [raiz])

    trabajo = tmp_path / "proyectos" / "proyecto-demo"
    assert r["trabajo"] == str(trabajo)
    assert (trabajo / "fuente" / "estados-financieros" / "EEFF.xlsx").is_file()
    assert (trabajo / "fuente" / "documentacion-legal" / "contrato.txt").is_file()

    por_nombre = {d["archivo"]: d for d in r["documentos"]}
    assert por_nombre["EEFF.xlsx"]["estado"] == "ok"
    assert por_nombre["EEFF.xlsx"]["extractor"] == excel.EXTRACTOR
    assert por_nombre["EEFF.xlsx"]["original_bytes"] > 0
    assert por_nombre["EEFF.xlsx"]["extracto_bytes"] > 0
    assert por_nombre["EEFF.xlsx"]["tiempo_s"] >= 0

    assert por_nombre["contrato.txt"]["estado"] == "sin_extractor"
    assert por_nombre["contrato.txt"]["extractor"] == "ninguno"
    assert por_nombre["contrato.txt"]["extracto_bytes"] == 0


def test_ignora_carpetas_que_empiezan_con_guion_bajo(tmp_path: Path):
    raiz = tmp_path / "origen"
    (raiz / "_extractos").mkdir(parents=True)
    (raiz / "_extractos" / "viejo.txt").write_text("x", encoding="utf8")
    (raiz / "_extractos_ocr").mkdir(parents=True)
    (raiz / "_extractos_ocr" / "vacio.txt").write_text("", encoding="utf8")
    (raiz / "real.txt").write_text("contenido real", encoding="utf8")

    r = procesar_archivos.procesar("Ignora", [raiz])

    nombres = {d["archivo"] for d in r["documentos"]}
    assert nombres == {"real.txt"}, (
        f"se ingirieron archivos de una carpeta con guion bajo: {nombres}"
    )

    trabajo = tmp_path / "proyectos" / "ignora"
    assert not (trabajo / "fuente" / "extractos").exists()
    assert not (trabajo / "fuente" / "extractos-ocr").exists()


def test_resumen_reparto_por_estado_y_tope_de_lectura(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(tool_authority, "MAX_READ_BYTES", 5)
    raiz = tmp_path / "origen"
    raiz.mkdir()
    (raiz / "a.txt").write_text("x", encoding="utf8")  # 1 byte -- cabe
    (raiz / "b.txt").write_text("contenido mas largo", encoding="utf8")  # no cabe

    r = procesar_archivos.procesar("Tope", [raiz])

    por_nombre = {d["archivo"]: d for d in r["documentos"]}
    assert por_nombre["a.txt"]["cabe_en_tope"] is True
    assert por_nombre["b.txt"]["cabe_en_tope"] is False

    resumen = r["resumen"]
    assert resumen["total_documentos"] == 2
    assert resumen["por_estado"] == {"sin_extractor": 2}
    assert resumen["caben_en_tope"] == 1
    assert resumen["tiempo_total_s"] >= 0


def test_slug_de_proyecto_crea_la_carpeta_esperada(tmp_path: Path):
    raiz = tmp_path / "origen"
    raiz.mkdir()
    (raiz / "a.txt").write_text("x", encoding="utf8")

    r = procesar_archivos.procesar("Lácteos Victoria & Cía (Grupo)", [raiz])

    assert r["slug"] == "lacteos-victoria-cia-grupo"
    assert (
        tmp_path / "proyectos" / "lacteos-victoria-cia-grupo" / "fuente" / "a.txt"
    ).is_file()


def test_procesar_un_solo_archivo_sin_carpeta_va_a_la_raiz_de_fuente(tmp_path: Path):
    origen = tmp_path / "suelto.txt"
    origen.write_text("solo", encoding="utf8")

    r = procesar_archivos.procesar("Suelto", [origen])

    trabajo = tmp_path / "proyectos" / "suelto"
    assert (trabajo / "fuente" / "suelto.txt").is_file()
    assert r["documentos"][0]["archivo"] == "suelto.txt"


def test_main_invoca_procesar_con_proyecto_y_rutas(tmp_path: Path, monkeypatch, capsys):
    raiz = tmp_path / "origen"
    raiz.mkdir()
    (raiz / "a.txt").write_text("x", encoding="utf8")

    monkeypatch.setattr(sys, "argv", ["procesar_archivos.py", "CLI Demo", str(raiz)])
    rc = procesar_archivos.main()

    assert rc == 0
    salida = capsys.readouterr().out
    assert "cli-demo" in salida
    assert (tmp_path / "proyectos" / "cli-demo" / "fuente" / "a.txt").is_file()
