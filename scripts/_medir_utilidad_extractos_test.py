"""Medidor de utilidad de extractos -- spec §7.B (corregido 2026-09-21).

Mapa criterio -> test (B.1 es de Fernando, no medible por código, no tiene
test acá):

  B.2 (binario, atado a MAX_READ_BYTES) ->
      test_b2_binario_requiere_extracto_chico_y_original_grande,
      test_b2_es_falso_si_el_original_YA_cabia,
      test_totales_b2_exige_TODOS_los_documentos_aplicables_no_alguno
  B.3 (caché: segundo pase, cero extracciones) ->
      test_segunda_pasada_sobre_los_mismos_archivos_no_extrae_de_nuevo
  B.4 (parcial no puede pasar del 60%) ->
      test_totales_porcentaje_parcial_bajo_60_PASA,
      test_totales_porcentaje_parcial_sobre_60_NO_PASA
  totales / conteos ->
      test_totales_cuenta_extractos_que_caben
  veredicto agregado (mutación "siempre aprobado") ->
      test_medir_end_to_end_arma_veredicto_y_puede_dar_NO_GO
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

# `scripts/` no es un paquete (sin `__init__.py`) -- mismo patrón que
# `task-9-brief.md` y `scripts/_check_mirror_sync_test.py`: se agrega la
# raíz del repo a `sys.path` para que el namespace package implícito
# resuelva, sin depender de que PYTHONPATH ya lo traiga.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motor_registry import tool_authority

from procesamiento import ingesta
from procesamiento.ficha import Ficha
from scripts.medir_utilidad_extractos import _contar_extracciones, _medir_documento, _totales, medir


@pytest.fixture(autouse=True)
def _workspace_root_es_tmp(tmp_path, monkeypatch):
    """Mismo patrón que `procesamiento/_ingesta_test.py`: `medir()` llama a
    `ingesta.ingerir`, que jailea contra `tool_authority.WORKSPACE_ROOT` --
    se parchea a un tempdir por test, nunca al workspace real."""
    monkeypatch.setattr(tool_authority, "WORKSPACE_ROOT", tmp_path.resolve())


def _libro(destino: Path, filas: int = 5) -> Path:
    wb = openpyxl.Workbook()
    for fila in range(1, filas + 1):
        wb.active[f"A{fila}"] = f"CUENTA CONTABLE NUMERO {fila}"
        wb.active[f"B{fila}"] = fila * 1000
    wb.save(destino)
    return destino


def _ficha(estado: str, detalle: dict | None = None) -> Ficha:
    return Ficha(
        sha256="a" * 64,
        origen="fuente/e.xlsx",
        extractor="openpyxl",
        extractor_version="3.1.5",
        fecha="2026-09-21T00:00:00-06:00",
        estado=estado,
        detalle=detalle or {},
    )


def _doc(estado: str, criterio_b2: bool = True) -> dict:
    return {
        "estado": estado,
        "extracto_cabe": criterio_b2,
        "original_no_cabe": True,
        "criterio_b2": criterio_b2,
    }


# ---------------------------------------------------------------------------
# B.3 -- segundo pase sobre los mismos archivos: cero extracciones reales.
# ---------------------------------------------------------------------------


def test_segunda_pasada_sobre_los_mismos_archivos_no_extrae_de_nuevo(tmp_path: Path):
    archivo = _libro(tmp_path / "e.xlsx")
    trabajo = tmp_path / "trabajo"

    r = medir([archivo], trabajo)

    assert r["extracciones_primera_pasada"] == 1
    assert r["extracciones_segunda_pasada"] == 0, (
        f"la segunda pasada volvio a extraer {r['extracciones_segunda_pasada']} "
        "veces: el cache no sirve"
    )
    assert r["veredicto"]["b3_cache_cero_extracciones"] is True


def test_contar_extracciones_refleja_una_extraccion_real_no_un_valor_fijo(tmp_path: Path):
    """Con el cache SANO, un segundo pase da 0 de por sí -- eso solo no
    prueba que `_contar_extracciones` cuente de verdad (una función que
    devolviera `0` a ciegas pasaría igual). Acá se rompe el cache a
    propósito (se borra una salida listada, I-5) para que la llamada
    SIGUIENTE tenga que reextraer -- si `_contar_extracciones` no cuenta
    de verdad (la mutación "no cuenta bien la segunda pasada"), esto lo
    nota."""
    archivo = _libro(tmp_path / "e.xlsx")
    trabajo = tmp_path / "trabajo"

    ficha = ingesta.ingerir(archivo, trabajo)
    carpeta = ingesta.ruta_procesado(trabajo, ficha.sha256)
    salidas = ficha.detalle["_salidas_ingesta"]
    assert salidas, "el fixture no genero ninguna salida -- ajustar _libro"
    (carpeta / salidas[0]).unlink()  # rompe I-5: cache incompleto, fuerza reextraccion

    n = _contar_extracciones([archivo], trabajo)

    assert n == 1, (
        f"se esperaba UNA extraccion real tras romper el cache, se contaron {n}"
    )


# ---------------------------------------------------------------------------
# B.2 -- binario por documento: extracto cabe Y original no cabe.
# ---------------------------------------------------------------------------


def test_b2_binario_requiere_extracto_chico_y_original_grande(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(tool_authority, "MAX_READ_BYTES", 100)

    carpeta = tmp_path / "procesado"
    carpeta.mkdir()
    (carpeta / "hoja1.csv").write_text("x" * 10, encoding="utf8")  # extracto: 10 B

    archivo = tmp_path / "grande.xlsx"
    archivo.write_bytes(b"0" * 500)  # original: 500 B > umbral de 100

    d = _medir_documento(archivo, _ficha("ok"), carpeta, tiempo_s=0.01)

    assert d["original_no_cabe"] is True
    assert d["extracto_cabe"] is True
    assert d["criterio_b2"] is True


def test_b2_es_falso_si_el_original_YA_cabia(tmp_path: Path, monkeypatch):
    """Si el original ya cabía en el límite, extraerlo no resolvió nada --
    el criterio mide la diferencia entre "no se puede usar" y "se puede
    usar", así que no aplica cuando ya se podía usar el original."""
    monkeypatch.setattr(tool_authority, "MAX_READ_BYTES", 1_000)

    carpeta = tmp_path / "procesado"
    carpeta.mkdir()
    (carpeta / "hoja1.csv").write_text("x" * 10, encoding="utf8")

    archivo = tmp_path / "chico.xlsx"
    archivo.write_bytes(b"0" * 500)  # original: 500 B, cabe en el umbral de 1000

    d = _medir_documento(archivo, _ficha("ok"), carpeta, tiempo_s=0.01)

    assert d["original_no_cabe"] is False
    assert d["criterio_b2"] is False


def test_medir_documento_declara_excede_tope_lectura_desde_la_ficha(
    tmp_path: Path, monkeypatch
):
    """I-6 (final-hallazgos.md, ronda de cierre): `_medir_documento` lee el
    flag `excede_tope_lectura` que `ingesta.ingerir` escribe en la ficha
    cuando el extracto TOTAL supera `MAX_READ_BYTES` aunque el original
    cupiera -- caso medido: xlsx-04, original 159.077 B (cabe), extracto
    17.861.532 B (89x el tope), 'ok'."""
    monkeypatch.setattr(tool_authority, "MAX_READ_BYTES", 1_000)

    carpeta = tmp_path / "procesado"
    carpeta.mkdir()
    (carpeta / "hoja1.csv").write_text("x" * 2_000, encoding="utf8")

    archivo = tmp_path / "chico.xlsx"
    archivo.write_bytes(b"0" * 100)  # original: 100 B, cabe en el umbral de 1000

    ficha = _ficha("ok", detalle={"excede_tope_lectura": True})
    d = _medir_documento(archivo, ficha, carpeta, tiempo_s=0.01)

    assert d["original_no_cabe"] is False
    assert d["excede_tope_lectura"] is True


def test_totales_b2_ve_extracto_que_excede_el_tope_aunque_el_original_cupiera():
    """I-6: el criterio §7.B.2 viejo sólo evaluaba documentos cuyo ORIGINAL
    no cabía -- un extracto que el propio sistema INFLÓ hasta superar el
    tope (el original SÍ cabía) quedaba invisible. `excede_tope_lectura`
    trae a la evaluación este caso aunque `original_no_cabe` sea False."""
    docs = [
        {
            **_doc("ok", criterio_b2=False),
            "original_no_cabe": False,
            "excede_tope_lectura": True,
        },
    ]

    t = _totales(docs)

    assert t["documentos_donde_aplica_b2"] == 1
    assert t["documentos_donde_aplica_b2_ok"] == 0
    assert t["b2_extracto_util_pass"] is False, (
        "I-6 REABIERTO: un extracto que excede el tope aunque el original "
        "cupiera quedo invisible para B.2"
    )


def test_totales_b2_exige_TODOS_los_documentos_aplicables_no_alguno():
    docs = [_doc("ok", criterio_b2=True), _doc("ok", criterio_b2=False)]

    t = _totales(docs)

    assert t["documentos_donde_aplica_b2"] == 2
    assert t["documentos_donde_aplica_b2_ok"] == 1
    assert t["b2_extracto_util_pass"] is False, (
        "un solo documento aplicable que no cabe tiene que tumbar el criterio"
    )


def test_totales_cuenta_extractos_que_caben():
    docs = [
        _doc("ok", criterio_b2=True),
        {**_doc("ok", criterio_b2=False), "extracto_cabe": False},
        {**_doc("sin_extractor", criterio_b2=False), "extracto_cabe": False},
    ]

    t = _totales(docs)

    assert t["total_documentos"] == 3
    assert t["extractos_que_caben"] == 1


# ---------------------------------------------------------------------------
# B.4 -- que "parcial" no sea más del 60% de la muestra.
# ---------------------------------------------------------------------------


def test_totales_porcentaje_parcial_bajo_60_PASA():
    docs = [_doc("ok")] * 4 + [_doc("parcial")] * 2  # 33.33%

    t = _totales(docs)

    assert t["porcentaje_parcial"] == pytest.approx(33.33, abs=0.01)
    assert t["b4_parcial_bajo_umbral_pass"] is True


def test_totales_porcentaje_parcial_sobre_60_NO_PASA():
    docs = [_doc("ok")] * 2 + [_doc("parcial")] * 5  # 71.43%

    t = _totales(docs)

    assert t["porcentaje_parcial"] == pytest.approx(71.43, abs=0.01)
    assert t["b4_parcial_bajo_umbral_pass"] is False


# ---------------------------------------------------------------------------
# Veredicto agregado, de punta a punta -- lo que atrapa un medidor que
# "siempre sale aprobado".
# ---------------------------------------------------------------------------


def test_medir_end_to_end_arma_veredicto_y_puede_dar_NO_GO(tmp_path: Path, monkeypatch):
    """Fuerza `MAX_READ_BYTES` por debajo de cualquier extracto real --
    ningún documento aplicable a B.2 puede pasar, y el veredicto agregado
    tiene que reflejarlo. Prueba directa contra la mutación "el medidor
    siempre sale aprobado": si `go` se hardcodea en `True`, esto lo nota."""
    monkeypatch.setattr(tool_authority, "MAX_READ_BYTES", 1)

    archivo = _libro(tmp_path / "e.xlsx")
    trabajo = tmp_path / "trabajo"

    r = medir([archivo], trabajo)

    assert r["totales"]["documentos_donde_aplica_b2"] == 1
    assert r["totales"]["b2_extracto_util_pass"] is False
    assert r["veredicto"]["b2_utilidad_binaria"] is False
    assert r["go"] is False
