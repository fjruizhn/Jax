# tests/test_ejecutor_contratos_pausa.py
"""La pausa PROPIA del Ejecutor (C5 la pone; el proxy la obedece). No es el interruptor
global de JAX: un falso positivo del auditor frena al Ejecutor, no a la Mesa.
Fail-closed: sin saber dónde está, o sin poder mirarla, se la da por PUESTA."""
import json
import os

import pytest

from jax.ejecutor.contratos import pausa as P


def test_la_ruta_sale_del_entorno_y_tiene_que_ser_absoluta(tmp_path):
    assert P.ruta_de_la_pausa({P.VARIABLE_RUTA: str(tmp_path / "PAUSA")}) == tmp_path / "PAUSA"
    for env in ({}, {P.VARIABLE_RUTA: "  "}, {P.VARIABLE_RUTA: "relativa/PAUSA"}):
        with pytest.raises(P.PausaSinConfigurar):
            P.ruta_de_la_pausa(env)


def test_poner_una_vez_y_leer(tmp_path):
    ruta = tmp_path / "PAUSA"
    assert P.pausa_puesta(ruta) is False
    assert P.poner_pausa(ruta, {"origen": "c5", "motivo": "fuera_de_mision", "paso": 3}) is True
    assert P.pausa_puesta(ruta) is True
    doc = json.loads(ruta.read_text())
    assert (doc["origen"], doc["motivo"], doc["paso"]) == ("c5", "fuera_de_mision", 3) and doc["momento"]
    # La segunda no pisa la primera: el motivo que quedó es el del primer freno.
    assert P.poner_pausa(ruta, {"origen": "c5", "motivo": "auditor_caido", "paso": None}) is False
    assert json.loads(ruta.read_text())["motivo"] == "fuera_de_mision"
    assert [p.name for p in tmp_path.iterdir()] == ["PAUSA"], "no quedan temporales"


def test_sin_poder_mirarla_esta_puesta(tmp_path):
    archivo = tmp_path / "no-es-directorio"
    archivo.write_text("x")
    assert P.pausa_puesta(archivo / "PAUSA") is True  # ENOTDIR


@pytest.mark.skipif(os.geteuid() == 0, reason="root lee todo")
def test_directorio_ilegible_esta_puesta(tmp_path):
    cerrado = tmp_path / "cerrado"
    cerrado.mkdir()
    cerrado.chmod(0)
    try:
        assert P.pausa_puesta(cerrado / "PAUSA") is True
    finally:
        cerrado.chmod(0o700)


def test_latido_fresco_viejo_o_ausente(tmp_path):
    ruta = tmp_path / "latido"
    assert P.latido_fresco(ruta, 5, ahora=1000.0) is False
    P.latir(ruta)
    mtime = ruta.stat().st_mtime
    assert P.latido_fresco(ruta, 5, ahora=mtime + 4) is True
    assert P.latido_fresco(ruta, 5, ahora=mtime + 6) is False
    assert P.latido_fresco(tmp_path / "x" / "latido", 5) is False
