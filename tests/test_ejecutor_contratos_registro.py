# tests/test_ejecutor_contratos_registro.py
"""Registro intocable del Ejecutor (C3): cadena de sha256, O_APPEND, fsync.
Cada forma de tocarlo por detrás se detecta; un registro que no cuadra no se abre."""
import fcntl
import json
import os
import threading

import pytest

from jax.ejecutor.contratos import registro as R


def _lineas(ruta):
    return ruta.read_bytes().splitlines()


def test_anota_y_la_cadena_cuadra(tmp_path):
    ruta = tmp_path / "registro.jsonl"
    reg = R.Registro(ruta)
    assert [reg.anotar({"evento": "x", "i": i}) for i in range(3)] == [1, 2, 3]
    reg.cerrar()
    assert R.verificar_cadena(ruta) == R.Verificacion(True, 3, None, None)
    primera = json.loads(_lineas(ruta)[0])
    assert primera["prev"] == R.GENESIS and primera["n"] == 1 and primera["momento"].endswith("+00:00")


def test_reabrir_continua_la_cadena(tmp_path):
    ruta = tmp_path / "registro.jsonl"
    reg = R.Registro(ruta)
    reg.anotar({"evento": "a"})
    reg.cerrar()
    reg = R.Registro(ruta)
    assert reg.anotar({"evento": "b"}) == 2
    reg.cerrar()
    assert R.verificar_cadena(ruta).ok is True


def test_abre_en_modo_append(tmp_path):
    reg = R.Registro(tmp_path / "registro.jsonl")
    try:
        assert fcntl.fcntl(reg._fd, fcntl.F_GETFL) & os.O_APPEND
    finally:
        reg.cerrar()


def _tres(tmp_path):
    ruta = tmp_path / "registro.jsonl"
    reg = R.Registro(ruta)
    for i in range(3):
        reg.anotar({"evento": "x", "comando": f"cmd-{i}"})
    reg.cerrar()
    return ruta


def test_editar_una_linea_se_detecta_en_la_siguiente(tmp_path):
    ruta = _tres(tmp_path)
    lineas = _lineas(ruta)
    lineas[1] = lineas[1].replace(b"cmd-1", b"cmd-X")
    ruta.write_bytes(b"\n".join(lineas) + b"\n")
    assert R.verificar_cadena(ruta) == R.Verificacion(False, 3, 3, "prev_no_cuadra")


def test_borrar_una_linea_se_detecta(tmp_path):
    ruta = _tres(tmp_path)
    lineas = _lineas(ruta)
    ruta.write_bytes(lineas[0] + b"\n" + lineas[2] + b"\n")
    assert R.verificar_cadena(ruta) == R.Verificacion(False, 2, 2, "n_no_cuadra")


def test_linea_incompleta_no_abre_y_se_reporta(tmp_path):
    ruta = _tres(tmp_path)
    with open(ruta, "ab") as f:
        f.write(b'{"evento":"a medias"')
    with pytest.raises(R.RegistroCorrupto) as e:
        R.Registro(ruta)
    assert e.value.codigo == "ultima_linea_incompleta"
    assert R.verificar_cadena(ruta) == R.Verificacion(False, 4, 4, "ultima_linea_incompleta")


def test_cola_ilegible_no_abre(tmp_path):
    ruta = tmp_path / "registro.jsonl"
    ruta.write_bytes(b"no es json\n")
    with pytest.raises(R.RegistroCorrupto) as e:
        R.Registro(ruta)
    assert e.value.codigo == "ultima_linea_ilegible"


def test_escritores_concurrentes_no_rompen_la_cadena(tmp_path):
    ruta = tmp_path / "registro.jsonl"
    reg = R.Registro(ruta)
    hilos = [threading.Thread(target=lambda: [reg.anotar({"evento": "x"}) for _ in range(200)]) for _ in range(4)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join(30)
    reg.cerrar()
    assert R.verificar_cadena(ruta) == R.Verificacion(True, 800, None, None)


def test_escritura_parcial_deja_el_registro_roto_y_lo_dice(tmp_path, monkeypatch):
    reg = R.Registro(tmp_path / "registro.jsonl")
    monkeypatch.setattr(R.os, "write", lambda fd, datos: len(datos) - 1)
    with pytest.raises(OSError):
        reg.anotar({"evento": "x"})
    monkeypatch.undo()
    with pytest.raises(OSError):
        reg.anotar({"evento": "y"})
    reg.cerrar()


def test_un_evento_no_puede_pisar_la_cadena(tmp_path):
    reg = R.Registro(tmp_path / "registro.jsonl")
    reg.anotar({"evento": "x", "n": 999, "prev": "falso"})
    reg.cerrar()
    assert R.verificar_cadena(tmp_path / "registro.jsonl").ok is True
