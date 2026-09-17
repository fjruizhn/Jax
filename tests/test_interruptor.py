"""El interruptor de JAX: dónde está el freno y si está puesto
(plan 2026-09-16-frente-b-kill-switch, Task 1).

Antes, cada lector tenía su ruta escrita a mano y miraba con `Path.exists()`,
que devuelve False ante un PermissionError: con el archivo puesto y el
directorio ilegible, el freno se leía SUELTO (medido, Python 3.14.4).

Corre con:
  PYTHONPATH=.:las_manos python -m pytest tests/test_interruptor.py -v

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path

import pytest

from jax.core import interruptor

VARIABLE = "JAX_KILL_SWITCH_PATH"
ES_ROOT = os.geteuid() == 0


def test_sin_variable_no_hay_ruta(monkeypatch):
    monkeypatch.delenv(VARIABLE, raising=False)
    with pytest.raises(interruptor.InterruptorSinConfigurar, match=VARIABLE):
        interruptor.ruta_del_interruptor()


def test_variable_vacia_es_lo_mismo_que_ausente(monkeypatch):
    monkeypatch.setenv(VARIABLE, "   ")
    with pytest.raises(interruptor.InterruptorSinConfigurar):
        interruptor.ruta_del_interruptor()


def test_ruta_relativa_se_rechaza(monkeypatch):
    monkeypatch.setenv(VARIABLE, "PAUSE")
    with pytest.raises(interruptor.InterruptorSinConfigurar, match="absoluta"):
        interruptor.ruta_del_interruptor()


def test_sin_variable_mirar_el_freno_tambien_lanza(monkeypatch):
    monkeypatch.delenv(VARIABLE, raising=False)
    with pytest.raises(interruptor.InterruptorSinConfigurar):
        interruptor.interruptor_activo()


def test_la_ruta_se_lee_en_cada_llamada(monkeypatch, tmp_path):
    monkeypatch.setenv(VARIABLE, str(tmp_path / "a" / "PAUSE"))
    assert interruptor.ruta_del_interruptor() == tmp_path / "a" / "PAUSE"
    monkeypatch.setenv(VARIABLE, str(tmp_path / "b" / "PAUSE"))
    assert interruptor.ruta_del_interruptor() == tmp_path / "b" / "PAUSE"


def test_suelto_si_el_archivo_no_existe(monkeypatch, tmp_path):
    monkeypatch.setenv(VARIABLE, str(tmp_path / "PAUSE"))
    assert interruptor.interruptor_activo() is False


def test_puesto_si_el_archivo_existe(monkeypatch, tmp_path):
    (tmp_path / "PAUSE").write_text("")
    monkeypatch.setenv(VARIABLE, str(tmp_path / "PAUSE"))
    assert interruptor.interruptor_activo() is True


@pytest.mark.skipif(ES_ROOT, reason="root atraviesa cualquier permiso")
def test_directorio_ilegible_cuenta_como_puesto(monkeypatch, tmp_path):
    carpeta = tmp_path / "interruptor"
    carpeta.mkdir()
    (carpeta / "PAUSE").write_text("")
    monkeypatch.setenv(VARIABLE, str(carpeta / "PAUSE"))
    carpeta.chmod(0)
    try:
        # El defecto que se cierra: pathlib lo lee SUELTO.
        assert (carpeta / "PAUSE").exists() is False
        assert interruptor.interruptor_activo() is True
    finally:
        carpeta.chmod(0o700)


def test_un_componente_que_no_es_directorio_cuenta_como_puesto(monkeypatch, tmp_path):
    archivo = tmp_path / "archivo"
    archivo.write_text("")
    monkeypatch.setenv(VARIABLE, str(archivo / "PAUSE"))
    assert interruptor.interruptor_activo() is True


def test_escribir_pausa_publica_contenido_completo_y_no_pisa(tmp_path):
    ruta = tmp_path / "PAUSE"
    assert interruptor.escribir_pausa(ruta, '{"user_id": "7"}') is True
    assert ruta.read_text() == '{"user_id": "7"}'
    assert stat.S_IMODE(ruta.stat().st_mode) == 0o660
    assert interruptor.escribir_pausa(ruta, '{"user_id": "8"}') is False
    assert ruta.read_text() == '{"user_id": "7"}'
    assert [p.name for p in tmp_path.iterdir()] == ["PAUSE"]  # sin temporales


def test_borrar_pausa_es_idempotente(tmp_path):
    ruta = tmp_path / "PAUSE"
    interruptor.escribir_pausa(ruta, "{}")
    assert interruptor.borrar_pausa(ruta) is True
    assert not ruta.exists()
    assert interruptor.borrar_pausa(ruta) is False


def test_escribir_en_un_directorio_que_no_existe_lanza(tmp_path):
    with pytest.raises(FileNotFoundError):
        interruptor.escribir_pausa(tmp_path / "no-existe" / "PAUSE", "{}")


def test_correr_devuelve_el_resultado(monkeypatch, tmp_path):
    monkeypatch.setenv(VARIABLE, str(tmp_path / "PAUSE"))

    async def rapida():
        return 42

    assert asyncio.run(interruptor.correr_con_interruptor(rapida(), intervalo=0.01)) == 42


def test_correr_con_el_freno_puesto_no_arranca(monkeypatch, tmp_path):
    (tmp_path / "PAUSE").write_text("")
    monkeypatch.setenv(VARIABLE, str(tmp_path / "PAUSE"))
    arranco = []

    async def tarea():
        arranco.append(True)

    with pytest.raises(interruptor.InterruptorActivado, match="killed_by_switch"):
        asyncio.run(interruptor.correr_con_interruptor(tarea(), intervalo=0.01))
    assert arranco == []


def test_correr_cancela_lo_que_corre_cuando_aparece_el_freno(monkeypatch, tmp_path):
    ruta = tmp_path / "PAUSE"
    monkeypatch.setenv(VARIABLE, str(ruta))
    limpieza = []

    async def larga():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            # Lo que hace hyde_sandbox.run_sandboxed_claude: proc.kill().
            limpieza.append("cancelada")
            raise

    async def escenario():
        loop = asyncio.get_running_loop()

        async def poner():
            await asyncio.sleep(0.05)
            interruptor.escribir_pausa(ruta, "{}")

        loop.create_task(poner())
        inicio = loop.time()
        try:
            await interruptor.correr_con_interruptor(larga(), intervalo=0.01)
        except interruptor.InterruptorActivado as exc:
            return str(exc), loop.time() - inicio
        return None, loop.time() - inicio

    mensaje, duracion = asyncio.run(escenario())
    assert mensaje is not None and "killed_by_switch" in mensaje
    assert limpieza == ["cancelada"]
    assert duracion < 1.0


def test_el_conftest_aisla_la_ruta_de_produccion():
    assert not str(interruptor.ruta_del_interruptor()).startswith("/etc/jax")
