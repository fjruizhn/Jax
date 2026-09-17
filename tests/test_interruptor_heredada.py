"""La ruta vieja del freno sigue frenando (frente B, Task H).

Requisito del controlador principal del frente B (2026-09-17), que prevalece
sobre el plan: gente y scripts pausan creando la ruta vieja. Si con el cambio a
JAX_KILL_SWITCH_PATH esa ruta dejara de leerse, quien pause así creería que
frenó, y no. Mientras exista, TODO lector la trata como freno PUESTO, falla
cerrado si no se puede mirar, y avisa con un WARNING que nombra la ruta nueva.
La regla vive en el módulo interruptor compartido (rulings R11-R15).

Ningún test toca la ruta real: el conftest de la raíz la desvía a un temporal
inexistente en cada objeto módulo, y acá se la apunta a un tmp_path.

Corre con:
  PYTHONPATH=.:las_manos python -m pytest tests/test_interruptor_heredada.py -v

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import errno
import importlib
import importlib.util
import logging
import os
from pathlib import Path

import pytest

import interruptor as interruptor_pelado
from jax.core import interruptor
from jacobs import policy
from motor_registry import worker
from motor_registry.models import JobStatus
from tests.test_interruptor_lectores import _job

VARIABLE = "JAX_KILL_SWITCH_PATH"
LITERAL_VIEJO = "/etc/jax/" + "PAUSE"
ES_ROOT = os.geteuid() == 0
MODULOS = (interruptor, interruptor_pelado)


def _apuntar_heredada(monkeypatch, ruta):
    # raising=False: contra el código viejo el atributo no existe y el test
    # tiene que caer en su assert, no en el monkeypatch.
    for modulo in MODULOS:
        monkeypatch.setattr(modulo, "RUTA_HEREDADA", Path(ruta), raising=False)


@pytest.fixture
def nueva(monkeypatch, tmp_path):
    ruta = tmp_path / "interruptor" / "PAUSE"
    ruta.parent.mkdir()
    monkeypatch.setenv(VARIABLE, str(ruta))
    return ruta


@pytest.fixture
def vieja(monkeypatch, tmp_path):
    ruta = tmp_path / "vieja" / "PAUSE"
    ruta.parent.mkdir()
    _apuntar_heredada(monkeypatch, ruta)
    return ruta


def _avisos(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING]


def test_la_constante_es_la_ruta_que_la_gente_usa():
    # El conftest la desvía en los módulos importados; una carga fresca del
    # archivo muestra el valor real sin tocar el disco.
    spec = importlib.util.spec_from_file_location("interruptor_fresco", interruptor.__file__)
    fresco = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresco)
    assert fresco.RUTA_HEREDADA == Path(LITERAL_VIEJO)


def test_el_conftest_aisla_la_ruta_heredada():
    importlib.import_module("interruptor")
    for modulo in MODULOS:
        assert str(getattr(modulo, "RUTA_HEREDADA", LITERAL_VIEJO)) != LITERAL_VIEJO


def test_solo_la_heredada_frena_con_y_sin_ruta_explicita(nueva, vieja, caplog):
    vieja.write_text("")
    caplog.set_level(logging.WARNING)
    assert interruptor.interruptor_activo() is True
    assert interruptor.interruptor_activo(nueva) is True
    avisos = _avisos(caplog)
    assert len(avisos) == 1
    assert str(vieja) in avisos[0].getMessage()
    assert str(nueva) in avisos[0].getMessage()


def test_pausa_presente_no_mira_la_heredada(nueva, vieja):
    vieja.write_text("")
    assert interruptor.pausa_presente(nueva) is False
    assert interruptor.pausa_presente(vieja) is True


@pytest.mark.skipif(ES_ROOT, reason="root atraviesa cualquier permiso")
def test_heredada_ilegible_cuenta_como_puesto(nueva, vieja, caplog):
    vieja.write_text("")
    caplog.set_level(logging.WARNING)
    vieja.parent.chmod(0)
    try:
        assert interruptor.interruptor_activo() is True
    finally:
        vieja.parent.chmod(0o700)
    assert len(_avisos(caplog)) == 1


@pytest.mark.parametrize("error", [
    PermissionError(errno.EACCES, "sin permiso"),
    OSError(errno.EIO, "error de E/S"),
])
def test_heredada_con_stat_roto_cuenta_como_puesto(nueva, vieja, monkeypatch, caplog, error):
    """Sin depender de la versión de Python ni de ser root: sólo la heredada
    no se puede mirar, la nueva está ausente."""
    stat_real = os.stat

    def stat_selectivo(ruta, *args, **kwargs):
        if Path(ruta) == vieja:
            raise error
        return stat_real(ruta, *args, **kwargs)

    monkeypatch.setattr(interruptor.os, "stat", stat_selectivo)
    caplog.set_level(logging.WARNING)
    assert interruptor.interruptor_activo() is True
    assert len(_avisos(caplog)) == 1


def test_sin_ninguna_de_las_dos_esta_suelto_y_callado(nueva, vieja, caplog):
    caplog.set_level(logging.WARNING)
    assert interruptor.interruptor_activo() is False
    assert interruptor.interruptor_activo(nueva) is False
    assert _avisos(caplog) == []


def test_un_aviso_por_episodio(nueva, vieja, caplog):
    caplog.set_level(logging.WARNING)
    vieja.write_text("")
    assert interruptor.interruptor_activo() is True
    assert interruptor.interruptor_activo() is True
    assert len(_avisos(caplog)) == 1
    vieja.unlink()
    assert interruptor.interruptor_activo() is False
    assert len(_avisos(caplog)) == 1
    vieja.write_text("")
    assert interruptor.interruptor_activo() is True
    assert len(_avisos(caplog)) == 2


def test_sin_variable_sigue_lanzando_aunque_la_heredada_exista(monkeypatch, vieja):
    vieja.write_text("")
    monkeypatch.delenv(VARIABLE, raising=False)
    with pytest.raises(interruptor.InterruptorSinConfigurar):
        interruptor.interruptor_activo()


def test_correr_con_interruptor_corta_cuando_aparece_la_heredada(nueva, vieja):
    async def larga():
        await asyncio.sleep(30)

    async def escenario():
        loop = asyncio.get_running_loop()

        async def poner():
            await asyncio.sleep(0.05)
            vieja.write_text("")

        loop.create_task(poner())
        with pytest.raises(interruptor.InterruptorActivado, match="killed_by_switch"):
            await interruptor.correr_con_interruptor(larga(), intervalo=0.01)
        return loop.time()

    asyncio.run(escenario())
    assert not nueva.exists()


def test_jacobs_policy_frena_con_solo_la_heredada(nueva, vieja):
    assert policy.check_kill_switch() is False
    vieja.write_text("")
    assert policy.check_kill_switch() is True


def test_las_manos_no_llama_al_modelo_con_solo_la_heredada(tmp_path, nueva, vieja):
    vieja.write_text("")
    llamadas = []

    async def transporte(**kwargs):
        llamadas.append(kwargs)
        return {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "usage": {}}

    estado = asyncio.run(_job(tmp_path, nueva, transporte))
    assert estado["status"] == JobStatus.FAILED.value
    assert "killed_by_switch" in estado["error"]
    assert llamadas == []


def test_las_manos_mata_el_motor_en_vuelo_cuando_aparece_la_heredada(tmp_path, monkeypatch, nueva, vieja):
    monkeypatch.setattr(worker, "_KILL_SWITCH_INTERVAL", 0.05)
    cancelado = []

    async def escenario():
        en_vuelo = asyncio.Event()

        async def transporte(**kwargs):
            en_vuelo.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelado.append(True)
                raise

        async def durante(tarea):
            await asyncio.wait_for(en_vuelo.wait(), 2.0)
            vieja.write_text("")

        return await _job(tmp_path, nueva, transporte, durante)

    estado = asyncio.run(escenario())
    assert estado["status"] == JobStatus.FAILED.value
    assert "killed_by_switch" in estado["error"]
    assert cancelado == [True]


# --- Barrera de sesión sobre el freno de producción (revisión final del
# frente B, 2026-09-17). Hasta ese día el conftest de la raíz no vigilaba el
# freno: un test que creara o borrara el directorio del interruptor o la ruta
# heredada de producción pasaba en verde. Sólo tmp_path: la barrera real nunca
# se ejercita contra /etc.

def test_la_barrera_vigila_el_freno_real():
    import conftest
    assert conftest.HEREDADA_DE_PRODUCCION == Path(LITERAL_VIEJO)
    assert conftest.FRENO_DE_PRODUCCION == Path("/etc/jax/interruptor")


def _al_pasado(*rutas):
    import time
    viejo = time.time() - 100
    for r in rutas:
        os.utime(r, (viejo, viejo))


def test_barrera_sin_nada_en_disco_son_cero_cambios(tmp_path):
    import conftest
    import time
    assert conftest.cambios_del_freno_de_produccion(
        tmp_path / "no-dir", tmp_path / "no-PAUSE", time.time() - 1, False) == []


def test_barrera_detecta_la_heredada_que_aparece(tmp_path):
    import conftest
    import time
    inicio = time.time() - 1
    archivo = tmp_path / "PAUSE"
    archivo.write_text("")
    _al_pasado(archivo)  # aunque el mtime mienta: no existía al empezar
    assert conftest.cambios_del_freno_de_produccion(
        tmp_path / "no-dir", archivo, inicio, False) == [str(archivo)]


def test_barrera_detecta_la_heredada_que_cambia_o_desaparece(tmp_path):
    import conftest
    import time
    inicio = time.time() - 1
    archivo = tmp_path / "PAUSE"
    archivo.write_text("")
    _al_pasado(archivo)
    assert conftest.cambios_del_freno_de_produccion(tmp_path / "no-dir", archivo, inicio, True) == []
    os.utime(archivo, (inicio + 10, inicio + 10))
    assert conftest.cambios_del_freno_de_produccion(
        tmp_path / "no-dir", archivo, inicio, True) == [str(archivo)]
    archivo.unlink()
    assert conftest.cambios_del_freno_de_produccion(
        tmp_path / "no-dir", archivo, inicio, True) == [f"{archivo} (desapareció)"]


def test_barrera_detecta_altas_en_el_directorio(tmp_path):
    import conftest
    import time
    directorio = tmp_path / "interruptor"
    directorio.mkdir()
    viejo = directorio / "viejo"
    viejo.write_text("")
    _al_pasado(viejo, directorio)
    inicio = time.time() - 1
    assert conftest.cambios_del_freno_de_produccion(directorio, tmp_path / "no-PAUSE", inicio, False) == []
    (directorio / "PAUSE").write_text("")
    cambios = conftest.cambios_del_freno_de_produccion(directorio, tmp_path / "no-PAUSE", inicio, False)
    assert str(directorio / "PAUSE") in cambios
