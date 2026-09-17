"""Semáforo de dos carriles para el acceso a Ollama (Fase 2, §3.4).

ENTRE PROCESOS a propósito: la Mesa (`jax-platform`) y el Ejecutor (el arnés
del usuario `axioma`) son procesos distintos. Por eso estos tests usan
`multiprocessing` y no hilos: un lock por proceso no coordinaría nada y unos
tests con hilos darían verde sin probar lo único que importa.
"""
import fcntl
import multiprocessing as mp
import time

import pytest

from jax.ejecutor.cita import Motivo
from jax.ejecutor.prioridad import (
    ESPERA_AGOTADA, EsperaAgotada, carril_ejecutor, carril_mesa, hay_mesa_esperando,
)

# `fork` explícito: los objetivos son funciones anidadas en los tests y no se
# pueden picklear. En Linux/3.12 es el predeterminado, pero dejarlo escrito
# evita que un cambio del predeterminado convierta estos tests en errores
# de pickle que se leerían como "el semáforo no anda".
_CTX = mp.get_context("fork")


def _rematar(p):
    """Ningún proceso huérfano, pase lo que pase en el test."""
    p.join(5)
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join(5)


def test_la_mesa_entra_aunque_el_ejecutor_este_trabajando(tmp_path):
    with carril_ejecutor(tmp_path, tope_s=1):
        with carril_mesa(tmp_path):
            pass  # si esto bloquea, el diseño está mal: la Mesa entra SIEMPRE


def test_el_ejecutor_espera_si_hay_mesa_esperando(tmp_path):
    def mesa(listo, suelte):
        with carril_mesa(tmp_path):
            listo.set()
            suelte.wait(5)
    listo, suelte = _CTX.Event(), _CTX.Event()
    p = _CTX.Process(target=mesa, args=(listo, suelte)); p.start()
    try:
        assert listo.wait(5) is True
        assert hay_mesa_esperando(tmp_path) is True
        with pytest.raises(EsperaAgotada):
            with carril_ejecutor(tmp_path, tope_s=0.5):
                pass
    finally:
        suelte.set(); _rematar(p)


def test_sin_mesa_el_ejecutor_entra_enseguida(tmp_path):
    t0 = time.monotonic()
    with carril_ejecutor(tmp_path, tope_s=5):
        pass
    assert time.monotonic() - t0 < 1


def test_el_tope_vencido_FALLA_y_no_se_cuela(tmp_path):
    """Si colarse fuera una opción, la prioridad no existiría (§5 del spec)."""
    def mesa(listo, suelte):
        with carril_mesa(tmp_path):
            listo.set(); suelte.wait(5)
    listo, suelte = _CTX.Event(), _CTX.Event()
    p = _CTX.Process(target=mesa, args=(listo, suelte)); p.start()
    try:
        assert listo.wait(5) is True
        with pytest.raises(EsperaAgotada) as agotada:
            with carril_ejecutor(tmp_path, tope_s=0.2):
                pytest.fail("se coló: el carril no debió concederse")
        # Sin prosa: el argumento es un Motivo con el tope.
        assert agotada.value.args == (Motivo(ESPERA_AGOTADA, (("tope_s", 0.2),)),)
    finally:
        suelte.set(); _rematar(p)


def test_el_carril_se_suelta_aunque_el_cuerpo_lance(tmp_path):
    with pytest.raises(ValueError):
        with carril_ejecutor(tmp_path, tope_s=1):
            raise ValueError("boom")
    with carril_ejecutor(tmp_path, tope_s=1):
        pass  # si el anterior no soltó, esto se cuelga


def test_sin_mesa_no_hay_mesa_esperando(tmp_path):
    """El sondeo no puede dar un falso positivo con todo quieto: si diera
    siempre True, el Ejecutor no entraría nunca y los demás tests del tope
    pasarían por la razón equivocada."""
    assert hay_mesa_esperando(tmp_path) is False


def test_el_sondeo_no_le_roba_el_carril_a_la_mesa(tmp_path):
    """`hay_mesa_esperando` sondea y suelta: dos sondeos seguidos no pueden
    dejar `mesa.lock` tomado y bloquear a la Mesa de verdad."""
    assert hay_mesa_esperando(tmp_path) is False
    assert hay_mesa_esperando(tmp_path) is False
    with carril_mesa(tmp_path):
        pass


def test_dos_ejecutores_no_entran_a_la_vez(tmp_path):
    """El carril del Ejecutor es exclusivo ENTRE PROCESOS: si no lo fuera,
    dos misiones pisarían la GPU a la vez y el reparto no existiría."""
    dentro, suelte = _CTX.Event(), _CTX.Event()

    def primero():
        with carril_ejecutor(tmp_path, tope_s=5):
            dentro.set()
            suelte.wait(5)

    p = _CTX.Process(target=primero); p.start()
    try:
        assert dentro.wait(5) is True
        with open(tmp_path / "ejecutor.lock", "r+") as f:
            with pytest.raises(OSError):
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        suelte.set(); _rematar(p)


def test_la_mesa_espera_a_otra_mesa(tmp_path):
    """Dos peticiones de persona sí se serializan entre sí: `carril_mesa` es
    exclusivo. 'La Mesa entra siempre' es frente al Ejecutor, no frente a
    otra Mesa."""
    dentro, suelte = _CTX.Event(), _CTX.Event()

    def mesa():
        with carril_mesa(tmp_path):
            dentro.set()
            suelte.wait(5)

    p = _CTX.Process(target=mesa); p.start()
    try:
        assert dentro.wait(5) is True
        with open(tmp_path / "mesa.lock", "r+") as f:
            with pytest.raises(OSError):
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        suelte.set(); _rematar(p)


def test_el_carril_del_ejecutor_se_suelta_al_salir(tmp_path):
    """Soltar entre pasos es el mecanismo de §3.4: al salir del `with`, otro
    proceso tiene que poder tomarlo."""
    with carril_ejecutor(tmp_path, tope_s=1):
        pass

    def segundo(ok):
        with carril_ejecutor(tmp_path, tope_s=2):
            ok.set()

    ok = _CTX.Event()
    p = _CTX.Process(target=segundo, args=(ok,)); p.start()
    try:
        assert ok.wait(5) is True
    finally:
        _rematar(p)
