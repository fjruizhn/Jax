"""Semáforo de dos carriles para el acceso a Ollama (Fase 2, §3.4).

ENTRE PROCESOS a propósito: la Mesa (`jax-platform`) y el Ejecutor (el arnés
del usuario `axioma`) son procesos distintos. Por eso estos tests usan
`multiprocessing` y no hilos: un lock por proceso no coordinaría nada y unos
tests con hilos darían verde sin probar lo único que importa.
"""
import asyncio
import fcntl
import multiprocessing as mp
import threading
import time

import pytest

from jax.ejecutor.cita import Motivo
from jax.ejecutor.prioridad import (
    ESPERA_AGOTADA, EsperaAgotada, carril_ejecutor, carril_ejecutor_async, carril_mesa,
    carril_mesa_async, hay_mesa_esperando,
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


# ---------------------------------------------------------------------------
# Carriles ASYNC (§3.4 bis). El worker de la Mesa y el proxy del Ejecutor
# corren dentro de un event loop: un `flock` bloqueante ahí congela a todos.
# ---------------------------------------------------------------------------
def _mesa_retiene(raiz, listo, suelte, hasta_s):
    """Objetivo de proceso: toma el carril de Mesa y lo retiene hasta que le
    avisen o venza `hasta_s` (así un test roto no deja un proceso colgado)."""
    with carril_mesa(raiz):
        listo.set()
        suelte.wait(hasta_s)


def _lock_libre(ruta) -> bool:
    # "a": si nadie llegó a crear el fichero, el lock está libre por definición.
    with open(ruta, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(f, fcntl.LOCK_UN)
        return True


def test_async_mesa_espera_sin_congelar_el_event_loop(tmp_path):
    """LO QUE IMPORTA de la versión async: mientras otro proceso retiene el
    carril de Mesa, el loop sigue atendiendo. Un contador avanza cada 10 ms;
    si el flock se tomara en el loop, no habría ticks hasta entrar."""
    listo, suelte = _CTX.Event(), _CTX.Event()
    # El retenedor suelta SOLO al segundo: si el loop se congela, nadie podría
    # avisarle y el test se colgaría en vez de fallar.
    p = _CTX.Process(target=_mesa_retiene, args=(tmp_path, listo, suelte, 1.0))
    p.start()
    try:
        assert listo.wait(5) is True

        async def escenario():
            ticks = []

            async def contador():
                while True:
                    ticks.append(time.monotonic())
                    await asyncio.sleep(0.01)

            c = asyncio.create_task(contador())
            await asyncio.sleep(0)
            t0 = time.monotonic()
            async with carril_mesa_async(tmp_path):
                t_entrada = time.monotonic()
            c.cancel()
            await asyncio.wait({c})
            return t0, t_entrada, ticks

        t0, t_entrada, ticks = asyncio.run(asyncio.wait_for(escenario(), 10))
        assert t_entrada - t0 >= 0.5, "entró sin esperar a la otra Mesa"
        durante = [t for t in ticks if t0 < t < t_entrada - 0.1]
        assert len(durante) >= 20, f"el loop se congeló: {len(durante)} ticks"
    finally:
        suelte.set(); _rematar(p)


def test_async_ejecutor_sondea_sin_congelar_el_event_loop(tmp_path):
    listo, suelte = _CTX.Event(), _CTX.Event()
    p = _CTX.Process(target=_mesa_retiene, args=(tmp_path, listo, suelte, 5))
    p.start()
    try:
        assert listo.wait(5) is True

        async def escenario():
            ticks = []

            async def contador():
                while True:
                    ticks.append(time.monotonic())
                    await asyncio.sleep(0.01)

            c = asyncio.create_task(contador())
            await asyncio.sleep(0)
            t0 = time.monotonic()
            with pytest.raises(EsperaAgotada):
                async with carril_ejecutor_async(tmp_path, tope_s=0.6):
                    pytest.fail("se coló")
            t1 = time.monotonic()
            c.cancel()
            await asyncio.wait({c})
            return [t for t in ticks if t0 < t < t1]

        durante = asyncio.run(asyncio.wait_for(escenario(), 10))
        assert len(durante) >= 20, f"el loop se congeló: {len(durante)} ticks"
    finally:
        suelte.set(); _rematar(p)


def test_async_la_mesa_entra_aunque_el_ejecutor_este_trabajando(tmp_path):
    async def escenario():
        async with carril_ejecutor_async(tmp_path, tope_s=1):
            async with carril_mesa_async(tmp_path):
                return True
    assert asyncio.run(asyncio.wait_for(escenario(), 5)) is True


def test_async_tope_vencido_FALLA_con_motivo_y_no_se_cuela(tmp_path):
    listo, suelte = _CTX.Event(), _CTX.Event()
    p = _CTX.Process(target=_mesa_retiene, args=(tmp_path, listo, suelte, 5))
    p.start()
    try:
        assert listo.wait(5) is True

        async def escenario():
            with pytest.raises(EsperaAgotada) as agotada:
                async with carril_ejecutor_async(tmp_path, tope_s=0.2):
                    pytest.fail("se coló: el carril no debió concederse")
            return agotada.value.args

        args = asyncio.run(asyncio.wait_for(escenario(), 5))
        assert args == (Motivo(ESPERA_AGOTADA, (("tope_s", 0.2),)),)
    finally:
        suelte.set(); _rematar(p)


def test_async_ejecutor_entra_cuando_la_mesa_suelta(tmp_path):
    listo, suelte = _CTX.Event(), _CTX.Event()
    p = _CTX.Process(target=_mesa_retiene, args=(tmp_path, listo, suelte, 0.4))
    p.start()
    try:
        assert listo.wait(5) is True

        async def escenario():
            t0 = time.monotonic()
            async with carril_ejecutor_async(tmp_path, tope_s=5):
                return time.monotonic() - t0

        espera = asyncio.run(asyncio.wait_for(escenario(), 10))
        assert espera >= 0.2
    finally:
        suelte.set(); _rematar(p)


def test_async_dos_ejecutores_no_entran_a_la_vez_y_el_tope_cubre_esa_espera(tmp_path):
    """El segundo Ejecutor espera al primero DENTRO del tope: con el proxy,
    Claude Code manda peticiones en paralelo, y una espera sin tope sobre
    `ejecutor.lock` colgaría la misión en vez de fallarla."""
    dentro, suelte = _CTX.Event(), _CTX.Event()

    def primero():
        with carril_ejecutor(tmp_path, tope_s=5):
            dentro.set(); suelte.wait(5)

    p = _CTX.Process(target=primero); p.start()
    try:
        assert dentro.wait(5) is True

        async def escenario():
            with pytest.raises(EsperaAgotada):
                async with carril_ejecutor_async(tmp_path, tope_s=0.3):
                    pytest.fail("dos Ejecutores a la vez")

        asyncio.run(asyncio.wait_for(escenario(), 5))
    finally:
        suelte.set(); _rematar(p)


def test_sync_el_tope_cubre_la_espera_por_otro_ejecutor(tmp_path):
    """Misma semántica en la versión sync. Antes esperaba `ejecutor.lock` con
    un LOCK_EX bloqueante SIN tope: el tope sólo cubría a la Mesa."""
    dentro, suelte = _CTX.Event(), _CTX.Event()

    def primero():
        with carril_ejecutor(tmp_path, tope_s=5):
            dentro.set(); suelte.wait(5)

    p = _CTX.Process(target=primero); p.start()
    resultado = []

    def segundo():
        try:
            with carril_ejecutor(tmp_path, tope_s=0.3):
                resultado.append("entro")
        except EsperaAgotada:
            resultado.append("agotada")

    try:
        assert dentro.wait(5) is True
        h = threading.Thread(target=segundo, daemon=True); h.start()
        h.join(3)
        assert resultado == ["agotada"], f"colgado o colado: {resultado}"
    finally:
        suelte.set(); _rematar(p)


def test_async_cancelar_mientras_la_mesa_espera_suelta_el_lock(tmp_path):
    """Cancelación REAL de la tarea mientras espera: el hilo que sondea no
    puede quedarse con el lock cuando la otra Mesa suelte."""
    listo, suelte = _CTX.Event(), _CTX.Event()
    p = _CTX.Process(target=_mesa_retiene, args=(tmp_path, listo, suelte, 5))
    p.start()
    try:
        assert listo.wait(5) is True

        async def escenario():
            async def pide():
                async with carril_mesa_async(tmp_path):
                    await asyncio.sleep(60)
            t = asyncio.create_task(pide())
            await asyncio.sleep(0.2)
            assert not t.done()
            t.cancel()
            await asyncio.wait({t})
            assert t.cancelled()
            suelte.set()
            await asyncio.to_thread(_rematar, p)
            # Tiempo de sobra para que un hilo huérfano tomara el lock.
            await asyncio.sleep(0.3)
            return await asyncio.to_thread(_lock_libre, tmp_path / "mesa.lock")

        assert asyncio.run(asyncio.wait_for(escenario(), 10)) is True
    finally:
        suelte.set(); _rematar(p)


def test_async_cancelar_dentro_del_carril_de_mesa_lo_suelta(tmp_path):
    async def escenario():
        dentro = asyncio.Event()

        async def pide():
            async with carril_mesa_async(tmp_path):
                dentro.set()
                await asyncio.sleep(60)
        t = asyncio.create_task(pide())
        await asyncio.wait_for(dentro.wait(), 5)
        assert await asyncio.to_thread(_lock_libre, tmp_path / "mesa.lock") is False
        t.cancel()
        await asyncio.wait({t})
        return await asyncio.to_thread(_lock_libre, tmp_path / "mesa.lock")

    assert asyncio.run(asyncio.wait_for(escenario(), 10)) is True


def test_async_cancelar_mientras_el_ejecutor_espera_no_deja_lock(tmp_path):
    listo, suelte = _CTX.Event(), _CTX.Event()
    p = _CTX.Process(target=_mesa_retiene, args=(tmp_path, listo, suelte, 5))
    p.start()
    try:
        assert listo.wait(5) is True

        async def escenario():
            async def pide():
                async with carril_ejecutor_async(tmp_path, tope_s=30):
                    await asyncio.sleep(60)
            t = asyncio.create_task(pide())
            await asyncio.sleep(0.2)
            assert not t.done()
            t.cancel()
            await asyncio.wait({t})
            assert t.cancelled()
            suelte.set()
            await asyncio.to_thread(_rematar, p)
            await asyncio.sleep(0.3)
            return await asyncio.to_thread(_lock_libre, tmp_path / "ejecutor.lock")

        assert asyncio.run(asyncio.wait_for(escenario(), 10)) is True
    finally:
        suelte.set(); _rematar(p)


def test_async_cancelar_dentro_del_carril_del_ejecutor_lo_suelta(tmp_path):
    async def escenario():
        dentro = asyncio.Event()

        async def pide():
            async with carril_ejecutor_async(tmp_path, tope_s=1):
                dentro.set()
                await asyncio.sleep(60)
        t = asyncio.create_task(pide())
        await asyncio.wait_for(dentro.wait(), 5)
        assert await asyncio.to_thread(_lock_libre, tmp_path / "ejecutor.lock") is False
        t.cancel()
        await asyncio.wait({t})
        return await asyncio.to_thread(_lock_libre, tmp_path / "ejecutor.lock")

    assert asyncio.run(asyncio.wait_for(escenario(), 10)) is True


def test_async_el_carril_se_suelta_aunque_el_cuerpo_lance(tmp_path):
    async def escenario():
        with pytest.raises(ValueError):
            async with carril_mesa_async(tmp_path):
                raise ValueError("boom")
        with pytest.raises(ValueError):
            async with carril_ejecutor_async(tmp_path, tope_s=1):
                raise ValueError("boom")
        return (await asyncio.to_thread(_lock_libre, tmp_path / "mesa.lock"),
                await asyncio.to_thread(_lock_libre, tmp_path / "ejecutor.lock"))

    assert asyncio.run(asyncio.wait_for(escenario(), 5)) == (True, True)
