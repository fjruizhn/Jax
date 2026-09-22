# tests/test_ejecutor_proxy_freno.py
"""C4 en el proxy: con el interruptor de JAX puesto, o con la pausa del Ejecutor (C5), el
cerebro no responde (423 legible sin tocar el upstream) y un stream en curso se corta en
menos de un segundo, soltando el carril. Sin saber dónde está el interruptor, no arranca."""
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

import httpx
import pytest

from jax.core import interruptor
from jax.ejecutor import proxy_carril
from jax.ejecutor.contratos import pausa as P
from tests.test_ejecutor_proxy_carril import _CUERPO, MODELO_PERMITIDO, Proxy, Upstream, _correr


@pytest.fixture
def freno(tmp_path, monkeypatch):
    ruta = tmp_path / "interruptor" / "PAUSE"
    ruta.parent.mkdir()
    monkeypatch.setenv(interruptor.VARIABLE_RUTA, str(ruta))
    return ruta


def _error(tipo):
    return {"type": "error", "error": {"type": tipo}}


def test_con_el_interruptor_puesto_423_sin_tocar_el_upstream(tmp_path, freno):
    freno.write_text("{}")

    async def escenario():
        async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 2) as px, httpx.AsyncClient() as cli:
            r = await cli.post(px.url + "/v1/messages", content=_CUERPO)
            h = await cli.head(px.url + "/api/hello")
            return r.status_code, r.headers.get("x-should-retry"), r.json(), h.status_code, len(up.recibidas)

    assert _correr(escenario()) == (423, "false", _error(proxy_carril.KILL_SWITCH_ACTIVO), 423, 0)


def _poner_interruptor(px, ruta):
    ruta.write_text("{}")


def _poner_pausa_c5(px, ruta):
    P.poner_pausa(px.cfg.pausa, {"origen": "prueba_c4"})


def _quitar_interruptor(px, ruta):
    ruta.unlink()


def _quitar_pausa_c5(px, ruta):
    px.cfg.pausa.unlink()


@pytest.mark.parametrize("poner, quitar", [(_poner_interruptor, _quitar_interruptor),
                                           (_poner_pausa_c5, _quitar_pausa_c5)],
                         ids=["interruptor_de_jax", "pausa_del_ejecutor"])
def test_el_freno_corta_el_stream_en_vuelo_y_suelta_el_carril(tmp_path, freno, poner, quitar):
    async def escenario():
        async with Upstream(n_trozos=3) as up, Proxy(up.url, tmp_path, 5) as px, httpx.AsyncClient() as cli:
            cortado, demora, inicio = False, None, None
            try:
                async with cli.stream("POST", px.url + "/v1/messages", content=_CUERPO) as r:
                    trozos = r.aiter_raw()
                    await asyncio.wait_for(anext(trozos), 3)
                    inicio = time.monotonic()
                    poner(px, freno)
                    async for _ in trozos:
                        pass
            except httpx.RemoteProtocolError:
                cortado, demora = True, time.monotonic() - inicio
            quitar(px, freno)
            up.avanzar.set()
            up.n_trozos = 1  # la siguiente respuesta no espera a que el test avance
            siguiente = await asyncio.wait_for(cli.post(px.url + "/v1/messages", content=_CUERPO), 4)
            return cortado, demora, siguiente.status_code

    cortado, demora, siguiente = _correr(escenario())
    assert cortado is True
    assert demora < 1.0, f"el corte tardó {demora:.2f} s"
    assert siguiente == 200, "el carril quedó tomado después del freno"


def test_freno_puesto_mientras_espera_el_carril_da_423_legible(tmp_path, freno):
    """Sin cabeceras enviadas todavía, el corte es un 423 que el arnés lee (y no reintenta)."""
    async def escenario():
        async with Upstream(n_trozos=2) as up, Proxy(up.url, tmp_path, 30) as px, httpx.AsyncClient() as cli:
            primera = asyncio.create_task(cli.post(px.url + "/v1/messages", content=_CUERPO))
            while not up.recibidas:
                await asyncio.sleep(0.02)
            # La primera tiene el carril (su upstream espera `avanzar`); la segunda hace cola.
            segunda = asyncio.create_task(cli.post(px.url + "/v1/messages", content=_CUERPO))
            # Esperar el ESTADO, no un rato: con `sleep` el runner lento ponía el freno antes de
            # que la segunda llegara a la cola y el corte salía como RemoteProtocolError (CI
            # 2026-09-17). `esperando_carril()` dice cuántas hay esperando de verdad.
            limite = time.monotonic() + 5
            while proxy_carril.esperando_carril() < 1:
                assert time.monotonic() < limite, "la segunda petición nunca llegó a la cola"
                await asyncio.sleep(0.02)
            # Las dos suposiciones del test, comprobadas EN EL INSTANTE del corte.
            #
            # Falló en CI el 2026-09-20 con `httpx.RemoteProtocolError: peer closed
            # connection without sending complete message body`. Causa raíz encontrada
            # el 2026-09-21 (reproducida 2/150 bajo carga de CPU, y de forma
            # determinística en test_freno_justo_al_tomar_el_carril_no_toca_upstream,
            # más abajo): `_reenviar` no re-chequeaba el freno justo al TOMAR el
            # carril. Si la primera soltaba el carril (cortada por su propio vigía
            # del freno) en la misma ventana en que la segunda esperaba, la segunda
            # podía ganar el carril recién liberado y llegar al upstream antes de que
            # SU PROPIO vigía del freno —que sondea cada INTERVALO_DE_SONDEO, no en
            # cada instrucción— se enterara. Arreglado con un re-chequeo síncrono
            # dentro de `_reenviar` al tomar el carril (`jax/ejecutor/proxy_carril.py`).
            # Estas dos asserts se quedan como red: si la causa vuelve a abrirse por
            # otra vía, el mensaje dice CUÁL suposición se rompió en vez de un error
            # de protocolo indescifrable.
            assert not primera.done(), (
                "la primera soltó el carril antes del freno: la segunda ya estaba "
                "recibiendo cuerpo, y el corte sale como error de protocolo en vez "
                "de como 423. El test no probó lo que dice probar")
            assert len(up.recibidas) == 1, (
                f"el upstream recibió {len(up.recibidas)} peticiones, no 1: la segunda "
                "ya había pasado del carril cuando cayó el freno")

            inicio = time.monotonic()
            freno.write_text("{}")
            r = await asyncio.wait_for(segunda, 3)
            demora = time.monotonic() - inicio
            primera.cancel()
            await asyncio.gather(primera, return_exceptions=True)
            return r.status_code, r.headers.get("x-should-retry"), r.json(), demora, len(up.recibidas)

    estado, reintento, cuerpo, demora, recibidas = _correr(escenario())
    assert (estado, reintento, cuerpo, recibidas) == (423, "false", _error(proxy_carril.KILL_SWITCH_ACTIVO), 1)
    assert demora < 1.0


def test_freno_justo_al_tomar_el_carril_no_toca_upstream(tmp_path, freno, monkeypatch):
    """Causa raíz de la intermitencia de CI 2026-09-20 (ver el comentario del test de
    arriba), reproducida SIN depender de carga ni de suerte: el freno cae en la misma
    fracción de segundo en que una petición en cola obtiene el carril, antes de que
    `_reenviar` construya nada para el upstream. Contra el código de antes del
    2026-09-21 esto llegaba al upstream (recibía `data: trozo-0`); con el re-chequeo
    al tomar el carril, no."""
    real_carril = proxy_carril.carril_ejecutor_async

    @asynccontextmanager
    async def carril_que_frena_al_conceder(raiz, tope_s):
        async with real_carril(raiz, tope_s):
            # El instante exacto de la carrera: el carril se concede y, ANTES de que
            # `_reenviar` llegue a construir la petición al upstream, el freno ya cayó.
            freno.write_text("{}")
            yield

    monkeypatch.setattr(proxy_carril, "carril_ejecutor_async", carril_que_frena_al_conceder)

    async def escenario():
        async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 5) as px, httpx.AsyncClient() as cli:
            r = await cli.post(px.url + "/v1/messages", content=_CUERPO)
            return r.status_code, r.headers.get("x-should-retry"), r.json(), len(up.recibidas)

    assert _correr(escenario()) == (423, "false", _error(proxy_carril.KILL_SWITCH_ACTIVO), 0)


def test_vigia_sin_latido_justo_al_tomar_el_carril_no_toca_upstream(tmp_path):
    """El re-chequeo al tomar el carril (arriba) usa el predicado ANCHO
    (`_frenado_con_tope`, que mira el latido), no el que sondea el vigía de `atender()`
    (que NO lo mira). Sin esto: `tope_s` de producción es 90 s contra
    `JAX_EJECUTOR_VIGIA_LATIDO_MAX_S` de 30 s -hasta 3 veces la edad máxima del latido-, y
    una petición podía esperar el carril, obtenerlo, y llegar al upstream con el auditor
    de C5 ya sin latido: nada en el camino de espera del carril miraba el latido. Contra
    el código de antes de este re-chequeo (y contra `_freno_puesto_ahora` en su lugar) B
    llega al upstream; con `_frenado_con_tope`, no."""
    async def escenario():
        async with Upstream(n_trozos=2) as up, Proxy(up.url, tmp_path, 30) as px, httpx.AsyncClient() as cli:
            primera = asyncio.create_task(cli.post(px.url + "/v1/messages", content=_CUERPO))
            while not up.recibidas:
                await asyncio.sleep(0.02)
            segunda = asyncio.create_task(cli.post(px.url + "/v1/messages", content=_CUERPO))
            limite = time.monotonic() + 5
            while proxy_carril.esperando_carril() < 1:
                assert time.monotonic() < limite, "la segunda petición nunca llegó a la cola"
                await asyncio.sleep(0.02)
            assert not primera.done()
            assert len(up.recibidas) == 1

            # El latido queda viejo (px.cfg.latido_max_s es 3600, fijo en el helper
            # Proxy): nada lo renueva y nadie más lo mira hasta que segunda tome el
            # carril. Mismo patrón que test_sin_latido_del_vigia_423.
            viejo = px.cfg.latido.stat().st_mtime - (px.cfg.latido_max_s + 1)
            os.utime(px.cfg.latido, (viejo, viejo))

            # Libera el carril SIN pasar por el freno (ni interruptor ni pausa se
            # tocan en este test): sólo el latido está viejo. Si el vigía de
            # `atender()` (predicado angosto) fuera lo único que corta, primera
            # seguiría streameando para siempre -- se cancela a mano, como hace el
            # test hermano de la carrera del freno.
            primera.cancel()
            await asyncio.gather(primera, return_exceptions=True)

            r = await asyncio.wait_for(segunda, 3)
            return r.status_code, r.headers.get("x-should-retry"), r.json(), len(up.recibidas)

    assert _correr(escenario()) == (
        423, "false", _error(proxy_carril.VIGIA_SIN_LATIDO), 1)


def test_frenado_colgado_al_tomar_el_carril_es_fail_closed(tmp_path, monkeypatch):
    """MINOR 1 de la revisión adversarial (2026-09-21): el re-chequeo al tomar el
    carril corre DENTRO de la sección crítica (el carril es un `flock` entre
    procesos). Si el `os.stat` que lo resuelve se cuelga -montaje en mal estado de
    la raíz del interruptor o de la pausa-, `_frenado_con_tope` no puede esperarlo
    para siempre: eso retendría el carril indefinidamente y ni el propio vigía de
    `atender()` podría cortarlo (usa el mismo executor de hilos). Vencido el tope,
    fail-closed: se trata como frenado, no como libre."""
    llamadas = {"n": 0}
    frenado_real = proxy_carril._Proxy._frenado

    async def _frenado_lenta_desde_el_carril(self):
        llamadas["n"] += 1
        if llamadas["n"] <= 2:
            # Los dos chequeos ANTES de tomar el carril (`_reenviar`, antes de
            # `carril_ejecutor_async`): rápidos, como siempre. Lo que se cuelga es
            # sólo el de DENTRO de la sección crítica, vía `_frenado_con_tope`.
            return await frenado_real(self)
        await asyncio.sleep(10)
        return None  # nunca se llega acá dentro del tope del test

    monkeypatch.setattr(proxy_carril._Proxy, "_frenado", _frenado_lenta_desde_el_carril)
    monkeypatch.setattr(proxy_carril, "_TOPE_FRENADO_AL_TOMAR_CARRIL_S", 0.05)

    async def escenario():
        async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 5) as px, httpx.AsyncClient() as cli:
            r = await asyncio.wait_for(cli.post(px.url + "/v1/messages", content=_CUERPO), 3)
            return r.status_code, r.headers.get("x-should-retry"), r.json(), len(up.recibidas)

    assert _correr(escenario()) == (
        423, "false", _error(proxy_carril.FRENO_INDETERMINADO), 0)


def test_con_el_freno_puesto_lo_que_ya_corrio_se_anota(tmp_path, freno):
    """Los resultados de herramientas que ya corrieron llegan al registro de C3 aunque el
    cerebro no responda: el freno no borra el rastro."""
    freno.write_text("{}")
    cuerpo = {"model": MODELO_PERMITIDO, "max_tokens": 5, "messages": [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_freno", "name": "Bash",
                                            "input": {"command": "true"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_freno", "content": "ok"}]}]}

    async def escenario():
        async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 2) as px, httpx.AsyncClient() as cli:
            r = await cli.post(px.url + "/v1/messages", json=cuerpo)
            return r.status_code, len(up.recibidas)

    assert _correr(escenario()) == (423, 0)
    eventos = [json.loads(l) for l in (tmp_path / "registro.jsonl").read_text().splitlines()]
    assert [e["evento"] for e in eventos][-1] == "resultado_devuelto", eventos


def test_sin_ruta_del_interruptor_el_proxy_no_arranca(tmp_path, monkeypatch):
    px = Proxy("http://127.0.0.1:9", tmp_path, 1)
    monkeypatch.delenv(interruptor.VARIABLE_RUTA, raising=False)
    with pytest.raises(interruptor.InterruptorSinConfigurar):
        _correr(proxy_carril.arrancar(px.cfg))
