# tests/test_ejecutor_proxy_freno.py
"""C4 en el proxy: con el interruptor de JAX puesto, o con la pausa del Ejecutor (C5), el
cerebro no responde (423 legible sin tocar el upstream) y un stream en curso se corta en
menos de un segundo, soltando el carril. Sin saber dónde está el interruptor, no arranca."""
import asyncio
import json
import time

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
