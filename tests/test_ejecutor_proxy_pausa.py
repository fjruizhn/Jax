# tests/test_ejecutor_proxy_pausa.py
"""C5 en el proxy: con la pausa del Ejecutor puesta, o sin un vigía que lata, no hay
cerebro (423 sin tocar el upstream); y si el freno aparece mientras el cerebro responde,
el tool_use queda anotado pero su trozo final NO llega al arnés."""
import json
import os

import httpx

from jax.ejecutor import proxy_carril
from jax.ejecutor.contratos import pausa as P
from jax.ejecutor.contratos import registro as R
from jax.ejecutor.contratos.canario_upstream import UpstreamCanario, guion_bash
from tests.test_ejecutor_proxy_carril import MODELO_PERMITIDO, Proxy, Upstream, _correr

# El modelo fijado por el proxy (SP3): con otro, el 403 de modelo taparía lo que se mide acá.
_BASE = {"model": MODELO_PERMITIDO, "stream": True, "max_tokens": 5, "tools": [{"name": "Bash", "input_schema": {"type": "object"}}]}


def _error(tipo):
    return {"type": "error", "error": {"type": tipo}}


def test_con_la_pausa_puesta_423_sin_tocar_el_upstream(tmp_path, caplog):
    async def escenario():
        async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 2) as px, httpx.AsyncClient() as cli:
            P.poner_pausa(px.cfg.pausa, {"origen": "c5", "motivo": "fuera_de_mision", "paso": 3})
            r = await cli.post(px.url + "/v1/messages", json={**_BASE, "messages": []})
            h = await cli.head(px.url + "/api/hello")
            return r.status_code, r.headers.get("x-should-retry"), r.json(), h.status_code, len(up.recibidas)

    assert _correr(escenario()) == (423, "false", _error(proxy_carril.EJECUTOR_PAUSADO), 423, 0)
    # El 423 a un HEAD sale sin cuerpo (RFC 9110 §9.3.2); con cuerpo, h11 lanza dentro del handler.
    assert "Unhandled exception" not in caplog.text


def test_sin_latido_del_vigia_423(tmp_path):
    async def escenario():
        async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 2) as px, httpx.AsyncClient() as cli:
            viejo = px.cfg.latido.stat().st_mtime - 7200
            os.utime(px.cfg.latido, (viejo, viejo))
            r1 = await cli.post(px.url + "/v1/messages", json={**_BASE, "messages": []})
            px.cfg.latido.unlink()
            r2 = await cli.post(px.url + "/v1/messages", json={**_BASE, "messages": []})
            return r1.status_code, r1.json(), r2.status_code, len(up.recibidas)

    assert _correr(escenario()) == (423, _error(proxy_carril.VIGIA_SIN_LATIDO), 423, 0)


def test_con_vigia_vivo_y_sin_pausa_pasa(tmp_path):
    async def escenario():
        async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 2) as px, httpx.AsyncClient() as cli:
            r = await cli.post(px.url + "/v1/messages", json={**_BASE, "messages": []})
            return r.status_code, len(up.recibidas)

    assert _correr(escenario()) == (200, 1)


def _pausa_al_anotar(monkeypatch, cfg_de):
    """Simula al vigía frenando justo cuando el proxy anota el tool_use."""
    original = R.Registro.anotar

    def y_pausa(self, evento):
        n = original(self, evento)
        if evento.get("evento") == "herramienta_pedida":
            P.poner_pausa(cfg_de().pausa, {"origen": "c5", "motivo": "fuera_de_mision", "paso": n})
        return n
    monkeypatch.setattr(R.Registro, "anotar", y_pausa)


def test_si_c5_frena_en_vuelo_el_tool_use_no_llega_por_stream(tmp_path, monkeypatch):
    caja = {}

    async def escenario():
        async with UpstreamCanario([guion_bash("toolu_c5", "cat /etc/jax/.env")], "127.0.0.1", 0) as up, \
                Proxy(f"http://127.0.0.1:{up.puerto}", tmp_path, 2) as px, httpx.AsyncClient() as cli:
            caja["px"] = px
            _pausa_al_anotar(monkeypatch, lambda: caja["px"].cfg)
            recibido, cortado = b"", False
            try:
                async with cli.stream("POST", px.url + "/v1/messages", json={**_BASE, "messages": []}) as r:
                    async for trozo in r.aiter_raw():
                        recibido += trozo
            except httpx.RemoteProtocolError:
                cortado = True
            return recibido, cortado

    recibido, cortado = _correr(escenario())
    assert cortado is True and b"content_block_stop" not in recibido
    eventos = [json.loads(l) for l in (tmp_path / "registro.jsonl").read_text().splitlines()]
    assert [e["evento"] for e in eventos] == ["registro_abierto", "herramienta_pedida"], "anotado aunque no se entregue"


def test_si_c5_frena_en_vuelo_el_tool_use_no_llega_sin_stream(tmp_path, monkeypatch):
    caja = {}

    async def escenario():
        async with UpstreamCanario([guion_bash("toolu_c5b", "cat /etc/jax/.env")], "127.0.0.1", 0) as up, \
                Proxy(f"http://127.0.0.1:{up.puerto}", tmp_path, 2) as px, httpx.AsyncClient() as cli:
            caja["px"] = px
            _pausa_al_anotar(monkeypatch, lambda: caja["px"].cfg)
            r = await cli.post(px.url + "/v1/messages", json={**_BASE, "stream": False, "messages": []})
            return r.status_code, r.content

    estado, cuerpo = _correr(escenario())
    assert estado == 423 and b"toolu_c5b" not in cuerpo


def test_el_freno_va_antes_que_la_politica_de_rutas_y_de_modelo(tmp_path):
    """Orden de _reenviar tras juntar C5 y SP3: con el Ejecutor frenado, TODA petición
    recibe 423 -- también una ruta cerrada o un modelo ajeno, que sin freno darían 403 --,
    y nada llega al upstream. Un freno no depende de que la petición sea válida."""
    async def escenario():
        async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 2) as px, httpx.AsyncClient() as cli:
            P.poner_pausa(px.cfg.pausa, {"origen": "c5", "motivo": "fuera_de_mision", "paso": 1})
            ruta = await cli.get(px.url + "/api/tags")
            modelo = await cli.post(px.url + "/v1/messages", json={**_BASE, "model": "otro", "messages": []})
            salida = await cli.post(px.url + "/v1/messages", json={**_BASE, "max_tokens": 10 ** 6, "messages": []})
            ilegible = await cli.post(px.url + "/v1/messages", content=b"{no es json")
            return ([(r.status_code, r.json()) for r in (ruta, modelo, salida, ilegible)], len(up.recibidas))

    esperado = (423, _error(proxy_carril.EJECUTOR_PAUSADO))
    assert _correr(escenario()) == ([esperado] * 4, 0)
