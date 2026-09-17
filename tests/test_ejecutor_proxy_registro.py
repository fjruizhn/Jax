# tests/test_ejecutor_proxy_registro.py
"""C3 en el proxy: lo que el cerebro pide queda anotado ANTES de que el arnés reciba el
final del bloque; si no se puede anotar, el bloque no llega. Sin registro no hay acción."""
import asyncio
import gzip
import json

import h11
import httpx
import pytest

from jax.ejecutor import proxy_carril
from jax.ejecutor.contratos import registro as R
from jax.ejecutor.contratos.canario_upstream import UpstreamCanario, _stream_tool_use, guion_bash
from tests.test_ejecutor_proxy_carril import MAX_SALIDA_TOKENS, MODELO_PERMITIDO, Proxy, Upstream, _correr

_BASE = {"model": MODELO_PERMITIDO, "stream": True, "max_tokens": 5, "tools": [{"name": "Bash", "input_schema": {"type": "object"}}]}


def _resultado(tool_use_id, contenido):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": contenido}]}


def _registro(tmp_path):
    return [json.loads(l) for l in (tmp_path / "registro.jsonl").read_text().splitlines()]


def test_tool_use_y_resultado_quedan_encadenados_y_una_sola_vez(tmp_path):
    async def escenario():
        async with UpstreamCanario([guion_bash("toolu_1", "uptime")], "127.0.0.1", 0) as up, \
                Proxy(f"http://127.0.0.1:{up.puerto}", tmp_path, 2) as px, httpx.AsyncClient() as cli:
            url = px.url + "/v1/messages"
            e1 = (await cli.post(url, json={**_BASE, "messages": [{"role": "user", "content": "x"}]})).status_code
            recordatorio = {"role": "system", "content": [{"type": "text", "text": "r"}]}
            e2 = (await cli.post(url, json={**_BASE, "messages": [_resultado("toolu_1", "arriba"), recordatorio]})).status_code
            e3 = (await cli.post(url, json={**_BASE, "messages": [_resultado("toolu_1", "arriba")]})).status_code
            return e1, e2, e3

    assert _correr(escenario()) == (200, 200, 200)
    eventos = _registro(tmp_path)
    assert [e["evento"] for e in eventos] == ["registro_abierto", "herramienta_pedida", "resultado_devuelto"]
    assert (eventos[1]["tool_use_id"], eventos[1]["herramienta"], eventos[1]["entrada"]) == ("toolu_1", "Bash", {"command": "uptime"})
    assert eventos[2]["tool_use_id"] == "toolu_1" and eventos[2]["es_error"] is False
    assert R.verificar_cadena(tmp_path / "registro.jsonl") == R.Verificacion(True, 3, None, None)


def test_respuesta_no_stream_se_anota_antes_de_entregarse(tmp_path):
    async def escenario():
        async with UpstreamCanario([guion_bash("toolu_2", "ls")], "127.0.0.1", 0) as up, \
                Proxy(f"http://127.0.0.1:{up.puerto}", tmp_path, 2) as px, httpx.AsyncClient() as cli:
            r = await cli.post(px.url + "/v1/messages", json={**_BASE, "stream": False, "messages": []})
            return r.status_code, r.json()["content"][0]["id"]

    assert _correr(escenario()) == (200, "toolu_2")
    assert [e["evento"] for e in _registro(tmp_path)] == ["registro_abierto", "herramienta_pedida"]


def test_sin_registro_el_bloque_no_llega(tmp_path, monkeypatch):
    original = R.Registro.anotar

    def roto(self, evento):
        if evento.get("evento") == "herramienta_pedida":
            raise OSError("disco_lleno")
        return original(self, evento)

    async def escenario():
        async with UpstreamCanario([guion_bash("toolu_3", "rm -rf /tmp/x")], "127.0.0.1", 0) as up, \
                Proxy(f"http://127.0.0.1:{up.puerto}", tmp_path, 2) as px, httpx.AsyncClient() as cli:
            monkeypatch.setattr(R.Registro, "anotar", roto)
            recibido, cortado = b"", False
            try:
                async with cli.stream("POST", px.url + "/v1/messages", json={**_BASE, "messages": []}) as r:
                    async for trozo in r.aiter_raw():
                        recibido += trozo
            except httpx.RemoteProtocolError:
                cortado = True
            return recibido, cortado

    recibido, cortado = _correr(escenario())
    assert cortado is True
    assert b"content_block_stop" not in recibido and b"toolu_3" not in recibido


def test_un_resultado_que_no_se_puede_anotar_no_sube_al_cerebro(tmp_path, monkeypatch):
    def roto(self, evento):
        raise OSError("disco_lleno")

    async def escenario():
        async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 2) as px, httpx.AsyncClient() as cli:
            monkeypatch.setattr(R.Registro, "anotar", roto)
            r = await cli.post(px.url + "/v1/messages", json={**_BASE, "messages": [_resultado("t", "x")]})
            return r.status_code, r.json(), len(up.recibidas)

    assert _correr(escenario()) == (502, {"type": "error", "error": {"type": proxy_carril.REGISTRO_FALLO}}, 0)


def test_pide_identidad_y_rechaza_respuesta_comprimida(tmp_path):
    async def atender(reader, writer):
        conn = h11.Connection(h11.SERVER)
        while not isinstance(ev := conn.next_event(), (h11.EndOfMessage, h11.ConnectionClosed)):
            if ev is h11.NEED_DATA:
                conn.receive_data(await reader.read(65536))
        cuerpo = gzip.compress(json.dumps({"content": [{"type": "tool_use", "id": "t", "name": "Bash",
                                                        "input": {"command": "x"}}]}).encode())
        writer.write(conn.send(h11.Response(status_code=200, headers=[
            ("content-type", "application/json"), ("content-encoding", "gzip"), ("content-length", str(len(cuerpo)))])))
        writer.write(conn.send(h11.Data(data=cuerpo)))
        writer.write(conn.send(h11.EndOfMessage()))
        await writer.drain()
        writer.close()

    async def escenario():
        srv = await asyncio.start_server(atender, "127.0.0.1", 0)
        try:
            async with Proxy(f"http://127.0.0.1:{srv.sockets[0].getsockname()[1]}", tmp_path, 2) as px, \
                    httpx.AsyncClient() as cli:
                r = await cli.post(px.url + "/v1/messages", headers={"accept-encoding": "gzip"},
                                   json={**_BASE, "stream": False, "messages": []})
            async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 2) as px, httpx.AsyncClient() as cli:
                await cli.post(px.url + "/v1/messages", headers={"accept-encoding": "gzip, br"},
                               json={**_BASE, "messages": []})
                pedida = up.recibidas[0][2]
            return r.status_code, r.json()["error"]["type"], pedida
        finally:
            srv.close()
            await srv.wait_closed()

    estado, codigo, cabeceras = _correr(escenario())
    assert (estado, codigo) == (502, proxy_carril.REGISTRO_ILEGIBLE)
    assert cabeceras[b"accept-encoding"] == b"identity"


def test_registro_corrupto_no_arranca(tmp_path):
    (tmp_path / "registro.jsonl").write_bytes(b'{"n":1,"prev":"0"}\n{"n":2')
    cfg = proxy_carril.Config(upstream="http://127.0.0.1:9", raiz=tmp_path, tope_s=1, host="127.0.0.1", puerto=0,
                              registro=tmp_path / "registro.jsonl", pausa=tmp_path / "PAUSA",
                              latido=tmp_path / "latido", latido_max_s=60, modelo=MODELO_PERMITIDO,
                              max_salida_tokens=MAX_SALIDA_TOKENS)
    with pytest.raises(R.RegistroCorrupto):
        _correr(proxy_carril.arrancar(cfg))


def test_cuerpo_2xx_que_no_se_puede_leer_no_se_entrega(tmp_path):
    # El arnés decide si es stream por lo que PIDIÓ, no por el content-type: un SSE servido
    # como JSON no lo lee el proxy pero sí el arnés. Lo que el proxy no puede leer, no sale.
    sse = _stream_tool_use(guion_bash("toolu_9", "rm -rf /tmp/x"))

    async def atender(reader, writer):
        conn = h11.Connection(h11.SERVER)
        while not isinstance(ev := conn.next_event(), (h11.EndOfMessage, h11.ConnectionClosed)):
            if ev is h11.NEED_DATA:
                conn.receive_data(await reader.read(65536))
        writer.write(conn.send(h11.Response(status_code=200, headers=[
            ("content-type", "application/json"), ("content-length", str(len(sse)))])))
        writer.write(conn.send(h11.Data(data=sse)))
        writer.write(conn.send(h11.EndOfMessage()))
        await writer.drain()
        writer.close()

    async def escenario():
        srv = await asyncio.start_server(atender, "127.0.0.1", 0)
        try:
            async with Proxy(f"http://127.0.0.1:{srv.sockets[0].getsockname()[1]}", tmp_path, 2) as px, \
                    httpx.AsyncClient() as cli:
                r = await cli.post(px.url + "/v1/messages", json={**_BASE, "messages": []})
            return r.status_code, r.json()["error"]["type"], b"toolu_9" in r.content
        finally:
            srv.close()
            await srv.wait_closed()

    assert _correr(escenario()) == (502, proxy_carril.REGISTRO_ILEGIBLE, False)


def test_stream_comprimido_no_se_entrega(tmp_path):
    # Un SSE comprimido el lector no lo ve y el arnés sí lo descomprime: sin el chequeo de
    # codificación, la herramienta pasaría sin anotar.
    sse = gzip.compress(_stream_tool_use(guion_bash("toolu_8", "rm -rf /tmp/x")))

    async def atender(reader, writer):
        conn = h11.Connection(h11.SERVER)
        while not isinstance(ev := conn.next_event(), (h11.EndOfMessage, h11.ConnectionClosed)):
            if ev is h11.NEED_DATA:
                conn.receive_data(await reader.read(65536))
        writer.write(conn.send(h11.Response(status_code=200, headers=[
            ("content-type", "text/event-stream"), ("content-encoding", "gzip"), ("content-length", str(len(sse)))])))
        writer.write(conn.send(h11.Data(data=sse)))
        writer.write(conn.send(h11.EndOfMessage()))
        await writer.drain()
        writer.close()

    async def escenario():
        srv = await asyncio.start_server(atender, "127.0.0.1", 0)
        try:
            async with Proxy(f"http://127.0.0.1:{srv.sockets[0].getsockname()[1]}", tmp_path, 2) as px, \
                    httpx.AsyncClient() as cli:
                r = await cli.post(px.url + "/v1/messages", json={**_BASE, "messages": []})
            return r.status_code, r.json()["error"]["type"]
        finally:
            srv.close()
            await srv.wait_closed()

    assert _correr(escenario()) == (502, proxy_carril.REGISTRO_ILEGIBLE)
    assert [e["evento"] for e in _registro(tmp_path)] == ["registro_abierto"]


@pytest.mark.parametrize("metodo, ruta", [
    ("POST", "/api/pull"), ("DELETE", "/api/delete"), ("POST", "/api/create"), ("POST", "/api/chat"),
    ("POST", "/api/generate"), ("PUT", "/v1/messages"), ("POST", "/v1/messages/../../api/pull"),
])
def test_rutas_fuera_de_la_api_de_mensajes_no_llegan_al_upstream(tmp_path, metodo, ruta):
    # Medido 2026-09-17 con el arnés real (2.1.273) por el proxy: HEAD /api/hello y POST /v1/messages.
    # El upstream es el Ollama de producción: por el proxy, la jaula no borra, baja ni crea modelos,
    # ni pide inferencia por una ruta donde el registro no ve herramientas.
    async def escenario():
        async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 2) as px, httpx.AsyncClient() as cli:
            r = await cli.request(metodo, px.url + ruta, content=b"{}")
            return r.status_code, r.json()["error"]["type"], len(up.recibidas)

    assert _correr(escenario()) == (403, proxy_carril.RUTA_NO_PERMITIDA, 0)


@pytest.mark.parametrize("metodo, ruta", [("HEAD", "/api/hello"), ("POST", "/v1/messages")])
def test_las_rutas_del_arnes_si_llegan(tmp_path, metodo, ruta):
    # SP3 (2026-09-17): ya no pasa cualquier GET/HEAD. Sólo lo que el arnés manda de verdad.
    async def escenario():
        async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 2) as px, httpx.AsyncClient() as cli:
            await cli.request(metodo, px.url + ruta + "?beta=true",
                              content=json.dumps({**_BASE, "messages": []}).encode() if metodo == "POST" else None)
            return len(up.recibidas)

    assert _correr(escenario()) == 1
