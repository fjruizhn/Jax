"""Proxy con carril del Ejecutor (Fase 2, §3.4 bis).

Claude Code apunta `ANTHROPIC_BASE_URL` al proxy; el proxy toma
`carril_ejecutor_async` POR PETICIÓN y reenvía a Ollama con streaming. Estos
tests no usan Ollama: levantan un upstream falso en el mismo loop que cuenta
las peticiones que recibe y responde SSE a paso controlado por el test (un
`asyncio.Event` por trozo), así "llega por partes" no depende de relojes.

La Mesa se simula en OTRO PROCESO (`fork` explícito, como
test_ejecutor_prioridad.py): el carril coordina procesos, no tareas.
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import time

import h11
import httpx
import pytest

from jax.ejecutor.cita import Motivo
from jax.ejecutor.prioridad import carril_mesa
from jax.ejecutor.proxy_carril import (
    CONFIG_FALTA, CONFIG_INVALIDA, ESPERA_AGOTADA, UPSTREAM_INALCANZABLE, Config,
    ConfigInvalida, arrancar, config_desde_entorno,
)

_CTX = mp.get_context("fork")


def _rematar(p):
    p.join(5)
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join(5)


def _mesa_retiene(raiz, listo, suelte, hasta_s):
    with carril_mesa(raiz):
        listo.set()
        suelte.wait(hasta_s)


# --------------------------------------------------------------------------
# Upstream falso
# --------------------------------------------------------------------------

class Upstream:
    """Servidor HTTP mínimo. `modo`:
    - "sse": 200 text/event-stream; manda `trozo-0`, y cada trozo siguiente
      sólo cuando el test hace `avanzar.set()`; `n_trozos` en total.
    - "error": 500 con cuerpo corto.
    - "se_cae": cabeceras + un trozo y corta la conexión a lo bruto.
    """

    def __init__(self, modo="sse", n_trozos=3):
        self.modo = modo
        self.n_trozos = n_trozos
        self.recibidas = []
        self._abiertas = set()
        self.avanzar = asyncio.Event()
        self.server = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"

    async def __aenter__(self):
        self.server = await asyncio.start_server(self._atender, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *exc):
        # Desde 3.12 `wait_closed` espera a que cierren TODAS las conexiones:
        # una que quedó esperando `avanzar` se corta a mano.
        self.server.close()
        for w in self._abiertas:
            w.transport.abort()
        await self.server.wait_closed()

    async def _atender(self, reader, writer):
        self._abiertas.add(writer)
        conn = h11.Connection(h11.SERVER)
        req, cuerpo = None, b""
        while True:
            ev = conn.next_event()
            if ev is h11.NEED_DATA:
                conn.receive_data(await reader.read(65536))
                continue
            if isinstance(ev, h11.Request):
                req = ev
            elif isinstance(ev, h11.Data):
                cuerpo += ev.data
            elif isinstance(ev, (h11.EndOfMessage, h11.ConnectionClosed)):
                break
        self.recibidas.append((req.method, req.target, dict(req.headers), cuerpo))
        try:
            if self.modo == "error":
                writer.write(conn.send(h11.Response(
                    status_code=500, headers=[("content-length", "4")])))
                writer.write(conn.send(h11.Data(data=b"roto")))
                writer.write(conn.send(h11.EndOfMessage()))
                await writer.drain()
                return
            writer.write(conn.send(h11.Response(
                status_code=200, headers=[("content-type", "text/event-stream")])))
            for i in range(self.n_trozos):
                if i:
                    self.avanzar.clear()
                    await self.avanzar.wait()
                writer.write(conn.send(h11.Data(data=f"data: trozo-{i}\n\n".encode())))
                await writer.drain()
                if self.modo == "se_cae":
                    writer.transport.abort()
                    return
            writer.write(conn.send(h11.EndOfMessage()))
            await writer.drain()
        except ConnectionError:  # fail-soft: upstream falso de test; el proxy ya cortó y el test mide eso
            return
        finally:
            self._abiertas.discard(writer)
            writer.close()


class Proxy:
    def __init__(self, upstream_url, raiz, tope_s):
        self.cfg = Config(upstream=upstream_url, raiz=raiz, tope_s=tope_s,
                          host="127.0.0.1", puerto=0, registro=raiz / "registro.jsonl")

    async def __aenter__(self):
        self.server = await arrancar(self.cfg)
        puerto = self.server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{puerto}"
        self.puerto = puerto
        return self

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()


def _correr(coro, tope=20):
    return asyncio.run(asyncio.wait_for(coro, tope))


_CABECERAS = {"authorization": "Bearer llave-secreta-XYZ", "x-api-key": "llave-secreta-XYZ"}
_CUERPO = b'{"messages":[{"role":"user","content":"dato-de-cliente-ABC"}]}'


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_una_peticion_pasa_y_el_stream_llega_por_partes(tmp_path):
    """El test NO deja que el upstream mande el trozo 1 hasta que el cliente
    haya recibido el 0. Si el proxy bufferizara la respuesta entera, el
    cliente no recibiría nada y esto vence por tiempo."""
    async def escenario():
        async with Upstream() as up, Proxy(up.url, tmp_path, 2) as px:
            async with httpx.AsyncClient() as cli:
                async with cli.stream("POST", px.url + "/v1/messages?beta=true",
                                      headers=_CABECERAS, content=_CUERPO) as r:
                    assert r.status_code == 200
                    assert r.headers["content-type"] == "text/event-stream"
                    recibido = b""
                    trozos = r.aiter_raw()
                    for i in range(3):
                        esperado = f"data: trozo-{i}\n\n".encode()
                        while esperado not in recibido:
                            recibido += await asyncio.wait_for(anext(trozos), 3)
                        up.avanzar.set()
            metodo, destino, cabeceras, cuerpo = up.recibidas[0]
            return len(up.recibidas), metodo, destino, cabeceras, cuerpo

    n, metodo, destino, cabeceras, cuerpo = _correr(escenario())
    assert n == 1
    assert (metodo, destino, cuerpo) == (b"POST", b"/v1/messages?beta=true", _CUERPO)
    assert cabeceras[b"authorization"] == b"Bearer llave-secreta-XYZ"


def test_el_reenvio_no_tiene_tope_de_lectura(tmp_path, monkeypatch):
    """El primer byte de Ollama puede tardar lo que tarde su cola: el reenvío
    va sin tope de lectura (connect 10 s). Desde E-24 el proxy usa el cliente de
    jax/core/cliente_http_compartido.py, cuyo default es 5 s: el timeout lo pone
    CADA petición. Sin eso, un primer byte de más de 5 s sería un 502."""
    vistos = []
    send_real = httpx.AsyncClient.send

    async def send_espia(self, request, **kw):
        vistos.append((request.url.port, request.extensions.get("timeout")))
        return await send_real(self, request, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "send", send_espia)

    async def escenario():
        async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 2) as px:
            async with httpx.AsyncClient(timeout=10) as cli:
                r = await cli.post(px.url + "/v1/messages", content=_CUERPO)
            return r.status_code, int(up.url.rsplit(":", 1)[1])

    estado, puerto_up = _correr(escenario())
    assert estado == 200
    al_upstream = [t for puerto, t in vistos if puerto == puerto_up]
    assert al_upstream == [{"connect": 10.0, "read": None, "write": None, "pool": None}]


def test_con_la_mesa_en_el_carril_la_peticion_espera_y_entra_cuando_suelta(tmp_path):
    listo, suelte = _CTX.Event(), _CTX.Event()
    p = _CTX.Process(target=_mesa_retiene, args=(tmp_path, listo, suelte, 0.6))
    p.start()
    try:
        assert listo.wait(5) is True

        async def escenario():
            async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 5) as px:
                async with httpx.AsyncClient(timeout=10) as cli:
                    t0 = time.monotonic()
                    r = await cli.post(px.url + "/v1/messages", content=_CUERPO)
                    return r.status_code, time.monotonic() - t0, len(up.recibidas)

        estado, espera, n = _correr(escenario())
        assert estado == 200
        assert espera >= 0.3, "no esperó a la Mesa"
        assert n == 1
    finally:
        suelte.set(); _rematar(p)


def test_con_la_mesa_y_tope_corto_503_sin_tocar_el_upstream(tmp_path):
    listo, suelte = _CTX.Event(), _CTX.Event()
    p = _CTX.Process(target=_mesa_retiene, args=(tmp_path, listo, suelte, 10))
    p.start()
    try:
        assert listo.wait(5) is True

        async def escenario():
            async with Upstream() as up, Proxy(up.url, tmp_path, 0.3) as px:
                async with httpx.AsyncClient(timeout=10) as cli:
                    r = await cli.post(px.url + "/v1/messages", content=_CUERPO)
                    return r, len(up.recibidas)

        r, n = _correr(escenario())
        assert r.status_code == 503
        assert n == 0, "la petición llegó al upstream: se coló"
        # Formato de máquina: código y datos, sin frases.
        assert r.json() == {"type": "error",
                            "error": {"type": ESPERA_AGOTADA, "tope_s": 0.3}}
        assert r.headers["x-should-retry"] == "false"
    finally:
        suelte.set(); _rematar(p)


def test_otra_peticion_espera_mientras_dura_el_stream(tmp_path):
    """El carril se retiene hasta que TERMINA el stream, no hasta que llegan
    las cabeceras: con A a mitad de su stream, B da 503 sin tocar el
    upstream; terminado A, C entra."""
    async def escenario():
        async with Upstream(n_trozos=2) as up, Proxy(up.url, tmp_path, 0.4) as px:
            async with httpx.AsyncClient(timeout=10) as cli:
                async with cli.stream("POST", px.url + "/v1/messages", content=_CUERPO) as a:
                    trozos = a.aiter_raw()
                    await asyncio.wait_for(anext(trozos), 3)  # A está a mitad
                    # En stream: si B se colara, su cuerpo tampoco terminaría y
                    # el test caería por tiempo en vez de decir por qué.
                    async with cli.stream("POST", px.url + "/v1/messages",
                                          content=_CUERPO) as b:
                        estado_b, n_durante = b.status_code, len(up.recibidas)
                    up.avanzar.set()
                    async for _ in trozos:
                        pass
                up.n_trozos = 1
                c = await cli.post(px.url + "/v1/messages", content=_CUERPO)
                return estado_b, n_durante, c.status_code

    estado_b, n_durante, estado_c = _correr(escenario())
    assert estado_b == 503
    assert n_durante == 1, "B llegó al upstream mientras A seguía en su stream"
    assert estado_c == 200, "C no entró tras terminar el stream de A"


def test_el_carril_se_suelta_si_el_cliente_corta_a_mitad_del_stream(tmp_path):
    """El upstream se queda callado después del primer trozo: el proxy no
    tiene nada que escribir, así que sólo puede enterarse del corte mirando la
    conexión del cliente."""
    async def escenario():
        async with Upstream(n_trozos=5) as up, Proxy(up.url, tmp_path, 2) as px:
            reader, writer = await asyncio.open_connection("127.0.0.1", px.puerto)
            writer.write(b"POST /v1/messages HTTP/1.1\r\nhost: x\r\ncontent-length: %d\r\n\r\n"
                         % len(_CUERPO) + _CUERPO)
            await writer.drain()
            leido = b""
            while b"trozo-0" not in leido:
                leido += await asyncio.wait_for(reader.read(4096), 3)
            writer.close()  # el cliente corta a mitad
            await writer.wait_closed()
            async with httpx.AsyncClient(timeout=10) as cli:
                async with cli.stream("POST", px.url + "/v1/messages", content=_CUERPO) as r:
                    return r.status_code, len(up.recibidas)

    estado, n = _correr(escenario())
    assert estado == 200, "el carril quedó tomado tras el corte del cliente"
    assert n == 2


def test_el_carril_se_suelta_si_el_upstream_devuelve_error(tmp_path):
    async def escenario():
        async with Upstream(modo="error") as up, Proxy(up.url, tmp_path, 0.5) as px:
            async with httpx.AsyncClient(timeout=10) as cli:
                r1 = await cli.post(px.url + "/v1/messages", content=_CUERPO)
                r2 = await cli.post(px.url + "/v1/messages", content=_CUERPO)
                return r1.status_code, r1.content, r2.status_code, len(up.recibidas)

    e1, c1, e2, n = _correr(escenario())
    assert (e1, c1) == (500, b"roto"), "el error del upstream se reenvía tal cual"
    assert e2 == 500 and n == 2


def test_el_carril_se_suelta_si_el_upstream_se_cae_a_mitad(tmp_path):
    async def escenario():
        async with Upstream(modo="se_cae") as up, Proxy(up.url, tmp_path, 0.5) as px:
            async with httpx.AsyncClient(timeout=10) as cli:
                with pytest.raises(httpx.HTTPError):
                    async with cli.stream("POST", px.url + "/v1/messages", content=_CUERPO) as r:
                        async for _ in r.aiter_raw():
                            pass
                async with cli.stream("POST", px.url + "/v1/messages", content=_CUERPO) as r2:
                    return r2.status_code, len(up.recibidas)

    estado, n = _correr(escenario())
    assert estado == 200 and n == 2


def test_el_carril_se_suelta_si_el_upstream_no_existe(tmp_path):
    async def escenario():
        # Un puerto que acabamos de soltar: nadie escucha.
        srv = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
        muerto = f"http://127.0.0.1:{srv.sockets[0].getsockname()[1]}"
        srv.close(); await srv.wait_closed()
        async with Proxy(muerto, tmp_path, 0.3) as px:
            async with httpx.AsyncClient(timeout=10) as cli:
                r1 = await cli.post(px.url + "/v1/messages", content=_CUERPO)
                r2 = await cli.post(px.url + "/v1/messages", content=_CUERPO)
                return r1, r2.status_code

    r1, e2 = _correr(escenario())
    assert r1.status_code == 502
    assert r1.json() == {"type": "error", "error": {"type": UPSTREAM_INALCANZABLE}}
    assert e2 == 502, "la segunda no entró: el carril quedó tomado"


def test_ni_la_llave_ni_el_cuerpo_aparecen_en_los_logs(tmp_path, caplog):
    """Camino feliz Y camino del 503, con TODOS los loggers en DEBUG (httpx y
    httpcore incluidos): la llave y los datos del cliente no salen."""
    caplog.set_level(logging.DEBUG)

    async def peticion(tope_s):
        async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, tope_s) as px:
            async with httpx.AsyncClient(timeout=10) as cli:
                r = await cli.post(px.url + "/v1/messages", headers=_CABECERAS,
                                   content=_CUERPO)
                return r.status_code

    estado_ok = _correr(peticion(1))
    listo, suelte = _CTX.Event(), _CTX.Event()
    p = _CTX.Process(target=_mesa_retiene, args=(tmp_path, listo, suelte, 10))
    p.start()
    try:
        assert listo.wait(5) is True
        estado_rechazo = _correr(peticion(0.2))
    finally:
        suelte.set(); _rematar(p)

    assert (estado_ok, estado_rechazo) == (200, 503)
    propios = [r for r in caplog.records if r.name.startswith("jax.ejecutor.proxy_carril")]
    assert propios, "el proxy no logueó nada: el test no probaría nada"
    texto = caplog.text + "".join(repr(r.args) + repr(r.msg) for r in caplog.records)
    assert "llave-secreta-XYZ" not in texto
    assert "dato-de-cliente-ABC" not in texto


_ENTORNO = {
    "JAX_PROXY_CARRIL_UPSTREAM": "http://ollama.invalid:9/",
    "JAX_PROXY_CARRIL_RAIZ": "/srv/ejemplo/locks",
    "JAX_PROXY_CARRIL_TOPE_S": "120",
    "JAX_PROXY_CARRIL_PUERTO": "8199",
    "JAX_EJECUTOR_REGISTRO": "/var/log/jax-ejecutor/registro.jsonl",
}


def test_config_sale_del_entorno_sin_upstream_hardcodeado():
    cfg = config_desde_entorno(_ENTORNO)
    assert cfg.upstream == "http://ollama.invalid:9"
    assert (str(cfg.raiz), cfg.tope_s, cfg.puerto) == ("/srv/ejemplo/locks", 120.0, 8199)
    assert cfg.host == "127.0.0.1", "sin HOST, sólo loopback: el proxy no autentica"
    assert str(cfg.registro) == "/var/log/jax-ejecutor/registro.jsonl"


@pytest.mark.parametrize("variable", sorted(_ENTORNO))
def test_config_sin_una_obligatoria_falla_cerrado(variable):
    env = {k: v for k, v in _ENTORNO.items() if k != variable}
    with pytest.raises(ConfigInvalida) as err:
        config_desde_entorno(env)
    assert err.value.args == (Motivo(CONFIG_FALTA, (("variable", variable),)),)


@pytest.mark.parametrize("variable,valor", [
    ("JAX_PROXY_CARRIL_TOPE_S", "mucho"), ("JAX_PROXY_CARRIL_TOPE_S", "-1"),
    ("JAX_PROXY_CARRIL_PUERTO", "8199.5"), ("JAX_PROXY_CARRIL_UPSTREAM", "ollama:11434"),
    ("JAX_EJECUTOR_REGISTRO", "relativa/registro.jsonl"),
])
def test_config_invalida_falla_cerrado(variable, valor):
    with pytest.raises(ConfigInvalida) as err:
        config_desde_entorno({**_ENTORNO, variable: valor})
    assert err.value.args == (Motivo(CONFIG_INVALIDA, (("variable", variable),)),)
