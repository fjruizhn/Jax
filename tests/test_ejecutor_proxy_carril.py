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
import dataclasses
import json
import logging
import multiprocessing as mp
import time

import h11
import httpx
import pytest

from jax.ejecutor.cita import Motivo
from jax.ejecutor.contratos.pausa import latir
from jax.ejecutor.prioridad import carril_mesa
from jax.ejecutor import proxy_carril
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


#: El único modelo que la jaula puede pedir en estos tests, y su tope de salida.
MODELO_PERMITIDO = "modelo-permitido"
MAX_SALIDA_TOKENS = 1024


class Proxy:
    def __init__(self, upstream_url, raiz, tope_s, pensamiento="libre"):
        # C5: sin pausa del Ejecutor y con un vigía que acaba de latir (lo prueban
        # test_ejecutor_proxy_pausa.py); estos tests miran el carril y el registro.
        self.cfg = Config(upstream=upstream_url, raiz=raiz, tope_s=tope_s,
                          host="127.0.0.1", puerto=0, registro=raiz / "registro.jsonl",
                          pausa=raiz / "PAUSA", latido=raiz / "latido", latido_max_s=3600,
                          modelo=MODELO_PERMITIDO, max_salida_tokens=MAX_SALIDA_TOKENS,
                          pensamiento=pensamiento)
        latir(self.cfg.latido)

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
_CUERPO = (b'{"model":"modelo-permitido","max_tokens":1024,'
           b'"messages":[{"role":"user","content":"dato-de-cliente-ABC"}]}')


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
    "JAX_EJECUTOR_PAUSA": "/etc/jax/interruptor/EJECUTOR_PAUSA",
    "JAX_EJECUTOR_VIGIA_LATIDO": "/var/lib/jax-ejecutor/vigia.latido",
    "JAX_EJECUTOR_VIGIA_LATIDO_MAX_S": "30",
    "JAX_PROXY_CARRIL_MODELO": "modelo-de-prueba-carril",
    "JAX_PROXY_CARRIL_MAX_SALIDA_TOKENS": "1024",
    "JAX_PROXY_CARRIL_PENSAMIENTO": "apagado",
}


def test_config_sale_del_entorno_sin_upstream_hardcodeado():
    cfg = config_desde_entorno(_ENTORNO)
    assert cfg.upstream == "http://ollama.invalid:9"
    assert (str(cfg.raiz), cfg.tope_s, cfg.puerto) == ("/srv/ejemplo/locks", 120.0, 8199)
    assert cfg.host == "127.0.0.1", "sin HOST, sólo loopback: el proxy no autentica"
    assert str(cfg.registro) == "/var/log/jax-ejecutor/registro.jsonl"
    assert (str(cfg.pausa), str(cfg.latido), cfg.latido_max_s) == (
        "/etc/jax/interruptor/EJECUTOR_PAUSA", "/var/lib/jax-ejecutor/vigia.latido", 30.0)
    assert (cfg.modelo, cfg.max_salida_tokens) == ("modelo-de-prueba-carril", 1024)


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
    ("JAX_EJECUTOR_PAUSA", "relativa/PAUSA"), ("JAX_EJECUTOR_VIGIA_LATIDO", "relativa/latido"),
    ("JAX_EJECUTOR_VIGIA_LATIDO_MAX_S", "0"), ("JAX_EJECUTOR_VIGIA_LATIDO_MAX_S", "nan"),
    ("JAX_EJECUTOR_VIGIA_LATIDO_MAX_S", "inf"),
    ("JAX_PROXY_CARRIL_MAX_SALIDA_TOKENS", "0"), ("JAX_PROXY_CARRIL_MAX_SALIDA_TOKENS", "1024.5"),
    ("JAX_PROXY_CARRIL_MAX_SALIDA_TOKENS", "-5"),
])
def test_config_invalida_falla_cerrado(variable, valor):
    with pytest.raises(ConfigInvalida) as err:
        config_desde_entorno({**_ENTORNO, variable: valor})
    assert err.value.args == (Motivo(CONFIG_INVALIDA, (("variable", variable),)),)


def test_la_cola_del_carril_se_ve_y_vuelve_a_cero(tmp_path):
    """`esperando_carril()` cuenta las que esperan el carril y vuelve a cero al soltarlo. Sin este
    estado, una prueba sólo puede dormir un rato y cruzar los dedos (rojo intermitente en CI,
    2026-09-17). Cuenta también la que se va por el tope: si no, la cola quedaría inflada."""
    async def escenario():
        # n_trozos=2: el upstream de la primera espera `avanzar`, así que retiene el carril
        # mientras la segunda hace cola (con 1 trozo la primera terminaba antes y no había cola).
        async with Upstream(n_trozos=2) as up, Proxy(up.url, tmp_path, 30) as px, httpx.AsyncClient() as cli:
            assert proxy_carril.esperando_carril() == 0
            primera = asyncio.create_task(cli.post(px.url + "/v1/messages", content=_CUERPO))
            while not up.recibidas:
                await asyncio.sleep(0.02)
            segunda = asyncio.create_task(cli.post(px.url + "/v1/messages", content=_CUERPO))
            limite = time.monotonic() + 5
            while proxy_carril.esperando_carril() < 1:
                assert time.monotonic() < limite
                await asyncio.sleep(0.02)
            en_cola = proxy_carril.esperando_carril()
            up.avanzar.set()
            await asyncio.gather(primera, segunda, return_exceptions=True)
            limite = time.monotonic() + 5
            while proxy_carril.esperando_carril() != 0:
                assert time.monotonic() < limite
                await asyncio.sleep(0.02)
            return en_cola

    assert _correr(escenario()) == 1


# --------------------------------------------------------------------------
# Pensamiento del cerebro impuesto por el proxy (decisión de Fernando, 2026-10-03)
# --------------------------------------------------------------------------
# Con `JAX_PROXY_CARRIL_PENSAMIENTO=apagado` el proxy SOBRESCRIBE `thinking` en lo que reenvía;
# con `libre` el cuerpo pasa byte a byte. Se valida siempre sobre el cuerpo original: una
# petición rechazada nunca llega al upstream. (Vive acá y no en un archivo aparte para que corra
# en el job de CI que ya lista este archivo.)

VARIABLE_PENSAMIENTO = "JAX_PROXY_CARRIL_PENSAMIENTO"
_DESHABILITADO = {"type": "disabled"}


def _mensaje_p(**extra):
    base = {"model": MODELO_PERMITIDO, "max_tokens": MAX_SALIDA_TOKENS,
            "messages": [{"role": "user", "content": "hola ñandú"}]}
    base.update(extra)
    return json.dumps(base).encode()


def _enviar_p(tmp_path, pensamiento, cuerpo, ruta="/v1/messages"):
    async def escenario():
        async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 2, pensamiento=pensamiento) as px, \
                httpx.AsyncClient() as cli:
            r = await cli.post(px.url + ruta, content=cuerpo)
            return r.status_code, list(up.recibidas)

    return _correr(escenario())


@pytest.mark.parametrize("del_cliente", [
    {}, {"thinking": {"type": "enabled", "budget_tokens": 2048}},
    {"thinking": {"type": "disabled"}}, {"thinking": None}, {"thinking": "raro"},
])
def test_apagado_el_upstream_recibe_disabled_y_content_length_coincide(tmp_path, del_cliente):
    estado, recibidas = _enviar_p(tmp_path, "apagado", _mensaje_p(**del_cliente))
    assert estado == 200 and len(recibidas) == 1
    _, _, cabeceras, cuerpo = recibidas[0]
    pedido = json.loads(cuerpo)
    assert pedido["thinking"] == _DESHABILITADO
    assert pedido["model"] == MODELO_PERMITIDO and pedido["messages"][0]["content"] == "hola ñandú"
    assert int(cabeceras[b"content-length"]) == len(cuerpo)
    assert b"transfer-encoding" not in cabeceras


def test_apagado_conserva_el_resto_del_pedido(tmp_path):
    original = json.loads(_mensaje_p(stream=True, temperature=0.2, system="s"))
    _, recibidas = _enviar_p(tmp_path, "apagado", json.dumps(original).encode())
    enviado = json.loads(recibidas[0][3])
    assert enviado == {**original, "thinking": _DESHABILITADO}


def test_libre_el_cuerpo_llega_identico(tmp_path):
    # Espacios y orden a propósito raros: si se re-serializara, ya no serían idénticos.
    cuerpo = (b'{ "messages": [{"role":"user","content":"hola"}],  "max_tokens": 1024, '
              b'"model": "modelo-permitido", "thinking": {"type":"enabled","budget_tokens":2048} }')
    estado, recibidas = _enviar_p(tmp_path, "libre", cuerpo)
    assert estado == 200 and recibidas[0][3] == cuerpo
    assert int(recibidas[0][2][b"content-length"]) == len(cuerpo)


@pytest.mark.parametrize("pensamiento", ["apagado", "libre"])
@pytest.mark.parametrize("cuerpo, ruta", [
    (_mensaje_p(model="otro-modelo"), "/v1/messages"),
    (_mensaje_p(max_tokens=MAX_SALIDA_TOKENS + 1), "/v1/messages"),
    (b"{", "/v1/messages"),
    (_mensaje_p(), "/api/show"),
])
def test_una_peticion_rechazada_nunca_llega_al_upstream(tmp_path, pensamiento, cuerpo, ruta):
    estado, recibidas = _enviar_p(tmp_path, pensamiento, cuerpo, ruta)
    assert estado == 403 and recibidas == []


def test_la_reescritura_que_falla_no_reenvia(tmp_path, monkeypatch):
    def roto(cuerpo):
        raise ValueError("no se puede")

    monkeypatch.setattr(proxy_carril, "_con_pensamiento_apagado", roto)

    async def escenario():
        async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 2, pensamiento="apagado") as px, \
                httpx.AsyncClient() as cli:
            r = await cli.post(px.url + "/v1/messages", content=_mensaje_p())
            return r.status_code, r.json()["error"]["type"], len(up.recibidas)

    assert _correr(escenario()) == (502, proxy_carril.REESCRITURA_FALLO, 0)


def test_apagado_queda_anotado_sin_volcar_el_cuerpo(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=proxy_carril.log.name):
        _enviar_p(tmp_path, "apagado", _mensaje_p())
    assert "pensamiento=apagado" in caplog.text
    assert "hola ñandú" not in caplog.text


def test_libre_no_anota_pensamiento_apagado(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=proxy_carril.log.name):
        _enviar_p(tmp_path, "libre", _mensaje_p())
    assert "pensamiento=apagado" not in caplog.text


@pytest.mark.parametrize("valor", ["apagado", "libre"])
def test_config_acepta_solo_los_dos_valores(valor):
    assert config_desde_entorno({**_ENTORNO, VARIABLE_PENSAMIENTO: valor}).pensamiento == valor


def test_config_sin_la_variable_falla_cerrado():
    with pytest.raises(ConfigInvalida) as err:
        config_desde_entorno({k: v for k, v in _ENTORNO.items() if k != VARIABLE_PENSAMIENTO})
    assert err.value.args == (Motivo(CONFIG_FALTA, (("variable", VARIABLE_PENSAMIENTO),)),)


@pytest.mark.parametrize("valor", ["Apagado", "APAGADO", "off", "encendido", "true", "libre;", "apagado,libre"])
def test_config_con_valor_invalido_falla_cerrado(valor):
    with pytest.raises(ConfigInvalida) as err:
        config_desde_entorno({**_ENTORNO, VARIABLE_PENSAMIENTO: valor})
    assert err.value.args == (Motivo(CONFIG_INVALIDA, (("variable", VARIABLE_PENSAMIENTO),)),)


def test_el_config_no_tiene_valor_por_omision():
    campo = {f.name: f for f in dataclasses.fields(proxy_carril.Config)}["pensamiento"]
    assert campo.default is dataclasses.MISSING


# --- Ronda de la auditoría de escalón 3 sobre el pensamiento apagado (2026-10-03) ---------------

def _posteo(tmp_path, pensamiento, cuerpo, preparar=None):
    """Un POST /v1/messages por el proxy; devuelve (estado, cabeceras de respuesta, recibidas)."""
    async def escenario():
        async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 2, pensamiento=pensamiento) as px, \
                httpx.AsyncClient() as cli:
            if preparar:
                preparar(px)
            r = await cli.post(px.url + "/v1/messages", content=cuerpo)
            return r.status_code, r.headers, list(up.recibidas)

    return _correr(escenario())


_CLAVES_DE_PENSAMIENTO_VARIANTES = ["Thinking", "THINKING", "thinKing", "THINK", "Reasoning_Effort"]


@pytest.mark.parametrize("clave", _CLAVES_DE_PENSAMIENTO_VARIANTES + ["thinking", "think", "reasoning_effort"])
def test_la_reescritura_borra_toda_clave_de_pensamiento_por_casefold(clave):
    sucio = json.dumps({"model": "m", "max_tokens": 5, clave: "high"}).encode()
    assert json.loads(proxy_carril._con_pensamiento_apagado(sucio)) == {
        "model": "m", "max_tokens": 5, "thinking": _DESHABILITADO}


@pytest.mark.parametrize("extra", [{"think": True}, {"reasoning_effort": "high"},
                                   {"think": True, "reasoning_effort": "high",
                                    "thinking": {"type": "enabled", "budget_tokens": 9}}])
def test_apagado_think_y_reasoning_effort_no_llegan_al_upstream(tmp_path, extra):
    estado, _, recibidas = _posteo(tmp_path, "apagado", _mensaje_p(**extra))
    assert estado == 200
    enviado = json.loads(recibidas[0][3])
    assert enviado["thinking"] == _DESHABILITADO and "think" not in enviado and "reasoning_effort" not in enviado


@pytest.mark.parametrize("pensamiento", ["apagado", "libre"])
@pytest.mark.parametrize("clave", _CLAVES_DE_PENSAMIENTO_VARIANTES)
def test_variante_de_capitalizacion_de_pensamiento_es_ambigua_y_no_llega(tmp_path, pensamiento, clave):
    estado, _, recibidas = _posteo(tmp_path, pensamiento, _mensaje_p(**{clave: "high"}))
    assert estado == 403 and recibidas == []


@pytest.mark.parametrize("pensamiento", ["apagado", "libre"])
@pytest.mark.parametrize("cuerpo", [
    b'{"model":"modelo-permitido","Model":"otro","max_tokens":1024,"messages":[]}',
    b'{"Model":"otro","max_tokens":1024,"messages":[]}',
    b'{"model":"modelo-permitido","Max_Tokens":999999,"messages":[]}',
    b'{"model":"modelo-permitido","max_tokens":1024,"Messages":[{"role":"user","content":"x"}]}',
    b'{"model":"modelo-permitido","max_tokens":1024,"max_tokens":1024,"messages":[]}',
    b'{"model":"modelo-permitido","max_tokens":1024,"SYSTEM":"x","messages":[]}',
    b'{"model":"modelo-permitido","max_tokens":1024,"Foo":1,"foo":2,"messages":[]}',
])
def test_claves_que_go_leeria_distinto_dan_403_y_no_llegan(tmp_path, pensamiento, cuerpo):
    estado, _, recibidas = _posteo(tmp_path, pensamiento, cuerpo)
    assert estado == 403 and recibidas == []


def test_claves_ajenas_en_otra_capitalizacion_sin_choque_si_pasan(tmp_path):
    cuerpo = b'{"model":"modelo-permitido","max_tokens":1024,"messages":[],"Extra":1}'
    estado, _, recibidas = _posteo(tmp_path, "libre", cuerpo)
    assert estado == 200 and recibidas[0][3] == cuerpo


def test_surrogate_suelto_con_apagado_se_reenvia_y_no_da_502(tmp_path):
    cuerpo = (b'{"model":"modelo-permitido","max_tokens":1024,"messages":[{"role":"user","content":'
              b'[{"type":"tool_result","tool_use_id":"t1","content":"\\ud83d"}]}]}')
    estado, _, recibidas = _posteo(tmp_path, "apagado", cuerpo)
    assert estado == 200 and len(recibidas) == 1
    enviado = json.loads(recibidas[0][3])
    assert enviado["thinking"] == _DESHABILITADO
    assert enviado["messages"][0]["content"][0]["content"] == "\ud83d"


def test_502_de_reescritura_lleva_x_should_retry_false(tmp_path, monkeypatch):
    def roto(cuerpo):
        raise ValueError("no")

    monkeypatch.setattr(proxy_carril, "_con_pensamiento_apagado", roto)
    estado, cabeceras, recibidas = _posteo(tmp_path, "apagado", _mensaje_p())
    assert (estado, cabeceras.get("x-should-retry"), recibidas) == (502, "false", [])


def test_apagado_con_la_pausa_puesta_423_sin_reenviar(tmp_path):
    from jax.ejecutor.contratos import pausa as P
    estado, cabeceras, recibidas = _posteo(
        tmp_path, "apagado", _mensaje_p(),
        preparar=lambda px: P.poner_pausa(px.cfg.pausa, {"origen": "c5", "motivo": "x", "paso": 1}))
    assert (estado, cabeceras.get("x-should-retry"), recibidas) == (423, "false", [])


def test_apagado_sin_latido_del_vigia_423_sin_reenviar(tmp_path):
    estado, _, recibidas = _posteo(tmp_path, "apagado", _mensaje_p(),
                                   preparar=lambda px: px.cfg.latido.unlink())
    assert (estado, recibidas) == (423, [])


def _con_resultado(tool_use_id="toolu_1"):
    return json.dumps({"model": MODELO_PERMITIDO, "max_tokens": MAX_SALIDA_TOKENS, "messages": [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": "ok"}]}]}).encode()


def _eventos(tmp_path):
    return [json.loads(l)["evento"] for l in (tmp_path / "registro.jsonl").read_text().splitlines()]


def test_c3_con_apagado_anota_el_resultado_una_sola_vez(tmp_path):
    async def escenario():
        async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 2, pensamiento="apagado") as px, \
                httpx.AsyncClient() as cli:
            return [(await cli.post(px.url + "/v1/messages", content=_con_resultado())).status_code for _ in range(2)]

    assert _correr(escenario()) == [200, 200]
    assert _eventos(tmp_path).count("resultado_devuelto") == 1


def test_c3_un_502_de_reescritura_no_duplica_la_anotacion_al_reintentar(tmp_path, monkeypatch):
    original = proxy_carril._con_pensamiento_apagado
    llamadas = []

    def falla_la_primera(cuerpo):
        llamadas.append(1)
        if len(llamadas) == 1:
            raise ValueError("falla una vez")
        return original(cuerpo)

    monkeypatch.setattr(proxy_carril, "_con_pensamiento_apagado", falla_la_primera)

    async def escenario():
        async with Upstream(n_trozos=1) as up, Proxy(up.url, tmp_path, 2, pensamiento="apagado") as px, \
                httpx.AsyncClient() as cli:
            return [(await cli.post(px.url + "/v1/messages", content=_con_resultado())).status_code for _ in range(2)]

    assert _correr(escenario()) == [502, 200]
    assert _eventos(tmp_path).count("resultado_devuelto") == 1


def test_c3_ve_lo_mismo_que_la_validacion_un_pedido_ambiguo_no_se_anota():
    from jax.ejecutor.contratos import lectura
    ambiguo = b'{"messages":[],"Messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":"x"}]}]}'
    assert lectura.resultados_de_peticion(ambiguo) is None
    with pytest.raises(lectura.PedidoAmbiguo):
        lectura.cargar_pedido(ambiguo)
