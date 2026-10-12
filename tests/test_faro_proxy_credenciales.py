"""Contrato de seguridad del proxy de credenciales de Fase 0.5.

Estas pruebas usan un upstream local deliberadamente: prueban que las peticiones
rechazadas no lo alcanzan y que el secreto nunca vuelve a la jaula.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import h11
import pytest

from jax.faro.perfiles_credenciales import PerfilCredencial, perfil_para_motor
from jax.faro.proxy_credenciales import ConfigProxyCredenciales, ServidorProxyCredenciales


class Upstream:
    def __init__(self, status=200):
        self.recibidas = []
        self.status = status

    async def __aenter__(self):
        self.server = await asyncio.start_server(self._atender, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *_):
        self.server.close()
        await self.server.wait_closed()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"

    async def _atender(self, reader, writer):
        conn = h11.Connection(h11.SERVER)
        req = None
        while True:
            ev = conn.next_event()
            if ev is h11.NEED_DATA:
                conn.receive_data(await reader.read(65536))
            elif isinstance(ev, h11.Request):
                req = ev
            elif isinstance(ev, (h11.EndOfMessage, h11.ConnectionClosed)):
                break
        self.recibidas.append(req)
        writer.write(conn.send(h11.Response(status_code=self.status, headers=[("set-cookie", "sesion=secreta"), ("location", "https://no-debe-seguirse.invalid"), ("content-length", "2")])))
        writer.write(conn.send(h11.Data(data=b"ok")))
        writer.write(conn.send(h11.EndOfMessage()))
        await writer.drain()
        writer.close()


def _cfg(tmp_path: Path, upstream: str, **extra) -> ConfigProxyCredenciales:
    tmp_path.mkdir(exist_ok=True)
    token = tmp_path / "token"
    token.write_text("token-prueba\n", encoding="utf-8")
    os.chmod(token, 0o400)
    uid = extra.pop("uid_esperado", os.geteuid())
    mismo_uid = extra.pop("solo_pruebas_mismo_uid", True)
    return ConfigProxyCredenciales(
        ruta_socket=tmp_path / "run.proxy.sock",
        ruta_token=token,
        uid_esperado=uid,
        solo_pruebas_mismo_uid=mismo_uid,
        perfil=PerfilCredencial("claude", upstream, "/v1/messages", "clave-real-de-prueba", frozenset({"POST"}), frozenset({"modelo-prueba"})),
        **extra,
    )


def _correr(coro):
    return asyncio.run(asyncio.wait_for(coro, 10))


async def _pedir(proxy, *, modelo: str, cabeceras=(), metodo="POST", destino="/v1/messages", token="token-prueba"):
    reader, writer = await asyncio.open_unix_connection(str(proxy.ruta_socket))
    conn = h11.Connection(h11.CLIENT)
    cuerpo = (f'{{"model":"{modelo}"}}').encode()
    writer.write(b"FARO-TOKEN " + token.encode() + b"\n")
    writer.write(conn.send(h11.Request(method=metodo, target=destino, headers=[("host", "faro-proxy"), ("content-type", "application/json"), ("content-length", str(len(cuerpo))), *cabeceras])))
    writer.write(conn.send(h11.Data(data=cuerpo)))
    writer.write(conn.send(h11.EndOfMessage()))
    await writer.drain()
    respuesta, cuerpo = None, b""
    while True:
        ev = conn.next_event()
        if ev is h11.NEED_DATA:
            conn.receive_data(await reader.read(65536))
        elif isinstance(ev, h11.Response):
            respuesta = ev
        elif isinstance(ev, h11.Data):
            cuerpo += ev.data
        elif isinstance(ev, (h11.EndOfMessage, h11.ConnectionClosed)):
            writer.close()
            await writer.wait_closed()
            return respuesta, cuerpo


def test_codex_no_tiene_adaptador_y_falla_cerrado():
    with pytest.raises(ValueError, match="NO_S1"):
        perfil_para_motor("codex", "https://ejemplo.invalid")


def test_rechaza_modelo_ajeno_antes_de_contactar_upstream(tmp_path):
    async def escenario():
        async with Upstream() as up:
            async with ServidorProxyCredenciales(_cfg(tmp_path, up.url)) as proxy:
                r, _ = await _pedir(proxy, modelo="otro")
                return r.status_code, len(up.recibidas)
    assert _correr(escenario()) == (403, 0)


def test_freno_puesto_no_contacta_upstream(tmp_path):
    async def escenario():
        async with Upstream() as up:
            async with ServidorProxyCredenciales(_cfg(tmp_path, up.url, freno=lambda: True)) as proxy:
                r, _ = await _pedir(proxy, modelo="modelo-prueba")
                return r.status_code, len(up.recibidas)
    assert _correr(escenario()) == (423, 0)


@pytest.mark.parametrize("metodo,destino,cabeceras", [
    ("CONNECT", "/v1/messages", ()),
    ("POST", "/otra", ()),
    ("POST", "http://ajeno.invalid/v1/messages", ()),
    ("POST", "/v1/messages", (("upgrade", "websocket"),)),
    ("POST", "/v1/messages", (("forwarded", "host=ajeno.invalid"),)),
])
def test_pedidos_de_escape_se_rechazan_antes_del_upstream(tmp_path, metodo, destino, cabeceras):
    async def escenario():
        async with Upstream() as up:
            async with ServidorProxyCredenciales(_cfg(tmp_path, up.url)) as proxy:
                r, _ = await _pedir(proxy, modelo="modelo-prueba", metodo=metodo, destino=destino, cabeceras=cabeceras)
                return r.status_code, len(up.recibidas)
    assert _correr(escenario()) == (403, 0)


def test_redirect_del_upstream_no_revela_location(tmp_path):
    async def escenario():
        async with Upstream(status=302) as up:
            async with ServidorProxyCredenciales(_cfg(tmp_path, up.url)) as proxy:
                r, _ = await _pedir(proxy, modelo="modelo-prueba")
                return r.status_code, dict(r.headers), len(up.recibidas)
    estado, cabeceras, n = _correr(escenario())
    assert (estado, n) == (502, 1)
    assert b"location" not in cabeceras


def test_respuesta_mayor_al_limite_no_sale_de_proxy(tmp_path):
    async def escenario():
        async with Upstream() as up:
            async with ServidorProxyCredenciales(_cfg(tmp_path, up.url, max_respuesta_bytes=1)) as proxy:
                r, _ = await _pedir(proxy, modelo="modelo-prueba")
                return r.status_code, len(up.recibidas)
    assert _correr(escenario()) == (502, 1)


def test_cierre_no_borra_socket_reemplazado(tmp_path):
    async def escenario():
        async with Upstream() as up:
            proxy = ServidorProxyCredenciales(_cfg(tmp_path, up.url))
            await proxy.__aenter__()
            os.unlink(proxy.ruta_socket)
            reemplazo = await asyncio.start_unix_server(lambda r, w: None, path=str(proxy.ruta_socket))
            try:
                await proxy.__aexit__(None, None, None)
                return os.path.lexists(proxy.ruta_socket)
            finally:
                reemplazo.close()
                await reemplazo.wait_closed()
                if os.path.lexists(proxy.ruta_socket):
                    os.unlink(proxy.ruta_socket)
    assert _correr(escenario()) is True


def test_token_de_otra_ejecucion_no_cruza_hacia_upstream(tmp_path):
    async def escenario():
        async with Upstream() as up:
            cfg_a = _cfg(tmp_path / "a", up.url)
            os.chmod(cfg_a.ruta_token, 0o600)
            cfg_a.ruta_token.write_text("a\n")
            os.chmod(cfg_a.ruta_token, 0o400)
            cfg_b = _cfg(tmp_path / "b", up.url)
            os.chmod(cfg_b.ruta_token, 0o600)
            cfg_b.ruta_token.write_text("b\n")
            os.chmod(cfg_b.ruta_token, 0o400)
            async with ServidorProxyCredenciales(cfg_a) as proxy_a, ServidorProxyCredenciales(cfg_b) as proxy_b:
                with pytest.raises((ConnectionResetError, BrokenPipeError, h11.RemoteProtocolError)):
                    await _pedir(proxy_a, modelo="modelo-prueba", token="b")
                return len(up.recibidas)
    assert _correr(escenario()) == 0


def test_uid_distinto_se_cierra_antes_de_leer_token_o_contactar_upstream(tmp_path):
    async def escenario():
        async with Upstream() as up:
            cfg = _cfg(tmp_path, up.url, uid_esperado=os.geteuid() + 1, solo_pruebas_mismo_uid=False)
            async with ServidorProxyCredenciales(cfg) as proxy:
                reader, writer = await asyncio.open_unix_connection(str(proxy.ruta_socket))
                writer.write(b"FARO-TOKEN token-prueba\nPOST /v1/messages HTTP/1.1\r\n\r\n")
                await writer.drain()
                try:
                    datos = await reader.read(1)
                except ConnectionResetError:
                    datos = b""
                writer.close()
                return datos, len(up.recibidas)
    assert _correr(escenario()) == (b"", 0)


def test_no_reenvia_cookie_autorizacion_ni_devuelve_set_cookie(tmp_path):
    async def escenario():
        async with Upstream() as up:
            async with ServidorProxyCredenciales(_cfg(tmp_path, up.url)) as proxy:
                r, _ = await _pedir(proxy, modelo="modelo-prueba", cabeceras=[("cookie", "cliente=secreto"), ("authorization", "Bearer ajena"), ("x-api-key", "ajena")])
                return r, up.recibidas[0]
    respuesta, req = _correr(escenario())
    cabeceras = dict(req.headers)
    assert respuesta.status_code == 200
    assert b"set-cookie" not in dict(respuesta.headers)
    assert b"cookie" not in cabeceras
    assert b"x-api-key" not in cabeceras
    assert cabeceras[b"authorization"] == b"Bearer clave-real-de-prueba"

