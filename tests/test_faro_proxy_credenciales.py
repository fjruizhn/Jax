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
    def __init__(self):
        self.recibidas = []

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
        writer.write(conn.send(h11.Response(status_code=200, headers=[("set-cookie", "sesion=secreta"), ("content-length", "2")])))
        writer.write(conn.send(h11.Data(data=b"ok")))
        writer.write(conn.send(h11.EndOfMessage()))
        await writer.drain()
        writer.close()


def _cfg(tmp_path: Path, upstream: str, **extra) -> ConfigProxyCredenciales:
    token = tmp_path / "token"
    token.write_text("token-prueba\n", encoding="utf-8")
    os.chmod(token, 0o400)
    return ConfigProxyCredenciales(
        ruta_socket=tmp_path / "run.proxy.sock",
        ruta_token=token,
        uid_esperado=os.geteuid(),
        solo_pruebas_mismo_uid=True,
        perfil=PerfilCredencial("claude", upstream, "/v1/messages", "clave-real-de-prueba", frozenset({"POST"}), frozenset({"modelo-prueba"})),
        **extra,
    )


def _correr(coro):
    return asyncio.run(asyncio.wait_for(coro, 10))


async def _pedir(proxy, *, modelo: str, cabeceras=()):
    reader, writer = await asyncio.open_unix_connection(str(proxy.ruta_socket))
    conn = h11.Connection(h11.CLIENT)
    cuerpo = (f'{{"model":"{modelo}"}}').encode()
    writer.write(b"FARO-TOKEN token-prueba\n")
    writer.write(conn.send(h11.Request(method="POST", target="/v1/messages", headers=[("host", "local"), ("content-type", "application/json"), ("content-length", str(len(cuerpo))), *cabeceras])))
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


def test_no_reenvia_cookie_autorizacion_ni_devuelve_set_cookie(tmp_path):
    async def escenario():
        async with Upstream() as up:
            async with ServidorProxyCredenciales(_cfg(tmp_path, up.url)) as proxy:
                r, _ = await _pedir(proxy, modelo="modelo-prueba", cabeceras=[("cookie", "cliente=secreto"), ("authorization", "Bearer ajena")])
                return r, up.recibidas[0]
    respuesta, req = _correr(escenario())
    cabeceras = dict(req.headers)
    assert respuesta.status_code == 200
    assert b"set-cookie" not in dict(respuesta.headers)
    assert b"cookie" not in cabeceras
    assert cabeceras[b"authorization"] == b"Bearer clave-real-de-prueba"

