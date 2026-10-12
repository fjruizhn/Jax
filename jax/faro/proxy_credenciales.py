"""Proxy Unix por ejecución para S-1.

El único extremo que escucha es un socket Unix propio de la ejecución. El
kernel identifica al par antes de leer bytes y un token de una sola ejecución
es la segunda barrera. El perfil, incluido destino y credencial, es inmutable.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import socket
import stat
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import h11
import httpx

from .config import ConfigFaroInvalida
from .perfiles_credenciales import PerfilCredencial

_PEERCRED = struct.Struct("3i")


def _credenciales_del_par(writer: asyncio.StreamWriter) -> tuple[int, int, int] | None:
    sock = writer.get_extra_info("socket")
    if sock is None:
        return None
    try:
        return _PEERCRED.unpack(sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _PEERCRED.size))
    except (OSError, struct.error):
        return None

_MAX_CABECERA = 64 * 1024
_MAX_CUERPO = 1024 * 1024
_HOP = {b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization", b"te", b"trailer", b"transfer-encoding", b"upgrade"}
_SECRETOS = _HOP | {b"cookie", b"set-cookie", b"authorization"}


@dataclass(frozen=True)
class ConfigProxyCredenciales:
    ruta_socket: Path
    ruta_token: Path
    uid_esperado: int
    perfil: PerfilCredencial
    solo_pruebas_mismo_uid: bool = False
    freno: Callable[[], bool] = field(default=lambda: False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.ruta_socket.is_absolute() or not self.ruta_token.is_absolute():
            raise ConfigFaroInvalida("socket y token del proxy deben ser rutas absolutas")
        if self.uid_esperado in (0, os.geteuid()) and not self.solo_pruebas_mismo_uid:
            raise ConfigFaroInvalida("el uid del motor debe diferir del servicio del proxy")


class ServidorProxyCredenciales:
    def __init__(self, cfg: ConfigProxyCredenciales):
        self.cfg = cfg
        self.ruta_socket = cfg.ruta_socket
        self._token = b""
        self._server: asyncio.AbstractServer | None = None
        self._client: httpx.AsyncClient | None = None
        self._tasks: set[asyncio.Task] = set()

    async def __aenter__(self):
        self._leer_token()
        if os.path.lexists(self.ruta_socket):
            raise FileExistsError(self.ruta_socket)
        self._client = httpx.AsyncClient(trust_env=False, follow_redirects=False)
        try:
            self._server = await asyncio.start_unix_server(self._atender, path=str(self.ruta_socket), limit=_MAX_CABECERA)
            os.chmod(self.ruta_socket, 0o600)
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, *_):
        if self._server is not None:
            self._server.close()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._server is not None:
            await self._server.wait_closed()
            self._server = None
        try:
            st = os.lstat(self.ruta_socket)
            if stat.S_ISSOCK(st.st_mode):
                os.unlink(self.ruta_socket)
        except FileNotFoundError:
            pass
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._token = b""

    def _leer_token(self) -> None:
        try:
            st = os.lstat(self.cfg.ruta_token)
            if not stat.S_ISREG(st.st_mode) or st.st_mode & 0o022:
                raise ConfigFaroInvalida("token del proxy inseguro")
            with open(self.cfg.ruta_token, "rb") as f:
                self._token = f.readline(1024).strip()
        except OSError as exc:
            raise ConfigFaroInvalida("no se puede leer token del proxy") from exc
        if not self._token:
            raise ConfigFaroInvalida("token del proxy vacio")

    async def _respuesta(self, conn, writer, status: int, body: bytes = b""):
        writer.write(conn.send(h11.Response(status_code=status, headers=[("content-length", str(len(body))), ("connection", "close")])))
        if body:
            writer.write(conn.send(h11.Data(data=body)))
        writer.write(conn.send(h11.EndOfMessage()))
        await writer.drain()

    async def _leer(self, conn, reader):
        req, body = None, bytearray()
        while True:
            event = conn.next_event()
            if event is h11.NEED_DATA:
                data = await reader.read(65536)
                if not data:
                    conn.receive_data(b"")
                else:
                    conn.receive_data(data)
            elif isinstance(event, h11.Request):
                req = event
            elif isinstance(event, h11.Data):
                body += event.data
                if len(body) > _MAX_CUERPO:
                    raise ValueError("cuerpo excedido")
            elif isinstance(event, h11.EndOfMessage):
                return req, bytes(body)
            elif isinstance(event, h11.ConnectionClosed):
                raise ValueError("conexion cerrada")

    def _validar(self, req, body: bytes) -> None:
        if req is None or req.method.decode().upper() not in self.cfg.perfil.metodos:
            raise PermissionError("metodo")
        target = req.target.decode("ascii", "strict")
        if target != self.cfg.perfil.ruta or "://" in target or "?" in target or "#" in target:
            raise PermissionError("ruta")
        headers = {k.lower(): v for k, v in req.headers}
        if b"upgrade" in headers or b"connection" in headers or b"proxy-authorization" in headers:
            raise PermissionError("upgrade")
        try:
            document = json.loads(body)
        except (TypeError, ValueError) as exc:
            raise PermissionError("json") from exc
        if not isinstance(document, dict) or document.get("model") not in self.cfg.perfil.modelos:
            raise PermissionError("modelo")

    async def _reenviar(self, conn, writer, req, body: bytes) -> None:
        self._validar(req, body)
        if self.cfg.freno():
            raise RuntimeError("freno")
        assert self._client is not None
        headers = [(k.decode(), v.decode()) for k, v in req.headers if k.lower() not in _SECRETOS and k.lower() != b"host"]
        headers.append(("authorization", f"Bearer {self.cfg.perfil.credencial}"))
        request = self._client.build_request(req.method.decode(), self.cfg.perfil.upstream + self.cfg.perfil.ruta, headers=headers, content=body)
        response = await self._client.send(request, stream=True)
        if 300 <= response.status_code < 400:
            await response.aclose()
            await self._respuesta(conn, writer, 502)
            return
        response_headers = [(k, v) for k, v in response.headers.multi_items() if k.lower().encode() not in _SECRETOS]
        response_headers.extend([("connection", "close")])
        writer.write(conn.send(h11.Response(status_code=response.status_code, headers=response_headers)))
        try:
            async for chunk in response.aiter_raw():
                writer.write(conn.send(h11.Data(data=chunk)))
                await writer.drain()
            writer.write(conn.send(h11.EndOfMessage()))
            await writer.drain()
        finally:
            await response.aclose()

    async def _atender(self, reader, writer) -> None:
        task = asyncio.current_task()
        self._tasks.add(task)
        conn = h11.Connection(h11.SERVER)
        try:
            cred = _credenciales_del_par(writer)
            if cred is None or cred[1] != self.cfg.uid_esperado:
                return
            raw_token = await asyncio.wait_for(reader.readline(), 5)
            if not hmac.compare_digest(raw_token, b"FARO-TOKEN " + self._token + b"\n"):
                return
            req, body = await self._leer(conn, reader)
            try:
                await self._reenviar(conn, writer, req, body)
            except PermissionError:
                await self._respuesta(conn, writer, 403)
            except RuntimeError as exc:
                if str(exc) == "freno":
                    await self._respuesta(conn, writer, 423)
                else:
                    raise
            except (httpx.HTTPError, OSError):
                await self._respuesta(conn, writer, 502)
        except (asyncio.CancelledError, ConnectionError, h11.ProtocolError):
            raise
        finally:
            self._tasks.discard(task)
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
