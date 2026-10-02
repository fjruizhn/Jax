"""El transporte del Puerto: un socket Unix por ejecucion, `<dir>/<run_id>.sock`, y la
identidad que sale DEL SOCKET (spec §3 «Transporte»).

- El servicio que crea la ejecucion construye una `Ejecucion` (usuario, tenant, faceta, motor,
  pipeline, run_id, entry_point, uid de la jaula) y abre el socket de ESE run. Lo que dicen las
  credenciales del par (`SO_PEERCRED`, las pone el kernel, no el cliente) se compara con el uid
  esperado: otro uid se cierra sin hablar MCP y queda registrado. Dentro de una conexion
  aceptada, la identidad es la `Ejecucion` + las credenciales del par; el cuerpo de los pedidos
  no aporta ninguna.
- El socket se crea en un directorio del usuario del servicio, sin escritura de grupo/otros, y
  con modo 0600 (si el servicio corre como root y la jaula usa otro uid, el socket se entrega a ese
  uid). Un `run_id` que ya tiene socket NO se pisa. Al cerrar, el socket se borra.
- Las corrientes son `asyncio` puras (nada bloqueante); se adaptan a lo que espera el helper
  oficial `mcp.server.stdio.stdio_server(stdin=, stdout=)`, que acepta corrientes explicitas.
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import stat
import struct
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from mcp.server.stdio import stdio_server

from . import freno as _freno
from .bitacora import Bitacora
from .config import ConfigFaroInvalida, ConfigPuerto
from .identidad import Ejecucion, Identidad
from .paquete import PaqueteCargado
from .puerto import construir_servidor, servidor_de_bajo_nivel

logger = logging.getLogger(__name__)

_PEERCRED = struct.Struct("3i")


def credenciales_del_par(escritor: asyncio.StreamWriter) -> tuple[int, int, int] | None:
    """(pid, uid, gid) del otro extremo, segun el kernel (`SO_PEERCRED`); None si no se pueden leer."""
    sock = escritor.get_extra_info("socket")
    if sock is None:
        return None
    try:
        return _PEERCRED.unpack(sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _PEERCRED.size))
    except (OSError, struct.error):
        return None


class _Lineas:
    """Lo que `stdio_server` lee: `async for linea in stdin`. Un mensaje mas largo que el limite
    cierra la conexion (fallo cerrado), nunca se trunca."""

    def __init__(self, lector: asyncio.StreamReader):
        self._lector = lector

    async def __aiter__(self) -> AsyncIterator[str]:
        while True:
            try:
                linea = await self._lector.readline()
            except ValueError:
                logger.warning("mensaje MCP mas largo que el limite configurado: se cierra la conexion")
                return
            if not linea:
                return
            yield linea.decode("utf-8", errors="replace")


class _Salida:
    """Lo que `stdio_server` escribe: `await stdout.write(texto)` y `await stdout.flush()`."""

    def __init__(self, escritor: asyncio.StreamWriter):
        self._escritor = escritor

    async def write(self, texto: str) -> None:
        self._escritor.write(texto.encode("utf-8"))

    async def flush(self) -> None:
        await self._escritor.drain()


class ServidorPuerto:
    def __init__(self, cfg: ConfigPuerto, ejecucion: Ejecucion, paquete: PaqueteCargado, bitacora: Bitacora,
                 *, freno: Callable[[], bool] | None = None):
        self._cfg = cfg
        self.ejecucion = ejecucion
        self._paquete = paquete
        self._bitacora = bitacora
        self._freno = freno if freno is not None else _freno.puesto
        self.ruta_socket: Path = Path(cfg.socket_dir) / f"{ejecucion.run_id}.sock"
        self._servidor: asyncio.AbstractServer | None = None
        self._tareas: set[asyncio.Task] = set()

    def _validar_directorio(self) -> None:
        d = Path(self._cfg.socket_dir)
        try:
            st = os.lstat(d)
        except OSError as exc:
            raise ConfigFaroInvalida(f"el directorio de sockets {d} no se puede leer: {type(exc).__name__}") from exc
        if not stat.S_ISDIR(st.st_mode):
            raise ConfigFaroInvalida(f"{d} no es un directorio")
        if st.st_uid != os.geteuid():
            raise ConfigFaroInvalida(f"{d} tiene que ser del usuario del servicio (uid {os.geteuid()})")
        if st.st_mode & 0o022:
            raise ConfigFaroInvalida(f"{d} no puede tener escritura de grupo ni de otros (modo {stat.S_IMODE(st.st_mode):o})")

    async def __aenter__(self) -> "ServidorPuerto":
        self._validar_directorio()
        if os.path.lexists(self.ruta_socket):
            raise FileExistsError(f"ya hay un socket para el run {self.ejecucion.run_id}: {self.ruta_socket}")
        self._servidor = await asyncio.start_unix_server(self._atender, path=str(self.ruta_socket),
                                                         limit=self._cfg.max_mensaje)
        try:
            os.chmod(self.ruta_socket, 0o600)
            if os.geteuid() == 0 and self.ejecucion.uid_esperado != 0:
                os.chown(self.ruta_socket, self.ejecucion.uid_esperado, -1)
        except OSError:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, *_exc) -> None:
        if self._servidor is not None:
            self._servidor.close()
        tareas = list(self._tareas)
        for t in tareas:
            t.cancel()
        await asyncio.gather(*tareas, return_exceptions=True)
        if self._servidor is not None:
            await self._servidor.wait_closed()
            self._servidor = None
        try:
            os.unlink(self.ruta_socket)
        except FileNotFoundError:  # fail-soft: el socket ya no esta (alguien lo borro antes); el objetivo del cierre es justamente que no exista
            pass

    async def _atender(self, lector: asyncio.StreamReader, escritor: asyncio.StreamWriter) -> None:
        tarea = asyncio.current_task()
        self._tareas.add(tarea)
        id_conexion = uuid.uuid4().hex
        try:
            cred = credenciales_del_par(escritor)
            e = self.ejecucion
            if cred is None or cred[1] != e.uid_esperado:
                self._bitacora.registrar(
                    "conexion_rechazada", id_conexion=id_conexion, run_id=e.run_id, id_correlacion=e.id_correlacion,
                    entry_point=e.entry_point, sha_paquete=self._paquete.sha,
                    peer_pid=cred[0] if cred else -1, peer_uid=cred[1] if cred else -1,
                    motivo="par_sin_credenciales" if cred is None else "uid_distinto_del_esperado")
                return
            identidad = Identidad(e, id_conexion, peer_pid=cred[0], peer_uid=cred[1], peer_gid=cred[2])
            mcp = construir_servidor(self._paquete, identidad=identidad, bitacora=self._bitacora, freno=self._freno)
            bajo = servidor_de_bajo_nivel(mcp)
            async with stdio_server(stdin=_Lineas(lector), stdout=_Salida(escritor)) as (leer, escribir):
                await bajo.run(leer, escribir, bajo.create_initialization_options())
        except asyncio.CancelledError:
            raise
        except Exception:  # fail-closed: la conexion se cierra en el finally (nada se sirve mas por ella) y el error queda en el log
            logger.exception("la conexion %s del run %s termino con error", id_conexion, self.ejecucion.run_id)
        finally:
            self._tareas.discard(tarea)
            escritor.close()
            try:
                await escritor.wait_closed()
            except OSError:  # fail-soft: el par ya cerro o reseteo la conexion; la conexion ya se da por terminada y no hay nada que recuperar
                pass
