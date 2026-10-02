"""El transporte del Puerto: un socket Unix por ejecucion, `<dir>/<run_id>.sock`, y la
identidad que sale DEL SOCKET (spec §3 «Transporte»).

QUIEN PUEDE HABLAR (auditoria MAJOR-2/3)
- El servicio corre como usuario `faro` SIN privilegios: con euid 0 no arranca. No hace `chown`.
- El directorio de sockets es del usuario del servicio y esta en 0700 (cualquier permiso de grupo u
  otros lo rechaza). A la jaula NO se le monta el directorio: se le entrega por BIND DE ARCHIVO solo su
  `<run_id>.sock` y su `<run_id>.token`. Como el socket tiene que poder conectarlo el uid de la jaula
  (que no es el del servicio), su modo es 0666; dentro de un directorio 0700 solo llega quien lo
  recibio por bind. El uid de la jaula por ejecucion queda para la fase 1.
- Defensa en profundidad para que otra ejecucion del MISMO uid no pueda suplantar a la dueña: un token
  aleatorio por ejecucion (`secrets.token_urlsafe(32)`), en un archivo 0444 del mismo directorio (la
  jaula dueña lo recibe por bind), que el rele manda como primera linea (`FARO-TOKEN <token>\\n`). Se
  compara en tiempo constante; sin el, no se habla MCP. Nunca se registra.
- Orden de comprobaciones por conexion: credenciales del par (`SO_PEERCRED`, las pone el kernel) ->
  token. Cada rechazo queda en la bitacora.

MEMORIA (auditoria MAJOR-4)
- `PresupuestoBytes`: presupuesto GLOBAL de bytes en vuelo. Un mensaje a medio leer cobra sus bytes
  (x3: cadena, decodificacion y analisis) hasta que se entrega; una respuesta cobra sus bytes hasta
  que el par la lee. Una conexion ociosa no cobra nada: NO HAY TOPE DE CONEXIONES (D-4); lo que el
  servicio retiene queda acotado por el presupuesto, no por cuantos hablan. Lo que no cabe espera
  (el par queda frenado por el kernel); un mensaje a medias o una escritura que el par no lee se cortan
  tras `mensaje_timeout_s` devolviendo lo que retenian. Un mensaje de mas de `max_mensaje` cierra la
  conexion.
- Las corrientes son `asyncio` puras (nada bloqueante) y se adaptan a lo que espera el helper
  oficial `mcp.server.stdio.stdio_server(stdin=, stdout=)`, que acepta corrientes explicitas.
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import os
import secrets
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
from .logs import asegurar_logging
from .paquete import PaqueteCargado
from .puerto import construir_servidor, servidor_de_bajo_nivel

logger = logging.getLogger(__name__)

_PEERCRED = struct.Struct("3i")
_TROZO = 16384            # lo que se lee de una vez: la memoria sin cobrar por conexion es de este orden
_FACTOR_COPIAS = 3        # un mensaje en vuelo ocupa ~3 copias (bytes, texto, objeto analizado)
_PREFIJO_HANDSHAKE = b"FARO-TOKEN "
MODO_SOCKET = 0o666       # ver el docstring: dentro de un directorio 0700
MODO_TOKEN = 0o444


class PresupuestoBytes:
    """Semaforo por bytes. `adquirir(n)` espera hasta que haya `n` libres (acotado al total: pedir
    mas que el total toma todo, no se cuelga) y devuelve lo cobrado; `liberar(n)` lo devuelve."""

    def __init__(self, total: int):
        self.total = total
        self._libre = total
        self._cond = asyncio.Condition()

    @property
    def libre(self) -> int:
        return self._libre

    async def adquirir(self, n: int) -> int:
        n = max(0, min(n, self.total))
        async with self._cond:
            await self._cond.wait_for(lambda: self._libre >= n)
            self._libre -= n
        return n

    async def liberar(self, n: int) -> None:
        n = max(0, min(n, self.total))
        async with self._cond:
            self._libre = min(self.total, self._libre + n)
            self._cond.notify_all()


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
    """Lo que `stdio_server` lee: `async for linea in stdin`. Lee por trozos chicos, cobra al presupuesto
    los bytes del mensaje a medias y los devuelve cuando el mensaje se entrega (o la conexion termina)."""

    def __init__(self, lector: asyncio.StreamReader, presupuesto: PresupuestoBytes, max_mensaje: int, plazo_s: float):
        self._lector, self._presupuesto, self._max, self._plazo = lector, presupuesto, max_mensaje, plazo_s

    async def __aiter__(self) -> AsyncIterator[str]:
        parcial = bytearray()
        cobrado = 0
        try:
            while True:
                try:
                    trozo = await asyncio.wait_for(self._lector.read(_TROZO), self._plazo if parcial else None)
                except TimeoutError:
                    logger.warning("mensaje MCP a medias sin completarse en %ss: se cierra la conexion", self._plazo)
                    return
                if not trozo:
                    return
                partes = trozo.split(b"\n")
                for i, parte in enumerate(partes):
                    if len(parcial) + len(parte) > self._max:
                        logger.warning("mensaje MCP mas largo que el limite configurado (%s bytes): se cierra la conexion", self._max)
                        return
                    if parte:
                        try:
                            cobrado += await asyncio.wait_for(self._presupuesto.adquirir(len(parte) * _FACTOR_COPIAS), self._plazo)
                        except TimeoutError:
                            logger.warning("sin presupuesto de memoria para el mensaje en %ss: se cierra la conexion", self._plazo)
                            return
                        parcial += parte
                    if i < len(partes) - 1:
                        linea = bytes(parcial) + b"\n"
                        parcial.clear()
                        yield linea.decode("utf-8", errors="replace")
                        await self._presupuesto.liberar(cobrado)
                        cobrado = 0
        finally:
            if cobrado:
                await self._presupuesto.liberar(cobrado)


class _Salida:
    """Lo que `stdio_server` escribe: `await stdout.write(texto)` y `await stdout.flush()`. Los bytes de
    una respuesta cobran al presupuesto hasta que el par los lee (o el plazo corta la conexion)."""

    def __init__(self, escritor: asyncio.StreamWriter, presupuesto: PresupuestoBytes, plazo_s: float):
        self._escritor, self._presupuesto, self._plazo = escritor, presupuesto, plazo_s
        self._cobrado = 0

    async def write(self, texto: str) -> None:
        datos = texto.encode("utf-8")
        try:
            self._cobrado += await asyncio.wait_for(self._presupuesto.adquirir(len(datos)), self._plazo)
        except TimeoutError:
            raise ConnectionError("sin presupuesto de memoria para la respuesta") from None
        self._escritor.write(datos)

    async def flush(self) -> None:
        try:
            await asyncio.wait_for(self._escritor.drain(), self._plazo)
        except TimeoutError:
            raise ConnectionError("el par no lee la respuesta") from None
        finally:
            await self.devolver()

    async def devolver(self) -> None:
        if self._cobrado:
            cobrado, self._cobrado = self._cobrado, 0
            await self._presupuesto.liberar(cobrado)


class ServidorPuerto:
    def __init__(self, cfg: ConfigPuerto, ejecucion: Ejecucion, paquete: PaqueteCargado, bitacora: Bitacora,
                 *, freno: Callable[[], bool] | None = None):
        self._cfg = cfg
        self.ejecucion = ejecucion
        self._paquete = paquete
        self._bitacora = bitacora
        self._freno = freno if freno is not None else _freno.puesto
        self.ruta_socket: Path = Path(cfg.socket_dir) / f"{ejecucion.run_id}.sock"
        self.ruta_token: Path = Path(cfg.socket_dir) / f"{ejecucion.run_id}.token"
        self.presupuesto = PresupuestoBytes(cfg.presupuesto_bytes)
        self._token = b""
        self._servidor: asyncio.AbstractServer | None = None
        self._tareas: set[asyncio.Task] = set()

    def _validar_entorno(self) -> None:
        if os.geteuid() == 0:
            raise ConfigFaroInvalida("el Puerto no corre como root: usa el usuario de servicio sin privilegios (`faro`)")
        d = Path(self._cfg.socket_dir)
        try:
            st = os.lstat(d)
        except OSError as exc:
            raise ConfigFaroInvalida(f"el directorio de sockets {d} no se puede leer: {type(exc).__name__}") from exc
        if not stat.S_ISDIR(st.st_mode):
            raise ConfigFaroInvalida(f"{d} no es un directorio")
        if st.st_uid != os.geteuid():
            raise ConfigFaroInvalida(f"{d} tiene que ser del usuario del servicio (uid {os.geteuid()})")
        if st.st_mode & 0o077:
            raise ConfigFaroInvalida(
                f"{d} tiene que ser privado (0700): ningun permiso de grupo ni de otros (tiene {stat.S_IMODE(st.st_mode):04o})")

    def _escribir_token(self) -> None:
        self._token = secrets.token_urlsafe(32).encode()
        fd = os.open(self.ruta_token, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(self._token + b"\n")
            os.fchmod(f.fileno(), MODO_TOKEN)

    async def __aenter__(self) -> "ServidorPuerto":
        self._validar_entorno()
        asegurar_logging()      # antes del primer MCPServer: el SDK configura el logging si nadie lo hizo
        if os.path.lexists(self.ruta_socket) or os.path.lexists(self.ruta_token):
            raise FileExistsError(f"ya hay un socket o un token para el run {self.ejecucion.run_id}: {self.ruta_socket}")
        self._escribir_token()
        try:
            self._servidor = await asyncio.start_unix_server(self._atender, path=str(self.ruta_socket), limit=_TROZO,
                                                             backlog=socket.SOMAXCONN)
            os.chmod(self.ruta_socket, MODO_SOCKET)
        except BaseException:
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
            try:
                await asyncio.wait_for(self._servidor.wait_closed(), 5)
            except TimeoutError:  # fail-soft: el apagado no puede colgarse por una conexion rara; ya se cancelaron todas las tareas
                logger.warning("el cierre del servidor no termino en 5s: se sigue con el apagado")
            self._servidor = None
        for ruta in (self.ruta_socket, self.ruta_token):
            try:
                os.unlink(ruta)
            except FileNotFoundError:  # fail-soft: ya no esta (alguien lo borro antes); el objetivo del cierre es justamente que no exista
                pass
        self._token = b""

    async def _rechazar(self, id_conexion: str, cred, motivo: str) -> None:
        e = self.ejecucion
        try:
            await self._bitacora.registrar(
                "conexion_rechazada", id_conexion=id_conexion, run_id=e.run_id, id_correlacion=e.id_correlacion,
                entry_point=e.entry_point, sha_paquete=self._paquete.sha,
                peer_pid=cred[0] if cred else -1, peer_uid=cred[1] if cred else -1, motivo=motivo)
        except Exception:  # fail-closed: la conexion se rechaza igual; solo la anotacion fallo y queda en el log
            logger.exception("no se pudo registrar el rechazo de la conexion %s", id_conexion)

    async def _handshake_valido(self, lector: asyncio.StreamReader) -> bool:
        esperado = _PREFIJO_HANDSHAKE + self._token + b"\n"
        try:
            recibido = await asyncio.wait_for(lector.readline(), self._cfg.handshake_s)
        except (TimeoutError, ValueError, ConnectionError):
            return False
        return hmac.compare_digest(recibido, esperado)

    async def _atender(self, lector: asyncio.StreamReader, escritor: asyncio.StreamWriter) -> None:
        tarea = asyncio.current_task()
        self._tareas.add(tarea)
        id_conexion = uuid.uuid4().hex
        salida = None
        try:
            cred = credenciales_del_par(escritor)
            e = self.ejecucion
            if cred is None or cred[1] != e.uid_esperado:
                await self._rechazar(id_conexion, cred, "par_sin_credenciales" if cred is None else "uid_distinto_del_esperado")
                return
            if not await self._handshake_valido(lector):
                await self._rechazar(id_conexion, cred, "token_invalido")
                return
            identidad = Identidad(e, id_conexion, peer_pid=cred[0], peer_uid=cred[1], peer_gid=cred[2])
            mcp = construir_servidor(self._paquete, identidad=identidad, bitacora=self._bitacora, freno=self._freno)
            bajo = servidor_de_bajo_nivel(mcp)
            salida = _Salida(escritor, self.presupuesto, self._cfg.mensaje_timeout_s)
            entrada = _Lineas(lector, self.presupuesto, self._cfg.max_mensaje, self._cfg.mensaje_timeout_s)
            async with stdio_server(stdin=entrada, stdout=salida) as (leer, escribir):
                await bajo.run(leer, escribir, bajo.create_initialization_options())
        except asyncio.CancelledError:
            raise
        except Exception:  # fail-closed: la conexion se cierra en el finally (nada se sirve mas por ella) y el error queda en el log
            logger.exception("la conexion %s del run %s termino con error", id_conexion, self.ejecucion.run_id)
        finally:
            self._tareas.discard(tarea)
            if salida is not None:
                await salida.devolver()
            escritor.close()
            try:
                await escritor.wait_closed()
            except OSError:  # fail-soft: el par ya cerro o reseteo la conexion; la conexion ya se da por terminada y no hay nada que recuperar
                pass
