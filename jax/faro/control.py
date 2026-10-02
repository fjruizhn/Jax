"""El canal de control del Faro (plan 0.3c, R3): QUIEN puede pedir crear (o cerrar) una `Ejecucion`.

El orquestador de jax-platform es quien decide usuario, tenant, faceta, motor, pipeline, `entry_point` y el uid de
la jaula; el MOTOR no decide nada de eso (spec §3 y §4: «la fija quien autentica, nunca el modelo»). Este modulo
es la unica puerta por la que nace una `Ejecucion`, y esta cerrada:

AUTENTICACION. Un socket Unix `control.sock` en `JAX_FARO_CONTROL_DIR` (distinto del directorio de sockets del
Puerto, que es 0700 y a la jaula ni se monta). La autenticacion es `SO_PEERCRED`, que pone el kernel: el uid del
par tiene que ser EXACTAMENTE `JAX_FARO_ORQUESTADOR_UID`, que nunca es root, ni el usuario del servicio, ni uno
de los uids de las jaulas. Un par que no lo es se cierra SIN LEER NADA de lo que mande. Un motor con el token de
su ejecucion tiene otro uid (el de su jaula) y el token no sirve aqui: el control no mira tokens.

ACCESO AL ARCHIVO. El plan decia «0600 del usuario faro»; un orquestador de OTRO uid no podria abrir un socket
0600 de `faro` (conectar exige escritura), asi que el socket es 0660 y su directorio no admite NADA para «otros»
(ni escritura ni lectura ni paso) ni escritura de grupo: solo `faro` y el grupo del directorio (al que se anade
el usuario del orquestador) llegan al archivo; la autenticacion real sigue siendo el uid del kernel. Es una
desviacion declarada del plan (ver «Desviaciones» alli).

EL PEDIDO. Una linea JSON. `crear` lleva SOLO `usuario, tenant, faceta, motor, pipeline, entry_point, uid_jaula`;
cualquier otro campo (un `run_id`, un `id_correlacion`, un `uid_esperado`...) se IGNORA: el `run_id` y el
`id_correlacion` los genera el Faro y el uid de la jaula sale de `uid_jaula`, validado contra el rango
configurado (`JAX_FARO_JAULA_UID_MIN/MAX`), distinto de root, del servicio y del orquestador, y de la jaula de
cualquier otra ejecucion viva. `cerrar` lleva `run_id`. No hay otra operacion.

BITACORA. Cada creacion y cada rechazo van a la bitacora (`control_creado`, `control_rechazado`; el cierre,
`control_cerrado`) y el `Avisador` (observador) los manda a Telegram. Si la bitacora durable no puede anotar una
creacion, la creacion se DESHACE: no hay ejecucion sin registro. Un rechazo se mantiene aunque no se pueda anotar.

D-4: no hay tope de agentes ni de ejecuciones concurrentes aqui; el unico limite es el rango de uids de jaula
(un recurso del sistema operativo, que se dimensiona grande) y la memoria del Puerto (su presupuesto).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pwd
import re
import socket
import stat
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .bitacora import Bitacora
from .config import ConfigFaroInvalida
from .identidad import RE_TENANT, Ejecucion
from .transporte import ServidorPuerto, credenciales_del_par

logger = logging.getLogger(__name__)

NOMBRE_SOCKET = "control.sock"
MODO_SOCKET_CONTROL = 0o660
PLAZO_S_POR_DEFECTO = 5.0
MAX_PEDIDO_POR_DEFECTO = 64 * 1024
CAMPOS_DE_IDENTIDAD = ("usuario", "tenant", "faceta", "motor", "pipeline", "entry_point")
_MAX_CAMPO = 200
_UID_MAX = 2 ** 32 - 2      # 2**32-1 es «sin uid» para el kernel


# Cuentas que ejecutan codigo guiado por modelos: jax-platform y LAS MANOS (`jaxsvc`), el ecosistema Axioma
# (`axioma`) y las sesiones de Fernando (`fruiz`). El orquestador que crea ejecuciones NO puede ser ninguna de
# ellas: quien controla el codigo de un modelo controlaria quien pide las ejecuciones. Es una lista NEGATIVA
# versionada; se resuelve por nombre con `pwd` (los uids cambian de una maquina a otra) y se puede EXTENDER con
# `JAX_FARO_ORQUESTADOR_PROHIBIDOS` (nombres separados por comas), nunca acortar. El orquestador debe ser una
# cuenta propia que no ejecute codigo de modelos (plan, 0.10).
CUENTAS_CON_CODIGO_DE_MODELOS = ("jaxsvc", "axioma", "fruiz")
_RE_CUENTA = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")


def _resolver_cuentas(env: Mapping[str, str], getpwnam: Callable = pwd.getpwnam) -> dict[int, str]:
    extra = [n.strip() for n in (env.get("JAX_FARO_ORQUESTADOR_PROHIBIDOS") or "").split(",") if n.strip()]
    for n in extra:
        if not _RE_CUENTA.fullmatch(n):
            raise ConfigFaroInvalida(f"JAX_FARO_ORQUESTADOR_PROHIBIDOS: {n!r} no es un nombre de cuenta valido")
    resueltas: dict[int, str] = {}
    for nombre in (*CUENTAS_CON_CODIGO_DE_MODELOS, *extra):
        try:
            resueltas.setdefault(int(getpwnam(nombre).pw_uid), nombre)
        except KeyError:
            continue                    # una cuenta que no existe en esta maquina no puede ser el orquestador
    return resueltas


def cuentas_prohibidas(env: Mapping[str, str], getpwnam: Callable = pwd.getpwnam) -> set[int]:
    """Los uids que no pueden ser orquestador ni estar en el rango de jaulas."""
    return set(_resolver_cuentas(env, getpwnam))


@dataclass(frozen=True)
class ConfigControl:
    """`JAX_FARO_CONTROL_DIR`, `JAX_FARO_ORQUESTADOR_UID`, `JAX_FARO_JAULA_UID_MIN` y `JAX_FARO_JAULA_UID_MAX` son
    obligatorias (falla cerrado). `JAX_FARO_CONTROL_PLAZO_S` y `JAX_FARO_CONTROL_MAX_PEDIDO` son opcionales.
    `solo_pruebas_mismo_uid` NO sale del entorno: es un argumento (como en `ServidorPuerto`)."""
    control_dir: Path
    orquestador_uid: int
    jaula_uid_min: int
    jaula_uid_max: int
    plazo_s: float = PLAZO_S_POR_DEFECTO
    max_pedido: int = MAX_PEDIDO_POR_DEFECTO
    solo_pruebas_mismo_uid: bool = False
    cuentas_con_codigo_de_modelos: tuple[tuple[int, str], ...] = ()      # (uid, nombre); lo llena desde_entorno

    def __post_init__(self) -> None:
        if not Path(self.control_dir).is_absolute():
            raise ConfigFaroInvalida(f"JAX_FARO_CONTROL_DIR tiene que ser una ruta absoluta, no {str(self.control_dir)!r}")
        for nombre, v in (("JAX_FARO_ORQUESTADOR_UID", self.orquestador_uid), ("JAX_FARO_JAULA_UID_MIN", self.jaula_uid_min),
                          ("JAX_FARO_JAULA_UID_MAX", self.jaula_uid_max)):
            if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= _UID_MAX:
                raise ConfigFaroInvalida(f"{nombre} tiene que ser un uid (entero entre 0 y {_UID_MAX})")
        euid = os.geteuid()
        if self.orquestador_uid == 0:
            raise ConfigFaroInvalida("JAX_FARO_ORQUESTADOR_UID no puede ser root: el orquestador no corre con privilegios")
        if self.orquestador_uid == euid and not self.solo_pruebas_mismo_uid:
            raise ConfigFaroInvalida(
                f"JAX_FARO_ORQUESTADOR_UID={self.orquestador_uid} es el usuario del servicio: cualquier proceso de `faro` "
                "pasaria por el orquestador")
        for uid, nombre in self.cuentas_con_codigo_de_modelos:
            if self.orquestador_uid == uid:
                raise ConfigFaroInvalida(
                    f"JAX_FARO_ORQUESTADOR_UID={uid} es la cuenta `{nombre}`, que ejecuta codigo guiado por modelos: "
                    "el orquestador tiene que ser una cuenta propia")
        if not 1 <= self.jaula_uid_min <= self.jaula_uid_max:
            raise ConfigFaroInvalida("el rango de uids de jaula (JAX_FARO_JAULA_UID_MIN..MAX) tiene que cumplir 1 <= MIN <= MAX")
        for nombre, v in (("root", 0), ("el usuario del servicio", euid), ("el orquestador", self.orquestador_uid)):
            if self.jaula_uid_min <= v <= self.jaula_uid_max:
                raise ConfigFaroInvalida(f"el rango de uids de jaula incluye a {nombre} (uid {v}): una jaula no puede ser {nombre}")
        for uid, nombre in self.cuentas_con_codigo_de_modelos:
            if self.jaula_uid_min <= uid <= self.jaula_uid_max:
                raise ConfigFaroInvalida(f"el rango de uids de jaula incluye a `{nombre}` (uid {uid}), que ejecuta codigo de modelos")
        if not self.plazo_s > 0:
            raise ConfigFaroInvalida("JAX_FARO_CONTROL_PLAZO_S tiene que ser positivo")
        if not isinstance(self.max_pedido, int) or self.max_pedido < 1024:
            raise ConfigFaroInvalida("JAX_FARO_CONTROL_MAX_PEDIDO tiene que ser un entero de al menos 1024")

    @property
    def ruta_socket(self) -> Path:
        return Path(self.control_dir) / NOMBRE_SOCKET

    @classmethod
    def desde_entorno(cls, env: Mapping[str, str], *, solo_pruebas_mismo_uid: bool = False,
                      getpwnam: Callable = pwd.getpwnam) -> "ConfigControl":
        def pedir(nombre: str) -> str:
            valor = (env.get(nombre) or "").strip()
            if not valor:
                raise ConfigFaroInvalida(f"{nombre} no esta definida: sin canal de control autenticado el Faro no arranca")
            return valor

        def entero(nombre: str) -> int:
            try:
                return int(pedir(nombre))
            except ValueError as exc:
                raise ConfigFaroInvalida(f"{nombre} no es un entero") from exc

        def opcional(nombre: str, tipo, defecto):
            valor = (env.get(nombre) or "").strip()
            try:
                return tipo(valor) if valor else defecto
            except ValueError as exc:
                raise ConfigFaroInvalida(f"{nombre} no es un numero valido") from exc

        return cls(control_dir=Path(pedir("JAX_FARO_CONTROL_DIR")), orquestador_uid=entero("JAX_FARO_ORQUESTADOR_UID"),
                   jaula_uid_min=entero("JAX_FARO_JAULA_UID_MIN"), jaula_uid_max=entero("JAX_FARO_JAULA_UID_MAX"),
                   plazo_s=opcional("JAX_FARO_CONTROL_PLAZO_S", float, PLAZO_S_POR_DEFECTO),
                   max_pedido=opcional("JAX_FARO_CONTROL_MAX_PEDIDO", int, MAX_PEDIDO_POR_DEFECTO),
                   solo_pruebas_mismo_uid=solo_pruebas_mismo_uid,
                   # En las pruebas de un solo usuario el «orquestador» ES este proceso (fruiz): la lista no aplica.
                   cuentas_con_codigo_de_modelos=() if solo_pruebas_mismo_uid else tuple(_resolver_cuentas(env, getpwnam).items()))


class _Rechazo(Exception):
    def __init__(self, motivo: str, **extra):
        super().__init__(motivo)
        self.motivo = motivo
        self.extra = extra          # run_id, id_correlacion...: van a la bitacora y a la respuesta


def _texto(pedido: Mapping, campo: str) -> str:
    v = pedido.get(campo)
    if not isinstance(v, str) or not v.strip() or len(v) > _MAX_CAMPO or any(ord(c) < 32 or ord(c) == 127 for c in v):
        raise _Rechazo(f"campo_invalido:{campo}")
    v = v.strip()
    if campo == "tenant" and not RE_TENANT.fullmatch(v):
        raise _Rechazo("campo_invalido:tenant")     # el mismo patron que las claves de los topes
    return v


def validar_directorio_control(cfg: ConfigControl) -> None:
    """El directorio del socket de control: del usuario del servicio, sin escritura de grupo ni NADA para otros.
    Lo llama `ServidorControl` al abrir y `servicio.arrancar` al arrancar (un 0777 no deja arrancar)."""
    d = Path(cfg.control_dir)
    try:
        st = os.lstat(d)
    except OSError as exc:
        raise ConfigFaroInvalida(f"el directorio de control {d} no se puede leer: {type(exc).__name__}") from exc
    if not stat.S_ISDIR(st.st_mode):
        raise ConfigFaroInvalida(f"{d} no es un directorio")
    if st.st_uid != os.geteuid():
        raise ConfigFaroInvalida(f"{d} tiene que ser del usuario del servicio (uid {os.geteuid()})")
    if st.st_mode & 0o027:
        raise ConfigFaroInvalida(
            f"{d} no admite escritura de grupo ni NINGUN permiso para otros (tiene {stat.S_IMODE(st.st_mode):04o}); "
            "use 0750 o 0710 con el grupo del orquestador")


class ServidorControl:
    """`crear_puerto(ejecucion) -> ServidorPuerto` es el del `Servicio` (con su bitacora y su UNICO presupuesto)."""

    def __init__(self, cfg: ConfigControl, crear_puerto: Callable[[Ejecucion], ServidorPuerto], bitacora: Bitacora,
                 *, leer_credenciales: Callable = credenciales_del_par, jaula_viva: Callable[[int], bool] | None = None):
        """`jaula_viva(uid) -> bool` (gancho para el lanzador, y para el scope de 0.6): mientras devuelva True, el
        uid de una ejecucion CERRADA no se reasigna (un proceso suyo podria seguir vivo con ese uid)."""
        self._cfg = cfg
        self._crear_puerto = crear_puerto
        self._bitacora = bitacora
        self._credenciales = leer_credenciales
        self._servidor: asyncio.AbstractServer | None = None
        self._tareas: set[asyncio.Task] = set()
        self._ejecuciones: dict[str, ServidorPuerto] = {}
        self._uids: dict[int, str] = {}                 # uid de jaula -> run_id de la ejecucion viva que lo usa
        self._jaula_viva = jaula_viva
        self._retenidos: set[int] = set()               # uids de ejecuciones cerradas cuya jaula sigue viva

    @property
    def ruta_socket(self) -> Path:
        return self._cfg.ruta_socket

    @property
    def ejecuciones(self) -> Mapping[str, ServidorPuerto]:
        return dict(self._ejecuciones)

    # -- ciclo de vida -------------------------------------------------------------------------
    def _validar_entorno(self) -> None:
        if os.geteuid() == 0:
            raise ConfigFaroInvalida("el canal de control no corre como root: usa el usuario sin privilegios `faro`")
        validar_directorio_control(self._cfg)

    async def __aenter__(self) -> "ServidorControl":
        self._validar_entorno()
        if os.path.lexists(self.ruta_socket):
            raise FileExistsError(f"ya hay un socket de control: {self.ruta_socket}")
        self._servidor = await asyncio.start_unix_server(self._atender, path=str(self.ruta_socket),
                                                         limit=self._cfg.max_pedido, backlog=socket.SOMAXCONN)
        os.chmod(self.ruta_socket, MODO_SOCKET_CONTROL)
        return self

    async def __aexit__(self, *_exc) -> None:
        if self._servidor is not None:
            self._servidor.close()
        for t in list(self._tareas):
            t.cancel()
        await asyncio.gather(*self._tareas, return_exceptions=True)
        if self._servidor is not None:
            try:
                await asyncio.wait_for(self._servidor.wait_closed(), 5)
            except TimeoutError:  # fail-soft: el apagado no puede colgarse; ya se cancelaron todas las tareas
                logger.warning("el cierre del canal de control no termino en 5s: se sigue con el apagado")
            self._servidor = None
        for run_id, srv in list(self._ejecuciones.items()):
            await self._cerrar_ejecucion(run_id, srv)
        try:
            os.unlink(self.ruta_socket)
        except FileNotFoundError:  # fail-soft: ya no esta; el objetivo del cierre es que no exista
            pass

    # -- anotacion ------------------------------------------------------------------------------
    async def _rechazar(self, cred, motivo: str, escritor=None, **extra) -> None:
        try:
            await self._bitacora.registrar(
                "control_rechazado", motivo=motivo, peer_pid=cred[0] if cred else -1, peer_uid=cred[1] if cred else -1, **extra)
        except Exception:  # fail-closed: el rechazo se mantiene; solo la anotacion fallo y queda en el log
            logger.exception("no se pudo registrar el rechazo de control (%s)", motivo)
        if escritor is not None:
            await self._responder(escritor, {"ok": False, "error": motivo, **{k: v for k, v in extra.items() if k in ("run_id", "id_correlacion")}})

    async def _responder(self, escritor: asyncio.StreamWriter, cuerpo: dict) -> None:
        try:
            escritor.write(json.dumps(cuerpo, ensure_ascii=True).encode() + b"\n")
            await asyncio.wait_for(escritor.drain(), self._cfg.plazo_s)
        except (ConnectionError, TimeoutError, OSError):  # fail-soft: el par se fue; lo hecho (o rechazado) ya esta anotado
            pass

    # -- una conexion ---------------------------------------------------------------------------
    async def _atender(self, lector: asyncio.StreamReader, escritor: asyncio.StreamWriter) -> None:
        tarea = asyncio.current_task()
        self._tareas.add(tarea)
        try:
            cred = self._credenciales(escritor)
            if cred is None or cred[1] != self._cfg.orquestador_uid:
                # No se lee NADA de lo que mande y no se le responde: no es el orquestador.
                await self._rechazar(cred, "uid_no_autorizado" if cred else "par_sin_credenciales")
                return
            try:
                linea = await asyncio.wait_for(lector.readline(), self._cfg.plazo_s)
            except (TimeoutError, ConnectionError):
                await self._rechazar(cred, "pedido_ilegible", escritor)
                return
            except ValueError:      # linea mas larga que `max_pedido`
                await self._rechazar(cred, "pedido_demasiado_largo", escritor)
                return
            try:
                pedido = json.loads(linea)
                if not isinstance(pedido, dict):
                    raise ValueError("el pedido no es un objeto")
            except ValueError:
                await self._rechazar(cred, "pedido_invalido", escritor)
                return
            op = pedido.get("op")
            try:
                if op == "crear":
                    respuesta = await self._crear(cred, pedido)
                elif op == "cerrar":
                    respuesta = await self._cerrar(cred, pedido)
                else:
                    raise _Rechazo("operacion_desconocida")
            except _Rechazo as r:
                await self._rechazar(cred, r.motivo, escritor, operacion=_corto(op), **r.extra)
                return
            await self._responder(escritor, respuesta)
        except asyncio.CancelledError:
            raise
        except Exception:  # fail-closed: la conexion se cierra sin respuesta; el error queda en el log
            logger.exception("el canal de control termino una conexion con error")
        finally:
            self._tareas.discard(tarea)
            escritor.close()
            try:
                await escritor.wait_closed()
            except OSError:  # fail-soft: el par ya cerro la conexion
                pass

    # -- operaciones ----------------------------------------------------------------------------
    def _validar_uid_jaula(self, pedido: Mapping) -> int:
        uid = pedido.get("uid_jaula")
        c = self._cfg
        if isinstance(uid, bool) or not isinstance(uid, int) or not c.jaula_uid_min <= uid <= c.jaula_uid_max:
            raise _Rechazo("uid_jaula_invalido")
        if uid in (0, os.geteuid(), c.orquestador_uid):      # el rango ya los excluye; defensa en profundidad
            raise _Rechazo("uid_jaula_invalido")
        if uid in self._uids:
            if uid in self._retenidos and self._jaula_viva is not None and not self._jaula_viva(uid):
                self._retenidos.discard(uid)
                self._uids.pop(uid, None)           # la jaula ya murio: el uid se libera
            else:
                raise _Rechazo("uid_jaula_en_uso")
        return uid

    async def _crear(self, cred, pedido: Mapping) -> dict:
        campos = {c: _texto(pedido, c) for c in CAMPOS_DE_IDENTIDAD}      # lo demas del pedido se IGNORA
        uid = self._validar_uid_jaula(pedido)
        run_id = "r-" + uuid.uuid4().hex[:24]
        id_correlacion = uuid.uuid4().hex
        try:
            ejecucion = Ejecucion(run_id=run_id, id_correlacion=id_correlacion, uid_esperado=uid, **campos)
        except ConfigFaroInvalida:
            raise _Rechazo("pedido_invalido") from None
        self._uids[uid] = run_id                  # se reserva ya: dos pedidos a la vez no comparten uid
        srv = None
        try:
            srv = self._crear_puerto(ejecucion)
            await srv.__aenter__()
            await self._bitacora.registrar("control_creado", run_id=run_id, id_correlacion=id_correlacion, uid_jaula=uid,
                                           peer_uid=cred[1], **campos)
        except BaseException as exc:
            self._uids.pop(uid, None)
            if srv is not None:
                try:
                    await srv.__aexit__(None, None, None)
                except Exception:  # fail-soft: ya se esta deshaciendo la creacion; el error de fondo es el que sube
                    logger.exception("no se pudo deshacer el Puerto de %s", run_id)
            if isinstance(exc, asyncio.CancelledError):
                raise
            identificadores = {"run_id": run_id, "id_correlacion": id_correlacion, "uid_jaula": uid}
            if isinstance(exc, ConfigFaroInvalida):
                raise _Rechazo("ejecucion_invalida", **identificadores) from None
            logger.error("no se pudo crear la ejecucion %s (%s)", run_id, type(exc).__name__)
            raise _Rechazo("no_se_pudo_crear_o_anotar", **identificadores) from None
        self._ejecuciones[run_id] = srv
        return {"ok": True, "run_id": run_id, "id_correlacion": id_correlacion, "uid_jaula": uid}

    async def _cerrar_ejecucion(self, run_id: str, srv: ServidorPuerto) -> None:
        self._ejecuciones.pop(run_id, None)
        uid = srv.ejecucion.uid_esperado
        if self._jaula_viva is not None and self._jaula_viva(uid):
            self._retenidos.add(uid)                # sigue ocupado hasta que su jaula muera
        else:
            self._uids.pop(uid, None)
        try:
            await srv.__aexit__(None, None, None)
        except Exception:  # fail-soft: el cierre sigue; el socket de esa ejecucion ya no se atiende
            logger.exception("no se pudo cerrar bien el Puerto de %s", run_id)

    async def _cerrar(self, cred, pedido: Mapping) -> dict:
        run_id = pedido.get("run_id")
        srv = self._ejecuciones.get(run_id) if isinstance(run_id, str) else None
        if srv is None:
            raise _Rechazo("run_id_desconocido")
        await self._cerrar_ejecucion(run_id, srv)
        try:
            await self._bitacora.registrar("control_cerrado", run_id=run_id, peer_uid=cred[1])
        except Exception:  # fail-closed: la ejecucion ya esta cerrada; solo la anotacion fallo y queda en el log
            logger.exception("no se pudo registrar el cierre de %s", run_id)
        return {"ok": True, "run_id": run_id}


def _corto(valor: object) -> str:
    return str(valor)[:32]
