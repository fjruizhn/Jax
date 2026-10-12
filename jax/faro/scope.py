"""La unidad de systemd (cgroup) por ejecucion del Faro (plan 0.6 / R-2, plan 2026-10-09-faro-scope-r2).

Cada jaula corre dentro de un SERVICIO TRANSITORIO del user manager, `faro-jaula-u<uid>.service`, bajo el
slice explicito de Faro, con limites FINITOS de memoria, CPU y tareas: `MemoryMax`, `CPUQuota` y
`TasksMax` son obligatorios (no hay valores por defecto: sin los tres, el lanzamiento falla cerrado). El
servicio usa `Type=exec` (activa solo cuando el exec de verdad ocurrio), `--pipe` (el stdio de la jaula
pasa por systemd-run, que espera a que termine), `--collect` (la unidad muerta se descarga y su nombre
queda libre), `KillMode=control-group` y `OOMPolicy=kill` (matar la unidad mata TODO su arbol).

POR QUE `.service` Y NO `.scope` (desviacion declarada del plan 0.6): el futuro servicio de sistema corre
como `User=faro` y no se autorizo a `faro` a administrar el system manager ni a un broker privilegiado;
el user manager no puede adoptar un PID nacido fuera de su subarbol (no probado), asi que es el manager
quien crea el proceso DENTRO del cgroup. El contrato de aislamiento y limites por cgroup es el mismo.

EL UID ES EL NOMBRE. `ServidorControl` solo permite una ejecucion viva por uid; el nombre determinista
`faro-jaula-u<uid>.service` permite consultar el estado tras reinicios, pero NO basta para resolver carreras
o conservar ownership tras un crash. Antes del start se adquiere un lease durable exclusivo en
`JAX_FARO_SCOPE_STATE_DIR`; solo se quita despues de confirmar el cgroup vacio y esperar la salida del
cliente `systemd-run --pipe`. Al arrancar el servicio, todos los leases se reconcilian antes de abrir el
socket de control. Directorio, registros corruptos, identidad ajena, jobs pendientes o consultas inciertas
bloquean la admision; nunca se borra una cuarentena por no encontrarla en memoria.

CONTRATO DE ESTADO (trivalente). `estado(uid)` devuelve:
- LIVE: unidad activa y cgroup poblado; tambien cuando la unidad ya se apaga pero QUEDAN procesos suyos
  en el cgroup (hijos y nietos del proceso principal: exactamente lo que R-2 exige retener);
- EMPTY: systemd confirma que no existe unidad activa (LoadState=not-found, o inactiva) y el cgroup ya
  esta vacio o eliminado (`cgroup.events` con `populated 0`, o sin directorio);
- UNKNOWN: manager, unidad, ControlGroup, cgroupfs o proceso no permiten concluir (consulta rota, salida
  ilegible, cgroup activo sin ControlGroup, contradicciones como unidad activa con cgroup vacio).
SOLO EMPTY libera un uid. LIVE y UNKNOWN lo conservan/rechazan: el fallo es cerrado.

El `ControlGroup` se LEE de systemd (`systemctl --user show`), nunca se deduce por concatenacion; la
poblacion se comprueba en `<ControlGroup>/cgroup.events` del cgroupfs (legible: el subarbol del user
manager esta delegado a su usuario).

IDENTIDAD POR INTENTO (cierre del BLOCK 1 del veredicto Tier 3 sobre f2476099). La vida del `systemd-run`
propio NO prueba propiedad: un perdedor de la carrera puede tardar en recibir su rechazo. Cada `arrancar`
genera un id de intento y lo planta en `--property=Description=FARO-R2-<id>`; la unidad solo se da por
NUESTRA si `show` devuelve exactamente esa Description (determinista, sin temporizacion), y la limpieza de
un arranque fallido o cancelado SOLO mata si la Description vigente sigue siendo la del intento.

ALLOWLIST CERRADA DE ENTORNO (decision de Fernando, 2026-10-09; cierra el MAJOR 3). El entorno del user
manager llega PERSISTENTE al servicio transitorio (el entorno de systemd-run no se hereda), y la cadena
elevacion+bwrap corre ANTES del `--clearenv` de bwrap. Por eso el comando de la unidad empieza con
`/usr/bin/env -i`: la elevacion y bwrap reciben exactamente lo que el argv declara y NADA del manager.
(`--setenv`+`UnsetEnvironment=` no cierra un conjunto desconocido de variables; `env -i` si.)
"""
from __future__ import annotations

import asyncio
import contextlib
import enum
import hashlib
import logging
import os
import re
import stat
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigFaroInvalida
from .scope_lease import LeaseStage, LeaseStore, LeaseStoreError, LeaseStoreNotFound, ScopeLease

logger = logging.getLogger(__name__)

SYSTEMD_RUN = "/usr/bin/systemd-run"
SYSTEMCTL = "/usr/bin/systemctl"
ENV_LIMPIA = "/usr/bin/env"
RAIZ_CGROUP = Path("/sys/fs/cgroup")
PREFIJO_UNIDAD = "faro-jaula-u"
PREFIJO_INTENTO = "FARO-R2-"
#: Lo unico que llega a systemd-run/systemctl: PATH y, SI EXISTEN, las dos variables que ubican el bus
#: del user manager (sin XDG_RUNTIME_DIR, `--user` no conecta: medido en hall9000, systemd 259). No son
#: secretos y systemd-run NO los hereda el servicio (el servicio recibe el entorno del manager, no el
#: suyo). 0.10 debe fijarlas en la unidad del servicio de sistema (`faro` con linger).
_VARS_DEL_BUS = ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")


def entorno_consulta() -> dict[str, str]:
    env = {"PATH": "/usr/bin:/bin"}
    for var in _VARS_DEL_BUS:
        valor = os.environ.get(var)
        if valor:
            env[var] = valor
    return env

_PLAZO_CONSULTA_S = 5.0
_UID_MAX = 2 ** 32 - 2      # 2**32-1 es «sin uid» para el kernel
_RE_MEMORIA = re.compile(r"^([0-9]+)([KMGTP]?)$", re.IGNORECASE)
_RE_CPU = re.compile(r"^([0-9]+(?:\.[0-9]+)?)%$")
_RE_TAREAS = re.compile(r"^[0-9]+$")
_RE_SLICE = re.compile(r"^[A-Za-z0-9:@_.-]+\.slice$")
_MULT_MEMORIA = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4, "P": 1024 ** 5}
_RE_TIMESPAN = re.compile(r"^([0-9]+(?:\.[0-9]+)?)(us|µs|ms|s|min|m|h|d|w)$")
_US = {"us": 1, "µs": 1, "ms": 1_000, "s": 1_000_000, "m": 60_000_000, "min": 60_000_000,
       "h": 3_600_000_000, "d": 86_400_000_000, "w": 604_800_000_000}

_ACTIVAS = {"active", "activating", "reloading"}          # hay (o puede haber) vida gestionada
_INACTIVAS = {"inactive", "failed", "deactivating"}       # la unidad se apago (o esta apagandose)


def tiempospan_a_us(texto: str) -> int:
    """«500ms», «2s», «1min 30s» (como imprime systemd `--value` sus tiempos) a microsegundos."""
    tokens = str(texto).split()
    if not tokens:
        raise ValueError(f"timespan ilegible: {texto!r}")
    total = 0.0
    for token in tokens:
        m = _RE_TIMESPAN.match(token)
        if m is None:
            raise ValueError(f"timespan ilegible: {texto!r}")
        total += float(m.group(1)) * _US[m.group(2)]
    return round(total)


def nombre_unidad(uid: int) -> str:
    """El nombre DETERMINISTA de la unidad de un uid de jaula: sin el, no hay forma de preguntar por el."""
    if isinstance(uid, bool) or not isinstance(uid, int) or not 1 <= uid <= _UID_MAX:
        raise ConfigFaroInvalida(f"el uid de la unidad tiene que ser un entero entre 1 y {_UID_MAX}")
    return f"{PREFIJO_UNIDAD}{uid}.service"


def nombre_unidad_intento(uid: int, intento: str) -> str:
    """Nombre inmutable por intento para que una limpieza obsoleta no alcance al siguiente lease."""
    nombre_unidad(uid)
    if not isinstance(intento, str) or not intento or "\x00" in intento:
        raise ConfigFaroInvalida("el id de intento debe ser texto no vacío")
    sufijo = intento if re.fullmatch(r"[0-9a-f]{32}", intento) else hashlib.sha256(intento.encode()).hexdigest()[:32]
    return f"{PREFIJO_UNIDAD}{uid}-{sufijo}.service"


class EstadoUnidad(enum.Enum):
    """El trivalor del contrato: solo EMPTY libera; LIVE y UNKNOWN conservan/rechazan el uid."""
    LIVE = "live"
    EMPTY = "empty"
    UNKNOWN = "unknown"


class _EstadoIdentidad(enum.Enum):
    AUSENTE = "absent"
    PROPIA = "ours"
    AJENA = "other"
    DESCONOCIDA = "unknown"


class ScopeError(Exception):
    """Un arranque, verificacion o kill que no se pudo completar: falla cerrado."""


class _UnidadAjena(Exception):
    """La unidad activa bajo el nombre de este uid NO es de esta llamada (perdio la carrera): no se mata."""


@dataclass
class _ScopeHandle:
    uid: int
    intento: str
    unidad: str
    proceso: object
    client_task: asyncio.Task | None = None
    wait_rc: int | None = None
    wait_ok: bool = False
    drain_error: BaseException | None = None
    materialization_seen: bool = False
    finalize_task: asyncio.Task | None = None
    close_task: asyncio.Task | None = None


def _validar_comando(comando: object) -> list[str]:
    if (not isinstance(comando, (list, tuple)) or isinstance(comando, (str, bytes)) or not comando
            or not all(isinstance(c, str) and "\x00" not in c and c for c in comando)
            or not comando[0].startswith("/")):
        raise ConfigFaroInvalida(
            "el comando de la unidad tiene que ser una lista de textos cuyo primer elemento es un ejecutable absoluto")
    return list(comando)


@dataclass(frozen=True)
class ConfigScope:
    """`JAX_FARO_SCOPE_MEMORY_MAX`, `_CPU_QUOTA` y `_TASKS_MAX` son obligatorias y finitas: sin los tres
    limites no hay unidad (nunca infinity: un limite infinito no es un limite). El slice es el explicito
    de Faro; los plazos, opcionales."""
    memory_max: str
    cpu_quota: str
    tasks_max: str
    state_dir: Path | None = None
    slice: str = "faro.slice"
    plazo_arranque_s: float = 10.0
    plazo_vacio_s: float = 10.0

    def __post_init__(self) -> None:
        m = _RE_MEMORIA.match(self.memory_max or "")
        if m is None or int(m.group(1)) == 0:
            raise ConfigFaroInvalida(
                f"JAX_FARO_SCOPE_MEMORY_MAX={self.memory_max!r} no es un limite finito: tiene que ser bytes con"
                " sufijo K/M/G/T/P (nunca infinity ni porcentaje); sin limite de memoria no hay unidad")
        q = _RE_CPU.match(self.cpu_quota or "")
        if q is None or float(q.group(1)) <= 0:
            raise ConfigFaroInvalida(
                f"JAX_FARO_SCOPE_CPU_QUOTA={self.cpu_quota!r} no es un limite finito: tiene que ser un porcentaje"
                " positivo (p. ej. 200%); sin limite de CPU no hay unidad")
        if _RE_TAREAS.fullmatch(self.tasks_max or "") is None or int(self.tasks_max) == 0:
            raise ConfigFaroInvalida(
                f"JAX_FARO_SCOPE_TASKS_MAX={self.tasks_max!r} no es un limite finito: tiene que ser un entero"
                " positivo (con sufijo K/M/G); sin tope de tareas no hay unidad")
        if not _RE_SLICE.match(self.slice or ""):
            raise ConfigFaroInvalida(f"JAX_FARO_SCOPE_SLICE={self.slice!r} no es un nombre de slice (algo.slice)")
        if self.state_dir is not None and not Path(self.state_dir).is_absolute():
            raise ConfigFaroInvalida("JAX_FARO_SCOPE_STATE_DIR tiene que ser una ruta absoluta")
        for nombre, v in (("JAX_FARO_SCOPE_PLAZO_ARRANQUE_S", self.plazo_arranque_s),
                          ("JAX_FARO_SCOPE_PLAZO_VACIO_S", self.plazo_vacio_s)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
                raise ConfigFaroInvalida(f"{nombre} tiene que ser positivo")

    def bytes_memoria(self) -> int:
        n, sufijo = _RE_MEMORIA.match(self.memory_max).groups()
        return int(n) * _MULT_MEMORIA[sufijo.upper()]

    def usec_cpu(self) -> int:
        """`CPUQuota=200%` son 2s de CPU por segundo: systemd lo informa como `CPUQuotaPerSecUSec`."""
        return round(float(_RE_CPU.match(self.cpu_quota).group(1)) / 100 * 1_000_000)

    def tareas(self) -> int:
        return int(self.tasks_max)

    @classmethod
    def desde_entorno(cls, env) -> "ConfigScope":
        def pedir(nombre: str) -> str:
            valor = (env.get(nombre) or "").strip()
            if not valor:
                raise ConfigFaroInvalida(
                    f"{nombre} no esta definida: sin los tres limites finitos no hay unidad por ejecucion")
            return valor

        def opcional(nombre: str, tipo, defecto):
            valor = (env.get(nombre) or "").strip()
            try:
                return tipo(valor) if valor else defecto
            except ValueError as exc:
                raise ConfigFaroInvalida(f"{nombre} no es un numero valido") from exc

        return cls(memory_max=pedir("JAX_FARO_SCOPE_MEMORY_MAX"), cpu_quota=pedir("JAX_FARO_SCOPE_CPU_QUOTA"),
                   tasks_max=pedir("JAX_FARO_SCOPE_TASKS_MAX"), state_dir=Path(pedir("JAX_FARO_SCOPE_STATE_DIR")),
                   slice=opcional("JAX_FARO_SCOPE_SLICE", str, "faro.slice"),
                   plazo_arranque_s=opcional("JAX_FARO_SCOPE_PLAZO_ARRANQUE_S", float, 10.0),
                   plazo_vacio_s=opcional("JAX_FARO_SCOPE_PLAZO_VACIO_S", float, 10.0))


class ScopeManager:
    """Arranca, consulta y mata la unidad transitoria de cada uid. Todo es asincrono (subprocesos, nunca
    bloqueante) y falla cerrado: lo que no se puede concluir es UNKNOWN, y un arranque que no se puede
    VERIFICAR (activa, con los tres limites efectivos) se mata y se reporta como error."""

    def __init__(self, cfg: ConfigScope, *, systemd_run: str = SYSTEMD_RUN, systemctl: str = SYSTEMCTL,
                 raiz_cgroup: Path = RAIZ_CGROUP, crear_proceso: Callable | None = None,
                 intervalo_s: float = 0.05, plazo_consulta_s: float = _PLAZO_CONSULTA_S,
                 lease_store: LeaseStore | None = None):
        self._cfg = cfg
        self._systemd_run = systemd_run
        self._systemctl = systemctl
        self._raiz_cgroup = Path(raiz_cgroup)
        self._crear_proceso = crear_proceso if crear_proceso is not None else asyncio.create_subprocess_exec
        self._intervalo_s = intervalo_s
        self._plazo_consulta_s = plazo_consulta_s
        self._leases = lease_store if lease_store is not None else (
            LeaseStore(cfg.state_dir) if cfg.state_dir is not None else None)
        self._leases_recuperados = False
        self._recovery_lock = asyncio.Lock()
        self._limpiezas: set[asyncio.Task] = set()     # limpiezas de cancelacion en vuelo (MAJOR 2)
        self._resultados_limpieza: dict[tuple[int, str], object] = {}
        self._arranques: set[asyncio.Task] = set()
        self._arranques_uid: dict[int, set[asyncio.Task]] = {}
        self._uids_parando: set[int] = set()
        self._cerrando = False
        self._fd_instancia: int | None = None
        self._token_instancia: object | None = None
        self._adquisicion_instancia: asyncio.Task | None = None
        self._handles: dict[int, _ScopeHandle] = {}
        self._cuarentena: dict[int, str] = {}

    async def adquirir_instancia(self) -> object:
        """Exclusión de vida completa antes de recovery/socket; flock cruza procesos."""
        if self._leases is None:
            raise ScopeError("falta almacén durable para lock exclusivo de instancia")
        if self._fd_instancia is not None or self._adquisicion_instancia is not None:
            raise ScopeError("la instancia única ya fue adquirida por este gestor")
        tarea = asyncio.create_task(asyncio.to_thread(self._leases.acquire_instance))
        self._adquisicion_instancia = tarea
        try:
            fd = await asyncio.shield(tarea)
            token = object()
            self._fd_instancia = fd
            self._token_instancia = token
            return token
        except asyncio.CancelledError:
            # to_thread no detiene el worker al cancelar al caller. Drenarlo y cerrar
            # el fd que eventualmente obtuvo antes de propagar la cancelación.
            async def drenar_y_cerrar() -> None:
                try:
                    fd_obtenido = await asyncio.shield(tarea)
                except LeaseStoreError:  # fail-soft: el worker rechazó adquirir; no devolvió fd y sigue la cancelación original
                    return
                try:
                    await asyncio.to_thread(os.close, fd_obtenido)
                except OSError as exc:
                    self._fd_instancia = fd_obtenido
                    self._token_instancia = object()
                    raise ScopeError("cancelación adquirió el lock pero no pudo cerrar su fd; lock retenido") from exc

            limpieza = asyncio.create_task(drenar_y_cerrar())
            while not limpieza.done():
                try:
                    await asyncio.shield(limpieza)
                except asyncio.CancelledError:
                    continue
            limpieza.result()
            raise
        except LeaseStoreError as exc:
            raise ScopeError("no se pudo asegurar instancia única del Faro") from exc
        finally:
            if self._adquisicion_instancia is tarea:
                self._adquisicion_instancia = None

    async def liberar_instancia(self, propietario: object) -> None:
        if propietario is not self._token_instancia or self._fd_instancia is None:
            raise ScopeError("el caller no posee el lock exclusivo de instancia")
        fd = self._fd_instancia
        cierre = asyncio.create_task(asyncio.to_thread(os.close, fd))
        cancelada = False
        try:
            await asyncio.shield(cierre)
        except asyncio.CancelledError:
            cancelada = True
            while not cierre.done():
                try:
                    await asyncio.shield(cierre)
                except asyncio.CancelledError:  # fail-soft: se drena el close exacto antes de propagar cancelación
                    continue
            cierre.result()
        if propietario is self._token_instancia and fd == self._fd_instancia:
            self._fd_instancia = None
            self._token_instancia = None
        if cancelada:
            raise asyncio.CancelledError

    async def recuperar_leases(self) -> None:
        """Reconcilia cuarentenas durables antes de admitir conexiones o nuevos lanzamientos.

        PREPARED prueba que no hubo llamada externa; REQUESTED sin handle conserva cuarentena
        solo para ese UID. CLIENT_CLOSED se reconcilia por identidad exacta.
        """
        if self._leases is None:
            raise ScopeError("falta JAX_FARO_SCOPE_STATE_DIR: no hay cuarentena durable de R-2")
        async with self._recovery_lock:
            if self._leases_recuperados:
                return
            try:
                leases = await asyncio.to_thread(self._leases.all)
            except LeaseStoreError as exc:
                raise ScopeError("no se pueden leer las cuarentenas durables de R-2") from exc
            for lease in leases:
                try:
                    await self._reconciliar_lease_de_arranque(lease)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # fail-soft: se conserva ese UID en cuarentena y continúa el recovery de hermanas
                    # Un error operativo de un UID no debe impedir reconciliar las
                    # hermanas; el lease problemático queda en cuarentena durable.
                    self._cuarentena[lease.uid] = f"recovery inconcluso: {type(exc).__name__}"
                    logger.exception("recovery de uid %s quedó en cuarentena", lease.uid)
            self._leases_recuperados = True

    async def _reconciliar_lease_de_arranque(self, lease: ScopeLease) -> None:
        unidad, intento = lease.unit, lease.attempt_id
        if lease.stage in (LeaseStage.REQUESTED, LeaseStage.ACCEPTED):
            # Sin el sender causal, ni una observación vieja del receptor prueba que
            # no haya otra petición en tránsito.
            self._cuarentena[lease.uid] = f"{lease.stage.value} sin handle causal recuperable"
            return
        identidad = await self._identidad(lease.uid, intento)
        trabajo = await self._hay_trabajo(unidad)
        if identidad is _EstadoIdentidad.DESCONOCIDA or trabajo is None:
            self._cuarentena[lease.uid] = "recovery de namespace inconcluso"
            return
        if identidad is _EstadoIdentidad.AJENA or trabajo:
            self._cuarentena[lease.uid] = "unidad/job sin identidad recuperable"
            return
        if lease.stage is LeaseStage.PREPARED:
            if identidad is not _EstadoIdentidad.AUSENTE or not await self._namespace_vacio(lease.uid):
                self._cuarentena[lease.uid] = "PREPARED contradice unidad presente"
                return
            await self._liberar_lease(lease.uid, intento)
            return
        if lease.stage is LeaseStage.CLIENT_CLOSED:
            if identidad is _EstadoIdentidad.PROPIA:
                try:
                    await self.matar(lease.uid, intento=intento)
                except Exception:  # fail-soft: el kill puede perder la carrera; solo identidad AUSENTE permite seguir
                    # Una unidad propia puede ser recolectada justo antes del kill.
                    # Solo la segunda observación AUSENTE permite pasar al kernel.
                    identidad = await self._identidad(lease.uid, intento)
                    if identidad is not _EstadoIdentidad.AUSENTE:
                        self._cuarentena[lease.uid] = "CLIENT_CLOSED kill/identidad no concluyentes"
                        return
                if await self.esperar_vacio(lease.uid, intento=intento, plazo_s=self._plazo_consulta_s) is not EstadoUnidad.EMPTY:
                    self._cuarentena[lease.uid] = "CLIENT_CLOSED sigue poblado"
                    return
            elif identidad is not _EstadoIdentidad.AUSENTE:
                self._cuarentena[lease.uid] = "CLIENT_CLOSED identidad no concluyente"
                return
            if not await self._namespace_vacio(lease.uid):
                self._cuarentena[lease.uid] = "CLIENT_CLOSED namespace no vacío"
                return
            if await self._estado_kernel_lease(lease) != 0:
                self._cuarentena[lease.uid] = "CLIENT_CLOSED sin evidencia kernel EMPTY"
                return
            await self._liberar_lease(lease.uid, intento)

    async def intento_actual(self, uid: int) -> str:
        """Devuelve el attempt esperado; los cierres deben conservarlo de extremo a extremo."""
        if self._leases is None:
            raise ScopeError("falta el almacén durable de leases R-2")
        try:
            lease = await asyncio.to_thread(self._leases.read, uid)
            return lease.attempt_id
        except LeaseStoreError as exc:
            raise ScopeError(f"no se pudo leer el handle durable del uid {uid}") from exc

    async def liberar(self, uid: int, *, intento: str) -> None:
        """Solo PREPARED o CLIENT_CLOSED con EMPTY puede liberarse externamente."""
        if self._leases is None:
            raise ScopeError("falta el almacén durable de leases R-2")
        try:
            lease = await asyncio.to_thread(self._leases.read, uid)
            if lease.attempt_id != intento or lease.stage in (LeaseStage.REQUESTED, LeaseStage.ACCEPTED):
                raise ScopeError(f"{lease.stage.value} no se libera sin cierre causal del cliente")
            if await self.esperar_vacio(uid, intento=intento, plazo_s=0.05) is not EstadoUnidad.EMPTY:
                raise ScopeError("namespace no está vacío; lease retenido")
            if not await self._namespace_vacio(uid):
                raise ScopeError("namespace contiene otra unidad/job del UID; lease retenido")
            await self._liberar_lease(uid, intento)
        except LeaseStoreError as exc:
            raise ScopeError(f"no se pudo liberar el lease durable del uid {uid}") from exc

    async def _reservar_lease(self, uid: int, intento: str) -> None:
        if self._leases is None:
            raise ScopeError("falta JAX_FARO_SCOPE_STATE_DIR: se rechaza arrancar sin lease durable")
        unidad = nombre_unidad_intento(uid, intento)
        tarea = asyncio.create_task(asyncio.to_thread(self._leases.acquire, uid, intento, unidad))
        try:
            await asyncio.shield(tarea)
        except asyncio.CancelledError:
            # El hilo puede completar aunque se cancele la espera. No lanzar systemd-run;
            # esperar su resultado y quitar solo el lease recién adquirido.
            try:
                await asyncio.shield(tarea)
            except LeaseStoreError:  # fail-soft: el worker no creó lease; no se lanza systemd-run y se propaga cancelación
                pass
            else:
                await self._liberar_lease(uid, intento)
            raise
        except LeaseStoreError as exc:
            raise ScopeError(f"no se pudo reservar el uid {uid} en cuarentena durable") from exc

    async def _liberar_lease(self, uid: int, intento: str) -> None:
        if self._leases is None:
            raise ScopeError("falta el almacén durable de leases R-2")
        try:
            lease = await asyncio.to_thread(self._leases.read, uid)
            if lease.attempt_id != intento:
                raise ScopeError(f"el lease de uid {uid} cambió antes de liberar")
            if lease.stage is LeaseStage.PREPARED:
                if not await self._namespace_vacio(uid):
                    raise ScopeError(f"namespace de uid {uid} no está vacío para liberar PREPARED")
                verificada_kernel = False
            elif lease.stage is LeaseStage.CLIENT_CLOSED:
                kernel = await asyncio.to_thread(self._leer_cgroup_protegido, lease.cgroup_path, lease)
                if kernel is None:
                    raise ScopeError(f"evidencia kernel de uid {uid} no es verificable")
                if not kernel[3] and kernel[0] != 0:
                    raise ScopeError(f"cgroup de uid {uid} sigue poblado o no es legible")
                if not await self._namespace_vacio(uid):
                    raise ScopeError(f"namespace de uid {uid} no está vacío para liberar")
                verificada_kernel = True
            else:
                raise ScopeError(f"stage {lease.stage.value} no autoriza liberar uid {uid}")
            await asyncio.to_thread(self._leases.release, uid, intento, cgroup_verified=verificada_kernel)
            self._uids_parando.discard(uid)
        except (LeaseStoreError, OSError) as exc:
            raise ScopeError(f"no se pudo liberar el lease durable del uid {uid}") from exc

    def _registrar_limpieza(self, tarea: asyncio.Task, uid: int, intento: str) -> asyncio.Task:
        setattr(tarea, "_faro_uid", uid)
        setattr(tarea, "_faro_intento", intento)
        self._limpiezas.add(tarea)
        def terminado(completada: asyncio.Task) -> None:
            self._limpiezas.discard(completada)
            clave = (uid, intento)
            if completada.cancelled():
                self._resultados_limpieza[clave] = asyncio.CancelledError()
            else:
                error = completada.exception()
                resultado = error if error is not None else completada.result()
                if isinstance(resultado, BaseException) or resultado is not True:
                    self._resultados_limpieza[clave] = resultado
        tarea.add_done_callback(terminado)
        return tarea

    async def _transicionar_lease(self, uid: int, intento: str, desde: LeaseStage, hacia: LeaseStage,
                                 *, cgroup_path: str | None = None, cgroup_dev: int | None = None,
                                 cgroup_ino: int | None = None) -> ScopeLease:
        if self._leases is None:
            raise ScopeError("falta el almacén durable de leases R-2")
        try:
            return await asyncio.to_thread(self._leases.transition, uid, intento, desde, hacia,
                                            cgroup_path=cgroup_path, cgroup_dev=cgroup_dev,
                                            cgroup_ino=cgroup_ino)
        except LeaseStoreError as exc:
            raise ScopeError(f"no se pudo persistir {hacia.value} para uid {uid}") from exc

    async def _crear_systemd_run(self, *argv, **kwargs):
        """Lanza systemd-run sin permitir que stdout/stderr sin consumidor atasquen la jaula."""
        proceso = await self._crear_proceso(*argv, **kwargs)
        stderr = getattr(proceso, "stderr", None)
        if stderr is not None and callable(getattr(stderr, "read", None)):
            cola = bytearray()
            setattr(proceso, "_faro_stderr_tail", cola)
            tarea = asyncio.create_task(self._drenar_stderr(stderr, cola))
            setattr(proceso, "_faro_stderr_task", tarea)
        return proceso

    async def _esperar_cliente_handle(self, handle: _ScopeHandle) -> None:
        """Único wait/drain por proceso; el error de drain no borra la prueba independiente de wait."""
        try:
            handle.wait_rc = await handle.proceso.wait()
            handle.wait_ok = True
        except BaseException:
            raise
        try:
            await self._esperar_drenaje_stderr(handle.proceso)
        except BaseException as exc:  # fail-soft: el error queda guardado y el owner lo recibe tras cleanup seguro
            handle.drain_error = exc

    def _registrar_handle(self, uid: int, intento: str, unidad: str, proceso: object) -> _ScopeHandle:
        existente = self._handles.get(uid)
        if existente is not None and existente.intento != intento:
            raise ScopeError(f"uid {uid} ya tiene un handle de otro attempt")
        if existente is not None:
            return existente
        handle = _ScopeHandle(uid, intento, unidad, proceso)
        handle.client_task = asyncio.create_task(self._esperar_cliente_handle(handle))
        self._handles[uid] = handle
        return handle

    async def _aceptar_handle(self, handle: _ScopeHandle) -> None:
        lease = await asyncio.to_thread(self._leases.read, handle.uid)
        if lease.attempt_id != handle.intento:
            raise ScopeError(f"lease cambió para uid {handle.uid}")
        if lease.stage is LeaseStage.REQUESTED:
            observado = await self._consultar(handle.unidad, ("Description", "ControlGroup", "Delegate",
                                                                 "MemoryMax", "CPUQuotaPerSecUSec", "TasksMax"))
            if (observado is None or observado["Description"].strip() != self.description_de_intento(handle.intento)
                    or observado["Delegate"].strip().lower() != "no"):
                raise ScopeError(f"no se pudo verificar identidad/cgroup del attempt {handle.intento}")
            self._comparar_limites(observado, handle.unidad)
            path = observado["ControlGroup"].strip()
            lectura = await asyncio.to_thread(self._leer_cgroup_protegido, path, None)
            if lectura is None or lectura[0] is None or lectura[3]:
                raise ScopeError(f"no se pudo observar evidencia kernel del cgroup de uid {handle.uid}")
            await self._transicionar_lease(
                handle.uid, handle.intento, LeaseStage.REQUESTED, LeaseStage.ACCEPTED,
                cgroup_path=path, cgroup_dev=lectura[1], cgroup_ino=lectura[2])
        elif lease.stage is not LeaseStage.ACCEPTED:
            raise ScopeError(f"stage {lease.stage.value} no acepta materialización de uid {handle.uid}")
        handle.materialization_seen = True

    async def finalizar_natural(self, uid: int, proceso: object) -> int:
        handle = self._handles.get(uid)
        if handle is None or handle.proceso is not proceso:
            raise ScopeError(f"no existe handle exacto para finalizar uid {uid}")
        if handle.finalize_task is None:
            handle.finalize_task = asyncio.create_task(self._finalizar_natural(uid, proceso))
        return await asyncio.shield(handle.finalize_task)

    async def _finalizar_natural(self, uid: int, proceso: object) -> int:
        """Cierra causalmente el cliente natural exacto y libera solo tras CLIENT_CLOSED + EMPTY."""
        handle = self._handles.get(uid)
        resultados_uid = [(clave, resultado) for clave, resultado in self._resultados_limpieza.items()
                          if clave[0] == uid]
        for clave, resultado in resultados_uid:
            if isinstance(resultado, BaseException):
                raise ScopeError(f"limpieza fallida para uid {uid}; lease retenido") from resultado
            if resultado is not True:
                raise ScopeError(f"limpieza inconclusa para uid {uid}; lease retenido")
        if handle is None or handle.proceso is not proceso:
            raise ScopeError(f"no existe handle exacto para finalizar uid {uid}")
        assert handle.client_task is not None
        await asyncio.shield(handle.client_task)
        if not handle.wait_ok:
            raise ScopeError(f"wait del cliente de uid {uid} no demuestra cierre")
        lease = await asyncio.to_thread(self._leases.read, uid) if self._leases is not None else None
        if lease is None or lease.attempt_id != handle.intento:
            raise ScopeError(f"lease no coincide con el handle de uid {uid}")
        if lease.stage is not LeaseStage.ACCEPTED:
            self._cuarentena[uid] = f"{lease.stage.value} sin prueba ACCEPTED"
            raise ScopeError("lease no aceptado por el manager; cuarentena retenida")
        await self._transicionar_lease(uid, handle.intento, LeaseStage.ACCEPTED, LeaseStage.CLIENT_CLOSED)
        estado = await self.esperar_vacio(uid, intento=handle.intento, plazo_s=self._cfg.plazo_vacio_s)
        if estado is not EstadoUnidad.EMPTY:
            raise ScopeError(f"unidad {handle.unidad} sigue sin EMPTY; lease retenido")
        if not await self._namespace_vacio(uid):
            self._cuarentena[uid] = "namespace por UID contiene unidades/jobs"
            raise ScopeError(f"namespace de uid {uid} no está vacío; lease retenido")
        await self._liberar_lease(uid, handle.intento)
        self._handles.pop(uid, None)
        for clave, _ in resultados_uid:
            self._resultados_limpieza.pop(clave, None)
        self._cuarentena.pop(uid, None)
        if handle.drain_error is not None:
            raise ScopeError(f"stderr del cliente de uid {uid} no se pudo drenar") from handle.drain_error
        return int(handle.wait_rc)

    @staticmethod
    async def _drenar_stderr(stderr, cola: bytearray) -> None:
        """Consume stderr en bloques y conserva solo la cola acotada para diagnósticos de arranque."""
        while bloque := await stderr.read(4096):
            cola.extend(bloque)
            if len(cola) > 200:
                del cola[:-200]

    @staticmethod
    async def _esperar_drenaje_stderr(proceso) -> None:
        tarea = getattr(proceso, "_faro_stderr_task", None)
        if tarea is not None:
            await tarea
            setattr(proceso, "_faro_stderr_task", None)

    async def esperar_cliente(self, proceso) -> int:
        """Reap systemd-run y cierra su lector de stderr después del fin de la unidad."""
        for handle in self._handles.values():
            if handle.proceso is proceso:
                assert handle.client_task is not None
                await asyncio.shield(handle.client_task)
                if not handle.wait_ok:
                    raise ScopeError("wait del cliente no concluyó")
                return int(handle.wait_rc)
        rc = await proceso.wait()
        await self._esperar_drenaje_stderr(proceso)
        return rc

    # -- nombres y argv ---------------------------------------------------------------------
    def unidad(self, uid: int) -> str:
        return nombre_unidad(uid)

    @staticmethod
    def description_de_intento(intento: str) -> str:
        return f"{PREFIJO_INTENTO}{intento}"

    def argv_arrancar(self, uid: int, comando: Sequence[str], *, intento: str) -> list[str]:
        """`intento` es el id UNICO de este arranque: viaja en la Description de la unidad y es lo unico
        que prueba propiedad (la vida del systemd-run propio no: BLOCK 1). El comando corre bajo
        `env -i`: allowlist cerrada, nada del entorno persistente del manager (MAJOR 3)."""
        if not isinstance(intento, str) or not intento or "\x00" in intento or " " in intento:
            raise ConfigFaroInvalida("el id de intento tiene que ser un texto sin espacios ni NUL")
        comando = _validar_comando(comando)
        c = self._cfg
        return [
            self._systemd_run, "--user", "--quiet", "--pipe", "--collect",
            f"--unit={nombre_unidad_intento(uid, intento)}", f"--slice={c.slice}",
            "--property=Type=exec", "--property=KillMode=control-group", "--property=OOMPolicy=kill",
            "--property=Delegate=no",
            f"--property=Description={self.description_de_intento(intento)}",
            f"--property=MemoryMax={c.memory_max}", f"--property=CPUQuota={c.cpu_quota}",
            f"--property=TasksMax={c.tasks_max}",
            ENV_LIMPIA, "-i",
            *comando,
        ]

    def argv_matar(self, uid: int, *, intento: str) -> list[str]:
        unidad = nombre_unidad_intento(uid, intento)
        return [self._systemctl, "--user", "kill", unidad, "--kill-whom=all", "--signal=SIGKILL"]

    # -- subprocesos ------------------------------------------------------------------------
    async def _correr(self, argv: Sequence[str], *, stdin=asyncio.subprocess.DEVNULL,
                      plazo_s: float | None = None) -> tuple[int, bytes, bytes]:
        """Corre un comando de systemd sin bloquear el bucle; el plazo vencido mata el subproceso."""
        proceso = await self._crear_proceso(
            *argv, stdin=stdin, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=entorno_consulta())
        try:
            salida, error = await asyncio.wait_for(proceso.communicate(), plazo_s or self._plazo_consulta_s)
        except TimeoutError:
            with contextlib.suppress(Exception):
                proceso.kill()
                await proceso.wait()
            raise
        return (proceso.returncode if proceso.returncode is not None else -1), salida, error

    async def _consultar(self, unidad: str, propiedades: Sequence[str]) -> dict[str, str] | None:
        """`systemctl --user show <unidad> --property=...` como diccionario `clave=valor`. Sin `--value`:
        systemd imprime las propiedades en SU orden interno (medido en 259), asi que la correspondencia
        es por clave, nunca posicional. None si la consulta no es concluyente."""
        argv = [self._systemctl, "--user", "show", unidad,
                *[f"--property={p}" for p in propiedades]]
        rc, salida, error = await self._correr(argv)
        if rc != 0:
            logger.warning("systemctl show de %s salio %d: %s", unidad, rc, error.decode(errors="replace").strip())
            return None
        res: dict[str, str] = {}
        for linea in salida.decode(errors="replace").splitlines():
            clave, sep, valor = linea.partition("=")
            if sep:
                res.setdefault(clave, valor)
        if set(res) != {str(p) for p in propiedades}:
            logger.warning("systemctl show de %s devolvio %s, se esperaban %s", unidad, sorted(res),
                           sorted(map(str, propiedades)))
            return None
        return res

    def _leer_cgroup_protegido(self, control_group: str, esperado: ScopeLease | None
                               ) -> tuple[int | None, int | None, int | None, bool] | None:
        """Lee cgroup.events bajo la raíz con dirfd/O_NOFOLLOW y verifica el inode persistido.

        Retorna (populated, dev, ino, path_absent). Solo ausencia de componente se distingue
        de permisos, enlaces, contenido ilegible o sustitución de inode, que son UNKNOWN.
        """
        if (not isinstance(control_group, str) or not control_group.startswith("/")
                or "\x00" in control_group):
            return None
        if esperado is not None and control_group != esperado.cgroup_path:
            return None
        partes = control_group.split("/")[1:]
        if not partes or any(p in ("", ".", "..") for p in partes):
            return None
        flags_dir = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        descriptores: list[int] = []
        try:
            try:
                actual = os.open(self._raiz_cgroup, flags_dir)
            except OSError:
                return None
            descriptores.append(actual)
            for parte in partes:
                try:
                    actual = os.open(parte, flags_dir, dir_fd=actual)
                except FileNotFoundError:
                    return None, None, None, True
                except OSError:
                    return None
                descriptores.append(actual)
            datos_dir = os.fstat(actual)
            if esperado is not None and (datos_dir.st_dev, datos_dir.st_ino) != (esperado.cgroup_dev, esperado.cgroup_ino):
                return None
            try:
                fd_eventos = os.open("cgroup.events", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                     dir_fd=actual)
            except OSError:
                return None
            try:
                datos_eventos = os.fstat(fd_eventos)
                if not stat.S_ISREG(datos_eventos.st_mode):
                    return None
                fragmentos = []
                total = 0
                while True:
                    bloque = os.read(fd_eventos, min(4096, 65537 - total))
                    if not bloque:
                        break
                    fragmentos.append(bloque)
                    total += len(bloque)
                    if total > 65536:
                        return None
                contenido = b"".join(fragmentos)
            finally:
                os.close(fd_eventos)
            try:
                lineas = contenido.decode("utf-8").splitlines()
            except UnicodeDecodeError:
                return None
            valores = [linea.split() for linea in lineas if linea.startswith("populated ")]
            if len(valores) != 1 or len(valores[0]) != 2 or valores[0][1] not in ("0", "1"):
                return None
            return int(valores[0][1]), datos_dir.st_dev, datos_dir.st_ino, False
        finally:
            for descriptor in reversed(descriptores):
                with contextlib.suppress(OSError):
                    os.close(descriptor)

    async def _estado_kernel_lease(self, lease: ScopeLease) -> int | None:
        if lease.stage not in (LeaseStage.ACCEPTED, LeaseStage.CLIENT_CLOSED):
            return None
        lectura = await asyncio.to_thread(self._leer_cgroup_protegido, lease.cgroup_path, lease)
        if lectura is None:
            return None
        if lectura[3]:
            return 0 if lease.stage is LeaseStage.CLIENT_CLOSED else None
        return lectura[0]

    # -- el trivalor ------------------------------------------------------------------------
    async def estado(self, uid: int, *, intento: str | None = None) -> EstadoUnidad:
        if uid in self._cuarentena:
            if self._leases is not None:
                try:
                    retenido = await asyncio.to_thread(self._leases.read, uid)
                    if (retenido.stage in (LeaseStage.ACCEPTED, LeaseStage.CLIENT_CLOSED)
                            and await self._estado_kernel_lease(retenido) == 1):
                        return EstadoUnidad.LIVE
                except LeaseStoreError:  # fail-soft: evidencia de kernel no legible es UNKNOWN y retiene el UID
                    pass
            return EstadoUnidad.UNKNOWN
        lease = None
        if self._leases is not None:
            try:
                lease = await asyncio.to_thread(self._leases.read, uid)
                if lease.stage in (LeaseStage.REQUESTED, LeaseStage.ACCEPTED) and uid not in self._handles:
                    self._cuarentena[uid] = f"{lease.stage.value} sin handle"
                    return EstadoUnidad.UNKNOWN
                if intento is not None and lease.attempt_id != intento:
                    return EstadoUnidad.UNKNOWN
                if intento is None:
                    intento = lease.attempt_id
            except LeaseStoreNotFound:
                if intento is None:
                    return EstadoUnidad.EMPTY if await self._namespace_vacio(uid) else EstadoUnidad.UNKNOWN
            except LeaseStoreError:
                return EstadoUnidad.UNKNOWN
        if intento is None:
            return EstadoUnidad.UNKNOWN
        unidad = nombre_unidad_intento(uid, intento)
        try:
            res = await self._consultar(unidad, ("LoadState", "ActiveState", "ControlGroup"))
            if res is None:
                return EstadoUnidad.UNKNOWN
            load, activo, cg = res["LoadState"].strip(), res["ActiveState"].strip(), res["ControlGroup"].strip()
            if load == "not-found":
                trabajo = await self._hay_trabajo(unidad)
                if trabajo is not False:
                    return EstadoUnidad.UNKNOWN
                # La doble consulta solo corrobora el estado actual del namespace;
                # la causalidad del cliente se demuestra por ACCEPTED + wait, no por este scan.
                res = await self._consultar(unidad, ("LoadState", "ActiveState", "ControlGroup"))
                if res is None:
                    return EstadoUnidad.UNKNOWN
                load, activo, cg = res["LoadState"].strip(), res["ActiveState"].strip(), res["ControlGroup"].strip()
                if load == "not-found":
                    if lease is not None and lease.stage in (LeaseStage.ACCEPTED, LeaseStage.CLIENT_CLOSED):
                        kernel = await self._estado_kernel_lease(lease)
                        if kernel == 1:
                            return EstadoUnidad.LIVE
                        if kernel != 0 or lease.stage is not LeaseStage.CLIENT_CLOSED:
                            return EstadoUnidad.UNKNOWN
                    return EstadoUnidad.EMPTY if await self._namespace_vacio(uid) else EstadoUnidad.UNKNOWN
            if load != "loaded" or activo not in _ACTIVAS | _INACTIVAS:
                return EstadoUnidad.UNKNOWN
            if activo in _ACTIVAS:
                if not cg:
                    return EstadoUnidad.UNKNOWN
                esperado = lease if lease is not None and lease.stage in (LeaseStage.ACCEPTED, LeaseStage.CLIENT_CLOSED) else None
                lectura = await asyncio.to_thread(self._leer_cgroup_protegido, cg, esperado)
                if lectura is None or lectura[3]:
                    return EstadoUnidad.UNKNOWN
                return EstadoUnidad.LIVE if lectura[0] == 1 else EstadoUnidad.UNKNOWN
            if not cg:
                if lease is not None and lease.stage in (LeaseStage.ACCEPTED, LeaseStage.CLIENT_CLOSED):
                    kernel = await self._estado_kernel_lease(lease)
                    if kernel == 1:
                        return EstadoUnidad.LIVE
                    if kernel is None or (kernel == 0 and lease.stage is not LeaseStage.CLIENT_CLOSED):
                        return EstadoUnidad.UNKNOWN
                return EstadoUnidad.EMPTY if await self._namespace_vacio(uid) else EstadoUnidad.UNKNOWN
            esperado = lease if lease is not None and lease.stage in (LeaseStage.ACCEPTED, LeaseStage.CLIENT_CLOSED) else None
            lectura = await asyncio.to_thread(self._leer_cgroup_protegido, cg, esperado)
            if lectura is None:
                return EstadoUnidad.UNKNOWN
            poblado = lectura[0]
            if lectura[3]:
                poblado = 0 if lease is not None and lease.stage is LeaseStage.CLIENT_CLOSED else None
            if poblado == 1:
                return EstadoUnidad.LIVE             # quedan procesos (hijos/nietos) aunque la unidad se apague
            return (EstadoUnidad.EMPTY if await self._namespace_vacio(uid) else EstadoUnidad.UNKNOWN
                    ) if poblado == 0 else EstadoUnidad.UNKNOWN
        except Exception:  # fail-closed: cualquier cosa rara es UNKNOWN, nunca EMPTY
            logger.exception("no se pudo concluir el estado de %s", unidad)
            return EstadoUnidad.UNKNOWN

    async def _namespace_vacio(self, uid: int) -> bool:
        """Dos lecturas completas y estables de unidades/jobs; UID exacto, sin confundir prefijos."""
        patron = re.compile(rf"^{re.escape(PREFIJO_UNIDAD)}{uid}(?:-[0-9a-f]{{32}})?\.service$")
        async def scan() -> tuple[set[str], set[str]] | None:
            rc, salida, error = await self._correr([self._systemctl, "--user", "list-units", "--all",
                                                     "--type=service", "--plain", "--no-legend", "--no-pager"])
            if rc != 0:
                return None
            try:
                text = salida.decode("utf-8")
            except UnicodeDecodeError:
                return None
            units = set()
            for line in text.splitlines():
                fields = line.split()
                if len(fields) < 5 or not fields[0].endswith((".service", ".socket", ".target", ".slice", ".scope", ".timer", ".path", ".mount", ".automount", ".swap")):
                    return None
                name = fields[0]
                if name.startswith(f"{PREFIJO_UNIDAD}{uid}"):
                    if not patron.fullmatch(name):
                        return None
                    units.add(name)
            rc, jobs, error = await self._correr([self._systemctl, "--user", "list-jobs", "--no-legend", "--no-pager"])
            if rc != 0:
                return None
            try:
                job_units = set()
                for line in jobs.decode("utf-8").splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    if len(parts) != 4 or not parts[0].isdigit() or not parts[1].endswith((".service", ".socket", ".target", ".slice", ".scope", ".timer", ".path", ".mount", ".automount", ".swap")):
                        return None
                    if parts[1].startswith(f"{PREFIJO_UNIDAD}{uid}"):
                        if not patron.fullmatch(parts[1]):
                            return None
                        job_units.add(parts[1])
            except UnicodeDecodeError:
                return None
            return units, job_units
        first = await scan()
        if first is None or first != await scan():
            return False
        return not first[0] and not first[1]

    async def _hay_trabajo(self, unidad: str) -> bool | None:
        """Indica si systemd tiene un job para esta unidad; None si no se pudo leer la cola."""
        rc, salida, error = await self._correr([self._systemctl, "--user", "list-jobs", "--no-legend", "--no-pager"])
        if rc != 0:
            logger.warning("systemctl list-jobs salio %d: %s", rc, error.decode(errors="replace").strip())
            return None
        try:
            return any(len(campos := linea.split()) >= 2 and campos[1] == unidad
                       for linea in salida.decode("utf-8").splitlines())
        except UnicodeDecodeError:
            return None

    async def _identidad(self, uid: int, intento: str) -> _EstadoIdentidad:
        """Clasifica ausencia/propiedad/otra unidad sin confiar en la Description sintetizada de not-found."""
        res = await self._consultar(nombre_unidad_intento(uid, intento), ("LoadState", "Description"))
        if res is None:
            return _EstadoIdentidad.DESCONOCIDA
        carga = res["LoadState"].strip()
        if carga == "not-found":
            return _EstadoIdentidad.AUSENTE
        if carga != "loaded":
            return _EstadoIdentidad.DESCONOCIDA
        if res["Description"] == self.description_de_intento(intento):
            return _EstadoIdentidad.PROPIA
        return _EstadoIdentidad.AJENA

    async def control_group(self, uid: int) -> str | None:
        """La ruta del cgroup que REPORTA systemd para la unidad (lectura, nunca concatenacion)."""
        intento = None
        if self._leases is not None:
            try:
                intento = (await asyncio.to_thread(self._leases.read, uid)).attempt_id
            except LeaseStoreNotFound:
                return None
            except LeaseStoreError:
                return None
        unidad = nombre_unidad(uid) if intento is None else nombre_unidad_intento(uid, intento)
        res = await self._consultar(unidad, ("ControlGroup",))
        return res["ControlGroup"].strip() if res else None

    # -- arranque ---------------------------------------------------------------------------
    async def arrancar(self, uid: int, comando: Sequence[str]) -> object:
        """Registra cada lanzamiento antes del primer await para que el cierre lo pueda cancelar y drenar."""
        if self._cerrando:
            raise ScopeError("el gestor de unidades está cerrando; se rechazan nuevos arranques")
        if self._fd_instancia is None or self._token_instancia is None:
            raise ScopeError("arrancar requiere el lock de instancia adquirido previamente")
        tarea = asyncio.current_task()
        if tarea is None:
            raise RuntimeError("arrancar requiere una tarea asyncio")
        if uid in self._uids_parando or uid in self._cuarentena:
            raise ScopeError(f"uid {uid} está en STOPPING; se conserva la cuarentena")
        self._arranques.add(tarea)
        self._arranques_uid.setdefault(uid, set()).add(tarea)
        try:
            if not self._leases_recuperados:
                await self.recuperar_leases()
            if not await self._namespace_vacio(uid):
                raise ScopeError(f"namespace de uid {uid} ocupado o no concluyente; no se admite arranque")
            return await self._arrancar(uid, comando)
        finally:
            self._arranques.discard(tarea)
            activos = self._arranques_uid.get(uid)
            if activos is not None:
                activos.discard(tarea)
                if not activos:
                    self._arranques_uid.pop(uid, None)

    def bloquear_uid(self, uid: int) -> None:
        """Sin punto de suspensión: evita nuevos starts antes de cualquier operación async."""
        self._uids_parando.add(uid)

    async def cerrar_uid(self, uid: int) -> None:
        self.bloquear_uid(uid)
        handle = self._handles.get(uid)
        if handle is not None and handle.finalize_task is not None and not handle.finalize_task.done():
            # El watcher puede estar esperando wait(). Kill exacto primero para que
            # ambos callers compartan ese mismo wait/drain y resultado durable.
            identidad = await self._identidad(uid, handle.intento)
            if identidad is _EstadoIdentidad.PROPIA and await self.estado(uid, intento=handle.intento) is not EstadoUnidad.EMPTY:
                await self._aceptar_handle(handle)
                await self.matar(uid, intento=handle.intento)
            await asyncio.shield(handle.finalize_task)
            return
        if (handle is not None and handle.close_task is not None
                and not handle.close_task.done()):
            await asyncio.shield(handle.close_task)
            return
        tarea = asyncio.create_task(self._cerrar_uid(uid))
        if handle is not None:
            handle.close_task = tarea
        await asyncio.shield(tarea)

    async def _cerrar_uid(self, uid: int) -> None:
        """Detiene starts del UID, usa el handle capturado y solo libera con prueba causal + EMPTY."""
        self.bloquear_uid(uid)
        actuales = set(self._arranques_uid.get(uid, ()))
        tarea_actual = asyncio.current_task()
        actuales.discard(tarea_actual)
        for tarea in actuales:
            if not tarea.done():
                tarea.cancel()
        if actuales:
            resultados = await asyncio.gather(*actuales, return_exceptions=True)
            errores = [r for r in resultados if isinstance(r, BaseException)
                       and not isinstance(r, asyncio.CancelledError)]
            if errores:
                raise BaseExceptionGroup(f"no se drenaron arranques del uid {uid}", errores)
        # Las tareas de cleanup pertenecen a attempts concretos; no se espera a otros UIDs.
        handle = self._handles.get(uid)
        resultados_uid = [(clave, resultado) for clave, resultado in self._resultados_limpieza.items()
                          if clave[0] == uid]
        for clave, resultado in resultados_uid:
            if isinstance(resultado, BaseException):
                raise ScopeError(f"limpieza fallida para uid {uid}; lease retenido") from resultado
            if resultado is not True:
                raise ScopeError(f"limpieza inconclusa para uid {uid}; lease retenido")
        tareas_uid = {t for t in self._limpiezas if getattr(t, "_faro_uid", None) == uid}
        while tareas_uid:
            tareas = tuple(tareas_uid)
            hechos, pendientes = await asyncio.wait(tareas, timeout=self._plazo_drenaje_s())
            if pendientes:
                raise ScopeError(f"limpiezas del uid {uid} siguen en vuelo; lease retenido")
            for tarea in hechos:
                self._limpiezas.discard(tarea)
                if tarea.cancelled():
                    raise ScopeError(f"limpieza cancelada para uid {uid}; lease retenido")
                error = tarea.exception()
                if error is not None:
                    raise ScopeError(f"limpieza fallida para uid {uid}; lease retenido") from error
                if tarea.result() is not True:
                    raise ScopeError(f"limpieza inconclusa para uid {uid}; lease retenido")
            tareas_uid.difference_update(hechos)
        if handle is None:
            try:
                lease = await asyncio.to_thread(self._leases.read, uid) if self._leases is not None else None
            except LeaseStoreNotFound:
                lease = None
            except LeaseStoreError as exc:
                raise ScopeError(f"lease de uid {uid} no es legible; queda retenido") from exc
            if lease is None:
                if await self.estado(uid) is EstadoUnidad.EMPTY:
                    for clave, _ in resultados_uid:
                        self._resultados_limpieza.pop(clave, None)
                    self._uids_parando.discard(uid)
                    return
                raise ScopeError(f"uid {uid} tiene namespace sin lease/handle; queda en cuarentena")
            if lease.stage is LeaseStage.PREPARED:
                if (await self.estado(uid, intento=lease.attempt_id) is EstadoUnidad.EMPTY
                        and await self._namespace_vacio(uid)):
                    await self._liberar_lease(uid, lease.attempt_id)
                    return
            self._cuarentena[uid] = f"{lease.stage.value} sin handle causal"
            raise ScopeError(f"uid {uid} conserva REQUESTED sin handle; no se puede probar cierre causal")
        lease = await asyncio.to_thread(self._leases.read, uid)
        if lease.attempt_id != handle.intento:
            self._cuarentena[uid] = "handle y lease pertenecen a attempts distintos"
            raise ScopeError(f"handle/lease de uid {uid} no coinciden")
        if lease.stage is LeaseStage.CLIENT_CLOSED:
            # El wait y el CAS ACCEPTED→CLIENT_CLOSED ya sucedieron en el cierre natural.
            # Reintentar solo reconcilia la unidad/cgroup durable; nunca vuelve a esperar al proceso.
            if not handle.wait_ok or not handle.materialization_seen:
                self._cuarentena[uid] = "CLIENT_CLOSED sin handle/espera/aceptación verificables"
                raise ScopeError("CLIENT_CLOSED no coincide con el handle causal conservado")
            identidad = await self._identidad(uid, handle.intento)
            if identidad is _EstadoIdentidad.PROPIA:
                try:
                    await self.matar(uid, intento=handle.intento)
                except Exception:
                    # Puede haberse recolectado entre show y kill. Solo AUSENTE exacto
                    # permite seguir a las pruebas durables de kernel y namespace.
                    identidad = await self._identidad(uid, handle.intento)
                    if identidad is not _EstadoIdentidad.AUSENTE:
                        self._cuarentena[uid] = "kill de CLIENT_CLOSED no concluyente"
                        raise ScopeError("la unidad CLIENT_CLOSED sigue presente o incierta")
            elif identidad is not _EstadoIdentidad.AUSENTE:
                self._cuarentena[uid] = "identidad CLIENT_CLOSED no concluyente"
                raise ScopeError("no se puede reconciliar la unidad CLIENT_CLOSED")
            if await self.esperar_vacio(uid, intento=handle.intento,
                                        plazo_s=self._cfg.plazo_vacio_s) is not EstadoUnidad.EMPTY:
                self._cuarentena[uid] = "CLIENT_CLOSED sin prueba kernel EMPTY"
                raise ScopeError(f"unidad/cgroup de uid {uid} sigue poblado o incierto")
            if not await self._namespace_vacio(uid):
                self._cuarentena[uid] = "namespace por UID contiene unidades/jobs"
                raise ScopeError(f"namespace de uid {uid} no está vacío; lease retenido")
            await self._liberar_lease(uid, handle.intento)
            self._handles.pop(uid, None)
            for clave, _ in resultados_uid:
                self._resultados_limpieza.pop(clave, None)
            self._cuarentena.pop(uid, None)
            return
        identidad = await self._identidad(uid, handle.intento)
        if identidad is _EstadoIdentidad.PROPIA:
            await self._aceptar_handle(handle)
            if await self.estado(uid, intento=handle.intento) is not EstadoUnidad.EMPTY:
                await self.matar(uid, intento=handle.intento)
        elif identidad in (_EstadoIdentidad.AJENA, _EstadoIdentidad.DESCONOCIDA):
            raise ScopeError(f"identidad de unidad de uid {uid} cambió; se retiene lease")
        if not handle.materialization_seen:
            self._cuarentena[uid] = "REQUESTED sin observación propia de materialización"
            raise ScopeError("REQUESTED sin prueba causal suficiente; lease retenido")
        assert handle.client_task is not None
        await asyncio.shield(handle.client_task)
        if not handle.wait_ok:
            raise ScopeError(f"wait de uid {uid} no concluyó; lease retenido")
        lease = await asyncio.to_thread(self._leases.read, uid)
        if lease.attempt_id != handle.intento:
            raise ScopeError(f"handle/lease de uid {uid} no coinciden")
        if lease.stage is not LeaseStage.ACCEPTED:
            self._cuarentena[uid] = f"{lease.stage.value} sin prueba ACCEPTED"
            raise ScopeError("lease sin ACCEPTED; cuarentena retenida")
        await self._transicionar_lease(uid, handle.intento, LeaseStage.ACCEPTED, LeaseStage.CLIENT_CLOSED)
        if await self.esperar_vacio(uid, intento=handle.intento, plazo_s=self._cfg.plazo_vacio_s) is not EstadoUnidad.EMPTY:
            self._cuarentena[uid] = "unidad del handle sigue poblada"
            raise ScopeError(f"unidad de uid {uid} no confirma EMPTY; lease retenido")
        if not await self._namespace_vacio(uid):
            self._cuarentena[uid] = "namespace por UID contiene unidades/jobs"
            raise ScopeError(f"namespace de uid {uid} no está vacío; lease retenido")
        await self._liberar_lease(uid, handle.intento)
        self._handles.pop(uid, None)
        for clave, _ in resultados_uid:
            self._resultados_limpieza.pop(clave, None)
        if handle.drain_error is not None:
            raise ScopeError(f"stderr de uid {uid} no se pudo drenar tras cleanup seguro") from handle.drain_error

    async def _arrancar(self, uid: int, comando: Sequence[str]) -> object:
        """Arranca la unidad del uid con `comando` (la elevacion + bwrap ya van dentro) y no la da por
        lista hasta VERIFICARLA activa, con los tres limites efectivos y con la DESCRIPTION de ESTE
        intento: la identidad es determinista, no se infiere por temporizacion ni por la vida del
        systemd-run propio. Lo arrancado y no verificado se mata SOLO si sigue llevando nuestra
        Description; una unidad de otra ejecucion, jamas. La creacion del subproceso corre como TAREA
        propia: una cancelacion nunca deja un hijo sin dueno — la limpieza termina el hijo (cierra la
        ventana de creacion) y luego vigila por identidad, como tarea registrada que
        `esperar_limpiezas` completa aunque la cancelacion se repita (MAJOR 2)."""
        intento = uuid.uuid4().hex
        unidad = nombre_unidad_intento(uid, intento)
        completo = self.argv_arrancar(uid, comando, intento=intento)
        await self._reservar_lease(uid, intento)
        if not await self._namespace_vacio(uid):
            await self._liberar_lease(uid, intento)  # PREPARED prueba que no se llamó a systemd-run.
            raise ScopeError(f"namespace de uid {uid} dejó de estar vacío antes del REQUESTED")
        # Una consulta/lease persistidos como PREPARED prueban que la llamada externa
        # todavía no pudo empezar. Desde aquí toda incertidumbre queda en cuarentena.
        await self._transicionar_lease(uid, intento, LeaseStage.PREPARED, LeaseStage.REQUESTED)
        creacion = asyncio.create_task(self._crear_systemd_run(
            *completo, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE, env=entorno_consulta()))
        try:
            proceso = await asyncio.shield(creacion)
        except asyncio.CancelledError:
            await self._limpiar_cancelacion(uid, intento, creacion)
            raise
        except Exception as exc:
            tarea = asyncio.create_task(self._reconciliar_sin_handle(uid, intento))
            self._registrar_limpieza(tarea, uid, intento)
            try:
                resultado = await asyncio.wait_for(asyncio.shield(tarea), self._plazo_drenaje_s())
            except TimeoutError as espera:
                raise ScopeError(f"{unidad} falló al crear y su efecto externo sigue en reconciliación; UID retenido") from espera
            if resultado is not True:
                raise ScopeError(f"{unidad} falló al crear y no se confirmó estado seguro") from exc
            self._limpiezas.discard(tarea)
            raise ScopeError(f"no se pudo lanzar {unidad}: {type(exc).__name__}") from exc
        self._registrar_handle(uid, intento, unidad, proceso)
        if getattr(proceso, "returncode", None) is not None:
            # systemd-run murio de entrada (la unidad ya existe — perdio la carrera por el nombre —,
            # propiedad rechazada, manager caido): no hay unidad NUESTRA que limpiar, y la que exista
            # es de OTRA ejecucion: no se toca.
            motivo = await self._error(proceso)
            await self._limpiar_en_tarea(uid, intento, proceso)
            raise ScopeError(f"{unidad} no arranco: rc={proceso.returncode} {motivo}")
        try:
            return await self._verificar(proceso, unidad, intento)
        except asyncio.CancelledError:
            await self._limpiar_en_tarea(uid, intento, proceso)
            raise
        except _UnidadAjena as exc:
            raise ScopeError(str(exc)) from exc      # no se mata: la unidad activa es de otra ejecucion
        except Exception as exc:
            await self._limpiar_en_tarea(uid, intento, proceso)
            raise ScopeError(f"{unidad} no se pudo verificar ({exc})") from exc

    async def _verificar(self, proceso, unidad: str, intento: str) -> object:
        fin = time.monotonic() + self._cfg.plazo_arranque_s
        while True:
            rc = getattr(proceso, "returncode", None)
            res = await self._consultar(unidad, ("ActiveState", "ControlGroup", "Description", "Delegate",
                                                 "MemoryMax", "CPUQuotaPerSecUSec", "TasksMax"))
            if res is not None and res["ActiveState"].strip() == "active" and res["ControlGroup"].strip():
                if res["Description"] != self.description_de_intento(intento):
                    # La unidad activa bajo este nombre lleva OTRA identidad: es de otro intento
                    # (ganamos/perdimos la carrera contra el). No se toca. (BLOCK 1)
                    raise _UnidadAjena(f"{unidad} esta activa con Description {res['Description']!r}:"
                                       " es de otro intento")
                self._comparar_limites(res, unidad)
                if res["Delegate"].strip().lower() != "no":
                    raise RuntimeError(f"Delegate efectivo {res['Delegate'].strip()!r} != 'no'")
                for handle in self._handles.values():
                    if handle.unidad == unidad and handle.intento == intento:
                        await self._aceptar_handle(handle)
                return proceso
            if rc is not None:
                # con --pipe, systemd-run solo termina antes que el servicio si ALGO fallo en el arranque
                raise RuntimeError(f"systemd-run termino con rc={rc} sin unidad activa: {await self._error(proceso)}")
            if time.monotonic() >= fin:
                raise RuntimeError(f"la unidad no quedo activa en {self._cfg.plazo_arranque_s}s "
                                   f"(ActiveState={res['ActiveState'].strip() if res else 'sin consulta'})")
            await asyncio.sleep(self._intervalo_s)

    def _comparar_limites(self, res: dict[str, str], unidad: str) -> None:
        """Los tres limites EFECTIVOS que reporta systemd tienen que ser exactamente los configurados:
        un infinity o un valor distinto significa que la unidad no es la que se pidio."""
        esperados = (("MemoryMax", self._cfg.bytes_memoria(), int),
                     ("CPUQuotaPerSecUSec", self._cfg.usec_cpu(), tiempospan_a_us),
                     ("TasksMax", self._cfg.tareas(), int))
        for nombre, esperado, parsear in esperados:
            try:
                efectivo = parsear(res[nombre].strip())
            except (ValueError, KeyError):
                raise RuntimeError(f"{nombre} ilegible: {res.get(nombre)!r}") from None
            if efectivo != esperado:
                raise RuntimeError(f"{nombre} efectivo {res[nombre].strip()!r} != configurado {esperado}")

    async def _error(self, proceso) -> str:
        cola = getattr(proceso, "_faro_stderr_tail", None)
        if cola is not None:
            await asyncio.wait_for(self._esperar_drenaje_stderr(proceso), 1.0)
            return bytes(cola).decode(errors="replace").strip()[-200:]
        try:
            _, err = await asyncio.wait_for(proceso.communicate(), 1.0)
            return err.decode(errors="replace").strip()[:200]
        except Exception:  # fail-soft: solo falta diagnóstico stderr; el error original de arranque se conserva
            return ""

    async def _limpiar_cancelacion(self, uid: int, intento: str, creacion: asyncio.Task) -> None:
        """Entrega la limpieza a una tarea propia antes del primer await cancelable.

        Una segunda cancelación del llamador puede cortar esta espera, pero no la tarea registrada:
        ésta conserva el spawn hasta recibir su handle y luego cierra la ventana de creación antes de
        vigilar la unidad por identidad. No se usa un timeout para declarar ausente un hijo que aún no
        entregó su handle.
        """
        tarea = asyncio.create_task(self._resolver_cancelacion(uid, intento, creacion))
        self._registrar_limpieza(tarea, uid, intento)
        try:
            resultado = await asyncio.wait_for(asyncio.shield(tarea), self._plazo_drenaje_s())
            if resultado is not True:
                raise ScopeError(f"limpieza de {nombre_unidad(uid)} no pudo confirmar estado seguro")
            self._limpiezas.discard(tarea)
        except TimeoutError as exc:
            raise ScopeError(f"limpieza de {nombre_unidad(uid)} sigue en curso; UID retenido") from exc
        except asyncio.CancelledError:
            raise                    # la tarea registrada conserva el ownership hasta completar

    async def _resolver_cancelacion(self, uid: int, intento: str, creacion: asyncio.Task) -> bool:
        """Espera el resultado real del spawn y reconcilia el manager incluso si la entrega del handle
        falla después de que la solicitud externa haya podido surtir efecto."""
        try:
            proceso = await creacion
        except Exception:  # fail-soft: spawn ambiguo va a reconciliación sin handle y permanece en cuarentena
            return await self._reconciliar_sin_handle(uid, intento)
        self._registrar_handle(uid, intento, nombre_unidad_intento(uid, intento), proceso)
        return await self._terminar_y_liberar(uid, intento, proceso)

    async def _terminar_y_liberar(self, uid: int, intento: str, proceso) -> bool:
        seguro = await self._terminar_intento(uid, intento, proceso)
        if seguro:
            handle = self._handles.get(uid)
            if (handle is None or handle.intento != intento or not handle.wait_ok
                    or not handle.materialization_seen):
                return False
            lease = await asyncio.to_thread(self._leases.read, uid)
            if lease.stage is not LeaseStage.ACCEPTED:
                return False
            await self._transicionar_lease(uid, intento, LeaseStage.ACCEPTED, LeaseStage.CLIENT_CLOSED)
            if not await self._namespace_vacio(uid):
                return False
            await self._liberar_lease(uid, intento)
            self._handles.pop(uid, None)
            if handle.drain_error is not None:
                raise ScopeError(f"stderr del cliente de uid {uid} no se pudo drenar") from handle.drain_error
        return seguro

    async def _limpiar_en_tarea(self, uid: int, intento: str, proceso) -> None:
        """La limpieza corre como TAREA propia y registrada: una cancelacion (incluso repetida) de quien
        llamo no la interrumpe; se espera aqui bajo shield y, si la cancelacion llega ahi tambien,
        `esperar_limpiezas` (o el apagado del canal de control) la completa igual."""
        tarea = asyncio.create_task(self._terminar_y_liberar(uid, intento, proceso))
        self._registrar_limpieza(tarea, uid, intento)
        try:
            resultado = await asyncio.wait_for(asyncio.shield(tarea), self._plazo_drenaje_s())
            if resultado is not True:
                raise ScopeError(f"limpieza de {nombre_unidad(uid)} no pudo confirmar estado seguro")
            self._limpiezas.discard(tarea)
        except TimeoutError as exc:
            raise ScopeError(f"limpieza de {nombre_unidad(uid)} sigue en curso; UID retenido") from exc
        except asyncio.CancelledError:
            raise                    # la tarea sigue en background; nadie la cancela

    async def _terminar_intento(self, uid: int, intento: str, proceso) -> bool:
        """Reconcilia una cancelación usando el lease y el cierre causal del cliente systemd-run.

        No mata al cliente: primero espera la unidad propia o que systemd-run termine por sí mismo.
        Solo cuando el cliente terminó y la identidad/job confirman ausencia se puede liberar el UID.
        """
        unidad = nombre_unidad_intento(uid, intento)
        ultima_identidad = _EstadoIdentidad.DESCONOCIDA
        ultimo_trabajo: bool | None = None
        if proceso is None:
            return False
        while True:
            try:
                identidad = await self._identidad(uid, intento)
                ultima_identidad = identidad
                cliente_termino = getattr(proceso, "returncode", None) is not None
                if identidad is _EstadoIdentidad.PROPIA:
                    handle = self._handles.get(uid)
                    if handle is not None and handle.intento == intento:
                        try:
                            await self._aceptar_handle(handle)
                        except Exception:  # fail-soft: conserva REQUESTED, contiene la unidad exacta y no declara cierre
                            # Intento propio pero inválido: contiene la unidad exacta,
                            # conserva REQUESTED y no declara cierre causal.
                            await self.matar(uid, intento=intento)
                            return False
                    estado = await self.estado(uid, intento=intento)
                    if estado is not EstadoUnidad.EMPTY:
                        await self.matar(uid, intento=intento)
                        estado = await self.esperar_vacio(uid, intento=intento, plazo_s=self._plazo_consulta_s)
                    if estado is EstadoUnidad.EMPTY:
                        # --pipe mantiene vivo systemd-run hasta concluir la unidad. Esperarlo
                        # cierra el canal causal antes de liberar el UID y reap el hijo local.
                        await self.esperar_cliente(proceso)
                        if getattr(proceso, "returncode", None) is not None:
                            return True
                elif identidad is _EstadoIdentidad.AJENA:
                    if cliente_termino:
                        return False
                elif identidad is _EstadoIdentidad.AUSENTE and cliente_termino:
                    trabajo = await self._hay_trabajo(unidad)
                    ultimo_trabajo = trabajo
                    if trabajo is False:
                        # Releer tras consultar jobs puede descubrir una unidad ya materializada,
                        # pero no es una barrera causal: sin handle el lease REQUESTED se conserva.
                        identidad = await self._identidad(uid, intento)
                        ultima_identidad = identidad
                        if identidad is _EstadoIdentidad.AUSENTE:
                            return True
                        if identidad is _EstadoIdentidad.AJENA:
                            return False
                        if identidad is _EstadoIdentidad.PROPIA:
                            await self.matar(uid, intento=intento)
                            estado = await self.esperar_vacio(uid, intento=intento, plazo_s=self._plazo_consulta_s)
                            if estado is EstadoUnidad.EMPTY and getattr(proceso, "returncode", None) is not None:
                                return True
            except Exception:  # fail-soft: reaper registra error, conserva lease/UID y reintenta la observación
                logger.exception("limpieza de %s falló; el reaper conserva el UID en cuarentena", unidad)
            logger.warning("limpieza de %s sigue inconclusa: identidad=%s job=%s; se conserva el reaper",
                           unidad, ultima_identidad.value, ultimo_trabajo)
            await asyncio.sleep(self._intervalo_s)

    async def _reconciliar_sin_handle(self, uid: int, intento: str) -> bool:
        """Contiene solo una unidad exacta observada; sin sender causal siempre retiene REQUESTED."""
        unidad = nombre_unidad_intento(uid, intento)
        try:
            identidad = await self._identidad(uid, intento)
            trabajo = await self._hay_trabajo(unidad)
            if identidad is _EstadoIdentidad.AUSENTE:
                identidad = await self._identidad(uid, intento)
            if identidad is _EstadoIdentidad.PROPIA:
                await self.matar(uid, intento=intento)
            vacio = await self._namespace_vacio(uid)
        except Exception:  # fail-soft: consulta no concluyente queda en cuarentena y no autoriza liberación
            logger.exception("no se pudo inspeccionar spawn sin handle de %s", unidad)
            identidad, trabajo, vacio = _EstadoIdentidad.DESCONOCIDA, None, False
        self._cuarentena[uid] = (f"spawn sin handle: identidad={identidad.value}, job={trabajo}, namespace_empty={vacio}")
        logger.error("spawn de %s sin handle causal; UID retenido en cuarentena durable", unidad)
        return False

    def _plazo_drenaje_s(self) -> float:
        return max(self._cfg.plazo_arranque_s * 2 + self._plazo_consulta_s, self._plazo_consulta_s)

    async def _descripcion(self, uid: int) -> str | None:
        """La Description que reporta systemd para la unidad del uid; None si la consulta no concluye."""
        res = await self._consultar(nombre_unidad(uid), ("Description",))
        return res["Description"] if res is not None else None

    async def esperar_limpiezas(self) -> None:
        """Cierra admisión, cancela y drena lanzamientos activos, y no absorbe limpiezas inciertas."""
        self.cerrar_admision()
        plazo = max(self._cfg.plazo_arranque_s * 2 + self._plazo_consulta_s, self._plazo_consulta_s)
        actual = asyncio.current_task()
        arranques = [t for t in self._arranques if t is not actual and not t.done()]
        for tarea in arranques:
            tarea.cancel()
        if arranques:
            _, pendientes = await asyncio.wait(arranques, timeout=plazo)
            if pendientes:
                raise ScopeError("lanzamientos activos no terminaron su cancelación; apagado queda incompleto")

        while self._limpiezas:
            tareas = tuple(self._limpiezas)
            hechos, pendientes = await asyncio.wait(tareas, timeout=plazo)
            if pendientes:
                raise ScopeError("limpiezas siguen en vuelo; UID queda retenido y apagado queda incompleto")
            fallidas = []
            for tarea in hechos:
                try:
                    resultado = tarea.result()
                except BaseException as exc:  # fail-soft: se agrega al error de apagado y las demás limpiezas continúan
                    fallidas.append(type(exc).__name__)
                    continue
                if resultado is True:
                    self._limpiezas.discard(tarea)
                else:
                    fallidas.append("inconclusa")
            if fallidas:
                raise ScopeError("limpiezas de unidad no concluyentes (%s); UID queda retenido" % ", ".join(fallidas))
        fallidas = []
        for clave, resultado in tuple(self._resultados_limpieza.items()):
            if isinstance(resultado, BaseException) or resultado is not True:
                fallidas.append(f"uid={clave[0]} attempt={clave[1]}")
            else:
                self._resultados_limpieza.pop(clave, None)
        if fallidas:
            raise ScopeError("limpiezas de unidad no concluyentes (%s); UID queda retenido" % ", ".join(fallidas))

    def cerrar_admision(self) -> None:
        """Impide nuevos lanzamientos antes de iniciar la cancelación del servicio."""
        self._cerrando = True

    # -- muerte -----------------------------------------------------------------------------
    async def matar(self, uid: int, *, intento: str) -> None:
        """SIGKILL a TODOS los procesos de la unidad (KillMode=control-group, --kill-whom=all), sin tocar
        a las hermanas. Fallar es visible (ScopeError); quien llama decide con `estado`."""
        unidad = nombre_unidad_intento(uid, intento)
        identidad = await self._identidad(uid, intento)
        if identidad is not _EstadoIdentidad.PROPIA:
            raise ScopeError(f"se rechaza matar {unidad}: el attempt esperado ya no es propietario")
        rc, _, error = await self._correr(self.argv_matar(uid, intento=intento))
        if rc != 0:
            raise ScopeError(f"no se pudo matar {unidad}: rc={rc} {error.decode(errors='replace').strip()[:200]}")

    async def esperar_vacio(self, uid: int, *, intento: str | None = None, plazo_s: float | None = None) -> EstadoUnidad:
        """Espera la confirmacion EMPTY de la unidad del uid. Vence el plazo, devuelve lo que haya
        (LIVE/UNKNOWN): NUNCA EMPTY sin confirmacion; el uid queda retenido."""
        fin = time.monotonic() + (plazo_s if plazo_s is not None else self._cfg.plazo_vacio_s)
        if intento is None and self._leases is not None:
            try:
                intento = (await asyncio.to_thread(self._leases.read, uid)).attempt_id
            except LeaseStoreNotFound:  # fail-soft: sin lease se consulta namespace; la ausencia no se presume vacía
                pass
            except LeaseStoreError:
                return EstadoUnidad.UNKNOWN
        estado = await self.estado(uid, intento=intento)
        while estado is not EstadoUnidad.EMPTY and time.monotonic() < fin:
            await asyncio.sleep(self._intervalo_s)
            estado = await self.estado(uid, intento=intento)
        return estado
