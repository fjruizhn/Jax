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

EL UID ES EL NOMBRE. `ServidorControl` solo permite una ejecucion viva por uid; la unidad determinista
`faro-jaula-u<uid>.service` hace que (a) systemd arbitre atomicamente dos arranques concurrentes con el
mismo uid (el segundo falla: la unidad ya existe) y (b) un reinicio del Faro recupere el estado consultando
por el nombre, sin memoria propia.

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
"""
from __future__ import annotations

import asyncio
import contextlib
import enum
import logging
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigFaroInvalida

logger = logging.getLogger(__name__)

SYSTEMD_RUN = "/usr/bin/systemd-run"
SYSTEMCTL = "/usr/bin/systemctl"
RAIZ_CGROUP = Path("/sys/fs/cgroup")
PREFIJO_UNIDAD = "faro-jaula-u"
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


class EstadoUnidad(enum.Enum):
    """El trivalor del contrato: solo EMPTY libera; LIVE y UNKNOWN conservan/rechazan el uid."""
    LIVE = "live"
    EMPTY = "empty"
    UNKNOWN = "unknown"


class ScopeError(Exception):
    """Un arranque, verificacion o kill que no se pudo completar: falla cerrado."""


class _UnidadAjena(Exception):
    """La unidad activa bajo el nombre de este uid NO es de esta llamada (perdio la carrera): no se mata."""


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
                   tasks_max=pedir("JAX_FARO_SCOPE_TASKS_MAX"),
                   slice=opcional("JAX_FARO_SCOPE_SLICE", str, "faro.slice"),
                   plazo_arranque_s=opcional("JAX_FARO_SCOPE_PLAZO_ARRANQUE_S", float, 10.0),
                   plazo_vacio_s=opcional("JAX_FARO_SCOPE_PLAZO_VACIO_S", float, 10.0))


class ScopeManager:
    """Arranca, consulta y mata la unidad transitoria de cada uid. Todo es asincrono (subprocesos, nunca
    bloqueante) y falla cerrado: lo que no se puede concluir es UNKNOWN, y un arranque que no se puede
    VERIFICAR (activa, con los tres limites efectivos) se mata y se reporta como error."""

    def __init__(self, cfg: ConfigScope, *, systemd_run: str = SYSTEMD_RUN, systemctl: str = SYSTEMCTL,
                 raiz_cgroup: Path = RAIZ_CGROUP, crear_proceso: Callable | None = None,
                 intervalo_s: float = 0.05, plazo_consulta_s: float = _PLAZO_CONSULTA_S):
        self._cfg = cfg
        self._systemd_run = systemd_run
        self._systemctl = systemctl
        self._raiz_cgroup = Path(raiz_cgroup)
        self._crear_proceso = crear_proceso if crear_proceso is not None else asyncio.create_subprocess_exec
        self._intervalo_s = intervalo_s
        self._plazo_consulta_s = plazo_consulta_s

    # -- nombres y argv ---------------------------------------------------------------------
    def unidad(self, uid: int) -> str:
        return nombre_unidad(uid)

    def argv_arrancar(self, uid: int, comando: Sequence[str]) -> list[str]:
        comando = _validar_comando(comando)
        c = self._cfg
        return [
            self._systemd_run, "--user", "--quiet", "--pipe", "--collect",
            f"--unit={nombre_unidad(uid)}", f"--slice={c.slice}",
            "--property=Type=exec", "--property=KillMode=control-group", "--property=OOMPolicy=kill",
            f"--property=MemoryMax={c.memory_max}", f"--property=CPUQuota={c.cpu_quota}",
            f"--property=TasksMax={c.tasks_max}",
            *comando,
        ]

    def argv_matar(self, uid: int) -> list[str]:
        return [self._systemctl, "--user", "kill", nombre_unidad(uid), "--kill-whom=all", "--signal=SIGKILL"]

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

    def _poblado(self, control_group: str) -> int | None:
        """`populated` de `cgroup.events`: 1 hay procesos, 0 no hay, None no se puede concluir. Un
        cgroup cuya lectura da FileNotFoundError ya fue ELIMINADO: eso es 0 (vacio), no un error."""
        eventos = self._raiz_cgroup / control_group.removeprefix("/") / "cgroup.events"
        try:
            contenido = eventos.read_text()
        except FileNotFoundError:
            return 0                    # el cgroup ya no existe: vacio ELIMINADO
        except OSError as exc:
            logger.warning("no se puede leer %s: %s", eventos, type(exc).__name__)
            return None
        for linea in contenido.splitlines():
            clave, _, valor = linea.partition(" ")
            if clave == "populated":
                if valor.strip() == "1":
                    return 1
                if valor.strip() == "0":
                    return 0
                return None
        return None

    # -- el trivalor ------------------------------------------------------------------------
    async def estado(self, uid: int) -> EstadoUnidad:
        unidad = nombre_unidad(uid)     # un uid basura es un bug del que llama: no se disfraza de UNKNOWN
        try:
            res = await self._consultar(unidad, ("LoadState", "ActiveState", "ControlGroup"))
            if res is None:
                return EstadoUnidad.UNKNOWN
            load, activo, cg = res["LoadState"].strip(), res["ActiveState"].strip(), res["ControlGroup"].strip()
            if load == "not-found":
                return EstadoUnidad.EMPTY            # el manager confirma que la unidad no existe (y su cgroup ya se vacio: --collect no descarga con procesos dentro)
            if load != "loaded" or activo not in _ACTIVAS | _INACTIVAS:
                return EstadoUnidad.UNKNOWN
            if activo in _ACTIVAS:
                if not cg:
                    return EstadoUnidad.UNKNOWN
                return EstadoUnidad.LIVE if self._poblado(cg) == 1 else EstadoUnidad.UNKNOWN
            if not cg:
                return EstadoUnidad.EMPTY            # inactiva y sin cgroup: no hay donde vivir
            poblado = self._poblado(cg)
            if poblado == 1:
                return EstadoUnidad.LIVE             # quedan procesos (hijos/nietos) aunque la unidad se apague
            return EstadoUnidad.EMPTY if poblado == 0 else EstadoUnidad.UNKNOWN
        except Exception:  # fail-closed: cualquier cosa rara es UNKNOWN, nunca EMPTY
            logger.exception("no se pudo concluir el estado de %s", unidad)
            return EstadoUnidad.UNKNOWN

    async def control_group(self, uid: int) -> str | None:
        """La ruta del cgroup que REPORTA systemd para la unidad (lectura, nunca concatenacion)."""
        res = await self._consultar(nombre_unidad(uid), ("ControlGroup",))
        return res["ControlGroup"].strip() if res else None

    # -- arranque ---------------------------------------------------------------------------
    async def arrancar(self, uid: int, comando: Sequence[str]) -> object:
        """Arranca la unidad del uid con `comando` (la elevacion + bwrap ya van dentro) y no la da por
        lista hasta VERIFICARLA activa con los tres limites efectivos y con NUESTRO systemd-run vivo:
        con `--pipe`, nuestro proceso vivo es lo que prueba que el servicio conectado es el nuestro. Lo
        arrancado y no verificado se mata; una unidad que resulto ser de otra ejecucion, jamas."""
        unidad = nombre_unidad(uid)
        completo = self.argv_arrancar(uid, comando)
        try:
            proceso = await self._crear_proceso(
                *completo, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, env=entorno_consulta())
        except Exception as exc:
            raise ScopeError(f"no se pudo lanzar {unidad}: {type(exc).__name__}") from exc
        if getattr(proceso, "returncode", None) is not None:
            # systemd-run murio de entrada (la unidad ya existe — perdio la carrera por el nombre —,
            # propiedad rechazada, manager caido): no hay unidad NUESTRA que limpiar, y la que exista
            # es de OTRA ejecucion: no se toca.
            raise ScopeError(f"{unidad} no arranco: rc={proceso.returncode} {await self._error(proceso)}")
        try:
            return await self._verificar(proceso, unidad)
        except asyncio.CancelledError:
            await self._matar_mejor_esfuerzo(uid)
            raise
        except _UnidadAjena as exc:
            raise ScopeError(str(exc)) from exc      # no se mata: la unidad activa es de otra ejecucion
        except Exception as exc:
            await self._matar_mejor_esfuerzo(uid)
            raise ScopeError(f"{unidad} no se pudo verificar ({exc})") from exc

    async def _verificar(self, proceso, unidad: str) -> object:
        fin = time.monotonic() + self._cfg.plazo_arranque_s
        while True:
            rc = getattr(proceso, "returncode", None)
            res = await self._consultar(unidad, ("ActiveState", "ControlGroup", "MemoryMax",
                                                 "CPUQuotaPerSecUSec", "TasksMax"))
            if res is not None and res["ActiveState"].strip() == "active" and res["ControlGroup"].strip():
                if rc is not None:
                    # unidad activa PERO nuestro systemd-run ya termino: con --pipe eso es una
                    # contradiccion — la unidad activa es de OTRA ejecucion (perdimos la carrera).
                    raise _UnidadAjena(f"{unidad} esta activa pero nuestro systemd-run termino rc={rc}: no es nuestra")
                self._comparar_limites(res, unidad)
                # un perdedor de la carrera tarda unos ms en morir: asentar antes de dar por lista
                await asyncio.sleep(self._intervalo_s)
                if getattr(proceso, "returncode", None) is not None:
                    raise _UnidadAjena(f"{unidad} esta activa pero systemd-run termino rc={proceso.returncode}")
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
        try:
            _, err = await asyncio.wait_for(proceso.communicate(), 1.0)
            return err.decode(errors="replace").strip()[:200]
        except Exception:  # solo decorar el mensaje de fallo de arranque; sin stderr no afea el error
            return ""

    async def _matar_mejor_esfuerzo(self, uid: int) -> None:
        with contextlib.suppress(Exception):
            await self.matar(uid)

    # -- muerte -----------------------------------------------------------------------------
    async def matar(self, uid: int) -> None:
        """SIGKILL a TODOS los procesos de la unidad (KillMode=control-group, --kill-whom=all), sin tocar
        a las hermanas. Fallar es visible (ScopeError); quien llama decide con `estado`."""
        unidad = nombre_unidad(uid)
        rc, _, error = await self._correr(self.argv_matar(uid))
        if rc != 0:
            raise ScopeError(f"no se pudo matar {unidad}: rc={rc} {error.decode(errors='replace').strip()[:200]}")

    async def esperar_vacio(self, uid: int, *, plazo_s: float | None = None) -> EstadoUnidad:
        """Espera la confirmacion EMPTY de la unidad del uid. Vence el plazo, devuelve lo que haya
        (LIVE/UNKNOWN): NUNCA EMPTY sin confirmacion; el uid queda retenido."""
        fin = time.monotonic() + (plazo_s if plazo_s is not None else self._cfg.plazo_vacio_s)
        estado = await self.estado(uid)
        while estado is not EstadoUnidad.EMPTY and time.monotonic() < fin:
            await asyncio.sleep(self._intervalo_s)
            estado = await self.estado(uid)
        return estado
