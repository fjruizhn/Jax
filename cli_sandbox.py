"""
cli_sandbox -- nucleo comun del confinamiento con bubblewrap para los CLIs de
suscripcion (claude, codex, kimi) y su unico punto de entrada, `run_cli`.

Spec: docs/superpowers/specs/2026-10-01-facetas-por-suscripcion-design.md
(§1 arquitectura, §3 historial, §4 errores, §D titular, §E/§G paso 1).
Decisiones de Fernando 2026-10-01: D-2 (solo user_id 1 y 8), D-4 (Hyde migra a
este nucleo solo si la golden `_hyde_wrap_golden_test.py` queda identica byte a
byte), D-5 (compuerta de titular).

QUE HAY ACA

1. NUCLEO COMUN, sin saber de ningun CLI en particular: `argv_base` (argv de
   bwrap), `env_minimo`, `flock_compartido_adquirir`/`flock_liberar` y `ejecutar` (el
   unico `create_subprocess_exec` del modulo). `hyde_sandbox` lo usa tal cual.
2. PERFILES en codigo (`PERFILES`): claude, codex, kimi. La base solo elegira
   una CLAVE de perfil; nunca una ruta de binario ni flags (la confianza sigue
   a quien escribio el valor).
3. TITULAR (`exigir_titular` / `Titular`): la compuerta de la suscripcion. Un
   Titular solo lo construye `exigir_titular`, caduca a los `TITULAR_TTL_S` y
   `run_cli` lo exige. Es disciplina, no una barrera contra codigo hostil dentro
   del proceso (ver la docstring de `Titular`).
4. `run_cli`: arma el sandbox del perfil, lanza el CLI SIN herramientas, lee
   la salida estructurada y clasifica los errores por evento + exit code.
5. `transporte_efectivo`: el transporte que se deriva del proveedor.

ENTORNO (misma disciplina que hyde_sandbox, B-1 de la auditoria 2026-09-27):
la frontera es el `env=` de `asyncio.create_subprocess_exec`, que recibe el
diccionario de `env_minimo` TAL CUAL, nunca fusionado con os.environ. El argv
no lleva ningun secreto: se lee por /proc/<pid>/cmdline. El system prompt y la
memoria viajan por un directorio por llamada (`/run/jax-cli/<uuid>/`, 0700),
montado en solo lectura como /work, y el prompt del usuario por stdin.

CONFIGURACION (no hay rutas ni SHA de binarios en la base ni en el argv de
nadie): variables de entorno de /etc/jax/.env (root:jaxsvc 640, las escribe
root, cambiarlas exige reiniciar):
  JAX_SUSCRIPCION_TITULARES   "1,8" -- quienes pueden usar la suscripcion.
                              DESVIO CONSCIENTE de §2 del spec: NO en la tabla
                              de configuracion (§D, H-4): una fila la puede
                              escribir cualquier superadmin por el PUT generico.
  JAX_CLI_ROOT                raiz de binarios fijados (default /opt/jax-cli)
  JAX_CLI_<CODEX|KIMI>_VERSION / _SHA256
                              version fijada y el SHA256 del MANIFIESTO de su
                              directorio (`sha_manifiesto`: ruta, modo y sha256 de
                              cada archivo del arbol, no solo el del binario: bwrap
                              monta y ejecuta el directorio entero). La version que
                              se instala se decide en la operacion (paso 11: la que
                              se pruebe, no la 0.159); sin SHA configurado el perfil
                              NO arranca (BinarioAlterado).
  JAX_CLI_CRED_ROOT           raiz de las credenciales de suscripcion
                              (default /srv/jax-data/cli-suscripcion)
  JAX_CLI_RUN_DIR             directorio por llamada (default /run/jax-cli)
  JAX_CLI_LOCK_DIR            locks por perfil. SIN DEFAULT: ausente o vacia, el CLI no
                              arranca (SandboxUnavailable, falla cerrado). El paso de
                              host la pone en /run/jax-locks/cli y la siembra con
                              tmpfiles.d (`d /run/jax-locks/cli 0700 <usuario> -`). Debe
                              ser del euid y sin escritura de grupo/otros. Estos locks
                              son por euid (un solo usuario por CLI); el de Hyde, que
                              comparten dos usuarios, va por grupo en hyde_sandbox
  JAX_CLI_MAX_PROMPT_CHARS    tope del prompt (default 32000); `max_chars` del llamador solo
                              puede bajarlo (min)
  JAX_CLI_<PERFIL>_RANURAS    llamadas concurrentes por perfil, 1..16 (default el del
                              perfil; NO lo decide el llamador de run_cli)
  JAX_CLI_TIMEOUT_MAX_CHAT_S / JAX_CLI_TIMEOUT_MAX_JACOBS_S
                              tope del timeout por entry_point (default 180 / 600); un
                              entry_point sin tope (canary, repl) se rechaza

CACHES (cada uno declara su invalidacion en el mismo commit que lo crea):
  - `_CACHE_SHA`: SHA256 del manifiesto del directorio de un binario, clave = ruta
    del binario, firma = la de TODO el arbol (por entrada: dev, inode, mtime_ns,
    ctime_ns, size, modo, uid). Se invalida solo: si cambia cualquier stat de
    cualquier entrada, se re-hashea. Es la invalidacion que pide el spec §1. En cada
    llamada, ademas, se recorre el arbol entero y se exige que cada entrada sea de
    root y sin escritura de grupo/otros, que ningun symlink salga del arbol, y que los
    ancestros de JAX_CLI_ROOT hasta `/` sean de root y no escribibles.
  - `_CACHE_FEATURES` / `_CACHE_FEATURES_FALLO`: el exito y el fallo de `verificar_features`,
    misma clave (ruta del binario) y misma firma del arbol que `_CACHE_SHA`; cambia el arbol,
    se vuelve a verificar. `features list` corre bajo una ranura del perfil.
  - La lista de titulares y la verificacion en `jax_users` NO se cachean: se
    leen del entorno y de la base en cada llamada (el borrado de un usuario
    surte efecto en la siguiente llamada).

LO QUE ESTE MODULO NO VERIFICO (§8 del spec): el esquema de eventos de
`codex exec --json` en sus errores de cuota/sesion, el esquema de
`kimi --output-format stream-json`, y si `kimi -p` lee el prompt de stdin. El
perfil kimi y el perfil codex quedan con `canal_prompt_verificado=False` y
`run_cli` los rechaza (falla cerrado) hasta que se verifique el canal del system
prompt (prueba manual del §5); los parsers estan marcados.

En honor al Prof. Raul Jacobs.
"""
from __future__ import annotations

import asyncio
import contextvars
import fcntl
import grp
import json
import logging
import math
import os
import re
import secrets
import shutil
import stat
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("cli_sandbox")


# --------------------------------------------------------------------------
# Excepciones
# --------------------------------------------------------------------------

class ErrorCLI(Exception):
    """Base de los errores tipados de `run_cli` (spec §4). `clase` es el
    nombre estable que va a los logs y a la telemetria."""
    clase = "ErrorCLI"


class SandboxUnavailable(ErrorCLI):
    """bwrap no disponible o no ejecutable en runtime. Fail-closed (P10):
    el llamador NO debe atrapar esto para degradar a ejecución sin
    sandbox -- el CLI simplemente no arranca. Es un ErrorCLI para que el log
    de `run_cli` lo registre con su `clase` (auditoria 2026-10-02, MAJOR-2);
    Hyde importa esta misma clase y la atrapa igual que antes (sigue siendo
    una Exception)."""
    clase = "SandboxUnavailable"


class CuotaAgotada(ErrorCLI):
    clase = "CuotaAgotada"


class SesionVencida(ErrorCLI):
    clase = "SesionVencida"


class ErrorProtocolo(ErrorCLI):
    clase = "ErrorProtocolo"


class TimeoutCLI(ErrorCLI, TimeoutError):
    clase = "Timeout"


class LockTimeout(ErrorCLI, TimeoutError):
    clase = "LockTimeout"


class BinarioAlterado(ErrorCLI):
    clase = "BinarioAlterado"


class FeaturesNoPermitidas(ErrorCLI):
    clase = "FeaturesNoPermitidas"


class PerfilNoSoportado(ErrorCLI):
    clase = "PerfilNoSoportado"


class MensajeDemasiadoLargo(ErrorCLI):
    clase = "MensajeDemasiadoLargo"


class TitularNoAutorizado(ErrorCLI):
    """La compuerta de la suscripcion se nego. `codigo` es la clave i18n que
    ve el usuario (`suscripcion_solo_titular`, o `verificacion_no_disponible`
    cuando no se pudo comprobar); `motivo` es solo para el log."""
    clase = "TitularNoAutorizado"

    def __init__(self, codigo: str, motivo: str = ""):
        super().__init__(f"{codigo}: {motivo}" if motivo else codigo)
        self.codigo = codigo
        self.motivo = motivo


# --------------------------------------------------------------------------
# NUCLEO COMUN -- sin saber de ningun CLI
# --------------------------------------------------------------------------

LANG = "C.UTF-8"
SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# /etc puntual para DNS + TLS + NSS -- NUNCA /etc entero (expondria
# /etc/jax/.env, root:fruiz 0660, el grupo fruiz SI tiene lectura real).
ETC_RO_PATHS = (
    "/etc/resolv.conf", "/etc/nsswitch.conf", "/etc/hosts",
    "/etc/ssl", "/etc/passwd", "/etc/group",
)

_BWRAP_BIN = shutil.which("bwrap") or "/usr/bin/bwrap"


def verificar_bwrap(bwrap_bin: str, quien: str) -> None:
    """Fail-closed (P10): sin bwrap ejecutable, `quien` no arranca."""
    if not (bwrap_bin and os.path.isfile(bwrap_bin) and os.access(bwrap_bin, os.X_OK)):
        raise SandboxUnavailable(
            f"bwrap no encontrado o no ejecutable ({bwrap_bin!r}) -- "
            f"{quien} no arranca sin confinamiento (fail-closed, P10)"
        )


def argv_base(bwrap_bin: str, etc_paths: tuple[str, ...] = ETC_RO_PATHS) -> list[str]:
    """Argv comun de bwrap: namespaces, /proc, /dev, /tmp efimero, la base del
    SO en solo lectura y el /etc puntual. SIN --clearenv/--setenv (B-1,
    2026-09-27): la frontera de entorno es el `env` de `env_minimo`, que el
    llamador pasa tal cual a create_subprocess_exec."""
    argv = [
        bwrap_bin,
        "--unshare-all", "--share-net",  # red completa: es la unica forma de que bwrap deje llegar a la API del proveedor
        "--die-with-parent",
        "--new-session",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        # base del SO -- necesaria para que corran node/git/python3/bash/etc.
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
    ]
    for optional_root in ("/lib64", "/bin", "/sbin"):
        if os.path.isdir(optional_root) or os.path.islink(optional_root):
            argv += ["--ro-bind", optional_root, optional_root]
    for etc_path in etc_paths:
        if os.path.exists(etc_path):
            argv += ["--ro-bind", etc_path, etc_path]
    return argv


def env_minimo(
    home: str, extra: Optional[dict[str, str]] = None, *, path: str = SAFE_PATH,
) -> dict[str, str]:
    """Entorno MINIMO y COMPLETO del proceso confinado. El llamador lo pasa
    TAL CUAL como `env=`; nunca se fusiona con os.environ."""
    env = {"HOME": home, "PATH": path, "LANG": LANG}
    if extra:
        env.update(extra)
    return env


def _preparar_dir_locks(directorio: Path) -> int:
    """Devuelve un fd del directorio de locks, ya verificado: es un directorio
    REAL (no un symlink), es del euid del proceso y ni el grupo ni otros pueden
    escribir en el. Lo crea con 0700 si no existe. Cualquier otra cosa es
    SandboxUnavailable: un lock que otro usuario puede reemplazar o sembrar con
    symlinks no da exclusion mutua ni es seguro de abrir (auditoria 2026-10-02,
    MAJOR-3). El llamador cierra el fd. En produccion el directorio lo siembra el
    host con tmpfiles.d (JAX_CLI_LOCK_DIR=/run/jax-locks/cli); este modulo no
    puede crear nada bajo /run, solo bajo un padre que ya sea del euid."""
    try:
        os.makedirs(directorio, mode=0o700, exist_ok=True)
        fd = os.open(directorio, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise SandboxUnavailable(
            f"directorio de locks {str(directorio)!r} no usable ({exc.strerror or type(exc).__name__}) "
            "-- falla cerrado: sin exclusion mutua segura no se lanza"
        ) from None
    st = os.fstat(fd)
    if st.st_uid != os.geteuid() or st.st_mode & 0o022:
        os.close(fd)
        raise SandboxUnavailable(
            f"directorio de locks {str(directorio)!r} inseguro: debe ser del euid ({os.geteuid()}) "
            f"y sin escritura de grupo ni de otros (dueno={st.st_uid}, modo={stat.S_IMODE(st.st_mode):o}); "
            "corregirlo con chown/chmod -- falla cerrado"
        )
    return fd


def _abrir_lock(dir_fd: int, nombre: str):
    """Abre (o crea, 0600) el archivo de lock `nombre` DENTRO del directorio ya
    verificado, sin seguir symlinks (O_NOFOLLOW) y SIN truncar: el lock no
    necesita contenido, y un `open(..., "w")` truncaria el destino de un symlink
    plantado. Exige archivo regular del euid; si no, SandboxUnavailable."""
    try:
        fd = os.open(
            nombre, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=dir_fd,
        )
    except OSError as exc:
        raise SandboxUnavailable(
            f"no se pudo abrir el lock {nombre!r} sin seguir symlinks ({exc.strerror or type(exc).__name__})"
        ) from None
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid():
        os.close(fd)
        raise SandboxUnavailable(f"el lock {nombre!r} no es un archivo regular del euid")
    return os.fdopen(fd, "r+")


def _esperar_flock(fh, timeout: float, descripcion: str, detalle: str, ruta: str):
    """BLOQUEANTE. Sondea `flock(LOCK_EX | LOCK_NB)` sobre `fh` hasta `timeout`;
    agotado, cierra `fh` y lanza TimeoutError (falla cerrado: no se lanza sin
    exclusion mutua real). flock(2) vale igual sobre un fd de solo lectura."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except BlockingIOError:
            if time.monotonic() >= deadline:
                fh.close()
                raise TimeoutError(
                    f"no se pudo adquirir el lock cross-proceso de {descripcion} "
                    f"en {timeout}s ({ruta}) -- {detalle} "
                    "Fail-closed: no se lanza sin exclusion mutua real."
                )
            time.sleep(_LOCK_POLL_S)


TMPFILES_LOCKS = "/etc/tmpfiles.d/jax-locks.conf"


def _linea_tmpfiles(directorio: str, nombre: str, grupo: str) -> str:
    """La linea de `/etc/tmpfiles.d/jax-locks.conf` que siembra el lock `nombre`:
    el modo compartido no crea nada, lo crea el host al arrancar."""
    return f"f {directorio}/{nombre} 0640 root {grupo} -"


def _sin_escritura_ajena(st) -> bool:
    return not st.st_mode & 0o022


def flock_compartido_adquirir(
    directorio: str, nombre: str, grupo: str, timeout: float, descripcion: str, detalle: str = "", *,
    uid_esperado: int = 0, gid_esperado: Optional[int] = None,
):
    """BLOQUEANTE -- solo via asyncio.to_thread. MODO COMPARTIDO del lock: lo usan
    procesos de usuarios distintos (las_manos/jaxsvc y el REPL/fruiz), asi que el
    duenyo no es el euid sino ROOT y el acceso es por GRUPO. Lo siembra
    `tmpfiles.d` (`f <directorio>/<nombre> 0640 root <grupo> -`); este modulo NUNCA
    crea nada (sin O_CREAT).

    Verificaciones, todas sobre fds ya abiertos (fstat, sin carreras por ruta):
      - el directorio se abre con O_RDONLY|O_DIRECTORY|O_NOFOLLOW: uid == uid_esperado,
        gid == el del `grupo` y sin escritura de grupo ni de otros;
      - el archivo se abre RELATIVO a ese fd, O_RDONLY|O_NOFOLLOW|O_CLOEXEC y sin
        O_CREAT (un symlink o un archivo ausente fallan): archivo regular, mismo uid,
        mismo gid y sin escritura de grupo ni de otros. flock(LOCK_EX) funciona sobre
        un fd de solo lectura.
    Cualquier otra cosa es SandboxUnavailable (falla cerrado) y, si falta el
    archivo o el grupo, el mensaje nombra la linea de tmpfiles.d que hay que poner.

    `uid_esperado` (0 en produccion) y `gid_esperado` (None = el del `grupo` real)
    se inyectan por keyword solo para los tests, que no corren como root; no hay
    ningun flag global, y quien llama desde produccion no los pasa."""
    linea = _linea_tmpfiles(directorio, nombre, grupo)
    falta = f"; falta la linea `{linea}` en {TMPFILES_LOCKS}"
    gid = gid_esperado
    if gid is None:
        try:
            gid = grp.getgrnam(grupo).gr_gid
        except KeyError:
            raise SandboxUnavailable(
                f"el grupo {grupo!r} del lock compartido no existe{falta} (y el grupo mismo)"
            ) from None
    try:
        dfd = os.open(directorio, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise SandboxUnavailable(
            f"directorio de locks compartido {directorio!r} no usable ({exc.strerror or type(exc).__name__})"
            f"{falta} (o su linea `d {directorio} 0750 root {grupo} -`)"
        ) from None
    try:
        sd = os.fstat(dfd)
        if sd.st_uid != uid_esperado or sd.st_gid != gid or not _sin_escritura_ajena(sd):
            raise SandboxUnavailable(
                f"directorio de locks compartido {directorio!r} inseguro: debe ser del uid {uid_esperado}, "
                f"del gid {gid} ({grupo!r}) y sin escritura de grupo ni de otros "
                f"(uid={sd.st_uid}, gid={sd.st_gid}, modo={stat.S_IMODE(sd.st_mode):o}) -- falla cerrado"
            )
        try:
            fd = os.open(nombre, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dfd)
        except FileNotFoundError:
            raise SandboxUnavailable(
                f"el lock compartido {nombre!r} no existe en {directorio!r}{falta}"
            ) from None
        except OSError as exc:
            raise SandboxUnavailable(
                f"no se pudo abrir el lock compartido {nombre!r} sin seguir symlinks "
                f"({exc.strerror or type(exc).__name__})"
            ) from None
    finally:
        os.close(dfd)
    st = os.fstat(fd)
    if (
        not stat.S_ISREG(st.st_mode) or st.st_uid != uid_esperado or st.st_gid != gid
        or not _sin_escritura_ajena(st)
    ):
        os.close(fd)
        raise SandboxUnavailable(
            f"el lock compartido {nombre!r} debe ser un archivo regular del uid {uid_esperado}, del gid {gid} "
            f"({grupo!r}) y sin escritura de grupo ni de otros (modo={stat.S_IMODE(st.st_mode):o}) -- falla cerrado"
        )
    return _esperar_flock(os.fdopen(fd, "r"), timeout, descripcion, detalle, f"{directorio}/{nombre}")


def flock_liberar(fh) -> None:
    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    fh.close()


_LOCK_POLL_S = 0.05


async def ejecutar(
    argv: list[str], env: dict[str, str], entrada: bytes, timeout: float, *,
    adquirir: Callable[[float], object], liberar: Callable[[object], None],
    cwd: Optional[str] = None,
):
    """Unico `create_subprocess_exec` del nucleo. `adquirir(timeout)` y
    `liberar(handle)` son BLOQUEANTES (corren en to_thread). Con timeout o
    cancelacion mata el proceso, cosecha el zombie y RE-LANZA sin envolver
    (CancelledError debe seguir siendo CancelledError). Devuelve
    (proc, stdout, stderr) crudos. `env=` se pasa explicito y tal cual."""
    lock_fh = await asyncio.to_thread(adquirir, timeout)
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=entrada), timeout=timeout,
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            proc.kill()
            await proc.wait()
            raise
    finally:
        await asyncio.to_thread(liberar, lock_fh)
    return proc, stdout, stderr


def argv_confinado_cli(
    bwrap_bin: str, *, work_host: str, home_sandbox: str,
    binds_rw: list[tuple[str, str]], binds_ro: list[tuple[str, str]], cmd: list[str],
    dirs: tuple[str, ...] | list[str] = (),
) -> list[str]:
    """Confinamiento comun de los perfiles de CLI (codex, kimi): el argv base,
    un $HOME tmpfs, los binds del perfil y /work (solo lectura) con --chdir.
    Sin repos, sin ~/.nvm, sin nada de /home/fruiz: los CLIs de chat no
    necesitan codigo."""
    argv = argv_base(bwrap_bin)
    argv += ["--tmpfs", home_sandbox]
    for host, dest in binds_ro:
        argv += ["--ro-bind", host, dest]
    for host, dest in binds_rw:
        argv += ["--bind", host, dest]
    argv += ["--ro-bind", work_host, "/work"]
    # Directorios vacios 0700 dentro del sandbox (bajo /tmp, que es un tmpfs propio): p. ej. el
    # CODEX_HOME efimero de `features list`, que el CLI exige que EXISTA.
    for d in dirs:
        argv += ["--perms", "0700", "--dir", d]
    # La raiz que arma bwrap es un tmpfs ESCRIBIBLE (efimero, en RAM): sin esto
    # un `echo x > /etc/x` funciona dentro del sandbox. Se remonta solo lectura
    # DESPUES de crear todos los puntos de montaje; /tmp y el $HOME son tmpfs
    # propios y siguen escribibles. (No se agrega a `argv_base`: el argv de
    # Hyde esta congelado por la golden, D-4.)
    argv += ["--remount-ro", "/", "--chdir", "/work", "--"]
    argv += cmd
    return argv


# --------------------------------------------------------------------------
# TRANSPORTE EFECTIVO (spec §2: derivado del proveedor)
# --------------------------------------------------------------------------

def transporte_efectivo(facet_transport: Optional[str], provider_auth_type: Optional[str]) -> Optional[str]:
    """Si el proveedor es de suscripcion (`auth_type='subprocess'`) el
    transporte es `'subprocess'`; si no, el de la faceta. Una sola funcion para
    los resolvedores, contrato_dispatch, prevuelo_catalogo y el catalogo de
    motores (adenda §C)."""
    if provider_auth_type == "subprocess":
        return "subprocess"
    return facet_transport


# --------------------------------------------------------------------------
# TITULAR (spec §D, D-2, D-5)
# --------------------------------------------------------------------------

#: Puntos de entrada reconocidos. `canary` lo fija solo el codigo de la sonda,
#: dentro del proceso; nunca viene de un request.
ENTRY_POINTS = frozenset({"chat", "canary", "jacobs", "repl"})

TITULARES_ENV = "JAX_SUSCRIPCION_TITULARES"
_RE_LISTA = re.compile(r"^\s*[0-9]+(\s*,\s*[0-9]+)*\s*$")

_SELLO = object()

#: Vigencia de un Titular, en segundos monotonicos desde que `exigir_titular` lo
#: emitio. `run_cli` rechaza uno mas viejo: la verificacion contra `jax_users` es
#: de AHORA, y un titular guardado y reusado minutos o horas despues ya no la
#: respalda (el usuario pudo ser dado de baja). Constante de codigo a proposito
#: (no sale de la base ni de un request): 30 s cubre de sobra el camino
#: `exigir_titular` -> `run_cli` de un mismo pedido de chat.
TITULAR_TTL_S = 30.0

#: Solo vale True dentro de `exigir_titular`, de forma sincrona, mientras
#: construye el Titular. Es lo que hace que cualquier otra construccion
#: (`Titular(...)`, `dataclasses.replace`) termine en TypeError.
_EMITIENDO: contextvars.ContextVar[bool] = contextvars.ContextVar("cli_sandbox_emitiendo", default=False)


@dataclass(frozen=True)
class Titular:
    """Prueba de que `exigir_titular` autorizo a este usuario, en este tenant y
    por este punto de entrada, hace menos de `TITULAR_TTL_S` segundos.

    LA FRONTERA QUE EXISTE, exactamente:
      1. En ejecucion, por las vias normales, solo `exigir_titular` lo construye:
         `Titular(...)`, `dataclasses.replace`, `copy`, `deepcopy` y `pickle`
         terminan en TypeError; el sello y la marca de tiempo no son argumentos del
         constructor; y `run_cli` rechaza uno caducado o de otro entry_point.
      2. NO es una barrera contra codigo que corre DENTRO del proceso: quien importe
         este modulo puede leer `_SELLO`, entrar a `_EMITIENDO`, fijar `emitido_mono`
         con `object.__setattr__` o llamar `object.__new__(Titular)` y obtener un
         Titular valido. Ninguna de esas vias esta cerrada en ejecucion.
      3. Lo que las hace VISIBLES es un control de revision, no de ejecucion:
         policy/tests/test_titular_solo_via_exigir_titular.py falla el CI si, fuera de
         este archivo y de su test, el AST nombra `_SELLO`, `_EMITIENDO` o
         `emitido_mono`, declara una subclase de `Titular` o llama `object.__new__`
         sobre ella. No ve codigo fuera de los repos escaneados (en CI solo jax) ni
         formas que el AST no muestra (`exec`/`eval` de texto, nombres calculados).
    Saltarse la compuerta exige, pues, un acto deliberado que una revision ve, y no un
    descuido; contra un llamador malicioso la defensa es la revision del codigo."""
    user_id: int
    tenant_id: int
    entry_point: str
    _sello: object = field(init=False, default=None, repr=False, compare=False)
    emitido_mono: float = field(init=False, default=0.0, repr=False, compare=False)

    def __post_init__(self):
        if not _EMITIENDO.get():
            raise TypeError("Titular solo lo construye cli_sandbox.exigir_titular")

    def __copy__(self):
        raise TypeError("un Titular no se copia: se vuelve a pedir con exigir_titular")

    def __deepcopy__(self, memo):
        raise TypeError("un Titular no se copia: se vuelve a pedir con exigir_titular")

    def __reduce__(self):
        raise TypeError("un Titular no se serializa: se vuelve a pedir con exigir_titular")

    def __reduce_ex__(self, protocolo):
        raise TypeError("un Titular no se serializa: se vuelve a pedir con exigir_titular")


def _es_entero(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _titulares_configurados() -> frozenset[int]:
    raw = os.environ.get(TITULARES_ENV)
    if raw is None or not _RE_LISTA.match(raw):
        raise TitularNoAutorizado(
            "verificacion_no_disponible", f"{TITULARES_ENV} ausente o con formato invalido",
        )
    return frozenset(int(x) for x in raw.split(","))


def _connect_timeout() -> int:
    try:
        from db_connect_config import db_connect_timeout_seconds
    except ImportError:
        try:
            from jax.core.db_connect_config import db_connect_timeout_seconds
        except ImportError:
            return 10  # mismo default declarado por db_connect_config
    return db_connect_timeout_seconds()


async def _consultar_usuario(user_id: int) -> Optional[dict]:
    """SELECT por clave primaria sobre `jax_users`, una conexion nueva por
    llamada: sin cache. Solo lectura. Devuelve None si la fila no existe."""
    import aiomysql  # lazy: el resto del modulo no necesita la base

    host, port = os.environ.get("JAX_DB_HOST"), os.environ.get("JAX_DB_PORT")
    if not host or not port:
        raise RuntimeError("JAX_DB_HOST/JAX_DB_PORT no estan seteados")
    timeout = _connect_timeout()
    async with asyncio.timeout(timeout):
        conn = await aiomysql.connect(
            host=host, port=int(port),
            user=os.getenv("JAX_DB_USER", ""), password=os.getenv("JAX_DB_PASSWORD", ""),
            db=os.getenv("JAX_DB_NAME", "jax_memory"), charset="utf8mb4",
            autocommit=True, connect_timeout=timeout,
        )
        try:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "SELECT tenant_id, status, deleted_at FROM jax_users WHERE user_id = %s",
                    (user_id,),
                )
                return await cur.fetchone()
        finally:
            conn.close()


async def exigir_titular(user_id, tenant_id, entry_point) -> Titular:
    """Unico punto de paso a la suscripcion. Falla CERRADO: cualquier duda es
    una negativa. Exige (1) entry_point conocido, (2) user_id y tenant_id
    enteros positivos (None se niega), (3) user_id en la lista del entorno,
    (4) en `jax_users`: fila existente, status='active', deleted_at NULL y el
    tenant_id que corresponde. La verificacion de (4) va a la base en cada
    llamada, sin cache."""
    if not isinstance(entry_point, str) or entry_point not in ENTRY_POINTS:
        raise TitularNoAutorizado("suscripcion_solo_titular", "entry_point desconocido")
    if not _es_entero(user_id) or not _es_entero(tenant_id):
        raise TitularNoAutorizado("suscripcion_solo_titular", "identidad ausente o de tipo invalido")
    if user_id not in _titulares_configurados():
        logger.warning("titular negado: user_id=%s entry_point=%s (fuera de la lista)", user_id, entry_point)
        raise TitularNoAutorizado("suscripcion_solo_titular", "usuario fuera de la lista")
    try:
        fila = await _consultar_usuario(user_id)
    except Exception as exc:  # fail-closed: base caida, timeout, config
        logger.warning("titular negado: no se pudo verificar user_id=%s (%s)", user_id, type(exc).__name__)
        raise TitularNoAutorizado("verificacion_no_disponible", type(exc).__name__) from None
    if (
        not fila
        or fila.get("status") != "active"
        or fila.get("deleted_at") is not None
        or fila.get("tenant_id") != tenant_id
    ):
        logger.warning("titular negado: user_id=%s tenant_id=%s no activo o de otro tenant", user_id, tenant_id)
        raise TitularNoAutorizado("suscripcion_solo_titular", "cuenta no activa o de otro tenant")
    token = _EMITIENDO.set(True)
    try:
        t = Titular(user_id=user_id, tenant_id=tenant_id, entry_point=entry_point)
    finally:
        _EMITIENDO.reset(token)
    object.__setattr__(t, "_sello", _SELLO)
    object.__setattr__(t, "emitido_mono", time.monotonic())
    return t


# --------------------------------------------------------------------------
# PERFILES
# --------------------------------------------------------------------------

_CODEX_FLAGS = (
    "exec", "-", "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config",
    "--ignore-rules", "-s", "read-only",
)
# EVIDENCIA 2026-10-02 (ronda 2, codex 0.160.0, CODEX_HOME efimero en /tmp, sin login ni
# cuota): `--disable unified_exec` NO apaga `unified_exec`; `codex features list` lo sigue
# mostrando en `true` con `--disable unified_exec`, con `-c features.unified_exec=false`
# (antes o despues de `features list`), con `features --disable unified_exec list` y con
# `features.unified_exec = false` escrito en el config.toml del CODEX_HOME (`codex features
# disable unified_exec`). No se encontro una forma que lo apague; el flag queda en la lista
# de abajo solo porque no hace dano, y `unified_exec` NO esta en la lista permitida: con el
# binario real, `verificar_features` falla cerrado (junto con otras 26 features activas sin
# clasificar, ver abajo) hasta que un humano decida. Sin esa decision codex no se lanza.
# Herramientas apagadas (spec §1). Nombres tal cual los lista `codex features
# list` de 0.160.0; el golden de _cli_sandbox_test.py los congela y se
# re-verifica al fijar la version (paso 11). `web_search` no es una feature: se
# apaga con -c. NO se usa --dangerously-bypass-approvals-and-sandbox.
_CODEX_DISABLE = (
    "shell_tool", "unified_exec", "apps", "browser_use", "browser_use_external",
    "browser_use_full_cdp_access", "computer_use", "image_generation", "plugins",
    "multi_agent", "memories", "hooks",
    # Auditoria 2026-10-02 (MAJOR-5): ambas vienen activas por defecto en 0.160.
    # `unbounded_connection_retries` reintenta sin tope (equivale al
    # KIMI_CODE_INFINITE_RETRY que el perfil de Kimi NUNCA activa);
    # `daemon_auto_start` levanta un proceso residente fuera del confinamiento por llamada.
    "unbounded_connection_retries", "daemon_auto_start",
)

# Features que PUEDEN quedar activadas con los --disable de arriba puestos: lista
# PERMITIDA, no prohibida. `codex features list` crece con cada version y las
# nuevas vienen activadas por defecto; una herramienta nueva no puede colarse por
# no estar en la lista prohibida. Solo hay features sin ninguna accion sobre
# archivos, procesos ni red: las "removed" (el CLI ya no las ofrece, no hay nada
# que apagar) y las de formato de protocolo/compactacion. Sacada de la salida real
# de `codex features list --disable ...` de 0.160.0; se RE-VERIFICA al fijar la
# version (paso 11) con `verificar_features`. Cualquier otra activada la decide un
# humano: apagarla con --disable (y agregarla a _CODEX_DISABLE) o permitirla aqui.
_CODEX_FEATURES_PERMITIDAS = (
    "collaboration_modes", "item_ids", "resize_all_images", "sqlite", "steer",
    "terminal_resize_reflow", "tool_search_always_defer_mcp_tools", "tui_app_server",
    "unified_exec_zsh_fork",
    "compaction_image_budget", "content_item_kinds", "enable_request_compression", "mentions_v2",
)


def _comando_codex(binario: str, modelo: str) -> list[str]:
    cmd = [binario, *_CODEX_FLAGS]
    for f in _CODEX_DISABLE:
        cmd += ["--disable", f]
    cmd += [
        "-c", 'web_search="disabled"',
        # `model_instructions_file` figura en el binario 0.160 (strings); que
        # la clave se honre en ejecucion NO esta verificado (§8).
        "-c", "model_instructions_file=/work/sistema.md",
        "-c", "project_doc_max_bytes=0",
        "-m", modelo,
    ]
    return cmd


def comando_features_codex(binario: str) -> list[str]:
    """argv de `codex features list` con los MISMOS --disable que usa el perfil:
    `verificar_features` se alimenta con la salida de este comando, que muestra el
    estado EFECTIVO (las apagadas salen en false)."""
    cmd = [binario, "features", "list"]
    for f in _CODEX_DISABLE:
        cmd += ["--disable", f]
    return cmd


def verificar_features(salida_de_features_list: str, perfil: str = "codex") -> None:
    """Falla cerrado (FeaturesNoPermitidas) si la salida de `codex features list`
    (con los --disable del perfil, ver `comando_features_codex`) trae CUALQUIER
    feature activada que no este en la lista permitida del perfil, o una linea que
    no se pueda interpretar, o ninguna. Formato (0.160): `nombre  etapa  true|false`,
    la etapa puede llevar espacios ("under development"). Se usa en el paso 11, al
    fijar la version del CLI. Devuelve None si todo esta en regla."""
    p = PERFILES.get(perfil)
    if p is None or p.features_permitidas is None:
        raise PerfilNoSoportado(f"el perfil {perfil!r} no declara una lista de features permitidas")
    permitidas = set(p.features_permitidas)
    activadas_no_permitidas, ilegibles, vistas = [], [], 0
    for linea in salida_de_features_list.splitlines():
        if not linea.strip():
            continue
        partes = linea.split()
        if len(partes) < 3 or partes[-1] not in ("true", "false"):
            ilegibles.append(linea.strip()[:60])
            continue
        vistas += 1
        if partes[-1] == "true" and partes[0] not in permitidas:
            activadas_no_permitidas.append(partes[0])
    if ilegibles:
        raise FeaturesNoPermitidas(f"lineas de `features list` ilegibles: {ilegibles}")
    if not vistas:
        raise FeaturesNoPermitidas("la salida de `features list` esta vacia")
    if activadas_no_permitidas:
        raise FeaturesNoPermitidas(
            "features activadas fuera de la lista permitida: " + ", ".join(sorted(set(activadas_no_permitidas)))
        )


def _comando_kimi(binario: str, modelo: str) -> list[str]:
    # `-p -`: el prompt por stdin NO esta verificado (§8); ver
    # canal_prompt_verificado. Nunca --yolo ni --auto.
    return [
        binario, "--agent-file", "/work/faceta.md", "-m", modelo,
        "--output-format", "stream-json", "-p", "-",
    ]


def _sistema_plano(system_prompt: str) -> str:
    return system_prompt


def _sistema_kimi(system_prompt: str) -> str:
    # frontmatter `tools: []` + system prompt en el cuerpo (spec §1).
    return f"---\ntools: []\n---\n{system_prompt}\n"


def _eventos(stdout: bytes) -> list[dict]:
    out = []
    for linea in stdout.decode("utf-8", "replace").splitlines():
        linea = linea.strip()
        if not linea.startswith("{"):
            continue
        try:
            ev = json.loads(linea)
        except ValueError:
            continue
        if isinstance(ev, dict):
            out.append(ev)
    return out


# Codigos estructurados de error. NO VERIFICADO contra la CLI real (§8): se
# clasifica solo por campos de eventos estructurados y por exit code, nunca por
# palabras en stderr; un codigo desconocido cae en ErrorProtocolo (fail-closed).
_CODIGOS_CUOTA = frozenset({
    "usage_limit_reached", "rate_limit_exceeded", "insufficient_quota", "quota_exceeded",
})
_CODIGOS_SESION = frozenset({
    "token_expired", "invalid_token", "unauthorized", "refresh_token_reused",
    "refresh_token_expired", "invalid_grant", "not_logged_in",
})


def _codigo_de_error(ev: dict) -> Optional[str]:
    err = ev.get("error")
    for fuente in (ev, err if isinstance(err, dict) else {}):
        for k in ("code", "type"):
            v = fuente.get(k)
            if isinstance(v, str) and v in (_CODIGOS_CUOTA | _CODIGOS_SESION):
                return v
    return None


def _clasificar_errores(eventos: list[dict]) -> None:
    for ev in eventos:
        if ev.get("type") in ("error", "turn.failed") or "error" in ev:
            codigo = _codigo_de_error(ev)
            if codigo in _CODIGOS_CUOTA:
                raise CuotaAgotada(codigo)
            if codigo in _CODIGOS_SESION:
                raise SesionVencida(codigo)
            raise ErrorProtocolo("evento de error sin codigo reconocido")


def _parsear_codex(stdout: bytes, returncode: int) -> tuple[str, int, int]:
    eventos = _eventos(stdout)
    _clasificar_errores(eventos)
    if returncode != 0:
        raise ErrorProtocolo(f"codex termino con exit code {returncode} sin evento de error")
    texto, tin, tout = None, 0, 0
    for ev in eventos:
        if ev.get("type") == "item.completed":
            item = ev.get("item") or {}
            if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                texto = item["text"]  # solo el mensaje FINAL; el razonamiento se descarta
        elif ev.get("type") == "turn.completed":
            uso = ev.get("usage") or {}
            tin = int(uso.get("input_tokens") or 0)
            tout = int(uso.get("output_tokens") or 0)
    if texto is None:
        raise ErrorProtocolo("la salida de codex no trae un mensaje final")
    return texto, tin, tout


def _parsear_kimi(stdout: bytes, returncode: int) -> tuple[str, int, int]:
    # NO VERIFICADO (§8): esquema supuesto de stream-json. Mientras
    # canal_prompt_verificado sea False, run_cli no llega aca.
    eventos = _eventos(stdout)
    _clasificar_errores(eventos)
    if returncode != 0:
        raise ErrorProtocolo(f"kimi termino con exit code {returncode} sin evento de error")
    texto, tin, tout = None, 0, 0
    for ev in eventos:
        if ev.get("role") == "assistant":
            c = ev.get("content")
            if isinstance(c, str):
                texto = c
            elif isinstance(c, list):
                partes = [p.get("text") for p in c if isinstance(p, dict) and p.get("type") == "text"]
                if partes:
                    texto = "".join(p for p in partes if isinstance(p, str))
        uso = ev.get("usage")
        if isinstance(uso, dict):
            tin = int(uso.get("input_tokens") or tin)
            tout = int(uso.get("output_tokens") or tout)
    if texto is None:
        raise ErrorProtocolo("la salida de kimi no trae un mensaje final")
    return texto, tin, tout


@dataclass(frozen=True)
class Perfil:
    nombre: str
    via_run_cli: bool
    home_sandbox: str
    bin_nombre: str = ""
    cred_subdir: str = ""
    cred_destino: str = ""
    env_extra: tuple[tuple[str, str], ...] = ()
    purgar: tuple[str, ...] = ()
    # Lista PERMITIDA del directorio de credencial: tras cada llamada se borra TODO
    # lo que no figure aqui (codex deja ejecutables, hooks y config en CODEX_HOME,
    # que es ese mismo directorio montado RW). None = no se purga por exclusion.
    purgar_excepto: Optional[tuple[str, ...]] = None
    archivo_nombre: str = ""
    canal_prompt_verificado: bool = True
    features_permitidas: Optional[tuple[str, ...]] = None  # solo codex; ver verificar_features
    # `features list` del perfil (solo codex): como se arma el comando y el entorno EFIMERO en
    # el que corre (sin la credencial: listar features no pide login ni cuota)
    _armar_features: Optional[Callable[[str], list[str]]] = None
    env_features: tuple[tuple[str, str], ...] = ()
    ranuras: int = 2
    espera_lock_s: float = 10.0  # espera corta del chat (spec §1); no es el timeout
    _armar_comando: Optional[Callable[[str, str], list[str]]] = None
    _armar_sistema: Callable[[str], str] = _sistema_plano
    _parsear: Optional[Callable[[bytes, int], tuple[str, int, int]]] = None

    def comando(self, binario: str, modelo: str) -> list[str]:
        if self._armar_comando is None:
            raise PerfilNoSoportado(self.nombre)
        return self._armar_comando(binario, modelo)

    def env_fijo(self) -> dict[str, str]:
        return env_minimo(self.home_sandbox, dict(self.env_extra))

    def archivo_sistema(self, system_prompt: str) -> str:
        return self._armar_sistema(system_prompt)

    def parsear(self, stdout: bytes, returncode: int) -> tuple[str, int, int]:
        if self._parsear is None:
            raise PerfilNoSoportado(self.nombre)
        return self._parsear(stdout, returncode)


_HOME_CLI = "/home/cli-sandbox"

PERFILES: dict[str, Perfil] = {
    # Hyde (con herramientas) sigue por hyde_sandbox; `claude` esta aca para que
    # el nucleo y los tests tengan UNA tabla de perfiles, pero `run_cli` lo
    # rechaza (adenda §E: "claude sigue cerrado").
    "claude": Perfil(nombre="claude", via_run_cli=False, home_sandbox="/home/hyde-sandbox"),
    "codex": Perfil(
        nombre="codex", via_run_cli=True, home_sandbox=_HOME_CLI,
        bin_nombre="codex", cred_subdir="codex", cred_destino=f"{_HOME_CLI}/.codex",
        env_extra=(("CODEX_HOME", f"{_HOME_CLI}/.codex"), ("CODEX_SQLITE_HOME", "/tmp/codex-sqlite")),
        archivo_nombre="sistema.md",
        purgar_excepto=("auth.json",),
        # Igual que Kimi: `model_instructions_file` figura en el binario pero que
        # se honre en ejecucion NO esta verificado (§8, prueba manual del §5). Sin
        # eso, el system prompt (persona y memoria) podria no llegar al modelo: no
        # se lanza hasta verificarlo.
        canal_prompt_verificado=False,
        features_permitidas=_CODEX_FEATURES_PERMITIDAS,
        _armar_features=comando_features_codex,
        env_features=(("CODEX_HOME", "/tmp/codex-features-home"), ("CODEX_SQLITE_HOME", "/tmp/codex-features-sqlite")),
        _armar_comando=_comando_codex, _parsear=_parsear_codex,
    ),
    "kimi": Perfil(
        nombre="kimi", via_run_cli=True, home_sandbox=_HOME_CLI,
        bin_nombre="kimi", cred_subdir="kimi", cred_destino=f"{_HOME_CLI}/.kimi-code",
        env_extra=(
            ("KIMI_CODE_HOME", f"{_HOME_CLI}/.kimi-code"),
            ("KIMI_CODE_NO_AUTO_UPDATE", "1"), ("KIMI_CLI_NO_AUTO_UPDATE", "1"),
            ("KIMI_DISABLE_TELEMETRY", "1"), ("KIMI_DISABLE_CRON", "1"),
            # NUNCA KIMI_CODE_INFINITE_RETRY (spec §1)
        ),
        purgar=("sessions", "user-history", "logs", "telemetry"),
        archivo_nombre="faceta.md",
        canal_prompt_verificado=False,
        _armar_comando=_comando_kimi, _armar_sistema=_sistema_kimi, _parsear=_parsear_kimi,
    ),
}


# --------------------------------------------------------------------------
# Binarios fijados
# --------------------------------------------------------------------------

_RE_VERSION = re.compile(r"^[0-9][0-9A-Za-z._+-]{0,31}$")
_RE_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RE_MODELO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:-]{0,63}$")

#: ruta del binario -> (firma del ARBOL, sha256 del manifiesto observado). La firma
#: del arbol es el SHA256 de (ruta relativa, tipo, dev, inode, mtime_ns, ctime_ns,
#: size, modo, uid) de TODAS las entradas del directorio del binario: bwrap monta y
#: ejecuta el directorio entero, no solo el archivo (MAJOR-15). Se invalida sola: si
#: cambia cualquier stat de cualquier entrada, se re-hashea. `ctime` lo fija el
#: kernel en cada escritura y NO se puede restaurar con utime(): reescribir un
#: archivo y devolverle el mtime no engana a la cache (MAJOR-4). Lo que la cache no
#: garantiza (un cambio dentro del mismo tick del reloj del kernel) lo cubre que
#: todo el arbol sea de root y no escribible: quien podria cambiarlo no es un usuario
#: comun. El recorrido (lstat de cada entrada, dueno y permisos) corre en CADA
#: llamada; lo que se cachea es el hash del contenido, que es lo caro.
_CACHE_SHA: dict[str, tuple[str, str]] = {}

_MANIFIESTO_VERSION = "jax-cli-manifiesto-v1"


def _firma_archivo(st) -> tuple[int, ...]:
    return (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_ctime_ns, st.st_size)


def _sha256_archivo(ruta: str, *, esperado=None) -> str:
    """SHA256 del archivo `ruta`, abierto SIN seguir symlinks. Si se pasa `esperado`
    (el stat que vio el recorrido), el archivo abierto tiene que ser ese mismo
    (dev, inode): si lo cambiaron entre el recorrido y el hash, BinarioAlterado."""
    import hashlib
    h = hashlib.sha256()
    try:
        fd = os.open(ruta, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise BinarioAlterado(f"no se pudo abrir {ruta!r} para hashearlo ({exc.strerror or type(exc).__name__})") from None
    with os.fdopen(fd, "rb") as f:
        if esperado is not None:
            st = os.fstat(f.fileno())
            if (st.st_dev, st.st_ino) != (esperado.st_dev, esperado.st_ino):
                raise BinarioAlterado(f"{ruta!r} cambio entre el recorrido y el hash")
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _inspeccionar_arbol(directorio: str, uid_esperado: int, perfil: str) -> list:
    """Recorre TODO el arbol de `directorio` con `os.fwalk(follow_symlinks=False)` y
    exige, en cada entrada: dueno == `uid_esperado`, sin escritura de grupo ni de
    otros, y tipo permitido (archivo regular, directorio o symlink). Un symlink solo
    se acepta si, resuelto de verdad (`realpath`, cadenas incluidas), queda DENTRO del
    arbol. Cualquier error al recorrer (un subdirectorio ilegible) es BinarioAlterado:
    `fwalk` ignora esos errores por defecto, y un arbol que no se pudo leer entero no
    esta verificado. Devuelve [(ruta relativa, tipo, stat, destino del symlink o None)]
    en el orden del recorrido; la raiz es la entrada ".". No hashea nada."""
    def malo(que: str):
        raise BinarioAlterado(f"{perfil}: {que}")

    def al_fallar(exc: OSError):
        malo(f"no se pudo recorrer el arbol del binario ({exc.strerror or type(exc).__name__})")

    try:
        top = os.lstat(directorio)
    except OSError:
        malo("el directorio del binario no existe")
    if not stat.S_ISDIR(top.st_mode):
        malo("el directorio del binario no es un directorio real (¿symlink?)")
    raiz_real = os.path.realpath(directorio)
    entradas: list = []

    def revisar(rel: str, st) -> str:
        if st.st_uid != uid_esperado:
            malo(f"{rel!r} no es del dueno esperado")
        if stat.S_ISLNK(st.st_mode):
            return "l"  # el modo de un symlink no significa nada; cuenta su destino
        if st.st_mode & 0o022:
            malo(f"{rel!r} admite escritura de grupo u otros")
        if stat.S_ISREG(st.st_mode):
            return "f"
        if stat.S_ISDIR(st.st_mode):
            return "d"
        malo(f"{rel!r} es de un tipo no permitido (solo archivo, directorio o symlink)")

    for dirpath, dirnames, filenames, dfd in os.fwalk(directorio, follow_symlinks=False, onerror=al_fallar):
        if dirpath == directorio or not entradas:
            sd = os.fstat(dfd)
            entradas.append((".", revisar(".", sd), sd, None))
        reldir = os.path.relpath(dirpath, directorio)
        for nombre in sorted(dirnames + filenames):
            rel = nombre if reldir == "." else os.path.join(reldir, nombre)
            st = os.stat(nombre, dir_fd=dfd, follow_symlinks=False)
            tipo = revisar(rel, st)
            destino = None
            if tipo == "l":
                destino = os.readlink(nombre, dir_fd=dfd)
                real = os.path.realpath(os.path.join(dirpath, nombre))
                if real != raiz_real and not real.startswith(raiz_real + os.sep):
                    malo(f"el symlink {rel!r} sale del arbol del binario")
            entradas.append((rel, tipo, st, destino))
    if not entradas:
        malo("el arbol del binario no se pudo recorrer")
    return entradas


def _firma_arbol(entradas) -> str:
    import hashlib
    h = hashlib.sha256()
    for rel, tipo, st, destino in entradas:
        h.update(repr((rel, tipo, _firma_archivo(st), st.st_mode, st.st_uid, destino)).encode("utf-8", "surrogateescape"))
    return h.hexdigest()


def _manifiesto(directorio: str, entradas) -> str:
    """SHA256 del MANIFIESTO del arbol: el texto canonico (JSON compacto, ASCII) de
    la lista ORDENADA por ruta de `[ruta relativa, tipo, modo en octal, sha256 de los
    archivos regulares | destino de los symlinks | ""]`, precedido por la version del
    formato. No incluye dueno, mtime, ctime ni inodes: no depende de quien corra la
    verificacion ni de cuando se copio el arbol, solo del contenido, los nombres, los
    modos y los symlinks."""
    import hashlib
    filas = []
    for rel, tipo, st, destino in entradas:
        if tipo == "f":
            ruta = os.path.join(directorio, rel)
            extra = _sha256_archivo(ruta, esperado=st)
        elif tipo == "l":
            extra = destino
        else:
            extra = ""
        filas.append([rel, tipo, "%04o" % stat.S_IMODE(st.st_mode), extra])
    filas.sort(key=lambda f: f[0])
    texto = _MANIFIESTO_VERSION + "\n" + json.dumps(filas, separators=(",", ":"))
    return hashlib.sha256(texto.encode("ascii")).hexdigest()


def sha_manifiesto(directorio: str, *, uid_esperado: int = 0) -> str:
    """El valor que va en `JAX_CLI_<PERFIL>_SHA256` (paso 11 del despliegue): el SHA256
    del manifiesto del directorio del binario, p. ej.

        python3 -c 'import cli_sandbox as c; print(c.sha_manifiesto("/opt/jax-cli/codex/0.160.0"))'

    corrido DESPUES de instalar el arbol como root (todo de `uid_esperado`, 0 por
    defecto, y sin escritura de grupo ni de otros: si no, BinarioAlterado). Es la misma
    funcion que usa `run_cli` para verificar, asi que lo que se fija es lo que se
    compara. Cualquier archivo, modo o symlink que cambie en el arbol cambia el SHA."""
    return _manifiesto(directorio, _inspeccionar_arbol(directorio, uid_esperado, "manifiesto"))


def _verificar_dueno_y_permisos(ruta: str, raiz: str, uid_esperado: int, perfil: str) -> os.stat_result:
    """El binario es un archivo regular (no symlink), del `uid_esperado`, sin
    escritura de grupo ni de otros; cada directorio desde el suyo hasta `raiz`
    inclusive es un directorio real con el mismo dueno y sin esa escritura; y cada
    ANCESTRO de `raiz` hasta `/` es un directorio real de root (o del propio
    `uid_esperado`, que en produccion ES root) sin escritura de grupo ni de otros.
    Unica excepcion: `/tmp`, de escritura publica pero con el bit sticky, si esta en
    la ruta. Si no, BinarioAlterado: un binario (o la ruta que lleva a el) que alguien
    mas puede reemplazar no esta fijado por su SHA256, solo por suerte. El recorrido
    de TODO el directorio del binario lo hace `_inspeccionar_arbol`."""
    def malo(que: str):
        raise BinarioAlterado(f"{perfil}: {que}")

    if not os.path.isabs(raiz):
        malo("JAX_CLI_ROOT debe ser una ruta absoluta")
    try:
        st = os.lstat(ruta)
    except OSError:
        malo("binario fijado ausente")
    if not stat.S_ISREG(st.st_mode):
        malo("el binario fijado no es un archivo regular (¿symlink?)")
    if st.st_uid != uid_esperado or st.st_mode & 0o022:
        malo("el binario fijado no es del dueno esperado o admite escritura de grupo/otros")
    raiz = os.path.normpath(raiz)
    d = os.path.dirname(os.path.normpath(ruta))
    while True:
        try:
            sd = os.lstat(d)
        except OSError:
            malo(f"directorio {d!r} ilegible")
        if not stat.S_ISDIR(sd.st_mode) or sd.st_uid != uid_esperado or sd.st_mode & 0o022:
            malo(f"el directorio {d!r} no es del dueno esperado o admite escritura de grupo/otros")
        if d == raiz:
            break
        padre = os.path.dirname(d)
        if padre == d:  # llegamos a "/" sin pasar por la raiz: la ruta no esta bajo ella
            malo("el binario no esta bajo JAX_CLI_ROOT")
        d = padre
    # ancestros de la raiz hasta "/"
    while d != "/":
        d = os.path.dirname(d)
        try:
            sd = os.lstat(d)
        except OSError:
            malo(f"ancestro {d!r} ilegible")
        if not stat.S_ISDIR(sd.st_mode):
            malo(f"el ancestro {d!r} no es un directorio real (¿symlink?)")
        if sd.st_uid not in (0, uid_esperado):
            malo(f"el ancestro {d!r} no es de root")
        if sd.st_mode & 0o022 and not (d == "/tmp" and sd.st_uid == 0 and sd.st_mode & stat.S_ISVTX):
            malo(f"el ancestro {d!r} admite escritura de grupo u otros (el bit sticky solo se tolera en /tmp)")
    return st


def _resolver_binario(perfil: Perfil, *, uid_esperado: int = 0) -> tuple[str, str, str]:
    """(ruta del binario, directorio, version) fijados por la configuracion, ya
    verificados contra el SHA256 del MANIFIESTO de su directorio y contra su dueno y
    permisos (el arbol completo y los ancestros de JAX_CLI_ROOT). Lanza
    BinarioAlterado ante cualquier duda. `uid_esperado` (0 = root en produccion) es un
    parametro y no un flag global para que los tests sin root lo inyecten sin apagar
    el control; run_cli nunca lo recibe de un llamador."""
    pref = f"JAX_CLI_{perfil.nombre.upper()}"
    version = os.environ.get(f"{pref}_VERSION", "")
    sha = os.environ.get(f"{pref}_SHA256", "").strip().lower()
    if not _RE_VERSION.match(version):
        raise BinarioAlterado(f"{perfil.nombre}: version fijada ausente o invalida")
    if not _RE_SHA256.match(sha):
        raise BinarioAlterado(f"{perfil.nombre}: SHA256 fijado ausente o invalido")
    raiz = os.environ.get("JAX_CLI_ROOT", "/opt/jax-cli")
    directorio = os.path.join(raiz, perfil.nombre, version)
    ruta = os.path.join(directorio, perfil.bin_nombre)
    _verificar_dueno_y_permisos(ruta, raiz, uid_esperado, perfil.nombre)
    entradas = _inspeccionar_arbol(directorio, uid_esperado, perfil.nombre)
    firma = _firma_arbol(entradas)
    cacheado = _CACHE_SHA.get(ruta)
    if cacheado and cacheado[0] == firma:
        observado = cacheado[1]
    else:
        observado = _manifiesto(directorio, entradas)
        _CACHE_SHA[ruta] = (firma, observado)
    if observado != sha:
        raise BinarioAlterado(f"{perfil.nombre}: el SHA256 del manifiesto del directorio no coincide con el fijado")
    return ruta, directorio, version


#: ruta del binario -> firma del arbol para la que `verificar_features` ya dio verde. MISMA
#: clave y MISMA firma que `_CACHE_SHA`: si cambia cualquier stat del arbol, la firma cambia,
#: se re-hashea el manifiesto Y se vuelve a verificar.
_CACHE_FEATURES: dict[str, str] = {}

#: ruta del binario -> (firma del arbol, mensaje) del FALLO de `verificar_features` (feature
#: no permitida, salida vacia o ilegible, exit code, timeout de `features list`). MISMA clave
#: y firma que las dos de arriba, y la misma invalidacion: si cambia el arbol, se vuelve a
#: verificar (un operador que reinstala un binario sano no queda atado al fallo viejo). El
#: fallo se cachea igual que el exito (MINOR-23, reauditoria 2026-10-02): sin esto, cada
#: llamada con un binario malo relanzaba `features list` -- un proceso de bwrap por chat -- y
#: un binario roto era un amplificador. Se cachea el motivo, para que cada llamada falle con
#: el mismo mensaje. NO se cachean las esperas de ranura (LockTimeout): no son un fallo del
#: binario. Un fallo transitorio (p. ej. el timeout de `features list` con la maquina
#: saturada) tambien queda hasta que cambie el arbol o se reinicie el proceso: es la
#: contrapartida aceptada de fallar cerrado sin reintentar.
_CACHE_FEATURES_FALLO: dict[str, tuple[str, str]] = {}

#: Espera maxima de `codex features list` (arranca el binario una vez por firma).
FEATURES_TIMEOUT_S = 30.0


async def _features_del_binario(
    p: Perfil, ruta_bin: str, dir_bin: str, espera_lock_s: Optional[float] = None,
) -> str:
    """Corre `<binario> features list --disable ...` DENTRO del sandbox, con el mismo
    `ejecutar` del nucleo (el unico create_subprocess_exec), sin credenciales (listar
    features no pide login ni cuota) y con un CODEX_HOME efimero: el /tmp privado del
    sandbox, que desaparece con el proceso. Corre BAJO UNA RANURA del perfil, como la llamada
    real (MINOR-23): arrancar el binario cuenta para la concurrencia del perfil. Devuelve la
    salida; cualquier cosa que no sea un exit code 0 con salida es FeaturesNoPermitidas (falla
    cerrado); sin ranura libre es LockTimeout."""
    base = Path(os.environ.get("JAX_CLI_RUN_DIR", "/run/jax-cli"))
    rundir = base / f"features-{uuid.uuid4().hex}"
    await asyncio.to_thread(_preparar_rundir, base, rundir, p.archivo_nombre, "")
    n = _ranuras_de(p)
    espera = p.espera_lock_s if espera_lock_s is None else espera_lock_s
    try:
        argv = argv_confinado_cli(
            _BWRAP_BIN, work_host=str(rundir), home_sandbox=p.home_sandbox,
            binds_rw=[], binds_ro=[(dir_bin, dir_bin)], cmd=p._armar_features(ruta_bin),
            dirs=[v for _k, v in p.env_features],
        )
        env = env_minimo(p.home_sandbox, dict(p.env_features))
        proc, stdout, _stderr = await ejecutar(
            argv, env, b"", FEATURES_TIMEOUT_S,
            adquirir=lambda _t: _ranura_adquirir(p.nombre, n, espera),
            # sin credencial montada no hay nada que purgar: solo se suelta la ranura
            liberar=lambda h: flock_liberar(h[0]),
            cwd="/",
        )
    except LockTimeout:
        raise  # LockTimeout tambien es TimeoutError: no convertirlo en "features list no respondio"
    except asyncio.TimeoutError:
        raise FeaturesNoPermitidas(f"{p.nombre}: `features list` no respondio en {FEATURES_TIMEOUT_S}s") from None
    finally:
        await asyncio.to_thread(_borrar_rundir, rundir)
    if proc.returncode != 0:
        raise FeaturesNoPermitidas(f"{p.nombre}: `features list` termino con exit code {proc.returncode}")
    return stdout.decode("utf-8", "replace")


async def _asegurar_features(
    p: Perfil, ruta_bin: str, dir_bin: str, espera_lock_s: Optional[float] = None,
) -> None:
    """Verifica las features del binario (lista PERMITIDA, `verificar_features`) una vez por
    firma del arbol, el exito Y el fallo (MINOR-23). Los perfiles sin lista permitida (kimi)
    no pasan por aqui."""
    if p.features_permitidas is None or p._armar_features is None:
        return
    firma = _CACHE_SHA.get(ruta_bin, ("",))[0]
    if firma:
        if _CACHE_FEATURES.get(ruta_bin) == firma:
            return
        fallo = _CACHE_FEATURES_FALLO.get(ruta_bin)
        if fallo and fallo[0] == firma:
            raise FeaturesNoPermitidas(fallo[1])
    try:
        verificar_features(await _features_del_binario(p, ruta_bin, dir_bin, espera_lock_s), p.nombre)
    except FeaturesNoPermitidas as exc:
        if firma:
            _CACHE_FEATURES_FALLO[ruta_bin] = (firma, str(exc))
        raise
    if firma:
        _CACHE_FEATURES[ruta_bin] = firma
        _CACHE_FEATURES_FALLO.pop(ruta_bin, None)


# --------------------------------------------------------------------------
# Historial (spec §3)
# --------------------------------------------------------------------------

MAX_PROMPT_CHARS = 32_000
_ETIQUETAS = {"user": "Usuario", "assistant": "Asistente"}


def _normalizar_historial(historial) -> list[tuple[str, str]]:
    out = []
    for t in historial or []:
        if isinstance(t, dict):
            rol, texto = t.get("role"), t.get("content")
        else:
            rol, texto = t
        out.append((_ETIQUETAS.get(rol, "Usuario"), str(texto)))
    return out


def armar_conversacion(historial, mensaje: str, max_chars: int, nonce: Optional[str] = None) -> str:
    """Serializa historial + mensaje. Delimitadores con un NONCE aleatorio por
    llamada (un mensaje viejo no puede imitar el cierre del contexto) y
    etiquetas neutras. Recorta DESDE EL TURNO MAS VIEJO; el mensaje actual
    nunca se corta: si solo ya pasa el tope, MensajeDemasiadoLargo."""
    turnos = _normalizar_historial(historial)
    if nonce is None:
        while True:
            nonce = secrets.token_hex(8)
            if all(nonce not in t for _, t in turnos) and nonce not in mensaje:
                break

    if len(mensaje) > max_chars:
        raise MensajeDemasiadoLargo(
            f"el mensaje actual ({len(mensaje)} caracteres) no cabe en el tope de {max_chars}"
        )
    # Longitud EXACTA de la salida con k turnos: el prefijo y el sufijo fijos mas
    # la suma de las lineas y sus k-1 saltos. Se acumula desde el turno mas nuevo
    # y se corta en el primero que ya no cabe: O(n), sin re-serializar en cada
    # vuelta (MINOR-10) y con el mismo resultado que quitar turnos desde el mas
    # viejo hasta que quepa.
    fijo = (
        len(f"[Inicio del contexto {nonce}]\n")
        + len(f"\n[Fin del contexto {nonce}]\n\nMensaje actual:\n") + len(mensaje)
    )
    lineas: list[str] = []
    total = fijo - 1  # el primer turno no lleva salto previo
    for rol, texto in reversed(turnos):
        linea = f"{rol}: {texto}"
        total += len(linea) + 1
        if total > max_chars:
            break
        lineas.append(linea)
    if not lineas:
        return mensaje
    cuerpo = "\n".join(reversed(lineas))
    return (
        f"[Inicio del contexto {nonce}]\n{cuerpo}\n[Fin del contexto {nonce}]\n\n"
        f"Mensaje actual:\n{mensaje}"
    )


# --------------------------------------------------------------------------
# Locks por perfil (N ranuras). NO comparten directorio ni nombre con el de
# Hyde: un Thot en curso no espera a Hyde.
# --------------------------------------------------------------------------

def _lock_dir() -> Path:
    """Directorio de los locks por perfil: `JAX_CLI_LOCK_DIR`, SIN default. Ausente
    o vacio es SandboxUnavailable (falla cerrado): un nombre fijo en /tmp lo puede
    ocupar otro usuario antes que nosotros (auditoria 2026-10-02, MAJOR-14). El paso
    de host lo apunta a /run/jax-locks/cli, sembrado con tmpfiles.d; despues
    `_preparar_dir_locks` verifica que sea del euid y sin escritura ajena."""
    valor = os.environ.get("JAX_CLI_LOCK_DIR", "").strip()
    if not valor:
        raise SandboxUnavailable(
            "JAX_CLI_LOCK_DIR no esta definida: sin un directorio de locks seguro no se lanza "
            "(falla cerrado; el paso de host la fija en /run/jax-locks/cli con tmpfiles.d)"
        )
    return Path(valor)


_RANURAS_MAX = 16


def _ranuras_de(p: Perfil) -> int:
    """Numero de ranuras (llamadas concurrentes) del perfil. UNA sola fuente,
    leida aqui: `JAX_CLI_<PERFIL>_RANURAS` si es un entero 1..16 y, si no, el
    default del perfil. El llamador de `run_cli` no lo decide (auditoria
    2026-10-02, MINOR-8): un valor invalido nunca afloja el tope, vuelve al
    default."""
    crudo = os.environ.get(f"JAX_CLI_{p.nombre.upper()}_RANURAS", "").strip()
    if crudo.isdigit() and 1 <= int(crudo) <= _RANURAS_MAX:
        return int(crudo)
    return p.ranuras


#: Tope del `timeout` de `run_cli` por punto de entrada: (variable de entorno, default
#: en segundos). Un entry_point que no figura aqui NO tiene tope declarado y `run_cli`
#: lo rechaza (falla cerrado). `canary` y `repl` son entry_points validos para
#: `exigir_titular` pero no tienen tope: hay que declararselo antes de usarlos.
TIMEOUT_MAX_POR_ENTRY: dict[str, tuple[str, float]] = {
    "chat": ("JAX_CLI_TIMEOUT_MAX_CHAT_S", 180.0),
    "jacobs": ("JAX_CLI_TIMEOUT_MAX_JACOBS_S", 600.0),
}


def timeout_maximo_para(entry_point) -> float:
    """Tope del `timeout` de `run_cli` para `entry_point`: `JAX_CLI_TIMEOUT_MAX_CHAT_S`
    (chat, default 180 s) o `JAX_CLI_TIMEOUT_MAX_JACOBS_S` (jacobs, default 600 s) si es
    un numero finito positivo y, si no, el default: un valor roto vuelve al default, no
    lo afloja. Un entry_point sin tope declarado es ValueError."""
    if not isinstance(entry_point, str) or entry_point not in TIMEOUT_MAX_POR_ENTRY:
        raise ValueError(f"entry_point {entry_point!r} sin tope de timeout declarado")
    var, default = TIMEOUT_MAX_POR_ENTRY[entry_point]
    try:
        v = float(os.environ.get(var, ""))
    except ValueError:
        return default
    return v if math.isfinite(v) and v > 0 else default


def tope_prompt_chars() -> int:
    """Tope de configuracion del prompt: `JAX_CLI_MAX_PROMPT_CHARS` si es un entero
    positivo y, si no, `MAX_PROMPT_CHARS`. Lo que pide el llamador de `run_cli` solo
    puede BAJARLO (`min`), nunca subirlo."""
    crudo = os.environ.get("JAX_CLI_MAX_PROMPT_CHARS", "").strip()
    return int(crudo) if crudo.isascii() and crudo.isdigit() and int(crudo) > 0 else MAX_PROMPT_CHARS


def _numero_finito(v) -> bool:
    """int o float (no bool) finito. Un entero demasiado grande para un float no es
    finito: se rechaza en vez de dejar escapar un OverflowError."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        return math.isfinite(v)
    except OverflowError:
        return False


def _ranura_adquirir(perfil: str, ranuras: int, espera: float):
    """BLOQUEANTE (to_thread). Primera ranura libre de `ranuras`; si no hay
    ninguna en `espera` segundos, LockTimeout. Un directorio o un lock inseguro
    es SandboxUnavailable (no se salta a la ranura siguiente: algo esta mal)."""
    dfd = _preparar_dir_locks(_lock_dir())
    try:
        deadline = time.monotonic() + espera
        while True:
            for i in range(ranuras):
                fh = _abrir_lock(dfd, f"{perfil}.{i}.lock")
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return (fh, i)
                except BlockingIOError:
                    fh.close()
            if time.monotonic() >= deadline:
                raise LockTimeout(f"sin ranura libre para el perfil {perfil} en {espera}s")
            time.sleep(_LOCK_POLL_S)
    finally:
        os.close(dfd)


def _cred_host(p: Perfil) -> str:
    """Directorio de credencial dedicado del perfil en el host (UNA sola fuente: lo usan
    `run_cli` y `purgar_al_arranque`)."""
    return os.path.join(os.environ.get("JAX_CLI_CRED_ROOT", "/srv/jax-data/cli-suscripcion"), p.cred_subdir)


def _quitar(ruta: str, perfil: Perfil) -> None:
    """Quita `ruta` (directorio, archivo o symlink) sin seguir symlinks. Lo que no se
    pueda quitar se REGISTRA con la ruta y el error y queda en disco para la proxima
    purga: `rmtree(onexc=...)` en vez de `ignore_errors=True`, que lo tragaba."""
    def al_fallar(_func, ruta_fallida, exc):
        logger.warning(
            "purga: no se pudo quitar %s de la credencial de %s (%s)",
            ruta_fallida, perfil.nombre, getattr(exc, "strerror", None) or type(exc).__name__,
        )

    try:
        if os.path.islink(ruta) or not os.path.isdir(ruta):
            os.unlink(ruta)  # un enlace se desvincula; nunca se toca su destino
        else:
            shutil.rmtree(ruta, onexc=al_fallar)
    except OSError as exc:
        al_fallar(os.unlink, ruta, exc)


def _purgar_credenciales(perfil: Perfil, cred_host: str) -> None:
    """BLOQUEANTE. Con TODAS las ranuras del perfil tomadas por quien llama: borra
    los directorios de `purgar` y todo lo que no este en `purgar_excepto`. No sigue
    symlinks: un enlace se desvincula, nunca se borra su destino. Lo que no se pueda
    quitar se registra (warning con ruta y error), no se ignora."""
    for nombre in perfil.purgar:
        ruta = os.path.join(cred_host, nombre)
        if os.path.lexists(ruta):
            _quitar(ruta, perfil)
    if perfil.purgar_excepto is None:
        return
    try:
        entradas = list(os.scandir(cred_host))
    except OSError as exc:
        logger.warning("purga: no se pudo listar la credencial de %s (%s)", perfil.nombre, exc.strerror)
        return
    for e in entradas:
        if e.name not in perfil.purgar_excepto:
            _quitar(e.path, perfil)


def _purgar_si_libre(perfil: Perfil, ranuras: int, cred_host: str, propia: Optional[int]) -> bool:
    """BLOQUEANTE. Toma, sin esperar, todas las ranuras del perfil salvo la `propia` (la
    que ya tiene quien llama; None = ninguna) y, SOLO si las consiguio todas, purga. Si
    otra esta en uso la purga queda para la proxima (se borraria el estado de una llamada
    en curso). Los locks se abren con la misma disciplina que al adquirir (sin symlinks,
    sin truncar, directorio verificado). Devuelve si purgo."""
    if not (perfil.purgar or perfil.purgar_excepto is not None):
        return False
    otras = []
    dfd = _preparar_dir_locks(_lock_dir())
    try:
        libres = True
        for j in range(ranuras):
            if j == propia:
                continue
            f2 = _abrir_lock(dfd, f"{perfil.nombre}.{j}.lock")
            try:
                fcntl.flock(f2.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                otras.append(f2)
            except BlockingIOError:
                f2.close()
                libres = False
                break
        if libres:
            _purgar_credenciales(perfil, cred_host)
        return libres
    finally:
        os.close(dfd)
        for f2 in otras:
            flock_liberar(f2)


def purgar_al_arranque(perfil: str) -> bool:
    """BLOQUEANTE -- llamar via asyncio.to_thread. Purga el estado que el CLI del `perfil`
    dejo en su directorio de credencial si el proceso anterior murio entre el CLI y la
    purga de `_ranura_liberar` (kill -9, apagon, OOM): sin esto ese estado -- ejecutables,
    hooks, config -- quedaria hasta la proxima llamada exitosa. Lo invoca, al arrancar el
    proceso y antes de servir la primera llamada, el llamador de los pasos 6 (Thot, chat) y
    9 (Jacobs); este modulo no lo dispara solo. Con la misma regla que la purga normal: solo
    si NINGUNA ranura esta en uso (otro proceso puede estar sirviendo una llamada); devuelve
    True si purgo y False si no (ranura ocupada, perfil sin nada que purgar o directorio de
    credencial ausente, que se registra). Un directorio de locks inseguro es
    SandboxUnavailable (falla cerrado). Perfil desconocido o que no se sirve por run_cli:
    PerfilNoSoportado."""
    p = PERFILES.get(perfil) if isinstance(perfil, str) else None
    if p is None or not p.via_run_cli:
        raise PerfilNoSoportado(f"perfil {perfil!r} no se sirve por run_cli")
    cred_host = _cred_host(p)
    if not os.path.isdir(cred_host):
        logger.warning("purga al arrancar: no existe el directorio de credencial de %s", p.nombre)
        return False
    return _purgar_si_libre(p, _ranuras_de(p), cred_host, None)


def _ranura_liberar(handle, perfil: Perfil, ranuras: int, cred_host: Optional[str]) -> None:
    """BLOQUEANTE (to_thread). Con la ranura todavia tomada, purga el estado
    que el CLI deja en su home dedicado (Kimi no tiene --ephemeral) -- pero solo
    si NINGUNA otra ranura esta en uso (ver `_purgar_si_libre`); si hay otra
    corriendo, la purga queda para la proxima."""
    fh, idx = handle
    try:
        if cred_host:
            _purgar_si_libre(perfil, ranuras, cred_host, idx)
    finally:
        flock_liberar(fh)


# --------------------------------------------------------------------------
# run_cli
# --------------------------------------------------------------------------

_RE_BLANCOS = re.compile(r"\s+")


def _campo_log(valor, tope: int = 200) -> str:
    """Un valor para la linea de log de `run_cli`: texto de UNA sola linea y de largo acotado.
    Todo campo de texto de esa linea pasa por aqui (no solo `motivo`): un valor con saltos de
    linea puede fabricar una segunda linea de log con una `clase=ok` falsa (MINOR-22). Los
    espacios y saltos (\\n, \\r, tabs, separadores unicode) se colapsan en uno solo."""
    return _RE_BLANCOS.sub(" ", str(valor))[:tope]


@dataclass(frozen=True)
class ResultadoCLI:
    texto: str
    tokens_in: int
    tokens_out: int
    version_cli: str
    clase_error: Optional[str] = None  # None en exito; los errores se LANZAN tipados


def _escribir_privado(ruta: Path, contenido: str) -> None:
    fd = os.open(ruta, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(contenido)


def _preparar_rundir(base: Path, rundir: Path, archivo: str, contenido: str) -> None:
    """BLOQUEANTE (to_thread): crea `base` (0700) y el directorio por llamada y
    escribe el system prompt. Si algo falla, no deja el directorio a medias."""
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.mkdir(rundir, 0o700)
    try:
        _escribir_privado(rundir / archivo, contenido)
    except BaseException:
        shutil.rmtree(rundir, ignore_errors=True)
        raise


def _borrar_rundir(rundir: Path) -> None:
    """BLOQUEANTE (to_thread)."""
    shutil.rmtree(rundir, ignore_errors=True)


async def run_cli(
    perfil: str, system_prompt: str, historial, mensaje: str, modelo: str, timeout: float, *,
    titular: Titular, correlation_id: str, entry_point: str,
    espera_lock_s: Optional[float] = None, max_chars: Optional[int] = None,
) -> ResultadoCLI:
    """Lanza el CLI del perfil SIN herramientas dentro del sandbox y devuelve el
    mensaje final. Exige un `Titular` autorizado por `exigir_titular` para este
    mismo `entry_point`: sin el, no se crea nada ni se lanza nada. Los errores
    son excepciones tipadas (ErrorCLI y subclases), con `.clase` estable."""
    t0 = time.monotonic()
    clase = None
    motivo = ""
    version = ""
    try:
        if (
            not isinstance(titular, Titular) or titular._sello is not _SELLO
            or titular.entry_point != entry_point
        ):
            raise TitularNoAutorizado("suscripcion_solo_titular", "titular ausente, no autorizado o de otro entry_point")
        edad = time.monotonic() - titular.emitido_mono
        if not 0 <= edad <= TITULAR_TTL_S:
            raise TitularNoAutorizado("suscripcion_solo_titular", "titular caducado")
        p = PERFILES.get(perfil)
        if p is None or not p.via_run_cli:
            raise PerfilNoSoportado(f"perfil {perfil!r} no se sirve por run_cli")
        if not isinstance(modelo, str) or not _RE_MODELO.match(modelo):
            raise ValueError("modelo con formato invalido")
        tope_timeout = timeout_maximo_para(entry_point)  # ValueError si el entry_point no tiene tope
        if not _numero_finito(timeout) or not 0 < timeout <= tope_timeout:
            raise ValueError(
                f"timeout invalido: se exige 0 < timeout <= {tope_timeout} para el entry_point {entry_point!r}"
            )
        if espera_lock_s is not None and (
            not _numero_finito(espera_lock_s) or not 0 <= espera_lock_s <= p.espera_lock_s
        ):
            raise ValueError(f"espera_lock_s invalida: se exige un numero finito en 0..{p.espera_lock_s}")
        tope = tope_prompt_chars()
        if max_chars is not None:
            if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
                raise ValueError("max_chars invalido: se exige un entero positivo")
            tope = min(max_chars, tope)  # el llamador solo puede BAJAR el tope de configuracion
        if not p.canal_prompt_verificado:
            raise ErrorProtocolo(f"{p.nombre}: el canal del prompt no esta verificado (spec §8); no se lanza")
        conversacion = armar_conversacion(historial, mensaje, tope)
        ruta_bin, dir_bin, version = await asyncio.to_thread(_resolver_binario, p)
        verificar_bwrap(_BWRAP_BIN, p.nombre)
        await _asegurar_features(p, ruta_bin, dir_bin, espera_lock_s)
        cred_host = _cred_host(p)
        if not os.path.isdir(cred_host):
            raise SandboxUnavailable(f"{p.nombre}: falta el directorio de credencial dedicado")

        base = Path(os.environ.get("JAX_CLI_RUN_DIR", "/run/jax-cli"))
        rundir = base / uuid.uuid4().hex
        try:
            await asyncio.to_thread(
                _preparar_rundir, base, rundir, p.archivo_nombre, p.archivo_sistema(system_prompt),
            )
            argv = argv_confinado_cli(
                _BWRAP_BIN, work_host=str(rundir), home_sandbox=p.home_sandbox,
                binds_rw=[(cred_host, p.cred_destino)], binds_ro=[(dir_bin, dir_bin)],
                cmd=p.comando(ruta_bin, modelo),
            )
            n = _ranuras_de(p)
            espera = p.espera_lock_s if espera_lock_s is None else espera_lock_s
            proc, stdout, _stderr = await ejecutar(
                argv, p.env_fijo(), conversacion.encode("utf-8"), timeout,
                adquirir=lambda _t: _ranura_adquirir(p.nombre, n, espera),
                liberar=lambda h: _ranura_liberar(h, p, n, cred_host),
                cwd="/",
            )
        except LockTimeout:
            raise  # LockTimeout tambien es TimeoutError: no convertirlo en TimeoutCLI
        except asyncio.TimeoutError:
            raise TimeoutCLI(f"{p.nombre}: sin respuesta en {timeout}s") from None
        finally:
            await asyncio.to_thread(_borrar_rundir, rundir)
        texto, tin, tout = p.parsear(stdout, proc.returncode)
        return ResultadoCLI(texto=texto, tokens_in=tin, tokens_out=tout, version_cli=version)
    except asyncio.CancelledError:
        clase = "Cancelado"
        raise
    except BaseException as exc:  # noqa: BLE001 -- registra y RELANZA; nada se traga
        clase = getattr(exc, "clase", None) or type(exc).__name__
        # el motivo distingue las causas de una misma clase (SandboxUnavailable tiene
        # varias); sin saltos de linea, para que un mensaje no pueda fabricar otra linea
        motivo = _campo_log(exc)
        raise
    finally:
        logger.info(
            "run_cli correlation_id=%s entry_point=%s user_id=%s tenant_id=%s perfil=%s modelo=%s "
            "version_cli=%s clase=%s latencia_ms=%d motivo=%s",
            _campo_log(correlation_id), _campo_log(entry_point),
            _campo_log(getattr(titular, "user_id", None)), _campo_log(getattr(titular, "tenant_id", None)),
            _campo_log(perfil), _campo_log(modelo), _campo_log(version), _campo_log(clase or "ok"),
            int((time.monotonic() - t0) * 1000), motivo,
        )
