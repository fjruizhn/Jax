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
   bwrap), `env_minimo`, `flock_adquirir`/`flock_liberar` y `ejecutar` (el
   unico `create_subprocess_exec` del modulo). `hyde_sandbox` lo usa tal cual.
2. PERFILES en codigo (`PERFILES`): claude, codex, kimi. La base solo elegira
   una CLAVE de perfil; nunca una ruta de binario ni flags (la confianza sigue
   a quien escribio el valor).
3. TITULAR (`exigir_titular` / `Titular`): la compuerta de la suscripcion. Un
   Titular solo lo construye `exigir_titular`; `run_cli` lo exige.
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
                              version fijada y su SHA256. El spec fija la
                              version en el perfil, pero la que se instala se
                              decide en la operacion (paso 11: la que se
                              pruebe, no la 0.159); sin SHA configurado el
                              perfil NO arranca (BinarioAlterado).
  JAX_CLI_CRED_ROOT           raiz de las credenciales de suscripcion
                              (default /srv/jax-data/cli-suscripcion)
  JAX_CLI_RUN_DIR             directorio por llamada (default /run/jax-cli)
  JAX_CLI_LOCK_DIR            locks por perfil (default /tmp/jax-cli-locks)
  JAX_CLI_MAX_PROMPT_CHARS    tope del prompt (default 32000)

CACHES (cada uno declara su invalidacion en el mismo commit que lo crea):
  - `_CACHE_SHA`: SHA256 de un binario, clave = ruta, firma = (inode, mtime_ns,
    size). Se invalida solo: si cualquiera de los tres cambia, se re-hashea. Es
    la invalidacion que pide el spec §1.
  - La lista de titulares y la verificacion en `jax_users` NO se cachean: se
    leen del entorno y de la base en cada llamada (el borrado de un usuario
    surte efecto en la siguiente llamada).

LO QUE ESTE MODULO NO VERIFICO (§8 del spec): el esquema de eventos de
`codex exec --json` en sus errores de cuota/sesion, el esquema de
`kimi --output-format stream-json`, y si `kimi -p` lee el prompt de stdin. El
perfil kimi queda con `canal_prompt_verificado=False` y `run_cli` lo rechaza
(falla cerrado) hasta que se verifique; los parsers estan marcados.

En honor al Prof. Raul Jacobs.
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import re
import secrets
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("cli_sandbox")


# --------------------------------------------------------------------------
# Excepciones
# --------------------------------------------------------------------------

class SandboxUnavailable(Exception):
    """bwrap no disponible o no ejecutable en runtime. Fail-closed (P10):
    el llamador NO debe atrapar esto para degradar a ejecución sin
    sandbox -- el CLI simplemente no arranca."""


class ErrorCLI(Exception):
    """Base de los errores tipados de `run_cli` (spec §4). `clase` es el
    nombre estable que va a los logs y a la telemetria."""
    clase = "ErrorCLI"


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


def flock_adquirir(lock_path: Path, timeout: float, descripcion: str, detalle: str = ""):
    """BLOQUEANTE -- llamar SOLO via asyncio.to_thread. Sondea con LOCK_NB para
    poder fallar cerrado con un timeout explicito en vez de colgar el thread.
    Devuelve el file handle; el llamador lo pasa a `flock_liberar`."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w")
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
                    f"en {timeout}s ({lock_path}) -- {detalle} "
                    "Fail-closed: no se lanza sin exclusion mutua real."
                )
            time.sleep(_LOCK_POLL_S)


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


@dataclass(frozen=True)
class Titular:
    """Prueba de que `exigir_titular` autorizo a este usuario, en este tenant y
    por este punto de entrada. NO se construye a mano: sin el sello privado
    lanza TypeError. `run_cli` exige uno."""
    user_id: int
    tenant_id: int
    entry_point: str
    _sello: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._sello is not _SELLO:
            raise TypeError("Titular solo lo construye cli_sandbox.exigir_titular")


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
    return Titular(user_id=user_id, tenant_id=tenant_id, entry_point=entry_point, _sello=_SELLO)


# --------------------------------------------------------------------------
# PERFILES
# --------------------------------------------------------------------------

_CODEX_FLAGS = (
    "exec", "-", "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config",
    "--ignore-rules", "-s", "read-only",
)
# Herramientas apagadas (spec §1). Nombres tal cual los lista `codex features
# list` de 0.160.0; el golden de _cli_sandbox_test.py los congela y se
# re-verifica al fijar la version (paso 11). `web_search` no es una feature: se
# apaga con -c. NO se usa --dangerously-bypass-approvals-and-sandbox.
_CODEX_DISABLE = (
    "shell_tool", "unified_exec", "apps", "browser_use", "browser_use_external",
    "browser_use_full_cdp_access", "computer_use", "image_generation", "plugins",
    "multi_agent", "memories", "hooks",
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
    archivo_nombre: str = ""
    canal_prompt_verificado: bool = True
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

#: ruta -> ((inode, mtime_ns, size), sha256 observado). Se invalida solo: si la
#: firma cambia, se re-hashea.
_CACHE_SHA: dict[str, tuple[tuple[int, int, int], str]] = {}


def _sha256_archivo(ruta: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _resolver_binario(perfil: Perfil) -> tuple[str, str, str]:
    """(ruta del binario, directorio, version) fijados por la configuracion, ya
    verificados contra su SHA256. Lanza BinarioAlterado ante cualquier duda."""
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
    try:
        st = os.stat(ruta)
    except OSError:
        raise BinarioAlterado(f"{perfil.nombre}: binario fijado ausente") from None
    firma = (st.st_ino, st.st_mtime_ns, st.st_size)
    cacheado = _CACHE_SHA.get(ruta)
    if cacheado and cacheado[0] == firma:
        observado = cacheado[1]
    else:
        observado = _sha256_archivo(ruta)
        _CACHE_SHA[ruta] = (firma, observado)
    if observado != sha:
        raise BinarioAlterado(f"{perfil.nombre}: el SHA256 del binario no coincide con el fijado")
    return ruta, directorio, version


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

    def render(ts):
        if not ts:
            return mensaje
        cuerpo = "\n".join(f"{rol}: {texto}" for rol, texto in ts)
        return (
            f"[Inicio del contexto {nonce}]\n{cuerpo}\n[Fin del contexto {nonce}]\n\n"
            f"Mensaje actual:\n{mensaje}"
        )

    while True:
        out = render(turnos)
        if len(out) <= max_chars:
            return out
        if not turnos:
            raise MensajeDemasiadoLargo(
                f"el mensaje actual ({len(mensaje)} caracteres) no cabe en el tope de {max_chars}"
            )
        turnos = turnos[1:]


# --------------------------------------------------------------------------
# Locks por perfil (N ranuras). NO comparten directorio ni nombre con el de
# Hyde: un Thot en curso no espera a Hyde.
# --------------------------------------------------------------------------

def _lock_dir() -> Path:
    return Path(os.environ.get("JAX_CLI_LOCK_DIR", "/tmp/jax-cli-locks"))


def _ranura_adquirir(perfil: str, ranuras: int, espera: float):
    """BLOQUEANTE (to_thread). Primera ranura libre de `ranuras`; si no hay
    ninguna en `espera` segundos, LockTimeout."""
    d = _lock_dir()
    d.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + espera
    while True:
        for i in range(ranuras):
            fh = open(d / f"{perfil}.{i}.lock", "w")
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return (fh, i)
            except BlockingIOError:
                fh.close()
        if time.monotonic() >= deadline:
            raise LockTimeout(f"sin ranura libre para el perfil {perfil} en {espera}s")
        time.sleep(_LOCK_POLL_S)


def _ranura_liberar(handle, perfil: Perfil, ranuras: int, cred_host: Optional[str]) -> None:
    """BLOQUEANTE (to_thread). Con la ranura todavia tomada, purga el estado
    que el CLI deja en su home dedicado (Kimi no tiene --ephemeral) -- pero solo
    si NINGUNA otra ranura esta en uso (si no, se borraria el estado de una
    llamada en curso); si hay otra corriendo, la purga queda para la proxima."""
    fh, idx = handle
    otras = []
    try:
        if perfil.purgar and cred_host:
            d = _lock_dir()
            libres = True
            for j in range(ranuras):
                if j == idx:
                    continue
                f2 = open(d / f"{perfil.nombre}.{j}.lock", "w")
                try:
                    fcntl.flock(f2.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    otras.append(f2)
                except BlockingIOError:
                    f2.close()
                    libres = False
                    break
            if libres:
                for nombre in perfil.purgar:
                    shutil.rmtree(os.path.join(cred_host, nombre), ignore_errors=True)
    finally:
        for f2 in otras:
            flock_liberar(f2)
        flock_liberar(fh)


# --------------------------------------------------------------------------
# run_cli
# --------------------------------------------------------------------------

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


async def run_cli(
    perfil: str, system_prompt: str, historial, mensaje: str, modelo: str, timeout: float, *,
    titular: Titular, correlation_id: str, entry_point: str,
    ranuras: Optional[int] = None, espera_lock_s: Optional[float] = None,
    max_chars: Optional[int] = None,
) -> ResultadoCLI:
    """Lanza el CLI del perfil SIN herramientas dentro del sandbox y devuelve el
    mensaje final. Exige un `Titular` autorizado por `exigir_titular` para este
    mismo `entry_point`: sin el, no se crea nada ni se lanza nada. Los errores
    son excepciones tipadas (ErrorCLI y subclases), con `.clase` estable."""
    t0 = time.monotonic()
    clase = None
    version = ""
    try:
        if (
            not isinstance(titular, Titular) or titular._sello is not _SELLO
            or titular.entry_point != entry_point
        ):
            raise TitularNoAutorizado("suscripcion_solo_titular", "titular ausente, no autorizado o de otro entry_point")
        p = PERFILES.get(perfil)
        if p is None or not p.via_run_cli:
            raise PerfilNoSoportado(f"perfil {perfil!r} no se sirve por run_cli")
        if not isinstance(modelo, str) or not _RE_MODELO.match(modelo):
            raise ValueError("modelo con formato invalido")
        if not p.canal_prompt_verificado:
            raise ErrorProtocolo(f"{p.nombre}: el canal del prompt no esta verificado (spec §8); no se lanza")
        tope = max_chars or int(os.environ.get("JAX_CLI_MAX_PROMPT_CHARS", MAX_PROMPT_CHARS))
        conversacion = armar_conversacion(historial, mensaje, tope)
        ruta_bin, dir_bin, version = await asyncio.to_thread(_resolver_binario, p)
        verificar_bwrap(_BWRAP_BIN, p.nombre)
        cred_host = os.path.join(os.environ.get("JAX_CLI_CRED_ROOT", "/srv/jax-data/cli-suscripcion"), p.cred_subdir)
        if not os.path.isdir(cred_host):
            raise SandboxUnavailable(f"{p.nombre}: falta el directorio de credencial dedicado")

        base = Path(os.environ.get("JAX_CLI_RUN_DIR", "/run/jax-cli"))
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
        rundir = base / uuid.uuid4().hex
        os.mkdir(rundir, 0o700)
        try:
            _escribir_privado(rundir / p.archivo_nombre, p.archivo_sistema(system_prompt))
            argv = argv_confinado_cli(
                _BWRAP_BIN, work_host=str(rundir), home_sandbox=p.home_sandbox,
                binds_rw=[(cred_host, p.cred_destino)], binds_ro=[(dir_bin, dir_bin)],
                cmd=p.comando(ruta_bin, modelo),
            )
            n = ranuras or p.ranuras
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
            shutil.rmtree(rundir, ignore_errors=True)
        texto, tin, tout = p.parsear(stdout, proc.returncode)
        return ResultadoCLI(texto=texto, tokens_in=tin, tokens_out=tout, version_cli=version)
    except ErrorCLI as exc:
        clase = exc.clase
        raise
    except asyncio.CancelledError:
        clase = "Cancelado"
        raise
    finally:
        logger.info(
            "run_cli correlation_id=%s entry_point=%s perfil=%s modelo=%s version_cli=%s clase=%s latencia_ms=%d",
            correlation_id, entry_point, perfil, modelo, version, clase or "ok",
            int((time.monotonic() - t0) * 1000),
        )
