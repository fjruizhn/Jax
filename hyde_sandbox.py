"""
hyde_sandbox — confinamiento de bubblewrap para el subprocess `claude` de
Hyde. Fix de fondo del hallazgo P0 (ver jax-hyde-bash-sin-jail-p0 en
memoria): --allowedTools NUNCA fue una defensa real -- "Bash" pelado no
tiene jail, y "Bash(<cmd> *)" solo cubre cat/redireccion (python3 -c
"open(path).read()" y `git diff --no-index` la esquivan, confirmado
empiricamente). Este modulo confina a nivel de NAMESPACE DE MONTAJE: lo
que no esta bind-mounteado acá NO EXISTE dentro del sandbox, sin importar
que comando corra adentro -- no hay heuristica que esquivar.

Compartido entre jacobs/executor.py::_invoke_hyde (real, via el symlink
las_manos/jacobs -> jax/jacobs) y jax/muscles/subprocess_muscle.py (REPL
viejo). Vive en el repo root -- mismo patron que facet_resolver.py /
credential_resolver.py (repo root real, symlinkeados en las_manos/): este
archivo se symlinkea como las_manos/hyde_sandbox.py, y el REPL lo importa
directo porque su PYTHONPATH es $HOME/jax (repo root).

Alcance decidido por Fernando (opcion b, sesion sandbox 2026-08-22):
  - Lectura: los dos repos completos (~/jax, ~/jax-platform) -- Hyde puede
    leer codigo real para trabajo tecnico, como hacia el CLI viejo.
  - Escritura: SOLO dentro de HYDE_WORKSPACE_DIR. Escribir directo a los
    repos reales es una decision aparte, no entra en este alcance.
  - $HOME minimo, tmpfs efimero: NUNCA el ~/.claude real de Fernando.
    Hallazgo de esta misma sesion (T2): el $HOME real dispara hooks
    personales (PreToolUse: block-subagent-git-write.sh, SessionStart:
    superpowers) y carga todo el arbol de plugins (ruflo, token-optimizer,
    frontend-design, etc.) DENTRO de la ejecucion de Hyde, fuera del gate
    de --allowedTools -- eso es un camino de ejecucion de Fernando, no de
    Hyde, que ninguna gobernanza cubrio nunca (ver hallazgo aparte,
    conectado a "sub-agentes sin gobernanza", en memoria). Este modulo lo
    cierra para Hyde dandole un $HOME propio: solo credenciales (bind
    read-only EN VIVO desde el archivo real -- nunca copiadas, mismo
    criterio que el refresh de OAuth: leer en caliente, nunca stale) +
    trust del workspace + settings.json vacio. Sin hooks, sin plugins,
    sin historial.
  - Entorno: --clearenv + --setenv puntual. jax-las-manos.service carga
    TODOS los secretos de /etc/jax/.env como variables de entorno
    (DEEPSEEK_API_KEY, JAX_DB_PASSWORD, FERNET_KEY, KIMI_API_KEY, etc. --
    confirmado, 23 variables). Sin este --clearenv, el proceso de Hyde
    heredaria eso por default (asyncio.create_subprocess_exec hereda el
    entorno del padre si no se le pasa `env=`) -- un vector que ni
    siquiera necesita tocar el filesystem, ningun hallazgo previo lo
    cubria. Se resuelve ACÁ, en el wrapper, para que sea una sola fuente
    de verdad sin importar que pase el llamador.
  - Red: --share-net (host completo). bwrap NO tiene forma de acotar red
    por dominio/IP -- es namespace de red compartido o nada (unshare-net
    aislaria a Hyde de la API de Anthropic, que es su unica funcion). Un
    allowlist de red real requeriria un proxy egress aparte (proyecto
    propio, no entra en esta ronda) -- declarado, no resuelto.

Fail-closed (P10): sin bwrap disponible y ejecutable, SandboxUnavailable
sube y Hyde NO arranca. Nunca degrada a "corre sin confinamiento".

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

REAL_JAX_REPO = "/home/fruiz/jax"
REAL_JAX_PLATFORM_REPO = "/home/fruiz/jax-platform"
REAL_NVM_DIR = "/home/fruiz/.nvm"
REAL_CREDENTIALS = "/home/fruiz/.claude/.credentials.json"

# $HOME virtual DENTRO del sandbox -- nunca /home/fruiz real. Path elegido
# para no solapar con ningun bind de arriba (evita el problema de montar
# un tmpfs sobre un padre que ya tiene hijos bindeados -- mas simple y mas
# seguro que remontar /home/fruiz).
SANDBOX_HOME = "/home/hyde-sandbox"

# Template minimo de $HOME (solo trust + settings vacio, nunca
# credenciales -- esas se bindean en vivo desde REAL_CREDENTIALS). Vive
# fuera de cualquier repo bindeado -- no necesita estar en un path visible
# dentro del sandbox, bwrap lo lee del host al construir los binds.
_TEMPLATE_DIR = Path("/home/fruiz/.hyde-sandbox-home-template")

# /etc puntual para DNS + TLS + NSS -- NUNCA /etc entero (expondria
# /etc/jax/.env, root:fruiz 0660, el grupo fruiz SI tiene lectura real).
_ETC_RO_PATHS = (
    "/etc/resolv.conf", "/etc/nsswitch.conf", "/etc/hosts",
    "/etc/ssl", "/etc/passwd", "/etc/group",
)

_BWRAP_BIN = shutil.which("bwrap") or "/usr/bin/bwrap"

_SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class SandboxUnavailable(Exception):
    """bwrap no disponible o no ejecutable en runtime. Fail-closed (P10):
    el llamador NO debe atrapar esto para degradar a ejecución sin
    sandbox -- Hyde simplemente no arranca."""


def _ensure_home_template(workspace_dir: str) -> Path:
    """Crea/actualiza (idempotente) el template de $HOME que se bindea
    read-only dentro del tmpfs de SANDBOX_HOME. Se reescribe en cada
    llamada -- es texto minusculo, ningun costo real, y evita que quede
    desalineado si workspace_dir cambia alguna vez."""
    claude_dir = _TEMPLATE_DIR / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / "settings.json").write_text("{}\n", encoding="utf-8")
    (_TEMPLATE_DIR / ".claude.json").write_text(
        json.dumps({"projects": {workspace_dir: {"hasTrustDialogAccepted": True}}}) + "\n",
        encoding="utf-8",
    )
    return _TEMPLATE_DIR


def wrap_hyde_command(cmd: list[str], workspace_dir: str) -> list[str]:
    """Envuelve `cmd` (la invocacion real de `claude`) en un bwrap que lo
    confina a nivel de namespace de montaje. Devuelve la lista completa
    para pasar tal cual a asyncio.create_subprocess_exec -- el `cwd`/`env`
    que el llamador ya usaba siguen siendo validos (bwrap ignora el `env`
    heredado via --clearenv, y el `cwd` del proceso real lo fija --chdir
    adentro del sandbox, no el `cwd` del create_subprocess_exec externo).

    Lanza SandboxUnavailable si bwrap no esta disponible -- el llamador NO
    debe atrapar esta excepcion para caer a ejecucion sin sandbox."""
    if not (_BWRAP_BIN and os.path.isfile(_BWRAP_BIN) and os.access(_BWRAP_BIN, os.X_OK)):
        raise SandboxUnavailable(
            f"bwrap no encontrado o no ejecutable ({_BWRAP_BIN!r}) -- "
            "Hyde no arranca sin confinamiento (fail-closed, P10)"
        )

    os.makedirs(workspace_dir, exist_ok=True)
    template_dir = _ensure_home_template(workspace_dir)

    argv = [
        _BWRAP_BIN,
        "--unshare-all", "--share-net",  # red completa: es la unica forma de que bwrap deje llegar a la API de Anthropic
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--setenv", "HOME", SANDBOX_HOME,
        "--setenv", "PATH", _SAFE_PATH,
        "--setenv", "LANG", "C.UTF-8",
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

    for etc_path in _ETC_RO_PATHS:
        if os.path.exists(etc_path):
            argv += ["--ro-bind", etc_path, etc_path]

    # node/claude.exe -- fuera de /usr, vive bajo ~/.nvm.
    if os.path.isdir(REAL_NVM_DIR):
        argv += ["--ro-bind", REAL_NVM_DIR, REAL_NVM_DIR]

    # repos en lectura (decision de Fernando, opcion b) -- nunca los
    # directorios .old-pre-filter-repo-* hermanos: no se listan, no entran.
    argv += ["--ro-bind", REAL_JAX_REPO, REAL_JAX_REPO]
    if os.path.isdir(REAL_JAX_PLATFORM_REPO):
        argv += ["--ro-bind", REAL_JAX_PLATFORM_REPO, REAL_JAX_PLATFORM_REPO]

    # workspace: lectura+escritura, PISA el ro-bind de arriba para esa ruta
    # puntual (bwrap resuelve por orden de argumentos -- el ultimo bind
    # para una ruta dada gana; el resto de REAL_JAX_REPO sigue solo-lectura).
    argv += ["--bind", workspace_dir, workspace_dir]

    # $HOME minimo, efimero -- tmpfs fresco en CADA invocacion, nada
    # persiste entre corridas de Hyde. Credenciales bindeadas en vivo
    # desde el archivo real, solo lectura -- nunca copiadas.
    argv += [
        "--tmpfs", SANDBOX_HOME,
        "--ro-bind", str(template_dir / ".claude.json"), f"{SANDBOX_HOME}/.claude.json",
        "--ro-bind", str(template_dir / ".claude" / "settings.json"), f"{SANDBOX_HOME}/.claude/settings.json",
    ]
    if os.path.isfile(REAL_CREDENTIALS):
        argv += ["--ro-bind", REAL_CREDENTIALS, f"{SANDBOX_HOME}/.claude/.credentials.json"]

    argv += ["--chdir", workspace_dir, "--"]
    argv += cmd
    return argv


# Serializa TODAS las invocaciones de `claude` sandboxeado entre si, sin
# importar el llamador NI el proceso de SO -- antes cada uno tenia su
# propio mecanismo (Jacobs: HYDE_SEMAPHORE en jacobs/executor.py, un
# asyncio.Semaphore que solo sirve DENTRO de un proceso; REPL: ninguno).
# Jacobs corre DENTRO del proceso de las_manos (systemd jax-las-manos);
# el REPL (jax/core/main.py) es un proceso de SO SEPARADO -- confirmado
# por enumeracion real de imports, 2026-08-25. Un asyncio.Semaphore de
# modulo no cruza esa frontera. Se usa flock(2) sobre un archivo en su
# lugar -- un lock a nivel de kernel, visible por CUALQUIER proceso que
# abra el mismo path.
#
# DONDE vive ese archivo importa tanto como el lock mismo: NUNCA dentro
# de workspace_dir. workspace_dir se bindea READ-WRITE dentro del sandbox
# (ver wrap_hyde_command) -- el propio `claude` confinado, un `rm -rf` de
# limpieza o una tarea de Hyde a la que le pidieron "ordenar el
# workspace" podian BORRAR el archivo del lock. flock(2) es del inodo, no
# del path: borrar el path NO libera al que ya tiene el lock, pero el
# SIGUIENTE que llame open(path, "w") crea un inodo nuevo y toma su lock
# al instante -- dos `claude` corriendo a la vez, sin error y sin log, la
# garantia evaporada en silencio. Por eso el lock vive en el /tmp del
# HOST, que NO esta bind-mounteado (el sandbox recibe su propio
# `--tmpfs /tmp` privado, desconectado del host): fuera del alcance del
# proceso confinado.
#
# El nombre del archivo se deriva del workspace_dir resuelto (hash corto)
# para que workspaces distintos tengan locks independientes -- no un unico
# lock global. workspace_dir ya es JAX_WORKSPACE_DIR resuelto por cada
# llamador (fuente unica en /etc/jax/.env, ver
# jax-workspace-relocation-fix) -- el lock hereda esa misma fuente de
# verdad sin leer la env var de nuevo aca.
_CLAUDE_SUBPROCESS_LOCK_DIR_NAME = "jax-claude-subprocess-locks"
_CLAUDE_SUBPROCESS_LOCK_POLL_S = 0.05


def _lock_path_for_workspace(workspace_dir: str) -> Path:
    """Path del archivo de lock para `workspace_dir`. SIEMPRE fuera de
    workspace_dir (ver comentario de arriba) -- en /tmp del HOST (ruta fija,
    no tempfile.gettempdir()), que el sandbox no ve. Usa una ruta absoluta
    fija porque DOS procesos INDEPENDIENTES (las_manos systemd y REPL shell)
    deben computar EXACTAMENTE EL MISMO path sin depender del estado de
    entorno heredado (si TMPDIR/TEMP/TMP diferente, tomarian dos locks
    distintos, la misma clase de falla que este todo intenta cerrar, solo
    trasladada). El nombre es un hash corto del workspace resuelto:
    workspaces distintos -> locks independientes."""
    digest = hashlib.sha256(str(Path(workspace_dir).resolve()).encode("utf-8")).hexdigest()[:16]
    return Path("/tmp") / _CLAUDE_SUBPROCESS_LOCK_DIR_NAME / f"{digest}.lock"


def _acquire_cross_process_lock(workspace_dir: str, timeout: float):
    """BLOQUEANTE -- llamar SOLO via asyncio.to_thread, nunca en el event
    loop (flock(2) no tiene equivalente async). Sondea con LOCK_NB en vez
    de bloquear en LOCK_EX puro para poder fail-closed con un timeout
    explicito: si otro proceso (REPL o las_manos) tiene el lock mas de
    `timeout` segundos, lanza TimeoutError con mensaje explicito en vez de
    colgar el thread para siempre.

    Devuelve el file handle abierto -- el llamador debe pasarlo a
    _release_cross_process_lock (tambien via to_thread) cuando termine,
    en un finally."""
    lock_path = _lock_path_for_workspace(workspace_dir)
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
                    f"no se pudo adquirir el lock cross-proceso de subprocess "
                    f"'claude' en {timeout}s ({lock_path}) -- otro proceso "
                    "(REPL o las_manos) sigue teniendo un claude corriendo. "
                    "Fail-closed: no se lanza sin exclusion mutua real."
                )
            time.sleep(_CLAUDE_SUBPROCESS_LOCK_POLL_S)


def _release_cross_process_lock(fh) -> None:
    """BLOQUEANTE (aunque en la practica instantaneo) -- llamar via
    asyncio.to_thread por simetria con _acquire_cross_process_lock."""
    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    fh.close()


async def run_sandboxed_claude(
    cmd: list[str], workspace_dir: str, prompt: str, timeout: float,
) -> tuple["asyncio.subprocess.Process", bytes, bytes]:
    """Unico punto de entrada aprobado para lanzar `claude` como subproceso
    -- ver policy/tests/test_claude_subprocess_solo_via_sandbox.py, que
    falla el CI si aparece un create_subprocess_exec/create_subprocess_shell
    de un comando que mencione "claude" fuera de este modulo.

    Aplica wrap_hyde_command (sandbox de bwrap, fail-closed via
    SandboxUnavailable si no hay bwrap -- NO se atrapa acá) y serializa
    con un flock(2) cross-proceso derivado de workspace_dir (ver
    _acquire_cross_process_lock / _lock_path_for_workspace) -- no un
    asyncio.Semaphore, que no cruza la frontera real entre el proceso de
    las_manos (Jacobs) y el proceso del REPL.

    `timeout` se aplica INDEPENDIENTEMENTE a cada una de las dos esperas
    (adquisicion del lock, luego subprocess communicate) -- el peor caso
    combinado es hasta ~2×`timeout`, no un presupuesto compartido. Antes la espera del lock
    tenia una constante fija de 30s, mucho mas corta que los presupuestos
    reales (300s en la mayoria de los steps de Jacobs, 900s en
    reconcile/design/reason -- ver jacobs/models.py y jacobs/plan.py). Eso
    era una regresion funcional frente al asyncio.Semaphore que este lock
    reemplazo: Jacobs puede programar dos steps `hyde` en la misma ola
    paralela, y el segundo LEGITIMAMENTE esperaba a que terminara el
    primero. Con 30s fijos ese segundo step moria sin haber lanzado nada,
    y _run_one_step lo reportaba como "Timeout (300s)" a los 30 segundos
    (asyncio.TimeoutError ES TimeoutError desde 3.11) -- una trampa de
    depuracion. Sigue siendo fail-closed: agotado el presupuesto real,
    lanza TimeoutError explicito en vez de colgarse para siempre.

    Devuelve (proc, stdout, stderr) crudos -- la interpretacion de exit
    code / contenido de stderr queda en el llamador, cada uno con su
    propio contrato de excepciones (RuntimeError en Jacobs,
    MuscleInvocationError en el REPL viejo -- no se unifican acá).

    En timeout (de la corrida real O de la espera del lock) o
    cancelacion: mata el proceso si llego a lanzarse, cosecha el zombie
    con wait(), y RE-LANZA la excepcion SIN envolver -- CancelledError
    debe seguir siendo CancelledError (Jacobs cancela _dispatch_step desde
    afuera con su propio wait_for; envolverla rompe la propagacion real de
    cancelacion de asyncio). TimeoutError del lock y TimeoutError del
    wait_for son la misma clase (asyncio.TimeoutError es alias de
    TimeoutError desde Python 3.11) -- ambos llamadores ya distinguen por
    esa clase, no hace falta un tipo nuevo."""
    sandboxed_cmd = wrap_hyde_command(cmd, workspace_dir)

    # El presupuesto del lock es el del llamador, no una constante fija
    # (ver docstring) -- un step encolado espera lo que su step realmente
    # dura, como hacia el semaforo viejo.
    lock_fh = await asyncio.to_thread(
        _acquire_cross_process_lock, workspace_dir, timeout,
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *sandboxed_cmd,
            cwd=workspace_dir,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
                timeout=timeout,
            )
        except (asyncio.TimeoutError, asyncio.CancelledError):
            proc.kill()
            await proc.wait()
            raise
    finally:
        await asyncio.to_thread(_release_cross_process_lock, lock_fh)

    return proc, stdout, stderr
