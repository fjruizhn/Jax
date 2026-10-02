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
    cierra para Hyde dandole un $HOME propio: trust del workspace +
    settings.json vacio, sin hooks, sin plugins, sin historial. Mas la
    credencial de Anthropic, por UNA de dos vias (2026-09-27, decision de
    Fernando: NADA de API key para Hyde):
      1. `CLAUDE_CODE_OAUTH_TOKEN` en el entorno del proceso que llama a
         `wrap_hyde_command` -- la cuenta Max de Fernando via
         `claude setup-token`, guardada en /etc/jax/.env. Si esta presente
         (no vacia tras `.strip()` -- un valor de solo espacios cuenta como
         ausente), es la UNICA credencial que se usa y el archivo de
         REAL_CREDENTIALS NO se monta -- no hace falta y evita depender de
         que jaxsvc pueda leer un archivo 600 fruiz:fruiz que nunca le
         perteneció.
      2. Sin esa variable: el archivo REAL_CREDENTIALS (bind read-only EN
         VIVO desde el archivo real -- nunca copiado, mismo criterio que el
         refresh de OAuth: leer en caliente, nunca stale), pero SOLO si es
         legible por este proceso (`os.access(..., os.R_OK)`, no sólo
         `os.path.isfile`). Corrida como `jaxsvc` (desde el 2026-09-17) ese
         archivo real es 600 fruiz:fruiz -- `isfile` daba True (alcanza con
         poder recorrer directorios) y lo montaba igual, pero el `claude`
         de adentro no podia leerlo: Hyde quedaba con un bind inútil y sin
         credencial, en silencio. Ahora, si ninguna de las dos vías da una
         credencial usable, `wrap_hyde_command` NO arma el sandbox: loguea
         un warning (sin datos sensibles) y lanza `HydeCredentialUnavailable`
         -- fail-closed, mismo criterio que `CredentialUnavailableError` en
         `credential_resolver.py` (DEUDA.md, E-25): lanzar `claude` sabiendo
         que no puede autenticar sólo gasta el lock cross-proceso y el
         presupuesto de tiempo del llamador para terminar en un error de
         auth genérico.
    CORREGIDO 2026-09-27 (auditoría adversarial, hallazgo B-1 BLOCK sobre
    el primer intento de este mismo cambio): el token NUNCA va en el argv
    de bwrap. La primera versión lo pasaba con `--setenv
    CLAUDE_CODE_OAUTH_TOKEN <valor>`, y el argv completo de un proceso es
    legible por CUALQUIER usuario del host vía `/proc/<pid>/cmdline`
    (world-readable por defecto; a diferencia de `/proc/<pid>/environ`,
    que exige el mismo UID o CAP_SYS_PTRACE) -- verificado en hall9000:
    `/proc` no tiene `hidepid` montado, y `fruiz`/`axioma` ven con `ps` los
    procesos de `jaxsvc`. Ver la sección "Entorno" de abajo: ahora
    `wrap_hyde_command` devuelve `(argv, env)`, y el `env` -- nunca el
    argv -- es la única vía por la que cruza el token.
  - Entorno: la frontera es el parámetro `env=` de
    `asyncio.create_subprocess_exec`, no `--clearenv`/`--setenv` de bwrap.
    `wrap_hyde_command` devuelve una tupla `(argv, env)`: `env` es el
    entorno MÍNIMO y COMPLETO (HOME=SANDBOX_HOME, PATH segura, LANG, y
    CLAUDE_CODE_OAUTH_TOKEN si hay token) que el llamador debe pasar TAL
    CUAL -- nunca fusionado con `os.environ` -- como `env=` a
    `create_subprocess_exec` (lo hace `run_sandboxed_claude`, el único
    llamador aprobado). Como `env=` REEMPLAZA el entorno del proceso
    exec-ado en vez de heredarlo, bwrap arranca viendo EXACTAMENTE ese
    diccionario -- nunca los 20+ secretos que jax-las-manos.service carga
    de /etc/jax/.env (DEEPSEEK_API_KEY, JAX_DB_PASSWORD, FERNET_KEY,
    KIMI_API_KEY, CLAUDE_CODE_OAUTH_TOKEN, etc.) -- y por eso bwrap YA NO
    necesita `--clearenv` ni `--setenv`: sin esas banderas, bwrap
    simplemente hereda sin tocarlo el entorno de quien lo exec-ea, que es
    justo ese diccionario mínimo. Antes de este cambio (2026-09-27) la
    frontera era `--clearenv` + `--setenv` puntual dentro del argv de
    bwrap -- funcionaba para bloquear los secretos del padre, pero
    cualquier `--setenv` (incluida una credencial) terminaba en el argv,
    que es más expuesto que el entorno (ver el hallazgo B-1 de arriba).
    RIESGO RESIDUAL ACEPTADO (B-4, mismo que existía con el archivo, que
    además incluía el refreshToken): DENTRO del sandbox, el propio agente
    (que tiene Bash y red completa) puede leer su propio
    CLAUDE_CODE_OAUTH_TOKEN en su propio entorno -- eso no cambia con este
    fix, que sólo saca el token de la vista de OTROS procesos del host.
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
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import cli_sandbox
from cli_sandbox import SandboxUnavailable  # misma clase: la define el nucleo comun

logger = logging.getLogger("hyde_sandbox")

REAL_JAX_REPO = "/home/fruiz/jax"
REAL_JAX_PLATFORM_REPO = "/home/fruiz/jax-platform"
REAL_NVM_DIR = "/home/fruiz/.nvm"
REAL_CREDENTIALS = "/home/fruiz/.claude/.credentials.json"

# Variable de entorno que el CLI de Claude Code honra de forma nativa para
# autenticar con una cuenta Max/Pro via `claude setup-token` -- decision de
# Fernando (2026-09-27): NADA de API key para Hyde. Es la UNICA variable
# secreta que entra al `env` mínimo que wrap_hyde_command devuelve (ver esa
# función) -- nunca al argv de bwrap (B-1, auditoría adversarial 2026-09-27).
# Nombre en una constante, no repetido como literal, para que
# _hyde_sandbox_test.py y _hyde_containment_test.py no puedan desalinearse
# del valor real.
HYDE_OAUTH_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"

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
_ETC_RO_PATHS = cli_sandbox.ETC_RO_PATHS

_BWRAP_BIN = shutil.which("bwrap") or "/usr/bin/bwrap"

_SAFE_PATH = cli_sandbox.SAFE_PATH


class HydeCredentialUnavailable(Exception):
    """Ni HYDE_OAUTH_TOKEN_ENV en el entorno del proceso que arma el sandbox
    ni REAL_CREDENTIALS legible por ese mismo proceso. Fail-closed, mismo
    criterio que CredentialUnavailableError en credential_resolver.py
    (DEUDA.md, E-25 "retiro del fallback a .env de credenciales"): el
    llamador NO debe atrapar esto para lanzar `claude` sin credencial --
    haría gastar el lock cross-proceso y el presupuesto de tiempo del
    llamador para terminar en un error de autenticación genérico, en vez de
    fallar acá con un motivo explícito antes de tocar el filesystem del
    sandbox o el lock. El mensaje de esta excepción NUNCA lleva el valor de
    ninguna credencial -- sólo dice cuál de las dos vías faltó."""


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


def wrap_hyde_command(cmd: list[str], workspace_dir: str) -> tuple[list[str], dict[str, str]]:
    """Envuelve `cmd` (la invocacion real de `claude`) en un bwrap que lo
    confina a nivel de namespace de montaje. Devuelve `(argv, env)`:

    - `argv`: la lista completa para pasar tal cual a
      asyncio.create_subprocess_exec. El `cwd` del proceso real lo fija
      --chdir adentro del sandbox, no el `cwd` del create_subprocess_exec
      externo (ese sigue siendo válido para lo que el llamador ya hacía).
    - `env`: el entorno MÍNIMO Y COMPLETO que el llamador debe pasar TAL
      CUAL -- nunca fusionado con os.environ -- como `env=` a
      create_subprocess_exec. Esa es la frontera real de aislamiento de
      entorno (B-1, auditoría adversarial 2026-09-27): `argv` NUNCA lleva
      ninguna credencial -- el argv completo de un proceso es legible por
      cualquier usuario del host vía /proc/<pid>/cmdline (a diferencia de
      /proc/<pid>/environ, que exige el mismo UID o CAP_SYS_PTRACE). Por
      eso el argv de bwrap ya no usa --clearenv ni --setenv: sin esas
      banderas, bwrap hereda sin tocarlo el entorno de quien lo exec-ea,
      que es exactamente este `env` -- ni más (los secretos reales de
      jaxsvc) ni menos (HOME/PATH/LANG/token).

    ADVERTENCIA -- ESTA FUNCIÓN, POR SÍ SOLA, NO AÍSLA NADA DEL ENTORNO.
    Es una función PURA: arma el `argv` y calcula el `env` que hacen falta,
    pero no lanza ningún subproceso ni toca `os.environ`. El aislamiento
    real ocurre recién en el llamador, y sólo SI ese llamador pasa el
    `env` de retorno TAL CUAL -- sin fusionarlo con `os.environ`, ni con
    `{**os.environ, **env}`, ni con ningún otro merge -- al parámetro
    `env=` de `asyncio.create_subprocess_exec` (o de `subprocess.run`,
    para quien la use fuera de asyncio). Si un llamador ignora el `env` de
    retorno (no le pasa `env=` a create_subprocess_exec) o lo funde con el
    propio, el proceso exec-ado hereda el entorno REAL de quien llama --
    con los 20+ secretos de /etc/jax/.env si el llamador es jax-las-manos
    -- sin que `wrap_hyde_command` tenga forma de impedirlo: no controla
    el `env=` de nadie más que a través de lo que devuelve.

    El ÚNICO llamador aprobado hoy es `run_sandboxed_claude` (más abajo en
    este mismo módulo), que hace exactamente eso -- ver su docstring y
    `_hyde_sandbox_test.py::RunSandboxedClaudeWrappingTest`, que falla si
    el `env=` que llega a create_subprocess_exec no es el que esta función
    devolvió. Cualquier código nuevo que llame a `wrap_hyde_command`
    directo (sin pasar por `run_sandboxed_claude`) es responsable de
    reproducir esa misma disciplina -- ver `_hyde_containment_test.py`,
    cuyo `Caja.correr` lo hace a mano precisamente porque llama a esta
    función directo, sin `run_sandboxed_claude` de por medio.

    Lanza SandboxUnavailable si bwrap no esta disponible -- el llamador NO
    debe atrapar esta excepcion para caer a ejecucion sin sandbox. Lanza
    HydeCredentialUnavailable si no hay ninguna credencial de Anthropic
    usable (ni HYDE_OAUTH_TOKEN_ENV en el entorno, ni REAL_CREDENTIALS
    legible) -- mismo criterio fail-closed, ver esa excepción."""
    cli_sandbox.verificar_bwrap(_BWRAP_BIN, "Hyde")

    # Credencial de Anthropic -- se resuelve ACÁ (antes de tocar el
    # filesystem del sandbox) para no gastar el lock cross-proceso ni el
    # presupuesto de tiempo del llamador (ver run_sandboxed_claude) lanzando
    # `claude` sabiendo que no va a poder autenticar. Orden de precedencia:
    # 1) HYDE_OAUTH_TOKEN_ENV (cuenta Max de Fernando, `claude setup-token`,
    #    decisión de Fernando 2026-09-27: nunca una API key); 2) el archivo
    # REAL_CREDENTIALS, sólo si es legible por ESTE proceso -- `os.access`
    # con R_OK, no `os.path.isfile`: corriendo como `jaxsvc` (desde el
    # 2026-09-17) el archivo real es 600 fruiz:fruiz, `isfile` da True
    # (alcanza con poder recorrer directorios) pero `jaxsvc` no puede leerlo,
    # así que montarlo igual sólo dejaba un bind inútil sin que nadie se
    # enterara -- ver hallazgo verificado 2026-09-27 (`sudo -u jaxsvc test -r`).
    # `.strip()`: un valor de solo espacios en /etc/jax/.env no cuenta como
    # token presente (B-3, auditoría adversarial 2026-09-27).
    oauth_token = (os.environ.get(HYDE_OAUTH_TOKEN_ENV) or "").strip()
    credentials_file_readable = (
        os.path.isfile(REAL_CREDENTIALS) and os.access(REAL_CREDENTIALS, os.R_OK)
    )
    if not oauth_token and not credentials_file_readable:
        if os.path.exists(REAL_CREDENTIALS):
            motivo = (
                f"{HYDE_OAUTH_TOKEN_ENV} no está en el entorno y "
                f"REAL_CREDENTIALS existe pero no es legible por este proceso"
            )
        else:
            motivo = (
                f"{HYDE_OAUTH_TOKEN_ENV} no está en el entorno y no hay "
                "archivo de credenciales de Anthropic"
            )
        logger.warning("Hyde sin credencial de Anthropic usable (%s) -- no arranca", motivo)
        raise HydeCredentialUnavailable(
            f"Hyde no tiene credencial de Anthropic usable ({motivo}) -- "
            "no arranca sin poder autenticar (fail-closed)"
        )

    os.makedirs(workspace_dir, exist_ok=True)
    template_dir = _ensure_home_template(workspace_dir)

    # Base comun (namespaces, /proc, /dev, /tmp, /usr, /lib*, /etc puntual):
    # sale del nucleo cli_sandbox -- la golden _hyde_wrap_golden_test.py exige
    # que el resultado sea identico byte a byte al de antes del refactor (D-4).
    argv = cli_sandbox.argv_base(_BWRAP_BIN, _ETC_RO_PATHS)

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
    # persiste entre corridas de Hyde.
    argv += [
        "--tmpfs", SANDBOX_HOME,
        "--ro-bind", str(template_dir / ".claude.json"), f"{SANDBOX_HOME}/.claude.json",
        "--ro-bind", str(template_dir / ".claude" / "settings.json"), f"{SANDBOX_HOME}/.claude/settings.json",
    ]
    # Credencial de Anthropic -- ya validada arriba (una de las dos existe,
    # si no ya se lanzó HydeCredentialUnavailable). El token, cuando está,
    # GANA sobre el archivo: no hace falta el bind (evita depender de que
    # este proceso pueda leer un archivo que puede no ser suyo) y es la vía
    # que Fernando decidió como la única soportada hoy. A diferencia del
    # intento anterior (B-1), el token NUNCA se agrega al argv -- va sólo
    # en el `env` de retorno, más abajo.
    if not oauth_token and credentials_file_readable:
        argv += ["--ro-bind", REAL_CREDENTIALS, f"{SANDBOX_HOME}/.claude/.credentials.json"]

    argv += ["--chdir", workspace_dir, "--"]
    argv += cmd

    # Entorno mínimo y completo -- ver docstring de arriba (B-1). El
    # llamador lo pasa TAL CUAL como `env=`, nunca fusionado con
    # os.environ: eso es lo que impide que los secretos reales del proceso
    # que arma el sandbox (jaxsvc, con /etc/jax/.env cargado entero) lleguen
    # a bwrap o al `claude` de adentro.
    env = cli_sandbox.env_minimo(
        SANDBOX_HOME, {HYDE_OAUTH_TOKEN_ENV: oauth_token} if oauth_token else None,
        path=_SAFE_PATH,
    )

    return argv, env


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
    return cli_sandbox.flock_adquirir(
        _lock_path_for_workspace(workspace_dir), timeout, "subprocess 'claude'",
        "otro proceso (REPL o las_manos) sigue teniendo un claude corriendo.",
    )


def _release_cross_process_lock(fh) -> None:
    """BLOQUEANTE (aunque en la practica instantaneo) -- llamar via
    asyncio.to_thread por simetria con _acquire_cross_process_lock."""
    cli_sandbox.flock_liberar(fh)


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
    esa clase, no hace falta un tipo nuevo.

    `env=sandbox_env` se pasa EXPLÍCITO y TAL CUAL a
    create_subprocess_exec -- nunca fusionado con os.environ (B-1,
    auditoría adversarial 2026-09-27): ese diccionario mínimo (ver
    wrap_hyde_command) es la frontera real de aislamiento de entorno.
    Pasarlo es obligatorio -- sin `env=`, asyncio.create_subprocess_exec
    hereda el entorno completo de ESTE proceso (jax-las-manos, con los
    20+ secretos de /etc/jax/.env), que es exactamente el vector que
    wrap_hyde_command existe para cerrar."""
    sandboxed_cmd, sandbox_env = wrap_hyde_command(cmd, workspace_dir)

    # El presupuesto del lock es el del llamador, no una constante fija
    # (ver docstring) -- un step encolado espera lo que su step realmente
    # dura, como hacia el semaforo viejo. `adquirir`/`liberar` se resuelven por
    # nombre de modulo EN CADA LLAMADA (los tests los parchean aca).
    return await cli_sandbox.ejecutar(
        sandboxed_cmd, sandbox_env, prompt.encode("utf-8"), timeout,
        adquirir=lambda t: _acquire_cross_process_lock(workspace_dir, t),
        liberar=lambda fh: _release_cross_process_lock(fh),
        cwd=workspace_dir,
    )
