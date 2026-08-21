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

import json
import os
import shutil
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
