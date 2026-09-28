"""Todo git que corre `jaxsvc` en una misión de código pasa por aquí (spec 2026-09-28 v1.2,
§3.3 y §4.2).

- Entorno en LISTA BLANCA: de `os.environ` solo PATH y LANG; HOME es un directorio temporal
  propio (nada de `~/.gitconfig`, `~/.git-credentials` ni cachés de nadie). Nada de
  `/etc/jax/.env` ni de otras variables del proceso.
- `GIT_CONFIG_NOSYSTEM=1` y `GIT_CONFIG_GLOBAL=/dev/null`: solo cuenta la configuración del
  repositorio, y el único repositorio donde `jaxsvc` corre git con datos de la misión es el
  ESPEJO, que es suyo y privado.
- Cada llamada lleva `-c core.hooksPath=/dev/null -c credential.helper=` (lista vacía de
  ayudantes): ni ganchos ni ayudantes de credenciales, vengan de donde vengan.
- El token llega SOLO a las operaciones de red, por GIT_ASKPASS leyendo una variable del
  entorno DEL SUBPROCESO: nunca en argv, en la URL ni en ningún archivo. El script responde
  únicamente a los dos avisos exactos de `https://github.com`; cualquier otro (otro host,
  un host que solo CONTIENE github.com, una frase de paso) recibe salida vacía y código ≠ 0.
- Los errores: la salida de error de git puede incluir datos que no controlamos. Con
  `mostrar_error=False` no se incluye nada; si se incluye, son los últimos 300 caracteres
  con el token reemplazado por `***` ANTES de recortar."""
from __future__ import annotations

import asyncio
import contextlib
import math
import os
import signal
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

_ASKPASS = """#!/bin/sh
case "$1" in
  "Username for 'https://github.com': ") printf '%s\\n' x-access-token ;;
  "Password for 'https://x-access-token@github.com': ") printf '%s\\n' "$JAX_GIT_TOKEN_EFIMERO" ;;
  *) exit 1 ;;
esac
"""

BLINDAJE = ("-c", "core.hooksPath=/dev/null", "-c", "credential.helper=")
TOPE_ERROR = 300


class GitFallo(RuntimeError):
    """`args[0]` es el motivo, sin datos del clon. `salida` es el stdout crudo, para que quien
    llama CLASIFIQUE (p. ej. `push --porcelain`); nunca va al mensaje."""

    def __init__(self, motivo: str, salida: bytes = b""):
        super().__init__(motivo)
        self.salida = salida


def _heredadas() -> dict[str, str]:
    return {k: os.environ[k] for k in ("PATH", "LANG") if k in os.environ}


def entorno_minimo(home: Path) -> dict[str, str]:
    """Lo único que hereda un subproceso de la misión: PATH, LANG y un HOME propio."""
    return {**_heredadas(), "HOME": str(home)}


def entorno_base(home: Path) -> dict[str, str]:
    return {**entorno_minimo(home), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0"}


def entorno_red(home: Path, token: str) -> dict[str, str]:
    """Para `fetch`/`push` contra GitHub. Deja `askpass.sh` (0700, sin el token) en `home`."""
    script = home / "askpass.sh"
    script.write_text(_ASKPASS)
    script.chmod(0o700)
    return {**entorno_base(home), "GIT_ASKPASS": str(script), "JAX_GIT_TOKEN_EFIMERO": token}


@contextlib.contextmanager
def hogar_temporal() -> Iterator[Path]:
    """HOME de una operación: 0700 (mkdtemp), se borra al salir."""
    with tempfile.TemporaryDirectory(prefix="jax-git-") as d:
        yield Path(d)


def sanear(texto: str, token: str | None) -> str:
    if token:
        texto = texto.replace(token, "***")
    return texto[-TOPE_ERROR:]


VARIABLE_TOPE = "JAX_EJECUTOR_GIT_TOPE_S"
TOPE_S_POR_OMISION = 300.0


def tope_por_omision() -> float:
    """El tope de cada git: `JAX_EJECUTOR_GIT_TOPE_S`, o 300 s. Un valor presente e inválido es error
    (fail-closed), no el valor por omisión."""
    texto = os.environ.get(VARIABLE_TOPE)
    if texto is None:
        return TOPE_S_POR_OMISION
    valor = float(texto)
    if not math.isfinite(valor) or valor <= 0:
        raise ValueError(f"{VARIABLE_TOPE}_invalido")
    return valor


async def correr_git(args: Sequence[str], *, env: dict[str, str], error: str, sanear_con: str | None = None,
                     mostrar_error: bool = True, entrada: bytes | None = None, tope_s: float | None = None) -> bytes:
    """`git <BLINDAJE> <args>`. Devuelve stdout en bytes; si falla, `GitFallo(error[: …])`.
    `entrada`: lo que va por stdin (p. ej. oids para `cat-file --batch-check`); sin ella, stdin
    es /dev/null. `tope_s` (por omisión `tope_por_omision()`): al vencer se mata el GRUPO de procesos
    entero (git lanza hijos: upload-pack, ssh, askpass) y se lanza `GitFallo(error: tope_vencido)`."""
    tope = tope_por_omision() if tope_s is None else tope_s
    stdin = asyncio.subprocess.DEVNULL if entrada is None else asyncio.subprocess.PIPE
    proc = await asyncio.create_subprocess_exec("git", *BLINDAJE, *args, env=env, stdin=stdin,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                                                start_new_session=True)
    try:
        salida, errores = await asyncio.wait_for(proc.communicate(entrada), tope)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        await proc.wait()
        raise GitFallo(f"{error}: tope_vencido") from None
    if proc.returncode != 0:
        if not mostrar_error:
            raise GitFallo(error, salida)
        raise GitFallo(f"{error}: {sanear(errores.decode(errors='replace'), sanear_con)}", salida)
    return salida
