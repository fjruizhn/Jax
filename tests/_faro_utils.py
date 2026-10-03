"""Ayudas compartidas por las pruebas del Faro (no es un archivo de test: no empieza con
`test_` ni termina en `_test.py`). Un `repo-skills` de juguete con su `origin/main`, y el
arnes del Puerto: servidor en el bucle de la prueba y cliente MCP REAL por el rele stdio."""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from jax.faro import paquete
from jax.faro.bitacora import Bitacora
from jax.faro.identidad import Ejecucion
from jax.faro.transporte import ServidorPuerto
from mcp import Client
from mcp.client.stdio import StdioServerParameters

RAIZ = Path(__file__).resolve().parents[1]

_ENV_GIT = {
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, **_ENV_GIT}
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True, env=env).stdout.strip()


def _git_entrada(repo: Path, *args: str, entrada: str = "") -> str:
    env = {**os.environ, **_ENV_GIT}
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env=env, input=entrada).stdout.strip()


def _escribir(repo: Path, rel: str, datos: str | bytes, modo: int = 0o644) -> None:
    ruta = repo / rel
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(datos if isinstance(datos, bytes) else datos.encode())
    ruta.chmod(modo)


def _commit(repo: Path, msg: str = "c") -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def repo_de_juguete(tmp_path: Path, extra: dict | None = None) -> Path:
    """Un `repo-skills` de juguete con `origin/main` apuntando a su primer commit."""
    r = tmp_path / "repo-skills"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _escribir(r, paquete._FUENTE_CONSTITUCION, "# Nucleo comun\nregla uno\n")
    _escribir(r, "common/skills/alfa/SKILL.md", "---\nname: alfa\ndescription: la alfa mide el rendimiento\n---\ncuerpo alfa\n")
    _escribir(r, "common/skills/alfa/scripts/correr.sh", "#!/bin/sh\necho hola\n", 0o755)
    _escribir(r, "common/skills/beta/SKILL.md", "---\nname: beta\ndescription: la beta endurece contra inyeccion\n---\ncuerpo beta\n")
    _escribir(r, "common/skills/PROCEDENCIA.md", "procedencia\n")
    _escribir(r, "common/agents/explorador.md", "---\nname: explorador\ndescription: Tier 1. Descubre cosas.\ntools: Read, Bash\nmodel: haiku\n---\nCUERPO SECRETO DEL AGENTE\n")
    _escribir(r, "common/commands/ignorado.md", "no entra al paquete\n")
    for rel, datos in (extra or {}).items():
        _escribir(r, rel, datos)
    sha = _commit(r)
    _git(r, "update-ref", "refs/remotes/origin/main", sha)
    return r


def corre(coro):
    return asyncio.run(coro)


def ejecucion(**kw) -> Ejecucion:
    base = dict(run_id="run-1", usuario="u-real", tenant="t-real", faceta="hyde", motor="codex",
                pipeline="p-real", entry_point="repl", id_correlacion="corr-real", uid_esperado=os.getuid())
    base.update(kw)
    return Ejecucion(**base)


def servidor(*a, **kw) -> ServidorPuerto:
    """Un `ServidorPuerto` con la bandera SOLO-PRUEBAS que permite uid_esperado == el del servicio (las pruebas
    corren como un solo usuario). El servicio real no puede activarla: es un argumento del constructor, no
    sale del entorno ni de la configuracion."""
    return ServidorPuerto(*a, solo_pruebas_mismo_uid=True, **kw)


def paquete_listo(tmp_path: Path):
    """(cfg, cargado): un paquete de juguete construido y cargado, para las pruebas que solo necesitan uno."""
    from jax.faro.config import ConfigFaro
    from jax.faro.paquete import cargar_paquete, construir_paquete
    repo = repo_de_juguete(tmp_path)
    cfg = ConfigFaro(repo=repo, sha=_git(repo, "rev-parse", "HEAD"), destino=tmp_path / "ecosistema", uid_duenio=os.getuid())
    construir_paquete(cfg)
    return cfg, cargar_paquete(cfg)


@asynccontextmanager
async def puerto(cfg_puerto, cargado, ej=None, registros=None, bitacora=None, **kw):
    """Un Puerto REAL en un socket Unix temporal; `srv.registros` es la bitacora en memoria.
    `bitacora` permite pasar una propia (p. ej. con observadores); `registros` queda vacio entonces."""
    registros = registros if registros is not None else []
    bit = bitacora if bitacora is not None else Bitacora(emisores=[registros.append])
    async with servidor(cfg_puerto, ej or ejecucion(), cargado, bit, **kw) as srv:
        srv.registros = registros
        yield srv


@asynccontextmanager
async def cliente_por_rele(srv, *, token_file: Path | None = None, **kw):
    """Un cliente MCP real (SDK oficial) hablando con el Puerto POR EL RELE: el rele es un
    subproceso `python -m jax.faro.relay`, stdio de un lado y el socket Unix del otro. El token de
    la ejecucion se le pasa por ARCHIVO (`--token-file`), nunca por argv ni por entorno."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "jax.faro.relay", "--socket", str(srv.ruta_socket), "--token-file", str(token_file or srv.ruta_token)],
        env={"PYTHONPATH": str(RAIZ)}, cwd=str(RAIZ))
    async with Client(params, **kw) as c:
        yield c
