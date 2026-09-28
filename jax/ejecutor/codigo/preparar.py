"""Preparar la misión de código (spec 2026-09-28 §3.1). Corre como jaxsvc, FUERA de la jaula.

Idempotente: si `raiz/<mision_id>/repo` ya existe (turno >= 2), hace `fetch` y conserva la
rama -- no vuelve a clonar. `dar_acceso(ruta) -> Awaitable[None]` es
`cuenta_axioma.preparar_directorio_de_la_cuenta`/`dar_acceso_recursivo` ligado a la cuenta
`axioma` (inyectable en tests); (ruling del controlador 2026-09-28) se llama SIEMPRE con
una ruta que YA EXISTE -- `base` se crea con `mkdir(parents=True, exist_ok=True)` antes del
primer llamado, porque en producción `dar_acceso` exige la ruta ya creada (os.lstat, nunca
mkdir) y `git clone` necesita ese mismo directorio como cwd."""
from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from jax.ejecutor.codigo.entrega import rama_de_la_mision
from jax.ejecutor.codigo.git_token import entorno_git

# lockfile -> comando de instalación (sin red en la jaula: se instala aquí).
LOCKFILES = (
    ("requirements.txt", ("python3", "-m", "venv", ".venv"), (".venv/bin/pip", "install", "-r", "requirements.txt")),
    ("backend/requirements.txt", ("python3", "-m", "venv", "backend/.venv"),
     ("backend/.venv/bin/pip", "install", "-r", "backend/requirements.txt")),
    ("package-lock.json", None, ("npm", "ci", "--no-audit", "--no-fund")),
    ("frontend/package-lock.json", None, ("npm", "--prefix", "frontend", "ci", "--no-audit", "--no-fund")),
    ("composer.lock", None, ("composer", "install", "--no-interaction", "--no-scripts")),
)


@dataclass(frozen=True)
class Repo:
    owner_repo: str
    remoto_url: str
    comandos_prueba: tuple[str, ...]


@dataclass(frozen=True)
class Clon:
    ruta: Path
    rama: str
    rama_por_omision: str
    dependencias: tuple[str, ...]


async def _correr(*argv, cwd: Path, env=None) -> None:
    proc = await asyncio.create_subprocess_exec(*argv, cwd=str(cwd), env=env,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, errores = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"preparar_fallo: {argv[0]} {argv[1] if len(argv) > 1 else ''}: "
                           f"{errores.decode(errors='replace')[-400:]}")


async def rama_por_omision(cliente: httpx.AsyncClient, repo: str) -> str:
    r = await cliente.get(f"/repos/{repo}")
    r.raise_for_status()
    return r.json()["default_branch"]


async def preparar(repo: Repo, *, mision_id: str, raiz: Path, rama_por_omision: str, token: str, autor: str,
                   dar_acceso) -> Clon:
    rama = rama_de_la_mision(mision_id)
    base = raiz / mision_id
    ruta = base / "repo"
    with tempfile.TemporaryDirectory(prefix="jax-askpass-") as d:
        env = entorno_git(token, Path(d))
        if not ruta.exists():
            # `dar_acceso` (en producción, `cuenta_axioma.dar_acceso_recursivo`) exige la
            # ruta YA CREADA -- nunca hace mkdir. `git clone` también necesita `base` como
            # cwd existente. Se crea acá, antes del primer llamado a `dar_acceso`.
            await asyncio.to_thread(base.mkdir, parents=True, exist_ok=True)
            await dar_acceso(base)
            await _correr("git", "clone", "--branch", rama_por_omision, repo.remoto_url, str(ruta), cwd=base,
                          env=env)
            await _correr("git", "checkout", "-b", rama, cwd=ruta)
        else:
            await _correr("git", "fetch", "origin", cwd=ruta, env=env)
    nombre, correo = autor.rsplit(" <", 1)
    await _correr("git", "config", "user.name", nombre, cwd=ruta)
    await _correr("git", "config", "user.email", correo.rstrip(">"), cwd=ruta)
    instaladas = []
    for lock, previo, comando in LOCKFILES:
        if (ruta / lock).is_file():
            if previo:
                await _correr(*previo, cwd=ruta)
            await _correr(*comando, cwd=ruta)
            instaladas.append(lock)
    await dar_acceso(ruta)
    return Clon(ruta, rama, rama_por_omision, tuple(instaladas) or ("sin_lockfile",))
