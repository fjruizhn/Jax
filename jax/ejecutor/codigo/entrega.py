"""Entrega de la misión de código (spec 2026-09-28 §3.3). Única pieza con el token.

Empuja SOLO `axioma/<mision_id>` -- nunca `rama_por_omision`, ni ninguna otra rama --
y abre o actualiza el PR correspondiente, etiquetado `axioma`. El token viaja al
subproceso de `git` por `git_token.entorno_git` (GIT_ASKPASS + variable de entorno
del subproceso): nunca en argv, nunca en la URL, nunca en `.git/config`."""
from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path

import httpx

from jax.ejecutor.codigo.git_token import entorno_git

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
ETIQUETA = "axioma"


class EntregaRechazada(RuntimeError):
    pass


def rama_de_la_mision(mision_id: str) -> str:
    if not _UUID.match(mision_id):
        raise ValueError("mision_id_invalido")
    return f"axioma/{mision_id}"


def referencia_permitida(rama: str, rama_por_omision: str, mision_id: str) -> bool:
    return rama == rama_de_la_mision(mision_id) and rama != rama_por_omision


async def _git(clon: Path, *args: str, env: dict | None = None) -> str:
    proc = await asyncio.create_subprocess_exec("git", "-C", str(clon), *args, env=env,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    salida, errores = await proc.communicate()
    if proc.returncode != 0:
        raise EntregaRechazada(f"git_fallo: {' '.join(args[:2])}: {errores.decode(errors='replace')[-300:]}")
    return salida.decode().strip()


async def empujar(clon: Path, *, mision_id: str, rama_por_omision: str, remoto_url: str, token: str) -> None:
    rama = await _git(clon, "rev-parse", "--abbrev-ref", "HEAD")
    if not referencia_permitida(rama, rama_por_omision, mision_id):
        raise EntregaRechazada(f"rama_no_permitida: {rama}")
    with tempfile.TemporaryDirectory(prefix="jax-askpass-") as d:
        env = entorno_git(token, Path(d))
        await _git(clon, "push", "--force-with-lease", remoto_url, f"HEAD:refs/heads/{rama}", env=env)


async def abrir_o_actualizar_pr(cliente: httpx.AsyncClient, *, repo: str, rama: str, base: str,
                                titulo: str, cuerpo: str) -> str:
    duenio = repo.split("/", 1)[0]
    r = await cliente.get(f"/repos/{repo}/pulls", params={"head": f"{duenio}:{rama}", "state": "open"})
    r.raise_for_status()
    abiertos = r.json()
    if abiertos:
        n = abiertos[0]["number"]
        r = await cliente.patch(f"/repos/{repo}/pulls/{n}", json={"title": titulo, "body": cuerpo})
    else:
        r = await cliente.post(f"/repos/{repo}/pulls",
                               json={"title": titulo, "body": cuerpo, "head": rama, "base": base, "draft": False})
    r.raise_for_status()
    pr = r.json()
    et = await cliente.post(f"/repos/{repo}/issues/{pr['number']}/labels", json={"labels": [ETIQUETA]})
    et.raise_for_status()
    return pr["html_url"]
