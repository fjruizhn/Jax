"""Entrega de la misión de código (spec 2026-09-28 v1.2, §3.3). Única pieza con el token.

Dos repositorios por misión, y `jaxsvc` solo corre git con datos de la misión en el suyo:

- ESPEJO `<raiz>/<misión>/espejo.git` (bare, 0700, sin ACL para la cuenta): su `origin` es
  la URL DERIVADA de un `owner_repo` validado. Aquí -- y solo aquí -- se usa el token, se
  calcula el diff que revisa C1 y se empuja.
- CLON `<raiz>/<misión>/repo`: el de Qwen. Su `.git` es de quien lo puede reescribir, así que
  `jaxsvc` no ejecuta git dentro (`man git`, sección SECURITY: no es seguro correr git en un
  `.git` que viene de una fuente no confiable -- su config y sus ganchos se ejecutan).

Los commits de Qwen llegan al espejo con `git fetch --no-tags --upload-pack=<ssh a la cuenta>
<clon> +refs/heads/axioma/<id>:refs/heads/axioma/<id>`: el lado que lee el clon
(`upload-pack`) corre COMO la cuenta, que es lo que `man git` recomienda ("serve the
repository as an unprivileged user ... via ssh"). Después, `git fsck` en el espejo.

El empuje: `git push origin refs/heads/axioma/<id>:refs/heads/axioma/<id>` con
`--force-with-lease=refs/heads/axioma/<id>` -- el valor esperado es la rama de seguimiento
`refs/remotes/origin/axioma/<id>` DEL ESPEJO, que solo mueve un empuje nuestro (los `fetch`
del espejo nombran siempre y solo la rama por omisión). Si alguien más movió la rama en
GitHub, el lease lo detecta y no se pisa. Nunca etiquetas ni submódulos."""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

import httpx

from jax.ejecutor.codigo.diff import Cambio, parsear
from jax.ejecutor.codigo.git_token import (GitFallo, correr_git, entorno_base, entorno_red, hogar_temporal)
from jax.ejecutor.contratos.cuenta_axioma import Cuenta, ssh_a_la_cuenta

GITHUB = "https://github.com/"
ETIQUETA = "axioma"

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_OWNER_REPO = re.compile(r"[\w.-]+/[\w.-]+", re.ASCII)
_RAMA_BASE = re.compile(r"(?!-)(?!.*\.\.)(?!.*//)[A-Za-z0-9._/-]+(?<![/.])", re.ASCII)
# La ruta del clon viaja por ssh como texto que interpreta el shell de la cuenta: solo lo
# que no necesita comillas.
_RUTA_SEGURA = re.compile(r"/[A-Za-z0-9/._-]+", re.ASCII)


class EntregaRechazada(RuntimeError):
    pass


@dataclass(frozen=True)
class PrEntregado:
    url: str
    notas: tuple[str, ...]  # "pr_reabierto_nuevo": había uno cerrado y se abrió otro (no se reabre)


def rama_de_la_mision(mision_id: str) -> str:
    if not isinstance(mision_id, str) or not _UUID.fullmatch(mision_id):
        raise ValueError("mision_id_invalido")
    return f"axioma/{mision_id}"


def referencia_permitida(rama: str, rama_por_omision: str, mision_id: str) -> bool:
    return rama == rama_de_la_mision(mision_id) and rama != rama_por_omision


def validar_owner_repo(owner_repo: str) -> str:
    if (not isinstance(owner_repo, str) or not _OWNER_REPO.fullmatch(owner_repo)
            or any(parte in (".", "..") for parte in owner_repo.split("/"))):
        raise ValueError(f"owner_repo_invalido: {owner_repo!r} (se espera dueño/repo)")
    return owner_repo


def url_del_repo(owner_repo: str) -> str:
    """La URL del `origin` del espejo. Nunca un campo libre: siempre derivada de aquí."""
    return f"{GITHUB}{validar_owner_repo(owner_repo)}.git"


def validar_rama_base(rama: str) -> str:
    if not isinstance(rama, str) or not _RAMA_BASE.fullmatch(rama):
        raise ValueError(f"rama_por_omision_invalida: {rama!r}")
    return rama


def validar_ruta_segura(ruta: Path) -> str:
    texto = str(ruta)
    if not _RUTA_SEGURA.fullmatch(texto):
        raise ValueError(f"ruta_insegura: {texto!r}")
    return texto


def upload_pack_por_ssh(c: Cuenta) -> str:
    """El `--upload-pack` de producción: la MISMA cuenta, llave y puerto que el resto de
    `cuenta_axioma` (`ssh_a_la_cuenta`). git le agrega la ruta del clon al final."""
    return shlex.join(ssh_a_la_cuenta(c, "git-upload-pack"))


async def traer_del_clon(espejo: Path, clon: Path, *, mision_id: str, upload_pack: str) -> None:
    """Trae SOLO `axioma/<id>` del clon al espejo y verifica los objetos. Ningún error lleva
    salida de estos comandos: la escribe (o la provoca) lo que hay en el clon."""
    rama = rama_de_la_mision(mision_id)
    ruta_clon = validar_ruta_segura(clon)
    with hogar_temporal() as home:
        env = entorno_base(home)
        try:
            await correr_git(["-C", str(espejo), "fetch", "--no-tags", "--no-recurse-submodules", "--no-auto-gc",
                              "--no-auto-maintenance", "--no-write-fetch-head", f"--upload-pack={upload_pack}",
                              "--", ruta_clon, f"+refs/heads/{rama}:refs/heads/{rama}"],
                             env=env, error="traer_del_clon_fallo", mostrar_error=False)
            await correr_git(["-C", str(espejo), "fsck", "--no-dangling", "--no-progress"],
                             env=env, error="fsck_fallo", mostrar_error=False)
        except GitFallo as exc:
            raise EntregaRechazada(str(exc)) from None


async def diff_en_el_espejo(espejo: Path, *, mision_id: str, rama_por_omision: str) -> tuple[Cambio, ...]:
    """El diff que revisa C1: `origin/<base>...axioma/<id>`, calculado en el espejo."""
    rama = rama_de_la_mision(mision_id)
    base = validar_rama_base(rama_por_omision)
    with hogar_temporal() as home:
        try:
            salida = await correr_git(["-C", str(espejo), "diff", "--no-color", "--no-ext-diff", "--no-textconv",
                                       "--find-renames", "-U0", f"refs/remotes/origin/{base}...refs/heads/{rama}",
                                       "--"], env=entorno_base(home), error="diff_fallo")
        except GitFallo as exc:
            raise EntregaRechazada(str(exc)) from None
    return parsear(salida.decode("utf-8", errors="replace"))


async def empujar(espejo: Path, *, mision_id: str, rama_por_omision: str, token: str) -> None:
    rama = rama_de_la_mision(mision_id)
    if not referencia_permitida(rama, rama_por_omision, mision_id):
        raise EntregaRechazada(f"rama_no_permitida: {rama}")
    ref = f"refs/heads/{rama}"
    with hogar_temporal() as home:
        try:
            await correr_git(["-C", str(espejo), "push", "--porcelain", "--no-follow-tags",
                              "--recurse-submodules=no", f"--force-with-lease={ref}", "--", "origin", f"{ref}:{ref}"],
                             env=entorno_red(home, token), error="git_fallo: push", sanear_con=token)
        except GitFallo as exc:
            if b"stale info" in exc.salida:
                raise EntregaRechazada(
                    f"lease_rechazado: {ref} cambió en el remoto fuera de la misión; no se pisa") from None
            raise EntregaRechazada(str(exc)) from None


async def _listar(cliente: httpx.AsyncClient, repo: str, rama: str, estado: str) -> list[dict]:
    duenio = repo.split("/", 1)[0]
    r = await cliente.get(f"/repos/{repo}/pulls", params={"head": f"{duenio}:{rama}", "state": estado})
    r.raise_for_status()
    return r.json()


def _ya_existe(r: httpx.Response) -> bool:
    if r.status_code != 422:
        return False
    try:
        errores = r.json().get("errors") or []
    except ValueError:
        return False
    return any("already exists" in str(e.get("message", "")) for e in errores if isinstance(e, dict))


async def abrir_o_actualizar_pr(cliente: httpx.AsyncClient, *, repo: str, rama: str, base: str,
                                titulo: str, cuerpo: str) -> PrEntregado:
    """Abierto → se actualiza (título, cuerpo, `draft: false`). Cerrado → NO se reabre: se abre
    otro y se declara `pr_reabierto_nuevo`. Un 422 «already exists» (carrera) → se vuelve a
    listar y se actualiza el que apareció."""
    validar_owner_repo(repo)
    notas: tuple[str, ...] = ()
    abiertos = await _listar(cliente, repo, rama, "open")
    if not abiertos:
        if await _listar(cliente, repo, rama, "closed"):
            notas = ("pr_reabierto_nuevo",)
        r = await cliente.post(f"/repos/{repo}/pulls",
                               json={"title": titulo, "body": cuerpo, "head": rama, "base": base, "draft": False})
        if _ya_existe(r):
            notas = ()
            abiertos = await _listar(cliente, repo, rama, "open")
            if not abiertos:
                r.raise_for_status()
        else:
            r.raise_for_status()
            pr = r.json()
    if abiertos:
        n = abiertos[0]["number"]
        r = await cliente.patch(f"/repos/{repo}/pulls/{n}", json={"title": titulo, "body": cuerpo, "draft": False})
        r.raise_for_status()
        pr = r.json()
    et = await cliente.post(f"/repos/{repo}/issues/{pr['number']}/labels", json={"labels": [ETIQUETA]})
    et.raise_for_status()
    return PrEntregado(pr["html_url"], notas)
