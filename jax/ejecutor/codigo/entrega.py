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
    # MINOR-1 (re-revisión): `fetch.fsckObjects` rechaza un pack mal formado al recibirlo, y lo
    # recibido va a una ref TEMPORAL; la rama del espejo solo se mueve tras un fsck limpio.
    temporal = f"refs/jax/entrante/{rama}"
    with hogar_temporal() as home:
        env = entorno_base(home)
        try:
            await correr_git(["-c", "fetch.fsckObjects=true", "-C", str(espejo), "fetch", "--no-tags",
                              "--no-recurse-submodules", "--no-auto-gc", "--no-auto-maintenance",
                              "--no-write-fetch-head", f"--upload-pack={upload_pack}",
                              "--", ruta_clon, f"+refs/heads/{rama}:{temporal}"],
                             env=env, error="traer_del_clon_fallo", mostrar_error=False)
            await correr_git(["-C", str(espejo), "fsck", "--no-dangling", "--no-progress"],
                             env=env, error="fsck_fallo", mostrar_error=False)
            await correr_git(["-C", str(espejo), "update-ref", "--", f"refs/heads/{rama}", temporal],
                             env=env, error="traer_del_clon_fallo", mostrar_error=False)
        except GitFallo as exc:
            raise EntregaRechazada(str(exc)) from None
        finally:
            try:
                await correr_git(["-C", str(espejo), "update-ref", "-d", "--", temporal],
                                 env=env, error="limpiar_ref_temporal", mostrar_error=False)
            except GitFallo:  # fail-soft: la ref temporal no existía (el fetch falló antes); si quedara, el próximo fetch la pisa con "+" y nunca decide nada
                pass


async def diff_en_el_espejo(espejo: Path, *, mision_id: str, rama_por_omision: str) -> tuple[Cambio, ...]:
    """El diff que revisa C1: `origin/<base>...axioma/<id>`, calculado en el espejo.

    `--text` (2026-09-28, verificado): sin él, un byte NUL en un archivo hace que git diga
    «Binary files differ» y el diff no trae ninguna línea -- un secreto pasaba el barrido."""
    rama = rama_de_la_mision(mision_id)
    base = validar_rama_base(rama_por_omision)
    with hogar_temporal() as home:
        try:
            salida = await correr_git(["-c", "core.quotePath=false", "-C", str(espejo), "diff", "--no-color",
                                       "--no-ext-diff", "--no-textconv",
                                       "--text", "--find-renames", "-U0",
                                       f"refs/remotes/origin/{base}...refs/heads/{rama}", "--"],
                                      env=entorno_base(home), error="diff_fallo")
        except GitFallo as exc:
            raise EntregaRechazada(str(exc)) from None
    return parsear(salida.decode("utf-8", errors="replace"))


@dataclass(frozen=True)
class Commit:
    """Un commit de `origin/<base>..axioma/<id>` visto EN EL ESPEJO. `autor`/`committer` son
    (nombre, correo) crudos (sin .mailmap); `None` si la cabecera no se pudo leer."""
    sha: str
    autor: tuple[str, str] | None
    committer: tuple[str, str] | None
    cambios: tuple[Cambio, ...]


# La cabecera de cada commit en `git log -p`: empieza con \x01, que ninguna línea de un parche
# puede tener al principio (las de contenido llevan su prefijo +/-/espacio y las de cabecera son
# palabras fijas; una ruta con caracteres de control sale entre comillas).
_MARCA = "\x01jax-commit\x01"
_FORMATO = "--format=%x01jax-commit%x01%H%x00%an%x00%ae%x00%cn%x00%ce"
_OID = re.compile(r"[0-9a-f]{40}([0-9a-f]{24})?")


def _commit(cabecera: str, lineas: list[str]) -> Commit:
    campos = cabecera[len(_MARCA):].split("\x00")
    cambios = parsear("\n".join(lineas))
    if len(campos) != 5 or not _OID.fullmatch(campos[0]):
        return Commit(campos[0][:64], None, None, cambios)
    return Commit(campos[0], (campos[1], campos[2]), (campos[3], campos[4]), cambios)


async def commits_de_la_rama(espejo: Path, *, mision_id: str, rama_por_omision: str) -> tuple[Commit, ...]:
    """Cada commit de `origin/<base>..axioma/<id>` con su identidad y su parche, en el espejo.

    Lo usa la entrega para dos cosas: exigir la identidad de Axioma en author Y committer de
    TODOS los commits (Qwen puede fijar cualquier autor), y barrer secretos commit por commit (un
    secreto agregado y luego quitado dentro de la rama no está en el diff neto, pero llegaría a
    GitHub en el historial). `--text` por lo mismo que `diff_en_el_espejo`; `--no-renames` para
    que un archivo renombrado traiga su contenido; `--diff-merges=separate` porque, sin eso, `log
    -p` no muestra lo que un merge agrega por su cuenta; `--no-mailmap` para leer la identidad
    tal como está en el commit."""
    rama = rama_de_la_mision(mision_id)
    base = validar_rama_base(rama_por_omision)
    with hogar_temporal() as home:
        try:
            salida = await correr_git(["-c", "core.quotePath=false", "-C", str(espejo), "--no-replace-objects", "log",
                                       "--no-color", "--no-ext-diff",
                                       "--no-textconv", "--text", "--no-renames", "--no-mailmap", "--no-notes",
                                       "--no-show-signature", "--diff-merges=separate", "-p", "-U0", _FORMATO,
                                       f"refs/remotes/origin/{base}..refs/heads/{rama}", "--"],
                                      env=entorno_base(home), error="log_fallo")
        except GitFallo as exc:
            raise EntregaRechazada(str(exc)) from None
    commits: list[Commit] = []
    cabecera, lineas = None, []
    for linea in salida.decode("utf-8", errors="replace").split("\n"):
        if linea.startswith(_MARCA):
            if cabecera is not None:
                commits.append(_commit(cabecera, lineas))
            cabecera, lineas = linea, []
        elif cabecera is not None:
            lineas.append(linea)
    if cabecera is not None:
        commits.append(_commit(cabecera, lineas))
    return tuple(commits)


async def tamanos_en_el_espejo(espejo: Path, *, mision_id: str, rama_por_omision: str) -> dict[str, int]:
    """Tamaño en bytes (`git cat-file -s`) del blob de cada archivo que cambia en
    `origin/<base>...axioma/<id>`, tal como está EN LA RAMA DEL ESPEJO. Nunca `stat` en el clon:
    Qwen puede commitear un archivo grande y dejar uno chico en el disco. Lo borrado no se mide;
    un enlace simbólico mide lo que git guarda de él (el destino como texto); un submódulo
    (160000) no tiene blob."""
    rama = rama_de_la_mision(mision_id)
    base = validar_rama_base(rama_por_omision)
    tamanos: dict[str, int] = {}
    with hogar_temporal() as home:
        env = entorno_base(home)
        try:
            crudo = await correr_git(["-C", str(espejo), "diff", "--raw", "-z", "--no-renames", "--no-abbrev",
                                      "--no-ext-diff", "--no-textconv",
                                      f"refs/remotes/origin/{base}...refs/heads/{rama}", "--"],
                                     env=env, error="diff_fallo")
            partes = crudo.split(b"\0")
            for meta, ruta in zip(partes[0::2], partes[1::2]):
                _, modo_nuevo, _, oid, estado = meta.decode(errors="replace").lstrip(":").split(" ")
                if estado == "D" or modo_nuevo == "160000":
                    continue
                if not _OID.fullmatch(oid):
                    raise EntregaRechazada("tamano_ilegible")
                n = await correr_git(["-C", str(espejo), "cat-file", "-s", oid], env=env, error="cat_file_fallo")
                tamanos[ruta.decode("utf-8", errors="replace")] = int(n.decode().strip())
        except GitFallo as exc:
            raise EntregaRechazada(str(exc)) from None
    return tamanos


async def tamanos_de_la_rama(espejo: Path, *, mision_id: str, rama_por_omision: str) -> dict[str, int]:
    """Tamaño en bytes de TODO blob que el empuje de `axioma/<id>` llevaría a GitHub -- los de cada
    commit de `origin/<base>..axioma/<id>`, no solo los de la punta (ruling 4b de la Tarea 9: un
    archivo grande agregado y borrado dentro de la rama viaja igual en el historial). Medido EN EL
    ESPEJO: `rev-list --objects` y `cat-file --batch-check`; nunca `stat` en el clon.

    Por ruta, el mayor. Una línea de `rev-list` que no empieza con un oid (una ruta con un salto de
    línea, p. ej.) falla cerrado: `tamano_ilegible`."""
    rama = rama_de_la_mision(mision_id)
    base = validar_rama_base(rama_por_omision)
    rutas: dict[str, str] = {}
    with hogar_temporal() as home:
        env = entorno_base(home)
        try:
            listado = await correr_git(["-C", str(espejo), "--no-replace-objects", "rev-list", "--objects",
                                        f"refs/remotes/origin/{base}..refs/heads/{rama}", "--"],
                                       env=env, error="rev_list_fallo")
            for linea in listado.decode("utf-8", errors="replace").split("\n"):
                if not linea:
                    continue
                oid, _, ruta = linea.partition(" ")
                if not _OID.fullmatch(oid):
                    raise EntregaRechazada("tamano_ilegible")
                rutas.setdefault(oid, ruta)
            if not rutas:
                return {}
            tipos = await correr_git(["-C", str(espejo), "--no-replace-objects", "cat-file",
                                      "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
                                     env=env, error="cat_file_fallo", entrada="".join(f"{o}\n" for o in rutas).encode())
        except GitFallo as exc:
            raise EntregaRechazada(str(exc)) from None
    tamanos: dict[str, int] = {}
    for linea in tipos.decode().split("\n"):
        if not linea:
            continue
        partes = linea.split(" ")
        if len(partes) != 3 or partes[0] not in rutas or not partes[2].isdigit():
            raise EntregaRechazada("tamano_ilegible")  # incluye "<oid> missing"
        if partes[1] == "blob":
            ruta = rutas[partes[0]] or partes[0]
            tamanos[ruta] = max(tamanos.get(ruta, 0), int(partes[2]))
    return tamanos


async def empujar(espejo: Path, *, mision_id: str, rama_por_omision: str, token: str) -> None:
    rama = rama_de_la_mision(mision_id)
    if not referencia_permitida(rama, rama_por_omision, mision_id):
        raise EntregaRechazada(f"rama_no_permitida: {rama}")
    ref = f"refs/heads/{rama}"
    seguimiento = f"refs/remotes/origin/{rama}"
    base_push = ["-C", str(espejo), "push", "--porcelain", "--no-follow-tags", "--recurse-submodules=no"]
    with hogar_temporal() as home:
        env = entorno_red(home, token)
        try:
            await correr_git([*base_push, f"--force-with-lease={ref}", "--", "origin", f"{ref}:{ref}"],
                             env=env, error="git_fallo: push", sanear_con=token)
            return
        except GitFallo as exc:
            if b"stale info" not in exc.salida:
                raise EntregaRechazada(str(exc)) from None
        # MINOR-2 (re-revisión): reconciliación mínima tras un lease rechazado.
        try:
            remoto = await _oid_remoto(espejo, ref, env, token)
            local = (await correr_git(["-C", str(espejo), "rev-parse", "--verify", "--end-of-options", ref],
                                      env=env, error="git_fallo: rev-parse")).decode().strip()
            if remoto == local:
                # El empuje anterior sí llegó (p. ej. se cortó la red antes de la respuesta): éxito,
                # y el seguimiento se pone al día. (Con git 2.53 este caso ni siquiera llega aquí:
                # git lo da por «up to date» antes de mirar el lease.)
                await correr_git(["-C", str(espejo), "update-ref", "--", seguimiento, local],
                                 env=env, error="git_fallo: update-ref")
                return
            if remoto is None:
                # La rama ya no existe en GitHub (borrada al cerrar el PR, p. ej.): se crea de nuevo
                # SIN lease y SIN forzar -- si alguien la recrea en medio, no es avance rápido y falla.
                await correr_git([*base_push, "--", "origin", f"{ref}:{ref}"],
                                 env=env, error="git_fallo: push", sanear_con=token)
                return
        except GitFallo as exc:
            raise EntregaRechazada(str(exc)) from None
    raise EntregaRechazada(f"lease_rechazado: {ref} cambió en el remoto fuera de la misión; no se pisa")


async def _oid_remoto(espejo: Path, ref: str, env: dict[str, str], token: str) -> str | None:
    salida = await correr_git(["-C", str(espejo), "ls-remote", "--", "origin", ref],
                              env=env, error="git_fallo: ls-remote", sanear_con=token)
    for linea in salida.decode(errors="replace").splitlines():
        oid, _, nombre = linea.partition("\t")
        if nombre == ref:
            return oid
    return None


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
