"""Preparar la misión de código (spec 2026-09-28 v1.2, §3.1). Corre como jaxsvc, FUERA de la jaula.

Todo queda bajo `<raiz>/<misión>/` (la cuenta solo ATRAVIESA ese directorio):

- `espejo.git` -- bare, 0700, sin ACL para la cuenta. `origin` = URL derivada de `owner_repo`
  (`entrega.url_del_repo`). Solo se trae la rama por omisión; la rama de la misión en GitHub
  no se trae nunca (su seguimiento lo mueve solo nuestro empuje: ver `entrega.empujar`).
- `deps/` -- dependencias instaladas desde los archivos de bloqueo de la RAMA POR OMISIÓN
  (leídos del espejo, no del clon), con un entorno que no hereda nada del proceso salvo PATH
  y LANG, sin scripts de instalación. ACL de solo lectura para la cuenta.
- `repo/` -- el clon de trabajo de Qwen, clonado DESDE el espejo con `--no-hardlinks`, en
  `axioma/<misión>`; enlaces `.venv`/`node_modules`/`vendor` hacia `deps/`. ACL de escritura.
- `preparado.json` -- marca de que el turno 1 terminó.

Turno ≥ 2 (hay marca): solo `fetch` del espejo desde GitHub. No reinstala, no vuelve a aplicar
ACL (la ACL por omisión del clon gobierna lo nuevo) y no ejecuta NADA dentro del clon: su
`.git` y sus binarios ya son de quien los puede reescribir (`man git`, SECURITY).

Si la marca no está pero quedan restos (un turno 1 que se cayó a medias), se rehacen: la
cuenta todavía no trabajó ahí -- la marca se escribe antes de devolver el clon."""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import httpx

from jax.ejecutor.codigo.entrega import (rama_de_la_mision, url_del_repo, validar_rama_base, validar_ruta_segura)
from jax.ejecutor.codigo.git_token import (GitFallo, correr_git, entorno_base, entorno_minimo, entorno_red,
                                           hogar_temporal)
from jax.ejecutor.contratos import cuenta_axioma

_AUTOR = re.compile(r"(?P<nombre>[^<>\x00-\x1f\x7f]*[^<>\s\x00-\x1f\x7f])\s<(?P<correo>[^<>\s@]+@[^<>\s@]+)>")
_REQUIREMENTS = re.compile(r"(?:backend/)?requirements[^/]*\.txt")
_SIN_WHEEL = (b"No matching distribution found", b"Could not find a version that satisfies")
_DIRS_NPM = ("", "frontend")

Instalador = Callable[[list[str], Path, dict[str, str]], Awaitable[tuple[int, bytes]]]


@dataclass(frozen=True)
class Repo:
    owner_repo: str
    comandos_prueba: tuple[str, ...]


@dataclass(frozen=True)
class Clon:
    ruta: Path
    rama: str
    rama_por_omision: str
    dependencias: tuple[str, ...]  # lo instalado, o lo declarado: "sin_lockfile:<x>", "pip_sin_wheel:<x>", ...
    espejo: Path


@dataclass(frozen=True)
class Accesos:
    """Las tres ACL de la misión, ligadas a la cuenta (inyectables en pruebas)."""
    paso: Callable[[Path], Awaitable[None]]        # <misión>/: solo atravesar
    escritura: Callable[[Path], Awaitable[None]]   # repo/: rwX recursivo
    lectura: Callable[[Path], Awaitable[None]]     # deps/: r-X recursivo


def accesos_de_la_cuenta(c: cuenta_axioma.Cuenta) -> Accesos:
    return Accesos(paso=lambda r: cuenta_axioma.dar_paso(c, r),
                   escritura=lambda r: cuenta_axioma.dar_acceso_recursivo(c, r),
                   lectura=lambda r: cuenta_axioma.dar_acceso_recursivo(c, r, permiso="r-X"))


async def instalar_real(argv: list[str], cwd: Path, env: dict[str, str]) -> tuple[int, bytes]:
    proc = await asyncio.create_subprocess_exec(*argv, cwd=str(cwd), env=env, stdin=asyncio.subprocess.DEVNULL,
                                                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    _, errores = await proc.communicate()
    return proc.returncode, errores


def parsear_autor(autor: str) -> tuple[str, str]:
    m = _AUTOR.fullmatch(autor) if isinstance(autor, str) else None
    if m is None or m.group("nombre").startswith("-"):
        raise ValueError(f"autor_invalido: se espera 'Nombre <correo>', llegó {autor!r}")
    return m.group("nombre"), m.group("correo")


async def rama_por_omision(cliente: httpx.AsyncClient, repo: str) -> str:
    r = await cliente.get(f"/repos/{repo}")
    r.raise_for_status()
    return r.json()["default_branch"]


async def _git(args: list[str], env: dict[str, str], error: str, token: str | None = None) -> bytes:
    try:
        return await correr_git(args, env=env, error=f"preparar_fallo: {error}", sanear_con=token)
    except GitFallo as exc:
        raise RuntimeError(str(exc)) from None


async def _traer_la_base(espejo: Path, base: str, token: str) -> None:
    with hogar_temporal() as home:
        await _git(["-C", str(espejo), "fetch", "--no-tags", "--no-recurse-submodules", "--no-auto-gc",
                    "--no-auto-maintenance", "--no-write-fetch-head", "--", "origin",
                    f"+refs/heads/{base}:refs/remotes/origin/{base}"],
                   entorno_red(home, token), "fetch del espejo", token)


async def _crear_espejo(espejo: Path, *, url: str, base: str, rama: str, token: str) -> None:
    with hogar_temporal() as home:
        env = entorno_base(home)
        await _git(["init", "-q", "--bare", "--", str(espejo)], env, "init del espejo")
        await asyncio.to_thread(espejo.chmod, 0o700)
        await _git(["-C", str(espejo), "symbolic-ref", "HEAD", f"refs/heads/{base}"], env, "HEAD del espejo")
        await _git(["-C", str(espejo), "remote", "add", "-t", base, "--", "origin", url], env, "origin del espejo")
        # El seguimiento de la rama de la misión existe SOLO para el lease del empuje; ningún
        # fetch lo nombra (todos llevan su refspec explícito).
        await _git(["-C", str(espejo), "config", "--add", "remote.origin.fetch",
                    f"+refs/heads/{rama}:refs/remotes/origin/{rama}"], env, "refspec del espejo")
    await _traer_la_base(espejo, base, token)
    with hogar_temporal() as home:
        env = entorno_base(home)
        for ref in (f"refs/heads/{base}", f"refs/heads/{rama}"):
            await _git(["-C", str(espejo), "update-ref", "--", ref, f"refs/remotes/origin/{base}"], env,
                       "ramas del espejo")


async def _escribir(ruta: Path, datos: bytes) -> None:
    def escribir():
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(ruta, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(datos)
    await asyncio.to_thread(escribir)


async def _instalar_dependencias(espejo: Path, deps: Path, base: str, instalar: Instalador
                                 ) -> tuple[list[str], list[tuple[str, Path]]]:
    """Devuelve (lo instalado o declarado, enlaces a crear en el clon: (ruta relativa, destino))."""
    hechas: list[str] = []
    enlaces: list[tuple[str, Path]] = []
    await asyncio.to_thread(deps.mkdir, mode=0o700)
    with hogar_temporal() as home:
        env_git = entorno_base(home)
        commit = (await _git(["-C", str(espejo), "rev-parse", "--verify", "--end-of-options",
                              f"refs/remotes/origin/{base}^{{commit}}"], env_git, "rama por omisión")).decode().strip()
        listado = await _git(["-C", str(espejo), "ls-tree", "-r", "-z", "--name-only", "--end-of-options", commit],
                             env_git, "archivos de la rama por omisión")
        archivos = {n for n in listado.decode("utf-8", errors="replace").split("\0") if n}

        async def copiar(ruta_repo: str, destino: Path) -> None:
            datos = await _git(["-C", str(espejo), "cat-file", "blob", f"{commit}:{ruta_repo}"], env_git,
                               "leer archivo de bloqueo")
            await _escribir(destino / PurePosixPath(ruta_repo).name, datos)

        # Instalaciones: NADA del entorno del proceso salvo PATH y LANG (ni /etc/jax/.env, ni
        # NPM_TOKEN, ni PIP_*), HOME propio y temporal.
        env = entorno_minimo(home)

        async def correr(argv: list[str], cwd: Path) -> tuple[int, bytes]:
            return await instalar(argv, cwd, env)

        # Python: un venv por directorio (raíz, backend/), todos sus requirements*.txt adentro.
        por_dir: dict[str, list[str]] = {}
        for n in sorted(a for a in archivos if _REQUIREMENTS.fullmatch(a)):
            por_dir.setdefault("" if "/" not in n else str(PurePosixPath(n).parent), []).append(n)
        for d, reqs in por_dir.items():
            destino = deps / "python" / (d or "_raiz")
            for n in reqs:
                await copiar(n, destino)
            venv = destino / "venv"
            rc, _ = await correr(["python3", "-m", "venv", str(venv)], destino)
            if rc != 0:
                raise RuntimeError(f"preparar_fallo: venv en {d or '.'} (rc={rc})")
            for n in reqs:
                rc, err = await correr([str(venv / "bin" / "pip"), "install", "--only-binary=:all:", "-r",
                                        str(destino / PurePosixPath(n).name)], destino)
                if rc == 0:
                    hechas.append(n)
                elif any(m in err for m in _SIN_WHEEL):
                    hechas.append(f"pip_sin_wheel:{n}")
                else:
                    raise RuntimeError(f"preparar_fallo: pip install -r {n} (rc={rc})")
            enlaces.append((f"{d}/.venv" if d else ".venv", venv))

        # Node: `npm ci` sin scripts, solo en la raíz y en frontend/.
        for p in sorted(a for a in archivos if PurePosixPath(a).name == "package.json"
                        and "node_modules" not in PurePosixPath(a).parts):
            d = "" if "/" not in p else str(PurePosixPath(p).parent)
            lock = f"{d}/package-lock.json" if d else "package-lock.json"
            if lock not in archivos:
                hechas.append(f"sin_lockfile:{p}")
                continue
            if d not in _DIRS_NPM:
                hechas.append(f"npm_no_instalado:{p}")
                continue
            destino = deps / "node" / (d or "_raiz")
            await copiar(p, destino)
            await copiar(lock, destino)
            rc, _ = await correr(["npm", "ci", "--ignore-scripts", "--no-audit", "--no-fund"], destino)
            if rc != 0:
                raise RuntimeError(f"preparar_fallo: npm ci en {d or '.'} (rc={rc})")
            hechas.append(lock)
            enlaces.append((f"{d}/node_modules" if d else "node_modules", destino / "node_modules"))

        # PHP: `composer install` sin scripts ni plugins, en la raíz.
        if "composer.json" in archivos and "composer.lock" not in archivos:
            hechas.append("sin_lockfile:composer.json")
        elif "composer.lock" in archivos and "composer.json" in archivos:
            destino = deps / "php" / "_raiz"
            await copiar("composer.json", destino)
            await copiar("composer.lock", destino)
            rc, _ = await correr(["composer", "install", "--no-interaction", "--no-scripts", "--no-plugins"],
                                 destino)
            if rc != 0:
                raise RuntimeError(f"preparar_fallo: composer install (rc={rc})")
            hechas.append("composer.lock")
            enlaces.append(("vendor", destino / "vendor"))
    return hechas, enlaces


async def _crear_clon(espejo: Path, ruta: Path, *, rama: str, nombre: str, correo: str,
                      enlaces: list[tuple[str, Path]]) -> list[str]:
    """Clon recién hecho por nosotros: todavía es de confianza. Después de esto, jaxsvc no
    vuelve a ejecutar git adentro."""
    notas: list[str] = []
    with hogar_temporal() as home:
        env = entorno_base(home)
        await _git(["clone", "-q", "--no-hardlinks", "--branch", rama, "--", str(espejo), str(ruta)], env,
                   "clon de trabajo")
        config = str(ruta / ".git" / "config")
        await _git(["config", "--file", config, "--", "user.name", nombre], env, "autor del clon")
        await _git(["config", "--file", config, "--", "user.email", correo], env, "autor del clon")

    def enlazar():
        excluir = ruta / ".git" / "info" / "exclude"
        excluir.parent.mkdir(parents=True, exist_ok=True)
        with excluir.open("a") as f:
            # Un enlace no es un directorio para git: `node_modules/` en .gitignore no lo tapa.
            for rel, _ in enlaces:
                f.write(f"/{rel}\n")
        for rel, destino in enlaces:
            enlace = ruta / rel
            if os.path.lexists(enlace) or not enlace.parent.is_dir() or enlace.parent.is_symlink():
                notas.append(f"enlace_omitido:{rel}")
                continue
            os.symlink(destino, enlace)
    await asyncio.to_thread(enlazar)
    return notas


async def preparar(repo: Repo, *, mision_id: str, raiz: Path, rama_por_omision: str, token: str, autor: str,
                   accesos: Accesos, instalar: Instalador = instalar_real) -> Clon:
    # Todo se valida ANTES de crear un directorio o correr un comando.
    rama = rama_de_la_mision(mision_id)
    url = url_del_repo(repo.owner_repo)
    base = validar_rama_base(rama_por_omision)
    if base == rama:
        raise ValueError(f"rama_por_omision_invalida: {base!r} es la rama de la misión")
    nombre, correo = parsear_autor(autor)
    dir_mision = raiz / mision_id
    espejo, ruta, deps = dir_mision / "espejo.git", dir_mision / "repo", dir_mision / "deps"
    validar_ruta_segura(ruta)
    marca = dir_mision / "preparado.json"

    if await asyncio.to_thread(marca.is_file):
        estado = json.loads(await asyncio.to_thread(marca.read_text))
        await _traer_la_base(espejo, base, token)
        return Clon(ruta, rama, base, tuple(estado["dependencias"]), espejo)

    await asyncio.to_thread(dir_mision.mkdir, mode=0o700, parents=True, exist_ok=True)
    await accesos.paso(dir_mision)
    for resto in (espejo, ruta, deps):
        if await asyncio.to_thread(os.path.lexists, resto):
            await asyncio.to_thread(shutil.rmtree, resto)
    await _crear_espejo(espejo, url=url, base=base, rama=rama, token=token)
    hechas, enlaces = await _instalar_dependencias(espejo, deps, base, instalar)
    await accesos.lectura(deps)
    hechas += await _crear_clon(espejo, ruta, rama=rama, nombre=nombre, correo=correo, enlaces=enlaces)
    await accesos.escritura(ruta)
    dependencias = tuple(hechas) or ("sin_lockfile",)
    await _escribir(marca, json.dumps({"rama_por_omision": base, "dependencias": list(dependencias)}).encode())
    return Clon(ruta, rama, base, dependencias, espejo)
