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
import stat
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import httpx

from jax.ejecutor.codigo.entrega import (rama_de_la_mision, url_del_repo, validar_owner_repo, validar_rama_base,
                                         validar_ruta_segura)
from jax.ejecutor.codigo.sandbox import argv_sandbox
from jax.ejecutor.codigo.git_token import (GitFallo, correr_git, entorno_base, entorno_minimo, entorno_red,
                                           hogar_temporal)
from jax.ejecutor.contratos import cuenta_axioma

_AUTOR = re.compile(r"(?P<nombre>[^<>\x00-\x1f\x7f]*[^<>\s\x00-\x1f\x7f])\s<(?P<correo>[^<>\s@]+@[^<>\s@]+)>")
_REQUIREMENTS = re.compile(r"(?:backend/)?requirements[^/]*\.txt")
_SIN_WHEEL = (b"No matching distribution found", b"Could not find a version that satisfies")
_DIRS_NPM = ("", "frontend")

# El ejecutor de instalaciones recibe el argv COMPLETO de bwrap (sandbox.argv_sandbox).
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
    """`repo` se valida antes de armar la ruta de la API (MINOR-4); lo que responde GitHub se
    valida como nombre de rama antes de que llegue a un refspec."""
    r = await cliente.get(f"/repos/{validar_owner_repo(repo)}")
    r.raise_for_status()
    return validar_rama_base(r.json()["default_branch"])


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


async def _instalar_dependencias(espejo: Path, deps: Path, base: str, instalar: Instalador,
                                 node_bin: Path | None) -> tuple[list[str], list[tuple[str, Path]]]:
    """Devuelve (lo instalado o declarado, enlaces a crear en el clon: (ruta relativa, destino)).

    Cada instalación corre en `sandbox.argv_sandbox`: el único escribible es `deps/`, y los
    archivos de dependencias entran como COPIA de solo lectura (hecha aquí, fuera de `deps/`, en
    un temporal que se borra al terminar)."""
    hechas: list[str] = []
    enlaces: list[tuple[str, Path]] = []
    await asyncio.to_thread(deps.mkdir, mode=0o700)
    with hogar_temporal() as home:
        env_git = entorno_base(home)
        fuentes = home / "fuentes"
        commit = (await _git(["-C", str(espejo), "rev-parse", "--verify", "--end-of-options",
                              f"refs/remotes/origin/{base}^{{commit}}"], env_git, "rama por omisión")).decode().strip()
        listado = await _git(["-C", str(espejo), "ls-tree", "-r", "-z", "--name-only", "--end-of-options", commit],
                             env_git, "archivos de la rama por omisión")
        archivos = {n for n in listado.decode("utf-8", errors="replace").split("\0") if n}

        async def copiar(ruta_repo: str, carpeta: Path) -> Path:
            datos = await _git(["-C", str(espejo), "cat-file", "blob", f"{commit}:{ruta_repo}"], env_git,
                               "leer archivo de dependencias")
            copia = carpeta / PurePosixPath(ruta_repo).name
            await _escribir(copia, datos)
            return copia

        # El proceso bwrap no hereda más que PATH/LANG/HOME, y adentro --clearenv.
        env = entorno_minimo(home)

        async def correr(comando: list[str], destino: Path, solo_lectura) -> tuple[int, bytes]:
            # MINOR-C: una instalación anterior pudo dejar enlaces dentro de deps/.
            await asyncio.to_thread(_crear_bajo, deps, destino)
            argv = argv_sandbox(comando, deps=deps, cwd=destino, solo_lectura=solo_lectura, node_bin=node_bin)
            return await instalar(argv, destino, env)

        # Python: UNA invocación de pip por venv (raíz, backend/), con sus requirements*.txt
        # aceptados juntos. Los que traen opciones o requisitos que no son del índice no se
        # instalan: se declaran.
        por_dir: dict[str, list[str]] = {}
        for n in sorted(a for a in archivos if _REQUIREMENTS.fullmatch(a)):
            por_dir.setdefault("" if "/" not in n else str(PurePosixPath(n).parent), []).append(n)
        for d, reqs in por_dir.items():
            carpeta = fuentes / "python" / (d or "_raiz")
            copias = {n: await copiar(n, carpeta) for n in reqs}
            textos = {PurePosixPath(n).name: (await asyncio.to_thread(c.read_text, errors="replace"))
                      for n, c in copias.items()}
            rechazados = _requirements_rechazados(textos)
            aceptados = [n for n in reqs if PurePosixPath(n).name not in rechazados]
            hechas += [f"pip_opcion_rechazada:{n}" for n in reqs if PurePosixPath(n).name in rechazados]
            if not aceptados:
                continue
            destino = deps / "python" / (d or "_raiz")
            venv = destino / "venv"
            rc, _ = await correr(["python3", "-m", "venv", str(venv)], destino, [])
            if rc != 0:
                raise RuntimeError(f"preparar_fallo: venv en {d or '.'} (rc={rc})")
            pip = [str(venv / "bin" / "pip"), "install", "--only-binary=:all:"]
            for n in aceptados:
                pip += ["-r", str(copias[n])]
            rc, err = await correr(pip, destino, [(carpeta, carpeta)])
            if rc == 0:
                hechas += aceptados
            elif any(m in err for m in _SIN_WHEEL):
                hechas += [f"pip_sin_wheel:{n}" for n in aceptados]
            else:
                raise RuntimeError(f"preparar_fallo: pip install en {d or '.'} (rc={rc})")
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
            carpeta = fuentes / "node" / (d or "_raiz")
            destino = deps / "node" / (d or "_raiz")
            montajes = [(await copiar(f, carpeta), destino / PurePosixPath(f).name) for f in (p, lock)]
            # BLOCK-A: npm prepara las dependencias git aunque lleve --ignore-scripts. Se instala
            # solo si TODO el lockfile viene del registro con integridad.
            malos = _npm_rechazos(await asyncio.to_thread(montajes[1][0].read_text, errors="replace"))
            if malos:
                hechas += [f"npm_fuente_rechazada:{lock}:{m}" for m in malos]
                continue
            # --allow-git=none (npm >= 11; verificado en 11.13 del node de producción): con un
            # package.json que declare una dependencia git ausente del lockfile, npm ci bajaba el
            # tarball antes de fallar. Ahora ni lo baja.
            rc, _ = await correr(["npm", "ci", "--ignore-scripts", "--allow-git=none", "--no-audit", "--no-fund"],
                                 destino, montajes)
            if rc != 0:
                raise RuntimeError(f"preparar_fallo: npm ci en {d or '.'} (rc={rc})")
            hechas.append(lock)
            enlaces.append((f"{d}/node_modules" if d else "node_modules", destino / "node_modules"))

        # PHP: `composer install` sin scripts ni plugins, en la raíz.
        if "composer.json" in archivos and "composer.lock" not in archivos:
            hechas.append("sin_lockfile:composer.json")
        elif "composer.lock" in archivos and "composer.json" in archivos:
            carpeta = fuentes / "php" / "_raiz"
            destino = deps / "php" / "_raiz"
            montajes = [(await copiar(f, carpeta), destino / f) for f in ("composer.json", "composer.lock")]
            textos = [await asyncio.to_thread(c.read_text, errors="replace") for c, _ in montajes]
            malos = _composer_rechazos(*textos)
            if malos:
                hechas += [f"composer_fuente_rechazada:{m}" for m in malos]
            else:
                rc, _ = await correr(["composer", "install", "--no-interaction", "--no-scripts", "--no-plugins",
                                      "--prefer-dist"], destino, montajes)
                if rc != 0:
                    raise RuntimeError(f"preparar_fallo: composer install (rc={rc})")
                hechas.append("composer.lock")
                enlaces.append(("vendor", destino / "vendor"))
    return hechas, enlaces


def _crear_bajo(deps: Path, ruta: Path) -> None:
    """`mkdir -p` dentro de `deps/` que NO sigue enlaces: cada componente se mira con `lstat`;
    un enlace o algo que no es directorio → falla cerrado. Quien pudo dejar un enlace ahí es una
    instalación anterior (escribe en deps/ desde el sandbox), y un `mkdir` que lo siguiera
    crearía -- y luego montaría escribible -- un directorio fuera de deps/."""
    try:
        relativa = ruta.relative_to(deps)
    except ValueError:
        raise RuntimeError(f"preparar_fallo: fuera_de_deps: {ruta}") from None
    actual = deps
    for parte in relativa.parts:
        actual = actual / parte
        try:
            info = os.lstat(actual)
        except FileNotFoundError:
            os.mkdir(actual, 0o700)
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"preparar_fallo: enlace_en_deps: {actual}")


_REGISTRO_NPM = "https://registry.npmjs.org/"


def _npm_mala(e) -> bool:
    if not isinstance(e, dict) or e.get("link"):
        return True
    version = str(e.get("version", ""))
    resolved = e.get("resolved")
    # Una versión del registro es semver: `git+…`, `github:…`, `file:…`, `http(s):…`, `npm:alias`
    # o una ruta llevan ':' o '/'.
    if ":" in version or "/" in version:
        return True
    return not (isinstance(resolved, str) and resolved.startswith(_REGISTRO_NPM) and e.get("integrity"))


def _npm_rechazos(texto: str) -> list[str]:
    """Paquetes del package-lock.json que NO vienen del registro con integridad (v2/v3:
    `packages`; v1: `dependencies`, recursivo; si están los dos, se miran los dos)."""
    try:
        datos = json.loads(texto)
    except ValueError:
        return ["<ilegible>"]
    if not isinstance(datos, dict):
        return ["<ilegible>"]
    malos: set[str] = set()
    paquetes = datos.get("packages")
    if paquetes is not None:
        if not isinstance(paquetes, dict):
            return ["<ilegible>"]
        for clave, entrada in paquetes.items():
            if clave != "" and _npm_mala(entrada):
                malos.add(clave.rsplit("node_modules/", 1)[-1])

    def recorrer(deps) -> None:
        if not isinstance(deps, dict):
            malos.add("<ilegible>")
            return
        for nombre, entrada in deps.items():
            if _npm_mala(entrada):
                malos.add(nombre)
            if isinstance(entrada, dict) and "dependencies" in entrada:
                recorrer(entrada["dependencies"])
    if "dependencies" in datos:
        recorrer(datos["dependencies"])
    return sorted(malos)


_DIST_COMPOSER = ("https://api.github.com/repos/", "https://codeload.github.com/", "https://repo.packagist.org/")


def _composer_rechazos(composer_json: str, composer_lock: str) -> list[str]:
    """`composer.json:repositories` si declara repositorios que no son packagist (vcs, path, git,
    artifact, package…); `composer.lock:<paquete>` si su `dist` no es un zip de packagist/GitHub o
    su `source` no es git de github.com."""
    try:
        cj, cl = json.loads(composer_json), json.loads(composer_lock)
    except ValueError:
        return ["composer:<ilegible>"]
    if not isinstance(cj, dict) or not isinstance(cl, dict):
        return ["composer:<ilegible>"]
    malos: list[str] = []
    repos = cj.get("repositories")
    lista = repos.values() if isinstance(repos, dict) else (repos or [])
    for r in lista:
        if r is False or (isinstance(r, dict) and set(r) == {"packagist.org"} and r["packagist.org"] is False):
            continue  # apagar packagist no agrega fuentes
        if not (isinstance(r, dict) and r.get("type") == "composer"
                and str(r.get("url", "")).startswith("https://repo.packagist.org")):
            malos.append("composer.json:repositories")
            break
    for p in list(cl.get("packages") or []) + list(cl.get("packages-dev") or []):
        nombre = p.get("name", "<sin nombre>") if isinstance(p, dict) else "<ilegible>"
        dist = p.get("dist") if isinstance(p, dict) else None
        source = p.get("source") if isinstance(p, dict) else None
        dist_ok = (isinstance(dist, dict) and dist.get("type") == "zip"
                   and str(dist.get("url", "")).startswith(_DIST_COMPOSER))
        source_ok = source is None or (isinstance(source, dict) and source.get("type") == "git"
                                       and str(source.get("url", "")).startswith("https://github.com/"))
        if not (dist_ok and source_ok):
            malos.append(f"composer.lock:{nombre}")
    return malos


_COMENTARIO = re.compile(r"(^|\s)#.*$")
_INCLUIR = re.compile(r"(?:-r|--requirement)(?:\s+|=)(\S+)")


def _requirements_rechazados(textos: dict[str, str]) -> set[str]:
    """Nombres (del mismo directorio) que NO se instalan. Se rechaza un archivo con cualquier
    línea de opción (`--no-binary`, `--index-url`, `-f`, `-e`, `-c`, `--trusted-host`, `-i`…)
    salvo `-r <hermano>`, o con un requisito que no sale del índice (URL, `pkg @ …`, ruta local:
    pip los construye aunque diga `--only-binary`). Y el que incluye (`-r`) a uno rechazado."""
    incluye: dict[str, set[str]] = {}
    rechazados: set[str] = set()
    for nombre, texto in textos.items():
        incluye[nombre] = set()
        for linea in texto.replace("\\\n", " ").splitlines():
            linea = _COMENTARIO.sub("", linea).strip()
            if not linea:
                continue
            if linea.startswith("-"):
                m = _INCLUIR.fullmatch(linea)
                if m and m.group(1) in textos:
                    incluye[nombre].add(m.group(1))
                else:
                    rechazados.add(nombre)
            elif "://" in linea or "@" in linea.split(";", 1)[0] or linea.startswith((".", "/", "~")):
                rechazados.add(nombre)
    cambio = True
    while cambio:
        cambio = False
        for nombre, hijos in incluye.items():
            if nombre not in rechazados and hijos & rechazados:
                rechazados.add(nombre)
                cambio = True
    return rechazados


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
                   accesos: Accesos, instalar: Instalador = instalar_real, node_bin: Path | None = None) -> Clon:
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
    hechas, enlaces = await _instalar_dependencias(espejo, deps, base, instalar, node_bin)
    await accesos.lectura(deps)
    hechas += await _crear_clon(espejo, ruta, rama=rama, nombre=nombre, correo=correo, enlaces=enlaces)
    await accesos.escritura(ruta)
    dependencias = tuple(hechas) or ("sin_lockfile",)
    await _escribir(marca, json.dumps({"rama_por_omision": base, "dependencias": list(dependencias)}).encode())
    return Clon(ruta, rama, base, dependencias, espejo)
