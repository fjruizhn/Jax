"""Lectura de OBJETOS git del repo de skills del ecosistema: el unico lugar del Faro que lanza `git`.

Solo lectura y siempre contra un SHA (`ls-tree`, `cat-file`, `merge-base`, `rev-parse`): jamas el
arbol de trabajo del checkout. Va aparte de `paquete.py` para que el lanzamiento de subprocesos
quede en un modulo chico y revisable.

Mismo modelo de amenaza que el ensamblado de `lib/assemble.py` del repo de skills: ninguna
invocacion de git ejecuta hooks ni fsmonitor del repo que se esta leyendo.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

_GIT_SIN_HOOKS = ("-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false")


class FuenteInvalida(RuntimeError):
    """Lo que se quiere empaquetar no se puede empaquetar con garantias (SHA que no es de
    `origin/main`, symlink, plugin ausente...). La construccion no publica nada."""


def git(repo: Path, *args: str, entrada: bytes | None = None, aceptar: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess:
    try:
        r = subprocess.run(["git", *_GIT_SIN_HOOKS, "-C", str(repo), *args], input=entrada,
                           capture_output=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise FuenteInvalida(f"no se pudo ejecutar git sobre {repo}: {type(exc).__name__}") from exc
    if r.returncode not in aceptar:
        raise FuenteInvalida(f"git {args[0]} fallo (rc={r.returncode}): {r.stderr.decode(errors='replace')[:200].strip()}")
    return r


def exigir_sha_ancestro_de(repo: Path, sha: str, ref: str) -> None:
    """El commit existe y es ancestro de `ref` (en la practica `origin/main`). Si no se puede
    afirmar, se niega (fallo cerrado)."""
    r = git(repo, "cat-file", "-e", f"{sha}^{{commit}}", aceptar=(0, 1, 128))
    if r.returncode != 0:
        raise FuenteInvalida(f"el commit {sha} no existe en {repo}")
    r = git(repo, "merge-base", "--is-ancestor", sha, ref, aceptar=(0, 1, 128))
    if r.returncode != 0:
        # 1 = no es ancestro; 128 = la ref no existe.
        raise FuenteInvalida(f"el commit {sha} no es ancestro de {ref}: solo se fija un SHA de origin/main")


def resolver_ref(repo: Path, ref: str) -> str:
    """El SHA al que apunta `ref`, o "" si no se puede leer. No lanza."""
    try:
        r = git(repo, "rev-parse", "--verify", "-q", f"{ref}^{{commit}}", aceptar=(0, 1, 128))
    except FuenteInvalida:
        return ""
    return r.stdout.decode().strip() if r.returncode == 0 else ""


@dataclass(frozen=True)
class EntradaGit:
    modo: str
    oid: str
    ruta: str


def listar(repo: Path, sha: str, *rutas: str) -> list[EntradaGit]:
    r = git(repo, "ls-tree", "-r", "-z", "--full-tree", sha, "--", *rutas)
    entradas = []
    for crudo in r.stdout.split(b"\0"):
        if not crudo:
            continue
        meta, _, ruta = crudo.partition(b"\t")
        modo, _tipo, oid = meta.decode().split(" ")
        entradas.append(EntradaGit(modo, oid, ruta.decode("utf-8")))
    return entradas


def leer_blobs(repo: Path, oids: list[str]) -> dict[str, bytes]:
    unicos = list(dict.fromkeys(oids))
    if not unicos:
        return {}
    r = git(repo, "cat-file", "--batch", entrada=("\n".join(unicos) + "\n").encode())
    salida, i, blobs = r.stdout, 0, {}
    for oid in unicos:
        fin = salida.index(b"\n", i)
        cabecera = salida[i:fin].decode().split(" ")
        if len(cabecera) != 3 or cabecera[0] != oid or cabecera[1] != "blob":
            raise FuenteInvalida(f"git cat-file devolvio algo inesperado para {oid}: {cabecera}")
        tam = int(cabecera[2])
        blobs[oid] = salida[fin + 1:fin + 1 + tam]
        i = fin + 1 + tam + 1
    return blobs
