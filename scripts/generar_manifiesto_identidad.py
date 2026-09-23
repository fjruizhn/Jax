#!/usr/bin/env python3
r"""Genera el manifiesto de identidad de implementación que jax#260 exige.

POR QUÉ EXISTE (medido 2026-09-22): `las_manos/server.py::_configure_b7_trusted_runtime`
construye `TrustedImplementationIdentityProvider`, cuyo `.load()` lee
`/etc/jax/build/implementation-identity.json` y lo verifica con
`verify_build_manifest(..., repository_root="/srv/jax")`
(`policy/enforcement_evidence/implementation_identity.py`). Ese archivo no
existe en ningún lado del árbol -- `docs/operations/trusted-files.md` lo
listaba fail-closed sin decir quién lo genera -- y LAS MANOS no arranca
(`FileNotFoundError`), dejando producción parada en el commit anterior a
jax#260.

QUÉ PRODUCE, dos artefactos:
  1. el JSON de identidad (`--output`): las 10 claves EXACTAS que
     `implementation_identity_from_projection` acepta -- ni una de más ni
     una de menos.
  2. el blob del "build manifest" (`--manifest-output`): el `{schema_version,
     kind, files}` cuyo hash es `build_manifest_blob_hash` en (1). Sus bytes
     tienen que estar accesibles como `store.get_evidence_blob(...)` en la
     base de evidencia (`jax_evidence.evidence_blobs`, MariaDB) para que
     `verify_build_manifest` pueda leerlo en producción -- este script NO lo
     inserta ahí: requeriría las credenciales de producción que este script
     deliberadamente no toca (barrera de producción: nunca DB fuera de
     pytest). Ese paso vive aparte, en el runbook de despliegue
     (`docs/runbooks/deployment-rollback.md`).

ALGORITMO DE HASH: NO se reinventa. Cada archivo de
`_V1_REQUIRED_SOURCE_PATHS` (la MISMA constante que usa
`verify_build_manifest_bytes`, importada de ahí -- no copiada a mano) se
hashea con `policy.enforcement_evidence.ids.sha256_bytes`, que es
byte-por-byte el mismo cálculo que hace el verificador
(`"sha256:"+hashlib.sha256(path.read_bytes()).hexdigest()`). El conjunto de
`files` del manifiesto es EXACTAMENTE ese conjunto requerido, ni más ni
menos: agregar archivos de más ensancha en silencio lo que puede invalidar
el manifiesto sin que nadie lo haya decidido.

ÁRBOL SUCIO: un manifiesto de producción no puede describir bytes que no
están en un commit -- por default, si `git status --porcelain` no está
vacío, el script se niega (exit 1, no escribe nada). `--allow-dirty` es un
escape explícito para iteración local: genera igual, pero con
`source_state: "DIRTY"`, que ya de por sí hace STALE cualquier claim B7 que
dependa de esta identidad (`status_engine.derive_assertion`). Nunca hay un
tercer estado donde un árbol sucio produzca `CLEAN`.

CAMPOS SIN CONSUMIDOR HOY: `schema_versions` y `build_id` no los lee ningún
código de este repo (verificado con
`grep -rn "\.schema_versions\|\.build_id" --include=*.py .` -- el único hit
es la propia `projection()` de `ImplementationIdentity`). Inventar un valor
para un campo que nadie va a verificar sería exactamente el tipo de dato
fabricado que el Principio VIII prohíbe. Este script escribe los mismos
default del dataclass (`schema_versions: []`, `build_id: null`) y lo deja
dicho acá: el día que algo empiece a exigirlos, se decide su contenido real
en ese momento, con ese consumidor delante.

USO (hall9000, paso de despliegue -- ver el runbook para el resto del
procedimiento):
    python3 scripts/generar_manifiesto_identidad.py \
        --repo-root /srv/jax-prod/jax \
        --output /etc/jax/build/implementation-identity.json \
        --manifest-output /etc/jax/build/implementation-identity.manifest.json

Nunca escribe en `/etc/jax/` desde una sesión de desarrollo: sólo el
controlador de despliegue, corriendo con las credenciales del deploy,
ejecuta este comando contra las rutas reales de `/etc/jax/`.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_THIS_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_THIS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_THIS_REPO_ROOT))

from policy.enforcement_evidence.ids import sha256_bytes  # noqa: E402
from policy.enforcement_evidence.implementation_identity import (  # noqa: E402
    _V1_REQUIRED_SOURCE_PATHS,
)

_DEFAULT_OUTPUT = Path("/etc/jax/build/implementation-identity.json")


class ArbolSucioError(RuntimeError):
    """Un árbol con cambios sin commitear no puede convertirse en un
    manifiesto de producción CLEAN."""


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} falló en {repo_root}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def repository_id_desde_remoto(repo_root: Path) -> str:
    """`owner/repo`, la misma forma que `GITHUB_REPOSITORY` en CI (ver
    `.github/workflows/policy.yml` y `policy/enforcement_evidence/test_evidence.py`).
    Soporta remotos SSH (`git@github.com:owner/repo.git`) y HTTPS."""
    url = _git(repo_root, "remote", "get-url", "origin")
    if url.startswith("git@"):
        tail = url.split(":", 1)[-1]
    elif "github.com/" in url:
        tail = url.split("github.com/", 1)[-1]
    else:
        tail = url
    return tail[:-4] if tail.endswith(".git") else tail


def arbol_sucio(repo_root: Path) -> bool:
    return bool(_git(repo_root, "status", "--porcelain"))


def construir_manifiesto_de_build(repo_root: Path) -> bytes:
    """El `{schema_version, kind, files}` que `verify_build_manifest_bytes`
    exige -- mismo conjunto de claves, mismo algoritmo de hash, ni un
    archivo de más."""
    root = repo_root.resolve()
    files: dict[str, str] = {}
    for rel in sorted(_V1_REQUIRED_SOURCE_PATHS):
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"fuente requerida ausente en {root}: {rel}")
        files[rel] = sha256_bytes(path.read_bytes())
    payload = {"schema_version": "1.0", "kind": "JAX_BUILD_MANIFEST", "files": files}
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def construir_identidad(
    repo_root: Path, *, repository_id: str, allow_dirty: bool,
) -> tuple[dict, bytes]:
    sucio = arbol_sucio(repo_root)
    if sucio and not allow_dirty:
        raise ArbolSucioError(
            "el árbol tiene cambios sin commitear -- un manifiesto de "
            "producción no puede describir bytes que no están en un "
            "commit. Commiteá los cambios, o pasá --allow-dirty explícito "
            "si esto es intencional (el resultado queda marcado DIRTY, "
            "nunca CLEAN)."
        )
    manifest_bytes = construir_manifiesto_de_build(repo_root)
    identity = {
        "schema_version": "1.0",
        "kind": "JAX_IMPLEMENTATION_IDENTITY",
        "repository_id": repository_id,
        "source_revision_kind": "GIT",
        "git_commit_sha": _git(repo_root, "rev-parse", "HEAD"),
        "git_tree_id": _git(repo_root, "rev-parse", "HEAD^{tree}"),
        "source_state": "DIRTY" if sucio else "CLEAN",
        "build_manifest_blob_hash": sha256_bytes(manifest_bytes),
        # Ver docstring del módulo: sin consumidor hoy, no se inventa valor.
        "schema_versions": [],
        "build_id": None,
    }
    return identity, manifest_bytes


def escribir_atomico(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    """Nunca deja un archivo parcial: escribe a un temporal en el mismo
    directorio, fsync, chmod, y `os.replace` (atómico en el mismo
    filesystem)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:  # fail-soft: best-effort cleanup del temporal; el raise de abajo ya reporta el error real
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Genera implementation-identity.json + su build manifest (jax#260).",
    )
    parser.add_argument("--repo-root", type=Path, required=True,
                         help="raíz del checkout git a describir (p.ej. /srv/jax-prod/jax)")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT,
                         help=f"ruta del JSON de identidad (default: {_DEFAULT_OUTPUT})")
    parser.add_argument("--manifest-output", type=Path, default=None,
                         help="ruta del blob del build manifest a instalar en "
                              "jax_evidence.evidence_blobs (default: "
                              "<output>.manifest.json)")
    parser.add_argument("--repository-id", default=None,
                         help="por defecto se deriva de `git remote get-url origin`")
    parser.add_argument("--allow-dirty", action="store_true",
                         help="genera igual con un árbol sucio, marcado source_state=DIRTY")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    manifest_output = args.manifest_output or args.output.with_name(
        args.output.name + ".manifest.json"
    )

    try:
        repository_id = args.repository_id or repository_id_desde_remoto(repo_root)
        identity, manifest_bytes = construir_identidad(
            repo_root, repository_id=repository_id, allow_dirty=args.allow_dirty,
        )
    except (ArbolSucioError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    escribir_atomico(manifest_output, manifest_bytes)
    identity_bytes = (json.dumps(identity, indent=2, sort_keys=True) + "\n").encode("utf-8")
    escribir_atomico(args.output, identity_bytes)

    print(f"identidad escrita en {args.output} (source_state={identity['source_state']})")
    print(f"build manifest escrito en {manifest_output}")
    print(
        "Falta instalar el build manifest en jax_evidence.evidence_blobs -- "
        "ver docs/runbooks/deployment-rollback.md"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
