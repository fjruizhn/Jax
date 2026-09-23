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

Este script es SÓLO uno de los pasos del arranque -- el procedimiento
COMPLETO (variables de entorno, esquemas de MariaDB, permisos, el símlink
`/srv/jax`, y cómo confirmar que la identidad instalada no quedó STALE) está
en `docs/runbooks/implementation-identity.md`. Léelo antes de correr esto
contra producción.

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
     pytest). Ese paso vive aparte, en el runbook
     (`docs/runbooks/implementation-identity.md`).

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
`source_state: "DIRTY"`.

**ADVERTENCIA sobre `--allow-dirty` (ronda de revisión #264):**
`verify_build_manifest`/`verify_build_manifest_bytes` NUNCA leen
`source_state` -- sólo comparan hashes de archivo. Un manifiesto DIRTY
instalado en `/etc/jax/build/` arranca LAS MANOS igual de bien que uno
CLEAN, y `systemctl is-active` + `/health` 200 (la verificación que trae
este mismo runbook) también da verde igual. Lo único que nota la
diferencia es `status_engine.derive_assertion`, que hace STALE cualquier
claim B7 cuya identidad no sea CLEAN -- en silencio, sin que el arranque ni
el health-check lo digan. **Nunca instalar un artefacto DIRTY en
`/etc/jax/build/implementation-identity.json`** -- `--allow-dirty` es para
inspeccionar el manifiesto en un `--output` temporal, no para desplegar.
Ver el paso 7 del runbook para el chequeo que sí lo detecta.

CAMPOS SIN CONSUMIDOR HOY: `schema_versions` y `build_id` no los lee ningún
código de este repo (verificado con
`grep -rn "\.schema_versions\|\.build_id" --include=*.py .` -- el único hit
es la propia `projection()` de `ImplementationIdentity`). Inventar un valor
para un campo que nadie va a verificar sería exactamente el tipo de dato
fabricado que el Principio VIII prohíbe. Este script escribe los mismos
default del dataclass (`schema_versions: []`, `build_id: null`) y lo deja
dicho acá: el día que algo empiece a exigirlos, se decide su contenido real
en ese momento, con ese consumidor delante.

`repository_id` SIN CREDENCIALES: se deriva de `git remote get-url origin`.
Un remoto HTTPS con token embebido
(`https://x-access-token:ghp_...@github.com/o/r.git`) NUNCA debe terminar
dentro de `repository_id` -- esa cadena se inserta tal cual en
`jax_evidence.test_evidence_manifests.repository_id` y en la identidad
misma, ambas con triggers que prohíben UPDATE/DELETE: un token filtrado ahí
queda filtrado para siempre, sin forma de corregirlo salvo DROP+recrear la
fila (imposible, la tabla es inmutable) o rotar el token. Por eso la
extracción usa `urllib.parse.urlsplit(...).path`, que nunca incluye
`.netloc` (donde viven usuario/contraseña/token en una URL) -- ver
`repository_id_desde_remoto`.

`--output` NO TIENE DEFAULT: pasar `--output` explícito, o `--production`
para usar la ruta fija `/etc/jax/build/implementation-identity.json`. Un
default silencioso apuntando a esa ruta real habría hecho que cualquier
corrida de prueba sin `--output` intentara escribir ahí por accidente.

USO (hall9000, paso de despliegue -- ver
`docs/runbooks/implementation-identity.md` para el resto del
procedimiento):
    python3 scripts/generar_manifiesto_identidad.py \
        --repo-root /srv/jax-prod/jax --production

Nunca escribe en `/etc/jax/` desde una sesión de desarrollo: sólo el
controlador de despliegue, corriendo con las credenciales del deploy,
ejecuta este comando con `--production` contra las rutas reales.

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
from urllib.parse import urlsplit

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


class EstadoDelArbolIndeterminadoError(RuntimeError):
    """`git status` escribió en stderr aunque salió con código 0 -- no se
    puede confiar en que el stdout liste TODOS los archivos sin commitear
    (visto en producción: `dubious ownership` u otro `Permission denied`
    sobre archivos puntuales del checkout, con git igual devolviendo 0)."""


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

    Nunca devuelve credenciales. Remotos `https://user:token@host/o/r.git`:
    `urlsplit(...)` deja usuario/contraseña/token en `.netloc`, jamás en
    `.path` -- sólo `.path` se usa. Remotos SSH tipo scp
    (`git@github.com:owner/repo.git`) no llevan credencial en la URL
    (autenticación por llave), pero se les aplica la misma limpieza por si
    alguna vez hay un `usuario@` real ahí delante."""
    url = _git(repo_root, "remote", "get-url", "origin")
    return _repository_id_desde_url(url)


def _repository_id_desde_url(url: str) -> str:
    if "://" in url:
        # http(s)://[user[:pass]@]host/owner/repo(.git) -- las credenciales,
        # si las hay, quedan en .netloc y NUNCA llegan a .path.
        tail = urlsplit(url).path.lstrip("/")
    elif "@" in url and ":" in url:
        # scp-like: [user@]host:owner/repo(.git)
        tail = url.split(":", 1)[-1]
    else:
        tail = url
    # Defensa en profundidad: si por lo que sea quedó un "user@" colgando en
    # el tramo final (formato no contemplado arriba), se descarta también.
    tail = tail.rsplit("@", 1)[-1]
    return tail[:-4] if tail.endswith(".git") else tail


def arbol_sucio(repo_root: Path) -> bool:
    """`git status --porcelain`, pero fail-closed de verdad: un chequeo que
    ignora advertencias de stderr con exit 0 (dubious ownership sorteado a
    medias, Permission denied sobre un archivo puntual, etc.) puede leer
    CLEAN un árbol que en realidad no pudo inspeccionar completo."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git status falló en {repo_root}: {result.stderr.strip()}"
        )
    if result.stderr.strip():
        raise EstadoDelArbolIndeterminadoError(
            "git status escribió en stderr aunque su código de salida fue "
            "0 -- no se puede confiar en que el árbol esté limpio sin leer "
            "esto primero (¿'dubious ownership'? ¿permisos? corré este "
            "script como el dueño del checkout, o solucioná lo que sigue "
            f"antes de continuar):\n{result.stderr.strip()}"
        )
    return bool(result.stdout.strip())


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
            "nunca CLEAN -- y un artefacto DIRTY nunca se instala en "
            "/etc/jax/build/, ver la advertencia en el docstring de este "
            "script)."
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


def _construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera implementation-identity.json + su build manifest (jax#260).",
    )
    parser.add_argument("--repo-root", type=Path, required=True,
                         help="raíz del checkout git a describir (p.ej. /srv/jax-prod/jax)")
    parser.add_argument("--output", type=Path, default=None,
                         help="ruta del JSON de identidad -- obligatorio salvo "
                              "que se use --production")
    parser.add_argument("--production", action="store_true",
                         help=f"usa la ruta fija de despliegue ({_DEFAULT_OUTPUT} "
                              "y su .manifest.json); no combinar con --output")
    parser.add_argument("--manifest-output", type=Path, default=None,
                         help="ruta del blob del build manifest a instalar en "
                              "jax_evidence.evidence_blobs (default: "
                              "<output>.manifest.json)")
    parser.add_argument("--repository-id", default=None,
                         help="por defecto se deriva de `git remote get-url origin`")
    parser.add_argument("--allow-dirty", action="store_true",
                         help="genera igual con un árbol sucio, marcado source_state=DIRTY "
                              "-- nunca instalar ese resultado en /etc/jax/build/")
    return parser


def resolver_salidas(args: argparse.Namespace) -> tuple[Path, Path]:
    """Pura, sin tocar el filesystem: decide `(output, manifest_output)` a
    partir de los flags, o levanta `SystemExit` si la combinación es
    ambigua/insegura. Separada de `main()` para poder probarla sin arriesgar
    una escritura real en /etc/jax/."""
    if args.production and args.output is not None:
        raise SystemExit(
            "--production ya fija --output en la ruta de despliegue; no lo "
            "combines con --output explícito (elegí uno de los dos)."
        )
    if args.production:
        output = _DEFAULT_OUTPUT
    elif args.output is not None:
        output = args.output
    else:
        raise SystemExit(
            "--output es obligatorio (o pasá --production para usar la ruta "
            f"fija de despliegue, {_DEFAULT_OUTPUT})."
        )
    manifest_output = args.manifest_output or output.with_name(output.name + ".manifest.json")
    return output, manifest_output


def main(argv: list[str] | None = None) -> int:
    args = _construir_parser().parse_args(argv)

    try:
        output, manifest_output = resolver_salidas(args)
    except SystemExit as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    repo_root = args.repo_root.resolve()

    # PR#264 ronda 2 (BLOQUE-1): las dos escrituras atómicas estaban FUERA
    # de este try/except -- un PermissionError real de producción (p.ej.
    # /etc/jax/build/ sin crear todavía, mkdir() sobre /etc/jax con jaxsvc)
    # salía como traceback crudo en vez de un ERROR: prolijo con exit 1.
    try:
        repository_id = args.repository_id or repository_id_desde_remoto(repo_root)
        identity, manifest_bytes = construir_identidad(
            repo_root, repository_id=repository_id, allow_dirty=args.allow_dirty,
        )
        escribir_atomico(manifest_output, manifest_bytes)
        identity_bytes = (json.dumps(identity, indent=2, sort_keys=True) + "\n").encode("utf-8")
        escribir_atomico(output, identity_bytes)
    except (ArbolSucioError, EstadoDelArbolIndeterminadoError, FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"identidad escrita en {output} (source_state={identity['source_state']})")
    print(f"build manifest escrito en {manifest_output}")
    if identity["source_state"] == "DIRTY":
        print(
            "ADVERTENCIA: source_state=DIRTY -- este artefacto arranca LAS "
            "MANOS igual, pero deja cada claim B7 en STALE en silencio. "
            "NO instalar en /etc/jax/build/.",
            file=sys.stderr,
        )
    print(
        "Falta instalar el build manifest en jax_evidence.evidence_blobs -- "
        "ver docs/runbooks/implementation-identity.md"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
