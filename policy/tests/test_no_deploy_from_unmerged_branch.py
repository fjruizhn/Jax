#!/usr/bin/env python3
"""P11 — nada llega a axioma-ia.io desde una rama que no sea master.

Origen: dos incidentes reales el mismo dia (2026-08-19), documentados en
la correccion de plan commiteada a jax-platform/master
(commit edab23f, "fix(plan): saca el deploy de frontend a produccion de
Tasks 6 y 9") -- un build de una rama sin mergear llamando a un endpoint
que master no tenia, 404 en vivo para cualquier usuario real, revertido
dos veces en la misma sesion.

Enforcement mecanico: el bundle JS realmente servido en axioma-ia.io se
compara contra un rebuild limpio del HEAD real de origin/master. Si
difieren, lo que esta en produccion no vino de master -- exactamente el
incidente que esta regla nombra. No requiere acceso de escritura a la VM,
solo lectura del bundle publico via HTTPS y un `git rev-parse`/build local.

Requiere: conexion a axioma-ia.io y al checkout real de jax-platform en
este host (hall9000) -- no corre en un entorno sin ese checkout.

Corre con:
  python3 policy/tests/test_no_deploy_from_unmerged_branch.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import urllib.request

AXIOMA_URL = "https://axioma-ia.io"
JAX_PLATFORM_REPO = "/home/fruiz/jax-platform"


def _fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.read().decode("utf-8")


def _deployed_bundle_urls() -> list[str]:
    html = _fetch(AXIOMA_URL)
    # index.html referencia los assets hasheados por vite -- extraer las
    # rutas /assets/*.js reales, no asumir un nombre fijo.
    return [
        AXIOMA_URL + m
        for m in re.findall(r'src="(/assets/[^"]+\.js)"', html)
    ]


def _deployed_bundle_hash() -> str:
    urls = _deployed_bundle_urls()
    if not urls:
        raise RuntimeError("no se encontro ningun bundle .js referenciado en index.html")
    h = hashlib.sha256()
    for url in sorted(urls):
        h.update(_fetch(url).encode("utf-8"))
    return h.hexdigest()


def _rebuild_master_bundle_hash() -> str:
    head = subprocess.run(
        ["git", "-C", JAX_PLATFORM_REPO, "rev-parse", "origin/master"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    # Construye una copia limpia del HEAD real de origin/master en un
    # worktree temporal -- nunca toca el checkout de trabajo real.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["git", "-C", JAX_PLATFORM_REPO, "worktree", "add", "--detach", tmp, head],
            capture_output=True, text=True, check=True,
        )
        try:
            subprocess.run(
                ["npm", "install", "--silent"],
                cwd=f"{tmp}/frontend", capture_output=True, text=True, check=True,
            )
            subprocess.run(
                ["npm", "run", "build"],
                cwd=f"{tmp}/frontend", capture_output=True, text=True, check=True,
            )
            import pathlib
            dist = pathlib.Path(tmp) / "frontend" / "dist" / "assets"
            js_files = sorted(dist.glob("*.js"))
            h = hashlib.sha256()
            for f in js_files:
                h.update(f.read_text(encoding="utf-8").encode("utf-8"))
            return h.hexdigest()
        finally:
            subprocess.run(
                ["git", "-C", JAX_PLATFORM_REPO, "worktree", "remove", "--force", tmp],
                capture_output=True, text=True,
            )


def main() -> int:
    try:
        deployed = _deployed_bundle_hash()
    except Exception as e:  # noqa: BLE001 -- reportar la causa real, no fail-open
        print(f"FAIL — no se pudo leer el bundle desplegado en {AXIOMA_URL}: {e}")
        return 1

    try:
        master = _rebuild_master_bundle_hash()
    except subprocess.CalledProcessError as e:
        print(f"FAIL — no se pudo rebuildear origin/master: {e.stderr}")
        return 1

    if deployed != master:
        print(
            "FAIL — el bundle desplegado en axioma-ia.io NO coincide con un "
            "rebuild limpio de origin/master. Algo que no está en master llegó "
            "a producción."
        )
        print(f"  deployed sha256: {deployed}")
        print(f"  master   sha256: {master}")
        return 1

    print("OK — el bundle desplegado coincide con origin/master.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
