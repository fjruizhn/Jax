#!/usr/bin/env python3
"""
Compara la logica compartida de facet_resolver.py entre jax (canonico,
jax/core/facet_resolver.py -- las_manos/facet_resolver.py es symlink al
mismo archivo, ver Bloque 2, 2026-08-21) y jax-platform (copia real
aparte, backend/facet_resolver.py -- repo distinto, con su propio
credential_resolver.py local, no se puede symlinkear entre repos y
sobrevivir un clone fresco).

Compara solo las funciones/clases que DEBEN ser identicas (todo salvo el
import de credential_resolver, que es legitimamente distinto por diseno,
y load_facet_registry, que es bootstrap exclusivo del REPL y no existe en
jax-platform). Si algo mas diverge, el drift es real: alguien corrigio un
bug o cambio comportamiento en un lado y no en el otro.

NO arregla nada, diagnostico puro -- mismo espiritu que
find_unread_columns.py.

Uso:
  python3 scripts/check_facet_resolver_sync.py

Exit code 0 si esta sincronizado, 1 si hay drift real.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

JAX_CANONICAL = Path(__file__).resolve().parent.parent / "jax" / "core" / "facet_resolver.py"
JAX_PLATFORM_COPY = Path.home() / "jax-platform" / "backend" / "facet_resolver.py"

# Nombres que DEBEN coincidir byte-a-byte (fuente) entre ambos archivos.
# load_facet_registry queda afuera a proposito: exclusivo de jax (REPL).
SHARED_NAMES = [
    "FacetUnavailableError",
    "ResolvedFacet",
    "_CacheEntry",
    "_db_conn",
    "_query_facet",
    "resolve_facet",
]


def _extract(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    source = path.read_text()
    out = {}
    for node in tree.body:
        name = getattr(node, "name", None)
        if name in SHARED_NAMES:
            out[name] = ast.get_source_segment(source, node)
    return out


def main() -> int:
    if not JAX_CANONICAL.exists():
        print(f"FALTA: {JAX_CANONICAL}")
        return 1
    if not JAX_PLATFORM_COPY.exists():
        print(f"FALTA: {JAX_PLATFORM_COPY} (jax-platform no clonado en ~/jax-platform?)")
        return 1

    jax_funcs = _extract(JAX_CANONICAL)
    platform_funcs = _extract(JAX_PLATFORM_COPY)

    drift = False
    for name in SHARED_NAMES:
        a = jax_funcs.get(name)
        b = platform_funcs.get(name)
        if a is None or b is None:
            print(f"DRIFT: '{name}' falta en {'jax' if a is None else 'jax-platform'}")
            drift = True
        elif a != b:
            print(f"DRIFT: '{name}' difiere entre jax y jax-platform")
            drift = True

    if drift:
        print("\nfacet_resolver.py DIVERGIO entre jax y jax-platform -- revisar "
              "cual lado tiene el fix real y portarlo al otro a mano "
              "(no hay symlink cruzado entre repos posible).")
        return 1

    print("OK -- facet_resolver.py sincronizado entre jax y jax-platform "
          f"({len(SHARED_NAMES)} funciones/clases comparadas).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
