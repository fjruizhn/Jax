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
import os
import sys
from pathlib import Path

JAX_CANONICAL = Path(__file__).resolve().parent.parent / "jax" / "core" / "facet_resolver.py"
JAX_PLATFORM_COPY = Path(
    os.environ.get("JAX_PLATFORM_FACET_RESOLVER",
                   Path.home() / "jax-platform" / "backend" / "facet_resolver.py")
)

# Marcador que DECLARA una divergencia como deliberada. Vive en el codigo,
# junto a la divergencia, y no en una lista dentro de este script: una lista
# aparte se desincroniza igual que el codigo que pretende vigilar.
#
# POR QUE EXISTE: hasta 2026-09-01 este checker gritaba TRES veces y dos eran
# falsas (ResolvedFacet y _query_facet divergen a proposito por
# max_tokens_param). El drift REAL -- _db_conn sin el guard fail-closed contra
# el default a la instancia muerta 3306 -- quedaba escondido entre el ruido, y
# el script ademas no corria en ningun workflow. Un detector que no distingue
# lo esperado de lo anomalo entrena a ignorarlo.
MARCADOR = "DIVERGENCIA DELIBERADA"

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
    if not JAX_PLATFORM_COPY.exists():
        print(f"ERROR: no existe {JAX_PLATFORM_COPY}", file=sys.stderr)
        print("Seteá JAX_PLATFORM_FACET_RESOLVER si la copia vive en otra ruta.", file=sys.stderr)
        return 2
    a = _extract(JAX_CANONICAL)
    b = _extract(JAX_PLATFORM_COPY)

    drift, declaradas, faltantes = [], [], []
    for name in SHARED_NAMES:
        if name not in a or name not in b:
            faltantes.append(name)
            continue
        if a[name] == b[name]:
            continue
        # Divergencia declarada EN EL CODIGO, en cualquiera de las dos copias.
        if MARCADOR in a[name] or MARCADOR in b[name]:
            declaradas.append(name)
        else:
            drift.append(name)

    for name in declaradas:
        print(f"DECLARADA: '{name}' diverge a proposito (marcador '{MARCADOR}' en el codigo)")
    for name in faltantes:
        print(f"DRIFT: '{name}' falta en una de las dos copias")
    for name in drift:
        print(f"DRIFT: '{name}' difiere entre jax y jax-platform SIN declararlo")

    if drift or faltantes:
        print()
        print("facet_resolver.py DIVERGIO sin declararlo -- revisar cual lado tiene")
        print("el fix real y portarlo al otro a mano (no hay symlink cruzado entre")
        print("repos posible). Si la divergencia es DELIBERADA, declarala poniendo")
        print(f"'{MARCADOR}' en el docstring o comentario de ese simbolo, con la razon.")
        return 1

    print()
    print(f"facet_resolver.py sincronizado ({len(declaradas)} divergencia(s) declarada(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
