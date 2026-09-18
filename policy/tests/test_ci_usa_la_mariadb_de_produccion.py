#!/usr/bin/env python3
"""P12 — CI corre la MISMA MariaDB que produccion, con el tag exacto.

Origen: hasta el 2026-09-18 los workflows levantaban `mariadb:11.8` y
produccion corria 12.3.3 (`docker inspect mariadb-12-3-jax` en hall9000).
Una base distinta en CI no falla ruidoso: da VERDE sobre un plan de
consulta que produccion no usa. Ya nos costo un test de EXPLAIN que medio
la version del optimizador en vez de nuestro cambio -- MariaDB >= 11.1
reescribe `DATE(col) >= const`, asi que el mismo SQL daba dos planes y el
control no probaba nada.

Enforcement mecanico: TODO `image: mariadb:<tag>` de TODO workflow tiene
que declarar el tag de abajo. El defecto que esta regla ataca no es el
valor viejo -- es la DERIVA: cinco lugares que se editan por separado y
se separan solos. Si se mueve produccion, se mueve esta constante y las
cinco lineas en el mismo commit.

El tag es exacto a proposito. Un `mariadb:12.3` flotante se convierte en
otra version sin que nadie lo decida, que es el mismo defecto con mejor
disfraz.

Corre con:
  python3 policy/tests/test_ci_usa_la_mariadb_de_produccion.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# El tag EXACTO que corre produccion en hall9000. Verificado 2026-09-18 con
# `docker inspect mariadb-12-3-jax --format '{{.Config.Image}}'`.
TAG_DE_PRODUCCION = "mariadb:12.3.3"

RAIZ = Path(__file__).resolve().parents[2]
WORKFLOWS = RAIZ / ".github" / "workflows"

# `image: mariadb:<algo>`, con el sangrado que tenga.
IMAGEN = re.compile(r"^\s*image:\s*(mariadb:\S+)\s*$")


def infracciones() -> list[str]:
    encontradas: list[str] = []
    for archivo in sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml")):
        for numero, linea in enumerate(archivo.read_text().splitlines(), 1):
            hallado = IMAGEN.match(linea)
            if hallado and hallado.group(1) != TAG_DE_PRODUCCION:
                encontradas.append(
                    f"{archivo.relative_to(RAIZ)}:{numero}: {hallado.group(1)} "
                    f"-- produccion corre {TAG_DE_PRODUCCION}"
                )
    return encontradas


def main() -> int:
    if not WORKFLOWS.is_dir():
        print(f"FALLO: no existe {WORKFLOWS}", file=sys.stderr)
        return 1

    # Una regla que no ve ninguna linea no esta en verde: esta ciega. Si el
    # dia de manana los workflows dejan de declarar la imagen, esto lo dice
    # en vez de aprobar por vacio.
    declaradas = sum(
        1
        for archivo in list(WORKFLOWS.glob("*.yml")) + list(WORKFLOWS.glob("*.yaml"))
        for linea in archivo.read_text().splitlines()
        if IMAGEN.match(linea)
    )
    if declaradas == 0:
        print(
            "FALLO: ningun workflow declara `image: mariadb:...`. Esta regla "
            "quedaria en verde sin mirar nada -- revisala antes de borrarla.",
            file=sys.stderr,
        )
        return 1

    malas = infracciones()
    if malas:
        print(
            f"FALLO: {len(malas)} workflow(s) levantan una MariaDB distinta a "
            f"produccion. CI verde contra otra version no prueba el codigo:\n",
            file=sys.stderr,
        )
        for mala in malas:
            print(f"  {mala}", file=sys.stderr)
        return 1

    print(f"OK: {declaradas} servicio(s) de CI en {TAG_DE_PRODUCCION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
