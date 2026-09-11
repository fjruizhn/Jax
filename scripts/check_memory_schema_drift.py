#!/usr/bin/env python3
"""Compara `jax_memory_schema.sql` contra el esquema VIVO de la base.

POR QUE EXISTE: el archivo de esquema paso meses describiendo otra base.
Le faltaban columnas que el codigo usa (conversations.user_id/project_id,
de las que depende el scope de la busqueda semantica), el ENUM de
messages.role tenia 5 de 8 valores y `embedding` figuraba NULL sin VECTOR
KEY. Nadie lo noto porque ningun ALTER pasa por el archivo: se agregaban a
mano en produccion. Tercera instancia del mismo patron en un mes (antes:
jacobs_steps.depends_on y las 3 capabilities HTTP-directo). Regenerado el
2026-09-11 desde `SHOW CREATE TABLE`; este script es lo que impide que
vuelva a divergir en silencio.

QUE COMPARA: por cada `CREATE TABLE` del archivo, su texto contra el
`SHOW CREATE TABLE` de la base, normalizando lo que no es esquema
(`AUTO_INCREMENT=<n>`, espacios, `;` final). Como el archivo ES la salida de
`SHOW CREATE TABLE`, cualquier diferencia que queda es una diferencia real:
una columna, un tipo, un default, un indice.

Fail-closed: exit 0 solo si TODAS las tablas coinciden; 1 si alguna
difiere o falta; 2 si no se pudo leer el archivo o la base. Nunca "no
aplica".

Uso (hall9000, contra produccion; solo lectura):
    set -a; . /etc/jax/.env; set +a
    .venv/bin/python scripts/check_memory_schema_drift.py

Limite declarado: la salida de `SHOW CREATE TABLE` depende de la version del
servidor. El archivo se genero contra la de produccion (MariaDB 12.3); por
eso este chequeo corre contra produccion y no contra la 11.8 de CI.
"""
from __future__ import annotations

import difflib
import os
import re
import sys
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "jax_memory_schema.sql"

_CREATE_RE = re.compile(r"CREATE TABLE `(\w+)` \(.*?\n\)[^;\n]*", re.S)


def normalize(ddl: str) -> str:
    """Lo que no es esquema afuera: el contador AUTO_INCREMENT, espacios
    repetidos y el `;` final."""
    ddl = re.sub(r"\s+AUTO_INCREMENT=\d+", "", ddl)
    ddl = ddl.strip().rstrip(";").strip()
    return "\n".join(re.sub(r"\s+", " ", line).strip() for line in ddl.splitlines() if line.strip())


def parse_schema(text: str) -> dict[str, str]:
    """{tabla: DDL normalizado} de cada `CREATE TABLE` del archivo."""
    return {m.group(1): normalize(m.group(0)) for m in _CREATE_RE.finditer(text)}


def compare(archivo: dict[str, str], vivo: dict[str, str | None]) -> list[str]:
    """Diferencias legibles; lista vacia = sin drift. `vivo[t] is None`
    significa que la tabla no existe en la base."""
    problemas = []
    for tabla, ddl_archivo in archivo.items():
        ddl_vivo = vivo.get(tabla)
        if ddl_vivo is None:
            problemas.append(f"{tabla}: esta en el archivo y NO existe en la base")
            continue
        ddl_vivo = normalize(ddl_vivo)
        if ddl_vivo != ddl_archivo:
            diff = difflib.unified_diff(
                ddl_archivo.splitlines(), ddl_vivo.splitlines(),
                "archivo", "base", lineterm="", n=0)
            cuerpo = [l for l in diff if not l.startswith(("---", "+++", "@@"))]
            problemas.append(f"{tabla}: DIFIERE\n    " + "\n    ".join(cuerpo))
    return problemas


def _leer_vivo(tablas: list[str]) -> dict[str, str | None]:
    import pymysql  # dependencia de aiomysql; import tardio: los tests puros no la necesitan

    conn = pymysql.connect(
        host=os.environ["JAX_DB_HOST"], port=int(os.environ["JAX_DB_PORT"]),
        user=os.environ["JAX_DB_USER"], password=os.environ["JAX_DB_PASSWORD"],
        database=os.environ.get("JAX_DB_NAME", "jax_memory"))
    try:
        with conn.cursor() as cur:
            vivo: dict[str, str | None] = {}
            for t in tablas:
                cur.execute(
                    "SELECT 1 FROM information_schema.TABLES "
                    "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s", (t,))
                if not cur.fetchone():
                    vivo[t] = None
                    continue
                cur.execute(f"SHOW CREATE TABLE `{t}`")
                vivo[t] = cur.fetchone()[1]
            return vivo
    finally:
        conn.close()


def main() -> int:
    try:
        archivo = parse_schema(SCHEMA_PATH.read_text(encoding="utf-8"))
        if not archivo:
            print(f"ERROR: ningun CREATE TABLE en {SCHEMA_PATH}")
            return 2
        vivo = _leer_vivo(list(archivo))
    except Exception as e:  # fail-closed: sin lectura no hay veredicto verde, exit 2
        print(f"ERROR: no se pudo comparar ({type(e).__name__}: {e})")
        return 2
    problemas = compare(archivo, vivo)
    base = os.environ.get("JAX_DB_NAME", "jax_memory")
    if problemas:
        print(f"DRIFT: {len(problemas)} tabla(s) de {len(archivo)} difieren entre "
              f"{SCHEMA_PATH.name} y {base}:")
        for p in problemas:
            print(f"  - {p}")
        return 1
    print(f"OK: las {len(archivo)} tablas de {SCHEMA_PATH.name} coinciden con {base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
