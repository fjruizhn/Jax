"""DDL de las tablas de memoria, leido del esquema del repo.

Compartido por los tests de I/O de memoria (`test_memory_vector_zero_io.py` y
`test_memory_scope_denormalized.py`) en vez de copiado en cada uno: una segunda
copia del parser es un segundo lugar donde el esquema se desactualiza sin que
nadie lo note, que es exactamente el problema que `jax_memory_schema.sql` vino
a cerrar.

No es un modulo `test_*`: pytest no lo colecta, es una utilidad de tests.
"""
from __future__ import annotations

import re
from pathlib import Path

SCHEMA = Path(__file__).resolve().parents[1] / "jax_memory_schema.sql"

# Orden de creacion: facts depende de messages, messages de conversations.
TABLAS = ("conversations", "messages", "facts")


def ddl_del_archivo(tabla: str) -> str:
    m = re.search(rf"CREATE TABLE `{tabla}` \(.*?\n\)[^;\n]*",
                  SCHEMA.read_text(encoding="utf-8"), re.S)
    assert m, f"{tabla} no esta en {SCHEMA.name}: el test no podria crearla"
    return m.group(0)


def ddl() -> dict[str, str]:
    return {t: ddl_del_archivo(t) for t in TABLAS}


async def vaciar(conn) -> None:
    """Vacia las tablas de memoria con TRUNCATE, sobre UNA conexion.

    **Por que TRUNCATE y no DELETE.** Medido el 2026-09-20 en hall9000, contra
    MariaDB 12.3.3: despues de un DELETE masivo, el indice vectorial HNSW deja
    de encontrar filas -- y no se recupera. Reproducido en base aislada:

        300 filas -> DELETE de todo -> insertar 1 fila -> buscarla
          por el indice ................... []            <-- vacio, SIN error
          con IGNORE INDEX del vector ..... la encuentra
        insertar 60 filas mas: 0 de 61 encontradas por el indice.
        OPTIMIZE TABLE **no** lo repara. TRUNCATE si (reconstruye la tabla).

    O sea: un fixture que limpia con DELETE deja el indice envenenado para todo
    lo que venga despues en esa base, en silencio. Es el cero silencioso, pero
    en el motor de busqueda.

    Va sobre una sola conexion a proposito: `SET FOREIGN_KEY_CHECKS` es de
    sesion, y los helpers `_sql()` de los tests abren una conexion por llamada.
    Sin eso, el TRUNCATE de `conversations` falla por la FK de `messages`.
    """
    async with conn.cursor() as cur:
        await cur.execute("SET FOREIGN_KEY_CHECKS = 0")
        try:
            for tabla in reversed(TABLAS):
                await cur.execute(f"TRUNCATE TABLE {tabla}")
        finally:
            await cur.execute("SET FOREIGN_KEY_CHECKS = 1")
