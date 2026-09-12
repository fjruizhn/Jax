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
