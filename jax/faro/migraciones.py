"""Aplica las migraciones SQL versionadas del Faro (`ops/faro/migrations/NNN_*.sql`) con parametros
`${nombre}`. Los identificadores se validan antes de sustituirse: son los unicos datos que entran al SQL y
nunca vienen de un modelo ni de un pedido. Se aplican en orden; cada archivo es idempotente
(`CREATE TABLE IF NOT EXISTS`, `GRANT`)."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from string import Template

import aiomysql

_IDENT = re.compile(r"^[A-Za-z0-9_.%-]{1,64}$")


def _sentencias(sql: str) -> list[str]:
    sin_comentarios = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
    return [s.strip() for s in sin_comentarios.split(";") if s.strip()]


async def aplicar(con: aiomysql.Connection, directorio: Path, params: dict[str, str]) -> list[str]:
    """Devuelve `["001_x.sql:<sha12>", ...]` de lo aplicado."""
    for k, v in params.items():
        if not _IDENT.fullmatch(v):
            raise ValueError(f"parametro de migracion invalido: {k}")
    aplicadas: list[str] = []
    for archivo in sorted(Path(directorio).glob("[0-9][0-9][0-9]_*.sql")):
        texto = archivo.read_text(encoding="utf-8")
        async with con.cursor() as cur:
            for s in _sentencias(Template(texto).substitute(params)):
                await cur.execute(s)
        await con.commit()
        aplicadas.append(f"{archivo.name}:{hashlib.sha256(texto.encode()).hexdigest()[:12]}")
    return aplicadas
