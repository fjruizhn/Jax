#!/usr/bin/env python3
"""Llaves foraneas de `project_id` hacia `projects(id)` (Proyectos E1, T4).

Por cada tabla de contenido (conversations, messages, facts, decisions,
action_items) intenta
    ALTER TABLE <t> ADD CONSTRAINT fk_<t>_project FOREIGN KEY (project_id)
        REFERENCES projects(id), ALGORITHM=INPLACE, LOCK=SHARED
y escribe una linea JSON por tabla:
    {"tabla", "filas", "algoritmo", "segundos", "hnsw_intacto", "aplicada"}
(mas "ya_existia" / "error" / "motivo" cuando corresponde).

Uso:  python scripts/proyectos_e1_fks.py {--ensayar|--aplicar} [--database NOMBRE]
          [--tablas t1,t2] [--confirmo-produccion]

--ensayar  solo sobre una COPIA de jax_memory (se niega sobre jax_memory).
--aplicar  sobre jax_memory exige --confirmo-produccion (codigo 2 si falta).

Reglas por tabla:
  - Si la FK ya existe (information_schema, por columna, no por nombre): no
    crea nada, `ya_existia: true`, pero REVALIDA: cuenta huerfanos (si hay, error)
    y corre el detector HNSW. Repetir es seguro, no da por bueno lo previo.
  - `--tablas` limita lo que se toca; con --aplicar sobre jax_memory es obligatoria.
  - El ALTER usa LOCK=SHARED y lock_wait_timeout=5: si el lock de metadatos esta
    ocupado, la tabla sale `aplicada: false, motivo: lock_timeout` y sigue la
    siguiente.
  - ANTES del ALTER cuenta huerfanos; si hay, aborta ESA tabla con un error
    claro y sigue con las demas.
  - El ALTER corre con foreign_key_checks=0 en la sesion (unico modo en que
    MariaDB acepta INPLACE); por eso los huerfanos se cuentan antes Y despues.
  - Si MariaDB rechaza INPLACE/LOCK=SHARED (error 1846) registra
    `"algoritmo": "COPY"` y NO aplica esa FK.
  - TODA tabla procesada (hoy messages y facts tienen indice VECTOR): tras el
    ALTER corre el detector de indices vectoriales (`jax.memory.indice_vectorial`,
    el mismo de `scripts/revisar_indice_vectorial.py`) sobre los indices de esa
    tabla. `hnsw_intacto` es true/false; null si la tabla no tiene indice VECTOR
    (el detector no aplica). Solo mira, nunca repara.

Salida: 0 todas aplicadas (o ya existian) y hnsw sano; 1 alguna sin aplicar o
con hnsw roto; 2 argumentos/guarda.

Conexion: JAX_DB_HOST/PORT/USER/PASSWORD/NAME del entorno; nunca abre
/etc/jax/.env (el runbook carga el entorno).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiomysql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jax.core.db_connect_config import db_connect_timeout_seconds  # noqa: E402
from jax.memory import indice_vectorial as iv  # noqa: E402

TABLAS = ("conversations", "messages", "facts", "decisions", "action_items")
BASE_PRODUCCION = "jax_memory"
_ERRNO_NO_SOPORTADO = 1846          # "ALGORITHM=INPLACE / LOCK=... is not supported"
_ERRNO_LOCK_TIMEOUT = 1205          # "Lock wait timeout exceeded" (tambien el de metadatos)
#: Segundos que el ALTER espera el lock de metadatos antes de rendirse: el ALTER
#: encolado bloquea a todo el que llegue despues (incluido el chat).
LOCK_WAIT_TIMEOUT = 5


def _alter_sql(tabla: str) -> str:
    return (f"ALTER TABLE `{tabla}` ADD CONSTRAINT `fk_{tabla}_project` FOREIGN KEY (project_id) "
            f"REFERENCES projects(id), ALGORITHM=INPLACE, LOCK=SHARED")


async def _alter(cur, tabla: str) -> None:
    """MariaDB solo admite ADD FOREIGN KEY con ALGORITHM=INPLACE si la sesion
    tiene foreign_key_checks=OFF (con ON da 1846, medido en 12.3.3): por eso se
    apaga SOLO durante este ALTER y se restaura siempre. Sin validar, la
    garantia es el conteo de huerfanos de antes (en `_tabla`) y el de despues."""
    await cur.execute("SET SESSION foreign_key_checks=0")
    await cur.execute(f"SET SESSION lock_wait_timeout={int(LOCK_WAIT_TIMEOUT)}")
    try:
        await cur.execute(_alter_sql(tabla))
    finally:
        await cur.execute("SET SESSION foreign_key_checks=1")
        await cur.execute("SET SESSION lock_wait_timeout=DEFAULT")


async def _fk_existe(cur, tabla: str) -> bool:
    await cur.execute(
        "SELECT COUNT(*) FROM information_schema.KEY_COLUMN_USAGE WHERE TABLE_SCHEMA=DATABASE() "
        "AND TABLE_NAME=%s AND COLUMN_NAME='project_id' AND REFERENCED_TABLE_NAME='projects' "
        "AND REFERENCED_COLUMN_NAME='id'", (tabla,))
    return (await cur.fetchone())[0] > 0


async def _huerfanos(cur, tabla: str) -> list[int]:
    await cur.execute(
        f"SELECT DISTINCT project_id FROM `{tabla}` WHERE project_id IS NOT NULL "
        f"AND project_id NOT IN (SELECT id FROM projects) ORDER BY project_id LIMIT 20")
    return [int(r[0]) for r in await cur.fetchall()]


async def _hnsw(cur, tabla: str) -> bool | None:
    """None si la tabla no tiene indice VECTOR; False si alguno miente."""
    informes = [await iv.revisar_uno(cur, t, i, c)
                for t, i, c in await iv.indices_vectoriales(cur) if t == tabla]
    if not informes:
        return None
    return all(inf.sano for inf in informes)


async def _tabla(cur, tabla: str) -> dict:
    await cur.execute(f"SELECT COUNT(*) FROM `{tabla}`")
    res: dict = {"tabla": tabla, "filas": int((await cur.fetchone())[0]), "algoritmo": None,
                 "segundos": 0.0, "hnsw_intacto": None, "aplicada": False}
    existia = await _fk_existe(cur, tabla)
    huerfanos = await _huerfanos(cur, tabla)       # SIEMPRE: una FK creada sin validar no prueba nada
    if existia:
        res["ya_existia"] = True
        if huerfanos:
            res["error"] = (f"hay huerfanos en {tabla} bajo una FK que ya existia (primeros: {huerfanos}); "
                            f"se creo sin validar o entraron con foreign_key_checks=0. Corregirlos.")
            return res
        res["aplicada"] = True
        res["hnsw_intacto"] = await _hnsw(cur, tabla)
        return res
    if huerfanos:
        res["error"] = (f"quedan huerfanos en {tabla} (project_id sin fila en projects, primeros: "
                        f"{huerfanos}); correr proyectos_e1_migrar.py --aplicar antes. No se aplico la FK.")
        return res
    inicio = time.monotonic()
    try:
        await _alter(cur, tabla)
    except (aiomysql.Error, ) as e:
        res["segundos"] = round(time.monotonic() - inicio, 3)
        if e.args and e.args[0] == _ERRNO_LOCK_TIMEOUT:
            res.update(motivo="lock_timeout", error=f"lock de metadatos ocupado ({LOCK_WAIT_TIMEOUT} s): "
                       f"hay una transaccion abierta sobre {tabla}; reintentar cuando termine")
        elif e.args and e.args[0] == _ERRNO_NO_SOPORTADO:
            res.update(algoritmo="COPY", motivo=str(e.args[-1]))
        else:
            res["error"] = f"{type(e).__name__}: {e}"
        return res
    res.update(algoritmo="INPLACE", segundos=round(time.monotonic() - inicio, 3), aplicada=True)
    # INPLACE no valida las filas existentes: se comprueba de nuevo, por si
    # entro un huerfano entre el conteo previo y el ALTER.
    despues = await _huerfanos(cur, tabla)
    if despues:
        res.update(aplicada=False, error=(f"aparecieron huerfanos en {tabla} durante el ALTER: {despues}; la FK "
                                          f"quedo creada SIN validar: corregirlos o `DROP FOREIGN KEY "
                                          f"fk_{tabla}_project` (ver runbook)"))
        return res
    res["hnsw_intacto"] = await _hnsw(cur, tabla)
    return res


async def procesar(conn, emitir=None, tablas=TABLAS) -> list[dict]:
    """Procesa `tablas` con una conexion abierta; `emitir(res)` se llama por tabla."""
    out = []
    async with conn.cursor() as cur:
        for t in tablas:
            res = await _tabla(cur, t)
            out.append(res)
            if emitir:
                emitir(res)
    return out


def _ok(res: dict) -> bool:
    return res["aplicada"] and res["hnsw_intacto"] is not False


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ensayar", action="store_true", help="sobre una copia de jax_memory")
    g.add_argument("--aplicar", action="store_true", help="aplica las FKs")
    p.add_argument("--database", default=None, help="pisa a JAX_DB_NAME")
    p.add_argument("--tablas", default=None,
                   help="lista separada por comas (subconjunto de las cinco); obligatoria con --aplicar "
                        f"sobre {BASE_PRODUCCION}: solo las que el ensayo dio por buenas")
    p.add_argument("--confirmo-produccion", action="store_true",
                   help=f"obligatorio con --aplicar si la base es {BASE_PRODUCCION}")
    return p


async def _correr(database: str, tablas) -> list[dict]:
    conn = await aiomysql.connect(
        host=os.environ.get("JAX_DB_HOST", ""), port=int(os.environ.get("JAX_DB_PORT", "3306")),
        user=os.environ.get("JAX_DB_USER", ""), password=os.environ.get("JAX_DB_PASSWORD", ""),
        db=database, autocommit=True, connect_timeout=db_connect_timeout_seconds())
    try:
        return await procesar(conn, lambda r: print(json.dumps(r, ensure_ascii=False), flush=True), tablas)
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    database = args.database or os.environ.get("JAX_DB_NAME", "")
    if not database:
        print("falta la base: use --database o JAX_DB_NAME", file=sys.stderr)
        return 2
    tablas = TABLAS
    if args.tablas is not None:
        tablas = tuple(t.strip() for t in args.tablas.split(","))
        malas = [t for t in tablas if t not in TABLAS]
        if malas:
            print(f"--tablas: nombre(s) no valido(s) {malas}; validos: {', '.join(TABLAS)}", file=sys.stderr)
            return 2
    if args.ensayar and database == BASE_PRODUCCION:
        print(f"--ensayar es sobre una copia: se niega sobre {BASE_PRODUCCION}", file=sys.stderr)
        return 2
    if args.aplicar and database == BASE_PRODUCCION and not args.confirmo_produccion:
        print(f"--aplicar sobre {BASE_PRODUCCION} (produccion) exige --confirmo-produccion", file=sys.stderr)
        return 2
    if args.aplicar and database == BASE_PRODUCCION and args.tablas is None:
        print(f"--aplicar sobre {BASE_PRODUCCION} exige --tablas <las que el ensayo dio por buenas>",
              file=sys.stderr)
        return 2
    resultados = asyncio.run(_correr(database, tablas))
    return 0 if all(_ok(r) for r in resultados) else 1


if __name__ == "__main__":
    sys.exit(main())
