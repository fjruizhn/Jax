#!/usr/bin/env python3
"""Bajada a mano de la migración 005 (proyectos D1-D5, lifecycle ARCHIVED/
HIDDEN/DISABLED, procedencia de membresía, `jax_project_creation_request`).

POR QUE A MANO. `apply_project_authority_migration` (003+005) SI corre sola
al arrancar jax-platform (H1); su bajada NO -- perder ARCHIVED/HIDDEN y la
procedencia TENANT_ADMIN/pre_admin_* de las membresías es una decisión
operativa, nunca un efecto colateral de un despliegue.

`revert_project_lifecycle_migration` (jax/memory/project_authority_migrations.py)
falla cerrado por su cuenta si queda algún `jax_project_scope` ARCHIVED/HIDDEN
o algún `projects` hidden/disabled -- este script no repite esa guarda, solo
la deja correr y reporta el error tal cual si aparece.

Por defecto es un DRY RUN: conecta, corre las dos guardas de conteo (las
mismas que `revert_project_lifecycle_migration` corre primero) y reporta qué
haría, sin escribir nada. Con --aplicar corre la bajada real en una
transacción y hace COMMIT solo si termina sin excepción.

Uso (hall9000, con /etc/jax/.env cargado):
  python3 scripts/b9_revertir_005.py              # dry-run
  python3 scripts/b9_revertir_005.py --aplicar     # bajada real
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys


async def _pool_desde_env():
    import aiomysql

    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if raiz not in sys.path:
        sys.path.insert(0, raiz)
    from jax.core.db_connect_config import db_connect_timeout_seconds

    host, port = os.environ.get("JAX_DB_HOST"), os.environ.get("JAX_DB_PORT")
    if not host or not port:
        raise SystemExit("JAX_DB_HOST/JAX_DB_PORT sin setear: sourceá /etc/jax/.env")
    return await aiomysql.create_pool(
        host=host, port=int(port), user=os.getenv("JAX_DB_USER", ""),
        password=os.getenv("JAX_DB_PASSWORD", ""), db=os.getenv("JAX_DB_NAME", "jax_memory"),
        autocommit=False, minsize=1, maxsize=2,
        connect_timeout=db_connect_timeout_seconds())


async def _dry_run(pool) -> int:
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT COUNT(*) FROM jax_project_scope WHERE status IN ('ARCHIVED','HIDDEN')")
            (archivados_o_ocultos,) = await cur.fetchone()
            await cur.execute("SELECT COUNT(*) FROM projects WHERE status IN ('hidden','disabled')")
            (proyectos_hidden_disabled,) = await cur.fetchone()
        await conn.rollback()
    print(f"DRY RUN -- jax_project_scope ARCHIVED/HIDDEN: {archivados_o_ocultos}")
    print(f"DRY RUN -- projects hidden/disabled: {proyectos_hidden_disabled}")
    if archivados_o_ocultos or proyectos_hidden_disabled:
        print("La bajada real fallaría cerrada por estas filas (revert_project_lifecycle_migration).")
        return 1
    print("Sin filas bloqueantes. Correr con --aplicar para bajar la migración 005 de verdad.")
    return 0


async def _aplicar(pool) -> int:
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if raiz not in sys.path:
        sys.path.insert(0, raiz)
    from jax.memory.project_authority_migrations import revert_project_lifecycle_migration

    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            try:
                await revert_project_lifecycle_migration(cur)
            except Exception as exc:
                await conn.rollback()
                print(f"RECHAZADO: {exc}", file=sys.stderr)
                return 1
        await conn.commit()
    print("Migración 005 revertida.")
    return 0


async def _main(aplicar: bool) -> int:
    pool = await _pool_desde_env()
    try:
        return await (_aplicar(pool) if aplicar else _dry_run(pool))
    finally:
        pool.close()
        await pool.wait_closed()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--aplicar", action="store_true", help="corre la bajada real (por defecto: dry-run)")
    args = p.parse_args()
    sys.exit(asyncio.run(_main(args.aplicar)))
