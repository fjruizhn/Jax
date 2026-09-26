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

⚠️ RONDA 3 (2026-09-26), corrección de un hallazgo del auditor: ESTO NO ES
UNA TRANSACCIÓN, aunque la versión anterior de este script lo insinuaba
(`autocommit=False` + un `rollback()` en el `except`). MariaDB hace COMMIT
IMPLÍCITO antes de cada sentencia DDL (`ALTER TABLE`, `DROP TABLE`, `CREATE
INDEX`, ...); `revert_project_lifecycle_migration` corre SIETE pasos DDL en
secuencia (005a..005g invertidos), y si el CUARTO falla, los tres anteriores
YA ESTÁN COMMITEADOS PARA SIEMPRE -- ningún `rollback()` los deshace. Este
script corre con `autocommit=True` (lo que en verdad pasa) y, si algo falla
a mitad de camino, el esquema queda EN UN ESTADO INTERMEDIO. La única forma
de saber en qué paso quedó es leer `information_schema` a mano (se imprime
un recordatorio al fallar) -- no hay forma de "reintentar limpio" sin mirar
primero.

⚠️ SI EL CÓDIGO NUEVO SIGUE DESPLEGADO, ESTO NO SIRVE DE NADA A LA LARGA.
`apply_project_authority_migration` corre SOLA en cada arranque de
jax-platform (H1) y vuelve a crear exactamente lo que este script borra --
sus guardas por `information_schema` la hacen idempotente hacia ADELANTE,
no hacia atrás. Revertir 005 de verdad exige, en este orden: (1) desplegar
el código de `jax`/`jax-platform` SIN el llamador de
`apply_project_authority_migration`/`project_authority_migrations.py` (o sin
el despliegue entero que los trajo), (2) DESPUÉS correr este script. Si se
corre este script primero y jax-platform reinicia antes del paso 1, 005
vuelve a estar ahí en el próximo arranque -- no es un error del script, es
lo que la migración está diseñada a hacer.

Por defecto es un DRY RUN: conecta, corre las dos guardas de conteo (las
mismas que `revert_project_lifecycle_migration` corre primero) y reporta qué
haría, sin escribir nada (un `SELECT` no compromete nada, con o sin
autocommit). Con --aplicar corre la bajada real, paso por paso, tal como
`revert_project_lifecycle_migration` la define -- sin pretender que se puede
deshacer a mitad de camino.

Uso (hall9000, con /etc/jax/.env cargado):
  python3 scripts/b9_revertir_005.py              # dry-run
  python3 scripts/b9_revertir_005.py --aplicar     # bajada real, NO transaccional
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
    # autocommit=True a propósito: es lo que en verdad pasa con DDL (commit
    # implícito por sentencia). Fingir autocommit=False + rollback() era el
    # hallazgo de la ronda 3 -- prometía una atomicidad que MariaDB no da.
    return await aiomysql.create_pool(
        host=host, port=int(port), user=os.getenv("JAX_DB_USER", ""),
        password=os.getenv("JAX_DB_PASSWORD", ""), db=os.getenv("JAX_DB_NAME", "jax_memory"),
        autocommit=True, minsize=1, maxsize=2,
        connect_timeout=db_connect_timeout_seconds())


async def _dry_run(pool) -> int:
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT COUNT(*) FROM jax_project_scope WHERE status IN ('ARCHIVED','HIDDEN')")
            (archivados_o_ocultos,) = await cur.fetchone()
            await cur.execute("SELECT COUNT(*) FROM projects WHERE status IN ('hidden','disabled')")
            (proyectos_hidden_disabled,) = await cur.fetchone()
    print(f"DRY RUN -- jax_project_scope ARCHIVED/HIDDEN: {archivados_o_ocultos}")
    print(f"DRY RUN -- projects hidden/disabled: {proyectos_hidden_disabled}")
    if archivados_o_ocultos or proyectos_hidden_disabled:
        print("La bajada real fallaría cerrada por estas filas (revert_project_lifecycle_migration).")
        return 1
    print("Sin filas bloqueantes en este momento -- eso puede cambiar entre esta")
    print("corrida y --aplicar si algo más escribe mientras tanto (no hay lock).")
    print("Antes de --aplicar, confirmar que el código que llama a")
    print("apply_project_authority_migration() ya NO está desplegado (ver el")
    print("aviso del módulo) -- si sigue desplegado, el próximo arranque de")
    print("jax-platform vuelve a crear 005 entera.")
    return 0


async def _aplicar(pool) -> int:
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if raiz not in sys.path:
        sys.path.insert(0, raiz)
    from jax.memory.project_authority_migrations import revert_project_lifecycle_migration

    print("Corriendo la bajada -- NO es una transacción, cada paso DDL hace su")
    print("propio commit implícito. Si esto falla a mitad de camino, el esquema")
    print("queda en un estado intermedio: revisar SHOW CREATE TABLE")
    print("jax_project_scope / jax_project_membership e information_schema a")
    print("mano antes de reintentar.")
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            try:
                await revert_project_lifecycle_migration(cur)
            except Exception as exc:
                print(f"RECHAZADO A MITAD DE CAMINO (posible estado intermedio -- ver arriba): {exc}",
                     file=sys.stderr)
                return 1
    print("Migración 005 revertida.")
    print("Recordatorio: si el código que llama a apply_project_authority_migration()")
    print("sigue desplegado, el próximo arranque de jax-platform la vuelve a crear.")
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
