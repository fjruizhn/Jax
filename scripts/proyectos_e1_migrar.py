#!/usr/bin/env python3
"""Migracion de datos Proyectos E1 (spec §5.4).

1. Inicializa HAMURABI (`projects.id = 1`) con `bootstrap_existing_project`.
2. Crea el proyecto archivado «Evaluacion grounding SP3 · 2026-09-03».
3. Reescribe, en UNA transaccion, los `project_id` huerfanos de las cinco
   tablas de contenido hacia ese proyecto, con conteos verificados.

Uso:  python scripts/proyectos_e1_migrar.py {--verificar|--aplicar} --actor-user-id N
          [--database NOMBRE] [--salida-reversion RUTA] [--confirmo-produccion]

La conexion sale de JAX_DB_HOST/PORT/USER/PASSWORD/NAME del entorno; este guion
nunca abre /etc/jax/.env (el runbook carga el entorno). Nunca hace DELETE ni
toca FKs (eso es la Tarea 4).

La llave de idempotencia de la evaluacion es un UUID v5 derivado de un nombre
fijo: `create_project` exige un UUID canonico de 36 caracteres.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import aiomysql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jax.core.db_connect_config import db_connect_timeout_seconds  # noqa: E402
from jax.memory.b9 import MutationAuthorizationRequest, ScopeContext, Visibility  # noqa: E402
from jax.memory.b9_mariadb import MariaDBB9Store  # noqa: E402
from jax.memory.project_authority import LEGACY_PROJECT_TENANT_ID, ProjectAuthorityAdmin  # noqa: E402
from jax.memory.scope_authority import ProjectLifecycle  # noqa: E402

HAMURABI_ID = 1
TABLAS = ("conversations", "messages", "facts", "decisions", "action_items")
NOMBRE_EVALUACION = "Evaluación grounding SP3 · 2026-09-03"
LLAVE_EVALUACION_NOMBRE = "e1-evaluacion-grounding-sp3-20260903"
LLAVE_EVALUACION = str(uuid.uuid5(uuid.NAMESPACE_URL, LLAVE_EVALUACION_NOMBRE))
BASE_PRODUCCION = "jax_memory"
COMPONENTE = "proyectos-e1-migrar"


class HuerfanosRestantes(RuntimeError):
    """Tras el commit quedan huerfanos (aparecieron despues de medir)."""


class ConteosNoCoinciden(RuntimeError):
    """El UPDATE de una tabla no movio las filas contadas: se hizo ROLLBACK."""


def _huerfanos_sql() -> str:
    partes = [f"SELECT project_id FROM `{t}` WHERE project_id IS NOT NULL "
              f"AND project_id NOT IN (SELECT id FROM projects)" for t in TABLAS]
    return "SELECT DISTINCT project_id FROM (" + " UNION ".join(partes) + ") h ORDER BY project_id"


async def _huerfanos(cur) -> list[int]:
    await cur.execute(_huerfanos_sql())
    return [int(r["project_id"]) for r in await cur.fetchall()]


async def _filas(cur, ids: list[int]) -> dict[str, int]:
    if not ids:
        return {t: 0 for t in TABLAS}
    marcas = ",".join(["%s"] * len(ids))
    out = {}
    for t in TABLAS:
        await cur.execute(f"SELECT COUNT(*) AS c FROM `{t}` WHERE project_id IN ({marcas})", tuple(ids))
        out[t] = int((await cur.fetchone())["c"])
    return out


async def medir(pool) -> dict:
    """Solo lectura."""
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            ids = await _huerfanos(cur)
            filas = await _filas(cur, ids)
            await cur.execute("SELECT 1 AS x FROM jax_project_scope WHERE project_id=%s", (HAMURABI_ID,))
            hamurabi = await cur.fetchone() is not None
            await cur.execute("SELECT project_id FROM jax_project_creation_request WHERE idempotency_key=%s",
                              (LLAVE_EVALUACION,))
            fila = await cur.fetchone()
    return {"huerfanos_ids": ids, "filas_por_tabla": filas, "hamurabi_con_alcance": hamurabi,
            "evaluacion_project_id": int(fila["project_id"]) if fila else None}


def _escribir_reversion(ruta: str, destino: int, filas: list[dict]) -> None:
    """Crea el archivo (0600, O_EXCL: jamas pisa uno existente), escribe y
    hace fsync. Se llama ANTES del commit."""
    fd = os.open(ruta, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"evaluacion_project_id": destino, "filas": filas}, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        try:
            os.unlink(ruta)
        except OSError:
            pass
        raise


async def _reescribir_en_transaccion(pool, ids: list[int], destino: int, salida_reversion: str) -> dict[str, int]:
    """Las cinco tablas en UNA conexion: begin ... commit. Antes de cada UPDATE
    lee (FOR UPDATE) las filas huerfanas para el archivo de reversion, que se
    escribe y sincroniza antes del commit. Si algun conteo no coincide o la
    escritura falla: rollback (y el archivo creado se borra)."""
    marcas = ",".join(["%s"] * len(ids))
    creado = False
    async with pool.acquire() as conn:
        await conn.begin()
        try:
            async with conn.cursor() as cur:
                filas: list[dict] = []
                esperado: dict[str, int] = {}
                movido: dict[str, int] = {}
                for t in TABLAS:
                    await cur.execute(f"SELECT id, project_id FROM `{t}` WHERE project_id IN ({marcas}) FOR UPDATE",
                                      tuple(ids))
                    previas = await cur.fetchall()
                    esperado[t] = len(previas)
                    filas.extend({"tabla": t, "id": int(r["id"]), "project_id_anterior": int(r["project_id"])}
                                 for r in previas)
                    await cur.execute(f"UPDATE `{t}` SET project_id=%s WHERE project_id IN ({marcas})",
                                      (destino, *ids))
                    movido[t] = int(cur.rowcount)
                malas = {t: (esperado[t], movido[t]) for t in TABLAS if esperado[t] != movido[t]}
                if malas:
                    raise ConteosNoCoinciden(f"conteos distintos (esperado, movido) por tabla: {malas}")
                if await _huerfanos(cur):
                    raise ConteosNoCoinciden("quedan project_id huerfanos tras la reescritura")
            _escribir_reversion(salida_reversion, destino, filas)
            creado = True
            await conn.commit()
            return movido
        except BaseException:
            await conn.rollback()
            if creado:
                try:
                    os.unlink(salida_reversion)
                except OSError:
                    pass
            raise


def _scope(actor: int, project_id: int | None) -> ScopeContext:
    return ScopeContext(actor_principal=f"user:{actor}", actor_type="USER", subject_user_id=str(actor),
                        tenant_id=str(LEGACY_PROJECT_TENANT_ID),
                        project_id=str(project_id) if project_id is not None else None,
                        calling_component=COMPONENTE)


def _req(actor: int, op: str, project_id: int | None) -> MutationAuthorizationRequest:
    return MutationAuthorizationRequest(_scope(actor, project_id), op, Visibility.PROJECT_SHARED)


async def aplicar(pool, *, actor_user_id: int, salida_reversion: str) -> dict:
    """Idempotente: una segunda corrida no cambia nada y devuelve el mismo id."""
    if os.path.lexists(salida_reversion):
        raise FileExistsError(f"el archivo de reversion ya existe: {salida_reversion}")
    antes = await medir(pool)
    admin = ProjectAuthorityAdmin(MariaDBB9Store(pool))
    await admin.bootstrap_existing_project(
        _req(actor_user_id, "BOOTSTRAP_PROJECT", HAMURABI_ID), HAMURABI_ID, owner_user_id=actor_user_id)
    creado = await admin.create_project(
        _req(actor_user_id, "CREATE_PROJECT", None), name=NOMBRE_EVALUACION, description=None,
        idempotency_key=LLAVE_EVALUACION)
    ev = creado.project_id
    await admin.set_project_lifecycle(_req(actor_user_id, "SET_PROJECT_LIFECYCLE", ev), ev,
                                      ProjectLifecycle.ARCHIVED)
    ids = antes["huerfanos_ids"]
    if ids:
        movido = await _reescribir_en_transaccion(pool, ids, ev, salida_reversion)
    else:
        _escribir_reversion(salida_reversion, ev, [])
        movido = {t: 0 for t in TABLAS}
    despues = await medir(pool)
    if despues["huerfanos_ids"]:
        raise HuerfanosRestantes(
            f"quedan huerfanos tras el commit: {despues['huerfanos_ids']}; hay que volver a correr --aplicar")
    return {"antes": antes, "despues": despues, "evaluacion_project_id": ev, "filas_movidas": movido}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--verificar", action="store_true", help="solo lectura: imprime medir() como JSON")
    g.add_argument("--aplicar", action="store_true", help="escribe: imprime aplicar() como JSON")
    p.add_argument("--actor-user-id", type=int, required=True)
    p.add_argument("--database", default=None, help="pisa a JAX_DB_NAME")
    p.add_argument("--salida-reversion", default=None,
                   help="ruta (nueva, 0600) del JSON con el project_id anterior de cada fila; obligatoria con --aplicar")
    p.add_argument("--confirmo-produccion", action="store_true",
                   help=f"obligatorio con --aplicar si la base es {BASE_PRODUCCION}")
    return p


async def _correr(args, database: str) -> dict:
    pool = await aiomysql.create_pool(
        host=os.environ.get("JAX_DB_HOST", ""), port=int(os.environ.get("JAX_DB_PORT", "3306")),
        user=os.environ.get("JAX_DB_USER", ""), password=os.environ.get("JAX_DB_PASSWORD", ""),
        db=database, autocommit=True, minsize=1, maxsize=4, cursorclass=aiomysql.DictCursor,
        connect_timeout=db_connect_timeout_seconds())
    try:
        if args.verificar:
            return await medir(pool)
        return await aplicar(pool, actor_user_id=args.actor_user_id,
                             salida_reversion=args.salida_reversion)
    finally:
        pool.close()
        await pool.wait_closed()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    database = args.database or os.environ.get("JAX_DB_NAME", "")
    if not database:
        print("falta la base: use --database o JAX_DB_NAME", file=sys.stderr)
        return 2
    if args.aplicar and database == BASE_PRODUCCION and not args.confirmo_produccion:
        print(f"--aplicar sobre {BASE_PRODUCCION} (produccion) exige --confirmo-produccion", file=sys.stderr)
        return 2
    if args.aplicar and not args.salida_reversion:
        print("--aplicar exige --salida-reversion <ruta> (archivo nuevo)", file=sys.stderr)
        return 2
    try:
        resultado = asyncio.run(_correr(args, database))
    except (HuerfanosRestantes, ConteosNoCoinciden, FileExistsError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
