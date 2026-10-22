#!/usr/bin/env python3
"""Siembra y cambio de `axioma_config` `anti_fabricacion.modo`, siempre con su fila de auditoría.

    sembrar_modo.py                                        # simula: imprime el SQL, no toca nada
    sembrar_modo.py --verificar                            # 0 si la clave es 'sombra' o 'duro'; 1 si falta o es inválida
    sembrar_modo.py --aplicar --actor <quién>              # INSERT IGNORE de 'sombra' (nunca pisa)
    sembrar_modo.py --modo duro|sombra --actor <quién> --motivo "..."   # cambio explícito (nunca por defecto)

ORDEN DE ACTIVACIÓN (docs/runbooks/anti-fabricacion-activacion.md): sembrar `sombra` ANTES de
reiniciar LAS MANOS con el código del detector. Sin la clave el modo es `duro` por omisión.
Pasar a `duro` lo decide Fernando tras U5.

AUDITORÍA. Cada escritura deja una fila en `axioma_config_audit` (convención de
jax/ejecutor/contratos/eleccion_c5.py): ts (UTC), actor_user_id, config_key, valor_anterior,
valor_nuevo, origen='config' (el CHECK de la tabla solo admite config|smtp|catalogo_sync) e ip
NULL (el script no conoce la del operador). `--actor` es el `user_id` o el correo (o su parte
local, si es única) en `jax_users`; si no resuelve a EXACTAMENTE uno, aborta sin escribir.
LA TABLA NO TIENE COLUMNA PARA EL MOTIVO: `--motivo` es obligatorio en el cambio de modo y queda
impreso en la salida del script (pegarlo en el PR/pendiente); guardarlo en la base exige una
migración que no es de este cambio.

Escribe en la base de la conexión (JAX_DB_*): lo corre una persona con GO, nunca una prueba ni
una sesión sola. Las pruebas inyectan `conectar`.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from jax.anti_fabricacion import consecuencia  # noqa: E402

VALOR_INICIAL = "sombra"
MODOS = tuple(m.value for m in consecuencia.Modo)
SQL_SEMBRAR = "INSERT IGNORE INTO axioma_config (config_key, config_value) VALUES (%s, %s)"
SQL_CAMBIAR = "UPDATE axioma_config SET config_value = %s WHERE config_key = %s"
SQL_AUDITORIA = ("INSERT INTO axioma_config_audit (ts, actor_user_id, config_key, valor_anterior, valor_nuevo, origen, ip) "
                 "VALUES (%s, %s, %s, %s, %s, %s, %s)")
SQL_ACTOR = ("SELECT user_id FROM jax_users WHERE deleted_at IS NULL AND "
             "(user_id = %s OR email = %s OR email LIKE %s)")
ORIGEN_AUDITORIA = "config"


class UsoInvalido(Exception):
    pass


async def _leer(cur) -> str | None:
    await cur.execute(consecuencia.SQL_MODO, (consecuencia.CLAVE_MODO,))
    fila = await cur.fetchone()
    return None if not fila or fila[0] is None else str(fila[0]).strip().lower()


async def _resolver_actor(cur, actor: str) -> int:
    como_id = int(actor) if actor.isdigit() else -1
    await cur.execute(SQL_ACTOR, (como_id, actor, actor + "@%"))
    filas = await cur.fetchall()
    ids = {int(f[0]) for f in filas}
    if len(ids) != 1:
        raise UsoInvalido(f"--actor {actor!r} no resuelve a exactamente un usuario de jax_users ({len(ids)} coincidencias)")
    return ids.pop()


async def _auditar(cur, actor_id: int, anterior: str | None, nuevo: str) -> None:
    ts = datetime.now(timezone.utc).replace(tzinfo=None)
    await cur.execute(SQL_AUDITORIA, (ts, actor_id, consecuencia.CLAVE_MODO, anterior, nuevo, ORIGEN_AUDITORIA, None))


async def _sembrar(conectar, actor: str) -> str:
    conn = await conectar()
    try:
        async with conn.cursor() as cur:
            actor_id = await _resolver_actor(cur, actor)
            await cur.execute(SQL_SEMBRAR, (consecuencia.CLAVE_MODO, VALOR_INICIAL))
            if getattr(cur, "rowcount", 0) == 1:
                await _auditar(cur, actor_id, None, VALOR_INICIAL)
                resultado = f"sembrado (INSERT IGNORE): {consecuencia.CLAVE_MODO}={VALOR_INICIAL} (actor user_id={actor_id})"
            else:
                resultado = f"{consecuencia.CLAVE_MODO} ya existía: no se cambió nada ni se audita"
        commit = getattr(conn, "commit", None)
        if commit is not None:
            await commit()
        return resultado
    finally:
        conn.close()


async def _cambiar(conectar, modo: str, actor: str, motivo: str) -> str:
    conn = await conectar()
    try:
        async with conn.cursor() as cur:
            actor_id = await _resolver_actor(cur, actor)
            anterior = await _leer(cur)
            if anterior == modo:
                return f"{consecuencia.CLAVE_MODO} ya es {modo!r}: no se cambió nada ni se audita"
            if anterior is None:
                await cur.execute(SQL_SEMBRAR, (consecuencia.CLAVE_MODO, modo))
            else:
                await cur.execute(SQL_CAMBIAR, (modo, consecuencia.CLAVE_MODO))
            await _auditar(cur, actor_id, anterior, modo)
        commit = getattr(conn, "commit", None)
        if commit is not None:
            await commit()
        return (f"{consecuencia.CLAVE_MODO}: {anterior!r} -> {modo!r} (actor user_id={actor_id}). "
                f"MOTIVO (no hay columna en la tabla; registrarlo en el PR/pendiente): {motivo}")
    finally:
        conn.close()


async def _verificar(conectar) -> str | None:
    conn = await conectar()
    try:
        async with conn.cursor() as cur:
            return await _leer(cur)
    finally:
        conn.close()


def main(argv=None, *, conectar=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--aplicar", action="store_true", help="INSERT IGNORE de 'sombra' (no pisa nada); exige --actor")
    ap.add_argument("--verificar", action="store_true", help="comprueba que la clave existe y es válida")
    ap.add_argument("--modo", help="cambio explícito de modo (duro|sombra); exige --actor y --motivo")
    ap.add_argument("--actor", help="user_id o correo en jax_users de quien decide")
    ap.add_argument("--motivo", help="por qué (obligatorio con --modo)")
    a = ap.parse_args(argv)
    conectar = conectar or consecuencia._db_conn
    try:
        if sum(map(bool, (a.aplicar, a.verificar, a.modo))) > 1:
            raise UsoInvalido("--aplicar, --verificar y --modo son excluyentes")
        if a.verificar:
            valor = asyncio.run(_verificar(conectar))
            if valor in MODOS:
                print(f"{consecuencia.CLAVE_MODO}={valor}")
                return 0
            print(f"{consecuencia.CLAVE_MODO} falta o es inválida ({valor!r}): sembrar antes de reiniciar LAS MANOS",
                  file=sys.stderr)
            return 1
        if a.modo:
            if a.modo not in MODOS:
                raise UsoInvalido(f"--modo debe ser uno de {MODOS}")
            if not a.actor or not (a.motivo or "").strip():
                raise UsoInvalido("--modo exige --actor y --motivo")
            print(asyncio.run(_cambiar(conectar, a.modo, a.actor, a.motivo.strip())))
            return 0
        if a.aplicar:
            if not a.actor:
                raise UsoInvalido("--aplicar exige --actor")
            print(asyncio.run(_sembrar(conectar, a.actor)))
            return 0
    except UsoInvalido as exc:
        print(f"uso inválido: {exc}", file=sys.stderr)
        return 2
    print("SIMULACIÓN (no se escribe nada). Con --aplicar --actor <quién> se ejecutaría:")
    print(SQL_SEMBRAR.replace("%s", "'{}'").format(consecuencia.CLAVE_MODO, VALOR_INICIAL) + ";")
    print("y, si la fila se creó, una fila de axioma_config_audit (actor, ts UTC, valor_anterior NULL, valor_nuevo 'sombra').")
    return 0


if __name__ == "__main__":
    sys.exit(main())
