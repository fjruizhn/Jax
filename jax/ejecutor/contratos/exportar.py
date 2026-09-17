# jax/ejecutor/contratos/exportar.py
"""De la DB a /etc/jax-ejecutor/politica.json (C1/C2). Corre como `fruiz`.

Valida con el MISMO `politica.validar` del gancho antes de publicar: si no pasa,
no toca el archivo publicado y sale ≠ 0. Escritura atómica: temporal en el mismo
directorio (hereda el grupo `axioma` por el setgid del directorio), fsync, rename,
fsync del directorio.

Lee la DB de PRODUCCIÓN cuando corre con /etc/jax/.env: sólo SELECT.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from jax.ejecutor.contratos import politica

SQL_HOSTS = "SELECT nombre, ip, puerto, rol, es_local FROM ejecutor_host WHERE activo = 1 ORDER BY nombre"
SQL_REGLAS = ("SELECT id, codigo, tipo, herramientas, campo, patron, ambito_host, ambito_roles, es_canario, "
              "ejemplos_coincide, ejemplos_no_coincide FROM ejecutor_regla WHERE activa = 1 ORDER BY id")
SQL_RESPALDOS = ("SELECT host_nombre, MAX(restaurado_y_verificado_at) FROM ejecutor_punto_restauracion "
                 "GROUP BY host_nombre")
SQL_EDAD = "SELECT config_value FROM axioma_config WHERE config_key = 'ejecutor.c2_edad_max_s'"


class ExportacionImposible(RuntimeError):
    def __init__(self, codigo: str):
        super().__init__(codigo)
        self.codigo = codigo


def _roles(v) -> list:
    if not v:
        return []
    return sorted(v) if isinstance(v, (set, frozenset)) else sorted(v.split(","))


def documento(hosts, reglas, respaldos, edad_c2, generada_at: str) -> dict:
    try:
        edad = int(edad_c2)
    except (TypeError, ValueError):
        raise ExportacionImposible("c2_edad_max_invalida") from None
    if edad <= 0:
        raise ExportacionImposible("c2_edad_max_invalida")
    doc = {
        "version": politica.VERSION,
        "generada_at": generada_at,
        "hosts": [{"nombre": n, "ip": ip, "puerto": int(pt), "rol": rol, "es_local": bool(loc)}
                  for n, ip, pt, rol, loc in hosts],
        "reglas": [{"id": int(i), "codigo": c, "tipo": t, "herramientas": h, "campo": ca, "patron": pa,
                    "ambito_hosts": [ah] if ah else [], "ambito_roles": _roles(ar), "es_canario": bool(ec),
                    "ejemplos_coincide": json.loads(ej), "ejemplos_no_coincide": json.loads(no)}
                   for i, c, t, h, ca, pa, ah, ar, ec, ej, no in reglas],
        "respaldos": {n: m.replace(tzinfo=timezone.utc).isoformat() for n, m in respaldos if m is not None},
        "c2_edad_max_s": edad,
    }
    firmado = politica.firmar(doc)
    try:
        politica.validar(firmado)
    except politica.PoliticaIlegible as exc:
        raise ExportacionImposible("politica_invalida") from exc
    return firmado


async def leer(conn):
    async with conn.cursor() as cur:
        await cur.execute(SQL_HOSTS)
        hosts = await cur.fetchall()
        await cur.execute(SQL_REGLAS)
        reglas = await cur.fetchall()
        await cur.execute(SQL_RESPALDOS)
        respaldos = await cur.fetchall()
        await cur.execute(SQL_EDAD)
        fila = await cur.fetchone()
    return hosts, reglas, respaldos, (fila[0] if fila else None)


def escribir_atomico(ruta: Path, doc: dict) -> None:
    ruta = Path(ruta)
    datos = json.dumps(doc, ensure_ascii=True, sort_keys=True, indent=1).encode() + b"\n"
    fd, temporal = tempfile.mkstemp(dir=ruta.parent, prefix=".politica-", suffix=".tmp")
    try:
        os.fchmod(fd, 0o640)
        os.write(fd, datos)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(temporal, ruta)
    except OSError:
        os.unlink(temporal)
        raise
    dir_fd = os.open(ruta.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


async def exportar(ruta: Path, conectar) -> dict:
    conn = await conectar()
    try:
        filas = await leer(conn)
    finally:
        conn.close()
    doc = documento(*filas, datetime.now(timezone.utc).isoformat())
    await asyncio.to_thread(escribir_atomico, ruta, doc)
    return doc


def principal() -> int:
    from jacobs.store import get_conn
    ruta = os.environ.get("JAX_EJECUTOR_POLITICA", "").strip()
    if not ruta.startswith("/"):
        sys.stderr.write("codigo=\"config_falta\" variable=\"JAX_EJECUTOR_POLITICA\"\n")
        return 2
    try:
        doc = asyncio.run(exportar(Path(ruta), get_conn))
    except ExportacionImposible as exc:
        sys.stderr.write(f"codigo=\"{exc.codigo}\"\n")
        return 2
    sys.stdout.write(f"sha256=\"{doc['sha256']}\" reglas={len(doc['reglas'])} hosts={len(doc['hosts'])}\n")
    return 0


if __name__ == "__main__":
    sys.exit(principal())
