# jax/ejecutor/contratos/eleccion_c5.py
"""Quién audita al Ejecutor (C5). Spec 2026-09-15 §4: «nunca el mismo proveedor que
el cerebro: quien produce no aprueba». Y la compuerta que el spec no previó (índice de
SP1, punto 1): con cerebro local, todo auditor es de nube, y la Fase 0 prohibió que la
nube vea datos de clientes. Una misión que toca máquinas con datos de clientes sólo se
audita con auditor local o con `ejecutor.c5_auditor_admite_datos_de_clientes = true`,
que nace en false y abrirla es DECISIÓN de Fernando. Mientras tanto, esa misión NO
arranca. Una máquina que no está en el inventario, o dada de baja, cuenta como con datos
de clientes (cerrado). Una misión sin máquinas no arranca: no hay nada que decidir.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from jax.ejecutor.contratos.fallo import Fallo

CLAVES = ("ejecutor.cerebro_faceta", "ejecutor.auditor_faceta", "ejecutor.c5_lote_max", "ejecutor.c5_intervalo_s",
          "ejecutor.c5_max_tokens", "ejecutor.c5_auditor_admite_datos_de_clientes")
SQL_CONFIG = ("SELECT config_key, config_value FROM axioma_config WHERE config_key IN "
              f"({', '.join(['%s'] * len(CLAVES))})")
# Por clave primaria: sólo las máquinas de la misión.
SQL_HOSTS = "SELECT nombre, con_datos_de_clientes FROM ejecutor_host WHERE activo = 1 AND nombre IN ({})"


@dataclass(frozen=True)
class ConfigC5:
    cerebro_faceta: str
    auditor_faceta: str
    lote_max: int
    intervalo_s: float
    max_tokens: int
    admite_datos_de_clientes: bool


def config_desde_filas(filas: dict) -> ConfigC5:
    faltan = [c for c in CLAVES if not (filas.get(c) or "").strip()]
    if faltan:
        raise ValueError("config_c5_incompleta", faltan)
    admite = filas["ejecutor.c5_auditor_admite_datos_de_clientes"].strip()
    if admite not in ("true", "false"):
        raise ValueError("config_c5_invalida", "ejecutor.c5_auditor_admite_datos_de_clientes")
    lote, intervalo, tokens = (int(filas["ejecutor.c5_lote_max"]), float(filas["ejecutor.c5_intervalo_s"]),
                               int(filas["ejecutor.c5_max_tokens"]))
    if lote <= 0 or not math.isfinite(intervalo) or intervalo <= 0 or tokens <= 0:
        raise ValueError("config_c5_invalida", "numeros")
    return ConfigC5(filas["ejecutor.cerebro_faceta"].strip(), filas["ejecutor.auditor_faceta"].strip(), lote,
                    intervalo, tokens, admite == "true")


def validar_eleccion(*, proveedor_cerebro: str, proveedor_auditor: str, auditor_es_local: bool,
                     admite_datos_de_clientes: bool, hosts_mision, hosts_con_clientes, hosts_conocidos) -> tuple:
    fallos = []
    if not (proveedor_cerebro or "").strip() or not (proveedor_auditor or "").strip():
        fallos.append(Fallo("c5", "proveedor_desconocido"))
    elif proveedor_cerebro == proveedor_auditor:
        fallos.append(Fallo("c5", "auditor_mismo_proveedor_que_el_cerebro"))
    if not hosts_mision:
        fallos.append(Fallo("c5", "mision_sin_maquinas"))
    sensibles = sorted(h for h in hosts_mision if h in hosts_con_clientes or h not in hosts_conocidos)
    if sensibles and not (auditor_es_local or admite_datos_de_clientes):
        fallos.append(Fallo("c5", "auditor_no_admite_datos_de_clientes", (("hosts", tuple(sensibles)),)))
    return tuple(fallos)


async def leer_config(conn) -> ConfigC5:
    async with conn.cursor() as cur:
        await cur.execute(SQL_CONFIG, CLAVES)
        return config_desde_filas(dict(await cur.fetchall()))


async def es_local(conn, provider_id: str) -> bool:
    async with conn.cursor() as cur:
        await cur.execute("SELECT is_local FROM provider WHERE id = %s", (provider_id,))
        fila = await cur.fetchone()
    if fila is None:
        raise ValueError("proveedor_desconocido", provider_id)
    return bool(fila[0])


async def hosts_de_la_mision(conn, hosts) -> tuple:
    """(con datos de clientes, conocidos y activos), sólo entre las máquinas de la misión."""
    hosts = tuple(sorted(hosts))
    if not hosts:
        return frozenset(), frozenset()
    async with conn.cursor() as cur:
        await cur.execute(SQL_HOSTS.format(", ".join(["%s"] * len(hosts))), hosts)
        filas = await cur.fetchall()
    return frozenset(n for n, c in filas if c), frozenset(n for n, _ in filas)


async def verificar_eleccion(conn, *, cfg: ConfigC5, proveedor_cerebro: str, proveedor_auditor: str,
                             hosts_mision) -> tuple:
    """Lo que el arranque (plan 6) corre antes de cada misión: si devuelve fallos, no arranca."""
    con_clientes, conocidos = await hosts_de_la_mision(conn, hosts_mision)
    return validar_eleccion(proveedor_cerebro=proveedor_cerebro, proveedor_auditor=proveedor_auditor,
                            auditor_es_local=await es_local(conn, proveedor_auditor),
                            admite_datos_de_clientes=cfg.admite_datos_de_clientes, hosts_mision=frozenset(hosts_mision),
                            hosts_con_clientes=con_clientes, hosts_conocidos=conocidos)
