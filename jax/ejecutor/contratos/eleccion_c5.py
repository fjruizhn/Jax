# jax/ejecutor/contratos/eleccion_c5.py
"""Quién audita al Ejecutor (C5). Spec 2026-09-15 §4: «nunca el mismo proveedor que
el cerebro: quien produce no aprueba». Y la compuerta que el spec no previó (índice de
SP1, punto 1): con cerebro local, todo auditor es de nube, y la Fase 0 prohibió que la
nube vea datos de clientes. Una misión que toca máquinas con datos de clientes sólo se
audita con auditor local o con `ejecutor.c5_auditor_admite_datos_de_clientes = true`,
que nace en false y abrirla es DECISIÓN de Fernando. Mientras tanto, esa misión NO
arranca. Una máquina que no está en el inventario, o dada de baja, cuenta como con datos
de clientes (cerrado). Una misión sin máquinas no arranca: no hay nada que decidir.

Spec 2026-09-18-auditor-local-opcion.md §4/7 (DECISIÓN DE FERNANDO): con un auditor local
disponible, el auditor SE ELIGE según la máquina de la misión, no una clave global fija.
Máquina sin datos de clientes → `ejecutor.auditor_faceta` (hoy 'thot', el mejor auditor
cuando no hay nada que proteger); máquina con datos de clientes → `ejecutor.auditor_faceta_local`
(el segundo Ollama, sólo CPU, ver ops/ejecutor/ollama-cpu.service). Los dos nombres de
faceta salen de axioma_config — ninguno hardcodeado acá. La compuerta NO se borra: sigue
gobernando el caso que de verdad importa, que el auditor RESUELTO para la misión no sea
local de verdad (provider.is_local en la DB, no el nombre de la faceta) — ver
`validar_eleccion`, que ya distinguía `auditor_es_local` de `admite_datos_de_clientes`
antes de que existiera un auditor local real.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from jax.ejecutor.contratos.fallo import Fallo

CLAVES = ("ejecutor.cerebro_faceta", "ejecutor.auditor_faceta", "ejecutor.auditor_faceta_local",
          "ejecutor.c5_lote_max", "ejecutor.c5_intervalo_s", "ejecutor.c5_max_tokens",
          "ejecutor.c5_auditor_admite_datos_de_clientes")
SQL_CONFIG = ("SELECT config_key, config_value FROM axioma_config WHERE config_key IN "
              f"({', '.join(['%s'] * len(CLAVES))})")
# Por clave primaria: sólo las máquinas de la misión.
SQL_HOSTS = "SELECT nombre, con_datos_de_clientes FROM ejecutor_host WHERE activo = 1 AND nombre IN ({})"


@dataclass(frozen=True)
class ConfigC5:
    cerebro_faceta: str
    auditor_faceta: str
    auditor_faceta_local: str
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
    return ConfigC5(filas["ejecutor.cerebro_faceta"].strip(), filas["ejecutor.auditor_faceta"].strip(),
                    filas["ejecutor.auditor_faceta_local"].strip(), lote, intervalo, tokens, admite == "true")


def elegir_auditor_faceta(cfg: ConfigC5, *, hay_datos_de_clientes: bool) -> str:
    """Qué faceta audita ESTA misión. Sin nombre de faceta hardcodeado: los dos candidatos
    vienen de `cfg` (axioma_config). La verificación de que el elegido de verdad sirve
    -- proveedor distinto del cerebro, y si hace falta, local de verdad -- la sigue
    haciendo `validar_eleccion`/`verificar_eleccion`; esta función sólo decide CUÁL mirar."""
    return cfg.auditor_faceta_local if hay_datos_de_clientes else cfg.auditor_faceta


def sensibles(hosts_mision, hosts_con_clientes, hosts_conocidos) -> frozenset:
    """Máquinas de la misión que exigen auditor local (o compuerta abierta): con datos de
    clientes, o fuera del inventario/dadas de baja (cuentan como con datos de clientes,
    cerrado). Un mismo hecho gobierna dos preguntas: si la misión NECESITA auditor local
    (`elegir_auditor_faceta`) y si, sin uno, la compuerta la bloquea (`validar_eleccion`)."""
    return frozenset(h for h in hosts_mision if h in hosts_con_clientes or h not in hosts_conocidos)


def validar_proveedores(*, proveedor_cerebro: str, proveedor_auditor: str) -> tuple:
    """La mitad de la elección que no depende de la misión: quien produce no aprueba.
    La usa también el arranque del Ejecutor sin misión (plan 6)."""
    if not (proveedor_cerebro or "").strip() or not (proveedor_auditor or "").strip():
        return (Fallo("c5", "proveedor_desconocido"),)
    if proveedor_cerebro == proveedor_auditor:
        return (Fallo("c5", "auditor_mismo_proveedor_que_el_cerebro"),)
    return ()


def validar_eleccion(*, proveedor_cerebro: str, proveedor_auditor: str, auditor_es_local: bool,
                     admite_datos_de_clientes: bool, hosts_mision, hosts_con_clientes, hosts_conocidos) -> tuple:
    fallos = list(validar_proveedores(proveedor_cerebro=proveedor_cerebro, proveedor_auditor=proveedor_auditor))
    if not hosts_mision:
        fallos.append(Fallo("c5", "mision_sin_maquinas"))
    sens = sensibles(hosts_mision, hosts_con_clientes, hosts_conocidos)
    if sens and not (auditor_es_local or admite_datos_de_clientes):
        fallos.append(Fallo("c5", "auditor_no_admite_datos_de_clientes", (("hosts", tuple(sorted(sens))),)))
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


async def elegir_y_resolver_auditor(conn, *, cfg: ConfigC5, hosts_mision, resolve_facet):
    """Punto único de elección + resolución del auditor. Lo usan los TRES consumidores
    reales de C5 -- el arranque (la compuerta, `arranque.py::p_c5`), el turno del cerebro
    (`mision_servicio.py::auditar`) y el vigía en vivo (`vigia_servicio.py::_principal`) --
    así los tres auditan la MISMA misión con la MISMA faceta, nunca una decisión tomada
    dos veces con resultados distintos. `resolve_facet` se recibe INYECTADA (vive en
    `facet_resolver`, que este módulo no importa: la elección es pura sobre datos de la
    DB, la resolución de red es responsabilidad de quien llama) -- eso además la hace
    fácil de probar con un doble de prueba, sin credenciales ni HTTP real.

    Devuelve `(faceta_resuelta, hosts_con_clientes, hosts_conocidos)`: los dos últimos
    quedan para quien también necesite correr la compuerta (`validar_eleccion`) sin
    repetir la consulta de hosts."""
    if hosts_mision is None:
        con_clientes, conocidos = frozenset(), frozenset()
    else:
        con_clientes, conocidos = await hosts_de_la_mision(conn, hosts_mision)
    hay_datos = bool(sensibles(hosts_mision or frozenset(), con_clientes, conocidos))
    faceta = await resolve_facet(elegir_auditor_faceta(cfg, hay_datos_de_clientes=hay_datos))
    return faceta, con_clientes, conocidos


async def verificar_eleccion(conn, *, cfg: ConfigC5, proveedor_cerebro: str, proveedor_auditor: str,
                             hosts_mision) -> tuple:
    """Lo que el arranque (plan 6) corre antes de cada misión: si devuelve fallos, no arranca."""
    con_clientes, conocidos = await hosts_de_la_mision(conn, hosts_mision)
    return validar_eleccion(proveedor_cerebro=proveedor_cerebro, proveedor_auditor=proveedor_auditor,
                            auditor_es_local=await es_local(conn, proveedor_auditor),
                            admite_datos_de_clientes=cfg.admite_datos_de_clientes, hosts_mision=frozenset(hosts_mision),
                            hosts_con_clientes=con_clientes, hosts_conocidos=conocidos)
