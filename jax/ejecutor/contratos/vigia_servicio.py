# jax/ejecutor/contratos/vigia_servicio.py
"""El arranque de una misión del Ejecutor: exige los seis contratos, lanza el vigía de
C5 y SÓLO ENTONCES el proxy sirve (plan 6, con la enmienda del plan 4).

El proxy de C3 responde 423 a todo mientras no haya un vigía que lata
(`pausa.latido_fresco`). Este módulo es el único que late en producción, y lo hace
después de `arranque.exigir_contratos` con las máquinas de la misión. Orden:
1. exigir los contratos para `hosts` (con la compuerta de datos de clientes);
2. `desde_byte` = tamaño del registro en ese momento (la misión empieza acá);
3. el vigía corre y late → el proxy empieza a servir;
4. SIGTERM (fin normal): el vigía audita lo pendiente, termina, y el latido se BORRA,
   para que el proxy vuelva a 423 en el acto y no cuando el latido envejezca.
Cualquier otra salida del vigía pone la pausa `vigia_caido` (vigia.py); un SIGKILL deja
el latido envejecer y el proxy vuelve a 423 en `JAX_EJECUTOR_VIGIA_LATIDO_MAX_S`.

La unidad es `ops/ejecutor/ejecutor-vigia@.service` (`%i` = id de la misión); la misión
es un JSON `{"mision": texto, "hosts": [nombres]}` en `JAX_EJECUTOR_MISIONES/<id>.json`.
Salida: códigos `clave=valor` (formato.py), nunca texto para personas.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

from jax.ejecutor.contratos import arranque, cuenta_axioma, formato, pausa, politica, vigia
from jax.ejecutor.contratos import auditor as A

log = logging.getLogger("ejecutor.vigia_servicio")
VARIABLE_LATIDO_CADA_S = "JAX_EJECUTOR_VIGIA_LATIDO_CADA_S"


class MisionIlegible(ValueError):
    """`args[0]` es un código estable."""


@dataclass(frozen=True)
class Mision:
    texto: str
    hosts: frozenset


def mision_desde_bytes(datos: bytes) -> Mision:
    try:
        doc = json.loads(datos)
    except ValueError:
        raise MisionIlegible("mision_no_es_json") from None
    if not isinstance(doc, dict):
        raise MisionIlegible("mision_no_es_objeto")
    texto, hosts = doc.get("mision"), doc.get("hosts")
    if not isinstance(texto, str) or not texto.strip():
        raise MisionIlegible("mision_sin_texto")
    if (not isinstance(hosts, list) or not hosts
            or not all(isinstance(h, str) and h.strip() for h in hosts)):
        raise MisionIlegible("mision_sin_maquinas")
    return Mision(texto.strip(), frozenset(h.strip() for h in hosts))


def latido_cada_desde_entorno(env, latido_max_s: float) -> float:
    try:
        cada = float(env[VARIABLE_LATIDO_CADA_S])
    except (KeyError, ValueError):
        raise ValueError("latido_cada_invalido") from None
    if not 0 < cada < latido_max_s:
        raise ValueError("latido_cada_invalido")
    return cada


def _borrar_latido(ruta: Path) -> None:
    try:
        os.unlink(ruta)
    except FileNotFoundError:  # fail-soft: sin latido que borrar el proxy ya está en 423; es el estado buscado
        pass


async def correr_mision(ctx: arranque.Contexto, mision: Mision, *, latido_cada_s: float, lote_max: int,
                        intervalo_s: float, auditar, fin: asyncio.Event, exigir=arranque.exigir_contratos,
                        vigilar=vigia.vigilar, maquinas: tuple) -> None:
    """Lanza ContratosNoVerificados sin haber latido nunca si un contrato no está vivo."""
    if ctx.hosts_mision != mision.hosts:
        raise ValueError("contexto_de_otra_mision")
    await exigir(ctx)
    if fin.is_set():  # lo pararon mientras se verificaban los contratos: no se abre nada
        log.info("vigia_servicio mision_cancelada_antes_de_abrir")
        return
    desde = (await asyncio.to_thread(os.stat, ctx.registro)).st_size
    cfg = vigia.ConfigVigia(registro=ctx.registro, desde_byte=desde, mision=mision.texto, lote_max=lote_max,
                            intervalo_s=intervalo_s, pausa=ctx.pausa, latido=ctx.latido, latido_cada_s=latido_cada_s,
                            maquinas=maquinas)
    log.info("vigia_servicio mision_abierta desde_byte=%s hosts=%s", desde, ",".join(sorted(mision.hosts)))
    await vigilar(cfg, auditar, fin)
    # Sólo en el fin normal: con una excepción el latido se deja envejecer y vigia.py ya puso la pausa.
    await asyncio.to_thread(_borrar_latido, ctx.latido)
    log.info("vigia_servicio mision_cerrada")


async def _principal(ruta_mision: Path) -> int:
    from facet_resolver import resolve_facet
    from jacobs.store import conexion
    from jax.ejecutor.contratos import auditor_cliente, eleccion_c5

    mision = mision_desde_bytes(await asyncio.to_thread(ruta_mision.read_bytes))
    ctx = arranque.contexto_desde_entorno(os.environ, mision.hosts)
    latido_cada_s = latido_cada_desde_entorno(os.environ, ctx.latido_max_s)
    async with conexion(desechable=True) as conn:
        cfg = await eleccion_c5.leer_config(conn)
    auditor_f = await resolve_facet(cfg.auditor_faceta)
    doc = json.loads(await asyncio.to_thread(ctx.cuenta.politica.read_bytes))
    maquinas = A.maquinas_de(politica.validar(doc).hosts, mision.hosts)

    async def auditar(lote):
        return await auditor_cliente.auditar(lote, faceta=auditor_f, max_tokens=cfg.max_tokens)

    fin = asyncio.Event()
    loop = asyncio.get_running_loop()
    for senal in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(senal, fin.set)
    try:
        await correr_mision(ctx, mision, latido_cada_s=latido_cada_s, lote_max=cfg.lote_max,
                            intervalo_s=cfg.intervalo_s, auditar=auditar, fin=fin, maquinas=maquinas)
    except arranque.ContratosNoVerificados as exc:
        for f in exc.fallos:
            print(formato.campos((("contrato", f.contrato), ("codigo", f.codigo)) + tuple(f.datos)), flush=True)
        print(formato.campos((("arranco", False),)), flush=True)
        return 1
    print(formato.campos((("arranco", True), ("cerrada", True))), flush=True)
    return 0


def principal(argv) -> int:
    logging.basicConfig(level=logging.INFO)
    if len(argv) != 1:
        print(formato.campos((("codigo", "uso"), ("argumentos", "ruta_de_la_mision"))), file=sys.stderr)
        return 2
    try:
        return asyncio.run(_principal(Path(argv[0])))
    except (MisionIlegible, ValueError, KeyError, OSError, pausa.PausaSinConfigurar,
            cuenta_axioma.CuentaSinConfigurar) as exc:
        print(formato.campos((("arranco", False), ("codigo", "configuracion_invalida"),
                              ("tipo", type(exc).__name__), ("detalle", str(exc.args[0]) if exc.args else ""))),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(principal(sys.argv[1:]))
