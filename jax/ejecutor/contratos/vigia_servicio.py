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
import shlex
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

from jax.ejecutor.contratos import arranque, cuenta_axioma, formato, huella, pausa, politica, vigia
from jax.ejecutor.contratos import auditor as A

log = logging.getLogger("ejecutor.vigia_servicio")
VARIABLE_LATIDO_CADA_S = "JAX_EJECUTOR_VIGIA_LATIDO_CADA_S"
_TOPE_HUELLA_S = 30


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


async def _huella_de_cada_host(hosts_con_sudo: tuple, *, tomar_huella) -> dict:
    return {h: await tomar_huella(h) for h in hosts_con_sudo}


async def _verificar_huellas_al_cierre(pausa_ruta: Path, mision: Mision, huellas_iniciales: dict,
                                       hosts_con_sudo: tuple, *, tomar_huella, pausar) -> None:
    """M-1/M-2 (ronda 3): huella AL CERRAR contra `huellas_iniciales` (tomada al abrir,
    por `correr_mision`). Un hallazgo no declarado en el texto de la misión pone la
    pausa. Fail-soft por host, a propósito: si TOMAR la huella de cierre de UN host
    falla (máquina caída, ssh que no responde), se registra y no pausa por eso solo --
    y el resto de los hosts se sigue verificando igual. Perder la verificación no es lo
    mismo que encontrar un cambio, y esto no reemplaza a C6, que ya exige que la
    máquina esté viva."""
    for h in hosts_con_sudo:
        try:
            despues = await tomar_huella(h)
        except Exception as exc:  # fail-soft: ver docstring de la función
            log.error("vigia_servicio huella_no_verificada host=%s tipo=%s", h, type(exc).__name__)
            continue
        antes = huellas_iniciales.get(h)
        if antes is None:  # no se pudo tomar AL ABRIR (mismo criterio fail-soft): nada que comparar
            continue
        encontrados = huella.hallazgos(antes, despues, mision.texto)
        if encontrados:
            log.critical("vigia_servicio huella_cambio_no_declarado host=%s lineas=%d", h, len(encontrados))
            await asyncio.to_thread(pausar, pausa_ruta, {
                "origen": "huella", "motivo": "huella_cambio_no_declarado",
                "host": h, "detalle": list(encontrados[:20])})


async def correr_mision(ctx: arranque.Contexto, mision: Mision, *, latido_cada_s: float, lote_max: int,
                        intervalo_s: float, auditar, fin: asyncio.Event, exigir=arranque.exigir_contratos,
                        vigilar=vigia.vigilar, maquinas: tuple, hosts_con_sudo: tuple = (),
                        tomar_huella=None, pausar=pausa.poner_pausa) -> None:
    """Lanza ContratosNoVerificados sin haber latido nunca si un contrato no está vivo.

    `hosts_con_sudo`/`tomar_huella` (M-1/M-2, ronda 3): si se dan los dos, se toma una
    huella de cada host ANTES de latir y otra AL CERRAR (fin normal); un cambio no
    declarado en el texto de la misión pone la pausa. `tomar_huella=None` (el default):
    sin huella -- así los llamadores que no la necesitan (o corren en un entorno sin
    ssh/sudo, como los tests que no la ejercitan) no cambian de comportamiento."""
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
    huellas_iniciales = {}
    if tomar_huella is not None and hosts_con_sudo:
        huellas_iniciales = await _huella_de_cada_host(hosts_con_sudo, tomar_huella=tomar_huella)
    log.info("vigia_servicio mision_abierta desde_byte=%s hosts=%s", desde, ",".join(sorted(mision.hosts)))
    await vigilar(cfg, auditar, fin)
    # Sólo en el fin normal: con una excepción el latido se deja envejecer y vigia.py ya puso la pausa.
    await asyncio.to_thread(_borrar_latido, ctx.latido)
    if tomar_huella is not None and hosts_con_sudo:
        await _verificar_huellas_al_cierre(ctx.pausa, mision, huellas_iniciales, hosts_con_sudo,
                                           tomar_huella=tomar_huella, pausar=pausar)
    log.info("vigia_servicio mision_cerrada")


async def _principal(ruta_mision: Path) -> int:
    from facet_resolver import resolve_facet
    from jacobs.store import conexion
    from jax.ejecutor.contratos import auditor_cliente, eleccion_c5

    mision = mision_desde_bytes(await asyncio.to_thread(ruta_mision.read_bytes))
    ctx = arranque.contexto_desde_entorno(os.environ, mision.hosts)
    latido_cada_s = latido_cada_desde_entorno(os.environ, ctx.latido_max_s)
    # Spec 2026-09-18-auditor-local-opcion.md §4: mismo punto único que
    # mision_servicio.py::auditar y arranque.py::p_c5 -- las tres piezas de C5 auditan la
    # MISMA misión con la MISMA faceta, elegida según si sus máquinas cargan datos de
    # clientes.
    async with conexion(desechable=True) as conn:
        cfg = await eleccion_c5.leer_config(conn)
        auditor_f, _, _ = await eleccion_c5.elegir_y_resolver_auditor(
            conn, cfg=cfg, hosts_mision=mision.hosts, resolve_facet=resolve_facet)
    doc = json.loads(await asyncio.to_thread(ctx.cuenta.politica.read_bytes))
    hosts_pol = {h.nombre: h for h in politica.validar(doc).hosts}
    maquinas = A.maquinas_de(politica.validar(doc).hosts, mision.hosts)
    # M-1/M-2 (ronda 3): "con sudo" hoy equivale a "remota" -- hall9000 es la única local
    # y quedó sudo=false (M2, jaula bwrap con NoNewPrivs); las tres remotas tienen sudo
    # real (Fase 3). `politica.Host` no trae un campo `sudo` propio (eso vive en
    # maquinas.toml, host-bound, Fase 0, no se lee en runtime) -- si el día de mañana
    # una máquina remota pierde el sudo o una local lo gana, este criterio hay que
    # revisarlo junto con esa migración, no antes.
    hosts_con_sudo = tuple(sorted(n for n in mision.hosts if n in hosts_pol and not hosts_pol[n].es_local))

    async def auditar(lote):
        return await auditor_cliente.auditar(lote, faceta=auditor_f, max_tokens=cfg.max_tokens)

    async def tomar_huella(nombre_host: str) -> huella.Huella:
        h = hosts_pol[nombre_host]
        remoto = (f"ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=yes -p {int(h.puerto)} "
                 f"{ctx.cuenta.nombre}@{shlex.quote(h.ip)} {shlex.quote(huella.comando_huella(ctx.cuenta.nombre))}")
        rc, salida, errores = await cuenta_axioma.correr_en_la_cuenta(ctx.cuenta, remoto, tope_s=_TOPE_HUELLA_S)
        if rc != 0:
            raise RuntimeError(f"huella_rc_{rc}: {errores.decode(errors='replace')[:200]}")
        return huella.huella_desde_salida(nombre_host, salida)

    fin = asyncio.Event()
    loop = asyncio.get_running_loop()
    for senal in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(senal, fin.set)
    try:
        await correr_mision(ctx, mision, latido_cada_s=latido_cada_s, lote_max=cfg.lote_max,
                            intervalo_s=cfg.intervalo_s, auditar=auditar, fin=fin, maquinas=maquinas,
                            hosts_con_sudo=hosts_con_sudo, tomar_huella=tomar_huella)
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
