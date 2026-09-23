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

Este módulo NO es una unidad systemd (la plantilla `ejecutor-vigia@.service` que lo
prometía se retiró el 2026-09-22: código muerto, el journal nunca mostró un solo
arranque suyo -- ver DEUDA.md). Lo lanza `abrir_vigia` (`jax/ejecutor/mision_servicio.py`)
como SUBPROCESO DIRECTO por cada turno (`python -m jax.ejecutor.contratos.vigia_servicio
<ruta_mision>`), heredando la identidad del proceso que lo lanza.

CORREGIDO (bug de producción, jax#260, 2026-09-22; este párrafo decía lo contrario y
ERA FALSO): en producción ese proceso es `jax-platform`, y desde el 2026-09-17 (decisión
de Fernando, cuenta de servicio) `jax-platform.service` corre como `jaxsvc`
(`/etc/systemd/system/jax-platform.service.d/cuenta-de-servicio.conf: User=jaxsvc`),
NO como `fruiz` -- así que este módulo corre como `jaxsvc`, no como el administrador. Por
eso M-1/`_tomar_huella` más abajo ya NO arma el ssh con `revocacion.argv_admin` (sin
`-i`, resolución de identidad por DEFAULT): `jaxsvc` no puede leer `~fruiz/.ssh/*` (600,
dueño `fruiz`), y ese camino medía `vigia_no_latio=true rc=2` en producción -- el
Ejecutor bloqueado por completo. Usa en cambio `huella.argv_huella_servicio`, con una
llave PROPIA del servicio (`JAX_EJECUTOR_HUELLA_LLAVE`, jaxsvc:jaxsvc) autorizada por
comando forzado en cada remota (ver el docstring de ese módulo). El CONTROLADOR sigue
siendo, nominalmente, el mismo administrador (`fruiz`, vía `JAX_EJECUTOR_ADMIN_USUARIO`)
-- lo que cambió es la CREDENCIAL con la que se llega a esa cuenta, no de qué cuenta es
huésped ni que siga sin ser `axioma`.
La misión es un JSON `{"mision": texto, "hosts": [nombres]}` en
`JAX_EJECUTOR_MISIONES/<id>.json` (`<id>` = `Turno.id_vigia`, `<mision_id>-t<n>`, o el
UUID bare de la misión de humo). Salida: códigos `clave=valor` (formato.py), nunca texto
para personas.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

from jax.ejecutor.contratos import arranque, cuenta_axioma, formato, huella, pausa, politica, vigia
from jax.ejecutor.contratos import auditor as A

log = logging.getLogger("ejecutor.vigia_servicio")
VARIABLE_LATIDO_CADA_S = "JAX_EJECUTOR_VIGIA_LATIDO_CADA_S"
_TOPE_HUELLA_S = 30


def _hint_aceptar(host: str, mision_id: str = "<mision_id>") -> str:
    """M-1 (ronda 7): el mensaje de la pausa dice CÓMO salir -- no sólo qué pasó. Ver
    docs/ejecutor-huella-aceptar.md para el procedimiento completo."""
    return ("aceptar con: python -m jax.ejecutor.contratos.huella aceptar "
            f"--host {host} --mision {mision_id}  (ver docs/ejecutor-huella-aceptar.md)")


_SUFIJO_TURNO = re.compile(r"-t\d+$")


class HuellaHuerfanaNoResuelta(RuntimeError):
    """M-1 (ronda 6): una deuda de verificación huérfana (de esta u otra misión) que se
    cortó entre abrir y cerrar con éxito, y que la revisión al arranque NO pudo cerrar
    limpia (cambió, o no se pudo medir). La misión NUEVA no abre. `args[0]` es el host."""


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


def mision_id_desde_ruta(ruta_mision: Path) -> str:
    """El `mision_id` (ronda 4, M-1): la LÍNEA BASE de la huella es la de la APERTURA DE
    LA MISIÓN, no la del turno anterior -- para eso hace falta un identificador que sea
    el MISMO en todos los turnos. `ruta_mision.stem` es `<mision_id>` (misiones sueltas,
    p.ej. `mision_de_humo.py`) o `<mision_id>-t<n>` (`Turno.id_vigia`, mision_servicio.py):
    se quita el sufijo de turno si está, y lo que queda es el `mision_id` en los dos
    casos."""
    return _SUFIJO_TURNO.sub("", ruta_mision.stem)


def ruta_huella(misiones: Path, mision_id: str, host: str) -> Path:
    return huella.ruta_huella(misiones, mision_id, host)


async def verificar_huellas_huerfanas(misiones: Path, host: str, *, tomar_huella, pausar, pausa_ruta,
                                      rutas_extra: tuple = ()) -> bool:
    """M-1 (ronda 6; estados con nombre desde ronda 7). Antes de que CUALQUIER misión
    nueva abra su propia huella en `host`, revisa TODAS las marcas de OTRA vuelta --
    de esta misma misión (un turno que se cortó) o de otra:
    - `REPORTADA`: ya se comparó, salió sucia y ya pausó -- bloquea de una, SIN volver
      a medir (eso es tarea de `python -m jax.ejecutor.contratos.huella aceptar`, no de
      acá). El único camino de salida es la aceptación explícita.
    - `CERRADA`: ya se comparó limpia -- no es deuda, se salta.
    - `ABIERTA`: la comparación de cierre nunca se hizo (kill, reinicio) -- la ÚNICA que
      esta función vuelve a medir. Si cambió (o la huella de ahora sale vacía), pasa a
      `REPORTADA` (con diff) y pausa; si no se puede medir, PAUSA y deja la marca
      `ABIERTA` (se reintentará después); si sale limpia, pasa a `CERRADA`.
    Marca ilegible: fail-closed, pausa y no sigue. Devuelve `False` si la misión NO
    debe abrir."""
    for ruta in sorted(Path(misiones).glob(f"*/huella/{host}.json")):
        mision_id_de_la_ruta = ruta.parent.parent.name  # misiones/<mision_id>/huella/<host>.json
        try:
            marca = huella.leer_marca(ruta)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.error("vigia_servicio huella_huerfana_ilegible ruta=%s tipo=%s", ruta, type(exc).__name__)
            await asyncio.to_thread(pausar, pausa_ruta, {
                "origen": "huella", "motivo": "huella_no_medible", "host": host, "mision_id": mision_id_de_la_ruta,
                "detalle": ["huerfana_ilegible", str(ruta)]})
            return False
        if marca.estado == huella.CERRADA:
            continue
        if marca.estado == huella.REPORTADA:
            log.critical("vigia_servicio huella_reportada_sin_aceptar host=%s ruta=%s", host, ruta)
            await asyncio.to_thread(pausar, pausa_ruta, {
                "origen": "huella", "motivo": "huella_reportada_sin_aceptar", "host": host,
                "mision_id": mision_id_de_la_ruta, "detalle": [_hint_aceptar(host, mision_id_de_la_ruta)]})
            return False
        antes = marca.huella
        try:
            despues = await tomar_huella(host)
        except Exception as exc:  # fail-soft: no medible se reporta como huella_no_medible y PAUSA (fail-closed) -- nunca se sigue como si nada
            log.error("vigia_servicio huella_no_medible host=%s motivo=huerfana tipo=%s", host, type(exc).__name__)
            await asyncio.to_thread(pausar, pausa_ruta, {
                "origen": "huella", "motivo": "huella_no_medible", "host": host, "mision_id": mision_id_de_la_ruta,
                "detalle": ["huerfana", type(exc).__name__]})
            return False
        rutas = huella.RUTAS_DECLARADAS_POR_DEFAULT + rutas_extra
        if not huella.huella_valida(despues, rutas=rutas) or huella.cambio(antes, despues):
            log.critical("vigia_servicio huella_cambio_no_declarado host=%s motivo=huerfana", host)
            detalle = tuple(["huella_de_ahora_vacia"]
                            if not huella.huella_valida(despues, rutas=rutas)
                            else huella.lineas_agregadas_o_quitadas(antes, despues))
            await asyncio.to_thread(huella.escribir_marca, ruta,
                                    huella.Marca(huella=antes, estado=huella.REPORTADA, diff=detalle))
            await asyncio.to_thread(pausar, pausa_ruta, {
                "origen": "huella", "motivo": "huella_cambio_no_declarado", "host": host,
                "mision_id": mision_id_de_la_ruta,
                "detalle": ["huerfana"] + list(detalle[:20]) + [_hint_aceptar(host, mision_id_de_la_ruta)]})
            return False
        await asyncio.to_thread(huella.escribir_marca, ruta, huella.Marca(huella=antes, estado=huella.CERRADA))
    return True


async def huella_de_apertura_de_la_mision(*, misiones: Path, mision_id: str, host: str, tomar_huella,
                                          pausar, pausa_ruta, rutas_extra: tuple = ()) -> huella.Huella | None:
    """La huella de APERTURA de la MISIÓN (ronda 4, M-1) -- NO la del turno. Primero
    resuelve cualquier deuda huérfana en `host` (`verificar_huellas_huerfanas`); si esa
    revisión encuentra un problema, ESTA misión tampoco abre (devuelve `None`) --
    reportar el hallazgo no alcanza para blanquear la apertura nueva.

    Si ya hay una línea base persistida para `(mision_id, host)`, la carga TAL CUAL y
    NUNCA la vuelve a tomar: es el turno 1 el que la fija, y todos los turnos
    siguientes comparan contra ESE momento cero. Así un cambio que el turno N no llegó
    a ver no queda blanqueado cuando el turno N+1 vuelve a mirar.

    Deliberadamente SIN try/except sobre `tomar_huella` (M-3/M-4, rondas 4/6): si
    revienta la primera vez, la excepción se propaga -- una huella de apertura que no
    se pudo tomar no es "sin cambios", es que la misión no debe abrir. MINOR (ronda 6):
    una huella de apertura vacía (no parsea/no midió nada real) es el mismo fallo.

    MAJOR-L (ronda 7, auditoría adversarial 2026-09-22): la marca YA PERSISTIDA (el
    camino de arriba, "si ya hay una línea base... la carga TAL CUAL") se cargaba sin
    validarla contra las rutas EXIGIDAS de HOY -- sólo el camino "sin marca todavía"
    (el `except FileNotFoundError`) llamaba a `huella_valida()`. Si lo que hace falta
    vigilar cambió entre el turno 1 (que fijó esa línea base) y un turno posterior --
    el caso concreto: `admin_usuario` cambia entre turnos, así que
    `ruta_authorized_keys_admin` resuelve OTRA ruta -- la apertura seguía abriendo con
    una línea base que ya no representa lo que HOY hay que exigir, sin que nadie lo
    notara. Ahora las DOS ramas pasan por el mismo chequeo."""
    ok = await verificar_huellas_huerfanas(misiones, host, tomar_huella=tomar_huella, pausar=pausar,
                                           pausa_ruta=pausa_ruta, rutas_extra=rutas_extra)
    if not ok:
        return None
    ruta = ruta_huella(misiones, mision_id, host)
    rutas_exigidas = huella.RUTAS_DECLARADAS_POR_DEFAULT + rutas_extra
    try:
        marca = await asyncio.to_thread(huella.leer_marca, ruta)
        h = marca.huella
    except FileNotFoundError:
        h = await tomar_huella(host)
        if not huella.huella_valida(h, rutas=rutas_exigidas):
            raise RuntimeError("huella_apertura_vacia")
    else:
        if not huella.huella_valida(h, rutas=rutas_exigidas):
            raise RuntimeError("huella_apertura_persistida_invalida")
    await asyncio.to_thread(huella.escribir_marca, ruta, huella.Marca(huella=h, estado=huella.ABIERTA))
    return h


def hosts_con_sudo(hosts_mision, hosts_pol: dict) -> tuple:
    """Las máquinas de la misión que son remotas -- M-1/M-2 (ronda 3): "con sudo" hoy
    equivale a "remota" (comentario largo en `_principal`: hall9000 es la única local y
    quedó `sudo=false` en `maquinas.toml` -- y desde la noche del 2026-09-22 (MAJOR-2,
    ronda 9) eso también es literal: a `axioma` se le quitó el sudo en hall9000, no
    sólo quedó inerte por la jaula. Ver M-3/MAJOR-2, ronda 9, en ese comentario). Una
    máquina de la misión que no está en la política se omite acá -- ya la rechaza
    `arranque.exigir_contratos` antes de llegar a este punto."""
    return tuple(sorted(n for n in hosts_mision if n in hosts_pol and not hosts_pol[n].es_local))


async def _verificar_huellas_al_cierre(pausa_ruta: Path, huellas_iniciales: dict, hosts_con_sudo: tuple,
                                       misiones: Path, mision_id: str, *, tomar_huella, pausar,
                                       rutas_extra: tuple = ()) -> tuple:
    """«El CIERRE falla cerrado». Si la huella de cierre de un host NO SE PUEDE TOMAR
    (ssh caído, sudo denegado, lo que sea), sale vacía, o si no hubo huella de apertura
    que comparar, es un hallazgo `huella_no_medible` y PONE LA PAUSA. Sin declarado
    (ronda 6): cualquier cambio es `huella_cambio_no_declarado`, sin excepción. Una
    comparación limpia borra la marca `pendiente` (M-1) -- deja de ser deuda para el
    próximo arranque. Queda visible en el resultado del turno sin tocar el formato de
    salida del vigía: `pausar()` escribe en `ctx.pausa`, y `mision.py::correr_turno` YA
    lee esa pausa después de cerrar el vigía (`codigo = "pausa_puesta"` si está
    puesta). Devuelve `((host, motivo), ...)` -- vacío si todo midió limpio."""
    motivos = []
    for h in hosts_con_sudo:
        antes = huellas_iniciales.get(h)
        if antes is None:
            log.error("vigia_servicio huella_no_medible host=%s motivo=sin_apertura", h)
            motivos.append((h, "huella_no_medible"))
            await asyncio.to_thread(pausar, pausa_ruta, {
                "origen": "huella", "motivo": "huella_no_medible", "host": h, "mision_id": mision_id,
                "detalle": ["sin_huella_de_apertura"]})
            continue
        try:
            despues = await tomar_huella(h)
        except Exception as exc:  # fail-soft: no medible se reporta como huella_no_medible y PAUSA (fail-closed en el resultado) -- nunca se sigue como si nada
            log.error("vigia_servicio huella_no_medible host=%s tipo=%s", h, type(exc).__name__)
            motivos.append((h, "huella_no_medible"))
            await asyncio.to_thread(pausar, pausa_ruta, {
                "origen": "huella", "motivo": "huella_no_medible", "host": h, "mision_id": mision_id,
                "detalle": [type(exc).__name__]})
            continue
        if not huella.huella_valida(despues, rutas=huella.RUTAS_DECLARADAS_POR_DEFAULT + rutas_extra):
            log.error("vigia_servicio huella_no_medible host=%s motivo=huella_vacia", h)
            motivos.append((h, "huella_no_medible"))
            await asyncio.to_thread(pausar, pausa_ruta, {
                "origen": "huella", "motivo": "huella_no_medible", "host": h, "mision_id": mision_id,
                "detalle": ["huella_de_cierre_vacia"]})
            continue
        encontrados = huella.hallazgos(antes, despues)
        ruta = ruta_huella(misiones, mision_id, h)
        if encontrados:
            log.critical("vigia_servicio huella_cambio_no_declarado host=%s lineas=%d", h, len(encontrados))
            motivos.append((h, "huella_cambio_no_declarado"))
            await asyncio.to_thread(huella.escribir_marca, ruta,
                                    huella.Marca(huella=antes, estado=huella.REPORTADA, diff=tuple(encontrados)))
            await asyncio.to_thread(pausar, pausa_ruta, {
                "origen": "huella", "motivo": "huella_cambio_no_declarado", "host": h, "mision_id": mision_id,
                "detalle": list(encontrados[:20]) + [_hint_aceptar(h, mision_id)]})
            continue
        await asyncio.to_thread(huella.escribir_marca, ruta, huella.Marca(huella=antes, estado=huella.CERRADA))
    return tuple(motivos)


async def correr_mision(ctx: arranque.Contexto, mision: Mision, *, latido_cada_s: float, lote_max: int,
                        intervalo_s: float, auditar, fin: asyncio.Event, exigir=arranque.exigir_contratos,
                        vigilar=vigia.vigilar, maquinas: tuple, hosts_con_sudo: tuple = (),
                        misiones: Path | None = None, mision_id: str | None = None,
                        tomar_huella=None, pausar=pausa.poner_pausa, rutas_extra: tuple = ()) -> tuple:
    """Lanza ContratosNoVerificados sin haber latido nunca si un contrato no está vivo.

    `hosts_con_sudo`/`tomar_huella` (M-1/M-2, ronda 3; B-1/M-1/M-2 ronda 4;
    simplificado ronda 6): si se dan, se toma (o se carga, si ya existe -- ver
    `huella_de_apertura_de_la_mision`) una huella de cada host ANTES de latir y otra AL
    CERRAR (fin normal); cualquier cambio pone la pausa (sin declarado, ronda 6).
    `tomar_huella=None` (el default): sin huella -- así los llamadores que no la
    necesitan (o corren en un entorno sin ssh/sudo, como los tests que no la ejercitan)
    no cambian de comportamiento.

    M-1 (ronda 6): si CUALQUIER host tiene una deuda de verificación huérfana que no
    se pudo resolver limpia, la misión NO ABRE -- ni siquiera para los demás hosts:
    `huella_de_apertura_de_la_mision` devuelve `None` para ese host, y acá se propaga
    `HuellaHuerfanaNoResuelta` ANTES de `vigilar()` (no se abre el proxy; `_principal`
    la reporta con su propio código, no como un contrato más). Devuelve los motivos de
    pausa por huella (vacío si no se pidió huella o si todo midió limpio).

    `rutas_extra` (MAJOR-C, ronda 4): rutas ADEMÁS de `huella.RUTAS_DECLARADAS_POR_DEFAULT`
    que `huella_valida()` tiene que ver representadas -- típicamente el authorized_keys
    RESUELTO del administrador (`huella.ruta_authorized_keys_admin(admin_usuario)`),
    que `_principal` pasa porque es quien conoce la cuenta. Vacío por default: los
    llamadores de test que no la necesitan no cambian de comportamiento."""
    if ctx.hosts_mision != mision.hosts:
        raise ValueError("contexto_de_otra_mision")
    await exigir(ctx)
    if fin.is_set():  # lo pararon mientras se verificaban los contratos: no se abre nada
        log.info("vigia_servicio mision_cancelada_antes_de_abrir")
        return ()
    huellas_iniciales = {}
    if tomar_huella is not None and hosts_con_sudo:
        for h in hosts_con_sudo:
            baseline = await huella_de_apertura_de_la_mision(
                misiones=misiones, mision_id=mision_id, host=h, tomar_huella=tomar_huella,
                pausar=pausar, pausa_ruta=ctx.pausa, rutas_extra=rutas_extra)
            if baseline is None:
                log.critical("vigia_servicio mision_no_abre_por_huella_huerfana host=%s", h)
                raise HuellaHuerfanaNoResuelta(h)
            huellas_iniciales[h] = baseline
    desde = (await asyncio.to_thread(os.stat, ctx.registro)).st_size
    cfg = vigia.ConfigVigia(registro=ctx.registro, desde_byte=desde, mision=mision.texto, lote_max=lote_max,
                            intervalo_s=intervalo_s, pausa=ctx.pausa, latido=ctx.latido, latido_cada_s=latido_cada_s,
                            maquinas=maquinas)
    log.info("vigia_servicio mision_abierta desde_byte=%s hosts=%s", desde, ",".join(sorted(mision.hosts)))
    await vigilar(cfg, auditar, fin)
    # Sólo en el fin normal: con una excepción el latido se deja envejecer y vigia.py ya puso la pausa.
    await asyncio.to_thread(_borrar_latido, ctx.latido)
    pausas_de_huella = ()
    if tomar_huella is not None and hosts_con_sudo:
        pausas_de_huella = await _verificar_huellas_al_cierre(
            ctx.pausa, huellas_iniciales, hosts_con_sudo, misiones, mision_id,
            tomar_huella=tomar_huella, pausar=pausar, rutas_extra=rutas_extra)
    log.info("vigia_servicio mision_cerrada")
    return pausas_de_huella


async def correr_huella_por_ssh(argv: list, host: str, *, tope_s: float, correr=None) -> huella.Huella:
    """La ejecución REAL detrás de `_principal._tomar_huella` -- corre `argv` (ya
    armado por `huella.argv_huella_servicio`, ronda de arreglo del bug de producción
    jax#260, 2026-09-22), EXIGE rc==0 (M-4, ronda 6: un mutante que quite este chequeo
    dejaría pasar una huella de un comando que reventó a mitad de camino, con salida
    parcial, como si fuera limpia) y arma la `Huella` desde stdout. Extraída a nivel de
    módulo para poder probarla sin el resto de `_principal` (conexión DB, política,
    etc.) -- M-4 pide un test de esta pieza."""
    correr = correr or asyncio.create_subprocess_exec
    proc = await correr(*argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                        start_new_session=True)
    try:
        salida, errores = await asyncio.wait_for(proc.communicate(), tope_s)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        proc.kill()
        await proc.wait()
        raise
    if proc.returncode != 0:
        raise RuntimeError(f"huella_rc_{proc.returncode}: {errores.decode(errors='replace')[:200]}")
    return huella.huella_desde_salida(host, salida)


async def _principal(ruta_mision: Path) -> int:
    from facet_resolver import resolve_facet
    from jacobs.store import conexion
    from jax.ejecutor.contratos import auditor_cliente, eleccion_c5

    mision = mision_desde_bytes(await asyncio.to_thread(ruta_mision.read_bytes))
    ctx = arranque.contexto_desde_entorno(os.environ, mision.hosts)
    # Barrido (ronda 9): al arrancar el vigía, limpia los temporales huérfanos que un
    # kill puede haber dejado de una corrida anterior de `quitar_pausa_si` -- nunca la
    # pausa misma (ver `pausa.barrer_temporales_huerfanos`).
    await asyncio.to_thread(pausa.barrer_temporales_huerfanos, ctx.pausa)
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
    # M-1/M-2 (ronda 3): "con sudo" hoy equivale a "remota" -- hall9000 es la única
    # local y quedó `sudo=false` en `maquinas.toml`; las tres remotas tienen sudo real
    # (Fase 3). `politica.Host` no trae un campo `sudo` propio (eso vive en
    # maquinas.toml, host-bound, Fase 0, no se lee en runtime) -- si el día de mañana
    # una máquina remota pierde el sudo o una local lo gana, este criterio hay que
    # revisarlo junto con esa migración, no antes.
    #
    # M-3 (ronda 8, texto) -- CORREGIDO en ronda 9 (MAJOR-2, auditoría 7): esto decía
    # que axioma SÍ tenía sudo real en hall9000 (NOPASSWD:ALL) pero quedaba inerte por
    # NoNewPrivs en la jaula. Eso YA NO ES CIERTO -- HECHO NUEVO, verificado la noche
    # del 2026-09-22: a `axioma` se le QUITÓ el sudo en hall9000 (`sudo -l -U axioma` →
    # no permitido). Las remotas SÍ lo conservan. `maquinas.toml: sudo=false` para
    # hall9000 hoy es tanto "no utilizable desde la jaula" (NoNewPrivs, sigue siendo
    # cierto y sigue siendo una capa aparte) COMO "sin la concesión de sudoers": las
    # dos cosas, no sólo una. La única llave autorizada para axioma en hall9000 sigue
    # siendo la del CONTROLADOR (`/etc/jax/controlador/`, dueño `jaxsvc`, modo `600` --
    # axioma no puede ni leerla; verificado por Hyde) -- eso no cambió.
    #
    # EL CONTROL REAL (ronda 9): no es "axioma tiene o no tiene sudo" en abstracto --
    # son los permisos del registro de C3 y de `JAX_EJECUTOR_MISIONES` (ambos `jaxsvc`)
    # MÁS quién, concretamente, tiene sudo real hacia `jaxsvc` en esta máquina. Hoy en
    # hall9000 eso es sólo `fruiz`. Ver `huella.py::principal` y `_registrar_aceptacion`
    # para el mismo razonamiento aplicado a la CLI de aceptación de huellas.
    remotas_con_sudo = hosts_con_sudo(mision.hosts, hosts_pol)
    # M-1 (ronda 4; corregido -- bug de producción jax#260, 2026-09-22): la huella la
    # toma el CONTROLADOR (mismo administrador que `JAX_EJECUTOR_ADMIN_USUARIO`, nunca
    # `axioma` -- una cuenta sin privilegios no puede medirse a sí misma), pero YA NO
    # con la CREDENCIAL personal de ese administrador: este proceso corre como `jaxsvc`
    # (ver el docstring del módulo), que no puede leer `~fruiz/.ssh/*`. Usa la llave
    # PROPIA del servicio (`huella.argv_huella_servicio`, JAX_EJECUTOR_HUELLA_LLAVE) --
    # ver el docstring de ese módulo para el porqué completo.
    admin_usuario = os.environ["JAX_EJECUTOR_ADMIN_USUARIO"]
    huella_llave = Path(os.environ["JAX_EJECUTOR_HUELLA_LLAVE"])
    huella_known_hosts = Path(os.environ["JAX_EJECUTOR_HUELLA_KNOWN_HOSTS"])
    misiones_dir = Path(os.environ["JAX_EJECUTOR_MISIONES"])
    mision_id = mision_id_desde_ruta(ruta_mision)

    async def auditar(lote):
        return await auditor_cliente.auditar(lote, faceta=auditor_f, max_tokens=cfg.max_tokens)

    async def _tomar_huella(nombre_host: str) -> huella.Huella:
        """`huella.argv_huella_servicio` -- una sola forma, apertura y cierre comparan
        lo mismo. NUNCA `revocacion.argv_admin` (bug de producción jax#260, 2026-09-22:
        ese camino resolvía la identidad por default de ssh, y este proceso corre como
        `jaxsvc`, que no puede leer la llave personal de `admin_usuario`). La ejecución
        de verdad vive en `correr_huella_por_ssh` (a nivel de módulo, testeable aparte)."""
        h = hosts_pol[nombre_host]
        argv = huella.argv_huella_servicio(h, llave=huella_llave, known_hosts=huella_known_hosts,
                                           admin_usuario=admin_usuario, tope_s=_TOPE_HUELLA_S)
        return await correr_huella_por_ssh(argv, nombre_host, tope_s=_TOPE_HUELLA_S)

    fin = asyncio.Event()
    loop = asyncio.get_running_loop()
    for senal in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(senal, fin.set)
    try:
        pausas_de_huella = await correr_mision(
            ctx, mision, latido_cada_s=latido_cada_s, lote_max=cfg.lote_max,
            intervalo_s=cfg.intervalo_s, auditar=auditar, fin=fin, maquinas=maquinas,
            hosts_con_sudo=remotas_con_sudo, misiones=misiones_dir, mision_id=mision_id,
            tomar_huella=_tomar_huella,
            # MAJOR-C (ronda 4): el authorized_keys del administrador entra a las
            # rutas EXIGIDAS -- acá, y sólo acá, se conoce `admin_usuario` -- así que
            # `RUTAS_DECLARADAS_POR_DEFAULT` (fija) no podía incluirla por sí sola.
            rutas_extra=(huella.ruta_authorized_keys_admin(admin_usuario),))
    except arranque.ContratosNoVerificados as exc:
        for f in exc.fallos:
            print(formato.campos((("contrato", f.contrato), ("codigo", f.codigo)) + tuple(f.datos)), flush=True)
        print(formato.campos((("arranco", False),)), flush=True)
        return 1
    except HuellaHuerfanaNoResuelta as exc:
        # MINOR (ronda 6): código propio, NO un ContratosNoVerificados más -- así
        # `mision.py::correr_turno` no lo confunde con "vigia_no_latio" (el vigía nunca
        # llegó a latir, pero no porque un contrato esté caído: porque hay una deuda de
        # huella sin resolver de otra vuelta).
        print(formato.campos((("arranco", False), ("codigo", "huella_huerfana_no_resuelta"),
                              ("host", exc.args[0] if exc.args else ""))), flush=True)
        return 1
    # `huella_pausada` SIEMPRE presente (visible en el resultado del turno, ronda 4 M-1):
    # nunca queda en verde en silencio -- vacío/False es el caso limpio, explícito igual.
    print(formato.campos((("arranco", True), ("cerrada", True),
                          ("huella_pausada", bool(pausas_de_huella)),
                          ("huella_motivos", ",".join(f"{h}:{m}" for h, m in pausas_de_huella)))), flush=True)
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
    except RuntimeError as exc:
        # MINOR (ronda 6): un RuntimeError (huella_rc_*, huella_apertura_vacia, lo que
        # `_tomar_huella`/`huella_de_apertura_de_la_mision` propaguen sin capturar
        # arriba, M-3) se reporta con SU PROPIO código -- antes se colaba sin capturar
        # y el proceso moría con una traza sin código estable; `mision.py::correr_turno`
        # lo veía como "el vigía no latió" (`vigia_no_latio`), indistinguible de un
        # contrato lento. Acá queda explícito qué pasó.
        print(formato.campos((("arranco", False), ("codigo", "vigia_error_en_arranque"),
                              ("tipo", type(exc).__name__), ("detalle", str(exc.args[0]) if exc.args else ""))),
              file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(principal(sys.argv[1:]))
