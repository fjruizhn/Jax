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
<ruta_mision>`), heredando la identidad del proceso que lo lanza -- en producción,
`jax-platform` (`User=fruiz`), así que este módulo corre como `fruiz`. Eso es lo que hace
coherente a M-1 más abajo (la huella la toma el controlador COMO FRUIZ, nunca como
`axioma`): no es una cuenta de servicio aparte, es la misma identidad del proceso.
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
import shlex
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

from jax.ejecutor.contratos import arranque, cuenta_axioma, formato, huella, pausa, politica, revocacion, vigia
from jax.ejecutor.contratos import auditor as A

log = logging.getLogger("ejecutor.vigia_servicio")
VARIABLE_LATIDO_CADA_S = "JAX_EJECUTOR_VIGIA_LATIDO_CADA_S"
_TOPE_HUELLA_S = 30
_SUFIJO_TURNO = re.compile(r"-t\d+$")


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


def ruta_huella_apertura(misiones: Path, mision_id: str, host: str) -> Path:
    if not cuenta_axioma._MISION_ID_VALIDA.match(mision_id):
        raise cuenta_axioma.MisionIdInvalido(mision_id)
    if not host or "/" in host or host.strip() != host:
        raise ValueError("host_invalido")
    return Path(misiones) / mision_id / "huella" / f"{host}.json"


def _huella_a_json(h: huella.Huella) -> dict:
    return {
        "host": h.host, "controles": h.controles, "persistencia": h.persistencia,
        "log": None if h.log is None else {"inode": h.log.inode, "tamano": h.log.tamano, "sha256": h.log.sha256},
    }


def _huella_desde_json(d: dict) -> huella.Huella:
    log_ = d.get("log")
    return huella.Huella(host=d["host"], controles=d["controles"], persistencia=d["persistencia"],
                         log=None if log_ is None else huella.InfoLog(**log_))


def _persistir_huella_apertura(ruta: Path, h: huella.Huella) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".tmp")
    tmp.write_text(json.dumps(_huella_a_json(h)), encoding="utf-8")
    os.replace(tmp, ruta)


async def huella_de_apertura_de_la_mision(*, misiones: Path, mision_id: str, host: str, tomar_huella) -> huella.Huella:
    """La huella de APERTURA de la MISIÓN (ronda 4, M-1) -- NO la del turno. Si ya hay
    una persistida para `(mision_id, host)`, la carga TAL CUAL y NUNCA la vuelve a
    tomar: es el turno 1 el que la fija, y todos los turnos siguientes comparan contra
    ESE momento cero. Así un cambio que el turno N no llegó a ver (o que su cierre no
    llegó a medir) no queda blanqueado cuando el turno N+1 vuelve a mirar.

    Deliberadamente SIN try/except acá (M-3, ronda 4): si `tomar_huella` revienta la
    primera vez, la excepción se propaga -- una huella de apertura que no se pudo tomar
    no es "sin cambios", es que la misión no debe abrir."""
    ruta = ruta_huella_apertura(misiones, mision_id, host)
    try:
        datos = await asyncio.to_thread(ruta.read_text, "utf-8")
    except FileNotFoundError:
        datos = None
    if datos is not None:
        return _huella_desde_json(json.loads(datos))
    h = await tomar_huella(host)
    await asyncio.to_thread(_persistir_huella_apertura, ruta, h)
    return h


def hosts_con_sudo(hosts_mision, hosts_pol: dict) -> tuple:
    """Las máquinas de la misión que son remotas -- M-1/M-2 (ronda 3): "con sudo" hoy
    equivale a "remota" (comentario largo en `_principal`: hall9000 es la única local y
    quedó `sudo=false`; las tres remotas tienen sudo real, Fase 3). Una máquina de la
    misión que no está en la política se omite acá -- ya la rechaza
    `arranque.exigir_contratos` antes de llegar a este punto."""
    return tuple(sorted(n for n in hosts_mision if n in hosts_pol and not hosts_pol[n].es_local))


async def _verificar_huellas_al_cierre(pausa_ruta: Path, mision: Mision, huellas_iniciales: dict,
                                       hosts_con_sudo: tuple, *, tomar_huella_cierre, pausar) -> tuple:
    """Ronda 4 (M-1): «el CIERRE falla cerrado». Si la huella de cierre de un host NO SE
    PUEDE TOMAR (ssh caído, sudo denegado, lo que sea) o si no hubo huella de apertura
    que comparar, es un hallazgo `huella_no_medible` y PONE LA PAUSA -- ya no el
    `continue` silencioso de la ronda 3. Queda visible en el resultado del turno sin
    tocar el formato de salida del vigía: `pausar()` escribe en `ctx.pausa`, y
    `mision.py::correr_turno` YA lee esa pausa después de cerrar el vigía y hace
    `codigo = "pausa_puesta"` si está puesta -- el mismo camino que cualquier otra
    pausa. Devuelve `((host, motivo), ...)` -- vacío si todo midió limpio."""
    motivos = []
    for h in hosts_con_sudo:
        antes = huellas_iniciales.get(h)
        if antes is None:
            log.error("vigia_servicio huella_no_medible host=%s motivo=sin_apertura", h)
            motivos.append((h, "huella_no_medible"))
            await asyncio.to_thread(pausar, pausa_ruta, {
                "origen": "huella", "motivo": "huella_no_medible", "host": h, "detalle": ["sin_huella_de_apertura"]})
            continue
        try:
            despues = await tomar_huella_cierre(h, tamano_apertura_log=antes.log.tamano if antes.log else 0)
        except Exception as exc:  # fail-soft: no medible se reporta como huella_no_medible y PAUSA (fail-closed en el resultado, ronda 4 M-1) -- nunca se sigue como si nada
            log.error("vigia_servicio huella_no_medible host=%s tipo=%s", h, type(exc).__name__)
            motivos.append((h, "huella_no_medible"))
            await asyncio.to_thread(pausar, pausa_ruta, {
                "origen": "huella", "motivo": "huella_no_medible", "host": h, "detalle": [type(exc).__name__]})
            continue
        informe = huella.hallazgos_persistencia(antes, despues, mision.texto)
        if informe:
            log.warning("vigia_servicio huella_persistencia_cambiada host=%s lineas=%d", h, len(informe))
        encontrados = huella.hallazgos_controles(antes, despues, mision.texto)
        if encontrados:
            log.critical("vigia_servicio huella_cambio_no_declarado host=%s lineas=%d", h, len(encontrados))
            motivos.append((h, "huella_cambio_no_declarado"))
            await asyncio.to_thread(pausar, pausa_ruta, {
                "origen": "huella", "motivo": "huella_cambio_no_declarado",
                "host": h, "detalle": list(encontrados[:20])})
    return tuple(motivos)


async def correr_mision(ctx: arranque.Contexto, mision: Mision, *, latido_cada_s: float, lote_max: int,
                        intervalo_s: float, auditar, fin: asyncio.Event, exigir=arranque.exigir_contratos,
                        vigilar=vigia.vigilar, maquinas: tuple, hosts_con_sudo: tuple = (),
                        misiones: Path | None = None, mision_id: str | None = None,
                        tomar_huella_apertura=None, tomar_huella_cierre=None,
                        pausar=pausa.poner_pausa) -> tuple:
    """Lanza ContratosNoVerificados sin haber latido nunca si un contrato no está vivo.

    `hosts_con_sudo`/`tomar_huella_apertura`/`tomar_huella_cierre` (M-1/M-2, ronda 3;
    B-1/M-1/M-2 ronda 4): si se dan, se toma (o se carga, si ya existe -- ver
    `huella_de_apertura_de_la_mision`) una huella de cada host ANTES de latir y otra AL
    CERRAR (fin normal); un cambio no declarado en el texto de la misión pone la pausa.
    `tomar_huella_apertura=None` (el default): sin huella -- así los llamadores que no
    la necesitan (o corren en un entorno sin ssh/sudo, como los tests que no la
    ejercitan) no cambian de comportamiento. Devuelve los motivos de pausa por huella
    (vacío si no se pidió huella o si todo midió limpio)."""
    if ctx.hosts_mision != mision.hosts:
        raise ValueError("contexto_de_otra_mision")
    await exigir(ctx)
    if fin.is_set():  # lo pararon mientras se verificaban los contratos: no se abre nada
        log.info("vigia_servicio mision_cancelada_antes_de_abrir")
        return ()
    desde = (await asyncio.to_thread(os.stat, ctx.registro)).st_size
    cfg = vigia.ConfigVigia(registro=ctx.registro, desde_byte=desde, mision=mision.texto, lote_max=lote_max,
                            intervalo_s=intervalo_s, pausa=ctx.pausa, latido=ctx.latido, latido_cada_s=latido_cada_s,
                            maquinas=maquinas)
    huellas_iniciales = {}
    if tomar_huella_apertura is not None and hosts_con_sudo:
        huellas_iniciales = {
            h: await huella_de_apertura_de_la_mision(
                misiones=misiones, mision_id=mision_id, host=h, tomar_huella=tomar_huella_apertura)
            for h in hosts_con_sudo}
    log.info("vigia_servicio mision_abierta desde_byte=%s hosts=%s", desde, ",".join(sorted(mision.hosts)))
    await vigilar(cfg, auditar, fin)
    # Sólo en el fin normal: con una excepción el latido se deja envejecer y vigia.py ya puso la pausa.
    await asyncio.to_thread(_borrar_latido, ctx.latido)
    pausas_de_huella = ()
    if tomar_huella_apertura is not None and hosts_con_sudo:
        pausas_de_huella = await _verificar_huellas_al_cierre(
            ctx.pausa, mision, huellas_iniciales, hosts_con_sudo,
            tomar_huella_cierre=tomar_huella_cierre, pausar=pausar)
    log.info("vigia_servicio mision_cerrada")
    return pausas_de_huella


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
    remotas_con_sudo = hosts_con_sudo(mision.hosts, hosts_pol)
    # M-1 (ronda 4): la huella la toma el CONTROLADOR como `fruiz` (JAX_EJECUTOR_ADMIN_USUARIO),
    # NUNCA como `axioma` -- una cuenta sin privilegios no puede medirse a sí misma. Mismo
    # camino que ya usan `ops/ejecutor/_maquina.sh` y `scripts/ejecutor_contratos/revocar.py`
    # para C6/revocar: `revocacion.argv_admin` + UN solo `sudo -n sh -c '<script>'`.
    admin_usuario = os.environ["JAX_EJECUTOR_ADMIN_USUARIO"]
    misiones_dir = Path(os.environ["JAX_EJECUTOR_MISIONES"])
    mision_id = mision_id_desde_ruta(ruta_mision)

    async def auditar(lote):
        return await auditor_cliente.auditar(lote, faceta=auditor_f, max_tokens=cfg.max_tokens)

    async def _tomar_huella(nombre_host: str, comando_remoto: str, *, cierre: bool) -> huella.Huella:
        h = hosts_pol[nombre_host]
        argv = revocacion.argv_admin(h, admin_usuario, f"sudo -n sh -c {shlex.quote(comando_remoto)}")
        proc = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE, start_new_session=True)
        try:
            salida, errores = await asyncio.wait_for(proc.communicate(), _TOPE_HUELLA_S)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            proc.kill()
            await proc.wait()
            raise
        if proc.returncode != 0:
            raise RuntimeError(f"huella_rc_{proc.returncode}: {errores.decode(errors='replace')[:200]}")
        armar = huella.huella_de_cierre_desde_salida if cierre else huella.huella_de_apertura_desde_salida
        return armar(nombre_host, salida)

    async def tomar_huella_apertura(nombre_host: str) -> huella.Huella:
        return await _tomar_huella(nombre_host, huella.comando_apertura(ctx.cuenta.nombre), cierre=False)

    async def tomar_huella_cierre(nombre_host: str, *, tamano_apertura_log: int) -> huella.Huella:
        comando = huella.comando_cierre(ctx.cuenta.nombre, tamano_apertura_log=tamano_apertura_log)
        return await _tomar_huella(nombre_host, comando, cierre=True)

    fin = asyncio.Event()
    loop = asyncio.get_running_loop()
    for senal in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(senal, fin.set)
    try:
        pausas_de_huella = await correr_mision(
            ctx, mision, latido_cada_s=latido_cada_s, lote_max=cfg.lote_max,
            intervalo_s=cfg.intervalo_s, auditar=auditar, fin=fin, maquinas=maquinas,
            hosts_con_sudo=remotas_con_sudo, misiones=misiones_dir, mision_id=mision_id,
            tomar_huella_apertura=tomar_huella_apertura, tomar_huella_cierre=tomar_huella_cierre)
    except arranque.ContratosNoVerificados as exc:
        for f in exc.fallos:
            print(formato.campos((("contrato", f.contrato), ("codigo", f.codigo)) + tuple(f.datos)), flush=True)
        print(formato.campos((("arranco", False),)), flush=True)
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


if __name__ == "__main__":
    sys.exit(principal(sys.argv[1:]))
