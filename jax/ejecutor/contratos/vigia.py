# jax/ejecutor/contratos/vigia.py
"""Vigía de C5: sigue el registro de C3 (lo escribe el proxy, fuera de la jaula; no lo
que la jaula dice de sí misma) desde el byte en que empezó la misión, junta pasos,
audita por lote (lleno, vencido el intervalo o al terminar) y, si el auditor dice
pausar, pone la pausa DEL EJECUTOR con `origen=c5`, el motivo y el paso.

La pausa es la propia del Ejecutor (`pausa.py`), no el interruptor global de JAX:
ver la decisión en ese módulo.

FAIL-CLOSED en cinco puntos:
- el auditor devuelve algo ilegible → pausa `auditor_ilegible`;
- el auditor revienta (red, tope, lo que sea) → pausa `auditor_caido`;
- la cadena del registro no cuadra desde donde empezó la misión (una línea editada,
  borrada, insertada, o un `desde_byte` a mitad de línea) → pausa `registro_roto`;
- el propio vigía termina por cualquier cosa que no sea `fin` (excepción, cancelación,
  no poder latir) → pausa `vigia_caido`;
- el vigía muere sin poder escribir nada (SIGKILL): deja de latir y el proxy deja de
  servir (`pausa.latido_fresco`).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from jax.ejecutor.contratos import auditor as A
from jax.ejecutor.contratos import pausa as P
from jax.ejecutor.contratos.registro import GENESIS

log = logging.getLogger("ejecutor.vigia")
_SONDEO_S = 0.1
_LEER_ATRAS = 1 << 20


class RegistroRoto(RuntimeError):
    """La cadena del registro no cuadra. `args[0]` es el código."""


@dataclass(frozen=True)
class ConfigVigia:
    registro: Path
    desde_byte: int
    mision: str
    lote_max: int
    intervalo_s: float
    pausa: Path
    latido: Path
    latido_cada_s: float
    maquinas: tuple  # auditor.Maquina de la misión: los lotes las llevan al auditor


def _pausar(motivo: str, paso: int | None, cfg: ConfigVigia) -> None:
    puesta = P.poner_pausa(cfg.pausa, {"origen": "c5", "motivo": motivo, "paso": paso,
                                       "registro": str(cfg.registro), "desde_byte": cfg.desde_byte})
    log.critical("vigia pausa motivo=%s paso=%s nueva=%s", motivo, paso, puesta)


def _leer_desde(ruta: Path, desde: int) -> bytes:
    with open(ruta, "rb") as f:
        f.seek(desde)
        return f.read()


def _punto_de_partida(ruta: Path, desde: int) -> tuple[str, int]:
    """(`prev` esperado, `n` esperado) para la primera línea desde `desde`."""
    if desde == 0:
        return GENESIS, 1
    with open(ruta, "rb") as f:
        inicio = max(0, desde - _LEER_ATRAS)
        f.seek(inicio)
        atras = f.read(desde - inicio)
    if not atras.endswith(b"\n"):
        raise RegistroRoto("desde_a_mitad_de_linea")
    cuerpo = atras[:-1]
    corte = cuerpo.rfind(b"\n")
    if corte < 0 and inicio > 0:
        raise RegistroRoto("linea_demasiado_larga")
    anterior = cuerpo[corte + 1:]
    try:
        n = json.loads(anterior)["n"]
    except (ValueError, KeyError, TypeError):
        raise RegistroRoto("linea_ilegible") from None
    if isinstance(n, bool) or not isinstance(n, int):
        raise RegistroRoto("linea_ilegible")
    return hashlib.sha256(anterior).hexdigest(), n + 1


async def _latir_siempre(cfg: ConfigVigia) -> None:
    while True:
        await asyncio.to_thread(P.latir, cfg.latido)
        await asyncio.sleep(cfg.latido_cada_s)


async def vigilar(cfg: ConfigVigia, auditar, fin: asyncio.Event, *, pausar=None, reloj=time.monotonic) -> None:
    pausar = pausar or _pausar
    latidor = asyncio.create_task(_latir_siempre(cfg))
    try:
        prev, n_esperado = await asyncio.to_thread(_punto_de_partida, cfg.registro, cfg.desde_byte)
        offset, resto = cfg.desde_byte, b""
        pendientes: dict = {}          # tool_use_id -> Paso (en orden de llegada)
        primero_pendiente = None
        while True:
            if latidor.done():
                latidor.result()       # si no pudo latir, el vigía muere (y frena abajo)
            datos = await asyncio.to_thread(_leer_desde, cfg.registro, offset)
            offset += len(datos)
            resto += datos
            *completas, resto = resto.split(b"\n")
            for cruda in completas:
                try:
                    ev = json.loads(cruda)
                except ValueError:
                    raise RegistroRoto("linea_ilegible") from None
                if not isinstance(ev, dict) or ev.get("n") != n_esperado:
                    raise RegistroRoto("n_no_cuadra")
                if ev.get("prev") != prev:
                    raise RegistroRoto("prev_no_cuadra")
                prev, n_esperado = hashlib.sha256(cruda).hexdigest(), n_esperado + 1
                if ev.get("evento") == "herramienta_pedida":
                    pendientes[ev.get("tool_use_id")] = A.Paso(ev["n"], ev.get("herramienta"),
                                                               ev.get("entrada", ev.get("entrada_inicio")), None)
                    if primero_pendiente is None:
                        primero_pendiente = reloj()
                elif ev.get("evento") == "resultado_devuelto" and ev.get("tool_use_id") in pendientes:
                    p = pendientes[ev["tool_use_id"]]
                    pendientes[ev["tool_use_id"]] = A.Paso(p.n, p.herramienta, p.entrada, ev.get("es_error"))
            vencido = primero_pendiente is not None and reloj() - primero_pendiente >= cfg.intervalo_s
            if pendientes and (len(pendientes) >= cfg.lote_max or vencido or fin.is_set()):
                lote = A.Lote(cfg.mision, tuple(pendientes.values()), (), cfg.maquinas)
                pendientes, primero_pendiente = {}, None
                try:
                    revision = await auditar(lote)
                except A.AuditorIlegible as exc:
                    log.error("vigia auditor_ilegible codigo=%s", exc.codigo)
                    await asyncio.to_thread(pausar, "auditor_ilegible", None, cfg)
                    continue
                except Exception as exc:  # fail-soft: el vigía no se cae por el auditor; FRENA (pausa auditor_caido) y sigue leyendo
                    log.error("vigia auditor_caido tipo=%s", type(exc).__name__)
                    await asyncio.to_thread(pausar, "auditor_caido", None, cfg)
                    continue
                if revision.pausar:
                    await asyncio.to_thread(pausar, revision.motivo, revision.paso, cfg)
            elif fin.is_set():
                return
            await asyncio.sleep(_SONDEO_S)
    except RegistroRoto as exc:
        log.critical("vigia registro_roto codigo=%s", exc.args[0])
        await asyncio.to_thread(pausar, "registro_roto", None, cfg)
        raise
    except BaseException as exc:
        if not (isinstance(exc, asyncio.CancelledError) and fin.is_set()):
            await asyncio.to_thread(pausar, "vigia_caido", None, cfg)
        raise
    finally:
        latidor.cancel()
        await asyncio.gather(latidor, return_exceptions=True)
