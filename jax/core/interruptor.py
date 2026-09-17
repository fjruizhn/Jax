"""El interruptor de JAX (kill switch): dónde está el freno y si está puesto.

FUENTE ÚNICA para los procesos que lo LEEN (LAS MANOS con Jacobs adentro, el
REPL y `jax --task`) y para el único que lo ESCRIBE (jax-platform, que tiene
una copia en `backend/interruptor.py`: familia `interruptor` de
scripts/check_mirror_sync.py).

Hasta el 2026-09-16 la ruta estaba escrita a mano en seis lugares (uno con un
default silencioso) y cada lector miraba con `Path.exists()`, que devuelve
False ante un PermissionError: con el archivo puesto y el directorio
ilegible, el freno se leía SUELTO (medido en Python 3.14.4).

Reglas:
- La ruta sale de JAX_KILL_SWITCH_PATH y se lee en CADA llamada. Sin la
  variable, vacía o relativa: `InterruptorSinConfigurar`. Sin saber dónde
  está el freno no se ejecuta nada.
- `interruptor_activo` usa os.stat: sólo "no existe" es SUELTO; cualquier
  otro error (permiso, ENOTDIR, E/S) es PUESTO.
- `escribir_pausa` publica el archivo completo de una vez (temporal +
  os.link, que falla si ya existe) y `borrar_pausa` lo quita; los dos
  sincronizan el directorio.
- `correr_con_interruptor` (sólo jax) corre una corrutina y la cancela si el
  freno aparece; `hyde_sandbox.run_sandboxed_claude` mata su proceso al
  recibir la cancelación.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

VARIABLE_RUTA = "JAX_KILL_SWITCH_PATH"


class InterruptorSinConfigurar(RuntimeError):
    """JAX_KILL_SWITCH_PATH falta, está vacía o no es una ruta absoluta."""


def ruta_del_interruptor() -> Path:
    valor = os.environ.get(VARIABLE_RUTA, "").strip()
    if not valor:
        raise InterruptorSinConfigurar(
            f"{VARIABLE_RUTA} no está definida: sin saber dónde está el freno "
            "no se ejecuta nada (se define en /etc/jax/.env)"
        )
    ruta = Path(valor)
    if not ruta.is_absolute():
        raise InterruptorSinConfigurar(
            f"{VARIABLE_RUTA} tiene que ser una ruta absoluta, no {valor!r}"
        )
    return ruta


def interruptor_activo(ruta: Path | str | None = None) -> bool:
    objetivo = Path(ruta) if ruta is not None else ruta_del_interruptor()
    try:
        os.stat(objetivo)
    except FileNotFoundError:
        return False
    except OSError:  # fail-closed: sin poder mirar el freno (permiso, ENOTDIR, E/S) se lo da por PUESTO
        return True
    return True


def _sincronizar_directorio(directorio: Path) -> None:
    descriptor = os.open(directorio, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def escribir_pausa(ruta: Path, contenido: str) -> bool:
    """Pone el freno. True si lo puso esta llamada; False si ya estaba."""
    directorio = ruta.parent
    descriptor, temporal = tempfile.mkstemp(prefix=".pausa-", dir=directorio)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as archivo:
            archivo.write(contenido)
            archivo.flush()
            os.fsync(archivo.fileno())
        os.chmod(temporal, 0o660)
        try:
            os.link(temporal, ruta)
        except FileExistsError:
            return False
        _sincronizar_directorio(directorio)
        return True
    finally:
        os.unlink(temporal)


def borrar_pausa(ruta: Path) -> bool:
    """Quita el freno. True si lo quitó esta llamada; False si no estaba."""
    try:
        os.unlink(ruta)
    except FileNotFoundError:
        return False
    _sincronizar_directorio(ruta.parent)
    return True


class InterruptorActivado(RuntimeError):
    """El freno estaba puesto al empezar o apareció mientras la operación corría."""


INTERVALO_DE_SONDEO = 0.25  # segundos: el mismo del ssh_worker (CONTEXT.md §9, 2026-06-14)


async def correr_con_interruptor(corrutina, *, intervalo: float = INTERVALO_DE_SONDEO):
    ruta = ruta_del_interruptor()
    if interruptor_activo(ruta):
        corrutina.close()
        raise InterruptorActivado(f"killed_by_switch — {ruta} estaba puesto antes de empezar")
    tarea = asyncio.ensure_future(corrutina)
    try:
        while True:
            hechas, _ = await asyncio.wait({tarea}, timeout=intervalo)
            if hechas:
                return tarea.result()
            if interruptor_activo(ruta):
                tarea.cancel()
                await asyncio.gather(tarea, return_exceptions=True)
                raise InterruptorActivado(f"killed_by_switch — {ruta} apareció durante la ejecución")
    finally:
        if not tarea.done():
            tarea.cancel()
