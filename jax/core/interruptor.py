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
- `pausa_presente` usa os.stat sobre UN archivo: sólo "no existe" es
  SUELTO; cualquier otro error (permiso, ENOTDIR, E/S) es PUESTO.
- `interruptor_activo` es el freno de JAX: el archivo de JAX_KILL_SWITCH_PATH
  O la ruta heredada `RUTA_HEREDADA`. Requisito del controlador principal del
  frente B (2026-09-17): gente y scripts pausan creando la ruta vieja; si
  dejara de leerse, quien pause así creería que frenó, y no. La regla vive
  acá y no en cada lector. Con la heredada presente (o ilegible) se avisa
  con un WARNING que nombra la ruta nueva, una vez por episodio. Retirarla
  lo decide Fernando (DEUDA.md).
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
import logging
import os
import tempfile
from pathlib import Path

VARIABLE_RUTA = "JAX_KILL_SWITCH_PATH"

logger = logging.getLogger(__name__)

# Compatibilidad, NO configuración: es la ruta que la gente y los scripts ya
# usan para pausar a JAX (requisito del controlador principal del frente B,
# 2026-09-17). Por eso es una constante y no una variable de entorno: una
# variable que la pisara en producción apagaría la compatibilidad en silencio
# (ruling R15). Los tests la desvían con monkeypatch desde el conftest.
RUTA_HEREDADA = Path("/etc/jax/PAUSE")

# Anti-spam del WARNING, por proceso (ruling R12): True desde que se avisó de
# la heredada hasta que una lectura la encuentra ausente.
_heredada_avisada = False


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


def pausa_presente(ruta: Path | str) -> bool:
    """¿Está puesto ESTE archivo? Sólo "no existe" es False."""
    try:
        os.stat(ruta)
    except FileNotFoundError:
        return False
    except OSError:  # fail-closed: sin poder mirar el freno (permiso, ENOTDIR, E/S) se lo da por PUESTO
        return True
    return True


def _heredada_activa() -> bool:
    """La ruta heredada, con el WARNING una vez por episodio. El aviso nunca
    decide la lectura: lo que devuelve es siempre `pausa_presente`."""
    global _heredada_avisada
    presente = pausa_presente(RUTA_HEREDADA)
    if not presente:
        _heredada_avisada = False
        return False
    if not _heredada_avisada:
        _heredada_avisada = True
        nueva = os.environ.get(VARIABLE_RUTA, "").strip() or f"({VARIABLE_RUTA} sin definir)"
        logger.warning(
            "kill switch: la ruta VIEJA %s está puesta (o no se puede mirar) y JAX "
            "queda FRENADO por compatibilidad. La ruta del freno es %s: quitá %s "
            "en el servidor para soltarlo.",
            RUTA_HEREDADA, nueva, RUTA_HEREDADA,
        )
    return True


def interruptor_activo(ruta: Path | str | None = None) -> bool:
    objetivo = Path(ruta) if ruta is not None else ruta_del_interruptor()
    return pausa_presente(objetivo) or _heredada_activa()


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
        # Fix round 1 (2026-09-17): cancelar no alcanza. Un timeout externo
        # (asyncio.wait_for en _run_one_step) o una cancelación externa de
        # ESTA corrutina también pasan por acá, y si solo pedimos la
        # cancelación sin esperarla, la limpieza real del dispatch interno
        # (run_sandboxed_claude matando a Hyde, _invoke_motor cancelando el
        # job en LAS MANOS) puede seguir en curso cuando el error ya se
        # propagó y _fail_step ya corrió. gather() con return_exceptions
        # espera esa limpieza sin tragarse la CancelledError externa: no la
        # capturamos, así que sigue su curso normal después del finally.
        if not tarea.done():
            tarea.cancel()
            await asyncio.gather(tarea, return_exceptions=True)
