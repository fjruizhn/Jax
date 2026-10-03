"""Aviso del freno de extractores (jax-14, 2026-10-03).

Un 503 por falta de pdfplumber/openpyxl/python-docx no puede quedar mudo: el
2026-10-03 lo que faltaba se descubrio por casualidad. Reusa el canal que ya
existe, `jacobs.reaper.send_telegram_alert` (lo reusan `jacobs/aviso.py` y
`motor_registry/worker.py`); no escribe un cliente nuevo.

- Al arrancar: avisa siempre que falte algo.
- Desde el camino del POST: como mucho una vez por hora (marca de tiempo en
  memoria; vale por proceso -- un reinicio vuelve a avisar, que es lo correcto).
- Fire-and-forget con la referencia de la tarea guardada, igual que
  `jacobs/aviso.py::_AVISO_TASKS` (si no, el GC puede llevarsela a mitad).
- El texto lleva SOLO nombres de paquete: TELEGRAM_CHAT_ID es uno para todo el
  sistema, nunca va contenido de clientes (ver el ruling 4 de jacobs/aviso.py).
- Sin credenciales de Telegram no rompe: `send_telegram_alert` ya lo registra
  en el log y devuelve ok=False, que aqui tambien se registra.
- El log del 503 sale como mucho una vez por minuto (`log_si_toca`).
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import time

logger = logging.getLogger("las_manos.aviso_extractores")

def _segundos(nombre: str, por_defecto: float, minimo: float) -> float:
    """Lee un intervalo del entorno SIN lanzar nunca (se evalua al importar y
    `server.py` importa este modulo: un `ValueError` tumbaria LAS MANOS entero).
    No numerico o no finito -> el default, con un warning. Menos que `minimo`
    (cero y negativos incluidos) -> `minimo`: un 0 mandaria un Telegram por POST."""
    crudo = os.getenv(nombre)
    if crudo is None or not crudo.strip():
        return por_defecto
    try:
        valor = float(crudo)
    except ValueError:
        valor = float("nan")
    if not math.isfinite(valor):
        logger.warning("%s=%r no es un numero finito; se usa el valor por defecto (%ss)", nombre, crudo, por_defecto)
        return por_defecto
    if valor < minimo:
        logger.warning("%s=%r es menor que el minimo; se usa %ss", nombre, crudo, minimo)
        return minimo
    return valor


#: Sin hardcoding: ajustables por entorno, con los valores decididos por defecto
#: y un piso (aviso 60 s, log 10 s) para que ningun valor mande un aviso por POST.
INTERVALO_AVISO_S = _segundos("JAX_PROCESAMIENTO_AVISO_EXTRACTORES_S", 3600.0, 60.0)
INTERVALO_LOG_S = _segundos("JAX_PROCESAMIENTO_LOG_EXTRACTORES_S", 60.0, 10.0)

#: Referencia fuerte a las tareas en vuelo (anti-GC), con auto-limpieza.
_TAREAS: set[asyncio.Task] = set()
_ultimo_aviso: float | None = None
_ultimo_log: float | None = None


def _reloj() -> float:
    return time.monotonic()


def _reiniciar() -> None:
    """Para las pruebas: olvida las marcas de tiempo."""
    global _ultimo_aviso, _ultimo_log
    _ultimo_aviso = None
    _ultimo_log = None


def mensaje(faltan: list[str]) -> str:
    return (
        "LAS MANOS: faltan dependencias de extraccion de archivos "
        f"({', '.join(faltan)}). El procesamiento de los tipos afectados responde "
        "503 hasta instalar requirements-archivos.txt."
    )


async def _enviar(texto: str) -> None:
    from jacobs.reaper import send_telegram_alert

    try:
        resultado = await send_telegram_alert(texto)
    except Exception as exc:  # fail-soft: el aviso es accesorio; que Telegram falle no puede tumbar el arranque ni el POST, y queda en el log
        logger.error("aviso de extractores: fallo enviando a Telegram (%s)", type(exc).__name__)
        return
    if not resultado.get("ok"):
        logger.error("aviso de extractores NO entregado: %s", resultado.get("error"))


def _lanzar(texto: str) -> bool:
    try:
        tarea = asyncio.get_running_loop().create_task(_enviar(texto))
    except RuntimeError:
        logger.error("aviso de extractores: no hay loop de eventos, no se envia")
        return False
    _TAREAS.add(tarea)
    tarea.add_done_callback(_TAREAS.discard)
    return True


def avisar_al_arrancar(faltan: list[str]) -> bool:
    global _ultimo_aviso
    # DECISION (re-auditoria MINOR-N3): la marca se fija ANTES de saber si el envio
    # llego. Si Telegram esta caido, el aviso queda silenciado hasta que pase el
    # intervalo. Se acepta: reintentar en cada POST convertiria una caida de
    # Telegram en una tormenta de envios, y el fallo no es mudo (queda en el log).
    _ultimo_aviso = _reloj()
    return _lanzar(mensaje(faltan))


def avisar_si_toca(faltan: list[str]) -> bool:
    """True si salio un aviso; False si aun no pasa el intervalo."""
    global _ultimo_aviso
    ahora = _reloj()
    if _ultimo_aviso is not None and (ahora - _ultimo_aviso) < INTERVALO_AVISO_S:
        return False
    # Misma decision que en `avisar_al_arrancar` (MINOR-N3): marca antes de saber si
    # el envio llego; un Telegram caido silencia el aviso hasta el proximo intervalo.
    _ultimo_aviso = ahora
    return _lanzar(mensaje(faltan))


def log_si_toca() -> bool:
    global _ultimo_log
    ahora = _reloj()
    if _ultimo_log is not None and (ahora - _ultimo_log) < INTERVALO_LOG_S:
        return False
    _ultimo_log = ahora
    return True
