"""
LAS MANOS — Motor Registry: tareas asyncio vivas, por job_id.

`POST /motor/job/{id}/cancel` solo reescribía el status en el JobStore, y
worker.run nunca lo leía: el job seguía llamando al modelo, terminaba y se
cobraba (pipeline b8f80733, 2026-09-12: kimi siguió tres minutos después de
que Jacobs abortara el paso). Cancelar tiene que llegar a la tarea que
corre el job; worker.run ya sabe qué hacer con un CancelledError (corta la
llamada en vuelo, marca `cancelled`, registra el costo).

Módulo aparte de routes.py a propósito: routes.py abre el JobStore real al
importarse, y esto tiene que poder probarse sin tocarlo.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio

_RUNNING: dict[str, asyncio.Task] = {}


def register(job_id: str, task: asyncio.Task) -> None:
    """Registra la tarea del job; sale sola del registro al terminar."""
    _RUNNING[job_id] = task
    task.add_done_callback(lambda _t: _RUNNING.pop(job_id, None))


def cancel(job_id: str) -> bool:
    """Cancela la tarea del job si sigue viva. True si había algo que cortar."""
    task = _RUNNING.get(job_id)
    if task is None or task.done():
        return False
    task.cancel()
    return True
