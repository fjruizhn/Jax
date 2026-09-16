"""Dos carriles para el acceso a Ollama. La Mesa primero.

Spec: §3.4. Consecuencia pre-registrada de U5: p95 de espera en cola
62,71 s contra un umbral de 60.

ENTRE PROCESOS a propósito: la Mesa y el Ejecutor son procesos distintos,
así que un lock por proceso no coordina nada. Se usa `flock` sobre dos
ficheros, como `backup-hall9000.sh`.

`mesa.lock` lo toma la Mesa mientras usa la GPU. El Ejecutor no lo toma
nunca: lo SONDEA. Si está tomado, hay una petición de persona en curso y
el trabajo de máquina espera.
"""
from __future__ import annotations

import fcntl
import time
from contextlib import contextmanager
from pathlib import Path


class EsperaAgotada(RuntimeError):
    """El Ejecutor no consiguió carril antes del tope. La misión FALLA: si
    colarse fuera una opción, la prioridad no existiría."""


def _fichero(raiz, nombre: str) -> Path:
    p = Path(raiz) / nombre
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch(exist_ok=True)
    return p


@contextmanager
def carril_mesa(raiz):
    """La Mesa entra SIEMPRE. Bloquea sólo contra otra petición de Mesa."""
    with open(_fichero(raiz, "mesa.lock"), "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def hay_mesa_esperando(raiz) -> bool:
    """¿Hay una petición de Mesa usando la GPU ahora?"""
    with open(_fichero(raiz, "mesa.lock"), "r+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
        return False


@contextmanager
def carril_ejecutor(raiz, tope_s: float):
    """El Ejecutor entra sólo si no hay Mesa. Se suelta ENTRE PASOS."""
    limite = time.monotonic() + tope_s
    while hay_mesa_esperando(raiz):
        if time.monotonic() >= limite:
            raise EsperaAgotada(
                f"no se consiguió carril en {tope_s}s; la Mesa tiene prioridad")
        time.sleep(0.05)
    with open(_fichero(raiz, "ejecutor.lock"), "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
