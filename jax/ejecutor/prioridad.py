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

import asyncio
import fcntl
import os
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from jax.ejecutor.cita import Motivo

ESPERA_AGOTADA = "espera_agotada"

#: Cada cuánto re-intenta un hilo que espera `mesa.lock`, y cada cuánto sondea
#: el Ejecutor. No son topes: sólo la granularidad del sondeo.
_PASO_MESA_S = 0.01
_PASO_EJECUTOR_S = 0.05


class EsperaAgotada(RuntimeError):
    """El Ejecutor no consiguió carril antes del tope. La misión FALLA: si
    colarse fuera una opción, la prioridad no existiría."""


def _abrir(raiz, nombre: str):
    """El lock, abierto de SÓLO LECTURA (y creado si falta). `flock` no necesita escritura.

    SP3 (2026-09-17): la Mesa, el proxy y un arnés de otra cuenta comparten el directorio
    (grupo común, setgid). El fichero nace con el umask de quien llega primero (0644): con
    `r+` el otro miembro del grupo no podía abrirlo y su carril reventaba con PermissionError.
    Quien lo crea lo crea con su umask: esto no le da escritura a nadie."""
    p = Path(raiz) / nombre
    p.parent.mkdir(parents=True, exist_ok=True)
    return os.fdopen(os.open(p, os.O_RDONLY | os.O_CREAT | os.O_CLOEXEC, 0o660), "rb")


@contextmanager
def carril_mesa(raiz):
    """La Mesa entra SIEMPRE. Bloquea sólo contra otra petición de Mesa."""
    with _abrir(raiz, "mesa.lock") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def hay_mesa_esperando(raiz) -> bool:
    """¿Hay una petición de Mesa usando la GPU ahora?"""
    with _abrir(raiz, "mesa.lock") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
        return False


@contextmanager
def carril_ejecutor(raiz, tope_s: float):
    """El Ejecutor entra sólo si no hay Mesa. Se suelta ENTRE PASOS.

    El tope cubre TODA la espera, también la de otro Ejecutor. CORREGIDO
    2026-09-16 (§3.4 bis): antes, pasada la Mesa, esperaba `ejecutor.lock` con
    un `LOCK_EX` bloqueante sin tope — con el proxy, dos peticiones paralelas
    de Claude Code colgaban la segunda en vez de fallarla — y no re-miraba a
    la Mesa tras tomarlo."""
    limite = time.monotonic() + tope_s
    toma = _Toma()
    try:
        while True:
            _intentar_ejecutor(raiz, toma)
            if toma.tomada:
                break
            if time.monotonic() >= limite:
                # Sin prosa (política del ecosistema): el argumento es un Motivo.
                raise EsperaAgotada(Motivo(ESPERA_AGOTADA, (("tope_s", tope_s),)))
            time.sleep(_PASO_EJECUTOR_S)
        yield
    finally:
        toma.abandonar()


# ---------------------------------------------------------------------------
# Versiones ASYNC (§3.4 bis). El worker de la Mesa y el proxy del Ejecutor
# viven dentro de un event loop: un `flock` bloqueante ahí congela a TODOS
# (tercera de LAS CUATRO DEL RENDIMIENTO). El flock se toma en un hilo con
# `asyncio.to_thread` y la espera del Ejecutor es `asyncio.sleep`.
#
# Por qué el hilo no hace `LOCK_EX` bloqueante sino `LOCK_NB` con reintento:
# un hilo bloqueado en `flock` NO se puede cancelar. Si la tarea se cancela
# mientras espera, ese hilo seguiría ahí hasta que el otro suelte, tomaría el
# lock para nadie, y además retendría un hilo del executor (que `asyncio.run`
# espera hasta 300 s al cerrar). Con reintento, cancelar corta la espera
# dentro de un paso.
# ---------------------------------------------------------------------------


def _soltar(f) -> None:
    try:
        fcntl.flock(f, fcntl.LOCK_UN)
    finally:
        f.close()


class _Toma:
    """El traspaso de un lock tomado en un hilo al código async, a prueba de
    cancelación. El hilo ENTREGA el fichero ya bloqueado; si la tarea se
    canceló antes, el hilo lo suelta él mismo. `_guarda` cierra la carrera
    entre «el hilo acaba de tomarlo» y «la tarea acaba de abandonar»."""

    def __init__(self) -> None:
        self._guarda = threading.Lock()
        self.abandonada = threading.Event()
        self._f = None

    @property
    def tomada(self) -> bool:
        return self._f is not None

    def entregar(self, f) -> None:
        """Hilo: `f` ya tiene el flock."""
        with self._guarda:
            if not self.abandonada.is_set():
                self._f = f
                return
        _soltar(f)

    def abandonar(self) -> None:
        """Loop: salida normal, excepción o cancelación. Idempotente. Sólo hace
        syscalls que no bloquean (LOCK_UN y close)."""
        with self._guarda:
            self.abandonada.set()
            f, self._f = self._f, None
        if f is not None:
            _soltar(f)


def _esperar_mesa(raiz, toma: _Toma, paso_s: float) -> None:
    """Hilo: espera `mesa.lock` hasta tomarlo o hasta que abandonen."""
    f = _abrir(raiz, "mesa.lock")
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if toma.abandonada.wait(paso_s):
                f.close()
                return
            continue
        toma.entregar(f)
        return


def _intentar_ejecutor(raiz, toma: _Toma) -> None:
    """Hilo, UN intento sin bloquear: entra sólo si no hay Mesa, si
    `ejecutor.lock` está libre, y si sigue sin haber Mesa después de tomarlo
    (una Mesa que llegó en medio pasa primero)."""
    if hay_mesa_esperando(raiz):
        return
    f = _abrir(raiz, "ejecutor.lock")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return
    if hay_mesa_esperando(raiz):
        _soltar(f)
        return
    toma.entregar(f)


@asynccontextmanager
async def carril_mesa_async(raiz):
    """Como `carril_mesa`, sin congelar el loop. La cancelación de la tarea
    (esperando o dentro) suelta el lock."""
    toma = _Toma()
    try:
        await asyncio.to_thread(_esperar_mesa, raiz, toma, _PASO_MESA_S)
        yield
    finally:
        toma.abandonar()


@asynccontextmanager
async def carril_ejecutor_async(raiz, tope_s: float):
    """Como `carril_ejecutor`, sin congelar el loop. El tope cubre TODA la
    espera: la Mesa y otro Ejecutor. Vencido → `EsperaAgotada`, no se cuela."""
    limite = time.monotonic() + tope_s
    toma = _Toma()
    try:
        while True:
            await asyncio.to_thread(_intentar_ejecutor, raiz, toma)
            if toma.tomada:
                break
            if time.monotonic() >= limite:
                raise EsperaAgotada(Motivo(ESPERA_AGOTADA, (("tope_s", tope_s),)))
            await asyncio.sleep(_PASO_EJECUTOR_S)
        yield
    finally:
        toma.abandonar()
