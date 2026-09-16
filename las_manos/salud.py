"""Comprobaciones de salud de LAS MANOS, sin depender del servidor.

POR QUE VIVE APARTE (2026-09-16). Estaba dentro de `server.py`, y eso hacia que
su test necesitara importar el modulo entero —— FastAPI, el planner, la policy,
los workers, el motor registry, jacobs——. En CI ese import fallaba y los cinco
tests se SALTABAN en silencio: el test que demuestra que /health puede ponerse
rojo no corria justo donde importa. Un test que nadie ejecuta afirma un estado
que no existe.

Aca dentro no hay nada que importar salvo la capa de datos, asi que la logica
se prueba siempre. `server.py` sigue siendo quien decide el codigo HTTP.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Awaitable, Callable

# Cada cuanto se vuelve a comprobar de verdad. El endpoint recibe 20.000 req/s
# en las pruebas de carga y una consulta por peticion tumbaria la base que el
# chequeo vigila —— el remedio seria peor que la enfermedad.
TTL_SEGUNDOS = 5.0


class Salud:
    """Estado de las dependencias, cacheado con TTL y sin estampida."""

    def __init__(self, comprobaciones: dict[str, Callable[[], Awaitable[None]]],
                 ttl: float = TTL_SEGUNDOS, reloj: Callable[[], float] = time.monotonic):
        self._comprobaciones = comprobaciones
        self._ttl = ttl
        self._reloj = reloj
        self._lock = asyncio.Lock()
        self._t = 0.0
        self._valor: dict | None = None

    async def _medir(self) -> dict:
        fallos: list[str] = []
        for nombre, comprobar in self._comprobaciones.items():
            try:
                await comprobar()
            except Exception as exc:  # fail-closed: si una dependencia no responde, el servicio NO esta sano y lo dice
                fallos.append(f"{nombre}: {type(exc).__name__}: {exc}")
        return {"ok": not fallos, "fallos": fallos}

    async def estado(self) -> dict:
        ahora = self._reloj()
        if self._valor is not None and (ahora - self._t) < self._ttl:
            return self._valor
        async with self._lock:
            ahora = self._reloj()
            if self._valor is not None and (ahora - self._t) < self._ttl:
                return self._valor
            self._valor = await self._medir()
            self._t = ahora
            return self._valor

    def comprobado_hace(self) -> float:
        return round(self._reloj() - self._t, 2)

    def invalidar(self) -> None:
        """Para los tests: obliga a volver a medir."""
        self._valor = None
        self._t = 0.0


async def comprobar_base(get_conn) -> None:
    """La base responde. De ahi salen pipelines, catalogo de motores y reaper."""
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1")
            await cur.fetchone()
    finally:
        conn.close()


async def comprobar_audit(log_path: Path) -> None:
    """El log forense es escribible. Sin audit, ejecutar algo seria dejar una
    accion sin rastro —— y eso no es un servicio sano, es uno peligroso."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a"):
        pass
