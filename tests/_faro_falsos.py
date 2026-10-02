"""Un MariaDB de mentira para las pruebas de la bitacora y del servicio que no necesitan una base real:
`FalsoPool` imita lo que `EmisorTabla` usa de aiomysql (`acquire()` -> `cursor()` -> `execute`)."""
from __future__ import annotations

import asyncio

COLUMNAS = ("cadena_id", "seq", "momento", "evento", "run_id", "id_correlacion", "decision", "registro", "hash_previo", "hash")


class _Cursor:
    def __init__(self, pool): self.pool = pool
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

    async def execute(self, sql, params=None):
        if self.pool.modo == "colgar":
            await asyncio.sleep(3600)
        if self.pool.modo == "fallar":
            raise OSError("MariaDB caida (falso)")
        self.pool.filas.append(dict(zip(COLUMNAS, params)))


class _Con:
    def __init__(self, pool): self.pool = pool
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    def cursor(self): return _Cursor(self.pool)


class _Adquirir:
    def __init__(self, pool): self.pool = pool

    async def __aenter__(self):
        if self.pool.modo == "colgar_acquire":
            await asyncio.sleep(3600)
        return _Con(self.pool)

    async def __aexit__(self, *a): return False


class FalsoPool:
    def __init__(self, modo="ok"):
        self.modo, self.filas, self.cerrado = modo, [], False
    def acquire(self): return _Adquirir(self)
    def close(self): self.cerrado = True
    async def wait_closed(self): return None
