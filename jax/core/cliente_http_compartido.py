"""Cliente HTTP compartido por proceso (E-24, 2026-09-16).

Política 2 de LAS CUATRO DEL RENDIMIENTO: los clientes HTTP se comparten, no se
rehacen por request. Un httpx.AsyncClient por llamada abre un socket nuevo (y un
handshake TLS contra los proveedores) en cada pedido.

Uno por EVENT LOOP, no uno global: un AsyncClient queda atado al loop donde abrió
sus conexiones. Los servicios (uvicorn, REPL, workers) tienen un solo loop; los
tests traen uno por asyncio.run, y reusar el cliente de un loop cerrado revienta
con "Event loop is closed". Se guarda el último par (loop, cliente): si el loop
cambió o el cliente se cerró, se crea otro.

Invalidación / ciclo de vida: `cerrar_cliente_http()` al apagar (LAS MANOS en
shutdown; REPL y workers en su finally). El timeout lo pone CADA llamada; el del
cliente queda en el default de httpx (5 s), así que una llamada que lo olvide cae
en 5 s y no en "sin límite".

Un solo archivo real: las_manos/cliente_http_compartido.py es symlink.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio

import httpx

_loop: asyncio.AbstractEventLoop | None = None
_cliente: httpx.AsyncClient | None = None


def crear_cliente_http() -> httpx.AsyncClient:
    """ÚNICO lugar del árbol de servicio que construye un AsyncClient (lo vigila
    tests/test_cliente_http_compartido.py). MemoryDB lo usa para el suyo propio."""
    return httpx.AsyncClient()


def obtener_cliente_http() -> httpx.AsyncClient:
    global _loop, _cliente
    loop = asyncio.get_running_loop()
    if _cliente is None or _cliente.is_closed or _loop is not loop:
        _cliente = crear_cliente_http()
        _loop = loop
    return _cliente


async def cerrar_cliente_http() -> None:
    global _loop, _cliente
    if _cliente is not None and _loop is asyncio.get_running_loop():
        await _cliente.aclose()
    _cliente = None
    _loop = None
