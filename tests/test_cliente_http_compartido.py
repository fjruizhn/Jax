"""E-24 (2026-09-16): un httpx.AsyncClient por PROCESO (por event loop), no uno
por llamada. Política 2 de LAS CUATRO DEL RENDIMIENTO: los clientes HTTP se
comparten. Un cliente por llamada abre un socket (y un handshake TLS contra
los proveedores) por pedido: 14 sitios en Jacobs, LAS MANOS, el REPL y la memoria.

Uno por loop y no uno global: un AsyncClient queda atado al loop donde abrió
sus conexiones. Los servicios tienen un loop; los tests, uno por asyncio.run.
Ciclo de vida: cerrar_cliente_http() al apagar (LAS MANOS en shutdown, el REPL
y los workers en su finally); la memoria cierra el suyo en MemoryDB.close().
"""
from __future__ import annotations

import ast
import asyncio
import os
from pathlib import Path
from unittest.mock import patch

import httpx

os.environ["JAX_DB_NAME"] = "jax_memory_test"

from jax.core import cliente_http_compartido as chc  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
_PERMITIDO = (RAIZ / "jax" / "core" / "cliente_http_compartido.py").resolve()


def _construcciones(fuente: str) -> list[int]:
    lineas = []
    for n in ast.walk(ast.parse(fuente)):
        if isinstance(n, ast.Call):
            f = n.func
            nombre = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
            if nombre in ("AsyncClient", "Client"):
                lineas.append(n.lineno)
    return lineas


def test_ningun_modulo_de_servicio_construye_su_propio_cliente():
    hallazgos = []
    for arbol in ("jacobs", "jax", "las_manos"):
        for dirpath, dirnames, filenames in os.walk(RAIZ / arbol):
            dirnames[:] = [d for d in dirnames if d not in {".venv", "__pycache__", "tests"}]
            for nombre in filenames:
                p = Path(dirpath) / nombre
                if (p.suffix != ".py" or nombre.startswith("test_") or nombre.endswith("_test.py")
                        or p.is_symlink() or p.resolve() == _PERMITIDO):
                    continue
                hallazgos += [f"{p.relative_to(RAIZ)}:{l}" for l in _construcciones(p.read_text(encoding="utf-8"))]
    assert hallazgos == [], "usar obtener_cliente_http() (jax/core/cliente_http_compartido.py)"


def test_el_detector_ve_un_cliente_propio():
    assert _construcciones("import httpx\nasync def f():\n    async with httpx.AsyncClient() as c:\n        pass\n") == [3]


def test_dentro_de_un_loop_es_siempre_el_mismo_cliente():
    async def correr():
        a, b = chc.obtener_cliente_http(), chc.obtener_cliente_http()
        await chc.cerrar_cliente_http()
        return a, b
    a, b = asyncio.run(correr())
    assert a is b


def test_otro_loop_recibe_otro_cliente():
    async def uno():
        return chc.obtener_cliente_http()
    assert asyncio.run(uno()) is not asyncio.run(uno())


def test_cerrar_lo_cierra_y_el_siguiente_pedido_trae_uno_nuevo():
    async def correr():
        a = chc.obtener_cliente_http()
        await chc.cerrar_cliente_http()
        b = chc.obtener_cliente_http()
        await chc.cerrar_cliente_http()
        return a, b
    a, b = asyncio.run(correr())
    assert a.is_closed and a is not b


def test_cada_llamada_lleva_su_timeout():
    from jacobs import executor
    vistos = []

    async def handle(transport_self, request):
        vistos.append(request.extensions["timeout"])
        return httpx.Response(200, json={})

    async def correr():
        with patch("httpx.AsyncHTTPTransport.handle_async_request", handle):
            await executor._cancel_motor_job("job-x")

    asyncio.run(correr())
    assert vistos == [{"connect": 5, "read": 5, "write": 5, "pool": 5}]


def test_jacobs_reusa_el_cliente_entre_llamadas():
    from jacobs import executor
    clientes = []

    async def espia(client_self, request, **kwargs):
        clientes.append(client_self)
        return httpx.Response(200, json={}, request=request)

    async def correr():
        with patch.object(httpx.AsyncClient, "send", espia):
            await executor._cancel_motor_job("a")
            await executor._cancel_motor_job("b")

    asyncio.run(correr())
    assert len(clientes) == 2 and clientes[0] is clientes[1]


def test_la_memoria_reusa_su_cliente_y_lo_cierra(monkeypatch):
    from jax.memory import db as dbmod
    monkeypatch.setenv("JAX_OLLAMA_URL", "http://ollama.test:11434")
    clientes = []

    async def espia(client_self, request, **kwargs):
        clientes.append(client_self)
        return httpx.Response(200, json={"embeddings": [[0.1] * dbmod.EMBED.dim]}, request=request)

    async def correr():
        memoria = dbmod.MemoryDB()
        with patch.object(httpx.AsyncClient, "send", espia):
            await memoria.get_embedding("uno")
            await memoria.get_embedding("dos")
        await memoria.close()

    asyncio.run(correr())
    assert len(clientes) == 2 and clientes[0] is clientes[1] and clientes[0].is_closed


def test_las_manos_cierra_el_cliente_al_apagar():
    arbol = ast.parse((RAIZ / "las_manos" / "server.py").read_text(encoding="utf-8"))
    cierres = [
        f for f in ast.walk(arbol) if isinstance(f, ast.AsyncFunctionDef)
        and any(isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "on_event" and d.args
                and getattr(d.args[0], "value", None) == "shutdown" for d in f.decorator_list)
    ]
    assert any("cerrar_cliente_http" in ast.unparse(f) for f in cierres)


def test_el_repl_y_los_workers_cierran_el_cliente_al_salir():
    for rel in ("jax/core/main.py", "jax/memory/worker.py", "jax/memory/synthesis_worker.py", "jax/memory/embedding_worker.py"):
        assert "await cerrar_cliente_http()" in (RAIZ / rel).read_text(encoding="utf-8"), rel
