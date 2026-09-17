"""Candado con nombre de MariaDB para el cupo de MAX_PARALLEL_PIPELINES (ola
final F3, Ruling R31). Contra MariaDB real: dos conexiones = dos procesos
(GET_LOCK es por sesión del servidor, no por proceso de Python). Job
jacobs-gobernanza-db.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid

_db = os.environ.get("JAX_DB_NAME")
if _db and _db != "jax_memory_test":
    raise RuntimeError(f"JAX_DB_NAME={_db!r}: este archivo escribe filas y solo corre contra jax_memory_test.")
os.environ.setdefault("JAX_DB_NAME", "jax_memory_test")

import pytest  # noqa: E402

from jacobs import store  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus  # noqa: E402


async def _uno(sql, params=()):
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            return await cur.fetchone()
    finally:
        conn.close()


async def _borrar(pids):
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            for pid in pids:
                await cur.execute("DELETE FROM jacobs_events WHERE pipeline_id=%s", (pid,))
                await cur.execute("DELETE FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))
    finally:
        conn.close()


def test_dos_procesos_no_superan_el_cupo():
    """Cada "proceso" cuenta y, si hay cupo, tarda y crea un pipeline pending
    (un activo). Sin exclusión entre sesiones los dos ven el mismo conteo y
    crean los dos; con el candado el segundo recuenta después del primero."""
    creados: list[str] = []

    async def proceso(cupo: int):
        async with store.candado_de_activos() as conexion:
            activos = await store.pipeline_count_active(conexion=conexion)
            if activos >= cupo:
                return
            await asyncio.sleep(0.3)  # ensancha la carrera: el otro ya está esperando
            pid = str(uuid.uuid4())
            ahora = time.time()
            await store.pipeline_create(Pipeline(
                pipeline_id=pid, name="t-candado-activos", invoked_by="plataforma",
                mode="autonomous", status=PipelineStatus.pending, created_at=ahora, updated_at=ahora))
            creados.append(pid)

    async def cuerpo():
        await store.init_tables()
        base = await store.pipeline_count_active()
        try:
            await asyncio.gather(proceso(base + 1), proceso(base + 1))
        finally:
            await _borrar(creados)

    asyncio.run(cuerpo())
    assert len(creados) == 1


def test_get_lock_que_vence_falla_cerrado(monkeypatch):
    monkeypatch.setenv("JAX_PREVUELO_CANDADO_TIMEOUT_S", "1")

    async def cuerpo():
        tomado, soltar = asyncio.Event(), asyncio.Event()

        async def dueno():
            async with store.candado_de_activos():
                tomado.set()
                await soltar.wait()

        tarea = asyncio.create_task(dueno())
        # Acotado: si el dueño revienta antes de tomar el candado, el test
        # falla con su excepción en vez de esperar para siempre.
        esperar = asyncio.create_task(tomado.wait())
        await asyncio.wait({tarea, esperar}, timeout=5, return_when=asyncio.FIRST_COMPLETED)
        if not tomado.is_set():
            esperar.cancel()
            await tarea
            raise AssertionError("el dueño no tomó el candado")
        inicio = time.monotonic()
        try:
            with pytest.raises(store.CandadoNoDisponible, match="GET_LOCK"):
                async with store.candado_de_activos():
                    pass
            return time.monotonic() - inicio
        finally:
            soltar.set()
            await tarea

    espera = asyncio.run(cuerpo())
    assert 0.9 <= espera < 5


def test_el_candado_se_suelta_aunque_el_bloque_lance():
    async def cuerpo():
        with pytest.raises(ValueError):
            async with store.candado_de_activos():
                raise ValueError("a mitad de la escritura")
        libre = await _uno("SELECT IS_FREE_LOCK(%s)", (store.nombre_del_candado_de_activos(),))
        inicio = time.monotonic()
        async with store.candado_de_activos():
            pass
        return libre[0], time.monotonic() - inicio

    libre, espera = asyncio.run(cuerpo())
    assert libre == 1
    assert espera < 0.5


def test_el_nombre_del_candado_es_por_base():
    """Los candados con nombre son del SERVIDOR, no de la base: sin la base en
    el nombre, un test contra jax_memory_test serializaría a producción."""
    assert store.nombre_del_candado_de_activos().endswith(":jax_memory_test")
    assert len(store.nombre_del_candado_de_activos()) <= 64


def test_explain_de_las_consultas_del_candado():
    """GET_LOCK/RELEASE_LOCK no leen tablas; el recuento bajo el candado usa
    idx_pipelines_status (LAS CUATRO #1)."""
    async def cuerpo():
        await store.init_tables()
        conn = await store.get_conn()
        try:
            async with conn.cursor() as cur:
                salida = {}
                for nombre, sql, params in (
                    ("tomar", store._SQL_TOMAR_CANDADO, ("x", 0)),
                    ("soltar", store._SQL_SOLTAR_CANDADO, ("x",)),
                    ("contar", store._SQL_CONTAR_ACTIVOS, ()),
                ):
                    await cur.execute("EXPLAIN " + sql, params)
                    cols = [d[0] for d in cur.description]
                    salida[nombre] = [dict(zip(cols, r)) for r in await cur.fetchall()]
                return salida
        finally:
            conn.close()

    e = asyncio.run(cuerpo())
    for nombre in ("tomar", "soltar"):
        assert e[nombre][0]["table"] is None and e[nombre][0]["Extra"] == "No tables used", e[nombre]
    contar = e["contar"][0]
    assert contar["type"] not in ("ALL", "index") and contar["key"] == "idx_pipelines_status", contar
