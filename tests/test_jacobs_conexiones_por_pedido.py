"""Conexiones que abre CADA pedido de los cinco endpoints que corren el
pre-vuelo (Ruling R38, LAS CUATRO #2 y #4).

Origen medido (task-15-report.md, "Re-medición HTTP"): a c=50 sostenido,
`/jacobs/preflight` dio 0 %, 13 %, 49 % y 61 % de errores en cuatro corridas
iguales, con `pymysql.err.OperationalError: (2013, 'Lost connection to MySQL
server during query')` en `store.get_motor_governance()` -> `get_conn()`: una
conexión NUEVA por pedido, fuera del pool de la Task 15b. El inventario de
r38-report.md encontró lo mismo en todo el camino de crear, continue, resume y
approve-step (pipeline_get, steps_by_pipeline, pipeline_count_active,
pipeline_create, step_upsert, event_append).

La base es un doble que CUENTA cada conexión abierta y de dónde: por
`aiomysql.connect` (directa) o por `aiomysql.pool.connect` (lo que llama el
Pool REAL de aiomysql al llenarse). Contesta las consultas que los cinco
caminos hacen de verdad; una consulta que no conoce falla el test, así el
inventario no se queda corto en silencio. Ningún socket real: test puro.

Quedan DEDICADAS a propósito, y el test lo fija:
- `candado_de_activos` (F3, Ruling R31): GET_LOCK vive en la sesión; una
  conexión del pool que vuelve al pool se llevaría el candado a otro pedido.
- las escrituras condicionales con CLIENT.FOUND_ROWS (`pipeline_tomar_epoca`
  por `_ejecutar_condicional`, `continuar_transaccion`): sin el flag un UPDATE
  que escribe los mismos valores cuenta 0 filas y la época se daría por
  perdida; el pool no lleva el flag.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import aiomysql  # noqa: E402
import aiomysql.pool  # noqa: E402
import pymysql  # noqa: E402
import pytest  # noqa: E402
from fastapi import BackgroundTasks, HTTPException  # noqa: E402

from jacobs import continuar, policy, routes, store  # noqa: E402
from jacobs import prevuelo as pv  # noqa: E402
from jacobs.models import PipelineCreateRequest, PipelineStatus, StepSpec  # noqa: E402
from jacobs.prevuelo_reglas import CostoPaso, Veredicto  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
PID = "p-conexiones"
REF = 'inline:{"result": "hecho"}'
# Marcos que no son "el sitio" sino la cañería de la conexión.
_CANERIA = {"get_conn", "_ejecutar_condicional", "conexion_del_pool", "_pool_del_store",
            "__aenter__", "__aexit__", "_correr_endpoint"}


class _Lector:
    """Lo que el Pool de aiomysql mira de `conn._reader` al reusar."""

    def __init__(self):
        self.eof = False
        self.eof_received = False

    def at_eof(self):
        return self.eof

    def exception(self):
        return None


class _Cursor:
    def __init__(self, base):
        self._base = base
        self._fila = None
        self._filas = ()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, sql, params=None):
        self._fila, self._filas = None, ()
        s = " ".join(sql.split())
        b = self._base
        if s.startswith("SELECT `key`, has_tool_access FROM motor"):
            self._filas = (("kimi", 0),)
        elif s.startswith("SELECT `key`, risk_level"):
            self._filas = (("research", "low", 0, 0, 15, 0, "", None, "manual_only",
                            json.dumps(["plataforma"]), None, None),)
        elif s.startswith("SELECT capability_key, motor_key FROM capability_motor"):
            self._filas = ()
        elif s == store._SQL_CONTAR_ACTIVOS:
            self._fila = (0,)
        elif s.startswith("SELECT * FROM jacobs_pipelines"):
            self._fila = b.fila_pipeline()
        elif s.startswith("SELECT * FROM jacobs_steps"):
            self._filas = b.filas_pasos()
        elif s.startswith("SELECT GET_LOCK") or s.startswith("SELECT RELEASE_LOCK"):
            self._fila = (1,)
        elif s == store._SQL_BLOQUEAR_PIPELINE:
            self._fila = (0, b.status)
        elif s.startswith(("UPDATE ", "INSERT ")):
            b.escrituras.append((s, params))
            return 1
        else:
            raise AssertionError(f"consulta no prevista por el doble: {s[:120]}")
        return 0

    async def fetchone(self):
        return self._fila

    async def fetchall(self):
        return self._filas


class _Conexion:
    def __init__(self, base):
        self._base = base
        self.closed = False
        self._reader = _Lector()
        self.last_usage = time.monotonic()

    def cursor(self, *_clase):
        return _Cursor(self._base)

    def close(self):
        self.closed = True

    async def ensure_closed(self):
        self.closed = True

    def get_transaction_status(self):
        return False

    async def begin(self):
        pass

    async def commit(self):
        pass

    async def rollback(self):
        pass


def _sitio() -> str:
    """El primer marco del repo que no es cañería de conexión: la función que
    pidió la conexión (store.get_motor_governance, prevuelo, ...)."""
    for info in inspect.stack()[2:]:
        ruta = Path(info.filename)
        if RAIZ not in ruta.parents or "tests" in ruta.parts:
            continue
        if info.function in _CANERIA:
            continue
        return info.function
    return "?"


class _Base:
    def __init__(self, status: str, pasos: str):
        self.status = status
        self.pasos = pasos
        self.aperturas: list[tuple[str, str, bool]] = []  # (via, sitio, found_rows)
        self.escrituras: list[tuple[str, object]] = []
        self.falla_al_conectar: BaseException | None = None

    def _abrir(self, via, kwargs):
        if self.falla_al_conectar is not None:
            raise self.falla_al_conectar
        self.aperturas.append((via, _sitio(), "client_flag" in kwargs))
        return _Conexion(self)

    async def directa(self, *_a, **kwargs):
        return self._abrir("directa", kwargs)

    async def del_pool(self, *_a, **kwargs):
        return self._abrir("pool", kwargs)

    def fila_pipeline(self):
        return {
            "pipeline_id": PID, "name": "t", "invoked_by": "plataforma", "user_id": None,
            "tenant_id": None, "owner_ack_at": None, "run_epoch": 0, "mode": "autonomous",
            "status": self.status, "plan": "[]", "current_step_index": 1, "max_steps": 20,
            "context_refs": json.dumps({"objective": "o", "step_0_ref": REF}),
            "created_at": 1.0, "updated_at": 1.0,
        }

    def filas_pasos(self):
        comun = {"pipeline_id": PID, "motor": None, "capability": "research",
                 "input_ref": json.dumps({"prompt": "p"}), "timeout_seconds": 300,
                 "retries_allowed": 0, "skip_on_fail": 0, "trace_id": "", "started_at": None,
                 "finished_at": None, "error": None}
        return (
            {**comun, "step_id": "s0", "step_index": 0, "facet": "jekyll", "output_ref": REF,
             "status": "completed", "depends_on": "[]"},
            {**comun, "step_id": "s1", "step_index": 1, "facet": "hyde", "output_ref": None,
             "status": self.pasos, "depends_on": "[0]"},
        )


def _veredicto_ok():
    costo = CostoPaso(1, "hyde", None, 1, 0, 0, Decimal("0"), "acotado")
    return Veredicto(ok=True, violaciones=(), costo_max_usd=Decimal("0"),
                     pasos_costo=(costo,), sondeadas=())


async def _prevuelo_que_lee_del_pool(*_a, **_k):
    """El pre-vuelo real toma UNA conexión del pool para el catálogo (Task
    15b, cubierto en test_prevuelo_pool.py); acá se modela esa lectura."""
    async with store.conexion_del_pool():
        pass
    return _veredicto_ok()


@pytest.fixture
def entorno(monkeypatch):
    monkeypatch.setenv("JAX_DB_HOST", "127.0.0.1")
    monkeypatch.setenv("JAX_DB_PORT", "1")
    monkeypatch.setenv("JAX_DB_CONNECT_TIMEOUT_SECONDS", "1")
    monkeypatch.delenv("JAX_DB_POOL_MAX", raising=False)
    monkeypatch.delenv("JAX_PREVUELO_DB_POOL_MAX", raising=False)
    from jacobs import executor

    for modulo in (policy, routes, continuar, executor):
        monkeypatch.setattr(modulo, "check_kill_switch", lambda: False)
    monkeypatch.setattr(pv.sonda, "sondear", AsyncMock(side_effect=AssertionError("sin sondas")))

    def armar(status="aborted", pasos="failed"):
        base = _Base(status, pasos)
        monkeypatch.setattr(aiomysql, "connect", base.directa)
        monkeypatch.setattr(aiomysql.pool, "connect", base.del_pool)
        return base

    return armar


def _pedido_preflight():
    return routes.preflight(routes.PreflightRequest(
        invoked_by="plataforma", steps=[StepSpec(facet="jekyll", capability="research", prompt="p")]))


def _pedido_crear():
    return routes.create_pipeline(PipelineCreateRequest(
        name="t", objective="o", invoked_by="plataforma", mode="autonomous",
        steps=[StepSpec(facet="jekyll", capability="research", prompt="p")]), BackgroundTasks())


def _pedido_continue():
    return routes.continue_pipeline(PID, routes.ContinueRequest(invoked_by="plataforma"), BackgroundTasks())


def _pedido_resume():
    return routes.resume_pipeline(PID, routes.ResumeRequest(invoked_by="plataforma"), BackgroundTasks())


def _pedido_approve():
    return routes.approve_step(PID, routes.ApproveStepRequest(invoked_by="plataforma"), BackgroundTasks())


def _medir(base, pedido):
    """Dos pedidos seguidos en UN loop; devuelve las aperturas del SEGUNDO (el
    primero puede llenar el pool). El pool se cierra en el mismo loop."""
    async def cuerpo():
        try:
            await pedido()
            antes = len(base.aperturas)
            await pedido()
            return base.aperturas[antes:]
        finally:
            await store.cerrar_pool()
    return asyncio.run(cuerpo())


def _catalogo_vacio(monkeypatch):
    """El pre-vuelo REAL de /preflight con sus dos lectores mockeados: toma la
    conexión del pool (lo que se cuenta) sin simular el catálogo entero."""
    async def resolver(pasos, *, conexion):
        return {}

    async def leer(*, conexion, **_kw):
        return pv.Catalogo({}, {}, {}, frozenset(), {})

    monkeypatch.setattr(pv.prevuelo_catalogo, "resolver_motores", resolver)
    monkeypatch.setattr(pv.prevuelo_catalogo, "leer_catalogo", leer)


# ---------------------------------------------------------------------------
# Conexiones por pedido
# ---------------------------------------------------------------------------

def test_preflight_no_abre_ninguna_conexion_por_pedido(entorno, monkeypatch):
    """Expected contra 2fd3778: dos conexiones directas de
    get_motor_governance (build() y _validate_plan_capabilities) ->
    `assert [('directa', 'get_motor_governance', False), ...] == []`."""
    base = entorno()
    _catalogo_vacio(monkeypatch)
    assert _medir(base, _pedido_preflight) == []


def test_crear_solo_abre_la_conexion_dedicada_del_candado(entorno, monkeypatch):
    """Expected contra 2fd3778: además del candado, directas de
    pipeline_count_active, get_motor_governance x2, pipeline_create,
    step_upsert y event_append."""
    base = entorno()
    monkeypatch.setattr(routes, "prevuelo", _prevuelo_que_lee_del_pool)
    assert _medir(base, _pedido_crear) == [("directa", "candado_de_activos", False)]


def test_continue_solo_abre_el_candado_y_la_transaccion_con_found_rows(entorno, monkeypatch):
    """Expected contra 2fd3778: además, directas de pipeline_get,
    steps_by_pipeline, get_motor_governance y pipeline_count_active."""
    base = entorno(status="aborted", pasos="failed")
    monkeypatch.setattr(continuar, "prevuelo", _prevuelo_que_lee_del_pool)
    assert _medir(base, _pedido_continue) == [
        ("directa", "candado_de_activos", False),
        ("directa", "continuar_transaccion", True),
    ]


def test_resume_solo_abre_la_escritura_condicional_de_la_epoca(entorno, monkeypatch):
    """Expected contra 2fd3778: además, directas de pipeline_get,
    steps_by_pipeline, event_append y step_upsert."""
    base = entorno(status="interrupted", pasos="blocked")
    monkeypatch.setattr(routes, "prevuelo", _prevuelo_que_lee_del_pool)
    assert _medir(base, _pedido_resume) == [("directa", "pipeline_tomar_epoca", True)]


def test_approve_step_solo_abre_la_escritura_condicional_de_la_epoca(entorno, monkeypatch):
    """Expected contra 2fd3778: además, directas de pipeline_get,
    steps_by_pipeline, event_append y step_upsert."""
    base = entorno(status="interrupted", pasos="blocked_human_gate")
    monkeypatch.setattr(routes, "prevuelo", _prevuelo_que_lee_del_pool)
    assert _medir(base, _pedido_approve) == [("directa", "pipeline_tomar_epoca", True)]


def test_cincuenta_preflight_concurrentes_no_superan_el_tamano_del_pool(entorno, monkeypatch):
    """El defecto medido era de concurrencia: 50 pedidos a la vez abrían 100
    conexiones de gobernanza. Con el pool, a lo sumo JAX_DB_POOL_MAX
    en todo el proceso. Expected contra 2fd3778: 100 directas + 3 del pool."""
    monkeypatch.setenv("JAX_DB_POOL_MAX", "3")
    base = entorno()
    _catalogo_vacio(monkeypatch)

    async def cuerpo():
        try:
            await asyncio.gather(*[_pedido_preflight() for _ in range(50)])
        finally:
            await store.cerrar_pool()

    asyncio.run(cuerpo())
    assert [a for a in base.aperturas if a[0] == "directa"] == []
    assert 1 <= len(base.aperturas) <= 3


# ---------------------------------------------------------------------------
# Base caída: fail-closed
# ---------------------------------------------------------------------------

def test_preflight_con_la_base_caida_da_503_prevuelo_no_disponible(entorno, monkeypatch):
    """Spec §8: sin pre-vuelo, 503 prevuelo_no_disponible, nunca un 500
    genérico. Expected contra 2fd3778: el OperationalError de
    get_motor_governance sale de build() sin atrapar (el 500 de la carga)."""
    base = entorno()
    _catalogo_vacio(monkeypatch)
    base.falla_al_conectar = pymysql.err.OperationalError(2003, "Can't connect to MySQL server")

    async def cuerpo():
        try:
            with pytest.raises(HTTPException) as exc:
                await _pedido_preflight()
            return exc.value
        finally:
            await store.cerrar_pool()

    error = asyncio.run(cuerpo())
    assert error.status_code == 503
    assert error.detail["code"] == "prevuelo_no_disponible"
    assert "OperationalError" in error.detail["motivo"]


def test_preflight_con_rechazo_del_plan_sigue_dando_422(entorno, monkeypatch):
    """Control: el 503 nuevo no se come el rechazo del plan (PlanRejected)."""
    entorno()
    _catalogo_vacio(monkeypatch)
    from jacobs.plan import PlanRejected, PlanViolation

    rechazo = PlanRejected([PlanViolation(0, "jekyll", None, "research", "no")])
    monkeypatch.setattr(routes._plan_builder, "build", AsyncMock(side_effect=rechazo))

    async def cuerpo():
        try:
            with pytest.raises(HTTPException) as exc:
                await _pedido_preflight()
            return exc.value
        finally:
            await store.cerrar_pool()

    error = asyncio.run(cuerpo())
    assert error.status_code == 422 and error.detail["code"] == "plan_rechazado"


def test_crear_con_la_base_caida_conserva_su_error_y_no_escribe(entorno, monkeypatch):
    """Control (create conserva su comportamiento): el error de la base sube
    tal cual desde el primer acceso (el recuento de activos), sin pre-vuelo ni
    escrituras."""
    base = entorno()
    base.falla_al_conectar = pymysql.err.OperationalError(2003, "Can't connect to MySQL server")
    prevuelo = AsyncMock(return_value=_veredicto_ok())
    monkeypatch.setattr(routes, "prevuelo", prevuelo)

    async def cuerpo():
        try:
            with pytest.raises(pymysql.err.OperationalError):
                await _pedido_crear()
        finally:
            await store.cerrar_pool()

    asyncio.run(cuerpo())
    prevuelo.assert_not_awaited()
    assert base.aperturas == []


# ---------------------------------------------------------------------------
# Contención del pool: los trabajos de fondo esperan turno (R38, fix round 1, 2a)
# ---------------------------------------------------------------------------

def _pool_de_uno(monkeypatch):
    # Los dos nombres: 1d84e82 leía JAX_PREVUELO_DB_POOL_MAX, hoy JAX_DB_POOL_MAX.
    monkeypatch.setenv("JAX_DB_POOL_MAX", "1")
    monkeypatch.setenv("JAX_PREVUELO_DB_POOL_MAX", "1")


async def _acaparar_el_pool(segundos: float) -> asyncio.Task:
    """Toma la única conexión del pool y la retiene `segundos` (más que
    JAX_DB_CONNECT_TIMEOUT_SECONDS=1): contención con la base SANA."""
    ocupada = asyncio.Event()

    async def acaparar():
        async with store.conexion_del_pool():
            ocupada.set()
            await asyncio.sleep(segundos)

    tarea = asyncio.ensure_future(acaparar())
    await ocupada.wait()
    return tarea


def test_el_ejecutor_espera_turno_con_el_pool_lleno_y_termina(entorno, monkeypatch):
    """La revisión de 1d84e82: `event_append` de STEP_STARTED está fuera del
    try (executor.py) -- un turno del pool que vence salía del gather y mataba
    run_pipeline con el paso en running. Con la base sana, el ejecutor espera
    turno sin plazo y la corrida termina.
    Expected contra 1d84e82: TimeoutError desde run_pipeline."""
    from jacobs import executor
    from jacobs.models import Pipeline, PipelineStatus, Step

    _pool_de_uno(monkeypatch)
    base = entorno()
    monkeypatch.setattr(store, "pipeline_update_status_si_epoca", AsyncMock(return_value=True))
    monkeypatch.setattr(store, "pipeline_epoca_y_status", AsyncMock(return_value=(0, PipelineStatus.running)))
    monkeypatch.setattr(store, "step_upsert_si_epoca", AsyncMock(return_value=True))
    monkeypatch.setattr(executor, "_dispatch_step", AsyncMock(return_value={"result": "ok"}))
    monkeypatch.setattr(executor, "_persist_step_to_repo", AsyncMock())
    monkeypatch.setattr(executor, "save_if_large", lambda *a, **k: (None, {"result": "ok"}))
    pipeline = Pipeline(
        pipeline_id=PID, name="t", invoked_by="plataforma", mode="autonomous",
        status=PipelineStatus.running, run_epoch=0,
        plan=[Step(pipeline_id=PID, step_index=0, facet="jekyll", capability="research",
                   input={"prompt": "p"})],
    )

    async def cuerpo():
        try:
            tarea = await _acaparar_el_pool(1.5)
            await asyncio.wait_for(executor.run_pipeline(pipeline), 10)
            await tarea
        finally:
            await store.cerrar_pool()

    asyncio.run(cuerpo())
    tipos = [p[2] for s, p in base.escrituras if s.startswith("INSERT INTO jacobs_events")]
    assert tipos == ["PIPELINE_STARTED", "WAVE_STARTED", "STEP_STARTED", "STEP_COMPLETED",
                     "WAVE_COMPLETED", "PIPELINE_COMPLETED"]


def test_un_pedido_http_con_el_pool_lleno_sigue_dando_503_acotado(entorno, monkeypatch):
    """Control del otro lado de la decisión: fuera del ejecutor y del reaper
    la espera de turno sigue acotada por JAX_DB_CONNECT_TIMEOUT_SECONDS y el
    pedido responde 503 prevuelo_no_disponible, no queda colgado."""
    _pool_de_uno(monkeypatch)
    entorno()
    _catalogo_vacio(monkeypatch)

    async def cuerpo():
        try:
            tarea = await _acaparar_el_pool(3)
            inicio = time.monotonic()
            with pytest.raises(HTTPException) as exc:
                await asyncio.wait_for(_pedido_preflight(), 10)
            espera = time.monotonic() - inicio
            tarea.cancel()
            await asyncio.gather(tarea, return_exceptions=True)
            return exc.value, espera
        finally:
            await store.cerrar_pool()

    error, espera = asyncio.run(cuerpo())
    assert error.status_code == 503 and error.detail["code"] == "prevuelo_no_disponible"
    assert espera < 2.5


def test_sin_plazo_de_turno_la_base_caida_sigue_fallando_cerrado(entorno, monkeypatch):
    """Fail-closed para una caída REAL: sin plazo de turno, abrir la conexión
    sigue acotado por connect_timeout y el error sube (el ejecutor no se
    cuelga esperando una base muerta)."""
    base = entorno()
    base.falla_al_conectar = pymysql.err.OperationalError(2003, "Can't connect to MySQL server")

    async def cuerpo():
        try:
            with store.espera_de_turno_sin_plazo():
                with pytest.raises(pymysql.err.OperationalError):
                    await store.event_append(PID, "X")
        finally:
            await store.cerrar_pool()

    asyncio.run(cuerpo())


def test_el_reaper_barre_sin_plazo_de_turno(monkeypatch):
    """El reaper es el otro trabajo de fondo que escribe (REAPED): su barrido
    corre con la misma espera sin plazo que el ejecutor.
    Expected contra 1d84e82: AttributeError (no existía la marca)."""
    from jacobs import reaper

    vistas = []

    async def barrido():
        vistas.append(store.turno_sin_plazo())

    class _Corte(Exception):
        pass

    async def dormir(_s):
        raise _Corte

    monkeypatch.setattr(reaper, "reap_orphaned_pipelines", barrido)
    monkeypatch.setattr(reaper, "check_facet_health", AsyncMock())
    monkeypatch.setattr(reaper.asyncio, "sleep", dormir)
    with pytest.raises(_Corte):
        asyncio.run(reaper.start_reaper_loop())
    assert vistas == [True]
    assert store.turno_sin_plazo() is False
