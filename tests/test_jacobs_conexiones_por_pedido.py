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
import re
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
_SONDEAR_REAL = pv.sonda.sondear  # el fixture la apaga; el caso de sonda la vuelve a usar
PID = "p-conexiones"
REF = 'inline:{"result": "hecho"}'
# Marcos que no son "el sitio" sino la cañería de la conexión.
_CANERIA = {"get_conn", "_db_conn", "_ejecutar_condicional", "conexion_del_pool", "_pool_del_store", "_conexion_o_pool",
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
    def __init__(self, base, conn):
        self._base = base
        self._conn = conn
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
        elif s.startswith("SELECT * FROM jacobs_events"):
            self._filas = ()
        elif s.startswith("SELECT * FROM jacobs_steps"):
            self._filas = b.filas_pasos()
        elif s.startswith("SELECT GET_LOCK") or s.startswith("SELECT RELEASE_LOCK"):
            self._fila = (1,)
        elif s == store._SQL_BLOQUEAR_PIPELINE:
            self._fila = (0, b.status)
        elif s.startswith("SELECT f.transport, f.persona, p.base_url"):
            self._fila = ("http_openai_compat", None, "http://proveedor.invalid/v1", "deepseek", "m1", None)
        elif s.startswith("SELECT encrypted_value FROM credential"):
            self._fila = ("cifrado",)
        elif s.startswith("SELECT price_input_per_1m_usd, price_output_per_1m_usd FROM model WHERE provider_id"):
            self._fila = (Decimal("1"), Decimal("2"))
        elif s.startswith(("UPDATE ", "INSERT ")):
            if b.falla_en and s.startswith(b.falla_en):
                raise pymysql.err.OperationalError(2013, "Lost connection to MySQL server during query")
            if b.cuelga_en and s.startswith(b.cuelga_en):
                await asyncio.sleep(3600)
            escritura = (s, params, self._conn.sitio)
            if self._conn.en_transaccion:
                self._conn.pendientes.append(escritura)
            else:
                b.escrituras.append(escritura)  # autocommit: confirmada al escribir
            return 1
        else:
            raise AssertionError(f"consulta no prevista por el doble: {s[:120]}")
        return 0

    async def fetchone(self):
        return self._fila

    async def fetchall(self):
        return self._filas


class _Conexion:
    def __init__(self, base, sitio):
        self._base = base
        self.sitio = sitio
        self.en_transaccion = False
        self.pendientes: list = []
        self.closed = False
        self._reader = _Lector()
        self.last_usage = time.monotonic()

    def cursor(self, *_clase):
        return _Cursor(self._base, self)

    def close(self):
        # Cerrar la sesión descarta lo no confirmado (como el servidor).
        self.closed = True
        self.pendientes.clear()
        self.en_transaccion = False

    async def ensure_closed(self):
        self.closed = True

    def get_transaction_status(self):
        return False

    async def begin(self):
        self.en_transaccion = True

    async def commit(self):
        if self._base.commit_falla is not None:
            # El corte llega DESPUÉS de que el servidor recibió el COMMIT: el
            # doble confirma y el cliente ve el error (el caso incierto).
            self._base.escrituras.extend(self.pendientes)
            self.pendientes.clear()
            raise self._base.commit_falla
        if self._base.commit_cuelga:
            await asyncio.sleep(3600)
        self._base.escrituras.extend(self.pendientes)
        self.pendientes.clear()
        self.en_transaccion = False

    async def rollback(self):
        self.pendientes.clear()
        self.en_transaccion = False


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
        self.escrituras: list[tuple[str, object, str]] = []  # CONFIRMADAS: (sql, params, sitio de la conexión)
        self.conexiones: list = []
        self.vivas_max = 0
        self.falla_en: str | None = None
        self.commit_falla: BaseException | None = None
        self.commit_cuelga = False
        self.cuelga_en: str | None = None
        self.falla_al_conectar: BaseException | None = None

    def _abrir(self, via, kwargs):
        if self.falla_al_conectar is not None:
            raise self.falla_al_conectar
        sitio = _sitio()
        self.aperturas.append((via, sitio, "client_flag" in kwargs))
        conn = _Conexion(self, sitio)
        self.conexiones.append(conn)
        vivas = sum(1 for c in self.conexiones if not c.closed)
        self.vivas_max = max(self.vivas_max, vivas)
        return conn

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


def _pedido_crear(mode="autonomous"):
    return routes.create_pipeline(PipelineCreateRequest(
        name="t", objective="o", invoked_by="plataforma", mode=mode,
        steps=[StepSpec(facet="jekyll", capability="research", prompt="p")]), BackgroundTasks())


def _pedido_continue():
    return routes.continue_pipeline(PID, routes.ContinueRequest(invoked_by="plataforma"), BackgroundTasks())


def _pedido_resume():
    return routes.resume_pipeline(PID, routes.ResumeRequest(invoked_by="plataforma"), BackgroundTasks())


def _pedido_eventos():
    return routes.get_events(PID)


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


def test_el_pool_nunca_tiene_mas_conexiones_vivas_que_su_tamano(entorno, monkeypatch):
    """Control de R53 (2026-09-17): en la instancia aislada se contaron 11
    conexiones del proceso con JAX_DB_POOL_MAX=10. Medido: eran 10 a MariaDB
    (:3308) + 1 socket HTTP entrante (:17790) del propio servicio; el pool
    nunca pasó de 10 (1.000 muestras de `ss -tn "( dport = :3308 )"` durante
    lotes c=1/10/25/50 y cuatro sostenidas c=50). Acá se fija la propiedad en
    proceso: con N pedidos a la vez nunca hay más conexiones ABIERTAS Y SIN
    CERRAR que el tamaño del pool. Control: verde también antes de R53; se
    rompe con un pool que abra de más.

    QUÉ ASERCIÓN SOSTIENE LA AFIRMACIÓN (observación de la re-revisión final):
    la del CONTEO DE APERTURAS (`len(base.aperturas) <= 4`). Ésa es la que un
    pool que abra de más no puede pasar. `vivas_max` mide el máximo de
    conexiones abiertas y sin cerrar EN EL INSTANTE de cada apertura, así que
    con el mismo tope de aperturas nunca puede ser mayor: acompaña, no
    prueba."""
    monkeypatch.setenv("JAX_DB_POOL_MAX", "4")
    base = entorno()
    _catalogo_vacio(monkeypatch)

    async def cuerpo():
        try:
            await asyncio.gather(*[_pedido_preflight() for _ in range(60)])
        finally:
            await store.cerrar_pool()

    asyncio.run(cuerpo())
    assert len(base.aperturas) <= 4, base.aperturas  # la que sostiene la afirmación
    assert base.vivas_max <= 4, base.vivas_max


def test_get_events_no_abre_conexion_por_pedido(entorno):
    """m3 de la re-revisión final: GET /jacobs/pipeline/{id}/events era el
    último endpoint HTTP con conexión propia por pedido -- la misma forma que
    a c=50 dio 0/13/49/61 % de errores en get_motor_governance (R38).
    Expected contra 1da52bc: [('directa', 'events_by_pipeline', False)]."""
    base = entorno()
    assert _medir(base, _pedido_eventos) == []


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
    tipos = [p[2] for s, p, _ in base.escrituras if s.startswith("INSERT INTO jacobs_events")]
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


# ---------------------------------------------------------------------------
# Crear es atómico (R38, fix round 1, 2b) y dry_run (3)
# ---------------------------------------------------------------------------

def _escrituras_de_crear(base):
    return [(s.split(" (")[0].split(" SET")[0], sitio) for s, _p, sitio in base.escrituras]


def test_crear_escribe_todo_en_una_transaccion_sobre_la_conexion_del_candado(entorno, monkeypatch):
    """Pipeline, paso y PIPELINE_CREATED se confirman juntos por la conexión
    del GET_LOCK. Expected contra 1d84e82: cada escritura por su propia
    conexión del pool, en autocommit (sitios pipeline_create, step_upsert,
    event_append)."""
    base = entorno()
    monkeypatch.setattr(routes, "prevuelo", _prevuelo_que_lee_del_pool)

    async def cuerpo():
        try:
            return await _pedido_crear()
        finally:
            await store.cerrar_pool()

    asyncio.run(cuerpo())
    assert _escrituras_de_crear(base) == [
        ("INSERT INTO jacobs_pipelines", "candado_de_activos"),
        ("INSERT INTO jacobs_steps", "candado_de_activos"),
        ("INSERT INTO jacobs_events", "candado_de_activos"),
    ]


def test_crear_con_la_base_que_corta_a_mitad_da_503_sin_nada_escrito(entorno, monkeypatch):
    """La base corta al escribir el paso: nada queda confirmado (ni el
    pipeline sin pasos) y el pedido es 503 prevuelo_no_disponible.
    Expected contra 1d84e82: el pipeline quedaba confirmado y salía el
    OperationalError sin atrapar (un 500)."""
    base = entorno()
    base.falla_en = "INSERT INTO jacobs_steps"
    monkeypatch.setattr(routes, "prevuelo", _prevuelo_que_lee_del_pool)

    async def cuerpo():
        try:
            with pytest.raises(HTTPException) as exc:
                await _pedido_crear()
            return exc.value
        finally:
            await store.cerrar_pool()

    error = asyncio.run(cuerpo())
    assert error.status_code == 503 and error.detail["code"] == "prevuelo_no_disponible"
    assert base.escrituras == []


def test_crear_con_la_base_colgada_a_mitad_da_503_acotado_sin_nada_escrito(entorno, monkeypatch):
    """La escritura del paso se cuelga: el pedido vence a
    JAX_DB_CONNECT_TIMEOUT_SECONDS (1 s acá), responde 503 y no confirma nada.
    Expected contra 1d84e82: el pedido queda colgado (TimeoutError del
    wait_for de 5 s del test) con el pipeline ya confirmado."""
    base = entorno()
    base.cuelga_en = "INSERT INTO jacobs_steps"
    monkeypatch.setattr(routes, "prevuelo", _prevuelo_que_lee_del_pool)

    async def cuerpo():
        try:
            inicio = time.monotonic()
            with pytest.raises(HTTPException) as exc:
                await asyncio.wait_for(_pedido_crear(), 5)
            return exc.value, time.monotonic() - inicio
        finally:
            await store.cerrar_pool()

    error, espera = asyncio.run(cuerpo())
    assert error.status_code == 503 and error.detail["code"] == "prevuelo_no_disponible"
    assert espera < 3
    assert base.escrituras == []


def test_dry_run_solo_abre_el_candado_y_completa_en_la_misma_transaccion(entorno, monkeypatch):
    """Item 3 de la revisión: dry_run escribe además el completed y
    DRY_RUN_COMPLETE. Van en la transacción de crear (un dry_run no queda
    pending si la base corta después de crear).
    Expected contra 1d84e82: UPDATE y DRY_RUN_COMPLETE por el pool, fuera de
    la transacción."""
    base = entorno()
    monkeypatch.setattr(routes, "prevuelo", _prevuelo_que_lee_del_pool)
    assert _medir(base, lambda: _pedido_crear("dry_run")) == [("directa", "candado_de_activos", False)]
    segundo = _escrituras_de_crear(base)[len(_escrituras_de_crear(base)) // 2:]
    assert segundo == [
        ("INSERT INTO jacobs_pipelines", "candado_de_activos"),
        ("INSERT INTO jacobs_steps", "candado_de_activos"),
        ("INSERT INTO jacobs_events", "candado_de_activos"),
        ("UPDATE jacobs_pipelines", "candado_de_activos"),
        ("INSERT INTO jacobs_events", "candado_de_activos"),
    ]


# ---------------------------------------------------------------------------
# La rama de la sonda (R38, fix round 1, 1)
# ---------------------------------------------------------------------------

def _catalogo_que_sondea(monkeypatch):
    """jekyll con contrato, precio y credencial, SIN evento de salud: el
    pre-vuelo real la sondea. Los lectores del catálogo van mockeados (su
    conexión del pool ya está contada en los tests de arriba)."""
    from decimal import Decimal as D

    async def resolver(pasos, *, conexion):
        return {}

    async def leer(*, conexion, **_kw):
        return pv.Catalogo(
            {"jekyll": pv.prevuelo_catalogo.FilaFaceta(
                "jekyll", "http_openai_compat", None, "deepseek", "http://proveedor.invalid/v1", "m1")},
            {("deepseek", "m1"): pv.FilaModelo("max_tokens", 8192, D("1"), D("2"))},
            {}, frozenset({"deepseek"}), {},
        )

    monkeypatch.setattr(pv.prevuelo_catalogo, "resolver_motores", resolver)
    monkeypatch.setattr(pv.prevuelo_catalogo, "leer_catalogo", leer)


def _sonda_sin_red(monkeypatch, tmp_path):
    import credential_resolver
    import facet_resolver

    monkeypatch.setattr(pv.sonda, "sondear", _SONDEAR_REAL)
    llamada = AsyncMock(return_value={"usage": {"prompt_tokens": 3, "completion_tokens": 1}})
    monkeypatch.setattr(pv.sonda, "_post", llamada)  # ningún proveedor real
    monkeypatch.setattr(credential_resolver, "decrypt_secret", lambda _v: "clave-de-prueba")
    monkeypatch.setattr(credential_resolver, "_cache", {})
    monkeypatch.setattr(facet_resolver, "_cache", {})
    monkeypatch.setattr(facet_resolver, "FACET_SEAL_PATH", str(tmp_path / "sin-sello"))
    return llamada


def test_la_sonda_escribe_por_el_pool_y_los_resolvedores_solo_leen_sin_cache(entorno, monkeypatch, tmp_path):
    """Un pre-vuelo con salud vieja sondea. Primer pedido: los dos
    resolvedores COMPARTIDOS con el ejecutor (facet_resolver,
    credential_resolver) abren su conexión directa con la caché fría --
    excepción justificada en r38-report.md (módulos espejados con
    jax-platform, caché de 30 s por clave). Segundo pedido, dentro del TTL:
    ninguna conexión directa; el evento de salud y el uso de la sonda van por
    el pool. Expected contra 1d84e82: en el segundo pedido, directas de
    registrar_evento_de_sonda y record_direct_usage."""
    base = entorno()
    _catalogo_que_sondea(monkeypatch)
    llamada = _sonda_sin_red(monkeypatch, tmp_path)

    async def cuerpo():
        try:
            primero = await _pedido_preflight()
            n = len(base.aperturas)
            segundo = await _pedido_preflight()
            return primero, base.aperturas[:n], base.aperturas[n:], segundo
        finally:
            await store.cerrar_pool()

    primero, aperturas_1, aperturas_2, segundo = asyncio.run(cuerpo())
    assert primero["sondeadas"] == ["jekyll"] and segundo["sondeadas"] == ["jekyll"]
    assert llamada.await_count == 2
    assert [a for a in aperturas_1 if a[0] == "directa"] == [
        ("directa", "_query_facet", False),
        ("directa", "_query_active_credential", False),
    ]
    assert aperturas_2 == []
    tipos = [s.split(" (")[0] for s, _p, _sitio in base.escrituras]
    assert tipos == ["INSERT INTO facet_health_event", "INSERT INTO axioma_usage"] * 2


def test_sonda_con_la_base_caida_al_resolver_da_503(entorno, monkeypatch, tmp_path):
    """R13/R16 se conservan: una caída real de la base al resolver la faceta
    se propaga y el pedido es 503 prevuelo_no_disponible, nunca faceta_caida
    ni un veredicto. (Control: el cambio no toca los resolvedores.)"""
    base = entorno()
    _catalogo_que_sondea(monkeypatch)
    llamada = _sonda_sin_red(monkeypatch, tmp_path)
    real = base.directa

    async def directa_caida(*a, **kw):
        raise pymysql.err.OperationalError(2003, "Can't connect to MySQL server")

    monkeypatch.setattr(aiomysql, "connect", directa_caida)

    async def cuerpo():
        try:
            with pytest.raises(HTTPException) as exc:
                await _pedido_preflight()
            return exc.value
        finally:
            await store.cerrar_pool()

    error = asyncio.run(cuerpo())
    assert real is not None
    assert error.status_code == 503 and error.detail["code"] == "prevuelo_no_disponible"
    llamada.assert_not_awaited()


# ---------------------------------------------------------------------------
# Motor Registry: sus event_append corren en el loop del servicio (R38, 4)
# ---------------------------------------------------------------------------

_MOTOR_REGISTRY = RAIZ / "las_manos" / "motor_registry"
# Lo que ejecutaría código de pool en OTRO loop u otro hilo.
_OTRO_LOOP_U_HILO = {"new_event_loop", "run_coroutine_threadsafe", "run_in_executor",
                     "ThreadPoolExecutor", "Thread", "set_event_loop"}


def test_los_event_append_del_motor_registry_son_await_en_corrutinas_del_loop():
    """Control con evidencia (revisión de 1d84e82, 4): el pool es por loop y un
    loop ajeno vivo se niega. Los que llaman event_append en el Motor Registry
    (tool_authority.py y worker.py) lo hacen con `await` dentro de `async def`,
    y el trabajo nace con asyncio.create_task en motor_registry/routes.py
    (el loop de uvicorn). El único to_thread de worker.py envuelve
    subprocess.run, no una corrutina. En el paquete no hay nada que cree otro
    loop ni que corra corrutinas en un hilo."""
    import ast

    llamadas = 0
    for archivo in ("tool_authority.py", "worker.py"):
        arbol = ast.parse((_MOTOR_REGISTRY / archivo).read_text(encoding="utf-8"))
        padres = {hijo: nodo for nodo in ast.walk(arbol) for hijo in ast.iter_child_nodes(nodo)}
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Call) and getattr(nodo.func, "id", None) == "event_append":
                llamadas += 1
                assert isinstance(padres[nodo], ast.Await), f"{archivo}:{nodo.lineno} sin await"
                ancestro = padres[nodo]
                while not isinstance(ancestro, (ast.AsyncFunctionDef, ast.FunctionDef, ast.Module)):
                    ancestro = padres[ancestro]
                assert isinstance(ancestro, ast.AsyncFunctionDef), f"{archivo}:{nodo.lineno} fuera de async def"
            if isinstance(nodo, ast.Call) and getattr(nodo.func, "attr", None) == "to_thread":
                objetivo = nodo.args[0]
                assert getattr(objetivo, "attr", getattr(objetivo, "id", None)) in {
                    "run", "write_result"}, f"{archivo}:{nodo.lineno} to_thread de {ast.dump(objetivo)}"
    assert llamadas == 4  # tool_authority 3 + worker 1
    for ruta in _MOTOR_REGISTRY.glob("*.py"):
        if ruta.name.endswith("_test.py"):
            continue
        for nodo in ast.walk(ast.parse(ruta.read_text(encoding="utf-8"))):
            nombre = getattr(nodo, "attr", None) or getattr(nodo, "id", None)
            assert nombre not in _OTRO_LOOP_U_HILO, f"{ruta.name}:{nodo.lineno} usa {nombre}"
    rutas = ast.parse((_MOTOR_REGISTRY / "routes.py").read_text(encoding="utf-8"))
    assert any(isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "create_task"
               for n in ast.walk(rutas))


def test_un_event_append_del_motor_registry_usa_el_pool_del_loop_que_lo_corre(entorno):
    """La misma evidencia en ejecución: un rechazo de tool_authority escribe su
    evento por el pool creado en el loop que corre la corrutina."""
    from motor_registry import tool_authority

    base = entorno()

    async def cuerpo():
        try:
            await tool_authority._reject(job_id="j1", tool_name="read_file", caller="kimi", reason="x")
            return asyncio.get_running_loop(), store._pool_estado[0]
        finally:
            await store.cerrar_pool()

    corriendo, duenio = asyncio.run(cuerpo())
    assert duenio is corriendo
    assert [s.split(" (")[0] for s, _p, _x in base.escrituras] == ["INSERT INTO jacobs_events"]


# ---------------------------------------------------------------------------
# Gobernanza: UNA lectura por pedido (Ruling R43)
# ---------------------------------------------------------------------------

def _contar_gobernanza(monkeypatch):
    lecturas = []
    real = store.get_motor_governance

    async def contada():
        lecturas.append(1)
        return await real()

    monkeypatch.setattr(store, "get_motor_governance", contada)
    return lecturas


def _un_pedido(pedido):
    async def cuerpo():
        try:
            return await pedido()
        finally:
            await store.cerrar_pool()
    return asyncio.run(cuerpo())


def test_preflight_lee_la_gobernanza_una_vez(entorno, monkeypatch):
    """R43: build() leía la gobernanza y _validate_plan_capabilities la volvía
    a leer. Expected contra 36e539f: `assert 2 == 1`."""
    entorno()
    _catalogo_vacio(monkeypatch)
    lecturas = _contar_gobernanza(monkeypatch)
    _un_pedido(_pedido_preflight)
    assert len(lecturas) == 1


def test_crear_con_pasos_lee_la_gobernanza_una_vez(entorno, monkeypatch):
    """Expected contra 36e539f: `assert 2 == 1`."""
    entorno()
    monkeypatch.setattr(routes, "prevuelo", _prevuelo_que_lee_del_pool)
    lecturas = _contar_gobernanza(monkeypatch)
    _un_pedido(_pedido_crear)
    assert len(lecturas) == 1


def test_crear_por_objetivo_lee_la_gobernanza_una_vez(entorno, monkeypatch):
    """El camino del planificador (qwen, sin red): build() y el parseo del
    plan que devolvió el LLM usan la MISMA foto. Expected contra 36e539f:
    `assert 3 == 1` (build, _parse_plan_json y _validate_plan_capabilities)."""
    from types import SimpleNamespace

    import httpx

    from jacobs import plan as plan_mod

    entorno()
    monkeypatch.setattr(routes, "prevuelo", _prevuelo_que_lee_del_pool)
    lecturas = _contar_gobernanza(monkeypatch)
    monkeypatch.setattr(plan_mod, "resolve_facet", AsyncMock(return_value=SimpleNamespace(
        key="jax_local", transport="ollama", provider_id="ollama", model="qwen", base_url=None)))
    monkeypatch.setattr(plan_mod, "limite_de_salida", AsyncMock(return_value={"options": {"num_predict": 512}}))
    monkeypatch.setattr(plan_mod, "record_resolved_version_safe", AsyncMock())

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"model": "qwen", "message": {"content": json.dumps(
                [{"facet": "jekyll", "capability": "research", "prompt": "p"}])}}

    async def post(_self, url, json=None, **kw):
        assert url == plan_mod.OLLAMA_URL  # nunca un proveedor pago
        return _Resp()

    monkeypatch.setattr(httpx.AsyncClient, "post", post)

    def crear_por_objetivo():
        return routes.create_pipeline(PipelineCreateRequest(
            name="t", objective="o", invoked_by="plataforma", mode="autonomous"), BackgroundTasks())

    respuesta = _un_pedido(crear_por_objetivo)
    assert respuesta["step_count"] == 1
    assert len(lecturas) == 1


# ---------------------------------------------------------------------------
# COMMIT cortado: resultado incierto (Ruling R41)
# ---------------------------------------------------------------------------

def _crear_y_capturar_503(base, monkeypatch):
    monkeypatch.setattr(routes, "prevuelo", _prevuelo_que_lee_del_pool)

    async def cuerpo():
        try:
            with pytest.raises(HTTPException) as exc:
                await asyncio.wait_for(_pedido_crear(), 5)
            return exc.value
        finally:
            await store.cerrar_pool()

    return asyncio.run(cuerpo())


def test_commit_cortado_da_503_que_avisa_que_el_pipeline_puede_existir(entorno, monkeypatch):
    """La conexión se corta DURANTE el COMMIT: el servidor pudo haberlo
    confirmado. El 503 conserva code prevuelo_no_disponible y agrega un
    `detalle` que lo dice. Expected contra 36e539f: KeyError 'detalle'."""
    base = entorno()
    base.commit_falla = pymysql.err.OperationalError(2013, "Lost connection to MySQL server during query")
    error = _crear_y_capturar_503(base, monkeypatch)
    assert error.status_code == 503 and error.detail["code"] == "prevuelo_no_disponible"
    assert "puede existir" in error.detail["detalle"]
    assert "lista" in error.detail["detalle"] and "reintentar" in error.detail["detalle"]


def test_commit_colgado_que_vence_tambien_es_incierto(entorno, monkeypatch):
    """El COMMIT no responde y vence el plazo: tampoco se sabe si confirmó.
    Expected contra 36e539f: KeyError 'detalle'."""
    base = entorno()
    base.commit_cuelga = True
    error = _crear_y_capturar_503(base, monkeypatch)
    assert error.status_code == 503 and error.detail["code"] == "prevuelo_no_disponible"
    assert "puede existir" in error.detail["detalle"]


def test_falla_antes_del_commit_no_dice_incierto(entorno, monkeypatch):
    """Control: si falla una escritura ANTES del COMMIT, nada quedó escrito y
    el 503 no sugiere que el pipeline pueda existir."""
    base = entorno()
    base.falla_en = "INSERT INTO jacobs_steps"
    error = _crear_y_capturar_503(base, monkeypatch)
    assert error.status_code == 503 and error.detail["code"] == "prevuelo_no_disponible"
    assert "detalle" not in error.detail
    assert base.escrituras == []


def test_el_docstring_de_crear_declara_el_commit_incierto():
    """R41: el contrato se declara donde se lee el endpoint.
    Expected contra 36e539f: no lo menciona."""
    doc = routes.create_pipeline.__doc__ or ""
    assert "COMMIT" in doc and "incierto" in doc


# ---------------------------------------------------------------------------
# Excepción documentada: resolvedores directos en la rama de la sonda (R39)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ruta", ["jax/core/facet_resolver.py", "las_manos/credential_resolver.py"])
def test_la_conexion_directa_de_los_resolvedores_declara_por_que(ruta):
    """Ruling R39: facet_resolver y credential_resolver quedan con conexión
    propia (excepción aceptada). La razón se escribe EN el sitio, en el bloque
    de comentarios inmediatamente arriba de `_db_conn` (fuera del segmento que
    compara scripts/check_mirror_sync.py, así no es drift con jax-platform).
    Expected contra 36e539f: no hay nota."""
    lineas = (RAIZ / ruta).read_text(encoding="utf-8").splitlines()
    i = next(n for n, l in enumerate(lineas) if l.startswith("async def _db_conn("))
    bloque = []
    while i > 0 and lineas[i - 1].startswith("#"):
        i -= 1
        bloque.insert(0, lineas[i])
    nota = "\n".join(bloque)
    for parte in ("R39", "sonda", "ok fresco", "un vuelo por clave", "30 s", "jax-platform", "camino caliente"):
        assert parte in nota, f"{ruta}: falta '{parte}' en la nota sobre _db_conn:\n{nota}"


# ---------------------------------------------------------------------------
# Motor Registry: los trabajos de fondo esperan turno sin plazo (R38 fix round 3, N1)
# ---------------------------------------------------------------------------

def _trabajo_del_motor_registry(tmp_path, llamar):
    """Un job real de motor_registry.worker.run con el transporte falso
    `llamar` (mismo arnés que tests/test_motor_job_cancel_and_length.py)."""
    import test_motor_job_cancel_and_length as arnes
    from motor_registry import worker
    from motor_registry.catalog import MotorCatalog
    from motor_registry.job_store import JobStore

    tienda = JobStore(str(tmp_path / "jobs.jsonl"))
    job_id = tienda.create(caller="jacobs", capability="implementation", motor="kimi",
                           trace_id="t", prompt="prompt", recursion_depth=0)

    async def correr():
        with patch.dict(worker._TRANSPORT_DISPATCH, {"http_openai_compat": llamar}), \
             patch.object(worker, "resolve_credential_instrumented", AsyncMock(return_value="sk-fake")), \
             patch("contrato_dispatch._leer_contrato", AsyncMock(return_value=("max_tokens", 131072))), \
             patch("motor_registry.usage_writer.record_motor_usage", AsyncMock()), \
             patch("httpx.AsyncClient.post", AsyncMock(side_effect=AssertionError("red real"))):
            await worker.run(
                job_id=job_id, motor="kimi", capability="implementation", prompt="prompt",
                context={}, store=tienda, catalog=MotorCatalog(arnes._CFG),
                kill_switch_path=str(tmp_path / "PAUSE"),
            )
    return correr


def _llamada_que_rechaza_una_herramienta():
    from motor_registry import tool_authority

    async def llamar(**_kw):
        await tool_authority._reject(job_id="job-x", tool_name="write_file", caller="kimi", reason="fuera del sandbox")
        return {"choices": [{"message": {"content": "listo"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    return llamar


def test_un_job_del_motor_registry_espera_turno_y_audita_con_el_pool_lleno(entorno, monkeypatch, tmp_path):
    """N1 (re-review): motor_registry/routes.py lanza worker.run con create_task
    desde el handler HTTP, que tiene la espera acotada. Con el pool de 1
    ocupado 1,5 s (JAX_DB_CONNECT_TIMEOUT_SECONDS=1) y la base SANA, el
    TOOL_CALL_REJECTED se perdía (TimeoutError en el fail-soft). El job es
    trabajo de fondo: espera turno sin plazo.
    Expected contra bd977a4: `assert [] == ['TOOL_CALL_REJECTED']`."""
    _pool_de_uno(monkeypatch)
    base = entorno()
    trabajo = _trabajo_del_motor_registry(tmp_path, _llamada_que_rechaza_una_herramienta())

    async def cuerpo():
        try:
            acaparador = await _acaparar_el_pool(1.5)
            # Como en routes.py: la tarea nace en el contexto del pedido (acotado).
            assert store.turno_sin_plazo() is False
            await asyncio.wait_for(asyncio.create_task(trabajo()), 10)
            await acaparador
        finally:
            await store.cerrar_pool()

    asyncio.run(cuerpo())
    tipos = [p[2] for s, p, _ in base.escrituras if s.startswith("INSERT INTO jacobs_events")]
    assert tipos == ["TOOL_CALL_REJECTED"]


def test_un_job_del_motor_registry_con_la_base_caida_no_se_cuelga(entorno, monkeypatch, tmp_path):
    """Control fail-closed: sin plazo de TURNO, abrir la conexión sigue acotado;
    con la base caída el rechazo no se escribe (fail-soft que ya existía,
    logueado) y el job termina en segundos, no queda colgado."""
    base = entorno()
    base.falla_al_conectar = pymysql.err.OperationalError(2003, "Can't connect to MySQL server")
    trabajo = _trabajo_del_motor_registry(tmp_path, _llamada_que_rechaza_una_herramienta())

    async def cuerpo():
        try:
            inicio = time.monotonic()
            await asyncio.wait_for(asyncio.create_task(trabajo()), 10)
            return time.monotonic() - inicio
        finally:
            await store.cerrar_pool()

    assert asyncio.run(cuerpo()) < 5
    assert base.escrituras == []


# ---------------------------------------------------------------------------
# Texto viejo del pool (R38 fix round 3, m1)
# ---------------------------------------------------------------------------

_FRASES_VIEJAS_DEL_POOL = re.compile(r"\bpool de lectura\b|\bpool del pre-vuelo\b|\bes de lectura\b", re.IGNORECASE)
_ARCHIVOS_DEL_POOL = ("jacobs/store.py", "jacobs/prevuelo.py", "jacobs/prevuelo_catalogo.py",
                      "las_manos/motor_registry/routes.py", "las_manos/server.py",
                      "scripts/perfil_prevuelo.py", "tools/jacobs_relaunch.py",
                      "tests/test_prevuelo_pool.py", "tests/test_prevuelo_pool_db.py",
                      "tests/test_prevuelo_orquestador.py")


def test_el_pool_del_store_no_se_describe_como_de_lectura_ni_del_prevuelo():
    """Re-review m1: desde R38 el pool es del store entero y lleva escrituras
    sin condición; la razón de no llevar CLIENT.FOUND_ROWS es la semántica de
    las escrituras condicionales, no "es de lectura". El texto viejo engaña a
    quien decida qué meter en el pool.
    Expected contra bd977a4: store.py:70/110, motor_registry/routes.py:105,
    perfil_prevuelo.py:11, test_prevuelo_pool.py:226,
    test_prevuelo_orquestador.py:55, test_prevuelo_pool_db.py:1."""
    hallazgos = []
    for ruta in _ARCHIVOS_DEL_POOL:
        for n, linea in enumerate((RAIZ / ruta).read_text(encoding="utf-8").splitlines(), 1):
            if linea.lstrip().startswith("_FRASES_VIEJAS_DEL_POOL"):
                continue
            if _FRASES_VIEJAS_DEL_POOL.search(linea):
                hallazgos.append(f"{ruta}:{n}: {linea.strip()[:90]}")
    assert hallazgos == []


def test_store_lista_todas_las_funciones_que_usan_el_pool():
    """m1: el comentario de R38 en store.py enumera quién va por el pool; tiene
    que nombrar también a los dos escritores de la sonda (fix round 1, 1).
    Expected contra bd977a4: faltan registrar_evento_de_sonda y
    record_direct_usage."""
    texto = (RAIZ / "jacobs/store.py").read_text(encoding="utf-8")
    bloque = texto[texto.index("# Ruling R38 (2026-09-17): el pool es del STORE"):texto.index("_pool_estado: tuple[")]
    for nombre in ("registrar_evento_de_sonda", "record_direct_usage", "event_append", "get_motor_governance"):
        assert nombre in bloque, nombre


# ---------------------------------------------------------------------------
# Capas: motor_registry importa jacobs (m5 de la re-revisión final)
# ---------------------------------------------------------------------------

def test_la_dependencia_de_motor_registry_sobre_jacobs_esta_declarada():
    """m5: `worker.py` importa `jacobs.store` (capa alta) en tiempo de import
    desde el Motor Registry (capa baja). Funciona -- hay symlink
    las_manos/jacobs -> ../jacobs y todos los jobs de CI lo tienen -- pero
    tiene que estar DICHO donde se lee, y `jacobs/usage_writer.py` no puede
    seguir afirmando que los dos módulos no se importan entre sí.
    Expected contra 49ea939: falta la declaración en worker.py y
    usage_writer.py dice 'jacobs NO importa motor_registry' a secas."""
    worker = (RAIZ / "las_manos" / "motor_registry" / "worker.py").read_text(encoding="utf-8")
    cabeza = worker[:worker.index("import traceback")]
    for parte in ("jacobs.store", "symlink", "PYTHONPATH"):
        assert parte in cabeza, f"worker.py no declara la dependencia ({parte})"
    uso = (RAIZ / "jacobs" / "usage_writer.py").read_text(encoding="utf-8")
    # No basta con que siga diciendo "jacobs NO importa motor_registry" (sigue
    # siendo cierto): tiene que decir que la dirección contraria SÍ existe.
    assert "worker.py importa jacobs.store" in uso
    assert "la direccion CONTRARIA si existe" in uso


def test_el_ejecutor_y_el_motor_registry_comparten_la_misma_marca_de_turno():
    """Control del riesgo que hace que la marca NO se mueva a jax/core: si
    `espera_de_turno_sin_plazo` viviera en un módulo espejado, importarlo
    'a secas' desde las_manos y como `jax.core...` desde jacobs daría DOS
    módulos y DOS ContextVar -- la marca del job no la vería el pool y N1
    volvería en silencio. Acá se fija que es el MISMO objeto."""
    from motor_registry import worker

    assert worker.espera_de_turno_sin_plazo is store.espera_de_turno_sin_plazo
    with worker.espera_de_turno_sin_plazo():
        assert store.turno_sin_plazo() is True
    assert store.turno_sin_plazo() is False


def test_el_candado_de_creacion_declara_lo_que_serializa_hoy(entorno, monkeypatch):
    """m6: el comentario de T2 describía una sección crítica más chica que la
    real (hoy cubre build + pre-vuelo con sondas + transacción, y continuar
    toma el mismo objeto). Se fija que el texto lo diga y que el objeto sea
    realmente compartido. Expected contra 977fbe0: el texto no menciona el
    pre-vuelo ni la sonda."""
    from jacobs import candado

    texto = (RAIZ / "jacobs" / "candado.py").read_text(encoding="utf-8")
    for parte in ("pre-vuelo", "sonda", "continuar", "MISMO objeto"):
        assert parte in texto, parte
    assert continuar.candado_de_creacion is candado.candado_de_creacion
    assert routes._pipeline_create_lock is candado.candado_de_creacion
