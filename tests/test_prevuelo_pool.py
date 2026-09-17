"""El pre-vuelo no abre conexiones por pedido (Task 15b, LAS CUATRO #2).

Origen medido (task-15-report.md y task-15b-report.md): cada pre-vuelo abría
dos conexiones aiomysql nuevas -- `MotorCatalog.from_db()` y
`prevuelo_catalogo.leer_catalogo()` -- sin pool. Ahora las dos sacan su
conexión de UN pool por proceso (`jacobs.store.conexion_de_lectura`), creado
perezosamente y cerrado en el shutdown de LAS MANOS y al salir del CLI.

Los dobles reemplazan `aiomysql.connect` (lo que llamaba el código viejo) y
`aiomysql.pool.connect` (lo que llama el Pool REAL de aiomysql al llenarse):
el pool que se prueba es el de aiomysql, no uno de mentira. Ninguna
conexión real: tests puros.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import time
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import aiomysql  # noqa: E402
import aiomysql.pool  # noqa: E402
import pymysql  # noqa: E402
import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from jacobs import prevuelo as pv  # noqa: E402
from jacobs import routes, store  # noqa: E402
from jacobs.models import Step  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]


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
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, sql, params=None):
        self._conn.consultas += 1
        if self._conn.falla_al_consultar is not None:
            raise self._conn.falla_al_consultar

    async def fetchall(self):
        return ()

    async def fetchone(self):
        return None


class _Conexion:
    def __init__(self, n):
        self.n = n
        self.closed = False
        self.consultas = 0
        self.falla_al_consultar: BaseException | None = None
        self._reader = _Lector()
        self.last_usage = time.monotonic()

    def cursor(self):
        return _Cursor(self)

    def close(self):
        self.closed = True

    async def ensure_closed(self):
        self.closed = True

    def get_transaction_status(self):
        return False


class _Base:
    """Cuenta cada conexión que se abre, por cualquiera de los dos caminos."""

    def __init__(self):
        self.abiertas: list[_Conexion] = []
        self.falla_al_conectar: BaseException | None = None
        self.cuelga_al_conectar = False
        self.kwargs: list[dict] = []

    async def conectar(self, *args, **kwargs):
        self.kwargs.append(kwargs)
        if self.falla_al_conectar is not None:
            raise self.falla_al_conectar
        if self.cuelga_al_conectar:
            await asyncio.sleep(3600)
        conn = _Conexion(len(self.abiertas))
        self.abiertas.append(conn)
        return conn


@pytest.fixture
def base(monkeypatch):
    b = _Base()
    monkeypatch.setenv("JAX_DB_HOST", "127.0.0.1")
    monkeypatch.setenv("JAX_DB_PORT", "1")
    monkeypatch.setenv("JAX_DB_CONNECT_TIMEOUT_SECONDS", "1")
    monkeypatch.delenv("JAX_PREVUELO_DB_POOL_MAX", raising=False)
    monkeypatch.setattr(aiomysql, "connect", b.conectar)
    monkeypatch.setattr(aiomysql.pool, "connect", b.conectar)
    monkeypatch.setattr(pv.sonda, "sondear", AsyncMock(side_effect=AssertionError("sin sondas en este test")))
    yield b


def _plan():
    # Una faceta HTTP (leer_catalogo) y una de motor (MotorCatalog.from_db):
    # los dos lectores del catálogo en un mismo pre-vuelo.
    return [
        Step(pipeline_id="p", step_index=0, facet="jekyll", capability="research", input={"prompt": "x"}),
        Step(pipeline_id="p", step_index=1, facet="kimi", capability="generate", input={"prompt": "x"}),
    ]


def _correr(cuerpo):
    """Un event loop por test, y el pool se cierra en ESE loop (el pool queda
    atado al loop que lo creó)."""
    async def envuelto():
        try:
            return await cuerpo()
        finally:
            # getattr: contra el código anterior al pool (fcb0c6a) no existe, y
            # el rojo tiene que ser el del conteo, no un AttributeError.
            cerrar = getattr(store, "cerrar_pool", None)
            if cerrar is not None:
                await cerrar()
    return asyncio.run(envuelto())


def test_un_prevuelo_usa_una_sola_conexion_para_los_dos_lectores(base):
    """prevuelo_catalogo.py lo promete ("Una conexión por pre-vuelo"): el
    catálogo de motores y el resto del catálogo se leen por la MISMA conexión.
    Expected contra fcb0c6a: 2 conexiones (from_db y leer_catalogo abren cada
    una la suya) -> `assert 2 == 1`."""
    async def cuerpo():
        await pv.prevuelo(_plan(), {})
        return len(base.abiertas)

    assert _correr(cuerpo) == 1


def test_el_segundo_prevuelo_del_proceso_no_abre_conexiones(base):
    """Expected contra fcb0c6a: 2 conexiones nuevas en el segundo pre-vuelo
    (una de from_db, una de leer_catalogo) -> `assert 2 == 0`."""
    async def cuerpo():
        await pv.prevuelo(_plan(), {})
        despues_del_primero = len(base.abiertas)
        await pv.prevuelo(_plan(), {})
        return len(base.abiertas) - despues_del_primero

    assert _correr(cuerpo) == 0


def test_los_dos_lectores_reciben_la_misma_conexion(base, monkeypatch):
    vistas = []

    async def resolver(pasos, *, conexion):
        vistas.append(conexion)
        return {}

    async def leer(*, conexion, **kw):
        vistas.append(conexion)
        return pv.Catalogo({}, {}, {}, frozenset(), {})

    monkeypatch.setattr(pv.prevuelo_catalogo, "resolver_motores", resolver)
    monkeypatch.setattr(pv.prevuelo_catalogo, "leer_catalogo", leer)

    async def cuerpo():
        await pv.prevuelo(_plan(), {})

    _correr(cuerpo)
    assert len(vistas) == 2 and vistas[0] is vistas[1] is base.abiertas[0]


def test_prevuelos_concurrentes_no_superan_el_tamano_del_pool(base, monkeypatch):
    """25 pre-vuelos a la vez con JAX_PREVUELO_DB_POOL_MAX=3: a lo sumo 3
    conexiones abiertas en todo el proceso, no 2 por pedido."""
    monkeypatch.setenv("JAX_PREVUELO_DB_POOL_MAX", "3")

    async def cuerpo():
        await asyncio.gather(*[pv.prevuelo(_plan(), {}) for _ in range(25)])

    _correr(cuerpo)
    assert 1 <= len(base.abiertas) <= 3


def test_el_pool_lleva_connect_timeout_y_la_base_de_la_configuracion(base, monkeypatch):
    monkeypatch.setenv("JAX_DB_CONNECT_TIMEOUT_SECONDS", "7")

    async def cuerpo():
        await pv.prevuelo(_plan(), {})

    _correr(cuerpo)
    assert base.kwargs, "no se abrió ninguna conexión"
    for kw in base.kwargs:
        assert kw["connect_timeout"] == 7
        assert kw["db"] == "jax_memory_test"
        assert kw["autocommit"] is True
        # El pool del pre-vuelo es de LECTURA: sin CLIENT.FOUND_ROWS (las
        # escrituras condicionales siguen por get_conn(found_rows=True)).
        assert "client_flag" not in kw


def test_cerrar_pool_cierra_las_conexiones_y_el_siguiente_prevuelo_crea_otro(base):
    async def cuerpo():
        await pv.prevuelo(_plan(), {})
        abiertas = list(base.abiertas)
        await store.cerrar_pool()
        cerradas = all(c.closed for c in abiertas)
        await pv.prevuelo(_plan(), {})
        return abiertas, cerradas

    abiertas, cerradas = _correr(cuerpo)
    assert cerradas, "cerrar_pool() dejó conexiones abiertas"
    assert len(base.abiertas) > len(abiertas), "después de cerrar, el pool no se volvió a crear"


def test_cerrar_pool_sin_pool_no_falla(base):
    _correr(lambda: asyncio.sleep(0))
    assert base.abiertas == []


def test_base_caida_al_conectar_da_503_prevuelo_no_disponible(base):
    """Fail-closed (§8): el pool no convierte una base caída en un veredicto."""
    base.falla_al_conectar = pymysql.err.OperationalError(2003, "Can't connect to MySQL server")

    async def cuerpo():
        with pytest.raises(HTTPException) as exc:
            await routes._prevuelo_o_503(_plan(), {})
        return exc.value

    error = _correr(cuerpo)
    assert error.status_code == 503
    assert error.detail["code"] == "prevuelo_no_disponible"


def test_tamano_de_pool_invalido_da_503_sin_abrir_nada(base, monkeypatch):
    """Un typo en JAX_PREVUELO_DB_POOL_MAX no cae a un default: el pre-vuelo no
    está disponible (503) y no se abre ninguna conexión."""
    monkeypatch.setenv("JAX_PREVUELO_DB_POOL_MAX", "0")

    async def cuerpo():
        with pytest.raises(HTTPException) as exc:
            await routes._prevuelo_o_503(_plan(), {})
        return exc.value

    error = _correr(cuerpo)
    assert error.status_code == 503
    assert "JAX_PREVUELO_DB_POOL_MAX" in error.detail["motivo"]
    assert base.abiertas == []


def test_pool_agotado_espera_acotada_y_da_503(base, monkeypatch):
    """Con el pool lleno de conexiones ocupadas, pedir otra no espera para
    siempre: la espera se acota con JAX_DB_CONNECT_TIMEOUT_SECONDS y termina
    en 503 prevuelo_no_disponible, no en un pedido colgado."""
    monkeypatch.setenv("JAX_PREVUELO_DB_POOL_MAX", "1")

    async def cuerpo():
        ocupada = asyncio.Event()
        liberar = asyncio.Event()

        async def acaparar():
            async with store.conexion_de_lectura():
                ocupada.set()
                await liberar.wait()

        tarea = asyncio.ensure_future(acaparar())
        await asyncio.wait({tarea, asyncio.ensure_future(ocupada.wait())},
                           return_when=asyncio.FIRST_COMPLETED)
        if tarea.done():
            tarea.result()  # acaparar() falló antes de ocupar: que se vea su error
        inicio = time.monotonic()
        try:
            with pytest.raises(HTTPException) as exc:
                await asyncio.wait_for(routes._prevuelo_o_503(_plan(), {}), 10)
        finally:
            liberar.set()
            await tarea
        return exc.value, time.monotonic() - inicio

    error, espera = _correr(cuerpo)
    assert error.status_code == 503
    assert error.detail["code"] == "prevuelo_no_disponible"
    assert espera < 5


def test_conexion_colgada_al_abrir_da_503_acotado(base):
    base.cuelga_al_conectar = True

    async def cuerpo():
        inicio = time.monotonic()
        with pytest.raises(HTTPException) as exc:
            # wait_for del TEST: si el código no acota la espera, el test
            # falla con TimeoutError en vez de colgar la corrida.
            await asyncio.wait_for(routes._prevuelo_o_503(_plan(), {}), 10)
        return exc.value, time.monotonic() - inicio

    error, espera = _correr(cuerpo)
    assert error.status_code == 503 and error.detail["code"] == "prevuelo_no_disponible"
    assert espera < 5


def test_una_conexion_rota_no_vuelve_al_pool_ni_envenena_el_siguiente(base):
    """La consulta revienta a mitad de pedido (conexión perdida): ese pedido es
    503, la conexión se CIERRA y no vuelve al pool, y el siguiente pre-vuelo
    usa una conexión nueva y sale bien."""
    async def cuerpo():
        await pv.prevuelo(_plan(), {})
        rota = base.abiertas[0]
        rota.falla_al_consultar = pymysql.err.OperationalError(2013, "Lost connection to MySQL server")
        with pytest.raises(HTTPException) as exc:
            await routes._prevuelo_o_503(_plan(), {})
        consultas_de_la_rota = rota.consultas
        veredicto = await pv.prevuelo(_plan(), {})
        return rota, exc.value, consultas_de_la_rota, veredicto

    rota, error, consultas_de_la_rota, veredicto = _correr(cuerpo)
    assert error.status_code == 503
    assert rota.closed, "la conexión rota volvió al pool abierta"
    assert rota.consultas == consultas_de_la_rota, "el pedido siguiente reusó la conexión rota"
    assert len(base.abiertas) >= 2
    assert veredicto is not None


def test_un_error_de_python_a_mitad_de_lectura_tampoco_devuelve_la_conexion(base):
    """No sólo los errores de red: un error cualquiera dentro del bloque (p. ej.
    `_modo_valido` que lanza por un `capability.mode` inválido) puede dejar
    filas sin leer en el socket. Esa conexión no se reusa."""
    async def cuerpo():
        with pytest.raises(RuntimeError):
            async with store.conexion_de_lectura() as conn:
                usada = conn
                raise RuntimeError("capability 'x': mode None fuera de ...")
        async with store.conexion_de_lectura() as conn:
            siguiente = conn
        return usada, siguiente

    usada, siguiente = _correr(cuerpo)
    assert usada.closed
    assert siguiente is not usada


def test_una_conexion_que_la_base_corto_en_reposo_no_se_reusa(base):
    """wait_timeout de MariaDB: el servidor cierra una conexión ociosa. El pool
    la descarta al pedirla (EOF en el socket) en vez de entregarla."""
    async def cuerpo():
        async with store.conexion_de_lectura() as conn:
            primera = conn
        primera._reader.eof = True
        async with store.conexion_de_lectura() as conn:
            segunda = conn
        return primera, segunda

    primera, segunda = _correr(cuerpo)
    assert segunda is not primera
    assert primera.closed


def test_el_pool_de_otro_event_loop_no_se_usa_en_silencio(base):
    """Un pool queda atado al loop que lo creó: usarlo desde otro loop VIVO
    fallaría más tarde con un error críptico. Se niega de entrada, fail-closed.
    (R38: antes el loop dueño del test ya estaba cerrado; ese caso ahora
    reemplaza el pool -- test siguiente.)"""
    async def crear():
        async with store.conexion_de_lectura():
            pass

    duenio = asyncio.new_event_loop()
    try:
        duenio.run_until_complete(crear())  # sin cerrar_pool y con el loop VIVO
        async def usar():
            async with store.conexion_de_lectura():
                pass
        with pytest.raises(RuntimeError, match="otro event loop"):
            asyncio.run(usar())
    finally:
        duenio.run_until_complete(store.cerrar_pool())
        duenio.close()
    assert store._pool_de_lectura_estado is None


def test_al_apagarse_el_loop_el_pool_se_cierra_y_el_siguiente_crea_otro(base):
    """Ruling R38: el store entero pasa por el pool y los tests/scripts corren
    un asyncio.run por llamada. Al apagarse el loop (shutdown_asyncgens) el
    pool se cierra en ESE loop -- sin RuntimeError en el loop siguiente y sin
    sockets que el recolector tenga que cerrar sobre un loop muerto.
    Expected contra 2fd3778: el segundo asyncio.run choca con el pool viejo,
    que sigue abierto -> RuntimeError 'otro event loop'."""
    async def usar():
        async with store.conexion_de_lectura() as conn:
            return conn

    try:
        primera = asyncio.run(usar())  # sin cerrar_pool
        estado_tras_el_primer_loop = store._pool_de_lectura_estado
        segunda = asyncio.run(usar())
    finally:
        store._pool_de_lectura_estado = None
    assert estado_tras_el_primer_loop is None
    assert primera.closed, "la conexión del loop apagado quedó abierta"
    assert segunda is not primera and segunda.closed
    assert len(base.abiertas) == 2


def test_un_pool_de_un_loop_cerrado_sin_apagado_se_reemplaza(base):
    """Un loop cerrado a mano (loop.close() sin shutdown_asyncgens) no corre
    el guardián: su pool no puede volver a usarse nunca. El loop siguiente
    crea otro en vez de fallar con 'otro event loop'.
    Expected contra 2fd3778: RuntimeError 'otro event loop'."""
    async def usar():
        async with store.conexion_de_lectura() as conn:
            return conn

    viejo_loop = asyncio.new_event_loop()
    try:
        primera = viejo_loop.run_until_complete(usar())
        viejo = store._pool_de_lectura_estado
    finally:
        viejo_loop.close()
    try:
        segunda = asyncio.run(usar())
    finally:
        store._pool_de_lectura_estado = None
    assert viejo is not None
    assert segunda is not primera
    assert len(base.abiertas) == 2


def test_from_db_sin_conexion_inyectada_sigue_abriendo_y_cerrando_la_suya(base):
    """El worker de LAS MANOS y el ejecutor llaman `MotorCatalog.from_db()` sin
    argumentos: para ellos nada cambia (conexión propia, cerrada al final)."""
    from motor_registry.catalog import MotorCatalog

    asyncio.run(MotorCatalog.from_db())
    assert len(base.abiertas) == 1 and base.abiertas[0].closed


def test_from_db_con_conexion_inyectada_no_la_cierra(base):
    from motor_registry.catalog import MotorCatalog

    prestada = _Conexion(99)
    asyncio.run(MotorCatalog.from_db(conexion=prestada))
    assert base.abiertas == []
    assert not prestada.closed
    assert prestada.consultas == 3


def test_el_shutdown_de_las_manos_cierra_el_pool():
    """Expected contra fcb0c6a: `app.router.on_shutdown` vacío -> AssertionError."""
    las_manos = str(RAIZ / "las_manos")
    if las_manos not in sys.path:
        sys.path.insert(0, las_manos)
    import server

    cerrar = AsyncMock()
    with patch.object(server.jacobs_store, "cerrar_pool", cerrar):
        assert server.app.router.on_shutdown, "LAS MANOS no registra ningún handler de shutdown"
        for handler in server.app.router.on_shutdown:
            asyncio.run(handler())
    cerrar.assert_awaited_once()


def _cli():
    spec = importlib.util.spec_from_file_location("jacobs_relaunch_pool_test", RAIZ / "tools" / "jacobs_relaunch.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


@pytest.mark.parametrize("salida", [0, 1, 3, 4])
def test_el_cli_cierra_el_pool_antes_de_salir_con_cada_codigo(salida):
    from jacobs import continuar
    from jacobs.models import Pipeline, PipelineStatus

    cli = _cli()
    pipeline = Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma", mode="autonomous", run_epoch=2)
    respuesta = {"run_epoch": 2, "pasos_reusados": [], "pasos_a_correr": [0], "costo_max_usd": "0.100000"}
    estado = PipelineStatus.completed if salida == 0 else PipelineStatus.failed
    final = pipeline.model_copy(update={"status": estado})
    if salida == 1:
        servicio = AsyncMock(side_effect=continuar.ContinuarRechazado(409, "estado_no_continuable", {"mensaje": "m"}))
    elif salida == 4:
        servicio = AsyncMock(side_effect=OSError("base caída"))
    else:
        servicio = AsyncMock(return_value=(respuesta, pipeline))
    cerrar = AsyncMock()
    with patch("jacobs.continuar.continuar", servicio), \
         patch("jacobs.executor.run_pipeline", AsyncMock()), \
         patch("jacobs.store.pipeline_get", AsyncMock(return_value=final)), \
         patch("jacobs.store.steps_by_pipeline", AsyncMock(return_value=[])), \
         patch("jacobs.store.cerrar_pool", cerrar):
        rc = asyncio.run(cli.relanzar("p1", {}, Decimal("1")))
    assert rc == salida
    cerrar.assert_awaited_once()


def test_el_cli_conserva_su_codigo_si_cerrar_el_pool_falla(capsys):
    """Cerrar el pool al salir no puede tapar el resultado de la corrida: el
    proceso termina igual y sus sockets con él. Se avisa, no se cambia el código."""
    from jacobs import continuar

    cli = _cli()
    rechazo = continuar.ContinuarRechazado(409, "estado_no_continuable", {"mensaje": "m"})
    with patch("jacobs.continuar.continuar", AsyncMock(side_effect=rechazo)), \
         patch("jacobs.store.cerrar_pool", AsyncMock(side_effect=OSError("socket ya cerrado"))):
        rc = asyncio.run(cli.relanzar("p1", {}, None))
    assert rc == 1
    assert "pool" in capsys.readouterr().out


# --- Fix round 1 de la revisión (2026-09-17) ---------------------------------

def test_una_conexion_que_se_rompe_despierta_al_que_espera_turno(base, monkeypatch):
    """Hallazgo 1: aiomysql 0.3.2 `Pool.release` NO despierta a quien espera
    turno si la conexión devuelta ya está cerrada. Con POOL_MAX=1, A tiene la
    única conexión y revienta a los 0,2 s; B, que esperaba, tiene que recibir
    una conexión enseguida, no a los 3 s del timeout con un 503.
    Expected contra 607f84c: B tarda ~3 s -> TimeoutError."""
    monkeypatch.setenv("JAX_PREVUELO_DB_POOL_MAX", "1")
    monkeypatch.setenv("JAX_DB_CONNECT_TIMEOUT_SECONDS", "3")

    async def cuerpo():
        tomada = asyncio.Event()

        async def a():
            with pytest.raises(RuntimeError):
                async with store.conexion_de_lectura():
                    tomada.set()
                    await asyncio.sleep(0.2)
                    raise RuntimeError("se rompió a mitad de lectura")

        async def b():
            await tomada.wait()
            inicio = time.monotonic()
            async with store.conexion_de_lectura() as conn:
                return time.monotonic() - inicio, conn.closed

        _, (espera, cerrada) = await asyncio.gather(a(), b())
        return espera, cerrada

    espera, cerrada = _correr(cuerpo)
    assert espera < 1.0, f"B esperó {espera:.2f} s: nadie lo despertó cuando se liberó el lugar"
    assert not cerrada, "B recibió la conexión rota"


def test_dos_primeros_pedidos_a_la_vez_crean_un_solo_pool(base, monkeypatch):
    """Hallazgo 3: el candado de creación. Hoy minsize=0 hace create_pool casi
    instantáneo, pero con una pausa forzada adentro (lo que sería minsize>0,
    que conecta al crear) dos primeros pedidos concurrentes no deben crear dos
    pools. Rojo por mutación (sin el candado): 2 pools."""
    creados = []
    real = aiomysql.create_pool

    async def lento(*args, **kwargs):
        creados.append(kwargs)
        await asyncio.sleep(0.05)
        return await real(*args, **kwargs)

    monkeypatch.setattr(aiomysql, "create_pool", lento)

    async def cuerpo():
        async def pedir():
            async with store.conexion_de_lectura():
                await asyncio.sleep(0)
        await asyncio.gather(*[pedir() for _ in range(5)])

    _correr(cuerpo)
    assert len(creados) == 1


# --- Fix round 2 de la revisión (2026-09-17) ---------------------------------

@pytest.mark.parametrize("falla", ["revienta", "cuelga"])
def test_un_connect_fallido_no_se_queda_con_el_turno(base, monkeypatch, falla):
    """Con POOL_MAX=1, un connect que revienta (o que se cuelga hasta el
    timeout) durante una caída no puede quedarse con el único turno: cuando la
    base vuelve, el pedido siguiente tiene que conseguir conexión. Si el turno
    se perdiera, después de POOL_MAX connects fallidos todo pre-vuelo sería
    503 para siempre. Rojo por mutación: sin `turno.release()` en el except
    alrededor de `pool.acquire()` -> TimeoutError en el segundo pedido."""
    monkeypatch.setenv("JAX_PREVUELO_DB_POOL_MAX", "1")
    monkeypatch.setenv("JAX_DB_CONNECT_TIMEOUT_SECONDS", "1")

    async def cuerpo():
        if falla == "revienta":
            base.falla_al_conectar = pymysql.err.OperationalError(2003, "Can't connect to MySQL server")
            esperado = pymysql.err.OperationalError
        else:
            base.cuelga_al_conectar = True
            esperado = TimeoutError
        with pytest.raises(esperado):
            async with store.conexion_de_lectura():
                pass
        base.falla_al_conectar = None
        base.cuelga_al_conectar = False
        async with store.conexion_de_lectura() as conn:
            return conn.closed

    assert _correr(cuerpo) is False


def test_si_devolver_la_conexion_falla_el_turno_vuelve_igual(base, monkeypatch):
    """Si `pool.release()` lanza, el turno se suelta igual (finally): el pedido
    siguiente no se queda sin turno. Rojo por mutación: soltar el turno sólo si
    release no lanzó -> TimeoutError en el segundo pedido."""
    monkeypatch.setenv("JAX_PREVUELO_DB_POOL_MAX", "1")
    monkeypatch.setenv("JAX_DB_CONNECT_TIMEOUT_SECONDS", "1")
    release_real = aiomysql.pool.Pool.release
    llamadas = []

    def release_que_falla_una_vez(self, conn):
        fut = release_real(self, conn)  # la conexión vuelve al pool de verdad
        llamadas.append(conn)
        if len(llamadas) == 1:
            raise RuntimeError("release falló")
        return fut

    monkeypatch.setattr(aiomysql.pool.Pool, "release", release_que_falla_una_vez)

    async def cuerpo():
        with pytest.raises(RuntimeError, match="release falló"):
            async with store.conexion_de_lectura():
                pass
        async with store.conexion_de_lectura() as conn:
            return conn.closed

    assert _correr(cuerpo) is False
