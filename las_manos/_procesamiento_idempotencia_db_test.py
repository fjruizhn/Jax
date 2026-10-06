"""Idempotencia de `POST /procesamiento/trabajos` contra MariaDB real (job jacobs-gobernanza-db).

Origen (auditoria de E3, 2026-10-06): el despachador de documentos de jax-platform guarda en
MEMORIA el estado incierto de un envio. Si el proceso se corta entre el 202 de LAS MANOS y
`marcar_despachadas`, la fila vuelve a la cola y el OCR se duplica. La solucion es una clave de
idempotencia estable en el encabezado `Idempotency-Key`: LAS MANOS la guarda de forma durable
(tabla `procesamiento_idempotencia`, UNIQUE (identidad_servicio, clave)) y un reenvio devuelve el
MISMO `job_id` sin crear un trabajo nuevo.

Lo que prueba este archivo (con la base efimera de la sesion, nunca produccion):
  - el reclamo de una clave: el segundo da el mismo job_id; en concurrencia gana UNO solo;
  - la clave es por identidad de servicio; la misma clave con otro pedido es un conflicto (409);
  - sin clave el comportamiento es el de siempre (dos trabajos);
  - un reclamo huerfano (el proceso murio entre reclamar y crear el trabajo) se retoma pasada la
    gracia, y antes de ella el reintento espera (503) en vez de duplicar;
  - si crear el trabajo falla, el reclamo se libera;
  - la purga borra solo lo vencido; EXPLAIN de las consultas nuevas usa sus indices.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI

# Base de tests de ESTA sesion (decision de Fernando, 2026-09-17): nunca produccion.
from base_de_test import exigir_base_de_test  # noqa: E402

exigir_base_de_test()

import procesamiento_idempotencia as idem  # noqa: E402
import procesamiento_routes as rutas_mod  # noqa: E402
from auth_servicio import ENCABEZADO, IDENTIDAD_JACOBS, IDENTIDAD_PLATAFORMA, proteger  # noqa: E402
from jacobs import store  # noqa: E402
from procesamiento import dependencias  # noqa: E402
from processing_job_store import ProcessingJobStore  # noqa: E402
from processing_ownership import ProcessingOwnershipContext  # noqa: E402

UUID_PROYECTO = "0192f1d2-7c3a-7b4e-9a10-3f5e2d1c0b9a"
OWNER = ProcessingOwnershipContext("processing-owner.1", "1", "2", "3")
CREDENCIAL_PLATAFORMA = "p" * 43
CREDENCIAL_JACOBS = "j" * 43


def _clave() -> str:
    return f"jxp-doc-{uuid.uuid4().hex}"


def _hash(*rutas: str) -> str:
    return idem.hash_de_solicitud(OWNER, UUID_PROYECTO, list(rutas))


async def _borrar(clave: str) -> None:
    async with store.conexion() as conn:
        async with conn.cursor() as cur:
            await cur.execute(f"DELETE FROM {idem.NOMBRE_TABLA} WHERE clave = %s", (clave,))


async def _filas(clave: str) -> list[tuple]:
    async with store.conexion() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT identidad_servicio, job_id FROM {idem.NOMBRE_TABLA} WHERE clave = %s ORDER BY id", (clave,))
            return [tuple(f.values()) if isinstance(f, dict) else tuple(f) for f in await cur.fetchall()]


async def _envejecer(clave: str, segundos: int) -> None:
    async with store.conexion() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"UPDATE {idem.NOMBRE_TABLA} SET creado_en = creado_en - INTERVAL %s SECOND WHERE clave = %s",
                (segundos, clave))


# ---------------------------------------------------------------- la tabla y el reclamo

def test_init_tabla_es_idempotente():
    async def todo():
        await idem.init_tabla()
        await idem.init_tabla()          # repetirla no falla ni cambia nada
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT INDEX_NAME, NON_UNIQUE, COLUMN_NAME FROM information_schema.STATISTICS "
                    "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s ORDER BY INDEX_NAME, SEQ_IN_INDEX",
                    (idem.NOMBRE_TABLA,))
                return [tuple(f.values()) if isinstance(f, dict) else tuple(f) for f in await cur.fetchall()]
    indices = asyncio.run(todo())
    assert ("uq_procesamiento_idem_clave", 0, "identidad_servicio") in indices
    assert ("uq_procesamiento_idem_clave", 0, "clave") in indices


def test_el_segundo_reclamo_de_la_misma_clave_da_el_mismo_trabajo():
    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            a = await idem.reclamar(IDENTIDAD_PLATAFORMA, clave, _hash("a.pdf"), "job-a")
            b = await idem.reclamar(IDENTIDAD_PLATAFORMA, clave, _hash("a.pdf"), "job-b")
            return a, b, await _filas(clave)
        finally:
            await _borrar(clave)
    a, b, filas = asyncio.run(todo())
    assert a.gano and a.job_id == "job-a"
    assert not b.gano and b.job_id == "job-a"
    assert filas == [(IDENTIDAD_PLATAFORMA, "job-a")]


def test_reclamos_simultaneos_crean_un_solo_trabajo():
    """Dos (y mas) peticiones a la vez con la misma clave: sale UNA ganadora del indice unico y
    todas las demas leen el job_id de la ganadora."""
    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            reclamos = await asyncio.gather(*[
                idem.reclamar(IDENTIDAD_PLATAFORMA, clave, _hash("a.pdf"), f"job-{i}") for i in range(12)])
            return reclamos, await _filas(clave)
        finally:
            await _borrar(clave)
    reclamos, filas = asyncio.run(todo())
    ganadoras = [r for r in reclamos if r.gano]
    assert len(ganadoras) == 1
    assert {r.job_id for r in reclamos} == {ganadoras[0].job_id}
    assert len(filas) == 1


def test_la_clave_es_por_identidad_de_servicio():
    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            a = await idem.reclamar(IDENTIDAD_PLATAFORMA, clave, _hash("a.pdf"), "job-p")
            b = await idem.reclamar(IDENTIDAD_JACOBS, clave, _hash("a.pdf"), "job-j")
            return a, b, await _filas(clave)
        finally:
            await _borrar(clave)
    a, b, filas = asyncio.run(todo())
    assert a.gano and b.gano and a.job_id != b.job_id
    assert sorted(filas) == [(IDENTIDAD_JACOBS, "job-j"), (IDENTIDAD_PLATAFORMA, "job-p")]


def test_liberar_solo_borra_el_reclamo_propio():
    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            await idem.reclamar(IDENTIDAD_PLATAFORMA, clave, _hash("a.pdf"), "job-a")
            await idem.liberar(IDENTIDAD_PLATAFORMA, clave, "job-de-otro")     # no es suyo
            antes = await _filas(clave)
            await idem.liberar(IDENTIDAD_PLATAFORMA, clave, "job-a")
            return antes, await _filas(clave)
        finally:
            await _borrar(clave)
    antes, despues = asyncio.run(todo())
    assert antes == [(IDENTIDAD_PLATAFORMA, "job-a")]
    assert despues == []


def test_purga_borra_solo_lo_vencido():
    async def todo():
        await idem.init_tabla()
        vieja, nueva = _clave(), _clave()
        try:
            await idem.reclamar(IDENTIDAD_PLATAFORMA, vieja, _hash("a.pdf"), "job-v")
            await idem.reclamar(IDENTIDAD_PLATAFORMA, nueva, _hash("b.pdf"), "job-n")
            await _envejecer(vieja, idem.ttl_segundos() + 60)
            await idem.purgar_vencidas()
            return await _filas(vieja), await _filas(nueva)
        finally:
            await _borrar(vieja)
            await _borrar(nueva)
    vieja, nueva = asyncio.run(todo())
    assert vieja == []
    assert nueva == [(IDENTIDAD_PLATAFORMA, "job-n")]


def test_tomar_huerfana_es_un_cas_sobre_el_job_id():
    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            r = await idem.reclamar(IDENTIDAD_PLATAFORMA, clave, _hash("a.pdf"), "job-muerto")
            primera = await idem.tomar_huerfana(r, "job-nuevo-1")
            segunda = await idem.tomar_huerfana(r, "job-nuevo-2")      # r ya esta desactualizado
            return primera, segunda, await _filas(clave)
        finally:
            await _borrar(clave)
    primera, segunda, filas = asyncio.run(todo())
    assert primera is True and segunda is False
    assert filas == [(IDENTIDAD_PLATAFORMA, "job-nuevo-1")]


def test_explain_de_las_consultas_nuevas_usa_sus_indices():
    """EXPLAIN de la consulta REAL: la busqueda por (identidad, clave) va por el UNIQUE y la
    purga por el indice de `creado_en`; ninguna recorre la tabla."""
    async def explicar(sql, params):
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("EXPLAIN " + sql, params)
                return [dict(zip([d[0] for d in cur.description], f)) if not isinstance(f, dict) else f
                        for f in await cur.fetchall()]

    async def todo():
        await idem.init_tabla()
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                # Con la tabla vacia el optimizador elige `const`/`ALL` sin informar nada: se llena
                # con filas de relleno del mismo tamano para que el plan sea el de produccion.
                await cur.execute(
                    f"INSERT INTO {idem.NOMBRE_TABLA} (identidad_servicio, clave, solicitud_hash, job_id, creado_en) "
                    f"SELECT 'relleno', CONCAT('explain-', seq), REPEAT('0', 64), UUID(), "
                    f"NOW(6) - INTERVAL seq SECOND FROM seq_1_to_2000")
                await cur.execute(f"ANALYZE TABLE {idem.NOMBRE_TABLA}")
                await cur.fetchall()
        try:
            buscar = await explicar(idem.SQL_BUSCAR, ("relleno", "explain-5"))
            purgar = await explicar(idem.SQL_PURGAR.replace("DELETE", "SELECT id", 1)
                                    .replace(" LIMIT %s", ""), (3600,))
            return buscar, purgar
        finally:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(f"DELETE FROM {idem.NOMBRE_TABLA} WHERE identidad_servicio = 'relleno'")
    buscar, purgar = asyncio.run(todo())
    assert buscar[0]["key"] == "uq_procesamiento_idem_clave", buscar
    assert "filesort" not in str(buscar[0].get("Extra")).lower()
    assert purgar[0]["key"] == "idx_procesamiento_idem_creado", purgar


# ---------------------------------------------------------------- la ruta, de punta a punta

@pytest.fixture
def ruta(monkeypatch):
    """La ruta real de LAS MANOS con la base real; sin OCR (el worker es un AsyncMock) y con un
    almacen de trabajos propio. Entrega (app, store, ejecutar)."""
    tmp = tempfile.TemporaryDirectory()
    almacen = ProcessingJobStore(str(Path(tmp.name) / "jobs.jsonl"))
    ejecutar = AsyncMock()
    monkeypatch.setattr(rutas_mod, "_STORE", almacen)
    monkeypatch.setattr(rutas_mod, "_ejecutar_trabajo", ejecutar)
    monkeypatch.setattr(rutas_mod, "_SEMAFORO_TRABAJOS", asyncio.Semaphore(4))
    monkeypatch.setattr(rutas_mod, "_CONTROLES", {})
    monkeypatch.setattr(rutas_mod.proyecto_activo, "identidad_activa_del_proyecto", AsyncMock(return_value=True))
    monkeypatch.setattr(dependencias, "estado", lambda *a, **k: {"ok": True, "faltan": [], "motivos": {}})
    app = FastAPI()
    app.include_router(rutas_mod.router)
    proteger(app, {IDENTIDAD_PLATAFORMA: CREDENCIAL_PLATAFORMA.encode(), IDENTIDAD_JACOBS: CREDENCIAL_JACOBS.encode()})
    yield app, almacen, ejecutar
    tmp.cleanup()


def _cabeceras(clave: str | None, *, credencial: str = CREDENCIAL_PLATAFORMA) -> dict[str, str]:
    h = {ENCABEZADO: credencial,
         "X-Jax-Processing-Owner-Version": OWNER.version, "X-Jax-Processing-Tenant-Id": OWNER.tenant_id,
         "X-Jax-Processing-User-Id": OWNER.user_id, "X-Jax-Processing-Project-Id": OWNER.project_id}
    if clave is not None:
        h["Idempotency-Key"] = clave
    return h


def _cuerpo(*rutas: str) -> dict:
    return {"project_uuid": UUID_PROYECTO, "rutas": list(rutas)}


async def _post(app, clave, *rutas, **kw):
    transporte = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transporte, base_url="http://las-manos") as c:
        return await c.post("/procesamiento/trabajos", json=_cuerpo(*rutas), headers=_cabeceras(clave, **kw))


def test_reenvio_con_la_misma_clave_devuelve_el_mismo_job_sin_crear_otro(ruta):
    app, almacen, ejecutar = ruta

    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            return await _post(app, clave, "a.pdf"), await _post(app, clave, "a.pdf")
        finally:
            await _borrar(clave)
    r1, r2 = asyncio.run(todo())
    assert (r1.status_code, r2.status_code) == (202, 202), (r1.text, r2.text)
    assert r1.json()["job_id"] == r2.json()["job_id"]
    assert r2.headers.get("Idempotent-Replayed") == "true" and "Idempotent-Replayed" not in r1.headers
    assert r2.json()["estado"] in {"pending", "running", "completed"}
    assert len(almacen._index) == 1, "el reenvio no debe crear un trabajo nuevo"
    assert ejecutar.await_count == 1, "el OCR se programa UNA vez"


def test_dos_pedidos_simultaneos_con_la_misma_clave_crean_un_solo_trabajo(ruta):
    app, almacen, ejecutar = ruta

    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            return await asyncio.gather(*[_post(app, clave, "a.pdf") for _ in range(6)])
        finally:
            await _borrar(clave)
    respuestas = asyncio.run(todo())
    assert [r.status_code for r in respuestas] == [202] * 6, [r.text for r in respuestas]
    assert len({r.json()["job_id"] for r in respuestas}) == 1
    assert len(almacen._index) == 1
    assert ejecutar.await_count == 1


def test_sin_clave_el_comportamiento_es_el_de_siempre(ruta):
    app, almacen, ejecutar = ruta

    async def todo():
        await idem.init_tabla()
        return await _post(app, None, "a.pdf"), await _post(app, None, "a.pdf")
    r1, r2 = asyncio.run(todo())
    assert (r1.status_code, r2.status_code) == (202, 202)
    assert r1.json() == {"job_id": r1.json()["job_id"]}, "sin clave la respuesta no cambia de forma"
    assert r1.json()["job_id"] != r2.json()["job_id"]
    assert len(almacen._index) == 2


def test_la_misma_clave_con_otro_pedido_es_un_conflicto(ruta):
    app, almacen, ejecutar = ruta

    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            return await _post(app, clave, "a.pdf"), await _post(app, clave, "a.pdf", "b.pdf")
        finally:
            await _borrar(clave)
    r1, r2 = asyncio.run(todo())
    assert r1.status_code == 202
    assert r2.status_code == 409 and r2.json()["detail"]["code"] == "idempotency_key_reuse", r2.text
    assert len(almacen._index) == 1


def test_una_clave_con_formato_invalido_da_422(ruta):
    app, almacen, ejecutar = ruta

    async def todo():
        await idem.init_tabla()
        return [await _post(app, malo, "a.pdf") for malo in ("corta", "con espacio " * 3, "x" * 129, "a/b/c/d/e/f/g/h/i/j/k")]
    for r in asyncio.run(todo()):
        assert r.status_code == 422 and r.json()["detail"]["code"] == "idempotency_key_invalida", r.text
    assert len(almacen._index) == 0


def test_dos_encabezados_de_clave_son_ambiguos_y_dan_422(ruta):
    app, almacen, ejecutar = ruta

    async def todo():
        await idem.init_tabla()
        transporte = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        cabeceras = list(_cabeceras(_clave()).items()) + [("Idempotency-Key", _clave())]
        async with httpx.AsyncClient(transport=transporte, base_url="http://las-manos") as c:
            return await c.post("/procesamiento/trabajos", json=_cuerpo("a.pdf"), headers=cabeceras)
    r = asyncio.run(todo())
    assert r.status_code == 422 and r.json()["detail"]["code"] == "idempotency_key_invalida", r.text
    assert len(almacen._index) == 0


def test_un_reclamo_huerfano_reciente_hace_esperar_y_no_duplica(ruta, monkeypatch):
    """El proceso murio entre reclamar y crear el trabajo, pero hace segundos: el trabajo puede estar
    creandose ahora mismo. El reintento recibe 503 (reintentable), nunca un trabajo duplicado."""
    app, almacen, ejecutar = ruta
    monkeypatch.setenv("JAX_PROCESAMIENTO_IDEMPOTENCIA_ESPERA_MS", "100")

    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            await idem.reclamar(IDENTIDAD_PLATAFORMA, clave, _hash("a.pdf"), str(uuid.uuid4()))
            return await _post(app, clave, "a.pdf")
        finally:
            await _borrar(clave)
    r = asyncio.run(todo())
    assert r.status_code == 503 and r.json()["detail"]["code"] == "idempotencia_en_curso", r.text
    assert len(almacen._index) == 0 and ejecutar.await_count == 0


def test_un_reclamo_huerfano_viejo_se_retoma_y_crea_el_trabajo_una_vez(ruta):
    app, almacen, ejecutar = ruta

    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            muerto = str(uuid.uuid4())
            await idem.reclamar(IDENTIDAD_PLATAFORMA, clave, _hash("a.pdf"), muerto)
            await _envejecer(clave, idem.gracia_segundos() + 30)
            respuestas = await asyncio.gather(*[_post(app, clave, "a.pdf") for _ in range(4)])
            return muerto, respuestas, await _filas(clave)
        finally:
            await _borrar(clave)
    muerto, respuestas, filas = asyncio.run(todo())
    ok = [r for r in respuestas if r.status_code == 202]
    assert ok, [r.text for r in respuestas]
    assert {r.json()["job_id"] for r in ok} == {filas[0][1]} and filas[0][1] != muerto
    assert all(r.status_code in (202, 503) for r in respuestas)     # la perdedora de la carrera espera, no duplica
    assert len(almacen._index) == 1 and ejecutar.await_count == 1


def test_si_crear_el_trabajo_falla_el_reclamo_se_libera(ruta):
    app, almacen, ejecutar = ruta

    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            with patch.object(almacen, "create", side_effect=OSError("disco lleno")):
                r = await _post(app, clave, "a.pdf")
            cupo = rutas_mod._SEMAFORO_TRABAJOS._value
            return r, await _filas(clave), cupo, await _post(app, clave, "a.pdf")
        finally:
            await _borrar(clave)
    fallo, filas, cupo, reintento = asyncio.run(todo())
    assert fallo.status_code >= 500
    assert filas == [], "un reclamo sin trabajo no puede quedar apuntando a la nada"
    assert reintento.status_code == 202 and len(almacen._index) == 1
    assert cupo == 4, "el cupo se devolvio"


def test_si_la_base_de_claves_no_responde_falla_cerrado_sin_trabajo(ruta):
    """Con clave y sin poder garantizar la idempotencia, no se crea nada: 503 reintentable y el
    cupo vuelve. Es el camino que el despachador ya trata como 'reintentar'."""
    app, almacen, ejecutar = ruta

    async def todo():
        await idem.init_tabla()
        with patch.object(idem, "reclamar", AsyncMock(side_effect=ConnectionError("base caida"))):
            return await _post(app, _clave(), "a.pdf")
    r = asyncio.run(todo())
    assert r.status_code == 503 and r.json()["detail"]["code"] == "idempotencia_no_disponible", r.text
    assert len(almacen._index) == 0 and ejecutar.await_count == 0
    assert rutas_mod._SEMAFORO_TRABAJOS._value == 4


def test_el_reenvio_no_pide_cupo_aunque_este_lleno(ruta, monkeypatch):
    """Un reenvio solo lee: con el cupo agotado (429 para un pedido nuevo) devuelve igual el trabajo."""
    app, almacen, ejecutar = ruta

    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            primero = await _post(app, clave, "a.pdf")
            lleno = asyncio.Semaphore(1)
            await lleno.acquire()
            monkeypatch.setattr(rutas_mod, "_SEMAFORO_TRABAJOS", lleno)
            return primero, await _post(app, clave, "a.pdf"), await _post(app, _clave(), "a.pdf")
        finally:
            await _borrar(clave)
    primero, reenvio, nuevo = asyncio.run(todo())
    assert reenvio.status_code == 202 and reenvio.json()["job_id"] == primero.json()["job_id"]
    assert nuevo.status_code == 429
