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
import logging
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


def test_el_hash_corto_del_log_es_el_mismo_que_calcula_jax_platform():
    """Vector fijo, copiado a backend/tests/test_proyectos_documentos_despachador.py de jax-platform
    (`abreviar_clave`): con el mismo hash corto las lineas de log de las dos puntas se unen."""
    assert idem.abreviar("jxp-doc-a2dad605d3914f47985f83924083ec04") == "776783dd9d97"


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
    purga por el indice de `creado_en`; ninguna recorre la tabla. El relleno lleva una MARCA unica por
    corrida (como pide tests/test_delete_de_tablas_compartidas.py): dos sesiones sobre la misma base no se
    borran el relleno entre si."""
    marca = uuid.uuid4().hex                      # 32 caracteres: cabe en identidad_servicio

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
                # con filas de relleno para que el plan sea el de produccion.
                await cur.execute(
                    f"INSERT INTO {idem.NOMBRE_TABLA} (identidad_servicio, clave, solicitud_hash, job_id, creado_en) "
                    f"SELECT %s, CONCAT('explain-', seq), REPEAT('0', 64), UUID(), "
                    f"NOW(6) - INTERVAL seq SECOND FROM seq_1_to_2000", (marca,))
                await cur.execute(f"ANALYZE TABLE {idem.NOMBRE_TABLA}")
                await cur.fetchall()
        try:
            buscar = await explicar(idem.SQL_BUSCAR, (marca, "explain-5"))
            purgar = await explicar(idem.SQL_PURGAR.replace("DELETE", "SELECT id", 1)
                                    .replace(" LIMIT %s", ""), (3600,))
            return buscar, purgar
        finally:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(f"DELETE FROM {idem.NOMBRE_TABLA} WHERE identidad_servicio = %s", (marca,))
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


def test_un_trabajo_fallido_que_el_llamador_nunca_vio_se_reintenta_con_la_misma_clave(ruta):
    """LAS MANOS se reinicia y marca `failed` el trabajo que corria; el llamador (que no llego a conocer el
    job_id) reenvia con la misma clave. Devolverle ese trabajo fallido dejaria su documento en `error` por algo
    que nunca vio: se crea uno nuevo, una sola vez aunque lleguen varios reenvios a la vez."""
    app, almacen, ejecutar = ruta

    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            viejo = almacen.create(ownership=OWNER, caller=f"user:{OWNER.user_id}", capability="ingesta_archivos",
                                   motor="n/a", trace_id="t", prompt="p", recursion_depth=0)
            almacen.update(viejo, status="failed", error="proceso de LAS MANOS reiniciado")
            await idem.reclamar(IDENTIDAD_PLATAFORMA, clave, _hash("a.pdf"), viejo)
            respuestas = await asyncio.gather(*[_post(app, clave, "a.pdf") for _ in range(3)])
            return viejo, respuestas, await _filas(clave)
        finally:
            await _borrar(clave)
    viejo, respuestas, filas = asyncio.run(todo())
    assert all(r.status_code == 202 for r in respuestas), [r.text for r in respuestas]
    nuevos = {r.json()["job_id"] for r in respuestas}
    assert len(nuevos) == 1 and viejo not in nuevos and filas == [(IDENTIDAD_PLATAFORMA, nuevos.pop())]
    assert len(almacen._index) == 2 and ejecutar.await_count == 1


def test_la_ganadora_lenta_que_pierde_el_reclamo_cancela_su_trabajo_y_queda_uno(ruta, monkeypatch):
    """Gracia de 1 s y una demora de 1,6 s entre el INSERT y crear el trabajo: otro reintento vence la gracia y
    retoma el reclamo. La ganadora original, al crear, relee, ve que ya no es suya, CANCELA su trabajo antes de
    programar el OCR y devuelve el del reintento. Sin esto salian DOS trabajos vivos (OCR duplicado)."""
    app, almacen, ejecutar = ruta
    monkeypatch.setenv("JAX_PROCESAMIENTO_IDEMPOTENCIA_GRACIA_SEGUNDOS", "1")
    real = idem.reclamar

    async def lento(*a, **k):
        r = await real(*a, **k)
        if r.gano and not lento.ya:
            lento.ya = True
            await asyncio.sleep(1.6)
        return r
    lento.ya = False
    monkeypatch.setattr(idem, "reclamar", lento)

    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            async def segundo():
                await asyncio.sleep(1.2)
                return await _post(app, clave, "a.pdf")
            r1, r2 = await asyncio.gather(_post(app, clave, "a.pdf"), segundo())
            return r1, r2, await _filas(clave)
        finally:
            await _borrar(clave)
    r1, r2, filas = asyncio.run(todo())
    assert (r1.status_code, r2.status_code) == (202, 202), (r1.text, r2.text)
    assert r1.json()["job_id"] == r2.json()["job_id"] == filas[0][1]
    vivos = [j for j in almacen._index.values() if j["status"] != "cancelled"]
    assert len(vivos) == 1 and len(almacen._index) == 2
    assert ejecutar.await_count == 1, "el OCR se programa UNA vez"


def test_el_reenvio_la_toma_de_un_huerfano_y_el_reintento_de_un_fallido_quedan_en_el_log(ruta, caplog):
    app, almacen, ejecutar = ruta
    caplog.set_level(logging.INFO, logger="procesamiento_routes")

    async def todo():
        await idem.init_tabla()
        c_reenvio, c_huerfano, c_fallido = _clave(), _clave(), _clave()
        try:
            await _post(app, c_reenvio, "a.pdf")
            await _post(app, c_reenvio, "a.pdf")
            muerto = str(uuid.uuid4())
            await idem.reclamar(IDENTIDAD_PLATAFORMA, c_huerfano, _hash("a.pdf"), muerto)
            await _envejecer(c_huerfano, idem.gracia_segundos() + 30)
            await _post(app, c_huerfano, "a.pdf")
            viejo = almacen.create(ownership=OWNER, caller=f"user:{OWNER.user_id}", capability="ingesta_archivos",
                                   motor="n/a", trace_id="t", prompt="p", recursion_depth=0)
            almacen.update(viejo, status="failed", error="x")
            await idem.reclamar(IDENTIDAD_PLATAFORMA, c_fallido, _hash("a.pdf"), viejo)
            r = await _post(app, c_fallido, "a.pdf")
            return (c_reenvio, c_huerfano, c_fallido, muerto, viejo, r.json()["job_id"],
                    (await _filas(c_reenvio))[0][1])
        finally:
            for c in (c_reenvio, c_huerfano, c_fallido):
                await _borrar(c)
    c_reenvio, c_huerfano, c_fallido, muerto, viejo, nuevo, job_reenvio = asyncio.run(todo())
    texto = [r.getMessage() for r in caplog.records]

    def con(*partes):
        return [t for t in texto if all(p in t for p in partes)]
    assert con("reenvio", idem.abreviar(c_reenvio), job_reenvio)
    assert con("huerfano", idem.abreviar(c_huerfano), muerto)
    assert con("reintenta", idem.abreviar(c_fallido), viejo, nuevo)
    assert not [t for t in texto if c_reenvio in t or c_huerfano in t or c_fallido in t], "la clave entera no va al log"


# Contrato de idempotencia COMPARTIDO con jax-platform: la MISMA tabla vive copiada en
# backend/tests/test_proyectos_documentos_despachador.py (CONTRATO_IDEMPOTENCIA) y la falsa de LAS MANOS de
# la plataforma la cumple. Si se cambia una, se cambia la otra: cada fila es (estado del trabajo que ya tiene
# esa clave o None, el pedido es el mismo, resultado). `rejected` existe en `JobStatus` y en `_ESTADOS_QUE_SE_REINTENTAN`; el almacen de procesamiento no lo deja
# ESCRIBIR (vocabulario cerrado), asi que la prueba lo inyecta en el indice.
CONTRATO_IDEMPOTENCIA = (
    (None, True, "crea"),
    ("pending", True, "mismo"),
    ("running", True, "mismo"),
    ("cancelling", True, "mismo"),
    ("completed", True, "mismo"),
    ("failed", True, "crea"),
    ("cancelled", True, "crea"),
    ("rejected", True, "crea"),
    ("pending", False, "conflicto"),
    ("failed", False, "conflicto"),
    ("completed", False, "conflicto"),
    # Desenlaces que NO son un trabajo: 503 con un codigo estable que el despachador trata distinto.
    # `confirmado_ausente`: la clave esta confirmada para un trabajo que el almacen sano no conoce -> DESCONOCIDO
    # (503 idempotencia_estado_desconocido): el despachador salta ESE proyecto, nunca corta el ciclo ni retoma.
    # `almacen_sin_integridad`: el almacen perdio la integridad -> 503 procesamiento_no_disponible (global).
    ("confirmado_ausente", True, "desconocido"),
    ("almacen_sin_integridad", True, "no_disponible"),
)


@pytest.mark.parametrize("previo,mismo_pedido,esperado", CONTRATO_IDEMPOTENCIA)
def test_contrato_compartido_de_idempotencia(ruta, previo, mismo_pedido, esperado):
    app, almacen, ejecutar = ruta

    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            viejo = None
            if previo == "confirmado_ausente":
                viejo = str(uuid.uuid4())                                     # nunca existio en el almacen
                r = await idem.reclamar(IDENTIDAD_PLATAFORMA, clave, _hash("a.pdf"), viejo)
                assert await idem.confirmar(r.id, viejo)
                await _envejecer(clave, idem.gracia_segundos() + 30)
            elif previo is not None:
                viejo = almacen.create(ownership=OWNER, caller=f"user:{OWNER.user_id}", capability="ingesta_archivos",
                                       motor="n/a", trace_id="t", prompt="p", recursion_depth=0)
                if previo == "rejected":
                    almacen._index[viejo]["status"] = "rejected"          # fuera del vocabulario que `update` admite
                elif previo not in ("pending", "almacen_sin_integridad"):
                    almacen.update(viejo, status=previo)
                await idem.reclamar(IDENTIDAD_PLATAFORMA, clave, _hash("a.pdf" if mismo_pedido else "otro.pdf"), viejo)
                if previo == "almacen_sin_integridad":
                    almacen._history_integrity = False
            return viejo, await _post(app, clave, "a.pdf")
        finally:
            await _borrar(clave)
    viejo, r = asyncio.run(todo())
    if esperado == "conflicto":
        assert r.status_code == 409, r.text
        assert len(almacen._index) == 1
    elif esperado == "desconocido":
        assert r.status_code == 503 and r.json()["detail"]["code"] == "idempotencia_estado_desconocido", r.text
        assert len(almacen._index) == 0 and ejecutar.await_count == 0
    elif esperado == "no_disponible":
        assert r.status_code == 503 and r.json()["detail"]["code"] == "procesamiento_no_disponible", r.text
        assert ejecutar.await_count == 0
    elif esperado == "mismo":
        assert r.status_code == 202 and r.json()["job_id"] == viejo
        assert len(almacen._index) == 1
    else:
        assert r.status_code == 202 and r.json()["job_id"] != viejo
        assert len(almacen._index) == (1 if viejo is None else 2)


def _lento_tras_ganar(monkeypatch, segundos):
    real = idem.reclamar

    async def lento(*a, **k):
        r = await real(*a, **k)
        if r.gano and not lento.ya:
            lento.ya = True
            await asyncio.sleep(segundos)
        return r
    lento.ya = False
    monkeypatch.setattr(idem, "reclamar", lento)


def test_el_cas_del_reintento_demorado_no_deja_dos_trabajos_vivos(ruta, monkeypatch):
    """A6 del auditor: la ganadora lenta confirma ENTRE el chequeo del reintento y su CAS (el CAS llega 0,6 s
    tarde). Con `confirmado` el CAS exige confirmado=0 y falla: el reintento devuelve el trabajo de la
    ganadora. Antes, el CAS pasaba y quedaban dos trabajos con OCR."""
    app, almacen, ejecutar = ruta
    monkeypatch.setenv("JAX_PROCESAMIENTO_IDEMPOTENCIA_GRACIA_SEGUNDOS", "1")
    _lento_tras_ganar(monkeypatch, 1.6)

    def demorado(real_cas):
        async def cas_lento(*a, **k):
            await asyncio.sleep(0.6)
            return await real_cas(*a, **k)
        return cas_lento
    # LOS DOS CAS (huerfano y fallido): un reintento que usara siempre `tomar_fallido`, que no exige confirmado=0,
    # robaria el reclamo ya confirmado de la ganadora lenta y dejaria dos trabajos vivos.
    monkeypatch.setattr(idem, "tomar_huerfana", demorado(idem.tomar_huerfana))
    monkeypatch.setattr(idem, "tomar_fallido", demorado(idem.tomar_fallido))

    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            async def segundo():
                await asyncio.sleep(1.2)
                return await _post(app, clave, "a.pdf")
            r1, r2 = await asyncio.gather(_post(app, clave, "a.pdf"), segundo())
            return r1, r2, await _filas(clave)
        finally:
            await _borrar(clave)
    r1, r2, filas = asyncio.run(todo())
    assert (r1.status_code, r2.status_code) == (202, 202), (r1.text, r2.text)
    assert r1.json()["job_id"] == r2.json()["job_id"] == filas[0][1]
    assert [j["status"] for j in almacen._index.values() if j["status"] != "cancelled"] and len(
        [j for j in almacen._index.values() if j["status"] != "cancelled"]) == 1
    assert ejecutar.await_count == 1


def test_si_la_confirmacion_falla_se_cancela_el_trabajo_y_se_contesta_503(ruta, monkeypatch):
    """MINOR-B: sin poder confirmar no se programa el OCR: el trabajo propio queda terminal, se libera el
    reclamo y la respuesta es un 503 reintentable. Despues, el reenvio crea UNO."""
    app, almacen, ejecutar = ruta
    real = getattr(idem, "confirmar", None)
    fallos = {"n": 1}

    async def confirmar_que_falla(*a, **k):
        if fallos["n"]:
            fallos["n"] -= 1
            raise ConnectionError("base caida un instante")
        return await real(*a, **k)
    monkeypatch.setattr(idem, "confirmar", confirmar_que_falla, raising=False)

    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            r1 = await _post(app, clave, "a.pdf")
            tras = (len(almacen._index), [j["status"] for j in almacen._index.values()], await _filas(clave),
                    rutas_mod._SEMAFORO_TRABAJOS._value, dict(rutas_mod._CONTROLES), ejecutar.await_count)
            return r1, tras, await _post(app, clave, "a.pdf")
        finally:
            await _borrar(clave)
    r1, tras, r2 = asyncio.run(todo())
    assert r1.status_code == 503 and r1.json()["detail"]["code"] == "idempotencia_no_disponible", r1.text
    n, estados, filas, cupo, controles, ocr = tras
    assert n == 1 and estados[0] in ("failed", "cancelled") and filas == [] and cupo == 4 and controles == {}
    assert ocr == 0
    assert r2.status_code == 202 and ejecutar.await_count == 1


def test_perder_el_reclamo_y_no_poder_releer_tampoco_programa_el_ocr(ruta, monkeypatch):
    """A7 del auditor: la ganadora lenta pierde el reclamo y ademas falla la lectura de la tabla. 503, un solo OCR
    (el del reintento)."""
    app, almacen, ejecutar = ruta
    monkeypatch.setenv("JAX_PROCESAMIENTO_IDEMPOTENCIA_GRACIA_SEGUNDOS", "1")
    _lento_tras_ganar(monkeypatch, 1.6)
    real_buscar = idem.buscar
    t0 = {}

    async def buscar_falla(*a, **k):
        import time
        t0.setdefault("t", time.monotonic())
        if time.monotonic() - t0["t"] > 1.5 and not buscar_falla.ya:
            buscar_falla.ya = True
            raise ConnectionError("base caida un instante")
        return await real_buscar(*a, **k)
    buscar_falla.ya = False
    monkeypatch.setattr(idem, "buscar", buscar_falla)

    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            async def segundo():
                await asyncio.sleep(1.2)
                return await _post(app, clave, "a.pdf")
            return await asyncio.gather(_post(app, clave, "a.pdf"), segundo())
        finally:
            await _borrar(clave)
    r1, r2 = asyncio.run(todo())
    assert r2.status_code == 202
    assert r1.status_code in (202, 503)
    assert ejecutar.await_count == 1
    assert len([j for j in almacen._index.values() if j["status"] not in ("cancelled", "failed")]) == 1


def test_un_cancel_durante_la_confirmacion_encuentra_el_control(ruta, monkeypatch):
    """A9 del auditor (regresion de la ronda 6): el control se registra ANTES de cualquier await posterior al
    create. Un `POST .../cancel` que llega mientras se confirma el reclamo lo encuentra, y el worker arranca ya
    cancelado."""
    app, almacen, ejecutar = ruta
    real = getattr(idem, "confirmar", None)

    async def confirmar_lento(*a, **k):
        await asyncio.sleep(0.3)
        return await real(*a, **k)
    monkeypatch.setattr(idem, "confirmar", confirmar_lento, raising=False)

    async def cancel(job_id):
        transporte = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transporte, base_url="http://las-manos") as c:
            return await c.post(f"/procesamiento/trabajos/{job_id}/cancel", headers=_cabeceras(None))

    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            async def cancelador():
                for _ in range(300):
                    await asyncio.sleep(0.01)
                    if almacen._index:
                        jid = next(iter(almacen._index))
                        return jid, await cancel(jid)
            r1, (jid, rc) = await asyncio.gather(_post(app, clave, "a.pdf"), cancelador())
            return r1, jid, rc
        finally:
            await _borrar(clave)
    r1, jid, rc = asyncio.run(todo())
    assert r1.status_code == 202 and rc.status_code == 200, (r1.text, rc.text)
    control = ejecutar.call_args.kwargs["control"]
    assert control.cancelado is True, "el cancel no se perdio: el worker recibe el control ya cancelado"
    assert almacen._index[jid]["status"] == "cancelling"


def test_cancelar_el_pedido_durante_la_confirmacion_no_deja_un_trabajo_pending(ruta, monkeypatch):
    """A10 del auditor: el pedido se aborta (CancelledError) durante la confirmacion. El trabajo queda terminal,
    sin control ni cupo ni reclamo colgando, y el reenvio crea uno solo."""
    app, almacen, ejecutar = ruta

    real = getattr(idem, "confirmar", None)
    abortar = {"si": True}

    async def confirmar_abortado(*a, **k):
        if abortar["si"]:
            raise asyncio.CancelledError()
        return await real(*a, **k)
    monkeypatch.setattr(idem, "confirmar", confirmar_abortado, raising=False)

    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            with pytest.raises(asyncio.CancelledError):      # la cancelacion SE PROPAGA, no se traga
                await _post(app, clave, "a.pdf")
            tras = (dict(almacen._index), dict(rutas_mod._CONTROLES), rutas_mod._SEMAFORO_TRABAJOS._value,
                    await _filas(clave))
            abortar["si"] = False
            return tras, await _post(app, clave, "a.pdf")
        finally:
            await _borrar(clave)
    (indice, controles, cupo, filas), r2 = asyncio.run(todo())
    assert [j["status"] for j in indice.values()] and all(j["status"] in ("failed", "cancelled") for j in indice.values())
    assert controles == {} and cupo == 4 and filas == []
    assert r2.status_code == 202


def test_confirmado_luego_failed_y_reintento_programa_un_trabajo_nuevo_de_punta_a_punta(ruta):
    """El trabajo se creo y confirmo (confirmado=1), despues fallo (p. ej. LAS MANOS se reinicio) y el llamador, que
    nunca supo su job_id, reintenta con la misma clave: el trabajo NUEVO tiene que confirmar y PROGRAMAR su OCR.
    Si `tomar_fallido` no reiniciara `confirmado`, la confirmacion del trabajo nuevo fallaria y se cancelaria
    sin programar nada (el documento quedaria sin procesar)."""
    app, almacen, ejecutar = ruta

    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            r1 = await _post(app, clave, "a.pdf")
            almacen.update(r1.json()["job_id"], status="failed", error="reinicio")
            r2 = await _post(app, clave, "a.pdf")
            r3 = await _post(app, clave, "a.pdf")
            return r1, r2, r3, await _filas(clave)
        finally:
            await _borrar(clave)
    r1, r2, r3, filas = asyncio.run(todo())
    assert (r1.status_code, r2.status_code, r3.status_code) == (202, 202, 202)
    nuevo = r2.json()["job_id"]
    assert nuevo != r1.json()["job_id"] and r3.json()["job_id"] == nuevo and filas[0][1] == nuevo
    assert ejecutar.await_count == 2, "el trabajo nuevo SI programo su OCR"
    assert almacen._index[nuevo]["status"] not in ("cancelled", "failed")


def _con_almacen_sin_integridad(tmp_path):
    ruta_jsonl = tmp_path / "jobs.jsonl"
    ruta_jsonl.write_bytes(b'{"job_id": "x", "status": "pend')               # ultima linea truncada, sin \n
    almacen = ProcessingJobStore(str(ruta_jsonl))
    assert almacen.authoritative_history_intact is False
    return almacen


def test_sin_integridad_del_almacen_un_pedido_con_clave_da_503_antes_de_reclamar(ruta, tmp_path, monkeypatch):
    """A11 / MAJOR-1: con la historia truncada `authoritative_snapshot` devuelve None para todo. Con clave no se
    reclama nada (ni siquiera una fila en la tabla) y no se programa OCR; sin clave sigue el camino de siempre."""
    app, _almacen, ejecutar = ruta
    monkeypatch.setattr(rutas_mod, "_STORE", _con_almacen_sin_integridad(tmp_path))

    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            return await _post(app, clave, "a.pdf"), await _filas(clave)
        finally:
            await _borrar(clave)
    r, filas = asyncio.run(todo())
    assert r.status_code == 503 and r.json()["detail"]["code"] == "procesamiento_no_disponible", r.text
    assert filas == [] and ejecutar.await_count == 0


def test_a11c_perder_la_integridad_despues_de_crear_no_duplica_el_ocr(ruta, caplog):
    """A11c: el trabajo se crea y confirma con el almacen sano; despues el almacen pierde la integridad y la
    gracia vence una y otra vez con el llamador reintentando. Antes: un trabajo nuevo por reintento (5 OCR).
    Ahora: 503 siempre, 1 solo OCR."""
    app, almacen, ejecutar = ruta
    caplog.set_level(logging.ERROR, logger="procesamiento_routes")

    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            r1 = await _post(app, clave, "a.pdf")
            almacen._history_integrity = False
            salidas = []
            for _ in range(4):
                await _envejecer(clave, idem.gracia_segundos() + 30)
                salidas.append(await _post(app, clave, "a.pdf"))
            return r1, salidas
        finally:
            await _borrar(clave)
    r1, salidas = asyncio.run(todo())
    assert r1.status_code == 202 and all(r.status_code == 503 for r in salidas), [r.text for r in salidas]
    assert ejecutar.await_count == 1
    assert [t for t in caplog.records if "integridad" in t.getMessage()]


def test_confirmado_pero_ausente_es_desconocido_y_nunca_se_retoma(ruta, caplog):
    """MAJOR-1: confirmado=1 con el job_id ausente del almacen (sano) no es un huerfano: 503, error en el log y
    ni un trabajo nuevo."""
    app, almacen, ejecutar = ruta
    caplog.set_level(logging.ERROR, logger="procesamiento_routes")

    async def todo():
        await idem.init_tabla()
        clave = _clave()
        try:
            fantasma = str(uuid.uuid4())
            r = await idem.reclamar(IDENTIDAD_PLATAFORMA, clave, _hash("a.pdf"), fantasma)
            assert await idem.confirmar(r.id, fantasma)
            await _envejecer(clave, idem.gracia_segundos() + 30)
            return await _post(app, clave, "a.pdf"), fantasma, await _filas(clave)
        finally:
            await _borrar(clave)
    r, fantasma, filas = asyncio.run(todo())
    assert r.status_code == 503 and r.json()["detail"]["code"] == "idempotencia_estado_desconocido", r.text
    assert filas == [(IDENTIDAD_PLATAFORMA, fantasma)] and len(almacen._index) == 0 and ejecutar.await_count == 0
    assert [t for t in caplog.records if t.levelno == logging.ERROR and "DESCONOCIDO" in t.getMessage()]


def test_liberar_el_reclamo_no_traga_la_cancelacion_y_libera_igual(monkeypatch):
    """`_liberar_reclamo` relanza el CancelledError del pedido y la liberacion termina de todos modos."""
    real = idem.liberar
    avance = {"termino": False}

    async def todo():
        await idem.init_tabla()
        clave = _clave()
        entro = asyncio.Event()
        try:
            await idem.reclamar(IDENTIDAD_PLATAFORMA, clave, _hash("a.pdf"), "job-z")

            async def liberar_lento(*a, **k):
                entro.set()
                await asyncio.sleep(0.3)
                await real(*a, **k)
                avance["termino"] = True
            monkeypatch.setattr(idem, "liberar", liberar_lento)
            tarea = asyncio.ensure_future(rutas_mod._liberar_reclamo(IDENTIDAD_PLATAFORMA, clave, "job-z"))
            await entro.wait()
            tarea.cancel()
            with pytest.raises(asyncio.CancelledError):
                await tarea
            await asyncio.sleep(0.6)
            return await _filas(clave)
        finally:
            await _borrar(clave)
    assert asyncio.run(todo()) == [] and avance["termino"] is True


def test_si_liberar_falla_tras_una_cancelacion_el_error_queda_en_el_log(caplog):
    """MINOR-B: el pedido se cancela mientras espera la liberacion y la liberacion falla despues: nadie espera la
    tarea, y su excepcion se registra igual (hash corto de la clave y job_id) en vez de quedar sin recoger."""
    caplog.set_level(logging.ERROR, logger="procesamiento_routes")
    clave = _clave()

    async def todo():
        entro = asyncio.Event()
        real = idem.liberar

        async def liberar_que_falla(*a, **k):
            entro.set()
            await asyncio.sleep(0.2)
            raise ConnectionError("base caida al liberar")
        idem.liberar = liberar_que_falla
        try:
            tarea = asyncio.ensure_future(rutas_mod._liberar_reclamo(IDENTIDAD_PLATAFORMA, clave, "job-liberar"))
            await entro.wait()
            tarea.cancel()
            with pytest.raises(asyncio.CancelledError):
                await tarea
            await asyncio.sleep(0.5)
        finally:
            idem.liberar = real
    asyncio.run(todo())
    errores = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any(idem.abreviar(clave) in m and "job-liberar" in m and "ConnectionError" in m for m in errores), errores
    assert not any(clave in m for m in errores), "la clave entera no va al log"


def test_init_tabla_no_deja_warning_de_columna_duplicada_y_migra_una_tabla_vieja():
    import warnings

    async def sql(q):
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute(q)

    async def todo():
        await idem.init_tabla()
        await sql(f"ALTER TABLE {idem.NOMBRE_TABLA} DROP COLUMN confirmado")      # tabla de la ronda 2
        with warnings.catch_warnings(record=True) as primera:
            warnings.simplefilter("always")
            await idem.init_tabla()                                              # la agrega
        with warnings.catch_warnings(record=True) as segunda:
            warnings.simplefilter("always")
            await idem.init_tabla()                                              # ya esta: ni ALTER ni warning
        return primera, segunda
    primera, segunda = asyncio.run(todo())
    assert not [w for w in primera + segunda if "confirmado" in str(w.message) or "Duplicate" in str(w.message)]


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
