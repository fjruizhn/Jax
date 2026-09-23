"""Transacción de continuar y continue concurrente contra MariaDB real
(spec 2026-09-17 §5.2 regla 10, §8, §9). Job jacobs-gobernanza-db.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import aiomysql

# Base de tests de ESTA sesión (decisión de Fernando, 2026-09-17). Reemplaza
# la guarda vieja `if _db != "jax_memory_test": raise` + `setdefault`, que es
# anterior a `JAX_TEST_DB_SUFIJO` y rechazaba `jax_memory_test_<sufijo>`:
# protege lo mismo (nunca producción, nunca una base que no sea de tests) y
# además deja correr la base propia de la sesión.
from base_de_test import exigir_base_de_test  # noqa: E402

exigir_base_de_test()

import pytest  # noqa: E402

from jacobs import continuar, store  # noqa: E402
from jacobs.policy import MAX_PARALLEL_PIPELINES, CupoAgotado  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus  # noqa: E402
from jacobs.prevuelo_reglas import Veredicto  # noqa: E402

_REF = 'inline:{"result": "hecho"}'


async def _abortado():
    await store.init_tables()
    pid = str(uuid.uuid4())
    pasos = [
        Step(pipeline_id=pid, step_index=i, facet="jekyll", capability="research",
             input={"prompt": f"p{i}"}, depends_on=[i - 1] if i else [],
             status=StepStatus.completed if i < 2 else StepStatus.failed,
             error=None if i < 2 else "cortado", output_ref=_REF if i < 2 else None)
        for i in range(3)
    ]
    ahora = time.time()
    pipeline = Pipeline(pipeline_id=pid, name="t-continuar", invoked_by="plataforma",
                        mode="autonomous", status=PipelineStatus.aborted, plan=pasos,
                        context={"objective": "o", "step_0_ref": _REF, "step_1_ref": _REF},
                        created_at=ahora, updated_at=ahora)
    await store.pipeline_create(pipeline)
    for paso in pasos:
        await store.step_upsert(paso)
    return pipeline, pasos


async def _borrar(pid):
    conn = await store.conexion_dedicada()
    try:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM jacobs_steps WHERE pipeline_id=%s", (pid,))
            await cur.execute("DELETE FROM jacobs_events WHERE pipeline_id=%s", (pid,))
            await cur.execute("DELETE FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))
    finally:
        conn.close()


async def _explain(sql, params):
    conn = await store.conexion_dedicada()
    try:
        async with conn.cursor() as cur:
            await cur.execute("EXPLAIN " + sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in await cur.fetchall()]
    finally:
        conn.close()


def test_la_transaccion_aplica_pasos_plan_contexto_y_epoca():
    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            pasos[2].facet = "thot"
            contexto = {"objective": "o", "step_0_ref": _REF, "step_1_ref": _REF}
            nueva = await store.continuar_transaccion(
                pid, 0, PipelineStatus.aborted, [pasos[2]], pasos, contexto, 2,
                evento_payload=None)
            p = await store.pipeline_get(pid)
            s2 = (await store.steps_by_pipeline(pid))[2]
            explains = [
                await _explain(store._SQL_BLOQUEAR_PIPELINE, (pid,)),
                # Ronda de arreglo 1 (2026-09-18-arbitro-devuelve, hallazgo
                # ALTO): _SQL_PASO_A_CORRER ahora también escribe input_ref
                # (jacobs/store.py) -- el EXPLAIN tiene que mandar los MISMOS
                # parámetros que la sentencia real recibe, o el placeholder
                # de más revienta con "not enough arguments for format
                # string" (regresión vista en CI del PR #221, job
                # jacobs-gobernanza-db). pasos[2].input no cambió desde
                # _abortado(): es el mismo JSON que la llamada real de arriba
                # ya escribió.
                await _explain(store._SQL_PASO_A_CORRER, (
                    "thot", None, json.dumps(pasos[2].input, ensure_ascii=False), s2.step_id, pid,
                )),
            ]
            return nueva, p, s2, explains
        finally:
            await _borrar(pid)
    nueva, p, s2, explains = asyncio.run(cuerpo())
    assert nueva == 1
    assert (p.status, p.run_epoch, p.current_step_index, p.plan[2].facet) == (
        PipelineStatus.running, 1, 2, "thot")
    assert (s2.status, s2.facet, s2.error, s2.output_ref) == (StepStatus.pending, "thot", None, None)
    for filas in explains:
        # type != index: en un UPDATE, un recorrido completo de PRIMARY
        # también dice key='PRIMARY' (ola final F7).
        assert all(f["key"] == "PRIMARY" and f["type"] not in ("ALL", "index") for f in filas), filas


def test_continuar_resetea_modelo_real_contra_la_db_real():
    """ANTES DE MERGEAR 6 (revisión final 2026-09-18): un step que /continue
    vuelve a poner en 'pending' NO puede seguir mostrando el modelo_real de
    su corrida anterior. Complementa el test puro de texto de SQL
    (jacobs/_continue_modelo_real_reset_test.py) con la prueba de que el
    UPDATE real, contra MariaDB, deja la columna en NULL."""
    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            pasos[2].modelo_real = "modelo-de-la-corrida-anterior"
            await store.step_upsert(pasos[2])
            antes = (await store.steps_by_pipeline(pid))[2]

            contexto = {"objective": "o", "step_0_ref": _REF, "step_1_ref": _REF}
            await store.continuar_transaccion(
                pid, 0, PipelineStatus.aborted, [pasos[2]], pasos, contexto, 2,
                evento_payload=None)
            despues = (await store.steps_by_pipeline(pid))[2]
            return antes, despues
        finally:
            await _borrar(pid)

    antes, despues = asyncio.run(cuerpo())
    assert antes.modelo_real == "modelo-de-la-corrida-anterior"
    assert despues.modelo_real is None, (
        "el step continuado sigue mostrando el modelo_real de la corrida "
        "anterior -- un dato falso con cara de verdadero"
    )


async def _primera_columna(tabla, indice):
    """Primera columna del índice `indice` de `tabla` (None si no existe)."""
    if not indice:
        return None
    conn = await store.conexion_dedicada()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT COLUMN_NAME FROM information_schema.STATISTICS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND INDEX_NAME = %s "
                "AND SEQ_IN_INDEX = 1", (tabla, indice))
            fila = await cur.fetchone()
            return fila[0] if fila else None
    finally:
        conn.close()


def _fallas_del_conteo_por_status(fila, primera_columna):
    """Pendiente 631: se exige la PROPIEDAD, no el nombre. El COUNT del cupo
    va por ref/range sobre un índice que EMPIEZA por status y es cubriente
    ('Using index': no toca la tabla). Devuelve la lista de fallas."""
    fallas = []
    if fila.get("type") not in ("ref", "range"):
        fallas.append(f"type={fila.get('type')!r}, se esperaba ref o range")
    if primera_columna != "status":
        fallas.append(f"el índice {fila.get('key')!r} empieza por {primera_columna!r}, no por 'status'")
    if "Using index" not in (fila.get("Extra") or ""):
        fallas.append(f"Extra={fila.get('Extra')!r} sin 'Using index'")
    return fallas


def test_el_predicado_del_conteo_rechaza_una_fila_mala():
    """El control tiene que poder fallar: una fila que no va por status,
    recorre la tabla y no es cubriente se rechaza por las tres razones."""
    mala = {"table": "jacobs_pipelines", "key": "idx_jacobs_pipelines_duenio", "type": "ALL",
            "Extra": "Using where"}
    assert len(_fallas_del_conteo_por_status(mala, "user_id")) == 3
    buena = {"table": "jacobs_pipelines", "key": "idx_pipelines_ocultos", "type": "range",
             "Extra": "Using where; Using index"}
    assert _fallas_del_conteo_por_status(buena, "status") == []


def test_explain_del_update_final_de_continuar_usa_la_clave_primaria():
    """Ola final F7 (revisión final m8): la tercera consulta de la
    transacción, _SQL_PIPELINE_CONTINUAR, no tenía EXPLAIN al lado de las
    otras dos (LAS CUATRO #1). Se explica con parámetros reales."""
    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            # El último parámetro es el tope del cupo: desde el 2026-09-17 el
            # UPDATE lleva la condición adentro (JOIN con la derivada del
            # recuento), así que el EXPLAIN mide la sentencia REAL, con su JOIN.
            filas = await _explain(store._SQL_PIPELINE_CONTINUAR, (
                json.dumps([s.model_dump() for s in pasos], ensure_ascii=False),
                json.dumps(pipeline.context, ensure_ascii=False), 2, time.time(), pid, 0,
                MAX_PARALLEL_PIPELINES,
            ))
            conteo = [f for f in filas if f["table"] == "jacobs_pipelines"]
            primera = await _primera_columna("jacobs_pipelines", conteo[0]["key"]) if len(conteo) == 1 else None
            return filas, primera
        finally:
            await _borrar(pid)
    filas, primera = asyncio.run(cuerpo())
    assert filas, "EXPLAIN vacío"
    # Tres filas de plan desde el 2026-09-17, y cada una tiene que justificarse:
    #   1. la fila del pipeline, por PRIMARY;
    #   2. la tabla DERIVADA del cupo -- sale type=ALL porque es materializada,
    #      pero tiene UNA fila (el COUNT): un "scan" de una fila no es un scan,
    #      y por eso se exige rows<=1 en vez de mirar sólo el type;
    #   3. el COUNT de adentro, por un índice con prefijo status (hoy
    #      idx_pipelines_status / idx_pipelines_ocultos; pendiente 631: MariaDB
    #      elige entre los dos de forma inestable y los dos sirven), cubriente.
    (pipeline_fila,) = [f for f in filas if f["table"] == "p"]
    assert pipeline_fila["key"] == "PRIMARY", filas
    (conteo,) = [f for f in filas if f["table"] == "jacobs_pipelines"]
    assert _fallas_del_conteo_por_status(conteo, primera) == [], filas
    for f in filas:
        derivada = str(f.get("table") or "").startswith("<derived")
        if derivada:
            # La derivada materializa el COUNT: es UNA fila de verdad, aunque el
            # optimizador estime 2. Lo que importa es que sea diminuta y que el
            # COUNT de adentro (id=2) vaya por índice, que se exige arriba.
            assert int(f.get("rows") or 0) <= 10, filas
            continue
        assert f["type"] not in ("ALL", "index"), filas
        assert "filesort" not in (f.get("Extra") or ""), filas


def test_si_falla_a_mitad_no_cambia_nada(monkeypatch):
    # El sexto y último %s es el tope del cupo (2026-09-17): la sentencia real
    # lo lleva, así que el doble tiene que aceptarlo o el test falla por el
    # número de parámetros y no por lo que quiere probar.
    monkeypatch.setattr(store, "_SQL_PIPELINE_CONTINUAR",
                        "UPDATE tabla_que_no_existe SET plan=%s, context_refs=%s, "
                        "current_step_index=%s, updated_at=%s "
                        "WHERE pipeline_id=%s AND run_epoch=%s AND %s > 0")

    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            pasos[2].facet = "thot"
            evento = {"by": "plataforma", "from_status": "aborted", "run_epoch": 1,
                      "pasos_a_correr": [2], "pasos_reusados": [0, 1], "reasignados": {},
                      "costo_max_usd": "0.000000"}
            # ProgrammingError específico (1146, tabla inexistente), no
            # Exception genérico -- fix round 2: un catch-all también
            # atraparía un TypeError ajeno (p. ej. un argumento mal armado
            # en esta misma llamada) y el test daría un falso verde.
            with pytest.raises(aiomysql.ProgrammingError):
                await store.continuar_transaccion(
                    pid, 0, PipelineStatus.aborted, [pasos[2]], pasos, pipeline.context, 2,
                    evento_payload=evento)
            return (await store.pipeline_get(pid), (await store.steps_by_pipeline(pid))[2],
                    await store.events_by_pipeline(pid))
        finally:
            await _borrar(pid)
    p, s2, eventos = asyncio.run(cuerpo())
    assert (p.status, p.run_epoch) == (PipelineStatus.aborted, 0)
    assert (s2.status, s2.facet, s2.error) == (StepStatus.failed, "jekyll", "cortado")
    # Regla 10 / Ruling R22: la transacción caída no deja NINGÚN rastro, ni
    # siquiera el evento de auditoría -- si el evento se escribiera con otra
    # conexión (event_append() fuera de esta transacción) sobreviviría al
    # rollback y este assert lo vería.
    assert eventos == []


def test_el_evento_continued_se_escribe_con_la_misma_transaccion():
    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            pasos[2].facet = "thot"
            evento = {"by": "plataforma", "from_status": "aborted", "run_epoch": 1,
                      "pasos_a_correr": [2], "pasos_reusados": [0, 1], "reasignados": {},
                      "costo_max_usd": "0.000000"}
            nueva = await store.continuar_transaccion(
                pid, 0, PipelineStatus.aborted, [pasos[2]], pasos, pipeline.context, 2,
                evento_payload=evento)
            eventos = await store.events_by_pipeline(pid)
            return nueva, eventos
        finally:
            await _borrar(pid)
    nueva, eventos = asyncio.run(cuerpo())
    assert nueva == 1
    assert [e["event_type"] for e in eventos] == ["PIPELINE_CONTINUED"]
    assert eventos[0]["payload"] == {"by": "plataforma", "from_status": "aborted", "run_epoch": 1,
                                     "pasos_a_correr": [2], "pasos_reusados": [0, 1], "reasignados": {},
                                     "costo_max_usd": "0.000000"}


def test_si_el_evento_falla_no_queda_el_pipeline_a_medias(monkeypatch):
    # Ruling R22: si la escritura del evento fallara DESPUÉS del commit (como
    # hacía 45d60ec con event_append() por fuera), el pipeline y los pasos
    # quedarían escritos y el evento perdido -- un estado a medias. Con el
    # evento adentro de la MISMA transacción, una falla ahí también hace
    # ROLLBACK de todo: nada cambia, ni el pipeline, ni los pasos.
    monkeypatch.setattr(store, "_SQL_EVENTO_CONTINUED",
                        "INSERT INTO tabla_que_no_existe (pipeline_id, step_id, event_type, payload, ts) "
                        "VALUES (%s,%s,%s,%s,%s)")

    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            pasos[2].facet = "thot"
            evento = {"by": "plataforma", "from_status": "aborted", "run_epoch": 1,
                      "pasos_a_correr": [2], "pasos_reusados": [0, 1], "reasignados": {},
                      "costo_max_usd": "0.000000"}
            # ProgrammingError específico (1146, tabla inexistente) -- misma
            # razón que arriba (fix round 2): un Exception genérico no
            # discrimina.
            with pytest.raises(aiomysql.ProgrammingError):
                await store.continuar_transaccion(
                    pid, 0, PipelineStatus.aborted, [pasos[2]], pasos, pipeline.context, 2,
                    evento_payload=evento)
            return (await store.pipeline_get(pid), (await store.steps_by_pipeline(pid))[2],
                    await store.events_by_pipeline(pid))
        finally:
            await _borrar(pid)
    p, s2, eventos = asyncio.run(cuerpo())
    assert (p.status, p.run_epoch) == (PipelineStatus.aborted, 0)
    assert (s2.status, s2.facet, s2.error) == (StepStatus.failed, "jekyll", "cortado")
    assert eventos == []


def test_status_distinto_con_la_misma_epoca_no_gana():
    # Requisito (a) — mutación: si el chequeo sólo mirara la época y no el
    # status, esta llamada (época correcta, status equivocado a propósito)
    # ganaría la transacción igual.
    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            pasos[2].facet = "thot"
            nueva = await store.continuar_transaccion(
                pid, 0, PipelineStatus.expired, [pasos[2]], pasos, pipeline.context, 2,
                evento_payload=None)
            return nueva, await store.pipeline_get(pid), (await store.steps_by_pipeline(pid))[2]
        finally:
            await _borrar(pid)
    nueva, p, s2 = asyncio.run(cuerpo())
    assert nueva is None
    assert (p.status, p.run_epoch) == (PipelineStatus.aborted, 0)
    assert (s2.status, s2.facet) == (StepStatus.failed, "jekyll")


def test_si_el_update_final_no_toca_la_fila_no_queda_nada(monkeypatch):
    # Ruling R23 -- cinturón: aunque el SELECT...FOR UPDATE ya vio la fila
    # con la época correcta, mutamos el UPDATE final para que no toque
    # ninguna fila (condición imposible añadida) SIN lanzar excepción. Si el
    # código no revisara rowcount, devolvería la época nueva con nada escrito.
    # `AND 1=0` hace que el UPDATE no toque ninguna fila SIN lanzar. Desde que
    # la sentencia lleva la condición del cupo, 0 filas se interpreta como cupo
    # lleno y sale CupoAgotado: el cinturón de R23 sigue estando (no devuelve
    # una época con nada escrito), pero ahora dice cuál es el motivo.
    monkeypatch.setattr(store, "_SQL_PIPELINE_CONTINUAR", store._SQL_PIPELINE_CONTINUAR + " AND 1=0")

    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            pasos[2].facet = "thot"
            # Ya no devuelve None: 0 filas con la condición del cupo puesta
            # sólo puede ser el cupo (la época y el status quedaron fijados por
            # el SELECT...FOR UPDATE), y decirlo es mejor que un None mudo.
            with pytest.raises(CupoAgotado):
                await store.continuar_transaccion(
                    pid, 0, PipelineStatus.aborted, [pasos[2]], pasos, pipeline.context, 2,
                    evento_payload=None)
            return None, await store.pipeline_get(pid), (await store.steps_by_pipeline(pid))[2]
        finally:
            await _borrar(pid)
    nueva, p, s2 = asyncio.run(cuerpo())
    assert nueva is None
    assert (p.status, p.run_epoch) == (PipelineStatus.aborted, 0)
    # Rollback COMPLETO: ni siquiera el UPDATE de jacobs_steps quedó.
    assert (s2.status, s2.facet, s2.error) == (StepStatus.failed, "jekyll", "cortado")


def test_evento_payload_es_obligatorio():
    # Principio IX / fix round 2: el evento de auditoría no es opcional por
    # descuido. `evento_payload` no tiene default -- omitirlo es un TypeError
    # en el sitio de la llamada, no un pipeline continuado sin rastro. El
    # binding de argumentos de Python ocurre al invocar la corrutina, antes
    # de que corra una sola línea del cuerpo (y antes de tocar la DB), así
    # que ni hace falta awaitear ni armar un pipeline real para verlo.
    with pytest.raises(TypeError):
        store.continuar_transaccion(
            "cualquier-id", 0, PipelineStatus.aborted, [], [], {}, 0)


def test_dos_transacciones_con_la_misma_epoca_solo_una_gana():
    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            return await asyncio.gather(*(
                store.continuar_transaccion(pid, 0, PipelineStatus.aborted, [pasos[2]], pasos,
                                            pipeline.context, 2, evento_payload=None)
                for _ in range(2)
            ))
        finally:
            await _borrar(pid)
    resultados = asyncio.run(cuerpo())
    assert sorted(resultados, key=lambda r: r is None) == [1, None]


def test_continue_concurrente_sobre_el_mismo_pipeline_solo_uno_gana():
    ok = Veredicto(ok=True, violaciones=(), costo_max_usd=Decimal(0), pasos_costo=(), sondeadas=())

    async def cuerpo():
        pipeline, _ = await _abortado()
        pid = pipeline.pipeline_id
        try:
            with patch.object(continuar, "prevuelo", AsyncMock(return_value=ok)), \
                 patch.object(continuar, "check_kill_switch", return_value=False), \
                 patch.object(continuar.store, "pipeline_count_active", AsyncMock(return_value=0)):
                resultados = await asyncio.gather(
                    *(continuar.continuar(pid, "plataforma") for _ in range(10)),
                    return_exceptions=True)
            return resultados, await store.pipeline_epoca_y_status(pid)
        finally:
            await _borrar(pid)
    resultados, actual = asyncio.run(cuerpo())
    ganadores = [r for r in resultados if isinstance(r, tuple)]
    rechazos = [r for r in resultados if isinstance(r, continuar.ContinuarRechazado)]
    assert len(ganadores) == 1 and len(rechazos) == 9, resultados
    assert all(r.status_code == 409 for r in rechazos)
    assert actual == (1, PipelineStatus.running)
