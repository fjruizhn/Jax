"""Contrato de sub-pipelines de punta a punta por la ruta real (frente F, 2026-09-16).

El arnés (`jacobs/_arnes_ada.py`) hace de Ada: arma un padre `running` con un
paso de Ada, emite tokens con la función de servidor y pide hijos por
`routes.create_pipeline`. Token, base y eventos son los reales.

ROJO CONTRA MASTER (984ce46): `test_ada_con_token_inventado_no_crea_pipeline`
falló con "DID NOT RAISE": `validate_create` aceptaba cualquier string no vacío
como token y creaba el pipeline.

Corre con:
  bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -v tests/test_subpipeline_contrato_rutas.py"
"""
from __future__ import annotations

import os

from jacobs import _arnes_ada as ada  # primero: barrera de base de prueba

import asyncio  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from jacobs import store  # noqa: E402
from jacobs import subpipelines as sp  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.getenv("JAX_DB_HOST"),
    reason="necesita la MariaDB real con JAX_DB_NAME=jax_memory_test",
)

TOKEN_INVENTADO = "token-inventado-por-el-atacante"


def test_ada_con_token_inventado_no_crea_pipeline():
    async def escenario():
        padre, _paso = await ada.padre_en_ejecucion()
        try:
            with pytest.raises(HTTPException) as rechazo:
                await ada.pedir_hijo(TOKEN_INVENTADO, padre)
            return rechazo.value
        finally:
            await ada.cerrar(padre)

    rechazo = _correr(escenario)
    assert rechazo.status_code == 403
    assert "token_desconocido" in rechazo.detail
    assert TOKEN_INVENTADO not in rechazo.detail


def _correr(escenario):
    async def con_cierre():
        # Cada asyncio.run es un loop nuevo con su pool (jacobs/store.py): se
        # cierra antes de que el loop muera, para no dejar sockets colgados.
        try:
            return await escenario()
        finally:
            await store.cerrar_pool()
    return asyncio.run(con_cierre())


def test_camino_legitimo_crea_el_hijo_con_padre_y_profundidad_del_token():
    async def escenario():
        padre, paso = await ada.padre_en_ejecucion()
        try:
            token = await ada.emitir(padre, paso)
            respuesta = await ada.pedir_hijo(token, padre)
            hijo = await store.pipeline_get(respuesta["pipeline_id"])
            fila = await ada.fila_token(sp.hash_token(token))
            eventos_padre = await store.events_by_pipeline(padre)
            eventos_hijo = await store.events_by_pipeline(hijo.pipeline_id)
            return padre, paso, respuesta, hijo, fila, eventos_padre, eventos_hijo
        finally:
            await ada.cerrar(padre)

    padre, paso, respuesta, hijo, fila, eventos_padre, eventos_hijo = _correr(escenario)
    assert respuesta["status"] == "completed"
    assert (hijo.parent_pipeline_id, hijo.depth, hijo.invoked_by) == (padre, 1, "ada")
    assert fila["hijo_pipeline_id"] == hijo.pipeline_id
    assert fila["usado_at"] is not None
    creados = [e for e in eventos_padre if e["event_type"] == "SUBPIPELINE_CREADO"]
    assert [(e["payload"]["hijo_pipeline_id"], e["payload"]["depth"], e["step_id"])
            for e in creados] == [(hijo.pipeline_id, 1, paso)]
    creacion = [e for e in eventos_hijo if e["event_type"] == "PIPELINE_CREATED"]
    assert creacion[0]["payload"]["parent_pipeline_id"] == padre


def _rechazo_403(rechazo: HTTPException, motivo: sp.Motivo, token: str) -> None:
    assert rechazo.status_code == 403, rechazo.detail
    assert rechazo.detail == f"subpipeline_token rechazado: {motivo.value}"
    assert token not in rechazo.detail


def test_token_reusado_se_rechaza():
    async def escenario():
        padre, paso = await ada.padre_en_ejecucion()
        try:
            token = await ada.emitir(padre, paso)
            await ada.pedir_hijo(token, padre)
            with pytest.raises(HTTPException) as rechazo:
                await ada.pedir_hijo(token, padre)
            return token, rechazo.value
        finally:
            await ada.cerrar(padre)

    token, rechazo = _correr(escenario)
    _rechazo_403(rechazo, sp.Motivo.TOKEN_USADO, token)


def test_token_de_otro_padre_se_rechaza_y_no_se_quema():
    async def escenario():
        padre_a, paso_a = await ada.padre_en_ejecucion()
        padre_b, _ = await ada.padre_en_ejecucion()
        try:
            token_de_a = await ada.emitir(padre_a, paso_a)
            with pytest.raises(HTTPException) as rechazo:
                await ada.pedir_hijo(token_de_a, padre_b)
            fila = await ada.fila_token(sp.hash_token(token_de_a))
            return token_de_a, rechazo.value, fila
        finally:
            await ada.cerrar(padre_a)
            await ada.cerrar(padre_b)

    token, rechazo, fila = _correr(escenario)
    _rechazo_403(rechazo, sp.Motivo.PADRE_NO_COINCIDE, token)
    assert fila["usado_at"] is None


def test_token_vencido_se_rechaza():
    async def escenario():
        padre, paso = await ada.padre_en_ejecucion()
        try:
            token = await ada.emitir(padre, paso)
            await ada.ejecutar(
                "UPDATE jacobs_subpipeline_tokens SET vence_at = %s WHERE token_hash = %s",
                (time.time() - 1, sp.hash_token(token)),
            )
            with pytest.raises(HTTPException) as rechazo:
                await ada.pedir_hijo(token, padre)
            return token, rechazo.value
        finally:
            await ada.cerrar(padre)

    token, rechazo = _correr(escenario)
    _rechazo_403(rechazo, sp.Motivo.TOKEN_VENCIDO, token)


def test_padre_terminado_se_rechaza():
    async def escenario():
        padre, paso = await ada.padre_en_ejecucion()
        token = await ada.emitir(padre, paso)
        await ada.cerrar(padre)
        with pytest.raises(HTTPException) as rechazo:
            await ada.pedir_hijo(token, padre)
        return token, rechazo.value

    token, rechazo = _correr(escenario)
    _rechazo_403(rechazo, sp.Motivo.PADRE_INACTIVO, token)


def test_profundidad_excedida_se_rechaza(monkeypatch):
    async def escenario():
        padre, paso = await ada.padre_en_ejecucion(depth=3)
        try:
            monkeypatch.setenv(sp.ENV_MAX_PROFUNDIDAD, "4")
            token = await ada.emitir(padre, paso)  # token para un hijo de profundidad 4
            monkeypatch.setenv(sp.ENV_MAX_PROFUNDIDAD, "3")  # el máximo por defecto
            with pytest.raises(HTTPException) as rechazo:
                await ada.pedir_hijo(token, padre)
            return token, rechazo.value
        finally:
            await ada.cerrar(padre)

    token, rechazo = _correr(escenario)
    _rechazo_403(rechazo, sp.Motivo.PROFUNDIDAD_EXCEDIDA, token)


def test_el_cuerpo_no_fija_la_profundidad(monkeypatch):
    monkeypatch.delenv(sp.ENV_MAX_PROFUNDIDAD, raising=False)  # máximo por defecto (3)

    async def escenario():
        padre, paso = await ada.padre_en_ejecucion(depth=2)
        try:
            token = await ada.emitir(padre, paso)
            respuesta = await ada.pedir_hijo(
                token, padre, cuerpo_extra={"depth": 0, "subpipeline_depth": 0})
            return await store.pipeline_get(respuesta["pipeline_id"])
        finally:
            await ada.cerrar(padre)

    hijo = _correr(escenario)
    assert hijo.depth == 3


def test_kill_switch_rechaza_sin_consumir_el_token():
    async def escenario():
        padre, paso = await ada.padre_en_ejecucion()
        try:
            token = await ada.emitir(padre, paso)
            with pytest.raises(HTTPException) as frenado:
                await ada.pedir_hijo(token, padre, kill_switch=True)
            fila_frenado = await ada.fila_token(sp.hash_token(token))
            respuesta = await ada.pedir_hijo(token, padre)
            return frenado.value, fila_frenado, respuesta
        finally:
            await ada.cerrar(padre)

    frenado, fila_frenado, respuesta = _correr(escenario)
    assert frenado.status_code == 423
    assert fila_frenado["usado_at"] is None
    assert respuesta["status"] == "completed"


def test_rechazo_deja_evento_con_motivo_y_sin_el_token():
    async def escenario():
        padre, paso = await ada.padre_en_ejecucion()
        token = await ada.emitir(padre, paso)
        await ada.cerrar(padre)
        with pytest.raises(HTTPException):
            await ada.pedir_hijo(token, padre)
        rechazo = await ada.una_fila(
            "SELECT payload FROM jacobs_events WHERE event_type = 'SUBPIPELINE_RECHAZADO' "
            "AND JSON_VALUE(payload, '$.parent_pipeline_id_declarado') = %s",
            (padre,),
        )
        eventos_padre = await store.events_by_pipeline(padre)
        return token, rechazo, eventos_padre

    token, rechazo, eventos_padre = _correr(escenario)
    assert rechazo is not None
    assert json.loads(rechazo["payload"])["motivo"] == sp.Motivo.PADRE_INACTIVO.value
    assert token not in rechazo["payload"]
    assert all(token not in json.dumps(e["payload"]) for e in eventos_padre)


async def _plan_inejecutable(pipeline_id, objective, max_steps, steps_spec):
    raise HTTPException(status_code=422, detail="plan inejecutable (arnés)")


def test_plan_rechazado_quema_el_token_y_deja_evento():
    """Decisión del plan: el consumo va ANTES de planificar (planificar puede
    tardar 20-40 s con un LLM y no se sostiene una transacción ese tiempo). Un
    plan rechazado deja el token usado: Ada pide otro. Fail-closed."""
    async def escenario():
        padre, paso = await ada.padre_en_ejecucion()
        try:
            token = await ada.emitir(padre, paso)
            with pytest.raises(HTTPException) as rechazo:
                await ada.pedir_hijo(token, padre, plan=_plan_inejecutable)
            fila = await ada.fila_token(sp.hash_token(token))
            evento = await ada.una_fila(
                "SELECT payload FROM jacobs_events WHERE event_type = 'SUBPIPELINE_RECHAZADO' "
                "AND JSON_VALUE(payload, '$.parent_pipeline_id') = %s",
                (padre,),
            )
            return rechazo.value, fila, evento
        finally:
            await ada.cerrar(padre)

    rechazo, fila, evento = _correr(escenario)
    assert rechazo.status_code == 422
    assert fila["usado_at"] is not None
    payload = json.loads(evento["payload"])
    assert (payload["fase"], payload["motivo"]) == ("plan", sp.Motivo.PLAN_RECHAZADO.value)


def test_fallo_de_creacion_tras_el_consumo_deja_evento_y_propaga_el_error():
    """Revisión final, I-2: después de consumir, CUALQUIER excepción (no solo el
    plan rechazado) quema el token; tiene que quedar SUBPIPELINE_RECHAZADO con
    fase=creacion y el error original tiene que llegar al llamador, aunque
    escribir ese evento también falle (M-2)."""
    from unittest.mock import AsyncMock, patch

    async def escenario():
        padre, paso = await ada.padre_en_ejecucion()
        try:
            token = await ada.emitir(padre, paso)
            token_2 = await ada.emitir(padre, paso)
            with patch.object(store, "pipeline_create",
                              AsyncMock(side_effect=RuntimeError("fallo de escritura (arnés)"))), \
                 pytest.raises(RuntimeError, match="fallo de escritura"):
                await ada.pedir_hijo(token, padre)
            fila = await ada.fila_token(sp.hash_token(token))
            evento = await ada.una_fila(
                "SELECT payload FROM jacobs_events WHERE event_type = 'SUBPIPELINE_RECHAZADO' "
                "AND JSON_VALUE(payload, '$.parent_pipeline_id') = %s "
                "AND JSON_VALUE(payload, '$.fase') = 'creacion'",
                (padre,),
            )
            # El evento tampoco se puede escribir: el error que sube es el ORIGINAL.
            with patch.object(store, "pipeline_create",
                              AsyncMock(side_effect=RuntimeError("fallo de escritura (arnés)"))), \
                 patch.object(store, "event_append",
                              AsyncMock(side_effect=ConnectionError("base caída (arnés)"))), \
                 pytest.raises(RuntimeError, match="fallo de escritura"):
                await ada.pedir_hijo(token_2, padre)
            return token, fila, evento
        finally:
            await ada.cerrar(padre)

    token, fila, evento = _correr(escenario)
    assert fila["usado_at"] is not None
    assert evento is not None, "token quemado sin SUBPIPELINE_RECHAZADO fase=creacion"
    payload = json.loads(evento["payload"])
    assert payload["motivo"] == sp.Motivo.CREACION_FALLIDA.value
    assert payload["excepcion"] == "RuntimeError"
    assert "fallo de escritura" not in evento["payload"]
    assert token not in evento["payload"]


def test_el_hijo_hereda_la_identidad_del_padre():
    """Revisión final, I-3: el executor carga el uso a user_id/tenant_id del
    pipeline. La identidad del hijo sale del PADRE, no del cuerpo."""
    async def escenario():
        padre, paso = await ada.padre_en_ejecucion(user_id="u-padre", tenant_id="t-padre")
        try:
            sin_identidad = await ada.pedir_hijo(await ada.emitir(padre, paso), padre)
            misma_identidad = await ada.pedir_hijo(
                await ada.emitir(padre, paso), padre,
                cuerpo_extra={"user_id": "u-padre", "tenant_id": "t-padre"})
            return [await store.pipeline_get(r["pipeline_id"])
                    for r in (sin_identidad, misma_identidad)]
        finally:
            await ada.cerrar(padre)

    for hijo in _correr(escenario):
        assert (hijo.user_id, hijo.tenant_id) == ("u-padre", "t-padre")


def test_cuerpo_con_otra_identidad_se_rechaza_sin_quemar_el_token():
    async def escenario():
        padre, paso = await ada.padre_en_ejecucion(user_id="u-padre", tenant_id="t-padre")
        try:
            token = await ada.emitir(padre, paso)
            rechazos = []
            for extra in ({"user_id": "u-otro"}, {"tenant_id": "t-otro"}):
                with pytest.raises(HTTPException) as rechazo:
                    await ada.pedir_hijo(token, padre, cuerpo_extra=extra)
                rechazos.append(rechazo.value)
            fila = await ada.fila_token(sp.hash_token(token))
            evento = await ada.una_fila(
                "SELECT payload FROM jacobs_events WHERE event_type = 'SUBPIPELINE_RECHAZADO' "
                "AND pipeline_id = %s AND JSON_VALUE(payload, '$.motivo') = %s",
                (padre, sp.Motivo.IDENTIDAD_NO_COINCIDE.value),
            )
            return token, rechazos, fila, evento
        finally:
            await ada.cerrar(padre)

    token, rechazos, fila, evento = _correr(escenario)
    for rechazo in rechazos:
        _rechazo_403(rechazo, sp.Motivo.IDENTIDAD_NO_COINCIDE, token)
    assert fila["usado_at"] is None
    assert evento is not None, "rechazo por identidad sin evento en el padre"
    assert token not in evento["payload"]


async def _evento_quemado(padre: str, fase: str) -> dict | None:
    return await ada.una_fila(
        "SELECT pipeline_id, payload FROM jacobs_events "
        "WHERE event_type = 'SUBPIPELINE_RECHAZADO' "
        "AND JSON_VALUE(payload, '$.parent_pipeline_id') = %s "
        "AND JSON_VALUE(payload, '$.fase') = %s",
        (padre, fase),
    )


async def _hijos_de(padre: str) -> int:
    fila = await ada.una_fila(
        "SELECT COUNT(*) AS n FROM jacobs_pipelines WHERE parent_pipeline_id = %s", (padre,))
    return int(fila["n"])


def test_fallo_de_la_relectura_tras_el_update_deja_evento_y_no_crea_el_hijo():
    """Residual de I-2 (R13): el UPDATE del consumo ya se confirmó (token
    quemado) y la relectura SQL_TOKEN_CONSUMIDO falla con un error REAL de la
    base. Tiene que quedar SUBPIPELINE_RECHAZADO fase=consumo sin el token, el
    hijo no se crea y el error original sube al llamador."""
    import pymysql
    from unittest.mock import patch

    async def escenario():
        padre, paso = await ada.padre_en_ejecucion()
        try:
            token = await ada.emitir(padre, paso)
            relectura_rota = "SELECT * FROM tabla_inexistente_arnes_i2 WHERE token_hash = %s"
            with patch.object(store, "SQL_TOKEN_CONSUMIDO", relectura_rota), \
                 pytest.raises(pymysql.err.ProgrammingError):
                await ada.pedir_hijo(token, padre)
            fila = await ada.fila_token(sp.hash_token(token))
            evento = await _evento_quemado(padre, "consumo")
            return token, fila, evento, await _hijos_de(padre)
        finally:
            await ada.cerrar(padre)

    token, fila, evento, hijos = _correr(escenario)
    assert fila["usado_at"] is not None, "el arnés no quemó el token: el caso no se ejercitó"
    assert evento is not None, "token quemado en la relectura sin SUBPIPELINE_RECHAZADO"
    payload = json.loads(evento["payload"])
    assert payload["motivo"] == sp.Motivo.CREACION_FALLIDA.value
    assert payload["excepcion"] == "ProgrammingError"
    assert payload["token_ref"] == sp.token_ref(sp.hash_token(token))
    assert token not in evento["payload"]
    assert "tabla_inexistente" not in evento["payload"]
    assert evento["pipeline_id"] == fila["hijo_pipeline_id"]
    assert hijos == 0


def test_cancelacion_tras_quemar_el_token_deja_evento_y_se_relanza():
    """CancelledError es BaseException: un `except Exception` no la ve. Llega
    (a) entre el UPDATE confirmado y la relectura y (b) durante la creación.
    En los dos casos: evento best-effort, hijo no creado, cancelación relanzada."""
    from unittest.mock import AsyncMock, patch

    consumir_real = store.subpipeline_token_consumir

    async def consumir_y_cancelar(*args, **kwargs):
        await consumir_real(*args, **kwargs)  # UPDATE confirmado: token quemado
        raise asyncio.CancelledError()

    async def escenario():
        padre_a, paso_a = await ada.padre_en_ejecucion()
        padre_b, paso_b = await ada.padre_en_ejecucion()
        try:
            token_a = await ada.emitir(padre_a, paso_a)
            with patch.object(store, "subpipeline_token_consumir", consumir_y_cancelar), \
                 pytest.raises(asyncio.CancelledError):
                await ada.pedir_hijo(token_a, padre_a)
            token_b = await ada.emitir(padre_b, paso_b)
            with patch.object(store, "pipeline_create",
                              AsyncMock(side_effect=asyncio.CancelledError())), \
                 pytest.raises(asyncio.CancelledError):
                await ada.pedir_hijo(token_b, padre_b)
            return [
                (token_a, await ada.fila_token(sp.hash_token(token_a)),
                 await _evento_quemado(padre_a, "consumo"), await _hijos_de(padre_a)),
                (token_b, await ada.fila_token(sp.hash_token(token_b)),
                 await _evento_quemado(padre_b, "creacion"), await _hijos_de(padre_b)),
            ]
        finally:
            await ada.cerrar(padre_a)
            await ada.cerrar(padre_b)

    for token, fila, evento, hijos in _correr(escenario):
        assert fila["usado_at"] is not None
        assert evento is not None, "cancelación tras quemar el token sin SUBPIPELINE_RECHAZADO"
        payload = json.loads(evento["payload"])
        assert payload["excepcion"] == "CancelledError"
        assert token not in evento["payload"]
        assert hijos == 0
