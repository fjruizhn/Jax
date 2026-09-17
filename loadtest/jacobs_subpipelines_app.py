"""App de carga del contrato de sub-pipelines (frente F, 2026-09-16).

NUNCA contra producción: escribe pipelines, pasos, tokens y eventos, así que se
niega a arrancar sin JAX_DB_NAME=jax_memory_test.

Monta las rutas REALES de Jacobs. Lo único sustituido es `_build_plan_or_reject`
por el plan fijo de un paso del arnés: build() no cambió en este frente y en
el camino real llama a un LLM. Lo que se mide es lo que sí cambió: candado de
creación, conteo, validate_create, consumo atómico del token, inserts y eventos.

USO (ver la Task 8 del plan 2026-09-16-frente-f-contrato-subpipelines.md):
  PYTHONPATH=.:las_manos uvicorn loadtest.jacobs_subpipelines_app:app --port 7799
"""
from __future__ import annotations

import os

if os.environ.get("JAX_DB_NAME") != "jax_memory_test":
    raise RuntimeError(
        "jacobs_subpipelines_app solo corre con JAX_DB_NAME=jax_memory_test: "
        "escribe pipelines, tokens y eventos."
    )

import asyncio  # noqa: E402

from fastapi import FastAPI  # noqa: E402

from jacobs import _arnes_ada, routes, store  # noqa: E402

# JAX_CARGA_PLAN_MS (2026-09-17, medición del candado de creación): el
# planificador REAL tarda 1,3-8,7 s en el camino sano y hasta 20-40 s con el
# LLM cargado. Con el plan instantáneo del arnés, la sección crítica del
# candado viejo duraba microsegundos y el cuello quedaba SUBESTIMADO: lo que
# ponía a la Mesa a esperar detrás de Ada era justamente planificar adentro del
# candado. Este retardo lo representa. En 0 (el defecto) se comporta como antes.
_PLAN_MS = float(os.environ.get("JAX_CARGA_PLAN_MS", "0"))


async def _plan_con_retardo(pipeline_id, objective, max_steps, steps_spec):
    if _PLAN_MS:
        await asyncio.sleep(_PLAN_MS / 1000.0)
    return await _arnes_ada.plan_de_un_paso(pipeline_id, objective, max_steps, steps_spec)


routes._build_plan_or_reject = _plan_con_retardo

app = FastAPI(title="carga-contrato-subpipelines")
app.include_router(routes.router)

# Mismo middleware que LAS MANOS (auth_servicio.proteger): lee las credenciales
# JAX_LAS_MANOS_CREDENCIAL_* del entorno y falla cerrado si faltan. k6 manda
# la de jacobs en __ENV.CREDENCIAL.
from auth_servicio import proteger  # noqa: E402

proteger(app)


@app.on_event("startup")
async def _init() -> None:
    store.tamanio_pool()  # mismo orden que LAS MANOS: validar antes de conectar
    await store.init_tables()


@app.on_event("shutdown")
async def _cerrar() -> None:
    await store.cerrar_pool()
