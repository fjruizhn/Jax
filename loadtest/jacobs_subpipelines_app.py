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

from fastapi import FastAPI  # noqa: E402

from jacobs import _arnes_ada, routes, store  # noqa: E402

routes._build_plan_or_reject = _arnes_ada.plan_de_un_paso

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
