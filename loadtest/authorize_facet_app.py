"""App de carga de POST /motor/authorize-facet (2026-09-17, pool de Jacobs).

NUNCA contra producción: se niega a arrancar sin JAX_DB_NAME=jax_memory_test.
Solo LEE `facet.allowed_callers`, pero la barrera es la misma que la de
jacobs_subpipelines_app.py: una app de carga no elige base por accidente.

Monta el router REAL de Motor Registry (motor_registry.routes): lo que se mide
es el endpoint que consulta jax-platform antes de cada turno de la Mesa web a un
facet HTTP, con su conexión a la base. No arranca el catálogo de motores ni los
workers: /authorize-facet no los usa.

USO:
  cd las_manos && PYTHONPATH=.:.. uvicorn loadtest.authorize_facet_app:app --port 7798
"""
from __future__ import annotations

import os

if os.environ.get("JAX_DB_NAME") != "jax_memory_test":
    raise RuntimeError(
        "authorize_facet_app solo corre con JAX_DB_NAME=jax_memory_test."
    )

from fastapi import FastAPI  # noqa: E402

from jacobs import store  # noqa: E402
from motor_registry.routes import router  # noqa: E402

app = FastAPI(title="carga-authorize-facet")
app.include_router(router)


@app.on_event("shutdown")
async def _cerrar() -> None:
    await store.cerrar_pool()
