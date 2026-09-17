"""Arnés que hace de Ada contra Jacobs, para las pruebas del contrato de
sub-pipelines (frente F, 2026-09-16).

Escribe pipelines, pasos, tokens y eventos: SOLO contra jax_memory_test. Si el
proceso ya trae JAX_DB_NAME apuntando a otra base (típico después de sourcear
/etc/jax/.env), se niega a importar en vez de escribir ahí en silencio.

Lo único que sustituye es lo que no es el contrato: el planificador (llama a un
LLM) y el conteo de pipelines activos (el límite de 3 no es lo que se prueba).
El token, la base y los eventos son los reales.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import os

_nombre = os.environ.get("JAX_DB_NAME")
if _nombre is not None and _nombre != "jax_memory_test":
    raise RuntimeError(
        f"JAX_DB_NAME={_nombre!r}: el arnés de Ada escribe pipelines, pasos, tokens y "
        "eventos. Solo corre contra jax_memory_test (exportala después de sourcear "
        "/etc/jax/.env)."
    )
os.environ["JAX_DB_NAME"] = "jax_memory_test"

import time  # noqa: E402
import uuid  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402

import aiomysql  # noqa: E402
from fastapi import BackgroundTasks  # noqa: E402

from jacobs import routes, store  # noqa: E402
from jacobs.models import (  # noqa: E402
    Pipeline,
    PipelineCreateRequest,
    PipelineStatus,
    Step,
    StepStatus,
)


async def plan_de_un_paso(pipeline_id, objective, max_steps, steps_spec):
    return [Step(facet="jekyll", capability="summarize", pipeline_id=pipeline_id)]


async def padre_en_ejecucion(
    depth: int = 0,
    facet_del_paso: str = "ada",
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> tuple[str, str]:
    """Un pipeline `running` con un paso de Ada (`running` al crearse). Lo que habilita a Ada
    a pedir un hijo es que el PIPELINE padre esté `running`; el paso que delegó puede estar
    en cualquier estado (enmienda de Fernando 2026-09-16: en el modo "plan de delegación"
    los tokens se emiten después de que el step `delegate` terminó)."""
    pid = str(uuid.uuid4())
    ahora = time.time()
    paso = Step(
        pipeline_id=pid, step_index=0, facet=facet_del_paso,
        capability="delegate", status=StepStatus.running,
    )
    await store.pipeline_create(Pipeline(
        pipeline_id=pid, name="arnes-ada-padre", invoked_by="plataforma",
        mode="autonomous", status=PipelineStatus.running, plan=[paso],
        depth=depth, user_id=user_id, tenant_id=tenant_id,
        created_at=ahora, updated_at=ahora,
    ))
    await store.step_upsert(paso)
    return pid, paso.step_id


async def cerrar(pipeline_id: str) -> None:
    await store.pipeline_update_status(pipeline_id, PipelineStatus.completed)


async def una_fila(sql: str, args: tuple = ()) -> dict | None:
    async with store.conexion() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(sql, args)
            return await cur.fetchone()


async def ejecutar(sql: str, args: tuple = ()) -> int:
    async with store.conexion() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, args)
            return cur.rowcount


async def fila_token(token_hash: str) -> dict | None:
    return await una_fila(
        "SELECT * FROM jacobs_subpipeline_tokens WHERE token_hash = %s", (token_hash,)
    )


async def emitir(parent_pipeline_id: str, parent_step: str) -> str:
    from jacobs import subpipelines

    with patch("jacobs.policy.check_kill_switch", return_value=False):
        return await subpipelines.emitir_token_subpipeline(parent_pipeline_id, parent_step)


async def pedir_hijo(
    token: str,
    parent_pipeline_id: str,
    *,
    cuerpo_extra: dict | None = None,
    kill_switch: bool = False,
    plan=plan_de_un_paso,
) -> dict:
    """Lo que haría Ada: POST /jacobs/pipeline con su token. El cuerpo pasa por
    `model_validate` igual que el JSON de un pedido HTTP real."""
    cuerpo = {
        "name": "arnes-ada-hijo", "objective": "o", "invoked_by": "ada",
        "mode": "dry_run", "subpipeline_token": token,
        "parent_pipeline_id": parent_pipeline_id,
    }
    cuerpo.update(cuerpo_extra or {})
    req = PipelineCreateRequest.model_validate(cuerpo)
    with patch.object(routes, "_build_plan_or_reject", AsyncMock(side_effect=plan)), \
         patch.object(store, "pipeline_count_active", AsyncMock(return_value=1)), \
         patch("jacobs.policy.check_kill_switch", return_value=kill_switch):
        return await routes.create_pipeline(req, BackgroundTasks())
