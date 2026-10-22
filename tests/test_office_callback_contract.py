"""Classic HTTP/CLI gates must not expose an Office row, even before its step exists."""
from __future__ import annotations

import asyncio
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient

from jacobs import cupo, routes, store
from jacobs.office_contract import OFFICE_WORK_CONTRACT_VERSION, OfficeRunRequest, office_request_digest


def _request() -> OfficeRunRequest:
    data = {
        "contract_version": OFFICE_WORK_CONTRACT_VERSION,
        "office_job_id": uuid.uuid4(), "office_stage_id": uuid.uuid4(),
        "client_request_id": uuid.uuid4(), "project_id": 701, "tenant_id": 702,
        "usage_user_id": 703, "facet": "hipatia", "capability": "design",
        "timeout_seconds": 60, "accepted_cost_usd": "0.010000",
    }
    data["request_sha256"] = office_request_digest(data)
    return OfficeRunRequest(**data)


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(routes.router)
    return app


def test_classic_routes_hide_office_row_without_step_and_leave_no_events():
    async def reserve():
        await store.init_tables()
        result = await cupo.reservar_office_v1(_request(), limite=10000)
        await store.cerrar_pool()
        return result.pipeline_id

    async def state(pid):
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT office_phase,status FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))
                pipeline = await cur.fetchone()
                await cur.execute("SELECT COUNT(*) FROM jacobs_steps WHERE pipeline_id=%s", (pid,)); steps = (await cur.fetchone())[0]
                await cur.execute("SELECT COUNT(*) FROM jacobs_events WHERE pipeline_id=%s", (pid,)); events = (await cur.fetchone())[0]
        await store.cerrar_pool()
        return pipeline, steps, events

    pid = asyncio.run(reserve())
    before = asyncio.run(state(pid))
    try:
        requests = [
            ("get", f"/jacobs/pipeline/{pid}", None),
            ("get", f"/jacobs/pipeline/{pid}/results", None),
            ("get", f"/jacobs/pipeline/{pid}/events", None),
            ("post", f"/jacobs/pipeline/{pid}/cancel", None),
            ("post", f"/jacobs/pipeline/{pid}/discard", {"user_id": "u"}),
            ("post", f"/jacobs/pipeline/{pid}/recover", {"user_id": "u"}),
            ("post", f"/jacobs/pipeline/{pid}/hide", {"user_id": "u"}),
            ("post", f"/jacobs/pipeline/{pid}/restore", {"user_id": "u"}),
            ("post", f"/jacobs/pipeline/{pid}/resume", {"invoked_by": "plataforma"}),
            ("post", f"/jacobs/pipeline/{pid}/continue", {"invoked_by": "plataforma"}),
            ("post", f"/jacobs/pipeline/{pid}/continue/preflight", {"invoked_by": "plataforma"}),
            ("post", f"/jacobs/pipeline/{pid}/approve-step", {"invoked_by": "plataforma"}),
        ]
        with TestClient(_app(), raise_server_exceptions=False) as client:
            for method, path, body in requests:
                response = client.request(method.upper(), path, json=body)
                assert response.status_code == 404, (method, path, response.text)
        assert asyncio.run(state(pid)) == before
    finally:
        async def clean():
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("DELETE FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))
            await store.cerrar_pool()
        asyncio.run(clean())
