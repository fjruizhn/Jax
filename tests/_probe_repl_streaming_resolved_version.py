#!/usr/bin/env python3
"""
Probe manual — Bloque D (D1.2), corrida futura: verifica que
HttpMuscle._call_openai (REPL, jax/muscles/base.py) extrae resolved_version
del stream SSE real (un campo por chunk) y dispara la misma deteccion de
drift que ya se probo end-to-end para Jacobs
(_probe_resolved_version_drift.py). Es el codigo mas nuevo/riesgoso de este
wiring (parseo de streaming), por eso se prueba aparte.

Corre contra jax_memory_test. httpx se fake-ea (el .stream() de
AsyncClient), CERO llamada de red real a OpenAI — la credencial real de
jax_memory_test se resuelve de verdad, pero nunca sale por la red.

Uso: JAX_DB_NAME=jax_memory_test .venv/bin/python tests/_probe_repl_streaming_resolved_version.py
(desde la raiz del repo jax; requiere /etc/jax/.env ya sourceado)
"""
import asyncio
import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import aiomysql
import httpx

from jax.muscles.base import HttpMuscle

FACET = "thot"


class _FakeStreamResponse:
    def __init__(self, model: str, text: str):
        self.status_code = 200
        self._model = model
        self._text = text

    async def aiter_lines(self):
        chunk = {"model": self._model, "choices": [{"delta": {"content": self._text}}]}
        yield f"data: {json.dumps(chunk)}"
        yield "data: [DONE]"

    async def aread(self):
        return b""


class _FakeStreamCtx:
    def __init__(self, model: str, text: str):
        self._resp = _FakeStreamResponse(model, text)

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


async def _db_conn():
    return await aiomysql.connect(
        host=os.getenv("JAX_DB_HOST", "localhost"),
        port=int(os.getenv("JAX_DB_PORT", "3306")),
        user=os.getenv("JAX_DB_USER", ""),
        password=os.getenv("JAX_DB_PASSWORD", ""),
        db=os.getenv("JAX_DB_NAME", "jax_memory"),
        charset="utf8mb4",
        autocommit=True,
    )


async def _reset_baseline(conn):
    async with conn.cursor() as cur:
        await cur.execute(
            "UPDATE facet_binding SET resolved_version=NULL, resolved_version_checked_at=NULL "
            "WHERE facet_key=%s AND role='primary'",
            (FACET,),
        )
        await cur.execute(
            "DELETE FROM model_binding_proposal WHERE facet_key=%s AND reason='drift_detected' "
            "AND detail LIKE 'PROBE:%%'",
            (FACET,),
        )
        await conn.commit()


async def _fetch_state(conn):
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT resolved_version FROM facet_binding WHERE facet_key=%s AND role='primary'",
            (FACET,),
        )
        (resolved,) = await cur.fetchone()
        await cur.execute(
            "SELECT id, reason, status FROM model_binding_proposal "
            "WHERE facet_key=%s AND reason='drift_detected' AND status='pending' "
            "ORDER BY id DESC LIMIT 1",
            (FACET,),
        )
        proposal = await cur.fetchone()
    return resolved, proposal


async def main():
    if os.getenv("JAX_DB_NAME") != "jax_memory_test":
        print("ABORTA: JAX_DB_NAME debe ser jax_memory_test para este probe.")
        sys.exit(1)

    conn = await _db_conn()
    try:
        await _reset_baseline(conn)

        muscle = HttpMuscle(
            name=FACET, provider="openai", model_default="gpt-5.5",
            models_allowed=["gpt-5.5"], system_prompt="probe", timeout=30,
            authority_origin="", api_url="",
        )

        with patch.object(httpx.AsyncClient, "stream", lambda self, method, url, **kw: _FakeStreamCtx("gpt-5.5", "hola")):
            texto1 = await muscle._call_openai("hola", "gpt-5.5", history=None)
        resolved1, proposal1 = await _fetch_state(conn)
        print(f"PASO 1 (baseline, stream SSE real parseado): texto={texto1!r} resolved_version={resolved1!r} proposal={proposal1}")
        assert resolved1 == "gpt-5.5", f"no extrajo 'model' del chunk SSE: {resolved1!r}"
        assert proposal1 is None, "no debia haber drift en la primera observacion"

        with patch.object(httpx.AsyncClient, "stream", lambda self, method, url, **kw: _FakeStreamCtx("gpt-5.5-DRIFTED-BY-PROBE", "hola de nuevo")):
            texto2 = await muscle._call_openai("hola de nuevo", "gpt-5.5", history=None)
        resolved2, proposal2 = await _fetch_state(conn)
        print(f"PASO 2 (drift simulado): texto={texto2!r} resolved_version={resolved2!r} proposal={proposal2}")
        assert resolved2 == "gpt-5.5-DRIFTED-BY-PROBE"
        assert proposal2 is not None, "NO se creo proposal drift_detected — el wiring de streaming NO disparo"

        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE model_binding_proposal SET detail=CONCAT('PROBE: ', detail) WHERE id=%s",
                (proposal2[0],),
            )
            await conn.commit()

        print(f"\n✅ STREAMING SSE (REPL) DISPARA DRIFT END-TO-END: proposal id={proposal2[0]} status={proposal2[2]}")

        await _reset_baseline(conn)
        resolved_final, _ = await _fetch_state(conn)
        print(f"Estado final restaurado: resolved_version={resolved_final!r}")
    finally:
        conn.close()


if __name__ == "__main__":
    asyncio.run(main())
