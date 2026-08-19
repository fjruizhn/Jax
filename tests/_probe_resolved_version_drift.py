#!/usr/bin/env python3
"""
Probe manual — Bloque D (D1.2), corrida futura: demuestra que la deteccion
de drift DISPARA end-to-end a traves del wiring real de Jacobs
(_invoke_http_openai_compat), no solo que record_resolved_version() aislada
funciona (eso ya esta cubierto en jax-platform).

Corre contra jax_memory_test (conftest de jax-platform ya la sembro con el
esquema de Bloque D) — nunca toca jax_memory de produccion. httpx se
fake-ea (unittest.mock), CERO llamada de red real a DeepSeek: la credencial
real de jax_memory_test se resuelve (resolve_facet real, sin fake), pero
nunca sale por la red.

Uso: JAX_DB_NAME=jax_memory_test las_manos/.venv/bin/python tests/_probe_resolved_version_drift.py
(desde la raiz del repo jax; requiere /etc/jax/.env ya sourceado para
JAX_DB_HOST/USER/PASSWORD/FERNET_KEY)
"""
import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "las_manos"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import aiomysql
import httpx

from facet_resolver import resolve_facet
from jacobs.executor import _invoke_http_openai_compat

FACET = "jekyll"


class _FakeResponse:
    def __init__(self, model: str):
        self.status_code = 200
        self._model = model

    def json(self):
        return {
            "model": self._model,
            "choices": [{"message": {"content": f"hola, corro {self._model}"}}],
        }


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


async def _dump_state(conn, label):
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT provider_id, model_id, resolved_version FROM facet_binding "
            "WHERE facet_key=%s AND role='primary'",
            (FACET,),
        )
        binding = await cur.fetchone()
        await cur.execute(
            "SELECT id, reason, detail, status FROM model_binding_proposal "
            "WHERE facet_key=%s ORDER BY id DESC LIMIT 3",
            (FACET,),
        )
        proposals = await cur.fetchall()
    print(f"\n--- {label} ---")
    print(f"facet_binding[{FACET}]: provider_id={binding[0]} model_id={binding[1]} resolved_version={binding[2]!r}")
    print(f"model_binding_proposal (ultimas 3): {proposals}")
    return binding, proposals


async def main():
    if os.getenv("JAX_DB_NAME") != "jax_memory_test":
        print("ABORTA: JAX_DB_NAME debe ser jax_memory_test para este probe (no toca produccion).")
        sys.exit(1)

    conn = await _db_conn()
    try:
        await _reset_baseline(conn)
        await _dump_state(conn, "ESTADO INICIAL (reseteado por el probe)")

        f = await resolve_facet(FACET)  # credencial REAL de jax_memory_test, resuelta de verdad
        print(f"\nresolve_facet('{FACET}') -> provider_id={f.provider_id} model={f.model} transport={f.transport}")

        fake_post = AsyncMock(return_value=_FakeResponse(f.model))  # PASO 1: mismo modelo pedido -> baseline, sin drift
        with patch.object(httpx.AsyncClient, "post", fake_post):
            result1 = await _invoke_http_openai_compat(f, "hola", timeout=30)
        print(f"\nPASO 1 (baseline, resolved_version == model pedido): result={result1['result']!r}")
        await _dump_state(conn, "TRAS PASO 1 (debe seguir sin proposal)")

        drifted_model = f"{f.model}-DRIFTED-BY-PROBE"  # PASO 2: la API 'confirma' un modelo DISTINTO, sin tocar el binding
        fake_post2 = AsyncMock(return_value=_FakeResponse(drifted_model))
        with patch.object(httpx.AsyncClient, "post", fake_post2):
            result2 = await _invoke_http_openai_compat(f, "hola de nuevo", timeout=30)
        print(f"\nPASO 2 (drift simulado, proveedor 'confirma' {drifted_model!r}): result={result2['result']!r}")
        binding_after, proposals_after = await _dump_state(conn, "TRAS PASO 2 (debe existir 1 proposal drift_detected pendiente)")

        assert binding_after[2] == drifted_model, f"resolved_version no se actualizo: {binding_after}"
        fresh = [p for p in proposals_after if p[1] == "drift_detected" and p[3] == "pending"]
        assert fresh, "NO se creo ninguna proposal drift_detected pendiente — el wiring NO disparo"
        # marcar la proposal de este probe para que _reset_baseline la limpie en la proxima corrida
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE model_binding_proposal SET detail=CONCAT('PROBE: ', detail) WHERE id=%s",
                (fresh[0][0],),
            )
            await conn.commit()

        print(f"\n✅ DRIFT DISPARADO END-TO-END: proposal id={fresh[0][0]} reason={fresh[0][1]} status={fresh[0][3]}")
        print(f"   detail original: {fresh[0][2]}")

        await _reset_baseline(conn)  # deja jax_memory_test limpio para la proxima corrida
        await _dump_state(conn, "ESTADO FINAL (limpiado por el probe)")
    finally:
        conn.close()


if __name__ == "__main__":
    asyncio.run(main())
