"""Real MariaDB checks for opaque Office V1 storage (never production)."""
from __future__ import annotations

import asyncio
import importlib.util
import os
import uuid
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import aiomysql

from jacobs import cupo, store
from jacobs.models import PipelineStatus
from jacobs.office_contract import (
    OFFICE_WORK_CONTRACT_VERSION, OfficePhase, OfficeReceipt, OfficeRunRequest, canonical_usd,
    deterministic_step_id, office_request_digest,
)


pytestmark = pytest.mark.skipif(not os.getenv("JAX_DB_HOST"), reason="requires disposable MariaDB")


def _relaunch_cli_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "jacobs_relaunch.py"
    spec = importlib.util.spec_from_file_location("jacobs_relaunch_office_db", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_office_check_normalization_preserves_grouping_and_enum_literal_case():
    canonical = store._OFFICE_BINDING_CHECK_INFORMATION_SCHEMA
    regrouped = canonical.replace(
        " and (cast(`execution_contract` as char charset binary) = 'CLASSIC'",
        " and cast(`execution_contract` as char charset binary) = 'CLASSIC'",
        1,
    ).replace("`costo_max_aceptado_usd` is null),0)", "`costo_max_aceptado_usd` is null)),0)", 1)
    lower_enum = canonical.replace("'OFFICE_V1'", "'office_v1'", 1)
    assert store._normalizar_check_clause(regrouped) != store._normalizar_check_clause(canonical)
    assert store._normalizar_check_clause(lower_enum) != store._normalizar_check_clause(canonical)


def _request(**changes):
    data = {"contract_version": OFFICE_WORK_CONTRACT_VERSION, "office_job_id": uuid.uuid4(),
            "office_stage_id": uuid.uuid4(), "client_request_id": uuid.uuid4(), "project_id": 701,
            "tenant_id": 702, "usage_user_id": 703, "facet": "hipatia", "capability": "design",
            "timeout_seconds": 60, "accepted_cost_usd": Decimal("0.010000")}
    data.update(changes)
    data["request_sha256"] = office_request_digest(data)
    return OfficeRunRequest(**data)


async def _clean(ids):
    if not ids:
        return
    async with store.conexion() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM jacobs_pipelines WHERE pipeline_id IN (" + ",".join(["%s"] * len(ids)) + ")", ids)


def test_office_reservation_replay_conflict_and_epoch_cas_are_real_db():
    async def run():
        await store.init_tables()
        request = _request()
        results = await asyncio.gather(*[cupo.reservar_office_v1(request, limite=10000) for _ in range(2)])
        assert sum(item.created for item in results) == 1
        assert len({item.pipeline_id for item in results}) == 1
        pid = results[0].pipeline_id
        try:
            replay = await cupo.reservar_office_v1(request, limite=10000)
            assert replay and not replay.created and not replay.conflict and replay.pipeline_id == pid
            changed = _request(client_request_id=request.client_request_id)
            assert (await cupo.reservar_office_v1(changed, limite=10000)).conflict
            stage_conflict = _request(office_stage_id=request.office_stage_id)
            assert (await cupo.reservar_office_v1(stage_conflict, limite=10000)).conflict
            snap = await store.office_pipeline_by_client_request(request.tenant_id, str(request.client_request_id))
            assert snap and snap.phase == OfficePhase.RESERVED
            assert await store.office_prepare_step(pid, request)
            assert await store.office_claim_preflight(pid, 0, deterministic_step_id(pid))
            with pytest.raises(ValueError):
                await store.office_transition_phase(pid, 0, OfficePhase.RESERVED, OfficePhase.READY)
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("EXPLAIN SELECT pipeline_id FROM jacobs_pipelines WHERE invoked_by='plataforma' AND tenant_id=%s AND client_request_id=%s AND execution_contract='OFFICE_V1'", (request.tenant_id, str(request.client_request_id)))
                    plan = await cur.fetchone()
            assert plan[3] != "ALL"
        finally:
            await _clean([pid])
            await store.cerrar_pool()
    asyncio.run(run())


def test_office_opaque_delivery_store_is_atomic_and_only_one_claim_wins():
    """Catches a callback path persisting content, splitting receipt/event, or double-delivering."""
    async def run():
        await store.init_tables()
        request = _request()
        reserved = await cupo.reservar_office_v1(request, limite=10000)
        assert reserved and reserved.created
        pid, step_id = reserved.pipeline_id, deterministic_step_id(reserved.pipeline_id)
        try:
            assert await store.office_prepare_step(pid, request)
            assert await store.office_prepare_step(pid, request)  # exact replay only
            claims = await asyncio.gather(*[store.office_claim_preflight(pid, 0, step_id) for _ in range(2)])
            assert claims.count(True) == 1
            assert await store.office_mark_ready(pid, 0, step_id, "b" * 64, 1)
            delivery = await asyncio.gather(*[store.office_claim_delivery(pid, 0, step_id) for _ in range(2)])
            assert delivery.count(True) == 1
            receipt = OfficeReceipt(receipt_id=str(uuid.uuid4()), effect_kind="NOTE_CREATED",
                                    effect_id="note:1", output_sha256="c" * 64,
                                    office_stage_version=1)
            assert await store.office_store_receipt(pid, 0, step_id, receipt, "gemini-2.5-pro")
            snapshot = await store.office_execution_snapshot(pid)
            assert snapshot and snapshot.step_id == step_id and snapshot.correlation_id
            assert snapshot.materialization_sha256 == "b" * 64 and snapshot.receipt == receipt
            with pytest.raises(ValueError, match="terminal phase has no exit"):
                await store.office_finish_terminal(pid, 0, step_id, OfficePhase.RECEIPT_STORED,
                                                   "OFFICE_EXECUTION_FAILED", OfficePhase.FAILED)
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT office_phase,status FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))
                    assert await cur.fetchone() == ("RECEIPT_STORED", "completed")
                    await cur.execute("SELECT input_ref,output_ref,status FROM jacobs_steps WHERE step_id=%s", (step_id,))
                    input_ref, output_ref, status = await cur.fetchone()
                    assert "office_materialization_sha256" in input_ref and "prompt" not in input_ref
                    assert output_ref.startswith("inline:") and receipt.receipt_id in output_ref and status == "completed"
        finally:
            await _clean([pid])
            await store.cerrar_pool()
    asyncio.run(run())


def test_office_binding_check_expand_contract_recovers_old_and_both_states():
    """Catches dropping the legacy CHECK before a verified v2 replacement exists."""
    async def names():
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT CONSTRAINT_NAME FROM information_schema.CHECK_CONSTRAINTS WHERE CONSTRAINT_SCHEMA=DATABASE() AND CONSTRAINT_NAME IN ('chk_jacobs_office_binding','chk_jacobs_office_binding_v2')")
                return {row[0] for row in await cur.fetchall()}
    async def run():
        await store.init_tables()
        try:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("ALTER TABLE jacobs_pipelines DROP CONSTRAINT chk_jacobs_office_binding_v2")
                    await cur.execute("ALTER TABLE jacobs_pipelines ADD CONSTRAINT chk_jacobs_office_binding CHECK (" + store._OFFICE_BINDING_CHECK_V1_CLAUSE + ")")
            await store.init_tables()
            assert await names() == {"chk_jacobs_office_binding_v2"}
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("ALTER TABLE jacobs_pipelines DROP CONSTRAINT chk_jacobs_office_binding_v2")
                    await cur.execute("ALTER TABLE jacobs_pipelines ADD CONSTRAINT chk_jacobs_office_binding CHECK (" + store._OFFICE_BINDING_CHECK_V1_CLAUSE + ")")
            await store.init_tables()
            assert await names() == {"chk_jacobs_office_binding_v2"}
        finally:
            await store.init_tables()
            await store.cerrar_pool()
    asyncio.run(run())


def test_office_binding_check_fresh_neither_state_installs_v2_exactly():
    """A fresh schema with neither historical name repairs to the sole v2 gate."""
    async def names():
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT CONSTRAINT_NAME FROM information_schema.CHECK_CONSTRAINTS WHERE CONSTRAINT_SCHEMA=DATABASE() AND CONSTRAINT_NAME IN ('chk_jacobs_office_binding','chk_jacobs_office_binding_v2')")
                return {row[0] for row in await cur.fetchall()}

    async def run():
        await store.init_tables()
        try:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT COUNT(*) FROM jacobs_pipelines WHERE execution_contract='OFFICE_V1'")
                    assert (await cur.fetchone())[0] == 0
                    await cur.execute("ALTER TABLE jacobs_pipelines DROP CONSTRAINT chk_jacobs_office_binding_v2")
            assert await names() == set()
            await store.init_tables()
            assert await names() == {"chk_jacobs_office_binding_v2"}
        finally:
            await store.init_tables()
            await store.cerrar_pool()
    asyncio.run(run())


def test_office_binding_check_invalid_existing_row_fails_v2_add_and_preserves_old_gate(monkeypatch):
    """The exact v2 ALTER rejects an invalid old row without contracting v1."""
    async def names():
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT CONSTRAINT_NAME FROM information_schema.CHECK_CONSTRAINTS WHERE CONSTRAINT_SCHEMA=DATABASE() AND CONSTRAINT_NAME IN ('chk_jacobs_office_binding','chk_jacobs_office_binding_v2')")
                return {row[0] for row in await cur.fetchall()}

    async def run():
        await store.init_tables()
        reservation = await cupo.reservar_office_v1(_request(), limite=10000)
        assert reservation and reservation.created
        pid = reservation.pipeline_id
        attempted = []
        real_ddl = store._ddl_acotado

        async def trace_v2_add(cur, ddl):
            if "ADD CONSTRAINT chk_jacobs_office_binding_v2" in ddl:
                attempted.append(ddl)
            return await real_ddl(cur, ddl)

        try:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("ALTER TABLE jacobs_pipelines DROP CONSTRAINT chk_jacobs_office_binding_v2")
                    await cur.execute("ALTER TABLE jacobs_pipelines ADD CONSTRAINT chk_jacobs_office_binding CHECK (" + store._OFFICE_BINDING_CHECK_V1_CLAUSE + ")")
                    # This reproduces pre-existing corrupt data; it is not a production
                    # write path.  MariaDB must reject the v2 ADD against this row.
                    await cur.execute("SET SESSION check_constraint_checks=OFF")
                    await cur.execute("UPDATE jacobs_pipelines SET office_phase='FAILED',status='failed',office_rejection_code='ARBITRARY_UNREGISTERED_CODE' WHERE pipeline_id=%s", (pid,))
                    await cur.execute("SET SESSION check_constraint_checks=ON")
            monkeypatch.setattr(store, "_ddl_acotado", trace_v2_add)
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    with pytest.raises(aiomysql.Error) as caught:
                        await store._migrar_check_office_v2(cur)
            assert len(attempted) == 1
            assert "ADD CONSTRAINT chk_jacobs_office_binding_v2" in attempted[0]
            # MariaDB 12.3.3 reports the already-installed v1 constraint for
            # a row invalid under both clauses, but `attempted` above proves
            # this 4025 came from the real v2 ADD, not an earlier startup DDL.
            assert caught.value.args[0] == 4025
            assert "chk_jacobs_office_binding" in str(caught.value)
            assert await names() == {"chk_jacobs_office_binding"}
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT office_rejection_code FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))
                    assert (await cur.fetchone())[0] == "ARBITRARY_UNREGISTERED_CODE"
        finally:
            monkeypatch.setattr(store, "_ddl_acotado", real_ddl)
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SET SESSION check_constraint_checks=OFF")
                    await cur.execute("UPDATE jacobs_pipelines SET office_rejection_code='OFFICE_EXECUTION_FAILED' WHERE pipeline_id=%s", (pid,))
                    await cur.execute("SET SESSION check_constraint_checks=ON")
            await store.init_tables()
            await _clean([pid])
            await store.cerrar_pool()
    asyncio.run(run())


def test_office_binding_check_add_timeout_after_real_add_resumes_without_gap(monkeypatch):
    """An ambiguous ADD result leaves the old CHECK until a retry verifies v2."""
    async def names():
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT CONSTRAINT_NAME FROM information_schema.CHECK_CONSTRAINTS WHERE CONSTRAINT_SCHEMA=DATABASE() AND CONSTRAINT_NAME IN ('chk_jacobs_office_binding','chk_jacobs_office_binding_v2')")
                return {row[0] for row in await cur.fetchall()}
    async def run():
        await store.init_tables()
        real = store._ddl_acotado
        try:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("ALTER TABLE jacobs_pipelines DROP CONSTRAINT chk_jacobs_office_binding_v2")
                    await cur.execute("ALTER TABLE jacobs_pipelines ADD CONSTRAINT chk_jacobs_office_binding CHECK (" + store._OFFICE_BINDING_CHECK_V1_CLAUSE + ")")
            async def add_then_timeout(cur, ddl):
                if "ADD CONSTRAINT chk_jacobs_office_binding_v2" in ddl:
                    assert await real(cur, ddl)
                    return False
                return await real(cur, ddl)
            monkeypatch.setattr(store, "_ddl_acotado", add_then_timeout)
            with pytest.raises(RuntimeError, match="timed out adding Office binding v2 constraint"):
                await store.init_tables()
            assert await names() == {"chk_jacobs_office_binding", "chk_jacobs_office_binding_v2"}
            monkeypatch.setattr(store, "_ddl_acotado", real)
            await store.init_tables()
            assert await names() == {"chk_jacobs_office_binding_v2"}
        finally:
            monkeypatch.setattr(store, "_ddl_acotado", real)
            await store.init_tables()
            await store.cerrar_pool()
    asyncio.run(run())


def test_office_binding_check_allows_registered_input_stale_but_rejects_raw_code():
    async def run():
        await store.init_tables(); request = _request(); reservation = await cupo.reservar_office_v1(request, limite=10000)
        try:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("UPDATE jacobs_pipelines SET office_phase='FAILED',status='failed',office_rejection_code='OFFICE_INPUT_STALE' WHERE pipeline_id=%s", (reservation.pipeline_id,))
                    with pytest.raises(aiomysql.Error):
                        await cur.execute("UPDATE jacobs_pipelines SET office_rejection_code='ARBITRARY_RAW_CODE' WHERE pipeline_id=%s", (reservation.pipeline_id,))
                    # CHECK failure rolls the row back to the registered value.
                    await cur.execute("SELECT office_rejection_code FROM jacobs_pipelines WHERE pipeline_id=%s", (reservation.pipeline_id,))
                    assert (await cur.fetchone())[0] == "OFFICE_INPUT_STALE"
        finally:
            await _clean([reservation.pipeline_id]); await store.cerrar_pool()
    asyncio.run(run())


def test_office_store_rolls_back_when_opaque_event_insert_fails(monkeypatch):
    """Catches a phase/step commit surviving its required opaque audit event."""
    async def run():
        await store.init_tables(); request = _request(); reserved = await cupo.reservar_office_v1(request, limite=10000)
        pid = reserved.pipeline_id; step = deterministic_step_id(pid)
        try:
            assert await store.office_prepare_step(pid, request)
            assert await store.office_claim_preflight(pid, 0, step)
            async def broken(*_args): raise RuntimeError("event failure")
            monkeypatch.setattr(store, "_office_event", broken)
            with pytest.raises(RuntimeError, match="event failure"):
                await store.office_mark_ready(pid, 0, step, "d" * 64, 1)
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT office_phase FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,)); assert (await cur.fetchone())[0] == "PREFLIGHT_IN_FLIGHT"
                    await cur.execute("SELECT input_ref FROM jacobs_steps WHERE step_id=%s", (step,)); assert (await cur.fetchone())[0] == "{}"
        finally:
            await _clean([pid]); await store.cerrar_pool()
    asyncio.run(run())


@pytest.mark.parametrize("operation", [
    "prepare", "claim_preflight", "mark_ready", "claim_delivery", "store_receipt", "finish_terminal",
])
def test_failed_office_event_rolls_back_every_dedicated_transition(monkeypatch, operation):
    """Each Office transition is one DB transaction with its audit event."""
    async def row_state(pid, step):
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT office_phase,status,office_rejection_code FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))
                pipeline = await cur.fetchone()
                await cur.execute("SELECT status,input_ref,output_ref,error FROM jacobs_steps WHERE step_id=%s", (step,))
                step_row = await cur.fetchone()
                await cur.execute("SELECT COUNT(*) FROM jacobs_events WHERE pipeline_id=%s", (pid,))
                return pipeline, step_row, (await cur.fetchone())[0]

    async def run():
        await store.init_tables()
        request = _request()
        reserved = await cupo.reservar_office_v1(request, limite=10000)
        pid, step = reserved.pipeline_id, deterministic_step_id(reserved.pipeline_id)
        try:
            if operation != "prepare":
                assert await store.office_prepare_step(pid, request)
            if operation in {"mark_ready", "claim_delivery", "store_receipt", "finish_terminal"}:
                assert await store.office_claim_preflight(pid, 0, step)
            if operation in {"claim_delivery", "store_receipt"}:
                assert await store.office_mark_ready(pid, 0, step, "a" * 64, 1)
            if operation == "store_receipt":
                assert await store.office_claim_delivery(pid, 0, step)
            before = await row_state(pid, step)

            async def broken(*_args):
                raise RuntimeError("forced opaque event failure")
            monkeypatch.setattr(store, "_office_event", broken)
            with pytest.raises(RuntimeError, match="forced opaque event failure"):
                if operation == "prepare":
                    await store.office_prepare_step(pid, request)
                elif operation == "claim_preflight":
                    await store.office_claim_preflight(pid, 0, step)
                elif operation == "mark_ready":
                    await store.office_mark_ready(pid, 0, step, "b" * 64, 1)
                elif operation == "claim_delivery":
                    await store.office_claim_delivery(pid, 0, step)
                elif operation == "store_receipt":
                    await store.office_store_receipt(
                        pid, 0, step,
                        OfficeReceipt(receipt_id=str(uuid.uuid4()), effect_kind="NOTE_CREATED", effect_id="note:1", output_sha256="c" * 64, office_stage_version=1),
                        "catalog/model",
                    )
                else:
                    await store.office_finish_terminal(pid, 0, step, OfficePhase.PREFLIGHT_IN_FLIGHT,
                                                       "OFFICE_PREFLIGHT_FAILED", OfficePhase.FAILED)
            assert await row_state(pid, step) == before
        finally:
            await _clean([pid])
            await store.cerrar_pool()
    asyncio.run(run())


@pytest.mark.parametrize("input_ref", [
    "{}", "not-json", '{"office_materialization_sha256":"short","binding_version":1}',
    '{"office_materialization_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","binding_version":true}',
    '{"office_materialization_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","binding_version":1,"unexpected":2}',
])
def test_delivery_requires_exact_persisted_binding_without_side_effects(input_ref):
    async def run():
        await store.init_tables(); request = _request(); reservation = await cupo.reservar_office_v1(request, limite=10000)
        pid, step = reservation.pipeline_id, deterministic_step_id(reservation.pipeline_id)
        try:
            assert await store.office_prepare_step(pid, request)
            assert await store.office_claim_preflight(pid, 0, step)
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("UPDATE jacobs_steps SET input_ref=%s WHERE step_id=%s", (input_ref, step))
                    await cur.execute("UPDATE jacobs_pipelines SET office_phase='READY' WHERE pipeline_id=%s", (pid,))
                    await cur.execute("SELECT COUNT(*) FROM jacobs_events WHERE pipeline_id=%s", (pid,)); events = (await cur.fetchone())[0]
            assert not await store.office_claim_delivery(pid, 0, step)
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT office_phase,status FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,)); assert await cur.fetchone() == ("READY", "running")
                    await cur.execute("SELECT status FROM jacobs_steps WHERE step_id=%s", (step,)); assert (await cur.fetchone())[0] == "pending"
                    await cur.execute("SELECT COUNT(*) FROM jacobs_events WHERE pipeline_id=%s", (pid,)); assert (await cur.fetchone())[0] == events
        finally:
            await _clean([pid]); await store.cerrar_pool()
    asyncio.run(run())


def test_office_claim_rejects_stale_epoch_and_wrong_deterministic_step():
    async def run():
        await store.init_tables(); request = _request(); reserved = await cupo.reservar_office_v1(request, limite=10000)
        pid = reserved.pipeline_id; step = deterministic_step_id(pid)
        try:
            assert await store.office_prepare_step(pid, request)
            assert not await store.office_claim_preflight(pid, 1, step)
            with pytest.raises(ValueError, match="step ID"):
                await store.office_claim_preflight(pid, 0, str(uuid.uuid4()))
        finally:
            await _clean([pid]); await store.cerrar_pool()
    asyncio.run(run())


def test_prepare_replay_rejects_mismatched_persisted_step():
    async def run():
        await store.init_tables(); request = _request(); reserved = await cupo.reservar_office_v1(request, limite=10000)
        try:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("INSERT INTO jacobs_steps (step_id,pipeline_id,step_index,facet,capability,input_ref,status,timeout_seconds,retries_allowed,skip_on_fail,trace_id) VALUES (%s,%s,0,'wrong','wrong','{}','pending',1,0,0,%s)", (str(uuid.uuid4()), reserved.pipeline_id, str(uuid.uuid4())))
            assert not await store.office_prepare_step(reserved.pipeline_id, request)
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT COUNT(*) FROM jacobs_events WHERE pipeline_id=%s", (reserved.pipeline_id,))
                    assert (await cur.fetchone())[0] == 0
                    await cur.execute("DELETE FROM jacobs_steps WHERE pipeline_id=%s", (reserved.pipeline_id,))
            assert await store.office_prepare_step(reserved.pipeline_id, request)
        finally:
            await _clean([reserved.pipeline_id]); await store.cerrar_pool()
    asyncio.run(run())


def test_receipt_rejects_forged_models_and_unsafe_model_token_before_write():
    class ReceiptSubclass(OfficeReceipt):
        pass
    async def run():
        await store.init_tables(); request = _request(); reserved = await cupo.reservar_office_v1(request, limite=10000)
        pid, step = reserved.pipeline_id, deterministic_step_id(reserved.pipeline_id)
        try:
            assert await store.office_prepare_step(pid, request)
            assert await store.office_claim_preflight(pid, 0, step)
            assert await store.office_mark_ready(pid, 0, step, "a" * 64, 1)
            assert await store.office_claim_delivery(pid, 0, step)
            receipt = OfficeReceipt(receipt_id=str(uuid.uuid4()), effect_kind="NOTE_CREATED", effect_id="note:1", output_sha256="b" * 64, office_stage_version=1)
            forged = OfficeReceipt.model_construct(**(receipt.model_dump() | {"effect_kind": "bad"}))
            duck = SimpleNamespace(**receipt.model_dump())
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT office_phase,status FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))
                    before_pipeline = await cur.fetchone()
                    await cur.execute("SELECT COUNT(*) FROM jacobs_events WHERE pipeline_id=%s", (pid,))
                    before_events = (await cur.fetchone())[0]
            for value, model in ((forged, "catalog/model"), (ReceiptSubclass(**receipt.model_dump()), "catalog/model"),
                                 (duck, "catalog/model"), (receipt, "bad model")):
                with pytest.raises(ValueError):
                    await store.office_store_receipt(pid, 0, step, value, model)
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT office_phase,status FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))
                    assert await cur.fetchone() == before_pipeline
                    await cur.execute("SELECT COUNT(*) FROM jacobs_events WHERE pipeline_id=%s", (pid,))
                    assert (await cur.fetchone())[0] == before_events
        finally:
            await _clean([pid]); await store.cerrar_pool()
    asyncio.run(run())


def test_existing_classic_pipeline_keeps_classic_default_after_upgrade():
    async def run():
        await store.init_tables()
        pid = str(uuid.uuid4())
        try:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "INSERT INTO jacobs_pipelines (pipeline_id,name,invoked_by,mode,status,plan,"
                        "current_step_index,max_steps,context_refs,created_at,updated_at) "
                        "VALUES (%s,'office-classic-probe','plataforma','dry_run','completed','[]',0,1,'{}',0,0)",
                        (pid,))
                    await cur.execute("SELECT execution_contract, office_job_id, office_phase FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))
                    row = await cur.fetchone()
            assert row == ("CLASSIC", None, None)
        finally:
            await _clean([pid])
            await store.cerrar_pool()
    asyncio.run(run())


def test_real_office_row_is_invisible_to_continuar_and_cli_relaunch():
    """The real service rejects an Office row before the CLI can invoke an executor."""
    async def run():
        await store.init_tables()
        reservation = await cupo.reservar_office_v1(_request(), limite=10000)
        assert reservation and reservation.created
        pid = reservation.pipeline_id
        try:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "UPDATE jacobs_pipelines SET office_phase='FAILED',status='failed',"
                        "office_rejection_code='OFFICE_EXECUTION_FAILED' WHERE pipeline_id=%s",
                        (pid,),
                    )
            cli = _relaunch_cli_module()
            with patch("jacobs.executor.run_pipeline", AsyncMock()) as executor:
                # No continuation or storage seam is mocked: this calls the real
                # CLI, then the real continuar() against the disposable Office row.
                assert await cli.relanzar(pid, {}, None) == 1
                executor.assert_not_awaited()
        finally:
            await _clean([pid])
            await store.cerrar_pool()
    asyncio.run(run())


@pytest.mark.parametrize("amount", ["0", "0.000001", "99999999.999999"])
def test_office_budget_is_exactly_persisted_replayed_and_bound_to_the_digest(amount):
    async def run():
        await store.init_tables()
        request = _request(accepted_cost_usd=amount)
        reservation = await cupo.reservar_office_v1(request, limite=10000)
        assert reservation and reservation.created
        try:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT CAST(office_accepted_cost_usd AS CHAR) FROM jacobs_pipelines WHERE pipeline_id=%s",
                        (reservation.pipeline_id,),
                    )
                    assert (await cur.fetchone())[0] == canonical_usd(Decimal(amount))
            replay = await cupo.reservar_office_v1(request, limite=10000)
            assert replay and not replay.created and not replay.conflict and replay.pipeline_id == reservation.pipeline_id
            changed = _request(client_request_id=request.client_request_id, accepted_cost_usd="1.000000")
            if changed.request_sha256 == request.request_sha256:
                changed = _request(client_request_id=request.client_request_id, accepted_cost_usd="2.000000")
            conflict = await cupo.reservar_office_v1(changed, limite=10000)
            assert conflict and conflict.conflict
        finally:
            await _clean([reservation.pipeline_id])
            await store.cerrar_pool()
    asyncio.run(run())


def test_duplicate_and_conflict_do_not_self_deadlock_with_one_pool_slot(monkeypatch):
    """The duplicate path must query through its held cursor, not acquire slot two."""
    monkeypatch.setenv("JAX_JACOBS_DB_POOL_SIZE", "1")

    async def run():
        await store.cerrar_pool()
        await store.init_tables()
        request = _request()
        ids = []
        try:
            # A simultaneous duplicate is the deadlock-prone shape: if the
            # IntegrityError handler grabs another pool connection, both tasks
            # wait forever with pool size one.
            results = await asyncio.wait_for(asyncio.gather(
                cupo.reservar_office_v1(request, limite=10000),
                cupo.reservar_office_v1(request, limite=10000),
            ), timeout=3)
            ids.extend({result.pipeline_id for result in results})
            assert sum(result.created for result in results) == 1
            conflict = _request(client_request_id=request.client_request_id)
            result = await asyncio.wait_for(cupo.reservar_office_v1(conflict, limite=10000), timeout=3)
            assert result.conflict
        finally:
            await _clean(ids)
            await store.cerrar_pool()

    asyncio.run(run())


def test_office_phase_edges_are_epoch_fenced_and_terminal_states_do_not_revive():
    async def run():
        await store.init_tables(); request = _request(); reservation = await cupo.reservar_office_v1(request, limite=10000)
        pid, step = reservation.pipeline_id, deterministic_step_id(reservation.pipeline_id)
        try:
            with pytest.raises(ValueError, match="generic Office phase transitions are disabled"):
                await store.office_transition_phase(pid, 0, OfficePhase.RESERVED, OfficePhase.PREFLIGHT_IN_FLIGHT)
            assert await store.office_prepare_step(pid, request)
            # A terminal unknown outcome is truthful only after delivery began.
            with pytest.raises(ValueError, match="no exit"):
                await store.office_finish_terminal(pid, 0, step, OfficePhase.RESERVED,
                                                   "OFFICE_PREFLIGHT_UNKNOWN", OfficePhase.OUTCOME_UNKNOWN)
            assert await store.office_claim_preflight(pid, 0, step)
            assert await store.office_mark_ready(pid, 0, step, "a" * 64, 1)
            with pytest.raises(ValueError, match="no exit"):
                await store.office_finish_terminal(pid, 0, step, OfficePhase.READY,
                                                   "OFFICE_DELIVERY_OUTCOME_UNKNOWN", OfficePhase.OUTCOME_UNKNOWN)
        finally:
            await _clean([pid])
            await store.cerrar_pool()
    asyncio.run(run())


def test_office_database_check_enforces_full_snapshot_and_fails_closed_on_invalid_upgrade():
    async def run():
        await store.init_tables()
        reservation = await cupo.reservar_office_v1(_request(), limite=10000)
        assert reservation and reservation.created
        pid = reservation.pipeline_id
        try:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    for sql in (
                        "UPDATE jacobs_pipelines SET execution_contract='office_v1' WHERE pipeline_id=%s",
                        "UPDATE jacobs_pipelines SET office_phase='reserved' WHERE pipeline_id=%s",
                        "UPDATE jacobs_pipelines SET status='pending' WHERE pipeline_id=%s",
                        "UPDATE jacobs_pipelines SET office_phase='FAILED', status='failed', office_rejection_code=NULL WHERE pipeline_id=%s",
                        "UPDATE jacobs_pipelines SET office_phase='FAILED', status='failed', office_rejection_code='SENTINEL_OFFICE_CONTENT' WHERE pipeline_id=%s",
                        "UPDATE jacobs_pipelines SET office_phase='RECEIPT_STORED', status='completed', office_rejection_code='UNEXPECTED' WHERE pipeline_id=%s",
                        "UPDATE jacobs_pipelines SET office_phase='OUTCOME_UNKNOWN', status='outcome_unknown', office_rejection_code=NULL WHERE pipeline_id=%s",
                        "UPDATE jacobs_pipelines SET costo_max_aceptado_usd=1 WHERE pipeline_id=%s",
                        "UPDATE jacobs_pipelines SET invoked_by='other' WHERE pipeline_id=%s",
                        "UPDATE jacobs_pipelines SET mode='supervised' WHERE pipeline_id=%s",
                        "UPDATE jacobs_pipelines SET max_steps=2 WHERE pipeline_id=%s",
                        "UPDATE jacobs_pipelines SET tenant_id='0' WHERE pipeline_id=%s",
                        "UPDATE jacobs_pipelines SET office_job_id='ABC' WHERE pipeline_id=%s",
                    ):
                        with pytest.raises(aiomysql.Error):
                            await cur.execute(sql, (pid,))
                    await cur.execute(
                        "SELECT CHECK_CLAUSE FROM information_schema.CHECK_CONSTRAINTS "
                        "WHERE CONSTRAINT_SCHEMA=DATABASE() AND CONSTRAINT_NAME='chk_jacobs_office_binding_v2'"
                    )
                    check = (await cur.fetchone())[0]
                    assert store._normalizar_check_clause(check) == store._normalizar_check_clause(store._OFFICE_BINDING_CHECK_INFORMATION_SCHEMA)
                    await cur.execute("ALTER TABLE jacobs_pipelines DROP CONSTRAINT chk_jacobs_office_binding_v2")
                    await cur.execute("ALTER TABLE jacobs_pipelines ADD CONSTRAINT chk_jacobs_office_binding_v2 CHECK (office_project_id IS NULL OR office_project_id IS NOT NULL)")
                    await cur.execute("UPDATE jacobs_pipelines SET status='pending' WHERE pipeline_id=%s", (pid,))
            with pytest.raises(RuntimeError, match="unknown Office binding constraint content"):
                await store.init_tables()
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("UPDATE jacobs_pipelines SET status='running' WHERE pipeline_id=%s", (pid,))
                    await cur.execute("ALTER TABLE jacobs_pipelines DROP CONSTRAINT chk_jacobs_office_binding_v2")
                    await cur.execute("ALTER TABLE jacobs_pipelines ADD CONSTRAINT chk_jacobs_office_binding_v2 CHECK (" + store._OFFICE_BINDING_CHECK_CLAUSE + ")")
            await store.init_tables()
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("ALTER TABLE jacobs_pipelines MODIFY office_accepted_cost_usd DECIMAL(14,5) NULL")
            with pytest.raises(RuntimeError, match="DECIMAL\\(14,6\\)"):
                await store.init_tables()
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("ALTER TABLE jacobs_pipelines MODIFY office_accepted_cost_usd DECIMAL(14,6) NULL")
            await store.init_tables()
        finally:
            await _clean([pid])
            await store.cerrar_pool()
    asyncio.run(run())


def test_reaper_candidates_exclude_aged_office_and_keep_classic():
    async def run():
        await store.init_tables()
        office = await cupo.reservar_office_v1(_request(), limite=10000)
        assert office and office.created
        classic = str(uuid.uuid4())
        try:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("UPDATE jacobs_pipelines SET created_at=0, updated_at=0 WHERE pipeline_id=%s", (office.pipeline_id,))
                    await cur.execute(
                        "INSERT INTO jacobs_pipelines (pipeline_id,name,invoked_by,mode,status,plan,current_step_index,max_steps,context_refs,created_at,updated_at) VALUES (%s,'classic-reaper','plataforma','autonomous','running','[]',0,1,'{}',0,0)",
                        (classic,),
                    )
            candidates = await store.candidatos_del_reaper([PipelineStatus.running])
            ids = {pipeline.pipeline_id for pipeline, _timeout in candidates}
            assert classic in ids
            assert office.pipeline_id not in ids
        finally:
            await _clean([office.pipeline_id, classic])
            await store.cerrar_pool()
    asyncio.run(run())


def test_same_name_nonunique_office_idempotency_index_fails_closed():
    async def run():
        await store.init_tables()
        try:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("DROP INDEX uq_jacobs_office_client_request ON jacobs_pipelines")
                    await cur.execute("CREATE INDEX uq_jacobs_office_client_request ON jacobs_pipelines (invoked_by, tenant_id, client_request_id)")
            with pytest.raises(RuntimeError, match="correctness index"):
                await store.init_tables()
        finally:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("DROP INDEX IF EXISTS uq_jacobs_office_client_request ON jacobs_pipelines")
            await store.init_tables()
            await store.cerrar_pool()
    asyncio.run(run())


@pytest.mark.parametrize("index,columns,unique", [
    # Same names do not establish the index contract: both column order and
    # full-column uniqueness are part of the durable idempotency/CAS gates.
    ("uq_jacobs_office_client_request", "invoked_by, client_request_id, tenant_id", True),
    ("uq_jacobs_office_stage", "execution_contract, tenant_id, office_project_id", True),
    ("uq_jacobs_office_client_request", "invoked_by, tenant_id, client_request_id", False),
    ("uq_jacobs_office_stage", "execution_contract, tenant_id, office_stage_id", False),
    ("uq_jacobs_office_client_request", "invoked_by, tenant_id(1), client_request_id", True),
    ("uq_jacobs_office_stage", "execution_contract, tenant_id, office_stage_id(1)", True),
])
def test_every_office_correctness_index_requires_exact_order_and_uniqueness(index, columns, unique):
    async def run():
        await store.init_tables()
        try:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(f"DROP INDEX {index} ON jacobs_pipelines")
                    await cur.execute(f"CREATE {'UNIQUE ' if unique else ''}INDEX {index} ON jacobs_pipelines ({columns})")
            with pytest.raises(RuntimeError, match="correctness index"):
                await store.init_tables()
        finally:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(f"DROP INDEX IF EXISTS {index} ON jacobs_pipelines")
            await store.init_tables()
            await store.cerrar_pool()
    asyncio.run(run())


@pytest.mark.parametrize("index", ["uq_jacobs_office_client_request", "uq_jacobs_office_stage"])
def test_required_office_index_creation_timeout_fails_closed(monkeypatch, index):
    async def run():
        await store.init_tables()
        try:
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(f"DROP INDEX {index} ON jacobs_pipelines")
            real = store._crear_indice_acotado
            async def timeout(cur, table, name, ddl):
                if name == index:
                    return False
                return await real(cur, table, name, ddl)
            monkeypatch.setattr(store, "_crear_indice_acotado", timeout)
            with pytest.raises(RuntimeError, match="timed out creating required Office correctness index"):
                await store.init_tables()
            monkeypatch.setattr(store, "_crear_indice_acotado", real)
            await store.init_tables()
        finally:
            monkeypatch.setattr(store, "_crear_indice_acotado", real)
            await store.cerrar_pool()
    asyncio.run(run())


def test_visible_legacy_missing_index_and_each_bounded_timeout_resume(monkeypatch):
    async def canonical():
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT GENERATION_EXPRESSION FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='jacobs_pipelines' AND COLUMN_NAME='visible'")
                expression = (await cur.fetchone())[0]
                await cur.execute("SELECT GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) FROM information_schema.STATISTICS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='jacobs_pipelines' AND INDEX_NAME='idx_pipelines_visibles'")
                indexed = (await cur.fetchone())[0]
        assert store._normalizar_expresion_generada(expression) == store._normalizar_expresion_generada(store._EXPRESION_VISIBLE)
        assert indexed == "user_id,tenant_id,visible,created_at"

    async def legacy(with_index):
        async with store.conexion() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DROP INDEX IF EXISTS idx_pipelines_visibles ON jacobs_pipelines")
                await cur.execute("ALTER TABLE jacobs_pipelines DROP COLUMN visible, ALGORITHM=INPLACE, LOCK=SHARED")
                await cur.execute("ALTER TABLE jacobs_pipelines ADD COLUMN visible TINYINT(1) GENERATED ALWAYS AS (status NOT IN ('discarded','hidden') AND owner_ack_at IS NOT NULL) VIRTUAL, ALGORITHM=INPLACE, LOCK=SHARED")
                if with_index:
                    await cur.execute("CREATE INDEX idx_pipelines_visibles ON jacobs_pipelines (user_id, tenant_id, visible, created_at) ALGORITHM=INPLACE LOCK=NONE")

    async def run():
        await store.init_tables()
        real = store._ddl_acotado
        try:
            for needle, indexed in (("DROP INDEX idx_pipelines_visibles", True), ("DROP COLUMN visible", False), ("ADD COLUMN visible", False)):
                await legacy(indexed)
                async def timeout(cur, ddl, needle=needle):
                    if needle in ddl:
                        return False
                    return await real(cur, ddl)
                monkeypatch.setattr(store, "_ddl_acotado", timeout)
                with pytest.raises(RuntimeError, match="timed out"):
                    await store.init_tables()
                monkeypatch.setattr(store, "_ddl_acotado", real)
                await store.init_tables()
                await canonical()
            await legacy(False)
            await store.init_tables()
            await canonical()
        finally:
            monkeypatch.setattr(store, "_ddl_acotado", real)
            await store.init_tables()
            await store.cerrar_pool()
    asyncio.run(run())
