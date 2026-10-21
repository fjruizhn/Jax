"""Real MariaDB checks for opaque Office V1 storage (never production)."""
from __future__ import annotations

import asyncio
import os
import uuid
from decimal import Decimal

import pytest
import aiomysql

from jacobs import cupo, store
from jacobs.models import PipelineStatus
from jacobs.office_contract import (
    OFFICE_WORK_CONTRACT_VERSION, OfficePhase, OfficeRunRequest, canonical_usd, office_request_digest,
)


pytestmark = pytest.mark.skipif(not os.getenv("JAX_DB_HOST"), reason="requires disposable MariaDB")


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
            assert await store.office_transition_phase(pid, 0, OfficePhase.RESERVED, OfficePhase.PREFLIGHT_IN_FLIGHT)
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
        await store.init_tables()
        routes = [
            ((OfficePhase.PREFLIGHT_IN_FLIGHT,), None),
            ((OfficePhase.FAILED,), "OFFICE_PREFLIGHT_FAILED"),
            ((OfficePhase.CANCELLED_REVOKED,), "OFFICE_REVOKED"),
            ((OfficePhase.PREFLIGHT_IN_FLIGHT, OfficePhase.READY), None),
            ((OfficePhase.PREFLIGHT_IN_FLIGHT, OfficePhase.FAILED), "OFFICE_PREFLIGHT_FAILED"),
            ((OfficePhase.PREFLIGHT_IN_FLIGHT, OfficePhase.CANCELLED_REVOKED), "OFFICE_REVOKED"),
            ((OfficePhase.PREFLIGHT_IN_FLIGHT, OfficePhase.OUTCOME_UNKNOWN), "OFFICE_PREFLIGHT_UNKNOWN"),
            ((OfficePhase.PREFLIGHT_IN_FLIGHT, OfficePhase.READY, OfficePhase.FAILED), "OFFICE_EXECUTION_FAILED"),
            ((OfficePhase.PREFLIGHT_IN_FLIGHT, OfficePhase.READY, OfficePhase.CANCELLED_REVOKED), "OFFICE_REVOKED"),
        ]
        ids = []
        try:
            for phases, final_code in routes:
                reservation = await cupo.reservar_office_v1(_request(), limite=10000)
                assert reservation and reservation.created
                ids.append(reservation.pipeline_id)
                previous = OfficePhase.RESERVED
                for index, phase in enumerate(phases):
                    code = final_code if index == len(phases) - 1 else None
                    assert await store.office_transition_phase(reservation.pipeline_id, 0, previous, phase, rejection_code=code)
                    previous = phase
                if previous in {OfficePhase.FAILED, OfficePhase.CANCELLED_REVOKED, OfficePhase.OUTCOME_UNKNOWN, OfficePhase.RECEIPT_STORED}:
                    with pytest.raises(ValueError):
                        await store.office_transition_phase(reservation.pipeline_id, 0, previous, OfficePhase.READY)
            concurrent = await cupo.reservar_office_v1(_request(), limite=10000)
            assert concurrent and concurrent.created
            ids.append(concurrent.pipeline_id)
            winners = await asyncio.gather(*[
                store.office_transition_phase(concurrent.pipeline_id, 0, OfficePhase.RESERVED, OfficePhase.PREFLIGHT_IN_FLIGHT)
                for _ in range(2)
            ])
            assert winners.count(True) == 1
            assert not await store.office_transition_phase(concurrent.pipeline_id, 1, OfficePhase.PREFLIGHT_IN_FLIGHT, OfficePhase.READY)
            ready = await cupo.reservar_office_v1(_request(), limite=10000)
            assert ready and ready.created
            ids.append(ready.pipeline_id)
            assert await store.office_transition_phase(ready.pipeline_id, 0, OfficePhase.RESERVED, OfficePhase.PREFLIGHT_IN_FLIGHT)
            assert await store.office_transition_phase(ready.pipeline_id, 0, OfficePhase.PREFLIGHT_IN_FLIGHT, OfficePhase.READY)
            with pytest.raises(ValueError):
                await store.office_transition_phase(ready.pipeline_id, 0, OfficePhase.READY, OfficePhase.DELIVERY_IN_FLIGHT)
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("UPDATE jacobs_pipelines SET office_phase='DELIVERY_IN_FLIGHT' WHERE pipeline_id=%s", (ready.pipeline_id,))
            with pytest.raises(ValueError):
                await store.office_transition_phase(ready.pipeline_id, 0, OfficePhase.DELIVERY_IN_FLIGHT, OfficePhase.RECEIPT_STORED)
        finally:
            await _clean(ids)
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
                        "WHERE CONSTRAINT_SCHEMA=DATABASE() AND CONSTRAINT_NAME='chk_jacobs_office_binding'"
                    )
                    check = (await cur.fetchone())[0]
                    assert store._normalizar_check_clause(check) == store._normalizar_check_clause(store._OFFICE_BINDING_CHECK_INFORMATION_SCHEMA)
                    await cur.execute("ALTER TABLE jacobs_pipelines DROP CONSTRAINT chk_jacobs_office_binding")
                    await cur.execute("ALTER TABLE jacobs_pipelines ADD CONSTRAINT chk_jacobs_office_binding CHECK (office_project_id IS NULL OR office_project_id IS NOT NULL)")
                    await cur.execute("UPDATE jacobs_pipelines SET status='pending' WHERE pipeline_id=%s", (pid,))
            with pytest.raises(aiomysql.Error):
                await store.init_tables()
            async with store.conexion() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("UPDATE jacobs_pipelines SET status='running' WHERE pipeline_id=%s", (pid,))
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
