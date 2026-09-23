"""Block 6 evidence against the real MariaDB adapter (CI service only)."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import pytest

from policy.execution_control.authorization import (authorize_execution,
    build_execution_request, deserialize_execution_authorization)
from policy.execution_control.errors import (DecisionExecutionConflictError,
    UnknownExecutionError)
from policy.execution_control.service import create_execution
from policy.execution_control.service import _record
from policy.execution_control.storage import ExecutionEvent, MariaDBExecutionStore
from policy.execution_control.models import ExecutionEnvironment
from tests.policy.test_execution_request import catalog, record


pytestmark = pytest.mark.skipif(
    not os.environ.get("JAX_EXECUTION_TEST_MARIADB"),
    reason="requiere MariaDB aislada del job governed-execution-mariadb",
)


def _connection():
    import pymysql
    return pymysql.connect(host=os.environ["JAX_DB_HOST"], port=int(os.environ["JAX_DB_PORT"]),
        user=os.environ["JAX_DB_USER"], password=os.environ["JAX_DB_PASSWORD"],
        database=os.environ["JAX_DB_NAME"], charset="utf8mb4", autocommit=False)


def _apply_migration() -> None:
    sql = (Path(__file__).parents[2] / "policy/execution_control/migrations/001_governed_execution.sql").read_text()
    tables, triggers = sql.split("DELIMITER //", 1)
    triggers, _ = triggers.split("DELIMITER ;", 1)
    connection = _connection()
    try:
        cursor = connection.cursor()
        # CI runs this focused suite against one service database.  The
        # migration contains immutable trigger declarations, so replaying it
        # after the first test is not a valid migration operation.
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='jax_execution' AND table_name='execution_records'")
        if cursor.fetchone()[0]:
            return
        for statement in tables.split(";"):
            if statement.strip(): cursor.execute(statement)
        for statement in triggers.split("//"):
            if statement.strip(): cursor.execute(statement)
        connection.commit()
    finally:
        connection.close()

def _apply_evidence_migration() -> None:
    sql = (Path(__file__).parents[2] / "policy/enforcement_evidence/migrations/001_enforcement_evidence.sql").read_text()
    tables, triggers = sql.split("DELIMITER //", 1)
    triggers, _ = triggers.split("DELIMITER ;", 1)
    connection = _connection()
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='jax_evidence' AND table_name='evidence_blobs'")
        if cursor.fetchone()[0]:
            return
        for statement in tables.split(";"):
            if statement.strip(): cursor.execute(statement)
        for statement in triggers.split("//"):
            if statement.strip(): cursor.execute(statement)
        connection.commit()
    finally: connection.close()

def test_migraciones_son_idempotentes_aplicadas_dos_veces():
    """MINOR-7 (PR#264 ronda 2 de revisión): `IF NOT EXISTS` se agregó a
    los 22+10 `CREATE TRIGGER` de las dos migraciones para que un re-run a
    medio aplicar no muera en el primer trigger con "already exists" --
    pero `_apply_migration()`/`_apply_evidence_migration()` de arriba
    tienen un short-circuito (si la tabla ya existe, no ejecutan nada), así
    que esa idempotencia nunca se ejercitaba de verdad. Este test aplica
    las DOS migraciones completas DOS VECES seguidas, sin el
    short-circuito, y prueba que la segunda pasada no falla.

    Corre primero en el archivo (antes de `test_b7_real_mariadb_artifact_and_observation_relations`,
    que DROPea triggers a propósito para probar detección de manipulación)
    para no interferir con el estado que esos tests esperan encontrar: al
    terminar este test, el esquema queda con TODO presente -- el mismo
    estado final que dejaría una sola aplicación."""
    for ruta in (
        Path(__file__).parents[2] / "policy/execution_control/migrations/001_governed_execution.sql",
        Path(__file__).parents[2] / "policy/enforcement_evidence/migrations/001_enforcement_evidence.sql",
    ):
        sql = ruta.read_text()
        tables, triggers = sql.split("DELIMITER //", 1)
        triggers, _ = triggers.split("DELIMITER ;", 1)
        for intento in (1, 2):
            connection = _connection()
            try:
                cursor = connection.cursor()
                for statement in tables.split(";"):
                    if statement.strip():
                        cursor.execute(statement)
                for statement in triggers.split("//"):
                    if statement.strip():
                        cursor.execute(statement)
                connection.commit()
            except Exception as exc:
                connection.rollback()
                raise AssertionError(
                    f"{ruta.name} falló en el intento {intento} de 2 -- "
                    "IF NOT EXISTS no está haciendo su trabajo"
                ) from exc
            finally:
                connection.close()


def test_b7_evidence_blob_real_mariadb_immutability():
    """CI-only real DB proof: bytes deduplicate and trigger blocks mutation."""
    from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore
    _apply_evidence_migration()
    store = MariaDBEvidenceStore(_connection); blob = store.put_evidence_blob(b"b7-real-db")
    assert store.put_evidence_blob(b"b7-real-db").evidence_hash == blob.evidence_hash
    connection = _connection()
    try:
        with pytest.raises(Exception):
            cursor=connection.cursor(); cursor.execute("UPDATE jax_evidence.evidence_blobs SET size_bytes=1 WHERE evidence_hash=%s", (blob.evidence_hash,)); connection.commit()
        connection.rollback()
    finally: connection.close()

def test_b7_real_mariadb_artifact_and_observation_relations():
    from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore
    from policy.enforcement_evidence.control_registry import load_control_definition
    from policy.enforcement_evidence.models import (ImplementationIdentity, SourceState, ClaimScope, ClaimEnvironment)
    from policy.enforcement_evidence.trusted_lifecycle import EvidenceLifecycleService, RuntimeEvidenceRecorder
    from policy.enforcement_evidence.implementation_identity import _ControlledTestIdentityProvider
    _apply_evidence_migration(); store=MariaDBEvidenceStore(_connection); now=datetime.now(timezone.utc)
    manifest=store.put_evidence_blob(b"b7-manifest"); definition=load_control_definition("CTL.B6.GOVERNED_DISPATCH")
    identity=ImplementationIdentity("fjruizhn/Jax","a"*40,"b"*40,SourceState.CLEAN,manifest.evidence_hash)
    lifecycle=EvidenceLifecycleService(store,_ControlledTestIdentityProvider(identity)); recorder=RuntimeEvidenceRecorder(lifecycle,"ci",ClaimScope(ClaimEnvironment.CI))
    observation=recorder.record_governed_dispatch_denied()
    artifact=store.load_evidence_artifact(observation.evidence_artifact_hashes[0])
    loaded=store.load_evidence_artifact(artifact.artifact_hash)
    assert loaded.artifact_hash == artifact.artifact_hash
    assert _scalar("SELECT COUNT(*) FROM jax_evidence.observation_artifacts WHERE observation_id=%s",(observation.observation_id,)) == 1
    assert store.load_observation(observation.observation_id).observation_hash == observation.observation_hash
    connection=_connection()
    try:
        with pytest.raises(Exception):
            cursor=connection.cursor(); cursor.execute("DELETE FROM jax_evidence.evidence_artifacts WHERE artifact_hash=%s",(artifact.artifact_hash,)); connection.commit()
        connection.rollback()
    finally: connection.close()

def test_b7_real_mariadb_transaction_rollback_leaves_no_artifact():
    _apply_evidence_migration(); marker = "sha256:" + "f" * 64
    connection = _connection()
    try:
        cursor=connection.cursor()
        with pytest.raises(Exception):
            cursor.execute("INSERT INTO jax_evidence.evidence_artifacts(artifact_hash,canonical_artifact) VALUES (%s,%s)",(marker,"{}"))
            # A missing blob FK makes the relationship insertion fail; rollback
            # must erase the otherwise valid artifact row as one transaction.
            cursor.execute("INSERT INTO jax_evidence.evidence_artifact_blobs(artifact_hash,evidence_hash) VALUES (%s,%s)",(marker,"sha256:"+"e"*64))
            connection.commit()
        connection.rollback()
        cursor.execute("SELECT COUNT(*) FROM jax_evidence.evidence_artifacts WHERE artifact_hash=%s",(marker,))
        assert cursor.fetchone()[0] == 0
    finally: connection.close()

def test_b7_writer_failure_rolls_back_real_governed_execution():
    _apply_migration(); now=datetime.now(timezone.utc); decision=record()
    request=build_execution_request(decision, authenticated_caller_id="jacobs", capability="CAP", motor="m", environment=ExecutionEnvironment.SANDBOX, target_kind="JAX_WORKSPACE", target_value="JAX_WORKSPACE", prompt="p", context={"x": 1}, timeout_seconds=60)
    auth=authorize_execution(decision, request, catalog(), now_utc=now)
    store=MariaDBExecutionStore(_connection); store.insert_authorization(auth)
    def fail(_cursor, _record): raise RuntimeError("forced B7 persistence failure")
    store.execution_evidence_writer=fail
    with pytest.raises(RuntimeError): create_execution(store, auth, now_utc=now)
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_authorization_consumptions WHERE authorization_id=%s",(auth.authorization_id,)) == 0
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_records WHERE decision_id=%s",(decision.decision_id,)) == 0
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_events WHERE event_type='EXECUTION_CREATED'") == 0

def test_b7_dispatch_writer_failure_rolls_back_dispatch_event():
    from policy.execution_control.service import dispatch_execution
    _apply_migration(); now=datetime.now(timezone.utc); decision=record()
    request=build_execution_request(decision, authenticated_caller_id="jacobs", capability="CAP", motor="m", environment=ExecutionEnvironment.SANDBOX, target_kind="JAX_WORKSPACE", target_value="JAX_WORKSPACE", prompt="p", context={"x": 1}, timeout_seconds=60)
    auth=authorize_execution(decision, request, catalog(), now_utc=now)
    store=MariaDBExecutionStore(_connection); store.insert_authorization(auth); execution=create_execution(store, auth, now_utc=now)
    def fail(_cursor, _event): raise RuntimeError("forced B7 dispatch evidence failure")
    store.dispatch_evidence_writer=fail
    with pytest.raises(RuntimeError): dispatch_execution(store, execution, auth, now_utc=now, job_id="b7-job")
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_events WHERE execution_id=%s AND event_type='MOTOR_DISPATCHED'",(execution.execution_id,)) == 0

def test_b7_composed_writer_persists_with_same_create_transaction():
    """The installed writer uses the B6 cursor; no second B7 commit exists."""
    from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore
    from policy.enforcement_evidence.models import ImplementationIdentity, SourceState, ClaimScope, ClaimEnvironment
    from policy.enforcement_evidence.trusted_lifecycle import EvidenceLifecycleService, RuntimeEvidenceRecorder
    from policy.enforcement_evidence.implementation_identity import _ControlledTestIdentityProvider
    from policy.execution_control.service import configure_b7_execution_evidence
    _apply_migration(); _apply_evidence_migration(); now=datetime.now(timezone.utc); decision=record()
    evidence=MariaDBEvidenceStore(_connection); manifest=evidence.put_evidence_blob(b"runtime-manifest")
    identity=ImplementationIdentity("fjruizhn/Jax","a"*40,"b"*40,SourceState.CLEAN,manifest.evidence_hash)
    recorder=RuntimeEvidenceRecorder(EvidenceLifecycleService(evidence,_ControlledTestIdentityProvider(identity)),"ci",ClaimScope(ClaimEnvironment.CI,database_scope_id="ci-mariadb"))
    request=build_execution_request(decision, authenticated_caller_id="jacobs", capability="CAP", motor="m", environment=ExecutionEnvironment.SANDBOX, target_kind="JAX_WORKSPACE", target_value="JAX_WORKSPACE", prompt="p", context={"x": 1}, timeout_seconds=60)
    auth=authorize_execution(decision, request, catalog(), now_utc=now)
    store=MariaDBExecutionStore(_connection); store.insert_authorization(auth)
    configure_b7_execution_evidence(store,evidence,recorder)
    execution=create_execution(store,auth,now_utc=now)
    assert _scalar("SELECT COUNT(*) FROM jax_evidence.enforcement_observations WHERE canonical_observation LIKE %s",("%"+execution.execution_id+"%",)) == 1

def test_b7_composed_writer_persists_with_same_dispatch_transaction():
    from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore
    from policy.enforcement_evidence.models import ImplementationIdentity, SourceState, ClaimScope, ClaimEnvironment
    from policy.enforcement_evidence.trusted_lifecycle import EvidenceLifecycleService, RuntimeEvidenceRecorder
    from policy.enforcement_evidence.implementation_identity import _ControlledTestIdentityProvider
    from policy.execution_control.service import configure_b7_execution_evidence, dispatch_execution
    _apply_migration(); _apply_evidence_migration(); now=datetime.now(timezone.utc); decision=record()
    evidence=MariaDBEvidenceStore(_connection); manifest=evidence.put_evidence_blob(b"dispatch-runtime-manifest")
    identity=ImplementationIdentity("fjruizhn/Jax","c"*40,"d"*40,SourceState.CLEAN,manifest.evidence_hash)
    recorder=RuntimeEvidenceRecorder(EvidenceLifecycleService(evidence,_ControlledTestIdentityProvider(identity)),"ci",ClaimScope(ClaimEnvironment.CI,database_scope_id="ci-mariadb"))
    request=build_execution_request(decision, authenticated_caller_id="jacobs", capability="CAP", motor="m", environment=ExecutionEnvironment.SANDBOX, target_kind="JAX_WORKSPACE", target_value="JAX_WORKSPACE", prompt="p", context={"x": 1}, timeout_seconds=60)
    auth=authorize_execution(decision, request, catalog(), now_utc=now); store=MariaDBExecutionStore(_connection); store.insert_authorization(auth)
    configure_b7_execution_evidence(store,evidence,recorder); execution=create_execution(store,auth,now_utc=now)
    dispatch_execution(store,execution,auth,now_utc=now,job_id="b7-shared")
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_events WHERE execution_id=%s AND event_type='MOTOR_DISPATCHED'",(execution.execution_id,)) == 1
    assert _scalar("SELECT COUNT(*) FROM jax_evidence.enforcement_observations WHERE canonical_observation LIKE %s",("%MOTOR_DISPATCHED%",)) == 0 # event text is deliberately not evidence payload
    assert _scalar("SELECT COUNT(*) FROM jax_evidence.enforcement_observations WHERE canonical_observation LIKE %s",("%"+execution.execution_id+"%",)) == 2

def test_b7_live_inspector_observes_installed_execution_schema():
    from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore
    from policy.enforcement_evidence.models import ImplementationIdentity, SourceState, ClaimScope, ClaimEnvironment
    from policy.enforcement_evidence.trusted_lifecycle import EvidenceLifecycleService, RuntimeEvidenceRecorder
    from policy.enforcement_evidence.implementation_identity import _ControlledTestIdentityProvider
    from policy.enforcement_evidence.database_evidence import DatabaseControlInspector
    _apply_migration(); _apply_evidence_migration()
    evidence=MariaDBEvidenceStore(_connection); manifest=evidence.put_evidence_blob(b"inspection-manifest")
    identity=ImplementationIdentity("fjruizhn/Jax","e"*40,"f"*40,SourceState.CLEAN,manifest.evidence_hash)
    recorder=RuntimeEvidenceRecorder(EvidenceLifecycleService(evidence,_ControlledTestIdentityProvider(identity)),"ci",ClaimScope(ClaimEnvironment.CI,database_scope_id="ci-mariadb"))
    observation=DatabaseControlInspector(_connection,recorder,deployment_id="ci").inspect_one_decision_one_execution()
    assert observation.subject.identity.startswith("dbscope:sha256:")
    assert observation.outcome.value == "SATISFIED"


def _b8_runtime_identity_fixture(tmp_path):
    """Install a test deployment identity only through fixed composition."""
    from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore
    from policy.enforcement_evidence.models import ImplementationIdentity, SourceState
    from policy.enforcement_evidence.control_registry import _controls, load_control_definition
    from policy.enforcement_evidence.implementation_identity import _V1_REQUIRED_SOURCE_PATHS
    from policy.enforcement_evidence.trusted_lifecycle import EvidenceLifecycleService
    from policy.enforcement_evidence.implementation_identity import _ControlledTestIdentityProvider
    _apply_evidence_migration()
    evidence=MariaDBEvidenceStore(_connection)
    root=Path(__file__).parents[2]
    files={name:"sha256:"+hashlib.sha256((root/name).read_bytes()).hexdigest() for name in _V1_REQUIRED_SOURCE_PATHS}
    manifest=evidence.put_evidence_blob(json.dumps({"schema_version":"1.0","kind":"JAX_BUILD_MANIFEST","files":files},sort_keys=True,separators=(",",":")).encode())
    identity=ImplementationIdentity("fjruizhn/Jax","1"*40,"2"*40,SourceState.CLEAN,manifest.evidence_hash)
    EvidenceLifecycleService(evidence,_ControlledTestIdentityProvider(identity))
    # jax#266: el MISMO camino que corre el paso de despliegue
    # (scripts/sembrar_definiciones_de_control.py), incluido el contexto de
    # composición fija que `__persist_control_definition` ahora exige. Antes
    # esta fixture era el ÚNICO llamador del escritor en todo el repo -- esa
    # era justamente la señal de que producción no tenía camino para sembrar.
    from scripts.sembrar_definiciones_de_control import sembrar as _sembrar_definiciones
    _sembrar_definiciones(
        evidence,
        [(definition.control_id, definition.control_version) for definition in _controls.values()],
        emitir=lambda _linea: None)
    path=tmp_path/"implementation-identity.json"; path.write_text(json.dumps(identity.projection()),encoding="utf-8")
    return evidence, path

def _b7_authoritative_fingerprint():
    """Fingerprint every installed B7 table, including relationship tables.

    The query surface is declared read-only.  A schema-wide content digest is
    intentionally stronger than a selected table-count list: it detects any
    INSERT, DELETE, or in-place UPDATE in the actual installed B7 migration.
    """
    def normalize(value):
        if isinstance(value, bytes): return {"bytes_hex": value.hex()}
        if isinstance(value, datetime): return {"datetime": value.isoformat()}
        if value is None or isinstance(value, (bool, int, float, str)): return value
        return {"text": str(value)}
    connection=_connection()
    try:
        cursor=connection.cursor()
        cursor.execute("SELECT table_name FROM information_schema.tables "
                       "WHERE table_schema=%s AND table_type='BASE TABLE' ORDER BY table_name", ("jax_evidence",))
        result={}
        for (table,) in cursor.fetchall():
            cursor.execute("SELECT column_name FROM information_schema.columns "
                           "WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position", ("jax_evidence",table))
            columns=tuple(item[0] for item in cursor.fetchall())
            quoted=", ".join("`"+column.replace("`","``")+"`" for column in columns)
            cursor.execute("SELECT "+quoted+" FROM `jax_evidence`.`"+table.replace("`","``")+"`")
            rows=[json.dumps([normalize(value) for value in row], sort_keys=True, separators=(",",":"), ensure_ascii=True)
                  for row in cursor.fetchall()]
            rows.sort()
            result[table]={"row_count":len(rows),
                           "content_sha256":hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()}
        return result
    finally:
        connection.close()

def test_b8_jaxctl_control_real_mariadb_is_zero_write(monkeypatch, tmp_path, capsys):
    """The actual CLI/runtime/read-only B7 path cannot alter authoritative rows."""
    evidence, identity_path=_b8_runtime_identity_fixture(tmp_path)
    import policy.enforcement_evidence.implementation_identity as implementation_identity
    from jaxctl.commands import run
    monkeypatch.setattr(implementation_identity,"_DEPLOYMENT_IDENTITY_PATH",str(identity_path))
    before=_b7_authoritative_fingerprint()
    assert {"control_definitions","implementation_identities","evidence_blobs","evidence_artifacts",
            "evidence_artifact_blobs","enforcement_observations","observation_artifacts",
            "enforcement_assertions","assertion_artifacts","assertion_observations",
            "test_evidence_manifests"}.issubset(before)
    status=run(["control","CTL.B6.GOVERNED_DISPATCH","--version","1","--claim","ENFORCED","--scope",'{"environment":"SANDBOX_RUNTIME"}',"--subjects",'[{"subject_type":"EXECUTION","identity":"b8-readonly"}]',"--json"])
    output=capsys.readouterr().out
    assert status == 0 and '"persisted":false' in output
    assert '"classification":"AUTHORITATIVE_READONLY_DERIVATION"' in output
    assert _b7_authoritative_fingerprint() == before

def test_b8_jaxctl_control_unavailable_is_zero_write(monkeypatch, tmp_path, capsys):
    _evidence, identity_path=_b8_runtime_identity_fixture(tmp_path)
    import policy.enforcement_evidence.implementation_identity as implementation_identity
    from jaxctl.commands import run
    monkeypatch.setattr(implementation_identity,"_DEPLOYMENT_IDENTITY_PATH",str(identity_path)+".missing")
    before=_b7_authoritative_fingerprint()
    status=run(["control","CTL.B6.GOVERNED_DISPATCH","--version","1","--claim","ENFORCED","--scope",'{"environment":"SANDBOX_RUNTIME"}',"--subjects",'[{"subject_type":"EXECUTION","identity":"b8-unavailable"}]',"--json"])
    assert status == 2 and '"status":"UNAVAILABLE"' in capsys.readouterr().out
    assert _b7_authoritative_fingerprint() == before


def _scalar(sql, args=()):
    connection = _connection()
    try:
        cursor = connection.cursor(); cursor.execute(sql, args)
        return cursor.fetchone()[0]
    finally:
        connection.close()


def test_governed_execution_authoritative_mariadb_contract():
    _apply_migration()
    now = datetime.now(timezone.utc)
    decision = record()
    def bound_request(for_decision):
        return build_execution_request(for_decision, authenticated_caller_id="jacobs", capability="CAP", motor="m",
            environment=ExecutionEnvironment.SANDBOX, target_kind="JAX_WORKSPACE", target_value="JAX_WORKSPACE",
            prompt="p", context={"x": 1}, timeout_seconds=60)
    auth = authorize_execution(decision, bound_request(decision), catalog(), now_utc=now)
    store = MariaDBExecutionStore(_connection)
    store.insert_authorization(auth)

    loaded = store.load_authorization(auth.authorization_id)
    assert loaded._is_trusted() and loaded.execution_request._is_trusted()
    parsed = deserialize_execution_authorization(loaded.projection_without_hash() | {
        "execution_authorization_hash": loaded.execution_authorization_hash})
    assert not parsed._is_trusted()

    execution = create_execution(store, loaded, now_utc=now)
    assert execution.execution_id
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_records WHERE execution_id=%s", (execution.execution_id,)) == 1
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_authorization_consumptions WHERE authorization_id=%s", (auth.authorization_id,)) == 1
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_events WHERE execution_id=%s", (execution.execution_id,)) == 1
    assert _scalar("SELECT canonical_record FROM jax_execution.execution_records WHERE execution_id=%s", (execution.execution_id,))

    duplicate = authorize_execution(decision, bound_request(decision), catalog(), now_utc=now)
    store.insert_authorization(duplicate)
    with pytest.raises(DecisionExecutionConflictError):
        create_execution(store, store.load_authorization(duplicate.authorization_id), now_utc=now)

    # Independent sessions racing on one decision must leave one durable row.
    concurrent_decision = record()
    first = authorize_execution(concurrent_decision, bound_request(concurrent_decision), catalog(), now_utc=now)
    second = authorize_execution(concurrent_decision, bound_request(concurrent_decision), catalog(), now_utc=now)
    store.insert_authorization(first); store.insert_authorization(second)
    def create(auth_id):
        try:
            return create_execution(MariaDBExecutionStore(_connection),
                MariaDBExecutionStore(_connection).load_authorization(auth_id), now_utc=now)
        except Exception:  # fail-soft: the losing concurrent transaction is the expected uniqueness proof.
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(create, (first.authorization_id, second.authorization_id)))
    assert sum(item is not None for item in outcomes) == 1
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_records WHERE decision_id=%s", (concurrent_decision.decision_id,)) == 1

    # An error after the consumption insert rolls the whole transaction back.
    failing_decision = record()
    failing = authorize_execution(failing_decision, bound_request(failing_decision), catalog(), now_utc=now)
    store.insert_authorization(failing)
    failing_loaded = store.load_authorization(failing.authorization_id)
    failing_record = _record(failing_loaded, now_utc=now)
    with pytest.raises(Exception):
        store.create_execution(failing_loaded, failing_record,
            ExecutionEvent(failing_record.execution_id, "READY_TO_DISPATCH", "EXECUTION_CREATED", now, "x" * 129))
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_authorization_consumptions WHERE authorization_id=%s", (failing.authorization_id,)) == 0
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_records WHERE execution_id=%s", (failing_record.execution_id,)) == 0
    assert _scalar("SELECT COUNT(*) FROM jax_execution.execution_events WHERE execution_id=%s", (failing_record.execution_id,)) == 0

    connection = _connection()
    try:
        cursor = connection.cursor()
        for table in ("execution_records", "execution_events"):
            with pytest.raises(Exception):
                cursor.execute(f"UPDATE jax_execution.{table} SET event_type='X'" if table == "execution_events"
                    else f"UPDATE jax_execution.{table} SET decision_id=decision_id")
            connection.rollback()
            with pytest.raises(Exception):
                cursor.execute(f"DELETE FROM jax_execution.{table}")
            connection.rollback()
    finally:
        connection.close()

    # Tampered canonical bytes and duplicated row keys are both fail closed.
    connection = _connection()
    try:
        cursor = connection.cursor()
        cursor.execute("DROP TRIGGER jax_execution.execution_authorizations_no_update")
        cursor.execute("UPDATE jax_execution.execution_authorizations SET canonical_authorization=JSON_SET(canonical_authorization, '$.execution_request.motor', 'tampered') WHERE authorization_id=%s", (duplicate.authorization_id,))
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(Exception):
        store.load_authorization(duplicate.authorization_id)

    connection = _connection()
    try:
        cursor = connection.cursor()
        cursor.execute("UPDATE jax_execution.execution_authorizations SET execution_request_hash='sha256:" + "0" * 64 + "' WHERE authorization_id=%s", (auth.authorization_id,))
        cursor.execute("DROP TRIGGER jax_execution.execution_records_no_update")
        cursor.execute("UPDATE jax_execution.execution_records SET canonical_record=JSON_SET(canonical_record, '$.motor', 'tampered') WHERE execution_id=%s", (execution.execution_id,))
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(UnknownExecutionError):
        store.load_authorization(auth.authorization_id)
    with pytest.raises(Exception):
        store.load_execution(execution.execution_id)


def test_sembrar_definiciones_de_control_es_idempotente(tmp_path):
    """El paso de despliegue de jax#266, contra la MariaDB real y DESDE CERO.

    Defecto original, medido en vivo el 2026-09-22: `control_definitions`
    quedaba en CERO filas tras aplicar la migración porque en producción
    nadie la sembraba (el único escritor era el privado
    `__persist_control_definition`, llamado sólo por la fixture de este
    archivo). Con la tabla vacía, `readonly_status_snapshot` levanta
    `EvidenceArtifactIntegrityError("snapshot definition mismatch")` y
    `jaxctl control` sale `UNAVAILABLE` con exit 2 -- el paso 9 del runbook
    `docs/runbooks/implementation-identity.md`, que sólo acepta `SUPPORTED`,
    era inalcanzable en cualquier despliegue nuevo.

    ESTE TEST SE PARA EN UNA BASE VIRGEN A PROPÓSITO (revisión adversarial
    de jax#266, BLOCK-1). La versión anterior no lo hacía y **pasaba con la
    siembra destripada**: los tests B8 de este mismo archivo corren ANTES,
    su fixture ya siembra los 11 controles, `_apply_evidence_migration()`
    sale temprano si el esquema existe y la base es de sesión -- así que
    cuando este test llegaba, la tabla ya estaba completa y su primer
    `sembrar()` no insertaba una sola fila. Un control que no puede fallar no
    valida nada, y menos éste, que es la única prueba contra base real de un
    PR que existe para garantizar que alguien siembre.

    `DROP SCHEMA` y no `DELETE`: la tabla es append-only por diseño
    (`definitions_no_delete`), no hay forma de vaciarla desde adentro. Los
    tests que corren después se reabastecen solos -- sus fixtures llaman
    `_apply_evidence_migration()` y vuelven a sembrar.
    """
    from scripts.sembrar_definiciones_de_control import controles_empaquetados, sembrar
    from policy.enforcement_evidence.control_registry import load_control_definition
    from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore

    # FRENO: esto BORRA un esquema. Sólo puede correr contra una base de test.
    # `jax_evidence` no está cubierto por las barreras de `base_de_test.py`
    # (que miran `jax_memory`), así que el freno va acá, explícito.
    base = os.environ.get("JAX_DB_NAME", "")
    assert base.startswith("jax_memory_test"), (
        f"JAX_DB_NAME={base!r} no es una base de test: este test borra el esquema "
        "jax_evidence y NO puede correr contra producción")

    connection = _connection()
    try:
        cursor = connection.cursor()
        cursor.execute("DROP DATABASE IF EXISTS jax_evidence")
        connection.commit()
    finally:
        connection.close()
    _apply_evidence_migration()

    evidence = MariaDBEvidenceStore(_connection)
    controles = controles_empaquetados()

    # PRECONDICIÓN, medida y aseverada: la migración deja la tabla VACÍA.
    # Es el hecho que motivó todo el PR; si algún día la migración sembrara,
    # este assert avisa y el script pasa a ser redundante.
    assert _b7_authoritative_fingerprint()["control_definitions"]["row_count"] == 0

    lineas = []
    primera = sembrar(evidence, controles, emitir=lineas.append)
    # La PRIMERA pasada siembra de verdad: todas nuevas. Esto es lo que se
    # rompe si `sembrar()` deja de escribir -- verificado en rojo quitándole
    # la llamada a persistir (EvidenceBlobMissingError en la comprobación de
    # abajo) y también dejándola escribir sin contexto de composición fija.
    assert [ya_estaba for *_resto, ya_estaba in primera] == [False] * len(controles)
    assert len(lineas) == len(controles) and all("sembrado" in linea for linea in lineas)

    huella_tras_sembrar = _b7_authoritative_fingerprint()["control_definitions"]
    assert huella_tras_sembrar["row_count"] == len(controles)

    # Todas las definiciones del catálogo cargan desde la BASE, con los bytes
    # exactos de la definición empaquetada (load_control_definition del store
    # compara contra el registro y levanta si difieren).
    for control_id, control_version in controles:
        desde_la_base = evidence.load_control_definition(control_id, control_version)
        assert desde_la_base.control_definition_hash == load_control_definition(
            control_id, control_version).control_definition_hash

    # Segunda pasada: idempotente de verdad -- el MISMO digest de contenido de
    # la tabla, no sólo "no tiró excepción".
    segunda = sembrar(evidence, controles, emitir=lambda _linea: None)
    assert [ya_estaba for *_resto, ya_estaba in segunda] == [True] * len(controles)
    assert _b7_authoritative_fingerprint()["control_definitions"] == huella_tras_sembrar


def test_persistir_una_definicion_exige_el_contexto_de_composicion_fija():
    """Tener el store NO alcanza para escribir en `control_definitions`.

    (Revisión adversarial de jax#266, MAJOR-2.) `__persist_control_definition`
    era el ÚNICO escritor de este store que no pedía
    `_require_fixed_composition_write()`, y el PR original agregaba encima un
    método PÚBLICO que lo llamaba -- o sea, una capacidad de escritura sobre
    el esquema inmutable de evidencia entregada como método de objeto, viva en
    el store que `las_manos/server.py` construye al arrancar.
    """
    from policy.enforcement_evidence.control_registry import load_control_definition
    from policy.enforcement_evidence.errors import EvidenceArtifactUntrustedError
    from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore

    evidence = MariaDBEvidenceStore(_connection)

    # El store no expone NINGÚN escritor público para las definiciones.
    assert not [nombre for nombre in dir(evidence)
                if "persist" in nombre and not nombre.startswith("_")]

    with pytest.raises(EvidenceArtifactUntrustedError):
        evidence._MariaDBEvidenceStore__persist_control_definition(
            load_control_definition("CTL.B6.GOVERNED_DISPATCH", 1))
