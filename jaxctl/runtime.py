"""Fixed, read-only composition for jaxctl. No request controls dependencies."""
from __future__ import annotations
import os
from policy.enforcement_evidence.implementation_identity import TrustedImplementationIdentityProvider
from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore
from policy.enforcement_evidence.status_engine import EnforcementStatusService

class UnavailableSource(RuntimeError): pass

def _connection_factory():
    try:
        import pymysql
        host=os.environ["JAX_DB_HOST"]; port=int(os.environ["JAX_DB_PORT"])
    except (ImportError, KeyError, ValueError) as exc:
        raise UnavailableSource("MariaDB B7 composition unavailable") from exc
    return lambda: pymysql.connect(host=host,port=port,user=os.environ.get("JAX_DB_USER", ""),password=os.environ.get("JAX_DB_PASSWORD", ""),database=os.environ.get("JAX_DB_NAME", "jax_memory"),charset="utf8mb4",autocommit=False,connect_timeout=5)

def readonly_status_service():
    store=MariaDBEvidenceStore(_connection_factory())
    return EnforcementStatusService.for_readonly_query(store, TrustedImplementationIdentityProvider(store))

def health():
    # Reachability only; it intentionally does not claim authority integrity.
    factory=_connection_factory(); con=factory()
    try:
        cur=con.cursor(); cur.execute("SELECT 1"); cur.fetchone()
        return {"classification":"OPERATIONAL_DIAGNOSTIC","source":"MariaDB","status":"REACHABLE"}
    finally: con.close()

def decision(decision_id, replay=False):
    try:
        from policy.decision_record.storage import MariaDBDecisionRecordStore
        from policy.decision_record.service import load_decision
        value=load_decision(MariaDBDecisionRecordStore(_connection_factory()),decision_id)
        return {"classification":"AUTHORITATIVE_RUNTIME_DATA","source":"Block 5 DecisionRecord store","status":"FOUND","decision":value}
    except Exception as exc: raise UnavailableSource("Block 5 decision source unavailable") from exc

def execution(execution_id):
    try:
        from policy.execution_control.storage import MariaDBExecutionStore
        store=MariaDBExecutionStore(_connection_factory()); value=store.load_execution(execution_id)
        return {"classification":"AUTHORITATIVE_RUNTIME_DATA","source":"Block 6 governed execution store","status":"FOUND","execution":value,"events":store.events(execution_id)}
    except Exception as exc: raise UnavailableSource("Block 6 execution source unavailable") from exc

def evidence(identity):
    try:
        store=MariaDBEvidenceStore(_connection_factory())
        try: value=store.load_evidence_artifact(identity); kind="artifact"
        except Exception:
            data=store.get_evidence_blob(identity); value={"evidence_hash":identity,"size_bytes":len(data)}; kind="blob"
        return {"classification":"AUTHORITATIVE_RUNTIME_DATA","source":"Block 7 EvidenceStore","status":"FOUND","kind":kind,"evidence":value}
    except Exception as exc: raise UnavailableSource("Block 7 evidence source unavailable") from exc

def authority():
    try:
        from policy.authority_ledger.storage import MariaDBAuthorityLedgerStore
        from policy.authority_ledger.trusted_root import TrustedAuthorityRoot
        from policy.authority_ledger.trusted_checkpoint import TrustedCheckpointStore
        from policy.authority_ledger.replay import verify_authority_ledger
        store=MariaDBAuthorityLedgerStore(_connection_factory())
        state=verify_authority_ledger(store.get_genesis(),store.events(),TrustedAuthorityRoot.load(),TrustedCheckpointStore())
        return {"classification":"AUTHORITATIVE_RUNTIME_DATA","source":"Block 4 authority ledger","status":"VERIFIED","authority":state}
    except Exception as exc: raise UnavailableSource("Block 4 authority source unavailable") from exc
