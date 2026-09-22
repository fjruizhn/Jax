import pytest
from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore
from policy.enforcement_evidence.models import ImplementationIdentity, SourceState
from policy.enforcement_evidence.ids import sha256_bytes
from policy.enforcement_evidence.canonical import canonical_bytes
from policy.enforcement_evidence.control_registry import load_control_definition

def test_repeatable_read_is_explicit_store_boundary():
    calls=[]
    class Cur:
        def execute(self, sql): calls.append(sql)
    class Con:
        def cursor(self): return Cur()
        def commit(self): calls.append("commit")
        def rollback(self): calls.append("rollback")
        def close(self): calls.append("close")
    value=MariaDBEvidenceStore(Con).derive_in_repeatable_read(lambda cur: "derived")
    assert value=="derived"
    assert any("REPEATABLE READ" in x for x in calls)
    assert any("CONSISTENT SNAPSHOT" in x for x in calls)

def test_readonly_status_snapshot_uses_readonly_repeatable_read_and_rolls_back(monkeypatch):
    calls=[]
    class Cur:
        def execute(self, sql, *args): calls.append(sql)
        def fetchall(self): return ()
        def fetchone(self): return None
    class Con:
        def cursor(self): return Cur()
        def commit(self): calls.append("commit")
        def rollback(self): calls.append("rollback")
        def close(self): calls.append("close")
    store=MariaDBEvidenceStore(Con)
    with pytest.raises(Exception): store.readonly_status_snapshot("sha256:"+"a"*64, "CTL.B6.GOVERNED_DISPATCH", 1)
    assert any("READ ONLY" in call for call in calls)
    assert any("REPEATABLE READ" in call for call in calls)
    assert "rollback" in calls and "commit" not in calls

def test_readonly_snapshot_rejects_manifest_with_missing_raw_output_blob():
    manifest_bytes=b'{"files":{},"kind":"JAX_BUILD_MANIFEST","schema_version":"1.0"}'
    manifest_hash=sha256_bytes(manifest_bytes)
    identity=ImplementationIdentity("fjruizhn/Jax","a"*40,"b"*40,SourceState.CLEAN,manifest_hash)
    definition=load_control_definition("CTL.B6.GOVERNED_DISPATCH")
    missing="sha256:"+"f"*64
    ci_manifest={"implementation_identity_hash":identity.implementation_identity_hash,"raw_output_blob_hash":missing}
    class Cur:
        def __init__(self): self.last=None; self.args=()
        def execute(self, sql, args=()): self.last=sql; self.args=args
        def fetchone(self):
            if "implementation_identities" in self.last: return (canonical_bytes(identity.projection()).decode(),)
            if "control_definitions" in self.last: return (definition.control_id,definition.control_version,canonical_bytes(definition.projection()).decode())
            if "evidence_blobs" in self.last and self.args[0] == manifest_hash: return (len(manifest_bytes),manifest_bytes)
            return None
        def fetchall(self):
            if "enforcement_observations" in self.last: return ()
            if "test_evidence_manifests" in self.last: return ((identity.implementation_identity_hash,__import__("json").dumps(ci_manifest)),)
            return ()
    class Con:
        def __init__(self): self.cur=Cur()
        def cursor(self): return self.cur
        def rollback(self): pass
        def close(self): pass
    with pytest.raises(Exception):
        MariaDBEvidenceStore(Con).readonly_status_snapshot(identity.implementation_identity_hash,definition.control_id,definition.control_version)
