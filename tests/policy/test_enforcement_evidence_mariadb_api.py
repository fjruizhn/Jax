import pytest
from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore

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
    with pytest.raises(Exception): store.readonly_status_snapshot("sha256:"+"a"*64)
    assert any("READ ONLY" in call for call in calls)
    assert any("REPEATABLE READ" in call for call in calls)
    assert "rollback" in calls and "commit" not in calls
