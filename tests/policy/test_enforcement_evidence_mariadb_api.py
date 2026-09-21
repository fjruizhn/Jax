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
