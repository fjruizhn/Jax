# 4C1 storage review evidence

The implementation keeps raw provider text at the callback DTO boundary only.
`office_utf8_sha256` uses strict UTF-8 encoding and the DTO validators compare
the supplied digest without parsing or reserializing the text.

The six persistence operations use a pipeline-first transactional sequence and
write only identifiers, phases, hashes, stable codes, or receipt references to
Jacobs. Classic readers join/filter `execution_contract='CLASSIC'`; Office
bindings retain separately named readers.

The database evidence was run against the isolated C1 schema, never the shared
JXP schema. See the task report for commands and results.

Parent independent2026-10-21:80tests0.92s121fixturewarnings onprivatejax_memory_test_office_c1. Actualprepare/snapshot/claimSELECTs instrumented:5EXPLAINplansindexedwithoutfilesort/temp, receipt2026-10-21-office-c1-query-plans.json. No Officecallback/route/providerenabled.

Fix-round evidence, 2026-10-21: the private C1 runner completed 101 focused
Office/Classic compatibility cases. The new real-DB fault matrix injects the
opaque-event failure before insert for all six dedicated transitions and
compares pipeline state, step state, and event count before/after rollback.
Delivery’s persisted materialization binding is now validated under the same
pipeline-then-step locks as its phase claim; malformed, missing, or extra-key
bindings cause no event or state change. The CHECK migration test covers a
real v2 ADD followed by a simulated ambiguous timeout and a clean retry.

Parent independent finalfixcandidate:107tests1.14s191warnings; afterremovingunreachablegenericbody, focusedphasegate1passed0.12s. ExactSHA scopedreview required; prior3d09ae1cREJECTEDfindingsnotwaived.

Fix round 2, 2026-10-21: the earlier matrix claim was partial: it did not
exercise the fresh-neither CHECK state, invalid pre-existing data during the
v2 ADD, or a real Office row through the relaunch CLI.  The C1 disposable
schema now covers all three.  The invalid-row case records the real
`ADD CONSTRAINT chk_jacobs_office_binding_v2` through its DDL seam and gets
MariaDB 4025 from that exact ALTER; because the fixture is invalid under both
versions, MariaDB names the already-installed v1 constraint in its diagnostic.
The failed ADD leaves v1 installed and the corrupt row unchanged.  The
relaunch case invokes the real CLI and real `continuar()` with only the
executor stubbed, which is never called.  The available complete focused
command passed 75 tests in 1.10s with 216
fixture warnings.  `tests/test_office_cupo_db.py` is not present in this
checkout, so it was not silently omitted from a command that named it.

Final matrix closure: `OfficeReceipt.model_construct`, an `OfficeReceipt`
subclass, and a duck-shaped `SimpleNamespace` are each rejected before a
receipt write.  The focused disposable-DB case passed in 0.12s (5 fixture
warnings) and compares pipeline phase/status plus event count before and
after every rejected value.

Parent independiente finalfix2: comando cuatro archivos(incluyecallbackHTTP yCLI)75passed,216warnings,1.07s. Último duckobject+sinmutación focused1passed0.11s trasesecomando. Ningúnproducto editado en fix2. Scoped exactSHAreview pendiente.
