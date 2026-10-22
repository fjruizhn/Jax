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
