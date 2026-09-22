# B9 shared-memory boundary

Memory is operational history. It is not authority, evidence, execution
authorization, or current state. Consumers use the B9 Memory API and receive
`MemoryEnvelope` objects; model-facing callers construct `PromptMemoryContext`
instead of concatenating database memory strings.

Production scheduled memory jobs are owned by the corresponding systemd units.
The B9 migration SQL is additive and must be applied through the reviewed
database migration workflow, never automatically by importing an application.
