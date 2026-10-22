# Decisions carried from SDD ledger

2026-10-20 · Codex root · technical decisions within authorizeddevelopment; source signed sessionledger, not productionproof.

Ruling: scope global superuser authority to Virtual Office in JXP, not all classic Axioma/Faro — matches requested product and avoids unrelated authority expansion — if wrong, broader product behavior needs later design change.

Ruling: do not enforce at-most-one superuser — user did not request a singleton restriction — if wrong, provisioning policy needs revision.

Ruling: add an exact paired namespace contract marker before the JXP session barrier is enabled — API-presence checks cannot distinguish a reviewed core with all legacy consumer gates from a partial deployment — costs a synchronized version bump for future incompatible core changes. Namespace authority is unchanged. Task2 follows approvedTask1a51ca563.

Ruling: add a separate typed same-cursor Office membership adapter and exactpairedversion.2 before invitation acceptance — existingclassicmutators correctlydenyOffice, JXPrawINSERTwould bypasscanonicalauthority — costs migration/requestledger and grant/revokecontracts, no newidentitystore.

Ruling: grant needs active target/scope while revoke may clean dormant targets/archivedscopes under explicitcurrentadministrativeauthority — removal must not strand authority until reactivation — costs administrative operation separate fromcontentvisibility; inactive targettenant cleanup reservedOfficeroot.

Ruling: record invitation acceptance as a capability execution with a separately canonical inviter authority basis, rather than an interactive action by the inviter — the bearer can accept when inviter is offline — costs typed execution provenance bound into idempotency; no forged live session or pseudo-user. Add before Task3 first review.

Ruling: permanently bind invitation capability to one durable acceptance key, expose key to trusted authorizer, and retain every canonical stored role during revoke — independent audit found consumable invitation replay and elevated-row cleanup holes — costs a unique index/constraint upgrade that fails on inconsistent history; no capability enabled before clean review.
