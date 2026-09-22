# Authoritative query guide

This guide is not authority.

| Question | Source |
|---|---|
| Normative rule | `policy/rules/**`, resolver/ledger as applicable |
| Active authority | Block 4 ledger/effective-authority API |
| Decision | Block 5 `load_decision` / `replay_decision` |
| Execution | Block 6 through `jaxctl execution` |
| Control/enforcement | Block 7 `EnforcementStatusService` through `jaxctl control` |
| Evidence | Block 7 EvidenceStore through `jaxctl evidence` |
| Capability catalog | Motor Registry catalog |
| Runtime health | `jaxctl health` and health guide |
| Technical debt | `DEUDA.md` |
| History | `docs/historia/**` |
| Memory | Locator only; verify elsewhere |
