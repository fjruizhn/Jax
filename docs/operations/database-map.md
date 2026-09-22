# Database map

| Schema | Role | Migration/source | Backup |
|---|---|---|---|
| `jax_authority` | authoritative ledger | `policy/authority_ledger/migrations` | critical |
| `jax_decisions` | immutable DecisionRecords | `policy/decision_record/migrations` | critical |
| `jax_execution` | governed execution | `policy/execution_control/migrations` | critical |
| `jax_evidence` | authoritative evidence | `policy/enforcement_evidence/migrations` | critical |
| `jax_memory` | operational memory/catalog | `jax/memory`, `jacobs` | operational |

`jax_memory` is neither authority nor evidence. Restore authoritative schemas with their trusted roots and dependencies; B6/B7 writes can share a cross-schema transaction.
