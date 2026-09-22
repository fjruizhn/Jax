# Trusted external files

| Default | Consumer | Trust domain | Missing behavior / backup |
|---|---|---|---|
| `/etc/jax/authority/trusted-root.json` | B4 root loader | authority | fail closed; critical backup/recovery |
| `/var/lib/jax/authority/trusted-checkpoints.log` | B4 checkpoints | integrity | fail closed; critical backup/recovery |
| `/etc/jax/execution/trusted-approvers.json` | B6 approver loader | execution authority | fail closed; critical backup/recovery |
| `/etc/jax/build/implementation-identity.json` | B7 identity provider | deployment integrity | fail closed; critical backup/recovery |
| `/etc/jax/PAUSE` | LAS MANOS kill switch | operational safety | inspect through kill-switch runbook |

These paths are sensitive. Exact Unix modes are not asserted by this guide; any hardening recommendation is **RECOMMENDED**, not code-required.
