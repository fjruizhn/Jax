# JAX operational orientation

This file provides stable operational orientation. It is not canonical or normative authority, and it is not proof of current runtime state.

- For current authority, decisions, executions, evidence, health, deployment state, or configuration, use `jaxctl` and [the operations query guide](docs/operations/query-guide.md).
- For normative authority, use `policy/**` and verified authority artifacts.
- For procedures, use `docs/runbooks/**`.
- For technical debt and known limitations, use `DEUDA.md`.
- For historical narrative, plans, and old checkpoints, use `docs/historia/**` and the historical documentation directories.

Memory and old snapshots may suggest where to inspect, but never establish current truth. Dated snapshots must identify their source, observation time, scope, and deployment/revision identity.
- For the anti-fabrication structure detector (`jax/anti_fabricacion/`), its `axioma_config` mode key and the required activation order (seed `sombra` BEFORE restarting LAS MANOS), use [the activation runbook](docs/runbooks/anti-fabricacion-activacion.md).
