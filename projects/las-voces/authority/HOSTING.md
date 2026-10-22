# Ariadna governed host — LV-004

`ariadna_host.py` is the privileged local composition root. It owns the
`AuthorityEngine`, GitHub Actions verifier, canonical Human Authority verifier,
runtime, local lease, tick cadence, stop handling, health and kill-switch
checks. Planner/proposal data cannot replace any of those dependencies.

The GitHub adapter independently resolves a run and its jobs from GitHub's API
and binds the configured repository, run, job, workflow and exact commit SHA.
Missing credentials, network errors, or ambiguous API responses resolve to
`HUMAN_REQUIRED`; inconsistent available evidence is rejected. The human
adapter requires an append-only `activity:<event_id>` reference matching the
task, decision and commit. An actor string is never acceptance proof.

`config/ariadna-pm.json.example` is intentionally disabled by default. An
`ACTIVE_GOVERNED` lifecycle is not deployment permission. `dry-run` acquires a
local test lease and evaluates proposals but never calls a runtime effect.

The repository-owned `config/systemd/jax-ariadna-pm.service` follows the
existing JAX `jaxsvc` and `/srv/jax-prod/jax` convention. It is a template only:
this change neither copies it into `/etc/systemd/system` nor enables or starts
anything. Its filesystem restrictions have syntax coverage in tests, not a
claim that they have been exercised on Hall9000.

After a Human Authority merge/start decision, the operator must first review
the dedicated service account's access to the deployed checkout and create
`/etc/jax/ariadna-pm.json` from the example with
`production_effects_enabled: true`. Then, and only then, copy the unit, run
`systemctl daemon-reload`, and use `systemctl enable --now jax-ariadna-pm`.
Creating `/etc/jax/ariadna-pm.kill` is the independent kill switch: the next
cycle shuts down and releases its local lease. Remove it and explicitly start
the service to resume. Expected post-start check:
`/srv/jax-prod/jax/.venv/bin/python /srv/jax-prod/jax/scripts/ariadna_host.py --config /etc/jax/ariadna-pm.json --health`.

An unresolved LV-003 transition intent yields `RECONCILIATION_REQUIRED`, never
ready; the host does not repair history. The local `flock` lease is deliberately
not distributed consensus.
