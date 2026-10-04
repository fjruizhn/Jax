# F2-E structured projection — handoff

- Objective: implement the authorized additive/versioned F2-C structured JSON projection and F2-D exact-byte binding on a fresh JAX master branch.
- Starting JAX master: `f7386d8c3690832e5d9c21c79cab776998a4bfa1` (contains #345 and #348).
- First DTO candidate: Jacobs `POST /jacobs/pipeline` response; its small schema exposes the already-accredited `PIPELINE_STATUS` slot and avoids `steps`/pending `STEP_STATUS`.
- User authorization: Fernando approved immutable structured projection; server-owned claim slots; all remaining DTO values stay typed untrusted with per-field provenance; SR-03 remains live; F2-D binds exact canonical bytes; preserve client schema; reject old/unknown API versions.
- Done: created isolated worktree/branch from origin/master; reviewed current response, lifecycle and runtime-status contracts. No implementation changes yet.
- Need: verify the actual authenticated scope available to the Jacobs route; implement F2-C projection and F2-D byte-bound transport APIs; connect only the selected response; test authority-slot linkage, immutable bytes, SR-03, provenance, lifecycle/fail-closed behavior; measure CI floors; checkpoint and PR.
- Parallel dependency: c3 is auditing F2-E #344 delta at `44864db9ab65ecba3dd6288614c682328e372ba6`; keep that work separate.
- Production, Faro, F2-F and F2-G remain out of scope.
- Next command: inspect `las_manos/auth_servicio.py` and the full remaining `create_pipeline` response path to establish server-owned scope before writing the adapter.
