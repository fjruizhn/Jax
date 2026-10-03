# F2-E-SR3 Processing Job Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the authorized `PROCESSING_JOB_STATUS` predicate, backed only by
the canonical Processing JSONL store and an immutable Platform-authenticated
tenant/user/project owner context, without changing Motor Registry JobStore v1.

**Architecture:** LAS MANOS receives four authenticated, single-valued owner
headers from the Platform service credential and persists a
`ProcessingOwnershipContext` next to each processing job in
`procesamiento_jobs.jsonl`. `ProcessingJobStore` is a dedicated, server-owned
subclass of the existing `JobStore` over that path, reusing its fsync, index
and lock. A version `.3` runtime registry and resolver make a
fresh exact-scope read into F2-B evidence; F2-C/F2-D then render it through the
existing exact Platform tuple. No request body chooses ownership, store, source
identity, resolver, predicate, or status.

**Tech Stack:** Python 3.12, FastAPI/Pydantic, append+flush+fsync JSONL,
existing F2-A/F2-B/F2-C/F2-D governance modules, pytest, Platform FastAPI and
MariaDB queue repository.

**Spec:** Fernando's 2026-10-03 SR3 authorization: canonical
`PROCESSING_JOB_STATUS`; immutable Platform-authenticated tenant, user, and
project ownership; ownerless quarantine; dedicated Processing store; exact
scope and fresh reads. This plan does not authorize any other predicate.

## Global Constraints

- The only new predicate is `PROCESSING_JOB_STATUS`; Motor, pipeline, facet and
  engine predicates and their sources remain unchanged.
- Processing source is exactly `las_manos/logs/procesamiento_jobs.jsonl`, with
  `store_contract="processing-job-store-v1"` and
  `event_format="processing-job-event-v1"`.
- Closed status values are exactly `pending`, `running`, `cancelling`,
  `completed`, `failed`, and `cancelled`.
- `ProcessingOwnershipContext` is immutable and exact. Its IDs are positive,
  canonical ASCII decimal strings such as `"1"`, `"2"`, and `"3"`:
  `version="processing-owner.1"`, `tenant_id`, `user_id`, `project_id`.
  It comes only from the authenticated Platform service headers, never a
  request-body field.
- Required headers are `X-Jax-Processing-Owner-Version`,
  `X-Jax-Processing-Tenant-Id`, `X-Jax-Processing-User-Id`, and
  `X-Jax-Processing-Project-Id`. Duplicates, absence, non-canonical values,
  unknown owner headers, or owner values not matching the authenticated service
  scope are rejected before the route runs.
- Ownerless and legacy Processing events are quarantined: no status claim is
  resolved from them. Wrong tenant, user, or project returns `WRONG_SCOPE`.
- Sol refinement closed: the shared Project Documents DTO is a separate
  Spanish eight-state document workflow and has no `job_id`. It must never map
  to, be detected as, or accredit `PROCESSING_JOB_STATUS`.
- GET/cancel and claim resolution require the same exact owner scope:
  tenant + project + subject uploader + Platform service actor. Another member
  of the same project cannot reuse the uploader's claim.
- Preserve F2-E bounds: JSON decode layers `2`, strings `65,536` characters,
  nesting depth `32`, and nodes `1,024`. Preserve SR-03 separation between
  native Processing `{job_id, estado}` and generic Motor `{job_id, status}`.
- Runtime API, binding, and resolver versions become `.3`; governed-domain is
  version `6`; `f2-c.renderer.3` remains unchanged; envelope stays `f2-c.1` and
  uses the exact Platform tuple. Existing frozen schemas are additive only.
- Implementation may begin now. Do not integrate before Projects E2a
  deploy/migration is observed and JAX14 is notified. Do not modify production,
  credentials, MotorJobStore v1, or the
  measured F2-E-SR/load head.

## Review Focus

1. Duplicate or missing ownership headers must fail before Processing route
   code sees the request; Task 1 owns direct ASGI middleware tests.
2. `TrabajoRequest` no longer accepts body `usuario`; its existing
   `extra="forbid"` behavior returns 422, and Task 1 pins that body
   non-authority.
3. A JSONL job with no immutable owner must be unavailable/quarantined, never
   inherited from the requester; Task 2 pins this reader behavior.
4. A claim whose tenant/user matches but project differs must be `WRONG_SCOPE`;
   Task 2 pins all three exact owner dimensions.
5. Native `{job_id, estado}` is only `PROCESSING_JOB_STATUS`, while generic
   `{job_id, status}` remains only `JOB_STATUS`; Task 2 pins both forms and all
   bounds without rejecting unrelated native DTO fields.
6. A shared Project Documents DTO with one of its eight Spanish workflow states
   and no `job_id` must not trigger the Processing detector or receipt; Task 2
   pins this regression.

## File Structure

| Path | Change | Responsibility |
| --- | --- | --- |
| `las_manos/auth_servicio.py` | Modify | Authenticate and validate the four closed Processing owner headers for Platform-only Processing routes; expose a typed context through ASGI state. |
| `las_manos/processing_ownership.py` | Create | Immutable owner value object, canonical header parser, exact owner comparison, and no-body-authority validation. |
| `las_manos/processing_job_store.py` | Create | Dedicated append/read JSONL store facade, fresh authoritative snapshots, source configuration, legacy quarantine. |
| `las_manos/proyecto_activo.py` | Modify | Add the minimal canonical identity lookup that remains compatible with the existing project-status lookup. |
| `las_manos/procesamiento_routes.py` | Modify | Require `ProcessingOwnershipContext`, persist it at create, and use the dedicated store through the existing `_STORE` name without changing external status DTO fields. |
| `policy/vocabulary/predicates.yaml` | Modify | Add only canonical `PROCESSING_JOB_STATUS(processing_job_id,status)` with Processing JSONL source. |
| `policy/governance/runtime_status.py` | Modify | `.3` closed registry, source digest, `ProcessingJobStatusResolver`, exact argument key selection and fresh scope evidence. |
| `policy/governance/resolution.py` | Modify | Add the one Processing adapter kind to the already closed runtime-status dispatch set. |
| `policy/governance/governed_domain.py` | Modify | Domain version 6 and exact Processing native-shape recognition without widening Motor/pipeline/facet/engine recognition. |
| `tests/test_las_manos_auth_servicio.py` | Modify | Header duplicate/missing/extra/non-canonical/service identity tests. |
| `las_manos/_procesamiento_routes_test.py` | Modify | Owner persistence, immutable create/cancel/status behavior, legacy quarantine. |
| `tests/test_f2e_runtime_status.py` | Modify | Resolver, source configuration, freshness, scope and source mismatch tests. |
| `tests/test_governed_resolution.py` | Modify | F2-B receipt and runtime-evidence dispatch tests. |
| `tests/test_governed_renderer.py` | Modify | SR-03 native-shape, enum drift and bound tests. |
| Platform `backend/credencial_las_manos.py` | Modify | Build the four headers only from authenticated queue/repository records. |
| Platform `backend/proyectos_documentos/repositorio.py` | Modify | Canonically load tenant, uploader/user and project, including tenant-joined user validation. |
| Platform `backend/proyectos_documentos/despachador.py` | Modify | Send the closed owner context when creating/querying/cancelling Processing jobs; group and mutate only owner-bound rows. |
| Platform `backend/jax_engine/status_resolution.py` and `backend/api/governed_chat.py` | Modify | Consume the `.3` Processing receipt through the exact Platform F2-C tuple; never construct a caller-selected source. |
| Platform focused tests and exact-pair workflow | Modify | Queue-to-header and F2-B→F2-C/F2-D integration tests, then the existing exact-pair CI command. |

## Interfaces

The implementer must use these names and signatures. If a path or local helper
has moved, keep this public contract and adapt only the import location; do not
weaken the contract.

| Symbol | Exact contract |
| --- | --- |
| `ProcessingOwnershipContext` | Frozen value with `version: Literal["processing-owner.1"]`, and positive canonical ASCII-decimal `tenant_id`, `user_id`, `project_id`. |
| `processing_ownership_from_scope(scope)` | Returns the one validated `ProcessingOwnershipContext` in ASGI state or raises the route's closed authorization error. |
| `processing_owner_headers(context)` | Returns exactly the four required owner headers and their canonical values. |
| `ProcessingJobStore.create` | Existing `JobStore.create` keyword contract (`caller`, `capability`, `motor`, `trace_id`, `prompt`, `recursion_depth`, optional `pipeline_id` and `job_id`) plus required `ownership`. |
| `ProcessingJobStore.authoritative_snapshot(job_id)` | Returns the frozen owner snapshot (`view: MotorJobView`, typed `owner: ProcessingOwnershipContext`, `observed_at: datetime`) or `None`, from the locked inherited index. |
| `ProcessingJobStore.source_configuration()` | Returns the seven closed non-secret source configuration keys listed in Task 1 as `dict[str, object]`. |
| `ProcessingJobStatusResolver.evidence(arguments, scope)` | Accepts only canonical `{"processing_job_id", "status"}` and returns `RuntimeStatusEvidence`. |

The native Processing DTO `{job_id, estado}` maps server-side to canonical
predicate arguments `{"processing_job_id": job_id, "status": estado}`;
generic Motor `{job_id, status}` remains `JOB_STATUS` and never crosses that
mapping. `ProcessingJobStatusResolver.evidence` accepts only the canonical
arguments. It returns an observation containing the actual canonical source
result when
the snapshot owner exactly equals `(scope.tenant_id, scope.subject_id,
scope.project_id)`. The registry's existing canonical result/argument equality
check fails closed if a requested status differs. The exact scope also preserves
the authenticated Platform service actor. It returns
unavailable for missing/read-failed/ownerless/quarantined records and
wrong-scope for any owner or actor mismatch. The resolver chooses the server
singleton itself; neither an HTTP request nor Platform can supply a path or
store object.

## Task 1: JAX Processing source, authenticated ownership, and immutable JSONL

**Files:**
- Create: `las_manos/processing_ownership.py`
- Create: `las_manos/processing_job_store.py`
- Modify: `las_manos/auth_servicio.py`, `las_manos/proyecto_activo.py`, `las_manos/procesamiento_routes.py`
- Test: `tests/test_las_manos_auth_servicio.py`, `las_manos/_procesamiento_routes_test.py`

**Consumes:** the Platform service identity already enforced by
`auth_servicio.CredencialDeServicio`; existing Processing `JobStatus` values
and JSONL durability behavior.

**Produces:** `ProcessingOwnershipContext`, `ProcessingJobStore`, and a
server-owned `las_manos.procesamiento_routes._STORE` usable only by
Task 2's resolver.

- [ ] **Step 1: Write failing ASGI middleware tests for the owner-header envelope.**

```python
def test_processing_owner_headers_are_required_once_and_platform_only(client):
    headers = platform_service_headers()
    response = client.post("/procesamiento/trabajos", headers=headers, json=valid_body())
    assert response.status_code == 400

    headers.update(valid_processing_owner_headers())
    duplicate = raw_duplicate_header_request(client, headers,
        "X-Jax-Processing-Tenant-Id", "1", "1")
    assert duplicate.status_code == 400

def test_trabajo_request_rejects_legacy_usuario_as_extra(client):
    response = client.post("/procesamiento/trabajos", headers=platform_service_headers() | valid_processing_owner_headers(),
        json={"project_uuid": VALID_PROJECT_UUID, "rutas": ["/entrada.pdf"], "usuario": "2"})
    assert response.status_code == 422

def test_processing_owner_rejects_extra_noncanonical_and_body_usuario_authority(client):
    headers = platform_service_headers() | valid_processing_owner_headers()
    assert client.post("/procesamiento/trabajos", headers=headers | {
        "X-Jax-Processing-Unknown": "x"}, json=valid_body()).status_code == 400
    assert client.post("/procesamiento/trabajos", headers=headers,
        json=valid_body(usuario="attacker")).status_code == 422
```

- [ ] **Step 2: Run the new authentication tests and confirm they fail because Processing ownership validation is absent.**

Run: `PYTHONPATH=.:las_manos /home/fruiz/jax/.venv/bin/python -m pytest tests/test_las_manos_auth_servicio.py -q`

Expected: failures for missing/duplicate/extra Processing ownership headers and body authority.

- [ ] **Step 3: Implement the typed header contract and middleware gate.**

```python
PROCESSING_OWNER_HEADERS = (
    "x-jax-processing-owner-version", "x-jax-processing-tenant-id",
    "x-jax-processing-user-id", "x-jax-processing-project-id",
)

def processing_ownership_from_headers(headers: Sequence[tuple[bytes, bytes]]) -> ProcessingOwnershipContext:
    values: dict[str, str] = {}
    for raw_name, raw_value in headers:
        name = raw_name.decode("latin-1").lower()
        if name.startswith("x-jax-processing-owner-") and name not in PROCESSING_OWNER_HEADERS:
            raise ValueError("processing owner header unknown")
        if name in PROCESSING_OWNER_HEADERS:
            if name in values:
                raise ValueError("processing owner header duplicated")
            values[name] = raw_value.decode("ascii")
    if set(values) != set(PROCESSING_OWNER_HEADERS) or values["x-jax-processing-owner-version"] != "processing-owner.1":
        raise ValueError("processing owner headers invalid")
    identifiers = tuple(values[name] for name in (
        "x-jax-processing-tenant-id", "x-jax-processing-user-id",
        "x-jax-processing-project-id"))
    if any(not value.isascii() or not value.isdecimal() or value == "0" or value.startswith("0") for value in identifiers):
        raise ValueError("processing owner identifier noncanonical")
    return ProcessingOwnershipContext("processing-owner.1", *identifiers)
```

Install the context at `scope["state"]["processing_ownership"]` only after
the service identity is `plataforma` and the route is one of the three exact
Processing routes. Make `procesamiento_routes` obtain it through
`processing_ownership_from_scope(request.scope)`. Remove `usuario` from
`TrabajoRequest`; its existing `extra="forbid"` returns 422 if callers retain
that body field. It never selects persisted ownership.

Add `proyecto_activo.py`'s minimal canonical identity lookup alongside
`estado_del_proyecto(project_uuid)`: it returns only the canonical positive
decimal tenant, user/uploader and project IDs needed to cross-check the
Platform-authenticated context. Preserve the current status lookup and its
route compatibility; do not create a new generic project resolver.

- [ ] **Step 4: Write failing JSONL ownership and quarantine tests.**

```python
def test_processing_store_persists_immutable_owner_and_reads_fresh_snapshot(tmp_path):
    store = ProcessingJobStore(str(tmp_path / "procesamiento_jobs.jsonl"))
    owner = ProcessingOwnershipContext("processing-owner.1", "1", "2", "3")
    job_id = store.create(caller="2", capability="procesamiento", motor="local",
        trace_id="trace-1", prompt="", recursion_depth=0, ownership=owner)
    snapshot = store.authoritative_snapshot(job_id)
    assert snapshot is not None and snapshot.owner == owner
    assert snapshot.observed_at.tzinfo is not None

def test_ownerless_legacy_processing_event_is_quarantined(tmp_path):
    write_legacy_processing_event(tmp_path / "procesamiento_jobs.jsonl")
    assert ProcessingJobStore(str(tmp_path / "procesamiento_jobs.jsonl")).authoritative_snapshot("legacy") is None
```

- [ ] **Step 5: Implement `ProcessingJobStore` as a subclass without modifying `motor_registry.job_store.JobStore`.**

Reuse base `JobStore` append+flush+fsync, locked index and authoritative
snapshot behavior. Persist `ownership` as a nested canonical object on the
create event. Reject an event that attempts to alter it during update. The
subclass returns a fresh `observed_at` from its locked indexed snapshot, as the
closed SR model does; it does not reopen or fully replay JSONL per read. Return
no snapshot for malformed, legacy-ownerless, unknown-status,
duplicate-owner-key, or noncanonical-owner events. `source_configuration()`
returns exactly:

```python
{
    "store_contract": "processing-job-store-v1",
    "source_id": str(PROCESSING_JSONL_PATH.resolve()),
    "event_format": "processing-job-event-v1",
    "durability": "append-flush-fsync-v1",
    "ownership_contract": "platform-authenticated-processing-owner.1",
    "allowed_statuses": ["pending", "running", "cancelling", "completed", "failed", "cancelled"],
    "source_role": "authoritative-processing-job-status",
}
```

- [ ] **Step 6: Wire create/status/cancel to the new singleton and prove immutable owner behavior.**

Keep the existing `_STORE` name in `procesamiento_routes.py`, instantiate it as
`ProcessingJobStore`, pass the context only to `create`, and preserve it through every `update`, result and cancellation
event. Status/cancel handlers must obtain context first and return an
indistinguishable not-found or governed-unavailable response for an ownerless
or wrong-scope job; they must never expose a record from another owner.

- [ ] **Step 7: Run focused JAX source tests.**

Run: `PYTHONPATH=.:las_manos /home/fruiz/jax/.venv/bin/python -m pytest tests/test_las_manos_auth_servicio.py las_manos/_procesamiento_routes_test.py -q`

Expected: PASS, including duplicate/missing/extra/noncanonical header, body
`usuario`, immutable owner, legacy ownerless, wrong scope, and create/cancel
tests.

## Task 2: JAX F2-B/F2-C/F2-D Processing status registration

**Files:**
- Modify: `policy/vocabulary/predicates.yaml`, `policy/governance/runtime_status.py`, `policy/governance/resolution.py`, `policy/governance/governed_domain.py`
- Test: `tests/test_f2e_runtime_status.py`, `tests/test_f2e_runtime_status_remediation.py`, `tests/test_governed_resolution.py`, `tests/test_governed_renderer.py`

**Consumes:** Task 1's exact `ProcessingJobStore` source configuration and
fresh snapshot interface.

**Produces:** Runtime `.3` canonical `PROCESSING_JOB_STATUS` F2-B evidence,
the server-owned native DTO-to-canonical argument mapping, and an F2-C text
claim path that the exact Platform tuple can send to F2-D.

- [ ] **Step 1: Write failing registry and resolver tests.**

```python
def test_processing_job_status_requires_exact_tenant_user_project_scope(monkeypatch):
    resolver = ProcessingJobStatusResolver()
    args = {"processing_job_id": "proc-1", "status": "completed"}
    exact = _scope(tenant_id="1", subject_id="2", project_id="3", actor_id="service:jax")
    assert resolver.evidence(args, exact).observation.status is ResolutionStatus.RESOLVED
    assert resolver.evidence(args, _scope(tenant_id="1", subject_id="2", project_id="4", actor_id="service:jax")).observation.status is ResolutionStatus.WRONG_SCOPE
    assert resolver.evidence(args, _scope(tenant_id="1", subject_id="5", project_id="3", actor_id="service:jax")).observation.status is ResolutionStatus.WRONG_SCOPE
    assert resolver.evidence(args, _scope(tenant_id="1", subject_id="2", project_id="3", actor_id="service:other")).observation.status is ResolutionStatus.WRONG_SCOPE

```

In the same test module, install a `ProcessingJobStore` fixture at
`procesamiento_routes._STORE`, append an ownerless legacy line and then a line
with status `rejected`, and assert that calling
`ProcessingJobStatusResolver.evidence` for either canonical argument pair
returns `UNAVAILABLE`. Add a registry test that supplies a receipt observation
whose actual canonical `status` differs from the requested one and asserts the
existing result-equality gate returns `SOURCE_MISMATCH`.

- [ ] **Step 2: Run the failing runtime tests.**

Run: `PYTHONPATH=.:las_manos /home/fruiz/jax/.venv/bin/python -m pytest tests/test_f2e_runtime_status.py tests/test_governed_resolution.py -q`

Expected: FAIL because `.2` registry accepts only the four existing runtime predicates.

- [ ] **Step 3: Add only the authorized predicate and `.3` closed runtime composition.**

Add `PROCESSING_JOB_STATUS` to `predicates.yaml` with canonical arguments
`[processing_job_id, status]` and source `ProcessingJobStore`. In
`runtime_status.py`, change only the runtime composition versions to `.3`, add
the closed `_RUNTIME_SPECS` row:

```python
(AdapterKind.LAS_MANOS_PROCESSING_JOB_STATUS, "las-manos:processing-job-store",
 "authority:las-manos", 60, SourceScopeClass.EXACT_RESPONSE_SCOPE,
 "ProcessingJobStatusResolver")
```

Require all seven exact source-configuration keys documented in Task 1,
including ownership contract, allowed status sequence and source role.
`ProcessingJobStatusResolver` imports only `procesamiento_routes._STORE`,
requires exact canonical argument keys, exact closed status values and exact
owner scope plus the authenticated Platform actor. Add the server-owned mapping
from native `{job_id, estado}` to canonical `{processing_job_id, status}` before
the resolver; it is unavailable for malformed values. In `resolution.py`, add only
`AdapterKind.LAS_MANOS_PROCESSING_JOB_STATUS` to the runtime-evidence dispatch set. Do
not alter existing adapter enum values or Motor/pipeline/facet/engine behavior.

- [ ] **Step 4: Write SR-03 and domain bound regressions before changing the domain.**

```python
def test_native_processing_tool_data_is_rejected_before_claim_template_path():
    s = scope()
    candidate = GovernedResponseCandidate(
        "f2-c.1", "processing-native", s.request_id, s.trace_id, s, "web-chat", (),
        (ContentBlock(ContentBlockKind.TOOL_DATA, {"job_id": "x", "estado": "completed"}),), (), ())
    env = response._seal_candidate_for_server(candidate, contract_state=ContractState.VALID,
        governance_receipt=receipt())
    result = GovernedRenderer().render_text(env,
        RenderContext(None, {}, {}, {}, GovernedDomainRegistry(), None, lambda: NOW))
    assert result.text == GovernedRenderer.unavailable_text

def test_project_documents_eight_state_dto_does_not_match_processing_native_shape():
    s = scope()
    candidate = GovernedResponseCandidate(
        "f2-c.1", "document-workflow", s.request_id, s.trace_id, s, "web-chat", (),
        (ContentBlock(ContentBlockKind.TOOL_DATA, {"documento_id": "d-1", "estado": "pendiente_revision"}),), (), ())
    env = response._seal_candidate_for_server(candidate, contract_state=ContractState.VALID,
        governance_receipt=receipt())
    result = GovernedRenderer().render_text(env,
        RenderContext(None, {}, {}, {}, GovernedDomainRegistry(), None, lambda: NOW))
    assert result.text != GovernedRenderer.unavailable_text
```

- [ ] **Step 5: Implement domain version 6 without widening any existing native shape.**

Use the existing `GovernedDomainRegistry`/renderer test composition, rather
than exposing a new domain public method. Add a separate exact native
Processing `{job_id, estado}` detector that maps only to canonical
`{processing_job_id, status}`; retain generic Motor `{job_id, status}` and all
existing pipeline/facet/engine behavior. Accept additional native job DTO fields
without turning them into a schema contract; reject only unknown Processing
status values. Preserve the existing `2` JSON decode / `65,536` character /
`32` depth / `1,024` node traversal ceilings. Bump only the specified domain
and runtime/binding/resolver versions; keep `f2-c.renderer.3`, envelope
`f2-c.1` and the exact Platform tuple unchanged. Do not inspect or map the
Project Documents DTO's eight Spanish workflow values; it has no job identifier
and is not this source.

- [ ] **Step 6: Prove F2-B receipt and F2-C/F2-D revalidation.**

Add a test named
`test_processing_job_status_f2b_to_f2c_f2d_revalidates_exact_scope` that
creates a Task 1 JSONL job, resolves the F2-B receipt from canonical
`{processing_job_id,status}`, renders the resulting supported claim template
as F2-C text, and revalidates that text through F2-D's existing lifecycle
bytes. It must first prove the native `{job_id,estado}` `TOOL_DATA` is rejected.
Assert the `.3` binding/registry digest, owner scope digest equality for the
exact uploader subject and Platform service actor, and rejection after
receipt/status/source/owner mutation. The registry's existing result equality
check rejects a requested status that differs from the actual source status.
Add a separate assertion that another project member cannot reuse that receipt
or call GET/cancel for the uploader's job. This SR3 plan does not add a
structured F2-D byte format; authorized general JSON rendering remains later.

- [ ] **Step 7: Run JAX governance tests.**

Run: `PYTHONPATH=.:las_manos /home/fruiz/jax/.venv/bin/python -m pytest tests/test_f2e_runtime_status.py tests/test_f2e_runtime_status_remediation.py tests/test_governed_resolution.py tests/test_governed_renderer.py -q`

Expected: PASS. These paths were verified in this checkout; do not create a
separate domain test module.

## Task 3: Platform authenticated handoff and real exact-pair integration

**Files:**
- Modify: Platform `backend/credencial_las_manos.py`, `backend/proyectos_documentos/repositorio.py`, `backend/proyectos_documentos/despachador.py`, `backend/jax_engine/status_resolution.py`, `backend/api/governed_chat.py`
- Test: Platform `backend/tests/test_credencial_las_manos.py`, `backend/tests/test_proyectos_documentos_repositorio.py`, `backend/tests/test_proyectos_documentos_despachador.py`, `backend/tests/test_governed_chat_bridge.py`
- Modify: the existing exact-pair workflow only to add these focused tests; do not create a separate synthetic integration workflow.

**Consumes:** Task 1 header contract and Task 2 `.3` registry. Implementation
may start now; integration waits until Projects E2a deploy/migration is
observed and JAX14 has been notified.

**Produces:** a real queue SQL row → authenticated Platform owner headers →
LAS MANOS Processing route → JSONL → F2-B → exact F2-C tuple → F2-D chain.

- [ ] **Step 1: Write failing Platform repository and dispatcher tests.**

```python
def test_processing_dispatch_joins_user_to_tenant_and_builds_canonical_owner(repo):
    owner = repo.processing_owner_for_queue_row(queue_id)
    assert owner == {"version": "processing-owner.1", "tenant_id": "1",
                     "user_id": "2", "project_id": "3"}

def test_dispatch_uses_queue_owner_and_binds_queries_and_mutations_to_owner(dispatcher):
    dispatcher.enqueue_processing(queue_id)
    request = captured_las_manos_request()
    assert request.headers["X-Jax-Processing-User-Id"] == "2"
    assert owner_predicates_are_present_in_every_status_cancel_and_group_query()
```

- [ ] **Step 2: Run the Platform focused tests and confirm the new owner header path is absent.**

Run: `/home/fruiz/jax/.venv/bin/python /tmp/f2e-sr3-run-platform-tests.py nodb tests/test_credencial_las_manos.py tests/test_proyectos_documentos_repositorio.py tests/test_proyectos_documentos_despachador.py tests/test_governed_chat_bridge.py -q`

The wrapper preloads the test environment and replaces the environment loader
with a function returning `{}`, so this no-DB run neither reads `/etc` nor
contacts production. It is verified with 10 credential tests passing. Tests
whose fixture truly requires MariaDB are deferred to the one root-provided
isolated DB container and run with the same wrapper's `db` mode; they are not
silently treated as covered by the no-DB run.

Expected: FAIL until repository-derived ownership and `.3` bridging are wired.

- [ ] **Step 3: Implement canonical Platform ownership derivation and header emission.**

In `repositorio.py`, obtain tenant from the queue/project's canonical relation,
uploader user from the canonical row, and project ID from the same row; join the
user to tenant and reject a mismatch. In `despachador.py`, use that immutable
record for create/status/cancel/grouping/mutations, never request body
`usuario` field. In `credencial_las_manos.py`, attach the exact four headers only
when the Platform service credential is used. Do not make generic caller
headers configurable.

- [ ] **Step 4: Add the exact `.3` status bridge and tuple tests.**

Add the test to the existing Platform queue/bridge fixtures rather than
introducing a helper API. Its assertion sequence is fixed: create one real
queue SQL fixture with canonical owner IDs `1`/`2`/`3`; capture the four
authenticated headers; invoke the JAX Processing route; read the persisted
JSONL event; resolve F2-B with canonical
`PROCESSING_JOB_STATUS(processing_job_id,status)`; render the supported claim
template to real F2-C text; and revalidate that text through existing F2-D.
Assert the receipt predicate and exact scope digest at the real bridge result.

Also test malformed/missing/duplicate headers, tenant/user/project mismatch,
legacy ownerless Processing record, immutable re-create/update attempt,
status enum drift, stale source configuration, oversize/truncated JSONL line,
wrong F2-C/F2-D version, and the shared Project Documents eight-state Spanish
DTO without a job ID. The latter must not enter the Processing detector or
produce a receipt. The exact pair is F2-B claim → real text F2-C → F2-D;
it does not add a structured JSON F2-D path. The bridge must not select a
registry, resolver, source, template, envelope, or authority from a request.

- [ ] **Step 5: Run focused Platform tests and JAX/Platform exact pair.**

Run the no-DB focused Platform command from Step 2. When root provides the
isolated DB container, run the same command once with `db` in place of `nodb`
and the explicit fixture-requiring test paths selected by the test collector.
Then run the repository's existing exact-pair command discovered by:

```bash
cd /home/fruiz/worktrees/f2e-sr3-processing-platform
rg -n "exact-pair|PYTHONPATH=.*f2e|governed_chat_bridge" .github workflow* docs backend
```

Use the command found there unchanged, with JAX checkout fixed to
`/home/fruiz/worktrees/f2e-sr3-processing-jax`; record command, SHAs and
result in its test artifact. The required chain is real queue SQL →
authenticated headers → JAX route → Processing JSONL → F2-B receipt → F2-C/F2-D
bytes. A fake in-memory resolver alone does not satisfy this task.

- [ ] **Step 6: Finish implementation gates.**

Run `git diff --check` in both worktrees, ensure no unauthorized predicate or
MotorJobStore v1 change, and request a fresh Sol audit of the exact candidate
SHA after all code and evidence are frozen. The audit must explicitly review
the two fail-closed ownership boundaries, the `.3` registry closure, and the
real exact-pair chain. Do not merge, deploy, or alter the F2-E-SR load branch.

## Self-Review

- **Spec coverage:** Task 1 implements dedicated source, immutable owner
  headers, body-user prohibition, quarantine and source durability. Task 2
  implements the sole authorized predicate, `.3` registry, domain 6, SR-03,
  bounds and F2-B/F2-C/F2-D. Task 3 implements Platform queue ownership,
  service headers, owner-bound mutations and the real exact-pair chain.
- **No placeholders:** every planned file, interface, test behavior and command
  is concrete. The one intentionally conditional test path is resolved by an
  explicit `rg` command because the existing checkout may name its domain test
  module differently; it does not change the named behavior.
- **Type consistency:** Task 1 produces `ProcessingOwnershipContext`,
  `ProcessingJobStore` and its frozen owner snapshot; Task 2 consumes those
  exact contracts; Task 3 sends the same four header values and consumes only
  the `.3` predicate keys.
- **Review focus coverage:** header envelope/body authority is Task 1; legacy
  ownerless and exact three-part scope are Tasks 1–2; SR-03/native shape and
  all numeric bounds are Task 2.

## Execution Handoff

The plan is authorized for automatic implementation, but it is not itself an
implementation. Execute the three tasks in order; after Task 3, obtain a Sol
audit on the exact frozen candidate SHA before any integration action.

## Rollout Compatibility Note (2026-10-03)

Deploy JAX SR3 before Platform SR3. An old Platform dispatcher sends the
legacy body `usuario` field and no Processing owner headers; JAX SR3 rejects
that missing or invalid authenticated ownership envelope with the static
`403 processing_ownership_invalid` response before parsing the body or
appending a job. The deployed E2a dispatcher classifies that code as
`SIN_CULPA_DEL_DOCUMENTO` and retains the queue item for retry. It must not be
changed to a 400 document error and must not infer ownership or accept legacy
headers/body authority. After JAX SR3 is live, deploy Platform SR3, which emits
the four canonical headers and removes `usuario`; do not deploy new Platform
against old JAX because old JAX requires that legacy body field.
