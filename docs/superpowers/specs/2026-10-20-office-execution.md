# Office execution through Jacobs

2026-10-20 · DECISIÓN técnica Codex, authorized Office development by Fernando. Architecture source: readonly Sol office_workflow_contract, independently inspected executor save_if_large, direct HTTP payloads, service credentials, cupo and preflight paths. Not an enabled capability or production claim. JXP domain contract remains its office-workflow.md.

## Ownership and identity

Reuse existing run_pipeline, cupo, epochs/CAS, preflight/catalog/credential/output limits and usage_writer. No second executor/queue/provider client. Core stores opaque job/stage/project IDs, hashes, governed billing/model metadata and JXP receipts. Prompt/output live only in memory during provider transport and in authorized Office storage in JXP. Never persist raw Office content in classic Step.input/input_ref/context/output_ref/artifacts/repo/events/logs or exception text.

Pure jacobs/office_contract.py defines OFFICE_EXECUTION_CONTRACT=OFFICE_V1, OFFICE_WORK_CONTRACT_VERSION=office-work.1, strict OfficeRunRequest and immutable Office runtime policy. Request fields: exact contract_version, canonical UUID office_job_id/office_stage_id/client_request_id, positive project_id/tenant_id/usage_user_id, executable configured facet/capability, timeout_seconds from execution policy, required accepted Decimal USD cost, lowercase 64hex request_sha256. Reject extra/unknown types. No prompt/objective/references/provider/model/URL/recipientgroup/authorizedactor/name/mode/plan from request. Trusted plataforma supplies metadata after canonical JXP job authorization; these fields are accounting/binding claims, not Office authority.

Core recomputes canonical request digest over every field except digest, using existing canonical USD representation. Server fixes invoked_by=plataforma, mode=autonomous, max_steps=1, name and empty context. Deterministic UUIDv5 step/correlation IDs derive from pipeline ID + contract + index0, stable across safe retries.

## Persistence and reservation

Extend jacobs_pipelines, not a new Office queue: execution_contract NOTNULL DEFAULT CLASSIC; nullable office_job_id, office_stage_id, office_project_id, client_request_id, request_sha256, office_phase, office_rejection_code. Strict discriminator/binding CHECKs; unknown fails closed. Unique(invoked_by,tenant_id,client_request_id), unique(execution_contract,tenant_id,office_stage_id), indexed(execution_contract,office_phase,updated_at). Classic NULL metadata remains compatible. Atomic cupo INSERT...SELECT includes idempotency binding in SAME statement; no claim/cupo split race.

Phases: RESERVED → PREFLIGHT_IN_FLIGHT → READY → DELIVERY_IN_FLIGHT → RECEIPT_STORED, terminal FAILED/CANCELLED_REVOKED/OUTCOME_UNKNOWN. CAS RESERVED→PREFLIGHT_IN_FLIGHT immediately before preflight; unknown paid probe after crash is not automatically retried. Known rejection retained durably as failed/REJECTED, releases cupo, replay same rejection. Delivery CAS requires pipeline+epoch+running pipeline+running step and refreshes updated_at so stale reaper cannot win. Finish step+phase atomically, same epoch fencing.

## API and recovery

POST /jacobs/office-v1/runs additionally requires authenticated servicio identity plataforma (request.state.identidad_servicio), not any service credential. Disabled by default; missing valid configured JXP callbackbase/auth/pair gate fails closed. No input URL. Responses201new,200exactreplay,202inprogress,409key/hashconflict,422known preflight/cost rejection,503disabled/preflightunknown/commitunknown. Replaying known READY/pending can schedule existing run_pipeline again; existing pending→running CAS permits one winner, closing commit-before-background-scheduling gap. Unknown preflight/provider/delivery never blindly reruns generation.

All classic get/results/events/cancel/discard/recover/hide/restore/resume/continue/preflight/approve-step routes, relaunch CLI and list/visible SQL exclude non-CLASSIC, including JXP direct pipelines/admin-hidden SQL later. owner_ack_at is not isolation. Classic visibility expression/index includes execution_contract CLASSIC. Office events contain IDs, phase and stablecodes only. New capability stays disabled until JXP filters/callbacks/storage/pair/integration all exist.

## JXP callbacks

Configured base only, shared HTTP client, existing encabezado_propio(IDENTIDAD_JACOBS). No new credential family. Exact service identity Jacobs checked by JXP.

Materialize POST /internal/virtual-office/stages/{stage}/materialize: contract_version, phase PREFLIGHT|EXECUTION, pipeline_id, run_epoch, step_id, correlation_id, request_sha256. No actor/group/tenant supplied as authority. JXP locks stage/job and canonical current authority, derives actor/op/tenant/project/group from its DB, verifies/sets binding idempotently, resolves exact source versions. Returns MATERIALIZED prompt+materialization_sha256+binding_version or CANCELLED_REVOKED. EXECUTION compares prior preflight binding/hash; changed inputs fail INPUT_STALE. Byte limits from configured technical policy/catalog, no invented commercial quota. Core builds transient Step copy only, never upserts prompt.

Complete POST .../complete: same binding IDs+contract/request hash, output_sha256, untrusted exact output_json, modelo_real. JXP reauthorizes after LLM before effect; idempotency(stage,pipeline,epoch,step,outputdigest), exact replay same receipt, differentdigest409, revoked no artifact/message. Client response only MODEL/CLIENT_GROUP bound group, never publication. Receipt STORED|CANCELLED_REVOKED, receipt_id, effect_kind/effect_id, output_sha256, office_stage_version. Replace raw output with receipt BEFORE save_if_large; always omit classic repo copy/notification. Finish Office directly before classic devolucion/arbiter; no Office continue/resume/Hyde approval.

Fail POST .../fail: binding IDs, phase, stablecode, model_invoked bool and optional outputdigest still in memory. Never body/error/prompt/output. Auth+binding required; recording truthful failure history does not require actor still authorized. Existing matching completed receipt returns ALREADY_COMPLETED+receipt, core marks success without second effect. Otherwise FAILED/CANCELLED_REVOKED/OUTCOME_UNKNOWN. Unknown complete then unknown fail retains code only; JXP reconciles durable receipt/opaque state, never regenerates automatically.

## No-tools runtime policy

OFFICE_V1 means direct generation without tools and ONE generation invocation per stage. Repair is another explicit authorized/budgeted stage, not hidden retry. Immutable policy derived from pipeline discriminator, never browser/body/facet.params: tool_mode NONE; generation_calls1; allowed transports http_gemini/http_openai_compat/ollama; error_detail STABLE_CODE_ONLY. Configured facet/model/provider/persona/URL/credential still resolved from current catalog; incompatible transport fails OFFICE_TRANSPORT_NOT_ALLOWED, never silently changes provider. Validate after catalog resolution BEFORE probe and again after resolve_facet BEFORE provider call.

Classic runtime_policy=None behavior stays identical. Gemini Office payload omits tools/google_search, no grounding retry or source redirect resolution, sources/queries empty, groundedFalse. Classic Gemini keeps existing grounding/retry. OpenAI/Ollama Office final payloads must contain no tools/tool_choice/functions/function_call. Explicitly reject Motor/subprocess/unknown before invocation. Reuse shared client, output limits, resolution metadata and actual usage writer.

Preflight/runtime_policy optional keyword propagates through evaluar_paso/llamadas_max/probe. Office generation maxcalls1; existing health probe separate and accounted, no-tools. Probe coalescing key(facet,error_detail_policy) prevents classic detailed errors crossing into Office. Office probe stores stablecodes only; classic redacted detail unchanged. Paid probe crash leaves PREFLIGHT_IN_FLIGHT unknown, requiring explicit new decision/key, no auto repeat.

OfficeExecutionError has enumerated code, phase PREFLIGHT|GENERATION|DELIVERY, outcome FAILED|OUTCOME_UNKNOWN, provider_call_made bool. Map inside invokers BEFORE persistence/logging, raise fromNone, never str(exc)/body/URL/prompt/output/chained traceback. Codes: OFFICE_RUNTIME_POLICY_DENIED, OFFICE_PROVIDER_REJECTED/UNAVAILABLE/TIMEOUT/RESPONSE_INVALID, OFFICE_OUTPUT_TRUNCATED, OFFICE_PREFLIGHT_FAILED, OFFICE_DELIVERY_OUTCOME_UNKNOWN, OFFICE_EXECUTION_FAILED. Logs/events only correlation IDs/phase/code/exceptionclass/HTTPfamily. Timeout after send unknown, no auto retry. Facet health event Office details stable provider_http_4xx/5xx/provider_timeout/transport_error/config_error. Usage metadata/spool contains no content, preserves actual cost even post-response revocation.

## Delivery slices and required evidence

4A pure contracts/schema/atomic reserve/replay/CAS/classic visibility. 4B no-tools/opaque provider+probe policy. 4C callbacks/executor dedicated route/binding receipts. Single core writer per slice, root exactSHA review between meaningful checkpoints; disabled until all plus JXP gate. No production/provider spending in tests.

Tests: real DB two submits→one pipeline+cupo; samekey/hash replay and otherhash409; schema upgrade oldclassic preserved; epoch stale no complete; pre/postmodel revocation blocks/discards while usage retained; delivery ambiguity/fail reconciliation; sentinel prompt/output never appears DB/artifact/repo/events/logs even jacobs/httpx/httpcore DEBUG; save_if_large sees receipts only; cross tenant/project/stage binding denies; no-tools all3 final payloads and Gemini1call; Motor/subprocess denied BEFORE provider/probe; body secrets absent step/error/health/log; all classic routes/list/relaunch reject Office. Actual EXPLAIN indexes and load before enabling.

Existing test anchors: jacobs/_cupo_io_test.py, _store_indexes_test.py, _http_facet_admission_test.py, _direct_usage_test.py, _usage_writer_test.py; tests/test_run_epoch_db.py and test_las_manos_auth_servicio.py. JXP par/test credential counterparts after backend current tasks. Contract markers alone do not prove activation readiness.

## Task4A contract refinement · 2026-10-20

DECISIÓN técnica Codex following readonly Sol office_execution_contract_refinement and independent source inspection of legacy DECIMAL12,4, prevuelo_reglas.formatear_usd and PipelineStatus/cupo partition.

Office budget uses permanent subtype field office_accepted_cost_usd DECIMAL(14,6), one active budget authority per row: CLASSIC requires Office field NULL; OFFICE_V1 requires exact nonnegative Office budget and legacy costo_max_aceptado_usd NULL. Same atomic reservation INSERT persists consent. Preserve legacy Classic column and precision unchanged. Technical14,6 representation preserves eight integer digits, no commercial quota. Reject nonfinite/negative/overrange/>6fractional digits BEFORE canonical formatear_usd; no rounding consent upward. Equivalent1/1.0/1.000000 have same digest. Wire office-work.1 maps exactly to durable OFFICE_V1, no alias.

Generic allowed phase edges: RESERVED→PREFLIGHT_IN_FLIGHT|FAILED|CANCELLED_REVOKED; PREFLIGHT_IN_FLIGHT→READY|FAILED|CANCELLED_REVOKED|OUTCOME_UNKNOWN; READY→DELIVERY_IN_FLIGHT|FAILED|CANCELLED_REVOKED; DELIVERY_IN_FLIGHT→RECEIPT_STORED|FAILED|CANCELLED_REVOKED|OUTCOME_UNKNOWN. No terminal exits. Any later UNKNOWN reconciliation needs separate proof-bearing receipt API. Derive targetstatus in store, not callerchosen. Nonterminal running/codeNULL, RECEIPT_STORED completed/codeNULL, FAILED failed/coderequired, CANCELLED_REVOKED aborted/coderequired, OUTCOME_UNKNOWN new truthful terminal outcome_unknown/no cupo/coderequired. DB checks enforce snapshots with explicitNOTNULL and COALESCE false; CAS enforces history/epoch.

Request timeout positive only in4A;4B/C must use current catalog capability ceiling through plan._techo_segundos before probe/provider, including canonical fallback300. No static3600. Strict request types and immutable policy reject bool/coercion/mutable-set bypasses; duplicate lookup uses held cursor to avoid pool starvation. DirectSQL adversarialmatrix and poolsize1 tests required.
