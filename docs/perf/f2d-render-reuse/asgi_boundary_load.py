import asyncio
import hashlib
import json
import math
import os
import statistics
import time

from api.chat import ChatResponse, ContractResult
from api.governed_chat import project_provider_contract
from auth.models import AuthUser
from jax.memory.b9 import ScopeContext
from webchat_f2d.transport import prepare_governed_chat_response
from webchat_f2d import repository as outbox


class _FakeRepository:
    """In-memory F2-D outbox test seam; no database or external service."""
    def __init__(self):
        self.state = None
        self.payload = None

    async def prepare(self, unit, payload, *, tenant_id, project_id, subject_id, request_id,
                      previous_attempt_id=None):
        self.payload = payload
        projection = unit.durable_projection()
        self.authorization = outbox.PreparedTransportAuthorization._mint(
            outbox._AUTH_TOKEN,
            outbox_id="benchmark-outbox", attempt_id="benchmark-attempt",
            tenant_id=tenant_id, scope_digest=projection["scope_digest"],
            request_id=request_id, response_id=projection["response_id"],
            subject_id=subject_id, idempotency_key=projection["idempotency_key"],
            effective_output_digest=projection["effective_output_digest"],
            effective_projection_digest=projection["effective_projection_digest"],
            original_envelope_digest=projection["original_envelope_digest"],
            contract_state=projection["effective_contract_state"],
            transport_payload_digest="sha256:" + hashlib.sha256(payload).hexdigest(),
            payload=payload, unit=unit,
        )
        self.state = "OUTPUT_PREPARED"
        return self.authorization

    async def transition(self, authorization, target, *, failure_class=None, before_send=False):
        core = __import__("api.governed_chat", fromlist=["_lifecycle_core"])
        current = core._lifecycle_core().OutputLifecycleState(self.state)
        core._lifecycle_core().validate_lifecycle_transition(current, target,
            before_send=before_send)
        self.state = target.value

    async def record_secondary_event(self, authorization, event_type):
        return None

BASE_TEXT = "Respuesta de carga sobre planificacion y presupuesto del trimestre. "
TEXT = (BASE_TEXT * (16000 // len(BASE_TEXT) + 1))[:16000]
CONCURRENCIES = [1, 5, 10, 25]
REQUESTS_PER_LEVEL = int(os.environ.get("F2D_LOAD_REQUESTS", "100"))

async def noop():
    return None

async def one_request():
    started = time.perf_counter()
    # Fake provider latency, matching the existing harness's local HTTP fake
    # (sub-millisecond to low-millisecond) without model/GPU use.
    await asyncio.sleep(0.001)
    scope = ScopeContext(actor_principal="user:7", actor_type="USER", subject_user_id="7",
        tenant_id="1", project_id=None, calling_component="jax-platform-web-chat")
    contract = ContractResult(True, [], TEXT, None, None, TEXT)
    governed = project_provider_contract(contract, memory_scope=scope, user_id="7")
    if governed.transport_unit is None:
        raise RuntimeError("producer did not return governed transport unit")
    response = ChatResponse(facet="jax_local", response=governed.text,
        timestamp="2026-10-03T00:00:00Z", response_id=governed.response_id,
        envelope_digest=governed.envelope_digest,
        source_envelope_digest=governed.source_envelope_digest,
        contract_state=governed.contract_state, contract_degraded=governed.contract_degraded,
        governed_plain=True)
    user = AuthUser(user_id="7", tenant_id="1", role="operator", token_version=0)
    repo = _FakeRepository()
    prepared = await prepare_governed_chat_response(response=response,
        transport_unit=governed.transport_unit, user=user, memory_scope=scope,
        on_commit=noop, trusted_metadata={"facet": response.facet, "timestamp": response.timestamp,
            "contract_degraded": response.contract_degraded}, repository=repo)
    messages = []
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}
    async def send(message):
        messages.append(message)
    await prepared({"type": "http", "method": "POST", "path": "/api/chat"}, receive, send)
    bodies = [m["body"] for m in messages if m["type"] == "http.response.body"]
    if repo.state != "OUTPUT_COMMITTED_TO_TRANSPORT" or not bodies:
        raise RuntimeError(f"F2-D ASGI path failed: {repo.state}")
    body = json.loads(bodies[-1])
    if body["response"] != governed.text or len(governed.text) < 16000:
        raise RuntimeError("response did not preserve the full governed 16 KB output")
    return (time.perf_counter() - started) * 1000

async def run_level(concurrency):
    started = time.perf_counter()
    latencies = []
    workers = min(concurrency, REQUESTS_PER_LEVEL)
    per_worker = math.ceil(REQUESTS_PER_LEVEL / workers)
    async def worker():
        for _ in range(per_worker):
            latencies.append(await one_request())
    await asyncio.gather(*(worker() for _ in range(workers)))
    elapsed = time.perf_counter() - started
    latencies.sort()
    p95 = latencies[min(len(latencies) - 1, math.ceil(len(latencies) * .95) - 1)]
    print(json.dumps({"concurrency": concurrency, "requests": len(latencies),
        "rps": round(len(latencies) / elapsed, 2), "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(p95, 2), "max_ms": round(latencies[-1], 2),
        "elapsed_s": round(elapsed, 2)}), flush=True)

async def main():
    for concurrency in CONCURRENCIES:
        await run_level(concurrency)

asyncio.run(main())
