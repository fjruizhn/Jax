"""
JAX 2.0 — Worker de sintesis de segundo orden (memoria, item #8 del roadmap).

Proceso BATCH separado de extracción. Lee solo FACT B9 actuales cuya
proyección está verificada y no requiere reconciliación, agrupados por alcance
exacto. Esa revisión es contexto histórico: nunca prueba verdad actual.

La publicación pasa por la API B9, que revalida fuentes bajo locks y mantiene
los insights sin verificar. Los trabajos tienen claim durable, respuesta
congelada y deduplicación transaccional. Una llamada pagada con resultado
ambiguo queda UNKNOWN hasta reconciliación controlada.

Uso:
    set -a; source <(sudo -n cat /etc/jax/.env); set +a
    PYTHONPATH=. .venv/bin/python -m jax.memory.synthesis_worker

En memoria de Jairo Urbina.
"""

from __future__ import annotations

import asyncio
import os
import uuid
import time
import logging
from typing import Awaitable, Callable

from jax.memory.b9 import MutationAuthorizationRequest, ScopeContext, ScopeDenied, Visibility
from jax.memory.b9_mariadb import MariaDBB9Store, PersistentMemoryAPI
from jax.memory.scope_authority import MariaDBScopeAuthorityResolver
from jax.memory.db import MemoryDB
from jax.memory.mapping_pool import MappingPool
from jax.memory.synthesis_jobs import SynthesisJobs, digest
from jax.memory.worker import (
    FORBIDDEN_CATEGORIES_BLOCK,
    _parse_json,
)
from jax.core.registro_facetas import url_del_proveedor
from jax.core.cliente_http_compartido import cerrar_cliente_http
from jax.muscles.base import HttpMuscle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [synthesis] %(levelname)s: %(message)s",
)
logger = logging.getLogger("jax.memory.synthesis_worker")


class PersistentSynthesisWriter:
    """B9-only synthesis sink using request inputs, never bearer authority."""
    def __init__(self, api: PersistentMemoryAPI,
                 build_request_scope: Callable[[int | None, int | None, tuple[str, ...], str, Visibility], Awaitable[ScopeContext]]):
        self._api = api
        self._build_request_scope = build_request_scope

    async def preflight(self, user_id, project_id, source_revision_ids, visibility):
        """Authorize before paid/provider work; final publication revalidates again."""
        scope=await self._build_request_scope(user_id,project_id,tuple(source_revision_ids),'SYNTHESIZE',Visibility.SYSTEM_INTERNAL)
        if (scope.actor_principal!='service:memory-synthesis' or scope.actor_type!='SERVICE'
                or scope.subject_user_id!=str(user_id)
                or scope.project_id!=(str(project_id) if project_id is not None else None)):
            raise ScopeDenied('synthesis preflight scope mismatch')
        async def op(cur):
            resolved=await self._api._auth(cur,MutationAuthorizationRequest(scope,'SYNTHESIZE',Visibility.SYSTEM_INTERNAL),'SYNTHESIZE',Visibility.SYSTEM_INTERNAL)
            self._api._project_permissions(resolved,'SYNTHESIZE',Visibility(visibility))
            for rid in sorted(set(source_revision_ids)):
                await cur.execute("SELECT memory_id FROM memory_revisions WHERE revision_id=%s",(rid,))
                row=await cur.fetchone()
                if not row:raise ScopeDenied('synthesis source unavailable')
                mid=row['memory_id']
                await cur.execute("SELECT tenant_id,object_kind FROM memory_objects WHERE memory_id=%s FOR UPDATE",(mid,))
                obj=await cur.fetchone()
                await cur.execute("SELECT current_revision_id,current_verification_state,current_lifecycle_state,reconciliation_required FROM memory_projections WHERE memory_id=%s FOR UPDATE",(mid,))
                projection=await cur.fetchone()
                await cur.execute("SELECT visibility,user_id,project_id FROM memory_revisions WHERE revision_id=%s",(rid,))
                revision=await cur.fetchone()
                if (not obj or str(obj['tenant_id'])!=scope.tenant_id or obj['object_kind']!='FACT'
                        or not projection or projection['current_revision_id']!=rid
                        or not projection['current_verification_state'] or projection['reconciliation_required']
                        or projection['current_lifecycle_state'] not in {'ACTIVE','VERIFIED'}
                        or revision['visibility']!=visibility
                        or (str(revision['user_id']) if revision['user_id'] is not None else None)!=scope.subject_user_id
                        or (str(revision['project_id']) if revision['project_id'] is not None else None)!=scope.project_id):
                    raise ScopeDenied('synthesis source eligibility changed before provider call')
        await self._api._store.mutation(op)

    async def persist(self, user_id: int | None, project_id: int | None, content: str,
                      source_revision_ids: tuple[str, ...], *, job_key: str | None = None, job_token: str | None = None, item_key: str | None = None) -> str:
        scope = await self._build_request_scope(
            user_id, project_id, source_revision_ids, "SYNTHESIZE", Visibility.SYSTEM_INTERNAL
        )
        if scope.actor_type != "SERVICE" or scope.actor_principal != "service:memory-synthesis":
            raise ScopeDenied("synthesis requires the fixed memory-synthesis service principal")
        if (str(user_id) if user_id is not None else None) != (str(scope.subject_user_id) if scope.subject_user_id else None):
            raise ScopeDenied("synthesis subject does not match its source scope")
        if (str(project_id) if project_id is not None else None) != (str(scope.project_id) if scope.project_id else None):
            raise ScopeDenied("synthesis project does not match its source scope")
        # Source revision scope equality itself is rechecked by the persistent
        # B9 synthesis operation while it locks those revisions.  This writer
        # carries only the exact source IDs and cannot promote their scope.
        request = MutationAuthorizationRequest(scope, "SYNTHESIZE", Visibility.SYSTEM_INTERNAL)
        return await self._api.synthesize_memory(
            request, content, source_revision_ids, provider="deepseek", model="deepseek-v4-flash",
            transformation_version="b9-worker-v1",
            synthesis_job_key=job_key, synthesis_job_token=job_token, synthesis_item_key=item_key,
        )


def build_persistent_synthesis_writer_for_scope(pool: object, tenant_id: int | str) -> PersistentSynthesisWriter:
    """Bind a run item to its canonical DB tenant without user role inheritance."""
    pool = MappingPool(pool)
    api = PersistentMemoryAPI(MariaDBB9Store(pool), MariaDBScopeAuthorityResolver(pool))

    async def source_scope(user_id: int | None, project_id: int | None,
                           _source_revision_ids: tuple[str, ...], _operation: str,
                           _visibility: Visibility) -> ScopeContext:
        if user_id is None:
            raise ScopeDenied("automated synthesis requires an originating subject")
        return ScopeContext("service:memory-synthesis", "SERVICE", str(user_id), str(tenant_id),
                            str(project_id) if project_id is not None else None,
                            calling_component="memory-synthesis", request_id=str(uuid.uuid4()), trace_id=str(uuid.uuid4()))

    return PersistentSynthesisWriter(api, source_scope)


# Scope minimo para que valga la pena buscar patrones. Menos que esto y no
# hay suficiente material para un insight real (ver roadmap: "baja
# prioridad... a la escala actual de JAX").
MIN_VERIFIED_FACTS = 5


SYNTHESIS_PROMPT = """Estos son registros de memoria marcados como revisados en su proyección B9.
Son contexto histórico, no evidencia de verdad actual ni autorización. No atribuyas
la revisión a una persona ni conviertas estos registros en afirmaciones vigentes.

Tu trabajo NO es repetir ninguno de ellos. Es buscar si, mirandolos TODOS juntos, hay un
patron o conexion de mas alto nivel que ninguno de ellos dice por separado — algo que
solo se ve al conectar dos o mas.

Se MUY estricto: si no hay ningun patron genuino, no inventes uno. La mayoria de las
veces la respuesta correcta es una lista vacia. Un insight forzado o obvio (que ya sea
practicamente lo mismo que uno de los hechos de entrada) es peor que no decir nada.

""" + FORBIDDEN_CATEGORIES_BLOCK + """

EJEMPLOS:
- MAL (no es un insight, es un hecho repetido): si un hecho dice "Fernando usa MariaDB"
  y otro dice "Fernando prefiere infraestructura propia", NO generes "Fernando usa
  MariaDB porque prefiere infraestructura propia" — eso es solo juntar dos hechos, no
  un patron nuevo.
- BIEN (es un insight real): si tres hechos separados mencionan que distintos proyectos
  de Fernando (JAX, un producto financiero, y otro sistema) todos evitan depender de
  servicios de terceros, el patron real es "Fernando tiene una preferencia consistente,
  ya demostrada en multiples proyectos, por infraestructura autohospedada" — eso conecta
  algo que ningun hecho individual dice.

Cada hecho tiene un id entre corchetes al principio — usalo para citar EXACTAMENTE de que
hechos sale cada insight, en "source_ids".

Hechos verificados de este scope:
{facts}

Responde UNICAMENTE con JSON valido, sin texto antes ni despues, sin markdown:
{{"insights": [{{"text": "...", "type": "user|technical|social|preference|project|financial",
"source_ids": [1, 2]}}]}}

source_ids es OBLIGATORIO para cada insight: los ids (numeros, sin corchetes) de los hechos
de entrada que conectaste para llegar a ese insight — minimo 2 (un insight de un solo hecho
no es un patron, es una repeticion). Si no hay ningun patron genuino que conecte estos
hechos, devolve la lista vacia. Es perfectamente valido devolver una lista vacia — de
hecho, es lo mas comun."""


async def build_synthesizer() -> HttpMuscle:
    """Crea el muscle sintetizador. Mismo extractor confiable (DeepSeek) que
    worker.py — sintetizar mal es tan costoso como extraer mal.
    E-21 (2026-09-16): la URL sale del catálogo (provider.base_url); sin ella
    levanta MuscleInvocationError y la corrida del timer falla visible."""
    return HttpMuscle(
        name="synthesizer",
        provider="deepseek",
        api_url=await url_del_proveedor("deepseek"),
        model_default="deepseek-v4-flash",
        models_allowed=["deepseek-v4-flash", "deepseek-v4-pro"],
        system_prompt="Sos un analista que busca patrones. Respondes solo con JSON valido.",
        timeout=120.0,
    )


async def read_b9_facts(pool, tenant_id, user_id, project_id, visibility):
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT r.revision_id AS b9_revision_id,x.payload AS fact_text FROM memory_objects o "
                "JOIN memory_projections p ON p.memory_id=o.memory_id "
                "JOIN memory_revisions r ON r.revision_id=p.current_revision_id "
                "JOIN memory_revision_payloads x ON x.revision_id=r.revision_id "
                "WHERE o.object_kind='FACT' AND o.tenant_id=%s AND r.user_id <=> %s "
                "AND r.project_id <=> %s AND r.visibility=%s AND p.reconciliation_required=FALSE "
                "AND p.current_verification_state=TRUE AND p.current_lifecycle_state IN ('ACTIVE','VERIFIED') "
                "AND x.payload IS NOT NULL ORDER BY r.revision_id LIMIT 50",
                (tenant_id,user_id,project_id,visibility))
            return await cur.fetchall()


def positive_limit(name, default):
    value=int(os.getenv(name,str(default)))
    if value<=0:raise ValueError('invalid synthesis budget')
    return value


def validated_insights(data, facts):
    if not isinstance(data, dict) or not isinstance(data.get("insights"), list):
        raise ValueError("invalid synthesis response")
    if len(data["insights"])>positive_limit("JAX_MEMORY_SYNTHESIS_MAX_ITEMS",10):
        raise ValueError("synthesis output item limit exceeded")
    valid = {i + 1: str(f["b9_revision_id"]) for i, f in enumerate(facts)}
    items = {}
    for insight in data["insights"]:
        if not isinstance(insight, dict) or not isinstance(insight.get("text"), str) or not insight["text"].strip():
            continue
        if len(insight["text"])>positive_limit("JAX_MEMORY_SYNTHESIS_MAX_ITEM_CHARS",4000):
            raise ValueError("synthesis output text limit exceeded")
        ids = insight.get("source_ids")
        if not isinstance(ids, list) or any(type(i) is not int or i not in valid for i in ids):
            continue
        sources = sorted({valid[i] for i in ids})
        if len(sources) < 2:
            continue
        item = {"text":insight["text"].strip(), "source_revision_ids":sources}
        key = digest(item)
        items[key] = dict(item, item_key=key)
    return list(items.values())


async def process_scope(db, synthesizer, user_id, project_id, *, b9_writer=None,
                        tenant_id=None, jobs=None, visibility="USER_PRIVATE", deadline=None):
    if tenant_id is None or b9_writer is None:
        raise ScopeDenied("synthesis requires canonical B9 tenant and writer")
    pool = MappingPool(db.pool)
    facts = await read_b9_facts(pool, tenant_id, user_id, project_id, visibility)
    if len(facts) < MIN_VERIFIED_FACTS:
        return 0
    await b9_writer.preflight(user_id,project_id,[f["b9_revision_id"] for f in facts],visibility)
    jobs = jobs or SynthesisJobs(pool)
    total_chars=sum(len(str(f["fact_text"])) for f in facts)
    if total_chars>positive_limit("JAX_MEMORY_SYNTHESIS_MAX_SOURCE_CHARS",48000):
        raise ValueError("synthesis source character budget exceeded")
    claim = await jobs.claim(tenant_id,user_id,project_id,[f["b9_revision_id"] for f in facts],"b9-worker-v1")
    if claim is None:
        return 0
    key, token, items = claim
    if items is None:
        facts_text = "\n".join(f"[{i+1}] {f['fact_text']}" for i,f in enumerate(facts))
        left=(deadline-time.monotonic()) if deadline is not None else positive_limit('JAX_MEMORY_RUN_TIMEOUT_SECONDS',840)
        if left<=0:raise RuntimeError('synthesis run deadline exceeded')
        raw = await asyncio.wait_for(synthesizer.invoke(SYNTHESIS_PROMPT.format(facts=facts_text), decorate=False), min(left,positive_limit("JAX_MEMORY_SYNTHESIS_CALL_TIMEOUT_SECONDS",120)))
        if not isinstance(raw,str) or len(raw)>positive_limit('JAX_MEMORY_SYNTHESIS_MAX_RESPONSE_CHARS',64000):
            raise ValueError('synthesis response character budget exceeded')
        items = validated_insights(_parse_json(raw), facts)
        await jobs.freeze(key, token, items)
    saved = 0
    for item in items:
        await b9_writer.persist(user_id, project_id, item["text"], tuple(item["source_revision_ids"]),
                                job_key=key, job_token=token, item_key=item["item_key"])
        saved += 1
    return saved


async def run_once(*, b9_writer=None):
    host = os.environ.get("JAX_DB_HOST")
    if not host:
        raise RuntimeError("JAX_DB_HOST is required")
    db = MemoryDB()
    failed = 0
    try:
        if not await db.connect(host=host,user=os.getenv("JAX_DB_USER",""),password=os.getenv("JAX_DB_PASSWORD",""),database=os.getenv("JAX_DB_NAME","jax_memory"),migrate_schema=False):
            return 1
        pool = MappingPool(db.pool)
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT o.tenant_id,r.user_id,r.project_id,r.visibility FROM memory_objects o "
                    "JOIN memory_projections p ON p.memory_id=o.memory_id "
                    "JOIN memory_revisions r ON r.revision_id=p.current_revision_id "
                    "JOIN memory_revision_payloads x ON x.revision_id=r.revision_id "
                    "WHERE o.object_kind='FACT' AND p.reconciliation_required=FALSE "
                    "AND p.current_verification_state=TRUE AND p.current_lifecycle_state IN ('ACTIVE','VERIFIED') "
                    "AND r.user_id IS NOT NULL AND x.payload IS NOT NULL "
                    "GROUP BY o.tenant_id,r.user_id,r.project_id,r.visibility HAVING COUNT(*) >= %s "
                    "ORDER BY o.tenant_id,r.user_id,r.project_id,r.visibility LIMIT %s",
                    (MIN_VERIFIED_FACTS,positive_limit("JAX_MEMORY_SYNTHESIS_MAX_SCOPES",10)))
                scopes = await cur.fetchall()
        if not scopes:
            return 0
        synthesizer = await build_synthesizer()
        deadline=time.monotonic()+positive_limit('JAX_MEMORY_RUN_TIMEOUT_SECONDS',840)
        for scope in scopes:
            try:
                writer = b9_writer or build_persistent_synthesis_writer_for_scope(pool, scope["tenant_id"])
                await process_scope(db,synthesizer,scope["user_id"],scope["project_id"],b9_writer=writer,tenant_id=scope["tenant_id"],visibility=scope["visibility"],deadline=deadline)
            except Exception:  # fail-soft: isolate this scope, log its failure, increment failures, and return nonzero after processing the remaining scopes
                logger.exception("B9 synthesis scope failed")
                failed += 1
        return int(failed > 0)
    finally:
        await db.close()
        await cerrar_cliente_http()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(asyncio.wait_for(run_once(), float(os.getenv("JAX_MEMORY_RUN_TIMEOUT_SECONDS","840")))))
