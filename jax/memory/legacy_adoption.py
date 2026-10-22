"""Saved, private dry-run plans for controlled legacy adoption into B9.

No inference from content, no environment loading, no schema changes. The
persistent API revalidates source/authority inside the import transaction.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import stat
import uuid

from jax.memory.b9 import ObjectKind, Visibility, ScopeContext, MutationAuthorizationRequest
from jax.memory.mapping_pool import MappingPool

TABLE_KINDS = {"facts":ObjectKind.FACT,"decisions":ObjectKind.DECISION_MEMORY,"action_items":ObjectKind.ACTION_ITEM}
CONTENT_FIELDS = {"facts":("fact_text",),"decisions":("title","context","chosen_option","reasoning"),"action_items":("description",)}


def normalize(value):
    if isinstance(value, (datetime,date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", "strict")
    if isinstance(value, dict):
        return {str(k):normalize(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [normalize(v) for v in value]
    return value


def canonical(value):
    return json.dumps(normalize(value),sort_keys=True,separators=(",",":"),ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def legacy_content(table, row):
    if table == "decisions":
        return canonical({field:row.get(field) for field in CONTENT_FIELDS[table]})
    return str(row.get(CONTENT_FIELDS[table][0]) or "")


def snapshot_row(row):
    # Embeddings are derived data, not legacy ownership/content provenance.
    return normalize({k:v for k,v in row.items() if not k.startswith("embedding")})


def source_content(table, row):
    return legacy_content(table, row)


def row_digest(row):
    return digest(snapshot_row(row))


def source_eligible(table, row, *, observed_at=None):
    """Lifecycle eligibility; owner/authority are independently locked by API."""
    if table not in TABLE_KINDS or not source_content(table,row).strip():
        return False
    if table == "facts":
        if any(row.get(field) is not None for field in ("superseded_by","superseded_at","superseded_by_user")):
            return False
        if row.get("expires_at") is not None:
            expiry = row["expires_at"]
            if not isinstance(expiry,datetime):
                expiry = datetime.fromisoformat(str(expiry))
            if expiry <= (observed_at if observed_at is not None else datetime.now(tz=expiry.tzinfo)):
                return False
    if table == "action_items" and (row.get("status") != "pending" or row.get("completed_at") is not None):
        return False
    return True


def private_write(path, value):
    fd = os.open(os.fspath(path),os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os,"O_NOFOLLOW",0),0o600)
    with os.fdopen(fd,"w",encoding="utf-8") as out:
        out.write(canonical(value)+"\n")
        out.flush()
        os.fsync(out.fileno())


def private_read(path):
    fd = os.open(os.fspath(path),os.O_RDONLY | getattr(os,"O_NOFOLLOW",0))
    with os.fdopen(fd,"r",encoding="utf-8") as src:
        info=os.fstat(src.fileno())
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("adoption artifact must be a regular private 0600 file")
        return json.load(src)


def classify(table,row,users,projects,memberships,refs,*,observed_at):
    """Pure fail-closed owner/scope/lifecycle classification for the dry run."""
    uid=row.get("user_id")
    owner=users.get(str(uid)) if uid is not None else None
    if not owner or str(owner.get("status","")).lower() != "active" or owner.get("tenant_id") is None:
        return "OWNER_UNAVAILABLE",None
    tenant=owner["tenant_id"]
    project=row.get("project_id")
    if row.get("tenant_id") is not None and str(row["tenant_id"]) != str(tenant):
        return "TENANT_CONFLICT",None
    for ref in refs:
        if ref is None:
            return "REFERENCE_MISSING",None
        if str(ref.get("user_id")) != str(uid) or str(ref.get("tenant_id")) != str(tenant) or ref.get("project_id") != project:
            return "REFERENCE_SCOPE_CONFLICT",None
    if table=="facts":
        if any(row.get(field) is not None for field in ("superseded_by","superseded_at","superseded_by_user")):
            return "SUPERSEDED",None
        if row.get("expires_at") is not None:
            expires=datetime.fromisoformat(str(row["expires_at"]))
            # The snapshot carries DB NOW(), preserving the DB's clock/timezone.
            if expires <= observed_at:
                return "EXPIRED",None
    if table=="action_items" and (row.get("status") != "pending" or row.get("completed_at") is not None):
        return "ACTION_NOT_PENDING",None
    if not legacy_content(table,row).strip():
        return "EMPTY_CONTENT",None
    visibility=Visibility.USER_PRIVATE
    if project is not None:
        scope=projects.get(str(project))
        member=memberships.get((str(project),str(uid)))
        if not scope or str(scope.get("tenant_id"))!=str(tenant) or scope.get("status")!="ACTIVE":
            return "PROJECT_SCOPE_UNBOUND",None
        if not member or str(member.get("tenant_id"))!=str(tenant) or member.get("status")!="ACTIVE" or member.get("project_role") not in {"CONTRIBUTOR","REVIEWER","OWNER"}:
            return "PROJECT_MEMBERSHIP_UNAVAILABLE",None
        visibility=Visibility.PROJECT_SHARED
    return None,{"tenant_id":str(tenant),"user_id":str(uid) if project is None else None,"project_id":None if project is None else str(project),"visibility":visibility.value}


async def build_plan(raw_pool, namespace):
    if namespace != "legacy":
        raise ValueError("a stable legacy namespace is required")
    pool=MappingPool(raw_pool)
    async with pool.acquire() as conn:
        await conn.begin()
        try:
            async with conn.cursor() as cur:
                await cur.execute("SELECT NOW() observed_at")
                observed=(await cur.fetchone())["observed_at"]
                await cur.execute("SELECT user_id,tenant_id,status,role FROM jax_users ORDER BY user_id")
                users={str(r["user_id"]):normalize(r) for r in await cur.fetchall()}
                await cur.execute("SELECT project_id,tenant_id,status FROM jax_project_scope ORDER BY project_id")
                projects={str(r["project_id"]):normalize(r) for r in await cur.fetchall()}
                await cur.execute("SELECT project_id,user_id,tenant_id,status,project_role FROM jax_project_membership ORDER BY project_id,user_id")
                members={(str(r["project_id"]),str(r["user_id"])):normalize(r) for r in await cur.fetchall()}
                items=[]
                for table in TABLE_KINDS:
                    # Introspect schema, exclude only vector columns, never identifiers supplied by callers.
                    await cur.execute(f"SHOW COLUMNS FROM `{table}`")
                    columns=[r["Field"] for r in await cur.fetchall() if not str(r["Field"]).startswith("embedding")]
                    if any(not c.replace("_","").isalnum() for c in columns):
                        raise ValueError("unsupported source schema")
                    await cur.execute("SELECT "+",".join(f"`{c}`" for c in columns)+f" FROM `{table}` ORDER BY id")
                    rows=await cur.fetchall()
                    for raw in rows:
                        row=snapshot_row(raw)
                        refs=[]
                        if row.get("source_message_id") is not None:
                            await cur.execute("SELECT m.user_id,u.tenant_id,m.project_id,m.conversation_id FROM messages m LEFT JOIN jax_users u ON u.user_id=m.user_id WHERE m.id=%s",(row["source_message_id"],))
                            message=await cur.fetchone()
                            refs.append(normalize(message) if message else None)
                            if message and message.get("conversation_id") is not None:
                                await cur.execute("SELECT user_id,tenant_id,project_id FROM conversations WHERE id=%s",(message["conversation_id"],))
                                ref=await cur.fetchone();refs.append(normalize(ref) if ref else None)
                        if row.get("source_conversation_id") is not None:
                            await cur.execute("SELECT user_id,tenant_id,project_id FROM conversations WHERE id=%s",(row["source_conversation_id"],))
                            ref=await cur.fetchone();refs.append(normalize(ref) if ref else None)
                        reason,destination=classify(table,row,users,projects,members,refs,observed_at=observed)
                        items.append({"table":table,"legacy_id":str(row["id"]),"source_snapshot":row,"source_digest":digest(row),"content_digest":hashlib.sha256(legacy_content(table,row).encode()).hexdigest(),"references":refs,"destination":destination,"quarantine_reason":reason})
            await conn.rollback()  # read-only transaction; no persistent claims or writes.
        except BaseException:
            await conn.rollback()
            raise
    plan={"schema_version":1,"namespace":namespace,"observed_at":normalize(observed),"items":items}
    plan["plan_digest"]=digest(plan)
    return plan


def summary(plan):
    return {"total":len(plan["items"]),"eligible":sum(i["quarantine_reason"] is None for i in plan["items"]),"quarantine":dict(Counter(i["quarantine_reason"] for i in plan["items"] if i["quarantine_reason"]))}


def validate_plan(plan):
    if plan.get("schema_version")!=1 or not isinstance(plan.get("items"),list):
        raise ValueError("unsupported adoption plan")
    contents={k:v for k,v in plan.items() if k!="plan_digest"}
    if plan.get("plan_digest") != digest(contents):
        raise ValueError("adoption plan changed")
    seen=set()
    for item in plan["items"]:
        key=(item["table"],item["legacy_id"])
        if item["table"] not in TABLE_KINDS or key in seen or item["source_digest"]!=digest(item["source_snapshot"]):
            raise ValueError("invalid adoption source snapshot")
        seen.add(key)


def validate_backup(manifest):
    required={"facts","decisions","action_items","jax_users","messages","conversations","memory_objects","memory_revisions","memory_events","memory_projections","memory_legacy_bindings"}
    if manifest.get("restoration_verified") is not True or not manifest.get("restored_at") or not required.issubset(set(manifest.get("restored_tables",[]))):
        raise ValueError("backup restoration proof incomplete")
    if not isinstance(manifest.get("restored_counts"),dict) or any(name not in manifest["restored_counts"] for name in required):
        raise ValueError("backup restored counts incomplete")
    path=Path(manifest["backup_path"])
    if not path.is_file() or path.is_symlink() or path.stat().st_size!=manifest.get("backup_size") or path.stat().st_size<=0:
        raise ValueError("backup artifact missing or changed")
    sha=hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda:src.read(1024*1024),b""):
            sha.update(chunk)
    if sha.hexdigest()!=manifest.get("backup_sha256"):
        raise ValueError("backup artifact changed")


async def apply_plan(api,plan_path,backup_manifest_path,*,actor_user_id):
    plan=private_read(plan_path);validate_plan(plan)
    manifest=private_read(backup_manifest_path);validate_backup(manifest)
    result={"adopted":0,"quarantined":0}
    for item in plan["items"]:
        if item["quarantine_reason"]:
            result["quarantined"]+=1
            continue
        dest=item["destination"]
        scope=ScopeContext(f"user:{actor_user_id}","USER",str(actor_user_id),dest["tenant_id"],dest["project_id"],calling_component="legacy-adoption",request_id=str(uuid.uuid4()),trace_id=str(uuid.uuid4()))
        # The core API, not this plan, supplies authority and atomic binding.
        await api.import_legacy_memory(
            MutationAuthorizationRequest(scope,"IMPORT_LEGACY",Visibility.SYSTEM_INTERNAL),
            item["table"],plan["namespace"],item["legacy_id"],TABLE_KINDS[item["table"]],legacy_content(item["table"],item["source_snapshot"]),
            expected_source_digest=item["source_digest"],visibility=Visibility(dest["visibility"]),
            user_id=dest["user_id"],project_id=dest["project_id"])
        result["adopted"]+=1
    return result
