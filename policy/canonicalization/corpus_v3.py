"""Read-only candidate corpus validation for C14N/3."""
from __future__ import annotations
import hashlib, os, stat
from pathlib import Path, PurePosixPath
from .bootstrap_v3 import verified_bootstrap_v3, CANONICALIZER_IDENTITY
from .canonical_json import canonical_json_bytes
from .errors import SchemaValidationError
from .strict_yaml import load_strict_yaml
from .schemas_v3 import validate_authority,validate_manifest,validate_document
from .projection_v3 import normative_projection_v3
def _root(p): return Path(os.path.abspath(p))
def _safe(root, raw):
 if not isinstance(raw,str) or "\\" in raw or raw.startswith("/") or any(x in {"",".",".."} for x in raw.split("/")) or any(c in raw for c in "*?["): raise SchemaValidationError("locator inseguro")
 path=root/PurePosixPath(raw)
 try: mode=path.lstat().st_mode
 except FileNotFoundError as e: raise SchemaValidationError("locator ausente") from e
 if stat.S_ISLNK(mode) or not stat.S_ISREG(mode): raise SchemaValidationError("locator no regular")
 return path
def _digest(proj): return "sha256:"+hashlib.sha256(b"JAX-POLICY-CORPUS\0"+CANONICALIZER_IDENTITY.encode("ascii")+b"\0"+canonical_json_bytes(proj)).hexdigest()
def validate_candidate_corpus(repo_root):
 root=_root(repo_root); b=verified_bootstrap_v3(); bundle=b.bundle
 a=load_strict_yaml(_safe(root,bundle["fixed_root_locators"]["authority_meta_contract"])); m=load_strict_yaml(_safe(root,bundle["fixed_root_locators"]["authoritative_manifest"]))
 validate_authority(a,b); validate_manifest(m,b)
 if a["root_pair"]!={"authoritative_manifest_id":m["id"],"authoritative_manifest_kind":m["kind"]} or m["governed_by"]!={"authority_meta_contract_id":a["id"],"authority_meta_contract_kind":a["kind"]}: raise SchemaValidationError("root pair mismatch")
 if a["scope"]["jurisdiction"]!=m["corpus"]["jurisdiction"]: raise SchemaValidationError("jurisdiction mismatch")
 docs=[]; ids=set(); paths=set(); refs=set(); locs=set()
 for r in m["reference_documents"]:
  if r["reference_id"] in refs or r["source_locator"] in locs: raise SchemaValidationError("reference duplicate")
  refs.add(r["reference_id"]); locs.add(r["source_locator"]); _safe(root,r["source_locator"])
 for member in m["normative_documents"]:
  if member["id"] in ids or member["path"] in paths: raise SchemaValidationError("member duplicate")
  ids.add(member["id"]); paths.add(member["path"]); d=load_strict_yaml(_safe(root,member["path"])); validate_document(d,b)
  if member["id"]!=d["id"] or member["document_class"]!=d["document_class"] or member["normative_layer"]!=d["normative_layer"] or member["normative_effect"]!="ACTIVE_WHEN_CORPUS_ACTIVE": raise SchemaValidationError("member mismatch")
  docs.append((d,member))
 byid={d["id"]:d for d,_ in docs}; edges={x:[] for x in byid}
 for d,_ in docs:
  for target in d["relationships"]["supersedes"]:
   if target==d["id"] or target not in byid or d["id"] not in byid[target]["relationships"]["superseded_by"] or d["document_class"]!=byid[target]["document_class"]: raise SchemaValidationError("supersession invalid")
   edges[d["id"]].append(target)
  for target in d["relationships"]["superseded_by"]:
   if target==d["id"] or target not in byid or d["id"] not in byid[target]["relationships"]["supersedes"]: raise SchemaValidationError("relationship invalid")
 def visit(n,seen,done):
  if n in seen: raise SchemaValidationError("supersession cycle")
  if n not in done:
   seen.add(n); [visit(x,seen,done) for x in edges[n]]; seen.remove(n); done.add(n)
 done=set(); [visit(x,set(),done) for x in edges]
 proj={"canonicalizer_version":CANONICALIZER_IDENTITY,"bootstrap_bundle_id":b.bundle_id,"authority_meta_contract":normative_projection_v3(a,"authority",b),"authoritative_manifest":normative_projection_v3(m,"manifest",b),"documents":[normative_projection_v3(d,"document",b) for d,_ in sorted(docs,key=lambda x:x[0]["id"])]}
 return {"state":"VALID_CANDIDATE","canonicalizer_identity":CANONICALIZER_IDENTITY,"bootstrap_bundle_id":b.bundle_id,"policy_corpus_hash":_digest(proj)}
