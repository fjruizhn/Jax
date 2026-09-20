"""Read-only candidate corpus validation for C14N/3."""
from __future__ import annotations
import hashlib, os, stat, unicodedata
from pathlib import Path, PurePosixPath
from .bootstrap_v3 import verified_bootstrap_v3, CANONICALIZER_IDENTITY
from .canonical_json import canonical_json_bytes
from .errors import SchemaValidationError
from .strict_yaml import load_strict_yaml
from .schemas_v3 import validate_authority,validate_manifest,validate_document
from .projection_v3 import normative_projection_v3
def _root(p): return Path(os.path.abspath(p))
def _safe(root, raw):
 if not isinstance(raw,str) or unicodedata.normalize("NFC", raw) != raw or "\\" in raw or raw.startswith("/") or any(x in {"",".",".."} for x in raw.split("/")) or any(c in raw for c in "*?["):
  raise SchemaValidationError("locator inseguro")
 path=root/PurePosixPath(raw)
 try: mode=path.lstat().st_mode
 except FileNotFoundError as e: raise SchemaValidationError("locator ausente") from e
 if stat.S_ISLNK(mode) or not stat.S_ISREG(mode) or path.resolve().parent != (root / PurePosixPath(raw)).parent.resolve(): raise SchemaValidationError("locator no regular")
 return path
def _digest(proj): return "sha256:"+hashlib.sha256(b"JAX-POLICY-CORPUS\0"+CANONICALIZER_IDENTITY.encode("ascii")+b"\0"+canonical_json_bytes(proj)).hexdigest()
def validate_candidate_corpus(repo_root):
 root=_root(repo_root); b=verified_bootstrap_v3(); bundle=b.bundle
 a=load_strict_yaml(_safe(root,bundle["fixed_root_locators"]["authority_meta_contract"])); m=load_strict_yaml(_safe(root,bundle["fixed_root_locators"]["authoritative_manifest"]))
 validate_authority(a,b); validate_manifest(m,b)
 if a["kind"] != "JAX_AUTHORITY_META_CONTRACT" or m["kind"] != "JAX_AUTHORITATIVE_POLICY_MANIFEST": raise SchemaValidationError("root kind mismatch")
 if a["canonicalizer_version"] != CANONICALIZER_IDENTITY or m["canonicalizer_version"] != CANONICALIZER_IDENTITY: raise SchemaValidationError("canonicalizer mismatch")
 if a["bootstrap_bundle_id"] != b.bundle_id or m["bootstrap_bundle_id"] != b.bundle_id: raise SchemaValidationError("bootstrap mismatch")
 if a["root_pair"]!={"authoritative_manifest_id":m["id"],"authoritative_manifest_kind":m["kind"]} or m["governed_by"]!={"authority_meta_contract_id":a["id"],"authority_meta_contract_kind":a["kind"]}: raise SchemaValidationError("root pair mismatch")
 if a["scope"]["jurisdiction"] != "JAX" or m["corpus"]["jurisdiction"] != "JAX": raise SchemaValidationError("jurisdiction mismatch")
 if a["candidate_lifecycle"]["activation_mode"] != "CANDIDATE_ONLY" or m["candidate_lifecycle"] != {"activation_mode":"CANDIDATE_ONLY","authorizes_production":False}: raise SchemaValidationError("lifecycle mismatch")
 docs=[]; ids=set(); paths=set(); refs=set(); locs=set()
 for r in m["reference_documents"]:
  rid, locator = unicodedata.normalize("NFC",r["reference_id"]), unicodedata.normalize("NFC",r["source_locator"])
  if rid != r["reference_id"] or locator != r["source_locator"] or rid in refs or locator in locs: raise SchemaValidationError("reference duplicate or non-NFC")
  refs.add(rid); locs.add(locator); _safe(root,locator)
 for member in m["normative_documents"]:
  mid, locator=unicodedata.normalize("NFC",member["id"]), unicodedata.normalize("NFC",member["path"])
  if mid != member["id"] or locator != member["path"] or mid in ids or locator in paths: raise SchemaValidationError("member duplicate or non-NFC")
  ids.add(mid); paths.add(locator); d=load_strict_yaml(_safe(root,locator)); validate_document(d,b)
  if member["id"]!=d["id"] or member["document_class"]!=d["document_class"] or member["normative_layer"]!=d["normative_layer"] or member["normative_effect"]!="ACTIVE_WHEN_CORPUS_ACTIVE" or d["lifecycle_status"] != "CANDIDATE": raise SchemaValidationError("member mismatch")
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
