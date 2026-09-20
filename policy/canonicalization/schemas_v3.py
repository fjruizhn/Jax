"""Closed structural validation for candidate-only C14N/3 artifacts."""
from __future__ import annotations
import re
from typing import Any
from .errors import SchemaValidationError, BootstrapIntegrityError
from .bootstrap_v3 import CANONICALIZER_IDENTITY, VerifiedBootstrapV3
ID = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
CLASSES={"CONSTITUTIONAL_CORE","PRODUCT_POLICY","SUBORDINATE_POLICY"}
def _leaves(value: Any, prefix: str):
 if isinstance(value,dict):
  for k,v in value.items(): yield from _leaves(v,f"{prefix}.{k}")
 elif isinstance(value,list):
  for v in value: yield from _leaves(v,f"{prefix}[*]")
 else: yield prefix
def _registry_closed(value: Any, artifact: str, b: VerifiedBootstrapV3):
 expected={k for k in b.field_classes["fields"] if k.startswith(artifact+".")}
 actual=set(_leaves(value,artifact))
 missing = {x for x in expected - actual if "[*]" not in x}
 unknown = actual - expected
 if missing or unknown:
  raise BootstrapIntegrityError(f"schema/field registry mismatch for {artifact}: {sorted(missing | unknown)}")
def _obj(v:Any, keys:set[str], path:str):
 if not isinstance(v,dict) or set(v)!=keys: raise SchemaValidationError(f"{path}: campos cerrados inválidos")
def _str(v:Any,path:str):
 if not isinstance(v,str) or not v: raise SchemaValidationError(f"{path}: string requerido")
def _arr(v:Any,path:str):
 if not isinstance(v,list): raise SchemaValidationError(f"{path}: array requerido")
def _set(v,path,pattern=None, nonempty=False):
 _arr(v,path)
 if nonempty and not v: raise SchemaValidationError(f"{path}: no puede estar vacío")
 if any(not isinstance(x,str) or (pattern and not re.fullmatch(pattern,x)) for x in v) or len(set(v))!=len(v): raise SchemaValidationError(f"{path}: set inválido")
def validate_authority(a:Any,b:VerifiedBootstrapV3):
 required={"schema_version","kind","canonicalizer_version","bootstrap_bundle_id","id","contract_version","root_pair","scope","human_authority","candidate_lifecycle","precedence","normative_sources","protected_metanorms","meta_contract_amendment","exceptions","delegations","suspensions","interpretations","amendment_and_repeal","external_constraints","fail_closed"}; _obj(a,required,"authority")
 if a["schema_version"]!="1.0" or a["kind"]!="JAX_AUTHORITY_META_CONTRACT" or a["canonicalizer_version"]!=CANONICALIZER_IDENTITY or a["bootstrap_bundle_id"]!=b.bundle_id or a["id"]!="jax-authority-meta-contract": raise SchemaValidationError("authority identidad inválida")
 if a["candidate_lifecycle"]!={"activation_mode":"CANDIDATE_ONLY","activation_requires":["FUTURE_HUMAN_RATIFICATION_RECORD","EXACT_POLICY_CORPUS_HASH_BINDING"],"git_commit_is_ratification":False,"git_push_is_ratification":False}: raise SchemaValidationError("authority lifecycle inválido")
 if a["external_constraints"]!={"jax_normative":False,"effect":"CEILING_ONLY","may_grant_authority":False,"provenance_required_at_evaluation":True}: raise SchemaValidationError("external constraints inválidas")
 if a["normative_sources"]["permitted_document_classes"]!=["CONSTITUTIONAL_CORE","PRODUCT_POLICY","SUBORDINATE_POLICY"]: raise SchemaValidationError("clases inválidas")
 _set(a["scope"]["governs"],"scope.governs"); _set(a["scope"]["excludes"],"scope.excludes")
 if a["human_authority"]["constitutional_ratifier"]["actor_id"]!="human:fernando" or a["human_authority"]["constitutional_ratifier"]["actor_kind"]!="HUMAN_LOGICAL": raise SchemaValidationError("actor inválido")
 ids=[]
 for x in a["protected_metanorms"]:
  _obj(x,{"id","statement","non_waivable"},"protected_metanorm"); ids.append(x["id"])
  if not ID.fullmatch(x["id"].lower().replace("_","-")) or not isinstance(x["statement"],str) or x["non_waivable"] is not True: raise SchemaValidationError("metanorma inválida")
 if len(ids)!=len(set(ids)): raise SchemaValidationError("metanorma duplicada")
 _registry_closed(a,"authority",b)
def validate_manifest(m:Any,b:VerifiedBootstrapV3):
 keys={"schema_version","kind","canonicalizer_version","bootstrap_bundle_id","id","manifest_version","governed_by","candidate_lifecycle","corpus","membership","normative_documents","reference_documents"}; _obj(m,keys,"manifest")
 if m["schema_version"]!="1.0" or m["kind"]!="JAX_AUTHORITATIVE_POLICY_MANIFEST" or m["canonicalizer_version"]!=CANONICALIZER_IDENTITY or m["bootstrap_bundle_id"]!=b.bundle_id or m["id"]!="jax-authoritative-policy-manifest": raise SchemaValidationError("manifest identidad inválida")
 if m["candidate_lifecycle"]!={"activation_mode":"CANDIDATE_ONLY","authorizes_production":False}: raise SchemaValidationError("manifest lifecycle inválido")
 _arr(m["normative_documents"],"normative_documents"); _arr(m["reference_documents"],"reference_documents")
 for x in m["normative_documents"]: _obj(x,{"id","path","document_class","normative_layer","normative_effect"},"member")
 for x in m["reference_documents"]: _obj(x,{"reference_id","source_locator","source_type","legacy_status","display_name"},"reference")
 _registry_closed(m,"manifest",b)
def validate_document(d:Any, b: VerifiedBootstrapV3 | None = None):
 keys={"schema_version","kind","id","title","statement","scope","document_class","normative_layer","lifecycle_status","blocking","expected_enforcement","effective_semantics","relationships","origin","notes","history"}; _obj(d,keys,"document")
 if d["schema_version"]!="1.0" or d["kind"]!="JAX_NORMATIVE_POLICY_DOCUMENT" or not isinstance(d["id"],str) or not ID.fullmatch(d["id"]): raise SchemaValidationError("document identidad inválida")
 if d["document_class"] not in CLASSES or d["normative_layer"]!=d["document_class"] or d["lifecycle_status"]!="CANDIDATE": raise SchemaValidationError("document class/layer inválido")
 _obj(d["scope"],{"jurisdiction","subjects","actions","conditions_all"},"scope")
 if d["scope"]["jurisdiction"]!="JAX": raise SchemaValidationError("jurisdiction inválida")
 _set(d["scope"]["subjects"],"subjects",r"[A-Z][A-Z0-9_]*",True); _set(d["scope"]["actions"],"actions",r"[A-Z][A-Z0-9_]*",True); _set(d["scope"]["conditions_all"],"conditions",r"[A-Z][A-Z0-9_]*")
 if d["blocking"] not in ({"mode":"HARD_BLOCK"},{"mode":"SOFT_BLOCK"}) or d["expected_enforcement"] not in ({"mode":"REQUIRED"},{"mode":"ADVISORY"}) or d["effective_semantics"]!={"activation":"WHEN_CORPUS_ACTIVE","termination":"SUPERSEDED_OR_REPEALED_BY_RATIFIED_CORPUS"}: raise SchemaValidationError("semántica efectiva inválida")
 _obj(d["relationships"],{"supersedes","superseded_by"},"relationships"); _set(d["relationships"]["supersedes"],"supersedes",ID.pattern); _set(d["relationships"]["superseded_by"],"superseded_by",ID.pattern)
 _obj(d["origin"],{"source_type","source_ref"},"origin"); _arr(d["notes"],"notes"); _arr(d["history"],"history")
 if b is not None: _registry_closed(d,"document",b)
