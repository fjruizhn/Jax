"""Closed validation for C14N/3 candidate artifacts."""
from __future__ import annotations
import re
import unicodedata
from typing import Any
from .bootstrap_v3 import CANONICALIZER_IDENTITY, VerifiedBootstrapV3
from .errors import BootstrapIntegrityError, SchemaValidationError

ID_RE=re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
SEL_RE=re.compile(r"[A-Z][A-Z0-9_]*\Z")
CLASSES=("CONSTITUTIONAL_CORE","PRODUCT_POLICY","SUBORDINATE_POLICY")

ARRAYS={"authority.scope.governs":"SET_SCALAR","authority.scope.excludes":"SET_SCALAR","authority.human_authority.constitutional_ratifier.powers":"SET_SCALAR","authority.candidate_lifecycle.activation_requires":"SET_SCALAR","authority.precedence.ordered_document_layers":"ORDERED","authority.normative_sources.permitted_document_classes":"SET_SCALAR","authority.protected_metanorms":"SET_KEYED_ID","authority.meta_contract_amendment.requires":"SET_SCALAR","manifest.normative_documents":"SET_KEYED_ID","manifest.reference_documents":"DISPLAY","document.scope.subjects":"SET_SCALAR","document.scope.actions":"SET_SCALAR","document.scope.conditions_all":"SET_SCALAR","document.relationships.supersedes":"SET_SCALAR","document.relationships.superseded_by":"SET_SCALAR","document.notes":"DISPLAY","document.history":"DISPLAY","bootstrap.digest_resources":"ORDERED","bootstrap.allowed_document_classes":"SET_SCALAR"}

def fail(p,m): raise SchemaValidationError(f"{p}: {m}")
def obj(v,keys,p):
 if not isinstance(v,dict) or set(v)!=set(keys): fail(p,"campos cerrados inválidos")
 return v
def string(v,p,min=1,max=16384,pat=None,enum=None,const=None):
 if not isinstance(v,str) or not min<=len(v)<=max or unicodedata.normalize("NFC",v)!=v: fail(p,"string inválido")
 if pat and not re.fullmatch(pat,v): fail(p,"patrón inválido")
 if enum and v not in enum: fail(p,"enum inválido")
 if const is not None and v!=const: fail(p,"const inválida")
 return v
def boolean(v,p,const=None):
 if type(v) is not bool or const is not None and v is not const: fail(p,"booleano inválido")
def array(v,p,min=0,max=64):
 if not isinstance(v,list) or not min<=len(v)<=max: fail(p,"array inválido")
 return v
def scalar_set(v,p,min=0,pat=None,enum=None):
 values=[string(x,f"{p}[{i}]",pat=pat,enum=enum) for i,x in enumerate(array(v,p,min))]
 if len(set(values))!=len(values): fail(p,"duplicado semántico")
def exact(v,expected,p):
 obj(v,expected,p)
 for k,x in expected.items():
  if isinstance(x,bool): boolean(v[k],f"{p}.{k}",x)
  elif v[k]!=x: fail(f"{p}.{k}","valor inválido")
def _schema_leaf_patterns(schema, prefix=""):
 """Derive legal leaf patterns from the pinned JSON Schema shape."""
 if not isinstance(schema, dict):
  raise BootstrapIntegrityError("schema v3 inválido")
 typ=schema.get("type")
 if typ == "object":
  if schema.get("additionalProperties") is not False or not isinstance(schema.get("properties"), dict):
   raise BootstrapIntegrityError("schema object no cerrado")
  leaves=set()
  for name, child in schema["properties"].items():
   leaves |= _schema_leaf_patterns(child, f"{prefix}.{name}" if prefix else name)
  return leaves
 if typ == "array":
  if "items" not in schema:
   raise BootstrapIntegrityError("schema array sin items")
  return _schema_leaf_patterns(schema["items"], prefix + "[*]")
 # Scalars may express their type through const/enum; either is a legal leaf.
 if typ in {"string", "boolean", "number", "integer"} or "const" in schema or "enum" in schema:
  return {prefix}
 raise BootstrapIntegrityError("schema leaf inválido")

def verify_registry(b):
 fields=b.field_classes.get("fields"); arrays=b.field_classes.get("array_semantics")
 schema_sources={
  "authority": b.schemas["authority-meta-contract.schema.json"],
  "manifest": b.schemas["authoritative-policy-manifest.schema.json"],
  "document": b.schemas["normative-policy-document.schema.json"],
 }
 expected={f"{artifact}.{leaf}" for artifact,schema in schema_sources.items()
           for leaf in _schema_leaf_patterns(schema)}
 if not isinstance(fields,dict) or set(fields)!=expected or any(x not in {"NORMATIVE","PROVENANCE_REQUIRED","DISPLAY","FACTUAL_FORBIDDEN"} for x in fields.values()): raise BootstrapIntegrityError("schema/field registry no es biyectivo")
 if arrays!=ARRAYS: raise BootstrapIntegrityError("array registry v3 inválida")

def validate_authority(a:Any,b:VerifiedBootstrapV3):
 verify_registry(b); keys="schema_version kind canonicalizer_version bootstrap_bundle_id id contract_version root_pair scope human_authority candidate_lifecycle precedence normative_sources protected_metanorms meta_contract_amendment exceptions delegations suspensions interpretations amendment_and_repeal external_constraints fail_closed".split(); obj(a,keys,"authority")
 for k,c in {"schema_version":"1.0","kind":"JAX_AUTHORITY_META_CONTRACT","canonicalizer_version":CANONICALIZER_IDENTITY,"bootstrap_bundle_id":b.bundle_id,"id":"jax-authority-meta-contract","contract_version":"1.0"}.items(): string(a[k],f"authority.{k}",const=c)
 exact(a["root_pair"],{"authoritative_manifest_id":"jax-authoritative-policy-manifest","authoritative_manifest_kind":"JAX_AUTHORITATIVE_POLICY_MANIFEST"},"authority.root_pair")
 obj(a["scope"],["jurisdiction","governs","excludes"],"authority.scope"); string(a["scope"]["jurisdiction"],"authority.scope.jurisdiction",const="JAX"); scalar_set(a["scope"]["governs"],"authority.scope.governs"); scalar_set(a["scope"]["excludes"],"authority.scope.excludes")
 h=a["human_authority"]; obj(h,["constitutional_ratifier","mesa","agents"],"authority.human_authority"); r=h["constitutional_ratifier"]; obj(r,["actor_id","actor_kind","powers"],"authority.ratifier"); string(r["actor_id"],"authority.ratifier.actor_id",const="human:fernando"); string(r["actor_kind"],"authority.ratifier.actor_kind",const="HUMAN_LOGICAL"); scalar_set(r["powers"],"authority.ratifier.powers",1)
 exact(h["mesa"],{"role":"ADVISORY_REVIEW_AUDIT","normative_vote":False,"may_ratify":False},"authority.mesa"); exact(h["agents"],{"normative_vote":False,"may_ratify":False,"may_expand_authority":False},"authority.agents")
 l=a["candidate_lifecycle"]; obj(l,["activation_mode","activation_requires","git_commit_is_ratification","git_push_is_ratification"],"authority.lifecycle"); string(l["activation_mode"],"authority.lifecycle.activation_mode",const="CANDIDATE_ONLY"); scalar_set(l["activation_requires"],"authority.lifecycle.activation_requires",1); boolean(l["git_commit_is_ratification"],"authority.lifecycle.commit",False); boolean(l["git_push_is_ratification"],"authority.lifecycle.push",False)
 p=a["precedence"]; obj(p,["authority_meta_contract","ordered_document_layers","overlay_position","external_constraints","equal_rank_conflict","unresolved_conflict"],"authority.precedence"); string(p["authority_meta_contract"],"authority.precedence.authority_meta_contract",const="ROOT_META_LEVEL"); vals=array(p["ordered_document_layers"],"authority.precedence.layers",4,4); [string(x,"authority.precedence.layers[]",enum=("PROTECTED_METANORM",*CLASSES)) for x in vals]; exact({k:p[k] for k in p if k not in {"authority_meta_contract","ordered_document_layers"}},{"overlay_position":"POST_CORPUS_WITHIN_VALID_DELEGATED_SCOPE","external_constraints":"EXTERNAL_CEILING","equal_rank_conflict":"REJECT_UNTIL_RATIFIED_RESOLUTION","unresolved_conflict":"REJECT"},"authority.precedence")
 n=a["normative_sources"]; obj(n,["permitted_document_classes","manifest_classification_required","document_self_classification_authoritative","legacy_status_is_normative_force","unlisted_documents_have_normative_force"],"authority.sources"); scalar_set(n["permitted_document_classes"],"authority.sources.classes",3,enum=CLASSES); [boolean(n[k],f"authority.sources.{k}",x) for k,x in {"manifest_classification_required":True,"document_self_classification_authoritative":False,"legacy_status_is_normative_force":False,"unlisted_documents_have_normative_force":False}.items()]
 metas=array(a["protected_metanorms"],"authority.metas",5,5); ids=[]
 for i,x in enumerate(metas): obj(x,["id","statement","non_waivable"],f"authority.metas[{i}]"); ids.append(string(x["id"],f"authority.metas[{i}].id",pat=r"[A-Z][A-Z0-9_]*")); string(x["statement"],f"authority.metas[{i}].statement"); boolean(x["non_waivable"],f"authority.metas[{i}].non_waivable",True)
 if len(ids)!=len(set(ids)): fail("authority.metas","duplicado")
 ma=a["meta_contract_amendment"]; obj(ma,["requires","implicit_amendment","subordinate_document_may_amend","protected_metanorm_document_may_amend"],"authority.amendment"); scalar_set(ma["requires"],"authority.amendment.requires",1); string(ma["implicit_amendment"],"authority.amendment.implicit",const="forbidden"); boolean(ma["subordinate_document_may_amend"],"authority.amendment.subordinate",False); boolean(ma["protected_metanorm_document_may_amend"],"authority.amendment.protected",False)
 exact(a["exceptions"],{"storage":"SEPARATE_AUTHORITY_LEDGER","modifies_policy_corpus":False,"creates_precedent":False,"requires_explicit_scope":True,"requires_explicit_expiry_or_bound":True,"may_waive_non_waivable":False,"absent_or_invalid":"REJECT"},"authority.exceptions"); exact(a["delegations"],{"storage":"SEPARATE_AUTHORITY_LEDGER","default":"DENY","may_self_expand":False,"must_be_scope_bounded":True,"must_be_time_or_event_bounded":True,"may_override_protected_metanorm":False},"authority.delegations"); exact(a["suspensions"],{"storage":"SEPARATE_AUTHORITY_LEDGER","modifies_policy_corpus":False,"requires_explicit_scope":True,"requires_explicit_expiry_or_bound":True,"may_suspend_non_waivable":False,"absent_or_invalid":"REJECT"},"authority.suspensions")
 obj(a["interpretations"],["GENERAL_NORMATIVE","BINDING_PARTICULAR","EXPLANATORY"],"authority.interpretations"); exact(a["interpretations"]["GENERAL_NORMATIVE"],{"changes_policy_corpus":True,"requires_human_ratification":True},"authority.general"); exact(a["interpretations"]["BINDING_PARTICULAR"],{"storage":"SEPARATE_AUTHORITY_LEDGER","changes_policy_corpus":False,"changes_effective_authority_context":"WHEN_MATERIAL"},"authority.particular"); exact(a["interpretations"]["EXPLANATORY"],{"binding_force":"NONE","changes_policy_corpus":False,"changes_effective_authority_context":False},"authority.explanatory")
 exact(a["amendment_and_repeal"],{"changes_policy_corpus":True,"requires_human_ratification":True,"requires_new_policy_corpus_hash":True,"implicit_repeal":"forbidden"},"authority.repeal"); exact(a["external_constraints"],{"jax_normative":False,"effect":"CEILING_ONLY","may_grant_authority":False,"provenance_required_at_evaluation":True},"authority.external"); exact(a["fail_closed"],{"default":"REJECT","missing_or_invalid_root_pair":"REJECT","invalid_membership":"REJECT","unknown_document_class":"REJECT","use_of_non_active_candidate":"REJECT","invalid_or_expired_overlay":"REJECT","unresolved_precedence":"REJECT"},"authority.failclosed")

def validate_manifest(m:Any,b:VerifiedBootstrapV3):
 verify_registry(b); keys="schema_version kind canonicalizer_version bootstrap_bundle_id id manifest_version governed_by candidate_lifecycle corpus membership normative_documents reference_documents".split(); obj(m,keys,"manifest")
 for k,c in {"schema_version":"1.0","kind":"JAX_AUTHORITATIVE_POLICY_MANIFEST","canonicalizer_version":CANONICALIZER_IDENTITY,"bootstrap_bundle_id":b.bundle_id,"id":"jax-authoritative-policy-manifest","manifest_version":"1.0"}.items(): string(m[k],f"manifest.{k}",const=c)
 exact(m["governed_by"],{"authority_meta_contract_id":"jax-authority-meta-contract","authority_meta_contract_kind":"JAX_AUTHORITY_META_CONTRACT"},"manifest.governed_by"); exact(m["candidate_lifecycle"],{"activation_mode":"CANDIDATE_ONLY","authorizes_production":False},"manifest.lifecycle"); exact(m["corpus"],{"jurisdiction":"JAX","legacy_identity_mode":"COMPARE_ONLY"},"manifest.corpus"); exact(m["membership"],{"paths_are_explicit":True,"symlinks_allowed":False,"unlisted_documents_have_normative_force":False,"duplicate_ids":"forbidden","duplicate_paths":"forbidden","document_self_classification_authoritative":False},"manifest.membership")
 ids=[]
 for i,x in enumerate(array(m["normative_documents"],"manifest.members")):
  obj(x,["id","path","document_class","normative_layer","normative_effect"],f"manifest.member[{i}]"); ids.append(string(x["id"],f"manifest.member[{i}].id",pat=ID_RE.pattern)); string(x["path"],f"manifest.member[{i}].path"); string(x["document_class"],f"manifest.member[{i}].class",enum=CLASSES); string(x["normative_layer"],f"manifest.member[{i}].layer",enum=CLASSES); string(x["normative_effect"],f"manifest.member[{i}].effect",const="ACTIVE_WHEN_CORPUS_ACTIVE")
 if len(ids)!=len(set(ids)): fail("manifest.members","id duplicado")
 refs=[]
 for i,x in enumerate(array(m["reference_documents"],"manifest.references")):
  obj(x,["reference_id","source_locator","source_type","legacy_status","display_name"],f"manifest.reference[{i}]"); refs.append(string(x["reference_id"],f"manifest.reference[{i}].id",pat=ID_RE.pattern)); string(x["source_locator"],f"manifest.reference[{i}].locator"); string(x["source_type"],f"manifest.reference[{i}].source",enum=("LEGACY_CORPUS",)); string(x["legacy_status"],f"manifest.reference[{i}].status"); string(x["display_name"],f"manifest.reference[{i}].name",max=256)
 if len(refs)!=len(set(refs)): fail("manifest.references","id duplicado")

def validate_document(d:Any,b:VerifiedBootstrapV3|None=None):
 if b: verify_registry(b)
 keys="schema_version kind id title statement scope document_class normative_layer lifecycle_status blocking expected_enforcement effective_semantics relationships origin notes history".split(); obj(d,keys,"document")
 string(d["schema_version"],"document.schema_version",const="1.0"); string(d["kind"],"document.kind",const="JAX_NORMATIVE_POLICY_DOCUMENT"); string(d["id"],"document.id",pat=ID_RE.pattern); string(d["title"],"document.title",max=256); string(d["statement"],"document.statement",max=16384)
 s=d["scope"]; obj(s,["jurisdiction","subjects","actions","conditions_all"],"document.scope"); string(s["jurisdiction"],"document.scope.jurisdiction",const="JAX"); scalar_set(s["subjects"],"document.scope.subjects",1,SEL_RE.pattern); scalar_set(s["actions"],"document.scope.actions",1,SEL_RE.pattern); scalar_set(s["conditions_all"],"document.scope.conditions",0,SEL_RE.pattern)
 string(d["document_class"],"document.class",enum=CLASSES); string(d["normative_layer"],"document.layer",enum=CLASSES)
 if d["document_class"]!=d["normative_layer"]: fail("document.layer","class incompatible")
 string(d["lifecycle_status"],"document.lifecycle",const="CANDIDATE"); obj(d["blocking"],["mode"],"document.blocking"); string(d["blocking"]["mode"],"document.blocking.mode",enum=("HARD_BLOCK","SOFT_BLOCK")); obj(d["expected_enforcement"],["mode"],"document.enforcement"); string(d["expected_enforcement"]["mode"],"document.enforcement.mode",enum=("REQUIRED","ADVISORY")); exact(d["effective_semantics"],{"activation":"WHEN_CORPUS_ACTIVE","termination":"SUPERSEDED_OR_REPEALED_BY_RATIFIED_CORPUS"},"document.effective")
 rel=d["relationships"]; obj(rel,["supersedes","superseded_by"],"document.relationships"); scalar_set(rel["supersedes"],"document.supersedes",0,ID_RE.pattern); scalar_set(rel["superseded_by"],"document.superseded_by",0,ID_RE.pattern); origin=d["origin"]; obj(origin,["source_type","source_ref"],"document.origin"); string(origin["source_type"],"document.origin.type",enum=("HUMAN_NOMINATED",)); string(origin["source_ref"],"document.origin.ref",max=256)
 for n in ("notes","history"):
  for i,x in enumerate(array(d[n],f"document.{n}")): string(x,f"document.{n}[{i}]")
