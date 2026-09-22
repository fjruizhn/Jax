"""Deterministic, composition-owned claim derivation."""
from __future__ import annotations
from datetime import timedelta
from .models import *
from .control_registry import load_control_definition, require_trusted_definition
from .evidence_store import is_trusted_implementation_identity
from .errors import UnsupportedClaimLevelError, InvalidClaimScopeError

def _matching(definition, identity, observations, scope, subjects):
 return tuple(x for x in observations if x.control_id==definition.control_id and x.control_version==definition.control_version and x.control_definition_hash==definition.control_definition_hash and x.implementation_identity_hash==identity.implementation_identity_hash and x.scope==scope and x.subject in subjects)

def derive_assertion(definition, identity, observations, *, claim_level, scope, subjects, as_of_utc, written=True, tested=None):
 """Pure verdict logic. The composition service alone determines prerequisites."""
 require_trusted_definition(definition)
 if claim_level not in definition.supported_claim_levels: raise UnsupportedClaimLevelError(str(claim_level))
 if claim_level is ClaimLevel.ENFORCED and not subjects: raise InvalidClaimScopeError("ENFORCED requiere subjects")
 if not is_trusted_implementation_identity(identity): return AssertionVerdict.UNVERIFIABLE
 if identity.source_state is not SourceState.CLEAN: return AssertionVerdict.STALE
 matched=_matching(definition,identity,observations,scope,subjects)
 # Evidence dated after the claim cannot be used to prove the past.  This is
 # especially important for the one-hour installed-schema profile.
 if any(x.occurred_at_utc > as_of_utc for x in matched): return AssertionVerdict.STALE
 if any(x.outcome is ObservationOutcome.FAILED for x in matched): return AssertionVerdict.FAILED
 if any(x.outcome is ObservationOutcome.ERROR for x in matched): return AssertionVerdict.UNVERIFIABLE
 if claim_level is ClaimLevel.WRITTEN: return AssertionVerdict.SUPPORTED if written else AssertionVerdict.INSUFFICIENT_EVIDENCE
 if not written: return AssertionVerdict.INSUFFICIENT_EVIDENCE
 if claim_level is ClaimLevel.TESTED: return AssertionVerdict.SUPPORTED if tested is True else AssertionVerdict.INSUFFICIENT_EVIDENCE
 if tested is False: return AssertionVerdict.INSUFFICIENT_EVIDENCE
 if not matched: return AssertionVerdict.NOT_OBSERVED
 if any(x.occurred_at_utc < as_of_utc-timedelta(hours=24) for x in matched): return AssertionVerdict.STALE
 if tested is not True: return AssertionVerdict.INSUFFICIENT_EVIDENCE
 if {x.subject for x in matched if x.outcome is ObservationOutcome.SATISFIED} != set(subjects): return AssertionVerdict.NOT_OBSERVED
 if definition.control_id=="CTL.B6.ONE_DECISION_ONE_EXECUTION":
  db={x.subject.identity for x in observations if x.control_id==definition.control_id and x.implementation_identity_hash==identity.implementation_identity_hash and x.scope.database_scope_id==scope.database_scope_id and x.subject.subject_type is EvidenceSubjectType.DATABASE_SCHEMA and x.outcome is ObservationOutcome.SATISFIED and as_of_utc-timedelta(hours=1) <= x.occurred_at_utc <= as_of_utc}
  if scope.database_scope_id not in db: return AssertionVerdict.NOT_OBSERVED
  # Installed schema alone cannot certify the control.  The same derived DB
  # scope must also contain an actual successful create and a duplicate denial.
  created=any(x.outcome is ObservationOutcome.SATISFIED and x.subject.subject_type is EvidenceSubjectType.EXECUTION for x in matched)
  duplicate=any(x.control_id==definition.control_id and x.implementation_identity_hash==identity.implementation_identity_hash and x.scope==scope and x.outcome is ObservationOutcome.DENIED and as_of_utc-timedelta(hours=24) <= x.occurred_at_utc <= as_of_utc for x in observations)
  if not created or not duplicate: return AssertionVerdict.NOT_OBSERVED
 return AssertionVerdict.SUPPORTED

class _StatusDerivationService:
 """Shared deterministic derivation mechanics; not a query capability."""
 _readonly=False
 def __init__(self, lifecycle):
  # Lifecycle/identity/root are fixed at composition; requests only name a
  # claim.  In particular no request can redefine the source bytes that a
  # build manifest is compared against.
  self._lifecycle=lifecycle; self._store=lifecycle._store; self._identity=lifecycle._identity; self._identity_provider=None
 def _written(self, identity=None, manifest_bytes=None):
  identity=identity or self._identity
  if not is_trusted_implementation_identity(identity) or identity.source_state is not SourceState.CLEAN: return False
  try:
   # Provider verification includes the complete frozen B5/B6/B7 source and
   # migration set.  A subset claim is never enough for WRITTEN.
   if self._identity_provider is not None:
    if manifest_bytes is None: return False
    self._identity_provider.verify_loaded_identity_bytes(identity, manifest_bytes)
   else: self._lifecycle._verify_composed_identity_manifest()
   return True
  except Exception: return False  # fail-soft: unverifiable manifest is insufficient evidence.
 def _tested(self, definition, as_of, manifests=None):
  manifests=manifests if manifests is not None else self._store._test_manifests_for(self._identity.implementation_identity_hash); required=(definition.control_id,definition.control_version)
  for m in manifests:
   try:
    from datetime import datetime
    completed=datetime.fromisoformat(m["completed_at_utc"].replace("Z","+00:00"))
    started=datetime.fromisoformat(m["started_at_utc"].replace("Z","+00:00"))
    # Future-dated or internally inconsistent CI reports are ineligible.
    if completed.tzinfo is None or started.tzinfo is None or completed < started or completed > as_of or completed < as_of-timedelta(days=30): continue
    tests=[t for t in m["tests"] if any((b["control_id"],b["control_version"])==required for b in t["bindings"])]
    from .test_evidence import _TEST_CONTROL_MAP
    required_ids={test_id for test_id, controls in _TEST_CONTROL_MAP.items() if required in {(control_id, 1) for control_id in controls}}
    reported={t["test_id"]:t["result"] for t in tests}
    # A manifest is evidence of the whole frozen control suite, never a
    # caller-selected positive subset.  Any failed/error mapped test wins.
    if required_ids and set(reported) == required_ids and all(reported[x] == "PASSED" for x in required_ids): return True
   except Exception: pass  # fail-soft: malformed manifest is ineligible, never passing evidence.
  return False
 def _derive(self, *, control_id, control_version, claim_level, scope, subjects, as_of_utc):
  """One complete, composition-owned derivation shared by both public paths."""
  definition=load_control_definition(control_id,control_version)
  identity=self._identity; manifest_bytes=None
  if self._readonly and hasattr(self._store,"readonly_status_snapshot"):
   snapshot=self._store.readonly_status_snapshot(self._identity.implementation_identity_hash, control_id, control_version)
   if hasattr(snapshot,"observations"):
    identity=snapshot.identity; manifest_bytes=snapshot.manifest_bytes
    observations, manifests, trust_domains=snapshot.observations, snapshot.manifests, snapshot.trust_domains
   else:
    observations, manifests, trust_domains=snapshot
  else:
   observations, manifests, trust_domains=self._store.observations(), None, None
  subjects=tuple(subjects); written=self._written(identity,manifest_bytes); tested=self._tested(definition,as_of_utc,manifests) if written else False
  verdict=derive_assertion(definition,identity,observations,claim_level=claim_level,scope=scope,subjects=subjects,as_of_utc=as_of_utc,written=written,tested=tested)
  start=as_of_utc-(timedelta(hours=24) if claim_level is ClaimLevel.ENFORCED else timedelta(days=30))
  artifacts=tuple(sorted({h for o in observations for h in o.evidence_artifact_hashes}))
  if trust_domains is None:
   trust_domains=tuple(sorted({a.trust_domain for o in observations for h in o.evidence_artifact_hashes for a in (self._store.load_evidence_artifact(h),)},key=lambda x:x.value))
  return definition, identity, observations, subjects, verdict, start, artifacts, trust_domains

class EnforcementStatusService(_StatusDerivationService):
 """Persisted B7 assertion lifecycle; it deliberately has no read-only query API."""
 def evaluate_control_status(self, *, control_id, control_version, claim_level, scope, subjects, as_of_utc):
  definition, identity, observations, subjects, verdict, start, artifacts, _=self._derive(control_id=control_id,control_version=control_version,claim_level=claim_level,scope=scope,subjects=subjects,as_of_utc=as_of_utc)
  assertion=EnforcementAssertion(definition.control_id,definition.control_version,definition.control_definition_hash,claim_level,verdict,identity.implementation_identity_hash,scope,subjects,artifacts,tuple(o.observation_id for o in observations),as_of_utc,start,as_of_utc)
  return self._lifecycle._EvidenceLifecycleService__persist_assertion(assertion)

class _ReadonlyStatusDerivation(_StatusDerivationService):
 """Injectable derivation mechanics, deliberately unable to emit a status view.

 This type exists so deterministic snapshot handling can be tested.  It is
 not an authoritative query capability: it exposes only the shared raw
 derivation input/output and has no method that constructs a
 ``ControlStatusView``.
 """
 _readonly=True
 def __init__(self, store, identity_provider, identity_reference_hash):
  self._lifecycle=None; self._store=store; self._identity_provider=identity_provider
  self._identity=type("IdentityReference",(),{"implementation_identity_hash":identity_reference_hash})()

def _compose_runtime_readonly_derivation():
 """Bind the only production reader to fixed deployment configuration.

 This internal composition helper takes no dependencies.  Its return value is
 only a raw derivation helper, never an authoritative status-query object.
 """
 try:
  import os
  from .mariadb_store import MariaDBEvidenceStore
  from .implementation_identity import TrustedImplementationIdentityProvider
  import pymysql
  host=os.environ["JAX_DB_HOST"]; port=int(os.environ["JAX_DB_PORT"])
 except (ImportError, KeyError, ValueError) as exc:
  raise RuntimeError("MariaDB B7 composition unavailable") from exc
 def connect():
  return pymysql.connect(host=host,port=port,user=os.environ.get("JAX_DB_USER", ""),password=os.environ.get("JAX_DB_PASSWORD", ""),database=os.environ.get("JAX_DB_NAME", "jax_memory"),charset="utf8mb4",autocommit=False,connect_timeout=5)
 store=MariaDBEvidenceStore(connect)
 provider=TrustedImplementationIdentityProvider(store)
 # Deployment configuration supplies the identity reference.  The row and
 # manifest bytes are captured and verified inside the read-only snapshot.
 return _ReadonlyStatusDerivation(store,provider,provider.identity_reference_hash())

def query_control_status(*, control_id, control_version, claim_level, scope, subjects, as_of_utc):
 """Return an ephemeral B7 view from the fixed production composition.

 The semantic claim parameters above are the complete public input surface.
 Neither a caller-created store/provider/lifecycle nor an injectable query
 object can select the authoritative universe or emit this classification.
 """
 derivation=_compose_runtime_readonly_derivation()
 definition, identity, observations, subjects, verdict, start, artifacts, trust_domains=derivation._derive(control_id=control_id,control_version=control_version,claim_level=claim_level,scope=scope,subjects=subjects,as_of_utc=as_of_utc)
 return ControlStatusView(definition.control_id,definition.control_version,claim_level,verdict,identity.implementation_identity_hash,scope,subjects,as_of_utc,start,as_of_utc,tuple(sorted({o.reason_code for o in observations})),trust_domains,artifacts,tuple(o.observation_id for o in observations),as_of_utc)

def evaluate_control_status(store, definition, identity, *, claim_level, scope, subjects, as_of_utc):
 """Compatibility pure entrypoint; it cannot self-certify prerequisites."""
 return derive_assertion(definition,identity,store.observations(),claim_level=claim_level,scope=scope,subjects=subjects,as_of_utc=as_of_utc,written=False,tested=False)
