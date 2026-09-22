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

class EnforcementStatusService:
 """Fixed application composition; requests cannot choose evidence subsets."""
 def __init__(self, lifecycle):
  # Lifecycle/identity/root are fixed at composition; requests only name a
  # claim.  In particular no request can redefine the source bytes that a
  # build manifest is compared against.
  self._lifecycle=lifecycle; self._store=lifecycle._store; self._identity=lifecycle._identity; self._identity_provider=None
 @classmethod
 def for_readonly_query(cls, store, identity_provider):
  """Build a status reader from fixed composition without lifecycle writes.

  The identity must already be an authoritative stored identity.  This is the
  CLI/query seam; unlike ``EvidenceLifecycleService`` it never records it.
  """
  from .implementation_identity import TrustedImplementationIdentityProvider, _ControlledTestIdentityProvider
  if not isinstance(identity_provider, (TrustedImplementationIdentityProvider, _ControlledTestIdentityProvider)):
   raise AttributeError("identity provider must be fixed composition")
  candidate=identity_provider.load()
  value=store.load_implementation_identity(candidate.implementation_identity_hash)
  self=cls.__new__(cls); self._lifecycle=None; self._store=store; self._identity=value; self._identity_provider=identity_provider
  return self
 def _written(self):
  if not is_trusted_implementation_identity(self._identity) or self._identity.source_state is not SourceState.CLEAN: return False
  try:
   # Provider verification includes the complete frozen B5/B6/B7 source and
   # migration set.  A subset claim is never enough for WRITTEN.
   if self._identity_provider is not None: self._identity_provider.verify_loaded_identity(self._identity)
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
  if hasattr(self._store,"readonly_status_snapshot"):
   observations, manifests, trust_domains=self._store.readonly_status_snapshot(self._identity.implementation_identity_hash)
  else:
   observations, manifests, trust_domains=self._store.observations(), None, None
  subjects=tuple(subjects); written=self._written(); tested=self._tested(definition,as_of_utc,manifests) if written else False
  verdict=derive_assertion(definition,self._identity,observations,claim_level=claim_level,scope=scope,subjects=subjects,as_of_utc=as_of_utc,written=written,tested=tested)
  start=as_of_utc-(timedelta(hours=24) if claim_level is ClaimLevel.ENFORCED else timedelta(days=30))
  artifacts=tuple(sorted({h for o in observations for h in o.evidence_artifact_hashes}))
  if trust_domains is None:
   trust_domains=tuple(sorted({a.trust_domain for o in observations for h in o.evidence_artifact_hashes for a in (self._store.load_evidence_artifact(h),)},key=lambda x:x.value))
  return definition, observations, subjects, verdict, start, artifacts, trust_domains
 def query_control_status(self, *, control_id, control_version, claim_level, scope, subjects, as_of_utc):
  """Authoritative read-only status derivation; never persists an assertion."""
  definition, observations, subjects, verdict, start, artifacts, trust_domains=self._derive(control_id=control_id,control_version=control_version,claim_level=claim_level,scope=scope,subjects=subjects,as_of_utc=as_of_utc)
  return ControlStatusView(definition.control_id,definition.control_version,claim_level,verdict,self._identity.implementation_identity_hash,scope,subjects,as_of_utc,start,as_of_utc,tuple(sorted({o.reason_code for o in observations})),trust_domains,artifacts,tuple(o.observation_id for o in observations),as_of_utc)
 def evaluate_control_status(self, *, control_id, control_version, claim_level, scope, subjects, as_of_utc):
  definition, observations, subjects, verdict, start, artifacts, _=self._derive(control_id=control_id,control_version=control_version,claim_level=claim_level,scope=scope,subjects=subjects,as_of_utc=as_of_utc)
  assertion=EnforcementAssertion(definition.control_id,definition.control_version,definition.control_definition_hash,claim_level,verdict,self._identity.implementation_identity_hash,scope,subjects,artifacts,tuple(o.observation_id for o in observations),as_of_utc,start,as_of_utc)
  return self._lifecycle._EvidenceLifecycleService__persist_assertion(assertion)

def evaluate_control_status(store, definition, identity, *, claim_level, scope, subjects, as_of_utc):
 """Compatibility pure entrypoint; it cannot self-certify prerequisites."""
 return derive_assertion(definition,identity,store.observations(),claim_level=claim_level,scope=scope,subjects=subjects,as_of_utc=as_of_utc,written=False,tested=False)
