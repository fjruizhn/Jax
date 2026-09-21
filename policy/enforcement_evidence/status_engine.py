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
  db={x.subject.identity for x in observations if x.control_id==definition.control_id and x.implementation_identity_hash==identity.implementation_identity_hash and x.scope.database_scope_id==scope.database_scope_id and x.subject.subject_type is EvidenceSubjectType.DATABASE_SCHEMA and x.outcome is ObservationOutcome.SATISFIED and x.occurred_at_utc >= as_of_utc-timedelta(hours=1)}
  if scope.database_scope_id not in db: return AssertionVerdict.NOT_OBSERVED
 return AssertionVerdict.SUPPORTED

class EnforcementStatusService:
 """Fixed application composition; requests cannot choose evidence subsets."""
 def __init__(self, store, implementation_identity, *, repository_root):
  self._store=store; self._identity=implementation_identity; self._repository_root=repository_root; self._token=store._fixed_lifecycle_token()
 def _written(self):
  if not is_trusted_implementation_identity(self._identity) or self._identity.source_state is not SourceState.CLEAN: return False
  try:
   from .implementation_identity import verify_build_manifest
   manifest=verify_build_manifest(self._store,self._identity,repository_root=self._repository_root)
   return {"policy/enforcement_evidence/controls/v1.json","policy/enforcement_evidence/migrations/001_enforcement_evidence.sql"}.issubset(manifest["files"])
  except Exception: return False
 def _tested(self, definition, as_of):
  manifests=self._store._test_manifests_for(self._identity.implementation_identity_hash); required=(definition.control_id,definition.control_version)
  for m in manifests:
   try:
    from datetime import datetime
    if datetime.fromisoformat(m["completed_at_utc"].replace("Z","+00:00")) < as_of-timedelta(days=30): continue
    tests=[t for t in m["tests"] if any((b["control_id"],b["control_version"])==required for b in t["bindings"])]
    if tests and all(t["result"]=="PASSED" for t in tests): return True
   except Exception: pass
  return False
 def evaluate_control_status(self, *, control_id, control_version, claim_level, scope, subjects, as_of_utc):
  definition=load_control_definition(control_id,control_version); observations=self._store.observations()
  written=self._written(); tested=self._tested(definition,as_of_utc) if written else False
  verdict=derive_assertion(definition,self._identity,observations,claim_level=claim_level,scope=scope,subjects=tuple(subjects),as_of_utc=as_of_utc,written=written,tested=tested)
  start=as_of_utc-(timedelta(hours=24) if claim_level is ClaimLevel.ENFORCED else timedelta(days=30))
  artifacts=tuple(sorted({h for o in observations for h in o.evidence_artifact_hashes}))
  assertion=EnforcementAssertion(definition.control_id,definition.control_version,definition.control_definition_hash,claim_level,verdict,self._identity.implementation_identity_hash,scope,tuple(subjects),artifacts,tuple(o.observation_id for o in observations),as_of_utc,start,as_of_utc)
  return self._store._record_assertion(assertion,_token=self._token)

def evaluate_control_status(store, definition, identity, *, claim_level, scope, subjects, as_of_utc):
 """Compatibility pure entrypoint; it cannot self-certify prerequisites."""
 return derive_assertion(definition,identity,store.observations(),claim_level=claim_level,scope=scope,subjects=subjects,as_of_utc=as_of_utc,written=False,tested=False)
