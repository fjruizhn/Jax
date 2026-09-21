"""Pure bounded claim derivation; no model/caller-provided verdict exists."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from .models import *
from .control_registry import ControlDefinition, require_trusted_definition
from .evidence_store import is_trusted_implementation_identity
from .errors import UnsupportedClaimLevelError, InvalidClaimScopeError
def derive_assertion(definition, identity, observations, *, claim_level, scope, subjects, as_of_utc):
 require_trusted_definition(definition)
 if claim_level not in definition.supported_claim_levels: raise UnsupportedClaimLevelError(str(claim_level))
 if not subjects and claim_level is ClaimLevel.ENFORCED: raise InvalidClaimScopeError("ENFORCED requiere subjects")
 if not is_trusted_implementation_identity(identity): verdict=AssertionVerdict.UNVERIFIABLE
 elif identity.source_state is not SourceState.CLEAN: verdict=AssertionVerdict.STALE
 else:
  matching=[x for x in observations if x.control_id==definition.control_id and x.control_version==definition.control_version and x.implementation_identity_hash==identity.implementation_identity_hash and x.scope==scope and x.subject in subjects]
  if any(x.outcome is ObservationOutcome.FAILED for x in matching): verdict=AssertionVerdict.FAILED
  elif any(x.outcome is ObservationOutcome.ERROR for x in matching): verdict=AssertionVerdict.UNVERIFIABLE
  # A clean registered identity alone is not WRITTEN: the identity's build
  # manifest and every control implementation reference must be verified by
  # the authoritative profile evaluator.
  elif claim_level is ClaimLevel.WRITTEN: verdict=AssertionVerdict.INSUFFICIENT_EVIDENCE
  elif claim_level is ClaimLevel.TESTED: verdict=AssertionVerdict.INSUFFICIENT_EVIDENCE
  elif not matching: verdict=AssertionVerdict.NOT_OBSERVED
  elif any(x.occurred_at_utc < as_of_utc-timedelta(hours=24) for x in matching): verdict=AssertionVerdict.STALE
  # V1 profiles are deliberately fail-closed until a complete trusted test,
  # runtime and (where required) DB evidence query is supplied.  A manually
  # constructed or single persisted observation is never ENFORCED.
  else: verdict=AssertionVerdict.INSUFFICIENT_EVIDENCE
 return verdict

def evaluate_control_status(store, definition, identity, *, claim_level, scope, subjects, as_of_utc):
 """Composition service: derives only from the store's complete observation set."""
 if claim_level is ClaimLevel.TESTED:
  manifests=store._test_manifests_for(identity.implementation_identity_hash)
  required=(definition.control_id, definition.control_version)
  if any(any((b.get("control_id"),b.get("control_version"))==required and test.get("result")=="PASSED" for b in test["bindings"]) for m in manifests for test in m["tests"]):
   return AssertionVerdict.SUPPORTED
 return derive_assertion(definition, identity, store.observations(), claim_level=claim_level,
                         scope=scope, subjects=subjects, as_of_utc=as_of_utc)
