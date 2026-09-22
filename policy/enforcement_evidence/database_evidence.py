"""Fixed-composition live MariaDB inspection for installed control state."""
from __future__ import annotations
import json
import uuid
import hashlib
from datetime import datetime, timezone
from .errors import DatabaseObservationMismatchError
from .control_registry import load_control_definition
from .models import (EvidenceArtifact, EvidenceBlobRef, EvidenceClass,
                     EvidenceSubject, EvidenceSubjectType, EvidenceTrustDomain,
                     EvidenceType, EnforcementObservation, ObservationOutcome)

def inspect_database_control(connection_factory, control_id, control_version, *, deployment_id, observed_at_utc):
 """Read installed state only; callers cannot supply a schema projection."""
 if control_id != "CTL.B6.ONE_DECISION_ONE_EXECUTION" or control_version != 1: raise DatabaseObservationMismatchError(control_id)
 if not deployment_id: raise DatabaseObservationMismatchError("deployment_id required")
 if not isinstance(observed_at_utc, datetime) or observed_at_utc.tzinfo is None or observed_at_utc.astimezone(timezone.utc) > datetime.now(timezone.utc): raise DatabaseObservationMismatchError("observed_at_utc invalid")
 con=connection_factory()
 try:
  cur=con.cursor(); cur.execute("SELECT VERSION(), DATABASE(), @@server_id, @@hostname"); server_version,database_name,server_id,hostname=cur.fetchone()
  # This is derived from the inspected endpoint, not a request label.
  database_scope_id="dbscope:sha256:"+hashlib.sha256((str(deployment_id)+"|"+str(server_id)+"|"+str(hostname)+"|"+str(database_name)).encode()).hexdigest()
  cur.execute("SELECT index_name FROM information_schema.statistics WHERE table_schema='jax_execution' AND table_name='execution_records' AND column_name='decision_id' AND non_unique=0")
  unique_indexes=tuple(sorted(r[0] for r in cur.fetchall()))
  if not unique_indexes: raise DatabaseObservationMismatchError("unique decision_id absent")
  cur.execute("SELECT index_name FROM information_schema.statistics WHERE table_schema='jax_execution' AND table_name='execution_authorization_consumptions' AND non_unique=0")
  consumption_indexes=tuple(sorted(r[0] for r in cur.fetchall()))
  if not consumption_indexes: raise DatabaseObservationMismatchError("authorization consumption uniqueness absent")
  cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='jax_execution' AND table_name IN ('execution_records','execution_authorization_consumptions','execution_events')")
  tables=tuple(sorted(r[0] for r in cur.fetchall()))
  if tables != ('execution_authorization_consumptions','execution_events','execution_records'): raise DatabaseObservationMismatchError("execution structures absent")
  cur.execute("SELECT trigger_name,event_object_table,event_manipulation FROM information_schema.triggers WHERE trigger_schema='jax_execution' AND event_object_table IN ('execution_records','execution_events')")
  triggers=tuple(sorted(tuple(r) for r in cur.fetchall()))
  required={("execution_records","UPDATE"),("execution_records","DELETE"),("execution_events","UPDATE"),("execution_events","DELETE")}
  if not required.issubset({(r[1],r[2]) for r in triggers}): raise DatabaseObservationMismatchError("execution immutability triggers absent")
  return {"database_scope_id":database_scope_id,"observed_at_utc":observed_at_utc,"control_id":control_id,"control_version":control_version,"server_version":server_version,"server_id":server_id,"hostname":hostname,"database_name":database_name,"deployment_id":deployment_id,"execution_records_unique_indexes":unique_indexes,"consumption_unique_indexes":consumption_indexes,"tables":tables,"triggers":triggers,"installed":True}
 finally: con.close()

class DatabaseControlInspector:
 """Application-startup dependency which turns a live read into evidence."""
 def __init__(self, connection_factory, recorder, *, deployment_id):
  self._connection_factory=connection_factory; self._recorder=recorder; self._deployment_id=deployment_id
 def inspect_one_decision_one_execution(self):
  now=datetime.now(timezone.utc)
  installed=inspect_database_control(self._connection_factory,"CTL.B6.ONE_DECISION_ONE_EXECUTION",1,deployment_id=self._deployment_id,observed_at_utc=now)
  # Scope is derived by the live inspector.  The recorder receives the
  # observed profile as a whole and never accepts a caller-selected label.
  return self._recorder._record_database_profile(installed)
