"""Live DB inspection is composition-owned; no migration source is proof."""
from .errors import DatabaseObservationMismatchError
def inspect_database_control(connection_factory, control_id, control_version, *, database_scope_id, observed_at_utc):
 if control_id != "CTL.B6.ONE_DECISION_ONE_EXECUTION": raise DatabaseObservationMismatchError(control_id)
 con=connection_factory()
 try:
  cur=con.cursor(); cur.execute("SELECT COUNT(*) FROM information_schema.statistics WHERE table_schema='jax_execution' AND table_name='execution_records' AND column_name='decision_id' AND non_unique=0")
  if not cur.fetchone()[0]: raise DatabaseObservationMismatchError("unique decision_id absent")
  cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='jax_execution' AND table_name IN ('execution_records','execution_authorization_consumptions','execution_events')")
  if cur.fetchone()[0] != 3: raise DatabaseObservationMismatchError("execution structures absent")
  cur.execute("SELECT COUNT(*) FROM information_schema.triggers WHERE trigger_schema='jax_execution' AND event_object_table IN ('execution_records','execution_events')")
  if cur.fetchone()[0] < 4: raise DatabaseObservationMismatchError("immutability triggers absent")
  cur.execute("SELECT VERSION()")
  return {"database_scope_id":database_scope_id,"observed_at_utc":observed_at_utc,"control_id":control_id,"control_version":control_version,"server_version":cur.fetchone()[0],"installed":True}
 finally: con.close()
