-- B9 hardening is additive.  Existing revision.payload remains a transitional
-- compatibility column; all B9-managed payload writes use this separable row.

CREATE TABLE IF NOT EXISTS memory_revision_payloads (
  revision_id CHAR(36) NOT NULL PRIMARY KEY,
  payload LONGBLOB NULL,
  purged_at DATETIME(6) NULL,
  purge_reason VARCHAR(255) NULL,
  FOREIGN KEY (revision_id) REFERENCES memory_revisions(revision_id)
);

CREATE TABLE IF NOT EXISTS memory_legacy_bindings (
  legacy_source_type VARCHAR(64) NOT NULL,
  legacy_source_namespace VARCHAR(255) NOT NULL,
  legacy_source_key VARCHAR(255) NOT NULL,
  memory_id CHAR(36) NOT NULL,
  binding_state VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
  created_at DATETIME(6) NOT NULL,
  PRIMARY KEY (legacy_source_type, legacy_source_namespace, legacy_source_key),
  FOREIGN KEY (memory_id) REFERENCES memory_objects(memory_id)
);

ALTER TABLE memory_events
  ADD COLUMN IF NOT EXISTS actor_type VARCHAR(64) NULL,
  ADD COLUMN IF NOT EXISTS delegation VARCHAR(255) NULL,
  ADD COLUMN IF NOT EXISTS calling_component VARCHAR(255) NULL,
  ADD COLUMN IF NOT EXISTS request_id VARCHAR(255) NULL,
  ADD COLUMN IF NOT EXISTS trace_id VARCHAR(255) NULL;

-- ``memory_events`` remains operational append-only history.  These extra
-- fields distinguish the service actor from the subject it acts for.
