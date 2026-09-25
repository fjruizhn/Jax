-- B9 additive logical namespace in the existing shared jax_memory schema.
-- This file is tracked for review and must be applied by the established
-- migration workflow; it is intentionally never run automatically at import.

CREATE TABLE IF NOT EXISTS memory_objects (
  memory_id CHAR(36) NOT NULL PRIMARY KEY,
  object_kind VARCHAR(32) NOT NULL,
  tenant_id VARCHAR(128) NOT NULL,
  created_at DATETIME(6) NOT NULL,
  legacy_source_type VARCHAR(64) NULL,
  legacy_source_namespace VARCHAR(255) NULL,
  legacy_source_key VARCHAR(255) NULL,
  UNIQUE KEY uq_memory_legacy_binding (legacy_source_type, legacy_source_namespace, legacy_source_key),
  KEY idx_memory_objects_tenant (tenant_id)
);

CREATE TABLE IF NOT EXISTS memory_revisions (
  revision_id CHAR(36) NOT NULL PRIMARY KEY,
  memory_id CHAR(36) NOT NULL,
  content_digest CHAR(71) NOT NULL,
  visibility VARCHAR(32) NOT NULL,
  user_id VARCHAR(128) NULL,
  project_id VARCHAR(128) NULL,
  lifecycle_state VARCHAR(32) NOT NULL,
  created_at DATETIME(6) NOT NULL,
  payload LONGBLOB NULL,
  provenance_status VARCHAR(64) NOT NULL,
  prior_revision_id CHAR(36) NULL,
  FOREIGN KEY (memory_id) REFERENCES memory_objects(memory_id),
  KEY idx_memory_revisions_scope (memory_id, visibility, user_id, project_id)
);

CREATE TABLE IF NOT EXISTS memory_provenance (
  provenance_id CHAR(36) NOT NULL PRIMARY KEY,
  revision_id CHAR(36) NOT NULL,
  source_revisions JSON NULL,
  transformation_id VARCHAR(128) NOT NULL,
  transformation_version VARCHAR(64) NOT NULL,
  actor_principal VARCHAR(255) NOT NULL,
  actor_type VARCHAR(64) NOT NULL,
  subject_user_id VARCHAR(128) NULL,
  provider VARCHAR(128) NULL,
  model VARCHAR(255) NULL,
  created_at DATETIME(6) NOT NULL,
  limitations TEXT NULL,
  FOREIGN KEY (revision_id) REFERENCES memory_revisions(revision_id)
);

CREATE TABLE IF NOT EXISTS memory_events (
  event_id CHAR(36) NOT NULL PRIMARY KEY,
  memory_id CHAR(36) NOT NULL,
  revision_id CHAR(36) NULL,
  event_kind VARCHAR(32) NOT NULL,
  actor_principal VARCHAR(255) NOT NULL,
  subject_user_id VARCHAR(128) NULL,
  authority_source VARCHAR(255) NOT NULL,
  occurred_at DATETIME(6) NOT NULL,
  details JSON NULL,
  compensates_event_id CHAR(36) NULL,
  FOREIGN KEY (memory_id) REFERENCES memory_objects(memory_id),
  KEY idx_memory_events_object (memory_id, occurred_at)
);

CREATE TABLE IF NOT EXISTS memory_projections (
  memory_id CHAR(36) NOT NULL PRIMARY KEY,
  current_revision_id CHAR(36) NULL,
  current_lifecycle_state VARCHAR(32) NULL,
  current_verification_state BOOLEAN NOT NULL,
  canonical_history_digest CHAR(71) NOT NULL,
  reconciliation_required BOOLEAN NOT NULL DEFAULT FALSE,
  FOREIGN KEY (memory_id) REFERENCES memory_objects(memory_id)
);

CREATE TABLE IF NOT EXISTS embedding_spaces (
  embedding_space_id CHAR(71) NOT NULL PRIMARY KEY,
  schema_version VARCHAR(64) NOT NULL,
  provider_runtime_class VARCHAR(128) NOT NULL,
  model_identifier VARCHAR(255) NOT NULL,
  model_version_or_digest VARCHAR(255) NULL,
  dimension INT NOT NULL,
  normalization VARCHAR(128) NOT NULL,
  distance_semantics VARCHAR(128) NOT NULL,
  created_at DATETIME(6) NOT NULL
);

CREATE TABLE IF NOT EXISTS embedding_generations (
  generation_id CHAR(36) NOT NULL PRIMARY KEY,
  revision_id CHAR(36) NOT NULL,
  embedding_space_id CHAR(71) NOT NULL,
  generated_at DATETIME(6) NOT NULL,
  embedding_payload LONGBLOB NULL,
  FOREIGN KEY (revision_id) REFERENCES memory_revisions(revision_id),
  FOREIGN KEY (embedding_space_id) REFERENCES embedding_spaces(embedding_space_id),
  KEY idx_embedding_generation_revision (revision_id, embedding_space_id)
);

DELIMITER //
CREATE TRIGGER no_update_memory_events BEFORE UPDATE ON memory_events
FOR EACH ROW BEGIN SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'memory_events append-only'; END//
CREATE TRIGGER no_delete_memory_events BEFORE DELETE ON memory_events
FOR EACH ROW BEGIN SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'memory_events append-only'; END//
DELIMITER ;
