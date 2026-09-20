-- Block 4 authoritative event log.  The application DB user receives SELECT/INSERT
-- only; UPDATE/DELETE are intentionally absent.  Materialized state is a cache.
CREATE SCHEMA IF NOT EXISTS jax_authority;

CREATE TABLE IF NOT EXISTS jax_authority.authority_ledger_genesis (
  singleton TINYINT NOT NULL PRIMARY KEY,
  canonical_genesis LONGBLOB NOT NULL,
  genesis_hash CHAR(71) NOT NULL UNIQUE,
  CONSTRAINT exactly_one_genesis CHECK (singleton = 1)
);

CREATE TABLE IF NOT EXISTS jax_authority.authority_events (
  sequence BIGINT UNSIGNED NOT NULL PRIMARY KEY,
  event_id CHAR(36) NOT NULL UNIQUE,
  event_type VARCHAR(64) NOT NULL,
  actor_id VARCHAR(128) NOT NULL,
  canonical_intent LONGBLOB NULL,
  canonical_event LONGBLOB NOT NULL,
  evidence_refs LONGBLOB NOT NULL,
  previous_event_hash CHAR(71) NULL,
  event_hash CHAR(71) NOT NULL UNIQUE,
  signature TEXT NOT NULL,
  recorded_at_utc DATETIME(6) NOT NULL,
  CONSTRAINT authority_events_append_only_sequence CHECK (sequence > 0)
);

CREATE TABLE IF NOT EXISTS jax_authority.authority_ledger_head (
  singleton TINYINT NOT NULL PRIMARY KEY,
  sequence BIGINT UNSIGNED NOT NULL,
  head_event_id CHAR(36) NULL,
  head_event_hash CHAR(71) NULL,
  CONSTRAINT authority_head_singleton CHECK (singleton = 1)
);

DELIMITER //
CREATE TRIGGER jax_authority.no_update_authority_events BEFORE UPDATE ON jax_authority.authority_events
FOR EACH ROW BEGIN SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'authority_events append-only'; END//
CREATE TRIGGER jax_authority.no_delete_authority_events BEFORE DELETE ON jax_authority.authority_events
FOR EACH ROW BEGIN SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'authority_events append-only'; END//
CREATE TRIGGER jax_authority.no_update_genesis BEFORE UPDATE ON jax_authority.authority_ledger_genesis
FOR EACH ROW BEGIN SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'genesis immutable'; END//
CREATE TRIGGER jax_authority.no_delete_genesis BEFORE DELETE ON jax_authority.authority_ledger_genesis
FOR EACH ROW BEGIN SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'genesis immutable'; END//
DELIMITER ;

-- Deliberately no UPDATE/DELETE grants in this migration.  Any repair is a new
-- signed event and replay validates the predecessor hash chain.
