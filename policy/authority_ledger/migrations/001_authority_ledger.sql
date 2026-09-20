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
  canonical_intent LONGBLOB NOT NULL,
  evidence_refs LONGBLOB NOT NULL,
  previous_event_hash CHAR(71) NULL,
  event_hash CHAR(71) NOT NULL UNIQUE,
  signature TEXT NOT NULL,
  recorded_at_utc DATETIME(6) NOT NULL,
  CONSTRAINT authority_events_append_only_sequence CHECK (sequence > 0)
);

-- Deliberately no UPDATE/DELETE grants in this migration.  Any repair is a new
-- signed event and replay validates the predecessor hash chain.
