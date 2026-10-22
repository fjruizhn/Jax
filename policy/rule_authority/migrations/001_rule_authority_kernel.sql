-- Faro F1.1 durable decision store. One migration is applied to an isolated schema.
CREATE DATABASE IF NOT EXISTS jax_rule_authority
  CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;
USE jax_rule_authority;

CREATE TABLE rule_authority_audit_head (
  singleton TINYINT UNSIGNED NOT NULL PRIMARY KEY,
  audit_sequence BIGINT UNSIGNED NOT NULL,
  head_hash CHAR(71) CHARACTER SET ascii COLLATE ascii_bin NULL,
  head_kind VARCHAR(32) CHARACTER SET ascii COLLATE ascii_bin NULL,
  head_key VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL,
  CONSTRAINT chk_rule_authority_audit_head_singleton CHECK (singleton = 1),
  CONSTRAINT chk_rule_authority_audit_head_empty CHECK (
    (audit_sequence = 0 AND head_hash IS NULL AND head_kind IS NULL AND head_key IS NULL)
    OR (audit_sequence > 0 AND head_hash IS NOT NULL AND head_kind IS NOT NULL AND head_key IS NOT NULL)
  )
) ENGINE=InnoDB;

INSERT INTO rule_authority_audit_head
  (singleton, audit_sequence, head_hash, head_kind, head_key)
VALUES (1, 0, NULL, NULL, NULL);

CREATE TABLE rule_decisions (
  request_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL PRIMARY KEY,
  request_hash CHAR(71) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  request_catalog_hash CHAR(71) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  required_rule_id VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  status VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  reason_code VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL,
  decided_at_utc DATETIME(6) NOT NULL,
  canonical_decision LONGBLOB NOT NULL,
  previous_record_hash CHAR(71) CHARACTER SET ascii COLLATE ascii_bin NULL,
  record_hash CHAR(71) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  audit_sequence BIGINT UNSIGNED NOT NULL,
  UNIQUE KEY uq_rule_decisions_audit_sequence (audit_sequence),
  UNIQUE KEY uq_rule_decisions_permit_binding
    (request_id, request_hash, status, required_rule_id),
  KEY ix_rule_decisions_rule_status (required_rule_id, status),
  CONSTRAINT chk_rule_decisions_status CHECK (status IN ('DENY','MISSING_RULE','PERMIT')),
  CONSTRAINT chk_rule_decisions_request_hash CHECK (request_hash REGEXP '^sha256:[0-9a-f]{64}$'),
  CONSTRAINT chk_rule_decisions_catalog_hash CHECK (request_catalog_hash REGEXP '^sha256:[0-9a-f]{64}$'),
  CONSTRAINT chk_rule_decisions_record_hash CHECK (record_hash REGEXP '^sha256:[0-9a-f]{64}$'),
  CONSTRAINT chk_rule_decisions_previous_hash CHECK (
    previous_record_hash IS NULL OR previous_record_hash REGEXP '^sha256:[0-9a-f]{64}$'
  ),
  CONSTRAINT chk_rule_decisions_reason CHECK (
    (status = 'PERMIT' AND reason_code IS NULL)
    OR (status = 'DENY' AND reason_code IS NOT NULL)
    OR (status = 'MISSING_RULE' AND reason_code = 'RULE_NOT_FOUND')
  )
) ENGINE=InnoDB;

CREATE TABLE rule_permits (
  permit_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL PRIMARY KEY,
  request_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  request_hash CHAR(71) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  decision_status VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL DEFAULT 'PERMIT',
  rule_id VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  rule_path VARCHAR(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  rule_blob_oid CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  rule_content_hash CHAR(71) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  policy_revision CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  policy_tree_oid CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  policy_snapshot_hash CHAR(71) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  ratification_event_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  authority_ledger_checkpoint LONGBLOB NOT NULL,
  stop_checkpoint LONGBLOB NOT NULL,
  capability_id VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  capability_version VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  capability_class VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  capability_limits LONGBLOB NOT NULL,
  issued_at_utc DATETIME(6) NOT NULL,
  expires_at_utc DATETIME(6) NOT NULL,
  permit_hash CHAR(71) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  canonical_payload LONGBLOB NOT NULL,
  previous_record_hash CHAR(71) CHARACTER SET ascii COLLATE ascii_bin NULL,
  record_hash CHAR(71) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  audit_sequence BIGINT UNSIGNED NOT NULL UNIQUE,
  KEY ix_rule_permits_request_hash (request_id, request_hash),
  UNIQUE KEY uq_rule_permits_request_binding (permit_id, request_hash),
  KEY ix_rule_permits_rule_id (rule_id),
  KEY ix_rule_permits_ratification (ratification_event_id),
  KEY ix_rule_permits_expiration (expires_at_utc),
  CONSTRAINT fk_rule_permits_decision FOREIGN KEY
    (request_id, request_hash, decision_status, rule_id)
    REFERENCES rule_decisions (request_id, request_hash, status, required_rule_id),
  CONSTRAINT chk_rule_permits_hashes CHECK (
    request_hash REGEXP '^sha256:[0-9a-f]{64}$'
    AND rule_content_hash REGEXP '^sha256:[0-9a-f]{64}$'
    AND policy_snapshot_hash REGEXP '^sha256:[0-9a-f]{64}$'
    AND permit_hash REGEXP '^sha256:[0-9a-f]{64}$'
    AND record_hash REGEXP '^sha256:[0-9a-f]{64}$'
  ),
  CONSTRAINT chk_rule_permits_git_oids CHECK (
    CHAR_LENGTH(rule_blob_oid) IN (40,64)
    AND rule_blob_oid REGEXP '^([0-9a-f]{40}|[0-9a-f]{64})$'
    AND CHAR_LENGTH(policy_revision) IN (40,64)
    AND policy_revision REGEXP '^([0-9a-f]{40}|[0-9a-f]{64})$'
    AND CHAR_LENGTH(policy_tree_oid) IN (40,64)
    AND policy_tree_oid REGEXP '^([0-9a-f]{40}|[0-9a-f]{64})$'
  ),
  CONSTRAINT chk_rule_permits_previous_hash CHECK (
    previous_record_hash IS NULL OR previous_record_hash REGEXP '^sha256:[0-9a-f]{64}$'
  ),
  CONSTRAINT chk_rule_permits_time CHECK (expires_at_utc > issued_at_utc),
  CONSTRAINT chk_rule_permits_decision_status CHECK (decision_status = 'PERMIT')
) ENGINE=InnoDB;

CREATE TABLE rule_permit_consumptions (
  permit_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL PRIMARY KEY,
  request_hash CHAR(71) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  consumed_at_utc DATETIME(6) NOT NULL,
  canonical_payload LONGBLOB NOT NULL,
  consumption_hash CHAR(71) CHARACTER SET ascii COLLATE ascii_bin NOT NULL UNIQUE,
  previous_record_hash CHAR(71) CHARACTER SET ascii COLLATE ascii_bin NULL,
  record_hash CHAR(71) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  audit_sequence BIGINT UNSIGNED NOT NULL UNIQUE,
  CONSTRAINT fk_rule_permit_consumptions_permit FOREIGN KEY (permit_id, request_hash)
    REFERENCES rule_permits (permit_id, request_hash),
  CONSTRAINT chk_rule_permit_consumptions_hashes CHECK (
    request_hash REGEXP '^sha256:[0-9a-f]{64}$'
    AND consumption_hash REGEXP '^sha256:[0-9a-f]{64}$'
    AND record_hash REGEXP '^sha256:[0-9a-f]{64}$'
  ),
  CONSTRAINT chk_rule_permit_consumptions_previous_hash CHECK (
    previous_record_hash IS NULL OR previous_record_hash REGEXP '^sha256:[0-9a-f]{64}$'
  )
) ENGINE=InnoDB;

DELIMITER //
CREATE TRIGGER trg_rule_decisions_no_update
BEFORE UPDATE ON rule_decisions FOR EACH ROW
BEGIN
  SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'rule_decisions are append-only';
END//
CREATE TRIGGER trg_rule_decisions_no_delete
BEFORE DELETE ON rule_decisions FOR EACH ROW
BEGIN
  SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'rule_decisions are append-only';
END//
CREATE TRIGGER trg_rule_permits_no_update
BEFORE UPDATE ON rule_permits FOR EACH ROW
BEGIN
  SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'rule_permits are append-only';
END//
CREATE TRIGGER trg_rule_permits_no_delete
BEFORE DELETE ON rule_permits FOR EACH ROW
BEGIN
  SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'rule_permits are append-only';
END//
CREATE TRIGGER trg_rule_permit_consumptions_no_update
BEFORE UPDATE ON rule_permit_consumptions FOR EACH ROW
BEGIN
  SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'rule_permit_consumptions are append-only';
END//
CREATE TRIGGER trg_rule_permit_consumptions_no_delete
BEFORE DELETE ON rule_permit_consumptions FOR EACH ROW
BEGIN
  SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'rule_permit_consumptions are append-only';
END//
CREATE TRIGGER trg_rule_decisions_no_permit_until_step6
BEFORE INSERT ON rule_decisions FOR EACH ROW
BEGIN
  IF NEW.status = 'PERMIT' THEN
    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'PERMIT persistence requires step 6 atomic adapter';
  END IF;
END//
DELIMITER ;
