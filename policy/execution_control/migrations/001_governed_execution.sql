CREATE SCHEMA IF NOT EXISTS jax_execution;

CREATE TABLE IF NOT EXISTS jax_execution.execution_authorizations (
  authorization_id CHAR(36) PRIMARY KEY,
  decision_id CHAR(36) NOT NULL,
  execution_request_hash CHAR(71) NOT NULL,
  canonical_authorization_hash CHAR(71) NOT NULL UNIQUE,
  canonical_authorization JSON NOT NULL,
  created_at_utc DATETIME(6) NOT NULL
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS jax_execution.execution_authorization_consumptions (
  authorization_id CHAR(36) PRIMARY KEY,
  consumed_at_utc DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS jax_execution.human_approvals (
  human_approval_id CHAR(36) PRIMARY KEY,
  authorization_id CHAR(36) NOT NULL,
  approval_hash CHAR(71) NOT NULL UNIQUE,
  canonical_approval JSON NOT NULL
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS jax_execution.human_approval_consumptions (
  human_approval_id CHAR(36) PRIMARY KEY,
  consumed_at_utc DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS jax_execution.execution_records (
  execution_id CHAR(36) PRIMARY KEY,
  decision_id CHAR(36) NOT NULL UNIQUE,
  canonical_record_hash CHAR(71) NOT NULL UNIQUE,
  canonical_record JSON NOT NULL,
  created_at_utc DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS jax_execution.execution_events (
  event_sequence BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  execution_id CHAR(36) NOT NULL,
  state VARCHAR(40) NOT NULL,
  event_type VARCHAR(64) NOT NULL,
  event_at_utc DATETIME(6) NOT NULL,
  job_id VARCHAR(128) NULL
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS jax_execution.dry_run_artifacts (
  dry_run_id CHAR(36) PRIMARY KEY,
  execution_id CHAR(36) NOT NULL UNIQUE,
  artifact_hash CHAR(71) NOT NULL UNIQUE,
  canonical_artifact JSON NOT NULL
) ENGINE=InnoDB;

DELIMITER //
CREATE TRIGGER jax_execution.execution_records_no_update BEFORE UPDATE ON jax_execution.execution_records FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='immutable execution record'//
CREATE TRIGGER jax_execution.execution_records_no_delete BEFORE DELETE ON jax_execution.execution_records FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='immutable execution record'//
CREATE TRIGGER jax_execution.execution_events_no_update BEFORE UPDATE ON jax_execution.execution_events FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='immutable execution event'//
CREATE TRIGGER jax_execution.execution_events_no_delete BEFORE DELETE ON jax_execution.execution_events FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='immutable execution event'//
CREATE TRIGGER jax_execution.execution_authorizations_no_update BEFORE UPDATE ON jax_execution.execution_authorizations FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='immutable execution authorization'//
CREATE TRIGGER jax_execution.execution_authorizations_no_delete BEFORE DELETE ON jax_execution.execution_authorizations FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='immutable execution authorization'//
CREATE TRIGGER jax_execution.human_approvals_no_update BEFORE UPDATE ON jax_execution.human_approvals FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='immutable human approval'//
CREATE TRIGGER jax_execution.human_approvals_no_delete BEFORE DELETE ON jax_execution.human_approvals FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='immutable human approval'//
CREATE TRIGGER jax_execution.dry_run_artifacts_no_update BEFORE UPDATE ON jax_execution.dry_run_artifacts FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='immutable dry run artifact'//
CREATE TRIGGER jax_execution.dry_run_artifacts_no_delete BEFORE DELETE ON jax_execution.dry_run_artifacts FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='immutable dry run artifact'//
DELIMITER ;
