CREATE SCHEMA IF NOT EXISTS jax_decisions;

CREATE TABLE IF NOT EXISTS jax_decisions.decision_records (
  decision_id CHAR(36) NOT NULL PRIMARY KEY,
  decision_input_hash CHAR(71) NOT NULL,
  policy_corpus_hash CHAR(71) NOT NULL,
  effective_authority_context_hash CHAR(71) NOT NULL,
  authority_ledger_checkpoint_hash CHAR(71) NOT NULL,
  resolver_identity VARCHAR(128) NOT NULL,
  resolver_version VARCHAR(32) NOT NULL,
  decision_record_hash CHAR(71) NOT NULL UNIQUE,
  canonical_record LONGBLOB NOT NULL,
  recorded_at_utc DATETIME(6) NOT NULL,
  INDEX idx_decision_input_hash (decision_input_hash),
  INDEX idx_decision_policy_corpus (policy_corpus_hash),
  INDEX idx_decision_effective_context (effective_authority_context_hash),
  INDEX idx_decision_authority_checkpoint (authority_ledger_checkpoint_hash)
);

DELIMITER //
CREATE TRIGGER jax_decisions.no_update_decision_records BEFORE UPDATE ON jax_decisions.decision_records
FOR EACH ROW BEGIN SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'decision_records immutable'; END//
CREATE TRIGGER jax_decisions.no_delete_decision_records BEFORE DELETE ON jax_decisions.decision_records
FOR EACH ROW BEGIN SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'decision_records immutable'; END//
DELIMITER ;
