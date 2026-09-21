CREATE SCHEMA IF NOT EXISTS jax_evidence;
CREATE TABLE IF NOT EXISTS jax_evidence.control_definitions (control_definition_hash CHAR(71) PRIMARY KEY, control_id VARCHAR(128) NOT NULL, control_version INT UNSIGNED NOT NULL, canonical_definition JSON NOT NULL, UNIQUE(control_id, control_version)) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS jax_evidence.evidence_blobs (evidence_hash CHAR(71) PRIMARY KEY, size_bytes INT UNSIGNED NOT NULL, blob_bytes LONGBLOB NOT NULL) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS jax_evidence.implementation_identities (identity_hash CHAR(71) PRIMARY KEY, canonical_identity JSON NOT NULL) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS jax_evidence.evidence_artifacts (artifact_hash CHAR(71) PRIMARY KEY, canonical_artifact JSON NOT NULL) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS jax_evidence.evidence_artifact_blobs (artifact_hash CHAR(71) NOT NULL, evidence_hash CHAR(71) NOT NULL, PRIMARY KEY(artifact_hash,evidence_hash), FOREIGN KEY(artifact_hash) REFERENCES jax_evidence.evidence_artifacts(artifact_hash), FOREIGN KEY(evidence_hash) REFERENCES jax_evidence.evidence_blobs(evidence_hash)) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS jax_evidence.enforcement_observations (observation_id CHAR(36) PRIMARY KEY, observation_hash CHAR(71) NOT NULL UNIQUE, canonical_observation JSON NOT NULL) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS jax_evidence.observation_artifacts (observation_id CHAR(36) NOT NULL, artifact_hash CHAR(71) NOT NULL, PRIMARY KEY(observation_id, artifact_hash), FOREIGN KEY(observation_id) REFERENCES jax_evidence.enforcement_observations(observation_id), FOREIGN KEY(artifact_hash) REFERENCES jax_evidence.evidence_artifacts(artifact_hash)) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS jax_evidence.enforcement_assertions (assertion_hash CHAR(71) PRIMARY KEY, canonical_assertion JSON NOT NULL) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS jax_evidence.assertion_artifacts (assertion_hash CHAR(71) NOT NULL, artifact_hash CHAR(71) NOT NULL, PRIMARY KEY(assertion_hash, artifact_hash), FOREIGN KEY(assertion_hash) REFERENCES jax_evidence.enforcement_assertions(assertion_hash), FOREIGN KEY(artifact_hash) REFERENCES jax_evidence.evidence_artifacts(artifact_hash)) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS jax_evidence.assertion_observations (assertion_hash CHAR(71) NOT NULL, observation_id CHAR(36) NOT NULL, PRIMARY KEY(assertion_hash, observation_id), FOREIGN KEY(assertion_hash) REFERENCES jax_evidence.enforcement_assertions(assertion_hash), FOREIGN KEY(observation_id) REFERENCES jax_evidence.enforcement_observations(observation_id)) ENGINE=InnoDB;
DELIMITER //
CREATE TRIGGER jax_evidence.blobs_no_update BEFORE UPDATE ON jax_evidence.evidence_blobs FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='immutable evidence'//
CREATE TRIGGER jax_evidence.blobs_no_delete BEFORE DELETE ON jax_evidence.evidence_blobs FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='immutable evidence'//
CREATE TRIGGER jax_evidence.artifacts_no_update BEFORE UPDATE ON jax_evidence.evidence_artifacts FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='immutable evidence'//
CREATE TRIGGER jax_evidence.artifacts_no_delete BEFORE DELETE ON jax_evidence.evidence_artifacts FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='immutable evidence'//
CREATE TRIGGER jax_evidence.observations_no_update BEFORE UPDATE ON jax_evidence.enforcement_observations FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='immutable evidence'//
CREATE TRIGGER jax_evidence.observations_no_delete BEFORE DELETE ON jax_evidence.enforcement_observations FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='immutable evidence'//
CREATE TRIGGER jax_evidence.assertions_no_update BEFORE UPDATE ON jax_evidence.enforcement_assertions FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='immutable evidence'//
CREATE TRIGGER jax_evidence.assertions_no_delete BEFORE DELETE ON jax_evidence.enforcement_assertions FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='immutable evidence'//
CREATE TRIGGER jax_evidence.definitions_no_update BEFORE UPDATE ON jax_evidence.control_definitions FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='immutable evidence'//
CREATE TRIGGER jax_evidence.definitions_no_delete BEFORE DELETE ON jax_evidence.control_definitions FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='immutable evidence'//
DELIMITER ;
