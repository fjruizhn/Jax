-- Durable source identity and recovery markers; applying schema is a deployment action.
CREATE TABLE IF NOT EXISTS memory_extraction_jobs (
 conversation_id BIGINT PRIMARY KEY,
 state VARCHAR(16) NOT NULL DEFAULT 'READY',
 claim_token CHAR(36) NULL,
 lease_until DATETIME(6) NULL,
 attempts INT NOT NULL DEFAULT 0,
 next_attempt_at DATETIME(6) NULL,
 error_code VARCHAR(128) NULL,
 input_digest CHAR(71) NULL,
 frozen_output LONGTEXT NULL,
 output_digest CHAR(71) NULL,
 request_id CHAR(36) NULL,
 trace_id CHAR(36) NULL,
 run_id CHAR(36) NULL,
 updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
 KEY ix_memory_extraction_ready (state,next_attempt_at,lease_until)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS memory_extraction_results (
 conversation_id BIGINT NOT NULL,
 item_index INT NOT NULL,
 content_digest CHAR(71) NOT NULL,
 memory_id CHAR(36) NOT NULL,
 revision_id CHAR(36) NOT NULL,
 PRIMARY KEY (conversation_id,item_index),
 UNIQUE KEY uq_memory_extraction_revision (revision_id)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS memory_synthesis_jobs (
 job_key CHAR(64) PRIMARY KEY,
 tenant_id BIGINT NOT NULL, user_id BIGINT NOT NULL, project_id BIGINT NULL,
 source_revision_ids LONGTEXT NOT NULL,
 transformation_version VARCHAR(64) NOT NULL,
 state VARCHAR(16) NOT NULL,
 claim_token CHAR(36) NOT NULL,
 frozen_output LONGTEXT NULL,
 lease_until TIMESTAMP NULL,
 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
 updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS memory_synthesis_job_items (
 job_key CHAR(64) NOT NULL,item_key CHAR(64) NOT NULL,memory_id CHAR(36) NOT NULL,
 PRIMARY KEY(job_key,item_key)
) ENGINE=InnoDB;
-- Retrieval's tenant predicate and order must live on the same indexed row.
-- Existing tenants are derived exclusively from the canonical object table.
ALTER TABLE memory_revisions ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(128) NULL;
-- Older already-loaded API callers omit this additive column. The trigger
-- derives only the canonical object tenant; explicit mismatches still fail FK.
CREATE TRIGGER memory_revision_tenant_compat BEFORE INSERT ON memory_revisions
 FOR EACH ROW SET NEW.tenant_id=COALESCE(NEW.tenant_id,
 (SELECT tenant_id FROM memory_objects WHERE memory_id=NEW.memory_id));
UPDATE memory_revisions r JOIN memory_objects o ON o.memory_id=r.memory_id
 SET r.tenant_id=o.tenant_id WHERE r.tenant_id IS NULL;
ALTER TABLE memory_revisions MODIFY COLUMN tenant_id VARCHAR(128) NOT NULL;
ALTER TABLE memory_objects ADD UNIQUE INDEX IF NOT EXISTS uq_memory_object_tenant (memory_id,tenant_id);
ALTER TABLE memory_revisions ADD CONSTRAINT fk_memory_revision_tenant
 FOREIGN KEY (memory_id,tenant_id) REFERENCES memory_objects(memory_id,tenant_id);
ALTER TABLE memory_revisions ADD INDEX IF NOT EXISTS ix_memory_revision_private_feed
 (tenant_id,visibility,user_id,project_id,created_at,revision_id);
ALTER TABLE memory_revisions ADD INDEX IF NOT EXISTS ix_memory_revision_shared_feed
 (tenant_id,visibility,project_id,created_at,revision_id);
ALTER TABLE memory_provenance ADD INDEX IF NOT EXISTS ix_memory_provenance_revision_order
 (revision_id,created_at,provenance_id);
