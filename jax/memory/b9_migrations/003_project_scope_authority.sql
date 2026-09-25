-- JAX-owned operational authorization; not B9 memory, B4 ledger, or B7 evidence.
-- This tracked migration is never applied automatically by application code.

CREATE TABLE IF NOT EXISTS jax_project_scope (
  project_id INT NOT NULL, tenant_id INT NOT NULL, status VARCHAR(16) NOT NULL,
  created_at DATETIME(6) NOT NULL, created_by VARCHAR(128) NOT NULL,
  updated_at DATETIME(6) NOT NULL, version BIGINT NOT NULL DEFAULT 1,
  PRIMARY KEY (project_id), UNIQUE KEY uq_jax_project_scope_project_tenant (project_id, tenant_id),
  KEY idx_jax_project_scope_tenant_status (tenant_id, status),
  CONSTRAINT fk_jax_project_scope_project FOREIGN KEY (project_id) REFERENCES projects(id),
  CONSTRAINT fk_jax_project_scope_tenant FOREIGN KEY (tenant_id) REFERENCES jax_tenants(tenant_id),
  CONSTRAINT chk_jax_project_scope_status CHECK (status IN ('ACTIVE','DISABLED'))
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS jax_project_membership (
  membership_id CHAR(36) NOT NULL, project_id INT NOT NULL, tenant_id INT NOT NULL,
  user_id INT NOT NULL, project_role VARCHAR(16) NOT NULL, status VARCHAR(16) NOT NULL,
  created_at DATETIME(6) NOT NULL, created_by VARCHAR(128) NOT NULL,
  updated_at DATETIME(6) NOT NULL, version BIGINT NOT NULL DEFAULT 1,
  PRIMARY KEY (membership_id), UNIQUE KEY uq_jax_project_membership_subject (project_id, user_id),
  KEY idx_jax_project_membership_lookup (project_id, user_id, status),
  KEY idx_jax_project_membership_tenant_user (tenant_id, user_id),
  CONSTRAINT fk_jax_project_membership_scope FOREIGN KEY (project_id, tenant_id)
    REFERENCES jax_project_scope(project_id, tenant_id),
  CONSTRAINT fk_jax_project_membership_user FOREIGN KEY (user_id) REFERENCES jax_users(user_id),
  CONSTRAINT chk_jax_project_membership_role CHECK (project_role IN ('VIEWER','CONTRIBUTOR','REVIEWER','OWNER')),
  CONSTRAINT chk_jax_project_membership_status CHECK (status IN ('ACTIVE','REVOKED'))
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS jax_project_membership_event (
  event_id CHAR(36) NOT NULL, operation VARCHAR(32) NOT NULL,
  actor_principal VARCHAR(255) NOT NULL, actor_type VARCHAR(64) NOT NULL,
  project_id INT NOT NULL, target_user_id INT NULL, tenant_id INT NOT NULL,
  old_project_role VARCHAR(16) NULL, new_project_role VARCHAR(16) NULL,
  old_status VARCHAR(16) NULL, new_status VARCHAR(16) NULL, occurred_at DATETIME(6) NOT NULL,
  request_id VARCHAR(255) NULL, trace_id VARCHAR(255) NULL, PRIMARY KEY (event_id),
  KEY idx_jax_project_membership_event_project (project_id, occurred_at),
  CONSTRAINT fk_jax_project_membership_event_scope FOREIGN KEY (project_id, tenant_id)
    REFERENCES jax_project_scope(project_id, tenant_id)
) ENGINE=InnoDB;

CREATE TRIGGER IF NOT EXISTS no_update_jax_project_membership_event BEFORE UPDATE ON jax_project_membership_event
FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'jax_project_membership_event append-only';
CREATE TRIGGER IF NOT EXISTS no_delete_jax_project_membership_event BEFORE DELETE ON jax_project_membership_event
FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'jax_project_membership_event append-only';
