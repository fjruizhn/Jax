-- AUD-005: qualify legacy identity with the tenant of its bound memory object.
-- An existing binding without a matching object remains NULL and the NOT NULL
-- constraint fails closed; no tenant is inferred from a key or namespace.
ALTER TABLE memory_legacy_bindings
  ADD COLUMN tenant_id VARCHAR(128) NULL;

UPDATE memory_legacy_bindings b
JOIN memory_objects o ON o.memory_id=b.memory_id
SET b.tenant_id=o.tenant_id
WHERE b.tenant_id IS NULL;

ALTER TABLE memory_legacy_bindings
  MODIFY COLUMN tenant_id VARCHAR(128) NOT NULL,
  DROP PRIMARY KEY,
  ADD PRIMARY KEY (tenant_id, legacy_source_type, legacy_source_namespace, legacy_source_key);

ALTER TABLE memory_objects
  DROP INDEX uq_memory_legacy_binding,
  ADD UNIQUE KEY uq_memory_legacy_binding_tenant (tenant_id, legacy_source_type, legacy_source_namespace, legacy_source_key);
