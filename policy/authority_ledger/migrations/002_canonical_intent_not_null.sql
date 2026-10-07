-- Upgrade databases created before canonical_intent became mandatory.
-- MariaDB rejects this ALTER when legacy rows contain NULL; stop the upgrade
-- and require explicit integrity review instead of fabricating signed data.
ALTER TABLE jax_authority.authority_events
  MODIFY canonical_intent LONGBLOB NOT NULL;
