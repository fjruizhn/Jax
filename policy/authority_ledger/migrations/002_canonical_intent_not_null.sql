-- Upgrade databases created before canonical_intent became mandatory.
-- Fail closed on legacy rows containing NULL: do not fabricate signed data.
-- With a permissive sql_mode MariaDB would silently convert NULL to an empty
-- blob during MODIFY ... NOT NULL, so the NULL count is checked explicitly
-- (independent of sql_mode) and the ALTER itself runs under strict mode.
-- Any NULL aborts the upgrade and requires explicit integrity review.
SET SESSION sql_mode = 'STRICT_ALL_TABLES';

DELIMITER //
BEGIN NOT ATOMIC
  IF (SELECT COUNT(*) FROM jax_authority.authority_events
      WHERE canonical_intent IS NULL) > 0 THEN
    SIGNAL SQLSTATE '45000'
      SET MESSAGE_TEXT = 'authority_events has NULL canonical_intent; integrity review required';
  END IF;
  ALTER TABLE jax_authority.authority_events
    MODIFY canonical_intent LONGBLOB NOT NULL;
END//
DELIMITER ;
