-- V2 RulePermit bindings.  NULL preserves immutable historical V1 rows:
-- their canonical payload/hash remains auditably decodable but they cannot be
-- consumed after the adapter observes the absent bindings.
ALTER TABLE rule_permits
  ADD COLUMN policy_repository_id TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL,
  ADD COLUMN capability_contract_id TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL,
  ADD CONSTRAINT chk_rule_permits_binding_pair CHECK (
    (policy_repository_id IS NULL AND capability_contract_id IS NULL)
    OR (CHAR_LENGTH(TRIM(policy_repository_id)) > 0
        AND CHAR_LENGTH(TRIM(capability_contract_id)) > 0)
  );

DELIMITER //
CREATE TRIGGER trg_rule_permits_require_policy_capability_binding
BEFORE INSERT ON rule_permits FOR EACH ROW
BEGIN
  IF NEW.policy_repository_id IS NULL
     OR NEW.capability_contract_id IS NULL
     OR CHAR_LENGTH(TRIM(NEW.policy_repository_id)) = 0
     OR CHAR_LENGTH(TRIM(NEW.capability_contract_id)) = 0 THEN
    SIGNAL SQLSTATE '45000'
      SET MESSAGE_TEXT = 'RulePermit requires repository and capability-contract identity';
  END IF;
END//
DELIMITER ;
