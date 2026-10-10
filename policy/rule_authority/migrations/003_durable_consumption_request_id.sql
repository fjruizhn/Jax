-- Identidad durable del intento de consumo para recuperar únicamente el mismo retry.
-- NULL conserva filas históricas, cuyo ID de intento no puede reconstruirse sin
-- reescribir bytes firmados/encadenados; esas filas se niegan a retries nuevos.
ALTER TABLE rule_permit_consumptions
  ADD COLUMN consumption_request_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL,
  ADD UNIQUE KEY uq_rule_permit_consumptions_request_id (consumption_request_id);

DELIMITER //
CREATE TRIGGER trg_rule_permit_consumptions_require_request_id
BEFORE INSERT ON rule_permit_consumptions FOR EACH ROW
BEGIN
  IF NEW.consumption_request_id IS NULL
     OR NEW.consumption_request_id NOT REGEXP '^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$' THEN
    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'consumption_request_id must be UUIDv7';
  END IF;
END//
DELIMITER ;
