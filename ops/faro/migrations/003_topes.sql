-- El Faro 0.3b (P-4): el contador de los topes. Una fila por (clave, periodo); `clave` es "<tenant>|<recurso>".
--
-- El conteo es UN solo UPDATE atomico:  UPDATE faro_topes SET usado = LAST_INSERT_ID(usado + x)
--   WHERE clave = ? AND periodo = ? AND usado + x <= tope
-- (sin tope: la misma sentencia sin la ultima condicion, que MIDE). El tope NO se guarda aqui: lo trae cada
-- llamada, ya resuelto por quien evalua la regla (policy/faro/*.yaml); la tabla solo cuenta. Sin regla = sin
-- tope, pero con medicion y aviso. La clave primaria cubre el WHERE (se comprueba con EXPLAIN en las pruebas).
CREATE TABLE IF NOT EXISTS `${base}`.faro_topes (
  clave VARCHAR(128) NOT NULL,
  periodo VARCHAR(32) NOT NULL,
  usado BIGINT UNSIGNED NOT NULL DEFAULT 0,
  PRIMARY KEY (clave, periodo)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin
