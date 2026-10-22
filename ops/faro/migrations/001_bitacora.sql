-- El Faro: la bitacora durable y encadenada (auditoria MAJOR-5). Una fila por llamada o rechazo.
--
-- `registro` es el JSON CANONICO exacto que se hasheo (LONGTEXT, no JSON: el tipo JSON reordena y rompe el
-- hash). `hash = sha256(hash_previo | cadena_id | seq | registro)`. El proceso del Puerto lleva SU cadena
-- (`cadena_id` aleatorio por arranque, `seq` desde 0, `hash_previo` en cero para la primera fila): como el
-- usuario de la aplicacion solo tiene INSERT (ver 002) no puede leer la cola de la cadena anterior, asi que
-- cada arranque abre una cadena nueva y su primera fila es `inicio_cadena`. Una cadena se verifica sola; un
-- hueco, un cambio o un reordenamiento dentro de ella lo delata `jax.faro.bitacora_db.verificar_cadena`.
-- UNIQUE(cadena_id, seq) impide duplicar o reinsertar una posicion.
CREATE TABLE IF NOT EXISTS `${base}`.faro_bitacora (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  cadena_id CHAR(32) NOT NULL,
  seq BIGINT UNSIGNED NOT NULL,
  momento DOUBLE NOT NULL,
  evento VARCHAR(32) NOT NULL,
  run_id VARCHAR(64) NULL,
  id_correlacion VARCHAR(128) NULL,
  decision VARCHAR(16) NULL,
  registro LONGTEXT NOT NULL,
  hash_previo CHAR(64) NOT NULL,
  hash CHAR(64) NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_faro_bitacora_cadena_seq (cadena_id, seq),
  KEY ix_faro_bitacora_run (run_id, id),
  KEY ix_faro_bitacora_momento (momento),
  KEY ix_faro_bitacora_evento (evento, id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
