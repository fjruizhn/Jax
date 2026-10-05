-- Intentos fallidos de embedding por (revision, espacio) (auditoria 2026-10-05).
-- Sin esto, una revision cuyo embedding siempre falla ocupa la cabeza de la cola
-- (ORDER BY created_at LIMIT 50) y bloquea a las que vienen detras. El worker salta la fila
-- tras JAX_MEMORY_EMBED_MAX_ATTEMPTS intentos y lo registra; vector-health sigue en rojo por ella.
-- Para reintentar una fila saltada: DELETE FROM embedding_generation_attempts WHERE revision_id=...
-- Sin FK a proposito: el borrado de revisiones (purga) no debe depender de esta tabla.
CREATE TABLE IF NOT EXISTS embedding_generation_attempts (
 revision_id CHAR(36) NOT NULL,
 embedding_space_id CHAR(71) NOT NULL,
 attempts INT NOT NULL DEFAULT 0,
 last_error VARCHAR(255) NULL,
 last_attempt_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
 PRIMARY KEY (revision_id, embedding_space_id)
) ENGINE=InnoDB;
