-- Indice (conversation_id, turn_number, id) en messages (auditoria Jax#354, 2026-10-05).
-- Las dos lecturas de mensajes por conversacion (db.py get_conversation_messages, ORDER BY turn_number,
-- id ASC, y get_last_session_messages, DESC) usaban idx_conversation (solo conversation_id) y hacian
-- `Using filesort`. Con este indice el orden sale del propio indice. Idempotente.
--
-- COSTO, MEDIDO: es ALGORITHM=COPY con LOCK=SHARED. INPLACE, NOCOPY y LOCK=NONE FALLAN en esta tabla
-- (la VECTOR KEY y la FK ON DELETE CASCADE lo impiden), asi que MariaDB copia la tabla: durante la
-- copia (~1-3 s con 1.6k filas) se bloquea escrituras en messages (las lecturas siguen). Aplicar en
-- ventana tranquila. lock_wait_timeout=10 limita la espera por el bloqueo de metadatos: si un
-- escritor largo lo retiene y se agota (error 1205), el ALTER no cambio nada: reintentar mas tarde,
-- sin subir el limite.
SET SESSION lock_wait_timeout=10;
ALTER TABLE messages ADD INDEX IF NOT EXISTS idx_messages_conversation_turn (conversation_id, turn_number, id), ALGORITHM=COPY, LOCK=SHARED;
