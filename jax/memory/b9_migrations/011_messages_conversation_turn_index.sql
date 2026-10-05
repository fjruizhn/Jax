-- Indice (conversation_id, turn_number, id) en messages (auditoria Jax#354, 2026-10-05).
-- Las dos lecturas de mensajes por conversacion (db.py get_conversation_messages, ORDER BY turn_number,
-- id ASC, y get_last_session_messages, DESC) usaban idx_conversation (solo conversation_id) y hacian
-- `Using filesort`. Con este indice el orden sale del propio indice. Idempotente.
-- Es un ALTER sobre una tabla con VECTOR KEY: aplicar en ventana tranquila (1.6k filas hoy: segundos).
ALTER TABLE messages ADD INDEX IF NOT EXISTS idx_messages_conversation_turn (conversation_id, turn_number, id);
