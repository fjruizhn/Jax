-- Quita idx_conversation (conversation_id) de messages (re-auditoria Jax#354, 2026-10-05): lo cubre por
-- completo el indice de la 011 (idx_messages_conversation_turn, conversation_id es su prefijo), tambien
-- para la FK ON DELETE CASCADE. Verificado con EXPLAIN sobre la DDL real: los conteos por conversacion, las
-- dos lecturas ordenadas y el borrado en cascada pasan a usar el indice nuevo, sin filesort. Idempotente.
-- APLICAR DESPUES DE 011 (sin ella la FK se queda sin indice y MariaDB rechaza el DROP: falla cerrado).
--
-- COSTO: ALGORITHM=COPY con LOCK=SHARED (misma razon que la 011: VECTOR KEY y FK CASCADE).
-- Eso bloquea escrituras en messages ~1-3 s. Ventana tranquila. Si lock_wait_timeout=10 se agota (error 1205) no
-- cambio nada: reintentar mas tarde, sin subir el limite.
SET SESSION lock_wait_timeout=10;
ALTER TABLE messages DROP INDEX IF EXISTS idx_conversation, ALGORITHM=COPY, LOCK=SHARED;
