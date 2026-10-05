-- Indice (ended_at, started_at) en conversations (re-auditoria Jax#354, 2026-10-05).
-- stale_open_conversations filtra `ended_at IS NULL AND started_at < corte`: con el indice el plan
-- toca solo las abiertas (EXPLAIN antes/despues en tests/test_memory_aux_regressions.py). Idempotente.
-- NO se agrega indice (conversation_id, created_at) a messages: con 1.6k filas la subconsulta de ultima
-- actividad examina decenas de filas por conversacion candidata; no lo justifica el EXPLAIN y costaria
-- otra copia de la tabla con VECTOR KEY. Revisar si messages crece un orden de magnitud.
--
-- COSTO: ALGORITHM=COPY con LOCK=SHARED (conversations tiene FK entrantes de messages): la copia
-- bloquea escrituras en conversations ~1 s (364 filas). Ventana tranquila. Si lock_wait_timeout=10 se
-- agota (error 1205) no cambio nada: reintentar mas tarde, sin subir el limite.
SET SESSION lock_wait_timeout=10;
ALTER TABLE conversations ADD INDEX IF NOT EXISTS idx_conversations_open (ended_at, started_at), ALGORITHM=COPY, LOCK=SHARED;
