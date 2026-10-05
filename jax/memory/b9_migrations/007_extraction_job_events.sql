-- Auditoria de las decisiones humanas sobre la cola de extraccion (2026-10-05).
-- Hoy una sola: re-encolar un job en cuarentena tras revisarlo
-- (python -m jax.memory.worker --reencolar). Solo se agrega; idempotente.
-- Aplicarla es una accion de despliegue (README de esta carpeta): a mano, nunca al arrancar.
CREATE TABLE IF NOT EXISTS memory_extraction_job_events (
 event_id CHAR(36) NOT NULL PRIMARY KEY,
 conversation_id BIGINT NOT NULL,
 event_kind VARCHAR(32) NOT NULL,
 actor VARCHAR(128) NOT NULL,
 reason VARCHAR(512) NOT NULL,
 details LONGTEXT NULL,
 occurred_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
 KEY ix_memory_extraction_job_events_conv (conversation_id, occurred_at)
) ENGINE=InnoDB;
