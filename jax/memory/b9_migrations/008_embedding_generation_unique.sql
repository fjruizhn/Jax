-- Una sola generacion por (revision, espacio de embeddings) (auditoria 2026-10-05).
-- Hasta hoy solo lo garantizaba un SELECT ... FOR UPDATE en reembed_memory; el indice
-- idx_embedding_generation_revision es NO unico. Idempotente (IF NOT EXISTS).
--
-- ANTES de aplicar: no debe haber duplicados; con ellos el ALTER falla (1062) y no cambia nada:
--   SELECT revision_id, embedding_space_id, COUNT(*) FROM embedding_generations
--    GROUP BY 1,2 HAVING COUNT(*) > 1;
-- (medido 2026-10-05 sobre jax_memory: 0 grupos, 137 filas). Aplicarla es una accion de despliegue.
ALTER TABLE embedding_generations
 ADD UNIQUE INDEX IF NOT EXISTS uq_embedding_generation_revision_space (revision_id, embedding_space_id);
