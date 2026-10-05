-- Quita idx_embedding_generation_revision (revision_id, embedding_space_id): la UNIQUE de la
-- migracion 008 (uq_embedding_generation_revision_space, mismas columnas) la cubre por completo,
-- incluida la FK por revision_id. Idempotente. APLICAR DESPUES DE 008 (si no, la FK por
-- revision_id se queda sin indice y MariaDB rechaza el DROP: falla cerrado, sin cambios).
ALTER TABLE embedding_generations DROP INDEX IF EXISTS idx_embedding_generation_revision;
