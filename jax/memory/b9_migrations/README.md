# B9 shared memory migrations

The B9 schema is additive inside the physical `jax_memory` shared schema. It
is not executed by application import or worker startup. Apply it only through
the repository's reviewed database migration workflow.

`memory_events` is operational append-only history, not a B4 authority ledger
and not B7 evidence.

## Migration 005 lives only in Python

Migration 005 (project lifecycle -- ARCHIVED/HIDDEN/DISABLED, membership
provenance, idempotent project creation) is applied by
`project_authority_migrations.py::apply_project_authority_migration`, guarded
step-by-step against `information_schema`. There is deliberately **no**
`.sql` copy of it, unlike 003 (`003_project_scope_authority.sql` +
`project_authority_migrations.py`, H2 -- a known, already-accepted piece of
duplication). Adding a second source of truth for 005 would just grow that
same deuda; the guard-per-step Python hook is the only copy, and its bajada
is `revert_project_lifecycle_migration`, run by hand via
`scripts/b9_revertir_005.py --aplicar` (dry-run by default).

## Migraciones 007-009 (2026-10-05)

Solo SQL, aditivas e idempotentes, aplicadas a mano como el resto (nunca al arrancar):

- `007_extraction_job_events.sql`: auditoría del re-encolado de jobs en cuarentena.
- `008_embedding_generation_unique.sql`: UNIQUE (revision_id, embedding_space_id). Falla sin
  cambiar nada si hay duplicados: correr antes el SELECT que trae el propio archivo.
- `009_embedding_generation_attempts.sql`: contador de intentos fallidos por revisión.

Van ANTES del código que las usa (el worker de embeddings consulta `embedding_generation_attempts`
en cada corrida). Ver `docs/runbooks/memoria-cola-atascada.md`.
- `010_embedding_generation_drop_redundant_index.sql`: quita `idx_embedding_generation_revision`,
  cubierto por la UNIQUE de 008. Aplicar DESPUES de 008 (sin ella MariaDB rechaza el DROP con 1553:
  falla cerrado). Verificado con EXPLAIN y con las FK reales en una base desechable: las consultas por
  (revision, espacio) y por revision usan `uq_embedding_generation_revision_space`.
- Los intentos huerfanos de `embedding_generation_attempts` se limpian en la purga de contenido
  (`DELETE a FROM ...` en `content_purge`): las revisiones quedan de tombstone, asi que una FK
  ON DELETE CASCADE nunca dispararia.
- `011_messages_conversation_turn_index.sql`: indice `idx_messages_conversation_turn (conversation_id,
  turn_number, id)`; quita el `Using filesort` de las dos lecturas de mensajes por conversacion.
- `012_conversations_open_index.sql`: indice `idx_conversations_open (ended_at, started_at)` en `conversations`
  para `stale_open_conversations` (EXPLAIN antes/despues en las pruebas).
- `013_messages_drop_redundant_conversation_index.sql`: quita `idx_conversation` de `messages`, cubierto por el
  de la 011 (tambien para la FK CASCADE). Aplicar DESPUES de 011. Verificado con EXPLAIN sobre la DDL real.

Las tres van tambien en `jax_memory_schema.sql`.

**Costo de 011, 012 y 013 (medido):** son `ALGORITHM=COPY, LOCK=SHARED`. `INPLACE`, `NOCOPY` y `LOCK=NONE`
fallan en estas tablas (la `VECTOR KEY` de `messages` y la FK `ON DELETE CASCADE` lo impiden), asi que MariaDB
copia la tabla: durante ~1-3 s con el volumen actual se **bloquea escrituras** (las lecturas siguen). Cada
archivo empieza con `SET SESSION lock_wait_timeout=10;` para no esperar indefinidamente un bloqueo de
metadatos; si se agota (error 1205) el ALTER no cambio nada: **reintentar mas tarde, sin subir el limite**.
Aplicar en ventana tranquila. Probadas contra tablas creadas con la DDL del esquema versionado (`jax_memory_schema.sql`); verificado ademas por la auditoria con la DDL real de produccion, incluidas las `fk_*_project`. Incluye el camino del 1205.

## Marcha atrás de 007–013 (2026-10-05)

Orden INVERSO: 013, 012, 011, 010, 009, 008, 007. Todo con `IF [NOT] EXISTS`, repetible. Respaldar antes
(Principio VI). **Lo irreversible sin respaldo:** bajar 007 borra la auditoría del re-encolado y bajar
009 borra los contadores de intentos; exportarlos antes si importan. Los pasos sobre `messages` y
`conversations` son COPY con bloqueo de escrituras (ver arriba): misma ventana y mismo `lock_wait_timeout`.

```sql
SET SESSION lock_wait_timeout=10;
-- 013: restaurar idx_conversation ANTES de bajar el índice de la 011 (la FK por conversation_id lo necesita)
ALTER TABLE messages ADD INDEX IF NOT EXISTS idx_conversation (conversation_id), ALGORITHM=COPY, LOCK=SHARED;
-- 012
ALTER TABLE conversations DROP INDEX IF EXISTS idx_conversations_open, ALGORITHM=COPY, LOCK=SHARED;
-- 011 (solo después de restaurar idx_conversation)
ALTER TABLE messages DROP INDEX IF EXISTS idx_messages_conversation_turn, ALGORITHM=COPY, LOCK=SHARED;
-- 010: restaurar el índice ANTES de bajar la UNIQUE de 008 (la FK por revision_id lo necesita)
ALTER TABLE embedding_generations ADD INDEX IF NOT EXISTS idx_embedding_generation_revision (revision_id, embedding_space_id);
-- 009 (antes: SELECT * FROM embedding_generation_attempts si se quiere conservar el rastro)
DROP TABLE IF EXISTS embedding_generation_attempts;
-- 008 (solo después de restaurar el índice de la 010)
ALTER TABLE embedding_generations DROP INDEX IF EXISTS uq_embedding_generation_revision_space;
-- 007 (antes: SELECT * FROM memory_extraction_job_events si se quiere conservar la auditoría)
DROP TABLE IF EXISTS memory_extraction_job_events;
```

El código nuevo del worker usa 007 y 009 en cada corrida: si se baja el esquema hay que volver también
al código anterior (`5f4f5e0`), o el worker y la unidad de embeddings fallarán por tabla inexistente.
