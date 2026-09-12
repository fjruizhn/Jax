# Runbook — embeddings `nomic-embed-text` (768) → `bge-m3` (1024)

Ensayado completo el 2026-09-12 sobre una copia de producción en `jax_memory_test`:
`migrar` 43 s (1.607 messages + 113 facts, 0 fallidas), `activar` 0,8 s, `revertir` 0,8 s.
jax-platform no se toca (el chat importa `jax.memory.db`).

Por qué migrar (medido sobre `jax_memory` real, solo lectura, 2026-09-12):
- facts (113, 12 consultas): recall@1 nomic 5/12 → bge-m3 10/12.
- messages (1.607, 14 consultas): recall@1 5/14 → 11/14, recall@5 10/14 → 13/14,
  margen medio −0,005 → +0,063.

## 0. Deploy del código (no cambia comportamiento: los defaults son los de hoy)
- Merge del PR; `git -C ~/jax pull --ff-only`; verificar HEAD == `origin/master` en un comando aparte.
- Reiniciar `jax-platform` y `jax-las-manos`. Verificar: un turno en la Mesa web responde con memoria; journal sin errores.

## 1. Backup (antes de tocar datos)
```bash
set -a; . /etc/jax/.env; set +a
B=~/backups/jax_memory_pre_bge_$(date +%F-%H%M).sql      # contiene datos personales
( umask 077; mysqldump --single-transaction --skip-lock-tables --no-tablespaces \
    -h$JAX_DB_HOST -P$JAX_DB_PORT -u$JAX_DB_USER -p$JAX_DB_PASSWORD $JAX_DB_NAME \
    conversations messages facts > "$B" )                   # ~12 MB (medido)
```
Prueba de restauración (en `jax_memory_test`; borrar después, tiene datos reales):
```bash
mysql -h$JAX_DB_HOST -P$JAX_DB_PORT -u$JAX_DB_USER -p$JAX_DB_PASSWORD jax_memory_test < "$B"
# los tres conteos tienen que coincidir con producción (el 2026-09-12: 1607 / 113 / 352)
mysql ... jax_memory_test -e "SELECT COUNT(*) FROM messages; SELECT COUNT(*) FROM facts; SELECT COUNT(*) FROM conversations"
mysql ... jax_memory_test -e "DROP TABLE facts; DROP TABLE messages; DROP TABLE conversations"
```
Backup de la config: `sudo cp -a /etc/jax/.env /etc/jax/.env.pre-bge-$(date +%F-%H%M)`.

## 2. Migrar (el sistema sigue igual: el índice sigue en la columna vieja)
```bash
cd ~/jax && .venv/bin/python scripts/migrar_embeddings.py migrar --modelo bge-m3 --dim 1024 --columna embedding_bge_m3
```
~45 s; "fallidas" tiene que ser 0 en las dos tablas. Si no, correrlo otra vez (idempotente).

## 3. Config: agregar a `/etc/jax/.env`
Lo leen jax-platform, jax-las-manos, jax-memory-worker y jax-memory-synthesis:
```
JAX_MEMORY_EMBED_MODEL=bge-m3
JAX_MEMORY_EMBED_DIM=1024
JAX_MEMORY_EMBED_COLUMN=embedding_bge_m3
```

## 4. Activar (mueve el índice vectorial a la columna nueva; ~1 s)
```bash
.venv/bin/python scripts/migrar_embeddings.py activar --dim 1024 --columna embedding_bge_m3
```
Entre 4 y 5, un proceso con la config vieja busca sobre `embedding` SIN índice: funciona, con
búsqueda exacta (~79 ms en vez de ~1 ms). Hacer 4 y 5 seguidos.

## 5. Reiniciar, en este orden
`jax-platform`, `jax-las-manos`; `sudo systemctl start jax-memory-worker` (timer de 20 min) para no
esperar; synthesis lee el .env en su próxima corrida. Cerrar y abrir cualquier REPL de JAX abierto.
Las filas guardadas entre el paso 2 y este quedaron con la columna nueva en ceros: volver a correr
`migrar` (idempotente, solo toca esas); el worker también las repara en su pasada.

## 6. Verificar
- Env en los procesos: `sudo cat /proc/$(systemctl show jax-platform -p MainPID --value)/environ | tr '\0' '\n' | grep EMBED`
- Filas pendientes = 0: volver a correr `migrar` → 0 reparadas y 0 fallidas.
- `EXPLAIN` de la búsqueda usa `idx_embedding_bge_m3`.
- **Un turno de chat real con memoria** y journal de jax-platform sin "get_embedding fallo" ni
  "Embedding con dimension incorrecta". Obligatorio: `api/chat.py` atrapa CUALQUIER excepción al
  importar `MemoryDB` y sigue sin memoria — una config inválida no tumba el chat, lo deja sin
  memoria en silencio.

## Volver atrás (sin pérdida: la columna vieja nunca se toca)
**Desde el PR de seguimiento los defaults del código SON bge-m3.** Quitar las 3 variables, o
restaurar `.env.pre-bge-*` (que no las tiene), deja a los servicios en bge-m3: NO revierte nada.
1. FIJAR en `/etc/jax/.env` los valores de nomic (reemplazando los de bge-m3):
   ```
   JAX_MEMORY_EMBED_MODEL=nomic-embed-text
   JAX_MEMORY_EMBED_DIM=768
   JAX_MEMORY_EMBED_COLUMN=embedding
   ```
2. Reiniciar jax-platform y jax-las-manos (y el worker). Verificar el env en `/proc` como en el paso 6.
3. `.venv/bin/python scripts/migrar_embeddings.py revertir --columna embedding_bge_m3`
   con el `.env` cargado (devuelve el índice a `embedding` y borra la columna nueva; se NIEGA si la
   config activa sigue usando `embedding_bge_m3` — por eso 1 y 2 van antes). La columna activa la
   resuelve `embedding_config`, no un literal: sin la variable, cuenta el default.

Las filas creadas durante el período bge tienen la columna vieja en ceros: el worker de memoria las
re-embebe con nomic en su pasada.

## Después de ejecutar (PR de seguimiento) — hecho el 2026-09-12, salvo lo que tiene fecha
- [x] Regenerar `jax_memory_schema.sql` desde `SHOW CREATE TABLE` de producción (drift 9/9 OK).
- [x] Pasar los defaults de `jax/memory/embedding_config.py` a bge-m3 / 1024 / `embedding_bge_m3`,
  junto con el esquema: los tests de I/O de memoria crean las tablas desde el archivo y buscan sobre
  la columna configurada, así que uno sin el otro deja CI en rojo.
- [ ] **2026-09-26:** si no hubo que volver atrás, borrar la columna vieja `embedding` en otra
  migración, y borrar el backup del paso 1 (tiene datos personales).
- [x] Registrado en DEUDA.md.

## Ejecución en producción — 2026-09-12, ~15:50-15:56 CST
Paso 1: backup 11,9 MB restaurado en `_test` (1607/113/352 = producción) y borrado; `.env` respaldado
e idéntico por `cmp`. Paso 2: 1607 + 113, 0 fallidas, 43 s. Paso 4: 0,8 s. Paso 5: jax-platform
15:52:45, jax-las-manos 15:52:52, worker. Paso 6: variables en `/proc` de los dos servicios; `migrar`
→ 0/0; `EXPLAIN` de las consultas reales de messages y facts → `idx_embedding_bge_m3`; turno de chat
200 con sus dos filas embebidas; journal (con `sudo`: sin él `journalctl -u` da "No entries", que no
prueba nada) sin errores; y la búsqueda de `_semantic_context` en proceso encuentra una fila migrada
y una nueva a d=0,0000. Un control dentro del scope tiene que estar DENTRO del scope: la primera
fila elegida era de la sonda SP4 (`project_id` propio) y no podía volver.

Tiempo total estimado: ~5 min. La memoria nunca queda caída (a lo sumo búsqueda exacta unos segundos
entre 4 y 5).
