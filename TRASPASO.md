# TRASPASO — fix/deadlocks-save-message

**Estado: COMPLETO y verificado (2026-10-04).** Queda: revisión/auditoría de
escalón 3 y decisión de integración (fuera de mi alcance: no integrar ni
desplegar sin luz verde).

## SHA

- Rama: `fix/deadlocks-save-message` (desde `origin/master`).
- Head: **7c4b333** (`3019f67` migración UNIQUE; `f1f636b` + `cdb7098` fix y tests TDD).

## Qué era el bug

`save_message` es fire-and-forget y `@db_error_handler` traga los deadlocks
1213: cada uno **perdía el mensaje en silencio**. Medido en producción/F2-D:
20–48 deadlocks por cada ~2.400 turnos. Reproducido en base de prueba aislada
(`jax_memory_test_dlrepro`, 3308): **1563 deadlocks de 2400 intentos**, 850
guardados, turn_numbers duplicados.

## Diagnóstico (3 rondas, grafo InnoDB capturado cada vez)

1. Carrera MAX+1 sin UNIQUE: el UPDATE de embedding matcheaba ~35 filas del
   mismo turn y ciclaba con el INSERT ajeno vía las capas internas del índice
   vectorial HNSW (`messages#i#NN`).
2. Con INSERT+contador atómicos y candado FOR UPDATE sobre el rango, el
   next-key lock ataba la **frontera entre conversaciones adyacentes**.
3. El índice HNSW es **un solo recurso entre conversaciones**: todo INSERT
   escribe el vector default en ceros y compite por la misma zona del grafo;
   INSERT (conv 1) vs INSERT (conv 2) ciclaban en el supremo de la capa, con
   **1020** escapando al caller. MariaDB exige la columna del VECTOR KEY NOT
   NULL — no existe default NULL que difería el índice.

## Arreglo

- Transacción corta BEGIN…COMMIT: INSERT + contador juntos; rollback completo
  ante cualquier error.
- **Mutex GET_LOCK por base de datos** serializando todos los escritores del
  índice HNSW de este código (save_message, vectorización por id, backfill):
  un solo escritor a la vez → el ciclo es imposible por construcción.
  GET_LOCK no es transaccional: RELEASE_LOCK explícito en finally (el rollback
  de la víctima del deadlock no lo libera).
- Reintento acotado e idempotente ante 1213/1205/1062/**1020**, recalculando
  todo desde cero (la transacción muerta queda revertida entera).
- UNIQUE `uq_messages_conversation_turn` como respaldo estructural, con
  migrador que sanea duplicados históricos reenumerando al final en orden de
  id (sin pérdida) e idempotente. Corre en `ensure_schema` al arranque.

## Números

- Antes: 1563/2400 deadlocks, 850 guardados, turnos duplicados.
- Después: 2400/2400 guardados en **3 corridas limpias** (0 deadlocks, 0
  errores, contador exacto, sin turnos duplicados); carga del test **8/8**.
- Suite: archivo nuevo 7/7; B9 completo (11 archivos) 127 passed + 1 skip
  ajeno (exige la base `memb9_ci` del CI); regresión relacionada 32 passed +
  12 subtests; política de cableado 230 passed (pisos `^223` y `^7` intactos).

## Riesgos / notas

- La migración hace `ALTER TABLE messages ADD UNIQUE …` al arranque: en
  producción implica un rebuild de índice breve sobre `messages`.
- Escritores del índice que NO pasen por este código (otros repos, scripts
  como `scripts/migrar_embeddings.py`, jax-platform si duplica `db.py`) no
  toman el mutex: el reintento queda de respaldo pero el ciclo teórico con
  ellos no se puede cerrar desde aquí.
- El mutex es por base de datos (`jax_hnsw_write:<db>`): bases de test
  distintas no se bloquean entre sí.
- Se añadió `GRANT SELECT ON jax_memory.*` a `jax_test`@3308 (la siembra de
  `base_de_test` lo requiere; la base ya existía en el contenedor). Cambio de
  entorno de prueba, no de producción.
- No se desplegó ni integró nada.
