# Faro F1.1 paso 5: persistencia MariaDB

## Alcance

Implementar la migración `001_rule_authority_kernel.sql` y el adapter MariaDB del
`RuleDecisionStore` de #371. Mantener `get(request)` ligado a `request_id` y
`request_hash`; persistir decisiones antes de devolverlas; hacer `record` idempotente
para el mismo ID/hash y fallar cerrado ante hash ajeno, errores del store y `PERMIT`
hasta que el paso 6 pueda guardar también el `RulePermit` en la misma transacción.
La migración incluye las cuatro tablas de §11 y triggers de inmutabilidad. No se
implementan el modelo ni el consumo de `RulePermit`.

## Secuencia

1. Escribir pruebas de integración MariaDB para migración, columnas/índices, GRANTs,
   triggers (con usuario que sí tiene `UPDATE`/`DELETE`), lookup ligado al hash,
   persistencia antes del retorno, replay idempotente, rollback tras fallo intermedio,
   store desconectado y rechazo de `PERMIT`. Ejecutarlas contra MariaDB 12.3.3
   efímera (`sudo -n docker`, `--rm`, `--network none`) y verificar que fallen por
   ausencia de la implementación.
2. Crear migración con `rule_decisions`, `rule_permits`,
   `rule_permit_consumptions` y `rule_authority_audit_head`; índices sobre las
   búsquedas del diseño y triggers que impidan `UPDATE`/`DELETE` en las tres tablas
   append-only.
3. Implementar la cuenta de aplicación y sus GRANTs versionados, y el adapter
   transaccional usando las conexiones proporcionadas por el caller. Un commit
   idempotente devuelve la fila existente solo si su hash coincide; errores de DB
   se propagan como error cerrado.
4. Repetir pruebas hasta verde; matar por separado los mutantes de transacción,
   binding de hash y escritura posterior al retorno.
5. Añadir el job/paso de `policy.yml`, instalar `pymysql` en el job y fijar/verificar
   el piso en `ci/pisos.json`.
6. Ejecutar la suite del paso 5, suites Faro/Block 4 pertinentes, `piso.py verificar`,
   inspección de `EXPLAIN` de consultas reales y `git merge-tree` contra los tips
   actuales de #370/#371/#372. Actualizar `TRASPASO.md` en cada commit de trabajo.
7. Auditar el SHA exacto con un revisor Tier 3 distinto, borrar `TRASPASO.md`,
   escribir el reporte externo solicitado, empujar y abrir PR hacia
   `feat/faro-f1.1-evaluador`.

## Verificación

La prueba local crea solo un contenedor `mariadb:12.3.3` efímero sin red, por socket
Unix, y lo elimina incluso si falla. No usa `jax_memory`, 3308 ni `docker prune`.
Todo archivo temporal creado por esta tarea se elimina antes de cerrar.
