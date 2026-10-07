# Faro F1.1 paso 5 — persistencia MariaDB

**Fecha:** 2026-10-07 (America/Tegucigalpa)

**Autor:** Codex, por encargo del usuario

**Rama:** `feat/faro-f1.1-storage-mariadb`

**Base:** `feat/faro-f1.1-evaluador` en `31eb14bf7903b3cd400abb0bc312dcd004a72879`

## Qué se hizo y por qué

Se añadió `policy/rule_authority/migrations/001_rule_authority_kernel.sql`, con
`rule_decisions`, `rule_permits`, `rule_permit_consumptions` y
`rule_authority_audit_head`. Las tablas append-only tienen triggers contra `UPDATE` y
`DELETE`; la migración crea los índices para lookup por solicitud, regla/estado,
ratificación, expiración y consumo.

`MariaDBRuleDecisionStore` implementa `get(request)` y `record(request, decision)`.
El lookup liga `request_id` y `request_hash`; el registro valida la correspondencia de
la solicitud, usa una transacción explícita para append de decisión y actualización del
audit head, valida la proyección y su hash en lectura, y confirma antes de responder.
El provisioning versiona los GRANTs de la cuenta de aplicación: lectura/inserción en
registros inmutables y lectura/actualización únicamente del audit head.

El alcance pedido excluye RulePermit y su consumo. Por eso el adapter falla cerrado
ante `PERMIT` hasta que el paso 6 pueda insertar decisión y permiso en una sola
transacción. Las cuatro tablas sí quedan definidas desde la migración.

## Lecciones técnicas

- Ejecutar la integración con MariaDB `12.3.3` en un contenedor efímero
  `--network none`, usando socket Unix, evita depender de bases del host.
- La prueba dispara un error en la actualización del audit head después del `INSERT`;
  con autocommit habilitado en la conexión, verifica que el rollback elimina la fila
  parcial y prueba que el adapter inicia su propia transacción.
- Las pruebas negativas distinguen el fallo cerrado del store, el hash ajeno y la
  persistencia antes del retorno. Los triggers se ejercitan con una cuenta que sí tiene
  `UPDATE` y `DELETE`.
- Las cantidades y unidades de prueba se cargan de `policy/faro/catalogo-topes.json`;
  no se inventó un catálogo paralelo.

## Verificación

- Integración de este paso: `1 passed` contra MariaDB `12.3.3`; piso
  `authority-rule-storage/mariadb` comprobado con `piso.py verificar`.
- Suites Faro/Block 4 seleccionadas: `228 passed`.
- Los mutantes temporales de transacción ausente, hash no comparado y commit posterior
  al retorno fueron detectados por aserciones distintas y luego retirados.
- `policy.yml` y `ci/pisos.json` se parsearon correctamente. Ruff no está instalado en
  el entorno local; se revisó sintaxis Python con `ast.parse`.

## Pendientes

- Esperar que #370 fije r6 y que #371 se reapile; luego repetir el rebase de esta rama,
  el piso, las suites y `git merge-tree` frente a #370/#371/#372. Al momento del registro,
  los tips observados son #370 `85cee9ca`, #371 `31eb14bf` y #372 `57547cfe`.
- Implementar `RulePermit`, su inserción atómica con `PERMIT`, el consumo single-use y
  las carreras en el paso 6.
- El checkpoint externo y la reconciliación post-crash de auditoría durable de §11
  requieren su contrato posterior; este paso implementa el audit head transaccional de
  MariaDB.

## Alternativas descartadas

- No se escriben permisos ni consumos con placeholders para simular el paso 6: `PERMIT`
  devuelve error cerrado hasta que exista el modelo y su commit atómico.
- No se añade una librería nueva ni se accede a `jax_memory`/puerto 3308; el adapter usa
  la conexión MariaDB inyectada y las pruebas crean su propia base efímera.
- No se reapiló sobre un SHA supuesto de #370: la rama de #371 seguía en `31eb14bf` al
  comprobar los PRs, y el encargo pide repetir el rebase cuando esos tips cambien.

## Quién decidió

El usuario fijó la base, el alcance del paso 5, los nombres de las tablas, los límites
del entorno efímero y las pruebas obligatorias. Ante la incompatibilidad entre persistir
`PERMIT`+`RulePermit` en §11 y excluir `RulePermit` del paso 5, Codex adoptó el fallo
cerrado temporal de `PERMIT`; la pregunta de aclaración quedó pendiente al momento de
este registro.
