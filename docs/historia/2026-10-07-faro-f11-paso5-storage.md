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
También liga `request_catalog_hash` para detectar que el mismo ID/hash llegue con otro
catálogo, incluso si la decisión no incluye el campo opcional `catalogo`.
El provisioning versiona los GRANTs de la cuenta de aplicación: lectura/inserción en
registros inmutables y lectura/actualización únicamente del audit head. Antes, revoca
todos los privilegios anteriores de la cuenta para que una provisión repetida también
converja al mínimo privilegio.

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
- El audit head se bloquea antes de consultar un `request_id` ausente. Así los
  escritores se serializan antes de tomar gap locks de la tabla de decisiones.
- El modelo heredado incluye `RuleDecision.catalogo`; se incluye en el payload canónico
  para que `record()`/`get()` conserven el valor inmutable del store en memoria. El adapter
  verifica que el catálogo sea un mapa de tuplas de texto con la forma del pin de la
  solicitud; un catálogo inventado o malformado se rechaza.
- Los campos OID de permisos admiten 40 o 64 dígitos hexadecimales, en línea con los
  pins SHA-1/SHA-256 del snapshot; la prueba acepta ambos tamaños y rechaza longitud 41
  y salto de línea terminal.
- Claves foráneas compuestas solo permiten un RulePermit ligado al mismo request/hash,
  regla y decisión `PERMIT`; el consumo debe repetir el hash del permiso. La prueba
  MariaDB rechaza permisos sobre DENY, hashes/reglas distintos y consumos con otro hash.
- Hasta que el paso 6 escriba decisión y RulePermit atómicamente, un trigger impide a la
  cuenta app insertar decisiones `PERMIT`, y `get()` falla cerrado si encuentra una.
  El paso 6 debe retirar ese trigger en la misma migración que activa su adapter.

## Verificación

- Integración de este paso: `1 passed` contra MariaDB `12.3.3`; piso
  `authority-rule-storage/mariadb` comprobado con `piso.py verificar`.
- Suites Faro/Block 4 seleccionadas: `228 passed`.
- Los mutantes temporales de transacción ausente, hash no comparado y commit posterior
  al retorno fueron detectados por aserciones distintas y luego retirados.
- `policy.yml` y `ci/pisos.json` se parsearon correctamente. Ruff no está instalado en
  el entorno local; se revisó sintaxis Python con `ast.parse`.
- La prueba de privilegios inicia con `ALL PRIVILEGES ... WITH GRANT OPTION` y verifica
  `SHOW GRANTS` tras provisionar. La prueba de caída comprueba que la factory fue
  invocada; ocho escrituras UUID7 concurrentes completan sin deadlock.

## Pendientes

- Esperar que #370 fije r6 y que #371 se reapile; luego repetir el rebase de esta rama,
  el piso, las suites y `git merge-tree` frente a #370/#371/#372. Al momento del registro,
  los tips observados son #370 `85cee9ca`, #371 `31eb14bf` y #372 `57547cfe`.
- Implementar `RulePermit`, su inserción atómica con `PERMIT`, el consumo single-use y
  las carreras en el paso 6.
- El checkpoint externo y la reconciliación post-crash de auditoría durable de §11
  requieren su contrato posterior; este paso implementa el audit head transaccional de
  MariaDB.
- El audit head se valida por forma y avance transaccional, pero no se reconcilia con
  un checkpoint externo ni se verifica criptográficamente una cadena completa ante un
  administrador hostil. El adapter no declara terminado ese contrato anti-rollback.

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
