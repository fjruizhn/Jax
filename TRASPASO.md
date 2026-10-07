# Traspaso Faro F1.1 paso 5

## Estado (2026-10-07)

- Rama: `feat/faro-f1.1-storage-mariadb`, creada desde `origin/feat/faro-f1.1-evaluador` en `31eb14bf`.
- Worktree: `/home/fruiz/wt/jax-faro-f11-paso5`.
- Commit de implementación Codex publicado: HEAD de la rama remota.
- Baseline: `tests/policy/test_faro_rule_authority_models.py`: 30 passed.
- Implementados `policy/rule_authority/migrations/001_rule_authority_kernel.sql`,
  `storage.py`, `provisioning.py`, el error tipado, el test efímero y el paso de CI.
- La migración crea `rule_decisions`, `rule_permits`, `rule_permit_consumptions` y
  `rule_authority_audit_head`; los tres registros append-only tienen triggers contra
  `UPDATE`/`DELETE`. La cuenta versionada solo inserta/lee esos registros y actualiza
  el head.
- MariaDB 12.3.3 efímera: 1 passed en 5.64 s; piso
  `authority-rule-storage/mariadb` verificado. Suite Faro/Block 4 seleccionada:
  228 passed.
- Mutantes comprobados: quitar `begin()` deja una fila tras fallo parcial; omitir la
  comparación de hash hace que `get()` devuelva una fila ajena; omitir el commit hace
  que otra conexión no vea la decisión al retornar. Los tres fallan en aserciones.
- Correcciones de la segunda pasada: provisioning revoca privilegios previos antes de
  aplicar GRANTs; el test parte de `ALL PRIVILEGES ... WITH GRANT OPTION` y verifica
  `SHOW GRANTS`. El adapter conserva `RuleDecision.catalogo` en roundtrip y adquiere
  el audit head antes de revisar IDs ausentes, serializando escritores sin gap-lock
  invertido. El test de caída confirma una llamada real a la factory y la prueba
  concurrente escribe ocho IDs UUID7 distintos.
- OIDs de Git en `rule_permits` aceptan exactamente 40 o 64 dígitos hexadecimales,
  como los pins SHA-1/SHA-256 del snapshot; la prueba MariaDB hace una inserción
  transaccional con SHA-1, inserta SHA-256 y rechaza longitudes inválidas y un newline
  terminal.
- El catálogo guardado con una decisión debe ser un MappingProxyType de tuplas de texto
  y coincidir con el catálogo oficial de la solicitud; se rechazan mapas ajenos y la
  forma malformada que convertiría un string en caracteres.
- `request_catalog_hash` queda persistido y ligado a la fila canónica para que `get()`
  devuelva `None` si el mismo ID/hash llega con un catálogo distinto, aun cuando
  `RuleDecision.catalogo` sea `None`.
- Las claves foráneas de `rule_permits` exigen decisión con mismo request/hash/regla y
  `status=PERMIT`; las de consumo exigen el mismo permit/hash. La prueba verifica que la
  DB rechaza permiso sobre DENY, hash/regla ajenos y consumo con hash ajeno.
- Un trigger bloquea que la cuenta de aplicación persista `PERMIT` antes del paso 6;
  `get()` también falla cerrado si encuentra una fila PERMIT. El paso 6 tendrá que
  retirar el trigger dentro de su migración junto con el insert atómico.
- Alcance conservador: `DENY` y `MISSING_RULE` se guardan; `PERMIT` falla cerrado
  hasta que el paso 6 agregue `RulePermit` para insertarlo atómicamente.
- #370 r6 y el reapilado de #371 pueden mover la base; todavía falta reapilar esta
  rama y repetir pisos y merge-tree contra los tips fijados.
- Tips GitHub al último chequeo: #370 `85cee9ca`, #371 `31eb14bf`, #372 `57547cfe`.

## Siguiente comando después del reapilado de #371

`git fetch origin && git rebase --onto origin/feat/faro-f1.1-evaluador 31eb14bf feat/faro-f1.1-storage-mariadb`

Después del rebase, repetir la integración MariaDB con
`JAX_RULE_AUTHORITY_DOCKER_CMD='sudo -n docker'`, el piso, las suites Faro/Block 4,
EXPLAIN y merge-tree frente a #370/#371/#372; actualizar este archivo en cada commit.
El paso 6 de `RulePermit`/consumo queda fuera de esta rama.
