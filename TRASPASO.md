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
- MariaDB 12.3.3 efímera: 7 passed en 34.61 s (una base por prueba); piso
  `authority-rule-storage/mariadb` = `^7 passed in ` verificado con `piso.py verificar`.
  El monolito de 1 prueba se dividió en 7 (rollback, rollback fallido, idempotencia/binding,
  triggers append-only, restricciones/EXPLAIN, provisioning x2).
- Ronda 2 de #375: el `rollback()` fallido ya no se traga (`_LOG.exception` + marca
  `# fail-soft:`); los triggers de DELETE se ejercitan sobre filas sin hijos (DENY y permiso
  sin consumo) afirmando código 1644 y mensaje; `get()` con fila existente y otro catálogo
  devuelve None; provisioning aplica REVOKE/GRANT también a la cuenta homónima `@'%'` si
  existe y tiene `main()` de operador (`JAX_RULE_AUTHORITY_ADMIN_*`/`_APP_*`); se quitó el
  `pip install pymysql` suelto del paso de CI (el job ya lo instala antes).
- Nombre de tabla: el diseño §11 dice `rule_authority_decisions`; el encargo del paso 5 y la
  migración usan `rule_decisions`. No se renombró (decisión de Hyde); si se decide el nombre
  del diseño, cambia la migración, `storage.py`, el test y los GRANTs de provisioning en un
  mismo commit.
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
- Ronda 3 de #375 (2026-10-07): fusionado `origin/feat/faro-f1.1-evaluador` @ 002fc9ba (trae #370 final). `RuleDecision` ya no lleva `catalogo` y el catálogo del request es `CatalogoTopes` sellado: `storage.py` proyecta clases + `oid_pin`. MINOR (a): el `rollback()` de la rama `AuthorityStateError`/`RuleAuthorityStorageError` se protege (log + `RuleAuthorityStorageError`). MINOR (b): provisioning aplica REVOKE/GRANT a toda entrada homónima de `mysql.user`. Piso `authority-rule-storage/mariadb` = 9.

# Traspaso Faro F1.1 · #371 (paso 4) · ronda 3

## Estado (2026-10-07)

- Rama `feat/faro-f1.1-evaluador`, worktree `/home/fruiz/wt/jax-371-r3`. Base: #370 `feat/faro-f1.1-schema-snapshot` @ 28f1eac7 (antes 05183cb7) (fusionada con merge commit; conflictos de TRASPASO.md, ci/pisos.json y docs/ci/pisos.md resueltos conservando todos los pisos).
- Hecho en la ronda 3 (cierra el veredicto de ronda 2): catálogo por tipo exacto `CatalogoTopes`; `RuleLimits` = envoltorio de `LimitesObligatorios`/`Tope` de #370, solo vía `limites_de(regla, catalogo)`; `RuleEvaluation` exige el mismo catálogo y el `request_hash` lleva el OID; `type(x) is str` en unidades/moneda; enteros de `arguments` con rango; `store.record` exige `RuleDecision`; sin campo huérfano ni import duplicado; `pymysql==1.2.0` con hashes (`requirements-faro-mariadb-ci.txt`); historia corregida.
- Pisos medidos en hall9000 (Python 3.14.4): modelos 140, identity 659 (re-medido tras fusionar #370 final 28f1eac7), codec 19, ratificaciones 12, MariaDB 4. `tests/policy` completo con MariaDB efímera: 857 passed, 15 skipped.
- Mutantes: 36 mutaciones de `models.py`/`store.py` (28 de r3 + 8 de la corrección de la auditoría), todas muertas (tabla en `~/encargos-codex/entrega-faro-f11-371-r3.md`).

## Siguiente comando después del reapilado de #371

### Paso 5 (#375, MariaDB)

`git fetch origin && git rebase --onto origin/feat/faro-f1.1-evaluador 31eb14bf feat/faro-f1.1-storage-mariadb`

Después del rebase, repetir la integración MariaDB con
`JAX_RULE_AUTHORITY_DOCKER_CMD='sudo -n docker'`, el piso, las suites Faro/Block 4,
EXPLAIN y merge-tree frente a #370/#371/#372; actualizar este archivo en cada commit.
El paso 6 de `RulePermit`/consumo queda fuera de esta rama.

### Paso 4 (#371, evaluador)

- Auditoría de escalón 3 sobre el SHA final (otra invocación). Nada se declara aprobado aquí.
- Si #370 cambia de SHA, repetir la fusión y re-medir identity (el piso depende de la lista del paso).
- No hay evaluador en #371 (solo modelos y store): vigencia, STOP, snapshot y grants son pasos posteriores.

## Siguiente comando

```sh
JAX_AUTHORITY_LEDGER_DOCKER_CMD='sudo -n docker' PYTHONPATH=. python3 -B -m pytest -q tests/policy -p no:cacheprovider
```

- #375 r2 (2026-10-07): fusionado #371 `2b6c9cb3` (trae #370 final 28f1eac7, identity 659). Cerrados 2 MINOR de prueba: el comentario «M9» de `test_..._bound_to_request_and_catalog` ahora dice que el None sale de `row[0] != request.request_hash` y el chequeo de catálogo es defensa en profundidad; nueva prueba `_catalog_projection` rechaza lo que no es `CatalogoTopes` (3 casos; mutante sin el chequeo muere). Piso `authority-rule-storage/mariadb` = `^12 passed in ` (medido con MariaDB 12.3.3 efímera).
