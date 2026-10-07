# Traspaso Faro F1.1 paso 5

## Estado (2026-10-07)

- Rama: `feat/faro-f1.1-storage-mariadb`, creada desde `origin/feat/faro-f1.1-evaluador` en `31eb14bf`.
- Worktree: `/home/fruiz/wt/jax-faro-f11-paso5`.
- Baseline: `tests/policy/test_faro_rule_authority_models.py`: 30 passed.
- Implementados `policy/rule_authority/migrations/001_rule_authority_kernel.sql`,
  `storage.py`, `provisioning.py`, el error tipado, el test efímero y el paso de CI.
- La migración crea `rule_decisions`, `rule_permits`, `rule_permit_consumptions` y
  `rule_authority_audit_head`; los tres registros append-only tienen triggers contra
  `UPDATE`/`DELETE`. La cuenta versionada solo inserta/lee esos registros y actualiza
  el head.
- MariaDB 12.3.3 efímera: 1 passed; piso `authority-rule-storage/mariadb` verificado.
  Suite Faro/Block 4 seleccionada: 228 passed.
- Mutantes comprobados: quitar `begin()` deja una fila tras fallo parcial; omitir la
  comparación de hash hace que `get()` devuelva una fila ajena; omitir el commit hace
  que otra conexión no vea la decisión al retornar. Los tres fallan en aserciones.
- Alcance conservador: `DENY` y `MISSING_RULE` se guardan; `PERMIT` falla cerrado
  hasta que el paso 6 agregue `RulePermit` para insertarlo atómicamente.
- #370 r6 y el reapilado de #371 pueden mover la base; todavía falta reapilar esta
  rama y repetir pisos y merge-tree contra los tips fijados.

## Siguiente comando después del reapilado de #371

`git fetch origin && git rebase --onto origin/feat/faro-f1.1-evaluador 31eb14bf feat/faro-f1.1-storage-mariadb`

Después del rebase, repetir la integración MariaDB con
`JAX_RULE_AUTHORITY_DOCKER_CMD='sudo -n docker'`, el piso, las suites Faro/Block 4,
EXPLAIN y merge-tree frente a #370/#371/#372; actualizar este archivo en cada commit.
El paso 6 de `RulePermit`/consumo queda fuera de esta rama.
