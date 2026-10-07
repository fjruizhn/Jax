# Traspaso · Faro F1.1 #368/#369/#371

## Objetivo

Cerrar los hallazgos del veredicto de escalón 3 para #368, #369 y #371. El veredicto de
entrada está en `~/encargos-codex/jax-faro-f11-veredicto-368-369-371-r2.md`.

## Hecho en esta rama (#368)

- El modelo compara ambos sellos por identidad; el writer exige el sello de snapshot para
  `RATIFICATION_GRANTED` y sigue rechazando el sello de storage.
- `previous_event_hash` se lee y contrasta junto con las demás columnas denormalizadas.
- `canonical_intent` es NOT NULL en la migración inicial y en la migración de upgrade.
- Se agregó `policy.authority_ledger.provisioning` para crear la cuenta de aplicación y sus
  grants limitados, usando credenciales del entorno.
- Las pruebas K3, K8, D2 y D3 fallan al retirar el control correspondiente.
- MariaDB 12.3.3 efímera pasó migración, provisión, append/replay y triggers con un principal
  que tiene SELECT/UPDATE/DELETE.
- Suite `tests/policy/test_authority_ledger*.py`: 41 passed.
- El codec corrió 19 pruebas; el piso `authority-ledger-codec/codec` quedó en 19.

## Falta

- Repetir las pruebas de #368 tras completar #369 y confirmar ambos pisos en el SHA final.
- Incorporar esta base corregida a #369, luego cerrar sus pruebas/piso.
- #371 espera el SHA final de #370 antes del rebase sobre `origin/feat/faro-f1.1-schema-snapshot`.
- Re-medición final de los pisos sobre la base real apilada, merge-tree y reporte de entrega.

## Decisiones

- Se siguen los cierres obligatorios del veredicto r2 y la decisión indicada por Fernando para
  consumir el catálogo de topes de #370. No se escribe ni publica un veredicto de auditoría.
- No se usan llaves de Fernando ni bases persistentes; las pruebas MariaDB usan contenedor
  desechable, `--network none` y socket Unix.

## Siguiente comando

Desde el worktree #368, ejecutar:

```sh
JAX_AUTHORITY_LEDGER_DOCKER_CMD='sudo -n docker' PYTHONPATH=. python3 -B -m pytest -q tests/policy/test_authority_ledger*.py
```
