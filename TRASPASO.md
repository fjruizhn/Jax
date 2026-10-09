# Traspaso Codex · JAX Faro #373 · 2026-10-08

## Corrección Tier 3 · 2026-10-09

El auditor encontró que `TrustedPolicyPin` aceptaba subclases de `str` y
conservaba el objeto. Una implementación hostil de `__eq__` podía falsear la
comparación de procedencia contra `PinActivo`. `snapshot.py` ahora exige
`type(valor) is str` en los cuatro campos del pin (`repositorio`, `commit`,
`policy_tree_oid`, `procedencia`) antes de guardarlos. La regresión construye
una subclase cuyo `__eq__` siempre devuelve `True` y confirma que cada campo
se rechaza con `RuleSnapshotError`.

Evidencia local Python 3.14.4:

- regresión focal: 4 passed;
- `tests/policy/test_rule_authority_snapshot.py`: 49 passed;
- lista exacta `identity-foundation-shadow`: 795 passed, 0 skipped;
- `git diff --check`: limpio.

El verificador del piso falla cerrado porque `ci/pisos.json` aún exige 791.
La lista medida es 795; quien posea `ci/pisos.json` debe subir su patrón e
historia con esta medición. Este worktree no modifica ese archivo.

## Decisión y estado

Fernando decidió alinear `FormaLimites` al schema vigente: enum cerrado con `NINGUNA`,
`CANTIDAD`, `MONTO`; una acción `OBLIGATING` requiere exactamente cantidad o monto, y
`unidad`/`moneda` deben coincidir con la forma. No se amplía schema ni evaluador.

En `feat/faro-f1.1-providers-r2`, el rechazo Tier 3 anterior incluía prioridad de writer,
catálogo cerrado, tipos escalares exactos y esta incoherencia. El cambio local actual elimina
`CANTIDAD_Y_MONTO`, valida coherencia, ajusta las regresiones y el piso Identity Shadow a 791.
Evidencia local Python 3.14.4: providers 130 passed; lista exacta Identity Shadow 791 passed,
cero skipped; py_compile, JSON y diff-check limpios. El CI remoto todavía no cubre estos cambios.

## Dependencias y próximo paso

PR #373 debe actualizar su base a `master` después de mergear #379, incluir los cambios locales,
remedir el piso sobre el árbol final y recibir nueva auditoría Tier 3 del SHA exacto antes de
integrar. #371 consume #373 y además necesita la corrección de cola de #376; #375 y #378 están
apilados sobre #371. No desplegar.

```sh
cd /home/fruiz/wt/jax-faro-f11-providers && git status --short --branch
```
