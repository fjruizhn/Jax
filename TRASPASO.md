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

El piso `identity-foundation-shadow/policy` quedó actualizado a 795 con la
historia de la medición y de los cuatro ataques en `docs/ci/pisos.md`.

## Decisión y estado

Fernando decidió alinear `FormaLimites` al schema vigente: enum cerrado con `NINGUNA`,
`CANTIDAD`, `MONTO`; una acción `OBLIGATING` requiere exactamente cantidad o monto, y
`unidad`/`moneda` deben coincidir con la forma. No se amplía schema ni evaluador.

En `feat/faro-f1.1-providers-r2`, el rechazo Tier 3 anterior incluía prioridad de writer,
catálogo cerrado, tipos escalares exactos y esta incoherencia. Se eliminó `CANTIDAD_Y_MONTO`,
se validó coherencia y la regresión contra subclases hostiles; el piso Identity Shadow actual
es 795. El commit local `bcadb31d` recoge esos cambios. Evidencia local Python 3.14.4:
providers 131 passed; lista exacta Identity Shadow 795 passed, cero skipped; py_compile,
JSON y diff-check limpios. El CI remoto todavía no cubre estos cambios.

La auditoría de arquitectura determinó que el contrato normativo de la spec requiere leases
compartidos en orden `pin → clasificación → STOP`, mantenidos hasta persistir la decisión.
El cambio local sin commit añade `leases_de_emision(...)`, devuelve las tres vistas dentro de
un contexto y los libera en orden inverso; una regresión demuestra que los writers exclusivos
quedan bloqueados hasta salir. Provider suite: 131 passed; diff-check y py_compile limpios.
Este PR aún no incluye evaluador ni store; al reconstruir #371, `store.record()` debe quedar
dentro del contexto. El orden y retención de leases no se puede afirmar end-to-end hasta ese
cambio.

## Dependencias y próximo paso

PR #373 debe actualizar su base a `master` después de mergear #379, incluir los cambios locales,
remedir el piso sobre el árbol final y recibir nueva auditoría Tier 3 del SHA exacto antes de
integrar. #371 consume #373 y además necesita la corrección de cola de #376; #375 y #378 están
apilados sobre #371. No desplegar. Pendiente independiente: obtener la decisión de Fernando
sobre la semántica `OVERLAY_ISSUED` de #381 antes de cerrar ese PR.

```sh
cd /home/fruiz/wt/jax-faro-f11-providers && git status --short --branch
```
