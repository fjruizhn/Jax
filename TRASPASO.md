# Traspaso — soporte genérico para retiros exactos de pisos

**Fecha:** 2026-10-10 (America/Tegucigalpa)
**Rama/worktree:** `codex/faro-piso-retirement` / `/home/fruiz/worktrees/jax-faro-piso-retirement`
**Base inicial:** `bf6b05809e1c3505efd7447b33685796ca42e3b6`
**Estado:** cambios locales sin commit; no push, PR ni merge.

## Alcance implementado

- `policy.yml` obtiene historia completa, falla si permanece shallow y ejecuta el comparador extraído de la ref oficial de base en `$RUNNER_TEMP`; valida la ref merge del PR.
- `floor-retirements.json` usa esquema versionado cerrado y permanece vacío, sin grant efectivo.
- El comparador verifica registro solo en base; repo/PR/rama/SHA base; clave y definición canónica completa; digest; merge de introducción y padres; transición de base como merge dedicado cuyo parent1 es el estado previo, parent2 añade únicamente el registro, y el diff solo contiene ese registro. El hash esperado del diff usa rutas, modos y blob OIDs completos.
- Historia shallow, `bad object`, padres incompletos, grants inválidos/duplicados, claves múltiples, retiro parcial y diff diferente fallan cerrados. El comportamiento ordinario sin grant se conserva.
- Se añadieron pruebas de esquema/hash, historia incompleta, salida raw de diff, extracción del checker de base y registro vacío.
- La última prueba de integración valida el merge dedicado de base, grant solo vigente en la base exacta y rechazo ante ruta extra; el total final del comando exacto es `232 passed`. Se re-midió el piso asociado y quedó en `^232 passed`.
- Se actualizó `docs/ci/pisos.md` para documentar el contrato y el estado sin autorización.

## Verificación

```text
python3 -m pytest -q policy/tests/test_pisos_fuera_del_workflow.py policy/tests/test_pisos_migracion_desde_master.py policy/tests/test_comparar_pisos.py
232 passed in 1.57s (verificación independiente del coordinador)
```

## Pendiente de auditoría / coordinación

La rama #388 puede tocar `ci/pisos.json`; se actualizó únicamente `archivos-de-test-en-ci/pisos` de `^227` a `^232` según la medición exacta. El solape con #388 queda anotado: coordinar antes de publicar o integrar. No agregar el grant real aquí. #387 y #388 no se modificaron.

Auditoría Tier 3 independiente aún requerida sobre el SHA final; esta implementación Tier 2 no se autoaudita. Trabajo aún local: ninguna rama ajena fue modificada y no se publicó nada.

## VISTO PERO NO TOCADO

- #388: comparte `ci/pisos.json`; el único cambio local es la re-medición exacta del piso de estos tests. Revisar y reconciliar con su owner antes de publicar o integrar.
- #387 y #388: no se rebasaron ni modificaron; el grant está vacío y este trabajo no autoriza el retiro.
- Integración/publicación: requiere que el owner coordinador solicite la auditoría independiente y siga su runbook de PR.
