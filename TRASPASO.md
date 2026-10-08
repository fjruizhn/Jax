# Traspaso · fix/authority-ledger-sello-y-append

## Continuación 2026-10-08 · reapilado sobre master actual

Fernando pidió integrar los pendientes. El #370 ya está en master (`dee9426...`), por lo
que el SHA antiguo `d082cc89` dejó de ser integrable aunque tuvo APROBADO Tier 3 y CI
verde. El worktree `jax-377-fix`, rama `codex/integration-377`, está en merge local con
`origin/master` para actualizar la base. Se conservan los pisos de ledger y se añade el
piso de Identity Shadow de master (657); el Faro hereda 729. Conflictos resueltos en
`ci/pisos.json`; se mantuvo el handoff raíz porque esta tarea sigue abierta.

Siguiente: validar JSON/YAML y diff, medir pisos aplicables, completar el merge commit con
trailer de Codex, empujar la rama al PR #377, pedir auditoría Tier 3 y esperar CI sobre el
SHA nuevo. No reutilizar el APROBADO ni los checks de `d082cc89`.

## Objetivo

Cerrar las dos fallas serias del auditor de escalón 3 (ronda 3 de #368/#369),
presentes en master: (1) sello de ratificación falsificable con
`dataclasses.replace`/dict mutable, (2) `append` que firma eventos sin validar
contra el estado, envenenando un ledger append-only para siempre. Encargo:
`~/encargos-codex/encargo-glm-g2-ledger-sello-y-append.md`. Base: master
83f8f56b (ya trae #368/#369/#374).

## Hecho en esta rama

- `models.py`: sellos `_ratification_snapshot_seal` y
  `_rule_ratification_snapshot_seal` pasan a `init=False`; solo los estampan
  las fábricas `_from_validated_snapshot` (corpus) y
  el helper de prueba `tests/policy/_sellos_de_prueba.py` (Faro: en producción aún no hay vía que estampe el sello individual). `__post_init__` ya no
  puede ver sellos (siempre None en construcción): la frontera que firma
  (`append_authority_event`) es la que exige el sello correcto, como ya hacía.
- `static_policy_view_projection` se congela profundo al construir
  (`MappingProxyType`/tuplas hasta el fondo). `canonical_projection()` emite
  estructuras planas vía `plain()` (que ahora acepta `Mapping`): los bytes
  firmados NO cambian — vector dorado clavado a master lo prueba.
- `serialization.py` estampa los sellos de STORAGE tras construir (rehidratar
  no autoriza re-anexar; `append` sigue rechazándolos).
- `service.py`: antes de firmar, compara (no recalcula) la coherencia
  hash↔proyección congelada; antes de `store.append`, replay completo
  `verify_authority_ledger(events + [nuevo])` — si el estado rechaza el
  evento, no se escribe nada (fallo 2, error cerrado).
- Tests: `test_authority_ledger_seal_attacks.py` (4 ataques),
  `test_authority_ledger_golden_vectors.py` (6 tipos + genesis),
  `test_authority_ledger_append_validation_mariadb.py` (cero filas nuevas en
  MariaDB 12.3.3 efímera `--network none`). E4 de #369 y hermanas
  (`rule_ratifications`) ahora esperan el rechazo en append;
  `audit_001` y los contratos de construcción ajustados al sello `init=False`.

## Mutantes muertos (aplicar mutante → prueba FALLA → revertir)

| Mutante | Prueba que lo mata | Aserción |
|---|---|---|
| quitar el recálculo hash↔proyección | seal_attacks::test_hash_drift_after_sealing... | append lanza `no coincide con la proyección`; store queda () |
| sellos de vuelta `init=True` | seal_attacks::test_replace_attack... (+ variante Faro) | replace transporta el sello → append firma → pytest.raises explota |
| dict mutable (sin congelar) | seal_attacks::test_projection_is_frozen_deep... | la asignación NO lanza TypeError |
| congelado solo superficial | idem | mutar dict anidado NO lanza TypeError |
| quitar el replay previo | rule_ratifications E4-family (2) + append_validation_mariadb | el veneno se escribe / hay fila nueva |

## Pisos

- `authority-ledger-mariadb/integration`: 4 → **5** (incluye archivo nuevo).
- Nuevos: `authority-ledger-seal/attacks` **4**, `authority-ledger-golden/vectors` **7**.
- `tests-puros/out` SIN cambios: A/B local master↔rama idéntico
  (3928 passed, 3 skipped, 3 xfailed en hall9000; en el runner 3888/45 según
  su entorno — divergencia ambiental documentada en el propio piso).
- Verificados con `piso.py verificar` cada paso, + wireados (7), bash válido
  (3) y pisos fuera del workflow (227).

## Pendiente

- Auditoría de Hyde (escalón 3) y merge de Fernando.
- Observación (fuera de alcance, reportada en la entrega):
  `ValidatedCandidateCorpus._loader_seal` (Block 3) tiene el mismo patrón
  `init=True` falsificable por `replace`.

- Intents con proyección congelada (`MappingProxyType`) ya no son `copy.deepcopy`/`pickle`-ables; si hace falta copiarlos, usar `canonical_projection()` o `plain()`.
