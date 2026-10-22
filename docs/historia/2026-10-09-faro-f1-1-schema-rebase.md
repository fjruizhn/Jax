# JAX · Faro F1.1 · reapilado de #378 · 2026-10-09

**Tipo:** HISTORIA · **Responsable:** Codex, hall9000 · **Coordinación:** Fernando.

## Estado y causa

Al retomar la cadena, JAX #378 estaba abierto con head `4c3064a274179da371ab6893bf2d462e5fdc8a7b`, base `master@8c850bcec5cd0233db5487343210c0ba96689f97`, 62 checks verdes y estado `CONFLICTING`. `master` había avanzado desde entonces. #381 ya estaba integrado como `ec89ef1e`; el registro de auditorías de #371 (#384) entró en `da99e6e2`.

Fernando aclaró que GLM/ZCode había terminado su revisión F1.1 y que no se debía esperar otra respuesta. Las marcas ZCode de análisis y reapilado ya estaban `[x]` en `PENDIENTES.md`. La recomendación de secuencia se cotejó contra los PRs vivos: #381 ya estaba integrado; #378 y #375 seguían abiertos.

## Trabajo

Se creó el worktree propio `/home/fruiz/jax/.worktrees/codex-faro-378-rebase-20261009`, rama `codex/faro-378-rebase-20261009`, desde el head remoto de #378. La rama y el worktree ZCode se conservaron intactos. Se reaplicaron los seis commits del delta de #378 sobre `master@da99e6e22ccd1689950ed5ef13278fbd14ffb03c`.

Los conflictos estuvieron en el comentario del workflow `policy.yml`, los pisos y su documentación. Se conservó el baseline identity de master con #381, medido en 1003, y se sumaron 37 pruebas de #378: 34 de unidades/monedas del catálogo, una de catálogo forjado por subclase y dos de paridad JSON Schema/Python. El comportamiento de schema y las pruebas del PR se reaplicaron sin alterar su semántica.

## Verificación local

- Python 3.14.4, lista exacta del paso `Identity Foundation Shadow -- piso exacto de tests CORRIDOS` extraída de `.github/workflows/policy.yml`: **1040 passed, 0 skipped**.
- `.github/ci/piso.py verificar identity-foundation-shadow/policy`: rc=0, dentro del mismo paso.
- `tests/policy/test_rule_authority_schema.py`: **167 passed**.
- `ci/pisos.json` valida como JSON; `git diff --check` limpio.

## Decisiones y alternativas

- Fernando decidió que el reporte de GLM/ZCode estaba completo y que el trabajo podía seguir sin esperar respuesta adicional.
- Se usó una rama Codex nueva para no reescribir la rama de ZCode ni cambiar el head del PR #378 de forma concurrente.
- No se integró código ni se declaró listo el PR: el nuevo SHA requiere auditoría Tier 3 exacta y CI completa en un PR propio. Después se puede cerrar el PR antiguo como reemplazado. Antes de cualquier integración, esta sesión debe volver a consultar `bin/ventana estado`, ejecutar `verify-integration` para el SHA auditado y, después del merge, `post-merge-guard`.
