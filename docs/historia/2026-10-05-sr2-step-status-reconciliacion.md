# Reconciliación de SR2 STEP_STATUS con master

**Fecha:** 2026-10-05
**Tipo:** HISTORIA
**Fuente:** merge local de `origin/master` `d136cce5d88b3420a5066b40e7786c158118f29e`
en la rama `phase2/sr2-step-status-20261003` de JAX PR #341.

## Qué se hizo

Se hizo un merge regular, sin rebase ni reset, de `origin/master` en la rama
de SR2. Git lo resolvió automáticamente y se confirmó en
`92e8e3a12fbb9818f261db8b12daab007e70d479`.

## Por qué

La rama SR2 debía incorporar los cambios recientes de `master` antes de su
auditoría. El cambio concurrente de `master` agregó la cobertura contra
volúmenes huérfanos de MariaDB y elevó el piso de `tests-puros`; SR2 agrega su
propio piso exacto para la semántica de `STEP_STATUS`.

## Resultado verificado

- El paso SR2 sigue ejecutando
  `PYTHONPATH=.:las_manos:jax/core python -m pytest
  tests/test_f2e_step_status.py -q` y lo verifica contra
  `governance/f2e-sr2`.
- El piso SR2 permanece en `^26 passed` y la ejecución local dio `26 passed in
  0.15s`.
- El piso de `tests-puros/out` conserva el aumento de `master` a `^3473 passed,
  45 skipped`; la prueba nueva figura en sus dos listas del workflow.
- Los controles de cableado y pisos relevantes dieron `55 passed`:
  `policy/tests/test_archivos_de_test_wireados_en_ci.py`,
  `policy/tests/test_pisos_fuera_del_workflow.py` y
  `policy/tests/test_workflows_bash_valido.py`.

## Decisión y límites

**Quién decidió:** Fernando Ruiz fijó el orden y los límites de SR2 en la
recuperación del 2026-10-05; Codex decidió el merge regular y la verificación
técnica.

**Pendiente:** publicar el nuevo JAX HEAD, obtener CI verde para ese SHA,
actualizar en Platform #189 el pin exacto, reconciliar su master, ejecutar
el par completo y auditar ambos SHAs finales antes de cualquier integración.

**Alternativas descartadas:** no se reutilizó el veredicto del par anterior
porque sus SHAs ya no serían los finales; tampoco se mezcló el worktree
alterno de Platform con cambios locales sin publicar. Se inspeccionaron las
acciones de GitHub: pasar sus etiquetas a SHA exactos es otro cambio de
cadena de suministro y no forma parte de esta reconciliación.
