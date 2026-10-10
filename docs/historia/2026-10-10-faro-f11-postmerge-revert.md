# Revert requerido tras la integración de Faro F1.1 Storage

- **Fecha:** 2026-10-10.
- **Tipo:** HISTORIA.
- **Fuente:** GitHub Jax #385 y workflow `policy`, run `38022192036`; verificados por Codex en Hall9000 el 2026-10-10.
- **Integración afectada:** PR #385, head `111ec66505001087ee33618c46db46aa11713eba`, merge `9b2744c33c37fdfeec87bac03ddd00c9b76e8fb5`, padre 1 `da99e6e22ccd1689950ed5ef13278fbd14ffb03c`.

## Qué ocurrió

El `post-merge-guard` devolvió `REVERT_REQUIRED` por el workflow `policy` del merge. El único job fallido fue `tests-puros`, en el paso `Faro F1.1 Rule Authority MariaDB aislada -- piso exacto`: 11 pruebas pasaron y una perdió conexión al ejecutar `SELECT VERSION()` durante la preparación de la prueba. La corrida `pull_request` del mismo head había pasado la suite MariaDB completa (12/12).

La evidencia disponible no demuestra la causa raíz. El error ocurrió durante el setup de la conexión, antes del cuerpo de esa prueba; no hay evidencia suficiente para afirmar si fue una condición de disponibilidad de MariaDB u otra causa. No se modificó ni se desplegó una base de datos persistente.

## Acción y estado

El runbook `docs/runbooks/integracion-github-free.md` exige un PR de revert ante `REVERT_REQUIRED`. Se preparó en un worktree aislado el revert de primer padre de `9b2744c33c37fdfeec87bac03ddd00c9b76e8fb5`; el delta de reversión coincide con el árbol del padre 1. La reversión incluye `.github/workflows/policy.yml`, archivo reservado: Fernando integra el revert. Al redactar esta historia, el revert sigue local, sin publicar ni integrar, y requiere auditoría independiente, preflight y verificación posterior.

## Lecciones y pendientes

- La corrida pull request verde no sustituye al guard post merge; el fallo del run `push` mantiene abierto el procedimiento de reversión.
- Preservar logs de MariaDB y cubrir la disponibilidad real del servidor durante setup antes de atribuir la causa o proponer una corrección. Ese diagnóstico queda para un cambio posterior, separado del revert.
- Pendiente: auditoría y CI exacta del PR de revert; integración por Fernando; `post-merge-guard` sobre el tip resultante. Después, investigar la causa del fallo y reintroducir el almacenamiento con una nueva auditoría.

**Responsabilidad:** Codex preparó el revert siguiendo el runbook. La obligación de revertir deriva del guard; la causa técnica continúa sin determinar.
