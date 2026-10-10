# TRASPASO — refresco de historial Faro

## Objetivo
Actualizar el registro histórico de coordinación del grant de retiro F1.1 con los merges #396/#395 y la evidencia revisada el 2026-10-10.

## Hecho
- Branch `codex/faro-grant-handoff-refresh`, base `master@ec01b2191b1a7c1e05854a345a610d72f8b643ea`.
- Primera auditoría Tier 3 del SHA `cf3c8b1140e2ff0156c4f74341485356dde3aef9`: RECHAZADO (1 BLOCK, 1 MAJOR). BLOCK: el archivo de traspaso seguía incluido; MAJOR: la búsqueda Restic no tenía snapshots/comando/salida reproducibles.
- El registro histórico ahora rebaja explícitamente la búsqueda Restic anterior a reporte no verificado y documenta la comprobación en el host con hora UTC, comando, fuente y mirror JAX `master@7024529e`.
- El test de presencia de `/etc/jax/authority/trusted-root.json` retornó 1; la búsqueda local de los tres artefactos de autoridad no encontró archivos. `jaxctl` no está en PATH, así que el ledger sigue sin verificación.
- Hallé que `restic find --path` con cero snapshots para las rutas no prueba ausencia en otros snapshots. Detuve la búsqueda global antes de completarla y no la usaré como evidencia; el historial ahora marca el reporte Restic previo como no verificado.
- Estado de producción: mirror de JAX `master@7024529e1ede4b3a3de92eefe4f60e3d2f567f01`; root/receipt/checkpoint no están presentes en las rutas inspeccionadas; ledger no verificado por faltar `jaxctl`.
- Estado remoto sin cambio: #390 head `3e74c614`, #387 head `b5f70fef`, #388 head `3eac3fff`; todos siguen sin integración.
- No se modificaron código, reglas, PRs ajenos ni producción.

## Falta
- Revisar el diff, guardar los cambios documentales, eliminar `TRASPASO.md` antes de la auditoría final, publicar el SHA exacto, obtener auditoría Tier 3, CI, verify-integration, merge y post-merge-guard.
- Después rebasar #390 sobre el tip oficial nuevo y recalcular el comparador con el árbol final.

## Decisiones
- Mantener #388 en su worktree de owner.
- No generar ni sustituir la raíz confiable ausente; producción queda bloqueada hasta que exista fuente autorizada y ledger verificado.

## Siguiente comando
`git -C /home/fruiz/worktrees/jax-faro-grant-handoff-refresh diff --check && git -C /home/fruiz/worktrees/jax-faro-grant-handoff-refresh diff -- docs/historia/2026-10-10-faro-floor-retirement-grant.md`
