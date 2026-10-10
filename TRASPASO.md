# TRASPASO — refresco de historial Faro

## Objetivo
Actualizar el registro histórico de coordinación del grant de retiro F1.1 con los merges #396/#395 y la evidencia revisada el 2026-10-10.

## Hecho
- Branch `codex/faro-grant-handoff-refresh`, base `master@ec01b2191b1a7c1e05854a345a610d72f8b643ea`.
- Se actualizó `docs/historia/2026-10-10-faro-floor-retirement-grant.md` con estados fechados y fuente de la búsqueda Restic.
- No se modificaron código, reglas, PRs ajenos ni producción.

## Falta
- Revisar el diff, commit con firma Co-Authored-By de Codex, publicar PR, obtener auditoría Tier 3 del SHA exacto, CI, verify-integration, merge y post-merge-guard.
- Después rebasar #390 sobre el tip oficial nuevo y recalcular el comparador con el árbol final.

## Decisiones
- Mantener #388 en su worktree de owner.
- No generar ni sustituir la raíz confiable ausente; producción queda bloqueada hasta que exista fuente autorizada y ledger verificado.

## Siguiente comando
`git -C /home/fruiz/worktrees/jax-faro-grant-handoff-refresh diff --check && git -C /home/fruiz/worktrees/jax-faro-grant-handoff-refresh diff -- docs/historia/2026-10-10-faro-floor-retirement-grant.md`
