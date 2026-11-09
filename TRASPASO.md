# Traspaso · OP#225 mensajes FULLTEXT · 2026-11-09

## Objetivo

Dar a `messages` una señal textual de recuperación para que axioma-memoria pueda fusionarla con su señal vectorial, sin cambiar el ranking documental por cuotas de tipo.

## Hecho y verificado

- Fernando pidió corregir OP#225 comentario 1561; la rama propia parte de `origin/master` fd27989b.
- `messages` pertenece a JAX; axioma-memoria tiene solo SELECT sobre la tabla. Por eso el índice FULLTEXT se prepara aquí, en la ruta de migraciones JAX, antes de activar el lector en el otro repo.
- Arquitecto Sol recomendó índice textual y RRF plano; medición mostró que la búsqueda semántica global no recupera bien fragmentos parciales de mensajes largos.
- La rama claude-skills paralela es `codex/op225-rrf-shutdown-20261109`; ya contiene apagado acotado y prepara FULLTEXT de hechos. No integrar lector de mensajes antes de instalar este índice.
- JAX PR #349 de deadlocks en `messages` sigue abierto; comprobar compatibilidad antes de integrar.

## Falta

- Implementar y probar migración JAX idempotente con `lock_wait_timeout`, costo y rollback explícito; ensayar en base desechable.
- Auditoría Sol del SHA exacto y CI; PR propio, merge autorizado.
- Respaldo restaurado, DDL en producción con ventana vigente y EXPLAIN de consulta real; Biblioteca.
- Solo después, implementar lector SELECT y probar corpus de mensajes en axioma-memoria.

## Siguiente comando

`git status --short` en este worktree; revisar migración y prueba del implementador.

Decisión de trabajo: Fernando Ruiz pidió corregirlo; arquitectura técnica recomendada por Sol y adoptada por Codex. Este archivo es contexto, no autoridad de producción.
