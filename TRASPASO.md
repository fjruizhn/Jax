# Traspaso — Faro F1.1 avisos de decisión · PR #376

## Objetivo

Entregar los avisos del Rule Authority después de la decisión durable, sin hacerlos
parte de la autoridad. El aviso inmediato es fail-soft; el resumen diario conserva
entrega al menos una vez y no descarta avisos ante fallos.

## Estado verificado · 2026-10-07

- PR #376 (`feat/faro-f1.1-avisos-r2`) está apilado sobre el evaluador de #371.
- Su padre avanzó de `b967dff` a `d4ffe62` (r3). La sincronización solo conflictuó
  en `ci/pisos.json` y `docs/ci/pisos.md`.
- Se conservó el piso `faro-fase0/faro = 768`: el comando exacto del job pasó en
  el worktree con Python 3.14.4, **768 passed, 0 skipped**. La nota compara con el
  nuevo baseline `727` de #371 r3, por eso registra +41.
- El merge con #371 r3 incorpora también la corrección de readiness de MariaDB y
  los cambios de CI del padre.
- La punta publicada de #376 aún debe obtener CI nuevo después de este merge.
- PR #382 se mantiene como PR documental aparte; su rama se sincronizó con master y
  su base se cambió a master para evitar comparar el piso contra una rama vieja.
- PR #381 es independiente y sigue rechazado por auditoría; no reutilizar su SHA ni
  confundirlo con aprobación del ledger.

## Siguiente paso

1. Terminar la resolución de los dos archivos de pisos y revisar el diff.
2. Crear el merge commit con `Co-Authored-By: Codex <noreply@openai.com>` y publicar
   la rama de #376.
3. Confirmar que #376 queda mergeable y que los checks del SHA nuevo pasan, incluido
   `faro-fase0`, sin bajar pisos ni saltar pruebas.
4. Mantener ambos PR abiertos; esta ronda no integra a master.
