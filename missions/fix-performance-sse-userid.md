/using-superpowers
/ruflo

# Misión: performance + SSE lock + user_id hardcodeado

## Contexto

Tres frentes pendientes de la auditoría 2026-07-08 y de los hallazgos de
las últimas dos sesiones. Se atacan juntos por pedido explícito de Fernando,
pero son independientes entre sí — tratarlos como 3 sub-tareas separadas,
cada una con su propio test y su propio commit, no un commit gigante.

Infraestructura de tests ya existe (`backend/tests/`, `frontend` con vitest,
`jax_memory_test`, `scripts/test.sh`). Usarla para todo lo que se toque acá.

## Sub-tarea A — Zustand sin selectores + React.memo

1. Reconocimiento: confirmar el alcance real hoy (¿sigue siendo 80% de
   re-renders innecesarios? ¿0 archivos con React.memo en
   frontend/src/components, como decía el hallazgo original?).
2. Fix: introducir selectores de Zustand donde los componentes hoy
   suscriben el store completo en vez de la porción que usan. Agregar
   `React.memo` a los componentes que lo ameriten (listas, filas repetidas,
   cualquier cosa que rerenderice sin cambio de props).
3. No hay forma sencilla de testear "cantidad de re-renders" con vitest
   sin herramientas adicionales — si no hay ya algo para esto, no instalar
   nada nuevo sin preguntar. Documentar en el reporte cómo se verificó la
   mejora (ej. React DevTools Profiler manual, conteo de renders con un
   contador temporal) en vez de forzar un test automatizado que no aporta.

## Sub-tarea B — N+1 en /api/admin/keys (u otros endpoints admin similares)

1. Reconocimiento: confirmar dónde exactamente está el N+1 (¿una query por
   fila en vez de un JOIN o un IN?). Puede estar en `keys.py`,
   `dashboard.py`, u otro admin endpoint — no asumir cuál sin confirmar.
2. Fix: consolidar a una sola query (JOIN o `WHERE id IN (...)`) donde
   corresponda.
3. Test: si existe fixture con múltiples filas de datos, un test que
   cuente queries ejecutadas (via mock/spy sobre el cursor, o logging de
   queries) antes/después, confirmando que bajó de N+1 a O(1) o O(log n).
   Si no hay forma limpia de contar queries con la infraestructura actual,
   al menos un test funcional que confirme que el endpoint sigue
   devolviendo los datos correctos tras el fix.

## Sub-tarea C — SSE /api/events sin _ws_lifecycle_lock + user_id="1" hardcodeado

Dos hallazgos relacionados con aislamiento por usuario, tratar juntos:

1. `backend/api/events.py` (el endpoint SSE) no adquiere
   `_ws_lifecycle_lock` al hacer `event_bus.subscribe`/`unsubscribe`,
   dejando la misma clase de carrera que se cerró para WS en la sesión
   anterior, pero cross-channel (SSE vs WS). Confirmar primero si el
   endpoint sigue "dormido" (sin uso real en el frontend actual, como se
   documentó) — si sigue sin uso, el fix es de todas formas correcto
   aplicarlo ahora que se sabe del problema, no esperar a que se active.
2. `user_id="1"` hardcodeado en los pollers de `state.py` — esto es
   deuda de aislamiento real: los eventos de sistema no distinguen
   usuarios reales, van todos etiquetados al usuario 1. Confirmar el
   alcance exacto (¿todos los eventos de sistema, o solo algunos tipos?)
   antes de decidir el fix — puede requerir pasar el `user_id` real desde
   donde se origina el evento, no un fix trivial de una línea.

TDD igual que las sesiones anteriores: test que reproduce el problema
primero (para el lock: mismo patrón de neutralizar el lock y confirmar
pérdida, luego restaurar y confirmar que no se pierde; para el
hardcodeo: test que confirma que un evento de sistema real lleva el
user_id correcto, no "1" fijo).

## Reglas generales para toda la misión

- Worktree aislado (skill de git worktrees), no trabajar sobre master.
- Cada sub-tarea es su propio commit, con su propio ciclo RED/GREEN donde
  aplique test automatizado.
- NO reiniciar producción sin confirmación explícita de Fernando.
- NO mergear a master sin gate completo (`./scripts/test.sh`) pasando y
  sin reporte final de las 3 sub-tareas.
- Si alguna sub-tarea resulta ser mucho más grande de lo que este brief
  asume (ej. el N+1 real requiere reestructurar el schema), PARAR esa
  sub-tarea específica, reportar el hallazgo, y seguir con las otras dos
  — no intentar una reestructuración mayor sin aprobación.

## Reporte final

Por cada sub-tarea: diff, qué se verificó y cómo (test automatizado o
verificación manual documentada), y estado (completo / completo con
salvedad / bloqueado y por qué).
