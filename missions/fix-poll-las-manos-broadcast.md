/using-superpowers
/ruflo

# Misión: _poll_las_manos deja de mandarle el evento de salud solo al usuario "1"

## Contexto

Hallazgo de la misión `fix-performance-sse-userid` (2026-07-18): `backend/jax_engine/state.py:_poll_las_manos`
publica el evento `las_manos_health_changed` con `tenant_id="1"`/`user_id="1"` hardcodeado.
Impacto real confirmado (no cosmético): solo el usuario literal "1" en el tenant literal "1"
recibe el evento en vivo — cualquier otro usuario solo se entera del estado de LAS MANOS en el
próximo `loadState()` (refresh), no en tiempo real.

Se descartó agregar un `EventBus.broadcast()` nuevo (cruzaría la regla de CLAUDE.md "WS canal
por usuario — nunca por tenant"). Decisión de diseño confirmada por Fernando: **reusar el
`event_bus.publish(tenant_id, user_id)` que ya existe**, iterando sobre los usuarios realmente
conectados (`engine_state`) y publicando el evento una vez por usuario real — sigue siendo
estrictamente por-usuario, solo que ahora a varios usuarios en vez de uno hardcodeado. Sin
primitivo nuevo en `EventBus`.

## El fix

1. Reconocimiento: confirmar cómo `engine_state` expone los usuarios conectados hoy
   (`connected_users`, `_user_tenant_map` — ver `backend/jax_engine/state.py:68-76`) y si hay
   ya un método público para iterarlos, o si hay que agregar uno.
2. TDD: test que reproduce el bug primero — con 2+ usuarios reales "conectados" (vía
   `register_user`), confirmar que hoy el evento `las_manos_health_changed` solo le llega al
   hardcodeado "1" (RED). Después el fix: en vez de un único `publish(tenant_id="1", user_id="1", ...)`,
   iterar sobre los usuarios conectados y publicar el evento a cada uno con su tenant_id/user_id
   real. Confirmar que todos los usuarios conectados reciben el evento (GREEN).
3. Caso borde: cero usuarios conectados (nadie logueado cuando cambia el estado de salud) —
   no debe romper el poller, simplemente no publica nada.
4. Correr el suite completo antes/después.

## Reglas generales (mismas que la misión anterior)

- Worktree aislado, no trabajar sobre master.
- TDD real: RED contra el código sin modificar primero, GREEN después.
- NO reiniciar producción sin confirmación explícita de Fernando.
- NO mergear a master sin gate completo (`./scripts/test.sh`) pasando y sin reporte.
- Si el alcance resulta mucho más grande de lo que este brief asume, PARAR y reportar antes de
  intentar algo mayor.

## Reporte final

Diff, qué se verificó y cómo (RED/GREEN), estado final.
