faceta: hyde

# Fix: mensajes duplicados en el chat de LA CARA

## Problema
Cada respuesta de faceta aparece dos veces en el chat.
Causa probable: el frontend agrega el mensaje tanto al recibir la respuesta HTTP del POST /api/chat como al recibir el evento WebSocket facet_response_completed.

## Fix
En el frontend, elegir UNA sola fuente de verdad para agregar mensajes al chat:
- OPCIÓN A: Solo HTTP response agrega el mensaje. El evento WS solo actualiza estado (thinking/idle) pero NO agrega mensaje.
- OPCIÓN B: Solo WS agrega el mensaje. HTTP response solo devuelve {status: "ok"} sin contenido.

Recomendar OPCIÓN A — es más simple y predecible.

El evento WS facet_response_completed solo debe:
- Actualizar facet status a "idle"
- Actualizar activeFacet a null
- NO agregar mensaje al chat

## Verificación
1. Enviar mensaje → respuesta aparece UNA sola vez
2. El estado de la faceta sigue cambiando correctamente (thinking → idle)
3. El color del ojo sigue funcionando

Escribir resultado en ~/jax/missions/jax-platform-fix-duplicate_result.md
