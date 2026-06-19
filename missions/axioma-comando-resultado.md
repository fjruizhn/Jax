faceta: hyde

# Fix crítico: resultado del modo Comando siempre visible en Axioma

## Problema
El modo Comando ejecuta tareas correctamente pero el resultado no aparece en el chat.
El WS se desconecta durante tareas largas y el evento command_completed se pierde.

## Fix
En backend/api/command.py:
- Al completar la tarea, guardar el resultado en DB o en memoria del JAX Engine indexado por task_id
- Nuevo endpoint: GET /api/command/{task_id}/result → devuelve resultado aunque WS se haya perdido

En frontend useJaxStore.js:
- Al reconectar el WS: hacer GET /api/command/{task_id}/result para cada comando pendiente
- Al cargar el dashboard: verificar comandos completados sin mostrar y mostrarlos
- El mensaje de Hyde con spinner → reemplazar con resultado real al obtenerlo

## Verificación
1. Lanzar comando → desconectar WS manualmente → reconectar → resultado aparece
2. Refrescar página → comandos completados siguen visibles en el historial

Escribir resultado en ~/jax/missions/axioma-comando-resultado_result.md
