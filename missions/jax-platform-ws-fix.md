faceta: hyde

# Fix crítico: WebSocket reconexión + resultado comando en chat

## Problema 1: WS se desconecta y no reconecta
En useWebSocket.js el WS se cae durante tareas largas y muestra "disconnected".
Necesita reconexión automática con backoff exponencial.

## Fix 1: useWebSocket.js
- Reconexión automática cuando se cierra el WS
- Backoff: 1s, 2s, 4s, 8s, máximo 30s
- Mostrar estado "reconnecting..." en lugar de "disconnected"
- Al reconectar: re-suscribirse a todos los eventos
- No reconectar si el usuario cerró sesión (logout)

## Problema 2: Resultado del comando no aparece en chat
Cuando command_completed llega por WS, el frontend debe actualizar el mensaje de Hyde
(que está en estado "running" con spinner) con el resultado real.

## Fix 2: Manejo de command_completed en frontend
- El mensaje de Hyde con status="running" tiene el task_id
- Cuando llega evento command_completed con ese task_id:
  - Hacer GET /api/command/{task_id} para obtener resultado completo
  - Reemplazar el mensaje "running" con el resultado real
  - Renderizar markdown en el resultado
- Si WS estaba caído: al reconectar, hacer GET /api/command/{task_id} 
  para los tasks pendientes

## Problema 3: Ojo HAL y estado Hyde no cambian durante comando
- Al iniciar comando: publicar facet_status_changed hyde thinking
- Al terminar: publicar facet_status_changed hyde idle
- Esto ya debe estar en backend/api/command.py — verificar que se publica

## Verificaciones obligatorias
1. Enviar comando → WS puede desconectarse → reconecta automáticamente
2. Resultado del comando aparece en el chat al completarse
3. Hyde muestra "thinking" en panel izquierdo durante el comando
4. Ojo HAL cambia a naranja (#f97316) durante comando de Hyde
5. Al terminar: Hyde vuelve a "idle", ojo vuelve a azul
6. Probar con tarea larga (>60s) — WS no debe perderse el resultado

Escribir resultado en ~/jax/missions/jax-platform-ws-fix_result.md
