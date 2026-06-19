# Resultado de: jax-platform-ws-fix.md

Listo. Resumen de lo que se hizo:

---

**Fix aplicado en 3 archivos:**

**`websocket.js`** — Backoff exponencial real: 1s → 2s → 4s → 8s → 16s → 30s (tope). Status "reconnecting" durante los reintentos, "disconnected" solo al cerrar sesión. También corregido un bug: `clearInterval` → `clearTimeout`.

**`useJaxStore.js`** — Dos cambios:
1. `command_completed` ahora muestra el preview inmediatamente y luego hace GET `/api/command/{task_id}` para obtener el resultado completo (sin el truncado de 500 chars del payload WS).
2. Nueva acción `checkPendingTasks`: busca mensajes con `status='running'` e `id.startsWith('cmd-')` y fetchea su resultado.

**`useWebSocket.js`** — Llama `checkPendingTasks()` al reconectar (no en la primera conexión, solo en reconexiones). Usa `everConnectedRef` para distinguir los dos casos.

**Backend sin cambios** — `command.py` ya publicaba correctamente `facet_status_changed hyde thinking` al iniciar y `hyde idle` al terminar. El Ojo HAL naranja ya funcionaba a nivel de datos; el flujo WS era el que faltaba completar en el frontend.

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
