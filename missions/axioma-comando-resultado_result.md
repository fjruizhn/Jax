# Resultado de: axioma-comando-resultado.md

Listo. Resumen de lo que se hizo:

**Dos causas raíz, un fix de tres archivos:**

- **`useJaxStore.js`**: helpers `_loadPendingIds`/`_savePendingIds` para persistir en `localStorage['jax_pending_cmds']`; función `registerPendingCommand(taskId)` que graba al lanzar; `restorePendingTasks()` que reconstruye mensajes pendientes al arrancar; `checkPendingTasks` ahora hace polling cada 5s si quedan tareas en curso; y `command_completed` limpia el id del localStorage al completar

- **`useWebSocket.js`**: en primera conexión llama `restorePendingTasks()` (antes solo hacía `checkPendingTasks` en reconexión, lo que era inútil con `messages: []` vacío)

- **`BottomBar.jsx`**: llama `registerPendingCommand(taskId)` inmediatamente después del POST exitoso al backend

Build limpio (268 módulos, 0 errores), hot reload activo.

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
