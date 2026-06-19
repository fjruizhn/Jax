# Resultado de: jax-platform-fix-duplicate.md

Fix listo. Resumen:

**Causa raíz:** El evento `facet_response_completed` llegaba DOS VECES por WebSocket. En `_fire_completed` del backend se publicaba al `event_bus` (que ya tiene una suscripción que llama `ws_hub.send_to_user`) Y adicionalmente se llamaba `ws_hub.broadcast_to_tenant` directamente. Doble entrega → doble `addMessage`.

**Fix aplicado (Opción A):**

1. **`useJaxStore.js`**: El handler de `facet_response_completed` ya no agrega mensajes — solo resetea `activeFacet: null`. El estado thinking→idle sigue funcionando por el evento `facet_status_changed` separado.

2. **`BottomBar.jsx`**: Removida la condición `if (wsStatus !== 'connected')`. El HTTP response siempre agrega el mensaje, sin importar el estado del WS.

El ojo HAL y los colores de las facetas no se tocan — dependen de `facet_status_changed`, canal que no modificamos.

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
