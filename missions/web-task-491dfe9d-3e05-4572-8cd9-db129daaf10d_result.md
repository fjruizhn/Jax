# Resultado de: web-task-491dfe9d-3e05-4572-8cd9-db129daaf10d.md

Listo. Resumen de los dos cambios:

**CenterPanel.jsx** — `h-full` → `flex-1 min-h-0`  
El problema raíz: `h-full` en un flex child intenta ser 100% del contenedor padre, dejando sin espacio al BottomBar. Con `flex-1 min-h-0` el panel de conversación ocupa el espacio restante después de BottomBar y el `overflow-y-auto` interno funciona correctamente.

**Message.jsx** — se agrega `maxHeight: '400px', objectFit: 'contain'` a la imagen  
Evita que imágenes de 1024×1024px (DALL-E) ocupen toda la pantalla, respetando proporciones.

El BottomBar ya tenía `flex-shrink-0` así que queda anclado — no se necesita `position: fixed`.

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
