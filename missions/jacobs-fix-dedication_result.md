# Resultado de: jacobs-fix-dedication.md

Listo. Resumen del diagnóstico:

**Causa raíz:** El código fuente (`routes.py`) ya había sido editado a las 11:50 para eliminar `dedication` del return dict. El proceso uvicorn (PID 945152) arrancó a las 11:38, antes de ese edit, y tenía el código viejo en memoria. Systemd intentaba reiniciar pero fallaba porque el proceso fantasma ocupaba el puerto 7777 (231 reinicios fallidos).

**Acción:** Matar PID 945152. Systemd reinició solo con el código correcto (PID 952293).

**Verificación:** Los cuatro endpoints (`/jacobs/plan`, `/jacobs/pipeline`, `/jacobs/pipeline/{id}`, `/jacobs/pipeline/{id}/events`) devuelven respuestas sin `dedication`. El servicio está `active (running)`.

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
