faceta: hyde

# JAX Platform v0.2 — Thinking + Comando + Pipeline completos

## Contexto
JAX Platform v0.1 está viva en ~/jax-platform/.
Chat funciona con todas las facetas.
Tres cosas faltan: thinking state, modo Comando, modo Pipeline.

## TAREA 1 — Fix thinking state

### Problema
El panel izquierdo no cambia a "thinking" durante respuestas del chat.
El ojo HAL no cambia de color durante respuestas.

### Fix backend (backend/api/chat.py)
Antes de llamar a la faceta publicar:
{
  "event_type": "facet_status_changed",
  "tenant_id": "1",
  "user_id": "1",
  "facet": "jax_local",
  "status": "thinking",
  "timestamp": "ISO8601"
}

Después de recibir respuesta publicar:
{
  "event_type": "facet_status_changed",
  "facet": "jax_local",
  "status": "idle",
  "timestamp": "ISO8601"
}

### Fix frontend
En useJaxStore.js — el evento facet_status_changed debe:
1. Actualizar facets[facet_name].status = "thinking"|"idle"
2. Si status == "thinking": actualizar activeFacet = facet_name
3. HalEye lee activeFacet del store → cambia color e intensidad del pulso
4. FacetCard muestra "Pensando..." con animación cuando status == "thinking"

---

## TAREA 2 — Modo Comando

El modo Comando permite ejecutar tareas autónomas de Hyde desde LA CARA.
Es equivalente a: jax --task ~/jax/missions/archivo.md

### Backend — POST /api/command
Body:
{
  "command": "string — descripción de la tarea para Hyde",
  "mode": "execute"|"dry_run"
}

Lógica:
1. Crear archivo temporal ~/jax/missions/web-task-{uuid}.md con:
   - faceta: hyde
   - contenido: el command del usuario
2. Ejecutar: subprocess async → jax --task ~/jax/missions/web-task-{uuid}.md
3. No esperar resultado (es async) — devolver inmediatamente:
   {
     "task_id": "uuid",
     "status": "running",
     "mission_file": "web-task-{uuid}.md",
     "result_file": "web-task-{uuid}_result.md"
   }
4. Polling endpoint: GET /api/command/{task_id} → lee el _result.md si existe

Publicar evento WS cuando arranca:
{
  "event_type": "command_started",
  "task_id": "uuid",
  "command_preview": "primeras 100 chars"
}

Publicar evento WS cuando termina (polling del result file cada 5s):
{
  "event_type": "command_completed",
  "task_id": "uuid",
  "result_preview": "primeras 500 chars del result"
}

### Backend — GET /api/command/{task_id}
Lee ~/jax/missions/web-task-{task_id}_result.md si existe.
Si no existe: {"status": "running"}
Si existe: {"status": "completed", "result": contenido_completo}

### Frontend — Modo Comando
Cuando el usuario selecciona modo "Comando" en la barra inferior:
- El placeholder del input cambia a: "Describe la tarea para Hyde..."
- Al enviar: POST /api/command
- Mostrar en el chat:
  - Mensaje del usuario con la tarea
  - Mensaje de Hyde: "Iniciando tarea autónoma... [task_id]"
  - Spinner mientras corre
  - Cuando llega evento command_completed: mostrar resultado
- Hyde en panel izquierdo cambia a "thinking" durante la tarea
- Cuando termina: vuelve a "idle"

---

## TAREA 3 — Modo Pipeline (Jacobs desde LA CARA)

### Backend — POST /api/pipeline
Proxy a POST http://127.0.0.1:7777/jacobs/pipeline
Agregar tenant_id y user_id al request.
Publicar evento WS cuando pipeline cambia de estado (polling a Jacobs cada 3s).

### Backend — GET /api/pipeline/{id}
Proxy a GET http://127.0.0.1:7777/jacobs/pipeline/{id}

### Backend — POST /api/pipeline/{id}/resume
Proxy a POST http://127.0.0.1:7777/jacobs/pipeline/{id}/resume

### Backend — POST /api/pipeline/{id}/cancel
Proxy a POST http://127.0.0.1:7777/jacobs/pipeline/{id}/cancel

### Polling de Jacobs desde JAX Engine
En jax_engine/state.py, loop async cada 3s:
- GET http://127.0.0.1:7777/jacobs/pipeline/{id} para cada pipeline activo
- Si cambió el estado: publicar evento WS:
  {
    "event_type": "pipeline_step_changed",
    "pipeline_id": "...",
    "step_index": N,
    "step_status": "running|completed|failed|blocked",
    "facet": "hipatia"
  }
- Si step está "blocked" (supervised mode): publicar:
  {
    "event_type": "human_gate_requested",
    "pipeline_id": "...",
    "step_index": N,
    "message": "Jacobs espera aprobación para continuar"
  }

### Frontend — Modo Pipeline
Cuando usuario selecciona modo "Pipeline":
- Input cambia a: "Describe el objetivo del pipeline..."
- Al enviar: abrir modal con opciones:
  - Modo: dry_run | supervised
  - Facetas a usar (checkboxes): hipatia, jekyll, thot, kimi, ada
  - Botón "Planificar y ejecutar"
- POST /api/pipeline con steps generados automáticamente según facetas seleccionadas
- Panel derecho (Jacobs) muestra pipeline en tiempo real:
  - Nombre del pipeline
  - Barra de progreso general
  - Lista de steps con estado (pending/running/completed/failed/blocked)
  - Icono y color de la faceta de cada step
  - Duración de cada step completado
  - Botón "Aprobar" cuando step está blocked (modo supervised)
  - Botón "Cancelar pipeline"

### Frontend — Botón Aprobar Step
Cuando llega evento human_gate_requested:
- Toast notification: "Jacobs espera tu aprobación"
- En panel derecho: step bloqueado con botón "✓ Aprobar"
- Al hacer click: POST /api/pipeline/{id}/resume
- Ojo HAL pulsa en ámbar mientras espera aprobación

---

## TAREA 4 — Audit log real en panel derecho

Tab "AUDIT" en panel derecho debe mostrar eventos reales.
GET /api/audit → últimos 20 eventos del audit log de LAS MANOS.
Endpoint lee ~/jax/las_manos/logs/audit.jsonl (últimas 20 líneas).
Polling cada 10s.
Mostrar: timestamp, evento, faceta/capability, status.

---

## Verificaciones obligatorias

1. Chat → ojo HAL cambia de color + faceta muestra "Pensando..." en panel izquierdo
2. Modo Comando → escribir "analiza el archivo ~/jax/las_manos/motor_registry/worker.py y lista sus funciones" → Hyde ejecuta autónomo, resultado aparece en chat
3. Modo Pipeline → crear pipeline con hipatia+jekyll+thot, objetivo "analiza el mercado bancario de Honduras" → panel derecho muestra steps en tiempo real
4. Modo supervised → botón aprobar aparece cuando Jacobs pausa
5. Tab Audit → muestra eventos reales de LAS MANOS
6. Kill switch → click en KILL → confirmación → touch /etc/jax/PAUSE → LAS MANOS reporta kill_switch_active

## Dependencias adicionales si faltan
- aiofiles (leer result files async)
- watchfiles (opcional, para detectar cambios en result files)

## NO tocar
- ~/jax/ (JAX terminal)
- LAS MANOS internals
- Jacobs internals

Escribir resultado detallado en ~/jax/missions/jax-platform-v02-full_result.md

En memoria de Jairo Urbina. En honor al Prof. Raúl Jacobs.
