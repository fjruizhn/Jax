# Resultado de: jax-platform-pipeline-results.md

Listo. Resumen de lo que se hizo:

---

**Backend — cero cambios.** Ambos endpoints ya existían:
- `GET /jacobs/pipeline/{id}/results` en `jacobs/routes.py:364` (resuelve `output_ref` y devuelve texto + duración por step)
- `GET /api/pipelines/{id}/results` en `backend/api/pipelines.py:53` (proxy al anterior)

**`useJaxStore.js`** — handler nuevo para `pipeline_step_changed` cuando `payload.status === 'completed'`:
1. Llama `GET /api/pipelines/{id}/results`
2. Por cada step con resultado → mensaje al chat con la faceta correspondiente, primeras 800 chars
3. Mensaje final de Jacobs: "Pipeline completado — N steps, Xs totales"

**`StepCard.jsx`** — cuando el step está `completed`:
- Preview del resultado (100 chars en cursiva)
- Cursor pointer + click → agrega el output al chat sin API call extra

**`RightPanel.jsx`** — pasa `pipelineId` a `StepCard` para el click-to-expand.

Build confirmado limpio (264 módulos, sin errores). Pendiente: prueba manual con un pipeline real corriendo.

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
