faceta: hyde

# Fix: mostrar resultados del pipeline en el chat

## Problema
Cuando un pipeline de Jacobs completa, el chat solo muestra "Pipeline iniciado... Siguiendo progreso en panel derecho" pero nunca muestra los resultados de cada step ni el resumen final.

## Fix

### Backend — GET /api/pipelines/{id}/results
Nuevo endpoint que lee el contexto del pipeline de Jacobs y devuelve los outputs de cada step:
- GET http://127.0.0.1:7777/jacobs/pipeline/{id}
- Extraer context (step_0_ref, step_1_ref, step_2_ref)
- Parsear cada ref (inline:JSON o artifact_ref)
- Devolver:
{
  "pipeline_id": "...",
  "name": "...",
  "status": "completed",
  "steps": [
    {
      "step_index": 0,
      "facet": "hipatia",
      "result": "texto completo del resultado",
      "duration_seconds": 12.3
    }
  ]
}

### Frontend — Mostrar resultados cuando pipeline completa
Cuando llega evento pipeline_completed (o cuando polling detecta status=completed):
1. GET /api/pipelines/{id}/results
2. Para cada step completado, agregar mensaje al chat:
   - Icono y color de la faceta
   - Primeras 800 chars del resultado con "... [ver completo]"
   - Timestamp
3. Agregar mensaje final de Jacobs: "Pipeline completado — X steps, Y segundos totales"

### Panel derecho — mejorar StepCard
Cuando step está completed, mostrar:
- Duración del step
- Preview del resultado (primeras 100 chars)
- Click en el step → expande resultado completo en un modal o en el chat

## Verificaciones
1. Lanzar pipeline → al completar, resultados aparecen en chat por faceta
2. Cada mensaje tiene icono/color de su faceta
3. Panel derecho muestra duración de cada step
4. Click en step completado → muestra resultado completo

Escribir resultado en ~/jax/missions/jax-platform-pipeline-results_result.md
