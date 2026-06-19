faceta: hyde

# JAX Platform — Fix completo: pipeline results + autonomous + comando results

## TAREA 1 — Fix pipeline results (0 steps + duplicados)

### Diagnóstico primero
Hacer GET http://127.0.0.1:7777/jacobs/pipeline/{id} con el último pipeline completado.
Identificar exactamente cómo vienen los steps y el context en la respuesta.

### Fix: steps count
En useJaxStore.js cuando pipeline_completed llega:
- GET /api/pipelines/{id}/results
- Loguear response completo
- Si steps viene vacío, buscar en data.step_results, data.steps, data.pipeline.steps
- Mostrar "Pipeline completado — N steps" con el N correcto

### Fix: duplicados
Usar un Set pipeline_ids_processed en el store.
Si pipeline_id ya está en el Set al recibir pipeline_completed → ignorar.
Limpiar el Set cuando pipeline tiene más de 1 hora.

### Fix: panel derecho 0/3
El polling de /api/pipelines/{id} debe devolver steps con status.
Verificar que RightPanel recibe y muestra steps aunque pipeline esté completed.
Steps completados deben verse en verde con duración.

### Fix: resultados en chat por faceta
Cuando pipeline completa, para cada step con resultado:
- Mensaje en chat con icono/color de la faceta
- Primeras 800 chars del resultado
- Si resultado > 800 chars: agregar botón "Ver completo" que expande

---

## TAREA 2 — Habilitar modo autonomous en Jacobs

### En ~/jax/jacobs/policy.py
Remover o comentar el bloque que rechaza modo autonomous.
El modo autonomous debe ejecutar todos los steps sin pausar para aprobación.
Kill switch sigue funcionando — si /etc/jax/PAUSE existe, aborta.
Límites siguen activos — max 20 steps, max 3 pipelines paralelos.

### En ~/jax/jacobs/executor.py
En modo autonomous:
- No pausar entre steps
- No esperar aprobación
- Ejecutar step 1 → step 2 → step 3 en secuencia automática
- Si un step falla: abortar pipeline (default_on_fail = abort)
- Publicar eventos WS en cada transición de step

### Verificación autonomous
POST /jacobs/pipeline con mode=autonomous, 3 steps (jax_local, jax_local, jax_local)
→ debe completar sin intervención humana
→ panel derecho muestra progreso en tiempo real
→ chat muestra resultados al completar

---

## TAREA 3 — Fix Comando: resultado aparece en chat

### Problema
Modo Comando lanza tarea de Hyde pero el resultado no aparece en el chat.
El WS emite command_completed pero el frontend no actualiza el mensaje.

### Fix backend
En backend/api/command.py el polling del result file:
- Cuando result file existe: leer contenido completo
- Publicar evento WS:
{
  "event_type": "command_completed",
  "task_id": "uuid",
  "result": "contenido completo del result file",
  "status": "completed"|"failed"
}

### Fix frontend
En useJaxStore.js handler de command_completed:
- Encontrar el mensaje de Hyde con ese task_id (status="running")
- Reemplazar su contenido con el resultado real
- Renderizar markdown
- Cambiar status de "running" a "completed"
- Hyde en panel izquierdo vuelve a "idle"

### Fix: Hyde thinking durante comando
En backend/api/command.py:
- Al iniciar: publicar facet_status_changed hyde thinking
- Al terminar: publicar facet_status_changed hyde idle
- Ojo HAL debe cambiar a naranja durante el comando

### Verificación comando
1. Modo Comando → "lista los archivos en ~/jax/las_manos/"
2. Hyde muestra "thinking" en panel + ojo naranja
3. Al terminar: resultado aparece en chat reemplazando el spinner
4. Hyde vuelve a "idle"

---

## Verificaciones finales conjuntas

1. Pipeline autonomous 3 steps → completa solo → resultados en chat
2. Pipeline supervised 3 steps → apruebo cada step → resultados en chat  
3. Modo Comando → resultado aparece en chat
4. Sin duplicados en ningún caso
5. Panel derecho muestra steps con duración
6. Ojo HAL cambia correctamente en todos los modos

## NO tocar
- LAS MANOS internals
- ~/jax/ terminal
- Kill switch behavior

Escribir resultado detallado en ~/jax/missions/jax-platform-fix-all_result.md

En memoria de Jairo Urbina. En honor al Prof. Raúl Jacobs.
