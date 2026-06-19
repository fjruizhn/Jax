faceta: hyde

# Jacobs v0.2 — Executor real: facetas conectadas

__module_name__ = "Jacobs"
__dedication__ = "Prof. Raúl Jacobs — maestro, mentor, director."
__version__ = "0.2.0"

## Prerequisito
Jacobs v0.1 debe estar operativo antes de ejecutar esta misión.
Verificar: curl http://127.0.0.1:7777/jacobs/plan → respuesta válida.
Si v0.1 no está listo, ABORTAR esta misión.

## Contexto
Jacobs v0.1 tiene el skeleton completo: modelos, store, policy, routes, executor stub.
El executor stub simula steps sin invocar facetas reales.
Esta misión conecta el executor a las facetas reales de JAX.

## Tarea principal: executor.py real

Reemplazar el stub por invocaciones reales según step.facet:

### Facetas HTTP (hipatia, jekyll, thot, ada):
Usar httpx async para llamar directamente a la API de cada faceta.
Leer keys de /etc/jax/.env (ya cargadas en EnvironmentFile del servicio systemd).
Reutilizar el mismo patrón de base.py en ~/jax/jax/muscles/base.py.

Mapeo:
- hipatia → Gemini API (GEMINI_API_KEY, gemini-2.5-flash, grounding required_web)
- jekyll → DeepSeek API (DEEPSEEK_API_KEY, deepseek-v4-flash)
- thot → OpenAI API (OPENAI_API_KEY, gpt-5.5)
- ada → Z.ai API (ZHIPU_API_KEY, glm-5.2) — puede estar disabled, manejar gracefully

### Facetas Motor Registry (kimi):
POST http://127.0.0.1:7777/motor/dispatch con:
  caller = "jacobs"
  capability = step.capability
  prompt = step.input.get("prompt", "")
Esperar resultado con polling GET /motor/job/{job_id} cada 5s.
Timeout según step.timeout_seconds.

### Faceta local (jax_local):
POST http://localhost:11434/api/chat con:
  model = "qwen3:14b"
  messages = [{"role": "user", "content": prompt}]
  stream = false

### Hyde (ejecutor con human gate):
NO conectar en v0.2 — Hyde requiere human gate manual.
Si step.facet == "hyde": step → blocked_human_gate, pipeline pausa.
Fernando aprueba via POST /jacobs/pipeline/{id}/approve-step.

## Context propagation

Cada step recibe el output del step anterior como contexto.
Formato del input de cada step:
{
  "objective": "objetivo original del pipeline",
  "previous_outputs": [
    {"step_index": 0, "facet": "hipatia", "summary": "primeras 500 chars del output"},
    {"step_index": 1, "facet": "jekyll", "summary": "primeras 500 chars del output"}
  ],
  "prompt": "instrucción específica para este step"
}

Si output > 1MB → guardar en ~/jax/jacobs/artifacts/{pipeline_id}/{step_id}.json
En previous_outputs poner solo summary (500 chars), no el contenido completo.

## Nuevo endpoint: POST /jacobs/pipeline/{id}/approve-step

Aprueba el step bloqueado en hyde y lo ejecuta.
Solo válido si pipeline.mode == "supervised" o step.facet == "hyde".
Requiere que el step esté en status blocked_human_gate.

## Plan builder real (plan.py)

Conectar JAX Local (qwen3:14b via Ollama) para generar el plan desde el objetivo.
Prompt a JAX Local:
"Eres Jacobs, el Director. Dado este objetivo: {objective}
Genera un plan de ejecución con máximo {max_steps} steps.
Cada step debe tener: facet, capability, prompt específico.
Facetas disponibles: hipatia (investigar), jekyll (analizar), thot (criticar),
ada (diseñar arquitectura), kimi (coding), hyde (ejecutar cambios — requiere aprobación).
Responde SOLO con JSON válido. Sin explicaciones."

Validar que el JSON devuelto tiene máximo max_steps steps.
Si JAX Local devuelve texto inválido → usar plan_fallback con 3 steps genéricos.

## Prueba obligatoria en fuego

Pipeline de prueba real (modo supervised, 3 steps):
{
  "name": "Test pipeline real — investigar HAMMURABI",
  "objective": "Investiga qué es un sistema bancario core y lista sus 5 módulos principales",
  "invoked_by": "Fernando",
  "mode": "supervised",
  "steps": [
    {"facet": "hipatia", "capability": "research", "prompt": "Investiga qué es un sistema bancario core (core banking system) y lista sus 5 módulos principales con una descripción breve de cada uno."},
    {"facet": "jekyll", "capability": "analysis", "prompt": "Analiza la investigación anterior y desde una perspectiva humanista, reflexiona sobre cómo estos módulos afectan a las personas que usan el banco."},
    {"facet": "thot", "capability": "critique", "prompt": "Critica el análisis anterior. ¿Qué riesgos no se mencionaron? ¿Qué suposiciones son peligrosas?"}
  ]
}

Verificar:
1. Step 1 (Hipatia) ejecuta y devuelve resultado real con fuentes web
2. Step 2 (Jekyll) recibe output de step 1 y analiza
3. Step 3 (Thot) recibe outputs de steps 1 y 2 y critica
4. GET /jacobs/pipeline/{id} muestra status=completed
5. Cada step tiene trace_id y timestamps reales

## Verificaciones obligatorias
- py_compile en todos los archivos modificados
- Pipeline de prueba completa con 3 facetas reales
- Kill switch en vuelo: touch /etc/jax/PAUSE a mitad del pipeline → aborts
- Reiniciar LAS MANOS: sudo systemctl restart jax-las-manos → pipeline interrupted
- POST /jacobs/pipeline/{id}/resume → Fernando aprueba reanudación

## NO tocar
- motor_registry/ (ya funciona)
- Las candados de policy.py (no relajar límites)
- Modo autonomous (sigue deshabilitado)

Escribir resultado en ~/jax/missions/jacobs-v02_result.md

En memoria de Jairo Urbina. En honor al Prof. Raúl Jacobs.
