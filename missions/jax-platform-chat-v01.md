faceta: hyde

# JAX Platform — Chat real conectado a JAX

## Contexto
LA CARA v0.1 está viva en ~/jax-platform/.
Backend en puerto 8080, frontend en 5173.
El chat muestra "/chat aún no implementado" — hay que conectarlo real.

## Tarea

### 1. Backend — POST /api/chat
Nuevo endpoint en backend/api/chat.py:

POST /api/chat
Body: {"message": "hola", "facet": "jax_local"}
Response: {"facet": "jax_local", "response": "...", "timestamp": "..."}

Lógica:
- Leer el mensaje y la faceta solicitada
- Si facet == "jax_local": POST http://localhost:11434/api/chat (Ollama, qwen3:14b)
- Si facet == "jekyll": POST DeepSeek API (DEEPSEEK_API_KEY de /etc/jax/.env)
- Si facet == "hipatia": POST Gemini API (GEMINI_API_KEY)
- Si facet == "thot": POST OpenAI API (OPENAI_API_KEY, gpt-5.5)
- Si facet == "kimi": POST https://api.moonshot.ai/v1/chat/completions (KIMI_API_KEY)
- Si facet == "hyde": respuesta especial "Hyde opera en modo tarea autónoma — usa el modo Comando"
- Si facet == "ada": respuesta especial "Ada está pendiente — key Z.ai disponible semana 22-jun"

System prompts: leer de ~/jax/config/config.toml (sección [personalities.*])

Después de responder → publicar evento WebSocket:
{
  "event_type": "facet_response_completed",
  "tenant_id": "1",
  "user_id": "1",
  "facet": "jax_local",
  "message_preview": "primeras 100 chars",
  "timestamp": "ISO8601"
}

También publicar evento al inicio:
{
  "event_type": "facet_status_changed",
  "facet": "jax_local",
  "status": "thinking"
}

Y al terminar:
{
  "event_type": "facet_status_changed",
  "facet": "jax_local",
  "status": "idle"
}

### 2. Router automático
Si no se especifica facet, usar el router de JAX para decidir:
- Palabras técnicas (código, servidor, bash) → hyde (respuesta especial)
- Palabras de investigación (busca, investiga, noticias) → hipatia
- Palabras humanistas (poesía, filosofía, arte) → jekyll
- Crítica, auditoría, riesgo → thot
- Código, refactor, construir → kimi
- Default → jax_local

### 3. Historial de conversación
Mantener historial en memoria por user_id (dict en el JAX Engine state.py):
conversations: dict[user_id, list[{role, content, facet, timestamp}]]

Máximo 20 turnos en memoria (igual que JAX terminal).
Pasar historial a cada invocación de faceta.

### 4. Frontend — conectar chat real
En CenterPanel.jsx o donde esté el input:
- Al enviar mensaje: POST /api/chat con JWT
- Mostrar estado "pensando..." mientras espera
- Cuando llega respuesta: mostrar mensaje con icono de faceta y color
- El ojo HAL debe cambiar de color según la faceta activa (ya existe el evento WS)
- Selector de faceta en la barra inferior (dropdown o tabs)

### 5. Verificaciones obligatorias
1. Escribir "hola" en el chat → jax_local responde en español hondureño
2. Escribir "investiga qué es HAMMURABI" → hipatia responde con fuentes
3. Cambiar a Jekyll → escribir "qué es el barroco" → Jekyll responde formal
4. El ojo HAL cambia de color durante la respuesta
5. El estado de la faceta cambia a "thinking" en panel izquierdo durante respuesta
6. Historial persiste en la sesión (turno 2 recuerda turno 1)

## NO tocar
- LAS MANOS (~/jax/las_manos/)
- Jacobs (~/jax/jacobs/)
- JAX terminal (~/jax/)

Escribir resultado en ~/jax/missions/jax-platform-chat-v01_result.md

En memoria de Jairo Urbina. En honor al Prof. Raúl Jacobs.
