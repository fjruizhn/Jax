faceta: hyde

# Axioma — DALL-E 3 + fix título browser

## TAREA 1 — DALL-E 3 en Axioma

### Backend: POST /api/image/generate
- Usar OPENAI_API_KEY de /etc/jax/.env
- Llamar https://api.openai.com/v1/images/generations
- Body: {"model": "dall-e-3", "prompt": "...", "size": "1024x1024", "quality": "standard", "n": 1}
- Devolver: {"url": "...", "revised_prompt": "..."}
- Publicar evento WS: {"event_type": "image_generated", "url": "...", "prompt": "..."}

### Frontend: selector de modo "Imagen" en BottomBar
- Nuevo tab "Imagen" junto a Chat/Comando/Pipeline
- Input: "Describe la imagen que quieres generar..."
- Al enviar: POST /api/image/generate
- Mostrar en el chat: mensaje del usuario + imagen generada inline (tag <img>)
- El ojo HAL cambia a violeta (#7c3aed) durante la generación
- Texto i18n en es.js y en.js
- Funciona en dark y light mode

## TAREA 2 — Fix título tab browser
El tab del browser dice "JAX | Platform v0.1"
Verificar que index.html dice <title>Axioma</title>
Si no, corregirlo y hacer rebuild.

## Verificaciones
1. POST /api/image/generate con prompt "a HAL 9000 eye in space, cinematic" → devuelve URL
2. En Axioma modo Imagen → escribir prompt → imagen aparece en el chat
3. Tab del browser dice "Axioma"
4. Ojo HAL violeta durante generación

Escribir resultado en ~/jax/missions/axioma-dalle-titulo_result.md
