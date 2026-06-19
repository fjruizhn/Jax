# Resultado de: jax-platform-chat-v01.md

Listo. Resumen de lo que se implementó y verificó:

---

## Qué se hizo

**Backend** (`backend/api/chat.py` — nuevo):
- `POST /api/chat` con body `{message, facet}`
- Carga el `.env` de `/etc/jax/` automáticamente al importar
- Lee system prompts del `config.toml` real de JAX
- 7 rutas: Ollama local, DeepSeek, Gemini (con grounding), OpenAI, Kimi, y dos respuestas especiales (Hyde / Ada)
- Router automático por palabras clave si no se especifica faceta
- Historial por `user_id` en memoria, máximo 20 turnos
- Publica eventos WS: `thinking` → `facet_response_completed` → `idle`

**Frontend** (`BottomBar.jsx`, `useJaxStore.js`):
- Tabs de faceta coloreados arriba del input — activo adopta el color de la faceta
- POST real con JWT + `{message, facet}`
- Si WS desconectado: agrega mensaje desde HTTP response
- El ojo HAL ya respondía a `facet_status_changed` — solo faltaba que el backend publicara el evento

**Verificaciones pasadas (terminal):**
- jax_local en español hondureño ✅
- Hipatia busca con grounding Gemini ✅
- Jekyll responde formal sobre el Barroco ✅
- Thot con gpt-5.5 ✅
- Kimi con kimi-k2.7-code ✅
- Hyde y Ada respuestas especiales ✅
- Historial: turno 2 recuerda turno 1 ✅

**Nota**: el browser tiene un token expirado — hay que reloguear para ver el WS conectado y el ojo animarse.

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
