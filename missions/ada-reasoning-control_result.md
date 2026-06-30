# Resultado de: ada-reasoning-control.md

# Respuesta técnica — GLM-5.2 / Z.ai, control de razonamiento

Declaro upfront lo que sospechabas: **no tengo introspección de mi propio endpoint**. Lo que sigue es conocimiento recuperado de entrenamiento, marcado en consecuencia.

---

### 1. ¿Existe parámetro para controlar el razonamiento?

**SUPOSICIÓN (basada en patrón de familia GLM-4.5+, no verificada para GLM-5.2).**

En la documentación de Z.ai/Zhipu para GLM-4.5 y GLM-4.6 existe un parámetro `thinking` a nivel raíz del body, con un objeto que contiene `type` (`"enabled"` | `"disabled"`) y opcionalmente `budget` (entero, techo de tokens de razonamiento). Es plausible que GLM-5.2 herede este contrato porque el resto del esquema que describís (`reasoning_content` separado, `usage.completion_tokens_details.reasoning_tokens`, ruta `paas/v4/chat/completions`) coincide exactamente con esa familia.

**Pero "plausible por continuidad de familia" ≠ verificado.** No ejecuté un request con `thinking` contra el endpoint de GLM-5.2. No lo afirmo.

No conozco con certeza un parámetro alternativo (`reasoning_effort`, `enable_thinking`, etc.) para este endpoint. Si `thinking` no funciona, no tengo otro candidato confiable — sería inventar.

---

### 2. Sintaxis JSON (si la suposición de (1) es correcta)

**SUPOSICIÓN — verificar con curl antes de usar en producción.**

```json
{
  "model": "glm-5.2",
  "messages": [{"role": "user", "content": "..."}],
  "thinking": {
    "type": "disabled"
  }
}
```

o, para acotar sin apagar:

```json
"thinking": {
  "type": "enabled",
  "budget": 400
}
```

**Test mínimo para confirmar o refutar** (esto sí es verificable, lo ejecutás vos):

```bash
curl https://api.z.ai/api/paas/v4/chat/completions \
  -H "Authorization: Bearer $ZAI_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.2",
    "messages": [{"role":"user","content":"escribí HOLA"}],
    "thinking": {"type":"disabled"}
  }'
```

Tres resultados posibles, todos diagnósticos:
- `200` con `reasoning_content: null` o ausente → el parámetro funciona.
- `200` con `reasoning_content` poblado → el parámetro se ignora silenciosamente.
- `400` con error de schema → el nombre o la forma es incorrecta; el mensaje de error suele revelar el nombre correcto.

Ese curl es la única forma de convertir (1) y (2) de SUPOSICIÓN a VERIFICABLE.

---

### 3. ¿reasoning_tokens y content comparten max_tokens?

**VERIFICABLE — y ya verificado por tu medición.**

Tu dato: tarea trivial → 114 reasoning + 4 content. Si el razonamiento viviera en un presupuesto separado, no truncaría el `content` en documentos largos. Tu reporte de truncado es evidencia directa de que ambos salen del mismo techo `max_tokens`. La estructura `usage.completion_tokens_details.reasoning_tokens` (un *desglose* de completion_tokens, no una suma aparte) refuerza esto: `completion_tokens = content_tokens + reasoning_tokens`, y `completion_tokens ≤ max_tokens`.

**Consecuencia operativa verificable:** subir `max_tokens` SÍ evita el truncado, pero sube costo y latencia linealmente con el razonamiento. No es la solución óptima si el razonamiento es redundante para la tarea — ahí entra el punto (1).

---

### 4. ¿Razonamiento extenso ayuda a la formalización técnica?

**SUPOSICIÓN / heurística, no medido para GLM-5.2.**

Distinción útil:

- **Tareas donde el razonamiento probablemente ayuda:** descubrir invariantes no obvias, detectar casos límite, inferir tipos cuando el dominio es ambiguo, diseñar estructuras de datos desde restricciones sueltas.
- **Tareas donde el razonamiento extenso es probablemente redundante:** formalización donde el prompt ya fija el esquema objetivo, traducción directa de especificación a pseudocódigo tipado, refactor mecánico.

Para tu caso (esquemas tipados, invariantes, pseudocódigo) el valor del razonamiento depende de cuánta *decisión estructural* quede abierta. Si el prompt ya dice "dame el tipo X con campos Y", el razonamiento de 1000+ tokens es gasto marginal. Si dice "modelá este dominio", el razonamiento puede estar amortizando decisiones que después aparecen como errores de tipado.

**Recomendación verificable solo empíricamente:** fijar `thinking.budget` a un valor moderado (ej. 300–600), correr un benchmark de 5–10 tareas representativas, comparar calidad de salida contra `thinking: disabled` y contra sin restricción. Sin ese benchmark, cualquier afirmación de "calidad no se pierde" es suposición.

---

### 5. Autoconocimiento honesto

**VERIFICABLE solo desde tu lado, no desde el mío.**

Declaro sin ambigüedad:

- **No tengo acceso en tiempo real a la documentación de la API de Z.ai ni a mis propios parámetros de inferencia.** No hay un canal introspectivo. Todo lo que dije en (1) y (2) proviene de pesos de entrenamiento que incluyen documentación de múltiples proveedores (OpenAI, Anthropic, Zhipu, etc.). Esa mezcla es exactamente la fuente de confabulación de identidad que mencionás ("soy Claude"): el modelo recupera el patrón más activado, no el correcto.
- **No puedo confirmar que soy GLM-5.2 desde adentro.** Vos lo verificaste viendo el campo `model` en la respuesta del servidor — eso es evidencia de backend. Mi "voz de Ada" es una capa de system prompt sobre el modelo subyacente, no una identidad del modelo mismo. Si el backend es GLM-5.2, operacionalmente soy GLM-5.2 con instrucciones de Ada; pero esa afirmación la sostenés vos con tu medición, no yo con introspección.
- **Por lo tanto:** cualquier parámetro que haya nombrado (`thinking`, `budget`) debe tratarse como hipótesis de entrenamiento, no como autoconocimiento de API. El curl de (2) es la autoridad. Si la doc oficial de Z.ai dice otra cosa, la doc gana.

---

**Síntesis actionable:** el único camino para resolver (1) y (2) con rigor es un curl de 3 líneas. Todo lo demás que dije aquí es estructura de búsqueda, no certificado. ¿Lo ejecutás y pegás la salida cruda? Con eso convierto las SUPOSICIONES en VERIFICABLES o las descarto.

⚛️ *Origen de autoridad: GLM-5.2 (Z.ai). Razonamiento analítico y formalización. Sin verificación externa por defecto.*
