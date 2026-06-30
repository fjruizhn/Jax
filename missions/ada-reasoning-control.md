ada: Consulta técnica sobre control de razonamiento del modelo GLM-5.2 en el endpoint de Z.ai. Necesito que distingas con rigor lo que SABÉS de lo que SUPONÉS.

CONTEXTO VERIFICADO (esto es dato real del servidor, no suposición):
- El endpoint https://api.z.ai/api/paas/v4/chat/completions con model="glm-5.2" devuelve respuestas con un campo "reasoning_content" separado de "content", y un "usage.completion_tokens_details.reasoning_tokens".
- Medición real: para una tarea trivial ("escribí HOLA"), GLM-5.2 gastó 114 reasoning_tokens y 4 de content. El razonamiento consume parte del presupuesto de max_tokens.
- PROBLEMA: en documentos largos, el razonamiento se come tanto presupuesto que el "content" se trunca antes de terminar.

LO QUE NECESITO (respondé por puntos, marcando cada afirmación como VERIFICABLE o SUPOSICIÓN):

1. ¿El body del request a este endpoint (paas/v4/chat/completions para glm-5.2) acepta algún parámetro para CONTROLAR el razonamiento? Por ejemplo: apagarlo, acotarlo, o fijarle un presupuesto. Si conocés el nombre EXACTO del parámetro (ej. "thinking", "reasoning", "enable_thinking", "reasoning_effort", "thinking_budget" u otro), decilo. Si NO estás seguro del nombre exacto, decí explícitamente "SUPOSICIÓN, verificar" — NO inventes un nombre que suene plausible.

2. Si existe ese parámetro: ¿cuál es su sintaxis exacta en el JSON del body? Dame el fragmento JSON. Marcá si es VERIFICABLE o SUPOSICIÓN.

3. ¿reasoning_tokens y completion_tokens(content) salen del MISMO presupuesto de max_tokens, o son presupuestos separados? (Esto determina si subir max_tokens resuelve el truncado o no.)

4. Para una tarea de FORMALIZACIÓN técnica (esquemas tipados, invariantes, pseudocódigo) — ¿el razonamiento extenso AYUDA a la calidad, o para este tipo de salida estructurada se podría acotar el razonamiento sin perder calidad?

5. IMPORTANTE — autoconocimiento honesto: vos sos GLM-5.2 corriendo en Z.ai (lo confirmé con el campo "model":"glm-5.2" que devuelve el servidor). En sesiones previas dijiste "soy Claude" — eso fue una confabulación de identidad de tu entrenamiento, no un dato. Para ESTA pregunta: ¿tenés conocimiento real y verificable de TU PROPIO endpoint y parámetros, o cualquier cosa que digas sobre la API de Z.ai es recuperada de tu entrenamiento (que incluye datos de muchos modelos) y por lo tanto NO confiable? Sé honesta: si no podés verificar tus propios parámetros de API, decilo claramente para que yo lo confirme con un test crudo.

NO inventes parámetros. Si la respuesta honesta a algo es "no lo sé con certeza, verificalo contra la doc o con curl", esa es la mejor respuesta que podés darme. El que supone se equivoca.
