faceta: hyde

# JAX — Pipeline de Memoria Contextual Real con Embeddings

## CONTEXTO

La memoria de JAX tiene la infraestructura correcta pero le falta el pipeline de embeddings:

- MariaDB 11.8 en hall9000, base `jax_memory`
- Tabla `messages` ya tiene columna `embedding vector(768)` — actualmente NULL en todos los registros
- Tabla `facts` también tiene columna `embedding` — actualmente NULL
- Modelo de embeddings: `nomic-embed-text` via Ollama local (localhost:11434), produce exactamente 768 dimensiones — VERIFICADO
- 567 mensajes históricos sin vectorizar
- JAX ya tiene `~/jax/jax/memory/db.py` con toda la lógica de DB

## LO QUE HAY QUE CONSTRUIR

### Pieza 1 — Función de embedding en db.py

Agregar un método `get_embedding(text: str) -> list[float]` que:
- Llama a `http://localhost:11434/api/embeddings` con model `nomic-embed-text`
- Devuelve el vector como lista de 768 floats
- Es tolerante a fallos — si falla, devuelve None sin romper nada

### Pieza 2 — Worker de embeddings (archivo nuevo)

Crear `~/jax/jax/memory/embedding_worker.py` que:
- Se ejecuta como script independiente: `python -m jax.memory.embedding_worker`
- Procesa todos los mensajes con `embedding IS NULL` en lotes de 50
- Para cada mensaje llama a `get_embedding(content)` y actualiza la columna
- Hace lo mismo con la tabla `facts`
- Muestra progreso: "Procesando mensaje 45/567..."
- Tolerante a fallos — si un mensaje falla, lo loguea y continúa con el siguiente
- Al terminar reporta cuántos procesó y cuántos fallaron

### Pieza 3 — Búsqueda semántica en db.py

Agregar método `search_similar_messages(query: str, limit: int = 5) -> list[dict]` que:
- Vectoriza la query con `get_embedding(query)`
- Ejecuta búsqueda vectorial en MariaDB:
  ```sql
  SELECT m.content, m.role, m.created_at, c.started_at,
         VEC_DISTANCE(m.embedding, VEC_FromText(?)) as distancia
  FROM messages m
  JOIN conversations c ON m.conversation_id = c.id
  WHERE m.embedding IS NOT NULL
  ORDER BY distancia ASC
  LIMIT ?
  ```
- Devuelve lista de dicts con content, role, fecha y distancia
- Si no hay embeddings o falla, devuelve lista vacía (nunca None)

### Pieza 4 — Inyección semántica en main.py

Reemplazar el bloque actual de historial de sesión anterior (líneas ~329-338) por:
- Al arrancar JAX, vectorizar el primer mensaje del usuario (no disponible aún)
- En el loop del REPL, ANTES de invocar el músculo:
  1. Vectorizar el `user_text` actual
  2. Llamar a `search_similar_messages(user_text, limit=5)`
  3. Si hay resultados relevantes (distancia < 0.8), construir un string:
     ```
     Conversaciones relevantes de sesiones anteriores:
     [fecha] user: contenido
     [fecha] jax: respuesta
     ...
     ```
  4. Agregar ese contexto al historial temporal SOLO para esta invocación (no al historial permanente de la sesión)

## RESTRICCIONES

- Leer db.py y main.py completos antes de tocar nada
- Mostrar cada pieza como diff antes de aplicar — esperar confirmación
- El sistema debe ser completamente tolerante a fallos: si Ollama no responde o MariaDB falla, JAX sigue funcionando sin embeddings
- No romper nada de lo que ya funciona (guardado de mensajes, facts, historial de sesión)
- Pieza 1 y 2 primero — luego verificar que el worker procesa los 567 mensajes correctamente — luego Pieza 3 y 4
- Código limpio, comentado en español, consistente con el estilo de db.py

## VERIFICACIÓN ESPERADA

Después de implementar todo:
1. Correr el worker: `cd ~/jax && .venv/bin/python -m jax.memory.embedding_worker`
2. Verificar en DB: `SELECT COUNT(*) FROM messages WHERE embedding IS NOT NULL;` debe ser 567
3. Abrir JAX y preguntar sobre caperucita — debe traer contexto de la sesión donde se habló del cuento
4. Preguntar "qué sabes de mí" — debe traer facts Y conversaciones relevantes anteriores
