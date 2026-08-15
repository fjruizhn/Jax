# Mejoras de Memoria JAX (roadmap Beelink, 8 items) — Plan de Implementación

> **Para ejecución:** superpowers:executing-plans (inline, en esta misma sesión —
> ya tengo cargado el contexto completo de worker.py/db.py/chat.py/main.py, no
> hace falta re-descubrirlo en subagentes frescos). Checkboxes por tarea.

**Goal:** Portar las 8 mejoras evaluadas en la comparación JAX-vs-Beelink al
sistema de memoria real de JAX (`~/jax/jax/memory/{worker.py,db.py}` +
`~/jax-platform/backend/api/chat.py` + `~/jax/jax/core/main.py`).

**Architecture:** Todas las mejoras son incrementales sobre el pipeline
existente (extracción batch DeepSeek → `facts`/`decisions`/`action_items` con
`is_verified=FALSE` → retrieval vectorial `VEC_DISTANCE_COSINE`). Nada
reemplaza el modelo actual de scope de dos niveles (`user_id`/`project_id`) ni
el gate de revisión humana — las mejoras se insertan como pasos adicionales,
no como reemplazos.

**Tech Stack:** Python 3.14 async, aiomysql, MariaDB 11.8 VECTOR(768),
Ollama (`nomic-embed-text` para embeddings), DeepSeek API (extractor batch).

**Spec:** Comparación "JAX vs. Beelink.html" + "Mejoras para JAX.html"
(Google Drive, carpeta "Para Claude"; sketch original en memoria
`beelink-ai-dashboard-memory-system.md`, sesión `5aa29340` 2026-08-14/15).

## Global Constraints (CLAUDE.md — políticas del ecosistema)

- Backup obligatorio antes de modificar cualquier archivo existente
  (`*.backup-pre-<cambio>-<timestamp>`).
- `py_compile` en TODOS los .py modificados antes de dar la tarea por cerrada.
- Credenciales SOLO desde `/etc/jax/.env` — nunca hardcodeadas.
- Nada entra como verdad sin revisión humana: cualquier fact nuevo (incluida
  síntesis de item #8) se guarda con `is_verified=FALSE`.
- Fail-safe por defecto: ante incertidumbre (sin embedding, sin candidato
  claro, distancia ambigua) el sistema debe preferir INSERT sobre UPDATE, y
  preferir NO actuar sobre actuar mal — igual que ya hace la extracción con
  las categorías prohibidas de salud/etc.
- Cambios de schema son ALTER TABLE aditivos (columnas nullable, sin
  DROP/RENAME) — reversibles con un DROP COLUMN si algo sale mal. Anuncio la
  sentencia SQL antes de correrla contra `jax_memory` en vivo.
- Los items #5 y #6 se implementan pero con las reservas que dejó la sesión
  anterior explícitas en el código (comentario + flag apagado por defecto en
  #6) — el usuario pidió "todo" pero eso no borra la evaluación de riesgo ya
  hecha, solo dice que se construye igual.

---

## File Structure

- **Modify** `jax/jax_memory_schema.sql` — documentar `facts.user_id`/`project_id`
  ya existentes en la DB viva (drift) + agregar columnas nuevas de items #1/#3.
- **Modify** `jax/jax/memory/db.py` — `_find_nearest_fact()` (helper compartido
  por corrección y dedup), `supersede_fact()`, `search_similar_messages()`
  (blend de historial), `get_facts()` (ya soporta `fact_type`, se le agrega
  detección de intención), nuevo `detect_completeness_intent()`, scoring de
  decay opcional, reranking opcional.
- **Modify** `jax/jax/memory/worker.py` — `EXTRACTION_PROMPT` gana
  `is_correction`, `process_one()` rutea a corrección/dedup, chunking de
  conversaciones largas.
- **Modify** `jax-platform/backend/api/chat.py` — `_semantic_context()` pasa
  historial reciente a `search_similar_messages`, agrega bypass de
  completeness.
- **Modify** `jax/jax/core/main.py` — mismo blend/bypass en el loop del REPL
  (paridad con chat.py, líneas ~554-573).
- **Create** `jax/jax/memory/synthesis_worker.py` — job batch de síntesis de
  segundo orden (item #8), systemd timer separado, opt-in.
- **Test** `jax/tests/test_memory_correction.py` — funciones puras de
  `_find_nearest_fact`/bandas de distancia (corrección vs dedup), sin DB real
  (mismo patrón que `tests/test_jacobs_director.py`: réplica aislada de la
  lógica de umbrales, sin mockear aiomysql).
- **Test** `jax/tests/test_completeness_intent.py` — `detect_completeness_intent()`
  puro, sin DB.

---

### Task 1: Migración de schema — columnas de corrección/dedup

**Files:**
- Modify: `jax/jax_memory_schema.sql`
- Ejecutar ALTER contra `jax_memory` en vivo (hall9000)

**Columnas nuevas en `facts`:**
- `superseded_by INT NULL` — si no-NULL, este fact fue reemplazado por una
  corrección; el id apunta al fact nuevo.
- `superseded_at TIMESTAMP NULL`.
- Índice `idx_facts_active (superseded_by)` para que las queries de retrieval
  filtren `WHERE superseded_by IS NULL` barato.

No se necesita columna de dedup separada: un duplicado detectado simplemente
NO se inserta (se descarta en `save_fact`), no hay fact "duplicado" que marcar.

- [ ] Backup: `mysqldump jax_memory facts > /home/fruiz/jax/backups/facts-pre-correccion-$(date +%Y%m%d%H%M%S).sql`
- [ ] Actualizar `jax_memory_schema.sql`: agregar `user_id INT NULL`,
      `project_id INT NULL` (documentar el drift ya existente en la tabla
      `facts` — hoy el .sql no los tiene pero la DB viva sí) y las dos
      columnas nuevas, con sus índices, en el `CREATE TABLE facts`.
- [ ] Anunciar y ejecutar contra la DB viva:
  ```sql
  ALTER TABLE facts
    ADD COLUMN superseded_by INT NULL,
    ADD COLUMN superseded_at TIMESTAMP NULL,
    ADD INDEX idx_facts_active (superseded_by);
  ```
- [ ] Verificar con `SHOW CREATE TABLE facts` que las columnas quedaron.

### Task 2: `_find_nearest_fact()` — helper compartido corrección + dedup

**Files:**
- Modify: `jax/jax/memory/db.py`
- Test: `jax/tests/test_memory_correction.py`

**Diseño:** una sola búsqueda por vecino más cercano, scoped por
`user_id`/`project_id` igual que `search_similar_messages`. El *caller*
decide qué hacer según la banda de distancia:

- `distancia < DUP_THRESHOLD (0.05)` → duplicado casi textual → no insertar.
- `DUP_THRESHOLD <= distancia < CORRECTION_THRESHOLD (0.25)` → mismo tema,
  contenido distinto → candidato a corrección.
- `distancia >= CORRECTION_THRESHOLD` → no relacionado → INSERT normal.

Umbrales como constantes de módulo (no hardcodeados en línea, fáciles de
tunear si en producción generan falsos positivos).

- [ ] Backup: `cp jax/jax/memory/db.py jax/jax/memory/db.py.backup-pre-correccion-$(date +%Y%m%d%H%M%S)`
- [ ] Escribir función pura de clasificación de banda (testeable sin DB):

```python
DUP_DISTANCE_THRESHOLD = 0.05
CORRECTION_DISTANCE_THRESHOLD = 0.25


def classify_fact_distance(distancia: float) -> str:
    """Clasifica una distancia coseno contra el fact mas cercano.
    Devuelve 'duplicate' | 'correction_candidate' | 'unrelated'."""
    if distancia < DUP_DISTANCE_THRESHOLD:
        return "duplicate"
    if distancia < CORRECTION_DISTANCE_THRESHOLD:
        return "correction_candidate"
    return "unrelated"
```

- [ ] Agregar a `MemoryDB`:

```python
async def _find_nearest_fact(self, embedding: list, user_id: Optional[int],
                             project_id: Optional[int]) -> Optional[dict]:
    """Busca el fact activo (superseded_by IS NULL) mas cercano al embedding
    dado, scoped por user_id/project_id. None si no hay pool, no hay
    embedding, o no hay ningun fact en ese scope (fail-safe: caller inserta)."""
    if not self.pool or not embedding:
        return None
    vec_str = json.dumps(embedding)
    clauses = ["superseded_by IS NULL"]
    params: list = []
    scope = []
    if project_id is not None:
        scope.append("project_id = %s")
        params.append(project_id)
    if user_id is not None:
        scope.append("(project_id IS NULL AND user_id = %s)")
        params.append(user_id)
    if scope:
        clauses.append("(" + " OR ".join(scope) + ")")
    where = " AND ".join(clauses)
    try:
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "SELECT id, fact_text, "
                    "VEC_DISTANCE_COSINE(embedding, VEC_FromText(%s)) AS distancia "
                    f"FROM facts WHERE {where} "
                    "ORDER BY VEC_DISTANCE_COSINE(embedding, VEC_FromText(%s)) ASC "
                    "LIMIT 1",
                    ([vec_str] + params + [vec_str]),
                )
                row = await cur.fetchone()
                return dict(row) if row else None
    except Exception as e:
        logger.error(f"_find_nearest_fact fallo: {e}")
        return None
```

- [ ] `jax/tests/test_memory_correction.py` — importar `classify_fact_distance`
      de `jax.memory.db` y testear las 3 bandas + los bordes exactos
      (0.05 y 0.25):

```python
from jax.memory.db import classify_fact_distance


def test_duplicate_band():
    assert classify_fact_distance(0.0) == "duplicate"
    assert classify_fact_distance(0.049) == "duplicate"


def test_correction_band():
    assert classify_fact_distance(0.05) == "correction_candidate"
    assert classify_fact_distance(0.24) == "correction_candidate"


def test_unrelated_band():
    assert classify_fact_distance(0.25) == "unrelated"
    assert classify_fact_distance(0.9) == "unrelated"
```

- [ ] Correr: `cd ~/jax && .venv/bin/python -m pytest tests/test_memory_correction.py -v`
- [ ] `py_compile jax/jax/memory/db.py`
- [ ] Commit.

### Task 3: `supersede_fact()` + dedup/corrección en `save_fact()`

**Files:** Modify `jax/jax/memory/db.py`

**Interfaces:**
- Consumes: `_find_nearest_fact`, `classify_fact_distance` (Task 2).
- Produces: `save_fact(..., is_correction: bool = False) -> Optional[bool]`
  (firma extendida, retrocompatible — default `False` no cambia el
  comportamiento de callers existentes que no pasan el flag).

- [ ] Agregar `supersede_fact`:

```python
@db_error_handler
async def supersede_fact(self, old_fact_id: int, new_fact_id: int) -> Optional[bool]:
    """Marca old_fact_id como reemplazado por new_fact_id. No borra nada
    (auditoria: se puede reconstruir la historia de una correccion)."""
    if not self.pool:
        return None
    async with self.pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE facts SET superseded_by = %s, superseded_at = NOW() "
                "WHERE id = %s",
                (new_fact_id, old_fact_id),
            )
    return True
```

- [ ] Modificar `save_fact` (mantiene la firma actual, agrega
      `is_correction: bool = False` al final) para que, ANTES del INSERT,
      calcule el embedding primero (hoy se calcula despues, fuera del bloque
      de conexion — hay que adelantarlo), busque el vecino mas cercano, y
      decida:
      - `duplicate` y `not is_correction` → loggear y devolver `False` (no
        insertar, no es un error).
      - `correction_candidate` y `is_correction` → INSERT normal del fact
        nuevo, luego `supersede_fact(candidato_id, nuevo_id)`.
      - cualquier otro caso (`unrelated`, o sin candidato, o `is_correction`
        pero banda `duplicate`/`unrelated`) → INSERT normal, fail-safe.
- [ ] `py_compile jax/jax/memory/db.py`
- [ ] Commit.

### Task 4: `worker.py` — extracción reconoce correcciones

**Files:** Modify `jax/jax/memory/worker.py`

- [ ] Backup: `cp jax/jax/memory/worker.py jax/jax/memory/worker.py.backup-pre-correccion-$(date +%Y%m%d%H%M%S)`
- [ ] En `EXTRACTION_PROMPT`, agregar antes del bloque `Categorias:` una
      sección nueva explicando `is_correction` con ejemplo concreto
      ("Fernando dijo que el modelo era X, ahora dice que es Y" →
      `is_correction: true`), y actualizar el JSON de salida esperado:
      `{"text": "...", "type": "...", "is_correction": false}`.
- [ ] En `process_one`, cambiar el loop de facts para pasar
      `is_correction=f.get("is_correction", False)` a `db.save_fact`.
- [ ] `py_compile jax/jax/memory/worker.py`
- [ ] Commit.

### Task 5: Blend de contexto reciente en retrieval (item #2)

**Files:**
- Modify: `jax/jax/memory/db.py` (`search_similar_messages`)
- Modify: `jax-platform/backend/api/chat.py` (`_semantic_context`)
- Modify: `jax/jax/core/main.py` (loop del REPL, ~línea 554)

**Diseño:** la lógica de blend vive en un solo lugar (`db.py`) para no
duplicarla entre chat.py y main.py — ambos ya tienen su `history`/`historial`
disponible en el punto de llamada.

- [ ] En `db.py`, agregar `search_similar_messages(..., recent_history:
      Optional[list[dict]] = None)`. Construye el texto de query como:

```python
def _blend_query(user_text: str, recent_history: Optional[list[dict]]) -> str:
    """Combina el mensaje actual con los ultimos 4 turnos (500 chars c/u)
    para que el embedding capture el hilo de la conversacion, no solo la
    ultima frase (que puede ser ambigua: 'y eso por que?')."""
    if not recent_history:
        return user_text
    turnos = recent_history[-4:]
    piezas = [t["content"][:500] for t in turnos if t.get("content")]
    piezas.append(user_text)
    return "\n".join(piezas)
```

  y usarla para calcular `embedding = await self.get_embedding(_blend_query(query, recent_history))`
  (el resto de la función no cambia — sigue devolviendo mensajes por
  distancia contra ESE embedding).
- [ ] En `chat.py`, `_semantic_context(user_text, user_id, project_id, history)`
      gana el parámetro `history` y lo pasa a `search_similar_messages`. En
      el call site (línea ~576), pasar `_conversations.get(str(user_id), [])`
      (o la clave que use `_update_history` — verificar el tipo exacto de
      clave antes de escribir esta línea).
- [ ] En `main.py`, pasar `historial` (ya está en scope en el loop) a
      `db.search_similar_messages(..., recent_history=historial)`.
- [ ] `py_compile` los tres archivos.
- [ ] Commit.

### Task 6: Category bypass para preguntas de completeness (item #4)

**Files:**
- Modify: `jax/jax/memory/db.py` (nuevo `detect_completeness_intent`)
- Modify: `jax-platform/backend/api/chat.py`
- Modify: `jax/jax/core/main.py`
- Test: `jax/tests/test_completeness_intent.py`

**Diseño:** preguntas tipo "que proyectos tenes activos" / "cuales son mis
decisiones tecnicas" no se responden bien con similitud vectorial contra UN
fact — necesitan TODOS los facts de una categoría. Detección por keyword
(mismo estilo que el router híbrido de JAX, no requiere LLM):

```python
_COMPLETENESS_PATTERNS = {
    "project": ("que proyectos", "cuales proyectos", "en que proyectos"),
    "preference": ("mis preferencias", "que preferis", "como te gusta que"),
    "technical": ("que decisiones tecnicas", "que elegimos"),
}


def detect_completeness_intent(text: str) -> Optional[str]:
    """Detecta si el texto es una pregunta de 'dame todo lo que sepas de X'
    en vez de una pregunta puntual. Devuelve el fact_type a barrer completo,
    o None si es una pregunta normal (retrieval semantico de siempre)."""
    normalizado = text.lower()
    for fact_type, patrones in _COMPLETENESS_PATTERNS.items():
        if any(p in normalizado for p in patrones):
            return fact_type
    return None
```

- [ ] Agregar la función (y `_COMPLETENESS_PATTERNS`) a `db.py`.
- [ ] `jax/tests/test_completeness_intent.py`:

```python
from jax.memory.db import detect_completeness_intent


def test_detects_project_completeness():
    assert detect_completeness_intent("Que proyectos tenes activos?") == "project"


def test_normal_question_returns_none():
    assert detect_completeness_intent("Como se llama el gato de Fernando?") is None
```

- [ ] En `chat.py` y `main.py`, en el punto donde se arma `semantic_context`/
      `history_for_invocation`: si `detect_completeness_intent(user_text)`
      devuelve un `fact_type`, llamar `db.get_facts(only_unverified=False,
      fact_type=tipo, user_id=..., project_id=..., limit=20)` y agregar esos
      facts como contexto ADICIONAL (no en reemplazo del retrieval semántico
      normal — ambos aportan).
- [ ] `py_compile` todo, correr los dos tests nuevos.
- [ ] Commit.

### Task 7: Chunking de conversaciones largas (item #5)

**Files:** Modify `jax/jax/memory/worker.py`

**Nota de riesgo (heredada del roadmap):** el bug original de Beelink era de
corrupción/relevancia en SU hardware (Qwen local + AMD ROCm) — no confirmado
que aplique a JAX, que extrae con DeepSeek API (no local). Lo que SÍ es un
gap real e independiente de esa duda: `process_one` mete `conv_text` entero
en el prompt sin límite, y una conversación larga puede exceder el contexto
del extractor o inflar costo/latencia sin necesidad.

- [ ] En `worker.py`, agregar constante `MAX_CHARS_PER_EXTRACTION = 12000`
      (~3000 tokens, conservador para dejar espacio al prompt fijo) y una
      función pura:

```python
def _chunk_conversation(conv_text: str, max_chars: int = MAX_CHARS_PER_EXTRACTION) -> list[str]:
    """Parte conv_text en trozos de hasta max_chars, cortando en limites de
    linea (nunca a mitad de un mensaje). Conversaciones cortas: un solo
    trozo (comportamiento actual sin cambios)."""
    if len(conv_text) <= max_chars:
        return [conv_text]
    lineas = conv_text.split("\n")
    chunks, actual = [], []
    largo = 0
    for linea in lineas:
        if largo + len(linea) + 1 > max_chars and actual:
            chunks.append("\n".join(actual))
            actual, largo = [], 0
        actual.append(linea)
        largo += len(linea) + 1
    if actual:
        chunks.append("\n".join(actual))
    return chunks
```

- [ ] En `process_one`, reemplazar la única llamada al extractor por un loop
      sobre `_chunk_conversation(conv_text)`, mergeando los `facts`/
      `decisions`/`action_items` de cada chunk antes de guardar (el `save_fact`
      con dedup del Task 3 ya evita que un mismo hecho repetido entre chunks
      se duplique).
- [ ] Test puro en `test_memory_correction.py` o archivo nuevo
      `test_chunking.py`: conversación de 3 líneas no se parte; conversación
      sintética de 20000 chars se parte en 2+ trozos y el join de todos los
      trozos reconstruye el texto original línea por línea.
- [ ] `py_compile jax/jax/memory/worker.py`.
- [ ] Commit.

### Task 8: Time-decay opcional, apagado por defecto (item #6)

**Files:** Modify `jax/jax/memory/db.py`

**Nota de riesgo (heredada del roadmap, EXPLÍCITA en el código):** la sesión
anterior rechazó esto porque a la escala actual de JAX (~100 facts) no hace
falta y en Beelink causó una falla silenciosa de retrieval por horas. Se
construye porque el usuario lo pidió, pero con `DECAY_LAMBDA` en 0.0 por
defecto = decay matemáticamente inerte (multiplica por 1.0), para no
reintroducir esa fragilidad sin que alguien lo prenda a propósito.

- [ ] Agregar env var `JAX_MEMORY_DECAY_LAMBDA` (default `"0.0"`) leída en
      `search_similar_messages`/`get_facts` vía `float(os.getenv(...))`.
- [ ] Ajustar el `ORDER BY` de `search_similar_messages` para que, cuando
      `DECAY_LAMBDA > 0`, la distancia efectiva sea
      `distancia + DECAY_LAMBDA * DATEDIFF(NOW(), m.created_at)` en vez de
      la distancia cruda (mas viejo = mas "lejos"). Con `DECAY_LAMBDA=0.0`
      esto es una suma de cero: el SQL queda idéntico al de hoy.
- [ ] Documentar en un comentario arriba de la constante: por qué está en
      0.0, y bajo qué condición revisarlo (facts > ~1000 filas, ver
      memoria `beelink-ai-dashboard-memory-system.md`).
- [ ] `py_compile jax/jax/memory/db.py`.
- [ ] Commit.

### Task 9: Reranking opcional con cross-encoder (item #7)

**Files:**
- Modify: `jax/jax/memory/db.py`
- Modify: `jax/requirements.txt` (o el archivo de deps que use `.venv`)

**Nota:** agrega `sentence-transformers` (trae `torch`) como dependencia
OPCIONAL — el import va dentro de una función, con try/except, para que si
no está instalado el sistema siga funcionando exactamente como hoy (fail-safe
de disponibilidad, no solo de dato).

- [ ] Confirmar con el usuario ANTES de instalar el paquete en el venv de
      producción (`~/jax/.venv`) — es la única acción de esta tarea que toca
      algo más allá de código propio (instala un paquete de terceros con
      peso considerable, ~500MB+ con torch).
- [ ] Si se aprueba: `~/jax/.venv/bin/pip install sentence-transformers`.
- [ ] En `db.py`:

```python
_reranker = None

def _get_reranker():
    """Carga el cross-encoder de forma perezosa. Si no esta instalado el
    paquete, devuelve None y el caller sigue sin reranking (no rompe)."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except ImportError:
            _reranker = False
    return _reranker or None
```

  y en `search_similar_messages`, si `os.getenv("JAX_MEMORY_RERANK") == "1"`
  y hay más de `limit` candidatos crudos, pedir `limit * 3` de la DB y
  rerankear con `_get_reranker().predict([(query, r["content"]) for r in
  candidatos])`, devolviendo los `limit` mejores por score del cross-encoder
  en vez de por distancia coseno cruda.
- [ ] `py_compile jax/jax/memory/db.py`.
- [ ] Commit.

### Task 10: Síntesis de segundo orden (item #8) — batch job separado, opt-in

**Files:**
- Create: `jax/jax/memory/synthesis_worker.py`
- Modify: systemd (nuevo timer, NO se activa solo — se instala pero
  `systemctl enable` queda a decisión explícita del usuario)

**Nota de riesgo (la más alta de las 8, heredada del roadmap):** este es
literalmente el tipo de feature donde ocurrió el incidente de privacidad de
Beelink. Mitigaciones obligatorias, no opcionales:
- Solo lee facts con `is_verified = TRUE` (revisados por Fernando) — nunca
  sintetiza sobre ruido no confirmado.
- Reusa el mismo `EXTRACTION_PROMPT`-style filtro de categorías prohibidas
  (salud, etc.) ya shippeado en `worker.py` — se importa la lista de
  categorías prohibidas, no se reescribe.
- Los facts sintetizados se guardan con `fact_type` existente más apropiado
  y `is_verified=FALSE` — pasan por el mismo gate de revisión humana que
  cualquier otro fact, nunca se auto-verifican.
- Timer systemd separado del worker principal, con `systemctl enable`
  MANUAL — no se activa como parte de este despliegue.

- [ ] Crear `jax/jax/memory/synthesis_worker.py`, estructura análoga a
      `worker.py`: `build_synthesizer()` (mismo `HttpMuscle` DeepSeek),
      `SYNTHESIS_PROMPT` (recibe N facts verificados de un mismo scope, pide
      "que patron o insight de mas alto nivel conecta estos hechos, si lo
      hay" — devuelve JSON vacío si no hay nada, mismo espíritu estricto que
      `EXTRACTION_PROMPT`), `run_once()` que itera scopes con >= 5 facts
      verificados y llama `db.save_fact(..., source_facet="synthesis")`.
      Reutiliza `_parse_json` de `worker.py` (import, no copy-paste).
      Importa `NUNCA extraigas` de `EXTRACTION_PROMPT` por referencia
      (extraer esas líneas a una constante `FORBIDDEN_CATEGORIES_BLOCK` en
      `worker.py` que ambos archivos importan, en vez de duplicar el texto).
- [ ] Unidad systemd `jax-memory-synthesis.timer` +
      `jax-memory-synthesis.service`, calcado del patrón de
      `jax-las-manos`/`jax-platform` ya documentado en CLAUDE.md — instalar
      el archivo pero dejar deshabilitado (`systemctl daemon-reload` sí,
      `systemctl enable --now` NO).
- [ ] `py_compile jax/jax/memory/synthesis_worker.py`.
- [ ] Commit.

---

## Self-Review

**Cobertura del roadmap:** #1 correccion=Task2-4, #2 blend=Task5, #3
dedup=Task2-3 (misma banda de distancia), #4 completeness=Task6, #5
chunking=Task7, #6 decay=Task8 (inerte por defecto), #7 rerank=Task9
(opt-in, requiere OK explícito para instalar dependencia), #8
synthesis=Task10 (opt-in, requiere `systemctl enable` manual). Los 8 items
cubiertos.

**Orden de dependencia:** Task1 (schema) bloquea Task2-3. Task2-3 deben ir
juntas (mismo motivo que el roadmap: "hacer junto con #1, no antes" para
dedup). Task5-10 son independientes entre sí y de Task2-4 salvo Task7/10 que
necesitan confirmación explícita del usuario antes de instalar
dependencias/activar timers — se ejecutan al final por eso, no por
dificultad técnica.
