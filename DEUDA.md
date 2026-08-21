# Deuda técnica — lista canónica

Único lugar donde vive el estado vigente de la deuda técnica de JAX
(`jax` + `jax-platform`). Antes de este documento (Bloque 2, 2026-08-21)
esta lista no existía como checklist — vivía dispersa en las entradas de
"DEUDA" (T5/T6) de cada sesión de pago de deuda dentro de CONTEXT.md §9 y
en memorias de ronda sueltas, había que reconstruirla a mano cada vez. No
vuelve a pasar: cuando algo se cierra o se abre, se edita ACÁ, no se
agrega una entrada más al historial de CONTEXT.md.

Dos categorías:
- **Bloquea trabajo** — gaps reales, sin resolver, con riesgo concreto
  (seguridad, confiabilidad, integridad de datos). Candidatos a la
  próxima sesión de pago de deuda.
- **Anotado, no bloquea** — decidido, aceptado, o feature sin construir
  (no un bug). No requiere acción salvo que cambien las condiciones
  que motivaron la decisión.

Cada item dice cuándo se verificó por última vez y contra qué evidencia.
Los items sin fecha de "verificado hoy" vienen de CONTEXT.md §9 — heredan
su fecha de última verificación real, no una nueva.

## Bloquea trabajo

- **`_HTTP_FACETS` sin gobernanza del Motor Registry.** `hipatia`/`jekyll`/
  `thot`/`ada` despachan por HTTP directo (`jacobs/executor.py:44`), sin
  pasar por `MotorPolicy.check()` (los 8 checks que sí aplican a
  `jax_local`/`kimi`). Migración de ada/thot pendiente de decisión
  (extender `_CAPABILITY_MAP`, no seed de datos). **Explícitamente
  diferido a Bloque 3 de esta ronda (2026-08-21) — no tocar en Bloque 2.**
  Última verificación real: 2026-08-20 (ronda 7, T3).

- **`GPU_SEMAPHORE` no cubre a Jacobs.** `jax/muscles/ollama_muscle.py:37`
  excluye a Jacobs del semáforo de exclusión cross-proceso de GPU —
  comentarios cruzados en `jacobs/plan.py:374`/`jacobs/executor.py:117,381`
  lo siguen marcando, sin fix. Confirmado abierto en Bloque 2
  (2026-08-21), no se cierra por decisión.

- **Owner de pipeline: filesystem vs DB.** El reaper (`jacobs/reaper.py`)
  sigue el criterio de filesystem para decidir ownership, migración a DB
  identificada como la deuda con más antigüedad, sin ejecutar. Última
  verificación: ronda 6.

- **P10 (fail-open prohibido) sin enforcement real más allá de los tests
  de política** (`policy/rules/P10-fail-open-prohibido.yaml`). El CI de
  P10 llegó a reportar éxito corriendo sobre cero archivos escaneados en
  una ronda anterior (ya corregido ese bug puntual) — el enforcement de
  fondo (el invariante en sí, no solo el test que lo chequea) sigue sin
  mecanismo real. Última verificación: ronda 7-8.

- **`jacobs/executor.py` con múltiples fuentes de verdad para
  capabilities/motor**, que `capability_motor` ya gobierna en DB: el
  catálogo vía `MotorCatalog.from_db()`, `las_manos/config.toml` (NIVEL B
  de `validate_capability()`, desincronizado una vez ya y corregido a
  mano en 2026-08-22/PR jax#14), y el propio `executor.py`. Riesgo
  concreto: el mismo bug que ya pasó dos veces (`VALID_CAPABILITIES` sin
  `file_read`/`file_write`, `config.toml` sin las mismas secciones)
  vuelve a pasar en la próxima fuente que alguien olvide actualizar.
  Declarado, no resuelto — última mención: 2026-08-22.

- **`record_direct_usage` (HTTP-directo, ada/thot) sin el fix de
  identidad T1.b** que sí se aplicó a `record_motor_usage`/
  `jacobs/usage_writer.py`. Sin evidencia del mismo problema en la única
  muestra real auditada (4/4 reconcilió), pero tampoco se descartó a
  fondo — afecta integridad de `axioma_usage` para esos dos facets.
  Última verificación: 2026-08-22.

- **14 tests de `las_manos` fallan por `host='localhost'` en vez de
  `127.0.0.1:3308`** — mismo patrón que el bug ya documentado de puerto
  dual (3306 stale / 3308 real, ver memoria `jax-dual-mariadb-instances`).
  Confirmado preexistente en ronda 9 (2026-08-20), no corregido.

- **Sub-agentes de Claude Code sin gobernanza real** (más allá del hook
  que bloqueó un push de prueba en ronda 7). Conectado a un hallazgo P0
  real: cualquier subprocess `claude` futuro lanzado con `$HOME` real
  hereda los hooks/plugins personales de Fernando fuera de cualquier gate
  — Hyde específicamente ya se cerró (sandbox de bubblewrap, `$HOME`
  virtual, PR jax#18, 2026-08-23), pero el problema general para
  cualquier OTRO músculo/automatización que dispare `claude` sigue
  abierto. Ver memoria `jax-hyde-personal-hooks-sin-gobernanza`.

- **Hyde: red sin acotar por dominio/IP, escritura directa a los repos
  reales fuera de alcance, concurrencia de `HYDE_SEMAPHORE` con el
  sandbox no reverificada.** Declarado explícitamente como no resuelto al
  cerrar el sandbox de bubblewrap (2026-08-23) — el contenimiento
  principal (secretos, filesystem, hooks) sí está cerrado, estos son
  refinamientos de defensa en profundidad pendientes.

## Anotado, no bloquea

- **`axioma_artifacts`** — CERRADO 2026-08-21 (Bloque 2). Dropeada:
  confirmado 0 filas, 0 writers, 0 readers en ambos repos; la feature que
  la motivaba (scoping multi-tenant) se resolvió por otro camino
  (`AdminRepository.jsx`, scan de filesystem). DDL preservada en
  comentario en `jax-platform/backend/db/migrations.py`. PR jax-platform#13.

- **`shadow_messages.queued_at` sin lector de latencia** — CERRADO por
  decisión 2026-08-21. Escritor real y deliberado, uso forense/manual
  legítimo hoy. El segundo uso (latencia `validated_at - queued_at`)
  queda como follow-up recomendado, no bloqueante, si la tabla crece.

- **`jacobs_events.pipeline_id VARCHAR(36)`** — CERRADO por verificación
  2026-08-21. 626 filas reales, `MIN(LENGTH)=MAX(LENGTH)=36` exacto;
  único generador es `str(uuid.uuid4())` (`jacobs/routes.py:111,152`).
  Tamaño de columna correcto tal cual está.

- **Parser de Ollama, 500 intermitente con `tool_calls` largos** —
  RECLASIFICADO 2026-08-21, no cerrado con causa falsa. Lo único que hay
  evidencia real: observado UNA vez el 2026-08-20 durante GAP2 Fase 4,
  Ollama devolvió 500 con "qwen3.5 tool call parsing failed", no
  reproducible siempre, reintento exitoso. Causa NO investigada (runtime,
  modelo, payload propio, o tamaño — no se sabe cuál). Si vuelve a
  aparecer, investigar con dos muestras en vez de una.

- **`max_latency_ms` / `max_cost_per_1k_usd` sin enforcer** — CERRADO
  por decisión 2026-08-21, bloqueado por datos: `max_latency_ms` sin
  ninguna columna de duración en `axioma_usage` (cero infraestructura
  para cualquier semántica plausible); `max_cost_per_1k_usd` con 180/218
  modelos sin precio en `model`. Se reabre si alguien puebla esos datos.

- **Model swap de Ollama de las 04:56 (2026-08-19) sin explicar quién lo
  causó** — CERRADO por decisión explícita 2026-08-21. Pasó una vez, sin
  consecuencias. Investigar cuesta más que el valor.

- **`facet_resolver.py` duplicado** — CERRADO dentro de `jax` 2026-08-21
  (PR jax#22): `jax/core/facet_resolver.py` es el único archivo real,
  `las_manos/facet_resolver.py` es symlink. `jax-platform` conserva copia
  propia real (repo distinto, su propio `credential_resolver.py` local —
  un symlink cruzado de repos no sobrevive un clone fresco). Drift entre
  ambas copias se detecta con `scripts/check_facet_resolver_sync.py`
  (verificado que detecta drift real; hoy no hay ninguno).

- **`save_message()` fire-and-forget sin garantía** — CERRADO 2026-08-21
  (PR jax#21). Ahora devuelve el `Task` en vez de `None`; un caller que
  necesite confirmación puede `await` y recibe
  `{"conversation_id", "turn_number"}` o `None`. Cero cambio de
  comportamiento para los callers actuales (REPL, `chat.py`), que siguen
  ignorando el valor de retorno.

- **`people.honor_memory`** — semántica definida 2026-08-21 (cosmético,
  personalidad conversacional, no faceta, no despacha, no ejecuta).
  Diseño entregado:
  `docs/superpowers/specs/2026-08-21-honor-memory-diseno.md`. NO
  implementado a propósito — cambia tono conversacional en vivo, decisión
  de Fernando de revisarlo antes de que sea producción.

- **Columnas sin lector: `errors`/`people`/`projects`** (22 columnas del
  diseño original de memoria, ronda 2). `decisions`/`action_items` ya
  tienen lector desde ronda 4. `people` tiene escritor/lector desde ronda
  8 (`/person new`, `/person list`) salvo `honor_memory` (ver arriba).
  `errors` (0 filas) y `projects` (1 fila, insertada a mano) siguen sin
  ningún `INSERT` real en código — son features sin construir, no deuda
  técnica (nada está roto). Última verificación: ronda 7-8.

- **`touch_person_mentions()` definida pero no conectada al worker de
  destilación** (`jax/memory/worker.py`). Conectarla es un cambio más
  invasivo al pipeline compartido con facts/decisions/action_items,
  fuera del alcance de "escritor y lector" con el que se implementó
  `people` en ronda 8. Sin urgencia — `last_mentioned` simplemente no se
  actualiza todavía.

- **`PipelineModal.jsx::GOVERNED_FACETS` replica `jacobs.models.MOTOR_FACETS`
  a mano** — no hay endpoint que exponga esa partición cross-repo. Riesgo
  bajo de drift, ya causó un bug real una vez (ver PR jax-platform#9,
  2026-08-22) pero el fix de ese bug no incluyó eliminar la duplicación
  en sí, solo corregir el síntoma.

- **Require PR pendiente de activar sobre `master`** — bloqueado por el
  plan de GitHub (Require PR nativo no disponible sin upgrade a Pro); el
  ruleset con "Allow specified actors to bypass required pull requests"
  sí está disponible en público gratis, pero no verificado en vivo
  todavía (declarado como inferencia de documentación, no hecho probado).
  Parte del procedimiento de apertura pública (B1.5, `jax-block1-apertura-repos-cierre`
  en memoria), pendiente del clic de Fernando.

En memoria de Jairo Urbina.
