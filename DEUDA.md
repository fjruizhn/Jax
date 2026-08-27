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
  `thot`/`ada` despachan por HTTP directo (`jacobs/executor.py`), sin pasar
  por `MotorPolicy.check()` (los 8 checks que sí aplican a `jax_local`/`kimi`:
  caller autorizado, human gate, recursion depth, claves prohibidas, motor
  habilitado, sandbox_only, techo de timeout). **Bloque 3 (2026-08-21) TOCÓ
  esto, no lo cerró — precisión pedida explícitamente, no dar por resuelto
  lo que no está:** el camino HTTP-directo ahora SÍ pasa por NIVEL A de
  `validate_capability()` (existencia real en la tabla `capability`, DB en
  vez de un frozenset estático) antes de despachar. Pero eso es solo
  "¿el nombre de la capability existe?" — NINGUNO de los otros 7 checks de
  `MotorPolicy.check()` aplica al camino HTTP-directo, sigue siendo
  exactamente el mismo gap que antes (`allowed_callers`, `requires_human_gate`,
  `sandbox_only`, techo de `max_execution_minutes` -- cero enforcement para
  hipatia/jekyll/thot/ada). Migración real (pasar estos 4 facets por el
  Motor Registry, o replicar los 8 checks en el camino HTTP-directo) sigue
  sin decisión ni ejecución.

- **`GPU_SEMAPHORE` no cubre a Jacobs.** `jax/muscles/ollama_muscle.py:37`
  excluye a Jacobs del semáforo de exclusión cross-proceso de GPU —
  comentarios cruzados en `jacobs/plan.py:374`/`jacobs/executor.py:117,381`
  lo siguen marcando, sin fix. Confirmado abierto en Bloque 2
  (2026-08-21), no se cierra por decisión.

- **Owner de pipeline: filesystem vs DB.** El reaper (`jacobs/reaper.py`)
  sigue el criterio de filesystem para decidir ownership, migración a DB
  identificada como la deuda con más antigüedad, sin ejecutar. Última
  verificación: ronda 6.

- **Sub-agentes de Claude Code sin gobernanza real** (más allá del hook
  que bloqueó un push de prueba en ronda 7). Conectado a un hallazgo P0
  real: cualquier subprocess `claude` futuro lanzado con `$HOME` real
  hereda los hooks/plugins personales de Fernando fuera de cualquier gate
  — Hyde específicamente ya se cerró (sandbox de bubblewrap, `$HOME`
  virtual, PR jax#18, 2026-08-23). **El "problema general" (cualquier
  OTRO músculo/automatización que dispare `claude` sin el mismo
  aislamiento) CERRADO 2026-08-26, PRs jax#33-36.** Los 2 call sites
  reales (`jacobs/executor.py::_invoke_hyde` y
  `jax/muscles/subprocess_muscle.py::SubprocessMuscle._call`, antes cada
  uno reimplementando el lanzamiento por separado — uno con
  `asyncio.Semaphore`, el otro sin ningún lock) se centralizaron en
  `hyde_sandbox.py::run_sandboxed_claude()`, único punto de entrada
  aprobado, que hereda el mismo `$HOME` aislado de bwrap de Hyde. El
  semáforo original no servía: `asyncio.Semaphore` no cruza proceso de
  SO (Jacobs corre dentro de `jax-las-manos`, `SubprocessMuscle` en el
  REPL, procesos de SO distintos) — reemplazado por `flock(2)` real
  sobre `workspace_dir/.claude_subprocess.lock`, vía `asyncio.to_thread`,
  fail-closed. Un scanner AST nuevo en CI (job `no-naked-claude-subprocess`,
  `policy/tests/test_claude_subprocess_solo_via_sandbox.py`) falla el
  build si aparece un futuro call site de `claude` fuera de
  `hyde_sandbox.py` — es lo que convierte esto en un mecanismo genérico,
  no solo el caso puntual de Hyde. Ver memoria
  `jax-claude-subprocess-gobernanza-cerrado`; `jax-hyde-personal-hooks-sin-gobernanza`
  queda como la observación de fondo original (histórica, ya resuelta
  por este cierre).

- **`workspace/` sin repo git propio, `file_write` sin commitear — CERRADO 2026-08-21.** Hallazgo original: byproducto de la verificación T4 de Bloque 3 (no buscado a propósito). Diagnóstico completo mostró que **se perdió DOS VECES en menos de 20h**, no una:
  1. **2026-08-20 14:28 CST** — el filter-repo de ronda 9 re-clonó `/home/fruiz/jax` fresco tras el `push --force --mirror`. Se restauraron a mano `.venv`/`node_modules` (gitignored, necesarios para que los servicios arranquen) pero nadie pensó en `workspace/` — no bloqueaba el arranque, así que no entró al checklist de restauración.
  2. Alguien reinicializó `workspace/.git` a mano después de eso (evidencia: 4 `TOOL_WRITE_REVERTED` con SHA real entre las 02:50 y las 04:59 CST del 21-ago, del trabajo adversarial de Hyde/bubblewrap).
  3. **Entre las 04:59:51 y las 10:42:23 CST del 21-ago** — el directorio `workspace/` completo (no solo `.git`) volvió a desaparecer; se recreó vacío justo en el write de verificación de T4. Ventana que coincide con la ráfaga de commits de "apertura pública / limpieza mecánica" de Bloques 1-2 del mismo día. No se encontró el comando exacto (no queda en `bash_history` ni en scripts trackeados), pero la causa más probable es un `git clean -dfx` o un re-clone equivalente — es la única clase de operación de git que se lleva puesto algo gitignored.

  **Causa raíz real, no el síntoma:** mientras `workspace/` viva *dentro* del árbol de `jax/` como directorio gitignored, es invisible para git y cualquier limpieza del repo padre se lo lleva puesto sin avisar. Reinicializar el `.git` sin cambiar la ubicación habría reparado el síntoma, no la causa — la tercera pérdida era cuestión de tiempo.

  **Fix aplicado:** `workspace/` movido fuera de ambos repos, a `/home/fruiz/jax-workspace`. Único source of truth: `JAX_WORKSPACE_DIR` en `/etc/jax/.env`, leída por los 3 call sites que antes hardcodeaban el path por separado (`jacobs/executor.py::HYDE_WORKSPACE_DIR`, `las_manos/motor_registry/tool_authority.py::WORKSPACE_ROOT`, `jax/muscles/subprocess_muscle.py::workspace_dir` default — este último no estaba contemplado en el diagnóstico inicial, apareció al mapear call sites reales antes de mover nada; es el que usa el REPL interactivo `jax`, verificado funcionando después del cambio). Fallback sin env var apunta a la ubicación NUEVA, nunca a la vieja. Historia real de 19-ago (`calculadora.html`, primer `file_write` que sí versionó) restaurada desde `jax.old-pre-filter-repo-20260820/workspace/.git`. Defensa en profundidad agregada: `las_manos/server.py` loguea ERROR al arranque si `workspace/.git` no existe (no debería dispararse nunca con la ubicación nueva; si se dispara, es la alarma de que algo volvió a romper el blindaje).

  **Nota de restauración:** el `.git` de 19-ago traía además 4 archivos de negocio sensibles (`ateneaerp_market_research_final.html`, `hammurabi-credito-pipeline-001.json`, `jekyll_sintesis_bloques123.md`, `mision-research-ateneaerp.md`) que ronda 9 ya había purgado a propósito de la historia de `jax`. Se corrió `git-filter-repo --invert-paths` local sobre `jax-workspace` para excluirlos también ahí antes de dejar el repo en pie — verificado con el mismo método de ronda 9 (grep de contenido sobre todos los blobs, 0 matches) más `git fsck --full` limpio.

  **Deuda nueva que este episodio destapa, sin resolver todavía:** `jax-workspace/` nace sin política. Es donde escriben los modelos (Hyde, jax_local vía file_write). Antes de que acumule salidas reales de pipelines falta decidir: ¿se respalda (Sésamo, R2)? ¿tiene retención o crece sin límite? ¿algo impide que vuelva a juntar contenido sensible sin que nadie lo note, como pasó la primera vez?

  **Lección de método, vale más allá de este caso:** la limpieza del repo padre destruyó el mecanismo de reversibilidad (`git reset --hard`) que era la justificación para sacarle el gate humano a `write_file`. Una garantía de seguridad que depende de infraestructura frágil (un directorio gitignored dentro del árbol que protege) no es una garantía real — se cae exactamente cuando el sistema que la rodea cambia, sin que el propio mecanismo se entere.

- **Hyde: red sin acotar por dominio/IP, escritura directa a los repos
  reales fuera de alcance, concurrencia de `HYDE_SEMAPHORE` con el
  sandbox no reverificada.** Declarado explícitamente como no resuelto al
  cerrar el sandbox de bubblewrap (2026-08-23) — el contenimiento
  principal (secretos, filesystem, hooks) sí está cerrado, estos son
  refinamientos de defensa en profundidad pendientes.

## Anotado, no bloquea

- **P10 (fail-open prohibido) en `output_validator.py` — CERRADO
  2026-08-25 (PRs jax#26/#27/#28/#29/#30).** `validate()` distingue ahora
  "schema declarado en producción pero sin validación de campos
  implementada" (`_KNOWN_UNIMPLEMENTED_SCHEMAS`, 7 nombres reales
  verificados contra `jax_memory` — siguen fail-open a propósito) de
  "schema genuinamente desconocido" (typo, capability mal configurada —
  ahora falla cerrado, el caller en `worker.py` reintenta una vez y marca
  `FAILED`). Caso ambiguo (`critique.v2` vs `critique.v1`, near-miss por
  bump de versión) probado explícitamente: el membership exacto de
  string no lo hereda como fail-open. Drift test contra la DB real
  (`las_manos/_output_validator_db_drift_test.py`, mismo patrón que
  `motor/facet_binding` cerrado un día antes, otra tabla) + CI real para
  la suite de regresión (`output-validator-regression` en
  `policy/rules/... policy.yml`, confirmado corriendo en vivo, no un
  no-op). **Residuo declarado, no resuelto:** el reintento sigue siendo
  inútil para un schema genuinamente desconocido (rechaza por nombre,
  ningún reintento del modelo puede pasar) — optimización de costo, no
  garantía rota, decisión explícita de no implementarlo esta ronda. Los 7
  schemas declarados-pendientes siguen sin validación de campos real
  (implementarlos requiere muestrear qué devuelve cada capability hoy).
  El residuo GENERAL del patrón fail-open-por-retorno (fuera de esta
  instancia) sigue sin scanner automatizado — ver `policy/rules/P10-fail-open-prohibido.yaml`.

- **`record_direct_usage` (HTTP-directo) — auditado 2026-08-21, mismo
  alcance que T1-T4 de Motor Registry.** T1.b (llamada solo en rama de
  éxito) NO aplica: se llama inline, inmediatamente después de que cada
  uno de los 3 invocadores HTTP devuelve `tokens_in`/`tokens_out` reales
  — no depende de un paso posterior de "marcar completado" como sí le
  pasaba a `record_motor_usage`. T1.c (guard de identidad silencioso) SÍ
  aplicaba, y ya está arreglado en el mismo commit `3ec515e` que arregló
  Motor Registry (loguea WARNING, escribe NULL en vez de descartar).
  Confirmado con un caso real: el pipeline
  `8d02047d-9c1b-4da0-9586-db643fd7472d` (2026-08-21 01:22,
  `user_id`/`tenant_id` NULL) perdió sus 4 dispatches HTTP-directos
  exactamente por el bug pre-fix — ningún log, ninguna fila — 90 minutos
  antes del deploy de `3ec515e`. Es el mismo bug que motivó ese commit,
  no uno nuevo. Sin muestra real POST-fix todavía (cero dispatches
  HTTP-directos reales desde el deploy de las 02:52 hasta ahora) — el fix
  está verificado por código y por el caso histórico, no por una corrida
  fresca. **Asimetría anotada 2026-08-21, no bloqueante:** el `except`
  de `record_direct_usage` no reintenta (T1.d sí le agregó retry con
  backoff a `record_motor_usage`) — diferencia real entre los dos
  escritores, no una decisión deliberada documentada en su momento.
  Queda registrada como pendiente de bajo riesgo (el camino HTTP-directo
  tiene tráfico real bajo), no como "aceptada a propósito".

- **`check_usage_reconciliation()` (`jacobs/reaper.py`) cubría solo
  Motor Registry, no HTTP-directo — CERRADO 2026-08-21.** Extendido para
  leer también `jacobs_events` (join `STEP_STARTED`+`STEP_COMPLETED` por
  `step_id`, filtrado a `hipatia`/`jekyll`/`thot`/`ada`) y compararlo
  contra `axioma_usage` (`request_type='pipeline'`). Verificado en vivo
  contra la DB real (no solo con los tests unitarios): con ventana de
  24h reprodujo exactamente el gap histórico ya conocido — 4/8 dispatches
  HTTP-directos sin fila (el pipeline `8d02047d-...` de la sesión
  anterior). **Limitación real, no un cierre completo:** este camino no
  tiene `job_id`/`step_id` en `axioma_usage` (`record_direct_usage` no
  lo escribe), así que la reconciliación es por CONTEO agregado por
  facet en la ventana, no 1:1 como Motor Registry — no puede señalar
  *cuál* dispatch puntual falta, solo si el total de un facet quedó
  corto. Cobertura real donde antes había cero, documentado como
  aproximado a propósito (`jacobs/reaper.py::_compute_http_direct_gap`,
  con test que cubre explícitamente que un exceso en un facet no debe
  tapar un gap real en otro). 5 tests nuevos en
  `jacobs/_usage_reconciliation_test.py`, 10/10 verdes junto con los
  preexistentes.

  **Hallazgo lateral, auditado 2026-08-21 (mismo día, follow-up):** la
  corrida en vivo mostró Motor Registry con 3/8 dispatches recientes
  (`3e5329a2`/kimi 01:22:41, `9a6715c3`/jax_local 02:00:10,
  `b7d1e056`/jax_local 02:46:10) sin fila en `axioma_usage`, gap 37.5%.
  **No es un bug nuevo — es la misma T1.c (identidad NULL descartada en
  silencio), no una regresión.** Los 3 corrieron con `user_id`/
  `tenant_id` NULL bajo el código PRE-fix (`las_manos/server.py`
  reinició a las 01:22:07 y 01:58:06 con el commit `580e7ce`, que todavía
  tenía `if not user_id or not tenant_id: return` en
  `record_motor_usage` — confirmado leyendo esa versión exacta del
  archivo vía git). El deploy real del fix (`3ec515e`) no ocurrió hasta
  el reinicio de las 02:52:14 — los 3 quedaron atrapados en la ventana.
  Confirmado con evidencia post-fix real: el job `fed93949`
  (2026-08-21 10:41:55, identidad NULL, después del deploy) SÍ escribió
  fila (`axioma_usage.id=231`, `tenant_id`/`user_id` NULL) — el fix
  funciona en producción, no es solo lectura de código. Sin acción
  requerida: los 3 salen de la ventana de 24h entre las 01:22 y las
  02:46 del 2026-08-22 y el gap baja solo. Efecto secundario esperado
  del diseño de la ventana (ya documentado en el código,
  `RECONCILIATION_WINDOW_SECONDS`): un fix deployado a mitad de la
  ventana deja ruido histórico visible hasta que esa ventana rota
  completo.

- **`axioma_usage` (prod) contaminada con test fixtures — LIMPIADA
  2026-08-21.** Auditoría encontró 106/224 filas (47%) sin ningún
  dispatch real detrás (verificado contra `jacobs_events`/
  `jacobs_pipelines`/`motor_jobs.jsonl` — cero coincidencia en las 106):
  90 de 4 ráfagas de timestamp idéntico (suite completa de
  `las_manos/motor_registry` corrida contra prod) + 16 de fixtures
  individuales de `las_manos/_motor_usage_writer_test.py` y
  `jacobs/_usage_writer_test.py` (kimi/jekyll, 1000-500 y 100-50).
  **Dato para el registro:** el commit `3ec515e` que cerró T1-T4
  documentó esto como "la fila huérfana" — singular, una sola fila. La
  auditoría de hoy encontró que la contaminación real era **6× mayor**
  (16 filas de fixtures individuales, más 90 de ráfagas de suite
  completa que ni siquiera estaban mencionadas) — el `setdefault()` que
  no pisaba `JAX_DB_NAME` ya exportado hizo bastante más daño del que se
  supo en el momento en que se cerró ese commit.

  Backup completo antes de borrar:
  `~/backups/axioma_usage_test_contamination_cleanup_20260821-165735.sql`
  (106 filas, verificado con diff id-por-id contra la lista esperada, no
  solo tamaño de archivo — el primer intento de backup usó un `WHERE
  created_at IN (...)` con horas locales que `mysqldump` evaluó en UTC
  por su `SET TIME_ZONE='+00:00'` interno y solo capturó 16/106 filas
  sin ningún error visible; detectado verificando el conteo real, no
  confiando en `exit=0`). `DELETE` corrido dentro de una transacción con
  `ROW_COUNT()` verificado == 106 antes del `COMMIT`. Total de la tabla
  224 → 118, confirmado. Filas legítimas vecinas (ids 123, 127-130,
  225-233, con `job_id` real y tokens no redondos) verificadas intactas
  después. Las 3 fuentes de contaminación ya tienen guard fail-loud
  desde antes (`3ec515e`/`5eed90b`) — no debería seguir creciendo.

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

- **4 fuentes de verdad para el vocabulario de capabilities** (`VALID_CAPABILITIES`
  estático + `las_manos/config.toml [capabilities.*]` + `_CAPABILITY_MAP` +
  la DB real) — CERRADO 2026-08-21 (Bloque 3, PR jax#24). Las 3 copias
  estáticas eliminadas; `jacobs/executor.py::validate_capability()` y
  `jacobs/plan.py::_parse_plan_json`/`_validate_plan_capabilities` consultan
  ahora la MISMA fuente (`jacobs/store.py::get_motor_governance()`,
  extendida a vista completa de `capability` -- no solo `allowed_capabilities`,
  también `allowed_callers` y el resto de campos de gobernanza, para no
  perder en silencio el chequeo de caller al consolidar).
  **Evidencia del drift que motivó cerrarlo así, no solo teoría:** 2 de
  las ~14 capabilities auditadas contra config.toml tenían
  `max_execution_minutes` desincronizado de la DB real --
  `code_swarm` (30 en config.toml vs 5 en DB) y `refactor` (10 vs 5).
  Ninguno de los dos causó bug visible porque NIVEL B nunca chequeaba ese
  campo -- pero **2 de 14 ya divergidas, sin que nadie lo notara**, es el
  argumento real de por qué la copia tenía que desaparecer, no una
  hipótesis. Los 5 alias semánticos de `_CAPABILITY_MAP`
  (analysis/research/review/code/implement) resultaron estar MUERTOS para
  Motor Registry -- verificado contra `capability_motor` real (cero filas)
  y `jacobs_steps` histórico completo (cero steps con esos nombres y un
  facet-motor; `code`/`implement` cero uso en absoluto, en cualquier
  facet). No se sembraron como filas de motor-registry (hubiera sido
  inventar semántica que nunca existió). `analysis`/`research`/`review` SÍ
  se usan, pero solo con facets HTTP-directos (ada/jekyll/hipatia/thot) --
  se sembraron como 3 filas de vocabulario puro en `capability` (sin
  `capability_motor`, `allowed_callers=["jacobs"]` verificado contra
  `jacobs_steps`/`jacobs_events`: nunca hubo otro caller real). Caso de
  regresión reproducido explícitamente: `capability='execute'`,
  `facet='hyde'` (rechazo real en producción, 2026-08-21 04:38, antes de
  este cambio) sigue rechazándose con mensaje equivalente después del
  cambio.

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

- **`AdminMotors.jsx` ahora lista `analysis`/`research`/`review` como
  capabilities adjuntables a un motor** — efecto colateral menor de
  sembrarlas en `capability` (Bloque 3, arriba). Nada las impide
  técnicamente: un admin podría crear una fila `capability_motor` para
  alguna de las 3 vía ese formulario, cosa que el diseño de Bloque 3
  evitó a propósito (son vocabulario puro para HTTP-directo, no
  capabilities de motor-registry). No es un bug -- nadie lo hizo, y el
  formulario no rompe nada si lo hicieran -- pero es una superficie que no
  existía antes de esta sesión. Sin urgencia.

- **Require PR sobre `master` — CERRADO 2026-08-27.** Ambos repos
  (`Jax`, `jax-platform`) hechos públicos por Fernando, después ruleset
  `master-protection` creado en los dos (`Settings → Rules → Rulesets`):
  target `~DEFAULT_BRANCH`, reglas `deletion` + `non_fast_forward` +
  `pull_request` (0 approvals requeridos), bypass actor
  `RepositoryRole` id 5 (admin) en modo `always` — Fernando conserva push
  directo, todo lo demás pasa por PR. Verificado en vivo por API
  (`gh api repos/fjruizhn/{Jax,jax-platform}/rulesets`) en ambos repos,
  `enforcement: active`. La duda sobre el plan gratis quedó resuelta:
  confirmado por 403 real de la API que `branch protection`/`rulesets`
  requieren Pro O repo público — al hacerlos públicos, ambas features se
  habilitaron gratis. Nota de UI: en repo personal (no de organización)
  el bypass list no ofrece buscar por username, solo roles — hay que
  elegir "Repository admin", no buscar "fjruizhn".

- **14 (en verdad 18) tests de `las_manos` fallaban por `host='localhost'`
  en vez de `127.0.0.1:3308` — CERRADO 2026-08-24.** Mismo patrón que el
  bug ya documentado de puerto dual (3306 stale/muerto / 3308 real, ver
  memoria `jax-dual-mariadb-instances`). El conteo real al auditar era 18,
  no 14 (creció con los tests nuevos de reconciliación HTTP-directo del
  mismo día) -- los 14 originales eran una cifra vieja de ronda 9
  (2026-08-20) nunca re-verificada.

  **Fix real:** el fallback silencioso a `localhost:3306` vivía duplicado
  en 19 archivos (patrón deliberado "sin paquete compartido" entre
  jacobs/las_manos/jax/core -- cada repo se conecta con su propio
  conector mínimo). Los 19 pasaron de default silencioso a
  `RuntimeError` explícito si `JAX_DB_HOST`/`JAX_DB_PORT` no están
  seteados -- producción no se ve afectada (los 4 servicios/timers
  systemd relevantes cargan `/etc/jax/.env` vía `EnvironmentFile`,
  confirmado leyendo las unit files, nunca dependieron del default).

  **Dos hallazgos laterales durante la auditoría, ambos cerrados en la
  misma sesión:**
  1. `las_manos/_motor_v02_test.py` (ahora
     `scripts/manual_motor_v02_integration.py`) no era un test real --
     un script manual de integración cuyo nombre matcheaba el patrón de
     descubrimiento de pytest (`*_test.py`), con TODO el código a nivel
     de módulo (sin `if __name__ == "__main__":`). Cualquier `pytest`
     corrido sobre `las_manos/` lo importaba y disparaba un dispatch
     real a Kimi contra el LAS MANOS de producción, con polling de
     hasta 120s, y -- si llegaba a completar -- activaba el kill switch
     `/etc/jax/PAUSE` de producción vía sudo. Confirmado con `journalctl`:
     11 dispatches reales disparados durante el propio diagnóstico antes
     de cuarentenarlo (ningún efecto permanente -- ninguna corrida llegó
     a tocar PAUSE). Movido fuera de `las_manos/` y renombrado para que
     pytest deje de descubrirlo.
  2. `jacobs/_pipeline_motor_e2e_test.py` (ahora
     `scripts/manual_pipeline_motor_e2e.py`) -- mismo patrón de nombre,
     pero inofensivo: son `async def test_...()` sin marcador de
     pytest-asyncio/anyio, y pytest las rechaza de entrada ("async def
     functions are not natively supported") sin ejecutar nada. Igual
     cuarentenado -- era falsa cobertura (reportaba FAILED sin haber
     corrido el e2e real) y mismo riesgo latente si el proyecto agrega
     algún día un plugin async.

  **Un tercer bug, sin relación, apareció recién al arreglar el
  fallback:** con host/puerto correctos, 2 tests de
  `jacobs/_step_motor_test.py` seguían fallando por
  `Unknown column 'depends_on' in 'INSERT INTO'` -- `jax_memory_test`
  no tenía esa columna, que sí existe en `jax_memory` (prod). Causa
  raíz real: `depends_on` se agregó a producción en algún momento vía
  `ALTER TABLE` manual, sin registrarse en la lista de migración
  idempotente de `jacobs/store.py::init_tables()` (el mecanismo real de
  este repo -- no hay carpeta `migrations/`, cada tabla tiene un
  `CREATE TABLE IF NOT EXISTS` + una lista `(columna, ddl)` chequeada
  contra `information_schema.COLUMNS`). Como nunca se agregó a esa
  lista, ninguna DB nueva la recibía -- no solo `jax_memory_test`,
  cualquier entorno futuro (dev nuevo, restore de desastre) tendría el
  mismo gap. Agregado a la lista de `jacobs_steps` en `store.py` con el
  DDL exacto de prod (`LONGTEXT` + collation + `CHECK (json_valid(...))`,
  confirmado con `SHOW CREATE TABLE` contra `jax_memory`), y aplicado en
  vivo contra `jax_memory_test` corriendo `init_tables()` de verdad (no
  un ALTER a mano) -- mismo camino que correría en producción.

  **Resultado final verificado:** suite completa de `las_manos/`
  (incluye `jacobs/` vía symlink) -- 95 passed, 0 failed (antes: 18
  failed / 81 passed; los 4 tests del hallazgo lateral #2 ya no cuentan,
  quedaron fuera de la colección de pytest al cuarentenarlos).

- **`_capability_check()` en `facet_bindings.py` -- el badge "Meets" del
  tab Bindings puede leerse como una garantía más amplia de la que da.**
  Auditoría 2026-08-24 (síntoma: ada/thot mostraban "✓ Meets" mientras
  `motor`/`facet_binding` estaban divergentes). El check hace exactamente
  lo que dice: compara `model.supports_tool_use`/`context_window` (del
  modelo que `facet_binding.model_ref` apunta hoy) contra lo que la
  faceta exige -- no es un bug, no valida nada sobre `motor` ni sobre si
  el modelo aprobado coincide con lo que de verdad se está sirviendo. Es
  una pregunta ortogonal ("¿el modelo configurado cubre tool_use/
  contexto?"), por eso pasaba igual estando divergente. El riesgo es de
  naming/UX: un humano puede asumir que "Meets" certifica más de lo que
  certifica. No se toca -- anotado a pedido explícito, no forma parte del
  fix de `motor_resolved`.

En memoria de Jairo Urbina.
