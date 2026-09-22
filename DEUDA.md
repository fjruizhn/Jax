# Deuda técnica — lista canónica

> This is a non-authoritative ledger of technical debt, known limitations, and unresolved operational work. It cannot override policy, prove runtime or enforcement state, ratify authority, or authorize execution.

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
- **Una excepción dentro del `finally` de `conexion()` cuelga el apagado del servicio — hallado 2026-09-17 (ronda pre-vuelo y continuar), CÓDIGO Y ARREGLO DEL FRENTE F.**
  - **Archivo y función:** `jacobs/store.py`, el context manager `conexion()`
    (alias `conexion_del_pool()`), su bloque `finally`.
  - **Condición exacta:** si una excepción INESPERADA (no una de las que el
    bloque contempla) cae dentro de ese `finally` **antes** de que corra
    `pool.release(conn)` — por ejemplo al invalidar el envoltorio vigilado, al
    evaluar si la sesión es reutilizable o al cerrar la conexión — la conexión
    nunca vuelve al pool y queda en `_used`.
  - **Consecuencia:** `pool.wait_closed()` espera para siempre a que `_used`
    quede vacío, así que **el apagado del servicio se cuelga** (el lifespan de
    LAS MANOS no termina; `systemctl stop` va al timeout y mata el proceso).
  - **Cómo se destapó:** un `AttributeError` de un doble de test dentro de ese
    tramo, durante la ronda de pre-vuelo y continuar. No es un caso teórico.
  - **De quién es:** el pool de `jacobs/store.py` es del **frente F** (sesión
    `fruiz-47`); este código vino de master, **no** de la rama
    `feat/prevuelo-y-continuar`, y **esa rama NO lo toca**. El dueño ya
    respondió: el frente F toma el arreglo con la verificación propuesta.
  - **Verificación propuesta (acordada con el dueño):** un test que fuerce una
    excepción en ese tramo del `finally` y mida que `wait_closed()` **no**
    cuelga — con **timeout duro** (`asyncio.wait_for` / `timeout` de pytest),
    para que el control muera **en rojo** y no colgando la suite entera.
  - **Cierre:** lo borra el frente F citando su PR cuando lo cierre.
- **La ruta vieja del freno `/etc/jax/PAUSE` sigue frenando — frente B, Task H (2026-09-17).**
  - **DECISIÓN** (controlador principal del frente B, 2026-09-17; rulings R11-R15 del ledger
    `jax-platform/.superpowers/sdd/2026-09-16-frente-b-kill-switch/progress.md`): mientras exista
    `/etc/jax/PAUSE`, todo lector del interruptor (LAS MANOS server, ssh_worker, motor_registry,
    Jacobs policy/executor, el REPL, `jax --task` y jax-platform) lo trata como freno PUESTO, además
    del archivo de `JAX_KILL_SWITCH_PATH` (`/etc/jax/interruptor/PAUSE`, que no cambia). Si no se
    puede mirar (cualquier OSError que no sea ENOENT) cuenta como puesto. Un WARNING por episodio
    nombra la ruta vieja y la nueva. La regla vive una sola vez en `interruptor_activo` del módulo
    compartido (`jax/core/interruptor.py` + copia `jax-platform/backend/interruptor.py`, familia
    `interruptor` de `scripts/check_mirror_sync.py`). La plataforma decide sus escrituras con
    `pausa_presente` (sólo su archivo), informa con el freno completo más el campo `heredada`, y
    **nunca borra `/etc/jax/PAUSE`**: reanudar con la heredada puesta responde `activo: true,
    heredada: true` y la Mesa avisa (`killSwitchHeredada`).
  - **Por qué:** gente y scripts pausan creando esa ruta; si dejara de leerse, quien pause así
    creería que frenó, y no.
  - **PENDIENTE:** retirar la ruta heredada (constante `RUTA_HEREDADA`, la excepción del test de
    clase `tests/test_interruptor_sin_rutas_fijas.py`, el aviso de la Mesa) — la fecha la decide
    Fernando.
- **ESTADO — 2026-09-01.** Bloque único. Reemplaza los cinco bloques de
  cierre que la ronda de seguridad fue apilando en este mismo lugar; cada uno
  se declaraba "estado único" y convivía con los anteriores, que es
  exactamente el problema que este documento existe para evitar. Los ítems
  cerrados esa noche se movieron íntegros a la sección **"Cerrado — ronda de
  seguridad 2026-09-01"**, más abajo. Nada se borró.

  **Incidente de credenciales: CERRADO.** Un secreto real en toda la historia
  de ambos repos, rotado y verificado por tres vías. Cinco falsos positivos
  retractados antes de llegar a `master`. Detalle en la sección de cerrados.

  **Esperando a tercero — sin trabajo de este lado:**

  | Ítem | Bloqueado por |
  |---|---|
  | Ticket a GitHub Support (`$AUDIT/TICKET-GITHUB.md`) | su respuesta |

  La fila "Purga de dumps en R2 — 2026-09-08" **salió de esta tabla el
  2026-09-11: su premisa era falsa** (ver el párrafo de fechas vencidas).

  **Deuda técnica abierta:** los ítems que siguen en esta sección — al cierre
  del 2026-09-01 queda **uno**: el merge sin revisión en `master`, y su
  resolución ya está decidida (no se hace hoy, razones escritas abajo). Los
  otros dos que figuraban acá salieron el mismo día y ninguno por trabajo
  nuevo: `GPU_SEMAPHORE` era una **decisión tomada** mal etiquetada —se movió
  a "Anotado, no bloquea" y se le agregó el tripwire que le faltaba— y el
  owner de pipeline **ya estaba arreglado desde el 2026-08-20**. Cuarta y
  quinta instancia del patrón del día: el ítem describía un estado que ya no
  existía. De ahí la regla nueva de §7 de `CONTEXT.md`.
  **Fecha de control más próxima:** la de `kimi` (**2026-09-10**) se cerró el
  2026-09-11 — ver "Cerrado — kimi y la memoria vector cero" más abajo.
  **La "purga de dumps en R2" del 2026-09-08 nunca iba a ocurrir — medido
  2026-09-11.** La fecha salió de sumar solo el Bucket Lock de 7 días, sin
  mirar la retención ni el prune:

  | Hecho medido | Evidencia |
  |---|---|
  | Los dos repos (R2 y local) conservan **5 snapshots anteriores a la redacción** del 09-01: 07-13, 07-31, 08-23, 08-30, 08-31 | `restic snapshots` en ambos |
  | Cada uno lleva **2 líneas con hash bcrypt** en `jax_memory.sql`; el de 09-11 lleva **0** (la redacción funciona) | `restic dump` + conteo, sin imprimir un solo hash |
  | La retención (`--keep-daily 7 --keep-weekly 4 --keep-monthly 6`) los guarda **a propósito, hasta 6 meses** | `backup-hall9000.sh`, paso 6 |
  | En R2 **nunca se borra nada físicamente**: el paso 6 hace `forget` sin `--prune`, y sin prune restic no borra packs jamás | mismo paso; el comentario decía que "el espacio se libera cuando expira la inmutabilidad" — **falso**, corregido en el script el 2026-09-11 (y decía 10 días de lock: son 7) |

  Uno de los dos hashes es casi seguro el de **`user_id=2`, que sigue
  vigente** (su contraseña nunca se rotó). El repo está cifrado por restic:
  leerlo exige las credenciales de R2 **y** la contraseña del repo.

  **Decisión de Fernando, 2026-09-11: rotar la contraseña de `user_id=2`**
  (coordinada con la persona; mismo procedimiento que `user_id=1`), en vez de
  olvidar los snapshots — conserva los puntos de restauración de julio y
  agosto y deja sin valor los hashes viejos. **ROTADA el 2026-09-11 04:50:10**
  por orden de Fernando (la coordinación con la persona queda de su lado) —
  detalle y evidencia en el ítem de la segunda cuenta, más abajo. Desde ahí,
  los 2 hashes de los snapshots viejos son de valores que ya no abren nada.

  **R2 sin prune — CERRADO 2026-09-11.** `backup-hall9000.sh`, paso 6, hace
  ahora un **prune diferido** en R2 después del `forget`:
  `prune --max-unused unlimited --max-repack-size 0`, con `timeout 900`.
  Borra solo packs que ningún snapshot usa y **no reempaqueta** (reempaquetar
  borra packs en uso recientes, que el lock rechaza: el origen de los 20 min de
  reintentos de agosto). Un pack sin uso sale de un snapshot olvidado, y todo
  snapshot olvidado es anterior a los 7 diarios que se conservan: por
  construcción queda fuera del lock. Si falla, `RETENTION_OK=0` — registrado,
  no escondido.

  | Verificación | Resultado |
  |---|---|
  | dry-run, edades de los packs en R2 (`HeadObject`) | los 8 packs sin índice que se borrarían: 26 a 33 días; los 14 objetos recientes son de los snapshots en uso |
  | prune real | **52,5 MiB** borrados (39 packs viejos + 8 sin índice), 0 reempaquetados, 0 rechazos, 4 s |
  | `restic check` + restauración | sin errores; `jax_memory.sql` del último snapshot = staging, md5 idéntico |
  | **corrida real del script modificado** (05:16) | `✓ Forget en r2`, **`✓ Prune diferido en r2`**, servicio `success`, exit 0; `restic check` posterior sin errores |

  El script no está en git (`/opt/backup-scripts/`): respaldos
  `backup-hall9000.sh.bak-20260911-comentario-lock` y `-prune-r2`.

- **El REPL estaba roto para las 7 facetas — CERRADO Y EN MASTER 2026-09-14** (jax#153 →
  `fbab074`; decisión de Fernando: la lista permitida sale del catálogo `model`). Verificado
  desde `~/jax` (el checkout que corre el REPL): 7 facetas, ninguna rota. Sin servicio que
  reiniciar: el REPL lo toma al arrancar.** Verificado con el código de la rama contra la DB
  real: las 7 facetas con su modelo dentro de la lista, hipatia respondió, jekyll acepta
  el modo pesado, y la consulta nueva usa `uk_provider_model`. La revisión de
  jax#153 pidió tomar el proveedor del MODELO ASIGNADO (JOIN por `model_ref`) y no de
  `facet_binding.provider_id`, que el endpoint de aprobación puede dejar desalineado.
  **Anotado aparte:** los workers de memoria (`jax/memory/worker.py`,
  `synthesis_worker.py`) arman sus músculos con `deepseek-v4-flash` y una lista fija;
  hoy coinciden, pero sufren el mismo bug el día que cambie el modelo.

  Texto original (medido 2026-09-14): `main()` (en `jax/core/main.py`, antes del
  arreglo) pisa `model_default` con el modelo de
  `facet_binding`, pero `models_allowed` sigue siendo el de `config.toml`, y
  ninguno de los modelos vigentes está en su lista: invocar cualquier faceta
  desde el REPL da `ModelNotAllowedError` antes de llamar al proveedor.

  | Faceta | facet_binding | models_allowed del TOML |
  |---|---|---|
  | jax_local | qwen3.6:35b-a3b-q4_K_M | qwen3-coder:30b, qwen3:14b, qwen2.5:7b, llama3.2:3b |
  | hyde | claude-opus-5 | sonnet, opus, haiku |
  | jekyll | deepseek-flash | deepseek-v4-flash, deepseek-v4-pro |
  | hipatia | gemini-3.8-flash | gemini-2.5-flash, gemini-2.5-pro |
  | thot | gpt-5.6-terra | gpt-5.5, gpt-5.5-pro, gpt-5.4, gpt-5.4-mini, gpt-5.1-codex, gpt-4o |
  | ada | glm-5.3 | glm-5.2 |
  | kimi | kimi-k3 | kimi-k2.7-code, kimi-k2.6 |

  **Evidencia:** la misma secuencia de `main()` en proceso (registry → build_muscles →
  `hipatia.invoke`) lanzó `ModelNotAllowedError: [hipatia] modelo 'gemini-3.8-flash'
  no permitido`. La Mesa web y Jacobs NO están afectados: resuelven por
  `resolve_facet()` y no consultan `models_allowed`. Nadie lo vio porque el REPL no
  tiene uso (0 turnos en 14 días, medido el 2026-09-03). Es el patrón de siempre:
  una segunda fuente de verdad (`models_allowed` del TOML) que nadie mantiene.
  **Qué falta:** decidir de dónde sale la lista permitida (¿el catálogo `model` de la
  DB?) — no copiar el modelo vigente al TOML, que es el parche que lo reproduce.

- **`invoked_by` es un campo de AUTORIZACIÓN en Jacobs, con un nombre de persona como
  rol — CERRADO 2026-09-14 (tanda A, ver su sección). Texto original:** ABIERTO, medido 2026-09-14. `jacobs/routes.py:100` acepta solo
  `{"Fernando", "jax_local", "ada"}` y `policy.validate_resume` exige `"Fernando"`;
  jax-platform lo manda fijo (`PipelineModal.jsx` y `api/pipelines.py:161`). Cambiarlo
  por el usuario autenticado rompería crear y reanudar pipelines, así que NO se tocó en
  el lote 1 de la deuda sin fecha. Es contrato de autoridad (Principio IX): va con la
  tanda del validador de gobernanza.

- **El resolver de `CAPABILITY_AVAILABLE` consulta un catálogo que el Bloque 3
  vació — CERRADO 2026-09-14 (tanda A, ver su sección). Texto original:** verificado 2026-09-02. **Causa:** el Bloque 3 movió las
  capabilities a la DB (`MotorCatalog.from_db()`), pero
  `policy/governance/validator.py::load_validation_context()` sigue
  construyendo `MotorCatalog(config)` desde `las_manos/config.toml`, cuyo
  `[capabilities.*]` tiene 0 entradas. La rama `in_catalog` de
  `_resolve_capability_available` es código muerto en producción: un claim
  sobre una capability que existe solo en la DB da `FACT_MISMATCH`.
  **Evidencia:** `tests/test_governance_validator.py::test_capability_available_found_only_in_catalog_mode_unverified_but_accepted`
  estaba ROJO en `master` (`assert 'FACT_MISMATCH' == 'VALID'`) y nadie lo
  veía porque `tests/test_governance_*.py` no corría en ningún job de CI
  (arreglado el mismo día: job `governance`, PR jax#101). **Tripwire:**
  `tests/test_governance_validator.py::test_real_toml_catalog_is_empty_since_block3_so_catalog_branch_is_dead_in_production`
  fija el estado actual y se pone rojo cuando alguien arregle el validador —
  ese rojo es la señal de cerrar este ítem. **Qué falta:** que el validador
  lea el catálogo de la DB por el mismo camino que `jacobs`. **Por qué SP3
  puede diferirlo:** `grounding.build_snapshot()` lee solo `ctx.ops`, nunca
  la rama muerta (spec SP3 §3.3).

- **Bypass de admin en el ruleset de `master` — PUSH DIRECTO CERRADO
  2026-08-28; el merge sin revisión NO, y abajo está medido por qué.** Los dos
  repos tienen protección de `master` con bypass para admin (ver
  `jax-block1-apertura-repos-cierre`). Eso era higiene aceptada hasta que
  mostró su costo con un caso concreto.

  **Incidente, 2026-08-27:** el PR de la Task 4 de la ronda de alertas
  (`jax-platform` #25) se mergeó **con `no-fail-open-except` en FAILURE**.
  El comando de merge imprimió los checks —el rojo estaba en pantalla— y
  corrió `gh pr merge` en la misma cadena, sin condicionar nada. El bypass
  hizo que no encontrara ninguna resistencia. `master` quedó en rojo hasta
  el fix-forward (#26, `cab6f80`).

  **Por qué bloquea, y no es higiene:** hoy la única barrera entre un merge
  en rojo y `master` es la disciplina de quien mergea. Es **exactamente el
  modo de falla que la ronda de alertas de facets existe para eliminar** —
  "la regla se cumple cuando alguien se acuerda de mirar" — reproducido en
  la infraestructura que gobierna esa misma ronda. Un guard que depende de
  memoria humana no es un guard.

  **Segundo incidente, 2026-08-28, y es el que lo cerró:** un
  `cd <worktree> && git revert ...` falló por sintaxis; la llamada siguiente
  arrancó en el checkout principal, sobre `master`, y el `git revert HEAD`
  revirtió un PR ya mergeado (la décima lección de `CONTEXT.md`). El push
  salió con `Bypassed rule violations for refs/heads/master`. Se restauró en
  el acto (`b85e317` revert, `8dfc91f` reapply — **deliberadamente no
  reescritos, que quede el registro**), pero nada lo frenó. La diferencia con
  el primer incidente importa: aquél fue disciplina, éste fue **mecánico** —
  ninguna cantidad de cuidado previene un `cd` que no se ejecutó.

  **CERRADO 2026-08-28 — el push directo a `master`.** `bypass_mode` del
  ruleset pasó de `always` a `pull_request` en los DOS repos: el bypass
  sigue existiendo para el merge de un PR, pero **el push directo se rechaza
  server-side, también para admin**. No es una lectura de la documentación:
  se verificó con un push real (`GH013: Repository rule violations found for
  refs/heads/master`) con las mismas credenciales que 40 minutos antes
  bypasseaban. Defensa en profundidad: hook `pre-push` versionado en
  `ops/githooks` (`jax` #66, `jax-platform` #31) que decide por el **ref
  destino** que git pasa por stdin — no por el cwd, que es exactamente lo
  que estaba mal ese día. Se verificó reproduciendo el incidente completo
  (comando compuesto que falla, cwd equivocado, push a master): rechazado
  por el hook, y con `--no-verify` rechazado igual por el ruleset. Los hooks
  son compartidos entre worktrees (verificado), así que un `git config
  core.hooksPath ops/githooks` por clon cubre todos los árboles.

  **NO CERRADO, y ahora se sabe por qué: el merge sin revisión.**
  `required_approving_review_count` es 0. Subirlo a 1 **no sirve con una
  sola identidad**, y esto se midió, no se supuso (experimento en
  `jax-platform`, ruleset restaurado después):

  | Intento | Resultado |
  |---|---|
  | `gh pr review --approve` sobre un PR propio | `Can not approve your own pull request` |
  | `gh pr merge` con approvals=1 | `the base branch policy prohibits the merge` (BLOCKED) |
  | `gh pr merge --admin` | **mergeó igual** (`3acd8f3`) |

  O sea: con approvals=1 y una sola cuenta, el camino normal queda cerrado
  para todos —incluido Fernando, que no puede aprobar sus propios PRs— y el
  único camino es `--admin`, que el agente puede tipear igual de fácil. **No
  crea un gate humano: crea un flag.** El agujero del merge sin revisión
  sólo se cierra con identidad separada.

  **Identidad separada (bot/GitHub App): ANOTADA, no hecha — y es la ÚNICA
  opción que cierra el merge sin revisión. Es PROTECCIÓN, no auditoría.**
  Esta entrada decía antes que era separación para auditoría con una
  excepción de protección al margen; está al revés y la corrección importa,
  porque de eso depende con qué urgencia se lea: la separación de identidad
  es **el único mecanismo que convierte
  `required_approving_review_count = 1` en un gate real**. Con una sola
  cuenta, approvals=1 sólo bloquea el camino normal y deja `--admin` (ver la
  tabla de arriba). Con un bot de rol `write`: sus PRs los tiene que aprobar
  Fernando, y el bot **no tiene `--admin` que tipear**. La ganancia de
  auditoría —saber QUIÉN además de QUÉ— es un efecto secundario, no la
  razón.

  **Decisión de Fernando, 2026-08-28: no se hace hoy**, con las razones
  escritas para que la próxima ronda no las reconstruya:
  1. el agujero que queda (merge sin revisión) es el que **en la práctica sí
     tiene revisión** — cada PR de esta semana pasó por Fernando antes de
     mergearse. No es garantía mecánica, pero tampoco está desatendido;
  2. el que **sí estaba desatendido** —push directo por error mecánico, sin
     oportunidad de que nadie mirara— quedó cerrado hoy, por dos capas
     independientes y verificado con el escenario exacto del incidente;
  3. el costo **no está bien medido**: los comandos manuales de Fernando en
     la misma terminal saldrían como bot, y falta ver cómo interactúa con
     `require_extra_approval_for_unattributed_changes` (hoy en `true`). Es
     la clase de cambio que abre tres preguntas nuevas, y no hay un
     incidente que lo justifique.

  **Qué lo reabre:** un merge sin revisión que llegue a `master` y cause
  daño — es decir, el incidente que hoy no existe. Alcance concreto ya
  medido, para no rehacerlo: cuenta nueva con 2FA (paso interactivo de
  Fernando), invitación como colaborador con rol **write** —no admin, que es
  lo que la deja fuera del bypass—, llave SSH propia y `GH_CONFIG_DIR`/token
  aparte en hall9000.

  Ver la séptima lección de método en CONTEXT.md ("verificar y actuar en el
  mismo comando no es un gate") y la nota de §7 sobre el `cd` de un comando
  compuesto que falla.

- **El catálogo del Motor Registry no se enteraba de cambios en la DB — CERRADO
  Y DESPLEGADO 2026-09-12** (jax#140 → `e2e0a77`, `jax-las-manos` 14:41:15;
  jax-platform#58 → `668e79f`, `jax-platform` 14:45:26). Reusa el sello de
  `facet_resolver` (mismo archivo, mismo contrato): LAS MANOS hace un `stat`
  al entrar a `dispatch` (p50 0,70 µs) y recarga bajo lock si el sello es más
  nuevo que su carga (`from_db` p50 0,95 ms); recarga fallida → 503, nunca el
  catálogo viejo. Escritores que estampan después de commitear:
  `run_migrations`, `create_motor`, `update_motor` (los rebinds de facets ya lo
  hacían). **En vivo:** el sello pasó de 14:37:46 a 14:45:26.58 al arrancar
  jax-platform (reinicio pedido 14:45:25.95) y el dispatch siguiente respondió
  sin error con el techo de 900 s. **La regla operativa de abajo ya no hace
  falta**: con los dos desplegados, el orden de arranque deja de importar.
  **Hallazgo al desplegar, arreglado en el mismo PR:** la suite de
  jax-platform estampó el sello REAL en hall9000 (14:37:46): el fixture de
  sesión `client` corre `run_migrations` antes del aislamiento por función.
  `conftest` fija ahora `JAX_FACET_SEAL_PATH` a un temporal antes de cualquier
  import; control: mtime del sello real idéntico antes y después de la suite.

  Texto original del ítem, conservado:
  **ABIERTO, 2026-09-12, causó una regresión en producción.** `routes.py` carga
  `MotorCatalog.from_db()` **una sola vez**, al arrancar LAS MANOS (su propio
  docstring lo declara como limitación). Nada lo invalida: ni una migración de
  jax-platform ni un cambio desde el panel de admin.

  **Lo que pasó (HISTORIA):** el deploy de la ronda b8f80733 reinició
  `jax-las-manos` y `jax-platform` a la vez (13:03:15). La migración
  `generate` 5 → 15 corre al arrancar jax-platform, así que LAS MANOS ya había
  cargado el 5. Jacobs lee el techo de la DB (900 s) y el Motor Registry lo
  rechazaba contra su copia vieja (300 s): **de 13:03 a 13:29, todo paso de
  `generate` por kimi o jax_local fue rechazado.** Lo destapó la E2E de la
  cadena (pipeline `93fcd81f`), no un monitor. El motivo real además quedaba
  tapado por un `NameError` en la línea que loguea el rechazo (desde
  2026-06-29; arreglado en `fix/jacobs-rechazo-nameerror`).

  **Mitigado:** `jax-las-manos` reiniciado 13:29:52, verificado con un dispatch
  que se rechaza sin llamar al modelo:
  `excede el techo de 'generate' (15 min = 900s)`.

  **Regla operativa hasta que se arregle:** en un deploy que migra `capability`
  o `motor`, reiniciar **primero `jax-platform`** (migra) y **después
  `jax-las-manos`** (carga). Verificar con un dispatch con `timeout_seconds`
  = techo + 1: el rechazo muestra el techo que el proceso tiene en memoria.

  **Qué falta:** invalidación entre procesos, igual que el sello de
  `facet_resolver` (cerrado el 2026-09-11) para los facets. Es la misma clase
  de defecto: una copia en memoria de algo que vive en la DB.
  **Fecha de control: 2026-09-19.**

- **El login no tenía límite de intentos por IP — CERRADO Y DESPLEGADO
  2026-09-12** (jax-platform#59 → `3955f2a`, `jax-platform` 15:39:49; frontend
  `index-Cl-PZNlu.js`, backup `axioma-ia.io.backup-pre-ratelimit-20260912-154001`).
  Ventana deslizante en memoria (un solo proceso uvicorn, medido), por IP
  (`JAX_LOGIN_RATE_IP=20/60`) y por email (`10/300`), ANTES de la DB y del
  bcrypt; excedido → 429 con `Retry-After`, mismo cuerpo para todos. La IP sale
  de `X-Real-IP` solo si el peer está en `JAX_TRUSTED_PROXIES` (agregado a
  `/etc/jax/.env` = `172.16.20.11`, backup
  `/etc/jax/.env.backup-pre-trusted-proxies-20260912-152810`); sin Cloudflare en
  el camino (DNS directo, sin `cf-ray`, medido). **En vivo por la URL pública:**
  20 × 401 y después `429 retry-after: 56`; un control directo a `:8080` (otro
  peer) siguió en 401. Chequeo 1,75 µs/request.
  **Residuos CERRADOS el mismo día ("termina el login", jax-platform#60).**
  Medidos antes de tocar código, varios más graves que lo que decía este ítem:
  (1) bcrypt 5.0.0 LANZA `ValueError` con > 72 bytes (antes truncaba): login y
  reset daban 500, y una cuenta con contraseña larga fijada con el bcrypt viejo
  no podía entrar → `verify_password` trunca a 72 (semántica vieja exacta),
  reset rechaza > 72 con código; (2) **forgot-password enumeraba cuentas por
  tiempo** (real: DELETE + INSERT + SMTP síncrono hasta 10 s en el event loop;
  inexistente: instantáneo) — la misma propiedad que cerró #57, por otra
  puerta → el request no consulta nada, todo va en segundo plano
  (`add_safe_task`, SMTP en hilo, al email guardado); (3) forgot-password sin
  límite → comparte el del login y guarda la IP real, no la del proxy; (4)
  `email` sin tope con nginx a 50 MB → claves del LRU del tamaño que elija el
  atacante → `max_length=254`; (5) uvicorn 0.51 toma `--workers` de
  `WEB_CONCURRENCY` → `exigir_un_solo_proceso` al importar, probado levantando
  uvicorn con `WEB_CONCURRENCY=2`: los workers mueren con `RuntimeError`, el
  puerto nunca responde (el padre queda relanzándolos: systemd lo vería
  "active" — el aviso es el journal y el health check); (6) reset mostraba el
  `detail` crudo → códigos + i18n; (7) **ningún job de CI corría vitest** →
  job `frontend-tests`. **Declarado sin arreglo en la app:** un DDoS de >
  20.000 IPs se frena en el borde; el LRU por email no se vacía con IPs (son
  instancias separadas) y una clave atacada no se expulsa (cada intento la
  refresca). **DESPLEGADO y verificado en vivo** (jax-platform#60 → `ebf70cc`,
  canario de `frontend-tests` visto en rojo por el motivo inyectado sobre
  `99854c9` y revertido; `jax-platform` reiniciado 19:20:54, un proceso, sin
  hijos): login con contraseña de 100 bytes → 401 genérico en 0,155 s (antes
  500); forgot-password inexistente → 200 en 1 ms; email de 300 caracteres →
  422; journal 0 errores. Frontend `index-Bv42891O.js` servido por
  `axioma-ia.io` (backup `axioma-ia.io.backup-pre-login-residuos-20260912-192129`,
  idéntico por `diff -rq`). **Carga** (`jax/scripts/load_test.py`, que cuenta
  4xx como respuesta y solo 5xx como error): 2.400 peticiones a login y
  forgot-password con c=10 y c=50, **0 5xx**, p95 ≤ 18 ms a c=50 — son casi
  todas 429 del limitador desde una IP. El costo de bcrypt, medido aparte con
  25 logins secuenciales de emails distintos (el balde por email es de 300 s:
  repetir la carga con el mismo email mide solo el 429): **20 × 401 de 0,160 s
  de media (máx. 0,196 s) y después 5 × 429 en < 1 ms**, health 200. Es el
  diseño: una IP paga como mucho 20 bcrypt por minuto.

  Texto original: **ABIERTO, 2026-09-12.** Desde jax-platform#57 cualquier email
  costaba un bcrypt de ~155 ms (c=5 → 32 rps) sin ningún limitador.

- **El login de jax-platform dejaba saber qué cuentas existen — CERRADO Y
  DESPLEGADO 2026-09-12** (jax-platform#57 → `892ab45`, `jax-platform`
  reiniciado 14:35:02). Encontrado al contrastar el blueprint de Ricardo (§10)
  con `api/auth.py:39-55`: un email inexistente respondía 401 sin bcrypt
  (0,25 ms contra ~150 ms), y `403 inactivo` / `423 bloqueada` salían antes de
  verificar la contraseña. Ahora, sin la contraseña correcta, siempre el mismo
  401 genérico; un email inexistente verifica contra un hash de relleno de
  costo 12; el estado solo se revela con la contraseña correcta. Además
  `verify_password` pasó a `asyncio.to_thread` (bcrypt congelaba el event
  loop). **En vivo:** tres emails inexistentes → `401 "Usuario o contraseña
  incorrectos"` en 0,154-0,157 s. 5 tests, 3 mutaciones. **Residuo declarado:**
  con cuenta real y contraseña incorrecta hay un `UPDATE` extra del contador
  (ms, dentro de la varianza de bcrypt). `db/seed.py` es ruta de alto riesgo:
  el commit lleva `JAX_PRECOMMIT_ALLOW_PATH=1`, deliberado y revisado.

- **El tripwire de "sondas de medición" confunde `jax/ejecutor/contratos/contexto.py` con
  `scripts/ejecutor_fase0/contexto.py` por nombre — HALLADO 2026-09-22 (ronda 2 del contexto
  del Ejecutor, auditoría adversarial B3/M1), FUERA de ese encargo, sin arreglar.**
  - **Archivo y función:** `tests/test_payload_max_tokens_literal_tripwire.py::
    SondasDeMedicionTest::test_las_sondas_declaradas_no_son_codigo_de_servicio`.
  - **Condición exacta:** el test recorre `jax/`, `jacobs/` y `las_manos/` buscando, por
    TEXTO literal, `f"import {m}"` / `f"from {m} "` donde `m` es el STEM (sin ruta) de cada
    sonda declarada en `_SONDAS_DE_MEDICION` — entre ellas `scripts/ejecutor_fase0/
    contexto.py`. El commit `7d28a99` (ronda 1 de `feat/ejecutor-contexto-y-skills`,
    2026-09-22) agregó `jax/ejecutor/contratos/contexto.py`, un módulo SIN RELACIÓN con esa
    sonda que también se llama `contexto` — y el match por stem suelto no distingue los dos:
    cualquier `from jax.ejecutor.contratos import contexto` (hay tres: `arranque.py`,
    `cuenta_axioma.py`, y el propio `contexto.py` importándose por nombre en su docstring de
    módulo) se reporta como si `jax/` importara la sonda de medición de tok/s.
  - **Consecuencia:** el test sale ROJO — `tests-puros` (el job de CI que lo corre) no puede
    dar verde tal como está el árbol hoy. Reproducido con `git stash` contra `7d28a99` SIN
    ningún cambio de la ronda 2: el mismo rojo, así que es de la ronda 1, no de la 2.
  - **Cómo se descubrió:** al correr la suite completa de `tests-puros` para medir el piso de
    la aritmética que pide B3/M1 (ver `.github/workflows/policy.yml`, comentario junto al
    `grep -qE "^2467 passed, 30 skipped"`, con el número verificado A MANO aislando este
    hallazgo aparte, sin tocar el archivo real).
  - **Arreglo sugerido, NO aplicado (fuera del encargo de la ronda 2):** calificar el tripwire
    por RUTA de import (`scripts.ejecutor_fase0.contexto` / `scripts/ejecutor_fase0/contexto`),
    no por el stem suelto `contexto` — mismo criterio que ya usa para excluir `tests/`.
  - **De quién es:** ronda 1 de `feat/ejecutor-contexto-y-skills` (commit `7d28a99`), que
    eligió el nombre `contexto.py` sin correr esta suite completa antes de commitear.

## Medido — procesamiento de archivos: rendimiento, utilidad, caché y calidad de señal sobre 23 documentos reales (2026-09-21)

**Task 10 de la rama `feat/procesamiento-archivos`** (worktree
`jax-procesamiento`, ledger `.superpowers/sdd/2026-09-20-procesamiento-
archivos-nucleo/`). Cierra §7.A.2 (las 10 cifras) y §7.C (rendimiento) del
spec de `procesamiento/`, los dos criterios que ningún test de CI puede
cubrir porque dependen de archivos de clientes. **Ruling de alcance:** el
`task-10-brief.md` original pedía construir `scripts/
verificar_archivos_reales.py`; NO se construyó — `scripts/
medir_utilidad_extractos.py` (Task 9) ya mide lo mismo y más, y el brief se
escribió antes de que ese medidor existiera. Números tomados de
`task-9-report.md` y `task-9-defectos-report.md`, medidos contra 23
documentos financieros reales de Fernando (Nextcloud/macmini-bridge),
**ya borrados del disco** (ver esos reportes).

**Rendimiento (regla 4):** **2,70 s/página promedio sobre 88 páginas reales
escaneadas** (7 documentos, rango 2,01-3,50 s/página). Extrapolación
**serial** (no hay paralelismo en `ingesta.py`/`compuerta.py`, cada
`ingerir()` bloquea) a partir de ese promedio, asumiendo ~15 páginas por
documento (mediana de la muestra): 10 docs ≈ 6,7 min, 20 ≈ 13,5 min, 50 ≈
33,7 min. **Esto es extrapolación, no medición** — no se consiguió un
documento real de 50 páginas normales en la carpeta revisada. Y un solo
documento patológico puede costar más que diez normales: un PDF de
CamScanner con páginas a 9× el tamaño normal (1836×2376 pt) agotó el
timeout de `pdftoppm` (~300 s) y, antes del arreglo D-2 de esta misma
ronda, lo pagaba de nuevo en cada intento por no tener techo de
reintentos (ver "Cerrado" más abajo, D-1/D-2).

**Utilidad (§7.B, medido con `medir_utilidad_extractos.py`):** de 23
documentos reales, **13 de 13 documentos donde el criterio aplica**
(el original no cabía en el tope de 200.000 B de `file_read`) tienen un
extracto que sí cabe. Reducciones medidas por tipo: `.xlsx` 1,06×-268,5×,
`.pdf` nativo 13,1×-55,26×, `.pdf` escaneado 244,68×-415,53× (rango real de
`task-9-report.md`; el mayor factor individual es pdf-escaneado-02,
415,53×). Un `.xlsx` (`xlsx-04`) es la excepción conocida y ya reportada:
extracto 112× más grande que el original — no entra en los rangos de
arriba porque no es una reducción.

**Caché (§7.B.3):** 23 extracciones reales en la primera pasada sobre los
23 documentos, **0 en la segunda** — confirmado envolviendo
`compuerta.extraer` y contando llamadas reales, no asumiendo.

**Calidad de señal (§7.B.4):** **30,4 % de los documentos quedaron en
`parcial`** (7 de 23, todos PDF escaneados) — bajo el techo fijado del
60 %.

**Experimento de DPI (medido, código de producción SIN tocar — pedido
explícito, no aplicado):** reproducción sintética del tamaño de página
patológico (1836×2376 pt) con texto tamaño-factura y ruido realista. A
300 DPI (el actual, `ocr.py:DPI_RASTERIZADO`): 81,25 s, 20/20 campos OCR
correctos. A 150 DPI: 23,60 s (−71 %), también 20/20. A 75 DPI: 7,03 s,
pero cae a 5/20 (25 %). **El experimento no logró reproducir los ~300 s
reales del documento patológico** (su peor caso, a 300 DPI, fue 81 s, no
300 s) — se reporta como límite honesto del experimento, no como medición
del caso real; el factor ~3,7× de diferencia probablemente viene de
compresión JPEG real de una foto de cámara, que el ruido gaussiano
sintético no reproduce.

**B.1 (las 10 cifras) NO es medible por código — lo firma Fernando.**
Material preparado en `~/jax-workspace/verificacion-10-cifras/` (fuera del
repo, documento de cliente): el PDF original, el extracto completo, un
`COMPARAR.md` con la instrucción, y la lista de cifras que el sistema
marcó como dudosas (página + confianza) para que Fernando vea si el
sistema acertó al dudar.

- **PENDIENTE, fecha 2026-09-28:** calibrar el umbral de confianza del OCR
  con documentos reales (hoy es el default de `ocr.py`, nunca contrastado
  contra una muestra real más allá de esta ronda).
- **PENDIENTE, fecha 2026-09-28:** decidir el DPI de rasterizado —
  Fernando, con el número de arriba (150 DPI: −71 % de tiempo, sin
  pérdida de exactitud medida en la reproducción; 75 DPI: −91 % pero
  colapsa a 25 % de exactitud).

## Cerrado — ola final de arreglos de prevuelo-y-continuar (2026-09-17)

**HISTORIA 2026-09-17** (rama `feat/prevuelo-y-continuar`, sin mergear ni desplegar al escribir esto; revisión final de rama + Rulings R30-R33 del ledger `.superpowers/sdd/2026-09-17-prevuelo-y-continuar-jacobs/progress.md`). Regla de Fernando: ningún hallazgo se difiere, así que las dos entradas que la rama había anotado acá (Rulings R14 y R27) se cierran en la misma rama.

- **Cerrado — límite de activos entre procesos en `MAX_PARALLEL_PIPELINES` (antes "Anotado", desvío 7 del plan; ola final F3, Ruling R31).** El conteo contra el cupo se hacía bajo un `asyncio.Lock` de proceso y el CLI `tools/jacobs_relaunch.py` corre en otro proceso: un continue desde el CLI y un create de LAS MANOS podían contar a la vez y superar el cupo. **Arreglo:** `store.candado_de_activos()` — candado con nombre del servidor MariaDB (`GET_LOCK('jacobs_crear_o_continuar:<base>', JAX_PREVUELO_CANDADO_TIMEOUT_S)`, default 10 s) en UNA conexión dedicada; create y continue recuentan por esa conexión y escriben dentro del bloque; `RELEASE_LOCK` en `finally` y la conexión se cierra siempre (cerrar la sesión suelta el candado aunque `RELEASE_LOCK` falle). Si vence o la base no lo concede → falla cerrado, **503 `prevuelo_no_disponible`** (no 429: `limite_de_activos` afirmaría un cupo lleno que nadie midió). El conteo temprano sigue, sólo para no planificar ni sondear en vano. **Evidencia:** `tests/test_jacobs_candado_activos_db.py` (dos sesiones no superan el cupo — por mutación con un nombre de candado por sesión: `assert 2 == 1`; GET_LOCK que vence a 1 s; se suelta aunque el bloque lance; nombre por base; EXPLAIN sin tablas y recuento por `idx_pipelines_status`), y wiring puro en `tests/test_jacobs_preflight_endpoint.py`/`tests/test_jacobs_continuar.py` (recuento bajo el candado con el cupo que otro proceso llenó → 422 al crear / 429 al continuar; escritura dentro del candado; candado no disponible → 503 / se propaga sin escribir).

- **Cerrado — la sonda del pre-vuelo que vence sin registrar uso (antes "Anotado", Ruling R14; ola final F4, Ruling R30).** `jacobs/sonda.py::_registrar` sólo registraba uso con tokens medidos > 0: una sonda que VENCÍA o un 2xx SIN campo de uso no dejaba fila en `axioma_usage` aunque el proveedor la hubiera podido cobrar. **Arreglo:** esos dos casos se registran con tokens **estimados** — entrada ⌈`len(MENSAJE_DE_SONDA)` / `JAX_PREVUELO_CHARS_POR_TOKEN`⌉, salida el tope que pidió la sonda — y `request_type='preflight_probe_est'` (19 caracteres; `axioma_usage.request_type` es `VARCHAR(20)` y no tiene otra columna para marcar una estimación sin DDL). Con tokens medidos sigue `preflight_probe`. Se decide por si el pedido SALIÓ (pasada final R34): no registran `ConnectError`/`ConnectTimeout`/`PoolTimeout`/`UnsupportedProtocol`, un 4xx, la falla local (`config_error`) ni la base caída al resolver; registran estimado `ReadTimeout`/`WriteTimeout`, `ReadError`, `RemoteProtocolError`, cualquier 5xx (504/524 incluidos) y el timeout propio de la sonda. **Evidencia:** `tests/test_prevuelo_sonda.py` (+9: vence por `httpx` y por `wait_for`, 2xx sin usage en openai-compat, gemini y motor, controles con usage medido, falla local y base caída; rojo contra 4768484: 6 failed) y `tests/test_prevuelo_catalogo_db.py` (+1: los dos `request_type` entran en la columna real). **Aviso a la Mesa:** Admin → Costos agrupa por `request_type`, así que aparece la fila nueva `preflight_probe_est`.
## Retiro de la voz (2026-09-17) — Kokoro TTS y Whisper fuera del árbol

**DECISIÓN de Fernando, 2026-09-17.** Se retira la voz. No se reinstala hasta que
haya un plan de voz de verdad.

**Qué se retiró:**
- `jax/voice/` completo: `tts.py` (VoiceEngine/Kokoro), `ears.py` (EarEngine),
  `kokoro_worker.py`, `whisper_worker.py`, `__init__.py`.
- La puerta de arranque de `jax/core/main.py::main()`
  (`from jax.voice.tts import _python_de_kokoro` + su llamada). **`url_requerida
  ("JAX_OLLAMA_URL")`, en esa misma función, SE QUEDA:** esa variable sí está viva.
- Los comandos del REPL `/voz on`, `/voz off`, `/callate`, `/silencio`, `/escucha`,
  la locución de `run_task` y la del easter egg.
- Las claves `voice_id` / `voice_speed` de las 7 facetas en `config/config.toml`
  (nadie más las leía) y la columna "Voz" de la tabla de facetas de `CONTEXT.md`.
- La variable de entorno `JAX_KOKORO_PYTHON` (E-21). Ya no se necesita en
  `/etc/jax/.env`, y el pendiente de deploy que la pedía quedó corregido arriba.
- Los tests que describían la voz: `test_la_voz_toma_el_python_de_kokoro_del_entorno`
  (`tests/test_config_entorno.py`) y las 4 entradas de `_FUERA_DE_REQUIREMENTS`
  (`faster_whisper`, `kokoro`, `numpy`, `soundfile`) en
  `tests/test_requirements_completos.py`.
- De `requirements.txt` no salió nada: la voz nunca estuvo ahí (corría en su propio
  venv, por eso las 4 entradas vivían en la lista de excluidos).

**Por qué (VERDAD OPERACIONAL medida el 2026-09-17, no supuesta):**
1. El venv de la voz **no existe**: `~/kokoro-test` no está, no hay modelos en
   `/srv/jax-data/`, y `python3 -c "import kokoro"` da `ModuleNotFoundError`.
2. `JAX_KOKORO_PYTHON` **no está** en `/etc/jax/.env`.
3. `jax/core/main.py::main()` llamaba `_python_de_kokoro()` →
   `ruta_absoluta_requerida("JAX_KOKORO_PYTHON")`, que falla cerrado si la variable
   no está. **Resultado: el REPL de JAX no arrancaba**, y no por un problema de
   JAX sino por la puerta de una funcionalidad que no podía funcionar de todos
   modos. Un fail-closed correcto custodiando algo que ya no existe es una puerta
   de una casa demolida.

**Dónde queda la última versión funcional (para recuperarla):**
- Último commit que tocó `jax/voice/`: **`3b2887d`** (E-21, 2026-09-16).
- Último commit con `jax/voice/` presente en el árbol: **`351ec95`** (master al
  momento del retiro). Recuperación:
  `git checkout 351ec95 -- jax/voice/` — y `git show 351ec95:jax/core/main.py`
  para el cableado del REPL.

**Qué hace falta para traerla de vuelta (no es un `git checkout` y ya):**
1. Un plan de voz escrito: para qué, en qué camino de usuario, y quién la apaga.
2. Venv propio y reproducible, con `kokoro` / `faster-whisper` / `soundfile` /
   `numpy` fijados, y su ruta declarada en `/etc/jax/.env`.
3. Los modelos en disco, en una ruta bajo control (no `~/kokoro-test`).
4. Una prueba que se ejercite de verdad — Principio VII: un freno sin prueba no es
   freno; una voz sin prueba es una puerta más que rompe el arranque.
5. **La puerta de arranque NO vuelve a `main()`.** Si la voz vuelve, se resuelve
   perezosamente, al primer uso, y su falla degrada la voz, no el REPL.

**Controles del retiro:** `tests/test_retiro_de_la_voz.py` (el paquete no está, el
arranque no exige la variable, ningún módulo de servicio importa la voz). Los tres
fallan contra `351ec95`.
## Cerrado en código, merge y despliegue pendientes — UN solo mecanismo de cupo (2026-09-17)

Rama `perf/candado-creacion-a-la-base`, con merge-forward a `origin/master` `b450248` (jax#209).
**NO mergeada, NO desplegada.** Decisión de Fernando (arreglar de raíz antes del frente G) y del
coordinador (queda un solo mecanismo, y es el INSERT condicionado).

**LA REGLA, que reemplaza a dos candados: la condición del cupo viaja DENTRO de la escritura que lo
consume.** Un candado hay que acordarse de pedirlo; una condición adentro del INSERT o del UPDATE
se aplica sola aunque quien escriba un camino nuevo no sepa que hay un límite.

**Lo que había.** Dos mecanismos para el mismo invariante: `jacobs/candado.py::candado_de_creacion`
(un `asyncio.Lock` de proceso) y `store.candado_de_activos()` (un `GET_LOCK` del servidor MariaDB,
jax#209, que existía porque el de proceso no cruzaba al CLI). Los dos hacían cumplir el límite y los
dos serializaban: techo de ~43 delegaciones/s y la Mesa esperando detrás de Ada.

**EL HALLAZGO que amplió el alcance.** `resume` y `approve-step` mueven un pipeline de
`interrupted` —que NO cuenta como activo— a correr, **sin mirar el cupo**: ninguno de los dos
candados los tomaba. Con tres interrumpidos y tres `resume` se pasaba el límite. No es de jax#209 ni
de esta rama: es anterior. Un límite que dice que existe y no existe es peor que no tenerlo, porque
alguien dimensionó GPU, pool y techos de costo creyendo que como mucho hay tres.
**Verificado ROJO POR COMPORTAMIENTO contra `b450248`**, con un canario contra la base real: con el
cupo LLENO, `pipeline_tomar_epoca` admitió el resume y devolvió época 1 (`1 is not None`).

**Por qué no se podía mergear a medias.** Con crear usando el INSERT y continuar el GET_LOCK, los
dos no se ven: continuar cuenta 2 bajo su candado, crear inserta contando 2 filas **commiteadas**,
los dos commitean y quedan **4 activos con el límite en 3**. Dos mecanismos para un invariante es
peor que cualquiera de los dos solo.

**Cómo queda — los CUATRO caminos que ocupan cupo:**

| camino | forma | dónde va la condición |
|---|---|---|
| crear | INSERT | `cupo.SQL_RESERVAR`, reserva ANTES de planificar |
| continue | UPDATE | `store._SQL_PIPELINE_CONTINUAR`, JOIN con la derivada del recuento |
| resume | UPDATE | `store.pipeline_tomar_epoca(cupo_maximo=...)` |
| approve-step | UPDATE | ídem |

La partición de `PipelineStatus` y la lista de estados viven **una sola vez**, en `jacobs/policy.py`,
y de ahí salen las cuatro sentencias **y el recuento de `store`**. `interrupted` NO ocupa cupo:
semántica heredada, declarada, no cambiada acá.

**Verificado contra MariaDB 12.3** antes de escribir el código: 10, 25 y 50 reanimaciones
concurrentes respetan el cupo exacto con las dos formas de UPDATE (subconsulta directa y JOIN con
derivada; se eligió el JOIN por portabilidad, CI corre 11.8), y **50 creaciones CRUZADAS con 50
reanimaciones** —la carrera que rompía tener dos mecanismos— también, con cero errores.

**Deadlock: calibrado midiendo, no estimando.** El `INSERT`/`UPDATE` lee la tabla en la que escribe,
así que dos escrituras simultáneas se traban (1213) — y esos candados son justo lo que las hace
correctas. Se reintenta: 5 reintentos se agotaron 711 veces bajo carga; 12 dejaron 1 de cada
~530.000; con la unión (el UPDATE que completa la reserva pelea por los mismos candados de rango),
12 dejaron 23 de ~250.000. **Con 24 y espera creciente hasta 100 ms: cero.** Agotarlos levanta el
error, nunca devuelve "reservado" sin fila.

**Retirados con el candado:** `jacobs/candado.py`, `store.candado_de_activos`, `CandadoNoDisponible`,
`JAX_PREVUELO_CANDADO_TIMEOUT_S` + `candado_timeout_s()` (una variable que no lee nadie es una
trampa) y `tests/test_jacobs_candado_activos_db.py`. **Crear dejó de necesitar una conexión
DEDICADA** y sale de la lista de excepciones al pool: era la única que estaba ahí por ESPERAR.

**EXPLAIN de las sentencias REALES.** El COUNT del cupo va por `idx_pipelines_status` (ya existía:
sin migración). El UPDATE de continuar: la fila del pipeline por `PRIMARY`, el COUNT por
`idx_pipelines_status`, y la tabla derivada materializada con una fila. Sin `filesort` ni scans.

**Carga (2026-09-17, hall9000, k6 v2.2.0, instancias AISLADAS sobre `jax_memory_test`).** Antes =
`b450248`; después = la rama sobre ese mismo árbol. Mismo arnés, mismos offsets de token, 5 s de
rampa + 20 s sostenidos + 5 s de bajada, pool 10, **cupo efectivo 2 de 3** en las dos columnas (el
padre `running` que el arnés necesita ocupa un lugar; el pre-vuelo del arnés lo exige y aborta si
hay más — abortó dos veces de verdad, por filas vivas que dejaron las suites de base).

`JAX_CARGA_PLAN_MS=300` (planificador representado; el real tarda 1,3-8,7 s) — **la tabla que
prueba la tesis**:

| escenario | VUs | rps antes | p95 antes | rps después | p95 después |
|---|---|---|---|---|---|
| Mesa | 10 | 3,3 | 3.071,67 ms | **3.708,3** | **2,24 ms** |
| Mesa | 25 | 3,3 | 7.667,85 ms | **4.083,9** | **6,25 ms** |
| Ada (hijo con token real) | 10 | 3,2 | 3.083,17 ms | **3.778,7** | **2,07 ms** |
| Ada | 25 | 3,3 | 7.693,00 ms | **4.073,6** | **6,01 ms** |
| **Mesa (5 VUs) MIENTRAS Ada delega a 10** | 5 | 1,0 | 4.614,49 ms | **1.299,5** | **3,67 ms** |
| **Mesa (5 VUs) MIENTRAS Ada delega a 25** | 5 | **0,5** | **9.233,50 ms** | **1.146,1** | **7,61 ms** |

`JAX_CARGA_PLAN_MS=0` (plan instantáneo):

| escenario | VUs | rps antes | p95 antes | rps después | p95 después |
|---|---|---|---|---|---|
| Mesa | 10 | 330,3 | 42,01 ms | **1.199,6** | **22,62 ms** |
| Mesa | 25 | 311,4 | 97,20 ms | **1.375,8** | **43,66 ms** |
| Ada | 10 | 213,7 | 56,78 ms | **1.461,1** | **20,69 ms** |
| Ada | 25 | 160,1 | 324,78 ms | **2.642,1** | **21,88 ms** |
| **Mesa (5 VUs) con Ada a 10** | 5 | 104,5 | 64,11 ms | **837,9** | **15,66 ms** |
| **Mesa (5 VUs) con Ada a 25** | 5 | 56,2 | 140,04 ms | **564,6** | **19,64 ms** |

**El techo del candado se ve desnudo: 3,3 rps = exactamente 1/0,300 s.** Toda la creación, la de la
Mesa y la de Ada, pasaba por un solo planificador a la vez.

**¿La Mesa dejó de esperar detrás de Ada? SÍ.** Con el planificador representado y Ada delegando a
25 VUs: de **0,5 rps y p95 9,23 s** a **1.146,1 rps y p95 7,61 ms**.

**Degradación.** Antes, con el planificador representado, la Mesa no aguanta ni c=5 (p95 4,6 s
detrás de Ada). Después no se degrada en el rango medido: p95 2,24 → 6,25 ms de c=10 a c=25, muy por
debajo del umbral de 500 ms del arnés.

**Errores: CERO 500 en las cuatro corridas.** Con el plan instantáneo salieron **21 respuestas 503
`contencion_al_reservar` de ~250.000** (0,008 %): es el comportamiento nuevo funcionando — la
contención se contesta "volvé a intentar" con `Retry-After`, en vez de un 500 que manda a buscar un
defecto que no existe. Ninguna esperó más de 1 s (14-16 intentos, 0,90-1,00 s: el presupuesto
declarado haciéndose cumplir). Con el planificador representado, cero de todo. **Caduca** si cambia el esquema, el
volumen de datos, `MAX_PARALLEL_PIPELINES` o la infraestructura.

**Revisión del autor del mecanismo retirado (2026-09-17) — los cinco puntos:**

1. **El cupo se mira ANTES del pre-vuelo.** El pre-vuelo SONDEA facetas: es una llamada PAGA. Con el
   cupo revisado después, un `resume` rechazado por falta de lugar ya había gastado dinero.
   `routes._cupo_o_429()` es una compuerta barata antes de cualquier cosa que salga a la red; **no
   decide** (eso sigue en la condición del UPDATE), sólo ahorra el gasto. El orden está fijado por
   test: por posición en el código **y** por comportamiento (con el cupo lleno el pre-vuelo no se
   llama).
2. **La fila visible antes de planificar: averiguado, no supuesto.** Nace `pending`, con `plan='[]'`
   y `owner_ack_at` NULL. La consulta que lista los pipelines de la Mesa
   (`jax-platform ... SQL_PIPELINES_DEL_USUARIO`) filtra por `owner_ack_at IS NOT NULL`, y esa marca
   la escribe jax-platform **después** de que Jacobs responde: **no hay pipeline fantasma en la lista
   de nadie** durante la planificación (un hijo de Ada ni siquiera tiene dueño). Si el proceso muere
   ahí, la cosecha el reaper a los **300 s** — mismo mecanismo y mismo número que antes; lo que crece
   es la ventana (de milisegundos a los segundos que tarda planificar), no el plazo. **Declarado, no
   arreglado**: bajar los 300 s arriesgaría cosechar pendientes legítimos.
3. **La contención ya no es un 500.** Sale **503 `contencion_al_reservar`** con `Retry-After: 1` — no
   el 422 del cupo, que diría que el pedido es inválido, y no lo es. **Techo real de espera, medido:**
   sin presupuesto, 24 intentos con espera hasta 100 ms dan **2,06 s** en el camino del usuario; se
   declara `PRESUPUESTO_DE_ESPERA_SEGUNDOS = 1,0` y se hace cumplir (en la carga cortó a los 14-16
   intentos con 0,90-1,00 s). El cálculo está fijado por test. La traducción del código en la Mesa va
   en jax-platform, rama `feat/contencion-al-reservar` (es.js y en.js, más la lista de códigos de
   `errores.test.js`, que exige texto en los dos idiomas).
4. **Las CUATRO sentencias tienen test con base real**, y tres de los cuatro en el MISMO job:
   | sentencia | test | job |
   |---|---|---|
   | INSERT de la reserva (crear) | `jacobs/_cupo_io_test.py::CupoEnLaBaseTest` (25 y 50 corrutinas, EXPLAIN de la sentencia real, soltar, completar, fail-closed) y `::CreacionConcurrenteSinCandadoTest` (la ruta completa) | `subpipeline-contrato-db` |
   | UPDATE que revive (continue) | `jacobs/_cupo_io_test.py::ContinuarRespetaElCupoTest` (rechaza con el cupo lleno, admite con lugar) | `subpipeline-contrato-db` |
   | UPDATE de la época (resume y approve-step: **la misma sentencia**) | `jacobs/_cupo_io_test.py::ReanudarRespetaElCupoTest` (el hallazgo, rojo por comportamiento contra `b450248`) | `subpipeline-contrato-db` |
   | recuento del cupo | las de arriba (`cupo.activos()` en cada `asyncSetUp`) + `tests/test_jacobs_continuar_db.py` (EXPLAIN con el JOIN, y 0 filas → `CupoAgotado`) | `subpipeline-contrato-db` y `jacobs-gobernanza-db` |
   Que `resume` y `approve-step` usen esa sentencia lo fija `tests/test_cupo_en_todos_los_caminos.py`
   (puro): los dos endpoints pasan `cupo_maximo` y traducen `CupoAgotado`.
5. **`run_epoch` NO cambió de dueño**: anotado en `jacobs/store.py`, sobre las sentencias de época. El
   cupo les agregó una condición; quién incrementa la época, cuándo y con qué CAS sigue siendo
   continuar, resume y approve-step.

**Pisos de CI (MEDICIONES LOCALES; manda el runner):** `tests-puros` 2020 → **2049**;
`subpipeline-contrato-db` 121 → **135**; `jacobs-gobernanza-db` 94 → **87** (BAJA porque se retira el
test del GET_LOCK junto con el GET_LOCK; bajar un piso sólo se justifica cuando se retira
funcionalidad y se dice). Detector P10 en cero violaciones.

**El rojo de CI del 2026-09-17 y su diagnóstico (no era ni el cupo ni la versión).**
`CreacionConcurrenteSinCandadoTest::test_diez_creaciones_a_la_vez_admiten_exactamente_el_cupo` falló
en el runner con `AssertionError: 0 != 3` — admitió cero — y en local daba 3.

- **Descartado que sea de versión, con evidencia.** MariaDB 11.8 efímero en Docker (la del runner;
  producción corre 12.3.3): las dos formas de UPDATE condicionado y la carrera de 50 creaciones
  cruzadas con 50 reanimaciones dan el cupo EXACTO con cero errores, **igual en 11.8 que en 12.3**.
  Y el job entero (135 passed) pasa en esa 11.8 recién creada. La diferencia de optimizador que otra
  sesión midió ese día (el `GROUP BY` de una derivada ordena en 11.8 y no en 12.3) **no afecta a
  estas sentencias**: no dependen del orden, sólo del `COUNT`.
- **Descartado que el cupo estuviera tomado.** La precondición de la clase ya exigía
  `cupo.activos() == 0`; el error vino de la medición, así que la base arrancó limpia.
- **La causa real: el test medía la gobernanza, no el cupo.** `create_pipeline` corre el pre-vuelo
  (jax#209), que lee `facet`, `model`, `capability` y `credential` — tablas de las migraciones de
  **jax-platform**, no de `store.init_tables()`. Sin sustituirlo, las diez creaciones morían con 503
  `prevuelo_no_disponible`. En la `jax_memory_test` compartida de hall9000 esas tablas existen porque
  las sembró otro job: **el verde local dependía de un estado que el test no fijaba.**
- **Arreglo:** se sustituye el pre-vuelo (este test mide el cupo; el pre-vuelo tiene sus tests), las
  aserciones van en orden de diagnóstico (primero "todo rechazo es el 422 del cupo", después las
  cuentas) y la precondición aborta ruidosamente **con el censo de la tabla**
  (`SELECT status, COUNT(*) ... GROUP BY status`) en el mensaje, en las tres clases. Validado por
  mutación: sin el sustituto nombra la tabla que falta; con el cupo desactivado cae `10 != 3`; con
  tres `running` ajenos aborta con `3 != 0 ... [censo: completed=81, running=3]`.

- **PENDIENTE con fecha:**
  - [ ] **2026-09-17** Orden de merge contra el frente G: agrega `queued`, `awaiting_approval` y
        `waiting_children`. **Decidir cuáles ocupan cupo** y clasificarlos en
        `policy.ESTADOS_QUE_OCUPAN_CUPO` / `ESTADOS_SIN_CUPO`. Ya no se puede olvidar en silencio:
        la partición es exhaustiva y hay controles que se ponen rojos solos (validado por mutación
        con `queued`), más uno contra la tabla real. Decisión de Fernando.
  - [ ] **2026-09-17** El CI corre MariaDB **11.8** y producción **12.3.3**. En este trabajo se
        comprobó que para las sentencias del cupo las dos versiones coinciden, pero ese mismo día
        otra sesión midió una diferencia REAL de optimizador entre ellas (el `GROUP BY` de una
        derivada ordena en 11.8 y no en 12.3). Decidir si el CI se sube a 12.3: es una decisión de
        infraestructura, no un parche de SQL.
  - [ ] **2026-09-17** Las suites de base dejan pipelines VIVOS en `jax_memory_test`
        (`arnes-ada-padre`, `secreto de B`, `causa running`): ocupan cupo y hacen abortar cualquier
        medición posterior. El pre-vuelo del arnés los detecta, pero la limpieza es a mano. Cerrarlos
        en el `addAsyncCleanup` de cada suite.

## Cerrado en código, merge y despliegue pendientes — human gate de LAS MANOS sin emisión HTTP (2026-09-17)

- **HECHO (medido 2026-09-17, Mr. Hyde, rama `fix/human-gate-sin-auth` desde `c13d066`):** `POST /human_gate/token`
  (sin autenticación) devolvía 200 con un token a cualquier proceso local (TestClient sobre `server.app` de master: 200
  `token/ttl_seconds/expires_epoch`). Además `/motor/dispatch` aceptaba como `human_gate_token` cualquier string no vacío
  (`policy.py` sólo miraba presencia): `code_swarm` y `bug_hunt` (`requires_human_gate=1`) se autoaprobaban con `"x"`.
- **Quién la usaba:** nadie. Ningún código de jax ni de jax-platform (backend o frontend) llama la ruta; el log
  `las_manos/logs/gate.jsonl` no existía (nunca se emitió un token en producción) y `audit.jsonl` no tiene ningún
  evento `HUMAN_GATE`. El gate de pipelines de la UI (`approve-step`) es otro mecanismo y no se toca acá.
- **DECISIÓN (autonomía de Fernando, 2026-09-17):** el patrón del frente F. Tabla `las_manos_human_gate_tokens`
  (`init_tables()`): sólo sha256 (PK), `emitido_por`, `vence_at`, `usado_at`, `usado_en`; consumo atómico
  `UPDATE … WHERE usado_at IS NULL AND vence_at > now` (`las_manos/human_gate.py`). **Sin ruta HTTP de emisión:** la
  emite `las_manos/emitir_token_gate.py`, que necesita la credencial de la base (`/etc/jax/.env`, root:fruiz 0660).
  La jaula de Hyde no monta `/etc/jax` y `axioma` no la lee (`Permission denied`, verificado). `/execute` y
  `/motor/dispatch` consumen contra la base; el motor consume DESPUÉS de la política (un rechazo no quema el token).
  `[human_gate]` queda sólo con `token_ttl_seconds` (rango [10, 3600], validado al importar `server.py`).
- **Uso para el aprobador humano:** `set -a; . <(sudo -n cat /etc/jax/.env); set +a; ~/jax/las_manos/.venv/bin/python ~/jax/las_manos/emitir_token_gate.py`
  → imprime el token (una vez) y el vencimiento; va en `approval_token` o `human_gate_token`.
- **Pruebas:** `tests/test_human_gate_sin_emision_http.py` (24, tests-puros, piso 1127→1151) y
  `las_manos/_human_gate_io_test.py` (8, subpipeline-contrato-db, piso 101→109). Vistos en rojo contra master; mutaciones:
  reabrir la ruta → 3 rojos, quitar el consumo del motor → 2 rojos, quitar `usado_at IS NULL` → 2 rojos (un uso, carrera).
- **PENDIENTE 2026-09-18:** PR, CI, merge; deploy = reiniciar `jax-las-manos` (crea la tabla en el arranque, sin
  migración manual; comprobar antes `ActiveEnterTimestamp` contra los commits del checkout). Sin cambios en jax-platform.
- **CORREGIDO 2026-09-17 (Mr. Hyde, ver "LAS MANOS autentica a sus llamadores"):** ~~Sigue abierto (no es este arreglo), fecha 2026-09-24:~~ LAS MANOS no autenticaba al llamador. `facet_id`/`caller`/
  `invoked_by` son declarados por el cuerpo: `POST /jacobs/pipeline/{id}/approve-step` con `invoked_by="plataforma"`
  aprueba steps de Hyde bloqueados en human gate desde cualquier proceso local; `/execute` con `facet_id="hyde"` ejecuta
  en staging sin gate; `/motor/dispatch` elige `caller`. Ver la entrada de Hyde en la sección de C3.

## Cerrado en código, merge y despliegue pendientes — LAS MANOS autentica a sus llamadores (2026-09-17)

- **HECHO (medido 2026-09-17, Mr. Hyde, rama `fix/las-manos-auth-servicio` sobre `fix/human-gate-sin-auth`):** con
  `server.app` real de `c7d59cc` y la base sustituida, `POST /jacobs/pipeline/{id}/approve-step` SIN credencial y
  `{"invoked_by": "plataforma"}` → 200 y el paso de Hyde en `blocked_human_gate` pasaba a `pending`. La jaula de Hyde
  tiene red local (`--share-net`): Hyde se podía autoaprobar.
- **Llamadores reales (grep en jax y jax-platform + journal de 7 días de `jax-las-manos`):** jax-platform
  (`api/pipelines.py`: crear, leer, results, resume, cancel; `jax_engine/state.py`: sondeo de pipelines y `/health`;
  `api/chat.py`: `/motor/authorize-facet`; tablero: `/health`) y Jacobs dentro del propio proceso
  (`jacobs/executor.py`: `/motor/dispatch`, `/motor/job/{id}`, `/cancel`). **Nadie** llama `/execute`, `/plan` de LAS
  MANOS, `/audit/tail` ni `approve-step` (la UI de hoy no aprueba pasos en gate: sólo `resume`). El REPL no llama a
  LAS MANOS. El proxy del Ejecutor tampoco.
- **DECISIÓN (autonomía de Fernando, 2026-09-17): credencial de servicio, no SO_PEERCRED.** Hyde, jax-platform y LAS
  MANOS corren como `fruiz`: el uid del par no distingue a Hyde. Lo que los distingue es qué pueden leer: la jaula no
  monta `/etc/jax`, corre con `--clearenv` y `--unshare-all` (PID propio: no ve `/proc/<pid>/environ` de los
  servicios). `las_manos/auth_servicio.py`: middleware ASGI **deny by default** (público sólo `GET /health`), cabecera
  `X-Jax-Credencial-Servicio`, `hmac.compare_digest` contra todas, identidades `plataforma`
  (`JAX_LAS_MANOS_CREDENCIAL_PLATAFORMA`: `/jacobs/*` + `POST /motor/authorize-facet`; declara `invoked_by=plataforma`,
  `caller=jax_platform_chat`) y `jacobs` (`JAX_LAS_MANOS_CREDENCIAL_JACOBS`: `/motor/dispatch`, `/motor/job/*`,
  `POST /jacobs/pipeline`; declara `caller=jacobs`, `invoked_by=ada`). Un `invoked_by`/`caller` del cuerpo que no es de
  la credencial → 403 antes de la ruta; claves duplicadas → 400. **Sólo `plataforma` aprueba o reanuda.** Sin las
  variables (o cortas, o iguales) LAS MANOS no arranca. `/execute`, `/plan` y `/audit/tail` no los alcanza ninguna
  identidad. Rechazos con `code`, sin prosa. Una ruta nueva bajo `/jacobs/` queda sólo para `plataforma` sin tocar nada.
- **jax-platform:** `backend/credencial_las_manos.py::encabezados_las_manos()` en cada pedido a LAS MANOS salvo
  `/health` (rama `fix/las-manos-credencial-servicio`); guard AST en `tests/test_credencial_las_manos.py`.
- **Pruebas:** `tests/test_las_manos_auth_servicio.py` (37, tests-puros 1151 → 1188). Mutaciones en policy.yml.
  Carga (middleware aislado, 25 VUs, 15 s, hall9000): cuerpo de 1 KB p95 3,19 → 3,16 ms (10054 → 8650 req/s); cuerpo
  de 200 KB (peor caso: prompt armado de `/motor/dispatch`) p95 7,91 → 9,47 ms (4133 → 2647 req/s). Tráfico real:
  ~13 dispatch/semana.
- **Despliegue (PENDIENTE 2026-09-18):** agregar las dos variables a `/etc/jax/.env` (`secrets.token_urlsafe(32)`,
  distintas) → mergear y reiniciar **jax-platform primero** (manda la cabecera; LAS MANOS viejo la ignora) →
  mergear y reiniciar `jax-las-manos` con 0 pipelines en vuelo. Comprobar `ActiveEnterTimestamp` contra los commits.
- **Sigue abierto, fecha 2026-09-24:** (1) un proceso de `fruiz` FUERA de la jaula lee `/etc/jax/.env` (root:fruiz
  660) y cualquier `/proc/<pid>/environ` de `fruiz`: contra eso no hay credencial que alcance; el cierre es otra
  cuenta de servicio para LAS MANOS/jax-platform. (2) frente G (emisor de sub-pipelines de Ada) tiene que mandar
  `headers=encabezado_propio(IDENTIDAD_JACOBS)`. (3) los apps de carga de `loadtest/` montan los routers sin
  `proteger()`.
- **CORREGIDO 2026-09-17 (Mr. Hyde, verificado):**
  - **(3) CERRADO.** `loadtest/authorize_facet_app.py` y `loadtest/jacobs_subpipelines_app.py` llaman `proteger(app)`;
    los k6 mandan `X-Jax-Credencial-Servicio` desde `__ENV.CREDENCIAL` y abortan sin ella. Guardia
    `tests/test_loadtest_apps_protegidas.py` (AST; vista en rojo, mutación → 1 rojo). Medido contra `jax_memory_test`:
    sin credencial 401, identidad ajena 403, propia pasa; `authorize_facet.js` a 25 VUs con los tres caminos
    (permitido / fail-closed / caller ajeno) p95 5,81 ms, 6140 req/s, 100 % checks, 0 fallas.
  - **(2) CERRADO como propiedad.** No hay en el árbol un emisor de sub-pipelines de Ada que llame
    `POST /jacobs/pipeline` (`grep` sin coincidencias fuera de tests/loadtest). Los tres llamados de `jacobs/executor.py`
    ya mandan `encabezado_propio(IDENTIDAD_JACOBS)`. Un emisor futuro sin cabecera recibe 401 (el middleware es
    deny-by-default), así que no puede quedar abierto en silencio.
  - **(1) DECISIÓN DE FERNANDO, no deuda técnica.** Medido: `fruiz` tiene `(ALL : ALL) ALL` en sudo (con contraseña y
    caché de sesión). Una cuenta de servicio para LAS MANOS/jax-platform con `.env` `root:jaxsvc 640` sólo cierra el
    caso "proceso de `fruiz` sin sudo en caché"; cambia cómo operan y despliegan TODAS las sesiones de Mr. Hyde
    (checkouts, venvs, `/etc/jax/.env`, unidades). La jaula de Hyde no monta `.env` y el Ejecutor corre como `axioma`
    (sin acceso), que son los procesos no-operador. Queda como decisión del operador, con este análisis.

## Cerrado en código, merge y despliegue pendientes — Ejecutor SP1 plan 2: C3 registro intocable y cerco (2026-09-17)

Detalle y pruebas de «visto fallar» en CONTEXT.md §9 (2026-09-17 ~11:45). Rama local `feat/ejecutor-c3`
(`/home/fruiz/worktrees/jax-sp1-c3`), apilada sobre `feat/ejecutor-c1-c2` (PR #180), SIN PUBLICAR (un subagente
no puede escribir en el remoto). Orden: después de #180 y de su despliegue (el cerco sale del inventario de la
política que exporta el plan 1).

- [ ] **2026-09-18** Publicar la rama y abrir el PR (base `feat/ejecutor-c1-c2` o `master` si #180 ya entró); confirmar
  en el log del runner `1038 passed, 1 skipped` (tests-puros); si difiere, manda el runner y se corrige la línea de historia.
- [ ] **2026-09-18** Canario rojo por API (plan 2, Task 6 Step 4): en `_devolver`, mover `await _enviar(conn, writer,
  h11.Data(data=trozo))` antes del `try` que anota; `tests-puros` = `failure` sobre ese sha; revert = `success`.
- [ ] **2026-09-18** Despliegue (plan 2, Task 7 Steps 2–5): `/etc/jax/.env` con backup y sudoedit (`JAX_EJECUTOR_REGISTRO`,
  `JAX_EJECUTOR_CERCO_SONDAS`, `JAX_PROXY_CARRIL_{UPSTREAM,RAIZ,TOPE_S=90,PUERTO=18435,HOST}`), dos lectores idénticos;
  `ops/ejecutor/instalar_registro_y_cerco.sh` (respalda iptables-save/ip6tables-save y prueba la restauración en un
  netns ANTES de tocar la red); `probar_c3.py` → `c3_vivo=true`; `probar_c3_corte.py` → `c3_corta=true`; las cuatro
  roturas del Step 5; y verificar que siguen vivos Mesa, LAS MANOS :7777, Ollama :11434, MariaDB :3308, SSH :58291 de
  fruiz, la VM .11 y Sésamo .6. Rollback del cerco: `sudo nft delete table inet ejecutor_cerco`.

### Ejecutor SP1 plan 6 · arranque condicionado — publicar, instalar el vigía y lo que SP2 debe cumplir (2026-09-17, Mr. Hyde)

Rama `feat/ejecutor-arranque` (sobre master `6563cb4`), SIN PUBLICAR. `arranque.exigir_contratos`,
`vigia_servicio` + `ops/ejecutor/ejecutor-vigia@.service`, guardia de CI, `harness.correr` retirado.

- [ ] **2026-09-18** Publicar; confirmar pisos del runner: tests-puros `1486 passed, 1 skipped` (las dos listas),
  `no-naked-claude-subprocess` con `5 passed` en la guardia nueva. Canario rojo por API: un
  `jax/ejecutor/lanzador_de_prueba.py` con `remoto_claude` sin `exigir_contratos` ⇒ `no-naked-claude-subprocess` = failure; revert = success.
- [ ] **2026-09-18** Tras el merge, en el checkout de producción: `ops/ejecutor/instalar_vigia.sh` (la unidad plantilla; no
  arranca misiones). `/etc/jax/.env` YA tiene `JAX_EJECUTOR_VIGIA_LATIDO_CADA_S=5` y `JAX_EJECUTOR_MISIONES=/var/lib/jax-ejecutor-misiones`
  (backup `.env.backup-pre-ejecutor-arranque-20260917-080717`); el directorio existe (fruiz 0750).
- [ ] **RESERVADO A FERNANDO (ya estaba):** el arranque real hoy da `arrancaria=false` SÓLO por la parte remota de C4/C6
  (`freno_sin_remotos`; `llaves_no_son_de_root`, `sin_llave_del_freno`, `sin_revocador` en atemai, prod y bridge). Se
  cierra con `instalar_en_maquina.sh <m>` + `JAX_EJECUTOR_FRENO_REMOTOS` (DEUDA «plan 3»). Y toda misión: las cuatro
  máquinas tienen `con_datos_de_clientes=1` y la compuerta sigue en `false` ⇒ `auditor_no_admite_datos_de_clientes`.
- [x] **2026-09-24 · Obligaciones del transporte `harness` (SP2) que dejan vivos los contratos de SP1:**
  **CORREGIDO 2026-09-17 (Mr. Hyde):** 1–6 hechos y ejercitados en producción por SP2 (misión 9c2a9d8d desde
  `/api/ejecutor`: arranque verificado, vigía latiendo, afirmaciones con `proposito` citadas, `registro_cuadra`,
  `cadena_ok`, auditor legible, cierre del vigía). 7 no aplica hoy (el único cerebro es local; un cerebro de nube es
  una capacidad nueva con su propio proxy, no una obligación de SP2). 8 queda como está a propósito: `jax_local` es
  el cerebro medido en U3/V1–V4 y la faceta `ejecutor` sin binding hacía reventar C5; crearla sería una faceta
  duplicada del mismo modelo sin nada que la distinga.
  1. Lanzar sólo con `ejecutor-vigia@<mision>` activo y latiendo (`vigia_servicio` ya exige los contratos con los
     `hosts` reales); la guardia `test_ejecutor_lanza_solo_con_contratos.py` impone `exigir_contratos` a quien use
     `remoto_claude` o `vigilar`.
  2. El perfil `ejecutor` de `hyde_sandbox` conserva los montajes de solo lectura de `cuenta_axioma._jaula` y agrega
     `<workspace>/.claude` de solo lectura; NO monta credenciales de Anthropic.
  3. `ANTHROPIC_BASE_URL` = el proxy con carril (C3); ninguna otra URL de cerebro.
  4. Cada afirmación entregada lleva `proposito` y pasa por `auditor.aplicar_revision` antes de salir (C5).
  5. Fin de misión = `systemctl stop ejecutor-vigia@<mision>` (SIGTERM: audita lo pendiente y borra el latido).
  6. Informe final afirmación ↔ evidencia con las `Descartada` de citas y auditor.
  7. Kimi/GLM como cerebro: el proxy hoy tiene UN upstream (Ollama); un cerebro de nube necesita su propio proxy anotado.
  8. Crear la faceta `ejecutor` y devolver `ejecutor.cerebro_faceta` a `ejecutor` (hoy `jax_local`, ver abajo).
- [x] **2026-09-17 08:02** `axioma_config.ejecutor.cerebro_faceta` `ejecutor` → `jax_local` (la faceta `ejecutor` no tiene
  binding: C5 reventaba al resolverla). Dump `~/backups/axioma_config-pre-ejecutor-arranque-20260917-080235.sql`,
  restauración probada (md5 de las 21 filas igual en `jax_memory_test`). Reversión: `UPDATE axioma_config SET
  config_value='ejecutor' WHERE config_key='ejecutor.cerebro_faceta'`.

### Ejecutor SP1 plan 4 · C5 auditor en vivo — publicar, desplegar y DECIDIR (2026-09-17, Mr. Hyde)

Ramas SIN PUBLICAR: jax `feat/ejecutor-c5` (apilada sobre `feat/ejecutor-c3`), jax-platform `feat/ejecutor-config-c5`.
Orden: el PR de jax-platform se mergea ANTES (el job `jacobs-gobernanza-db` clona su master).

- [ ] **2026-09-18 · DECISIÓN RESERVADA A FERNANDO:** `ejecutor.c5_auditor_admite_datos_de_clientes` nace en `false`.
  Con cerebro local todo auditor de otro proveedor es nube; la Fase 0 prohibió que la nube vea datos de clientes.
  Opciones: (a) aceptar que el auditor de nube lea datos de clientes; (b) esperar un proveedor local distinto (Red
  Queen, Q3 2026); (c) auditar sólo misiones sin datos de clientes. Hasta decidir, una misión sobre .10/.11/.20 NO arranca.
- [ ] **2026-09-18** Publicar ambas ramas; confirmar pisos del runner: jax tests-puros `1135 passed, 1 skipped`,
  gobernanza-db `34 passed`; jax-platform con DB `1501`, sin DB `886`. Si difieren, manda el runner.
- [ ] **2026-09-18** Canario rojo por API: en `vigia.py`, `if not (isinstance(exc, asyncio.CancelledError) and fin.is_set()):`
  → `if False:` ⇒ `tests-puros` = failure; revert = success.
- [ ] **2026-09-18** Despliegue: `/etc/jax/.env` con backup y sudoedit: `JAX_EJECUTOR_PAUSA=/etc/jax/interruptor/EJECUTOR_PAUSA`
  (directorio del frente B), `JAX_EJECUTOR_VIGIA_LATIDO=/var/lib/jax-ejecutor/vigia.latido`,
  `JAX_EJECUTOR_VIGIA_LATIDO_MAX_S=30`. Sin ellas el proxy de C3 NO arranca (el instalador las exige). Tras
  desplegar jax-platform: `probar_c5.py --corridas 10 --cerebro jax_local` → `c5_vivo=true` con la config real.
- [x] **2026-09-17** Plan 3 (C4): el freno root actúa también con la pausa del Ejecutor (hecho, rama `feat/ejecutor-c4`).
- [x] **2026-09-17** Plan 6: `verificar_eleccion` y vigía con latido antes de abrir el proxy (rama `feat/ejecutor-arranque`).

### Ejecutor SP1 plan 3 · C4 freno en vuelo — publicar, desplegar el proxy y la parte remota (2026-09-17, Mr. Hyde)

Rama jax `feat/ejecutor-c4` (SIN PUBLICAR: la publica la sesión principal), desde `origin/master` (`f9c7073`,
frente B) con merge de `feat/ejecutor-c5` y `feat/ejecutor-c6` (sus PRs sin mergear). Si C5/C6 se rebasan antes,
los commits propios de C4 se reaplican encima.

- [ ] **2026-09-18** Publicar; piso del runner tests-puros `1273 passed, 1 skipped` (medido local 3.14); comprobar en
  el log que `test_ejecutor_freno_remoto.py` CORRE. Canario rojo por API: en `freno.py`,
  `salida["muertos"] = matar_pids(...)` → `salida["muertos"] = 0` ⇒ `tests-puros` = failure; revert = success.
- [ ] **2026-09-18** Tras mergear: reinstalar desde master `ops/ejecutor/instalar_freno.sh` (hoy corre desde la rama
  con `--rama-aprobada`, commit `f53b51b`; los archivos instalados son idénticos a la rama).
- [ ] **2026-09-18** Proxy (C4 en `proxy_carril.py`) sin desplegar: el proxy en vivo es el de C3 y el de la rama exige
  las variables de C5 (`JAX_EJECUTOR_VIGIA_LATIDO`, `_MAX_S`). Se despliega con C5.
- [ ] **2026-09-18 · RESERVADO A FERNANDO (servidores de clientes):** parte remota, máquina por máquina, en orden
  atemai → prod → bridge: `cd /home/fruiz/jax && set -a && . <(sudo -n cat /etc/jax/.env) && set +a &&
  ops/ejecutor/instalar_en_maquina.sh <m>` (SIN `--sin-freno`) `&& PYTHONPATH=.:las_manos python3
  scripts/ejecutor_contratos/probar_c6.py <m>`; luego agregar `<m>` a `JAX_EJECUTOR_FRENO_REMOTOS` (backup de `.env`),
  `sudo systemctl restart ejecutor-freno` y `PYTHONPATH=.:las_manos python3 scripts/ejecutor_contratos/probar_c4.py
  --remoto <m>` → `c4_vivo=true`. Rollback: `ops/ejecutor/revertir_en_maquina.sh <m>` y quitar `<m>` de la variable.
  Ensayado entero en contenedor (`scripts/ejecutor_contratos/probar_freno_remoto_en_contenedor.sh`, verde 3 veces).
  Hasta completarla, el latido dice `remotos_cargados=false` y el arranque (plan 6) no debe dejar ir a remotas.

### Hyde y cualquier proceso de `fruiz` alcanzan LAS MANOS sin autenticación — fecha: 2026-09-24

- **Hecho (medido 2026-09-17, Mr. Hyde, al planificar C3):** `127.0.0.1:7777` no pide credencial;
  `POST /human_gate/token` emite un token de aprobación a quien lo pida. `hyde_sandbox.py` usa `--share-net`:
  el `claude` de Hyde puede pedir un token y llamar `/execute`. El cerco de C3 lo cierra para la cuenta
  `axioma`, **no** para Hyde (corre como `fruiz`).
- **Por qué no se cierra en SP1:** es la misma propiedad (una jaula no puede autoaprobarse acciones), pero de
  otra faceta con otro dueño de proceso; el cerco por `meta skuid` no aplica. Opciones a decidir: token de
  servicio en LAS MANOS leído de `/etc/jax/.env` (que la jaula de Hyde no monta), o `--unshare-net` + proxy de
  salida para Hyde.
- **CORREGIDO 2026-09-17 (Mr. Hyde):** la emisión de tokens del human gate ya no tiene ruta HTTP (ver la sección
  "human gate de LAS MANOS sin emisión HTTP"). Lo que sigue abierto de esta entrada es la falta de autenticación del
  llamador (`approve-step`, `facet_id`, `caller` declarados por el cuerpo).
- **CORREGIDO 2026-09-17 (Mr. Hyde):** LAS MANOS exige credencial de servicio (sección "LAS MANOS autentica a sus
  llamadores"); lo que queda es un proceso de `fruiz` fuera de la jaula (ver allí).
- **Verificación de cierre:** desde un `claude` sandboxeado de Hyde, `curl -X POST 127.0.0.1:7777/human_gate/token`
  falla; el mismo pedido desde LAS MANOS/Jacobs funciona.

### El proxy del Ejecutor no fija el modelo: por `/v1/messages` la jaula puede pedir cualquiera — fecha: 2026-09-24 (SP2)

- **Hecho (medido 2026-09-17 al implementar C3):** el proxy deja pasar `POST /v1/messages` con el `model` que mande el
  arnés. Contra el Ollama de producción, un modelo distinto (o el mismo con otro contexto) desaloja al de la Mesa y a
  bge-m3 (V3, LEDGER SP3). El cerco y la lista de rutas no lo cubren.
- **Por qué no en C3:** el modelo permitido sale de `facet_binding` en vivo y el proxy no tiene DB; es el transporte
  `harness` de SP2 / el tope del proxy de SP3 (§6.2), que ya tienen que resolver el modelo del cerebro.
- **Verificación de cierre:** un `POST /v1/messages` con `model` distinto del del cerebro da 403 sin tocar el upstream.
- **CERRADO 2026-09-17 (SP3, desplegado):** `jax/ejecutor/proxy_carril.py` fija `JAX_PROXY_CARRIL_MODELO`
  (producción: `qwen3.6-mesa-131k`), tope de salida y rutas (`HEAD /api/hello`, `POST /v1/messages`). Tests
  `tests/test_ejecutor_proxy_modelo_y_rutas.py` (otro modelo, cuerpo ilegible, tope y GET arbitrario no llegan a
  Ollama). En vivo sin vigía el proxy responde 423 antes de mirar el modelo (la pausa va primero).

## CERRADO, DESPLEGADO Y PROBADO EN PRODUCCIÓN — frente B: kill switch real (2026-09-17)

**VERDAD OPERACIONAL 2026-09-17 ~15:40 CST (Mr. Hyde · sesión `fruiz-e5`; verificado con `git reflog`,
`journalctl`, `ps`, `sudo cat /proc/<pid>/environ`, `curl` y SQL).** El código del frente B **está vivo en
producción**. No lo desplegó esta sesión: el pull y el reinicio salieron de la migración a `jaxsvc` de
`fruiz-a1` (él lo confirmó; `fruiz-47` descartó ser el autor con el reloj de sus propios pulls).

- `git reflog` del checkout `/home/fruiz/jax`: `1490683 master@{15:22:50}: pull --ff-only`, después
  `046682f master@{15:30:30}`. `jax-las-manos` reiniciado 15:23:02 (journal), proceso 2272008,
  `cwd=/home/fruiz/jax/las_manos`, usuario `jaxsvc`.
- **El proceso corre `1490683` y el checkout está en `046682f`: los 2 commits de diferencia son de
  documentación** (`520f686` + el merge de #206). El código ejecutable es el actual. La comparación
  proceso-contra-checkout es la lección del incidente de las 04:46, y acá se hizo.
- `JAX_KILL_SWITCH_PATH=/etc/jax/interruptor/PAUSE` presente en `/proc/2272008/environ`.
- `/health` de LAS MANOS 200 con la forma de la auditoría fail-open:
  `{"status":"alive","kill_switch_active":false,"problemas":[],"comprobado_hace_s":0.0,"cache_ttl_s":5.0}`.
- `GET /api/admin/kill-switch` → `{"activo":false,"heredada":false,...}`: la ruta heredada `/etc/jax/PAUSE`
  no existe hoy.
- 0 líneas de `ERROR`/`PermissionError` en el journal desde el reinicio, corriendo como `jaxsvc`.
- El reinicio cumplió «0 pipelines en vuelo»: `jacobs_pipelines` sin ninguno activo (24 aborted,
  31 completed, 6 expired) y 0 turnos del Ejecutor `en_curso`.
- Sin `JAX_KOKORO_PYTHON` ni Whisper en el entorno: el retiro de la voz está adentro.

**VERDAD OPERACIONAL 2026-09-17 (Mr. Hyde, verificado con `stat`, `systemctl show` y `git log`).** Lado plataforma
MERGEADO Y EN PRODUCCIÓN: jax-platform#95 → `e715c28` (mergeado 04:23:49), `jax-platform.service` activo desde
04:46:16 y escribe el freno en `/etc/jax/interruptor/PAUSE`. Lado jax: rama `feat/kill-switch-real`
(`/home/fruiz/worktrees/jax-frente-b`), rebasada sobre `origin/master` `56d8c24` (frente E y Ejecutor C1/C2 adentro,
frente F NO), SIN publicar, SIN mergear, SIN desplegar: `jax-las-manos` sigue corriendo desde 03:23:03 con el código
que lee `/etc/jax/PAUSE`. Ledger: `jax-platform-frente-b/.superpowers/sdd/2026-09-16-frente-b-kill-switch/progress.md`.

**Qué hace el frente (lado jax):**
- **Módulo `interruptor` compartido** (`jax/core/interruptor.py`, symlink `las_manos/interruptor.py`, copia en
  `jax-platform/backend/interruptor.py`, familia `interruptor` de `scripts/check_mirror_sync.py`): la ruta del freno sale
  de `JAX_KILL_SWITCH_PATH` — OBLIGATORIA y absoluta (`InterruptorSinConfigurar` si falta); lectura fail-closed
  (cualquier `OSError` que no sea ENOENT = freno PUESTO); escritura atómica.
- **Lectores:** LAS MANOS (`server.py`, `ssh_worker`, `motor_registry` routes/worker — el job cancelado con el freno
  queda `killed_by_switch`), Jacobs (`policy.check_kill_switch` y cada step de `executor` corre bajo
  `correr_con_interruptor`, que frena en vuelo), REPL y `jax --task`. Ninguna ruta fija en el repo
  (`tests/test_interruptor_sin_rutas_fijas.py`).
- **La ruta heredada `/etc/jax/PAUSE` sigue frenando** mientras exista (fail-closed, WARNING por episodio): pedido de la
  sesión del Ejecutor `fruiz-a1`, cuyo contrato C4 (freno en vuelo) se construye encima. DECISIÓN completa en
  «Bloquea trabajo», arriba.
- **Canary de facetas y `probe_after_rebind` frenados** (lado plataforma, en #95): con el freno puesto devuelven
  `SALTADA_POR_FRENO`, sin llamar a proveedores ni escribir fila.
- **k6** `loadtest/kill-switch.js` (423 y admin, sólo con el freno puesto).

**Infraestructura (hecha 2026-09-17 ~04:05 por el controlador principal):** `/etc/jax/interruptor` `root:fruiz` `2770`;
`JAX_KILL_SWITCH_PATH` en `/etc/jax/.env` (backup `/etc/jax/.env.backup-pre-kill-switch-20260917-040558`).

**Incidente (HISTORIA, 2026-09-17 04:46):** otra sesión reinició `jax-platform` y desplegó #95 ANTES que el lado jax. Su
chequeo comparó el checkout contra master, pero no el proceso en marcha contra el checkout. **Exposición:** freno a
medias — el botón KILL de la Mesa escribe `/etc/jax/interruptor/PAUSE` y la Mesa responde 423, pero LAS MANOS y Jacobs
de producción siguen mirando `/etc/jax/PAUSE`: los pipelines en vuelo NO se frenan hasta desplegar esta rama.
**Lección: antes de reiniciar un servicio, comparar el proceso en marcha contra el checkout (`/proc/<pid>/cwd`,
`ActiveEnterTimestamp` contra la fecha del commit), no sólo el checkout contra master.**

**Rebase del 2026-09-17 sobre `56d8c24`:** conflictos en `.github/workflows/policy.yml` (listas de tests-puros: se
conservan las de E, Ejecutor y B; pisos re-medidos), `conftest.py` (aislamientos de E — URLs, `JAX_REPO_BASE`,
`JAX_FACET_SEAL_PATH` — más el del freno), `jacobs/policy.py` (E-13 `MAX_STEPS_PER_PIPELINE` desde `models.py` + el
freno desde `interruptor`) y `jax/core/main.py` (imports de E y del interruptor). El commit que sólo re-redactaba un
comentario de piso quedó vacío. **El tripwire de E-19 vio en rojo** el paso nuevo de `mirror-sync` que instalaba
`aiofiles`: ahora instala desde `requirements.txt`.

**Pisos (medidos dos veces, local 3.14):** tests-puros 979 → **1044 passed, 1 skipped** con `JAX_PLATFORM_REPO_ROOT`
inexistente (como el runner); `mirror-sync`: `check_mirror_sync.py` exit 0 (diez familias) y 15 + **2 passed** del
escritor real contra un worktree temporal de jax-platform `origin/master` `97e4b3e` (incluye #95);
`jacobs-gobernanza-db` **31** y `facet-health-io` **9** sin cambio contra `jax_memory_test`;
`policy/tests/test_no_fail_open_except.py` 21 passed.

- [x] **2026-09-17** Publicar la rama, PR, confirmar pisos en el runner, canario rojo por API y merge.
  `feat/kill-switch-real` es ancestro de `origin/master` (verificado con `git merge-base --is-ancestor`).
- [x] **2026-09-17** Deploy de jax con 0 pipelines en vuelo y `JAX_KILL_SWITCH_PATH` verificada en
  `/proc/2272008/environ` de `jax-las-manos` tras el reinicio. Evidencia completa en la VERDAD OPERACIONAL
  de arriba (incluye el journal sin errores y los 0 pipelines al momento del reinicio).
- [x] **2026-09-17 21:53–21:54 CST · PROBADO EN PRODUCCIÓN (Mr. Hyde · `fruiz-e5`).** El freno frena de verdad,
  incluido **el peor caso**: pipeline `b268d849-82a3-4d92-a25e-de812e8a28e6` creado por
  `POST /api/pipelines` con un paso en `jekyll`, confirmado en `running`; se activa el freno **con el pipeline
  en vuelo** y **3 segundos después** queda `aborted`, con el paso `failed` y motivo literal
  `killed_by_switch — /etc/jax/interruptor/PAUSE apareció durante la ejecución`.
  Puesta: `{"activo":true,"cambio":true,"heredada":false}` y `/health` de LAS MANOS con
  `kill_switch_active:true`. Quitada: `{"activo":false,"cambio":true,"heredada":false}`, el archivo
  `/etc/jax/interruptor/PAUSE` deja de existir, `/health` vuelve a `false` y `GET /api/state` de la Mesa
  responde 200. Ventana total ~3 minutos, coordinada por mensajes con `fruiz-cc` (medía conexiones) y
  `fruiz-47` (CI y merges), los dos avisados al abrir y al cerrar.
  **`jax_local` NO sirve para esta prueba:** su `facet.allowed_callers` es NULL y la regla es fail-closed, así
  que `jacobs` no puede llamarlo; se usa `jekyll` (deepseek), el más barato de los elegibles.
- [x] **2026-09-17 · CARGA MEDIDA, regla TIME-WAIT cumplida con margen.** `k6 loadtest/kill-switch.js`,
  50 VUs, con el freno PUESTO: **128.384 peticiones, 0 fallidas (0,00%), 4.279 req/s**, p95 global 27,57 ms
  — por escenario: freno 16,15 ms, admin 31,26 ms, activar 30,24 ms; los tres umbrales `p(95)<500` en verde y
  los 3 checks en verde. **TIME-WAIT 31 → 103**, contra la regla de < 10k. Línea base del día para referencia:
  6.976 → 8.173 con carga ajena → 3.437 en reposo.
- **HALLAZGO A FAVOR DEL CONTRATO (2026-09-17, bajo carga):** el escenario `activar_idempotente` pidió activar
  miles de veces con el freno ya puesto y `kill_switch_audit` quedó con **exactamente dos filas nuevas**
  (id 3 `activar`, id 4 `reanudar`, las dos `user_id=1`). Ni una fila de los activar redundantes, todos con
  `cambio:false`. La regla «activar no inventa un cambio con el freno ya puesto» aguanta concurrencia real,
  no sólo el test unitario.
- **REPL verificado tras el retiro de la voz:** `jax.core.main` importa sin `JAX_KOKORO_PYTHON` y expone
  `main()`. La puerta de arranque que lo mataba (`JAX_KOKORO_PYTHON no está seteada`) ya no existe.

**LECCIÓN (tercera aparición hoy) — el instrumento que se cuenta, se espera o se mata a sí mismo.**
Un patrón pasado a `pgrep -f` / `pkill -f` / `grep` sobre `ps` **matchea la propia línea de comando que lo
invoca**. Tres casos medidos el 2026-09-17, los tres dando una lectura falsa que parecía del sistema:
1. `pkill -f "authorize_facet_app:app"` mató el shell que lo ejecutaba → **salida 144** sin que fallara nada.
   Es la causa de un «código 144 sin anotar» que otra sesión estaba por registrar como hallazgo.
2. Una espera de «máquina libre» que grepeaba `ps -eo args` por `pytest|k6` contaba **las esperas de las otras
   sesiones** (sus `until` mencionan esos comandos): reportaba 7 procesos de carga con 0 corriendo. Corregida
   con `pgrep -x k6` y `pgrep -f 'python.*-m pytest'`; la comprobación decisiva del cierre de una app es el
   **puerto** (`ss -ltnp`), no el patrón.
3. Un contador de conexiones `ss … | grep -c ':3308'` contaba puertos efímeros del cliente 33080-33089 como
   conexiones a MariaDB, inventando 13 donde había 10 (otra sesión; cerrado con atribución en proceso).
**El repo ya tenía la lección escrita** en `scripts/check_ollama_num_parallel.py:127,150` («pgrep -f se ve a sí
mismo si el patrón aparece en la propia línea de comando», observado en su día). Que haya vuelto tres veces el
mismo día dice que la lección estaba en un comentario y no en una herramienta compartida: **la próxima vez que
haga falta contar procesos, se usa un helper con la exclusión adentro, no un `pgrep -f` a mano.**

## Cerrado en código, merge y despliegue pendientes — Ejecutor SP1 plan 1: C1 prohibiciones y C2 respaldo (2026-09-17)

Detalle y pruebas de «visto fallar» en CONTEXT.md §9 (2026-09-17 ~04:40). Ramas locales con commits, SIN PUBLICAR
(un subagente no puede escribir en el remoto): jax-platform `feat/ejecutor-tablas-c1-c2`
(`/home/fruiz/worktrees/jax-platform-sp1-c1`), jax `feat/ejecutor-c1-c2` (`/home/fruiz/worktrees/jax-sp1-c1`).

- [ ] **2026-09-18** Publicar la rama y abrir el PR de jax-platform; mergear ANTES que el de jax (`jacobs-gobernanza-db` clona su master).
- [ ] **2026-09-18** Publicar la rama y abrir el PR de jax; confirmar en el log del runner `979 passed, 1 skipped`
  (tests-puros) y `31 passed` (jacobs-gobernanza-db); si difiere, manda el runner.
- [ ] **2026-09-18** Canario rojo por API (plan, Task 8 Step 4): en `politica.py`, `return Decision(False, PROHIBIDO, …)`
  → `return Decision(True, PERMITIDO, None, ())`; `tests-puros` = `failure` sobre ese sha, revert = `success`.
- [ ] **2026-09-18** Despliegue (plan, Task 9 Step 2): dump de `axioma_config` restaurado en tabla temporal antes del
  restart de jax-platform; `JAX_EJECUTOR_*` e `JAX_EJECUTOR_INVENTARIO` en `/etc/jax/.env` con sudoedit (dos lectores
  idénticos); `exportar` → `reglas=14 hosts=4`; `instalar_contratos.sh`; `probar_c1.py` → `c1_vivo=true`, y repetir
  sobre la instalación real los tres «verlo fallar» (Steps 5–6).

## CERRADO, DESPLEGADO Y CON CARGA MEDIDA — frente F: contrato de sub-pipelines, I-2 y pool de conexiones de Jacobs (2026-09-17)

**VERDAD OPERACIONAL 2026-09-17 ~04:30 CST (Mr. Hyde, verificado en el worktree).** Rama jax `feat/contrato-subpipelines` rebasada sobre `origin/master` `de6964e` (frente E #177, frentes A #174 y C #175 ya adentro): 19 commits, sin publicar en GitHub, SIN mergear, SIN desplegar. jax-platform no se toca (master `c414eba`). Plan `docs/superpowers/plans/2026-09-16-frente-f-contrato-subpipelines.md`; ledger `jax-frente-f/.superpowers/sdd/2026-09-16-frente-f-contrato-subpipelines/progress.md` (+ `pr-body.md`, `enmienda.md`, `i2-report.md`, `pool-report.md`, `pool-carga.md`, `pool-ronda2-carga.md`).

**El hallazgo (verificado contra el código en `984ce46`):** el candado contra sub-pipelines de Ada era nominal. `jacobs/policy.py` aceptaba cualquier string no vacío como `subpipeline_token` (sin validez, dueño ni vigencia); `routes.py` nunca le pasaba `subpipeline_depth` a `validate_create`, así que el límite de profundidad era código muerto; `MAX_SUBPIPELINE_DEPTH = 1` era un literal. Cualquier cliente creaba un hijo de Ada con un token inventado.

**Contrato de sub-pipelines (qué quedó):**
- **Token emitido solo por el servidor** (`subpipelines.emitir_token_subpipeline`, sin ruta HTTP): 32 bytes, en la tabla `jacobs_subpipeline_tokens` se guarda SOLO el sha256 (`token_hash` PK, índice por `parent_pipeline_id`), atado a `parent_pipeline_id` + `parent_step` (paso de faceta `ada`) + `depth_hijo` + `vence_at`.
- **Consumo atómico en un solo `UPDATE`** dentro del `_pipeline_create_lock` de `create_pipeline`: token no usado, no vencido, del padre declarado, padre `running`, profundidad ≤ máximo. Carrera de 20 consumos → un ganador; carrera forzada con dos sesiones → la segunda pierde; `EXPLAIN` solo `PRIMARY`. Un rechazo por padre equivocado NO quema el token.
- **`jacobs_pipelines` gana `parent_pipeline_id` y `depth`** (`init_tables()`); el hijo hereda la identidad (`user_id`/`tenant_id`) del padre, y un cuerpo con otra identidad es 403 sin quemar el token (I-3). El cuerpo no fija la profundidad (`depth` en el JSON se ignora).
- **Forma por rol:** `invoked_by="ada"` exige token + padre; cualquier otro rol que los traiga → 422 visible. `POST /jacobs/plan` con `ada` → 403. Los validadores de `PipelineCreateRequest` son por campo: el 422 de forma ya no devuelve el token en `input` (I-1).
- **Todo rechazo deja `SUBPIPELINE_RECHAZADO`** con motivo, sin el token en claro (solo `token_ref`, 12 chars del sha256), visible desde el padre.
- **Enmienda de Fernando (GO 2026-09-16, prevalece sobre el plan):** `JAX_MAX_SUBPIPELINE_DEPTH` por defecto **3**, rango [1, 5]; **"padre activo" = `jacobs_pipelines.status='running'` del padre, solamente** — el paso de Ada que delega puede estar en cualquier estado (en "plan de delegación" Jacobs emite después de que `delegate` terminó); se conserva que `parent_step` exista, sea del padre y sea de `ada`.
- **Config validada al arrancar** (`config_subpipelines()`): `JAX_SUBPIPELINE_TOKEN_TTL_SECONDS` default 300, rango [10, 3600]; ausente → default, presente e inválida → error (fail-closed).

**I-2 (residual R13):** el consumo corría fuera del `try` que audita: si la relectura tras el `UPDATE` confirmado fallaba (error de base o invariante) o llegaba un `CancelledError`, el token quedaba quemado sin evento. Arreglo (`535cf0b`): consumo en `try/except BaseException` que audita best-effort (`fase="consumo"`, clase de la excepción, `token_ref`) y relanza SIEMPRE el original; el `except Exception` de la creación pasa a `BaseException`. Dos tests con error REAL de la base y con cancelación, vistos en rojo; mutación de cada `except` a `Exception` vuelve a romper el de la cancelación.

**Pool de conexiones de Jacobs:** `get_conn()` (una conexión `aiomysql` nueva por consulta) eliminado; `store.conexion()` sobre un pool por event loop con semáforo propio (sin `pool._wakeup()` privado), creación con candado y doble chequeo, descarte ante excepción/cancelación/transacción abierta/autocommit apagado, espera y cierre acotados por `JAX_DB_CONNECT_TIMEOUT_SECONDS`, cierre en el shutdown de LAS MANOS y red al terminar el loop. `JAX_JACOBS_DB_POOL_SIZE` default 10, rango [1, 50] (`max_connections`=151, `Max_used_connections`=95). Migrados además `motor_registry/facet_policy.py` (`POST /motor/authorize-facet`) y los dos escritores de `axioma_usage` (la falla del pool toma el mismo camino a la cola durable). `aiomysql==0.3.2` vigilado por `store.AIOMYSQL_REVISADO`. Detalle y mutaciones en `pool-report.md`.

**Carga (app aislada, `jax_memory_test`, k6 5s+20s+5s, "fallas" = `http_req_failed`; 200/403/422 esperados, el 422 de límite paralelo se cuenta aparte):**

| endpoint / escenario | VUs | antes (sin pool) fallas · p95 | después fallas · p95 | tras el rebase sobre `de6964e` fallas · p95 · rps |
|---|---|---|---|---|
| `/jacobs/pipeline` base | 25 | 40,28 % · 147,49 ms | 0,00 % · 76,82 ms | **0,00 %** · 119,87 ms · 285 |
| `/jacobs/pipeline` legítimo | 25 | 22,85 % · 178,77 ms | 0,00 % · 96,39 ms | **0,00 %** · 148,85 ms · 251 |
| `/jacobs/pipeline` inventado | 25 | 25,38 % · 121,80 ms | 0,00 % · 31,90 ms | **0,00 %** · 35,74 ms · 780 |
| `/motor/authorize-facet` | 25 | 50,07 % · 35,07 ms | 0,00 % · 4,47 ms | **0,00 %** · 4,88 ms · 5.846 |

Antes: 22-50 % de fallas en todas las celdas, TIME_WAIT 33.000-42.000 (el rango efímero entero), 14.762 / 79.886 tracebacks. Después: 0 %, TIME_WAIT 41-229, 0 tracebacks, 0 `TimeoutError` de pool. Tras el rebase: checks 100 %, 0 tracebacks; el p95 de base/legítimo subió (host con load average 8,8 por otros frentes, y la fila `test`/`Fernando` ajena activa dio 2.120 422 de límite en base): se lee como ruido del host compartido, no como regresión — el umbral (<1 % a 25 VUs) se cumple.

**Defecto del propio frente — `Row size too large` en `jacobs_pipelines` (HISTORIA, con lección):** `test_jacobs_pipelines_gana_parent_y_depth...` hacía `DROP COLUMN depth` y `init_tables()` lo volvía a agregar con `ALGORITHM=INSTANT`. Cada DROP+ADD instantáneo deja rastro en el formato de fila: tras muchas corridas `jax_memory_test` quedó sin `depth` y con `1118 Row size too large` en todo ADD (57 de 71 tests en rojo). Medido en bucle (MariaDB 12.3.3): DROP por defecto rompe en la corrida 27, `INPLACE` también en la 27, `ALGORITHM=COPY` llega a 60 sin error. Arreglo (`8b1c60e`): el DROP del test lleva `ALGORITHM=COPY`; la base de prueba se reparó con `ALTER TABLE ... FORCE, ALGORITHM=COPY`; 35 corridas seguidas "71 passed". En producción el ADD corre una sola vez. **Lección: un test que muta el esquema de una base persistente acumula; se mide en bucle, no en una corrida.**

**Regla de carga nacida hoy (DECISIÓN operativa del frente F, 2026-09-17):** la MariaDB de pruebas (`jax_memory_test`) vive en el MISMO servidor que producción (127.0.0.1:3308). En la Task 8, sin pool, la carga llevó el host a 25.969 sockets TIME_WAIT de 28.232 puertos efímeros (92 %) y el sembrado de tokens falló por agotamiento de puertos: eso le quita puertos a producción. **Toda carga contra esa MariaDB: pool acotado (nunca conexión por pedido), esperar TIME_WAIT < 5.000 antes de cada corrida y abortar si pasa de 10.000, app aislada en puerto propio levantada y bajada por el arnés, y limpiar las filas propias al terminar.**

**Rebase del 2026-09-17 sobre `de6964e` (Mr. Hyde) — conflictos:**
1. `jacobs/models.py`, `policy.py`, `routes.py`: E-13 (`MAX_STEPS_PER_PIPELINE` en `models.py`) + los campos, validadores por campo e `INVOKER_ADA` de F; el validador de `max_steps` usa la constante.
2. `jacobs/store.py`: el 4º SELECT de E-17 (facetas activas) dentro de `async with conexion()`.
3. `las_manos/server.py`: conviven los dos shutdown — cierre del cliente HTTP compartido (E-24) y del pool de Jacobs.
4. `.github/workflows/policy.yml`: listas de tests-puros unidas; el `pip install` del job `subpipeline-contrato-db` pasa a `-r requirements.txt` (el de F instalaba `aiofiles`, que `test_requirements_completos.py` prohíbe desde E-18) y los de los demás jobs quedan como los dejó E-19.
5. **Sin conflicto textual:** `tests/test_facetas_de_gobernanza_db.py` (E) usaba `store.get_conn()`, que F borró — migrado a `store.conexion()`; el escaneo AST de `_store_pool_test.py` lo habría marcado.

**Pisos (medidos dos veces tras el rebase, local 3.14):** tests-puros 853 → **872 passed, 1 skipped** (entorno limpio, sin `JAX_DB_HOST` ni checkout de jax-platform); `subpipeline-contrato-db` **71**; `jacobs-gobernanza-db` **29** (migraciones de jax-platform `c414eba`, sello en un temporal); `facet-health-io` 9; `jacobs/_store_pool_test.py` 35; `tests/test_cola_uso_escritores.py` 15 (con `PYTHONPATH=.:las_manos`, como el job); `policy/tests/test_no_fail_open_except.py` 21; `check_mirror_sync.py` contra un worktree temporal de jax-platform `origin/master` `c414eba`: exit 0, familias sincronizadas, y sus tests 14. **Ningún runner los confirmó todavía: si el runner da otro número, manda el runner.**

**Segundo rebase, 2026-09-17, sobre `origin/master` `f9c7073` (frente B #183 kill switch real + Ejecutor C3 #181) (Mr. Hyde) — REEMPLAZA la base y los pisos de arriba.** 24 commits (23 rebasados + 1 de ajuste), sin publicar. Conflictos:
1. `jacobs/policy.py`: import — conviven `from interruptor import interruptor_activo` (B, sin `KILL_SWITCH_PATH`) e `INVOKER_ADA` (F). El freno de los sub-pipelines sigue pasando por `policy.check_kill_switch()`, que ahora lee el interruptor.
2. `.github/workflows/policy.yml` (tres commits de pisos de F): listas de tests-puros unidas (archivos del interruptor + `test_subpipeline_contrato_puro.py`); comentarios de ambos conservados; piso final re-medido.
3. `CONTEXT.md` / `DEUDA.md`: secciones de C3 y de F, las dos.
4. **Sin conflicto textual pero roto:** `tests/test_ejecutor_politica_db.py` (Ejecutor C1/C2) importaba `jacobs.store.get_conn`, que F borró — migrado a `store.conexion()` (la limpieza en su propia conexión). No quedó ningún otro `get_conn` en el árbol.

**Pisos tras el segundo rebase (dos corridas cada uno, local 3.14):** tests-puros **1127 passed, 1 skipped** (1108 de master + 19 de F; env limpio, sin `JAX_DB_HOST`, `JAX_PLATFORM_REPO_ROOT=/tmp/no-existe`); `subpipeline-contrato-db` **101**; `jacobs-gobernanza-db` **31** (migraciones de jax-platform `0c6eb67` sobre `jax_memory_test`, sello en un temporal); `facet-health-io` 9; `policy/tests` 35 (`test_no_fail_open_except.py` 21); `check_mirror_sync.py` contra un worktree temporal de jax-platform `origin/master` `0c6eb67`: familias sincronizadas, sus tests 14, `test_cola_uso_escritores.py` 15, `test_interruptor_escritor_de_la_plataforma.py` 2. **Si el runner da otro número, manda el runner.**

**Pendientes (fechas propuestas por Hyde; Fernando las confirma o cambia):**
- **Publicar la rama, CI, canario y merge — control 2026-09-18.**
- ~~**Deploy de jax F — control 2026-09-18, con 0 pipelines en vuelo.** `/etc/jax/.env` (hoy ninguna de las tres está)~~ — **CORREGIDO Y DESPLEGADO 2026-09-17 ~15:40 (Mr. Hyde · `fruiz-e5`).** La frase «hoy ninguna de las tres está» era una VERDAD OPERACIONAL caducada. **Medido en el proceso vivo** (`sudo cat /proc/2272008/environ` de `jax-las-manos`, reiniciado 15:23:02 con 0 pipelines en vuelo): `JAX_SUBPIPELINE_TOKEN_TTL_SECONDS=300`, `JAX_MAX_SUBPIPELINE_DEPTH=3`, `JAX_JACOBS_DB_POOL_SIZE=10`, las tres presentes; las tres también escritas en `/etc/jax/.env`. **CARGA MEDIDA 2026-09-17 21:57 CST — HAY NÚMERO, HAY GO.** `k6 loadtest/authorize_facet.js`, 25 VUs, contra la app aislada `authorize_facet_app.py` en :7798 sobre `jax_memory_test` (nunca contra LAS MANOS de producción; la app se niega a arrancar con otra base): **174.576 peticiones, 0 fallidas (0,00%), 0 checks fallidos de 174.576, 5.819 req/s**, p95 6,88 ms (avg 3,56 ms, max 13,06 ms), umbral `p(95)<500` en verde. Los tres caminos rotados quedan cubiertos (facet permitido, facet sin `allowed_callers` fail-closed, y el caller que el middleware corta con 403). App cerrada al terminar: el puerto 7798 no escucha. Corrida en ventana coordinada con `fruiz-cc`, que mide contra esa MISMA base.
- **Ningún camino vivo emite tokens todavía:** `emitir_token_subpipeline` solo lo llaman el arnés y los tests, y ningún código de jax ni de jax-platform manda `invoked_by="ada"` a `/jacobs/pipeline` (grep 2026-09-17, Hyde). El emisor de Ada (modo "plan de delegación") es otro trabajo; hasta que exista, `ada` sin token es 422 visible.

## Cerrado en código y DESPLEGADO (gate cumplido), faltan E-24 y los controles fechados — frente E de la auditoría: limpieza, defectos y reglas en jax (2026-09-17)

**VERDAD OPERACIONAL 2026-09-17 ~02:20 CST (Mr. Hyde, verificado en el worktree).** Rama jax `fix/hallazgos-frente-e` rebasada sobre `origin/master` `0da32af` (frentes A #174, C #175, Ejecutor #172/#173/#176 ya adentro): 17 commits, sin publicar en GitHub, SIN mergear, SIN desplegar. Lado plataforma MERGEADO: jax-platform#92 → `c53ef30` (2026-09-17: `config_entorno` única que absorbe `config_de_entorno` de A, `JAX_OLLAMA_URL` obligatoria al arrancar, docstring de `owner_cleanup`), canario rojo `d234215` (`backend-tests` con y sin DB) leído por API. Spec `docs/superpowers/specs/2026-09-16-hallazgos-auditoria-jax-design.md` §E, plan `docs/superpowers/plans/2026-09-16-frente-e-jax-limpieza-defectos-reglas.md` (worktree `jax-hallazgos-docs`); ledger `jax-frente-e/.superpowers/sdd/2026-09-16-frente-e-jax-limpieza-defectos-reglas/progress.md` (+ `rebase-e-platform.md`, `unificar-config-entorno-report.md`).

**Qué se retiró y qué se arregló (E-01..E-24):**
- **Código muerto (E-01/02/04/05/06/07/08/09/20):** `_director_patch/*.py` (el control `_NO_PARSEA` se ejercita ahora con un `.py` roto de mentira), `StepResult`, `TRAFFIC_CLASSES`, `enabled_motors`, voces Piper (dos `.onnx` de ~63 MB), imports sin uso, `hyde_requires_human_gate`, `scripts/cleanup.sh` (nunca tuvo scheduler), `[motors.*]` de `config.toml` (la verdad es la tabla `motor`).
- **Una sola constante (E-13):** `MAX_STEPS_PER_PIPELINE` vive en `jacobs/models.py`; `policy.py`, `routes.py`, `plan.py` y el validador la importan (antes, literales `20` repartidos).
- **Facetas del planner desde la tabla `facet` (E-03/17/23):** una faceta desconocida o inactiva rechaza el plan (422 + `PLAN_REJECTED`) en vez de caer a `jax_local`; el menú de facetas de los prompts de Ada y qwen sale de las activas.
- **Un archivo real por módulo (E-10/11):** `crypto_secrets`, `credential_resolver`, `model_catalog` y `cliente_http_compartido` de `las_manos/` son symlinks a `jax/core`.
- **URLs de servicio desde el entorno, fail-closed (E-21):** `LAS_MANOS_URL`, `JAX_OLLAMA_URL`, `JAX_KOKORO_PYTHON` (esta última **retirada el 2026-09-17** con la voz) sin default; `config_entorno.url_requerida` exige URL base (sin path, query ni fragmento). Los workers de memoria leen la URL de DeepSeek de `provider.base_url`; un proveedor `deprecated` no da URL. Familia de espejos `config_entorno` en `scripts/check_mirror_sync.py` (copia verbatim en jax-platform).
- **Documentos en `JAX_REPO_BASE`, escritura en `asyncio.to_thread`, sin `aiofiles` (E-12/18/22); `requirements.txt` fuente única del CI con `cryptography`/`pyyaml` fijados (E-19).**
- **`_sin_autoetiqueta` con la cabecera de `authority_origin` y recorte simple del embedding (E-14/15).**
- **Errores de proveedor redactados con la credencial conocida ANTES de recortar (E-16)** en executor, Ada y REPL; el error de tarea se redacta antes de ir a disco.
- **Cliente HTTP compartido por proceso (E-24):** `jax/core/cliente_http_compartido.py`, cierre al apagar, timeout por llamada, tripwire que prohíbe construir `AsyncClient` fuera de ese archivo. Medido en modo aislado (0 errores): c=1 p95 3,55 → 0,47 ms (rps 291 → 2.562); c=10 p95 43,22 → 10,68 ms (rps 316 → 1.536).

**Rebase del 2026-09-17 (Mr. Hyde) — conflictos y lo que destapó:**
1. `jacobs/policy.py`: C había puesto sobre `MAX_STEPS_PER_PIPELINE` el comentario de la familia `tope_pipelines`, que habla de `MAX_PARALLEL_PIPELINES`. Se conservó E-13 (la constante vive en `models.py`) y el comentario quedó sobre la constante que de verdad se espeja.
2. `scripts/check_mirror_sync.py`: se conservan las NUEVE familias (`facet_resolver`, `crypto_secrets`, `credential_resolver`, `db_connect_config`, `contrato_dispatch`, `cola_uso`, `router_keywords`, `tope_pipelines`, `config_entorno`). `policy.yml` se mezcló solo; las dos listas de tests-puros verificadas idénticas entre sí (64 archivos, sin duplicados).
3. **Dos tripwires de E vieron en rojo código que entró por #173:** `jax/ejecutor/proxy_carril.py` construía su propio `httpx.AsyncClient` y `h11` no estaba en el mapa de `test_requirements_completos.py` (sí en `requirements.txt`). Arreglado en `4bb8053`: el proxy usa `crear_cliente_http()` y el timeout (connect 10 s, sin tope de lectura) viaja en cada petición — test nuevo visto en rojo con el cliente sin timeout por petición (read/pool 5,0).
4. **`facet-health-io` no habría colectado en CI:** desde E-24 `jacobs/` importa `cliente_http_compartido` por nombre corto (como corre dentro de LAS MANOS) y ese job no tenía `las_manos` en el path. Visto en rojo (`ModuleNotFoundError`) y en verde (9) con `PYTHONPATH: las_manos` (`20a7fb9`).

**Pisos (medidos dos veces, local 3.14):** tests-puros 742 → **850 passed, 1 skipped** simulando el runner sin checkout de jax-platform (master en el mismo entorno: 742/1, igual que su runner); +108 contados por archivo con `--collect-only`. `jacobs-gobernanza-db` 27 → **29** (master: 27) contra `jax_memory_test` con las migraciones de jax-platform `c53ef30`. Sin cambio y verdes: facet-health-io 9, facet-resolver-seal 13, plan-timeout-ceiling 16, mirror-sync 14 + cola_uso 15, governance 90, ollama-num-parallel 33, hyde-containment 14; `policy/tests/test_no_fail_open_except.py` 21. `check_mirror_sync.py` contra jax-platform `c53ef30`: exit 0, las nueve familias sincronizadas. **Los pisos no los confirmó todavía ningún runner: si el runner da otro número, manda el runner.**

**E-25 (B1.4, retiro del fallback a `.env` de credenciales) — ✅ CERRADO EN `jax` EL 2026-09-17.** La medición del 2026-09-17, sobre **30 días** de journal (no 7): **2.760 líneas `source=db`, 0 líneas `source=env_fallback`**, con **una rotación real** dentro de la ventana — la llave de Gemini, rotada el 2026-09-15. El criterio de salida escrito en B1.4 (7 días consecutivos sin `env_fallback`, incluyendo al menos una rotación real) **se cumple**. **Decisión de Fernando: se retira el fallback en los dos repos.** Hecho en `jax`: `jax/core/credential_resolver.py` pierde el mapa proveedor→variable de entorno y la función instrumentada de doble lectura; `jax/core/facet_resolver.py`, `jax/muscles/base.py` y `las_manos/motor_registry/worker.py` llaman directo a `resolve_credential()`. **La DB es la única fuente y sin credencial activa se falla cerrado** (`CredentialUnavailableError`) — nunca a una variable de entorno, que era un fail-open de la rotación: una llave revocada en la DB seguía viva mientras el `.env` la tuviera. Vigilado por `tests/test_credencial_sin_fallback_env.py` (3 tests, los 3 vistos en rojo contra `351ec95`; el primero con la línea `source=env_fallback` en el log capturado). `scripts/check_mirror_sync.py`: la familia `credential_resolver` pasa de 10 a 8 símbolos compartidos — declarar los dos retirados dejaría el checker en rojo permanente por símbolos que no existen en ninguna copia, y un checker siempre rojo se ignora. El gemelo de `jax-platform` va en su propio PR (otra sesión, en paralelo). **LO QUE SIGUE ABIERTO, fuera del resolver:** `/api/admin/keys` (jax-platform) escribe llaves a `/etc/jax/.env` y a `os.environ`; `scripts/manual_motor_v02_integration.py` (script manual de diagnóstico, no un servicio) lee `KIMI_API_KEY` del archivo; y `crypto_secrets.PROVIDER_ENV_KEYS` / `decrypt_provider_keys_in_env()` siguen existiendo — son familia espejada compartida y su retiro es una decisión aparte, no parte de B1.4.

**E-26:** 25 backups sin trackear en `/home/fruiz/jax`, los 25 idénticos byte a byte a blobs de git (`git hash-object`). Borrado pendiente del controlador principal, con re-verificación antes de borrar.

**Incidentes del 2026-09-17 (HISTORIA, con lección):**
- **jax#175 se mergeó con `jacobs-gobernanza-db` en rojo.** El gate usaba `set -euo pipefail` dentro del Bash tool en segundo plano y NO cortó. Causa del rojo: acoplamiento de import del frente C en jax-platform (`db/migrations` → `ajustes` → `auth.jwt`, que exigía `JAX_JWT_SECRET` al importarse) — arreglado de raíz en jax-platform#93 → `2bb90b2` (`auth/constantes.py`); rerun del job verde (intento 2, 07:56:28Z). **Lección: `set -e` no es un gate. Desde entonces todo merge pasa por un gate con `exit` explícito en cada paso, probado contra un canario rojo antes de confiar en su verde.**
- **Un agente que reprodujo ese job en hall9000 sin `JAX_FACET_SEAL_PATH` tocó el mtime de `/srv/jax-data/facet-cache-seal`** (01:51:21, sin cambio de datos): un toque del sello invalida la caché de facetas de todos los procesos. **Lección: una barrera que vive solo en la cabeza del que corre el job no viaja cuando otro copia los pasos.** Arreglo en esta rama (`20a7fb9`): el job exporta `JAX_FACET_SEAL_PATH=$RUNNER_TEMP/facet-cache-seal` por `GITHUB_ENV` antes de migraciones y tests. Verificado en local: con la variable, el sello se escribe en el temporal y el de `/srv` no cambia; la lista de tests-puros NO escribe el sello (corrida con la variable a un directorio vacío: sigue vacío). El mtime actual del sello real (02:07:11) coincide con el reinicio de `jax-platform.service` de las 02:07:10 del deploy de E-platform (journal): escritura del arranque, no de un test.

**Pendientes (fechas propuestas por Hyde; Fernando las confirma o cambia):**
- **Publicar la rama jax, CI, canario y merge — control 2026-09-18.** jax-platform ya tiene `c53ef30`, así que `mirror-sync` puede correr en verde.
- **Deploy de jax E — control 2026-09-18, con 0 pipelines en vuelo** (al apagar LAS MANOS se cierra el cliente compartido): `/etc/jax/.env` necesita `JAX_OLLAMA_URL` (el journal de sudo registra su escritura a las 02:06:58 para el deploy de jax-platform; verificar en `/proc/<pid>/environ` de `jax-las-manos` tras el reinicio) ~~y `JAX_KOKORO_PYTHON` (sin ella el REPL no arranca)~~ — **CORREGIDO 2026-09-17: `JAX_KOKORO_PYTHON` ya NO se necesita, la voz se retiró** (ver § "Retiro de la voz"); `JAX_REPO_BASE` existe desde el frente A. Gate previo: facetas `hipatia, jekyll, thot, ada, kimi, hyde, jax_local` en `active` y `provider.base_url` de deepseek presente (E-17 y los workers de memoria dependen de eso).
  - **DESPLEGADO Y GATE CUMPLIDO 2026-09-17 ~15:40 (Mr. Hyde · `fruiz-e5`), verificado en vivo:** en
    `/proc/2272008/environ` del `jax-las-manos` vivo están `JAX_OLLAMA_URL=http://localhost:11434`,
    `LAS_MANOS_URL=http://127.0.0.1:7777` y `JAX_REPO_BASE=/home/fruiz/jax/repo`. Gate previo medido con SQL:
    las 7 facetas en `active` (`ada, hipatia, hyde, jax_local, jekyll, kimi, thot`) y
    `provider.deepseek.base_url=https://api.deepseek.com/v1`, `status=active`. `JAX_KOKORO_PYTHON` confirmado
    como NO necesario: no está en el entorno del proceso y el servicio arrancó sin errores.
- **Medición de E-24 contra el Ollama de producción (modo `embedding`, antes/después) — en el deploy.** Sin ese número no hay GO del deploy de E-24.
- **E-25: decisión de Fernando sobre el retiro del fallback — control 2026-09-24.**
- **E-26: borrado de los 25 backups — control 2026-09-18.**

**Deuda nueva que este cierre destapa:** ver "Anotado — deuda residual del frente E" en `## Anotado, no bloquea`.

## Cerrado — frente C: los ajustes de Admin mandan de verdad (2026-09-17)

**VERDAD OPERACIONAL 2026-09-17 01:32 CST** (desplegado y verificado por el controlador principal). PR jax-platform#91 → `accc641` (mergeado 2026-09-17); familia de espejos `tope_pipelines` + guiones k6 en jax, commit `807cf06` (jax `984ce46..807cf06`, rama `feat/ajustes-que-mandan`). Plan `docs/superpowers/plans/2026-09-16-frente-c-ajustes.md` (worktree `jax-platform-hallazgos-docs`); ledger `jax-platform-frente-c/.superpowers/sdd/2026-09-16-frente-c-ajustes/progress.md` + `carga.md` (18 tasks, subagent-driven-development).

**Qué manda ahora de verdad:** los cinco ajustes de Admin → Configuración (`session_timeout_min`, `max_pipelines`, `web_task_retention_days`, `lang_default`, `system_name`) dejan de ser cosméticos. `backend/ajustes.py` es la única lectura tipada de esas cinco claves de `axioma_config` (consulta por PRIMARY, caché con TTL invalidado en el PUT); un valor ausente o inválido lanza `AjusteIlegible` → 503 `ajuste_ilegible`, nunca un default silencioso. Consumidores: emisión/vida del refresh (`api/auth.py`+`auth/jwt.py`), cupo de pipelines (`api/pipelines.py`+`jax_engine/resource_manager.py`), el reaper de owner files (`jax_engine/owner_cleanup.py`) y `GET /api/apariencia` (idioma inicial y nombre del sistema en el frontend).

**Discrepancias con el spec, resueltas con evidencia (no cambian la intención del ítem):**
1. **El refresh no rota — la sesión es de vida ABSOLUTA, no deslizante** (`api/auth.py` no rota la cookie; documentado como decisión previa). `session_timeout_min` fija el `exp` del JWT, el `max_age` de la cookie y, además, `/refresh` valida contra un `iat` propio — sin esto último, acortar el ajuste no alcanzaría a las sesiones ya abiertas hasta 7 días después. **Consecuencia (GATE, decidida por Fernando): cada sesión abierta antes del deploy vuelve a iniciar sesión una vez** (un refresh viejo no trae `iat` y da 401 `sesion_expirada`).
2. **`max_pipelines` no puede superar el candado global de Jacobs** (`jax/jacobs/policy.py`: `MAX_PARALLEL_PIPELINES = 3`, cuenta todos los pending/running de todos los tenants). El valor NO viaja a jax: es una cuota por tenant en la plataforma, acotada por arriba por ese candado. `backend/ajustes.py` lleva una copia textual de `MAX_PARALLEL_PIPELINES` y la familia `tope_pipelines` en `scripts/check_mirror_sync.py` (jax, commit `807cf06`) la vigila — verificado sincronizado contra `jax-platform` master (`accc641`) con `scripts/check_mirror_sync.py` y sus 14 tests, los dos en verde.
3. **`ws_notifications` sale de `DEFAULT_CONFIG`** (decisión de Fernando, coordinada con la Discrepancia 10 del frente A) y la migración la borra de la fila real una sola vez, con test.
4. **Retención de tareas web:** al vencer `web_task_retention_days`, `owner_cleanup.py` borra misión, resultado y dueño JUNTOS (misión+resultado primero, dueño al final, para que una caída a mitad deje un dueño huérfano que el ciclo siguiente termina de limpiar) — decisión de Fernando, reemplaza el GATE original del plan.

**Migración `ajustes_que_mandan_v1` (aplicada una sola vez, 01:32:52 CST) — antes / después, medido:**

| Ajuste | Antes | Después |
|---|---|---|
| `session_timeout_min` | 60 | 10080 |
| `max_pipelines` | 1 | 3 |
| `web_task_retention_days` | 7 | 30 |
| `lang_default` | es | es |
| `system_name` | Axioma | Axioma |
| `ws_notifications` | true | (fila borrada) |

EXPLAIN de la consulta de `ajustes.py`: `key=PRIMARY`. `GET /api/apariencia` en vivo → `theme_default: "dark"` (el plan esperaba `light`: dato anotado, no tocado — no es parte del alcance de este frente).

**Backup y prueba de restauración (Principio VI — un backup sin restauración probada no es backup):** dump `/home/fruiz/backups/axioma_config-pre-frente-c-20260917-013226.sql` (15 filas). La restauración prevista por el plan (`CREATE DATABASE` + reload) **falló**: `jax_user` no tiene permiso para crear bases. Probado en su lugar cargando el dump con la tabla renombrada `axioma_config_restore_check` dentro de `jax_memory_test` → **idéntica fila por fila** (MD5 de valor + `updated_at` comparado por fila), tabla de prueba borrada después. **Lección para la próxima vez: el usuario de servicio no puede crear bases nuevas — la restauración de un dump de una tabla existente se prueba con un nombre de tabla temporal en la base de test, nunca asumiendo `CREATE DATABASE` disponible.**

**Deploy:** `jax-platform.service` reiniciado, `NRestarts=0`, journal limpio. Frontend `index-DO0TEv_Q.js` (antes `index-ChCOZGPN.js`), backup `/www/wwwroot/axioma-ia.io.backup-pre-frente-c-20260917-013311`. Canario por API sobre el sha real: rojo en `d212f4b` (`backend-tests-con-db` y `backend-tests-no-db`), revert verde en `ef25862`. k6 `health.js`, 10 VUs, post-deploy: p95 0,52 ms, 0 fallas de 681.254 peticiones.

**Carga previa al deploy (instancia aislada, puertos propios — Ruling R6 del ledger; usuarios de prueba borrados al cerrar):** login p95 6,81→4,45 ms; refresh p95 5,02→5,63 ms (sin caché, TTL forzado a 0,001 s: 6,01 ms — sobrecosto del caché frío 0,38 ms, no es gate); crear pipeline p95 5,39→5,40 ms. GO en los tres criterios (±10%/+5ms de la base). **Advertencia sobre la medida de login (R17 del ledger): los usuarios de carga usan `bcrypt rounds=4` (fixture de test), no los 12 de producción — el número mide la sobrecarga de `ajustes.py`, no el costo real de bcrypt; no se repitió con rounds=12.**

**Efecto de usuario post-deploy (decisión de Fernando):** re-login único de todas las sesiones activas (consecuencia de la Discrepancia 1, `iat` nuevo en el refresh).

**Revisión final — encontró y arregló ANTES del push (regla "sin hallazgos diferidos"):** `Login.jsx` mostraba "usuario o contraseña incorrectos" ante un 503 `ajuste_ilegible` (en vez de un error genérico de servicio) y `client.js` convertía cualquier 5xx de `/refresh` en `sesion_expirada`; tests de 503 agregados en `/refresh` y `/me/password` (antes solo en login); `?.clave` defensivo en `AdminSettings.jsx` (`errorDeGuardado`); `policy.yml:562` corregido (el mensaje del piso decía 504, el piso real ya era 528).

**Pisos de CI (medidos dos veces cada uno):** con DB 1334→1386, sin DB 764→787, vitest 504→535 (base 6e50959, ya con el frente A mergeado). `scripts/_check_mirror_sync_test.py`: 14/14 verde contra jax-platform `accc641` (ambos frentes A y C mergeados).

**Pendiente sin fecha fija:** verificación en vivo con Fernando (Task 17 del plan) — cambiar cada uno de los cinco ajustes desde Admin, ver el efecto real, restaurar el valor.

**Deuda nueva que este cierre destapa:** ver "Anotado — deuda residual del frente C" en `## Anotado, no bloquea`, más abajo.

## Cerrado — hallazgos de la auditoría de sobre-ingeniería, frente A (2026-09-17)

**VERDAD OPERACIONAL 2026-09-17 00:28 CST** (desplegado y verificado por el controlador principal). PR jax-platform#90 → `6e50959` (mergeado 2026-09-17); PR de jax (familia de espejos `router_keywords`, A-22), commit `be38494` (jax `984ce46..be38494`). Spec `docs/superpowers/specs/2026-09-16-hallazgos-auditoria-design.md` (sección A, A-01..A-55; A-21 queda para el frente D), plan `docs/superpowers/plans/2026-09-16-frente-a-limpieza-defectos-reglas.md`. Frontend `index-ChCOZGPN.js` (antes `index-Bxqm5swQ.js`).

**Qué se borró y qué se arregló, por bloque (16 tareas, A-01..A-55):**
- **Código muerto y recortes sin cambio de conducta (Task 1, A-02/03/04/12/15/18/19/20/28/33/34):** `aiosmtplib` fuera de `requirements.txt`; `TokenPayload` sin uso borrado; `_load_jax_env` (parser a mano del `.env`, podía re-inyectar en corridas manuales valores YA descifrados) borrado de `chat.py`/`image.py`; `_user_tenant_map` (nadie lo leía) fuera de `jax_engine/state.py`; los dos tokens (access/refresh) salen de un solo `_crear_token`; `CORSMiddleware` solo admite `FRONTEND_ORIGIN`; `db/transaccion.py::AISLAMIENTOS` cerrado a `READ COMMITTED` (nadie pedía otro nivel); el router legado `api/admin/facet_models.py` (desregistrado desde Bloque C, 2026-08-10) **borrado** — la tabla `facet_models` queda como dato histórico sin escritor.
- **Locks sin `await` entre chequeo y acción (Task 2, A-24):** `EventBus`, `WebSocketHub` y `ResourceManager` tenían un `asyncio.Lock` alrededor de dicts/sets sin ningún `await` adentro — en asyncio esas secciones ya corren sin interrupción, el lock nunca se disputaba. Se sacaron los tres; `lifecycle_lock` (sí serializa secuencias con `await`) no se tocó.
- **DEFECTO real, ahora HISTORIA — `_safe_path` del repositorio admin (Task 3, A-07/08/26/31/40):** comparaba con `target.startswith(base)` **sin separador de ruta**, así que `documents/../../repo-x/secreto.txt` pasaba si existía una carpeta hermana cuyo nombre empezara con "repo" — **leía y borraba fuera del repositorio, con acceso de superadmin**. Sin carpetas hermanas de ese patrón en producción (verificado), pero explotable si alguna vez existieran. Arreglado con `Path.resolve()` + `is_relative_to(base)`; MIME real por extensión (antes todo lo que no fuera `.png` salía `image/jpeg`); `POST /repo/save` (sin llamador ni tests) borrado.
- **Admin SMTP/llaves/catálogo (Task 4, A-25/39/46/47):** solo las ramas de excepción comunes se traducen a 502/503 con código estable; `UnicodeEncodeError` (contraseña SMTP no-ASCII) nunca vuelve a loguear la excepción completa.
- **Rutas obligatorias desde el entorno, fail-closed (Task 5, A-41/54/55):** `JAX_REPO_PATH`, `JAX_CONFIG_PATH`, `JAX_AUDIT_LOG_PATH`, `JAX_MISSIONS_DIR`, `JAX_BIN`, `JAX_REPO_BASE` pasan a ser **obligatorias** (antes tres caían a `~/jax` si faltaban). Único lector: `config_de_entorno.ruta_requerida(nombre)`. **Hallazgo de paso:** `test_command_path_traversal.py` escribía en el `~/jax/missions` REAL — el conftest pasó a fijar directorios temporales.
- **Tablero que miente (Task 6, A-05/06/35/36/37/49):** `/api/admin/dashboard` marcaba `alive` con `status_code < 500` (un 404 contaba como vivo); ahora exige 200. `JAX_PLATFORM_URL` nueva (si falta, la tarjeta dice `sin_configurar`, nunca `alive`). "Pipelines completados" cuenta el total con índice `idx_pipelines_status`.
- **Labels crudos de facetas (Task 7, A-42/48):** `display_name` sale de la tabla `facet`, leído UNA vez en el `lifespan` tras `run_migrations()` (invalidación declarada: el reinicio del proceso). El tablero mostraba la clave (`jekyll`), no el nombre ("Dr. Jekyll").
- **Chat sin texto libre en `detail` (Task 8, A-13/16/22/51/53):** el backend manda `detail.code` (y `aviso.code`+`params` para avisos de chat); el frontend traduce con `codigoDe()`/`textoDeAviso()`. **A-22, mirror-sync de las keywords del auto-ruteo:** los 12 nombres de los sets no coincidían entre `jax-platform/backend/api/chat.py` y `jax/core/router.py` (`_KIMI_KW` vs `KIMI_KW`, etc.) y `check_mirror_sync.py` los comparaba por nombre — daría "falta" en los 12 aunque el contenido fuera idéntico byte a byte. Renombrados en la plataforma; familia `router_keywords` agregada en jax (commit `be38494`).
- **Mesa (image/command/pipelines/upload) sin texto libre (Task 9, A-14/30/51/53):** mismos códigos estables; `command_completed`/`GET /api/command/{id}` con `code` (`comando_sin_resultado|comando_fallo|comando_simulado`). **Residual R16, arreglado antes del push (regla "sin hallazgos diferidos"):** `_leer_tarea` leía el owner file Y el archivo de resultado en el mismo hilo y cualquier `OSError`/`ValueError` de los dos mapeaba a `404 tarea_no_encontrada` — un error transitorio de LECTURA DEL RESULTADO (no del dueño) se volvía éxito falso permanente en el store del frontend (violaba A-44). Separado en dos lecturas: dueño ilegible sigue 404; resultado ilegible → `500 tarea_resultado_ilegible`.
- **Login por regex (Task 10, A-50):** bloqueo de cuenta con código estable `cuenta_bloqueada` (423) + `Retry-After`.
- **Frontend: `errores.js` e i18n con códigos estables (Task 11/12, A-09/10/11/17/27/29/43/44/45/51/52/53):** `textoDeErrorDeMesa`/`textoDeAviso` traducen los códigos nuevos; paridad es/en con las mismas claves; 25 claves muertas de i18n borradas.
- **Confirmaciones/avisos sin diálogos del navegador (Task 13, A-23/32):** `politica/modales.test.js` verifica que ningún overlay a mano exista fuera de `Dialogo.jsx`. Arreglado, en la raíz y no con un parche local, un defecto real de foco en `Dialogo.jsx` (el `focus()` corría en el cleanup antes de que el trigger dejara de estar disabled) que `AdminSmtp.jsx` tapaba con `flushSync`.

**Discrepancias con el spec 1-10 (verificadas contra `26c9cd5`; ninguna cambia la intención del ítem):**
1. A-22: los 12 nombres no coincidían entre los dos repos (arriba).
2. A-24: el test que leía `hub._lock` se reescribió — la propiedad que queda es "un fallo o una desconexión durante el close no frena a las demás".
3. A-48: `/api/state` no tocaba la base; se leyó `display_name` UNA vez en el lifespan (no por pedido) para no romper los tests puros ni agregar una ida a la DB al camino caliente (medido en 127 ms a c=25).
4. A-55: `JAX_REPO_PATH`/`JAX_CONFIG_PATH` tenían default y producción no las definía — se volvieron obligatorias las SEIS rutas. Quedan fuera, con motivo escrito: `model_catalog.py` (`~/.claude/.credentials.json`, ubicación que define Claude Code) y el `cwd=Path.home()` del subproceso `jax` en `command.py` (directorio de trabajo de la CLI, no ruta de datos).
5. A-51: `upload.py` también lo reescribe el frente D (coordinación de merge, ver el archivo del PR).
6. A-51/A-53: un `detail` objeto rompía un slice en `chat.py:1075` (`detail[:100]` con `detail` dict) — se usa `motivo[:100]`.
7. A-36: no existía variable para la base de jax-platform → `JAX_PLATFORM_URL` nueva.
8. A-49: "pipelines completados" sin ventana de fecha en el spec → se cuenta el total, literal, con índice.
9. A-35: el registro histórico de P10 en jax (`_count_configured_keys`) es HISTORIA de un triage de 2026-08-19 y no se edita.
10. **A-17 — pregunta para Fernando, SIN decisión registrada al cierre de este documento:** la fila `ws_notifications` ya existe en `axioma_config` de producción (`_ensure_defaults` hace `INSERT IGNORE`); borrarla de `DEFAULT_CONFIG` no la borra de la base y `GET /api/admin/config` la sigue listando. El frente A no tocó datos de producción; el plan recomendó que la borre el frente C (que ya migra `axioma_config` con dump verificado fila por fila). **Pendiente sin fecha:** que el frente C la borre o Fernando decida.

**Carga (Task 15) y EXPLAIN:** primera corrida (`2b7f217`) dio **NO-GO**: `/api/health` con 10 lectores subía 15,7×/17,7×/19,5× contra un límite de 5× — `_ultimas_20_lineas` del audit recorría los 50 MB línea por línea en Python compitiendo por el GIL. Arreglado leyendo desde el final del archivo: 1,28×/2,12×/1,55×, `/api/audit` p95 de 165-181 ms → 4,9-6,2 ms. De paso, `/api/admin/repo/file` con un `.png` de 2 MB también congelaba el loop (`read_bytes`+`b64encode` de 2 MB reteniendo el GIL en 25 lecturas paralelas): bajó a 3,6×-4,5× acotando la concurrencia de lectura; y `SELECT COUNT(*) FROM jax_users WHERE locked_until > %s` hacía `ALL` — arreglado con `idx_jax_users_locked_until` (EXPLAIN: `range`, `Using where; Using index`). **Gate GO** final (`664cb0c`): `/api/state` sin regresión (−12%..+1,9%), 0 5xx inesperados, invariantes de `/api/command` en las 500 tareas, EXPLAIN de las tres consultas del tablero sin `filesort` ni `temporary`.

**Las 9 variables agregadas al `.env` (deploy 2026-09-17 00:28):** `JAX_REPO_PATH`, `JAX_CONFIG_PATH`, `JAX_AUDIT_LOG_PATH`, `JAX_MISSIONS_DIR`, `JAX_BIN`, `JAX_REPO_BASE`, `JAX_PLATFORM_URL`, `JAX_SEED_SUPERADMIN_EMAIL`, `JAX_SEED_TENANT_NAME`. Backup `/etc/jax/.env.backup-pre-frente-a-20260917-002811`; `jax-platform.service` reiniciado, `NRestarts=0`, journal limpio, variables vivas verificadas en `/proc/PID/environ`. Frontend: backup `/www/wwwroot/axioma-ia.io.backup-pre-frente-a-20260917-002830`, servido `index-ChCOZGPN.js`. **Rollback:** restaurar el backup del `.env` y `git checkout 26c9cd5` + reiniciar.

**Incidente del deploy y la lección (2026-09-17):** `JAX_SEED_TENANT_NAME` se escribió sin comillas ("Inversiones Diamante Negro"). `systemd` (`EnvironmentFile=`) lo lee bien, pero `set -a; . <(sudo -n cat /etc/jax/.env)` (cualquier script que sourcea el `.env` en un shell) falla con `Diamante: command not found` — el archivo queda roto sin aviso para ese segundo lector. Corregido con comillas, backup `/etc/jax/.env.backup-pre-comillas-tenant-20260917-002841`, verificado que bash y `systemd-run -p EnvironmentFile` leen el mismo valor. **Regla nueva: todo valor con espacios en `/etc/jax/.env` va entre comillas y se verifica parseándolo con los DOS lectores (systemd y un shell que lo sourcea), no solo con el que se usó para escribirlo.** Misma familia que "un default a `$HOME` en un import hace que los tests escriban en producción" (`test_command_path_traversal`, arriba): un archivo de configuración compartido tiene más de un lector, y cada arreglo se verifica contra todos, no contra el que se tuvo a mano.

**Pisos y canario:** vitest 448→504, con DB 1175→1334 (1 skip ambiental), sin DB 614→764 — medidos dos veces cada uno. Canario por API sobre el sha real: rojo en `e418c82` (`backend-tests-con-db` y `backend-tests-no-db`), revert verde en `e6e67f8`.

**Carga en producción post-deploy:** k6 `health.js`, 10 VUs, p95 0,52 ms, 0 fallas de 698.696 peticiones, 19.963 rps.

**Deuda nueva que este cierre destapa:** ver "Anotado — deuda residual del frente A" en `## Anotado, no bloquea`, más abajo.

**Pendientes con fecha:**
- Verificación en vivo con Fernando (claro/oscuro, es/en): tablero, labels de facetas, chat traducido, Admin Modelos/SMTP/Pipeline con Escape — sin fecha de control fijada; se cierra cuando Fernando la haga.
- Discrepancia 10 (arriba): decisión sobre `ws_notifications`, recomendado para el frente C.

## Cerrado — tanda A: gobernanza con el catálogo de la DB, `capability.mode`, rol `plataforma` (2026-09-14)

**VERDAD OPERACIONAL 2026-09-14 19:50 CST** (despliegue verificado en vivo). jax-platform#71 (PR-A) → Jax#156
(PR-B) → jax-platform#72 (PR-C), mergeados en ese orden con gate por `headSha`; desplegados junto con
jax-platform#73 (PR-J, jekyll). Checkouts: jax `f36ef3f`, jax-platform `e05c5cc`; frontend
`index-CckN9i1-.js` (respaldo `axioma-ia.io.backup-pre-tanda-a-20260914-195227`, idéntico por `diff -rq`).
Spec `docs/superpowers/specs/2026-09-14-gobernanza-catalogo-db-design.md` v3.1; plan con enmiendas v3/v4.

- **Cerrados de "Bloquea trabajo":** (1) *el resolver de `CAPABILITY_AVAILABLE` consultaba un catálogo que
  el Bloque 3 vació* — el validador recibe `await MotorCatalog.from_db()` y la rama `in_catalog` verifica
  nombre **y modo**; el snapshot suma la sección `catalog_capabilities` (`/catalog_capabilities/N`, sin mover
  `/capabilities/N`). HECHO corregido en la planificación: nunca hubo `FACT_MISMATCH` históricos (427 VALID,
  3 POINTER_MISMATCH, 1 ARGS_MISMATCH, 2 AUTHORITY_INVALID sobre `code_swarm`): el defecto real era que las
  capabilities de la DB no se podían **citar**. (2) *`invoked_by` era un nombre de persona usado como
  autorización* — ahora es el rol `plataforma`, lo pone el backend (pisa lo que mande el cliente); las filas
  viejas con "Fernando" quedan como historia.
- **`capability.mode` = `VARCHAR(16) NOT NULL` + `CHECK chk_capability_mode`** (spec v3). HECHO medido: con
  `ENUM ... NOT NULL` sin default, MariaDB 12.3 guarda el primer valor en silencio si se omite (fail-open);
  con VARCHAR+CHECK: omitir → 1364, inválido → 4025, NULL → 1048. Solo `file_write` es `mutating`.
- **Incidente 2026-09-14 ~15:11 CST (HISTORIA):** un script de verificación de la Tarea 1 cargó
  `/etc/jax/.env` fuera de pytest y corrió `run_migrations()` (versión ENUM), una mutación `ALTER ... DEFAULT`
  y su reversión, y filas `zz_test_probe*` borradas, contra `jax_memory` de producción. Verificado: 17 filas
  correctas, sin restos, servicios sanos. Decisión de Fernando: se dejó y se registra; el despliegue la
  convirtió a v3. **Causa:** el brief no advertía que `/etc/jax/.env` apunta a producción fuera de pytest.
  **Barrera:** todo brief con DB lleva la advertencia textual y el controller verifica prod de forma
  independiente (memoria `feedback-brief-barrera-db-produccion`).
- **Gobernanza async:** `governance_context.validation_context()` con recarga **compartida** por stamp
  (`asyncio.shield`) y acotada por `GOVERNANCE_RELOAD_TIMEOUT_SECONDS` (5.0); invalidación por el sello de
  facet_resolver; falla visible (SnapshotError en el chat, sin veredictos en sombra).
- **`connect_timeout` en todo aiomysql** de jax y jax-platform (`JAX_DB_CONNECT_TIMEOUT_SECONDS`, 10;
  aiomysql 0.3.2 no tenía límite): helper espejo `db_connect_config` + tripwires AST en los dos repos. Acota
  el socket TCP; la recarga completa la acota el timeout de arriba.
- **Hallazgos arreglados en el camino:** arnés `client` de tests (una falla dentro de `portal.call` mataba el
  portal de sesión: 635 → 352 passed / 184 failed); symlink `las_manos/jacobs` absoluto a
  `/home/fruiz/jax/jacobs` → relativo `../jacobs`.
- **Verificación en vivo:** columna `varchar(16)`/NO/sin default + `chk_capability_mode`; resolver con el
  contexto de prod (17 capabilities, ops∩DB vacío, 28 entradas, VALID/FACT_MISMATCH según modo); sonda de la
  Mesa por `run_shadow_validation` (read_only → VALID/OBSERVADO, mutating → FACT_NOT_IN_SNAPSHOT/INFERIDO);
  pipeline supervised creado con `invoked_by` falso → `plataforma`, `PIPELINE_RESUMED {"by": "plataforma"}`;
  journal limpio.
- **Rendimiento (LAS CUATRO #4), hall9000:** `validation_context()` en producción — frío p95 2,70 → 4,53 ms,
  caliente p95 4,42 → 5,31 µs; concurrente con la DB real: 50 turnos fríos simultáneos → **1** `from_db`,
  p95 4,03 ms. Snapshot 11 → 28 entradas, render 715 → 1.786 chars (~+270 tokens por chars/4). `tokens_in`
  de una sonda de kimi: 1720 → 3448, pero la sonda "después" arrastró el historial en memoria del turno
  anterior de jekyll (ChatRequest no permite conversación nueva): la comparación no aísla el snapshot.
- **Pisos:** jax-platform 709 / 349 / vitest 146; jax governance 90, tests-puros 176, jacobs-gobernanza-db 15.
- **SP4:** la línea base del 2026-09-03 deja de ser comparable (ver el ítem de SP4).

## Cerrado — el REPL, Ada y el executor toman modelo y tope del catálogo (PR-K, 2026-09-14)

**VERDAD OPERACIONAL 2026-09-14 ~21:29 CST.** jax#158 (`524edb7`), desplegado y verificado. Antes, jax
tenía modelo y `max_tokens` fijos en código en tres caminos: el REPL (`jax/muscles/base.py`), Ada
planificando (`ADA_MODEL`/`ADA_URL`) y el Motor Registry. Ahora:

- `jax/core/contrato_dispatch.py`: los validadores del contrato, copia textual de los de jax-platform
  (familia nueva en `scripts/check_mirror_sync.py`, así que el CI rompe si divergen).
- Ada planifica con `resolve_facet("ada")` (hoy glm-5.3); `ADA_MODEL` y `ADA_URL` eliminados.
- Si un cerebro falla (Ada → qwen → plan fijo), el pipeline deja un evento `PLAN_CEREBRO_FALLBACK` en
  `jacobs_events` con el motivo, además del log.
- El Motor Registry manda min(contrato, `motor.max_tokens`) con el parámetro del contrato; thot manda
  `max_completion_tokens` 128000.
- CI: `_direct_usage_test.py` pasó al job `jacobs-gobernanza-db` (necesita el esquema de jax-platform);
  los tests con DB crean sus propias filas. Pisos: tests-puros 285, jacobs-gobernanza-db 27,
  facet-health-io 8.

**Verificado en vivo:**
- Contrato con el código desplegado: ada, jekyll, kimi y thot OK; hipatia y hyde no exigen; jax_local 262144.
- jekyll 200 en 2,46 s; journal sin warnings desde el deploy.
- Pipeline `5d64a7c4`: objetivo trivial → cerebro qwen; paso thot con gpt-5.6-terra completo.
- Pipeline `ecacbc72`: objetivo formal → `Jacobs cerebro=Ada (formal)`, sin `PLAN_CEREBRO_FALLBACK`; paso
  ada con glm-5.3 completo.

## Cerrado — declarar el contrato de dispatch sin SQL a mano (PR-L, 2026-09-14)

**VERDAD OPERACIONAL 2026-09-14 ~21:07 CST.** jax-platform#74 (`99671eb`), desplegado y verificado: el
guard de #73 (409 al aprobar un modelo sin contrato) dejaba como único remedio SQL a mano, contra la regla
"cambiar el modelo NUNCA es un UPDATE a mano". Ahora `PUT /api/admin/models/{ref}/contrato-dispatch`
(superadmin, mismos validadores que el dispatch, tope ≤ INT de la columna, transacción, auditoría en
`model_catalog_audit` sin FK duras y con snapshot legible, invalida el sello). Los 409 de approve y de PUT
quedan registrados y se ven en Propuestas y Bindings con "Declarar contrato" (i18n es/en). Semillas de base
vacía coherentes (thot → gpt-5.6-terra, ada → glm-5.3; antes nacían sin contrato) con tripwire;
`/api/admin/keys` sin nombres de modelo literales. **Siembra de `ollama/qwen3.6:35b-a3b-q4_K_M` (jax_local):
`max_output_tokens = 262144`** — decisión de Fernando: su contexto (`ollama show`; la doc de Ollama dice que
`num_predict` por defecto es -1 = generación infinita, o sea que hoy no tenía tope). **Verificado en vivo:**
tabla de auditoría creada, qwen NULL/262144, jekyll 200, admin 200, frontend `index-LN3fcXbx.js`.
**Carga** (arnés local, `jax_memory_test`): 0 errores; p95 1,54 / 6,36 / 63 ms con c=1/10/50 (degrada entre
10 y 50; aceptado para un endpoint de superadmin esporádico).

- **Incidente menor (HISTORIA, 20:26:05 CST):** esa prueba de carga corrió con un arnés que aislaba la DB
  pero NO el sello de facet_resolver: cada PUT estampó `/srv/jax-data/facet-cache-seal` real y los
  servicios recargaron sus cachés una vez (sin cambio de datos; health 200, journal limpio). Arnés
  arreglado y verificado; la barrera de los briefs ahora incluye el sello (memoria
  `feedback-brief-barrera-db-produccion`).
- **CERRADO 2026-09-15 por la etapa 5 (jax-platform#83 → `a703c67`).**
  - La baja lógica (`POST /users/{id}/baja`) reemplaza al `DELETE` en duro, que ahora responde 405.
  - Es un UPDATE de columnas que no son clave y no cambia `user_id`. Por eso ninguna FK interviene.
    Verificado con un SELECT de solo lectura en `information_schema` de producción:
    - `credential.created_by`, `credential_audit.performed_by`, `facet_binding.approved_by` y
      `model_binding_proposal.decided_by` tienen `ON UPDATE RESTRICT` y `ON DELETE RESTRICT`.
    - `password_reset_tokens` y `user_api_keys` tienen `ON DELETE CASCADE`.
  - Ninguna auditoría queda bloqueada, y el historial de un actor dado de baja se sigue resolviendo
    (`user_audit` hace LEFT JOIN por `user_id`).

  Texto original:
  **Pendiente con fecha — etapa 5 de admin usuarios (baja lógica):** `credential_audit.performed_by`,
  `facet_binding.approved_by` y `model_binding_proposal.decided_by` tienen FK a `jax_users`: el
  `DELETE /api/admin/users/{id}` en duro de hoy da 500 para un usuario con historia. La etapa 5 reemplaza
  el borrado por una baja; al ejecutarla, verificar que ninguna auditoría quede bloqueada.

## Cerrado — jekyll caída en producción (2026-09-12 19:40 → 2026-09-14 19:50 CST)

**HISTORIA + VERDAD OPERACIONAL.** El catálogo detectó que DeepSeek renombró `deepseek-v4-flash` →
`deepseek-flash` (propuesta de drift #11) y al aprobarla el binding de jekyll pasó a la fila 2111, sin
`max_tokens_param` ni `max_output_tokens`, que el dispatch `http_openai_compat` exige (fail-closed). **Causa
raíz:** aprobar / PUT de facet-bindings no verificaba ese contrato. El canario lo detectó al instante y la
alerta llegó por Telegram (09-14 13:48, message_id 666), pero la faceta quedó caída ~48 h. **Arreglo**
(jax-platform#73): migración siembra `deepseek-flash` y `deepseek-v4-pro` con `max_tokens` / 393216 (doc
oficial de DeepSeek, leída el 2026-09-14; decisión de Fernando: el máximo documentado) y los dos escritores de
`facet_binding` responden 409 si el modelo destino no cumple el contrato de su transporte
(`modelo_sin_contrato_de_dispatch`) o es de otro proveedor (`modelo_de_otro_proveedor`). **Verificado en vivo:**
jekyll 200 a las 19:50:32, `facet_health_event` ok. **Siguen en esta sesión:** PR-L (camino admin auditado para
declarar el contrato de un modelo sin SQL a mano, y rastro del rechazo) y PR-K (el REPL `jax/muscles/base.py`
y Ada `jacobs/plan.py` mandan `max_tokens: 131072` fijo; pasan a leer el catálogo).

## Cerrado — pendientes del 2026-09-22 y la fila propia sin auto-acciones (2026-09-15)

**VERDAD OPERACIONAL 2026-09-15 17:21 CST.** jax-platform#85 → `a7780a5`, con jax#162 → `e2c4e2d` y jax#163 → `3e1f4f2` desplegados antes. Plan: `docs/superpowers/plans/2026-09-15-pendientes-2026-09-22.md`.

**Despliegue:**
- Frontend servido: `index-DepcGKgk.js` / `index-CFy-nEeS.css`. El md5 coincide con el del build.
- Backup `…backup-pre-pendientes-20260915-172147`, verificado idéntico antes del rsync.
- Dump previo `~/backups/usuarios-pre-pendientes-20260915-172133.sql`, verificado fila por fila (incluye `provider` y `axioma_usage`).
- **Sí hubo migración:** el ENUM `provider.api_key_transport` suma `header_goog_api_key`, la fila `gemini` pasa a usarlo y se crea el índice `idx_axioma_usage_periodo`. Verificado después del restart.
- **La sonda de hipatia respondió `ok` después del restart:** Google acepta la cabecera con la key de producción.

Pedido de Fernando (2026-09-15): "escóndelos los dos, y termina los pendientes del 2026-09-22".

- **Fila propia:** Admin → Usuarios ya no muestra "Fijar contraseña" ni "Dar de baja" en la fila del admin logueado. Esto revierte el Ruling F7. El backend los sigue rechazando con 403. El admin cambia su propia contraseña desde Mi cuenta.
- **CERRADO — AdminRepository con `window.confirm`:**
  - Borrar un archivo ahora pasa por ConfirmacionSuma.
  - La vista previa es un `Dialogo`, con #root inert, foco y Escape, y hay un solo modal a la vez.
  - Si una respuesta tardía llega, no cierra el diálogo de otro archivo.
  - Queda cero `window.confirm` en `src`.
- **CERRADO — historial de las bajas (U36):**
  - `GET /api/admin/users?bajas=true` (solo superadmin) devuelve `email_original`, que es lo que está antes del último `#baja-`, y `deleted_by_email`, que sale de un LEFT JOIN por PK.
  - El interruptor "Mostrar bajas" muestra esas filas en solo lectura, con la acción Historial.
  - La lista por defecto no cambió.
  - Carga a c=10: p95 20,8 ms. Satura entre c=10 y c=25 con 1000 usuarios, por CPU de Python.
- **CERRADO — endurecer `test_no_fail_open_except`:** _(Task 3; se completa)_. Estado actual:
  - Regla nueva: todo `except` amplio que no relanza lleva `# fail-soft: <razón>` en la línea del `except`. Loguear no exime.
  - Se auditaron 29 sitios:
    - 22 fail-soft legítimos, que ahora llevan la marca;
    - 2 eran ruido y se estrecharon o se quitaron;
    - 5 eran fail-open reales y se arreglaron con test: `chat.py` import y puerto, `audit.py` ilegible → 503, `models.py` sync ya no responde ok cuando falla.
  - El escaneo AST quedó en 0 sitios sin marca.

**Pruebas:**
- Con DB: 923 → 1076/1.
- Sin DB: 399 → 527.
- vitest: 404 → 421.
- Carga (U29) a c=25: `/api/state` p95 127 ms (antes 3288 ms), `/api/pipelines` 28,6 ms, chat con ids inválidos 14,4 ms, `/api/admin/usage` 334 ms el día y 452 ms el mes (antes ~2,7 s).

**Hallazgo:** el test estaba ciego en los worktrees. Salteaba cualquier ruta que contuviera `worktrees`, así que ahí no revisaba nada; CI no se veía afectado. Se arregló en jax-platform y quedó un test guardián.

- **PENDIENTE con fecha 2026-09-22 (propuesta):** la copia del test en `jax/policy/tests` probablemente tiene el mismo punto ciego y no tiene la regla nueva. Hay que llevarle la regla y el guardián con el mismo proceso.
- **CERRADO — hallazgos de seguridad de la auditoría** (decisión de Fernando del 2026-09-15: se arreglan en la misma rama):
  - **S1:** la key de Gemini ahora viaja en `x-goog-api-key`, tanto en la plataforma como en jax (jax#162 → `e2c4e2d`, desplegado).
    - `redactar_secretos` tiene reglas idénticas en los dos repos y aplica "redactar antes de recortar".
    - Hay un filtro de logs para httpx.
    - No hubo fuga guardada: 0 de 3061 filas y 0 líneas de journal tenían la key, así que no hubo que rotarla.
  - **S2:** `/api/state` y `/api/pipelines` devuelven solo lo del dueño.
    - El índice `idx_jacobs_pipelines_duenio` lo crea jax, que es el dueño de la tabla. Se construye con INPLACE/LOCK=NONE y con la espera de bloqueo acotada.
  - **S3:** `/api/audit` y `PUT /api/facets/{facet}/status` quedan solo para superadmin.
- **CERRADO — registro de uso (parcial, por decisión):**
  - Existe el contador `registros_perdidos`, visible en Admin → Costos.
  - Los ids se validan antes del LLM.
  - `/api/admin/usage` ahora es sargable, tiene índice sobre `created_at` y ordena por `SUM(cost_usd)`. Antes, con 102k filas, daba p95 de 2,7 s.
- **CERRADO 2026-09-15 — cola durable con reintento (era el PENDIENTE del 2026-09-29).**
  Desplegado: jax-platform#87 (`6443273`) + jax#165 (`d7fac0d`). La fila que no entra a
  `axioma_usage` ya no se pierde: se guarda en `/srv/jax-data/usage-spool` (un archivo por
  fila, `tmp`+`os.replace`, `fsync`) y una tarea de fondo la reinserta con `spool_id` e
  `INSERT IGNORE` contra un UNIQUE — un duplicado es un éxito, la fila ya está cobrada.
  Entran **los tres** escritores (plataforma, `jacobs`, `motor_registry`); sólo la plataforma
  drena. Verificado de punta a punta en producción: una fila depositada a mano en el respaldo
  llegó sola a la tabla en menos de un minuto, conservando la hora del TURNO, y el respaldo
  volvió a 0.
  - **Propiedad declarada, no descubierta:** con el intervalo de 60 s y el lote de 500, la
    recuperación va a **500 filas por minuto**. 10.000 filas = 20 min; el tope de 50.000 =
    1 h 40. Y la cola sólo baja si los fallos llegan a menos de ~8,3 por segundo.
  - **Lo que la prueba de carga encontró y se arregló antes de desplegar:** `encolar` corre
    en el camino del turno del usuario y pagaba O(n) en la profundidad de la cola — costo
    cuadrático sobre una caída, y realimentado. Medido a c=25: 6.834 ms por turno con la cola
    en el tope. Arreglado (un solo recorrido, barato): **286,7 ms**, y el throughput durante
    una caída pasó de 3,65 a 84,8 turnos/s.
  - **Límite conocido, medido y aceptado:** `encolar` toma un `asyncio.Lock` de todo el
    proceso, así que durante una caída el throughput topa en ~220 turnos/s sin importar la
    profundidad, y la latencia crece lineal con la concurrencia. Es acotado y no se
    realimenta. No se ataca hoy.
- **CERRADO 2026-09-16 — la fila venenosa** (era el PENDIENTE del 2026-09-29; Fernando:
  *"hacela ahora, no esperes al 29"*). Desplegado: jax-platform#88 → `9d05df2`. Tras 3
  intentos, la fila que la base rechaza siempre se mueve a `rechazadas/` con el motivo al
  lado, se cuenta y se ve en el monitor de costos.
  - Lo difícil no era la cuarentena, era **no aplicarla de más**: si se confunde "falló el
    ciclo porque la base está caída" con "falló esta fila porque la base la rechaza", una
    caída larga manda TODA la cola a cuarentena. Tres barandas, ejercitadas por mutación.
  - Se clasifica por **código** de error, no por clase de excepción: `pymysql` deja sin
    mapear el 1292 (fecha imposible) y lo manda a `OperationalError`, la misma clase que el
    1213 (deadlock) y el 1040 (demasiadas conexiones). Mirar la clase confundiría justo los
    dos casos que hay que separar.
- **CERRADO 2026-09-16 — el lock global de `encolar`** (era el límite declarado el 09-15;
  Fernando: *"arregla el lock global"*). Desplegado: jax-platform#89 → `26c9cd5` y
  jax#169 → `a42e396`. Los dos `fsync` salieron de la sección serializada.
  - Medido a c=25 con la base caída: **290 → 21 ms** en el tope, throughput **85 → 1.181
    turnos/s**, y **la profundidad de la cola dejó de importar** (17→21 ms de 0 a 49.000,
    contra 113→290 ms antes).
  - **La versión obvia era 7× PEOR y se midió antes de shipearla:** sacando el conteo del
    lock, 25 hilos contando el mismo directorio cuestan 230 veces uno, no 25 (bucle de
    Python con el GIL + `getdents` concurrentes). El trabajo no había que paralelizarlo,
    había que **no repetirlo**: una sola medición en vuelo.
  - **Lo que se paga, acotado:** el tope se puede sobrepasar en *(llamadas en vuelo − 1)* —
    24 filas sobre 50.000 a c=25. No deriva: el régimen queda en `[tope−1, tope+c−1]` y el
    error es sólo hacia arriba. `en_cola` se atrasa hasta ~c filas; `contar_pendientes()`
    sigue siendo exacto.
  - **Recuperación revisada:** con el lote en 500 y el intervalo de 60 s siguen siendo 500
    filas/minuto (20 min para 10.000, 1 h 40 para el tope). El lock no cambia eso: cambia lo
    que cuesta ENCOLAR durante la caída, no lo que tarda en drenar después.
- **PENDIENTE con fecha 2026-09-22:** llevar a `jax/policy/tests/test_no_fail_open_except.py` la regla nueva ("todo `except` amplio que no relanza lleva marca") y el guardián de cobertura. Hoy solo marca los `except: pass`.

## Cerrado — admin fija la contraseña, cambio obligatorio y sesión única (2026-09-15)

**VERDAD OPERACIONAL 2026-09-15 14:24 CST.** jax-platform#84 → `0b81ad7`. Plan:
`docs/superpowers/plans/2026-09-15-admin-fijar-password.md`, con la enmienda de la Task 3b.

**Deploy:**
- Frontend servido `index-BlCQtW_s.js`, con md5 igual al build (`f6ba6acc…`).
- Backup `axioma-ia.io.backup-pre-fijar-password-20260915-142422`, verificado idéntico antes del rsync.
- Dump previo `~/backups/usuarios-pre-fijar-password-20260915-142417.sql` (2 = 2, 1 = 1, 4 = 4), verificado fila por fila. El script aborta si no cuadra.
- Tras la migración, `must_change_password` con 0 filas marcadas (el script aborta si hay alguna).
- Sonda sin token: `POST /api/admin/users/{id}/password` responde 401. Antes daba 405.

**Decisiones de Fernando (2026-09-15):**
- El superadmin fija a mano la contraseña de otro usuario. Revierte U2; el enlace de recuperación se queda.
- El usuario tiene que cambiarla en su próximo login:
  - la nueva tiene que ser distinta de la que puso el admin (P1);
  - se le sigue pidiendo la actual (P2).
- **Sesión única:** una persona no puede tener dos sesiones abiertas. Salir mata la sesión en el servidor, y un login nuevo mata la vieja.

**Qué entra:**
- **Migración:** `jax_users.must_change_password`. La marca viaja en la única consulta de sesión (SELECT por PK), sin consultas extra.
- **Cumplimiento en el backend, negado por defecto (U34).** Con la marca puesta, solo pasan `/me`, `/me/password`, `/refresh` y `/logout`, que son los tres sitios de opt-in deliberados. Todo lo demás responde 403 `cambio_de_password_requerido`, incluidos WS y SSE.
  - Un test recorre las rutas servidas con `iter_route_contexts`, porque en FastAPI 0.139.2 `app.routes` no aplana los routers.
  - Un guard por AST impide otros opt-in e incluye los alias de import (F3). La indirección dinámica queda fuera de alcance (F4).
- **`POST /api/admin/users/{id}/password`:**
  - solo superadmin; nunca sobre sí mismo; 404 para una baja; inactivos permitidos;
  - bcrypt fuera de la transacción, que corre en READ COMMITTED con orden fijo;
  - un UPDATE que cambia hash, `token_version + 1`, la marca y el bloqueo;
  - borra los enlaces pendientes y audita `password_set_by_admin` sin la contraseña;
  - corta las sesiones después del commit.
- **F5:** completar un enlace de recuperación limpia la marca. Mientras la marca esté puesta, no acepta la contraseña del admin (400, token sin consumir), con bcrypt sin lock y re-chequeo bajo el bloqueo.
- **Sesión única (F2):**
  - el login exitoso sube la versión con UPDATE más relectura bajo el bloqueo de la fila, y solo si el hash verificado sigue siendo el actual (arreglo F1 del review);
  - el logout mata solo su propia sesión (arreglo F2 del review);
  - un login fallido nunca expulsa a nadie.
- **Frontend:**
  - modal "Fijar contraseña";
  - Mi cuenta obligatoria y no cerrable, con `Dialogo cerrable={false}`;
  - el interceptor convierte el 403 en la marca, sin bucles;
  - el logout avisa al servidor;
  - el aviso "se inició sesión en otro lugar";
  - F5 en ResetPassword.

**Carga medida antes del merge (U29), arnés pytest aislado:**
- p95 de `verificar_sesion` contra la base `a703c67`: `/me` 1,007 y 1,029; `/facets` 0,992 y 1,006, este último re-medido con 11 pares alternados porque el primer 1,103 era ruido (F6).
- El 403 cuesta lo mismo que un 401.
- Fijar a c=10: p95 162 ms. Logout: 13,6 ms. Reset con la marca: 316 ms, por un bcrypt de más.
- Carrera de K logins del mismo usuario: siempre exactamente un token válido.
- 0 respuestas 5xx y 0 deadlocks.

**Pruebas:**
- Con DB: 846 → 890/1. Sin DB: 371 → 374/517. vitest: 339 → 379.
- Cada arreglo de concurrencia tiene su test y se vio en rojo por mutación.

**Decisión de UI (Ruling F7):**
- ~~En la fila del propio admin, los botones de auto-acción ("Fijar contraseña", "Dar de baja") siguen visibles.~~
  **CERRADO 2026-09-15** (jax-platform#85 → `a7780a5`): Fernando decidió esconderlos ("escóndelos los dos"), y se
  esconden los dos juntos. El backend los sigue rechazando con 403 `auto_accion_prohibida`, como defensa en
  profundidad. El admin cambia su propia contraseña desde Mi cuenta.

**Límite aceptado:**
- Si el `POST /auth/logout` vence (a los 5 s) o falla la red, el cliente limpia su estado igual, pero la sesión sigue viva en el servidor.
- Es lo mismo que pasaba antes con el logout solo local.
- Esa sesión muere en el próximo login, porque ahora cada login invalida las sesiones anteriores.

**Pendientes con fecha:**
- ~~**2026-09-22 (propuesta):** endurecer `test_no_fail_open_except`.~~ **CERRADO 2026-09-15** (jax-platform#85 → `a7780a5`):
  todo `except` amplio que no relanza lleva `# fail-soft: <razón>`; loguear no exime; las funciones anidadas cuentan;
  un archivo que no se puede leer es violación. Se auditaron 29 sitios: 22 con marca, 2 eran ruido, 5 eran fail-open
  reales y se arreglaron. El test estaba ciego en los worktrees (se salteaba toda ruta con `worktrees`): arreglado,
  con un test guardián de cobertura.
- **Verificación en vivo de Fernando:**
  - fijar la contraseña a un usuario de prueba;
  - login con la marca: el diálogo no se cierra y se rechaza la misma contraseña;
  - una segunda sesión expulsa a la primera con el aviso;
  - salir mata la sesión;
  - todo en claro y en oscuro.

## Cerrado — admin usuarios etapa 5: editar el correo y dar de baja con ConfirmacionSuma (2026-09-15)

**VERDAD OPERACIONAL 2026-09-15 12:14 CST.** jax-platform#83 → `a703c67`. Plan:
`docs/superpowers/plans/2026-09-12-admin-usuarios-etapa-5-editar-baja.md`. Spec: §2, §3.2 y §3.5.
Frontend servido `index-d04xNGjp.js`, con md5 igual al build (`3916bf45…`). Backup
`axioma-ia.io.backup-pre-usuarios-etapa5-20260915-121449`, verificado idéntico antes del rsync.
Esquema verificado después con un SELECT: `email varchar(320)` con índice UNIQUE, `deleted_at
datetime`, `deleted_by int(11)`; `jax_users` sigue con 2 filas. Health 200, journal sin errores, cwd
= checkout con el commit. Sondas sin token: `POST /baja` pasó de 405 a 401 y `DELETE` de 401 a 405.

- **Incidente del backup previo (HISTORIA).** Antes del reinicio que aplica la migración se hizo el
  dump `~/backups/usuarios-pre-etapa5-20260915-121445.sql` (`jax_users`, `password_reset_tokens` y
  `user_admin_audit`).
  - El chequeo de filas del script informó `dump=0 DISTINTO` y el deploy **siguió igual**: imprimía,
    no cortaba. El regex esperaba `VALUES (` en la misma línea y mariadb-dump escribe una fila por
    línea.
  - Verificado después: el dump estaba completo y cuadra fila por fila con la DB (2 = 2, 1 = 1,
    4 = 4). La restauración no se probó.
  - La migración solo agrega y ensancha columnas, así que no se perdió nada.
  - Arreglo: el script cuenta bien las filas y **aborta** si no cuadran. Un chequeo que no corta no
    es un gate.

Qué entra:
- **"Eliminar" pasa a ser "Dar de baja"**, detrás de **ConfirmacionSuma** ("Resolvé a + b = ?"). El
  botón solo se habilita con la respuesta correcta. El componente está construido sobre `Dialogo` y
  sirve para cualquier borrado futuro.
- **La baja no borra la fila.** Deja `status='deleted'`, `deleted_at`/`deleted_by` y
  `token_version + 1`. Además:
  - Se cortan WS/SSE después del commit y se borran los enlaces de recuperación pendientes.
  - El correo se renombra (`<correo>#baja-<id>-<fecha>`), así que la dirección queda libre. El
    original queda en la auditoría.
  - El historial se conserva.
- **Transacción de la baja.** Corre en `transaccion(AISLAMIENTO_ADMIN)`, con el orden fijo de la
  etapa 3. Nadie se da de baja a sí mismo, y no se puede dar de baja al último superadmin activo.
- **`DELETE /users/{id}` responde 405.** `GET /users` excluye las bajas, y el enlace de recuperación
  a una baja responde 404.
- **Editar el correo** (PUT), con validación y códigos estables. El alta también devuelve códigos
  estables: `email_invalido`, `rol_invalido` y `email_ya_existe`. Un correo repetido nunca da 500,
  ni siquiera cuando la carrera pasa la comprobación previa.
- **Carreras cerradas en esta etapa:**
  - **"Enviar enlace" contra la baja (U31).** Bloquea al usuario por PK y crea el token en la misma
    transacción. El SMTP se envía después del commit.
  - **El forgot-password público (U31, U33).** Vuelve a bloquear por PK, nunca por el índice de
    email. Corre en READ COMMITTED: en REPEATABLE READ, el DELETE de tokens por el índice no único
    `user_id` tomaba gap locks, y dos pedidos de usuarios distintos terminaban en 1213.
  - **El dominio del correo solo admite `[A-Za-z0-9-]` (U32).** Antes aceptaba `#`, y alguien podía
    ocupar de antemano el nombre de baja de otro usuario.
- **Frontend:**
  - un modal a la vez;
  - después de la baja, el foco va a "+ Nuevo usuario" (U35);
  - la barra muestra al instante el correo propio editado;
  - i18n es/en completo, y se borraron las claves muertas.

**Antes de la baja se verificó, solo con lecturas contra producción:**
- Todas las FK a `jax_users.user_id` son `ON UPDATE RESTRICT`, y la baja no cambia `user_id`, así
  que ninguna auditoría queda bloqueada.
- `user_api_keys` guarda credenciales de proveedores, no sirve para autenticar.
- Login, sesión, refresh y forgot-password rechazan cualquier cuenta que no esté activa.

**Carga medida ANTES del merge** (gate U29; arnés pytest aislado sobre `8c34800`, ASGI en proceso):
- Con c=10: baja p95 12 ms (~1000 rps), PUT con email p95 11 ms, y la lista (131 usuarios, 100 de
  ellos dados de baja) p95 23 ms.
- Dos superadmins que se dan de baja mutuamente: 20 de 20 terminan con un 200 y un 409
  `ultimo_superadmin`.
- Toda carrera tuvo un solo ganador. 0 errores 5xx y 0 deadlocks.
- Saturación en ~1000 rps desde c≈10, por diseño: toda escritura de admin bloquea primero el conjunto
  de superadmins.

Pruebas: con DB 815 → 845/1; sin DB 369 → 371/475; vitest 315 → 336. Decisiones: Rulings U10-U12,
U20 y U28-U35 en el ledger.

- ~~**Pendiente con fecha 2026-09-22 (propuesta):** `AdminRepository.jsx` sigue borrando con
  `window.confirm` + `api.delete`.~~ **CERRADO 2026-09-15** (jax-platform#85 → `a7780a5`): borra con
  ConfirmacionSuma, la vista previa es un `Dialogo` con un solo modal a la vez, y no queda ningún
  `window.confirm` en `src`.
- ~~**Pendiente con fecha 2026-09-22 (propuesta), Ruling U36:** el historial de un usuario dado de
  baja no se puede abrir desde la UI.~~ **CERRADO 2026-09-15** (jax-platform#85 → `a7780a5`): el interruptor
  "Mostrar bajas" lista las bajas en solo lectura, con Historial como única acción, sobre
  `GET /api/admin/users?bajas=true` (solo superadmin).
  - La lista oculta las bajas y es la única entrada al modal de Historial. `GET /users/{id}/audit`
    sí devuelve esas filas.
  - No es una regresión: antes, un `DELETE` exitoso también sacaba al usuario de la lista.
  - Hace falta una vista de "bajas" o un filtro en la lista que permita abrir su historial.
- **Pendiente con fecha, verificación en vivo de Fernando:** la etapa 5 en claro y en oscuro
  (ConfirmacionSuma, correo en "Editar", toasts y la barra con el correo propio). Nadie la miró en un
  navegador todavía.

## Cerrado — admin usuarios etapa 4: Mi cuenta y enlace de recuperación por admin (2026-09-15)

**VERDAD OPERACIONAL 2026-09-15 06:03 CST.** jax-platform#82 → `453b128`. El plan es
`docs/superpowers/plans/2026-09-12-admin-usuarios-etapa-4-contrasenas.md` y cubre la spec §3.2 y §3.4.
El frontend servido es `index-BvtjkH_S.js`, con md5 idéntico al build local. Antes del rsync se hizo
el backup `axioma-ia.io.backup-pre-usuarios-etapa4-20260915-060236` y se comprobó idéntico al sitio.

Verificación después del despliegue: health 200 y journal sin errores. El cwd es el checkout con el
commit. `POST /api/auth/me/password` y `POST /api/admin/users/{id}/reset-link` responden 401 sin
token; antes del despliegue daban 405, así que las rutas quedaron vivas.

Qué entra:
- **Regla única de contraseña.** Mínimo 8 caracteres, contados como puntos de código, y máximo 72
  bytes. Está en backend (`auth/password_rules.py`) y frontend (`lib/reglasPassword.js`). La usan el
  alta, `/reset-password` y Mi cuenta. Antes el alta daba 500 con más de 72 bytes.
- **Mi cuenta** (`POST /api/auth/me/password`).
  - Exige la contraseña actual y tiene el límite del login.
  - Sube `token_version`: las otras sesiones mueren y esta recibe tokens nuevos.
  - Corta WS/SSE después del commit.
  - Una revocación o desactivación concurrente da 401 `sesion_invalida`: la re-lectura
    `FOR UPDATE` fija hash, `token_version` y `status`.
- **Enlace de recuperación por admin** (`POST /users/{id}/reset-link`, solo superadmin).
  - Si SMTP no está configurado o está dañado, 503. Si el envío falla, 502 con la respuesta del
    servidor. Si el usuario no existe, 404; si no está activo, 409.
  - Ante cualquier salida sin envío, incluida la cancelación, se borra el token exacto (U17).
  - Si falla la auditoría después de un envío exitoso, es fail-soft (U22).
- **Reset completado.**
  - Bloquea el usuario y después reclama el token de forma atómica; el orden tiene test.
  - Si el usuario no está activo, 400 `reset_token_invalido`, sin consumir el token (U21).
  - Sube `token_version` y corta las sesiones.
- **`_cortar_conexiones` pasa a `auth/conexiones.py`**, lo que rompe el ciclo auth↔admin.
- **Un único `components/Dialogo.jsx` para los cuatro modales de usuarios** (U27). Pone el portal,
  deja `#root` inert con un contador, lleva el foco adentro y lo devuelve, y cierra con Escape. En
  Admin → Usuarios hay un solo modal a la vez (U25).
- **Carrera cerrada.** Un 401 durante el cambio de la propia contraseña espera el token nuevo en vez
  de refrescar con la cookie vieja.

Pruebas: con DB 786 → 815/1; sin DB 366 → 369/447; vitest 257 → 315. Cada arreglo de concurrencia
tiene un test que se vio en rojo por mutación. Las decisiones y sus costos están en el ledger
(Rulings U8, U9 y U15-U27).

- **Carga, medida DESPUÉS del despliegue (2026-09-15 ~06:30 CST).** Esto fue un error: la etapa salió
  sin número, contra la regla 4 de LAS CUATRO. Se detectó al escribir esta entrada (Ruling U29), y desde
  ahí la carga es parte del gate de merge.
  - **Método.** Arnés pytest en un worktree de scratch en `453b128`, contra `jax_memory_test`, con el
    sello de facet_resolver aislado por conftest. Mide la app en proceso (ASGI), no la red. SMTP
    simulado con 150 ms. Cada usuario de prueba tiene su propia IP, porque el límite mira primero la IP
    (20/60) y después el email (10/300).
  - **Mi cuenta:**
    - plana hasta c=20 (p50 301 → 318 ms, 52 rps);
    - con c=10, p95 317 ms y 0 errores;
    - tope de ~77 rps y p95 de ~635 ms con c=50/100.
  - **Abuso:** 10 respuestas 400 y después 429. Primero corta el límite por email. El 429 cuesta p50
    2,9 ms, ~2.800 rps.
  - **reset-link:** p95 156-178 ms hasta c=20.
  - **reset-password:** p95 ~159 ms con c=10.
  - **Corrección:**
    - 0 respuestas 5xx;
    - filas de auditoría = éxitos;
    - `token_version` +1 exacto por cada éxito;
    - con K=10/20/50 pedidos sobre el mismo token, en 5 rondas cada uno, gana siempre exactamente uno;
      los demás reciben 400 `reset_token_usado`.
  - **Saturación entre c=20 y c=50.** La marca el ejecutor por defecto de 32 hilos, que comparten
    bcrypt y SMTP, junto con la CPU. Ni el pool de DB (10) ni el `FOR UPDATE` fueron el cuello.
  - **EXPLAIN en producción (solo lectura):**
    - por `token`: const por el índice UNIQUE `token`;
    - pendientes por `user_id`: `ref` en el índice `user_id`;
    - reclamo por `id`: const en PRIMARY.
  - **Veredicto:** aceptable para 2-10 admins.
  - **Límite anotado, sin fecha porque no bloquea:** con un SMTP real y lento, cada envío ocupa un hilo
    del mismo ejecutor que bcrypt usa en el login. Si algún día hay decenas de admins o envíos masivos,
    el SMTP pasa a un ejecutor propio.
- **Pendiente con fecha, verificación en vivo de Fernando (2026-09-15):**
  - Mi cuenta en dos ventanas, en claro y en oscuro.
  - Los labels visibles del alta.
  - "Enviar enlace" con un correo real a un buzón suyo (U19; el plan lo condiciona a su permiso).
  Yo no pude hacerla: no tengo credenciales de admin, y crear el usuario por SQL viola la barrera
  de la DB (U23).
- **Pregunta abierta a Fernando:** ¿el admin también fija una contraseña a mano? La decisión del
  2026-09-12 fue "por enlace" (U2).

## Cerrado — admin usuarios etapa 3: invariantes, auditoría, cerrar sesiones e historial (2026-09-15)

**VERDAD OPERACIONAL 2026-09-15 04:10 CST.** jax-platform#80 → `55fed34`. El frontend servido es
`index-DOh1EUQp.js` y el backup es `…pre-usuarios-etapa3-20260915-040951`. La migración se verificó
en producción: tabla `user_admin_audit`, índices `idx_jax_users_role_status` e
`idx_user_admin_audit_target_ts`, y el EXPLAIN usa el índice.

Qué entra:
- **Invariante de superadmin sin deadlock.** Siempre queda al menos un superadmin activo. Se usa
  `transaccion(AISLAMIENTO_ADMIN)`, que es READ COMMITTED, con orden fijo de bloqueos: el conjunto de
  superadmins y después el objetivo.
- **Sin auto-acciones.**
- **"Cerrar sesiones"** sube `token_version` y corta WS 4001 y SSE después del commit.
- **Historial por usuario:** `user_audit.registrar`, con 10 acciones.
- **PUT sin contraseña.**
- **Errores:** el backend devuelve códigos estables y el frontend los traduce.

**Riesgo aceptado (U14):** no se pudo revisar `innodb_trx` en producción porque el usuario de la app
no tiene PROCESS.

**VERDAD OPERACIONAL 2026-09-14 15:00 CST.** jax-platform#70 (`3d90f58`), plan
`docs/superpowers/plans/2026-09-12-admin-usuarios-etapa-2-sesiones.md`, spec §3.2. Antes, un
token seguía valiendo hasta vencer aunque el usuario fuera degradado, desactivado o borrado: el
rol salía del token. Ahora `jax_users.token_version` (NOT NULL DEFAULT 0) viaja como `tv` en
access y refresh (sin `tv` cuenta como 0), y **cada request autenticado, `/api/auth/refresh` y el
handshake del WebSocket** leen por PRIMARY KEY estado, rol, versión y email: si el usuario no
existe, no está activo o la versión cambió → el mismo 401; el rol vale el de la base. Fail-closed:
si la base falla, se niega (y en el WS queda en el log). El frontend dice por qué se cerró la
sesión (`sesion_invalida` / `sesion_expirada`, es/en). Sin caché de entrada (spec §3.2: sin
medición no hay caché; la medición no lo pide).

- **Hallazgos de la revisión final (arreglados antes del merge):** el interceptor del frontend
  también actuaba sobre los 401 de `/auth/login` y `/auth/refresh`, así que a cualquier visitante
  sin sesión le decía "tu sesión venció", cada contraseña equivocada mostraba dos cajas rojas y
  cada carga pedía `/auth/refresh` dos veces. Los tests no lo veían porque probaban el interceptor
  aislado y el Login con el store simulado → test de integración real del interceptor con el
  store. También: un `TimeoutError` fuera del mensaje de auth del WS se cerraba 4001 sin log, y
  `/api/auth/me` hacía dos consultas por PK (ahora una). **Lección:** un interceptor global de 401
  tiene que excluir los endpoints cuyo 401 es la respuesta normal (login, refresh, logout).
- **Carga antes del merge** (el plan la dejaba solo para producción): la app real con lifespan
  mínimo (solo el pool, sin canario de facetas ni workers) contra `jax_memory_test`, base
  `bfab4de` contra la rama, `/api/auth/me`, 0 errores: p95 0,45 → 0,41 ms (c=1), 2,65 → 2,23 ms
  (c=10), 18,13 → 17,61 ms (c=50).
- **Despliegue:** respaldo `/home/fruiz/backups/jax_users-pre-admin-usuarios-2-20260914-145504.sql`
  con **restauración probada** (`jax_user` no puede crear bases → restaurado en `jax_memory_test`
  con tabla y constraint renombradas: 2 filas y MD5 de contenido idénticos a producción, tabla de
  prueba borrada). `jax-platform` reiniciado 14:58:53 (migración aplicada, los 2 usuarios en 0,
  health 200, sin tracebacks). Frontend `index-DQg7ifzD.js` en `axioma-ia.io` (respaldo
  `axioma-ia.io.backup-pre-admin-usuarios-2-20260914-145912`, idéntico por `diff -rq`).
- **EXPLAIN en producción** de `SELECT status, role, token_version, email FROM jax_users WHERE
  user_id = %s`: `type=const`, `key=PRIMARY`, `rows=1`, sin filesort ni temporary.
- **En vivo (§5):** usuario de prueba superadmin → `/api/admin/users` 200; degradado a operator →
  403 con el mismo token; desactivado → `/api/auth/me` 401; baja 200 y 0 filas después.
- **Carga en producción** (`jax/scripts/load_test.py`, `/api/auth/me`, 0 errores en todas):

  | c | peticiones | rps antes → después | p50 ms | p95 ms antes → después | p99 ms antes → después |
  |---|---|---|---|---|---|
  | 1 | 200 | 2.563 → 2.263 | 0,33 → 0,37 | 0,53 → 0,57 | 0,76 → 0,78 |
  | 10 | 500 | 3.930 → 4.135 | 2,39 → 2,18 | 2,73 → 2,73 | 8,27 → 8,41 |
  | 50 | 1.000 | 4.368 → 5.141 | 10,64 → 9,18 | 15,81 → 11,37 | 18,00 → 13,65 |

  Línea base 13:28 CST (`bfab4de`), después 14:59 CST (`3d90f58`). Sin degradación: la consulta
  por PK no se ve dentro de la varianza.
- **CERRADO 2026-09-15 por la etapa 3 (jax-platform#80 → `55fed34`) y la etapa 4 (#82 → `453b128`):**
  `_cortar_conexiones` (fail-soft; desde #82 en `auth/conexiones.py`) corta WS 4001 y SSE después
  del commit, en cada lugar que sube `token_version`: `update_user`, `revoke_sessions`, Mi cuenta y el
  reset completado. Texto original:
  **Pasa a la etapa 3 (plan enmendado en #70):** los WebSocket ya abiertos no se cortan al
  degradar o desactivar (se verifica en el handshake); el corte va donde nace el incremento de
  `token_version` (`update_user`, `revoke_sessions`, `delete_user`), después del commit. En la
  etapa 2 nadie incrementa la versión todavía. `tenant_id` sigue saliendo del token (un solo
  tenant; aceptado por el spec).

## Cerrado — admin usuarios etapa 1: correo saliente (SMTP) desde Admin (2026-09-13)

jax-platform#67 (`ddd2bc8`), plan `docs/superpowers/plans/2026-09-12-admin-usuarios-etapa-1-smtp.md`,
spec §3.1. "¿Olvidaste tu contraseña?" nunca había enviado un correo: leía `SMTP_*`
del entorno, que no existían. Ahora la configuración vive en `axioma_config`, con la
contraseña cifrada con Fernet, y se carga desde Admin → "Correo (SMTP)".

- **Hallazgo de la revisión final (Crítico, arreglado antes del merge):** "Probar
  conexión" y "guardar con la máscara" reusaban la contraseña guardada contra
  CUALQUIER host que mandara el cliente. Con cifrado `none` y un listener propio,
  un token de superadmin (aunque fuera robado) sacaba la contraseña del buzón en
  claro. El plan mismo lo exigía (un test cambiaba el host conservando la
  contraseña). Ahora la contraseña guardada solo se reusa si host, puerto, cifrado
  y usuario coinciden con lo guardado. **Lección:** "la contraseña nunca sale" no
  es solo "no se devuelve en el GET"; también es "no se entrega a un tercero".
- También arreglados en dos olas: la reserva de `smtp.*` en `/api/admin/config`
  se saltaba primero con mayúsculas y, arreglado eso, con acentos, ancho completo
  y caracteres de ancho cero. La collation `utf8mb4_uca1400_ai_ci` los iguala a
  `smtp.password`, y `ON DUPLICATE KEY` pisaba la fila real en claro. Ahora lo
  decide la base con la collation de la columna, en el PUT y en el GET. También:
  el guardado pasa a una sola transacción (antes quedaban unos milisegundos el
  host nuevo con la contraseña vieja), los 500 por contraseña o usuario no ASCII,
  por saltos de línea en el remitente, por host IDNA inválido y por `FERNET_KEY`
  malformada, el límite en test-connection y la contraseña fuera del `repr`.
  **Lección:** una lista de exclusión en Python sobre una columna `_ai_ci` no
  protege; la igualdad la define la base, así que el chequeo lo tiene que hacer
  la base.
- **Modo claro:** rojos y verdes no tenían override en ninguna pantalla. El aviso
  de configuración corrupta medía 1,49:1 y se agregaron los overrides. El override
  de `text-slate-100` (título de todas las pantallas de Admin) quedó en ≥7,2:1.
  Medido con contraste WCAG en el navegador sobre un arnés sin login.
  **Lección:** "la UI ya usa esa clase" no prueba que se lea en modo claro.
- **Carga — VERDAD OPERACIONAL, 2026-09-13 18:04 CST.** Rama en su head final
  `16b94df`, levantada en 127.0.0.1:8091 contra `jax_memory_test`. 0 errores y 0
  tokens creados:
  | endpoint | c=10 rps / p95 | c=30 rps / p95 / p99 |
  |---|---|---|
  | `GET /api/admin/smtp` | 2416 / 4,9 ms | 2196 / 15,3 / 17,2 ms |
  | `GET /api/admin/config` (reserva por la base) | 1242 / 9,4 ms | 1183 / 28,6 / 40,6 ms |
  | `POST /api/auth/forgot-password` (email real) | 3592 / 2,8 ms | 3187 / 10,6 / 13,5 ms |
  **No medido:** test-connection y /smtp/test (salen a un servidor SMTP real;
  /smtp/test está limitado a 5/300) ni el trabajo en segundo plano de
  forgot-password. Se vuelve a medir si cambia `axioma_config` o el camino de
  forgot-password.
- **EXPLAIN:** `axioma_config IN (...)` hace range sobre PRIMARY (7 filas);
  `jax_users WHERE email = ? AND status = 'active'` es const por el índice único
  `email`.
- **Despliegue — VERDAD OPERACIONAL, 2026-09-13 18:20 CST:** backend `ddd2bc8`
  (cwd del proceso `/home/fruiz/jax-platform/backend`); frontend
  `index-DE6FPDVA.js` servido en axioma-ia.io; "Probar conexión" contra
  `mail.axioma-ia.io:587` STARTTLS: VERDE (Fernando, en la pantalla, 2026-09-13 ~18:30); `smtp.password` en la
  base empieza con `gAAAAA`.
- **Correo real (con permiso de Fernando) — VERDAD OPERACIONAL, 2026-09-13:** cinco
  envíos del sistema (`no-reply@axioma-ia.io` → `fernando.ruiz@rich-hn.com`, colas
  `B44755630D1` 18:50, `47D05562F46`/`9EE5F562F46` 19:21, `8AA20560572` 19:46,
  `44BBB562F46` 19:47), todos `status=sent` a `mail.rich-hn.com`. **No hay línea
  `Authentication-Results` del receptor** (no tengo acceso a ese buzón); el cierre se
  hizo con verificación independiente, dicho así y no disfrazado de esa línea:
  - **DKIM:** rspamd registra `DKIM_SIGNED{axioma-ia.io:s=default}` en las cinco colas;
    una firma del mismo camino (rspamd, `d=axioma-ia.io`, `s=default`) verificada contra
    la llave pública en DNS con dkimpy → **PASS**.
  - **SPF:** IP de salida de atemai medida = `38.7.24.147`; el SPF de axioma-ia.io
    (`v=spf1 a mx ip4:38.7.24.147 include:sendinblue.com ~all`) la lista → pass.
  - DMARC en `p=none`: los fallos no se rechazarían; subirlo es decisión aparte.
- **Cabeceras del receptor — VERDAD OPERACIONAL, 2026-09-13 20:22 y 20:46 CST.** Dos
  correos de prueba desde la ventanita de #68 a una bandeja de Gmail, con las cabeceras
  crudas leídas con el conector de Gmail. En los dos:
  `Authentication-Results: mx.google.com; dkim=pass header.i=@axioma-ia.io
  header.s=default; spf=pass (google.com: domain of no-reply@axioma-ia.io designates
  38.7.24.147 as permitted sender); dmarc=pass (p=NONE sp=NONE dis=NONE)
  header.from=axioma-ia.io`. **Criterio de cierre del spec §3.1 cumplido**: con esto,
  la verificación independiente de arriba queda confirmada por el receptor.
  El de las 20:22 cayó en **spam** y el de las 20:46 llegó a la bandeja como importante,
  con la misma autenticación: fue reputación inicial de un dominio remitente nuevo, no
  la firma. **Lo reabre** que algún receptor de clientes lo mande a spam de forma
  sostenida. Palancas conocidas, sin tomar: subir DMARC de `p=none` y un nombre de
  remitente propio (hoy "System Administrator").
- **Pedido de Fernando tras probarla en producción — jax-platform#68 (`48adfb7`,
  desplegado 2026-09-13 19:54 CST, `index-B4G6iYq-.js`):**
  - Al elegir cifrado, el puerto salta a su valor estándar (sin cifrado 25,
    STARTTLS 587, SSL/TLS 465) y sigue editable.
  - "Enviar correo de prueba" abre una ventanita, copiada del ERP. El destinatario
    viene prellenado con `smtp.test_to` (clave nueva y opcional, editable en la
    pantalla) o, si no hay, con el email de la sesión.
  - El destinatario y el remitente exigen **una sola dirección**
    (`validacion.direccion_unica_valida`). La revisión encontró que
    `postmaster,a@b.io` llegaba a dos destinatarios y que `x;y@b.io` iba a `x`
    mientras el log registraba `x;y@b.io`.
  - Inter queda registrada como fuente intencional en
    `frontend/.impeccable/config.json`, por decisión de Fernando.
  - Carga sobre `b3c043f`: c=30, `GET /api/admin/smtp` con p95 de 18,3 ms, 0
    errores.
  - **Lección:** un ruling mío decía que el login usa `email_valido`; era falso
    (valida con Pydantic) y el docstring falso se corrigió antes del merge.
    Verificar con grep quién llama algo antes de usarlo como motivo.
- **Proceso:** ejecución con subagentes (4 tareas + revisión final con opus). La
  Task 5 del plan no traía prueba de carga: se agregó por la política 4. Se
  trabajó en un worktree porque los servicios sirven desde el checkout principal.

## Cerrado — pipeline b8f80733 y la cadena en línea (2026-09-12)

El pipeline `b8f80733` ("esquematizar el ERP") abortó a las 10:00:25. Cinco
pasos terminaron; el 04 (kimi/`generate`) venció a los 300 s. Tres defectos
encadenados, más uno de fondo:

1. **kimi se quedó sin tokens pensando**: 7232 de 8000 tokens de salida en
   razonamiento, respuesta cortada (`finish_reason=length`) como texto libre.
2. **El worker lo trató como error de schema y reintentó** — otra llamada
   destinada a cortarse igual. Entre las dos se pasó de los 300 s.
3. **El job siguió vivo tras el aborto**: terminó 10:02:59 y se cobró
   ($0.216, el paso más caro) sin que nadie leyera la salida.
   `POST /motor/job/{id}/cancel` solo reescribía una etiqueta que `worker.run`
   nunca leía.
4. De fondo, **dos presupuestos para lo mismo**: el modal mandaba
   `timeout_seconds: 300` fijo en cada step (pisando el techo de la DB, la
   fuente única decidida el 2026-09-01), y cada llamada del worker usaba 600 s
   de `motor.default_timeout_seconds` aunque al job le quedara menos.

**Arreglos — CERRADOS Y DESPLEGADOS** (`jax#134` → `1bccf56`,
`jax-platform#54` → `75a7404`):
- `motor_registry/job_tasks.py` registra la tarea de cada job; `cancel` la
  corta (el `CancelledError` de `worker.run` ya marcaba `cancelled` y
  registraba costo). Jacobs cancela el job en los dos caminos de vencimiento y
  conserva el `TimeoutError` original si el aviso falla. Un `COMPLETED` tardío
  no pisa un `CANCELLED`.
- `finish_reason=length` + schema inválido → `failed` explícito, sin
  reintento, con el error que dice qué subir (`motor.max_tokens`).
  **Ampliado el 2026-09-14 (jax#152):** ahora falla todo corte por `length`, con o
  sin schema y con tool_calls — ver la entrada de los anotados de b8f80733.
- Cada llamada recibe `min(motor.default_timeout_seconds, lo que queda)`.
- El modal ya no manda `timeout_seconds`. `generate`: techo 5 → 15 min (GO de
  Fernando), por migración con guard `WHERE =5` que no pisa un ajuste manual.

**Verificación:** 14 tests nuevos (tests-puros 75 → 86; jax-platform backend
391 → 393; vitest +1). Seis mutaciones sobre copias del árbol — quitar cada
arreglo rompe su test; control verde antes y después. CI verde en ambos PRs.
**En vivo:** servicios reiniciados sin pipelines ni jobs en curso; `generate`
= 15 en `jax_memory`; `axioma-ia.io` sirve `index-CrXyp69-.js` (HTTP 200).
Backup previo del sitio: `/www/wwwroot/axioma-ia.io.backup-pre-b8f80733-20260912-130250`.

**Lección de método — un test que sale a la red.** El primer borrador de
`test_motor_job_cancel_and_length.py` llamó a la API real de Moonshot: el
`with patch.dict(...)` devolvía la corrutina sin esperarla y se cerraba antes
de que `worker.run` leyera el transporte. Falló con 404 (URL mal armada, clave
falsa), sin costo — por suerte, no por diseño. Corregido (el `await` va dentro
del `with`) y con una guarda que hace fallar fuerte cualquier `post` real.

**Deploy del frontend:** el primer `rsync --delete` falló (código 23) al no
poder borrar `.user.ini`, que aaPanel deja inmutable (`lsattr`: `i`) en el
docroot. La cadena `&&` cortó antes de tocar producción. Se repitió con
`--exclude .user.ini` en los dos saltos. **Para el próximo deploy:** siempre
con ese `--exclude`.

**La cadena en línea — DESPLEGADA** (`jax-platform#55` → `696c6e6`; ver la
tercera E2E, abajo). Pedido de Fernando: investigar →
maquetar/planificar → criticar → unificar → producir → auditar, en línea. El
ejecutor ya encadenaba por `depends_on` (y los facets HTTP reciben la salida de
sus dependencias igual que los de motor, `executor.py:809`); faltaba armar el
plan. `pipelineChain.js`: dependencias mínimas
(`[] [0] [0,1] [1,2] [3] [0,3,4]`), facetas por rol según `capability_motor`,
aviso de auditoría independiente antes de enviar. Tomado del blueprint de
Ricardo (`repo/comparacion/`, sin commitear): la auditoría verifica solo contra
la investigación (§7.3) y mide qué hallazgos de la crítica llegaron al
producto (§12). vitest 45 → 57.

**E2E real de la cadena — dos corridas (HISTORIA, 2026-09-12).** Payload
generado por el módulo real, objetivo chico, mode `supervised` (reanudado por
API en cada pausa: `supervised` corre una ola y espera).
- `93fcd81f`: 4/6, cayó en kimi. No era la cadena: el catálogo en memoria de
  LAS MANOS era anterior a la migración de `generate` (ver ítem abierto del
  catálogo, arriba), y el motivo lo tapaba un `NameError` (jax#135).
- `1bb0da78`: **6/6 completados en 370 s, ~$0.18.** Contexto recibido por
  paso, del log de Jacobs: 0 / 3.652 / 6.284 / 7.635 / 4.600 / 8.452
  caracteres. **Pero kimi no produjo**, y la auditoría lo detectó bien: marcó
  "No se entregó un producto" y no tomó el plan como evidencia (§7.3 del
  blueprint, funcionando).

**Tres defectos del camino de motores — CERRADOS Y DESPLEGADOS** (jax#136 →
`4a013de`, `jax-las-manos` reiniciado 14:07:49; jax#135 cerrado, su commit va
dentro de #136). Anteriores a la
cadena: los 21 pasos de motor completados desde junio pasaron por el 1 y
el 3. La cadena solo los hizo visibles (primer plan donde un paso de motor
dependía de verdad del anterior):
1. **El contexto no llegaba al motor.** `_dispatch_step` armaba el prompt
   con las dependencias y `_invoke_motor` lo reconstruía desde `step.input`:
   kimi recibió 721 caracteres, "produce con el plan unificado", sin el plan.
2. **El reintento pedía JSON de un schema sin campos.** `validate()` exigía
   JSON antes de mirar `_KNOWN_UNIMPLEMENTED_SCHEMAS`; la primera respuesta
   se descartaba y el reintento pedía "el JSON del schema 'generate.v1'".
   Kimi razonó (`_reasoning_content`) que no podía inventarlo y devolvió
   `SCHEMA_NOT_PROVIDED`, marcado `completed`.
3. **La salida se recortaba a 200 caracteres** y no se guardaba completa en
   ningún lado — ni en el JSONL ni en el journal. Lo perdido no se recupera.
   Ahora va a `motor_results/<job_id>.md`. **Sin retención:** ese directorio
   crece como el JSONL; se decide junto con la rotación de logs.

Verificación: 6 tests nuevos (tests-puros 87 → 93) + 7 subtests del
validador; cuatro mutaciones, cada una rompe al menos un test. **Residuo sin
explicar:** el control final de la copia de mutación dio 1/6; no se
reprodujo en 20 corridas sobre el árbol real, y la copia se borró antes de
ver qué test fue.

**Tercera E2E, ya con los arreglos y en `autonomous` — la cadena funciona de
punta a punta (HISTORIA, 2026-09-12 14:08).** Pipeline `b2d87971`: 6/6 en
221 s, sin una sola pausa, ~$0.14. Kimi recibió **5.326 caracteres** con la
dependencia del paso 4 (antes 721), no hubo reintento de schema, su salida
completa quedó en `motor_results/5afdd6c9….md` y la auditoría recibió 10.402
caracteres. **El producto es real:** un esquema de una página que declara
seguir el plan unificado y conserva las reglas que la crítica endureció.

**Desplegada:** jax-platform#55 → `696c6e6`; `axioma-ia.io` sirve
`index-DvCsO6rG.js` (HTTP 200, `index.html` con `no-cache`). Backup previo:
`/www/wwwroot/axioma-ia.io.backup-pre-cadena-20260912-141044`.
**DECISIÓN de Fernando (2026-09-12): la cadena corre en `autonomous` por
defecto; paralelo conserva `supervised`.** El modo sigue a la forma hasta que
el usuario elige uno.

**Dos observaciones de la tercera E2E — las dos RESUELTAS Y DESPLEGADAS el
mismo día** (GO de Fernando, "dale con todo"):
- **La auditoría medía contra lo que el plan DECLARABA, no contra la crítica**
  (recibía `[0, 3, 4]`). jax-platform#56 → `a9f6fe2`: la auditoría recibe
  `[0, 2, 3, 4]`; como crítica y auditoría ya no pueden ser la misma faceta,
  **critica jekyll y audita thot** por defecto; la instrucción mide contra la
  crítica original, hallazgo por hallazgo. **Cuarta E2E** (`3b43722c`): 6/6
  en 390 s, ~$0.20, la auditoría recibió 18.373 caracteres y midió **11
  llegaron / 0 rechazados / 0 perdidos**, marcando además qué cambios no
  respalda la investigación. Bundle `index-D2w6lbti.js`, backup
  `axioma-ia.io.backup-pre-auditoria-critica-20260912-143629`.
- **Las fuentes de hipatia no eran verificables** (redirecciones opacas de
  Google, sin cita) **y además no le llegaban al paso siguiente**:
  `_build_context_input` pasaba solo `result`. jax#139 → `629157b`
  (`jacobs/grounding_sources.py`): URL final siguiendo la redirección (solo el
  header `Location`, en paralelo, 4 s de tope; nunca inventa una URL) + los
  fragmentos que respalda cada fuente. **En vivo** (pipeline `23e4c504`,
  `jax-las-manos` reiniciado 14:41:15): fuente resuelta a
  `cheatsheetseries.owasp.org/…/Password_Storage_Cheat_Sheet.html` con 2 citas
  textuales; 0 sin resolver. Latencia medida: 0,43 s por paso (6 fuentes).
  El subagente hizo 2 llamadas reales a Gemini en vez de 1 (~$0.004).

**Prueba de carga — VERDAD OPERACIONAL, 2026-09-12 13:21 CST.**
`POST /jacobs/plan` con el plan real de 6 pasos (arma y valida sin persistir;
se eligió para no llenar `jacobs_pipelines` de filas `dry_run`):

| concurrencia | peticiones | errores | rps | p50 | p95 | p99 |
|---|---|---|---|---|---|---|
| 1 | 50 | 0 | — | 1,8 ms | 2,0 ms | 8,7 ms |
| 10 | 200 | 0 | 961 | 9,6 ms | 13,0 ms | 20,4 ms |
| 30 | 300 | 0 | 1007 | 28,1 ms | 40,5 ms | 46,2 ms |

El throughput se aplana en ~1000 rps desde c=10 y lo que crece es la
latencia: el camino se procesa de a uno. Sin errores. **No medido:** el
wrapper `POST /api/pipelines` de jax-platform (pide JWT) ni la ejecución
concurrente de pipelines reales (cuestan dinero). Vuelve a medirse si cambia
la gobernanza (`get_motor_governance`) o el volumen de `capability`.

## Cerrado — kimi y la memoria vector cero (2026-09-11)

Dos defectos distintos, encontrados en cadena: el segundo apareció al
verificar el primero con un turno real, y **sin ese turno se habría declarado
cerrado `kimi` con el chat caído para todo el scope individual**. Todo lo que
sigue se verificó en la máquina (checkout servido, DB en vivo, proceso
corriendo), no en el diff.

- **`kimi` en el chat de la Mesa web — CERRADO Y DESPLEGADO.** `jax-platform#49`
  (`1b47717`). `facet.transport='motor_registry'` era **una etiqueta sin
  lector**: Jacobs despacha a kimi por `_MOTOR_FACETS` **antes** de mirar
  `facet.transport`, y el worker de Motor Registry usa `motor.transport` (que ya
  era `http_openai_compat`). El único lector era el chat, que no despacha ese
  valor → `unsupported_transport` en cada barrido desde el 2026-08-20. La API
  estaba sana (llamada real con el payload exacto: HTTP 200, `stop`, 4.7 s).

  Se alineó con `ada`: `http_openai_compat` + gate `authorize-facet` +
  `allowed_callers`, por migración idempotente guardada. **Test de clase:**
  todo facet sembrado (salvo hyde) tiene que ser despachable por el chat —
  dio rojo exactamente en `kimi-motor_registry`. `CANARY_SWEEP_TIMEOUT_SECONDS`
  900 → 1080: el 900 contaba a kimi como retorno sin red, y el margen caía de
  ~32% a ~11% sin que nadie lo decidiera.

  **En vivo:** fila migrada en `jax_memory`; sonda `ok` a las 02:37:07 y de
  nuevo tras el segundo deploy (02:56:59, las 6 facetas `ok`); la alerta pasó
  de `down` a `ok` sola a las 02:41:34. Costo nuevo declarado: ~$0.08/día de
  sonda.

- **500 en `/api/chat` por embeddings "vector cero" — CERRADO Y DESPLEGADO.**
  `jax#116` (`9904a25`) + `jax-platform#50` (`0784d5f`). Al probar kimi,
  **todo turno del scope individual caía en cualquier faceta**: medido por el
  camino real, `search_similar_messages("hola", user_id=1)` devolvía 5 filas,
  las 5 con `distancia=None`, y `_semantic_context` hacía `None < 0.8` fuera de
  su `try`.

  **Causa:** `messages.embedding`/`facts.embedding` son `VECTOR(768) NOT NULL
  DEFAULT` vector cero (el `VECTOR KEY` exige NOT NULL). `VEC_DISTANCE_COSINE`
  contra norma cero da **NaN**, y el NaN engaña a toda medición ingenua —
  medido cada punto: `IS NULL` no lo atrapa; el `ORDER BY ... ASC` lo ubica en
  cualquier lado (acá primero: por eso funcionó el 09-03 y falló el 09-11 sin
  cambio de código); aiomysql lo entrega como `None` en un camino y como
  **`0.0`** en otro. El `0.0` era un defecto dormido de `add_fact`: un fact de
  vector cero parecería un duplicado exacto de cualquier fact nuevo.

  **Arreglo en la fuente, dos capas:** exclusión en SQL
  (`_nonzero_embedding_sql`, la única que ve el NaN entregado como `0.0`) +
  filtro de distancias finitas con WARNING; y el consumidor del chat pasa a ser
  fail-soft al consumir. Tests contra MariaDB real con rojo determinístico,
  mutación por capa, job nuevo `memory-vector-zero-io` con piso de 4 corridos.

  **En vivo:** tras el deploy, la misma búsqueda devuelve 5 filas **todas
  finitas**; 0 `TypeError` en el journal desde el reinicio.

- **Backfill de las 24 filas del 2026-06-09 — HECHO, y es un arreglo MANUAL.**
  Backup por id (`~/backups/messages_vector_cero_pre_backfill_20260911-025453.sql`,
  600), **restauración probada** en una tabla temporal (24 filas, md5 idéntico
  a producción), dry-run (24/24 con embedding), `UPDATE` guardado por "sigue en
  ceros": 24/24 reparadas, **0 filas en ceros**. **La causa que las produjo
  sigue viva** — ver el ítem abierto de `save_message()` en "Anotado, no
  bloquea". No se escribe "FIJO".

## Cerrado — ronda de seguridad 2026-09-01

- **Owner de pipeline: filesystem vs DB — CERRADO. Ya estaba arreglado hace
  doce días y el ítem lo ignoraba.** Figuraba en "Bloquea trabajo" como *la
  deuda con más antigüedad*, "migración a DB identificada, sin ejecutar",
  "última verificación: ronda 6". **Medido contra el árbol el 2026-09-01: no
  hay nada que hacer.** La migración se ejecutó en la **ronda 5, T1
  (2026-08-20)**.

  Evidencia, toda verificada hoy y no heredada:

  | Afirmación del ítem | Lo que dice el árbol |
  |---|---|
  | El reaper decide por filesystem | `jacobs/reaper.py:146` decide por `p.owner_ack_at is None` — columna de `jacobs_pipelines` |
  | Migración "sin ejecutar" | El docstring del propio `reaper.py:32` dice **RESUELTO (ronda 5, 2026-08-20, T1)** |
  | jax-platform escribe un owner file | `backend/api/pipelines.py:45` hace `UPDATE jacobs_pipelines SET owner_ack_at=%s`; `:64` hace el `SELECT` |
  | Cruce de repo por `~/jax/pipelines/` | **El directorio no existe** |
  | Quedan lectores del owner file | **Ninguno.** Los `_owner.json` que sobreviven son de **comandos** (`web-task-*_owner.json`, `api/command.py`, `jax_engine/owner_cleanup.py`) — feature distinta, se dejó intacta a propósito |

  **Por qué se registra en vez de sólo borrarse.** El ítem decía "última
  verificación: ronda 6" — es decir, **alguien lo revisó DESPUÉS del arreglo y
  siguió describiendo el estado previo**. No es una entrada vieja que nadie
  miró: es una entrada mirada y no medida. Cuarta instancia del mismo patrón
  en un solo día —la suite de jax-platform con "10 o 12 failures", los
  sub-agentes que su propio texto declaraba "CERRADO 2026-08-26", Hyde
  "fuera de alcance" que ya estaba cerrada con `--ro-bind`, y ésta— y la más
  extrema de las cuatro por distancia entre el texto y el hecho. De acá sale
  la regla operativa de §7 de `CONTEXT.md` sobre medir todo ítem contra el
  árbol antes de trabajarlo.


Se conservan íntegros: describen clases de defecto reutilizables y las
retractaciones, que no se borran. Ninguno requiere acción.

- **CERRADO (2026-09-01) — `_CAPABILITY_TIMEOUT_SECONDS` eliminado: el timeout
  por capability tiene UNA sola fuente.** El default sale ahora de
  `capability.max_execution_minutes`, la misma fila que ya daba el techo.

  **La equivalencia se probó antes de tocar nada:** medido contra las **17
  capabilities reales**, el dict y la columna coincidían en **todas**, y ninguna
  clave del dict faltaba en la DB. Reemplazar uno por otra es idéntico en
  comportamiento — no es una migración con riesgo, es borrar una copia.

  **El default ES el techo, y siempre lo fue.** Un step sin `timeout_seconds`
  explícito recibe todo el tiempo que su capability declara permitido, que es
  exactamente lo que hacía el dict. Lo único que cambia es de dónde sale.

  **Lo que la deduplicación habilitó, y es lo mejor del cambio:**
  - **Un invariante nuevo:** el default **no puede exceder el techo**, porque
    salen de la misma función y la misma fila. La rama del validador que
    distinguía "lo pidió el spec" de "código y DB divergieron" se borró: la
    segunda causa es ahora imposible, y ofrecer dos causas cuando una no puede
    pasar manda a investigar un camino que no existe.
  - **`scripts/check_timeout_consistency.py` se borró**, junto con su
    invocación al arrancar `las_manos`. Existía únicamente para comparar las dos
    fuentes; con una sola no hay nada que comparar. Es la forma correcta de
    retirar un detector: eliminando la clase de defecto que vigilaba, no el
    detector.
  - **Una sola consulta de gobernanza por build**, en vez de hasta **tres** en
    el camino del LLM (el hint de capabilities, la validación del plan del LLM,
    y `_validate_plan_capabilities`). Además de ahorrar viajes, garantiza que
    las tres decisiones se tomen sobre la **misma foto**: con tres consultas, un
    cambio de `capability` en el medio podía hacer que el default y el techo de
    un mismo step salieran de filas distintas.

  **Consumidores portados, no rotos:** `tools/jacobs_relaunch.py` lee ahora la
  DB con la misma función, y tres comentarios en `models.py`, `reaper.py` y
  `executor.py` que nombraban el símbolo eliminado quedaron actualizados — uno
  de ellos decía que el techo de timeout "sigue diferido a propósito", cosa que
  dejó de ser cierta el mismo día.

  **Este ítem lo había separado su propia entrada** ("se resuelve junto con la
  deduplicación, en una ronda aparte"), y la separación era correcta: el techo
  se podía exigir sin tocar el default, y hacerlo primero dejó este cambio
  reducido a borrar una copia con la equivalencia ya probada.

- **HALLAZGO (2026-09-01) — `PROVIDER_ENV_KEYS` divergía entre espejos:
  `jax-platform` no tenía `ZHIPU_API_KEY`. Severidad: baja hoy, media
  latente.** Arreglado en `jax-platform#42`. Se registra **como ítem propio y
  no como nota** dentro del cierre del checker, porque es un defecto de
  manejo de secretos y no un detalle del comparador.

  `PROVIDER_ENV_KEYS` es la lista de variables que
  `decrypt_provider_keys_in_env()` descifra en memoria. Las dos copias de `jax`
  tenían 6 claves; la de `jax-platform`, 5. Comparadas las **tres listas
  completas**: esa era la única divergencia, el orden coincidía y no sobraba
  nada en ningún lado.

  **Por qué baja hoy (medido, no supuesto):** `ZHIPU_API_KEY` no está en
  `/etc/jax/.env` —la de Z.ai es `ZAI_API_KEY`, presente en las dos listas—, no
  aparece en ningún otro archivo, y no estaba seteada en ninguno de los dos
  procesos vivos. El bucle hace `if raw:`, así que una clave ausente se saltea
  sin crear ni tocar nada.

  **Por qué media latente:** el día que alguien agregue `ZHIPU_API_KEY` al
  `.env` —un rename, un proveedor nuevo—, `jax-platform` **no la descifraría** y
  Mesa web mandaría el **ciphertext Fernet como API key** de Ada. El síntoma
  sería **"Ada anda en Jacobs y no en Mesa web"**: un fallo de proveedor
  externo, no de configuración local, y nadie miraría la lista de descifrado.

  **Precedente de la misma forma:** el mapa proveedor → variable de entorno que
  `credential_resolver` tuvo durante la ventana B1.4 (retirado el 2026-09-17 al
  cerrar E-25), señalado en su día como el símbolo cuyo drift produciría
  exactamente ese síntoma. Es
  la segunda instancia de la clase: **un mapa de secretos replicado, donde la
  copia incompleta no falla, sólo deja de hacer algo.**

- **CERRADO (2026-09-01) — `crypto_secrets.py`, cuarta familia del comparador
  de espejos.** `jax#PENDIENTE` + `jax-platform#42`.

  **Se midió antes de agregarla, y por eso no nació rota.** El primer intento
  de sumarla tal cual habría puesto **los 4 símbolos compartidos en rojo el día
  uno**, con **3 de las 4 diferencias siendo cosméticas** (estilo de anotación,
  redacción de docstrings, nombres de variable local). Eso es el estado del que
  esta ronda sacó a `facet_resolver`: un detector que grita por lo esperado
  entrena a ignorarlo. Y marcar tres diferencias de redacción como
  `DIVERGENCIA DELIBERADA` habría sido usar el marcador para mentir — no son
  decisiones de diseño, son copias que se despeinaron.

  **Forma real, verificada y no asumida:** tres archivos **reales**, ninguno
  symlink, ninguno generado por script. Dos dentro de `jax` (byte a byte
  idénticos) y uno en `jax-platform`. Misma forma que `credential_resolver` —y
  distinta de `facet_resolver`, donde `las_manos/` sí es symlink—.

  **Exclusiones declaradas**, mismo criterio que `load_facet_registry`:
  `encrypt_secret` y `decrypt_db_secret` existen **sólo** en `jax-platform` y no
  son drift. La razón ya estaba escrita en el docstring de la copia de `jax`:
  jax-platform es el lado que **cifra** (sync bidireccional BD→.env) y además
  lee `user_api_keys`, tabla suya; los procesos de JAX sólo **descifran**.

  **Alineación con cero cambio de comportamiento**, canónico elegido **por
  elemento y no por copia** (las tres cambiaron): la anotación directa y el
  docstring más informativo de jax-platform, el `-> None` explícito de jax, los
  nombres `env_key`/`raw`, y un docstring **nuevo** para
  `decrypt_provider_keys_in_env` —el de jax-platform nombraba `main.py`, falso
  en las copias de jax—. Verificado por esqueleto AST contra `HEAD` y por prueba
  directa de que `.get(k)` y `.get(k, "")` son equivalentes bajo `if raw:`.
  Suites idénticas antes y después: 336/186 en jax-platform, 55 en jax.

  **Probado rompiéndolo, en las tres copias:** quitar una key en jax-platform
  (el drift original) → rojo; cambiar `decrypt_secret` en `las_manos` —el tercer
  archivo, el que nadie comparaba— → rojo; cambiar `_get_fernet` en el canónico
  → rojo en los dos espejos. Restaurado → verde.

  **Fue configuración, no código:** agregar la familia es una entrada en
  `FAMILIAS`. La generalización de la ronda anterior aguantó.

- **RIESGO ACEPTADO (2026-09-01) — Hyde con red sin acotar. No es un
  pendiente: es una decisión tomada con la medición en la mano.** Decisión de
  Fernando tras el diagnóstico completo. **No se implementa el confinamiento de
  red**, y las razones están abajo para que nadie lo reabra por intuición.

  **Qué alcanza Hyde hoy** (medido desde adentro del sandbox, no inferido):

  | Destino | Resultado |
  |---|---|
  | `127.0.0.1:3308` (MariaDB) | **ALCANZA** |
  | `127.0.0.1:11434` (ollama) | **ALCANZA** |
  | `127.0.0.1:8080` (jax-platform) | **ALCANZA** |
  | `127.0.0.1:7777` (las_manos) | **ALCANZA** |
  | `1.1.1.1:443` (internet abierto) | **ALCANZA** |

  **El diseño por UID está MUERTO, con medición.** `bwrap --uid` **no cambia el
  UID que ve el kernel**: dentro del sandbox `id -u`=12345 y el socket, visto
  desde el host contra un listener local, sigue siendo `uid:1000`.
  `iptables -m owner --uid-owner` matchea ese valor, así que **no puede
  distinguir a Hyde de Fernando** mientras lo lance un proceso de éste. Llegar
  a un UID real distinto exigiría `sudoers` (rechazado) y leer los repos
  exigiría aflojar `/home/fruiz` (rechazado).

  **El match por cgroup SÍ funciona** — verificado, no supuesto. Kernel
  `7.0.0-30-generic`, `nftables v1.1.6`, `cgroup2fs`. Un proceso dentro del
  cgroup quedó **BLOQUEADO** y uno normal de fruiz **CONECTÓ**, con la misma
  regla activa. `iptables -m cgroup --path` existe (módulo `xt_cgroup`), y
  `jax-las-manos` puede administrar su propio subárbol con `Delegate=yes`
  (medido con una unidad desechable: systemd hace `chown` del cgroup al User).

  **La regla exacta, lista para usar si algún día se retoma** —
  `/etc/ufw/after.rules`, cadena `ufw-after-output`:

  ```
  -A ufw-after-output -m cgroup --path system.slice/jax-las-manos.service/hyde.scope \
     ! -d 160.79.104.10 -j DROP
  ```

  Persiste sola (ufw `enabled` la aplica al bootear), Docker no la reescribe
  (vive en las cadenas de ufw), y no toca las 6 reglas de entrada (están en
  `user.rules`). DNS no necesita entrada: sale por `127.0.0.53`, loopback.

  **POR QUÉ NO SE IMPLEMENTA — las dos razones:**

  1. **El loopback queda abierto, y eso deja la DB de producción alcanzable.**
     `ufw-before-output` acepta `oifname lo` **antes** de cualquier cadena
     custom, así que una regla en `ufw-after-output` nunca ve ese tráfico.
     Cerrarlo exigiría meter la regla en `before.rules` **por delante de ese
     accept**, que es load-bearing para todo el sistema.
  2. **La API de Anthropic es un canal de datos, y el control de red no lo
     toca.** Hyde arma los prompts: todo lo que puede leer —los dos repos
     completos— puede irse por el destino permitido. El control cerraría
     "destino arbitrario", no "contenido arbitrario al destino permitido", que
     es lo que en realidad se teme.

  **CONDICIÓN DE REVISIÓN — se retoma si pasa cualquiera de estas tres:**
  - **Hyde cambia de transporte** (hoy `facet.transport='subprocess'`, y
    ningún endpoint lo escribe: un rebind cambia el modelo, no el transporte).
  - **Hyde gana acceso a otras máquinas** (hoy sólo alcanza este host y la red
    pública desde él).
  - **Hyde deja de ser el único proceso en ese sandbox.**

  **COSTO DE LAS PRUEBAS, DECLARADO.** Diagnosticar esto consumió **dos
  llamadas reales a la API de Anthropic con las credenciales de Fernando** —
  una tarea de prueba ejecutada de punta a punta, con el destino de telemetría
  bloqueado y sin bloquear, para medir si `claude` funciona sin él. Autorizado
  y declarado en el momento. Se anota porque **el costo de verificar tiene que
  ser visible**: "¿funciona sin este destino?" no se responde leyendo código, y
  una ronda futura que repita el experimento va a pagar lo mismo.

  **Lo ya medido, para no volver a pagarlo:** el destino de IP rotante es
  **`us5.datadoghq.com`** (telemetría), identificado por el SNI del handshake
  real, y **`claude` funciona sin él** (tarea completada, RC=0, 10.8 s con
  bloqueo contra 9.7 s sin). El allowlist necesario sería **una sola IP**:
  `160.79.104.10:443`.

  **Lección de método** (vigesimoséptima de la familia, CONTEXT.md §9): un
  control puede estar correctamente diseñado, ser implementable, y aun así no
  valer la pena. Medirlo **antes** de construirlo es lo que evitó gastar la
  ronda.

- **CERRADO (2026-09-01) — el techo de ejecución declarado en la DB es un
  techo de verdad.** `_validate_plan_capabilities` lo hace cumplir.

  **Qué era.** `capability.max_execution_minutes` decía ser el límite de
  ejecución de una capability y nada lo hacía cumplir en plan-time: un
  `timeout_seconds` explícito en el spec de un step pisaba el default y llegaba
  intacto a `asyncio.wait_for`. El techo declarado no era un techo — era una
  columna que alguien podía editar en un panel admin creyendo que cambiaba algo.

  **Rechaza, no recorta.** Recortar en silencio convierte un error de
  configuración en comportamiento sordo: el plan corre con un timeout que nadie
  pidió y nadie ve. El mensaje nombra el valor pedido, el techo, la capability
  y **de dónde vino el valor** — distingue "lo pidió el spec" de "código y DB
  divergieron", que necesitan arreglos distintos.

  **Dos decisiones tomadas con datos, no por gusto:**

  1. **Aplica a TODOS los steps, no sólo a los de `MOTOR_FACETS`** —que es lo
     único que el validador miraba—. El techo es propiedad de la *capability*,
     no del motor, y `executor.py` envuelve **cada** step en
     `asyncio.wait_for(..., timeout=step.timeout_seconds)`, sea motor o facet
     HTTP. Un techo que no cubre a la mitad de los steps no es un techo. Se
     eliminó el `if not relevant: return` que dejaba pasar de largo cualquier
     plan sin steps de motor.
  2. **Capability sin techo declarado → 300 s, no "cualquier cosa".** Es el
     único caso ambiguo y se decidió midiendo: `assemble` está exenta **a
     propósito** en el planner y no tiene fila en `capability`. Aceptar
     cualquier timeout ahí sería fail-open; rechazar de plano rompería
     `assemble`. Recibe `_DEFAULT_TIMEOUT_SECONDS`, el mismo valor que el
     código ya le asigna. Medido en `jacobs_steps`: **6 steps `assemble`
     reales, todos con `timeout_seconds=300`** — la regla no rechaza ninguno.

  **Consecuencia declarada:** el validador ahora consulta la gobernanza
  **siempre**, no sólo cuando hay steps de motor. Si la DB no responde, un plan
  que antes se construía ahora falla — el techo no se puede verificar y se falla
  cerrado. Costo medido de la consulta: 3 SELECTs, 0.00024 s en el servidor.

  14 tests (job `plan-timeout-ceiling`), auto-verificados por mutación: quitando
  el chequeo caen 9. Medido además que **no rompe nada existente**: 24 passed en
  master y 24 con el cambio, sobre los 6 archivos de tests que tocan el
  validador.

- **CERRADO (2026-09-01) — el comparador de espejos cubre ahora
  `credential_resolver`, y se generalizó a FAMILIAS.**
  `scripts/check_facet_resolver_sync.py` → `scripts/check_mirror_sync.py`
  (renombrado con `git mv`, historia preservada). Job `facet-resolver-sync` →
  `mirror-sync`.

  **Cierra la CLASE, no la instancia.** El patrón declarado "sin paquete
  compartido" replica módulos a propósito, y su costo conocido es que un
  arreglo hecho en una copia y no en la otra queda invisible. Ese costo se
  cobró **tres veces en 2026**, siempre con la misma forma: `facet_resolver._db_conn`,
  los cuatro sitios del default a la instancia muerta `3306`, y
  `credential_resolver._db_conn`. Se generalizó en vez de copiarse: **un
  segundo comparador sería un espejo más, con el mismo defecto que viene a
  detectar.** Las familias se declaran como datos; agregar una es agregar una
  entrada.

  **LA MEDICIÓN CAMBIÓ EL DISEÑO.** El ítem asumía la forma de
  `facet_resolver` (dos archivos, porque `las_manos/` es un symlink). Medido
  antes de escribir nada: **`las_manos/credential_resolver.py` NO es un
  symlink, es un tercer archivo real.** O sea que esa familia puede driftear
  **dentro del propio repo `jax`**, sin cruzar repos — una copia más suelta que
  la de `facet_resolver`, y hasta hoy nadie la comparaba con nada. El
  comparador recorre todos los espejos, no el primero, y el reporte dice **cuál**
  diverge.

  Estado medido al cerrar: los tres archivos con el mismo conjunto de 10
  símbolos de nivel superior, todos idénticos. La única diferencia real es el
  import de `crypto_secrets`, que es un `ImportFrom` y no entra a la
  comparación — no necesita marcador.

  **PROBADO ROMPIÉNDOLO, Y LA PRUEBA ENCONTRÓ UN DEFECTO REAL.** El mecanismo
  de "divergencia deliberada" **no funcionaba para constantes de módulo**: el
  marcador se declara en el docstring y una constante no tiene, así que
  `ast.get_source_segment` devolvía la sentencia pelada sin los comentarios de
  alrededor. `FACET_SEAL_PATH`, agregado a la comparación ese mismo día, no se
  podía declarar de ninguna forma. Arreglado (`_bloque_declarativo`) y fijado
  con tests que se verificaron **revirtiendo el arreglo**: 3 en rojo.

  Casos verificados, los dos sentidos: divergencia sin declarar en cada una de
  las dos familias y en cada uno de los tres archivos → **rojo**; declarada con
  marcador (en el comentario de arriba y en el de la misma línea) → **verde**;
  un comentario sin marcador → **sigue rojo** (el arreglo no aflojó el
  detector); un espejo ilegible → **exit 2**, nunca verde en silencio.

  **La cuarta familia (`crypto_secrets.py`) ya está incorporada** — ver su
  entrada propia más arriba. Se midió antes de agregarla, y esa medición
  encontró drift real.

- **CERRADO (2026-09-01) — el default silencioso a `localhost:3306` sobrevivía
  en CUATRO sitios de `jax-platform`, incluido el pool principal.**
  PR `jax-platform#41`. **Nunca estuvo anotado acá**: se encontró barriendo el
  repo, no leyendo esta lista.

  **CORRIGE EL REGISTRO.** La entrada "14 (en verdad 18) tests de `las_manos`"
  de más abajo dice que el fallback silencioso "vivía duplicado en 19 archivos"
  y que "los 19 pasaron de default silencioso a `RuntimeError`". Los 19 eran
  **todos del repo `jax`**. jax-platform es un repo separado y quedó afuera del
  barrido. Q1 de esta misma ronda lo notó para **una** copia
  (`facet_resolver._db_conn`, portada a mano desde `jax/core`) pero no siguió.
  Quedaban cuatro:

  | Sitio | Qué es |
  |---|---|
  | `db/connection.py::get_pool()` | **el pool principal del backend — 22 módulos** |
  | `credential_resolver.py::_db_conn` | drift real: la copia de `jax` sí tenía el guard |
  | `api/chat.py::_ensure_memory` | conecta la memoria del chat |
  | `api/admin/dashboard.py` | no conecta: **muestra** el puerto en el panel de salud |

  Es la **tercera vez** que el mismo defecto se cierra en un repo y sobrevive
  en el otro. `localhost:3306` no es un default razonable: esa instancia **no
  existe** (la real es `:3308`), así que el default convertía "falta
  configuración" en "conecta a una instancia muerta".

  **Dos detalles que no son mecánicos:**
  - En `api/chat.py` el guard va **afuera del `try`**. Ese `except` deja
    `_memory_ready` en `False`, así que un `raise` adentro se traduciría en
    "memoria silenciosamente desactivada" — el mismo fallo mudo que el guard
    viene a eliminar, con otra cara. Configuración ausente = ruidoso; DB caída
    = sigue siendo fail-soft.
  - El tablero no conecta, **muestra**. Un `3306` inventado ahí no rompe nada:
    miente, y en un panel de estado eso es peor.

  **No se cerraron cuatro sitios: se cerró la regla.** Además de un test de
  comportamiento por sitio, hay un scanner AST sobre todo el árbol de
  producción — un sitio nuevo escrito mañana queda cubierto sin que nadie lo
  agregue a ninguna lista. Cerrar los cuatro sin eso sólo garantizaba que
  hubiera un quinto. Probado contra 5 mutaciones y 3 casos legítimos que no
  debe marcar, y sobre el código real: reintroduciendo el default en
  `get_pool`, 5 tests en rojo con el `ConnectionRefusedError` a
  `127.0.0.1:3306` en el log.

  `credential_resolver._db_conn` quedó **byte a byte idéntico** al de
  `jax/core` (verificado comparando el AST). El checker de drift que le faltaba
  **ya existe** — ver la entrada de `check_mirror_sync.py` más arriba.

- **CERRADO (2026-09-01) — el contenimiento del sandbox de Hyde ahora está
  EJERCITADO, no descrito.** `_hyde_containment_test.py` (14 tests) + job
  `hyde-containment` en CI.

  **El hueco no era el sandbox: era que nadie lo probaba.** `hyde_sandbox.py`
  afirmaba propiedades de seguridad en su docstring y acá —repos en
  solo-lectura, `$HOME` real no expuesto, entorno del padre no heredado— y esas
  propiedades se habían verificado **una sola vez a mano** al escribirlo
  (2026-08-23). El único test que existía, `_hyde_sandbox_test.py`, cubre el
  `flock` y los timeouts: la **serialización**, no el **confinamiento**.
  Cambiar un `--ro-bind` por un `--bind` no rompía ningún test — el sandbox
  seguía arrancando, Hyde seguía funcionando, y el confinamiento se perdía en
  silencio. Una propiedad de seguridad verificada una vez y nunca más es una
  propiedad supuesta.

  **Los tests ejecutan ataques reales** dentro del sandbox y, cuando el ataque
  es una escritura, afirman **sobre el host**: un `touch` puede "fallar" y aun
  así haber dejado el archivo, y ese caso es peor que el que se estaba
  probando. Se auto-verifican por mutación: `--ro-bind` → `--bind` pone 3 en
  rojo, quitar `--clearenv` pone 2.

  **El control positivo no es decorativo.** `test_el_workspace_si_es_escribible`
  existe porque sin él todos los tests de bloqueo pasarían igual si bwrap no
  arrancara — verde por la razón equivocada.

  **No usan las rutas de hall9000:** monkeypatchean las constantes del módulo a
  directorios temporales. Un test que sólo corre en la máquina de Fernando no
  corre en CI — y ese mismo día esa confusión costó tres tandas de arreglos a
  ciegas en la suite de jax-platform.

- **CERRADO (2026-09-01, sólo documentación) — sub-agentes de Claude Code sin
  gobernanza.** El ítem estaba en "Bloquea trabajo" pero **su propio texto ya
  decía que estaba cerrado**: `hyde_sandbox.py::run_sandboxed_claude()` como
  único punto de entrada, `flock(2)` cross-proceso, y el job
  `no-naked-claude-subprocess` fallando el build si aparece un call site de
  `claude` fuera de ahí (PRs jax#33-36, 2026-08-26). No había trabajo
  pendiente: había una entrada que nadie movió. Segundo caso el mismo día —ver
  la suite de jax-platform— de deuda que sigue listada como abierta porque el
  cierre se escribió adentro del ítem en vez de moverlo.

- **CERRADO (2026-09-01) — una BackgroundTask que lanza ya no se lleva puestas
  a las encoladas después.** PR `jax-platform#40`.

  **La premisa se re-verificó antes de tocar nada** —el ítem era del 08-27, y
  ese mismo día ya había aparecido otro ítem de esta lista cuyo diagnóstico
  estaba vencido—. Contra `fastapi 0.139.2 / starlette 1.3.1`: confirmada. La
  segunda tarea no corre y la excepción propaga.

  **El arreglo:** `jax_engine/background.py::add_safe_task()` envuelve cada
  tarea en su propio `try/except`. Fail-soft a propósito y **ruidoso** a
  propósito —la excepción se detiene ahí porque propagarla cancela las
  siguientes, y se registra con traceback porque tragarla en silencio cambiaría
  un modo de falla silencioso por otro—. El docstring dice además lo que **no**
  hace, para que nadie lo suponga: no reintenta, no persiste el fallo y no
  alerta. Una tarea cuyo resultado alguien necesite confirmar no es una
  BackgroundTask.

  **La regla quedó en "ninguna", no en "no más de una".** El ítem proponía "un
  test de política que falle si un endpoint encola más de una sin envoltorio".
  Se implementó más fuerte: **ningún** archivo de producción puede llamar
  `.add_task()` crudo. "Una sola tarea" es una propiedad de los llamadores de
  hoy, no una garantía del mecanismo, y un umbral de dos deja al primero
  desprotegido sin motivo.

  **El scanner está probado contra violaciones reales, no sólo escrito:** 5
  mutaciones parametrizadas (llamada directa, dentro de un `if`, en función
  anidada, dentro de un `try`, con otro nombre de variable), el caso negativo
  (`add_safe_task` no cuenta), y verificación sobre el código real
  —reintroduciendo a mano el `add_task` crudo, el test se pone en rojo—.
  Exige además haber escaneado >20 archivos: el scanner P10 de `jax` estuvo
  meses en verde sobre **cero**, y ese error no se repite por accidente.

  **Sin debilitar lo que ya se afirmaba:** los dos tests que decían "este
  endpoint encola `probe_after_rebind`" lo hacían por identidad, y se
  conservan así (el envoltorio expone `tarea_original`/`tarea_args`) en vez de
  relajarlos a "encola algo".

- **CERRADO (2026-09-01) — `jax-platform`: la suite del backend está en verde
  y CI la corre entera.** De **158 tests en CI a 308**. PR `jax-platform#39`.

  **La premisa de este ítem estaba vencida, y esa es la lección.** Decía que la
  suite era *inestable* ("10 o 12 failures sobre el mismo árbol limpio") y de
  ahí concluía que "cualquier criterio basado en el número es inservible" y que
  versionar un baseline quedaba "descartado por construcción". Medido contra
  `461a089`: **5 failures + 1 error, los mismos por nombre en 3 corridas
  seguidas.** Determinístico. La deuda se había descrito una vez y nadie volvió
  a medirla; el diagnóstico envejeció y la conclusión que colgaba de él —"esto
  no se puede meter en CI"— era falsa desde hacía tiempo.

  **Causa raíz, una sola para las 6:** `db/connection.py` guardaba **un pool
  global para todos los event loops**. Un pool de aiomysql queda atado al loop
  que lo creó (sus futuros internos, `Pool._wakeup`, viven ahí) y en la suite
  hay dos: el del portal de `TestClient` y el de pytest-asyncio. Ahora se
  indexa por loop. En producción no cambia nada —uvicorn corre un solo loop—.

  **Tres acoplamientos al `$HOME` que hacían imposible correr la suite fuera de
  hall9000**, encontrados sólo al reproducir el runner de verdad:
  `api/chat.py::CONFIG_PATH` y `shadow_validation.py::JAX_REPO` apuntaban a
  `~/jax` hardcodeado (30 y 9 tests), y `_query_facet` exige una credencial de
  proveedor que en un runner no existe (12 tests, con un mensaje que además
  miente: dice "sin binding activo" cuando el binding está y lo que falta es la
  credencial). Los dos primeros son ahora `JAX_CONFIG_PATH` y `JAX_REPO_PATH`,
  con el mismo default.

  **Bug de producto que la DB de desarrollo escondía:** `provider.base_url` de
  ollama quedaba **sin `/v1` en toda instalación nueva**. El seed insertaba sin
  `/v1` y la migración correctiva sólo actuaba `WHERE base_url IS NULL OR
  base_url = ''`, condición que ese mismo INSERT vuelve falsa. En `jax_memory`
  está bien por accidente histórico. Arreglado en las dos puntas. **No era
  detectable sin una base limpia en CI** — es el argumento entero de este
  trabajo, en un caso concreto.

  **`jacobs_pipelines` la crea el repo `jax`** (`jacobs/store.py::init_tables()`)
  y jax-platform sólo la lee. El job clona `jax` y ejecuta **su** `init_tables()`
  en vez de copiar el DDL: una copia sería una segunda fuente de verdad que se
  desincroniza sola. Sin esa tabla caen los 14 tests de propiedad de pipeline,
  que son los que cubren el IDOR cerrado en agosto.

  **Método, la parte que más costó:** las tres primeras tandas de arreglos se
  hicieron **a ciegas**, porque la corrida "local" usaba el `$HOME` de hall9000
  —con `~/jax` y con `/etc/jax/.env`, que es legible por el grupo `fruiz`— y por
  lo tanto **no reproducía el runner en absoluto**. Recién al correr todo en un
  contenedor sin ninguna de las dos cosas, contra un `mariadb:11.8` virgen, los
  números locales y los de CI coincidieron. Verde local sobre un entorno
  privilegiado no dice nada sobre un runner limpio.

  **El gate exige dos cosas, no una:** piso exacto de 308 passed **y máximo 1
  skip**. Sin lo segundo, una DB que no arranca convertiría los 150 tests en
  skips y el job seguiría verde sin haber probado nada.

- **CERRADO (2026-09-01) — `facet_resolver._cache` replicado en TRES
  procesos; ahora la invalidación cruza los tres.** Era el único ítem abierto
  de la ronda (Q3). Decisión de Fernando entre las dos opciones diseñadas:
  **sello en filesystem**.

  **Qué era.** `facet_resolver.py` está espejado a propósito en
  `jax-platform/backend`, `jax/core` y `jax/las_manos` —dos archivos reales,
  porque `las_manos/facet_resolver.py` es symlink de `jax/core`, y tres
  procesos—. Cada uno tenía **su propio `_cache`** con TTL de 30 s, y solo
  `jax-platform` lo invalidaba al rebindear. Jacobs y el REPL despachaban
  contra el binding **viejo** hasta 30 s después de aprobar un modelo nuevo, y
  la sonda por rebinding no lo podía ver: sondea el único camino invalidado.
  Punto ciego del detector, no solo un problema de frescura.

  **Cómo se cerró.** El escritor toca un archivo; cada `resolve_facet` compara
  su `mtime` contra el instante en que cacheó y, si el sello es más nuevo,
  **descarta** la entrada —también del camino de `serving_stale`, porque servir
  un valor que un escritor declaró superado es el binding viejo que esto vino a
  matar—. ~1 µs por `os.stat`, sin red y sin dependencia nueva: los tres
  procesos corren en hall9000, mismo filesystem.

  **El detalle que lo arruinaba en silencio, y por qué queda clavado.**
  `_CacheEntry.fetched_at` es `time.monotonic()`, que no es comparable con un
  `mtime`. Compararlos no da error: da un veredicto **constante** —"invalidar
  siempre" o "no invalidar nunca", según el origen de monotonic— y el bug queda
  igual, con código nuevo que aparenta resolverlo. La entrada guarda ahora los
  dos relojes y la comparación usa el de pared. Hay un test por cada dirección,
  con el origen de `monotonic` forzado, y se verificó por **mutación** que los
  dos se ponen en rojo si alguien vuelve a comparar contra monotonic.

  **Modo de falla, declarado en el código** (docstring de `_seal_mtime`): si el
  sello falta o es ilegible se vuelve al TTL de 30 s, el techo que ya existía.
  No es un fail-open nuevo —el sello solo puede **adelantar** una invalidación
  que el TTL iba a hacer igual—. Invalidar ante un sello ilegible convertiría un
  archivo faltante en "sin caché", que es una regresión de rendimiento
  silenciosa y un modo de falla nuevo introducido por el propio arreglo.

  **Verificado sobre los TRES procesos, no solo sobre Mesa web** —probar solo
  ahí reproduciría el mismo punto ciego que tenía la sonda—: rebind real, y
  jax-platform, jax-las-manos y el REPL sirviendo el modelo nuevo **sin
  reiniciar**, medidos por separado.

  **CI.** `facet-resolver-seal` (jax) corre los 11 tests por el camino de import
  de Jacobs y exige que corran, no que se salten; `backend-tests-no-db`
  (jax-platform) corre los otros 11 dentro de la suite, con el piso subido de
  143 a 158. `facet-resolver-sync` gatea que los dos espejos no diverjan, y se
  extendió para cubrir el sello: ahora compara también las constantes de módulo,
  porque dos espejos apuntando a **sellos distintos** dejarían todo lo demás
  idéntico y la invalidación no cruzaría —drift invisible dentro del mecanismo
  construido para cerrar un punto ciego—.

  PRs: `jax-platform#38`, `jax#88`.

- **CIERRE FINAL (2026-09-01). Un solo ítem vivo.**

  ### CERRADO — no hay terceros, no hay notificacion

  **Los 6 dominios del inventario son cuentas PROPIAS de Fernando en su propio
  servidor** (decisión y aclaración de Fernando, 2026-09-01). **No son clientes
  externos, no hay terceros involucrados y no hay obligación de notificar.**

  La clasificación como "información de terceros" fue **una inferencia
  razonable desde la base de datos** —aparecían como cuentas de panel
  distintas, con dominios distintos— **pero incorrecta**. Es el mismo modo de
  fallo que esta ronda ya registró: derivar el alcance de lo que el sistema
  muestra, sin confirmarlo con quien sabe.

  **Riesgo aceptado, misma categoría que las 3 listas de IPs propias.** Lo que
  quedó en la historia pública es información de infraestructura **propia**:
  18 blobs, 8 rutas, 16 alcanzables con un `git clone` normal. El inventario
  técnico se conserva en `$AUDIT/INVENTARIO-CLIENTES.md`; **el borrador de
  aviso que contiene no se usa.**

  **Único dato de tercero real de toda la ronda:** el correo de la persona
  física, **sacado de HEAD en el PR #80** e incluido en el ticket de purga.
  Sigue cubierto.

  ### ESPERANDO — sin trabajo de este lado

  - **GitHub Support:** ticket listo para pegar en `$AUDIT/TICKET-GITHUB.md`.
  - **Purga de R2:** 2026-09-08 ~01:00, bloqueada por el Bucket Lock.

  ### CERRADO esta noche — no son problemas

  - **La segunda cuenta NO es vulnerable.** `bcrypt` válido (`$2b$12$`, largo
    60) y `bcrypt.checkpw` de la aguja filtrada da **FALSO**, medido. El hash
    no cambió con la rotación de plataforma, pero eso es **higiene, no
    exposición**. **CERRADO.**
  - **Los dos dominios de cliente NO tienen vulnerabilidad explotable.**
    Certificados **válidos** hoy, **HTTPS 200**, vía Cloudflare. El
    diagnóstico del repo era de agosto y describía el certificado del
    **origen**, detrás del proxy, que los visitantes nunca vieron.
    **CERRADO.**

  ### Fuera de este cierre — higiene, sin fecha

  Modo SSL de Cloudflare y certificado del origen: **infraestructura propia,
  sin relación con el incidente**. Ítem separado, sin fecha. El diagnóstico
  medido queda registrado más abajo por si sirve cuando se retome.

- **MI ERROR: se trató un diagnóstico viejo como estado actual (2026-09-01).**
  Se le asignó urgencia a *"vulnerabilidad activa en dos dominios de
  clientes"* y **esa urgencia era falsa**. El diagnóstico del repo es de
  agosto y describía el certificado del **ORIGEN**; el borde sirve
  certificados válidos vía Cloudflare y **los visitantes nunca vieron el
  vencido**. Se dio por vigente sin medir el estado actual, y estuvo a punto
  de irse así en un aviso a terceros.

  Ver la vigésimo sexta lección en `CONTEXT.md` §9.

- **La plataforma NO escribió nada al "rotar" la segunda cuenta — NO hay un
  segundo almacén de credenciales.** Medido: única columna de credenciales en
  toda la instancia es `jax_users.password_hash` (en los dos esquemas); una
  sola fila para esa cuenta; `jax_users` sin escrituras posteriores al `ALTER`
  de las 05:34:16; `password_reset_tokens` **vacía y con `UPDATE_TIME` NULL**
  (tampoco fue un flujo de reset); el backend registró **5 líneas** en la
  ventana, todas de `credential_resolver`; y no corre ningún LDAP ni Keycloak.

  **El gate no dispara: no apareció un segundo almacén.** Lo que no se puede
  descartar desde acá es un servicio externo, pero nada en la máquina lo
  sugiere.

  **Procedimiento de rotación listo para ejecutar** en
  `/home/fruiz/security-audit-2026-09/ROTACION-user_id-2.md` (chmod 600), con
  el testigo previo, el `UPDATE` directo, y `updated_at` como verificación
  nueva. **No ejecutado.**

- **La rotación de la segunda cuenta NO ocurrió — NO CERRADO (2026-09-01).**
  Se informó que se había rotado desde la plataforma. **Medido: el
  `password_hash` de `user_id=2` es idéntico al que se registró en el
  inventario pre-rotación** (misma huella MD5), y `updated_at` está en `NULL`
  pese a que la columna existe desde hoy ~05:34 — cualquier `UPDATE` posterior
  la habría estampado.

  El hash **sí** es un bcrypt válido (`$2b$12$`, largo 60), así que **no** es
  el modo de fallo del marcador de 48 caracteres. Y `bcrypt.checkpw` de la
  aguja filtrada da **FALSO**, así que la cuenta no es vulnerable por esa vía.
  **Pero la fila no se tocó.** O la plataforma escribió en otro lado, o falló
  en silencio, o se rotó otra cuenta.

  Es **la misma clase que `user_id=1`**: creer que una rotación ocurrió cuando
  la fila no cambió. Ahí el testigo lo detectó; acá también.

- **Certificados de dos dominios de cliente — la vulnerabilidad documentada NO
  está vigente (medido 2026-09-01).** El repo publicó el diagnóstico de dos
  dominios con certificado vencido hace más de un año. **Medición de hoy:
  ambos sirven certificados válidos** (Google Trust Services, vencen
  2026-11-24 y 2026-11-19), responden **HTTP 200**, y tienen **Cloudflare
  delante**.

  **El certificado vencido era el del ORIGEN, detrás del proxy.** Los
  visitantes nunca lo vieron. **La incidencia no es explotable desde internet
  hoy.** Lo que queda por revisar —certificado del origen y modo SSL de
  Cloudflare, `Full` contra `Full (strict)`— **está dentro de la
  infraestructura propia**, no del lado del cliente. **Requiere autorización
  para tocar; no se cambió nada.**

  El borrador de aviso quedó **reescrito** con este estado: la versión
  anterior daba por vigente la vulnerabilidad y era incorrecta. Un aviso que
  reporta una exposición sin decir si lo expuesto sigue vigente deja al
  cliente sin saber qué hacer.

- **Dump nocturno sin hashes — CERRADO 2026-09-01, verificado corriendo.** El
  backup **fabricaba cada noche** un objeto inmutable en R2 con los
  `password_hash` adentro: creaba el problema que después había que esperar
  siete días a que caducara.

  **Se redacta el hash, no se excluye la tabla**, y la razón importa: excluir
  `jax_users` haría que una restauración **perdiera las cuentas** —emails,
  roles, tenants—, que son datos de negocio. Redactando solo el hash, la
  restauración conserva los usuarios y lo único que exige es un reset de
  contraseñas. Se redacta **cualquier** bcrypt del dump, no solo los de
  `jax_users`.

  **Verificado con una corrida real:** **34 tablas antes y 34 después** (sin
  pérdida), **0 hashes**, 2 marcadores de redacción, `jax_users` presente con
  sus cuentas, permisos 600, restic con dos snapshots nuevos. El script
  **falla el paso** si queda algún hash sin redactar — no lo deja pasar.

- **ESPERANDO A TERCERO — no son pendientes, no hay trabajo de este lado.**

  | Ítem | Estado | Bloqueado por |
  |---|---|---|
  | **GitHub Support** | Ticket redactado y ampliado con los 18 blobs de datos de clientes | **Respuesta de GitHub** |
  | **Purga de dumps en R2** | Procedimiento y fecha listos: **2026-09-08 ~01:00** | **El Bucket Lock — inmutabilidad por diseño.** Desde hoy los dumps nuevos ya no llevan hashes |

- **CIERRE TOTAL DE LA RONDA DE SEGURIDAD (2026-09-01).** Estado único.
  Nada figura como "pendiente": cada ítem lleva **CERRADO**, **LISTO PARA
  EJECUTAR** (con qué falta) o **NO DETERMINABLE** (con qué se buscó).

  ### CERRADO

  | Ítem | Evidencia |
  |---|---|
  | **Rotación de la credencial** | Huella cambiada + `bcrypt.checkpw` de la aguja en **falso** + login real |
  | **La aguja no abre la segunda cuenta** | `bcrypt.checkpw` contra su hash vivo |
  | Hook `pre-commit`, ambos repos | Probado rompiéndolo; 7 defectos corregidos antes de publicar |
  | Barrera de CI server-side | `required check` activo; caso crítico (merge con `--no-verify`) en rojo |
  | Inventario de los 162 candidatos | 27 con señal, resueltos |
  | Higiene de backup | `UMask=0077`, dirs 750, dumps 600; **corrida real verificada** |
  | `jax_users.updated_at` | `ALTER` aplicado; verificado en test, producción sin tocar |
  | 3 listas de IPs propias | **Riesgo aceptado, firme** |
  | Origen de `user_id=1` | Lo creó `run_seed()`; tenant 41 s antes, literal en un solo archivo |
  | Barrido del dominio de la segunda cuenta | 1.593 blobs, **cero credenciales** |
  | Familia de lecciones | **1-25**, sin huecos |
  | Datos de terceros fuera de HEAD | PR #80; `master` verificado limpio |

  ### LISTO PARA EJECUTAR — falta una decisión o un tercero, no trabajo

  | Ítem | Qué falta | Material |
  |---|---|---|
  | **Notificar a 6 clientes** | **Decisión de Fernando.** Es lo único con obligación posible hacia terceros | `INVENTARIO-CLIENTES.md` + borrador de aviso parametrizable |
  | **Ticket a GitHub Support** | Que Fernando lo pegue y lo envíe | `TICKET-GITHUB.md`, ampliado con los 18 blobs de datos de clientes |
  | **Rotar la segunda cuenta** | **HECHO 2026-09-11** — ver el ítem de la segunda cuenta | Procedimiento idéntico al de `user_id=1`, con testigo antes/después |
  | **Purga de dumps en R2** | **PREMISA FALSA (medido 2026-09-11)**: la retención guarda esos snapshots hasta 6 meses y R2 nunca hace prune — ver el bloque de estado. Resuelto por otra vía: rotación de `user_id=2` | Fecha derivada **solo** del lock, sin mirar retención ni prune |
  | **Excluir/anonimizar `jax_users` del dump** | Ronda propia | Hoy cada backup **crea** el problema que luego caduca |

  ### NO DETERMINABLE, con lo que se buscó

  | Pregunta | Se buscó en |
  |---|---|
  | Los 41 s entre el `INSERT` del tenant y el del usuario | Binlog (`log_bin=OFF`), `general_log` (OFF), `log_error` vacío, logs del contenedor (empiezan 2026-08-09), dumps anteriores (el más viejo es de 2026-07-08), `.bash_history` (sin timestamps) |
  | Si el hash viajó por `mariadb-dump` de la 11.8 a la 12.3 | Inverificable por comparación de huellas: **la 11.8 no existe** y su datadir está vacío. Resultó irrelevante: la fila de producción siempre tuvo ese valor |
  | `retain-until` exacto de los objetos de R2 | Sin `aws`/`rclone`/`s5cmd`/`mc`/`boto3` en el host. La fecha es **derivada**, no leída |

  ### CLASES NO CUBIERTAS — sin suavizar

  1. **142 rutas sin señal estructural**: no dispararon patrones, **que no es
     lo mismo que estar limpias**.
  2. **Objetos colgados server-side**: no hay método cliente. Solo Support.
  3. **Secretos codificados** (base64, URL-encode): el comparador es literal
     sobre bytes.
  4. **Secretos embebidos sin separadores**: inherente al match por token.
  5. **Clones de terceros**: `forks = 0` **no cubre un `git clone`**, que no
     deja rastro, y los repos fueron públicos ~2 meses y medio.
  6. **La contraseña de la segunda cuenta** nunca fue verificada contra nada
     más que la aguja conocida.

  ### Los cuatro falsos positivos de la ronda, y su causa única

  "Credencial viva en `master`" (era el marcador de `filter-repo`), "segunda
  contraseña" (era `tu_password`), "`access_token` desconocido" (era la
  cabecera JWT canónica), y "el seeder nunca creó `user_id=1`" (sí lo creó).
  **Los cuatro por clasificar por apariencia o por inferencia en vez de por
  comparación con el estado real.** Ninguno sobrevivió a `master`; los cuatro
  quedaron registrados, no borrados.

- **ROTACIÓN EJECUTADA Y VERIFICADA — el incidente está cerrado (2026-09-01).**

  `jax_users.user_id=1` en `jax_memory` (`127.0.0.1:3308`).

  | | Huella MD5 del hash |
  |---|---|
  | Original | `155c47c3263a6771cf8b854a698f443d` |
  | Intermedia (incidente, ver abajo) | marcador de 48 caracteres, no bcrypt |
  | **Nueva** | `7243bc8cdc39c63a3d91cf4ac8a9cf7b` |

  **Verificada por TRES vías independientes**, ninguna heredada:
  1. La huella cambió respecto de la del testigo pre-rotación.
  2. **`bcrypt.checkpw(aguja_filtrada, hash_nuevo)` → FALSO.** La contraseña
     que estuvo pública ~2 meses ya no abre producción.
  3. Login real en `axioma-ia.io` con el valor nuevo: OK.
  Prefijo `$2b$12$`, largo 60 — bcrypt cost 12, algoritmo sin cambios.

  ### Por qué la rotación anterior no había cubierto esta fila

  **CORRECCIÓN (2026-09-01, posterior): el seeder SÍ creó `user_id=1`.** Lo
  que sigue afirmaba lo contrario y era falso. Se deja registrado, no borrado.

  **Evidencia convergente, verificada:** `jax_tenants.tenant_id=1` es
  `'Inversiones Diamante Negro'`, `plan='superadmin'`, creado **2026-06-18
  17:11:51** — **41 segundos antes** que la fila de usuario. Ese literal
  exacto existe en **un solo archivo de todo `jax-platform`: `seed.py`**
  (verificado por `grep -rl`), y `run_seed()` inserta el tenant y el usuario
  en la misma función. El código estaba en disco y todavía sin commitear: se
  commiteó ~11 h después en `ed7719a`.

  **El email con punto no viene de ningún código:** son dos `UPDATE` hechos
  desde la propia Mesa el 2026-06-19, y por eso el gate `COUNT(*)` nunca los
  revirtió.

  **Qué NO cambia con esta corrección, y es lo que importaba:** la conclusión
  operativa se sostiene entera. El gate `COUNT(*)` significa que **el seeder
  no reescribe una fila existente**, así que la rotación tenía que ser un
  `UPDATE` directo — como se hizo. Lo que era falso era el relato del origen,
  no el procedimiento.

  Lo que se afirmaba antes, incorrecto:
  - Email en producción: `fernando.ruiz@rich-hn.com` (**con** punto). El
    seeder inserta `fernando@rich-hn.com` (**sin** punto).
  - Fila creada el **2026-06-18 17:12:32**, *anterior* al primer commit del
    repo (2026-06-19 04:42).
  - El gate del seeder es `COUNT(*)` sobre `user_id=1` ⇒ con la fila ya
    presente, **nunca la sobrescribió**.

  **Rotar por la vía del seeder no podía funcionar**, y por eso la rotación
  previa no tocó esta fila. La contraseña filtrada siguió siendo la de
  producción hasta hoy — **primer dato MEDIDO** sobre si el secreto llegó a
  producción; todo lo anterior era inferencia encadenada.

  **La tabla de entornos de Q2** ("el seeder escribió `user_id=1` con la aguja
  en cada entorno") queda marcada como **HEREDADA Y CONTRADICHA POR MEDICIÓN**.
  Llegó a producción, sí — pero por una razón distinta de la que suponíamos.

  ### Incidente durante la rotación, registrado

  El primer `UPDATE` guardó **el texto del marcador del comando de ejemplo**
  (48 caracteres, no un bcrypt) en el `password_hash` de producción. **Nadie
  pudo autenticarse como superadmin durante ~3 minutos.**

  Se detectó **por el `LENGTH`** en la verificación del testigo: 48 en vez de
  60. **Sin esa columna en el testigo, un hash inválido habría pasado por
  bueno** y el fallo se habría descubierto por un usuario sin poder entrar.

  Es la **misma clase que `da9fd5ec`**: la remediación introduce el defecto que
  venía a arreglar. Ver la duodécima lección en `CONTEXT.md` §9.

- **la **segunda cuenta** (`user_id=2`, rol `operator`) — cuenta con acceso, fuera de todo lo auditado
  (2026-09-01).** `user_id=2`, rol `operator`, `status=active`, creada
  2026-06-19 16:09:12. **Su contraseña no fue verificada contra nada**, y
  ningún barrido de la auditoría buscó el dominio de esa cuenta — el barrido
  por identidad cubrió `rich-hn.com` porque era el único dominio conocido.
  **Barrido CERRADO (2026-09-01): cero credenciales.** Enumeración completa de
  1.593 blobs en ambos repos con `refs/pull/*` traídas. la **segunda cuenta** (`user_id=2`, rol `operator`):
  **0 apariciones**. Búsqueda ampliada por regex (el usuario local contra cualquier dominio,
  el dominio): los mismos resultados. Ese dominio aparece en **7 blobs de
  `jax`**, todos como **dato de inventario**, y el escaneo de campos de
  credencial sobre los blobs completos —no solo las ventanas— dio **0
  coincidencias**. No hubo ningún valor que clasificar, así que el gate de
  palabras autodescriptivas no llegó a aplicarse.

  **Confirmado de paso:** la única variante de correo `@rich-hn.com` en la
  historia de ambos repos es `fernando@rich-hn.com` (54 apariciones).
  `fernando.ruiz@rich-hn.com` —el de producción— **no existe en ningún blob**.

  **Queda: la contraseña de esta cuenta nunca fue verificada contra nada.** No
  estaba filtrada en los repos, que es distinto de estar sana. Fecha de
  control: **2026-09-08**.

  **ROTADA 2026-09-11 04:50:10 — orden de Fernando.** Motivo que la volvió
  necesaria: su hash **vigente** estaba en 5 snapshots de restic (R2 y local)
  que la retención guarda hasta 6 meses (ver el bloque de estado, arriba).
  Procedimiento de `ROTACION-user_id-2.md`, con dos ajustes medidos:

  | Paso | Resultado |
  |---|---|
  | Testigo ANTES | `$2b$12$`, largo 60, huella `f6b33083…`, `updated_at` NULL — **idéntico al del 2026-09-01**: la fila no se tocó en 10 días |
  | Respaldo del hash previo, para revertir | `$AUDIT/user_id-2-hash-previo-20260911-045010.txt` (600) |
  | Valor nuevo | `secrets.token_urlsafe(18)`, hash con `bcrypt.hashpw(..., gensalt())` — la misma llamada del backend (`api/admin/users.py`) |
  | `UPDATE` | guardado por la huella del testigo (`WHERE MD5(password_hash)=…`): si la fila hubiera cambiado, no escribía nada. `rowcount=1` |
  | Testigo DESPUÉS | `$2b$12$`, largo 60, huella `520d73bf…` (distinta), **`updated_at` 2026-09-11 04:50:10** — la columna agregada el 09-01 para esto funcionó |
  | Verificación | `db.seed.verify_password` (la función del login) → **True** con el valor nuevo y **False** con uno alterado; `status=active`, sin `locked_until` |

  **Ajuste 1 — no hay "checkpw del valor viejo en falso":** el valor viejo de
  esta cuenta no lo conoce nadie. Se reemplazó por el contrapositivo de arriba.
  **Ajuste 2 — no se hizo login real por la API:** `/api/auth/login` escribe
  `last_login`, y habría registrado como entrada de la persona una que fue de
  Claude — ensuciando el mismo campo que mostró que la cuenta **no se usa desde
  el 2026-06-19** (`last_login` sigue ahí, sin tocar). El login real de la
  persona es la verificación final, de su lado.

  **Pendiente de Fernando:** entregar el valor por un canal **fuera de banda**
  (no por el correo de esa cuenta) — está en
  `$AUDIT/user_id-2-contrasena-nueva-20260911-045010.txt` (600) — y **borrar
  ese archivo** después. `jax_memory_test` no tiene `user_id=2` (medido): el
  paso 5 del procedimiento no aplica. **Observación, no decisión:** la cuenta
  no se usa hace casi 3 meses y tiene `failed_attempts=1` de antes de la
  rotación; si no hace falta, desactivarla cierra más que rotarla.

- **Inventario operativo de infraestructura de CLIENTES en historia pública —
  hallazgo lateral de L4 (2026-09-01). Decisión pendiente.** No es una
  credencial, y por eso ningún barrido de secretos lo iba a marcar. En
  `missions/` de `jax`, fuera de HEAD desde `f6c8e7d` pero **vivo en la
  historia**: los nombres de usuario de panel de **6 clientes**, sus dominios, la cantidad
  de cuentas de correo y el tamaño de buzón de cada uno, los nombres de sus
  bases de datos con el gestor de contenidos que revelan, rutas de Maildir, y —en las versiones anteriores a `99ad51a`— **IPs internas y el
  puerto SSH 58291**. Incluye además el diagnóstico de que **dos de esos dominios tienen
  certificados vencidos hace más de un año y su origen no responde a ACME** —
  una debilidad activa, documentada y no corregida.

  **El detalle por cliente NO va en este documento.** Vive en
  `/home/fruiz/security-audit-2026-09/INVENTARIO-CLIENTES.md` (chmod 600,
  fuera de ambos repos). Nombrar acá a los clientes sería devolver a HEAD
  exactamente lo que `f6c8e7d` sacó en agosto — y este documento es público.
  Es el mismo criterio que se aplicó desde el principio a la credencial
  ("referencia por ruta y por hallazgo, nunca por contenido"), que en la
  primera redacción **no se aplicó a los datos de clientes**.

  **Por qué esto no entra en el mismo cajón que las 3 listas de IPs propias:**
  aquello era topología de Fernando y la decisión de aceptar el riesgo era
  suya. **Esto es información de terceros** — clientes que no participaron de
  la decisión. Es reconocimiento útil para un atacante, con una debilidad
  concreta nombrada. **Decisión pendiente, y no es solo técnica.** Fecha de
  control: **2026-09-15**.

- **Dump nocturno con hashes legible por todo el host — severidad media.**
  `/srv/backup-adata/staging/mariadb-local/jax_memory.sql` (11.5 MB) queda
  `rw-rw-r--`. Lo escribe `backup-hall9000.service` como `User=fruiz`, **sin
  `UMask`** declarado, así que hereda el default. **Corrección propuesta, no
  aplicada: `UMask=0077` en la unidad systemd** — no toca el script, no
  requiere `chmod` ni regla de sudoers, y no rompe restic porque corre en el
  mismo servicio y con el mismo usuario. Los directorios `/srv/backup-adata`
  y `staging` son `755`/`775` y también convendría cerrarlos.

- **`jax_users` sin columna `updated_at` — deuda de esquema.** Solo hay
  `created_at`, `last_login`, `failed_attempts`, `locked_until`. **No se puede
  fechar cuándo cambió un `password_hash`**, que es exactamente lo que hizo
  falta en esta ronda y obligó a construir un testigo a mano.

- **PREGUNTA ABIERTA: ¿quién creó `user_id=1` el 2026-06-18, y por qué su
  email lleva punto y el del seeder no?** No hay respuesta; queda como
  pregunta, **no como respuesta inventada**. Fecha de control: **2026-09-08**.

- **CIERRE DE LA RONDA DE SEGURIDAD — estado único (2026-09-01).** Reemplaza
  cualquier reconstrucción de estado a partir de mensajes sueltos.

  ### Hallazgo real, total

  **UN secreto en toda la historia de ambos repos.** Contraseña del superadmin
  de seed, **8 caracteres**, en **7 rutas fuente** de **2 repos**, ventana
  desde **2026-06-19**, **`forks = 0`** en ambos (medido por API).

  ### Tres falsos positivos, los tres retractados antes de llegar a master

  | Afirmación | Qué era | Cómo se cayó |
  |---|---|---|
  | "credencial viva en el árbol de `master`" | marcador de `filter-repo` | `grep -c REMOVED` |
  | "segunda contraseña de la cuenta" | `tu_password`, placeholder en español | palabras autodescriptivas |
  | "`access_token` desconocido" | cabecera JWT canónica HS256 + `...` | decodificación de segmentos |

  **Los tres por la misma causa: clasificar por apariencia en vez de por
  comparación.** Ninguno llegó a `master`; los tres quedaron registrados, no
  borrados.

  ### Cerrado en git

  - **Hook `pre-commit`** en ambos repos, probado rompiéndolo, con 7 defectos
    corregidos que la primera batería no había probado.
  - **Barrera de CI server-side** (`secret-scan`), que cubre lo que el hook no
    puede ver — merge, rebase, cherry-pick, revert. **Required check activo en
    el ruleset de ambos repos.**
  - **Inventario de los 162 candidatos**, reducido a 27 con señal y resuelto.
  - **Familia de lecciones de método 1-24**, completa y sin huecos.
  - **`DEUDA.md` reconciliada** con el estado real.

  ### ABIERTO fuera de git — con fecha de control, porque un ítem sin fecha se evapora

  **(a) ROTACIÓN EN BASES — SEVERIDAD ALTA. Fecha de control: 2026-09-08.**
  Una sola cuenta: el superadmin de seed. Checklist de 7 pasos más abajo en
  este documento. **ORDEN CRÍTICO, y es contraintuitivo:**
  1. **PRIMERO** cambiar el hash de `user_id=1` en **cada base**.
  2. **DESPUÉS** declarar `JAX_SEED_ADMIN_PASSWORD` en el entorno.

  **Al revés no funciona y además engaña:** el gate del seeder es `COUNT(*)`,
  así que con la fila ya existente el seeder no vuelve a escribir nada — te
  quedás con el valor viejo en la base **creyendo que rotaste**.

  Bases, en orden: **prod 11.8 → 12.3 Docker → `jax_memory_test`** (que de paso
  convierte el "probable, no confirmado" en confirmado) **→ dev local**.

  **(b) DUMPS EN R2 con Bucket Lock (7 días).** Retienen el hash viejo hasta
  caducar y **no se pueden borrar antes**. **Fecha de control: 7 días después
  de que se ejecute (a)** — no antes, porque hasta que la rotación ocurra cada
  noche se crea un dump nuevo con el hash viejo. Es una dependencia, no una
  fecha fija.

  **(c) 3 LISTAS DE IPs PRIVADAS — RIESGO ACEPTADO, FIRME (decisión de
  Fernando, 2026-09-01). CERRADO.** `JAX_ENV_STAGING_HOSTS`,
  `JAX_ENV_PROD_HOSTS`, `JAX_ENV_BRIDGE_HOSTS`, en 80 blobs de ambos repos,
  ninguna en el árbol de hoy. **No se hace la reescritura de historial que las
  sacaría.**

  Se registra para que **no se relitigue**: si una auditoría futura vuelve a
  encontrarlas, **no es un hallazgo nuevo — es este ítem**, y la respuesta ya
  está dada. Este documento tenía la decisión escrita en dos estados
  contradictorios a la vez; queda en uno solo.

  Hechos medidos que la acompañan —**hechos, no los motivos de Fernando**, que
  no se registraron—: son direcciones **RFC1918**, no enrutables desde
  Internet; `forks = 0` en ambos repos; y la reescritura **no alcanzaría los
  `refs/pull/*`** de todos modos, el mismo límite que este documento establece
  para el secreto real.

  **Qué la reabre** — condiciones, no predicciones:
  - Que alguna pase a contener una IP **pública**.
  - Que se ejecute la purga con GitHub Support: ahí el costo marginal de
    incluir estas rutas es casi cero y conviene sumarlas.
  - Un incidente cuyo vector de entrada sea reconocimiento de red interna.

  **Ojo, y es distinto:** esta decisión cubre **la topología propia**. El
  inventario operativo de **clientes** que apareció después es de terceros que
  no participaron de ella, y va como ítem aparte.

  **(d) TICKET A GITHUB SUPPORT — redactado, SIN ENVIAR.**
  `/home/fruiz/security-audit-2026-09/TICKET-GITHUB.md` (chmod 600). Es la
  única vía a `refs/pull/*` y a objetos server-side. **Dato que importa para
  el ticket:** de los 8 blobs con el valor, siete viven **solo** en refs de PR,
  pero el `.pyc` sigue alcanzable desde **8 ramas vivas** — si Support solo
  purga refs de PR, ese queda.

  ### CLASES NO CUBIERTAS — sin suavizar

  1. **Secretos nunca conocidos en las 142 rutas sin señal estructural.** No
     dispararon ninguna de las 6 categorías; eso significa que **no contienen
     los patrones buscados, no que estén limpias**.
  2. **Objetos colgados del lado del servidor.** No hay método desde el
     cliente. Solo GitHub Support.
  3. **Secretos codificados** — base64, URL-encode, UTF-16. El comparador es
     **literal sobre bytes**: no ve una transformación del valor.
  4. **Secretos embebidos sin separadores alrededor** (`pw = "prefijo<valor>"`).
     Inherente al match por token.
  5. **Clones de terceros.** `forks = 0`, pero un `git clone` no deja rastro y
     los repos fueron públicos ~2 meses.

- **RETRACTACIÓN — el "segundo secreto" NO existe (2026-09-01).** Se reportó
  una segunda contraseña de producción de `fernando@rich-hn.com` en
  `missions/axioma-admin-y-login-fixes.md` y se escaló. **Era falso.** Queda
  registrado, no borrado, como el primer P0 falso de esta misma ronda.

  **La evidencia que lo cierra:**

  | Blob | Largo | Qué es en realidad |
  |---|---|---|
  | `9febaa5d…`, `abe82a30…` | 11 | **`tu_password`** — placeholder en español (palabras autodescriptivas `tu_`, `pass`; todo minúsculas; solo `[A-Za-z0-9_]`) |
  | `6d7c7ed6…` | 39 | **el marcador de `filter-repo`** — 39 es su largo exacto |

  **El cuadro de recall de los cuatro métodos queda INVÁLIDO.** Se construyó
  sobre 2 secretos reales y hay **uno**. El único dato que sobrevive es que
  `gitleaks` 8.30.1 no encontró el que sí existía.

  **Procedencia del error, cuatro capas otra vez:**
  1. El agente de lectura dirigida lo clasificó `DESCONOCIDO` — mecánicamente
     correcto: no coincidía con ninguna aguja.
  2. **Hipatia lo verificó con el chequeo equivocado y publicó la
     comprobación como concluyente.** Corrió el test de plantillas (`{`, `<`,
     `$`) y **no** el de palabras autodescriptivas.
  3. Se amplificó a "patrón de trabajo" y motivó un barrido por identidad.
  4. Llegó a los PRs #75 y #34.

  **Lo que vuelve este error distinto de los anteriores: el instrumento
  correcto existía y se había usado diez minutos antes.** El chequeo de
  palabras autodescriptivas se aplicó con éxito a los candidatos #4, #5 y #6
  de la categoría (b) —y descartó los tres correctamente— y no se aplicó a la
  única afirmación que se iba a escalar a P0. Ver la vigésimo cuarta lección
  en `CONTEXT.md` §9.

  **Estado real tras la retractación: UN secreto en toda la historia de ambos
  repos** — la contraseña del superadmin de seed. `fernando@rich-hn.com` no
  tiene nada que rotar por este lado.

  **Otros dos candidatos verificados y descartados en la misma ronda:** el
  `access_token` de `missions/axioma-login-prod-fix_result.md` es la
  **cabecera JWT canónica HS256 seguida de `...`** (36+3 caracteres, segmentos
  `[36,0,0,0]`, payload y firma vacíos) — idéntica en todos los tokens HS256
  que existen, cero material secreto; y `token_type` es literalmente `bearer`.

  **Lo que el barrido por identidad SÍ dejó, y es sólido:** una sola cuenta en
  todo el dominio; **ninguna** de las 9 claves de `/etc/jax/.env` en ningún
  blob de ninguno de los dos repos (control válido: sí encontró los valores
  *no* secretos del mismo archivo); y la confirmación —tercera, independiente—
  de que ningún valor real sigue en `refs/heads/*` y todos sobreviven por
  `refs/pull/N/head`.

- **No existe barrera de contenido en el camino rama → master — CERRADO
  2026-09-01** (era SEVERIDAD ALTA). Cerrado por el check de CI
  `secret-scan` (`ops/ci/scan-pr-secrets.py`), server-side, sobre el diff
  `base...head` completo del PR. **Probado rompiéndolo, con el caso crítico
  incluido:** PR limpio → verde; con la aguja → rojo; **la aguja introducida
  por MERGE de otra rama commiteada con `--no-verify` → ROJO**, que es
  precisamente lo que el hook local no puede ver; binario con la aguja →
  rojo; lista borrada → rojo (fail-closed).

  **El pepper NO va al runner** (decisión de Fernando, 2026-09-01): meterlo
  como secret de Actions crearía superficie nueva y un secret más que rotar.
  El check usa una lista paralela con **salt público por entrada y `scrypt`
  n=2^14** (~26 ms por comparación). Medido: ~2.500 tokens únicos en un diff
  real × 2 entradas ≈ **130 s de CI**. Crece **lineal** con las entradas —
  10 entradas serían ~11 min, y ahí hay que revisar el número.

  Diagnóstico original, que se conserva: Hallazgo del juez que atacó el hook `pre-commit`, más
  grande que lo que ese hook cierra.

  **Medido, no razonado** (con un hook sonda, `jax`, 2026-09-01):

  | Operación | ¿Ejecuta hooks de pre-commit? |
  |---|---|
  | `git commit`, `git commit --amend` | **SÍ** |
  | `merge`, `rebase`, `cherry-pick`, `revert`, `stash` | **NO** |

  Y el `pre-push` **solo mira el ref destino, no el contenido** — por diseño:
  se escribió para frenar un push que aterrizara en `master`, no para
  revisar qué lleva adentro.

  **La consecuencia es concreta y hoy está activa:** un commit hecho **antes**
  de activar `core.hooksPath`, o hecho con `--no-verify`, **entra a `master`
  por merge o rebase sin pasar jamás por ninguna revisión de contenido**. El
  hook cubre el commit directo y nada más. Todo el pasado de ambos repos está
  en esa condición, porque el hook es de hoy.

  **El hook local es defensa en profundidad, NO la barrera.** Escrito acá
  explícitamente para que nadie lo lea como cobertura: es evadible con
  `--no-verify`, con dos variables de entorno, y borrando una entrada de la
  lista en el working tree sin commitear. Las cuatro son actos explícitos, y
  ninguna deja rastro en el commit que llega a `master`.

  **Dirección de solución — NO implementada, y a propósito:** un job de CI que
  escanee el diff del PR contra la lista de hashes. Server-side, así que no lo
  evade `--no-verify` ni un `.gitattributes` local, y **gateado por la
  condición de merge**, que es lo único que convierte una revisión en una
  barrera. Es lo único que cierra la clase entera; el hook local solo cierra
  el commit directo. Requiere resolver dónde vive el pepper para el runner —
  un secret del repo— y esa decisión no está tomada.

- **P0 — Credencial de producción expuesta en repo público (GitGuardian,
  2026-09-01).** Hallazgo externo: GitGuardian alertó sobre una credencial
  de producción en claro en el repositorio **público** `fjruizhn/Jax`,
  archivo **`missions/axioma-login-prod-fix.md`** — un `curl` que llevaba
  usuario y contraseña embebidos.

  **Referencia por ruta y por hallazgo, nunca por contenido.** Este ítem no
  transcribe el valor, ni el usuario completo, ni el comando: registrar el
  secreto en la lista de deuda para "documentarlo bien" lo volvería a
  publicar en el mismo repo público. Quien necesite el detalle va a la
  alerta de GitGuardian, no a este archivo.

  **Estado de las piezas:**

  | Pieza | Estado |
  |---|---|
  | Rotación de la contraseña | **HECHA** |
  | Limpieza de HEAD (`missions/`, `CLAUDE.md`, `prompts/`, `policy/rules/OP02-05`) | **HECHA** en `f6c8e7d` (2026-08-21, B1.4) — `missions/` tiene 0 archivos trackeados en HEAD |
  | Barrido de credenciales en el historial completo | **PENDIENTE** |
  | Rotación de todo lo que aparezca en el barrido | **PENDIENTE** |
  | Solicitud de purga a GitHub Support | **PENDIENTE** |
  | Hook pre-commit anti-credenciales | **CERRADO** (ver abajo) |

  **El historial sigue VIVO, y esa es la distinción que importa.** HEAD está
  limpio desde el 2026-08-21; el contenido no. En el historial hay
  **4 versiones de blob** de `missions/axioma-login-prod-fix.md` y
  **6 versiones** de un segundo archivo que la alerta de GitGuardian no
  nombró, `missions/axioma-login-prod-fix_result.md`. Cualquier trabajo
  sobre este ítem cubre los dos, no solo el que salió en la alerta.

  **Alcanzabilidad medida (2026-09-01):** los blobs son alcanzables desde
  `master` **y desde 70 `refs/pull/*`**.

  **CONSECUENCIA — la reescritura de historial NO es remediación.** Las
  `refs/pull/*` las mantiene GitHub del lado del servidor y **no se borran
  con un push**: `filter-repo` + force-push reescribe las ramas y deja los
  blobs igual de fetcheables por sus refs de PR. Los forks, además,
  comparten object store con el repo de origen. Un repo que "se ve limpio"
  después de reescribir sigue sirviendo el secreto a quien pida el objeto
  por SHA.

  Por lo tanto la remediación real es, en este orden:
  1. **Rotar todo lo que aparezca en el barrido.**
  2. **Solicitar la purga a GitHub Support** — es la única vía que alcanza
     objetos server-side y refs de PR.

  La reescritura de historial queda como **higiene posterior, no como
  cierre**. Tratarla como cierre es exactamente el error que este ítem
  existe para prevenir: produce la apariencia de resolución sin la
  resolución.

  **RESTRICCIÓN VIGENTE — no reescribir historial hasta ver el barrido
  completo.** Nada de `filter-repo`, `rebase`, `gc`, `prune` ni force-push
  sobre ninguno de los dos repos mientras el barrido no esté hecho y leído.
  El motivo no es cautela genérica: reescribir ahora **destruye la
  evidencia** con la que se determina el alcance real, y deja sin responder
  la única pregunta que importa — qué OTROS secretos estuvieron expuestos.
  Una limpieza que borra el historial antes de haberlo leído produce un
  repo que *parece* limpio y un alcance que ya no se puede establecer.

  **ALCANCE REAL, MEDIDO EL 2026-09-01 — y una corrección de este mismo
  documento, ver más abajo.** Barrido por valor sobre mirrors frescos con
  `refs/pull/*` traídas, enumerando **todos** los objetos: 857/857 blobs
  leídos en `jax`, 684/684 en `jax-platform`. Reporte completo en
  `/home/fruiz/security-audit-2026-09/REPORTE-BARRIDO.md` (chmod 600, fuera
  de ambos repos).

  **HAY UNA SOLA CREDENCIAL, de 8 caracteres**: la del superadmin
  `fernando@rich-hn.com` (`user_id=1`, `tenant_id=1`, tenant "Inversiones
  Diamante Negro") en el backend de **jax-platform / Axioma**. No hay una
  segunda cuenta ni un segundo sistema — verificado leyendo el contexto de
  los blobs, no inferido.

  ### CORRECCIÓN — el "P0: credencial viva en master" fue FALSO

  Se reportó que `master:backend/tests/test_seed_admin_password.py` contenía
  una credencial viva. **Es falso y queda registrado, no borrado.** Lo que
  hay en ese archivo es el marcador que insertó `filter-repo` en la ronda 9:
  `***REMOVED-SEE-JAX-RONDA9-2026-08-20***`.

  | Blob | Contenido | Dónde vive |
  |---|---|---|
  | `2199fabd…` | **el marcador** (4× `REMOVED`, 4× `***`) | `master` + 8 ramas |
  | `f71bd511…` | **el valor real** (0 marcadores) | **solo `refs/pull`** |

  **Procedencia del error, porque el mecanismo importa más que el error:**
  ejecutor del barrido → juez "independiente" de TA2 → Hipatia → Fernando.
  **Cuatro capas, severidad creciente, ninguna corrió `grep -c REMOVED`** —
  un comando de dos segundos. El juez confirmó la conclusión equivocada
  porque corrió *el mismo método* que el ejecutor. Ver las lecciones
  decimotercera, decimocuarta y decimoquinta en `CONTEXT.md` §9.

  ### Dónde SÍ sobrevive el valor real (medido, no inferido)

  El `filter-repo` de la ronda 9 quedó **incompleto en dos frentes**:

  1. **`backend/db/__pycache__/seed.cpython-312.pyc`** contiene la
     contraseña real. Confirmado **por presencia** (comparación literal de
     la aguja contra los bytes del blob) más tres controles: el marcador NO
     está en el `.pyc`; la aguja SÍ está en el `seed.py` del linaje viejo; y
     NO está en `master:backend/db/seed.py`. Alcanzable desde **8 ramas
     vivas y 31 `refs/pull/*`**; ausente del árbol de hoy. `filter-repo`
     reescribió el `.py` y no tocó el bytecode que lo había compilado.
  2. **Los `refs/pull/*` conservan el linaje pre-scrub** — `seed.py` con la
     contraseña en texto plano en `refs/pull/1-7`. Confirmación empírica de
     lo que dice la CONSECUENCIA de arriba: reescribir ramas no las alcanza.

  **Ventana: desde 2026-06-19** en repos públicos. **`forks = 0`** en ambos
  (verificado por API) — única clase de exposición cerrada.

  **`gitleaks` no sirve como criterio de cierre acá, y se midió**: v8.30.1
  devolvió `[]` en `jax-platform` pese a que la contraseña está en texto
  plano en `seed.py` en refs que gitleaks demostradamente escanea. Ver la
  decimoséptima lección en `CONTEXT.md` §9.

  ### PENDIENTE — rotación en BASES, obligatoria — SEVERIDAD ALTA

  **La credencial se asume CONOCIDA por terceros. No es precaución.**
  Propiedades medidas, sin especular sobre si alguien la obtuvo:
  - **8 caracteres**, elegida por un humano (extraída del par de blobs A/B
    del mismo archivo, 92 líneas alineadas, una sola aguja).
  - Expuesta en repositorio **público** desde **2026-06-19**, ~2 meses, en
    `refs/pull/*` como texto plano y en un `.pyc` alcanzable desde 8 ramas.
  - A ese largo, **crackeable offline en tiempo trivial** aunque solo se
    tuviera el hash bcrypt; y acá no hacía falta el hash, estaba el texto.

  La consecuencia operativa es que la rotación no cierra un riesgo
  hipotético: cierra uno que hay que tratar como materializado.

  **Cuál es la contraseña vigente hoy — determinado por cronología de código
  contra cronología de entornos, sin tocar ninguna base:**

  El fallback aleatorio (`_resolve_seed_admin_password`) se introdujo en
  `da9fd5ec`, **2026-08-20**. Antes de eso la contraseña estaba hardcodeada.
  `run_seed()` se llama desde `main.py` desde el **primer commit del repo**
  (`5e28e9e`, 2026-06-19). Todo entorno sembrado antes del 2026-08-20 tiene
  la aguja.

  | Entorno | Vigente | Sostén |
  |---|---|---|
  | `jax_memory` prod 11.8 | **(a) la aguja** | journald de `jax-platform.service` arranca **2026-07-08**, seis semanas antes del fallback |
  | `jax_memory` 12.3 Docker | **(a) la aguja** | copiada por `mariadb-dump` de la 11.8; hereda la fila `user_id=1` |
  | Dumps en R2 | **(a) la aguja** | son dumps de las anteriores |
  | `jax_memory_test` (ambas) | **(a) probable**, no confirmado | los tests corrían desde antes del 2026-08-20 y el gate `COUNT(*)` hace que gane la primera siembra; no verificable sin tocar la base |
  | Dev local | **INDETERMINADO** | depende de cuándo levantó cada máquina; no determinable desde el código |

  **Si algún entorno resultara (b)** —sembrado después del 2026-08-20, con un
  `token_urlsafe(18)` generado y logueado una sola vez— ese superadmin tiene
  **una contraseña que nadie conoce**. Es un problema operativo distinto de
  la fuga y se resuelve aparte (resetear, no rotar). Hoy no hay evidencia de
  que ningún entorno esté en (b).

  ### PROCEDIMIENTO DE ROTACIÓN — REESCRITO 2026-09-01 con la topología REAL

  **El checklist anterior asumía 4 bases sobre 2 instancias y una vía de
  rotación por el seeder. Las dos cosas eran falsas.** Topología medida:
  **una sola instancia** (12.3 Docker, `127.0.0.1:3308`) con **dos esquemas**
  (`jax_memory`, `jax_memory_test`). La instancia 11.8 en `:3306` **no existe
  hoy** y su datadir `/var/lib/mysql` está vacío (20K, solo `lost+found`).

  1. **Sacar el testigo ANTES** de tocar nada:
     `SELECT LEFT(password_hash,12), LENGTH(password_hash), MD5(password_hash) FROM jax_users WHERE user_id=<id>;`
     **`LENGTH` no es opcional** — es lo que atrapó el marcador de 48
     caracteres guardado como hash en esta misma ronda.
  2. **Generar el valor con `secrets.token_urlsafe`**, nunca elegido por un
     humano: el largo de 8 caracteres fue lo que volvió crítico este caso.
  3. **`UPDATE` directo sobre la fila.** **NO por la vía del seeder:** su gate
     es `COUNT(*)`, así que con la fila presente no escribe nada — y en
     producción la fila ni siquiera la creó él.
  4. **Verificar por tres vías:** huella distinta, `bcrypt.checkpw` del valor
     viejo en **falso**, y login real.
  5. **`jax_memory_test`** se rota o se resiembra en el mismo pase.
  6. **R2:** los dumps con el hash viejo sobreviven hasta que caduque el
     Bucket Lock (7 días). Registrar la fecha y verificar el purgado.
  7. **Declarar `JAX_SEED_ADMIN_PASSWORD`** en el entorno para que el fallback
     deje de ser la ruta real en una base nueva. Es higiene futura, **no parte
     de la rotación** — no afecta filas existentes.

  ### POLÍTICA del reemplazo — no solo el valor

  1. **El reemplazo NO lo elige un humano.** Se genera
     (`secrets.token_urlsafe` o equivalente) y se inyecta por variable de
     entorno. **El camino ya existe en el código y no se usa.**
  2. **Declarar `JAX_SEED_ADMIN_PASSWORD` explícitamente en el entorno de
     cada despliegue** — hoy no está en `/etc/jax/.env`, así que la ruta
     real es el fallback, que es la ruta que loguea en claro.
  3. **Sin literales de contraseña en tests.** El test de regresión negativa
     debe comparar contra un valor inyectado, nunca hardcodeado: ese fue
     exactamente el mecanismo por el que `da9fd5ec` reintrodujo el secreto.


  **No es un cambio de repo.** El seeder corre en el lifespan de FastAPI
  (`backend/main.py:86`), en **cada arranque del backend**, y escribe
  `user_id=1` cuando no existe. El gate es `COUNT(*)`, así que **cambiar la
  contraseña por la app no re-dispara al seeder ni lo revierte**. Entornos
  donde escribió esa fila:

  | Entorno | Evidencia |
  |---|---|
  | `jax_memory` prod, MariaDB 11.8 :3306 | Directa |
  | `jax_memory`, MariaDB 12.3 Docker :3308 — copiada por `mariadb-dump`, **el hash viejo viajó tal cual** | Directa |
  | `jax_memory_test` en ambas instancias | Directa |
  | Máquinas de desarrollo local | Inferida fuerte |
  | **Dumps en R2** (`hall9000-critical-backup`, Bucket Lock 7 días) — retienen el hash hasta caducar | Inferida fuerte |
  | CI de GitHub Actions | **Descartada** — `JAX_CI_NO_DB=1` |

  ### Defecto latente detectado de paso (no disparado)

  `_resolve_seed_admin_password()` (`backend/db/seed.py:30-34`) **loguea la
  contraseña generada en claro** en nivel WARNING, y el docstring dice
  "nunca a un archivo" — inexacto, porque `jax-platform.service` corre bajo
  systemd y journald persiste en disco. **Medido: nunca se disparó.** La
  llamada está dentro del `if count == 0`, y el conteo en journald es **0**
  sobre la ventana `2026-07-08 → 2026-09-01`. `JAX_SEED_ADMIN_PASSWORD` no
  está declarada en `/etc/jax/.env`, así que la rama de fallback es la que
  correría en una base nueva. **No se propagó a backups**: `restic` respalda
  solo el staging con `--exclude='*.log'`; `/var/log` y el journal no están
  en el set.

  ### TA3 — barrido de `/etc/jax/.env`: LIMPIO

  **Ninguna de las 9 credenciales reales aparece en ningún blob**
  (`DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, `ZAI_API_KEY`, `JAX_DB_PASSWORD`,
  `OPENAI_API_KEY`, `TELEGRAM_BOT_TOKEN`, `KIMI_API_KEY`, `JAX_JWT_SECRET`,
  `FERNET_KEY`). Ninguna entra a la lista de rotación.

  **13 claves NO verificables por longitud o trivialidad — quedan a juicio
  manual, TA3 no las cubrió:** `LAS_MANOS_URL`, `JACOBS_URL`,
  `FRONTEND_ORIGIN`, `JAX_DB_USER`, `JAX_DB_NAME`, `JAX_SSH_USER`,
  `TELEGRAM_CHAT_ID`, `JAX_REPL_USER_ID`, `JAX_REPL_TENANT_ID`,
  `JAX_DB_PORT`, `JAX_SSH_PORT`, `JAX_WORKSPACE_DIR`, `JAX_DB_HOST`.

  **Topología interna en historia pública (no rotable):**
  `JAX_ENV_STAGING_HOSTS`, `JAX_ENV_PROD_HOSTS`, `JAX_ENV_BRIDGE_HOSTS` —
  listas de IPs privadas, presentes en 80 blobs de ambos repos, ninguna en
  el árbol de hoy. No hay nada que rotar: o se acepta como riesgo, o exige
  reescritura de historial adicional. **Decisión de Fernando, sin tomar.**

  ### Resto del pendiente
  1. Sacar el literal del test y del `.pyc` en cualquier scrub futuro
     (incluir `*.pyc` en los patrones).
  2. **162 rutas candidatas** de la clase "captura de salida de comandos"
     (73 son `*_result.md`) sin inspeccionar.
  3. Purga a GitHub Support — única vía que alcanza `refs/pull/*` y objetos
     server-side.
  4. ~~Hook pre-commit~~ — **CERRADO 2026-09-01**, ver la entrada dedicada
     más abajo.

  ### Hook pre-commit anti-credenciales — CERRADO 2026-09-01

  Instalado en **ambos repos** (`ops/githooks/pre-commit`, copias idénticas
  verificadas con `diff`), activo vía `core.hooksPath` ya configurado en los
  dos checkouts. **Verificado corriendo, no solo escrito**: un `*_result.md`
  staged es rechazado con exit 1 en `jax` y en `jax-platform`.

  **Compara por VALOR contra secretos ya conocidos, no por patrón ni
  entropía** — porque esa es la clase que falló: `gitleaks` 8.30.1 dio cero
  hallazgos sobre el repo con la contraseña en texto plano, y `no leaks
  found` sobre el archivo denunciado servido como texto plano a
  `gitleaks dir`. La contraseña real tiene 8 caracteres: no hay formato ni
  entropía que disparar.

  **HMAC-SHA256 con pepper fuera del repo** (`/etc/jax/precommit-pepper`,
  0600). La lista viaja en un repo **público**: un `sha256` pelado de una
  contraseña de 8 caracteres se crackea con wordlist en minutos, y publicarlo
  sería publicarla de nuevo; `bcrypt`/`argon2` serían seguros de publicar
  pero cuestan ~250 ms por comparación, inusable por commit.

  **Casos probados rompiéndolo** (ejecutados por un agente, juzgados por
  otro): commit limpio pasa · la aguja en un `.py` bloquea · **un
  `curl -u user:password` en un `.md` bloquea** (el escenario exacto que
  `gitleaks` no ve) · la aguja en un test de regresión bloquea **por el
  secreto, no por la ruta** (el caso de `da9fd5ec`) · `*_result.md` bloquea
  duro incluso con la marca de escape · lista borrada o corrupta rechaza
  (fail-closed) · un secreto **nuevo** pasa, demostrado, no asumido.

  **Siete defectos que la primera batería NO probó, encontrados por el juez
  y corregidos antes de publicar** — cuatro hacían que el hook aprobara sin
  haber comparado nada, que es justo lo que dice combatir:
  1. **`git` que falla aprobaba en silencio** (`check=False`, `returncode`
     ignorado). Ahora rechaza.
  2. **Sin canario**, rotar el pepper dejaba el hook comparando contra nada
     con apariencia de sano. Ahora la lista trae `HMAC(pepper, CANARY)` y se
     verifica al cargar.
  3. **`.gitattributes` con `-diff`** apagaba la revisión de una clase entera
     con una línea inocente — más barato que `--no-verify` y sin rastro.
  4. **Binarios** no producían líneas `+`: la clase del `.pyc` que originó
     todo esto. Ahora se escanean enteros desde el índice.
  5. Líneas de contenido que empiezan con `++` se descartaban como cabecera.
  6. `typechange` (`T`) quedaba fuera del `--diff-filter`.
  7. `anadir-secreto.py` aceptaba valores que el tokenizador nunca produciría
     (cortos, o con separadores — todo base64 con padding `=`), escribiendo
     **entradas muertas que parecían cobertura**. Ahora las rechaza.

  **EL LÍMITE, escrito para que nadie lo confunda con una defensa completa:
  este hook NO habría prevenido el incidente original.** La contraseña era
  nueva y ninguna lista la contenía. Previene la **REINTRODUCCIÓN**, que es
  exactamente lo que pasó en `da9fd5ec`: el commit que sacó la contraseña de
  `seed.py` la reescribió en el test de regresión que probaba que ya no se
  usaba. Una defensa contra secretos **nuevos** necesita otra cosa —
  inyección por env var sin literales en tests, o revisión obligatoria de
  rutas de alto riesgo.

  **Deuda que deja abierta, medida y no disimulada:** `merge`, `rebase`,
  `cherry-pick` y `revert` **no ejecutan hooks de pre-commit**, y el
  `pre-push` solo mira el ref destino, no el contenido. **Hoy no existe
  barrera de contenido en el camino rama → master.** Un commit hecho con
  `--no-verify` o anterior a la activación entra por merge sin pasar por acá.

  El diseño completo vive en `ops/githooks/README.md` — **no en este
  documento y no en un mensaje de chat**, que fue donde estuvo hasta hoy.

  **Por qué este ítem existe y es el primero de la lista:** el incidente
  estuvo abierto **sin estar registrado en ningún archivo de ninguno de los
  dos repos**. Una sesión nueva que reconstruyera el estado leyendo
  `DEUDA.md` — que es exactamente para lo que `DEUDA.md` existe — no se
  enteraba de que había un P0 de seguridad abierto.

- **`_HTTP_FACETS` sin gobernanza del Motor Registry — CERRADO Y DESPLEGADO
  (2026-08-27).** Mergeado (`jax` PR#39 → merge `abe1931`; `jax-platform`
  PR#16 → merge `766e03b`) y desplegado el mismo día.

  **Verificado corriendo, no solo mergeado** — la distinción importa, ver la
  lección de método en CONTEXT.md ("el código mergeado no es código
  corriendo"). Evidencia, toda del 2026-08-27 post-reinicio:
  1. `las_manos/motor_registry/facet_policy.py` existe en el checkout que
     sirve el servicio (`/home/fruiz/jax`, NO el worktree); el
     `migrations.py` desplegado de `jax-platform` ya contiene el ALTER y el
     seed de `facet.allowed_callers` (6 ocurrencias) — el drift de esquema
     quedó cerrado en la máquina, no en el papel.
  2. `jax-las-manos` reiniciado PRIMERO, `jax-platform` DESPUÉS (ese orden,
     por el riesgo descrito abajo). Ambos `active`, arranque sin excepciones.
  3. Endpoint probado en vivo, incluidos los casos negativos:
     `{"caller":"jax_platform_chat","facet":"hipatia"}` → `allowed:true`;
     `caller_fantasma` → `allowed:false` ("no autorizado");
     `facet:"jax_local"` (allowed_callers NULL) → `allowed:false`
     ("no configurado -- fail-closed"). **El gate gatea de verdad**, no solo
     deja pasar al legítimo.
  4. Chat real end-to-end con el enforcement encendido: **los 4 facets
     responden** (`hipatia`, `jekyll`, `thot`, `ada` — verificado
     2026-08-27 tras cerrar el bug de `thot`, ver su entrada más abajo).
     En la primera verificación `thot` devolvía 502 por una causa AJENA a
     esta ronda; el gate SÍ lo autorizaba
     (`check_facet_admission('jax_platform_chat','thot')` → `(True, 'OK')`
     verificado directo, con la credencial de openai resuelta DESPUÉS del
     gate en el log): la falla era aguas abajo, en la llamada al proveedor,
     y se cerró aparte.

  **Qué falta para que esté vigente (Task 9 del plan
  `docs/superpowers/plans/2026-08-27-http-facets-motor-registry-governance.md`),
  ya ejecutado — se deja registrado porque es el mismo orden que aplica a
  cualquier redeploy o rollback futuro de estos dos servicios:**
  1. Mergear ambas ramas (`jax` y `jax-platform`).
  2. Confirmar que `facet.allowed_callers` está poblado en `jax_memory`.
  3. Reiniciar **`jax-las-manos` PRIMERO**.
  4. Reiniciar **`jax-platform` DESPUÉS**.
  5. Verificar en el chat real de Mesa web que los 4 facets responden.

  **Por qué ese orden y no el inverso — riesgo operativo real:** si se
  reinicia `jax-platform` primero, `chat.py` empieza a llamar a
  `POST /motor/authorize-facet` contra un `las_manos` que todavía no tiene
  ese endpoint → 404 → el `except` fail-closed deniega → **los 4 facets
  quedan caídos en Mesa web durante toda la ventana**, y el usuario ve un
  mensaje de "acceso no autorizado" para lo que en realidad es una caída.
  Al revés no pasa nada malo: `las_manos` con el endpoint nuevo y
  `jax-platform` viejo simplemente no lo llama — el estado es el de hoy
  (sin gobernar), nunca peor. Ningún orden deja los facets MÁS
  desgobernados que hoy; el inverso solo causa la caída.

  **Rollback:** invierte el orden (`jax-platform` PRIMERO, después
  `jax-las-manos`) por la misma razón — rollbackear `las_manos` antes deja
  a `jax-platform` llamando un endpoint que ya no existe, y recrea la
  misma caída de los 4 facets.

- **Señal de facets caídos — CERRADA Y CORRIENDO (2026-08-27), con lo que
  NO cubre declarado abajo.** Era el ítem operativo más importante de la
  ronda anterior: `thot` estuvo **3 días** caído en la Mesa web y sólo se
  descubrió porque un deploy de otra cosa incluía un paso manual de
  verificación en el chat real.

  **Qué quedó corriendo**, verificado contra producción (no contra el
  plan): escritor instrumentado en `_invoke_facet` (el chokepoint único de
  salida de Mesa web); sonda horaria con guard anti-pytest y kill switch;
  sonda por rebinding colgada de los **dos** escritores de `facet_binding`,
  que baja la detección de días a minutos; lector y máquina de estados en
  `jacobs/facet_health.py`, evaluado en **cada** barrido del reaper (≤300 s);
  alerta por Telegram con supresión de 6 h. Tablas `facet_health_event`
  (fuente de verdad) y `facet_health_alert` (ledger de acuses, no una
  segunda fuente de verdad).

  **Los seis estados, materializados con datos reales el 2026-08-27** —
  ninguno razonado, todos producidos rompiendo algo a propósito y revertido
  con el valor anotado en disco y confirmado por `SELECT` posterior:

  | outcome | facet | cómo se produjo |
  |---|---|---|
  | `ok` | ada, hipatia, jax_local, jekyll, thot | tráfico real |
  | `unsupported_transport` | **kimi** | por diseño (ver abajo) |
  | `gate_denied` | ada | `allowed_callers` vaciado |
  | `gate_unreachable` | ada | `jax-las-manos` parado |
  | `config_error` | ada | `max_tokens_param` en NULL |
  | `probe_error` | ada | misma causa, otra capa |

  `gate_denied` y `gate_unreachable` salieron **distintos** — era el
  objetivo central del ítem (los estados 3 y 4 tenían dueños y acciones
  distintas y estaban colapsados).

  **El ciclo completo, no sólo la detección:** `ada` cayó, el detector la
  marcó `down`, se reparó, y el reaper registró la **recuperación
  automáticamente** a las `18:06:28` sin intervención. Un detector que
  avisa cuando algo se rompe pero no cuando se arregla obliga a mirar a
  mano para saber si sigue roto — eso quedó cubierto.

  **Criterio de aceptación del §5 del spec, cumplido:** `kimi` aparece
  caído (`unsupported_transport` en la tabla, `down` en el ledger). Era la
  prueba de que la v1 no tiene el hueco, no una observación de color.

  **QUÉ NO CUBRE ESTA v1** — declarado, porque una alerta que se cree más
  completa de lo que es sería una instancia del patrón dentro del ítem que
  existe para detectar el patrón:
  1. **Jacobs no queda cubierto.** `_dispatch_step` no se instrumentó. Un
     facet puede estar sano para Mesa web y fallar en un pipeline por su
     propio camino de admisión (`check_capability_admission`).
  2. **`kimi` y `hyde` no son sondeables de verdad** (transportes
     `motor_registry` y `subprocess`, que `_invoke_facet` no despacha).
     `kimi` sí se sondea y reporta `unsupported_transport`; `hyde` queda
     fuera del conjunto.
  3. **La sonda no prueba lo que prueba un usuario real:** prompt corto,
     sin historial, sin contexto semántico, sin tool use. Un facet que
     falla sólo con contexto largo pasa verde.
  4. **No detecta degradación de calidad**, sólo disponibilidad. Un facet
     que responde basura cuenta `ok`.
  5. **La salud se calcula sobre datos que pueden estar incompletos.** Si
     el escritor pierde filas en silencio, la salud miente. Mitigado
     parcialmente por `unknown` (la pérdida TOTAL se ve), no por pérdida
     parcial.
  6. **Un facet nuevo en el picker sin sondear** sólo queda cubierto por
     `unknown` si aparece en `config["personalities"]`. Agregado al
     frontend y no ahí, es invisible — la misma duplicación `FACET_ORDER`
     vs `personalities` que ya existe.

  Además, **la caída total del propio detector sólo se ve en el journal**:
  el `except` del reaper es fail-soft y sólo deja un `logger.error` cada
  300 s. Mitigado por el gate de esquema completo del deploy, no eliminado.

  **El riesgo residual cambió de forma el 2026-08-28 (`jax` PR#63), no
  desapareció.** La rama que sí avisa cuando el detector deja de producir
  datos — estado `unknown` → alerta agregada bajo `__system__` — pasó a
  tener cobertura automática: `jacobs/_facet_health_io_test.py`, 6 tests
  contra una MariaDB real (`jax_memory_test`) en el job de CI
  `facet-health-io`. Era la única rama del detector **nunca ejercitada**:
  `facet_health_alert` jamás tuvo una fila `__system__`, porque producirla
  en producción exige apagar la sonda dos horas. Y `check_facet_health()`
  era la única pieza del lector sin ningún test — la lógica pura podía
  correr impecablemente sobre datos que nunca llegaron.
  Las tres propiedades se fijan por separado (vencidos → `unknown` y nunca
  `ok`; la alerta va bajo `__system__` y **no** es lista vacía; la
  supresión de 6 h se respeta **y se levanta**), más la tabla vacía — el
  agujero del `if current and all(...)`, donde `{}` es falsy y el detector
  muerto produce silencio — y un contrapositivo, sin el cual un
  `check_facet_health()` que devolviera siempre `__system__` pasaría todo
  lo demás. Verificado rompiendo el job REAL: rojo con
  `assert [] == ['__system__']`, revertido, verde de nuevo.

  **QUÉ SIGUE SIN CUBRIR, y es lo que queda del riesgo original:** si
  `jax-platform` entero está caído, nadie escribe eventos; el lector vive
  en `jax-las-manos` y sí alertaría `__system__` — pero si el que se cae es
  `jax-las-manos`, **no queda nadie que alerte**. El detector no se vigila
  a sí mismo desde afuera, y eso no lo arregla ningún test: necesita un
  observador externo al par de servicios. Sigue abierto, ahora con el
  límite dicho con precisión en vez de como una frase general.

  **`kimi` va a alertar 4 veces por día, indefinidamente, y se deja así
  (decisión del 2026-08-27).** Es presión intencional hacia la decisión de
  producto pendiente (rutearlo por Motor Registry desde Mesa web, o sacarlo
  del picker). Silenciarlo sería el error exacto que esta ronda existe para
  no cometer: un facet roto que la herramienta decide no reportar porque
  lleva mucho roto. **Fecha de control: si al 2026-09-10 `kimi` sigue
  alertando, la presión no funcionó** y hay que decidir de otra forma —
  no bajarle el volumen a la alerta.

  **CERRADO 2026-09-11** — ni rutearla por Motor Registry ni sacarla del
  picker: la decisión de producto fue que kimi funcione en el chat, y la causa
  resultó ser una etiqueta sin lector. Ver "Cerrado — kimi y la memoria vector
  cero". La alerta hizo exactamente su trabajo: la mantuvo visible un mes y
  registró sola la recuperación.

- **La alerta afirma la capa equivocada: `probe_error` tapa a
  `config_error` (2026-08-27) — CERRADO Y DESPLEGADO (2026-08-28).**
  Se deja el diagnóstico completo abajo, sin borrar: describe una clase de
  defecto (dos capas escriben el mismo fallo y el lector elige la
  equivocada) que puede reaparecer en otro lector del ledger.

  **Cerrado por** `jax-platform` `74ce495` (guard de `_invoke_facet`,
  Task 1/2 — la alerta nombra la causa, no la capa) + `48da8e8`
  (`probe_facet` deja de escribir `probe_error`, elimina la doble
  escritura en origen). Ambos en `master`.

  **Verificado corriendo, no solo mergeado:** tras el deploy, un rebinding
  real produce **UNA sola fila clasificada, no dos** — desaparece la carrera
  de ~800 µs por la que `probe_error` le ganaba a `config_error` en el
  `MAX(ts)`. La solución tomada fue la segunda de las tres opciones que
  este ítem dejaba planteadas (que la sonda no registre su `probe_error`
  cuando la capa de abajo ya clasificó el mismo fallo), no la de
  precedencia por outcome.

  **El costo, con precisión:** el mensaje que le llega a Fernando dice
  **"la sonda falló"** cuando la causa accionable es otra — por ejemplo
  "la fila de `model` no declara `max_tokens_param`", que trae hasta el
  `UPDATE` a ejecutar. Es una alerta que **afirma la capa equivocada en el
  punto exacto donde alguien la lee para decidir qué hacer**. No es un
  detalle cosmético del registro: es el único texto que un humano ve, y
  manda a investigar la sonda en vez de la fila del catálogo.

  **Evidencia real, del deploy de la Task 8** (no razonada): al poner
  `model.max_tokens_param` en NULL para `ada` — el mismo defecto que rompió
  `thot` el 2026-08-24 — quedaron dos filas separadas por ~800 µs:

  ```
  ada  probe_error   canary_rebind  ModelDispatchConfigError: ...  18:02:45.443634
  ada  config_error  canary_rebind  ModelDispatchConfigError: ...  18:02:45.442861
  ```

  Correcto por capas, y no es bug de ninguna de las dos: `_invoke_facet`
  clasifica `config_error` y **re-lanza** (no puede volverse fail-open);
  `probe_after_rebind` lo captura y registra `probe_error`, que es la
  verdad de *su* capa. El defecto está en el **lector**: toma `MAX(ts)` por
  facet, y `probe_error` gana por microsegundos. La distinción que la Task
  3.5 construyó a propósito (`config_error` separado de `provider_error`,
  porque el problema es NUESTRO y no del proveedor) se pierde en el último
  paso.

  **Por qué BLOQUEA y no queda anotado:** el ítem entero existe para que
  una alerta diga qué se rompió. Una que nombra la capa equivocada
  reintroduce, en el consumidor, el problema que el detector vino a
  resolver — igual que `thot`, alguien va a mirar el lugar equivocado.

  **Qué haría falta (no diseñado — es diseño, no un typo):** precedencia
  por outcome en vez de por `ts`; o que la sonda no registre su
  `probe_error` cuando la capa de abajo ya escribió un evento clasificado
  para el mismo facet en la misma operación; o incluir el `detail` en el
  texto de la alerta. Elegir exige decidir qué significa el ledger cuando
  dos capas describen el mismo fallo.

- **`thot` caído en la Mesa web — CERRADO 2026-08-27.** `_call_openai_compat`
  mandaba `max_tokens` con un valor fijo; `gpt-5.6-terra` rechazaba primero
  el NOMBRE del parámetro y después el VALOR. Encontrado durante la
  verificación en vivo del despliegue de gobernanza de `_HTTP_FACETS`, **no
  causado por ella** — probado, no supuesto:
  - Error real de la API: `HTTP 400 ... "Unsupported parameter: 'max_tokens'
    is not supported with this model. Use 'max_completion_tokens' instead."`
    (el backend lo propaga como 502 al cliente).
  - `backend/api/chat.py:559` manda `"max_tokens": 131072` fijo. Esa línea
    entró el **2026-08-18** (commit `f8bd8c9`, "manda max_tokens explícito"),
    nueve días antes de la ronda de gobernanza, y **ninguno** de los 3
    commits de esa rama la tocó (verificado: `git show <sha> -- chat.py |
    grep -c max_tokens` → 0 en `b017fbf`, `5a3f6c4` y `bd62db6`).
  - El gate nuevo NO es la causa: `check_facet_admission('jax_platform_chat',
    'thot')` devuelve `(True, "OK")`, y el log muestra
    `credential_resolution provider=openai` DESPUÉS del gate — o sea que
    autorizó y falló aguas abajo, en la llamada al proveedor.
  - Fecha probable de rotura: **2026-08-24 11:08**, cuando `thot` se rebindeó
    a `gpt-5.6-terra` (`facet_binding.approved_at`). El modelo anterior
    aceptaba `max_tokens`; el nuevo exige `max_completion_tokens`. Es decir,
    `thot` llevaba 3 días roto en la Mesa web sin que nadie lo notara —
    dato que vale por sí solo: **no hay alerta que avise cuando un facet deja
    de responder.**
  - **NOMBRE del parámetro: ARREGLADO y desplegado (2026-08-27, PR
    jax-platform#17, merge `6800a32`).** Columna nueva `model.max_tokens_param`
    (ENUM): el catálogo declara qué parámetro exige cada modelo y
    `_call_openai_compat` lo lee vía el JOIN que `facet_resolver` ya hacía.
    NULL falla RUIDOSO (log `ERROR` con el `UPDATE` exacto a correr), no
    asume — si el default fuera el parámetro viejo, el próximo modelo nuevo
    se rompería igual pero en silencio. Sembrados los 4 modelos que usan el
    camino openai-compat (`gpt-5.6-terra` → nuevo; `deepseek-v4-flash`,
    `glm-5.3`, `kimi-k3` → viejo); sembrar solo el de `thot` habría tumbado
    `jekyll` y `ada`. Verificado en prod post-deploy por SELECT.
  - **VALOR del tope: ARREGLADO y desplegado (2026-08-27, PR
    jax-platform#18, merge `35105ae`).** Arreglado el nombre, apareció el
    valor: `HTTP 400: "max_tokens is too large: 131072. This model supports
    at most 128000 completion tokens"`. El `131072` también estaba
    hardcodeado y también es propiedad por modelo. **`context_window` NO
    servía para derivarlo:** `gpt-5.6-terra` tiene `context_window=1050000`
    (ventana total) contra un tope de *completion* de 128000 — hechos
    distintos, y el segundo no estaba en el catálogo. Columna hermana
    `model.max_output_tokens` (INT), mismo contrato. Se ELIMINÓ la constante
    `_MAX_OUTPUT_TOKENS` en vez de dejarla como fallback: dejarla era el
    default silencioso que convertiría al próximo modelo nuevo en otro
    `thot`, pero mudo. Validación de tipo además de NULL (rechaza `0`,
    negativos, no-int, `bool`) — el ENUM protege a `max_tokens_param`, un
    `INT` no protege contra un `0` de un backfill.
  - **CERRADO — verificado con `thot` RESPONDIENDO, no con el código
    mergeado (2026-08-27).** Los 4 facets en el chat real de la Mesa web
    post-deploy: `hipatia` OK, `jekyll` OK, **`thot` OK**, `ada` OK.
    El ítem no se dio por cerrado hasta ese output, a propósito: el primer
    despliegue (PR #17) había MOVIDO el error sin arreglar el facet, y solo
    la verificación en el chat lo dijo. Es la regla de CONTEXT.md ("el
    código mergeado no es código corriendo") aplicada a un fix.
  - **Lección: el primer despliegue MOVIÓ el error, no arregló el facet.**
    Se dio por "arreglado" hasta que la verificación en el chat real dijo lo
    contrario — la misma regla de CONTEXT.md ("el código mergeado no es
    código corriendo") aplicada a un fix, no a un cierre.

  ---

  **Qué se construyó.** Jacobs y Mesa web quedan gobernados DE FORMA
  DISTINTA — no es el mismo check aplicado dos veces, son dos mecanismos
  separados:

  **Jacobs (`jax`, commits `dbc5585`/`4f4eb6b`/`ab7d241`/`ba8234d`):**
  `MotorPolicy.check()` (`las_manos/motor_registry/policy.py`) se partió en
  `check_capability_admission()` (checks 1-5: capability existe, caller
  autorizado, human gate, recursion depth, claves prohibidas) más un
  wrapper que preserva `check()` exactamente igual para `kimi`/`jax_local`
  — cero cambio de comportamiento, suite existente sin modificar y sin
  fallar. `jacobs/executor.py::validate_capability()` gana un bloque NIVEL C
  (`jacobs/executor.py:646-670`) que llama a `check_capability_admission()`
  para los 4 facets de `_HTTP_FACETS` únicamente. Fail-closed real, no solo
  de nombre: una falla de DB propaga en vez de tragarse silenciosamente
  (probado con un mock de fallo de DB contra `hipatia`/`research`,
  `jacobs/_http_facet_admission_test.py::HttpFacetAdmissionFailClosedTest::test_db_caida_al_leer_catalogo_no_deja_pasar_el_step`).
  Checks 6-7 (resolver motor, `motor.sandbox_only`) son N/A para un facet
  HTTP — no hay motor que resolver. Check 8 (techo de timeout) sigue SIN
  activar para este camino — ver la entrada de `_CAPABILITY_TIMEOUT_SECONDS`
  más abajo, con la razón estructural.

  **Mesa web (`jax-platform`, misma rama, commits `849956b`/`b017fbf`/
  `5a3f6c4`):** NO usa `MotorPolicy` ni la tabla `capability` en absoluto.
  Un turno de chat es texto libre enrutado a un facet por keyword-matching,
  sin ningún mapeo facet→capability real (verificado leyendo `chat.py`
  completo antes de diseñar esto — una versión anterior del diseño asumía
  ese mapeo y era falsa, corregida antes de implementar). Se construyó un
  check nuevo y más chico, `check_facet_admission()`
  (`las_manos/motor_registry/facet_policy.py`, repo `jax`, corre
  server-side dentro de `las_manos`), sobre una columna nueva
  `facet.allowed_callers` (NULLABLE; migración idempotente en
  `jax-platform`, commit `849956b`, guardada con `WHERE ... IS NULL` para
  no pisar un valor manual futuro), expuesto como `POST
  /motor/authorize-facet` (`jax`) y llamado desde `_invoke_facet` en
  `backend/api/chat.py` (`jax-platform`) antes de despachar. Fail-closed
  confirmado con un caso real, no solo un mock prolijo: `las_manos` caído
  de verdad (conexión rechazada, no un mock con error prolijo) deniega
  igual que una respuesta explícita `allowed=False`, con logging que
  distingue "no se pudo verificar" de "denegado de verdad". Los checks 2-8
  de `MotorPolicy` son N/A para este camino POR DISEÑO — están atados a la
  tabla `capability`, que Mesa web no consulta para este check — no es que
  "quedaron pendientes".

  **El gate de Mesa web se llavea por TRANSPORTE, no por nombre de facet.**
  Corregido en la revisión final del branch: era un frozenset de nombres
  (`{"hipatia","jekyll","thot","ada"}`) al lado de un dispatch que rutea
  por `facet.transport` — dos fuentes de verdad que divergen a la primera
  fila nueva. Hoy gatea sobre `f.transport in ("http_gemini",
  "http_openai_compat")`. Verificado por SELECT contra `jax_memory`: esos
  transportes cubren exactamente `ada`/`hipatia`/`jekyll`/`thot` y nada
  más, así que el cambio es preservador de comportamiento hoy y
  fail-closed para cualquier facet HTTP futuro.

  **`facet.allowed_callers` NO gobierna a Jacobs — trampa de modelo mental
  documentada, no cerrada.** La columna contiene `"jacobs"`, pero nada
  consulta ese valor para Jacobs: Jacobs se gobierna por
  `capability.allowed_callers` vía `check_capability_admission()`. Un
  operador que quisiera cortarle el acceso a `hipatia` editaría
  `facet.allowed_callers`, sacaría `"jacobs"`, y Jacobs seguiría
  despachando igual, sin error ni aviso. No se cambió el dato sembrado (los
  tests dependen de él y reseedearlo cascadea); se documentó en los dos
  lugares donde alguien lo leería: el DDL de `facet` en
  `jax-platform/backend/db/migrations.py` y el docstring de
  `check_facet_admission()`. **Follow-up candidato:** hacer que Jacobs
  también consulte `check_facet_admission()`, para que la columna pase a
  ser el gate real de nivel facet para AMBOS caminos y deje de enseñar un
  modelo equivocado.

  **Lo que sigue explícitamente SIN gobernar, para no leerse como cobertura
  total:**

  - **El REPL interactivo `jax` — TERCER camino de dispatch, fuera de
    alcance por decisión de esta ronda.** No toca ninguno de los dos
    mecanismos. Despacha en `jax/core/main.py:400` (modo tarea, invoke en
    `:422`) y `jax/core/main.py:680` (bucle interactivo, invoke en `:738`)
    vía `muscles[faceta].invoke(...)`, sobre el dict que arma
    `build_muscles()` (`jax/core/main.py:64-95`) desde
    `config/config.toml`, donde `jekyll`/`hipatia`/`thot`/`ada` son todos
    `type = "http"` (`config/config.toml:114`, `:158`, `:211`, `:273`;
    secciones `[personalities.*]` en `:113`, `:157`, `:210`, `:272`).
    Verificado por grep: cero ocurrencias de `MotorPolicy`,
    `check_capability_admission`, `check_facet_admission`,
    `authorize-facet` o `allowed_callers` en todo `jax/`. **Además
    despacha `kimi` como `type = "http"` (`config/config.toml:323`,
    sección en `:322`), o sea que el REPL saltea el Motor Registry incluso
    para un facet-motor** — hueco preexistente de la historia original de
    los 8 checks, NO creado por esta rama. Razonamiento operativo para
    dejarlo afuera, para que un lector futuro sepa que se consideró y no
    que se pasó por alto: es una herramienta local e interactiva, el caller
    es el dueño sentado en una terminal, y no cruza ninguna frontera de
    privilegio — el gate protegería al operador de sí mismo. Si el REPL
    algún día se expone a otro caller (script, servicio, sesión remota
    compartida), deja de ser cierto y hay que gobernarlo.

  - **`_HTTP_FACETS` en el repo `jax` sigue llaveado por NOMBRE, no por
    transporte** (`jacobs/models.py:22`, consumido en
    `jacobs/executor.py:659`). Es el mismo defecto estructural que se
    corrigió del lado de Mesa web: agregar un facet HTTP nuevo lo dejaría
    fuera del bloque NIVEL C. No se tocó esta ronda porque el
    restructure del lado `jax` es más invasivo (`_HTTP_FACETS` se usa
    también para ruteo de dispatch, no solo para el gate). Deuda
    registrada, no resuelta.

  - **El techo de `max_execution_minutes`/timeout (check 8)** no se activó
    para ningún camino — ver `_CAPABILITY_TIMEOUT_SECONDS` más abajo.

  - **`capability.sandbox_only`** sigue vestigial — ver la entrada de abajo.

  - **`human_gate_token` para Jacobs pasa hoy, pero el invariante no está
    garantizado en ninguna parte.** Corregido en la revisión final: la
    versión anterior de esta entrada decía "ninguna de las 5 capabilities
    relevantes lo requiere", y el número y el encuadre estaban mal.
    Verificado por SELECT contra `jax_memory` 2026-08-27: los steps
    históricos con facet HTTP en `jacobs_steps` usaron **9 capabilities
    distintas**, no 5 — `ada`: analysis/assemble/design/generate/reconcile;
    `hipatia`: research; `jekyll`: analysis; `thot`:
    critique/review/validate_consistency. De esas 9, `assemble` ni siquiera
    es fila de `capability` (es mecánica: se cortocircuita en
    `jacobs/executor.py:635` y `:690`, nunca llega a admisión); las otras 8
    sí existen y todas tienen `requires_human_gate=0`. Pero el encuadre
    correcto NO es "ninguna capability relevante exige gate": es **"ninguna
    capability OBSERVADA históricamente exige gate, y nada obliga a que
    siga siendo así"**. `bug_hunt` y `code_swarm` SÍ tienen
    `requires_human_gate=1` y SÍ listan `"jacobs"` en `allowed_callers`
    (verificado en `jax_memory` y `jax_memory_test`), y nada impide que el
    planner se las asigne a un facet HTTP:
    `_validate_plan_capabilities` (`jacobs/plan.py:294`, filtro en `:308`)
    solo inspecciona `MOTOR_FACETS`, y el cierre de vocabulario del planner
    (`jacobs/plan.py:639`) las deja pasar porque existen en la tabla.
    Consecuencia real del código que se shippeó: un step así **falla duro**
    — NIVEL C pasa `human_gate_token=None` fijo
    (`jacobs/executor.py:667`) y la denegación vuelve como `str`, no como
    `CapabilityUnbound`, así que `_dispatch_step` no reintenta ni reenruta.
    Es el comportamiento correcto (fail-closed), pero es una falla sin
    reintento y sin mecanismo hoy de conseguir un token real. Cubierto por
    test:
    `jacobs/_http_facet_admission_test.py::HttpFacetAdmissionTest::test_capability_con_human_gate_es_denegada`.

- **`workspace/` sin repo git propio, `file_write` sin commitear — CERRADO 2026-08-21.** Hallazgo original: byproducto de la verificación T4 de Bloque 3 (no buscado a propósito). Diagnóstico completo mostró que **se perdió DOS VECES en menos de 20h**, no una:
  1. **2026-08-20 14:28 CST** — el filter-repo de ronda 9 re-clonó `/home/fruiz/jax` fresco tras el `push --force --mirror`. Se restauraron a mano `.venv`/`node_modules` (gitignored, necesarios para que los servicios arranquen) pero nadie pensó en `workspace/` — no bloqueaba el arranque, así que no entró al checklist de restauración.
  2. Alguien reinicializó `workspace/.git` a mano después de eso (evidencia: 4 `TOOL_WRITE_REVERTED` con SHA real entre las 02:50 y las 04:59 CST del 21-ago, del trabajo adversarial de Hyde/bubblewrap).
  3. **Entre las 04:59:51 y las 10:42:23 CST del 21-ago** — el directorio `workspace/` completo (no solo `.git`) volvió a desaparecer; se recreó vacío justo en el write de verificación de T4. Ventana que coincide con la ráfaga de commits de "apertura pública / limpieza mecánica" de Bloques 1-2 del mismo día. No se encontró el comando exacto (no queda en `bash_history` ni en scripts trackeados), pero la causa más probable es un `git clean -dfx` o un re-clone equivalente — es la única clase de operación de git que se lleva puesto algo gitignored.

  **Causa raíz real, no el síntoma:** mientras `workspace/` viva *dentro* del árbol de `jax/` como directorio gitignored, es invisible para git y cualquier limpieza del repo padre se lo lleva puesto sin avisar. Reinicializar el `.git` sin cambiar la ubicación habría reparado el síntoma, no la causa — la tercera pérdida era cuestión de tiempo.

  **Fix aplicado:** `workspace/` movido fuera de ambos repos, a `/home/fruiz/jax-workspace`. Único source of truth: `JAX_WORKSPACE_DIR` en `/etc/jax/.env`, leída por los 3 call sites que antes hardcodeaban el path por separado (`jacobs/executor.py::HYDE_WORKSPACE_DIR`, `las_manos/motor_registry/tool_authority.py::WORKSPACE_ROOT`, `jax/muscles/subprocess_muscle.py::workspace_dir` default — este último no estaba contemplado en el diagnóstico inicial, apareció al mapear call sites reales antes de mover nada; es el que usa el REPL interactivo `jax`, verificado funcionando después del cambio). Fallback sin env var apunta a la ubicación NUEVA, nunca a la vieja. Historia real de 19-ago (`calculadora.html`, primer `file_write` que sí versionó) restaurada desde `jax.old-pre-filter-repo-20260820/workspace/.git`. Defensa en profundidad agregada: `las_manos/server.py` loguea ERROR al arranque si `workspace/.git` no existe (no debería dispararse nunca con la ubicación nueva; si se dispara, es la alarma de que algo volvió a romper el blindaje).

  **Nota de restauración:** el `.git` de 19-ago traía además 4 archivos de negocio sensibles (`ateneaerp_market_research_final.html`, `hammurabi-credito-pipeline-001.json`, `jekyll_sintesis_bloques123.md`, `mision-research-ateneaerp.md`) que ronda 9 ya había purgado a propósito de la historia de `jax`. Se corrió `git-filter-repo --invert-paths` local sobre `jax-workspace` para excluirlos también ahí antes de dejar el repo en pie — verificado con el mismo método de ronda 9 (grep de contenido sobre todos los blobs, 0 matches) más `git fsck --full` limpio.

  **Deuda nueva que este episodio destapa, sin resolver todavía:** `jax-workspace/` nace sin política. Es donde escriben los modelos (Hyde, jax_local vía file_write). Antes de que acumule salidas reales de pipelines falta decidir: ¿se respalda (Sésamo, R2)? ¿tiene retención o crece sin límite? ¿algo impide que vuelva a juntar contenido sensible sin que nadie lo note, como pasó la primera vez?

  **Lección de método, vale más allá de este caso:** la limpieza del repo padre destruyó el mecanismo de reversibilidad (`git reset --hard`) que era la justificación para sacarle el gate humano a `write_file`. Una garantía de seguridad que depende de infraestructura frágil (un directorio gitignored dentro del árbol que protege) no es una garantía real — se cae exactamente cuando el sistema que la rodea cambia, sin que el propio mecanismo se entere.


## Anotado, no bloquea

- **Anotado con fecha 2026-09-22 — `idx_pipelines_status (status)` quedó redundante con `idx_pipelines_ocultos (status, descartado_at)` (Task 1, fix round 1, spec `2026-09-22-descartar-pipelines`).** `idx_pipelines_ocultos` empieza por la misma columna (`status`) que `idx_pipelines_status`: por la regla del prefijo izquierdo de un índice compuesto, MariaDB puede resolver con el nuevo cualquier consulta que hoy elige el viejo filtrando solo por `status`. No se retira en este PR: el viejo puede tener lectores que esta ronda no auditó (el reaper vía `store.candidatos_del_reaper`, `pipeline_count_active`, el candado del cupo), y borrarlo a ciegas es exactamente el tipo de "arreglo" que la Regla Absoluta prohíbe. Retirarlo va en su **PROPIO PR**, con: (a) `EXPLAIN` de la consulta real del reaper (y de cualquier otro caller que filtre `jacobs_pipelines` solo por `status`) contra el índice nuevo, sin filesort ni caída a scan completo; (b) un grep de todos los callers que arman `WHERE status = ...`/`WHERE status IN (...)` sobre `jacobs_pipelines` para confirmar que ninguno depende de una propiedad de `idx_pipelines_status` que `idx_pipelines_ocultos` no cubra (por ejemplo, un `FORCE INDEX`/`USE INDEX` explícito, si existiera).

- **`ejecutor_host.sudo` y `.machine_id` quedan desactualizadas y SIN LECTOR en el código —
  DECISIÓN 2026-09-22 (ronda 2 del contexto del Ejecutor, auditoría adversarial M6).**
  La migración que las llenaba (`jax/memory/migrations.py::ensure_schema()`,
  `_EJECUTOR_HOST_MACHINE_ID`, agregada en la ronda 1) se QUITÓ: ninguna pieza del código lee
  esas dos columnas (verificado con `grep` sobre el árbol — sólo el propio migrador y su test
  las tocaban), y la fuente única del sudo/machine-id que de verdad importa (lo que
  `generar_claude_md.py` pone en el CLAUDE.md de axioma) es `scripts/ejecutor_fase0/
  maquinas.toml`, que ya lo tenía. Mantener la migración corriendo en CADA `connect()` del
  memory worker/LAS MANOS/síntesis tenía dos costos sin beneficio: (1) pisaba, en cada
  arranque, cualquier corrección manual que Fernando hiciera directo en la DB (el propio
  ledger, `~/ejecutor-producto/LEDGER.md`, ya decía "se corrigen por migración en PR, no a
  mano" — pero una migración que se REPITE en cada connect es peor que una corrida una vez);
  (2) ataba la salud de la memoria (`ensure_schema()` es lo que decide si `jax_memory` está al
  día) a una tabla de otro dominio (`ejecutor_host`, de jax-platform) — ver B3 en el job
  `memory-vector-zero-io` (`.github/workflows/policy.yml`) para el patrón de acoplamiento que
  ese job ya vigila para otras columnas.
  - **Verificado en producción, 2026-09-22 (SELECT de solo lectura, puerto 3308):** las cinco
    filas de `ejecutor_host` (`atemai`, `bridge`, `ejecutor-prueba`, `hall9000`, `prod`) están
    HOY con `sudo=0` y `machine_id=NULL` — la migración de la ronda 1 nunca llegó a
    producción (esta rama no está desplegada), así que quitarla no revierte nada que
    estuviera en uso.
  - **`ejecutor-prueba` (auditoría, ítem MINOR):** no tiene línea en `maquinas.toml` (nunca
    tuvo sudo real — VM desechable de la Fase 2) y su `machine_id` en la DB es `NULL`/no
    verificado hoy; no se inventó un valor. Con esta migración retirada, no queda ningún
    artefacto de este repo donde agregarle una línea de machine-id tenga sentido.
  - **Si algún día algo SÍ necesita leer `ejecutor_host.sudo`/`.machine_id` desde jax:**
    escribir ese lector primero (Principio IX — el contrato antes que la capacidad), y recién
    ahí decidir si hace falta una migración de nuevo, en el repo que corresponda
    (jax-platform, dueño de la tabla) o acá si el lector vive en jax.

- **Anotado con fecha 2026-10-17 — revisar el tamaño del pool del store de Jacobs sólo si el uso real lo pide (Ruling R52, 2026-09-17).** `JAX_DB_POOL_MAX=10` (default derivado en `jacobs/store.py::db_pool_max`). La carga final del pre-vuelo dio el umbral 10× NO CUMPLIDO (p95 c25/c1 = 17,14×; c50/c1 = 30,39×) con 0 errores en todas las concurrencias y p95 absoluto 2,74/17,15/46,97/83,26 ms a c=1/10/25/50. La sesión principal lo aceptó por escrito: la relación mide encolamiento contra un pool de 10 con concurrencia de 25 y 50, muy por encima de la demanda real (`MAX_PARALLEL_PIPELINES=3`; el pre-vuelo lo dispara una persona desde la Mesa), y el pool se dimensionó contra una MariaDB compartida (151 conexiones, 96 en uso).
  - **Qué lo dispara (observación, no calendario):** uso real por encima de **10 pre-vuelos concurrentes sostenidos**. Dónde se ve: (a) en el journal de `jax-las-manos`, 503 `prevuelo_no_disponible` cuyo motivo es un `TimeoutError` esperando turno del pool (un pedido HTTP espera a lo sumo `JAX_DB_CONNECT_TIMEOUT_SECONDS`), que es el síntoma de cola llena; (b) en `jacobs_events` / la Mesa, `/jacobs/preflight` y `POST /jacobs/pipeline` solapados en la misma ventana de segundos por más de 10 pedidos; (c) en el perfil (`scripts/perfil_prevuelo.py trabajadores`), la fase `acquire` dominando el total con la concurrencia real medida, no con una inventada.
  - **Qué hacer si pasa:** volver a medir con la concurrencia real observada (lotes y sostenida, el procedimiento de la Task 15) y recién entonces evaluar subir el pool, contra el presupuesto de conexiones del servidor MariaDB compartido. Sin ese número medido, no se toca: subir el pool para que el ratio dé bien sería acomodar la medición al criterio.
  - **Evidencia y decisión:** `CONTEXT.md` §9 (entrada del 2026-09-17 sobre HEAD `cec13ac`), `r38-report.md` y Rulings R44/R52/R53 del ledger `2026-09-17-prevuelo-y-continuar-jacobs`.

- **Anotado — C3 del Ejecutor (2026-09-17, Mr. Hyde). Ninguno bloquea:**
  - **El carril del Ejecutor sondea cada 50 ms** (`prioridad._PASO_EJECUTOR_S`). Carga de C3 (200 peticiones con
    `tool_use`, upstream falso): a concurrencia 1 el registro agrega +2,2/+2,5 ms de p95 (cumple ≤ 10 ms); a
    concurrencia 2, en 1 de 5 rondas el p95 salta a +52,8 ms. Medido por fases: `anotar` p95 1,2 ms, `fsync` p95
    1,03 ms, upstream 0,5 ms; el salto es la espera del carril (el registro alarga la retención ~2 ms y dos peticiones
    chocan más). Frente a una inferencia de segundos no se nota; si alguna vez importa, el paso del carril es la palanca.
  - **El cerco es de red (TCP/UDP):** los sockets unix del sistema (dbus, systemd, snapd, libvirt-ro) siguen con su
    propia autorización, como para cualquier usuario sin privilegios. Ningún servicio de JAX escucha por socket unix
    alcanzable por `axioma` (docker.sock es `root:docker 660`; verificado 2026-09-17).
  - **Si alguien habilita `nftables.service`**, su `/etc/nftables.conf` hace `flush ruleset` y borra el cerco hasta el
    próximo `systemctl restart ejecutor-cerco`. Hoy está `disabled` (verificado 2026-09-17); `probar_c3.py` lo detecta
    (`cerco_abierto`).

- **Anotado — deuda residual del frente F: contrato de sub-pipelines y pool de Jacobs (2026-09-17). Dueño: próxima ronda de pago de deuda.** Ninguno bloquea; en CI cada job levanta su propia MariaDB vacía:
  - **`jax_memory_test` compartida entre frentes da flakes de tests que leen "la última fila" o limpian una tabla entera:** tras el rebase, en 16 corridas locales (13 de la suite completa, 3 de `jacobs` + `las_manos`) fallaron una sola vez cada uno `las_manos/_motor_usage_writer_test.py::test_record_motor_usage_sin_identidad_escribe_con_null_y_loguea` (lee la última fila de `axioma_usage`) y `jacobs/_facet_health_io_test.py::test_la_supresion_de_6h_se_respeta_en_la_alerta_agregada` (vacía `facet_health_event`); no se reprodujeron en las demás corridas. Misma causa que el flake de `test_tablero.py` del frente C: base de test compartida con otros frentes corriendo en paralelo. Arreglo de raíz: base de test propia por frente, o tests que filtren por su propia fila.
  - **Dos fallas locales preexistentes en master `de6964e`, idénticas en la rama** (estado de la `jax_memory_test` local, en CI la base es nueva): `las_manos/_catalog_from_db_test.py::test_from_db_carga_kimi_y_ada` (`max_tokens` 0 de la fila de kimi) y `tests/test_memory_scope_denormalized.py::test_la_busqueda_usa_el_indice_vectorial_y_no_une_con_conversations` (el plan usa `idx_msg_scope` con filesort, no `idx_embedding`, con 1-2 filas en la base).
  - **`jacobs/_pipeline_identity_test.py` en master deja una fila `test`/`Fernando` `pending` por corrida** que cuenta contra `MAX_PARALLEL_PIPELINES=3` y tapa las cargas de Jacobs con 422; la rama F la expira en `addAsyncCleanup`. Hasta el merge, correr la suite de master contra `jax_memory_test` ensucia la base (visto 2026-09-17: tres filas de corridas de master, expiradas a mano por id).

- **Anotado — deuda residual del frente E de la auditoría (jax, triage de la revisión final y del rebase, 2026-09-17). Dueño: próxima ronda de pago de deuda.** Ninguno bloquea:
  - **`facet` se lee por `status` sin índice** (`jacobs/store.py`, 4º SELECT de la validación de plan, E-17): catálogo de 7 filas, recorrido completo. El costo de 0,00024 s citado en `store.py` se midió con 3 SELECTs (2026-08-21); el 4º no está medido.
  - **El `lifespan` de jax-platform no valida `LAS_MANOS_URL`** al arrancar (sí `JAX_OLLAMA_URL`).
  - **Docstring de `jacobs/executor.py:424-426`** describe la lista fija vieja de facetas; `_CLEANROOM_RULE` nombra a kimi.
  - **`cargar_registro` y `facet_resolver` ignoran `provider.status`** (a diferencia de `url_del_proveedor`, que ya no da URL a un `deprecated`). Es espejo de dos repos: se cambia en los dos a la vez.
  - **El `conftest.py` raíz de jax no fija `JAX_FACET_SEAL_PATH`** (jax-platform sí). Hoy la lista de tests-puros no escribe el sello (verificado 2026-09-17) y `jacobs-gobernanza-db` lo fija en el workflow; una corrida local de los tests de DB sin la variable SÍ lo toca.
  - **`tests/test_memory_scope_denormalized.py::test_encuentra_el_mensaje_del_proyecto_compartido` falla en la `jax_memory_test` local**, igual en master `0da32af` y en la rama (estado de la base compartida, no regresión); en CI la base es nueva.
  - **Tripwires con límites declarados:** el de errores de proveedor redactados solo ve `X.text[:N]` y `aread()`; el de cliente HTTP compartido no ve alias de `AsyncClient`; el cierre en REPL/workers se verifica por substring.
  - **`embedding_worker` cierra el cliente fuera de `finally`; `MemoryDB.close` no cierra el pool si `_http.aclose()` lanza.**
  - **`ControlesFueraDelCheckoutTest` queda después de `if __name__ == "__main__"`** en dos archivos de tripwire (pytest los colecta; cosmético al correrlos con `python` directo).

- **Anotado — deuda residual del frente C: los ajustes de Admin mandan de verdad (2026-09-17). jax-platform; dueño: próxima ronda de pago de deuda.** Ninguno bloquea; quedan acá para no perderse (regla "sin hallazgos diferidos" no aplica a los que la revisión final marcó explícitamente como aceptados, no como pendientes de arreglo):
  - **Medida de login de la carga (R17 del ledger) no representa el costo real de bcrypt en producción:** los usuarios de la carga se crean con `bcrypt rounds=4` (fixture de test, `tests/identidades.py`), no los 12 de producción. El GO del gate de carga es válido para lo que `ajustes.py` agrega, no para el costo de login end-to-end. Repetir con `rounds=12` si alguna vez se necesita ese número (≈15 min, según el ledger).
  - **`CacheDeAjustes` indexa por la grafía guardada de la clave** — un valor de `axioma_config` con una collation distinta a la canónica (caso borde, solo alcanzable borrando a mano la fila canónica) podría no invalidar igual que la fila esperada. Parqueado por la revisión final; la mitigación real sería un test + un `lower()` explícito por collation.
  - **`test_grounding_config_revalidation` es flake por pérdida de conexión de MariaDB**, reproducido igual CON y SIN el cambio de este frente (visto en Task 2 y de nuevo tras el rebase de Task 12) — no es una regresión de este cierre, preexiste en `master`. Sin arreglar.
  - **`test_tablero.py` (frente A) tiene un flake reportado por compartir la base `jax_memory_test` con la carga de otros frentes corriendo en simultáneo** (mismo síntoma que el "ruido de contención" documentado en `carga.md` para los tests de carga de este frente) — anotado por instrucción del ledger sin una reproducción aislada propia en esta sesión; si vuelve a aparecer, aislar con una base de test propia por frente en vez de compartir `jax_memory_test`.
  - **Carrera check-then-admit en `api/pipelines.py:117-142`** (mencionada ya en la deuda residual del frente A): sigue sin lock porque el área es de `resource_manager.py`, que este frente sí tocó — no se resolvió, queda con el mismo dueño.

- **Anotado — deuda residual del frente A de la auditoría de sobre-ingeniería (triage del review final, 2026-09-17). jax-platform; dueño: próxima ronda de pago de deuda.** Ninguno bloquea; el review final de la rama los marcó como minor y la regla "sin hallazgos diferidos" exige que queden acá, con archivo y motivo, en vez de perderse:
  - **`api/pipelines.py:117-142`, carrera check-then-admit** entre `can_start_pipeline` y `admit_pipeline` (awaits en el medio; el lock que se quitó en A-24 tampoco la cubría). Es del área del frente C (`resource_manager.py`).
  - **`jax_engine/state.py:29`, `LAS_MANOS_URL` con default literal** `http://127.0.0.1:7777`; la tarjeta de LAS MANOS del tablero nunca dice `sin_configurar`.
  - **`api/admin/dashboard.py`, sondas de servicios secuenciales** (peor caso ~6 s); falta `asyncio.gather` para paralelizarlas (preexistente a la ronda).
  - **`api/command.py`, `motivo` expone la ruta absoluta de `JAX_BIN` y la URL interna de Jacobs** en el texto de error (preexistente; la redacción de secretos no cubre rutas/URLs).
  - **`api/command.py`, `.tmp` huérfano del owner file** si el proceso muere entre el `write` y el `os.replace`.
  - **`api/command.py`, `publish`/`set_facet_status` sin guarda dentro del `except` del fallo sin persistir** (si publicar falla tras un fallo de ejecución, hyde puede quedar en `thinking`; preexistente a la ronda).
  - **`backend/tests/test_mesa_codigos.py`, el guard AST es laxo:** acepta cualquier `ast.Name` como `detail` y no ve `fastapi.HTTPException(...)` construido inline.
  - **`Login.test.jsx`, el `grep /minuto/`** es un instrumento romo que no falla contra el código viejo (los otros 3 tests de conducta de A-50 sí).
  - **El `motivo` de un error se renderiza como Markdown** en el chat — una imagen remota es posible desde ahí (preexistente).
  - **`frontend/src/politica/modales.test.js:15`** solo detecta el literal `"fixed inset-0"`, no `"inset-0 fixed"` ni `position: fixed` inline.
  - **Escape no cierra `rotate` (`AdminFacetsModels.jsx`) ni `PipelineModal` mientras el pedido está en vuelo** (preexistente; SMTP sí bloquea Escape durante el pedido).
  - **El conteo de usuarios activos del tablero recorre todo `idx_jax_users_role_status`** (index scan completo, no range) — fuera del alcance de la ronda de arreglos de carga (R12).
  - **`api/admin/repository.py:169-170`, el semáforo de lectura se libera al cancelar el pedido con el hilo todavía leyendo** (ventana de concurrencia >1 breve).
  - **`api/audit.py`, un byte UTF-8 inválido fuera de la cola pedida (o una línea sin `\n` final / con `\r` suelto) ya no da 503**, desde que `_ultimas_20_lineas` lee desde el final del archivo (por diseño: ya no se lee el resto del archivo).
  - **Warning de `httpx`/`starlette.testclient`** preexistente en la suite sin DB (solo ruido en el log de CI, sin efecto en resultados).
  - **`api/admin/repository.py`, `delete_file` sobre un symlink borra el destino** (no el link) dentro de la misma carpeta permitida — conducta preexistente, no introducida por A-40; el listado ahora la muestra con su propia ruta.
  - **`backend/main.py`, el comentario de `facet_models_router` desregistrado queda pegado arriba de `models_router`** y puede leerse como anotación de ese router — la ronda de arreglo despachada durante la ejecución no llegó a commit (ver el ledger, nota del controlador de cierre).
  - **Coordinación pendiente con el frente D:** los códigos nuevos `archivo_demasiado_grande`/`pdf_ilegible` ya se traducen en `errores.js`, pero `BottomBar.jsx:90-91` (bloque de adjuntos, del frente D) sigue mostrando `t.attachError` genérico — se resuelve cuando el frente D mergee sobre esta rama.

- **Anotados en la etapa 1 de admin usuarios (2026-09-13).**
  - **Dependencia de la etapa 4 — CERRADO 2026-09-15 (jax-platform#82 → `453b128`):**
    `POST /users/{id}/reset-link` reusa el núcleo (`_crear_enlace_de_recuperacion` +
    `_send_reset_email`) sin el envoltorio fail-soft. Da 503 si SMTP no está configurado o está
    dañado, 502 con la respuesta del servidor si el envío falla, y no deja vivo ningún token no
    entregado. Texto original: `_procesar_recuperacion` se traga todo y
    devuelve `None`, así que el "enlace de restablecimiento" por admin de la
    etapa 4 no puede reusarlo para devolver 503 si falta SMTP (spec §3.4). La
    etapa 4 tiene que llamar `smtp_config.cargar_settings()` y
    `_send_reset_email` (que SÍ lanza) por su cuenta.
  - **`DeprecationWarning` de `datetime.utcnow()` — CERRADO Y DESPLEGADO 2026-09-14**
    (jax#150 → `a996c07`; jax-platform#69 → `bfab4de`; `jax-platform` 12:38:07 y `jax-las-manos` 12:38:09, cwd = checkout con el commit, NRestarts=0, journal limpio; frontend `index-BK3OWw2W.js`, md5 local = servido). Eran 18 llamadas (16 de código, 2 de tests), no solo las de `jax_engine`.
    `tiempo.utc_ahora()` devuelve UTC SIN zona: una hora con zona compararía contra
    `expires_at`/`locked_until` (DATETIME sin zona) y lanzaría TypeError. Guard por AST
    contra `utcnow`/`utcfromtimestamp`. Suite con DB: 417 → 159 avisos, 0 de utcnow.
    En vivo: login 401 en 0,197 s, forgot-password 200, reset con token inventado 400.
    Carga del login (2.400 peticiones, `scripts/load_test.py`): 0 errores, p95 2,69 ms
    a c=10 y 13,86 ms a c=50. Texto original: ruido preexistente en `jax_engine`.
  - **test-connection devuelve el banner del servidor — ACEPTADO, reconfirmado por
    Fernando 2026-09-14** remoto al superadmin:
    sirve como sondeo de puertos internos. Se acepta porque es solo superadmin y
    está limitado (`JAX_SMTP_CONN_RATE`). Lo reabre que el endpoint deje de ser
    exclusivo de superadmin.
  - **La pantalla de Configuración traga los errores del PUT `/api/admin/config` —
    CERRADO Y DESPLEGADO 2026-09-14** (jax#150 → `a996c07`; jax-platform#69 → `bfab4de`; `jax-platform` 12:38:07 y `jax-las-manos` 12:38:09, cwd = checkout con el commit, NRestarts=0, journal limpio; frontend `index-BK3OWw2W.js`, md5 local = servido). Cada código con su texto (es/en), uno
    desconocido cae en el genérico, la carga fallida se dice y deja Guardar
    deshabilitado (antes mandaba `[]` y mostraba "Guardado"). `codigoDe()` compartido
    en `src/api/errores.js`; `AlertaError.jsx` como único lugar del estilo del aviso.
    Texto original: no había texto para esos códigos y quien guardaba no veía por qué.
  - **Contraste de textos secundarios en modo oscuro — CERRADO Y DESPLEGADO
    2026-09-14** (jax-platform#75 → `b08395f`; ver el ítem del tema). El texto
    secundario es el token `texto-tenue` de `src/tema/tokens.css`: oscuro
    `129 144 166` (`#8190a6`) = 4,51 sobre superficie y 5,50 sobre fondo y hundido;
    claro `97 113 136` (`#617188`) = 4,54 / 4,75 / 4,97 sobre fondo / superficie /
    hundido (calculados por Hyde 2026-09-15 con la fórmula WCAG). Lo exige
    `src/tema/contraste.test.js`: los pares `texto-tenue`/{fondo, superficie,
    hundido} a 4,5 en los dos temas, y un control que afirma que el viejo
    `slate-500` (3,75) queda por debajo; el canario del PR 1 (bajarlo a 3,75) puso
    `frontend-tests` en rojo. Texto original: "`text-slate-500` sobre el fondo
    oscuro mide 3,75:1, por debajo del AA de 4,5 (medido en AdminSmtp el
    2026-09-13). Es la convención de todas las pantallas. Arreglarlo es una
    decisión del sistema de diseño, no de una pantalla."
  - **Cancelación durante el rollback de `smtp_config.guardar_filas` — ACEPTADO,
    reconfirmado por Fernando 2026-09-14:** si la tarea
    se cancela justo en el rollback, se loguea el `CancelledError` en vez del error
    original de la base. La conexión no vuelve sucia al pool: `Pool.release` de
    aiomysql cierra las que quedan a mitad de transacción. No se cambió porque
    atrapar `BaseException` en la limpieza arriesga tragarse cancelaciones. Lo
    reabre ver ese caso en un log real.
  - **Pares de contraste por debajo de 4,5 que hoy no se usan — CERRADO Y
    DESPLEGADO 2026-09-15** (jax-platform#79 → `65c02de` borró
    `lightModeOverrides.test.js` y el bloque de rojos/verdes, 35 → 4 overrides;
    #81 → `1d8b787` borró la capa `html.light-mode` entera). Ya no hay overrides:
    `src/tema/contraste.test.js` MIDE el contraste de cada par declarado (91) en los
    dos temas en lugar de exigir que exista un override. Texto original: "`#b91c1c`
    sobre `#fecaca` da 4,47 y el verde sobre `#e2e8f0` da 4,07. El test de modo
    claro exige que exista un override, no que el contraste alcance. Se reabre si
    una pantalla combina `text-red-400` con `hover:bg-red-900` o pone texto verde
    sobre `bg-slate-700`."
  - **Fuente Inter — CERRADO 2026-09-14 — DECISIÓN de Fernando (2026-09-13,
    reafirmada 2026-09-14 con el spec del tema).** Inter se queda y se carga en el
    bundle con `@fontsource/inter` 5.3.0 (versión exacta en `package.json`; latin
    400/500/600/700 importados en `src/main.jsx`, `font-display: swap`;
    jax-platform#75 → `b08395f`). La advertencia del hook de impeccable queda
    registrada en `frontend/.impeccable/config.json` (`overused-font` = `inter`) con
    el motivo existente del 2026-09-13: "Fernando confirmed (2026-09-13, chat):
    Inter es la fuente de la plataforma desde v0.2 (5e28e9e, 2026-06-19); se
    mantiene por coherencia visual". Costo medido en el ítem "Bundle y primer
    pintado del tema". Texto original: "marcada como "sobreusada" por el hook de
    impeccable en `frontend/src/index.css:8`: preexistente. DECISIÓN de Fernando
    2026-09-14: se decide en el Lote 3, junto con los tokens de diseño del tema
    claro/oscuro."

- **Anotados en la ronda del pipeline b8f80733 (2026-09-12).** Ninguno
  bloquea; cada uno dice qué lo reabre.
  - **Embeddings en español: `bge-m3` MEDIDO en JAX (2026-09-12 ~14:30), gana
    con claridad; migración pendiente de ronda propia.** Spike de solo lectura
    sobre los 113 `facts` vigentes, 12 consultas con verdad verificada
    (paráfrasis, inglés→español, dos "difíciles" de preferencia, un control),
    distancia coseno exacta en Python:
    | Brazo | Dim | Recall@1 | Recall@5 | Margen medio |
    |---|---|---|---|---|
    | nomic (como hoy) | 768 | 5/12 | 7/12 | −0,024 |
    | nomic + prefijos | 768 | 6/12 | 7/12 | −0,020 |
    | **bge-m3** | 1024 | **10/12** | **12/12** | **+0,073** |
    Confirma el blueprint de Ricardo (§3): nomic hunde la respuesta correcta
    (rango 36 y 68-73 en dos casos); los prefijos no lo arreglan. bge-m3 no
    desalojó a jax_local (664 MB VRAM); el modelo quedó descargado (1,16 GB).
    **PREPARADO el mismo día (jax#143), corte en producción PENDIENTE.** Las
    dos condiciones previas se cumplieron: sobre `messages` (1.607, 14
    consultas) recall@1 5/14 → **11/14**, recall@5 10/14 → 13/14; y el HNSW
    con `VECTOR(1024)` da 100 % con `ef=400` medido por distancia (una
    comparación por conjuntos falla por los duplicados exactos de `messages`;
    el 93,3 % documentado probablemente arrastra ese sesgo, sin verificar).
    Modelo/dimensión/columna por env (`JAX_MEMORY_EMBED_*`, defaults = hoy:
    desplegar el código no cambia nada). Migración en tres pasos
    (`scripts/migrar_embeddings.py` migrar/activar/revertir) porque MariaDB
    12.3.3 no admite dos índices vectoriales por tabla; la columna vieja nunca
    se toca. Ensayo sobre copia de producción: migrar 43 s, activar 0,8 s,
    revertir 0,8 s. Además: `embedding_worker.py` buscaba `embedding IS NULL`,
    imposible con `VECTOR NOT NULL` — nunca encontraba nada. **Cómo ejecutar:**
    `docs/runbooks/migracion-embeddings-bge-m3.md` (backup + restauración
    probada, turno de chat con memoria obligatorio: `api/chat.py` traga las
    excepciones de `MemoryDB`). Límites: consultas escritas por el subagente,
    sin el filtro por usuario/proyecto del chat.
    **CORTE EJECUTADO 2026-09-12 ~15:50-15:56 CST, verificado en vivo.** Backup
    de conversations/messages/facts (11,9 MB, `~/backups/jax_memory_pre_bge_2026-09-12-1551.sql`,
    umask 077) restaurado en `jax_memory_test` con los tres conteos iguales a
    producción (1607/113/352) y borrado después; `.env` respaldado
    (`/etc/jax/.env.pre-bge-2026-09-12-1551`, idéntico por `cmp`). `migrar`:
    1607 + 113 re-embebidas, 0 fallidas, 43 s. `activar`: 0,8 s. Reinicio
    jax-platform (15:52:45) → jax-las-manos (15:52:52) → worker. Evidencia:
    las 3 variables en `/proc/<pid>/environ` de los dos servicios; `migrar`
    otra vez → 0/0; `EXPLAIN` de las consultas REALES de messages (scope
    individual, sin JOIN) y de facts → `key=idx_embedding_bge_m3`; turno de
    chat real (`origin=probe`) 200 y sus dos filas con embedding bge-m3;
    journal de los tres servicios (con sudo: fruiz sola no lo lee y da
    "No entries", que NO es evidencia) 173 líneas, ninguna de embeddings ni
    error. Segundo método para "con memoria": la búsqueda de `_semantic_context`
    reproducida en proceso con la config viva — una fila migrada (1007) y la
    del turno de hoy (1623) se encuentran a sí mismas con d=0,0000. El primer
    control falló por elegirlo mal (la fila 1616 es de la sonda SP4, con
    `project_id` propio, fuera del scope individual) y se declara así.
    **Seguimiento (este PR):** defaults a bge-m3, esquema regenerado (drift
    9/9 OK), y dos interacciones que el runbook no preveía: (1) con defaults
    bge-m3, "quitar las 3 variables" ya NO revierte — el paso de volver atrás
    pasa a FIJARLAS a nomic; (2) `migrar_embeddings.py revertir` resolvía la
    columna activa con el literal `"embedding"`: con defaults nuevos y la
    variable ausente habría borrado la columna que usan los servicios. Ahora
    sale de `embedding_config`, con test que se vio rojo antes del arreglo.
    **Pendiente del 2026-09-26 — ADELANTADO y HECHO el 2026-09-12 a pedido
    de Fernando.** Columna `embedding` (768) retirada de messages y facts con
    `migrar_embeddings.py retirar` (nuevo: se niega con la columna activa o
    con la indexada; 3 tests de I/O). Antes, sin REPL abierto, y con la
    restauración probada del snapshot restic `75e6d3e2` (local + R2) en un
    MariaDB 12.3.3 descartable: 1607/113/352, 35 tablas. Después: solo
    `embedding_bge_m3` con su índice, conteos intactos, turno de chat real
    200 con sus filas embebidas. Borrado el backup local del paso 1. Volver a
    nomic ya no es `revertir`: es `migrar` hacia `embedding_nomic` + `activar`
    (runbook). Visto de paso: el backup-hall9000 de las 07:34 dio `Failed`
    solo por el prune diferido de R2 (rc=124, timeout); el snapshot sí se
    subió a local y a R2.
  - **El `HttpMuscle` del REPL armaba fuentes opacas de Gemini — CERRADO
    2026-09-12** (jax#142 → `92676b4`). `grounding_sources.py` pasó a
    `jax/core/` (sin copia) y llega a LAS MANOS por el symlink relativo
    `las_manos/grounding_sources.py`, igual que `facet_resolver`: el REPL no
    puede depender de `jacobs/` y LAS MANOS no importa `jax.*`. En vivo tras
    reiniciar `jax-las-manos` (15:29:13): 2 fuentes resueltas, 3 citas.
  - **Las fuentes se deduplicaban por la redirección de Google, no por la URL
    final — CERRADO Y DESPLEGADO 2026-09-14** (jax#150 → `a996c07`; jax-platform#69 → `bfab4de`; `jax-platform` 12:38:07 y `jax-las-manos` 12:38:09, cwd = checkout con el commit, NRestarts=0, journal limpio; frontend `index-BK3OWw2W.js`, md5 local = servido). `resolve_redirects` fusiona
    las que llegan a la misma URL final (queda la primera, citas unidas en orden);
    las no resueltas no se fusionan. 6 tests, cada uno validado por mutación. **En
    vivo, por el camino de Jacobs** (`resolve_facet` + `_invoke_http_gemini`,
    gemini-3.8-flash): 2 fuentes armadas, 2 en el resultado, 2 URLs finales únicas.
    **Límite de esa evidencia:** esa respuesta no traía duplicados, así que prueba que
    no hay regresión con datos reales, no la fusión en acción — eso lo prueban los
    tests. Texto original: pipeline `04e02b09`, `[1]` y `[2]` con la misma URL.
  - **El frontend no tiene tema claro/oscuro — CORREGIDO 2026-09-14 (el ítem era
    falso tal como estaba escrito: existía un modo claro manual, la capa
    `html.light-mode` de `index.css` y el interruptor de la barra; lo verdadero era
    que no había tokens ni garantía de contraste) y CERRADO Y DESPLEGADO
    2026-09-15 04:16 CST.** Cinco PRs de jax-platform, cada uno con gate por
    headSha (11 checks), backup `BACKUP-IDENTICO` y md5 local = servido:
    | PR | Qué | Merge | Deploy | Frontend | Backup en la VM dev |
    |---|---|---|---|---|---|
    | #75 | PR 1: tokens, Login/Reset, `/api/apariencia`, script en línea, Inter | `b08395f` | 2026-09-14 22:51-22:52 | `index--vE4OXEG.js` (md5 `1aa5f60bbee0a35130d29c4850ba1fb4`) | `axioma-ia.io.backup-pre-tema-pr1-20260914-225145` |
    | #77 | Arreglos de la verificación en vivo de Fernando (HalEye animado en el Login, guardar el predeterminado fija la elección del admin, claro gris `slate-100`, ojo con tokens) | `89e9b71` | 2026-09-14 23:59 | `index-CPhAT0Nt.js` (md5 `367312870d2ff68e3a5059b7bd63766d`) | `…backup-pre-tema-fix-vivo-20260914-235854` |
    | #78 | PR 2: administración (13 archivos) + token `obsoleto` | `04c7e3a` | 2026-09-15 02:39 | `index-BEf4mekM.js` | `…backup-pre-tema-pr2-20260915-023902` |
    | #79 | PR 3: chat y pipelines (16 archivos, colores de faceta del store) | `65c02de` | 2026-09-15 03:52 | `index-CQ6VeUCD.js` | `…backup-pre-tema-pr3-20260915-035243` |
    | #81 | PR 4: Dashboard, escaneo de todo `src` y hojas de estilo, fin de la capa `html.light-mode` | `1d8b787` | 2026-09-15 04:16 | `index-DLo6BH_z.js` | `…backup-pre-tema-pr4-20260915-041556` |
    **Evidencia, re-medida por Hyde 2026-09-15 04:17 CST (sólo lectura):**
    `/home/fruiz/jax-platform` en `1d8b787`; `https://axioma-ia.io/login` sirve
    `index-DLo6BH_z.js` (md5 `9692df0145273fbd0ab2b87aec524204`) e
    `index-74p7HosF.css` (md5 `06add6c770a190a77619c28d3dd64b7d`), los dos iguales
    al `dist/` local; `light-mode` en `frontend/src` + `index.html` = 5 líneas, las 5
    en `src/tema/contraste.test.js` y todas son aserciones que lo PROHÍBEN (fuera
    de ese test: 0; en `dist/`: 0). `src/tema/tokens.js` importado con node:
    **48 tokens** (`TOKENS`) y **91 pares AA** (`PARES`), verificados en los dos
    temas por `src/tema/contraste.test.js` en CI (el job `frontend-tests`, visto en
    rojo con el canario del PR 1: `texto-tenue` bajado a 3,75 → failure). Son 91 y
    no los 97 del plan: la decisión de Fernando "superficie opaca + borde del color
    de la faceta" quitó los 9 pares fondo/faceta-* y `obsoleto` sumó 3. El mismo
    test escanea TODO `src` (ya no una lista de migrados) contra clases de paleta
    cruda y hex, y las hojas de estilo fuera de `tokens.css` contra hex, `rgb`
    literal y `light-mode`. `theme_default` funciona: `GET /api/apariencia` → 200,
    `cache-control: no-cache`, `{"theme_default":"dark"}`, y el script en línea de
    `index.html` lo aplica antes del primer pintado (desde `jax_theme_default`).
    Revisión visual de Fernando: PR 1 (con #77) y PR 2 "se ve bien"; chat,
    pipelines y Dashboard en los dos temas quedan para su vistazo de la mañana
    del 2026-09-15 (sin sesión de Hyde). Texto original: "El frontend no tiene
    tema claro/oscuro en ninguna pantalla. Medido: ni variables CSS en
    `src/index.css`, ni una clase `dark:`, ni `darkMode` en Tailwind; todo es
    `slate-*` y hex fijos. Incumple la política "Dark/Light mode — SIEMPRE" en
    toda la app, no en un componente. La cadena siguió las clases del modal;
    arreglarlo es una ronda propia de tokens de diseño."
  - **Carga de GET /api/apariencia — VERDAD OPERACIONAL, 2026-09-14 (hora
    estimada ~23:58, ver corrección) CST.** Rama `d9bcc8d` (PR 1), uvicorn en
    `127.0.0.1:8091`, 1 worker, contra `jax_memory_test`; aislado (sello de
    facetas y HOME temporales, `CANARY_INTERVAL_SECONDS=0`). Respuesta: 200,
    `no-cache`, `{"theme_default":"dark"}`.
    | endpoint | c | n | errores | rps | p50 | p95 | p99 |
    |---|---|---|---|---|---|---|---|
    | /api/apariencia | 1 | 100 | 0 | 1721 | 0,5 ms | 0,8 ms | 3,6 ms |
    | /api/apariencia | 10 | 1000 | 0 | 1606 | 4,6 ms | 9,5 ms | 85,3 ms |
    | /api/apariencia | 30 | 3000 | 0 | 997 | 20,3 ms | 82,6 ms | 119,3 ms |
    | /api/apariencia | 100 | 5000 | 0 | 628 | 71,0 ms | 639,8 ms | 1301,5 ms |
    | /api/health (sin base, referencia) | 30 | 3000 | 0 | 1004 | 19,6 ms | 81,6 ms | 130,4 ms |
    A c=30 el endpoint con base empata con `/api/health` del mismo proceso: la
    consulta por PK no suma latencia medible; degrada entre c=30 y c=100 como
    cualquier endpoint del proceso de un worker. El criterio del plan (p95 28,6 ms
    de `GET /api/admin/config` del 2026-09-13) se midió **en otras condiciones**;
    la comparación honesta es contra `/api/health` de la misma corrida. Barrera:
    sello real y `~/jax/missions` sin cambio, 0 tracebacks. Un primer intento se
    descartó (el uvicorn no arrancó; todo fue conexión rechazada). **DECISIÓN de
    Fernando (2026-09-14 22:51 CST): sin caché** para `/api/apariencia`.
    Corrección: las horas ~23:35/23:45/~23:58 del registro de medidas fueron
    estimadas, no leídas del reloj (a las 22:51 seguía siendo 2026-09-14); los
    números no cambian.
  - **Bundle y primer pintado del tema — VERDAD OPERACIONAL, 2026-09-14 21:38 →
    2026-09-15 04:02 CST.** Lighthouse 12, desktop, `vite preview` de `dist`, 3
    corridas, mediana; FCP/CLS del Login. Bytes gzip con `gzip -9`.
    | Medida (HEAD) | JS crudo / gzip | CSS crudo / gzip | FCP simulado oscuro / claro | FCP observado | CLS |
    |---|---|---|---|---|---|
    | Base (`9138e36`) | 561780 / 171197 B | 29102 / 6170 B | 405 / 241 ms | 53-56 / 52-60 ms | 0 |
    | PR 1 (`def18d8`) | 562514 / 171562 B | 34722 / 7337 B | 522 / 282 ms | 52 / 48-52 ms | 0 |
    | #77 (`3a7c06c`) | 563176 / 171796 B | 34789 / 7341 B | 525 / 282 ms | 52-56 / 52-54 ms | 0 |
    | PR 2 (`4ae3cc8`) | 564380 / 172170 B | 34824 / 7195 B | 522 / 282 ms | 52-56 / 49-53 ms | 0 |
    | PR 3 (`bfe7e86`) | 565303 / 172021 B | 28009 / 6237 B | 522 / 282 ms | 53-57 / 51-55 ms | 0 |
    | PR 4 (`091d660`) | 566126 / 172221 B | 27443 / 6121 B | 525 / 282 ms | 51-59 / 50-61 ms | 0 |
    Inter: 4 woff2 (96744 B) con `font-display: swap`, + 4 woff de respaldo que
    un navegador moderno no baja (fuentes en disco antes del PR 1: 82212 B).
    Resumen: JS +4346 B (+0,8 %), CSS −1659 B (se fue la capa `html.light-mode`).
    **El +117 ms de FCP simulado en oscuro no es del navegador:** Lighthouse corre
    en `simulate` (modelo Lantern) y ahora cuenta en la cadena crítica los 3 woff2
    de Inter que pide el Login; ningún recurso bloquea el render y el FCP
    **observado** no cambia (≈50-60 ms). No se agregó `size-adjust`: CLS 0,000 en
    todas las medidas. **Incertidumbre declarada:** claro corre con perfil
    persistente (caché tibia) y oscuro en frío; no se comparan entre sí, sólo
    antes↔después dentro de cada tema.
  - **npm audit del frontend — CERRADO Y DESPLEGADO 2026-09-14 23:05 CST**
    (jax-platform#76 → `0907f4b`, con GO de Fernando de las 23:04). `npm audit fix`
    sin `--force`: 9 → 0 vulnerabilidades (6 high: browserslist, nanoid, postcss,
    react-router, react-router-dom, undici; 3 moderate: @vitest/mocker,
    baseline-browser-mapping, vitest), todo patch/minor, sólo `package-lock.json`;
    al bundle sólo llegan react-router/react-router-dom 7.18.0 → 7.18.3. `npm audit`
    en producción = 0; frontend `index-DkQdXnxb.js`, md5 local = servido
    (`1ff0f985b370d0bbf46ba62410c88a3e`); backup
    `axioma-ia.io.backup-pre-npm-audit-20260914-230516` `BACKUP-IDENTICO`.
  - **Fechas sin zona horaria en el resto de la API — PENDIENTE, fecha
    2026-09-22 (propuesta por Hyde; Fernando la confirma o la mueve).** Anotado 2026-09-15 (Ruling U7, etapa 3 de admin usuarios): la
    sesión de MariaDB corre en CST (`SYSTEM`, `NOW()` = UTC−6, medido por pytest) y
    los endpoints serializan `TIMESTAMP`/`DATETIME` sin zona con `isoformat()`: el
    navegador los lee como hora local. La etapa 3 sólo arregló los suyos
    (`last_login`/`created_at` por `UNIX_TIMESTAMP` y enviados ISO `+00:00`; `ts` de
    `user_admin_audit` escrito en UTC por `registrar`, único escritor; su DEFAULT
    sigue en CST). Falta revisar TODOS los demás endpoints que serializan fechas.
    Costo mientras tanto: horas corridas en otras pantallas.
  - **vitest no corre en el CI de jax-platform — ya estaba CERRADO, el ítem estaba
    vencido (medido 2026-09-14).** El job `frontend-tests` existe desde el 2026-09-12
    (jax-platform#60, con canario visto en rojo) y hoy exige 125 tests exactos. Otra
    instancia de §7: un ítem que describe un estado que ya no existe.
  - **Una cadena en modo `supervised` pide una aprobación por paso.**
    `supervised` corre UNA ola y pausa (`executor.py:966`); en paralelo eso era
    una sola pausa, en cadena son cinco. El botón "Aprobar" de `RightPanel`
    reanuda. **Decidido el mismo día por Fernando: la cadena va en
    `autonomous` por defecto** (jax-platform#55). `supervised` sigue
    disponible a mano, con su pausa por paso.
  - **Aprobar en `RightPanel` tragaba el error — CERRADO Y DESPLEGADO 2026-09-14** (jax#150 → `a996c07`; jax-platform#69 → `bfab4de`; `jax-platform` 12:38:07 y `jax-las-manos` 12:38:09, cwd = checkout con el commit, NRestarts=0, journal limpio; frontend `index-BK3OWw2W.js`, md5 local = servido).
    Aprobar y Cancelar (mismo defecto) muestran su fallo, atado a su pipeline: no queda
    sobre otro ni cuando el pipeline ya no espera aprobación. Texto original:
    `handleResume` solo hacía `console.error`.
  - **Cancelar kimi: MEDIDO 2026-09-14, no se cobró el pedido cortado (una muestra).**
    La documentación oficial de Moonshot no dice si el servidor para ni qué cobra al cortar
    (guía de streaming: "the request's token consumption cannot be determined"; solo documenta
    que un 429 no se cobra). Medido con `GET /v1/users/me/balance` (USD, 5 decimales): un stream
    de `kimi-k3` cortado a los 150 fragmentos (7 s) dejó el saldo en 0,00000 de gasto durante
    ~15 min; el CONTROL (un pedido completo, 104+400 tokens) sí se descontó, 0,00919 USD, visible
    entre 90 y 180 s. **Hallazgo aparte, abierto:** 0,00919 cobrado contra 0,00631 a precio de
    lista (3/15 USD por 1M) — si `model` guarda el de lista, `axioma_usage` subregistra kimi.
  - **`finish_reason=length` sin schema marcaba `completed` — DECISIÓN de Fernando,
    2026-09-14: revierte la del 2026-08-10; DESPLEGADO (jax#152 → `d5ccbfa`; `jax-las-manos`
    reiniciado 13:18:17, proceso en el commit, 0 errores). Límite de la evidencia: en vivo no se
    forzó un corte real (exigiría bajar `max_tokens` de un motor en producción); lo prueban los
    tests y el despliegue.** Una salida cortada que pasa por completa es fail-open (P10). En
    `worker.py` la rama de corte va ahora ANTES de mirar tool_calls o schema: falla
    siempre, sin reintento, también con schema que acepta texto libre y con
    tool_calls (hueco hermano que encontró la revisión: se ejecutaban herramientas
    con argumentos truncados). Con `max_tokens=0` el error dice que el motor no lo
    declara. Impacto medido antes de desplegar: 0 de 57 jobs de `motor_jobs.jsonl`
    estaban cortados. Texto original: decisión del 2026-08-10 ("el dato queda para
    diagnóstico, no bloquea"), el arreglo solo cubría el camino con schema.
  - **El nombre del pipeline llevaba `Pipeline: ` fijo — CERRADO Y DESPLEGADO
    2026-09-14** (jax#150 → `a996c07`; jax-platform#69 → `bfab4de`; `jax-platform` 12:38:07 y `jax-las-manos` 12:38:09, cwd = checkout con el commit, NRestarts=0, journal limpio; frontend `index-BK3OWw2W.js`, md5 local = servido). Sale de `t.pipelineName()`; el test usa un spy porque el texto
    de es.js coincide con el prefijo viejo (verificado por mutación). `invoked_by`,
    que está al lado, NO se tocó: es autorización (ver "Bloquea trabajo").

- **El `JOIN` a `conversations` anula el indice vectorial HNSW de `messages` —
  CERRADO 2026-09-11 (jax#128), desplegado y verificado.** El arreglo candidato que
  esta entrada proponía —desnormalizar `user_id`/`project_id` a `messages`— se hizo
  esa misma noche, con su migrador (`jax/memory/migrations.py`, el primero que
  tienen esas tablas) y backfill de las 1.607 filas.

  **Y trajo consigo un cambio que hay que saber: usar el índice HNSW vuelve la
  búsqueda APROXIMADA.** Con el default de MariaDB (`mhnsw_ef_search=20`) el
  recall@5 medido fue **50,7 %** —media memoria perdida, sin un solo error ni una
  línea de log—; se fijó en **400** (93,3 %, 0,92 ms contra 49 ms de la búsqueda
  exacta) en el `init_command` de la conexión, con un test que falla si vuelve a
  20. Configurable por `JAX_MEMORY_HNSW_EF_SEARCH`, y **hay que volver a medirlo
  cuando `messages` crezca un orden de magnitud**.

  Texto original del diagnóstico, conservado:
  `messages` tiene `idx_embedding` VECTOR (HNSW, MariaDB 12.3), pero
  `search_similar_messages` no lo usa: su `EXPLAIN` da
  `Using temporary; Using filesort` y elige `idx_conv_project`.

  | Consulta | ms |
  |---|---|
  | La real (JOIN + scope + filtro anti-NaN) | **58,5** |
  | Sin el filtro anti-NaN | 46,5 |
  | **Con el filtro, sin el JOIN** | **0,4** (usa el indice) |
  | Canonica `ORDER BY VEC_DISTANCE … LIMIT` | 0,2 |

  **El JOIN es la causa, no el filtro de vector cero de jax#116** — el filtro
  cuesta 12 ms de los 58, el JOIN cuesta el resto y ademas mata el indice.
  **Raiz:** el scope (`user_id`/`project_id`) vive en `conversations`, asi que
  hay que unir para filtrar. **Arreglo candidato:** desnormalizar esas dos
  columnas a `messages` (con el costo de mantenerlas sincronizadas, que es la
  decision real). **Por que no se hizo hoy:** es cambio de esquema en la tabla
  mas grande y toca el camino de cada turno de chat — no se cuela en un PR de
  otra cosa. **Hoy no duele** (1.149 filas, 12 MB); duele lineal.

- **Linea base de carga — 2026-09-11, `/api/health` de jax-platform.**
  Primera medicion de carga del ecosistema (antes no habia herramienta).

  **CORREGIDA el mismo dia, y la correccion es la parte interesante.** La
  primera version del arnes usaba `httpx` y midio c=1 2.605 rps / p95 0,48 ms;
  c=10 1.649 / 12,1 ms; c=50 1.275 / 100,5 ms; c=100 2.109 / 48,2 ms. Al
  reescribirlo sobre stdlib (porque en atem-ai no hay httpx), los mismos
  niveles dieron **c=1 5.815 rps / p95 0,22 ms; c=10 7.522 / 1,9 ms; c=50
  5.638 / 6,53 ms** — hasta 4,5x mas rps y un p95 quince veces menor. El
  servidor es el mismo: **lo que se estaba midiendo era el cliente**. El p95 de
  100 ms a c=50 que parecia degradacion del servicio era overhead del arnes.

  **TERCERA MEDICION, con k6 (2026-09-11, mismo dia): el arnes de stdlib
  tambien subestimaba.** Mismo endpoint, 50 concurrentes:

  | Instrumento | rps | p95 |
  |---|---|---|
  | arnes con httpx | 1.275 | 100,5 ms |
  | arnes con stdlib | 5.638 | 6,53 ms |
  | **k6 v2.2.0** | **20.399** | **3,01 ms** |

  El arnes de Python esta limitado por el GIL: sus hilos no corren en paralelo
  de verdad. **El servicio aguanta 16x mas de lo que decia la primera
  medicion.** Para saber cuanto aguanta de verdad, la referencia es k6
  (`loadtest/health.js`, con thresholds que dan exit != 0); el arnes de stdlib
  queda para medir en cualquier maquina sin instalar nada — sabiendo que su
  numero es un piso, no el techo.

  **CAMINOS AUTENTICADOS, 2026-09-12.** Faltaba la línea base de lo que no es un
  health check. `loadtest/api-autenticada.js` mide tres endpoints de SOLO LECTURA
  que pasan por el JWT y tocan la base (`/api/pipelines`,
  `/api/motors/capabilities`, `/api/facets`), 20 concurrentes: **2.055 rps, p95
  13,5 ms, cero errores**, thresholds en verde. Diez veces más caro que
  `/api/health` (20.399 rps, p95 3 ms), que es lo esperable: ahí se ve el costo
  del JWT y de la base.

  **Lo que sigue sin medirse, y por qué:** el turno de chat con búsqueda semántica
  y el despacho de pipelines. Medirlos bajo carga contra producción invoca a
  Ollama —GPU real— y **escribe turnos en la memoria**: la prueba ensuciaría justo
  los datos que el sistema usa para recordar. Necesitan un entorno con su propia
  base y su propio Ollama; es una ronda aparte, no un número que se saque de paso.

  *(Al medirlo, el generador de tokens falló en silencio por `JAX_JWT_SECRET` sin
  cargar y k6 reportó 100 % de fallos. Cuarta vez en dos días que el instrumento
  miente antes que el servicio: verificar con un `curl` suelto antes de creerle a
  una corrida entera.)*

  **Leccion, y vale mas que el numero:** una prueba de carga mide el sistema
  MAS el instrumento. Antes de reportar degradacion hay que descartar que el
  cuello sea el que mide — con un cliente distinto, o mirando si el servidor
  esta ocioso mientras el arnes sufre. Un arnes lento no da un error: da un
  numero pesimista y creible. **Cero errores en todos los niveles, en las dos
  versiones.** Es un endpoint trivial: mide el event loop,
  no la app. **Falta la linea base de los caminos caros** (chat con busqueda
  semantica, pipelines), que necesitan auth y escriben — hacerlas contra la base
  de test, no contra produccion. Es VERDAD OPERACIONAL: caduca si cambia el
  esquema, el volumen o la infraestructura.

- **El brazo negativo de la sonda de SP4 no mide fabricación: hay que cerrar
  la puerta de `ssh_exec` — 2026-09-04, causa medida. DECISIÓN de Fernando 2026-09-14: se
  espera a retomar SP4.** En el corrido dirigido
  del 2026-09-03, **82 de los 120 turnos negativos (68%)** terminaron con la
  faceta afirmando `ssh_exec` o `ssh_exec_readonly`, capabilities que SÍ están
  en el snapshot. Eso no es fabricación y por eso el pre-registro los clasifica
  como no responsivos, pero deja el resultado sin fuerza: **el 0% de fabricación
  es sobre 38 turnos efectivos, no sobre 120**, y no distingue "no fabrica" de
  "no necesitó fabricar". **Qué falta:** ítems negativos que no tengan salida
  por una capability real — o excluir `ssh_exec`/`ssh_exec_readonly` del
  snapshot inyectado para ese brazo, que es un cambio del mecanismo y por lo
  tanto exige su propio pre-registro y su propio corrido. **No se trabaja
  ahora**, por decisión explícita. **Fecha de control:** al retomar SP4.
  **2026-09-14 (tanda A v2):** el snapshot suma `catalog_capabilities` (28 entradas en vez de 11). La
  línea base del 2026-09-03 deja de ser directamente comparable; al retomar SP4 hace falta una nueva, con su
  pre-registro (spec tanda A v2 §2, decisión 4).

- **`save_message()` no reintenta el embedding — CERRADO Y DESPLEGADO
  2026-09-11 (jax#118, `47589ae`).** `MemoryDB.backfill_zero_embeddings()`
  reintenta hasta 50 filas en ceros por tabla (`messages` y `facts`) en **cada**
  pasada del worker, **antes** del `return` temprano de "no hay conversaciones"
  — que es el caso de casi todas las pasadas. `UPDATE` guardado por "sigue en
  ceros"; si Ollama sigue caído la fila queda y se reintenta a los 20 min; un
  fallo del recálculo no frena la extracción. 6 tests (4 contra MariaDB real),
  todos en rojo antes, y por mutación cada pieza tiene uno que cae sin ella.

  **En vivo, dicho con su alcance real:** la pasada de las 04:51:20 corrió con
  el código nuevo (checkout en `47589ae`), terminó `Deactivated successfully`
  sin un solo `ERROR`, y dejó `messages`/`facts` con **0 filas en ceros**
  (medido con `VEC_DISTANCE_EUCLIDEAN`). Con 0 pendientes el paso no loguea, así
  que lo que el journal prueba es que su `SELECT` corre en producción sin
  fallar; el camino con filas reales lo prueban los tests contra MariaDB. No se
  fabricó una fila en ceros en producción para verlo loguear.

  Texto original del ítem, conservado:

  **`save_message()` no reintenta el embedding: una fila que nace en vector
  cero queda así para siempre — ABIERTO, 2026-09-11.** Es la causa viva detrás
  del 500 del 2026-09-11 (ver "Cerrado — kimi y la memoria vector cero"). La
  búsqueda ya excluye esas filas, así que **no rompen nada**; lo que queda es
  que **se pierden de la memoria en silencio**: un mensaje guardado durante una
  caída de Ollama no vuelve a aparecer en ninguna búsqueda, y nada lo avisa.
  Las 24 del 2026-06-09 se repararon **a mano** (un backfill, no un arreglo:
  ver la lección "un arreglo manual no es un arreglo"). **Qué falta:** que algo
  recalcule los embeddings en ceros — candidato natural, el worker de memoria
  (`jax-memory-worker.timer`, cada 20 min), con el mismo `UPDATE` guardado por
  "sigue en ceros" del backfill. **Cómo medirlo:** `SELECT COUNT(*) FROM
  messages WHERE VEC_DISTANCE_EUCLIDEAN(embedding, <cero>) = 0` — **no** con
  `IS NULL` ni con la distancia coseno, que dan NaN y engañan (así se midió mal
  el 2026-09-03). **Fecha de control: 2026-09-25.**

- **`jax_memory_schema.sql` desactualizado — CERRADO 2026-09-11, con el
  chequeo automatizado que la nota de patrón pedía a la tercera instancia.**
  Tres piezas, cada una verificada:

  | Pieza | Evidencia |
  |---|---|
  | Archivo **regenerado** desde `SHOW CREATE TABLE` de producción (9 tablas, sin `AUTO_INCREMENT=N`, cabecera y semillas conservadas) | las diferencias reales eran 6 columnas faltantes (`conversations.tenant_id/user_id/project_id`, `decisions.user_id`, `action_items.user_id/project_id`), el ENUM de `role` y `embedding`; el resto era cosmético (`int` vs `int(11)`, `boolean` vs `tinyint(1)`) |
  | **El archivo se ejecuta en CI**: `test_memory_vector_zero_io.py` crea sus tablas desde él (ya no lleva copia del DDL) | 10 passed contra MariaDB con las tablas creadas desde el archivo |
  | **`scripts/check_memory_schema_drift.py`**: compara el `SHOW CREATE TABLE` vivo contra el archivo; exit 0/1/2, fail-closed | contra `jax_memory`: **OK, 9 de 9**. Mutación contra la base real (archivo sin `project_id`): **DRIFT, exit 1**. 6 tests puros del comparador, en `tests-puros` |

  **Límite declarado:** el chequeo corre en hall9000 contra producción (manual,
  como el tripwire de `OLLAMA_NUM_PARALLEL`), no en CI: la salida de
  `SHOW CREATE TABLE` depende de la versión y CI usa MariaDB 11.8 contra la
  12.3 de producción. **Cuándo correrlo:** después de cualquier `ALTER` sobre
  estas 9 tablas, y antes de dar por cerrada una ronda que las toque.

  Texto original, conservado:

  **`jax_memory_schema.sql` desactualizado respecto de producción — ABIERTO,
  2026-09-11.** Medido con `SHOW CREATE TABLE` contra `jax_memory`: al archivo
  le faltan `conversations.tenant_id/user_id/project_id` (el scope de la
  búsqueda depende de ellos), declara `embedding VECTOR(768) NULL` donde
  producción tiene `NOT NULL DEFAULT` vector cero más `VECTOR KEY`, y el ENUM de
  `messages.role` tiene 5 de los 8 valores. **Ningún `ALTER` del código explica
  la diferencia.** Tercera instancia del patrón "ALTER a mano en producción que
  nunca entró al camino de creación" (después de `depends_on` y de las 3
  capabilities HTTP-directo): una instalación nueva desde este archivo no
  corre la búsqueda semántica. Por eso `tests/test_memory_vector_zero_io.py`
  declara el DDL de producción en vez de leer el archivo. La nota de patrón de
  más abajo decía "si aparece una tercera instancia, es momento de escribir un
  chequeo automatizado": apareció. **Fecha de control: 2026-09-25.**

- **Tests de `tests/` sin job de CI — CERRADO 2026-09-11, y era mucho más que
  un archivo.** Al medir el ítem de abajo: **18 de 26 archivos de `tests/` no
  los nombraba ningún job.** Clasificados uno por uno en un venv limpio (no en
  hall9000), con el mismo `pip install` que usa el job:

  | Grupo | Archivos | Destino |
  |---|---|---|
  | Puros, pasan limpios | 11 (`memory_helpers`, `chunking`, `memory_correction`, `jacobs_director`, `validate_capability_typed`, ...) | job nuevo **`tests-puros`** |
  | Script sin `test_`, **ROTO** | `test_jacobs_timeout_by_capability.py`: pytest le colectaba 0 tests; hecho test, **falló** — el refactor del 2026-09-01 cambió la firma de `_from_spec` (ahora recibe `caps`) y el archivo nunca se actualizó. Hardcodeaba además `~/jax` en `sys.path` | arreglado con `caps` fabricado = valores reales de producción (medidos: 15/15/15/5 min, `assemble` sin fila); los 7 checks pasan sin cambiar una sola expectativa → `tests-puros` |
  | Contra la gobernanza real | `plan_validation`, `valid_capabilities_includes_file_tools`, `plan_capability_hint` (1 de sus 6 va contra DB a propósito) | job nuevo **`jacobs-gobernanza-db`**: clona `jax-platform` y corre **sus** migraciones, sin copiar DDL |
  | LAS MANOS en proceso | `audit_traffic_class`, `envelope_brutal`, `thot_connection` | **ABIERTO**, ítem propio abajo |

  Pisos medidos: `tests-puros` **62 corridos** (incluye los 6 del chequeo de
  esquema), `jacobs-gobernanza-db` **14 corridos**. El segundo no se pudo
  reproducir contra una base **vacía** (sin Docker en hall9000): eso lo prueba
  el CI, dicho así.

- **El sello de `facet_resolver` puede perder una invalidación: `os.utime(p,
  None)` escribe un `mtime` que va detrás de `time.time()` — CERRADO 2026-09-11
  (jax#123 + jax-platform#51), desplegado y verificado en vivo.** Lo que sigue
  es el texto del diagnóstico, conservado; el cierre y **el límite de su
  evidencia** están al final del ítem. Apareció como rojo intermitente de
  `facet-resolver-seal` en jax#120 (un PR que no toca nada de ese job):
  `test_invalidate_facet_cache_lo_ven_los_otros_procesos` falló en la corrida
  `push` y pasó en la `pull_request` del **mismo commit**; el reintento pasó.
  En las 24 corridas anteriores del job, cero fallos.

  **Mecanismo:** `_entrada_sellada` compara `st_mtime >= entry.fetched_at_wall`.
  `fetched_at_wall` es `time.time()`; el sello se escribe con
  `os.utime(FACET_SEAL_PATH, None)`, que le pone al archivo la hora **gruesa** del
  kernel. Si un `invalidate` cae a menos de un milisegundo de un cacheo, el
  sello queda **anterior** a la entrada y la invalidación se pierde hasta el TTL
  (30 s). jax#114 cerró el empate exacto (`>` → `>=`); esto es otra cosa.

  | Medición (20.000 iteraciones: `time.time()`, tocar, leer `st_mtime`) | `mtime` anterior al `time.time()` previo |
  |---|---|
  | `/srv/jax-data` (xfs, **donde vive el sello real**), `os.utime(p, None)` — como hoy | **1 / 20.000**, −0,693 ms |
  | mismo filesystem, `os.utime(p, (t, t))` con `t = time.time()` | **0 / 20.000** |

  **Corrección registrada:** la primera medición se hizo en `/tmp` (tmpfs), dio
  0/20.000 con los dos métodos y se leyó como "hipótesis refutada" — **era el
  filesystem equivocado**. El sello no vive ahí.

  **Alcance:** en producción es raro (hace falta invalidar a menos de 1 ms de
  un cacheo en otro proceso); en CI hace inestable un test. **Arreglo propuesto,
  no aplicado:** escribir el sello con hora explícita del mismo reloj que
  compara, `os.utime(FACET_SEAL_PATH, (t, t))`. Es de una línea, pero
  `facet_resolver` está espejado en `jax` y `jax-platform` (`mirror-sync` exige
  que coincidan) y desplegarlo reinicia `jax-platform` y `jax-las-manos`:
  **espera GO de Fernando.** **Fecha de control: 2026-09-18.**

  **CIERRE — 2026-09-11, GO de Fernando.** `_tocar_sello` muestrea
  `time.time()` y escribe `os.utime(p, (ahora, ahora))`: el sello deja de
  depender de un reloj que el código no controla y pasa a llevar el mismo con
  el que se compara `entry.fetched_at_wall`. Espejado en los dos repos
  (`check_mirror_sync.py` en verde — atrapó una línea duplicada al portarlo,
  antes de que llegara a ningún test). Pisos de CI: `facet-resolver-seal`
  12 → 13; en `jax-platform`, con DB 390 → 391 y sin DB 205 → 206, los dos
  medidos en el árbol antes de escribirlos. Desplegado con reinicio de
  `jax-platform` y `jax-las-manos` (18:39 CST): los dos `active`, `/api/health`
  200, las 6 facetas sondeables en `ok`, sin alertas nuevas, sin warnings en
  journal.

  **Test:** la carrera real es de sub-milisegundo, así que esperarla en un test
  sería un flake y no una prueba. El test **amplifica la misma diferencia**
  adelantando `time.time()` una hora: con `None` la invalidación se pierde de
  forma determinística, con hora explícita no. Verificado ROJO antes del
  arreglo en los dos repos.

  **LO QUE LA MEDICIÓN DE HOY NO PRUEBA — y queda escrito para que nadie lo lea
  al revés.** Se repitió el experimento sobre el filesystem real con el código
  nuevo (0/20.000 atrasados, xfs `/srv/jax-data`), pero **el control con el
  código viejo tampoco reprodujo el defecto**:

  | Corrida de control (`os.utime(p, None)`) | Resultado |
  |---|---|
  | xfs `/srv/jax-data`, 3 × 20.000 | 0 atrasados |
  | xfs `/srv/jax-data`, 200.000 | 0 atrasados |
  | ext4 (`/`) y tmpfs (`/tmp`), 200.000 cada uno | 0 atrasados |
  | inodo nuevo por iteración (hipótesis de timestamps multigrano de xfs) | 0/20.000 |

  O sea: **el 1/20.000 registrado esta misma mañana no se pudo re-confirmar**
  en esta máquina y con este kernel (7.0.0-31-generic), y por lo tanto el
  0/20.000 del código nuevo **no es evidencia de nada por sí solo** — un
  control que no falla no valida al tratamiento. Lo que sostiene el cambio es
  el argumento estructural (el sello ya no depende del reloj del kernel) y el
  test determinístico. **Queda abierta la pregunta de qué condición produjo
  el 1/20.000 y el flake de CI de jax#120**; si el flake reaparece con este
  arreglo puesto, la hipótesis del mecanismo estaba equivocada y hay que
  volver a diagnosticar, no a parchear. Quinta lección del día: una medición
  sin control es una suposición con decimales.

  **Verificación en vivo del camino de escritura real** (sello de producción,
  un `_tocar_sello()` cuesta que los tres procesos re-consulten una vez):
  `mtime` del sello **+0,046 ms por delante** del `time.time()` muestreado
  justo antes.

- **Los 3 tests de LAS MANOS en proceso no pueden correr en CI — ABIERTO,
  2026-09-11, causa medida.** `test_audit_traffic_class.py`,
  `test_envelope_brutal.py` y `test_thot_connection.py` levantan
  `las_manos/server.py` con `TestClient`, y el servidor toma el audit log de
  `las_manos/config.toml`: `audit_log = "/home/fruiz/jax/las_manos/logs/audit.jsonl"`
  — **ruta absoluta de hall9000**. En un runner esa ruta no existe; y en
  hall9000, desde el checkout que sea, **escriben en el audit log REAL** (dos
  de ellos lo dicen en su docstring). No se corrieron por eso. **Qué falta:** que
  el servidor reciba la ruta del audit log por entorno (con el valor actual como
  default de producción), y recién ahí meterlos a un job con el log apuntando a
  un temporal. Es un cambio de cómo arranca LAS MANOS, no de los tests.
  **Fecha de control: 2026-09-25.**

- **`tests/test_memory_helpers.py` no lo corre ningún job de CI — CERRADO, y ya lo
  estaba cuando se escribió esta entrada (revisado 2026-09-12).** El texto pedía
  "nombrarlo en un job, con piso"; `grep` sobre `.github/workflows/policy.yml` lo
  encuentra **dos veces**, en el job `tests-puros` y en su piso exacto. Entró ahí
  en jax#120, **el mismo día** en que se redactó este ítem, y nadie volvió a
  mirar. Otra instancia del patrón que persigue §7 de `CONTEXT.md`: un ítem que
  describe un estado que ya no existe. Lo destapó el inventario del 2026-09-12,
  midiendo cada entrada contra el árbol en vez de leerla.

- **500 en `/api/chat` por distancias NULL en `_semantic_context` --
  CERRADO 2026-09-11, ver "Cerrado — kimi y la memoria vector cero".** Lo que
  sigue es el texto del 2026-09-03, **conservado y corregido, no borrado**: su
  medición de "cero vectores cero en producción" era **falsa** — había 24 filas
  así desde el 2026-06-09. La consulta con la que se midió no está registrada;
  lo más probable es que la engañara el mismo NaN (una distancia NaN no es
  NULL para el servidor). La sospecha del final ("vector de consulta
  degenerado") también era parcial: la causa real eran filas guardadas de
  norma cero. Texto original:

  **500 en `/api/chat` por distancias NULL en `_semantic_context` --
  2026-09-03, NO REPRODUCIDO.** Observado una sola vez, en un montaje de dev
  levantado a mano (uvicorn aparte en :8099, `JAX_DB_NAME=jax_memory_test`,
  `JAX_REPO_PATH` apuntando a una copia con el `config.toml` roto a propósito).
  Cada turno terminaba en `TypeError: '<' not supported between instances of
  'NoneType' and 'float'` en `api/chat.py::_semantic_context`, en la línea
  `[r for r in similares if r["distancia"] < 0.8]`: `search_similar_messages()`
  devolvía filas con `distancia = NULL`. `VEC_DISTANCE_COSINE` devuelve NULL con
  un vector cero de cualquiera de los dos lados, así que la sospecha es un
  vector de consulta degenerado, pero **no está probada**.

  **Por qué queda anotado y no cerrado:** no se reprodujo. Contra la misma DB,
  en proceso limpio, `search_similar_messages()` devolvió 0 filas y ningún
  error; y en producción no aparece: los 7 turnos de la verificación en vivo
  del 2026-09-03 respondieron 200, y `messages`/`facts` de `jax_memory` no
  tienen ni un embedding NULL ni uno de vector cero (medido). Puede ser un
  artefacto del montaje (DB de test con `messages` vacía, historial en memoria
  del proceso) y no un defecto del código.

  **Lo que importa si es real:** el filtro está en el camino de TODO turno de
  chat y **no es fail-soft** -- el `try/except` de más arriba cubre la consulta,
  no la comparación, así que una sola fila con `distancia` NULL tumba el turno
  entero con 500. Eso es más grave que el ranking que se pierde.

  **Qué haría falta para cerrarlo:** reproducirlo con intención (forzar un
  embedding de consulta cero o una fila con vector cero y correr el endpoint), o
  descartar la clase entera acotando el filtro
  (`r.get("distancia") is not None and r["distancia"] < 0.8`), que es una línea y
  convierte el 500 en un candidato menos. **Fecha de control: 2026-09-17.**

- **`GROUNDING_UNAVAILABLE` no puede aparecer en producción por su causa
  realista — CERRADO 2026-09-12 (jax#131 + jax-platform#52), desplegado.** Las dos
  mitades que pedía el "qué falta":

  1. **Cortocircuitar el paso 0 antes de `_validation_context()`** — hecho. Ahora,
     con un `SnapshotError`, se emite un veredicto `GROUNDING_UNAVAILABLE` por
     claim y la fila se marca validada, en vez de morir releyendo la misma config
     rota. El veredicto lo produce `governance_validator.verdict_sin_grounding()`,
     extraída del paso 0 de `validate()`: **una sola definición**, no una copia en
     el caller. **Lo que sí se pierde, declarado:** el barrido de vocabulario, que
     sin `term_categories` no se puede hacer.
  2. **Revalidar el contexto (régimen B)** — ya estaba hecho desde el 2026-09-03
     (`governance_context` con `stat()` por construcción, y
     `backend/tests/test_grounding_config_revalidation.py`). El ítem lo pedía como
     pendiente y llevaba nueve días cerrado: se comprobó contra el código antes de
     tocar nada.

  El test que fijaba el defecto se **reescribió** para exigir el comportamiento
  correcto —mismo escenario provocado, ahora esperando los veredictos— y se vio en
  rojo antes del arreglo. Texto original del diagnóstico, conservado: El spec §4.2 dice
  que el paso 0 "se aplica a todo claim del turno". Provocado con
  `las_manos/config.toml` ilegible, en el endpoint real, en proceso, contra
  `jax_memory_test`:

  | régimen | qué ve el usuario | fila en `shadow_messages` | veredictos |
  |---|---|---|---|
  | **A** — caché fría (primer turno tras reiniciar), config rota | HTTP **200**, respuesta normal, `contract_degraded: false`; prompt SIN bloque de hechos | `grounding_snapshot={"error": "TOMLDecodeError: ..."}`, `sha256='ERROR'`, **`validated_at NULL`**, `has_claim=1` | **CERO** |
  | **B** — caché caliente, la config se rompe después | HTTP 200, y el prompt SÍ trae el bloque de hechos | snapshot **válido**, `sha256` de siempre (`6d007427…`) | los normales |
  | (control) snapshot roto + contexto sano | HTTP 200 | `sha256='ERROR'`, `validated_at` NO NULL | **1 fila `GROUNDING_UNAVAILABLE/INFERIDO`** |

  **Régimen A** confirma el hallazgo de la revisión final: `run_shadow_validation`
  llama a `_validation_context()` ANTES del loop de claims, así que la misma
  config ilegible que produjo el `SnapshotError` lo hace lanzar de nuevo; la task
  muere fail-closed (visible: `validated_at NULL`) sin escribir un solo veredicto.
  **Régimen B** es el hallazgo nuevo y el más incómodo: `lru_cache(maxsize=1)`
  sirve el contexto cacheado, así que una config que se rompe con el proceso ya
  caliente **no produce ni `ERROR` ni degradación** — el snapshot sale igual, con
  datos que ya no están en el disco. La fila de control muestra que el estado
  funciona como el spec pide cuando el contexto está sano y el snapshot no, que es
  el caso que ejercitan los tests con `SnapshotError` inyectado.

  **Qué falta:** (1) cortocircuitar el paso 0 antes de `_validation_context()`
  — `accredit()` y el paso 0 de `validate()` no necesitan ni `ctx` ni
  `predicates`, así que con un `SnapshotError` se pueden escribir los veredictos
  `GROUNDING_UNAVAILABLE` de todos los claims y saltar el vocab sweep;
  (2) decidir si `validation_context()` debe revalidar (mtime o hash del config)
  para que el régimen B deje de ser invisible. **Causa:** orden del plan de SP3
  (Task 5) más la caché de proceso, no un slip de implementación.
  **Fecha de control:** al abrir SP4 (análisis de datos de grounding).

- **`GPU_SEMAPHORE` no cubre a Jacobs — DECISIÓN TOMADA (2026-09-01), no
  pendiente.** Estuvo listado como deuda abierta desde el Bloque 2
  (2026-08-21) diciendo "no se cierra por decisión", que es una frase que se
  lee como pendiente. No lo es: es una decisión con la medición en la mano, y
  desde hoy con un tripwire que la protege.

  **El hecho.** `jax/muscles/ollama_muscle.py::GPU_SEMAPHORE` es un
  `asyncio.Semaphore` **de proceso**, no cross-proceso. Jacobs corre en el
  proceso `jax-las-manos` y tiene tres caminos propios a Ollama que no pasan
  por él: `jacobs/plan.py::_llm_plan`, `jacobs/executor.py::_invoke_ollama`,
  y `las_manos/motor_registry/worker.py` con transporte `ollama`. Eso es
  cierto y sigue siendo cierto. **No es un bug: es la consecuencia esperada de
  un semáforo de proceso.**

  **El invariante que hace que no importe, medido** (2026-08-28,
  `scripts/gpu_concurrency_probe.py`): no hace falta exclusión mutua
  cross-proceso **porque Ollama serializa**, y serializa porque
  `OLLAMA_NUM_PARALLEL=1`. La exclusión ya existe afuera del código de JAX.

  **Lo que faltaba, y es lo que se agregó hoy: el invariante no tenía
  tripwire.** Toda la decisión cuelga de un valor que vive en la unidad de
  systemd de un servicio de terceros. Si ese valor deja de ser 1 la decisión
  no se degrada — **se invierte**: los tres caminos pasan a correr en paralelo
  real contra la misma GPU, y nadie se entera. Una decisión que depende de un
  invariante sin tripwire es una suposición con fecha de vencimiento
  desconocida.

  **Tripwire: `scripts/check_ollama_num_parallel.py`, job `ollama-num-parallel`.**
  Lee `/proc/<pid>/environ` del `ollama serve` **vivo**, no el archivo de
  unidad — la unidad declara, el proceso ejecuta, y pueden diferir (drop-in
  posterior, `systemctl set-environment`, arranque a mano, unidad editada sin
  recargar). Fail-closed en todos los caminos: variable ausente, valor
  distinto de 1, environ ilegible, environ vacío o proceso inexistente son
  **rojo**, nunca "no aplica".

  **Probado rompiéndolo, en vivo, los dos sentidos** (2026-09-01):

  | Fuente | Resultado |
  |---|---|
  | Ollama real de hall9000 (pid 3376, descubrimiento automático, `sudo`) | **VERDE**, exit 0 — `OLLAMA_NUM_PARALLEL=1` |
  | Proceso vivo real con el valor en **2** | **ROJO**, exit 1 |
  | Proceso vivo real con el valor en **1** | **VERDE**, exit 0 |

  **Lo que el job de CI NO hace, dicho para que nadie lo lea de más.** No
  verifica el invariante en producción, y no puede: los once workflows corren
  en `ubuntu-latest`, ahí no hay ningún Ollama, y el de hall9000 corre como
  usuario `ollama` con `/proc/<pid>/environ` en 0400 — leerlo **exige root**.
  Lo que CI verifica es que **el tripwire funciona**, apuntándolo a procesos
  vivos de verdad lanzados con el valor puesto a mano: mismo camino de lectura
  de `/proc`, mismo parser, mismo veredicto; lo único distinto es cuál es el
  proceso. 33 tests con piso exacto, más un paso que exige que el tripwire dé
  **rojo** en un runner sin Ollama — si ahí diera verde, sería fail-open.
  **El chequeo contra el Ollama real es de hall9000 y es manual:**
  `sudo python3 scripts/check_ollama_num_parallel.py`.

  **Referencias de línea corregidas de paso, y la causa de fondo anotada.**
  Las tres estaban mal: el semáforo se citaba como
  `jax/muscles/ollama_muscle.py:37` cuando estaba en la 49 (y tras el cambio
  de hoy quedó en la 59), y los comentarios cruzados como `jacobs/plan.py:374`
  y `jacobs/executor.py:117,381` cuando están en la 645 y la 351. **No se
  arreglaron poniendo los números nuevos: se cambiaron a referencia por
  símbolo** (`ollama_muscle.py::GPU_SEMAPHORE`), porque el número de línea se
  pudre solo y volver a escribirlo sólo posterga el mismo defecto. También se
  corrigió el comentario del propio `ollama_muscle.py`, que decía "ese valor
  no lo fija nadie explícitamente" — falso: la unidad lo declara.

  **Qué lo reabre:** el tripwire en rojo. Ahí la decisión hay que rehacerla
  entera, no ajustarla. Condiciones de validez completas en
  `docs/superpowers/specs/2026-08-25-gpu-concurrency-resultado.md`.


- **`capability.sandbox_only` — columna sin lector, vestigial.** Verificado
  2026-08-27: `grep -rn "cap.sandbox_only\|capability.sandbox_only\|entry\[.sandbox_only.\]\|entry.get(.sandbox_only"` → 0 resultados en todo el repo. Las 5 filas
  con valor `1` (research/analysis/design/reconcile/validate_consistency)
  nunca se comparan contra nada. El único `sandbox_only` real es
  `motor.sandbox_only` (`policy.py` check 7), una columna DISTINTA, de la
  tabla `motor`. No se le inventa semántica esta ronda (el candidato obvio,
  egress de red, es el ítem "Hyde: red sin acotar por dominio/IP" ya
  diferido a propósito). Pendiente: darle lector real o dropearla.

- **`research`/`analysis`/`review` (capabilities HTTP-directo) existían en
  PROD pero en NINGUNA migración — CERRADO 2026-08-27 (jax-platform,
  commit `27cb73c`).** Segunda instancia confirmada del mismo patrón que
  `depends_on` (ver la entrada "14 (en verdad 18) tests" más abajo: fila o
  columna agregada a mano en producción, nunca registrada en el camino de
  migración idempotente) — ya no es un caso aislado, es un patrón.
  Verificado durante el cierre de gobernanza de `_HTTP_FACETS` (Task 6):
  `_CAPABILITY_SEED` en `jax-platform/backend/db/migrations.py` tenía 12
  entradas, ninguna de las tres; `jax_memory` (prod) tenía las tres,
  sembradas a mano en algún momento (`allowed_callers=["jacobs"]`);
  `jax_memory_test` tenía cero. Consecuencia real, no solo teórica:
  ninguna DB recién creada (test, dev, restore de desastre) las recibía
  jamás, y eso bloqueaba en silencio a los tests de gobernanza de poder
  ejercitar el par canónico `hipatia`/`research`. Arreglado agregando las
  tres a `_CAPABILITY_SEED` con los valores exactos de prod
  (`sandbox_only=1`, `requires_human_gate=0`, `max_execution_minutes=5`,
  `allowed_callers=["jacobs"]`, cero filas `capability_motor` — HTTP-directo
  puro; `risk_level=low` para research/analysis, `medium` para review),
  verificado en vivo contra `jax_memory_test` post-fix (las tres presentes,
  prod sin cambios — `INSERT IGNORE` no-opea donde la fila ya existe).
  **Nota de patrón, vale más que cualquiera de las dos instancias por
  separado:** dos bugs de la misma clase exacta (ALTER/INSERT manual a
  producción, nunca incorporado a la migración idempotente) encontrados en
  un mes. Si aparece una tercera instancia, es momento de escribir un
  chequeo automatizado que compare el vocabulario real de prod contra lo
  que la migración sembraría en una DB fresca, en vez de seguir
  encontrándolas una por una durante trabajo no relacionado.

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
  ambas copias se detecta con `scripts/check_mirror_sync.py` (era
  `check_facet_resolver_sync.py`; renombrado el 2026-09-01 al generalizarse)
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

  **Ampliación 2026-08-27: el check SÍ lee `context_window` y SÍ maneja
  NULL bien, pero está INERTE — verificado, no supuesto.** Se investigó a
  raíz de que `glm-5.3` (el modelo de `ada`) tiene `context_window = NULL`
  en producción. Hallazgos:
  - **Hay lector**, uno solo: `_capability_check()` en
    `backend/api/admin/facet_bindings.py` — no es columna sin lector.
  - **Trata NULL con honestidad**, no asume: `if context_window is None and
    min_context_tokens > 0: return "unknown"`. Devuelve "unknown", no "ok".
  - **Pero esa rama es inalcanzable hoy:** las 7 facetas tienen
    `min_context_tokens = 0` (verificado por SELECT), así que la condición
    `> 0` nunca se cumple y el flujo cae a `"ok"`. El badge "✓ Meets" de
    `ada` está pasando un check que **nunca miró nada sobre contexto**.
  - Son 3 modelos con `context_window` NULL, no solo el de `ada`:
    `glm-5.3` (ada), `claude-opus-5` (hyde), `qwen3.6:35b-a3b-q4_K_M`
    (jax_local).

  **No es el perfil de `thot`** (un campo consumido por un valor
  hardcodeado que rompió). Es un tercer perfil, y vale distinguirlo: dato
  faltante + guard correcto + umbral que desactiva el guard. Nada se rompe
  hoy; lo que se pierde es la señal — nadie se va a enterar de que a 3
  modelos les falta el dato, porque el único lugar que lo miraría dice
  "ok". Si alguien pone `min_context_tokens > 0` en una faceta, esas 3
  pasan a "unknown" de golpe y va a parecer una regresión nueva.

- **`api/chat.py` acopla el backend a `~/jax` en import time —
  QUÉ SE PIERDE cuando no está (2026-08-27).** `backend/api/chat.py` hace
  `sys.path.insert(0, ~/jax)` e importa `MemoryDB` del OTRO repo. Si esa
  ruta no existe, `MemoryDB` queda `None` y `_ensure_memory()` corta con
  `return False`.

  **"Degrada elegante" es precisamente cómo se ve un fail-open desde
  afuera, así que queda escrito qué se pierde, no solo que no revienta:**
  el chat sigue respondiendo, pero **sin memoria semántica** — no lee ni
  escribe contra `jax_memory` (la misma DB que usa el REPL, ver
  `jax-memoria-semantica-dos-niveles` en memoria), y `_get_conv_uuid()`
  devuelve `None`, así que los turnos dejan de agruparse en conversaciones
  identificables. Un usuario no ve un error: ve un asistente que no
  recuerda nada de sesiones previas, indistinguible de uno que sí tiene
  memoria pero no encontró contexto relevante.

  Descubierto porque 2 tests pasaban en local y fallaban en el runner de CI
  (parcheaban `_memory`/`_memory_ready` pero no `MemoryDB`, y el
  cortocircuito ocurre antes de mirar los parches). Reproducido cambiando
  SOLO `HOME`. Arreglo de fondo, no hecho: volverlo una variable
  (`JAX_CORE_PATH`) en vez de una ruta implícita del home.

En memoria de Jairo Urbina.

- **Certificados del origen — diagnóstico MEDIDO 2026-09-01.**

  | | Borde (lo que ve el visitante) | Origen (`172.16.20.20`) |
  |---|---|---|
  | Certificado | **válido**, Google Trust Services | **VENCIDO** — Let's Encrypt R11 |
  | Dominio A | vence 2026-11-24 | venció **2025-05-03** |
  | Dominio B | vence 2026-11-19 | venció **2025-05-22** |
  | HTTP | 200 | 200 en `:80` y en `:443` |

  **Por qué falla ACME, y no es lo que decía el diagnóstico viejo:** el
  challenge `http-01` lo resuelve Let's Encrypt **desde internet**, y el
  origen está en una **IP privada** (`172.16.20.20`) que no es enrutable desde
  afuera. La petición llega al borde de Cloudflare, no al origen. **No es un
  puerto cerrado ni un firewall: es que el origen no es alcanzable por
  diseño.** Medido: desde la red interna el origen sí responde en `:80`.

  **Modo SSL de Cloudflare: NO DETERMINABLE desde afuera.** No hay token de
  Cloudflare en el host. `Full (strict)` **queda descartado por medición** —
  con un certificado vencido el borde devolvería `526` y devuelve `200`.
  Quedan `Full` y `Flexible`, y el origen responde 200 en ambos puertos, así
  que lo observable no los distingue. **Hay que mirarlo en el panel.**

  **Y la distinción importa:** si el modo fuera **`Flexible`**, el tramo
  Cloudflare→origen viaja **sin cifrar**, aunque el visitante vea el candado.
  Eso sí sería un problema real, y no lo cubre ningún certificado del borde.

  **Corrección propuesta, una sola para los dos dominios: certificado Origin
  CA de Cloudflare + modo `Full (strict)`.** Es gratuito, dura 15 años, está
  diseñado exactamente para orígenes detrás del proxy, y **elimina el problema
  de renovación de raíz** — no necesita ACME, así que la IP privada deja de
  importar. `Full (strict)` además obliga a que el tramo interno sea cifrado y
  validado. La alternativa —cambiar Hestia a challenge `DNS-01`— también
  funciona pero conserva la renovación periódica y su modo de fallo.

  **No se cambió nada.** Requiere tu autorización: son dominios de clientes.
