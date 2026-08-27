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

- **No existe ninguna señal que avise cuando un facet deja de responder
  (2026-08-27).** El ítem operativo más importante que dejó esta ronda, y el
  complemento de la lección de método de CONTEXT.md ("verificar el estado
  desplegado antes de declarar cerrado").

  **Qué señal existe hoy: NINGUNA.** No es "existe pero es débil". Verificado
  al diagnosticar la caída de `thot`:
  - `/health` de `jax-platform` reporta `las_manos: alive|down` — la
    salud del *servicio*, no la de cada facet. Con `las_manos` arriba y los
    4 facets caídos, `/health` dice que todo está bien.
  - `facet.status` (`active`/`degraded`/`disabled`) es un campo de
    configuración: lo escribe un humano o un admin, nadie lo deriva de que
    las llamadas estén fallando. Hoy `thot` figura `active` estando roto.
  - `model.status` (`available`/`degraded`/...) lo mueve el sync del
    catálogo contra `/v1/models` del proveedor — dice si el proveedor
    todavía OFRECE el modelo, no si nuestras llamadas a él funcionan. El
    modelo de `thot` existe y responde: lo que falla es el contrato de
    parámetros de NUESTRA llamada. Ese sync jamás lo vería.
  - Los errores por turno de chat se propagan al usuario y quedan en
    `journalctl`. Nada los cuenta, los agrega ni los alerta.

  **Consecuencia medida, no hipotética:** `thot` estuvo **3 días** caído en
  la Mesa web (roto el 2026-08-24 al rebindearse a `gpt-5.6-terra`,
  descubierto el 2026-08-27) y solo se detectó porque el despliegue de otra
  cosa incluyó un paso de verificación manual en el chat real. Sin ese paso
  seguiría caído.

  **Por qué BLOQUEA:** "verificar el estado desplegado antes de declarar
  cerrado" es una regla que hoy solo se cumple cuando alguien se acuerda de
  mirar. Todo lo que se cerró esta semana — la gobernanza de `_HTTP_FACETS`,
  el gate fail-closed de Mesa web, el sandbox de Hyde — puede romperse en
  silencio exactamente igual que `thot`, y el modo de falla de un gate
  fail-closed es *denegar todo*: se vería idéntico a "el facet no responde".
  Sin detección, un fail-closed que se dispara por error es indistinguible
  de uno que funciona.

  **Qué haría falta (no diseñado todavía, es la próxima ronda):** una señal
  derivada del tráfico real, no de configuración — algo que cuente
  éxito/fallo por facet en la ventana reciente y alerte cuando un facet que
  venía respondiendo deja de hacerlo. Los dos chokepoints de salida ya
  existen y son únicos (`_invoke_facet` en Mesa web, `_dispatch_step` en
  Jacobs), así que el lugar donde instrumentar no es la incógnita; la
  incógnita es dónde vive el estado, cuál es el umbral y por qué canal
  avisa. **Primer ítem de la próxima ronda, por decisión explícita de
  Fernando.**

- **`facet_resolver._cache` replicado en TRES procesos; solo uno se invalida
  al rebindear (2026-08-27).** `facet_resolver.py` está espejado a propósito
  en `jax-platform/backend`, `jax/core` y `jax/las_manos` — el patrón
  declarado de "sin paquete compartido", cada repo con su conector mínimo.
  Cada espejo tiene **su propio `_cache` en su propio proceso**, con
  `FACET_CACHE_TTL_SECONDS` (default 30 s).

  La Task 5 de la ronda de alertas agregó `invalidate_facet_cache()` y la
  llama desde `probe_after_rebind`, así que tras aprobar un binding el
  proceso de `jax-platform` ve el modelo nuevo de inmediato. **Los otros dos
  no.** Jacobs (`las_manos`) y el REPL (`jax/core`) pueden seguir
  despachando contra el binding **viejo** hasta 30 s después del rebind, y
  la sonda no lo puede ver: sondea por el camino de Mesa web, que es el
  único invalidado.

  **Por qué importa y no es teórico:** es el mismo estado replicado en tres
  lugares con un solo escritor de invalidación — la forma exacta que ya
  produjo el incidente de `motor.model_ref` (2026-08-19 y de nuevo el
  08-24, documentado en el docstring de `approve_proposal`) y las 4 fuentes
  de verdad de capabilities que el Bloque 3 tuvo que colapsar. Ventana
  chica (30 s) pero determinística, y justo en el momento de mayor riesgo:
  inmediatamente después de un cambio de modelo.

  **Es un punto ciego del detector, no solo un problema de frescura.** La
  sonda por rebinding **no puede ver ese estado**: sondea por el camino de
  Mesa web (`_invoke_facet`), que es el único de los tres procesos cuya
  caché se invalida. Si Jacobs o el REPL despachan contra el binding viejo
  durante esos 30 s, la sonda reporta `ok` — y reporta la verdad *de su
  propio camino*. El instrumento que esta ronda construyó para que un facet
  roto no pase 3 días sin avisar tiene, por construcción, un camino que no
  observa. Anotar la ventana "donde el operador la vea" no alcanza para
  esto: el operador miraría un tablero verde que es correcto y aun así
  incompleto.

  **Qué haría falta (no diseñado):** una señal de invalidación entre
  procesos, o bajar el TTL a costa de más queries, o aceptar la ventana
  explícitamente y documentarla donde el operador la vea — cualquiera de
  las tres deja el punto ciego abierto salvo la primera. No se resuelve
  hoy — queda con el caso concreto. Descubierto por la revisión de la
  Task 5, no buscado.

- **Bypass de admin en el ruleset de `master`: la única barrera contra un
  merge en rojo es que alguien se acuerde de mirar (2026-08-27).** Los dos
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

  **Qué haría falta (no diseñado):** una identidad de bot separada, sin
  bypass, que sea la que abre y mergea PRs, dejando el bypass de admin
  como escape manual y explícito de Fernando. No se resuelve hoy; queda
  con el caso concreto para que la próxima ronda no tenga que reconstruir
  el argumento. Ver la séptima lección de método en CONTEXT.md ("verificar
  y actuar en el mismo comando no es un gate") — el corolario operativo
  mitiga el lado del agente, pero no cierra el agujero de infraestructura.

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

- **`_CAPABILITY_TIMEOUT_SECONDS` (jacobs/plan.py) duplica
  `capability.max_execution_minutes` (DB) sin lectura en vivo.** Verificado
  2026-08-27 durante el cierre de gobernanza de `_HTTP_FACETS`: el default
  de `step.timeout_seconds` sale de un dict hardcodeado
  (`jacobs/plan.py:106-110`); un `timeout_seconds` explícito en un spec de
  step lo pisa SIN validar contra el techo de la DB
  (`_validate_plan_capabilities` no lo chequea, y ni siquiera aplica a
  `_HTTP_FACETS`). `scripts/check_timeout_consistency.py` verifica que
  coincidan, pero es manual, no una garantía en runtime. El check 8 real de
  `MotorPolicy.check()` solo corre server-side, dentro de `las_manos`
  (`las_manos/motor_registry/routes.py:95`, antes de crear el `MotorJob`
  pero después de que Jacobs ya hizo la llamada HTTP a `/motor/dispatch`),
  y solo para `kimi`/`jax_local` -- nunca corre en Jacobs, ni antes de que
  Jacobs decida despachar. Decisión
  explícita: NO se activa un admission-check contra esto en la ronda de
  `_HTTP_FACETS` (validaría contra un valor que puede ya estar desincronizado
  del que el ejecutor real usa) -- se resuelve junto con la deduplicación,
  en una ronda aparte.

- **`jax-platform`: suite de tests del backend con fallos preexistentes en
  este entorno de desarrollo.** Verificado 2026-08-27
  durante el cierre de gobernanza de `_HTTP_FACETS` (Task 7, integración de
  `_invoke_facet` en `chat.py`): diff contra un stash-baseline muestra
  EXACTAMENTE el mismo conjunto de failures, por nombre, con y sin los
  cambios de esta ronda -- no es una regresión de este trabajo.
  **EL NÚMERO NO ES ESTABLE — corregido dos veces el mismo día antes de
  entender por qué (2026-08-27).** Historia completa, porque la lección
  está en la secuencia y no en la cifra: una medición dio 10, otra 12, una
  tercera (a mano, dos corridas seguidas) volvió a dar 10 y se escribió acá
  como "el número correcto". **Las tres estaban midiendo algo inestable.**
  Al portar el CI se corrió el experimento que faltaba: el MISMO árbol
  limpio da 10 y después 12 (3 corridas de cada resultado). La causa es la
  misma que la de los fallos: `jax_memory_test` es una DB COMPARTIDA, y el
  conteo depende del estado en que la dejó la corrida anterior.
  **Consecuencia práctica:** cualquier criterio de "el conjunto de fallos
  no cambió" basado en el NÚMERO es inservible acá — hay que comparar
  NOMBRES. Y descarta por construcción la opción de versionar un baseline,
  que era el diseño más obvio para meter esta suite en CI. Causa raíz identificada: un
  pool de conexiones `aiomysql` reusado entre distintos event loops de
  `asyncio` -- defecto de aislamiento de tests, no defecto de producto.
  Evidencia directa de eso, no solo inferencia: **los failures y el
  error pasan TODOS en verde cuando se corren sus archivos por separado**
  (verificado 2026-08-27, archivo por archivo) -- solo fallan cuando
  comparten proceso con el resto de la suite. No existe hoy, para
  `jax-platform`, un equivalente
  al cierre que `jax`/`las_manos` logró 2026-08-24 (95 passed / 0 failed,
  ver la entrada "14 (en verdad 18) tests" más abajo) -- ninguna auditoría
  llevó esa suite a verde de referencia. Se deja abierto a propósito, no
  como "ruido conocido, ignorar": una suite permanentemente un poco roja
  entrena a ignorarla, que es exactamente cómo una regresión real termina
  escondida en el ruido. Sin fix esta ronda (fuera de alcance de la
  gobernanza de `_HTTP_FACETS`) -- candidato a sesión de pago de deuda
  propia.

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
  reales fuera de alcance.** Declarado explícitamente como no resuelto al
  cerrar el sandbox de bubblewrap (2026-08-23) — el contenimiento
  principal (secretos, filesystem, hooks) sí está cerrado, estos son
  refinamientos de defensa en profundidad pendientes. **La tercera parte
  original de este bullet ("concurrencia de `HYDE_SEMAPHORE` con el
  sandbox no reverificada") ya no aplica — CERRADO 2026-08-26 (PRs
  jax#33-36).** `HYDE_SEMAPHORE` (`asyncio.Semaphore`) se eliminó del
  código (confirmado por grep: solo queda en comentarios/docstrings como
  referencia histórica), reemplazado por `flock(2)` cross-proceso en
  `hyde_sandbox.py::run_sandboxed_claude()` — ver
  `jax-claude-subprocess-gobernanza-cerrado` en memoria.

## Anotado, no bloquea

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
