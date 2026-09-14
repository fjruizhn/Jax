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

- **El resolver de `CAPABILITY_AVAILABLE` consulta un catálogo que el Bloque 3
  vació — verificado 2026-09-02.** **Causa:** el Bloque 3 movió las
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

  **Precedente de la misma forma:** `_PROVIDER_ENV_KEY_MAP` en
  `credential_resolver` —el otro mapa proveedor→variable de entorno—, señalado
  el mismo día como el símbolo cuyo drift produciría exactamente ese síntoma. Es
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

- **Anotados en la etapa 1 de admin usuarios (2026-09-13).**
  - **Dependencia de la etapa 4:** `_procesar_recuperacion` se traga todo y
    devuelve `None`, así que el "enlace de restablecimiento" por admin de la
    etapa 4 no puede reusarlo para devolver 503 si falta SMTP (spec §3.4). La
    etapa 4 tiene que llamar `smtp_config.cargar_settings()` y
    `_send_reset_email` (que SÍ lanza) por su cuenta.
  - **`DeprecationWarning` de `datetime.utcnow()`** en `jax_engine/schemas.py` y
    `state.py`: ruido preexistente en toda corrida de pytest. Lo reabre Python
    3.15 o una limpieza de warnings.
  - **test-connection devuelve el banner del servidor** remoto al superadmin:
    sirve como sondeo de puertos internos. Se acepta porque es solo superadmin y
    está limitado (`JAX_SMTP_CONN_RATE`). Lo reabre que el endpoint deje de ser
    exclusivo de superadmin.
  - **La pantalla de Configuración traga los errores del PUT `/api/admin/config`**:
    no hay texto para `config_clave_reservada` ni para
    `config_collation_desconocida`, y quien intenta guardar una clave reservada
    no ve por qué no se guardó. Es preexistente. Lo reabre el próximo cambio en
    AdminSettings.
  - **Contraste de textos secundarios en modo oscuro:** `text-slate-500` sobre el
    fondo oscuro mide 3,75:1, por debajo del AA de 4,5 (medido en AdminSmtp el
    2026-09-13). Es la convención de todas las pantallas. Arreglarlo es una
    decisión del sistema de diseño, no de una pantalla.
  - **Cancelación durante el rollback de `smtp_config.guardar_filas`:** si la tarea
    se cancela justo en el rollback, se loguea el `CancelledError` en vez del error
    original de la base. La conexión no vuelve sucia al pool: `Pool.release` de
    aiomysql cierra las que quedan a mitad de transacción. No se cambió porque
    atrapar `BaseException` en la limpieza arriesga tragarse cancelaciones. Lo
    reabre ver ese caso en un log real.
  - **Pares de contraste por debajo de 4,5 que hoy no se usan:** `#b91c1c` sobre
    `#fecaca` da 4,47 y el verde sobre `#e2e8f0` da 4,07. El test de modo claro
    exige que exista un override, no que el contraste alcance. Se reabre si una
    pantalla combina `text-red-400` con `hover:bg-red-900` o pone texto verde
    sobre `bg-slate-700`.
  - **Fuente Inter** marcada como "sobreusada" por el hook de impeccable en
    `frontend/src/index.css:8`: preexistente, pendiente de decisión de Fernando.

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
  - **Las fuentes se deduplican por la redirección de Google, no por la URL
    final.** Dos redirecciones distintas que resuelven al mismo documento salen
    como dos fuentes (visto en vivo, pipeline `04e02b09`: `[1]` y `[2]` con la
    misma URL). Menor: no esconde nada, repite. Se arregla deduplicando después
    de `resolve_redirects` y fusionando las citas.
  - **El frontend no tiene tema claro/oscuro en ninguna pantalla.** Medido: ni
    variables CSS en `src/index.css`, ni una clase `dark:`, ni `darkMode` en
    Tailwind; todo es `slate-*` y hex fijos. Incumple la política "Dark/Light
    mode — SIEMPRE" en toda la app, no en un componente. La cadena siguió las
    clases del modal; arreglarlo es una ronda propia de tokens de diseño.
  - **vitest no corre en el CI de jax-platform.** 57 tests del frontend solo se
    verifican a mano. Qué lo cierra: un job con piso exacto, como el backend.
  - **Una cadena en modo `supervised` pide una aprobación por paso.**
    `supervised` corre UNA ola y pausa (`executor.py:966`); en paralelo eso era
    una sola pausa, en cadena son cinco. El botón "Aprobar" de `RightPanel`
    reanuda. **Decidido el mismo día por Fernando: la cadena va en
    `autonomous` por defecto** (jax-platform#55). `supervised` sigue
    disponible a mano, con su pausa por paso.
  - **Aprobar en `RightPanel` traga el error.** `handleResume` solo hace
    `console.error`: si `/resume` falla, en la interfaz no pasa nada.
  - **Cancelar kimi corta nuestro lado, no necesariamente la facturación.**
    Kimi es cloud: cerrar la conexión no está verificado que detenga lo que
    Moonshot ya estaba generando (blueprint de Ricardo §11, mismo límite). Lo
    garantizado es que no hay segunda llamada.
  - **`finish_reason=length` sin schema sigue marcando `completed`.** Decisión
    del 2026-08-10 (`_worker_max_tokens_test.py`: "el dato queda para
    diagnóstico, no bloquea"), no tocada en esta ronda: el arreglo solo cubre
    el camino con schema. Se reabre si una salida cortada sin schema llega a
    un consumidor como si estuviera completa.
  - **El nombre del pipeline lleva `Pipeline: ` fijo en `PipelineModal.jsx`**
    (dato guardado, no texto de interfaz). Menor; se va con la próxima pasada
    de i18n del modal.

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
  la puerta de `ssh_exec` — 2026-09-04, causa medida.** En el corrido dirigido
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

