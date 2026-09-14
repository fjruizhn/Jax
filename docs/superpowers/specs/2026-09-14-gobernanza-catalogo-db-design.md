# Gobernanza con el catálogo de la DB, e `invoked_by` como rol — diseño (tanda A)

**Fecha:** 2026-09-14 · **Aprobado por:** Fernando (diseño en chat, 2026-09-14 ~14:30) ·
**Enmendado:** 2026-09-14 (tarde), por decisión de Fernando — ver §0 ·
**Repos:** `jax` (validador, grounding, catálogo, Jacobs) y `jax-platform` (migración, contexto de
gobernanza, API de pipelines).

## 0. Historial de este documento

| Versión | Commit | Qué cambió |
|---|---|---|
| v1 | `0e44acd` | Diseño original: "solo el resolver", snapshot sin cambios. |
| v2 | este commit | Enmienda. La planificación (plan `0c33478`, "Tarea 0") midió que con v1 el arreglo **no tenía efecto visible**: en la Mesa un claim sobre una capability de la DB nunca llega al resolver, porque la acreditación lo corta antes. Fernando (2026-09-14): *"Arréglalo completo, nada para más adelante"*. Se revierte la decisión "snapshot sin cambios", se agrega la columna `capability.mode` y el resolver pasa a verificar el modo también en el catálogo. Lo corregido se marca **CORREGIDO** o **REVERTIDA**; el texto de v1 queda a la vista donde importa. |

## 1. Por qué

Dos defectos de `DEUDA.md` ("Bloquea trabajo"):

1. **El resolver de `CAPABILITY_AVAILABLE` consulta un catálogo vacío.** El Bloque 3 movió las
   capabilities a la DB (`MotorCatalog.from_db()`), pero `policy/governance/validator.py::load_validation_context()`
   sigue armando `MotorCatalog(config)` desde `las_manos/config.toml`, cuyo `[capabilities.*]` está
   vacío. En producción lo usa `jax-platform/backend/governance_context.py` (validación en sombra de la
   Mesa). El tripwire
   `test_real_toml_catalog_is_empty_since_block3_so_catalog_branch_is_dead_in_production` fija ese estado.

   > **v1 decía** que un claim `INFERIDO` sobre una capability que existe solo en la DB (`generate`,
   > `file_write`...) "sale `FACT_MISMATCH` — un falso negativo".
   >
   > **CORREGIDO (2026-09-14, medido):** eso nunca se midió y es falso. En producción,
   > `shadow_claim_verdicts` para `CAPABILITY_AVAILABLE` tiene: `OBSERVADO/VALID` 427,
   > `INFERIDO/POINTER_MISMATCH` 3, `INFERIDO/ARGS_MISMATCH` 1 y `NULL/AUTHORITY_INVALID` 2 (los 2 sobre
   > `code_swarm`). **Cero `FACT_MISMATCH`.** El defecto real es otro y es más grave:
   >
   > - **Una capability de la DB no se puede citar.** `grounding.build_snapshot()` arma el snapshot solo
   >   con `ctx.ops`. Sin una línea que citar, `grounding.accredit()` da `NO_POINTER`,
   >   `POINTER_MISMATCH` o `FACT_NOT_IN_SNAPSHOT`, y `validator.validate()` corta en el paso 4 o 5,
   >   **antes** del resolver. Los 2 `AUTHORITY_INVALID` sobre `code_swarm` son facetas que intentaron
   >   hablar de una capability real y no tenían con qué.
   > - **La rama `in_catalog` del resolver es inalcanzable** desde la Mesa. Arreglar solo el catálogo
   >   del resolver (v1) no cambiaba ningún veredicto en producción.

2. **`invoked_by` es un campo de autorización con un nombre de persona como rol.** `jacobs/routes.py:100`
   acepta solo `{"Fernando", "jax_local", "ada"}` y `policy.validate_resume` exige `"Fernando"`;
   jax-platform lo manda fijo (`PipelineModal.jsx`, `api/pipelines.py:161`).

## 2. Decisiones (Fernando, 2026-09-14)

**De v1, vigentes:**

- **`invoked_by` pasa a ser el rol `plataforma`**: "pedido de jax-platform en nombre de un usuario
  autenticado". La identidad sigue viajando en `user_id`/`tenant_id`.
- El validador sigue **puro** y **recibe** el catálogo. `governance_context.validation_context()` es async
  con caché por mtimes + sello, recarga bajo `asyncio.Lock` con `MotorCatalog.from_db()` y falla visible
  (nunca catálogo vacío). Tripwire ops ∩ DB vacío. Medición de latencia. Despliegue con la ventana de
  segundos anunciada.

**De v1, REVERTIDA:**

- ~~**Solo el resolver.** El snapshot que se inyecta en el prompt (SP3) NO cambia: sigue listando
  `ctx.ops`.~~ → **REVERTIDA 2026-09-14 (tarde).** Con el snapshot sin cambios el arreglo no tenía efecto
  visible (ver §1).

**Nuevas (v2, 2026-09-14 tarde):**

1. **"Arréglalo completo, nada para más adelante."** El snapshot inyectado **lista también las
   capabilities de la DB**. Las facetas pueden citarlas como `OBSERVADO`; acreditación y resolver las ven.
2. **Fuente del modo: columna nueva `capability.mode ENUM('read_only','mutating')`**, por migración
   idempotente en `jax-platform/backend/db/migrations.py`. Sembrada: `mutating` **solo** para `file_write`
   (la única capability que cambia el estado del sistema); todas las demás (`generate`, `design`,
   `refactor`, `implementation`, `code_swarm`, `file_read`, ...) `read_only`: producen texto o parches sin
   aplicarlos.
3. **`MotorCatalog`/`CapabilityEntry` llevan `mode`**, y el resolver verifica el modo también para las
   capabilities del catálogo (`FACT_MISMATCH` si no coincide). Se acaba el "mode no verificable ahí".
4. **Consecuencia aceptada — SP4 deja de ser comparable.** El snapshot crece (§4), así que la línea base de
   la sonda dirigida de SP4 (2026-09-03, `docs/superpowers/specs/2026-09-03-sp4-citacion-dirigida-design.md`)
   ya no es directamente comparable con corridas posteriores: el mecanismo que mide cambió. Al retomar
   SP4 hace falta una línea base nueva, con su propio pre-registro. Se declara acá, no se descubre después.

## 3. Diseño

### 3.1 La columna `capability.mode` (jax-platform)

- **DDL.** `CREATE_CAPABILITY` gana `mode ENUM('read_only','mutating') NOT NULL`, **sin `DEFAULT`**.
- **Semilla, una sola fuente.** Un dict `_CAPABILITY_MODE` en `migrations.py` con las 17 capabilities que
  siembran las migraciones (`_CAPABILITY_SEED`, 15, + `_FILE_CAPABILITY_SEED`, 2, que hoy es una lista
  local de `_seed_file_tools_capabilities` y pasa a constante de módulo): `file_write` → `mutating`, el
  resto → `read_only`. Los `INSERT IGNORE INTO capability` pasan `mode` explícito, leído de ese dict.
- **Bases que ya existen** (producción), en `run_migrations()`, idempotente:
  1. `_COLUMNS`: `ALTER TABLE capability ADD COLUMN mode ENUM('read_only','mutating') NULL` si la columna
     no existe. Nace **NULL a propósito**: un `ADD COLUMN ... NOT NULL` sin default le pone a las filas
     existentes el primer valor del ENUM (`read_only`) — `file_write` quedaría `read_only` en silencio
     hasta el `UPDATE`, y el DDL de MariaDB hace commit implícito.
  2. Después de las semillas, `_backfill_capability_mode(cur)`: `UPDATE capability SET mode=%s WHERE
     `key`=%s AND mode IS NULL` para cada entrada de `_CAPABILITY_MODE`.
  3. `_enforce_capability_mode_not_null(cur)`: si quedan filas con `mode IS NULL` (una capability que no
     sembró ninguna migración), **lanza `RuntimeError` con sus nombres**: la migración falla y
     jax-platform no arranca. Si no quedan y la columna todavía admite NULL,
     `ALTER TABLE capability MODIFY COLUMN mode ENUM('read_only','mutating') NOT NULL`.
- **El default: `NOT NULL` sin `DEFAULT` (decisión de este diseño, con su porqué).**
  - Un `DEFAULT 'read_only'` es fail-open: la próxima capability que cambie estado nacería declarada de
    solo lectura, y el resolver le daría `VALID` a una faceta que diga que no muta nada.
  - Un `DEFAULT 'mutating'` es fail-closed pero miente en el otro sentido y queda igual de invisible.
  - Sin default, **la base obliga a declararlo**: `sql_mode` de producción tiene `STRICT_TRANS_TABLES`
    (medido 2026-09-14, `@@GLOBAL.sql_mode`), así que un `INSERT` sin `mode` falla con el error 1364
    ("Field 'mode' doesn't have a default value"). Hoy ningún endpoint crea capabilities
    (`api/admin/motors.py` solo las **asocia** a un motor, y responde 400 si no existen; verificado
    2026-09-14): toda fila nace en las migraciones, que la declaran.
  - **Tripwires de CI** que lo sostienen: (a) puro — las claves de `_CAPABILITY_MODE` son exactamente las
    capabilities sembradas y solo `file_write` es `mutating`; (b) con DB — `SHOW COLUMNS` da
    `Null=NO` y `Default=NULL`, y un `INSERT` sin `mode` falla. Ese último es a la vez el CONTROL de la
    decisión: si alguien le pone un default, se pone rojo.
- **Invalidación.** `run_migrations()` ya estampa el sello de `facet_resolver` después del commit: LAS
  MANOS y `governance_context` recargan su catálogo.

### 3.2 `MotorCatalog` lleva el modo (jax, `las_manos/motor_registry/catalog.py`)

- `CAPABILITY_MODES = frozenset({"read_only", "mutating"})`.
- `CapabilityEntry.mode: str = "mutating"`. El default solo lo usa el constructor por dict (tests) y es
  **fail-closed**, en la línea de los otros defaults de ese constructor (`risk_level="high"`,
  `requires_human_gate=True`): un claim de solo lectura sobre una capability sin modo declarado sale
  `FACT_MISMATCH`, nunca `VALID`.
- `from_db()` lee `mode` en el `SELECT` de `capability` y lo pasa explícito. Un valor fuera de
  `CAPABILITY_MODES` (incluido `None`) **lanza `RuntimeError`** con el nombre de la capability: la base
  garantiza el ENUM y el NOT NULL, y si eso se rompe se ve.
- `MotorCatalog.capabilities() -> tuple[CapabilityEntry, ...]`, ordenado por nombre: la vista pública que
  usa `build_snapshot` (hoy solo existe `get_capability(name)`; nadie de afuera debería leer
  `_capabilities`).
- Con esto `jax` depende de que la columna exista: **la migración va primero** (§6).

### 3.3 Validador (jax, `policy/governance/validator.py`) — sigue puro

- `load_validation_context(repo_root, config_paths_allowlist, catalog)` **recibe** el `MotorCatalog`; deja
  de construirlo del TOML. El validador no toca la DB.
- `_resolve_capability_available`, rama `in_catalog`: compara `claim.args["mode"]` con `entry.mode`.
  Distinto → `FACT_MISMATCH` ("`'<name>'` tiene mode real `'<x>'` en el catálogo, el claim afirma
  `'<y>'`"). Igual → `VALID` ("verificado en catálogo de capabilities, mode=...").
- `SOURCE_CONFLICT` (nombre en `ops` y en el catálogo) no cambia y se evalúa primero.
- **Medido 2026-09-14:** 0 nombres en común entre `ops` del TOML (11) y `capability` de la DB (**17** —
  **CORREGIDO**: v1 decía 18; no existe `validate`), así que activar el catálogo no produce
  `SOURCE_CONFLICT`.
- Las dos fuentes del modo quedan separadas por construcción: para `ops`,
  `las_manos.envelope.MUTATING_CAPABILITIES`; para el catálogo, `capability.mode`. El tripwire ops ∩ DB
  vacío (§5) es lo que impide que un mismo nombre tenga dos modos.

### 3.4 Snapshot (jax, `policy/governance/grounding.py`)

- **Esquema de punteros: sección nueva, los punteros viejos no se mueven.**
  - `/capabilities/N` sigue siendo exactamente lo que es hoy: las `ops` del TOML ordenadas por nombre,
    con el modo de `MUTATING_CAPABILITIES`. Mismo contenido, mismos índices (`write_file` sigue en
    `/capabilities/10` en producción).
  - **Sección nueva `catalog_capabilities`**: `/catalog_capabilities/N`, las capabilities de
    `ctx.catalog.capabilities()` ordenadas por nombre, con `args = {"name", "mode"}` y el modo de
    `CapabilityEntry.mode`. `_POINTER_RE` (`^/([a-z_]+)/(0|[1-9][0-9]*)$`) ya la acepta.
  - Por qué una sección y no mezclar las dos listas en `/capabilities/N`: un solo orden mezclado
    renumeraría las `ops` cada vez que se agregue una capability a la DB, y todo test o análisis que
    cite `/capabilities/N` cambiaría de sentido sin aviso. Con dos secciones, lo existente es estable y
    lo nuevo se ve como nuevo.
- `SECTION_PREDICATE = {"capabilities": "CAPABILITY_AVAILABLE", "catalog_capabilities":
  "CAPABILITY_AVAILABLE"}`. El comentario "crece SOLO cuando un predicado gana resolver" se cumple: la
  sección nueva la re-resuelve la rama `in_catalog`, que ahora verifica nombre **y** modo.
- **Invariante de SP3 §3 ("todo hecho inyectado tiene quién lo re-resuelva")**: se sigue cumpliendo. SP3
  §3.3 decía que `build_snapshot` no debía tocar `ctx.catalog` *mientras la rama fuera código muerto*;
  con este diseño deja de serlo y verifica lo mismo que se inyecta. Se anota en SP3 §3.3 con fecha.
- `canonical_json` pasa a `{"capabilities": [...], "catalog_capabilities": [...]}`. La sección va
  **siempre**, aunque esté vacía (un catálogo vacío es una observación, no una ausencia). El `sha256` de
  todo snapshot cambia: es por turno y nadie lo compara entre versiones.
- `render()` imprime las dos secciones, cada una con su encabezado (`  capabilities:` y
  `  catalog_capabilities:`).
- Un error leyendo el catálogo dentro de `build_snapshot` entra en el mismo `try` y sale como
  `GroundingBuildError` (P10), igual que un `ctx.ops` roto.
- **Efecto sobre los tests existentes:** los que citan `/capabilities/N` no cambian (usan contextos con
  `MotorCatalog({})`: la sección nueva queda vacía y los índices de `ops` son los mismos). Cambian los que
  comparan el `canonical_json` literal, si existen; se buscan con `grep` en el plan.
- **La acreditación no cambia de lógica.** Un claim con el modo equivocado y un puntero correcto sigue
  saliendo `FACT_NOT_IN_SNAPSHOT` en la acreditación (ninguna entrada tiene esos args), **antes** del
  resolver. El `FACT_MISMATCH` del resolver aparece cuando el claim llega acreditado pero el catálogo
  cambió entre el snapshot del turno y la validación en sombra (se recarga por el sello), y en el camino
  sin acreditación (`validate(..., accreditation=None)` con `OBSERVADO`).

### 3.5 jax-platform — `governance_context.validation_context()` async

- Pasa a `async def validation_context()`. Llamadores: `api/chat.py` (`_build_snapshot_or_raise` /
  `_build_grounding`, llamado en `chat()` línea ~1115 → `await`) y `shadow_validation.py:315` (`await`).
- **Clave de caché:** los mtimes de los 3 archivos de config (como hoy) **+** el mtime del sello de
  `facet_resolver` (`_seal_mtime()`), que ya estampan las migraciones y el admin de motores y
  capabilities. Si la clave cambió → recarga; si no → el mismo objeto.
- **Recarga:** `catalog = await MotorCatalog.from_db()` y `load_validation_context(..., catalog)`, bajo un
  `asyncio.Lock` (N turnos a la vez no disparan N recargas) — el mismo patrón que LAS MANOS
  (`motor_registry/routes.py::_ensure_catalog_fresh`). La lectura de YAML/TOML va en `asyncio.to_thread`.
- **Falla visible, nunca catálogo vacío:** si la DB no responde, la excepción sube. En el chat,
  `_build_grounding` ya la convierte en `SnapshotError` → `grounding_snapshot_sha256='ERROR'`; la validación
  en sombra no escribe veredictos (hoy igual). Servir un catálogo vacío o viejo repetiría el falso negativo
  en silencio, o daría `VALID` a una capability revocada (P10).
- Snapshot y validación leen el **mismo** `validation_context()`: el snapshot del turno y el contexto de
  la validación coinciden salvo que el sello cambie entre los dos (ver §3.4, último punto).
- `validation_context.cache_clear` se conserva para los tests.

### 3.6 `invoked_by` = `plataforma`

- **Jacobs (jax):** `routes.py` y `policy.VALID_INVOKERS` (`jacobs/models.py:51`) aceptan
  `{"plataforma", "jax_local", "ada"}`; `validate_resume` exige `"plataforma"`. `create` lo valida
  `PipelineCreateRequest`.
- **jax-platform:** `api/pipelines.py` pone `invoked_by = "plataforma"` en el backend al crear (pisa lo
  que mande el cliente, igual que ya pisa `user_id`/`tenant_id`) y al reanudar. `PipelineModal.jsx` deja de
  mandarlo.
- Las filas viejas de `jacobs_pipelines` con `"Fernando"` quedan como están: son historia.
- Los tests que fabrican `Pipeline(invoked_by="Fernando")` sin pasar por la validación no cambian; los que
  pasan por `routes`/`policy` pasan a `"plataforma"`.

## 4. Rendimiento (LAS CUATRO)

- **Índices:** `from_db()` lee tres tablas chicas completas (17 capabilities), sin `WHERE` ni `ORDER BY`
  sobre tablas que crezcan, y solo al recargar. La columna nueva no se filtra: no pide índice.
- **Caché:** la clave nueva agrega un `stat` (µs) por turno; la recarga (~1 ms, medido para `from_db` en
  LAS MANOS el 2026-09-12) solo cuando algo cambió. Invalidación declarada: el sello de facet_resolver.
- **Async:** todo por aiomysql; nada bloqueante en el camino del turno.
- **Tamaño del snapshot (consecuencia de la decisión 1):** medido el 2026-09-14 renderizando con los datos
  reales: hoy 11 entradas y 715 caracteres; con las 17 capabilities de la DB, 28 entradas y ~1.786
  caracteres (~180 → ~450 tokens, estimados como caracteres/4, no con un tokenizador). El plan lo vuelve
  a medir con el código real y con `axioma_usage.tokens_in` de un turno de sonda antes y después.
- **Carga:** latencia en proceso de `validation_context()` con caché fría y caliente, antes y después. No se
  hace carga sobre `/api/chat` (llama al modelo); se declara así. Crear/reanudar pipelines: la carga
  existente no cambia de forma (un literal distinto).

## 5. Tests

- **jax-platform, migración:** puros — `_CAPABILITY_MODE` cubre exactamente lo sembrado y solo
  `file_write` es `mutating`. Con DB — forma de la columna (`NOT NULL`, sin default); valores sembrados
  (17 filas); `INSERT` sin `mode` falla (CONTROL del "sin default"); una base vieja (columna NULL, valores
  NULL) queda rellenada y en `NOT NULL` tras `run_migrations()`; una fila no sembrada con `mode` NULL hace
  fallar la migración con su nombre.
- **jax, catálogo:** el constructor por dict da `mutating` sin modo declarado y respeta el declarado; un
  modo inválido lanza; `capabilities()` sale ordenado. Con DB (job `jacobs-gobernanza-db`, que clona
  jax-platform y corre sus migraciones): `from_db()` trae `file_write=mutating` y `generate=read_only`.
- **jax, validador:** el tripwire del TOML vacío se pone rojo, como estaba previsto, y se reemplaza por
  `test_load_validation_context_usa_el_catalogo_que_recibe` + su CONTROL. La rama `in_catalog` verifica
  el modo (VALID con el modo real, FACT_MISMATCH con el otro). El test existente de la rama con catálogo
  sintético declara `mode` explícito.
- **jax, grounding:** sección `catalog_capabilities` con punteros ordenados y modo del catálogo; los
  punteros de `ops` no se mueven al agregar capabilities al catálogo; acreditar una entrada del catálogo
  da `ACCREDITED` y `validate()` da `VALID`; el modo equivocado da `FACT_NOT_IN_SNAPSHOT`; un catálogo que
  explota da `GroundingBuildError`; `render()` muestra las dos secciones.
- **jax, Jacobs:** `"Fernando"` se rechaza y `"plataforma"` se acepta, al crear, planificar, reanudar y
  aprobar.
- **jax-platform, contexto y Mesa:** el contexto trae el catálogo de la DB; tocar el sello fuerza la
  recarga; N turnos, una recarga; si la DB falla, error visible (frío y caliente) y `SnapshotError` en el
  chat. **Con DB, de punta a punta por `run_shadow_validation`:** un claim sobre `generate` citando su
  línea de `catalog_capabilities` queda `VALID`/`OBSERVADO`; con el modo equivocado,
  `FACT_NOT_IN_SNAPSHOT`; con el catálogo cambiado entre snapshot y validación, `FACT_MISMATCH`.
  `create`/`resume` mandan `"plataforma"` aunque el cliente mande otra cosa.
- **Tripwire (con DB):** la intersección entre `ops` del TOML y `capability` de la DB es vacía, con su
  CONTROL (un `[ops.generate]` agregado da la intersección y `SOURCE_CONFLICT`).
- Pisos exactos de CI subidos con el número medido; cada test nuevo visto en rojo; cada tripwire, en rojo
  por mutación.

## 6. Despliegue

**Orden de merge (tres PRs, por el acoplamiento de los CI).** El CI de jax clona `jax-platform` master
para crear el esquema (`jacobs-gobernanza-db`), y el CI de jax-platform clona `jax` master.

1. **jax-platform PR-A — solo la migración de `capability.mode`.** Compatible con `jax` master: el
   `from_db()` viejo no lee la columna. Se mergea primero. No se despliega sola.
2. **jax PR-B** — catálogo con `mode`, validador, snapshot, Jacobs. Su CI ya ve la columna.
3. **jax-platform PR-C** — `governance_context` async, `invoked_by`, tests de la Mesa y tripwires. Su CI
   clona el `jax` nuevo.

Entre el merge de PR-B y el de PR-C, el CI de `jax-platform` master (y el de cualquier rama que no tenga
PR-C, la etapa 2 incluida) falla en los tests de gobernanza: `master` todavía llama a
`load_validation_context` con dos argumentos. Por eso PR-B y PR-C se mergean **uno detrás del otro**, y
se avisa al ejecutor de la etapa 2.

**Despliegue:** `git pull` de los dos checkouts de producción; reinicio de **`jax-platform` primero** (sus
migraciones crean y siembran la columna al arrancar; se verifica con `SHOW COLUMNS` antes de seguir) y
**`jax-las-manos` después** (su `from_db()` nuevo lee `mode`). **Entre los dos reinicios (segundos) crear
o reanudar un pipeline puede fallar** (uno manda `plataforma` y el otro todavía espera `Fernando`); se
avisa antes. El frontend se publica con el procedimiento de `jax/CONTEXT.md` §7 (CORREGIDO 2026-09-14).

**Verificación en vivo:** columna y valores en producción (17 filas, solo `file_write` `mutating`);
el resolver con el contexto de producción; un claim de sonda (`origin='probe'`) sobre una capability de
la DB citando su línea de `catalog_capabilities`, que queda `VALID`/`OBSERVADO`, y el mismo con el modo
equivocado, que queda `FACT_NOT_IN_SNAPSHOT` en la Mesa (el `FACT_MISMATCH` del resolver se verifica en
proceso, sin acreditación); un turno real de la Mesa cuyo snapshot incluye `catalog_capabilities`; crear
y reanudar un pipeline con el rol nuevo; journal limpio.

## 7. Fuera de alcance

- ~~Capabilities de la DB en el snapshot del prompt (sub-proyecto aparte, ver §2).~~ → **Dentro de
  alcance desde v2** (§2, decisión 1).
- ~~Verificar el "modo" (lectura/escritura) de las capabilities de la DB: el catálogo no lo tiene.~~ →
  **Dentro de alcance desde v2** (§2, decisiones 2 y 3).
- `ops` del TOML como fuente: queda como está; su migración a la DB es otra tanda.
- Una línea base nueva de SP4: se hace al retomar SP4, con su propio pre-registro (§2, decisión 4).
- Que `jacobs/store.py::get_motor_governance()` exponga `mode`: hoy nada lo consume allí.
