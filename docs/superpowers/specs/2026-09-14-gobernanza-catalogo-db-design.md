# Gobernanza con el catálogo de la DB, e `invoked_by` como rol — diseño (tanda A)

**Fecha:** 2026-09-14 · **Aprobado por:** Fernando (diseño en chat, 2026-09-14 ~14:30) ·
**Repos:** `jax` (validador, Jacobs) y `jax-platform` (contexto de gobernanza, API de pipelines).

## 1. Por qué

Dos defectos de `DEUDA.md` ("Bloquea trabajo"):

1. **El resolver de `CAPABILITY_AVAILABLE` consulta un catálogo vacío.** El Bloque 3 movió las
   capabilities a la DB (`MotorCatalog.from_db()`), pero `policy/governance/validator.py::load_validation_context()`
   sigue armando `MotorCatalog(config)` desde `las_manos/config.toml`, cuyo `[capabilities.*]` está
   vacío. En producción lo usa `jax-platform/backend/governance_context.py` (validación en sombra de la
   Mesa): un claim `INFERIDO` sobre una capability que existe solo en la DB (`generate`, `file_write`...)
   sale `FACT_MISMATCH` — un falso negativo. El tripwire
   `test_real_toml_catalog_is_empty_since_block3_so_catalog_branch_is_dead_in_production` fija ese estado.
2. **`invoked_by` es un campo de autorización con un nombre de persona como rol.** `jacobs/routes.py:100`
   acepta solo `{"Fernando", "jax_local", "ada"}` y `policy.validate_resume` exige `"Fernando"`;
   jax-platform lo manda fijo (`PipelineModal.jsx`, `api/pipelines.py:161`).

## 2. Decisiones (Fernando, 2026-09-14)

- **Solo el resolver.** El snapshot que se inyecta en el prompt (SP3) NO cambia: sigue listando
  `ctx.ops`. Sumar las capabilities de la DB al snapshot sería un sub-proyecto propio (modo de cada una,
  prompt más grande, línea base de SP4 no comparable).
- **`invoked_by` pasa a ser el rol `plataforma`**: "pedido de jax-platform en nombre de un usuario
  autenticado". La identidad sigue viajando en `user_id`/`tenant_id`.

## 3. Diseño

### 3.1 jax — validador puro (sin I/O)

- `load_validation_context(repo_root, config_paths_allowlist, catalog)` **recibe** el `MotorCatalog`;
  deja de construirlo del TOML. El validador sigue sin tocar la DB.
- El resolver no cambia: con el catálogo real, la rama `in_catalog` deja de estar muerta y da `VALID`
  ("mode no verificable ahí, aceptado sin contradicción"), como ya estaba escrito.
- **Medido 2026-09-14:** 0 nombres en común entre `ops` del TOML (11) y `capability` de la DB (18), así
  que activar el catálogo no produce `SOURCE_CONFLICT`.

### 3.2 jax-platform — `governance_context.validation_context()` async

- Pasa a `async def validation_context()`. Llamadores: `api/chat.py` (`_build_snapshot_or_raise` /
  `_build_grounding`, llamado en `chat()` línea ~1115 → `await`) y `shadow_validation.py:315` (`await`).
- **Clave de caché:** los mtimes de los 3 archivos de config (como hoy) **+** el mtime del sello de
  `facet_resolver` (`_seal_mtime()`), que ya estampan las migraciones y el admin de motores y
  capabilities. Si la clave cambió → recarga; si no → el mismo objeto.
- **Recarga:** `catalog = await MotorCatalog.from_db()` y `load_validation_context(..., catalog)`, bajo un
  `asyncio.Lock` (N turnos a la vez no disparan N recargas) — el mismo patrón que LAS MANOS
  (`motor_registry/routes.py::_ensure_catalog_fresh`).
- **Falla visible, nunca catálogo vacío:** si la DB no responde, la excepción sube. En el chat,
  `_build_grounding` ya la convierte en `SnapshotError` → `grounding_snapshot_sha256='ERROR'`; la validación
  en sombra no escribe veredictos (hoy igual). Servir un catálogo vacío repetiría el falso negativo en
  silencio (P10).
- `validation_context.cache_clear` se conserva para los tests.

### 3.3 `invoked_by` = `plataforma`

- **Jacobs (jax):** `routes.py` y `policy.VALID_INVOKERS` aceptan `{"plataforma", "jax_local", "ada"}`;
  `validate_resume` exige `"plataforma"`.
- **jax-platform:** `api/pipelines.py` pone `invoked_by = "plataforma"` en el backend al crear (pisa lo
  que mande el cliente, igual que ya pisa `user_id`/`tenant_id`) y al reanudar. `PipelineModal.jsx` deja de
  mandarlo.
- Las filas viejas de `jacobs_pipelines` con `"Fernando"` quedan como están: son historia.
- Los tests que fabrican `Pipeline(invoked_by="Fernando")` sin pasar por la validación no cambian; los que
  pasan por `routes`/`policy` pasan a `"plataforma"`.

## 4. Rendimiento (LAS CUATRO)

- **Índices:** `from_db()` ya existe y lee tres tablas chicas; solo corre al recargar.
- **Caché:** la clave nueva agrega un `stat` (µs) por turno; la recarga (~1 ms, medido para `from_db` en
  LAS MANOS el 2026-09-12) solo cuando algo cambió. Invalidación declarada: el sello de facet_resolver.
- **Async:** todo por aiomysql; nada bloqueante en el camino del turno.
- **Carga:** latencia en proceso de `validation_context()` con caché fría y caliente, antes y después. No se
  hace carga sobre `/api/chat` (llama al modelo); se declara así. Crear/reanudar pipelines: la carga
  existente no cambia de forma (un literal distinto).

## 5. Tests

- **jax:** el tripwire se pone rojo, como estaba previsto, y se reemplaza por
  `test_load_validation_context_usa_el_catalogo_que_recibe` (una capability que solo existe en el catálogo
  pasado → `VALID`). `_real_ctx` pasa un catálogo explícito. Jacobs: `"Fernando"` se rechaza y
  `"plataforma"` se acepta, al crear y al reanudar.
- **jax-platform:** la validación en sombra da `VALID` para una capability que solo está en la DB; tocar el
  sello fuerza la recarga; si la DB falla, error visible y no catálogo vacío; `create`/`resume` mandan
  `"plataforma"` aunque el cliente mande otra cosa. **Tripwire nuevo (con DB):** la intersección entre
  `ops` del TOML y `capability` de la DB es vacía — si alguien agrega un nombre repetido, se entera antes
  de que aparezca `SOURCE_CONFLICT` en producción.
- Pisos exactos de CI subidos con el número medido; cada test nuevo visto en rojo.

## 6. Despliegue

Los dos repos juntos: merge de los dos PRs, `jax-platform` primero y `jax-las-manos` después. **Entre los
dos reinicios (segundos) crear o reanudar un pipeline puede fallar** (uno manda `plataforma` y el otro
todavía espera `Fernando`, o al revés); se avisa antes. Verificación en vivo: un claim de la Mesa sobre una
capability de la DB sale `VALID` (turno de sonda con `origin='probe'`); crear y reanudar un pipeline con el
rol nuevo; journal limpio.

## 7. Fuera de alcance

- Capabilities de la DB en el snapshot del prompt (sub-proyecto aparte, ver §2).
- Verificar el "modo" (lectura/escritura) de las capabilities de la DB: el catálogo no lo tiene.
- `ops` del TOML como fuente: queda como está; su migración a la DB es otra tanda.
