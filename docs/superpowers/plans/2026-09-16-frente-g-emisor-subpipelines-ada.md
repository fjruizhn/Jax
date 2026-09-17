# Frente G · Emisor de sub-pipelines de Ada — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que Ada reparta un objetivo en sub-pipelines por sí sola dentro de techos holgados. Hay dos modos: (1) un plan de delegación validado estrictamente y (2) tools en marcha, este último solo si GLM pasa la medición. Todo lo que exceda un techo pide aprobación de Fernando y le avisa por Telegram, nunca en silencio. La cola respeta el límite global y ningún hijo queda huérfano gastando.

**Architecture:**
- **Decisión:** `jacobs/delegacion.py` es el único punto de decisión. Carga el árbol, proyecta los techos (`jacobs/delegacion_techos.py`, puro), emite los tokens del frente F y crea hijos `queued` por `routes.crear_hijo_de_ada`, o pide aprobación.
- **Cola:** `jacobs/cola.py` admite en orden FIFO bajo el mismo `_pipeline_create_lock`. Se frena con el kill switch, vence las esperas y reencola tras una aprobación.
- **Consumo:** el consumo del árbol vive en `jacobs_arbol_consumo`. Lo actualizan los dos escritores de uso en la misma transacción que `axioma_usage`.
- **Executor:** valida el step `delegate` con el validador estricto `plan_delegacion.v1` (`motor_registry/esquema_plan_delegacion.py`) y hace un reintento. Después delega, deja al padre en `waiting_children` (fuera del cupo) y lo reanuda con un step `integrate` cuando terminan todos los hijos.
- **Aprobaciones:** se exponen en Jacobs (`/jacobs/delegaciones/*`) y en el admin de jax-platform (`/admin/delegaciones`).
- **Etapa 2:** mide el tool-calling de GLM con 5 llamadas reales (gate) y migra `ada` a `MOTOR_FACETS`.
- **Etapa 3:** agrega `lanzar_subpipeline` y `leer_resultado_subpipeline` al bucle GAP2. `tool_authority` las resuelve llamando a la MISMA `delegacion.py`.

**Tech Stack:** Python 3.12 en CI de jax y 3.14 en los venvs de hall9000. FastAPI 0.139, Pydantic 2, aiomysql 0.3.2, MariaDB (12.3.3 en producción en `127.0.0.1:3308`, 11.8 en CI), pytest y unittest, k6 v2.2.0 (`/home/fruiz/bin/k6`). En jax-platform: React 19, vitest y react-i18next vía `i18n/index.jsx`.

**Spec:** `/home/fruiz/worktrees/jax-hallazgos-docs/docs/superpowers/specs/2026-09-16-emisor-subpipelines-ada-design.md` (aprobado por Fernando el 2026-09-16, con GO para la medición de GLM, el árbol real en vivo con techos bajados y los deploys).
- Depende del plan del frente F **con su ENMIENDA**: `/home/fruiz/worktrees/jax-hallazgos-docs/docs/superpowers/plans/2026-09-16-frente-f-contrato-subpipelines.md`.
- Usa interfaces del frente B (`/home/fruiz/worktrees/jax-platform-hallazgos-docs/docs/superpowers/plans/2026-09-16-frente-b-kill-switch.md`) y del frente E (`/home/fruiz/worktrees/jax-hallazgos-docs/docs/superpowers/plans/2026-09-16-frente-e-jax-limpieza-defectos-reglas.md`).
- Reglas comunes: `/home/fruiz/worktrees/jax-platform-hallazgos-docs/docs/superpowers/specs/2026-09-16-hallazgos-auditoria-design.md` §0 y "Decisiones de Fernando al GO".

**Ramas y worktrees de ejecución:**
- jax: `feat/emisor-subpipelines-ada` en `/home/fruiz/worktrees/jax-frente-g` (repo `/home/fruiz/jax`, remoto `fjruizhn/Jax`), base = `origin/master` **después** de mergear E, B y F (orden de merge de Fernando: E → B → F → G).
- jax-platform (gemela): `feat/emisor-subpipelines-ada` en `/home/fruiz/worktrees/jax-platform-frente-g` (repo `/home/fruiz/jax-platform`, remoto `fjruizhn/jax-platform`), base = `origin/master` después de A → C → B → D.

**Base medida al escribir:** jax `origin/master` = `772c1be`, que no es `bd95237`. `git diff --stat bd95237 origin/master -- jacobs las_manos loadtest` sale vacío: sólo cambió `.github/workflows/policy.yml` (pisos de la Fase 2 del Ejecutor). jax-platform `origin/master` = `26c9cd5`. Plan escrito contra esos árboles más las interfaces que declaran E, B y F.

---

## Discrepancias con el spec

Cada una se verificó contra el código de `origin/master`. El plan sigue el código real y lo dice acá.

1. **`_CAPABILITY_MAP` ya no existe** (prerrequisito §3.2 (a)). `jacobs/executor.py:546-555` (docstring de `_invoke_motor`) lo declara eliminado en el Bloque 3 (2026-08-21). Además, `analysis` y `review` hoy son **filas propias** de `capability`: `jax-platform/backend/db/migrations.py`, `_CAPABILITY_SEED`, entradas `research`/`analysis`/`review` con `allowed_callers=["jacobs"]` y sin `capability_motor`. El diagnóstico de la ronda 8 (`CONTEXT.md:558`) es anterior a esos dos cambios. **Plan:** la "traducción" se reemplaza por una auditoría de las capabilities reales de `ada` en producción (Task 20, SELECT de sólo lectura) y filas `capability_motor(<cap>, 'ada')` sembradas por jax-platform para cada una que pase los 8 checks de `MotorPolicy`. `assemble` queda exento del chequeo de plan (es mecánico: `executor.py:849`), porque si no, mover `ada` a `MOTOR_FACETS` rechazaría todo plan con ensamble (`plan.py:369-398`).
2. **`plan_delegacion.v1` nunca estuvo en `_KNOWN_UNIMPLEMENTED_SCHEMAS`** (`output_validator.py:54-57`). Hoy un schema así cae en "no reconocido", que falla cerrado. **Plan:** entra a `SCHEMAS` (para que el test de drift `las_manos/_output_validator_db_drift_test.py:35` lo cubra) y además con un validador estricto registrado en `_VALIDADORES_ESTRICTOS`.
3. **`output_validator` sólo corre en el worker del Motor Registry** (`worker.py:817`). En la etapa 1 Ada va por HTTP directo (`models.py:22`, `executor.py:953-965`), y ahí nadie valida la salida. **Plan:** el executor llama al MISMO validador para el step `delegate` y hace el reintento con el error explícito. Desde la etapa 2 (ada por el motor), el reintento ya lo hace el worker (`worker.py:820-840`) y Jacobs sólo re-valida, sin reintentar dos veces.
4. **El techo de profundidad no es aprobable.** F rechaza el consumo con `t.depth_hijo <= %s` (plan F, `SQL_CONSUMIR_TOKEN`), así que una aprobación de Fernando no puede crear ese hijo. **Plan:** exceder la profundidad deja evento `DELEGACION_EXCEDE_TECHO` con `aprobable=false`, avisa por Telegram y falla el step, sin `awaiting_approval`. Los otros cuatro techos sí piden aprobación.
5. **No existe el estado `cancelled` en `PipelineStatus`** (`models.py:26-34`). La cancelación por API usa `aborted` (`routes.py:242-243`). **Plan:** los hijos cortados pasan a `aborted` con el evento `SUBPIPELINE_CANCELADO_POR_PADRE {padre, motivo}`, así no aparece un tercer estado terminal con el mismo significado.
6. **La cola se traba con `MAX_PARALLEL_PIPELINES=3`** (`policy.py:17`) si los padres que esperan ocupan cupo. `store.pipeline_count_active` cuenta `pending` y `running` (`store.py:357-367`). Con raíz, hijo A y hijo B delegando a la vez, los tres cupos quedan ocupados por padres que esperan, y los nietos se quedan en `queued` hasta vencer. El spec no lo trata. **Plan:** estado nuevo `waiting_children`, que no cuenta como activo. El padre sale de `running` recién después de consumir todos los tokens (la ENMIENDA de F exige padre `running` al consumir).
7. **Los hijos entran siempre por la cola**, no sólo cuando se supera el límite. La admisión es un único camino, FIFO, bajo `_pipeline_create_lock`, así que el límite global no tiene dos implementaciones. `validate_create` deja de aplicar el límite de paralelos a `ada`, porque lo aplica el despachador. `SUBPIPELINE_ENCOLADO` se escribe siempre y `SUBPIPELINE_ADMITIDO` al entrar.
8. **`depende_de` necesita un estado de espera.** Un hijo con dependencias queda `queued` con `queued_at NULL` y el despachador sólo mira `queued_at NOT NULL`. La espera máxima (`JAX_ADA_COLA_MAX_ESPERA_SEGUNDOS`) cuenta desde que se cumplen las dependencias, no desde la creación: si no, un hijo que espera a otro largo vencería por culpa ajena.
9. **Datos que el spec no lista:**
   - la tabla `jacobs_delegacion_aprobaciones` (qué se pidió, quién aprobó y cuándo; el spec dice "auditado con usuario y hora", y los eventos no se pueden listar por tipo sin índice);
   - en `jacobs_arbol_consumo`, las columnas `consumo_incompleto`, `techo_pipelines_aprobado` y `techo_hijos_por_step_aprobado` (sin estas dos, aprobar un exceso de pipelines o de hijos no tendría efecto);
   - los índices `idx_jacobs_pipelines_padre`, `idx_jacobs_pipelines_raiz (root_pipeline_id, status)` e `idx_jacobs_pipelines_cola (status, queued_at)`.
10. **"El flujo de aprobación de Jacobs expuesto en el admin" no existe.**
    - `approve-step` sólo sirve para Hyde o para modo supervised (`routes.py:304-346`).
    - jax-platform no lo expone: `api/pipelines.py` sólo reenvía `resume` y `cancel`.

    **Plan:** rutas nuevas en Jacobs (`/jacobs/delegaciones/pendientes|{id}/aprobar|{id}/rechazar`), un proxy de superadmin en jax-platform (`/api/admin/delegaciones`) y la pantalla `AdminDelegaciones`. Aprobar va con `Dialogo`; rechazar, con `ConfirmacionSuma`, porque aborta. El rechazo explícito se agrega porque el spec no define otra salida de `awaiting_approval`.
11. **"En la misma transacción que registra el uso" no existe todavía:**
    - los dos escritores abren una conexión `autocommit=True` sin transacción (`jacobs/usage_writer.py:149-163`, `motor_registry/usage_writer.py:155-170`);
    - `motor_registry` no conoce el pipeline (`motor_registry/usage_writer.py:218-224`).

    **Plan:**
    - el escritor hace `conn.begin()` → INSERT en `axioma_usage` → suma en `jacobs_arbol_consumo` → `commit`;
    - `root_pipeline_id` viaja en `MotorDispatchRequest`;
    - si la base cae y la fila va al respaldo (`cola_uso`), el árbol queda subcontado. En ese caso se marca `consumo_incompleto=1` y la próxima delegación de ese árbol exige aprobación (fail-closed).
12. **La cadencia del despachador no puede ser la del reaper.** El reaper duerme 300 s (`reaper.py:69`), y admitir un hijo 5 minutos tarde no sirve. **Plan:** `cola.bucle_despachador()` arranca en `_jacobs_init` junto al reaper, con su propia variable `JAX_ADA_COLA_CADENCIA_SEGUNDOS`. También es nueva `JAX_ADA_RESUMEN_HIJO_CARACTERES`, el largo del resumen por hijo que recibe la integración. **`JAX_ADA_MAX_DEPTH` NO se crea:** el spec pide una sola fuente (`JAX_MAX_SUBPIPELINE_DEPTH` de F), y si alguien la pone en `/etc/jax/.env` el arranque falla.
13. **Cancelar o abortar hoy no detiene un run en curso.**
    - `run_pipeline` escribe `running` después de cada ola (`executor.py:1172-1174`) y `completed` al final (`executor.py:1206-1208`), pisando un `aborted` de `cancel_pipeline`.
    - Sin arreglar eso, "padre abortado → sin huérfanos" no se puede cumplir.

    **Plan:** transiciones condicionales (`store.pipeline_transicion`) y una lectura del estado antes de cada ola. El step que ya está en vuelo termina; B corta en vuelo sólo con el kill switch.
14. **Forma de los hijos:**
    - `mode=autonomous` y un solo step (`steps=[StepSpec]` pasa por `_from_spec`, sin LLM);
    - `user_id`/`tenant_id` y `owner_ack_at` heredados del padre.

    Sin `owner_ack_at`, el reaper cosecha un hijo `interrupted` a los 600 s (`reaper.py:143-151`) aunque Fernando pudiera aprobarlo.
15. **Orden de la etapa 2.** La medición de GLM va primero. La migración de Ada sólo se hace si pasa: el spec §2 la lista como requisito del modo 2, no del modo 1, y migrar sin modo 2 cambiaría el camino de todas las capabilities de Ada sin necesidad.

    **Aprobación en el modo 2:** F exige el padre `running` para consumir un token. Por eso un exceso detectado por la tool se aprueba igual que en el modo 1: al terminar el step, el padre pasa a `awaiting_approval` con los sub-objetivos pendientes guardados, y aprobar lo vuelve a `running` para lanzar.
16. **Las tools se ofrecen según los datos.** Poner `motor.has_tool_access=TRUE` en ada (spec §3.2 (c)) le daría también `read_file`/`write_file`: el worker entrega `TOOLS_CATALOG` entero (`worker.py:556`) y `tool_authority` no mira `allowed_motors` (`tool_authority.py:180-204`). **Plan:** el worker ofrece cada tool sólo si el motor está en `allowed_motors` de la capability que la tool mapea. Para jax_local y kimi no cambia nada, porque ya están en `file_read`/`file_write`.
17. **`tokens_sin_precio`:** el spec crea la columna pero no dice qué hace. **Recomendación implementada:** se informa en `DELEGACION_PROPUESTA` y en la aprobación, pero no bloquea, porque el techo de tokens cubre esos tokens. Se le pregunta a Fernando en la revisión (Task 14, Step 1).
18. **Los prompts de formato y de integración son texto para el LLM, no UI.** `instrucciones_de_formato()` e `armar_prompt_integracion()` viven en código, igual que `_EVIDENCE_RULE` (`executor.py:145-152`) y los prompts del planner (`plan.py:160-195`). No van a i18n.
19. **El deploy de jax-platform va primero.** Las filas `delegate`/`integrate` las siembra `run_migrations()` al arrancar el backend (`backend/main.py:91`). El job `jacobs-gobernanza-db` de jax clona jax-platform `master` (`policy.yml:962-978`), así que los tests de DB de G sólo pasan en CI después de mergear la rama gemela.

## Solapamientos

| Archivo | Frente G (este plan) | Otro frente | Qué hacer |
|---|---|---|---|
| `jacobs/routes.py` | `crear_hijo_de_ada`, `_crear_pipeline`, rutas `/delegaciones/*`, cascada en `cancel_pipeline` | **F** (`create_pipeline` consume el token), **E-13** (`plan_only`), **B** (`check_kill_switch`) | G parte de F ya mergeado: extrae el bloque de F a `_crear_pipeline` sin cambiar su orden. Correr la suite de F (`tests/test_subpipeline_contrato_rutas.py`) después de cada cambio |
| `jacobs/policy.py` | `validate_create` no aplica el límite de paralelos a `ada` | **F** (firma), **E-13**, **E-20**, **B** | Una línea dentro de `validate_create` |
| `jacobs/models.py` | 3 estados nuevos, `root_pipeline_id`/`queued_at`, constantes de capability | **F** (`parent_pipeline_id`/`depth`), **E-02/E-03/E-13** | Merge textual |
| `jacobs/store.py` | tablas, columnas, índices y funciones de delegación | **F** (tokens, `pipeline_create`), **E-03** (`get_motor_governance` devuelve `facets`) | G usa `governance["facets"]` de E; `pipeline_create` recibe dos columnas más sobre la versión de F |
| `jacobs/executor.py` | `_run_one_step` (delegate), `run_pipeline` (transiciones, espera, hook), uso con raíz, `_invoke_motor` con raíz | **B** (`correr_con_interruptor` en `_run_one_step`), **E-24** (`obtener_cliente_http`), **E-16**, **E-12/E-22** | G envuelve **dentro** de `correr_con_interruptor`. Los tests de G no parchean `httpx.AsyncClient`: parchean `_dispatch_step` o `_TRANSPORT_DISPATCH` |
| `jacobs/plan.py` | `_check_delegacion` en `build()` | **E-03/E-17** (`_check_facets` en `build()`) | Se agrega debajo de `_check_facets` |
| `jacobs/reaper.py` | hook de terminación después de `REAPED` | **E-24** (Telegram con cliente compartido) | Merge textual |
| `las_manos/server.py` | `config_delegacion()` y arranque de `bucle_despachador` en `_jacobs_init` | **F** (`config_subpipelines()`), **B** (freno), **E-21/E-24** | Mismo bloque que F; el orden queda F, G, `init_tables()` |
| `las_manos/motor_registry/{models,routes,worker,usage_writer,tool_authority,output_validator,tools_catalog}.py` | raíz y pipeline en el dispatch, uso transaccional, tools de sub-pipelines, esquema estricto | **B** (freno del worker), **E-24** (cliente), **E-08** | Merge textual. En `worker.py` el reparto de tools por `allowed_motors` es un bloque propio |
| `.github/workflows/policy.yml` (jax) | archivos nuevos en `tests-puros` y en `jacobs-gobernanza-db`, `output-validator-regression` | **E, B, F** y la Fase 2 del Ejecutor (piso 712 en `772c1be`) | **Nunca sumar a mano:** rebasear y medir en el runner |
| `/etc/jax/.env` | 8 variables `JAX_ADA_*` | **B** (`JAX_KILL_SWITCH_PATH`), **E** (`JAX_OLLAMA_URL`…), **F** (`JAX_SUBPIPELINE_TOKEN_TTL_SECONDS`, `JAX_MAX_SUBPIPELINE_DEPTH`) | Backup con timestamp; `grep -c` para que ninguna quede duplicada |
| jax-platform `backend/db/migrations.py` | semillas `delegate`/`integrate` (etapa 1); `capability_motor` de ada (etapa 2) | **A**, **C** (migraciones propias) | Entradas nuevas en las listas; el test de conteo `test_capability_motor_seed_count_is_26` cambia en la etapa 2 |
| jax-platform `frontend/src/i18n/{es,en}.js`, `pages/Admin.jsx`, `components/admin/AdminSidebar.jsx` | claves de delegaciones, estados nuevos, ruta y menú | **A** (A-10/A-11 borran claves), **B** (claves del freno), **C** (ajustes de admin) | Merge textual; paridad es/en verificada por su test |
| jax-platform `frontend/src/components/BottomBar/pipelineChain.js` | `GOVERNED_FACETS` suma `ada` (etapa 2) | **D** (adjuntos) | Merge textual |
| `jax/DEUDA.md`, `jax/CONTEXT.md` | entradas del frente G | **Todos** | PR de docs al final, rebaseado |

**Orden de merge:** jax E → B → F → G y jax-platform A → C → B → D → G, según la decisión de Fernando al GO. **Dentro de G, la gemela de jax-platform (Task 1) se mergea y despliega ANTES que el PR de jax**, por la Discrepancia 19.

## Global Constraints

Copiadas de §0 del spec común (aplican a todas las tareas):

- **TDD:** test rojo contra el código viejo antes del arreglo, porque un control que no falla no valida. El rojo se pega en el ledger con el comando y su salida.
- **i18n:** ningún texto visible literal; es/en en paridad.
- **Tema:** dark/light con tokens.
- **Diálogos:** nada de `confirm/alert/prompt`, ni desnudos ni con `window.`. Se usan `components/Dialogo.jsx` y `ConfirmacionSuma` para lo destructivo.
- **Sin hardcoding:** la config va en `/etc/jax/.env` o en la DB.
- **Fail-closed.** Todo caché declara su invalidación en el mismo commit. Este frente no agrega cachés: la config son `os.environ.get` por llamada, igual que F.
- **Las cuatro del rendimiento:**
  - índice verificado con EXPLAIN sobre la consulta real;
  - nada bloqueante en `async def` (`_load_ref` lee disco: va en `asyncio.to_thread`);
  - **prueba de carga con número registrado** para todo endpoint nuevo o modificado en camino de usuario.
- **Tests con barrera de DB de producción:** todo archivo que escribe importa primero `jacobs/_arnes_ada.py` (barrera de F) o fija y verifica `JAX_DB_NAME=jax_memory_test`.
- **CI:**
  - todo test nuevo lo corre un job y va en **las dos listas** del job;
  - los pisos se miden en el runner;
  - se verifica rompiéndolo, con rojo sobre el sha real leído por API.
- **Mirror-sync:** si se toca un símbolo espejado (`jax/scripts/check_mirror_sync.py`), el cambio va en los dos repos en el mismo paso. Este frente no toca ninguna familia; la Task 12 lo verifica.
- **Deploy:**
  - backend jax: `sudo systemctl restart jax-las-manos.service` con 0 pipelines en vuelo;
  - jax-platform: `sudo systemctl restart jax-platform.service` con 0 pipelines en vuelo;
  - frontend: build → rsync a `/tmp/axioma-deploy/` en la VM dev → `sudo rsync -a --delete --chown=www:www --exclude .user.ini` a `/www/wwwroot/axioma-ia.io/`, con backup previo.
- **Biblioteca:** registro en `jax/DEUDA.md` y `jax/CONTEXT.md` antes de cerrar.
- **Marca P10:** todo `except` amplio que siga de largo lleva `# fail-soft: <razón>` o `# fail-closed: <razón>` en la línea del `except`.
- **Nunca cambiar de rama en `/home/fruiz/jax` ni en `/home/fruiz/jax-platform`:** son los checkouts de producción.

Propias de este frente (valores exactos del spec §4, más las Discrepancias 9 y 12):

- Variables, validadas al arrancar LAS MANOS (`las_manos/server.py::_jacobs_init`). Si la variable falta, rige el valor por defecto. Si está presente pero vacía, no es un número o cae fuera de rango, lanza `ValueError` y el arranque cae:

  | Variable | Defecto | Rango |
  |---|---|---|
  | `JAX_ADA_MAX_HIJOS_POR_STEP` | 10 | [1, 50] |
  | `JAX_ADA_MAX_PIPELINES_ARBOL` | 40 | [1, 500] |
  | `JAX_ADA_MAX_TOKENS_ARBOL` | 2000000 | [1000, 100000000] |
  | `JAX_ADA_MAX_USD_ARBOL` | 10 | [0.01, 1000] |
  | `JAX_ADA_RESERVA_TOKENS_HIJO` | 50000 | [0, 10000000] |
  | `JAX_ADA_COLA_MAX_ESPERA_SEGUNDOS` | 3600 | [60, 86400] |
  | `JAX_ADA_COLA_CADENCIA_SEGUNDOS` | 10 | [1, 300] |
  | `JAX_ADA_RESUMEN_HIJO_CARACTERES` | 2000 | [200, 20000] |

  La profundidad se lee sólo de `JAX_MAX_SUBPIPELINE_DEPTH` (F: defecto 3, rango [1, 5]). `JAX_ADA_MAX_DEPTH` presente tumba el arranque.
- Estados nuevos de `jacobs_pipelines.status`: `queued`, `awaiting_approval` y `waiting_children`. Los terminales son `completed`, `aborted`, `expired` y `failed`.
- Eventos en `jacobs_events`:
  - del spec: `DELEGACION_PROPUESTA`, `DELEGACION_APROBADA` (`por: auto|fernando`), `DELEGACION_EXCEDE_TECHO`, `DELEGACION_TECHO_CRUZADO`, `SUBPIPELINE_ENCOLADO`, `SUBPIPELINE_ADMITIDO` y `SUBPIPELINE_EXPIRADO_EN_COLA`;
  - declarados por este plan: `DELEGACION_PLAN_INVALIDO`, `DELEGACION_SIN_MEDICION`, `DELEGACION_FALLIDA`, `DELEGACION_RECHAZADA`, `DELEGACION_NOTIFICACION_FALLIDA`, `DELEGACION_HIJOS_TERMINADOS`, `PIPELINE_ESPERA_HIJOS`, `PIPELINE_DETENIDO`, `SUBPIPELINE_CANCELADO_POR_PADRE` y `SUBPIPELINE_DEPENDENCIA_FALLIDA`.
- Capabilities nuevas, filas de jax-platform:
  - `delegate`: `risk_level='medium'`, `sandbox_only=1`, `requires_human_gate=0`, `max_execution_minutes=15`, `max_recursion_depth=0`, `output_schema='plan_delegacion.v1'`, `allowed_callers=["jacobs"]`, `mode='read_only'`;
  - `integrate`: `risk_level='low'`, lo demás igual, con `output_schema` NULL.
- Códigos HTTP de las rutas de aprobación en Jacobs:
  - 200 si sale bien;
  - 403 si `invoked_by` no es `plataforma`;
  - 404 si la aprobación no existe;
  - 409 si ya está resuelta;
  - 422 con `{"code": "techo_insuficiente", "faltan": [...]}`;
  - 423 con el kill switch puesto (sólo `aprobar`).
- Variables de shell que usan los comandos (declararlas en cada terminal):
  ```bash
  WT=/home/fruiz/worktrees/jax-frente-g
  WTP=/home/fruiz/worktrees/jax-platform-frente-g
  PY=/home/fruiz/jax/.venv/bin/python
  PYP=/home/fruiz/jax-platform/backend/.venv/bin/python
  L=$WT/.superpowers/sdd/2026-09-16-emisor-subpipelines-ada/progress.md
  DBTEST='set -a; . /etc/jax/.env; set +a; export JAX_DB_NAME=jax_memory_test'
  PUROS='env -u JAX_DB_HOST -u JAX_DB_PORT -u JAX_DB_USER -u JAX_DB_PASSWORD JAX_DB_NAME=jax_memory_test JAX_KILL_SWITCH_PATH=/tmp/jax-frente-g-sin-freno/PAUSE'
  ```
  - `DBTEST` carga host, puerto y credenciales de la MariaDB real y pisa `JAX_DB_NAME` con la base de prueba.
  - `PUROS` corre sin base y con un freno en una ruta que no existe; si el conftest de B ya fija `JAX_KILL_SWITCH_PATH`, esa asignación queda.
- **Ledger de la ejecución:** `$L`. Cada rojo visto se pega ahí con el comando y su salida.

---

## File Structure

| Repo | Archivo | Acción | Responsabilidad |
|---|---|---|---|
| jax | `jacobs/config_delegacion.py` | Crear | Techos y cadencias por entorno con rango; prohíbe `JAX_ADA_MAX_DEPTH` |
| jax | `las_manos/motor_registry/esquema_plan_delegacion.py` | Crear | Validador estricto puro de `plan_delegacion.v1`, orden topológico, chequeo contra catálogo, instrucciones de formato |
| jax | `las_manos/motor_registry/output_validator.py` | Modificar | `plan_delegacion.v1` en `SCHEMAS` + `_VALIDADORES_ESTRICTOS` |
| jax | `jacobs/delegacion_techos.py` | Crear | Puro: `Consumo`, `Proyeccion`, `TechoExcedido`, `proyectar`, `evaluar_techos`, `techos_cruzados`, `ResumenHijo`, `armar_prompt_integracion` |
| jax | `jacobs/models.py` | Modificar | Estados `queued`/`awaiting_approval`/`waiting_children`; `root_pipeline_id`, `queued_at`; `CAPABILITY_DELEGAR`, `CAPABILITY_INTEGRAR`, `ESTADOS_TERMINALES` |
| jax | `jacobs/store.py` | Modificar | Columnas, índices, `jacobs_arbol_consumo`, `jacobs_delegacion_aprobaciones`, transiciones condicionales y consultas de árbol, cola y aprobaciones |
| jax | `jacobs/usage_writer.py`, `las_manos/motor_registry/usage_writer.py` | Modificar | Uso + consumo del árbol en una transacción; devuelven `bool` |
| jax | `jacobs/delegacion.py` | Crear | Único punto de decisión: `delegar`, esperas, reanudación, cascada, aprobar/rechazar, tool de sub-pipelines |
| jax | `jacobs/cola.py` | Crear | `despachar`, `vencer_esperas`, `bucle_despachador`, `lanzar_en_fondo` |
| jax | `jacobs/routes.py` | Modificar | `_crear_pipeline` (bloque de F), `crear_hijo_de_ada`, raíz en todo pipeline nuevo, cascada al cancelar, rutas de delegaciones |
| jax | `jacobs/policy.py` | Modificar | El límite de paralelos no aplica a `ada` |
| jax | `jacobs/executor.py` | Modificar | Step `delegate`, transiciones condicionales, espera de hijos, guard de `integrate`, hook de terminación, raíz en el uso |
| jax | `jacobs/plan.py` | Modificar | `_check_delegacion` (`delegate` sólo ada; `integrate` sólo lo agrega Jacobs) |
| jax | `jacobs/reaper.py` | Modificar | Hook de terminación después de `REAPED` |
| jax | `las_manos/server.py` | Modificar | `config_delegacion()` al arrancar; `bucle_despachador` en background |
| jax | `las_manos/motor_registry/models.py`, `routes.py`, `worker.py` | Modificar | `root_pipeline_id` (etapa 1); `pipeline_id`/`step_id` y tools por `allowed_motors` (etapa 3) |
| jax | `las_manos/motor_registry/tools_subpipelines.py` | Crear (etapa 2) | Declaraciones y validador puro de argumentos de las dos tools |
| jax | `las_manos/motor_registry/tool_authority.py`, `tools_catalog.py` | Modificar (etapa 3) | Mapeo y ejecución de las dos tools por `delegacion.py` |
| jax | `jacobs/models.py`, `jacobs/reaper.py`, `jacobs/_http_facet_admission_test.py` | Modificar (etapa 2) | `ada` pasa a `MOTOR_FACETS` |
| jax | `scripts/medir_tool_calling_glm.py` | Crear (etapa 2) | Medición de 5 llamadas reales a GLM con calificación determinista |
| jax | `jacobs/_arnes_delegacion.py` | Crear | Arnés de pruebas (barrera de F) |
| jax | tests (ver cada tarea) | Crear | Puros, de DB, E2E y de tools |
| jax | `loadtest/jacobs_delegacion_app.py`, `loadtest/jacobs_delegacion.js` | Crear | Carga aislada de delegar, despachar y listar aprobaciones |
| jax | `.github/workflows/policy.yml` | Modificar | Listas y pisos |
| jax-platform | `backend/db/migrations.py` | Modificar | Semillas `delegate`/`integrate`; `capability_motor` de ada (etapa 2) |
| jax-platform | `backend/api/admin/delegaciones.py`, `backend/main.py` | Crear/Modificar | Proxy de superadmin a las rutas de Jacobs |
| jax-platform | `frontend/src/pages/admin/AdminDelegaciones.jsx` (+test), `pages/Admin.jsx`, `components/admin/AdminSidebar.jsx`, `i18n/es.js`, `i18n/en.js` | Crear/Modificar | Pantalla de aprobaciones y claves |
| jax-platform | `frontend/src/components/BottomBar/pipelineChain.js` | Modificar (etapa 2) | `GOVERNED_FACETS` suma `ada` |
| jax-platform | `loadtest/admin_delegaciones.js` | Crear | Carga del proxy de admin |
| jax | `DEUDA.md`, `CONTEXT.md` | Modificar | Biblioteca |

---

# ETAPA 1 · Modo 1: plan de delegación de punta a punta

### Task 0: Preparación, verificación de la base y ledger

**Files:**
- Create: `$L` (ledger)

**Interfaces:**
- Consumes (del árbol mergeado, verificado acá, no supuesto):
  - F: `jacobs.subpipelines.emitir_token_subpipeline(parent_pipeline_id: str, parent_step: str) -> str`, `consumir_token_subpipeline(token, parent_pipeline_id, hijo_pipeline_id) -> TokenConsumido | ConsumoRechazado`, `config_subpipelines() -> ConfigSubpipelines(ttl_segundos, max_profundidad)`, `EmisionRechazada(.motivo)`, `jacobs_pipelines.parent_pipeline_id/depth`, `jacobs_subpipeline_tokens(token_hash, parent_pipeline_id, parent_step, depth_hijo, emitido_at, vence_at, usado_at, hijo_pipeline_id)`, `jacobs._arnes_ada` (barrera + `una_fila`, `ejecutar`), `models.INVOKER_ADA`, `PipelineCreateRequest.parent_pipeline_id`.
  - B: `interruptor.interruptor_activo()`, `jacobs.policy.check_kill_switch()`, `interruptor.correr_con_interruptor(corrutina)`, jax-platform `kill_switch.exigir_mesa_libre`, `api/errores.js::textoDeKillSwitch(t, err)`.
  - E: `store.get_motor_governance()["facets"]` (frozenset de `facet.key` activos), `plan._check_facets`, `cliente_http_compartido.obtener_cliente_http()`.
- Produces: el sha base anotado y el ledger.

- [ ] **Step 1: Worktrees y ramas**

```bash
git -C /home/fruiz/jax fetch origin
git -C /home/fruiz/jax log --oneline -5 origin/master
git -C /home/fruiz/jax worktree add /home/fruiz/worktrees/jax-frente-g -b feat/emisor-subpipelines-ada origin/master
git -C /home/fruiz/jax-platform fetch origin
git -C /home/fruiz/jax-platform log --oneline -5 origin/master
git -C /home/fruiz/jax-platform worktree add /home/fruiz/worktrees/jax-platform-frente-g -b feat/emisor-subpipelines-ada origin/master
git -C /home/fruiz/worktrees/jax-frente-g branch --show-current
git -C /home/fruiz/worktrees/jax-platform-frente-g branch --show-current
mkdir -p /home/fruiz/worktrees/jax-frente-g/.superpowers/sdd/2026-09-16-emisor-subpipelines-ada
```
Expected: las dos últimas líneas imprimen `feat/emisor-subpipelines-ada`. Anotar los dos sha base en `$L`.

- [ ] **Step 2: Verificar que E, B y F están mergeados (gate de arranque)**

```bash
cd $WT && pwd && git grep -n "async def emitir_token_subpipeline\|async def consumir_token_subpipeline\|MAX_PROFUNDIDAD_POR_DEFECTO = 3" -- jacobs/subpipelines.py
cd $WT && git grep -n "paso_status\|s.status = 'running'" -- jacobs/subpipelines.py jacobs/store.py || echo "sin condiciones de paso (ENMIENDA aplicada)"
cd $WT && git grep -n "def interruptor_activo\|async def correr_con_interruptor" -- jax/core/interruptor.py
cd $WT && git grep -n '"facets"' -- jacobs/store.py && git grep -n "def _check_facets" -- jacobs/plan.py
cd $WT && git grep -n "def obtener_cliente_http" -- jax/core/cliente_http_compartido.py
cd $WTP && pwd && git grep -n "async def exigir_mesa_libre" -- backend/kill_switch.py && git grep -n "export function textoDeKillSwitch" -- frontend/src/api/errores.js
```
Expected: cada `grep` encuentra su símbolo y la segunda línea imprime "sin condiciones de paso (ENMIENDA aplicada)". **Si alguno falta, parar:** G no se escribe contra interfaces supuestas (Principio I). Avisar a Fernando qué frente falta.

- [ ] **Step 3: Releer los archivos que G toca, ya con E/B/F encima**

Leer completos en `$WT`: `jacobs/routes.py`, `jacobs/policy.py`, `jacobs/models.py`, `jacobs/store.py`, `jacobs/executor.py` (al menos `_dispatch_step`, `_run_one_step`, `run_pipeline`, `_fail_step`), `jacobs/plan.py::build`, `jacobs/reaper.py`, `jacobs/usage_writer.py`, `las_manos/server.py::_jacobs_init`, `las_manos/motor_registry/{models,routes,worker,usage_writer,output_validator}.py`. Anotar en `$L` las líneas reales donde se aplicarán los cambios de las Tasks 2-10. Los números de línea de este plan son de `772c1be`; **manda el árbol**.

- [ ] **Step 4: Venv y base de prueba**

```bash
$PY -c "import pytest, aiomysql, fastapi, pydantic, httpx; print('deps ok', fastapi.__version__)"
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY - <<'PY'
import asyncio
from jacobs import _arnes_ada as ada
async def main():
    for t in ('capability', 'facet', 'axioma_usage', 'model', 'jacobs_subpipeline_tokens'):
        f = await ada.una_fila('SELECT COUNT(*) AS n FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s', (t,))
        print(t, f['n'])
    f = await ada.una_fila(\"SELECT GROUP_CONCAT(\`key\`) AS k FROM facet WHERE status='active'\")
    print('facetas activas', f['k'])
asyncio.run(main())
PY"
```
Expected: `deps ok 0.139.0`, las cinco tablas con `1` y facetas activas que incluyen `ada`, `jekyll` y `hipatia`. Si falta una tabla de jax-platform en `jax_memory_test`, correr la suite de jax-platform una vez (sus migraciones corren al arrancar la app de test): `cd $WTP/backend && $PYP -m pytest -q tests/test_motor_migrations.py`. **Nunca crear tablas a mano.**

- [ ] **Step 5: Commit del ledger**

```bash
git -C $WT add .superpowers/sdd/2026-09-16-emisor-subpipelines-ada/progress.md
git -C $WT commit -m "chore(frente-g): ledger del emisor de sub-pipelines de Ada

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 1: jax-platform · capabilities `delegate` e `integrate`

**Files:**
- Modify: `$WTP/backend/db/migrations.py` (`_CAPABILITY_SEED`, después de la entrada `review`)
- Create: `$WTP/backend/tests/test_capabilities_delegacion.py`

**Interfaces:**
- Produces: filas `capability('delegate')` y `capability('integrate')` con los valores de Global Constraints. `_CAPABILITY_MODE` las incluye solo por comprensión (`{fila[0]: "read_only" for fila in _CAPABILITY_SEED}`), y el tripwire `tests/test_capability_mode.py::test_tripwire_cada_capability_sembrada_declara_su_modo` sigue verde sin tocarlo.

- [ ] **Step 1: Test que falla**

Create `$WTP/backend/tests/test_capabilities_delegacion.py`:

```python
"""Frente G (2026-09-16): capabilities `delegate` e `integrate` del emisor de
sub-pipelines de Ada. Las siembra jax-platform (dueña de `capability`); las
consume jax (jacobs/delegacion.py, executor, output_validator).

Contra la base migrada del fixture `client` (jax_memory_test).
"""
from __future__ import annotations

import json

from db import migrations


async def _fila(key):
    from db.connection import get_pool
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT risk_level, sandbox_only, requires_human_gate, max_execution_minutes, "
                "max_recursion_depth, output_schema, allowed_callers, mode "
                "FROM capability WHERE `key`=%s",
                (key,),
            )
            return await cur.fetchone()


def test_delegate_esta_en_la_semilla_con_su_schema():
    fila = next(f for f in migrations._CAPABILITY_SEED if f[0] == "delegate")
    assert fila[6] == "plan_delegacion.v1"
    assert fila[9] == ["jacobs"]
    assert migrations._CAPABILITY_MODE["delegate"] == "read_only"


def test_integrate_esta_en_la_semilla_sin_schema():
    fila = next(f for f in migrations._CAPABILITY_SEED if f[0] == "integrate")
    assert fila[6] is None
    assert fila[9] == ["jacobs"]
    assert migrations._CAPABILITY_MODE["integrate"] == "read_only"


def test_delegate_sembrada_en_la_base(client):
    fila = client.portal.call(_fila, "delegate")
    assert fila is not None
    risk, sandbox, gate, minutos, recursion, schema, callers, mode = fila
    assert (risk, bool(sandbox), bool(gate), minutos, recursion, schema, mode) == (
        "medium", True, False, 15, 0, "plan_delegacion.v1", "read_only")
    assert json.loads(callers) == ["jacobs"]


def test_integrate_sembrada_en_la_base(client):
    fila = client.portal.call(_fila, "integrate")
    assert fila is not None
    risk, sandbox, gate, minutos, recursion, schema, callers, mode = fila
    assert (risk, bool(sandbox), bool(gate), minutos, recursion, schema, mode) == (
        "low", True, False, 15, 0, None, "read_only")
    assert json.loads(callers) == ["jacobs"]
```

- [ ] **Step 2: Verlo fallar**

```bash
cd $WTP/backend && pwd && JAX_REPO_PATH=$WT $PYP -m pytest -q tests/test_capabilities_delegacion.py 2>&1 | tee -a $L | tail -6
```
Expected: `4 failed`. Los dos puros fallan con `StopIteration` y los dos de base, con `assert None is not None`.

- [ ] **Step 3: Implementar**

En `backend/db/migrations.py`, dentro de `_CAPABILITY_SEED`, después de `("review", ...)`:

```python
    # Frente G (2026-09-16): emisor de sub-pipelines de Ada. `delegate` produce
    # un plan_delegacion.v1 que Jacobs valida estrictamente
    # (jax/las_manos/motor_registry/esquema_plan_delegacion.py) antes de emitir
    # tokens; `integrate` lo agrega Jacobs al padre cuando terminan los hijos.
    # Solo jacobs las pide. Sin capability_motor en la etapa 1: Ada va por HTTP
    # directo (jax/jacobs/models.py::HTTP_FACETS). 15 min = el techo de design.
    ("delegate", "medium", True, False, 15, 0, "plan_delegacion.v1", None, None,
     ["jacobs"], None),
    ("integrate", "low", True, False, 15, 0, None, None, None,
     ["jacobs"], None),
```

- [ ] **Step 4: Verde y suites vecinas**

```bash
cd $WTP/backend && JAX_REPO_PATH=$WT $PYP -m pytest -q tests/test_capabilities_delegacion.py tests/test_capability_mode.py tests/test_motor_migrations.py 2>&1 | tail -3
```
Expected: todo `passed`, con `test_capability_motor_seed_count_is_26` intacto (esta tarea no agrega `capability_motor`).

- [ ] **Step 5: CI de jax-platform y commit**

En `$WTP/.github/workflows/policy.yml`, job `backend-tests-con-db`: si el job corre la suite por directorio, no hay lista que tocar, así que sólo cambia el piso. Medirlo con `cd $WTP/backend && JAX_REPO_PATH=$WT $PYP -m pytest -q 2>&1 | tail -1`, sumar exactamente lo medido (+4) y agregar la línea de historia `# +4 el 2026-09-16 (frente G): test_capabilities_delegacion.py -- delegate/integrate sembradas. Vistos en rojo.`

```bash
git -C $WTP add backend/db/migrations.py backend/tests/test_capabilities_delegacion.py .github/workflows/policy.yml
git -C $WTP commit -m "feat(migraciones): capabilities delegate e integrate para el emisor de sub-pipelines de Ada

Frente G. delegate declara output_schema plan_delegacion.v1; integrate sin
schema. Solo jacobs. 4 tests vistos en rojo.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```
El push y el PR de esta rama van en la Task 14 (con la pantalla de la Task 11). **Se mergea y despliega antes que el PR de jax** (Discrepancia 19).

---
### Task 2: Techos por entorno validados al arrancar

**Files:**
- Create: `jacobs/config_delegacion.py`
- Create: `tests/test_config_delegacion.py`
- Modify: `las_manos/server.py` (`_jacobs_init`, junto a la validación de `config_subpipelines()` de F)

**Interfaces:**
- Produces:
  - `jacobs.config_delegacion.ENV_MAX_HIJOS_POR_STEP = "JAX_ADA_MAX_HIJOS_POR_STEP"`, `ENV_MAX_PIPELINES_ARBOL`, `ENV_MAX_TOKENS_ARBOL`, `ENV_MAX_USD_ARBOL`, `ENV_RESERVA_TOKENS_HIJO`, `ENV_COLA_MAX_ESPERA`, `ENV_COLA_CADENCIA`, `ENV_RESUMEN_HIJO`, `ENV_PROHIBIDA_MAX_DEPTH = "JAX_ADA_MAX_DEPTH"`
  - `ConfigDelegacion(max_hijos_por_step: int, max_pipelines_arbol: int, max_tokens_arbol: int, max_usd_arbol: float, reserva_tokens_hijo: int, cola_max_espera_segundos: int, cola_cadencia_segundos: int, resumen_hijo_caracteres: int)` (dataclass frozen)
  - `config_delegacion() -> ConfigDelegacion` (lanza `ValueError` con el nombre de la variable)

- [ ] **Step 1: Tests que fallan**

Create `tests/test_config_delegacion.py`:

```python
"""Frente G (2026-09-16): techos del emisor de sub-pipelines de Ada, por
entorno, con rango declarado y fail-closed. La profundidad NO es de este
módulo: es JAX_MAX_SUBPIPELINE_DEPTH del frente F (una sola fuente).

ROJO CONTRA LA BASE (E+B+F): ModuleNotFoundError jacobs.config_delegacion y,
creado el módulo, test_las_manos_valida_los_techos_al_arrancar.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ["JAX_DB_NAME"] = "jax_memory_test"

from jacobs import config_delegacion as cd  # noqa: E402

TODAS = [
    cd.ENV_MAX_HIJOS_POR_STEP, cd.ENV_MAX_PIPELINES_ARBOL, cd.ENV_MAX_TOKENS_ARBOL,
    cd.ENV_MAX_USD_ARBOL, cd.ENV_RESERVA_TOKENS_HIJO, cd.ENV_COLA_MAX_ESPERA,
    cd.ENV_COLA_CADENCIA, cd.ENV_RESUMEN_HIJO, cd.ENV_PROHIBIDA_MAX_DEPTH,
]


@pytest.fixture(autouse=True)
def entorno_limpio(monkeypatch):
    for nombre in TODAS:
        monkeypatch.delenv(nombre, raising=False)


def test_nombres_exactos_del_spec():
    assert (cd.ENV_MAX_HIJOS_POR_STEP, cd.ENV_MAX_PIPELINES_ARBOL, cd.ENV_MAX_TOKENS_ARBOL,
            cd.ENV_MAX_USD_ARBOL, cd.ENV_RESERVA_TOKENS_HIJO, cd.ENV_COLA_MAX_ESPERA) == (
        "JAX_ADA_MAX_HIJOS_POR_STEP", "JAX_ADA_MAX_PIPELINES_ARBOL", "JAX_ADA_MAX_TOKENS_ARBOL",
        "JAX_ADA_MAX_USD_ARBOL", "JAX_ADA_RESERVA_TOKENS_HIJO", "JAX_ADA_COLA_MAX_ESPERA_SEGUNDOS")


def test_defaults_del_spec():
    c = cd.config_delegacion()
    assert (c.max_hijos_por_step, c.max_pipelines_arbol, c.max_tokens_arbol,
            c.max_usd_arbol, c.reserva_tokens_hijo, c.cola_max_espera_segundos) == (
        10, 40, 2_000_000, 10.0, 50_000, 3600)
    assert (c.cola_cadencia_segundos, c.resumen_hijo_caracteres) == (10, 2000)


def test_valores_validos_se_leen(monkeypatch):
    monkeypatch.setenv(cd.ENV_MAX_HIJOS_POR_STEP, "2")
    monkeypatch.setenv(cd.ENV_MAX_USD_ARBOL, "0.5")
    monkeypatch.setenv(cd.ENV_RESERVA_TOKENS_HIJO, "0")
    c = cd.config_delegacion()
    assert (c.max_hijos_por_step, c.max_usd_arbol, c.reserva_tokens_hijo) == (2, 0.5, 0)


@pytest.mark.parametrize("nombre, valor", [
    ("JAX_ADA_MAX_HIJOS_POR_STEP", "0"),
    ("JAX_ADA_MAX_HIJOS_POR_STEP", "51"),
    ("JAX_ADA_MAX_PIPELINES_ARBOL", "501"),
    ("JAX_ADA_MAX_TOKENS_ARBOL", "dos millones"),
    ("JAX_ADA_MAX_TOKENS_ARBOL", "999"),
    ("JAX_ADA_MAX_USD_ARBOL", "0"),
    ("JAX_ADA_MAX_USD_ARBOL", "nan"),
    ("JAX_ADA_MAX_USD_ARBOL", "inf"),
    ("JAX_ADA_RESERVA_TOKENS_HIJO", "-1"),
    ("JAX_ADA_COLA_MAX_ESPERA_SEGUNDOS", "59"),
    ("JAX_ADA_COLA_CADENCIA_SEGUNDOS", ""),
    ("JAX_ADA_RESUMEN_HIJO_CARACTERES", "199"),
])
def test_valor_invalido_tumba_con_el_nombre(monkeypatch, nombre, valor):
    monkeypatch.setenv(nombre, valor)
    with pytest.raises(ValueError, match=nombre):
        cd.config_delegacion()


def test_jax_ada_max_depth_no_es_una_segunda_fuente(monkeypatch):
    monkeypatch.setenv("JAX_ADA_MAX_DEPTH", "3")
    with pytest.raises(ValueError, match="JAX_MAX_SUBPIPELINE_DEPTH"):
        cd.config_delegacion()


def test_las_manos_valida_los_techos_al_arrancar():
    fuente = (Path(__file__).resolve().parents[1] / "las_manos" / "server.py").read_text(encoding="utf-8")
    inicio = fuente.index("async def _jacobs_init")
    fin = fuente.index("await jacobs_store.init_tables()", inicio)
    assert "config_delegacion()" in fuente[inicio:fin]
```

- [ ] **Step 2: Verlos fallar**

```bash
cd $WT && pwd && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_config_delegacion.py 2>&1 | tee -a $L | tail -4
```
Expected: `ModuleNotFoundError: No module named 'jacobs.config_delegacion'`, que es un error de colección. El rojo por aserción viene en el Step 4.

- [ ] **Step 3: Crear el módulo**

Create `jacobs/config_delegacion.py`:

```python
"""
Jacobs — Techos del emisor de sub-pipelines de Ada (frente G, 2026-09-16).

Spec: docs/superpowers/specs/2026-09-16-emisor-subpipelines-ada-design.md §4.
Ada lanza sola DENTRO de estos techos; lo que los excede pide aprobación de
Fernando (jacobs/delegacion.py). Se validan al arrancar LAS MANOS
(server.py::_jacobs_init): un valor inválido tumba el arranque.

La profundidad NO vive acá: es JAX_MAX_SUBPIPELINE_DEPTH del contrato del
frente F (jacobs/subpipelines.py). JAX_ADA_MAX_DEPTH presente es un error: dos
fuentes para el mismo límite divergen solas.

Sin caché: ocho os.environ.get por llamada, nada caro que guardar ni invalidar.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

ENV_MAX_HIJOS_POR_STEP = "JAX_ADA_MAX_HIJOS_POR_STEP"
ENV_MAX_PIPELINES_ARBOL = "JAX_ADA_MAX_PIPELINES_ARBOL"
ENV_MAX_TOKENS_ARBOL = "JAX_ADA_MAX_TOKENS_ARBOL"
ENV_MAX_USD_ARBOL = "JAX_ADA_MAX_USD_ARBOL"
ENV_RESERVA_TOKENS_HIJO = "JAX_ADA_RESERVA_TOKENS_HIJO"
ENV_COLA_MAX_ESPERA = "JAX_ADA_COLA_MAX_ESPERA_SEGUNDOS"
ENV_COLA_CADENCIA = "JAX_ADA_COLA_CADENCIA_SEGUNDOS"
ENV_RESUMEN_HIJO = "JAX_ADA_RESUMEN_HIJO_CARACTERES"
ENV_PROHIBIDA_MAX_DEPTH = "JAX_ADA_MAX_DEPTH"

# (defecto, mínimo, máximo). Los rangos son límites del diseño, no gustos: moverlos
# es una decisión de Fernando, no un ajuste de .env.
_ENTEROS: dict[str, tuple[int, int, int]] = {
    ENV_MAX_HIJOS_POR_STEP: (10, 1, 50),
    ENV_MAX_PIPELINES_ARBOL: (40, 1, 500),
    ENV_MAX_TOKENS_ARBOL: (2_000_000, 1_000, 100_000_000),
    ENV_RESERVA_TOKENS_HIJO: (50_000, 0, 10_000_000),
    ENV_COLA_MAX_ESPERA: (3600, 60, 86_400),
    ENV_COLA_CADENCIA: (10, 1, 300),
    ENV_RESUMEN_HIJO: (2000, 200, 20_000),
}
_USD: tuple[float, float, float] = (10.0, 0.01, 1000.0)


@dataclass(frozen=True)
class ConfigDelegacion:
    max_hijos_por_step: int
    max_pipelines_arbol: int
    max_tokens_arbol: int
    max_usd_arbol: float
    reserva_tokens_hijo: int
    cola_max_espera_segundos: int
    cola_cadencia_segundos: int
    resumen_hijo_caracteres: int


def _entero(nombre: str) -> int:
    defecto, minimo, maximo = _ENTEROS[nombre]
    crudo = os.environ.get(nombre)
    if crudo is None:
        return defecto
    try:
        valor = int(crudo.strip())
    except ValueError as exc:
        raise ValueError(f"{nombre}={crudo!r} no es un entero") from exc
    if not minimo <= valor <= maximo:
        raise ValueError(f"{nombre}={valor} fuera de rango [{minimo}, {maximo}]")
    return valor


def _usd() -> float:
    defecto, minimo, maximo = _USD
    crudo = os.environ.get(ENV_MAX_USD_ARBOL)
    if crudo is None:
        return defecto
    try:
        valor = float(crudo.strip())
    except ValueError as exc:
        raise ValueError(f"{ENV_MAX_USD_ARBOL}={crudo!r} no es un número") from exc
    if not math.isfinite(valor) or not minimo <= valor <= maximo:
        raise ValueError(f"{ENV_MAX_USD_ARBOL}={crudo!r} fuera de rango [{minimo}, {maximo}]")
    return valor


def config_delegacion() -> ConfigDelegacion:
    if os.environ.get(ENV_PROHIBIDA_MAX_DEPTH) is not None:
        raise ValueError(
            f"{ENV_PROHIBIDA_MAX_DEPTH} no existe: la profundidad de sub-pipelines es "
            "JAX_MAX_SUBPIPELINE_DEPTH (contrato del frente F), una sola fuente"
        )
    return ConfigDelegacion(
        max_hijos_por_step=_entero(ENV_MAX_HIJOS_POR_STEP),
        max_pipelines_arbol=_entero(ENV_MAX_PIPELINES_ARBOL),
        max_tokens_arbol=_entero(ENV_MAX_TOKENS_ARBOL),
        max_usd_arbol=_usd(),
        reserva_tokens_hijo=_entero(ENV_RESERVA_TOKENS_HIJO),
        cola_max_espera_segundos=_entero(ENV_COLA_MAX_ESPERA),
        cola_cadencia_segundos=_entero(ENV_COLA_CADENCIA),
        resumen_hijo_caracteres=_entero(ENV_RESUMEN_HIJO),
    )
```

- [ ] **Step 4: Rojo por aserción (servidor viejo)**

```bash
cd $WT && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_config_delegacion.py 2>&1 | tee -a $L | tail -3
```
Expected: `1 failed, 16 passed`. El que falla es `test_las_manos_valida_los_techos_al_arrancar`.

- [ ] **Step 5: Validar al arrancar**

En `las_manos/server.py::_jacobs_init`, inmediatamente después del bloque de F que llama a `config_subpipelines()` (y antes de `await jacobs_store.init_tables()`):

```python
    # Frente G (2026-09-16): techos del emisor de sub-pipelines de Ada. Un valor
    # inválido (o JAX_ADA_MAX_DEPTH, que no debe existir) tumba LAS MANOS acá.
    from jacobs.config_delegacion import config_delegacion
    config_delegacion()
```

- [ ] **Step 6: Verde y commit**

```bash
cd $WT && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_config_delegacion.py tests/test_subpipeline_contrato_puro.py 2>&1 | tail -2
git -C $WT add jacobs/config_delegacion.py tests/test_config_delegacion.py las_manos/server.py $L
git -C $WT commit -m "feat(jacobs): techos del emisor de Ada por entorno, validados al arrancar

JAX_ADA_* con rango declarado; JAX_ADA_MAX_DEPTH prohibida (la profundidad es
JAX_MAX_SUBPIPELINE_DEPTH de F). 17 tests puros; 1 visto en rojo por aserción.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```
Expected del primer comando: `17 passed` más los puros de F en verde.

---

### Task 3: Esquema estricto `plan_delegacion.v1` y el validador de salidas

**Files:**
- Create: `las_manos/motor_registry/esquema_plan_delegacion.py`
- Create: `tests/test_plan_delegacion_esquema.py`
- Modify: `las_manos/motor_registry/output_validator.py` (`SCHEMAS`, `_VALIDADORES_ESTRICTOS`, `validate`)
- Modify: `las_manos/_output_validator_test.py` (clase nueva)

**Interfaces:**
- Produces (módulo puro, sin I/O, importable con `PYTHONPATH=las_manos`):
  - `ESQUEMA = "plan_delegacion.v1"`, `CAPABILITY_DELEGAR = "delegate"`, `FACETA_DELEGADORA = "ada"`, `CAPABILITIES_NO_DELEGABLES = frozenset({"integrate", "assemble"})`, `MAX_SUBOBJETIVOS = 50`
  - `class PlanInvalido(ValueError)`
  - `SubObjetivo(id: str, objetivo: str, faceta: str, capability: str, depende_de: tuple[str, ...], skip_on_fail: bool)` y `PlanDelegacion(subobjetivos: tuple[SubObjetivo, ...], integracion_objetivo: str)` (dataclasses frozen)
  - `parsear_plan_delegacion(texto: str) -> PlanDelegacion` (lanza `PlanInvalido`)
  - `validar_texto(texto: str) -> str | None` (None = válido)
  - `orden_topologico(plan: PlanDelegacion) -> tuple[SubObjetivo, ...]` (lanza `PlanInvalido` si hay ciclo)
  - `plan_a_dict(plan: PlanDelegacion) -> dict`
  - `errores_de_catalogo(plan: PlanDelegacion, facetas_activas: frozenset[str], capabilities: dict[str, dict]) -> list[str]`
  - `instrucciones_de_formato(facetas_activas: frozenset[str], capabilities: dict[str, dict]) -> str`
  - `texto_de_reintento(motivo: str) -> str`
  - `output_validator.validate(content, "plan_delegacion.v1")` → `validated=True` sólo si el validador estricto no devuelve motivo; si no, `warning` = motivo.

- [ ] **Step 1: Tests puros que fallan**

Create `tests/test_plan_delegacion_esquema.py`:

```python
"""Frente G (2026-09-16): plan_delegacion.v1 se valida ESTRICTAMENTE (spec
§3.1.2). Nunca se lanza un plan parcial: un solo campo mal y el plan entero es
inválido, con un motivo que Ada puede usar en su único reintento.

ROJO CONTRA LA BASE: ModuleNotFoundError motor_registry.esquema_plan_delegacion.
"""
from __future__ import annotations

import json

import pytest

from motor_registry import esquema_plan_delegacion as esq

CAPS = {
    "analysis": {"allowed_callers": ["jacobs"]},
    "research": {"allowed_callers": ["jacobs"]},
    "delegate": {"allowed_callers": ["jacobs"]},
    "integrate": {"allowed_callers": ["jacobs"]},
    "solo_hyde": {"allowed_callers": ["hyde"]},
}
FACETAS = frozenset({"ada", "jekyll", "hipatia", "kimi"})


def _sub(i, **kw):
    base = {"id": f"s{i}", "objetivo": f"objetivo {i}", "faceta": "jekyll",
            "capability": "analysis", "depende_de": [], "skip_on_fail": False}
    base.update(kw)
    return base


def _texto(subs, integracion=None):
    return json.dumps({"subobjetivos": subs, "integracion": integracion or {"objetivo": "integrá"}})


def test_plan_valido_se_parsea_en_orden():
    plan = esq.parsear_plan_delegacion(_texto([_sub(1), _sub(2, depende_de=["s1"], skip_on_fail=True)]))
    assert [s.id for s in plan.subobjetivos] == ["s1", "s2"]
    assert plan.subobjetivos[1].depende_de == ("s1",)
    assert plan.subobjetivos[1].skip_on_fail is True
    assert plan.integracion_objetivo == "integrá"


def test_acepta_un_solo_cerco_markdown_json():
    plan = esq.parsear_plan_delegacion("```json\n" + _texto([_sub(1)]) + "\n```")
    assert len(plan.subobjetivos) == 1


@pytest.mark.parametrize("texto, fragmento", [
    ("no es json", "no es JSON"),
    ("[]", "objeto JSON"),
    (json.dumps({"subobjetivos": []}), "faltan ['integracion']"),
    (json.dumps({"subobjetivos": [], "integracion": {"objetivo": "x"}, "extra": 1}), "claves no permitidas ['extra']"),
    (_texto([]), "lista no vacía"),
    (_texto([_sub(1, extra=True)]), "claves no permitidas ['extra']"),
    (_texto([{k: v for k, v in _sub(1).items() if k != "faceta"}]), "faltan ['faceta']"),
    (_texto([_sub(1, objetivo="   ")]), "objetivo tiene que ser un texto no vacío"),
    (_texto([_sub(1, skip_on_fail=0)]), "skip_on_fail tiene que ser true o false"),
    (_texto([_sub(1, depende_de="s0")]), "lista de ids"),
    (_texto([_sub(1), _sub(1)]), "repetido"),
    (_texto([_sub(1, depende_de=["s9"])]), "no está en el plan"),
    (_texto([_sub(1, depende_de=["s1"])]), "depende de sí mismo"),
    (_texto([_sub(1)], integracion={"objetivo": ""}), "integracion.objetivo"),
    (_texto([_sub(1, id="x" * 65)]), "excede 64"),
])
def test_invalidos_con_motivo(texto, fragmento):
    with pytest.raises(esq.PlanInvalido) as exc:
        esq.parsear_plan_delegacion(texto)
    assert fragmento in str(exc.value)


def test_ciclo_en_depende_de_se_rechaza():
    texto = _texto([_sub(1, depende_de=["s2"]), _sub(2, depende_de=["s1"]), _sub(3)])
    with pytest.raises(esq.PlanInvalido, match=r"ciclo en depende_de entre \['s1', 's2'\]"):
        esq.parsear_plan_delegacion(texto)


def test_mas_de_cincuenta_subobjetivos_se_rechaza():
    with pytest.raises(esq.PlanInvalido, match="50"):
        esq.parsear_plan_delegacion(_texto([_sub(i) for i in range(51)]))


def test_orden_topologico_respeta_dependencias():
    plan = esq.parsear_plan_delegacion(_texto([
        _sub(1, depende_de=["s3"]), _sub(2), _sub(3, depende_de=["s2"])]))
    assert [s.id for s in esq.orden_topologico(plan)] == ["s2", "s3", "s1"]


def test_validar_texto_devuelve_none_o_motivo():
    assert esq.validar_texto(_texto([_sub(1)])) is None
    assert "no es JSON" in esq.validar_texto("{")


def test_plan_a_dict_vuelve_a_parsear_igual():
    plan = esq.parsear_plan_delegacion(_texto([_sub(1), _sub(2, depende_de=["s1"])]))
    assert esq.parsear_plan_delegacion(json.dumps(esq.plan_a_dict(plan))) == plan


def test_errores_de_catalogo():
    plan = esq.parsear_plan_delegacion(_texto([
        _sub(1, faceta="inventada"),
        _sub(2, capability="no_existe"),
        _sub(3, capability="integrate"),
        _sub(4, capability="solo_hyde"),
        _sub(5, capability="delegate", faceta="jekyll"),
        _sub(6, capability="delegate", faceta="ada"),
    ]))
    errores = esq.errores_de_catalogo(plan, FACETAS, CAPS)
    assert any("'s1'" in e and "tabla `facet`" in e for e in errores)
    assert any("'s2'" in e and "tabla `capability`" in e for e in errores)
    assert any("'s3'" in e and "no se delega" in e for e in errores)
    assert any("'s4'" in e and "jacobs" in e for e in errores)
    assert any("'s5'" in e and "solo ada delega" in e for e in errores)
    assert not any("'s6'" in e for e in errores)


def test_instrucciones_listan_solo_lo_delegable():
    texto = esq.instrucciones_de_formato(FACETAS, CAPS)
    assert "plan_delegacion.v1" in texto
    assert "analysis" in texto and "delegate" in texto
    assert "integrate" not in texto.split("Capabilities delegables:")[1]
    assert "solo_hyde" not in texto


def test_texto_de_reintento_lleva_el_motivo():
    assert "ciclo en depende_de" in esq.texto_de_reintento("ciclo en depende_de entre ['a', 'b']")
```

- [ ] **Step 2: Verlos fallar**

```bash
cd $WT && pwd && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_plan_delegacion_esquema.py 2>&1 | tee -a $L | tail -3
```
Expected: `ModuleNotFoundError: No module named 'motor_registry.esquema_plan_delegacion'`.

- [ ] **Step 3: Implementar el esquema**

Create `las_manos/motor_registry/esquema_plan_delegacion.py`:

```python
"""
LAS MANOS — Motor Registry: esquema ESTRICTO plan_delegacion.v1 (frente G,
2026-09-16).

Spec: docs/superpowers/specs/2026-09-16-emisor-subpipelines-ada-design.md §3.1.
Módulo PURO (sin I/O): lo usan output_validator.py (camino del Motor Registry)
y jacobs/executor.py (camino HTTP directo de Ada), para que haya UN validador.

"Estricto" significa: claves exactas (ni faltan ni sobran), tipos exactos
(skip_on_fail es bool, no 0/1), ids únicos, depende_de solo a ids del mismo
plan, sin autodependencias ni ciclos. Un plan con un solo defecto es inválido
entero: nunca se lanza un plan parcial.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass

ESQUEMA = "plan_delegacion.v1"
CAPABILITY_DELEGAR = "delegate"
FACETA_DELEGADORA = "ada"
# integrate la agrega Jacobs al padre; assemble es el ensamble mecánico del
# padre (executor._assemble_mechanical): ninguna tiene sentido como hijo.
CAPABILITIES_NO_DELEGABLES = frozenset({"integrate", "assemble"})
MAX_SUBOBJETIVOS = 50  # cota del ESQUEMA; el techo operativo es JAX_ADA_MAX_HIJOS_POR_STEP
LARGO_MAX_ID = 64
LARGO_MAX_OBJETIVO = 4000
LARGO_MAX_NOMBRE = 50

_CLAVES_PLAN = frozenset({"subobjetivos", "integracion"})
_CLAVES_SUBOBJETIVO = frozenset({"id", "objetivo", "faceta", "capability", "depende_de", "skip_on_fail"})
_CLAVES_INTEGRACION = frozenset({"objetivo"})


class PlanInvalido(ValueError):
    """El texto no es un plan_delegacion.v1 válido. str(exc) es el motivo."""


@dataclass(frozen=True)
class SubObjetivo:
    id: str
    objetivo: str
    faceta: str
    capability: str
    depende_de: tuple[str, ...]
    skip_on_fail: bool


@dataclass(frozen=True)
class PlanDelegacion:
    subobjetivos: tuple[SubObjetivo, ...]
    integracion_objetivo: str


def _sin_cerco(texto: str) -> str:
    t = texto.strip()
    if t.startswith("```") and t.endswith("```") and len(t) >= 6:
        primera, _, resto = t[3:-3].partition("\n")
        if primera.strip() in ("", "json"):
            return resto.strip()
    return t


def _claves_exactas(obj: object, esperadas: frozenset[str], donde: str) -> dict:
    if not isinstance(obj, dict):
        raise PlanInvalido(f"{donde} tiene que ser un objeto JSON")
    faltan = sorted(esperadas - obj.keys())
    if faltan:
        raise PlanInvalido(f"{donde}: faltan {faltan}")
    sobran = sorted(obj.keys() - esperadas)
    if sobran:
        raise PlanInvalido(f"{donde}: claves no permitidas {sobran}")
    return obj


def _texto(obj: dict, campo: str, largo: int, donde: str) -> str:
    valor = obj[campo]
    if not isinstance(valor, str) or not valor.strip():
        raise PlanInvalido(f"{donde}.{campo} tiene que ser un texto no vacío")
    if len(valor) > largo:
        raise PlanInvalido(f"{donde}.{campo} excede {largo} caracteres")
    return valor.strip()


def orden_topologico(plan: PlanDelegacion) -> tuple[SubObjetivo, ...]:
    por_id = {s.id: s for s in plan.subobjetivos}
    hechos: set[str] = set()
    orden: list[SubObjetivo] = []
    listos = deque(s.id for s in plan.subobjetivos if not s.depende_de)
    en_cola = set(listos)
    while listos:
        ident = listos.popleft()
        orden.append(por_id[ident])
        hechos.add(ident)
        for s in plan.subobjetivos:
            if s.id not in hechos and s.id not in en_cola and set(s.depende_de) <= hechos:
                listos.append(s.id)
                en_cola.add(s.id)
    if len(orden) != len(plan.subobjetivos):
        raise PlanInvalido(f"ciclo en depende_de entre {sorted(set(por_id) - hechos)}")
    return tuple(orden)


def parsear_plan_delegacion(texto: str) -> PlanDelegacion:
    if not isinstance(texto, str):
        raise PlanInvalido("la salida no es texto")
    try:
        datos = json.loads(_sin_cerco(texto))
    except ValueError as exc:
        raise PlanInvalido(f"no es JSON: {exc}") from exc
    _claves_exactas(datos, _CLAVES_PLAN, "plan")
    crudos = datos["subobjetivos"]
    if not isinstance(crudos, list) or not crudos:
        raise PlanInvalido("plan.subobjetivos tiene que ser una lista no vacía")
    if len(crudos) > MAX_SUBOBJETIVOS:
        raise PlanInvalido(f"plan.subobjetivos tiene {len(crudos)}; el esquema admite hasta {MAX_SUBOBJETIVOS}")
    subs: list[SubObjetivo] = []
    ids: set[str] = set()
    for i, crudo in enumerate(crudos):
        donde = f"subobjetivos[{i}]"
        _claves_exactas(crudo, _CLAVES_SUBOBJETIVO, donde)
        ident = _texto(crudo, "id", LARGO_MAX_ID, donde)
        if ident in ids:
            raise PlanInvalido(f"{donde}.id '{ident}' repetido")
        ids.add(ident)
        deps = crudo["depende_de"]
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            raise PlanInvalido(f"{donde}.depende_de tiene que ser una lista de ids (texto)")
        if not isinstance(crudo["skip_on_fail"], bool):
            raise PlanInvalido(f"{donde}.skip_on_fail tiene que ser true o false")
        subs.append(SubObjetivo(
            id=ident,
            objetivo=_texto(crudo, "objetivo", LARGO_MAX_OBJETIVO, donde),
            faceta=_texto(crudo, "faceta", LARGO_MAX_NOMBRE, donde),
            capability=_texto(crudo, "capability", LARGO_MAX_NOMBRE, donde),
            depende_de=tuple(deps),
            skip_on_fail=crudo["skip_on_fail"],
        ))
    for s in subs:
        if len(set(s.depende_de)) != len(s.depende_de):
            raise PlanInvalido(f"'{s.id}' repite dependencias")
        for dep in s.depende_de:
            if dep == s.id:
                raise PlanInvalido(f"'{s.id}' depende de sí mismo")
            if dep not in ids:
                raise PlanInvalido(f"'{s.id}' depende de '{dep}', que no está en el plan")
    integracion = _claves_exactas(datos["integracion"], _CLAVES_INTEGRACION, "integracion")
    plan = PlanDelegacion(tuple(subs), _texto(integracion, "objetivo", LARGO_MAX_OBJETIVO, "integracion"))
    orden_topologico(plan)
    return plan


def validar_texto(texto: str) -> str | None:
    try:
        parsear_plan_delegacion(texto)
    except PlanInvalido as exc:
        return str(exc)
    return None


def plan_a_dict(plan: PlanDelegacion) -> dict:
    return {
        "subobjetivos": [
            {"id": s.id, "objetivo": s.objetivo, "faceta": s.faceta, "capability": s.capability,
             "depende_de": list(s.depende_de), "skip_on_fail": s.skip_on_fail}
            for s in plan.subobjetivos
        ],
        "integracion": {"objetivo": plan.integracion_objetivo},
    }


def errores_de_catalogo(
    plan: PlanDelegacion, facetas_activas: frozenset[str], capabilities: dict[str, dict],
) -> list[str]:
    """Contra la gobernanza REAL (store.get_motor_governance(): facets de E-03 y
    capabilities). Lista vacía = el plan se puede despachar."""
    errores: list[str] = []
    for s in plan.subobjetivos:
        if s.faceta not in facetas_activas:
            errores.append(f"'{s.id}': faceta '{s.faceta}' no existe o no está activa en la tabla `facet`")
        if s.capability in CAPABILITIES_NO_DELEGABLES:
            errores.append(f"'{s.id}': capability '{s.capability}' no se delega (la agrega Jacobs)")
            continue
        entrada = capabilities.get(s.capability)
        if entrada is None:
            errores.append(f"'{s.id}': capability '{s.capability}' no existe en la tabla `capability`")
            continue
        if "jacobs" not in entrada.get("allowed_callers", []):
            errores.append(f"'{s.id}': capability '{s.capability}' no admite a jacobs como caller")
        if s.capability == CAPABILITY_DELEGAR and s.faceta != FACETA_DELEGADORA:
            errores.append(f"'{s.id}': solo {FACETA_DELEGADORA} delega")
    return errores


def instrucciones_de_formato(facetas_activas: frozenset[str], capabilities: dict[str, dict]) -> str:
    """Texto para el LLM (no es UI): se agrega al prompt del step delegate."""
    delegables = sorted(
        k for k, v in capabilities.items()
        if k not in CAPABILITIES_NO_DELEGABLES and "jacobs" in v.get("allowed_callers", [])
    )
    return (
        f"FORMATO DE SALIDA ({ESQUEMA}, obligatorio): respondé SOLO con un objeto JSON, sin texto antes ni después:\n"
        '{"subobjetivos": [{"id": "<texto corto único>", "objetivo": "<tarea autocontenida>", '
        '"faceta": "<faceta>", "capability": "<capability>", "depende_de": ["<id>"], '
        '"skip_on_fail": false}], "integracion": {"objetivo": "<qué hacer con los resultados>"}}\n'
        f"Facetas activas: {', '.join(sorted(facetas_activas))}.\n"
        f"Capabilities delegables: {', '.join(delegables)}.\n"
        "depende_de solo nombra ids del mismo plan y no puede formar ciclos. "
        "skip_on_fail=true deja arrancar a quienes dependen de ese sub-objetivo aunque falle."
    )


def texto_de_reintento(motivo: str) -> str:
    return (
        f"Tu respuesta anterior no es un {ESQUEMA} válido: {motivo}. "
        "Respondé de nuevo, SOLO con el JSON del esquema, sin texto adicional."
    )
```

- [ ] **Step 4: Verde de los puros**

```bash
cd $WT && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_plan_delegacion_esquema.py 2>&1 | tail -2
```
Expected: `25 passed`: 10 tests simples más los 15 casos parametrizados de `test_invalidos_con_motivo`.

- [ ] **Step 5: Test del validador de salidas que falla**

Agregar al final de `las_manos/_output_validator_test.py`, antes de `if __name__ == "__main__":`:

```python
class OutputValidatorPlanDelegacionEstrictoTest(unittest.TestCase):
    """Frente G (2026-09-16): plan_delegacion.v1 se valida ESTRICTO. Antes caía
    en 'Schema no reconocido' (no estaba ni en SCHEMAS ni en la lista de
    pendientes) y fallaba cerrado incluso con un plan perfecto."""

    PLAN = ('{"subobjetivos": [{"id": "a", "objetivo": "x", "faceta": "jekyll", '
            '"capability": "analysis", "depende_de": [], "skip_on_fail": false}], '
            '"integracion": {"objetivo": "y"}}')

    def test_plan_valido_valida(self):
        result = validate(self.PLAN, "plan_delegacion.v1")
        self.assertTrue(result["validated"])
        self.assertIsNone(result["warning"])
        self.assertEqual(result["parsed"]["integracion"], {"objetivo": "y"})

    def test_campos_de_primer_nivel_no_alcanzan(self):
        content = '{"subobjetivos": [{"id": "a"}], "integracion": {"objetivo": "y"}}'
        result = validate(content, "plan_delegacion.v1")
        self.assertFalse(result["validated"])
        self.assertIn("faltan", result["warning"])

    def test_ciclo_no_valida(self):
        content = ('{"subobjetivos": ['
                   '{"id": "a", "objetivo": "x", "faceta": "j", "capability": "c", "depende_de": ["b"], "skip_on_fail": false},'
                   '{"id": "b", "objetivo": "x", "faceta": "j", "capability": "c", "depende_de": ["a"], "skip_on_fail": false}],'
                   ' "integracion": {"objetivo": "y"}}')
        result = validate(content, "plan_delegacion.v1")
        self.assertFalse(result["validated"])
        self.assertIn("ciclo", result["warning"])

    def test_esta_cubierto_para_el_test_de_drift(self):
        from motor_registry.output_validator import SCHEMAS
        self.assertIn("plan_delegacion.v1", SCHEMAS)
```

```bash
cd $WT && PYTHONPATH=las_manos $PY -m pytest -q las_manos/_output_validator_test.py 2>&1 | tee -a $L | tail -3
```
Expected: `4 failed`. `test_plan_valido_valida` y `test_esta_cubierto…` fallan porque el schema "no es reconocido"; `test_campos…` y `test_ciclo…` fallan porque el `warning` no contiene "faltan" ni "ciclo", sólo "no reconocido".

- [ ] **Step 6: Implementar en `output_validator.py`**

Agregar el import después de `from typing import Any`:

```python
from motor_registry.esquema_plan_delegacion import ESQUEMA as _PLAN_DELEGACION
from motor_registry.esquema_plan_delegacion import validar_texto as _validar_plan_delegacion
```

En `SCHEMAS`, al final:

```python
    # Frente G (2026-09-16). Los campos de primer nivel están acá para que el
    # test de drift contra la DB lo vea cubierto; la validación real es la
    # estricta de _VALIDADORES_ESTRICTOS (claves exactas, tipos, ciclos).
    _PLAN_DELEGACION: ["subobjetivos", "integracion"],
```

Después de `_KNOWN_UNIMPLEMENTED_SCHEMAS`:

```python
# Schemas con validador estricto propio (frente G): se aplican ANTES del
# chequeo superficial de campos, que no ve ni tipos ni ciclos.
_VALIDADORES_ESTRICTOS = {
    _PLAN_DELEGACION: _validar_plan_delegacion,
}
```

En `validate`, inmediatamente después del bloque `if schema_name in _KNOWN_UNIMPLEMENTED_SCHEMAS: ... return result`:

```python
    estricto = _VALIDADORES_ESTRICTOS.get(schema_name)
    if estricto is not None:
        motivo = estricto(content)
        if motivo is not None:
            result["warning"] = f"Salida no cumple '{schema_name}': {motivo}"
            return result
        try:
            result["parsed"] = json.loads(content)
        except ValueError:  # fail-soft: el validador estricto ya aceptó el texto; solo un cerco markdown impide re-parsearlo acá, y `parsed` es informativo (el caller usa el content)
            result["parsed"] = None
        result["validated"] = True
        return result
```

- [ ] **Step 7: Verde, drift y commit**

```bash
cd $WT && PYTHONPATH=las_manos $PY -m pytest -q las_manos/_output_validator_test.py 2>&1 | tail -2
cd $WT && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_plan_delegacion_esquema.py las_manos/_worker_tool_loop_test.py 2>&1 | tail -2
cd $WT && $PY -m pytest -q policy/tests/test_no_fail_open_except.py 2>&1 | tail -2
```
Expected: todo verde, y la marca P10 del `except ValueError` nuevo queda aceptada por el escáner.

El test de drift (`las_manos/_output_validator_db_drift_test.py`) no está en CI: se corre a mano contra `jax_memory_test` **después** de la Task 1:
```bash
bash -c "$DBTEST; cd $WT/las_manos && PYTHONPATH=. $PY _output_validator_db_drift_test.py" 2>&1 | tail -3
```
Expected: `OK`. Con `delegate` sembrada y `plan_delegacion.v1` fuera de `SCHEMAS`, este test fallaba.

```bash
git -C $WT add las_manos/motor_registry/esquema_plan_delegacion.py las_manos/motor_registry/output_validator.py las_manos/_output_validator_test.py tests/test_plan_delegacion_esquema.py $L
git -C $WT commit -m "feat(motor_registry): plan_delegacion.v1 con validador estricto (claves, tipos, ciclos)

Un solo módulo puro lo usan output_validator (motor) y el executor (HTTP de
Ada). Antes el schema caía en 'no reconocido' y fallaba cerrado con cualquier
plan. 25 puros + 4 del validador, vistos en rojo.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---
### Task 4: Estados, esquema y consultas del árbol, la cola y las aprobaciones

**Files:**
- Modify: `jacobs/models.py` (`PipelineStatus`, constantes, `Pipeline`)
- Modify: `jacobs/store.py` (`init_tables`, `_INDICES`, `pipeline_create`, `_row_to_pipeline`, sección nueva "Delegación (frente G)" antes de `# Audit events`)
- Create: `jacobs/_arnes_delegacion.py`
- Create: `jacobs/_delegacion_io_test.py` (clases `EsquemaDelegacionTest`, `ConsultasDelegacionTest`, `ExplainDelegacionTest`)
- Create: `tests/test_delegacion_modelos_puro.py`

**Interfaces:**
- Consumes: `jacobs._arnes_ada` (barrera, `una_fila`, `ejecutar`) del frente F; `store.pipeline_create` con `parent_pipeline_id`/`depth` (F).
- Produces:
  - `PipelineStatus.queued | awaiting_approval | waiting_children`; `models.ESTADOS_TERMINALES: frozenset[PipelineStatus]` = {completed, failed, aborted, expired}; `models.CAPABILITY_DELEGAR = "delegate"`, `models.CAPABILITY_INTEGRAR = "integrate"`; `Pipeline.root_pipeline_id: str | None = None`, `Pipeline.queued_at: float | None = None`.
  - Tablas `jacobs_arbol_consumo` y `jacobs_delegacion_aprobaciones`; columnas `jacobs_pipelines.root_pipeline_id VARCHAR(36) NULL` y `queued_at DOUBLE NULL`; índices `idx_jacobs_pipelines_padre (parent_pipeline_id)`, `idx_jacobs_pipelines_raiz (root_pipeline_id, status)`, `idx_jacobs_pipelines_cola (status, queued_at)`.
  - En `store` (todas async salvo las constantes):
    - `SQL_COLA_CANDIDATOS`, `SQL_COLA_VENCIDOS`, `SQL_HIJOS`, `SQL_ARBOL_CONTAR`, `SQL_HIJOS_DEL_PASO_CONTAR`, `SQL_APROBACIONES_PENDIENTES`, `SQL_APROBACION_PENDIENTE_DE`, `SQL_ARBOL_CONSUMO_SUMAR` (str)
    - `pipeline_transicion(pipeline_id: str, desde: tuple[PipelineStatus, ...], hacia: PipelineStatus) -> bool`
    - `pipeline_status(pipeline_id: str) -> PipelineStatus | None`
    - `pipeline_guardar_contexto(pipeline_id: str, current_step_index: int, context: dict) -> None`
    - `pipeline_plan_guardar(pipeline_id: str, plan: list[Step]) -> None`
    - `step_actualizar_input(step_id: str, input_data: dict) -> None`
    - `pipeline_raiz_fijar(pipeline_id: str) -> None`
    - `pipeline_heredar_dueno(hijo_pipeline_id: str, padre_pipeline_id: str) -> None`
    - `pipeline_fijar_cola(pipeline_ids: list[str], queued_at: float) -> int`
    - `pipeline_reencolar(pipeline_id: str, queued_at: float) -> bool`
    - `pipelines_hijos(parent_pipeline_id: str) -> list[Pipeline]`
    - `pipelines_del_arbol_contar(root_pipeline_id: str) -> int`
    - `pipelines_del_arbol_en_estado(root_pipeline_id: str, estados: tuple[PipelineStatus, ...]) -> list[Pipeline]`
    - `pipelines_en_estado(estado: PipelineStatus) -> list[Pipeline]`
    - `hijos_del_paso_contar(parent_pipeline_id: str, parent_step: str) -> int`
    - `cola_candidatos(limite: int) -> list[Pipeline]`, `cola_vencidos(antes_de: float) -> list[Pipeline]`
    - `arbol_consumo_leer(root_pipeline_id: str) -> dict | None`, `arbol_consumo_sumar_en(cur, root_pipeline_id: str, tokens: int, costo_usd: float | None, ahora: float) -> None`, `arbol_sumar_pipelines(root_pipeline_id: str, n: int) -> None`, `arbol_marcar_consumo_incompleto(root_pipeline_id: str) -> None`, `arbol_limpiar_consumo_incompleto(root_pipeline_id: str) -> None`, `arbol_fijar_techos(root_pipeline_id: str, techos: dict) -> None` (claves `techo_tokens_aprobado`, `techo_usd_aprobado`, `techo_pipelines_aprobado`, `techo_hijos_por_step_aprobado`; las ausentes no se tocan)
    - `aprobacion_crear(root_pipeline_id: str, pipeline_id: str, step_id: str, tipo: str, techos: list[dict], plan: dict | None) -> int`, `aprobacion_actualizar_plan(aprobacion_id: int, techos: list[dict], plan: dict) -> None`, `aprobacion_pendiente_de(pipeline_id: str, step_id: str, tipo: str) -> dict | None`, `aprobacion_pendiente_de_raiz(root_pipeline_id: str, tipo: str) -> dict | None`, `aprobaciones_pendientes(limite: int) -> list[dict]`, `aprobacion_obtener(aprobacion_id: int) -> dict | None`, `aprobacion_resolver(aprobacion_id: int, estado: str, por: str, techos_aprobados: dict | None, ahora: float) -> bool`
  - Arnés `jacobs._arnes_delegacion`: `padre_delegador(*, depth=0, root=None, parent=None, mode="autonomous") -> tuple[Pipeline, Step]`, `plan_json(n: int, *, faceta="jekyll", capability="analysis", depende: dict[int, list[int]] | None = None, skip: set[int] | None = None) -> str`, `plan(n, **kw) -> PlanDelegacion`, `vaciar_activos() -> int`, `limpiar_raiz(root: str) -> None`, `eventos(pipeline_id: str, tipo: str) -> list[dict]`, `tokens_emitidos(padre: str) -> int`, `TELEGRAM_OK: dict`, `EXPLAIN(sql: str, args: tuple) -> list[dict]`.

- [ ] **Step 1: Test puro de modelos que falla**

Create `tests/test_delegacion_modelos_puro.py`:

```python
"""Frente G (2026-09-16): estados y campos del emisor de sub-pipelines.
ROJO CONTRA LA BASE: AttributeError en PipelineStatus.queued y en las constantes."""
from __future__ import annotations

import os

os.environ["JAX_DB_NAME"] = "jax_memory_test"

from jacobs import models  # noqa: E402
from motor_registry import esquema_plan_delegacion as esq  # noqa: E402


def test_estados_nuevos():
    assert models.PipelineStatus.queued.value == "queued"
    assert models.PipelineStatus.awaiting_approval.value == "awaiting_approval"
    assert models.PipelineStatus.waiting_children.value == "waiting_children"


def test_terminales_no_incluyen_esperas():
    t = models.ESTADOS_TERMINALES
    assert {s.value for s in t} == {"completed", "failed", "aborted", "expired"}


def test_constantes_de_capability_coinciden_con_el_esquema():
    assert models.CAPABILITY_DELEGAR == esq.CAPABILITY_DELEGAR == "delegate"
    assert models.CAPABILITY_INTEGRAR == "integrate"
    assert models.CAPABILITY_INTEGRAR in esq.CAPABILITIES_NO_DELEGABLES


def test_pipeline_lleva_raiz_y_cola():
    p = models.Pipeline(name="x", invoked_by="plataforma", mode="autonomous")
    assert (p.root_pipeline_id, p.queued_at) == (None, None)
```

```bash
cd $WT && pwd && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_delegacion_modelos_puro.py 2>&1 | tee -a $L | tail -3
```
Expected: `4 failed`.

- [ ] **Step 2: Arnés de pruebas**

Create `jacobs/_arnes_delegacion.py`:

```python
"""Arnés del emisor de sub-pipelines de Ada (frente G, 2026-09-16).

Escribe pipelines, pasos, aprobaciones y consumo: SOLO contra jax_memory_test.
La barrera es la del frente F: se importa jacobs/_arnes_ada PRIMERO, que se niega
a importar con JAX_DB_NAME apuntando a otra base.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

from jacobs import _arnes_ada as ada  # primero: barrera de base de prueba

import json  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402

import aiomysql  # noqa: E402

from jacobs import store  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus  # noqa: E402
from motor_registry.esquema_plan_delegacion import PlanDelegacion, parsear_plan_delegacion  # noqa: E402

TELEGRAM_OK = {"ok": True, "message_id": 1, "error": None}
_NO_TERMINALES = ("pending", "running", "queued", "waiting_children", "awaiting_approval", "interrupted")


async def padre_delegador(
    *, depth: int = 0, root: str | None = None, parent: str | None = None, mode: str = "autonomous",
) -> tuple[Pipeline, Step]:
    """Un pipeline `running` cuyo step 0 es `ada/delegate` ya `completed`: el
    estado exacto en que el executor llama a delegacion.delegar()."""
    pid = str(uuid.uuid4())
    ahora = time.time()
    paso = Step(
        pipeline_id=pid, step_index=0, facet="ada", capability="delegate",
        status=StepStatus.completed, input={"prompt": "delegá"}, timeout_seconds=900,
    )
    p = Pipeline(
        pipeline_id=pid, name="arnes-g-padre", invoked_by="plataforma", user_id="1", tenant_id="1",
        mode=mode, status=PipelineStatus.running, plan=[paso], depth=depth,
        parent_pipeline_id=parent, root_pipeline_id=root or pid,
        context={"objective": "objetivo del padre"}, created_at=ahora, updated_at=ahora,
    )
    await store.pipeline_create(p)
    await store.step_upsert(paso)
    await ada.ejecutar("UPDATE jacobs_pipelines SET owner_ack_at=%s WHERE pipeline_id=%s", (ahora, pid))
    return p, paso


def plan_json(
    n: int, *, faceta: str = "jekyll", capability: str = "analysis",
    depende: dict[int, list[int]] | None = None, skip: set[int] | None = None,
) -> str:
    depende = depende or {}
    skip = skip or set()
    return json.dumps({
        "subobjetivos": [
            {"id": f"s{i}", "objetivo": f"sub-objetivo {i}", "faceta": faceta, "capability": capability,
             "depende_de": [f"s{d}" for d in depende.get(i, [])], "skip_on_fail": i in skip}
            for i in range(n)
        ],
        "integracion": {"objetivo": "integrá los resultados"},
    })


def plan(n: int, **kw) -> PlanDelegacion:
    return parsear_plan_delegacion(plan_json(n, **kw))


async def vaciar_activos() -> int:
    """Pasa a expired todo lo no terminal de jax_memory_test: restos de otras
    corridas llenarían el cupo global de 3 y la cola no admitiría nada."""
    marcas = ",".join(["%s"] * len(_NO_TERMINALES))
    return await ada.ejecutar(
        f"UPDATE jacobs_pipelines SET status='expired' WHERE status IN ({marcas})", _NO_TERMINALES,
    )


async def limpiar_raiz(root: str) -> None:
    marcas = ",".join(["%s"] * len(_NO_TERMINALES))
    await ada.ejecutar(
        f"UPDATE jacobs_pipelines SET status='expired' WHERE root_pipeline_id=%s AND status IN ({marcas})",
        (root, *_NO_TERMINALES),
    )
    await ada.ejecutar("DELETE FROM jacobs_delegacion_aprobaciones WHERE root_pipeline_id=%s", (root,))
    await ada.ejecutar("DELETE FROM jacobs_arbol_consumo WHERE root_pipeline_id=%s", (root,))


async def eventos(pipeline_id: str, tipo: str) -> list[dict]:
    return [e for e in await store.events_by_pipeline(pipeline_id) if e["event_type"] == tipo]


async def tokens_emitidos(padre: str) -> int:
    fila = await ada.una_fila(
        "SELECT COUNT(*) AS n FROM jacobs_subpipeline_tokens WHERE parent_pipeline_id=%s", (padre,))
    return int(fila["n"])


async def EXPLAIN(sql: str, args: tuple) -> list[dict]:  # noqa: N802 -- nombre del comando SQL
    conn = await store.get_conn()
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute("EXPLAIN " + sql, args)
            return list(await cur.fetchall())
    finally:
        conn.close()
```

- [ ] **Step 3: Tests de base que fallan**

Create `jacobs/_delegacion_io_test.py`:

```python
"""Emisor de sub-pipelines de Ada contra una MariaDB REAL (frente G, 2026-09-16).

Necesita el esquema de jax-platform (capability, facet, axioma_usage, model):
corre en el job jacobs-gobernanza-db, que clona jax-platform y aplica SUS
migraciones. Las tablas de Jacobs las crea store.init_tables() en cada setUp.

Barrera: jacobs/_arnes_delegacion importa jacobs/_arnes_ada primero.

Corre con:
  bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -v jacobs/_delegacion_io_test.py"
"""
from __future__ import annotations

from jacobs import _arnes_delegacion as g  # primero: barrera de base de prueba

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import unittest  # noqa: E402
import uuid  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402

from jacobs import _arnes_ada as ada  # noqa: E402
from jacobs import store  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus, Step  # noqa: E402

_MARCA = "arnes-g-relleno"


@unittest.skipUnless(os.getenv("JAX_DB_HOST"), "necesita la MariaDB real (jax_memory_test)")
class _ConBaseG(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await store.init_tables()
        await g.vaciar_activos()
        self.raices: list[str] = []
        for destino in ("jacobs.policy.check_kill_switch", "jacobs.executor.check_kill_switch"):
            freno = patch(destino, return_value=False)
            freno.start()
            self.addCleanup(freno.stop)
        telegram = patch("jacobs.reaper.send_telegram_alert", AsyncMock(return_value=g.TELEGRAM_OK))
        self.telegram = telegram.start()
        self.addCleanup(telegram.stop)
        entorno = patch.dict(os.environ)
        entorno.start()
        self.addCleanup(entorno.stop)
        # Freno en vuelo de B (correr_con_interruptor) apuntando a una ruta que no
        # existe: DBTEST carga /etc/jax/.env y los tests no miran el freno real.
        os.environ["JAX_KILL_SWITCH_PATH"] = "/tmp/jax-frente-g-sin-freno/PAUSE"
        for nombre in list(os.environ):
            if nombre.startswith("JAX_ADA_") or nombre in ("JAX_MAX_SUBPIPELINE_DEPTH", "JAX_SUBPIPELINE_TOKEN_TTL_SECONDS"):
                os.environ.pop(nombre)

    async def asyncTearDown(self):
        for root in self.raices:
            await g.limpiar_raiz(root)
        await ada.ejecutar("DELETE FROM jacobs_pipelines WHERE name=%s", (_MARCA,))

    async def padre(self, **kw):
        p, paso = await g.padre_delegador(**kw)
        self.raices.append(p.root_pipeline_id)
        return p, paso

    async def pipeline(self, *, status: PipelineStatus, root: str | None = None, parent: str | None = None,
                       queued_at: float | None = None, name: str = _MARCA) -> Pipeline:
        pid = str(uuid.uuid4())
        ahora = time.time()
        p = Pipeline(pipeline_id=pid, name=name, invoked_by="ada", mode="autonomous", status=status,
                     root_pipeline_id=root or pid, parent_pipeline_id=parent, queued_at=queued_at,
                     created_at=ahora, updated_at=ahora)
        await store.pipeline_create(p)
        return p


class EsquemaDelegacionTest(_ConBaseG):
    async def test_init_tables_crea_las_tablas_nuevas(self):
        await ada.ejecutar("DROP TABLE IF EXISTS jacobs_arbol_consumo")
        await ada.ejecutar("DROP TABLE IF EXISTS jacobs_delegacion_aprobaciones")
        await store.init_tables()
        for tabla in ("jacobs_arbol_consumo", "jacobs_delegacion_aprobaciones"):
            fila = await ada.una_fila(
                "SELECT COUNT(*) AS n FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s",
                (tabla,))
            self.assertEqual(fila["n"], 1, tabla)

    async def test_columnas_e_indices_aun_si_la_tabla_ya_existia(self):
        for indice in ("idx_jacobs_pipelines_raiz", "idx_jacobs_pipelines_cola", "idx_jacobs_pipelines_padre"):
            await ada.ejecutar(f"DROP INDEX IF EXISTS {indice} ON jacobs_pipelines")
        await ada.ejecutar(
            "ALTER TABLE jacobs_pipelines DROP COLUMN IF EXISTS root_pipeline_id, DROP COLUMN IF EXISTS queued_at")
        await store.init_tables()
        sql = ("SELECT GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) AS cols FROM information_schema.STATISTICS "
               "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND INDEX_NAME=%s")
        esperados = {
            ("jacobs_pipelines", "idx_jacobs_pipelines_padre"): "parent_pipeline_id",
            ("jacobs_pipelines", "idx_jacobs_pipelines_raiz"): "root_pipeline_id,status",
            ("jacobs_pipelines", "idx_jacobs_pipelines_cola"): "status,queued_at",
            ("jacobs_delegacion_aprobaciones", "idx_delegacion_aprobaciones_estado"): "estado,creada_at",
            ("jacobs_delegacion_aprobaciones", "idx_delegacion_aprobaciones_raiz"): "root_pipeline_id,estado",
            ("jacobs_delegacion_aprobaciones", "idx_delegacion_aprobaciones_paso"): "pipeline_id,step_id,estado",
        }
        for (tabla, indice), cols in esperados.items():
            self.assertEqual((await ada.una_fila(sql, (tabla, indice)))["cols"], cols, indice)

    async def test_init_tables_idempotente(self):
        await store.init_tables()
        await store.init_tables()
        fila = await ada.una_fila(
            "SELECT COUNT(*) AS n FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() "
            "AND TABLE_NAME='jacobs_pipelines' AND COLUMN_NAME IN ('root_pipeline_id','queued_at')")
        self.assertEqual(fila["n"], 2)

    async def test_pipeline_create_conserva_raiz_y_cola(self):
        p = await self.pipeline(status=PipelineStatus.queued, root="raiz-x", queued_at=123.5)
        leido = await store.pipeline_get(p.pipeline_id)
        self.assertEqual((leido.root_pipeline_id, leido.queued_at, leido.status),
                         ("raiz-x", 123.5, PipelineStatus.queued))


class ConsultasDelegacionTest(_ConBaseG):
    async def test_transicion_es_condicional(self):
        p = await self.pipeline(status=PipelineStatus.queued, queued_at=time.time())
        self.assertTrue(await store.pipeline_transicion(p.pipeline_id, (PipelineStatus.queued,), PipelineStatus.pending))
        self.assertFalse(await store.pipeline_transicion(p.pipeline_id, (PipelineStatus.queued,), PipelineStatus.pending))
        self.assertEqual(await store.pipeline_status(p.pipeline_id), PipelineStatus.pending)

    async def test_cola_es_fifo_y_excluye_los_que_esperan_dependencias(self):
        ahora = time.time()
        tarde = await self.pipeline(status=PipelineStatus.queued, queued_at=ahora + 2)
        temprano = await self.pipeline(status=PipelineStatus.queued, queued_at=ahora + 1)
        await self.pipeline(status=PipelineStatus.queued, queued_at=None)
        ids = [c.pipeline_id for c in await store.cola_candidatos(limite=10)]
        self.assertEqual(ids[:2], [temprano.pipeline_id, tarde.pipeline_id])
        self.assertEqual(len([i for i in ids if i in (temprano.pipeline_id, tarde.pipeline_id)]), 2)

    async def test_cola_vencidos(self):
        viejo = await self.pipeline(status=PipelineStatus.queued, queued_at=time.time() - 5000)
        nuevo = await self.pipeline(status=PipelineStatus.queued, queued_at=time.time())
        ids = {c.pipeline_id for c in await store.cola_vencidos(time.time() - 3600)}
        self.assertIn(viejo.pipeline_id, ids)
        self.assertNotIn(nuevo.pipeline_id, ids)

    async def test_fijar_cola_solo_toca_los_null_en_queued(self):
        esperando = await self.pipeline(status=PipelineStatus.queued, queued_at=None)
        ya = await self.pipeline(status=PipelineStatus.queued, queued_at=10.0)
        n = await store.pipeline_fijar_cola([esperando.pipeline_id, ya.pipeline_id], 99.0)
        self.assertEqual(n, 1)
        self.assertEqual((await store.pipeline_get(ya.pipeline_id)).queued_at, 10.0)
        self.assertEqual((await store.pipeline_get(esperando.pipeline_id)).queued_at, 99.0)

    async def test_reencolar_conserva_la_espera_de_dependencias(self):
        dep = await self.pipeline(status=PipelineStatus.awaiting_approval, queued_at=None)
        listo = await self.pipeline(status=PipelineStatus.awaiting_approval, queued_at=1.0)
        self.assertTrue(await store.pipeline_reencolar(dep.pipeline_id, 50.0))
        self.assertTrue(await store.pipeline_reencolar(listo.pipeline_id, 50.0))
        self.assertIsNone((await store.pipeline_get(dep.pipeline_id)).queued_at)
        self.assertEqual((await store.pipeline_get(listo.pipeline_id)).queued_at, 50.0)
        self.assertFalse(await store.pipeline_reencolar(listo.pipeline_id, 60.0))

    async def test_arbol_hijos_y_conteos(self):
        raiz = await self.pipeline(status=PipelineStatus.waiting_children)
        h1 = await self.pipeline(status=PipelineStatus.queued, root=raiz.pipeline_id, parent=raiz.pipeline_id)
        await self.pipeline(status=PipelineStatus.completed, root=raiz.pipeline_id, parent=raiz.pipeline_id)
        self.assertEqual(await store.pipelines_del_arbol_contar(raiz.pipeline_id), 3)
        self.assertEqual(len(await store.pipelines_hijos(raiz.pipeline_id)), 2)
        en_cola = await store.pipelines_del_arbol_en_estado(raiz.pipeline_id, (PipelineStatus.queued,))
        self.assertEqual([p.pipeline_id for p in en_cola], [h1.pipeline_id])

    async def test_consumo_del_arbol_acumula_y_marca(self):
        raiz = "raiz-consumo-" + uuid.uuid4().hex[:8]
        self.raices.append(raiz)
        conn = await store.get_conn()
        try:
            async with conn.cursor() as cur:
                await store.arbol_consumo_sumar_en(cur, raiz, 100, 0.25, time.time())
                await store.arbol_consumo_sumar_en(cur, raiz, 50, None, time.time())
        finally:
            conn.close()
        await store.arbol_sumar_pipelines(raiz, 3)
        await store.arbol_marcar_consumo_incompleto(raiz)
        await store.arbol_fijar_techos(raiz, {"techo_tokens_aprobado": 9000})
        fila = await store.arbol_consumo_leer(raiz)
        self.assertEqual((fila["tokens_total"], fila["tokens_sin_precio"], fila["pipelines_total"]), (150, 50, 4))
        self.assertAlmostEqual(float(fila["costo_usd_total"]), 0.25)
        self.assertEqual((fila["consumo_incompleto"], fila["techo_tokens_aprobado"], fila["techo_usd_aprobado"]),
                         (1, 9000, None))
        await store.arbol_limpiar_consumo_incompleto(raiz)
        self.assertEqual((await store.arbol_consumo_leer(raiz))["consumo_incompleto"], 0)

    async def test_aprobacion_se_resuelve_una_sola_vez(self):
        raiz = "raiz-aprob-" + uuid.uuid4().hex[:8]
        self.raices.append(raiz)
        aid = await store.aprobacion_crear(raiz, raiz, "paso-1", "excede",
                                           [{"techo": "tokens_arbol", "proyectado": 10, "limite": 5, "aprobable": True}],
                                           {"subobjetivos": [], "integracion": {"objetivo": "x"}})
        self.assertEqual((await store.aprobacion_pendiente_de(raiz, "paso-1", "excede"))["id"], aid)
        self.assertIn(aid, [a["id"] for a in await store.aprobaciones_pendientes(100)])
        self.assertTrue(await store.aprobacion_resolver(aid, "aprobada", "1", {"techo_tokens_aprobado": 10}, time.time()))
        self.assertFalse(await store.aprobacion_resolver(aid, "rechazada", "1", None, time.time()))
        fila = await store.aprobacion_obtener(aid)
        self.assertEqual((fila["estado"], fila["resuelta_por"]), ("aprobada", "1"))
        self.assertEqual(fila["techos"][0]["techo"], "tokens_arbol")
        self.assertIsNone(await store.aprobacion_pendiente_de(raiz, "paso-1", "excede"))

    async def test_step_input_y_plan_se_guardan(self):
        p, paso = await self.padre()
        await store.step_actualizar_input(paso.step_id, {"prompt": "nuevo"})
        paso.input = {"prompt": "nuevo"}
        await store.pipeline_plan_guardar(p.pipeline_id, [paso])
        self.assertEqual((await store.steps_by_pipeline(p.pipeline_id))[0].input, {"prompt": "nuevo"})
        self.assertEqual((await store.pipeline_get(p.pipeline_id)).plan[0].input, {"prompt": "nuevo"})


class ExplainDelegacionTest(_ConBaseG):
    """Política 1 de LAS CUATRO DEL RENDIMIENTO: EXPLAIN sobre la SQL REAL
    (las constantes de store), con volumen, no el diseño en la cabeza."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        ahora = time.time()
        filas = [(str(uuid.uuid4()), _MARCA, "ada", "autonomous", "completed", "[]", "{}", ahora, ahora,
                  "raiz-relleno-" + str(i % 40), None) for i in range(2000)]
        filas += [(str(uuid.uuid4()), _MARCA, "ada", "autonomous", "queued", "[]", "{}", ahora, ahora,
                   "raiz-relleno-q", ahora + i) for i in range(40)]
        conn = await store.get_conn()
        try:
            async with conn.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO jacobs_pipelines (pipeline_id, name, invoked_by, mode, status, plan, context_refs, "
                    "created_at, updated_at, root_pipeline_id, queued_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    filas)
                aprobaciones = [("raiz-relleno-a", filas[i][0], "paso", "excede", "[]",
                                 "aprobada" if i % 50 else "pendiente", ahora + i) for i in range(600)]
                await cur.executemany(
                    "INSERT INTO jacobs_delegacion_aprobaciones "
                    "(root_pipeline_id, pipeline_id, step_id, tipo, techos, estado, creada_at) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s)", aprobaciones)
                await cur.execute("ANALYZE TABLE jacobs_pipelines")
                await cur.fetchall()
                await cur.execute("ANALYZE TABLE jacobs_delegacion_aprobaciones")
                await cur.fetchall()
        finally:
            conn.close()

    async def asyncTearDown(self):
        await ada.ejecutar("DELETE FROM jacobs_delegacion_aprobaciones WHERE root_pipeline_id='raiz-relleno-a'")
        await super().asyncTearDown()

    def _usa(self, plan, indice):
        claves = [f.get("key") for f in plan]
        self.assertIn(indice, claves, plan)
        self.assertFalse(any("filesort" in (f.get("Extra") or "") for f in plan), plan)

    async def test_candidatos_de_la_cola(self):
        self._usa(await g.EXPLAIN(store.SQL_COLA_CANDIDATOS, (10,)), "idx_jacobs_pipelines_cola")

    async def test_vencidos_de_la_cola(self):
        self._usa(await g.EXPLAIN(store.SQL_COLA_VENCIDOS, (time.time(),)), "idx_jacobs_pipelines_cola")

    async def test_hijos_de_un_padre(self):
        self._usa(await g.EXPLAIN(store.SQL_HIJOS, ("raiz-relleno-3",)), "idx_jacobs_pipelines_padre")

    async def test_contar_el_arbol(self):
        self._usa(await g.EXPLAIN(store.SQL_ARBOL_CONTAR, ("raiz-relleno-3",)), "idx_jacobs_pipelines_raiz")

    async def test_aprobaciones_pendientes(self):
        self._usa(await g.EXPLAIN(store.SQL_APROBACIONES_PENDIENTES, (100,)), "idx_delegacion_aprobaciones_estado")


if __name__ == "__main__":
    unittest.main()
```

```bash
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_delegacion_io_test.py" 2>&1 | tee -a $L | tail -8
```
Expected: los 18 en rojo (`failed` o `error`), con `AttributeError` (`root_pipeline_id`, `pipeline_transicion`, `SQL_COLA_CANDIDATOS`…), o error de colección si `_arnes_delegacion` ya importa algo que falta. Un rojo por conexión no cuenta: arreglar el entorno y repetir.

- [ ] **Step 4: Implementar los modelos**

En `jacobs/models.py`, dentro de `class PipelineStatus`, después de `expired`:

```python
    # Frente G (2026-09-16): emisor de sub-pipelines de Ada.
    queued            = "queued"             # hijo esperando cupo (queued_at NOT NULL) o dependencias (NULL)
    awaiting_approval = "awaiting_approval"  # delegación que excede un techo: espera a Fernando
    waiting_children  = "waiting_children"   # padre que ya delegó: fuera del cupo hasta que terminan sus hijos
```

Después de la clase `StepStatus`:

```python
ESTADOS_TERMINALES = frozenset({
    PipelineStatus.completed, PipelineStatus.failed, PipelineStatus.aborted, PipelineStatus.expired,
})

# Frente G: capabilities del emisor. Mismo valor que
# las_manos/motor_registry/esquema_plan_delegacion.py::CAPABILITY_DELEGAR (el
# test tests/test_delegacion_modelos_puro.py lo exige): models no importa
# motor_registry porque jax-platform importa jacobs sin las_manos en su path.
CAPABILITY_DELEGAR = "delegate"
CAPABILITY_INTEGRAR = "integrate"
```

En `class Pipeline`, después de los campos de F (`parent_pipeline_id`, `depth`):

```python
    # Frente G: raíz del árbol (índice para contar y frenar el árbol) y el
    # instante desde el que un hijo espera cupo (NULL = espera dependencias).
    root_pipeline_id:   str | None = None
    queued_at:          float | None = None
```

- [ ] **Step 5: Implementar el store**

En `jacobs/store.py::init_tables`, en la lista de columnas de `jacobs_pipelines` (después de las de F):

```python
                # Frente G (2026-09-16): raíz del árbol de delegación y espera
                # en cola. INSTANT explícito, mismo criterio que F.
                ("root_pipeline_id", "ALTER TABLE jacobs_pipelines ADD COLUMN "
                    "root_pipeline_id VARCHAR(36) NULL, ALGORITHM=INSTANT"),
                ("queued_at", "ALTER TABLE jacobs_pipelines ADD COLUMN "
                    "queued_at DOUBLE NULL, ALGORITHM=INSTANT"),
```

En `_INDICES`, al final:

```python
    # Frente G (2026-09-16). Acotados: jacobs_pipelines es tabla viva de producción.
    # padre -> hijos de un padre (cascada, dependencias, reanudar);
    # raiz,status -> contar el árbol y frenar sus no-arrancados;
    # status,queued_at -> la cola FIFO sin filesort.
    ("jacobs_pipelines", "idx_jacobs_pipelines_padre",
     "CREATE INDEX idx_jacobs_pipelines_padre ON jacobs_pipelines (parent_pipeline_id) "
     "ALGORITHM=INPLACE LOCK=NONE", True),
    ("jacobs_pipelines", "idx_jacobs_pipelines_raiz",
     "CREATE INDEX idx_jacobs_pipelines_raiz ON jacobs_pipelines (root_pipeline_id, status) "
     "ALGORITHM=INPLACE LOCK=NONE", True),
    ("jacobs_pipelines", "idx_jacobs_pipelines_cola",
     "CREATE INDEX idx_jacobs_pipelines_cola ON jacobs_pipelines (status, queued_at) "
     "ALGORITHM=INPLACE LOCK=NONE", True),
```

En `init_tables`, después del `CREATE TABLE IF NOT EXISTS jacobs_subpipeline_tokens` de F y **antes** del bucle de `_INDICES`:

```python
            # Frente G (2026-09-16): consumo vivo por árbol. Lo suman los
            # escritores de uso EN LA MISMA TRANSACCIÓN que axioma_usage
            # (arbol_consumo_sumar_en). Tabla nueva: índices en el CREATE.
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS jacobs_arbol_consumo (
                    root_pipeline_id              VARCHAR(36)   NOT NULL PRIMARY KEY,
                    pipelines_total               INT           NOT NULL DEFAULT 0,
                    tokens_total                  BIGINT        NOT NULL DEFAULT 0,
                    costo_usd_total               DECIMAL(14,6) NOT NULL DEFAULT 0,
                    tokens_sin_precio             BIGINT        NOT NULL DEFAULT 0,
                    consumo_incompleto            TINYINT(1)    NOT NULL DEFAULT 0,
                    techo_tokens_aprobado         BIGINT        NULL,
                    techo_usd_aprobado            DECIMAL(14,6) NULL,
                    techo_pipelines_aprobado      INT           NULL,
                    techo_hijos_por_step_aprobado INT           NULL,
                    actualizado_at                DOUBLE        NOT NULL
                ) ENGINE=InnoDB
            """)
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS jacobs_delegacion_aprobaciones (
                    id               BIGINT AUTO_INCREMENT PRIMARY KEY,
                    root_pipeline_id VARCHAR(36) NOT NULL,
                    pipeline_id      VARCHAR(36) NOT NULL,
                    step_id          VARCHAR(36) NOT NULL,
                    tipo             VARCHAR(16) NOT NULL,
                    techos           JSON        NOT NULL,
                    plan             JSON        NULL,
                    estado           VARCHAR(16) NOT NULL,
                    creada_at        DOUBLE      NOT NULL,
                    resuelta_at      DOUBLE      NULL,
                    resuelta_por     VARCHAR(50) NULL,
                    techos_aprobados JSON        NULL,
                    INDEX idx_delegacion_aprobaciones_estado (estado, creada_at),
                    INDEX idx_delegacion_aprobaciones_raiz (root_pipeline_id, estado),
                    INDEX idx_delegacion_aprobaciones_paso (pipeline_id, step_id, estado)
                ) ENGINE=InnoDB
            """)
```

Reemplazar `pipeline_create` (sobre la versión de F) por:

```python
async def pipeline_create(p: Pipeline) -> None:
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO jacobs_pipelines
                    (pipeline_id, name, invoked_by, mode, status,
                     plan, current_step_index, max_steps, context_refs,
                     created_at, updated_at, user_id, tenant_id,
                     parent_pipeline_id, depth, root_pipeline_id, queued_at)
                VALUES (%s,%s,%s,%s,%s, %s,%s,%s,%s, %s,%s, %s,%s, %s,%s, %s,%s)
                """,
                (
                    p.pipeline_id, p.name, p.invoked_by, p.mode, p.status.value,
                    json.dumps([s.model_dump() for s in p.plan], ensure_ascii=False),
                    p.current_step_index, p.max_steps,
                    json.dumps(p.context, ensure_ascii=False),
                    p.created_at, p.updated_at,
                    p.user_id, p.tenant_id,
                    p.parent_pipeline_id, p.depth, p.root_pipeline_id, p.queued_at,
                ),
            )
    finally:
        conn.close()
```

En `_row_to_pipeline`, después de `depth=...` (F):

```python
        root_pipeline_id=row.get("root_pipeline_id"),
        queued_at=row.get("queued_at"),
```

Antes de `#  Audit events`, sección nueva:

```python
# ----------------------------------------------------------------
#  Delegación de Ada (frente G, 2026-09-16)
# ----------------------------------------------------------------
# Cada decisión de estado es UN UPDATE condicionado (pipeline_transicion): dos
# caminos que compiten (despachador y cancelación, reanudación y reaper) no
# pueden pisarse; el que pierde ve rowcount 0.

SQL_COLA_CANDIDATOS = """
    SELECT * FROM jacobs_pipelines
     WHERE status = 'queued' AND queued_at IS NOT NULL
     ORDER BY queued_at
     LIMIT %s
"""
SQL_COLA_VENCIDOS = """
    SELECT * FROM jacobs_pipelines
     WHERE status = 'queued' AND queued_at IS NOT NULL AND queued_at < %s
"""
SQL_HIJOS = "SELECT * FROM jacobs_pipelines WHERE parent_pipeline_id = %s"
SQL_ARBOL_CONTAR = "SELECT COUNT(*) FROM jacobs_pipelines WHERE root_pipeline_id = %s"
SQL_HIJOS_DEL_PASO_CONTAR = """
    SELECT COUNT(*) FROM jacobs_subpipeline_tokens
     WHERE parent_pipeline_id = %s AND parent_step = %s AND hijo_pipeline_id IS NOT NULL
"""
SQL_APROBACIONES_PENDIENTES = """
    SELECT a.id, a.root_pipeline_id, a.pipeline_id, a.step_id, a.tipo, a.techos, a.creada_at,
           p.name AS pipeline_name
      FROM jacobs_delegacion_aprobaciones a
      JOIN jacobs_pipelines p ON p.pipeline_id = a.pipeline_id
     WHERE a.estado = 'pendiente'
     ORDER BY a.creada_at
     LIMIT %s
"""
SQL_APROBACION_PENDIENTE_DE = """
    SELECT * FROM jacobs_delegacion_aprobaciones
     WHERE pipeline_id = %s AND step_id = %s AND tipo = %s AND estado = 'pendiente'
"""
SQL_ARBOL_CONSUMO_SUMAR = """
    INSERT INTO jacobs_arbol_consumo
        (root_pipeline_id, pipelines_total, tokens_total, costo_usd_total, tokens_sin_precio, actualizado_at)
    VALUES (%s, 1, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        tokens_total      = tokens_total + VALUE(tokens_total),
        costo_usd_total   = costo_usd_total + VALUE(costo_usd_total),
        tokens_sin_precio = tokens_sin_precio + VALUE(tokens_sin_precio),
        actualizado_at    = VALUE(actualizado_at)
"""


async def _ejecutar(sql: str, args: tuple = ()) -> int:
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(sql, args)
            return cur.rowcount
    finally:
        conn.close()


async def _filas(sql: str, args: tuple = ()) -> list[dict]:
    conn = await get_conn()
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(sql, args)
            return list(await cur.fetchall())
    finally:
        conn.close()


async def pipeline_transicion(
    pipeline_id: str, desde: tuple[PipelineStatus, ...], hacia: PipelineStatus,
) -> bool:
    marcas = ",".join(["%s"] * len(desde))
    n = await _ejecutar(
        f"UPDATE jacobs_pipelines SET status=%s, updated_at=%s WHERE pipeline_id=%s AND status IN ({marcas})",
        (hacia.value, time.time(), pipeline_id, *(d.value for d in desde)),
    )
    return n == 1


async def pipeline_status(pipeline_id: str) -> PipelineStatus | None:
    filas = await _filas("SELECT status FROM jacobs_pipelines WHERE pipeline_id=%s", (pipeline_id,))
    return PipelineStatus(filas[0]["status"]) if filas else None


async def pipeline_guardar_contexto(pipeline_id: str, current_step_index: int, context: dict) -> None:
    """Como pipeline_update_status pero SIN tocar status: después de una ola el
    executor ya no puede pisar un aborted/waiting_children escrito por otro camino."""
    await _ejecutar(
        "UPDATE jacobs_pipelines SET current_step_index=%s, context_refs=%s, updated_at=%s WHERE pipeline_id=%s",
        (current_step_index, json.dumps(context, ensure_ascii=False), time.time(), pipeline_id),
    )


async def pipeline_plan_guardar(pipeline_id: str, plan: list[Step]) -> None:
    await _ejecutar(
        "UPDATE jacobs_pipelines SET plan=%s, updated_at=%s WHERE pipeline_id=%s",
        (json.dumps([s.model_dump() for s in plan], ensure_ascii=False), time.time(), pipeline_id),
    )


async def step_actualizar_input(step_id: str, input_data: dict) -> None:
    """step_upsert NO actualiza input_ref en el ON DUPLICATE KEY (a propósito:
    el input se fija al planificar). La integración es la única excepción."""
    await _ejecutar(
        "UPDATE jacobs_steps SET input_ref=%s WHERE step_id=%s",
        (json.dumps(input_data, ensure_ascii=False), step_id),
    )


async def pipeline_raiz_fijar(pipeline_id: str) -> None:
    await _ejecutar(
        "UPDATE jacobs_pipelines SET root_pipeline_id=pipeline_id WHERE pipeline_id=%s AND root_pipeline_id IS NULL",
        (pipeline_id,),
    )


async def pipeline_heredar_dueno(hijo_pipeline_id: str, padre_pipeline_id: str) -> None:
    await _ejecutar(
        "UPDATE jacobs_pipelines h JOIN jacobs_pipelines p ON p.pipeline_id=%s "
        "SET h.owner_ack_at=p.owner_ack_at WHERE h.pipeline_id=%s",
        (padre_pipeline_id, hijo_pipeline_id),
    )


async def pipeline_fijar_cola(pipeline_ids: list[str], queued_at: float) -> int:
    if not pipeline_ids:
        return 0
    marcas = ",".join(["%s"] * len(pipeline_ids))
    return await _ejecutar(
        f"UPDATE jacobs_pipelines SET queued_at=%s, updated_at=%s "
        f"WHERE pipeline_id IN ({marcas}) AND status='queued' AND queued_at IS NULL",
        (queued_at, time.time(), *pipeline_ids),
    )


async def pipeline_reencolar(pipeline_id: str, queued_at: float) -> bool:
    """awaiting_approval -> queued. Un hijo que esperaba dependencias (queued_at
    NULL) vuelve a esperarlas; uno que esperaba cupo reinicia su espera."""
    n = await _ejecutar(
        "UPDATE jacobs_pipelines SET status='queued', "
        "queued_at = CASE WHEN queued_at IS NULL THEN NULL ELSE %s END, updated_at=%s "
        "WHERE pipeline_id=%s AND status='awaiting_approval'",
        (queued_at, time.time(), pipeline_id),
    )
    return n == 1


async def pipelines_hijos(parent_pipeline_id: str) -> list[Pipeline]:
    filas = await _filas(SQL_HIJOS, (parent_pipeline_id,))
    return sorted((_row_to_pipeline(f) for f in filas), key=lambda p: p.created_at)


async def pipelines_del_arbol_contar(root_pipeline_id: str) -> int:
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(SQL_ARBOL_CONTAR, (root_pipeline_id,))
            (n,) = await cur.fetchone()
            return int(n)
    finally:
        conn.close()


async def pipelines_del_arbol_en_estado(
    root_pipeline_id: str, estados: tuple[PipelineStatus, ...],
) -> list[Pipeline]:
    marcas = ",".join(["%s"] * len(estados))
    filas = await _filas(
        f"SELECT * FROM jacobs_pipelines WHERE root_pipeline_id=%s AND status IN ({marcas})",
        (root_pipeline_id, *(e.value for e in estados)),
    )
    return [_row_to_pipeline(f) for f in filas]


async def pipelines_en_estado(estado: PipelineStatus) -> list[Pipeline]:
    return [_row_to_pipeline(f) for f in await _filas(
        "SELECT * FROM jacobs_pipelines WHERE status=%s", (estado.value,))]


async def hijos_del_paso_contar(parent_pipeline_id: str, parent_step: str) -> int:
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(SQL_HIJOS_DEL_PASO_CONTAR, (parent_pipeline_id, parent_step))
            (n,) = await cur.fetchone()
            return int(n)
    finally:
        conn.close()


async def cola_candidatos(limite: int) -> list[Pipeline]:
    return [_row_to_pipeline(f) for f in await _filas(SQL_COLA_CANDIDATOS, (limite,))]


async def cola_vencidos(antes_de: float) -> list[Pipeline]:
    return [_row_to_pipeline(f) for f in await _filas(SQL_COLA_VENCIDOS, (antes_de,))]


async def arbol_consumo_leer(root_pipeline_id: str) -> dict | None:
    filas = await _filas("SELECT * FROM jacobs_arbol_consumo WHERE root_pipeline_id=%s", (root_pipeline_id,))
    return filas[0] if filas else None


async def arbol_consumo_sumar_en(
    cur, root_pipeline_id: str, tokens: int, costo_usd: float | None, ahora: float,
) -> None:
    """Corre en el CURSOR del llamador, dentro de SU transacción (los escritores
    de uso): o quedan la fila de axioma_usage y la suma del árbol, o ninguna."""
    sin_precio = tokens if costo_usd is None else 0
    await cur.execute(SQL_ARBOL_CONSUMO_SUMAR, (root_pipeline_id, tokens, costo_usd or 0, sin_precio, ahora))


async def arbol_sumar_pipelines(root_pipeline_id: str, n: int) -> None:
    await _ejecutar(
        "INSERT INTO jacobs_arbol_consumo (root_pipeline_id, pipelines_total, actualizado_at) VALUES (%s, %s, %s) "
        "ON DUPLICATE KEY UPDATE pipelines_total = pipelines_total + %s, actualizado_at = VALUE(actualizado_at)",
        (root_pipeline_id, 1 + n, time.time(), n),
    )


async def arbol_marcar_consumo_incompleto(root_pipeline_id: str) -> None:
    await _ejecutar(
        "INSERT INTO jacobs_arbol_consumo (root_pipeline_id, consumo_incompleto, actualizado_at) VALUES (%s, 1, %s) "
        "ON DUPLICATE KEY UPDATE consumo_incompleto = 1, actualizado_at = VALUE(actualizado_at)",
        (root_pipeline_id, time.time()),
    )


async def arbol_limpiar_consumo_incompleto(root_pipeline_id: str) -> None:
    await _ejecutar(
        "UPDATE jacobs_arbol_consumo SET consumo_incompleto=0, actualizado_at=%s WHERE root_pipeline_id=%s",
        (time.time(), root_pipeline_id),
    )


_COLUMNAS_TECHO = ("techo_tokens_aprobado", "techo_usd_aprobado", "techo_pipelines_aprobado",
                   "techo_hijos_por_step_aprobado")


async def arbol_fijar_techos(root_pipeline_id: str, techos: dict) -> None:
    columnas = [c for c in _COLUMNAS_TECHO if techos.get(c) is not None]
    if not columnas:
        return
    await _ejecutar(
        f"INSERT INTO jacobs_arbol_consumo (root_pipeline_id, {', '.join(columnas)}, actualizado_at) "
        f"VALUES (%s, {', '.join(['%s'] * len(columnas))}, %s) ON DUPLICATE KEY UPDATE "
        + ", ".join(f"{c} = VALUE({c})" for c in columnas) + ", actualizado_at = VALUE(actualizado_at)",
        (root_pipeline_id, *(techos[c] for c in columnas), time.time()),
    )


def _aprobacion_de_fila(fila: dict) -> dict:
    for campo in ("techos", "plan", "techos_aprobados"):
        if isinstance(fila.get(campo), str):
            fila[campo] = json.loads(fila[campo])
    return fila


async def aprobacion_crear(
    root_pipeline_id: str, pipeline_id: str, step_id: str, tipo: str, techos: list[dict], plan: dict | None,
) -> int:
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO jacobs_delegacion_aprobaciones "
                "(root_pipeline_id, pipeline_id, step_id, tipo, techos, plan, estado, creada_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,'pendiente',%s)",
                (root_pipeline_id, pipeline_id, step_id, tipo, json.dumps(techos, ensure_ascii=False),
                 json.dumps(plan, ensure_ascii=False) if plan is not None else None, time.time()),
            )
            return int(cur.lastrowid)
    finally:
        conn.close()


async def aprobacion_actualizar_plan(aprobacion_id: int, techos: list[dict], plan: dict) -> None:
    await _ejecutar(
        "UPDATE jacobs_delegacion_aprobaciones SET techos=%s, plan=%s WHERE id=%s AND estado='pendiente'",
        (json.dumps(techos, ensure_ascii=False), json.dumps(plan, ensure_ascii=False), aprobacion_id),
    )


async def aprobacion_pendiente_de(pipeline_id: str, step_id: str, tipo: str) -> dict | None:
    filas = await _filas(SQL_APROBACION_PENDIENTE_DE, (pipeline_id, step_id, tipo))
    return _aprobacion_de_fila(filas[0]) if filas else None


async def aprobacion_pendiente_de_raiz(root_pipeline_id: str, tipo: str) -> dict | None:
    filas = await _filas(
        "SELECT * FROM jacobs_delegacion_aprobaciones WHERE root_pipeline_id=%s AND estado='pendiente' AND tipo=%s",
        (root_pipeline_id, tipo),
    )
    return _aprobacion_de_fila(filas[0]) if filas else None


async def aprobaciones_pendientes(limite: int) -> list[dict]:
    return [_aprobacion_de_fila(f) for f in await _filas(SQL_APROBACIONES_PENDIENTES, (limite,))]


async def aprobacion_obtener(aprobacion_id: int) -> dict | None:
    filas = await _filas("SELECT * FROM jacobs_delegacion_aprobaciones WHERE id=%s", (aprobacion_id,))
    return _aprobacion_de_fila(filas[0]) if filas else None


async def aprobacion_resolver(
    aprobacion_id: int, estado: str, por: str, techos_aprobados: dict | None, ahora: float,
) -> bool:
    n = await _ejecutar(
        "UPDATE jacobs_delegacion_aprobaciones SET estado=%s, resuelta_por=%s, resuelta_at=%s, techos_aprobados=%s "
        "WHERE id=%s AND estado='pendiente'",
        (estado, por, ahora, json.dumps(techos_aprobados) if techos_aprobados is not None else None, aprobacion_id),
    )
    return n == 1
```

- [ ] **Step 6: Verde**

```bash
cd $WT && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_delegacion_modelos_puro.py 2>&1 | tail -1
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_delegacion_io_test.py jacobs/_subpipeline_contrato_io_test.py jacobs/_store_indexes_test.py" 2>&1 | tail -3
```
Expected: `4 passed`, y en la segunda todo `passed`: los 18 nuevos (4 de esquema, 9 de consultas y 5 de EXPLAIN) más los de F y los de índices. Si un EXPLAIN no usa el índice con 2000 filas, **no se fuerza con `FORCE INDEX`**: se pega el plan en `$L` y se diagnostica antes de seguir (LAS CUATRO, política 1).

- [ ] **Step 7: Mutación del EXPLAIN y commit**

En `store.SQL_COLA_CANDIDATOS`, cambiar `ORDER BY queued_at` por `ORDER BY created_at`. Correr `-k test_candidatos_de_la_cola`. Expected: `1 failed` por `Using filesort`. Restaurar la línea y pegar las dos salidas en `$L`.

```bash
git -C $WT add jacobs/models.py jacobs/store.py jacobs/_arnes_delegacion.py jacobs/_delegacion_io_test.py tests/test_delegacion_modelos_puro.py $L
git -C $WT commit -m "feat(jacobs): estados, esquema y consultas del árbol de delegación de Ada

queued/awaiting_approval/waiting_children; root_pipeline_id y queued_at;
jacobs_arbol_consumo y jacobs_delegacion_aprobaciones; transiciones
condicionales; cola FIFO con índice (status, queued_at). 4 puros + 18 de DB
vistos en rojo; EXPLAIN con 2040 pipelines y 600 aprobaciones, validado por mutación.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---
### Task 5: El uso suma en el árbol en la misma transacción

**Files:**
- Modify: `jacobs/usage_writer.py` (`record_direct_usage`)
- Modify: `las_manos/motor_registry/usage_writer.py` (`record_motor_usage`)
- Modify: `las_manos/motor_registry/models.py` (`MotorDispatchRequest`)
- Modify: `las_manos/motor_registry/routes.py` (`dispatch`)
- Modify: `las_manos/motor_registry/worker.py` (`run`, `_report_usage`)
- Modify: `jacobs/executor.py` (`_dispatch_step`: rama de transportes HTTP; `_invoke_motor`: payload)
- Modify: `jacobs/_delegacion_io_test.py` (clase `UsoArbolTest`)
- Create: `tests/test_delegacion_uso_puro.py`

**Interfaces:**
- Consumes: `store.arbol_consumo_sumar_en(cur, root, tokens, costo_usd, ahora)`, `store.arbol_marcar_consumo_incompleto(root)` (Task 4).
- Produces:
  - `jacobs.usage_writer.record_direct_usage(user_id, tenant_id, facet, provider_id, model, tokens_in, tokens_out, *, root_pipeline_id: str | None = None) -> bool` (True = fila y suma del árbol confirmadas en la base; False = respaldo o pérdida)
  - `motor_registry.usage_writer.record_motor_usage(user_id, tenant_id, facet, provider_id, model, tokens_in, tokens_out, *, job_id=None, status="unknown", root_pipeline_id: str | None = None) -> bool`
  - `MotorDispatchRequest.root_pipeline_id: str | None = None`; `worker.run(..., root_pipeline_id: str | None = None)`
  - El payload de `executor._invoke_motor` lleva `"root_pipeline_id": pipeline.root_pipeline_id`.

- [ ] **Step 1: Tests de base que fallan**

Agregar en `jacobs/_delegacion_io_test.py`, antes de `if __name__ == "__main__":`:

```python
class UsoArbolTest(_ConBaseG):
    """El consumo del árbol se suma EN LA MISMA transacción que axioma_usage
    (spec §4): o quedan las dos cosas o ninguna."""

    MODELO = "test-g-model"

    async def asyncTearDown(self):
        await ada.ejecutar("DELETE FROM axioma_usage WHERE model=%s", (self.MODELO,))
        await super().asyncTearDown()

    def raiz(self) -> str:
        r = "raiz-uso-" + uuid.uuid4().hex[:8]
        self.raices.append(r)
        return r

    async def filas_de_uso(self) -> int:
        return int((await ada.una_fila("SELECT COUNT(*) AS n FROM axioma_usage WHERE model=%s", (self.MODELO,)))["n"])

    async def test_directo_suma_tokens_y_costo_en_el_arbol(self):
        from jacobs import usage_writer
        raiz = self.raiz()
        with patch.object(usage_writer, "_lookup_model_price", AsyncMock(return_value=(1.0, 2.0))):
            ok = await usage_writer.record_direct_usage("1", "1", "jekyll", "prov", self.MODELO, 1000, 500,
                                                        root_pipeline_id=raiz)
        self.assertTrue(ok)
        fila = await store.arbol_consumo_leer(raiz)
        self.assertEqual((fila["tokens_total"], fila["tokens_sin_precio"]), (1500, 0))
        self.assertAlmostEqual(float(fila["costo_usd_total"]), 0.002)
        self.assertEqual(await self.filas_de_uso(), 1)

    async def test_directo_sin_precio_va_a_tokens_sin_precio(self):
        from jacobs import usage_writer
        raiz = self.raiz()
        with patch.object(usage_writer, "_lookup_model_price", AsyncMock(return_value=(None, None))):
            await usage_writer.record_direct_usage("1", "1", "jekyll", "prov", self.MODELO, 10, 5,
                                                   root_pipeline_id=raiz)
        fila = await store.arbol_consumo_leer(raiz)
        self.assertEqual((fila["tokens_total"], fila["tokens_sin_precio"]), (15, 15))

    async def test_directo_es_atomico(self):
        from jacobs import usage_writer
        raiz = self.raiz()
        with patch.object(usage_writer, "_lookup_model_price", AsyncMock(return_value=(1.0, 1.0))), \
             patch("jacobs.store.arbol_consumo_sumar_en", AsyncMock(side_effect=RuntimeError("falla simulada"))):
            ok = await usage_writer.record_direct_usage("1", "1", "jekyll", "prov", self.MODELO, 10, 5,
                                                        root_pipeline_id=raiz)
        self.assertFalse(ok)
        self.assertEqual(await self.filas_de_uso(), 0, "sin la suma del árbol no puede quedar la fila de uso")
        self.assertIsNone(await store.arbol_consumo_leer(raiz))

    async def test_motor_suma_en_el_arbol_y_es_atomico(self):
        from motor_registry import usage_writer as muw
        raiz = self.raiz()
        with patch.object(muw, "_lookup_model_price", AsyncMock(return_value=(1.0, 1.0))):
            self.assertTrue(await muw.record_motor_usage("1", "1", "kimi", "prov", self.MODELO, 100, 100,
                                                         job_id="job-g", status="completed", root_pipeline_id=raiz))
        self.assertEqual((await store.arbol_consumo_leer(raiz))["tokens_total"], 200)
        with patch.object(muw, "_lookup_model_price", AsyncMock(return_value=(1.0, 1.0))), \
             patch.object(muw, "_WRITE_RETRY_DELAY_SECONDS", 0), \
             patch("jacobs.store.arbol_consumo_sumar_en", AsyncMock(side_effect=RuntimeError("falla simulada"))):
            self.assertFalse(await muw.record_motor_usage("1", "1", "kimi", "prov", self.MODELO, 7, 7,
                                                          job_id="job-g2", status="failed", root_pipeline_id=raiz))
        self.assertEqual(await self.filas_de_uso(), 1)
        self.assertEqual((await store.arbol_consumo_leer(raiz))["tokens_total"], 200)

    async def test_sin_raiz_no_toca_el_arbol(self):
        from jacobs import usage_writer
        with patch.object(usage_writer, "_lookup_model_price", AsyncMock(return_value=(1.0, 1.0))):
            self.assertTrue(await usage_writer.record_direct_usage("1", "1", "jekyll", "prov", self.MODELO, 1, 1))
        self.assertEqual(await self.filas_de_uso(), 1)
```

```bash
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_delegacion_io_test.py -k UsoArbol" 2>&1 | tee -a $L | tail -6
```
Expected: `5 failed`. Cuatro caen por `TypeError: ... unexpected keyword argument 'root_pipeline_id'` y `test_sin_raiz…` cae por `assert None is True`, porque la función vieja devuelve `None`.

- [ ] **Step 2: Tests puros del cableado que fallan**

Create `tests/test_delegacion_uso_puro.py`:

```python
"""Frente G (2026-09-16): la raíz del árbol llega a los dos escritores de uso,
y si la fila no quedó en la base el árbol se marca consumo_incompleto
(fail-closed: la próxima delegación de ese árbol pide aprobación).

ROJO CONTRA LA BASE: los escritores no reciben root_pipeline_id y nadie marca."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

from jacobs import executor  # noqa: E402
from jacobs.models import Pipeline, Step  # noqa: E402


def _resolved():
    from facet_resolver import ResolvedFacet
    return ResolvedFacet(key="jekyll", provider_id="deepseek", base_url="https://api.example.test/v1",
                         model="m", credential="k", transport="http_openai_compat", persona=None, params=None)


class DispatchDirectoTest(unittest.IsolatedAsyncioTestCase):
    async def _correr(self, resultado_uso: bool):
        pipeline = Pipeline(pipeline_id="p1", name="n", invoked_by="plataforma", mode="autonomous",
                            user_id="1", tenant_id="1", root_pipeline_id="raiz-1")
        step = Step(pipeline_id="p1", facet="jekyll", capability="analysis", input={"prompt": "x"})
        uso = AsyncMock(return_value=resultado_uso)
        marcar = AsyncMock()
        with patch.object(executor, "validate_capability", AsyncMock(return_value=None)), \
             patch.object(executor, "resolve_facet", AsyncMock(return_value=_resolved())), \
             patch.object(executor, "_invoke_http_openai_compat",
                          AsyncMock(return_value={"success": True, "result": "r", "tokens_in": 3, "tokens_out": 4})), \
             patch.object(executor, "record_direct_usage", uso), \
             patch("jacobs.store.arbol_marcar_consumo_incompleto", marcar):
            await executor._dispatch_step(step, pipeline)
        return uso, marcar

    async def test_pasa_la_raiz_al_escritor(self):
        uso, marcar = await self._correr(True)
        self.assertEqual(uso.await_args.kwargs["root_pipeline_id"], "raiz-1")
        marcar.assert_not_awaited()

    async def test_si_la_fila_no_quedo_marca_consumo_incompleto(self):
        _uso, marcar = await self._correr(False)
        marcar.assert_awaited_once_with("raiz-1")


class DispatchMotorTest(unittest.IsolatedAsyncioTestCase):
    async def test_invoke_motor_manda_la_raiz(self):
        pipeline = Pipeline(pipeline_id="p1", name="n", invoked_by="plataforma", mode="autonomous",
                            root_pipeline_id="raiz-1")
        step = Step(pipeline_id="p1", facet="kimi", capability="generate", input={"prompt": "x"})
        respuesta = MagicMock()
        respuesta.raise_for_status = lambda: None
        respuesta.json = lambda: {"status": "rejected", "rejected_reason": "probe"}
        cliente = MagicMock()
        cliente.post = AsyncMock(return_value=respuesta)
        with patch.object(executor, "obtener_cliente_http", return_value=cliente):
            with self.assertRaises(RuntimeError):
                await executor._invoke_motor(step, pipeline, 30, "prompt")
        self.assertEqual(cliente.post.await_args.kwargs["json"]["root_pipeline_id"], "raiz-1")

    def test_el_request_del_motor_acepta_la_raiz(self):
        from motor_registry.models import MotorDispatchRequest
        req = MotorDispatchRequest(caller="jacobs", capability="generate", prompt="x", root_pipeline_id="raiz-1")
        self.assertEqual(req.root_pipeline_id, "raiz-1")


class WorkerUsoTest(unittest.IsolatedAsyncioTestCase):
    async def test_worker_pasa_la_raiz_a_record_motor_usage(self):
        from motor_registry import worker
        from motor_registry.catalog import MotorCatalog
        from motor_registry.job_store import JobStore
        cfg = {"motors": {"kimi": {"enabled": True, "provider": "moonshot", "provider_id": "moonshot",
                                   "api_url": "https://x.test/v1", "model": "k2", "sandbox_only": True,
                                   "default_timeout_seconds": 30, "transport": "http_openai_compat"}},
               "capabilities": {"generate": {"allowed_callers": ["jacobs"], "requires_human_gate": False,
                                             "max_execution_minutes": 15, "output_schema": ""}}}
        respuesta = {"choices": [{"message": {"content": "hecho"}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 5, "completion_tokens": 6}}
        uso = AsyncMock(return_value=True)
        store = JobStore(tempfile.mkdtemp(prefix="jax-g-jobs-") + "/jobs.jsonl")
        job_id = store.create(caller="jacobs", capability="generate", motor="kimi", trace_id="t", prompt="x",
                              recursion_depth=0)
        with patch.dict(worker._TRANSPORT_DISPATCH, {"http_openai_compat": AsyncMock(return_value=respuesta)}), \
             patch.object(worker, "_limite_del_motor", AsyncMock(return_value=({"max_tokens": 100}, "catalogo"))), \
             patch.object(worker, "resolve_credential_instrumented", AsyncMock(return_value="sk")), \
             patch("motor_registry.usage_writer.record_motor_usage", uso):
            await worker.run(job_id=job_id, motor="kimi", capability="generate", prompt="x", context={},
                             store=store, catalog=MotorCatalog(cfg), kill_switch_path="/tmp/jax-g-sin-freno/PAUSE",
                             caller="jacobs", timeout_seconds=30, root_pipeline_id="raiz-1")
        self.assertEqual(uso.await_args.kwargs["root_pipeline_id"], "raiz-1")
```

```bash
cd $WT && pwd && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_delegacion_uso_puro.py 2>&1 | tee -a $L | tail -6
```
Expected: `5 failed`: `KeyError: 'root_pipeline_id'` en los dos del dispatch directo y en el de `_invoke_motor`, `ValidationError extra_forbidden` y `TypeError` en `worker.run`.

- [ ] **Step 3: Escritor directo**

En `jacobs/usage_writer.py`, agregar `import time` a los imports. Reemplazar la firma y el bloque `try:` de `record_direct_usage` hasta el `return` de éxito por:

```python
async def record_direct_usage(
    user_id: str | None,
    tenant_id: str | None,
    facet: str,
    provider_id: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    *,
    root_pipeline_id: str | None = None,
) -> bool:
```

En el docstring, agregar al final:

```
    Frente G (2026-09-16): con root_pipeline_id, la fila de axioma_usage y la
    suma en jacobs_arbol_consumo van en UNA transacción: el techo del árbol se
    compara contra un número vivo, nunca contra uno que perdió un turno. Devuelve
    True solo si las dos quedaron confirmadas; False si la fila fue al respaldo
    o se perdió (el executor marca entonces el árbol consumo_incompleto).
```

Y el cuerpo desde `creado_en = _ahora_iso()` hasta el primer `return` queda así:

```python
    creado_en = _ahora_iso()
    cost = None
    try:
        conn = await aiomysql.connect(**_db_cfg(), connect_timeout=db_connect_timeout_seconds())
        try:
            price_in, price_out = await _lookup_model_price(conn, provider_id, model)
            if price_in is not None and price_out is not None:
                cost = (tokens_in * float(price_in) + tokens_out * float(price_out)) / 1_000_000
            # Sin commit no queda nada: si algo falla entre begin() y commit(),
            # conn.close() del finally descarta la transacción entera.
            await conn.begin()
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO axioma_usage (tenant_id, user_id, facet, model, tokens_in, tokens_out, cost_usd, request_type) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, 'pipeline')",
                    (
                        int(tenant_id) if tenant_id is not None else None,
                        int(user_id) if user_id is not None else None,
                        facet, model, tokens_in, tokens_out, cost,
                    ),
                )
                if root_pipeline_id:
                    from jacobs.store import arbol_consumo_sumar_en
                    await arbol_consumo_sumar_en(cur, root_pipeline_id, tokens_in + tokens_out, cost, time.time())
            await conn.commit()
        finally:
            conn.close()
        return True
    except Exception as e:  # fail-soft: la contabilidad no puede tumbar un step ya completado; la fila va al respaldo, no a la basura, y el False le dice al executor que marque el árbol
        motivo = f"{type(e).__name__}: {e}"
```

Los dos `return` del final (respaldo y pérdida) pasan a `return False`.

- [ ] **Step 4: Escritor del motor**

En `las_manos/motor_registry/usage_writer.py`, agregar `import time`. La firma suma `root_pipeline_id: str | None = None` después de `status` y devuelve `-> bool`. Dentro del `for attempt`, el bloque `try:` queda:

```python
        try:
            conn = await aiomysql.connect(**_db_cfg(), connect_timeout=db_connect_timeout_seconds())
            try:
                price_in, price_out = await _lookup_model_price(conn, provider_id, model)
                if price_in is not None and price_out is not None:
                    cost = (tokens_in * float(price_in) + tokens_out * float(price_out)) / 1_000_000
                # Frente G: fila de uso + suma del árbol en UNA transacción; sin
                # commit, conn.close() descarta las dos.
                await conn.begin()
                async with conn.cursor() as cur:
                    await cur.execute(
                        "INSERT INTO axioma_usage "
                        "(tenant_id, user_id, facet, model, tokens_in, tokens_out, cost_usd, request_type, status, job_id) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, 'motor', %s, %s)",
                        (
                            int(tenant_id) if tenant_id is not None else None,
                            int(user_id) if user_id is not None else None,
                            facet, model, tokens_in, tokens_out, cost, status, job_id,
                        ),
                    )
                    if root_pipeline_id:
                        from jacobs.store import arbol_consumo_sumar_en
                        await arbol_consumo_sumar_en(cur, root_pipeline_id, tokens_in + tokens_out, cost, time.time())
                await conn.commit()
            finally:
                conn.close()
            return True
```

Los `return` del respaldo y de la pérdida pasan a `return False`.

- [ ] **Step 5: Dispatch, worker y executor**

`las_manos/motor_registry/models.py`, en `MotorDispatchRequest`, antes de `model_config`:

```python
    # Frente G (2026-09-16): raíz del árbol de delegación del pipeline de Jacobs
    # que pidió el job. Solo alimenta jacobs_arbol_consumo (el techo del árbol);
    # no da autoridad. Mismo límite de confianza que user_id/tenant_id (arriba).
    root_pipeline_id: str | None = None
```

`las_manos/motor_registry/routes.py::dispatch`, en la llamada a `motor_worker.run(...)`, agregar `root_pipeline_id=req.root_pipeline_id,`.

`las_manos/motor_registry/worker.py::run`: agregar el parámetro `root_pipeline_id: str | None = None` después de `timeout_seconds`. En `_report_usage`, reemplazar la llamada por:

```python
            from motor_registry.usage_writer import record_motor_usage
            registrado = await record_motor_usage(
                user_id, tenant_id, motor, provider_id, motor_entry.model,
                cumulative_prompt_tokens, cumulative_completion_tokens,
                job_id=job_id, status=status, root_pipeline_id=root_pipeline_id,
            )
            if not registrado and root_pipeline_id:
                try:
                    from jacobs.store import arbol_marcar_consumo_incompleto
                    await arbol_marcar_consumo_incompleto(root_pipeline_id)
                except Exception:  # fail-soft: la base no respondió ni para marcar; queda el ERROR con la raíz, y el step de Jacobs que espera este job falla igual al no poder escribir su propio estado
                    logger.error("job %s: no se pudo marcar consumo_incompleto del árbol %s", job_id, root_pipeline_id, exc_info=True)
```

`jacobs/executor.py::_dispatch_step`, en la rama de transportes HTTP, reemplazar la llamada a `record_direct_usage(...)` por:

```python
        registrado = await record_direct_usage(
            pipeline.user_id, pipeline.tenant_id, step.facet,
            f.provider_id, f.model,
            result.get("tokens_in", 0), result.get("tokens_out", 0),
            root_pipeline_id=pipeline.root_pipeline_id,
        )
        # Frente G: si la fila fue al respaldo, el consumo del árbol quedó
        # corto; la próxima delegación de este árbol pide aprobación. Si ni
        # esto responde, sube: el step falla limpio (la base está caída).
        if not registrado and pipeline.root_pipeline_id:
            await store.arbol_marcar_consumo_incompleto(pipeline.root_pipeline_id)
```

`jacobs/executor.py::_invoke_motor`, en `payload`, después de `"tenant_id"`:

```python
        # Frente G: el worker suma el uso del job en el árbol de delegación.
        "root_pipeline_id": pipeline.root_pipeline_id,
```

- [ ] **Step 6: Verde y regresión**

```bash
cd $WT && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_delegacion_uso_puro.py tests/test_cola_uso_escritores.py tests/test_invoke_motor_rechazo.py tests/test_motor_contexto_y_salida.py las_manos/_worker_tool_loop_test.py las_manos/_worker_max_tokens_test.py 2>&1 | tail -2
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_delegacion_io_test.py jacobs/_direct_usage_test.py" 2>&1 | tail -2
cd $WT && $PY -m pytest -q policy/tests/test_no_fail_open_except.py tests/test_no_blocking_in_async.py 2>&1 | tail -2
```
Expected: todo `passed`. Si `tests/test_cola_uso_escritores.py` afirma `is None` sobre el resultado de un escritor, se actualiza esa aserción a `is False` en el mismo commit, con una nota en `$L`.

- [ ] **Step 7: Mutación y commit**

En `jacobs/usage_writer.py`, mover `await conn.commit()` a justo después del `INSERT INTO axioma_usage` (antes de la suma del árbol). Correr `-k test_directo_es_atomico`. Expected: `1 failed` (`filas_de_uso` = 1). Restaurar y pegar las dos salidas en `$L`.

```bash
git -C $WT add jacobs/usage_writer.py las_manos/motor_registry/usage_writer.py las_manos/motor_registry/models.py las_manos/motor_registry/routes.py las_manos/motor_registry/worker.py jacobs/executor.py jacobs/_delegacion_io_test.py tests/test_delegacion_uso_puro.py $L
git -C $WT commit -m "feat(uso): el consumo del árbol de delegación se suma en la misma transacción que axioma_usage

Los dos escritores devuelven bool; sin la fila en la base el árbol queda
consumo_incompleto (fail-closed). La raíz viaja en MotorDispatchRequest. 5 de
DB + 5 puros vistos en rojo; atomicidad validada por mutación.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Proyección y techos (puro)

**Files:**
- Create: `jacobs/delegacion_techos.py`
- Create: `tests/test_delegacion_techos_puro.py`

**Interfaces:**
- Consumes: `ConfigDelegacion` (Task 2).
- Produces:
  - `TECHO_PROFUNDIDAD = "profundidad"`, `TECHO_HIJOS_POR_STEP = "hijos_por_step"`, `TECHO_PIPELINES_ARBOL = "pipelines_arbol"`, `TECHO_TOKENS_ARBOL = "tokens_arbol"`, `TECHO_USD_ARBOL = "usd_arbol"`, `TECHO_CONSUMO_INCOMPLETO = "consumo_incompleto"`
  - `CAMPO_APROBADO: dict[str, str]` (techo → columna `techo_*_aprobado`)
  - `Consumo` (frozen; `Consumo.de_fila(fila: dict | None) -> Consumo`), `Proyeccion(profundidad_hijo, hijos_step, pipelines_arbol, tokens, usd)`, `TechoExcedido(techo, proyectado, limite, aprobable)` con `.a_dict() -> dict`
  - `limites(consumo: Consumo, cfg: ConfigDelegacion) -> dict[str, float]`
  - `proyectar(*, depth_padre: int, hijos_existentes_step: int, hijos_nuevos: int, pipelines_arbol_actual: int, consumo: Consumo, cfg: ConfigDelegacion) -> Proyeccion`
  - `evaluar_techos(p: Proyeccion, consumo: Consumo, cfg: ConfigDelegacion, max_profundidad: int) -> list[TechoExcedido]`
  - `techos_cruzados(consumo: Consumo, cfg: ConfigDelegacion) -> list[TechoExcedido]`
  - `techos_no_cubiertos(excedidos: list[dict], techos: dict, aceptar_consumo_incompleto: bool) -> list[dict]`
  - `ResumenHijo(subobjetivo_id, pipeline_id, objetivo, estado, resumen, output_ref)`, `acotar(texto: str, largo: int) -> str`, `armar_prompt_integracion(objetivo: str, hijos: list[ResumenHijo]) -> str`

- [ ] **Step 1: Tests que fallan**

Create `tests/test_delegacion_techos_puro.py`:

```python
"""Frente G (2026-09-16): proyección de techos del árbol (spec §3.1.3 y §5).
Puro: la decisión de 'dentro o fuera' no toca la base, así que se prueba entera
acá; delegacion.py solo junta los números y actúa.

ROJO CONTRA LA BASE: ModuleNotFoundError jacobs.delegacion_techos."""
from __future__ import annotations

import os

os.environ["JAX_DB_NAME"] = "jax_memory_test"

from decimal import Decimal  # noqa: E402

from jacobs import delegacion_techos as dt  # noqa: E402
from jacobs.config_delegacion import ConfigDelegacion  # noqa: E402

CFG = ConfigDelegacion(max_hijos_por_step=10, max_pipelines_arbol=40, max_tokens_arbol=2_000_000,
                       max_usd_arbol=10.0, reserva_tokens_hijo=50_000, cola_max_espera_segundos=3600,
                       cola_cadencia_segundos=10, resumen_hijo_caracteres=2000)


def _proy(**kw):
    base = dict(depth_padre=0, hijos_existentes_step=0, hijos_nuevos=3, pipelines_arbol_actual=1,
                consumo=dt.Consumo(), cfg=CFG)
    base.update(kw)
    return dt.proyectar(**base)


def test_proyeccion_del_spec():
    p = _proy(depth_padre=1, hijos_existentes_step=2, hijos_nuevos=3, pipelines_arbol_actual=5,
              consumo=dt.Consumo(tokens_total=100_000, costo_usd_total=1.5))
    assert (p.profundidad_hijo, p.hijos_step, p.pipelines_arbol, p.tokens, p.usd) == (2, 5, 8, 250_000, 1.5)


def test_dentro_de_techos_no_excede_nada():
    assert dt.evaluar_techos(_proy(), dt.Consumo(), CFG, max_profundidad=3) == []


def test_profundidad_excedida_no_es_aprobable():
    [e] = dt.evaluar_techos(_proy(depth_padre=3), dt.Consumo(), CFG, max_profundidad=3)
    assert (e.techo, e.proyectado, e.limite, e.aprobable) == ("profundidad", 4, 3, False)


def test_cada_techo_aprobable_se_detecta():
    consumo = dt.Consumo(tokens_total=1_990_000, costo_usd_total=10.01)
    p = _proy(hijos_existentes_step=9, hijos_nuevos=2, pipelines_arbol_actual=39, consumo=consumo)
    techos = {e.techo: e for e in dt.evaluar_techos(p, consumo, CFG, max_profundidad=3)}
    assert set(techos) == {"hijos_por_step", "pipelines_arbol", "tokens_arbol", "usd_arbol"}
    assert all(e.aprobable for e in techos.values())
    assert techos["tokens_arbol"].proyectado == 2_090_000


def test_igual_al_limite_no_excede():
    p = _proy(hijos_existentes_step=7, hijos_nuevos=3, pipelines_arbol_actual=37)
    assert dt.evaluar_techos(p, dt.Consumo(), CFG, max_profundidad=3) == []


def test_consumo_incompleto_pide_aprobacion():
    consumo = dt.Consumo(consumo_incompleto=True)
    [e] = dt.evaluar_techos(_proy(consumo=consumo), consumo, CFG, max_profundidad=3)
    assert (e.techo, e.aprobable) == ("consumo_incompleto", True)


def test_techo_aprobado_reemplaza_al_del_entorno_solo_para_ese_arbol():
    consumo = dt.Consumo(techo_hijos_por_step_aprobado=20, techo_tokens_aprobado=5_000_000)
    p = _proy(hijos_nuevos=15, consumo=consumo)
    assert dt.evaluar_techos(p, consumo, CFG, max_profundidad=3) == []
    assert dt.evaluar_techos(p, dt.Consumo(), CFG, max_profundidad=3) != []


def test_de_fila_convierte_decimal_y_nulos():
    c = dt.Consumo.de_fila({"pipelines_total": 3, "tokens_total": 10, "costo_usd_total": Decimal("0.500000"),
                            "tokens_sin_precio": 2, "consumo_incompleto": 1, "techo_tokens_aprobado": None,
                            "techo_usd_aprobado": Decimal("12.5"), "techo_pipelines_aprobado": None,
                            "techo_hijos_por_step_aprobado": 11})
    assert (c.costo_usd_total, c.consumo_incompleto, c.techo_usd_aprobado, c.techo_hijos_por_step_aprobado) == (
        0.5, True, 12.5, 11)
    assert dt.Consumo.de_fila(None) == dt.Consumo()


def test_cruce_a_mitad_de_camino_sin_reserva():
    assert dt.techos_cruzados(dt.Consumo(tokens_total=2_000_000), CFG) == []
    cruzados = dt.techos_cruzados(dt.Consumo(tokens_total=2_000_001, costo_usd_total=10.5), CFG)
    assert {e.techo for e in cruzados} == {"tokens_arbol", "usd_arbol"}


def test_techos_no_cubiertos_por_la_aprobacion():
    excedidos = [
        {"techo": "tokens_arbol", "proyectado": 2_100_000, "limite": 2_000_000, "aprobable": True},
        {"techo": "hijos_por_step", "proyectado": 12, "limite": 10, "aprobable": True},
        {"techo": "consumo_incompleto", "proyectado": 1, "limite": 0, "aprobable": True},
    ]
    faltan = dt.techos_no_cubiertos(excedidos, {"techo_tokens_aprobado": 2_100_000,
                                                "techo_hijos_por_step_aprobado": 11}, False)
    assert [f["techo"] for f in faltan] == ["hijos_por_step", "consumo_incompleto"]
    assert dt.techos_no_cubiertos(excedidos, {"techo_tokens_aprobado": 3_000_000,
                                              "techo_hijos_por_step_aprobado": 12}, True) == []


def test_prompt_de_integracion_declara_cada_hijo_y_los_fallos():
    hijos = [
        dt.ResumenHijo("s0", "p0", "investigar A", "completed", "A es verdad", "inline:{}"),
        dt.ResumenHijo("s1", "p1", "investigar B", "aborted", "[sin resultado: timeout]", None),
    ]
    texto = dt.armar_prompt_integracion("integrá A y B", hijos)
    assert texto.startswith("integrá A y B")
    for fragmento in ("s0", "p0", "completed", "A es verdad", "inline:{}", "s1", "aborted", "sin output_ref"):
        assert fragmento in texto
    assert "fallo declarado" in texto


def test_acotar():
    assert dt.acotar("abc", 5) == "abc"
    corto = dt.acotar("x" * 300, 200)
    assert corto.startswith("x" * 200) and "recortado a 200 caracteres de 300" in corto
```

```bash
cd $WT && pwd && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_delegacion_techos_puro.py 2>&1 | tee -a $L | tail -3
```
Expected: `ModuleNotFoundError: No module named 'jacobs.delegacion_techos'`.

- [ ] **Step 2: Implementar**

Create `jacobs/delegacion_techos.py`:

```python
"""
Jacobs — Techos del árbol de delegación de Ada: la parte PURA (frente G, 2026-09-16).

Spec: docs/superpowers/specs/2026-09-16-emisor-subpipelines-ada-design.md §3.1.3, §5.
delegacion.py junta los números (base) y actúa; acá se decide. Un techo aprobado
por Fernando (columna techo_*_aprobado de ESE árbol) reemplaza al del entorno
solo para ese árbol. La profundidad no se aprueba: el contrato de F rechaza el
consumo de un token más hondo que JAX_MAX_SUBPIPELINE_DEPTH.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from jacobs.config_delegacion import ConfigDelegacion

TECHO_PROFUNDIDAD = "profundidad"
TECHO_HIJOS_POR_STEP = "hijos_por_step"
TECHO_PIPELINES_ARBOL = "pipelines_arbol"
TECHO_TOKENS_ARBOL = "tokens_arbol"
TECHO_USD_ARBOL = "usd_arbol"
TECHO_CONSUMO_INCOMPLETO = "consumo_incompleto"

CAMPO_APROBADO = {
    TECHO_TOKENS_ARBOL: "techo_tokens_aprobado",
    TECHO_USD_ARBOL: "techo_usd_aprobado",
    TECHO_PIPELINES_ARBOL: "techo_pipelines_aprobado",
    TECHO_HIJOS_POR_STEP: "techo_hijos_por_step_aprobado",
}


def _entero_o_none(valor) -> int | None:
    return None if valor is None else int(valor)


def _real_o_none(valor) -> float | None:
    return None if valor is None else float(valor)


@dataclass(frozen=True)
class Consumo:
    pipelines_total: int = 0
    tokens_total: int = 0
    costo_usd_total: float = 0.0
    tokens_sin_precio: int = 0
    consumo_incompleto: bool = False
    techo_tokens_aprobado: int | None = None
    techo_usd_aprobado: float | None = None
    techo_pipelines_aprobado: int | None = None
    techo_hijos_por_step_aprobado: int | None = None

    @classmethod
    def de_fila(cls, fila: dict | None) -> "Consumo":
        if not fila:
            return cls()
        return cls(
            pipelines_total=int(fila.get("pipelines_total") or 0),
            tokens_total=int(fila.get("tokens_total") or 0),
            costo_usd_total=float(fila.get("costo_usd_total") or 0),
            tokens_sin_precio=int(fila.get("tokens_sin_precio") or 0),
            consumo_incompleto=bool(fila.get("consumo_incompleto")),
            techo_tokens_aprobado=_entero_o_none(fila.get("techo_tokens_aprobado")),
            techo_usd_aprobado=_real_o_none(fila.get("techo_usd_aprobado")),
            techo_pipelines_aprobado=_entero_o_none(fila.get("techo_pipelines_aprobado")),
            techo_hijos_por_step_aprobado=_entero_o_none(fila.get("techo_hijos_por_step_aprobado")),
        )


@dataclass(frozen=True)
class Proyeccion:
    profundidad_hijo: int
    hijos_step: int
    pipelines_arbol: int
    tokens: int
    usd: float


@dataclass(frozen=True)
class TechoExcedido:
    techo: str
    proyectado: float
    limite: float
    aprobable: bool

    def a_dict(self) -> dict:
        return asdict(self)


def limites(consumo: Consumo, cfg: ConfigDelegacion) -> dict[str, float]:
    return {
        TECHO_HIJOS_POR_STEP: consumo.techo_hijos_por_step_aprobado or cfg.max_hijos_por_step,
        TECHO_PIPELINES_ARBOL: consumo.techo_pipelines_aprobado or cfg.max_pipelines_arbol,
        TECHO_TOKENS_ARBOL: consumo.techo_tokens_aprobado or cfg.max_tokens_arbol,
        TECHO_USD_ARBOL: consumo.techo_usd_aprobado or cfg.max_usd_arbol,
    }


def proyectar(
    *, depth_padre: int, hijos_existentes_step: int, hijos_nuevos: int,
    pipelines_arbol_actual: int, consumo: Consumo, cfg: ConfigDelegacion,
) -> Proyeccion:
    """Spec §3.1.3: tokens = consumido + reserva x hijos nuevos; USD = consumido
    con precio (sin reserva: no hay precio antes de saber qué modelo corre)."""
    return Proyeccion(
        profundidad_hijo=depth_padre + 1,
        hijos_step=hijos_existentes_step + hijos_nuevos,
        pipelines_arbol=pipelines_arbol_actual + hijos_nuevos,
        tokens=consumo.tokens_total + cfg.reserva_tokens_hijo * hijos_nuevos,
        usd=consumo.costo_usd_total,
    )


def evaluar_techos(
    p: Proyeccion, consumo: Consumo, cfg: ConfigDelegacion, max_profundidad: int,
) -> list[TechoExcedido]:
    excedidos: list[TechoExcedido] = []
    if p.profundidad_hijo > max_profundidad:
        excedidos.append(TechoExcedido(TECHO_PROFUNDIDAD, p.profundidad_hijo, max_profundidad, False))
    lim = limites(consumo, cfg)
    for techo, valor in (
        (TECHO_HIJOS_POR_STEP, p.hijos_step), (TECHO_PIPELINES_ARBOL, p.pipelines_arbol),
        (TECHO_TOKENS_ARBOL, p.tokens), (TECHO_USD_ARBOL, p.usd),
    ):
        if valor > lim[techo]:
            excedidos.append(TechoExcedido(techo, valor, lim[techo], True))
    if consumo.consumo_incompleto:
        excedidos.append(TechoExcedido(TECHO_CONSUMO_INCOMPLETO, 1, 0, True))
    return excedidos


def techos_cruzados(consumo: Consumo, cfg: ConfigDelegacion) -> list[TechoExcedido]:
    """Spec §5: el consumo REAL (sin reserva) ya pasó el techo a mitad de camino."""
    lim = limites(consumo, cfg)
    cruzados: list[TechoExcedido] = []
    if consumo.tokens_total > lim[TECHO_TOKENS_ARBOL]:
        cruzados.append(TechoExcedido(TECHO_TOKENS_ARBOL, consumo.tokens_total, lim[TECHO_TOKENS_ARBOL], True))
    if consumo.costo_usd_total > lim[TECHO_USD_ARBOL]:
        cruzados.append(TechoExcedido(TECHO_USD_ARBOL, consumo.costo_usd_total, lim[TECHO_USD_ARBOL], True))
    return cruzados


def techos_no_cubiertos(excedidos: list[dict], techos: dict, aceptar_consumo_incompleto: bool) -> list[dict]:
    """Los excedidos que la aprobación de Fernando NO cubre: sin esto, aprobar
    con un techo menor que lo proyectado volvería a pedir aprobación en bucle."""
    faltan: list[dict] = []
    for e in excedidos:
        if not e["aprobable"]:
            faltan.append(e)
        elif e["techo"] == TECHO_CONSUMO_INCOMPLETO:
            if not aceptar_consumo_incompleto:
                faltan.append(e)
        else:
            valor = techos.get(CAMPO_APROBADO[e["techo"]])
            if valor is None or valor < e["proyectado"]:
                faltan.append(e)
    return faltan


@dataclass(frozen=True)
class ResumenHijo:
    subobjetivo_id: str
    pipeline_id: str
    objetivo: str
    estado: str
    resumen: str
    output_ref: str | None


def acotar(texto: str, largo: int) -> str:
    if len(texto) <= largo:
        return texto
    return texto[:largo] + f"\n[... recortado a {largo} caracteres de {len(texto)}]"


def armar_prompt_integracion(objetivo: str, hijos: list[ResumenHijo]) -> str:
    """Texto para el LLM (no es UI). Spec §3.1.5: por hijo, id, sub-objetivo,
    estado, resumen acotado y referencia a su output_ref."""
    partes = [objetivo, "", f"Resultados de los sub-pipelines delegados ({len(hijos)}):"]
    for h in hijos:
        partes.append(
            f"\n--- {h.subobjetivo_id} · pipeline {h.pipeline_id} · estado {h.estado} ---\n"
            f"Sub-objetivo: {h.objetivo}\n"
            f"Resumen: {h.resumen}\n"
            f"Referencia completa: {h.output_ref or 'sin output_ref'}"
        )
    partes.append(
        "\nIntegrá SOLO lo que dicen estos resultados. Un sub-pipeline que no está en estado "
        "'completed' es un fallo declarado: nombralo como tal, no lo rellenes."
    )
    return "\n".join(partes)
```

- [ ] **Step 3: Verde y commit**

```bash
cd $WT && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_delegacion_techos_puro.py 2>&1 | tail -1
git -C $WT add jacobs/delegacion_techos.py tests/test_delegacion_techos_puro.py $L
git -C $WT commit -m "feat(jacobs): proyección de techos del árbol de delegación (puro)

Profundidad no aprobable; hijos, pipelines, tokens (con reserva) y USD
aprobables por árbol; cruce a mitad de camino sin reserva; prompt de
integración con fallos declarados. 12 tests vistos en rojo.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```
Expected del primer comando: `12 passed`.

---
### Task 7: `delegacion.delegar` — dentro de techos lanza, fuera pide aprobación

**Files:**
- Modify: `jacobs/policy.py` (`validate_create`: límite de paralelos)
- Modify: `jacobs/routes.py` (`create_pipeline` → `_crear_pipeline` + `crear_hijo_de_ada`)
- Modify: `tests/test_subpipeline_contrato_rutas.py` (dos aserciones de F: `completed` → `queued`)
- Create: `jacobs/delegacion.py`
- Create: `tests/test_delegacion_politica_puro.py`
- Modify: `jacobs/_delegacion_io_test.py` (clase `DelegarTest`)

**Interfaces:**
- Consumes: `emitir_token_subpipeline`, `EmisionRechazada`, `config_subpipelines` (F); `config_delegacion` (Task 2); `parsear_plan_delegacion`, `orden_topologico`, `errores_de_catalogo`, `plan_a_dict`, `PlanDelegacion` (Task 3); el store de la Task 4; `delegacion_techos` (Task 6); `store.get_motor_governance()["facets"]` (E-03).
- Produces:
  - `routes._crear_pipeline(req: PipelineCreateRequest, contexto_delegacion: dict | None = None) -> tuple[Pipeline, list[Step]]` (bajo `_pipeline_create_lock`, sin lanzar la ejecución). Todo pipeline nuevo nace con `root_pipeline_id`; un hijo de Ada nace `queued` con `queued_at NULL`, identidad y `owner_ack_at` del padre.
  - `routes.crear_hijo_de_ada(req: PipelineCreateRequest, contexto_delegacion: dict) -> dict` → `{"pipeline_id", "status": "queued", "step_count"}`
  - `delegacion.ResultadoDelegacion(str, Enum)`: `LANZADA = "lanzada"`, `ESPERA_APROBACION = "espera_aprobacion"`, `RECHAZADA = "rechazada"`
  - `delegacion.Delegacion(resultado: ResultadoDelegacion, motivo: str | None = None, hijos: tuple[str, ...] = (), aprobacion_id: int | None = None)`
  - `delegacion.delegar(padre: Pipeline, paso: Step, plan: PlanDelegacion, *, por: str = "auto") -> Delegacion`
  - `delegacion.raiz_de(p: Pipeline) -> str`
  - Contexto de un hijo: `context["delegacion"] = {"subobjetivo_id": str, "parent_step": str, "depende_de": list[str] (pipeline_ids de hermanos), "skip_on_fail": bool}`
  - Step de integración agregado al padre: `Step(facet="ada", capability="integrate", depends_on=[paso.step_index], input={"prompt": objetivo, "objetivo_integracion": objetivo, "delegacion_step_id": paso.step_id})`

- [ ] **Step 1: Tests que fallan**

Create `tests/test_delegacion_politica_puro.py`:

```python
"""Frente G (2026-09-16): el límite de pipelines paralelos NO se aplica a la
creación de un hijo de Ada: el hijo nace `queued` y lo admite el despachador
(jacobs/cola.py) bajo el mismo candado. Una sola implementación del límite.

ROJO CONTRA LA BASE (F): validate_create rechaza ada con active_count >= 3."""
from __future__ import annotations

import os
from unittest.mock import patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

from jacobs import policy  # noqa: E402


def test_ada_no_choca_con_el_limite_de_paralelos():
    with patch.object(policy, "check_kill_switch", return_value=False):
        r = policy.validate_create("ada", "autonomous", 1, 99, subpipeline_token="t", parent_pipeline_id="p")
    assert r.ok, r.reason


def test_plataforma_sigue_con_el_limite():
    with patch.object(policy, "check_kill_switch", return_value=False):
        r = policy.validate_create("plataforma", "autonomous", 1, policy.MAX_PARALLEL_PIPELINES)
    assert not r.ok
```

Agregar en `jacobs/_delegacion_io_test.py`, antes de `if __name__ == "__main__":`:

```python
class DelegarTest(_ConBaseG):
    async def delegar(self, padre, paso, plan):
        from jacobs import delegacion
        return await delegacion.delegar(padre, paso, plan)

    async def test_plan_valido_crea_n_hijos_con_token_encolados(self):
        from jacobs.delegacion import ResultadoDelegacion
        p, paso = await self.padre()
        d = await self.delegar(p, paso, g.plan(3))
        self.assertEqual(d.resultado, ResultadoDelegacion.LANZADA, d.motivo)
        self.assertEqual(len(d.hijos), 3)
        self.assertEqual(await g.tokens_emitidos(p.pipeline_id), 3)
        usados = await ada.una_fila(
            "SELECT COUNT(*) AS n FROM jacobs_subpipeline_tokens WHERE parent_pipeline_id=%s AND usado_at IS NOT NULL",
            (p.pipeline_id,))
        self.assertEqual(usados["n"], 3)
        padre_db = await ada.una_fila("SELECT owner_ack_at FROM jacobs_pipelines WHERE pipeline_id=%s", (p.pipeline_id,))
        for hijo_id in d.hijos:
            h = await store.pipeline_get(hijo_id)
            self.assertEqual((h.status, h.parent_pipeline_id, h.depth, h.root_pipeline_id, h.invoked_by),
                             (PipelineStatus.queued, p.pipeline_id, 1, p.pipeline_id, "ada"))
            self.assertIsNotNone(h.queued_at)
            self.assertEqual((h.user_id, h.tenant_id), ("1", "1"))
            self.assertEqual([(s.facet, s.capability) for s in h.plan], [("jekyll", "analysis")])
            self.assertEqual(h.context["delegacion"]["parent_step"], paso.step_id)
            fila = await ada.una_fila("SELECT owner_ack_at FROM jacobs_pipelines WHERE pipeline_id=%s", (hijo_id,))
            self.assertEqual(fila["owner_ack_at"], padre_db["owner_ack_at"])
            self.assertEqual(len(await g.eventos(hijo_id, "SUBPIPELINE_ENCOLADO")), 1)
        padre = await store.pipeline_get(p.pipeline_id)
        self.assertEqual([s.capability for s in padre.plan], ["delegate", "integrate"])
        self.assertEqual(padre.plan[1].depends_on, [0])
        self.assertEqual(padre.plan[1].input["delegacion_step_id"], paso.step_id)
        aprobada = await g.eventos(p.pipeline_id, "DELEGACION_APROBADA")
        self.assertEqual(aprobada[0]["payload"]["por"], "auto")
        self.assertEqual(len(await g.eventos(p.pipeline_id, "DELEGACION_PROPUESTA")), 1)
        self.assertEqual((await store.arbol_consumo_leer(p.pipeline_id))["pipelines_total"], 4)

    async def test_dependencias_esperan_con_queued_at_null(self):
        p, paso = await self.padre()
        d = await self.delegar(p, paso, g.plan(2, depende={1: [0]}))
        primero, segundo = [await store.pipeline_get(h) for h in d.hijos]
        self.assertIsNotNone(primero.queued_at)
        self.assertIsNone(segundo.queued_at)
        self.assertEqual(segundo.context["delegacion"]["depende_de"], [primero.pipeline_id])

    async def test_cada_techo_excedido_pide_aprobacion_con_cero_tokens(self):
        from jacobs.delegacion import ResultadoDelegacion
        casos = [
            ("hijos_por_step", {"JAX_ADA_MAX_HIJOS_POR_STEP": "2"}, None),
            ("pipelines_arbol", {"JAX_ADA_MAX_PIPELINES_ARBOL": "3"}, None),
            ("tokens_arbol", {"JAX_ADA_MAX_TOKENS_ARBOL": "1000"}, None),
            ("usd_arbol", {"JAX_ADA_MAX_USD_ARBOL": "0.01"}, 0.02),
        ]
        for techo, entorno, costo_previo in casos:
            with self.subTest(techo=techo), patch.dict(os.environ, entorno):
                p, paso = await self.padre()
                if costo_previo is not None:
                    conn = await store.get_conn()
                    try:
                        async with conn.cursor() as cur:
                            await store.arbol_consumo_sumar_en(cur, p.pipeline_id, 10, costo_previo, time.time())
                    finally:
                        conn.close()
                self.telegram.reset_mock()
                d = await self.delegar(p, paso, g.plan(3))
                self.assertEqual(d.resultado, ResultadoDelegacion.ESPERA_APROBACION)
                self.assertEqual(await g.tokens_emitidos(p.pipeline_id), 0)
                self.assertEqual(await store.pipelines_hijos(p.pipeline_id), [])
                self.assertEqual(await store.pipeline_status(p.pipeline_id), PipelineStatus.awaiting_approval)
                aprobacion = await store.aprobacion_obtener(d.aprobacion_id)
                self.assertEqual((aprobacion["estado"], aprobacion["tipo"]), ("pendiente", "excede"))
                self.assertIn(techo, [t["techo"] for t in aprobacion["techos"]])
                self.assertEqual(len(aprobacion["plan"]["subobjetivos"]), 3)
                excede = await g.eventos(p.pipeline_id, "DELEGACION_EXCEDE_TECHO")
                self.assertIn(techo, [t["techo"] for t in excede[-1]["payload"]["techos"]])
                self.telegram.assert_awaited_once()

    async def test_profundidad_excedida_falla_sin_aprobacion(self):
        from jacobs.delegacion import ResultadoDelegacion
        os.environ["JAX_MAX_SUBPIPELINE_DEPTH"] = "1"
        p, paso = await self.padre(depth=1)
        d = await self.delegar(p, paso, g.plan(1))
        self.assertEqual(d.resultado, ResultadoDelegacion.RECHAZADA)
        self.assertIn("profundidad", d.motivo)
        self.assertIsNone(await store.aprobacion_pendiente_de(p.pipeline_id, paso.step_id, "excede"))
        self.assertEqual(await g.tokens_emitidos(p.pipeline_id), 0)
        self.assertEqual(await store.pipeline_status(p.pipeline_id), PipelineStatus.running)
        self.telegram.assert_awaited_once()

    async def test_consumo_incompleto_pide_aprobacion(self):
        from jacobs.delegacion import ResultadoDelegacion
        p, paso = await self.padre()
        await store.arbol_marcar_consumo_incompleto(p.pipeline_id)
        d = await self.delegar(p, paso, g.plan(1))
        self.assertEqual(d.resultado, ResultadoDelegacion.ESPERA_APROBACION)
        self.assertEqual(await g.tokens_emitidos(p.pipeline_id), 0)

    async def test_base_caida_al_medir_no_delega(self):
        from jacobs.delegacion import ResultadoDelegacion
        p, paso = await self.padre()
        with patch.object(store, "arbol_consumo_leer", AsyncMock(side_effect=OSError("base caída (simulada)"))):
            d = await self.delegar(p, paso, g.plan(2))
        self.assertEqual((d.resultado, d.motivo), (ResultadoDelegacion.RECHAZADA, "sin_medicion"))
        self.assertEqual(await g.tokens_emitidos(p.pipeline_id), 0)
        self.assertEqual(len(await g.eventos(p.pipeline_id, "DELEGACION_SIN_MEDICION")), 1)

    async def test_plan_parcial_nunca_se_lanza(self):
        from jacobs import delegacion
        from jacobs.subpipelines import EmisionRechazada, Motivo, emitir_token_subpipeline
        p, paso = await self.padre()
        llamadas = {"n": 0}

        async def emision_que_falla_al_tercero(padre_id, paso_id):
            llamadas["n"] += 1
            if llamadas["n"] == 3:
                raise EmisionRechazada(Motivo.KILL_SWITCH_ACTIVO)
            return await emitir_token_subpipeline(padre_id, paso_id)

        with patch.object(delegacion, "emitir_token_subpipeline", emision_que_falla_al_tercero):
            d = await delegacion.delegar(p, paso, g.plan(3))
        self.assertEqual(d.resultado, delegacion.ResultadoDelegacion.RECHAZADA)
        self.assertEqual(d.motivo, "kill_switch_activo")
        hijos = await store.pipelines_hijos(p.pipeline_id)
        self.assertEqual(len(hijos), 2)
        for h in hijos:
            self.assertEqual((h.status, h.queued_at), (PipelineStatus.aborted, None))
            self.assertEqual(len(await g.eventos(h.pipeline_id, "SUBPIPELINE_CANCELADO_POR_PADRE")), 1)
        self.assertEqual([s.capability for s in (await store.pipeline_get(p.pipeline_id)).plan], ["delegate"])

    async def test_catalogo_invalido_rechaza_sin_tokens(self):
        from jacobs.delegacion import ResultadoDelegacion
        p, paso = await self.padre()
        d = await self.delegar(p, paso, g.plan(1, faceta="inventada"))
        self.assertEqual(d.resultado, ResultadoDelegacion.RECHAZADA)
        self.assertIn("tabla `facet`", d.motivo)
        self.assertEqual(await g.tokens_emitidos(p.pipeline_id), 0)
```

En `tests/test_subpipeline_contrato_rutas.py` (de F), en `test_camino_legitimo_crea_el_hijo_con_padre_y_profundidad_del_token` y en `test_kill_switch_rechaza_sin_consumir_el_token`, cambiar `assert respuesta["status"] == "completed"` por:

```python
    # Frente G (2026-09-16): un hijo de Ada ya no corre al crearse; nace
    # `queued` y lo admite el despachador (jacobs/cola.py).
    assert respuesta["status"] == "queued"
```

```bash
cd $WT && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_delegacion_politica_puro.py 2>&1 | tee -a $L | tail -2
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_delegacion_io_test.py -k Delegar tests/test_subpipeline_contrato_rutas.py" 2>&1 | tee -a $L | tail -12
```
Expected:
- Primero: `1 failed, 1 passed`. Falla `test_ada_no_choca…`.
- Segundo: los 8 de `DelegarTest` en rojo (`ModuleNotFoundError: jacobs.delegacion`) y los 2 de F con `assert 'completed' == 'queued'`.

- [ ] **Step 2: La política**

En `jacobs/policy.py::validate_create`, reemplazar el bloque `if active_count >= MAX_PARALLEL_PIPELINES:` por:

```python
    # Frente G (2026-09-16): un hijo de Ada no se rechaza por cupo. Nace
    # `queued` y lo admite jacobs/cola.py::despachar() bajo el MISMO candado de
    # creación: el límite global tiene una sola implementación, la de la cola.
    if invoked_by != INVOKER_ADA and active_count >= MAX_PARALLEL_PIPELINES:
        return PolicyResult(
            ok=False,
            reason=(
                f"Ya hay {active_count} pipelines activos. "
                f"Límite duro: {MAX_PARALLEL_PIPELINES}"
            ),
        )
```

- [ ] **Step 3: La creación en `routes.py`**

Reemplazar `create_pipeline` (versión de F) por estas dos funciones más la ruta:

```python
async def _crear_pipeline(
    req: PipelineCreateRequest, contexto_delegacion: dict | None = None,
) -> tuple[Pipeline, list[Step]]:
    """Cuerpo de la creación bajo el candado (frentes F y G). NO lanza la
    ejecución: la ruta HTTP lo hace para los pipelines de la plataforma y la
    cola (jacobs/cola.py) para los hijos de Ada."""
    async with _pipeline_create_lock:
        active_count = await store.pipeline_count_active()
        policy = validate_create(
            invoked_by=req.invoked_by,
            mode=req.mode,
            max_steps=req.max_steps,
            active_count=active_count,
            subpipeline_token=req.subpipeline_token,
            parent_pipeline_id=req.parent_pipeline_id,
        )
        if not policy.ok:
            status_code = 423 if "kill switch" in policy.reason.lower() else 422
            raise HTTPException(status_code=status_code, detail=policy.reason)

        pipeline_id = str(uuid.uuid4())

        # Frente F: el hijo de Ada consume su token acá (después de
        # validate_create, antes de planificar). Padre y profundidad salen de la
        # fila del token, nunca del cuerpo.
        parent_pipeline_id: str | None = None
        parent_step: str | None = None
        depth = 0
        padre: Pipeline | None = None
        if req.invoked_by == INVOKER_ADA:
            consumo = await consumir_token_subpipeline(
                req.subpipeline_token, req.parent_pipeline_id, pipeline_id,
            )
            if isinstance(consumo, ConsumoRechazado):
                raise HTTPException(
                    status_code=403,
                    detail=f"subpipeline_token rechazado: {consumo.motivo.value}",
                )
            parent_pipeline_id = consumo.parent_pipeline_id
            parent_step = consumo.parent_step
            depth = consumo.depth
            # Frente G: identidad, dueño y raíz salen del PADRE (fila del
            # servidor), no del cuerpo del pedido.
            padre = await store.pipeline_get(parent_pipeline_id)
            if padre is None:
                raise HTTPException(status_code=403, detail="subpipeline_token rechazado: padre_desconocido")

        steps_spec = [s.model_dump() for s in req.steps] if req.steps else None
        try:
            steps = await _build_plan_or_reject(pipeline_id, req.objective, req.max_steps, steps_spec)
        except HTTPException:
            if parent_pipeline_id is not None:
                await store.event_append(pipeline_id, "SUBPIPELINE_RECHAZADO", {
                    "fase": "plan",
                    "motivo": Motivo.PLAN_RECHAZADO.value,
                    "parent_pipeline_id": parent_pipeline_id,
                })
            raise

        for step in steps:
            step.pipeline_id = pipeline_id

        now = time.time()
        contexto: dict = {"objective": req.objective}
        if contexto_delegacion is not None:
            contexto["delegacion"] = contexto_delegacion
        pipeline = Pipeline(
            pipeline_id=pipeline_id,
            name=req.name,
            invoked_by=req.invoked_by,
            user_id=padre.user_id if padre else req.user_id,
            tenant_id=padre.tenant_id if padre else req.tenant_id,
            parent_pipeline_id=parent_pipeline_id,
            depth=depth,
            root_pipeline_id=(padre.root_pipeline_id or padre.pipeline_id) if padre else pipeline_id,
            status=PipelineStatus.queued if padre else PipelineStatus.pending,
            mode=req.mode,
            plan=steps,
            max_steps=req.max_steps,
            context=contexto,
            created_at=now,
            updated_at=now,
        )

        await store.pipeline_create(pipeline)
        for step in steps:
            await store.step_upsert(step)
        await store.event_append(pipeline_id, "PIPELINE_CREATED", {
            "name": req.name, "mode": req.mode, "steps": len(steps),
            "parent_pipeline_id": parent_pipeline_id, "depth": depth,
            "root_pipeline_id": pipeline.root_pipeline_id,
        })
        if padre is not None:
            await store.pipeline_heredar_dueno(pipeline_id, parent_pipeline_id)
            await store.event_append(
                parent_pipeline_id, "SUBPIPELINE_CREADO",
                {"hijo_pipeline_id": pipeline_id, "depth": depth},
                parent_step,
            )
            await store.event_append(pipeline_id, "SUBPIPELINE_ENCOLADO", {
                "parent_pipeline_id": parent_pipeline_id,
                "subobjetivo_id": (contexto_delegacion or {}).get("subobjetivo_id"),
            })
    return pipeline, steps


async def crear_hijo_de_ada(req: PipelineCreateRequest, contexto_delegacion: dict) -> dict:
    """Frente G: la usa jacobs/delegacion.py dentro del proceso (no es ruta).
    El hijo queda `queued` con queued_at NULL: delegacion.py fija la cola
    cuando el plan entero quedó creado (nunca se lanza un plan parcial)."""
    pipeline, steps = await _crear_pipeline(req, contexto_delegacion)
    return {"pipeline_id": pipeline.pipeline_id, "status": pipeline.status.value, "step_count": len(steps)}


@router.post("/pipeline")
async def create_pipeline(req: PipelineCreateRequest, background: BackgroundTasks) -> dict:
    """Crea un pipeline y lo ejecuta en background (plataforma), o lo encola
    (hijo de Ada que entra por HTTP: lo admite el despachador)."""
    pipeline, steps = await _crear_pipeline(req)
    pipeline_id = pipeline.pipeline_id

    if pipeline.status == PipelineStatus.queued:
        await store.pipeline_fijar_cola([pipeline_id], time.time())
        return {
            "pipeline_id": pipeline_id,
            "status": "queued",
            "mode": req.mode,
            "step_count": len(steps),
        }

    # dry_run: no ejecuta en background, solo completa inmediatamente
    if req.mode == "dry_run":
        await store.pipeline_update_status(pipeline_id, PipelineStatus.completed)
        await store.event_append(pipeline_id, "DRY_RUN_COMPLETE")
        return {
            "pipeline_id": pipeline_id,
            "status": "completed",
            "mode": "dry_run",
            "plan": [s.model_dump() for s in steps],
            "note": "dry_run — ningún step fue ejecutado",
        }

    background.add_task(run_pipeline, pipeline)

    return {
        "pipeline_id": pipeline_id,
        "status": "running",
        "mode": req.mode,
        "step_count": len(steps),
        "message": "Pipeline iniciado. Consultar GET /jacobs/pipeline/{id}",
    }
```

- [ ] **Step 4: `jacobs/delegacion.py`**

Create `jacobs/delegacion.py`:

```python
"""
Jacobs — Emisor de sub-pipelines de Ada (frente G, 2026-09-16).

Spec: docs/superpowers/specs/2026-09-16-emisor-subpipelines-ada-design.md.
ÚNICO punto de decisión de la delegación: el modo 1 (step `delegate` con un
plan_delegacion.v1, desde executor.py) y el modo 2 (tool `lanzar_subpipeline`,
desde motor_registry/tool_authority.py) llaman a ESTE módulo. Mismos techos,
mismo token (frente F), misma cola (cola.py), misma aprobación.

Reglas que este módulo hace cumplir:
- Dentro de techos: un token por sub-objetivo (emitir_token_subpipeline) y un
  hijo `queued` por routes.crear_hijo_de_ada. Todo o nada: si falla uno, lo ya
  creado se aborta antes de que la cola lo vea (queued_at NULL).
- Fuera de techos: padre a awaiting_approval, evento DELEGACION_EXCEDE_TECHO,
  aviso a Telegram y CERO tokens. La profundidad no se aprueba (contrato F).
- Sin poder medir (base caída): no se delega.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum

from jacobs import store
from jacobs.config_delegacion import config_delegacion
from jacobs.delegacion_techos import Consumo, TechoExcedido, evaluar_techos, proyectar
from jacobs.models import (
    CAPABILITY_INTEGRAR,
    INVOKER_ADA,
    Pipeline,
    PipelineCreateRequest,
    PipelineStatus,
    Step,
    StepSpec,
)
from jacobs.subpipelines import EmisionRechazada, config_subpipelines, emitir_token_subpipeline
from motor_registry.esquema_plan_delegacion import (
    PlanDelegacion,
    errores_de_catalogo,
    orden_topologico,
    plan_a_dict,
)

logger = logging.getLogger("jacobs.delegacion")

LARGO_MAX_NOMBRE_HIJO = 200


class ResultadoDelegacion(str, Enum):
    LANZADA = "lanzada"
    ESPERA_APROBACION = "espera_aprobacion"
    RECHAZADA = "rechazada"


@dataclass(frozen=True)
class Delegacion:
    resultado: ResultadoDelegacion
    motivo: str | None = None
    hijos: tuple[str, ...] = ()
    aprobacion_id: int | None = None


def raiz_de(p: Pipeline) -> str:
    return p.root_pipeline_id or p.pipeline_id


async def _notificar(pipeline_id: str, texto: str) -> None:
    """Telegram (texto de servidor, como las alertas del reaper). Si no se
    entrega, queda el evento: nunca un aviso perdido sin rastro."""
    from jacobs import reaper
    resultado = await reaper.send_telegram_alert(texto)
    if not resultado["ok"]:
        await store.event_append(pipeline_id, "DELEGACION_NOTIFICACION_FALLIDA", {"error": resultado["error"]})


def _texto_techos(excedidos: list[TechoExcedido]) -> str:
    return "; ".join(f"{e.techo}: {e.proyectado:g} > {e.limite:g}" for e in excedidos)


async def delegar(padre: Pipeline, paso: Step, plan: PlanDelegacion, *, por: str = "auto") -> Delegacion:
    cfg = config_delegacion()
    max_profundidad = config_subpipelines().max_profundidad
    raiz = raiz_de(padre)
    try:
        governance = await store.get_motor_governance()
        await store.pipeline_raiz_fijar(padre.pipeline_id)
        consumo = Consumo.de_fila(await store.arbol_consumo_leer(raiz))
        en_arbol = await store.pipelines_del_arbol_contar(raiz)
        existentes = await store.hijos_del_paso_contar(padre.pipeline_id, paso.step_id)
    except Exception as exc:  # fail-closed: sin medir los techos no se delega (spec §5); el llamador falla el step con este motivo
        logger.error("Delegación de %s sin medición: %s", padre.pipeline_id, exc, exc_info=True)
        try:
            await store.event_append(
                padre.pipeline_id, "DELEGACION_SIN_MEDICION",
                {"error": f"{type(exc).__name__}: {exc}"}, paso.step_id,
            )
        except Exception:  # fail-soft: la base tampoco acepta el evento; el ERROR de arriba ya lo dejó en el journal y el resultado sigue siendo RECHAZADA
            logger.error("Tampoco se pudo registrar DELEGACION_SIN_MEDICION de %s", padre.pipeline_id, exc_info=True)
        return Delegacion(ResultadoDelegacion.RECHAZADA, "sin_medicion")

    errores = errores_de_catalogo(plan, governance["facets"], governance["capabilities"])
    if errores:
        await store.event_append(padre.pipeline_id, "DELEGACION_PLAN_INVALIDO",
                                 {"fase": "catalogo", "errores": errores}, paso.step_id)
        return Delegacion(ResultadoDelegacion.RECHAZADA, "plan_invalido: " + "; ".join(errores))

    proyeccion = proyectar(
        depth_padre=padre.depth, hijos_existentes_step=existentes,
        hijos_nuevos=len(plan.subobjetivos), pipelines_arbol_actual=en_arbol,
        consumo=consumo, cfg=cfg,
    )
    await store.event_append(padre.pipeline_id, "DELEGACION_PROPUESTA", {
        "subobjetivos": [s.id for s in plan.subobjetivos],
        "proyeccion": asdict(proyeccion),
        "tokens_sin_precio": consumo.tokens_sin_precio,
    }, paso.step_id)

    excedidos = evaluar_techos(proyeccion, consumo, cfg, max_profundidad)
    if excedidos:
        return await _pedir_aprobacion(padre, paso, plan, raiz, excedidos)
    return await _lanzar(padre, paso, plan, raiz, por)


async def _pedir_aprobacion(
    padre: Pipeline, paso: Step, plan: PlanDelegacion, raiz: str, excedidos: list[TechoExcedido],
) -> Delegacion:
    techos = [e.a_dict() for e in excedidos]
    await store.event_append(padre.pipeline_id, "DELEGACION_EXCEDE_TECHO",
                             {"techos": techos, "raiz": raiz}, paso.step_id)
    no_aprobables = [e for e in excedidos if not e.aprobable]
    if no_aprobables:
        await _notificar(padre.pipeline_id, (
            f"JAX · Ada no puede delegar en el pipeline {padre.pipeline_id[:8]}: "
            f"{_texto_techos(no_aprobables)}. Ese límite no se aprueba; el step falla."
        ))
        return Delegacion(ResultadoDelegacion.RECHAZADA,
                          "techo_no_aprobable: " + ", ".join(e.techo for e in no_aprobables))

    existente = await store.aprobacion_pendiente_de(padre.pipeline_id, paso.step_id, "excede")
    if existente is not None:
        aprobacion_id = existente["id"]
        await store.aprobacion_actualizar_plan(aprobacion_id, techos, plan_a_dict(plan))
    else:
        aprobacion_id = await store.aprobacion_crear(
            raiz, padre.pipeline_id, paso.step_id, "excede", techos, plan_a_dict(plan))

    if not await store.pipeline_transicion(padre.pipeline_id, (PipelineStatus.running,),
                                           PipelineStatus.awaiting_approval):
        return Delegacion(ResultadoDelegacion.RECHAZADA, "padre_no_corre", aprobacion_id=aprobacion_id)

    await _notificar(padre.pipeline_id, (
        f"JAX · Ada pide aprobación (árbol {raiz[:8]}, pipeline {padre.pipeline_id[:8]}): "
        f"{_texto_techos(excedidos)}. Aprobar o rechazar en Admin → Delegaciones."
    ))
    return Delegacion(ResultadoDelegacion.ESPERA_APROBACION, aprobacion_id=aprobacion_id)


async def _lanzar(padre: Pipeline, paso: Step, plan: PlanDelegacion, raiz: str, por: str) -> Delegacion:
    from jacobs import routes

    creados: list[str] = []
    por_id: dict[str, str] = {}
    try:
        for so in orden_topologico(plan):
            token = await emitir_token_subpipeline(padre.pipeline_id, paso.step_id)
            req = PipelineCreateRequest(
                name=f"{padre.name} · {so.id}"[:LARGO_MAX_NOMBRE_HIJO],
                objective=so.objetivo,
                invoked_by=INVOKER_ADA,
                mode="autonomous",
                max_steps=1,
                steps=[StepSpec(facet=so.faceta, capability=so.capability, prompt=so.objetivo)],
                subpipeline_token=token,
                parent_pipeline_id=padre.pipeline_id,
            )
            creado = await routes.crear_hijo_de_ada(req, {
                "subobjetivo_id": so.id,
                "parent_step": paso.step_id,
                "depende_de": [por_id[d] for d in so.depende_de],
                "skip_on_fail": so.skip_on_fail,
            })
            por_id[so.id] = creado["pipeline_id"]
            creados.append(creado["pipeline_id"])
    except Exception as exc:  # fail-closed: nunca se lanza un plan parcial (spec §5); lo creado se aborta antes de que la cola lo vea
        for hijo in creados:
            if await store.pipeline_transicion(hijo, (PipelineStatus.queued,), PipelineStatus.aborted):
                await store.event_append(hijo, "SUBPIPELINE_CANCELADO_POR_PADRE",
                                         {"padre": padre.pipeline_id, "motivo": "plan_parcial"})
        if isinstance(exc, EmisionRechazada):
            motivo = exc.motivo.value
        else:
            motivo = f"{type(exc).__name__}: {getattr(exc, 'detail', exc)}"
        await store.event_append(padre.pipeline_id, "DELEGACION_FALLIDA",
                                 {"motivo": motivo, "creados_abortados": creados}, paso.step_id)
        return Delegacion(ResultadoDelegacion.RECHAZADA, motivo)

    sin_dependencias = [por_id[so.id] for so in plan.subobjetivos if not so.depende_de]
    await store.pipeline_fijar_cola(sin_dependencias, time.time())
    await store.arbol_sumar_pipelines(raiz, len(creados))
    await _agregar_integracion(padre, paso, plan.integracion_objetivo)
    await store.event_append(padre.pipeline_id, "DELEGACION_APROBADA",
                             {"por": por, "hijos": por_id}, paso.step_id)
    return Delegacion(ResultadoDelegacion.LANZADA, hijos=tuple(creados))


async def _agregar_integracion(padre: Pipeline, paso: Step, objetivo: str) -> None:
    ya = any(
        s.capability == CAPABILITY_INTEGRAR and s.input.get("delegacion_step_id") == paso.step_id
        for s in padre.plan
    )
    if ya:
        return
    integracion = Step(
        pipeline_id=padre.pipeline_id,
        step_index=len(padre.plan),
        facet="ada",
        capability=CAPABILITY_INTEGRAR,
        depends_on=[paso.step_index],
        timeout_seconds=paso.timeout_seconds,
        input={"prompt": objetivo, "objetivo_integracion": objetivo, "delegacion_step_id": paso.step_id},
    )
    padre.plan.append(integracion)
    await store.step_upsert(integracion)
    await store.pipeline_plan_guardar(padre.pipeline_id, padre.plan)
```

- [ ] **Step 5: Verde**

```bash
cd $WT && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_delegacion_politica_puro.py tests/test_subpipeline_contrato_puro.py tests/test_jacobs_invoked_by_rol.py 2>&1 | tail -1
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_delegacion_io_test.py tests/test_subpipeline_contrato_rutas.py jacobs/_subpipeline_contrato_io_test.py" 2>&1 | tail -3
cd $WT && $PY -m pytest -q policy/tests/test_no_fail_open_except.py tests/test_no_blocking_in_async.py 2>&1 | tail -1
```
Expected: todo `passed`. En `_delegacion_io_test.py` quedan 31: las 18 de la Task 4, las 5 de la Task 5 y las 8 de `DelegarTest`.

- [ ] **Step 6: Mutación y commit**

En `_lanzar`, agregar `await store.pipeline_fijar_cola([creado["pipeline_id"]], time.time())` justo después de `creados.append(creado["pipeline_id"])`, o sea, fijar la cola hijo por hijo al crearlo. Correr `-k test_plan_parcial_nunca_se_lanza`. Expected: `1 failed`: los hijos abortados quedan con `queued_at` no nulo, y el despachador ya los habría podido admitir antes del fallo. Borrar la línea agregada y pegar las dos salidas en `$L`.

```bash
git -C $WT add jacobs/policy.py jacobs/routes.py jacobs/delegacion.py tests/test_delegacion_politica_puro.py tests/test_subpipeline_contrato_rutas.py jacobs/_delegacion_io_test.py $L
git -C $WT commit -m "feat(jacobs): delegacion.delegar -- dentro de techos lanza hijos encolados, fuera pide aprobación

Único punto de decisión. Un token por sub-objetivo, hijos queued con identidad
y dueño del padre, integración agregada al padre; techos excedidos ->
awaiting_approval + Telegram + cero tokens; profundidad no aprobable; base
caída -> no delega; plan parcial -> se aborta lo creado. validate_create deja
el cupo a la cola. 1 puro + 8 de DB vistos en rojo; F: hijos 'queued'.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---
### Task 8: La cola, el cruce a mitad de camino, las dependencias y la reanudación del padre

**Files:**
- Create: `jacobs/cola.py`
- Modify: `jacobs/delegacion.py` (al final: esperas, cascada, dependencias, reanudación, cruce)
- Modify: `las_manos/server.py` (`_jacobs_init`: arranque del despachador junto al reaper)
- Modify: `tests/test_delegacion_politica_puro.py` (test de arranque)
- Modify: `jacobs/_delegacion_io_test.py` (clases `ColaTest`, `ArbolTest`)

**Interfaces:**
- Consumes: Tasks 4, 6 y 7; `executor.run_pipeline(pipeline)`, `executor._load_ref(ref) -> dict`, `executor.RefIlegible` (existentes).
- Produces:
  - `cola.lanzar_en_fondo(corrutina) -> asyncio.Task` (guarda la referencia; loguea la excepción al terminar)
  - `cola.despachar() -> list[str]` (ids admitidos; `[]` con kill switch)
  - `cola.vencer_esperas(ahora: float | None = None) -> list[str]`
  - `cola.bucle_despachador() -> None` (no retorna)
  - `delegacion.al_terminar_pipeline(pipeline_id: str) -> None` (hook: cascada si no terminó bien, dependientes, reanudar padre, despachar)
  - `delegacion.cortar_descendientes(pipeline_id: str, motivo: str) -> int`
  - `delegacion.pasar_a_espera_de_hijos(pipeline_id: str) -> None`
  - `delegacion.hay_hijos_sin_terminar(pipeline_id: str) -> bool`
  - `delegacion.reanudar_padre_si_corresponde(padre_pipeline_id: str) -> bool`
  - `delegacion.reconciliar_esperas() -> int`
  - `delegacion.frenar_arbol_por_cruce(raiz: str, cruzados: list[TechoExcedido]) -> int`

- [ ] **Step 1: Tests que fallan**

Agregar a `tests/test_delegacion_politica_puro.py`:

```python
from pathlib import Path  # noqa: E402


def test_las_manos_arranca_el_despachador_junto_al_reaper():
    fuente = (Path(__file__).resolve().parents[1] / "las_manos" / "server.py").read_text(encoding="utf-8")
    inicio = fuente.index("async def _jacobs_init")
    bloque = fuente[inicio:fuente.index("\n\napp.include_router", inicio)]
    assert "start_reaper_loop()" in bloque
    assert "lanzar_en_fondo(bucle_despachador())" in bloque
```

Agregar a `jacobs/_delegacion_io_test.py`, antes de `if __name__ == "__main__":`:

```python
class _ConColaG(_ConBaseG):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        corre = patch("jacobs.executor.run_pipeline", AsyncMock())
        self.run_pipeline = corre.start()
        self.addCleanup(corre.stop)

    async def delegar_y_esperar(self, n, **kw):
        from jacobs import delegacion
        p, paso = await self.padre()
        d = await delegacion.delegar(p, paso, g.plan(n, **kw))
        self.assertEqual(d.resultado, delegacion.ResultadoDelegacion.LANZADA, d.motivo)
        return p, paso, d

    async def terminar(self, pipeline_id, estado, resultado="hecho"):
        """Simula que el hijo corrió: step con salida inline y estado final."""
        pasos = await store.steps_by_pipeline(pipeline_id)
        await ada.ejecutar(
            "UPDATE jacobs_steps SET status=%s, output_ref=%s WHERE step_id=%s",
            ("completed" if estado == "completed" else "failed",
             "inline:" + json.dumps({"result": resultado}) if estado == "completed" else None,
             pasos[0].step_id))
        await ada.ejecutar("UPDATE jacobs_pipelines SET status=%s WHERE pipeline_id=%s", (estado, pipeline_id))


class ColaTest(_ConColaG):
    async def test_admite_fifo_hasta_el_limite_global(self):
        from jacobs import cola
        base = time.time()
        encolados = [await self.pipeline(status=PipelineStatus.queued, queued_at=base + i) for i in range(5)]
        admitidos = await cola.despachar()
        self.assertEqual(admitidos, [p.pipeline_id for p in encolados[:3]])
        self.assertEqual(self.run_pipeline.call_count, 3)
        for p in encolados[:3]:
            self.assertEqual(await store.pipeline_status(p.pipeline_id), PipelineStatus.pending)
            self.assertEqual(len(await g.eventos(p.pipeline_id, "SUBPIPELINE_ADMITIDO")), 1)
        self.assertEqual(await cola.despachar(), [])
        self.assertEqual(await store.pipeline_status(encolados[3].pipeline_id), PipelineStatus.queued)

    async def test_con_kill_switch_no_admite_nada(self):
        from jacobs import cola
        p = await self.pipeline(status=PipelineStatus.queued, queued_at=time.time())
        with patch("jacobs.policy.check_kill_switch", return_value=True):
            self.assertEqual(await cola.despachar(), [])
        self.assertEqual(await store.pipeline_status(p.pipeline_id), PipelineStatus.queued)
        self.run_pipeline.assert_not_called()

    async def test_espera_maxima_vence(self):
        from jacobs import cola
        viejo = await self.pipeline(status=PipelineStatus.queued, queued_at=time.time() - 3601)
        nuevo = await self.pipeline(status=PipelineStatus.queued, queued_at=time.time())
        self.assertEqual(await cola.vencer_esperas(), [viejo.pipeline_id])
        self.assertEqual(await store.pipeline_status(viejo.pipeline_id), PipelineStatus.expired)
        self.assertEqual(await store.pipeline_status(nuevo.pipeline_id), PipelineStatus.queued)
        self.assertEqual(len(await g.eventos(viejo.pipeline_id, "SUBPIPELINE_EXPIRADO_EN_COLA")), 1)

    async def test_cruce_a_mitad_de_camino_frena_lo_pendiente(self):
        from jacobs import cola
        p, _paso, d = await self.delegar_y_esperar(2)
        await ada.ejecutar("UPDATE jacobs_pipelines SET status='waiting_children' WHERE pipeline_id=%s", (p.pipeline_id,))
        corriendo = await self.pipeline(status=PipelineStatus.running, root=p.pipeline_id, parent=p.pipeline_id)
        conn = await store.get_conn()
        try:
            async with conn.cursor() as cur:
                await store.arbol_consumo_sumar_en(cur, p.pipeline_id, 2_000_001, 0.1, time.time())
        finally:
            conn.close()
        self.telegram.reset_mock()
        self.assertEqual(await cola.despachar(), [])
        for hijo in d.hijos:
            self.assertEqual(await store.pipeline_status(hijo), PipelineStatus.awaiting_approval)
        self.assertEqual(await store.pipeline_status(corriendo.pipeline_id), PipelineStatus.running)
        aprobacion = await store.aprobacion_pendiente_de_raiz(p.pipeline_id, "cruzado")
        self.assertEqual(aprobacion["techos"][0]["techo"], "tokens_arbol")
        self.assertEqual(sorted(aprobacion["plan"]["frenados"]), sorted(d.hijos))
        self.assertEqual(len(await g.eventos(p.pipeline_id, "DELEGACION_TECHO_CRUZADO")), 1)
        self.telegram.assert_awaited_once()


class ArbolTest(_ConColaG):
    async def test_padre_abortado_no_deja_huerfanos(self):
        from jacobs import delegacion
        p, _paso, d = await self.delegar_y_esperar(3)
        nieto = await self.pipeline(status=PipelineStatus.queued, root=p.pipeline_id, parent=d.hijos[0],
                                    queued_at=time.time())
        await store.pipeline_transicion(p.pipeline_id, (PipelineStatus.running,), PipelineStatus.aborted)
        await delegacion.al_terminar_pipeline(p.pipeline_id)
        for pid in (*d.hijos, nieto.pipeline_id):
            self.assertEqual(await store.pipeline_status(pid), PipelineStatus.aborted)
            self.assertEqual(len(await g.eventos(pid, "SUBPIPELINE_CANCELADO_POR_PADRE")), 1)
        self.run_pipeline.assert_not_called()

    async def test_dependiente_arranca_cuando_termina_su_dependencia(self):
        from jacobs import delegacion
        _p, _paso, d = await self.delegar_y_esperar(2, depende={1: [0]})
        await self.terminar(d.hijos[0], "completed")
        await delegacion.al_terminar_pipeline(d.hijos[0])
        self.assertIsNotNone((await store.pipeline_get(d.hijos[1])).queued_at)

    async def test_dependencia_fallida_sin_skip_no_arranca_y_se_declara(self):
        from jacobs import delegacion
        _p, _paso, d = await self.delegar_y_esperar(2, depende={1: [0]})
        await self.terminar(d.hijos[0], "aborted")
        await delegacion.al_terminar_pipeline(d.hijos[0])
        self.assertEqual(await store.pipeline_status(d.hijos[1]), PipelineStatus.aborted)
        self.assertEqual(len(await g.eventos(d.hijos[1], "SUBPIPELINE_DEPENDENCIA_FALLIDA")), 1)

    async def test_dependencia_fallida_con_skip_on_fail_deja_arrancar(self):
        from jacobs import delegacion
        _p, _paso, d = await self.delegar_y_esperar(2, depende={1: [0]}, skip={0})
        await self.terminar(d.hijos[0], "aborted")
        await delegacion.al_terminar_pipeline(d.hijos[0])
        hijo = await store.pipeline_get(d.hijos[1])
        # al_terminar_pipeline despacha la cola al final: el dependiente ya entró.
        self.assertEqual(hijo.status, PipelineStatus.pending)
        self.assertIsNotNone(hijo.queued_at)
        self.assertEqual(len(await g.eventos(d.hijos[1], "SUBPIPELINE_ADMITIDO")), 1)

    async def test_padre_se_reanuda_con_la_integracion_cuando_terminan_todos(self):
        from jacobs import delegacion
        p, _paso, d = await self.delegar_y_esperar(2)
        await delegacion.pasar_a_espera_de_hijos(p.pipeline_id)
        self.assertEqual(await store.pipeline_status(p.pipeline_id), PipelineStatus.waiting_children)
        await ada.ejecutar("UPDATE jacobs_pipelines SET status='queued' WHERE pipeline_id IN (%s,%s)", d.hijos)
        await self.terminar(d.hijos[0], "completed", "resultado cero")
        await delegacion.al_terminar_pipeline(d.hijos[0])
        self.assertEqual(await store.pipeline_status(p.pipeline_id), PipelineStatus.waiting_children)
        await self.terminar(d.hijos[1], "aborted")
        self.run_pipeline.reset_mock()
        await delegacion.al_terminar_pipeline(d.hijos[1])
        self.assertEqual(await store.pipeline_status(p.pipeline_id), PipelineStatus.running)
        relanzado = [c.args[0] for c in self.run_pipeline.call_args_list if c.args[0].pipeline_id == p.pipeline_id]
        self.assertEqual(len(relanzado), 1)
        prompt = relanzado[0].plan[1].input["prompt"]
        for fragmento in ("s0", "s1", "resultado cero", "completed", "aborted", "fallo declarado"):
            self.assertIn(fragmento, prompt)
        self.assertEqual(len(await g.eventos(p.pipeline_id, "DELEGACION_HIJOS_TERMINADOS")), 1)

    async def test_reconciliar_corta_hijos_encolados_de_un_padre_abortado(self):
        from jacobs import delegacion
        p, _paso, d = await self.delegar_y_esperar(2)
        await ada.ejecutar("UPDATE jacobs_pipelines SET status='aborted' WHERE pipeline_id=%s", (p.pipeline_id,))
        await delegacion.reconciliar_esperas()
        for hijo in d.hijos:
            self.assertEqual(await store.pipeline_status(hijo), PipelineStatus.aborted)

    async def test_reconciliar_reanuda_si_el_hook_se_perdio(self):
        from jacobs import delegacion
        p, _paso, d = await self.delegar_y_esperar(1)
        await store.pipeline_transicion(p.pipeline_id, (PipelineStatus.running,), PipelineStatus.waiting_children)
        await self.terminar(d.hijos[0], "completed")
        self.assertEqual(await delegacion.reconciliar_esperas(), 1)
        self.assertEqual(await store.pipeline_status(p.pipeline_id), PipelineStatus.running)
```

```bash
cd $WT && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_delegacion_politica_puro.py 2>&1 | tee -a $L | tail -2
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_delegacion_io_test.py -k 'ColaTest or ArbolTest'" 2>&1 | tee -a $L | tail -12
```
Expected: `1 failed, 2 passed` en el primero y `11 failed` en el segundo (`ModuleNotFoundError: jacobs.cola` o `AttributeError` de `delegacion`).

- [ ] **Step 2: `jacobs/cola.py`**

Create `jacobs/cola.py`:

```python
"""
Jacobs — Cola de sub-pipelines de Ada (frente G, 2026-09-16).

Spec §3.1.4. Todo hijo de Ada nace `queued`; se admite ACÁ, en orden FIFO por
queued_at, bajo el MISMO candado de creación (routes._pipeline_create_lock) y
con el MISMO límite global (policy.MAX_PARALLEL_PIPELINES): una sola
implementación del cupo. Un padre en waiting_children no ocupa cupo.

- Kill switch puesto: no se admite nada (spec §3.3).
- Espera máxima (JAX_ADA_COLA_MAX_ESPERA_SEGUNDOS, desde queued_at): expired.
- Árbol que cruzó un techo a mitad de camino: sus hijos sin arrancar pasan a
  awaiting_approval (delegacion.frenar_arbol_por_cruce).

El bucle corre cada JAX_ADA_COLA_CADENCIA_SEGUNDOS, además de las llamadas por
evento (al delegar, al terminar un pipeline, al aprobar).

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import asyncio
import logging
import time

from jacobs import policy, store
from jacobs.config_delegacion import config_delegacion
from jacobs.delegacion_techos import Consumo, techos_cruzados
from jacobs.models import Pipeline, PipelineStatus

logger = logging.getLogger("jacobs.cola")

# Candidatos leídos por cupo libre: deja saltar hijos de árboles frenados sin
# otra consulta. No es un límite funcional: lo que no entra se ve en la vuelta
# siguiente.
CANDIDATOS_POR_CUPO = 10

_TAREAS: set[asyncio.Task] = set()


def _al_terminar_tarea(tarea: asyncio.Task) -> None:
    _TAREAS.discard(tarea)
    if tarea.cancelled():
        return
    exc = tarea.exception()
    if exc is not None:
        logger.error("Cola: tarea de fondo terminó con error: %r", exc, exc_info=exc)


def lanzar_en_fondo(corrutina) -> asyncio.Task:
    """asyncio.create_task sin referencia puede ser recolectada a mitad de camino;
    acá queda en _TAREAS hasta terminar, y su excepción no muere en silencio."""
    tarea = asyncio.create_task(corrutina)
    _TAREAS.add(tarea)
    tarea.add_done_callback(_al_terminar_tarea)
    return tarea


async def despachar() -> list[str]:
    if policy.check_kill_switch():
        logger.info("Cola: kill switch puesto -- no se admite ningún sub-pipeline")
        return []
    from jacobs import delegacion, executor, routes

    cfg = config_delegacion()
    admitidos: list[Pipeline] = []
    cruces: dict[str, list] = {}
    async with routes._pipeline_create_lock:
        libres = policy.MAX_PARALLEL_PIPELINES - await store.pipeline_count_active()
        if libres <= 0:
            return []
        consumos: dict[str, Consumo] = {}
        for candidato in await store.cola_candidatos(limite=libres * CANDIDATOS_POR_CUPO):
            if len(admitidos) >= libres:
                break
            raiz = candidato.root_pipeline_id or candidato.pipeline_id
            if raiz not in consumos:
                consumos[raiz] = Consumo.de_fila(await store.arbol_consumo_leer(raiz))
                cruzados = techos_cruzados(consumos[raiz], cfg)
                if cruzados:
                    cruces[raiz] = cruzados
            if raiz in cruces:
                continue
            if not await store.pipeline_transicion(candidato.pipeline_id, (PipelineStatus.queued,),
                                                   PipelineStatus.pending):
                continue
            await store.event_append(candidato.pipeline_id, "SUBPIPELINE_ADMITIDO", {
                "espera_segundos": round(time.time() - (candidato.queued_at or time.time()), 3),
            })
            candidato.status = PipelineStatus.pending
            admitidos.append(candidato)
    # Fuera del candado: Telegram y la cascada no retienen la creación de pipelines.
    for raiz, cruzados in cruces.items():
        await delegacion.frenar_arbol_por_cruce(raiz, cruzados)
    for p in admitidos:
        lanzar_en_fondo(executor.run_pipeline(p))
    return [p.pipeline_id for p in admitidos]


async def vencer_esperas(ahora: float | None = None) -> list[str]:
    from jacobs import delegacion

    cfg = config_delegacion()
    ahora = time.time() if ahora is None else ahora
    vencidos: list[str] = []
    for p in await store.cola_vencidos(ahora - cfg.cola_max_espera_segundos):
        if not await store.pipeline_transicion(p.pipeline_id, (PipelineStatus.queued,), PipelineStatus.expired):
            continue
        await store.event_append(p.pipeline_id, "SUBPIPELINE_EXPIRADO_EN_COLA", {
            "espera_segundos": round(ahora - (p.queued_at or ahora), 3),
            "maximo_segundos": cfg.cola_max_espera_segundos,
        })
        vencidos.append(p.pipeline_id)
        await delegacion.al_terminar_pipeline(p.pipeline_id)
    return vencidos


async def bucle_despachador() -> None:
    from jacobs import delegacion

    while True:
        try:
            await vencer_esperas()
            await delegacion.reconciliar_esperas()
            await despachar()
        except Exception:  # fail-soft: bucle de fondo como el del reaper; nunca tumba LAS MANOS, el error queda en el journal y la próxima vuelta reintenta
            logger.error("Cola: la vuelta del despachador falló", exc_info=True)
        await asyncio.sleep(config_delegacion().cola_cadencia_segundos)
```

- [ ] **Step 3: Esperas, cascada y reanudación en `delegacion.py`**

Agregar a los imports de `jacobs/delegacion.py`:

```python
import asyncio
import json

from jacobs.delegacion_techos import ResumenHijo, acotar, armar_prompt_integracion
from jacobs.models import ESTADOS_TERMINALES
```

Agregar al final de `jacobs/delegacion.py`:

```python
_NO_TERMINALES = (
    PipelineStatus.pending, PipelineStatus.running, PipelineStatus.interrupted,
    PipelineStatus.queued, PipelineStatus.awaiting_approval, PipelineStatus.waiting_children,
)


async def al_terminar_pipeline(pipeline_id: str) -> None:
    """Hook de fin de un pipeline (executor, reaper, cancelación, cola, rechazo).
    Idempotente: se puede llamar de más sin efecto doble (transiciones condicionales)."""
    p = await store.pipeline_get(pipeline_id)
    if p is None or p.status not in ESTADOS_TERMINALES:
        return
    if p.status != PipelineStatus.completed:
        await cortar_descendientes(p.pipeline_id, f"padre_{p.status.value}")
    if p.parent_pipeline_id is not None:
        await _resolver_dependientes(p)
        await reanudar_padre_si_corresponde(p.parent_pipeline_id)
    from jacobs import cola
    await cola.despachar()


async def cortar_descendientes(pipeline_id: str, motivo: str) -> int:
    """Spec §5: padre abortado -> hijos sin terminar a aborted (los que corren
    frenan antes de su próxima ola: executor.run_pipeline relee el estado).
    Recursivo: ningún nieto sigue gastando."""
    cortados = 0
    for h in await store.pipelines_hijos(pipeline_id):
        if h.status not in ESTADOS_TERMINALES and await store.pipeline_transicion(
            h.pipeline_id, _NO_TERMINALES, PipelineStatus.aborted,
        ):
            await store.event_append(h.pipeline_id, "SUBPIPELINE_CANCELADO_POR_PADRE",
                                     {"padre": pipeline_id, "motivo": motivo})
            cortados += 1
        cortados += await cortar_descendientes(h.pipeline_id, motivo)
    return cortados


async def _resolver_dependientes(hijo: Pipeline) -> None:
    """Spec §5: hijo fallado sin skip_on_fail -> sus dependientes no arrancan;
    completado (o fallado con skip_on_fail) -> arrancan si no esperan a nadie más."""
    hermanos = await store.pipelines_hijos(hijo.parent_pipeline_id)
    por_id = {h.pipeline_id: h for h in hermanos}
    for h in hermanos:
        if h.status != PipelineStatus.queued or h.queued_at is not None:
            continue
        deps = h.context.get("delegacion", {}).get("depende_de", [])
        if hijo.pipeline_id not in deps:
            continue
        estados = [por_id.get(d) for d in deps]
        fallidas = [
            d for d in estados
            if d is None or (
                d.status in ESTADOS_TERMINALES and d.status != PipelineStatus.completed
                and not d.context.get("delegacion", {}).get("skip_on_fail", False)
            )
        ]
        if fallidas:
            if await store.pipeline_transicion(h.pipeline_id, (PipelineStatus.queued,), PipelineStatus.aborted):
                await store.event_append(h.pipeline_id, "SUBPIPELINE_DEPENDENCIA_FALLIDA", {
                    "dependencias": {d.pipeline_id: d.status.value for d in fallidas if d is not None},
                })
                await al_terminar_pipeline(h.pipeline_id)
            continue
        if all(d.status in ESTADOS_TERMINALES for d in estados):
            await store.pipeline_fijar_cola([h.pipeline_id], time.time())


async def hay_hijos_sin_terminar(pipeline_id: str) -> bool:
    return any(h.status not in ESTADOS_TERMINALES for h in await store.pipelines_hijos(pipeline_id))


async def pasar_a_espera_de_hijos(pipeline_id: str) -> None:
    """El padre suelta su cupo (waiting_children no cuenta como activo) DESPUÉS de
    que todos sus tokens se consumieron (F exige padre running al consumir)."""
    if await store.pipeline_transicion(pipeline_id, (PipelineStatus.running,), PipelineStatus.waiting_children):
        await store.event_append(pipeline_id, "PIPELINE_ESPERA_HIJOS", {})
    from jacobs import cola
    await cola.despachar()
    await reanudar_padre_si_corresponde(pipeline_id)


async def _resumen_de_hijo(h: Pipeline, largo: int) -> ResumenHijo:
    from jacobs.executor import RefIlegible, _load_ref

    pasos = await store.steps_by_pipeline(h.pipeline_id)
    ref = next((s.output_ref for s in reversed(pasos) if s.output_ref), None)
    if ref:
        try:
            datos = await asyncio.to_thread(_load_ref, ref)
            texto = str(datos.get("result") or datos.get("text") or json.dumps(datos, ensure_ascii=False))
        except RefIlegible as exc:
            texto = f"[resultado ilegible: {exc}]"
    else:
        error = next((s.error for s in reversed(pasos) if s.error), None)
        texto = f"[sin resultado: {error or h.status.value}]"
    d = h.context.get("delegacion", {})
    return ResumenHijo(
        subobjetivo_id=d.get("subobjetivo_id", ""), pipeline_id=h.pipeline_id,
        objetivo=h.context.get("objective", ""), estado=h.status.value,
        resumen=acotar(texto, largo), output_ref=ref,
    )


async def reanudar_padre_si_corresponde(padre_pipeline_id: str) -> bool:
    padre = await store.pipeline_get(padre_pipeline_id)
    if padre is None or padre.status != PipelineStatus.waiting_children:
        return False
    hijos = await store.pipelines_hijos(padre_pipeline_id)
    if any(h.status not in ESTADOS_TERMINALES for h in hijos):
        return False
    largo = config_delegacion().resumen_hijo_caracteres
    por_paso: dict[str, list[ResumenHijo]] = {}
    for h in hijos:
        paso_id = h.context.get("delegacion", {}).get("parent_step")
        por_paso.setdefault(paso_id, []).append(await _resumen_de_hijo(h, largo))
    for step in padre.plan:
        if step.capability != CAPABILITY_INTEGRAR or padre.context.get(f"step_{step.step_index}_ref"):
            continue
        resumenes = por_paso.get(step.input.get("delegacion_step_id"), [])
        step.input = {**step.input, "prompt": armar_prompt_integracion(step.input["objetivo_integracion"], resumenes)}
        await store.step_actualizar_input(step.step_id, step.input)
    await store.pipeline_plan_guardar(padre_pipeline_id, padre.plan)
    if not await store.pipeline_transicion(padre_pipeline_id, (PipelineStatus.waiting_children,),
                                           PipelineStatus.running):
        return False
    await store.event_append(padre_pipeline_id, "DELEGACION_HIJOS_TERMINADOS",
                             {"hijos": {h.pipeline_id: h.status.value for h in hijos}})
    from jacobs import cola, executor
    padre.status = PipelineStatus.running
    cola.lanzar_en_fondo(executor.run_pipeline(padre))
    return True


async def reconciliar_esperas() -> int:
    """Red del bucle despachador: si un hook de fin se perdió (proceso caído entre
    el estado final de un pipeline y su hook), ni el padre queda esperando para
    siempre ni un hijo encolado de un padre ya terminado llega a arrancar."""
    for h in await store.pipelines_en_estado(PipelineStatus.queued):
        if h.parent_pipeline_id is None:
            continue
        estado_padre = await store.pipeline_status(h.parent_pipeline_id)
        if estado_padre in ESTADOS_TERMINALES and estado_padre != PipelineStatus.completed:
            await cortar_descendientes(h.parent_pipeline_id, "reconciliacion")
    reanudados = 0
    for padre in await store.pipelines_en_estado(PipelineStatus.waiting_children):
        for h in await store.pipelines_hijos(padre.pipeline_id):
            if h.status in ESTADOS_TERMINALES:
                await _resolver_dependientes(h)
        if await reanudar_padre_si_corresponde(padre.pipeline_id):
            reanudados += 1
    return reanudados


async def frenar_arbol_por_cruce(raiz: str, cruzados: list[TechoExcedido]) -> int:
    """Spec §5: el árbol cruzó tokens o USD -> lo que aún no arrancó espera
    aprobación; lo que ya corre termina su step."""
    frenados: list[str] = []
    for h in await store.pipelines_del_arbol_en_estado(raiz, (PipelineStatus.queued,)):
        if await store.pipeline_transicion(h.pipeline_id, (PipelineStatus.queued,), PipelineStatus.awaiting_approval):
            frenados.append(h.pipeline_id)
    if not frenados:
        return 0
    techos = [e.a_dict() for e in cruzados]
    # La aprobación guarda QUÉ se frenó: aprobar reencola exactamente esos y
    # rechazar aborta exactamente esos (un padre del árbol que espera SU propia
    # aprobación también está en awaiting_approval y no es de este cruce).
    existente = await store.aprobacion_pendiente_de_raiz(raiz, "cruzado")
    if existente is None:
        await store.aprobacion_crear(raiz, raiz, "", "cruzado", techos, {"frenados": frenados})
    else:
        previos = (existente.get("plan") or {}).get("frenados", [])
        await store.aprobacion_actualizar_plan(existente["id"], techos, {"frenados": sorted(set(previos) | set(frenados))})
    await store.event_append(raiz, "DELEGACION_TECHO_CRUZADO", {"techos": techos, "frenados": frenados})
    if existente is None:
        await _notificar(raiz, (
            f"JAX · El árbol {raiz[:8]} cruzó su techo a mitad de camino ({_texto_techos(cruzados)}). "
            f"{len(frenados)} sub-pipeline(s) sin arrancar esperan aprobación en Admin → Delegaciones."
        ))
    return len(frenados)
```

- [ ] **Step 4: Arranque en LAS MANOS**

En `las_manos/server.py::_jacobs_init`, inmediatamente después de `asyncio.create_task(start_reaper_loop())`:

```python
    # Frente G (2026-09-16): despachador de la cola de sub-pipelines de Ada.
    # Bucle propio (cadencia JAX_ADA_COLA_CADENCIA_SEGUNDOS): el del reaper
    # duerme 300 s y admitiría un hijo 5 minutos tarde.
    from jacobs.cola import bucle_despachador, lanzar_en_fondo
    lanzar_en_fondo(bucle_despachador())
```

- [ ] **Step 5: Verde**

```bash
cd $WT && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_delegacion_politica_puro.py 2>&1 | tail -1
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_delegacion_io_test.py" 2>&1 | tail -2
cd $WT && $PY -m pytest -q policy/tests/test_no_fail_open_except.py tests/test_no_blocking_in_async.py 2>&1 | tail -1
```
Expected: `3 passed`; `42 passed` (31 + 4 de `ColaTest` + 7 de `ArbolTest`); escáneres en verde.

- [ ] **Step 6: Mutaciones y commit**

1. En `cola.despachar`, cambiar `libres = policy.MAX_PARALLEL_PIPELINES - await store.pipeline_count_active()` por `libres = 99`. Correr `-k test_admite_fifo`. Expected: `1 failed` (5 admitidos). Restaurar.
2. En `cortar_descendientes`, borrar la llamada recursiva `cortados += await cortar_descendientes(h.pipeline_id, motivo)`. Correr `-k huerfanos`. Expected: `1 failed` (el nieto sigue `queued`). Restaurar.

Pegar las cuatro salidas en `$L`.

```bash
git -C $WT add jacobs/cola.py jacobs/delegacion.py las_manos/server.py tests/test_delegacion_politica_puro.py jacobs/_delegacion_io_test.py $L
git -C $WT commit -m "feat(jacobs): cola FIFO de sub-pipelines, cruce de techo, dependencias y reanudación del padre

Admisión única bajo el candado de creación y el límite global; kill switch
frena la cola; espera máxima -> expired; cruce a mitad de camino -> lo pendiente
a awaiting_approval + Telegram; padre abortado -> descendientes abortados;
dependencias con skip_on_fail; padre reanudado con la integración y los fallos
declarados; reconciliación por si se pierde un hook. 1 puro + 11 de DB en rojo,
2 guardas por mutación.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---
### Task 9: El executor delega, espera, se detiene y avisa al terminar

**Files:**
- Modify: `jacobs/executor.py` (imports, `_dispatch_step`, funciones nuevas `_plan_de_salida`/`_dispatch_delegate`/`_al_terminar`, `_run_one_step`, `run_pipeline`)
- Modify: `jacobs/routes.py` (`cancel_pipeline`)
- Modify: `jacobs/reaper.py` (`reap_orphaned_pipelines`)
- Modify: `jacobs/plan.py` (`_check_delegacion` y su llamada en `build()`)
- Create: `tests/test_delegacion_executor_puro.py`
- Modify: `jacobs/_delegacion_io_test.py` (clase `EjecucionTest`)

**Interfaces:**
- Consumes: Tasks 3, 7 y 8; B: `correr_con_interruptor` ya envuelve `_dispatch_step` en `_run_one_step`; E-03: `governance["facets"]`.
- Produces:
  - `executor.PlanDelegacionInvalido(RuntimeError)`
  - `executor._plan_de_salida(raw_output: dict) -> tuple[PlanDelegacion | None, str | None]`
  - `executor._dispatch_delegate(step: Step, pipeline: Pipeline) -> tuple[dict, PlanDelegacion]`
  - `executor._al_terminar(pipeline_id: str) -> None`
  - En `pipeline.context`, `f"delegacion_{step_id}"` ∈ {`"lanzada"`, `"espera_aprobacion"`, `"rechazada"`}
  - `run_pipeline`:
    - arranca sólo desde `pending`, `running` o `interrupted`;
    - antes de cada ola relee el estado y se detiene si no es `running`;
    - nunca pisa un estado que escribió otro camino;
    - pasa a `waiting_children` después de una ola que delegó;
    - no corre `integrate` con hijos sin terminar;
    - llama a `_al_terminar` en cada salida terminal.
  - `plan._check_delegacion(steps: list[Step]) -> list[PlanViolation]`: `delegate` sólo con faceta `ada`; `integrate` nunca viene de un plan (lo agrega Jacobs).

- [ ] **Step 1: Tests puros que fallan**

Create `tests/test_delegacion_executor_puro.py`:

```python
"""Frente G (2026-09-16): el step `delegate` en el executor. Spec §5: plan
inválido -> UN reintento con el error explícito; si falla otra vez, step failed
sin hijos. En el camino del Motor Registry el reintento ya lo hace el worker
(worker.py) y Jacobs no reintenta dos veces.

ROJO CONTRA LA BASE: AttributeError _dispatch_delegate / _check_delegacion."""
from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, patch

import pytest

os.environ["JAX_DB_NAME"] = "jax_memory_test"

from jacobs import executor, plan as plan_mod  # noqa: E402
from jacobs.models import Pipeline, Step  # noqa: E402

GOBERNANZA = {
    "capabilities": {"analysis": {"allowed_callers": ["jacobs"], "allowed_motors": [], "max_execution_minutes": 5},
                     "delegate": {"allowed_callers": ["jacobs"], "allowed_motors": [], "max_execution_minutes": 15}},
    "motors": {},
    "facets": frozenset({"ada", "jekyll"}),
}
PLAN_OK = json.dumps({"subobjetivos": [{"id": "a", "objetivo": "x", "faceta": "jekyll", "capability": "analysis",
                                        "depende_de": [], "skip_on_fail": False}],
                      "integracion": {"objetivo": "y"}})
PLAN_CICLO = json.dumps({"subobjetivos": [
    {"id": "a", "objetivo": "x", "faceta": "jekyll", "capability": "analysis", "depende_de": ["b"], "skip_on_fail": False},
    {"id": "b", "objetivo": "x", "faceta": "jekyll", "capability": "analysis", "depende_de": ["a"], "skip_on_fail": False}],
    "integracion": {"objetivo": "y"}})


def _pipeline_y_step(facet="ada"):
    p = Pipeline(pipeline_id="p1", name="n", invoked_by="plataforma", mode="autonomous", root_pipeline_id="p1")
    s = Step(pipeline_id="p1", facet=facet, capability="delegate", input={"prompt": "repartí esto"})
    return p, s


def _correr(salidas, facet="ada"):
    p, s = _pipeline_y_step(facet)
    despacho = AsyncMock(side_effect=[{"result": r} for r in salidas])
    evento = AsyncMock()
    with patch.object(executor, "_dispatch_step", despacho), \
         patch("jacobs.store.get_motor_governance", AsyncMock(return_value=GOBERNANZA)), \
         patch("jacobs.store.event_append", evento):
        try:
            resultado = asyncio.run(executor._dispatch_delegate(s, p))
        except executor.PlanDelegacionInvalido as exc:
            resultado = exc
    return resultado, despacho, evento, s


def test_plan_valido_a_la_primera():
    resultado, despacho, evento, _s = _correr([PLAN_OK])
    raw, plan = resultado
    assert [so.id for so in plan.subobjetivos] == ["a"]
    assert despacho.await_count == 1
    evento.assert_not_awaited()


def test_invalido_una_vez_reintenta_con_el_error_y_restaura_el_prompt():
    prompts = []
    p, s = _pipeline_y_step()

    async def despacho(step, pipeline):
        prompts.append(step.input["prompt"])
        return {"result": "esto no es json" if len(prompts) == 1 else PLAN_OK}

    with patch.object(executor, "_dispatch_step", despacho), \
         patch("jacobs.store.get_motor_governance", AsyncMock(return_value=GOBERNANZA)), \
         patch("jacobs.store.event_append", AsyncMock()) as evento:
        _raw, plan = asyncio.run(executor._dispatch_delegate(s, p))
    assert len(prompts) == 2
    assert "no es un plan_delegacion.v1 válido" in prompts[1] and "no es JSON" in prompts[1]
    assert s.input["prompt"] == "repartí esto"
    assert evento.await_args.args[1] == "DELEGACION_PLAN_INVALIDO"


def test_invalido_dos_veces_falla_declarado():
    resultado, despacho, evento, _s = _correr(["no", "tampoco"])
    assert isinstance(resultado, executor.PlanDelegacionInvalido)
    assert despacho.await_count == 2
    assert [c.args[2]["intento"] for c in evento.await_args_list] == [1, 2]


def test_ciclo_en_depende_de_es_invalido():
    resultado, despacho, _evento, _s = _correr([PLAN_CICLO, PLAN_CICLO])
    assert isinstance(resultado, executor.PlanDelegacionInvalido)
    assert "ciclo" in str(resultado)


def test_faceta_desconocida_es_invalida():
    malo = PLAN_OK.replace('"jekyll"', '"inventada"')
    resultado, _d, _e, _s = _correr([malo, malo])
    assert "tabla `facet`" in str(resultado)


def test_por_el_motor_no_reintenta_dos_veces():
    resultado, despacho, _evento, _s = _correr(["no"], facet="kimi")
    assert isinstance(resultado, executor.PlanDelegacionInvalido)
    assert despacho.await_count == 1


def test_el_prompt_de_delegate_lleva_el_formato():
    from facet_resolver import ResolvedFacet
    p, s = _pipeline_y_step()
    f = ResolvedFacet(key="ada", provider_id="zhipu", base_url="https://x.test", model="glm",
                      credential="k", transport="http_openai_compat", persona=None, params=None)
    invocar = AsyncMock(return_value={"success": True, "result": PLAN_OK, "tokens_in": 1, "tokens_out": 1})
    with patch.object(executor, "validate_capability", AsyncMock(return_value=None)), \
         patch.object(executor, "resolve_facet", AsyncMock(return_value=f)), \
         patch.object(executor, "_invoke_http_openai_compat", invocar), \
         patch.object(executor, "record_direct_usage", AsyncMock(return_value=True)), \
         patch("jacobs.store.get_motor_governance", AsyncMock(return_value=GOBERNANZA)):
        asyncio.run(executor._dispatch_step(s, p))
    prompt = invocar.await_args.args[1]
    assert "plan_delegacion.v1" in prompt and "Facetas activas: ada, jekyll" in prompt


def test_plan_rechaza_delegate_fuera_de_ada_e_integrate_pedido():
    pasos = [Step(step_index=0, facet="jekyll", capability="delegate"),
             Step(step_index=1, facet="ada", capability="integrate"),
             Step(step_index=2, facet="ada", capability="delegate")]
    violaciones = plan_mod._check_delegacion(pasos)
    assert [(v.step_index, v.capability) for v in violaciones] == [(0, "delegate"), (1, "integrate")]
```

```bash
cd $WT && pwd && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_delegacion_executor_puro.py 2>&1 | tee -a $L | tail -3
```
Expected: `8 failed` (`AttributeError`).

- [ ] **Step 2: Tests de base que fallan**

Agregar a `jacobs/_delegacion_io_test.py`, antes de `if __name__ == "__main__":`:

```python
class EjecucionTest(_ConBaseG):
    async def pipeline_de_dos_olas(self) -> Pipeline:
        pid = str(uuid.uuid4())
        ahora = time.time()
        pasos = [Step(pipeline_id=pid, step_index=0, facet="jekyll", capability="analysis", input={"prompt": "a"}),
                 Step(pipeline_id=pid, step_index=1, facet="jekyll", capability="analysis", input={"prompt": "b"},
                      depends_on=[0])]
        p = Pipeline(pipeline_id=pid, name=_MARCA, invoked_by="plataforma", mode="autonomous",
                     status=PipelineStatus.pending, plan=pasos, root_pipeline_id=pid,
                     context={"objective": "o"}, created_at=ahora, updated_at=ahora)
        await store.pipeline_create(p)
        for s in pasos:
            await store.step_upsert(s)
        return p

    async def test_una_cancelacion_a_mitad_de_camino_no_se_pisa(self):
        from jacobs import executor
        p = await self.pipeline_de_dos_olas()
        despachados = []

        async def despacho(step, pipeline):
            despachados.append(step.step_index)
            await store.pipeline_transicion(pipeline.pipeline_id, (PipelineStatus.running,), PipelineStatus.aborted)
            return {"success": True, "result": "hecho"}

        with patch.object(executor, "_dispatch_step", despacho), \
             patch.object(executor, "_persist_step_to_repo", AsyncMock()):
            await executor.run_pipeline(p)
        self.assertEqual(despachados, [0])
        self.assertEqual(await store.pipeline_status(p.pipeline_id), PipelineStatus.aborted)
        self.assertEqual(len(await g.eventos(p.pipeline_id, "PIPELINE_DETENIDO")), 1)

    async def test_no_arranca_un_pipeline_ya_abortado(self):
        from jacobs import executor
        p = await self.pipeline_de_dos_olas()
        await ada.ejecutar("UPDATE jacobs_pipelines SET status='aborted' WHERE pipeline_id=%s", (p.pipeline_id,))
        with patch.object(executor, "_dispatch_step", AsyncMock()) as despacho:
            await executor.run_pipeline(p)
        despacho.assert_not_awaited()
        self.assertEqual(await store.pipeline_status(p.pipeline_id), PipelineStatus.aborted)

    async def test_cancelar_por_api_corta_los_hijos(self):
        from jacobs import delegacion, routes
        p, paso = await self.padre()
        d = await delegacion.delegar(p, paso, g.plan(2))
        with patch("jacobs.executor.run_pipeline", AsyncMock()):
            await routes.cancel_pipeline(p.pipeline_id)
        for hijo in d.hijos:
            self.assertEqual(await store.pipeline_status(hijo), PipelineStatus.aborted)

    async def test_cancelar_un_hijo_encolado_esta_permitido(self):
        from jacobs import delegacion, routes
        p, paso = await self.padre()
        d = await delegacion.delegar(p, paso, g.plan(1))
        with patch("jacobs.executor.run_pipeline", AsyncMock()):
            respuesta = await routes.cancel_pipeline(d.hijos[0])
        self.assertEqual(respuesta["status"], "aborted")

    async def test_el_reaper_corta_los_hijos_del_padre_cosechado(self):
        from jacobs import delegacion, reaper
        p, paso = await self.padre()
        d = await delegacion.delegar(p, paso, g.plan(2))
        await ada.ejecutar("UPDATE jacobs_pipelines SET updated_at=%s WHERE pipeline_id=%s",
                           (time.time() - reaper.RUNNING_STALE_SECONDS - 60, p.pipeline_id))
        with patch("jacobs.executor.run_pipeline", AsyncMock()):
            await reaper.reap_orphaned_pipelines()
        self.assertEqual(await store.pipeline_status(p.pipeline_id), PipelineStatus.expired)
        for hijo in d.hijos:
            self.assertEqual(await store.pipeline_status(hijo), PipelineStatus.aborted)
```

```bash
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_delegacion_io_test.py -k EjecucionTest" 2>&1 | tee -a $L | tail -8
```
Expected: `4 failed, 1 passed`:
- el de cancelación a mitad falla porque el executor viejo pisa el `aborted` con `running`, corre la ola 1 y termina `completed`;
- el de "ya abortado" falla porque despacha igual;
- los dos de cascada (API y reaper) fallan porque los hijos quedan `queued`;
- `test_cancelar_un_hijo_encolado_esta_permitido` **pasa**: es el control de que el 409 nuevo (`ESTADOS_TERMINALES`) no bloquea un hijo `queued`, y se valida por mutación en el Step 6.

- [ ] **Step 3: Executor**

En `jacobs/executor.py`, agregar a los imports:

```python
from jacobs.models import CAPABILITY_DELEGAR, CAPABILITY_INTEGRAR
from motor_registry.esquema_plan_delegacion import (
    PlanDelegacion,
    PlanInvalido,
    errores_de_catalogo,
    instrucciones_de_formato,
    parsear_plan_delegacion,
    texto_de_reintento,
)
```

En `_dispatch_step`, reemplazar `prompt    = _enrich_prompt(ctx_input)` por:

```python
    prompt    = _enrich_prompt(ctx_input)
    if step.capability == CAPABILITY_DELEGAR:
        # Frente G: Ada tiene que responder plan_delegacion.v1; el formato y el
        # vocabulario vivo (facetas activas, capabilities delegables) van en el prompt.
        governance = await store.get_motor_governance()
        prompt += "\n\n" + instrucciones_de_formato(governance["facets"], governance["capabilities"])
```

Después de `_dispatch_step`, agregar:

```python
class PlanDelegacionInvalido(RuntimeError):
    """El step delegate no produjo un plan_delegacion.v1 válido (spec §5)."""


async def _plan_de_salida(raw_output: dict) -> tuple[PlanDelegacion | None, str | None]:
    try:
        plan = parsear_plan_delegacion(str(raw_output.get("result", "")))
    except PlanInvalido as exc:
        return None, str(exc)
    governance = await store.get_motor_governance()
    errores = errores_de_catalogo(plan, governance["facets"], governance["capabilities"])
    if errores:
        return None, "; ".join(errores)
    return plan, None


async def _dispatch_delegate(step: Step, pipeline: Pipeline) -> tuple[dict, PlanDelegacion]:
    """Spec §5: UN reintento con el error explícito; si falla otra vez, el step
    falla sin hijos. Por el Motor Registry el worker ya reintentó con el mismo
    validador (output_validator -> esquema_plan_delegacion): acá no se repite."""
    raw = await _dispatch_step(step, pipeline)
    plan, motivo = await _plan_de_salida(raw)
    if plan is not None:
        return raw, plan
    await store.event_append(pipeline.pipeline_id, "DELEGACION_PLAN_INVALIDO",
                             {"intento": 1, "motivo": motivo}, step.step_id)
    if step.facet in _MOTOR_FACETS:
        raise PlanDelegacionInvalido(f"plan_delegacion.v1 inválido (el Motor Registry ya reintentó): {motivo}")
    original = step.input.get("prompt", "")
    step.input["prompt"] = f"{original}\n\n{texto_de_reintento(motivo)}"
    try:
        raw = await _dispatch_step(step, pipeline)
    finally:
        step.input["prompt"] = original
    plan, motivo = await _plan_de_salida(raw)
    if plan is None:
        await store.event_append(pipeline.pipeline_id, "DELEGACION_PLAN_INVALIDO",
                                 {"intento": 2, "motivo": motivo}, step.step_id)
        raise PlanDelegacionInvalido(f"plan_delegacion.v1 inválido tras un reintento: {motivo}")
    return raw, plan


async def _al_terminar(pipeline_id: str) -> None:
    from jacobs import delegacion
    try:
        await delegacion.al_terminar_pipeline(pipeline_id)
    except Exception:  # fail-soft: el pipeline ya quedó en su estado final; cascada, dependientes y reanudación los repite delegacion.reconciliar_esperas() en cada vuelta del despachador
        logger.error("Jacobs: hook de fin de %s falló", pipeline_id, exc_info=True)
```

En `_run_one_step`, reemplazar el bloque `raw_output = await asyncio.wait_for(...)` (el de B, con `correr_con_interruptor`) por:

```python
    plan_delegacion: PlanDelegacion | None = None
    try:
        if step.capability == CAPABILITY_DELEGAR:
            raw_output, plan_delegacion = await asyncio.wait_for(
                correr_con_interruptor(_dispatch_delegate(step, pipeline)),
                timeout=step.timeout_seconds,
            )
        else:
            raw_output = await asyncio.wait_for(
                # El freno en vuelo (frente B): ver el comentario original de B.
                correr_con_interruptor(_dispatch_step(step, pipeline)),
                timeout=step.timeout_seconds,
            )
```

(El `try:` original de la función pasa a ser este; el resto del cuerpo del `try` queda igual.) Después del bloque `try/except` de `_persist_step_to_repo` y **antes** del `return True`:

```python
        if plan_delegacion is not None:
            # Frente G: el token queda atado a ESTE step (ya completed) y el padre
            # sigue running: es el orden que exige el contrato F (ENMIENDA).
            from jacobs import delegacion
            hecha = await delegacion.delegar(pipeline, step, plan_delegacion)
            pipeline.context[f"delegacion_{step.step_id}"] = hecha.resultado.value
            if hecha.resultado is delegacion.ResultadoDelegacion.RECHAZADA:
                await _fail_step(pipeline, step, i, f"delegación rechazada: {hecha.motivo}")
                return False
```

En `run_pipeline`:

1. En la rama `dry_run`, después de `await store.event_append(pipeline_id, "DRY_RUN_COMPLETE", ...)`, agregar `await _al_terminar(pipeline_id)`.
2. Reemplazar `await store.pipeline_update_status(pipeline_id, PipelineStatus.running, pipeline.current_step_index, pipeline.context)` (el del arranque) por:

```python
    # Frente G: solo arranca desde un estado vivo. Un pipeline cancelado o
    # cosechado mientras esperaba (cola, background task) no revive.
    if not await store.pipeline_transicion(
        pipeline_id, (PipelineStatus.pending, PipelineStatus.running, PipelineStatus.interrupted),
        PipelineStatus.running,
    ):
        estado = await store.pipeline_status(pipeline_id)
        await store.event_append(pipeline_id, "PIPELINE_DETENIDO",
                                 {"estado": estado.value if estado else None, "antes_de": "arrancar"})
        return
    await store.pipeline_guardar_contexto(pipeline_id, pipeline.current_step_index, pipeline.context)
```

3. Al principio del `for wave_num, wave in enumerate(waves):`, antes del chequeo del kill switch:

```python
        # Frente G: una cancelación, el reaper o un rechazo pudieron cambiar el
        # estado mientras corría la ola anterior. Se relee: nunca se pisa.
        estado = await store.pipeline_status(pipeline_id)
        if estado != PipelineStatus.running:
            await store.event_append(pipeline_id, "PIPELINE_DETENIDO",
                                     {"estado": estado.value if estado else None, "wave": wave_num})
            return
```

4. En la rama del kill switch, después de `await store.pipeline_update_status(pipeline_id, PipelineStatus.aborted)`, agregar `await _al_terminar(pipeline_id)`.
5. En el gate de Hyde, reemplazar `await store.pipeline_update_status(pipeline_id, PipelineStatus.interrupted, wave[0], pipeline.context)` por:

```python
            if not await store.pipeline_transicion(pipeline_id, (PipelineStatus.running,), PipelineStatus.interrupted):
                return
            await store.pipeline_guardar_contexto(pipeline_id, wave[0], pipeline.context)
```

6. Después del gate de Hyde y antes de `# ---- EJECUTAR LA OLA EN PARALELO ----`:

```python
        # Frente G: la integración no corre con hijos sin terminar (relanzador,
        # /resume o carrera con el hook): el padre vuelve a esperar.
        if any(pipeline.plan[i].capability == CAPABILITY_INTEGRAR for i in wave):
            from jacobs import delegacion
            if await delegacion.hay_hijos_sin_terminar(pipeline_id):
                await delegacion.pasar_a_espera_de_hijos(pipeline_id)
                return
```

7. Reemplazar `await store.pipeline_update_status(pipeline_id, PipelineStatus.running, next_idx, pipeline.context)` (después del `gather`) por `await store.pipeline_guardar_contexto(pipeline_id, next_idx, pipeline.context)`.
8. En la rama `if failed:`, después del `event_append(... "PIPELINE_ABORTED" ...)`, agregar `await _al_terminar(pipeline_id)`.
9. Después de `WAVE_COMPLETED` y antes de la rama `supervised`:

```python
        # Frente G: una ola que delegó deja al padre esperando a sus hijos (fuera
        # del cupo) o a Fernando (awaiting_approval ya lo escribió delegacion).
        resultados = {pipeline.context.get(f"delegacion_{pipeline.plan[i].step_id}") for i in wave}
        if "espera_aprobacion" in resultados:
            return
        if "lanzada" in resultados:
            from jacobs import delegacion
            await delegacion.pasar_a_espera_de_hijos(pipeline_id)
            return
```

10. En la rama `supervised`, reemplazar `await store.pipeline_update_status(pipeline_id, PipelineStatus.interrupted, next_idx, pipeline.context)` por:

```python
            if not await store.pipeline_transicion(pipeline_id, (PipelineStatus.running,), PipelineStatus.interrupted):
                return
            await store.pipeline_guardar_contexto(pipeline_id, next_idx, pipeline.context)
```

11. Reemplazar el cierre (`# ---- Todas las olas terminaron ----` hasta `PIPELINE_COMPLETED`) por:

```python
    # ---- Todas las olas terminaron ----
    if await store.pipeline_transicion(pipeline_id, (PipelineStatus.running,), PipelineStatus.completed):
        await store.pipeline_guardar_contexto(pipeline_id, len(pipeline.plan), pipeline.context)
        await store.event_append(pipeline_id, "PIPELINE_COMPLETED")
    else:
        estado = await store.pipeline_status(pipeline_id)
        await store.event_append(pipeline_id, "PIPELINE_DETENIDO",
                                 {"estado": estado.value if estado else None, "antes_de": "completar"})
    await _al_terminar(pipeline_id)
```

- [ ] **Step 4: Cancelación, reaper y plan**

`jacobs/routes.py::cancel_pipeline`, reemplazar desde el `if pipeline.status in (...)` hasta el `return` por:

```python
    if pipeline.status in ESTADOS_TERMINALES:
        raise HTTPException(
            status_code=409,
            detail=f"Pipeline ya finalizado con status '{pipeline.status.value}'",
        )
    await store.pipeline_update_status(pipeline_id, PipelineStatus.aborted)
    await store.event_append(pipeline_id, "PIPELINE_CANCELLED", {"by": "API request"})
    # Frente G: la cancelación corta el árbol (spec §5, sin huérfanos) y avisa al padre.
    from jacobs import delegacion
    await delegacion.al_terminar_pipeline(pipeline_id)
    return {"pipeline_id": pipeline_id, "status": "aborted"}
```

y agregar `ESTADOS_TERMINALES` al import de `jacobs.models` en `routes.py`.

`jacobs/reaper.py::reap_orphaned_pipelines`, dentro del `try:` que cosecha, después del `event_append(... "REAPED" ...)`:

```python
            # Frente G: un padre cosechado corta su árbol; un hijo cosechado avisa
            # a su padre (dependientes e integración con el fallo declarado).
            from jacobs import delegacion
            await delegacion.al_terminar_pipeline(p.pipeline_id)
```

`jacobs/plan.py`, después de `_check_facets` (E-17):

```python
def _check_delegacion(steps: list) -> list[PlanViolation]:
    """Frente G (2026-09-16): `delegate` solo lo corre ada (el contrato F exige
    un step de ada para emitir tokens) e `integrate` nunca viene de un plan: lo
    agrega Jacobs al padre cuando delegó (jacobs/delegacion.py)."""
    from jacobs.models import CAPABILITY_DELEGAR, CAPABILITY_INTEGRAR
    violaciones: list[PlanViolation] = []
    for s in steps:
        if s.capability == CAPABILITY_DELEGAR and s.facet != "ada":
            violaciones.append(PlanViolation(s.step_index, s.facet, s.motor, s.capability,
                                             "solo la faceta ada delega (capability 'delegate')"))
        elif s.capability == CAPABILITY_INTEGRAR:
            violaciones.append(PlanViolation(s.step_index, s.facet, s.motor, s.capability,
                                             "'integrate' no se pide en un plan: la agrega Jacobs al delegar"))
    return violaciones
```

En `build()`, justo después del bloque de `facet_violations` (E-17):

```python
        delegacion_violations = _check_delegacion(steps)
        if delegacion_violations:
            raise PlanRejected(delegacion_violations)
```

- [ ] **Step 5: Verde y regresión amplia**

```bash
cd $WT && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_delegacion_executor_puro.py tests/test_jacobs_director.py tests/test_jacobs_interruptor.py tests/test_plan_facetas_de_la_tabla.py tests/test_jacobs_ref_ilegible_no_es_exito.py tests/test_invoke_motor_cancel_on_timeout.py 2>&1 | tail -2
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_delegacion_io_test.py tests/test_subpipeline_contrato_rutas.py jacobs/_pipeline_identity_test.py jacobs/_direct_usage_test.py jacobs/_http_facet_admission_test.py" 2>&1 | tail -3
cd $WT && $PY -m pytest -q policy/tests/test_no_fail_open_except.py tests/test_no_blocking_in_async.py 2>&1 | tail -1
```
Expected: todo `passed`. En `_delegacion_io_test.py` son 47: 42 más los 5 de `EjecucionTest`. Si un test existente que parcheaba `store.pipeline_update_status` espera la llamada `running` del arranque, se ajusta al contrato nuevo (`pipeline_transicion` + `pipeline_guardar_contexto`) en este commit, con una nota en `$L`.

- [ ] **Step 6: Mutación y commit**

1. En `run_pipeline`, borrar el bloque del punto 3 (la relectura del estado antes de cada ola). Correr `-k test_una_cancelacion_a_mitad_de_camino_no_se_pisa`. Expected: `1 failed` (`despachados == [0, 1]`). Restaurar.
2. En `cancel_pipeline`, cambiar `if pipeline.status in ESTADOS_TERMINALES:` por `if pipeline.status not in (PipelineStatus.pending, PipelineStatus.running, PipelineStatus.interrupted):`. Correr `-k test_cancelar_un_hijo_encolado_esta_permitido`. Expected: `1 failed` (409). Restaurar.

Pegar las cuatro salidas en `$L`.

```bash
git -C $WT add jacobs/executor.py jacobs/routes.py jacobs/reaper.py jacobs/plan.py tests/test_delegacion_executor_puro.py jacobs/_delegacion_io_test.py $L
git -C $WT commit -m "feat(jacobs): el executor delega, espera a los hijos y no pisa una cancelación

Step delegate con plan_delegacion.v1 estricto y UN reintento (no doble por el
motor); delegar al completar; ola que delegó -> waiting_children o
awaiting_approval; integrate no corre con hijos pendientes; transiciones
condicionales al arrancar, entre olas y al completar; hook de fin en cada
salida terminal; cancelar y el reaper cortan el árbol; plan: delegate solo ada,
integrate nunca pedido. 8 puros + 4 de DB en rojo (1 control); relectura y 409
validados por mutación.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---
### Task 10: Aprobar o rechazar una delegación (Jacobs)

**Files:**
- Modify: `jacobs/delegacion.py` (al final: `aprobar`, `rechazar`, excepciones)
- Modify: `jacobs/routes.py` (modelos y tres rutas `/jacobs/delegaciones/*`)
- Modify: `jacobs/_delegacion_io_test.py` (clase `AprobacionTest`)

**Interfaces:**
- Consumes: Tasks 4, 6, 7 y 8; `policy.validate_resume` (sólo `plataforma`).
- Produces:
  - `delegacion.AprobacionNoEncontrada(LookupError)`, `delegacion.AprobacionYaResuelta(RuntimeError)`, `delegacion.TechoInsuficiente(ValueError)` con `.faltan: list[dict]`
  - `delegacion.aprobar(aprobacion_id: int, user_id: str, techos: dict, *, aceptar_consumo_incompleto: bool = False) -> dict`. `techos` tiene las claves `techo_tokens_aprobado`, `techo_usd_aprobado`, `techo_pipelines_aprobado` y `techo_hijos_por_step_aprobado`, y vale `None` = no se aprueba ese techo.
  - `delegacion.rechazar(aprobacion_id: int, user_id: str) -> dict`
  - Rutas: `GET /jacobs/delegaciones/pendientes` → `{"aprobaciones": [{id, root_pipeline_id, pipeline_id, step_id, tipo, techos, creada_at, pipeline_name}]}` (hasta `LIMITE_APROBACIONES_LISTADAS = 100`)
  - `POST /jacobs/delegaciones/{aprobacion_id}/aprobar` body `AprobarDelegacionRequest(invoked_by: str, user_id: str, techo_tokens: int | None, techo_usd: float | None, techo_pipelines: int | None, techo_hijos_por_step: int | None, aceptar_consumo_incompleto: bool = False)`
  - `POST /jacobs/delegaciones/{aprobacion_id}/rechazar` body `RechazarDelegacionRequest(invoked_by: str, user_id: str)`
  - Códigos de error, siempre con `detail` estable:
    - 403: `invoked_by` no autorizado;
    - 404: `aprobacion_no_encontrada`;
    - 409: `aprobacion_ya_resuelta`;
    - 422: `{"code": "techo_insuficiente", "faltan": [...]}`;
    - 423: `kill_switch_activo`, sólo al aprobar.

- [ ] **Step 1: Tests que fallan**

Agregar a `jacobs/_delegacion_io_test.py`, antes de `if __name__ == "__main__":`:

```python
class AprobacionTest(_ConColaG):
    async def excede(self, n=3):
        from jacobs import delegacion
        os.environ["JAX_ADA_MAX_HIJOS_POR_STEP"] = "2"
        p, paso = await self.padre()
        d = await delegacion.delegar(p, paso, g.plan(n))
        self.assertEqual(d.resultado, delegacion.ResultadoDelegacion.ESPERA_APROBACION)
        return p, paso, d.aprobacion_id

    async def test_aprobar_lanza_con_el_techo_de_ese_arbol(self):
        from jacobs import delegacion
        p, paso, aid = await self.excede()
        r = await delegacion.aprobar(aid, "7", {"techo_hijos_por_step_aprobado": 3})
        self.assertEqual((r["resultado"], len(r["hijos"])), ("lanzada", 3))
        self.assertEqual(await g.tokens_emitidos(p.pipeline_id), 3)
        self.assertEqual(await store.pipeline_status(p.pipeline_id), PipelineStatus.waiting_children)
        self.assertEqual((await store.arbol_consumo_leer(p.pipeline_id))["techo_hijos_por_step_aprobado"], 3)
        fila = await store.aprobacion_obtener(aid)
        self.assertEqual((fila["estado"], fila["resuelta_por"]), ("aprobada", "7"))
        self.assertIsNotNone(fila["resuelta_at"])
        por = [e["payload"]["por"] for e in await g.eventos(p.pipeline_id, "DELEGACION_APROBADA")]
        self.assertIn("fernando", por)

    async def test_otro_arbol_no_hereda_el_techo_aprobado(self):
        from jacobs import delegacion
        _p, _paso, aid = await self.excede()
        await delegacion.aprobar(aid, "7", {"techo_hijos_por_step_aprobado": 3})
        otro, paso_otro = await self.padre()
        d = await delegacion.delegar(otro, paso_otro, g.plan(3))
        self.assertEqual(d.resultado, delegacion.ResultadoDelegacion.ESPERA_APROBACION)

    async def test_techo_insuficiente_no_resuelve_ni_lanza(self):
        from jacobs import delegacion
        p, _paso, aid = await self.excede()
        with self.assertRaises(delegacion.TechoInsuficiente) as exc:
            await delegacion.aprobar(aid, "7", {"techo_hijos_por_step_aprobado": 2})
        self.assertEqual(exc.exception.faltan[0]["techo"], "hijos_por_step")
        self.assertEqual((await store.aprobacion_obtener(aid))["estado"], "pendiente")
        self.assertEqual(await g.tokens_emitidos(p.pipeline_id), 0)

    async def test_no_se_aprueba_dos_veces(self):
        from jacobs import delegacion
        _p, _paso, aid = await self.excede()
        await delegacion.aprobar(aid, "7", {"techo_hijos_por_step_aprobado": 3})
        with self.assertRaises(delegacion.AprobacionYaResuelta):
            await delegacion.aprobar(aid, "7", {"techo_hijos_por_step_aprobado": 3})

    async def test_rechazar_aborta_al_padre_sin_tokens(self):
        from jacobs import delegacion
        p, _paso, aid = await self.excede()
        await delegacion.rechazar(aid, "7")
        self.assertEqual(await store.pipeline_status(p.pipeline_id), PipelineStatus.aborted)
        self.assertEqual((await store.aprobacion_obtener(aid))["estado"], "rechazada")
        self.assertEqual(await g.tokens_emitidos(p.pipeline_id), 0)
        self.assertEqual(len(await g.eventos(p.pipeline_id, "DELEGACION_RECHAZADA")), 1)

    async def cruzado(self):
        from jacobs import cola, delegacion
        p, paso = await self.padre()
        d = await delegacion.delegar(p, paso, g.plan(2))
        await store.pipeline_transicion(p.pipeline_id, (PipelineStatus.running,), PipelineStatus.waiting_children)
        conn = await store.get_conn()
        try:
            async with conn.cursor() as cur:
                await store.arbol_consumo_sumar_en(cur, p.pipeline_id, 2_000_001, 0.1, time.time())
        finally:
            conn.close()
        await cola.despachar()
        aprobacion = await store.aprobacion_pendiente_de_raiz(p.pipeline_id, "cruzado")
        return p, d, aprobacion["id"]

    async def test_aprobar_un_cruce_reencola_lo_frenado(self):
        from jacobs import delegacion
        p, d, aid = await self.cruzado()
        r = await delegacion.aprobar(aid, "7", {"techo_tokens_aprobado": 3_000_000})
        self.assertEqual((r["tipo"], r["reencolados"]), ("cruzado", 2))
        for hijo in d.hijos:
            self.assertIn(await store.pipeline_status(hijo), (PipelineStatus.queued, PipelineStatus.pending))
        self.assertEqual(self.run_pipeline.call_count, 2)

    async def test_rechazar_un_cruce_aborta_lo_frenado_y_reanuda_la_integracion(self):
        from jacobs import delegacion
        p, d, aid = await self.cruzado()
        await delegacion.rechazar(aid, "7")
        for hijo in d.hijos:
            self.assertEqual(await store.pipeline_status(hijo), PipelineStatus.aborted)
        self.assertEqual(await store.pipeline_status(p.pipeline_id), PipelineStatus.running)

    async def test_rutas(self):
        from fastapi import HTTPException
        from jacobs import routes
        p, _paso, aid = await self.excede()
        listado = await routes.delegaciones_pendientes()
        fila = next(a for a in listado["aprobaciones"] if a["id"] == aid)
        self.assertEqual(fila["pipeline_name"], "arnes-g-padre")
        with self.assertRaises(HTTPException) as e403:
            await routes.aprobar_delegacion(aid, routes.AprobarDelegacionRequest(invoked_by="ada", user_id="7"))
        self.assertEqual(e403.exception.status_code, 403)
        with patch("jacobs.routes.check_kill_switch", return_value=True), self.assertRaises(HTTPException) as e423:
            await routes.aprobar_delegacion(aid, routes.AprobarDelegacionRequest(invoked_by="plataforma", user_id="7"))
        self.assertEqual((e423.exception.status_code, e423.exception.detail), (423, "kill_switch_activo"))
        with self.assertRaises(HTTPException) as e422:
            await routes.aprobar_delegacion(aid, routes.AprobarDelegacionRequest(invoked_by="plataforma", user_id="7"))
        self.assertEqual((e422.exception.status_code, e422.exception.detail["code"]), (422, "techo_insuficiente"))
        with self.assertRaises(HTTPException) as e404:
            await routes.rechazar_delegacion(10**12, routes.RechazarDelegacionRequest(invoked_by="plataforma", user_id="7"))
        self.assertEqual((e404.exception.status_code, e404.exception.detail), (404, "aprobacion_no_encontrada"))
        ok = await routes.rechazar_delegacion(aid, routes.RechazarDelegacionRequest(invoked_by="plataforma", user_id="7"))
        self.assertEqual(ok["estado"], "rechazada")
        with self.assertRaises(HTTPException) as e409:
            await routes.rechazar_delegacion(aid, routes.RechazarDelegacionRequest(invoked_by="plataforma", user_id="7"))
        self.assertEqual((e409.exception.status_code, e409.exception.detail), (409, "aprobacion_ya_resuelta"))
```

```bash
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_delegacion_io_test.py -k AprobacionTest" 2>&1 | tee -a $L | tail -10
```
Expected: `8 failed` (`AttributeError: module 'jacobs.delegacion' has no attribute 'aprobar'` / `routes` sin `delegaciones_pendientes`).

- [ ] **Step 2: `aprobar` y `rechazar`**

Agregar a los imports de `jacobs/delegacion.py`: `from jacobs.delegacion_techos import techos_no_cubiertos` y `from motor_registry.esquema_plan_delegacion import parsear_plan_delegacion`. Agregar al final del archivo:

```python
class AprobacionNoEncontrada(LookupError):
    pass


class AprobacionYaResuelta(RuntimeError):
    pass


class TechoInsuficiente(ValueError):
    def __init__(self, faltan: list[dict]) -> None:
        super().__init__("techo_insuficiente")
        self.faltan = faltan


_CAMPOS_TECHO = ("techo_tokens_aprobado", "techo_usd_aprobado", "techo_pipelines_aprobado",
                 "techo_hijos_por_step_aprobado")


async def _pendiente(aprobacion_id: int) -> dict:
    fila = await store.aprobacion_obtener(aprobacion_id)
    if fila is None:
        raise AprobacionNoEncontrada(aprobacion_id)
    if fila["estado"] != "pendiente":
        raise AprobacionYaResuelta(fila["estado"])
    return fila


async def aprobar(
    aprobacion_id: int, user_id: str, techos: dict, *, aceptar_consumo_incompleto: bool = False,
) -> dict:
    """Spec §5: aprobar fija techo_*_aprobado SOLO para ese árbol, auditado con
    usuario y hora (jacobs_delegacion_aprobaciones.resuelta_por/resuelta_at).
    Una aprobación que no cubre lo proyectado no se acepta (volvería a pedirse)."""
    fila = await _pendiente(aprobacion_id)
    faltan = techos_no_cubiertos(fila["techos"], techos, aceptar_consumo_incompleto)
    if faltan:
        raise TechoInsuficiente(faltan)
    fijados = {c: techos[c] for c in _CAMPOS_TECHO if techos.get(c) is not None}
    if not await store.aprobacion_resolver(
        aprobacion_id, "aprobada", user_id,
        {**fijados, "aceptar_consumo_incompleto": aceptar_consumo_incompleto}, time.time(),
    ):
        raise AprobacionYaResuelta("carrera")
    raiz = fila["root_pipeline_id"]
    await store.arbol_fijar_techos(raiz, fijados)
    if aceptar_consumo_incompleto:
        await store.arbol_limpiar_consumo_incompleto(raiz)
    await store.event_append(fila["pipeline_id"], "DELEGACION_APROBADA", {
        "por": "fernando", "user_id": user_id, "aprobacion_id": aprobacion_id, "techos": fijados,
    }, fila["step_id"] or None)

    if fila["tipo"] == "cruzado":
        reencolados = 0
        for hijo_id in (fila.get("plan") or {}).get("frenados", []):
            if await store.pipeline_reencolar(hijo_id, time.time()):
                reencolados += 1
        from jacobs import cola
        await cola.despachar()
        return {"estado": "aprobada", "tipo": "cruzado", "reencolados": reencolados}

    padre = await store.pipeline_get(fila["pipeline_id"])
    paso = next((s for s in padre.plan if s.step_id == fila["step_id"]), None) if padre else None
    if paso is None or not await store.pipeline_transicion(
        fila["pipeline_id"], (PipelineStatus.awaiting_approval,), PipelineStatus.running,
    ):
        return {"estado": "aprobada", "tipo": "excede", "resultado": "no_lanzada", "motivo": "padre_no_espera"}
    padre.status = PipelineStatus.running
    plan = parsear_plan_delegacion(json.dumps(fila["plan"]))
    hecha = await delegar(padre, paso, plan, por="fernando")
    if hecha.resultado is ResultadoDelegacion.LANZADA:
        await pasar_a_espera_de_hijos(padre.pipeline_id)
    elif hecha.resultado is ResultadoDelegacion.RECHAZADA:
        if await store.pipeline_transicion(padre.pipeline_id, (PipelineStatus.running,), PipelineStatus.aborted):
            await store.event_append(padre.pipeline_id, "DELEGACION_FALLIDA",
                                     {"motivo": hecha.motivo, "tras": "aprobacion"}, paso.step_id)
            await al_terminar_pipeline(padre.pipeline_id)
    return {"estado": "aprobada", "tipo": "excede", "resultado": hecha.resultado.value, "hijos": list(hecha.hijos)}


async def rechazar(aprobacion_id: int, user_id: str) -> dict:
    """Fernando dice que no: lo que esperaba la aprobación se aborta (nunca
    queda esperando para siempre) y el árbol sigue con el fallo declarado."""
    fila = await _pendiente(aprobacion_id)
    if not await store.aprobacion_resolver(aprobacion_id, "rechazada", user_id, None, time.time()):
        raise AprobacionYaResuelta("carrera")
    await store.event_append(fila["pipeline_id"], "DELEGACION_RECHAZADA",
                             {"por": "fernando", "user_id": user_id, "aprobacion_id": aprobacion_id},
                             fila["step_id"] or None)
    if fila["tipo"] == "cruzado":
        abortados = 0
        for hijo_id in (fila.get("plan") or {}).get("frenados", []):
            if await store.pipeline_transicion(hijo_id, (PipelineStatus.awaiting_approval,), PipelineStatus.aborted):
                await store.event_append(hijo_id, "SUBPIPELINE_CANCELADO_POR_PADRE",
                                         {"padre": fila["root_pipeline_id"], "motivo": "aprobacion_rechazada"})
                abortados += 1
                await al_terminar_pipeline(hijo_id)
        return {"estado": "rechazada", "tipo": "cruzado", "abortados": abortados}
    if await store.pipeline_transicion(fila["pipeline_id"], (PipelineStatus.awaiting_approval,), PipelineStatus.aborted):
        await al_terminar_pipeline(fila["pipeline_id"])
    return {"estado": "rechazada", "tipo": "excede"}
```

- [ ] **Step 3: Rutas**

En `jacobs/routes.py`, agregar `Field` al import de pydantic (`from pydantic import BaseModel, Field`) y al final del archivo:

```python
# ----------------------------------------------------------------
#  Delegaciones de Ada que esperan a Fernando (frente G, 2026-09-16)
# ----------------------------------------------------------------
# Las pide jax-platform (/api/admin/delegaciones, superadmin) en nombre del
# usuario. Mismo límite de confianza que resume/approve-step: solo 'plataforma'.

LIMITE_APROBACIONES_LISTADAS = 100


class AprobarDelegacionRequest(BaseModel):
    invoked_by: str
    user_id: str = Field(min_length=1, max_length=50)
    techo_tokens: int | None = Field(default=None, ge=1)
    techo_usd: float | None = Field(default=None, gt=0)
    techo_pipelines: int | None = Field(default=None, ge=1)
    techo_hijos_por_step: int | None = Field(default=None, ge=1)
    aceptar_consumo_incompleto: bool = False


class RechazarDelegacionRequest(BaseModel):
    invoked_by: str
    user_id: str = Field(min_length=1, max_length=50)


@router.get("/delegaciones/pendientes")
async def delegaciones_pendientes() -> dict:
    return {"aprobaciones": await store.aprobaciones_pendientes(LIMITE_APROBACIONES_LISTADAS)}


@router.post("/delegaciones/{aprobacion_id}/aprobar")
async def aprobar_delegacion(aprobacion_id: int, req: AprobarDelegacionRequest) -> dict:
    politica = validate_resume(req.invoked_by)
    if not politica.ok:
        raise HTTPException(status_code=403, detail=politica.reason)
    if check_kill_switch():
        raise HTTPException(status_code=423, detail="kill_switch_activo")
    from jacobs import delegacion
    try:
        return await delegacion.aprobar(aprobacion_id, req.user_id, {
            "techo_tokens_aprobado": req.techo_tokens,
            "techo_usd_aprobado": req.techo_usd,
            "techo_pipelines_aprobado": req.techo_pipelines,
            "techo_hijos_por_step_aprobado": req.techo_hijos_por_step,
        }, aceptar_consumo_incompleto=req.aceptar_consumo_incompleto)
    except delegacion.AprobacionNoEncontrada:
        raise HTTPException(status_code=404, detail="aprobacion_no_encontrada")
    except delegacion.AprobacionYaResuelta:
        raise HTTPException(status_code=409, detail="aprobacion_ya_resuelta")
    except delegacion.TechoInsuficiente as exc:
        raise HTTPException(status_code=422, detail={"code": "techo_insuficiente", "faltan": exc.faltan})


@router.post("/delegaciones/{aprobacion_id}/rechazar")
async def rechazar_delegacion(aprobacion_id: int, req: RechazarDelegacionRequest) -> dict:
    politica = validate_resume(req.invoked_by)
    if not politica.ok:
        raise HTTPException(status_code=403, detail=politica.reason)
    from jacobs import delegacion
    try:
        return await delegacion.rechazar(aprobacion_id, req.user_id)
    except delegacion.AprobacionNoEncontrada:
        raise HTTPException(status_code=404, detail="aprobacion_no_encontrada")
    except delegacion.AprobacionYaResuelta:
        raise HTTPException(status_code=409, detail="aprobacion_ya_resuelta")
```

- [ ] **Step 4: Verde y commit**

```bash
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_delegacion_io_test.py tests/test_subpipeline_contrato_rutas.py" 2>&1 | tail -2
cd $WT && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_subpipeline_contrato_puro.py -k ruta_http 2>&1 | tail -1
cd $WT && $PY -m pytest -q policy/tests/test_no_fail_open_except.py 2>&1 | tail -1
```
Expected: `55 passed` en `_delegacion_io_test.py` (47 + 8) y los de F en verde. El test de F que prohíbe rutas con `token`/`subpipeline` en el path sigue verde, porque las rutas nuevas son `delegaciones`.

```bash
git -C $WT add jacobs/delegacion.py jacobs/routes.py jacobs/_delegacion_io_test.py $L
git -C $WT commit -m "feat(jacobs): aprobar o rechazar una delegación de Ada, techo por árbol auditado

Aprobar fija techo_*_aprobado solo para ese árbol (usuario y hora en
jacobs_delegacion_aprobaciones); exceso -> relanza la delegación; cruce ->
reencola exactamente lo frenado. Rechazar aborta lo que esperaba y el árbol
sigue con el fallo declarado. Rutas /jacobs/delegaciones/* solo para
plataforma; 423 con kill switch al aprobar. 8 de DB vistos en rojo.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---
### Task 11: jax-platform · Admin → Delegaciones (proxy de superadmin y pantalla)

**Files:**
- Create: `$WTP/backend/api/admin/delegaciones.py`
- Modify: `$WTP/backend/main.py` (import e `include_router`)
- Create: `$WTP/backend/tests/test_admin_delegaciones.py`
- Create: `$WTP/frontend/src/pages/admin/AdminDelegaciones.jsx`
- Create: `$WTP/frontend/src/pages/admin/AdminDelegaciones.test.jsx`
- Modify: `$WTP/frontend/src/pages/Admin.jsx`, `$WTP/frontend/src/components/admin/AdminSidebar.jsx`
- Modify: `$WTP/frontend/src/i18n/es.js`, `$WTP/frontend/src/i18n/en.js`

**Interfaces:**
- Consumes: rutas de Jacobs de la Task 10; B: `kill_switch.exigir_mesa_libre`, `interruptor.escribir_pausa/borrar_pausa/ruta_del_interruptor`, `api/errores.js::textoDeKillSwitch(t, err)`; `api/pipelines.py::JACOBS_URL`, `INVOKED_BY_PLATAFORMA`; `http_client.get_http_client`; `auth.middleware.require_superadmin`.
- Produces:
  - `GET /api/admin/delegaciones` → el JSON de Jacobs.
  - `POST /api/admin/delegaciones/{aprobacion_id}/aprobar` con body `{techo_tokens?, techo_usd?, techo_pipelines?, techo_hijos_por_step?, aceptar_consumo_incompleto?}` (extra prohibido). Agrega `invoked_by="plataforma"` y el `user_id` del token, y responde 423 con el freno puesto.
  - `POST /api/admin/delegaciones/{aprobacion_id}/rechazar`.
  - Errores: el `status` y el `detail` de Jacobs se propagan tal cual; si Jacobs no responde, 502 `jacobs_no_disponible`.
  - Frontend: la ruta `/admin/delegaciones` y la función exportada `valoresIniciales(techos) -> {techo_*: string}`.
  - Claves i18n (es/en):
    - `adminDelegaciones`, `delegacionesTitulo`, `delegacionesVacio`, `delegacionesPipeline`, `delegacionesTipo`, `delegacionesTechos`, `delegacionesCreada`;
    - `delegacionesTipos: {excede, cruzado}`, `delegacionesTechoNombre: {tokens_arbol, usd_arbol, pipelines_arbol, hijos_por_step, consumo_incompleto}`;
    - `delegacionesProyectadoSobreLimite(proyectado, limite)`, `delegacionesAprobar`, `delegacionesRechazar`, `delegacionesAprobarTitulo`, `delegacionesAprobarAyuda`, `delegacionesAceptarIncompleto`;
    - `delegacionesRechazarTitulo`, `delegacionesRechazarMensaje(nombre)`, `delegacionesAprobada`, `delegacionesRechazada`, `delegacionesErrorCargar`;
    - `adminErrors.{techo_insuficiente, aprobacion_ya_resuelta, aprobacion_no_encontrada, jacobs_no_disponible, jacobs_error}`;
    - `pipelineStatusLabels.{queued, awaiting_approval, waiting_children}`.

- [ ] **Step 1: Test de backend que falla**

Create `$WTP/backend/tests/test_admin_delegaciones.py`:

```python
"""Frente G (2026-09-16): Admin -> Delegaciones. Proxy de superadmin a las
rutas de Jacobs (/jacobs/delegaciones/*). La identidad y el rol los pone este
backend; el cliente no puede mandar invoked_by ni user_id (extra prohibido).
Aprobar ejecuta (lanza sub-pipelines): con el freno puesto, 423."""
from __future__ import annotations

from unittest.mock import AsyncMock

import httpx

from api.admin import delegaciones
from auth.jwt import create_access_token
from tests.identidades import cabeceras

TENANT = "test-admin-delegaciones"


def _superadmin():
    return {"Authorization": f"Bearer {create_access_token('1', TENANT, 'superadmin')}"}


class _Resp:
    def __init__(self, status, datos):
        self.status_code = status
        self._datos = datos
        self.content = b"{}"

    def json(self):
        return self._datos


class _Cliente:
    def __init__(self, resp=None, exc=None):
        self.llamadas = []
        self.resp, self.exc = resp, exc

    async def _llamar(self, metodo, url, **kw):
        self.llamadas.append((metodo, url, kw))
        if self.exc:
            raise self.exc
        return self.resp

    async def get(self, url, **kw):
        return await self._llamar("GET", url, **kw)

    async def post(self, url, **kw):
        return await self._llamar("POST", url, **kw)


def _con(monkeypatch, cliente):
    monkeypatch.setattr(delegaciones, "get_http_client", AsyncMock(return_value=cliente))
    return cliente


def test_listar_reenvia_a_jacobs(client, monkeypatch):
    cliente = _con(monkeypatch, _Cliente(_Resp(200, {"aprobaciones": [{"id": 1}]})))
    r = client.get("/api/admin/delegaciones", headers=_superadmin())
    assert r.status_code == 200, r.text
    assert r.json() == {"aprobaciones": [{"id": 1}]}
    assert cliente.llamadas[0][1].endswith("/delegaciones/pendientes")


def test_solo_superadmin(client, monkeypatch):
    cliente = _con(monkeypatch, _Cliente(_Resp(200, {})))
    r = client.get("/api/admin/delegaciones", headers=cabeceras(client, "delegaciones", "operator", TENANT))
    assert r.status_code == 403
    assert cliente.llamadas == []


def test_aprobar_pone_rol_e_identidad_del_servidor(client, monkeypatch):
    cliente = _con(monkeypatch, _Cliente(_Resp(200, {"estado": "aprobada"})))
    r = client.post("/api/admin/delegaciones/5/aprobar", headers=_superadmin(),
                    json={"techo_hijos_por_step": 3, "aceptar_consumo_incompleto": False})
    assert r.status_code == 200, r.text
    metodo, url, kw = cliente.llamadas[0]
    assert (metodo, url.endswith("/delegaciones/5/aprobar")) == ("POST", True)
    assert kw["json"]["invoked_by"] == "plataforma"
    assert kw["json"]["user_id"] == "1"
    assert kw["json"]["techo_hijos_por_step"] == 3


def test_el_cliente_no_puede_mandar_invoked_by(client, monkeypatch):
    cliente = _con(monkeypatch, _Cliente(_Resp(200, {})))
    r = client.post("/api/admin/delegaciones/5/aprobar", headers=_superadmin(),
                    json={"invoked_by": "ada", "user_id": "99"})
    assert r.status_code == 422
    assert cliente.llamadas == []


def test_aprobar_con_el_freno_puesto_es_423_sin_llamar_a_jacobs(client, monkeypatch):
    from interruptor import borrar_pausa, escribir_pausa, ruta_del_interruptor
    cliente = _con(monkeypatch, _Cliente(_Resp(200, {})))
    ruta = ruta_del_interruptor()
    escribir_pausa(ruta, '{"accion": "test-admin-delegaciones"}')
    try:
        r = client.post("/api/admin/delegaciones/5/aprobar", headers=_superadmin(), json={})
    finally:
        borrar_pausa(ruta)
    assert r.status_code == 423
    assert r.json()["detail"] == "kill_switch_activo"
    assert cliente.llamadas == []


def test_el_error_de_jacobs_se_propaga_con_su_codigo(client, monkeypatch):
    detalle = {"code": "techo_insuficiente", "faltan": [{"techo": "hijos_por_step"}]}
    _con(monkeypatch, _Cliente(_Resp(422, {"detail": detalle})))
    r = client.post("/api/admin/delegaciones/5/aprobar", headers=_superadmin(), json={})
    assert r.status_code == 422
    assert r.json()["detail"] == detalle


def test_jacobs_caido_es_502_con_codigo(client, monkeypatch):
    _con(monkeypatch, _Cliente(exc=httpx.ConnectError("sin LAS MANOS")))
    r = client.post("/api/admin/delegaciones/5/rechazar", headers=_superadmin())
    assert r.status_code == 502
    assert r.json()["detail"] == "jacobs_no_disponible"


def test_rechazar_reenvia(client, monkeypatch):
    cliente = _con(monkeypatch, _Cliente(_Resp(200, {"estado": "rechazada"})))
    r = client.post("/api/admin/delegaciones/9/rechazar", headers=_superadmin())
    assert r.status_code == 200, r.text
    assert cliente.llamadas[0][2]["json"] == {"invoked_by": "plataforma", "user_id": "1"}
```

```bash
cd $WTP/backend && pwd && JAX_REPO_PATH=$WT $PYP -m pytest -q tests/test_admin_delegaciones.py 2>&1 | tee -a $L | tail -3
```
Expected: `ImportError: cannot import name 'delegaciones' from 'api.admin'`.

- [ ] **Step 2: Backend**

Create `$WTP/backend/api/admin/delegaciones.py`:

```python
"""Frente G (2026-09-16): Admin -> Delegaciones de Ada.

Proxy de superadmin a Jacobs (jax/jacobs/routes.py, /jacobs/delegaciones/*).
Ada lanza sub-pipelines sola dentro de techos; lo que excede espera acá a
Fernando. Aprobar fija un techo SOLO para ese árbol, con usuario y hora (los
guarda Jacobs). El rol (`plataforma`) y la identidad los pone ESTE backend:
el cuerpo del cliente no los acepta (extra prohibido).

Aprobar ejecuta (puede lanzar sub-pipelines): con el kill switch puesto, 423
(frente B, exigir_mesa_libre). Rechazar detiene: nunca se frena.
"""
from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.pipelines import INVOKED_BY_PLATAFORMA, JACOBS_URL
from auth.middleware import require_superadmin
from auth.models import AuthUser
from http_client import get_http_client
from kill_switch import exigir_mesa_libre

router = APIRouter(prefix="/api/admin/delegaciones")

# Aprobar crea los hijos dentro del request de Jacobs (token + plan fijo, sin
# LLM): medido en la carga de la Task 13; 30 s es margen sobre ese p99.
JACOBS_DELEGACIONES_TIMEOUT = float(os.getenv("JACOBS_DELEGACIONES_TIMEOUT", "30.0"))


class AprobarDelegacion(BaseModel):
    techo_tokens: int | None = Field(default=None, ge=1)
    techo_usd: float | None = Field(default=None, gt=0)
    techo_pipelines: int | None = Field(default=None, ge=1)
    techo_hijos_por_step: int | None = Field(default=None, ge=1)
    aceptar_consumo_incompleto: bool = False

    model_config = {"extra": "forbid"}


async def _reenviar(metodo: str, ruta: str, cuerpo: dict | None = None) -> dict:
    cliente = await get_http_client()
    try:
        if metodo == "GET":
            r = await cliente.get(f"{JACOBS_URL}{ruta}", timeout=JACOBS_DELEGACIONES_TIMEOUT)
        else:
            r = await cliente.post(f"{JACOBS_URL}{ruta}", json=cuerpo, timeout=JACOBS_DELEGACIONES_TIMEOUT)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="jacobs_no_disponible") from exc
    try:
        datos = r.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="jacobs_error") from exc
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=datos.get("detail", "jacobs_error"))
    return datos


@router.get("")
async def listar_delegaciones(user: AuthUser = Depends(require_superadmin)) -> dict:
    return await _reenviar("GET", "/delegaciones/pendientes")


@router.post("/{aprobacion_id}/aprobar")
async def aprobar_delegacion(
    aprobacion_id: int,
    cuerpo: AprobarDelegacion,
    user: AuthUser = Depends(require_superadmin),
    _mesa_libre: AuthUser = Depends(exigir_mesa_libre),
) -> dict:
    return await _reenviar("POST", f"/delegaciones/{aprobacion_id}/aprobar", {
        **cuerpo.model_dump(), "invoked_by": INVOKED_BY_PLATAFORMA, "user_id": str(user.user_id),
    })


@router.post("/{aprobacion_id}/rechazar")
async def rechazar_delegacion(aprobacion_id: int, user: AuthUser = Depends(require_superadmin)) -> dict:
    return await _reenviar("POST", f"/delegaciones/{aprobacion_id}/rechazar", {
        "invoked_by": INVOKED_BY_PLATAFORMA, "user_id": str(user.user_id),
    })
```

En `$WTP/backend/main.py`: `from api.admin.delegaciones import router as admin_delegaciones_router` junto a los otros routers de admin, y `app.include_router(admin_delegaciones_router)` después de `app.include_router(admin_motors_router)`.

```bash
cd $WTP/backend && JAX_REPO_PATH=$WT $PYP -m pytest -q tests/test_admin_delegaciones.py 2>&1 | tail -1
```
Expected: `8 passed`.

- [ ] **Step 3: Test de frontend que falla**

Create `$WTP/frontend/src/pages/admin/AdminDelegaciones.test.jsx`:

```jsx
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import '@testing-library/jest-dom'

// Frente G (2026-09-16): Admin -> Delegaciones. Aprobar va en Dialogo con los
// techos prellenados con lo proyectado; rechazar aborta, así que va con
// ConfirmacionSuma. Errores traducidos por código en un toast.
vi.mock('../../api/client', () => ({ default: { get: vi.fn(), post: vi.fn() } }))
const addToastMock = vi.fn()
vi.mock('../../store/useJaxStore', () => ({
  useJaxStore: (selector) => selector({ addToast: addToastMock }),
}))

import api from '../../api/client'
import AdminDelegaciones, { valoresIniciales } from './AdminDelegaciones'
import { I18nProvider } from '../../i18n/index.jsx'
import es from '../../i18n/es.js'
import en from '../../i18n/en.js'

const EXCEDE = {
  id: 5, tipo: 'excede', pipeline_id: 'p-1', root_pipeline_id: 'p-1', step_id: 's-1',
  pipeline_name: 'Investigar costos', creada_at: 1789000000,
  techos: [{ techo: 'hijos_por_step', proyectado: 12, limite: 10, aprobable: true },
           { techo: 'usd_arbol', proyectado: 10.5, limite: 10, aprobable: true }],
}

function renderizar() {
  return render(<I18nProvider><AdminDelegaciones /></I18nProvider>)
}

beforeEach(() => {
  localStorage.clear()
  api.get.mockReset()
  api.post.mockReset()
  addToastMock.mockReset()
  api.get.mockResolvedValue({ data: { aprobaciones: [EXCEDE] } })
})

afterEach(() => { vi.restoreAllMocks() })

describe('AdminDelegaciones', () => {
  it('lista la aprobación con su tipo y sus techos traducidos', async () => {
    renderizar()
    expect(await screen.findByText('Investigar costos')).toBeInTheDocument()
    expect(screen.getByText(es.delegacionesTipos.excede)).toBeInTheDocument()
    expect(screen.getByText(es.delegacionesTechoNombre.hijos_por_step)).toBeInTheDocument()
    expect(api.get).toHaveBeenCalledWith('/admin/delegaciones')
  })

  it('sin pendientes muestra el vacío', async () => {
    api.get.mockResolvedValue({ data: { aprobaciones: [] } })
    renderizar()
    expect(await screen.findByText(es.delegacionesVacio)).toBeInTheDocument()
  })

  it('valoresIniciales prellena con lo proyectado (entero hacia arriba, USD tal cual)', () => {
    expect(valoresIniciales(EXCEDE.techos)).toEqual({ techo_hijos_por_step: '12', techo_usd: '10.5' })
  })

  it('aprobar abre un diálogo propio y envía los techos como números', async () => {
    api.post.mockResolvedValue({ data: { estado: 'aprobada' } })
    renderizar()
    fireEvent.click(await screen.findByRole('button', { name: es.delegacionesAprobar }))
    const dialogo = screen.getByRole('dialog')
    fireEvent.click(within(dialogo).getByRole('button', { name: es.delegacionesAprobar }))
    await waitFor(() => expect(api.post).toHaveBeenCalledWith(
      '/admin/delegaciones/5/aprobar',
      { techo_hijos_por_step: 12, techo_usd: 10.5, aceptar_consumo_incompleto: false },
    ))
    expect(addToastMock).toHaveBeenCalledWith({ type: 'success', message: es.delegacionesAprobada })
  })

  it('rechazar pide la suma antes de abortar', async () => {
    vi.spyOn(Math, 'random').mockReturnValue(0)
    api.post.mockResolvedValue({ data: { estado: 'rechazada' } })
    renderizar()
    fireEvent.click(await screen.findByRole('button', { name: es.delegacionesRechazar }))
    const dialogo = screen.getByRole('dialog')
    fireEvent.change(within(dialogo).getByLabelText(es.confirmSumLabel(10, 1)), { target: { value: '11' } })
    fireEvent.click(within(dialogo).getByRole('button', { name: es.delegacionesRechazar }))
    await waitFor(() => expect(api.post).toHaveBeenCalledWith('/admin/delegaciones/5/rechazar'))
  })

  it('un techo insuficiente se explica traducido', async () => {
    api.post.mockRejectedValue({ response: { status: 422, data: { detail: { code: 'techo_insuficiente', faltan: [] } } } })
    renderizar()
    fireEvent.click(await screen.findByRole('button', { name: es.delegacionesAprobar }))
    fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: es.delegacionesAprobar }))
    await waitFor(() => expect(addToastMock).toHaveBeenCalledWith(
      { type: 'error', message: es.adminErrors.techo_insuficiente }))
  })

  it('las claves nuevas existen en es y en', () => {
    for (const idioma of [es, en]) {
      for (const clave of ['queued', 'awaiting_approval', 'waiting_children']) {
        expect(idioma.pipelineStatusLabels[clave]).toBeTruthy()
      }
      for (const codigo of ['techo_insuficiente', 'aprobacion_ya_resuelta', 'aprobacion_no_encontrada', 'jacobs_no_disponible', 'jacobs_error']) {
        expect(idioma.adminErrors[codigo]).toBeTruthy()
      }
      expect(Object.keys(idioma.delegacionesTechoNombre).sort()).toEqual(
        ['consumo_incompleto', 'hijos_por_step', 'pipelines_arbol', 'tokens_arbol', 'usd_arbol'])
    }
  })
})
```

```bash
cd $WTP/frontend && pwd && npx vitest run src/pages/admin/AdminDelegaciones.test.jsx 2>&1 | tee -a $L | tail -5
```
Expected: falla la colección (`Failed to resolve import "./AdminDelegaciones"`).

- [ ] **Step 4: Pantalla, ruta, menú e i18n**

Create `$WTP/frontend/src/pages/admin/AdminDelegaciones.jsx`:

```jsx
import { useState, useEffect, useCallback } from 'react'
import { useI18n, localeFor } from '../../i18n/index.jsx'
import api from '../../api/client'
import { useJaxStore } from '../../store/useJaxStore'
import Dialogo from '../../components/Dialogo'
import ConfirmacionSuma from '../../components/ConfirmacionSuma'
import { mensajeDeError } from './erroresAdmin'
import { textoDeKillSwitch } from '../../api/errores'

// Frente G (2026-09-16): delegaciones de Ada que esperan a Fernando. Aprobar
// fija techos SOLO para ese árbol (Jacobs guarda usuario y hora); rechazar
// aborta lo que esperaba, por eso pide la suma. Solo tokens de tema.
const CAMPO_DE_TECHO = {
  tokens_arbol: 'techo_tokens',
  usd_arbol: 'techo_usd',
  pipelines_arbol: 'techo_pipelines',
  hijos_por_step: 'techo_hijos_por_step',
}
const CAMPO = 'w-full bg-hundido border border-borde-control rounded-lg px-3 py-2 text-sm text-texto focus:outline-none focus:border-foco'

export function valoresIniciales(techos) {
  const valores = {}
  for (const techo of techos) {
    const campo = CAMPO_DE_TECHO[techo.techo]
    if (!campo) continue
    valores[campo] = campo === 'techo_usd' ? String(techo.proyectado) : String(Math.ceil(techo.proyectado))
  }
  return valores
}

export default function AdminDelegaciones() {
  const { t, lang } = useI18n()
  const addToast = useJaxStore((s) => s.addToast)
  const [aprobaciones, setAprobaciones] = useState(null)
  const [aprobando, setAprobando] = useState(null)
  const [valores, setValores] = useState({})
  const [aceptarIncompleto, setAceptarIncompleto] = useState(false)
  const [rechazando, setRechazando] = useState(null)
  const [enviando, setEnviando] = useState(false)
  const numero = (n) => Number(n).toLocaleString(localeFor(lang))

  const cargar = useCallback(async () => {
    try {
      const r = await api.get('/admin/delegaciones')
      setAprobaciones(r.data.aprobaciones)
    } catch {
      addToast({ type: 'error', message: t.delegacionesErrorCargar })
    }
  }, [addToast, t])

  useEffect(() => { cargar() }, [cargar])

  function abrirAprobar(aprobacion) {
    setValores(valoresIniciales(aprobacion.techos))
    setAceptarIncompleto(false)
    setAprobando(aprobacion)
  }

  async function aprobar(e) {
    e.preventDefault()
    if (enviando) return
    setEnviando(true)
    const cuerpo = {}
    for (const [campo, valor] of Object.entries(valores)) {
      cuerpo[campo] = campo === 'techo_usd' ? Number(valor) : parseInt(valor, 10)
    }
    cuerpo.aceptar_consumo_incompleto = aceptarIncompleto
    try {
      await api.post(`/admin/delegaciones/${aprobando.id}/aprobar`, cuerpo)
      addToast({ type: 'success', message: t.delegacionesAprobada })
      setAprobando(null)
      await cargar()
    } catch (err) {
      addToast({ type: 'error', message: textoDeKillSwitch(t, err) || mensajeDeError(t, err) })
    } finally {
      setEnviando(false)
    }
  }

  async function rechazar() {
    try {
      await api.post(`/admin/delegaciones/${rechazando.id}/rechazar`)
      addToast({ type: 'success', message: t.delegacionesRechazada })
      setRechazando(null)
      await cargar()
    } catch (err) {
      addToast({ type: 'error', message: mensajeDeError(t, err) })
    }
  }

  return (
    <div className="space-y-4">
      <h1 className="text-lg font-semibold text-texto">{t.delegacionesTitulo}</h1>
      {aprobaciones !== null && aprobaciones.length === 0 && (
        <p className="text-sm text-texto-tenue">{t.delegacionesVacio}</p>
      )}
      {aprobaciones !== null && aprobaciones.length > 0 && (
        <table className="w-full text-sm border border-borde rounded-lg overflow-hidden">
          <thead className="bg-superficie text-texto-suave text-xs uppercase">
            <tr>
              <th className="px-4 py-2 text-left">{t.delegacionesPipeline}</th>
              <th className="px-4 py-2 text-left">{t.delegacionesTipo}</th>
              <th className="px-4 py-2 text-left">{t.delegacionesTechos}</th>
              <th className="px-4 py-2 text-left">{t.delegacionesCreada}</th>
              <th className="px-4 py-2" />
            </tr>
          </thead>
          <tbody>
            {aprobaciones.map((a) => (
              <tr key={a.id} className="border-t border-borde">
                <td className="px-4 py-3 text-texto">{a.pipeline_name}</td>
                <td className="px-4 py-3 text-texto-suave">{t.delegacionesTipos[a.tipo]}</td>
                <td className="px-4 py-3">
                  <ul className="space-y-1">
                    {a.techos.map((techo) => (
                      <li key={techo.techo} className="text-xs text-texto">
                        <span className="font-medium">{t.delegacionesTechoNombre[techo.techo]}</span>{' '}
                        <span className="text-texto-tenue">
                          {t.delegacionesProyectadoSobreLimite(numero(techo.proyectado), numero(techo.limite))}
                        </span>
                      </li>
                    ))}
                  </ul>
                </td>
                <td className="px-4 py-3 text-xs text-texto-tenue">
                  {new Date(a.creada_at * 1000).toLocaleString(localeFor(lang))}
                </td>
                <td className="px-4 py-3 text-right space-x-2 whitespace-nowrap">
                  <button type="button" onClick={() => abrirAprobar(a)}
                          className="px-3 py-1.5 rounded-lg bg-accion hover:bg-accion-hover text-sobre-color text-xs font-semibold">
                    {t.delegacionesAprobar}
                  </button>
                  <button type="button" onClick={() => setRechazando(a)}
                          className="px-3 py-1.5 rounded-lg text-xs text-peligro-texto hover:bg-peligro-fondo">
                    {t.delegacionesRechazar}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {aprobando && (
        <Dialogo idTitulo="delegacion-aprobar-titulo" titulo={t.delegacionesAprobarTitulo} onCerrar={() => setAprobando(null)}>
          <form onSubmit={aprobar} className="space-y-3">
            <p className="text-xs text-texto-tenue">{t.delegacionesAprobarAyuda}</p>
            {aprobando.techos.filter((techo) => CAMPO_DE_TECHO[techo.techo]).map((techo) => {
              const campo = CAMPO_DE_TECHO[techo.techo]
              return (
                <div key={campo}>
                  <label htmlFor={`delegacion-${campo}`} className="block text-sm text-texto mb-1">
                    {t.delegacionesTechoNombre[techo.techo]}
                  </label>
                  <input id={`delegacion-${campo}`} type="number" min={techo.proyectado}
                         step={campo === 'techo_usd' ? '0.01' : '1'} value={valores[campo] ?? ''}
                         onChange={(e) => setValores((v) => ({ ...v, [campo]: e.target.value }))}
                         className={CAMPO} />
                </div>
              )
            })}
            {aprobando.techos.some((techo) => techo.techo === 'consumo_incompleto') && (
              <label className="flex items-center gap-2 text-sm text-texto">
                <input type="checkbox" checked={aceptarIncompleto} onChange={(e) => setAceptarIncompleto(e.target.checked)} />
                {t.delegacionesAceptarIncompleto}
              </label>
            )}
            <div className="flex gap-2 justify-end pt-2">
              <button type="button" onClick={() => setAprobando(null)}
                      className="px-3 py-1.5 rounded-lg text-sm text-texto-suave hover:text-texto">
                {t.adminCreateCancel}
              </button>
              <button type="submit" disabled={enviando}
                      className="px-4 py-1.5 rounded-lg bg-accion hover:bg-accion-hover text-sobre-color text-sm font-semibold disabled:opacity-50">
                {t.delegacionesAprobar}
              </button>
            </div>
          </form>
        </Dialogo>
      )}

      {rechazando && (
        <ConfirmacionSuma
          titulo={t.delegacionesRechazarTitulo}
          mensaje={t.delegacionesRechazarMensaje(rechazando.pipeline_name)}
          textoConfirmar={t.delegacionesRechazar}
          onConfirmar={rechazar}
          onCancelar={() => setRechazando(null)}
        />
      )}
    </div>
  )
}
```

**Tokens de tema:** antes de escribir, verificar con `grep -n "accion-hover\|peligro-texto\|peligro-fondo" $WTP/frontend/tailwind.config.js $WTP/frontend/src/index.css` que existen. Si alguno no existe, usar el par declarado que ya usan `AdminUsers.jsx` para el mismo rol (acción principal y acción destructiva secundaria), nunca un color literal.

En `$WTP/frontend/src/pages/Admin.jsx`: `import AdminDelegaciones from './admin/AdminDelegaciones'` y `<Route path="delegaciones" element={<AdminDelegaciones />} />` después de la de `costs`.

En `$WTP/frontend/src/components/admin/AdminSidebar.jsx`, en `NAV_ITEMS` después de `costs`: `{ path: 'delegaciones', labelKey: 'adminDelegaciones', icon: '⑂' },`.

En `$WTP/frontend/src/i18n/es.js` (en `pipelineStatusLabels`, en `adminErrors` y junto a las claves de admin):

```js
  // pipelineStatusLabels (frente G):
    queued: 'En cola',
    awaiting_approval: 'Esperando aprobación de techos',
    waiting_children: 'Esperando sub-pipelines',
  // adminErrors (frente G):
    techo_insuficiente: 'El techo aprobado no cubre lo proyectado.',
    aprobacion_ya_resuelta: 'Esta delegación ya fue resuelta.',
    aprobacion_no_encontrada: 'La delegación ya no existe.',
    jacobs_no_disponible: 'Jacobs no responde. Probá de nuevo en un momento.',
    jacobs_error: 'Jacobs respondió con un error inesperado.',
  // claves de primer nivel (frente G):
  adminDelegaciones: 'Delegaciones',
  delegacionesTitulo: 'Delegaciones de Ada que esperan aprobación',
  delegacionesVacio: 'No hay delegaciones pendientes.',
  delegacionesPipeline: 'Pipeline',
  delegacionesTipo: 'Motivo',
  delegacionesTechos: 'Techos',
  delegacionesCreada: 'Pedida',
  delegacionesTipos: { excede: 'Excede un techo al delegar', cruzado: 'Cruzó un techo a mitad de camino' },
  delegacionesTechoNombre: {
    tokens_arbol: 'Tokens del árbol',
    usd_arbol: 'USD del árbol',
    pipelines_arbol: 'Pipelines del árbol',
    hijos_por_step: 'Sub-pipelines por paso',
    consumo_incompleto: 'Consumo sin registrar completo',
  },
  delegacionesProyectadoSobreLimite: (proyectado, limite) => `${proyectado} (límite ${limite})`,
  delegacionesAprobar: 'Aprobar',
  delegacionesRechazar: 'Rechazar',
  delegacionesAprobarTitulo: 'Aprobar techos para este árbol',
  delegacionesAprobarAyuda: 'Los techos aprobados valen solo para este árbol de sub-pipelines.',
  delegacionesAceptarIncompleto: 'Acepto seguir con consumo sin registrar completo',
  delegacionesRechazarTitulo: 'Rechazar la delegación',
  delegacionesRechazarMensaje: (nombre) => `Se abortará lo que espera en "${nombre}". Los resultados que ya existan se conservan.`,
  delegacionesAprobada: 'Delegación aprobada.',
  delegacionesRechazada: 'Delegación rechazada.',
  delegacionesErrorCargar: 'No se pudieron cargar las delegaciones.',
```

En `$WTP/frontend/src/i18n/en.js`, las mismas claves:

```js
  // pipelineStatusLabels (frente G):
    queued: 'Queued',
    awaiting_approval: 'Awaiting ceiling approval',
    waiting_children: 'Waiting for sub-pipelines',
  // adminErrors (frente G):
    techo_insuficiente: 'The approved ceiling does not cover the projection.',
    aprobacion_ya_resuelta: 'This delegation was already resolved.',
    aprobacion_no_encontrada: 'The delegation no longer exists.',
    jacobs_no_disponible: 'Jacobs is not responding. Try again in a moment.',
    jacobs_error: 'Jacobs returned an unexpected error.',
  // top-level keys (frente G):
  adminDelegaciones: 'Delegations',
  delegacionesTitulo: 'Ada delegations awaiting approval',
  delegacionesVacio: 'No pending delegations.',
  delegacionesPipeline: 'Pipeline',
  delegacionesTipo: 'Reason',
  delegacionesTechos: 'Ceilings',
  delegacionesCreada: 'Requested',
  delegacionesTipos: { excede: 'Exceeds a ceiling when delegating', cruzado: 'Crossed a ceiling midway' },
  delegacionesTechoNombre: {
    tokens_arbol: 'Tree tokens',
    usd_arbol: 'Tree USD',
    pipelines_arbol: 'Tree pipelines',
    hijos_por_step: 'Sub-pipelines per step',
    consumo_incompleto: 'Incompletely recorded usage',
  },
  delegacionesProyectadoSobreLimite: (projected, limit) => `${projected} (limit ${limit})`,
  delegacionesAprobar: 'Approve',
  delegacionesRechazar: 'Reject',
  delegacionesAprobarTitulo: 'Approve ceilings for this tree',
  delegacionesAprobarAyuda: 'Approved ceilings apply only to this sub-pipeline tree.',
  delegacionesAceptarIncompleto: 'I accept continuing with incompletely recorded usage',
  delegacionesRechazarTitulo: 'Reject the delegation',
  delegacionesRechazarMensaje: (name) => `Whatever is waiting in "${name}" will be aborted. Existing results are kept.`,
  delegacionesAprobada: 'Delegation approved.',
  delegacionesRechazada: 'Delegation rejected.',
  delegacionesErrorCargar: 'Could not load delegations.',
```

- [ ] **Step 5: Verde, suites y commit**

```bash
cd $WTP/frontend && npx vitest run src/pages/admin/AdminDelegaciones.test.jsx 2>&1 | tail -3
cd $WTP/frontend && npx vitest run 2>&1 | tail -3
cd $WTP/frontend && npm run build 2>&1 | tail -3
cd $WTP/backend && JAX_REPO_PATH=$WT $PYP -m pytest -q tests/test_admin_delegaciones.py tests/test_capabilities_delegacion.py 2>&1 | tail -1
```
Expected:
- `7 passed` en el test nuevo;
- la suite entera de vitest en verde, incluidos el escaneo de diálogos del navegador (`politica/dialogosDelNavegador.test.js`), la paridad de claves es/en y `RightPanel.test.jsx`;
- el build sin errores;
- backend: `12 passed`.

Si el test de paridad de i18n exige orden o ubicación, se acomoda la clave y no el test.

Verificar también en ambos temas: `npm run dev` en `$WTP/frontend`, entrar a `/admin/delegaciones` con claro y con oscuro y mirar que no quede ningún color fijo. Anotar el resultado en `$L`.

```bash
git -C $WTP add backend/api/admin/delegaciones.py backend/main.py backend/tests/test_admin_delegaciones.py frontend/src/pages/admin/AdminDelegaciones.jsx frontend/src/pages/admin/AdminDelegaciones.test.jsx frontend/src/pages/Admin.jsx frontend/src/components/admin/AdminSidebar.jsx frontend/src/i18n/es.js frontend/src/i18n/en.js
git -C $WTP commit -m "feat(admin): Delegaciones de Ada -- aprobar techos por árbol o rechazar

Proxy de superadmin a /jacobs/delegaciones/* (rol e identidad del servidor,
extra prohibido, 423 con el freno al aprobar). Pantalla con Dialogo para
aprobar (techos prellenados con lo proyectado) y ConfirmacionSuma para
rechazar; i18n es/en y estados nuevos del pipeline. 8 backend + 7 vitest
vistos en rojo.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---
### Task 12: El árbol de punta a punta (matriz del spec §6) y suites completas

**Files:**
- Create: `tests/test_delegacion_arbol_e2e.py`

**Interfaces:**
- Consumes: todo lo anterior. Sólo se sustituye `executor._dispatch_step` (la llamada al LLM) y `executor._persist_step_to_repo` (la copia `.md`). `run_pipeline`, la cola, los tokens, la base, los eventos y las rutas de aprobación son los reales.
- Produces: la matriz de pruebas del emisor del spec §6, ejecutada sobre el flujo completo.

- [ ] **Step 1: Escribir la matriz**

Create `tests/test_delegacion_arbol_e2e.py`:

```python
"""Emisor de sub-pipelines de Ada de punta a punta (frente G, 2026-09-16).

Matriz del spec §6 sobre el flujo REAL: run_pipeline, cola, tokens de F, base,
eventos y rutas de aprobación. Lo único sustituido es la llamada al modelo
(_dispatch_step) y la copia .md de cortesía (_persist_step_to_repo).

Necesita el esquema de jax-platform: corre en jacobs-gobernanza-db.
Barrera: jacobs/_arnes_delegacion importa jacobs/_arnes_ada primero.
"""
from __future__ import annotations

from jacobs import _arnes_delegacion as g  # primero: barrera de base de prueba

import asyncio  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import unittest  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402

from jacobs import executor, routes, store  # noqa: E402
from jacobs.models import PipelineCreateRequest, PipelineStatus, StepSpec  # noqa: E402

TERMINALES = {PipelineStatus.completed, PipelineStatus.aborted, PipelineStatus.expired, PipelineStatus.failed}


def _plan(subs):
    return json.dumps({"subobjetivos": subs, "integracion": {"objetivo": "integrá todo"}})


def _sub(i, objetivo=None, depende=(), skip=False):
    return {"id": f"s{i}", "objetivo": objetivo or f"sub-objetivo {i}", "faceta": "jekyll",
            "capability": "analysis", "depende_de": [f"s{d}" for d in depende], "skip_on_fail": skip}


@unittest.skipUnless(os.getenv("JAX_DB_HOST"), "necesita la MariaDB real (jax_memory_test)")
class ArbolE2ETest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await store.init_tables()
        await g.vaciar_activos()
        entorno = patch.dict(os.environ)
        entorno.start()
        self.addCleanup(entorno.stop)
        os.environ["JAX_KILL_SWITCH_PATH"] = "/tmp/jax-frente-g-sin-freno/PAUSE"
        for nombre in [n for n in os.environ if n.startswith("JAX_ADA_")]:
            os.environ.pop(nombre)
        self.salidas_delegate: list[str] = []
        self.integraciones: list[str] = []
        self.liberar = asyncio.Event()
        self.liberar.set()
        for destino in ("jacobs.policy.check_kill_switch", "jacobs.executor.check_kill_switch"):
            freno = patch(destino, return_value=False)
            freno.start()
            self.addCleanup(freno.stop)
        despacho = patch("jacobs.executor._dispatch_step", self._despacho)
        despacho.start()
        self.addCleanup(despacho.stop)
        copia = patch("jacobs.executor._persist_step_to_repo", AsyncMock())
        copia.start()
        self.addCleanup(copia.stop)
        telegram = patch("jacobs.reaper.send_telegram_alert", AsyncMock(return_value=g.TELEGRAM_OK))
        self.telegram = telegram.start()
        self.addCleanup(telegram.stop)
        self.raices: list[str] = []

    async def asyncTearDown(self):
        self.liberar.set()
        for root in self.raices:
            await g.limpiar_raiz(root)

    async def _despacho(self, step, pipeline):
        if step.capability == "delegate":
            return {"success": True, "result": self.salidas_delegate.pop(0)}
        if step.capability == "integrate":
            self.integraciones.append(step.input["prompt"])
            return {"success": True, "result": "integrado"}
        prompt = step.input.get("prompt", "")
        await self.liberar.wait()
        if prompt.startswith("falla"):
            raise RuntimeError("falla simulada del sub-objetivo")
        return {"success": True, "result": f"hecho: {prompt}"}

    async def lanzar_padre(self, *salidas):
        self.salidas_delegate.extend(salidas)
        req = PipelineCreateRequest(
            name="e2e-g-padre", objective="objetivo grande", invoked_by="plataforma",
            user_id="1", tenant_id="1", mode="autonomous",
            steps=[StepSpec(facet="ada", capability="delegate", prompt="repartí el objetivo")],
        )
        padre, _steps = await routes._crear_pipeline(req)
        self.raices.append(padre.pipeline_id)
        await executor.run_pipeline(padre)
        return padre

    async def esperar(self, pipeline_id, estados, limite=30.0):
        fin = time.monotonic() + limite
        estado = None
        while time.monotonic() < fin:
            estado = await store.pipeline_status(pipeline_id)
            if estado in estados:
                return estado
            await asyncio.sleep(0.2)
        self.fail(f"{pipeline_id} no llegó a {estados}: está en {estado}")

    async def test_plan_valido_tres_hijos_integra_y_completa(self):
        padre = await self.lanzar_padre(_plan([_sub(0), _sub(1), _sub(2)]))
        await self.esperar(padre.pipeline_id, {PipelineStatus.completed})
        hijos = await store.pipelines_hijos(padre.pipeline_id)
        self.assertEqual({h.status for h in hijos}, {PipelineStatus.completed})
        self.assertEqual(await g.tokens_emitidos(padre.pipeline_id), 3)
        [prompt] = self.integraciones
        for i in range(3):
            self.assertIn(f"s{i}", prompt)
            self.assertIn(f"hecho: sub-objetivo {i}", prompt)
        tipos = [e["event_type"] for e in await store.events_by_pipeline(padre.pipeline_id)]
        orden = ["DELEGACION_PROPUESTA", "DELEGACION_APROBADA", "PIPELINE_ESPERA_HIJOS",
                 "DELEGACION_HIJOS_TERMINADOS", "PIPELINE_COMPLETED"]
        self.assertEqual([t for t in tipos if t in orden], orden)
        for h in hijos:
            self.assertEqual(len(await g.eventos(h.pipeline_id, "SUBPIPELINE_ADMITIDO")), 1)

    async def test_plan_invalido_dos_veces_falla_sin_hijos(self):
        padre = await self.lanzar_padre("esto no es un plan", '{"subobjetivos": []}')
        self.assertEqual(await store.pipeline_status(padre.pipeline_id), PipelineStatus.aborted)
        self.assertEqual(await g.tokens_emitidos(padre.pipeline_id), 0)
        self.assertEqual(await store.pipelines_hijos(padre.pipeline_id), [])
        self.assertEqual(len(await g.eventos(padre.pipeline_id, "DELEGACION_PLAN_INVALIDO")), 2)

    async def test_ciclo_y_despues_plan_valido(self):
        ciclo = _plan([_sub(0, depende=(1,)), _sub(1, depende=(0,))])
        padre = await self.lanzar_padre(ciclo, _plan([_sub(0)]))
        await self.esperar(padre.pipeline_id, {PipelineStatus.completed})
        motivo = (await g.eventos(padre.pipeline_id, "DELEGACION_PLAN_INVALIDO"))[0]["payload"]["motivo"]
        self.assertIn("ciclo", motivo)

    async def test_hijo_fallado_sin_skip_y_la_integracion_recibe_el_fallo(self):
        padre = await self.lanzar_padre(_plan([_sub(0, objetivo="falla a propósito"), _sub(1, depende=(0,))]))
        await self.esperar(padre.pipeline_id, {PipelineStatus.completed})
        s0, s1 = sorted(await store.pipelines_hijos(padre.pipeline_id),
                        key=lambda h: h.context["delegacion"]["subobjetivo_id"])
        self.assertEqual((s0.status, s1.status), (PipelineStatus.aborted, PipelineStatus.aborted))
        self.assertEqual(len(await g.eventos(s1.pipeline_id, "SUBPIPELINE_DEPENDENCIA_FALLIDA")), 1)
        [prompt] = self.integraciones
        self.assertIn("estado aborted", prompt)
        self.assertIn("fallo declarado", prompt)

    async def test_techo_bajado_pide_aprobacion_y_al_aprobar_lanza(self):
        os.environ["JAX_ADA_MAX_HIJOS_POR_STEP"] = "2"
        padre = await self.lanzar_padre(_plan([_sub(0), _sub(1), _sub(2)]))
        self.assertEqual(await store.pipeline_status(padre.pipeline_id), PipelineStatus.awaiting_approval)
        self.assertEqual(await g.tokens_emitidos(padre.pipeline_id), 0)
        self.telegram.assert_awaited()
        [pendiente] = [a for a in (await routes.delegaciones_pendientes())["aprobaciones"]
                       if a["pipeline_id"] == padre.pipeline_id]
        await routes.aprobar_delegacion(pendiente["id"], routes.AprobarDelegacionRequest(
            invoked_by="plataforma", user_id="1", techo_hijos_por_step=3))
        await self.esperar(padre.pipeline_id, {PipelineStatus.completed})
        self.assertEqual(len(await store.pipelines_hijos(padre.pipeline_id)), 3)

    async def test_cancelar_el_padre_no_deja_huerfanos_gastando(self):
        self.liberar.clear()
        padre = await self.lanzar_padre(_plan([_sub(i) for i in range(5)]))
        await asyncio.sleep(1.0)
        estados = [h.status for h in await store.pipelines_hijos(padre.pipeline_id)]
        self.assertEqual(estados.count(PipelineStatus.running), 3)
        self.assertEqual(estados.count(PipelineStatus.queued), 2)
        await routes.cancel_pipeline(padre.pipeline_id)
        self.liberar.set()
        for h in await store.pipelines_hijos(padre.pipeline_id):
            await self.esperar(h.pipeline_id, TERMINALES)
        await asyncio.sleep(1.0)
        hijos = await store.pipelines_hijos(padre.pipeline_id)
        self.assertEqual({h.status for h in hijos}, {PipelineStatus.aborted})
        self.assertEqual(self.integraciones, [])
        self.assertEqual(await store.pipeline_status(padre.pipeline_id), PipelineStatus.aborted)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Rojo contra el commit de la Task 6, verde en la rama**

El rojo se lee en un worktree temporal del commit de la Task 6, que tiene el esquema y los techos pero no delega:

```bash
T6=$(git -C $WT log --format=%H --grep="proyección de techos del árbol de delegación" -1)
git -C /home/fruiz/jax worktree add /tmp/jax-frente-g-rojo $T6
cp $WT/tests/test_delegacion_arbol_e2e.py /tmp/jax-frente-g-rojo/tests/
bash -c "$DBTEST; cd /tmp/jax-frente-g-rojo && PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_delegacion_arbol_e2e.py" 2>&1 | tee -a $L | tail -8
git -C /home/fruiz/jax worktree remove --force /tmp/jax-frente-g-rojo
```
Expected: `6 failed`, todos con `AttributeError: module 'jacobs.routes' has no attribute '_crear_pipeline'`. En ese commit la creación, la delegación y la cola todavía no existen.

En la rama:

```bash
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_delegacion_arbol_e2e.py" 2>&1 | tee -a $L | tail -3
for i in 1 2 3 4 5; do bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_delegacion_arbol_e2e.py" 2>&1 | tail -1; done
```
Expected: `6 passed` las seis veces. Si alguno es intermitente, **no se aceptan esperas más largas como arreglo**: se diagnostica la carrera con `superpowers:systematic-debugging`.

- [ ] **Step 3: Suites completas, escáneres y espejos**

```bash
cd $WT && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q \
  $(sed -n '/^  tests-puros:/,/^      - name: Piso exacto/p' .github/workflows/policy.yml | grep -oE '(tests|las_manos)/[A-Za-z0-9_./]+\.py' | sort -u) \
  tests/test_config_delegacion.py tests/test_plan_delegacion_esquema.py tests/test_delegacion_modelos_puro.py \
  tests/test_delegacion_uso_puro.py tests/test_delegacion_techos_puro.py tests/test_delegacion_politica_puro.py \
  tests/test_delegacion_executor_puro.py 2>&1 | tail -2
cd $WT && PYTHONPATH=las_manos $PY -m pytest -q las_manos/_output_validator_test.py 2>&1 | tail -1
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_delegacion_io_test.py tests/test_delegacion_arbol_e2e.py jacobs/_subpipeline_contrato_io_test.py tests/test_subpipeline_contrato_rutas.py jacobs/_direct_usage_test.py tests/test_plan_validation.py tests/test_contrato_dispatch_db.py" 2>&1 | tail -2
cd $WT && $PY -m pytest -q policy/tests/test_no_fail_open_except.py tests/test_no_blocking_in_async.py 2>&1 | tail -1
cd $WT && JAX_PLATFORM_REPO_ROOT=$WTP python3 scripts/check_mirror_sync.py; echo "mirror-sync exit=$?"
```
Expected:
- los puros dan el piso de `tests-puros` de la base más 69 (17 + 25 + 4 + 5 + 12 + 3 + 8), con 0 failed;
- el validador de salidas y la base dan todo `passed`;
- los escáneres, en verde;
- `mirror-sync exit=0`: este frente no toca ninguna familia espejada. Si reporta algo, parar, porque el cambio es de los dos repos.

- [ ] **Step 4: Commit**

```bash
git -C $WT add tests/test_delegacion_arbol_e2e.py $L
git -C $WT commit -m "test(jacobs): el árbol de delegación de Ada de punta a punta (matriz del spec §6)

Tres hijos que integran y completan; plan inválido dos veces sin hijos; ciclo y
reintento; hijo fallado con dependiente y fallo declarado en la integración;
techo bajado -> aprobación -> lanzamiento; cancelar el padre sin huérfanos.
Flujo real (run_pipeline, cola, tokens, base); solo se sustituye el LLM. 6 en
rojo contra el commit de la Task 6.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: Prueba de carga de la delegación, la cola y las aprobaciones

**Files:**
- Create: `loadtest/jacobs_delegacion_app.py`
- Create: `loadtest/sembrar_carga_delegacion.py`
- Create: `loadtest/jacobs_delegacion.js`
- Create: `$WTP/loadtest/admin_delegaciones.js`

**Interfaces:**
- Consumes: `delegacion.delegar`, `cola.despachar`, rutas de la Task 10, arnés de la Task 4.
- Produces: números (rps, p50, p95, p99, tasa de error) por escenario y concurrencia, en `$L`, en los PRs y en la Biblioteca (Task 17).

Qué se mide y qué no (queda declarado en los archivos y en la Biblioteca):
- **Se mide** lo que agregó este frente en el camino de Jacobs:
  - `delegar` con 3 hijos y con 10 (el techo por defecto, peor caso): gobernanza, consumo, conteos, N tokens, N creaciones bajo el candado, integración y eventos;
  - `despachar` con la cola llena de lo que dejaron las delegaciones: candado, conteo, candidatos FIFO, consumo por raíz, transiciones y eventos;
  - `GET /jacobs/delegaciones/pendientes` con 1000 aprobaciones pendientes.
- **No se mide** el LLM ni la ejecución de los hijos: `executor.run_pipeline` se sustituye por una transición inmediata a `completed`.
- **Nunca contra producción:** la app y el sembrador se niegan a arrancar sin `JAX_DB_NAME=jax_memory_test`.
- **El proxy de jax-platform** se mide en producción **después** del deploy, sólo el `GET` (sin efectos) y con un token de superadmin que da Fernando (Task 16, Step 5). Aprobar y rechazar tienen efectos y se miden acá, en la ruta de Jacobs, que es donde está su costo.

- [ ] **Step 1: App y sembrador**

Create `loadtest/jacobs_delegacion_app.py`:

```python
"""App de carga del emisor de sub-pipelines de Ada (frente G, 2026-09-16).

NUNCA contra producción: escribe pipelines, tokens, aprobaciones y eventos. La
barrera de base es la de jacobs/_arnes_ada (se importa vía _arnes_delegacion).

Monta las rutas REALES de Jacobs y dos rutas de carga que llaman a las funciones
REALES (delegacion.delegar, cola.despachar). Lo único sustituido es
executor.run_pipeline (LLM y ejecución de los hijos): pasa el hijo admitido a
completed de inmediato, así la cola no se llena de pending eternos.

USO (Task 13 del plan 2026-09-16-frente-g-emisor-subpipelines-ada.md):
  PYTHONPATH=.:las_manos uvicorn loadtest.jacobs_delegacion_app:app --port 7798
"""
from __future__ import annotations

from jacobs import _arnes_delegacion as g  # primero: barrera de base de prueba

from fastapi import FastAPI  # noqa: E402

from jacobs import cola, delegacion, executor, routes, store  # noqa: E402
from jacobs.models import PipelineStatus  # noqa: E402


async def _correr_instantaneo(pipeline) -> None:
    await store.pipeline_transicion(pipeline.pipeline_id, (PipelineStatus.pending,), PipelineStatus.completed)


executor.run_pipeline = _correr_instantaneo

app = FastAPI(title="carga-emisor-subpipelines")
app.include_router(routes.router)


@app.on_event("startup")
async def _init() -> None:
    await store.init_tables()


@app.post("/carga/delegar")
async def carga_delegar(n: int = 3) -> dict:
    padre, paso = await g.padre_delegador()
    hecha = await delegacion.delegar(padre, paso, g.plan(n))
    # Como hace el executor después de la ola: el padre suelta su cupo. Sin esto
    # los padres `running` llenan el límite global y `despachar` mediría solo el
    # camino "sin lugar", no el de admitir con la cola llena.
    await store.pipeline_transicion(padre.pipeline_id, (PipelineStatus.running,), PipelineStatus.waiting_children)
    return {"resultado": hecha.resultado.value, "hijos": len(hecha.hijos), "raiz": padre.pipeline_id}


@app.post("/carga/despachar")
async def carga_despachar() -> dict:
    return {"admitidos": len(await cola.despachar())}
```

Create `loadtest/sembrar_carga_delegacion.py`:

```python
"""Prepara jax_memory_test para la carga del emisor (frente G, 2026-09-16).

  preparar  -> pasa a expired lo no terminal que quedó de otros tests y siembra
               1000 aprobaciones pendientes (con su pipeline) bajo la raíz
               'raiz-carga-g'
  limpiar   -> borra lo sembrado

Solo jax_memory_test (barrera de jacobs/_arnes_ada).
"""
from __future__ import annotations

from jacobs import _arnes_delegacion as g  # primero: barrera de base de prueba

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402

from jacobs import _arnes_ada as ada  # noqa: E402
from jacobs import store  # noqa: E402

RAIZ = "raiz-carga-g"
TECHOS = json.dumps([{"techo": "tokens_arbol", "proyectado": 3, "limite": 2, "aprobable": True}])


async def preparar(n: int) -> None:
    await store.init_tables()
    print("no terminales pasados a expired:", await g.vaciar_activos())
    ahora = time.time()
    ids = [str(uuid.uuid4()) for _ in range(n)]
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.executemany(
                "INSERT INTO jacobs_pipelines (pipeline_id, name, invoked_by, mode, status, plan, context_refs, "
                "created_at, updated_at, root_pipeline_id) VALUES (%s,'carga-g','plataforma','autonomous',"
                "'completed','[]','{}',%s,%s,%s)",
                [(pid, ahora, ahora, RAIZ) for pid in ids])
            await cur.executemany(
                "INSERT INTO jacobs_delegacion_aprobaciones (root_pipeline_id, pipeline_id, step_id, tipo, techos, "
                "estado, creada_at) VALUES (%s,%s,'paso','excede',%s,'pendiente',%s)",
                [(RAIZ, pid, TECHOS, ahora + i) for i, pid in enumerate(ids)])
    finally:
        conn.close()
    print("aprobaciones sembradas:", n)


async def limpiar() -> None:
    print("aprobaciones borradas:", await ada.ejecutar(
        "DELETE FROM jacobs_delegacion_aprobaciones WHERE root_pipeline_id=%s", (RAIZ,)))
    print("pipelines borrados:", await ada.ejecutar("DELETE FROM jacobs_pipelines WHERE name='carga-g'"))
    print("no terminales pasados a expired:", await g.vaciar_activos())


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="accion", required=True)
    s = sub.add_parser("preparar")
    s.add_argument("--n", type=int, default=1000)
    sub.add_parser("limpiar")
    args = p.parse_args()
    asyncio.run(preparar(args.n) if args.accion == "preparar" else limpiar())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Escenarios k6**

Create `loadtest/jacobs_delegacion.js`:

```javascript
// Carga del emisor de sub-pipelines de Ada -- politica 4 de LAS CUATRO DEL
// RENDIMIENTO (frente G, 2026-09-16). Contra la app AISLADA
// loadtest/jacobs_delegacion_app.py (jax_memory_test), nunca contra produccion.
//
// ESCENARIO:
//   delegar3    -- delegar un plan de 3 sub-objetivos (caso tipico)
//   delegar10   -- delegar 10 (el techo por defecto: peor caso)
//   despachar   -- una vuelta del despachador con la cola llena
//   pendientes  -- GET /jacobs/delegaciones/pendientes con 1000 pendientes
//
// USO: k6 run -e ESCENARIO=delegar3 -e VUS=10 loadtest/jacobs_delegacion.js

import http from 'k6/http';
import { check } from 'k6';

const BASE = __ENV.BASE || 'http://127.0.0.1:7798';
const ESCENARIO = __ENV.ESCENARIO || 'delegar3';
const VUS = parseInt(__ENV.VUS || '10', 10);
const P95_MS = ESCENARIO === 'delegar10' ? 1000 : 500;

export const options = {
  stages: [
    { duration: '5s', target: VUS },
    { duration: '20s', target: VUS },
    { duration: '5s', target: 0 },
  ],
  thresholds: {
    http_req_duration: [`p(95)<${P95_MS}`],
    http_req_failed: ['rate<0.01'],
    checks: ['rate>0.99'],
  },
};

export default function () {
  if (ESCENARIO === 'delegar3' || ESCENARIO === 'delegar10') {
    const n = ESCENARIO === 'delegar3' ? 3 : 10;
    const r = http.post(`${BASE}/carga/delegar?n=${n}`, null, { tags: { escenario: ESCENARIO } });
    check(r, { lanzada: (res) => res.status === 200 && res.json('resultado') === 'lanzada' });
  } else if (ESCENARIO === 'despachar') {
    const r = http.post(`${BASE}/carga/despachar`, null, { tags: { escenario: ESCENARIO } });
    check(r, { ok: (res) => res.status === 200 });
  } else {
    const r = http.get(`${BASE}/jacobs/delegaciones/pendientes`, { tags: { escenario: ESCENARIO } });
    check(r, { lista: (res) => res.status === 200 && res.json('aprobaciones').length > 0 });
  }
}
```

Los umbrales son p95 < 1000 ms para `delegar10` (10 creaciones seriales bajo el candado) y p95 < 500 ms para los demás.

Create `$WTP/loadtest/admin_delegaciones.js`:

```javascript
// Carga del proxy Admin -> Delegaciones (frente G, 2026-09-16), SOLO lectura:
// GET /api/admin/delegaciones. Aprobar y rechazar tienen efectos (lanzan o
// abortan) y se miden en la app aislada de Jacobs, no en produccion.
//
// USO: k6 run -e TOKEN=<access token superadmin> -e VUS=10 loadtest/admin_delegaciones.js
// El token se pasa por entorno y NUNCA se escribe en el archivo.

import http from 'k6/http';
import { check } from 'k6';

const BASE = __ENV.BASE || 'http://127.0.0.1:8080';
const TOKEN = __ENV.TOKEN;
const VUS = parseInt(__ENV.VUS || '10', 10);

export const options = {
  stages: [
    { duration: '5s', target: VUS },
    { duration: '15s', target: VUS },
    { duration: '5s', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],
    http_req_failed: ['rate<0.01'],
    checks: ['rate>0.99'],
  },
};

export default function () {
  const r = http.get(`${BASE}/api/admin/delegaciones`, { headers: { Authorization: `Bearer ${TOKEN}` } });
  check(r, { ok: (res) => res.status === 200 });
}
```

- [ ] **Step 3: Preparar la base y levantar la app**

```bash
S=$(mktemp -d /tmp/jax-frente-g-carga-XXXX); echo $S
ss -ltn | grep -c ':7798 ' || true
bash -c "$DBTEST; export JAX_KILL_SWITCH_PATH=$S/PAUSE; cd $WT && PYTHONPATH=.:las_manos $PY loadtest/sembrar_carga_delegacion.py preparar --n 1000"
```
Expected: `ss` da `0` y el sembrador imprime `aprobaciones sembradas: 1000`.

Levantar la app en background (`run_in_background`):
```bash
bash -c "$DBTEST; export JAX_KILL_SWITCH_PATH=$S/PAUSE; cd $WT && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/uvicorn loadtest.jacobs_delegacion_app:app --host 127.0.0.1 --port 7798 --log-level warning"
```
Verificar con `curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7798/jacobs/delegaciones/pendientes`. Expected: `200`.

- [ ] **Step 4: Medir a c=5, 10 y 25**

```bash
for VUS in 5 10 25; do
  for ESC in delegar3 delegar10 despachar pendientes; do
    /home/fruiz/bin/k6 run -e ESCENARIO=$ESC -e VUS=$VUS $WT/loadtest/jacobs_delegacion.js 2>&1 | tee $S/$ESC-$VUS.txt | grep -E 'http_req_duration|http_reqs|checks|http_req_failed'
  done
done
```
- Expected: cada corrida sale con código 0 y sus umbrales cumplidos.
- `despachar` corre después de las delegaciones del mismo `VUS`, así la cola está llena: es el peor caso de candidatos.

Registrar en `$L` una tabla con escenario, VUS, rps, p50, p95, p99, tasa de error y checks. Registrar también la concurrencia desde la que p95 más que duplica al de c=5.

Si un umbral falla, **no hay GO**:
1. Diagnosticar con `EXPLAIN ANALYZE` de la consulta `store.SQL_*` más lenta.
2. No instalar ningún perfilador sin preguntar.
3. Avisar a Fernando con el número.

- [ ] **Step 5: Apagar, limpiar y commit**

Detener uvicorn (TaskStop del proceso en background) y confirmar que `ss -ltn | grep -c ':7798 '` da `0`. Después:

```bash
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY loadtest/sembrar_carga_delegacion.py limpiar"
rm -rf $S
git -C $WT add loadtest/jacobs_delegacion_app.py loadtest/sembrar_carga_delegacion.py loadtest/jacobs_delegacion.js $L
git -C $WT commit -m "loadtest: carga aislada de delegar (3 y 10 hijos), despachar y pendientes

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git -C $WTP add loadtest/admin_delegaciones.js
git -C $WTP commit -m "loadtest: GET /api/admin/delegaciones (solo lectura, para producción)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: Revisión, CI (listas, pisos y canarios) y PRs de la etapa 1

**Files:**
- Modify: `.github/workflows/policy.yml` (jax): `tests-puros` y `jacobs-gobernanza-db`
- Modify: `$WTP/.github/workflows/policy.yml`: pisos de `backend-tests-con-db` y `frontend-tests`

**Interfaces:**
- Consumes: todos los tests de las Tasks 1-12 y los números de la Task 13.
- Produces: dos PRs en verde, con el canario visto en rojo sobre el sha real. La gemela de jax-platform se mergea primero.

> **Quién ejecuta:** el `push`, la creación del PR, el canario y el merge los corre el **controlador principal**. El hook de la sesión bloquea escrituras a git compartido desde sub-agentes (`CONTEXT.md`, deuda "sub-agentes sin gobernanza").

- [ ] **Step 1: Revisión independiente y respuesta a Fernando**

Pedir revisión de las dos ramas con `superpowers:requesting-code-review`: el ejecutor no juzga su propio gate. Todo hallazgo se arregla antes de mergear; nada se difiere.

Mandarle a Fernando un resumen en el chat:
- los números de la Task 13;
- los EXPLAIN de la Task 4;
- los valores de `.env` que van a producción (la tabla de Global Constraints, sin cambios);
- que el deploy reinicia `jax-platform` y después `jax-las-manos`;
- **la pregunta de la Discrepancia 17** (`tokens_sin_precio` informa y no bloquea: ¿lo confirma?).

El GO de deploy ya está dado (aprobación del spec, 2026-09-16). Si Fernando responde algo distinto sobre `tokens_sin_precio`, se implementa antes del merge, con su test en rojo.

- [ ] **Step 2: CI de jax**

En `.github/workflows/policy.yml`:

- `tests-puros`: agregar al final de **las dos** listas (la del `pytest -v` y la del piso, esta con ` \` de continuación):
  ```
  tests/test_config_delegacion.py tests/test_plan_delegacion_esquema.py
  tests/test_delegacion_modelos_puro.py tests/test_delegacion_uso_puro.py
  tests/test_delegacion_techos_puro.py tests/test_delegacion_politica_puro.py
  tests/test_delegacion_executor_puro.py
  ```
  Piso: el de la base rebasada más 69, **medido en el runner** y nunca sumado a mano. Línea de historia:
  `# +69 el 2026-09-16 (frente G): config de techos (17), esquema plan_delegacion.v1 (25), modelos (4), uso con raíz (5), techos (12), política y arranque (3), executor delegate (8). Vistos en rojo; guardas por mutación.`
- `jacobs-gobernanza-db`: agregar `jacobs/_delegacion_io_test.py tests/test_delegacion_arbol_e2e.py` a las dos listas. Piso: el de la base más 61 (55 + 6), también medido en el runner. Línea de historia:
  `# +61 el 2026-09-16 (frente G): _delegacion_io_test.py (55: esquema, consultas, EXPLAIN con volumen, uso transaccional, delegar, cola, árbol, ejecución, aprobación) + test_delegacion_arbol_e2e.py (6: matriz del spec §6). Requiere jax-platform con las capabilities delegate/integrate en master.`
- Si el `env:` de `jacobs-gobernanza-db` no trae `JAX_KILL_SWITCH_PATH` (lo agrega B donde lo exige), agregar `JAX_KILL_SWITCH_PATH: /tmp/jax-ci-sin-freno/PAUSE`.

Antes de subir, medir las dependencias en un venv limpio, igual que F:
```bash
S=$(mktemp -d /tmp/jax-frente-g-venv-XXXX)
python3.12 -m venv $S/v && $S/v/bin/pip -q install -r $WTP/backend/requirements.txt pytest
cd $WT && PYTHONPATH=.:las_manos $S/v/bin/python -m pytest --collect-only -q jacobs/_delegacion_io_test.py tests/test_delegacion_arbol_e2e.py 2>&1 | tail -2
rm -rf $S
```
Expected: `61 tests collected`, sin errores de colección.

- [ ] **Step 3: CI de jax-platform**

En `$WTP/.github/workflows/policy.yml`:
- piso de `backend-tests-con-db`: +12 (`test_capabilities_delegacion.py` 4 + `test_admin_delegaciones.py` 8), medido;
- piso de `frontend-tests`: +7 (`AdminDelegaciones.test.jsx`), medido con `npx vitest run --reporter=json` como el paso existente.

Cada uno con su línea de historia fechada. Commit en `$WTP`.

- [ ] **Step 4: jax-platform — rebase, subida, PR, canario y merge (controlador)**

```bash
git -C $WTP fetch origin && git -C $WTP rebase origin/master
cd $WTP/backend && JAX_REPO_PATH=$WT $PYP -m pytest -q 2>&1 | tail -1
cd $WTP/frontend && npx vitest run 2>&1 | tail -2
git -C $WTP push -u origin feat/emisor-subpipelines-ada
gh pr create --repo fjruizhn/jax-platform --base master --head feat/emisor-subpipelines-ada \
  --title "Emisor de sub-pipelines de Ada: capabilities delegate/integrate y Admin → Delegaciones" \
  --body-file $WT/.superpowers/sdd/2026-09-16-emisor-subpipelines-ada/pr-plataforma.md
```

Antes de crear el PR, escribir `pr-plataforma.md` con:
- el propósito;
- las Discrepancias 10 y 19;
- los tests vistos en rojo;
- el orden de deploy (esta rama antes que la de jax);
- la línea final `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

Esperar los checks sobre el sha real (Monitor sobre `gh api repos/fjruizhn/jax-platform/commits/$SHA/check-runs --jq '.check_runs[] | [.name, .status, .conclusion] | @tsv'`). Expected: todos `completed success`.

**Canario:**
1. En `frontend/src/pages/admin/AdminDelegaciones.jsx`, cambiar `Math.ceil` por `Math.floor`.
2. Commit `canario: valoresIniciales redondea hacia abajo (debe poner rojo frontend-tests)` y subirlo.
3. Esperar `frontend-tests completed failure` y verificar en `gh run view --log-failed` que cae `valoresIniciales prellena…`.
4. Revertir con `git revert --no-edit HEAD`, subir y esperar todo en `success`.
5. Pegar los dos sha en `$L` y en el PR.

Merge: `gh pr merge --repo fjruizhn/jax-platform feat/emisor-subpipelines-ada --merge`.

- [ ] **Step 5: jax — rebase, subida, PR, canario y merge (controlador)**

Con la gemela ya en master (el job `jacobs-gobernanza-db` clona master):

```bash
git -C $WT fetch origin && git -C $WT rebase origin/master
cd $WT && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_delegacion_executor_puro.py tests/test_delegacion_politica_puro.py 2>&1 | tail -1
git -C $WT push -u origin feat/emisor-subpipelines-ada
gh pr create --repo fjruizhn/Jax --base master --head feat/emisor-subpipelines-ada \
  --title "Emisor de sub-pipelines de Ada (modo 1: plan de delegación)" \
  --body-file $WT/.superpowers/sdd/2026-09-16-emisor-subpipelines-ada/pr-jax.md
```

`pr-jax.md` lleva:
- el propósito;
- las Discrepancias 1-19;
- la matriz de pruebas con lo visto en rojo;
- las mutaciones;
- la tabla de carga de la Task 13;
- los EXPLAIN;
- el orden de deploy;
- la línea final `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

Esperar los checks sobre el sha real. Expected: `tests-puros`, `jacobs-gobernanza-db`, `output-validator-regression`, `subpipeline-contrato-db` y el resto en `success`.

**Canario:**
1. En `jacobs/store.py::pipeline_fijar_cola`, borrar `AND queued_at IS NULL`.
2. Commit `canario: fijar_cola pisa queued_at ya fijado (debe poner rojo jacobs-gobernanza-db)` y subirlo.
3. Esperar `jacobs-gobernanza-db completed failure` y verificar en el log que cae `test_fijar_cola_solo_toca_los_null_en_queued`.
4. `git revert --no-edit HEAD`, subir y todo `success`.
5. Pegar los sha en `$L` y en el PR.

Merge: `gh pr merge --repo fjruizhn/Jax feat/emisor-subpipelines-ada --merge`.

---

### Task 15: Deploy de la etapa 1 con 0 pipelines en vuelo (controlador)

**Files:**
- Modify (producción, con `sudo`): `/etc/jax/.env`

**Interfaces:**
- Consumes: los dos PRs mergeados (Task 14).
- Produces: jax-platform con las capabilities y la pantalla; LAS MANOS con el emisor, la cola y las variables. Todo verificado en vivo.

- [ ] **Step 1: Cero en vuelo y backups verificados**

```bash
S=$(mktemp -d /tmp/jax-frente-g-deploy-XXXX)
cat > $S/en_vuelo.py <<'PY'
import asyncio
from jacobs import store
async def main():
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT DATABASE()")
            print("base:", (await cur.fetchone())[0])
            await cur.execute("SELECT pipeline_id, status, name FROM jacobs_pipelines WHERE status IN "
                              "('pending','running','queued','waiting_children','awaiting_approval')")
            print("en vuelo:", await cur.fetchall())
    finally:
        conn.close()
asyncio.run(main())
PY
bash -c 'set -a; . /etc/jax/.env; set +a; cd /home/fruiz/jax && PYTHONPATH=.:las_manos las_manos/.venv/bin/python "$1"' _ $S/en_vuelo.py
```
Expected: `base: jax_memory` y `en vuelo: ()`. Si hay alguno, esperar a que termine y repetir: **nunca se reinicia con un pipeline corriendo**.

```bash
cat > $S/backup.sh <<'SH'
set -euo pipefail
set -a; . /etc/jax/.env; set +a
B=~/backups/pre-emisor-ada-$(date +%F-%H%M).sql
( umask 077; mysqldump --single-transaction --skip-lock-tables --no-tablespaces \
    -h$JAX_DB_HOST -P$JAX_DB_PORT -u$JAX_DB_USER -p$JAX_DB_PASSWORD $JAX_DB_NAME \
    jacobs_pipelines jacobs_steps capability capability_motor > "$B" )
echo "backup: $B"
sed -e 's/`jacobs_pipelines`/`g_restore_pipelines`/g' -e 's/`jacobs_steps`/`g_restore_steps`/g' \
    -e 's/`capability_motor`/`g_restore_capability_motor`/g' -e 's/`capability`/`g_restore_capability`/g' \
    -e 's/CONSTRAINT `\([A-Za-z0-9_]*\)`/CONSTRAINT `g_restore_\1`/g' "$B" > "$B.probe"
mysql -h$JAX_DB_HOST -P$JAX_DB_PORT -u$JAX_DB_USER -p$JAX_DB_PASSWORD jax_memory_test < "$B.probe"
mysql -h$JAX_DB_HOST -P$JAX_DB_PORT -u$JAX_DB_USER -p$JAX_DB_PASSWORD -N -e "
  SELECT 'pipelines', (SELECT COUNT(*) FROM jax_memory.jacobs_pipelines), (SELECT COUNT(*) FROM jax_memory_test.g_restore_pipelines);
  SELECT 'steps', (SELECT COUNT(*) FROM jax_memory.jacobs_steps), (SELECT COUNT(*) FROM jax_memory_test.g_restore_steps);
  SELECT 'capability', (SELECT COUNT(*) FROM jax_memory.capability), (SELECT COUNT(*) FROM jax_memory_test.g_restore_capability);
  SELECT 'capability_motor', (SELECT COUNT(*) FROM jax_memory.capability_motor), (SELECT COUNT(*) FROM jax_memory_test.g_restore_capability_motor);"
mysql -h$JAX_DB_HOST -P$JAX_DB_PORT -u$JAX_DB_USER -p$JAX_DB_PASSWORD jax_memory_test -e \
  "DROP TABLE g_restore_pipelines, g_restore_steps, g_restore_capability, g_restore_capability_motor"
rm -f "$B.probe"
SH
bash $S/backup.sh
TS=$(date +%F-%H%M); sudo cp -a /etc/jax/.env /etc/jax/.env.pre-emisor-ada-$TS && sudo ls -l /etc/jax/.env.pre-emisor-ada-$TS
```
Expected: en cada fila las dos cifras coinciden y la copia del `.env` existe. **Un backup sin restauración probada no es backup:** si una cifra no coincide o el `.probe` falla al cargar, parar y diagnosticar.

La prueba renombra tablas **y** restricciones. InnoDB exige nombres de FK únicos en toda la base, y `jax_memory_test` ya tiene las FK originales de `capability`/`capability_motor`. Las referencias a `motor` quedan igual, porque esa tabla existe en `jax_memory_test` gracias a las migraciones.

- [ ] **Step 2: jax-platform (migraciones al arrancar)**

```bash
git -C /home/fruiz/jax-platform status --short | head
git -C /home/fruiz/jax-platform pull --ff-only && git -C /home/fruiz/jax-platform log --oneline -1
sudo systemctl restart jax-platform.service && systemctl is-active jax-platform.service
sudo -n /usr/bin/journalctl -u jax-platform.service --since '-3 min' --no-pager | grep -iE "traceback|error" || echo "sin errores en el arranque"
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/api/health
( set -a; . /etc/jax/.env; set +a
  mariadb -h "$JAX_DB_HOST" -P "$JAX_DB_PORT" -u "$JAX_DB_USER" -p"$JAX_DB_PASSWORD" jax_memory -N -e \
  "SELECT \`key\`, output_schema, allowed_callers, mode, max_execution_minutes FROM capability WHERE \`key\` IN ('delegate','integrate')" )
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/api/admin/delegaciones
```
Expected:
- `status` limpio en lo trackeado;
- el merge en el `log`;
- `active` y "sin errores en el arranque";
- `/api/health` responde `200`;
- dos filas: `delegate plan_delegacion.v1 ["jacobs"] read_only 15` y `integrate NULL ["jacobs"] read_only 15`;
- la ruta nueva sin token da `401`.

- [ ] **Step 3: Frontend**

```bash
cd /home/fruiz/jax-platform/frontend && pwd && npm run build && ls dist/assets/index-*.js
set -a; . /etc/jax/.env; set +a
ssh -p "$JAX_SSH_PORT" "$JAX_SSH_USER@172.16.20.11" "B=/www/wwwroot/axioma-ia.io.backup-pre-emisor-ada-$(date +%Y%m%d-%H%M%S); sudo cp -a /www/wwwroot/axioma-ia.io \$B && sudo diff -rq /www/wwwroot/axioma-ia.io \$B && echo backup-ok \$B"
rsync -a --delete --exclude .user.ini -e "ssh -p $JAX_SSH_PORT" /home/fruiz/jax-platform/frontend/dist/ "$JAX_SSH_USER@172.16.20.11:/tmp/axioma-deploy/"
ssh -p "$JAX_SSH_PORT" "$JAX_SSH_USER@172.16.20.11" "sudo rsync -a --delete --exclude .user.ini --chown=www:www /tmp/axioma-deploy/ /www/wwwroot/axioma-ia.io/"
curl -s https://axioma-ia.io/ | grep -o 'index-[A-Za-z0-9_-]*\.js'
```
Expected: `backup-ok` con su nombre, y el `index-*.js` servido igual al recién construido. Anotar el hash y el nombre del backup.

- [ ] **Step 4: Variables y LAS MANOS**

```bash
sudo grep -cE '^JAX_ADA_' /etc/jax/.env || true
printf '\n# Emisor de sub-pipelines de Ada (frente G, 2026-09-16)\nJAX_ADA_MAX_HIJOS_POR_STEP=10\nJAX_ADA_MAX_PIPELINES_ARBOL=40\nJAX_ADA_MAX_TOKENS_ARBOL=2000000\nJAX_ADA_MAX_USD_ARBOL=10\nJAX_ADA_RESERVA_TOKENS_HIJO=50000\nJAX_ADA_COLA_MAX_ESPERA_SEGUNDOS=3600\nJAX_ADA_COLA_CADENCIA_SEGUNDOS=10\nJAX_ADA_RESUMEN_HIJO_CARACTERES=2000\n' | sudo tee -a /etc/jax/.env >/dev/null
sudo grep -nE '^JAX_ADA_' /etc/jax/.env
sudo grep -c '^JAX_ADA_MAX_DEPTH=' /etc/jax/.env || true
```
Expected: el primer `grep` da `0`, después aparecen exactamente las 8 líneas y `JAX_ADA_MAX_DEPTH` da `0`. Si el primero no da 0, no se agregan duplicados: se revisa qué hay.

Repetir el chequeo de 0 en vuelo del Step 1 inmediatamente antes de reiniciar:
```bash
bash -c 'set -a; . /etc/jax/.env; set +a; cd /home/fruiz/jax && PYTHONPATH=.:las_manos las_manos/.venv/bin/python "$1"' _ $S/en_vuelo.py
git -C /home/fruiz/jax pull --ff-only && git -C /home/fruiz/jax log --oneline -1
sudo systemctl restart jax-las-manos.service && systemctl is-active jax-las-manos.service
journalctl -u jax-las-manos.service --since "-3 min" --no-pager | grep -iE "error|traceback|ValueError|delegaci|cola" || echo "sin errores en el arranque"
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7777/health
curl -s http://127.0.0.1:7777/jacobs/delegaciones/pendientes
```
Expected: `en vuelo: ()`, el merge en el `log`, `active`, "sin errores en el arranque", `200` y `{"aprobaciones":[]}`.

- [ ] **Step 5: Esquema, índices y EXPLAIN en producción**

```bash
cat > $S/verificar.py <<'PY'
import asyncio
from jacobs import store
async def main():
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT DATABASE()"); print("base:", (await cur.fetchone())[0])
            for t in ("jacobs_arbol_consumo", "jacobs_delegacion_aprobaciones"):
                await cur.execute("SELECT COUNT(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s", (t,))
                print(t, (await cur.fetchone())[0])
            await cur.execute("SELECT INDEX_NAME, GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) FROM information_schema.STATISTICS "
                              "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='jacobs_pipelines' AND INDEX_NAME LIKE 'idx_jacobs_pipelines_%' GROUP BY INDEX_NAME")
            print("indices:", await cur.fetchall())
            await cur.execute("SELECT COUNT(*) FROM jacobs_pipelines"); print("pipelines:", (await cur.fetchone())[0])
            for nombre in ("SQL_COLA_CANDIDATOS", "SQL_HIJOS", "SQL_ARBOL_CONTAR"):
                args = (10,) if nombre == "SQL_COLA_CANDIDATOS" else ("x",)
                await cur.execute("EXPLAIN " + getattr(store, nombre), args)
                print(nombre, await cur.fetchall())
    finally:
        conn.close()
asyncio.run(main())
PY
bash -c 'set -a; . /etc/jax/.env; set +a; cd /home/fruiz/jax && PYTHONPATH=.:las_manos las_manos/.venv/bin/python "$1"' _ $S/verificar.py
```
Expected: `base: jax_memory`; las dos tablas con `1`; los índices `padre`, `raiz` (`root_pipeline_id,status`), `cola` (`status,queued_at`) y `duenio`. Los EXPLAIN se pegan en `$L` **tal cual**: con 0 filas en cola MariaDB puede cortar con "Impossible WHERE" y eso no dice nada del plan. La verificación válida con volumen es la de la Task 4, y queda dicho así. Si un índice falta, buscar en el journal el `ERROR` de `_crear_indice_acotado` (espera de lock vencida): el próximo arranque lo reintenta, pero **no se sigue** a la Task 16 sin los índices.

---

### Task 16: Verificación en vivo con un árbol real chico y techos bajados (controlador + Fernando)

**Files:**
- Modify (producción, temporal): `/etc/jax/.env` (`JAX_ADA_MAX_HIJOS_POR_STEP=2` y restauración)

**Interfaces:**
- Consumes: el deploy de la Task 15. Gasta: 1 llamada de Ada para delegar, 3 de jekyll y 1 de Ada para integrar (GO de Fernando, spec §6).
- Produces: evidencia en vivo del ciclo completo: exceso → Telegram → Admin → aprobación → cola → integración → completado, con el consumo del árbol cuadrado contra `axioma_usage`.

- [ ] **Step 1: Bajar el techo a propósito**

Con 0 en vuelo (script del Task 15, Step 1):
```bash
sudo sed -i 's/^JAX_ADA_MAX_HIJOS_POR_STEP=10$/JAX_ADA_MAX_HIJOS_POR_STEP=2/' /etc/jax/.env
sudo grep -n '^JAX_ADA_MAX_HIJOS_POR_STEP=' /etc/jax/.env
sudo systemctl restart jax-las-manos.service && systemctl is-active jax-las-manos.service
```
Expected: `JAX_ADA_MAX_HIJOS_POR_STEP=2` y `active`.

- [ ] **Step 2: Lanzar el árbol**

Leer la identidad de Fernando (sólo lectura):
```bash
( set -a; . /etc/jax/.env; set +a
  mariadb -h "$JAX_DB_HOST" -P "$JAX_DB_PORT" -u "$JAX_DB_USER" -p"$JAX_DB_PASSWORD" jax_memory -N -e \
  "SELECT user_id, tenant_id, email FROM jax_users WHERE role='superadmin' ORDER BY user_id LIMIT 1" )
```
Confirmar con Fernando que es su cuenta. Con esos valores en `UID_F` y `TID_F`:

```bash
INICIO=$(date +%s)
cat > $S/arbol.json <<JSON
{"name": "verificacion-frente-g", "objective": "Árbol chico de verificación del emisor de Ada",
 "invoked_by": "plataforma", "user_id": "$UID_F", "tenant_id": "$TID_F", "mode": "autonomous",
 "steps": [{"facet": "ada", "capability": "delegate",
   "prompt": "Delegá exactamente 3 sub-objetivos independientes, cada uno a la faceta jekyll con capability analysis: (a) explicar en 3 líneas qué es un kill switch en un sistema de agentes; (b) explicar en 3 líneas qué es una cola FIFO; (c) explicar en 3 líneas qué es un árbol de dependencias. Integración: unir los tres textos en un párrafo breve."}]}
JSON
curl -s -X POST http://127.0.0.1:7777/jacobs/pipeline -H 'Content-Type: application/json' -d @$S/arbol.json | tee $S/creado.json
PADRE=$(python3 -c "import json;print(json.load(open('$S/creado.json'))['pipeline_id'])")
( set -a; . /etc/jax/.env; set +a
  mariadb -h "$JAX_DB_HOST" -P "$JAX_DB_PORT" -u "$JAX_DB_USER" -p"$JAX_DB_PASSWORD" jax_memory -e \
  "UPDATE jacobs_pipelines SET owner_ack_at=UNIX_TIMESTAMP() WHERE pipeline_id='$PADRE' AND owner_ack_at IS NULL" )
```
El `UPDATE` de `owner_ack_at` es el mismo que hace jax-platform al crear (`api/pipelines.py::_record_pipeline_owner`). Se hace porque el pedido entró directo a Jacobs, y así el árbol aparece en la Mesa de Fernando. Queda declarado en la Biblioteca.

Vigilar en sólo lectura hasta `awaiting_approval` (Monitor, sin `sleep` en primer plano):
```bash
( set -a; . /etc/jax/.env; set +a
  mariadb -h "$JAX_DB_HOST" -P "$JAX_DB_PORT" -u "$JAX_DB_USER" -p"$JAX_DB_PASSWORD" jax_memory -e \
  "SELECT status FROM jacobs_pipelines WHERE pipeline_id='$PADRE';
   SELECT event_type, payload FROM jacobs_events WHERE pipeline_id='$PADRE' ORDER BY id" )
```
Expected:
- `awaiting_approval`;
- eventos `PIPELINE_CREATED`, `PIPELINE_STARTED`, `STEP_COMPLETED`, `DELEGACION_PROPUESTA` y `DELEGACION_EXCEDE_TECHO` (con `hijos_por_step: 3 > 2`);
- 0 filas en `jacobs_subpipeline_tokens` para `$PADRE`;
- Fernando recibe el Telegram.

Si el step `delegate` falla dos veces por plan inválido, **eso también es un resultado**: se anota con los dos `DELEGACION_PLAN_INVALIDO` y se le pregunta a Fernando si repetir. No se corrige el prompt a escondidas.

- [ ] **Step 3: Fernando aprueba en el Admin**

Fernando abre **Admin → Delegaciones** y revisa, en claro y en oscuro, en es y en en:
- la fila con el tipo "Excede un techo al delegar";
- el techo "Sub-pipelines por paso 3 (límite 2)".

Pulsa **Aprobar**. En el diálogo, Escape cierra y el foco vuelve al botón; lo reabre, deja `3` y confirma. Aparece el toast "Delegación aprobada".

Vigilar hasta `completed`:
```bash
( set -a; . /etc/jax/.env; set +a
  mariadb -h "$JAX_DB_HOST" -P "$JAX_DB_PORT" -u "$JAX_DB_USER" -p"$JAX_DB_PASSWORD" jax_memory -e \
  "SELECT pipeline_id, parent_pipeline_id, depth, status, queued_at FROM jacobs_pipelines WHERE root_pipeline_id='$PADRE' ORDER BY created_at;
   SELECT estado, resuelta_por, FROM_UNIXTIME(resuelta_at) FROM jacobs_delegacion_aprobaciones WHERE root_pipeline_id='$PADRE';
   SELECT COUNT(*), SUM(usado_at IS NOT NULL) FROM jacobs_subpipeline_tokens WHERE parent_pipeline_id='$PADRE';
   SELECT pipelines_total, tokens_total, costo_usd_total, tokens_sin_precio, consumo_incompleto, techo_hijos_por_step_aprobado FROM jacobs_arbol_consumo WHERE root_pipeline_id='$PADRE';
   SELECT SUM(tokens_in + tokens_out), SUM(cost_usd) FROM axioma_usage WHERE request_type='pipeline' AND user_id=$UID_F AND UNIX_TIMESTAMP(created_at) >= $INICIO;" )
```
Expected:
- 4 filas en el árbol: el padre `completed` y 3 hijos `completed` con `depth=1`;
- la aprobación `aprobada` con el `user_id` de Fernando y la hora;
- tokens `3, 3`;
- consumo con `pipelines_total=4` y `techo_hijos_por_step_aprobado=3`;
- **`tokens_total` igual a la suma de `axioma_usage` de la ventana**, y lo mismo con el costo (con precio). Es la prueba en vivo de la misma transacción. Si Fernando usó otro pipeline en la ventana, la suma de `axioma_usage` es mayor: se anota y se filtra por `facet IN ('ada','jekyll')`.

Pegar también el resultado de la integración: `curl -s http://127.0.0.1:7777/jacobs/pipeline/$PADRE/results`. El último step (`integrate`) tiene que nombrar los tres sub-objetivos.

- [ ] **Step 4: Restaurar el techo**

Con 0 en vuelo:
```bash
sudo sed -i 's/^JAX_ADA_MAX_HIJOS_POR_STEP=2$/JAX_ADA_MAX_HIJOS_POR_STEP=10/' /etc/jax/.env
sudo grep -n '^JAX_ADA_MAX_HIJOS_POR_STEP=' /etc/jax/.env
sudo diff <(sudo grep -v '^JAX_ADA_' /etc/jax/.env) <(sudo grep -v '^JAX_ADA_' /etc/jax/.env.pre-emisor-ada-$TS) && echo "resto del .env intacto"
sudo systemctl restart jax-las-manos.service && systemctl is-active jax-las-manos.service
```
Expected: `=10`, "resto del .env intacto" y `active`.

- [ ] **Step 5: k6 del proxy en producción (sólo lectura)**

Fernando pasa un access token de superadmin para esta ventana:
```bash
for VUS in 5 10 25; do /home/fruiz/bin/k6 run -e TOKEN=<token> -e VUS=$VUS /home/fruiz/jax-platform/loadtest/admin_delegaciones.js 2>&1 | grep -E 'http_req_duration|http_reqs|checks'; done
```
Expected: exit 0 y umbrales en verde. Anotar rps y p95 por concurrencia. El token no se guarda en ningún archivo ni en `$L`. Después: `rm -rf $S`.

---

### Task 17: Biblioteca de la etapa 1

**Files:**
- Modify: `/home/fruiz/jax/DEUDA.md`, `/home/fruiz/jax/CONTEXT.md` (rama `docs/emisor-subpipelines-ada`, PR)
- Create: `/home/fruiz/.claude/projects/-home-fruiz/memory/jax-emisor-subpipelines-ada-2026-09-16.md` y su línea en `MEMORY.md`

- [ ] **Step 1: Entradas**

En `DEUDA.md`, en la sección de cerrados más reciente, agregar el bloque **"Emisor de sub-pipelines de Ada, modo 1 (frente G) — CERRADO <fecha>"** con estos puntos:
- **DECISIÓN (Fernando, 2026-09-16):** dos modos permanentes; Ada lanza sola dentro de techos y lo que excede pide aprobación. Más la respuesta sobre `tokens_sin_precio` (Discrepancia 17).
- **HISTORIA:**
  - las Discrepancias 1-19, cada una con su resolución;
  - el deadlock de cupo evitado con `waiting_children`;
  - la cancelación que no detenía un run (Discrepancia 13), visto en rojo.
- **VERDAD OPERACIONAL** (con fecha, comando y evidencia):
  - capabilities en `jax_memory`;
  - tablas, índices y variables;
  - `index-*.js` servido y nombre del backup;
  - el árbol real de la Task 16 con sus 4 pipelines;
  - el consumo cuadrado contra `axioma_usage`.
- **Carga:**
  - la tabla de la Task 13 (escenario, VUS, rps, p95, p99) y el punto de degradación;
  - el k6 del proxy en producción;
  - la nota: *caduca si cambia el esquema, el volumen o el límite global de pipelines*.
- **EXPLAIN:** los de la Task 4 (con volumen) y los de producción tal cual.
- **CI:** listas, pisos y los sha de los canarios (rojo) y de los reverts (verde), en los dos repos.
- **PENDIENTE con fecha:**
  - la etapa 2 (medición de GLM), **hoy** (Task 18 en adelante);
  - borrar el backup del frontend en la VM el día que Fernando dé por buena la verificación de la Task 16 (fecha concreta, anotada).

En `CONTEXT.md` §9, una entrada de 6 a 10 líneas que remita a `DEUDA.md` para el detalle. Las lecciones:
- un límite global con padres que esperan se traba si esperar ocupa cupo;
- "cancelado" no es "detenido" si el runner pisa el estado;
- la misma transacción se prueba cuadrando contra la tabla de cobro.

- [ ] **Step 2: PR de docs (controlador)**

```bash
git -C /home/fruiz/jax fetch origin
git -C /home/fruiz/jax worktree add /home/fruiz/worktrees/jax-docs-frente-g -b docs/emisor-subpipelines-ada origin/master
```
Editar en ese worktree y después:
```bash
git -C /home/fruiz/worktrees/jax-docs-frente-g add DEUDA.md CONTEXT.md
git -C /home/fruiz/worktrees/jax-docs-frente-g commit -m "docs: emisor de sub-pipelines de Ada, modo 1 cerrado y desplegado (frente G)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git -C /home/fruiz/worktrees/jax-docs-frente-g push -u origin docs/emisor-subpipelines-ada
gh pr create --repo fjruizhn/Jax --base master --head docs/emisor-subpipelines-ada --title "docs: emisor de sub-pipelines de Ada (frente G, modo 1)" --body "Registro en la Biblioteca: evidencia en vivo, carga, EXPLAIN, CI y pendientes con fecha.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```
Esperar el CI en verde, mergear y después `git -C /home/fruiz/jax pull --ff-only`.

- [ ] **Step 3: Memoria**

Crear el archivo de memoria con firma y tipo (DECISIÓN, HISTORIA, VERDAD OPERACIONAL con fecha, PENDIENTE con fecha) y agregar su línea a `MEMORY.md`. El ledger `$L` **sigue** en el worktree hasta cerrar las etapas 2 y 3 (Task 24).

---

# ETAPA 2 · Medición de tool-calling de GLM (gate) y migración de Ada al camino gobernado

> **Gate de la etapa.** La medición (Task 19) va **primero**: sin ella, la migración no tiene propósito (Discrepancia 15).
> - **Si la medición no pasa**, las Tasks 20-23 **no se ejecutan**. Se registra con fecha de revisión y se salta a la Task 24.
> - **Si pasa**, siguen las Tasks 20 y 21 (migración) y la etapa 3.

### Task 18: Declaración de las tools de sub-pipelines y script de medición

**Files:**
- Create: `las_manos/motor_registry/tools_subpipelines.py`
- Create: `scripts/medir_tool_calling_glm.py`
- Create: `tests/test_tools_subpipelines.py`
- Create: `tests/test_medicion_glm.py`

**Interfaces:**
- Produces:
  - `tools_subpipelines.TOOL_LANZAR = "lanzar_subpipeline"`, `TOOL_LEER = "leer_resultado_subpipeline"`, `NOMBRES: frozenset[str]`
  - `TOOLS_SUBPIPELINES: list[dict]` (forma OpenAI `tools`, `additionalProperties: false`)
  - `validar_argumentos(nombre: str, argumentos_json: str) -> tuple[dict | None, str | None]`. Para `lanzar_subpipeline` devuelve `{objetivo, faceta, capability, depende_de: list[str], skip_on_fail: bool}` con los valores por defecto aplicados; para `leer_resultado_subpipeline`, `{pipeline_id}`.
  - `scripts/medir_tool_calling_glm.py`: `PROMPTS: list[str]` (5), `UMBRAL_VALIDAS = 4`, `calificar(respuesta: dict, nombres_declarados: frozenset[str]) -> dict` con `{tool_calls, inventadas, valida, motivos}`, `resumir(calificaciones: list[dict]) -> dict` con `{llamadas, validas, invenciones, umbral_validas, pasa}`, `medir(salida: Path) -> dict` (async, gasta), `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Tests que fallan**

Create `tests/test_tools_subpipelines.py`:

```python
"""Frente G (2026-09-16): tools lanzar_subpipeline y leer_resultado_subpipeline.
Declaración y validación estricta de argumentos (puro). La misma validación
califica la medición de GLM (scripts/medir_tool_calling_glm.py) y autoriza la
ejecución en tool_authority (etapa 3).

ROJO CONTRA LA BASE: ModuleNotFoundError motor_registry.tools_subpipelines."""
from __future__ import annotations

import json

import pytest

from motor_registry import tools_subpipelines as ts


def test_declara_las_dos_tools_con_forma_openai():
    nombres = [t["function"]["name"] for t in ts.TOOLS_SUBPIPELINES]
    assert nombres == ["lanzar_subpipeline", "leer_resultado_subpipeline"]
    for t in ts.TOOLS_SUBPIPELINES:
        assert t["type"] == "function"
        assert t["function"]["parameters"]["additionalProperties"] is False
    assert ts.NOMBRES == frozenset(nombres)


def test_lanzar_valido_aplica_defaults():
    args, motivo = ts.validar_argumentos("lanzar_subpipeline",
                                         json.dumps({"objetivo": "x", "faceta": "jekyll", "capability": "analysis"}))
    assert motivo is None
    assert args == {"objetivo": "x", "faceta": "jekyll", "capability": "analysis",
                    "depende_de": [], "skip_on_fail": False}


def test_leer_valido():
    assert ts.validar_argumentos("leer_resultado_subpipeline", '{"pipeline_id": "abc"}') == ({"pipeline_id": "abc"}, None)


@pytest.mark.parametrize("nombre, argumentos, fragmento", [
    ("lanzar_subpipeline", "no json", "no es JSON"),
    ("lanzar_subpipeline", "[]", "objeto JSON"),
    ("lanzar_subpipeline", '{"objetivo": "x", "faceta": "j"}', "faltan ['capability']"),
    ("lanzar_subpipeline", '{"objetivo": "x", "faceta": "j", "capability": "c", "extra": 1}', "no permitidos ['extra']"),
    ("lanzar_subpipeline", '{"objetivo": " ", "faceta": "j", "capability": "c"}', "objetivo"),
    ("lanzar_subpipeline", '{"objetivo": "x", "faceta": "j", "capability": "c", "skip_on_fail": "si"}', "skip_on_fail"),
    ("lanzar_subpipeline", '{"objetivo": "x", "faceta": "j", "capability": "c", "depende_de": "p1"}', "depende_de"),
    ("leer_resultado_subpipeline", "{}", "faltan ['pipeline_id']"),
    ("read_file", '{"path": "a"}', "no es una tool de sub-pipelines"),
])
def test_invalidos(nombre, argumentos, fragmento):
    args, motivo = ts.validar_argumentos(nombre, argumentos)
    assert args is None
    assert fragmento in motivo
```

Create `tests/test_medicion_glm.py`:

```python
"""Frente G (2026-09-16): calificación determinista de la medición de
tool-calling de GLM (spec §6: pasa si >= 4/5 llamadas producen tool_calls
válidos con argumentos que validan el schema, y 0 invenciones de nombres).
Puro: no llama a ninguna API.

ROJO CONTRA LA BASE: el script no existe."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("medir_glm", RAIZ / "scripts" / "medir_tool_calling_glm.py")
medir = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(medir)

NOMBRES = frozenset({"read_file", "write_file", "lanzar_subpipeline", "leer_resultado_subpipeline"})


def _resp(*llamadas):
    return {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": a}}
        for i, (n, a) in enumerate(llamadas)]}}]}


def test_cinco_prompts():
    assert len(medir.PROMPTS) == 5 and medir.UMBRAL_VALIDAS == 4


def test_llamada_valida():
    c = medir.calificar(_resp(("lanzar_subpipeline", '{"objetivo": "x", "faceta": "hipatia", "capability": "research"}')), NOMBRES)
    assert (c["valida"], c["tool_calls"], c["inventadas"]) == (True, 1, [])


def test_nombre_inventado_no_es_valida_y_se_cuenta():
    c = medir.calificar(_resp(("crear_subtarea", '{"objetivo": "x"}')), NOMBRES)
    assert c["valida"] is False and c["inventadas"] == ["crear_subtarea"]


def test_argumentos_que_no_validan():
    c = medir.calificar(_resp(("lanzar_subpipeline", '{"objetivo": "x", "faceta": "hipatia"}')), NOMBRES)
    assert c["valida"] is False and "faltan" in c["motivos"][0]


def test_sin_tool_calls_no_es_valida():
    c = medir.calificar({"choices": [{"message": {"content": "no uso herramientas"}}]}, NOMBRES)
    assert c["valida"] is False and c["motivos"] == ["sin tool_calls"]


@pytest.mark.parametrize("validas, inventadas, pasa", [(5, 0, True), (4, 0, True), (3, 0, False), (5, 1, False)])
def test_criterio_del_spec(validas, inventadas, pasa):
    calificaciones = [{"valida": i < validas, "inventadas": ["x"] if i < inventadas else []} for i in range(5)]
    assert medir.resumir(calificaciones)["pasa"] is pasa


def test_se_niega_sin_confirmar_el_gasto(tmp_path):
    assert medir.main(["--salida", str(tmp_path)]) == 2


def test_se_niega_a_guardar_dentro_del_repo():
    assert medir.main(["--salida", str(RAIZ / "mediciones"), "--confirmo-gasto"]) == 2
```

```bash
cd $WT && pwd && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_tools_subpipelines.py tests/test_medicion_glm.py 2>&1 | tee -a $L | tail -4
```
Expected: errores de colección (`ModuleNotFoundError` y `FileNotFoundError` del script).

- [ ] **Step 2: Implementar las tools**

Create `las_manos/motor_registry/tools_subpipelines.py`:

```python
"""
LAS MANOS — Motor Registry: tools de sub-pipelines de Ada (frente G, 2026-09-16).

Spec §3.2: lanzar_subpipeline(objetivo, faceta, capability, depende_de,
skip_on_fail) y leer_resultado_subpipeline(pipeline_id). Acá vive SOLO la
declaración (forma OpenAI `tools`) y la validación estricta de argumentos: la
autoridad la resuelve tool_authority.py en cada llamada, y la decisión (techos,
token, cola) la toma jacobs/delegacion.py, la MISMA del modo 1.

Las propiedades válidas salen de la declaración: una sola fuente.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import json

TOOL_LANZAR = "lanzar_subpipeline"
TOOL_LEER = "leer_resultado_subpipeline"
LARGO_MAX_OBJETIVO = 4000
LARGO_MAX_NOMBRE = 50
LARGO_MAX_ID = 36
MAX_DEPENDENCIAS = 50

TOOLS_SUBPIPELINES: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_LANZAR,
            "description": (
                "Lanza un sub-pipeline hijo con UN sub-objetivo para una faceta. Devuelve su "
                "pipeline_id y su estado: encolado, o pendiente de aprobación si excede un techo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "objetivo": {"type": "string", "description": "Tarea autocontenida del sub-pipeline."},
                    "faceta": {"type": "string", "description": "Faceta activa que lo ejecuta."},
                    "capability": {"type": "string", "description": "Capability que ejecuta la faceta."},
                    "depende_de": {
                        "type": "array", "items": {"type": "string"},
                        "description": "pipeline_id de sub-pipelines lanzados antes en esta misma tarea.",
                    },
                    "skip_on_fail": {
                        "type": "boolean",
                        "description": "true: quienes dependen de este arrancan aunque falle.",
                    },
                },
                "required": ["objetivo", "faceta", "capability"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_LEER,
            "description": "Lee el estado y un resumen del resultado de un sub-pipeline de tu propio árbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_id": {"type": "string", "description": "pipeline_id que devolvió lanzar_subpipeline."},
                },
                "required": ["pipeline_id"],
                "additionalProperties": False,
            },
        },
    },
]

NOMBRES = frozenset(t["function"]["name"] for t in TOOLS_SUBPIPELINES)
_PROPIEDADES = {t["function"]["name"]: frozenset(t["function"]["parameters"]["properties"]) for t in TOOLS_SUBPIPELINES}
_REQUERIDAS = {t["function"]["name"]: frozenset(t["function"]["parameters"]["required"]) for t in TOOLS_SUBPIPELINES}


def _texto(args: dict, campo: str, largo: int) -> str:
    valor = args[campo]
    if not isinstance(valor, str) or not valor.strip():
        raise ValueError(f"{campo} tiene que ser un texto no vacío")
    if len(valor) > largo:
        raise ValueError(f"{campo} excede {largo} caracteres")
    return valor.strip()


def validar_argumentos(nombre: str, argumentos_json: str) -> tuple[dict | None, str | None]:
    if nombre not in NOMBRES:
        return None, f"'{nombre}' no es una tool de sub-pipelines"
    try:
        args = json.loads(argumentos_json) if argumentos_json else {}
    except (json.JSONDecodeError, TypeError) as exc:
        return None, f"arguments no es JSON válido: {exc}"
    if not isinstance(args, dict):
        return None, "arguments tiene que ser un objeto JSON"
    sobran = sorted(set(args) - _PROPIEDADES[nombre])
    if sobran:
        return None, f"argumentos no permitidos {sobran}"
    faltan = sorted(_REQUERIDAS[nombre] - set(args))
    if faltan:
        return None, f"faltan {faltan}"
    try:
        if nombre == TOOL_LEER:
            return {"pipeline_id": _texto(args, "pipeline_id", LARGO_MAX_ID)}, None
        deps = args.get("depende_de", [])
        if (not isinstance(deps, list) or len(deps) > MAX_DEPENDENCIAS
                or not all(isinstance(d, str) and 0 < len(d) <= LARGO_MAX_ID for d in deps)):
            return None, "depende_de tiene que ser una lista de pipeline_id"
        skip = args.get("skip_on_fail", False)
        if not isinstance(skip, bool):
            return None, "skip_on_fail tiene que ser true o false"
        return {
            "objetivo": _texto(args, "objetivo", LARGO_MAX_OBJETIVO),
            "faceta": _texto(args, "faceta", LARGO_MAX_NOMBRE),
            "capability": _texto(args, "capability", LARGO_MAX_NOMBRE),
            "depende_de": deps,
            "skip_on_fail": skip,
        }, None
    except ValueError as exc:
        return None, str(exc)
```

- [ ] **Step 3: Implementar el script de medición**

Create `scripts/medir_tool_calling_glm.py`:

```python
#!/usr/bin/env python3
"""Medición del tool-calling del modelo de Ada: gate del modo 2 (frente G, spec §6).

PASA si >= 4 de 5 llamadas producen tool_calls válidos (nombres declarados y
argumentos que validan el schema) y hay 0 invenciones de nombres de tool.

Mismo formato que el bucle GAP2 del Motor Registry: el motor `ada` del catálogo
(DB), su credencial, su límite de salida (worker._limite_del_motor), la
identidad que inyecta el worker, y worker._call_http_openai_compat con
TOOLS_CATALOG + TOOLS_SUBPIPELINES. No ejecuta ninguna tool: solo califica lo
que el modelo pide.

GASTA: 5 llamadas reales a la API del modelo de Ada. Se niega sin
--confirmo-gasto. Guarda lo crudo (pedido SIN credencial + respuesta +
calificación) en --salida, que tiene que estar FUERA del repo.

USO (lo corre el controlador, con GO de Fernando):
  set -a; . /etc/jax/.env; set +a
  cd /home/fruiz/jax && PYTHONPATH=.:las_manos las_manos/.venv/bin/python \
      scripts/medir_tool_calling_glm.py --salida /home/fruiz/mediciones/glm-tool-calling-$(date +%F) --confirmo-gasto

Exit: 0 pasa, 1 no pasa, 2 uso inválido.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parents[1]
UMBRAL_VALIDAS = 4
_ID_INEXISTENTE = "3f1c2a9e-0b7d-4c55-9a51-2f4e8d6b1c10"

PROMPTS = [
    "Necesito saber cuánto cuesta hoy instalar paneles solares en Honduras. Lanzá un sub-pipeline con la "
    "faceta hipatia y la capability research para investigarlo. Usá la herramienta; no respondas con texto.",
    "Lanzá un sub-pipeline con la faceta jekyll y la capability analysis cuyo objetivo sea explicar en tres "
    "líneas qué es una cola FIFO. Si falla, lo que dependa de él tiene que poder arrancar igual.",
    f"Leé el resultado del sub-pipeline {_ID_INEXISTENTE} con la herramienta que corresponde.",
    "Lanzá un sub-pipeline con la faceta kimi y la capability generate para escribir una función en Python "
    f"que sume dos números. Tiene que esperar al sub-pipeline {_ID_INEXISTENTE}.",
    "Antes de seguir necesitás el resultado del sub-pipeline 9a7b6c5d-1e2f-4a3b-8c9d-0e1f2a3b4c5d. "
    "Pedilo con la herramienta; no inventes el resultado.",
]


def _argumentos_invalidos(nombre: str, argumentos_json: str) -> str | None:
    from motor_registry.tools_subpipelines import NOMBRES, validar_argumentos
    if nombre in NOMBRES:
        _args, motivo = validar_argumentos(nombre, argumentos_json)
        return motivo
    try:
        args = json.loads(argumentos_json or "{}")
    except ValueError as exc:
        return f"arguments no es JSON: {exc}"
    requeridos = {"read_file": ("path",), "write_file": ("path", "content")}[nombre]
    faltan = [r for r in requeridos if not isinstance(args, dict) or not isinstance(args.get(r), str) or not args[r]]
    return f"faltan {faltan}" if faltan else None


def calificar(respuesta: dict, nombres_declarados: frozenset[str]) -> dict:
    mensaje = ((respuesta.get("choices") or [{}])[0]).get("message") or {}
    llamadas = mensaje.get("tool_calls") or []
    motivos: list[str] = []
    inventadas: list[str] = []
    if not llamadas:
        motivos.append("sin tool_calls")
    for tc in llamadas:
        funcion = tc.get("function") or {}
        nombre = funcion.get("name", "")
        if nombre not in nombres_declarados:
            inventadas.append(nombre)
            continue
        motivo = _argumentos_invalidos(nombre, funcion.get("arguments", ""))
        if motivo:
            motivos.append(f"{nombre}: {motivo}")
    if inventadas:
        motivos.append(f"nombres inventados: {inventadas}")
    return {"tool_calls": len(llamadas), "inventadas": inventadas, "valida": not motivos, "motivos": motivos}


def resumir(calificaciones: list[dict]) -> dict:
    validas = sum(1 for c in calificaciones if c["valida"])
    invenciones = sum(len(c["inventadas"]) for c in calificaciones)
    return {
        "llamadas": len(calificaciones), "validas": validas, "invenciones": invenciones,
        "umbral_validas": UMBRAL_VALIDAS,
        "pasa": len(calificaciones) == len(PROMPTS) and validas >= UMBRAL_VALIDAS and invenciones == 0,
    }


async def _escribir(ruta: Path, datos: dict) -> None:
    await asyncio.to_thread(ruta.write_text, json.dumps(datos, ensure_ascii=False, indent=2), "utf-8")


async def medir(salida: Path) -> dict:
    from credential_resolver import resolve_credential_instrumented
    from motor_registry import worker
    from motor_registry.catalog import MotorCatalog
    from motor_registry.identity_context import build_identity_context
    from motor_registry.tools_catalog import TOOLS_CATALOG
    from motor_registry.tools_subpipelines import TOOLS_SUBPIPELINES

    catalogo = await MotorCatalog.from_db()
    ada = catalogo.get_motor("ada")
    if ada is None:
        raise RuntimeError("el motor 'ada' no está en el catálogo")
    clave = await resolve_credential_instrumented(ada.provider_id or "ada")
    limite, origen = await worker._limite_del_motor(ada)
    herramientas = TOOLS_CATALOG + TOOLS_SUBPIPELINES
    nombres = frozenset(t["function"]["name"] for t in herramientas)
    calificaciones: list[dict] = []
    for i, prompt in enumerate(PROMPTS, start=1):
        identidad = build_identity_context(
            motor_name="ada", capabilities=["delegate"], catalog={},
            predicates=worker._REFORMAS_V3_PREDICATES, task_id=f"medicion-glm-{i}",
        )
        mensajes = [{"role": "user", "content": identidad + "\n---\n" + prompt}]
        inicio = time.monotonic()
        try:
            respuesta = await worker._call_http_openai_compat(
                api_url=ada.api_url, model=ada.model, api_key=clave, messages=mensajes,
                timeout=float(ada.default_timeout_seconds), limite=limite, tools=herramientas,
            )
            error = None
        except Exception as exc:  # fail-soft: una llamada que falla cuenta como NO válida (no se reintenta ni se descarta) y queda registrada con su error
            respuesta, error = {}, f"{type(exc).__name__}: {exc}"
        calificacion = calificar(respuesta, nombres)
        if error:
            calificacion["valida"] = False
            calificacion["motivos"].append(f"error de API: {error}")
        calificacion["segundos"] = round(time.monotonic() - inicio, 2)
        calificaciones.append(calificacion)
        await _escribir(salida / f"llamada-{i}.json", {
            "pedido": {"api_url": ada.api_url, "model": ada.model, "limite": limite, "origen_limite": origen,
                       "tools": [t["function"]["name"] for t in herramientas], "messages": mensajes},
            "respuesta": respuesta, "error": error, "calificacion": calificacion,
        })
        print(f"llamada {i}: valida={calificacion['valida']} tool_calls={calificacion['tool_calls']} "
              f"motivos={calificacion['motivos']}")
    resumen = resumir(calificaciones)
    resumen.update({"modelo": ada.model, "fecha": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    await _escribir(salida / "resumen.json", resumen)
    return resumen


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Medición de tool-calling de GLM (gate del modo 2).")
    p.add_argument("--salida", required=True)
    p.add_argument("--confirmo-gasto", action="store_true")
    args = p.parse_args(argv)
    salida = Path(args.salida).resolve()
    if not args.confirmo_gasto:
        print("Se niega: gasta 5 llamadas reales. Repetir con --confirmo-gasto (GO de Fernando).")
        return 2
    if salida == RAIZ_REPO or RAIZ_REPO in salida.parents:
        print(f"Se niega: {salida} está dentro del repo; los resultados crudos van fuera.")
        return 2
    salida.mkdir(parents=True, exist_ok=True)
    resumen = asyncio.run(medir(salida))
    print(json.dumps(resumen, ensure_ascii=False))
    return 0 if resumen["pasa"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verde y commit**

```bash
cd $WT && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_tools_subpipelines.py tests/test_medicion_glm.py 2>&1 | tail -1
cd $WT && $PY -m pytest -q policy/tests/test_no_fail_open_except.py tests/test_no_blocking_in_async.py 2>&1 | tail -1
git -C $WT add las_manos/motor_registry/tools_subpipelines.py scripts/medir_tool_calling_glm.py tests/test_tools_subpipelines.py tests/test_medicion_glm.py $L
git -C $WT commit -m "feat(motor_registry): tools de sub-pipelines declaradas y script de medición de GLM

lanzar_subpipeline / leer_resultado_subpipeline con validación estricta de
argumentos (una sola fuente: la declaración). Script con calificación
determinista del spec §6 (>= 4/5 válidas y 0 invenciones), que se niega sin
--confirmo-gasto y fuera del repo. 12 + 11 puros vistos en rojo.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```
Expected: `23 passed` (12 de tools: 3 simples + 9 parametrizados; 11 de medición: 5 simples + 4 parametrizados + 2 de negativa).

Estos dos archivos van al PR de la etapa 2. Las tools **no** se ofrecen a ningún motor hasta la Task 22.

---

### Task 19: Medición real y gate (controlador)

**Files:**
- Create (fuera del repo): `/home/fruiz/mediciones/glm-tool-calling-<fecha>/llamada-{1..5}.json`, `resumen.json`

**Interfaces:**
- Consumes: el script de la Task 18, ejecutado desde el worktree `$WT` (no hace falta desplegar, porque no toca servicios).
- Produces: el veredicto `pasa` y el registro que decide si siguen las Tasks 20-23.

- [ ] **Step 1: Correr la medición (gasta, con GO)**

```bash
M=/home/fruiz/mediciones/glm-tool-calling-$(date +%F)
bash -c "set -a; . /etc/jax/.env; set +a; cd $WT && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python scripts/medir_tool_calling_glm.py --salida $M --confirmo-gasto"; echo "exit=$?"
ls -l $M && cat $M/resumen.json
grep -c "$(bash -c 'set -a; . /etc/jax/.env; set +a; echo ${ZHIPU_API_KEY:-no-hay-variable}')" $M/*.json || true
```
Expected:
- 5 líneas `llamada N: …`, el JSON del resumen y `exit=0` (pasa) o `exit=1` (no pasa);
- `ls` muestra 6 archivos;
- el último `grep` da `0` en cada archivo: la credencial no está en lo crudo.

La variable de la credencial de Zhipu tiene otro nombre en `.env`: se toma el que devuelve `resolve_credential_instrumented` (tabla `credential`). Si la credencial vive cifrada en la DB, el chequeo equivalente es `grep -c "Bearer" $M/*.json`, que también tiene que dar 0. **No se repite ninguna llamada:** una que falla por red cuenta como no válida (spec: 5 llamadas).

Pegar en `$L` el `resumen.json` y, por cada llamada, `tool_calls`, `valida` y `motivos`.

- [ ] **Step 2: Gate**

- **`pasa: true`** → seguir con la Task 20. Contarle a Fernando el resultado (validas/5, invenciones, modelo).
- **`pasa: false`** → **las Tasks 20, 21, 22 y 23 no se ejecutan.**
  1. Contarle a Fernando el resultado con los motivos.
  2. Registrar en la Task 24:
     - **HISTORIA:** la medición, con fecha, modelo y resumen;
     - **DECISIÓN (spec §2):** "el modo 1 queda operativo y el 2 espera a un modelo que pase";
     - **PENDIENTE con fecha de revisión 2026-10-16:** re-medir con `scripts/medir_tool_calling_glm.py`, o antes si cambia el `facet_binding` de `ada`. El que cambie el binding tiene que correr la medición.
  3. Los archivos de la Task 18 **quedan** en master, igual que el gate. Van en un PR chico de la etapa 2 (Task 21, Step 5, sólo esos archivos y sus listas de CI) y en ese caso **nada se ofrece a ningún motor**.

---

### Task 20: Auditoría de las capabilities de Ada y semillas de la migración (sólo si pasó la Task 19)

**Files:**
- Modify: `$WTP/backend/db/migrations.py` (`_ADA_CAPABILITY_MOTOR`, `_seed_ada_capability_motor`, llamada en `run_migrations`)
- Modify: `$WTP/backend/tests/test_motor_migrations.py` (conteo 26 → 26 + N)
- Create: `$WTP/backend/tests/test_ada_gobernada_semillas.py`
- Modify: `$WTP/frontend/src/components/BottomBar/pipelineChain.js` (`GOVERNED_FACETS`, `HTTP_FACETS`) y su test

**Interfaces:**
- Consumes: producción en sólo lectura (controlador).
- Produces:
  - `capability_motor(<cap>, 'ada')` para cada capability de la auditoría que no tenga fila, más `delegate` e `integrate`;
  - en el frontend, `GOVERNED_FACETS = ['jax_local', 'kimi', 'ada']` y `HTTP_FACETS = ['hipatia', 'jekyll', 'thot']`.

- [ ] **Step 1: Auditoría en producción (sólo lectura, controlador)**

```bash
( set -a; . /etc/jax/.env; set +a
  mariadb -h "$JAX_DB_HOST" -P "$JAX_DB_PORT" -u "$JAX_DB_USER" -p"$JAX_DB_PASSWORD" jax_memory -e "
  SELECT s.capability, COUNT(*) AS pasos, MAX(s.timeout_seconds) AS timeout_max,
         c.\`key\` IS NOT NULL AS existe, c.allowed_callers, c.requires_human_gate, c.max_recursion_depth,
         c.max_execution_minutes,
         (SELECT GROUP_CONCAT(cm.motor_key ORDER BY cm.priority) FROM capability_motor cm WHERE cm.capability_key = s.capability) AS motores
    FROM jacobs_steps s LEFT JOIN capability c ON c.\`key\` = s.capability
   WHERE s.facet = 'ada'
   GROUP BY s.capability, c.\`key\`, c.allowed_callers, c.requires_human_gate, c.max_recursion_depth, c.max_execution_minutes;
  SELECT \`key\`, max_tokens, default_timeout_seconds, sandbox_only, has_tool_access, status FROM motor_resolved WHERE \`key\` = 'ada';
  SELECT COUNT(*) AS turnos, MAX(tokens_out) AS max_salida,
         SUM(tokens_out > 0.8 * (SELECT max_tokens FROM motor WHERE \`key\`='ada')) AS cerca_del_tope
    FROM axioma_usage WHERE facet = 'ada' AND request_type = 'pipeline';" ) | tee -a $L
```

Aplicar los 8 checks de `MotorPolicy` por capability y anotar la tabla en `$L`. Los checks son:
1. existe;
2. `jacobs` está en `allowed_callers`;
3. sin gate humano;
4. `recursion 0 <= max`;
5. claves de contexto: el `input` de los steps sólo usa `prompt`/`name`;
6. `ada` está en `motores` (o se agrega);
7. el motor es `sandbox_only`;
8. `timeout_max <= max_execution_minutes * 60`.

Reglas, sin excepciones:
- `assemble` no entra: es mecánico, nunca despacha.
- Una capability que falla el check 1, 2, 3, 4 u 8 → **parar**, presentarla a Fernando y no migrar. Es decisión de vocabulario suya, como en la ronda 7 (T4.c).
- Si `cerca_del_tope > 0`, los turnos de Ada por HTTP ya salían cerca de `motor.max_tokens`. Por el motor, el worker corta con `finish_reason=length` y falla el step (`worker.py:766-803`) → **parar** y presentarle el número a Fernando. Si decide subir el presupuesto, lo hace desde **Admin → Motores** (PATCH existente, `api/admin/motors.py`), nunca un UPDATE a mano. Después se repite este Step.
- Las capabilities que pasan y no tienen `ada` en `motores` van a `_ADA_CAPABILITY_MOTOR`.

- [ ] **Step 2: Test de semillas que falla**

Create `$WTP/backend/tests/test_ada_gobernada_semillas.py`:

```python
"""Frente G etapa 2: ada pasa al Motor Registry (jax/jacobs/models.py::MOTOR_FACETS).
Cada capability que ada usa en producción (auditoría en el ledger del frente G)
y las del emisor (delegate/integrate) necesitan su fila capability_motor, o el
plan se rechaza antes de persistir (jax/jacobs/plan.py::_validate_plan_capabilities)."""
from __future__ import annotations

from db import migrations


async def _motores(capability):
    from db.connection import get_pool
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT motor_key FROM capability_motor WHERE capability_key=%s", (capability,))
            return {r[0] for r in await cur.fetchall()}


def test_la_lista_incluye_las_del_emisor():
    assert {"delegate", "integrate", "analysis"} <= set(migrations._ADA_CAPABILITY_MOTOR)


def test_cada_capability_de_la_lista_tiene_a_ada(client):
    for capability in migrations._ADA_CAPABILITY_MOTOR:
        assert "ada" in client.portal.call(_motores, capability), capability
```

```bash
cd $WTP/backend && JAX_REPO_PATH=$WT $PYP -m pytest -q tests/test_ada_gobernada_semillas.py 2>&1 | tee -a $L | tail -3
```
Expected: `2 failed` (`AttributeError: _ADA_CAPABILITY_MOTOR`).

- [ ] **Step 3: Semillas**

En `$WTP/backend/db/migrations.py`, después de `_seed_thot_motor`:

```python
# Frente G etapa 2 (auditoría del <fecha>, ledger 2026-09-16-emisor-subpipelines-ada):
# ada pasa al Motor Registry. Sin su fila en capability_motor, un plan con
# ada/<cap> se rechaza antes de persistir. `analysis` no tenía ninguna; delegate
# e integrate son del emisor. Se agregan acá, en orden alfabético y con la fila
# de la auditoría que las justifica, las demás capabilities usadas sin fila de ada.
_ADA_CAPABILITY_MOTOR = ["analysis", "delegate", "integrate"]


async def _seed_ada_capability_motor(cur) -> None:
    await cur.execute("SELECT 1 FROM motor WHERE `key`='ada'")
    if await cur.fetchone() is None:
        return  # motor no existe todavía -- mismo guard que el resto de las semillas
    for capability_key in _ADA_CAPABILITY_MOTOR:
        await cur.execute("SELECT 1 FROM capability WHERE `key`=%s", (capability_key,))
        if await cur.fetchone() is None:
            continue
        await cur.execute(
            "INSERT IGNORE INTO capability_motor (capability_key, motor_key, priority) VALUES (%s, 'ada', 0)",
            (capability_key,),
        )
```

En `run_migrations`, inmediatamente después de `await _seed_thot_motor(cur)`: `await _seed_ada_capability_motor(cur)`.

En `tests/test_motor_migrations.py`, renombrar `test_capability_motor_seed_count_is_26` a `test_capability_motor_seed_count` y reemplazar el `26` por `26 + len(migrations._ADA_CAPABILITY_MOTOR)`, con el `import` de `migrations` si falta. Agregar una línea al docstring: `# +N el <fecha> (frente G etapa 2): filas de ada.`

- [ ] **Step 4: Frontend**

En `$WTP/frontend/src/components/BottomBar/pipelineChain.js`:

```js
// Motor Registry: se valida contra capability_motor y llevan motor fijo.
// ada entra el <fecha> (frente G etapa 2): pasa de HTTP directo al Motor Registry
// (jax/jacobs/models.py::MOTOR_FACETS), con sus filas de capability_motor.
export const GOVERNED_FACETS = ['jax_local', 'kimi', 'ada']
// HTTP directo: su admisión mira la capability (allowed_callers), no un
// binding de motor -- jacobs/executor.py::validate_capability, NIVEL C.
export const HTTP_FACETS = ['hipatia', 'jekyll', 'thot']
```

Buscar las aserciones que fijan las particiones viejas: `grep -rn "HTTP_FACETS\|GOVERNED_FACETS\|'ada'" $WTP/frontend/src --include=*.test.*`. Actualizarlas en este commit: la partición es dato espejado de jax. Agregar a `pipelineChain.test.js`:

```js
it('ada es faceta gobernada (frente G etapa 2): aparece solo donde capability_motor la admite', () => {
  const plan = CHAIN_ROLES.find(r => r.id === 'plan')
  const research = CHAIN_ROLES.find(r => r.id === 'research')
  const capabilities = { design: ['ada', 'kimi'], research: [] }
  expect(facetOptionsFor(plan, capabilities)).toContain('ada')
  expect(facetOptionsFor(research, capabilities)).not.toContain('ada')
  expect(buildChainSteps('o', { ...defaultFacetsByRole() }, Object.fromEntries(CHAIN_ROLES.map(r => [r.id, 'x'])))[1].motor).toBe('ada')
})
```

**Cambio de comportamiento que se declara en el PR y se verifica en vivo (Task 21, Step 6):** en la cadena, `ada` deja de aparecer en roles cuya capability no la lista en `capability_motor` (por ejemplo `research`), y sus steps llevan `motor: 'ada'`.

- [ ] **Step 5: Verde y commit**

```bash
cd $WTP/backend && JAX_REPO_PATH=$WT $PYP -m pytest -q tests/test_ada_gobernada_semillas.py tests/test_motor_migrations.py tests/test_capability_mode.py 2>&1 | tail -1
cd $WTP/frontend && npx vitest run 2>&1 | tail -2
git -C $WTP add backend/db/migrations.py backend/tests/test_motor_migrations.py backend/tests/test_ada_gobernada_semillas.py frontend/src/components/BottomBar/pipelineChain.js frontend/src/components/BottomBar/pipelineChain.test.js
git -C $WTP commit -m "feat(migraciones): ada al Motor Registry -- capability_motor de sus capabilities y del emisor

Auditoría de producción en el ledger del frente G (8 checks por capability).
Frontend: ada pasa a GOVERNED_FACETS (espejo de jacobs/models.py). 2 backend +
1 vitest vistos en rojo.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 21: jax · `ada` en `MOTOR_FACETS`, CI, deploy y verificación (sólo si pasó la Task 19)

**Files:**
- Modify: `jacobs/models.py` (`HTTP_FACETS`, `MOTOR_FACETS`)
- Modify: `jacobs/plan.py` (`_validate_plan_capabilities`: exentar `assemble`)
- Modify: `jacobs/_http_facet_admission_test.py` (el caso de `bug_hunt` usa `thot`, no `ada`)
- Create: `tests/test_ada_gobernada.py`
- Modify: `.github/workflows/policy.yml`

**Interfaces:**
- Consumes: las semillas de la Task 20 (mergeadas y desplegadas antes).
- Produces: `models.MOTOR_FACETS = frozenset({"kimi", "jax_local", "ada"})`, `models.HTTP_FACETS = frozenset({"hipatia", "jekyll", "thot"})`. Ada despacha por `/motor/dispatch` con los 8 checks y su uso se registra con `request_type='motor'`.

- [ ] **Step 1: Tests que fallan**

Create `tests/test_ada_gobernada.py`:

```python
"""Frente G etapa 2: ada pasa al Motor Registry (spec §3.2 (b)). Prerrequisito
del modo 2: el bucle GAP2 y tool_authority solo existen en ese camino.

ROJO CONTRA LA BASE: ada sigue en HTTP_FACETS y el plan con ada/assemble se
rechazaría por capability_motor."""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest

os.environ["JAX_DB_NAME"] = "jax_memory_test"

from jacobs import models  # noqa: E402
from jacobs import plan as plan_mod  # noqa: E402
from jacobs.models import Step  # noqa: E402

GOBERNANZA = {
    "capabilities": {"design": {"allowed_motors": ["ada", "kimi"], "allowed_callers": ["jacobs"], "max_execution_minutes": 15}},
    "motors": {"ada": False, "kimi": False},
    "facets": frozenset({"ada", "kimi"}),
}


def test_ada_es_faceta_de_motor():
    assert "ada" in models.MOTOR_FACETS
    assert "ada" not in models.HTTP_FACETS
    assert not (models.MOTOR_FACETS & models.HTTP_FACETS)


def test_assemble_de_ada_no_exige_capability_motor():
    pasos = [Step(step_index=0, facet="ada", capability="design"),
             Step(step_index=1, facet="ada", capability="assemble", depends_on=[0])]
    with patch("jacobs.store.get_motor_governance", AsyncMock(return_value=GOBERNANZA)):
        asyncio.run(plan_mod._validate_plan_capabilities(pasos))


def test_ada_con_una_capability_sin_su_fila_se_rechaza():
    with patch("jacobs.store.get_motor_governance", AsyncMock(return_value=GOBERNANZA)):
        with pytest.raises(plan_mod.PlanRejected, match="capability_motor"):
            asyncio.run(plan_mod._validate_plan_capabilities([Step(step_index=0, facet="ada", capability="research")]))
```

```bash
cd $WT && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_ada_gobernada.py 2>&1 | tee -a $L | tail -3
```
Expected: `2 failed, 1 passed`.
- `test_ada_es_faceta_de_motor` falla porque la partición vieja todavía está.
- `test_ada_con_una_capability_sin_su_fila…` falla porque ada todavía no pasa por el chequeo de motor: DID NOT RAISE.
- `test_assemble…` pasa (ada no es de motor todavía). Es el control del exento, validado por mutación en el Step 3.

- [ ] **Step 2: Implementar**

`jacobs/models.py`:

```python
# Frente G etapa 2 (<fecha>): ada pasa al Motor Registry -- 8 checks de
# MotorPolicy, bucle GAP2 y uso por job. Auditoría y semillas: ledger del frente
# G y jax-platform/backend/db/migrations.py::_ADA_CAPABILITY_MOTOR.
HTTP_FACETS = frozenset({"hipatia", "jekyll", "thot"})
MOTOR_FACETS = frozenset({"kimi", "jax_local", "ada"})
```

`jacobs/plan.py::_validate_plan_capabilities`, reemplazar `relevant = [s for s in steps if (s.motor or s.facet) in MOTOR_FACETS]` por:

```python
    # `assemble` es el ensamble mecánico (executor._assemble_mechanical): nunca
    # despacha, así que no necesita binding de motor. Antes no hacía falta
    # exentarlo porque ninguna faceta de motor lo usaba; ada sí (frente G etapa 2).
    relevant = [s for s in steps if (s.motor or s.facet) in MOTOR_FACETS and s.capability != "assemble"]
```

`jacobs/_http_facet_admission_test.py::test_capability_con_human_gate_es_denegada`: cambiar `Step(facet="ada", capability="bug_hunt", ...)` por `Step(facet="thot", capability="bug_hunt", ...)` y agregar al docstring: `(Frente G etapa 2: ada pasó a MOTOR_FACETS; el caso HTTP se prueba con thot.)`. Verificar antes que `bug_hunt` lista `jacobs` en `allowed_callers`. Si no, se elige otra capability con gate y `jacobs`, y se anota cuál.

- [ ] **Step 3: Verde, mutación y suites**

```bash
cd $WT && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_ada_gobernada.py tests/test_delegacion_executor_puro.py tests/test_dispatch_step_reroute.py tests/test_validate_capability_typed.py tests/test_contrato_dispatch_repl_ada.py 2>&1 | tail -1
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_http_facet_admission_test.py jacobs/_plan_timeout_ceiling_test.py tests/test_plan_validation.py jacobs/_delegacion_io_test.py tests/test_delegacion_arbol_e2e.py" 2>&1 | tail -1
```
Expected: todo `passed`.

Mutación: quitar `and s.capability != "assemble"` y correr `-k assemble`. Expected: `1 failed`. Restaurar y pegar las dos salidas en `$L`.

- [ ] **Step 4: CI y commit**

`tests/test_ada_gobernada.py`, `tests/test_tools_subpipelines.py` y `tests/test_medicion_glm.py` van a las dos listas de `tests-puros`, con el piso medido en el runner (+3 +12 +11).

```bash
git -C $WT add jacobs/models.py jacobs/plan.py jacobs/_http_facet_admission_test.py tests/test_ada_gobernada.py .github/workflows/policy.yml $L
git -C $WT commit -m "feat(jacobs): ada pasa al Motor Registry (MOTOR_FACETS)

Prerrequisito del modo 2 del emisor. assemble exento del binding de motor (es
mecánico). El caso HTTP con gate humano se prueba con thot. 2 vistos en rojo
+ 1 control validado por mutación.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: PRs, canario, merge y deploy (controlador)**

Mismo procedimiento que las Tasks 14 y 15:
1. **jax-platform primero:** revisión, PR, canario (`GOVERNED_FACETS` sin `'ada'` → rojo en `frontend-tests`), merge, backup verificado de `capability_motor`, restart de `jax-platform` y deploy del frontend con backup.
2. Verificar en producción: `SELECT capability_key FROM capability_motor WHERE motor_key='ada' ORDER BY 1`, que tiene que listar las de `_ADA_CAPABILITY_MOTOR`.
3. **jax después:** revisión, rebase, PR, canario (`MOTOR_FACETS` sin `"ada"` → rojo en `tests-puros`), merge, 0 en vuelo, pull y restart de `jax-las-manos`.

Si la Task 19 **no pasó**, en este paso sólo se sube el PR chico con los archivos de la Task 18 y sus listas de CI, sin las Tasks 20 y 21.

- [ ] **Step 6: Verificación en vivo del camino gobernado (controlador + Fernando)**

Fernando lanza desde la Mesa una cadena corta con `ada` en `plan` (design) y lo demás por defecto. Vigilar en sólo lectura:

```bash
tail -n 0 -F /home/fruiz/jax/las_manos/logs/motor_jobs.jsonl | /home/fruiz/jax/.venv/bin/python -u -c "
import sys, json
for l in sys.stdin:
    d = json.loads(l)
    if d.get('motor') == 'ada': print(d['job_id'][:8], d.get('caller'), d.get('capability'), d['status'], (d.get('error') or '')[:90])"
```
Expected: un job `ada` con `caller=jacobs`, `capability=design` y estado `completed`.

Después:
```bash
( set -a; . /etc/jax/.env; set +a
  mariadb -h "$JAX_DB_HOST" -P "$JAX_DB_PORT" -u "$JAX_DB_USER" -p"$JAX_DB_PASSWORD" jax_memory -e \
  "SELECT facet, request_type, status, tokens_out, job_id FROM axioma_usage WHERE facet='ada' ORDER BY id DESC LIMIT 3" )
```
Expected: una fila `request_type='motor'` con `job_id`. Fernando confirma en claro/oscuro que la cadena muestra `ada` sólo donde corresponde.

Si el step de `ada` falla por `finish_reason=length`, **no se sube el tope a mano**: se registra y se vuelve al Step 1 de la Task 20 con el número.

---

# ETAPA 3 · Modo 2: tools en el bucle GAP2 (sólo si pasó la Task 19 y se desplegó la Task 21)

### Task 22: `lanzar_subpipeline` y `leer_resultado_subpipeline` por la MISMA `delegacion.py`

**Files:**
- Modify: `jacobs/delegacion.py` (`delegar`, `_pedir_aprobacion` y `_lanzar` con `externas` y `transicionar_padre`; `aprobar` lee `externas`; funciones nuevas `delegar_por_tool`, `leer_resultado_por_tool`, `resultado_de_tools`)
- Modify: `jacobs/executor.py` (`_run_one_step`: resultado de tools; `run_pipeline`: transición de `espera_aprobacion`; `_invoke_motor`: `pipeline_id`/`step_id`)
- Modify: `las_manos/motor_registry/models.py` (`MotorDispatchRequest.pipeline_id/step_id`)
- Modify: `las_manos/motor_registry/routes.py` (`dispatch`)
- Modify: `las_manos/motor_registry/worker.py` (`herramientas_para`, `run`)
- Modify: `las_manos/motor_registry/tool_authority.py` (mapa, ejecutables, `_herramienta_subpipeline`)
- Modify: `las_manos/_worker_tool_loop_test.py` (`allowed_motors` de `file_read`/`file_write` en la config de prueba)
- Create: `tests/test_tools_subpipelines_autoridad.py`
- Modify: `jacobs/_delegacion_io_test.py` (clase `ToolTest`)

**Interfaces:**
- Consumes: Tasks 7-10 y 18; `tool_authority._reject`.
- Produces:
  - `delegacion.delegar(padre, paso, plan, *, por="auto", externas: dict[str, list[str]] | None = None, transicionar_padre: bool = True) -> Delegacion`
  - `delegacion.delegar_por_tool(pipeline_id: str, step_id: str, argumentos: dict) -> dict`. Devuelve `{"ok": True, "estado": "encolado", "pipeline_id"}`, `{"ok": True, "estado": "pendiente_de_aprobacion", "aprobacion_id"}` o `{"ok": False, "motivo"}`.
  - `delegacion.leer_resultado_por_tool(pipeline_id: str, hijo_pipeline_id: str) -> dict`. Devuelve `{"ok": True, "pipeline_id", "estado", "resumen", "output_ref"}` o `{"ok": False, "motivo": "fuera_de_tu_arbol"}`.
  - `delegacion.resultado_de_tools(pipeline_id: str, step_id: str) -> str | None` (`"espera_aprobacion"`, `"lanzada"` o `None`).
  - `worker.herramientas_para(motor: str, catalog: MotorCatalog) -> list[dict]`
  - `tool_authority.authorize_and_execute_tool_call(..., jacobs_pipeline_id: str | None = None, jacobs_step_id: str | None = None)`
  - `MotorDispatchRequest.pipeline_id: str | None = None`, `step_id: str | None = None`; `worker.run(..., pipeline_id=None, step_id=None)`.
  - En una aprobación de tipo `excede`, la columna `plan` puede traer `"externas": {subobjetivo_id: [pipeline_id]}` (dependencias de hermanos ya lanzados).

- [ ] **Step 1: Tests puros que fallan**

Create `tests/test_tools_subpipelines_autoridad.py`:

```python
"""Frente G etapa 3 (spec §3.2): las tools de sub-pipelines pasan por
tool_authority de CERO en cada llamada y deciden con la MISMA delegacion.py.
Discrepancia 16: cada tool se ofrece solo a los motores que su capability
admite (has_tool_access ya no entrega el catálogo entero).

ROJO CONTRA LA BASE: herramientas_para no existe y las tools no están mapeadas."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

from motor_registry import tool_authority, worker  # noqa: E402
from motor_registry.catalog import MotorCatalog  # noqa: E402
from motor_registry.job_store import JobStore  # noqa: E402

_CAP = {"allowed_callers": ["jacobs"], "requires_human_gate": False, "max_execution_minutes": 15,
        "max_recursion_depth": 0, "output_schema": ""}
CFG = {
    "motors": {
        "ada": {"enabled": True, "provider_id": "zhipu", "api_url": "https://x.test/v1", "model": "glm",
                "sandbox_only": True, "default_timeout_seconds": 30, "transport": "http_openai_compat",
                "has_tool_access": True},
        "jax_local": {"enabled": True, "provider_id": "ollama", "api_url": "http://localhost:11434/v1",
                      "model": "qwen", "sandbox_only": True, "default_timeout_seconds": 30,
                      "transport": "ollama", "has_tool_access": True},
    },
    "capabilities": {
        "delegate": {**_CAP, "allowed_motors": ["ada"]},
        "design": {**_CAP, "allowed_motors": ["ada"]},
        "file_read": {**_CAP, "allowed_motors": ["jax_local"], "forbidden_paths": [".env"]},
        "file_write": {**_CAP, "allowed_motors": ["jax_local"], "forbidden_paths": [".env"]},
    },
}
LANZAR = json.dumps({"objetivo": "x", "faceta": "jekyll", "capability": "analysis"})


def _nombres(tools):
    return sorted(t["function"]["name"] for t in tools)


class OfertaTest(unittest.TestCase):
    def test_cada_motor_recibe_solo_las_tools_que_su_capability_admite(self):
        catalogo = MotorCatalog(CFG)
        self.assertEqual(_nombres(worker.herramientas_para("ada", catalogo)),
                         ["lanzar_subpipeline", "leer_resultado_subpipeline"])
        self.assertEqual(_nombres(worker.herramientas_para("jax_local", catalogo)), ["read_file", "write_file"])
        self.assertEqual(worker.herramientas_para("kimi", catalogo), [])


class AutoridadTest(unittest.IsolatedAsyncioTestCase):
    async def _llamar(self, nombre, argumentos, caller="jacobs", pipeline_id="p1", step_id="s1"):
        with patch.object(tool_authority, "event_append", AsyncMock()), \
             patch("jacobs.delegacion.delegar_por_tool", AsyncMock(return_value={"ok": True, "estado": "encolado", "pipeline_id": "h1"})) as lanzar, \
             patch("jacobs.delegacion.leer_resultado_por_tool", AsyncMock(return_value={"ok": True, "estado": "completed"})) as leer:
            r = await tool_authority.authorize_and_execute_tool_call(
                tool_name=nombre, arguments_json=argumentos, caller=caller, job_id="job-1",
                catalog=MotorCatalog(CFG), jacobs_pipeline_id=pipeline_id, jacobs_step_id=step_id)
        return r, lanzar, leer

    async def test_lanzar_valido_se_ejecuta_por_delegacion(self):
        r, lanzar, _leer = await self._llamar("lanzar_subpipeline", LANZAR)
        self.assertEqual(r["decision"], "executed")
        self.assertEqual(json.loads(r["content"])["pipeline_id"], "h1")
        lanzar.assert_awaited_once_with("p1", "s1", {"objetivo": "x", "faceta": "jekyll", "capability": "analysis",
                                                     "depende_de": [], "skip_on_fail": False})

    async def test_argumentos_invalidos_se_rechazan_sin_delegar(self):
        r, lanzar, _ = await self._llamar("lanzar_subpipeline", '{"objetivo": "x"}')
        self.assertEqual(r["decision"], "rejected")
        lanzar.assert_not_awaited()

    async def test_sin_pipeline_de_jacobs_se_rechaza(self):
        r, lanzar, _ = await self._llamar("lanzar_subpipeline", LANZAR, pipeline_id=None, step_id=None)
        self.assertEqual(r["decision"], "rejected")
        self.assertIn("pipeline de Jacobs", r["reason"])
        lanzar.assert_not_awaited()

    async def test_caller_no_autorizado_se_rechaza(self):
        r, lanzar, _ = await self._llamar("lanzar_subpipeline", LANZAR, caller="hyde")
        self.assertEqual(r["decision"], "rejected")
        lanzar.assert_not_awaited()

    async def test_leer_va_por_delegacion(self):
        r, _lanzar, leer = await self._llamar("leer_resultado_subpipeline", '{"pipeline_id": "h9"}')
        self.assertEqual(r["decision"], "executed")
        leer.assert_awaited_once_with("p1", "h9")


class CableadoTest(unittest.IsolatedAsyncioTestCase):
    async def test_invoke_motor_manda_pipeline_y_step(self):
        from jacobs import executor
        from jacobs.models import Pipeline, Step
        pipeline = Pipeline(pipeline_id="p1", name="n", invoked_by="plataforma", mode="autonomous", root_pipeline_id="p1")
        step = Step(pipeline_id="p1", facet="ada", capability="design", input={"prompt": "x"})
        respuesta = MagicMock()
        respuesta.raise_for_status = lambda: None
        respuesta.json = lambda: {"status": "rejected", "rejected_reason": "probe"}
        cliente = MagicMock()
        cliente.post = AsyncMock(return_value=respuesta)
        with patch.object(executor, "obtener_cliente_http", return_value=cliente):
            with self.assertRaises(RuntimeError):
                await executor._invoke_motor(step, pipeline, 30, "prompt")
        cuerpo = cliente.post.await_args.kwargs["json"]
        self.assertEqual((cuerpo["pipeline_id"], cuerpo["step_id"]), ("p1", step.step_id))

    async def test_worker_ofrece_las_tools_de_ada_y_pasa_los_ids(self):
        llamadas = []

        async def transporte(**kw):
            llamadas.append(kw)
            if len(llamadas) == 1:
                return {"choices": [{"message": {"content": "", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "lanzar_subpipeline", "arguments": LANZAR}}]},
                    "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
            return {"choices": [{"message": {"content": "listo"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

        store = JobStore(tempfile.mkdtemp(prefix="jax-g-tools-") + "/jobs.jsonl")
        job_id = store.create(caller="jacobs", capability="design", motor="ada", trace_id="t", prompt="x",
                              recursion_depth=0)
        autorizar = AsyncMock(return_value={"tool_name": "lanzar_subpipeline", "decision": "executed",
                                            "reason": None, "content": "{}"})
        with patch.dict(worker._TRANSPORT_DISPATCH, {"http_openai_compat": transporte}), \
             patch.object(worker, "_limite_del_motor", AsyncMock(return_value=({"max_tokens": 100}, "catalogo"))), \
             patch.object(worker, "resolve_credential_instrumented", AsyncMock(return_value="sk")), \
             patch.object(worker, "authorize_and_execute_tool_call", autorizar), \
             patch("motor_registry.usage_writer.record_motor_usage", AsyncMock(return_value=True)):
            await worker.run(job_id=job_id, motor="ada", capability="design", prompt="x", context={},
                             store=store, catalog=MotorCatalog(CFG), kill_switch_path="/tmp/jax-g-sin-freno/PAUSE",
                             caller="jacobs", timeout_seconds=30, pipeline_id="p1", step_id="s1")
        self.assertEqual(_nombres(llamadas[0]["tools"]), ["lanzar_subpipeline", "leer_resultado_subpipeline"])
        self.assertEqual((autorizar.await_args.kwargs["jacobs_pipeline_id"], autorizar.await_args.kwargs["jacobs_step_id"]),
                         ("p1", "s1"))
```

```bash
cd $WT && pwd && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_tools_subpipelines_autoridad.py 2>&1 | tee -a $L | tail -3
```
Expected: `8 failed`.

- [ ] **Step 2: Tests de base que fallan**

Agregar a `jacobs/_delegacion_io_test.py`, antes de `if __name__ == "__main__":`:

```python
class ToolTest(_ConColaG):
    """Las mismas pruebas del emisor, por la tool (spec §6, modo 2)."""

    async def lanzar(self, padre, paso, **kw):
        from jacobs import delegacion
        args = {"objetivo": "sub por tool", "faceta": "jekyll", "capability": "analysis",
                "depende_de": [], "skip_on_fail": False}
        args.update(kw)
        return await delegacion.delegar_por_tool(padre.pipeline_id, paso.step_id, args)

    async def test_dentro_de_techos_encola_con_token_y_prepara_la_integracion(self):
        p, paso = await self.padre()
        r = await self.lanzar(p, paso)
        self.assertEqual((r["ok"], r["estado"]), (True, "encolado"))
        self.assertEqual(await g.tokens_emitidos(p.pipeline_id), 1)
        hijo = await store.pipeline_get(r["pipeline_id"])
        self.assertEqual((hijo.status, hijo.parent_pipeline_id), (PipelineStatus.queued, p.pipeline_id))
        padre = await store.pipeline_get(p.pipeline_id)
        self.assertEqual(padre.plan[-1].capability, "integrate")
        from jacobs import delegacion
        self.assertEqual(await delegacion.resultado_de_tools(p.pipeline_id, paso.step_id), "lanzada")

    async def test_depende_de_un_hermano_espera(self):
        p, paso = await self.padre()
        a = await self.lanzar(p, paso)
        b = await self.lanzar(p, paso, depende_de=[a["pipeline_id"]])
        hijo_b = await store.pipeline_get(b["pipeline_id"])
        self.assertIsNone(hijo_b.queued_at)
        self.assertEqual(hijo_b.context["delegacion"]["depende_de"], [a["pipeline_id"]])

    async def test_depende_de_ajeno_se_rechaza_sin_token(self):
        p, paso = await self.padre()
        r = await self.lanzar(p, paso, depende_de=["no-es-mi-hermano"])
        self.assertFalse(r["ok"])
        self.assertIn("fuera de esta tarea", r["motivo"])
        self.assertEqual(await g.tokens_emitidos(p.pipeline_id), 0)

    async def test_exceso_pide_aprobacion_acumula_y_al_aprobar_lanza(self):
        from jacobs import delegacion
        os.environ["JAX_ADA_MAX_HIJOS_POR_STEP"] = "1"
        p, paso = await self.padre()
        a = await self.lanzar(p, paso)
        self.assertEqual(a["estado"], "encolado")
        b = await self.lanzar(p, paso, depende_de=[a["pipeline_id"]])
        c = await self.lanzar(p, paso)
        self.assertEqual((b["estado"], c["estado"]), ("pendiente_de_aprobacion", "pendiente_de_aprobacion"))
        self.assertEqual(b["aprobacion_id"], c["aprobacion_id"])
        self.assertEqual(await store.pipeline_status(p.pipeline_id), PipelineStatus.running)
        self.assertEqual(await delegacion.resultado_de_tools(p.pipeline_id, paso.step_id), "espera_aprobacion")
        self.assertEqual(await g.tokens_emitidos(p.pipeline_id), 1)
        fila = await store.aprobacion_obtener(b["aprobacion_id"])
        self.assertEqual(len(fila["plan"]["subobjetivos"]), 2)
        self.assertEqual(list(fila["plan"]["externas"].values()), [[a["pipeline_id"]]])
        await store.pipeline_transicion(p.pipeline_id, (PipelineStatus.running,), PipelineStatus.awaiting_approval)
        r = await delegacion.aprobar(b["aprobacion_id"], "7", {"techo_hijos_por_step_aprobado": 3})
        self.assertEqual((r["resultado"], len(r["hijos"])), ("lanzada", 2))
        dependiente = [await store.pipeline_get(h) for h in r["hijos"]]
        self.assertIn([a["pipeline_id"]], [h.context["delegacion"]["depende_de"] for h in dependiente])

    async def test_leer_solo_su_arbol(self):
        from jacobs import delegacion
        p, paso = await self.padre()
        a = await self.lanzar(p, paso)
        propio = await delegacion.leer_resultado_por_tool(p.pipeline_id, a["pipeline_id"])
        self.assertEqual((propio["ok"], propio["estado"]), (True, "queued"))
        otro, paso_otro = await self.padre()
        ajeno = await self.lanzar(otro, paso_otro)
        self.assertEqual(await delegacion.leer_resultado_por_tool(p.pipeline_id, ajeno["pipeline_id"]),
                         {"ok": False, "motivo": "fuera_de_tu_arbol"})

    async def test_padre_que_no_corre_no_delega(self):
        p, paso = await self.padre()
        await store.pipeline_transicion(p.pipeline_id, (PipelineStatus.running,), PipelineStatus.aborted)
        self.assertEqual(await self.lanzar(p, paso), {"ok": False, "motivo": "padre_no_corre"})

    async def test_con_kill_switch_no_hay_token(self):
        p, paso = await self.padre()
        with patch("jacobs.policy.check_kill_switch", return_value=True):
            r = await self.lanzar(p, paso)
        self.assertEqual((r["ok"], r["motivo"]), (False, "kill_switch_activo"))
        self.assertEqual(await g.tokens_emitidos(p.pipeline_id), 0)

    async def test_padre_abortado_corta_los_hijos_de_la_tool(self):
        from jacobs import delegacion
        p, paso = await self.padre()
        a = await self.lanzar(p, paso)
        await store.pipeline_transicion(p.pipeline_id, (PipelineStatus.running,), PipelineStatus.aborted)
        await delegacion.al_terminar_pipeline(p.pipeline_id)
        self.assertEqual(await store.pipeline_status(a["pipeline_id"]), PipelineStatus.aborted)
```

```bash
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_delegacion_io_test.py -k ToolTest" 2>&1 | tee -a $L | tail -10
```
Expected: `8 failed` (`AttributeError: delegar_por_tool`).

- [ ] **Step 3: `delegacion.py`**

Agregar a los imports: `from motor_registry.esquema_plan_delegacion import FACETA_DELEGADORA, SubObjetivo`.

Reemplazar la firma y el cuerpo final de `delegar` por:

```python
async def delegar(
    padre: Pipeline, paso: Step, plan: PlanDelegacion, *, por: str = "auto",
    externas: dict[str, list[str]] | None = None, transicionar_padre: bool = True,
) -> Delegacion:
```
El cuerpo queda igual hasta `excedidos = evaluar_techos(...)`. Las dos últimas líneas pasan a:

```python
    externas = externas or {}
    if excedidos:
        return await _pedir_aprobacion(padre, paso, plan, raiz, excedidos, externas, transicionar_padre)
    return await _lanzar(padre, paso, plan, raiz, por, externas)
```

En `_pedir_aprobacion`, la firma suma `externas: dict[str, list[str]], transicionar_padre: bool`. Las dos llamadas que guardan el plan usan `_plan_guardable(plan, externas)`. La transición del padre queda así:

```python
    if transicionar_padre and not await store.pipeline_transicion(
        padre.pipeline_id, (PipelineStatus.running,), PipelineStatus.awaiting_approval,
    ):
        return Delegacion(ResultadoDelegacion.RECHAZADA, "padre_no_corre", aprobacion_id=aprobacion_id)
```

Agregar antes de `_pedir_aprobacion`:

```python
def _plan_guardable(plan: PlanDelegacion, externas: dict[str, list[str]]) -> dict:
    """El plan que espera aprobación. `externas` (modo 2) son dependencias de
    hermanos YA lanzados: no son ids del plan, así que viajan aparte."""
    datos = plan_a_dict(plan)
    if externas:
        datos["externas"] = externas
    return datos
```

En `_lanzar`, la firma suma `externas: dict[str, list[str]]`. La entrada `"depende_de"` del contexto pasa a `[por_id[d] for d in so.depende_de] + externas.get(so.id, [])`. `sin_dependencias` pasa a `[por_id[so.id] for so in plan.subobjetivos if not so.depende_de and not externas.get(so.id)]`. Justo después de `await store.pipeline_fijar_cola(sin_dependencias, time.time())`:

```python
    # Modo 2: un hijo que depende de hermanos ya lanzados arranca cuando ellos
    # terminan; si ya terminaron todos, se resuelve ahora (el hook ya pasó).
    for so_id, deps in externas.items():
        for dep_id in deps:
            dep = await store.pipeline_get(dep_id)
            if dep is not None and dep.status in ESTADOS_TERMINALES:
                await _resolver_dependientes(dep)
```

En `aprobar`, reemplazar `plan = parsear_plan_delegacion(json.dumps(fila["plan"]))` y la llamada a `delegar` por:

```python
    datos = dict(fila["plan"])
    externas = datos.pop("externas", None)
    plan = parsear_plan_delegacion(json.dumps(datos))
    hecha = await delegar(padre, paso, plan, por="fernando", externas=externas)
```

Agregar al final del archivo:

```python
async def delegar_por_tool(pipeline_id: str, step_id: str, argumentos: dict) -> dict:
    """Modo 2 (spec §3.2): tool_authority llama ACÁ con argumentos ya validados
    (tools_subpipelines.validar_argumentos). Mismos techos, mismo token, misma
    cola que el modo 1: un sub-objetivo por llamada, por delegar()."""
    padre = await store.pipeline_get(pipeline_id)
    if padre is None or padre.status != PipelineStatus.running:
        return {"ok": False, "motivo": "padre_no_corre"}
    paso = next((s for s in padre.plan if s.step_id == step_id), None)
    if paso is None or paso.facet != FACETA_DELEGADORA:
        return {"ok": False, "motivo": "paso_no_es_ada"}
    hermanos = {
        h.pipeline_id for h in await store.pipelines_hijos(pipeline_id)
        if h.context.get("delegacion", {}).get("parent_step") == step_id
    }
    ajenas = [d for d in argumentos["depende_de"] if d not in hermanos]
    if ajenas:
        return {"ok": False, "motivo": f"depende_de fuera de esta tarea: {ajenas}"}

    pendiente = await store.aprobacion_pendiente_de(pipeline_id, step_id, "excede")
    if pendiente is not None:
        # Ya hay una delegación de este step esperando a Fernando: lo nuevo se
        # suma a ESA aprobación (se lanza todo junto al aprobar, o nada).
        datos = dict(pendiente["plan"])
        externas = datos.pop("externas", {})
        previo = parsear_plan_delegacion(json.dumps(datos))
        so = SubObjetivo(id=f"tool-{len(hermanos) + len(previo.subobjetivos) + 1}", objetivo=argumentos["objetivo"],
                         faceta=argumentos["faceta"], capability=argumentos["capability"],
                         depende_de=(), skip_on_fail=argumentos["skip_on_fail"])
        if argumentos["depende_de"]:
            externas[so.id] = list(argumentos["depende_de"])
        nuevo = PlanDelegacion(previo.subobjetivos + (so,), previo.integracion_objetivo)
        await store.aprobacion_actualizar_plan(pendiente["id"], pendiente["techos"], _plan_guardable(nuevo, externas))
        return {"ok": True, "estado": "pendiente_de_aprobacion", "aprobacion_id": pendiente["id"]}

    so = SubObjetivo(id=f"tool-{len(hermanos) + 1}", objetivo=argumentos["objetivo"], faceta=argumentos["faceta"],
                     capability=argumentos["capability"], depende_de=(), skip_on_fail=argumentos["skip_on_fail"])
    # Texto para el LLM (no es UI): el objetivo de la integración del modo 2 es
    # la tarea del step que lanzó los hijos.
    integracion = f"Integrá los resultados de los sub-pipelines que lanzaste para esta tarea: {paso.input.get('prompt', '')}"
    externas = {so.id: list(argumentos["depende_de"])} if argumentos["depende_de"] else None
    hecha = await delegar(padre, paso, PlanDelegacion((so,), integracion), externas=externas, transicionar_padre=False)
    if hecha.resultado is ResultadoDelegacion.LANZADA:
        return {"ok": True, "estado": "encolado", "pipeline_id": hecha.hijos[0]}
    if hecha.resultado is ResultadoDelegacion.ESPERA_APROBACION:
        return {"ok": True, "estado": "pendiente_de_aprobacion", "aprobacion_id": hecha.aprobacion_id}
    return {"ok": False, "motivo": hecha.motivo}


async def leer_resultado_por_tool(pipeline_id: str, hijo_pipeline_id: str) -> dict:
    """Spec §3.2: solo lee pipelines de su propio árbol (misma raíz, no él mismo)."""
    padre = await store.pipeline_get(pipeline_id)
    hijo = await store.pipeline_get(hijo_pipeline_id)
    if padre is None or hijo is None or hijo.pipeline_id == padre.pipeline_id or raiz_de(hijo) != raiz_de(padre):
        return {"ok": False, "motivo": "fuera_de_tu_arbol"}
    resumen = await _resumen_de_hijo(hijo, config_delegacion().resumen_hijo_caracteres)
    return {"ok": True, "pipeline_id": hijo.pipeline_id, "estado": hijo.status.value,
            "resumen": resumen.resumen, "output_ref": resumen.output_ref}


async def resultado_de_tools(pipeline_id: str, step_id: str) -> str | None:
    """Lo que dejaron las tools de un step, para que el executor decida después
    de la ola (igual que con el step delegate del modo 1)."""
    if await store.aprobacion_pendiente_de(pipeline_id, step_id, "excede") is not None:
        return ResultadoDelegacion.ESPERA_APROBACION.value
    if await store.hijos_del_paso_contar(pipeline_id, step_id) > 0:
        return ResultadoDelegacion.LANZADA.value
    return None
```

- [ ] **Step 4: Executor, dispatch, worker y autoridad**

`jacobs/executor.py::_run_one_step`: inmediatamente después del bloque `if plan_delegacion is not None: ...` (Task 9):

```python
        if plan_delegacion is None and step.facet in _MOTOR_FACETS:
            # Frente G modo 2: el step pudo lanzar sub-pipelines por tool.
            from jacobs import delegacion
            por_tools = await delegacion.resultado_de_tools(pipeline.pipeline_id, step.step_id)
            if por_tools is not None:
                pipeline.context[f"delegacion_{step.step_id}"] = por_tools
```

`jacobs/executor.py::run_pipeline`, en el bloque de la Task 9 (punto 9), reemplazar `if "espera_aprobacion" in resultados: return` por:

```python
        if "espera_aprobacion" in resultados:
            # Modo 1: delegacion ya lo pasó a awaiting_approval. Modo 2: la tool
            # no puede (F exige padre running mientras el step consume tokens),
            # así que se hace acá, al cerrar la ola.
            await store.pipeline_transicion(pipeline_id, (PipelineStatus.running,), PipelineStatus.awaiting_approval)
            return
```

`jacobs/executor.py::_invoke_motor`, en el `payload`:

```python
        # Frente G modo 2: las tools de sub-pipelines necesitan saber de qué
        # pipeline y step de Jacobs salen (tool_authority las verifica contra la base).
        "pipeline_id": pipeline.pipeline_id,
        "step_id":     step.step_id,
```

`las_manos/motor_registry/models.py::MotorDispatchRequest`, después de `root_pipeline_id`:

```python
    # Frente G modo 2: pipeline y step de Jacobs que pidieron el job. No dan
    # autoridad: delegacion.py verifica en la base que el pipeline corre y que
    # el step es de ada antes de emitir un token (contrato F).
    pipeline_id: str | None = None
    step_id: str | None = None
```

`las_manos/motor_registry/routes.py::dispatch`, en `motor_worker.run(...)`: `pipeline_id=req.pipeline_id, step_id=req.step_id,`.

`las_manos/motor_registry/worker.py`: agregar los imports `from motor_registry.tools_subpipelines import TOOLS_SUBPIPELINES` y `from motor_registry.tool_authority import TOOL_CAPABILITY_MAP`. Antes de `run`:

```python
def herramientas_para(motor: str, catalog: MotorCatalog) -> list[dict]:
    """Frente G (Discrepancia 16): cada tool solo a los motores que la
    capability que la tool mapea admite en capability_motor. has_tool_access
    dice SI un motor recibe tools; esto dice CUÁLES. Para jax_local y kimi no
    cambia nada (ya estaban en file_read/file_write); ada recibe solo las de
    sub-pipelines (delegate), no read_file/write_file."""
    ofrecidas: list[dict] = []
    for tool in TOOLS_CATALOG + TOOLS_SUBPIPELINES:
        cap = catalog.get_capability(TOOL_CAPABILITY_MAP.get(tool["function"]["name"], ""))
        if cap is not None and motor in cap.allowed_motors:
            ofrecidas.append(tool)
    return ofrecidas
```

En `run`: agregar los parámetros `pipeline_id: str | None = None, step_id: str | None = None` después de `root_pipeline_id`. Reemplazar `tools_for_call = TOOLS_CATALOG if motor_entry.has_tool_access else None` por `tools_for_call = (herramientas_para(motor, catalog) or None) if motor_entry.has_tool_access else None`. En la llamada a `authorize_and_execute_tool_call(...)`, agregar `jacobs_pipeline_id=pipeline_id, jacobs_step_id=step_id,`.

`las_manos/_worker_tool_loop_test.py`: en `_MOTOR_CFG["capabilities"]`, agregar `"allowed_motors": ["jax_local"]` a `file_read` y a `file_write`. Con la regla nueva, un motor sólo recibe las tools que su `capability_motor` admite; la config de prueba no lo declaraba. En producción esa fila existe (`_seed_file_tools_capabilities`).

`las_manos/motor_registry/tool_authority.py`:

```python
from motor_registry.tools_subpipelines import NOMBRES as TOOLS_DE_SUBPIPELINES
from motor_registry.tools_subpipelines import TOOL_LANZAR, validar_argumentos as validar_argumentos_de_subpipeline

TOOL_CAPABILITY_MAP: dict[str, str] = {
    "read_file": "file_read",
    "write_file": "file_write",
    # Frente G modo 2: las dos tools del emisor de Ada se autorizan contra
    # `delegate` (allowed_callers, gate); la decisión es jacobs/delegacion.py.
    "lanzar_subpipeline": "delegate",
    "leer_resultado_subpipeline": "delegate",
}

EXECUTABLE_TOOLS: frozenset[str] = frozenset({"read_file", "write_file", *TOOLS_DE_SUBPIPELINES})
```

La firma de `authorize_and_execute_tool_call` suma `jacobs_pipeline_id: str | None = None, jacobs_step_id: str | None = None`. Inmediatamente después del bloque `if cap.requires_human_gate: ... return await _reject(...)`:

```python
    if tool_name in TOOLS_DE_SUBPIPELINES:
        return await _herramienta_subpipeline(
            job_id=job_id, tool_name=tool_name, arguments_json=arguments_json, caller=caller,
            capability=capability_key, jacobs_pipeline_id=jacobs_pipeline_id, jacobs_step_id=jacobs_step_id,
        )
```

Y la función, antes de `_read_file`:

```python
async def _herramienta_subpipeline(
    *, job_id: str, tool_name: str, arguments_json: str, caller: str, capability: str,
    jacobs_pipeline_id: str | None, jacobs_step_id: str | None,
) -> dict:
    args, motivo = validar_argumentos_de_subpipeline(tool_name, arguments_json)
    if motivo is not None:
        return await _reject(job_id=job_id, tool_name=tool_name, caller=caller, capability=capability, reason=motivo)
    if not jacobs_pipeline_id or not jacobs_step_id:
        return await _reject(
            job_id=job_id, tool_name=tool_name, caller=caller, capability=capability,
            reason="sin pipeline de Jacobs: los sub-pipelines solo se lanzan desde un step de ada",
        )
    from jacobs import delegacion
    if tool_name == TOOL_LANZAR:
        resultado = await delegacion.delegar_por_tool(jacobs_pipeline_id, jacobs_step_id, args)
    else:
        resultado = await delegacion.leer_resultado_por_tool(jacobs_pipeline_id, args["pipeline_id"])
    if not resultado.get("ok"):
        return await _reject(job_id=job_id, tool_name=tool_name, caller=caller, capability=capability,
                             reason=resultado.get("motivo", "rechazado por la delegación"))
    return {"tool_name": tool_name, "decision": "executed", "reason": None,
            "content": json.dumps(resultado, ensure_ascii=False)}
```

- [ ] **Step 5: Verde, regresión y commit**

```bash
cd $WT && $PUROS PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_tools_subpipelines_autoridad.py tests/test_tools_subpipelines.py las_manos/_worker_tool_loop_test.py las_manos/_worker_max_tokens_test.py tests/test_delegacion_executor_puro.py tests/test_delegacion_uso_puro.py 2>&1 | tail -1
cd $WT && PYTHONPATH=las_manos $PY las_manos/_tool_authority_test.py 2>&1 | tail -3
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_delegacion_io_test.py tests/test_delegacion_arbol_e2e.py" 2>&1 | tail -1
cd $WT && $PY -m pytest -q policy/tests/test_no_fail_open_except.py tests/test_no_blocking_in_async.py 2>&1 | tail -1
```
Expected: todo `passed`, con `_delegacion_io_test.py` en 63 (55 + 8).

Mutación: en `delegar_por_tool`, borrar el bloque `ajenas = ...` / `if ajenas:`. Correr `-k depende_de_ajeno`. Expected: `1 failed`. Restaurar y pegar las salidas en `$L`.

```bash
git -C $WT add jacobs/delegacion.py jacobs/executor.py las_manos/motor_registry/models.py las_manos/motor_registry/routes.py las_manos/motor_registry/worker.py las_manos/motor_registry/tool_authority.py las_manos/_worker_tool_loop_test.py tests/test_tools_subpipelines_autoridad.py jacobs/_delegacion_io_test.py $L
git -C $WT commit -m "feat(motor_registry): modo 2 del emisor -- lanzar/leer sub-pipelines por tool con la misma delegacion.py

tool_authority re-autoriza cada llamada contra `delegate` y delega en
delegacion.py (mismos techos, token, cola y aprobación; dependencias de
hermanos ya lanzados; aprobación acumulada del step; lectura solo del propio
árbol). El worker ofrece cada tool solo a los motores que su capability admite.
8 puros + 8 de DB vistos en rojo; guarda de dependencias por mutación.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 23: Carga, CI, deploy (jax antes que `has_tool_access`) y verificación en vivo del modo 2

**Files:**
- Modify: `loadtest/jacobs_delegacion_app.py` (ruta `/carga/tool`), `loadtest/jacobs_delegacion.js` (escenario `tool`)
- Modify: `.github/workflows/policy.yml` (jax)
- Modify: `$WTP/backend/db/migrations.py` (`_seed_ada_has_tool_access`) y `$WTP/backend/tests/test_ada_gobernada_semillas.py`

**Interfaces:**
- Consumes: Task 22.
- Produces: `motor.has_tool_access=TRUE` para `ada` **después** de que el worker con el reparto por `allowed_motors` está en producción. Sin ese orden, ada recibiría `read_file`/`write_file` (Discrepancia 16).

- [ ] **Step 1: Carga del camino de la tool**

En `loadtest/jacobs_delegacion_app.py`, agregar:

```python
@app.post("/carga/tool")
async def carga_tool() -> dict:
    """Una llamada de lanzar_subpipeline por tool_authority (autoridad de cero +
    delegacion.delegar_por_tool), con un catálogo leído de la base de prueba."""
    import json
    from motor_registry.catalog import MotorCatalog
    from motor_registry.tool_authority import authorize_and_execute_tool_call
    padre, paso = await g.padre_delegador()
    r = await authorize_and_execute_tool_call(
        tool_name="lanzar_subpipeline",
        arguments_json=json.dumps({"objetivo": "carga", "faceta": "jekyll", "capability": "analysis"}),
        caller="jacobs", job_id="carga-tool", catalog=await MotorCatalog.from_db(),
        jacobs_pipeline_id=padre.pipeline_id, jacobs_step_id=paso.step_id,
    )
    await store.pipeline_transicion(padre.pipeline_id, (PipelineStatus.running,), PipelineStatus.waiting_children)
    return {"decision": r["decision"]}
```

En `loadtest/jacobs_delegacion.js`, dentro de `default`, antes del `else` final:

```javascript
  } else if (ESCENARIO === 'tool') {
    const r = http.post(`${BASE}/carga/tool`, null, { tags: { escenario: ESCENARIO } });
    check(r, { ejecutada: (res) => res.status === 200 && res.json('decision') === 'executed' });
```

Repetir los Steps 3-5 de la Task 13 con `ESC=tool` a c=5, 10 y 25. Umbral: p95 < 500 ms. `MotorCatalog.from_db()` corre por pedido, igual que en el dispatch real, **que recarga el catálogo por sello**. Si domina el p95, se anota con la medición y se compara con el camino real (`_ensure_catalog_fresh`, que no recarga por pedido): el número que vale para producción es el del camino real, y se dice así. Registrar la tabla en `$L`.

- [ ] **Step 2: Semilla de `has_tool_access` (jax-platform)**

`$WTP/backend/tests/test_ada_gobernada_semillas.py`, agregar:

```python
async def _tool_access(motor):
    from db.connection import get_pool
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT has_tool_access FROM motor WHERE `key`=%s", (motor,))
            return (await cur.fetchone())[0]


def test_ada_tiene_tool_access_para_el_modo_2(client):
    assert bool(client.portal.call(_tool_access, "ada")) is True
```

Rojo: `cd $WTP/backend && JAX_REPO_PATH=$WT $PYP -m pytest -q tests/test_ada_gobernada_semillas.py -k tool_access`. Expected: `1 failed`.

`$WTP/backend/db/migrations.py`, después de `_seed_ada_capability_motor`:

```python
async def _seed_ada_has_tool_access(cur) -> None:
    """Frente G etapa 3: ada recibe tools en el bucle GAP2. CUÁLES decide
    capability_motor (jax/las_manos/motor_registry/worker.py::herramientas_para):
    solo las de sub-pipelines (delegate), no read_file/write_file. Se despliega
    DESPUÉS del worker con ese reparto (orden declarado en el plan del frente G)."""
    await cur.execute("UPDATE motor SET has_tool_access=TRUE WHERE `key`='ada'")
```

Y en `run_migrations`, después de `await _seed_ada_capability_motor(cur)`: `await _seed_ada_has_tool_access(cur)`. Verde y commit en `$WTP`.

- [ ] **Step 3: CI**

- jax `tests-puros`: `tests/test_tools_subpipelines_autoridad.py` en las dos listas (+8).
- jax `jacobs-gobernanza-db`: piso +8 (`ToolTest`).
- jax-platform `backend-tests-con-db`: +3 (`test_ada_gobernada_semillas.py`).

Todo medido en el runner.

- [ ] **Step 4: PRs, canarios, merge y deploy en ESTE orden (controlador)**

1. **jax primero:**
   - revisión, rebase, PR y CI en verde;
   - canario: en `herramientas_para`, reemplazar la condición por `if cap is not None:` → rojo en `tests-puros` (`test_cada_motor_recibe_solo…`), y revert;
   - merge;
   - 0 en vuelo, pull y restart de `jax-las-manos` (Task 15, Steps 1 y 4);
   - verificar en producción que ada **todavía** no recibe tools: `SELECT has_tool_access FROM motor WHERE \`key\`='ada'` → `0`.
2. **jax-platform después:**
   - revisión, PR y canario: quitar la llamada a `_seed_ada_has_tool_access` → rojo en `backend-tests-con-db`, y revert;
   - merge;
   - backup verificado de `motor` (mismo procedimiento de restauración que la Task 15, con `g_restore_motor`);
   - restart de `jax-platform`;
   - verificar `has_tool_access=1` para `ada` y que `jax_local` y `kimi` quedaron igual que antes (anotar los valores previos del backup);
   - LAS MANOS recarga el catálogo por el sello (`_ensure_catalog_fresh`), sin reiniciar.

- [ ] **Step 5: Verificación en vivo del modo 2 (controlador; entra en el GO de deploys A-G del spec)**

Con 0 en vuelo, por la API de Jacobs y con la identidad de Fernando (igual que la Task 16, Step 2):

```bash
cat > $S/tool.json <<JSON
{"name": "verificacion-frente-g-modo-2", "objective": "Verificación del modo 2 del emisor",
 "invoked_by": "plataforma", "user_id": "$UID_F", "tenant_id": "$TID_F", "mode": "autonomous",
 "steps": [{"facet": "ada", "capability": "design",
   "prompt": "Usá la herramienta lanzar_subpipeline UNA vez para lanzar un sub-pipeline con la faceta jekyll y la capability analysis cuyo objetivo sea explicar en tres líneas qué es una cola FIFO. Después respondé solo: listo."}]}
JSON
curl -s -X POST http://127.0.0.1:7777/jacobs/pipeline -H 'Content-Type: application/json' -d @$S/tool.json | tee $S/tool-creado.json
```
Vigilar `motor_jobs.jsonl` y la base, en sólo lectura. Expected:
- el job de `ada` pide `lanzar_subpipeline`;
- el evento `TOOL_CALL_REJECTED` no aparece;
- el padre pasa por `waiting_children`;
- el hijo `jekyll/analysis` sale con `SUBPIPELINE_ADMITIDO` y queda `completed`;
- el padre corre `integrate` y queda `completed`;
- el consumo del árbol cuadra contra `axioma_usage` de la ventana (`request_type IN ('motor','pipeline')`).

Si GLM no llama la tool, **eso es un resultado**: se anota con el `_tool_loop_history` del job y se le cuenta a Fernando. No se reescribe el prompt hasta que funcione.

---

### Task 24: Biblioteca final, archivo del ledger y limpieza

**Files:**
- Modify: `/home/fruiz/jax/DEUDA.md`, `/home/fruiz/jax/CONTEXT.md` (rama `docs/emisor-subpipelines-ada-etapas-2-3`)
- Modify: `/home/fruiz/.claude/projects/-home-fruiz/memory/jax-emisor-subpipelines-ada-2026-09-16.md`

- [ ] **Step 1: Entradas según el gate**

- **Si la Task 19 NO pasó:**
  - **HISTORIA:** la medición, con fecha, modelo, `validas/5`, invenciones y la ruta de lo crudo en `/home/fruiz/mediciones/…`;
  - **DECISIÓN (spec §2):** el modo 1 queda operativo y el modo 2 espera;
  - **PENDIENTE (2026-10-16):** re-medir con `scripts/medir_tool_calling_glm.py`, o antes si cambia el `facet_binding` de `ada`, con el comando exacto;
  - **VERDAD OPERACIONAL:** ada sigue en `HTTP_FACETS`, y ningún motor recibe las tools de sub-pipelines.
- **Si pasó:**
  - **HISTORIA:** la medición, la auditoría de capabilities de ada con su tabla de 8 checks, la migración y el modo 2;
  - **VERDAD OPERACIONAL:** `MOTOR_FACETS`, las filas de `capability_motor`, `has_tool_access`, el job real de la Task 21 (Step 6) y el árbol por tool de la Task 23 (Step 5);
  - los números de carga del escenario `tool`;
  - los canarios;
  - el orden de deploy y por qué (Discrepancia 16).
- **En los dos casos:** las lecciones en `CONTEXT.md` §9 y la línea de memoria actualizada, con tipo y fecha.

- [ ] **Step 2: PR de docs, archivo y limpieza (controlador)**

PR de docs con el mismo procedimiento que la Task 17, Step 2. Después:
```bash
git -C /home/fruiz/jax pull --ff-only
cp -a $WT/.superpowers/sdd/2026-09-16-emisor-subpipelines-ada /home/fruiz/jax/.superpowers/sdd/
git -C /home/fruiz/jax worktree remove /home/fruiz/worktrees/jax-frente-g
git -C /home/fruiz/jax-platform worktree remove /home/fruiz/worktrees/jax-platform-frente-g
git -C /home/fruiz/jax worktree remove /home/fruiz/worktrees/jax-docs-frente-g
git -C /home/fruiz/jax branch -d feat/emisor-subpipelines-ada docs/emisor-subpipelines-ada
git -C /home/fruiz/jax-platform branch -d feat/emisor-subpipelines-ada
```
Cerrar los subagentes entregados. El backup del frontend en la VM se borra en la fecha anotada en la Task 17, no antes.

---

## Self-review (hecho al escribir el plan)

**1. Cobertura del spec:**

| Requisito del spec | Dónde se cumple |
|---|---|
| §1 Ada lanza sola dentro de techos; lo que excede pide aprobación y notifica; nunca rechazo silencioso | Task 7 (`delegar`, `_pedir_aprobacion` con Telegram y evento), Task 8 (cruce), Tasks 10-11 (aprobación) |
| §2 dos modos permanentes; orden modo 1 → migración + medición → modo 2; si GLM no pasa se registra con fecha | Etapas 1-3; gate en la Task 19; registro en la Task 24 |
| §3.1.1 step de Ada con capability `delegate` (fila, `allowed_callers` jacobs, `output_schema`) | Task 1 (semilla), Task 9 (`_check_delegacion`, `_dispatch_delegate`) |
| §3.1.2 validación estricta de `plan_delegacion.v1`; faceta contra `facet`; capability contra `capability`; `depende_de` sin ciclos y sólo al mismo plan | Task 3; `errores_de_catalogo` con `governance["facets"]` de E-03 |
| §3.1.3 `delegacion.py` único punto de decisión; proyección; token de F; `create_pipeline` endurecido; exceso → `awaiting_approval` + evento + Telegram sin tokens | Tasks 6 y 7 |
| §3.1.4 cola `queued` por límite global, despachador FIFO con cadencia configurable, espera máxima → `expired` | Task 8 (Discrepancias 7, 8 y 12) |
| §3.1.5 integración con id, sub-objetivo, estado, resumen acotado y `output_ref` | Task 6 (`armar_prompt_integracion`), Task 8 (`reanudar_padre_si_corresponde`) |
| §3.2 tools, `tool_authority` con la misma `delegacion.py`; leer sólo el propio árbol; prerrequisitos (a)-(d) | Tasks 18 y 22; (a) Discrepancia 1 + Task 20; (b) Task 21; (c) Task 23; (d) Task 19 |
| §3.3 jax-platform nunca emite tokens ni fija profundidad; kill switch corta el árbol y frena la cola | F (sin cambios); Task 8 (`despachar` con freno), Task 9 (cascada), B (freno en vuelo) |
| §4 datos (`root_pipeline_id`, `queued`, `queued_at`, `jacobs_arbol_consumo` en la misma transacción, techos en `.env` validados, eventos) | Tasks 2, 4 y 5; Global Constraints |
| §5 cruce a mitad, aprobación por árbol auditada, plan inválido con un reintento, hijo fallado con `skip_on_fail`, padre abortado sin huérfanos, DB caída → no delega | Tasks 7-10 y 12 |
| §6 matriz del emisor | Task 12 (E2E) + tests de las Tasks 7-10 |
| §6 carga con p95 registrado | Tasks 13 y 23 (y k6 del proxy en la Task 16) |
| §6 en vivo: árbol chico con techos bajados para forzar una aprobación | Task 16 |
| §6 modo 2: medición (≥ 4/5 válidas, 0 invenciones) y las mismas pruebas por la tool | Tasks 18-19 y 22 (`ToolTest`) |
| Reglas comunes: TDD, i18n, tema, diálogos propios, sin hardcoding, fail-closed, cuatro del rendimiento, barrera de DB, CI con canario, mirror-sync, deploy con 0 en vuelo, Biblioteca | Global Constraints; cada tarea; Task 12 (mirror-sync); Tasks 14/21/23 (canarios); Tasks 15/21/23 (deploy); Tasks 17/24 |

**2. Placeholders:** los únicos valores que el plan no puede conocer de antemano son datos de ejecución y quedan con su procedimiento exacto:
- la fecha de ejecución (`<fecha>` en comentarios de semillas);
- las capabilities sin fila de ada que salgan de la auditoría (Task 20, Step 1, con la regla de decisión y el SQL);
- la identidad de Fernando (Task 16, Step 2, con el SELECT y la confirmación);
- los pisos de CI, que por regla se miden en el runner y no se suman a mano.

No hay "TBD" ni pasos sin código.

**3. Consistencia de nombres:**

| Nombre | Dónde se define | Dónde se usa, con la misma firma |
|---|---|---|
| `delegar(padre, paso, plan, *, por, externas, transicionar_padre)` | Tasks 7 y 22 | Tasks 9, 10, 22 y 23 |
| `ResultadoDelegacion` (`lanzada`, `espera_aprobacion`, `rechazada`) | Task 7 | executor, Tasks 9 y 22 |
| `store.pipeline_transicion(pipeline_id, desde, hacia)` | Task 4 | Tasks 7-10, 12, 13 y 22 |
| `store.SQL_*` | Task 4 | tests de EXPLAIN y Task 15 |
| `Consumo`, `evaluar_techos`, `techos_cruzados`, `techos_no_cubiertos` | Task 6 | Tasks 7, 8 y 10 |
| `routes._crear_pipeline` / `crear_hijo_de_ada` | Task 7 | Tasks 12 y 13 |
| `cola.despachar` / `lanzar_en_fondo` | Task 8 | Tasks 8-10 |
| `validar_argumentos`, `TOOLS_SUBPIPELINES`, `NOMBRES` | Task 18 | Tasks 22 y 23 |

Columnas y claves:
- `techo_*_aprobado`: columnas (Task 4) = `CAMPO_APROBADO` (Task 6) = claves de `aprobar` (Task 10). Los campos HTTP `techo_tokens`/`techo_usd`/`techo_pipelines`/`techo_hijos_por_step` se traducen en la ruta de Jacobs y se usan igual en jax-platform (Task 11).
- Eventos: los nombres de Global Constraints coinciden con los que emiten y buscan los tests.

**4. Cuentas de tests:**

| Archivo | Tests |
|---|---|
| `tests/test_config_delegacion.py` | 17 |
| `tests/test_plan_delegacion_esquema.py` | 25 |
| `tests/test_delegacion_modelos_puro.py` | 4 |
| `tests/test_delegacion_uso_puro.py` | 5 |
| `tests/test_delegacion_techos_puro.py` | 12 |
| `tests/test_delegacion_politica_puro.py` | 3 |
| `tests/test_delegacion_executor_puro.py` | 8 |
| Total etapa 1 en `tests-puros` | **+69** |
| `las_manos/_output_validator_test.py` | +4 |
| `jacobs/_delegacion_io_test.py` | 55 en la etapa 1 (esquema 4, consultas 9, EXPLAIN 5, uso 5, delegar 8, cola 4, árbol 7, ejecución 5, aprobación 8); +8 en la etapa 3 = 63 |
| `tests/test_delegacion_arbol_e2e.py` | 6 |
| jax-platform backend | 4 + 8 (etapa 1), +2 +1 (etapas 2-3) |
| jax-platform vitest | 7 (etapa 1), +1 (etapa 2) |
| Etapa 2 | tools 12, medición 11, ada gobernada 3 |
| Etapa 3 | autoridad 8 |
