# El árbitro devuelve el trabajo mal hecho — reporte de implementación

**Fecha:** 2026-09-18
**Rama:** `feat/arbitro-devuelve` (worktree `/home/fruiz/worktrees/jax-devolucion`)
**Spec:** `docs/superpowers/specs/2026-09-18-arbitro-devuelve-design.md`
**Caso real:** pipeline `e570ac1c-8cae-4423-b397-d354f30b4328` (verificado contra
`jacobs_pipelines`/`jacobs_steps` en la base real, `jax_memory`, solo lectura)

## 1. Qué se construyó

Ningún mecanismo nuevo de reintento: una devolución es `store.continuar_transaccion`
(la MISMA función que `jacobs/continuar.py` usa para revivir pipelines aborted/expired)
llamada con el paso que el árbitro señaló — y todo lo que depende de él — invalidado,
la crítica inyectada en su prompt, y la época subida.

### Archivos nuevos

- `jacobs/veredicto.py` — `parsear_veredicto(texto, n_pasos_totales) -> VeredictoArbitro | None`.
  Lee un bloque ` ```veredicto ` al final de la respuesta del árbitro. Fallo cerrado:
  sin bloque, JSON roto, clave faltante, `paso` fuera de rango (o apuntando al árbitro
  mismo), `motivo` vacío o `cita` que no coincide con `paso` -> `None`, nunca una excepción.
- `jacobs/devolucion.py` — `evaluar_y_devolver(pipeline) -> (resultado, payload)`.
  Se llama al terminar la última ola de `_correr_pipeline` (la del árbitro, que
  depende de todos los demás pasos y por eso siempre corre sola al final), antes de
  marcar el pipeline `completed`. `pasos_afectados()` calcula el cierre transitivo de
  dependientes; `_inyectar_critica()` agrega la crítica al `input["prompt"]` del paso
  devuelto (nunca toca `facet`: el árbitro no reescribe, §3.5).
- `jacobs/_veredicto_test.py` (20 tests, puro) — incluye, como PRIMER test, la prosa
  REAL del step 6 de e570ac1c contra `parsear_veredicto`: da `None` porque hoy no
  trae bloque `veredicto` (rojo del caso real, antes de que existiera el módulo).
- `jacobs/_devolucion_test.py` (13 tests, puro, mockea `jacobs.store` y
  `jacobs.prevuelo.prevuelo`) — los 5 escenarios del spec §5.
- `jacobs/_devolucion_persistencia_test.py` (10 tests, DB real vía
  `base_de_test.py`) — migración idempotente, persistencia del tope por los DOS
  caminos de creación reales, `get_tope_devoluciones()` fail-closed, y el control
  que prueba que `aplicar_cupo=False` evita que una devolución se frene contra su
  propio cupo (con `aplicar_cupo=True` en la MISMA situación saturada, sale
  `CupoAgotado` — freno confirmado, no solo declarado).

### Archivos modificados

- `jacobs/plan.py` — `PROMPT_ARBITRO` pide, además de la prosa, el bloque
  ` ```veredicto ` con `{"decision": "aprobar"}` o `{"decision": "devolver", "paso",
  "motivo", "cita"}`.
- `jacobs/executor.py` — `_correr_pipeline`, al terminar todas las olas, llama a
  `devolucion.evaluar_y_devolver` (import perezoso dentro de la función — evita el
  ciclo executor->devolucion->executor, mismo patrón que ya usa `jacobs/continuar.py`
  con `_load_ref`/`RefIlegible`). `RESULTADO_DEVUELTO` actualiza el `Pipeline` en
  memoria y vuelve a llamar `_correr_pipeline` (misma corrida, nueva época — no se
  agenda un background task nuevo, ya estamos en uno). `RESULTADO_COMPLETAR`/
  `RESULTADO_TOPE` siguen el camino de siempre.
- `jacobs/models.py` — `Pipeline.costo_max_aceptado_usd: Decimal | None` y
  `Pipeline.devoluciones: int = 0`.
- `jacobs/store.py`:
  - migración `jacobs_pipelines`: `costo_max_aceptado_usd DECIMAL(12,4) NULL` y
    `devoluciones INT NOT NULL DEFAULT 0` (ALGORITHM=INSTANT, en la lista de
    `init_tables()`, verificado idempotente).
  - `pipeline_create()` y `_row_to_pipeline()` incluyen las dos columnas.
  - `get_tope_devoluciones()` — lee `axioma_config.jacobs.tope_devoluciones`;
    sin fila, valor no numérico o negativo -> `0` (fail-closed, mismo criterio que
    `arbitro_faceta`).
  - `continuar_transaccion()` — 4 parámetros nuevos, todos keyword-only con default
    que reproduce el comportamiento de siempre (ningún caller existente cambia):
    `evento_tipo` ("PIPELINE_DEVUELTO" en vez de "PIPELINE_CONTINUED"),
    `aplicar_cupo` (False para una devolución — ver más abajo por qué),
    `incrementar_devoluciones`, `costo_max_aceptado_usd` (persiste un tope nuevo
    dado en un `/continue` humano, no solo lo valida para ese pedido).
- `jacobs/cupo.py` — `SQL_COMPLETAR`/`completar_reserva()` ahora escriben
  `costo_max_aceptado_usd`. **Este es el fix real**: `routes.py` NO usa
  `store.pipeline_create()` para crear pipelines — usa `cupo.reservar_cupo()` +
  `cupo.completar_reserva()`. Verificado contra el código: ninguno de los dos
  escribía `costo_max_aceptado_usd` antes de esta ronda (coincide exactamente con
  lo que el spec §3.4 daba por verificado).
- `jacobs/continuar.py` — pasa `costo_max_aceptado_usd` a `continuar_transaccion()`
  y al `Pipeline` devuelto, para que un `/continue` humano con un tope nuevo quede
  disponible para una devolución automática posterior.
- `jacobs/routes.py` — la creación real (`Pipeline(...)` antes de
  `cupo.completar_reserva`) pasa `costo_max_aceptado_usd=req.costo_max_aceptado_usd`.

## 2. Los cinco puntos del spec

1. **Veredicto accionable, fallo cerrado** — `jacobs/veredicto.py`. Sin estructura
   válida, `jacobs/devolucion.py` deja el evento `VEREDICTO_SIN_ESTRUCTURA` y el
   pipeline completa como hoy.
2. **La crítica viaja en el contexto** — `_inyectar_critica()` la pone en
   `input["prompt"]` del paso devuelto; el ejecutor arma ese prompt en
   `_build_context_input`/`_enrich_prompt` exactamente igual que con cualquier otro
   dato de entrada.
3. **Tope de dos, en configuración** — `axioma_config.jacobs.tope_devoluciones`,
   leído por `store.get_tope_devoluciones()` en el momento en que se evalúa el
   veredicto (no cacheado desde el plan-build). **Pendiente de un paso manual**:
   la fila no existe hoy en `axioma_config` (ni en `jax_memory` de producción ni en
   la plantilla de test) — sin ella el tope es `0` (fail-closed): el código está
   listo, pero el sistema no devuelve nada hasta que alguien inserte
   `('jacobs.tope_devoluciones', '2')`. Deliberado: mergear esto no empieza a
   gastar presupuesto solo, hace falta una decisión explícita.
4. **`costo_max_aceptado_usd` persistido** — por los dos caminos reales de
   creación (`cupo.completar_reserva`, que es el que usa `POST /jacobs/pipeline`) y
   por `/continue`. **Declarado, no verificado más allá de lo que ya hacía
   `continuar()`**: la comparación es contra el ESTIMADO de prevuelo para los pasos
   afectados vs. el tope COMPLETO, no un saldo neto de gasto acumulado — Jacobs no
   lleva ese ledger por pipeline hoy (mismo criterio que ya usa
   `continuar()::costo_supera_lo_aceptado`, no una regresión de esta ronda).
5. **El árbitro devuelve, no reescribe** — `_inyectar_critica()` nunca toca
   `paso.facet`; lo rehace la MISMA faceta que lo produjo. Probado en
   `test_el_arbitro_no_reescribe_solo_devuelve`.

## 3. Decisiones de diseño donde el spec dejó margen

- **Formato del veredicto**: un bloque ` ```veredicto ` con JSON, no un campo
  separado del contrato del step ni un `output_schema` nuevo — la `capability`
  "critique" ya trae `output_schema='critique.v1'` pero no se valida en ningún
  lado del código actual; no se construyó infraestructura de validación de schema
  para esto.
- **`aplicar_cupo=False` para la devolución**: el pipeline que se devuelve está
  `running` AHORA MISMO — ya cuenta en `SQL_JOIN_CUPO`. Reusar el UPDATE
  cupo-gateado de `continuar_transaccion` tal cual habría hecho que, con el cupo
  exactamente lleno, un pipeline se negara a devolverse A SÍ MISMO (`cupo_x.c < N`
  da `False` aunque nadie pida un lugar nuevo). Verificado con un test que prueba
  las dos ramas: con `aplicar_cupo=False` la devolución avanza en cupo saturado;
  con `aplicar_cupo=True`, en la MISMA situación, sale `CupoAgotado` (control,
  Principio VII).
- **Recursión en memoria en vez de un nuevo background task**: `_correr_pipeline`
  se vuelve a llamar a sí misma tras una devolución exitosa, en vez de agendar un
  `run_pipeline` nuevo — ya estamos dentro de un background task, y un `/continue`
  real de todas formas dispara exactamente esa misma re-entrada
  (`_correr_pipeline` desde cero, con `PIPELINE_STARTED` de nuevo). El tope de
  devoluciones acota la profundidad de la recursión.
- **"Las dos versiones" del §3.3** — no se construyó un mecanismo de diff ni de
  versionado nuevo: la versión previa y la posterior quedan en `jacobs_events`
  (cada `PIPELINE_DEVUELTO` trae los refs invalidados) y el evento
  `DEVOLUCION_TOPE_ALCANZADO` trae la crítica del último intento. "Avisa" reusa el
  aviso de Telegram que ya dispara toda finalización de pipeline
  (`_disparar_aviso_fin`), no un canal nuevo.
- **Nombre de la config**: `jacobs.tope_devoluciones` (no `ejecutor.*`) — es del
  orquestador de Jacobs, no del Ejecutor de Contratos que también usa el prefijo
  `ejecutor.` para su propio auditor.

## 4. Verificación

- Los 2 módulos nuevos (`veredicto.py`, `devolucion.py`) confirmados
  **rojos por la razón correcta** (`ModuleNotFoundError`) antes de escribir la
  implementación.
- 43 tests nuevos, puros (33) + DB real (10), todos verdes contra este código.
- Suite de regresión dirigida (continuar, épocas, cupo, subpipelines, resume,
  preflight, relaunch CLI, árbitro, menú sin árbitro): 366 passed, 0 failed.
- `policy/tests/test_no_fail_open_except.py` y
  `policy/tests/test_archivos_de_test_wireados_en_ci.py`: verdes — los tres
  archivos de test nuevos quedan wireados a CI (dos en `tests-puros`, uno en
  `subpipeline-contrato-db`).
- Pisos de CI actualizados en el mismo commit:
  - `tests-puros`: 2192 -> 2225 passed (+33, 0 skipped nuevos). Medido LOCAL
    (hall9000) dio 2206 -> 2240 (+34, no +33) porque en este checkout
    `test_cola_uso_escritores.py` no tiene el mirror desactualizado de
    `~/jax-platform` que el baseline local documentaba como "1 failed ajeno" --
    reconciliado sumando el delta limpio (+33, verificado en aislamiento) sobre
    el número YA corregido para el runner (2192), no sobre el local. **No
    verificado contra el runner real de GitHub Actions** -- declarado.
  - `subpipeline-contrato-db`: 141 -> 151 passed, medido completo localmente
    contra `jax_memory_test_<sufijo>` con la gobernanza real (facet/capability/
    motor clonados de jax-platform en ese entorno).
- Caso real (§5, criterio 1): no re-corrido end-to-end contra un LLM real (fuera
  de alcance de esta sesión -- necesitaría lanzar el pipeline completo con
  árbitro en producción/staging). Verificado en su lugar: (a) la prosa REAL del
  pipeline e570ac1c no dispara devolución hoy (primer test de
  `_veredicto_test.py`); (b) el mismo caso, CON el bloque estructurado que el
  nuevo prompt le va a pedir al árbitro, sí dispara una devolución correcta a
  `ada` (paso 4) con la crítica inyectada y solo los pasos 4/5/6 invalidados
  (`_devolucion_test.py`).

## 5. Pendientes / no verificado

- **`axioma_config.jacobs.tope_devoluciones` no tiene fila en producción**: sin
  ella el tope es 0 y el árbitro nunca devuelve (fail-closed, a propósito). Falta
  que alguien con acceso a producción corra
  `INSERT INTO axioma_config (config_key, config_value) VALUES
  ('jacobs.tope_devoluciones', '2')` -- deliberadamente fuera de esta rama.
- Piso de `tests-puros` no confirmado contra el runner real de GitHub Actions
  (ver §4).
- Verificación en vivo del caso real completo (correr el pipeline con árbitro
  contra ADA/thot reales y confirmar que la arquitectura corregida sale sin
  BIGSERIAL/TIMESTAMPTZ) -- no hecha en esta sesión.
