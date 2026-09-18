# Pre-vuelo antes de gastar y continuar pipelines abortados (Jacobs) · Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que Jacobs rechace un pipeline ANTES de gastar (tope de salida, credencial, salud con sonda, costo máximo) y que un pipeline `aborted`/`expired` se pueda continuar reusando lo ya hecho, con un solo ejecutor por pipeline.

**Architecture:** El pre-vuelo vive en Jacobs en cuatro módulos chicos: reglas puras (`prevuelo_reglas.py`), lectura del catálogo en una conexión (`prevuelo_catalogo.py`), sonda de disponibilidad (`sonda.py`) y el orquestador (`prevuelo.py`). Crear y continuar lo corren dentro del candado. Continuar es un servicio (`continuar.py`) que usan el endpoint y el CLI. La exclusión de ejecutores es una época (`jacobs_pipelines.run_epoch`): toda escritura del ejecutor es condicional a su época y a `status='running'`.

**Tech Stack:** Python 3.12 (runner) / 3.14 (venv local), FastAPI, aiomysql, httpx, pytest, MariaDB 11.8 (CI) / 12.3 (hall9000).

**Spec:** `docs/superpowers/specs/2026-09-17-prevuelo-y-continuar-design.md` (§4, §5, §5.4, §8, §9 en lo que cae en `jax`). Leerlo entero antes de empezar. Lo de jax-platform (§6, §7, DDL) es del plan P.

---

## Desvíos del spec

Cada uno se verificó contra el código de `origin/master` (`044cb99`). Donde el spec y el código no cierran, manda el código medido; la resolución está al lado.

1. **§4.2 — la unidad de un paso de motor es el MOTOR RESUELTO, no la faceta.** `StepSpec` no tiene campo `motor` (`jacobs/models.py:134-152`), así que todo paso creado por la Mesa llega con `step.motor=None`, y el Motor Registry elige "el primero habilitado" de `capability_motor` (`las_manos/motor_registry/policy.py:171-183`; `executor.py:559`). Medir costo, tope y salud de `motor_resolved` por nombre de faceta mediría otro motor. **Resolución:** `MotorPolicy.motor_que_despacharia()` (público, envuelve `_resolve_motor`) y la clave de salud del paso es ese motor.
2. **§4.6 — `llamadas_max`.** (a) El reintento de grounding de `_invoke_http_gemini` corre para TODO paso `http_gemini`, sin mirar política (`executor.py:307-327`): ×2 siempre en ese transporte. (b) En el Motor Registry los multiplicadores no se componen: el reintento de schema consume una iteración del MISMO bucle, acotado por `MAX_TOOL_LOOP_ITERATIONS` (`worker.py:636-650`, `805-839`). **Resolución:** motor con herramientas → `MAX_TOOL_LOOP_ITERATIONS`; motor sin herramientas → `min(2 si hay reintento de schema, si no 1; MAX_TOOL_LOOP_ITERATIONS)`. "Schema con reintento" = nombre no vacío y fuera de `_KNOWN_UNIMPLEMENTED_SCHEMAS` (un nombre desconocido también reintenta, `output_validator.py:134-140`), expuesto como `output_validator.puede_pedir_reintento()`.
3. **§4.6 — `tokens_in_max`.** La fórmula del spec omite partes del prompt real: la regla de evidencia, el objetivo, los encabezados, los resúmenes de 500 caracteres de los pasos previos cuando el paso no declara `depends_on` (`executor.py:155-263`) y el contexto de identidad que antepone el Motor Registry (`worker.py:541-548`). Con ella se SUBESTIMA. **Resolución:** se arma el prompt con las MISMAS funciones del ejecutor (`_build_context_input` + `_enrich_prompt`), con cada dependencia sin ref rellenada al peor caso (`MAX_DEP_CONTEXT_CHARS`), más persona y contexto de identidad.
4. **§4.1 — firma.** `prevuelo(pasos, contexto, *, pendientes=None, user_id=None, tenant_id=None)`: armar el prompt necesita el plan COMPLETO (índices de dependencias) aunque sólo se evalúen los pendientes, y la sonda necesita identidad para registrar su uso. Los tipos (`Veredicto`, `Violacion`, `CostoPaso`, `Despacho`) viven en `jacobs/prevuelo_reglas.py` para que `sonda.py` y `prevuelo.py` no se importen en círculo.
5. **§4.4 — `sin_contrato_de_salida` también en `jax_local`/ollama.** El spec lo pide "en un paso que cobra", pero el despacho de ollama falla cerrado igual sin `max_output_tokens` (`contrato_dispatch.py:321-325`). Rechazarlo antes es el mismo fallo, sin esperar al paso.
6. **§4.7 — cuerpo de `POST /jacobs/preflight`.** Acepta además `objective`, `user_id`, `tenant_id` opcionales (el objetivo entra al prompt; la identidad atribuye el costo de la sonda). `steps` es obligatorio con al menos 1: sin pasos, `PlanBuilder.build()` planifica con un LLM pago (`plan.py:450-453`) y un pre-vuelo no puede gastar. **Kill switch: SÍ lo mira** (423 `{"code": "kill_switch", ...}`, después de validar `invoked_by` y antes de `build()`). Corregido el 2026-09-17 por la re-revisión final (I1) y el Ruling R54: este desvío decía "no mira el kill switch: no ejecuta nada", cierto cuando se escribió y falso desde que entró la sonda del §4.5, que paga una llamada al proveedor y la registra en `axioma_usage`. Un freno que no cubre el único endpoint nuevo que gasta no es un freno (Principio VII).
7. **§5.2 regla 4 y §5.4 — el candado es de proceso.** `_pipeline_create_lock` es un `asyncio.Lock` (`routes.py:45-56`); el CLI corre en otro proceso. **Resolución:** el candado se mueve a `jacobs/candado.py` y lo toman endpoint y CLI; entre procesos, la exclusión la da la transacción de continuar (`SELECT … FOR UPDATE` + `UPDATE … WHERE run_epoch=%s`). El límite de activos visto desde el CLI queda con carrera contra LAS MANOS: se registra en `DEUDA.md` con fecha (Task 14).
8. **§5.3 — también las escrituras de PASOS son condicionales.** El spec nombra `pipeline_update_status`, pero el ejecutor escribe `jacobs_steps` en `_run_one_step`, el kill switch, el gate de hyde y `_fail_step` (`executor.py:1028, 1052, 1127, 1144, 1274`). Sin condición, una corrida superada reescribe `completed`/`failed` sobre pasos que `continue` acaba de poner `pending`. **Resolución:** `store.step_upsert_si_epoca()`; un paso que pierde la época antes de despachar NO despacha.
9. **§5.3 — `approve-step` también incrementa la época.** Lanza `run_pipeline` igual que `resume` (`routes.py:305-375`); un doble approve lanzaría dos ejecutores.
10. **§5.2 regla 7 — validación del plan siempre, y marcas de hyde.** `_check_cleanroom` + `_validate_plan_capabilities` corren aunque no haya reasignación (el plan vigente puede haberse vuelto inválido); sin reasignación el código es `plan_rechazado`. Se quitan del contexto las `hyde_approved_<step_id>` de los pasos a rehacer: la aprobación humana de una corrida no autoriza la siguiente. Pasos guardados cuyo `step_index` no es `0..N-1` → 409 `plan_inconsistente`.
11. **§5.2 regla 2 — `failed` existe** en `PipelineStatus` (`models.py:30`) y el spec no lo lista: 409 como los demás no continuables.
12. **§4.7/§5.1 — forma de los errores.** `HTTPException(detail={...})` → FastAPI responde `{"detail": {"code": ..., ...}}`. Cuando el detalle es un dict se aplana junto a `code`; si es texto o lista va en `detalle`. Los montos viajan como TEXTO con 6 decimales redondeados hacia arriba (`DECIMAL(10,6)` de `axioma_usage.cost_usd`).
13. **§6.1 — `costo_max_aceptado_usd`** se compara contra la suma de los pasos acotados. Un paso no acotado no bloquea en Jacobs: la Mesa pide confirmación siempre que haya uno, y el veredicto lo declara (`hay_no_acotados`).
14. **§4.4 — la semilla de `min_output_tokens` NO trae valores en este plan.** Medirlos requiere `SELECT` sobre producción y leer `las_manos/logs/motor_jobs.jsonl` de producción, que los límites de esta ronda no autorizan. La Task 13 entrega el script y el comando; los valores se agregan a este archivo al ejecutarla con GO, y los consume la migración del plan P.
15. **§5.4 — `--from-step` desaparece del CLI.** Qué se reusa lo decide la ref legible de cada paso (regla 5), no un número de quien corre el comando.
16. **§4.5 — payload de la sonda.** Mismo endpoint, resolver y credencial que el ejecutor, pero payload mínimo propio: las `_invoke_*` mandan `google_search` y el tope COMPLETO del modelo, una sonda cara. Sólo se sondean claves cuyos pasos no tienen otra violación (el despacho se rechazaría igual).
17. **§8 "se cuenta"** — contador en proceso `sonda.registros_perdidos()`; jax no exporta métricas.
18. **§5.3 — `RUN_SUPERSEDED`** lleva `{epoca, epoca_actual, status_actual}`.
19. **§4.5 — `facet_health.py` decía "quien escribe es jax-platform".** Ahora también Jacobs (`source='preflight'`); el LECTOR sigue siendo único (este módulo).
20. **§8 — continue con error inesperado** (DB caída en el análisis o en el pre-vuelo) → 503 `prevuelo_no_disponible`, igual que crear.

## Global Constraints

- **Dependencia del plan P (jax-platform) — se mergea ANTES que esta rama.** Este plan CONSUME: `capability.min_output_tokens INT NOT NULL DEFAULT 0`, el valor `'preflight'` del ENUM `facet_health_event.source`, y `axioma_usage.request_type='preflight_probe'` (la columna es `VARCHAR(20)`: entra sin DDL). El job `jacobs-gobernanza-db` clona `jax-platform` **master** y corre `run_migrations()`: hasta que P esté en master, los tests DB de las Tasks 6, 10 y 15 dan rojo en CI por esquema, no por código. Localmente corren contra `jax_memory_test` migrada con la rama de P (Task 6, Step 1).
- **Base de tests, nunca producción.** Tests DB: `set -a; source /etc/jax/.env; set +a; export JAX_DB_NAME=jax_memory_test` en el MISMO comando. Cada archivo de tests DB lleva la barrera: si `JAX_DB_NAME` apunta a otra base, `RuntimeError` al importar. Los tests puros fuerzan `os.environ["JAX_DB_NAME"] = "jax_memory_test"` antes de importar `jacobs` y mockean todo acceso a DB (patrón `tests/test_jacobs_invoked_by_rol.py`).
- **Nada de proveedores pagos ni servicios de producción (:7777, :8080)** en tests ni en la prueba de carga. La instancia de carga es aislada (puerto 17790) y la corrida que se lanza en la prueba de `continue` para en el gate de hyde, sin despacho.
- **TDD con rojo visto.** Cada test nuevo se corre contra el código viejo ANTES de implementar y falla con el Expected escrito en su paso. Un test que ya pasa antes se declara "control" en el paso.
- **P10.** Todo `except` amplio que no relanza lleva en la MISMA línea `# fail-soft: <razón de este sitio>` o `# fail-closed: <razón>`. Lo verifica `policy/tests/test_no_fail_open_except.py`.
- **Sin hardcoding.** Timeouts, divisores y topes de la sonda por variable de entorno validada (`jacobs/prevuelo_config.py`, patrón de `jax/core/db_connect_config.py`: valor inválido → `RuntimeError` con el nombre de la variable, nunca default silencioso). Ningún límite de salida literal en un payload (`tests/test_payload_max_tokens_literal_tripwire.py`).
- **Textos en español.** Mensajes de error, eventos y detalles de violaciones.
- **Secretos.** Todo error que se guarda (evento, `facet_health_event.detail`, `jacobs_steps.error`, detalle 503) pasa por `redactar_secretos`/`recortar_redactado` (`jax/core/redaccion.py`).
- **LAS CUATRO DEL RENDIMIENTO.** Cada consulta nueva tiene su `EXPLAIN` sobre la consulta REAL en un test DB. Nada bloqueante en `async def`: lectura de artifacts y armado del prompt por `asyncio.to_thread`. Carga de `/jacobs/preflight` a c=25 medida (Task 15) antes de pedir GO.
- **CI — listas duplicadas y pisos.** En `.github/workflows/policy.yml`, cada test puro nuevo entra en las DOS listas del job `tests-puros` (la de `pytest -v` y la del paso "Piso exacto de tests CORRIDOS") y cada test DB en las DOS del job `jacobs-gobernanza-db`. El número del `grep -qE "^N passed"` sube en el mismo commit, con una línea de comentario fechada. Los números de este plan (tests-puros 742 → 858 con "1 skipped"; gobernanza-db 27 → 46) son lo que deberían dar; **el piso lo fija el runner**: si el runner cuenta otro número, se pone el del runner y se escribe por qué (Task 14).
- **Conteo local.** El venv local no es el runner. Antes de la Task 1 se mide la línea base local de la lista de `tests-puros` (Task 1, Step 1) y cada task espera `base_local + k`.
- **Git.** Solo en `/home/fruiz/worktrees/jax-prevuelo` (rama `feat/prevuelo-y-continuar`), con `git -C /home/fruiz/worktrees/jax-prevuelo` y `pwd; git branch --show-current` en el mismo comando antes de cada commit. Nunca `git stash` pelado. `git push`, PR, merge y deploy: solo con GO del controlador (Task 14/15 lo marcan). Commits terminan con `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- **Convivencia (§10).** Los frentes E/F tocan `routes.py`, `executor.py`, `store.py`, `policy.py`: antes de pedir merge, `git fetch` + rebase sobre `origin/master`, preservando lo de ellos.

**Prefijos de comando** (se usan en todos los pasos):

```bash
# PUROS
cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q <archivos>
# DB (jax_memory_test)
cd /home/fruiz/worktrees/jax-prevuelo && set -a && source /etc/jax/.env && set +a && export JAX_DB_NAME=jax_memory_test PYTHONPATH=.:las_manos && /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -v <archivos>
```


**Verificación previa del plan (2026-09-17, al escribirlo).** Todo el código de este plan se aplicó en una copia descartable del árbol (`044cb99` + spec) y se corrió: los 116 tests puros nuevos pasan y la lista de `tests-puros` da `743 → 859 passed, 1 xfailed` en local (runner: `742 → 858`); los rojos de las Tasks 3, 4 y 12 se vieron contra el código viejo con los mensajes que figuran en sus pasos; `tests/test_run_epoch_db.py` (6) y `tests/test_jacobs_continuar_db.py` (4) pasan contra `jax_memory_test`. De `tests/test_prevuelo_catalogo_db.py` se verificaron los que no dependen del esquema de P (lectura, credencial, salud y los tres EXPLAIN con `source='chat'`); los que escriben `source='preflight'` o leen `min_output_tokens` esperan a P. Esa copia no es la ejecución: cada task se vuelve a correr con su rojo en el worktree.

---

## File Structure

| Archivo | Responsabilidad |
|---|---|
| `jacobs/prevuelo_config.py` (nuevo) | Lee y valida `JAX_PREVUELO_SONDA_TIMEOUT_S`, `JAX_PREVUELO_CHARS_POR_TOKEN`, `JAX_PREVUELO_SONDA_MAX_TOKENS` |
| `jacobs/prevuelo_reglas.py` (nuevo) | Tipos del veredicto y reglas PURAS: contrato, tope efectivo, credencial, llamadas, costo |
| `jacobs/prevuelo_catalogo.py` (nuevo) | Una conexión: facetas, modelos, `min_output_tokens`, credenciales activas, salud; resolución de motores |
| `jacobs/sonda.py` (nuevo) | Llamada mínima por transporte, registro en `facet_health_event` y en `axioma_usage` |
| `jacobs/prevuelo.py` (nuevo) | Orquestador: arma despachos, mide el prompt, sondea en paralelo, arma el veredicto |
| `jacobs/candado.py` (nuevo) | `candado_de_creacion` (antes `routes._pipeline_create_lock`) |
| `jacobs/continuar.py` (nuevo) | Servicio de continuar: `analizar`, `previsualizar`, `continuar` |
| `jacobs/models.py` | `Pipeline.run_epoch`, `PipelineCreateRequest.costo_max_aceptado_usd` |
| `jacobs/store.py` | Columna `run_epoch`, escrituras condicionales, `pipeline_tomar_epoca`, `continuar_transaccion`, `step_upsert` actualiza `facet` |
| `jacobs/executor.py` | Época en `run_pipeline`/`_run_one_step`/`_fail_step`, `RUN_SUPERSEDED`, un solo `PIPELINE_ABORTED` |
| `jacobs/routes.py` | `POST /jacobs/preflight`, pre-vuelo en crear, época en resume/approve, `continue` y `continue/preflight` |
| `jacobs/facet_health.py` | Lector de último evento de proveedor, escritor de eventos de sonda |
| `jacobs/usage_writer.py` | `record_direct_usage(..., request_type=)` |
| `las_manos/motor_registry/policy.py` | `MotorPolicy.motor_que_despacharia()` |
| `las_manos/motor_registry/output_validator.py` | `puede_pedir_reintento()` |
| `tools/jacobs_relaunch.py` | CLI sin lógica propia sobre `continuar.py` |
| `scripts/medir_min_output_tokens.py` (nuevo) | Medición de solo lectura para la semilla de `min_output_tokens` |
| `.github/workflows/policy.yml` | Listas y pisos de `tests-puros` y `jacobs-gobernanza-db` |

---

### Task 1: Configuración del pre-vuelo

**Files:**
- Create: `jacobs/prevuelo_config.py`
- Test: `tests/test_prevuelo_config.py`
- Modify: `.github/workflows/policy.yml` (job `tests-puros`, las dos listas y el piso)

**Interfaces:**
- Consumes: nada.
- Produces: `prevuelo_config.sonda_timeout_s() -> int` (default 20), `prevuelo_config.chars_por_token() -> int` (default 2), `prevuelo_config.sonda_max_tokens() -> int` (default 16); constantes `SONDA_TIMEOUT_S`, `CHARS_POR_TOKEN`, `SONDA_MAX_TOKENS` con el nombre de cada variable. Valor no entero o ≤ 0 → `RuntimeError` que nombra la variable.

- [ ] **Step 1: Medir la línea base local de `tests-puros`**

Run (la lista exacta del paso "Piso exacto" del job `tests-puros`, tal como está hoy):
```bash
cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q \
  tests/test_capability_unbound_schema.py tests/test_chunking.py tests/test_completeness_intent.py tests/test_dispatch_step_reroute.py \
  tests/test_identity_context.py tests/test_jacobs_director.py tests/test_memory_correction.py tests/test_memory_helpers.py \
  tests/test_read_audit_log_unbound.py tests/test_template_hash.py tests/test_validate_capability_typed.py \
  tests/test_jacobs_timeout_by_capability.py tests/test_memory_schema_drift.py tests/test_no_blocking_in_async.py tests/test_load_test_harness.py \
  tests/test_hnsw_recall_tripwire.py tests/test_motor_job_cancel_and_length.py tests/test_invoke_motor_cancel_on_timeout.py \
  tests/test_invoke_motor_rechazo.py tests/test_motor_contexto_y_salida.py tests/test_hipatia_fuentes.py tests/test_motor_catalog_sello.py \
  tests/test_repl_fuentes.py tests/test_memory_embedding_config.py tests/test_fuentes_dedup.py tests/test_repl_modelos_permitidos.py \
  tests/test_motor_catalog_mode.py tests/test_jacobs_invoked_by_rol.py tests/test_catalog_from_db_timeout.py \
  tests/test_aiomysql_connect_timeout_tripwire.py tests/test_contrato_dispatch_repl_ada.py tests/test_payload_max_tokens_literal_tripwire.py \
  tests/test_motor_contrato_dispatch.py tests/test_ejecutor_fase0.py tests/test_ejecutor_cita.py tests/test_ejecutor_captura.py \
  tests/test_ejecutor_hechos.py tests/test_ejecutor_herramientas.py tests/test_ejecutor_prioridad.py tests/test_ejecutor_transporte.py \
  tests/test_ejecutor_proxy_carril.py tests/test_jacobs_ref_ilegible_no_es_exito.py tests/test_memoria_no_traga_fallos_de_escritura.py \
  tests/test_reaper_reconciliacion_no_reporta_verde_si_fallo.py tests/test_health_de_las_manos_puede_fallar.py tests/test_degradaciones_declaradas.py \
  las_manos/_worker_max_tokens_test.py las_manos/_worker_tool_loop_test.py tests/test_redaccion.py tests/test_gemini_key_en_cabecera.py \
  tests/test_store_indice_duenio.py tests/test_cola_uso_escritores.py tests/test_respaldo_de_uso_aislado.py 2>&1 | tail -3
```
Expected: todo verde. Anotar la línea final (`N passed, S skipped, X xfailed`) en el ledger de la ejecución como **BASE_LOCAL**. Medido al escribir este plan (2026-09-17, venv de `las_manos`, Python 3.14): `743 passed, 1 xfailed` — el skip del runner corre en local porque hay checkout de jax-platform. El runner dice `742 passed, 1 skipped, 1 xfailed`; si local difiere (p. ej. el skip de `test_la_copia_es_identica_al_canonico_de_jax_platform` corre porque hay checkout de jax-platform), se usa BASE_LOCAL para las comparaciones locales de todo el plan.

- [ ] **Step 2: Escribir el test que falla**

`tests/test_prevuelo_config.py`:
```python
"""Configuración del pre-vuelo (spec 2026-09-17 §4.5, §4.6): se lee en cada
llamada y un valor inválido FALLA con el nombre de la variable, nunca cae a un
default silencioso (mismo trato que JAX_DB_CONNECT_TIMEOUT_SECONDS).

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import os

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import pytest  # noqa: E402

from jacobs import prevuelo_config as pc  # noqa: E402


def test_timeout_de_sonda_por_defecto(monkeypatch):
    monkeypatch.delenv(pc.SONDA_TIMEOUT_S, raising=False)
    assert pc.sonda_timeout_s() == 20


def test_chars_por_token_por_defecto_es_2(monkeypatch):
    monkeypatch.delenv(pc.CHARS_POR_TOKEN, raising=False)
    assert pc.chars_por_token() == 2


def test_max_tokens_de_sonda_por_defecto(monkeypatch):
    monkeypatch.delenv(pc.SONDA_MAX_TOKENS, raising=False)
    assert pc.sonda_max_tokens() == 16


def test_valor_valido_del_entorno_manda(monkeypatch):
    monkeypatch.setenv(pc.CHARS_POR_TOKEN, "3")
    assert pc.chars_por_token() == 3


def test_no_numerico_lanza_con_el_nombre(monkeypatch):
    monkeypatch.setenv(pc.SONDA_TIMEOUT_S, "veinte")
    with pytest.raises(RuntimeError, match="JAX_PREVUELO_SONDA_TIMEOUT_S"):
        pc.sonda_timeout_s()


def test_cero_lanza(monkeypatch):
    monkeypatch.setenv(pc.SONDA_MAX_TOKENS, "0")
    with pytest.raises(RuntimeError, match="mayor que cero"):
        pc.sonda_max_tokens()


def test_negativo_lanza(monkeypatch):
    monkeypatch.setenv(pc.CHARS_POR_TOKEN, "-2")
    with pytest.raises(RuntimeError, match="JAX_PREVUELO_CHARS_POR_TOKEN"):
        pc.chars_por_token()
```

- [ ] **Step 3: Verlo fallar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_prevuelo_config.py 2>&1 | tail -4`
Expected: `ERROR tests/test_prevuelo_config.py` con `ImportError: cannot import name 'prevuelo_config' from 'jacobs'` y `1 error`.

- [ ] **Step 4: Implementar**

`jacobs/prevuelo_config.py`:
```python
"""Configuración del pre-vuelo de Jacobs (spec 2026-09-17 §4.5 y §4.6).

Mismo patrón que jax/core/db_connect_config.py: se lee en CADA llamada (un
cambio de entorno vale en el próximo pre-vuelo, sin reiniciar) y un valor
inválido NO cae a un default silencioso -- lanza RuntimeError con el nombre de
la variable. Un typo en el entorno que desactivara la sonda o cambiara el
divisor de costo sería un fail-open con apariencia de configuración.

Defaults declarados, no medidos:
- JAX_PREVUELO_SONDA_TIMEOUT_S=20: la sonda pide 16 tokens; 20 s cubre la
  latencia de arranque de un proveedor sano sin dejar al usuario esperando el
  timeout de un paso entero (300 s). Se revisa con los números de la Task 15.
- JAX_PREVUELO_CHARS_POR_TOKEN=2: cota INFERIOR de caracteres por token.
  Menos caracteres por token = más tokens = más costo: sobreestima, nunca
  subestima (spec §4.6).
- JAX_PREVUELO_SONDA_MAX_TOKENS=16: la sonda mide disponibilidad, no calidad;
  se manda el menor entre esto y model.max_output_tokens.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import os

SONDA_TIMEOUT_S = "JAX_PREVUELO_SONDA_TIMEOUT_S"
CHARS_POR_TOKEN = "JAX_PREVUELO_CHARS_POR_TOKEN"
SONDA_MAX_TOKENS = "JAX_PREVUELO_SONDA_MAX_TOKENS"


def _entero_positivo(nombre: str, crudo: str) -> int:
    try:
        valor = int(crudo)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{nombre}={crudo!r} no es un entero: corregilo en el entorno "
            f"(/etc/jax/.env) -- no se asume ningún valor"
        ) from exc
    if valor <= 0:
        raise RuntimeError(f"{nombre}={valor} tiene que ser mayor que cero")
    return valor


def sonda_timeout_s() -> int:
    """Segundos que la sonda espera al proveedor."""
    return _entero_positivo(SONDA_TIMEOUT_S, os.getenv(SONDA_TIMEOUT_S, "20"))


def chars_por_token() -> int:
    """Divisor de caracteres a tokens para el costo máximo de entrada."""
    return _entero_positivo(CHARS_POR_TOKEN, os.getenv(CHARS_POR_TOKEN, "2"))


def sonda_max_tokens() -> int:
    """Techo de tokens de salida que pide la sonda."""
    return _entero_positivo(SONDA_MAX_TOKENS, os.getenv(SONDA_MAX_TOKENS, "16"))
```

- [ ] **Step 5: Verlo pasar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_prevuelo_config.py 2>&1 | tail -2`
Expected: `7 passed`.

- [ ] **Step 6: CI — listas y piso**

En `.github/workflows/policy.yml`, job `tests-puros`:
1. En la lista de `pytest -v`, reemplazar la línea `          tests/test_respaldo_de_uso_aislado.py` (la que cierra esa lista, antes de `      - name: Piso exacto de tests CORRIDOS`) por:
```
          tests/test_respaldo_de_uso_aislado.py
          tests/test_prevuelo_config.py
```
2. En la lista del piso, reemplazar `            tests/test_respaldo_de_uso_aislado.py \` por:
```
            tests/test_respaldo_de_uso_aislado.py \
            tests/test_prevuelo_config.py \
```
3. Reemplazar
```
          grep -qE "^742 passed, 1 skipped" /tmp/out || {
            echo "PISO ROTO: se esperaban 742 tests CORRIDOS y exactamente 1 skipped."; exit 1; }
```
por
```
          # 742 -> 749 el 2026-09-17 (pre-vuelo): test_prevuelo_config.py (+7) -- timeout,
          #   divisor y techo de la sonda por entorno; un valor inválido falla con el nombre.
          grep -qE "^749 passed, 1 skipped" /tmp/out || {
            echo "PISO ROTO: se esperaban 749 tests CORRIDOS y exactamente 1 skipped."; exit 1; }
```

Todas las tasks siguientes editan estas mismas tres zonas: agregan su archivo DESPUÉS del último agregado en cada lista y reemplazan la línea `grep`/`echo` vigente por la nueva, dejando el comentario de la task anterior.

- [ ] **Step 7: Commit**

```bash
cd /home/fruiz/worktrees/jax-prevuelo && pwd && git branch --show-current && \
git -C /home/fruiz/worktrees/jax-prevuelo add jacobs/prevuelo_config.py tests/test_prevuelo_config.py .github/workflows/policy.yml && \
git -C /home/fruiz/worktrees/jax-prevuelo commit -m "feat(jacobs): configuración validada del pre-vuelo

Timeout y techo de la sonda y divisor de caracteres por token, por entorno;
un valor inválido falla con el nombre de la variable.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Época de corrida en el store

**Files:**
- Modify: `jacobs/models.py:82-104` (`Pipeline`)
- Modify: `jacobs/store.py` (imports, `get_conn`, `init_tables`, `pipeline_create`, `_row_to_pipeline`, `step_upsert`, funciones nuevas)
- Test: `tests/test_run_epoch_db.py`
- Modify: `.github/workflows/policy.yml` (job `jacobs-gobernanza-db`)

**Interfaces:**
- Consumes: nada.
- Produces:
  - `Pipeline.run_epoch: int = 0`
  - `store.get_conn(found_rows: bool = False)` — con `found_rows=True` el `UPDATE` cuenta filas que CUMPLEN el `WHERE`, no las que cambiaron.
  - `store.pipeline_epoca_y_status(pipeline_id: str) -> tuple[int, PipelineStatus] | None`
  - `store.pipeline_update_status_si_epoca(pipeline_id: str, epoca: int, status: PipelineStatus, current_step_index: int | None = None, context: dict | None = None, *, desde: tuple[PipelineStatus, ...] = (PipelineStatus.running,)) -> bool`
  - `store.step_upsert_si_epoca(s: Step, epoca: int) -> bool` (el paso ya existe; escribe status, facet, motor, output_ref, timeout, tiempos y error sólo si el pipeline está `running` en esa época)
  - `store.pipeline_tomar_epoca(pipeline_id: str, epoca_leida: int, desde: tuple[PipelineStatus, ...], context: dict | None = None) -> int | None` (incrementa y devuelve la nueva; `None` si otro la tomó)
  - `store.step_upsert` ahora también actualiza `facet` en el `ON DUPLICATE KEY UPDATE`.
  - Constantes SQL `_SQL_EPOCA_Y_STATUS`, `_SQL_STEP_SI_EPOCA` y constructores `_sql_update_si_epoca(con_indice, con_contexto, n_desde)`, `_sql_tomar_epoca(con_contexto, n_desde)` (los EXPLAIN corren ESTAS).

- [ ] **Step 1: Escribir el test DB que falla**

`tests/test_run_epoch_db.py`:
```python
"""Época de corrida (spec 2026-09-17 §5.3) contra MariaDB real: la columna, las
escrituras condicionales y el incremento atómico. Job jacobs-gobernanza-db.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid

_db = os.environ.get("JAX_DB_NAME")
if _db and _db != "jax_memory_test":
    raise RuntimeError(
        f"JAX_DB_NAME={_db!r}: este archivo escribe filas y solo corre contra jax_memory_test."
    )
os.environ.setdefault("JAX_DB_NAME", "jax_memory_test")

from jacobs import store  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus  # noqa: E402


async def _crear(status=PipelineStatus.running, epoca=0):
    await store.init_tables()
    pid = str(uuid.uuid4())
    paso = Step(pipeline_id=pid, step_index=0, facet="jekyll", capability="research",
                input={"prompt": "p"})
    ahora = time.time()
    pipeline = Pipeline(pipeline_id=pid, name="t-epoca", invoked_by="plataforma",
                        mode="autonomous", status=status, plan=[paso],
                        context={"objective": "o"}, created_at=ahora, updated_at=ahora,
                        run_epoch=epoca)
    await store.pipeline_create(pipeline)
    await store.step_upsert(paso)
    return pipeline, paso


async def _borrar(pid):
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM jacobs_steps WHERE pipeline_id=%s", (pid,))
            await cur.execute("DELETE FROM jacobs_events WHERE pipeline_id=%s", (pid,))
            await cur.execute("DELETE FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))
    finally:
        conn.close()


async def _explain(sql, params):
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("EXPLAIN " + sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in await cur.fetchall()]
    finally:
        conn.close()


def test_init_tables_crea_run_epoch_con_default_cero():
    async def cuerpo():
        await store.init_tables()
        conn = await store.get_conn()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COLUMN_DEFAULT, IS_NULLABLE, DATA_TYPE FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='jacobs_pipelines' "
                    "AND COLUMN_NAME='run_epoch'")
                return await cur.fetchone()
        finally:
            conn.close()
    fila = asyncio.run(cuerpo())
    assert fila is not None, "jacobs_pipelines no tiene run_epoch"
    assert (str(fila[0]), fila[1], fila[2]) == ("0", "NO", "int")


def test_update_condicional_respeta_epoca_y_status():
    async def cuerpo():
        pipeline, _ = await _crear(status=PipelineStatus.running, epoca=2)
        pid = pipeline.pipeline_id
        try:
            otra_epoca = await store.pipeline_update_status_si_epoca(pid, 1, PipelineStatus.aborted)
            otro_status = await store.pipeline_update_status_si_epoca(
                pid, 2, PipelineStatus.completed, desde=(PipelineStatus.pending,))
            vigente = await store.pipeline_update_status_si_epoca(pid, 2, PipelineStatus.aborted)
            releido = await store.pipeline_get(pid)
            return otra_epoca, otro_status, vigente, releido.status, releido.run_epoch
        finally:
            await _borrar(pid)
    assert asyncio.run(cuerpo()) == (False, False, True, PipelineStatus.aborted, 2)


def test_step_upsert_si_epoca_solo_con_la_epoca_vigente_y_running():
    async def cuerpo():
        pipeline, paso = await _crear(status=PipelineStatus.running, epoca=1)
        pid = pipeline.pipeline_id
        try:
            paso.status = StepStatus.completed
            ajena = await store.step_upsert_si_epoca(paso, 0)
            propia = await store.step_upsert_si_epoca(paso, 1)
            tras_propia = (await store.steps_by_pipeline(pid))[0].status
            await store.pipeline_update_status_si_epoca(pid, 1, PipelineStatus.aborted)
            paso.status = StepStatus.failed
            paso.error = "tarde"
            no_running = await store.step_upsert_si_epoca(paso, 1)
            final = (await store.steps_by_pipeline(pid))[0]
            return ajena, propia, tras_propia, no_running, final.status, final.error
        finally:
            await _borrar(pid)
    assert asyncio.run(cuerpo()) == (
        False, True, StepStatus.completed, False, StepStatus.completed, None)


def test_tomar_epoca_una_sola_vez():
    async def cuerpo():
        pipeline, _ = await _crear(status=PipelineStatus.interrupted, epoca=5)
        pid = pipeline.pipeline_id
        try:
            resultados = await asyncio.gather(
                store.pipeline_tomar_epoca(pid, 5, (PipelineStatus.interrupted,)),
                store.pipeline_tomar_epoca(pid, 5, (PipelineStatus.interrupted,)),
            )
            return sorted(resultados, key=lambda r: r is None), await store.pipeline_epoca_y_status(pid)
        finally:
            await _borrar(pid)
    resultados, actual = asyncio.run(cuerpo())
    assert resultados == [6, None]
    assert actual == (6, PipelineStatus.interrupted)


def test_step_upsert_actualiza_la_faceta():
    async def cuerpo():
        pipeline, paso = await _crear()
        try:
            paso.facet = "thot"
            await store.step_upsert(paso)
            return (await store.steps_by_pipeline(pipeline.pipeline_id))[0].facet
        finally:
            await _borrar(pipeline.pipeline_id)
    assert asyncio.run(cuerpo()) == "thot"


def test_explain_de_las_consultas_de_epoca_usa_la_clave_primaria():
    async def cuerpo():
        pipeline, paso = await _crear(status=PipelineStatus.running, epoca=0)
        pid = pipeline.pipeline_id
        try:
            lectura = await _explain(store._SQL_EPOCA_Y_STATUS, (pid,))
            update = await _explain(store._sql_update_si_epoca(False, False, 1),
                                    ("aborted", time.time(), pid, 0, "running"))
            pasos = await _explain(store._SQL_STEP_SI_EPOCA, (
                "completed", "jekyll", None, None, 300, None, None, None, paso.step_id, pid, 0))
            tomar = await _explain(store._sql_tomar_epoca(False, 1), (time.time(), pid, 0, "running"))
            return lectura, update, pasos, tomar
        finally:
            await _borrar(pid)
    for filas in asyncio.run(cuerpo()):
        assert filas, "EXPLAIN vacío"
        assert all(f["key"] == "PRIMARY" for f in filas), filas
        assert all("filesort" not in (f.get("Extra") or "") and
                   "temporary" not in (f.get("Extra") or "") for f in filas), filas
```

- [ ] **Step 2: Verlo fallar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && set -a && source /etc/jax/.env && set +a && export JAX_DB_NAME=jax_memory_test PYTHONPATH=.:las_manos && /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_run_epoch_db.py 2>&1 | tail -8`
Expected: `6 failed`. `test_init_tables_crea_run_epoch_con_default_cero` con `AssertionError: jacobs_pipelines no tiene run_epoch`; `test_step_upsert_actualiza_la_faceta` con `AssertionError: assert 'jekyll' == 'thot'`; los otros cuatro con `AttributeError: module 'jacobs.store' has no attribute ...` o con el `run_epoch` que el modelo viejo ignora.

- [ ] **Step 3: Implementar — modelo**

En `jacobs/models.py`, dentro de `class Pipeline`, reemplazar
```python
    current_step_index: int = 0
```
por
```python
    current_step_index: int = 0
    # Época de corrida (spec 2026-09-17 §5.3): la toma cada ejecutor al
    # arrancar; resume, approve-step y continue la INCREMENTAN. Toda escritura
    # del ejecutor es condicional a su época y a status='running': una corrida
    # superada (cancelada, vencida, continuada por otro) no escribe nada.
    run_epoch:          int = 0
```

- [ ] **Step 4: Implementar — store**

En `jacobs/store.py`:

1. Después de `import aiomysql` agregar:
```python
from pymysql.constants import CLIENT
```

2. Reemplazar `get_conn` entera por:
```python
async def get_conn(found_rows: bool = False) -> aiomysql.Connection:
    # connect_timeout explícito (no en _db_cfg()): hallazgo de revisión,
    # Tarea 2b (tanda A, ronda de arreglo 1, 2026-09-14) -- sin esto,
    # aiomysql espera sin límite si la DB se cuelga.
    #
    # found_rows (2026-09-17, época de corrida): por defecto MariaDB devuelve
    # las filas CAMBIADAS de un UPDATE, no las que cumplen el WHERE. Un UPDATE
    # condicional que escribe los mismos valores devolvería 0 y el ejecutor
    # creería haber perdido la época. Con CLIENT.FOUND_ROWS el conteo es de
    # filas encontradas.
    extra = {"client_flag": CLIENT.FOUND_ROWS} if found_rows else {}
    return await aiomysql.connect(**_db_cfg(), connect_timeout=db_connect_timeout_seconds(), **extra)
```

3. En `init_tables`, en la lista de columnas de `jacobs_pipelines`, reemplazar
```python
                ("owner_ack_at", "ALTER TABLE jacobs_pipelines ADD COLUMN owner_ack_at DOUBLE NULL"),
            ]:
```
por
```python
                ("owner_ack_at", "ALTER TABLE jacobs_pipelines ADD COLUMN owner_ack_at DOUBLE NULL"),
                # 2026-09-17 (spec prevuelo-y-continuar §5.3): época de corrida.
                ("run_epoch", "ALTER TABLE jacobs_pipelines ADD COLUMN run_epoch INT NOT NULL DEFAULT 0"),
            ]:
```

4. En `pipeline_create`, reemplazar el `INSERT` y su tupla por:
```python
            await cur.execute(
                """
                INSERT INTO jacobs_pipelines
                    (pipeline_id, name, invoked_by, mode, status,
                     plan, current_step_index, max_steps, context_refs,
                     created_at, updated_at, user_id, tenant_id, run_epoch)
                VALUES (%s,%s,%s,%s,%s, %s,%s,%s,%s, %s,%s, %s,%s, %s)
                """,
                (
                    p.pipeline_id, p.name, p.invoked_by, p.mode, p.status.value,
                    json.dumps([s.model_dump() for s in p.plan], ensure_ascii=False),
                    p.current_step_index, p.max_steps,
                    json.dumps(p.context, ensure_ascii=False),
                    p.created_at, p.updated_at,
                    p.user_id, p.tenant_id, p.run_epoch,
                ),
            )
```

5. En `_row_to_pipeline`, reemplazar
```python
        owner_ack_at=row.get("owner_ack_at"),
```
por
```python
        owner_ack_at=row.get("owner_ack_at"),
        run_epoch=int(row.get("run_epoch") or 0),
```

6. En `step_upsert`, reemplazar
```python
                ON DUPLICATE KEY UPDATE
                    status=VALUES(status),
```
por
```python
                ON DUPLICATE KEY UPDATE
                    status=VALUES(status),
                    facet=VALUES(facet),
```

7. Agregar después de `pipeline_count_active`:
```python
# ----------------------------------------------------------------
#  Época de corrida (spec 2026-09-17 §5.3)
# ----------------------------------------------------------------
# Un solo ejecutor por pipeline. `cancel`, el kill switch y el reaper cambian
# el STATUS; `resume`, `approve-step` y `continue` INCREMENTAN la época. El
# ejecutor escribe sólo si el pipeline sigue en SU época y `running`: si no,
# perdió, registra RUN_SUPERSEDED una vez y termina sin escribir más.
# Todas van por clave primaria (EXPLAIN en tests/test_run_epoch_db.py).

_SQL_EPOCA_Y_STATUS = "SELECT run_epoch, status FROM jacobs_pipelines WHERE pipeline_id=%s"

_SQL_STEP_SI_EPOCA = (
    "UPDATE jacobs_steps s JOIN jacobs_pipelines p ON p.pipeline_id = s.pipeline_id "
    "SET s.status=%s, s.facet=%s, s.motor=%s, s.output_ref=%s, s.timeout_seconds=%s, "
    "    s.started_at=%s, s.finished_at=%s, s.error=%s "
    "WHERE s.step_id=%s AND p.pipeline_id=%s AND p.run_epoch=%s AND p.status='running'"
)


def _sql_update_si_epoca(con_indice: bool, con_contexto: bool, n_desde: int) -> str:
    sets = ["status=%s", "updated_at=%s"]
    if con_indice:
        sets.append("current_step_index=%s")
    if con_contexto:
        sets.append("context_refs=%s")
    desde = ",".join(["%s"] * n_desde)
    return (
        f"UPDATE jacobs_pipelines SET {', '.join(sets)} "
        f"WHERE pipeline_id=%s AND run_epoch=%s AND status IN ({desde})"
    )


def _sql_tomar_epoca(con_contexto: bool, n_desde: int) -> str:
    extra = ", context_refs=%s" if con_contexto else ""
    desde = ",".join(["%s"] * n_desde)
    return (
        f"UPDATE jacobs_pipelines SET run_epoch=run_epoch+1, updated_at=%s{extra} "
        f"WHERE pipeline_id=%s AND run_epoch=%s AND status IN ({desde})"
    )


async def _ejecutar_condicional(sql: str, params: tuple | list) -> int:
    conn = await get_conn(found_rows=True)
    try:
        async with conn.cursor() as cur:
            return await cur.execute(sql, params)
    finally:
        conn.close()


async def pipeline_epoca_y_status(pipeline_id: str) -> tuple[int, PipelineStatus] | None:
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(_SQL_EPOCA_Y_STATUS, (pipeline_id,))
            fila = await cur.fetchone()
    finally:
        conn.close()
    if not fila:
        return None
    return int(fila[0]), PipelineStatus(fila[1])


async def pipeline_update_status_si_epoca(
    pipeline_id: str,
    epoca: int,
    status: PipelineStatus,
    current_step_index: int | None = None,
    context: dict | None = None,
    *,
    desde: tuple[PipelineStatus, ...] = (PipelineStatus.running,),
) -> bool:
    """True si escribió: el pipeline estaba en `epoca` y en uno de `desde`."""
    params: list = [status.value, time.time()]
    if current_step_index is not None:
        params.append(current_step_index)
    if context is not None:
        params.append(json.dumps(context, ensure_ascii=False))
    params += [pipeline_id, epoca, *(d.value for d in desde)]
    sql = _sql_update_si_epoca(current_step_index is not None, context is not None, len(desde))
    return await _ejecutar_condicional(sql, params) == 1


async def step_upsert_si_epoca(s: Step, epoca: int) -> bool:
    """Escritura de un paso YA EXISTENTE desde el ejecutor. True si escribió."""
    params = (
        s.status.value, s.facet, s.motor, s.output_ref, s.timeout_seconds,
        s.started_at, s.finished_at, s.error,
        s.step_id, s.pipeline_id, epoca,
    )
    return await _ejecutar_condicional(_SQL_STEP_SI_EPOCA, params) == 1


async def pipeline_tomar_epoca(
    pipeline_id: str,
    epoca_leida: int,
    desde: tuple[PipelineStatus, ...],
    context: dict | None = None,
) -> int | None:
    """Incrementa la época si nadie la tomó desde que se leyó. Devuelve la
    nueva, o None si otro pedido ganó (doble resume, doble approve)."""
    params: list = [time.time()]
    if context is not None:
        params.append(json.dumps(context, ensure_ascii=False))
    params += [pipeline_id, epoca_leida, *(d.value for d in desde)]
    filas = await _ejecutar_condicional(_sql_tomar_epoca(context is not None, len(desde)), params)
    return epoca_leida + 1 if filas == 1 else None
```

- [ ] **Step 5: Verlo pasar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && set -a && source /etc/jax/.env && set +a && export JAX_DB_NAME=jax_memory_test PYTHONPATH=.:las_manos && /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_run_epoch_db.py jacobs/_step_motor_test.py 2>&1 | tail -3`
Expected: `10 passed` (6 nuevos + 4 existentes de `_step_motor_test.py`).

Si `test_explain_de_las_consultas_de_epoca_usa_la_clave_primaria` muestra una fila con `key` distinto de `PRIMARY`, NO se relaja el test: se pega el EXPLAIN en el ledger y se consulta al controlador.

- [ ] **Step 6: Regresión pura**

Run: la lista de Task 1 Step 1.
Expected: BASE_LOCAL + 7 passed (el modelo acepta `run_epoch`; nada más cambia).

- [ ] **Step 7: CI — job `jacobs-gobernanza-db`**

En `.github/workflows/policy.yml`, job `jacobs-gobernanza-db`:
1. Reemplazar (lista de `pytest -v`)
```
          jacobs/_direct_usage_test.py
      - name: Piso exacto de tests CORRIDOS
```
por
```
          jacobs/_direct_usage_test.py
          tests/test_run_epoch_db.py
      - name: Piso exacto de tests CORRIDOS
```
2. Reemplazar (lista del piso)
```
            jacobs/_direct_usage_test.py \
            2>&1 | tee /tmp/out
```
por
```
            jacobs/_direct_usage_test.py \
            tests/test_run_epoch_db.py \
            2>&1 | tee /tmp/out
```
3. Reemplazar
```
          grep -qE "^27 passed" /tmp/out || {
            echo "PISO ROTO: se esperaban 27 tests CORRIDOS."; exit 1; }
```
por
```
          # 27 -> 33 el 2026-09-17 (época de corrida): test_run_epoch_db.py (+6) -- columna,
          #   UPDATE condicional por época y status, pasos condicionales, incremento único
          #   y EXPLAIN por PRIMARY.
          grep -qE "^33 passed" /tmp/out || {
            echo "PISO ROTO: se esperaban 33 tests CORRIDOS."; exit 1; }
```

- [ ] **Step 8: Commit**

```bash
cd /home/fruiz/worktrees/jax-prevuelo && pwd && git branch --show-current && \
git -C /home/fruiz/worktrees/jax-prevuelo add jacobs/models.py jacobs/store.py tests/test_run_epoch_db.py .github/workflows/policy.yml && \
git -C /home/fruiz/worktrees/jax-prevuelo commit -m "feat(jacobs): época de corrida en el store

Columna run_epoch, escrituras condicionales a la época y a running (pipeline y
pasos), incremento atómico para resume/approve, step_upsert actualiza facet.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---
### Task 3: El ejecutor respeta su época

**Files:**
- Modify: `jacobs/executor.py:1019-1210` (`_run_one_step`, `run_pipeline`) y `:1263-1285` (`_fail_step`)
- Test: `tests/test_jacobs_epoca_ejecutor.py`
- Modify: `tests/test_gemini_key_en_cabecera.py:165-168` (el test de `_fail_step` parchea la escritura condicional)
- Modify: `.github/workflows/policy.yml` (job `tests-puros`)

**Interfaces:**
- Consumes (Task 2): `Pipeline.run_epoch`, `store.pipeline_epoca_y_status`, `store.pipeline_update_status_si_epoca`, `store.step_upsert_si_epoca`.
- Produces:
  - `run_pipeline(pipeline)` usa `pipeline.run_epoch` como SU época; nunca la modifica.
  - Evento `RUN_SUPERSEDED {epoca, epoca_actual, status_actual}`: una vez por corrida superada.
  - Evento `PIPELINE_ABORTED {at_wave, failed_steps, errores: {"<i>": error}}`: uno solo, escrito por la ola.
  - `_fail_step(pipeline, step, step_index, error)` ya no escribe `aborted` ni `PIPELINE_ABORTED`.
  - `_DESDE_ARRANQUE = (PipelineStatus.pending, PipelineStatus.running, PipelineStatus.interrupted)`: estados desde los que arranca una corrida.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_jacobs_epoca_ejecutor.py`:
```python
"""Un solo ejecutor por pipeline (spec 2026-09-17 §5.3), sin DB ni red.

La tienda falsa implementa la API VIEJA (pipeline_update_status, step_upsert)
y la NUEVA (condicionales por época): así los tests corren contra el ejecutor
de hoy y fallan por comportamiento, no por un AttributeError.

Hoy cancelar no detiene nada: la escritura de fin de ola vuelve a poner
`running` y la ola siguiente se despacha.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os

os.environ["JAX_DB_NAME"] = "jax_memory_test"

from unittest.mock import AsyncMock  # noqa: E402

from jacobs import executor  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus, Step  # noqa: E402


class TiendaFalsa:
    def __init__(self, status="pending", epoca=0):
        self.status = status
        self.epoca = epoca
        self.contexto: dict = {}
        self.pasos: dict[int, str] = {}
        self.eventos: list[tuple[str, dict]] = []

    async def pipeline_update_status(self, pipeline_id, status, current_step_index=None, context=None):
        self.status = status.value
        if context is not None:
            self.contexto = dict(context)

    async def pipeline_update_status_si_epoca(self, pipeline_id, epoca, status,
                                              current_step_index=None, context=None, *,
                                              desde=(PipelineStatus.running,)):
        if epoca != self.epoca or self.status not in {d.value for d in desde}:
            return False
        self.status = status.value
        if context is not None:
            self.contexto = dict(context)
        return True

    async def pipeline_epoca_y_status(self, pipeline_id):
        return self.epoca, PipelineStatus(self.status)

    async def step_upsert(self, s):
        self.pasos[s.step_index] = s.status.value

    async def step_upsert_si_epoca(self, s, epoca):
        if epoca != self.epoca or self.status != "running":
            return False
        self.pasos[s.step_index] = s.status.value
        return True

    async def event_append(self, pipeline_id, event_type, payload=None, step_id=None):
        self.eventos.append((event_type, payload or {}))

    def tipos(self):
        return [t for t, _ in self.eventos]


def _pipeline(n_pasos=2, epoca=0):
    pasos = [
        Step(pipeline_id="p1", step_index=i, facet="jekyll", capability="research",
             input={"prompt": f"paso {i}"}, depends_on=[i - 1] if i else [])
        for i in range(n_pasos)
    ]
    return Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma", mode="autonomous",
                    plan=pasos, context={"objective": "o"}, run_epoch=epoca)


def _correr(monkeypatch, tienda, pipeline, despachar):
    monkeypatch.setattr(executor, "store", tienda)
    monkeypatch.setattr(executor, "check_kill_switch", lambda: False)
    monkeypatch.setattr(executor, "save_if_large", lambda pid, sid, raw: (None, raw))
    monkeypatch.setattr(executor, "_persist_step_to_repo", AsyncMock())
    monkeypatch.setattr(executor, "_dispatch_step", despachar)
    asyncio.run(executor.run_pipeline(pipeline))


def test_cancelar_durante_la_ola_no_se_reescribe_running(monkeypatch):
    tienda, despachados = TiendaFalsa(), []

    async def despachar(step, pipeline):
        despachados.append(step.step_index)
        tienda.status = "aborted"  # POST /cancel mientras el paso corre
        return {"success": True, "result": "hecho"}

    _correr(monkeypatch, tienda, _pipeline(2), despachar)
    assert tienda.status == "aborted"
    assert despachados == [0]
    assert tienda.tipos().count("RUN_SUPERSEDED") == 1
    assert "PIPELINE_COMPLETED" not in tienda.tipos()


def test_continuado_por_otro_la_corrida_vieja_no_escribe_su_contexto(monkeypatch):
    tienda, despachados = TiendaFalsa(), []

    async def despachar(step, pipeline):
        despachados.append(step.step_index)
        tienda.epoca = 1  # POST /continue tomó el pipeline
        return {"success": True, "result": "tarde"}

    _correr(monkeypatch, tienda, _pipeline(2), despachar)
    assert "step_0_ref" not in tienda.contexto
    assert despachados == [0]
    assert tienda.tipos().count("RUN_SUPERSEDED") == 1


def test_un_solo_pipeline_aborted_con_los_errores(monkeypatch):
    tienda = TiendaFalsa()

    async def despachar(step, pipeline):
        raise RuntimeError("se cayó el proveedor")

    _correr(monkeypatch, tienda, _pipeline(1), despachar)
    abortados = [p for t, p in tienda.eventos if t == "PIPELINE_ABORTED"]
    assert len(abortados) == 1
    assert abortados[0] == {"at_wave": 0, "failed_steps": [0],
                            "errores": {"0": "se cayó el proveedor"}}
    assert tienda.status == "aborted"


def test_arranque_con_epoca_ajena_no_despacha(monkeypatch):
    tienda, despachados = TiendaFalsa(status="pending", epoca=1), []

    async def despachar(step, pipeline):
        despachados.append(step.step_index)
        return {"success": True, "result": "no debería"}

    _correr(monkeypatch, tienda, _pipeline(2, epoca=0), despachar)
    assert despachados == []
    assert tienda.status == "pending"
    assert tienda.tipos().count("RUN_SUPERSEDED") == 1


def test_relee_la_epoca_antes_de_cada_ola(monkeypatch):
    tienda, despachados = TiendaFalsa(), []

    def _cancelar_si_ya_persistio_la_ola_0():
        if "step_0_ref" in tienda.contexto and tienda.status == "running":
            tienda.status = "aborted"

    nuevo_original = tienda.pipeline_update_status_si_epoca
    viejo_original = tienda.pipeline_update_status

    async def nuevo(*a, **k):
        ok = await nuevo_original(*a, **k)
        if ok:
            _cancelar_si_ya_persistio_la_ola_0()
        return ok

    async def viejo(*a, **k):
        await viejo_original(*a, **k)
        _cancelar_si_ya_persistio_la_ola_0()

    tienda.pipeline_update_status_si_epoca = nuevo
    tienda.pipeline_update_status = viejo

    async def despachar(step, pipeline):
        despachados.append(step.step_index)
        return {"success": True, "result": "ok"}

    _correr(monkeypatch, tienda, _pipeline(2), despachar)
    assert despachados == [0]
    assert tienda.status == "aborted"
    assert tienda.tipos().count("RUN_SUPERSEDED") == 1


def test_corrida_normal_completa_y_persiste_el_contexto(monkeypatch):
    """CONTROL: pasa también con el ejecutor de hoy."""
    tienda = TiendaFalsa()

    async def despachar(step, pipeline):
        return {"success": True, "result": f"r{step.step_index}"}

    _correr(monkeypatch, tienda, _pipeline(2), despachar)
    assert tienda.status == "completed"
    assert {"step_0_ref", "step_1_ref"} <= set(tienda.contexto)
    assert "RUN_SUPERSEDED" not in tienda.tipos()
    assert tienda.tipos().count("PIPELINE_COMPLETED") == 1
```

- [ ] **Step 2: Verlos fallar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_jacobs_epoca_ejecutor.py 2>&1 | tail -9`
Expected: `5 failed, 1 passed`:
- `test_cancelar_durante_la_ola_no_se_reescribe_running`: `AssertionError: assert 'completed' == 'aborted'`
- `test_continuado_por_otro_la_corrida_vieja_no_escribe_su_contexto`: `AssertionError: assert 'step_0_ref' not in {...}`
- `test_un_solo_pipeline_aborted_con_los_errores`: `AssertionError: assert 2 == 1`
- `test_arranque_con_epoca_ajena_no_despacha`: `AssertionError: assert [0, 1] == []`
- `test_relee_la_epoca_antes_de_cada_ola`: `AssertionError: assert [0, 1] == [0]`
- `test_corrida_normal_completa_y_persiste_el_contexto`: PASS (control).

- [ ] **Step 3: Implementar — `_run_one_step`**

En `jacobs/executor.py` reemplazar `async def _run_one_step(...)` completa (desde `async def _run_one_step` hasta su `return False` final) por:
```python
async def _run_one_step(step: Step, i: int, pipeline: Pipeline) -> bool:
    """Ejecuta un step individual. Devuelve True si completó, False si falló o
    si la corrida perdió su época.

    Época (spec 2026-09-17 §5.3): cada escritura del paso es condicional a
    `pipeline.run_epoch` y a status='running'. Si la primera falla, el paso NO
    se despacha (no se gasta en una corrida superada). Si falla la de
    `completed`, el resultado tardío se descarta y su ref no entra al contexto.
    Lo despachado no se interrumpe: es el mismo límite que el kill switch
    entre olas. run_pipeline descubre la pérdida en su escritura de fin de
    ola y registra RUN_SUPERSEDED una sola vez.
    """
    epoca = pipeline.run_epoch
    step.status     = StepStatus.running
    step.started_at = time.time()
    if not await store.step_upsert_si_epoca(step, epoca):
        return False
    await store.event_append(
        pipeline.pipeline_id, "STEP_STARTED",
        {"step_index": i, "facet": step.facet, "capability": step.capability},
        step.step_id,
    )

    try:
        raw_output = await asyncio.wait_for(
            _dispatch_step(step, pipeline),
            timeout=step.timeout_seconds,
        )

        ref, inline = save_if_large(pipeline.pipeline_id, step.step_id, raw_output)
        if ref:
            step.output_ref = ref
        else:
            step.output_ref = f"inline:{json.dumps(inline, ensure_ascii=False)}"

        step.status      = StepStatus.completed
        step.finished_at = time.time()
        if not await store.step_upsert_si_epoca(step, epoca):
            return False
        pipeline.context[f"step_{i}_ref"] = step.output_ref
        await store.event_append(
            pipeline.pipeline_id, "STEP_COMPLETED",
            {"step_index": i, "output_ref": step.output_ref},
            step.step_id,
        )
        try:
            await _persist_step_to_repo(
                pipeline_id=pipeline.pipeline_id,
                pipeline_name=pipeline.name,
                step_index=i,
                facet=step.facet,
                capability=step.capability,
                raw_output=raw_output,
            )
        except Exception as _persist_err:  # noqa: BLE001  # fail-soft: es la copia .md de cortesía en ~/jax/repo/documents -- el output canónico ya quedó en output_ref y en store.step_upsert_si_epoca antes de este try, nadie lee ese .md
            logger.warning("No se pudo persistir step %d al repo: %s", i, _persist_err)
        return True

    except asyncio.TimeoutError:
        await _fail_step(pipeline, step, i, f"Timeout ({step.timeout_seconds}s)")
        return False
    except Exception as exc:  # noqa: BLE001  # fail-soft: no traga nada -- convierte cualquier error del step en fallo EXPLÍCITO vía _fail_step (status=failed + STEP_FAILED + error) y devuelve False, que es lo que la ola usa para cortar el pipeline
        await _fail_step(pipeline, step, i, str(exc))
        return False
```

- [ ] **Step 4: Implementar — `run_pipeline`**

Reemplazar `async def run_pipeline(...)` completa por:
```python
# Estados desde los que arranca una corrida: pending (recién creado), running
# (continue ya lo dejó así con la época nueva), interrupted (resume y
# approve-step no cambian el status, sólo la época).
_DESDE_ARRANQUE = (PipelineStatus.pending, PipelineStatus.running, PipelineStatus.interrupted)


async def _perdio_la_epoca(pipeline: Pipeline) -> None:
    """La corrida perdió su época o el pipeline dejó de estar `running`
    (cancelado, vencido por el reaper, continuado o reanudado por otro
    pedido). Se registra UNA vez y quien llama termina sin escribir nada más."""
    actual = await store.pipeline_epoca_y_status(pipeline.pipeline_id)
    epoca_actual = actual[0] if actual else None
    status_actual = actual[1].value if actual else None
    logger.warning(
        "Jacobs %s: corrida de la época %s superada (época actual %s, status %s) -- termina sin escribir",
        pipeline.pipeline_id, pipeline.run_epoch, epoca_actual, status_actual,
    )
    await store.event_append(pipeline.pipeline_id, "RUN_SUPERSEDED", {
        "epoca": pipeline.run_epoch,
        "epoca_actual": epoca_actual,
        "status_actual": status_actual,
    })


async def run_pipeline(pipeline: Pipeline) -> None:
    """
    Ejecuta el pipeline por OLAS topológicas. Dentro de cada ola, los steps
    corren EN PARALELO (asyncio.gather). El orden entre olas respeta depends_on.

    Modos:
      dry_run    — no ejecuta nada, completa inmediatamente.
      supervised — ejecuta UNA ola y pausa (status=interrupted); espera /resume.
                   La granularidad de aprobación es la OLA, no el step.
    Hyde: si un step de la ola es hyde sin aprobar, la ola NO se ejecuta y el
    pipeline se interrumpe hasta /approve-step.

    Época (spec 2026-09-17 §5.3): `pipeline.run_epoch` es la época de ESTA
    corrida. Toda escritura de estado es condicional a ella y a
    status='running'; antes de cada ola se relee (una consulta por PK). Si no
    coincide, RUN_SUPERSEDED una vez y termina.
    """
    pipeline_id = pipeline.pipeline_id
    epoca = pipeline.run_epoch

    if pipeline.mode == "dry_run":
        if not await store.pipeline_update_status_si_epoca(
            pipeline_id, epoca, PipelineStatus.completed, desde=_DESDE_ARRANQUE,
        ):
            await _perdio_la_epoca(pipeline)
            return
        await store.event_append(pipeline_id, "DRY_RUN_COMPLETE", {"steps": len(pipeline.plan)})
        return

    if not await store.pipeline_update_status_si_epoca(
        pipeline_id, epoca, PipelineStatus.running,
        pipeline.current_step_index, pipeline.context, desde=_DESDE_ARRANQUE,
    ):
        await _perdio_la_epoca(pipeline)
        return
    await store.event_append(pipeline_id, "PIPELINE_STARTED", {"run_epoch": epoca})

    # Estado derivado del DAG, no de un cursor lineal: un step está "hecho" si
    # tiene su ref en context (sobrevive a /resume y a /continue).
    done = {
        i for i in range(len(pipeline.plan))
        if pipeline.context.get(f"step_{i}_ref")
    }

    waves = _compute_waves(pipeline.plan, done)
    logger.info(
        "Jacobs director: %d olas, tamaños=%s (ya completos: %s, época %s)",
        len(waves), [len(w) for w in waves], sorted(done), epoca,
    )

    for wave_num, wave in enumerate(waves):
        # ---- ¿Sigue siendo mi corrida? ----
        if await store.pipeline_epoca_y_status(pipeline_id) != (epoca, PipelineStatus.running):
            await _perdio_la_epoca(pipeline)
            return

        # ---- Kill switch: antes de cada ola ----
        if check_kill_switch():
            for i in wave:
                step = pipeline.plan[i]
                step.status = StepStatus.failed
                step.error  = "Kill switch activo"
                if not await store.step_upsert_si_epoca(step, epoca):
                    await _perdio_la_epoca(pipeline)
                    return
            if not await store.pipeline_update_status_si_epoca(pipeline_id, epoca, PipelineStatus.aborted):
                await _perdio_la_epoca(pipeline)
                return
            await store.event_append(
                pipeline_id, "KILL_SWITCH_ABORTED", {"wave": wave_num, "steps": wave}
            )
            return

        # ---- Hyde gate: si algún step de la ola es hyde sin aprobar, interrumpir ----
        hyde_pending = [
            i for i in wave
            if pipeline.plan[i].facet == "hyde"
            and not pipeline.context.get(f"hyde_approved_{pipeline.plan[i].step_id}")
        ]
        if hyde_pending:
            for i in hyde_pending:
                step = pipeline.plan[i]
                step.status = StepStatus.blocked_human_gate
                if not await store.step_upsert_si_epoca(step, epoca):
                    await _perdio_la_epoca(pipeline)
                    return
                await store.event_append(
                    pipeline_id, "STEP_BLOCKED_HUMAN_GATE",
                    {"step_index": i, "facet": "hyde"}, step.step_id,
                )
            if not await store.pipeline_update_status_si_epoca(
                pipeline_id, epoca, PipelineStatus.interrupted, wave[0], pipeline.context,
            ):
                await _perdio_la_epoca(pipeline)
                return
            await store.event_append(
                pipeline_id, "PIPELINE_INTERRUPTED",
                {"at_wave": wave_num, "hyde_steps": hyde_pending,
                 "reason": "hyde — requiere /approve-step"},
            )
            return

        # ---- EJECUTAR LA OLA EN PARALELO ----
        await store.event_append(
            pipeline_id, "WAVE_STARTED",
            {"wave": wave_num, "steps": wave, "parallel": len(wave)},
        )
        results = await asyncio.gather(*[
            _run_one_step(pipeline.plan[i], i, pipeline)
            for i in wave
        ])

        # Persistir avance del context tras la ola completa, SOLO si sigue
        # siendo mi corrida: esta es la escritura que antes resucitaba un
        # pipeline cancelado a `running`.
        next_idx = max(wave) + 1
        if not await store.pipeline_update_status_si_epoca(
            pipeline_id, epoca, PipelineStatus.running, next_idx, pipeline.context,
        ):
            await _perdio_la_epoca(pipeline)
            return

        # ---- ¿Algún step falló sin skip_on_fail? → abortar (UN evento) ----
        failed = [
            i for i, ok in zip(wave, results)
            if not ok and not pipeline.plan[i].skip_on_fail
        ]
        if failed:
            if not await store.pipeline_update_status_si_epoca(pipeline_id, epoca, PipelineStatus.aborted):
                await _perdio_la_epoca(pipeline)
                return
            await store.event_append(
                pipeline_id, "PIPELINE_ABORTED",
                {"at_wave": wave_num, "failed_steps": failed,
                 "errores": {str(i): pipeline.plan[i].error for i in failed}},
            )
            return

        await store.event_append(
            pipeline_id, "WAVE_COMPLETED", {"wave": wave_num, "steps": wave}
        )

        # ---- Supervised: pausar después de cada ola ----
        if pipeline.mode == "supervised":
            if not await store.pipeline_update_status_si_epoca(
                pipeline_id, epoca, PipelineStatus.interrupted, next_idx, pipeline.context,
            ):
                await _perdio_la_epoca(pipeline)
                return
            await store.event_append(
                pipeline_id, "PIPELINE_INTERRUPTED",
                {"after_wave": wave_num, "next_index": next_idx,
                 "reason": "supervised — awaiting /resume"},
            )
            return

    # ---- Todas las olas terminaron ----
    if not await store.pipeline_update_status_si_epoca(
        pipeline_id, epoca, PipelineStatus.completed, len(pipeline.plan), pipeline.context,
    ):
        await _perdio_la_epoca(pipeline)
        return
    await store.event_append(pipeline_id, "PIPELINE_COMPLETED")
```

- [ ] **Step 5: Implementar — `_fail_step`**

Reemplazar `async def _fail_step(...)` completa por:
```python
async def _fail_step(
    pipeline: Pipeline, step: Step, step_index: int, error: str
) -> None:
    # Ruling T6-6: aca se ESCRIBE el error de un paso (jacobs_steps.error y el
    # evento STEP_FAILED, que jax-platform muestra). Se redacta en el punto de
    # escritura para que ningun llamador -- ni el str(exc) generico de
    # _run_one_step -- pueda guardar un secreto en claro.
    #
    # 2026-09-17 (spec §5.3): ya NO escribe `aborted` ni PIPELINE_ABORTED. Lo
    # hace run_pipeline al cerrar la ola, una sola vez y con los errores de
    # todos los pasos caídos: antes salían DOS PIPELINE_ABORTED con payloads
    # distintos. Y la escritura del paso es condicional a la época.
    error = redactar_secretos(error)
    step.status      = StepStatus.failed
    step.error       = error
    step.finished_at = time.time()
    if not await store.step_upsert_si_epoca(step, pipeline.run_epoch):
        return
    await store.event_append(
        pipeline.pipeline_id, "STEP_FAILED",
        {"step_index": step_index, "error": error},
        step.step_id,
    )
```

- [ ] **Step 6: Ajustar el test de redacción de `_fail_step`**

En `tests/test_gemini_key_en_cabecera.py`, dentro de `JacobsFailStepRedactaTest.test_el_error_guardado_y_los_eventos_van_redactados`, reemplazar
```python
        with patch.object(executor.store, "step_upsert", AsyncMock()), \
             patch.object(executor.store, "event_append", AsyncMock()) as ev, \
             patch.object(executor.store, "pipeline_update_status", AsyncMock()):
```
por
```python
        with patch.object(executor.store, "step_upsert_si_epoca", AsyncMock(return_value=True)), \
             patch.object(executor.store, "event_append", AsyncMock()) as ev:
```
y en su docstring reemplazar `y los eventos STEP_FAILED / PIPELINE_ABORTED)` por `y el evento STEP_FAILED)`.

- [ ] **Step 7: Verlos pasar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_jacobs_epoca_ejecutor.py tests/test_gemini_key_en_cabecera.py tests/test_jacobs_ref_ilegible_no_es_exito.py tests/test_hipatia_fuentes.py 2>&1 | tail -3`
Expected: todo passed, `0 failed`.

Luego la lista completa de Task 1 Step 1 más `tests/test_prevuelo_config.py tests/test_jacobs_epoca_ejecutor.py`.
Expected: BASE_LOCAL + 13 passed.

- [ ] **Step 8: CI y commit**

`tests-puros`: agregar `tests/test_jacobs_epoca_ejecutor.py` a las dos listas (después de `tests/test_prevuelo_config.py`) y reemplazar la línea `grep`/`echo` de 749 por:
```
          # 749 -> 755 el 2026-09-17 (época de corrida): test_jacobs_epoca_ejecutor.py (+6) --
          #   cancelar durante una ola ya no reescribe running, la corrida superada no escribe
          #   contexto ni despacha, relee la época por ola, un solo PIPELINE_ABORTED. 5 vistos
          #   en rojo; la corrida normal es control.
          grep -qE "^755 passed, 1 skipped" /tmp/out || {
            echo "PISO ROTO: se esperaban 755 tests CORRIDOS y exactamente 1 skipped."; exit 1; }
```

```bash
cd /home/fruiz/worktrees/jax-prevuelo && pwd && git branch --show-current && \
git -C /home/fruiz/worktrees/jax-prevuelo add jacobs/executor.py tests/test_jacobs_epoca_ejecutor.py tests/test_gemini_key_en_cabecera.py .github/workflows/policy.yml && \
git -C /home/fruiz/worktrees/jax-prevuelo commit -m "fix(jacobs): un solo ejecutor por pipeline

Cancelar, el reaper o un continue ya detienen la corrida vieja: toda escritura
del ejecutor es condicional a su época y a running, relee la época antes de
cada ola y registra RUN_SUPERSEDED una vez. Un solo PIPELINE_ABORTED por ola.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `resume` y `approve-step` toman la época

**Files:**
- Modify: `jacobs/routes.py:251-375` (`resume_pipeline`, `approve_step`)
- Test: `tests/test_jacobs_resume_epoca.py`
- Modify: `.github/workflows/policy.yml` (job `tests-puros`)

**Interfaces:**
- Consumes (Task 2): `store.pipeline_tomar_epoca(pipeline_id, epoca_leida, desde, context=None) -> int | None`.
- Produces: `POST /jacobs/pipeline/{id}/resume` y `/approve-step` responden 409 si otro pedido ya tomó la época, y agregan `run_epoch` a su respuesta; lanzan `run_pipeline` con `pipeline.run_epoch = nueva`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_jacobs_resume_epoca.py`:
```python
"""resume y approve-step incrementan la época (spec 2026-09-17 §5.3; approve-step
por el desvío 9 del plan): un pedido doble no lanza dos ejecutores. Sin DB.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import pytest  # noqa: E402
from fastapi import BackgroundTasks, HTTPException  # noqa: E402

from jacobs import routes  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus  # noqa: E402


def _interrumpido(epoca=3):
    return Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma", mode="autonomous",
                    status=PipelineStatus.interrupted, context={"objective": "o"}, run_epoch=epoca)


def _paso_hyde():
    return Step(step_id="s-hyde", pipeline_id="p1", step_index=0, facet="hyde",
                capability="execute", status=StepStatus.blocked_human_gate)


def test_resume_toma_la_epoca_y_lanza_con_ella():
    tomar, bg = AsyncMock(return_value=4), BackgroundTasks()
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=_interrumpido())), \
         patch.object(routes.store, "steps_by_pipeline", AsyncMock(return_value=[])), \
         patch.object(routes.store, "pipeline_tomar_epoca", tomar, create=True), \
         patch.object(routes.store, "event_append", AsyncMock()), \
         patch.object(routes, "check_kill_switch", return_value=False):
        r = asyncio.run(routes.resume_pipeline(
            "p1", routes.ResumeRequest(invoked_by="plataforma"), bg))
    tomar.assert_awaited_once_with("p1", 3, (PipelineStatus.interrupted,))
    assert r["run_epoch"] == 4
    assert bg.tasks[0].args[0].run_epoch == 4


def test_resume_doble_el_segundo_recibe_409_y_no_lanza():
    bg, upsert = BackgroundTasks(), AsyncMock()
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=_interrumpido())), \
         patch.object(routes.store, "steps_by_pipeline", AsyncMock(return_value=[])), \
         patch.object(routes.store, "pipeline_tomar_epoca", AsyncMock(return_value=None), create=True), \
         patch.object(routes.store, "step_upsert", upsert), \
         patch.object(routes.store, "event_append", AsyncMock()), \
         patch.object(routes, "check_kill_switch", return_value=False), \
         pytest.raises(HTTPException) as e:
        asyncio.run(routes.resume_pipeline("p1", routes.ResumeRequest(invoked_by="plataforma"), bg))
    assert e.value.status_code == 409
    assert bg.tasks == []
    upsert.assert_not_awaited()


def test_approve_step_persiste_las_marcas_de_hyde_al_tomar_la_epoca():
    tomar, bg = AsyncMock(return_value=8), BackgroundTasks()
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=_interrumpido(epoca=7))), \
         patch.object(routes.store, "steps_by_pipeline", AsyncMock(return_value=[_paso_hyde()])), \
         patch.object(routes.store, "pipeline_tomar_epoca", tomar, create=True), \
         patch.object(routes.store, "pipeline_update_status", AsyncMock()), \
         patch.object(routes.store, "step_upsert", AsyncMock()), \
         patch.object(routes.store, "event_append", AsyncMock()), \
         patch.object(routes, "check_kill_switch", return_value=False):
        r = asyncio.run(routes.approve_step(
            "p1", routes.ApproveStepRequest(invoked_by="plataforma"), bg))
    tomar.assert_awaited_once()
    args = tomar.await_args.args
    assert args[:3] == ("p1", 7, (PipelineStatus.interrupted,))
    assert args[3]["hyde_approved_s-hyde"] is True
    assert r["run_epoch"] == 8
    assert bg.tasks[0].args[0].run_epoch == 8


def test_approve_step_doble_409_sin_tocar_pasos():
    bg, upsert = BackgroundTasks(), AsyncMock()
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=_interrumpido())), \
         patch.object(routes.store, "steps_by_pipeline", AsyncMock(return_value=[_paso_hyde()])), \
         patch.object(routes.store, "pipeline_tomar_epoca", AsyncMock(return_value=None), create=True), \
         patch.object(routes.store, "pipeline_update_status", AsyncMock()), \
         patch.object(routes.store, "step_upsert", upsert), \
         patch.object(routes.store, "event_append", AsyncMock()), \
         patch.object(routes, "check_kill_switch", return_value=False), \
         pytest.raises(HTTPException) as e:
        asyncio.run(routes.approve_step("p1", routes.ApproveStepRequest(invoked_by="plataforma"), bg))
    assert e.value.status_code == 409
    assert bg.tasks == []
    upsert.assert_not_awaited()
```

- [ ] **Step 2: Verlos fallar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_jacobs_resume_epoca.py 2>&1 | tail -6`
Expected: `4 failed`: los dos "toma la época" con `AssertionError: Expected mock to have been awaited once. Awaited 0 times.`; los dos "doble" con `Failed: DID NOT RAISE <class 'fastapi.exceptions.HTTPException'>`.

- [ ] **Step 3: Implementar — resume**

En `jacobs/routes.py`, dentro de `resume_pipeline`, reemplazar desde `    await store.event_append(pipeline_id, "PIPELINE_RESUMED", {"by": req.invoked_by})` hasta el `return {...}` inclusive por:
```python
    # Época (spec 2026-09-17 §5.3): se toma ANTES de tocar pasos. Si otro
    # resume ganó, este no escribe ni lanza nada: dos ejecutores del mismo
    # pipeline es exactamente lo que la época existe para impedir.
    nueva_epoca = await store.pipeline_tomar_epoca(
        pipeline_id, pipeline.run_epoch, (PipelineStatus.interrupted,),
    )
    if nueva_epoca is None:
        raise HTTPException(
            status_code=409,
            detail="Otro pedido ya reanudó o cambió este pipeline mientras se procesaba este",
        )
    await store.event_append(
        pipeline_id, "PIPELINE_RESUMED", {"by": req.invoked_by, "run_epoch": nueva_epoca},
    )

    # Desbloquear TODOS los steps en estado blocked (una ola supervised pudo
    # dejar varios). El executor recalcula las olas desde los refs en context,
    # así que basta con poner los blocked en pending para que entren a su ola.
    steps = await store.steps_by_pipeline(pipeline_id)
    for s in steps:
        if s.status == StepStatus.blocked:
            s.status = StepStatus.pending
            await store.step_upsert(s)

    pipeline.plan = steps
    pipeline.run_epoch = nueva_epoca
    background.add_task(run_pipeline, pipeline)

    return {
        "pipeline_id": pipeline_id,
        "status": "resuming",
        "from_index": pipeline.current_step_index,
        "run_epoch": nueva_epoca,
    }
```

- [ ] **Step 4: Implementar — approve-step**

En `approve_step`, reemplazar desde `    approved_indices = []` hasta el `return {...}` inclusive por:
```python
    for current_step in gated:
        if current_step.facet == "hyde":
            pipeline.context[f"hyde_approved_{current_step.step_id}"] = True

    # Época (desvío 9 del plan 2026-09-17): approve-step lanza run_pipeline
    # igual que resume. Las marcas hyde_approved_* viajan en el MISMO UPDATE
    # que toma la época (antes iban en un pipeline_update_status aparte).
    nueva_epoca = await store.pipeline_tomar_epoca(
        pipeline_id, pipeline.run_epoch, (PipelineStatus.interrupted,), pipeline.context,
    )
    if nueva_epoca is None:
        raise HTTPException(
            status_code=409,
            detail="Otro pedido ya aprobó o cambió este pipeline mientras se procesaba este",
        )

    approved_indices = []
    for current_step in gated:
        await store.event_append(
            pipeline_id, "STEP_APPROVED",
            {"step_index": current_step.step_index, "facet": current_step.facet,
             "by": req.invoked_by, "run_epoch": nueva_epoca},
            current_step.step_id,
        )
        current_step.status = StepStatus.pending
        current_step.error  = None
        await store.step_upsert(current_step)
        approved_indices.append(current_step.step_index)

    pipeline.plan = steps
    pipeline.run_epoch = nueva_epoca
    background.add_task(run_pipeline, pipeline)

    return {
        "pipeline_id":     pipeline_id,
        "status":          "resuming",
        "approved_steps":  approved_indices,
        "run_epoch":       nueva_epoca,
    }
```

- [ ] **Step 5: Verlos pasar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_jacobs_resume_epoca.py tests/test_jacobs_invoked_by_rol.py 2>&1 | tail -3`
Expected: `12 passed` (4 nuevos + 8 de `test_jacobs_invoked_by_rol.py`).

- [ ] **Step 6: CI y commit**

`tests-puros`: agregar `tests/test_jacobs_resume_epoca.py` a las dos listas; reemplazar la línea de 755 por:
```
          # 755 -> 759 el 2026-09-17: test_jacobs_resume_epoca.py (+4) -- resume y approve-step
          #   toman la época; el pedido doble recibe 409 sin tocar pasos. 4 vistos en rojo.
          grep -qE "^759 passed, 1 skipped" /tmp/out || {
            echo "PISO ROTO: se esperaban 759 tests CORRIDOS y exactamente 1 skipped."; exit 1; }
```

```bash
cd /home/fruiz/worktrees/jax-prevuelo && pwd && git branch --show-current && \
git -C /home/fruiz/worktrees/jax-prevuelo add jacobs/routes.py tests/test_jacobs_resume_epoca.py .github/workflows/policy.yml && \
git -C /home/fruiz/worktrees/jax-prevuelo commit -m "fix(jacobs): resume y approve-step toman la época

Un resume o approve doble ya no lanza dos ejecutores: el segundo recibe 409.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Reglas puras del pre-vuelo

**Files:**
- Create: `jacobs/prevuelo_reglas.py`
- Modify: `las_manos/motor_registry/output_validator.py` (función pública `puede_pedir_reintento`)
- Test: `tests/test_prevuelo_reglas.py`
- Modify: `.github/workflows/policy.yml` (job `tests-puros`)

**Interfaces:**
- Consumes: `contrato_dispatch.errores_del_contrato`, `contrato_dispatch._max_output_tokens_value`, `contrato_dispatch.ModelDispatchConfigError`.
- Produces (todo en `jacobs/prevuelo_reglas.py`):
  - `REGLAS: frozenset[str]` = `{"tope_insuficiente", "sin_contrato_de_salida", "credencial_ausente", "faceta_caida", "faceta_inexistente"}`
  - `TRANSPORTES_QUE_COBRAN = frozenset({"http_openai_compat", "http_gemini"})`
  - `@dataclass(frozen=True) Violacion(paso: int, faceta: str, regla: str, detalle: str)` + `to_dict()`
  - `@dataclass(frozen=True) CostoPaso(paso: int, faceta: str, modelo: str | None, llamadas_max: int, tokens_in_max: int, tokens_out_max: int, usd_max: Decimal | None, motivo: str)` + `to_dict()` (usd como `str` o `None`). Motivos: `"acotado"`, `"sin_precio"`, `"local"`, `"suscripcion"`, `"mecanico"`, `"sin_contrato_de_salida"`, `"faceta_inexistente"`.
  - `@dataclass(frozen=True) Veredicto(ok: bool, violaciones: tuple[Violacion, ...], costo_max_usd: Decimal, pasos_costo: tuple[CostoPaso, ...], sondeadas: tuple[str, ...])` + propiedad `hay_no_acotados: bool` + `to_dict() -> {"ok", "violaciones", "costo_max_usd": str, "pasos_costo", "sondeadas", "hay_no_acotados"}`
  - `@dataclass(frozen=True) Despacho(clave_salud: str, via_motor: bool, transporte: str, provider_id: str, base_url: str | None, modelo: str, max_tokens_param: str | None, max_output_tokens: int | None, motor_max_tokens: int, precio_in: Decimal | None, precio_out: Decimal | None, tiene_herramientas: bool, schema_con_reintento: bool, persona: str | None)`
  - `errores_de_contrato(d: Despacho) -> list[str]`
  - `tope_efectivo(d: Despacho) -> tuple[int, str]` (tope, qué subir: `"motor.max_tokens"` o `"model.max_output_tokens"`)
  - `llamadas_max(d: Despacho, max_iteraciones: int) -> int`
  - `tokens_de_entrada(chars: int, chars_por_token: int) -> int`
  - `costo_usd(llamadas: int, tokens_in: int, tokens_out: int, precio_in: Decimal, precio_out: Decimal) -> Decimal` (6 decimales, hacia arriba)
  - `costo_sin_evaluar(step: Step) -> CostoPaso | None` (`assemble` → mecánico; `hyde` → suscripción)
  - `evaluar_paso(paso: int, faceta: str, d: Despacho | None, *, min_output_tokens: int, credencial_activa: bool, chars_entrada: int, chars_por_token: int, max_iteraciones: int, detalle_inexistente: str = "") -> tuple[list[Violacion], CostoPaso]`
  - `armar_veredicto(violaciones: list[Violacion], costos: list[CostoPaso], sondeadas: list[str]) -> Veredicto`
  - `output_validator.puede_pedir_reintento(schema_name: str | None) -> bool`

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_prevuelo_reglas.py`:
```python
"""Reglas puras del pre-vuelo (spec 2026-09-17 §4.1, §4.3, §4.4, §4.6; desvíos
2, 5 y 13 del plan). Sin DB ni red.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import json
import os
from decimal import Decimal

os.environ["JAX_DB_NAME"] = "jax_memory_test"

from jacobs import prevuelo_reglas as pr  # noqa: E402
from jacobs.models import Step  # noqa: E402
from motor_registry.output_validator import puede_pedir_reintento  # noqa: E402


def _despacho(**cambios):
    base = dict(
        clave_salud="jekyll", via_motor=False, transporte="http_openai_compat",
        provider_id="deepseek", base_url="https://api.example/v1", modelo="deepseek-v4-flash",
        max_tokens_param="max_tokens", max_output_tokens=8192, motor_max_tokens=0,
        precio_in=Decimal("0.27"), precio_out=Decimal("1.10"),
        tiene_herramientas=False, schema_con_reintento=False, persona=None,
    )
    base.update(cambios)
    return pr.Despacho(**base)


def _evaluar(d, **k):
    args = dict(min_output_tokens=0, credencial_activa=True, chars_entrada=2000,
                chars_por_token=2, max_iteraciones=5)
    args.update(k)
    return pr.evaluar_paso(3, "jekyll", d, **args)


def test_faceta_inexistente_bloquea():
    v, c = pr.evaluar_paso(0, "zz", None, min_output_tokens=0, credencial_activa=False,
                           chars_entrada=0, chars_por_token=2, max_iteraciones=5)
    assert [x.regla for x in v] == ["faceta_inexistente"]
    assert c.usd_max is None and c.motivo == "faceta_inexistente"


def test_contrato_completo_y_credencial_no_bloquea():
    v, _ = _evaluar(_despacho())
    assert v == []


def test_openai_compat_sin_max_tokens_param_bloquea():
    v, c = _evaluar(_despacho(max_tokens_param=None))
    assert [x.regla for x in v] == ["sin_contrato_de_salida"]
    assert "UPDATE model SET max_tokens_param" in v[0].detalle
    assert c.usd_max is None and c.motivo == "sin_contrato_de_salida"


def test_gemini_sin_max_output_tokens_bloquea():
    v, _ = _evaluar(_despacho(transporte="http_gemini", max_tokens_param=None, max_output_tokens=None))
    assert [x.regla for x in v] == ["sin_contrato_de_salida"]
    assert "max_output_tokens" in v[0].detalle


def test_tope_igual_al_minimo_pasa():
    v, _ = _evaluar(_despacho(max_output_tokens=8192), min_output_tokens=8192)
    assert v == []


def test_motor_por_debajo_del_minimo_dice_subir_motor_max_tokens():
    d = _despacho(clave_salud="kimi", via_motor=True, provider_id="moonshot", modelo="kimi-k3",
                  max_output_tokens=131072, motor_max_tokens=8000)
    v, _ = _evaluar(d, min_output_tokens=16384)
    assert [x.regla for x in v] == ["tope_insuficiente"]
    assert "8000" in v[0].detalle and "motor.max_tokens" in v[0].detalle and "kimi" in v[0].detalle


def test_catalogo_por_debajo_del_minimo_dice_model_max_output_tokens():
    v, _ = _evaluar(_despacho(max_output_tokens=4096), min_output_tokens=8192)
    assert [x.regla for x in v] == ["tope_insuficiente"]
    assert "model.max_output_tokens" in v[0].detalle and "deepseek-v4-flash" in v[0].detalle


def test_motor_max_tokens_cero_usa_el_tope_del_catalogo():
    d = _despacho(via_motor=True, max_output_tokens=131072, motor_max_tokens=0)
    _, c = _evaluar(d)
    assert c.tokens_out_max == 131072


def test_credencial_ausente_bloquea_en_transporte_que_cobra():
    v, _ = _evaluar(_despacho(), credencial_activa=False)
    assert [x.regla for x in v] == ["credencial_ausente"]
    assert "deepseek" in v[0].detalle


def test_ollama_sin_credencial_no_bloquea_y_cuesta_cero():
    d = _despacho(transporte="ollama", via_motor=True, max_tokens_param=None,
                  precio_in=None, precio_out=None)
    v, c = _evaluar(d, credencial_activa=False)
    assert v == []
    assert c.usd_max == Decimal(0) and c.motivo == "local"


def test_costo_acotado_es_la_formula_redondeada_hacia_arriba():
    _, c = _evaluar(_despacho(), chars_entrada=2001)
    # (1001 * 0.27 + 8192 * 1.10) / 1e6 = 0.00928147 -> 0.009282
    assert (c.tokens_in_max, c.tokens_out_max, c.llamadas_max) == (1001, 8192, 1)
    assert c.usd_max == Decimal("0.009282") and c.motivo == "acotado"


def test_gemini_directo_cuenta_el_reintento_de_grounding():
    _, c = _evaluar(_despacho(transporte="http_gemini", max_tokens_param=None))
    assert c.llamadas_max == 2


def test_motor_con_schema_validable_cuenta_el_reintento():
    _, c = _evaluar(_despacho(via_motor=True, schema_con_reintento=True))
    assert c.llamadas_max == 2


def test_motor_sin_schema_validable_una_llamada():
    _, c = _evaluar(_despacho(via_motor=True, schema_con_reintento=False))
    assert c.llamadas_max == 1


def test_motor_con_herramientas_cuenta_todas_las_iteraciones():
    _, c = _evaluar(_despacho(via_motor=True, tiene_herramientas=True, schema_con_reintento=True),
                    max_iteraciones=5)
    assert c.llamadas_max == 5


def test_precio_null_no_acotado_y_no_bloquea():
    v, c = _evaluar(_despacho(precio_in=None))
    assert v == []
    assert c.usd_max is None and c.motivo == "sin_precio"


def test_tokens_de_entrada_redondea_hacia_arriba():
    assert [pr.tokens_de_entrada(n, 2) for n in (0, 1, 4, 5)] == [0, 1, 2, 3]


def test_hyde_no_se_evalua_y_cuesta_cero_por_suscripcion():
    c = pr.costo_sin_evaluar(Step(step_index=1, facet="hyde", capability="execute"))
    assert c is not None and c.usd_max == Decimal(0) and c.motivo == "suscripcion"


def test_assemble_no_se_evalua_y_es_mecanico():
    c = pr.costo_sin_evaluar(Step(step_index=2, facet="ada", capability="assemble"))
    assert c is not None and c.usd_max == Decimal(0) and c.motivo == "mecanico"
    assert pr.costo_sin_evaluar(Step(step_index=0, facet="jekyll", capability="research")) is None


def _costo(paso, usd):
    return pr.CostoPaso(paso, "jekyll", "m", 1, 1, 1, None if usd is None else Decimal(usd), "acotado")


def test_veredicto_suma_solo_los_acotados():
    costos = [_costo(0, "0.100000"), _costo(1, None), _costo(2, "0.200000")]
    v = pr.armar_veredicto([], costos, [])
    assert v.ok and v.costo_max_usd == Decimal("0.300000") and v.hay_no_acotados
    malo = pr.armar_veredicto([pr.Violacion(0, "jekyll", "faceta_caida", "x")], costos, ["jekyll"])
    assert not malo.ok and malo.sondeadas == ("jekyll",)


def test_to_dict_serializa_decimales_como_texto():
    v = pr.armar_veredicto([], [_costo(0, "0.100000"), _costo(1, None), _costo(2, "0.200000")], [])
    d = json.loads(json.dumps(v.to_dict()))
    assert d["costo_max_usd"] == "0.300000"
    assert d["pasos_costo"][1]["usd_max"] is None
    assert d["hay_no_acotados"] is True


def test_puede_pedir_reintento_segun_el_validador():
    assert puede_pedir_reintento("code_patch.v1") is True
    assert puede_pedir_reintento("critique.v1") is False
    assert puede_pedir_reintento("") is False
    assert puede_pedir_reintento(None) is False
    assert puede_pedir_reintento("typo.v9") is True
```

- [ ] **Step 2: Verlos fallar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_prevuelo_reglas.py 2>&1 | tail -4`
Expected: `1 error` de colección: `ImportError: cannot import name 'prevuelo_reglas' from 'jacobs'`.

- [ ] **Step 3: Implementar — validador**

En `las_manos/motor_registry/output_validator.py`, después de la definición de `_KNOWN_UNIMPLEMENTED_SCHEMAS`, agregar:
```python
def puede_pedir_reintento(schema_name: str | None) -> bool:
    """True si worker.py puede gastar un SEGUNDO turno por este schema: la
    validación puede fallar y el bucle reintenta una vez (worker.py, rama
    `validation_retried`). Un schema declarado-pendiente acepta texto libre y
    nunca reintenta; un nombre desconocido falla cerrado y SÍ reintenta.

    Público para el pre-vuelo de Jacobs (2026-09-17): el costo máximo de un
    paso de motor cuenta ese turno sin copiar la lista de pendientes."""
    return bool(schema_name) and schema_name not in _KNOWN_UNIMPLEMENTED_SCHEMAS
```

- [ ] **Step 4: Implementar — reglas**

`jacobs/prevuelo_reglas.py`:
```python
"""Jacobs — reglas PURAS del pre-vuelo (spec 2026-09-17 §4).

Sin DB ni red: reciben lo que el catálogo dijo (prevuelo_catalogo) y devuelven
violaciones y costo. El orquestador (jacobs/prevuelo.py) junta todo.

Los tipos viven acá y no en prevuelo.py para que sonda.py pueda importarlos
sin un import circular (prevuelo -> sonda -> prevuelo).

Cuatro chequeos (D4 de Fernando): tope suficiente, credencial viva, faceta
sana (la sonda, en el orquestador) y costo máximo. Un paso que no se puede
acotar NO bloquea: se declara (`hay_no_acotados`) y la Mesa pide confirmación.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal

from contrato_dispatch import ModelDispatchConfigError, _max_output_tokens_value, errores_del_contrato

from jacobs.models import Step

REGLAS = frozenset({
    "tope_insuficiente", "sin_contrato_de_salida", "credencial_ausente",
    "faceta_caida", "faceta_inexistente",
})

# Transportes que cobran por token y necesitan credencial de proveedor
# (§4.3). ollama es local y subprocess (hyde) es suscripción.
TRANSPORTES_QUE_COBRAN = frozenset({"http_openai_compat", "http_gemini"})

_UN_MILLON = Decimal(1_000_000)
# axioma_usage.cost_usd es DECIMAL(10,6): mismo grano, redondeado HACIA ARRIBA
# (un costo máximo que redondea para abajo deja de ser máximo).
_GRANO_USD = Decimal("0.000001")


@dataclass(frozen=True)
class Violacion:
    paso: int
    faceta: str
    regla: str
    detalle: str

    def __post_init__(self) -> None:
        if self.regla not in REGLAS:
            raise ValueError(f"regla de pre-vuelo desconocida: {self.regla!r}")

    def to_dict(self) -> dict:
        return {"paso": self.paso, "faceta": self.faceta, "regla": self.regla, "detalle": self.detalle}


@dataclass(frozen=True)
class CostoPaso:
    paso: int
    faceta: str
    modelo: str | None
    llamadas_max: int
    tokens_in_max: int
    tokens_out_max: int
    usd_max: Decimal | None
    motivo: str

    def to_dict(self) -> dict:
        return {
            "paso": self.paso, "faceta": self.faceta, "modelo": self.modelo,
            "llamadas_max": self.llamadas_max, "tokens_in_max": self.tokens_in_max,
            "tokens_out_max": self.tokens_out_max,
            "usd_max": None if self.usd_max is None else str(self.usd_max),
            "motivo": self.motivo,
        }


@dataclass(frozen=True)
class Veredicto:
    ok: bool
    violaciones: tuple[Violacion, ...]
    costo_max_usd: Decimal
    pasos_costo: tuple[CostoPaso, ...]
    sondeadas: tuple[str, ...]

    @property
    def hay_no_acotados(self) -> bool:
        return any(c.usd_max is None for c in self.pasos_costo)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "violaciones": [v.to_dict() for v in self.violaciones],
            "costo_max_usd": str(self.costo_max_usd),
            "pasos_costo": [c.to_dict() for c in self.pasos_costo],
            "sondeadas": list(self.sondeadas),
            "hay_no_acotados": self.hay_no_acotados,
        }


@dataclass(frozen=True)
class Despacho:
    """Lo que el despacho REAL de un paso va a usar, leído del catálogo."""
    clave_salud: str           # faceta HTTP, o el motor que resuelve MotorPolicy
    via_motor: bool
    transporte: str            # http_openai_compat | http_gemini | ollama | subprocess
    provider_id: str
    base_url: str | None
    modelo: str
    max_tokens_param: str | None
    max_output_tokens: int | None
    motor_max_tokens: int      # 0 = sin presupuesto propio (solo via_motor)
    precio_in: Decimal | None  # USD por millón
    precio_out: Decimal | None
    tiene_herramientas: bool
    schema_con_reintento: bool
    persona: str | None


def errores_de_contrato(d: Despacho) -> list[str]:
    """Los MISMOS validadores que el despacho: http_openai_compat exige nombre
    y tope (contrato_dispatch.faltantes_del_contrato); ollama exige tope
    (limite_de_salida); http_gemini exige el tope de la fila porque sin él no
    hay costo máximo (§4.4). subprocess no lleva."""
    if d.transporte == "subprocess":
        return []
    if d.transporte == "http_openai_compat":
        return [str(e) for _, e in errores_del_contrato(d.modelo, d.max_tokens_param, d.max_output_tokens)]
    try:
        _max_output_tokens_value(d.modelo, d.max_output_tokens)
    except ModelDispatchConfigError as e:
        return [str(e)]
    return []


def tope_efectivo(d: Despacho) -> tuple[int, str]:
    """El mismo cálculo que worker._limite_del_motor: el menor entre
    motor.max_tokens (0 = sin presupuesto) y el tope del catálogo. Presupone
    contrato válido."""
    if d.via_motor and d.motor_max_tokens and d.motor_max_tokens < d.max_output_tokens:
        return d.motor_max_tokens, "motor.max_tokens"
    return d.max_output_tokens, "model.max_output_tokens"


def llamadas_max(d: Despacho, max_iteraciones: int) -> int:
    """Desvío 2 del plan. Motor: el bucle de worker.py está acotado por
    MOTOR_TOOL_LOOP_MAX_ITERATIONS y el reintento de schema es una iteración
    de ese bucle. HTTP gemini: _invoke_http_gemini reintenta sin grounding."""
    if d.via_motor:
        if d.tiene_herramientas:
            return max_iteraciones
        return min(2 if d.schema_con_reintento else 1, max_iteraciones)
    if d.transporte == "http_gemini":
        return 2
    return 1


def tokens_de_entrada(chars: int, chars_por_token: int) -> int:
    return -(-chars // chars_por_token)


def costo_usd(llamadas: int, tokens_in: int, tokens_out: int,
              precio_in: Decimal, precio_out: Decimal) -> Decimal:
    bruto = Decimal(llamadas) * (Decimal(tokens_in) * precio_in + Decimal(tokens_out) * precio_out) / _UN_MILLON
    return bruto.quantize(_GRANO_USD, rounding=ROUND_CEILING)


def costo_sin_evaluar(step: Step) -> CostoPaso | None:
    """Pasos que no consultan catálogo: `assemble` es mecánico
    (executor._assemble_mechanical) y `hyde` es suscripción (sin tope,
    credencial ni sonda, igual que el canario de la Mesa)."""
    if step.capability == "assemble":
        return CostoPaso(step.step_index, step.facet, None, 0, 0, 0, Decimal(0), "mecanico")
    if step.facet == "hyde":
        return CostoPaso(step.step_index, step.facet, None, 0, 0, 0, Decimal(0), "suscripcion")
    return None


def evaluar_paso(
    paso: int,
    faceta: str,
    d: Despacho | None,
    *,
    min_output_tokens: int,
    credencial_activa: bool,
    chars_entrada: int,
    chars_por_token: int,
    max_iteraciones: int,
    detalle_inexistente: str = "",
) -> tuple[list[Violacion], CostoPaso]:
    if d is None:
        detalle = detalle_inexistente or f"la faceta '{faceta}' no tiene fila activa con binding primario"
        return ([Violacion(paso, faceta, "faceta_inexistente", detalle)],
                CostoPaso(paso, faceta, None, 0, 0, 0, None, "faceta_inexistente"))

    violaciones: list[Violacion] = []
    tokens_in = tokens_de_entrada(chars_entrada, chars_por_token)
    llamadas = llamadas_max(d, max_iteraciones)

    errores = errores_de_contrato(d)
    tokens_out = 0
    if errores:
        violaciones.append(Violacion(paso, faceta, "sin_contrato_de_salida", " | ".join(errores)))
    else:
        tokens_out, que_subir = tope_efectivo(d)
        if tokens_out < min_output_tokens:
            de_quien = d.clave_salud if que_subir == "motor.max_tokens" else d.modelo
            violaciones.append(Violacion(
                paso, faceta, "tope_insuficiente",
                f"tope efectivo {tokens_out} < capability.min_output_tokens {min_output_tokens}: "
                f"subí {que_subir} de '{de_quien}' a {min_output_tokens} o más",
            ))

    if d.transporte in TRANSPORTES_QUE_COBRAN and not credencial_activa:
        violaciones.append(Violacion(
            paso, faceta, "credencial_ausente",
            f"el proveedor '{d.provider_id}' no tiene credencial con state='active' en `credential`",
        ))

    if errores:
        costo = CostoPaso(paso, faceta, d.modelo, llamadas, tokens_in, 0, None, "sin_contrato_de_salida")
    elif d.transporte == "ollama":
        costo = CostoPaso(paso, faceta, d.modelo, llamadas, tokens_in, tokens_out, Decimal(0), "local")
    elif d.precio_in is None or d.precio_out is None:
        costo = CostoPaso(paso, faceta, d.modelo, llamadas, tokens_in, tokens_out, None, "sin_precio")
    else:
        costo = CostoPaso(paso, faceta, d.modelo, llamadas, tokens_in, tokens_out,
                          costo_usd(llamadas, tokens_in, tokens_out, d.precio_in, d.precio_out),
                          "acotado")
    return violaciones, costo


def armar_veredicto(violaciones: list[Violacion], costos: list[CostoPaso],
                    sondeadas: list[str]) -> Veredicto:
    total = sum((c.usd_max for c in costos if c.usd_max is not None), Decimal(0))
    return Veredicto(
        ok=not violaciones,
        violaciones=tuple(violaciones),
        costo_max_usd=total,
        pasos_costo=tuple(sorted(costos, key=lambda c: c.paso)),
        sondeadas=tuple(sorted(sondeadas)),
    )
```

- [ ] **Step 5: Verlos pasar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_prevuelo_reglas.py las_manos/_output_validator_test.py 2>&1 | tail -3`
Expected: `0 failed`; `tests/test_prevuelo_reglas.py` aporta `22 passed`.

- [ ] **Step 6: Tripwires**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_payload_max_tokens_literal_tripwire.py tests/test_no_blocking_in_async.py policy/tests/test_no_fail_open_except.py 2>&1 | tail -2`
Expected: `0 failed`.

- [ ] **Step 7: CI y commit**

`tests-puros`: agregar `tests/test_prevuelo_reglas.py` a las dos listas; reemplazar la línea de 759 por:
```
          # 759 -> 781 el 2026-09-17 (pre-vuelo): test_prevuelo_reglas.py (+22) -- contrato,
          #   tope efectivo, credencial, llamadas por transporte, costo redondeado hacia arriba,
          #   no acotados que no bloquean, hyde/assemble y puede_pedir_reintento.
          grep -qE "^781 passed, 1 skipped" /tmp/out || {
            echo "PISO ROTO: se esperaban 781 tests CORRIDOS y exactamente 1 skipped."; exit 1; }
```

```bash
cd /home/fruiz/worktrees/jax-prevuelo && pwd && git branch --show-current && \
git -C /home/fruiz/worktrees/jax-prevuelo add jacobs/prevuelo_reglas.py las_manos/motor_registry/output_validator.py tests/test_prevuelo_reglas.py .github/workflows/policy.yml && \
git -C /home/fruiz/worktrees/jax-prevuelo commit -m "feat(jacobs): reglas puras del pre-vuelo

Contrato de salida, tope efectivo, credencial, llamadas máximas por transporte
y costo máximo; los pasos sin precio se declaran no acotados.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Catálogo, salud y motor resuelto

**Files:**
- Create: `jacobs/prevuelo_catalogo.py`
- Modify: `jacobs/facet_health.py` (docstring, lector de último evento de proveedor, escritor de eventos de sonda)
- Modify: `las_manos/motor_registry/policy.py` (`MotorPolicy.motor_que_despacharia`)
- Test: `tests/test_prevuelo_salud_pura.py` (puro), `tests/test_prevuelo_catalogo_db.py` (DB)
- Modify: `.github/workflows/policy.yml` (jobs `tests-puros` y `jacobs-gobernanza-db`)

**Interfaces:**
- Consumes: `store.get_conn()`, `MotorCatalog.from_db()`, `facet_health.HEALTH_WINDOW_SECONDS`.
- Produces:
  - `MotorPolicy.motor_que_despacharia(requested: str | None, capability: str) -> str | None`
  - `facet_health.OUTCOMES_DE_PROVEEDOR = ("ok", "provider_error")`, `facet_health.SOURCE_PREVUELO = "preflight"`
  - `facet_health.sql_ultimo_evento_de_proveedor(n_claves: int) -> str`
  - `facet_health.salud_de_proveedor(ultimo: tuple[float, str] | None, ahora: float) -> str` → `"sana"` | `"sondear"`
  - `facet_health.ultimo_evento_de_proveedor(cur, claves: set[str], ahora: float) -> dict[str, tuple[float, str]]`
  - `facet_health.registrar_evento_de_sonda(clave: str, outcome: str, detalle: str | None, ts: float) -> None` (detalle recortado a 255)
  - `prevuelo_catalogo.FilaFaceta(key, transport, persona, provider_id, base_url, model_id)`
  - `prevuelo_catalogo.FilaModelo(max_tokens_param, max_output_tokens, precio_in, precio_out)`
  - `prevuelo_catalogo.MotorResuelto(clave, transporte, provider_id, base_url, modelo, max_tokens, tiene_herramientas, output_schema)`
  - `prevuelo_catalogo.Catalogo(facetas: dict[str, FilaFaceta], modelos: dict[tuple[str, str], FilaModelo], min_output_tokens: dict[str, int], proveedores_con_credencial: frozenset[str], salud: dict[str, tuple[float, str]])`
  - `prevuelo_catalogo.sql_facetas(n)`, `sql_modelos(n)`, `sql_min_output_tokens(n)`, `sql_credenciales(n)`
  - `async prevuelo_catalogo.resolver_motores(pasos: list[Step]) -> dict[int, MotorResuelto | None]`
  - `async prevuelo_catalogo.leer_catalogo(*, facetas: set[str], motores: list[MotorResuelto], capabilities: set[str], ahora: float) -> Catalogo`

- [ ] **Step 1: Preparar `jax_memory_test` con el esquema del plan P**

El plan P debe tener en su rama (`/home/fruiz/worktrees/jax-platform-prevuelo`) `capability.min_output_tokens` y `'preflight'` en `facet_health_event.source`. Migrar la base de TESTS con esa rama:
```bash
cd /home/fruiz/worktrees/jax-platform-prevuelo/backend && set -a && source /etc/jax/.env && set +a && export JAX_DB_NAME=jax_memory_test && \
/home/fruiz/jax-platform/backend/.venv/bin/python - <<'PY'
import asyncio
from db.connection import close_pool
from db.migrations import run_migrations
async def main():
    await run_migrations()
    await close_pool()
asyncio.run(main())
print("run_migrations OK")
PY
```
Verificar:
```bash
cd /home/fruiz/worktrees/jax-prevuelo && set -a && source /etc/jax/.env && set +a && export JAX_DB_NAME=jax_memory_test PYTHONPATH=.:las_manos && \
/home/fruiz/jax/las_manos/.venv/bin/python - <<'PY'
import asyncio
from jacobs import store
async def main():
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT DATABASE()")
            print(await cur.fetchone())
            await cur.execute("SELECT COLUMN_TYPE FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='facet_health_event' AND COLUMN_NAME='source'")
            print(await cur.fetchone())
            await cur.execute("SELECT COUNT(*) FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='capability' AND COLUMN_NAME='min_output_tokens'")
            print(await cur.fetchone())
    finally:
        conn.close()
asyncio.run(main())
PY
```
Expected: `('jax_memory_test',)`, un ENUM que contiene `'preflight'`, y `(1,)`. Si falta cualquiera: PARAR y reportar al controlador que la rama de P no está lista. No se simula el esquema.

- [ ] **Step 2: Escribir los tests que fallan**

`tests/test_prevuelo_salud_pura.py`:
```python
"""Salud de nivel proveedor y motor resuelto, sin DB (spec 2026-09-17 §4.5,
desvío 1 del plan).

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ["JAX_DB_NAME"] = "jax_memory_test"

from jacobs import facet_health as fh  # noqa: E402
from motor_registry.policy import MotorPolicy  # noqa: E402


def test_ok_reciente_es_sana():
    assert fh.salud_de_proveedor((1000.0, "ok"), 1060.0) == "sana"


def test_sin_evento_o_fuera_de_ventana_se_sondea():
    assert fh.salud_de_proveedor(None, 1060.0) == "sondear"
    assert fh.salud_de_proveedor((0.0, "ok"), fh.HEALTH_WINDOW_SECONDS + 1.0) == "sondear"


def test_provider_error_reciente_se_vuelve_a_medir():
    assert fh.salud_de_proveedor((1000.0, "provider_error"), 1060.0) == "sondear"


def test_la_consulta_solo_cuenta_eventos_de_proveedor():
    sql = fh.sql_ultimo_evento_de_proveedor(2)
    assert "outcome IN ('ok','provider_error')" in sql
    assert sql.count("%s") == 3  # dos claves + el inicio de la ventana


def test_motor_que_despacharia_respeta_el_pedido_y_la_prioridad():
    class _Catalogo:
        def get_capability(self, nombre):
            return SimpleNamespace(allowed_motors=["kimi", "ada"]) if nombre == "implementation" else None

        def get_motor(self, nombre):
            return {"kimi": SimpleNamespace(enabled=False), "ada": SimpleNamespace(enabled=True)}.get(nombre)

    politica = MotorPolicy(_Catalogo())
    assert politica.motor_que_despacharia(None, "implementation") == "ada"
    assert politica.motor_que_despacharia("kimi", "implementation") is None
    assert politica.motor_que_despacharia("ada", "implementation") == "ada"
    assert politica.motor_que_despacharia(None, "no_existe") is None
```

`tests/test_prevuelo_catalogo_db.py`:
```python
"""Lectura del catálogo del pre-vuelo contra el esquema REAL de jax-platform
(job jacobs-gobernanza-db). Cada test siembra SU proveedor, modelo, faceta,
binding y credencial sintéticos y los borra: no depende de semillas.

Requiere el esquema del plan P: capability.min_output_tokens y
facet_health_event.source con 'preflight'.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from decimal import Decimal

_db = os.environ.get("JAX_DB_NAME", "")
if not _db.endswith("_test"):
    raise RuntimeError(f"JAX_DB_NAME={_db!r}: este test solo corre contra una base *_test.")

from jacobs import facet_health as fh  # noqa: E402
from jacobs import prevuelo_catalogo as pc  # noqa: E402
from jacobs import store  # noqa: E402


class _Semilla:
    def __init__(self):
        sufijo = uuid.uuid4().hex[:8]
        self.proveedor = f"zz-pv-{sufijo}"
        self.faceta = f"zz-pv-{sufijo}"
        self.modelo = f"zz-modelo-{sufijo}"


async def _sembrar(cur, s, *, credencial="active", faceta_status="active"):
    await cur.execute(
        "INSERT INTO provider (id, display_name, base_url, auth_type, is_local) "
        "VALUES (%s, 'prevuelo test', 'https://zz.example/v1', 'api_key', FALSE)", (s.proveedor,))
    await cur.execute(
        "INSERT INTO model (provider_id, model_id, is_alias, status, source, source_checked_at, "
        "max_tokens_param, max_output_tokens, price_input_per_1m_usd, price_output_per_1m_usd) "
        "VALUES (%s, %s, FALSE, 'available', 'manual', NOW(), 'max_tokens', 4096, 0.2700, 1.1000)",
        (s.proveedor, s.modelo))
    await cur.execute("SELECT id FROM model WHERE provider_id=%s AND model_id=%s", (s.proveedor, s.modelo))
    (model_ref,) = await cur.fetchone()
    await cur.execute(
        "INSERT INTO facet (`key`, display_name, transport, status) "
        "VALUES (%s, 'prevuelo test', 'http_openai_compat', %s)", (s.faceta, faceta_status))
    await cur.execute(
        "INSERT INTO facet_binding (facet_key, provider_id, model_id, role, model_ref) "
        "VALUES (%s, %s, %s, 'primary', %s)", (s.faceta, s.proveedor, s.modelo, model_ref))
    if credencial:
        await cur.execute(
            "INSERT INTO credential (provider_id, env_key, encrypted_value, state) "
            "VALUES (%s, 'ZZ_PV_KEY', 'no-se-descifra', %s)", (s.proveedor, credencial))


async def _limpiar(cur, s):
    await cur.execute("DELETE FROM facet_health_event WHERE facet LIKE %s", (s.faceta + "%",))
    await cur.execute("DELETE FROM credential WHERE provider_id=%s", (s.proveedor,))
    await cur.execute("DELETE FROM facet_binding WHERE facet_key=%s", (s.faceta,))
    await cur.execute("DELETE FROM facet WHERE `key`=%s", (s.faceta,))
    await cur.execute("DELETE FROM model WHERE provider_id=%s", (s.proveedor,))
    await cur.execute("DELETE FROM provider WHERE id=%s", (s.proveedor,))


def _con_semilla(cuerpo, **kw):
    async def correr():
        s = _Semilla()
        conn = await store.get_conn()
        try:
            async with conn.cursor() as cur:
                await _sembrar(cur, s, **kw)
            return s, await cuerpo(s, conn)
        finally:
            async with conn.cursor() as cur:
                await _limpiar(cur, s)
            conn.close()
    return asyncio.run(correr())


async def _evento(conn, faceta, outcome, source, ts):
    async with conn.cursor() as cur:
        await cur.execute(
            "INSERT INTO facet_health_event (facet, outcome, source, detail, ts) VALUES (%s, %s, %s, NULL, %s)",
            (faceta, outcome, source, ts))


async def _explain(conn, sql, params):
    async with conn.cursor() as cur:
        await cur.execute("EXPLAIN " + sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in await cur.fetchall()]


def test_lee_faceta_modelo_precio_y_contrato():
    async def cuerpo(s, conn):
        return await pc.leer_catalogo(facetas={s.faceta}, motores=[], capabilities=set(), ahora=time.time())
    s, cat = _con_semilla(cuerpo)
    fila = cat.facetas[s.faceta]
    assert (fila.provider_id, fila.model_id, fila.base_url, fila.transport) == (
        s.proveedor, s.modelo, "https://zz.example/v1", "http_openai_compat")
    assert cat.modelos[(s.proveedor, s.modelo)] == pc.FilaModelo(
        "max_tokens", 4096, Decimal("0.2700"), Decimal("1.1000"))
    assert s.proveedor in cat.proveedores_con_credencial


def test_faceta_inactiva_no_aparece():
    async def cuerpo(s, conn):
        return await pc.leer_catalogo(facetas={s.faceta}, motores=[], capabilities=set(), ahora=time.time())
    s, cat = _con_semilla(cuerpo, faceta_status="disabled")
    assert s.faceta not in cat.facetas


def test_credencial_revocada_no_cuenta():
    async def cuerpo(s, conn):
        return await pc.leer_catalogo(facetas={s.faceta}, motores=[], capabilities=set(), ahora=time.time())
    s, cat = _con_semilla(cuerpo, credencial="revoked")
    assert s.proveedor not in cat.proveedores_con_credencial


def test_salud_toma_el_ultimo_evento_de_proveedor_e_ignora_los_del_gate():
    ahora = time.time()

    async def cuerpo(s, conn):
        await _evento(conn, s.faceta, "ok", "chat", ahora - 600)
        await _evento(conn, s.faceta, "provider_error", "preflight", ahora - 300)
        await _evento(conn, s.faceta, "unsupported_transport", "canary_periodic", ahora - 10)
        await _evento(conn, s.faceta, "gate_denied", "chat", ahora - 5)
        return await pc.leer_catalogo(facetas={s.faceta}, motores=[], capabilities=set(), ahora=ahora)
    s, cat = _con_semilla(cuerpo)
    assert cat.salud[s.faceta] == (ahora - 300, "provider_error")


def test_salud_fuera_de_ventana_no_cuenta():
    ahora = time.time()

    async def cuerpo(s, conn):
        await _evento(conn, s.faceta, "ok", "preflight", ahora - fh.HEALTH_WINDOW_SECONDS - 5)
        return await pc.leer_catalogo(facetas={s.faceta}, motores=[], capabilities=set(), ahora=ahora)
    s, cat = _con_semilla(cuerpo)
    assert s.faceta not in cat.salud


def test_registrar_evento_de_sonda_escribe_source_preflight_y_recorta():
    ahora = time.time()

    async def cuerpo(s, conn):
        await fh.registrar_evento_de_sonda(s.faceta, "ok", "x" * 300, ahora)
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT outcome, source, CHAR_LENGTH(detail) FROM facet_health_event WHERE facet=%s",
                (s.faceta,))
            return await cur.fetchall()
    _, filas = _con_semilla(cuerpo)
    assert list(filas) == [("ok", "preflight", 255)]


def test_min_output_tokens_sale_de_la_capability():
    async def cuerpo(s, conn):
        async with conn.cursor() as cur:
            await cur.execute("SELECT min_output_tokens FROM capability WHERE `key`='research'")
            (esperado,) = await cur.fetchone()
        cat = await pc.leer_catalogo(facetas=set(), motores=[], capabilities={"research"}, ahora=time.time())
        return esperado, cat.min_output_tokens
    _, (esperado, minimos) = _con_semilla(cuerpo)
    assert minimos == {"research": int(esperado)}


def test_explain_salud_usa_idx_facet_ts():
    ahora = time.time()

    async def cuerpo(s, conn):
        # Volumen para que el optimizador no prefiera un scan por tabla chica.
        for n in range(30):
            for k in range(10):
                await _evento(conn, f"{s.faceta}-rel-{n:02d}", "ok", "chat", ahora - k)
        await _evento(conn, s.faceta, "ok", "preflight", ahora)
        return await _explain(conn, fh.sql_ultimo_evento_de_proveedor(1),
                              (s.faceta, ahora - fh.HEALTH_WINDOW_SECONDS))
    _, filas = _con_semilla(cuerpo)
    assert any(f["key"] == "idx_facet_ts" for f in filas), filas
    assert all("filesort" not in (f.get("Extra") or "") for f in filas), filas


def test_explain_credencial_y_modelos_usan_sus_indices():
    async def cuerpo(s, conn):
        cred = await _explain(conn, pc.sql_credenciales(1), (s.proveedor,))
        modelos = await _explain(conn, pc.sql_modelos(1), (s.proveedor, s.modelo))
        return cred, modelos
    _, (cred, modelos) = _con_semilla(cuerpo)
    assert [f["key"] for f in cred] == ["idx_provider_state"], cred
    assert [f["key"] for f in modelos] == ["uk_provider_model"], modelos
```

- [ ] **Step 3: Verlos fallar**

Run (puros): `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_prevuelo_salud_pura.py 2>&1 | tail -7`
Expected: `5 failed` con `AttributeError: module 'jacobs.facet_health' has no attribute 'salud_de_proveedor'` / `'sql_ultimo_evento_de_proveedor'` y `AttributeError: 'MotorPolicy' object has no attribute 'motor_que_despacharia'`.

Run (DB): `cd /home/fruiz/worktrees/jax-prevuelo && set -a && source /etc/jax/.env && set +a && export JAX_DB_NAME=jax_memory_test PYTHONPATH=.:las_manos && /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_prevuelo_catalogo_db.py 2>&1 | tail -4`
Expected: `1 error` de colección: `ImportError: cannot import name 'prevuelo_catalogo' from 'jacobs'`.

- [ ] **Step 4: Implementar — política de motores**

En `las_manos/motor_registry/policy.py`, dentro de `class MotorPolicy`, justo antes de `def _resolve_motor`, agregar:
```python
    def motor_que_despacharia(self, requested: str | None, capability: str) -> str | None:
        """El motor que check() resolvería para (requested, capability), o None
        si ninguno. Público para el pre-vuelo de Jacobs (2026-09-17): medir
        costo, tope y salud del motor que DE VERDAD despacha un paso con
        motor=None, sin copiar la regla de prioridad."""
        cap = self._catalog.get_capability(capability)
        if cap is None:
            return None
        return self._resolve_motor(requested, cap)
```

- [ ] **Step 5: Implementar — salud**

En `jacobs/facet_health.py`:

1. Reemplazar las dos primeras líneas del docstring del módulo
```python
"""Lector UNICO de salud de facets. La salud se calcula EXCLUSIVAMENTE
aca, desde facet_health_event. Quien escribe esa tabla es
jax-platform/backend/facet_health.py; quien la lee es solo este modulo.
```
por
```python
"""Lector UNICO de salud de facets. La salud se calcula EXCLUSIVAMENTE
aca, desde facet_health_event. La escriben jax-platform/backend/
facet_health.py (chat y canarios) y, desde 2026-09-17, la sonda del
pre-vuelo de Jacobs (source='preflight', registrar_evento_de_sonda abajo);
quien la LEE es solo este modulo.
```

2. Agregar al final del archivo:
```python
# ----------------------------------------------------------------
#  Pre-vuelo (spec 2026-09-17 §4.5)
# ----------------------------------------------------------------
# Solo cuentan los eventos de NIVEL PROVEEDOR. Los gate_*, unbound y
# unsupported_transport son del gate de la Mesa (kimi escribe siempre
# unsupported_transport porque la Mesa no lo despacha) y probe_error es una
# falla de la sonda: ninguno dice nada del proveedor.
OUTCOMES_DE_PROVEEDOR = ("ok", "provider_error")
SOURCE_PREVUELO = "preflight"
_LARGO_DETALLE = 255  # facet_health_event.detail VARCHAR(255)

_SQL_EVENTO_DE_SONDA = (
    "INSERT INTO facet_health_event (facet, outcome, source, detail, ts) "
    "VALUES (%s, %s, %s, %s, %s)"
)


def sql_ultimo_evento_de_proveedor(n_claves: int) -> str:
    """Último evento ok/provider_error por clave dentro de la ventana. Va por
    idx_facet_ts (facet, ts) -- EXPLAIN en tests/test_prevuelo_catalogo_db.py."""
    ph = ",".join(["%s"] * n_claves)
    return (
        "SELECT e.facet, e.ts, e.outcome FROM facet_health_event e "
        "JOIN (SELECT facet, MAX(ts) mt FROM facet_health_event "
        f"      WHERE facet IN ({ph}) AND ts >= %s AND outcome IN ('ok','provider_error') "
        "      GROUP BY facet) m "
        "  ON m.facet = e.facet AND m.mt = e.ts "
        "WHERE e.outcome IN ('ok','provider_error')"
    )


def salud_de_proveedor(ultimo: tuple[float, str] | None, ahora: float) -> str:
    """PURA. 'sana' solo con un `ok` dentro de la ventana; cualquier otra cosa
    (sin evento, provider_error, viejo) se vuelve a medir: decide un dato
    fresco, no uno de hace una hora."""
    if ultimo is None or ultimo[0] < ahora - HEALTH_WINDOW_SECONDS:
        return "sondear"
    return "sana" if ultimo[1] == _OK else "sondear"


async def ultimo_evento_de_proveedor(cur, claves: set[str], ahora: float) -> dict[str, tuple[float, str]]:
    if not claves:
        return {}
    ordenadas = sorted(claves)
    await cur.execute(
        sql_ultimo_evento_de_proveedor(len(ordenadas)),
        (*ordenadas, ahora - HEALTH_WINDOW_SECONDS),
    )
    return {faceta: (float(ts), outcome) for faceta, ts, outcome in await cur.fetchall()}


async def registrar_evento_de_sonda(clave: str, outcome: str, detalle: str | None, ts: float) -> None:
    """Escribe el resultado de una sonda del pre-vuelo. Quien llama decide qué
    hacer si falla (sonda.py: el veredicto se mantiene y se cuenta)."""
    if outcome not in OUTCOMES_DE_PROVEEDOR:
        raise ValueError(f"outcome de sonda inválido: {outcome!r}")
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                _SQL_EVENTO_DE_SONDA,
                (clave, outcome, SOURCE_PREVUELO, detalle[:_LARGO_DETALLE] if detalle else None, ts),
            )
    finally:
        conn.close()
```

- [ ] **Step 6: Implementar — catálogo**

`jacobs/prevuelo_catalogo.py`:
```python
"""Jacobs — lectura del catálogo para el pre-vuelo (spec 2026-09-17 §4.2-§4.5).

Una conexión por pre-vuelo, no por paso: facetas HTTP con su binding, filas de
`model` (contrato y precio), capability.min_output_tokens, credenciales
ACTIVAS y el último evento de salud de nivel proveedor. Los motores se
resuelven con MotorCatalog + MotorPolicy (la misma regla que despacha).

Credencial (§4.3): se lee la TABLA, no el resolver. resolve_credential_
instrumented cae a .env aunque la credencial esté revocada y resolve_facet
puede servir una copia de hasta 300 s; el pre-vuelo no hereda ninguno de los
dos (su retiro es del frente E / B1.4).

Sin caché, a propósito: son cinco consultas por PK o índice sobre tablas de
decenas de filas, y el pre-vuelo existe para ver la verdad de AHORA. La carga
se mide en la Task 15 del plan.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from jacobs import store
from jacobs.facet_health import ultimo_evento_de_proveedor
from jacobs.models import Step


@dataclass(frozen=True)
class FilaFaceta:
    key: str
    transport: str
    persona: str | None
    provider_id: str
    base_url: str | None
    model_id: str


@dataclass(frozen=True)
class FilaModelo:
    max_tokens_param: str | None
    max_output_tokens: int | None
    precio_in: Decimal | None
    precio_out: Decimal | None


@dataclass(frozen=True)
class MotorResuelto:
    clave: str
    transporte: str
    provider_id: str
    base_url: str | None
    modelo: str
    max_tokens: int
    tiene_herramientas: bool
    output_schema: str


@dataclass(frozen=True)
class Catalogo:
    facetas: dict[str, FilaFaceta]
    modelos: dict[tuple[str, str], FilaModelo]
    min_output_tokens: dict[str, int]
    proveedores_con_credencial: frozenset[str]
    salud: dict[str, tuple[float, str]]


def _ph(n: int) -> str:
    return ",".join(["%s"] * n)


def sql_facetas(n: int) -> str:
    return (
        "SELECT f.`key`, f.transport, f.persona, b.provider_id, p.base_url, m.model_id "
        "FROM facet f "
        "JOIN facet_binding b ON b.facet_key = f.`key` AND b.role = 'primary' "
        "JOIN provider p ON p.id = b.provider_id "
        "JOIN model m ON m.id = b.model_ref "
        f"WHERE f.`key` IN ({_ph(n)}) AND f.status = 'active'"
    )


def sql_modelos(n: int) -> str:
    return (
        "SELECT provider_id, model_id, max_tokens_param, max_output_tokens, "
        "price_input_per_1m_usd, price_output_per_1m_usd FROM model WHERE "
        + " OR ".join(["(provider_id = %s AND model_id = %s)"] * n)
    )


def sql_min_output_tokens(n: int) -> str:
    return f"SELECT `key`, min_output_tokens FROM capability WHERE `key` IN ({_ph(n)})"


def sql_credenciales(n: int) -> str:
    return (
        "SELECT DISTINCT provider_id FROM credential "
        f"WHERE provider_id IN ({_ph(n)}) AND state = 'active'"
    )


def _decimal(valor) -> Decimal | None:
    return None if valor is None else Decimal(valor)


async def resolver_motores(pasos: list[Step]) -> dict[int, MotorResuelto | None]:
    """Por step_index, el motor que el Motor Registry usaría (None si ninguno)."""
    if not pasos:
        return {}
    from motor_registry.catalog import MotorCatalog
    from motor_registry.policy import MotorPolicy

    catalogo = await MotorCatalog.from_db()
    politica = MotorPolicy(catalogo)
    salida: dict[int, MotorResuelto | None] = {}
    for paso in pasos:
        clave = politica.motor_que_despacharia(paso.motor, paso.capability)
        entrada = catalogo.get_motor(clave) if clave else None
        cap = catalogo.get_capability(paso.capability)
        salida[paso.step_index] = None if entrada is None else MotorResuelto(
            clave=clave,
            transporte=entrada.transport,
            provider_id=entrada.provider_id,
            base_url=entrada.api_url or None,
            modelo=entrada.model,
            max_tokens=entrada.max_tokens,
            tiene_herramientas=entrada.has_tool_access,
            output_schema=cap.output_schema if cap else "",
        )
    return salida


async def leer_catalogo(
    *,
    facetas: set[str],
    motores: list[MotorResuelto],
    capabilities: set[str],
    ahora: float,
) -> Catalogo:
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            filas_facetas: dict[str, FilaFaceta] = {}
            if facetas:
                claves = sorted(facetas)
                await cur.execute(sql_facetas(len(claves)), claves)
                for key, transport, persona, provider_id, base_url, model_id in await cur.fetchall():
                    filas_facetas[key] = FilaFaceta(key, transport, persona, provider_id, base_url, model_id)

            pares = sorted(
                {(f.provider_id, f.model_id) for f in filas_facetas.values()}
                | {(m.provider_id, m.modelo) for m in motores}
            )
            modelos: dict[tuple[str, str], FilaModelo] = {}
            if pares:
                await cur.execute(sql_modelos(len(pares)), [x for par in pares for x in par])
                for provider_id, model_id, param, max_out, p_in, p_out in await cur.fetchall():
                    modelos[(provider_id, model_id)] = FilaModelo(
                        param, None if max_out is None else int(max_out), _decimal(p_in), _decimal(p_out))

            minimos: dict[str, int] = {}
            if capabilities:
                caps = sorted(capabilities)
                await cur.execute(sql_min_output_tokens(len(caps)), caps)
                minimos = {key: int(valor) for key, valor in await cur.fetchall()}

            proveedores = sorted({par[0] for par in pares})
            con_credencial: frozenset[str] = frozenset()
            if proveedores:
                await cur.execute(sql_credenciales(len(proveedores)), proveedores)
                con_credencial = frozenset(r[0] for r in await cur.fetchall())

            salud = await ultimo_evento_de_proveedor(
                cur, set(filas_facetas) | {m.clave for m in motores}, ahora,
            )
    finally:
        conn.close()
    return Catalogo(filas_facetas, modelos, minimos, con_credencial, salud)
```

- [ ] **Step 7: Verlos pasar**

Run (puros): `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_prevuelo_salud_pura.py jacobs/_facet_health_test.py 2>&1 | tail -2`
Expected: `0 failed`; `tests/test_prevuelo_salud_pura.py` aporta `5 passed`.

Run (DB): `cd /home/fruiz/worktrees/jax-prevuelo && set -a && source /etc/jax/.env && set +a && export JAX_DB_NAME=jax_memory_test PYTHONPATH=.:las_manos && /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -v tests/test_prevuelo_catalogo_db.py 2>&1 | tail -12`
Expected: `9 passed`. Si un EXPLAIN no usa el índice esperado, se pega su salida en el ledger y se consulta: no se relaja el test.

- [ ] **Step 8: CI y commit**

`tests-puros`: agregar `tests/test_prevuelo_salud_pura.py` a las dos listas; reemplazar la línea de 781 por:
```
          # 781 -> 786 el 2026-09-17: test_prevuelo_salud_pura.py (+5) -- salud de nivel
          #   proveedor (gate_* no cuenta, provider_error se vuelve a medir) y motor resuelto.
          grep -qE "^786 passed, 1 skipped" /tmp/out || {
            echo "PISO ROTO: se esperaban 786 tests CORRIDOS y exactamente 1 skipped."; exit 1; }
```
`jacobs-gobernanza-db`: agregar `tests/test_prevuelo_catalogo_db.py` después de `tests/test_run_epoch_db.py` en las dos listas; reemplazar la línea de 33 por:
```
          # 33 -> 42 el 2026-09-17 (pre-vuelo): test_prevuelo_catalogo_db.py (+9) -- faceta,
          #   modelo, precio, credencial activa, salud de proveedor, evento de sonda y EXPLAIN
          #   (idx_facet_ts, idx_provider_state, uk_provider_model). Requiere el esquema de P.
          grep -qE "^42 passed" /tmp/out || {
            echo "PISO ROTO: se esperaban 42 tests CORRIDOS."; exit 1; }
```

```bash
cd /home/fruiz/worktrees/jax-prevuelo && pwd && git branch --show-current && \
git -C /home/fruiz/worktrees/jax-prevuelo add jacobs/prevuelo_catalogo.py jacobs/facet_health.py las_manos/motor_registry/policy.py tests/test_prevuelo_salud_pura.py tests/test_prevuelo_catalogo_db.py .github/workflows/policy.yml && \
git -C /home/fruiz/worktrees/jax-prevuelo commit -m "feat(jacobs): catálogo, salud y motor resuelto para el pre-vuelo

Una conexión por pre-vuelo: facetas, contrato y precio, min_output_tokens,
credenciales activas de la tabla y último evento de salud de proveedor. El
motor de un paso es el que resolvería MotorPolicy.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: La sonda

**Files:**
- Create: `jacobs/sonda.py`
- Modify: `jacobs/usage_writer.py:105-193` (`record_direct_usage` recibe `request_type`)
- Test: `tests/test_prevuelo_sonda.py`
- Modify: `.github/workflows/policy.yml` (job `tests-puros`)

**Interfaces:**
- Consumes: `prevuelo_config.sonda_timeout_s()`, `prevuelo_config.sonda_max_tokens()` (Task 1); `prevuelo_reglas.Despacho` (Task 5); `facet_health.registrar_evento_de_sonda` (Task 6); `facet_resolver.resolve_facet`; `credential_resolver.resolve_credential_instrumented`; `motor_registry.worker._call_http_openai_compat`; `executor.OLLAMA_URL`.
- Produces:
  - `sonda.ResultadoSonda(ok: bool, detalle: str | None, tokens_in: int = 0, tokens_out: int = 0)`
  - `async sonda.sondear(clave: str, d: Despacho, *, user_id: str | None = None, tenant_id: str | None = None) -> ResultadoSonda` — nunca lanza salvo `CancelledError`; registra el evento y el uso.
  - `sonda.registros_perdidos() -> int`
  - `sonda.MENSAJE_DE_SONDA`, `sonda.REQUEST_TYPE_SONDA = "preflight_probe"`
  - `usage_writer.record_direct_usage(user_id, tenant_id, facet, provider_id, model, tokens_in, tokens_out, request_type: str = "pipeline")`

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_prevuelo_sonda.py`:
```python
"""La sonda del pre-vuelo (spec 2026-09-17 §4.5, §8). Sin red ni DB: httpx, el
resolver, el registro de salud y el de uso van mockeados.

«Responde» = 2xx del proveedor, aunque el contenido venga cortado: mide
disponibilidad, no calidad.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import json
import os
from decimal import Decimal
from unittest.mock import ANY, AsyncMock, patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import httpx  # noqa: E402

from facet_resolver import ResolvedFacet  # noqa: E402
from jacobs import sonda, usage_writer  # noqa: E402
from jacobs.prevuelo_reglas import Despacho  # noqa: E402


def _despacho(**cambios):
    base = dict(
        clave_salud="jekyll", via_motor=False, transporte="http_openai_compat",
        provider_id="deepseek", base_url="https://api.example/v1", modelo="deepseek-v4-flash",
        max_tokens_param="max_tokens", max_output_tokens=8192, motor_max_tokens=0,
        precio_in=Decimal("0.27"), precio_out=Decimal("1.10"),
        tiene_herramientas=False, schema_con_reintento=False, persona=None,
    )
    base.update(cambios)
    return Despacho(**base)


def _faceta(**cambios):
    base = dict(key="jekyll", provider_id="deepseek", base_url="https://api.example/v1",
                model="deepseek-v4-flash", credential="k-secreta", transport="http_openai_compat",
                persona=None, params=None)
    base.update(cambios)
    return ResolvedFacet(**base)


class _Resp:
    def __init__(self, status, cuerpo):
        self.status_code = status
        self._cuerpo = cuerpo
        self.text = cuerpo if isinstance(cuerpo, str) else json.dumps(cuerpo)

    def json(self):
        return self._cuerpo


_OK_OPENAI = {"choices": [{"message": {"content": "o"}, "finish_reason": "length"}],
              "usage": {"prompt_tokens": 9, "completion_tokens": 16}}


def _sondear(monkeypatch, d, respuesta, faceta=None, registrar=None, uso=None):
    capturado = {}

    async def post(self, url, headers=None, json=None, **kw):
        capturado.update(url=url, headers=headers, json=json)
        if isinstance(respuesta, Exception):
            raise respuesta
        return respuesta

    monkeypatch.setenv("JAX_PREVUELO_SONDA_MAX_TOKENS", "16")
    monkeypatch.setenv("JAX_PREVUELO_SONDA_TIMEOUT_S", "1")
    with patch("httpx.AsyncClient.post", post), \
         patch.object(sonda, "resolve_facet", AsyncMock(return_value=faceta or _faceta())), \
         patch.object(sonda.facet_health, "registrar_evento_de_sonda", registrar or AsyncMock()), \
         patch.object(sonda, "record_direct_usage", uso or AsyncMock()):
        r = asyncio.run(sonda.sondear(d.clave_salud, d))
    return r, capturado


def test_openai_compat_2xx_aunque_venga_cortada_responde(monkeypatch):
    r, c = _sondear(monkeypatch, _despacho(), _Resp(200, _OK_OPENAI))
    assert r.ok and r.detalle is None
    assert c["url"] == "https://api.example/v1/chat/completions"
    assert c["json"]["max_tokens"] == 16 and c["json"]["stream"] is False
    assert c["headers"]["Authorization"] == "Bearer k-secreta"


def test_el_limite_es_el_menor_entre_config_y_catalogo(monkeypatch):
    d = _despacho(max_tokens_param="max_completion_tokens", max_output_tokens=8)
    _, c = _sondear(monkeypatch, d, _Resp(200, _OK_OPENAI))
    assert c["json"]["max_completion_tokens"] == 8
    assert "max_tokens" not in c["json"]


def test_gemini_manda_cabecera_y_maxOutputTokens_sin_herramientas(monkeypatch):
    f = _faceta(key="hipatia", provider_id="gemini", base_url="https://g.example/v1beta",
                model="gemini-x", credential="AIza-secreta", transport="http_gemini")
    d = _despacho(clave_salud="hipatia", transporte="http_gemini", provider_id="gemini",
                  modelo="gemini-x", max_tokens_param=None, max_output_tokens=65536)
    cuerpo = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}],
              "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1}}
    r, c = _sondear(monkeypatch, d, _Resp(200, cuerpo), faceta=f)
    assert r.ok
    assert c["url"] == "https://g.example/v1beta/models/gemini-x:generateContent"
    assert c["headers"] == {"x-goog-api-key": "AIza-secreta"}
    assert c["json"]["generationConfig"] == {"maxOutputTokens": 16}
    assert "tools" not in c["json"]


def test_http_5xx_no_responde_y_el_detalle_va_redactado(monkeypatch):
    r, _ = _sondear(monkeypatch, _despacho(), _Resp(503, "caído, la key k-secreta no sirve"))
    assert not r.ok
    assert "503" in r.detalle and "k-secreta" not in r.detalle


def test_timeout_da_faceta_caida_con_los_segundos(monkeypatch):
    r, _ = _sondear(monkeypatch, _despacho(), httpx.ReadTimeout("lento"))
    assert not r.ok and r.detalle == "timeout de sonda (1s)"


def test_motor_usa_el_cliente_del_worker_con_su_credencial(monkeypatch):
    d = _despacho(clave_salud="kimi", via_motor=True, provider_id="moonshot", modelo="kimi-k3",
                  base_url="https://api.moonshot.example/v1", max_output_tokens=131072,
                  motor_max_tokens=8000)
    llamar = AsyncMock(return_value=_OK_OPENAI)
    monkeypatch.setenv("JAX_PREVUELO_SONDA_MAX_TOKENS", "16")
    with patch("motor_registry.worker._call_http_openai_compat", llamar), \
         patch.object(sonda, "resolve_credential_instrumented", AsyncMock(return_value="k-moon")), \
         patch.object(sonda.facet_health, "registrar_evento_de_sonda", AsyncMock()), \
         patch.object(sonda, "record_direct_usage", AsyncMock()):
        r = asyncio.run(sonda.sondear("kimi", d))
    assert r.ok
    kw = llamar.await_args.kwargs
    assert (kw["api_url"], kw["model"], kw["api_key"], kw["limite"]) == (
        "https://api.moonshot.example/v1", "kimi-k3", "k-moon", {"max_tokens": 16})


def test_motor_ollama_sin_credencial_y_con_max_tokens(monkeypatch):
    d = _despacho(clave_salud="jax_local", via_motor=True, transporte="ollama", provider_id="ollama",
                  modelo="qwen3-coder:30b", base_url="http://localhost:11434/v1",
                  max_tokens_param=None, precio_in=None, precio_out=None)
    llamar, credencial = AsyncMock(return_value=_OK_OPENAI), AsyncMock()
    monkeypatch.setenv("JAX_PREVUELO_SONDA_MAX_TOKENS", "16")
    with patch("motor_registry.worker._call_http_openai_compat", llamar), \
         patch.object(sonda, "resolve_credential_instrumented", credencial), \
         patch.object(sonda.facet_health, "registrar_evento_de_sonda", AsyncMock()), \
         patch.object(sonda, "record_direct_usage", AsyncMock()):
        asyncio.run(sonda.sondear("jax_local", d))
    credencial.assert_not_awaited()
    assert llamar.await_args.kwargs["api_key"] == ""
    assert llamar.await_args.kwargs["limite"] == {"max_tokens": 16}


def test_registra_el_evento_con_el_resultado(monkeypatch):
    registrar = AsyncMock()
    _sondear(monkeypatch, _despacho(), _Resp(200, _OK_OPENAI), registrar=registrar)
    registrar.assert_awaited_once_with("jekyll", "ok", None, ANY)
    registrar = AsyncMock()
    r, _ = _sondear(monkeypatch, _despacho(), _Resp(500, "error"), registrar=registrar)
    registrar.assert_awaited_once_with("jekyll", "provider_error", r.detalle, ANY)


def test_si_no_se_puede_registrar_el_veredicto_se_mantiene_y_se_cuenta(monkeypatch):
    antes = sonda.registros_perdidos()
    r, _ = _sondear(monkeypatch, _despacho(), _Resp(200, _OK_OPENAI),
                    registrar=AsyncMock(side_effect=RuntimeError("base caída")))
    assert r.ok
    assert sonda.registros_perdidos() == antes + 1


def test_el_uso_de_la_sonda_se_registra_como_preflight_probe(monkeypatch):
    uso = AsyncMock()
    _sondear(monkeypatch, _despacho(), _Resp(200, _OK_OPENAI), uso=uso)
    uso.assert_awaited_once_with(None, None, "jekyll", "deepseek", "deepseek-v4-flash", 9, 16,
                                 request_type="preflight_probe")


def test_record_direct_usage_encola_con_el_request_type_pedido(monkeypatch):
    # Host y puerto de mentira: sin ellos _db_cfg() lanza ANTES de conectar y
    # el test no ejercitaría el camino de la conexión caída.
    monkeypatch.setenv("JAX_DB_HOST", "127.0.0.1")
    monkeypatch.setenv("JAX_DB_PORT", "1")
    encolar = AsyncMock(return_value="spool-1")
    with patch.object(usage_writer.aiomysql, "connect", AsyncMock(side_effect=OSError("sin base"))), \
         patch.object(usage_writer, "encolar_uso", encolar):
        asyncio.run(usage_writer.record_direct_usage(
            "1", "1", "jekyll", "deepseek", "m", 1, 2, request_type="preflight_probe"))
    assert encolar.await_args.args[0]["request_type"] == "preflight_probe"


def test_record_direct_usage_inserta_con_el_request_type_pedido(monkeypatch):
    monkeypatch.setenv("JAX_DB_HOST", "127.0.0.1")
    monkeypatch.setenv("JAX_DB_PORT", "1")
    ejecutados = []

    class _Cur:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def execute(self, sql, params=None):
            ejecutados.append((sql, params))

        async def fetchone(self):
            return None

    class _Conn:
        def cursor(self):
            return _Cur()

        def close(self):
            pass

    with patch.object(usage_writer.aiomysql, "connect", AsyncMock(return_value=_Conn())):
        asyncio.run(usage_writer.record_direct_usage(
            "1", "1", "jekyll", "deepseek", "m", 1, 2, request_type="preflight_probe"))
        asyncio.run(usage_writer.record_direct_usage("1", "1", "jekyll", "deepseek", "m", 1, 2))
    inserts = [p for sql, p in ejecutados if sql.startswith("INSERT INTO axioma_usage")]
    assert [p[-1] for p in inserts] == ["preflight_probe", "pipeline"]
```

- [ ] **Step 2: Verlos fallar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_prevuelo_sonda.py 2>&1 | tail -4`
Expected: `1 error` de colección: `ImportError: cannot import name 'sonda' from 'jacobs'`.

- [ ] **Step 3: Implementar — `request_type` en el escritor de uso**

En `jacobs/usage_writer.py`:
1. Reemplazar la firma
```python
async def record_direct_usage(
    user_id: str | None,
    tenant_id: str | None,
    facet: str,
    provider_id: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
) -> None:
```
por
```python
async def record_direct_usage(
    user_id: str | None,
    tenant_id: str | None,
    facet: str,
    provider_id: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    request_type: str = "pipeline",
) -> None:
```
2. Reemplazar
```python
                    "INSERT INTO axioma_usage (tenant_id, user_id, facet, model, tokens_in, tokens_out, cost_usd, request_type) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, 'pipeline')",
                    (
                        int(tenant_id) if tenant_id is not None else None,
                        int(user_id) if user_id is not None else None,
                        facet, model, tokens_in, tokens_out, cost,
                    ),
```
por
```python
                    "INSERT INTO axioma_usage (tenant_id, user_id, facet, model, tokens_in, tokens_out, cost_usd, request_type) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        int(tenant_id) if tenant_id is not None else None,
                        int(user_id) if user_id is not None else None,
                        facet, model, tokens_in, tokens_out, cost, request_type,
                    ),
```
3. Reemplazar `        "request_type": "pipeline",` (en el `encolar_uso`) por `        "request_type": request_type,`.
4. Al docstring del módulo, después del párrafo de `request_type='pipeline'`, agregar:
```
request_type='preflight_probe' (2026-09-17): la sonda del pre-vuelo de Jacobs
(jacobs/sonda.py) paga una llamada mínima; se registra aparte para que se vea.
```

- [ ] **Step 4: Implementar — la sonda**

`jacobs/sonda.py`:
```python
"""Jacobs — sonda de disponibilidad del pre-vuelo (spec 2026-09-17 §4.5, §8).

Una llamada MÍNIMA por el transporte real de la clave: mismo endpoint,
resolver y credencial que el ejecutor, pero payload propio (sin google_search
y con el menor entre JAX_PREVUELO_SONDA_MAX_TOKENS y el tope del catálogo):
las _invoke_* mandarían herramientas y el tope completo del modelo.

«Responde» = 2xx del proveedor aunque el contenido venga cortado: mide
disponibilidad, no calidad. Timeout propio (JAX_PREVUELO_SONDA_TIMEOUT_S).

Cada resultado se registra en facet_health_event (source='preflight', outcome
ok/provider_error) para que el próximo pre-vuelo dentro de la ventana no
vuelva a sondear, y su uso en axioma_usage (request_type='preflight_probe'):
se paga y se ve. Si el registro de salud falla, el veredicto se mantiene (el
dato es la respuesta del proveedor, no la fila), se loguea WARNING y se cuenta.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx
from credential_resolver import resolve_credential_instrumented
from facet_resolver import resolve_facet
from redaccion import recortar_redactado

from jacobs import facet_health, prevuelo_config
from jacobs.prevuelo_reglas import Despacho
from jacobs.usage_writer import record_direct_usage

logger = logging.getLogger("jacobs.sonda")

MENSAJE_DE_SONDA = "Respondé solamente: ok"
REQUEST_TYPE_SONDA = "preflight_probe"
_LARGO_DETALLE = 200


@dataclass(frozen=True)
class ResultadoSonda:
    ok: bool
    detalle: str | None
    tokens_in: int = 0
    tokens_out: int = 0


_registros_perdidos = 0


def registros_perdidos() -> int:
    """Eventos de sonda que no se pudieron escribir desde que arrancó el proceso."""
    return _registros_perdidos


async def _post(url: str, headers: dict, payload: dict, timeout: int, secretos: list[str]) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
    if not 200 <= resp.status_code < 300:
        raise RuntimeError(f"HTTP {resp.status_code}: {recortar_redactado(resp.text, _LARGO_DETALLE, secretos)}")
    return resp.json()


async def _llamar(clave: str, d: Despacho, timeout: int) -> ResultadoSonda:
    limite = min(prevuelo_config.sonda_max_tokens(), d.max_output_tokens)
    mensajes = [{"role": "user", "content": MENSAJE_DE_SONDA}]

    if d.via_motor:
        # El MISMO cliente que worker.py usa para despachar el motor.
        from motor_registry import worker
        api_key = "" if d.transporte == "ollama" else await resolve_credential_instrumented(d.provider_id)
        campo = "max_tokens" if d.transporte == "ollama" else d.max_tokens_param
        try:
            data = await worker._call_http_openai_compat(
                api_url=d.base_url, model=d.modelo, api_key=api_key,
                messages=mensajes, timeout=timeout, limite={campo: limite},
            )
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"HTTP {exc.response.status_code}: "
                f"{recortar_redactado(exc.response.text, _LARGO_DETALLE, [api_key])}"
            ) from exc
        uso = data.get("usage") or {}
        return ResultadoSonda(True, None, uso.get("prompt_tokens", 0), uso.get("completion_tokens", 0))

    f = await resolve_facet(clave)
    if f.transport == "http_gemini":
        data = await _post(
            f"{f.base_url}/models/{f.model}:generateContent",
            {"x-goog-api-key": f.credential},
            {"contents": [{"role": "user", "parts": [{"text": MENSAJE_DE_SONDA}]}],
             "generationConfig": {"maxOutputTokens": limite}},
            timeout, [f.credential],
        )
        uso = data.get("usageMetadata") or {}
        return ResultadoSonda(True, None, uso.get("promptTokenCount", 0), uso.get("candidatesTokenCount", 0))
    if f.transport == "http_openai_compat":
        data = await _post(
            f"{f.base_url}/chat/completions",
            {"Authorization": f"Bearer {f.credential}", "Content-Type": "application/json"},
            {"model": f.model, "messages": mensajes, "stream": False, d.max_tokens_param: limite},
            timeout, [f.credential],
        )
        uso = data.get("usage") or {}
        return ResultadoSonda(True, None, uso.get("prompt_tokens", 0), uso.get("completion_tokens", 0))
    if f.transport == "ollama":
        from jacobs.executor import OLLAMA_URL
        data = await _post(
            OLLAMA_URL, {},
            {"model": f.model, "messages": mensajes, "stream": False, "options": {"num_predict": limite}},
            timeout, [],
        )
        return ResultadoSonda(True, None, data.get("prompt_eval_count", 0), data.get("eval_count", 0))
    raise RuntimeError(f"transporte '{f.transport}' de '{clave}' no tiene sonda")


async def _registrar(clave: str, d: Despacho, r: ResultadoSonda,
                     user_id: str | None, tenant_id: str | None) -> None:
    global _registros_perdidos
    outcome = "ok" if r.ok else "provider_error"
    try:
        await facet_health.registrar_evento_de_sonda(clave, outcome, r.detalle, time.time())
    except Exception as exc:  # fail-soft: el veredicto sale de la respuesta del proveedor, no de esta fila; sin ella el próximo pre-vuelo vuelve a sondear, y la pérdida queda contada en registros_perdidos() y en el WARNING
        _registros_perdidos += 1
        logger.warning(
            "pre-vuelo: no se pudo registrar la sonda de '%s' en facet_health_event (%s); "
            "van %d registros perdidos",
            clave, recortar_redactado(f"{type(exc).__name__}: {exc}", _LARGO_DETALLE), _registros_perdidos,
        )
    if r.tokens_in or r.tokens_out:
        await record_direct_usage(
            user_id, tenant_id, clave, d.provider_id, d.modelo, r.tokens_in, r.tokens_out,
            request_type=REQUEST_TYPE_SONDA,
        )


async def sondear(clave: str, d: Despacho, *, user_id: str | None = None,
                  tenant_id: str | None = None) -> ResultadoSonda:
    timeout = prevuelo_config.sonda_timeout_s()
    try:
        resultado = await asyncio.wait_for(_llamar(clave, d, timeout), timeout=timeout)
    except (asyncio.TimeoutError, httpx.TimeoutException):
        resultado = ResultadoSonda(False, f"timeout de sonda ({timeout}s)")
    except Exception as exc:  # fail-closed: una sonda que no completó la llamada da faceta_caida, nunca sana; el motivo redactado viaja en el veredicto
        resultado = ResultadoSonda(False, recortar_redactado(f"{type(exc).__name__}: {exc}", _LARGO_DETALLE))
    await _registrar(clave, d, resultado, user_id, tenant_id)
    return resultado
```

- [ ] **Step 5: Verlos pasar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_prevuelo_sonda.py tests/test_cola_uso_escritores.py tests/test_payload_max_tokens_literal_tripwire.py tests/test_no_blocking_in_async.py policy/tests/test_no_fail_open_except.py 2>&1 | tail -3`
Expected: `0 failed`; `tests/test_prevuelo_sonda.py` aporta `12 passed`.

Run DB (el escritor de uso sigue sano contra el esquema real): `cd /home/fruiz/worktrees/jax-prevuelo && set -a && source /etc/jax/.env && set +a && export JAX_DB_NAME=jax_memory_test PYTHONPATH=.:las_manos && /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q jacobs/_direct_usage_test.py 2>&1 | tail -2`
Expected: `7 passed`.

- [ ] **Step 6: CI y commit**

`tests-puros`: agregar `tests/test_prevuelo_sonda.py` a las dos listas; reemplazar la línea de 786 por:
```
          # 786 -> 798 el 2026-09-17: test_prevuelo_sonda.py (+12) -- 2xx cortado responde,
          #   límite mínimo, gemini sin herramientas, 5xx redactado, timeout, motor con el
          #   cliente del worker, evento preflight, registro perdido contado, uso preflight_probe.
          grep -qE "^798 passed, 1 skipped" /tmp/out || {
            echo "PISO ROTO: se esperaban 798 tests CORRIDOS y exactamente 1 skipped."; exit 1; }
```

```bash
cd /home/fruiz/worktrees/jax-prevuelo && pwd && git branch --show-current && \
git -C /home/fruiz/worktrees/jax-prevuelo add jacobs/sonda.py jacobs/usage_writer.py tests/test_prevuelo_sonda.py .github/workflows/policy.yml && \
git -C /home/fruiz/worktrees/jax-prevuelo commit -m "feat(jacobs): sonda de disponibilidad del pre-vuelo

Llamada mínima por el transporte real, registrada en facet_health_event
(source=preflight) y en axioma_usage (request_type=preflight_probe).

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: El orquestador del pre-vuelo

**Files:**
- Create: `jacobs/prevuelo.py`
- Test: `tests/test_prevuelo_orquestador.py`
- Modify: `.github/workflows/policy.yml` (job `tests-puros`)

**Interfaces:**
- Consumes: Task 1 (`chars_por_token`), Task 5 (reglas y tipos), Task 6 (`resolver_motores`, `leer_catalogo`, `Catalogo`, `FilaModelo`, `MotorResuelto`, `salud_de_proveedor`), Task 7 (`sonda.sondear`, `ResultadoSonda`); `executor.MAX_DEP_CONTEXT_CHARS`, `executor._build_context_input`, `executor._enrich_prompt`; `motor_registry.output_validator.puede_pedir_reintento`; `motor_registry.worker.MAX_TOOL_LOOP_ITERATIONS`, `motor_registry.worker._REFORMAS_V3_PREDICATES`; `motor_registry.identity_context.build_identity_context`.
- Produces: `async prevuelo.prevuelo(pasos: list[Step], contexto: dict, *, pendientes: set[int] | None = None, user_id: str | None = None, tenant_id: str | None = None) -> Veredicto`. `pendientes=None` evalúa todos. Un error de DB se PROPAGA (quien llama responde 503). Hooks de test: `prevuelo._ahora()`, `prevuelo._max_iteraciones()`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_prevuelo_orquestador.py`:
```python
"""Orquestador del pre-vuelo (spec 2026-09-17 §4; desvíos 1, 3, 4 y 16 del
plan). Catálogo y sonda mockeados; el armado del prompt es el REAL del ejecutor.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import json
import os
from decimal import Decimal
from unittest.mock import AsyncMock

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import pytest  # noqa: E402

from jacobs import prevuelo as pv  # noqa: E402
from jacobs.executor import MAX_DEP_CONTEXT_CHARS  # noqa: E402
from jacobs.models import Step  # noqa: E402
from jacobs.prevuelo_catalogo import Catalogo, FilaFaceta, FilaModelo, MotorResuelto  # noqa: E402
from jacobs.sonda import ResultadoSonda  # noqa: E402

AHORA = 1_000_000.0


def _catalogo(salud=None, credencial=True):
    return Catalogo(
        facetas={
            "jekyll": FilaFaceta("jekyll", "http_openai_compat", None, "deepseek",
                                 "https://api.deepseek.example/v1", "deepseek-v4-flash"),
            "hipatia": FilaFaceta("hipatia", "http_gemini", None, "gemini",
                                  "https://g.example/v1beta", "gemini-x"),
        },
        modelos={
            ("deepseek", "deepseek-v4-flash"): FilaModelo("max_tokens", 8192, Decimal("0.27"), Decimal("1.10")),
            ("gemini", "gemini-x"): FilaModelo(None, 65536, Decimal("0.30"), Decimal("2.50")),
            ("moonshot", "kimi-k3"): FilaModelo("max_tokens", 131072, Decimal("0.60"), Decimal("2.50")),
        },
        min_output_tokens={},
        proveedores_con_credencial=frozenset({"deepseek", "gemini", "moonshot"}) if credencial else frozenset(),
        salud=salud if salud is not None else {
            "jekyll": (AHORA - 60, "ok"), "hipatia": (AHORA - 60, "ok"), "kimi": (AHORA - 60, "ok")},
    )


def _paso(i, facet, capability="research", deps=None, prompt="investigá"):
    return Step(pipeline_id="p", step_index=i, facet=facet, capability=capability,
                input={"prompt": prompt}, depends_on=deps or [])


def _instalar(monkeypatch, catalogo, motores=None, sondear=None):
    leer = AsyncMock(return_value=catalogo)
    monkeypatch.setattr(pv.prevuelo_catalogo, "leer_catalogo", leer)
    monkeypatch.setattr(pv.prevuelo_catalogo, "resolver_motores", AsyncMock(return_value=motores or {}))
    sonda = sondear or AsyncMock(return_value=ResultadoSonda(True, None))
    monkeypatch.setattr(pv.sonda, "sondear", sonda)
    monkeypatch.setattr(pv, "_ahora", lambda: AHORA)
    monkeypatch.setattr(pv, "_max_iteraciones", lambda: 5)
    monkeypatch.setenv("JAX_PREVUELO_CHARS_POR_TOKEN", "2")
    return leer, sonda


def _correr(pasos, contexto=None, **kw):
    return asyncio.run(pv.prevuelo(pasos, contexto or {"objective": "o"}, **kw))


def test_crear_evalua_todos_los_pasos_y_suma_el_costo(monkeypatch):
    _instalar(monkeypatch, _catalogo())
    v = _correr([_paso(0, "jekyll"), _paso(1, "hipatia", deps=[0])])
    assert v.ok
    assert [c.paso for c in v.pasos_costo] == [0, 1]
    assert v.costo_max_usd == sum(c.usd_max for c in v.pasos_costo)


def test_continuar_solo_evalua_los_pendientes(monkeypatch):
    _instalar(monkeypatch, _catalogo())
    v = _correr([_paso(0, "jekyll"), _paso(1, "hipatia", deps=[0])], pendientes={1})
    assert [c.paso for c in v.pasos_costo] == [1]


def test_salud_fresca_no_sondea(monkeypatch):
    _, sonda = _instalar(monkeypatch, _catalogo())
    v = _correr([_paso(0, "jekyll")])
    sonda.assert_not_awaited()
    assert v.sondeadas == ()


def test_sin_evento_sondea_y_si_responde_pasa(monkeypatch):
    _, sonda = _instalar(monkeypatch, _catalogo(salud={}))
    v = _correr([_paso(0, "jekyll")])
    assert sonda.await_args.args[0] == "jekyll"
    assert v.ok and v.sondeadas == ("jekyll",)


def test_sonda_que_no_responde_bloquea_con_faceta_caida(monkeypatch):
    caida = AsyncMock(return_value=ResultadoSonda(False, "timeout de sonda (20s)"))
    _instalar(monkeypatch, _catalogo(salud={}), sondear=caida)
    v = _correr([_paso(0, "jekyll")])
    assert not v.ok
    assert [(x.regla, x.detalle) for x in v.violaciones] == [("faceta_caida", "timeout de sonda (20s)")]


def test_una_sola_sonda_por_clave_aunque_haya_varios_pasos(monkeypatch):
    _, sonda = _instalar(monkeypatch, _catalogo(salud={}))
    _correr([_paso(0, "jekyll"), _paso(1, "jekyll", deps=[0])])
    assert sonda.await_count == 1


def test_las_sondas_corren_en_paralelo(monkeypatch):
    estado = {"en_curso": 0, "max": 0}

    async def sondear(clave, d, **kw):
        estado["en_curso"] += 1
        estado["max"] = max(estado["max"], estado["en_curso"])
        await asyncio.sleep(0.01)
        estado["en_curso"] -= 1
        return ResultadoSonda(True, None)

    _instalar(monkeypatch, _catalogo(salud={}), sondear=sondear)
    _correr([_paso(0, "jekyll"), _paso(1, "hipatia")])
    assert estado["max"] == 2


def test_dependencia_sin_ref_cuenta_el_peor_caso(monkeypatch):
    _instalar(monkeypatch, _catalogo())
    v = _correr([_paso(0, "jekyll"), _paso(1, "jekyll", deps=[0])])
    assert v.pasos_costo[1].tokens_in_max >= MAX_DEP_CONTEXT_CHARS // 2


def test_dependencia_con_ref_mide_lo_real(monkeypatch):
    _instalar(monkeypatch, _catalogo())
    contexto = {"objective": "o", "step_0_ref": "inline:" + json.dumps({"result": "corto"})}
    v = _correr([_paso(0, "jekyll"), _paso(1, "jekyll", deps=[0])], contexto, pendientes={1})
    assert v.pasos_costo[0].tokens_in_max < 2000


def test_solo_hyde_no_lee_el_catalogo(monkeypatch):
    leer, _ = _instalar(monkeypatch, _catalogo())
    v = _correr([_paso(0, "hyde", capability="execute")])
    leer.assert_not_awaited()
    assert v.ok and v.pasos_costo[0].motivo == "suscripcion"


def test_un_error_de_la_base_se_propaga(monkeypatch):
    _instalar(monkeypatch, _catalogo())
    monkeypatch.setattr(pv.prevuelo_catalogo, "leer_catalogo", AsyncMock(side_effect=OSError("base caída")))
    with pytest.raises(OSError):
        _correr([_paso(0, "jekyll")])


def test_paso_de_motor_usa_el_motor_resuelto(monkeypatch):
    motor = MotorResuelto("kimi", "http_openai_compat", "moonshot", "https://api.moonshot.example/v1",
                          "kimi-k3", 8000, False, "")
    _, sonda = _instalar(monkeypatch, _catalogo(salud={}), motores={0: motor})
    v = _correr([_paso(0, "kimi", capability="generate")])
    clave, despacho = sonda.await_args.args
    assert clave == "kimi" and despacho.via_motor
    assert v.pasos_costo[0].tokens_out_max == 8000


def test_paso_con_violacion_no_se_sondea(monkeypatch):
    _, sonda = _instalar(monkeypatch, _catalogo(salud={}, credencial=False))
    v = _correr([_paso(0, "jekyll")])
    sonda.assert_not_awaited()
    assert [x.regla for x in v.violaciones] == ["credencial_ausente"]
```

- [ ] **Step 2: Verlos fallar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_prevuelo_orquestador.py 2>&1 | tail -4`
Expected: `1 error` de colección: `ImportError: cannot import name 'prevuelo' from 'jacobs'`.

- [ ] **Step 3: Implementar**

`jacobs/prevuelo.py`:
```python
"""Jacobs — pre-vuelo antes de gastar (spec 2026-09-17 §4).

Corre obligatoriamente al crear y al continuar (dentro del candado), y por su
propio endpoint (sin escribir). Para cada paso a evaluar:

1. arma el Despacho real: faceta HTTP con su binding, o el motor que
   resolvería MotorPolicy (desvío 1 del plan);
2. mide el prompt con las MISMAS funciones del ejecutor, rellenando cada
   dependencia que todavía no tiene ref con el peor caso (desvío 3);
3. aplica las reglas puras (prevuelo_reglas);
4. sondea en paralelo las claves sin un `ok` fresco, una vez por clave y solo
   si sus pasos no tienen otra violación (desvío 16).

Un error de la base se propaga: sin pre-vuelo no se crea (quien llama responde
503 prevuelo_no_disponible). Nada de esto es fail-open.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import asyncio
import json
import time

from motor_registry.output_validator import puede_pedir_reintento

from jacobs import prevuelo_catalogo, prevuelo_config, sonda
from jacobs.executor import MAX_DEP_CONTEXT_CHARS, _build_context_input, _enrich_prompt
from jacobs.facet_health import salud_de_proveedor
from jacobs.models import MOTOR_FACETS, Pipeline, Step
from jacobs.prevuelo_catalogo import Catalogo, FilaModelo, MotorResuelto
from jacobs.prevuelo_reglas import (
    CostoPaso,
    Despacho,
    Veredicto,
    Violacion,
    armar_veredicto,
    costo_sin_evaluar,
    evaluar_paso,
)

# Largo de un uuid real: el contexto de identidad del Motor Registry lo incluye.
_TASK_ID_DE_MEDIDA = "00000000-0000-0000-0000-000000000000"
# Una dependencia que todavía no corrió: el armado la recorta a
# MAX_DEP_CONTEXT_CHARS (o a 500 si el paso no declara depends_on).
_RELLENO_DE_DEPENDENCIA = "inline:" + json.dumps({"result": "x" * MAX_DEP_CONTEXT_CHARS})
_SIN_FILA = FilaModelo(None, None, None, None)
_SEPARADOR_DE_IDENTIDAD = "\n---\n"  # worker.run: identity + "\n---\n" + prompt


def _ahora() -> float:
    return time.time()


def _max_iteraciones() -> int:
    from motor_registry.worker import MAX_TOOL_LOOP_ITERATIONS
    return MAX_TOOL_LOOP_ITERATIONS


def _despacho(step: Step, catalogo: Catalogo, motor: MotorResuelto | None) -> tuple[Despacho | None, str]:
    if step.facet in MOTOR_FACETS:
        if motor is None:
            return None, (
                f"ningún motor habilitado despacha la capability '{step.capability}' "
                f"para la faceta '{step.facet}' (capability_motor / motor.status)"
            )
        fila = catalogo.modelos.get((motor.provider_id, motor.modelo), _SIN_FILA)
        return Despacho(
            clave_salud=motor.clave, via_motor=True, transporte=motor.transporte,
            provider_id=motor.provider_id, base_url=motor.base_url, modelo=motor.modelo,
            max_tokens_param=fila.max_tokens_param, max_output_tokens=fila.max_output_tokens,
            motor_max_tokens=motor.max_tokens, precio_in=fila.precio_in, precio_out=fila.precio_out,
            tiene_herramientas=motor.tiene_herramientas,
            schema_con_reintento=puede_pedir_reintento(motor.output_schema),
            persona=None,
        ), ""
    faceta = catalogo.facetas.get(step.facet)
    if faceta is None:
        return None, f"la faceta '{step.facet}' no tiene fila activa en `facet` con binding primario"
    fila = catalogo.modelos.get((faceta.provider_id, faceta.model_id), _SIN_FILA)
    return Despacho(
        clave_salud=faceta.key, via_motor=False, transporte=faceta.transport,
        provider_id=faceta.provider_id, base_url=faceta.base_url, modelo=faceta.model_id,
        max_tokens_param=fila.max_tokens_param, max_output_tokens=fila.max_output_tokens,
        motor_max_tokens=0, precio_in=fila.precio_in, precio_out=fila.precio_out,
        tiene_herramientas=False, schema_con_reintento=False, persona=faceta.persona,
    ), ""


def _chars_de_entrada(step: Step, pasos: list[Step], contexto: dict, d: Despacho) -> int:
    """SÍNCRONA (lee artifacts de disco): se llama por asyncio.to_thread."""
    peor = dict(contexto)
    for j in range(step.step_index):
        if not peor.get(f"step_{j}_ref"):
            peor[f"step_{j}_ref"] = _RELLENO_DE_DEPENDENCIA
    medida = Pipeline(name="prevuelo", invoked_by="plataforma", mode="autonomous",
                      plan=pasos, context=peor)
    chars = len(_enrich_prompt(_build_context_input(step, medida))) + len(d.persona or "")
    if d.via_motor:
        from motor_registry.identity_context import build_identity_context
        from motor_registry.worker import _REFORMAS_V3_PREDICATES
        chars += len(build_identity_context(
            motor_name=d.clave_salud, capabilities=[step.capability], catalog={},
            predicates=_REFORMAS_V3_PREDICATES, task_id=_TASK_ID_DE_MEDIDA,
        )) + len(_SEPARADOR_DE_IDENTIDAD)
    return chars


async def prevuelo(
    pasos: list[Step],
    contexto: dict,
    *,
    pendientes: set[int] | None = None,
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> Veredicto:
    por_indice = {s.step_index: s for s in pasos}
    indices = sorted(por_indice) if pendientes is None else sorted(pendientes)

    costos: list[CostoPaso] = []
    evaluables: list[Step] = []
    for i in indices:
        corto = costo_sin_evaluar(por_indice[i])
        if corto is None:
            evaluables.append(por_indice[i])
        else:
            costos.append(corto)
    if not evaluables:
        return armar_veredicto([], costos, [])

    ahora = _ahora()
    motores = await prevuelo_catalogo.resolver_motores([s for s in evaluables if s.facet in MOTOR_FACETS])
    catalogo = await prevuelo_catalogo.leer_catalogo(
        facetas={s.facet for s in evaluables if s.facet not in MOTOR_FACETS},
        motores=[m for m in motores.values() if m is not None],
        capabilities={s.capability for s in evaluables},
        ahora=ahora,
    )
    chars_por_token = prevuelo_config.chars_por_token()
    max_iteraciones = _max_iteraciones()

    violaciones: list[Violacion] = []
    despachables: dict[int, Despacho] = {}
    for step in evaluables:
        d, detalle = _despacho(step, catalogo, motores.get(step.step_index))
        chars = 0 if d is None else await asyncio.to_thread(_chars_de_entrada, step, pasos, contexto, d)
        propias, costo = evaluar_paso(
            step.step_index, step.facet, d,
            min_output_tokens=catalogo.min_output_tokens.get(step.capability, 0),
            credencial_activa=d is not None and d.provider_id in catalogo.proveedores_con_credencial,
            chars_entrada=chars, chars_por_token=chars_por_token,
            max_iteraciones=max_iteraciones, detalle_inexistente=detalle,
        )
        violaciones += propias
        costos.append(costo)
        if d is not None and not propias:
            despachables[step.step_index] = d

    a_sondear: dict[str, Despacho] = {}
    for d in despachables.values():
        if salud_de_proveedor(catalogo.salud.get(d.clave_salud), ahora) == "sondear":
            a_sondear.setdefault(d.clave_salud, d)
    claves = sorted(a_sondear)
    resultados = await asyncio.gather(*(
        sonda.sondear(c, a_sondear[c], user_id=user_id, tenant_id=tenant_id) for c in claves
    ))
    caidas = {c: r for c, r in zip(claves, resultados) if not r.ok}
    for i, d in sorted(despachables.items()):
        r = caidas.get(d.clave_salud)
        if r is not None:
            violaciones.append(Violacion(i, por_indice[i].facet, "faceta_caida",
                                         r.detalle or "la sonda no obtuvo respuesta"))
    return armar_veredicto(violaciones, costos, claves)
```

- [ ] **Step 4: Verlos pasar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_prevuelo_orquestador.py tests/test_no_blocking_in_async.py 2>&1 | tail -2`
Expected: `0 failed`; `tests/test_prevuelo_orquestador.py` aporta `13 passed`.

- [ ] **Step 5: CI y commit**

`tests-puros`: agregar `tests/test_prevuelo_orquestador.py` a las dos listas; reemplazar la línea de 798 por:
```
          # 798 -> 811 el 2026-09-17: test_prevuelo_orquestador.py (+13) -- pendientes,
          #   sonda solo sin ok fresco, una por clave y en paralelo, peor caso de dependencias
          #   con el armado real, hyde sin catálogo, error de base propagado, motor resuelto.
          grep -qE "^811 passed, 1 skipped" /tmp/out || {
            echo "PISO ROTO: se esperaban 811 tests CORRIDOS y exactamente 1 skipped."; exit 1; }
```

```bash
cd /home/fruiz/worktrees/jax-prevuelo && pwd && git branch --show-current && \
git -C /home/fruiz/worktrees/jax-prevuelo add jacobs/prevuelo.py tests/test_prevuelo_orquestador.py .github/workflows/policy.yml && \
git -C /home/fruiz/worktrees/jax-prevuelo commit -m "feat(jacobs): orquestador del pre-vuelo

Arma el despacho real de cada paso, mide el prompt con el armado del
ejecutor, aplica las reglas y sondea en paralelo lo que no tiene un ok fresco.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: `POST /jacobs/preflight` y el pre-vuelo al crear

**Files:**
- Modify: `jacobs/models.py:108-131` (`PipelineCreateRequest`)
- Modify: `jacobs/routes.py` (imports, `PreflightRequest`, `_prevuelo_o_503`, `preflight`, `create_pipeline`)
- Test: `tests/test_jacobs_preflight_endpoint.py`
- Modify: `.github/workflows/policy.yml` (job `tests-puros`)

**Interfaces:**
- Consumes: `prevuelo.prevuelo` (Task 8), `Veredicto.to_dict()` (Task 5).
- Produces:
  - `PipelineCreateRequest.costo_max_aceptado_usd: Decimal | None = None` (negativo → `ValidationError`)
  - `routes.PreflightRequest(invoked_by: str, steps: list[StepSpec] (min 1), objective: str = "", user_id: str | None = None, tenant_id: str | None = None)`
  - `POST /jacobs/preflight` → 200 `Veredicto.to_dict()`; 403 invocador; 422 plan rechazado o más de 20 pasos; 503 `{"code": "prevuelo_no_disponible", "motivo"}`. No escribe.
  - `POST /jacobs/pipeline`: 422 `{"code": "prevuelo_rechazado", **veredicto}` + evento `PREVUELO_RECHAZADO`; 409 `{"code": "costo_supera_lo_aceptado", "costo_max_aceptado_usd", **veredicto}`; 503 como arriba; OK/dry_run suman `costo_max_usd` (texto) y `pasos_costo`, también en `PIPELINE_CREATED`.
  - `routes._prevuelo_o_503(steps, contexto, *, pendientes=None, user_id=None, tenant_id=None) -> Veredicto`

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_jacobs_preflight_endpoint.py`:
```python
"""POST /jacobs/preflight y el pre-vuelo obligatorio al crear (spec 2026-09-17
§4.7, §6.1, §8). Sin DB: el planificador, el pre-vuelo y el store van mockeados.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
from contextlib import ExitStack
from decimal import Decimal
from unittest.mock import AsyncMock, patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import pytest  # noqa: E402
from fastapi import BackgroundTasks, HTTPException  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from jacobs import policy, routes  # noqa: E402
from jacobs.models import PipelineCreateRequest, Step, StepSpec  # noqa: E402
from jacobs.plan import PlanRejected, PlanViolation  # noqa: E402
from jacobs.prevuelo_reglas import CostoPaso, Veredicto, Violacion  # noqa: E402


def _veredicto(ok=True, usd="0.500000"):
    costo = CostoPaso(0, "jekyll", "deepseek-v4-flash", 1, 100, 8192, Decimal(usd), "acotado")
    violaciones = () if ok else (Violacion(0, "jekyll", "credencial_ausente", "sin credencial"),)
    return Veredicto(ok=ok, violaciones=violaciones, costo_max_usd=Decimal(usd),
                     pasos_costo=(costo,), sondeadas=())


def _pasos():
    return [Step(step_index=0, facet="jekyll", capability="research", input={"prompt": "p"})]


def _spec():
    return [StepSpec(facet="jekyll", capability="research", prompt="p")]


def _parches(pila, veredicto=None, prevuelo=None, build=None):
    m = {}
    for nombre in ("pipeline_create", "step_upsert", "event_append", "pipeline_update_status"):
        m[nombre] = pila.enter_context(patch.object(routes.store, nombre, AsyncMock()))
    m["pipeline_count_active"] = pila.enter_context(
        patch.object(routes.store, "pipeline_count_active", AsyncMock(return_value=0)))
    m["build"] = pila.enter_context(patch.object(
        routes._plan_builder, "build", build or AsyncMock(return_value=_pasos())))
    m["prevuelo"] = pila.enter_context(patch.object(
        routes, "prevuelo", prevuelo or AsyncMock(return_value=veredicto or _veredicto()), create=True))
    pila.enter_context(patch.object(policy, "check_kill_switch", return_value=False))
    return m


def _crear(**cambios):
    datos = dict(name="t", objective="o", invoked_by="plataforma", mode="autonomous", steps=_spec())
    datos.update(cambios)
    return PipelineCreateRequest(**datos)


def test_preflight_devuelve_200_con_el_veredicto_aunque_rechace():
    with ExitStack() as pila:
        _parches(pila, veredicto=_veredicto(ok=False))
        r = asyncio.run(routes.preflight(routes.PreflightRequest(invoked_by="plataforma", steps=_spec())))
    assert r["ok"] is False
    assert r["violaciones"][0]["regla"] == "credencial_ausente"


def test_preflight_rechaza_un_invocador_no_valido_con_403():
    with ExitStack() as pila:
        m = _parches(pila)
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.preflight(routes.PreflightRequest(invoked_by="Fernando", steps=_spec())))
    assert e.value.status_code == 403
    m["prevuelo"].assert_not_awaited()


def test_preflight_no_escribe_nada_ni_con_plan_rechazado():
    with ExitStack() as pila:
        m = _parches(pila)
        asyncio.run(routes.preflight(routes.PreflightRequest(invoked_by="plataforma", steps=_spec())))
        for nombre in ("pipeline_create", "step_upsert", "event_append", "pipeline_update_status"):
            m[nombre].assert_not_awaited()
    rechazo = PlanRejected([PlanViolation(0, "jekyll", None, "research", "no va")])
    with ExitStack() as pila:
        m = _parches(pila, build=AsyncMock(side_effect=rechazo))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.preflight(routes.PreflightRequest(invoked_by="plataforma", steps=_spec())))
        m["event_append"].assert_not_awaited()
    assert e.value.status_code == 422


def test_preflight_con_la_base_caida_da_503():
    with ExitStack() as pila:
        _parches(pila, prevuelo=AsyncMock(side_effect=OSError("base caída")))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.preflight(routes.PreflightRequest(invoked_by="plataforma", steps=_spec())))
    assert e.value.status_code == 503
    assert e.value.detail["code"] == "prevuelo_no_disponible"


def test_crear_con_prevuelo_rechazado_da_422_sin_crear_y_deja_evento():
    with ExitStack() as pila:
        m = _parches(pila, veredicto=_veredicto(ok=False))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.create_pipeline(_crear(), BackgroundTasks()))
        m["pipeline_create"].assert_not_awaited()
        tipos = [c.args[1] for c in m["event_append"].await_args_list]
    assert e.value.status_code == 422
    assert e.value.detail["code"] == "prevuelo_rechazado"
    assert tipos == ["PREVUELO_RECHAZADO"]


def test_crear_con_costo_mayor_al_aceptado_da_409_sin_crear():
    with ExitStack() as pila:
        m = _parches(pila, veredicto=_veredicto(usd="0.500000"))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.create_pipeline(
                _crear(costo_max_aceptado_usd=Decimal("0.40")), BackgroundTasks()))
        m["pipeline_create"].assert_not_awaited()
    assert e.value.status_code == 409
    assert e.value.detail["code"] == "costo_supera_lo_aceptado"
    assert e.value.detail["costo_max_usd"] == "0.500000"


def test_crear_ok_devuelve_el_costo_y_lo_deja_en_el_evento():
    bg = BackgroundTasks()
    with ExitStack() as pila:
        m = _parches(pila)
        r = asyncio.run(routes.create_pipeline(_crear(costo_max_aceptado_usd=Decimal("1")), bg))
        creado = [c.args[2] for c in m["event_append"].await_args_list if c.args[1] == "PIPELINE_CREATED"]
    assert r["costo_max_usd"] == "0.500000" and r["pasos_costo"][0]["paso"] == 0
    assert creado[0]["costo_max_usd"] == "0.500000"
    assert len(bg.tasks) == 1


def test_dry_run_tambien_corre_el_prevuelo():
    with ExitStack() as pila:
        m = _parches(pila)
        r = asyncio.run(routes.create_pipeline(_crear(mode="dry_run"), BackgroundTasks()))
        m["prevuelo"].assert_awaited_once()
    assert r["costo_max_usd"] == "0.500000"


def test_el_prevuelo_corre_despues_de_build_y_antes_de_crear():
    orden = []

    async def build(**kw):
        orden.append("build")
        return _pasos()

    async def prevuelo(*a, **kw):
        orden.append("prevuelo")
        return _veredicto()

    with ExitStack() as pila:
        m = _parches(pila, build=AsyncMock(side_effect=build), prevuelo=AsyncMock(side_effect=prevuelo))
        m["pipeline_create"].side_effect = lambda p: orden.append("crear")
        asyncio.run(routes.create_pipeline(_crear(), BackgroundTasks()))
    assert orden == ["build", "prevuelo", "crear"]


def test_crear_con_la_base_caida_da_503_sin_crear():
    with ExitStack() as pila:
        m = _parches(pila, prevuelo=AsyncMock(side_effect=OSError("base caída")))
        with pytest.raises(HTTPException) as e:
            asyncio.run(routes.create_pipeline(_crear(), BackgroundTasks()))
        m["pipeline_create"].assert_not_awaited()
    assert e.value.status_code == 503


def test_costo_max_aceptado_negativo_es_invalido():
    with pytest.raises(ValidationError):
        _crear(costo_max_aceptado_usd=Decimal("-1"))
```

- [ ] **Step 2: Verlos fallar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_jacobs_preflight_endpoint.py 2>&1 | tail -13`
Expected: `11 failed`: los cuatro de preflight con `AttributeError: module 'jacobs.routes' has no attribute 'PreflightRequest'`; los de crear con `Failed: DID NOT RAISE <class 'fastapi.exceptions.HTTPException'>` (422, 409, 503) o `KeyError: 'costo_max_usd'`; `test_el_prevuelo_corre_despues_de_build_y_antes_de_crear` con `AssertionError: assert ['build', 'crear'] == ['build', 'prevuelo', 'crear']`; `test_costo_max_aceptado_negativo_es_invalido` con `Failed: DID NOT RAISE <class 'pydantic_core._pydantic_core.ValidationError'>`.

- [ ] **Step 3: Implementar — modelo**

En `jacobs/models.py`:
1. Agregar `from decimal import Decimal` después de `import uuid`.
2. En `PipelineCreateRequest`, reemplazar
```python
    subpipeline_token: str | None = None
```
por
```python
    subpipeline_token: str | None = None
    # Spec 2026-09-17 §6.1: el costo que el humano confirmó en la Mesa. Si el
    # pre-vuelo interno da MÁS, Jacobs responde 409 costo_supera_lo_aceptado
    # sin crear: la condición la hace cumplir quien gasta.
    costo_max_aceptado_usd: Decimal | None = None
```
3. En `validate_fields`, antes de `return self`, agregar:
```python
        if self.costo_max_aceptado_usd is not None and self.costo_max_aceptado_usd < 0:
            raise ValueError("costo_max_aceptado_usd no puede ser negativo")
```

- [ ] **Step 4: Implementar — rutas**

En `jacobs/routes.py`:

1. Reemplazar
```python
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
```
por
```python
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from redaccion import recortar_redactado
```
2. Después de `from jacobs.plan import PlanBuilder, PlanRejected` agregar:
```python
from jacobs.prevuelo import prevuelo
from jacobs.prevuelo_reglas import Veredicto
```
3. Después de la función `_build_plan_or_reject` agregar:
```python
async def _prevuelo_o_503(
    steps: list[Step],
    contexto: dict,
    *,
    pendientes: set[int] | None = None,
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> Veredicto:
    """Spec 2026-09-17 §8: sin pre-vuelo no se crea ni se continúa. Un error
    (DB caída, catálogo ilegible) es 503 prevuelo_no_disponible, nunca un 500
    genérico ni un pase libre."""
    try:
        return await prevuelo(steps, contexto, pendientes=pendientes,
                              user_id=user_id, tenant_id=tenant_id)
    except Exception as exc:
        motivo = recortar_redactado(f"{type(exc).__name__}: {exc}", 300)
        logger.error("pre-vuelo no disponible: %s", motivo, exc_info=True)
        raise HTTPException(
            status_code=503, detail={"code": "prevuelo_no_disponible", "motivo": motivo},
        ) from exc


def _resumen_de_costo(veredicto: Veredicto) -> dict:
    return {
        "costo_max_usd": str(veredicto.costo_max_usd),
        "pasos_costo": [c.to_dict() for c in veredicto.pasos_costo],
    }


# ----------------------------------------------------------------
#  POST /jacobs/preflight  — pre-vuelo sin escribir
# ----------------------------------------------------------------

class PreflightRequest(BaseModel):
    invoked_by: str
    # Obligatorio y no vacío: sin pasos, build() planifica con un LLM pago y un
    # pre-vuelo no puede gastar (desvío 6 del plan 2026-09-17).
    steps:      list[StepSpec] = Field(min_length=1)
    objective:  str = ""
    user_id:    str | None = None
    tenant_id:  str | None = None


@router.post("/preflight")
async def preflight(req: PreflightRequest) -> dict:
    """Pre-vuelo de un plan explícito (spec 2026-09-17 §4.7). Responde 200 con
    el veredicto AUNQUE rechace. No crea filas, no escribe eventos, no toma el
    candado. La sonda sí registra su evento de salud y su uso: se paga.

    El camino por objetivo (planificación por LLM) NO pasa por acá: lo gastado
    en planificar ya está gastado; crear corre el pre-vuelo sobre el plan que
    devolvió el LLM."""
    if req.invoked_by not in VALID_INVOKERS:
        raise HTTPException(status_code=403, detail=f"invoked_by '{req.invoked_by}' no autorizado")
    if len(req.steps) > 20:
        raise HTTPException(status_code=422, detail=f"{len(req.steps)} pasos excede el límite duro (20)")
    steps_spec = [s.model_dump() for s in req.steps]
    try:
        steps = await _plan_builder.build(
            pipeline_id=str(uuid.uuid4()), objective=req.objective,
            max_steps=len(steps_spec), steps_spec=steps_spec,
        )
    except PlanRejected as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    veredicto = await _prevuelo_o_503(
        steps, {"objective": req.objective}, user_id=req.user_id, tenant_id=req.tenant_id,
    )
    return veredicto.to_dict()
```
4. En `create_pipeline`, reemplazar desde
```python
        # Asignar pipeline_id a cada step
        for step in steps:
            step.pipeline_id = pipeline_id
```
hasta el final de la función por:
```python
        # Asignar pipeline_id a cada step
        for step in steps:
            step.pipeline_id = pipeline_id

        # Pre-vuelo (spec 2026-09-17 §4.7): DESPUÉS de build() y ANTES de
        # crear filas, dentro del candado. dry_run también ("¿cuánto costaría?").
        veredicto = await _prevuelo_o_503(
            steps, {"objective": req.objective}, user_id=req.user_id, tenant_id=req.tenant_id,
        )
        if not veredicto.ok:
            cuerpo = {"code": "prevuelo_rechazado", **veredicto.to_dict()}
            await store.event_append(pipeline_id, "PREVUELO_RECHAZADO", cuerpo)
            raise HTTPException(status_code=422, detail=cuerpo)
        if (req.costo_max_aceptado_usd is not None
                and veredicto.costo_max_usd > req.costo_max_aceptado_usd):
            raise HTTPException(status_code=409, detail={
                "code": "costo_supera_lo_aceptado",
                "costo_max_aceptado_usd": str(req.costo_max_aceptado_usd),
                **veredicto.to_dict(),
            })
        costo = _resumen_de_costo(veredicto)

        now = time.time()
        pipeline = Pipeline(
            pipeline_id=pipeline_id,
            name=req.name,
            invoked_by=req.invoked_by,
            user_id=req.user_id,
            tenant_id=req.tenant_id,
            mode=req.mode,
            plan=steps,
            max_steps=req.max_steps,
            context={"objective": req.objective},
            created_at=now,
            updated_at=now,
        )

        await store.pipeline_create(pipeline)
        for step in steps:
            await store.step_upsert(step)
        await store.event_append(pipeline_id, "PIPELINE_CREATED", {
            "name": req.name, "mode": req.mode, "steps": len(steps), **costo,
        })

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
            **costo,
        }

    background.add_task(run_pipeline, pipeline)

    return {
        "pipeline_id": pipeline_id,
        "status": "running",
        "mode": req.mode,
        "step_count": len(steps),
        "message": "Pipeline iniciado. Consultar GET /jacobs/pipeline/{id}",
        **costo,
    }
```

Nota: el `except Exception` de `_prevuelo_o_503` relanza (`raise HTTPException ... from exc`), así que P10 no pide marca.

- [ ] **Step 5: Verlos pasar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_jacobs_preflight_endpoint.py tests/test_jacobs_invoked_by_rol.py tests/test_jacobs_resume_epoca.py policy/tests/test_no_fail_open_except.py 2>&1 | tail -2`
Expected: `0 failed`; `tests/test_jacobs_preflight_endpoint.py` aporta `11 passed`.

- [ ] **Step 6: CI y commit**

`tests-puros`: agregar `tests/test_jacobs_preflight_endpoint.py` a las dos listas; reemplazar la línea de 811 por:
```
          # 811 -> 822 el 2026-09-17: test_jacobs_preflight_endpoint.py (+11) -- /preflight
          #   200 sin escribir, 403, 503; crear corre el pre-vuelo entre build y crear, 422 con
          #   evento, 409 por costo aceptado, dry_run incluido. 11 vistos en rojo.
          grep -qE "^822 passed, 1 skipped" /tmp/out || {
            echo "PISO ROTO: se esperaban 822 tests CORRIDOS y exactamente 1 skipped."; exit 1; }
```

```bash
cd /home/fruiz/worktrees/jax-prevuelo && pwd && git branch --show-current && \
git -C /home/fruiz/worktrees/jax-prevuelo add jacobs/models.py jacobs/routes.py tests/test_jacobs_preflight_endpoint.py .github/workflows/policy.yml && \
git -C /home/fruiz/worktrees/jax-prevuelo commit -m "feat(jacobs): pre-vuelo obligatorio al crear y endpoint /preflight

Crear rechaza con 422 prevuelo_rechazado antes de gastar, 409 si el costo
supera lo aceptado y 503 si el pre-vuelo no puede correr.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Servicio de continuar

**Files:**
- Create: `jacobs/candado.py`
- Create: `jacobs/continuar.py`
- Modify: `jacobs/routes.py:45-56` (el candado pasa a `jacobs/candado.py`)
- Modify: `jacobs/store.py` (`continuar_transaccion` y sus SQL)
- Test: `tests/test_jacobs_continuar.py` (puro), `tests/test_jacobs_continuar_db.py` (DB)
- Modify: `.github/workflows/policy.yml` (jobs `tests-puros` y `jacobs-gobernanza-db`)

**Interfaces:**
- Consumes: Task 2 (`Pipeline.run_epoch`), Task 8 (`prevuelo.prevuelo`), `executor._load_ref`/`RefIlegible`, `plan._check_cleanroom`/`_validate_plan_capabilities`/`PlanRejected`, `policy.validate_resume`/`check_kill_switch`/`MAX_PARALLEL_PIPELINES`.
- Produces:
  - `candado.candado_de_creacion: asyncio.Lock` (y `routes._pipeline_create_lock` es el mismo objeto)
  - `store.continuar_transaccion(pipeline_id: str, epoca_leida: int, status_leido: PipelineStatus, pasos_a_correr: list[Step], plan: list[Step], context: dict, current_step_index: int) -> int | None` y las constantes `_SQL_BLOQUEAR_PIPELINE`, `_SQL_PASO_A_CORRER`, `_SQL_PIPELINE_CONTINUAR`
  - `continuar.ESTADOS_CONTINUABLES = frozenset({PipelineStatus.aborted, PipelineStatus.expired})`
  - `continuar.ContinuarRechazado(status_code: int, code: str, detalle: dict | list | str)` con `.cuerpo() -> dict` (`{"code", **detalle}` si es dict; si no `{"code", "detalle"}`)
  - `continuar.Analisis(pipeline, plan, contexto, pasos_a_correr, pasos_reusados, reasignados)`
  - `async continuar.analizar(pipeline_id: str, invoked_by: str, reasignar: dict[str, str] | None) -> Analisis` — no escribe
  - `async continuar.previsualizar(pipeline_id, invoked_by, reasignar=None, user_id=None, tenant_id=None) -> dict` → `{continuable, motivo, pasos_a_correr, pasos_reusados, veredicto}`; 403/404 se relanzan
  - `async continuar.continuar(pipeline_id, invoked_by, reasignar=None, user_id=None, tenant_id=None, costo_max_aceptado_usd: Decimal | None = None) -> tuple[dict, Pipeline]` — respuesta `{pipeline_id, status: "running", run_epoch, pasos_a_correr, pasos_reusados, costo_max_usd, pasos_costo}` y el `Pipeline` a lanzar. Códigos: 403 `invocador_no_autorizado`, 404 `no_existe`, 409 `estado_no_continuable` | `plan_inconsistente` | `costo_supera_lo_aceptado`, 422 `reasignacion_invalida` | `plan_rechazado` | `prevuelo_rechazado`, 423 `kill_switch`, 429 `limite_de_activos`. Evento `PIPELINE_CONTINUED {by, from_status, run_epoch, pasos_a_correr, pasos_reusados, reasignados, costo_max_usd}`.

- [ ] **Step 1: Escribir los tests puros que fallan**

`tests/test_jacobs_continuar.py`:
```python
"""Continuar un pipeline abortado o vencido (spec 2026-09-17 §5.1-§5.2;
desvíos 7, 10 y 11 del plan). Sin DB: store, pre-vuelo y gobernanza mockeados.
Las refs inline se leen de verdad; la ilegible es un artifact que no existe.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
from contextlib import ExitStack, contextmanager
from decimal import Decimal
from unittest.mock import AsyncMock, patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import pytest  # noqa: E402

from jacobs import continuar  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus  # noqa: E402
from jacobs.prevuelo_reglas import CostoPaso, Veredicto, Violacion  # noqa: E402

_NADA = object()
_REF_A = 'inline:{"result": "a"}'
_REF_B = 'inline:{"result": "b"}'
_ILEGIBLE = "artifact://jacobs/no-existe/tampoco/output.json"


def _pasos(capability_2="research", facet_1="jekyll"):
    pasos = []
    for i in range(3):
        pasos.append(Step(
            step_id=f"s{i}", pipeline_id="p1", step_index=i,
            facet=facet_1 if i == 1 else "jekyll",
            capability=capability_2 if i == 2 else "research",
            input={"prompt": f"p{i}"}, depends_on=[i - 1] if i else [],
            status=StepStatus.completed if i < 2 else StepStatus.failed,
            error=None if i < 2 else "cortado",
            started_at=1.0, finished_at=2.0,
            output_ref=_REF_A if i == 0 else (_REF_B if i == 1 else None),
        ))
    return pasos


def _abortado(status=PipelineStatus.aborted, contexto=None, epoca=2):
    return Pipeline(
        pipeline_id="p1", name="t", invoked_by="plataforma", user_id="7", tenant_id="1",
        mode="autonomous", status=status, run_epoch=epoca,
        context=contexto if contexto is not None else
        {"objective": "o", "step_0_ref": _REF_A, "step_1_ref": _REF_B},
    )


def _veredicto(ok=True, usd="0.250000"):
    costo = CostoPaso(2, "jekyll", "deepseek-v4-flash", 1, 100, 8192, Decimal(usd), "acotado")
    violaciones = () if ok else (Violacion(2, "jekyll", "faceta_caida", "timeout de sonda (20s)"),)
    return Veredicto(ok=ok, violaciones=violaciones, costo_max_usd=Decimal(usd),
                     pasos_costo=(costo,), sondeadas=())


@contextmanager
def _entorno(pipeline=_NADA, pasos=None, activos=0, veredicto=None, transaccion=3, kill=False):
    with ExitStack() as pila:
        m = {}
        m["get"] = pila.enter_context(patch.object(continuar.store, "pipeline_get", AsyncMock(
            return_value=_abortado() if pipeline is _NADA else pipeline)))
        m["pasos"] = pila.enter_context(patch.object(continuar.store, "steps_by_pipeline", AsyncMock(
            return_value=pasos if pasos is not None else _pasos())))
        m["activos"] = pila.enter_context(patch.object(
            continuar.store, "pipeline_count_active", AsyncMock(return_value=activos)))
        m["tx"] = pila.enter_context(patch.object(
            continuar.store, "continuar_transaccion", AsyncMock(return_value=transaccion)))
        m["evento"] = pila.enter_context(patch.object(continuar.store, "event_append", AsyncMock()))
        m["prevuelo"] = pila.enter_context(patch.object(
            continuar, "prevuelo", AsyncMock(return_value=veredicto or _veredicto())))
        m["capacidades"] = pila.enter_context(patch.object(
            continuar, "_validate_plan_capabilities", AsyncMock()))
        pila.enter_context(patch.object(continuar, "check_kill_switch", return_value=kill))
        yield m


def _continuar(**kw):
    return asyncio.run(continuar.continuar("p1", kw.pop("invoked_by", "plataforma"), **kw))


def _rechazo(**kw):
    with pytest.raises(continuar.ContinuarRechazado) as e:
        _continuar(**kw)
    return e.value


def test_un_abortado_se_continua_reusando_lo_que_tiene_ref():
    with _entorno() as m:
        r, p = _continuar()
        args = m["tx"].await_args.args
    assert r["pasos_reusados"] == [0, 1] and r["pasos_a_correr"] == [2]
    assert r["run_epoch"] == 3 and p.run_epoch == 3 and p.status == PipelineStatus.running
    assert args[:3] == ("p1", 2, PipelineStatus.aborted)
    assert [s.step_index for s in args[3]] == [2]


def test_un_vencido_tambien_se_continua():
    with _entorno(pipeline=_abortado(status=PipelineStatus.expired)) as m:
        r, _ = _continuar()
        assert m["tx"].await_args.args[2] == PipelineStatus.expired
    assert r["status"] == "running"


def test_los_demas_estados_dan_409_con_el_status():
    for status in (PipelineStatus.completed, PipelineStatus.running, PipelineStatus.pending,
                   PipelineStatus.interrupted, PipelineStatus.failed):
        with _entorno(pipeline=_abortado(status=status)) as m:
            e = _rechazo()
            m["tx"].assert_not_awaited()
        assert (e.status_code, e.code, e.cuerpo()["status"]) == (409, "estado_no_continuable", status.value)


def test_invocador_distinto_de_plataforma_403():
    with _entorno():
        e = _rechazo(invoked_by="ada")
    assert (e.status_code, e.code) == (403, "invocador_no_autorizado")


def test_pipeline_inexistente_404():
    with _entorno(pipeline=None):
        e = _rechazo()
    assert (e.status_code, e.code) == (404, "no_existe")


def test_kill_switch_423():
    with _entorno(kill=True) as m:
        e = _rechazo()
        m["tx"].assert_not_awaited()
    assert (e.status_code, e.code) == (423, "kill_switch")


def test_ref_ilegible_se_rehace_y_sale_del_contexto():
    contexto = {"objective": "o", "step_0_ref": _REF_A, "step_1_ref": _ILEGIBLE}
    with _entorno(pipeline=_abortado(contexto=contexto)) as m:
        r, _ = _continuar()
        ctx = m["tx"].await_args.args[5]
    assert r["pasos_reusados"] == [0] and r["pasos_a_correr"] == [1, 2]
    assert "step_1_ref" not in ctx and ctx["step_0_ref"] == _REF_A


def test_los_pasos_a_correr_quedan_pendientes_limpios():
    with _entorno() as m:
        _continuar()
        paso = m["tx"].await_args.args[3][0]
    assert (paso.status, paso.error, paso.started_at, paso.finished_at, paso.output_ref) == (
        StepStatus.pending, None, None, None, None)


def test_reasignar_cambia_la_faceta_y_recalcula_el_motor():
    with _entorno() as m:
        _continuar(reasignar={"2": "kimi"})
        paso = m["tx"].await_args.args[3][0]
        evento = [c.args[2] for c in m["evento"].await_args_list if c.args[1] == "PIPELINE_CONTINUED"][0]
    assert (paso.facet, paso.motor) == ("kimi", "kimi")
    assert evento["reasignados"] == {"2": {"de": "jekyll", "a": "kimi"}}
    with _entorno() as m:
        _continuar(reasignar={"2": "thot"})
        paso = m["tx"].await_args.args[3][0]
    assert (paso.facet, paso.motor) == ("thot", None)


def test_reasignar_un_paso_reusado_da_422():
    with _entorno() as m:
        e = _rechazo(reasignar={"0": "thot"})
        m["tx"].assert_not_awaited()
        m["prevuelo"].assert_not_awaited()
    assert (e.status_code, e.code) == (422, "reasignacion_invalida")


def test_reasignar_a_una_faceta_desconocida_da_422():
    with _entorno() as m:
        e = _rechazo(reasignar={"2": "gpt"})
        m["tx"].assert_not_awaited()
    assert (e.status_code, e.code) == (422, "reasignacion_invalida")
    assert "gpt" in e.cuerpo()["detalle"][0]["motivo"]


def test_reasignacion_que_rompe_el_cleanroom_da_422_sin_escribir():
    pasos = _pasos(capability_2="critique", facet_1="thot")
    with _entorno(pasos=pasos) as m:
        e = _rechazo(reasignar={"2": "thot"})
        m["tx"].assert_not_awaited()
        m["evento"].assert_not_awaited()
    assert (e.status_code, e.code) == (422, "reasignacion_invalida")
    assert "cleanroom" in e.cuerpo()["detalle"][0]["reason"]


def test_limite_de_activos_da_429():
    with _entorno(activos=3) as m:
        e = _rechazo()
        m["tx"].assert_not_awaited()
    assert (e.status_code, e.code) == (429, "limite_de_activos")


def test_prevuelo_rechazado_da_422_sin_escribir_el_pipeline():
    with _entorno(veredicto=_veredicto(ok=False)) as m:
        e = _rechazo()
        m["tx"].assert_not_awaited()
        tipos = [c.args[1] for c in m["evento"].await_args_list]
    assert (e.status_code, e.code) == (422, "prevuelo_rechazado")
    assert tipos == ["PREVUELO_RECHAZADO"]


def test_costo_mayor_al_aceptado_da_409_sin_escribir():
    with _entorno() as m:
        e = _rechazo(costo_max_aceptado_usd=Decimal("0.10"))
        m["tx"].assert_not_awaited()
    assert (e.status_code, e.code) == (409, "costo_supera_lo_aceptado")


def test_si_otro_pedido_gano_la_transaccion_da_409():
    with _entorno(transaccion=None) as m:
        e = _rechazo()
        tipos = [c.args[1] for c in m["evento"].await_args_list]
    assert (e.status_code, e.code) == (409, "estado_no_continuable")
    assert "PIPELINE_CONTINUED" not in tipos


def test_el_evento_continued_lleva_todo_el_detalle():
    with _entorno() as m:
        _continuar()
        evento = [c.args[2] for c in m["evento"].await_args_list if c.args[1] == "PIPELINE_CONTINUED"][0]
    assert evento == {"by": "plataforma", "from_status": "aborted", "run_epoch": 3,
                      "pasos_a_correr": [2], "pasos_reusados": [0, 1], "reasignados": {},
                      "costo_max_usd": "0.250000"}


def test_la_marca_de_hyde_del_paso_a_rehacer_se_quita():
    contexto = {"objective": "o", "step_0_ref": _REF_A, "step_1_ref": _REF_B,
                "hyde_approved_s0": True, "hyde_approved_s2": True}
    with _entorno(pipeline=_abortado(contexto=contexto)) as m:
        _continuar()
        ctx = m["tx"].await_args.args[5]
    assert "hyde_approved_s2" not in ctx and ctx["hyde_approved_s0"] is True


def test_el_prevuelo_solo_mira_los_pendientes():
    with _entorno() as m:
        _continuar()
        kw = m["prevuelo"].await_args.kwargs
    assert kw["pendientes"] == {2}
    assert (kw["user_id"], kw["tenant_id"]) == ("7", "1")


def test_previsualizar_no_escribe_y_explica_por_que_no():
    with _entorno() as m:
        r = asyncio.run(continuar.previsualizar("p1", "plataforma"))
        m["tx"].assert_not_awaited()
        m["evento"].assert_not_awaited()
    assert r["continuable"] is True and r["motivo"] is None and r["veredicto"]["ok"] is True
    with _entorno(pipeline=_abortado(status=PipelineStatus.completed)):
        r = asyncio.run(continuar.previsualizar("p1", "plataforma"))
    assert r["continuable"] is False and r["motivo"]["code"] == "estado_no_continuable"
    with _entorno(activos=3):
        r = asyncio.run(continuar.previsualizar("p1", "plataforma"))
    assert r["continuable"] is False and r["motivo"]["code"] == "limite_de_activos"
```

- [ ] **Step 2: Escribir los tests DB que fallan**

`tests/test_jacobs_continuar_db.py`:
```python
"""Transacción de continuar y continue concurrente contra MariaDB real
(spec 2026-09-17 §5.2 regla 10, §8, §9). Job jacobs-gobernanza-db.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, patch

_db = os.environ.get("JAX_DB_NAME")
if _db and _db != "jax_memory_test":
    raise RuntimeError(f"JAX_DB_NAME={_db!r}: este archivo escribe filas y solo corre contra jax_memory_test.")
os.environ.setdefault("JAX_DB_NAME", "jax_memory_test")

import pytest  # noqa: E402

from jacobs import continuar, store  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus  # noqa: E402
from jacobs.prevuelo_reglas import Veredicto  # noqa: E402

_REF = 'inline:{"result": "hecho"}'


async def _abortado():
    await store.init_tables()
    pid = str(uuid.uuid4())
    pasos = [
        Step(pipeline_id=pid, step_index=i, facet="jekyll", capability="research",
             input={"prompt": f"p{i}"}, depends_on=[i - 1] if i else [],
             status=StepStatus.completed if i < 2 else StepStatus.failed,
             error=None if i < 2 else "cortado", output_ref=_REF if i < 2 else None)
        for i in range(3)
    ]
    ahora = time.time()
    pipeline = Pipeline(pipeline_id=pid, name="t-continuar", invoked_by="plataforma",
                        mode="autonomous", status=PipelineStatus.aborted, plan=pasos,
                        context={"objective": "o", "step_0_ref": _REF, "step_1_ref": _REF},
                        created_at=ahora, updated_at=ahora)
    await store.pipeline_create(pipeline)
    for paso in pasos:
        await store.step_upsert(paso)
    return pipeline, pasos


async def _borrar(pid):
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM jacobs_steps WHERE pipeline_id=%s", (pid,))
            await cur.execute("DELETE FROM jacobs_events WHERE pipeline_id=%s", (pid,))
            await cur.execute("DELETE FROM jacobs_pipelines WHERE pipeline_id=%s", (pid,))
    finally:
        conn.close()


async def _explain(sql, params):
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("EXPLAIN " + sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in await cur.fetchall()]
    finally:
        conn.close()


def test_la_transaccion_aplica_pasos_plan_contexto_y_epoca():
    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            pasos[2].facet = "thot"
            contexto = {"objective": "o", "step_0_ref": _REF, "step_1_ref": _REF}
            nueva = await store.continuar_transaccion(
                pid, 0, PipelineStatus.aborted, [pasos[2]], pasos, contexto, 2)
            p = await store.pipeline_get(pid)
            s2 = (await store.steps_by_pipeline(pid))[2]
            explains = [
                await _explain(store._SQL_BLOQUEAR_PIPELINE, (pid,)),
                await _explain(store._SQL_PASO_A_CORRER, ("thot", None, s2.step_id, pid)),
            ]
            return nueva, p, s2, explains
        finally:
            await _borrar(pid)
    nueva, p, s2, explains = asyncio.run(cuerpo())
    assert nueva == 1
    assert (p.status, p.run_epoch, p.current_step_index, p.plan[2].facet) == (
        PipelineStatus.running, 1, 2, "thot")
    assert (s2.status, s2.facet, s2.error, s2.output_ref) == (StepStatus.pending, "thot", None, None)
    for filas in explains:
        assert all(f["key"] == "PRIMARY" for f in filas), filas


def test_si_falla_a_mitad_no_cambia_nada(monkeypatch):
    monkeypatch.setattr(store, "_SQL_PIPELINE_CONTINUAR",
                        "UPDATE tabla_que_no_existe SET plan=%s, context_refs=%s, "
                        "current_step_index=%s, updated_at=%s WHERE pipeline_id=%s AND run_epoch=%s")

    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            pasos[2].facet = "thot"
            with pytest.raises(Exception):
                await store.continuar_transaccion(
                    pid, 0, PipelineStatus.aborted, [pasos[2]], pasos, pipeline.context, 2)
            return await store.pipeline_get(pid), (await store.steps_by_pipeline(pid))[2]
        finally:
            await _borrar(pid)
    p, s2 = asyncio.run(cuerpo())
    assert (p.status, p.run_epoch) == (PipelineStatus.aborted, 0)
    assert (s2.status, s2.facet, s2.error) == (StepStatus.failed, "jekyll", "cortado")


def test_dos_transacciones_con_la_misma_epoca_solo_una_gana():
    async def cuerpo():
        pipeline, pasos = await _abortado()
        pid = pipeline.pipeline_id
        try:
            return await asyncio.gather(*(
                store.continuar_transaccion(pid, 0, PipelineStatus.aborted, [pasos[2]], pasos,
                                            pipeline.context, 2)
                for _ in range(2)
            ))
        finally:
            await _borrar(pid)
    resultados = asyncio.run(cuerpo())
    assert sorted(resultados, key=lambda r: r is None) == [1, None]


def test_continue_concurrente_sobre_el_mismo_pipeline_solo_uno_gana():
    ok = Veredicto(ok=True, violaciones=(), costo_max_usd=Decimal(0), pasos_costo=(), sondeadas=())

    async def cuerpo():
        pipeline, _ = await _abortado()
        pid = pipeline.pipeline_id
        try:
            with patch.object(continuar, "prevuelo", AsyncMock(return_value=ok)), \
                 patch.object(continuar, "check_kill_switch", return_value=False), \
                 patch.object(continuar.store, "pipeline_count_active", AsyncMock(return_value=0)):
                resultados = await asyncio.gather(
                    *(continuar.continuar(pid, "plataforma") for _ in range(10)),
                    return_exceptions=True)
            return resultados, await store.pipeline_epoca_y_status(pid)
        finally:
            await _borrar(pid)
    resultados, actual = asyncio.run(cuerpo())
    ganadores = [r for r in resultados if isinstance(r, tuple)]
    rechazos = [r for r in resultados if isinstance(r, continuar.ContinuarRechazado)]
    assert len(ganadores) == 1 and len(rechazos) == 9, resultados
    assert all(r.status_code == 409 for r in rechazos)
    assert actual == (1, PipelineStatus.running)
```

- [ ] **Step 3: Verlos fallar**

Run (puros): `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_jacobs_continuar.py 2>&1 | tail -4`
Expected: `1 error` de colección: `ImportError: cannot import name 'continuar' from 'jacobs'`.

Run (DB): `cd /home/fruiz/worktrees/jax-prevuelo && set -a && source /etc/jax/.env && set +a && export JAX_DB_NAME=jax_memory_test PYTHONPATH=.:las_manos && /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_jacobs_continuar_db.py 2>&1 | tail -4`
Expected: `1 error` de colección, la misma causa.

- [ ] **Step 4: Implementar — candado**

`jacobs/candado.py`:
```python
"""Candado de proceso para crear y continuar pipelines (spec 2026-09-17 §5.2
regla 4). Vivía en routes.py como _pipeline_create_lock; se muda acá para que
continuar.py (endpoint y CLI) tome el MISMO objeto sin importar routes.

Su justificación sigue siendo la de T2 (2026-08-19, ver routes.py): Jacobs
corre en un solo proceso uvicorn y un asyncio.Lock no reserva conexión DB
mientras build() llama a un LLM. Límite declarado (desvío 7 del plan): no
cruza procesos. Entre LAS MANOS y el CLI, lo que serializa un continue es la
transacción con SELECT ... FOR UPDATE y la época.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import asyncio

candado_de_creacion = asyncio.Lock()
```

En `jacobs/routes.py` reemplazar la línea
```python
_pipeline_create_lock = asyncio.Lock()
```
por
```python
from jacobs.candado import candado_de_creacion as _pipeline_create_lock  # noqa: E402  (2026-09-17: compartido con continuar.py)
```

- [ ] **Step 5: Implementar — transacción en el store**

En `jacobs/store.py`, después de `pipeline_tomar_epoca`, agregar:
```python
_SQL_BLOQUEAR_PIPELINE = "SELECT run_epoch, status FROM jacobs_pipelines WHERE pipeline_id=%s FOR UPDATE"
_SQL_PASO_A_CORRER = (
    "UPDATE jacobs_steps SET facet=%s, motor=%s, status='pending', output_ref=NULL, "
    "started_at=NULL, finished_at=NULL, error=NULL WHERE step_id=%s AND pipeline_id=%s"
)
_SQL_PIPELINE_CONTINUAR = (
    "UPDATE jacobs_pipelines SET status='running', run_epoch=run_epoch+1, plan=%s, "
    "context_refs=%s, current_step_index=%s, updated_at=%s "
    "WHERE pipeline_id=%s AND run_epoch=%s"
)


async def continuar_transaccion(
    pipeline_id: str,
    epoca_leida: int,
    status_leido: PipelineStatus,
    pasos_a_correr: list[Step],
    plan: list[Step],
    context: dict,
    current_step_index: int,
) -> int | None:
    """Escrituras de continue en UNA transacción (spec 2026-09-17 §5.2 regla
    10): bloquea la fila del pipeline, confirma que nadie la cambió desde el
    análisis (misma época y mismo status), resetea los pasos a correr, reescribe
    plan y contexto, pone running e incrementa la época. Devuelve la época
    nueva, o None si otro pedido ganó. Un error a mitad hace ROLLBACK: nada
    cambia."""
    conn = await get_conn(found_rows=True)
    try:
        await conn.begin()
        try:
            async with conn.cursor() as cur:
                await cur.execute(_SQL_BLOQUEAR_PIPELINE, (pipeline_id,))
                fila = await cur.fetchone()
                if fila is None or int(fila[0]) != epoca_leida or fila[1] != status_leido.value:
                    await conn.rollback()
                    return None
                for paso in pasos_a_correr:
                    await cur.execute(_SQL_PASO_A_CORRER, (paso.facet, paso.motor, paso.step_id, pipeline_id))
                await cur.execute(_SQL_PIPELINE_CONTINUAR, (
                    json.dumps([s.model_dump() for s in plan], ensure_ascii=False),
                    json.dumps(context, ensure_ascii=False),
                    current_step_index, time.time(), pipeline_id, epoca_leida,
                ))
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise
    finally:
        conn.close()
    return epoca_leida + 1
```

- [ ] **Step 6: Implementar — el servicio**

`jacobs/continuar.py`:
```python
"""Jacobs — continuar un pipeline abortado o vencido (spec 2026-09-17 §5).

UNA función de servicio para el endpoint (POST /jacobs/pipeline/{id}/continue
y /continue/preflight) y para el CLI (tools/jacobs_relaunch.py): ninguno tiene
lógica propia.

Reglas (§5.2): sólo `plataforma` (el dueño lo verifica jax-platform); estados
aborted (cualquier causa, D2) o expired; kill switch 423; dentro del candado y
contra MAX_PARALLEL_PIPELINES; se reusan los pasos con ref LEGIBLE y se
rehacen todos los demás; reasignar solo pasos a correr, con clean-room y
gobernanza sobre el plan completo; pre-vuelo sobre los pendientes; escrituras
en una transacción que incrementa la época.

Lo de antes de la transacción NO escribe filas de estado. El único registro
previo es el evento PREVUELO_RECHAZADO (auditoría, igual que al crear).

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from decimal import Decimal

from jacobs import store
from jacobs.candado import candado_de_creacion
from jacobs.executor import RefIlegible, _load_ref
from jacobs.models import MOTOR_FACETS, VALID_FACETS, Pipeline, PipelineStatus, Step, StepStatus
from jacobs.plan import PlanRejected, _check_cleanroom, _validate_plan_capabilities
from jacobs.policy import MAX_PARALLEL_PIPELINES, check_kill_switch, validate_resume
from jacobs.prevuelo import prevuelo

logger = logging.getLogger("jacobs.continuar")

ESTADOS_CONTINUABLES = frozenset({PipelineStatus.aborted, PipelineStatus.expired})


class ContinuarRechazado(Exception):
    def __init__(self, status_code: int, code: str, detalle):
        super().__init__(f"{status_code} {code}: {detalle}")
        self.status_code = status_code
        self.code = code
        self.detalle = detalle

    def cuerpo(self) -> dict:
        if isinstance(self.detalle, dict):
            return {"code": self.code, **self.detalle}
        return {"code": self.code, "detalle": self.detalle}


@dataclass
class Analisis:
    pipeline: Pipeline
    plan: list[Step]
    contexto: dict
    pasos_a_correr: list[int]
    pasos_reusados: list[int]
    reasignados: dict[str, dict] = field(default_factory=dict)


def _ref_legible(ref: str) -> bool:
    """SÍNCRONA (lee artifacts de disco): se llama por asyncio.to_thread."""
    try:
        _load_ref(ref)
    except RefIlegible as exc:
        logger.warning("continuar: ref ilegible, el paso se rehace (%s)", exc)
        return False
    return True


def _texto_limite(activos: int) -> str:
    return f"Ya hay {activos} pipelines activos. Límite duro: {MAX_PARALLEL_PIPELINES}"


async def analizar(pipeline_id: str, invoked_by: str, reasignar: dict[str, str] | None) -> Analisis:
    politica = validate_resume(invoked_by)
    if not politica.ok:
        raise ContinuarRechazado(403, "invocador_no_autorizado", politica.reason)
    pipeline = await store.pipeline_get(pipeline_id)
    if pipeline is None:
        raise ContinuarRechazado(404, "no_existe", f"Pipeline '{pipeline_id}' no encontrado")
    if pipeline.status not in ESTADOS_CONTINUABLES:
        mensaje = ("tiene /resume" if pipeline.status == PipelineStatus.interrupted
                   else "solo se continúan pipelines aborted o expired")
        raise ContinuarRechazado(409, "estado_no_continuable",
                                 {"status": pipeline.status.value, "mensaje": mensaje})
    if check_kill_switch():
        raise ContinuarRechazado(423, "kill_switch", "Kill switch activo — no se puede continuar")

    # Los pasos VIGENTES (jacobs_steps), no la foto de creación en `plan`.
    plan = await store.steps_by_pipeline(pipeline_id)
    if [s.step_index for s in plan] != list(range(len(plan))):
        raise ContinuarRechazado(409, "plan_inconsistente",
                                 "los pasos guardados no son 0..N-1 sin huecos")

    contexto = dict(pipeline.context)
    reusados: list[int] = []
    a_correr: list[int] = []
    for paso in plan:
        clave = f"step_{paso.step_index}_ref"
        ref = contexto.get(clave)
        if ref and await asyncio.to_thread(_ref_legible, ref):
            reusados.append(paso.step_index)
            continue
        contexto.pop(clave, None)
        # La aprobación humana de hyde era para la corrida anterior (desvío 10).
        contexto.pop(f"hyde_approved_{paso.step_id}", None)
        paso.status = StepStatus.pending
        paso.error = None
        paso.started_at = None
        paso.finished_at = None
        paso.output_ref = None
        a_correr.append(paso.step_index)

    reasignados: dict[str, dict] = {}
    invalidas: list[dict] = []
    for clave, faceta in (reasignar or {}).items():
        try:
            indice = int(clave)
        except (TypeError, ValueError):
            invalidas.append({"paso": clave, "motivo": "el índice de paso no es un entero"})
            continue
        if indice not in a_correr:
            invalidas.append({"paso": indice, "motivo": "solo se reasignan pasos a correr, no los reusados"})
            continue
        if faceta not in VALID_FACETS:
            invalidas.append({"paso": indice, "motivo": f"faceta desconocida: '{faceta}'"})
            continue
        paso = plan[indice]
        reasignados[str(indice)] = {"de": paso.facet, "a": faceta}
        paso.facet = faceta
        paso.motor = faceta if faceta in MOTOR_FACETS else None
    if invalidas:
        raise ContinuarRechazado(422, "reasignacion_invalida", invalidas)

    codigo = "reasignacion_invalida" if reasignados else "plan_rechazado"
    violaciones = _check_cleanroom(plan)
    if violaciones:
        raise ContinuarRechazado(422, codigo, [v.to_dict() for v in violaciones])
    try:
        await _validate_plan_capabilities(plan)
    except PlanRejected as exc:
        raise ContinuarRechazado(422, codigo, [v.to_dict() for v in exc.violations]) from exc

    return Analisis(pipeline, plan, contexto, a_correr, reusados, reasignados)


def _pasos(a: Analisis) -> dict:
    return {"pasos_a_correr": a.pasos_a_correr, "pasos_reusados": a.pasos_reusados}


async def _prevuelo_de(a: Analisis, user_id: str | None, tenant_id: str | None):
    return await prevuelo(
        a.plan, a.contexto, pendientes=set(a.pasos_a_correr),
        user_id=user_id or a.pipeline.user_id, tenant_id=tenant_id or a.pipeline.tenant_id,
    )


async def previsualizar(pipeline_id: str, invoked_by: str, reasignar: dict[str, str] | None = None,
                        user_id: str | None = None, tenant_id: str | None = None) -> dict:
    """Reglas 1-8 sin escribir (§5.1). 403 y 404 se relanzan; el resto se
    devuelve como `continuable: False` con su motivo."""
    try:
        a = await analizar(pipeline_id, invoked_by, reasignar)
    except ContinuarRechazado as r:
        if r.status_code in (403, 404):
            raise
        return {"continuable": False, "motivo": r.cuerpo(),
                "pasos_a_correr": [], "pasos_reusados": [], "veredicto": None}
    veredicto = await _prevuelo_de(a, user_id, tenant_id)
    activos = await store.pipeline_count_active()
    motivo = None
    if activos >= MAX_PARALLEL_PIPELINES:
        motivo = {"code": "limite_de_activos", "detalle": _texto_limite(activos)}
    elif not veredicto.ok:
        motivo = {"code": "prevuelo_rechazado", "detalle": "el pre-vuelo encontró violaciones: ver `veredicto`"}
    return {"continuable": motivo is None, "motivo": motivo, **_pasos(a), "veredicto": veredicto.to_dict()}


async def continuar(pipeline_id: str, invoked_by: str, reasignar: dict[str, str] | None = None,
                    user_id: str | None = None, tenant_id: str | None = None,
                    costo_max_aceptado_usd: Decimal | None = None) -> tuple[dict, Pipeline]:
    async with candado_de_creacion:
        a = await analizar(pipeline_id, invoked_by, reasignar)
        activos = await store.pipeline_count_active()
        if activos >= MAX_PARALLEL_PIPELINES:
            raise ContinuarRechazado(429, "limite_de_activos", _texto_limite(activos))
        veredicto = await _prevuelo_de(a, user_id, tenant_id)
        if not veredicto.ok:
            await store.event_append(pipeline_id, "PREVUELO_RECHAZADO",
                                     {"code": "prevuelo_rechazado", **veredicto.to_dict()})
            raise ContinuarRechazado(422, "prevuelo_rechazado", veredicto.to_dict())
        if costo_max_aceptado_usd is not None and veredicto.costo_max_usd > costo_max_aceptado_usd:
            raise ContinuarRechazado(409, "costo_supera_lo_aceptado", {
                "costo_max_aceptado_usd": str(costo_max_aceptado_usd), **veredicto.to_dict(),
            })

        indice = min(a.pasos_a_correr) if a.pasos_a_correr else len(a.plan)
        nueva = await store.continuar_transaccion(
            pipeline_id, a.pipeline.run_epoch, a.pipeline.status,
            [a.plan[i] for i in a.pasos_a_correr], a.plan, a.contexto, indice,
        )
        if nueva is None:
            raise ContinuarRechazado(409, "estado_no_continuable", {
                "status": None,
                "mensaje": "otro pedido cambió el pipeline mientras se preparaba este: volvé a consultarlo",
            })
        continuado = a.pipeline.model_copy(update={
            "plan": a.plan, "context": a.contexto, "status": PipelineStatus.running,
            "run_epoch": nueva, "current_step_index": indice,
        })
        await store.event_append(pipeline_id, "PIPELINE_CONTINUED", {
            "by": invoked_by, "from_status": a.pipeline.status.value, "run_epoch": nueva,
            **_pasos(a), "reasignados": a.reasignados, "costo_max_usd": str(veredicto.costo_max_usd),
        })

    respuesta = {
        "pipeline_id": pipeline_id, "status": "running", "run_epoch": nueva, **_pasos(a),
        "costo_max_usd": str(veredicto.costo_max_usd),
        "pasos_costo": [c.to_dict() for c in veredicto.pasos_costo],
    }
    return respuesta, continuado
```

- [ ] **Step 7: Verlos pasar**

Run (puros): `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_jacobs_continuar.py tests/test_jacobs_preflight_endpoint.py tests/test_jacobs_resume_epoca.py tests/test_no_blocking_in_async.py policy/tests/test_no_fail_open_except.py 2>&1 | tail -2`
Expected: `0 failed`; `tests/test_jacobs_continuar.py` aporta `20 passed`.

Run (DB): `cd /home/fruiz/worktrees/jax-prevuelo && set -a && source /etc/jax/.env && set +a && export JAX_DB_NAME=jax_memory_test PYTHONPATH=.:las_manos && /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -v tests/test_jacobs_continuar_db.py tests/test_run_epoch_db.py 2>&1 | tail -12`
Expected: `10 passed` (4 + 6).

- [ ] **Step 8: CI y commit**

`tests-puros`: agregar `tests/test_jacobs_continuar.py` a las dos listas; reemplazar la línea de 822 por:
```
          # 822 -> 842 el 2026-09-17 (continuar): test_jacobs_continuar.py (+20) -- estados,
          #   refs ilegibles, pasos limpios, reasignación válida e inválida (clean-room),
          #   límite, pre-vuelo y costo sin escribir, transacción perdida, marcas de hyde.
          grep -qE "^842 passed, 1 skipped" /tmp/out || {
            echo "PISO ROTO: se esperaban 842 tests CORRIDOS y exactamente 1 skipped."; exit 1; }
```
`jacobs-gobernanza-db`: agregar `tests/test_jacobs_continuar_db.py` después de `tests/test_prevuelo_catalogo_db.py` en las dos listas; reemplazar la línea de 42 por:
```
          # 42 -> 46 el 2026-09-17 (continuar): test_jacobs_continuar_db.py (+4) -- la
          #   transacción aplica todo o nada, dos con la misma época: gana una, y diez
          #   continue concurrentes sobre el mismo pipeline: gana uno. EXPLAIN por PRIMARY.
          grep -qE "^46 passed" /tmp/out || {
            echo "PISO ROTO: se esperaban 46 tests CORRIDOS."; exit 1; }
```

```bash
cd /home/fruiz/worktrees/jax-prevuelo && pwd && git branch --show-current && \
git -C /home/fruiz/worktrees/jax-prevuelo add jacobs/candado.py jacobs/continuar.py jacobs/routes.py jacobs/store.py tests/test_jacobs_continuar.py tests/test_jacobs_continuar_db.py .github/workflows/policy.yml && \
git -C /home/fruiz/worktrees/jax-prevuelo commit -m "feat(jacobs): servicio para continuar pipelines abortados o vencidos

Reusa los pasos con ref legible, rehace el resto con reasignación opcional
validada, corre el pre-vuelo sobre los pendientes y escribe todo en una
transacción que incrementa la época.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Endpoints `continue` y `continue/preflight`

**Files:**
- Modify: `jacobs/routes.py` (imports, modelos y dos endpoints nuevos)
- Test: `tests/test_jacobs_continue_endpoint.py`
- Modify: `.github/workflows/policy.yml` (job `tests-puros`)

**Interfaces:**
- Consumes (Task 10): `continuar.continuar`, `continuar.previsualizar`, `continuar.ContinuarRechazado`.
- Produces:
  - `routes.ContinueRequest(invoked_by: str, reasignar: dict[str, str] | None = None, user_id: str | None = None, tenant_id: str | None = None, costo_max_aceptado_usd: Decimal | None = None)`
  - `routes.ContinuePreflightRequest(invoked_by: str, reasignar: dict[str, str] | None = None, user_id: str | None = None, tenant_id: str | None = None)`
  - `POST /jacobs/pipeline/{pipeline_id}/continue` → 200 respuesta del servicio + `run_pipeline` en background; rechazo → `HTTPException(status_code, detail=cuerpo())`; otro error → 503 `prevuelo_no_disponible`.
  - `POST /jacobs/pipeline/{pipeline_id}/continue/preflight` → 200 `{continuable, motivo, pasos_a_correr, pasos_reusados, veredicto}`; 403/404 como HTTP; otro error → 503.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_jacobs_continue_endpoint.py`:
```python
"""Endpoints de continuar (spec 2026-09-17 §5.1). El servicio va mockeado: lo
que se prueba acá es el mapeo HTTP y que la corrida se lance con la época nueva.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
from decimal import Decimal
from unittest.mock import AsyncMock, patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import pytest  # noqa: E402
from fastapi import BackgroundTasks, HTTPException  # noqa: E402

from jacobs import continuar, routes  # noqa: E402
from jacobs.models import Pipeline  # noqa: E402


def test_continue_ok_lanza_la_corrida_con_la_epoca_nueva():
    pipeline = Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma", mode="autonomous", run_epoch=5)
    servicio = AsyncMock(return_value=({"pipeline_id": "p1", "run_epoch": 5}, pipeline))
    bg = BackgroundTasks()
    with patch.object(continuar, "continuar", servicio):
        r = asyncio.run(routes.continue_pipeline("p1", routes.ContinueRequest(
            invoked_by="plataforma", reasignar={"4": "ada"}, costo_max_aceptado_usd=Decimal("1.5")), bg))
    assert r == {"pipeline_id": "p1", "run_epoch": 5}
    servicio.assert_awaited_once_with("p1", "plataforma", reasignar={"4": "ada"}, user_id=None,
                                      tenant_id=None, costo_max_aceptado_usd=Decimal("1.5"))
    assert bg.tasks[0].func is routes.run_pipeline and bg.tasks[0].args[0].run_epoch == 5


def test_un_rechazo_del_servicio_sale_con_su_status_y_cuerpo():
    rechazo = continuar.ContinuarRechazado(409, "estado_no_continuable", {"status": "completed", "mensaje": "m"})
    bg = BackgroundTasks()
    with patch.object(continuar, "continuar", AsyncMock(side_effect=rechazo)), \
         pytest.raises(HTTPException) as e:
        asyncio.run(routes.continue_pipeline("p1", routes.ContinueRequest(invoked_by="plataforma"), bg))
    assert e.value.status_code == 409
    assert e.value.detail == {"code": "estado_no_continuable", "status": "completed", "mensaje": "m"}
    assert bg.tasks == []


def test_continue_con_la_base_caida_da_503():
    with patch.object(continuar, "continuar", AsyncMock(side_effect=OSError("base caída"))), \
         pytest.raises(HTTPException) as e:
        asyncio.run(routes.continue_pipeline(
            "p1", routes.ContinueRequest(invoked_by="plataforma"), BackgroundTasks()))
    assert e.value.status_code == 503 and e.value.detail["code"] == "prevuelo_no_disponible"


def test_continue_preflight_devuelve_200_y_no_lanza():
    esperado = {"continuable": False, "motivo": {"code": "limite_de_activos", "detalle": "x"},
                "pasos_a_correr": [2], "pasos_reusados": [0, 1], "veredicto": None}
    servicio = AsyncMock(return_value=esperado)
    with patch.object(continuar, "previsualizar", servicio):
        r = asyncio.run(routes.continue_preflight(
            "p1", routes.ContinuePreflightRequest(invoked_by="plataforma", reasignar={"2": "thot"})))
    assert r == esperado
    servicio.assert_awaited_once_with("p1", "plataforma", reasignar={"2": "thot"}, user_id=None, tenant_id=None)


def test_continue_preflight_de_un_pipeline_inexistente_da_404():
    rechazo = continuar.ContinuarRechazado(404, "no_existe", "Pipeline 'p1' no encontrado")
    with patch.object(continuar, "previsualizar", AsyncMock(side_effect=rechazo)), \
         pytest.raises(HTTPException) as e:
        asyncio.run(routes.continue_preflight("p1", routes.ContinuePreflightRequest(invoked_by="plataforma")))
    assert e.value.status_code == 404
    assert e.value.detail == {"code": "no_existe", "detalle": "Pipeline 'p1' no encontrado"}
```

- [ ] **Step 2: Verlos fallar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_jacobs_continue_endpoint.py 2>&1 | tail -7`
Expected: `5 failed`, cada uno con `AttributeError: module 'jacobs.routes' has no attribute 'ContinueRequest'` (o `'ContinuePreflightRequest'`).

- [ ] **Step 3: Implementar**

En `jacobs/routes.py`:
1. Agregar `from decimal import Decimal` después de `import uuid`.
2. Después de `from jacobs import store` agregar `from jacobs import continuar as servicio_continuar`.
3. Después del endpoint `resume_pipeline` (antes del bloque de `approve-step`), agregar:
```python
# ----------------------------------------------------------------
#  POST /jacobs/pipeline/{id}/continue  (spec 2026-09-17 §5)
# ----------------------------------------------------------------

class ContinueRequest(BaseModel):
    invoked_by:             str
    reasignar:              dict[str, str] | None = None
    user_id:                str | None = None
    tenant_id:              str | None = None
    costo_max_aceptado_usd: Decimal | None = None


class ContinuePreflightRequest(BaseModel):
    invoked_by: str
    reasignar:  dict[str, str] | None = None
    user_id:    str | None = None
    tenant_id:  str | None = None


def _no_disponible(exc: Exception) -> HTTPException:
    motivo = recortar_redactado(f"{type(exc).__name__}: {exc}", 300)
    logger.error("continuar no disponible: %s", motivo, exc_info=True)
    return HTTPException(status_code=503, detail={"code": "prevuelo_no_disponible", "motivo": motivo})


@router.post("/pipeline/{pipeline_id}/continue")
async def continue_pipeline(
    pipeline_id: str, req: ContinueRequest, background: BackgroundTasks
) -> dict:
    """Continúa un pipeline aborted o expired (spec §5.1): reusa los pasos con
    ref legible, rehace el resto (con reasignación opcional) y corre el
    pre-vuelo sobre los pendientes. 403/404/409/422/423/429 con
    {"code", ...}; 503 si el análisis o el pre-vuelo no pueden correr."""
    try:
        respuesta, pipeline = await servicio_continuar.continuar(
            pipeline_id, req.invoked_by, reasignar=req.reasignar, user_id=req.user_id,
            tenant_id=req.tenant_id, costo_max_aceptado_usd=req.costo_max_aceptado_usd,
        )
    except servicio_continuar.ContinuarRechazado as rechazo:
        raise HTTPException(status_code=rechazo.status_code, detail=rechazo.cuerpo()) from rechazo
    except Exception as exc:
        raise _no_disponible(exc) from exc
    background.add_task(run_pipeline, pipeline)
    return respuesta


@router.post("/pipeline/{pipeline_id}/continue/preflight")
async def continue_preflight(pipeline_id: str, req: ContinuePreflightRequest) -> dict:
    """Reglas 1-8 de continuar SIN escribir (spec §5.1): lo que la Mesa muestra
    antes de confirmar."""
    try:
        return await servicio_continuar.previsualizar(
            pipeline_id, req.invoked_by, reasignar=req.reasignar,
            user_id=req.user_id, tenant_id=req.tenant_id,
        )
    except servicio_continuar.ContinuarRechazado as rechazo:
        raise HTTPException(status_code=rechazo.status_code, detail=rechazo.cuerpo()) from rechazo
    except Exception as exc:
        raise _no_disponible(exc) from exc
```

- [ ] **Step 4: Verlos pasar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_jacobs_continue_endpoint.py tests/test_jacobs_continuar.py tests/test_jacobs_preflight_endpoint.py tests/test_jacobs_invoked_by_rol.py 2>&1 | tail -2`
Expected: `0 failed`; `tests/test_jacobs_continue_endpoint.py` aporta `5 passed`.

- [ ] **Step 5: CI y commit**

`tests-puros`: agregar `tests/test_jacobs_continue_endpoint.py` a las dos listas; reemplazar la línea de 842 por:
```
          # 842 -> 847 el 2026-09-17: test_jacobs_continue_endpoint.py (+5) -- continue lanza
          #   con la época nueva, rechazos con su status y cuerpo, 503, continue/preflight.
          grep -qE "^847 passed, 1 skipped" /tmp/out || {
            echo "PISO ROTO: se esperaban 847 tests CORRIDOS y exactamente 1 skipped."; exit 1; }
```

```bash
cd /home/fruiz/worktrees/jax-prevuelo && pwd && git branch --show-current && \
git -C /home/fruiz/worktrees/jax-prevuelo add jacobs/routes.py tests/test_jacobs_continue_endpoint.py .github/workflows/policy.yml && \
git -C /home/fruiz/worktrees/jax-prevuelo commit -m "feat(jacobs): endpoints continue y continue/preflight

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: El relanzador CLI sin lógica propia

**Files:**
- Modify (reescritura completa): `tools/jacobs_relaunch.py`
- Test: `tests/test_jacobs_relaunch_cli.py`
- Modify: `.github/workflows/policy.yml` (job `tests-puros`)

**Interfaces:**
- Consumes (Task 10): `continuar.continuar`, `continuar.ContinuarRechazado`; `executor.run_pipeline`; `store.pipeline_get`, `store.steps_by_pipeline`.
- Produces: `relanzar(pipeline_id: str, reasignar: dict[str, str], costo_max_aceptado: Decimal | None) -> int` (0 completado, 1 rechazado, 3 terminó en otro estado); `parsear_reasignar(valores: list[str]) -> dict[str, str]` (lanza `argparse.ArgumentTypeError`); `main(argv=None)`. Uso: `python tools/jacobs_relaunch.py <pipeline_id> [--reasignar PASO=FACETA ...] [--costo-max-aceptado USD]`.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_jacobs_relaunch_cli.py`:
```python
"""El relanzador CLI es un cliente de jacobs/continuar.py (spec 2026-09-17 §5.4):
sin escrituras propias y sin leer /etc/jax/.env al importarse.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import importlib.util
import os
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ["JAX_DB_NAME"] = "jax_memory_test"

import pytest  # noqa: E402

from jacobs import continuar  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus  # noqa: E402

RUTA = Path(__file__).resolve().parents[1] / "tools" / "jacobs_relaunch.py"


def _modulo():
    spec = importlib.util.spec_from_file_location("jacobs_relaunch_bajo_test", RUTA)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_llama_a_continuar_y_corre_la_corrida_que_devuelve():
    cli = _modulo()
    pipeline = Pipeline(pipeline_id="p1", name="t", invoked_by="plataforma", mode="autonomous", run_epoch=2)
    final = pipeline.model_copy(update={"status": PipelineStatus.completed})
    respuesta = {"run_epoch": 2, "pasos_reusados": [0], "pasos_a_correr": [1], "costo_max_usd": "0.100000"}
    with patch("jacobs.continuar.continuar", AsyncMock(return_value=(respuesta, pipeline))) as servicio, \
         patch("jacobs.executor.run_pipeline", AsyncMock()) as corrida, \
         patch("jacobs.store.pipeline_get", AsyncMock(return_value=final)), \
         patch("jacobs.store.steps_by_pipeline", AsyncMock(return_value=[])):
        rc = asyncio.run(cli.relanzar("p1", {"1": "ada"}, Decimal("0.5")))
    assert rc == 0
    servicio.assert_awaited_once_with("p1", "plataforma", reasignar={"1": "ada"},
                                      costo_max_aceptado_usd=Decimal("0.5"))
    corrida.assert_awaited_once_with(pipeline)


def test_un_rechazo_sale_con_1_sin_correr():
    cli = _modulo()
    rechazo = continuar.ContinuarRechazado(409, "estado_no_continuable", {"status": "completed", "mensaje": "m"})
    with patch("jacobs.continuar.continuar", AsyncMock(side_effect=rechazo)), \
         patch("jacobs.executor.run_pipeline", AsyncMock()) as corrida:
        rc = asyncio.run(cli.relanzar("p1", {}, None))
    assert rc == 1
    corrida.assert_not_awaited()


def test_parsear_reasignar_valido():
    assert _modulo().parsear_reasignar(["4=ada", " 5 = thot "]) == {"4": "ada", "5": "thot"}


def test_parsear_reasignar_invalido_lanza():
    cli = _modulo()
    for malo in ("ada", "x=ada", "4="):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parsear_reasignar([malo])


def test_no_escribe_pasos_ni_estado_por_su_cuenta():
    arbol = ast.parse(RUTA.read_text(encoding="utf-8"))
    llamados = {n.func.attr for n in ast.walk(arbol)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    prohibidos = {"step_upsert", "step_upsert_si_epoca", "pipeline_update_status",
                  "pipeline_update_status_si_epoca", "event_append", "continuar_transaccion"}
    assert not (llamados & prohibidos), llamados & prohibidos


def test_el_env_de_produccion_solo_se_lee_desde_una_funcion():
    arbol = ast.parse(RUTA.read_text(encoding="utf-8"))

    def permitido(nodo):
        if isinstance(nodo, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign)):
            return True
        if isinstance(nodo, ast.Expr) and isinstance(nodo.value, ast.Constant):
            return True  # docstring
        return (isinstance(nodo, ast.If) and isinstance(nodo.test, ast.Compare)
                and isinstance(nodo.test.left, ast.Name) and nodo.test.left.id == "__name__")

    sueltos = [ast.dump(n)[:60] for n in arbol.body if not permitido(n)]
    assert sueltos == []
```

- [ ] **Step 2: Verlos fallar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_jacobs_relaunch_cli.py 2>&1 | tail -8`
Expected: `6 failed`: los dos de `relanzar` con `AttributeError: module 'jacobs_relaunch_bajo_test' has no attribute 'relanzar'`; los dos de `parsear_reasignar` con `AttributeError: ... has no attribute 'parsear_reasignar'`; `test_no_escribe_pasos_ni_estado_por_su_cuenta` con `AssertionError` que lista `step_upsert`, `pipeline_update_status`, `event_append`; `test_el_env_de_produccion_solo_se_lee_desde_una_funcion` con `AssertionError` que lista el `If`/`Expr` sueltos del módulo viejo. (Solo este archivo se corre contra el CLI viejo: su import lee `/etc/jax/.env` con `setdefault`, con `JAX_DB_NAME` ya forzado a la base de tests.)

- [ ] **Step 3: Implementar**

Reemplazar `tools/jacobs_relaunch.py` entero por:
```python
#!/usr/bin/env python3
"""
Jacobs — relanzador de pipelines abortados o vencidos (CLI).

Desde 2026-09-17 (spec prevuelo-y-continuar §5.4) NO tiene lógica propia:
llama a jacobs/continuar.py::continuar, la MISMA función que
POST /jacobs/pipeline/{id}/continue, y después corre la corrida en este
proceso. Los tres agujeros de la versión anterior se cierran por construcción:
  1. usaba la foto de `plan` de la creación -> continuar lee jacobs_steps;
  2. no borraba las refs de los pasos a rehacer -> continuar las quita;
  3. se saltaba el candado y el límite -> continuar los aplica. El candado es
     de proceso: entre este CLI y LAS MANOS serializa la transacción con
     SELECT ... FOR UPDATE y la época (desvío 7 del plan).

`--from-step` ya no existe: qué se reusa lo decide la ref legible de cada
paso, no un número elegido a mano.

Uso:
    python tools/jacobs_relaunch.py <pipeline_id> [--reasignar 4=ada ...] [--costo-max-aceptado 1.50]

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ENV_PATH = "/etc/jax/.env"


def _cargar_env(ruta: str = ENV_PATH) -> None:
    """/etc/jax/.env -> os.environ sin pisar lo ya seteado. SOLO desde main():
    antes corría al importar el módulo, y cualquier import (un test) quedaba
    con el entorno de producción cargado."""
    if not os.path.exists(ruta):
        return
    with open(ruta, encoding="utf-8") as fh:
        for linea in fh:
            linea = linea.strip()
            if not linea or linea.startswith("#") or "=" not in linea:
                continue
            clave, valor = linea.split("=", 1)
            os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))


def _asegurar_rutas() -> None:
    """Raíz del repo (jacobs) y las_manos (facet_resolver, contrato_dispatch,
    motor_registry), como corre LAS MANOS. Antes era ~/jax fijo: desde un
    worktree importaba el código del checkout de producción."""
    for ruta in (str(RAIZ), str(RAIZ / "las_manos")):
        if ruta not in sys.path:
            sys.path.insert(0, ruta)


def parsear_reasignar(valores: list[str]) -> dict[str, str]:
    salida: dict[str, str] = {}
    for valor in valores:
        indice, separador, faceta = valor.partition("=")
        if not separador or not indice.strip().isdigit() or not faceta.strip():
            raise argparse.ArgumentTypeError(
                f"--reasignar espera PASO=FACETA (por ejemplo 4=ada); llegó {valor!r}")
        salida[indice.strip()] = faceta.strip()
    return salida


def _costo(valor: str) -> Decimal:
    try:
        costo = Decimal(valor)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"--costo-max-aceptado no es un número: {valor!r}") from exc
    if costo < 0:
        raise argparse.ArgumentTypeError("--costo-max-aceptado no puede ser negativo")
    return costo


async def relanzar(pipeline_id: str, reasignar: dict[str, str], costo_max_aceptado: Decimal | None) -> int:
    from jacobs import continuar, store
    from jacobs.executor import run_pipeline
    from jacobs.models import INVOKER_PLATAFORMA, PipelineStatus

    try:
        respuesta, pipeline = await continuar.continuar(
            pipeline_id, INVOKER_PLATAFORMA,
            reasignar=reasignar or None, costo_max_aceptado_usd=costo_max_aceptado,
        )
    except continuar.ContinuarRechazado as rechazo:
        print(f"✗ {rechazo.status_code} {json.dumps(rechazo.cuerpo(), ensure_ascii=False)}")
        return 1

    print(
        f"▶ Continuando {pipeline_id} (época {respuesta['run_epoch']}): "
        f"reusados={respuesta['pasos_reusados']} a correr={respuesta['pasos_a_correr']} "
        f"costo máximo=US$ {respuesta['costo_max_usd']}"
    )
    await run_pipeline(pipeline)

    final = await store.pipeline_get(pipeline_id)
    for s in await store.steps_by_pipeline(pipeline_id):
        duracion = round(s.finished_at - s.started_at, 1) if s.started_at and s.finished_at else None
        print(f"  idx={s.step_index} {s.facet}/{s.capability} status={s.status.value} "
              f"timeout={s.timeout_seconds} out={(s.output_ref or '')[:11]} dur={duracion} err={s.error}")
    print(f"=== FINAL: pipeline status={final.status.value} ===")
    return 0 if final.status == PipelineStatus.completed else 3


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Continúa un pipeline de Jacobs abortado o vencido (misma función que el endpoint).")
    parser.add_argument("pipeline_id")
    parser.add_argument("--reasignar", action="append", default=[], metavar="PASO=FACETA")
    parser.add_argument("--costo-max-aceptado", type=_costo, default=None, metavar="USD")
    args = parser.parse_args(argv)
    try:
        reasignar = parsear_reasignar(args.reasignar)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    _cargar_env()
    _asegurar_rutas()
    sys.exit(asyncio.run(relanzar(args.pipeline_id, reasignar, args.costo_max_aceptado)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verlos pasar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_jacobs_relaunch_cli.py tests/test_no_blocking_in_async.py tests/test_aiomysql_connect_timeout_tripwire.py 2>&1 | tail -2`
Expected: `0 failed`; `tests/test_jacobs_relaunch_cli.py` aporta `6 passed`.

Run: `cd /home/fruiz/worktrees/jax-prevuelo && /home/fruiz/jax/las_manos/.venv/bin/python tools/jacobs_relaunch.py --help | head -3`
Expected: la línea de uso con `--reasignar PASO=FACETA` y `--costo-max-aceptado USD`; sin `--from-step`.

- [ ] **Step 5: CI y commit**

`tests-puros`: agregar `tests/test_jacobs_relaunch_cli.py` a las dos listas; reemplazar la línea de 847 por:
```
          # 847 -> 853 el 2026-09-17: test_jacobs_relaunch_cli.py (+6) -- el CLI llama a
          #   continuar.py, no escribe por su cuenta y no carga /etc/jax/.env al importarse.
          grep -qE "^853 passed, 1 skipped" /tmp/out || {
            echo "PISO ROTO: se esperaban 853 tests CORRIDOS y exactamente 1 skipped."; exit 1; }
```

```bash
cd /home/fruiz/worktrees/jax-prevuelo && pwd && git branch --show-current && \
git -C /home/fruiz/worktrees/jax-prevuelo add tools/jacobs_relaunch.py tests/test_jacobs_relaunch_cli.py .github/workflows/policy.yml && \
git -C /home/fruiz/worktrees/jax-prevuelo commit -m "refactor(jacobs): el relanzador CLI usa continuar.py

Cierra sus tres agujeros por construcción (foto vieja del plan, refs de
pasos a rehacer, candado y límite) y deja de cargar el .env al importarse.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: Medición de la semilla de `min_output_tokens`

**Files:**
- Create: `scripts/medir_min_output_tokens.py`
- Test: `tests/test_medir_min_output_tokens.py`
- Modify: `.github/workflows/policy.yml` (job `tests-puros`)
- Modify (al ejecutar con GO): este plan, sección nueva "Semilla medida de min_output_tokens" al final

**Interfaces:**
- Consumes: `jacobs.store.get_conn` (solo `SELECT`), `las_manos/logs/motor_jobs.jsonl`.
- Produces: `redondear(n: int) -> int` (múltiplo de 1024 hacia arriba; ≤ 0 → 0), `maximos_de_jobs(lineas: Iterable[str]) -> tuple[dict[str, int], int]` (máximos por capability de jobs `completed` con `_usage.completion_tokens`, y cantidad de líneas rotas), `combinar(*fuentes: dict[str, int]) -> dict[str, int]`, `_SQL_HTTP_DIRECTO`, `HOLGURA_S`, CLI `--jobs RUTA`. Imprime la tabla y los `UPDATE capability SET min_output_tokens=...` que consume la migración del plan P. No escribe nada.

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_medir_min_output_tokens.py`:
```python
"""Aritmética de la medición de la semilla de capability.min_output_tokens
(spec 2026-09-17 §4.4). La lectura de producción no se testea acá: la hace el
script con GO, en solo lectura.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from medir_min_output_tokens import combinar, maximos_de_jobs, redondear  # noqa: E402


def test_redondea_hacia_arriba_al_multiplo_de_1024():
    assert [redondear(n) for n in (1, 1024, 1025, 7997)] == [1024, 1024, 2048, 8192]


def test_cero_o_negativo_queda_en_cero():
    assert redondear(0) == 0 and redondear(-5) == 0


def test_maximos_de_jobs_solo_completed_con_completion_tokens():
    lineas = [
        json.dumps({"status": "completed", "capability": "generate", "_usage": {"completion_tokens": 3000}}),
        json.dumps({"status": "completed", "capability": "generate", "_usage": {"completion_tokens": 5000}}),
        json.dumps({"status": "failed", "capability": "generate", "_usage": {"completion_tokens": 8000}}),
        json.dumps({"status": "completed", "capability": "design", "_usage": {}}),
        "",
    ]
    assert maximos_de_jobs(lineas) == ({"generate": 5000}, 0)


def test_una_linea_rota_se_cuenta_no_se_esconde():
    lineas = ["{roto", json.dumps({"status": "completed", "capability": "reason", "_usage": {"completion_tokens": 10}})]
    assert maximos_de_jobs(lineas) == ({"reason": 10}, 1)


def test_combinar_toma_el_maximo_por_capability():
    assert combinar({"a": 10, "b": 5}, {"a": 7, "c": 1}) == {"a": 10, "b": 5, "c": 1}
```

- [ ] **Step 2: Verlos fallar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_medir_min_output_tokens.py 2>&1 | tail -3`
Expected: `1 error` de colección: `ModuleNotFoundError: No module named 'medir_min_output_tokens'`.

- [ ] **Step 3: Implementar**

`scripts/medir_min_output_tokens.py`:
```python
#!/usr/bin/env python3
"""Mide la semilla de capability.min_output_tokens (spec 2026-09-17 §4.4).

SOLO LEE. Máximo de tokens de salida de corridas COMPLETADAS por capability,
redondeado hacia arriba a múltiplo de 1024, de dos fuentes:
  1. Motor Registry: las_manos/logs/motor_jobs.jsonl, `_usage.completion_tokens`
     de los jobs `completed` (incluye razonamiento: es lo que el tope tiene que
     dejar pasar).
  2. Transportes HTTP directos de Jacobs: axioma_usage.tokens_out
     (request_type='pipeline') unido a jacobs_steps completados por faceta y
     ventana de tiempo. Si dos pasos de la misma faceta se solapan, gana el
     mayor: cota superior, que es la dirección segura para un mínimo.
Una capability sin corridas medibles queda en 0 y se declara.

Consulta de UNA vez, fuera del camino caliente: su EXPLAIN se registra al
correrla (Task 13, Step 6 del plan) y un scan se acepta por ser un script
puntual.

Uso (con GO, contra producción, solo lectura):
  set -a; source /etc/jax/.env; set +a
  PYTHONPATH=.:las_manos python scripts/medir_min_output_tokens.py --jobs las_manos/logs/motor_jobs.jsonl

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path

MULTIPLO = 1024
HOLGURA_S = 5  # axioma_usage.created_at es TIMESTAMP (segundos enteros)

_SQL_HTTP_DIRECTO = (
    "SELECT s.capability, MAX(u.tokens_out) FROM jacobs_steps s "
    "JOIN axioma_usage u ON u.facet = s.facet AND u.request_type = 'pipeline' "
    " AND UNIX_TIMESTAMP(u.created_at) BETWEEN s.started_at - %s AND s.finished_at + %s "
    "WHERE s.status = 'completed' AND s.started_at IS NOT NULL AND s.finished_at IS NOT NULL "
    "GROUP BY s.capability"
)


def redondear(n: int) -> int:
    if n <= 0:
        return 0
    return math.ceil(n / MULTIPLO) * MULTIPLO


def maximos_de_jobs(lineas: Iterable[str]) -> tuple[dict[str, int], int]:
    maximos: dict[str, int] = {}
    rotas = 0
    for linea in lineas:
        linea = linea.strip()
        if not linea:
            continue
        try:
            job = json.loads(linea)
        except json.JSONDecodeError:
            rotas += 1
            continue
        if job.get("status") != "completed":
            continue
        tokens = (job.get("_usage") or {}).get("completion_tokens")
        capability = job.get("capability")
        if not capability or not isinstance(tokens, int):
            continue
        maximos[capability] = max(maximos.get(capability, 0), tokens)
    return maximos, rotas


def combinar(*fuentes: dict[str, int]) -> dict[str, int]:
    salida: dict[str, int] = {}
    for fuente in fuentes:
        for capability, valor in fuente.items():
            salida[capability] = max(salida.get(capability, 0), valor)
    return salida


async def _maximos_http_directo() -> tuple[dict[str, int], list[str]]:
    from jacobs import store
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(_SQL_HTTP_DIRECTO, (HOLGURA_S, HOLGURA_S))
            maximos = {cap: int(valor or 0) for cap, valor in await cur.fetchall()}
            await cur.execute("SELECT `key` FROM capability ORDER BY `key`")
            capabilities = [r[0] for r in await cur.fetchall()]
    finally:
        conn.close()
    return maximos, capabilities


def main() -> int:
    parser = argparse.ArgumentParser(description="Mide la semilla de capability.min_output_tokens (solo lee).")
    parser.add_argument("--jobs", required=True, type=Path, help="ruta a motor_jobs.jsonl")
    args = parser.parse_args()

    with args.jobs.open(encoding="utf-8") as fh:
        de_jobs, rotas = maximos_de_jobs(fh)
    de_http, capabilities = asyncio.run(_maximos_http_directo())
    medidos = combinar(de_jobs, de_http)

    print(f"motor_jobs.jsonl: {len(de_jobs)} capabilities medidas, {rotas} líneas ilegibles")
    print(f"axioma_usage + jacobs_steps: {len(de_http)} capabilities medidas")
    print("| capability | max tokens medidos | min_output_tokens | fuente |")
    print("|---|---|---|---|")
    for cap in capabilities:
        fuentes = [n for n, d in (("motor_jobs", de_jobs), ("http_directo", de_http)) if cap in d]
        fuente = ", ".join(fuentes) or "sin corridas medibles"
        print(f"| {cap} | {medidos.get(cap, 0)} | {redondear(medidos.get(cap, 0))} | {fuente} |")
    print()
    for cap in capabilities:
        print(f"UPDATE capability SET min_output_tokens={redondear(medidos.get(cap, 0))} WHERE `key`='{cap}';")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Verlos pasar**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_medir_min_output_tokens.py tests/test_no_blocking_in_async.py tests/test_aiomysql_connect_timeout_tripwire.py 2>&1 | tail -2`
Expected: `0 failed`; el archivo nuevo aporta `5 passed`.

- [ ] **Step 5: CI y commit**

`tests-puros`: agregar `tests/test_medir_min_output_tokens.py` a las dos listas; reemplazar la línea de 853 por:
```
          # 853 -> 858 el 2026-09-17: test_medir_min_output_tokens.py (+5) -- redondeo a 1024,
          #   solo jobs completed, líneas rotas contadas, máximo por capability.
          grep -qE "^858 passed, 1 skipped" /tmp/out || {
            echo "PISO ROTO: se esperaban 858 tests CORRIDOS y exactamente 1 skipped."; exit 1; }
```

```bash
cd /home/fruiz/worktrees/jax-prevuelo && pwd && git branch --show-current && \
git -C /home/fruiz/worktrees/jax-prevuelo add scripts/medir_min_output_tokens.py tests/test_medir_min_output_tokens.py .github/workflows/policy.yml && \
git -C /home/fruiz/worktrees/jax-prevuelo commit -m "feat(scripts): medición de solo lectura de min_output_tokens

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Medir (requiere GO del controlador para leer producción)**

Sin GO explícito para leer `jax_memory` y `/home/fruiz/jax/las_manos/logs/motor_jobs.jsonl`, este paso NO se ejecuta y se reporta como pendiente al controlador. Con GO:
```bash
cd /home/fruiz/worktrees/jax-prevuelo && set -a && source /etc/jax/.env && set +a && \
PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python scripts/medir_min_output_tokens.py \
  --jobs /home/fruiz/jax/las_manos/logs/motor_jobs.jsonl
```
y el EXPLAIN de la consulta, también de solo lectura:
```bash
cd /home/fruiz/worktrees/jax-prevuelo && set -a && source /etc/jax/.env && set +a && \
PYTHONPATH=.:las_manos:scripts /home/fruiz/jax/las_manos/.venv/bin/python - <<'PY'
import asyncio
from medir_min_output_tokens import _SQL_HTTP_DIRECTO, HOLGURA_S
from jacobs import store
async def main():
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("EXPLAIN " + _SQL_HTTP_DIRECTO, (HOLGURA_S, HOLGURA_S))
            for fila in await cur.fetchall():
                print(fila)
    finally:
        conn.close()
asyncio.run(main())
PY
```
Expected: la tabla con una fila por capability y los `UPDATE`. Agregar al final de este plan una sección `## Semilla medida de min_output_tokens` con fecha, quién la midió, el comando, la tabla completa, el EXPLAIN y la lista de capabilities en 0 "sin corridas medibles"; pasar los `UPDATE` al controlador para la migración del plan P. Commit solo de este archivo:
```bash
cd /home/fruiz/worktrees/jax-prevuelo && pwd && git branch --show-current && \
git -C /home/fruiz/worktrees/jax-prevuelo add docs/superpowers/plans/2026-09-17-prevuelo-y-continuar-jacobs.md && \
git -C /home/fruiz/worktrees/jax-prevuelo commit -m "docs(plan): semilla medida de min_output_tokens

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: Cierre en CI — pisos del runner, canarios, deuda y rebase

**Files:**
- Modify: `.github/workflows/policy.yml` (solo si el runner cuenta otro número)
- Modify: `DEUDA.md`
- Test: todas las suites nuevas, en sus jobs

**Interfaces:**
- Consumes: todo lo anterior.
- Produces: rama con CI verde y pisos confirmados por el runner; dos canarios rojos vistos; deuda fechada registrada.

- [ ] **Step 1: Controles de clase localmente**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q policy/tests/test_no_fail_open_except.py tests/test_no_blocking_in_async.py tests/test_aiomysql_connect_timeout_tripwire.py tests/test_payload_max_tokens_literal_tripwire.py las_manos/_output_validator_test.py jacobs/_facet_health_test.py 2>&1 | tail -2`
Expected: `0 failed`.

- [ ] **Step 2: Lista completa de `tests-puros` localmente**

Run: la lista de Task 1 Step 1 más, al final: `tests/test_prevuelo_config.py tests/test_jacobs_epoca_ejecutor.py tests/test_jacobs_resume_epoca.py tests/test_prevuelo_reglas.py tests/test_prevuelo_salud_pura.py tests/test_prevuelo_sonda.py tests/test_prevuelo_orquestador.py tests/test_jacobs_preflight_endpoint.py tests/test_jacobs_continuar.py tests/test_jacobs_continue_endpoint.py tests/test_jacobs_relaunch_cli.py tests/test_medir_min_output_tokens.py`
Expected: BASE_LOCAL + 116 passed, mismos skipped/xfailed que BASE_LOCAL.

- [ ] **Step 3: Lista completa de `jacobs-gobernanza-db` localmente**

Run: `cd /home/fruiz/worktrees/jax-prevuelo && set -a && source /etc/jax/.env && set +a && export JAX_DB_NAME=jax_memory_test PYTHONPATH=.:las_manos && /home/fruiz/jax/las_manos/.venv/bin/python -m pytest -q tests/test_plan_validation.py tests/test_valid_capabilities_includes_file_tools.py tests/test_plan_capability_hint.py tests/test_catalog_mode_db.py tests/test_contrato_dispatch_db.py jacobs/_direct_usage_test.py tests/test_run_epoch_db.py tests/test_prevuelo_catalogo_db.py tests/test_jacobs_continuar_db.py 2>&1 | tail -2`
Expected: `46 passed`.

- [ ] **Step 4: Deuda fechada**

Agregar a `DEUDA.md`, en la sección de ítems abiertos y con el formato que ya usa el archivo:
```markdown
- **[2026-09-17 · vence 2026-10-15] Límite de activos entre procesos.** `jacobs/continuar.py` toma
  `candado_de_creacion` (asyncio.Lock de proceso). El CLI `tools/jacobs_relaunch.py` corre en otro
  proceso: la transacción con `FOR UPDATE` y la época impiden un doble continue, pero el conteo contra
  `MAX_PARALLEL_PIPELINES` desde el CLI tiene carrera con un create de LAS MANOS. Arreglo: contar y
  reservar el cupo dentro de la misma transacción. Origen: plan prevuelo-y-continuar, desvío 7.
- **[2026-09-17 · vence 2026-10-15] Sonda que vence sin registrar uso.** Una sonda con timeout no
  conoce sus tokens y no escribe `axioma_usage` (`jacobs/sonda.py::_registrar`). El proveedor pudo
  haber cobrado la llamada. Arreglo: registrar la llamada con tokens estimados (el techo de la sonda)
  y marca de estimación.
```

- [ ] **Step 5: Rebase sobre master**

```bash
cd /home/fruiz/worktrees/jax-prevuelo && git -C /home/fruiz/worktrees/jax-prevuelo fetch origin && \
git -C /home/fruiz/worktrees/jax-prevuelo log --oneline HEAD..origin/master | head -30
```
Si hay commits nuevos en `origin/master` (frentes E/F): `git -C /home/fruiz/worktrees/jax-prevuelo rebase origin/master`, resolviendo conflictos en `routes.py`, `executor.py`, `store.py`, `policy.py` y `policy.yml` preservando lo de ellos (en `policy.yml`: unir las listas y sumar los incrementos de piso de ambos lados; el número final lo confirma el runner). Repetir Steps 1-3 después del rebase.

- [ ] **Step 6: Commit**

```bash
cd /home/fruiz/worktrees/jax-prevuelo && pwd && git branch --show-current && \
git -C /home/fruiz/worktrees/jax-prevuelo add DEUDA.md && \
git -C /home/fruiz/worktrees/jax-prevuelo commit -m "docs(deuda): límite entre procesos y uso de sondas vencidas

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 7: Subir la rama y canarios en el runner (requiere GO del controlador)**

Precondición: el plan P está mergeado en `jax-platform` master. Si no, `jacobs-gobernanza-db` da rojo por esquema, no por código: se declara y se espera.

1. Con GO del controlador: publicar la rama `feat/prevuelo-y-continuar` en `origin` y abrir el PR con `gh pr create` (cuerpo que termine con `🤖 Generated with [Claude Code](https://claude.com/claude-code)`).
2. Canario `tests-puros`: commit temporal que en `tests/test_prevuelo_reglas.py` cambia `Decimal("0.009282")` por `Decimal("0.009281")`.
3. Canario `jacobs-gobernanza-db`: en el MISMO commit temporal, en `tests/test_jacobs_continuar_db.py` cambiar `assert len(ganadores) == 1 and len(rechazos) == 9, resultados` por `assert len(ganadores) == 2 and len(rechazos) == 8, resultados`. Publicar. Expected en el runner: `tests-puros` rojo nombrando `test_costo_acotado_es_la_formula_redondeada_hacia_arriba`, y `jacobs-gobernanza-db` rojo nombrando `test_continue_concurrente_sobre_el_mismo_pipeline_solo_uno_gana`.
4. `git -C /home/fruiz/worktrees/jax-prevuelo revert --no-edit HEAD` (commit de reversión: no se reescribe historia publicada) y publicar.
5. Expected: todos los jobs verdes; `tests-puros` imprime `858 passed, 1 skipped, 1 xfailed` y `jacobs-gobernanza-db` imprime `46 passed`. Si el runner cuenta OTRO número, se pone el del runner en el `grep` con un comentario "MEDIDO por el runner (run #N)" que explica la diferencia, y se vuelve a publicar.
6. Registrar en el ledger de la ejecución la URL del run rojo de cada canario y la del run verde final.

---

### Task 15: Carga de `/jacobs/preflight` y `continue` concurrente

**Files:**
- Modify: `CONTEXT.md` (§9, entrada fechada con los números)

**Interfaces:**
- Consumes: la rama completa; `scripts/load_test.py`; `jax_memory_test` migrada con P (Task 6, Step 1).
- Produces: p95, p99, rps y tasa de error de `/jacobs/preflight` a c=1/10/25 con catálogo y salud frescos (sin sondas); resultado de 25 `continue` concurrentes sobre el mismo pipeline. Sin número medido, no hay GO.

- [ ] **Step 1: Sembrar credenciales y salud frescas en `jax_memory_test`**

```bash
cd /home/fruiz/worktrees/jax-prevuelo && set -a && source /etc/jax/.env && set +a && export JAX_DB_NAME=jax_memory_test PYTHONPATH=.:las_manos && \
/home/fruiz/jax/las_manos/.venv/bin/python - <<'PY'
import asyncio, time, uuid
from jacobs import store
CLAVES_HTTP = ("hipatia", "jekyll", "thot")
async def main():
    marca = f"carga-{uuid.uuid4().hex[:8]}"
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT DATABASE()")
            assert (await cur.fetchone())[0] == "jax_memory_test"
            await cur.execute("SELECT `key` FROM motor")
            motores = [r[0] for r in await cur.fetchall()]
            await cur.execute("SELECT DISTINCT provider_id FROM facet_binding WHERE role='primary' AND facet_key IN (%s,%s,%s)", CLAVES_HTTP)
            proveedores = {r[0] for r in await cur.fetchall()}
            await cur.execute("SELECT DISTINCT mo.provider_id FROM motor_resolved m JOIN model mo ON mo.id = m.model_ref")
            proveedores |= {r[0] for r in await cur.fetchall()}
            for proveedor in sorted(proveedores):
                await cur.execute("INSERT INTO credential (provider_id, env_key, encrypted_value, state) VALUES (%s, %s, 'carga-no-se-descifra', 'active')", (proveedor, marca))
            ahora = time.time()
            for clave in (*CLAVES_HTTP, *motores):
                await cur.execute("INSERT INTO facet_health_event (facet, outcome, source, detail, ts) VALUES (%s, 'ok', 'preflight', %s, %s)", (clave, marca, ahora))
    finally:
        conn.close()
    print(marca)
asyncio.run(main())
PY
```
Expected: imprime la MARCA (`carga-xxxxxxxx`). Anotarla: la limpieza del Step 6 borra por ella.

- [ ] **Step 2: Levantar la instancia aislada (en segundo plano)**

Run (con `run_in_background`):
```bash
cd /home/fruiz/worktrees/jax-prevuelo/las_manos && set -a && source /etc/jax/.env && set +a && export JAX_DB_NAME=jax_memory_test && \
/home/fruiz/jax/las_manos/.venv/bin/python -c 'import uvicorn; from fastapi import FastAPI; from jacobs.routes import router; app = FastAPI(); app.include_router(router); uvicorn.run(app, host="127.0.0.1", port=17790, log_level="warning")'
```
Solo el router de Jacobs: sin reaper, sin tocar :7777 ni :8080.

- [ ] **Step 3: Verificar el camino feliz sin sondas**

```bash
mkdir -p /tmp/claude-1000 && cat > /tmp/claude-1000/preflight-carga.json <<'JSON'
{"invoked_by": "plataforma", "objective": "prueba de carga del pre-vuelo",
 "steps": [
  {"facet": "hipatia", "capability": "research", "prompt": "investigá la ley"},
  {"facet": "jekyll", "capability": "analysis", "prompt": "analizá", "depends_on": [0]},
  {"facet": "thot", "capability": "critique", "prompt": "criticá el análisis", "depends_on": [1]},
  {"facet": "kimi", "capability": "generate", "prompt": "generá el informe", "depends_on": [0, 1, 2]}
 ]}
JSON
curl -s -X POST http://127.0.0.1:17790/jacobs/preflight -H 'Content-Type: application/json' --data @/tmp/claude-1000/preflight-carga.json | python3 -m json.tool | head -40
```
Expected: `"ok": true` y `"sondeadas": []`. Si aparece `sin_contrato_de_salida` o `faceta_inexistente`: PARAR y reportar (la semilla de la base de tests no alcanza; no se inventan filas de `model`). Si una capability no existe: cambiarla en el JSON por una de `SELECT key FROM capability` y anotar cuál. Un `sondeadas` no vacío invalida la medición (habría llamado a un proveedor): PARAR.

- [ ] **Step 4: Carga a c=1, 10 y 25**

```bash
cd /home/fruiz/worktrees/jax-prevuelo && for c in 1 10 25; do \
python3 scripts/load_test.py --url http://127.0.0.1:17790/jacobs/preflight -X POST -c $c -n $((c*40)) \
  -H 'Content-Type: application/json' --cuerpo "$(cat /tmp/claude-1000/preflight-carga.json)"; done
```
Expected: tres JSON con rps, p50/p95/p99 y `errores: 0`. Anotarlos. Si hay errores o el p95 a c=25 supera 10 veces el de c=1, NO hay GO: se investiga con EXPLAIN y perfilado antes de seguir.

- [ ] **Step 5: `continue` concurrente sobre el mismo pipeline**

Crear un pipeline abortado cuyo único paso a correr es `hyde` (la corrida ganadora para en el gate humano, sin despacho a ningún proveedor):
```bash
cd /home/fruiz/worktrees/jax-prevuelo && set -a && source /etc/jax/.env && set +a && export JAX_DB_NAME=jax_memory_test PYTHONPATH=.:las_manos && \
/home/fruiz/jax/las_manos/.venv/bin/python - <<'PY'
import asyncio, time, uuid
from jacobs import store
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus
async def main():
    await store.init_tables()
    pid = str(uuid.uuid4())
    ref = 'inline:{"result": "hecho"}'
    pasos = [
        Step(pipeline_id=pid, step_index=0, facet="jekyll", capability="research", input={"prompt": "p"},
             status=StepStatus.completed, output_ref=ref),
        Step(pipeline_id=pid, step_index=1, facet="hyde", capability="research", input={"prompt": "p"},
             depends_on=[0], status=StepStatus.failed, error="cortado"),
    ]
    ahora = time.time()
    await store.pipeline_create(Pipeline(pipeline_id=pid, name="carga-continue", invoked_by="plataforma",
        mode="autonomous", status=PipelineStatus.aborted, plan=pasos,
        context={"objective": "o", "step_0_ref": ref}, created_at=ahora, updated_at=ahora))
    for paso in pasos:
        await store.step_upsert(paso)
    print(pid)
asyncio.run(main())
PY
```
Con el PID impreso en la variable `PID`:
```bash
cd /home/fruiz/worktrees/jax-prevuelo && \
python3 scripts/load_test.py --url "http://127.0.0.1:17790/jacobs/pipeline/$PID/continue" -X POST -c 25 -n 25 \
  -H 'Content-Type: application/json' --cuerpo '{"invoked_by": "plataforma"}'; \
set -a && source /etc/jax/.env && set +a && export JAX_DB_NAME=jax_memory_test PYTHONPATH=.:las_manos && \
/home/fruiz/jax/las_manos/.venv/bin/python - "$PID" <<'PY'
import asyncio, sys
from jacobs import store
async def main(pid):
    print(await store.pipeline_epoca_y_status(pid))
    print([e["event_type"] for e in await store.events_by_pipeline(pid)])
asyncio.run(main(sys.argv[1]))
PY
```
Expected: el arnés informa `errores: 24` (24 respuestas 409; cuenta todo 4xx como error y sale con 1, esperado acá); la época queda en `(1, <PipelineStatus.interrupted: 'interrupted'>)` (la corrida ganadora paró en el gate de hyde); en los eventos hay exactamente UN `PIPELINE_CONTINUED` y un `PIPELINE_INTERRUPTED`, sin `STEP_STARTED`. Anotar el p95 de las 25.

- [ ] **Step 6: Limpiar la base de tests y bajar la instancia**

Detener el proceso en segundo plano del Step 2. Luego, con `MARCA` del Step 1 y `PID` del Step 5:
```bash
cd /home/fruiz/worktrees/jax-prevuelo && set -a && source /etc/jax/.env && set +a && export JAX_DB_NAME=jax_memory_test PYTHONPATH=.:las_manos && \
/home/fruiz/jax/las_manos/.venv/bin/python - "$MARCA" "$PID" <<'PY'
import asyncio, sys
from jacobs import store
async def main(marca, pid):
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT DATABASE()")
            assert (await cur.fetchone())[0] == "jax_memory_test"
            await cur.execute("DELETE FROM credential WHERE env_key=%s", (marca,))
            await cur.execute("DELETE FROM facet_health_event WHERE detail=%s", (marca,))
            for tabla in ("jacobs_steps", "jacobs_events", "jacobs_pipelines"):
                await cur.execute(f"DELETE FROM {tabla} WHERE pipeline_id=%s", (pid,))
    finally:
        conn.close()
asyncio.run(main(*sys.argv[1:]))
PY
rm -f /tmp/claude-1000/preflight-carga.json
```
Expected: sale 0.

- [ ] **Step 7: Registrar en la Biblioteca y commit**

Agregar al final de `CONTEXT.md` §9 una entrada con este formato, completando CADA número con lo medido en los Steps 4 y 5 (una entrada con un número sin completar no se commitea):
```markdown
- **2026-09-17 · Pre-vuelo y continuar (Jacobs) — carga medida** (VERDAD OPERACIONAL; caduca si
  cambia el esquema del catálogo, el volumen de `facet_health_event` o la máquina). Instancia aislada
  :17790 sobre `jax_memory_test`, catálogo y salud frescos (sin sondas), plan de 4 pasos (hipatia,
  jekyll, thot, kimi): c=1 → R1 rps, p95 P1 ms; c=10 → R10 rps, p95 P10 ms; c=25 → R25 rps,
  p95 P25 ms, p99 Q25 ms, 0 errores. 25 `continue` concurrentes sobre el mismo pipeline: 1 ganó,
  24 recibieron 409, época final 1, un solo PIPELINE_CONTINUED; p95 PC ms. Medido por QUIÉN con la
  Task 15 de `docs/superpowers/plans/2026-09-17-prevuelo-y-continuar-jacobs.md`.
```

```bash
cd /home/fruiz/worktrees/jax-prevuelo && pwd && git branch --show-current && \
git -C /home/fruiz/worktrees/jax-prevuelo add CONTEXT.md && \
git -C /home/fruiz/worktrees/jax-prevuelo commit -m "docs(context): carga del pre-vuelo y continue concurrente medidos

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Cobertura del spec (self-review)

| Spec | Task |
|---|---|
| §4.1 entrada/salida, REGLAS, pendientes, hyde, assemble | 5, 8 |
| §4.2 resolución por paso, una lectura por pre-vuelo, motores | 6, 8 |
| §4.3 credencial viva leída de la tabla | 5, 6 |
| §4.4 tope efectivo, `sin_contrato_de_salida`, `tope_insuficiente`, semilla medida | 5, 13 |
| §4.5 salud de nivel proveedor, sonda, `source='preflight'`, `preflight_probe`, paralelo, timeout propio | 1, 6, 7, 8 |
| §4.6 costo máximo, multiplicadores, precio NULL no acotado, `jax_local`, divisor configurable | 1, 5, 8 |
| §4.7 `/jacobs/preflight`, crear con 422/409/evento, dry_run, `_from_objective` | 9 |
| §5.1 contrato de continue y continue/preflight | 10, 11 |
| §5.2 reglas 1-10 | 10 |
| §5.3 época: columna, escrituras condicionales, relectura por ola, RUN_SUPERSEDED, un PIPELINE_ABORTED, resume incrementa, step_upsert con facet, plan reescrito | 2, 3, 4, 10 |
| §5.4 relanzador sobre continuar.py | 12 |
| §8 503, timeout de sonda, registro de sonda fail-soft contado, carreras continue/continue y cancel/continue | 3, 7, 9, 10, 11 |
| §9 tests puros y DB, canarios de CI, carga c=25 aislada, continue concurrente, EXPLAIN | 1-15 |
| §10 rebase preservando frentes E/F | 14 |

Fuera de este plan (plan P, jax-platform): DDL de `capability.min_output_tokens`, ENUM `facet_health_event.source='preflight'`, `request_type` en la Mesa, ajuste `pipeline_confirmar_usd`, backend y frontend de la Mesa (§6), datos A y F (§7), rescate de `ef9b2d6e` (§11, después del deploy).

---

## Semilla medida de min_output_tokens

**La medición NO se repitió en esta rama (Ruling R2 del ledger de este
plan).** El brief de la Task 13 pedía correr `scripts/medir_min_output_tokens.py`
contra producción "con GO del controlador"; el controlador confirmó que el
plan P (jax-platform) ya la había medido una vez, el mismo día, contra la
misma base (`jax_memory`), con el mismo criterio de solo lectura. Medir dos
veces no cambia el dato y sí duplica el riesgo de tocar producción sin
necesidad — "el que supone se equivoca" corre en el sentido contrario acá:
suponer que hace falta medir de nuevo sería el error. Esta sección copia,
con referencia, lo que P ya midió y verificó.

- **Quién midió:** plan P (jax-platform), Task 2.
- **Cuándo:** 2026-09-17, contra `jax_memory` (producción), sesión `SET
  SESSION TRANSACTION READ ONLY` + `START TRANSACTION READ ONLY` + `ROLLBACK`
  al final — sin escritura alguna.
- **Commit:** `d79b0f9` en `/home/fruiz/worktrees/jax-platform-prevuelo`
  (rama `feat/prevuelo-y-continuar-mesa`), mensaje `feat(prevuelo): semilla
  medida de min_output_tokens y topes de salida`.
- **Ledger de origen:** `/home/fruiz/worktrees/jax-platform-prevuelo/.superpowers/sdd/2026-09-17-prevuelo-y-continuar-mesa/progress.md`,
  bloque "MEDICIÓN DE SEMILLAS" (línea 61), y `task-2-report.md` del mismo
  directorio (tabla completa, evidencia TDD y preocupaciones).

### Valores medidos

```python
MIN_OUTPUT_TOKENS_MEDIDOS_2026_09_17 = {
    "analysis": 16384, "critique": 13312, "design": 14336, "file_write": 2048,
    "generate": 14336, "reconcile": 21504, "research": 7168, "validate_consistency": 3072,
}
```

Sin corridas medibles, quedan en 0 (sin mínimo, no bloquean): `architecture_review`,
`bug_hunt`, `code_swarm`, `file_read`, `implementation`, `pipeline_analysis`,
`reason`, `refactor`, `review`.

### Topes de salida agregados (§7 F, GET de metadata de Gemini, sin costo)

`gemini/gemini-2.5-flash` → 65536. `gemini/gemini-3.8-flash` → 65536 (este
último es el binding primario de `hipatia` en producción; su fila en
`model` tenía `max_output_tokens` NULL antes de esta medición).

### Método de P que `scripts/medir_min_output_tokens.py` sigue (fix rounds 1-2: alineado, no aproximado)

Este script sigue el método real de P en TRES respectos. De los tres, sólo
DOS eran divergencias de la primera versión (commit `a215959`) — la ventana
del JOIN y la exclusión de filas ambiguas —, encontradas en la revisión
(`git -C jax-platform-prevuelo show d79b0f9`, brief y salida de Task 2 de P)
y corregidas en el commit siguiente. El COLLATE (punto 1) NO fue una
divergencia: ya estaba correcto desde el commit original de la Task 13 (era
parte del alcance desde el principio, no un fix de esta ronda); se lista acá
porque también es parte del método de P que el script sigue.

1. **COLLATE del JOIN (error 1267) — correcto desde el commit original,
   `a215959`, no un fix de esta ronda.** `axioma_usage.facet` quedó en
   `utf8mb4_uca1400_ai_ci` y `jacobs_steps.facet` en `utf8mb4_unicode_ci`
   (esquemas creados en momentos distintos): un JOIN `ON u.facet = s.facet`
   falla contra producción con `pymysql.err.OperationalError: (1267,
   "Illegal mix of collations...")`. `_SQL_HTTP_DIRECTO` de este script lleva
   `ON u.facet = s.facet COLLATE utf8mb4_uca1400_ai_ci`; las claves de faceta
   son ASCII en minúscula, así que la igualdad da lo mismo con cualquiera de
   las dos collations. Verificado con un test puro
   (`tests/test_medir_min_output_tokens.py::test_el_join_http_directo_lleva_la_collate_del_esquema_real`).
2. **Ventana del JOIN, IDÉNTICA a la de P — corregida en fix round 1.** P usa
   `UNIX_TIMESTAMP(u.created_at) BETWEEN FLOOR(s.started_at) AND
   CEIL(s.finished_at) + 5` — holgura SÓLO del lado derecho (`created_at` es
   `TIMESTAMP`, segundos enteros; `started_at`/`finished_at` son `DOUBLE`;
   FLOOR/CEIL evitan perder una fila por el truncamiento). La primera
   versión de este script restaba holgura TAMBIÉN del lado izquierdo
   (`s.started_at - %s`), una ventana más ancha que la de P — no era una
   adaptación de esquema, era una divergencia de comportamiento que subía la
   probabilidad de fila ambigua sin necesidad. Corregido a `FLOOR(s.started_at)
   AND CEIL(s.finished_at) + %s`, con un test puro que lo fija.
3. **Exclusión de filas ambiguas (P la hace; la primera versión NO la
   hacía) — corregida en fix round 1.** P no arma el máximo con
   `MAX()+GROUP BY` en SQL: trae `(u.id, s.capability, u.tokens_out)` fila
   por fila y excluye en Python las filas cuyo `id` de `axioma_usage` cae en
   la ventana de pasos de MÁS de una capability a la vez ("filas ambiguas",
   **0 encontradas en la medición real de P** — su propia salida dice
   `filas de uso ambiguas excluidas: []`). Sin esa exclusión, un
   `MAX()+GROUP BY` le atribuye una fila compartida a TODAS las capabilities
   cuya ventana la toca — no sólo la infla hacia arriba (lado "seguro" de un
   mínimo), se la puede atribuir ENTERA a una capability que no la generó.
   **Ejemplo HIPOTÉTICO** (construido para el test que fija el mecanismo, NO
   observado en la medición real de P — que dio cero filas ambiguas): SI un
   paso `reconcile` de la faceta `thot` terminara en t=100 con una fila de
   20664 tokens y un paso `file_write` de la MISMA faceta arrancara en
   t=103, sin exclusión `file_write` mediría 20664/21504 en vez de su valor
   REAL medido por P, 1301/2048. `maximos_http()` reproduce el
   post-procesamiento de P: agrupa por `id`, excluye `len(capabilities) > 1`,
   descarta `tokens_out == 0`, toma el máximo de lo que queda. Cinco tests
   puros lo cubren: fila ambigua excluida con el ejemplo hipotético de
   arriba, máximo sin ambiguas con cero descartado, filas vacías, y un
   control que fija que `_SQL_HTTP_DIRECTO` NO contiene `GROUP BY` (el
   agrupado es post-procesamiento en Python, no SQL).

Los DOS puntos de comportamiento (2 y 3, ventana y exclusión de ambiguas) se
vieron en ROJO contra el commit `a215959` antes de corregirse; el COLLATE
(punto 1) no, porque ya estaba bien desde el commit original — nunca fue
rojo. Evidencia completa: `task-13-report.md`, secciones "Fix round 1" y
"Fix round 2". Ninguno de los tres queda como "adaptación aceptada" — el
script sigue el método de P tal cual, no una aproximación propia.

Fix round 2 (revisión, 2026-09-17) además endureció la prueba de solo
lectura (detecta `.query(...)` además de `.execute`/`.executemany`/
`.callproc`/`.commit`, una referencia guardada en variable sin llamar de
inmediato, y una sentencia con `;` que esconde una segunda), movió el
`ROLLBACK` a su propio `finally` (corre aunque la consulta falle, con
`conn.close()` después) y acotó el resolutor de "constantes de módulo" al
nivel de módulo real (no cualquier asignación del árbol). Detalle completo
en `task-13-report.md`, sección "Fix round 2".

### Consumo

Los `UPDATE capability SET min_output_tokens=... WHERE key=...` que
`scripts/medir_min_output_tokens.py` imprime (y que este documento fija con
los valores de arriba) los consume la migración de P: `backend/db/migrations.py::_semilla_min_output_tokens_v1`
(marcador `capability_min_output_tokens_v1` en `axioma_migracion_de_datos`,
una sola vez).
