# Frente F · Contrato de sub-pipelines (Ada multiagente) — Implementation Plan

> ## ENMIENDA DEL CONTROLLER — GO de Fernando, 2026-09-16 (prevalece sobre todo el plan)
> Alineación con el diseño del emisor de Ada (sesión de diseño del mismo día):
> 1. **`JAX_MAX_SUBPIPELINE_DEPTH` por defecto = 3**, rango [1, 5]. Todo test, harness, k6,
>    `.env` y verificación que asuma 1 se escribe con 3 (los tests de "excede el máximo"
>    construyen un padre en depth=3 y piden un hijo de depth 4).
> 2. **"Padre activo" = `jacobs_pipelines.status='running'` del padre, SOLAMENTE.** El token
>    queda atado a `parent_step` (el step de Ada que produjo la delegación), **en cualquier
>    estado**: en el modo "plan de delegación" Jacobs emite los tokens DESPUÉS de que el step
>    `delegate` terminó. Se elimina toda condición `s.status='running'` / `paso_status` de la
>    emisión y del consumo (SQL, diagnóstico y tests); se CONSERVA la verificación de que
>    `parent_step` existe, pertenece a `parent_pipeline_id` y es de la faceta `ada`. Test nuevo:
>    padre `running` + step de Ada `completed` → emisión y consumo aceptados; padre `completed`
>    → rechazados.


> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que un pipeline con `invoked_by="ada"` solo se cree presentando un `subpipeline_token` emitido por el servidor para un padre y un paso de Ada que están corriendo; que ese token se consuma de forma atómica una sola vez, y que la profundidad del hijo salga de la fila del token y nunca del cuerpo del pedido.

**Architecture:** Módulo nuevo `jacobs/subpipelines.py` con la config (entorno, validada al arrancar LAS MANOS), la emisión del lado del servidor (`emitir_token_subpipeline`, sin ruta HTTP) y el consumo (`consumir_token_subpipeline`). `jacobs/store.py` gana la tabla `jacobs_subpipeline_tokens` (se guarda solo el sha256), las columnas `parent_pipeline_id`/`depth` en `jacobs_pipelines` (las dos por `init_tables()`, idempotente) y dos sentencias atómicas: un `INSERT … SELECT` para emitir y un `UPDATE … JOIN` para consumir, que afecta una fila o ninguna. `jacobs/routes.py::create_pipeline` consume el token dentro del candado de creación, justo después de `validate_create` (que pasa a controlar solo la forma por rol) y antes de planificar.

**Tech Stack:** Python 3.12, FastAPI 0.139, Pydantic 2, aiomysql 0.3.2, MariaDB (12.3.3 en producción en `127.0.0.1:3308`, 11.8 en CI), pytest, k6 v2.2.0 (`~/bin/k6`).

**Spec:** `/home/fruiz/worktrees/jax-hallazgos-docs/docs/superpowers/specs/2026-09-16-hallazgos-auditoria-jax-design.md`, sección **F**. Las reglas comunes vienen de `/home/fruiz/worktrees/jax-platform-hallazgos-docs/docs/superpowers/specs/2026-09-16-hallazgos-auditoria-design.md` §0.

**Rama y worktree de ejecución:** `feat/contrato-subpipelines` en `/home/fruiz/worktrees/jax-frente-f` (repo `/home/fruiz/jax`, remoto `fjruizhn/Jax`, base `master` en `bd95237` o en lo que haya en master después de los frentes E y B; ver **Solapamientos**).

---

## Discrepancias con el spec

Cada una se verificó contra el código en `bd95237`. El plan sigue el código real y lo dice acá.

1. **"Validación en `validate_create`": no puede ser atómica ahí.** `jacobs/policy.py:32-78` es una función **síncrona y pura**: no toca la base. Consumir un token de forma atómica exige un `UPDATE` en la base. Por eso:
   - `validate_create` controla la **forma** según el rol: `ada` tiene que traer token y padre, y los demás roles no pueden traer ninguno de los dos.
   - La **validez** del token la decide `subpipelines.consumir_token_subpipeline()` (async, un solo `UPDATE`), que se llama en `routes.create_pipeline` justo después de `validate_create`, dentro del mismo `_pipeline_create_lock` (`routes.py:145`).
   - `validate_create` no tiene otros llamadores (`grep -rn validate_create`: solo `routes.py:147`).
2. **El candado de hoy es solo nominal (el hallazgo, confirmado):**
   - `policy.py:48-52` acepta **cualquier string no vacío** como token.
   - `routes.py:147-153` **nunca le pasa `subpipeline_depth`**: la profundidad vale siempre 0 y el control de `policy.py:69-73` es código muerto.
   - `MAX_SUBPIPELINE_DEPTH = 1` (`policy.py:18`) es un literal. El plan lo reemplaza por `JAX_MAX_SUBPIPELINE_DEPTH`.
3. **Tiempos: `DOUBLE` (epoch) en vez de `NOW()`.** Todas las tablas de Jacobs guardan el tiempo como `DOUBLE` con `time.time()` (`store.py:153-154`, `217`). jax-platform documenta que los `DATETIME` dependen de la zona horaria de la sesión (`jax-platform/backend/db/migrations.py:545`). `emitido_at`, `vence_at` y `usado_at` son `DOUBLE`, y el `UPDATE` recibe `usado_at=%s` con `time.time()`.
4. **"Token de otro padre" no se puede detectar con el cuerpo que describe el spec.** Si el padre sale solo de la fila del token, un token de P1 presentado desde el contexto de P2 crearía un hijo de P1 sin que nada lo note. El pedido de Ada declara `parent_pipeline_id` y tiene que coincidir con el de la fila; esa condición va dentro del `WHERE` del `UPDATE`.
   - Declarar el padre **no** da autoridad: la profundidad y el padre que se guardan salen de la fila.
   - Un rechazo por padre equivocado **no quema** el token. Si lo quemara, cualquiera que viera pasar un token podría dejar sin hijo al pipeline legítimo.
5. **"El padre sigue activo" se define así:** `jacobs_pipelines.status='running'` (ver ENMIENDA: el paso emisor puede estar en cualquier estado).
   - Un padre `interrupted` (pausa de supervised o de Hyde) no tiene ningún paso de Ada corriendo.
   - La emisión exige además que `jacobs_steps.facet='ada'`.
6. **`jacobs_pipelines` no tiene columna de profundidad.** El spec dice "el hijo guarda `parent_pipeline_id` y `depth`". Las dos columnas se crean en `init_tables()`, que es la dueña real de esa tabla: la crea jax (`store.py:137`). El CI de jax-platform la levanta clonando jax y corriendo **su** `init_tables()` (`jax-platform/.github/workflows/policy.yml:564-643`). Las filas existentes quedan con `depth=0` y `parent_pipeline_id=NULL`.
7. **`POST /jacobs/plan` acepta `invoked_by="ada"` sin token** (`routes.py:106`), y planificar llama a un LLM. El spec no lo menciona, pero es el mismo agujero por otra puerta. Ada no planifica por `/plan`: responde 403.
8. **jax-platform reenvía el cuerpo del cliente casi tal cual** (`jax-platform/backend/api/pipelines.py:121-128`). Solo pisa `user_id`, `tenant_id` e `invoked_by="plataforma"`, así que un cliente podría mandar `subpipeline_token` o `parent_pipeline_id`. **jax-platform no se toca:**
   - Con este plan, `plataforma` + token o padre → **422** en Jacobs, un rechazo visible. Borrarlos en silencio en la plataforma escondería el intento.
   - `depth`/`subpipeline_depth` en el cuerpo se ignoran porque ningún código los lee, y hay un test que lo prueba.
   - El frontend no envía ninguno de esos campos (`grep -rn "subpipeline\|parent_pipeline" jax-platform/frontend/src`: 0).
9. **Nombre de columna:** la PK del spec se llama `hash`; en el plan es `token_hash`, para no chocar con la función `HASH` ni confundir en los `JOIN`.
10. **Índice por `parent_pipeline_id`:** en este frente ninguna consulta filtra por esa columna (todas van por PK). Se crea igual porque lo pide el spec y porque el emisor (listar o cortar los hijos de un padre) lo va a usar. Queda anotado para que no parezca un índice huérfano.

## Hechos verificados que fijan el diseño

- **Conexiones:** Jacobs abre una conexión por llamada, sin pool (`store.py:51-55`), con `autocommit=True` (`store.py:47`) y sin `SET TRANSACTION ISOLATION`, así que rige el valor del servidor. Cada sentencia del contrato es un único statement autocommit.
- **Atomicidad del consumo:** el `UPDATE … WHERE usado_at IS NULL` toma el candado de fila de la PK. En `REPEATABLE-READ` y en `READ-COMMITTED`, un segundo `UPDATE` concurrente espera ese candado y después relee la versión confirmada, así que ve `usado_at` ya puesto y afecta 0 filas. La Task 5 lo prueba forzando el intercalado con dos sesiones reales, y la Task 1 mide el nivel de aislamiento real.
- **Candado de creación:** `_pipeline_create_lock` es de proceso (`routes.py:56`). Serializa los consumos dentro de LAS MANOS; la base cubre el caso de más de un proceso.
- **Kill switch:** se lee en `validate_create` (`policy.py:75`) y antes de cada ola (`executor.py:1122`). Un hijo corre por el mismo `run_pipeline`, así que respeta el mismo interruptor. El orden del plan asegura que un 423 del kill switch **no consume** el token.
- **La emisión no tiene ruta HTTP.** Las rutas de Jacobs no tienen autenticación propia (`routes.py`: ningún `Depends`). Una ruta de emisión permitiría que cualquier proceso que llegue a `127.0.0.1:7777` fabrique tokens. El emisor va a correr dentro del proceso de Jacobs, en la ejecución del paso de Ada, y llama a la función directamente. Un test vigila que no aparezca una ruta con `token` o `subpipeline` en el path.
- **Retención de tokens:** la tabla crece al ritmo de los sub-pipelines y `jacobs_pipelines` tampoco se poda. Todas las consultas van por PK. No se agrega poda en este frente; queda declarado.
- **Sin caché:** la config son dos `os.environ.get` por llamada. No hay nada caro que cachear ni que invalidar.

## Global Constraints

Copiadas de §0 del spec común (aplican a todas las tareas):

- TDD: test rojo contra el código viejo antes del arreglo (un control que no falla no valida).
- i18n: ningún texto visible literal; es/en en paridad. Dark/light con tokens. Nada de `confirm/alert/prompt`. *(Este frente no toca UI: sin textos visibles nuevos.)*
- Sin hardcoding: config en `/etc/jax/.env` o en DB.
- Fail-closed. Todo caché declara su invalidación en el mismo commit.
- Las cuatro del rendimiento: índice verificado con EXPLAIN sobre la consulta real, nada bloqueante en `async def`, **prueba de carga con número registrado** para todo endpoint nuevo o modificado en camino de usuario.
- Tests con barrera de DB de producción (`conftest.py` fuerza `jax_memory_test`). En este repo el patrón es: si `JAX_DB_NAME` está seteada y no es `jax_memory_test`, `RuntimeError`; si no, se fuerza `jax_memory_test` (`jacobs/_direct_usage_test.py:25-35`).
- CI: todo test nuevo lo corre un job; se verifica rompiéndolo (rojo sobre el sha real por API).
- Mirror-sync: si se toca un símbolo espejado (`jax/scripts/check_mirror_sync.py`), el cambio es de los dos repos en el mismo paso.
- Deploy: backend `sudo systemctl restart jax-las-manos.service` (Jacobs vive en LAS MANOS, `las_manos/server.py:284`) con 0 pipelines en vuelo.
- Registro en la Biblioteca (`jax/DEUDA.md` y `jax/CONTEXT.md`) antes de cerrar.

Propias de este frente:

- Variables nuevas: `JAX_SUBPIPELINE_TOKEN_TTL_SECONDS` (por defecto 300, rango [10, 3600]) y `JAX_MAX_SUBPIPELINE_DEPTH` (por defecto 3, rango [1, 5]). Un valor inválido tumba el arranque de LAS MANOS.
- Token: `secrets.token_urlsafe(32)`. En la base solo se guarda `sha256(token)` en hex. Ningún evento, log ni `detail` HTTP lleva el token: los eventos usan `token_ref = sha256[:12]`.
- Eventos en `jacobs_events`: `SUBPIPELINE_TOKEN_EMITIDO`, `SUBPIPELINE_CREADO`, `SUBPIPELINE_RECHAZADO` (con `fase` ∈ {`emision`, `consumo`, `plan`} y `motivo`).
- Códigos HTTP de la creación de un hijo: token rechazado → **403**; forma inválida (validate_create) → **422**; kill switch → **423**.
- Variables de shell que usan los comandos (declararlas en cada terminal):
  ```bash
  WT=/home/fruiz/worktrees/jax-frente-f
  PY=/home/fruiz/jax/.venv/bin/python
  DBTEST='set -a; . /etc/jax/.env; set +a; export JAX_DB_NAME=jax_memory_test'
  ```
  `DBTEST` carga host, puerto y credenciales de la MariaDB real y **pisa** `JAX_DB_NAME` con la base de prueba antes de que arranque Python. Sin el `export` final, la barrera de los tests levanta `RuntimeError`, que es exactamente para lo que está.
- Ledger de la ejecución: `$WT/.superpowers/sdd/2026-09-16-contrato-subpipelines/progress.md`. Cada rojo visto se pega ahí con el comando y su salida.

---

## File Structure

| Archivo | Acción | Responsabilidad |
|---|---|---|
| `jacobs/subpipelines.py` | Crear | Config por entorno, `Motivo`, hash, clasificación pura de motivos, emisión y consumo (orquestan `store` y los eventos) |
| `jacobs/store.py` | Modificar | DDL de `jacobs_subpipeline_tokens`, columnas `parent_pipeline_id`/`depth`, SQL constantes `SQL_EMITIR_TOKEN`/`SQL_CONSUMIR_TOKEN`/`SQL_TOKEN_CONSUMIDO`/`SQL_DIAGNOSTICO_TOKEN`, funciones `subpipeline_*`, `pipeline_create`/`_row_to_pipeline` con las columnas nuevas |
| `jacobs/models.py` | Modificar | `INVOKER_ADA`; `Pipeline.parent_pipeline_id`/`depth`; `PipelineCreateRequest.parent_pipeline_id` y validación de forma por rol |
| `jacobs/policy.py` | Modificar | `validate_create` sin `subpipeline_depth`, con `parent_pipeline_id`; se borra `MAX_SUBPIPELINE_DEPTH` |
| `jacobs/routes.py` | Modificar | `/plan` rechaza `ada`; `create_pipeline` consume el token, guarda padre y profundidad, eventos |
| `las_manos/server.py` | Modificar | `_jacobs_init` valida `config_subpipelines()` antes de `init_tables()` |
| `jacobs/_arnes_ada.py` | Crear | Arnés de prueba que hace de Ada (padre, emisión, pedido de hijo, consultas de apoyo) con barrera de base |
| `tests/test_subpipeline_contrato_puro.py` | Crear | 18 tests sin base (job `tests-puros`) |
| `jacobs/_subpipeline_contrato_io_test.py` | Crear | 19 tests contra MariaDB: esquema, emisión, consumo, carreras, EXPLAIN |
| `tests/test_subpipeline_contrato_rutas.py` | Crear | 11 tests de punta a punta por la ruta con el arnés (ataques y camino legítimo) |
| `.github/workflows/policy.yml` | Modificar | Job nuevo `subpipeline-contrato-db` (piso 30) y `tests-puros` (+18) |
| `loadtest/jacobs_subpipelines_app.py` | Crear | App de carga aislada (solo `jax_memory_test`) |
| `loadtest/sembrar_tokens_subpipeline.py` | Crear | Siembra un padre corriendo y N tokens para la carga |
| `loadtest/jacobs_subpipelines.js` | Crear | Escenarios k6: `base`, `legitimo`, `inventado` |
| `DEUDA.md`, `CONTEXT.md` | Modificar | Biblioteca |

---

### Task 1: Preparación, arnés de Ada y prueba roja contra master

**Files:**
- Create: `jacobs/_arnes_ada.py`
- Create: `tests/test_subpipeline_contrato_rutas.py`

**Interfaces:**
- Produces (usado por las Tasks 4-8):
  - `jacobs._arnes_ada.plan_de_un_paso(pipeline_id, objective, max_steps, steps_spec) -> list[Step]` (async)
  - `padre_en_ejecucion(depth: int = 0, facet_del_paso: str = "ada") -> tuple[str, str]` (async; devuelve `(pipeline_id, step_id)`)
  - `cerrar(pipeline_id: str) -> None` (async)
  - `una_fila(sql: str, args: tuple = ()) -> dict | None`, `ejecutar(sql: str, args: tuple = ()) -> int`, `fila_token(token_hash: str) -> dict | None` (async)
  - `emitir(parent_pipeline_id: str, parent_step: str) -> str` (async; kill switch forzado apagado)
  - `pedir_hijo(token, parent_pipeline_id, *, cuerpo_extra=None, kill_switch=False, plan=plan_de_un_paso) -> dict` (async; llama a `routes.create_pipeline`)

- [ ] **Step 1: Crear el worktree y la rama**

```bash
git -C /home/fruiz/jax fetch origin
git -C /home/fruiz/jax log --oneline -1 origin/master
git -C /home/fruiz/jax worktree add /home/fruiz/worktrees/jax-frente-f -b feat/contrato-subpipelines origin/master
git -C /home/fruiz/worktrees/jax-frente-f branch --show-current
mkdir -p /home/fruiz/worktrees/jax-frente-f/.superpowers/sdd/2026-09-16-contrato-subpipelines
```
Expected: la última línea imprime `feat/contrato-subpipelines`. Anotar el sha base en `progress.md`. Si origin/master ya no es `bd95237`, releer `jacobs/policy.py`, `jacobs/models.py`, `jacobs/routes.py` y `las_manos/server.py` completos antes de seguir: los frentes E y B tocan esos archivos (ver **Solapamientos**).

- [ ] **Step 2: Verificar el venv y medir el nivel de aislamiento real**

```bash
$PY -c "import pytest, aiomysql, fastapi, cryptography, httpx, pydantic, yaml; print('deps ok', fastapi.__version__)"
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY - <<'PY'
import asyncio
from jacobs import store
async def main():
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute('SELECT DATABASE(), VERSION(), @@GLOBAL.transaction_isolation, @@SESSION.transaction_isolation')
            print(await cur.fetchone())
    finally:
        conn.close()
asyncio.run(main())
PY"
```
Expected: `deps ok 0.139.0` y una tupla `('jax_memory_test', '12.3.3-MariaDB…', 'REPEATABLE-READ', 'REPEATABLE-READ')`, o `READ-COMMITTED`. Pegar la salida en `progress.md`. Si sale otro nivel, **parar y avisar a Fernando**: el argumento de atomicidad de la Task 5 está escrito para esos dos.

- [ ] **Step 3: Escribir el arnés de Ada**

Create `jacobs/_arnes_ada.py`:

```python
"""Arnés que hace de Ada contra Jacobs, para las pruebas del contrato de
sub-pipelines (frente F, 2026-09-16).

Escribe pipelines, pasos, tokens y eventos: SOLO contra jax_memory_test. Si el
proceso ya trae JAX_DB_NAME apuntando a otra base (típico después de sourcear
/etc/jax/.env), se niega a importar en vez de escribir ahí en silencio.

Lo único que sustituye es lo que no es el contrato: el planificador (llama a un
LLM) y el conteo de pipelines activos (el límite de 3 no es lo que se prueba).
El token, la base y los eventos son los reales.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import os

_nombre = os.environ.get("JAX_DB_NAME")
if _nombre is not None and _nombre != "jax_memory_test":
    raise RuntimeError(
        f"JAX_DB_NAME={_nombre!r}: el arnés de Ada escribe pipelines, pasos, tokens y "
        "eventos. Solo corre contra jax_memory_test (exportala después de sourcear "
        "/etc/jax/.env)."
    )
os.environ["JAX_DB_NAME"] = "jax_memory_test"

import time  # noqa: E402
import uuid  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402

import aiomysql  # noqa: E402
from fastapi import BackgroundTasks  # noqa: E402

from jacobs import routes, store  # noqa: E402
from jacobs.models import (  # noqa: E402
    Pipeline,
    PipelineCreateRequest,
    PipelineStatus,
    Step,
    StepStatus,
)


async def plan_de_un_paso(pipeline_id, objective, max_steps, steps_spec):
    return [Step(facet="jekyll", capability="summarize", pipeline_id=pipeline_id)]


async def padre_en_ejecucion(depth: int = 0, facet_del_paso: str = "ada") -> tuple[str, str]:
    """Un pipeline `running` con un paso `running`: el único estado desde el que
    Ada puede pedir un hijo."""
    pid = str(uuid.uuid4())
    ahora = time.time()
    paso = Step(
        pipeline_id=pid, step_index=0, facet=facet_del_paso,
        capability="delegate", status=StepStatus.running,
    )
    await store.pipeline_create(Pipeline(
        pipeline_id=pid, name="arnes-ada-padre", invoked_by="plataforma",
        mode="autonomous", status=PipelineStatus.running, plan=[paso],
        depth=depth, created_at=ahora, updated_at=ahora,
    ))
    await store.step_upsert(paso)
    return pid, paso.step_id


async def cerrar(pipeline_id: str) -> None:
    await store.pipeline_update_status(pipeline_id, PipelineStatus.completed)


async def una_fila(sql: str, args: tuple = ()) -> dict | None:
    conn = await store.get_conn()
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(sql, args)
            return await cur.fetchone()
    finally:
        conn.close()


async def ejecutar(sql: str, args: tuple = ()) -> int:
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(sql, args)
            return cur.rowcount
    finally:
        conn.close()


async def fila_token(token_hash: str) -> dict | None:
    return await una_fila(
        "SELECT * FROM jacobs_subpipeline_tokens WHERE token_hash = %s", (token_hash,)
    )


async def emitir(parent_pipeline_id: str, parent_step: str) -> str:
    from jacobs import subpipelines

    with patch("jacobs.policy.check_kill_switch", return_value=False):
        return await subpipelines.emitir_token_subpipeline(parent_pipeline_id, parent_step)


async def pedir_hijo(
    token: str,
    parent_pipeline_id: str,
    *,
    cuerpo_extra: dict | None = None,
    kill_switch: bool = False,
    plan=plan_de_un_paso,
) -> dict:
    """Lo que haría Ada: POST /jacobs/pipeline con su token. El cuerpo pasa por
    `model_validate` igual que el JSON de un pedido HTTP real."""
    cuerpo = {
        "name": "arnes-ada-hijo", "objective": "o", "invoked_by": "ada",
        "mode": "dry_run", "subpipeline_token": token,
        "parent_pipeline_id": parent_pipeline_id,
    }
    cuerpo.update(cuerpo_extra or {})
    req = PipelineCreateRequest.model_validate(cuerpo)
    with patch.object(routes, "_build_plan_or_reject", AsyncMock(side_effect=plan)), \
         patch.object(store, "pipeline_count_active", AsyncMock(return_value=1)), \
         patch("jacobs.policy.check_kill_switch", return_value=kill_switch):
        return await routes.create_pipeline(req, BackgroundTasks())
```

- [ ] **Step 4: Escribir la primera prueba de ataque**

Create `tests/test_subpipeline_contrato_rutas.py`:

```python
"""Contrato de sub-pipelines de punta a punta por la ruta real (frente F, 2026-09-16).

El arnés (`jacobs/_arnes_ada.py`) hace de Ada: arma un padre `running` con un
paso de Ada `running`, emite tokens con la función de servidor y pide hijos por
`routes.create_pipeline`. Token, base y eventos son los reales.

ROJO CONTRA MASTER (bd95237): `test_ada_con_token_inventado_no_crea_pipeline`
falló con "DID NOT RAISE": `validate_create` aceptaba cualquier string no vacío
como token y creaba el pipeline.

Corre con:
  bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -v tests/test_subpipeline_contrato_rutas.py"
"""
from __future__ import annotations

import os

from jacobs import _arnes_ada as ada  # primero: barrera de base de prueba

import asyncio  # noqa: E402

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.getenv("JAX_DB_HOST"),
    reason="necesita la MariaDB real con JAX_DB_NAME=jax_memory_test",
)

TOKEN_INVENTADO = "token-inventado-por-el-atacante"


def test_ada_con_token_inventado_no_crea_pipeline():
    async def escenario():
        padre, _paso = await ada.padre_en_ejecucion()
        try:
            with pytest.raises(HTTPException) as rechazo:
                await ada.pedir_hijo(TOKEN_INVENTADO, padre)
            return rechazo.value
        finally:
            await ada.cerrar(padre)

    rechazo = asyncio.run(escenario())
    assert rechazo.status_code == 403
    assert "token_desconocido" in rechazo.detail
    assert TOKEN_INVENTADO not in rechazo.detail
```

- [ ] **Step 5: Correrla contra el código de master y verla fallar**

```bash
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -v tests/test_subpipeline_contrato_rutas.py" 2>&1 | tee -a $WT/.superpowers/sdd/2026-09-16-contrato-subpipelines/progress.md
```
Expected: `FAILED tests/test_subpipeline_contrato_rutas.py::test_ada_con_token_inventado_no_crea_pipeline - Failed: DID NOT RAISE <class 'fastapi.exceptions.HTTPException'>`. Es la prueba de que hoy un token inventado crea un pipeline. Si falla por otra causa (colección, conexión), arreglar el entorno y repetir: un rojo por la razón equivocada no vale.

- [ ] **Step 6: Commit (rojo a propósito; ningún job de CI lo corre todavía)**

```bash
git -C $WT add jacobs/_arnes_ada.py tests/test_subpipeline_contrato_rutas.py .superpowers/sdd/2026-09-16-contrato-subpipelines/progress.md
git -C $WT commit -m "test(jacobs): ada con token inventado crea un pipeline en master (rojo)

Arnés que hace de Ada contra la ruta real y la primera prueba de ataque del
contrato de sub-pipelines. Falla contra bd95237 con DID NOT RAISE.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Config, forma del pedido y política (sin base)

**Files:**
- Create: `jacobs/subpipelines.py`
- Create: `tests/test_subpipeline_contrato_puro.py`
- Modify: `jacobs/models.py:51-58` (invocadores), `jacobs/models.py:82-131` (`Pipeline`, `PipelineCreateRequest`)
- Modify: `jacobs/policy.py:12-78`
- Modify: `jacobs/routes.py:21-35` (imports), `jacobs/routes.py:97-115` (`plan_only`)
- Modify: `las_manos/server.py:245` (`_jacobs_init`)

**Interfaces:**
- Produces:
  - `jacobs.subpipelines.ENV_TTL = "JAX_SUBPIPELINE_TOKEN_TTL_SECONDS"`, `ENV_MAX_PROFUNDIDAD = "JAX_MAX_SUBPIPELINE_DEPTH"`
  - `ConfigSubpipelines(ttl_segundos: int, max_profundidad: int)` (dataclass frozen), `config_subpipelines() -> ConfigSubpipelines` (lanza `ValueError` con el nombre de la variable)
  - `Motivo(str, Enum)`: `TOKEN_DESCONOCIDO`, `TOKEN_USADO`, `TOKEN_VENCIDO`, `PADRE_NO_COINCIDE`, `PADRE_DESCONOCIDO`, `PADRE_INACTIVO`, `PASO_DESCONOCIDO`, `PASO_INACTIVO`, `PASO_NO_ES_ADA`, `PROFUNDIDAD_EXCEDIDA`, `KILL_SWITCH_ACTIVO`, `ESTADO_CAMBIO`, `PLAN_RECHAZADO` (valores en minúscula, iguales al nombre)
  - `TokenConsumido(parent_pipeline_id: str, parent_step: str, depth: int)`, `ConsumoRechazado(motivo: Motivo)` (dataclasses frozen), `EmisionRechazada(Exception)` con `.motivo`
  - `hash_token(token: str) -> str`, `token_ref(token_hash: str) -> str`
  - `motivo_emision(diagnostico: dict, parent_pipeline_id: str, max_profundidad: int) -> Motivo`
  - `motivo_consumo(diagnostico: dict | None, parent_pipeline_id: str, ahora: float, max_profundidad: int) -> Motivo`
  - `jacobs.models.INVOKER_ADA = "ada"`; `Pipeline.parent_pipeline_id: str | None = None`, `Pipeline.depth: int = 0`; `PipelineCreateRequest.parent_pipeline_id: str | None`
  - `policy.validate_create(invoked_by, mode, max_steps, active_count, subpipeline_token=None, parent_pipeline_id=None) -> PolicyResult`

- [ ] **Step 1: Escribir los tests puros**

Create `tests/test_subpipeline_contrato_puro.py`:

```python
"""Contrato de sub-pipelines: la parte que NO necesita base (frente F, 2026-09-16).

Vistos en rojo contra el código de master (Task 2, Step 4 del plan):
plataforma con token, ada sin token/padre en el request, la firma de
validate_create, /plan con ada y la validación de config al arrancar.

`test_hijo_respeta_el_kill_switch_antes_de_su_primera_ola` y
`test_la_emision_no_tiene_ruta_http` son GUARDAS: ya pasan con el código viejo
y se validan por mutación (Step 7), no por rojo.

Corre con (sin base):
  cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -v tests/test_subpipeline_contrato_puro.py
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from jacobs import executor, policy, routes
from jacobs import subpipelines as sp
from jacobs.models import Pipeline, PipelineCreateRequest, PipelineStatus, Step

_NO_PLANIFICAR = AsyncMock(side_effect=AssertionError("el test no debe llegar a planificar"))


def _sin_config(monkeypatch):
    monkeypatch.delenv(sp.ENV_TTL, raising=False)
    monkeypatch.delenv(sp.ENV_MAX_PROFUNDIDAD, raising=False)


def test_config_por_defecto_acotada(monkeypatch):
    _sin_config(monkeypatch)
    assert sp.config_subpipelines() == sp.ConfigSubpipelines(ttl_segundos=300, max_profundidad=3)


def test_config_lee_el_entorno(monkeypatch):
    _sin_config(monkeypatch)
    monkeypatch.setenv(sp.ENV_TTL, "90")
    monkeypatch.setenv(sp.ENV_MAX_PROFUNDIDAD, "2")
    assert sp.config_subpipelines() == sp.ConfigSubpipelines(ttl_segundos=90, max_profundidad=2)


def test_config_no_entera_falla(monkeypatch):
    _sin_config(monkeypatch)
    monkeypatch.setenv(sp.ENV_TTL, "cinco minutos")
    with pytest.raises(ValueError, match=sp.ENV_TTL):
        sp.config_subpipelines()


def test_config_fuera_de_rango_falla(monkeypatch):
    for nombre, valor in (
        (sp.ENV_TTL, "9"), (sp.ENV_TTL, "3601"),
        (sp.ENV_MAX_PROFUNDIDAD, "0"), (sp.ENV_MAX_PROFUNDIDAD, "4"),
    ):
        _sin_config(monkeypatch)
        monkeypatch.setenv(nombre, valor)
        with pytest.raises(ValueError, match=nombre):
            sp.config_subpipelines()


def test_config_vacia_falla_en_vez_de_tomar_el_defecto(monkeypatch):
    _sin_config(monkeypatch)
    monkeypatch.setenv(sp.ENV_MAX_PROFUNDIDAD, "")
    with pytest.raises(ValueError, match=sp.ENV_MAX_PROFUNDIDAD):
        sp.config_subpipelines()


def test_hash_token_es_sha256_hex():
    assert sp.hash_token("abc") == hashlib.sha256(b"abc").hexdigest()
    assert sp.token_ref(sp.hash_token("abc")) == hashlib.sha256(b"abc").hexdigest()[:12]


def test_plataforma_no_puede_presentar_token_ni_padre():
    for extra in ({"subpipeline_token": "x"}, {"parent_pipeline_id": "p"}):
        with pytest.raises(ValidationError):
            PipelineCreateRequest(
                name="t", objective="o", invoked_by="plataforma", mode="dry_run", **extra)


def test_ada_exige_token_y_padre():
    for extra in ({}, {"subpipeline_token": "x"}, {"parent_pipeline_id": "p"}):
        with pytest.raises(ValidationError):
            PipelineCreateRequest(name="t", objective="o", invoked_by="ada", mode="dry_run", **extra)
    req = PipelineCreateRequest(
        name="t", objective="o", invoked_by="ada", mode="dry_run",
        subpipeline_token="x", parent_pipeline_id="p",
    )
    assert (req.subpipeline_token, req.parent_pipeline_id) == ("x", "p")


def test_la_profundidad_nunca_la_pone_el_llamador():
    assert "subpipeline_depth" not in inspect.signature(policy.validate_create).parameters
    assert not hasattr(policy, "MAX_SUBPIPELINE_DEPTH")


def test_validate_create_rechaza_plataforma_con_token():
    with patch.object(policy, "check_kill_switch", return_value=False):
        r = policy.validate_create("plataforma", "dry_run", 3, 0, subpipeline_token="x")
    assert not r.ok
    assert "subpipeline_token" in r.reason


def test_validate_create_ada_sin_padre_se_rechaza():
    with patch.object(policy, "check_kill_switch", return_value=False):
        r = policy.validate_create("ada", "dry_run", 3, 0, subpipeline_token="x")
    assert not r.ok
    assert "parent_pipeline_id" in r.reason


def test_validate_create_ada_con_token_y_padre_pasa_solo_la_forma():
    with patch.object(policy, "check_kill_switch", return_value=False):
        r = policy.validate_create(
            "ada", "dry_run", 3, 0, subpipeline_token="x", parent_pipeline_id="p")
    assert r.ok, r.reason


def test_plan_no_acepta_ada():
    req = routes.PlanRequest(name="t", objective="o", invoked_by="ada", mode="dry_run")
    with patch.object(routes, "_build_plan_or_reject", _NO_PLANIFICAR), \
         patch.object(routes, "check_kill_switch", return_value=False), \
         pytest.raises(HTTPException) as rechazo:
        asyncio.run(routes.plan_only(req))
    assert rechazo.value.status_code == 403


def test_motivo_de_emision():
    base = {
        "padre_status": "running", "padre_depth": 0,
        "paso_status": "running", "paso_facet": "ada", "paso_pipeline_id": "P",
    }
    casos = [
        ({"padre_status": None, "padre_depth": None}, sp.Motivo.PADRE_DESCONOCIDO),
        ({"paso_status": None, "paso_facet": None, "paso_pipeline_id": None}, sp.Motivo.PASO_DESCONOCIDO),
        ({"paso_pipeline_id": "OTRO"}, sp.Motivo.PASO_DESCONOCIDO),
        ({"padre_status": "completed"}, sp.Motivo.PADRE_INACTIVO),
        ({"paso_facet": "jekyll"}, sp.Motivo.PASO_NO_ES_ADA),
        ({"paso_status": "completed"}, sp.Motivo.PASO_INACTIVO),
        ({"padre_depth": 1}, sp.Motivo.PROFUNDIDAD_EXCEDIDA),
        ({}, sp.Motivo.ESTADO_CAMBIO),
    ]
    for cambio, esperado in casos:
        assert sp.motivo_emision({**base, **cambio}, "P", 1) == esperado, cambio


def test_motivo_de_consumo():
    ahora = 1000.0
    base = {
        "usado_at": None, "vence_at": 2000.0, "parent_pipeline_id": "P",
        "depth_hijo": 1, "padre_status": "running", "paso_status": "running",
    }
    assert sp.motivo_consumo(None, "P", ahora, 1) == sp.Motivo.TOKEN_DESCONOCIDO
    casos = [
        ({"usado_at": 999.0}, sp.Motivo.TOKEN_USADO),
        ({"vence_at": 1000.0}, sp.Motivo.TOKEN_VENCIDO),
        ({"parent_pipeline_id": "OTRO"}, sp.Motivo.PADRE_NO_COINCIDE),
        ({"depth_hijo": 2}, sp.Motivo.PROFUNDIDAD_EXCEDIDA),
        ({"padre_status": "aborted"}, sp.Motivo.PADRE_INACTIVO),
        ({"padre_status": None}, sp.Motivo.PADRE_INACTIVO),
        ({"paso_status": "failed"}, sp.Motivo.PASO_INACTIVO),
        ({}, sp.Motivo.ESTADO_CAMBIO),
    ]
    for cambio, esperado in casos:
        assert sp.motivo_consumo({**base, **cambio}, "P", ahora, 1) == esperado, cambio


def test_hijo_respeta_el_kill_switch_antes_de_su_primera_ola():
    hijo = Pipeline(
        pipeline_id="hijo", name="h", invoked_by="ada", mode="autonomous",
        parent_pipeline_id="padre", depth=1,
        plan=[Step(pipeline_id="hijo", facet="jekyll", capability="summarize")],
    )
    eventos: list[str] = []

    async def _evento(pipeline_id, event_type, payload=None, step_id=None):
        eventos.append(event_type)

    with patch.object(executor.store, "pipeline_update_status", AsyncMock()) as estado, \
         patch.object(executor.store, "event_append", AsyncMock(side_effect=_evento)), \
         patch.object(executor.store, "step_upsert", AsyncMock()), \
         patch.object(executor, "check_kill_switch", return_value=True), \
         patch.object(executor, "_dispatch_step",
                      AsyncMock(side_effect=AssertionError("un hijo frenado no despacha"))):
        asyncio.run(executor.run_pipeline(hijo))
    assert "KILL_SWITCH_ABORTED" in eventos
    assert estado.await_args_list[-1].args[1] == PipelineStatus.aborted


def test_la_emision_no_tiene_ruta_http():
    rutas = [r.path for r in routes.router.routes]
    assert not [p for p in rutas if "token" in p or "subpipeline" in p], rutas


def test_las_manos_valida_la_config_al_arrancar():
    fuente = (Path(__file__).resolve().parents[1] / "las_manos" / "server.py").read_text(encoding="utf-8")
    inicio = fuente.index("async def _jacobs_init")
    fin = fuente.index("await jacobs_store.init_tables()", inicio)
    assert "config_subpipelines()" in fuente[inicio:fin]
```

- [ ] **Step 2: Correrlos y verlos fallar por importación**

```bash
cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_subpipeline_contrato_puro.py 2>&1 | tail -5
```
Expected: `ModuleNotFoundError: No module named 'jacobs.subpipelines'`. Es un error de colección; el rojo que cuenta viene en el Step 4.

- [ ] **Step 3: Crear la parte pura de `jacobs/subpipelines.py`**

Create `jacobs/subpipelines.py`:

```python
"""
Jacobs — Contrato de sub-pipelines (Ada multiagente).

Frente F del spec 2026-09-16 (docs/superpowers/specs/2026-09-16-hallazgos-auditoria-jax-design.md).
Hasta este frente, `validate_create` aceptaba para `invoked_by="ada"` CUALQUIER
string no vacío como `subpipeline_token`, y la profundidad nunca llegaba a la
política (routes.py no la pasaba): el candado existía en el nombre y no en el
efecto. Acá vive el contrato real:

- Emisión (solo servidor, SIN ruta HTTP: las rutas de Jacobs no autentican):
  `emitir_token_subpipeline()`.
- Consumo atómico: `consumir_token_subpipeline()` -- un UPDATE condicionado,
  una fila afectada o rechazo.
- La profundidad sale de la fila del token, nunca del cuerpo del pedido.

Sin caché: la config son dos os.environ.get por llamada, nada caro que valga la
pena guardar ni invalidar. Se valida también al arrancar LAS MANOS
(server.py::_jacobs_init): un valor inválido tumba el arranque.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from enum import Enum

ENV_TTL = "JAX_SUBPIPELINE_TOKEN_TTL_SECONDS"
ENV_MAX_PROFUNDIDAD = "JAX_MAX_SUBPIPELINE_DEPTH"

# 300 s cubre el peor caso medido de espera por el candado de creación
# (routes._pipeline_create_lock serializa build() de 20-40 s, hasta 3 en cola).
TTL_POR_DEFECTO = 300
MAX_PROFUNDIDAD_POR_DEFECTO = 1

# Límites del contrato, no preferencias: un TTL de horas deja vivo demasiado
# tiempo un token filtrado, y más de 3 niveles es un árbol de agentes que nadie
# audita a mano. Moverlos es una decisión de Fernando, no un ajuste de .env.
TTL_MINIMO, TTL_MAXIMO = 10, 3600
PROFUNDIDAD_MINIMA, PROFUNDIDAD_MAXIMA = 1, 3

TOKEN_BYTES = 32
LARGO_TOKEN_REF = 12


@dataclass(frozen=True)
class ConfigSubpipelines:
    ttl_segundos: int
    max_profundidad: int


def _entero_acotado(nombre: str, por_defecto: int, minimo: int, maximo: int) -> int:
    crudo = os.environ.get(nombre)
    if crudo is None:
        return por_defecto
    try:
        valor = int(crudo.strip())
    except ValueError as exc:
        raise ValueError(f"{nombre}={crudo!r} no es un entero") from exc
    if not minimo <= valor <= maximo:
        raise ValueError(f"{nombre}={valor} fuera de rango [{minimo}, {maximo}]")
    return valor


def config_subpipelines() -> ConfigSubpipelines:
    """Lee y valida la config. Variable ausente -> default acotado; presente
    pero vacía, no entera o fuera de rango -> ValueError (fail-closed)."""
    return ConfigSubpipelines(
        ttl_segundos=_entero_acotado(ENV_TTL, TTL_POR_DEFECTO, TTL_MINIMO, TTL_MAXIMO),
        max_profundidad=_entero_acotado(
            ENV_MAX_PROFUNDIDAD, MAX_PROFUNDIDAD_POR_DEFECTO,
            PROFUNDIDAD_MINIMA, PROFUNDIDAD_MAXIMA,
        ),
    )


class Motivo(str, Enum):
    TOKEN_DESCONOCIDO    = "token_desconocido"
    TOKEN_USADO          = "token_usado"
    TOKEN_VENCIDO        = "token_vencido"
    PADRE_NO_COINCIDE    = "padre_no_coincide"
    PADRE_DESCONOCIDO    = "padre_desconocido"
    PADRE_INACTIVO       = "padre_inactivo"
    PASO_DESCONOCIDO     = "paso_desconocido"
    PASO_INACTIVO        = "paso_inactivo"
    PASO_NO_ES_ADA       = "paso_no_es_ada"
    PROFUNDIDAD_EXCEDIDA = "profundidad_excedida"
    KILL_SWITCH_ACTIVO   = "kill_switch_activo"
    # La sentencia atómica dijo que no, pero el diagnóstico posterior ya no ve
    # la causa (el estado cambió entre las dos lecturas). Se rechaza igual.
    ESTADO_CAMBIO        = "estado_cambio"
    PLAN_RECHAZADO       = "plan_rechazado"


@dataclass(frozen=True)
class TokenConsumido:
    parent_pipeline_id: str
    parent_step: str
    depth: int


@dataclass(frozen=True)
class ConsumoRechazado:
    motivo: Motivo


class EmisionRechazada(Exception):
    def __init__(self, motivo: Motivo) -> None:
        super().__init__(motivo.value)
        self.motivo = motivo


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_ref(token_hash: str) -> str:
    """Lo único del token que puede ir a un evento o a un log."""
    return token_hash[:LARGO_TOKEN_REF]


def motivo_emision(diagnostico: dict, parent_pipeline_id: str, max_profundidad: int) -> Motivo:
    """Por qué el INSERT … SELECT de la emisión no insertó. `diagnostico` sale de
    store.subpipeline_emision_diagnostico()."""
    if diagnostico["padre_status"] is None:
        return Motivo.PADRE_DESCONOCIDO
    if diagnostico["paso_status"] is None or diagnostico["paso_pipeline_id"] != parent_pipeline_id:
        return Motivo.PASO_DESCONOCIDO
    if diagnostico["padre_status"] != "running":
        return Motivo.PADRE_INACTIVO
    if diagnostico["paso_facet"] != "ada":
        return Motivo.PASO_NO_ES_ADA
    if diagnostico["paso_status"] != "running":
        return Motivo.PASO_INACTIVO
    if diagnostico["padre_depth"] + 1 > max_profundidad:
        return Motivo.PROFUNDIDAD_EXCEDIDA
    return Motivo.ESTADO_CAMBIO


def motivo_consumo(
    diagnostico: dict | None, parent_pipeline_id: str, ahora: float, max_profundidad: int,
) -> Motivo:
    """Por qué el UPDATE del consumo afectó 0 filas. `diagnostico` sale de
    store.subpipeline_token_diagnostico() (None = el hash no existe)."""
    if diagnostico is None:
        return Motivo.TOKEN_DESCONOCIDO
    if diagnostico["usado_at"] is not None:
        return Motivo.TOKEN_USADO
    if diagnostico["vence_at"] <= ahora:
        return Motivo.TOKEN_VENCIDO
    if diagnostico["parent_pipeline_id"] != parent_pipeline_id:
        return Motivo.PADRE_NO_COINCIDE
    if diagnostico["depth_hijo"] > max_profundidad:
        return Motivo.PROFUNDIDAD_EXCEDIDA
    if diagnostico["padre_status"] != "running":
        return Motivo.PADRE_INACTIVO
    if diagnostico["paso_status"] != "running":
        return Motivo.PASO_INACTIVO
    return Motivo.ESTADO_CAMBIO
```

- [ ] **Step 4: Correr contra models/policy/routes/server viejos: rojo por aserción**

```bash
cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_subpipeline_contrato_puro.py 2>&1 | tee -a .superpowers/sdd/2026-09-16-contrato-subpipelines/progress.md | tail -15
```
Expected: `8 failed, 10 passed`. Tienen que fallar exactamente estos:
- `test_plataforma_no_puede_presentar_token_ni_padre`: DID NOT RAISE
- `test_ada_exige_token_y_padre`: DID NOT RAISE
- `test_la_profundidad_nunca_la_pone_el_llamador`
- `test_validate_create_rechaza_plataforma_con_token`
- `test_validate_create_ada_sin_padre_se_rechaza`
- `test_validate_create_ada_con_token_y_padre_pasa_solo_la_forma`: TypeError por `parent_pipeline_id`
- `test_plan_no_acepta_ada`: AssertionError de `_NO_PLANIFICAR`
- `test_las_manos_valida_la_config_al_arrancar`

Si la lista es distinta, parar y entender por qué antes de seguir.

- [ ] **Step 5: Implementar models, policy, routes y server**

En `jacobs/models.py`, reemplazar las líneas 55-56:

```python
INVOKER_PLATAFORMA = "plataforma"
# Ada solo crea sub-pipelines: presenta un subpipeline_token emitido por Jacobs
# para un padre y un paso de Ada en ejecución (jacobs/subpipelines.py).
INVOKER_ADA = "ada"
VALID_INVOKERS = frozenset({INVOKER_PLATAFORMA, "jax_local", INVOKER_ADA})
```

En `class Pipeline`, justo después de `owner_ack_at`:

```python
    # Frente F (2026-09-16): un hijo de Ada guarda de quién es hijo y a qué
    # profundidad. Los dos salen de la fila del token, nunca del pedido.
    parent_pipeline_id: str | None = None
    depth:              int = 0
```

Reemplazar `class PipelineCreateRequest` completa (líneas 108-131):

```python
class PipelineCreateRequest(BaseModel):
    name:               str
    objective:          str
    invoked_by:         str
    user_id:            str | None = None
    tenant_id:          str | None = None
    mode:               str
    max_steps:          int = 20
    steps:              list[StepSpec] | None = None
    # Frente F: solo para invoked_by="ada". La profundidad NO es un campo: un
    # `depth` o `subpipeline_depth` en el JSON se ignora (nadie lo lee).
    subpipeline_token:  str | None = Field(default=None, min_length=1, max_length=128)
    parent_pipeline_id: str | None = Field(default=None, min_length=1, max_length=36)

    @model_validator(mode="after")
    def validate_fields(self) -> "PipelineCreateRequest":
        if self.invoked_by not in VALID_INVOKERS:
            raise ValueError(
                f"invoked_by '{self.invoked_by}' inválido. Aceptados: {sorted(VALID_INVOKERS)}"
            )
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"mode '{self.mode}' inválido. Aceptados: {sorted(VALID_MODES)}"
            )
        if self.max_steps < 1 or self.max_steps > 20:
            raise ValueError("max_steps debe estar entre 1 y 20 (límite duro v0.1)")
        trae_contrato = self.subpipeline_token is not None or self.parent_pipeline_id is not None
        if self.invoked_by != INVOKER_ADA and trae_contrato:
            raise ValueError(
                "subpipeline_token y parent_pipeline_id solo los acepta invoked_by='ada'"
            )
        if self.invoked_by == INVOKER_ADA and (
            self.subpipeline_token is None or self.parent_pipeline_id is None
        ):
            raise ValueError("invoked_by='ada' exige subpipeline_token y parent_pipeline_id")
        return self
```

En `jacobs/policy.py`, reemplazar las líneas 12-78 (import, constantes y `validate_create`):

```python
from jacobs.models import INVOKER_ADA, INVOKER_PLATAFORMA, VALID_INVOKERS

KILL_SWITCH_PATH = Path("/etc/jax/PAUSE")

MAX_STEPS_PER_PIPELINE  = 20
MAX_PARALLEL_PIPELINES  = 3
# MAX_SUBPIPELINE_DEPTH se borró (frente F, 2026-09-16): era un literal que
# ningún llamador alimentaba. La profundidad vive en la fila del token y el
# límite en JAX_MAX_SUBPIPELINE_DEPTH (jacobs/subpipelines.py).


@dataclass
class PolicyResult:
    ok:     bool
    reason: str


def check_kill_switch() -> bool:
    """True si el kill switch está activo."""
    return KILL_SWITCH_PATH.exists()


def validate_create(
    invoked_by: str,
    mode: str,
    max_steps: int,
    active_count: int,
    subpipeline_token: str | None = None,
    parent_pipeline_id: str | None = None,
) -> PolicyResult:
    """Valida si se puede crear un pipeline nuevo.

    Para ada controla la FORMA (token y padre presentes; nadie más puede
    traerlos). La VALIDEZ del token no se decide acá: es un consumo atómico en
    la base (subpipelines.consumir_token_subpipeline), que routes.create_pipeline
    corre justo después de esta función, dentro del mismo candado."""

    if invoked_by not in VALID_INVOKERS:
        return PolicyResult(
            ok=False,
            reason=f"invoked_by '{invoked_by}' no autorizado. Aceptados: {sorted(VALID_INVOKERS)}",
        )

    if invoked_by == INVOKER_ADA and (not subpipeline_token or not parent_pipeline_id):
        return PolicyResult(
            ok=False,
            reason="ada requiere subpipeline_token y parent_pipeline_id para invocar Jacobs",
        )

    if invoked_by != INVOKER_ADA and (subpipeline_token or parent_pipeline_id):
        return PolicyResult(
            ok=False,
            reason=(
                f"invoked_by '{invoked_by}' no puede presentar subpipeline_token "
                "ni parent_pipeline_id"
            ),
        )

    if max_steps > MAX_STEPS_PER_PIPELINE:
        return PolicyResult(
            ok=False,
            reason=f"max_steps={max_steps} excede límite duro ({MAX_STEPS_PER_PIPELINE})",
        )

    if active_count >= MAX_PARALLEL_PIPELINES:
        return PolicyResult(
            ok=False,
            reason=(
                f"Ya hay {active_count} pipelines activos. "
                f"Límite duro: {MAX_PARALLEL_PIPELINES}"
            ),
        )

    if check_kill_switch():
        return PolicyResult(ok=False, reason="Kill switch activo — Jacobs detenido")

    return PolicyResult(ok=True, reason="OK")
```

En `jacobs/routes.py`, cambiar el import de modelos (líneas 21-29) para incluir `INVOKER_ADA`:

```python
from jacobs.models import (
    INVOKER_ADA,
    VALID_INVOKERS,
    Pipeline,
    PipelineCreateRequest,
    PipelineStatus,
    Step,
    StepSpec,
    StepStatus,
)
```

En `plan_only`, justo después del bloque `if req.invoked_by not in VALID_INVOKERS:` (antes del kill switch):

```python
    # Frente F: Ada no planifica por acá. Un sub-pipeline entra solo por
    # POST /pipeline con un subpipeline_token; /plan no consume tokens y
    # planificar llama a un LLM.
    if req.invoked_by == INVOKER_ADA:
        raise HTTPException(
            status_code=403,
            detail="ada no planifica por /plan: un sub-pipeline se crea por /pipeline con subpipeline_token",
        )
```

En `las_manos/server.py::_jacobs_init`, inmediatamente antes de `await jacobs_store.init_tables()`:

```python
    # Frente F (2026-09-16): la config del contrato de sub-pipelines se valida
    # al arrancar. Un valor inválido en /etc/jax/.env tumba LAS MANOS acá
    # (fail-closed) en vez de descubrirse en el primer hijo de Ada.
    from jacobs.subpipelines import config_subpipelines
    config_subpipelines()
```

- [ ] **Step 6: Correr los puros y la suite de invocadores existente**

```bash
cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_subpipeline_contrato_puro.py tests/test_jacobs_invoked_by_rol.py 2>&1 | tail -3
```
Expected: `26 passed` (18 nuevos + 8 existentes).

- [ ] **Step 7: Validar las dos guardas por mutación**

1. Comentar en `jacobs/executor.py` el `if check_kill_switch():` del bucle de olas (línea ~1122) y reemplazarlo por `if False:`. Correr:
   ```bash
   cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_subpipeline_contrato_puro.py::test_hijo_respeta_el_kill_switch_antes_de_su_primera_ola
   ```
   Expected: `1 failed` (AssertionError "un hijo frenado no despacha"). Revertir con `git -C $WT checkout jacobs/executor.py`.
2. Agregar temporalmente en `jacobs/routes.py` `@router.post("/subpipeline-token")` sobre una función vacía `async def _mutacion(): return {}`. Correr `test_la_emision_no_tiene_ruta_http`. Expected: `1 failed`. Revertir la mutación a mano (no con checkout: routes.py tiene cambios del Step 5).

Anotar las dos salidas en `progress.md`.

- [ ] **Step 8: Commit**

```bash
git -C $WT add jacobs/subpipelines.py jacobs/models.py jacobs/policy.py jacobs/routes.py las_manos/server.py tests/test_subpipeline_contrato_puro.py .superpowers/sdd/2026-09-16-contrato-subpipelines/progress.md
git -C $WT commit -m "feat(jacobs): forma del contrato de sub-pipelines por rol y config validada al arrancar

- ada exige subpipeline_token y parent_pipeline_id; plataforma y jax_local no
  pueden traerlos (422).
- validate_create pierde subpipeline_depth: la profundidad nunca viene del
  llamador. MAX_SUBPIPELINE_DEPTH (literal sin alimentar) se borra.
- /plan rechaza ada (403).
- JAX_SUBPIPELINE_TOKEN_TTL_SECONDS / JAX_MAX_SUBPIPELINE_DEPTH con rango,
  validadas en _jacobs_init.
18 tests puros; 8 vistos en rojo contra master, 2 guardas validadas por mutación.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Esquema en `init_tables()`

**Files:**
- Modify: `jacobs/store.py:157-163` (columnas de `jacobs_pipelines`), `jacobs/store.py:210-219` (después de `jacobs_events`), `jacobs/store.py:260-282` (`pipeline_create`), `jacobs/store.py:370-397` (`_row_to_pipeline`)
- Create: `jacobs/_subpipeline_contrato_io_test.py`

**Interfaces:**
- Consumes: `Pipeline.parent_pipeline_id`, `Pipeline.depth` (Task 2); `jacobs._arnes_ada.una_fila/ejecutar/padre_en_ejecucion/cerrar` (Task 1)
- Produces: la tabla `jacobs_subpipeline_tokens(token_hash CHAR(64) PK, parent_pipeline_id VARCHAR(36), parent_step VARCHAR(36), depth_hijo INT, emitido_at DOUBLE, vence_at DOUBLE, usado_at DOUBLE NULL, hijo_pipeline_id VARCHAR(36) NULL, INDEX idx_subpipeline_tokens_padre (parent_pipeline_id))` y las columnas `jacobs_pipelines.parent_pipeline_id VARCHAR(36) NULL` y `jacobs_pipelines.depth INT NOT NULL DEFAULT 0`. Produce también `_ConBase`, la clase base de los tests de base.

- [ ] **Step 1: Escribir los tests de esquema**

Create `jacobs/_subpipeline_contrato_io_test.py`:

```python
"""Contrato de sub-pipelines contra una MariaDB REAL (frente F, 2026-09-16):
esquema, emisión, consumo atómico, carreras y plan de la consulta.

Las tablas de Jacobs las crea ESTE repo (store.init_tables), así que el job de
CI no necesita el esquema de jax-platform.

Seguridad: la barrera de base vive en jacobs/_arnes_ada.py, que se importa
PRIMERO: con JAX_DB_NAME apuntando a otra base que no sea jax_memory_test, el
módulo entero levanta RuntimeError.

Corre con:
  bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -v jacobs/_subpipeline_contrato_io_test.py"
"""
from __future__ import annotations

from jacobs import _arnes_ada as ada  # primero: barrera de base de prueba

import asyncio  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import unittest  # noqa: E402
import uuid  # noqa: E402
from unittest.mock import patch  # noqa: E402

import aiomysql  # noqa: E402

from jacobs import store  # noqa: E402
from jacobs import subpipelines as sp  # noqa: E402
from jacobs.models import Pipeline, PipelineStatus  # noqa: E402

COLUMNAS_TOKENS = [
    "token_hash", "parent_pipeline_id", "parent_step", "depth_hijo",
    "emitido_at", "vence_at", "usado_at", "hijo_pipeline_id",
]


@unittest.skipUnless(os.getenv("JAX_DB_HOST"), "necesita la MariaDB real (jax_memory_test)")
class _ConBase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await store.init_tables()
        freno = patch("jacobs.policy.check_kill_switch", return_value=False)
        freno.start()
        self.addCleanup(freno.stop)
        entorno = patch.dict(os.environ)
        entorno.start()
        self.addCleanup(entorno.stop)
        os.environ.pop(sp.ENV_TTL, None)
        os.environ.pop(sp.ENV_MAX_PROFUNDIDAD, None)
        self.padres: list[str] = []

    async def asyncTearDown(self):
        for pid in self.padres:
            await ada.cerrar(pid)

    async def padre(self, **kw) -> tuple[str, str]:
        pid, paso = await ada.padre_en_ejecucion(**kw)
        self.padres.append(pid)
        return pid, paso

    async def columnas(self, tabla: str) -> list[str]:
        conn = await store.get_conn()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s ORDER BY ORDINAL_POSITION",
                    (tabla,),
                )
                return [r[0] for r in await cur.fetchall()]
        finally:
            conn.close()


class EsquemaTest(_ConBase):
    async def test_init_tables_crea_la_tabla_de_tokens_sin_columna_para_el_token(self):
        # Se BORRA y se recrea: la jax_memory_test local puede tenerla de una
        # corrida anterior, y entonces el test pasaría sin init_tables().
        await ada.ejecutar("DROP TABLE IF EXISTS jacobs_subpipeline_tokens")
        await store.init_tables()
        self.assertEqual(await self.columnas("jacobs_subpipeline_tokens"), COLUMNAS_TOKENS)

    async def test_tokens_tienen_pk_por_hash_e_indice_por_padre(self):
        sql = (
            "SELECT GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) AS cols "
            "FROM information_schema.STATISTICS WHERE TABLE_SCHEMA=DATABASE() "
            "AND TABLE_NAME='jacobs_subpipeline_tokens' AND INDEX_NAME=%s"
        )
        self.assertEqual((await ada.una_fila(sql, ("PRIMARY",)))["cols"], "token_hash")
        self.assertEqual(
            (await ada.una_fila(sql, ("idx_subpipeline_tokens_padre",)))["cols"],
            "parent_pipeline_id",
        )

    async def test_jacobs_pipelines_gana_parent_y_depth_aun_si_la_tabla_ya_existia(self):
        await ada.ejecutar(
            "ALTER TABLE jacobs_pipelines DROP COLUMN IF EXISTS parent_pipeline_id, "
            "DROP COLUMN IF EXISTS depth"
        )
        await store.init_tables()
        columnas = await self.columnas("jacobs_pipelines")
        self.assertIn("parent_pipeline_id", columnas)
        self.assertIn("depth", columnas)

    async def test_init_tables_es_idempotente_con_el_contrato(self):
        await store.init_tables()
        await store.init_tables()
        self.assertEqual(await self.columnas("jacobs_subpipeline_tokens"), COLUMNAS_TOKENS)
        self.assertEqual((await self.columnas("jacobs_pipelines")).count("depth"), 1)

    async def test_pipeline_create_conserva_parent_y_depth(self):
        pid = str(uuid.uuid4())
        ahora = time.time()
        await store.pipeline_create(Pipeline(
            pipeline_id=pid, name="hijo", invoked_by="ada", mode="dry_run",
            status=PipelineStatus.completed, parent_pipeline_id="padre-x", depth=1,
            created_at=ahora, updated_at=ahora,
        ))
        leido = await store.pipeline_get(pid)
        self.assertEqual((leido.parent_pipeline_id, leido.depth), ("padre-x", 1))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verlos fallar contra el store viejo**

```bash
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_subpipeline_contrato_io_test.py" 2>&1 | tee -a $WT/.superpowers/sdd/2026-09-16-contrato-subpipelines/progress.md | tail -8
```
Expected: `5 failed`. Las columnas dan `[]` en lugar de la lista, `cols` da `None`, falta `parent_pipeline_id` y el pipeline leído vuelve con `(None, 0)`.

- [ ] **Step 3: Implementar el esquema**

En `jacobs/store.py`, agregar a la lista de columnas de `jacobs_pipelines` (después de la entrada `owner_ack_at`):

```python
                # Frente F (2026-09-16): de quién es hijo un pipeline de Ada y a
                # qué profundidad. ALGORITHM=INSTANT explícito: si MariaDB no
                # puede agregarla sin copiar la tabla, FALLA en vez de bloquear
                # las escrituras de Jacobs mientras copia.
                ("parent_pipeline_id", "ALTER TABLE jacobs_pipelines ADD COLUMN "
                    "parent_pipeline_id VARCHAR(36) NULL, ALGORITHM=INSTANT"),
                ("depth", "ALTER TABLE jacobs_pipelines ADD COLUMN "
                    "depth INT NOT NULL DEFAULT 0, ALGORITHM=INSTANT"),
```

Inmediatamente después del `CREATE TABLE IF NOT EXISTS jacobs_events (...)`:

```python
            # Frente F (2026-09-16): contrato de sub-pipelines. Se guarda SOLO
            # el sha256 del token. Tabla nueva -> el índice va en el CREATE (no
            # hay filas que migrar). Tiempos en DOUBLE epoch como el resto de
            # Jacobs: inmunes a la zona horaria de la sesión.
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS jacobs_subpipeline_tokens (
                    token_hash         CHAR(64)    NOT NULL PRIMARY KEY,
                    parent_pipeline_id VARCHAR(36) NOT NULL,
                    parent_step        VARCHAR(36) NOT NULL,
                    depth_hijo         INT         NOT NULL,
                    emitido_at         DOUBLE      NOT NULL,
                    vence_at           DOUBLE      NOT NULL,
                    usado_at           DOUBLE      NULL,
                    hijo_pipeline_id   VARCHAR(36) NULL,
                    INDEX idx_subpipeline_tokens_padre (parent_pipeline_id)
                ) ENGINE=InnoDB
            """)
```

Reemplazar `pipeline_create` (el `INSERT` y sus parámetros):

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
                     parent_pipeline_id, depth)
                VALUES (%s,%s,%s,%s,%s, %s,%s,%s,%s, %s,%s, %s,%s, %s,%s)
                """,
                (
                    p.pipeline_id, p.name, p.invoked_by, p.mode, p.status.value,
                    json.dumps([s.model_dump() for s in p.plan], ensure_ascii=False),
                    p.current_step_index, p.max_steps,
                    json.dumps(p.context, ensure_ascii=False),
                    p.created_at, p.updated_at,
                    p.user_id, p.tenant_id,
                    p.parent_pipeline_id, p.depth,
                ),
            )
    finally:
        conn.close()
```

En `_row_to_pipeline`, después de `owner_ack_at=row.get("owner_ack_at"),`:

```python
        parent_pipeline_id=row.get("parent_pipeline_id"),
        depth=int(row.get("depth") or 0),
```

- [ ] **Step 4: Correr esquema, índices existentes e identidad**

```bash
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_subpipeline_contrato_io_test.py jacobs/_store_indexes_test.py jacobs/_pipeline_identity_test.py" 2>&1 | tail -3
```
Expected: todo `passed` (5 nuevos + 3 de índices + los de identidad), 0 failed.

- [ ] **Step 5: Commit**

```bash
git -C $WT add jacobs/store.py jacobs/_subpipeline_contrato_io_test.py .superpowers/sdd/2026-09-16-contrato-subpipelines/progress.md
git -C $WT commit -m "feat(jacobs): tabla jacobs_subpipeline_tokens y parent/depth en jacobs_pipelines

Por init_tables(), idempotente; ALTER con ALGORITHM=INSTANT. Solo el sha256
del token. 5 tests contra MariaDB vistos en rojo.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Emisión del lado del servidor

**Files:**
- Modify: `jacobs/store.py` (sección nueva al final, antes de `# Audit events`)
- Modify: `jacobs/subpipelines.py` (imports y funciones al final)
- Modify: `jacobs/_subpipeline_contrato_io_test.py` (clase `EmisionTest`, antes de `if __name__`)

**Interfaces:**
- Consumes: `Motivo`, `EmisionRechazada`, `hash_token`, `token_ref`, `motivo_emision`, `config_subpipelines` (Task 2); el esquema (Task 3)
- Produces:
  - `store.SQL_EMITIR_TOKEN: str`
  - `store.subpipeline_token_emitir(token_hash: str, parent_pipeline_id: str, parent_step: str, emitido_at: float, vence_at: float, max_profundidad: int) -> int | None` (la profundidad del hijo, o None si no insertó)
  - `store.subpipeline_emision_diagnostico(parent_pipeline_id: str, parent_step: str) -> dict` (claves `padre_status`, `padre_depth`, `paso_status`, `paso_facet`, `paso_pipeline_id`)
  - `subpipelines.emitir_token_subpipeline(parent_pipeline_id: str, parent_step: str) -> str` (async; lanza `EmisionRechazada`). **Es la API que va a llamar el emisor.**

- [ ] **Step 1: Escribir los tests de emisión**

Agregar en `jacobs/_subpipeline_contrato_io_test.py`, antes de `if __name__ == "__main__":`:

```python
class EmisionTest(_ConBase):
    async def test_guarda_solo_el_hash(self):
        padre, paso = await self.padre()
        token = await sp.emitir_token_subpipeline(padre, paso)
        self.assertGreaterEqual(len(token), 43)  # 32 bytes en base64url
        self.assertIsNotNone(await ada.fila_token(sp.hash_token(token)))
        self.assertIsNone(await ada.fila_token(token))
        fila = await ada.una_fila(
            "SELECT COUNT(*) AS n FROM jacobs_subpipeline_tokens "
            "WHERE parent_step=%s OR hijo_pipeline_id=%s OR parent_pipeline_id=%s",
            (token, token, token),
        )
        self.assertEqual(fila["n"], 0)

    async def test_depth_hijo_es_la_del_padre_mas_uno_y_vence_con_el_ttl(self):
        os.environ[sp.ENV_MAX_PROFUNDIDAD] = "3"
        os.environ[sp.ENV_TTL] = "120"
        padre, paso = await self.padre(depth=2)
        antes = time.time()
        token = await sp.emitir_token_subpipeline(padre, paso)
        fila = await ada.fila_token(sp.hash_token(token))
        self.assertEqual(fila["depth_hijo"], 3)
        self.assertEqual((fila["parent_pipeline_id"], fila["parent_step"]), (padre, paso))
        self.assertAlmostEqual(fila["vence_at"] - fila["emitido_at"], 120, delta=0.01)
        self.assertGreaterEqual(fila["emitido_at"], antes)
        self.assertIsNone(fila["usado_at"])
        self.assertIsNone(fila["hijo_pipeline_id"])

    async def test_deja_evento_emitido_sin_el_token(self):
        padre, paso = await self.padre()
        token = await sp.emitir_token_subpipeline(padre, paso)
        eventos = [e for e in await store.events_by_pipeline(padre)
                   if e["event_type"] == "SUBPIPELINE_TOKEN_EMITIDO"]
        self.assertEqual(len(eventos), 1)
        self.assertEqual(eventos[0]["step_id"], paso)
        self.assertEqual(eventos[0]["payload"]["token_ref"], sp.token_ref(sp.hash_token(token)))
        self.assertEqual(eventos[0]["payload"]["depth_hijo"], 1)
        self.assertNotIn(token, json.dumps(eventos[0]["payload"]))

    async def _rechazo(self, padre: str, paso: str) -> sp.Motivo:
        with self.assertRaises(sp.EmisionRechazada) as ctx:
            await sp.emitir_token_subpipeline(padre, paso)
        eventos = [e for e in await store.events_by_pipeline(padre)
                   if e["event_type"] == "SUBPIPELINE_RECHAZADO"]
        self.assertTrue(eventos, "un rechazo de emisión tiene que dejar evento")
        self.assertEqual(eventos[-1]["payload"]["fase"], "emision")
        self.assertEqual(eventos[-1]["payload"]["motivo"], ctx.exception.motivo.value)
        n = await ada.una_fila(
            "SELECT COUNT(*) AS n FROM jacobs_subpipeline_tokens WHERE parent_pipeline_id=%s",
            (padre,),
        )
        self.assertEqual(n["n"], 0, "un rechazo no puede dejar un token emitido")
        return ctx.exception.motivo

    async def test_rechaza_padre_que_no_corre(self):
        padre, paso = await self.padre()
        await ada.cerrar(padre)
        self.assertEqual(await self._rechazo(padre, paso), sp.Motivo.PADRE_INACTIVO)

    async def test_rechaza_paso_que_no_es_de_ada(self):
        padre, paso = await self.padre(facet_del_paso="jekyll")
        self.assertEqual(await self._rechazo(padre, paso), sp.Motivo.PASO_NO_ES_ADA)

    async def test_rechaza_profundidad_excedida(self):
        padre, paso = await self.padre(depth=1)  # con el máximo por defecto (1)
        self.assertEqual(await self._rechazo(padre, paso), sp.Motivo.PROFUNDIDAD_EXCEDIDA)

    async def test_rechaza_con_kill_switch_activo(self):
        padre, paso = await self.padre()
        with patch("jacobs.policy.check_kill_switch", return_value=True):
            motivo = await self._rechazo(padre, paso)
        self.assertEqual(motivo, sp.Motivo.KILL_SWITCH_ACTIVO)
```

- [ ] **Step 2: Verlos fallar**

```bash
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_subpipeline_contrato_io_test.py -k Emision" 2>&1 | tail -4
```
Expected: `7 failed`, con `AttributeError: module 'jacobs.subpipelines' has no attribute 'emitir_token_subpipeline'`.

- [ ] **Step 3: Implementar el store de emisión**

En `jacobs/store.py`, antes de la sección `#  Audit events`:

```python
# ----------------------------------------------------------------
#  Contrato de sub-pipelines (frente F, 2026-09-16)
# ----------------------------------------------------------------
# Las dos sentencias que deciden son UNA sola cada una, autocommit: nada de
# SELECT-y-después-escribir, que es exactamente la ventana de carrera que este
# contrato cierra. Si no afectan una fila, un diagnóstico APARTE nombra el
# motivo; el diagnóstico solo etiqueta el rechazo, no lo decide.

SQL_EMITIR_TOKEN = """
    INSERT INTO jacobs_subpipeline_tokens
        (token_hash, parent_pipeline_id, parent_step, depth_hijo, emitido_at, vence_at)
    SELECT %s, p.pipeline_id, s.step_id, p.depth + 1, %s, %s
      FROM jacobs_pipelines p
      JOIN jacobs_steps s ON s.step_id = %s AND s.pipeline_id = p.pipeline_id
     WHERE p.pipeline_id = %s
       AND p.status = 'running'
       AND s.facet = 'ada'
       AND p.depth + 1 <= %s
"""


async def subpipeline_token_emitir(
    token_hash: str,
    parent_pipeline_id: str,
    parent_step: str,
    emitido_at: float,
    vence_at: float,
    max_profundidad: int,
) -> int | None:
    """Inserta el hash solo si el padre y su paso de Ada están corriendo y el
    hijo no excede la profundidad. Devuelve depth_hijo, o None si no insertó."""
    conn = await get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                SQL_EMITIR_TOKEN,
                (token_hash, emitido_at, vence_at, parent_step, parent_pipeline_id, max_profundidad),
            )
            if cur.rowcount != 1:
                return None
            await cur.execute(
                "SELECT depth_hijo FROM jacobs_subpipeline_tokens WHERE token_hash = %s",
                (token_hash,),
            )
            (depth_hijo,) = await cur.fetchone()
            return int(depth_hijo)
    finally:
        conn.close()


async def subpipeline_emision_diagnostico(parent_pipeline_id: str, parent_step: str) -> dict:
    conn = await get_conn()
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                "SELECT status, depth FROM jacobs_pipelines WHERE pipeline_id = %s",
                (parent_pipeline_id,),
            )
            padre = await cur.fetchone()
            await cur.execute(
                "SELECT status, facet, pipeline_id FROM jacobs_steps WHERE step_id = %s",
                (parent_step,),
            )
            paso = await cur.fetchone()
    finally:
        conn.close()
    return {
        "padre_status": padre["status"] if padre else None,
        "padre_depth": int(padre["depth"]) if padre else None,
        "paso_status": paso["status"] if paso else None,
        "paso_facet": paso["facet"] if paso else None,
        "paso_pipeline_id": paso["pipeline_id"] if paso else None,
    }
```

- [ ] **Step 4: Implementar `emitir_token_subpipeline`**

En `jacobs/subpipelines.py`, reemplazar el bloque de imports por:

```python
from __future__ import annotations

import hashlib
import os
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from typing import NoReturn

from jacobs import policy, store
```

Y agregar al final del archivo:

```python
async def _rechazar_emision(parent_pipeline_id: str, parent_step: str, motivo: Motivo) -> NoReturn:
    await store.event_append(
        parent_pipeline_id, "SUBPIPELINE_RECHAZADO",
        {"fase": "emision", "motivo": motivo.value}, parent_step,
    )
    raise EmisionRechazada(motivo)


async def emitir_token_subpipeline(parent_pipeline_id: str, parent_step: str) -> str:
    """Emite un token de un solo uso para que el paso de Ada `parent_step` del
    pipeline `parent_pipeline_id` cree UN hijo.

    Solo servidor: no hay ruta HTTP (las rutas de Jacobs no autentican). La
    llama el emisor dentro del proceso de Jacobs. Devuelve el token en claro, y
    es la única vez que existe: en la base queda su sha256 y en el evento, los
    primeros 12 caracteres del hash."""
    cfg = config_subpipelines()
    if policy.check_kill_switch():
        await _rechazar_emision(parent_pipeline_id, parent_step, Motivo.KILL_SWITCH_ACTIVO)
    token = secrets.token_urlsafe(TOKEN_BYTES)
    token_hash = hash_token(token)
    emitido_at = time.time()
    vence_at = emitido_at + cfg.ttl_segundos
    depth_hijo = await store.subpipeline_token_emitir(
        token_hash, parent_pipeline_id, parent_step, emitido_at, vence_at, cfg.max_profundidad,
    )
    if depth_hijo is None:
        diagnostico = await store.subpipeline_emision_diagnostico(parent_pipeline_id, parent_step)
        await _rechazar_emision(
            parent_pipeline_id, parent_step,
            motivo_emision(diagnostico, parent_pipeline_id, cfg.max_profundidad),
        )
    await store.event_append(
        parent_pipeline_id, "SUBPIPELINE_TOKEN_EMITIDO",
        {"token_ref": token_ref(token_hash), "depth_hijo": depth_hijo, "vence_at": vence_at},
        parent_step,
    )
    return token
```

- [ ] **Step 5: Correr emisión, esquema y puros**

```bash
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_subpipeline_contrato_io_test.py tests/test_subpipeline_contrato_puro.py" 2>&1 | tail -3
```
Expected: `30 passed` (12 de base + 18 puros).

- [ ] **Step 6: Commit**

```bash
git -C $WT add jacobs/store.py jacobs/subpipelines.py jacobs/_subpipeline_contrato_io_test.py
git -C $WT commit -m "feat(jacobs): emitir_token_subpipeline, emisión atómica solo de servidor

INSERT … SELECT que exige padre y paso de Ada corriendo y profundidad dentro
del límite; kill switch frena la emisión; evento SUBPIPELINE_TOKEN_EMITIDO con
token_ref, nunca el token. 7 tests contra MariaDB vistos en rojo.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Consumo atómico, carreras y EXPLAIN

**Files:**
- Modify: `jacobs/store.py` (después de `subpipeline_emision_diagnostico`)
- Modify: `jacobs/subpipelines.py` (al final)
- Modify: `jacobs/_subpipeline_contrato_io_test.py` (clase `ConsumoTest`)

**Interfaces:**
- Consumes: `motivo_consumo`, `TokenConsumido`, `ConsumoRechazado`, `token_ref`, `hash_token`, `config_subpipelines` (Task 2); `emitir_token_subpipeline` (Task 4)
- Produces:
  - `store.SQL_CONSUMIR_TOKEN: str`, con parámetros en este orden: `(usado_at, hijo_pipeline_id, token_hash, ahora, parent_pipeline_id, max_profundidad)`
  - `store.SQL_TOKEN_CONSUMIDO: str`, `store.SQL_DIAGNOSTICO_TOKEN: str`
  - `store.subpipeline_token_consumir(token_hash: str, parent_pipeline_id: str, hijo_pipeline_id: str, ahora: float, max_profundidad: int) -> dict | None`
  - `store.subpipeline_token_diagnostico(token_hash: str) -> dict | None`
  - `subpipelines.consumir_token_subpipeline(token: str, parent_pipeline_id: str, hijo_pipeline_id: str) -> TokenConsumido | ConsumoRechazado` (async)

- [ ] **Step 1: Escribir los tests de consumo**

Agregar en `jacobs/_subpipeline_contrato_io_test.py`, antes de `if __name__ == "__main__":`:

```python
class ConsumoTest(_ConBase):
    async def test_consumo_legitimo_marca_usado_y_ata_al_hijo(self):
        padre, paso = await self.padre()
        token = await sp.emitir_token_subpipeline(padre, paso)
        hijo = str(uuid.uuid4())
        resultado = await sp.consumir_token_subpipeline(token, padre, hijo)
        self.assertEqual(
            resultado, sp.TokenConsumido(parent_pipeline_id=padre, parent_step=paso, depth=1))
        fila = await ada.fila_token(sp.hash_token(token))
        self.assertEqual(fila["hijo_pipeline_id"], hijo)
        self.assertIsNotNone(fila["usado_at"])

    async def test_token_inexistente_es_desconocido_y_deja_evento_sin_el_token(self):
        hijo = str(uuid.uuid4())
        resultado = await sp.consumir_token_subpipeline("no-existe-este-token", "padre-x", hijo)
        self.assertEqual(resultado, sp.ConsumoRechazado(sp.Motivo.TOKEN_DESCONOCIDO))
        eventos = await store.events_by_pipeline(hijo)
        self.assertEqual([e["event_type"] for e in eventos], ["SUBPIPELINE_RECHAZADO"])
        self.assertEqual(eventos[0]["payload"]["fase"], "consumo")
        self.assertNotIn("no-existe-este-token", json.dumps(eventos[0]["payload"]))

    async def test_rechazo_por_otro_padre_no_quema_el_token(self):
        padre, paso = await self.padre()
        token = await sp.emitir_token_subpipeline(padre, paso)
        self.assertEqual(
            await sp.consumir_token_subpipeline(token, str(uuid.uuid4()), str(uuid.uuid4())),
            sp.ConsumoRechazado(sp.Motivo.PADRE_NO_COINCIDE),
        )
        self.assertIsInstance(
            await sp.consumir_token_subpipeline(token, padre, str(uuid.uuid4())),
            sp.TokenConsumido,
        )

    async def test_carrera_de_veinte_consumos_tiene_un_solo_ganador(self):
        padre, paso = await self.padre()
        for ronda in range(10):
            token = await sp.emitir_token_subpipeline(padre, paso)
            resultados = await asyncio.gather(*[
                sp.consumir_token_subpipeline(token, padre, str(uuid.uuid4()))
                for _ in range(20)
            ])
            ganadores = [r for r in resultados if isinstance(r, sp.TokenConsumido)]
            perdedores = [r for r in resultados if not isinstance(r, sp.TokenConsumido)]
            self.assertEqual(len(ganadores), 1, f"ronda {ronda}: {resultados}")
            self.assertEqual(
                set(perdedores), {sp.ConsumoRechazado(sp.Motivo.TOKEN_USADO)}, f"ronda {ronda}")

    async def test_carrera_forzada_el_segundo_espera_el_candado_y_pierde(self):
        """Intercalado forzado con dos sesiones reales: A consume y NO confirma;
        B tiene que quedar esperando el candado de fila y, al confirmar A,
        releer y perder. Sin `usado_at IS NULL` en el WHERE, B también gana."""
        padre, paso = await self.padre()
        token = await sp.emitir_token_subpipeline(padre, paso)
        max_prof = sp.config_subpipelines().max_profundidad
        conn_a = await store.get_conn()
        try:
            await conn_a.autocommit(False)
            async with conn_a.cursor() as cur:
                ahora = time.time()
                await cur.execute(
                    store.SQL_CONSUMIR_TOKEN,
                    (ahora, "hijo-a", sp.hash_token(token), ahora, padre, max_prof),
                )
                self.assertEqual(cur.rowcount, 1)
            tarea_b = asyncio.create_task(sp.consumir_token_subpipeline(token, padre, "hijo-b"))
            await asyncio.sleep(1.0)
            self.assertFalse(
                tarea_b.done(),
                "B no esperó el candado de fila de A: el consumo no es atómico",
            )
            await conn_a.commit()
        finally:
            conn_a.close()
        resultado_b = await asyncio.wait_for(tarea_b, timeout=15)
        self.assertEqual(resultado_b, sp.ConsumoRechazado(sp.Motivo.TOKEN_USADO))
        self.assertEqual((await ada.fila_token(sp.hash_token(token)))["hijo_pipeline_id"], "hijo-a")

    async def test_nivel_de_aislamiento_declarado(self):
        fila = await ada.una_fila("SELECT @@SESSION.transaction_isolation AS nivel")
        self.assertIn(
            fila["nivel"], {"REPEATABLE-READ", "READ-COMMITTED"},
            "el argumento de atomicidad del consumo (UPDATE con candado de fila que "
            "relee la versión confirmada) está escrito para estos dos niveles",
        )

    async def test_explain_del_consumo_usa_claves_primarias(self):
        padre, paso = await self.padre()
        for _ in range(30):  # volumen: con tablas casi vacías el optimizador puede elegir ALL
            await sp.emitir_token_subpipeline(padre, paso)
        token = await sp.emitir_token_subpipeline(padre, paso)
        conn = await store.get_conn()
        try:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                ahora = time.time()
                # La SQL REAL (la constante que ejecuta el store), no una copia.
                await cur.execute(
                    "EXPLAIN " + store.SQL_CONSUMIR_TOKEN,
                    (ahora, "hijo-explain", sp.hash_token(token), ahora, padre, 1),
                )
                plan = await cur.fetchall()
        finally:
            conn.close()
        por_tabla = {f["table"]: f for f in plan}
        self.assertEqual(set(por_tabla), {"t", "p", "s"}, plan)
        for alias, fila in por_tabla.items():
            self.assertIn(fila["type"], {"const", "eq_ref"}, f"{alias}: {fila}")
            self.assertEqual(fila["key"], "PRIMARY", f"{alias}: {fila}")
```

- [ ] **Step 2: Verlos fallar**

```bash
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_subpipeline_contrato_io_test.py -k Consumo" 2>&1 | tail -4
```
Expected: `6 failed, 1 passed`. Los 6 caen por `AttributeError` (`consumir_token_subpipeline` / `SQL_CONSUMIR_TOKEN`). `test_nivel_de_aislamiento_declarado` pasa: es una medición del entorno, no del cambio, y se deja dicho en `progress.md`.

- [ ] **Step 3: Implementar el store del consumo**

En `jacobs/store.py`, después de `subpipeline_emision_diagnostico`:

```python
# El consumo. Un UPDATE multi-tabla autocommit: toma el candado de fila de la PK
# del token; un segundo consumo concurrente espera ese candado y, al liberarse,
# RELEE la versión confirmada (REPEATABLE-READ y READ-COMMITTED), ve usado_at
# puesto y afecta 0 filas. Probado con intercalado forzado en
# jacobs/_subpipeline_contrato_io_test.py. Plan: t por PK (const), p y s por PK.
SQL_CONSUMIR_TOKEN = """
    UPDATE jacobs_subpipeline_tokens t
      JOIN jacobs_pipelines p ON p.pipeline_id = t.parent_pipeline_id
      JOIN jacobs_steps s     ON s.step_id = t.parent_step
                             AND s.pipeline_id = t.parent_pipeline_id
       SET t.usado_at = %s, t.hijo_pipeline_id = %s
     WHERE t.token_hash = %s
       AND t.usado_at IS NULL
       AND t.vence_at > %s
       AND t.parent_pipeline_id = %s
       AND t.depth_hijo <= %s
       AND p.status = 'running'
"""

SQL_TOKEN_CONSUMIDO = """
    SELECT parent_pipeline_id, parent_step, depth_hijo
      FROM jacobs_subpipeline_tokens
     WHERE token_hash = %s AND hijo_pipeline_id = %s
"""

SQL_DIAGNOSTICO_TOKEN = """
    SELECT t.usado_at, t.vence_at, t.parent_pipeline_id, t.depth_hijo,
           p.status AS padre_status, s.status AS paso_status
      FROM jacobs_subpipeline_tokens t
      LEFT JOIN jacobs_pipelines p ON p.pipeline_id = t.parent_pipeline_id
      LEFT JOIN jacobs_steps s     ON s.step_id = t.parent_step
     WHERE t.token_hash = %s
"""


async def subpipeline_token_consumir(
    token_hash: str,
    parent_pipeline_id: str,
    hijo_pipeline_id: str,
    ahora: float,
    max_profundidad: int,
) -> dict | None:
    """Consume el token para `hijo_pipeline_id`. Devuelve la fila consumida
    (parent_pipeline_id, parent_step, depth_hijo) o None si no afectó una fila."""
    conn = await get_conn()
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                SQL_CONSUMIR_TOKEN,
                (ahora, hijo_pipeline_id, token_hash, ahora, parent_pipeline_id, max_profundidad),
            )
            if cur.rowcount != 1:
                return None
            await cur.execute(SQL_TOKEN_CONSUMIDO, (token_hash, hijo_pipeline_id))
            return await cur.fetchone()
    finally:
        conn.close()


async def subpipeline_token_diagnostico(token_hash: str) -> dict | None:
    conn = await get_conn()
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(SQL_DIAGNOSTICO_TOKEN, (token_hash,))
            return await cur.fetchone()
    finally:
        conn.close()
```

- [ ] **Step 4: Implementar `consumir_token_subpipeline`**

Al final de `jacobs/subpipelines.py`:

```python
async def consumir_token_subpipeline(
    token: str, parent_pipeline_id: str, hijo_pipeline_id: str,
) -> TokenConsumido | ConsumoRechazado:
    """Consume el token de forma atómica para crear `hijo_pipeline_id`.

    Aceptado solo si el hash existe, no venció, no se usó, fue emitido para
    `parent_pipeline_id`, su profundidad cabe en el límite vigente y el padre y
    su paso de Ada siguen corriendo. La profundidad y el padre que se devuelven
    salen de la FILA. Todo rechazo deja SUBPIPELINE_RECHAZADO con el motivo y
    sin el token."""
    cfg = config_subpipelines()
    token_hash = hash_token(token)
    fila = await store.subpipeline_token_consumir(
        token_hash, parent_pipeline_id, hijo_pipeline_id, time.time(), cfg.max_profundidad,
    )
    if fila is not None:
        return TokenConsumido(
            parent_pipeline_id=fila["parent_pipeline_id"],
            parent_step=fila["parent_step"],
            depth=int(fila["depth_hijo"]),
        )
    diagnostico = await store.subpipeline_token_diagnostico(token_hash)
    motivo = motivo_consumo(diagnostico, parent_pipeline_id, time.time(), cfg.max_profundidad)
    await store.event_append(hijo_pipeline_id, "SUBPIPELINE_RECHAZADO", {
        "fase": "consumo",
        "motivo": motivo.value,
        "parent_pipeline_id_declarado": parent_pipeline_id,
        "token_ref": token_ref(token_hash) if diagnostico is not None else None,
    })
    return ConsumoRechazado(motivo)
```

- [ ] **Step 5: Correr todo el archivo de base y los puros**

```bash
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_subpipeline_contrato_io_test.py tests/test_subpipeline_contrato_puro.py" 2>&1 | tail -3
```
Expected: `37 passed` (19 de base + 18 puros).

- [ ] **Step 6: Validar la atomicidad y el EXPLAIN por mutación**

1. En `store.SQL_CONSUMIR_TOKEN`, borrar la línea `AND t.usado_at IS NULL`. Correr:
   ```bash
   bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q jacobs/_subpipeline_contrato_io_test.py -k 'carrera'"
   ```
   Expected: `2 failed`, el de veinte con más de un ganador y el forzado con B aceptado. Restaurar la línea.
2. En `store.SQL_CONSUMIR_TOKEN`, cambiar `WHERE t.token_hash = %s` por `WHERE LEFT(t.token_hash, 64) = %s`. Correr `-k explain`. Expected: `1 failed` (`type` = `ALL`). Restaurar.

Pegar las dos salidas en `progress.md`. Confirmar `git -C $WT diff jacobs/store.py` limpio respecto del Step 3 antes del commit.

- [ ] **Step 7: Commit**

```bash
git -C $WT add jacobs/store.py jacobs/subpipelines.py jacobs/_subpipeline_contrato_io_test.py .superpowers/sdd/2026-09-16-contrato-subpipelines/progress.md
git -C $WT commit -m "feat(jacobs): consumo atómico del subpipeline_token

UPDATE … JOIN condicionado (una fila o rechazo), diagnóstico aparte para el
motivo, evento SUBPIPELINE_RECHAZADO sin el token. Carrera de 20 consumos y
carrera forzada con dos sesiones: un solo ganador. EXPLAIN de la SQL real: t, p
y s por PRIMARY. Validado por mutación (sin usado_at IS NULL cae la carrera;
con LEFT() cae el EXPLAIN).

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Cableado en la ruta de creación y matriz de ataques

**Files:**
- Modify: `jacobs/routes.py:30-35` (imports), `jacobs/routes.py:137-208` (`create_pipeline`)
- Modify: `tests/test_subpipeline_contrato_rutas.py` (import de `sp` y 10 tests)

**Interfaces:**
- Consumes: `consumir_token_subpipeline`, `ConsumoRechazado`, `Motivo`, `hash_token`, `ENV_MAX_PROFUNDIDAD` (Tasks 2-5); `_arnes_ada.*` (Task 1)
- Produces: comportamiento HTTP de `POST /jacobs/pipeline` con `invoked_by="ada"`: 200 con hijo creado, 403 si el token se rechaza (con `detail="subpipeline_token rechazado: <motivo>"`), 422 si la forma es inválida o el plan se rechaza, 423 si está el kill switch. Eventos `SUBPIPELINE_CREADO` en el padre (`step_id=parent_step`, payload `{hijo_pipeline_id, depth}`) y `SUBPIPELINE_RECHAZADO` con `fase="plan"` en el hijo candidato (payload `{fase, motivo, parent_pipeline_id}`).

- [ ] **Step 1: Agregar la matriz de ataques y el camino legítimo**

En `tests/test_subpipeline_contrato_rutas.py`, cambiar el bloque de imports posterior al del arnés por:

```python
import asyncio  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from jacobs import store  # noqa: E402
from jacobs import subpipelines as sp  # noqa: E402
```

Y agregar al final del archivo:

```python
def _correr(escenario):
    return asyncio.run(escenario())


def test_camino_legitimo_crea_el_hijo_con_padre_y_profundidad_del_token():
    async def escenario():
        padre, paso = await ada.padre_en_ejecucion()
        try:
            token = await ada.emitir(padre, paso)
            respuesta = await ada.pedir_hijo(token, padre)
            hijo = await store.pipeline_get(respuesta["pipeline_id"])
            fila = await ada.fila_token(sp.hash_token(token))
            eventos_padre = await store.events_by_pipeline(padre)
            eventos_hijo = await store.events_by_pipeline(hijo.pipeline_id)
            return padre, paso, respuesta, hijo, fila, eventos_padre, eventos_hijo
        finally:
            await ada.cerrar(padre)

    padre, paso, respuesta, hijo, fila, eventos_padre, eventos_hijo = _correr(escenario)
    assert respuesta["status"] == "completed"
    assert (hijo.parent_pipeline_id, hijo.depth, hijo.invoked_by) == (padre, 1, "ada")
    assert fila["hijo_pipeline_id"] == hijo.pipeline_id
    assert fila["usado_at"] is not None
    creados = [e for e in eventos_padre if e["event_type"] == "SUBPIPELINE_CREADO"]
    assert [(e["payload"]["hijo_pipeline_id"], e["payload"]["depth"], e["step_id"])
            for e in creados] == [(hijo.pipeline_id, 1, paso)]
    creacion = [e for e in eventos_hijo if e["event_type"] == "PIPELINE_CREATED"]
    assert creacion[0]["payload"]["parent_pipeline_id"] == padre


def _rechazo_403(rechazo: HTTPException, motivo: sp.Motivo, token: str) -> None:
    assert rechazo.status_code == 403, rechazo.detail
    assert rechazo.detail == f"subpipeline_token rechazado: {motivo.value}"
    assert token not in rechazo.detail


def test_token_reusado_se_rechaza():
    async def escenario():
        padre, paso = await ada.padre_en_ejecucion()
        try:
            token = await ada.emitir(padre, paso)
            await ada.pedir_hijo(token, padre)
            with pytest.raises(HTTPException) as rechazo:
                await ada.pedir_hijo(token, padre)
            return token, rechazo.value
        finally:
            await ada.cerrar(padre)

    token, rechazo = _correr(escenario)
    _rechazo_403(rechazo, sp.Motivo.TOKEN_USADO, token)


def test_token_de_otro_padre_se_rechaza_y_no_se_quema():
    async def escenario():
        padre_a, paso_a = await ada.padre_en_ejecucion()
        padre_b, _ = await ada.padre_en_ejecucion()
        try:
            token_de_a = await ada.emitir(padre_a, paso_a)
            with pytest.raises(HTTPException) as rechazo:
                await ada.pedir_hijo(token_de_a, padre_b)
            fila = await ada.fila_token(sp.hash_token(token_de_a))
            return token_de_a, rechazo.value, fila
        finally:
            await ada.cerrar(padre_a)
            await ada.cerrar(padre_b)

    token, rechazo, fila = _correr(escenario)
    _rechazo_403(rechazo, sp.Motivo.PADRE_NO_COINCIDE, token)
    assert fila["usado_at"] is None


def test_token_vencido_se_rechaza():
    async def escenario():
        padre, paso = await ada.padre_en_ejecucion()
        try:
            token = await ada.emitir(padre, paso)
            await ada.ejecutar(
                "UPDATE jacobs_subpipeline_tokens SET vence_at = %s WHERE token_hash = %s",
                (time.time() - 1, sp.hash_token(token)),
            )
            with pytest.raises(HTTPException) as rechazo:
                await ada.pedir_hijo(token, padre)
            return token, rechazo.value
        finally:
            await ada.cerrar(padre)

    token, rechazo = _correr(escenario)
    _rechazo_403(rechazo, sp.Motivo.TOKEN_VENCIDO, token)


def test_padre_terminado_se_rechaza():
    async def escenario():
        padre, paso = await ada.padre_en_ejecucion()
        token = await ada.emitir(padre, paso)
        await ada.cerrar(padre)
        with pytest.raises(HTTPException) as rechazo:
            await ada.pedir_hijo(token, padre)
        return token, rechazo.value

    token, rechazo = _correr(escenario)
    _rechazo_403(rechazo, sp.Motivo.PADRE_INACTIVO, token)


def test_profundidad_excedida_se_rechaza(monkeypatch):
    async def escenario():
        padre, paso = await ada.padre_en_ejecucion(depth=1)
        try:
            monkeypatch.setenv(sp.ENV_MAX_PROFUNDIDAD, "2")
            token = await ada.emitir(padre, paso)  # token para un hijo de profundidad 2
            monkeypatch.setenv(sp.ENV_MAX_PROFUNDIDAD, "1")
            with pytest.raises(HTTPException) as rechazo:
                await ada.pedir_hijo(token, padre)
            return token, rechazo.value
        finally:
            await ada.cerrar(padre)

    token, rechazo = _correr(escenario)
    _rechazo_403(rechazo, sp.Motivo.PROFUNDIDAD_EXCEDIDA, token)


def test_el_cuerpo_no_fija_la_profundidad(monkeypatch):
    monkeypatch.setenv(sp.ENV_MAX_PROFUNDIDAD, "2")

    async def escenario():
        padre, paso = await ada.padre_en_ejecucion(depth=1)
        try:
            token = await ada.emitir(padre, paso)
            respuesta = await ada.pedir_hijo(
                token, padre, cuerpo_extra={"depth": 0, "subpipeline_depth": 0})
            return await store.pipeline_get(respuesta["pipeline_id"])
        finally:
            await ada.cerrar(padre)

    hijo = _correr(escenario)
    assert hijo.depth == 2


def test_kill_switch_rechaza_sin_consumir_el_token():
    async def escenario():
        padre, paso = await ada.padre_en_ejecucion()
        try:
            token = await ada.emitir(padre, paso)
            with pytest.raises(HTTPException) as frenado:
                await ada.pedir_hijo(token, padre, kill_switch=True)
            fila_frenado = await ada.fila_token(sp.hash_token(token))
            respuesta = await ada.pedir_hijo(token, padre)
            return frenado.value, fila_frenado, respuesta
        finally:
            await ada.cerrar(padre)

    frenado, fila_frenado, respuesta = _correr(escenario)
    assert frenado.status_code == 423
    assert fila_frenado["usado_at"] is None
    assert respuesta["status"] == "completed"


def test_rechazo_deja_evento_con_motivo_y_sin_el_token():
    async def escenario():
        padre, paso = await ada.padre_en_ejecucion()
        token = await ada.emitir(padre, paso)
        await ada.cerrar(padre)
        with pytest.raises(HTTPException):
            await ada.pedir_hijo(token, padre)
        rechazo = await ada.una_fila(
            "SELECT payload FROM jacobs_events WHERE event_type = 'SUBPIPELINE_RECHAZADO' "
            "AND JSON_VALUE(payload, '$.parent_pipeline_id_declarado') = %s",
            (padre,),
        )
        eventos_padre = await store.events_by_pipeline(padre)
        return token, rechazo, eventos_padre

    token, rechazo, eventos_padre = _correr(escenario)
    assert rechazo is not None
    assert json.loads(rechazo["payload"])["motivo"] == sp.Motivo.PADRE_INACTIVO.value
    assert token not in rechazo["payload"]
    assert all(token not in json.dumps(e["payload"]) for e in eventos_padre)


async def _plan_inejecutable(pipeline_id, objective, max_steps, steps_spec):
    raise HTTPException(status_code=422, detail="plan inejecutable (arnés)")


def test_plan_rechazado_quema_el_token_y_deja_evento():
    """Decisión del plan: el consumo va ANTES de planificar (planificar puede
    tardar 20-40 s con un LLM y no se sostiene una transacción ese tiempo). Un
    plan rechazado deja el token usado: Ada pide otro. Fail-closed."""
    async def escenario():
        padre, paso = await ada.padre_en_ejecucion()
        try:
            token = await ada.emitir(padre, paso)
            with pytest.raises(HTTPException) as rechazo:
                await ada.pedir_hijo(token, padre, plan=_plan_inejecutable)
            fila = await ada.fila_token(sp.hash_token(token))
            evento = await ada.una_fila(
                "SELECT payload FROM jacobs_events WHERE event_type = 'SUBPIPELINE_RECHAZADO' "
                "AND JSON_VALUE(payload, '$.parent_pipeline_id') = %s",
                (padre,),
            )
            return rechazo.value, fila, evento
        finally:
            await ada.cerrar(padre)

    rechazo, fila, evento = _correr(escenario)
    assert rechazo.status_code == 422
    assert fila["usado_at"] is not None
    payload = json.loads(evento["payload"])
    assert (payload["fase"], payload["motivo"]) == ("plan", sp.Motivo.PLAN_RECHAZADO.value)
```

- [ ] **Step 2: Verlos fallar contra la ruta sin cablear**

```bash
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_subpipeline_contrato_rutas.py" 2>&1 | tee -a $WT/.superpowers/sdd/2026-09-16-contrato-subpipelines/progress.md | tail -14
```
Expected: `10 failed, 1 passed`.
- Pasa `test_kill_switch_rechaza_sin_consumir_el_token`: `validate_create` ya devuelve 423 y el camino viejo crea el hijo. Es guarda de orden y se valida por mutación en el Step 5.
- `test_camino_legitimo…` falla con `parent_pipeline_id None`.
- Los ataques fallan con `DID NOT RAISE`.
- `test_plan_rechazado…` falla porque el evento no existe (`evento` es `None`).

- [ ] **Step 3: Cablear `create_pipeline`**

En `jacobs/routes.py`, agregar después del import de `jacobs.policy`:

```python
from jacobs.subpipelines import ConsumoRechazado, Motivo, consumir_token_subpipeline
```

Reemplazar el cuerpo de `create_pipeline` desde `async with _pipeline_create_lock:` hasta el cierre del bloque `with` (antes de `# dry_run:`):

```python
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

        # Frente F (2026-09-16): un hijo de Ada consume su token ACÁ, después de
        # validate_create (un 423 del kill switch no lo quema) y ANTES de
        # planificar (no se sostiene una transacción los 20-40 s del LLM). Padre
        # y profundidad salen de la fila del token, nunca del cuerpo.
        parent_pipeline_id: str | None = None
        parent_step: str | None = None
        depth = 0
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

        # Asignar pipeline_id a cada step
        for step in steps:
            step.pipeline_id = pipeline_id

        now = time.time()
        pipeline = Pipeline(
            pipeline_id=pipeline_id,
            name=req.name,
            invoked_by=req.invoked_by,
            user_id=req.user_id,
            tenant_id=req.tenant_id,
            parent_pipeline_id=parent_pipeline_id,
            depth=depth,
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
            "name": req.name, "mode": req.mode, "steps": len(steps),
            "parent_pipeline_id": parent_pipeline_id, "depth": depth,
        })
        if parent_pipeline_id is not None:
            await store.event_append(
                parent_pipeline_id, "SUBPIPELINE_CREADO",
                {"hijo_pipeline_id": pipeline_id, "depth": depth},
                parent_step,
            )
```

- [ ] **Step 4: Correr la ruta, la base, los puros y los invocadores**

```bash
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_subpipeline_contrato_rutas.py jacobs/_subpipeline_contrato_io_test.py tests/test_subpipeline_contrato_puro.py tests/test_jacobs_invoked_by_rol.py" 2>&1 | tail -3
```
Expected: `56 passed` (11 + 19 + 18 + 8).

- [ ] **Step 5: Validar la guarda del kill switch por mutación**

En `jacobs/routes.py`, mover temporalmente el bloque `if req.invoked_by == INVOKER_ADA: … depth = consumo.depth` para que quede **antes** de `policy = validate_create(`. Hay que usar `pipeline_id = str(uuid.uuid4())` antes del bloque. Correr:
```bash
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q tests/test_subpipeline_contrato_rutas.py -k kill_switch"
```
Expected: `1 failed`. Quedó `fila_frenado["usado_at"]` con valor: el 423 quemó el token. Revertir el movimiento y volver a correr el Step 4. Pegar las dos salidas en `progress.md`.

- [ ] **Step 6: Suite completa de tests puros y escáneres de política**

La lista es la del job `tests-puros` de `.github/workflows/policy.yml`, copiada tal cual, más el archivo nuevo:

```bash
cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -q \
  $(sed -n '/^  tests-puros:/,/^      - name: Piso exacto/p' .github/workflows/policy.yml | grep -oE '(tests|las_manos)/[A-Za-z0-9_./]+\.py' | sort -u) \
  tests/test_subpipeline_contrato_puro.py 2>&1 | tail -3
cd $WT && $PY -m pytest -q policy/tests/test_no_fail_open_except.py tests/test_no_blocking_in_async.py 2>&1 | tail -3
cd $WT && python3 scripts/check_mirror_sync.py
```
Expected:
- Primera: `512 passed, 1 skipped`, que es el piso vigente de master (494) + 18. Si la base cambió por E/B, el número es el piso de esa base + 18.
- Segunda: todo `passed`.
- Tercera: sale 0, porque este frente no toca símbolos espejados. Si reporta alguno, parar: el cambio es de los dos repos.

- [ ] **Step 7: Commit**

```bash
git -C $WT add jacobs/routes.py tests/test_subpipeline_contrato_rutas.py .superpowers/sdd/2026-09-16-contrato-subpipelines/progress.md
git -C $WT commit -m "feat(jacobs): create_pipeline exige y consume el subpipeline_token de Ada

Consumo después de validate_create (el 423 no quema el token) y antes de
planificar. Padre y profundidad desde la fila; eventos SUBPIPELINE_CREADO en
el padre y SUBPIPELINE_RECHAZADO (fase plan). Matriz de ataques con el arnés:
inventado, reusado, de otro padre, vencido, padre terminado, profundidad
excedida, cuerpo que intenta fijar depth, kill switch, evento sin token, plan
rechazado. 10 vistos en rojo; la guarda del kill switch validada por mutación.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: CI (job nuevo, pisos y canario) y PR

**Files:**
- Modify: `.github/workflows/policy.yml`: job nuevo `subpipeline-contrato-db` después de `facet-health-io`; `tests-puros`, las dos listas y el piso

**Interfaces:**
- Consumes: los tres archivos de test de las Tasks 1-6.
- Produces: un job de CI que corre 30 tests contra MariaDB con piso exacto, y `tests-puros` con el piso +18.

- [ ] **Step 1: Medir las dependencias del job nuevo en un venv limpio**

```bash
S=$(mktemp -d /tmp/jax-frente-f-venv-XXXX)
python3.12 -m venv $S/v && $S/v/bin/pip -q install pytest aiomysql==0.3.2 pydantic httpx cryptography pyyaml aiofiles fastapi==0.139.0
cd $WT && PYTHONPATH=.:las_manos $S/v/bin/python -m pytest --collect-only -q jacobs/_subpipeline_contrato_io_test.py tests/test_subpipeline_contrato_rutas.py 2>&1 | tail -3
```
Expected: `30 tests collected`, sin errores de colección. Si falta un módulo, agregarlo a la lista del `pip install` y repetir hasta que colecte limpio. Anotar la lista final en `progress.md`. Después: `rm -rf $S`.

- [ ] **Step 2: Agregar el job**

En `.github/workflows/policy.yml`, después del job `facet-health-io` (termina en el `grep -qE "^9 passed"`), agregar:

```yaml
  # Frente F (2026-09-16): contrato de sub-pipelines contra una MariaDB REAL.
  # Las tablas de Jacobs (jacobs_pipelines/steps/events y la nueva
  # jacobs_subpipeline_tokens) las crea ESTE repo con store.init_tables(): no
  # hace falta clonar jax-platform. PYTHONPATH incluye las_manos/ porque la
  # ruta importa executor -> credential_resolver (sin eso, error de coleccion).
  # Deps medidas en un venv limpio antes de escribir esta lista.
  #
  # Lo que prueba y por que necesita base: la carrera de consumos (20 a la vez
  # y un intercalado forzado con dos sesiones) y el EXPLAIN de la SQL real no
  # existen sin InnoDB de verdad. Un doble de la base daria verde sin probar
  # la atomicidad.
  subpipeline-contrato-db:
    runs-on: ubuntu-latest
    services:
      mariadb:
        image: mariadb:11.8
        env:
          MARIADB_ROOT_PASSWORD: ci
          MARIADB_DATABASE: jax_memory_test
        ports: ["3306:3306"]
        options: >-
          --health-cmd="healthcheck.sh --connect --innodb_initialized"
          --health-interval=5s --health-timeout=5s --health-retries=20
    env:
      JAX_DB_HOST: 127.0.0.1
      JAX_DB_PORT: "3306"
      JAX_DB_USER: root
      JAX_DB_PASSWORD: ci
      JAX_DB_NAME: jax_memory_test
      PYTHONPATH: .:las_manos
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install pytest aiomysql==0.3.2 pydantic httpx cryptography pyyaml aiofiles fastapi==0.139.0
      - run: python -m pytest -v jacobs/_subpipeline_contrato_io_test.py tests/test_subpipeline_contrato_rutas.py
      - name: Piso exacto de tests CORRIDOS
        # Los dos archivos se saltan sin JAX_DB_HOST: un skip masivo se ve verde.
        run: |
          python -m pytest -q jacobs/_subpipeline_contrato_io_test.py \
            tests/test_subpipeline_contrato_rutas.py 2>&1 | tee /tmp/out
          # 30 el 2026-09-16 (frente F): 19 contra la base (esquema 5, emision 7,
          #   consumo 7 con la carrera de 20, la carrera forzada y el EXPLAIN) + 11
          #   por la ruta con el arnes que hace de Ada. Vistos en rojo contra
          #   master; carrera, EXPLAIN y orden del kill switch validados por mutacion.
          grep -qE "^30 passed" /tmp/out || {
            echo "PISO ROTO: se esperaban 30 tests CORRIDOS (no skipeados)."; exit 1; }
```

Si el Step 1 cambió la lista de paquetes, usar la medida. Si el frente B ya está en master y exige `JAX_KILL_SWITCH_PATH` al importar, agregarla al `env:` del job apuntando a una ruta que no existe (`/tmp/jax-ci-sin-freno/PAUSE`).

- [ ] **Step 3: Sumar los puros a `tests-puros`**

En el job `tests-puros`, agregar `tests/test_subpipeline_contrato_puro.py` al final de **las dos** listas: la del `pytest -v` y la del piso, esta última con ` \` de continuación. Agregar a la historia del piso la línea:

```yaml
          # 494 -> 512 el 2026-09-16 (frente F): test_subpipeline_contrato_puro.py (+18) -- forma del pedido por rol, validate_create sin profundidad del llamador, /plan sin ada, config con rango validada al arrancar, motivos de emision/consumo, hijo frenado por el kill switch, sin ruta HTTP de emision. 8 vistos en rojo contra master; 2 guardas validadas por mutacion.
```

Cambiar el `grep` del piso a `"^512 passed, 1 skipped"`, o al piso de la base rebasada + 18.

- [ ] **Step 4: Push, PR y CI en verde**

```bash
git -C $WT add .github/workflows/policy.yml .superpowers/sdd/2026-09-16-contrato-subpipelines/progress.md
git -C $WT commit -m "ci: job subpipeline-contrato-db (piso 30) y tests-puros +18

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git -C $WT push -u origin feat/contrato-subpipelines
gh pr create --repo fjruizhn/Jax --base master --head feat/contrato-subpipelines \
  --title "Contrato de sub-pipelines: token de Ada emitido por el servidor y consumido atómicamente" \
  --body-file $WT/.superpowers/sdd/2026-09-16-contrato-subpipelines/pr-body.md
```

Antes de `gh pr create`, escribir `pr-body.md` con:
- el hallazgo, con `policy.py:48-52` y `routes.py:147-153`;
- las Discrepancias de este plan;
- la matriz de ataques;
- las mutaciones;
- los números de carga como "pendientes de la Task 8";
- el cierre `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

Después:
```bash
SHA=$(git -C $WT rev-parse HEAD)
gh api repos/fjruizhn/Jax/commits/$SHA/check-runs --jq '.check_runs[] | [.name, .status, .conclusion] | @tsv'
```
Repetir la consulta con Monitor (sin `sleep` en primer plano) hasta que todos queden `completed`. Expected: `subpipeline-contrato-db completed success`, `tests-puros completed success` y el resto `success`.

- [ ] **Step 5: Canario — romper a propósito y ver rojo sobre el sha real**

```bash
sed -i 's/       AND t.usado_at IS NULL$/       AND 1 = 1/' $WT/jacobs/store.py
git -C $WT diff --stat
git -C $WT commit -am "canario: consumo sin usado_at IS NULL (debe poner rojo subpipeline-contrato-db)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git -C $WT push
CANARIO=$(git -C $WT rev-parse HEAD)
gh api repos/fjruizhn/Jax/commits/$CANARIO/check-runs --jq '.check_runs[] | select(.name=="subpipeline-contrato-db") | [.status, .conclusion] | @tsv'
```
Expected, cuando termina: `completed	failure`. Verificar en `gh run view --log-failed` que las fallas son las dos de `carrera` y no una colección rota. Después revertir **con commit nuevo** (nada de force-push):
```bash
git -C $WT revert --no-edit HEAD
git -C $WT push
gh api repos/fjruizhn/Jax/commits/$(git -C $WT rev-parse HEAD)/check-runs --jq '.check_runs[] | [.name, .conclusion] | @tsv'
```
Expected: todos `success`. Pegar los dos sha y sus conclusiones en `progress.md` y en el PR.

---

### Task 8: Prueba de carga de la creación con validación del token

**Files:**
- Create: `loadtest/jacobs_subpipelines_app.py`
- Create: `loadtest/sembrar_tokens_subpipeline.py`
- Create: `loadtest/jacobs_subpipelines.js`

**Interfaces:**
- Consumes: `routes.router`, `store.init_tables`, `subpipelines.emitir_token_subpipeline`, `_arnes_ada.plan_de_un_paso`
- Produces: números registrados (rps, p95, tasas) por escenario y concurrencia, en `progress.md`, el PR, `DEUDA.md` y `CONTEXT.md` (Task 11)

Qué se mide y qué no (queda declarado en los archivos y en la Biblioteca):
- **Se mide** el camino que cambió: candado de creación → conteo de activos → `validate_create` → consumo atómico (o rechazo con diagnóstico y evento) → `pipeline_create` → pasos → eventos → `dry_run` completado.
- **No se mide** `build()`: no cambió en este frente y en el camino real llama a un LLM y a la gobernanza. Se sustituye por el plan fijo de un paso del arnés.
- **Nunca contra producción:** la app se niega a arrancar sin `JAX_DB_NAME=jax_memory_test`.
- Escenarios:
  - `base`: `jax_local` sin token, para tener el antes con el mismo sustituto.
  - `legitimo`: tokens sembrados, uno por iteración.
  - `inventado`: el peor caso del atacante, que es un rechazo con diagnóstico y un evento escrito por pedido.

- [ ] **Step 1: Escribir la app de carga**

Create `loadtest/jacobs_subpipelines_app.py`:

```python
"""App de carga del contrato de sub-pipelines (frente F, 2026-09-16).

NUNCA contra producción: escribe pipelines, pasos, tokens y eventos, así que se
niega a arrancar sin JAX_DB_NAME=jax_memory_test.

Monta las rutas REALES de Jacobs. Lo único sustituido es `_build_plan_or_reject`
por el plan fijo de un paso del arnés: build() no cambió en este frente y en
el camino real llama a un LLM. Lo que se mide es lo que sí cambió: candado de
creación, conteo, validate_create, consumo atómico del token, inserts y eventos.

USO (ver la Task 8 del plan 2026-09-16-frente-f-contrato-subpipelines.md):
  PYTHONPATH=.:las_manos uvicorn loadtest.jacobs_subpipelines_app:app --port 7799
"""
from __future__ import annotations

import os

if os.environ.get("JAX_DB_NAME") != "jax_memory_test":
    raise RuntimeError(
        "jacobs_subpipelines_app solo corre con JAX_DB_NAME=jax_memory_test: "
        "escribe pipelines, tokens y eventos."
    )

from fastapi import FastAPI  # noqa: E402

from jacobs import _arnes_ada, routes, store  # noqa: E402

routes._build_plan_or_reject = _arnes_ada.plan_de_un_paso

app = FastAPI(title="carga-contrato-subpipelines")
app.include_router(routes.router)


@app.on_event("startup")
async def _init() -> None:
    await store.init_tables()
```

- [ ] **Step 2: Escribir el sembrador**

Create `loadtest/sembrar_tokens_subpipeline.py`:

```python
"""Siembra un padre `running` con un paso de Ada `running` y N tokens, para la
carga del contrato de sub-pipelines (frente F, 2026-09-16).

Solo jax_memory_test (la barrera vive en jacobs/_arnes_ada.py).

  sembrar --n 60000 --salida DIR/tokens.json   -> tokens.json (lista) y tokens.json.padre
  cerrar  --padre PIPELINE_ID                  -> deja el padre en completed

Los tokens son secretos de prueba: la salida va a un directorio temporal, nunca
al repo, y se borra al terminar la medición.
"""
from __future__ import annotations

import argparse
import asyncio
import json

from jacobs import _arnes_ada as ada
from jacobs import store, subpipelines


async def sembrar(n: int, salida: str, lote: int) -> None:
    await store.init_tables()
    padre, paso = await ada.padre_en_ejecucion()
    tokens: list[str] = []
    while len(tokens) < n:
        k = min(lote, n - len(tokens))
        tokens += await asyncio.gather(*[
            subpipelines.emitir_token_subpipeline(padre, paso) for _ in range(k)
        ])
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(tokens, f)
    with open(salida + ".padre", "w", encoding="utf-8") as f:
        f.write(padre)
    print(f"padre={padre} tokens={len(tokens)} salida={salida}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="accion", required=True)
    s = sub.add_parser("sembrar")
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--salida", required=True)
    s.add_argument("--lote", type=int, default=50)
    c = sub.add_parser("cerrar")
    c.add_argument("--padre", required=True)
    args = p.parse_args()
    if args.accion == "sembrar":
        asyncio.run(sembrar(args.n, args.salida, args.lote))
    else:
        asyncio.run(ada.cerrar(args.padre))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Escribir el escenario k6**

Create `loadtest/jacobs_subpipelines.js`:

```javascript
// Carga de POST /jacobs/pipeline con el contrato de sub-pipelines -- politica 4
// de LAS CUATRO DEL RENDIMIENTO (frente F, 2026-09-16).
//
// Contra la app AISLADA loadtest/jacobs_subpipelines_app.py (jax_memory_test),
// nunca contra LAS MANOS de produccion.
//
// ESCENARIO:
//   base      -- jax_local sin token (el "antes", mismo plan sustituto)
//   legitimo  -- ada con un token sembrado distinto por iteracion
//   inventado -- ada con tokens inventados: el peor caso del atacante (rechazo
//                con diagnostico y un evento escrito por pedido)
//
// 422 "pipelines activos" es el limite duro de 3 (MAX_PARALLEL_PIPELINES)
// haciendo su trabajo bajo concurrencia: se cuenta aparte, no como error.
//
// USO:
//   k6 run -e ESCENARIO=legitimo -e VUS=25 -e TOKENS=$S/tokens.json \
//          -e PADRE=$(cat $S/tokens.json.padre) loadtest/jacobs_subpipelines.js

import http from 'k6/http';
import { check } from 'k6';
import exec from 'k6/execution';
import { Counter } from 'k6/metrics';
import { SharedArray } from 'k6/data';

const BASE = __ENV.BASE || 'http://127.0.0.1:7799';
const ESCENARIO = __ENV.ESCENARIO || 'legitimo';
const VUS = parseInt(__ENV.VUS || '25', 10);
const PADRE = __ENV.PADRE || '';

const TOKENS = ESCENARIO === 'legitimo'
  ? new SharedArray('tokens', () => JSON.parse(open(__ENV.TOKENS)))
  : [];

const aceptados = new Counter('aceptados');
const limiteParalelo = new Counter('limite_paralelo');
const rechazosToken = new Counter('rechazos_token');

http.setResponseCallback(http.expectedStatuses(200, 403, 422));

export const options = {
  stages: [
    { duration: '5s', target: VUS },
    { duration: '20s', target: VUS },
    { duration: '5s', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],
    http_req_failed: ['rate<0.01'],
    checks: ['rate>0.99'],
  },
};

function cuerpo() {
  const base = { name: 'carga-subpipeline', objective: 'o', mode: 'dry_run' };
  if (ESCENARIO === 'base') {
    return { ...base, invoked_by: 'jax_local' };
  }
  const i = exec.scenario.iterationInTest;
  if (ESCENARIO === 'inventado') {
    return { ...base, invoked_by: 'ada', parent_pipeline_id: PADRE,
             subpipeline_token: `inventado-${exec.vu.idInTest}-${i}` };
  }
  if (i >= TOKENS.length) {
    exec.test.abort(`tokens agotados en la iteracion ${i}: sembrar mas`);
  }
  return { ...base, invoked_by: 'ada', parent_pipeline_id: PADRE, subpipeline_token: TOKENS[i] };
}

export default function () {
  const r = http.post(`${BASE}/jacobs/pipeline`, JSON.stringify(cuerpo()), {
    headers: { 'Content-Type': 'application/json' },
    tags: { escenario: ESCENARIO },
  });
  const limite = r.status === 422 && String(r.body).includes('pipelines activos');
  if (r.status === 200) aceptados.add(1);
  else if (limite) limiteParalelo.add(1);
  else if (r.status === 403) rechazosToken.add(1);
  check(r, {
    'respuesta esperada': (res) => (ESCENARIO === 'inventado'
      ? res.status === 403
      : res.status === 200 || limite),
  });
}
```

- [ ] **Step 3b: Commit de las herramientas de carga**

```bash
git -C $WT add loadtest/jacobs_subpipelines_app.py loadtest/sembrar_tokens_subpipeline.py loadtest/jacobs_subpipelines.js
git -C $WT commit -m "loadtest: carga aislada de la creación de hijos con token (base, legitimo, inventado)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Preparar la base de prueba y levantar la app**

```bash
S=$(mktemp -d /tmp/jax-frente-f-carga-XXXX); echo $S
ss -ltn | grep -c ':7799 ' || true   # esperado 0
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY - <<'PY'
import asyncio
from jacobs import _arnes_ada as ada
async def main():
    fila = await ada.una_fila(\"SELECT COUNT(*) AS n FROM jacobs_pipelines WHERE status IN ('pending','running')\")
    print('activos antes', fila['n'])
    if fila['n']:
        n = await ada.ejecutar(\"UPDATE jacobs_pipelines SET status='expired' WHERE status IN ('pending','running')\")
        print('restos de tests pasados a expired (solo jax_memory_test):', n)
asyncio.run(main())
PY"
```
Expected: se imprimen los activos que había. Si había, se pasan a `expired`, porque son restos de corridas de test en `jax_memory_test` que dejarían todo en 422 por el límite de 3. El arnés impide que esto corra contra otra base.

Levantar la app en background (run_in_background):
```bash
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos /home/fruiz/jax/las_manos/.venv/bin/uvicorn loadtest.jacobs_subpipelines_app:app --host 127.0.0.1 --port 7799 --log-level warning"
```
Verificar con `curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7799/jacobs/pipeline/no-existe`. Expected: `404`.

- [ ] **Step 5: Medir los tres escenarios a c=10, 25 y 50**

Para cada `VUS` en `10 25 50`:
```bash
~/bin/k6 run -e ESCENARIO=base -e VUS=$VUS $WT/loadtest/jacobs_subpipelines.js 2>&1 | tee $S/base-$VUS.txt | grep -E 'http_req_duration|http_reqs|aceptados|limite_paralelo|checks'

bash -c "$DBTEST; export JAX_SUBPIPELINE_TOKEN_TTL_SECONDS=3600; cd $WT && PYTHONPATH=.:las_manos $PY loadtest/sembrar_tokens_subpipeline.py sembrar --n 60000 --salida $S/tokens-$VUS.json"
~/bin/k6 run -e ESCENARIO=legitimo -e VUS=$VUS -e TOKENS=$S/tokens-$VUS.json -e PADRE=$(cat $S/tokens-$VUS.json.padre) $WT/loadtest/jacobs_subpipelines.js 2>&1 | tee $S/legitimo-$VUS.txt | grep -E 'http_req_duration|http_reqs|aceptados|limite_paralelo|rechazos_token|checks'

~/bin/k6 run -e ESCENARIO=inventado -e VUS=$VUS -e PADRE=$(cat $S/tokens-$VUS.json.padre) $WT/loadtest/jacobs_subpipelines.js 2>&1 | tee $S/inventado-$VUS.txt | grep -E 'http_req_duration|http_reqs|rechazos_token|checks'

bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY loadtest/sembrar_tokens_subpipeline.py cerrar --padre $(cat $S/tokens-$VUS.json.padre)"
```
- Expected en cada corrida: k6 sale con código 0, con los thresholds cumplidos.
- Si aborta con "tokens agotados", repetir ese `VUS` sembrando `--n 120000`.
- Si un threshold falla, **no es un GO**. Diagnosticar con `EXPLAIN`/`ANALYZE` antes de tocar nada y avisar a Fernando con el número.

Registrar en `progress.md` una tabla con estas columnas: escenario, VUS, rps, p95 ms, p99 ms, aceptados, límite paralelo, rechazos. Registrar también la diferencia de p95 entre `legitimo` y `base` a c=25, que es el costo del contrato, y la concurrencia desde la que empieza a degradar.

- [ ] **Step 6: Apagar, limpiar y registrar**

Detener uvicorn (TaskStop del proceso en background), `ss -ltn | grep -c ':7799 '` → 0, y `rm -rf $S` (los tokens son secretos de prueba). Agregar la tabla al cuerpo del PR (`gh pr edit --body-file`).

---

### Task 9: Verificación de índices con EXPLAIN sobre la consulta real con volumen

**Files:** ninguno (verificación; la evidencia va a `progress.md` y al PR)

**Interfaces:**
- Consumes: `store.SQL_CONSUMIR_TOKEN`, `store.SQL_EMITIR_TOKEN`, `store.SQL_DIAGNOSTICO_TOKEN`, la `jax_memory_test` que quedó con volumen después de la Task 8

- [ ] **Step 1: EXPLAIN y ANALYZE de las tres sentencias con la tabla cargada**

```bash
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY - <<'PY'
import asyncio, time, uuid
from unittest.mock import patch
import aiomysql
from jacobs import _arnes_ada as ada, store, subpipelines as sp

async def main():
    padre, paso = await ada.padre_en_ejecucion()
    with patch('jacobs.policy.check_kill_switch', return_value=False):
        token = await sp.emitir_token_subpipeline(padre, paso)
    h = sp.hash_token(token)
    print('filas tokens:', (await ada.una_fila('SELECT COUNT(*) AS n FROM jacobs_subpipeline_tokens'))['n'])
    print('filas pipelines:', (await ada.una_fila('SELECT COUNT(*) AS n FROM jacobs_pipelines'))['n'])
    print('filas steps:', (await ada.una_fila('SELECT COUNT(*) AS n FROM jacobs_steps'))['n'])
    conn = await store.get_conn()
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            ahora = time.time()
            for nombre, sql, args in (
                ('EMITIR', store.SQL_EMITIR_TOKEN, (sp.hash_token(str(uuid.uuid4())), ahora, ahora + 60, paso, padre, 1)),
                ('CONSUMIR', store.SQL_CONSUMIR_TOKEN, (ahora, 'hijo-explain', h, ahora, padre, 1)),
                ('DIAGNOSTICO', store.SQL_DIAGNOSTICO_TOKEN, (h,)),
            ):
                await cur.execute('EXPLAIN ' + sql, args)
                print(nombre)
                for fila in await cur.fetchall():
                    print('  ', {k: fila[k] for k in ('table', 'type', 'key', 'rows', 'Extra')})
            await cur.execute('ANALYZE ' + store.SQL_CONSUMIR_TOKEN, (ahora, 'hijo-analyze', h, ahora, padre, 1))
            print('ANALYZE CONSUMIR')
            for fila in await cur.fetchall():
                print('  ', {k: fila.get(k) for k in ('table', 'type', 'key', 'rows', 'r_rows', 'r_filtered')})
    finally:
        conn.close()
    await ada.cerrar(padre)

asyncio.run(main())
PY" 2>&1 | tee -a $WT/.superpowers/sdd/2026-09-16-contrato-subpipelines/progress.md
```
Expected:
- `filas tokens` en decenas de miles.
- En EMITIR, CONSUMIR y DIAGNOSTICO, cada fila de `t`, `p` y `s` con `type` ∈ {`const`, `eq_ref`} y `key = PRIMARY`, sin `ALL` y sin `Using filesort`/`Using temporary`.
- En ANALYZE CONSUMIR, `r_rows` = 1 por tabla.

Si aparece un `ALL`, **es un bloqueo del GO**: diagnosticar antes de seguir.

- [ ] **Step 2: Correr el test de EXPLAIN con el volumen real**

```bash
bash -c "$DBTEST; cd $WT && PYTHONPATH=.:las_manos $PY -m pytest -v jacobs/_subpipeline_contrato_io_test.py -k explain"
```
Expected: `1 passed`. Pegar las salidas en el PR.

---

### Task 10: GO, merge y deploy con 0 pipelines en vuelo

**Files:**
- Modify (producción, con `sudo`): `/etc/jax/.env`

**Interfaces:**
- Consumes: PR en verde (Task 7), números de carga (Task 8) y EXPLAIN (Task 9).

- [ ] **Step 1: Revisión y GO de Fernando**

Pedir revisión del PR con `superpowers:requesting-code-review`. El ejecutor no juzga su propio gate. Arreglar todo hallazgo antes de mergear (nada diferido).

Presentar a Fernando:
- los números de carga;
- el EXPLAIN;
- la confirmación de los valores de `.env`: `JAX_SUBPIPELINE_TOKEN_TTL_SECONDS=300`, `JAX_MAX_SUBPIPELINE_DEPTH=3`;
- que el deploy reinicia `jax-las-manos`.

**Sin GO explícito no se sigue.**

- [ ] **Step 2: Merge y actualización del checkout de producción**

```bash
gh pr merge --repo fjruizhn/Jax feat/contrato-subpipelines --merge
git -C /home/fruiz/jax fetch origin
git -C /home/fruiz/jax status --short | head
readlink /proc/$(systemctl show -p MainPID --value jax-las-manos.service)/cwd
```
Expected: el `status` sale limpio en lo trackeado y el `cwd` es `/home/fruiz/jax/las_manos`. Si el checkout tiene cambios sin commitear en archivos que toca el merge, **parar**. Todavía no se hace `pull`: primero va la verificación de vuelo.

- [ ] **Step 3: Verificar 0 pipelines en vuelo**

```bash
S=$(mktemp -d /tmp/jax-frente-f-deploy-XXXX)
cat > $S/en_vuelo.py <<'PY'
import asyncio
from jacobs import store
async def main():
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT DATABASE()")
            print("base:", (await cur.fetchone())[0])
            await cur.execute(
                "SELECT pipeline_id, status, name FROM jacobs_pipelines "
                "WHERE status IN ('pending','running')")
            print("en vuelo:", await cur.fetchall())
    finally:
        conn.close()
asyncio.run(main())
PY
bash -c 'set -a; . /etc/jax/.env; set +a; cd /home/fruiz/jax && PYTHONPATH=.:las_manos las_manos/.venv/bin/python "$1"' _ $S/en_vuelo.py
curl -s http://127.0.0.1:7777/health
```
Expected: `base: jax_memory` y `en vuelo: ()`. Si hay alguno, esperar a que termine y repetir; nunca reiniciar con un pipeline corriendo. Guardar la salida de `/health`.

- [ ] **Step 4: Backups verificados (tabla y config)**

```bash
cat > $S/backup.sh <<'SH'
set -euo pipefail
set -a; . /etc/jax/.env; set +a
B=~/backups/jacobs-pipelines-pre-subpipelines-$(date +%F-%H%M).sql
( umask 077; mysqldump --single-transaction --skip-lock-tables --no-tablespaces \
    -h$JAX_DB_HOST -P$JAX_DB_PORT -u$JAX_DB_USER -p$JAX_DB_PASSWORD $JAX_DB_NAME \
    jacobs_pipelines jacobs_steps > "$B" )
echo "backup: $B"
sed -e 's/`jacobs_pipelines`/`jacobs_pipelines_restore_probe`/g' \
    -e 's/`jacobs_steps`/`jacobs_steps_restore_probe`/g' "$B" \
  | mysql -h$JAX_DB_HOST -P$JAX_DB_PORT -u$JAX_DB_USER -p$JAX_DB_PASSWORD jax_memory_test
mysql -h$JAX_DB_HOST -P$JAX_DB_PORT -u$JAX_DB_USER -p$JAX_DB_PASSWORD -N -e "
  SELECT 'prod pipelines', COUNT(*) FROM jax_memory.jacobs_pipelines;
  SELECT 'restore pipelines', COUNT(*) FROM jax_memory_test.jacobs_pipelines_restore_probe;
  SELECT 'prod steps', COUNT(*) FROM jax_memory.jacobs_steps;
  SELECT 'restore steps', COUNT(*) FROM jax_memory_test.jacobs_steps_restore_probe;"
mysql -h$JAX_DB_HOST -P$JAX_DB_PORT -u$JAX_DB_USER -p$JAX_DB_PASSWORD jax_memory_test \
  -e "DROP TABLE jacobs_pipelines_restore_probe; DROP TABLE jacobs_steps_restore_probe"
SH
bash $S/backup.sh
sudo cp -a /etc/jax/.env /etc/jax/.env.pre-subpipelines-$(date +%F-%H%M)
sudo ls -l /etc/jax/.env.pre-subpipelines-*
```
Expected: los conteos de prod y de restore coinciden de a pares, y la copia del `.env` existe. Un backup sin restauración probada no es backup: si los conteos no coinciden, **parar**.

- [ ] **Step 5: Variables en `/etc/jax/.env`**

```bash
sudo grep -cE '^(JAX_SUBPIPELINE_TOKEN_TTL_SECONDS|JAX_MAX_SUBPIPELINE_DEPTH)=' /etc/jax/.env || true
printf '\n# Contrato de sub-pipelines (frente F, 2026-09-16)\nJAX_SUBPIPELINE_TOKEN_TTL_SECONDS=300\nJAX_MAX_SUBPIPELINE_DEPTH=3\n' | sudo tee -a /etc/jax/.env >/dev/null
sudo grep -nE '^(JAX_SUBPIPELINE_TOKEN_TTL_SECONDS|JAX_MAX_SUBPIPELINE_DEPTH)=' /etc/jax/.env
```
Expected: el primer `grep` da `0` y el último muestra exactamente las dos líneas nuevas. Si el primero no da 0, no agregar duplicados: revisar qué hay.

- [ ] **Step 6: Pull y reinicio**

Repetir el Step 3 inmediatamente antes (sigue en `()`). Después:
```bash
git -C /home/fruiz/jax pull --ff-only
git -C /home/fruiz/jax log --oneline -1
sudo systemctl restart jax-las-manos.service
systemctl is-active jax-las-manos.service
journalctl -u jax-las-manos.service --since "-3 min" --no-pager | grep -iE "error|traceback|subpipeline|ValueError" || echo "sin errores en el arranque"
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7777/health
```
Expected: el `log` muestra el merge del PR, el servicio queda `active`, el journal dice `sin errores en el arranque` y `/health` responde `200`.

- [ ] **Step 7: Verificar el esquema, el índice y el contrato en producción**

```bash
cat > $S/verificar.py <<'PY'
import asyncio
from jacobs import store
async def main():
    conn = await store.get_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT DATABASE()")
            print("base:", (await cur.fetchone())[0])
            await cur.execute(
                "SELECT COLUMN_NAME FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='jacobs_subpipeline_tokens' ORDER BY ORDINAL_POSITION")
            print("tokens:", [r[0] for r in await cur.fetchall()])
            await cur.execute(
                "SELECT COLUMN_NAME, COLUMN_DEFAULT FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='jacobs_pipelines' "
                "AND COLUMN_NAME IN ('parent_pipeline_id','depth')")
            print("pipelines:", await cur.fetchall())
            await cur.execute(
                "SELECT INDEX_NAME, GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) "
                "FROM information_schema.STATISTICS WHERE TABLE_SCHEMA=DATABASE() "
                "AND TABLE_NAME='jacobs_subpipeline_tokens' GROUP BY INDEX_NAME")
            print("indices:", await cur.fetchall())
            await cur.execute("SELECT depth, COUNT(*) FROM jacobs_pipelines GROUP BY depth")
            print("depth existentes:", await cur.fetchall())
    finally:
        conn.close()
asyncio.run(main())
PY
bash -c 'set -a; . /etc/jax/.env; set +a; cd /home/fruiz/jax && PYTHONPATH=.:las_manos las_manos/.venv/bin/python "$1"' _ $S/verificar.py
curl -s -w '\n%{http_code}\n' -X POST http://127.0.0.1:7777/jacobs/pipeline -H 'Content-Type: application/json' \
  -d '{"name":"verificacion-frente-f","objective":"o","invoked_by":"ada","mode":"dry_run","subpipeline_token":"verificacion-inventado","parent_pipeline_id":"00000000-0000-0000-0000-000000000000"}'
curl -s -w '\n%{http_code}\n' -X POST http://127.0.0.1:7777/jacobs/pipeline -H 'Content-Type: application/json' \
  -d '{"name":"verificacion-frente-f","objective":"o","invoked_by":"plataforma","mode":"dry_run","subpipeline_token":"x"}'
```
Expected:
- `base: jax_memory` y `tokens:` con las 8 columnas en orden.
- `pipelines:` con las dos columnas.
- `indices:` con `PRIMARY → token_hash` e `idx_subpipeline_tokens_padre → parent_pipeline_id`.
- `depth existentes:` todo en 0.
- El primer curl devuelve `403` con `subpipeline_token rechazado: token_desconocido`.
- El segundo curl devuelve `422` (ValidationError de forma).

El primer curl deja **un** evento `SUBPIPELINE_RECHAZADO` en producción: es la evidencia del contrato vivo, y queda declarado en la Biblioteca.

Sobre el EXPLAIN en producción: con 0 tokens reales, el optimizador de MariaDB corta en "Impossible WHERE noticed after reading const tables", y un EXPLAIN de un token inexistente no dice nada del plan. La verificación válida del plan es la de la Task 9, con volumen, más los índices confirmados acá. Queda dicho así.

- [ ] **Step 8: Verificación en vivo con Fernando**

Pedirle a Fernando que cree un pipeline desde la Mesa (axioma-ia.io). Después, con el mismo patrón del Step 7:
```sql
SELECT pipeline_id, invoked_by, parent_pipeline_id, depth FROM jacobs_pipelines ORDER BY created_at DESC LIMIT 1
```
Expected: `invoked_by='plataforma'`, `parent_pipeline_id=NULL`, `depth=0`, y el pipeline corre normal. Anotar el resultado. Borrar `$S` (`rm -rf $S`).

---

### Task 11: Biblioteca

**Files:**
- Modify: `/home/fruiz/jax/DEUDA.md` (vía rama `docs/contrato-subpipelines` y PR, igual que el resto de las entradas; master está protegido)
- Modify: `/home/fruiz/jax/CONTEXT.md` §9
- Create: `/home/fruiz/.claude/projects/-home-fruiz/memory/jax-contrato-subpipelines-2026-09-16.md` y su línea en `MEMORY.md`

- [ ] **Step 1: Escribir las entradas**

En `DEUDA.md`, en la sección de cerrados más reciente, agregar el bloque **"Contrato de sub-pipelines (frente F) — CERRADO 2026-09-16"** con:
- **HISTORIA.** `validate_create` aceptaba cualquier string como token (`policy.py:48-52` en `bd95237`) y la profundidad nunca llegaba (`routes.py:147-153`). Rojo visto en `tests/test_subpipeline_contrato_rutas.py`.
- **DECISIÓN (Fernando, 2026-09-16).** Ada se queda; el contrato se construye completo antes del emisor (Principio IX).
- **VERDAD OPERACIONAL** (verificada con fecha, comando y evidencia): tabla, columnas e índices en `jax_memory`; variables en `/etc/jax/.env`; 403 con token inventado en producción; pipeline de la Mesa con `depth=0`.
- **Números de carga:** la tabla de la Task 8 con fecha, concurrencia, p95 de cada escenario, costo del contrato frente a `base` y punto de degradación. Nota: *caduca si cambia el esquema o el volumen*.
- **EXPLAIN** de la Task 9 con el conteo de filas.
- **Decisiones técnicas con motivo** (las Discrepancias 1, 3, 4 y 5; token quemado por plan rechazado; sin poda de tokens; sin ruta HTTP de emisión).
- **CI:** job `subpipeline-contrato-db` (piso 30) y `tests-puros` +18, con los sha del canario (rojo) y del revert (verde).
- **PENDIENTE con fecha (hoy, 2026-09-16):** sesión de diseño del emisor con Fernando. Llevar las preguntas abiertas:
  1. ¿Un hijo sigue corriendo si su padre se aborta o se cancela? Hoy sí, y habría que decidir si hay cascada.
  2. ¿Qué capacidades puede delegar Ada?
  3. Límites de costo por árbol.
  4. ¿Un hijo exige human gate?
  5. ¿Cómo recibe Ada el token dentro de su paso?

En `CONTEXT.md` §9, una entrada de historial de 5-8 líneas que apunte a DEUDA.md para el detalle.

- [ ] **Step 2: PR de docs y merge**

```bash
git -C /home/fruiz/jax fetch origin
git -C /home/fruiz/jax worktree add /home/fruiz/worktrees/jax-docs-frente-f -b docs/contrato-subpipelines origin/master
```
Hacer las ediciones del Step 1 en ese worktree y después:
```bash
git -C /home/fruiz/worktrees/jax-docs-frente-f add DEUDA.md CONTEXT.md
git -C /home/fruiz/worktrees/jax-docs-frente-f commit -m "docs: contrato de sub-pipelines cerrado y desplegado (frente F)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git -C /home/fruiz/worktrees/jax-docs-frente-f push -u origin docs/contrato-subpipelines
gh pr create --repo fjruizhn/Jax --base master --head docs/contrato-subpipelines --title "docs: contrato de sub-pipelines (frente F)" --body "Registro en la Biblioteca del frente F: evidencia, números de carga, EXPLAIN, CI y el pendiente de hoy (diseño del emisor).

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```
Esperar CI verde y mergear. Después:
```bash
git -C /home/fruiz/jax pull --ff-only
```

- [ ] **Step 3: Memoria y limpieza**

- Crear el archivo de memoria con firma y tipo (HISTORIA / DECISIÓN / VERDAD OPERACIONAL con fecha / PENDIENTE con fecha) y agregar su línea al índice `MEMORY.md`.
- Archivar el ledger: copiar `$WT/.superpowers/sdd/2026-09-16-contrato-subpipelines/` a `/home/fruiz/jax/.superpowers/sdd/`, como con los ledgers anteriores.
- Borrar los worktrees `jax-frente-f` y `jax-docs-frente-f` y las ramas locales ya mergeadas:
  ```bash
  git -C /home/fruiz/jax worktree remove /home/fruiz/worktrees/jax-frente-f
  git -C /home/fruiz/jax worktree remove /home/fruiz/worktrees/jax-docs-frente-f
  git -C /home/fruiz/jax branch -d feat/contrato-subpipelines docs/contrato-subpipelines
  ```
- Cerrar los subagentes entregados.

---

## Solapamientos

| Archivo | Frente F | Frente E | Frente B |
|---|---|---|---|
| `jacobs/policy.py` | Firma y cuerpo de `validate_create`; borra `MAX_SUBPIPELINE_DEPTH`; importa `INVOKER_ADA` | **E-13**: `MAX_STEPS_PER_PIPELINE` se muda a `models.py` (mismas líneas 16-18). **E-20**: borra `hyde_requires_human_gate` (96-98) | Lector del kill switch: `KILL_SWITCH_PATH` → `JAX_KILL_SWITCH_PATH`, error si falta |
| `jacobs/models.py` | `INVOKER_ADA`, `Pipeline.parent_pipeline_id/depth`, `PipelineCreateRequest` | **E-02**: borra `StepResult` (155-160). **E-03**: `VALID_FACETS` (47-49). **E-13**: agrega `MAX_STEPS_PER_PIPELINE`, y el validador de `max_steps` (líneas 129-130) lo importa | — |
| `jacobs/routes.py` | Imports, `plan_only` (ada 403), `create_pipeline` | **E-13**: `plan_only` usa la constante en vez del `20` literal (101-105) | Si B cambia el nombre del lector en `routes`, afecta los `patch.object(routes, "check_kill_switch")` |
| `jacobs/plan.py` | **No se toca** | E-16 (`plan.py:655`), E-17 (`plan.py:799`) | — |
| `jacobs/executor.py` | **No se toca** (solo un test lo parchea) | E-16, E-18, E-24 | Lector del kill switch (`executor.py:38,1122`): el test puro parchea `executor.check_kill_switch` |
| `las_manos/server.py` | `_jacobs_init` llama a `config_subpipelines()` | **E-21**: validación de URLs por entorno al arrancar (mismo lugar conceptual) | Quita `kill_switch_path` de config y exige la variable al arrancar (`server.py:76, 208`) |
| `.github/workflows/policy.yml` | Job nuevo y piso de `tests-puros` | **E-12**: quita `aiofiles` del CI (L557), así que el `pip install` del job nuevo también lo tiene que quitar; pisos de `tests-puros` | Pisos y, si B lo exige, `JAX_KILL_SWITCH_PATH` en el `env:` de los jobs que importan `jacobs.policy` |

**Orden de merge propuesto: E → B → F.**
- E y B son más chicos y cambian las líneas sobre las que F se apoya: constantes de `policy.py`, validador de `models.py`, lector del kill switch y arranque de `server.py`.
- Si F va último, rebasa una vez sobre un árbol ya estable y:
  1. sus tests del kill switch parchean el lector real que dejó B, que según el spec de B sigue siendo `jacobs.policy.check_kill_switch`, con otra fuente;
  2. el `pip install` y el `env:` del job nuevo nacen con los cambios de E-12 y B;
  3. los pisos de `tests-puros` se miden una sola vez sobre la base final (Task 6 Step 6 y Task 7 Step 3 ya dicen "piso de la base + 18").
- Si F llega antes por tiempos, E y B rebasan sobre F. Los conflictos son textuales y chicos (constantes de `policy.py`, bloque del validador en `models.py`, pisos del yml), y lo que se re-mide son los pisos.
- **En cualquier orden:** `git -C $WT rebase origin/master` y la suite de la Task 6 Step 6 antes de pedir revisión.

---

## Self-review (hecho al escribir el plan)

**Cobertura del spec F:**

| Requisito del spec | Dónde se cumple |
|---|---|
| Emisión con `token_urlsafe`, solo hash, columnas, índice por padre, TTL por env | Tasks 2-4 |
| Validación: hash existe, no vencido, no usado, padre activo | Task 5 |
| Consumo atómico con una fila o rechazo | Task 5, con mutación |
| Profundidad desde la fila y `MAX_SUBPIPELINE_DEPTH` por env | Tasks 2, 5 y 6 (`test_el_cuerpo_no_fija_la_profundidad`) |
| El hijo guarda `parent_pipeline_id` y `depth` | Tasks 3 y 6 |
| Eventos `EMITIDO`, `CREADO`, `RECHAZADO` sin el token | Tasks 4, 5 y 6 |
| Kill switch | Task 2 (executor), Task 4 (emisión) y Task 6 (orden, con mutación) |
| Ataques: inventado, reusado, otro padre, vencido, padre terminado, profundidad, carrera | Tasks 1, 5 y 6 |
| Camino legítimo con evento | Task 6 |
| Arnés que hace de Ada | Task 1 |
| Emisor fuera de alcance | Preguntas registradas para la sesión de hoy en la Task 11 |

**Reglas comunes:**

| Regla | Dónde se cumple |
|---|---|
| Rojo contra código viejo | Tasks 1-6 |
| Barrera de base | `_arnes_ada` |
| CI roto a propósito | Task 7 |
| EXPLAIN real | Tasks 5 y 9 |
| Carga con número | Task 8 |
| Nada bloqueante en async | Task 6 Step 6 |
| Mirror-sync | Task 6 Step 6 |
| Deploy con 0 en vuelo | Task 10 |
| Biblioteca | Task 11 |
| i18n y tema | No aplican: sin UI |

**Consistencia de nombres:**
- `SQL_CONSUMIR_TOKEN` con parámetros `(usado_at, hijo, hash, ahora, padre, max)` se usa igual en store (Task 5), en la carrera forzada, en el test de EXPLAIN y en la Task 9.
- `emitir_token_subpipeline(parent_pipeline_id, parent_step) -> str` y `consumir_token_subpipeline(token, parent_pipeline_id, hijo_pipeline_id)` son iguales en las Tasks 4-8.
- `Motivo.*.value` es el mismo en el `detail` 403 y en los eventos.
- `padre_en_ejecucion(depth=, facet_del_paso=)` es igual en las Tasks 1, 3-6 y 8.

**Cuentas:**
- `tests/test_subpipeline_contrato_puro.py` tiene 18 tests.
- `jacobs/_subpipeline_contrato_io_test.py` tiene 19: esquema 5, emisión 7 y consumo 7.
- `tests/test_subpipeline_contrato_rutas.py` tiene 11: 1 de la Task 1 y 10 de la Task 6.
- Job nuevo: 30.
