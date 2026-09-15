# Gobernanza con el catálogo de la DB, `capability.mode` e `invoked_by` como rol (tanda A, v2) — Plan de implementación

> **Para agentes ejecutores:** SUB-SKILL OBLIGATORIA: usar superpowers:subagent-driven-development (recomendado) o superpowers:executing-plans para implementar este plan tarea por tarea. Los pasos usan casillas (`- [ ]`) para el seguimiento.

**Objetivo:** que las facetas de la Mesa puedan citar las capabilities de la DB como `OBSERVADO`, que el resolver de `CAPABILITY_AVAILABLE` las verifique con nombre **y modo** contra el catálogo real, y que `invoked_by` deje de ser un nombre de persona usado como rol (`"Fernando"` → `"plataforma"`).

**Arquitectura:** jax-platform agrega la columna `capability.mode` (NOT NULL, sin default) con una migración idempotente. En jax, `MotorCatalog`/`CapabilityEntry` cargan el modo. El validador sigue puro: recibe el catálogo, y su rama `in_catalog` verifica el modo. El snapshot suma la sección `catalog_capabilities`, sin mover los punteros `/capabilities/N` de las `ops`. En jax-platform, `governance_context.validation_context()` pasa a async: carga `await MotorCatalog.from_db()`, cachea con los mtimes de config más el sello de `facet_resolver`, recarga una sola vez bajo `asyncio.Lock` y falla visible. Jacobs acepta el rol `plataforma`, que jax-platform pone en el backend.

**Stack:** Python 3.14 (venvs de hall9000) / 3.12 y 3.14 (runners de CI), FastAPI, aiomysql, pydantic, pytest + pytest-asyncio, MariaDB (12.3 en hall9000, 11.8 en CI), React 19 + vitest.

**Spec:** `docs/superpowers/specs/2026-09-14-gobernanza-catalogo-db-design.md`, **v2** (repo `jax`, commit `b2bfa43`, enmienda de `0e44acd`). El spec es la autoridad; este plan argumenta desde él. Quien ejecuta lee los dos. Este plan reemplaza al de `0c33478`, que paraba en una "Tarea 0" y quedó resuelta por la decisión de Fernando del 2026-09-14 (spec §0).

## Restricciones globales

Cada tarea las incluye implícitamente.

- **Checkouts de producción intocables.** Ningún comando de las Tareas 1–10 edita, prueba en el lugar ni commitea en `/home/fruiz/jax` ni en `/home/fruiz/jax-platform`: los servicios corren desde ahí. Solo la Tarea 11, después del merge, hace `git pull --ff-only` allí. Importar módulos de esos árboles para medir se hace con `PYTHONDONTWRITEBYTECODE=1`, que no escribe `__pycache__`.
- **Worktrees:**
  - jax → `/home/fruiz/worktrees/jax-gobernanza-catalogo` (rama `feat/gobernanza-catalogo-db`, ya existe; contiene el spec).
  - jax-platform → `/home/fruiz/worktrees/jax-platform-gobernanza`, con **dos ramas**: `feat/capability-mode` (PR-A, Tarea 1, desde `origin/master`) y `feat/gobernanza-catalogo-db` (PR-C, Tareas 6–8, apilada sobre `feat/capability-mode` y rebasada sobre `master` cuando PR-A se mergee).
  - Usar siempre `git -C <ruta>` y rutas absolutas. Si un comando necesita `cd`, `pwd` va en el MISMO comando.
- **Intérpretes:**
  - jax-platform → `/home/fruiz/jax-platform/backend/.venv/bin/python`. Solo se USA el intérprete: no se edita ese árbol. `frontend/node_modules` del worktree es un symlink a `/home/fruiz/jax-platform/frontend/node_modules`.
  - jax → venvs limpios en el scratchpad de quien ejecuta, con EXACTAMENTE el `pip install` del job de CI que corre esos tests (lección "reproducir el runner, no el local"). `/home/fruiz/jax/.venv` no tiene `fastapi` ni `pytest-asyncio`: medido el 2026-09-14. Tres venvs:
    - `venv-jax-gov`: job `governance`, `pip install pytest pyyaml pydantic aiomysql==0.3.2`.
    - `venv-jax-puros`: job `tests-puros`, `pip install pytest pytest-asyncio aiomysql httpx cryptography pydantic pyyaml aiofiles fastapi==0.139.0`.
    - `venv-jax-db`: job `jacobs-gobernanza-db`, `pip install -r <requirements.txt de jax-platform> pytest`.
  - Antes de crearlos, se relee el workflow ACTUAL: si las líneas cambiaron, mandan las del workflow.
- **`JAX_REPO_PATH`:** toda corrida de tests o mediciones de jax-platform en el worktree exporta `JAX_REPO_PATH=/home/fruiz/worktrees/jax-gobernanza-catalogo`, salvo donde el paso diga otra cosa. Sin eso, `governance_context` importa el validador del checkout de producción.
- **Trabajo concurrente:** en `/home/fruiz/worktrees/jax-platform-etapa2` (rama `feat/admin-usuarios-etapa2-sesiones`) corre otro plan que toca auth del backend y sus tests. Este plan no depende de él. **El que mergee segundo rebasa** y en los pisos de CI suma los dos deltas: nunca pisa el número del otro. Ese plan agregó `token_version`/`tv` a los tokens: la firma de `create_access_token` se lee del código desplegado antes de usarla (Tarea 11).
- **TDD:** cada test nuevo se ve en ROJO **por la razón declarada** antes del arreglo. Si falla por otra razón (import, fixture, colección), primero se corrige eso y se vuelve a ver el rojo correcto. Hay tests que pasan antes del cambio a propósito, porque existen para que otro signifique algo o para fijar lo que el cambio no debe mover. Esos se declaran como tales en el paso y se validan por mutación.
- **Controles declarados como controles:** un test que existe para que otro signifique algo lleva en el docstring la palabra `CONTROL` y dice qué controla. Un tripwire lleva `TRIPWIRE`. "Un control que no falla no valida": cada tripwire y cada control se ve en rojo por mutación, no solo en verde.
- **Mutaciones:** se restauran desde un backup real (`cp <archivo> $SCRATCH/<archivo>.bak` antes; `cp` de vuelta después), nunca con `git checkout`, porque sobre un archivo untracked no restaura nada (CONTEXT.md de jax §7). Cada mutación se anota en el cuerpo del PR con el test que la cazó.
- **P10, sin fail-open:** ningún `except` amplio nuevo. Si hiciera falta uno, lleva `# fail-soft: <razón>` en la MISMA línea. Una recarga fallida sube la excepción: no se traga ni se sirve el catálogo viejo o vacío. Una capability sin modo declarado frena la migración.
- **LAS CUATRO:**
  1. Índices: `from_db()` lee tres tablas chicas completas (17 capabilities), sin `WHERE` ni `ORDER BY` sobre tablas que crezcan, y solo al recargar. `mode` no se filtra. Se declara, no hay `EXPLAIN` que hacer.
  2. Caché con invalidación declarada en el mismo commit: el sello de `facet_resolver`, que `run_migrations()` ya estampa.
  3. Solo async: aiomysql, y la lectura de YAML/TOML va en `asyncio.to_thread`.
  4. Latencia en proceso antes y después, y tamaño del snapshot (caracteres, entradas y `axioma_usage.tokens_in` de un turno de sonda) antes y después, escritos en `DEUDA.md` de jax con fecha (Tareas 9 y 11). No se hace carga sobre `/api/chat`, porque llama al modelo, y así se declara.
- **i18n es/en:** este plan no agrega texto visible al usuario. El snapshot es texto de prompt, no de UI. Si una tarea terminara agregando texto de UI, va en `frontend/src/i18n/es.js` y `en.js`.
- **Pisos de CI exactos:** nunca se copia un número de este plan sin medir. Se lee el valor ACTUAL del workflow, porque el plan concurrente puede haberlo subido. Se sube en el delta MEDIDO en la rama y se agrega un comentario con fecha, qué tests y "visto en rojo". Valores leídos el 2026-09-14, solo como referencia:
  - jax: `governance` 80, `tests-puros` 143, `jacobs-gobernanza-db` 14.
  - jax-platform (`origin/master` `bfab4de`): `PISO_PASSED` 611 y `MAX_SKIPS` 1 (job `backend-tests-con-db`), `JAX_CI_MIN_PASSED` 301 (job `backend-tests-no-db`) y vitest 125 (job `frontend-tests`).
  - Los deltas "esperados" de cada tarea son estimaciones para detectar sorpresas. Manda el medido.
- **Commits:** cada mensaje termina con

  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01LNXrHCwsZAVfrQj56maFHB
  ```

  El cuerpo de cada PR termina con `🤖 Generated with [Claude Code](https://claude.com/claude-code)` y la línea `https://claude.ai/code/session_01LNXrHCwsZAVfrQj56maFHB`.
- **PRs:** tres, en este orden de merge (spec §6): **PR-A** jax-platform `feat/capability-mode` → **PR-B** jax `feat/gobernanza-catalogo-db` → **PR-C** jax-platform `feat/gobernanza-catalogo-db`. El gate se mide sobre el `headSha` exacto del PR (`gh run list --commit <sha>`), no sobre "el último run". No se mergea con un check en rojo ni pendiente. El merge y el despliegue van con GO de Fernando.
- **No se pushea nada** hasta la Tarea 10.
- **Despliegue:** primero `jax-platform` (sus migraciones crean la columna) y después `jax-las-manos`, anunciando antes la ventana de segundos en la que crear o reanudar un pipeline puede fallar.

---

## Mapa de archivos

**jax-platform, PR-A** (`/home/fruiz/worktrees/jax-platform-gobernanza`, rama `feat/capability-mode`):

| Archivo | Responsabilidad | Tarea |
|---|---|---|
| `backend/db/migrations.py` | `mode` en `CREATE_CAPABILITY`; `_CAPABILITY_MODE`; `_FILE_CAPABILITY_SEED` a nivel de módulo; semillas con `mode`; `_COLUMNS`; `_backfill_capability_mode`; `_enforce_capability_mode_not_null` | 1 |
| `backend/tests/test_capability_mode.py` (nuevo) | tripwires puros de la semilla; forma de la columna, valores, `INSERT` sin modo, base vieja, fila huérfana | 1 |
| `.github/workflows/policy.yml` | `PISO_PASSED`, `JAX_CI_MIN_PASSED` | 1 |

**jax, PR-B** (`/home/fruiz/worktrees/jax-gobernanza-catalogo`):

| Archivo | Responsabilidad | Tarea |
|---|---|---|
| `las_manos/motor_registry/catalog.py` | `CAPABILITY_MODES`, `CapabilityEntry.mode`, `from_db()` lee `mode`, `MotorCatalog.capabilities()` | 2 |
| `tests/test_motor_catalog_mode.py` (nuevo, `tests-puros`) | constructor por dict, modo inválido, orden | 2 |
| `tests/test_catalog_mode_db.py` (nuevo, `jacobs-gobernanza-db`) | `from_db()` trae el modo sembrado | 2 |
| `policy/governance/validator.py` | `load_validation_context(..., catalog)`; la rama `in_catalog` verifica el modo | 3 |
| `tests/test_governance_validator.py` | `_real_ctx(catalog)`; el tripwire viejo se reemplaza; tests del modo | 3 |
| `policy/governance/grounding.py` | sección `catalog_capabilities`, `SECTION_PREDICATE`, `render()` por sección | 4 |
| `tests/test_governance_grounding.py` | sección nueva, estabilidad de punteros, acreditación, `render`, P10 | 4 |
| `docs/superpowers/specs/2026-09-02-reformas-fase2-sp3-grounding-design.md` | nota fechada en §3.3 | 4 |
| `jacobs/models.py`, `jacobs/policy.py`, `jacobs/routes.py` | rol `plataforma` | 5 |
| `tests/test_jacobs_invoked_by_rol.py` (nuevo, `tests-puros`) | rol al crear, planificar, reanudar y aprobar | 5 |
| `.github/workflows/policy.yml` | pisos de `governance`, `tests-puros`, `jacobs-gobernanza-db`; archivos nuevos en sus listas | 2–5 |
| `DEUDA.md` | cierres, números medidos, nota de SP4 (PR de docs, post-deploy) | 11 |

**jax-platform, PR-C** (mismo worktree, rama `feat/gobernanza-catalogo-db`):

| Archivo | Responsabilidad | Tarea |
|---|---|---|
| `backend/governance_context.py` | `async validation_context()`, clave con el sello, lock, falla visible | 6 |
| `backend/api/chat.py` | `_build_snapshot_or_raise` / `_build_grounding` async; `await` en `chat()` | 6 |
| `backend/shadow_validation.py` | `await _validation_context()` | 6 |
| `backend/tests/test_governance_context_catalogo.py` (nuevo) | catálogo de la DB, sello, lock, DB caída, acreditación real, catálogo cambiado | 6 |
| `backend/tests/test_grounding_config_revalidation.py`, `test_shadow_origin.py`, `test_shadow_validation.py`, `test_shadow_validation_grounding.py` | llamadas async; `from_db` parcheado donde el test es puro | 6 |
| `backend/tests/test_catalogo_db_en_la_mesa.py` (nuevo) | de punta a punta por `run_shadow_validation`; tripwire ops∩DB y su control | 7 |
| `backend/api/pipelines.py`, `backend/tests/test_pipelines_identity_injection.py` | `invoked_by = "plataforma"` al crear y reanudar | 8 |
| `frontend/src/components/BottomBar/PipelineModal.jsx` (+ `.test.jsx`) | deja de mandar `invoked_by` | 8 |
| `.github/workflows/policy.yml` | `PISO_PASSED`, `MAX_SKIPS` si aplica, `JAX_CI_MIN_PASSED`, vitest | 6–8 |

**Por qué el tripwire ops∩DB va en jax-platform (Tarea 7):** jax-platform es el único proceso donde las dos fuentes se juntan en el mismo objeto, el `ValidationContext` de `await governance_context.validation_context()`. Eso es exactamente lo que ven el resolver y el snapshot en producción: `ctx.ops` del TOML clonado de jax y `ctx.catalog` de la DB migrada. En jax habría que recomponer esa unión a mano, y eso sería una segunda definición del contexto.

---

### Tarea 1: columna `capability.mode` (jax-platform, PR-A)

> **ENMIENDA v3 (2026-09-14, spec v3 §0 y §3.1) — manda sobre el texto de esta tarea que la contradiga.**
>
> 1. **Tipo:** `mode VARCHAR(16) NOT NULL` + `CONSTRAINT chk_capability_mode CHECK (mode IN ('read_only','mutating'))`,
>    sin default, en `CREATE_CAPABILITY`. En `_COLUMNS`: `ALTER TABLE capability ADD COLUMN mode VARCHAR(16) NULL`.
>    Donde abajo dice `ENUM('read_only','mutating')`, va este tipo.
> 2. **`_enforce_capability_mode_not_null` pasa a asegurar la forma completa**, idempotente, en este orden:
>    (a) filas con `mode IS NULL` → `RuntimeError` con sus nombres (sin cambios); (b) si `COLUMN_TYPE` no es
>    `varchar(16)` o `IS_NULLABLE = 'YES'` → `ALTER TABLE capability MODIFY COLUMN mode VARCHAR(16) NOT NULL`
>    (convierte también el `ENUM` que ya está en producción, conservando los valores); (c) si en
>    `information_schema.CHECK_CONSTRAINTS` no hay `chk_capability_mode` para `capability` →
>    `ALTER TABLE capability ADD CONSTRAINT chk_capability_mode CHECK (mode IN ('read_only','mutating'))`
>    (repetirlo da 1826, por eso se consulta antes). El nombre de la función puede cambiar a
>    `_asegurar_forma_de_capability_mode`; se declara.
> 3. **Tests** (`test_capability_mode.py`), además de adaptar la forma esperada a
>    `("NO", None, "varchar(16)")` + el CHECK presente con su cláusula:
>    - **Nunca `pytest.raises` dentro de `client.portal.call`:** la corrutina captura y DEVUELVE el error
>      (código o excepción) y la aserción va afuera. `Failed` es `BaseException` y, dentro del portal de
>      sesión, lo mata para el resto de la corrida (medido: 635 → 352 passed / 184 failed / 108 errors).
>    - CONTROL del sin-default: `INSERT` sin `mode` → **1364**.
>    - Test nuevo, CONTROL del CHECK: `INSERT` con `mode='escritura'` → **4025**.
>    - Test nuevo, **el estado de producción de hoy:** la columna como `ENUM('read_only','mutating') NOT NULL`
>      (sin CHECK) → `run_migrations()` la deja `varchar(16)`/`NO`, con el CHECK y los 17 valores intactos.
>    - La base vieja (sin columna) y la fila huérfana siguen, con la aserción fuera del portal. Si
>      `DROP COLUMN mode` choca con el CHECK, se quita el CHECK primero y se declara.
>    - Los rojos esperados cambian donde corresponda (p. ej. la forma da `enum(...)` contra `varchar(16)` con
>      el código a medio hacer): se anota el rojo VISTO con su razón.
> 4. **Endurecer el arnés** (hallazgo de esta tarea; regla de Fernando: sin hallazgos diferidos). En
>    `backend/tests/conftest.py`, el fixture `client` envuelve `c.portal.call` para que una excepción de
>    `pytest.outcomes.OutcomeException` (o cualquier `BaseException` que no sea `KeyboardInterrupt`/
>    `SystemExit`/`GeneratorExit`/cancelación) lanzada DENTRO de la función se capture adentro y se
>    relance AFUERA del portal: el test falla como corresponde y el portal sigue vivo para el resto de la
>    sesión. Test nuevo (`backend/tests/test_arnes_portal.py`, con `client`): un `pytest.fail` dentro de
>    `client.portal.call` sale como `Failed` afuera, y la llamada siguiente al portal sigue funcionando.
>    Rojo visto: sin el envoltorio, la segunda llamada da `RuntimeError: This portal is not running`.
>    Mutación: sacar el envoltorio pone rojo ese test.
> 5. **Mutaciones** además de las de abajo: (d) quitar el paso (c) del CHECK → cae el CONTROL 4025 y la
>    forma; (e) quitar el paso (b) → cae el test del estado ENUM de producción.
> 6. **Pisos:** los deltas cambian (+2 tests de migración y el del arnés): se sube el MEDIDO.
> 7. Commit: `backend/db/migrations.py backend/tests/test_capability_mode.py backend/tests/test_arnes_portal.py
>    backend/tests/conftest.py .github/workflows/policy.yml`; en el mensaje, una línea sobre el tipo v3 y otra
>    sobre el arnés.
> 8. **Efecto en otras tareas:** Tarea 2, el comentario de `CAPABILITY_MODES` dice "mismo conjunto que el
>    CHECK de la columna" en vez de "mismo ENUM". Tarea 11, Paso 3: producción YA tiene la columna
>    (`enum`, incidente del 2026-09-14): el respaldo se llama `capability-pre-v3` y la precondición es
>    "17 filas, solo `file_write` mutating, 0 NULL"; Paso 4: la columna esperada es `NO`, `NULL`,
>    `varchar(16)` y `chk_capability_mode` presente en `information_schema.CHECK_CONSTRAINTS`.

**Archivos:**
- Modificar: `backend/db/migrations.py`, en estos puntos: `CREATE_CAPABILITY` (~402-440); después de `_CAPABILITY_SEED` (~589-630); `_seed_motors_and_capabilities` (~653-712); `_seed_file_tools_capabilities` (~761-840); `_COLUMNS` (~1098-1266); `run_migrations` (~1625-1679).
- Crear: `backend/tests/test_capability_mode.py`
- Modificar: `.github/workflows/policy.yml` (`PISO_PASSED`, `JAX_CI_MIN_PASSED`)

**Interfaces:**
- Produce:
  - `migrations._CAPABILITY_MODE: dict[str, str]`, con las 17 capabilities sembradas.
  - `migrations._FILE_CAPABILITY_SEED: list[tuple]`, la lista que hoy es local en `_seed_file_tools_capabilities`.
  - La columna `capability.mode ENUM('read_only','mutating') NOT NULL`, sin default.
- La Tarea 2 (jax) la lee con `SELECT ... mode FROM capability`.

- [ ] **Paso 1: crear el worktree y medir la base**

```bash
git -C /home/fruiz/jax-platform fetch origin
git -C /home/fruiz/jax-platform worktree add -b feat/capability-mode /home/fruiz/worktrees/jax-platform-gobernanza origin/master
ln -s /home/fruiz/jax-platform/frontend/node_modules /home/fruiz/worktrees/jax-platform-gobernanza/frontend/node_modules
git -C /home/fruiz/worktrees/jax-platform-gobernanza log --oneline -1   # anotar el sha base
grep -rn "INTO capability" /home/fruiz/worktrees/jax-platform-gobernanza/backend/tests /home/fruiz/worktrees/jax-gobernanza-catalogo/tests /home/fruiz/worktrees/jax-gobernanza-catalogo/jacobs /home/fruiz/worktrees/jax-gobernanza-catalogo/las_manos --include=*.py
```

Esperado del `grep`: nada (medido el 2026-09-14). Si aparece un test que inserta en `capability`, ese `INSERT` pasa a declarar `mode`, en este commit si es de jax-platform y en la Tarea 2 si es de jax. Con la columna sin default, ese test fallaría; en jax, además, fallaría en CI apenas se mergee PR-A.

Base de las dos modalidades de CI, sobre el worktree sin cambios (el código de jax del worktree de jax todavía es igual a `master`):

```bash
cd /home/fruiz/worktrees/jax-platform-gobernanza/backend && pwd && export JAX_REPO_PATH=/home/fruiz/worktrees/jax-gobernanza-catalogo && \
  /home/fruiz/jax-platform/backend/.venv/bin/python -m pytest -q -rs 2>&1 | tail -3 | tee "$SCRATCH/base-con-db.txt" && \
  JAX_CI_NO_DB=1 /home/fruiz/jax-platform/backend/.venv/bin/python -m pytest -q -rs 2>&1 | tail -3 | tee "$SCRATCH/base-no-db.txt"
```

Anotar passed/skipped de cada una: son la base de los deltas de las Tareas 1, 6 y 7.

- [ ] **Paso 2: escribir los tests (rojos)**

Crear `backend/tests/test_capability_mode.py`:

```python
"""
capability.mode (tanda A v2, spec 2026-09-14 §3.1).

`mode` dice si una capability cambia el estado del sistema ('mutating') o
solo produce texto/parches sin aplicarlos ('read_only'). Decisión de
Fernando (2026-09-14): 'mutating' SOLO file_write. Lo lee
MotorCatalog.from_db() (jax) y lo verifica el resolver de
CAPABILITY_AVAILABLE contra lo que afirma una faceta.

NOT NULL y SIN DEFAULT a propósito: un default 'read_only' haría nacer de
solo lectura a la próxima capability que mute (fail-open), y uno 'mutating'
mentiría igual de invisible. Sin default, un INSERT sin modo falla
(STRICT_TRANS_TABLES, medido en producción el 2026-09-14): el modo se
declara o la fila no entra.

Los dos primeros son PUROS. El resto usa `client` (base migrada) y, cuando
rompe el esquema a propósito, lo deja como estaba con run_migrations().
"""
from __future__ import annotations

import pymysql
import pytest

from db import migrations

_INFO_MODE = (
    "SELECT IS_NULLABLE, COLUMN_DEFAULT, COLUMN_TYPE FROM information_schema.COLUMNS "
    "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'capability' AND COLUMN_NAME = 'mode'"
)


def _sembradas() -> set[str]:
    return ({fila[0] for fila in migrations._CAPABILITY_SEED}
            | {fila[0] for fila in migrations._FILE_CAPABILITY_SEED})


def test_tripwire_cada_capability_sembrada_declara_su_modo():
    """TRIPWIRE. Una capability nueva en la semilla sin su modo en
    _CAPABILITY_MODE rompe acá, antes de que su INSERT falle en producción."""
    assert set(migrations._CAPABILITY_MODE) == _sembradas()
    assert set(migrations._CAPABILITY_MODE.values()) <= {"read_only", "mutating"}


def test_solo_file_write_es_mutating():
    """Decisión de Fernando, 2026-09-14. Cambiarla es cambiar este test a
    propósito, con fecha y quién decidió."""
    assert {k for k, v in migrations._CAPABILITY_MODE.items() if v == "mutating"} == {"file_write"}


async def _sql(sentencia, args=None):
    from db.connection import get_pool
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sentencia, args)
            return await cur.fetchall()


def test_la_columna_mode_es_not_null_y_sin_default(client):
    assert client.portal.call(_sql, _INFO_MODE) == (("NO", None, "enum('read_only','mutating')"),)


def test_los_valores_sembrados_son_los_de_capability_mode(client):
    filas = dict(client.portal.call(_sql, "SELECT `key`, mode FROM capability"))
    assert {k: filas.get(k) for k in migrations._CAPABILITY_MODE} == migrations._CAPABILITY_MODE
    assert {k for k, v in filas.items() if v == "mutating"} == {"file_write"}


def test_control_un_insert_sin_mode_falla(client):
    """CONTROL de la decisión "sin default": si alguien le pone DEFAULT a
    la columna, este INSERT entra y el test se pone rojo."""
    async def correr():
        try:
            with pytest.raises(pymysql.err.MySQLError) as error:
                await _sql("INSERT INTO capability (`key`, risk_level, max_execution_minutes, allowed_callers) "
                           "VALUES ('zz_sin_modo', 'low', 5, '[]')")
            return error.value.args[0]
        finally:
            await _sql("DELETE FROM capability WHERE `key` = 'zz_sin_modo'")

    assert client.portal.call(correr) == 1364  # ER_NO_DEFAULT_FOR_FIELD


def test_una_base_vieja_queda_rellenada_y_not_null(client):
    """Producción hoy: la tabla existe sin la columna. run_migrations() la
    agrega NULL, rellena desde _CAPABILITY_MODE y la pasa a NOT NULL."""
    async def correr():
        await _sql("ALTER TABLE capability DROP COLUMN mode")
        await migrations.run_migrations()
        return await _sql(_INFO_MODE), dict(await _sql("SELECT `key`, mode FROM capability"))

    info, filas = client.portal.call(correr)
    assert info == (("NO", None, "enum('read_only','mutating')"),)
    assert {k: filas.get(k) for k in migrations._CAPABILITY_MODE} == migrations._CAPABILITY_MODE


def test_una_capability_no_sembrada_sin_modo_frena_la_migracion(client):
    """Una fila que ninguna migración sembró (SQL a mano) no recibe un modo
    inventado: la migración falla con su nombre y jax-platform no arranca."""
    async def correr():
        await _sql("ALTER TABLE capability MODIFY COLUMN mode ENUM('read_only','mutating') NULL")
        await _sql("INSERT INTO capability (`key`, risk_level, max_execution_minutes, allowed_callers) "
                   "VALUES ('zz_huerfana', 'low', 5, '[]')")
        try:
            with pytest.raises(RuntimeError, match="zz_huerfana"):
                await migrations.run_migrations()
        finally:
            await _sql("DELETE FROM capability WHERE `key` = 'zz_huerfana'")
            await migrations.run_migrations()
        return await _sql(_INFO_MODE)

    assert client.portal.call(correr) == (("NO", None, "enum('read_only','mutating')"),)
```

- [ ] **Paso 3: correr y ver el rojo**

```bash
cd /home/fruiz/worktrees/jax-platform-gobernanza/backend && pwd && JAX_REPO_PATH=/home/fruiz/worktrees/jax-gobernanza-catalogo \
  /home/fruiz/jax-platform/backend/.venv/bin/python -m pytest tests/test_capability_mode.py -v 2>&1 | tail -15
```

Esperado: los 7 en ROJO, todos porque la columna y la semilla todavía no existen:
- Los 2 puros dan `AttributeError: module 'db.migrations' has no attribute '_CAPABILITY_MODE'`.
- La forma de la columna da `assert () == (('NO', None, ...),)`.
- Los valores dan un error 1054 `Unknown column 'mode'`.
- El control da `DID NOT RAISE`: sin la columna, el `INSERT` entra.
- La base vieja da 1091 `Can't DROP COLUMN 'mode'`.
- La huérfana da 1054 en el `MODIFY`.

- [ ] **Paso 4: implementar en `migrations.py`**

En `CREATE_CAPABILITY`, después de `requires_human_gate BOOLEAN NOT NULL DEFAULT FALSE,`:

```sql
  -- Tanda A v2 (2026-09-14, decisión de Fernando): ¿la capability cambia el
  -- estado del sistema? 'mutating' solo file_write; el resto produce texto o
  -- parches sin aplicarlos. NOT NULL y SIN DEFAULT a propósito: un INSERT
  -- sin modo falla (STRICT_TRANS_TABLES). Lo lee MotorCatalog.from_db() (jax)
  -- y lo verifica el resolver de CAPABILITY_AVAILABLE. Spec jax
  -- 2026-09-14-gobernanza-catalogo-db-design.md §3.1.
  mode ENUM('read_only','mutating') NOT NULL,
```

Sacar la lista `file_capabilities` de `_seed_file_tools_capabilities` a nivel de módulo, justo después de `_CAPABILITY_SEED`, sin cambiar su contenido ni sus comentarios, con el nombre `_FILE_CAPABILITY_SEED`. Debajo:

```python
# Modo de cada capability sembrada (tanda A v2, 2026-09-14, decisión de
# Fernando): 'mutating' SOLO file_write, la única que cambia el estado del
# sistema. Las demás producen texto o parches sin aplicarlos. UNA fuente:
# la usan los INSERT de las semillas y _backfill_capability_mode. Sin
# default a propósito (ver CREATE_CAPABILITY). tests/test_capability_mode.py
# es el tripwire de que cubre exactamente lo sembrado.
_CAPABILITY_MODE: dict[str, str] = {
    **{fila[0]: "read_only" for fila in _CAPABILITY_SEED},
    "file_read": "read_only",
    "file_write": "mutating",
}
```

En los DOS `INSERT IGNORE INTO capability` (`_seed_motors_and_capabilities` y `_seed_file_tools_capabilities`), agregar `mode` a la lista de columnas, un `%s` más, y `_CAPABILITY_MODE[key]` como último valor. El `for` de `_seed_file_tools_capabilities` pasa a iterar `_FILE_CAPABILITY_SEED`. `_CAPABILITY_MODE[key]` es `KeyError` si falta el modo: ruidoso, y el tripwire puro lo ve antes.

Al final de `_COLUMNS`:

```python
    # Tanda A v2 (2026-09-14): en bases que ya tienen `capability` nace NULL
    # A PROPÓSITO. Un ADD COLUMN ... NOT NULL sin default le pondría a las
    # filas existentes el primer valor del ENUM ('read_only') -- file_write
    # quedaría de solo lectura en silencio hasta el UPDATE, y el DDL hace
    # commit implícito. _backfill_capability_mode la rellena desde
    # _CAPABILITY_MODE y _enforce_capability_mode_not_null la pasa a NOT NULL.
    ("capability", "mode",
     "ALTER TABLE capability ADD COLUMN mode ENUM('read_only','mutating') NULL"),
```

Funciones nuevas, antes de `run_migrations`:

```python
async def _backfill_capability_mode(cur) -> None:
    """Rellena `mode` en las filas que ya existían sin columna. Solo toca
    NULL: nunca pisa un modo ya declarado."""
    for key, mode in _CAPABILITY_MODE.items():
        await cur.execute(
            "UPDATE capability SET mode=%s WHERE `key`=%s AND mode IS NULL", (mode, key)
        )


async def _enforce_capability_mode_not_null(cur) -> None:
    """Si queda una capability sin modo (una fila que ninguna migración
    sembró), la migración FALLA con su nombre: no se le inventa un modo
    (P10). Si no queda ninguna y la columna todavía admite NULL, pasa a
    NOT NULL, sin default."""
    await cur.execute("SELECT `key` FROM capability WHERE mode IS NULL ORDER BY `key`")
    sin_modo = [fila[0] for fila in await cur.fetchall()]
    if sin_modo:
        raise RuntimeError(
            f"capability sin mode declarado: {sin_modo}. Ninguna migración las sembró: "
            "declarar su modo en _CAPABILITY_MODE (db/migrations.py) o borrarlas. "
            "Sin default a propósito, ver spec jax 2026-09-14-gobernanza-catalogo-db-design.md §3.1."
        )
    await cur.execute(
        "SELECT IS_NULLABLE FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'capability' AND COLUMN_NAME = 'mode'"
    )
    fila = await cur.fetchone()
    if fila is not None and fila[0] == "YES":
        await cur.execute(
            "ALTER TABLE capability MODIFY COLUMN mode ENUM('read_only','mutating') NOT NULL"
        )
```

En `run_migrations`, después de `await _raise_generate_execution_ceiling(cur)`:

```python
            # Después de TODAS las semillas de capability: las filas nuevas ya
            # entraron con su modo; las viejas se rellenan y la columna queda
            # NOT NULL. Una fila huérfana frena acá (ver la función).
            await _backfill_capability_mode(cur)
            await _enforce_capability_mode_not_null(cur)
```

- [ ] **Paso 5: ver el verde, en las dos modalidades**

```bash
cd /home/fruiz/worktrees/jax-platform-gobernanza/backend && pwd && export JAX_REPO_PATH=/home/fruiz/worktrees/jax-gobernanza-catalogo && \
  /home/fruiz/jax-platform/backend/.venv/bin/python -m pytest tests/test_capability_mode.py -v 2>&1 | tail -10 && \
  /home/fruiz/jax-platform/backend/.venv/bin/python -m pytest -q -rs 2>&1 | tail -3 && \
  JAX_CI_NO_DB=1 /home/fruiz/jax-platform/backend/.venv/bin/python -m pytest -q -rs 2>&1 | tail -3
```

Esperado: `7 passed`. Suite completa sin fallos ni errores. Contra la base: con DB +7 passed y los mismos skips; sin DB +2 passed y +5 skipped.

**Mutaciones** (backup con `cp`, restauración con `cp`):
- (a) `DEFAULT 'read_only'` en el `MODIFY` y en el `CREATE`: caen `test_control_un_insert_sin_mode_falla` y la forma de la columna.
- (b) Sacar el `raise` de `_enforce_capability_mode_not_null`: cae `test_una_capability_no_sembrada_sin_modo_frena_la_migracion`. El `MODIFY ... NOT NULL` con filas NULL da otro error, no `RuntimeError` con el nombre.
- (c) `"file_write": "read_only"` en `_CAPABILITY_MODE`: cae `test_solo_file_write_es_mutating`.

Después de las mutaciones, correr una vez `tests/test_capability_mode.py` para dejar la base de tests migrada con el código bueno.

- [ ] **Paso 6: pisos**

En `.github/workflows/policy.yml`, leer `PISO_PASSED` y `JAX_CI_MIN_PASSED` ACTUALES y subirlos por el delta medido en el Paso 5 (esperado +7 y +2). Comentario en el estilo del archivo:

```text
# Subido de <actual> a <actual+7> (2026-09-14, tanda A v2, capability.mode):
# tests/test_capability_mode.py -- 2 puros (la semilla declara el modo de
# todo lo sembrado; solo file_write es mutating) y 5 con `client` (columna
# NOT NULL sin default, valores sembrados, CONTROL: INSERT sin mode falla
# 1364, base vieja rellenada, fila huérfana frena la migración). Vistos en
# rojo antes del arreglo; mutaciones en el PR. Medido: <passed> / <skips>.
```

Para `JAX_CI_MIN_PASSED`, el mismo comentario con "+2 puros; los 5 con `client` se saltean". `MAX_SKIPS` (job con DB) no cambia: los 5 corren con DB.

- [ ] **Paso 7: commit**

```bash
git -C /home/fruiz/worktrees/jax-platform-gobernanza add backend/db/migrations.py backend/tests/test_capability_mode.py .github/workflows/policy.yml
git -C /home/fruiz/worktrees/jax-platform-gobernanza commit -m "$(cat <<'EOF'
feat(migraciones): capability.mode, NOT NULL y sin default

Columna nueva ENUM('read_only','mutating'). Sembrada desde
_CAPABILITY_MODE: mutating solo file_write (decisión de Fernando,
2026-09-14). En bases existentes nace NULL, se rellena y pasa a NOT NULL;
una capability no sembrada frena la migración con su nombre. Sin default:
un INSERT sin modo falla. Tripwire de la semilla y CONTROL del sin-default.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LNXrHCwsZAVfrQj56maFHB
EOF
)"
```

---

### Tarea 2: `MotorCatalog` lleva el modo (jax, PR-B)

**Archivos:**
- Modificar: `las_manos/motor_registry/catalog.py` (dataclass `CapabilityEntry` ~66-83, `_load` ~119-134, `from_db` ~217-252, método nuevo)
- Crear: `tests/test_motor_catalog_mode.py`, `tests/test_catalog_mode_db.py`
- Modificar: `.github/workflows/policy.yml`, jobs `tests-puros` y `jacobs-gobernanza-db`

**Interfaces:**
- Consume: la columna `capability.mode` (Tarea 1).
- Produce, todo lo que usan las Tareas 3 y 4:
  - `catalog.CAPABILITY_MODES = frozenset({"read_only", "mutating"})`.
  - `CapabilityEntry.mode: str`, con default `"mutating"` solo en el constructor por dict.
  - `MotorCatalog.capabilities() -> tuple[CapabilityEntry, ...]`, ordenado por `name`.

- [ ] **Paso 1: venvs limpios** (líneas leídas del workflow actual; ver Restricciones)

```bash
python3.14 -m venv "$SCRATCH/venv-jax-puros" && "$SCRATCH/venv-jax-puros/bin/pip" install -q pytest pytest-asyncio aiomysql httpx cryptography pydantic pyyaml aiofiles fastapi==0.139.0
python3.14 -m venv "$SCRATCH/venv-jax-db" && "$SCRATCH/venv-jax-db/bin/pip" install -q -r /home/fruiz/worktrees/jax-platform-gobernanza/backend/requirements.txt pytest
git -C /home/fruiz/worktrees/jax-gobernanza-catalogo status --short   # esperado: vacío
```

- [ ] **Paso 2: tests (rojos)**

`tests/test_motor_catalog_mode.py`:

```python
"""
MotorCatalog lleva el modo de cada capability (tanda A v2, spec 2026-09-14 §3.2).

`mode` sale de la columna capability.mode (jax-platform, NOT NULL sin
default). El constructor por dict (solo tests) usa 'mutating' si no se
declara: fail-closed, en la línea de risk_level='high' y
requires_human_gate=True. Un claim de solo lectura sobre una capability sin
modo declarado da FACT_MISMATCH, nunca VALID.
"""
from __future__ import annotations

import pytest

from las_manos.motor_registry.catalog import CAPABILITY_MODES, MotorCatalog


def test_los_modos_son_los_del_enum_de_la_columna():
    assert CAPABILITY_MODES == frozenset({"read_only", "mutating"})


def test_sin_mode_declarado_el_constructor_por_dict_es_fail_closed():
    assert MotorCatalog({"capabilities": {"x": {}}}).get_capability("x").mode == "mutating"


def test_el_mode_declarado_se_respeta():
    assert MotorCatalog({"capabilities": {"x": {"mode": "read_only"}}}).get_capability("x").mode == "read_only"


def test_un_mode_invalido_lanza_con_el_nombre_de_la_capability():
    with pytest.raises(RuntimeError, match="'x'"):
        MotorCatalog({"capabilities": {"x": {"mode": "escritura"}}})


def test_capabilities_sale_ordenado_por_nombre():
    cat = MotorCatalog({"capabilities": {"b": {}, "a": {}, "c": {}}})
    assert [c.name for c in cat.capabilities()] == ["a", "b", "c"]
```

`tests/test_catalog_mode_db.py`:

```python
"""
MotorCatalog.from_db() trae capability.mode (tanda A v2). Corre en el job
jacobs-gobernanza-db, cuya base la crean las migraciones de jax-platform
(clonado de master): requiere PR-A (capability.mode) mergeado. Solo lee.
"""
from __future__ import annotations

import asyncio

from las_manos.motor_registry.catalog import MotorCatalog


def test_from_db_trae_el_modo_sembrado_de_cada_capability():
    modos = {c.name: c.mode for c in asyncio.run(MotorCatalog.from_db()).capabilities()}
    assert modos["file_write"] == "mutating"
    assert modos["generate"] == "read_only"
    assert {n for n, m in modos.items() if m == "mutating"} == {"file_write"}
```

- [ ] **Paso 3: ver el rojo**

```bash
cd /home/fruiz/worktrees/jax-gobernanza-catalogo && pwd && PYTHONPATH=.:las_manos "$SCRATCH/venv-jax-puros/bin/python" -m pytest tests/test_motor_catalog_mode.py -v 2>&1 | tail -8
cd /home/fruiz/worktrees/jax-gobernanza-catalogo && pwd && set -a && . /etc/jax/.env && set +a && JAX_DB_NAME=jax_memory_test \
  PYTHONPATH=.:las_manos "$SCRATCH/venv-jax-db/bin/python" -m pytest tests/test_catalog_mode_db.py -v 2>&1 | tail -6
```

Esperado:
- En el primero, error de colección con `ImportError: cannot import name 'CAPABILITY_MODES'`. Es la razón declarada: el catálogo todavía no conoce el modo.
- En el segundo, `AttributeError: 'MotorCatalog' object has no attribute 'capabilities'`, contra `jax_memory_test`, que la Tarea 1 dejó migrada con la columna (el `client` de jax-platform la migra). Si da `Unknown column 'mode'`, esa base no pasó por la Tarea 1: correr antes `tests/test_capability_mode.py` de jax-platform.

- [ ] **Paso 4: implementar en `catalog.py`**

Debajo de los imports:

```python
# capability.mode (jax-platform, db/migrations.py): ¿la capability cambia el
# estado del sistema? Mismo ENUM que la columna. Tanda A v2, 2026-09-14.
CAPABILITY_MODES = frozenset({"read_only", "mutating"})


def _modo_valido(capability: str, mode: object) -> str:
    """La DB garantiza ENUM y NOT NULL; si eso se rompe, se ve (P10)."""
    if mode not in CAPABILITY_MODES:
        raise RuntimeError(
            f"capability '{capability}': mode {mode!r} fuera de {sorted(CAPABILITY_MODES)} "
            "(ver capability.mode en jax-platform/backend/db/migrations.py)."
        )
    return mode  # type: ignore[return-value]
```

Al final de `CapabilityEntry`:

```python
    # Tanda A v2 (2026-09-14): 'read_only' | 'mutating', de capability.mode
    # (from_db lo pasa SIEMPRE explícito). El default solo lo usa el
    # constructor por dict (tests) y es fail-closed, como risk_level='high'.
    mode: str = "mutating"
```

En `_load`, dentro del `CapabilityEntry(...)` de capabilities: `mode=_modo_valido(name, cfg.get("mode", "mutating")),`.

En `from_db`, el `SELECT` de `capability` termina en `"       auditor_motor, mode "`. El `for` desempaqueta `(..., fallback_mode, callers, forbidden, auditor_motor, mode)` y el `CapabilityEntry(...)` suma `mode=_modo_valido(key, mode),`. Actualizar el docstring de `from_db`: "motor/capability (con `mode`, tanda A v2)/capability_motor".

Método nuevo, después de `get_capability`:

```python
    def capabilities(self) -> tuple[CapabilityEntry, ...]:
        """Todas las capabilities, ordenadas por nombre. Lo usa
        policy/governance/grounding.build_snapshot: el orden fijo es lo que
        hace deterministas los punteros y el hash del snapshot."""
        return tuple(self._capabilities[n] for n in sorted(self._capabilities))
```

- [ ] **Paso 5: ver el verde, con las listas completas de los dos jobs**

Correr el paso "Piso exacto" del job `tests-puros`, copiado del workflow ACTUAL, con `tests/test_motor_catalog_mode.py` agregado al final. Correr también el de `jacobs-gobernanza-db` con `tests/test_catalog_mode_db.py` agregado, con `venv-jax-db`, `/etc/jax/.env` y `JAX_DB_NAME=jax_memory_test`.

Esperado: `tests-puros` = piso actual + 5 (hoy 148) y `jacobs-gobernanza-db` = piso actual + 1 (hoy 15). Los tests existentes que arman `MotorCatalog(dict)` siguen verdes: el campo nuevo tiene default.

**Mutación:** hacer que `_modo_valido` devuelva `mode` sin chequear: cae `test_un_mode_invalido_lanza_con_el_nombre_de_la_capability`.

- [ ] **Paso 6: workflow**

- Job `tests-puros`: `tests/test_motor_catalog_mode.py` en las DOS listas (el `run: >-` y el "Piso exacto"). Piso de `143` al medido, con este comentario:
  `# <actual> -> <medido> el 2026-09-14 (tanda A v2): test_motor_catalog_mode.py (+5) -- CapabilityEntry.mode, fail-closed por dict, modo inválido lanza, capabilities() ordenado. Vistos en rojo.`
- Job `jacobs-gobernanza-db`: `tests/test_catalog_mode_db.py` en sus DOS listas. Piso `14` → medido, con este comentario:
  `# <actual> -> <medido> el 2026-09-14 (tanda A v2): test_catalog_mode_db.py (+1) -- from_db() trae capability.mode. Requiere jax-platform con capability.mode (PR-A) en master. Visto en rojo.`

- [ ] **Paso 7: commit**

```bash
git -C /home/fruiz/worktrees/jax-gobernanza-catalogo add las_manos/motor_registry/catalog.py tests/test_motor_catalog_mode.py tests/test_catalog_mode_db.py .github/workflows/policy.yml
git -C /home/fruiz/worktrees/jax-gobernanza-catalogo commit -m "$(cat <<'EOF'
feat(motor_registry): CapabilityEntry.mode desde capability.mode

from_db() lee la columna nueva de jax-platform y valida el ENUM; el
constructor por dict usa 'mutating' si no se declara (fail-closed).
MotorCatalog.capabilities() ordenado por nombre, para el snapshot.
tests-puros +5, jacobs-gobernanza-db +1.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LNXrHCwsZAVfrQj56maFHB
EOF
)"
```

---

### Tarea 3: el validador recibe el catálogo y verifica su modo (jax, PR-B)

**Archivos:**
- Modificar: `policy/governance/validator.py:78-90` (`load_validation_context`) y `:185-193` (rama `in_catalog`)
- Modificar: `tests/test_governance_validator.py`: `_real_ctx` (`:229-231`), `test_capability_available_catalog_branch_with_synthetic_catalog_is_valid` (`:254-269`), el tripwire `:272-301`, y tests nuevos
- Modificar: `.github/workflows/policy.yml`, job `governance`

**Interfaces:**
- Consume: `CapabilityEntry.mode` (Tarea 2).
- Produce: `validator.load_validation_context(repo_root: Path, config_paths_allowlist: frozenset[str], catalog: MotorCatalog) -> ValidationContext`. `ctx.catalog` es el objeto recibido (identidad, no copia). La Tarea 6 lo llama con `await MotorCatalog.from_db()`.

- [ ] **Paso 1: venv limpio del job `governance`**

```bash
python3.14 -m venv "$SCRATCH/venv-jax-gov" && "$SCRATCH/venv-jax-gov/bin/pip" install -q pytest pyyaml pydantic aiomysql==0.3.2
```

- [ ] **Paso 2: tests (rojos)**

Reemplazar `_real_ctx`:

```python
def _real_ctx(catalog: MotorCatalog | None = None) -> "validator.ValidationContext":
    """ops y allowlist REALES (config.toml y closed_vocabulary.yaml del repo).
    El catálogo lo pasa el test: desde 2026-09-14 el validador ya no lo arma
    del TOML (spec tanda A, §3.3). Sin catálogo explícito va uno vacío, que es
    lo que necesitan los tests de la rama `ops`."""
    vocab = loaders.load_vocabulary()
    return validator.load_validation_context(
        REPO_ROOT, vocab.config_paths, catalog if catalog is not None else MotorCatalog({})
    )
```

En `test_capability_available_catalog_branch_with_synthetic_catalog_is_valid`:
- El catálogo pasa a `MotorCatalog({"capabilities": {"code_swarm": {"allowed_motors": ["kimi"], "mode": "read_only"}}})`.
- El docstring suma: "Desde la tanda A v2 (2026-09-14) la rama verifica el modo: el catálogo lo declara explícito, y sin declararlo el constructor por dict da 'mutating' (fail-closed)."
- Es una adaptación: el test no es nuevo y no se espera verlo en rojo.

Borrar entero `test_real_toml_catalog_is_empty_since_block3_so_catalog_branch_is_dead_in_production` y agregar en su lugar:

```python
def test_load_validation_context_usa_el_catalogo_que_recibe():
    """Reemplaza al tripwire `test_real_toml_catalog_is_empty_since_block3_so_
    catalog_branch_is_dead_in_production` (2026-09-02 -> 2026-09-14). Ese test
    fijaba que el validador armaba el catálogo desde el TOML vacío; el arreglo
    de la tanda A (spec 2026-09-14 v2, §3.3) lo pone rojo, como estaba
    previsto, y este describe el estado nuevo: el catálogo es el que pasa el
    llamador (en producción, jax-platform con `await MotorCatalog.from_db()`).

    Lo que pedía el tripwire, hecho: DEUDA.md se cierra en el despliegue; SP3
    §3.3 lleva una nota fechada; `grounding.SECTION_PREDICATE` suma
    `catalog_capabilities` (tests/test_governance_grounding.py)."""
    catalogo = MotorCatalog({"capabilities": {"generate": {"allowed_motors": ["kimi"], "mode": "read_only"}}})
    ctx = _real_ctx(catalogo)
    assert ctx.catalog is catalogo
    claim = _claim(predicate="CAPABILITY_AVAILABLE", args={"name": "generate", "mode": "read_only"})
    verdict = validator.validate(claim, PREDICATES, ctx)
    assert verdict.status == "VALID", verdict.detail
    assert "catálogo" in verdict.detail


def test_control_sin_la_capability_en_el_catalogo_el_mismo_claim_es_fact_mismatch():
    """CONTROL de `test_load_validation_context_usa_el_catalogo_que_recibe`:
    mismo claim, mismo repo, catálogo vacío -> FACT_MISMATCH. El VALID de
    arriba sale del catálogo recibido y no de `ops` ni de otra fuente."""
    ctx = _real_ctx(MotorCatalog({}))
    claim = _claim(predicate="CAPABILITY_AVAILABLE", args={"name": "generate", "mode": "read_only"})
    assert validator.validate(claim, PREDICATES, ctx).status == "FACT_MISMATCH"


def test_rama_catalogo_con_el_modo_equivocado_es_fact_mismatch():
    """Tanda A v2: el catálogo tiene el modo (capability.mode). Afirmar que
    file_write es de solo lectura es falso."""
    ctx = _real_ctx(MotorCatalog({"capabilities": {"file_write": {"mode": "mutating"}}}))
    claim = _claim(predicate="CAPABILITY_AVAILABLE", args={"name": "file_write", "mode": "read_only"})
    verdict = validator.validate(claim, PREDICATES, ctx)
    assert verdict.status == "FACT_MISMATCH"
    assert "mutating" in verdict.detail


def test_rama_catalogo_con_el_modo_real_mutating_es_valid():
    ctx = _real_ctx(MotorCatalog({"capabilities": {"file_write": {"mode": "mutating"}}}))
    claim = _claim(predicate="CAPABILITY_AVAILABLE", args={"name": "file_write", "mode": "mutating"})
    assert validator.validate(claim, PREDICATES, ctx).status == "VALID"
```

- [ ] **Paso 3: ver el rojo**

```bash
cd /home/fruiz/worktrees/jax-gobernanza-catalogo && pwd && "$SCRATCH/venv-jax-gov/bin/python" -m pytest tests/test_governance_validator.py -q 2>&1 | tail -15
```

Esperado:
- ROJO en todo test que use `_real_ctx`, con `TypeError: load_validation_context() takes 2 positional arguments but 3 were given`.
- Una vez corregida la firma (Paso 4, primera mitad), sigue ROJO `test_rama_catalogo_con_el_modo_equivocado_es_fact_mismatch`: da `VALID` por "mode no verificable ahí".
- Así se ven los dos rojos por su razón: primero la firma, después el resolver.

- [ ] **Paso 4: implementar**

`load_validation_context`:

```python
def load_validation_context(
    repo_root: Path,
    config_paths_allowlist: frozenset[str],
    catalog: MotorCatalog,
) -> ValidationContext:
    """`ops` sale de `las_manos/config.toml`; el catálogo de capabilities lo
    RECIBE. El validador no toca la DB: el llamador lo carga (jax-platform,
    `governance_context`, con `await MotorCatalog.from_db()`).

    Hasta el 2026-09-14 se armaba acá con `MotorCatalog(config)` desde el
    TOML, cuyo `[capabilities.*]` quedó vacío con el Bloque 3: la rama
    `in_catalog` de `_resolve_capability_available` era código muerto en
    producción. Spec: docs/superpowers/specs/
    2026-09-14-gobernanza-catalogo-db-design.md (v2) §3.3."""
    config_path = repo_root / "las_manos" / "config.toml"
    with config_path.open("rb") as f:
        config = tomllib.load(f)
    return ValidationContext(
        ops=frozenset(config.get("ops", {}).keys()),
        mutating_capabilities=frozenset(MUTATING_CAPABILITIES),
        catalog=catalog,
        config_paths_allowlist=config_paths_allowlist,
        repo_root=repo_root,
    )
```

Ver el rojo del resolver (Paso 3). Después, reemplazar el bloque `if in_catalog:` de `_resolve_capability_available`:

```python
    if in_catalog:
        # Tanda A v2 (2026-09-14): el catálogo tiene el modo (capability.mode,
        # jax-platform). Hasta hoy esta rama aceptaba cualquier modo "sin
        # contradicción". `ops` usa MUTATING_CAPABILITIES; el catálogo, su
        # columna. Dos fuentes para dos conjuntos de nombres DISJUNTOS
        # (tripwire en jax-platform: test_catalogo_db_en_la_mesa.py).
        real_mode = ctx.catalog.get_capability(name).mode
        if mode != real_mode:
            return Verdict(
                status="FACT_MISMATCH",
                predicate="CAPABILITY_AVAILABLE",
                detail=(
                    f"'{name}' tiene mode real '{real_mode}' en el catálogo de "
                    f"capabilities, el claim afirma '{mode}'."
                ),
            )
        return Verdict(
            status="VALID",
            predicate="CAPABILITY_AVAILABLE",
            detail=f"'{name}' verificado en catálogo de capabilities, mode='{real_mode}'.",
        )
```

Actualizar el docstring del módulo: "Recibe el catálogo de capabilities del llamador (no toca la DB)".

```bash
git -C /home/fruiz/worktrees/jax-gobernanza-catalogo grep -n "load_validation_context(" -- '*.py'
```

Esperado: solo `validator.py` y `tests/test_governance_validator.py`.

- [ ] **Paso 5: verde con la suite del job**

```bash
cd /home/fruiz/worktrees/jax-gobernanza-catalogo && pwd && "$SCRATCH/venv-jax-gov/bin/python" -m pytest tests/test_governance_claims.py tests/test_governance_loaders.py tests/test_governance_validator.py tests/test_governance_vocab_sweep.py tests/test_governance_renderer.py tests/test_governance_grounding.py -q 2>&1 | tail -3
```

Esperado: piso actual − 1 + 4 (hoy `83 passed`).

**Mutación:** volver la rama `in_catalog` al `VALID` incondicional: cae `test_rama_catalogo_con_el_modo_equivocado_es_fact_mismatch`.

- [ ] **Paso 6: piso del job `governance`** (se sube de nuevo en la Tarea 4; cada commit deja su número)

Reemplazar el valor en los dos lugares (`grep -qE "^N passed"` y el mensaje) y agregar al comentario del paso:

```yaml
        # <actual> -> <medido> el 2026-09-14 (tanda A v2): el tripwire del
        # catálogo TOML vacío se reemplaza (-1) por
        # test_load_validation_context_usa_el_catalogo_que_recibe + su CONTROL
        # (+2) y la rama in_catalog verifica el modo (+2). El tripwire se vio en
        # rojo con el arreglo, como estaba previsto; los nuevos, antes.
```

- [ ] **Paso 7: commit**

```bash
git -C /home/fruiz/worktrees/jax-gobernanza-catalogo add policy/governance/validator.py tests/test_governance_validator.py .github/workflows/policy.yml
git -C /home/fruiz/worktrees/jax-gobernanza-catalogo commit -m "$(cat <<'EOF'
fix(gobernanza): el validador recibe el catálogo y verifica su modo

load_validation_context() deja de armar MotorCatalog desde el TOML (vacío
desde el Bloque 3) y recibe el del llamador. La rama in_catalog compara el
modo del claim con capability.mode: FACT_MISMATCH si no coincide. El
tripwire del catálogo vacío se reemplaza por el estado nuevo y su control.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LNXrHCwsZAVfrQj56maFHB
EOF
)"
```

---

### Tarea 4: el snapshot lista las capabilities de la DB (jax, PR-B)

**Archivos:**
- Modificar: `policy/governance/grounding.py`: docstring del módulo (1-18), `SECTION_PREDICATE` (27-30), `build_snapshot` (114-135), `render` (138-148) y el comentario de `accredit.mismatch` (184-190)
- Modificar: `tests/test_governance_grounding.py` (tests nuevos después de `test_render_lists_every_entry_with_its_pointer_and_args` y de `test_empty_ops_is_a_valid_snapshot_not_an_error`)
- Modificar: `docs/superpowers/specs/2026-09-02-reformas-fase2-sp3-grounding-design.md` §3.3 (nota fechada al final de la sección)
- Modificar: `.github/workflows/policy.yml`, job `governance`

**Interfaces:**
- Consume: `MotorCatalog.capabilities()` y `CapabilityEntry.mode` (Tarea 2).
- Produce:
  - `SECTION_PREDICATE = {"capabilities": "CAPABILITY_AVAILABLE", "catalog_capabilities": "CAPABILITY_AVAILABLE"}`.
  - Punteros `/catalog_capabilities/N`, por nombre. `/capabilities/N` no se mueve.
  - `canonical_json = {"capabilities": [...], "catalog_capabilities": [...]}`.
  - La Tarea 7 cita `/catalog_capabilities/N`.

- [ ] **Paso 1: buscar tests que dependan del `canonical_json` literal**

```bash
git -C /home/fruiz/worktrees/jax-gobernanza-catalogo grep -n 'canonical_json ==\|"capabilities": \[' -- tests/
git -C /home/fruiz/worktrees/jax-platform-gobernanza grep -n 'canonical_json ==\|"capabilities": \[' -- backend/tests/
```

Esperado (medido el 2026-09-14):
- En jax, solo `test_sha256_is_over_the_canonical_json`, que compara el JSON consigo mismo.
- En jax-platform, `test_reclassify_provenance_mismatch.py`, que arma su propio JSON de filas viejas: no depende de `build_snapshot`.
- Cualquier otro que compare un literal se actualiza en esta tarea y se declara en el PR.

- [ ] **Paso 2: tests**

En `tests/test_governance_grounding.py`, después de `CTX_A`:

```python
def _ctx_con_catalogo(ops: set[str], catalogo: dict) -> validator.ValidationContext:
    return validator.ValidationContext(
        ops=frozenset(ops),
        mutating_capabilities=frozenset({"write_file"}),
        catalog=MotorCatalog({"capabilities": catalogo}),
        config_paths_allowlist=frozenset(),
        repo_root=REPO_ROOT,
    )


# Contexto C (tanda A v2): las ops de CTX_A + dos capabilities de la DB.
# Por nombre: file_write en /catalog_capabilities/0, generate en /1.
CTX_C = _ctx_con_catalogo(
    {"write_file", "read_file"},
    {"generate": {"mode": "read_only"}, "file_write": {"mode": "mutating"}},
)
```

Después de `test_render_lists_every_entry_with_its_pointer_and_args`:

```python
# --- tanda A v2 (2026-09-14): sección catalog_capabilities --------------------

def test_seccion_catalog_capabilities_ordenada_y_con_el_modo_del_catalogo():
    snap = grounding.build_snapshot(CTX_C)
    e0, e1 = snap.lookup("/catalog_capabilities/0"), snap.lookup("/catalog_capabilities/1")
    assert e0.args == {"name": "file_write", "mode": "mutating"}
    assert e1.args == {"name": "generate", "mode": "read_only"}
    assert e0.predicate == e1.predicate == "CAPABILITY_AVAILABLE"


def test_los_punteros_de_ops_no_se_mueven_al_sumar_el_catalogo():
    """Pasa también ANTES del cambio, a propósito: fija lo que el cambio no
    debe mover (spec v2 §3.4). Se valida por mutación: mezclar las dos
    listas en un solo orden lo pone rojo."""
    def ops(snap):
        return [(e.pointer, e.args) for e in snap.entries if e.pointer.startswith("/capabilities/")]
    assert ops(grounding.build_snapshot(CTX_C)) == ops(grounding.build_snapshot(CTX_A))


def test_canonical_json_trae_las_dos_secciones_aunque_el_catalogo_este_vacio():
    assert json.loads(grounding.build_snapshot(CTX_A).canonical_json) == {
        "capabilities": [{"mode": "read_only", "name": "read_file"},
                         {"mode": "mutating", "name": "write_file"}],
        "catalog_capabilities": [],
    }


def test_una_capability_del_catalogo_citada_bien_es_observado_y_valid():
    snap = grounding.build_snapshot(CTX_C)
    raw = _raw(args={"name": "generate", "mode": "read_only"}, pointer="/catalog_capabilities/1")
    assert grounding.accredit(raw, snap).outcome == "ACCREDITED"
    v = _validate(raw, snap, ctx=CTX_C)
    assert v.status == "VALID", v.detail


def test_control_el_modo_equivocado_contra_el_catalogo_es_fact_not_in_snapshot():
    """CONTROL del anterior: mismo puntero, modo falso -> ninguna entrada
    del snapshot lo respalda. Pasa también antes del cambio (ahí no había
    sección); con la sección, prueba que acreditar mira el modo y no solo el
    nombre."""
    snap = grounding.build_snapshot(CTX_C)
    raw = _raw(args={"name": "generate", "mode": "mutating"}, pointer="/catalog_capabilities/1")
    assert _validate(raw, snap, ctx=CTX_C).status == "FACT_NOT_IN_SNAPSHOT"


def test_render_muestra_las_dos_secciones():
    text = grounding.render(grounding.build_snapshot(CTX_C))
    assert "  capabilities:\n" in text
    assert "  catalog_capabilities:\n" in text
    assert "/catalog_capabilities/1: name=generate, mode=read_only" in text
    assert text.index("  capabilities:") < text.index("  catalog_capabilities:")
```

Después de `test_empty_ops_is_a_valid_snapshot_not_an_error`:

```python
def test_7d_un_catalogo_que_explota_da_GroundingBuildError():
    """P10: leer el catálogo entra en el mismo try que ctx.ops."""
    class CatalogoRoto:
        def capabilities(self):
            raise OSError("catálogo ilegible")

    class Ctx:
        ops = frozenset({"read_file"})
        mutating_capabilities = frozenset()
        catalog = CatalogoRoto()

    with pytest.raises(grounding.GroundingBuildError):
        grounding.build_snapshot(Ctx())
```

- [ ] **Paso 3: ver el rojo**

```bash
cd /home/fruiz/worktrees/jax-gobernanza-catalogo && pwd && "$SCRATCH/venv-jax-gov/bin/python" -m pytest tests/test_governance_grounding.py -q 2>&1 | tail -12
```

Esperado: 5 ROJOS, porque el snapshot todavía no lee el catálogo:
- La sección da `AttributeError: 'NoneType' object has no attribute 'args'`.
- El `canonical_json` da `assert {...} == {..., 'catalog_capabilities': []}`.
- La citación bien hecha da `'FACT_NOT_IN_SNAPSHOT' == 'ACCREDITED'`.
- `render` da `assert '  catalog_capabilities:\n' in ...`.
- El catálogo roto da `DID NOT RAISE`.

Los otros 2 (punteros de ops y el CONTROL del modo) pasan ya, como declaran.

- [ ] **Paso 4: implementar en `grounding.py`**

```python
# Sección del snapshot -> predicado que acredita. Crece SOLO cuando un
# predicado gana resolver (spec SP3 §3): no agregar entradas acá sin
# resolver en validator._RESOLVERS.
#   capabilities          -> `ops` de las_manos/config.toml (rama in_ops)
#   catalog_capabilities  -> capability de la DB (rama in_catalog, que desde
#                            la tanda A v2, 2026-09-14, verifica nombre Y modo)
# Dos secciones y no una lista mezclada: los punteros /capabilities/N de las
# ops no se mueven cuando la DB gana una capability (spec tanda A v2 §3.4).
SECTION_PREDICATE: dict[str, str] = {
    "capabilities": "CAPABILITY_AVAILABLE",
    "catalog_capabilities": "CAPABILITY_AVAILABLE",
}
```

```python
def build_snapshot(ctx) -> Snapshot:
    """Snapshot desde ctx.ops + ctx.mutating_capabilities (sección
    `capabilities`) y ctx.catalog (sección `catalog_capabilities`, con el
    modo de capability.mode). Cada sección ordenada por name (spec SP3
    §5.2): mismo contenido => mismo hash y mismos punteros, en cualquier
    orden de archivo o de filas."""
    try:
        mutating = ctx.mutating_capabilities
        data = {
            "capabilities": [
                normalize_args({"name": name, "mode": "mutating" if name in mutating else "read_only"})
                for name in sorted(ctx.ops)
            ],
            "catalog_capabilities": [
                normalize_args({"name": e.name, "mode": e.mode})
                for e in sorted(ctx.catalog.capabilities(), key=lambda e: e.name)
            ],
        }
    except Exception as e:
        raise GroundingBuildError(f"ValidationContext inutilizable: {type(e).__name__}: {e}") from e

    canonical = _canonical(data)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    entries = tuple(
        SnapshotEntry(pointer=f"/{section}/{i}", predicate=SECTION_PREDICATE[section], args=c)
        for section, lista in data.items()
        for i, c in enumerate(lista)
    )
    return Snapshot(entries=entries, canonical_json=canonical, sha256=digest)


def render(snapshot: Snapshot) -> str:
    """Bloque para el system prompt. NO incluye el hash (spec §5.1): el
    modelo solo cita la línea; provenance_ref lo escribe el servidor. Una
    cabecera por sección, en el orden de SECTION_PREDICATE, aunque esté
    vacía: cero entradas es una observación."""
    lines = [
        "HECHOS VERIFICADOS — leídos del sistema por el servidor. "
        "Para afirmar uno, poné su evidence_pointer en el claim.",
    ]
    for section in SECTION_PREDICATE:
        lines.append(f"  {section}:")
        prefix = f"/{section}/"
        for e in snapshot.entries:
            if e.pointer.startswith(prefix):
                lines.append(f"    {e.pointer}: " + ", ".join(f"{k}={v}" for k, v in e.args.items()))
    return "\n".join(lines)
```

En el docstring del módulo, reemplazar el párrafo del invariante por:

```text
Invariante (spec §3): todo hecho inyectado tiene quién lo re-resuelva. Por
eso el snapshot se genera desde las MISMAS fuentes que consulta
_resolve_capability_available: ctx.ops (rama in_ops) y, desde la tanda A v2
(2026-09-14), ctx.catalog con su modo (rama in_catalog). No desde una
lista curada.
```

En el comentario de `mismatch()` de `accredit`, reemplazar "Hoy build_snapshot() no puede producir dos entradas con args idénticos -- itera sorted(ctx.ops), ops es un frozenset (nombres únicos), y el nombre es parte de los args de cada entrada" por:

```text
Hoy build_snapshot() no puede producir dos entradas con args idénticos:
cada sección tiene nombres únicos (ops es un frozenset, capability.key es
PK), y entre secciones un mismo nombre es SOURCE_CONFLICT, vigilado por el
tripwire ops∩DB de jax-platform (test_catalogo_db_en_la_mesa.py).
```

En el spec de SP3, al final de §3.3:

```markdown
> **Nota 2026-09-14 (tanda A v2, `docs/superpowers/specs/2026-09-14-gobernanza-catalogo-db-design.md`):**
> la rama `in_catalog` dejó de ser código muerto. El validador recibe el catálogo de la DB y verifica
> nombre **y** modo (`capability.mode`), así que `build_snapshot` ahora sí lee `ctx.catalog`: sección
> `catalog_capabilities` (`/catalog_capabilities/N`), sin mover los punteros `/capabilities/N`. El
> invariante de §3 se sigue cumpliendo: cada línea inyectada la re-resuelve una rama viva. El tripwire
> citado arriba se reemplazó por `test_load_validation_context_usa_el_catalogo_que_recibe`.
```

- [ ] **Paso 5: verde**

Correr el mismo comando de la suite del job `governance` que en la Tarea 3, Paso 5. Esperado: el piso de la Tarea 3 + 7 (hoy `90 passed`).

**Mutaciones:**
- (a) Mezclar las secciones: sumar el catálogo a la lista `capabilities` y dejar `catalog_capabilities` vacía. Caen `test_los_punteros_de_ops_no_se_mueven_al_sumar_el_catalogo` y la sección.
- (b) Sacar `"catalog_capabilities"` de `SECTION_PREDICATE`. Cae con `KeyError` todo lo que construye el snapshot, que es ruidoso y es lo buscado.
- (c) Construir la sección nueva fuera del `try`. Cae `test_7d_un_catalogo_que_explota_da_GroundingBuildError`.

- [ ] **Paso 6: piso de `governance`**: del valor que dejó la Tarea 3 al medido. Comentario: `# <antes> -> <medido> el 2026-09-14 (tanda A v2): sección catalog_capabilities en el snapshot (+7: sección, punteros de ops estables, canonical con dos secciones, citación VALID + su CONTROL, render, P10). Rojos vistos; los 2 que pasan antes son declarados y validados por mutación.`

- [ ] **Paso 7: commit**

```bash
git -C /home/fruiz/worktrees/jax-gobernanza-catalogo add policy/governance/grounding.py tests/test_governance_grounding.py docs/superpowers/specs/2026-09-02-reformas-fase2-sp3-grounding-design.md .github/workflows/policy.yml
git -C /home/fruiz/worktrees/jax-gobernanza-catalogo commit -m "$(cat <<'EOF'
feat(grounding): el snapshot lista las capabilities de la DB

Sección nueva catalog_capabilities (/catalog_capabilities/N, por nombre,
con capability.mode); los punteros /capabilities/N de las ops no se
mueven. SECTION_PREDICATE la mapea a CAPABILITY_AVAILABLE, que la
re-resuelve la rama in_catalog. render() por sección. Nota en SP3 §3.3.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LNXrHCwsZAVfrQj56maFHB
EOF
)"
```

---

### Tarea 5: Jacobs acepta el rol `plataforma` (jax, PR-B)

**Archivos:**
- Modificar: `jacobs/models.py:51` (`VALID_INVOKERS`)
- Modificar: `jacobs/policy.py:12` (import) y `:81-88` (`validate_resume`)
- Modificar: `jacobs/routes.py:20-27` (import), `:100-104` (`plan_only`) y los docstrings de `resume_pipeline` (`:252`) y `approve_step` (`:302-306`)
- Crear: `tests/test_jacobs_invoked_by_rol.py`
- Modificar: `.github/workflows/policy.yml`, job `tests-puros`

**Interfaces:**
- Produce: `jacobs.models.INVOKER_PLATAFORMA = "plataforma"` y `jacobs.models.VALID_INVOKERS = frozenset({"plataforma", "jax_local", "ada"})`. El literal `"plataforma"` es el contrato con jax-platform (Tarea 8). `create` lo valida `PipelineCreateRequest` contra `VALID_INVOKERS`.
- `policy.validate_resume(invoked_by: str) -> PolicyResult` da `ok=True` solo con `"plataforma"`.

- [ ] **Paso 1: escribir el test (rojo)**

Crear `tests/test_jacobs_invoked_by_rol.py`:

```python
"""
Jacobs — `invoked_by` es un ROL, no el nombre de una persona (tanda A, 2026-09-14).

Hasta hoy `VALID_INVOKERS` era {"Fernando", "jax_local", "ada"} y
`validate_resume` exigía "Fernando": un nombre de persona usado como campo de
AUTORIZACIÓN (Principio IX). Decisión de Fernando (spec 2026-09-14, §2):
`invoked_by` pasa a ser el rol "plataforma" -- "pedido de jax-platform en
nombre de un usuario autenticado"; la identidad viaja en user_id/tenant_id.

Nada sale a la red ni a la DB: `store.pipeline_get` y `_build_plan_or_reject`
se parchean en TODOS los casos, también en los que esperan un rechazo -- con el
código viejo "Fernando" pasaba el control y llegaba a la DB o al planificador.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

# Forzado, no setdefault: mismo guard que jacobs/_pipeline_identity_test.py.
os.environ["JAX_DB_NAME"] = "jax_memory_test"

import pytest  # noqa: E402
from fastapi import BackgroundTasks, HTTPException  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from jacobs import policy, routes  # noqa: E402
from jacobs.models import VALID_INVOKERS, PipelineCreateRequest  # noqa: E402

_NO_PLANIFICAR = AsyncMock(side_effect=AssertionError("el test no debe llegar a planificar"))


def test_los_invocadores_validos_son_plataforma_jax_local_y_ada():
    assert VALID_INVOKERS == frozenset({"plataforma", "jax_local", "ada"})


def test_crear_rechaza_fernando_como_invocador():
    with patch.object(policy, "check_kill_switch", return_value=False):
        r = policy.validate_create("Fernando", "supervised", 3, 0)
    assert not r.ok
    assert "Fernando" in r.reason


def test_crear_acepta_plataforma():
    with patch.object(policy, "check_kill_switch", return_value=False):
        r = policy.validate_create("plataforma", "supervised", 3, 0)
    assert r.ok, r.reason


def test_el_request_de_creacion_rechaza_fernando_y_acepta_plataforma():
    with pytest.raises(ValidationError):
        PipelineCreateRequest(name="t", objective="o", invoked_by="Fernando", mode="supervised")
    req = PipelineCreateRequest(name="t", objective="o", invoked_by="plataforma", mode="supervised")
    assert req.invoked_by == "plataforma"


def test_validate_resume_solo_acepta_plataforma():
    assert policy.validate_resume("plataforma").ok
    for otro in ("Fernando", "jax_local", "ada", ""):
        assert not policy.validate_resume(otro).ok, otro


def test_plan_rechaza_fernando_y_deja_pasar_plataforma():
    def plan(invoked_by):
        return routes.plan_only(routes.PlanRequest(
            name="t", objective="o", invoked_by=invoked_by, mode="dry_run"))

    with patch.object(routes, "_build_plan_or_reject", _NO_PLANIFICAR), \
         patch.object(routes, "check_kill_switch", return_value=False), \
         pytest.raises(HTTPException) as rechazo:
        asyncio.run(plan("Fernando"))
    assert rechazo.value.status_code == 403

    # "plataforma" pasa el control de invocador; lo frena el kill switch
    # (forzado a propósito) para que el test no llegue a planificar.
    with patch.object(routes, "_build_plan_or_reject", _NO_PLANIFICAR), \
         patch.object(routes, "check_kill_switch", return_value=True), \
         pytest.raises(HTTPException) as frenado:
        asyncio.run(plan("plataforma"))
    assert frenado.value.status_code == 423


def test_reanudar_rechaza_fernando_y_deja_pasar_plataforma():
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=None)):
        with pytest.raises(HTTPException) as rechazo:
            asyncio.run(routes.resume_pipeline(
                "p-inexistente", routes.ResumeRequest(invoked_by="Fernando"), BackgroundTasks()))
        assert rechazo.value.status_code == 403

        # "plataforma" pasa la política y llega a buscar el pipeline (404).
        with pytest.raises(HTTPException) as no_existe:
            asyncio.run(routes.resume_pipeline(
                "p-inexistente", routes.ResumeRequest(invoked_by="plataforma"), BackgroundTasks()))
        assert no_existe.value.status_code == 404


def test_aprobar_paso_rechaza_fernando_y_deja_pasar_plataforma():
    with patch.object(routes.store, "pipeline_get", AsyncMock(return_value=None)):
        with pytest.raises(HTTPException) as rechazo:
            asyncio.run(routes.approve_step(
                "p-inexistente", routes.ApproveStepRequest(invoked_by="Fernando"), BackgroundTasks()))
        assert rechazo.value.status_code == 403

        with pytest.raises(HTTPException) as no_existe:
            asyncio.run(routes.approve_step(
                "p-inexistente", routes.ApproveStepRequest(invoked_by="plataforma"), BackgroundTasks()))
        assert no_existe.value.status_code == 404
```

Antes de correrlo, confirmar contra el código las firmas que el test supone: `validate_create(invoked_by, mode, ...)` en `policy.py`, `PlanRequest`/`ResumeRequest`/`ApproveStepRequest` en `routes.py`, y `check_kill_switch` importado en `routes`. Esto se hace con `grep -n "def validate_create\|class PlanRequest\|class ResumeRequest\|class ApproveStepRequest\|check_kill_switch" jacobs/policy.py jacobs/routes.py jacobs/models.py`. Si una firma difiere, se ajusta el test a la firma real antes de ver el rojo.

- [ ] **Paso 2: ver el rojo**

```bash
cd /home/fruiz/worktrees/jax-gobernanza-catalogo && pwd && PYTHONPATH=.:las_manos "$SCRATCH/venv-jax-puros/bin/python" -m pytest tests/test_jacobs_invoked_by_rol.py -v 2>&1 | tail -15
```

Esperado: los 8 en ROJO por la razón declarada:
- El set de invocadores no coincide.
- `validate_create("Fernando")` da `ok=True`.
- `"plataforma"` es rechazado, y en `plan_only` da 403 donde se esperaba 423.
- El request da `DID NOT RAISE ValidationError`.
- Reanudar y aprobar con `"Fernando"` dan 404 donde se esperaba 403.

**Si alguno falla en la colección** (`ModuleNotFoundError`), es una dependencia que falta en el `pip install` del job: se agrega en el workflow con un comentario "medido en venv limpio" y se vuelve a ver el rojo correcto.

- [ ] **Paso 3: implementar**

`jacobs/models.py`, línea 51:

```python
# `invoked_by` es un ROL de quien pide, no el nombre de una persona (tanda A,
# 2026-09-14, decisión de Fernando). "plataforma" = pedido de jax-platform en
# nombre de un usuario autenticado; QUIÉN es viaja en user_id/tenant_id. Las
# filas viejas de jacobs_pipelines con "Fernando" quedan como están: historia.
INVOKER_PLATAFORMA = "plataforma"
VALID_INVOKERS = frozenset({INVOKER_PLATAFORMA, "jax_local", "ada"})
```

`jacobs/policy.py`: import `from jacobs.models import INVOKER_PLATAFORMA, VALID_INVOKERS` y:

```python
def validate_resume(invoked_by: str) -> PolicyResult:
    """Solo la plataforma (jax-platform, en nombre de un usuario autenticado y
    dueño del pipeline -- eso lo verifica jax-platform antes de reenviar)
    puede reanudar un pipeline interrumpido o aprobar un paso."""
    if invoked_by != INVOKER_PLATAFORMA:
        return PolicyResult(
            ok=False,
            reason=(
                f"invoked_by '{invoked_by}' no puede reanudar ni aprobar: "
                f"solo '{INVOKER_PLATAFORMA}'"
            ),
        )
    return PolicyResult(ok=True, reason="OK")
```

`jacobs/routes.py`:
- Agregar `VALID_INVOKERS` al import de `jacobs.models`.
- En `plan_only`, `if req.invoked_by not in VALID_INVOKERS:` en lugar del literal.
- Docstrings: en `resume_pipeline`, `"""Reanuda un pipeline interrumpido. Solo el rol 'plataforma' puede hacerlo."""`. En `approve_step`, la línea `Solo Fernando puede aprobar.` pasa a `Solo el rol 'plataforma' puede aprobar.`

```bash
git -C /home/fruiz/worktrees/jax-gobernanza-catalogo grep -n '"Fernando"' -- jacobs/ las_manos/ tests/
```

Esperado: solo fixtures que fabrican `Pipeline(invoked_by="Fernando")` sin pasar por la validación (`jacobs/_pipeline_identity_test.py`, `jacobs/_direct_usage_test.py`). El spec §3.6 dice que esas no cambian. Cualquier otro sitio que valide contra `"Fernando"` se corrige en este commit.

- [ ] **Paso 4: verde con la lista completa del job**: el "Piso exacto" de `tests-puros` del workflow ACTUAL (que ya incluye `test_motor_catalog_mode.py`), con `tests/test_jacobs_invoked_by_rol.py` al final. Esperado: el piso de la Tarea 2 + 8 (hoy `156 passed`).

**Mutación:** volver `validate_resume` a exigir `"Fernando"`. Caen `test_validate_resume_solo_acepta_plataforma`, reanudar y aprobar.

- [ ] **Paso 5: workflow** — `tests/test_jacobs_invoked_by_rol.py` en las DOS listas de `tests-puros`. Piso al medido, con este comentario:
  `# <antes> -> <medido> el 2026-09-14 (tanda A): test_jacobs_invoked_by_rol.py (+8) -- invoked_by es el rol "plataforma" y no "Fernando": al crear (policy y request), planificar, reanudar y aprobar. Vistos en rojo antes del arreglo.`

- [ ] **Paso 6: commit**

```bash
git -C /home/fruiz/worktrees/jax-gobernanza-catalogo add jacobs/models.py jacobs/policy.py jacobs/routes.py tests/test_jacobs_invoked_by_rol.py .github/workflows/policy.yml
git -C /home/fruiz/worktrees/jax-gobernanza-catalogo commit -m "$(cat <<'EOF'
fix(jacobs): invoked_by es el rol "plataforma", no un nombre de persona

VALID_INVOKERS = {plataforma, jax_local, ada}; validate_resume exige
"plataforma"; plan_only usa VALID_INVOKERS en vez de un literal duplicado.
tests-puros +8.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LNXrHCwsZAVfrQj56maFHB
EOF
)"
```

---

### Tarea 6: `governance_context` async, con el catálogo de la DB (jax-platform, PR-C)

**Archivos:**
- Modificar: `backend/governance_context.py` (desde `from __future__` hasta el final; el docstring se conserva y se amplía)
- Modificar: `backend/api/chat.py:629-647` y `:1115`
- Modificar: `backend/shadow_validation.py:315`
- Crear: `backend/tests/test_governance_context_catalogo.py`
- Modificar: `backend/tests/test_grounding_config_revalidation.py`, `test_shadow_origin.py` (`:128`, `:177`), `test_shadow_validation.py` (`:50`), `test_shadow_validation_grounding.py` (`:89`)
- Modificar: `.github/workflows/policy.yml` (`PISO_PASSED`, `JAX_CI_MIN_PASSED`)

**Interfaces:**
- Consume:
  - `validator.load_validation_context(repo_root, allowlist, catalog)` (Tarea 3).
  - `MotorCatalog.from_db()` con `mode` (Tarea 2).
  - `grounding.build_snapshot` con la sección nueva (Tarea 4).
  - `facet_resolver._seal_mtime() -> float | None` y `facet_resolver._tocar_sello()` (jax-platform, sin cambios).
- Produce:
  - `async def governance_context.validation_context() -> tuple[ValidationContext, dict, dict]`.
  - `governance_context.validation_context.cache_clear() -> None`, que vacía la caché y crea un lock nuevo.
  - `async def api.chat._build_snapshot_or_raise()` y `async def api.chat._build_grounding()`.

- [ ] **Paso 1: rama de PR-C**

```bash
git -C /home/fruiz/worktrees/jax-platform-gobernanza switch -c feat/gobernanza-catalogo-db
git -C /home/fruiz/worktrees/jax-platform-gobernanza log --oneline -2   # arriba: el commit de la Tarea 1
```

- [ ] **Paso 2: medir el "antes"** (lo usa la Tarea 9)

Crear `$SCRATCH/medir_validation_context.py`. No se commitea: es una herramienta de medición.

```python
"""Latencia en proceso de governance_context.validation_context(), caché fría
(recarga completa) y caliente (hit), y tamaño del snapshot que produce.
Sirve para la versión sync (master) y la async (rama). Solo lectura: from_db()
hace SELECT; el sello solo se statea."""
import asyncio
import inspect
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import governance_context as gc  # noqa: E402
import grounding as g  # noqa: E402

N = int(os.environ.get("N", "200"))


async def llamar():
    r = gc.validation_context()
    if inspect.isawaitable(r):
        r = await r
    return r


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


async def main():
    frio, caliente = [], []
    for _ in range(N):
        gc.validation_context.cache_clear()
        t = time.perf_counter()
        await llamar()
        frio.append((time.perf_counter() - t) * 1e3)
    for _ in range(N):
        t = time.perf_counter()
        await llamar()
        caliente.append((time.perf_counter() - t) * 1e6)
    ctx, _, _ = await llamar()
    snap = g.build_snapshot(ctx)
    texto = g.render(snap)
    print(f"N={N} frio_ms p50={pct(frio, 50):.3f} p95={pct(frio, 95):.3f} max={max(frio):.3f}")
    print(f"N={N} caliente_us p50={pct(caliente, 50):.2f} p95={pct(caliente, 95):.2f} max={max(caliente):.2f}")
    print(f"snapshot entradas={len(snap.entries)} render_chars={len(texto)} "
          f"canonical_chars={len(snap.canonical_json)} tokens_estimados(chars/4)={len(texto) // 4}")


asyncio.run(main())
```

Correrlo con el `governance_context` todavía sin cambios y el validador de producción (`/home/fruiz/jax`, dos argumentos), que solo se importa y no escribe bytecode:

```bash
cd /home/fruiz/worktrees/jax-platform-gobernanza/backend && pwd && set -a && . /etc/jax/.env && set +a && \
  PYTHONDONTWRITEBYTECODE=1 JAX_REPO_PATH=/home/fruiz/jax \
  /home/fruiz/jax-platform/backend/.venv/bin/python "$SCRATCH/medir_validation_context.py" | tee "$SCRATCH/latencia-antes.txt"
```

Esperado en la línea del snapshot: 11 entradas y 715 caracteres de `render` (medido el 2026-09-14). Si da otro número, se anota el medido.

- [ ] **Paso 3: escribir el test nuevo (rojo)**

Crear `backend/tests/test_governance_context_catalogo.py`:

```python
"""
governance_context con el catálogo de la DB (tanda A v2, spec 2026-09-14 §3.5).

Hasta hoy el contexto de gobernanza armaba el catálogo de capabilities desde
`las_manos/config.toml`, vacío desde el Bloque 3. Ahora `validation_context()`
es async, carga `await MotorCatalog.from_db()` y cachea con una clave que suma
el mtime del sello de facet_resolver (lo estampan las migraciones y el admin de
motores y capabilities) a los mtimes de los tres archivos de config.

Salvo el último, estos tests son PUROS: `MotorCatalog.from_db` se parchea, así
que corren también en el job sin DB. El sello es el archivo aislado por
función que pone conftest (`_sello_de_facets_aislado`), nunca el real.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

import claims as governance_claims
import facet_resolver
import governance_context
import grounding as governance_grounding
import validator as governance_validator

MotorCatalog = governance_validator.MotorCatalog


def _catalogo(**modos):
    return MotorCatalog({"capabilities": {n: {"allowed_motors": ["kimi"], "mode": m} for n, m in modos.items()}})


@pytest.fixture
def from_db(monkeypatch):
    """from_db falso que devuelve un catálogo NUEVO en cada llamada (así una
    recarga se distingue de un hit por identidad) y cuenta las llamadas."""
    fake = AsyncMock(side_effect=lambda: _catalogo(generate="read_only", file_write="mutating"))
    monkeypatch.setattr(MotorCatalog, "from_db", fake)
    governance_context.validation_context.cache_clear()
    yield fake
    governance_context.validation_context.cache_clear()


def _raw(nombre, modo, ptr):
    return {"predicate": "CAPABILITY_AVAILABLE", "args": {"name": nombre, "mode": modo}, "evidence_pointer": ptr}


def _validar(raw, snap, ctx, predicates):
    acc = governance_grounding.accredit(raw, snap)
    claim = governance_claims.Claim(
        predicate=raw["predicate"], args=governance_grounding.normalize_args(raw["args"]),
        authority=acc.authority, provenance_ref=acc.provenance_ref,
        evidence_pointer=raw["evidence_pointer"], scope="mesa_web")
    return acc, governance_validator.validate(claim, predicates, ctx, accreditation=acc)


def _puntero(snap, nombre):
    return next(e.pointer for e in snap.entries
                if e.pointer.startswith("/catalog_capabilities/") and e.args["name"] == nombre)


def test_el_contexto_trae_el_catalogo_de_la_db(from_db):
    ctx, _, _ = asyncio.run(governance_context.validation_context())
    assert ctx.catalog.get_capability("generate").mode == "read_only"
    assert from_db.await_count == 1


def test_sin_cambios_no_recarga_y_devuelve_el_mismo_objeto(from_db):
    async def dos_turnos():
        return (await governance_context.validation_context(),
                await governance_context.validation_context())

    primero, segundo = asyncio.run(dos_turnos())
    assert segundo is primero
    assert from_db.await_count == 1


def test_tocar_el_sello_fuerza_la_recarga(from_db):
    async def correr():
        antes = await governance_context.validation_context()
        assert facet_resolver._tocar_sello(), "no se pudo estampar el sello aislado del test"
        return antes, await governance_context.validation_context()

    antes, despues = asyncio.run(correr())
    assert despues is not antes
    assert from_db.await_count == 2


def test_db_caida_en_frio_falla_visible(monkeypatch):
    monkeypatch.setattr(
        MotorCatalog, "from_db", AsyncMock(side_effect=ConnectionRefusedError("DB caída, simulada")))
    governance_context.validation_context.cache_clear()
    try:
        with pytest.raises(ConnectionRefusedError):
            asyncio.run(governance_context.validation_context())
    finally:
        governance_context.validation_context.cache_clear()


def test_db_caida_en_caliente_no_sirve_el_catalogo_viejo(from_db):
    """Con el sello nuevo y la DB caída, servir el contexto cacheado sería un
    gate fail-open (podría seguir dando VALID a una capability revocada)."""
    async def correr():
        await governance_context.validation_context()
        from_db.side_effect = ConnectionRefusedError("DB caída, simulada")
        assert facet_resolver._tocar_sello()
        with pytest.raises(ConnectionRefusedError):
            await governance_context.validation_context()

    asyncio.run(correr())


def test_n_turnos_a_la_vez_disparan_una_sola_recarga(monkeypatch):
    async def lento():
        await asyncio.sleep(0.05)
        return _catalogo(generate="read_only")

    fake = AsyncMock(side_effect=lento)
    monkeypatch.setattr(MotorCatalog, "from_db", fake)
    governance_context.validation_context.cache_clear()

    async def veinte_turnos():
        return await asyncio.gather(*(governance_context.validation_context() for _ in range(20)))

    try:
        resultados = asyncio.run(veinte_turnos())
    finally:
        governance_context.validation_context.cache_clear()
    assert fake.await_count == 1
    assert all(r is resultados[0] for r in resultados)


def test_la_mesa_acredita_y_valida_una_capability_de_la_db(from_db):
    """El camino de la Mesa, sin nada armado a mano: el snapshot del
    contexto real lista `generate`, la citación se acredita OBSERVADO y el
    resolver la confirma con nombre y modo."""
    ctx, predicates, _ = asyncio.run(governance_context.validation_context())
    snap = governance_grounding.build_snapshot(ctx)
    acc, v = _validar(_raw("generate", "read_only", _puntero(snap, "generate")), snap, ctx, predicates)
    assert (acc.outcome, acc.authority) == ("ACCREDITED", "OBSERVADO")
    assert v.status == "VALID", v.detail


def test_control_una_capability_que_no_esta_en_la_db_no_tiene_linea_que_citar(from_db):
    """CONTROL del anterior: mismo camino, nombre ausente de la DB y de
    `ops` -> FACT_NOT_IN_SNAPSHOT. El VALID de arriba sale del catálogo."""
    ctx, predicates, _ = asyncio.run(governance_context.validation_context())
    snap = governance_grounding.build_snapshot(ctx)
    _, v = _validar(_raw("totalmente_inventado_xyz", "read_only", "/catalog_capabilities/0"), snap, ctx, predicates)
    assert v.status == "FACT_NOT_IN_SNAPSHOT"


def test_si_el_catalogo_cambia_entre_el_snapshot_y_la_validacion_el_resolver_da_fact_mismatch(monkeypatch):
    """El FACT_MISMATCH del resolver en la Mesa: la faceta citó bien el
    snapshot de SU turno, pero antes de la validación en sombra alguien
    cambió el modo en la DB y estampó el sello. El resolver ve el catálogo
    nuevo y lo dice (spec v2 §3.4, último punto)."""
    modos = iter(["read_only", "mutating"])
    monkeypatch.setattr(MotorCatalog, "from_db", AsyncMock(side_effect=lambda: _catalogo(generate=next(modos))))
    governance_context.validation_context.cache_clear()

    async def correr():
        ctx_turno, _, _ = await governance_context.validation_context()
        snap = governance_grounding.build_snapshot(ctx_turno)
        assert facet_resolver._tocar_sello()
        ctx_validacion, predicates, _ = await governance_context.validation_context()
        return snap, ctx_validacion, predicates

    try:
        snap, ctx, predicates = asyncio.run(correr())
    finally:
        governance_context.validation_context.cache_clear()
    acc, v = _validar(_raw("generate", "read_only", _puntero(snap, "generate")), snap, ctx, predicates)
    assert acc.outcome == "ACCREDITED"
    assert v.status == "FACT_MISMATCH"
    assert "mutating" in v.detail


def test_build_grounding_con_la_db_caida_da_SnapshotError(monkeypatch):
    import api.chat as chat

    monkeypatch.setattr(
        MotorCatalog, "from_db", AsyncMock(side_effect=ConnectionRefusedError("DB caída, simulada")))
    governance_context.validation_context.cache_clear()
    try:
        resultado = asyncio.run(chat._build_grounding())
    finally:
        governance_context.validation_context.cache_clear()
    assert isinstance(resultado, governance_grounding.SnapshotError)
    assert "ConnectionRefusedError" in resultado.reason


def test_con_la_db_real_el_contexto_trae_las_capabilities_sembradas_con_su_modo(client):
    """Sin parches: from_db contra jax_memory_test migrada por el fixture
    `client` (capability.mode incluida)."""
    governance_context.validation_context.cache_clear()
    try:
        ctx, _, _ = client.portal.call(governance_context.validation_context)
    finally:
        governance_context.validation_context.cache_clear()
    assert ctx.catalog.get_capability("generate").mode == "read_only"
    assert ctx.catalog.get_capability("file_write").mode == "mutating"
```

- [ ] **Paso 4: ver el rojo**

```bash
cd /home/fruiz/worktrees/jax-platform-gobernanza/backend && pwd && JAX_REPO_PATH=/home/fruiz/worktrees/jax-gobernanza-catalogo \
  /home/fruiz/jax-platform/backend/.venv/bin/python -m pytest tests/test_governance_context_catalogo.py -v 2>&1 | tail -20
```

Esperado: los 11 en ROJO. Con el validador de la Tarea 3 y el `governance_context` viejo, la reconstrucción falla con `TypeError: load_validation_context() missing 1 required positional argument: 'catalog'`: el contexto todavía no carga el catálogo. En los tests de la DB caída, `pytest.raises(ConnectionRefusedError)` ve un `TypeError`, y `asyncio.run(...)` sobre una tupla da `ValueError: a coroutine was expected`. Un rojo por otra razón (fixture, import) se corrige primero.

- [ ] **Paso 5: implementar `governance_context.py`**

Conservar el docstring actual y agregarle esta sección antes del `"""` de cierre:

```text
Catálogo de la DB (tanda A v2, 2026-09-14). Desde el Bloque 3 las
capabilities viven en la DB; hasta hoy este contexto las armaba del TOML
vacío. Ahora:

  * `validation_context()` es ASYNC: el catálogo sale de
    `await MotorCatalog.from_db()` (aiomysql), con capability.mode. Los
    YAML/TOML se leen en `asyncio.to_thread`: nada bloqueante en el turno.
  * El MISMO contexto alimenta el snapshot del prompt (api/chat.py) y la
    validación en sombra (shadow_validation.py): lo que se inyecta es lo que
    se verifica.
  * CLAVE DE CACHÉ = mtimes de los tres archivos + mtime del sello de
    facet_resolver (`_seal_mtime()`). INVALIDACIÓN DECLARADA: el sello, que
    estampan tras commitear las migraciones y el admin de motores/
    capabilities (y los rebinds de facets). Si la clave cambió, recarga; si
    no, el mismo objeto. La clave se toma ANTES de consultar: un sello
    estampado con la consulta en vuelo deja la clave guardada vieja y el
    próximo turno recarga (mismo criterio que LAS MANOS,
    motor_registry/routes.py::_load_catalog).
  * Sello ausente o ilegible: `_seal_mtime()` da None = "sin señal", nunca
    "invalidar" (contrato de facet_resolver). La clave queda estable y los
    cambios del catálogo entran al reiniciar. En hall9000 el sello existe
    (/srv/jax-data/facet-cache-seal, verificado 2026-09-14); el despliegue lo
    vuelve a verificar.
  * UNA recarga a la vez (`asyncio.Lock` con doble chequeo).
  * FALLA VISIBLE: si from_db() lanza, la excepción sube y la caché queda
    como estaba (no se sirve: la clave ya no coincide). En el chat,
    `_build_grounding` la convierte en SnapshotError
    (grounding_snapshot_sha256='ERROR'); la validación en sombra no escribe
    veredictos. Servir un catálogo vacío o viejo repetiría el falso negativo
    en silencio, o daría VALID a una capability revocada (P10).
  * Costo: un stat más por turno; la recarga solo cuando cambió algo.
    Latencia medida antes/después: DEUDA.md de jax.
  * Si la DB cuelga en vez de rechazar, la recarga espera lo que espere
    aiomysql.connect -- igual que el resto del turno, que ya depende de la
    misma DB por el pool.
```

Código (reemplaza desde `from __future__ import annotations` hasta el final):

```python
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

JAX_REPO = Path(os.getenv("JAX_REPO_PATH", os.path.expanduser("~/jax")))
if str(JAX_REPO) not in sys.path:
    sys.path.insert(0, str(JAX_REPO))
if str(JAX_REPO / "policy" / "governance") not in sys.path:
    sys.path.insert(0, str(JAX_REPO / "policy" / "governance"))

import facet_resolver  # noqa: E402  (su sello invalida también este caché)
import loaders as governance_loaders  # noqa: E402
import validator as governance_validator  # noqa: E402

_FileStamp = tuple[tuple[str, int | None], ...]
_Stamp = tuple[_FileStamp, float | None]


def _source_stamp() -> _FileStamp:
    """(ruta, mtime_ns) de los tres archivos fuente. Solo stat(): no abre nada.

    Un archivo que no se puede statear entra como None -- también invalida, y
    la reconstrucción falla ruidosamente, que es el comportamiento buscado
    (P10: ningún gate falla abierto). JAX_REPO se lee en cada llamada a
    propósito: los tests lo reapuntan.
    """
    rutas = (
        JAX_REPO / "las_manos" / "config.toml",
        governance_loaders.PREDICATES_FILE,
        governance_loaders.VOCABULARY_FILE,
    )
    marca: list[tuple[str, int | None]] = []
    for ruta in rutas:
        try:
            marca.append((str(ruta), os.stat(ruta).st_mtime_ns))
        except OSError:
            marca.append((str(ruta), None))
    return tuple(marca)


def _stamp() -> _Stamp:
    return (_source_stamp(), facet_resolver._seal_mtime())


_cache: tuple[_Stamp, tuple] | None = None
_lock = asyncio.Lock()


def _build_static(catalog):
    """La parte de disco (YAML/TOML): corre en un hilo, no en el loop."""
    vocabulary = governance_loaders.load_vocabulary()
    ctx = governance_validator.load_validation_context(JAX_REPO, vocabulary.config_paths, catalog)
    predicates = governance_loaders.load_predicates()
    return ctx, predicates, vocabulary.term_categories


async def _build():
    catalog = await governance_validator.MotorCatalog.from_db()
    return await asyncio.to_thread(_build_static, catalog)


async def validation_context():
    """(ValidationContext, predicates, term_categories) de gobernanza."""
    global _cache
    stamp = _stamp()
    cached = _cache
    if cached is not None and cached[0] == stamp:
        return cached[1]
    async with _lock:
        stamp = _stamp()  # otro turno pudo recargar mientras esperábamos
        cached = _cache
        if cached is not None and cached[0] == stamp:
            return cached[1]
        value = await _build()
        _cache = (stamp, value)
        return value


def _cache_clear() -> None:
    """Vacía la caché y crea un lock nuevo: un test que usó el lock en su
    propio loop no se lo deja atado al siguiente."""
    global _cache, _lock
    _cache = None
    _lock = asyncio.Lock()


# Compatibilidad: quien tenía `validation_context.cache_clear()` lo sigue
# teniendo (lo usan los tests para forzar una reconstrucción).
validation_context.cache_clear = _cache_clear
```

Antes de dar por buena la clave, verificar que `governance_loaders.PREDICATES_FILE` y `VOCABULARY_FILE` existen con esos nombres en `policy/governance/loaders.py` del worktree de jax. El `_source_stamp` actual de master ya los usa: se copia de ahí, no de este plan, si difieren.

- [ ] **Paso 6: llamadores con `await`**

`backend/api/chat.py`, líneas 629-647:

```python
async def _build_snapshot_or_raise() -> "governance_grounding.Snapshot":
    """Separado de _build_grounding para poder parchearlo en tests."""
    ctx, _, _ = await _governance_context()
    return governance_grounding.build_snapshot(ctx)


async def _build_grounding() -> "governance_grounding.Snapshot | governance_grounding.SnapshotError":
```

El docstring y el cuerpo quedan igual, salvo `return await _build_snapshot_or_raise()` dentro del `try`. Línea ~1115: `grounding = await _build_grounding()`.

`backend/shadow_validation.py:315`: `ctx, predicates, term_categories = await _validation_context()`.

```bash
git -C /home/fruiz/worktrees/jax-platform-gobernanza grep -n "validation_context()\|_build_grounding()\|_build_snapshot_or_raise()" -- backend
```

Todo llamador que no esté en esta lista ni en la del Paso 7 se adapta y se declara en el PR.

- [ ] **Paso 7: adaptar los tests existentes**

- `test_grounding_config_revalidation.py`:
  - En el fixture `repo_copia`, agregar `monkeypatch` a la firma y, antes del `cache_clear()`: `monkeypatch.setattr(governance_context.governance_validator.MotorCatalog, "from_db", AsyncMock(side_effect=lambda: governance_context.governance_validator.MotorCatalog({})))`. Importar `AsyncMock` de `unittest.mock` y `asyncio`. Sin eso, en el job sin DB `from_db` intenta `aiomysql.connect`, y la Regla 2 de conftest solo cubre `create_pool`.
  - Cada `governance_context.validation_context()` pasa a `asyncio.run(governance_context.validation_context())`, y `chat._build_grounding()` a `asyncio.run(chat._build_grounding())`.
  - En `test_sin_cambios_en_disco_el_contexto_no_se_reconstruye`, las dos llamadas van en la MISMA corrida: `primero, segundo = asyncio.run(_dos())`, con `async def _dos(): return (await governance_context.validation_context(), await governance_context.validation_context())`.
  - `CLAIM` cita `/capabilities/10` (`write_file`): no cambia, porque los punteros de `ops` son estables (Tarea 4).
- `test_shadow_origin.py:128` y `:177`: `ctx, _, _ = client.portal.call(validation_context)`.
- `test_shadow_validation.py:50` (`_grounding()`) y `test_shadow_validation_grounding.py:89` (`_snapshot()`): `ctx, _, _ = asyncio.run(governance_context.validation_context())`, con `import asyncio`. Si el test ya corre dentro de `client` (loop del portal), usar `client.portal.call(governance_context.validation_context)`, como en `test_shadow_origin.py`: `asyncio.run` dentro de otro loop falla con `RuntimeError: asyncio.run() cannot be called from a running event loop`.
- `test_chat_grounding_wiring.py` y `test_shadow_validation.py::..._context_load_fails_before_the_insert`: sin cambios. `patch.object` sobre un `async def` crea un `AsyncMock`, y el `side_effect` que lanza se dispara al hacer `await`. Se confirma en el Paso 8.

- [ ] **Paso 8: verde, en las dos modalidades de CI**

```bash
cd /home/fruiz/worktrees/jax-platform-gobernanza/backend && pwd && export JAX_REPO_PATH=/home/fruiz/worktrees/jax-gobernanza-catalogo && \
  /home/fruiz/jax-platform/backend/.venv/bin/python -m pytest -q -rs 2>&1 | tail -5 && \
  JAX_CI_NO_DB=1 /home/fruiz/jax-platform/backend/.venv/bin/python -m pytest -q -rs 2>&1 | tail -5
```

Esperado: 0 failed y 0 errors en las dos. Con DB, passed = (base + Tarea 1) + 11 y los mismos skips (`MAX_SKIPS` no cambia). Sin DB, passed = (base + Tarea 1) + 10 y skipped + 1, porque el último test pide `client`.

**Mutaciones** (backup y restauración con `cp`):
- (a) Sacar el doble chequeo dentro del lock: `test_n_turnos_a_la_vez_disparan_una_sola_recarga` cae con `await_count == 20`.
- (b) Sacar `facet_resolver._seal_mtime()` de `_stamp()`: caen `test_tocar_el_sello_fuerza_la_recarga` y el del catálogo cambiado.
- (c) Envolver `_build()` en un `try/except` que devuelva el `_cache` viejo: cae `test_db_caida_en_caliente_no_sirve_el_catalogo_viejo`.

- [ ] **Paso 9: pisos**: `PISO_PASSED` y `JAX_CI_MIN_PASSED` ACTUALES (que ya tienen la Tarea 1) + delta medido (esperado +11 y +10). Comentario:

```text
# Subido de <actual> a <actual+11> (2026-09-14, tanda A v2, catálogo de la DB):
# test_governance_context_catalogo.py -- 10 puros (catálogo de la DB, hit sin
# recarga, recarga por sello, DB caída en frío y en caliente, una recarga con
# 20 turnos, la Mesa acredita y valida una capability de la DB + CONTROL,
# catálogo cambiado entre snapshot y validación -> FACT_MISMATCH,
# SnapshotError) y 1 con `client` (from_db real con capability.mode). Vistos
# en rojo antes del arreglo; mutaciones en el PR. Medido: <n> / <skips>.
```

- [ ] **Paso 10: commit**

```bash
git -C /home/fruiz/worktrees/jax-platform-gobernanza add backend/governance_context.py backend/api/chat.py backend/shadow_validation.py backend/tests/test_governance_context_catalogo.py backend/tests/test_grounding_config_revalidation.py backend/tests/test_shadow_origin.py backend/tests/test_shadow_validation.py backend/tests/test_shadow_validation_grounding.py .github/workflows/policy.yml
git -C /home/fruiz/worktrees/jax-platform-gobernanza commit -m "$(cat <<'EOF'
fix(gobernanza): el contexto de validación usa el catálogo de la DB

validation_context() pasa a async: carga MotorCatalog.from_db() (con
capability.mode), cachea con los mtimes de config + el sello de
facet_resolver, recarga una sola vez bajo asyncio.Lock y, si la DB falla,
la excepción sube (SnapshotError en el chat) en vez de servir un catálogo
vacío o viejo. El mismo contexto arma el snapshot y valida. Requiere el
jax de la tanda A v2.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LNXrHCwsZAVfrQj56maFHB
EOF
)"
```

---

### Tarea 7: la Mesa de punta a punta y el tripwire ops∩DB (jax-platform, PR-C)

**Archivos:**
- Crear: `backend/tests/test_catalogo_db_en_la_mesa.py`
- Modificar: `.github/workflows/policy.yml` (`PISO_PASSED`; `MAX_SKIPS` no cambia; `JAX_CI_MIN_PASSED` no cambia)

**Interfaces:**
- Consume:
  - `await governance_context.validation_context()` (Tarea 6).
  - `shadow_validation.run_shadow_validation(conv_uuid, shadow_message_id, facet, contract, grounding, origin)`, que inserta la fila de `shadow_messages` y los veredictos.
  - `api.chat.ContractResult`.

- [ ] **Paso 1: escribir los tests**

```python
"""
La Mesa con las capabilities de la DB, de punta a punta, y el TRIPWIRE ops∩DB
(tanda A v2, 2026-09-14).

1. Un claim sobre `generate` que cita su línea de catalog_capabilities queda
   VALID/OBSERVADO en shadow_claim_verdicts, por el MISMO camino que un turno
   real (run_shadow_validation). Con el modo equivocado, FACT_NOT_IN_SNAPSHOT:
   la acreditación lo corta antes del resolver (spec v2 §3.4). Hasta la v2 las
   capabilities de la DB no se podían citar: en producción, 2 AUTHORITY_INVALID
   sobre code_swarm (medido 2026-09-14).
2. TRIPWIRE: `ops` del TOML y `capability` de la DB no comparten nombres. Un
   nombre repetido da SOURCE_CONFLICT y dos líneas del snapshot con el mismo
   hecho. Medido el 2026-09-14: 11 ops, 17 capabilities, intersección vacía.
   Mira el MISMO objeto que ven resolver y snapshot. Alcance declarado: ve lo
   que SIEMBRAN las migraciones; el despliegue lo mide contra producción.
"""
from __future__ import annotations

import shutil
import uuid

import claims as governance_claims
import governance_context
import grounding as governance_grounding
import validator as governance_validator
from api.chat import ContractResult


def _contract(claims):
    return ContractResult(contract_parsed=True, claims=claims, analysis="a", judgment=None,
                          degradation_reason=None, raw_text="...")


async def _veredictos(shadow_message_id):
    from db.connection import get_pool
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT predicate, status, authority, evidence_pointer FROM shadow_claim_verdicts "
                "WHERE shadow_message_id = %s ORDER BY id", (shadow_message_id,))
            return await cur.fetchall()


def _snapshot_real(client):
    governance_context.validation_context.cache_clear()
    ctx, _, _ = client.portal.call(governance_context.validation_context)
    return ctx, governance_grounding.build_snapshot(ctx)


def _correr_la_mesa(client, snap, modo):
    from shadow_validation import run_shadow_validation
    ptr = next(e.pointer for e in snap.entries
               if e.pointer.startswith("/catalog_capabilities/") and e.args["name"] == "generate")
    smid = str(uuid.uuid4())
    client.portal.call(run_shadow_validation, "conv-tanda-a", smid, "jekyll", _contract([
        {"predicate": "CAPABILITY_AVAILABLE", "args": {"name": "generate", "mode": modo}, "evidence_pointer": ptr},
    ]), snap, "test")
    return ptr, client.portal.call(_veredictos, smid)


def test_en_la_mesa_un_claim_sobre_generate_queda_valid_observado(client):
    try:
        _, snap = _snapshot_real(client)
        ptr, filas = _correr_la_mesa(client, snap, "read_only")
    finally:
        governance_context.validation_context.cache_clear()
    assert filas == (("CAPABILITY_AVAILABLE", "VALID", "OBSERVADO", ptr),)


def test_en_la_mesa_el_modo_equivocado_queda_fact_not_in_snapshot(client):
    try:
        _, snap = _snapshot_real(client)
        ptr, filas = _correr_la_mesa(client, snap, "mutating")
    finally:
        governance_context.validation_context.cache_clear()
    assert filas == (("CAPABILITY_AVAILABLE", "FACT_NOT_IN_SNAPSHOT", "INFERIDO", ptr),)


def _en_ambos(ctx):
    return sorted(n for n in ctx.ops if ctx.catalog.get_capability(n) is not None)


def test_tripwire_ops_del_toml_y_capabilities_de_la_db_no_comparten_nombres(client):
    """TRIPWIRE. Ver docstring del módulo, punto 2."""
    try:
        ctx, _ = _snapshot_real(client)
    finally:
        governance_context.validation_context.cache_clear()
    assert ctx.ops, "ops vacío: el tripwire no estaría mirando nada"
    assert ctx.catalog.get_capability("generate") is not None, (
        "catálogo sin la capability sembrada: el tripwire no estaría mirando la DB")
    assert _en_ambos(ctx) == [], (
        f"{_en_ambos(ctx)} está en ops del TOML Y en capability de la DB: el resolver "
        "de CAPABILITY_AVAILABLE va a dar SOURCE_CONFLICT. Renombrar uno de los dos.")


def test_control_el_tripwire_ve_un_nombre_repetido(client, tmp_path):
    """CONTROL del TRIPWIRE: con un config.toml que agrega `[ops.generate]`, la
    intersección da ['generate'] y el resolver da SOURCE_CONFLICT."""
    destino = tmp_path / "jax"
    (destino / "las_manos").mkdir(parents=True)
    shutil.copy(governance_context.JAX_REPO / "las_manos" / "config.toml", destino / "las_manos" / "config.toml")
    with open(destino / "las_manos" / "config.toml", "a", encoding="utf-8") as f:
        f.write("\n[ops.generate]\n")
    anterior = governance_context.JAX_REPO
    governance_context.JAX_REPO = destino
    governance_context.validation_context.cache_clear()
    try:
        ctx, predicates, _ = client.portal.call(governance_context.validation_context)
    finally:
        governance_context.JAX_REPO = anterior
        governance_context.validation_context.cache_clear()
    assert _en_ambos(ctx) == ["generate"]
    claim = governance_claims.Claim(
        predicate="CAPABILITY_AVAILABLE", args={"name": "generate", "mode": "read_only"},
        authority="OBSERVADO", provenance_ref="test", evidence_pointer="test", scope="mesa_web")
    assert governance_validator.validate(claim, predicates, ctx).status == "SOURCE_CONFLICT"
```

Antes de correrlo, confirmar en `shadow_validation._insert_claim_verdict` que `authority` y `evidence_pointer` se guardan como el test espera (`authority` derivada por el servidor, `evidence_pointer` tal cual). Si la tupla leída difiere en forma (por ejemplo, `evidence_pointer` truncado), se ajusta la aserción a lo que guarda el código y se declara.

- [ ] **Paso 2: verde, y ROJO por mutación**

```bash
cd /home/fruiz/worktrees/jax-platform-gobernanza/backend && pwd && JAX_REPO_PATH=/home/fruiz/worktrees/jax-gobernanza-catalogo \
  /home/fruiz/jax-platform/backend/.venv/bin/python -m pytest tests/test_catalogo_db_en_la_mesa.py -v 2>&1 | tail -8
```

Esperado: `4 passed`. Estos tests nacen verdes porque las Tareas 2–6 ya están. Su rojo se ve de dos maneras, y las dos se anotan en el PR:
- **(a) Contra el jax de producción.** Correr los 2 de la Mesa con `JAX_REPO_PATH=/home/fruiz/jax` y `PYTHONDONTWRITEBYTECODE=1`. Caen, porque ese jax no tiene la sección `catalog_capabilities` (`StopIteration` en `next(...)`) y el `governance_context` nuevo lo llama con 3 argumentos (`TypeError`). Es la prueba de que no pasan por casualidad.
- **(b) Mutaciones en el worktree de jax**, con backup y `cp`:
  - Agregar `[ops.generate]` a `las_manos/config.toml`: el TRIPWIRE cae con `['generate'] está en ops del TOML Y en capability de la DB`.
  - Sacar la sección `catalog_capabilities` de `build_snapshot`: caen los 2 de la Mesa.

- [ ] **Paso 3: pisos**: `PISO_PASSED` + medido (esperado +4). En `JAX_CI_MIN_PASSED` no cambia el número (los 4 piden `client` y se saltean sin DB). Se anota en su comentario: `skipped +4 (test_catalogo_db_en_la_mesa.py, los 4 con client)`. Medir con los dos comandos del Paso 8 de la Tarea 6. Comentario de `PISO_PASSED`: `# +4 (2026-09-14, tanda A v2): test_catalogo_db_en_la_mesa.py -- la Mesa de punta a punta (VALID/OBSERVADO; modo equivocado FACT_NOT_IN_SNAPSHOT), TRIPWIRE ops∩DB vacío y su CONTROL. Rojos por mutación y contra el jax de producción.`

- [ ] **Paso 4: commit**

```bash
git -C /home/fruiz/worktrees/jax-platform-gobernanza add backend/tests/test_catalogo_db_en_la_mesa.py .github/workflows/policy.yml
git -C /home/fruiz/worktrees/jax-platform-gobernanza commit -m "$(cat <<'EOF'
test(gobernanza): la Mesa cita capabilities de la DB; tripwire ops∩DB

De punta a punta por run_shadow_validation: generate citada con su línea
de catalog_capabilities queda VALID/OBSERVADO; con el modo equivocado,
FACT_NOT_IN_SNAPSHOT. Tripwire: ops del TOML y capability de la DB no
comparten nombres, con su control. Validados por mutación.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LNXrHCwsZAVfrQj56maFHB
EOF
)"
```

---

### Tarea 8: jax-platform manda `invoked_by = "plataforma"` (PR-C)

**Archivos:**
- Modificar: `backend/api/pipelines.py`: constante nueva, `create_pipeline` (`:95-96`) y `resume_pipeline` (`:161`)
- Modificar: `backend/tests/test_pipelines_identity_injection.py`
- Modificar: `frontend/src/components/BottomBar/PipelineModal.jsx:172` y `:187`
- Modificar: `frontend/src/components/BottomBar/PipelineModal.test.jsx` (test nuevo)
- Modificar: `.github/workflows/policy.yml` (piso de vitest)

**Interfaces:**
- Consume: el contrato de la Tarea 5, `invoked_by == "plataforma"`. `create` lo valida `PipelineCreateRequest` en Jacobs.
- Produce: `api.pipelines.INVOKED_BY_PLATAFORMA = "plataforma"`.

- [ ] **Paso 1: tests (rojos)**

En `test_pipelines_identity_injection.py`, al final de `test_create_pipeline_inyecta_identidad_real`:

```python
    # tanda A (2026-09-14): invoked_by es un ROL que pone el backend, igual que
    # la identidad; lo que mande el cliente se pisa.
    assert captured["json"]["invoked_by"] == "plataforma"
```

En `test_resume_pipeline_inyecta_identidad_real`, reemplazar el comentario viejo ("`"invoked_by": "Fernando"` is kept as the human-readable label...") por:

```python
    # tanda A (2026-09-14): el backend declara el rol "plataforma" (Jacobs
    # exige ese rol para reanudar); la identidad real va en user_id/tenant_id.
    assert captured["json"]["invoked_by"] == "plataforma"
```

Docstring del módulo: `(hoy "Fernando" fijo)` pasa a `(hasta 2026-09-14, "Fernando" fijo; ahora el rol "plataforma", puesto por el backend)`.

Antes, confirmar cómo esos tests mandan el cuerpo del cliente (`grep -n "invoked_by\|captured" backend/tests/test_pipelines_identity_injection.py`). Si el create no manda un `invoked_by` distinto de `"plataforma"`, se hace que mande `"cliente-mintiendo"`: sin eso la aserción no probaría que el backend pisa el valor.

En `PipelineModal.test.jsx`, dentro de `describe('PipelineModal -- cadena en línea', ...)`:

```jsx
  // tanda A (2026-09-14): invoked_by es un rol que pone el backend
  // (api/pipelines.py); el cliente no lo declara, en ninguna de las dos formas.
  it('no manda invoked_by en ninguna de las dos formas', async () => {
    for (const layout of ['chain', 'parallel']) {
      let submitted = null
      const { unmount } = renderModal({ onSubmit: (p) => { submitted = p; return Promise.resolve() } }, { layout })
      await waitFor(() => expect(screen.getByText(/Planificar y ejecutar/i)).not.toBeDisabled())
      if (layout === 'parallel') {
        // En paralelo sin facetas elegidas el submit no hace nada: elegir una.
        fireEvent.click(screen.getAllByRole('checkbox')[0])
      }
      fireEvent.click(screen.getByText(/Planificar y ejecutar/i))
      await waitFor(() => expect(submitted).not.toBeNull())
      expect(submitted).not.toHaveProperty('invoked_by')
      unmount()
    }
  })
```

Antes de correrlo, confirmar en el `.test.jsx` cómo seleccionan una faceta los tests existentes del layout paralelo (`grep -n "checkbox\|fireEvent.click\|renderModal" PipelineModal.test.jsx`) y usar ese mismo selector y la misma firma de `renderModal`.

- [ ] **Paso 2: ver el rojo**

```bash
cd /home/fruiz/worktrees/jax-platform-gobernanza/backend && pwd && JAX_REPO_PATH=/home/fruiz/worktrees/jax-gobernanza-catalogo \
  /home/fruiz/jax-platform/backend/.venv/bin/python -m pytest tests/test_pipelines_identity_injection.py -v 2>&1 | tail -6
cd /home/fruiz/worktrees/jax-platform-gobernanza/frontend && pwd && PATH=/home/fruiz/.nvm/versions/node/v24.16.0/bin:$PATH npx vitest run src/components/BottomBar/PipelineModal.test.jsx 2>&1 | tail -12
```

Esperado:
- En create, `AssertionError: assert 'cliente-mintiendo' == 'plataforma'`.
- En resume, `assert 'Fernando' == 'plataforma'`.
- En vitest, `expected { … } to not have property "invoked_by"`.

- [ ] **Paso 3: implementar**

`backend/api/pipelines.py`, bajo `JACOBS_PIPELINE_TIMEOUT`:

```python
# `invoked_by` es el ROL de quien le pide a Jacobs, no una persona (tanda A,
# 2026-09-14, decisión de Fernando): "plataforma" = pedido de jax-platform en
# nombre de un usuario autenticado. La identidad viaja en user_id/tenant_id y
# la pone este backend; el rol también -- nunca se toma del cliente.
INVOKED_BY_PLATAFORMA = "plataforma"
```

- En `create_pipeline`, junto a `user_id`/`tenant_id`: `body["invoked_by"] = INVOKED_BY_PLATAFORMA`.
- En `resume_pipeline`: `json={"invoked_by": INVOKED_BY_PLATAFORMA, "user_id": user.user_id, "tenant_id": user.tenant_id}`.
- `PipelineModal.jsx`: borrar las dos líneas `invoked_by: 'Fernando',`.

- [ ] **Paso 4: verde**

Correr los dos comandos del Paso 2, más la suite completa de vitest:

```bash
cd /home/fruiz/worktrees/jax-platform-gobernanza/frontend && pwd && PATH=/home/fruiz/.nvm/versions/node/v24.16.0/bin:$PATH npx vitest run --reporter=default --reporter=json --outputFile="$SCRATCH/vitest.json" >/dev/null; node -e 'const r=require(process.argv[1]);console.log(r.numPassedTests,r.numFailedTests)' "$SCRATCH/vitest.json"
```

Esperado: piso actual + 1 y `0` fallidos (hoy `126 0`). Los tests de backend de este archivo no cambian de cantidad: se agregaron aserciones a tests que ya existían, y se declara así en el PR.

- [ ] **Paso 5: piso de vitest** — en el job `frontend-tests`, `r.numPassedTests !== <actual>` y su mensaje al medido, con este comentario:
  `// <actual> -> <medido> el 2026-09-14 (tanda A): PipelineModal.test.jsx (+1) -- el modal no manda invoked_by en ninguna de las dos formas (lo pone el backend como el rol "plataforma"). Visto en rojo antes.`

- [ ] **Paso 6: commit**

```bash
git -C /home/fruiz/worktrees/jax-platform-gobernanza add backend/api/pipelines.py backend/tests/test_pipelines_identity_injection.py frontend/src/components/BottomBar/PipelineModal.jsx frontend/src/components/BottomBar/PipelineModal.test.jsx .github/workflows/policy.yml
git -C /home/fruiz/worktrees/jax-platform-gobernanza commit -m "$(cat <<'EOF'
fix(pipelines): invoked_by es el rol "plataforma" y lo pone el backend

create y resume mandan invoked_by="plataforma" pisando lo que mande el
cliente, como ya se hacía con user_id/tenant_id; el modal deja de mandarlo.
vitest +1.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LNXrHCwsZAVfrQj56maFHB
EOF
)"
```

---

### Tarea 9: latencia y tamaño del snapshot, antes y después

**Archivos:** ninguno en los repos. Los números van a los PRs y a `DEUDA.md` (Tarea 11).

- [ ] **Paso 1: medir el "después"** con el script de la Tarea 6, Paso 2, contra las ramas. La DB es la de producción, solo lectura, que todavía NO tiene `capability.mode`. Por eso el "después" se mide contra `jax_memory_test`, que sí la tiene, y se declara:

```bash
cd /home/fruiz/worktrees/jax-platform-gobernanza/backend && pwd && set -a && . /etc/jax/.env && set +a && \
  PYTHONDONTWRITEBYTECODE=1 JAX_DB_NAME=jax_memory_test JAX_REPO_PATH=/home/fruiz/worktrees/jax-gobernanza-catalogo \
  /home/fruiz/jax-platform/backend/.venv/bin/python "$SCRATCH/medir_validation_context.py" | tee "$SCRATCH/latencia-despues.txt"
```

Qué incluye cada número:
- La recarga "fría" del después incluye `from_db()` (3 SELECT) y `asyncio.to_thread`.
- El "caliente" suma un `stat` del sello. El sello real solo se statea; el script no lo escribe.
- El "después" definitivo, contra `jax_memory` de producción, se repite en la Tarea 11, Paso 5, ya desplegado.

- [ ] **Paso 2: juzgar.**
  - Caliente: p95 del después menor a 50 µs.
  - Frío: p95 menor a 20 ms.
  - Snapshot: 28 entradas (11 ops + 17 capabilities) y un `render` de ~1.800 caracteres (estimado del spec: 1.786).
  - **Umbral de parada:** si el `render` pasa de 2.500 caracteres, o la diferencia de `tokens_in` de la Tarea 11 pasa de 600 tokens, la estimación del spec estaba mal. Se busca la causa y se consulta a Fernando antes del GO. No se sube el umbral para que pase. Si la latencia se pasa, no hay GO: se busca la causa (¿el hilo? ¿el connect?) y se mide de nuevo.
- [ ] **Paso 3: anotar** en el PR-C:
  - Antes y después, `N`, fecha y máquina (hall9000).
  - Qué base se usó en cada medición.
  - Lo declarado como no medido: sin carga sobre `/api/chat` (llama al modelo); crear y reanudar pipelines no cambian de forma (un literal distinto).

---

### Tarea 10: PRs, gate por `headSha` y orden de merge

> **ENMIENDA v4 (2026-09-14 noche, revisión final de las tres ramas) — manda sobre el texto de esta tarea.**
> - **Dos rojos de `mirror-sync` que el texto no preveía (reproducidos):** (a) el CI de PR-B corrido ANTES
>   del merge de PR-A falla (`backend/db_connect_config.py` todavía no existe en jax-platform master); (b)
>   entre el merge de PR-A y el de PR-B, cualquier corrida sobre jax **master** da drift en
>   `credential_resolver._db_conn`. Por eso: se abre y se gatea PR-A primero; PR-B se pushea/abre **después**
>   del merge de PR-A (o, si ya corrió, `gh run rerun` de su run sobre el mismo `headSha`, y se exige que
>   `mirror-sync` esté en success); PR-A → PR-B → PR-C seguidos y **sin ningún push a jax master entre el merge
>   de PR-A y el de PR-B**. Se avisa a Fernando que ese intervalo existe.
> - PR-A vive en el worktree `/home/fruiz/worktrees/jax-platform-capability-mode` (rama `feat/capability-mode`,
>   5 commits: T1 + T1b); el push se hace desde ahí (las refs se comparten, pero es la rama de ese worktree).
> - PR-C ya está rebasado sobre PR-A: `git log origin/master..feat/gobernanza-catalogo-db` muestra PR-A + los
>   commits de las Tareas 6–8 **y** `c8640d0` (pisos tras el rebase). El rebase del Paso 5 se rehace solo si
>   PR-A cambia antes del merge.
> - El Paso 8 (avisar al ejecutor de la etapa 2) queda sin objeto: la etapa 2 se mergeó antes (Ruling 1).
> - Cuerpos de los PRs: borradores en el scratchpad de la sesión (`tanda-a/pr-{a,b,c}-body.md`).

**Acoplamiento declarado (spec §6):**
- El job `jacobs-gobernanza-db` de jax clona `jax-platform` master: PR-B necesita PR-A mergeado para su `test_catalog_mode_db.py`.
- El CI de jax-platform clona `jax` master: PR-C necesita PR-B mergeado.
- Entre el merge de PR-B y el de PR-C, **cualquier** run de CI de jax-platform sin PR-C (master, la etapa 2) falla en los tests de gobernanza. Por eso PR-B y PR-C van uno detrás del otro.
- PR-A sola es inofensiva: el `from_db()` viejo no lee la columna, y si alguien reinicia jax-platform en el medio, la migración corre y no rompe nada.

- [ ] **Paso 1: estado remoto antes de abrir** (lección "verificar estado remoto antes de un PR")

```bash
git -C /home/fruiz/worktrees/jax-gobernanza-catalogo fetch origin && git -C /home/fruiz/worktrees/jax-gobernanza-catalogo log --oneline origin/master -3
git -C /home/fruiz/worktrees/jax-platform-gobernanza fetch origin && git -C /home/fruiz/worktrees/jax-platform-gobernanza log --oneline origin/master -3
gh pr list --repo fjruizhn/Jax --state open; gh pr list --repo fjruizhn/jax-platform --state open
```

Si `master` avanzó, se rebasa y, si hay conflicto en un piso, se suman los dos deltas. Antes de pushear se vuelven a correr los comandos de verde de las Tareas 1–8.

- [ ] **Paso 2: GO de Fernando** para la secuencia completa: los tres merges y el despliegue (Tarea 11). Se le presentan el acoplamiento de arriba, la ventana de segundos del despliegue, la medición de la Tarea 9 y la consecuencia sobre SP4 (spec §2, decisión 4).
- [ ] **Paso 3: PR-A.**
  - `git -C /home/fruiz/worktrees/jax-platform-gobernanza push -u origin feat/capability-mode` y `gh pr create --repo fjruizhn/jax-platform --base master --head feat/capability-mode`.
  - Cuerpo: resumen; la decisión del default (spec §3.1); pisos; mutaciones; "primero de tres: PR-B (Jax) y PR-C dependen de este"; atribución.
  - Gate:

```bash
SHA=$(git -C /home/fruiz/worktrees/jax-platform-gobernanza rev-parse feat/capability-mode)
gh pr view <n> --repo fjruizhn/jax-platform --json headRefOid -q .headRefOid   # tiene que ser == $SHA
gh run list --repo fjruizhn/jax-platform --commit "$SHA" --json name,status,conclusion
```

  Se esperan todos los checks `completed/success` (`frontend-tests`, `backend-tests-con-db`, `backend-tests-no-db`, `no-fail-open-except` y los que haya). Merge: `gh pr merge <n> --repo fjruizhn/jax-platform --merge`.
- [ ] **Paso 4: PR-B.**
  - `git -C /home/fruiz/worktrees/jax-gobernanza-catalogo push -u origin feat/gobernanza-catalogo-db` y `gh pr create --repo fjruizhn/Jax`.
  - Cuerpo: resumen; spec v2 y la decisión de Fernando (2026-09-14, con la cita); pisos (`governance`, `tests-puros`, `jacobs-gobernanza-db`); rojos vistos; mutaciones; "requiere jax-platform#<PR-A> (mergeado); acompaña a jax-platform#<PR-C>"; atribución.
  - Gate por `headSha` (mismo comando contra `fjruizhn/Jax`, en particular `governance`, `tests-puros`, `jacobs-gobernanza-db`, `mirror-sync` y `no-fail-open-except`). Si `jacobs-gobernanza-db` corrió antes del merge de PR-A, se hace `gh run rerun` sobre el run del sha.
- [ ] **Paso 5: PR-C.**
  - `git -C /home/fruiz/worktrees/jax-platform-gobernanza rebase origin/master feat/gobernanza-catalogo-db`. Los commits de PR-A ya están en master y se descartan solos. Verificarlo: `git log --oneline origin/master..feat/gobernanza-catalogo-db` muestra solo los commits de las Tareas 6–8.
  - Volver a correr los verdes de las Tareas 6–8. Push y `gh pr create --repo fjruizhn/jax-platform`.
  - Cuerpo: resumen; latencia y tamaño del snapshot (Tarea 9); mutaciones de las Tareas 6 y 7; "requiere Jax#<PR-B> mergeado primero; su CI queda rojo en gobernanza hasta ese merge" (así se declara); atribución.
- [ ] **Paso 6: merge de PR-B** (`gh pr merge <n> --repo fjruizhn/Jax --merge`, solo si el gate del Paso 4 sigue verde sobre el mismo sha). Enseguida, `gh run rerun <run-id> --repo fjruizhn/jax-platform` sobre el run del `headSha` de PR-C, para que clone el jax nuevo.
- [ ] **Paso 7: gate de PR-C sobre su `headSha`**, todo en success. Después, merge.
- [ ] **Paso 8: avisar al ejecutor de la etapa 2** (`/home/fruiz/worktrees/jax-platform-etapa2`): `master` cambió los pisos, la tabla `capability` (columna `mode`) y `governance_context` (async). Rebasa él y suma deltas.

---

### Tarea 11: despliegue, verificación en vivo y cierre en DEUDA

> **ENMIENDA v4 (2026-09-14 noche, revisión final) — manda sobre el texto de esta tarea.**
> - **Paso 3:** producción YA tiene `capability.mode` como `enum('read_only','mutating') NOT NULL` desde el
>   incidente de la Tarea 1 (Fernando: se deja y se registra). Precondición: 17 filas, solo `file_write`
>   mutating, 0 NULL, 0 filas `zz%`. El respaldo se llama `capability-pre-v3`.
> - **Paso 4 — condición de parada corregida:** después de reiniciar jax-platform, la columna tiene que dar
>   `varchar(16)` / `NO` / sin default **y** `chk_capability_mode` presente en `information_schema.CHECK_CONSTRAINTS`
>   (junto a los 2 CHECK json que ya existían). `enum(...)` después del reinicio = la migración no corrió → PARAR.
> - **Paso 5:** además de la medición en proceso, repetir `medir_concurrente.py` (N=1/20/50 turnos fríos y
>   calientes a la vez, recarga compartida real) contra `jax_memory` de producción en solo lectura.
> - **Paso 11 (DEUDA.md), agregar:**
>   1. **Incidente 2026-09-14 ~15:11 CST** (HISTORIA + VERDAD OPERACIONAL): un script de verificación de la
>      Tarea 1 cargó `/etc/jax/.env` fuera de pytest y corrió `run_migrations()` (versión ENUM), una mutación
>      `ALTER ... DEFAULT` y su reversión, y filas `zz_test_probe*` borradas, contra `jax_memory` de producción.
>      Verificado por el controller: 17 filas correctas, sin restos, servicios sanos. Decisión de Fernando:
>      se deja y se registra; el despliegue la convierte a v3. **Causa:** el brief no advertía que
>      `/etc/jax/.env` apunta a producción fuera de pytest. **Barrera** (Ruling 8, memoria
>      `feedback-brief-barrera-db-produccion`): todo brief con DB lleva la advertencia textual y el controller
>      verifica prod de forma independiente.
>   2. **El ENUM no obligaba a declarar el modo** (HECHO medido, spec v3): omitir una columna ENUM NOT NULL
>      guarda el primer valor sin error; por eso VARCHAR(16)+CHECK.
>   3. **Variables nuevas** también en `jax/CONTEXT.md` (junto a `CREDENTIAL_CACHE_TTL_SECONDS`):
>      `GOVERNANCE_RELOAD_TIMEOUT_SECONDS` (jax-platform, default 5.0: acota la recarga completa del catálogo) y
>      `JAX_DB_CONNECT_TIMEOUT_SECONDS` (jax y jax-platform, default 10: acota el socket TCP de todo aiomysql);
>      las dos se validan en cada lectura (P10). Hoy no están en `/etc/jax/.env` y aplican los defaults.
>   4. Los números de carga concurrente (M4) y el symlink `las_manos/jacobs` pasado a relativo.
> - **Paso 12:** borrar también los auxiliares del scratchpad (`prod_readonly.py`, `prc-db.txt`, `chain.txt`,
>   `catalog.py.backup-task2`, `jax-congelado-168ce00` con `git worktree remove`) y la rama local
>   `respaldo/pr-c-pre-rebase`.

- [ ] **Paso 1: línea base del tamaño del prompt, ANTES de desplegar.**
  - Leer la firma actual de `create_access_token` en `/home/fruiz/jax-platform/backend/auth/jwt.py`: la etapa 2 agregó `tv`/`token_version`. Leer también los `user_id`, `tenant_id`, rol y, si corresponde, `token_version` de la cuenta de Fernando, con una consulta de solo lectura a `jax_users`, verificando antes los nombres de columna con `SHOW COLUMNS FROM jax_users`.
  - Generar el token con el venv de producción y `/etc/jax/.env`.
  - `POST http://127.0.0.1:8080/api/chat` con `{"message": "¿Qué capabilities tiene hoy este sistema, de lectura y de escritura, incluidas generate y file_write? Citá cada una.", "facet": "jekyll", "origin": "probe"}`, en una conversación nueva.
  - Anotar `tokens_in` de la fila de `axioma_usage` de ese turno (`SELECT tokens_in, created_at FROM axioma_usage WHERE facet='jekyll' AND user_id=<id> ORDER BY id DESC LIMIT 1`).
- [ ] **Paso 2: anunciar la ventana a Fernando.** "Reinicio jax-platform y después jax-las-manos. Durante unos segundos, crear o reanudar un pipeline puede fallar (uno manda `plataforma` y el otro todavía espera `Fernando`)."
- [ ] **Paso 3: backup con restauración probada y precondición de la migración**

```bash
set -a && . /etc/jax/.env && set +a
M="mariadb -h $JAX_DB_HOST -P $JAX_DB_PORT -u $JAX_DB_USER -p$JAX_DB_PASSWORD"
$M -N jax_memory -e 'SELECT `key` FROM capability ORDER BY `key`' | tee "$SCRATCH/capabilities-pre.txt"   # esperado: las 17 de _CAPABILITY_MODE
mariadb-dump -h "$JAX_DB_HOST" -P "$JAX_DB_PORT" -u "$JAX_DB_USER" -p"$JAX_DB_PASSWORD" jax_memory capability capability_motor > "$SCRATCH/capability-pre-mode.sql"
$M jax_memory_test < "$SCRATCH/capability-pre-mode.sql" && $M -N jax_memory_test -e 'SELECT COUNT(*) FROM capability'   # restauración probada: 17
git -C /home/fruiz/jax log --oneline -1 | tee "$SCRATCH/sha-jax-antes.txt"; git -C /home/fruiz/jax-platform log --oneline -1 | tee "$SCRATCH/sha-jax-platform-antes.txt"
```

Detalles del paso:
- Si la lista de capabilities no es exactamente la de `_CAPABILITY_MODE`, **se para**: la migración fallaría a propósito con la fila huérfana y jax-platform no arrancaría. Se consulta a Fernando.
- La restauración va a `jax_memory_test`, la base de tests: queda con el esquema viejo y la próxima corrida de tests la migra, que es el camino "base vieja" de la Tarea 1.
- Rollback, si hiciera falta: `git -C <checkout> checkout <sha anterior>` en los dos checkouts y reiniciar los dos servicios. La columna puede quedar: el código viejo no la lee.

- [ ] **Paso 4: traer el código y reiniciar, en orden**

```bash
git -C /home/fruiz/jax status --short && git -C /home/fruiz/jax pull --ff-only && git -C /home/fruiz/jax log --oneline -1
git -C /home/fruiz/jax-platform status --short && git -C /home/fruiz/jax-platform pull --ff-only && git -C /home/fruiz/jax-platform log --oneline -1
ls -la /srv/jax-data/facet-cache-seal          # el sello existe (si no, parar: el caché no se invalidaría)
date -Is | tee "$SCRATCH/deploy-inicio.txt"
sudo systemctl restart jax-platform && sleep 5 && systemctl is-active jax-platform && curl -fsS http://127.0.0.1:8080/api/health; echo
$M -N jax_memory -e "SELECT IS_NULLABLE, COLUMN_DEFAULT, COLUMN_TYPE FROM information_schema.COLUMNS WHERE TABLE_SCHEMA='jax_memory' AND TABLE_NAME='capability' AND COLUMN_NAME='mode'; SELECT \`key\`, mode FROM capability WHERE mode='mutating';"
sudo systemctl restart jax-las-manos && sleep 3 && systemctl is-active jax-las-manos && curl -fsS http://127.0.0.1:7777/health; echo
for u in jax-platform jax-las-manos; do p=$(systemctl show -p MainPID --value $u); echo "$u pid=$p cwd=$(readlink /proc/$p/cwd)"; done
```

Qué se espera y cuándo parar:
- Si `status --short` no está vacío, se para y se pregunta: no se pisa trabajo ajeno.
- La columna tiene que dar `NO	NULL	enum('read_only','mutating')`, y la única fila `mutating` tiene que ser `file_write`.
- **Si jax-platform no queda `active`, o la columna no está, no se reinicia LAS MANOS**, porque su `from_db()` nuevo necesita la columna. Se mira `journalctl -u jax-platform -n 80` y se aplica el rollback.

- [ ] **Paso 5: verificación en vivo 1, el resolver y el snapshot con el contexto de producción** (proceso aparte, solo lectura)

```bash
cd /home/fruiz/jax-platform/backend && pwd && set -a && . /etc/jax/.env && set +a && PYTHONDONTWRITEBYTECODE=1 \
/home/fruiz/jax-platform/backend/.venv/bin/python - <<'PY'
import asyncio, os, sys
sys.path.insert(0, os.getcwd())
import governance_context as gc, grounding as g, validator as v, claims as c

def claim(n, m): return c.Claim(predicate="CAPABILITY_AVAILABLE", args={"name": n, "mode": m},
                                authority="OBSERVADO", provenance_ref="x", evidence_pointer="x", scope="mesa_web")
async def main():
    ctx, preds, _ = await gc.validation_context()
    caps = {e.name: e.mode for e in ctx.catalog.capabilities()}
    print("capabilities de la DB:", len(caps), caps)                                     # 17, solo file_write mutating
    print("ops∩DB:", sorted(set(ctx.ops) & set(caps)))                                   # []
    snap = g.build_snapshot(ctx); texto = g.render(snap)
    print("snapshot entradas:", len(snap.entries), "render_chars:", len(texto))           # 28 / ~1800
    # Resolver directo, sin acreditación (camino legado, OBSERVADO):
    print("generate/read_only ->", v.validate(claim("generate", "read_only"), preds, ctx).status)      # VALID
    print("generate/mutating  ->", v.validate(claim("generate", "mutating"), preds, ctx).status)       # FACT_MISMATCH
    print("file_write/read_only ->", v.validate(claim("file_write", "read_only"), preds, ctx).status)  # FACT_MISMATCH
asyncio.run(main())
PY
```

Además, repetir `medir_validation_context.py` (Tarea 6, Paso 2) con `JAX_REPO_PATH=/home/fruiz/jax` contra `jax_memory`, y guardarlo en `$SCRATCH/latencia-despues-produccion.txt`.

- [ ] **Paso 6: verificación en vivo 2, claim de sonda por el camino real de la Mesa** (`run_shadow_validation`, `origin='probe'`; escribe filas de sonda en producción, como las sondas de SP4)

```bash
cd /home/fruiz/jax-platform/backend && pwd && set -a && . /etc/jax/.env && set +a && PYTHONDONTWRITEBYTECODE=1 \
/home/fruiz/jax-platform/backend/.venv/bin/python - <<'PY'
import asyncio, os, sys, uuid
sys.path.insert(0, os.getcwd())
import governance_context as gc, grounding as g
from api.chat import ContractResult
from shadow_validation import run_shadow_validation
from db.connection import get_pool, close_pool

async def main():
    ctx, _, _ = await gc.validation_context()
    snap = g.build_snapshot(ctx)
    ptr = next(e.pointer for e in snap.entries if e.pointer.startswith("/catalog_capabilities/") and e.args["name"] == "generate")
    for modo in ("read_only", "mutating"):
        smid = str(uuid.uuid4())
        contrato = ContractResult(contract_parsed=True, analysis="sonda tanda A v2", judgment=None,
                                  degradation_reason=None, raw_text="sonda tanda A v2",
                                  claims=[{"predicate": "CAPABILITY_AVAILABLE",
                                           "args": {"name": "generate", "mode": modo}, "evidence_pointer": ptr}])
        await run_shadow_validation("sonda-tanda-a-v2", smid, "jekyll", contrato, snap, "probe")
        pool = await get_pool()
        async with pool.acquire() as conn, conn.cursor() as cur:
            await cur.execute("SELECT status, authority, evidence_pointer FROM shadow_claim_verdicts WHERE shadow_message_id=%s", (smid,))
            print(modo, smid, await cur.fetchall())
    await close_pool()
asyncio.run(main())
PY
```

Esperado:
- `read_only` → `(('VALID', 'OBSERVADO', '/catalog_capabilities/N'),)`.
- `mutating` → `(('FACT_NOT_IN_SNAPSHOT', 'INFERIDO', ...),)`. En la Mesa, el modo equivocado lo corta la acreditación, antes del resolver (spec v2 §3.4). El `FACT_MISMATCH` del resolver quedó verificado en el Paso 5 y, por el camino de la Mesa con el catálogo cambiado, en `test_si_el_catalogo_cambia_entre_el_snapshot_y_la_validacion_el_resolver_da_fact_mismatch`.
- Si `async with pool.acquire() as conn, conn.cursor() as cur` no es válido en esta versión, se anidan los dos `async with`.

- [ ] **Paso 7: verificación en vivo 3, un turno real de la Mesa.** Repetir el turno de sonda del Paso 1: mismo mensaje, faceta, `origin='probe'` y conversación nueva. Después:

```sql
SELECT shadow_message_id, grounding_snapshot_sha256, validated_at,
       JSON_LENGTH(grounding_snapshot, '$.catalog_capabilities') AS n_catalogo
  FROM shadow_messages WHERE origin='probe' AND conv_uuid <> 'sonda-tanda-a-v2'
 ORDER BY id DESC LIMIT 1;
SELECT predicate, status, authority, evidence_pointer, JSON_UNQUOTE(JSON_EXTRACT(args,'$.name'))
  FROM shadow_claim_verdicts WHERE shadow_message_id = '<id>' ORDER BY id;
SELECT tokens_in FROM axioma_usage WHERE facet='jekyll' AND user_id=<id> ORDER BY id DESC LIMIT 1;
```

Esperado:
- `sha256` distinto de `'ERROR'`, `validated_at` no NULL y `n_catalogo = 17`.
- Ningún `SOURCE_CONFLICT`.
- Si la faceta citó una capability de la DB, `OBSERVADO/VALID`.
- La diferencia de `tokens_in` con el Paso 1 es el costo del snapshot (umbral de la Tarea 9).
- Si el modelo no declaró ningún claim sobre capabilities de la DB, se repite una vez con otra formulación y se anota. No es un fallo del sistema, porque lo mecánico ya lo verificó el Paso 6.

- [ ] **Paso 8: verificación en vivo 4, crear y reanudar un pipeline con el rol nuevo.**
  - Desde la UI, o con `POST /api/pipelines` y el token del Paso 1, crear un pipeline `supervised` de objetivo mínimo.
  - Esperar `interrupted` y reanudar (`POST /api/pipelines/<id>/resume`). Luego cancelar, para no gastar de más.
  - Verificar:

```sql
SELECT pipeline_id, invoked_by, user_id, tenant_id, status FROM jacobs_pipelines WHERE pipeline_id='<id>';   -- invoked_by='plataforma'
```

  En los eventos de Jacobs del pipeline se espera `PIPELINE_RESUMED` con `{"by": "plataforma"}`. Si la tabla de eventos no se llama como se espera, se busca con `SHOW TABLES LIKE 'jacobs%'`.
- [ ] **Paso 9: journal limpio**

```bash
journalctl -u jax-platform -u jax-las-manos --since "$(cat "$SCRATCH/deploy-inicio.txt")" -p warning --no-pager | tail -40
```

No se espera ver trazas de `validation_context`, `from_db`, `SnapshotError`, `capability sin mode` ni `invoked_by`. Cualquier aviso nuevo se explica o se trata como fallo.
- [ ] **Paso 10: frontend**, con el procedimiento de `/home/fruiz/jax/CONTEXT.md` §7 ("CORREGIDO 2026-09-14": build → rsync a `/tmp/axioma-deploy/` en la VM dev → `sudo rsync -a --delete --chown=www:www` a `/www/wwwroot/axioma-ia.io/`, con `--exclude .user.ini` en los DOS saltos; backup antes). El backend ya pisa `invoked_by`, así que el bundle viejo sigue funcionando: esto solo publica el modal nuevo.
  - El build se hace en el worktree, llevado al `master` desplegado, para no escribir en el checkout de producción. Las variables `JAX_SSH_PORT`/`JAX_SSH_USER` y el host `172.16.20.11` son las del despliegue de la etapa 4 (`jax-platform/docs/superpowers/plans/2026-09-12-admin-usuarios-etapa-4-contrasenas.md`, ~1131). Se verifican con `ssh -p "$JAX_SSH_PORT" "$JAX_SSH_USER@172.16.20.11" hostname` antes.

```bash
git -C /home/fruiz/worktrees/jax-platform-gobernanza checkout --detach origin/master && git -C /home/fruiz/worktrees/jax-platform-gobernanza log --oneline -1
cd /home/fruiz/worktrees/jax-platform-gobernanza/frontend && pwd && PATH=/home/fruiz/.nvm/versions/node/v24.16.0/bin:$PATH npm run build
ssh -p "$JAX_SSH_PORT" "$JAX_SSH_USER@172.16.20.11" "sudo cp -a /www/wwwroot/axioma-ia.io /www/wwwroot/axioma-ia.io.backup-pre-tanda-a-$(date +%Y%m%d-%H%M%S)"
rsync -a --delete --exclude .user.ini -e "ssh -p $JAX_SSH_PORT" /home/fruiz/worktrees/jax-platform-gobernanza/frontend/dist/ "$JAX_SSH_USER@172.16.20.11:/tmp/axioma-deploy/"
ssh -p "$JAX_SSH_PORT" "$JAX_SSH_USER@172.16.20.11" "sudo rsync -a --delete --exclude .user.ini --chown=www:www /tmp/axioma-deploy/ /www/wwwroot/axioma-ia.io/"
curl -fsS https://axioma-ia.io/ | grep -o 'index-[A-Za-z0-9_-]*\.js'   # == el index-*.js de frontend/dist/index.html
```

- [ ] **Paso 11: DEUDA.md (PR de docs en jax).** En el worktree de jax, rama nueva `docs/deuda-gobernanza-catalogo` desde el `origin/master` actualizado:
  - Cerrar **"El resolver de `CAPABILITY_AVAILABLE` consulta un catálogo que el Bloque 3 vació"**. Va como CERRADO 2026-09-14, con los tres PRs y lo medido en vivo (Pasos 5–7: 17 capabilities, ops∩DB vacío, sonda `VALID/OBSERVADO` y `FACT_NOT_IN_SNAPSHOT`, snapshot con 28 entradas). También el hallazgo de la planificación como HECHO con procedencia: cero `FACT_MISMATCH` históricos en `CAPABILITY_AVAILABLE` (427 VALID, 3 POINTER_MISMATCH, 1 ARGS_MISMATCH, 2 AUTHORITY_INVALID sobre `code_swarm`) y 17 capabilities, no 18.
  - Cerrar **"`invoked_by` es un campo de AUTORIZACIÓN en Jacobs..."**: CERRADO 2026-09-14, rol `plataforma`, verificación del Paso 8, y las filas viejas con "Fernando" quedan como historia.
  - Entrada de rendimiento (LAS CUATRO, punto 4), con fecha y máquina:
    - Latencia de `validation_context()`, fría y caliente, p50/p95/max, antes y después, en `jax_memory_test` y en producción, con `N`.
    - Tamaño del snapshot: entradas y caracteres, antes y después, más la diferencia de `tokens_in` de los turnos de sonda.
    - "sin carga sobre /api/chat (llama al modelo), declarado".
  - En el ítem de SP4 (brazo negativo, ~línea 2827), agregar: "**2026-09-14 (tanda A v2):** el snapshot suma `catalog_capabilities` (28 entradas en vez de 11). La línea base del 2026-09-03 deja de ser directamente comparable; al retomar SP4 hace falta una nueva, con su pre-registro (spec tanda A v2 §2, decisión 4)."
  - Commit con las líneas de atribución, PR, gate por `headSha`, merge y `git -C /home/fruiz/jax pull --ff-only` (solo docs, sin reinicio).
- [ ] **Paso 12: limpiar.** Cuando los PRs estén mergeados y lo medido esté en `DEUDA.md`:
  - `git -C /home/fruiz/jax-platform worktree remove /home/fruiz/worktrees/jax-platform-gobernanza`
  - `git -C /home/fruiz/jax worktree remove /home/fruiz/worktrees/jax-gobernanza-catalogo`
  - Borrar las ramas locales mergeadas, los venvs y los archivos del scratchpad (lección "limpiar scratchpad tras publicar"). El dump de `capability` se guarda hasta confirmar 24 h sin incidentes y después se borra.

---

## Autorrevisión (contra el spec v2)

- §1 (defecto real: no se puede citar; rama inalcanzable) → Tareas 4 y 7 lo arreglan. La Tarea 11, Pasos 6–7, lo verifica en producción.
- §2 decisión 1 (snapshot con las capabilities de la DB) → Tarea 4; de punta a punta, Tarea 7.
- §2 decisión 2 / §3.1 (`capability.mode`, semilla, NOT NULL sin default, tripwires) → Tarea 1.
- §2 decisión 3 / §3.2–3.3 (`CapabilityEntry.mode`, `from_db`, resolver verifica el modo) → Tareas 2 y 3.
- §2 decisión 4 (SP4 no comparable) → la medición está en las Tareas 6 (antes), 9 y 11 (después) y la nota en DEUDA.md en la Tarea 11, Paso 11.
- §3.4 (esquema de punteros, `SECTION_PREDICATE`, `render`, invariante de SP3, efecto sobre tests existentes) → Tarea 4, que busca los tests que dependen del JSON literal en el Paso 1. La Tarea 6, Paso 7, confirma que `/capabilities/10` sigue válido.
- §3.5 (async, clave con el sello, lock, falla visible, mismo contexto para snapshot y validación) → Tarea 6.
- §3.6 (Jacobs `plataforma`; jax-platform lo pone; el modal no lo manda; filas viejas intactas) → Tareas 5 y 8.
- §4 (LAS CUATRO) → Restricciones globales, Tarea 6 (docstring) y Tareas 9 y 11.
- §5 (tests) → Tareas 1–8; cada uno con su rojo declarado, y los que pasan antes, declarados y validados por mutación.
- §6 (tres PRs en orden, despliegue con `SHOW COLUMNS` entre reinicios, ventana anunciada, frontend por CONTEXT.md §7, verificación en vivo) → Tareas 10 y 11.
- §7 (fuera de alcance) → respetado: `ops` sigue en el TOML, sin línea base nueva de SP4 y sin `mode` en `get_motor_governance()`.
- **Nombres consistentes entre tareas:** `_CAPABILITY_MODE`, `_FILE_CAPABILITY_SEED`, `CAPABILITY_MODES`, `CapabilityEntry.mode`, `MotorCatalog.capabilities()`, `load_validation_context(repo_root, allowlist, catalog)`, `SECTION_PREDICATE["catalog_capabilities"]`, `/catalog_capabilities/N`, `validation_context()` async + `.cache_clear()`, `_build_grounding()` async, `INVOKER_PLATAFORMA` (jax), `INVOKED_BY_PLATAFORMA` (jax-platform) y el literal `"plataforma"`.
- **Placeholders:** los `<actual>`, `<medido>`, `<n>` y `<id>` son valores que se leen o se miden al ejecutar, por regla de las Restricciones globales ("nunca se copia un número sin medir"). No son contenido por completar.
