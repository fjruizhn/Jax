# Gobernanza con el catálogo de la DB e `invoked_by` como rol (tanda A) — Plan de implementación

> **Para agentes ejecutores:** SUB-SKILL OBLIGATORIA: usar superpowers:subagent-driven-development (recomendado) o superpowers:executing-plans para implementar este plan tarea por tarea. Los pasos usan casillas (`- [ ]`) para el seguimiento.

**Objetivo:** que el resolver de `CAPABILITY_AVAILABLE` consulte el catálogo real de la DB (no el `[capabilities.*]` vacío del TOML) y que `invoked_by` deje de ser un nombre de persona usado como rol (`"Fernando"` → `"plataforma"`).

**Arquitectura:** el validador de `jax` sigue puro: `load_validation_context()` **recibe** el `MotorCatalog`. En `jax-platform`, `governance_context.validation_context()` pasa a ser `async`, carga el catálogo con `await MotorCatalog.from_db()` y lo cachea con una clave que suma el mtime del sello de `facet_resolver` a los mtimes de los tres archivos de config. Una recarga a la vez (`asyncio.Lock`). Si la DB falla, la excepción sube: nunca se sirve un catálogo vacío. Jacobs acepta el rol `plataforma` y exige ese rol para reanudar y aprobar. jax-platform lo pone en el backend y el modal deja de mandarlo.

**Stack:** Python 3.14 (venvs de hall9000) / 3.12 y 3.14 (runners de CI), FastAPI, aiomysql, pydantic, pytest + pytest-asyncio, React 19 + vitest.

**Spec:** `docs/superpowers/specs/2026-09-14-gobernanza-catalogo-db-design.md` (repo `jax`, commit `0e44acd`). El spec es la autoridad; este plan argumenta desde él. Quien ejecuta lee los dos.

## Restricciones globales

Cada tarea las incluye implícitamente.

- **Checkouts de producción intocables.** Ningún comando de las Tareas 1–7 edita, prueba en el lugar ni commitea en `/home/fruiz/jax` ni en `/home/fruiz/jax-platform`: los servicios corren desde ahí. Solo la Tarea 8, después del merge, hace `git pull --ff-only` allí. Importar módulos de esos árboles para medir se hace con `PYTHONDONTWRITEBYTECODE=1`, que no escribe `__pycache__`.
- **Worktrees:** jax → `/home/fruiz/worktrees/jax-gobernanza-catalogo` (rama `feat/gobernanza-catalogo-db`). jax-platform → `/home/fruiz/worktrees/jax-platform-gobernanza` (rama `feat/gobernanza-catalogo-db`, creada desde `origin/master`). Usar siempre `git -C <ruta>` y rutas absolutas; si un comando necesita `cd`, `pwd` va en el MISMO comando.
- **Intérpretes:** jax-platform → `/home/fruiz/jax-platform/backend/.venv/bin/python` (solo se USA el intérprete: no se edita ese árbol). `frontend/node_modules` del worktree es un symlink a `/home/fruiz/jax-platform/frontend/node_modules`. jax → venv limpio en el scratchpad con EXACTAMENTE el `pip install` del job de CI que corre esos tests (lección "reproducir el runner, no el local"). `/home/fruiz/jax/.venv` no tiene `fastapi` ni `pytest-asyncio`: medido el 2026-09-14.
- **`JAX_REPO_PATH`:** toda corrida de tests o mediciones de jax-platform en el worktree exporta `JAX_REPO_PATH=/home/fruiz/worktrees/jax-gobernanza-catalogo`. Sin eso, `governance_context` importa el validador del checkout de producción.
- **Trabajo concurrente:** en `/home/fruiz/worktrees/jax-platform-etapa2` (rama `feat/admin-usuarios-etapa2-sesiones`) se ejecuta otro plan que toca auth del backend y sus tests. Este plan no depende de él. **El que mergee segundo rebasa**, y en los pisos de CI suma los dos deltas: nunca pisa el número del otro.
- **TDD:** cada test nuevo se ve en ROJO **por la razón declarada** antes del arreglo. Si falla por otra razón (import, fixture, colección), primero se corrige eso y se vuelve a ver el rojo correcto.
- **Controles declarados como controles:** un test que existe para que otro signifique algo lleva en el docstring la palabra `CONTROL` y dice qué controla. Un tripwire lleva `TRIPWIRE`. "Un control que no falla no valida": cada tripwire se ve en rojo por mutación, no solo en verde.
- **P10, sin fail-open:** ningún `except` amplio nuevo. Si hiciera falta uno, lleva `# fail-soft: <razón>` en la MISMA línea. Una recarga fallida sube la excepción: no se traga ni se sirve el catálogo viejo o vacío.
- **LAS CUATRO:** (1) índices: `from_db()` lee tres tablas chicas completas, sin `WHERE` ni `ORDER BY` sobre tablas que crezcan, y solo al recargar. Se declara, no hay `EXPLAIN` que hacer. (2) Caché con invalidación declarada en el mismo commit: el sello de `facet_resolver`. (3) Solo async: aiomysql, y la lectura de YAML/TOML va en `asyncio.to_thread`. (4) Latencia en proceso antes y después, escrita en `DEUDA.md` de jax con fecha (Tarea 6 → Tarea 8). No se hace carga sobre `/api/chat`, porque llama al modelo, y así se declara.
- **i18n es/en:** este plan no agrega texto visible al usuario. Si una tarea terminara agregando alguno, va en `frontend/src/i18n/es.js` y `en.js`.
- **Pisos de CI exactos:** nunca se copia un número de este plan sin medir. Se lee el valor ACTUAL del workflow (el plan concurrente puede haberlo subido), se sube en el delta MEDIDO en la rama y se agrega un comentario con fecha, qué tests y "visto en rojo". Hoy (2026-09-14): jax `tests-puros` 143 y `governance` 80; jax-platform `PISO_PASSED` 611, `JAX_CI_MIN_PASSED` 301 y vitest 125.
- **Commits:** cada mensaje termina con

  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01LNXrHCwsZAVfrQj56maFHB
  ```

  El cuerpo de cada PR termina con `🤖 Generated with [Claude Code](https://claude.com/claude-code)` y la línea `https://claude.ai/code/session_01LNXrHCwsZAVfrQj56maFHB`.
- **PRs:** el gate se mide sobre el `headSha` exacto del PR (`gh run list --commit <sha>`), no sobre "el último run". No se mergea con un check en rojo ni pendiente. El merge y el despliegue van con GO de Fernando.
- **Despliegue:** primero `jax-platform` y después `jax-las-manos`, anunciando antes la ventana de segundos en la que crear o reanudar un pipeline puede fallar.

## Hallazgo de la planificación: parar antes de ejecutar (Tarea 0)

Medido el 2026-09-14 contra el código y la DB de producción (solo lectura):

1. **En la Mesa, un claim sobre una capability que existe solo en la DB no llega nunca al resolver.** `shadow_validation.run_shadow_validation` siempre valida con `accreditation`. `grounding.accredit()` solo da `ACCREDITED` si el `evidence_pointer` apunta a una entrada del snapshot, y el snapshot se arma solo con `ctx.ops` (`grounding.build_snapshot`). La decisión §2 del spec deja el snapshot así. Un claim sobre `generate` da entonces `AUTHORITY_INVALID` (sin puntero), `POINTER_MISMATCH` o `FACT_NOT_IN_SNAPSHOT` (con puntero), en el paso 4 o 5 de `validator.validate()`, **antes** del resolver. En producción, `shadow_claim_verdicts` para `CAPABILITY_AVAILABLE` tiene hoy: `OBSERVADO/VALID` 427, `INFERIDO/POINTER_MISMATCH` 3, `INFERIDO/ARGS_MISMATCH` 1, `NULL/AUTHORITY_INVALID` 2 (los 2 sobre `code_swarm`). **Cero `FACT_MISMATCH`.** El §1 del spec ("sale `FACT_MISMATCH`, un falso negativo") y dos puntos no se cumplen tal como están escritos: la verificación en vivo del §6 ("un claim de la Mesa sobre una capability de la DB sale `VALID`") y el test del §5 ("la validación en sombra da `VALID` para una capability que solo está en la DB").
2. **La tabla `capability` de producción tiene 17 filas, no 18.** No está `validate`. La intersección con los 11 `ops` sigue vacía (medido).

**Qué hace este plan con eso.** Implementa el diseño del spec tal cual (el catálogo real llega al resolver) y verifica solo lo que es alcanzable: el resolver, alimentado con el contexto REAL que arma jax-platform, da `VALID` para una capability de la DB. Además deja un TRIPWIRE que fija que en la Mesa ese claim sigue sin llegar al resolver mientras el snapshot no incluya el catálogo. El cambio es, por ahora, latente en la Mesa. Lo que sí cambia en producción es que un nombre repetido entre `ops` y la DB pasa a dar `SOURCE_CONFLICT` (hoy es imposible: intersección vacía y tripwire en la Tarea 5).

- [ ] **Paso 0.1: presentar el hallazgo a Fernando y esperar su decisión antes de la Tarea 1.** Opciones:
  - (a) Seguir con este plan: el fix queda latente en la Mesa hasta el sub-proyecto del snapshot, y la verificación en vivo es la de la Tarea 8.
  - (b) Ampliar el alcance para sumar las capabilities de la DB al snapshot. Eso revierte la decisión §2 y requiere un spec nuevo.

  Anotar la decisión, con fecha, en el PR de jax. Si elige (b), este plan se detiene aquí.

---

## Mapa de archivos

**jax** (`/home/fruiz/worktrees/jax-gobernanza-catalogo`):

| Archivo | Responsabilidad | Tarea |
|---|---|---|
| `policy/governance/validator.py` | `load_validation_context(repo_root, allowlist, catalog)`: recibe el catálogo | 1 |
| `tests/test_governance_validator.py` | `_real_ctx` con catálogo explícito; el tripwire viejo se reemplaza | 1 |
| `jacobs/models.py` | `VALID_INVOKERS = {"plataforma","jax_local","ada"}`, `INVOKER_PLATAFORMA` | 2 |
| `jacobs/policy.py` | `validate_resume` exige `"plataforma"` | 2 |
| `jacobs/routes.py` | `plan_only` usa `VALID_INVOKERS` (fuera el literal duplicado); docstrings | 2 |
| `tests/test_jacobs_invoked_by_rol.py` (nuevo) | rol al crear, planificar, reanudar y aprobar | 2 |
| `.github/workflows/policy.yml` | pisos de `governance` y `tests-puros`; archivo nuevo en `tests-puros` | 1, 2 |
| `DEUDA.md` | cierre de los dos ítems y números de latencia (PR de docs, post-deploy) | 8 |

**jax-platform** (`/home/fruiz/worktrees/jax-platform-gobernanza`):

| Archivo | Responsabilidad | Tarea |
|---|---|---|
| `backend/governance_context.py` | `async validation_context()`, clave con el sello, lock, falla visible | 3 |
| `backend/api/chat.py` | `_build_snapshot_or_raise` / `_build_grounding` async; `await` en `chat()` | 3 |
| `backend/shadow_validation.py` | `await _validation_context()` | 3 |
| `backend/tests/test_governance_context_catalogo.py` (nuevo) | catálogo de la DB, sello, lock, DB caída | 3 |
| `backend/tests/test_grounding_config_revalidation.py` | fixture parchea `from_db`; llamadas async | 3 |
| `backend/tests/test_shadow_origin.py`, `test_shadow_validation.py`, `test_shadow_validation_grounding.py` | llamadas a `validation_context()` con `await` | 3 |
| `backend/api/pipelines.py` | `invoked_by = "plataforma"` al crear y reanudar | 4 |
| `backend/tests/test_pipelines_identity_injection.py` | afirma `"plataforma"` aunque el cliente mande otra cosa | 4 |
| `frontend/src/components/BottomBar/PipelineModal.jsx` (+ `.test.jsx`) | deja de mandar `invoked_by` | 4 |
| `backend/tests/test_tripwires_catalogo_db.py` (nuevo) | tripwire ops∩DB vacío + control + tripwire Mesa | 5 |
| `.github/workflows/policy.yml` | `PISO_PASSED`, `JAX_CI_MIN_PASSED`, vitest | 3, 4, 5 |

**Por qué el tripwire ops∩DB va en jax-platform y no en el job `jacobs-gobernanza-db` de jax (Tarea 5):** jax-platform es el único proceso donde las dos fuentes se juntan en el mismo objeto. El test puede afirmar sobre el `ValidationContext` que produce `await governance_context.validation_context()`, que es exactamente lo que ve el resolver en producción: `ctx.ops` del TOML clonado de jax (`JAX_REPO_PATH=/tmp/jax`) y `ctx.catalog` de la DB migrada por las migraciones de jax-platform. En jax habría que recomponer esa unión a mano, y eso sería una segunda definición del contexto. El job `backend-tests-con-db` ya tiene las dos cosas (clon de jax + MariaDB migrada) y un piso exacto.

---

### Tarea 1: el validador de jax recibe el catálogo

**Archivos:**
- Modificar: `policy/governance/validator.py:78-90` (`load_validation_context`)
- Modificar: `tests/test_governance_validator.py:229-231` (`_real_ctx`) y `:272-301` (se reemplaza el tripwire)
- Modificar: `.github/workflows/policy.yml`, job `governance`, paso "Piso exacto de tests CORRIDOS"

**Interfaces:**
- Consume: `las_manos.motor_registry.catalog.MotorCatalog` (sin cambios).
- Produce: `validator.load_validation_context(repo_root: Path, config_paths_allowlist: frozenset[str], catalog: MotorCatalog) -> ValidationContext`. `ctx.catalog` es el objeto recibido (identidad, no copia). La Tarea 3 lo llama con el catálogo de `await MotorCatalog.from_db()`.

- [ ] **Paso 1: preparar el venv limpio del job `governance`**

```bash
SCRATCH=<scratchpad de la sesión que ejecuta>
python3.14 -m venv "$SCRATCH/venv-jax-gov"
"$SCRATCH/venv-jax-gov/bin/pip" install -q pytest pyyaml pydantic aiomysql==0.3.2
git -C /home/fruiz/worktrees/jax-gobernanza-catalogo status --short   # esperado: vacío
```

- [ ] **Paso 2: escribir los tests (rojos)**

En `tests/test_governance_validator.py`, reemplazar `_real_ctx` (líneas 229-231) por:

```python
def _real_ctx(catalog: MotorCatalog | None = None) -> "validator.ValidationContext":
    """ops y allowlist REALES (config.toml y closed_vocabulary.yaml del repo).
    El catálogo lo pasa el test: desde 2026-09-14 el validador ya no lo arma
    del TOML (spec tanda A, §3.1). Sin catálogo explícito va uno vacío, que es
    lo que necesitan los tests de la rama `ops`."""
    vocab = loaders.load_vocabulary()
    return validator.load_validation_context(
        REPO_ROOT, vocab.config_paths, catalog if catalog is not None else MotorCatalog({})
    )
```

Borrar entero `test_real_toml_catalog_is_empty_since_block3_so_catalog_branch_is_dead_in_production` (líneas 272-301). En su lugar:

```python
def test_load_validation_context_usa_el_catalogo_que_recibe():
    """Reemplaza al tripwire `test_real_toml_catalog_is_empty_since_block3_so_
    catalog_branch_is_dead_in_production` (2026-09-02 -> 2026-09-14). Ese test
    fijaba que el validador armaba el catálogo desde el TOML vacío; el arreglo
    de la tanda A (spec 2026-09-14, §3.1) lo pone rojo, como estaba previsto,
    y este describe el estado nuevo: el catálogo es el que pasa el llamador
    (en producción, jax-platform con `await MotorCatalog.from_db()`), y una
    capability que existe solo ahí resuelve VALID por la rama `in_catalog`.

    §3.3 del spec de SP3 y `grounding.SECTION_PREDICATE`, revisados como pedía
    el tripwire: el snapshot sigue listando solo `ctx.ops` (decisión §2 de la
    tanda A), así que en la Mesa un claim sobre `generate` todavía no llega al
    resolver -- lo fija el tripwire de jax-platform
    `test_tripwires_catalogo_db.py::test_tripwire_en_la_mesa_un_claim_sobre_capability_de_la_db_no_llega_al_resolver`."""
    catalogo = MotorCatalog({"capabilities": {"generate": {"allowed_motors": ["kimi"]}}})
    ctx = _real_ctx(catalogo)
    assert ctx.catalog is catalogo
    claim = _claim(
        predicate="CAPABILITY_AVAILABLE",
        args={"name": "generate", "mode": "read_only"},
    )
    verdict = validator.validate(claim, PREDICATES, ctx)
    assert verdict.status == "VALID", verdict.detail
    assert "catálogo" in verdict.detail


def test_control_sin_la_capability_en_el_catalogo_el_mismo_claim_es_fact_mismatch():
    """CONTROL de `test_load_validation_context_usa_el_catalogo_que_recibe`:
    mismo claim, mismo repo, catálogo vacío -> FACT_MISMATCH. Prueba que el
    VALID de arriba sale del catálogo recibido y no de `ops` ni de otra
    fuente."""
    ctx = _real_ctx(MotorCatalog({}))
    claim = _claim(
        predicate="CAPABILITY_AVAILABLE",
        args={"name": "generate", "mode": "read_only"},
    )
    verdict = validator.validate(claim, PREDICATES, ctx)
    assert verdict.status == "FACT_MISMATCH"
```

- [ ] **Paso 3: correr y ver el rojo**

```bash
cd /home/fruiz/worktrees/jax-gobernanza-catalogo && pwd && "$SCRATCH/venv-jax-gov/bin/python" -m pytest tests/test_governance_validator.py -q 2>&1 | tail -15
```

Esperado: ROJO en todo test que use `_real_ctx`, con `TypeError: load_validation_context() takes 2 positional arguments but 3 were given`. Es la razón declarada: el validador todavía no recibe el catálogo.

- [ ] **Paso 4: implementar**

En `policy/governance/validator.py`, reemplazar `load_validation_context` (líneas 78-90) por:

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
    TOML, cuyo `[capabilities.*]` quedó vacío con el Bloque 3 (capabilities a
    la DB): la rama `in_catalog` de `_resolve_capability_available` era código
    muerto en producción. Spec: docs/superpowers/specs/
    2026-09-14-gobernanza-catalogo-db-design.md §3.1."""
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

Confirmar que no queda ningún otro llamador en jax:

```bash
git -C /home/fruiz/worktrees/jax-gobernanza-catalogo grep -n "load_validation_context(" -- '*.py'
```

Esperado: solo `validator.py` y `tests/test_governance_validator.py`.

- [ ] **Paso 5: ver el verde, con la suite del job completa**

```bash
cd /home/fruiz/worktrees/jax-gobernanza-catalogo && pwd && "$SCRATCH/venv-jax-gov/bin/python" -m pytest tests/test_governance_claims.py tests/test_governance_loaders.py tests/test_governance_validator.py tests/test_governance_vocab_sweep.py tests/test_governance_renderer.py tests/test_governance_grounding.py -q 2>&1 | tail -3
```

Esperado: `81 passed` (80 − 1 tripwire + 2 nuevos). Si el número difiere, se usa el MEDIDO y se explica en el comentario del piso.

- [ ] **Paso 6: subir el piso del job `governance`**

En `.github/workflows/policy.yml`, en el paso "Piso exacto de tests CORRIDOS" del job `governance`, leer el valor actual (hoy `80`) y reemplazarlo por el medido en los dos lugares (`grep -qE "^81 passed"` y el mensaje `se esperaban 81`). Agregar al final del comentario del paso:

```yaml
        # 80 -> 81 el 2026-09-14 (tanda A, catálogo de la DB): se reemplaza el
        # tripwire del catálogo TOML vacío (-1) por
        # test_load_validation_context_usa_el_catalogo_que_recibe y su CONTROL
        # (+2). El tripwire se vio en rojo con el arreglo, como estaba previsto;
        # los nuevos, en rojo antes del arreglo (TypeError de la firma vieja).
```

- [ ] **Paso 7: commit**

```bash
git -C /home/fruiz/worktrees/jax-gobernanza-catalogo add policy/governance/validator.py tests/test_governance_validator.py .github/workflows/policy.yml
git -C /home/fruiz/worktrees/jax-gobernanza-catalogo commit -m "$(cat <<'EOF'
fix(gobernanza): el validador recibe el catálogo de capabilities

load_validation_context() deja de armar MotorCatalog desde el TOML (vacío
desde el Bloque 3) y recibe el catálogo del llamador. El tripwire del
catálogo vacío se reemplaza por el test del estado nuevo y su control.
Piso de governance 80 -> 81.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LNXrHCwsZAVfrQj56maFHB
EOF
)"
```

---

### Tarea 2: Jacobs acepta el rol `plataforma`

**Archivos:**
- Modificar: `jacobs/models.py:51` (`VALID_INVOKERS`)
- Modificar: `jacobs/policy.py:12` (import) y `:81-88` (`validate_resume`)
- Modificar: `jacobs/routes.py:20-27` (import), `:100-104` (`plan_only`) y los docstrings de `resume_pipeline` (`:252`) y `approve_step` (`:302-306`)
- Crear: `tests/test_jacobs_invoked_by_rol.py`
- Modificar: `.github/workflows/policy.yml`, job `tests-puros` (las dos listas de archivos y el piso)

**Interfaces:**
- Produce: `jacobs.models.INVOKER_PLATAFORMA = "plataforma"` y `jacobs.models.VALID_INVOKERS = frozenset({"plataforma", "jax_local", "ada"})`. El literal `"plataforma"` es el contrato con jax-platform (Tarea 4).
- `policy.validate_resume(invoked_by: str) -> PolicyResult` da `ok=True` solo con `"plataforma"`.

- [ ] **Paso 1: venv limpio del job `tests-puros`**

```bash
python3.14 -m venv "$SCRATCH/venv-jax-puros"
"$SCRATCH/venv-jax-puros/bin/pip" install -q pytest pytest-asyncio aiomysql httpx cryptography pydantic pyyaml aiofiles fastapi==0.139.0
```

- [ ] **Paso 2: escribir el test (rojo)**

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

- [ ] **Paso 3: correr y ver el rojo**

```bash
cd /home/fruiz/worktrees/jax-gobernanza-catalogo && pwd && PYTHONPATH=.:las_manos "$SCRATCH/venv-jax-puros/bin/python" -m pytest tests/test_jacobs_invoked_by_rol.py -v 2>&1 | tail -15
```

Esperado: los 8 en ROJO por la razón declarada. El set de invocadores no coincide. `validate_create("Fernando")` da `ok=True`. `"plataforma"` es rechazado, y en `plan_only` da 403 donde se esperaba 423. `DID NOT RAISE ValidationError`. Reanudar y aprobar con `"Fernando"` dan 404 donde se esperaba 403. **Si alguno falla en la colección** (`ModuleNotFoundError`), es una dependencia que falta en el `pip install` del job: se agrega en el workflow con un comentario "medido en venv limpio" y se vuelve a ver el rojo correcto. No se sigue sin entenderlo.

- [ ] **Paso 4: implementar**

`jacobs/models.py`, línea 51:

```python
# `invoked_by` es un ROL de quien pide, no el nombre de una persona (tanda A,
# 2026-09-14, decisión de Fernando). "plataforma" = pedido de jax-platform en
# nombre de un usuario autenticado; QUIÉN es viaja en user_id/tenant_id. Las
# filas viejas de jacobs_pipelines con "Fernando" quedan como están: historia.
INVOKER_PLATAFORMA = "plataforma"
VALID_INVOKERS = frozenset({INVOKER_PLATAFORMA, "jax_local", "ada"})
```

`jacobs/policy.py`: import `from jacobs.models import INVOKER_PLATAFORMA, VALID_INVOKERS` y reemplazar `validate_resume`:

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

`jacobs/routes.py`: agregar `VALID_INVOKERS` al import de `jacobs.models` y reemplazar el literal de `plan_only` (línea 100):

```python
    if req.invoked_by not in VALID_INVOKERS:
```

Cambiar los docstrings: en `resume_pipeline`, `"""Reanuda un pipeline interrumpido. Solo el rol 'plataforma' puede hacerlo."""`. En `approve_step`, reemplazar la línea `Solo Fernando puede aprobar.` por `Solo el rol 'plataforma' puede aprobar.`

Buscar restos:

```bash
git -C /home/fruiz/worktrees/jax-gobernanza-catalogo grep -n '"Fernando"' -- jacobs/ las_manos/ tests/
```

Esperado: solo fixtures que fabrican `Pipeline(invoked_by="Fernando")` sin pasar por la validación (`jacobs/_pipeline_identity_test.py`, `jacobs/_direct_usage_test.py`). El spec §3.3 dice que esas no cambian. Cualquier otro sitio que valide contra `"Fernando"` se corrige y se agrega a este commit.

- [ ] **Paso 5: ver el verde, con la lista completa del job**

Correr el comando del paso "Piso exacto" del job `tests-puros`, copiado del workflow ACTUAL con `tests/test_jacobs_invoked_by_rol.py` agregado al final:

```bash
cd /home/fruiz/worktrees/jax-gobernanza-catalogo && pwd && PYTHONPATH=.:las_manos "$SCRATCH/venv-jax-puros/bin/python" -m pytest -q <lista del workflow> tests/test_jacobs_invoked_by_rol.py 2>&1 | tail -3
```

Esperado: `151 passed` (143 + 8), o el piso actual + 8 si otro PR lo subió.

- [ ] **Paso 6: workflow**

En el job `tests-puros`, agregar `tests/test_jacobs_invoked_by_rol.py` a las DOS listas (el `run: >-` y el paso "Piso exacto"). Subir el `grep -qE "^143 passed"` y su mensaje al valor medido, con este comentario:

```yaml
          # 143 -> 151 el 2026-09-14 (tanda A): test_jacobs_invoked_by_rol.py (+8) --
          #   invoked_by es el rol "plataforma" y no "Fernando": al crear (policy y
          #   request), planificar, reanudar y aprobar. Vistos en rojo antes del arreglo.
```

- [ ] **Paso 7: commit**

```bash
git -C /home/fruiz/worktrees/jax-gobernanza-catalogo add jacobs/models.py jacobs/policy.py jacobs/routes.py tests/test_jacobs_invoked_by_rol.py .github/workflows/policy.yml
git -C /home/fruiz/worktrees/jax-gobernanza-catalogo commit -m "$(cat <<'EOF'
fix(jacobs): invoked_by es el rol "plataforma", no un nombre de persona

VALID_INVOKERS = {plataforma, jax_local, ada}; validate_resume exige
"plataforma"; plan_only usa VALID_INVOKERS en vez de un literal duplicado.
tests-puros 143 -> 151.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LNXrHCwsZAVfrQj56maFHB
EOF
)"
```

---

### Tarea 3: `governance_context` async, con el catálogo de la DB y el sello en la clave

**Archivos:**
- Modificar: `backend/governance_context.py` (entero, abajo)
- Modificar: `backend/api/chat.py:629-647` y `:1115`
- Modificar: `backend/shadow_validation.py:315`
- Crear: `backend/tests/test_governance_context_catalogo.py`
- Modificar: `backend/tests/test_grounding_config_revalidation.py`, `test_shadow_origin.py` (`:128`, `:177`), `test_shadow_validation.py` (`:50`), `test_shadow_validation_grounding.py` (`:89`)
- Modificar: `.github/workflows/policy.yml` (`PISO_PASSED`, `JAX_CI_MIN_PASSED`)

**Interfaces:**
- Consume: `validator.load_validation_context(repo_root, allowlist, catalog)` (Tarea 1); `MotorCatalog.from_db()` (jax, `las_manos/motor_registry/catalog.py:146`, async, lee `JAX_DB_*`); `facet_resolver._seal_mtime() -> float | None` (jax-platform, sin cambios: es un espejo verificado por `mirror-sync`).
- Produce: `async def governance_context.validation_context() -> tuple[ValidationContext, dict, dict]` y `governance_context.validation_context.cache_clear() -> None` (vacía la caché y crea un lock nuevo). `async def api.chat._build_snapshot_or_raise()` y `async def api.chat._build_grounding()`.

- [ ] **Paso 1: crear el worktree de jax-platform**

```bash
git -C /home/fruiz/jax-platform fetch origin
git -C /home/fruiz/jax-platform worktree add -b feat/gobernanza-catalogo-db /home/fruiz/worktrees/jax-platform-gobernanza origin/master
ln -s /home/fruiz/jax-platform/frontend/node_modules /home/fruiz/worktrees/jax-platform-gobernanza/frontend/node_modules
git -C /home/fruiz/worktrees/jax-platform-gobernanza log --oneline -1   # anotar el sha base
```

- [ ] **Paso 2: medir el "antes" (lo usa la Tarea 6)**

Crear `$SCRATCH/medir_validation_context.py`. No se commitea: es una herramienta de medición y su resultado va a `DEUDA.md`.

```python
"""Latencia en proceso de governance_context.validation_context(): caché fría
(recarga completa) y caliente (hit). Sirve para la versión sync (master) y la
async (rama). Solo lectura: from_db() hace SELECT; el sello solo se statea."""
import asyncio
import inspect
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import governance_context as gc  # noqa: E402

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
    print(f"N={N} frio_ms p50={pct(frio, 50):.3f} p95={pct(frio, 95):.3f} max={max(frio):.3f}")
    print(f"N={N} caliente_us p50={pct(caliente, 50):.2f} p95={pct(caliente, 95):.2f} max={max(caliente):.2f}")


asyncio.run(main())
```

Correrlo contra el worktree todavía sin cambios. El validador viejo (dos argumentos) está en `/home/fruiz/jax`, que solo se importa, sin escribir bytecode:

```bash
cd /home/fruiz/worktrees/jax-platform-gobernanza/backend && pwd && set -a && . /etc/jax/.env && set +a && \
  PYTHONDONTWRITEBYTECODE=1 JAX_REPO_PATH=/home/fruiz/jax \
  /home/fruiz/jax-platform/backend/.venv/bin/python "$SCRATCH/medir_validation_context.py" | tee "$SCRATCH/latencia-antes.txt"
```

- [ ] **Paso 3: escribir el test nuevo (rojo)**

Crear `backend/tests/test_governance_context_catalogo.py`:

```python
"""
governance_context con el catálogo de la DB (tanda A, spec 2026-09-14 §3.2).

Hasta hoy el contexto de gobernanza armaba el catálogo de capabilities desde
`las_manos/config.toml`, vacío desde el Bloque 3: el resolver de
CAPABILITY_AVAILABLE nunca veía una capability de la DB. Ahora
`validation_context()` es async, carga `await MotorCatalog.from_db()` y cachea
con una clave que suma el mtime del sello de facet_resolver (lo estampan las
migraciones y el admin de motores y capabilities) a los mtimes de los tres
archivos de config.

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


def _catalogo(*nombres):
    return MotorCatalog({"capabilities": {n: {"allowed_motors": ["kimi"]} for n in nombres}})


@pytest.fixture
def from_db(monkeypatch):
    """from_db falso que devuelve un catálogo NUEVO en cada llamada (así una
    recarga se distingue de un hit por identidad) y cuenta las llamadas."""
    fake = AsyncMock(side_effect=lambda: _catalogo("generate"))
    monkeypatch.setattr(MotorCatalog, "from_db", fake)
    governance_context.validation_context.cache_clear()
    yield fake
    governance_context.validation_context.cache_clear()


def test_el_contexto_trae_el_catalogo_de_la_db(from_db):
    ctx, _, _ = asyncio.run(governance_context.validation_context())
    assert ctx.catalog.get_capability("generate") is not None
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
        return _catalogo("generate")

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


def _acreditado_a_mano():
    return governance_grounding.Accreditation(
        authority="OBSERVADO", provenance_ref="tool_result:sha256:" + "0" * 64,
        evidence_pointer_raw="/capabilities/0", outcome="ACCREDITED",
        detail="acreditado a mano: ejercita el resolver, no el snapshot",
    )


def _claim(nombre):
    return governance_claims.Claim(
        predicate="CAPABILITY_AVAILABLE", args={"name": nombre, "mode": "read_only"},
        authority="OBSERVADO", provenance_ref="tool_result:sha256:" + "0" * 64,
        evidence_pointer="/capabilities/0", scope="mesa_web",
    )


def test_el_resolver_con_el_contexto_de_produccion_da_valid_para_una_capability_de_la_db(from_db):
    """El contexto que arma jax-platform, pasado al validador real: una
    capability que solo existe en la DB resuelve VALID por la rama in_catalog.
    La acreditación va armada a mano porque el snapshot lista solo `ops`
    (decisión §2 del spec); ver el TRIPWIRE de la Mesa en
    test_tripwires_catalogo_db.py."""
    ctx, predicates, _ = asyncio.run(governance_context.validation_context())
    v = governance_validator.validate(_claim("generate"), predicates, ctx, accreditation=_acreditado_a_mano())
    assert v.status == "VALID", v.detail


def test_control_una_capability_que_no_esta_en_la_db_sigue_siendo_fact_mismatch(from_db):
    """CONTROL del test de arriba: mismo camino, nombre ausente del catálogo
    falso y de `ops` -> FACT_MISMATCH. El VALID de arriba sale del catálogo."""
    ctx, predicates, _ = asyncio.run(governance_context.validation_context())
    v = governance_validator.validate(
        _claim("totalmente_inventado_xyz"), predicates, ctx, accreditation=_acreditado_a_mano())
    assert v.status == "FACT_MISMATCH"


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


def test_con_la_db_real_el_contexto_trae_las_capabilities_sembradas(client):
    """Sin parches: from_db contra jax_memory_test migrada por el fixture
    `client`. `generate` la siembra db/migrations.py."""
    governance_context.validation_context.cache_clear()
    try:
        ctx, _, _ = client.portal.call(governance_context.validation_context)
    finally:
        governance_context.validation_context.cache_clear()
    assert ctx.catalog.get_capability("generate") is not None
```

- [ ] **Paso 4: correr y ver el rojo**

```bash
cd /home/fruiz/worktrees/jax-platform-gobernanza/backend && pwd && JAX_REPO_PATH=/home/fruiz/worktrees/jax-gobernanza-catalogo \
  /home/fruiz/jax-platform/backend/.venv/bin/python -m pytest tests/test_governance_context_catalogo.py -v 2>&1 | tail -20
```

Esperado: los 11 en ROJO. Con el validador de la Tarea 1 y el `governance_context` viejo, la reconstrucción falla con `TypeError: load_validation_context() missing 1 required positional argument: 'catalog'`: el contexto todavía no carga el catálogo. En los tests de la DB caída, `pytest.raises(ConnectionRefusedError)` ve un `TypeError`, y `asyncio.run(...)` sobre una tupla da `ValueError: a coroutine was expected`. Un rojo por otra razón (fixture, import) se corrige primero.

- [ ] **Paso 5: implementar `governance_context.py`**

Reemplazar, desde `from __future__ import annotations` hasta el final del archivo, conservando el docstring actual y agregándole esta sección al final (antes del `"""` de cierre):

```text
Catálogo de la DB (tanda A, 2026-09-14). Desde el Bloque 3 las capabilities
viven en la DB; hasta hoy este contexto las armaba del TOML vacío y el
resolver de CAPABILITY_AVAILABLE nunca las veía. Ahora:

  * `validation_context()` es ASYNC: el catálogo sale de
    `await MotorCatalog.from_db()` (aiomysql). Los YAML/TOML se leen en
    `asyncio.to_thread`: nada bloqueante en el camino del turno.
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
    cambios del catálogo entran al reiniciar. Declarado, no supuesto: en
    hall9000 el sello existe (/srv/jax-data/facet-cache-seal, verificado
    2026-09-14) y la Tarea 8 del plan lo vuelve a verificar al desplegar.
  * UNA recarga a la vez (`asyncio.Lock` con doble chequeo): N turnos que ven
    el sello nuevo a la par no disparan N consultas.
  * FALLA VISIBLE: si from_db() lanza, la excepción sube y la caché queda
    como estaba (no se sirve: la clave ya no coincide). En el chat,
    `_build_grounding` la convierte en SnapshotError
    (grounding_snapshot_sha256='ERROR'); la validación en sombra no escribe
    veredictos. Servir un catálogo vacío o viejo repetiría el falso negativo
    en silencio, o daría VALID a una capability revocada (P10).
  * Costo: un stat más por turno (~1 us); la recarga (from_db ~1 ms, medido
    en LAS MANOS el 2026-09-12, más la reconstrucción de config) solo cuando
    cambió algo. Latencia medida antes/después: DEUDA.md de jax.
  * Si la DB cuelga en vez de rechazar, la recarga espera lo que espere
    aiomysql.connect -- igual que el resto del turno, que ya depende de la
    misma DB por el pool.
```

Código:

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

- [ ] **Paso 6: llamadores con `await`**

`backend/api/chat.py`, líneas 629-647:

```python
async def _build_snapshot_or_raise() -> "governance_grounding.Snapshot":
    """Separado de _build_grounding para poder parchearlo en tests."""
    ctx, _, _ = await _governance_context()
    return governance_grounding.build_snapshot(ctx)


async def _build_grounding() -> "governance_grounding.Snapshot | governance_grounding.SnapshotError":
```

(el docstring y el cuerpo quedan igual, salvo `return await _build_snapshot_or_raise()` dentro del `try`). Línea 1115: `grounding = await _build_grounding()`.

`backend/shadow_validation.py:315`: `ctx, predicates, term_categories = await _validation_context()`.

Buscar otros llamadores sync:

```bash
git -C /home/fruiz/worktrees/jax-platform-gobernanza grep -n "validation_context()\|_build_grounding()\|_build_snapshot_or_raise()" -- backend
```

- [ ] **Paso 7: adaptar los tests existentes**

- `test_grounding_config_revalidation.py`:
  - En el fixture `repo_copia`, agregar `monkeypatch` a la firma y, antes del `cache_clear()`, esta línea: `monkeypatch.setattr(governance_context.governance_validator.MotorCatalog, "from_db", AsyncMock(side_effect=lambda: governance_context.governance_validator.MotorCatalog({})))`. Importar `AsyncMock` de `unittest.mock` y `asyncio`. Así los tests que hoy son puros siguen puros: sin eso, en el job sin DB `from_db` intenta `aiomysql.connect` y la Regla 2 de conftest solo cubre `create_pool`.
  - Cada `governance_context.validation_context()` pasa a `asyncio.run(governance_context.validation_context())`, y `chat._build_grounding()` a `asyncio.run(chat._build_grounding())`.
  - En `test_sin_cambios_en_disco_el_contexto_no_se_reconstruye`, las dos llamadas van en la MISMA corrida: `primero, segundo = asyncio.run(_dos())`, con `async def _dos(): return (await governance_context.validation_context(), await governance_context.validation_context())`. Se afirma `segundo is primero`.
- `test_shadow_origin.py:128` y `:177`: `ctx, _, _ = client.portal.call(validation_context)`.
- `test_shadow_validation.py:50` (`_grounding()`) y `test_shadow_validation_grounding.py:89` (`_snapshot()`): `ctx, _, _ = asyncio.run(governance_context.validation_context())`, con `import asyncio`.
- `test_chat_grounding_wiring.py`: sin cambios. `patch.object` sobre un `async def` crea un `AsyncMock`, y el `side_effect` que lanza se dispara al hacer `await`. Se confirma en el Paso 8.
- `test_shadow_validation.py::..._context_load_fails_before_the_insert`: sin cambios, por la misma razón (`patch.object(shadow_validation, "_validation_context", side_effect=RuntimeError)`).

- [ ] **Paso 8: ver el verde, en las dos modalidades de CI**

```bash
cd /home/fruiz/worktrees/jax-platform-gobernanza/backend && pwd && export JAX_REPO_PATH=/home/fruiz/worktrees/jax-gobernanza-catalogo && \
  /home/fruiz/jax-platform/backend/.venv/bin/python -m pytest -q -rs 2>&1 | tail -5 && \
  JAX_CI_NO_DB=1 /home/fruiz/jax-platform/backend/.venv/bin/python -m pytest -q -rs 2>&1 | tail -5
```

Esperado: 0 failed y 0 errors en las dos. Con DB, passed = base + 11. Sin DB, passed = base + 10: el último test pide `client` y se saltea. La base es lo que dan los mismos comandos sobre `origin/master`, que se mide ANTES (en otro worktree temporal o anotado del run de CI de master). Así el delta queda medido y no supuesto.

**Mutación obligatoria** (se revierte después; se anota en el PR):
- (a) Sacar el doble chequeo dentro del lock: `test_n_turnos_a_la_vez_disparan_una_sola_recarga` cae con `await_count == 20`.
- (b) Sacar `facet_resolver._seal_mtime()` de `_stamp()`: cae `test_tocar_el_sello_fuerza_la_recarga`.
- (c) Envolver `_build()` en un `try/except` que devuelva el `_cache` viejo: cae `test_db_caida_en_caliente_no_sirve_el_catalogo_viejo`.

- [ ] **Paso 9: pisos de jax-platform**

En `.github/workflows/policy.yml`, leer el `PISO_PASSED` y el `JAX_CI_MIN_PASSED` ACTUALES y subirlos por el delta medido en el Paso 8 (esperado +11 y +10), con comentario en el estilo del archivo:

```text
# Subido de <actual> a <actual+11> (2026-09-14, tanda A, catálogo de la DB):
# test_governance_context_catalogo.py -- 10 puros (catálogo de la DB, hit sin
# recarga, recarga por sello, DB caída en frío y en caliente, una recarga con
# 20 turnos, resolver VALID + CONTROL, SnapshotError) y 1 con `client` (from_db
# real contra la base migrada). Vistos en rojo antes del arreglo. Medido: <n> / <skips>.
```

Para `JAX_CI_MIN_PASSED`, el mismo comentario con "+10 puros; el de `client` se saltea".

- [ ] **Paso 10: commit**

```bash
git -C /home/fruiz/worktrees/jax-platform-gobernanza add backend/governance_context.py backend/api/chat.py backend/shadow_validation.py backend/tests/test_governance_context_catalogo.py backend/tests/test_grounding_config_revalidation.py backend/tests/test_shadow_origin.py backend/tests/test_shadow_validation.py backend/tests/test_shadow_validation_grounding.py .github/workflows/policy.yml
git -C /home/fruiz/worktrees/jax-platform-gobernanza commit -m "$(cat <<'EOF'
fix(gobernanza): el contexto de validación usa el catálogo de la DB

validation_context() pasa a async: carga MotorCatalog.from_db(), cachea
con los mtimes de config + el sello de facet_resolver, recarga una sola
vez bajo asyncio.Lock y, si la DB falla, la excepción sube (SnapshotError
en el chat) en vez de servir un catálogo vacío o viejo. chat y shadow
validation hacen await. Requiere el validador de jax con catálogo recibido.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LNXrHCwsZAVfrQj56maFHB
EOF
)"
```

---

### Tarea 4: jax-platform manda `invoked_by = "plataforma"`

**Archivos:**
- Modificar: `backend/api/pipelines.py` (constante nueva; `create_pipeline` `:95-96`; `resume_pipeline` `:161`)
- Modificar: `backend/tests/test_pipelines_identity_injection.py`
- Modificar: `frontend/src/components/BottomBar/PipelineModal.jsx:172` y `:187`
- Modificar: `frontend/src/components/BottomBar/PipelineModal.test.jsx` (test nuevo)
- Modificar: `.github/workflows/policy.yml` (piso de vitest)

**Interfaces:**
- Consume: el contrato de la Tarea 2, `invoked_by == "plataforma"`.
- Produce: `api.pipelines.INVOKED_BY_PLATAFORMA = "plataforma"`.

- [ ] **Paso 1: tests (rojos)**

En `test_pipelines_identity_injection.py`, al final de `test_create_pipeline_inyecta_identidad_real`:

```python
    # tanda A (2026-09-14): invoked_by es un ROL que pone el backend, igual que
    # la identidad; lo que mande el cliente se pisa.
    assert captured["json"]["invoked_by"] == "plataforma"
```

En `test_resume_pipeline_inyecta_identidad_real`, reemplazar el comentario viejo ("`"invoked_by": "Fernando"` is kept as the human-readable label...") y agregar:

```python
    # tanda A (2026-09-14): el backend declara el rol "plataforma" (Jacobs
    # exige ese rol para reanudar); la identidad real va en user_id/tenant_id.
    assert captured["json"]["invoked_by"] == "plataforma"
```

Actualizar el docstring del módulo: `(hoy "Fernando" fijo)` pasa a `(hasta 2026-09-14, "Fernando" fijo; ahora el rol "plataforma", puesto por el backend)`.

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

Antes de correrlo, confirmar en el `.test.jsx` cómo seleccionan una faceta los tests existentes del layout paralelo (`grep -n "checkbox\|fireEvent.click" PipelineModal.test.jsx`) y usar ese mismo selector. Si no es un checkbox, se reemplaza la línea por el que usan ellos.

- [ ] **Paso 2: ver el rojo**

```bash
cd /home/fruiz/worktrees/jax-platform-gobernanza/backend && pwd && JAX_REPO_PATH=/home/fruiz/worktrees/jax-gobernanza-catalogo \
  /home/fruiz/jax-platform/backend/.venv/bin/python -m pytest tests/test_pipelines_identity_injection.py -v 2>&1 | tail -6
cd /home/fruiz/worktrees/jax-platform-gobernanza/frontend && pwd && PATH=/home/fruiz/.nvm/versions/node/v24.16.0/bin:$PATH npx vitest run src/components/BottomBar/PipelineModal.test.jsx 2>&1 | tail -12
```

Esperado: `AssertionError: assert 'cliente-mintiendo' == 'plataforma'` (create), `assert 'Fernando' == 'plataforma'` (resume) y, en vitest, `expected { … } to not have property "invoked_by"`.

- [ ] **Paso 3: implementar**

`backend/api/pipelines.py`, bajo `JACOBS_PIPELINE_TIMEOUT`:

```python
# `invoked_by` es el ROL de quien le pide a Jacobs, no una persona (tanda A,
# 2026-09-14, decisión de Fernando): "plataforma" = pedido de jax-platform en
# nombre de un usuario autenticado. La identidad viaja en user_id/tenant_id y
# la pone este backend; el rol también -- nunca se toma del cliente.
INVOKED_BY_PLATAFORMA = "plataforma"
```

En `create_pipeline`, junto a `user_id`/`tenant_id`: `body["invoked_by"] = INVOKED_BY_PLATAFORMA`. En `resume_pipeline`: `json={"invoked_by": INVOKED_BY_PLATAFORMA, "user_id": user.user_id, "tenant_id": user.tenant_id}`.

`PipelineModal.jsx`: borrar las dos líneas `invoked_by: 'Fernando',`.

- [ ] **Paso 4: verde**

Los mismos dos comandos del Paso 2, más la suite completa de vitest:

```bash
cd /home/fruiz/worktrees/jax-platform-gobernanza/frontend && pwd && PATH=/home/fruiz/.nvm/versions/node/v24.16.0/bin:$PATH npx vitest run --reporter=default --reporter=json --outputFile="$SCRATCH/vitest.json" >/dev/null; node -e 'const r=require(process.argv[1]);console.log(r.numPassedTests,r.numFailedTests)' "$SCRATCH/vitest.json"
```

Esperado: `126 0` (125 + 1, o el piso actual + 1). Los tests de backend de este archivo no cambian de cantidad: se agregaron aserciones a tests que ya existían, y se declara así en el PR.

- [ ] **Paso 5: piso de vitest**

En el job `frontend-tests`, subir `r.numPassedTests !== 125` y su mensaje al valor medido, con este comentario:

```js
            // 125 -> 126 el 2026-09-14 (tanda A): PipelineModal.test.jsx (+1) --
            // el modal no manda invoked_by en ninguna de las dos formas (lo pone
            // el backend como el rol "plataforma"). Visto en rojo antes.
```

- [ ] **Paso 6: commit**

```bash
git -C /home/fruiz/worktrees/jax-platform-gobernanza add backend/api/pipelines.py backend/tests/test_pipelines_identity_injection.py frontend/src/components/BottomBar/PipelineModal.jsx frontend/src/components/BottomBar/PipelineModal.test.jsx .github/workflows/policy.yml
git -C /home/fruiz/worktrees/jax-platform-gobernanza commit -m "$(cat <<'EOF'
fix(pipelines): invoked_by es el rol "plataforma" y lo pone el backend

create y resume mandan invoked_by="plataforma" pisando lo que mande el
cliente, como ya se hacía con user_id/tenant_id; el modal deja de mandarlo.
vitest 125 -> 126.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LNXrHCwsZAVfrQj56maFHB
EOF
)"
```

---

### Tarea 5: tripwires del catálogo en jax-platform

**Archivos:**
- Crear: `backend/tests/test_tripwires_catalogo_db.py`
- Modificar: `.github/workflows/policy.yml` (`PISO_PASSED`, `JAX_CI_MIN_PASSED`)

**Interfaces:**
- Consume: `await governance_context.validation_context()` (Tarea 3) y `grounding.build_snapshot` / `accredit` (jax, sin cambios).

- [ ] **Paso 1: escribir los tests**

```python
"""
TRIPWIRES del catálogo de capabilities (tanda A, 2026-09-14).

1. `ops` del TOML y `capability` de la DB no comparten nombres. Con el catálogo
   real en el resolver, un nombre repetido da SOURCE_CONFLICT en producción
   (dos fuentes de verdad para el mismo nombre). Medido el 2026-09-14: 11 ops,
   17 capabilities en producción, intersección vacía. Este test lo mira sobre
   el MISMO objeto que ve el resolver: el ValidationContext de
   governance_context, con el TOML del clon de jax y la DB migrada.
   Alcance declarado: ve las capabilities que SIEMBRAN las migraciones; una
   fila agregada a mano por el admin en producción no pasa por acá (la
   Tarea 8 del plan la mide contra producción al desplegar).
2. En la Mesa, un claim sobre una capability que solo existe en la DB NO llega
   al resolver: el snapshot lista solo `ops` (decisión §2 del spec) y la
   acreditación lo corta antes. Si alguien suma el catálogo al snapshot, este
   test se pone ROJO: es la señal de revisar SP3 §3.3, SECTION_PREDICATE y la
   línea base de SP4.
"""
from __future__ import annotations

import asyncio
import shutil
from unittest.mock import AsyncMock

import claims as governance_claims
import governance_context
import grounding as governance_grounding
import validator as governance_validator


def _en_ambos(ctx):
    return sorted(n for n in ctx.ops if ctx.catalog.get_capability(n) is not None)


def test_tripwire_ops_del_toml_y_capabilities_de_la_db_no_comparten_nombres(client):
    """TRIPWIRE 1. Ver docstring del módulo."""
    governance_context.validation_context.cache_clear()
    try:
        ctx, _, _ = client.portal.call(governance_context.validation_context)
    finally:
        governance_context.validation_context.cache_clear()
    assert ctx.ops, "ops vacío: el tripwire no estaría mirando nada"
    assert ctx.catalog.get_capability("generate") is not None, (
        "catálogo sin la capability sembrada: el tripwire no estaría mirando la DB")
    assert _en_ambos(ctx) == [], (
        f"{_en_ambos(ctx)} está en ops del TOML Y en capability de la DB: el resolver "
        "de CAPABILITY_AVAILABLE va a dar SOURCE_CONFLICT. Renombrar uno de los dos.")


def test_control_el_tripwire_ve_un_nombre_repetido(client, tmp_path):
    """CONTROL del TRIPWIRE 1: con un config.toml que agrega `[ops.generate]`, la
    intersección da ['generate'] y el resolver da SOURCE_CONFLICT. Sin este
    control el tripwire podría estar verde por no ver nada."""
    destino = tmp_path / "jax"
    (destino / "las_manos").mkdir(parents=True)
    original = governance_context.JAX_REPO / "las_manos" / "config.toml"
    shutil.copy(original, destino / "las_manos" / "config.toml")
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


def test_tripwire_en_la_mesa_un_claim_sobre_capability_de_la_db_no_llega_al_resolver(monkeypatch):
    """TRIPWIRE 2. Puro: from_db parcheado con un catálogo que tiene `generate`."""
    MotorCatalog = governance_validator.MotorCatalog
    monkeypatch.setattr(MotorCatalog, "from_db", AsyncMock(
        side_effect=lambda: MotorCatalog({"capabilities": {"generate": {}}})))
    governance_context.validation_context.cache_clear()
    try:
        ctx, predicates, _ = asyncio.run(governance_context.validation_context())
    finally:
        governance_context.validation_context.cache_clear()
    snapshot = governance_grounding.build_snapshot(ctx)
    raw = {"predicate": "CAPABILITY_AVAILABLE", "args": {"name": "generate", "mode": "read_only"},
           "evidence_pointer": "/capabilities/0"}
    acc = governance_grounding.accredit(raw, snapshot)
    claim = governance_claims.Claim(
        predicate=raw["predicate"], args=governance_grounding.normalize_args(raw["args"]),
        authority=acc.authority, provenance_ref=acc.provenance_ref,
        evidence_pointer=raw["evidence_pointer"], scope="mesa_web")
    verdict = governance_validator.validate(claim, predicates, ctx, accreditation=acc)
    assert verdict.status == "FACT_NOT_IN_SNAPSHOT", (
        f"dio {verdict.status}: el snapshot ya incluye capabilities de la DB. Revisar "
        "SP3 §3.3, grounding.SECTION_PREDICATE y la línea base de SP4, y reemplazar este test.")
```

- [ ] **Paso 2: verde, y ROJO por mutación (un tripwire se valida rompiéndolo)**

```bash
cd /home/fruiz/worktrees/jax-platform-gobernanza/backend && pwd && JAX_REPO_PATH=/home/fruiz/worktrees/jax-gobernanza-catalogo \
  /home/fruiz/jax-platform/backend/.venv/bin/python -m pytest tests/test_tripwires_catalogo_db.py -v 2>&1 | tail -6
```

Esperado: `3 passed`. Mutaciones, cada una revertida después y anotada en el PR:
- (a) En la copia de jax del worktree, agregar `[ops.generate]` a `las_manos/config.toml` sin commitear: el TRIPWIRE 1 cae con `['generate'] está en ops del TOML Y en capability de la DB`. Revertir con `git -C /home/fruiz/worktrees/jax-gobernanza-catalogo checkout las_manos/config.toml`.
- (b) En `grounding.build_snapshot` del worktree de jax, sumar temporalmente los nombres de `ctx.catalog._capabilities` a `ops`: el TRIPWIRE 2 cae con `dio VALID`. Revertir.

- [ ] **Paso 3: pisos**

Leer los valores actuales. `PISO_PASSED` +3 (2 con `client` + 1 puro) y `JAX_CI_MIN_PASSED` +1, con comentario ("tripwires del catálogo: intersección ops∩DB vacía y su CONTROL con `client`; la Mesa no llega al resolver, puro. Validados por mutación."). Medir con los dos comandos del Paso 8 de la Tarea 3.

- [ ] **Paso 4: commit**

```bash
git -C /home/fruiz/worktrees/jax-platform-gobernanza add backend/tests/test_tripwires_catalogo_db.py .github/workflows/policy.yml
git -C /home/fruiz/worktrees/jax-platform-gobernanza commit -m "$(cat <<'EOF'
test(gobernanza): tripwires del catálogo de capabilities

ops del TOML y capability de la DB no comparten nombres (con su control),
y en la Mesa un claim sobre una capability de la DB sigue sin llegar al
resolver mientras el snapshot liste solo ops. Validados por mutación.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LNXrHCwsZAVfrQj56maFHB
EOF
)"
```

---

### Tarea 6: latencia de `validation_context()`, antes y después

**Archivos:** ninguno en los repos. Los números van a `DEUDA.md` en la Tarea 8.

- [ ] **Paso 1: medir el "después"** con el mismo script del Paso 2 de la Tarea 3, contra la rama:

```bash
cd /home/fruiz/worktrees/jax-platform-gobernanza/backend && pwd && set -a && . /etc/jax/.env && set +a && \
  PYTHONDONTWRITEBYTECODE=1 JAX_REPO_PATH=/home/fruiz/worktrees/jax-gobernanza-catalogo \
  /home/fruiz/jax-platform/backend/.venv/bin/python "$SCRATCH/medir_validation_context.py" | tee "$SCRATCH/latencia-despues.txt"
```

La recarga "fría" del después incluye `from_db()` contra `jax_memory` (solo SELECT) y `asyncio.to_thread`. El "caliente" suma un `stat` del sello. El sello real solo se statea; el script no lo escribe.

- [ ] **Paso 2: juzgar el resultado.** Caliente: p95 del después menor a 50 µs. Frío: p95 menor a 20 ms. Si se pasa, no hay GO. Se busca la causa (¿el hilo? ¿el connect?) y se mide de nuevo. No se sube el umbral para que pase.
- [ ] **Paso 3: anotar** en el PR de jax-platform: antes y después, `N`, fecha, máquina (hall9000), y lo que se declaró no medido: sin carga sobre `/api/chat` (llama al modelo); crear y reanudar pipelines no cambian de forma (un literal distinto).

---

### Tarea 7: PRs, gate por `headSha` y orden de merge

**Acoplamiento declarado.** El CI de jax-platform clona `jax` master (`git clone --depth=1 .../Jax.git /tmp/jax`). Hasta que el PR de jax esté mergeado, los tests de gobernanza del PR de jax-platform llaman al validador viejo y fallan. En la otra dirección: desde que el PR de jax se mergea hasta que se mergea el de jax-platform, **cualquier** run de CI de jax-platform (el de la etapa 2 incluido) falla en esos tests, porque `master` todavía llama con dos argumentos. Por eso se mergean uno detrás del otro, y la etapa 2 rebasa después.

- [ ] **Paso 1: verificar el estado remoto antes de abrir** (lección "verificar estado remoto antes de un PR")

```bash
git -C /home/fruiz/worktrees/jax-gobernanza-catalogo fetch origin && git -C /home/fruiz/worktrees/jax-gobernanza-catalogo log --oneline origin/master -3
git -C /home/fruiz/worktrees/jax-platform-gobernanza fetch origin && git -C /home/fruiz/worktrees/jax-platform-gobernanza log --oneline origin/master -3
gh pr list --repo fjruizhn/Jax --state open; gh pr list --repo fjruizhn/jax-platform --state open
```

Si `master` avanzó, se rebasa y, si hay conflicto en un piso, se suman los dos deltas. Antes de pushear se vuelven a correr los comandos de verde de las Tareas 1–5.

- [ ] **Paso 2: PR de jax.** `git -C … push -u origin feat/gobernanza-catalogo-db`, y `gh pr create --repo fjruizhn/Jax`. El cuerpo lleva: resumen, la decisión de Fernando del Paso 0.1 (con fecha), los hallazgos de la Tarea 0, los pisos (80→81, 143→151), lo que se vio en rojo, y que **acompaña a jax-platform#<n>** (se despliegan juntos). Termina con las líneas de atribución de las Restricciones globales.
- [ ] **Paso 3: gate de jax sobre el `headSha`**

```bash
SHA=$(git -C /home/fruiz/worktrees/jax-gobernanza-catalogo rev-parse HEAD)
gh pr view <n> --repo fjruizhn/Jax --json headRefOid -q .headRefOid   # tiene que ser == $SHA
gh run list --repo fjruizhn/Jax --commit "$SHA" --json name,status,conclusion
```

Se espera que todos los workflows del sha estén `completed/success`, en particular `governance`, `tests-puros`, `mirror-sync` y `no-fail-open-except`. Si alguno está en rojo, no se sigue.

- [ ] **Paso 4: PR de jax-platform** (push y `gh pr create --repo fjruizhn/jax-platform`), con la latencia de la Tarea 6, las mutaciones de las Tareas 3 y 5 y "requiere Jax#<n> mergeado primero". Su CI va a estar rojo en gobernanza hasta el merge de jax, y así se declara en el cuerpo.
- [ ] **Paso 5: GO de Fernando** para mergear los dos y desplegar (Tarea 8). Se le anuncian el acoplamiento de arriba y la ventana de segundos del despliegue.
- [ ] **Paso 6: merge de jax** (`gh pr merge <n> --repo fjruizhn/Jax --merge`, solo si el gate del Paso 3 sigue verde sobre el mismo sha). Enseguida, `gh run rerun <run-id> --repo fjruizhn/jax-platform` sobre el run del `headSha` del PR de jax-platform, para que clone el jax nuevo.
- [ ] **Paso 7: gate de jax-platform sobre su `headSha`** (mismo comando del Paso 3 contra `fjruizhn/jax-platform`: `frontend-tests`, `backend-tests-con-db`, `backend-tests-no-db`, `no-fail-open-except`, todo en success). Después, merge.
- [ ] **Paso 8: avisar al ejecutor de la etapa 2** (`/home/fruiz/worktrees/jax-platform-etapa2`) que `master` cambió los pisos y `governance_context`: rebasa él y suma deltas.

---

### Tarea 8: despliegue, verificación en vivo y cierre en DEUDA

- [ ] **Paso 1: anunciar la ventana a Fernando.** "Reinicio jax-platform y después jax-las-manos. Durante unos segundos, crear o reanudar un pipeline puede fallar (uno manda `plataforma` y el otro todavía espera `Fernando`)."
- [ ] **Paso 2: traer el código a los checkouts de producción** (lo único que se hace en ellos)

```bash
git -C /home/fruiz/jax status --short && git -C /home/fruiz/jax pull --ff-only && git -C /home/fruiz/jax log --oneline -1
git -C /home/fruiz/jax-platform status --short && git -C /home/fruiz/jax-platform pull --ff-only && git -C /home/fruiz/jax-platform log --oneline -1
```

Si `status --short` no está vacío, se para y se pregunta: no se pisa trabajo ajeno. Traer el código de jax no cambia a ningún proceso vivo: el validador ya está importado en memoria.

- [ ] **Paso 3: reiniciar, en orden, y medir**

```bash
ls -la /srv/jax-data/facet-cache-seal          # el sello existe (si no, parar: el caché no se invalidaría)
date -Is | tee "$SCRATCH/deploy-inicio.txt"
sudo systemctl restart jax-platform && sleep 3 && systemctl is-active jax-platform
sudo systemctl restart jax-las-manos && sleep 3 && systemctl is-active jax-las-manos
for u in jax-platform jax-las-manos; do p=$(systemctl show -p MainPID --value $u); echo "$u pid=$p cwd=$(readlink /proc/$p/cwd)"; done
curl -fsS http://127.0.0.1:8080/health; echo; curl -fsS http://127.0.0.1:7777/health; echo
```

Si alguno de los dos `/health` no existe con esa ruta, se usa el endpoint de salud del CONTEXT.md del repo, verificado antes de correr esto.

- [ ] **Paso 4: verificación en vivo 1, el resolver con el contexto de producción** (proceso aparte, solo lectura, con el código ya desplegado)

```bash
cd /home/fruiz/jax-platform/backend && pwd && set -a && . /etc/jax/.env && set +a && PYTHONDONTWRITEBYTECODE=1 \
/home/fruiz/jax-platform/backend/.venv/bin/python - <<'PY'
import asyncio, os, sys
sys.path.insert(0, os.getcwd())
import governance_context as gc, grounding as g, validator as v, claims as c

acc = g.Accreditation(authority="OBSERVADO", provenance_ref="tool_result:sha256:" + "0"*64,
                      evidence_pointer_raw="/capabilities/0", outcome="ACCREDITED", detail="verificación en vivo")
def claim(n): return c.Claim(predicate="CAPABILITY_AVAILABLE", args={"name": n, "mode": "read_only"},
                             authority="OBSERVADO", provenance_ref="x", evidence_pointer="/capabilities/0", scope="mesa_web")
async def main():
    ctx, preds, _ = await gc.validation_context()
    caps = sorted(ctx.catalog._capabilities)
    print("capabilities de la DB:", len(caps), caps)
    print("ops∩DB:", sorted(set(ctx.ops) & set(caps)))                              # esperado: []
    print("generate ->", v.validate(claim("generate"), preds, ctx, accreditation=acc).status)   # VALID
    print("inventada ->", v.validate(claim("totalmente_inventado_xyz"), preds, ctx, accreditation=acc).status)  # FACT_MISMATCH (control)
asyncio.run(main())
PY
```

- [ ] **Paso 5: verificación en vivo 2, un turno de sonda (`origin='probe'`)** que haga que la Mesa declare claims `CAPABILITY_AVAILABLE`. Token: `auth.jwt.create_access_token(user_id, tenant_id, role)` en el venv de producción, con el `user_id`, `tenant_id` y rol de la cuenta de Fernando. Esos datos salen de una consulta de solo lectura a la tabla de usuarios; los nombres de columna se verifican con `SHOW COLUMNS` antes. POST a `http://127.0.0.1:8080/api/chat` con `{"message": "¿Qué capabilities tiene hoy este sistema, de lectura y de escritura, incluidas generate y file_write?", "facet": "jekyll", "origin": "probe"}`. Después:

```sql
SELECT shadow_message_id, grounding_snapshot_sha256, validated_at FROM shadow_messages
 WHERE origin='probe' AND created_at >= '<deploy-inicio>' ORDER BY created_at DESC LIMIT 1;
SELECT predicate, status, authority, JSON_UNQUOTE(JSON_EXTRACT(args,'$.name')) FROM shadow_claim_verdicts
 WHERE shadow_message_id = '<id>' ORDER BY id;
```

Antes de correrla, verificar con `SHOW COLUMNS FROM shadow_messages` que las columnas `origin` y `created_at` existen con esos nombres. Esperado: `sha256` distinto de `'ERROR'` (el contexto async con el catálogo de la DB se construyó), `validated_at` no NULL, claims sobre `ops` con `OBSERVADO/VALID`, ninguno `SOURCE_CONFLICT`. Los claims sobre capabilities de la DB dan lo que fija el TRIPWIRE 2 (no llegan al resolver), salvo que el Paso 0.1 haya decidido otra cosa. Si el modelo no declaró ningún claim, se repite una vez con otra formulación y se anota.

- [ ] **Paso 6: verificación en vivo 3, crear y reanudar un pipeline con el rol nuevo.** Desde la UI, o con `POST /api/pipelines` y el token del Paso 5, se crea un pipeline `supervised` de objetivo mínimo. Se espera `interrupted` y se reanuda (`POST /api/pipelines/<id>/resume`). Luego se cancela, para no gastar de más. Verificar:

```sql
SELECT pipeline_id, invoked_by, user_id, tenant_id, status FROM jacobs_pipelines WHERE pipeline_id='<id>';   -- invoked_by='plataforma'
```

y en los eventos de Jacobs del pipeline, `PIPELINE_RESUMED` con `{"by": "plataforma"}`. Si la tabla de eventos no se llama como se espera, se busca con `SHOW TABLES LIKE 'jacobs%'`.

- [ ] **Paso 7: journal limpio**

```bash
journalctl -u jax-platform -u jax-las-manos --since "$(cat "$SCRATCH/deploy-inicio.txt")" -p warning --no-pager | tail -40
```

Se espera no ver trazas de `validation_context`, `from_db`, `SnapshotError` ni `invoked_by`. Cualquier aviso nuevo se explica o se trata como fallo.

- [ ] **Paso 8: frontend.** El backend ya pisa `invoked_by`, así que el bundle viejo, que manda `'Fernando'`, sigue funcionando. Aun así, el modal nuevo se publica con el procedimiento de despliegue del frontend registrado en la Biblioteca de jax-platform (su `CONTEXT.md`). Si no está documentado, **se pregunta a Fernando**, no se improvisa. Se verifica que `axioma-ia.io` sirva el `index-*.js` nuevo.
- [ ] **Paso 9: DEUDA.md (PR de docs en jax).** En el worktree de jax, rama nueva desde el `origin/master` actualizado, `docs/deuda-gobernanza-catalogo`:
  - Cerrar **"El resolver de `CAPABILITY_AVAILABLE` consulta un catálogo que el Bloque 3 vació"**: CERRADO 2026-09-14, con los PRs, lo medido en vivo (Paso 4) y el alcance real, que es latente en la Mesa hasta que el snapshot incluya el catálogo (TRIPWIRE 2 en jax-platform; decisión del Paso 0.1).
  - Cerrar **"`invoked_by` es un campo de AUTORIZACIÓN en Jacobs..."**: CERRADO 2026-09-14, rol `plataforma`, verificación del Paso 6, y que las filas viejas con "Fernando" quedan como historia.
  - Agregar la entrada de rendimiento (LAS CUATRO, punto 4), con fecha: latencia de `validation_context()`, fría y caliente, p50/p95/max antes y después (Tareas 3 y 6), `N`, máquina, "sin carga sobre /api/chat (llama al modelo), declarado".
  - Anotar el hallazgo de la Tarea 0 como HECHO con procedencia: 17 capabilities en producción (no 18: falta `validate`) y cero `FACT_MISMATCH` históricos en `CAPABILITY_AVAILABLE`.

  Commit con las líneas de atribución, PR, gate por `headSha`, merge y `git -C /home/fruiz/jax pull --ff-only` (solo docs, sin reinicio).
- [ ] **Paso 10: limpiar.** Cuando los PRs estén mergeados y lo medido esté en `DEUDA.md`: `git -C /home/fruiz/jax-platform worktree remove /home/fruiz/worktrees/jax-platform-gobernanza`, `git -C /home/fruiz/jax worktree remove /home/fruiz/worktrees/jax-gobernanza-catalogo`, y borrar los venvs y archivos del scratchpad.

---

## Autorrevisión (contra el spec)

- §3.1 (validador recibe el catálogo, resolver sin cambios) → Tarea 1.
- §3.2 (async, clave con el sello, lock, falla visible, `cache_clear`) → Tarea 3.
- §3.3 (Jacobs `plataforma`, jax-platform lo pone, el modal no lo manda, filas viejas intactas, fixtures sin cambio) → Tareas 2 y 4.
- §4 (LAS CUATRO) → Restricciones globales, Tarea 3 (docstring) y Tarea 6.
- §5 (tests; tripwire nuevo ops∩DB) → Tareas 1–5.
- §6 (despliegue en orden, ventana, verificación en vivo) → Tareas 7 y 8. La verificación "claim de la Mesa → VALID" se ajustó por el hallazgo de la Tarea 0 y queda sujeta al Paso 0.1.
- §7 (fuera de alcance) → respetado: el snapshot no cambia y no se verifica el "modo" de las capabilities de la DB.
- Nombres consistentes entre tareas: `load_validation_context(repo_root, allowlist, catalog)`, `validation_context()` async + `.cache_clear()`, `_build_grounding()` async, `INVOKER_PLATAFORMA` (jax), `INVOKED_BY_PLATAFORMA` (jax-platform), el literal `"plataforma"`.
