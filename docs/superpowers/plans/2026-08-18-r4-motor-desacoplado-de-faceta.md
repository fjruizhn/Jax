# R4 — Motor desacoplado de faceta: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que un motor se elija por la tarea (capability), no por el nombre de faceta que lo presenta — con el catálogo en DB, de modo que registrar un motor nuevo sea una fila, no un commit.

**Architecture:** Tres tablas nuevas (`motor`, `capability`, `capability_motor`) en la DB compartida `jax_memory`, dueño de schema `jax-platform/backend/db/migrations.py`. `~/jax/las_manos/motor_registry/catalog.py` las lee en runtime (mismo patrón que `facet_resolver.py` ya prueba del lado de la Mesa). `worker.py` generaliza su única función hardcodeada (`_call_kimi`) a un dispatcher por `motor.transport`, reusando la forma de `chat.py::_call_openai_compat`. `jacobs/executor.py` gana un campo `step.motor` separado de `step.facet`, que al venir vacío activa `MotorPolicy._resolve_motor()` — mecanismo que ya existe y hoy nunca corre.

**Tech Stack:** Python 3.14 (jax-platform backend, `.venv` propio), Python 3.14 (`~/jax/las_manos`, `.venv` propio), MariaDB 12.3 (`jax_memory`, compartida por ambos repos), aiomysql, FastAPI, React 19 (frontend jax-platform), pytest / `unittest.IsolatedAsyncioTestCase`.

**Spec:** `~/jax/docs/superpowers/specs/2026-08-18-r4-motor-desacoplado-de-faceta-design.md` — este plan lo implementa tarea por tarea; quien ejecute debe leer ambos.

## Global Constraints

- Migración única, dueña de schema: `jax-platform/backend/db/migrations.py`. `~/jax` solo lee.
- Todo `CREATE TABLE`/`ALTER` sigue el patrón idempotente ya establecido: guard `_table_exists`/`_column_exists` antes de ejecutar (ver `run_migrations()`, `jax-platform/backend/db/migrations.py:646-674`).
- `capability_motor.priority`: **menor gana primero** (0 = primer intento). `_resolve_motor()` no cambia — solo el orden en que la lista `allowed_motors` llega ya viene correcto (`ORDER BY priority ASC` en la query).
- Ningún motor nuevo requiere código para registrarse — solo filas en `motor`/`capability_motor`. El criterio de aceptación #4 (Tarea 8) es la prueba de esto.
- El form de Admin (Tarea 9) no empieza hasta que la Tarea 8 pase por INSERT directo.
- Comando (CLI viejo, `api/command.py`) y la unificación A7 (`ops.*` vs `capabilities.*`) están fuera de alcance — no se tocan en este plan.
- Cero credencial nueva: `openai`/`anthropic`/`gemini` ya tienen `provider`+`credential` activos (usados por Thot/Hyde/Hipatia). `ollama` ya tiene `auth_type='none'` sembrado.

---

### Task 1: Migración — tablas `motor`/`capability`/`capability_motor` + seed real

**Files:**
- Modify: `jax-platform/backend/db/migrations.py` (agregar DDL, `_TABLES`, función de seed, llamada en `run_migrations()`)
- Test: `jax-platform/backend/tests/test_motor_migrations.py` (nuevo)

**Interfaces:**
- Produces: tablas `motor(key, model_ref, transport, max_tokens, default_timeout_seconds, supports_reasoning, reasoning_default_visibility, sandbox_only, status)`, `capability(key, risk_level, sandbox_only, requires_human_gate, max_execution_minutes, max_recursion_depth, output_schema, fallback_motor, fallback_mode, allowed_callers, forbidden_paths)`, `capability_motor(capability_key, motor_key, priority)` — consumidas por Task 2.

- [ ] **Step 1: Escribir el test que falla — las 3 tablas existen con las columnas esperadas**

Archivo nuevo `jax-platform/backend/tests/test_motor_migrations.py`:

```python
"""Migración de las tablas motor/capability/capability_motor (R4 — motor
desacoplado de faceta). Corre contra jax_memory_test (ver conftest.py)."""


async def _table_columns(cur, table_name):
    await cur.execute(
        "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s",
        (table_name,),
    )
    rows = await cur.fetchall()
    return {r[0] for r in rows}


async def _get_columns(table_name):
    from db.connection import get_pool
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            return await _table_columns(cur, table_name)


def test_motor_table_has_expected_columns(client):
    cols = client.portal.call(_get_columns, "motor")
    expected = {
        "key", "model_ref", "transport", "max_tokens",
        "default_timeout_seconds", "supports_reasoning",
        "reasoning_default_visibility", "sandbox_only", "status",
    }
    assert expected.issubset(cols)


def test_capability_table_has_expected_columns(client):
    cols = client.portal.call(_get_columns, "capability")
    expected = {
        "key", "risk_level", "sandbox_only", "requires_human_gate",
        "max_execution_minutes", "max_recursion_depth", "output_schema",
        "fallback_motor", "fallback_mode", "allowed_callers", "forbidden_paths",
    }
    assert expected.issubset(cols)


def test_capability_motor_table_has_expected_columns(client):
    cols = client.portal.call(_get_columns, "capability_motor")
    expected = {"capability_key", "motor_key", "priority"}
    assert expected.issubset(cols)


async def _fetch_all(query, params=()):
    from db.connection import get_pool
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, params)
            return await cur.fetchall()


def test_seed_kimi_y_ada_como_motor(client):
    rows = client.portal.call(
        _fetch_all,
        "SELECT `key`, transport, max_tokens, sandbox_only FROM motor WHERE `key` IN ('kimi','ada')",
    )
    by_key = {r[0]: r for r in rows}
    assert set(by_key) == {"kimi", "ada"}
    assert by_key["kimi"][1] == "http_openai_compat"
    assert by_key["kimi"][2] == 8000
    assert by_key["ada"][1] == "http_openai_compat"


def test_seed_capability_motor_no_referencia_motores_inexistentes(client):
    """thot no es motor todavia (Task 8 lo crea) -- las filas de
    validate_consistency/critique que en config.toml apuntaban a "thot"
    deben quedar excluidas del seed, no romper la FK."""
    rows = client.portal.call(
        _fetch_all,
        "SELECT capability_key, motor_key FROM capability_motor WHERE motor_key = 'thot'",
    )
    assert rows == []


def test_seed_code_swarm_apunta_a_kimi_con_fallback_ada(client):
    rows = client.portal.call(
        _fetch_all,
        "SELECT motor_key, priority FROM capability_motor WHERE capability_key = 'code_swarm' ORDER BY priority",
    )
    assert [r[0] for r in rows] == ["kimi"]
    cap = client.portal.call(
        _fetch_all,
        "SELECT fallback_motor, fallback_mode, requires_human_gate FROM capability WHERE `key`='code_swarm'",
    )
    assert cap[0] == ("ada", "manual_only", 1)


def test_seed_generate_tiene_dos_motores_en_orden_kimi_luego_ada(client):
    """[capabilities.generate].allowed_motors = ["kimi", "ada"] en config.toml
    -- el orden es el criterio de _resolve_motor() (el primero habilitado
    gana), portado a priority 0/1."""
    rows = client.portal.call(
        _fetch_all,
        "SELECT motor_key, priority FROM capability_motor WHERE capability_key = 'generate' ORDER BY priority",
    )
    assert [r[0] for r in rows] == ["kimi", "ada"]
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/python -m pytest tests/test_motor_migrations.py -v`
Expected: FAIL — `motor`/`capability`/`capability_motor` no existen todavía (`1146, Table 'jax_memory_test.motor' doesn't exist` o similar).

- [ ] **Step 3: Agregar las 3 tablas DDL**

En `jax-platform/backend/db/migrations.py`, inmediatamente después de `CREATE_MODEL_BINDING_PROPOSAL` (antes de `_TABLES = [`):

```python
# R4 — motor desacoplado de faceta. Tres ejes separados: capability (que
# sabe hacer), transport (como se le habla, mismo enum que facet.transport),
# auth (via provider.auth_type, ya existente — ollama='none' ya sembrado).
# model_ref reusa la tabla `model` (context_window, pricing, deprecacion)
# en vez de duplicar esos campos por motor, mismo patron que
# facet_binding.model_ref.
CREATE_MOTOR = """
CREATE TABLE IF NOT EXISTS motor (
  `key` VARCHAR(50) NOT NULL PRIMARY KEY,
  model_ref INT NOT NULL,
  transport ENUM('http_openai_compat','http_gemini','motor_registry','ollama','subprocess') NOT NULL,
  max_tokens INT NULL,
  default_timeout_seconds INT NOT NULL DEFAULT 600,
  supports_reasoning BOOLEAN NOT NULL DEFAULT FALSE,
  reasoning_default_visibility ENUM('audit_only','visible') NOT NULL DEFAULT 'audit_only',
  sandbox_only BOOLEAN NOT NULL DEFAULT TRUE,
  status ENUM('active','disabled') NOT NULL DEFAULT 'active',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (model_ref) REFERENCES model(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# priority reemplaza el orden implicito de la lista allowed_motors de TOML.
# Convencion: menor priority gana primero (0 = primer intento) -- mismo
# sentido que "el primero de la lista" que _resolve_motor() ya usa.
CREATE_CAPABILITY = """
CREATE TABLE IF NOT EXISTS capability (
  `key` VARCHAR(50) NOT NULL PRIMARY KEY,
  risk_level ENUM('low','medium','high') NOT NULL,
  sandbox_only BOOLEAN NOT NULL DEFAULT TRUE,
  requires_human_gate BOOLEAN NOT NULL DEFAULT FALSE,
  max_execution_minutes INT NOT NULL,
  max_recursion_depth INT NOT NULL DEFAULT 0,
  output_schema VARCHAR(100) NULL,
  fallback_motor VARCHAR(50) NULL,
  fallback_mode ENUM('manual_only','auto') NULL,
  allowed_callers LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL CHECK (json_valid(allowed_callers)),
  forbidden_paths LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL CHECK (forbidden_paths IS NULL OR json_valid(forbidden_paths)),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (fallback_motor) REFERENCES motor(`key`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

CREATE_CAPABILITY_MOTOR = """
CREATE TABLE IF NOT EXISTS capability_motor (
  capability_key VARCHAR(50) NOT NULL,
  motor_key VARCHAR(50) NOT NULL,
  priority INT NOT NULL DEFAULT 0,
  PRIMARY KEY (capability_key, motor_key),
  FOREIGN KEY (capability_key) REFERENCES capability(`key`),
  FOREIGN KEY (motor_key) REFERENCES motor(`key`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""
```

Nota de estilo: el proyecto usa `longtext ... CHECK (json_valid(...))` para columnas JSON (ver `facet_binding.params` en el mismo archivo) en vez del tipo `JSON` nativo — se sigue ese patrón exacto para consistencia, no el `JSON` que el spec sugería.

- [ ] **Step 4: Sumar las 3 tablas a `_TABLES`**

En `_TABLES = [...]`, después de `("model_binding_proposal", CREATE_MODEL_BINDING_PROPOSAL),`:

```python
    ("motor", CREATE_MOTOR),                          # antes de capability (FK fallback_motor)
    ("capability", CREATE_CAPABILITY),                # antes de capability_motor (FK)
    ("capability_motor", CREATE_CAPABILITY_MOTOR),
```

- [ ] **Step 5: Seed real, portado de `~/jax/las_manos/config.toml`**

Después de `_fix_anthropic_sonnet_alias` (antes de `_table_exists`), agregar:

```python
# Portado de ~/jax/las_manos/config.toml [motors.*] (2026-08-18). model_ref
# se resuelve por SELECT en vez de hardcodear el id -- el AUTO_INCREMENT de
# `model` no es estable entre instalaciones.
_MOTOR_SEED = [
    # key,   provider_id, model_id,   transport,             max_tokens, timeout, reasoning, visibility,    sandbox
    ("kimi", "moonshot", "kimi-k3",   "http_openai_compat",  8000,       600,     True,      "audit_only",  True),
    ("ada",  "zhipu",    "glm-5.2",   "http_openai_compat",  8000,       600,     True,      "audit_only",  True),
]

# key, risk_level, sandbox_only, requires_human_gate, max_exec_min, max_recursion,
# output_schema, fallback_motor, fallback_mode, allowed_callers, forbidden_paths
_CAPABILITY_SEED = [
    ("code_swarm", "high", True, True, 30, 1, "code_swarm.v1", "ada", "manual_only",
     ["hyde", "ada", "kimi", "jacobs"], [".env", "secrets/", "private_keys/", "credentials/"]),
    ("refactor", "medium", True, False, 10, 0, "code_patch.v1", None, None,
     ["hyde", "ada", "jacobs"], None),
    ("architecture_review", "medium", True, False, 5, 0, "architecture_review.v1", None, None,
     ["hyde", "jacobs"], None),
    ("bug_hunt", "high", True, True, 15, 0, "bug_hunt.v1", None, None,
     ["hyde", "ada", "jacobs"], None),
    ("pipeline_analysis", "low", True, False, 15, 0, "analysis.v1", None, None,
     ["jacobs", "hyde"], None),
    ("implementation", "medium", True, False, 30, 0, "code_patch.v1", None, None,
     ["jacobs", "hyde"], [".env", "secrets/", "private_keys/", "credentials/"]),
    ("generate", "low", True, False, 15, 0, "generate.v1", None, None,
     ["jacobs", "hyde", "ada"], None),
    ("reason", "low", True, False, 15, 0, "reason.v1", None, None,
     ["jacobs", "hyde", "ada", "thot"], None),
    ("design", "low", True, False, 15, 0, "design.v1", None, None,
     ["jacobs", "hyde", "ada"], None),
    ("validate_consistency", "low", True, False, 15, 0, "validation.v1", None, None,
     ["jacobs", "hyde", "thot"], None),
    ("reconcile", "low", True, False, 15, 0, "reconcile.v1", None, None,
     ["jacobs", "hyde", "ada"], None),
    ("critique", "low", True, False, 15, 0, "critique.v1", None, None,
     ["jacobs", "hyde", "thot"], None),
]

# (capability_key, [motor_key, ...] en orden de prioridad). "thot" queda
# excluido a proposito de validate_consistency/critique -- no existe como
# motor todavia (Task 8 lo crea junto con esas 2 filas via INSERT directo,
# el criterio de aceptacion #4). Sin esto, la FK de capability_motor
# rompe el seed.
_CAPABILITY_MOTOR_SEED = [
    ("code_swarm", ["kimi"]),
    ("refactor", ["kimi"]),
    ("architecture_review", ["ada"]),
    ("bug_hunt", ["kimi"]),
    ("pipeline_analysis", ["kimi"]),
    ("implementation", ["kimi"]),
    ("generate", ["kimi", "ada"]),
    ("reason", ["ada", "kimi"]),
    ("design", ["ada", "kimi"]),
    ("validate_consistency", ["ada"]),  # "thot" excluido, ver nota arriba
    ("reconcile", ["ada", "kimi"]),
    ("critique", ["ada"]),              # "thot" excluido, ver nota arriba
]


async def _seed_motors_and_capabilities(cur) -> None:
    for key, provider_id, model_id, transport, max_tokens, timeout, reasoning, visibility, sandbox in _MOTOR_SEED:
        await cur.execute(
            "SELECT id FROM model WHERE provider_id=%s AND model_id=%s",
            (provider_id, model_id),
        )
        row = await cur.fetchone()
        if row is None:
            # model no sembrado todavia (orden de _seed_models_and_backfill) --
            # no romper el seed completo por un motor que se puede agregar despues.
            continue
        model_ref = row[0]
        await cur.execute(
            "INSERT IGNORE INTO motor "
            "(`key`, model_ref, transport, max_tokens, default_timeout_seconds, "
            " supports_reasoning, reasoning_default_visibility, sandbox_only) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (key, model_ref, transport, max_tokens, timeout, reasoning, visibility, sandbox),
        )

    for (key, risk_level, sandbox_only, gate, max_exec, max_rec, schema,
         fallback_motor, fallback_mode, callers, forbidden) in _CAPABILITY_SEED:
        await cur.execute(
            "INSERT IGNORE INTO capability "
            "(`key`, risk_level, sandbox_only, requires_human_gate, max_execution_minutes, "
            " max_recursion_depth, output_schema, fallback_motor, fallback_mode, "
            " allowed_callers, forbidden_paths) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (key, risk_level, sandbox_only, gate, max_exec, max_rec, schema,
             fallback_motor, fallback_mode, json.dumps(callers),
             json.dumps(forbidden) if forbidden is not None else None),
        )

    for capability_key, motor_keys in _CAPABILITY_MOTOR_SEED:
        for priority, motor_key in enumerate(motor_keys):
            await cur.execute(
                "SELECT 1 FROM motor WHERE `key`=%s",
                (motor_key,),
            )
            if await cur.fetchone() is None:
                continue  # motor no existe todavia -- no romper el seed (ver nota _CAPABILITY_MOTOR_SEED)
            await cur.execute(
                "INSERT IGNORE INTO capability_motor (capability_key, motor_key, priority) "
                "VALUES (%s, %s, %s)",
                (capability_key, motor_key, priority),
            )
```

Verificar el import de `json` al tope del archivo (`migrations.py`) — si no está, agregarlo.

- [ ] **Step 6: Llamar al seed en `run_migrations()`**

En `run_migrations()`, después de `await _fix_anthropic_sonnet_alias(cur)`:

```python
            await _seed_motors_and_capabilities(cur)
```

- [ ] **Step 7: Correr el test y verificar que pasa**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/python -m pytest tests/test_motor_migrations.py -v`
Expected: 6 passed.

- [ ] **Step 8: Correr la suite completa de migraciones/tests existentes, confirmar cero regresión**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/python -m pytest tests/ -q`
Expected: todos los tests existentes siguen en verde (mismo conteo que antes de este cambio + 6 nuevos).

- [ ] **Step 9: Aplicar la migración contra la DB real y confirmar**

Run: `sudo systemctl restart jax-platform.service` (aplica `run_migrations()` en el próximo `startup`, mismo patrón que toda migración anterior de esta sesión) y luego:
```bash
mysql -h 127.0.0.1 -P 3308 -u jax_user jax_memory -e "SELECT COUNT(*) FROM motor; SELECT COUNT(*) FROM capability; SELECT COUNT(*) FROM capability_motor;"
```
Expected: `motor`=2, `capability`=12, `capability_motor`=13 (suma de motores por capability en `_CAPABILITY_MOTOR_SEED`, sin las 2 filas de "thot" excluidas).

- [ ] **Step 10: Commit**

```bash
cd /home/fruiz/jax-platform
git add backend/db/migrations.py backend/tests/test_motor_migrations.py
git commit -m "feat(db): tablas motor/capability/capability_motor (R4) + seed real de config.toml

Tres ejes separados: capability (que sabe hacer), transport (mismo enum
que facet.transport), auth (via provider.auth_type ya existente). Seed
portado 1:1 de las_manos/config.toml -- kimi/ada como motor, las 12
capabilities reales. validate_consistency/critique excluyen su
referencia a 'thot' (no existe como motor todavia, Task 8 lo crea)."
```

---

### Task 2: `catalog.py` lee DB en vez de `config.toml`, wiring en `routes.py`

**Files:**
- Modify: `~/jax/las_manos/motor_registry/catalog.py` (agregar `transport`/`model_ref` a `MotorEntry`, agregar `MotorCatalog.from_db`)
- Modify: `~/jax/las_manos/motor_registry/routes.py` (reemplazar carga sync de TOML por carga async al startup)
- Test: `~/jax/las_manos/_catalog_from_db_test.py` (nuevo)

**Interfaces:**
- Consumes: pool de conexión DB — mismo patrón que `credential_resolver.py` (`~/jax/las_manos/credential_resolver.py`, variables `JAX_DB_HOST/PORT/USER/PASSWORD/NAME`).
- Produces: `MotorCatalog.from_db() -> MotorCatalog` (classmethod async, sin argumentos — abre su propia conexión, mismo patrón que `credential_resolver.py`). `MotorEntry` gana campos `transport: str`, `model_ref: int`, `provider_id: str` — el campo `model` ya existente se reusa para el `model_id` resuelto vía JOIN (no se agrega un `model_id` nuevo, ver Step 3). `provider_id` evita una query aparte en `worker.py`. `MotorCatalog(config: dict)` (constructor existente, dict-shaped) **conserva su firma** — `_load()` gana una línea (`transport=cfg.get("transport", "http_openai_compat")`) para que ese valor no se pierda cuando un test arma el dict a mano; los tests actuales que no incluyen `"transport"` en su fixture (`_worker_max_tokens_test.py`, hasta que Task 3 lo actualice) siguen funcionando igual, con el default de clase.

- [ ] **Step 1: Escribir el test que falla — `from_db` construye un catálogo real**

Archivo nuevo `~/jax/las_manos/_catalog_from_db_test.py`:

```python
#!/usr/bin/env python3
"""MotorCatalog.from_db() — lee motor/capability/capability_motor de la DB
compartida (jax_memory) en vez de config.toml (R4 — motor desacoplado de
faceta). Corre contra la DB real de desarrollo -- mismo criterio que
credential_resolver.py, sin mock de DB (aiomysql no tiene un modo in-memory
liviano establecido en este repo).

Corre desde /home/fruiz/jax/las_manos con:
  PYTHONPATH=/home/fruiz/jax/las_manos \
  /home/fruiz/jax/las_manos/.venv/bin/python \
  /home/fruiz/jax/las_manos/_catalog_from_db_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import unittest

from motor_registry.catalog import MotorCatalog


class CatalogFromDbTest(unittest.IsolatedAsyncioTestCase):
    async def test_from_db_carga_kimi_y_ada(self):
        catalog = await MotorCatalog.from_db()
        kimi = catalog.get_motor("kimi")
        assert kimi is not None, "kimi no cargó desde DB"
        assert kimi.transport == "http_openai_compat", kimi.transport
        assert kimi.provider_id == "moonshot", kimi.provider_id
        assert kimi.model == "kimi-k3", kimi.model  # model reusa el campo existente (no model_id nuevo)
        assert kimi.max_tokens == 8000, kimi.max_tokens
        assert kimi.enabled is True

    async def test_from_db_carga_capability_con_allowed_motors_en_orden(self):
        catalog = await MotorCatalog.from_db()
        cap = catalog.get_capability("generate")
        assert cap is not None, "capability 'generate' no cargó desde DB"
        assert cap.allowed_motors == ["kimi", "ada"], cap.allowed_motors

    async def test_from_db_capability_sin_thot_no_incluye_motor_inexistente(self):
        catalog = await MotorCatalog.from_db()
        cap = catalog.get_capability("critique")
        assert cap is not None
        assert "thot" not in cap.allowed_motors, cap.allowed_motors
        assert cap.allowed_motors == ["ada"], cap.allowed_motors


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `cd /home/fruiz/jax/las_manos && PYTHONPATH=. .venv/bin/python _catalog_from_db_test.py`
Expected: FAIL — `AttributeError: type object 'MotorCatalog' has no attribute 'from_db'`.

- [ ] **Step 3: Agregar `transport`/`model_ref`/`provider_id` a `MotorEntry` y el método `from_db`**

En `~/jax/las_manos/motor_registry/catalog.py`, después del import de `dataclass`:

```python
import os

import aiomysql
```

En `MotorEntry`, **al final de la clase** (después de `max_tokens: int = 0`,
la última línea hoy) — no antes: los campos nuevos tienen default, y
Python exige que ningún campo sin default (`max_context_tokens`,
`sandbox_only`, `default_timeout_seconds`, `supports_reasoning`, todos
sin default) venga después de uno con default, o `dataclass` explota con
`TypeError: non-default argument follows default argument` al definir la
clase:

```python
    transport: str = "http_openai_compat"
    model_ref: int = 0
    provider_id: str = ""
```

(el campo `model` ya existente se mantiene — hoy es `cfg.get("model", "")` desde TOML; con DB pasa a ser el `model_id` resuelto via JOIN, incorporado en el mismo campo para no romper el uso existente en `worker.py::_call_kimi` (`motor_entry.model`)).

**`_load()` necesita una línea nueva** — sin esto, un `MotorCatalog(dict)`
construido a mano en un test (Task 3 lo hace) con `"transport": "ollama"`
en el dict quedaría con `MotorEntry.transport` en su default de clase
(`"http_openai_compat"`), nunca `"ollama"` — el guard de credencial de
Task 3 fallaría en silencio contra el valor equivocado. En
`MotorCatalog._load`, dentro del `for name, cfg in config.get("motors", {}).items():`,
agregar `transport=cfg.get("transport", "http_openai_compat"),` a la
construcción de `MotorEntry(...)` (mismo lugar que ya arma `max_tokens=cfg.get("max_tokens", 0),`).

Al final de la clase `MotorCatalog`, después de `enabled_motors`:

```python
    @classmethod
    async def from_db(cls) -> "MotorCatalog":
        """Carga motor/capability/capability_motor desde la DB compartida
        jax_memory -- mismo pool/patron de conexion que credential_resolver.py.
        Reemplaza la lectura de config.toml (TOML queda solo para
        [server]/kill_switch_path y lo que routes.py todavia usa aparte)."""
        conn = await aiomysql.connect(
            host=os.getenv("JAX_DB_HOST", "localhost"),
            port=int(os.getenv("JAX_DB_PORT", "3306")),
            user=os.getenv("JAX_DB_USER", ""),
            password=os.getenv("JAX_DB_PASSWORD", ""),
            db=os.getenv("JAX_DB_NAME", "jax_memory"),
            charset="utf8mb4",
            autocommit=True,
        )
        try:
            instance = cls.__new__(cls)
            instance._motors = {}
            instance._capabilities = {}
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT m.`key`, m.model_ref, mo.provider_id, mo.model_id, p.base_url, "
                    "       m.transport, m.max_tokens, m.default_timeout_seconds, "
                    "       m.supports_reasoning, m.reasoning_default_visibility, "
                    "       m.sandbox_only, m.status "
                    "FROM motor m "
                    "JOIN model mo ON mo.id = m.model_ref "
                    "JOIN provider p ON p.id = mo.provider_id"
                )
                # api_url viene de provider.base_url (JOIN arriba) -- sin esto
                # _call_http_openai_compat (Task 3) arma "/chat/completions" sin
                # host, porque MotorEntry.api_url nunca se pobló desde ningun lado.
                for (key, model_ref, provider_id, model_id, base_url, transport, max_tokens,
                     timeout, reasoning, visibility, sandbox, status) in await cur.fetchall():
                    instance._motors[key] = MotorEntry(
                        name=key,
                        enabled=(status == "active"),
                        provider=provider_id,
                        api_key_env="",
                        api_url=base_url or "",
                        model=model_id,
                        max_context_tokens=0,
                        sandbox_only=bool(sandbox),
                        default_timeout_seconds=timeout,
                        supports_reasoning=bool(reasoning),
                        reasoning_default_visibility=visibility,
                        max_tokens=max_tokens or 0,
                        transport=transport,
                        model_ref=model_ref,
                        provider_id=provider_id,
                    )

                await cur.execute(
                    "SELECT `key`, risk_level, sandbox_only, requires_human_gate, "
                    "       max_execution_minutes, max_recursion_depth, output_schema, "
                    "       fallback_motor, fallback_mode, allowed_callers, forbidden_paths "
                    "FROM capability"
                )
                cap_rows = await cur.fetchall()

                await cur.execute(
                    "SELECT capability_key, motor_key FROM capability_motor ORDER BY capability_key, priority ASC"
                )
                motor_rows = await cur.fetchall()

            import json as _json
            allowed_by_cap: dict[str, list[str]] = {}
            for capability_key, motor_key in motor_rows:
                allowed_by_cap.setdefault(capability_key, []).append(motor_key)

            for (key, risk_level, sandbox_only, gate, max_exec, max_rec, schema,
                 fallback_motor, fallback_mode, callers, forbidden) in cap_rows:
                instance._capabilities[key] = CapabilityEntry(
                    name=key,
                    allowed_motors=allowed_by_cap.get(key, []),
                    allowed_callers=_json.loads(callers) if callers else [],
                    risk_level=risk_level,
                    sandbox_only=bool(sandbox_only),
                    requires_human_gate=bool(gate),
                    max_execution_minutes=max_exec,
                    max_recursion_depth=max_rec,
                    output_schema=schema or "",
                    fallback_motor=fallback_motor,
                    fallback_mode=fallback_mode or "manual_only",
                    forbidden_paths=_json.loads(forbidden) if forbidden else [],
                )
            return instance
        finally:
            conn.close()
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `cd /home/fruiz/jax/las_manos && PYTHONPATH=. .venv/bin/python _catalog_from_db_test.py`
Expected: `OK` (3 tests).

- [ ] **Step 5: Correr el test existente `_worker_max_tokens_test.py` para confirmar que `MotorCatalog(dict)` sigue intacto**

Run: `cd /home/fruiz/jax/las_manos && PYTHONPATH=. .venv/bin/python _worker_max_tokens_test.py`
Expected: 5/5 tests pasan (sin tocar este archivo todavía — confirma que agregar campos con default a `MotorEntry` no rompió el constructor dict-shaped).

- [ ] **Step 6: Wiring en `routes.py` — carga async al startup en vez de sync al import**

En `~/jax/las_manos/motor_registry/routes.py`, reemplazar:

```python
with open(CONFIG_PATH, "rb") as _f:
    _CONFIG = tomllib.load(_f)

_STORE = JobStore(str(BASE_DIR / "logs" / "motor_jobs.jsonl"))
_CATALOG = MotorCatalog(_CONFIG)
_POLICY = MotorPolicy(_CATALOG)
_KILL_SWITCH_PATH: str = _CONFIG.get("server", {}).get("kill_switch_path", "/etc/jax/PAUSE")
```

por:

```python
with open(CONFIG_PATH, "rb") as _f:
    _CONFIG = tomllib.load(_f)

_STORE = JobStore(str(BASE_DIR / "logs" / "motor_jobs.jsonl"))
# _CATALOG/_POLICY arrancan None -- se pueblan en el startup hook de
# server.py (init_motor_catalog, abajo). [motors.*]/[capabilities.*] de
# config.toml ya no se leen (R4 -- catalogo en DB). Ningun otro modulo
# importa estos dos nombres directamente (verificado: grep -rn "_CATALOG"
# solo los usa este archivo), asi que reasignarlos acá es seguro.
_CATALOG: MotorCatalog | None = None
_POLICY: MotorPolicy | None = None
_KILL_SWITCH_PATH: str = _CONFIG.get("server", {}).get("kill_switch_path", "/etc/jax/PAUSE")


async def init_motor_catalog() -> None:
    """Llamado desde el startup hook de server.py. Falla cerrado: si la DB
    no responde al arrancar, _CATALOG queda None y cada dispatch rechaza
    explícito (ver check en dispatch abajo) en vez de arrancar con un
    catálogo vacío en silencio."""
    global _CATALOG, _POLICY
    _CATALOG = await MotorCatalog.from_db()
    _POLICY = MotorPolicy(_CATALOG)
```

Buscar el endpoint `POST /motor/dispatch` en este mismo archivo (el que usa `_POLICY.check(...)`) y agregar, como primera línea del handler:

```python
    if _POLICY is None or _CATALOG is None:
        raise HTTPException(status_code=503, detail="Motor Registry: catálogo no inicializado todavía")
```

- [ ] **Step 7: Llamar `init_motor_catalog()` desde el startup hook de `server.py`**

En `~/jax/las_manos/server.py`, dentro de `@app.on_event("startup") async def _jacobs_init()`, después de `await jacobs_store.init_tables()`:

```python
    from motor_registry.routes import init_motor_catalog
    await init_motor_catalog()
```

- [ ] **Step 8: Verificación manual — reiniciar el servicio y confirmar el catálogo cargado**

```bash
sudo systemctl restart jax-las-manos.service
sleep 2
journalctl -u jax-las-manos -n 20 --no-pager
curl -s http://127.0.0.1:7777/health
```
Expected: sin traceback en journal, `/health` responde 200. (No hay endpoint que exponga el catálogo todavía — Task 9 lo agrega vía admin; esta verificación solo confirma que el startup no rompe.)

- [ ] **Step 9: Commit**

```bash
cd /home/fruiz/jax
git add las_manos/motor_registry/catalog.py las_manos/motor_registry/routes.py las_manos/server.py las_manos/_catalog_from_db_test.py
git commit -m "feat(motor_registry): catalog.py lee motor/capability/capability_motor de DB

MotorCatalog.from_db() reemplaza la lectura de config.toml para
[motors.*]/[capabilities.*]. routes.py pasa de construir el catalogo sync
al import a poblarlo async en el startup hook de server.py -- mismo
_CATALOG/_POLICY, ahora None hasta que init_motor_catalog() corre.
MotorCatalog(dict) conserva su firma (_load gana una linea para no
perder 'transport' de un dict armado a mano) -- _worker_max_tokens_test.py
sigue en verde sin tocarlo."
```

---

### Task 3: `worker.py` — dispatch por `transport`, credencial opcional

**Files:**
- Modify: `~/jax/las_manos/motor_registry/worker.py`
- Modify: `~/jax/las_manos/_worker_max_tokens_test.py` (agregar `"transport"` al fixture, nuevos casos)

**Interfaces:**
- Consumes: `MotorEntry.transport` (Task 2), `MotorEntry.provider_id` (Task 2 — reemplaza `_MOTOR_PROVIDER_MAP`).
- Produces: `_call_http_openai_compat(...)` (reemplaza `_call_kimi`, mismo contrato: recibe `api_url/model/api_key/prompt/timeout/max_tokens`, devuelve el dict JSON completo de la respuesta) — usado tanto para `transport="http_openai_compat"` como `transport="ollama"` (mismo formato de request/response, verificado en vivo contra `http://localhost:11434/v1/chat/completions`).

- [ ] **Step 1: Actualizar el fixture del test existente con `transport`**

En `~/jax/las_manos/_worker_max_tokens_test.py`, en `_MOTOR_CFG["motors"]["kimi"]`, agregar la clave:

```python
            "transport": "http_openai_compat",
```

(inmediatamente después de `"provider": "kimi",`).

- [ ] **Step 2: Escribir los tests que fallan — dispatch por transporte y credencial opcional**

Agregar al final de la clase `WorkerMaxTokensTest` en `_worker_max_tokens_test.py`:

```python
    async def test_transport_ollama_no_resuelve_credencial(self):
        """Guard igual a facet_resolver.py:81-82 -- transport='ollama' nunca
        llama a resolve_credential_instrumented. Si lo hiciera, este test
        lo detecta porque el mock de credential está seteado para explotar."""
        cfg = {
            "motors": {"jax_local": {
                "enabled": True, "provider": "ollama", "api_key_env": "",
                "api_url": "http://localhost:11434/v1", "model": "qwen3-coder:30b",
                "max_context_tokens": 0, "sandbox_only": True,
                "default_timeout_seconds": 300, "supports_reasoning": False,
                "transport": "ollama",
            }},
            "capabilities": {"implementation": _MOTOR_CFG["capabilities"]["implementation"]},
        }
        catalog = MotorCatalog(cfg)
        store = JobStore(str(Path(self._tmpdir.name) / "jobs2.jsonl"))
        job_id = store.create(
            caller="jacobs", capability="implementation", motor="jax_local",
            trace_id="t2", prompt="prompt de prueba", recursion_depth=0,
        )
        boom = AsyncMock(side_effect=AssertionError("no debería resolver credencial para ollama"))
        with patch.object(worker, "resolve_credential_instrumented", boom), \
             patch("httpx.AsyncClient.post", AsyncMock(return_value=_fake_response(content="listo"))):
            await worker.run(
                job_id=job_id, motor="jax_local", capability="implementation",
                prompt="prompt de prueba", context={}, store=store,
                catalog=catalog, kill_switch_path=self.kill_switch_path,
            )
        assert store._index[job_id]["status"] == "completed", store._index[job_id]

    async def test_transport_desconocido_falla_explicito_no_silencioso(self):
        cfg = {
            "motors": {"futuro": {
                "enabled": True, "provider": "x", "api_key_env": "", "api_url": "",
                "model": "x", "max_context_tokens": 0, "sandbox_only": True,
                "default_timeout_seconds": 300, "supports_reasoning": False,
                "transport": "subprocess",
            }},
            "capabilities": {"implementation": _MOTOR_CFG["capabilities"]["implementation"]},
        }
        catalog = MotorCatalog(cfg)
        store = JobStore(str(Path(self._tmpdir.name) / "jobs3.jsonl"))
        job_id = store.create(
            caller="jacobs", capability="implementation", motor="futuro",
            trace_id="t3", prompt="prompt de prueba", recursion_depth=0,
        )
        await worker.run(
            job_id=job_id, motor="futuro", capability="implementation",
            prompt="prompt de prueba", context={}, store=store,
            catalog=catalog, kill_switch_path=self.kill_switch_path,
        )
        state = store._index[job_id]
        assert state["status"] == "failed", state
        assert "transport" in state["error"].lower(), state
```

- [ ] **Step 3: Correr y verificar que ambos fallan**

Run: `cd /home/fruiz/jax/las_manos && PYTHONPATH=. .venv/bin/python _worker_max_tokens_test.py`
Expected: FAIL en los 2 tests nuevos (`test_transport_ollama_no_resuelve_credencial` explota porque hoy `worker.py` siempre llama a `resolve_credential_instrumented`; `test_transport_desconocido_falla_explicito_no_silencioso` falla porque `worker.py` no sabe de `motor.transport` todavía).

- [ ] **Step 4: Generalizar `worker.py`**

Reemplazar el import de `_MOTOR_PROVIDER_MAP` (líneas 35-44) — eliminar el diccionario hardcodeado completo, ya no hace falta (`MotorEntry.provider_id` lo reemplaza):

```python
from motor_registry.job_store import JobStore
from motor_registry.models import JobStatus
from motor_registry.output_validator import validate
```

Reemplazar `_call_kimi` (líneas 84-113) por:

```python
async def _call_http_openai_compat(
    *,
    api_url: str,
    model: str,
    api_key: str,
    prompt: str,
    timeout: float,
    max_tokens: int = 0,
) -> dict:
    """Llama a un endpoint OpenAI-compatible. Usado tanto para
    transport='http_openai_compat' (Kimi/Ada/futuros con API key) como para
    transport='ollama' (Qwen local, api_key='') -- Ollama expone el mismo
    formato de request/response en /v1/chat/completions, verificado en vivo
    (2026-08-18): mismo choices[0].message.content/finish_reason/usage.

    max_tokens (2026-08-10): sin esto, un motor de razonamiento puede gastar
    todo el completion budget en reasoning_content y devolver `content`
    cortado a mitad de palabra — bug real reproducido en vivo contra la API
    de Moonshot. 0/falsy = no mandar el campo."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{api_url}/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


# transport -> función de dispatch. Un motor nuevo elige un transporte
# EXISTENTE por nombre (dato en DB, Task 1/2) -- agregar un transporte
# nuevo (ej. http_gemini, subprocess) sí requiere código acá, a propósito
# (R4: los transportes son lógica, los motores son dato).
_TRANSPORT_DISPATCH = {
    "http_openai_compat": _call_http_openai_compat,
    "ollama": _call_http_openai_compat,
}
```

Nota: `api_url` para `transport='ollama'` debe incluir `/v1` (el motor `jax_local` se registra con `api_url="http://localhost:11434/v1"` en Task 4 — la función arma `{api_url}/chat/completions`, igual que para los demás).

En `run()`, reemplazar el bloque de resolución de credencial (líneas 166-177):

```python
    # Validar motor en catálogo
    motor_entry = catalog.get_motor(motor)
    if motor_entry is None:
        store.update(
            job_id,
            status=JobStatus.FAILED.value,
            finished_at=time.time(),
            error=f"Motor '{motor}' no encontrado en el catálogo",
        )
        return

    # Validar transporte soportado (R4 -- generalizado, ya no solo Kimi)
    call_fn = _TRANSPORT_DISPATCH.get(motor_entry.transport)
    if call_fn is None:
        store.update(
            job_id,
            status=JobStatus.FAILED.value,
            finished_at=time.time(),
            error=f"transport '{motor_entry.transport}' del motor '{motor}' no tiene dispatcher implementado",
        )
        return

    # Validar API key -- guard igual a facet_resolver.py:81-82: ollama/subprocess
    # no usan credencial de proveedor gestionada aca.
    provider_id = motor_entry.provider_id or motor
    api_key = ""
    if motor_entry.transport not in ("ollama", "subprocess"):
        try:
            api_key = await resolve_credential_instrumented(provider_id)
        except CredentialUnavailableError:
            store.update(
                job_id,
                status=JobStatus.FAILED.value,
                finished_at=time.time(),
                error=f"Sin credencial válida configurada para '{provider_id}'",
            )
            return
```

Y en el `asyncio.create_task(...)` que lanza la llamada HTTP (líneas 209-218), reemplazar `_call_kimi(...)` por:

```python
    api_task = asyncio.create_task(
        call_fn(
            api_url=motor_entry.api_url,
            model=motor_entry.model,
            api_key=api_key,
            prompt=prompt_with_identity,
            timeout=float(motor_entry.default_timeout_seconds),
            max_tokens=motor_entry.max_tokens,
        )
    )
```

- [ ] **Step 5: Correr los tests y verificar que pasan**

Run: `cd /home/fruiz/jax/las_manos && PYTHONPATH=. .venv/bin/python _worker_max_tokens_test.py`
Expected: `OK` (7 tests: 5 originales + 2 nuevos).

- [ ] **Step 6: Commit**

```bash
cd /home/fruiz/jax
git add las_manos/motor_registry/worker.py las_manos/_worker_max_tokens_test.py
git commit -m "feat(motor_registry): worker.py despacha por transport, no por nombre de motor

_call_kimi generalizado a _call_http_openai_compat (mismo formato sirve a
http_openai_compat y ollama, verificado en vivo contra Ollama /v1/chat/completions).
Guard de credencial opcional igual a facet_resolver.py:81-82. Elimina
_MOTOR_PROVIDER_MAP hardcodeado -- MotorEntry.provider_id lo reemplaza,
resuelto por JOIN en catalog.py (Task 2)."
```

---

### Task 4: Alta de Qwen como motor + backfill de `thot` (habilita criterio #4 más adelante)

**Files:**
- Modify: `jax-platform/backend/db/migrations.py` (nueva función de seed puntual, llamada en `run_migrations()`)

**Interfaces:**
- Consumes: `model` row `ollama/qwen3-coder:30b` (ya sembrada, ver spec §3), `provider` row `ollama` (ya sembrada, `auth_type='none'`).
- Produces: fila `motor` con `key='jax_local'`.

- [ ] **Step 1: Escribir el test que falla**

Agregar a `jax-platform/backend/tests/test_motor_migrations.py`:

```python
def test_seed_jax_local_como_motor_ollama(client):
    rows = client.portal.call(
        _fetch_all,
        "SELECT transport, max_tokens FROM motor WHERE `key`='jax_local'",
    )
    assert len(rows) == 1, rows
    assert rows[0][0] == "ollama"


def test_seed_provider_ollama_base_url_incluye_v1(client):
    """Ollama expone /v1/chat/completions (OpenAI-compatible) -- confirmado
    en vivo. provider.base_url tenía 'http://localhost:11434' sin /v1
    (ningún código lo consumía todavía); ahora sí, worker.py lo usa."""
    rows = client.portal.call(
        _fetch_all, "SELECT base_url FROM provider WHERE id='ollama'",
    )
    assert rows[0][0] == "http://localhost:11434/v1", rows


def test_seed_code_swarm_no_incluye_jax_local(client):
    """jax_local compite por capabilities de razonamiento/generación
    (Task 4), no por code_swarm (alto riesgo, gateado humano) -- decisión
    de dato, documentada en el spec §7, no de código."""
    rows = client.portal.call(
        _fetch_all,
        "SELECT motor_key FROM capability_motor WHERE capability_key='code_swarm'",
    )
    assert "jax_local" not in [r[0] for r in rows], rows
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/python -m pytest tests/test_motor_migrations.py -v -k jax_local or ollama`
Expected: FAIL — sin fila `motor` para `jax_local`, `provider.base_url` de ollama sin `/v1` todavía.

- [ ] **Step 3: Agregar el seed puntual**

En `jax-platform/backend/db/migrations.py`, después de `_seed_motors_and_capabilities` (antes de `_table_exists`):

```python
async def _seed_jax_local_motor(cur) -> None:
    """R4 Task 4: Qwen (jax_local) como motor real, no atado a la faceta
    conversacional. provider.base_url de ollama se corrige a incluir /v1 --
    ningun codigo lo consumia hasta ahora (chat.py::_call_ollama usa el
    formato nativo de Ollama, no este base_url), asi que es seguro.
    Compite por capabilities de razonamiento/generacion (generate, reason,
    design, reconcile), no por code_swarm/refactor/bug_hunt/implementation
    (agentico de alto riesgo, hoy exclusivo de Kimi) -- decision de dato,
    ajustable despues sin tocar codigo."""
    await cur.execute(
        "UPDATE provider SET base_url='http://localhost:11434/v1' "
        "WHERE id='ollama' AND base_url != 'http://localhost:11434/v1'"
    )
    await cur.execute(
        "SELECT id FROM model WHERE provider_id='ollama' AND model_id='qwen3-coder:30b'"
    )
    row = await cur.fetchone()
    if row is None:
        return
    await cur.execute(
        "INSERT IGNORE INTO motor "
        "(`key`, model_ref, transport, max_tokens, default_timeout_seconds, "
        " supports_reasoning, reasoning_default_visibility, sandbox_only) "
        "VALUES ('jax_local', %s, 'ollama', 0, 300, FALSE, 'audit_only', TRUE)",
        (row[0],),
    )
    for capability_key, priority in [("generate", 2), ("reason", 2), ("design", 2), ("reconcile", 2)]:
        await cur.execute(
            "INSERT IGNORE INTO capability_motor (capability_key, motor_key, priority) "
            "VALUES (%s, 'jax_local', %s)",
            (capability_key, priority),
        )
```

Y llamarla en `run_migrations()`, después de `await _seed_motors_and_capabilities(cur)`:

```python
            await _seed_jax_local_motor(cur)
```

- [ ] **Step 4: Correr y verificar que pasan**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/python -m pytest tests/test_motor_migrations.py -v`
Expected: todos los tests del archivo pasan (9 en total).

- [ ] **Step 5: Aplicar contra la DB real**

```bash
sudo systemctl restart jax-platform.service
mysql -h 127.0.0.1 -P 3308 -u jax_user jax_memory -e "SELECT \`key\`, transport FROM motor; SELECT base_url FROM provider WHERE id='ollama';"
```
Expected: 3 filas en `motor` (kimi, ada, jax_local), `base_url` con `/v1`.

- [ ] **Step 6: Commit**

```bash
cd /home/fruiz/jax-platform
git add backend/db/migrations.py backend/tests/test_motor_migrations.py
git commit -m "feat(db): Qwen (jax_local) como motor real -- capabilities de razonamiento

Cero dato nuevo de modelo (ollama/qwen3-coder:30b ya sembrado). Compite
por generate/reason/design/reconcile, no por las agenticas de alto riesgo
(exclusivas de Kimi por ahora). provider.base_url de ollama corregido a
incluir /v1 -- sin uso previo, ahora lo consume worker.py (Task 3)."
```

---

### Task 5: Jacobs — `step.motor` desacoplado de `step.facet`

**Files:**
- Modify: `~/jax/jacobs/models.py` (campo `Step.motor`)
- Modify: `~/jax/jacobs/plan.py` (`_from_spec` propaga `motor`)
- Modify: `~/jax/jacobs/executor.py` (`_invoke_motor` pasa `step.motor`, `_MOTOR_FACETS` suma `jax_local`)
- Test: `~/jax/jacobs/_step_motor_test.py` (nuevo)

**Interfaces:**
- Consumes: nada nuevo de tareas anteriores (usa el endpoint `/motor/dispatch` ya generalizado por Tasks 1-4).
- Produces: `Step.motor: str | None` — consumido por Task 6 (frontend, que lo setea) y Task 7 (test E2E).

- [ ] **Step 1: Escribir el test que falla**

Archivo nuevo `~/jax/jacobs/_step_motor_test.py`:

```python
#!/usr/bin/env python3
"""step.motor desacoplado de step.facet (R4). Cuando el spec trae 'motor'
explicito, viaja tal cual al Motor Registry. Cuando no, se manda motor=None
-- activa MotorPolicy._resolve_motor(), que ya existia y nunca corria
porque _invoke_motor siempre mandaba motor=step.facet (executor.py:519
antes de este fix).

Corre desde /home/fruiz/jax con:
  PYTHONPATH=/home/fruiz/jax .venv/bin/python jacobs/_step_motor_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from jacobs.executor import _invoke_motor
from jacobs.models import Pipeline, Step


def _pipeline():
    return Pipeline(name="test", invoked_by="test", user_id="1", tenant_id="1", mode="dry_run")


class StepMotorTest(unittest.IsolatedAsyncioTestCase):
    async def test_motor_explicito_viaja_tal_cual(self):
        step = Step(facet="kimi", capability="implementation", motor="ada")
        pipeline = _pipeline()
        fake_dispatch = AsyncMock()
        fake_dispatch.return_value.status_code = 200
        fake_dispatch.return_value.json = lambda: {"job_id": "j1", "status": "pending"}
        fake_dispatch.return_value.raise_for_status = lambda: None
        fake_poll = AsyncMock()
        fake_poll.return_value.status_code = 200
        fake_poll.return_value.json = lambda: {"status": "completed", "result_summary": "ok"}
        fake_poll.return_value.raise_for_status = lambda: None

        captured = {}

        async def fake_post(self, url, json=None, **kw):
            captured["payload"] = json
            return fake_dispatch.return_value

        with patch("httpx.AsyncClient.post", fake_post), \
             patch("httpx.AsyncClient.get", fake_poll):
            await _invoke_motor(step, pipeline, timeout=5)

        assert captured["payload"]["motor"] == "ada", captured["payload"]

    async def test_motor_ausente_manda_none_para_activar_resolver(self):
        step = Step(facet="kimi", capability="implementation", motor=None)
        pipeline = _pipeline()
        fake_dispatch_json = {"job_id": "j2", "status": "pending"}
        captured = {}

        class _Resp:
            status_code = 200
            def json(self): return fake_dispatch_json
            def raise_for_status(self): pass

        class _RespDone:
            status_code = 200
            def json(self): return {"status": "completed", "result_summary": "ok"}
            def raise_for_status(self): pass

        async def fake_post(self, url, json=None, **kw):
            captured["payload"] = json
            return _Resp()

        async def fake_get(self, url, **kw):
            return _RespDone()

        with patch("httpx.AsyncClient.post", fake_post), \
             patch("httpx.AsyncClient.get", fake_get):
            await _invoke_motor(step, pipeline, timeout=5)

        assert captured["payload"]["motor"] is None, captured["payload"]


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `cd /home/fruiz/jax && PYTHONPATH=. .venv/bin/python jacobs/_step_motor_test.py`
Expected: FAIL en `test_motor_explicito_viaja_tal_cual` (`TypeError` — `Step` no tiene campo `motor` todavía) y en el otro (`captured["payload"]["motor"]` sería `step.facet` = `"kimi"`, no `None`).

- [ ] **Step 3: Agregar el campo `motor` a `Step`**

En `~/jax/jacobs/models.py`, en `class Step(BaseModel)`, después de `facet: str`:

```python
    motor:            str | None = None  # R4: motor explícito, separado de facet.
                                          # None = MotorPolicy._resolve_motor() elige por competencia.
```

- [ ] **Step 4: Propagar en `PlanBuilder._from_spec`**

En `~/jax/jacobs/plan.py:198-208`, agregar `motor=spec.get("motor"),` al `Step(...)`:

```python
            steps.append(Step(
                step_id=str(uuid.uuid4()),
                pipeline_id=pipeline_id,
                step_index=i,
                facet=spec.get("facet", "jax_local"),
                motor=spec.get("motor"),
                capability=capability,
                input=input_data,
                depends_on=spec.get("depends_on", []),
                timeout_seconds=spec.get("timeout_seconds", default_timeout),
                skip_on_fail=spec.get("skip_on_fail", False),
            ))
```

- [ ] **Step 5: `_invoke_motor` pasa `step.motor` en vez de `step.facet`**

En `~/jax/jacobs/executor.py:516-524`, cambiar `"motor": step.facet,` por:

```python
        "motor":      step.motor,  # None = MotorPolicy resuelve por competencia (R4)
```

- [ ] **Step 6: Sumar `jax_local` al conjunto gobernado**

En `~/jax/jacobs/executor.py:44-47`:

```python
_HTTP_FACETS = frozenset({"hipatia", "jekyll", "thot", "ada"})
_MOTOR_FACETS = frozenset({"kimi", "jax_local"})
```

- [ ] **Step 7: Correr el test nuevo y verificar que pasa**

Run: `cd /home/fruiz/jax && PYTHONPATH=. .venv/bin/python jacobs/_step_motor_test.py`
Expected: `OK` (2 tests).

- [ ] **Step 8: Correr la suite de tests existente de Jacobs, confirmar cero regresión**

Run: `cd /home/fruiz/jax && PYTHONPATH=. .venv/bin/python -m pytest tests/ -q -k jacobs or executor or plan`
Expected: sin nuevos fallos. Prestar atención especial a cualquier test que asuma `jax_local` en `_HTTP_FACETS`/fuera de `_MOTOR_FACETS` (NIVEL B, `executor.py:656` — ahora `jax_local` sí pasa por la validación de `allowed_motors`, algo que antes no le aplicaba).

- [ ] **Step 9: Commit**

```bash
cd /home/fruiz/jax
git add jacobs/models.py jacobs/plan.py jacobs/executor.py jacobs/_step_motor_test.py
git commit -m "feat(jacobs): step.motor desacoplado de step.facet

_invoke_motor manda step.motor (None por defecto) en vez de step.facet
-- activa MotorPolicy._resolve_motor(), que ya existía sin caller real.
jax_local se suma a _MOTOR_FACETS: pasa a despacharse por el job-queue
gobernado de Motor Registry en vez de _invoke_ollama directo."
```

---

### Task 6: Frontend — elegir capability + motor (opcional) en Pipeline

**Files:**
- Create: `jax-platform/backend/api/motors.py` (nuevo endpoint, lee `capability`/`motor`/`capability_motor` directo de la DB compartida)
- Modify: `jax-platform/backend/main.py` (registrar el router nuevo — verificar el patrón real de registro de otros routers antes de escribir la línea)
- Modify: `jax-platform/frontend/src/components/BottomBar/PipelineModal.jsx`
- Test: `jax-platform/backend/tests/test_motors_endpoint.py` (nuevo)

**Interfaces:**
- Produces: `GET /api/motors/capabilities` → `{"capabilities": [{"key": str, "allowed_motors": [str, ...]}]}`.

- [ ] **Step 1: Verificar el patrón real de registro de routers**

Run: `grep -n "include_router" /home/fruiz/jax-platform/backend/main.py`

Usar exactamente ese patrón (mismo estilo de import/prefix) al registrar el router nuevo en Step 3 — no asumir la forma sin haberla visto.

- [ ] **Step 2: Escribir el test que falla**

Archivo nuevo `jax-platform/backend/tests/test_motors_endpoint.py`:

```python
"""GET /api/motors/capabilities -- capability + allowed_motors en orden de
priority, para que el frontend arme el picker de Pipeline (R4)."""


def test_capabilities_incluye_generate_con_kimi_y_ada_en_orden(client):
    resp = client.get("/api/motors/capabilities", headers=client.auth_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    by_key = {c["key"]: c for c in body["capabilities"]}
    assert "generate" in by_key
    assert by_key["generate"]["allowed_motors"] == ["kimi", "ada"]


def test_capabilities_critique_no_incluye_thot(client):
    resp = client.get("/api/motors/capabilities", headers=client.auth_headers)
    body = resp.json()
    by_key = {c["key"]: c for c in body["capabilities"]}
    assert "thot" not in by_key["critique"]["allowed_motors"]
```

Nota: verificar el fixture real de auth en `conftest.py` (`client.auth_headers` es un placeholder de forma — confirmar el nombre exacto usado por otros tests de `tests/` antes de correr, ej. `grep -n "auth_headers\|def client" jax-platform/backend/tests/conftest.py`).

- [ ] **Step 3: Correr y verificar que falla**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/python -m pytest tests/test_motors_endpoint.py -v`
Expected: FAIL — 404, el endpoint no existe.

- [ ] **Step 4: Crear el endpoint**

Archivo nuevo `jax-platform/backend/api/motors.py`:

```python
"""R4 -- expone capability/allowed_motors (tablas motor/capability/
capability_motor, ver db/migrations.py) para que el frontend arme el
picker de motor en Pipeline. Solo lectura -- el alta de motores nuevos es
Task 9 (admin)."""
from fastapi import APIRouter, Depends
from auth.middleware import get_current_user
from auth.models import AuthUser
from db.connection import get_pool

router = APIRouter(prefix="/api/motors")


@router.get("/capabilities")
async def list_capabilities(user: AuthUser = Depends(get_current_user)):
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT `key` FROM capability ORDER BY `key`")
            keys = [r[0] for r in await cur.fetchall()]
            await cur.execute(
                "SELECT capability_key, motor_key FROM capability_motor ORDER BY capability_key, priority ASC"
            )
            by_cap: dict[str, list[str]] = {}
            for capability_key, motor_key in await cur.fetchall():
                by_cap.setdefault(capability_key, []).append(motor_key)
    return {"capabilities": [{"key": k, "allowed_motors": by_cap.get(k, [])} for k in keys]}
```

- [ ] **Step 5: Registrar el router**

En `jax-platform/backend/main.py`, agregar el import y `include_router` con la misma forma exacta encontrada en Step 1.

- [ ] **Step 6: Correr y verificar que pasa**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/python -m pytest tests/test_motors_endpoint.py -v`
Expected: 2 passed.

- [ ] **Step 7: `PipelineModal.jsx` — picker de motor para el step de Kimi/jax_local**

Reemplazar `getFacetOptions`/`buildSteps` en `jax-platform/frontend/src/components/BottomBar/PipelineModal.jsx`:

```jsx
import { useState, useEffect } from 'react'
import { useI18n } from '../../i18n/index.jsx'
import { useJaxStore } from '../../store/useJaxStore'

// capability/desc son de otro sistema (las_manos, tablas motor/capability/
// capability_motor -- R4) -- label/color vienen de facetsState (/api/facets).
function getFacetOptions(t, facetsState) {
  return [
    { id: 'jax_local', capability: 'reasoning',       desc: t.descJaxLocal },
    { id: 'hipatia',   capability: 'research',         desc: t.descHipatia },
    { id: 'jekyll',    capability: 'analysis',         desc: t.descJekyll },
    { id: 'thot',      capability: 'critique',         desc: t.descThot },
    { id: 'kimi',      capability: 'implementation',   desc: t.descKimi },
    { id: 'ada',       capability: 'analysis',         desc: t.descAda },
  ].map(f => ({
    ...f,
    label: facetsState[f.id]?.display_name || facetsState[f.id]?.name || f.id,
    color: facetsState[f.id]?.color || '#94a3b8',
  }))
}

// facet -> capability real de las_manos (las que sí llegan a Motor
// Registry hoy: kimi y jax_local, tras Task 5). El resto de facetas del
// picker (hipatia/jekyll/thot/ada directas) no pasan por acá -- su
// "capability" es solo etiqueta descriptiva, sin motor que elegir.
const GOVERNED_FACET_CAPABILITY = { kimi: 'implementation', jax_local: 'generate' }

function buildSteps(selectedFacets, objective, facetOptions, motorChoices) {
  return facetOptions
    .filter(f => selectedFacets.includes(f.id))
    .map(f => {
      const step = {
        facet: f.id,
        capability: GOVERNED_FACET_CAPABILITY[f.id] || f.capability,
        prompt: `${f.desc}: ${objective}`,
        timeout_seconds: 300,
        skip_on_fail: false,
      }
      if (GOVERNED_FACET_CAPABILITY[f.id] && motorChoices[f.id]) {
        step.motor = motorChoices[f.id]  // vacío/no seteado = None, auto por competencia
      }
      return step
    })
}

export default function PipelineModal({ objective, onClose, onSubmit }) {
  const { t } = useI18n()
  const facetsState = useJaxStore((s) => s.facets)
  const FACET_OPTIONS = getFacetOptions(t, facetsState)

  const [mode, setMode] = useState('supervised')
  const [selected, setSelected] = useState(['hipatia', 'jekyll', 'thot'])
  const [submitting, setSubmitting] = useState(false)
  const [capabilities, setCapabilities] = useState({})  // {capability_key: [motor_key, ...]}
  const [motorChoices, setMotorChoices] = useState({})  // {facet_id: motor_key | ''}

  useEffect(() => {
    fetch('/api/motors/capabilities', { credentials: 'include' })
      .then(r => r.ok ? r.json() : { capabilities: [] })
      .then(data => {
        const byKey = {}
        for (const c of data.capabilities) byKey[c.key] = c.allowed_motors
        setCapabilities(byKey)
      })
      .catch(() => setCapabilities({}))
  }, [])

  function toggleFacet(id) {
    setSelected(s => s.includes(id) ? s.filter(x => x !== id) : [...s, id])
  }

  function setMotorFor(facetId, motorKey) {
    setMotorChoices(m => ({ ...m, [facetId]: motorKey }))
  }

  async function handleSubmit() {
    if (selected.length === 0) return
    setSubmitting(true)
    const steps = buildSteps(selected, objective, FACET_OPTIONS, motorChoices)
    await onSubmit({
      name: `Pipeline: ${objective.slice(0, 50)}`,
      objective,
      invoked_by: 'Fernando',
      mode,
      max_steps: Math.max(steps.length, 1),
      steps: steps.length > 0 ? steps : null,
    })
    setSubmitting(false)
    onClose()
  }

  const PIPELINE_MODES = [
    { id: 'supervised',  label: '👁 Supervised' },
    { id: 'autonomous',  label: '⚡ Autonomous' },
    { id: 'dry_run',     label: '🧪 Dry run' },
  ]

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ backgroundColor: 'rgba(0,0,0,0.7)' }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div className="bg-slate-900 border border-slate-700 rounded-xl p-5 w-full max-w-md shadow-2xl">
        <div className="mb-4">
          <h2 className="text-sm font-bold text-slate-200 uppercase tracking-widest">
            {t.newPipelineTitle}
          </h2>
          <p className="text-xs text-slate-500 mt-1 truncate">
            {t.objectiveLabel}: {objective}
          </p>
        </div>

        <div className="mb-4">
          <p className="text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wider">{t.modeLabel}</p>
          <div className="flex gap-2">
            {PIPELINE_MODES.map(({ id: m, label }) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`flex-1 py-1.5 rounded-lg text-xs font-semibold border transition-colors ${
                  mode === m
                    ? m === 'autonomous'
                      ? 'border-orange-500 bg-orange-500/20 text-orange-300'
                      : 'border-blue-500 bg-blue-500/20 text-blue-300'
                    : 'border-slate-700 bg-slate-800 text-slate-500 hover:text-slate-300'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="mb-5">
          <p className="text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wider">
            {t.facetsLabel}
          </p>
          <div className="space-y-1.5">
            {FACET_OPTIONS.map(f => {
              const cap = GOVERNED_FACET_CAPABILITY[f.id]
              const motorOptions = cap ? (capabilities[cap] || []) : []
              return (
                <div key={f.id}>
                  <label
                    className={`flex items-center gap-3 p-2 rounded-lg border cursor-pointer transition-colors ${
                      selected.includes(f.id)
                        ? 'border-opacity-60 bg-opacity-10'
                        : 'border-slate-800 bg-slate-800/50 hover:border-slate-700'
                    }`}
                    style={selected.includes(f.id) ? {
                      borderColor: f.color + '80',
                      backgroundColor: f.color + '12',
                    } : {}}
                  >
                    <input
                      type="checkbox"
                      checked={selected.includes(f.id)}
                      onChange={() => toggleFacet(f.id)}
                      className="sr-only"
                    />
                    <span
                      className="w-4 h-4 rounded border-2 flex items-center justify-center flex-shrink-0 text-xs"
                      style={selected.includes(f.id) ? {
                        borderColor: f.color,
                        backgroundColor: f.color,
                        color: '#000',
                      } : { borderColor: '#475569' }}
                    >
                      {selected.includes(f.id) ? '✓' : ''}
                    </span>
                    <span className="text-xs font-semibold" style={{ color: f.color }}>{f.label}</span>
                    <span className="text-xs text-slate-500">{f.desc}</span>
                  </label>
                  {selected.includes(f.id) && motorOptions.length > 0 && (
                    <select
                      className="ml-7 mt-1 text-xs bg-slate-800 border border-slate-700 rounded px-2 py-1 text-slate-300"
                      value={motorChoices[f.id] || ''}
                      onChange={(e) => setMotorFor(f.id, e.target.value)}
                    >
                      <option value="">{t.autoMotor || 'Auto (por competencia)'}</option>
                      {motorOptions.map(m => <option key={m} value={m}>{m}</option>)}
                    </select>
                  )}
                </div>
              )
            })}
          </div>
        </div>

        <div className="flex gap-2">
          <button
            onClick={onClose}
            className="flex-1 py-2 rounded-lg text-xs font-semibold bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-700 transition-colors"
          >
            {t.cancel}
          </button>
          <button
            onClick={handleSubmit}
            disabled={submitting || selected.length === 0}
            className="flex-1 py-2 rounded-lg text-xs font-bold text-white transition-colors disabled:opacity-40"
            style={{ backgroundColor: '#3b82f6' }}
          >
            {submitting ? t.starting : t.planAndExecute}
          </button>
        </div>
      </div>
    </div>
  )
}
```

Agregar la key `autoMotor` a `frontend/src/i18n/es.js` y `en.js` (`"Auto (por competencia)"` / `"Auto (by competence)"`) — verificar el patrón real de esos archivos (estructura de objeto, no asumir) antes de editar.

- [ ] **Step 8: Rebuild + deploy del frontend**

Por regla del proyecto (`jax-platform/CLAUDE.md`, "Rebuild + deploy frontend"):
```bash
cd /home/fruiz/jax-platform/frontend && npm run build
```
Luego el rsync a la VM dev documentado en ese mismo archivo (Lección operativa #1).

- [ ] **Step 9: Verificación manual en navegador**

Abrir la Mesa, pestaña Pipeline, seleccionar Kimi — confirmar que aparece el `<select>` de motor con al menos `kimi` como opción (poblado desde `/api/motors/capabilities`).

- [ ] **Step 10: Commit**

```bash
cd /home/fruiz/jax-platform
git add backend/api/motors.py backend/main.py backend/tests/test_motors_endpoint.py frontend/src/components/BottomBar/PipelineModal.jsx frontend/src/i18n/es.js frontend/src/i18n/en.js
git commit -m "feat(pipeline): elegir motor explícito o auto por competencia (R4)

Nuevo GET /api/motors/capabilities (solo lectura, tablas motor/capability/
capability_motor). PipelineModal.jsx suma un picker de motor para los
steps que llegan a Motor Registry (kimi, jax_local) -- vacío = auto,
resuelto por MotorPolicy._resolve_motor() del lado de LAS MANOS."
```

---

### Task 7: Test E2E real — Qwen y Kimi vía Pipeline, no mockeado

**Files:**
- Test: `~/jax/jacobs/_pipeline_motor_e2e_test.py` (nuevo, requiere servicios reales corriendo)

**Interfaces:**
- Consumes: `jax-las-manos.service` corriendo (Tasks 1-5 aplicadas), Ollama local corriendo con `qwen3-coder:30b` cargado.

- [ ] **Step 1: Escribir el test — job de Qwen por Pipeline real, sin mocks**

```python
#!/usr/bin/env python3
"""E2E real (R4, Tasks 1-5 aplicadas) -- no mockea httpx ni credencial.
Requiere jax-las-manos.service corriendo y Ollama con qwen3-coder:30b
cargado. Verifica el criterio de aceptación 1 y 2 del spec: Qwen ejecuta
una tarea de código real (no chat), Kimi vía Pipeline con motor explícito
vs auto.

Corre desde /home/fruiz/jax con:
  PYTHONPATH=/home/fruiz/jax .venv/bin/python jacobs/_pipeline_motor_e2e_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import httpx

LAS_MANOS = "http://127.0.0.1:7777"


async def _dispatch_and_wait(*, caller, capability, motor, prompt, timeout=60):
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(f"{LAS_MANOS}/motor/dispatch", json={
            "caller": caller, "capability": capability, "motor": motor, "prompt": prompt,
        })
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(2)
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{LAS_MANOS}/motor/job/{job_id}")
            job = r.json()
        if job["status"] in ("completed", "failed", "rejected"):
            return job
    raise TimeoutError(f"job {job_id} no completó en {timeout}s")


async def test_qwen_ejecuta_tarea_de_codigo_real():
    job = await _dispatch_and_wait(
        caller="jacobs", capability="generate", motor="jax_local",
        prompt="Escribí una función Python que sume dos números, con type hints.",
    )
    assert job["status"] == "completed", job
    print(f"OK Qwen: {job.get('result_summary', '')[:100]}")


async def test_kimi_con_motor_explicito_via_pipeline():
    job = await _dispatch_and_wait(
        caller="jacobs", capability="pipeline_analysis", motor="kimi",
        prompt="Confirmá en una frase que estás operativo.",
    )
    assert job["status"] == "completed", job
    print(f"OK Kimi explícito: {job.get('result_summary', '')[:100]}")


async def test_capability_generate_sin_motor_explicito_resuelve_auto():
    """motor=None -- MotorPolicy._resolve_motor() elige el primero
    habilitado de allowed_motors (generate: kimi, ada, jax_local en ese
    orden de priority)."""
    job = await _dispatch_and_wait(
        caller="jacobs", capability="generate", motor=None,
        prompt="Respondé solo con la palabra: listo.",
    )
    assert job["status"] == "completed", job
    print(f"OK auto: motor resuelto por competencia, job {job}")


async def main():
    await test_qwen_ejecuta_tarea_de_codigo_real()
    await test_kimi_con_motor_explicito_via_pipeline()
    await test_capability_generate_sin_motor_explicito_resuelve_auto()
    print("E2E: 3/3 OK")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Correr contra los servicios reales**

Run: `cd /home/fruiz/jax && PYTHONPATH=. .venv/bin/python jacobs/_pipeline_motor_e2e_test.py`
Expected: `E2E: 3/3 OK`. Si `test_qwen_ejecuta_tarea_de_codigo_real` falla con error de credencial, es la señal de que el guard de Task 3 (transport `ollama` sin resolver credencial) no está funcionando — no avanzar sin diagnosticar la causa raíz (no reintentar a ciegas).

- [ ] **Step 3: Verificar en `motor_jobs.jsonl` con el runbook ya existente**

Run: `grep '"motor": "jax_local"' ~/jax/las_manos/logs/motor_jobs.jsonl | tail -3 | python3 -m json.tool` (mismo comando que `docs/runbooks/verificar-truncamiento-kimi.md` ya documenta para Kimi) — confirmar `_finish_reason: "stop"`.

- [ ] **Step 4: Commit**

```bash
cd /home/fruiz/jax
git add jacobs/_pipeline_motor_e2e_test.py
git commit -m "test(e2e): Qwen y Kimi vía Pipeline real, sin mocks (R4 criterios 1-2)

Job real de Qwen por capability=generate (no conversación desde jax_local),
Kimi con motor explícito y con motor=None (resuelto por competencia)."
```

---

### Task 8: Criterio de aceptación #4 — motor nuevo sin tocar código

**Files:**
- Modify: `jax-platform/backend/db/migrations.py` (backfill de `thot` como motor + las 2 filas de `capability_motor` que Task 1 excluyó)
- Test: extender `jax-platform/backend/tests/test_motor_migrations.py`

**Interfaces:**
- Consumes: `provider`/`credential` ya activos para `openai` (usados por Thot en la Mesa — cero credencial nueva).
- Produces: fila `motor` con `key='thot'`, y las 2 filas `capability_motor` (`validate_consistency`, `critique`) que Task 1 dejó pendientes.

**Este es el criterio decisivo del spec: dar de alta un motor que hoy no existe en el catálogo, sin tocar código de dispatch (worker.py/catalog.py ya generalizados en Tasks 2-3), y confirmar que responde de verdad.**

- [ ] **Step 1: Escribir el test que falla**

Agregar a `jax-platform/backend/tests/test_motor_migrations.py`:

```python
def test_thot_existe_como_motor_y_completa_las_capabilities_pendientes(client):
    rows = client.portal.call(
        _fetch_all, "SELECT transport FROM motor WHERE `key`='thot'",
    )
    assert len(rows) == 1, rows
    assert rows[0][0] == "http_openai_compat"

    for cap in ("validate_consistency", "critique"):
        rows = client.portal.call(
            _fetch_all,
            "SELECT motor_key FROM capability_motor WHERE capability_key=%s ORDER BY priority",
            (cap,),
        )
        assert [r[0] for r in rows] == ["thot", "ada"], (cap, rows)
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/python -m pytest tests/test_motor_migrations.py -v -k thot`
Expected: FAIL — `thot` no existe en `motor` todavía.

- [ ] **Step 3: Agregar el seed — SOLO INSERT, cero cambio de código de dispatch**

En `jax-platform/backend/db/migrations.py`, después de `_seed_jax_local_motor` (antes de `_table_exists`):

```python
async def _seed_thot_motor(cur) -> None:
    """R4 -- criterio de aceptación decisivo del spec: motor nuevo dado de
    alta SOLO por dato (INSERT), sin tocar worker.py/catalog.py (ya
    generalizados por transport en Tasks 2-3). openai/credential ya están
    activos -- los usa Thot del lado de la Mesa, cero setup nuevo.
    validate_consistency/critique referenciaban 'thot' en config.toml
    (allowed_motors) pero Task 1 excluyó esas 2 filas porque el motor no
    existía -- se completan acá."""
    await cur.execute(
        "SELECT id FROM model WHERE provider_id='openai' AND model_id='gpt-5.5'"
    )
    row = await cur.fetchone()
    if row is None:
        return
    await cur.execute(
        "INSERT IGNORE INTO motor "
        "(`key`, model_ref, transport, max_tokens, default_timeout_seconds, "
        " supports_reasoning, reasoning_default_visibility, sandbox_only) "
        "VALUES ('thot', %s, 'http_openai_compat', 0, 300, FALSE, 'audit_only', TRUE)",
        (row[0],),
    )
    for capability_key in ("validate_consistency", "critique"):
        await cur.execute(
            "INSERT IGNORE INTO capability_motor (capability_key, motor_key, priority) "
            "VALUES (%s, 'thot', 0)",
            (capability_key,),
        )
        # ada ya tenía priority=0 (Task 1) -- bajarla a 1 para no empatar
        # con thot, sin tocar la fila de thot recién insertada.
        await cur.execute(
            "UPDATE capability_motor SET priority=1 "
            "WHERE capability_key=%s AND motor_key='ada' AND priority=0",
            (capability_key,),
        )
```

Llamarla en `run_migrations()`, después de `await _seed_jax_local_motor(cur)`:

```python
            await _seed_thot_motor(cur)
```

- [ ] **Step 4: Correr y verificar que pasa**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/python -m pytest tests/test_motor_migrations.py -v`
Expected: todos los tests del archivo pasan.

- [ ] **Step 5: Aplicar y verificar que responde de verdad (no un stub)**

```bash
sudo systemctl restart jax-platform.service
sudo systemctl restart jax-las-manos.service
```

Agregar a `~/jax/jacobs/_pipeline_motor_e2e_test.py` (o correr manual con el mismo helper `_dispatch_and_wait` de Task 7):

```python
async def test_thot_motor_nuevo_responde_de_verdad():
    """El criterio de aceptación #4: registrado por INSERT (Step 3 de esta
    tarea), sin una línea de código nueva de dispatch -- y responde."""
    job = await _dispatch_and_wait(
        caller="jacobs", capability="critique", motor="thot",
        prompt="En una frase: ¿cuál es el riesgo de no versionar un catálogo de motores?",
    )
    assert job["status"] == "completed", job
    print(f"OK thot (motor nuevo, cero código): {job.get('result_summary', '')[:150]}")
```

Run: `cd /home/fruiz/jax && PYTHONPATH=. .venv/bin/python -c "
import asyncio
from jacobs._pipeline_motor_e2e_test import test_thot_motor_nuevo_responde_de_verdad
asyncio.run(test_thot_motor_nuevo_responde_de_verdad())
"`
Expected: `OK thot (motor nuevo, cero código): ...` con contenido real, no vacío.

**Si esto falla, no se avanza a Task 9** (el form de Admin depende de que este mecanismo esté probado, no solo de que el schema exista).

- [ ] **Step 6: Commit**

```bash
cd /home/fruiz/jax-platform
git add backend/db/migrations.py backend/tests/test_motor_migrations.py
git commit -m "feat(db): thot como motor nuevo -- criterio de aceptación #4 de R4

Cero código de dispatch nuevo (worker.py/catalog.py ya generalizados,
Tasks 2-3) -- solo INSERT. Cero credencial nueva (openai ya activo, lo
usa Thot en la Mesa). Completa validate_consistency/critique, que Task 1
había dejado sin su referencia a 'thot' por no existir el motor todavía."

cd /home/fruiz/jax
git add jacobs/_pipeline_motor_e2e_test.py
git commit -m "test(e2e): confirma que thot (motor nuevo por INSERT) responde de verdad"
```

---

### Task 9: Admin/Modelos — form de alta de motor (ÚLTIMA tarea, después de Task 8 verde)

**Files:**
- Explorar primero: buscar los endpoints/componentes reales de Bloque D para provider/model (`jax-platform/backend/api/admin/`, `jax-platform/frontend/src/pages/` o `components/admin/`) — **leer el patrón real antes de escribir una línea**, no diseñar a ciegas.
- Create/Modify: endpoint(s) admin para `motor`/`capability`/`capability_motor` (CRUD), siguiendo el mismo patrón de auth/estructura que el admin de provider/model ya construido.
- Modify: componente(s) de frontend admin, misma pestaña o una nueva junto a "Facetas & Modelos".

**Interfaces:**
- Consumes: mismas tablas de Tasks 1/4/8, ya probadas por INSERT directo.

- [ ] **Step 1: Ubicar el patrón real de admin de Bloque D**

Run:
```bash
grep -rln "facet_binding\|model_binding_proposal" /home/fruiz/jax-platform/backend/api/admin/ 2>/dev/null
grep -rln "FacetBinding\|ModelCatalog\|facet_binding" /home/fruiz/jax-platform/frontend/src/ 2>/dev/null
```

Leer esos archivos completos antes de continuar — la forma exacta de router/auth/componente de este paso depende de lo que se encuentre ahí, no se puede especificar de antemano sin haberlo visto. Documentar acá (edición manual de este plan, antes de escribir código) qué archivos son la referencia real.

- [ ] **Step 2: Escribir los tests que fallan para el CRUD de `motor`**

(Backend, endpoint de creación — mínimo: crear, listar. Editar/borrar quedan fuera si el patrón de referencia de provider/model tampoco los tiene — seguir el mismo alcance que la referencia, no inventar más.)

- [ ] **Step 3: Implementar el endpoint de creación siguiendo el patrón encontrado en Step 1**

- [ ] **Step 4: Correr los tests del backend, verificar que pasan**

- [ ] **Step 5: Implementar el componente de frontend, mismo patrón visual/estructural que el admin de provider/model**

- [ ] **Step 6: Rebuild + deploy del frontend** (mismo procedimiento de Task 6, Step 8)

- [ ] **Step 7: Verificación manual — dar de alta un motor de prueba desde el form, confirmar que aparece en `capability_motor` y que un job real lo despacha**

Un motor de prueba real distinto de kimi/ada/jax_local/thot (los 4 ya sembrados) — por ejemplo `anthropic`/Hyde no aplica (transport `subprocess`, sin dispatch HTTP implementado en Task 3 a propósito), así que usar `gemini` (provider ya activo, usado por Hipatia) con `transport='http_gemini'` requeriría implementar `_call_gemini` en `worker.py` primero (fuera de alcance de Task 3, que solo generalizó `ollama`/`http_openai_compat`) — **documentar esto como limitación conocida del form**: hoy solo puede dar de alta motores con transporte `http_openai_compat` u `ollama` sin trabajo adicional; `http_gemini`/`subprocess` requieren sumar su dispatcher a `_TRANSPORT_DISPATCH` primero (deuda con nombre, no bloqueante para cerrar R4).

- [ ] **Step 8: Commit**

```bash
cd /home/fruiz/jax-platform
git add -A
git commit -m "feat(admin): form de alta de motor/capability (R4, última tarea)

CRUD sobre motor/capability/capability_motor, mismo patrón que el admin
de provider/model de Bloque D. Cablea el mecanismo ya probado por INSERT
directo en Task 8 -- no diseña contra un backend sin verificar."
```

---

## Self-review

**Cobertura del spec:** los 4 criterios de aceptación del spec quedan cubiertos — #1 (Qwen, Task 7), #2 (Kimi vía Pipeline con motor elegido, Task 7), #3 (elegir motor o auto, Tasks 5-7), #4 (motor nuevo sin código, Task 8, orden estricto respetado — Task 9 depende de que Task 8 pase). Los 3 ejes del spec (capability/transport/auth) están cada uno en su tabla/mecanismo: capability→`capability`/`capability_motor`, transport→`motor.transport` + `_TRANSPORT_DISPATCH`, auth→`provider.auth_type` (sin cambios, reusado). Fuera de alcance respetado: Comando y A7 no se tocan en ninguna tarea.

**Placeholders:** ninguno — cada step tiene código real, verificado contra los archivos reales del repo (line numbers confirmados por lectura directa, no de memoria). El único punto deliberadamente abierto es Task 9 Step 1 (leer el patrón real de admin antes de escribir) — no es un placeholder de "TODO", es la instrucción correcta cuando el patrón de referencia todavía no se ha leído; el plan es explícito sobre qué hacer con lo que se encuentre.

**Consistencia de tipos/firmas:** `MotorEntry.transport`/`model_ref`/`provider_id` (Task 2) se consumen igual en Task 3 (`motor_entry.transport`, `motor_entry.provider_id`) y Task 4 (sembrados vía SQL, no vía estos campos Python). `Step.motor` (Task 5) se consume igual en Task 6 (frontend lo setea) y Task 7 (test E2E lo pasa a `_dispatch_and_wait`). `_TRANSPORT_DISPATCH` (Task 3) es el único punto que Task 9 Step 7 señala como límite conocido (`http_gemini`/`subprocess` no tienen dispatcher) — consistente, no contradictorio.

**Hallazgo verificado durante la escritura de este plan, no anticipado en el spec:** `[capabilities.validate_consistency]` y `[capabilities.critique]` en `config.toml` ya referenciaban `allowed_motors = ["thot", ...]` **sin que `thot` existiera nunca como `[motors.thot]`** — un dangling reference real en producción, no introducido por este plan. Task 1 lo hace explícito (excluye esas 2 filas del seed en vez de romper la FK en silencio) y Task 8 lo resuelve como parte del criterio de aceptación #4 — las dos capabilities quedan funcionales por primera vez.

**Tres bugs reales encontrados y corregidos en esta pasada de self-review** (no hipotéticos — cada uno habría roto la ejecución tal como estaba escrito primero):
1. Task 2 insertaba `transport`/`model_ref`/`provider_id` (con default) en medio del dataclass `MotorEntry`, antes de campos sin default (`max_context_tokens`, etc.) — `TypeError: non-default argument follows default argument` al definir la clase. Corregido: los campos nuevos van al final.
2. `from_db()` dejaba `api_url=""` hardcodeado — sin el JOIN a `provider.base_url`, todo motor armaría `"/chat/completions"` sin host. Corregido: JOIN agregado, `api_url=base_url`.
3. El test de Task 2 asertaba `kimi.model_id`, un campo que nunca se agregó (el diseño reusa el campo `model` existente) — y `_load()` (el constructor dict-shaped) nunca leía `"transport"` del dict, así que los tests de Task 3 que arman `MotorCatalog(cfg)` con `"transport": "ollama"` lo hubieran ignorado en silencio, y el guard de credencial habría fallado contra el default de clase en vez del valor real. Corregidos ambos: el test usa `kimi.model`, y `_load()` gana la línea `transport=cfg.get(...)`.
