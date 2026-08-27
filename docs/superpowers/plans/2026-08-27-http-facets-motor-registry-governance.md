# `_HTTP_FACETS` Motor Registry Governance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cerrar el gap de gobernanza de `hipatia`/`jekyll`/`thot`/`ada` en los dos caminos que los despachan HTTP-directo (Jacobs y Mesa web), sin cambiar comportamiento de `kimi`/`jax_local`.

**Architecture:** Partir `MotorPolicy.check()` (`las_manos/motor_registry/policy.py`) en `check_capability_admission()` (checks 1-5, sin motor resuelto) + wrapper — Jacobs la llama en memoria (mismo proceso que `las_manos`). Mesa web no tiene concepto de `capability` por mensaje, así que usa un check distinto y más chico, `check_facet_admission()`, sobre una columna nueva `facet.allowed_callers`, expuesto vía un endpoint nuevo `POST /motor/authorize-facet`, fail-closed.

**Tech Stack:** Python 3.12, FastAPI, aiomysql, `unittest.IsolatedAsyncioTestCase`/`unittest.mock`, pytest (jax-platform), MariaDB (`jax_memory`).

**Spec:** `docs/superpowers/specs/2026-08-27-http-facets-motor-policy-governance-design.md`

## Global Constraints

- **Cero cambio de comportamiento para `kimi`/`jax_local`.** Si un test existente necesita modificarse para pasar, PARAR y reportar — no es un fix, es una señal de que el refactor cambió comportamiento.
- **Fail-closed en todo.** DB no responde, `las_manos` no responde, timeout, respuesta inesperada → siempre deniega. Nunca "no pude verificar, sigo igual".
- **Todo dato nuevo (columnas, valores de `allowed_callers`) preserva el acceso que existe hoy** — esta ronda cierra un gap estructural, no cambia quién puede qué.
- **`capability.sandbox_only` se marca vestigial, no se le inventa semántica.** El techo de timeout (`max_execution_minutes`) NO se activa esta ronda — queda declarado explícitamente como deuda separada.
- **`facet.allowed_callers` (columna nueva) nace CON su lector y CON test negativo real, en el mismo tramo del plan** — no repetir el destino de `capability.sandbox_only` (columna sin lector desde el día uno). Por eso la migración (Task 3) va ANTES de escribir el lector (Task 4), no después.
- Migraciones de datos son idempotentes con guarda `WHERE` (mismo patrón que `_fix_file_write_gate_and_auditor` en `jax-platform/backend/db/migrations.py`) — nunca pisan un valor manual futuro.
- **Si un "Expected" de este plan no se cumple al correr un paso, PARAR y reportar — nunca ajustar el código para forzar que se cumpla.**

---

### Task 1: Caracterización de `MotorPolicy.check()` — tests ANTES de tocar el código

**Files:**
- Create: `las_manos/motor_registry/_policy_test.py`

**Interfaces:**
- Consumes: `MotorPolicy` / `MotorCatalog` (`las_manos/motor_registry/policy.py`, `catalog.py`) — sin cambios, código actual.
- Produces: una suite que debe seguir pasando SIN MODIFICAR después de la Task 2 — es el contrato de "cero cambio de comportamiento".

Hoy no existe ningún test dedicado a `MotorPolicy.check()` (verificado: `grep -rln "MotorPolicy" --include="*.py"` no encuentra ningún archivo de test que lo ejercite directo). Esta task lo cubre por primera vez, ANTES del refactor de la Task 2, para tener una base real contra la cual medir "cero cambio".

- [ ] **Step 1: Escribir el test file completo**

```python
#!/usr/bin/env python3
"""MotorPolicy.check() -- caracterizacion ANTES de partirlo en
check_capability_admission() + wrapper (docs/superpowers/specs/
2026-08-27-http-facets-motor-policy-governance-design.md). No existia
ningun test dedicado a check() -- esta suite es la base real contra la
que se mide "cero cambio de comportamiento" en la Task 2. Debe seguir
pasando SIN MODIFICAR despues del split.

Catalogo armado a mano (MotorCatalog(dict), constructor dict-shaped que
el modulo conserva para tests -- ver catalog.py:83-135), sin DB real:
check() es "modulo puro, sin I/O" por diseno.

Corre desde /home/fruiz/jax/las_manos con:
  PYTHONPATH=/home/fruiz/jax/las_manos \
  /home/fruiz/jax/las_manos/.venv/bin/python \
  /home/fruiz/jax/las_manos/motor_registry/_policy_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import unittest

from motor_registry.catalog import MotorCatalog
from motor_registry.policy import MotorPolicy


def _catalog() -> MotorCatalog:
    return MotorCatalog({
        "motors": {
            "kimi": {
                "enabled": True, "sandbox_only": True,
                "transport": "http_openai_compat",
            },
        },
        "capabilities": {
            "implementation": {
                "allowed_motors": ["kimi"],
                "allowed_callers": ["jacobs", "hyde"],
                "risk_level": "medium",
                "sandbox_only": True,
                "requires_human_gate": False,
                "max_execution_minutes": 5,
                "max_recursion_depth": 0,
                "output_schema": "code_patch.v1",
            },
        },
    })


class MotorPolicyCheckTest(unittest.TestCase):
    def setUp(self):
        self.policy = MotorPolicy(_catalog())

    def test_caller_autorizado_pasa(self):
        result = self.policy.check(
            caller="jacobs", capability="implementation", motor=None,
            context_keys=[], recursion_depth=0, human_gate_token=None,
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.resolved_motor, "kimi")

    def test_caller_no_autorizado_rechaza(self):
        result = self.policy.check(
            caller="caller_fantasma", capability="implementation", motor=None,
            context_keys=[], recursion_depth=0, human_gate_token=None,
        )
        self.assertFalse(result.allowed)
        self.assertIn("no autorizado", result.reason)

    def test_capability_desconocida_rechaza(self):
        result = self.policy.check(
            caller="jacobs", capability="no_existe", motor=None,
            context_keys=[], recursion_depth=0, human_gate_token=None,
        )
        self.assertFalse(result.allowed)
        self.assertIn("desconocida", result.reason)

    def test_recursion_depth_excede_limite_rechaza(self):
        result = self.policy.check(
            caller="jacobs", capability="implementation", motor=None,
            context_keys=[], recursion_depth=1, human_gate_token=None,
        )
        self.assertFalse(result.allowed)
        self.assertIn("recursion_depth", result.reason)

    def test_clave_prohibida_en_context_rechaza(self):
        result = self.policy.check(
            caller="jacobs", capability="implementation", motor=None,
            context_keys=["prompt", "api_key"], recursion_depth=0,
            human_gate_token=None,
        )
        self.assertFalse(result.allowed)
        self.assertIn("prohibidas", result.reason)

    def test_timeout_excede_techo_rechaza(self):
        result = self.policy.check(
            caller="jacobs", capability="implementation", motor=None,
            context_keys=[], recursion_depth=0, human_gate_token=None,
            timeout_seconds=301,
        )
        self.assertFalse(result.allowed)
        self.assertIn("excede el techo", result.reason)

    def test_motor_no_sandbox_only_rechaza(self):
        catalog = MotorCatalog({
            "motors": {
                "kimi": {"enabled": True, "sandbox_only": False},
            },
            "capabilities": {
                "implementation": {
                    "allowed_motors": ["kimi"], "allowed_callers": ["jacobs"],
                    "risk_level": "medium", "sandbox_only": True,
                    "requires_human_gate": False, "max_execution_minutes": 5,
                    "max_recursion_depth": 0, "output_schema": "",
                },
            },
        })
        result = MotorPolicy(catalog).check(
            caller="jacobs", capability="implementation", motor=None,
            context_keys=[], recursion_depth=0, human_gate_token=None,
        )
        self.assertFalse(result.allowed)
        self.assertIn("sandbox_only", result.reason)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Correr el test y confirmar que pasa contra el código actual (sin tocar)**

Run: `cd /home/fruiz/jax/las_manos && PYTHONPATH=/home/fruiz/jax/las_manos .venv/bin/python motor_registry/_policy_test.py -v`
Expected: `OK`, 7 tests, 0 failures. Si algo falla acá, el test está mal escrito contra el comportamiento real — corregir el test, NO el código (`policy.py` no se toca en esta task).

- [ ] **Step 3: Commit**

```bash
git add las_manos/motor_registry/_policy_test.py
git commit -m "test(motor-registry): caracteriza MotorPolicy.check() antes del split"
```

---

### Task 2: Partir `MotorPolicy.check()` en `check_capability_admission()` + wrapper

**Files:**
- Modify: `las_manos/motor_registry/policy.py`
- Modify: `las_manos/motor_registry/_policy_test.py` (agregar tests nuevos, NO tocar los de Task 1)

**Interfaces:**
- Consumes: `CapabilityEntry`, `MotorCatalog` (`catalog.py`, sin cambios).
- Produces: `MotorPolicy.check_capability_admission(caller: str, capability: str, context_keys: list[str], recursion_depth: int, human_gate_token: str | None) -> MotorPolicyResult` — usado por Task 6 (Jacobs).
  `MotorPolicy.check(...)` conserva firma y comportamiento exactos (Task 1 lo prueba).

- [ ] **Step 1: Escribir el test nuevo (debe fallar — la función no existe todavía)**

Agregar al FINAL de `las_manos/motor_registry/_policy_test.py` (no tocar las clases/tests existentes):

```python
class CheckCapabilityAdmissionTest(unittest.TestCase):
    """check_capability_admission() -- subconjunto de check() (checks 1-5,
    SIN motor resuelto ni techo de timeout). No requiere que exista un
    motor para la capability -- a diferencia de check(), que fallaria en
    el check 6 (resolver motor) para una capability sin allowed_motors."""

    def setUp(self):
        self.catalog = MotorCatalog({
            "motors": {},
            "capabilities": {
                "research": {
                    "allowed_motors": [],  # HTTP-directo: sin motor, a proposito
                    "allowed_callers": ["jacobs"],
                    "risk_level": "low", "sandbox_only": True,
                    "requires_human_gate": False, "max_execution_minutes": 5,
                    "max_recursion_depth": 0, "output_schema": "",
                },
            },
        })
        self.policy = MotorPolicy(self.catalog)

    def test_caller_autorizado_pasa_sin_necesitar_motor(self):
        result = self.policy.check_capability_admission(
            caller="jacobs", capability="research",
            context_keys=[], recursion_depth=0, human_gate_token=None,
        )
        self.assertTrue(result.allowed)

    def test_caller_no_autorizado_rechaza(self):
        result = self.policy.check_capability_admission(
            caller="jax_platform_chat", capability="research",
            context_keys=[], recursion_depth=0, human_gate_token=None,
        )
        self.assertFalse(result.allowed)
        self.assertIn("no autorizado", result.reason)

    def test_capability_desconocida_rechaza(self):
        result = self.policy.check_capability_admission(
            caller="jacobs", capability="no_existe",
            context_keys=[], recursion_depth=0, human_gate_token=None,
        )
        self.assertFalse(result.allowed)
        self.assertIn("desconocida", result.reason)

    def test_no_tiene_parametro_timeout_seconds(self):
        # El check 8 (techo) queda fuera de esta función a propósito
        # (decisión del spec, punto 2) -- confirmamos que la firma no lo
        # acepta, para que nadie lo reintroduzca sin querer.
        import inspect
        sig = inspect.signature(MotorPolicy.check_capability_admission)
        self.assertNotIn("timeout_seconds", sig.parameters)
        self.assertNotIn("motor", sig.parameters)
```

- [ ] **Step 2: Correr, confirmar que falla (la función no existe)**

Run: `cd /home/fruiz/jax/las_manos && PYTHONPATH=/home/fruiz/jax/las_manos .venv/bin/python motor_registry/_policy_test.py -v`
Expected: FAIL — `AttributeError: 'MotorPolicy' object has no attribute 'check_capability_admission'`

- [ ] **Step 3: Refactor de `policy.py`**

Reemplazar el cuerpo de `MotorPolicy.check()` (líneas 57-144 actuales) por:

```python
    def check_capability_admission(
        self,
        *,
        caller: str,
        capability: str,
        context_keys: list[str],
        recursion_depth: int,
        human_gate_token: str | None,
    ) -> MotorPolicyResult:
        """Checks 1-5 de check() -- capability existe, caller autorizado,
        human gate, recursion depth, claves prohibidas. NO incluye
        resolucion de motor (6-7) ni techo de timeout (8): ninguno aplica
        a un dispatch que no pasa por un motor resuelto (facets HTTP-
        directos, docs/superpowers/specs/2026-08-27-http-facets-motor-
        policy-governance-design.md, decision del punto 2 -- el techo NO
        se activa esta ronda). check() la llama como su primer paso;
        firma SIN motor/timeout_seconds a proposito, ningun check de acá
        los necesita."""
        cap = self._catalog.get_capability(capability)
        if cap is None:
            return MotorPolicyResult(False, f"Capability desconocida: '{capability}'")

        if caller not in cap.allowed_callers:
            return MotorPolicyResult(
                False,
                f"Caller '{caller}' no autorizado para '{capability}'. "
                f"Autorizados: {cap.allowed_callers}",
            )

        if cap.requires_human_gate and not (human_gate_token and human_gate_token.strip()):
            return MotorPolicyResult(
                False,
                f"Capability '{capability}' requiere human_gate_token y no fue provisto",
            )

        if recursion_depth > cap.max_recursion_depth:
            return MotorPolicyResult(
                False,
                f"recursion_depth={recursion_depth} excede máximo "
                f"{cap.max_recursion_depth} para '{capability}'",
            )

        bad_keys = [k for k in context_keys if k.lower() in FORBIDDEN_CONTEXT_KEYS]
        if bad_keys:
            return MotorPolicyResult(
                False,
                f"Context contiene claves prohibidas: {bad_keys}. "
                "Las manos no tocan secretos.",
            )

        return MotorPolicyResult(allowed=True, reason=f"OK: '{caller}' → '{capability}' (admisión)")

    def check(
        self,
        *,
        caller: str,
        capability: str,
        motor: str | None,
        context_keys: list[str],
        recursion_depth: int,
        human_gate_token: str | None,
        timeout_seconds: int | None = None,
    ) -> MotorPolicyResult:
        """Valida el dispatch completo. Devuelve al PRIMER fallo.
        Checks 1-5 delegados a check_capability_admission() -- ver esa
        docstring. 6-8 (resolver motor, sandbox_only, techo) sin cambios
        de esta refactorización (docs/superpowers/specs/2026-08-27-
        http-facets-motor-policy-governance-design.md, Requisito 1: cero
        cambio de comportamiento)."""
        admission = self.check_capability_admission(
            caller=caller, capability=capability, context_keys=context_keys,
            recursion_depth=recursion_depth, human_gate_token=human_gate_token,
        )
        if not admission.allowed:
            return admission

        cap = self._catalog.get_capability(capability)

        resolved = self._resolve_motor(motor, cap)
        if resolved is None:
            return MotorPolicyResult(
                False,
                f"No hay motor habilitado disponible para '{capability}'. "
                f"Motores permitidos: {cap.allowed_motors}",
            )

        motor_entry = self._catalog.get_motor(resolved)
        if motor_entry and not motor_entry.sandbox_only:
            return MotorPolicyResult(
                False,
                f"Motor '{resolved}' no es sandbox_only. "
                "Motor Registry v0.1 solo admite motores sandbox.",
            )

        if timeout_seconds is not None:
            max_seconds = cap.max_execution_minutes * 60
            if timeout_seconds > max_seconds:
                return MotorPolicyResult(
                    False,
                    f"timeout_seconds={timeout_seconds} excede el techo de "
                    f"'{capability}' ({cap.max_execution_minutes} min = {max_seconds}s)",
                )

        return MotorPolicyResult(
            allowed=True,
            reason=f"OK: '{caller}' → '{capability}' vía motor '{resolved}'",
            resolved_motor=resolved,
            requires_human_gate=cap.requires_human_gate,
        )
```

No tocar `_resolve_motor()` ni el docstring del módulo (actualizarlo solo para mencionar el split, sin cambiar la lista de 8 checks que documenta).

- [ ] **Step 4: Correr TODOS los tests del archivo — Task 1 + Task 2, todos deben pasar**

Run: `cd /home/fruiz/jax/las_manos && PYTHONPATH=/home/fruiz/jax/las_manos .venv/bin/python motor_registry/_policy_test.py -v`
Expected: `OK`, 11 tests (7 de Task 1 + 4 de Task 2), 0 failures. Los 7 de Task 1 pasan **sin haber sido modificados** — es la prueba del Requisito 1. Si alguno de los 7 falla, PARAR y reportar, no editarlo para que pase.

- [ ] **Step 5: Commit**

```bash
git add las_manos/motor_registry/policy.py las_manos/motor_registry/_policy_test.py
git commit -m "refactor(motor-registry): parte MotorPolicy.check() en check_capability_admission() + wrapper"
```

---

### Task 3: DB migration — `facet.allowed_callers` + `capability.sandbox_only` vestigial (jax-platform)

**Files:**
- Modify: `jax-platform/backend/db/migrations.py`
- Create: `jax-platform/backend/tests/test_facet_allowed_callers_migration.py`

**Interfaces:**
- Produces: columna `facet.allowed_callers` (NULLABLE, JSON), poblada para `hipatia`/`jekyll`/`thot`/`ada` con `["jacobs", "jax_platform_chat"]`. Consumida por `check_facet_admission()` (Task 4, repo `jax`, misma DB `jax_memory`).

Esta migración va ANTES del lector (Task 4) a propósito — Requisito explícito: `facet.allowed_callers` nace con su lector, no como `capability.sandbox_only` (columna sin lector desde el día uno, ver Task 8).

- [ ] **Step 1: DDL — agregar la columna a `CREATE_FACET`**

En `CREATE_FACET` (línea 222-238), agregar antes de `status`:

```python
CREATE_FACET = """
CREATE TABLE IF NOT EXISTS facet (
  `key` VARCHAR(50) NOT NULL PRIMARY KEY,
  display_name VARCHAR(100) NOT NULL,
  icon VARCHAR(10) NULL,
  color_hex VARCHAR(7) NULL,
  persona TEXT NULL,
  transport ENUM('http_openai_compat','http_gemini','motor_registry','ollama','subprocess') NOT NULL,
  requires_tool_use BOOLEAN NOT NULL DEFAULT FALSE,
  requires_structured_output BOOLEAN NOT NULL DEFAULT FALSE,
  min_context_tokens INT NOT NULL DEFAULT 0,
  max_latency_ms INT NULL,
  max_cost_per_1k_usd DECIMAL(10,6) NULL,
  auto_selectable BOOLEAN NOT NULL DEFAULT TRUE,
  allowed_callers LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL CHECK (allowed_callers IS NULL OR json_valid(allowed_callers)),
  status ENUM('active','degraded','disabled') NOT NULL DEFAULT 'active',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""
```

La tabla ya existe en producción (`CREATE TABLE IF NOT EXISTS` no la altera) — la columna real se agrega vía `_COLUMNS`/`_column_exists()`, mismo patrón que el resto del archivo. Ubicar la lista `_COLUMNS` (buscar `_COLUMNS = [` cerca de la lista `_TABLES`) y agregar:

```python
    ("facet", "allowed_callers",
     "ALTER TABLE facet ADD COLUMN allowed_callers LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL "
     "CHECK (allowed_callers IS NULL OR json_valid(allowed_callers))"),
```

- [ ] **Step 2: Comentario vestigial en `CREATE_CAPABILITY`**

En `CREATE_CAPABILITY` (línea 340-356), agregar comentario ARRIBA de la línea `sandbox_only`:

```python
CREATE_CAPABILITY = """
CREATE TABLE IF NOT EXISTS capability (
  `key` VARCHAR(50) NOT NULL PRIMARY KEY,
  risk_level ENUM('low','medium','high') NOT NULL,
  -- VESTIGIAL (verificado 2026-08-27, ver DEUDA.md): ningun lector en el
  -- codigo real compara este valor contra nada. El sandbox_only que SI se
  -- enforce es motor.sandbox_only (columna distinta, tabla motor). No
  -- confiar en este valor -- pendiente decidir lector real o drop.
  sandbox_only BOOLEAN NOT NULL DEFAULT TRUE,
  requires_human_gate BOOLEAN NOT NULL DEFAULT FALSE,
  ...
```

(Nota para quien ejecute esta task: el comentario SQL de línea `--` dentro de un docstring de Python triple-comillas es válido — MariaDB lo interpreta como comentario SQL real al ejecutar el `CREATE TABLE`.)

- [ ] **Step 3: Migración idempotente del seed de `allowed_callers`**

Agregar función nueva, después de `_eliminate_motor_model_ref_denormalization` (mismo estilo que `_fix_file_write_gate_and_auditor`):

```python
async def _seed_http_facet_allowed_callers(cur) -> None:
    """Gobernanza de _HTTP_FACETS (docs/superpowers/specs/2026-08-27-
    http-facets-motor-policy-governance-design.md): hipatia/jekyll/thot/
    ada quedan con allowed_callers=["jacobs","jax_platform_chat"] --
    mismo acceso que ya existia informalmente (ninguno de los dos estaba
    bloqueado antes de esta ronda), ahora explicito. kimi/jax_local/hyde
    quedan NULL a proposito -- fuera de alcance esta ronda, fail-closed
    por diseno (ver facet_policy.py::check_facet_admission en el repo jax).

    Guard WHERE allowed_callers IS NULL: no pisa un valor manual futuro
    si alguien ya lo configuro distinto."""
    await cur.execute(
        "UPDATE facet SET allowed_callers = %s "
        "WHERE `key` IN ('hipatia','jekyll','thot','ada') AND allowed_callers IS NULL",
        (json.dumps(["jacobs", "jax_platform_chat"]),),
    )
```

Registrar en `run_migrations()`, después de `_eliminate_motor_model_ref_denormalization(cur)`:

```python
            await _seed_http_facet_allowed_callers(cur)
```

- [ ] **Step 4: Test — idempotencia + valores correctos**

```python
"""facet.allowed_callers se siembra para los 4 facets HTTP-directos,
idempotente (correr dos veces no duplica ni pisa un valor manual)."""
import json

import pytest

from db.migrations import _seed_http_facet_allowed_callers


@pytest.mark.asyncio
async def test_seed_sets_allowed_callers_for_the_4_http_facets(db_cursor):
    await _seed_http_facet_allowed_callers(db_cursor)
    await db_cursor.execute(
        "SELECT `key`, allowed_callers FROM facet WHERE `key` IN "
        "('hipatia','jekyll','thot','ada') ORDER BY `key`"
    )
    rows = {key: json.loads(val) for key, val in await db_cursor.fetchall()}
    for facet_key in ("hipatia", "jekyll", "thot", "ada"):
        assert rows[facet_key] == ["jacobs", "jax_platform_chat"]


@pytest.mark.asyncio
async def test_seed_does_not_overwrite_manual_value(db_cursor):
    await db_cursor.execute(
        "UPDATE facet SET allowed_callers = %s WHERE `key` = 'hipatia'",
        (json.dumps(["solo_jacobs"]),),
    )
    await _seed_http_facet_allowed_callers(db_cursor)
    await db_cursor.execute("SELECT allowed_callers FROM facet WHERE `key` = 'hipatia'")
    (val,) = await db_cursor.fetchone()
    assert json.loads(val) == ["solo_jacobs"]


@pytest.mark.asyncio
async def test_seed_leaves_out_of_scope_facets_null(db_cursor):
    await _seed_http_facet_allowed_callers(db_cursor)
    await db_cursor.execute(
        "SELECT allowed_callers FROM facet WHERE `key` IN ('kimi','jax_local','hyde')"
    )
    for (val,) in await db_cursor.fetchall():
        assert val is None
```

Verificar en `backend/tests/conftest.py` si existe un fixture `db_cursor` reusable (buscar `def db_cursor` o `def cursor` en ese archivo); si no existe, usar el mismo patrón de conexión que otros tests de migración en el archivo (ej. `test_admin_models_endpoints.py`, que ya ejercita `_eliminate_motor_model_ref_denormalization`) — copiar su fixture de DB, no inventar uno nuevo.

- [ ] **Step 5: Correr**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/python -m pytest tests/test_facet_allowed_callers_migration.py -v`
Expected: 3 passed.

- [ ] **Step 6: Aplicar la migración contra la DB real de desarrollo**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/python -c "import asyncio; from db.migrations import run_migrations; asyncio.run(run_migrations())"`
Expected: sin errores. Confirmar en vivo:
`mysql ... -e "SELECT \`key\`, allowed_callers FROM facet;"` — los 4 facets muestran `["jacobs", "jax_platform_chat"]`, los otros 3 siguen `NULL`.

Este paso es lo que hace que la Task 4 (el lector) tenga datos reales contra los cuales correr sus tests — sin este Step 6, la Task 4 va a fallar con "no tiene allowed_callers configurado" para `hipatia`, y eso es correcto: significa que esta task no terminó, no que la Task 4 esté mal escrita.

- [ ] **Step 7: Commit**

```bash
git add backend/db/migrations.py backend/tests/test_facet_allowed_callers_migration.py
git commit -m "feat(db): facet.allowed_callers -- gobernanza de nivel facet para _HTTP_FACETS"
```

---

### Task 4: `check_facet_admission()` — check de nivel FACET para Mesa web

**Files:**
- Create: `las_manos/motor_registry/facet_policy.py`
- Create: `las_manos/motor_registry/_facet_policy_test.py`

**Interfaces:**
- Consumes: tabla `facet`, columna `allowed_callers` (agregada y sembrada en Task 3 — ya aplicada contra la DB de desarrollo al llegar acá).
- Produces: `async def check_facet_admission(caller: str, facet: str) -> tuple[bool, str]` — usado por Task 5 (endpoint).

- [ ] **Step 1: Escribir el test (falla — el módulo no existe)**

```python
#!/usr/bin/env python3
"""check_facet_admission() -- gobernanza de nivel FACET para callers que
NO tienen concepto de capability (Mesa web -- ver docs/superpowers/specs/
2026-08-27-http-facets-motor-policy-governance-design.md, seccion 3, la
correccion sobre por que esto NO reusa check_capability_admission()).

Corre contra la DB real de desarrollo, mismo criterio que
_catalog_from_db_test.py -- sin mock de aiomysql. Requiere que la Task 3
(migracion de facet.allowed_callers) ya haya corrido contra esta DB.

Corre desde /home/fruiz/jax/las_manos con:
  PYTHONPATH=/home/fruiz/jax/las_manos \
  /home/fruiz/jax/las_manos/.venv/bin/python \
  /home/fruiz/jax/las_manos/motor_registry/_facet_policy_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_DB_NAME", os.environ.get("JAX_DB_NAME", "jax_memory"))

from motor_registry.facet_policy import check_facet_admission


class CheckFacetAdmissionTest(unittest.IsolatedAsyncioTestCase):
    async def test_caller_autorizado_pasa(self):
        # hipatia se siembra con allowed_callers incluyendo "jacobs" (Task 3).
        allowed, reason = await check_facet_admission("jacobs", "hipatia")
        self.assertTrue(allowed, reason)

    async def test_caller_no_autorizado_rechaza(self):
        allowed, reason = await check_facet_admission("caller_fantasma", "hipatia")
        self.assertFalse(allowed)
        self.assertIn("no autorizado", reason)

    async def test_facet_sin_allowed_callers_configurado_rechaza(self):
        # jax_local/kimi/hyde quedan NULL a propósito (fuera de alcance,
        # spec seccion 3) -- NULL debe denegar, no "lista vacia = todos".
        allowed, reason = await check_facet_admission("jacobs", "jax_local")
        self.assertFalse(allowed)
        self.assertIn("no configurado", reason)

    async def test_facet_inexistente_rechaza(self):
        allowed, reason = await check_facet_admission("jacobs", "no_existe")
        self.assertFalse(allowed)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Correr, confirmar que falla**

Run: `cd /home/fruiz/jax/las_manos && PYTHONPATH=/home/fruiz/jax/las_manos .venv/bin/python motor_registry/_facet_policy_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'motor_registry.facet_policy'`

- [ ] **Step 3: Implementar `facet_policy.py`**

```python
"""LAS MANOS — Motor Registry: gobernanza de nivel FACET.

check_capability_admission()/check() (policy.py) gobiernan dispatch por
CAPABILITY -- tiene sentido para un step de pipeline con un objetivo
concreto. Mesa web no tiene eso: un turno de chat es texto libre enrutado
a un facet por keyword-matching, sin capability asociada (verificado
leyendo jax-platform/backend/api/chat.py completo -- ver
docs/superpowers/specs/2026-08-27-http-facets-motor-policy-governance-
design.md, seccion 3). check_facet_admission() responde la pregunta que
SI tiene sentido para ese camino: "¿puede este caller hablar con este
facet?" -- nada mas. No toca la tabla `capability`.

Modulo con I/O real (a diferencia de policy.py, "puro, sin I/O") --
consulta facet.allowed_callers directo, mismo patron de conexion que
facet_resolver.py::_db_conn().

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import json
import os

import aiomysql


async def _db_conn() -> aiomysql.Connection:
    host = os.environ.get("JAX_DB_HOST")
    port = os.environ.get("JAX_DB_PORT")
    if not host or not port:
        raise RuntimeError(
            "JAX_DB_HOST/JAX_DB_PORT no están seteados -- sin default "
            "silencioso a localhost:3306 (esa instancia está muerta, ver "
            "memoria jax-dual-mariadb-instances). Sourceá /etc/jax/.env o "
            "exportalos a mano antes de conectar."
        )
    return await aiomysql.connect(
        host=host,
        port=int(port),
        user=os.getenv("JAX_DB_USER", ""),
        password=os.getenv("JAX_DB_PASSWORD", ""),
        db=os.getenv("JAX_DB_NAME", "jax_memory"),
        charset="utf8mb4",
        autocommit=True,
    )


async def check_facet_admission(caller: str, facet: str) -> tuple[bool, str]:
    """Fail-closed: facet inexistente, allowed_callers NULL, o caller
    ausente de la lista -> (False, razon). Nunca deja pasar por duda."""
    conn = await _db_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT allowed_callers FROM facet WHERE `key`=%s",
                (facet,),
            )
            row = await cur.fetchone()
    finally:
        conn.close()

    if row is None:
        return False, f"facet desconocido: '{facet}'"

    (allowed_callers_raw,) = row
    if allowed_callers_raw is None:
        return False, f"facet '{facet}' no tiene allowed_callers configurado -- fail-closed"

    allowed_callers = json.loads(allowed_callers_raw)
    if caller not in allowed_callers:
        return False, f"caller '{caller}' no autorizado para facet '{facet}'. Autorizados: {allowed_callers}"

    return True, f"OK: '{caller}' → facet '{facet}'"
```

- [ ] **Step 4: Correr, confirmar que pasa**

Run: `cd /home/fruiz/jax/las_manos && PYTHONPATH=/home/fruiz/jax/las_manos .venv/bin/python motor_registry/_facet_policy_test.py -v`
Expected: `OK`, 4 tests. Si `test_caller_autorizado_pasa` falla con "no tiene allowed_callers configurado", la Task 3 (Step 6) no corrió contra esta DB — es un problema de la Task 3, no de este código; volver ahí, no editar este test.

- [ ] **Step 5: Commit**

```bash
git add las_manos/motor_registry/facet_policy.py las_manos/motor_registry/_facet_policy_test.py
git commit -m "feat(motor-registry): check_facet_admission -- gobernanza de nivel facet para callers sin capability"
```

---

### Task 5: `POST /motor/authorize-facet` — endpoint fail-closed

**Files:**
- Modify: `las_manos/motor_registry/routes.py`
- Create: `las_manos/motor_registry/_authorize_facet_endpoint_test.py`

**Interfaces:**
- Consumes: `check_facet_admission()` (Task 4).
- Produces: `POST /motor/authorize-facet` — request `{"caller": str, "facet": str}`, response `{"allowed": bool, "reason": str}`. Usado por Task 7 (Mesa web).

- [ ] **Step 1: Escribir el test (falla — el endpoint no existe)**

```python
#!/usr/bin/env python3
"""POST /motor/authorize-facet -- endpoint test end-to-end (FastAPI
TestClient real, sin mockear check_facet_admission: si la DB no responde,
este test lo va a mostrar, que es exactamente lo que queremos saber).

Corre desde /home/fruiz/jax/las_manos con:
  PYTHONPATH=/home/fruiz/jax/las_manos \
  /home/fruiz/jax/las_manos/.venv/bin/python \
  /home/fruiz/jax/las_manos/motor_registry/_authorize_facet_endpoint_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import unittest

from fastapi.testclient import TestClient
from server import app


class AuthorizeFacetEndpointTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_caller_autorizado_devuelve_allowed_true(self):
        resp = self.client.post(
            "/motor/authorize-facet",
            json={"caller": "jacobs", "facet": "hipatia"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["allowed"])

    def test_caller_no_autorizado_devuelve_allowed_false(self):
        resp = self.client.post(
            "/motor/authorize-facet",
            json={"caller": "caller_fantasma", "facet": "hipatia"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["allowed"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Correr, confirmar que falla**

Run: `cd /home/fruiz/jax/las_manos && PYTHONPATH=/home/fruiz/jax/las_manos .venv/bin/python motor_registry/_authorize_facet_endpoint_test.py -v`
Expected: FAIL — 404 (ruta no existe).

- [ ] **Step 3: Agregar el endpoint a `routes.py`**

Agregar import al bloque de imports existente (línea 26-33):

```python
from motor_registry.facet_policy import check_facet_admission
```

Agregar modelo de request/response en `las_manos/motor_registry/models.py` (junto a `MotorDispatchRequest`):

```python
class FacetAuthorizeRequest(BaseModel):
    caller: str
    facet: str


class FacetAuthorizeResponse(BaseModel):
    allowed: bool
    reason: str
```

Agregar endpoint a `routes.py`, después de `dispatch()`:

```python
from motor_registry.models import FacetAuthorizeRequest, FacetAuthorizeResponse


@router.post("/authorize-facet", response_model=FacetAuthorizeResponse)
async def authorize_facet(req: FacetAuthorizeRequest) -> FacetAuthorizeResponse:
    """Sincrono, sin job ni polling -- solo corre check_facet_admission()
    y devuelve el veredicto. Usado por jax-platform (Mesa web) antes de
    despachar a un facet HTTP-directo -- ver docs/superpowers/specs/
    2026-08-27-http-facets-motor-policy-governance-design.md."""
    allowed, reason = await check_facet_admission(req.caller, req.facet)
    return FacetAuthorizeResponse(allowed=allowed, reason=reason)
```

- [ ] **Step 4: Correr, confirmar que pasa**

Run: `cd /home/fruiz/jax/las_manos && PYTHONPATH=/home/fruiz/jax/las_manos .venv/bin/python motor_registry/_authorize_facet_endpoint_test.py -v`
Expected: `OK`, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add las_manos/motor_registry/routes.py las_manos/motor_registry/models.py las_manos/motor_registry/_authorize_facet_endpoint_test.py
git commit -m "feat(motor-registry): endpoint POST /motor/authorize-facet"
```

---

### Task 6: Jacobs — `validate_capability()` gobierna `_HTTP_FACETS`

**Files:**
- Modify: `jacobs/executor.py`
- Modify: `tests/test_validate_capability_typed.py` (agregar, sin tocar los tests existentes)

**Interfaces:**
- Consumes: `MotorPolicy.check_capability_admission()` (Task 2), `store.get_motor_governance()` (sin cambios).
- Produces: `validate_capability()` rechaza steps HTTP-directos con caller no autorizado (mismo contrato de retorno: `str` para rechazo NIVEL A/B, no reenrutable). Fail-closed también si `MotorCatalog.from_db()` no puede leer la DB (ver Step 3-4).

**Modo de fallo si la DB no responde (verificado, no asumido por analogía):**
`MotorCatalog.from_db()` (`las_manos/motor_registry/catalog.py:146-169`) no atrapa ninguna excepción — si `JAX_DB_HOST`/`JAX_DB_PORT` faltan, lanza `RuntimeError` explícito; si `aiomysql.connect()` falla, esa excepción se propaga tal cual. Esa excepción sube por `validate_capability()` (sin try/except propio en este punto) → `_dispatch_step()` (tampoco la atrapa) → `_run_one_step()`, que SÍ la atrapa (`except Exception as exc: await _fail_step(...)`) y marca el step `failed` con el motivo real. Resultado: **DB caída → step falla, nunca despacha** — mismo patrón fail-closed que ya usa el resto de `validate_capability()` (documentado en su propio docstring para NIVEL A/B). El Step 4 de esta task agrega el test que lo confirma en vivo, no solo por lectura de código.

**Costo medido en vivo (no por analogía con otro call site):** 5 corridas reales contra la DB de hall9000, `MotorCatalog.from_db()` solo: `[0.0021, 0.0020, 0.0018, 0.0010, 0.0011]` segundos, promedio **1.59ms** (incluye abrir una conexión `aiomysql` nueva sin pooling + 3 SELECTs: `motor`, `capability`, `capability_motor`). Insignificante para un dispatch HTTP que después tarda segundos en la llamada real al proveedor.

- [ ] **Step 1: Escribir el test (falla — el nuevo check no corre todavía)**

Agregar al final de `tests/test_validate_capability_typed.py`:

```python
def test_http_facet_caller_no_autorizado_es_rechazado():
    """NIVEL C nuevo (esta ronda): un facet HTTP-directo (hipatia/jekyll/
    thot/ada) con un caller fuera de allowed_callers debe rechazarse --
    antes de esta ronda este chequeo no corria en absoluto para estos 4
    facets (ver DEUDA.md, bullet _HTTP_FACETS sin gobernanza)."""
    from motor_registry.catalog import MotorCatalog
    from motor_registry.policy import MotorPolicy

    catalog = MotorCatalog({
        "motors": {},
        "capabilities": {
            "research": {
                "allowed_motors": [], "allowed_callers": ["jacobs"],
                "risk_level": "low", "sandbox_only": True,
                "requires_human_gate": False, "max_execution_minutes": 5,
                "max_recursion_depth": 0, "output_schema": "",
            },
        },
    })
    policy = MotorPolicy(catalog)
    result = policy.check_capability_admission(
        caller="caller_no_autorizado", capability="research",
        context_keys=[], recursion_depth=0, human_gate_token=None,
    )
    assert not result.allowed
    assert "no autorizado" in result.reason
```

(Este test ejercita `check_capability_admission()` directo, mismo patrón de "réplica aislada" que el resto del archivo — evita importar `jacobs.executor` directo, que dispara I/O de red/DB al import. La integración real se confirma en el Step 4-5 de abajo, corriendo `validate_capability()` de verdad.)

- [ ] **Step 2: Correr, confirmar que pasa (la función ya existe desde Task 2 — este test documenta el contrato, no bloquea)**

Run: `cd /home/fruiz/jax && .venv/bin/python -m pytest tests/test_validate_capability_typed.py -v`
Expected: pasa junto con los tests existentes del archivo.

- [ ] **Step 3: Modificar `validate_capability()` en `jacobs/executor.py`**

Ubicar el bloque NIVEL A actual (líneas 641-644) y agregar el nuevo check inmediatamente después, ANTES del bloque NIVEL B existente:

```python
    # ---- NIVEL A: existencia real en la DB (TODOS los facets) ----
    entry = caps.get(cap)
    if entry is None:
        return f"capability desconocida: '{cap}' no está en la tabla `capability`"

    # ---- NIVEL C (2026-08-27): admisión para _HTTP_FACETS ----
    # Antes de esta ronda, hipatia/jekyll/thot/ada nunca pasaban por
    # ninguno de los checks de MotorPolicy (DEUDA.md, bullet _HTTP_FACETS
    # sin gobernanza). check_capability_admission() cubre allowed_callers/
    # requires_human_gate/recursion_depth/claves prohibidas -- import
    # directo, Jacobs corre en el mismo proceso que las_manos (docs/
    # superpowers/specs/2026-08-27-http-facets-motor-policy-governance-
    # design.md). NO incluye techo de timeout (decisión explícita, punto 2
    # del spec) ni resolución de motor (N/A -- facets HTTP no son motores).
    # Fail-closed: si MotorCatalog.from_db() no puede leer la DB, la
    # excepción se propaga sin capturarla acá (mismo criterio P10 que el
    # resto de esta función) -- _run_one_step falla el step limpio, nunca
    # despacha sin haber podido verificar.
    if step.facet in _HTTP_FACETS:
        from motor_registry.catalog import MotorCatalog
        from motor_registry.policy import MotorPolicy
        catalog = await MotorCatalog.from_db()
        policy = MotorPolicy(catalog)
        admission = policy.check_capability_admission(
            caller="jacobs", capability=cap,
            context_keys=list(step.input.keys()),
            recursion_depth=0, human_gate_token=None,
        )
        if not admission.allowed:
            return admission.reason

    # ---- NIVEL B: contrato de motor (SOLO facets-motor, hoy kimi/jax_local) ----
    if step.facet in _MOTOR_FACETS:
```

- [ ] **Step 4: Tests de integración real — camino feliz Y fail-closed cuando la DB no responde**

```python
#!/usr/bin/env python3
"""validate_capability() rechaza un facet HTTP-directo cuando el caller
'jacobs' no está en allowed_callers -- contra la DB real de test, sin
mockear -- y falla cerrado (nunca dispatcha) si MotorCatalog.from_db()
no puede leer la DB. Ver Task 6 del plan de gobernanza de _HTTP_FACETS.

Corre desde /home/fruiz/jax con:
  PYTHONPATH=/home/fruiz/jax .venv/bin/python -m unittest jacobs._http_facet_admission_test

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, patch

_existing_db_name = os.environ.get("JAX_DB_NAME")
if _existing_db_name and _existing_db_name != "jax_memory_test":
    raise RuntimeError(
        f"JAX_DB_NAME={_existing_db_name!r} ya está seteado -- este test "
        f"corre contra jax_memory_test, no contra esa DB."
    )
os.environ.setdefault("JAX_DB_NAME", "jax_memory_test")

from jacobs.executor import validate_capability
from jacobs.models import Step


class HttpFacetAdmissionTest(unittest.IsolatedAsyncioTestCase):
    async def test_research_con_caller_jacobs_pasa(self):
        # 'research' (hipatia) tiene allowed_callers=["jacobs"] sembrado
        # -- jacobs SIEMPRE fue el caller real de este camino.
        step = Step(facet="hipatia", capability="research", input={"prompt": "x"})
        result = await validate_capability(step)
        self.assertIsNone(result)


class HttpFacetAdmissionFailClosedTest(unittest.IsolatedAsyncioTestCase):
    async def test_db_caida_al_leer_catalogo_no_deja_pasar_el_step(self):
        """Simula MotorCatalog.from_db() fallando (DB caida/timeout real,
        no un mock que devuelve un error prolijo) -- confirma que
        validate_capability() PROPAGA la excepcion en vez de tragarla, que
        es lo que _run_one_step necesita para fallar el step cerrado."""
        step = Step(facet="hipatia", capability="research", input={"prompt": "x"})
        with patch(
            "motor_registry.catalog.MotorCatalog.from_db",
            new=AsyncMock(side_effect=RuntimeError("DB no responde (simulado)")),
        ):
            with self.assertRaises(RuntimeError):
                await validate_capability(step)


if __name__ == "__main__":
    unittest.main()
```

Este archivo confirma el camino feliz contra datos reales (jax_memory_test debe tener el seed de `capability` aplicado — si `research` no existe ahí, correr las migraciones de jax-platform contra esa DB primero, mismo requisito que ya tienen los tests vecinos de este directorio) y el camino fail-closed sin necesitar apagar una DB de verdad.

- [ ] **Step 5: Correr ambos**

Run: `cd /home/fruiz/jax && .venv/bin/python -m pytest tests/test_validate_capability_typed.py -v && PYTHONPATH=/home/fruiz/jax .venv/bin/python -m unittest jacobs._http_facet_admission_test -v`
Expected: todo `OK`/passed — 2 tests totales en el archivo (1 de `HttpFacetAdmissionTest` + 1 de `HttpFacetAdmissionFailClosedTest`).

- [ ] **Step 6: Commit**

```bash
git add jacobs/executor.py tests/test_validate_capability_typed.py jacobs/_http_facet_admission_test.py
git commit -m "feat(jacobs): validate_capability gobierna _HTTP_FACETS via check_capability_admission, fail-closed"
```

---

### Task 7: Mesa web — `_invoke_facet` gobierna los 4 facets HTTP-directos, fail-closed

**Files:**
- Modify: `jax-platform/backend/api/chat.py`
- Modify: `jax-platform/backend/tests/test_chat_facet_validation.py`

**Interfaces:**
- Consumes: `POST /motor/authorize-facet` (Task 5), vía `http_client.get_http_client()` (patrón ya usado en `chat.py`).
- Produces: `_invoke_facet` deniega ANTES de dispatchar a `hipatia`/`jekyll`/`thot`/`ada` si `jax_platform_chat` no está autorizado, o si `las_manos` no responde.

**Shape de la respuesta de Gemini — confirmado contra el código real ANTES de escribir el test (Requisito explícito, no inferencia):**
`_call_gemini()` (`backend/api/chat.py:552-579`) parsea la respuesta como
`data["candidates"][0]["content"]["parts"][0]["text"]`, y el uso como
`data.get("usageMetadata") or {}` → `usage.get("promptTokenCount", 0)` /
`usage.get("candidatesTokenCount", 0)`. El fake del Step 1 usa exactamente
este shape.

- [ ] **Step 1: Escribir los tests (fallan — el pre-check no existe todavía)**

Agregar a `backend/tests/test_chat_facet_validation.py`:

```python
import httpx
import http_client
from tests.test_chat_contract_wrapper import _FakePostClient, _FakeResponse


class _FakeFailingPostClient:
    """Simula las_manos caido de verdad -- conexion rechazada, no un
    mock que devuelve un error prolijo. Es el test que prueba que el
    gate gatea cuando mas importa (Requisito 3 del spec)."""
    async def post(self, url, **kwargs):
        raise httpx.ConnectError("Connection refused", request=None)


def test_chat_endpoint_denies_hipatia_when_authorize_facet_returns_false(client):
    token = create_access_token("test-authz-denied-user", "test-authz-denied-tenant", "operator")
    fake = _FakePostClient(_FakeResponse({"allowed": False, "reason": "caller no autorizado"}))
    original = http_client._client
    http_client._client = fake
    try:
        resp = client.post(
            "/api/chat",
            json={"message": "hola", "facet": "hipatia"},
            headers={"Authorization": f"Bearer {token}"},
        )
    finally:
        http_client._client = original
    assert resp.status_code == 200
    assert "no autorizado" in resp.json()["response"] or "no disponible" in resp.json()["response"]


def test_chat_endpoint_denies_hipatia_when_las_manos_is_down(client):
    """El caso critico: las_manos no responde en absoluto (ConnectError,
    no un 4xx/5xx prolijo). Fail-closed exige que esto tambien deniegue,
    no que se despache igual porque "no se pudo verificar"."""
    token = create_access_token("test-authz-down-user", "test-authz-down-tenant", "operator")
    original = http_client._client
    http_client._client = _FakeFailingPostClient()
    try:
        resp = client.post(
            "/api/chat",
            json={"message": "hola", "facet": "hipatia"},
            headers={"Authorization": f"Bearer {token}"},
        )
    finally:
        http_client._client = original
    assert resp.status_code == 200
    body = resp.json()["response"]
    assert "no autorizado" in body or "no disponible" in body


def test_chat_endpoint_allows_hipatia_when_authorize_facet_returns_true(client):
    token = create_access_token("test-authz-allowed-user", "test-authz-allowed-tenant", "operator")

    class _SequencedFakeClient:
        """Primera llamada = /motor/authorize-facet (allowed=True), segunda
        = la llamada real al proveedor del facet -- shape confirmado
        contra _call_gemini() (chat.py:552-579)."""
        def __init__(self):
            self._calls = 0

        async def post(self, url, **kwargs):
            self._calls += 1
            if "/motor/authorize-facet" in url:
                return _FakeResponse({"allowed": True, "reason": "OK"})
            return _FakeResponse({
                "candidates": [{"content": {"parts": [{"text": "hola desde hipatia"}]}}],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
            })

    original = http_client._client
    http_client._client = _SequencedFakeClient()
    try:
        resp = client.post(
            "/api/chat",
            json={"message": "hola", "facet": "hipatia"},
            headers={"Authorization": f"Bearer {token}"},
        )
    finally:
        http_client._client = original
    assert resp.status_code == 200
```

- [ ] **Step 2: Correr, confirmar que fallan (o pasan por casualidad si el fallback silencioso de hoy ya las deja pasar — confirmar CUÁL es el comportamiento actual antes de asumir "falla")**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/python -m pytest tests/test_chat_facet_validation.py -v -k authz`
Expected: los tres tests nuevos fallan (hoy no hay ningún pre-check, así que `test_chat_endpoint_denies_...` fallaría porque el mensaje SÍ se despacha). Si el resultado real difiere de este Expected, PARAR y reportar antes de seguir al Step 3 — no asumir que el Expected estaba bien.

- [ ] **Step 3: Implementar el pre-check en `_invoke_facet`**

Agregar constante cerca del import de `resolve_facet` en `chat.py`:

```python
_GOVERNED_HTTP_FACETS = frozenset({"hipatia", "jekyll", "thot", "ada"})
_JAX_PLATFORM_CHAT_CALLER = "jax_platform_chat"
```

Modificar `_invoke_facet`, inmediatamente después de `f = await resolve_facet(facet)` y su `except FacetUnavailableError`:

```python
    try:
        f = await resolve_facet(facet)
    except FacetUnavailableError:
        return f"⚠️ {facet} no está disponible: sin binding activo configurado.", None

    if facet in _GOVERNED_HTTP_FACETS:
        allowed = False
        try:
            hc = await get_http_client()
            resp = await hc.post(
                "http://127.0.0.1:7777/motor/authorize-facet",
                json={"caller": _JAX_PLATFORM_CHAT_CALLER, "facet": facet},
                timeout=5.0,
            )
            resp.raise_for_status()
            allowed = resp.json().get("allowed", False)
        except Exception:
            # Fail-closed (P10): cualquier falla -- timeout, conexión
            # rechazada, respuesta inesperada -- deniega. Nunca "no pude
            # verificar, sigo igual".
            allowed = False
        if not allowed:
            return f"⚠️ {facet} no está disponible: caller '{_JAX_PLATFORM_CHAT_CALLER}' no autorizado.", None
```

- [ ] **Step 4: Correr todos los tests del archivo — nuevos y existentes**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/python -m pytest tests/test_chat_facet_validation.py -v`
Expected: todos pasan, incluidos los tests pre-existentes del archivo (`test_chat_endpoint_rejects_overlong_unknown_facet`, etc.) sin haberlos tocado.

- [ ] **Step 5: Correr la suite completa de `chat.py` para confirmar que no se rompió nada tangencial**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/python -m pytest tests/test_chat_contract_wrapper.py tests/test_chat_resolved_version_capture.py tests/test_chat_usage_capture.py -v`
Expected: todos pasan.

- [ ] **Step 6: Commit**

```bash
git add backend/api/chat.py backend/tests/test_chat_facet_validation.py
git commit -m "feat(chat): _invoke_facet gobierna hipatia/jekyll/thot/ada via /motor/authorize-facet, fail-closed"
```

---

### Task 8: `DEUDA.md` — cierre del bullet (jax repo)

**Files:**
- Modify: `DEUDA.md`

**Interfaces:** ninguna — solo documentación.

- [ ] **Step 1: Reemplazar el bullet `_HTTP_FACETS sin gobernanza del Motor Registry`**

Localizar el bullet actual (`grep -n "_HTTP_FACETS sin gobernanza" DEUDA.md`) y reemplazar su contenido completo por el texto de cierre de la sección "`DEUDA.md` — actualización al cerrar" del spec (`docs/superpowers/specs/2026-08-27-http-facets-motor-policy-governance-design.md`), citando los PRs reales una vez existan (dejar el placeholder de número de PR para completar al mergear — este es el único lugar del plan donde un número de PR no puede conocerse de antemano; todo lo demás es texto final).

- [ ] **Step 2: Agregar entrada nueva — `capability.sandbox_only` vestigial**

En la sección "Anotado, no bloquea" de `DEUDA.md`:

```markdown
- **`capability.sandbox_only` — columna sin lector, vestigial.** Verificado
  2026-08-27: `grep -rn "cap.sandbox_only\|capability.sandbox_only\|entry\[.sandbox_only.\]\|entry.get(.sandbox_only"` → 0 resultados en todo el repo. Las 5 filas
  con valor `1` (research/analysis/design/reconcile/validate_consistency)
  nunca se comparan contra nada. El único `sandbox_only` real es
  `motor.sandbox_only` (`policy.py` check 7), una columna DISTINTA, de la
  tabla `motor`. No se le inventa semántica esta ronda (el candidato obvio,
  egress de red, es el ítem "Hyde: red sin acotar por dominio/IP" ya
  diferido a propósito). Pendiente: darle lector real o dropearla.
```

- [ ] **Step 3: Agregar entrada nueva — dedup de `_CAPABILITY_TIMEOUT_SECONDS`**

En la sección "Bloquea trabajo" (es deuda con riesgo real: un valor puede divergir de la DB sin que nada lo detecte salvo un script manual):

```markdown
- **`_CAPABILITY_TIMEOUT_SECONDS` (jacobs/plan.py) duplica
  `capability.max_execution_minutes` (DB) sin lectura en vivo.** Verificado
  2026-08-27 durante el cierre de gobernanza de `_HTTP_FACETS`: el default
  de `step.timeout_seconds` sale de un dict hardcodeado
  (`jacobs/plan.py:106-110`); un `timeout_seconds` explícito en un spec de
  step lo pisa SIN validar contra el techo de la DB
  (`_validate_plan_capabilities` no lo chequea, y ni siquiera aplica a
  `_HTTP_FACETS`). `scripts/check_timeout_consistency.py` verifica que
  coincidan, pero es manual, no una garantía en runtime. El check 8 real de
  `MotorPolicy.check()` solo corre server-side, solo para `kimi`/`jax_local`,
  después de crear el job -- nunca en Jacobs antes de despachar. Decisión
  explícita: NO se activa un admission-check contra esto en la ronda de
  `_HTTP_FACETS` (validaría contra un valor que puede ya estar desincronizado
  del que el ejecutor real usa) -- se resuelve junto con la deduplicación,
  en una ronda aparte.
```

- [ ] **Step 4: Verificación manual de que no quedaron placeholders salvo el número de PR anotado en Step 1**

Run: `grep -n "TBD\|TODO\|FIXME" DEUDA.md | grep -A2 -B2 "_HTTP_FACETS\|sandbox_only.*vestigial\|_CAPABILITY_TIMEOUT_SECONDS"` — debe devolver vacío (el placeholder del PR se completa al mergear, no queda como TBD literal en el texto final).

- [ ] **Step 5: Commit**

```bash
git add DEUDA.md
git commit -m "docs(deuda): cierra _HTTP_FACETS sin gobernanza; agrega sandbox_only vestigial y dedup de timeout"
```

---

### Task 9: Deploy + verificación en vivo (Requisitos 4 y 5)

**Files:** ninguno (operación, no código).

**Interfaces:** ninguna.

Esta task NO se dispatchea a un subagente fresco — requiere reiniciar servicios de producción y confirmar contra el chat real. Ejecutarla en la sesión principal, con cuidado.

- [ ] **Step 1: Confirmar que las Tasks 1-8 están mergeadas (o el branch listo) y que la migración de la Task 3 ya corrió contra la DB real (`/etc/jax/.env`, no una DB de test)**

Run: `mysql -h "$JAX_DB_HOST" -P "$JAX_DB_PORT" -u"$JAX_DB_USER" -p"$JAX_DB_PASSWORD" jax_memory -e "SELECT \`key\`, allowed_callers FROM facet WHERE \`key\` IN ('hipatia','jekyll','thot','ada');"`
Expected: las 4 filas muestran `["jacobs", "jax_platform_chat"]`.

- [ ] **Step 2: Reiniciar `jax-las-manos` (carga el `policy.py`/`routes.py`/`facet_policy.py` nuevos)**

Run: `sudo systemctl restart jax-las-manos.service && sleep 3 && systemctl is-active jax-las-manos.service`
Expected: `active`. Revisar log inmediato: `journalctl -u jax-las-manos.service -n 30 --no-pager` — sin excepciones al arrancar (`init_motor_catalog` corre en el startup hook, cualquier error de import ahí rompe el arranque).

- [ ] **Step 3: Reiniciar `jax-platform` backend (carga el `chat.py` nuevo)**

Run: `sudo systemctl restart jax-platform.service && sleep 3 && systemctl is-active jax-platform.service`
Expected: `active`. `journalctl -u jax-platform.service -n 30 --no-pager` sin excepciones.

- [ ] **Step 4: Verificación en vivo — endpoint directo**

Run: `curl -s -X POST http://127.0.0.1:7777/motor/authorize-facet -H "Content-Type: application/json" -d '{"caller":"jax_platform_chat","facet":"hipatia"}'`
Expected: `{"allowed":true,"reason":"OK: 'jax_platform_chat' → facet 'hipatia'"}`.

- [ ] **Step 5: Verificación en vivo — chat real, los 4 facets**

Usar `claude-in-chrome` (o el cliente HTTP autenticado que ya use la sesión) para mandar un mensaje real a la Mesa web con cada uno de los 4 facets (`hipatia`, `jekyll`, `thot`, `ada`) y confirmar respuesta real, no el mensaje de "no disponible". Este es el criterio de "no rompas nada" del Requisito 4/5 — no alcanza con que los tests unitarios pasen.

- [ ] **Step 6: Si algo falla — rollback**

Si cualquier facet devuelve "no autorizado" inesperadamente: confirmar primero que la migración de la Task 3 corrió contra la DB CORRECTA (`JAX_DB_HOST`/`JAX_DB_PORT` del entorno real, no `jax_memory_test`) antes de tocar código — el error más probable es la migración corrida contra la DB equivocada, no un bug de lógica (ya cubierto por los tests de las Tasks 1-7).

---

## Self-Review

**Cobertura del spec:** Problema (Tasks 2-7), Decisión 1 sandbox_only (Task 8 Step 2, Task 3 Step 2), Decisión 2 timeout ceiling diferido (Task 8 Step 3, declarado explícito en Task 2/6 docstrings), Approach C (Tasks 2,4,5,6), Arquitectura secciones 1-6 (Tasks 2, 6, 4, 5, 7, 3 respectivamente), Testing Requisitos 1-2 (Tasks 1-2 caracterización, Task 6/7 negativos, Task 7 fail-closed), Requisito 3 fail-closed (Task 6 fail-closed de DB + Task 7 Steps 1/3), Requisito 4/5 (Task 3 Step 6, Task 9), Qué queda sin gobernar (declarado en docstrings de Task 2/8, no oculto).

**Placeholder scan:** el único placeholder real es el número de PR en Task 8 Step 1 (no puede conocerse antes de abrir el PR) — explícitamente marcado como el único caso, no un patrón repetido.

**Consistencia de tipos:** `check_facet_admission()` devuelve `tuple[bool, str]` (Task 4) — el endpoint (Task 5) lo desempaqueta como `allowed, reason`. `check_capability_admission()` devuelve `MotorPolicyResult` (Task 2) — Task 6 lee `.allowed`/`.reason`, consistente con el tipo existente. `FacetAuthorizeRequest`/`FacetAuthorizeResponse` (Task 5) coinciden con el JSON que manda `chat.py` en Task 7 (`caller`/`facet` → `allowed`/`reason`).

**Orden de dependencias (por qué este orden, no otro):** Task 3 (migración `facet.allowed_callers`) va ANTES de Task 4 (su lector) a propósito — Global Constraints lo exige explícitamente: la columna nace con su lector en el mismo tramo, no antes, para no repetir el destino de `capability.sandbox_only`. Tasks 1→2 (caracterizar antes de refactorizar) y 4→5 (lector antes de exponerlo por HTTP) y 5→7 (endpoint antes de que Mesa web lo llame) siguen el mismo principio: nunca escribir un consumidor antes de que lo que consume exista y esté probado.
