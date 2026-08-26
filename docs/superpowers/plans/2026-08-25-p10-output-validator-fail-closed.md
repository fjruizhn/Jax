# P10 — output_validator.py Fail-Closed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cerrar la instancia concreta de fail-open que `DEUDA.md` y la propia regla P10 (`policy/rules/P10-fail-open-prohibido.yaml`) nombran como "residuo conocido": `las_manos/motor_registry/output_validator.py` valida `validated=True` para CUALQUIER schema desconocido, sin distinguir un typo/capability mal configurada de un schema legítimamente pendiente de implementar. Diferenciar los dos casos y fallar cerrado solo en el genuinamente desconocido, con test de regresión — sin romper las 7 capabilities de producción reales que hoy dependen del comportamiento fail-open para un schema que nunca se implementó.

**Architecture:** `output_validator.py::validate()` gana una tercera categoría explícita (`_KNOWN_UNIMPLEMENTED_SCHEMAS`, un frozenset de nombres declarados en la DB de producción pero sin validación de campos programada) entre "schema implementado" (valida campos) y "schema genuinamente desconocido" (falla cerrado). El caller (`las_manos/motor_registry/worker.py:712-726`) no cambia — ya tiene la lógica de reintento-luego-FAILED correcta, solo necesita que `validate()` le devuelva `validated=False` en el caso que hoy le llega enmascarado como éxito.

**Tech Stack:** Python 3.12, `unittest`, DB real `jax_memory` (MariaDB, puerto 3308) para verificar qué `output_schema` están en uso hoy.

**Spec:** `DEUDA.md` (raíz de `/home/fruiz/jax`), sección "Bloquea trabajo", ítem "P10 (fail-open prohibido) sin enforcement real..."; `policy/rules/P10-fail-open-prohibido.yaml`.

## Global Constraints

- **Hecho verificado en DB real (2026-08-25), crítico para este plan:** de las 12 capabilities con `output_schema` no vacío en la tabla `capability` de `jax_memory`, solo 4 tienen su schema implementado en `SCHEMAS` (`code_swarm.v1`, `bug_hunt.v1`, `architecture_review.v1`, `code_patch.v1` — este último usado por `implementation` y `refactor`). Las otras 8 filas (`critique`→`critique.v1`, `design`→`design.v1`, `generate`→`generate.v1`, `pipeline_analysis`→`analysis.v1`, `reason`→`reason.v1`, `reconcile`→`reconcile.v1`, `validate_consistency`→`validation.v1`) usan schemas que **no existen** en `SCHEMAS`. Cualquier fix que haga fail-closed a TODO schema desconocido rompería esas 7 capabilities (8 filas, 7 schemas únicos) en producción. Este plan NO puede tratar "desconocido" como un solo caso.
- No se implementan los 7 schemas pendientes en este plan — eso requiere saber qué campos devuelve cada capability realmente (trabajo de diseño aparte, se deja anotado como deuda nueva en Task 3).

---

### Task 1: Test de regresión que fija el comportamiento actual (antes de tocar código)

**Files:**
- Create: `las_manos/_output_validator_test.py`

**Interfaces:**
- Consume: `las_manos/motor_registry/output_validator.py::validate(content, schema_name, has_tool_calls=False) -> dict`

- [ ] **Step 1: Escribir los tests — algunos ya pasan hoy (fijan el comportamiento que NO debe cambiar), otros fallan hoy a propósito (fijan el comportamiento nuevo que Task 2 debe implementar)**

```python
#!/usr/bin/env python3
"""output_validator.validate() -- distingue "schema declarado pero sin
validacion de campos implementada" (7 nombres reales en uso hoy en la DB
de produccion, ver Global Constraints del plan) de "schema genuinamente
desconocido" (typo, capability mal configurada) -- P10 (DEUDA.md).

Antes de este fix, AMBOS casos devolvian validated=True con un warning
-- el caller (las_manos/motor_registry/worker.py) nunca se enteraba de
que una capability mal configurada estaba devolviendo basura sin
verificar nada. Ahora solo el primer caso (declarado, pendiente) sigue
fail-open; el segundo (no declarado en absoluto) falla cerrado.

Corre desde /home/fruiz/jax/las_manos con:
  PYTHONPATH=/home/fruiz/jax/las_manos \
  /home/fruiz/jax/las_manos/.venv/bin/python \
  /home/fruiz/jax/las_manos/_output_validator_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import unittest

from motor_registry.output_validator import validate


class OutputValidatorSchemaImplementadoTest(unittest.TestCase):
    """Comportamiento existente, NO debe cambiar."""

    def test_schema_implementado_con_todos_los_campos_valida(self):
        content = '{"diff": "...", "files_modified": ["a.py"], "description": "x"}'
        result = validate(content, "code_patch.v1")
        self.assertTrue(result["validated"])
        self.assertEqual(result["missing_fields"], [])

    def test_schema_implementado_con_campos_faltantes_no_valida(self):
        content = '{"diff": "..."}'
        result = validate(content, "code_patch.v1")
        self.assertFalse(result["validated"])
        self.assertIn("files_modified", result["missing_fields"])
        self.assertIn("description", result["missing_fields"])

    def test_sin_schema_name_valida_solo_por_ser_json(self):
        result = validate('{"cualquier_cosa": 1}', "")
        self.assertTrue(result["validated"])

    def test_has_tool_calls_se_saltea_la_validacion(self):
        result = validate("", "code_patch.v1", has_tool_calls=True)
        self.assertTrue(result["skipped"])
        self.assertFalse(result["validated"])

    def test_json_invalido_no_valida(self):
        result = validate("esto no es json", "code_patch.v1")
        self.assertFalse(result["validated"])
        self.assertIn("texto libre", result["warning"])


class OutputValidatorSchemaPendienteTest(unittest.TestCase):
    """Los 7 schemas declarados hoy en la DB de producción
    (capability.output_schema) sin validación de campos implementada --
    deben SEGUIR fail-open (romperían 7 capabilities reales si no)."""

    def test_schema_declarado_pendiente_sigue_fail_open(self):
        for schema_name in (
            "critique.v1", "design.v1", "generate.v1", "analysis.v1",
            "reason.v1", "reconcile.v1", "validation.v1",
        ):
            with self.subTest(schema=schema_name):
                result = validate('{"cualquier_cosa": 1}', schema_name)
                self.assertTrue(result["validated"], f"{schema_name} debe seguir fail-open")
                self.assertIn("declarado", result["warning"].lower())


class OutputValidatorSchemaDesconocidoTest(unittest.TestCase):
    """Comportamiento NUEVO -- schema que no es ni implementado ni
    declarado-pendiente ahora falla cerrado (P10)."""

    def test_schema_realmente_desconocido_falla_cerrado(self):
        result = validate('{"cualquier_cosa": 1}', "typo_que_nadie_declaro.v1")
        self.assertFalse(result["validated"])
        self.assertIn("no reconocido", result["warning"].lower())

    def test_schema_casi_declarado_no_hereda_el_fail_open_de_su_version_hermana(self):
        # Caso AMBIGUO real, agregado 2026-08-25 a pedido explícito del
        # usuario: 'critique.v2' comparte prefijo con 'critique.v1'
        # (declarado-pendiente, ver _KNOWN_UNIMPLEMENTED_SCHEMAS en Task 2)
        # pero NO es el mismo string -- es exactamente el borde donde un
        # bump de version en la DB de produccion, sin actualizar el
        # frozenset, podria colar fail-open por error si la comparacion
        # fuera por prefijo/fuzzy en vez de coincidencia EXACTA de string.
        # Sin este test, el diseño de Task 2 (membership exacto en un
        # frozenset) es una suposición sin verificar, no un hecho probado.
        result = validate('{"cualquier_cosa": 1}', "critique.v2")
        self.assertFalse(
            result["validated"],
            "un near-miss de un schema declarado-pendiente NO debe heredar su fail-open",
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Correr y confirmar el RED real (verificado en vivo 2026-08-25 — la predicción original de este plan estaba mal, corregida acá con evidencia real, no supuesta)**

Run: `PYTHONPATH=/home/fruiz/jax/las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest /home/fruiz/jax/las_manos/_output_validator_test.py -v`
Expected (esto es lo que realmente pasa contra el código sin modificar, confirmado con traceback crudo — NO lo que decía la primera versión de este plan): `OutputValidatorSchemaImplementadoTest` — 5/5 PASS. `OutputValidatorSchemaPendienteTest::test_schema_declarado_pendiente_sigue_fail_open` — FALLA en los 7 subTests (`AssertionError: 'declarado' not found in "schema '<nombre>' desconocido — validación de campos omitida"`) porque el código actual todavía no distingue "declarado-pendiente" de "desconocido" — los dos dicen "desconocido" hoy. `OutputValidatorSchemaDesconocidoTest::test_schema_realmente_desconocido_falla_cerrado` — FALLA (`AssertionError: True is not false`). `OutputValidatorSchemaDesconocidoTest::test_schema_casi_declarado_no_hereda_el_fail_open_de_su_version_hermana` — FALLA por el mismo motivo (`AssertionError: True is not false` — hoy CUALQUIER desconocido, incluido un near-miss, es fail-open). Total: 6 tests pasan, 2 métodos fallan (uno con 7 subfallos) + el nuevo método de near-miss también en rojo — todos por `AssertionError` sobre comportamiento real, ninguno por error estructural (ImportError, fixture rota). Si tu corrida muestra algo distinto (un ImportError, un 0 tests collected, etc.), DETENÉ y reportá — no es el mismo RED que este plan documenta.

**Nota para quien ejecute Task 2:** este archivo de test (`las_manos/_output_validator_test.py`) ya existe (Task 1, PR #26) — el nuevo método `test_schema_casi_declarado_no_hereda_el_fail_open_de_su_version_hermana` de arriba TODAVÍA NO está en ese archivo. Agregalo vos como parte del Step 1 de Task 2 (antes de tocar `output_validator.py`), confirmá que está en rojo por el motivo correcto, y recién ahí implementá el fix.

---

### Task 2: Implementar la distinción fail-open-declarado vs fail-closed-desconocido

**Files:**
- Modify: `las_manos/motor_registry/output_validator.py:24-107`
- Modify: `las_manos/_output_validator_test.py` (agregar el caso ambiguo/near-miss ANTES de tocar el código de producción)

**Interfaces:**
- Produce: `validate()` sin cambios de firma — mismo dict de retorno, mismas claves.

- [ ] **Step 1: Agregar el test del caso ambiguo (near-miss) al archivo de Task 1, confirmar que está en rojo por el motivo correcto**

En `las_manos/_output_validator_test.py`, dentro de `OutputValidatorSchemaDesconocidoTest` (después de `test_schema_realmente_desconocido_falla_cerrado`), agregar:

```python
    def test_schema_casi_declarado_no_hereda_el_fail_open_de_su_version_hermana(self):
        # Caso AMBIGUO real: 'critique.v2' comparte prefijo con
        # 'critique.v1' (declarado-pendiente, ver _KNOWN_UNIMPLEMENTED_SCHEMAS
        # abajo) pero NO es el mismo string -- el borde real de la
        # heurística, donde un bump de version en la DB sin actualizar el
        # frozenset podría colar fail-open si la comparación fuera por
        # prefijo/fuzzy en vez de coincidencia EXACTA de string.
        result = validate('{"cualquier_cosa": 1}', "critique.v2")
        self.assertFalse(
            result["validated"],
            "un near-miss de un schema declarado-pendiente NO debe heredar su fail-open",
        )
```

Correr: `PYTHONPATH=/home/fruiz/jax/las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest las_manos/_output_validator_test.py::OutputValidatorSchemaDesconocidoTest -v`
Esperado: el método nuevo FALLA con `AssertionError: True is not false` (mismo motivo que el resto — hoy cualquier desconocido, incluido este near-miss, es fail-open). Si falla por otra cosa (ImportError, nombre de test no encontrado), DETENÉ y reportá antes de seguir.

- [ ] **Step 2: Agregar el frozenset de schemas declarados-pendientes y actualizar la lógica**

En `las_manos/motor_registry/output_validator.py`, después de `SCHEMAS` (línea 37), agregar:

```python
# Schemas declarados HOY en capability.output_schema (DB de produccion,
# verificado 2026-08-25) pero sin validacion de campos implementada --
# fail-open EXPLICITO y ACOTADO a esta lista (deuda registrada, no un
# descuido): critique, design, generate, pipeline_analysis, reason,
# reconcile, validate_consistency. Implementarlos requiere saber que
# campos devuelve cada capability realmente -- fuera de este fix (ver
# DEUDA.md, item nuevo "schemas de capability sin validacion de campos").
#
# Cualquier OTRO nombre que llegue acá y no esté en SCHEMAS tampoco en
# esta lista es un caso distinto: typo, capability mal configurada, o un
# nombre nuevo que alguien se olvidó de declarar acá -- ESE sí falla
# cerrado (P10).
_KNOWN_UNIMPLEMENTED_SCHEMAS: frozenset[str] = frozenset({
    "critique.v1", "design.v1", "generate.v1", "analysis.v1",
    "reason.v1", "reconcile.v1", "validation.v1",
})
```

Reemplazar el bloque (líneas 101-107):
```python
    # Schema desconocido: warning pero no falla
    required = SCHEMAS.get(schema_name)
    if required is None:
        result["validated"] = True
        result["warning"] = f"Schema '{schema_name}' desconocido — validación de campos omitida"
        logger.warning("Schema desconocido: '%s'", schema_name)
        return result
```

Por:
```python
    required = SCHEMAS.get(schema_name)
    if required is None:
        if schema_name in _KNOWN_UNIMPLEMENTED_SCHEMAS:
            # Fail-open EXPLICITO: declarado en produccion, sin schema de
            # campos implementado todavia. Ver _KNOWN_UNIMPLEMENTED_SCHEMAS.
            result["validated"] = True
            result["warning"] = (
                f"Schema '{schema_name}' declarado pero sin schema de campos "
                "implementado — validación omitida (deuda conocida)"
            )
            logger.warning("Schema declarado-pendiente sin validar: '%s'", schema_name)
            return result
        # Schema realmente desconocido: ni implementado ni declarado como
        # pendiente -- typo o capability mal configurada. Fail-closed (P10):
        # el caller (worker.py) reintenta una vez y despues marca FAILED,
        # en vez de completar el job creyendo que algo se validó.
        result["warning"] = (
            f"Schema '{schema_name}' no reconocido (ni implementado ni "
            "declarado como pendiente) — posible typo o capability mal "
            "configurada"
        )
        logger.error("Schema NO reconocido, fallando cerrado: '%s'", schema_name)
        return result
```

Nota: `result["validated"]` ya arranca en `False` (línea 74, sin cambios) — el bloque nuevo simplemente NO lo pisa a `True` para el caso desconocido, dejando el default fail-closed.

- [ ] **Step 3: Actualizar el docstring del módulo (líneas 4-6) para que deje de afirmar una garantía que ya no es cierta**

Reemplazar:
```
Valida la respuesta del motor contra el schema declarado en la capability.
Falla abierto por schema: si el schema es desconocido, emite warning y sigue.
Nunca bloquea un job por un schema inválido o faltante.
```
Por:
```
Valida la respuesta del motor contra el schema declarado en la capability.
Fail-open ACOTADO (P10): un schema declarado pero sin validación de campos
implementada (ver _KNOWN_UNIMPLEMENTED_SCHEMAS) emite warning y sigue. Un
schema que ni siquiera está declarado ahí -- typo, capability mal
configurada -- falla cerrado: el caller lo trata como salida inválida.
```

- [ ] **Step 4: Correr los tests, confirmar que TODOS pasan ahora — incluido el caso ambiguo agregado en Step 1**

Run: `PYTHONPATH=/home/fruiz/jax/las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest las_manos/_output_validator_test.py -v`
Expected: 8/8 en verde (5 en `Implementado`, 1 en `Pendiente` con 7 subTests, 2 en `Desconocido` — el original `test_schema_realmente_desconocido_falla_cerrado` Y el nuevo `test_schema_casi_declarado_no_hereda_el_fail_open_de_su_version_hermana`). Pegá el resultado crudo en el reporte — no lo resumas como "todo pasó" sin la salida real.

- [ ] **Step 5: Regresión — correr la suite completa de `las_manos` para confirmar que ningún test existente dependía del comportamiento viejo**

Run: `cd /home/fruiz/jax/las_manos && PYTHONPATH=/home/fruiz/jax:/home/fruiz/jax/las_manos .venv/bin/python -m pytest . -x -q`
Expected: mismo resultado que antes de este cambio (ya confirmado por grep que `_worker_tool_loop_test.py` solo usa `output_schema=""` o `"code_patch.v1"`, ninguno de los 7 pendientes ni un schema inventado — cero tests deberían romperse).

- [ ] **Step 6: Commit**

```bash
cd /home/fruiz/jax
git add las_manos/motor_registry/output_validator.py las_manos/_output_validator_test.py
git commit -m "fix(P10): output_validator falla cerrado solo para schemas genuinamente desconocidos, no para los 7 declarados-pendientes en produccion"
```

---

### Task 3: Documentar el cierre en el corpus de políticas (DEUDA.md NO se toca en este plan — política explícita del usuario, 2026-08-25: solo el ítem de red de Hyde va a DEUDA.md, en el plan `hyde-semaforo-y-deuda-precision`)

**Files:**
- Modify: `policy/rules/P10-fail-open-prohibido.yaml`

- [ ] **Step 1: Actualizar `notes` en `policy/rules/P10-fail-open-prohibido.yaml` (agregar al final del bloque `notes`, sin borrar el texto existente)**

Agregar después del texto actual de `notes`:
```
  Cierre parcial 2026-08-25: output_validator.py (la instancia nombrada
  explícitamente arriba como "residuo conocido") ya no es fail-open para
  CUALQUIER schema desconocido -- solo para los 7 declarados-pendientes
  en producción (_KNOWN_UNIMPLEMENTED_SCHEMAS). Un schema realmente no
  declarado (typo, capability mal configurada) ahora falla cerrado, con
  test de regresión (las_manos/_output_validator_test.py). El residuo
  general del patrón (cualquier función que pueda fallar abierto por
  valor de retorno en código no registrado en este corpus) sigue sin
  scanner automatizado -- eso sigue siendo trabajo futuro, no resuelto acá.
```

- [ ] **Step 2: Commit (solo el YAML de política — DEUDA.md no se toca en este plan, por instrucción explícita del usuario)**

```bash
cd /home/fruiz/jax
git add policy/rules/P10-fail-open-prohibido.yaml
git commit -m "docs(policy): registra en P10 el cierre de la instancia output_validator.py"
```

**Nota para disclosure:** la deuda de los 7 schemas sin validación de campos implementada queda documentada únicamente en el código (`_KNOWN_UNIMPLEMENTED_SCHEMAS` en `output_validator.py`, con comentario explicando qué son y por qué). No se crea una entrada nueva en `DEUDA.md` — el usuario decide si la quiere ahí.

---

### Task 4: Fix wave de la revisión final — CI real para el test de regresión, drift test contra la DB, concurrency guard

**Contexto:** la revisión final de todo el rango (Tasks 1-3 + el fix de trigger de CI) encontró 3 hallazgos Important y 7 Minor. Por decisión explícita del usuario, esta ronda cierra 2 Important + 1 Minor. El otro Important (reintento inútil para schema genuinamente desconocido) se anota pero NO se implementa — es optimización de costo, no garantía rota. El resto de los Minor quedan sin tocar.

**Files:**
- Modify: `.github/workflows/policy.yml` (nuevo job + concurrency guard)
- Create: `las_manos/_output_validator_db_drift_test.py`

**Interfaces:**
- Consume: `SCHEMAS`, `_KNOWN_UNIMPLEMENTED_SCHEMAS` de `motor_registry.output_validator`; `MotorCatalog.from_db()` de `motor_registry.catalog` (patrón ya usado en `las_manos/_catalog_from_db_test.py`, incluye acceso directo a `catalog._capabilities` como ya hace `_worker_tool_loop_test.py:270`).

- [ ] **Step 1: Wirear `las_manos/_output_validator_test.py` a CI real (Important #3 de la revisión final — la nota del YAML de P10 cita este test como evidencia de cierre; hoy esa evidencia es manual, no CI)**

Este test NO toca la DB (confirmado: solo importa `validate` de `output_validator.py`, sin `aiomysql` ni credenciales) — a diferencia de `_catalog_from_db_test.py`/`_output_validator_db_drift_test.py` (Step 2), que sí la necesitan y por eso NO se wirean a CI en este task (no hay servicio de DB en el runner).

En `.github/workflows/policy.yml`, agregar un job nuevo (después de `no-fail-open-except`):

```yaml
  output-validator-regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install pytest
      - run: PYTHONPATH=las_manos python -m pytest las_manos/_output_validator_test.py -v
```

- [ ] **Step 2: Agregar el drift test contra la DB real (Important #1 — mismo bug de dos-fuentes-de-verdad que motor/facet_binding, en la tabla `capability`)**

Crear `las_manos/_output_validator_db_drift_test.py`:

```python
#!/usr/bin/env python3
"""output_validator._KNOWN_UNIMPLEMENTED_SCHEMAS es una segunda fuente de
verdad -- un snapshot codeado a mano de capability.output_schema tomado
2026-08-25. Sin este test, un cambio SOLO en la DB (agregar una capability
con un output_schema nuevo, o bumpear 'critique.v1' a 'critique.v2') hace
que ese schema empiece a fallar cerrado en producción sin que nadie lo haya
decidido a propósito -- mismo bug de dos-fuentes-de-verdad que
motor/facet_binding (cerrado 2026-08-24), en otra tabla.

Corre contra la DB real, mismo patrón que _catalog_from_db_test.py -- sin
mock, sin fixture local. NO wireado a CI (no hay servicio de DB en el
runner) -- correr a mano antes de cualquier cambio a capability.output_schema
en la DB de producción.

Corre desde /home/fruiz/jax/las_manos con:
  PYTHONPATH=/home/fruiz/jax/las_manos \
  /home/fruiz/jax/las_manos/.venv/bin/python \
  /home/fruiz/jax/las_manos/_output_validator_db_drift_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import unittest

from motor_registry.catalog import MotorCatalog
from motor_registry.output_validator import SCHEMAS, _KNOWN_UNIMPLEMENTED_SCHEMAS


class OutputValidatorDbDriftTest(unittest.IsolatedAsyncioTestCase):
    async def test_todos_los_output_schema_de_capability_estan_cubiertos(self):
        catalog = await MotorCatalog.from_db()
        db_schemas = {
            cap.output_schema
            for cap in catalog._capabilities.values()
            if cap.output_schema
        }
        covered = set(SCHEMAS.keys()) | _KNOWN_UNIMPLEMENTED_SCHEMAS
        uncovered = db_schemas - covered
        self.assertEqual(
            uncovered, set(),
            f"Schemas en capability.output_schema sin cobertura en SCHEMAS "
            f"ni _KNOWN_UNIMPLEMENTED_SCHEMAS: {uncovered} -- actualizar "
            "output_validator.py antes de que un job real falle en "
            "producción sin que nadie lo haya decidido a propósito"
        )


if __name__ == "__main__":
    unittest.main()
```

Correr: `PYTHONPATH=/home/fruiz/jax/las_manos /home/fruiz/jax/las_manos/.venv/bin/python -m pytest las_manos/_output_validator_db_drift_test.py -v` (con `/etc/jax/.env` sourceado — necesita DB real).
Esperado: PASS — verificado manualmente por el controller contra la DB real (2026-08-25) que los 11 schemas distintos en `capability.output_schema` están cubiertos exactamente por `SCHEMAS` (4) ∪ `_KNOWN_UNIMPLEMENTED_SCHEMAS` (7).

- [ ] **Step 3: Concurrency guard en `policy.yml` (Minor — sin esto, `branches: ["**"]` en push Y pull_request corre el job DOS veces por cada push a una rama con PR abierto)**

En `.github/workflows/policy.yml`, agregar después del bloque `on:` (antes de `jobs:`):

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

- [ ] **Step 4: Correr localmente y confirmar**

Run: `python3 policy/tests/test_no_fail_open_except.py && PYTHONPATH=las_manos python3 -m pytest las_manos/_output_validator_test.py -v`
Expected: ambos OK/verde — confirma que el YAML sigue siendo válido y que el comando que CI va a correr funciona igual en local.

- [ ] **Step 5: Commit**

```bash
cd /home/fruiz/jax
git add .github/workflows/policy.yml las_manos/_output_validator_db_drift_test.py
git commit -m "ci(P10): wirea el test de regresion a CI real, agrega drift test DB vs _KNOWN_UNIMPLEMENTED_SCHEMAS, concurrency guard"
```

**Nota para el reporte (obligatoria, no opcional):** anotar explícitamente el Important #2 de la revisión final (reintento inútil para schema genuinamente desconocido — un job con schema no reconocido gasta una llamada extra al modelo pidiéndole que cumpla un schema que no existe, antes de fallar) como hallazgo confirmado pero NO implementado en este task, por decisión explícita del usuario (optimización de costo, no garantía rota).

---

## Self-Review

- **Cobertura del spec:** el ítem de `DEUDA.md` pedía "enforcement real más allá de los tests de política" para la instancia nombrada — Task 2 lo cierra con código + test; Task 3 documenta honestamente que el patrón GENERAL (no esta instancia) sigue sin scanner, evitando sobre-declarar cobertura que no existe.
- **Riesgo de regresión verificado con evidencia real, no supuesto:** la consulta directa a `capability` en `jax_memory` (Global Constraints) es lo que evitó un fix ingenuo que hubiera roto 7 de 12 capabilities en producción.
- **Sin placeholders:** todo el código de Task 2 es el diff real a aplicar, no una descripción de qué hacer.
