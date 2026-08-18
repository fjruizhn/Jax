# REFORMAS-v3 Fase 1 — Desbloqueo de lectura — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar los tres ítems de REFORMAS-v3.md §5 "Fase 1 — Desbloqueo de lectura": la capability `read_audit_log` otorgada por contrato (no por identidad de facet), identidad+capabilities inyectadas a cada motor, y el rechazo tipado `CAPABILITY_UNBOUND` interceptado y re-enrutado por el scheduler (Jacobs).

**Architecture:** Dos subsistemas de autorización coexisten hoy en `jax` y esta fase no los unifica (eso es alcance de R3 completo, no de Fase 1): (1) `las_manos/policy.py` gatea las 11 operaciones `[ops.*]` — incluida `audit_log_read` — por `facets.<nombre>.allowed_ops`; (2) `jacobs/executor.py` + `motor_registry/policy.py` gatean las capabilities de pipeline (`generate`, `code_swarm`, etc.) por `allowed_motors`/`allowed_callers` del catálogo. `read_audit_log` (bullet 1 de Fase 1) vive en el subsistema (1) y se resuelve ahí: se saca de detrás de `allowed_ops` para que cualquier facet la use por contrato de tarea. `CAPABILITY_UNBOUND` + reroute (bullet 3) vive en el subsistema (2), en el único punto donde hoy existe pre-validación de capability con candidatos conocidos (`jacobs/executor.py:validate_capability`) pero sin tipo de rechazo estructurado ni reintento — hoy aborta el pipeline entero. La inyección de identidad (bullet 2) es прompt/contexto nuevo hacia los motores, sin punto de partida en código hoy — se agrega en el único lugar que arma el payload HTTP al motor (`motor_registry/worker.py`).

**Tech Stack:** Python 3.14, FastAPI (`las_manos/server.py`), Pydantic v2 (`motor_registry/models.py`), pytest + `fastapi.testclient.TestClient`, asyncio (`jacobs/executor.py`).

**Spec:** `/opt/jax/docs/REFORMAS-v3.md` §3 (R3, sha256 `4099a08c39713c79836eb1ab58fc42e0a3a1357767590cfe281c04ea7ede8660`) y §5 (Fase 1). Relevamiento del estado actual: este plan cita rutas/líneas verificadas contra el código real el 2026-08-15.

## Global Constraints

- No rediseñar R3 — implementar fiel a lo ya escrito en REFORMAS-v3.md, sin inventar mecanismos nuevos no descritos ahí.
- No tocar `las_manos/motor_registry/models.py`'s `MotorDispatchRequest`/`MotorDispatchResponse` existentes salvo para agregar, nunca para romper compatibilidad (hay tests que dependen de su forma actual).
- `IntentEnvelope.facet_id` (`envelope.py:41`) es `Literal["thot","hipatia","jekyll","hyde","jax_local"]` — no se amplía en esta fase (kimi/ada no llaman a `/execute` directo).
- Todo cambio en `las_manos/*.py` requiere `py_compile` antes de cerrar (regla de `jax/CLAUDE.md`).
- Backup obligatorio antes de modificar archivos existentes (`*.backup-pre-<cambio>*`), regla de `jax/CLAUDE.md`.
- Dos patrones de test establecidos, usar el que corresponda: `TestClient(server.app, raise_server_exceptions=False)` para HTTP (`tests/test_envelope_brutal.py`); réplica aislada de funciones puras, sin importar `jacobs.executor` directo, para lógica de executor (`tests/test_jacobs_director.py`).

---

### Task 1: Desbloquear `read_audit_log` por contrato, no por identidad de facet

**Files:**
- Modify: `las_manos/policy.py:62-67` (bloque `# 3) ¿La faceta puede ejecutar esta operación?`)
- Test: `tests/test_read_audit_log_unbound.py` (nuevo)

**Interfaces:**
- Consumes: `PolicyEngine.check(facet, operation, target_host, ...)` — firma existente, sin cambios.
- Produces: `PolicyResult(ok=True, reason="")` para `operation="audit_log_read"` sin importar el facet, siempre que `operation in self.ops` (paso 2, sin cambios).

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_read_audit_log_unbound.py
"""
REFORMAS-v3 Fase 1: read_audit_log se otorga por contrato de tarea, no por
identidad de facet — cualquier facet puede pedirla aunque su config.toml
[facets.<x>] allowed_ops no la incluya explícitamente.
"""
from __future__ import annotations

import sys
from pathlib import Path

LAS_MANOS = Path(__file__).resolve().parent.parent / "las_manos"
sys.path.insert(0, str(LAS_MANOS))

from policy import PolicyEngine  # noqa: E402


def _engine_without_audit_log_read_for(facet: str) -> PolicyEngine:
    config = {
        "facets": {
            facet: {
                "allowed_envs": ["local"],
                "allowed_ops": ["read_file"],  # NO incluye audit_log_read
                "can_write_prod": False,
            }
        },
        "ops": {"audit_log_read": {}, "read_file": {}},
        "environments": {"local": ["127.0.0.1"]},
    }
    return PolicyEngine(config)


def test_read_audit_log_otorgada_por_contrato_no_por_facet():
    engine = _engine_without_audit_log_read_for("hipatia")
    result = engine.check(
        facet="hipatia",
        operation="audit_log_read",
        target_host="127.0.0.1",
    )
    assert result.ok, f"debía otorgarse por contrato; razón de rechazo: {result.reason}"


def test_read_file_sigue_gateado_por_allowed_ops():
    """Control: el cambio NO afloja otras operaciones, solo audit_log_read."""
    engine = _engine_without_audit_log_read_for("hipatia")
    result = engine.check(
        facet="hipatia",
        operation="read_file",
        target_host="127.0.0.1",
    )
    assert result.ok  # read_file SÍ está en allowed_ops de este fixture

    engine2 = _engine_without_audit_log_read_for("hipatia")
    result2 = engine2.check(
        facet="hipatia",
        operation="list_dir",  # ni siquiera está en self.ops del fixture
        target_host="127.0.0.1",
    )
    assert not result2.ok
```

- [ ] **Step 2: Correr el test y confirmar que falla**

Run: `cd /home/fruiz/jax/las_manos && .venv/bin/python -m pytest ../tests/test_read_audit_log_unbound.py -v`
Expected: `test_read_audit_log_otorgada_por_contrato_no_por_facet` FAILS (`allowed_ops` de hipatia no incluye `audit_log_read`, hoy se rechaza).

- [ ] **Step 3: Backup y modificación mínima**

```bash
cp las_manos/policy.py las_manos/policy.py.backup-pre-read-audit-log-contrato
```

En `las_manos/policy.py`, reemplazar el bloque (líneas ~61-67):
```python
        # 3) ¿La faceta puede ejecutar esta operación?
        if operation not in facet_cfg["allowed_ops"]:
            return PolicyResult(
                False,
                f"Faceta '{facet}' no autorizada para operación '{operation}'. "
                f"Permitidas: {facet_cfg['allowed_ops']}"
            )
```
por:
```python
        # 3) ¿La faceta puede ejecutar esta operación?
        # REFORMAS-v3 §3 R3.3 — capabilities de solo lectura disponibles a
        # cualquier motor, otorgadas por contrato de tarea, no por identidad
        # de facet. Fase 1 desbloquea únicamente read_audit_log (mapeada acá
        # a la operación real 'audit_log_read'); las otras tres del §3.3
        # (read_own_config, list_facets_and_capabilities, read_memory) NO
        # están en el alcance de Fase 1 y siguen gateadas por allowed_ops.
        CONTRACT_GRANTED_READONLY_OPS = {"audit_log_read"}
        if operation not in CONTRACT_GRANTED_READONLY_OPS \
                and operation not in facet_cfg["allowed_ops"]:
            return PolicyResult(
                False,
                f"Faceta '{facet}' no autorizada para operación '{operation}'. "
                f"Permitidas: {facet_cfg['allowed_ops']}"
            )
```

- [ ] **Step 4: Correr el test y confirmar que pasa**

Run: `cd /home/fruiz/jax/las_manos && .venv/bin/python -m pytest ../tests/test_read_audit_log_unbound.py -v`
Expected: ambos tests PASS.

- [ ] **Step 5: py_compile + test de integración existente no roto**

Run:
```bash
cd /home/fruiz/jax/las_manos
.venv/bin/python -m py_compile policy.py
.venv/bin/python -m pytest ../tests/test_audit_traffic_class.py ../tests/test_envelope_brutal.py -v
```
Expected: todo PASS — este cambio no debe afectar el flujo de envelope/audit ya cubierto.

- [ ] **Step 6: Commit**

```bash
git add las_manos/policy.py tests/test_read_audit_log_unbound.py
git commit -m "feat(policy): read_audit_log otorgada por contrato de tarea (REFORMAS-v3 Fase 1)"
```

---

### Task 2: Esquema tipado `CAPABILITY_UNBOUND`

**Files:**
- Modify: `jacobs/plan.py` (agregar cerca de `VALID_CAPABILITIES`, cuya línea exacta puede variar — ubicar con `grep -n "^VALID_CAPABILITIES" jacobs/plan.py`)
- Test: `tests/test_capability_unbound_schema.py` (nuevo)

**Interfaces:**
- Produces: `CapabilityUnbound` — dataclass con `status: str` (siempre `"CAPABILITY_UNBOUND"`), `required: list[str]`, `candidates: list[str]`, `task_id: str`, y método `to_dict() -> dict` para serializar exactamente con la forma de REFORMAS-v3.md §3.1.4:
  ```json
  {"status": "CAPABILITY_UNBOUND", "required": ["read_audit_log"], "candidates": ["hyde", "gptoss_120b"], "task_id": "..."}
  ```
- Consumido por: Task 3 (`validate_capability`) y Task 4 (`_dispatch_step`).

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_capability_unbound_schema.py
"""REFORMAS-v3 §3.1.4 — forma exacta del rechazo tipado CAPABILITY_UNBOUND."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jacobs.plan import CapabilityUnbound  # noqa: E402


def test_capability_unbound_forma_exacta():
    cu = CapabilityUnbound(
        required=["read_audit_log"],
        candidates=["hyde", "gptoss_120b"],
        task_id="task-123",
    )
    assert cu.to_dict() == {
        "status": "CAPABILITY_UNBOUND",
        "required": ["read_audit_log"],
        "candidates": ["hyde", "gptoss_120b"],
        "task_id": "task-123",
    }


def test_capability_unbound_status_no_es_parametro():
    """status siempre es CAPABILITY_UNBOUND — no se puede pisar por afuera."""
    cu = CapabilityUnbound(required=["x"], candidates=[], task_id="t")
    assert cu.status == "CAPABILITY_UNBOUND"
```

- [ ] **Step 2: Correr y confirmar que falla**

Run: `.venv/bin/python -m pytest tests/test_capability_unbound_schema.py -v`
Expected: FAIL — `ImportError: cannot import name 'CapabilityUnbound'`.

- [ ] **Step 3: Implementación mínima**

```bash
cp jacobs/plan.py jacobs/plan.py.backup-pre-capability-unbound
```

Agregar a `jacobs/plan.py`, después de la definición de `VALID_CAPABILITIES`:
```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CapabilityUnbound:
    """Rechazo tipado — REFORMAS-v3.md §3.1.4. El scheduler lo intercepta y
    reenruta a uno de los candidates; el usuario nunca ve este estado."""
    required: list[str]
    candidates: list[str]
    task_id: str
    status: str = field(default="CAPABILITY_UNBOUND", init=False)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "required": self.required,
            "candidates": self.candidates,
            "task_id": self.task_id,
        }
```

- [ ] **Step 4: Correr y confirmar que pasa**

Run: `.venv/bin/python -m pytest tests/test_capability_unbound_schema.py -v`
Expected: PASS.

- [ ] **Step 5: py_compile + commit**

```bash
.venv/bin/python -m py_compile jacobs/plan.py
git add jacobs/plan.py tests/test_capability_unbound_schema.py
git commit -m "feat(jacobs): esquema tipado CapabilityUnbound (REFORMAS-v3 §3.1.4)"
```

---

### Task 3: `validate_capability` devuelve `CapabilityUnbound` tipado en vez de string

**Files:**
- Modify: `jacobs/executor.py:687-734` (función `validate_capability`)
- Test: `tests/test_validate_capability_typed.py` (nuevo — patrón réplica aislada, como `tests/test_jacobs_director.py`, sin importar `jacobs.executor` con sus dependencias de red/DB)

**Interfaces:**
- Consumes: `CapabilityUnbound` de Task 2 (`jacobs.plan`).
- Produces: `validate_capability(step: Step) -> CapabilityUnbound | None` (firma cambiada: antes `str | None`). Consumido por Task 4.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_validate_capability_typed.py
"""
Réplica aislada de la lógica NIVEL B de validate_capability (jacobs/executor.py),
mismo patrón que tests/test_jacobs_director.py — sin importar jacobs.executor
directo (evita dependencias de red/DB en el import).
"""
from __future__ import annotations

from dataclasses import dataclass

from jacobs.plan import CapabilityUnbound


@dataclass
class _FakeStep:
    facet: str
    capability: str


def _validate_nivel_b(step, catalog_caps: dict, motor_facets: frozenset, task_id: str):
    """Réplica exacta de jacobs/executor.py:718-733, retornando CapabilityUnbound
    tipado en vez de string — esto es lo que Task 3 implementa en el original."""
    if step.facet not in motor_facets:
        return None
    entry = catalog_caps.get(step.capability)
    if entry is None:
        return CapabilityUnbound(
            required=[step.capability], candidates=[], task_id=task_id,
        )
    if step.facet not in entry.get("allowed_motors", []):
        return CapabilityUnbound(
            required=[step.capability],
            candidates=list(entry.get("allowed_motors", [])),
            task_id=task_id,
        )
    return None


def test_motor_no_autorizado_devuelve_capability_unbound_tipado():
    step = _FakeStep(facet="ada", capability="code_swarm")
    catalog = {"code_swarm": {"allowed_motors": ["kimi"], "allowed_callers": ["jacobs"]}}
    result = _validate_nivel_b(step, catalog, frozenset({"kimi", "ada"}), "task-1")
    assert result is not None
    assert result.to_dict() == {
        "status": "CAPABILITY_UNBOUND",
        "required": ["code_swarm"],
        "candidates": ["kimi"],
        "task_id": "task-1",
    }


def test_motor_autorizado_devuelve_none():
    step = _FakeStep(facet="kimi", capability="code_swarm")
    catalog = {"code_swarm": {"allowed_motors": ["kimi"], "allowed_callers": ["jacobs"]}}
    result = _validate_nivel_b(step, catalog, frozenset({"kimi"}), "task-1")
    assert result is None
```

- [ ] **Step 2: Correr y confirmar que falla**

Run: `.venv/bin/python -m pytest tests/test_validate_capability_typed.py -v`
Expected: FAIL en el import de `CapabilityUnbound` si Task 2 no está mergeada aún; si ya está, este archivo en sí no falla (es una réplica autocontenida) — confirmar igual que pasa como base antes de tocar el original en Step 3.

- [ ] **Step 3: Modificar el original**

```bash
cp jacobs/executor.py jacobs/executor.py.backup-pre-capability-unbound-tipado
```

En `jacobs/executor.py`, agregar el import (junto a los existentes de `jacobs.plan`):
```python
from jacobs.plan import CapabilityUnbound  # + lo que ya se importa de ahí
```

Cambiar la firma y el cuerpo de `validate_capability` (líneas ~687-734) — reemplazar cada `return f"..."` del bloque NIVEL B (líneas ~722-733) por un `CapabilityUnbound`, y actualizar el docstring y el type hint:

```python
def validate_capability(step: Step) -> CapabilityUnbound | str | None:
    """... (docstring existente, agregar:)

    Devuelve CapabilityUnbound (tipado, REFORMAS-v3 §3.1.4) cuando el motivo
    de rechazo es un binding capability→motor ausente (NIVEL B) — el
    scheduler lo reenruta. Devuelve str para NIVEL A (vocabulario cerrado,
    no es un problema de binding, no tiene candidates que ofrecer).
    """
    cap = step.capability

    if cap == "assemble":
        return None

    # ---- NIVEL A: sin cambios, sigue devolviendo str ----
    if cap not in VALID_CAPABILITIES:
        return f"capability desconocida: '{cap}' no está en VALID_CAPABILITIES"

    # ---- NIVEL B: ahora devuelve CapabilityUnbound tipado ----
    if step.facet in _MOTOR_FACETS:
        if not _CATALOG_CAPS:
            return None
        resolved = _CAPABILITY_MAP.get(cap, cap)
        entry = _CATALOG_CAPS.get(resolved)
        if entry is None:
            return CapabilityUnbound(
                required=[resolved], candidates=[], task_id=step.step_id,
            )
        if step.facet not in entry.get("allowed_motors", []):
            return CapabilityUnbound(
                required=[resolved],
                candidates=list(entry.get("allowed_motors", [])),
                task_id=step.step_id,
            )
        if "jacobs" not in entry.get("allowed_callers", []):
            return CapabilityUnbound(
                required=[resolved], candidates=[], task_id=step.step_id,
            )
    return None
```

**Nota para quien ejecute este task:** verificar el nombre real del campo id en `Step` (`jacobs/models.py`) antes de escribir `step.step_id` — puede llamarse distinto (`id`, `step_id`, `trace_id`). Confirmar con `grep -n "class Step" -A 15 jacobs/models.py` y usar el nombre real.

- [ ] **Step 4: Correr los tests y confirmar que pasan**

Run: `.venv/bin/python -m pytest tests/test_validate_capability_typed.py tests/test_capability_unbound_schema.py -v`
Expected: PASS.

- [ ] **Step 5: py_compile + regresión de tests existentes que tocan executor**

Run:
```bash
.venv/bin/python -m py_compile jacobs/executor.py
.venv/bin/python -m pytest tests/test_jacobs_director.py tests/test_jacobs_timeout_by_capability.py -v
```
Expected: PASS. Si `test_jacobs_director.py` falla, es porque su réplica aislada asumía `validate_capability` devolviendo `str | None` — actualizar esa réplica para reflejar el tipo nuevo (no el comportamiento; el docstring del archivo exige que la réplica "coincida con executor", así que este es el punto donde se actualiza junto con el original).

- [ ] **Step 6: Commit**

```bash
git add jacobs/executor.py jacobs/plan.py tests/test_validate_capability_typed.py
git commit -m "feat(jacobs): validate_capability devuelve CapabilityUnbound tipado (NIVEL B)"
```

---

### Task 4: `_dispatch_step` intercepta `CapabilityUnbound` y reenruta

**Files:**
- Modify: `jacobs/executor.py:737-748` (inicio de `_dispatch_step`)
- Test: `tests/test_dispatch_step_reroute.py` (nuevo, patrón réplica aislada)

**Interfaces:**
- Consumes: `CapabilityUnbound` de Task 2/3.
- Produces: comportamiento nuevo de `_dispatch_step` — ante `CapabilityUnbound` con `candidates` no vacío, reintenta con el primer candidato no probado en vez de `raise ValueError`; si se agotan los candidatos, falla como hoy (mismo `ValueError`, mismo camino a `_fail_step`).

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_dispatch_step_reroute.py
"""
Réplica aislada de la lógica de reroute que Task 4 agrega al inicio de
jacobs/executor.py:_dispatch_step — mismo patrón que test_jacobs_director.py.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from jacobs.plan import CapabilityUnbound


@dataclass
class _FakeStep:
    facet: str
    capability: str


def _dispatch_with_reroute(step, validate_fn, max_attempts: int = 3):
    """Réplica de la lógica que Task 4 agrega antes del `raise ValueError`
    actual en _dispatch_step: reintenta con candidatos no probados."""
    tried = {step.facet}
    current = step
    for _ in range(max_attempts):
        result = validate_fn(current)
        if result is None:
            return current, None  # listo para dispatch real
        if isinstance(result, str):
            return current, result  # NIVEL A, no reenruta
        untried = [c for c in result.candidates if c not in tried]
        if not untried:
            return current, result  # candidatos agotados, falla como CapabilityUnbound
        tried.add(untried[0])
        current = replace(current, facet=untried[0])
    return current, result


def test_reroute_a_primer_candidato_no_probado():
    step = _FakeStep(facet="ada", capability="code_swarm")

    def fake_validate(s):
        if s.facet == "ada":
            return CapabilityUnbound(required=["code_swarm"], candidates=["kimi"], task_id="t1")
        return None  # kimi sí está autorizado

    final_step, error = _dispatch_with_reroute(step, fake_validate)
    assert error is None
    assert final_step.facet == "kimi"


def test_candidatos_agotados_falla_con_capability_unbound():
    step = _FakeStep(facet="ada", capability="code_swarm")

    def fake_validate(s):
        return CapabilityUnbound(required=["code_swarm"], candidates=["kimi"], task_id="t1")

    final_step, error = _dispatch_with_reroute(step, fake_validate, max_attempts=2)
    assert isinstance(error, CapabilityUnbound)


def test_nivel_a_no_reenruta():
    """Vocabulario desconocido (str, no CapabilityUnbound) no tiene candidatos
    — no se reenruta, falla directo como hoy."""
    step = _FakeStep(facet="ada", capability="capability-inventada")

    def fake_validate(s):
        return "capability desconocida: 'capability-inventada' no está en VALID_CAPABILITIES"

    final_step, error = _dispatch_with_reroute(step, fake_validate)
    assert error == "capability desconocida: 'capability-inventada' no está en VALID_CAPABILITIES"
    assert final_step.facet == "ada"  # no cambió
```

- [ ] **Step 2: Correr y confirmar que pasa como réplica autocontenida**

Run: `.venv/bin/python -m pytest tests/test_dispatch_step_reroute.py -v`
Expected: PASS (esta réplica no depende del original — confirma la lógica antes de tocar `executor.py`).

- [ ] **Step 3: Aplicar la misma lógica al original**

```bash
cp jacobs/executor.py jacobs/executor.py.backup-pre-reroute
```

Reemplazar en `jacobs/executor.py:737-748`:
```python
async def _dispatch_step(step: Step, pipeline: Pipeline) -> dict:
    """Selecciona el worker correcto según la faceta."""
    if step.capability == "assemble":
        return _assemble_mechanical(step, pipeline)

    cap_error = validate_capability(step)
    if cap_error:
        raise ValueError(f"Capability inválida (pre-dispatch): {cap_error}")

    ctx_input = _build_context_input(step, pipeline)
```
por:
```python
async def _dispatch_step(step: Step, pipeline: Pipeline) -> dict:
    """Selecciona el worker correcto según la faceta."""
    if step.capability == "assemble":
        return _assemble_mechanical(step, pipeline)

    # REFORMAS-v3 §3.1.4 — CAPABILITY_UNBOUND se intercepta y reenruta a un
    # candidate antes de abortar el pipeline. NIVEL A (str) no tiene
    # candidatos — falla igual que antes. El usuario nunca ve el estado
    # intermedio: si el reroute encuentra un candidato válido, el pipeline
    # sigue como si el step hubiera sido asignado a ese facet desde el inicio.
    tried_facets = {step.facet}
    cap_error = validate_capability(step)
    while isinstance(cap_error, CapabilityUnbound):
        untried = [c for c in cap_error.candidates if c not in tried_facets]
        if not untried:
            raise ValueError(
                f"Capability inválida (pre-dispatch, candidatos agotados): "
                f"{cap_error.to_dict()}"
            )
        step = replace(step, facet=untried[0])
        tried_facets.add(untried[0])
        cap_error = validate_capability(step)
    if isinstance(cap_error, str):
        raise ValueError(f"Capability inválida (pre-dispatch): {cap_error}")

    ctx_input = _build_context_input(step, pipeline)
```

**Nota para quien ejecute este task:** `Step` necesita soportar `dataclasses.replace()` (requiere que sea un `@dataclass`, no un Pydantic `BaseModel`) — confirmar con `grep -n "^class Step" -A 3 jacobs/models.py`. Si `Step` es Pydantic, usar `step.model_copy(update={"facet": untried[0]})` en su lugar y ajustar el import (`from dataclasses import replace` → no aplica).

Agregar el import correspondiente (`from dataclasses import replace` o nada, según lo anterior) y `from jacobs.plan import CapabilityUnbound` si Task 3 no lo dejó ya importado a nivel de módulo.

- [ ] **Step 4: Correr los tests y confirmar que pasan**

Run: `.venv/bin/python -m pytest tests/test_dispatch_step_reroute.py tests/test_validate_capability_typed.py tests/test_jacobs_director.py -v`
Expected: PASS.

- [ ] **Step 5: py_compile + commit**

```bash
.venv/bin/python -m py_compile jacobs/executor.py
git add jacobs/executor.py tests/test_dispatch_step_reroute.py
git commit -m "feat(jacobs): _dispatch_step reenruta ante CAPABILITY_UNBOUND antes de abortar"
```

---

## Checkpoint — parar y confirmar con Fernando antes de Task 5

Tasks 1-4 son de solo lectura y reenrutamiento (nunca cambian qué ve un
motor). Task 5 es distinta de categoría: toca `motor_registry/worker.py` y
antepone texto nuevo al prompt de *todos* los motores en producción — es
la tarea con más superficie para romper comportamiento existente de esta
fase.

Antes de arrancar Task 5:
- [ ] Confirmar que Tasks 1-4 están commiteadas y sus tests (incluida la
      regresión de Step 5 de cada task) pasaron limpio.
- [ ] Mostrarle a Fernando el diff acumulado de Tasks 1-4 y el texto exacto
      que `build_identity_context` va a prependear al prompt (Step 1-4 de
      Task 5 se pueden implementar y testear en aislado sin tocar
      `worker.py` — hacerlo primero, y parar ahí para el checkpoint antes
      del Step 5 que sí modifica `worker.py`).
- [ ] Confirmar los tres detalles marcados para verificar en el código real
      antes de asumir nada (ver "Puntos que el ejecutor debe verificar" en
      Self-Review): campo id de `Step`, si `Step` es dataclass o Pydantic
      (estos dos ya resueltos en Tasks 3-4, reusar lo confirmado), y el
      nombre real de la(s) función(es) `_call_<motor>` + variable de
      catálogo en scope de `worker.py` (este es nuevo para Task 5).
- [ ] Fernando aprueba seguir con el Step 5 de Task 5 (el wiring real en
      `worker.py`) antes de que el subagente lo ejecute.

No seguir a Step 5 de Task 5 sin ese OK explícito.

---

### Task 5: Identidad y capabilities inyectadas por motor

**Files:**
- Modify: `las_manos/motor_registry/worker.py:53-73` (función `_call_kimi`, y cualquier otra función `_call_<motor>` que arme el payload HTTP — confirmar con `grep -n "^def _call_" las_manos/motor_registry/worker.py`)
- Create: `las_manos/motor_registry/identity_context.py` (nuevo módulo — arma el bloque de identidad)
- Test: `tests/test_identity_context.py` (nuevo)

**Interfaces:**
- Produces: `build_identity_context(motor_name: str, capabilities: list[str], catalog: dict, task_id: str) -> str` — texto plano formateado, para prepender al prompt del motor.
- Consumes en Task 5: `las_manos/policy/vocabulary/predicates.yaml` (los 8 predicados de REFORMAS-v3 §3.1.3, ya versionados en el corpus mergeado a `master`), `motor_registry/catalog.py`'s `MotorCatalog` (roster de motores existentes).

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_identity_context.py
"""
REFORMAS-v3 §3.1.5 (R3.5) — cada motor recibe: quién es, qué capabilities
tiene en esta tarea, qué motores existen y qué puede cada uno, la lista de
predicados emitibles, y el protocolo de rechazo tipado.
"""
from __future__ import annotations

import sys
from pathlib import Path

LAS_MANOS = Path(__file__).resolve().parent.parent / "las_manos"
sys.path.insert(0, str(LAS_MANOS))

from motor_registry.identity_context import build_identity_context  # noqa: E402


def test_identity_context_incluye_los_cinco_elementos_de_r35():
    catalog = {
        "kimi": {"allowed_motors_for": ["code_swarm", "bug_hunt"]},
        "ada": {"allowed_motors_for": ["architecture_review"]},
    }
    predicates = ["CAPABILITY_AVAILABLE", "FACET_EXISTS", "ENGINE_STATUS",
                  "CONFIG_VALUE", "FILE_EXISTS", "AUDIT_EVENT_EXISTS",
                  "JOB_STATUS", "MEMORY_ENTRY_EXISTS"]

    ctx = build_identity_context(
        motor_name="kimi",
        capabilities=["code_swarm"],
        catalog=catalog,
        predicates=predicates,
        task_id="task-42",
    )

    assert "kimi" in ctx  # quién es
    assert "code_swarm" in ctx  # qué capabilities tiene en esta tarea
    assert "ada" in ctx  # qué otros motores existen
    assert "architecture_review" in ctx  # qué puede cada uno
    assert "AUDIT_EVENT_EXISTS" in ctx  # predicados emitibles
    assert "CAPABILITY_UNBOUND" in ctx  # protocolo de rechazo tipado
    assert "task-42" in ctx
```

- [ ] **Step 2: Correr y confirmar que falla**

Run: `cd /home/fruiz/jax && .venv/bin/python -m pytest tests/test_identity_context.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'motor_registry.identity_context'`.

- [ ] **Step 3: Implementación mínima**

```python
# las_manos/motor_registry/identity_context.py
"""
REFORMAS-v3.md §3.1.5 — identidad inyectada. Cada motor recibe, antes de su
prompt real: quién es, qué capabilities tiene en esta tarea, qué motores
existen y qué puede cada uno, la lista de predicados emitibles, y el
protocolo de rechazo tipado (CAPABILITY_UNBOUND).

Esto es SOLO contexto/plumbing (Fase 1) — no valida ni fuerza que el motor
lo respete. Esa validación (canal claim/analysis/judgment, barrido de
vocabulario) es R1, Fase 2, fuera de alcance acá.
"""
from __future__ import annotations


def build_identity_context(
    motor_name: str,
    capabilities: list[str],
    catalog: dict,
    predicates: list[str],
    task_id: str,
) -> str:
    other_motors = "\n".join(
        f"  - {name}: {', '.join(info.get('allowed_motors_for', []))}"
        for name, info in catalog.items()
        if name != motor_name
    )
    return (
        f"[IDENTIDAD — REFORMAS-v3 §3.1.5]\n"
        f"Sos el motor '{motor_name}'. Task id: {task_id}.\n"
        f"Capabilities otorgadas para esta tarea: {', '.join(capabilities)}.\n"
        f"Otros motores del ecosistema y qué pueden hacer:\n{other_motors}\n"
        f"Predicados emitibles en canal claim (lista cerrada, §3.1.3): "
        f"{', '.join(predicates)}.\n"
        f"Si necesitás una capability no otorgada, el rechazo tipado es "
        f"CAPABILITY_UNBOUND — no lo simules, no lo inventes en tu salida.\n"
    )
```

- [ ] **Step 4: Correr y confirmar que pasa**

Run: `.venv/bin/python -m pytest tests/test_identity_context.py -v`
Expected: PASS.

- [ ] **Step 5: Wirearlo en `_call_kimi` (y cada `_call_<motor>` que exista)**

Primero, confirmar el estado real de la firma antes de editar:
```bash
grep -n "^def _call_\|^async def _call_" las_manos/motor_registry/worker.py
```

```bash
cp las_manos/motor_registry/worker.py las_manos/motor_registry/worker.py.backup-pre-identity-context
```

En cada función `_call_<motor>` encontrada, el payload hoy (según relevamiento, `_call_kimi` líneas 53-73) es:
```python
payload = {"model": ..., "messages": [{"role": "user", "content": prompt}]}
```
Cambiar a:
```python
from motor_registry.identity_context import build_identity_context
# ... (import a nivel de módulo, junto a los existentes)

identity = build_identity_context(
    motor_name=<nombre del motor en esta función>,
    capabilities=[<capability de esta request>],
    catalog=<dict de catálogo disponible en este scope — confirmar variable real>,
    predicates=<lista de los 8 predicados — cargar de las_manos/policy/vocabulary/predicates.yaml o hardcodear la lista de nombres si cargar YAML acá es demasiado para Fase 1; anotar la decisión tomada>,
    task_id=<trace_id o job_id disponible en este scope>,
)
payload = {"model": ..., "messages": [{"role": "user", "content": identity + "\n---\n" + prompt}]}
```

**Nota para quien ejecute este task:** el catálogo de motores y sus capabilities-por-motor no tiene hoy una función lista que devuelva `{motor: [capabilities]}` invertido desde `MotorCatalog` (que indexa por motor individual, `catalog.py:53-93`) — construirla es parte de este step, iterando `MotorCatalog._motors` o el `config` crudo ya cargado, lo que esté disponible en el scope de `worker.py`. Si no hay acceso directo al `config` completo en ese scope, es válido (y preferible a inventar una ruta nueva) pasar solo `{motor_name: []}` como catálogo mínimo por ahora y dejar anotado en un comentario que el catálogo completo requiere exponer `MotorCatalog` en `worker.py` — no forzar una refactor mayor no pedida por Fase 1.

- [ ] **Step 6: py_compile + smoke test manual (no automatizable sin credenciales del motor real)**

Run: `.venv/bin/python -m py_compile las_manos/motor_registry/worker.py las_manos/motor_registry/identity_context.py`
Expected: sin errores.

Esto NO reemplaza un test de integración real contra Kimi (requiere `KIMI_API_KEY`, fuera de lo que un test automatizado debería disparar). Documentar en el commit que el wiring está probado por `py_compile` + unit test de `build_identity_context`, no por una llamada real al motor.

- [ ] **Step 7: Commit**

```bash
git add las_manos/motor_registry/identity_context.py las_manos/motor_registry/worker.py tests/test_identity_context.py
git commit -m "feat(motor_registry): inyecta identidad+capabilities+predicados al motor (REFORMAS-v3 §3.1.5)"
```

---

## Self-Review

**Cobertura de spec:**
- §5 Fase 1 bullet 1 ("Capability read_audit_log") → Task 1.
- §5 Fase 1 bullet 2 ("Identidad y capabilities inyectadas por motor") → Task 5, con los cinco elementos de R3.5 verificados en el test.
- §5 Fase 1 bullet 3 ("Rechazo tipado CAPABILITY_UNBOUND + intercepción en scheduler") → Tasks 2, 3, 4 (esquema, generación, interceptación/reroute).
- R3.4 "El usuario nunca ve este estado" → cubierto por el diseño de Task 4 (reroute exitoso es transparente; solo agotar candidatos produce un fallo visible, igual que hoy).

**Fuera de alcance, señalado explícitamente (no incluido en las tasks):** R1 (canales claim/analysis/judgment, validador de claims, renderer por plantilla) es Fase 2. Las otras tres capabilities de solo lectura de R3.3 (`read_own_config`, `list_facets_and_capabilities`, `read_memory`) no están en el texto de §5 Fase 1 y no se tocan. El token de compuerta humana (R3.6) tampoco.

**Puntos que el ejecutor debe verificar contra el código real antes de aplicar** (marcados inline arriba, no asumidos): nombre exacto del campo id en `Step` (Task 3), si `Step` es dataclass o Pydantic (Task 4), nombre real de la(s) función(es) `_call_<motor>` y la variable de catálogo disponible en su scope (Task 5). Esto es intencional — el relevamiento de esta sesión no llegó a confirmar esos tres detalles con certeza suficiente para hornear una respuesta única sin verificar primero.
