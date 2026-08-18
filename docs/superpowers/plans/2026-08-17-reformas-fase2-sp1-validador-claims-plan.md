# REFORMAS-v3 Fase 2, Sub-proyecto 1 — Validador de claims: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir `policy/governance/` — schema de claim, validador determinista contra fuentes reales, barrido de vocabulario cerrado y renderer por plantillas con hash versionado — probado íntegramente con claims sintéticos, sin tocar `chat.py` ni ningún consumidor de tráfico real.

**Architecture:** Cinco módulos nuevos en `policy/governance/` (uno de schema puro, dos de I/O con roles distintos — config estática vs. estado vivo —, dos puros de lógica) más una herramienta nueva en `policy/tools/` que extiende el patrón ya existente de `corpus_hash.py` a `render_templates.yaml`. Cero cambios a código fuera de `policy/`.

**Tech Stack:** Python 3.14, Pydantic 2.13.4, PyYAML 6.0.3, pytest 9.1.1 — todos ya instalados en el `.venv` de la **raíz del repo** (`/home/fruiz/jax/.venv`, NO `las_manos/.venv` — ese venv no tiene PyYAML). `tomllib` (stdlib) para leer `las_manos/config.toml`.

**Spec:** `docs/superpowers/specs/2026-08-17-reformas-fase2-sp1-validador-claims-design.md` (commit `0e0fe5f` en `master`) — este plan argumenta desde ese spec; los ejecutores deben leer ambos.

## Global Constraints

- **Todo se prueba con claims sintéticos.** Ningún test ni código de este plan invoca `chat.py`, la Mesa web, ni ningún endpoint HTTP real.
- **Dos módulos de I/O, no uno.** `loaders.py` lee config estática de policy (falla cerrado si el hash no coincide → el subsistema no arranca). `validator.py` lee estado vivo (`config.toml`, filesystem) → si un resolver falla, es un rechazo de ESE claim, no una falla de arranque. No mezclar responsabilidades entre los dos.
- **`claims.py`, `vocab_sweep.py`, `renderer.py` son puros** — reciben datos ya cargados, nunca abren un archivo.
- **`RESOLVER_NOT_IMPLEMENTED` es un veredicto explícito, nunca un paso silencioso.** Ningún claim de un predicado sin resolver real puede terminar en `VALID`.
- **`FILE_EXISTS` consulta la allowlist antes que el filesystem, siempre.** Un path fuera de `config_paths` nunca dispara `stat()`, `exists()` ni lectura de bytes.
- **Solo 2 de 8 predicados tienen resolver real esta ronda:** `CAPABILITY_AVAILABLE` y `FILE_EXISTS`. Los otros 6 (`FACET_EXISTS`, `ENGINE_STATUS`, `CONFIG_VALUE`, `AUDIT_EVENT_EXISTS`, `JOB_STATUS`, `MEMORY_ENTRY_EXISTS`) devuelven `RESOLVER_NOT_IMPLEMENTED` con motivo explícito en `detail`.
- **Todos los comandos de test corren con `/home/fruiz/jax/.venv/bin/python -m pytest`** desde la raíz del repo — no con `las_manos/.venv/bin/python` (le falta PyYAML) ni con `pytest` a secas (puede resolver a un intérprete sin las deps).
- **No correr `pytest -q` ciego desde la raíz del repo.** `_director_patch/test_jacobs_director.py` rompe la colección entera con `INTERNALERROR` (deuda preexistente, confirmada en `master` sin tocar — ver memoria `reformas-v3-progreso`). Apuntar siempre a archivos/paths específicos.

---

## File Structure

```
policy/
  VERSION                          # MODIFICAR — agregar línea templates_sha256:
  tools/
    template_hash.py               # CREAR — hash de render_templates.yaml
  governance/
    claims.py                      # CREAR — schema Pydantic puro (Claim)
    loaders.py                     # CREAR — I/O config estática, fail-closed
    validator.py                   # CREAR — I/O estado vivo + resolvers + Verdict
    vocab_sweep.py                 # CREAR — barrido léxico puro
    renderer.py                    # CREAR — motor de plantillas puro
tests/
  test_template_hash.py            # CREAR
  test_governance_loaders.py       # CREAR
  test_governance_claims.py        # CREAR
  test_governance_validator.py     # CREAR
  test_governance_vocab_sweep.py   # CREAR
  test_governance_renderer.py      # CREAR
```

No se crea `policy/governance/__init__.py`. El repo no usa paquetes instalados (no hay `pyproject.toml`/`setup.py`): cada test inserta el directorio del módulo directamente en `sys.path` e importa por nombre plano (`import loaders`, `import claims`, ...) — mismo patrón que `tests/test_envelope_brutal.py` usa para `las_manos/` (`sys.path.insert(0, str(LAS_MANOS)); import envelope`). `validator.py` es la única excepción: necesita importar `las_manos.motor_registry.catalog` y `las_manos.envelope` como paquetes reales, lo cual funciona porque `las_manos/__init__.py` y `las_manos/motor_registry/__init__.py` ya existen (vacíos) — inserta la RAÍZ del repo en `sys.path`, no `las_manos/` directamente.

---

### Task 1: `policy/tools/template_hash.py` — hash de templates

**Files:**
- Create: `policy/tools/template_hash.py`
- Test: `tests/test_template_hash.py`

**Interfaces:**
- Produces: `compute_hash() -> str` (sha256 hex de `policy/templates/render_templates.yaml`), usado por Task 3 (`loaders.py`) como referencia de qué algoritmo debe reproducir `_current_templates_hash()`.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_template_hash.py
"""
Test de policy/tools/template_hash.py — hash de render_templates.yaml,
mismo algoritmo que corpus_hash.py pero para un solo archivo.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

POLICY_TOOLS = Path(__file__).resolve().parent.parent / "policy" / "tools"
sys.path.insert(0, str(POLICY_TOOLS))

import template_hash  # noqa: E402

TEMPLATES_FILE = Path(__file__).resolve().parent.parent / "policy" / "templates" / "render_templates.yaml"


def test_compute_hash_matches_manual_sha256():
    expected = hashlib.sha256(TEMPLATES_FILE.read_bytes()).hexdigest()
    assert template_hash.compute_hash() == expected
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_template_hash.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'template_hash'`

- [ ] **Step 3: Implementar `template_hash.py`**

```python
#!/usr/bin/env python3
"""
Calcula el sha256 de policy/templates/render_templates.yaml, byte a byte
— mismo algoritmo que policy/tools/corpus_hash.py (que hashea el corpus
de rules/), aplicado a un solo archivo en vez de a un directorio.

Uso:
    policy/tools/template_hash.py            imprime el hash
    policy/tools/template_hash.py --write    lo escribe en VERSION, línea
                                              'templates_sha256:' (nueva
                                              si no existe, reemplazada si
                                              ya existe)
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

POLICY_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_FILE = POLICY_DIR / "templates" / "render_templates.yaml"
VERSION_FILE = POLICY_DIR / "VERSION"


def compute_hash() -> str:
    return hashlib.sha256(TEMPLATES_FILE.read_bytes()).hexdigest()


def main() -> int:
    if not TEMPLATES_FILE.exists():
        print(f"ERROR: no existe {TEMPLATES_FILE}", file=sys.stderr)
        return 1

    digest = compute_hash()
    print(f"templates_sha256: {digest}")

    if "--write" in sys.argv:
        if not VERSION_FILE.exists():
            print(f"ERROR: no existe {VERSION_FILE}", file=sys.stderr)
            return 1
        content = VERSION_FILE.read_text(encoding="utf-8")
        if re.search(r"^templates_sha256:.*$", content, flags=re.MULTILINE):
            new_content = re.sub(
                r"^templates_sha256:.*$",
                f"templates_sha256: {digest}",
                content,
                count=1,
                flags=re.MULTILINE,
            )
        else:
            sep = "" if content.endswith("\n") else "\n"
            new_content = f"{content}{sep}templates_sha256: {digest}\n"
        VERSION_FILE.write_text(new_content, encoding="utf-8")
        print(f"\nVERSION actualizado: {VERSION_FILE}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_template_hash.py -v`
Expected: PASS

- [ ] **Step 5: Escribir el hash real en VERSION**

Run: `/home/fruiz/jax/.venv/bin/python policy/tools/template_hash.py --write`

Verificar que `policy/VERSION` quedó con tres líneas (`version:`, `sha256:`, `templates_sha256:`). Este paso es real, no un test — sin él, Task 3 no tiene un hash correcto contra el cual `loaders.py` pueda validar en el caso feliz.

- [ ] **Step 6: Commit**

```bash
git add policy/tools/template_hash.py tests/test_template_hash.py policy/VERSION
git commit -m "policy: agregar template_hash.py y hash de render_templates.yaml en VERSION"
```

---

### Task 2: `policy/governance/claims.py` — schema de claim

**Files:**
- Create: `policy/governance/claims.py`
- Test: `tests/test_governance_claims.py`

**Interfaces:**
- Produces: `class Claim(BaseModel)` con campos `predicate: str`, `args: dict[str, str]`, `authority: Literal["EJECUTADO", "OBSERVADO", "RECUPERADO", "INFERIDO"]`, `provenance_ref: str`, `evidence_pointer: str`, `scope: str`. Usado por Task 4 (`validator.py`), Task 6 (`vocab_sweep.py` indirectamente vía el llamador) y Task 7 (`renderer.py`).

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_governance_claims.py
"""
Test de policy/governance/claims.py — schema estructural del claim.
Dos capas de validación en el subsistema: esta (Pydantic) y la semántica
de validator.py (Task 4).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

GOVERNANCE = Path(__file__).resolve().parent.parent / "policy" / "governance"
sys.path.insert(0, str(GOVERNANCE))

from pydantic import ValidationError

import claims  # noqa: E402


def _valid_kwargs(**overrides):
    base = dict(
        predicate="FILE_EXISTS",
        args={"path": "las_manos/config.toml", "hash": "a" * 64},
        authority="OBSERVADO",
        provenance_ref="tool_call:read_file:abc123",
        evidence_pointer="las_manos/config.toml",
        scope="jax",
    )
    base.update(overrides)
    return base


def test_claim_valid_construction():
    claim = claims.Claim(**_valid_kwargs())
    assert claim.predicate == "FILE_EXISTS"
    assert claim.authority == "OBSERVADO"


def test_claim_rejects_invalid_authority():
    with pytest.raises(ValidationError):
        claims.Claim(**_valid_kwargs(authority="ADIVINADO"))


def test_claim_rejects_missing_field():
    kwargs = _valid_kwargs()
    del kwargs["provenance_ref"]
    with pytest.raises(ValidationError):
        claims.Claim(**kwargs)


def test_claim_accepts_all_four_authority_values():
    for value in ("EJECUTADO", "OBSERVADO", "RECUPERADO", "INFERIDO"):
        claim = claims.Claim(**_valid_kwargs(authority=value))
        assert claim.authority == value
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_claims.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'claims'`

- [ ] **Step 3: Implementar `claims.py`**

```python
"""
policy/governance — Schema de claim.

REFORMAS-v3.md §3.1 (R1). Capa 1 de dos: estructural (este módulo,
Pydantic) y semántica (validator.py). Un claim mal tipado no llega nunca
al validador semántico — mismo principio de dos capas que
las_manos/envelope.py usa para IntentEnvelope.

Este módulo es PURO: sin I/O, sin red, testeable en aislamiento.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Claim(BaseModel):
    predicate: str
    args: dict[str, str]
    authority: Literal["EJECUTADO", "OBSERVADO", "RECUPERADO", "INFERIDO"]
    provenance_ref: str
    evidence_pointer: str
    scope: str
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_claims.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add policy/governance/claims.py tests/test_governance_claims.py
git commit -m "policy: agregar schema Claim (governance sub-proyecto 1, Task 2)"
```

---

### Task 3: `policy/governance/loaders.py` — I/O de config estática, fail-closed primero

**Files:**
- Create: `policy/governance/loaders.py`
- Test: `tests/test_governance_loaders.py`

**Interfaces:**
- Consumes: nada de tasks anteriores.
- Produces: `PredicateSpec` (dataclass: `name: str`, `args: tuple[str, ...]`, `source_of_truth: str`), `TemplateSpec` (dataclass: `status: str`, `template: str | None`), `ClosedVocabulary` (dataclass: `flattened: frozenset[str]`, `config_paths: frozenset[str]`), `load_predicates() -> dict[str, PredicateSpec]`, `load_vocabulary() -> ClosedVocabulary`, `load_templates() -> dict[str, TemplateSpec]`. Usados por Task 4 (`validator.py`, predicates + vocabulary), Task 6 (`vocab_sweep.py` vía el llamador, `.flattened`) y Task 7 (`renderer.py`, templates).

**El primer test de este task es el fail-closed de `load_templates()`, no el último.** Es el que verifica que el subsistema no arranca con hash roto — la clase de prueba que faltó en `backup-hall9000.sh` (retención fail-open, un mes de fallo diario invisible porque nadie probó que fallara cuando debía). Si este test quedara al final de la lista, es el primero que se recorta cuando aparece presión de tiempo; por eso va primero.

- [ ] **Step 1: Escribir el test que falla — fail-closed por hash roto**

```python
# tests/test_governance_loaders.py
"""
Test de policy/governance/loaders.py — I/O de config ESTÁTICA de policy
(predicates.yaml, closed_vocabulary.yaml, render_templates.yaml + hash).

El primer test es el fail-closed de load_templates(): si el hash de
render_templates.yaml no coincide con el registrado en VERSION, el
subsistema no debe cargar nada — ni parcial, ni con warning. Va primero
a propósito (ver plan, Task 3): es la prueba que faltó en
backup-hall9000.sh y no debe ser la que se recorta bajo presión.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

GOVERNANCE = Path(__file__).resolve().parent.parent / "policy" / "governance"
sys.path.insert(0, str(GOVERNANCE))

import loaders  # noqa: E402


def test_load_templates_fails_closed_on_hash_mismatch(tmp_path, monkeypatch):
    templates_file = tmp_path / "render_templates.yaml"
    templates_file.write_text(
        "templates:\n  FOO:\n    status: definida\n    template: 'x'\n",
        encoding="utf-8",
    )
    version_file = tmp_path / "VERSION"
    version_file.write_text(
        "version: 0.1.0\nsha256: deadbeef\ntemplates_sha256: 0000000000\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(loaders, "TEMPLATES_FILE", templates_file)
    monkeypatch.setattr(loaders, "VERSION_FILE", version_file)

    with pytest.raises(RuntimeError, match="Hash de"):
        loaders.load_templates()
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_loaders.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'loaders'`

- [ ] **Step 3: Implementar el fail-closed de `load_templates()` (mínimo)**

```python
# policy/governance/loaders.py
"""
policy/governance — I/O de config ESTÁTICA de policy: predicates.yaml,
closed_vocabulary.yaml, render_templates.yaml + hash.

Config versionada por commit. Si `load_templates()` no puede verificar el
hash contra VERSION, el subsistema NO ARRANCA — no hay carga parcial ni
degradación silenciosa. Mismo patrón de falla que ya causó tres
incidentes independientes en este ecosistema (retención de backups,
_HTTP_FACETS, output_validator.py de motor_registry): este módulo existe
específicamente para no ser el cuarto.

Distinto de validator.py: ese lee ESTADO VIVO (config.toml, filesystem),
que cambia sin commits — si un resolver individual falla ahí, es un
rechazo de ese claim, no una falla de arranque del subsistema entero.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

GOVERNANCE_DIR = Path(__file__).resolve().parent
POLICY_DIR = GOVERNANCE_DIR.parent
PREDICATES_FILE = POLICY_DIR / "vocabulary" / "predicates.yaml"
VOCABULARY_FILE = POLICY_DIR / "vocabulary" / "closed_vocabulary.yaml"
TEMPLATES_FILE = POLICY_DIR / "templates" / "render_templates.yaml"
VERSION_FILE = POLICY_DIR / "VERSION"


@dataclass(frozen=True)
class TemplateSpec:
    status: str
    template: str | None


def _current_templates_hash() -> str:
    return hashlib.sha256(TEMPLATES_FILE.read_bytes()).hexdigest()


def _recorded_templates_hash() -> str:
    content = VERSION_FILE.read_text(encoding="utf-8")
    match = re.search(r"^templates_sha256:\s*(\S+)\s*$", content, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(
            f"{VERSION_FILE} no tiene línea 'templates_sha256:' — correr "
            "policy/tools/template_hash.py --write"
        )
    return match.group(1)


def load_templates() -> dict[str, TemplateSpec]:
    current = _current_templates_hash()
    recorded = _recorded_templates_hash()
    if current != recorded:
        raise RuntimeError(
            f"Hash de {TEMPLATES_FILE} no coincide con VERSION (esperado "
            f"{recorded}, calculado {current}) — el subsistema de "
            "gobernanza no arranca con templates sin verificar."
        )
    return {}
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_loaders.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add policy/governance/loaders.py tests/test_governance_loaders.py
git commit -m "policy: fail-closed de load_templates() por hash roto (governance, Task 3.1)"
```

- [ ] **Step 6: Escribir el test que falla — `load_templates()` caso feliz**

Agregar a `tests/test_governance_loaders.py`:

```python
def test_load_templates_happy_path_returns_real_specs():
    templates = loaders.load_templates()
    assert templates["CAPABILITY_AVAILABLE"].status == "definida"
    assert templates["CAPABILITY_AVAILABLE"].template == (
        "La capability {name} está disponible en modo {mode}."
    )
    assert templates["FACET_EXISTS"].status == "pendiente"
    assert templates["FACET_EXISTS"].template is None
```

- [ ] **Step 7: Correr el test y verificar que falla**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_loaders.py -v`
Expected: FAIL — `load_templates()` hoy devuelve `{}` siempre (implementación mínima del Step 3)

- [ ] **Step 8: Completar `load_templates()` para parsear el YAML real**

En `policy/governance/loaders.py`, reemplazar el `return {}` final de `load_templates()`:

```python
import yaml  # agregar al bloque de imports, junto a hashlib/re

# ... (sin cambios arriba) ...

def load_templates() -> dict[str, TemplateSpec]:
    current = _current_templates_hash()
    recorded = _recorded_templates_hash()
    if current != recorded:
        raise RuntimeError(
            f"Hash de {TEMPLATES_FILE} no coincide con VERSION (esperado "
            f"{recorded}, calculado {current}) — el subsistema de "
            "gobernanza no arranca con templates sin verificar."
        )
    data = yaml.safe_load(TEMPLATES_FILE.read_text(encoding="utf-8"))
    return {
        name: TemplateSpec(status=entry["status"], template=entry.get("template"))
        for name, entry in data["templates"].items()
    }
```

- [ ] **Step 9: Correr el test y verificar que pasa**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_loaders.py -v`
Expected: PASS (2 tests)

- [ ] **Step 10: Commit**

```bash
git add policy/governance/loaders.py tests/test_governance_loaders.py
git commit -m "policy: load_templates() parsea render_templates.yaml (governance, Task 3.2)"
```

- [ ] **Step 11: Escribir el test que falla — `load_predicates()`**

Agregar a `tests/test_governance_loaders.py`:

```python
def test_load_predicates_returns_all_eight():
    predicates = loaders.load_predicates()
    assert len(predicates) == 8
    assert predicates["CAPABILITY_AVAILABLE"].args == ("name", "mode")
    assert predicates["FILE_EXISTS"].args == ("path", "hash")
    assert predicates["MEMORY_ENTRY_EXISTS"].source_of_truth == "MariaDB jax_memory"
```

- [ ] **Step 12: Correr el test y verificar que falla**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_loaders.py -v`
Expected: FAIL con `AttributeError: module 'loaders' has no attribute 'load_predicates'`

- [ ] **Step 13: Implementar `load_predicates()`**

En `policy/governance/loaders.py`, agregar debajo de `TemplateSpec`:

```python
@dataclass(frozen=True)
class PredicateSpec:
    name: str
    args: tuple[str, ...]
    source_of_truth: str
```

Y al final del archivo:

```python
def load_predicates() -> dict[str, PredicateSpec]:
    data = yaml.safe_load(PREDICATES_FILE.read_text(encoding="utf-8"))
    return {
        entry["name"]: PredicateSpec(
            name=entry["name"],
            args=tuple(entry["args"]),
            source_of_truth=entry["source_of_truth"],
        )
        for entry in data["predicates"]
    }
```

- [ ] **Step 14: Correr el test y verificar que pasa**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_loaders.py -v`
Expected: PASS (3 tests)

- [ ] **Step 15: Commit**

```bash
git add policy/governance/loaders.py tests/test_governance_loaders.py
git commit -m "policy: load_predicates() (governance, Task 3.3)"
```

- [ ] **Step 16: Escribir el test que falla — `load_vocabulary()`**

Agregar a `tests/test_governance_loaders.py`:

```python
def test_load_vocabulary_flattens_categories_and_keeps_config_paths_separate():
    vocab = loaders.load_vocabulary()
    assert "code_swarm" in vocab.flattened   # capabilities
    assert "ssh_exec" in vocab.flattened     # ops
    assert "hyde" in vocab.flattened         # facets_las_manos / facets_jax
    assert "jax/policy/" in vocab.config_paths
    assert "las_manos/config.toml" in vocab.config_paths
    # config_paths no debe filtrarse al vocabulario léxico plano
    assert "jax/policy/" not in vocab.flattened
```

- [ ] **Step 17: Correr el test y verificar que falla**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_loaders.py -v`
Expected: FAIL con `AttributeError: module 'loaders' has no attribute 'load_vocabulary'`

- [ ] **Step 18: Implementar `load_vocabulary()`**

En `policy/governance/loaders.py`, agregar debajo de `PredicateSpec`:

```python
@dataclass(frozen=True)
class ClosedVocabulary:
    flattened: frozenset[str]
    config_paths: frozenset[str]
```

Y al final del archivo:

```python
def load_vocabulary() -> ClosedVocabulary:
    data = yaml.safe_load(VOCABULARY_FILE.read_text(encoding="utf-8"))
    flattened: set[str] = set()
    for key, value in data.items():
        if key == "config_paths":
            continue
        if isinstance(value, dict):
            flattened.update(value.keys())
    config_paths = frozenset(data.get("config_paths") or [])
    return ClosedVocabulary(flattened=frozenset(flattened), config_paths=config_paths)
```

- [ ] **Step 19: Correr el test y verificar que pasa**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_loaders.py -v`
Expected: PASS (4 tests)

- [ ] **Step 20: Commit**

```bash
git add policy/governance/loaders.py tests/test_governance_loaders.py
git commit -m "policy: load_vocabulary() (governance, Task 3.4 — loaders.py completo)"
```

---

### Task 4: `policy/governance/validator.py` — validación semántica y resolvers

**Files:**
- Create: `policy/governance/validator.py`
- Test: `tests/test_governance_validator.py`

**Interfaces:**
- Consumes: `claims.Claim` (Task 2); `loaders.PredicateSpec`, `loaders.ClosedVocabulary` (Task 3); `las_manos.motor_registry.catalog.MotorCatalog` (existente); `las_manos.envelope.MUTATING_CAPABILITIES` (existente).
- Produces: `class Verdict(BaseModel)` con `status: Literal["VALID", "UNKNOWN_PREDICATE", "ARGS_MISMATCH", "RESOLVER_NOT_IMPLEMENTED", "FACT_MISMATCH", "AUTHORITY_INVALID", "SOURCE_CONFLICT", "PATH_NOT_ALLOWED"]`, `predicate: str`, `detail: str`; `ValidationContext` (dataclass); `load_validation_context(repo_root: Path, config_paths_allowlist: frozenset[str]) -> ValidationContext`; `validate(claim: Claim, predicates: dict[str, PredicateSpec], ctx: ValidationContext) -> Verdict`. Usado por Task 7 (`renderer.py` — solo consume `Verdict`, indirectamente vía el llamador que decide si renderiza).

Este task tiene tres partes con requisitos explícitos del diseño, cada una verificada con test, no solo por lectura del código:
1. **Chequeos estructurales primero** (orden fijo: predicado conocido → args coinciden → `authority != INFERIDO`), antes de tocar cualquier resolver.
2. **`FILE_EXISTS`: allowlist antes que `stat()`/`exists()`/lectura de bytes**, siempre — con un test que lo prueba forzando una excepción si el código toca el filesystem para un path no permitido, no solo verificándolo por lectura.
3. **`RESOLVER_NOT_IMPLEMENTED` es un rechazo real** — un claim de `ENGINE_STATUS` (o cualquiera de los otros 5 sin resolver) nunca puede terminar en `VALID`, con test explícito. Esto es fácil de "arreglar" en seis meses para que dé menos ruido; el test es lo que lo impide.

- [ ] **Step 1: Escribir los tests que fallan — chequeos estructurales**

```python
# tests/test_governance_validator.py
"""
Test de policy/governance/validator.py — validación semántica de claims
contra fuentes reales (config.toml, motor_registry, filesystem).

Orden de este archivo, a propósito:
  1. Chequeos estructurales (predicado conocido, args, authority) — antes
     de cualquier resolver.
  2. RESOLVER_NOT_IMPLEMENTED como rechazo real (ENGINE_STATUS nunca
     pasa silenciosamente).
  3. FILE_EXISTS: allowlist antes que filesystem, con test que lo hace
     explotar si el código toca disco para un path no permitido.
  4. CAPABILITY_AVAILABLE contra el config.toml real.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GOVERNANCE = REPO_ROOT / "policy" / "governance"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GOVERNANCE))

import claims  # noqa: E402
import loaders  # noqa: E402
import validator  # noqa: E402
from las_manos.motor_registry.catalog import MotorCatalog  # noqa: E402

FILE_EXISTS_SPEC = loaders.PredicateSpec(
    name="FILE_EXISTS", args=("path", "hash"), source_of_truth="Sistema de archivos"
)
CAPABILITY_AVAILABLE_SPEC = loaders.PredicateSpec(
    name="CAPABILITY_AVAILABLE", args=("name", "mode"), source_of_truth="Registro de capabilities"
)
ENGINE_STATUS_SPEC = loaders.PredicateSpec(
    name="ENGINE_STATUS", args=("name", "status"), source_of_truth="Health check"
)
PREDICATES = {
    "FILE_EXISTS": FILE_EXISTS_SPEC,
    "CAPABILITY_AVAILABLE": CAPABILITY_AVAILABLE_SPEC,
    "ENGINE_STATUS": ENGINE_STATUS_SPEC,
}


def _claim(**overrides):
    base = dict(
        predicate="FILE_EXISTS",
        args={"path": "las_manos/config.toml", "hash": "a" * 64},
        authority="OBSERVADO",
        provenance_ref="test",
        evidence_pointer="test",
        scope="jax",
    )
    base.update(overrides)
    return claims.Claim(**base)


def _empty_ctx() -> "validator.ValidationContext":
    return validator.ValidationContext(
        ops=frozenset(),
        mutating_capabilities=frozenset(),
        catalog=MotorCatalog({}),
        config_paths_allowlist=frozenset(),
        repo_root=REPO_ROOT,
    )


def test_validate_unknown_predicate():
    claim = _claim(predicate="NOT_A_REAL_PREDICATE", args={})
    verdict = validator.validate(claim, PREDICATES, _empty_ctx())
    assert verdict.status == "UNKNOWN_PREDICATE"


def test_validate_args_mismatch():
    claim = _claim(predicate="FILE_EXISTS", args={"path": "x"})  # falta 'hash'
    verdict = validator.validate(claim, PREDICATES, _empty_ctx())
    assert verdict.status == "ARGS_MISMATCH"


def test_validate_authority_inferido_rejected():
    claim = _claim(authority="INFERIDO")
    verdict = validator.validate(claim, PREDICATES, _empty_ctx())
    assert verdict.status == "AUTHORITY_INVALID"
```

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_validator.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'validator'`

- [ ] **Step 3: Implementar `validator.py` — `Verdict`, `ValidationContext` y los tres chequeos estructurales**

```python
# policy/governance/validator.py
"""
policy/governance — Validación semántica de claims contra fuentes reales.

Capa 2 de dos (la 1 es claims.py, estructural). Recibe un Claim ya válido
estructuralmente y lo despacha contra las fuentes de verdad reales del
sistema (config.toml de las_manos, motor_registry, filesystem).

Distinto de loaders.py: ese lee config ESTÁTICA versionada por commit (si
falla, el subsistema no arranca). Este lee ESTADO VIVO que cambia sin
commits — si un resolver individual falla, es un rechazo de ESE claim,
no una falla de arranque.

RESOLVER_NOT_IMPLEMENTED es un veredicto, no una excepción sin capturar:
P07 ("no existe bypass en producción") aplicado al tipo de retorno — no
hay forma de expresar "falló pero seguí" porque Verdict no lo permite.

Este módulo SÍ hace I/O (config.toml, filesystem) — a diferencia de
claims.py, vocab_sweep.py y renderer.py, que son puros.
"""
from __future__ import annotations

import hashlib
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from las_manos.envelope import MUTATING_CAPABILITIES  # noqa: E402
from las_manos.motor_registry.catalog import MotorCatalog  # noqa: E402

import claims  # noqa: E402


class Verdict(BaseModel):
    status: Literal[
        "VALID",
        "UNKNOWN_PREDICATE",
        "ARGS_MISMATCH",
        "RESOLVER_NOT_IMPLEMENTED",
        "FACT_MISMATCH",
        "AUTHORITY_INVALID",
        "SOURCE_CONFLICT",
        "PATH_NOT_ALLOWED",
    ]
    predicate: str
    detail: str


@dataclass(frozen=True)
class ValidationContext:
    ops: frozenset[str]
    mutating_capabilities: frozenset[str]
    catalog: MotorCatalog
    config_paths_allowlist: frozenset[str]
    repo_root: Path


def load_validation_context(
    repo_root: Path, config_paths_allowlist: frozenset[str]
) -> ValidationContext:
    config_path = repo_root / "las_manos" / "config.toml"
    with config_path.open("rb") as f:
        config = tomllib.load(f)
    return ValidationContext(
        ops=frozenset(config.get("ops", {}).keys()),
        mutating_capabilities=frozenset(MUTATING_CAPABILITIES),
        catalog=MotorCatalog(config),
        config_paths_allowlist=config_paths_allowlist,
        repo_root=repo_root,
    )


_RESOLVERS: dict[str, Callable[["claims.Claim", ValidationContext], Verdict]] = {}
_UNIMPLEMENTED_REASONS: dict[str, str] = {}


def validate(
    claim: "claims.Claim", predicates: dict, ctx: ValidationContext
) -> Verdict:
    spec = predicates.get(claim.predicate)
    if spec is None:
        return Verdict(
            status="UNKNOWN_PREDICATE",
            predicate=claim.predicate,
            detail=f"'{claim.predicate}' no está en predicates.yaml.",
        )

    if set(claim.args.keys()) != set(spec.args):
        return Verdict(
            status="ARGS_MISMATCH",
            predicate=claim.predicate,
            detail=(
                f"Args esperados {sorted(spec.args)}, "
                f"recibidos {sorted(claim.args.keys())}."
            ),
        )

    if claim.authority == "INFERIDO":
        return Verdict(
            status="AUTHORITY_INVALID",
            predicate=claim.predicate,
            detail="authority=INFERIDO prohibido en canal claim (§3.1.4).",
        )

    resolver = _RESOLVERS.get(claim.predicate)
    if resolver is not None:
        return resolver(claim, ctx)

    reason = _UNIMPLEMENTED_REASONS.get(
        claim.predicate, f"{claim.predicate}: resolver no implementado."
    )
    return Verdict(
        status="RESOLVER_NOT_IMPLEMENTED", predicate=claim.predicate, detail=reason
    )
```

- [ ] **Step 4: Correr los tests y verificar que pasan**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_validator.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add policy/governance/validator.py tests/test_governance_validator.py
git commit -m "policy: Verdict + chequeos estructurales de validate() (governance, Task 4.1)"
```

- [ ] **Step 6: Escribir el test que falla — `RESOLVER_NOT_IMPLEMENTED` como rechazo real**

Agregar a `tests/test_governance_validator.py`:

```python
def test_engine_status_is_resolver_not_implemented_never_valid():
    claim = _claim(
        predicate="ENGINE_STATUS",
        args={"name": "kimi", "status": "healthy"},
    )
    verdict = validator.validate(claim, PREDICATES, _empty_ctx())
    assert verdict.status == "RESOLVER_NOT_IMPLEMENTED"
    assert verdict.status != "VALID"
    assert "ENGINE_STATUS" in verdict.detail
```

- [ ] **Step 7: Correr el test y verificar que falla**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_validator.py -v`
Expected: FAIL — hoy `ENGINE_STATUS` no está en `_UNIMPLEMENTED_REASONS`, cae en el mensaje genérico sin mencionar "ENGINE_STATUS" en el texto explicativo real (el assert de contenido del detail falla)

- [ ] **Step 8: Poblar `_UNIMPLEMENTED_REASONS` con los 6 predicados sin resolver**

En `policy/governance/validator.py`, reemplazar la línea `_UNIMPLEMENTED_REASONS: dict[str, str] = {}`:

```python
_UNIMPLEMENTED_REASONS: dict[str, str] = {
    "ENGINE_STATUS": (
        "ENGINE_STATUS: sin fuente de verdad en el dominio de jax. La "
        "tabla 'model' de jax-platform tiene semántica distinta "
        "(disponibilidad del provider, no salud del motor). Ver "
        "REFORMAS-v3 §3.1.3 y el spec de gobernanza, sección 'Por qué "
        "solo dos resolvers reales'."
    ),
    "FACET_EXISTS": (
        "FACET_EXISTS: sin fuente ni consumidor identificado esta "
        "ronda. Ver spec de gobernanza, 'Fuera de alcance, "
        "explícitamente'."
    ),
    "CONFIG_VALUE": (
        "CONFIG_VALUE: sin fuente ni consumidor identificado esta "
        "ronda. Ver spec de gobernanza, 'Fuera de alcance, "
        "explícitamente'."
    ),
    "AUDIT_EVENT_EXISTS": (
        "AUDIT_EVENT_EXISTS: sin fuente ni consumidor identificado esta "
        "ronda. Ver spec de gobernanza, 'Fuera de alcance, "
        "explícitamente'."
    ),
    "JOB_STATUS": (
        "JOB_STATUS: sin fuente ni consumidor identificado esta ronda. "
        "Ver spec de gobernanza, 'Fuera de alcance, explícitamente'."
    ),
    "MEMORY_ENTRY_EXISTS": (
        "MEMORY_ENTRY_EXISTS: sin fuente ni consumidor identificado "
        "esta ronda. Ver spec de gobernanza, 'Fuera de alcance, "
        "explícitamente'."
    ),
}
```

- [ ] **Step 9: Correr el test y verificar que pasa**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_validator.py -v`
Expected: PASS (4 tests)

- [ ] **Step 10: Commit**

```bash
git add policy/governance/validator.py tests/test_governance_validator.py
git commit -m "policy: RESOLVER_NOT_IMPLEMENTED con motivo por predicado (governance, Task 4.2)"
```

- [ ] **Step 11: Escribir el test que falla — `FILE_EXISTS` allowlist antes que filesystem**

Agregar a `tests/test_governance_validator.py`:

```python
def test_file_exists_rejects_path_outside_allowlist_without_touching_disk(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError(
            "tocó el filesystem antes de chequear la allowlist — "
            "PATH_NOT_ALLOWED debe devolverse sin exists()/read_bytes()"
        )

    monkeypatch.setattr(Path, "exists", _boom)
    monkeypatch.setattr(Path, "read_bytes", _boom)

    ctx = validator.ValidationContext(
        ops=frozenset(),
        mutating_capabilities=frozenset(),
        catalog=MotorCatalog({}),
        config_paths_allowlist=frozenset({"las_manos/config.toml"}),
        repo_root=REPO_ROOT,
    )
    claim = _claim(args={"path": "/etc/shadow", "hash": "0" * 64})

    verdict = validator.validate(claim, PREDICATES, ctx)

    assert verdict.status == "PATH_NOT_ALLOWED"


def test_file_exists_allowed_path_valid_hash():
    real_file = REPO_ROOT / "las_manos" / "config.toml"
    actual_hash = hashlib.sha256(real_file.read_bytes()).hexdigest()
    ctx = validator.ValidationContext(
        ops=frozenset(),
        mutating_capabilities=frozenset(),
        catalog=MotorCatalog({}),
        config_paths_allowlist=frozenset({"las_manos/config.toml"}),
        repo_root=REPO_ROOT,
    )
    claim = _claim(args={"path": "las_manos/config.toml", "hash": actual_hash})

    verdict = validator.validate(claim, PREDICATES, ctx)

    assert verdict.status == "VALID"


def test_file_exists_allowed_path_wrong_hash():
    ctx = validator.ValidationContext(
        ops=frozenset(),
        mutating_capabilities=frozenset(),
        catalog=MotorCatalog({}),
        config_paths_allowlist=frozenset({"las_manos/config.toml"}),
        repo_root=REPO_ROOT,
    )
    claim = _claim(args={"path": "las_manos/config.toml", "hash": "f" * 64})

    verdict = validator.validate(claim, PREDICATES, ctx)

    assert verdict.status == "FACT_MISMATCH"


def test_file_exists_allowed_directory_prefix_nonexistent_file():
    ctx = validator.ValidationContext(
        ops=frozenset(),
        mutating_capabilities=frozenset(),
        catalog=MotorCatalog({}),
        config_paths_allowlist=frozenset({"policy/"}),
        repo_root=REPO_ROOT,
    )
    claim = _claim(args={"path": "policy/no_existe_este_archivo.yaml", "hash": "0" * 64})

    verdict = validator.validate(claim, PREDICATES, ctx)

    assert verdict.status == "FACT_MISMATCH"
```

- [ ] **Step 12: Correr los tests y verificar que fallan**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_validator.py -v`
Expected: FAIL — `FILE_EXISTS` no está en `_RESOLVERS`, cae en `RESOLVER_NOT_IMPLEMENTED` para las 4 aserciones nuevas

- [ ] **Step 13: Implementar el resolver `FILE_EXISTS`**

En `policy/governance/validator.py`, agregar antes de `_RESOLVERS`:

```python
def _normalize_path(path: str, repo_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def _path_allowed(path: str, allowlist: frozenset[str], repo_root: Path) -> bool:
    """Solo aritmética de paths — CERO llamadas a exists()/stat()/read_bytes()."""
    candidate = _normalize_path(path, repo_root)
    for entry in allowlist:
        is_dir_entry = entry.endswith("/")
        entry_path = _normalize_path(
            entry.rstrip("/") if is_dir_entry else entry, repo_root
        )
        if is_dir_entry:
            try:
                candidate.relative_to(entry_path)
                return True
            except ValueError:
                continue
        elif candidate == entry_path:
            return True
    return False


def _resolve_file_exists(claim: "claims.Claim", ctx: ValidationContext) -> Verdict:
    path = claim.args["path"]
    expected_hash = claim.args["hash"]

    if not _path_allowed(path, ctx.config_paths_allowlist, ctx.repo_root):
        return Verdict(
            status="PATH_NOT_ALLOWED",
            predicate="FILE_EXISTS",
            detail="Path fuera de la allowlist de config_paths.",
        )

    candidate = _normalize_path(path, ctx.repo_root)
    if not candidate.exists():
        return Verdict(
            status="FACT_MISMATCH", predicate="FILE_EXISTS", detail=f"'{path}' no existe."
        )

    actual_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        return Verdict(
            status="FACT_MISMATCH",
            predicate="FILE_EXISTS",
            detail=(
                f"'{path}' existe pero su hash no coincide (esperado "
                f"{expected_hash}, real {actual_hash})."
            ),
        )
    return Verdict(
        status="VALID", predicate="FILE_EXISTS", detail=f"'{path}' existe con hash verificado."
    )
```

Y cambiar la línea `_RESOLVERS: dict[...] = {}` a:

```python
_RESOLVERS: dict[str, Callable[["claims.Claim", ValidationContext], Verdict]] = {
    "FILE_EXISTS": _resolve_file_exists,
}
```

- [ ] **Step 14: Correr los tests y verificar que pasan**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_validator.py -v`
Expected: PASS (8 tests)

- [ ] **Step 15: Commit**

```bash
git add policy/governance/validator.py tests/test_governance_validator.py
git commit -m "policy: resolver FILE_EXISTS, allowlist antes que filesystem (governance, Task 4.3)"
```

- [ ] **Step 16: Escribir los tests que fallan — `CAPABILITY_AVAILABLE`**

Agregar a `tests/test_governance_validator.py`:

```python
def _real_ctx() -> "validator.ValidationContext":
    vocab = loaders.load_vocabulary()
    return validator.load_validation_context(REPO_ROOT, vocab.config_paths)


def test_capability_available_found_only_in_ops_read_only_mode_matches():
    ctx = _real_ctx()
    claim = _claim(
        predicate="CAPABILITY_AVAILABLE",
        args={"name": "ssh_exec_readonly", "mode": "read_only"},
    )
    verdict = validator.validate(claim, PREDICATES, ctx)
    assert verdict.status == "VALID"


def test_capability_available_found_only_in_ops_mode_mismatch():
    ctx = _real_ctx()
    claim = _claim(
        predicate="CAPABILITY_AVAILABLE",
        args={"name": "ssh_exec", "mode": "read_only"},  # ssh_exec ES mutante
    )
    verdict = validator.validate(claim, PREDICATES, ctx)
    assert verdict.status == "FACT_MISMATCH"


def test_capability_available_found_only_in_catalog_mode_unverified_but_accepted():
    ctx = _real_ctx()
    claim = _claim(
        predicate="CAPABILITY_AVAILABLE",
        args={"name": "code_swarm", "mode": "read_only"},
    )
    verdict = validator.validate(claim, PREDICATES, ctx)
    assert verdict.status == "VALID"


def test_capability_available_not_found_anywhere():
    ctx = _real_ctx()
    claim = _claim(
        predicate="CAPABILITY_AVAILABLE",
        args={"name": "totalmente_inventado_xyz", "mode": "read_only"},
    )
    verdict = validator.validate(claim, PREDICATES, ctx)
    assert verdict.status == "FACT_MISMATCH"


def test_capability_available_source_conflict_when_present_in_both():
    # Sintético a propósito: hoy [ops.*] y [capabilities.*] son disjuntos
    # en config.toml (0 solapamiento, verificado en el spec). Se fabrica
    # el conflicto a mano para probar la rama SOURCE_CONFLICT sin
    # depender de que el config real cambie.
    ctx = validator.ValidationContext(
        ops=frozenset({"code_swarm"}),
        mutating_capabilities=frozenset(),
        catalog=MotorCatalog({"capabilities": {"code_swarm": {}}}),
        config_paths_allowlist=frozenset(),
        repo_root=REPO_ROOT,
    )
    claim = _claim(
        predicate="CAPABILITY_AVAILABLE",
        args={"name": "code_swarm", "mode": "read_only"},
    )
    verdict = validator.validate(claim, PREDICATES, ctx)
    assert verdict.status == "SOURCE_CONFLICT"
    assert "code_swarm" in verdict.detail
```

- [ ] **Step 17: Correr los tests y verificar que fallan**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_validator.py -v`
Expected: FAIL — `CAPABILITY_AVAILABLE` no está en `_RESOLVERS`, cae en `RESOLVER_NOT_IMPLEMENTED`

- [ ] **Step 18: Implementar el resolver `CAPABILITY_AVAILABLE`**

En `policy/governance/validator.py`, agregar antes de la línea `_RESOLVERS = {...}`:

```python
def _resolve_capability_available(
    claim: "claims.Claim", ctx: ValidationContext
) -> Verdict:
    name = claim.args["name"]
    mode = claim.args["mode"]
    in_ops = name in ctx.ops
    in_catalog = ctx.catalog.get_capability(name) is not None

    if in_ops and in_catalog:
        return Verdict(
            status="SOURCE_CONFLICT",
            predicate="CAPABILITY_AVAILABLE",
            detail=(
                f"'{name}' presente en ops y en capabilities — dos "
                "fuentes de verdad para el mismo nombre (ver P04, "
                "tensión documentada en el corpus)."
            ),
        )
    if in_ops:
        derived_mode = "mutating" if name in ctx.mutating_capabilities else "read_only"
        if mode != derived_mode:
            return Verdict(
                status="FACT_MISMATCH",
                predicate="CAPABILITY_AVAILABLE",
                detail=(
                    f"'{name}' tiene mode real '{derived_mode}', el "
                    f"claim afirma '{mode}'."
                ),
            )
        return Verdict(
            status="VALID",
            predicate="CAPABILITY_AVAILABLE",
            detail=f"'{name}' verificado en ops, mode='{derived_mode}'.",
        )
    if in_catalog:
        return Verdict(
            status="VALID",
            predicate="CAPABILITY_AVAILABLE",
            detail=(
                f"'{name}' verificado en catálogo de capabilities (mode "
                "no verificable ahí, aceptado sin contradicción)."
            ),
        )
    return Verdict(
        status="FACT_MISMATCH",
        predicate="CAPABILITY_AVAILABLE",
        detail=f"'{name}' no está en ops ni en capabilities.",
    )
```

Y cambiar `_RESOLVERS`:

```python
_RESOLVERS: dict[str, Callable[["claims.Claim", ValidationContext], Verdict]] = {
    "FILE_EXISTS": _resolve_file_exists,
    "CAPABILITY_AVAILABLE": _resolve_capability_available,
}
```

- [ ] **Step 19: Correr los tests y verificar que pasan**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_validator.py -v`
Expected: PASS (13 tests)

- [ ] **Step 20: Commit**

```bash
git add policy/governance/validator.py tests/test_governance_validator.py
git commit -m "policy: resolver CAPABILITY_AVAILABLE, dos fuentes + SOURCE_CONFLICT (governance, Task 4.4 — validator.py completo)"
```

---

### Task 5: `policy/governance/vocab_sweep.py` — barrido léxico

**Files:**
- Create: `policy/governance/vocab_sweep.py`
- Test: `tests/test_governance_vocab_sweep.py`

**Interfaces:**
- Consumes: `frozenset[str]` (típicamente `loaders.ClosedVocabulary.flattened`, Task 3).
- Produces: `sweep(text: str, vocabulary: frozenset[str]) -> list[str]`. Sin consumidor en este sub-proyecto (el llamador que decide claim-vs-rechazo es sub-proyecto 2) — probado standalone con vocabularios sintéticos.

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_governance_vocab_sweep.py
"""
Test de policy/governance/vocab_sweep.py — barrido léxico puro, sin I/O.
"""
from __future__ import annotations

import sys
from pathlib import Path

GOVERNANCE = Path(__file__).resolve().parent.parent / "policy" / "governance"
sys.path.insert(0, str(GOVERNANCE))

import vocab_sweep  # noqa: E402

VOCAB = frozenset({"trae a hyde", "hyde", "code_swarm"})


def test_sweep_no_matches_returns_empty():
    assert vocab_sweep.sweep("un texto sin nada prohibido", VOCAB) == []


def test_sweep_finds_known_term():
    assert vocab_sweep.sweep("invocá a hyde ahora", VOCAB) == ["hyde"]


def test_sweep_finds_multiple_terms_sorted():
    result = vocab_sweep.sweep("code_swarm y también trae a hyde", VOCAB)
    assert result == sorted(result)
    assert "code_swarm" in result
    assert "trae a hyde" in result
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_vocab_sweep.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'vocab_sweep'`

- [ ] **Step 3: Implementar `vocab_sweep.py`**

```python
"""
policy/governance — Barrido léxico contra el vocabulario cerrado
(REFORMAS-v3.md §3.1.5).

Puro: recibe el vocabulario ya cargado (loaders.load_vocabulary().flattened),
nunca abre un archivo. Decidir qué hacer con los términos encontrados
(reformular como claim, rechazar el bloque) es responsabilidad del
llamador — sub-proyecto 2, fuera de alcance acá.
"""
from __future__ import annotations


def sweep(text: str, vocabulary: frozenset[str]) -> list[str]:
    return sorted(term for term in vocabulary if term and term in text)
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_vocab_sweep.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add policy/governance/vocab_sweep.py tests/test_governance_vocab_sweep.py
git commit -m "policy: vocab_sweep.py — barrido léxico puro (governance, Task 5)"
```

---

### Task 6: CHECKPOINT — revisión humana de los templates antes del renderer

**No hay código en este task.** Es una pausa deliberada, no un checkpoint automático de revisión-entre-tasks: los cuatro módulos anteriores (`claims.py`, `loaders.py`, `validator.py`, `vocab_sweep.py`) son internos — nadie los lee directamente, solo los consume código. `renderer.py` es distinto: decide **cómo se lee la salida verificada**, y eso es lo que responde Q14/Q15 (si la salida verificada se lee peor que la prosa libre, la presión de diseño para saltársela aparece ahí, no antes).

- [ ] **Step 1: Presentar a Fernando, antes de escribir `renderer.py`:**
  - Los 3 templates `definida` de `policy/templates/render_templates.yaml`:
    - `CAPABILITY_AVAILABLE`: `"La capability {name} está disponible en modo {mode}."`
    - `AUDIT_EVENT_EXISTS`: `"Existe un evento de auditoría con hash {event_hash}."`
    - `FILE_EXISTS`: `"El archivo {path} existe (hash {hash})."`
  - Un ejemplo renderizado de cada uno con datos de prueba, para juzgar legibilidad real (no solo el template en abstracto).
  - La pregunta explícita: ¿esta forma de leer un claim verificado es aceptable, o hace falta ajustar el fraseo de las plantillas antes de fijarlas con hash (Task 7 las deja versionadas — cambiarlas después implica recalcular `templates_sha256` y recorrer Task 1 de nuevo)?

- [ ] **Step 2: Leer las 3 plantillas en voz alta (Fernando, 2026-08-18).** Si suenan mecánicas al leerlas, el problema es la plantilla, no la restricción — hay margen para redactar mejor sin agregar lógica al renderer. Ese es el momento de decidir Q15 (¿la salida verificada se lee peor que la prosa libre?), con las 3 plantillas reales delante, no en abstracto.

- [ ] **Step 3: Registrar qué queda fuera con solo 2 de 8 predicados resueltos (Fernando, 2026-08-18).** Con `CAPABILITY_AVAILABLE` y `FILE_EXISTS` como únicos resolvers reales, la mayoría del contenido que un modelo querría afirmar como claim no va a ser expresable todavía — los otros 6 predicados (`FACET_EXISTS`, `ENGINE_STATUS`, `CONFIG_VALUE`, `AUDIT_EVENT_EXISTS`, `JOB_STATUS`, `MEMORY_ENTRY_EXISTS`) devuelven `RESOLVER_NOT_IMPLEMENTED`. Este es el primer momento donde ese recorte se ve concreto en vez de solo declarado en el spec — es el dato empírico de Q14. Anotar la impresión de Fernando acá o en la memoria `reformas-v3-progreso` antes de seguir.

- [ ] **Step 4: Esperar aprobación explícita de Fernando antes de avanzar a Task 7.** Si pide cambios de fraseo, editarlos en `policy/templates/render_templates.yaml`, volver a correr `policy/tools/template_hash.py --write` (Task 1, Step 5) y commitear ese ajuste antes de tocar `renderer.py`.

---

### Task 7: `policy/governance/renderer.py` — motor de plantillas

**Files:**
- Create: `policy/governance/renderer.py`
- Test: `tests/test_governance_renderer.py`

**Interfaces:**
- Consumes: `claims.Claim` (Task 2), `loaders.TemplateSpec` (Task 3, vía `loaders.load_templates()`).
- Produces: `render(claim: Claim, templates: dict[str, TemplateSpec]) -> str`. Sin consumidor en este sub-proyecto (sub-proyecto 2 lo conecta a tráfico real) — probado standalone con templates ya cargados y claims sintéticos.

- [ ] **Step 1: Escribir los tests que fallan**

```python
# tests/test_governance_renderer.py
"""
Test de policy/governance/renderer.py — motor de plantillas, puro.
Deliberadamente agnóstico de qué predicados tienen resolver real: prueba
con templates ya cargados y claims sintéticos, sin pasar por validator.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

GOVERNANCE = Path(__file__).resolve().parent.parent / "policy" / "governance"
sys.path.insert(0, str(GOVERNANCE))

import claims  # noqa: E402
import loaders  # noqa: E402
import renderer  # noqa: E402

TEMPLATES = {
    "CAPABILITY_AVAILABLE": loaders.TemplateSpec(
        status="definida", template="La capability {name} está disponible en modo {mode}."
    ),
    "FACET_EXISTS": loaders.TemplateSpec(status="pendiente", template=None),
}


def _claim(**overrides):
    base = dict(
        predicate="CAPABILITY_AVAILABLE",
        args={"name": "code_swarm", "mode": "read_only"},
        authority="OBSERVADO",
        provenance_ref="test",
        evidence_pointer="test",
        scope="jax",
    )
    base.update(overrides)
    return claims.Claim(**base)


def test_render_known_predicate_with_definida_template():
    claim = _claim()
    text = renderer.render(claim, TEMPLATES)
    assert text == "La capability code_swarm está disponible en modo read_only."


def test_render_raises_for_pendiente_template():
    claim = _claim(predicate="FACET_EXISTS", args={"name": "hyde", "engine": "kimi"})
    with pytest.raises(ValueError, match="plantilla"):
        renderer.render(claim, TEMPLATES)


def test_render_raises_for_predicate_without_template_entry():
    claim = _claim(predicate="JOB_STATUS", args={"job_id": "1", "status": "done"})
    with pytest.raises(ValueError, match="plantilla"):
        renderer.render(claim, TEMPLATES)
```

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_renderer.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'renderer'`

- [ ] **Step 3: Implementar `renderer.py`**

```python
"""
policy/governance — Motor de plantillas por predicado (REFORMAS-v3.md
§3.1.6).

Puro: recibe templates ya cargados y verificados por hash (loaders.py),
nunca abre un archivo. No puede emitir texto para un predicado sin
plantilla 'definida' — §3.1.6: "El renderer no puede emitir texto para
un predicado sin plantilla registrada". Deliberadamente agnóstico de qué
predicados tienen resolver real en validator.py — esa desconexión es lo
que permite probarlo con claims sintéticos, sin esperar al resolver.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import claims
    import loaders


def render(claim: "claims.Claim", templates: dict) -> str:
    spec = templates.get(claim.predicate)
    if spec is None or spec.status != "definida" or spec.template is None:
        raise ValueError(
            f"'{claim.predicate}' no tiene plantilla 'definida' — el "
            "renderer no puede emitir texto para un predicado sin "
            "plantilla registrada (§3.1.6)."
        )
    return spec.template.format(**claim.args)
```

- [ ] **Step 4: Correr los tests y verificar que pasan**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_renderer.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add policy/governance/renderer.py tests/test_governance_renderer.py
git commit -m "policy: renderer.py — motor de plantillas puro (governance, Task 7)"
```

---

### Task 8: Suite completa y cierre del sub-proyecto 1

**Files:** ninguno nuevo — solo verificación.

- [ ] **Step 1: Correr la suite completa de governance**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_template_hash.py tests/test_governance_claims.py tests/test_governance_loaders.py tests/test_governance_validator.py tests/test_governance_vocab_sweep.py tests/test_governance_renderer.py -v`
Expected: PASS — 28 tests, 0 failures, 0 errors (1 en test_template_hash.py + 4 en test_governance_claims.py + 4 en test_governance_loaders.py + 13 en test_governance_validator.py + 3 en test_governance_vocab_sweep.py + 3 en test_governance_renderer.py)

- [ ] **Step 2: Confirmar que `policy/VERSION` tiene las tres líneas esperadas**

Run: `cat policy/VERSION`
Expected: `version:`, `sha256:` (sin cambios, corpus de rules/), `templates_sha256:` (nueva, de Task 1)

- [ ] **Step 3: Actualizar la memoria de progreso**

Editar `reformas-v3-progreso` (memoria) marcando Sub-proyecto 1 de Fase 2 como implementado y testeado, con el link a este plan y la lista de commits reales (`git log --oneline` desde el commit de Task 1 hasta el de Task 7).

- [ ] **Step 4: Próximo paso — no forma parte de este plan**

Sub-proyecto 2 (separación de canales `claim`/`analysis`/`judgment` en el contrato de salida de las facetas, manejo de salida malformada de Kimi, integración real en shadow validation sobre la Mesa web) necesita su propio spec — repetir `superpowers:brainstorming` + spec antes de plan, no continuar directo desde acá.
