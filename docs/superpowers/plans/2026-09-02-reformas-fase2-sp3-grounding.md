# REFORMAS Fase 2 SP3 — Grounding por snapshot inyectado — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que un claim de `CAPABILITY_AVAILABLE` emitido por la Mesa web pueda acreditar `authority=OBSERVADO` citando una línea de un snapshot que el servidor inyectó en el prompt, y llegar al resolver — en vez de morir en `AUTHORITY_INVALID` como hoy el 100%.

**Architecture:** Módulo puro nuevo `policy/governance/grounding.py` en `jax` (construir/renderizar/hashear el snapshot y acreditar un claim contra él); `validator.validate()` gana un orden de chequeos normativo con dos estados nuevos. En `jax-platform`, `chat.py` construye el snapshot por turno, lo anexa al system prompt sin el hash, y lo pasa como quinto argumento obligatorio a `run_shadow_validation`, que acredita antes de validar y persiste snapshot, sha256, authority y evidence_pointer en columnas nuevas.

**Tech Stack:** Python 3.12, pydantic, PyYAML, pytest; aiomysql + MariaDB 11.8 (service container en CI); FastAPI (`BackgroundTasks`).

**Spec:** `docs/superpowers/specs/2026-09-02-reformas-fase2-sp3-grounding-design.md` (repo `jax`). El plan argumenta desde el spec; leerlo primero.

## Global Constraints

- **Dos repos, dos ramas, un PR por repo.** `jax`: rama `reformas-fase2-sp3-grounding` (ya existe, tiene el spec). `jax-platform`: rama `reformas-fase2-sp3-grounding` a crear desde `origin/master` (`36b7728` o posterior — hacer `git fetch` antes). Verificar con `git -C <ruta> rev-parse --abbrev-ref HEAD` en el MISMO comando que cualquier `git` — un `cd` fallido deja el shell en otro checkout.
- **Orden de merge:** `jax` primero (el `grounding.py` lo importa `jax-platform` por `sys.path` desde `JAX_REPO_PATH`); el job de CI de `jax-platform` clona `master` de `jax`, así que hasta que `jax` esté mergeado el CI de `jax-platform` NO puede pasar. Aceptado: los PRs se abren en orden.
- **Python para tests de `jax`:** `/home/fruiz/jax/.venv/bin/python` (tiene PyYAML). NO `las_manos/.venv`.
- **Python para tests de `jax-platform`:** `cd /home/fruiz/jax-platform/backend && python -m pytest ...`. Los tests con DB corren contra `jax_memory_test` (`conftest.py` fija `JAX_DB_NAME`). Para el modo sin DB: `JAX_CI_NO_DB=1`.
- **P10 (NORMATIVA):** ningún `except` con cuerpo `pass` sin `# fail-soft: <razón>` en la misma línea; el scanner `no-fail-open-except` corre en los dos repos. Ningún gate falla abierto.
- **Serialización canónica (spec §5.2):** `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`, lista `capabilities` ordenada por `name`.
- **Normalización de args (spec §5.3):** `str(v).strip()` por valor; UNA función, llamada por los dos lados.
- **Pisos de CI se MIDEN antes/después con el comando del job, no se estiman.** Cada cambio de piso lleva comentario con la medición.
- **Commits:** mensaje en español, estilo `tipo(alcance): frase -- detalle`, con el trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` y `Claude-Session: https://claude.ai/code/session_011AVPMngV4W1np18dUPa542`.
- **No tocar:** el sufijo de contrato para restringir predicados (los 8 siguen), `FACET_EXISTS`, tool-calling, frontend.

---

## File Structure

### `jax`

| archivo | responsabilidad |
|---|---|
| `policy/governance/grounding.py` *(nuevo, puro)* | `normalize_args`, `build_snapshot`, `render`, `snapshot_sha256` (vía `Snapshot.sha256`), `accredit`. Tipos: `SnapshotEntry`, `Snapshot`, `SnapshotError`, `Accreditation`, `GroundingBuildError`. Cero I/O. |
| `policy/governance/validator.py` | `Verdict.status` gana `PROVENANCE_MISMATCH` y `GROUNDING_UNAVAILABLE`; `validate()` gana `accreditation: Accreditation | None = None` y el orden 0–6 del spec §4.1. |
| `tests/test_governance_grounding.py` *(nuevo)* | Pruebas 1–8 + 9.1b del spec, todas puras. |
| `tests/test_governance_validator.py` | Arreglo del test rojo preexistente (catálogo TOML vacío desde Bloque 3). |
| `.github/workflows/policy.yml` | Job nuevo `governance` que corre `tests/test_governance_*.py` con piso exacto. |
| `DEUDA.md` | Ítem nuevo: la rama `in_catalog` del resolver lee un catálogo que el Bloque 3 vació. |

### `jax-platform`

| archivo | responsabilidad |
|---|---|
| `backend/governance_context.py` *(nuevo)* | `validation_context()` con `lru_cache(maxsize=1)` — movido de `shadow_validation.py` para que `chat.py` y `shadow_validation.py` compartan el MISMO contexto sin import circular. |
| `backend/api/chat.py` | Parser conserva `evidence_pointer` (y `authority` si el modelo lo manda); sufijo de contrato admite `evidence_pointer`; `_build_grounding()`; `grounding` viaja por `_invoke_facet` → `_invoke_facet_dispatch` (anexa `render()`) y al background task. |
| `backend/shadow_validation.py` | Quinto argumento obligatorio `grounding`; acreditar antes de validar; persistir snapshot/sha256/authority/evidence_pointer; truncado a 100 con original en `detail`. |
| `backend/db/migrations.py` | 4 columnas nuevas en `_COLUMNS` + en los `CREATE TABLE` (instalación nueva). |
| `backend/tests/test_chat_contract_prompt.py` | Regresión de la línea 576 + parser conserva `evidence_pointer`. |
| `backend/tests/test_shadow_validation_grounding.py` *(nuevo)* | §9.2: cada estado → su fila; `NULL`/`ERROR`/hash; truncado; toda fila nueva con sha256 no-NULL. |
| `backend/tests/test_chat_grounding_wiring.py` *(nuevo)* | §9.3: un objeto, dos consumidores. |
| `.github/workflows/policy.yml` | Pisos `PISO_PASSED` (con DB) y `JAX_CI_MIN_PASSED` (sin DB), medidos. |

---

## Task 0: Gobernanza en CI de `jax`, y el test rojo que nadie veía

**Por qué es tarea 0 y no un extra:** los tests de `grounding.py` (Tasks 1–3) van a `tests/test_governance_grounding.py`. Ningún job de `.github/workflows/policy.yml` de `jax` corre `tests/test_governance_*.py` hoy (medido 2026-09-02: `grep pytest .github/workflows/*.yml` no los menciona). Sin este job, todo lo que este plan escribe en `jax` quedaría verde por ausencia. Y al correrlos localmente hay **1 rojo en `master`**: `test_capability_available_found_only_in_catalog_mode_unverified_but_accepted` da `FACT_MISMATCH` en vez de `VALID`, porque `MotorCatalog(config)` construido desde `las_manos/config.toml` está **vacío** desde que el Bloque 3 movió las capabilities a la DB (`[capabilities.*]` del TOML tiene 0 entradas; `MotorCatalog.from_db()` es el camino real). La rama `in_catalog` de `_resolve_capability_available` es código muerto en producción. Arreglar el validador para leer la DB está FUERA de este plan; lo que se hace acá es dejar de tener un test que bendice un estado que no existe, registrar la deuda, y poner los tests en CI.

**Files:**
- Modify: `tests/test_governance_validator.py:254-261`
- Modify: `.github/workflows/policy.yml` (job nuevo al final)
- Modify: `DEUDA.md` (sección "Bloquea trabajo", ítem nuevo)

**Interfaces:**
- Produces: job `governance` en CI con piso exacto; `tests/test_governance_*.py` corriendo en CI.

- [ ] **Step 1: Ver el rojo con tus propios ojos**

Run: `cd /home/fruiz/jax && git rev-parse --abbrev-ref HEAD && .venv/bin/python -m pytest tests/test_governance_validator.py -q 2>&1 | tail -3`
Expected: rama `reformas-fase2-sp3-grounding`; `1 failed, N passed`; el failed es `test_capability_available_found_only_in_catalog_mode_unverified_but_accepted` con `assert 'FACT_MISMATCH' == 'VALID'`.

- [ ] **Step 2: Reemplazar el test rojo por dos que digan la verdad**

Reemplazar la función `test_capability_available_found_only_in_catalog_mode_unverified_but_accepted` (líneas 254–261) por:

```python
def test_capability_available_catalog_branch_with_synthetic_catalog_is_valid():
    """La rama `in_catalog` del resolver, ejercitada con un catálogo
    ARMADO A MANO. No usa _real_ctx(): ver el test siguiente."""
    ctx = validator.ValidationContext(
        ops=frozenset(),
        mutating_capabilities=frozenset(),
        catalog=MotorCatalog({"capabilities": {"code_swarm": {"allowed_motors": ["kimi"]}}}),
        config_paths_allowlist=frozenset(),
        repo_root=REPO_ROOT,
    )
    claim = _claim(
        predicate="CAPABILITY_AVAILABLE",
        args={"name": "code_swarm", "mode": "read_only"},
    )
    verdict = validator.validate(claim, PREDICATES, ctx)
    assert verdict.status == "VALID"


def test_real_toml_catalog_is_empty_since_block3_so_catalog_branch_is_dead_in_production():
    """Hasta el 2026-09-02 este archivo tenía un test que afirmaba que
    `code_swarm` se encontraba en el catálogo REAL (_real_ctx) y daba VALID.
    Estaba rojo en master y nadie lo veía porque tests/test_governance_*.py
    no corría en CI. La causa: el Bloque 3 (2026-08-2x) movió las
    capabilities a la DB (`MotorCatalog.from_db()`); `[capabilities.*]` del
    TOML quedó vacío, y `load_validation_context()` sigue construyendo
    `MotorCatalog(config)` desde el TOML. Resultado: en producción la rama
    `in_catalog` del resolver NUNCA se toma. Este test fija ese hecho para
    que deje de ser invisible; la deuda está en DEUDA.md."""
    ctx = _real_ctx()
    assert ctx.catalog.get_capability("code_swarm") is None
    claim = _claim(
        predicate="CAPABILITY_AVAILABLE",
        args={"name": "code_swarm", "mode": "read_only"},
    )
    verdict = validator.validate(claim, PREDICATES, ctx)
    assert verdict.status == "FACT_MISMATCH"
```

- [ ] **Step 3: Correr la suite de gobernanza entera y anotar el número**

Run: `cd /home/fruiz/jax && .venv/bin/python -m pytest tests/test_governance_claims.py tests/test_governance_loaders.py tests/test_governance_validator.py tests/test_governance_vocab_sweep.py tests/test_governance_renderer.py -q 2>&1 | tail -1`
Expected: `39 passed` (38 anteriores − 1 reemplazado + 2 nuevos). **Anotar el número real**: es el piso del job.

- [ ] **Step 4: Job `governance` en CI**

Agregar al final de `.github/workflows/policy.yml` de `jax` (misma indentación que `ollama-num-parallel`):

```yaml
  governance:
    # policy/governance/ (SP1 de REFORMAS Fase 2, 2026-08-18) tenía 38 tests
    # que NO corrían en ningún job hasta el 2026-09-02. Uno estaba rojo en
    # master (catálogo TOML vacío desde el Bloque 3) y nadie lo veía. Este
    # job existe para que eso no vuelva a pasar; SP3 agrega
    # tests/test_governance_grounding.py acá mismo.
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install pytest pyyaml pydantic
      - name: Suite de gobernanza
        run: python -m pytest tests/test_governance_claims.py tests/test_governance_loaders.py tests/test_governance_validator.py tests/test_governance_vocab_sweep.py tests/test_governance_renderer.py -v
      - name: Piso exacto de tests CORRIDOS
        # Exacto y no "al menos": un piso con holgura deja perder tests en
        # silencio. Medido 2026-09-02 con .venv local antes de este job: 39.
        run: |
          python -m pytest tests/test_governance_claims.py tests/test_governance_loaders.py tests/test_governance_validator.py tests/test_governance_vocab_sweep.py tests/test_governance_renderer.py -q 2>&1 | tee /tmp/gov
          grep -qE "^39 passed" /tmp/gov || {
            echo "PISO ROTO: se esperaban 39 tests CORRIDOS."; exit 1; }
```

Si el número del Step 3 no fue 39, usar el medido en las dos ocurrencias.

- [ ] **Step 5: Verificar que el job corre en un entorno limpio (sin `.venv`, sin `/etc/jax`)**

Run: `cd /home/fruiz/jax && python3 -m venv /tmp/claude-1000/-home-fruiz/5af0a683-1b2f-4ae0-8a06-226199ac76b7/scratchpad/venv-gov && /tmp/claude-1000/-home-fruiz/5af0a683-1b2f-4ae0-8a06-226199ac76b7/scratchpad/venv-gov/bin/pip install -q pytest pyyaml pydantic && /tmp/claude-1000/-home-fruiz/5af0a683-1b2f-4ae0-8a06-226199ac76b7/scratchpad/venv-gov/bin/python -m pytest tests/test_governance_claims.py tests/test_governance_loaders.py tests/test_governance_validator.py tests/test_governance_vocab_sweep.py tests/test_governance_renderer.py -q 2>&1 | tail -1`
Expected: el mismo número que el Step 3. Si difiere, el job va a fallar en CI — averiguar qué dependencia falta ANTES de commitear (lección: reproducir el runner, no el local).

- [ ] **Step 6: Registrar la deuda**

En `DEUDA.md`, sección `## Bloquea trabajo`, agregar después del bloque ESTADO:

```markdown
- **El resolver de `CAPABILITY_AVAILABLE` consulta un catálogo que el Bloque 3
  vació — encontrado 2026-09-02.** `validator.load_validation_context()`
  construye `MotorCatalog(config)` desde `las_manos/config.toml`, pero desde el
  Bloque 3 las capabilities viven en la DB (`MotorCatalog.from_db()`) y
  `[capabilities.*]` del TOML tiene 0 entradas. La rama `in_catalog` de
  `_resolve_capability_available` es código muerto en producción: un claim
  sobre una capability de la DB da `FACT_MISMATCH` aunque exista. Lo tapaba un
  test que afirmaba lo contrario y que estaba rojo en `master` sin que nadie
  lo viera, porque `tests/test_governance_*.py` no corría en CI (arreglado el
  mismo día: job `governance`). **Qué falta:** que el validador lea el catálogo
  de la DB por el mismo camino que `jacobs`, o que el snapshot de SP3 y el
  resolver declaren explícitamente que solo cubren `ops`. Verificado contra el
  árbol el 2026-09-02, con `tests/test_governance_validator.py::test_real_toml_catalog_is_empty_since_block3_so_catalog_branch_is_dead_in_production`
  como testigo.
```

- [ ] **Step 7: Commit**

```bash
cd /home/fruiz/jax && git add tests/test_governance_validator.py .github/workflows/policy.yml DEUDA.md && git commit -F - <<'EOF'
ci(governance): los 38 tests de policy/governance entran a CI -- uno estaba rojo en master y nadie lo veia

El catalogo que construye load_validation_context() desde el TOML esta vacio
desde el Bloque 3 (capabilities en DB). El test que afirmaba lo contrario se
reemplaza por uno que ejercita la rama con catalogo sintetico y otro que fija
el hecho medido. Deuda registrada en DEUDA.md.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011AVPMngV4W1np18dUPa542
EOF
```

---

## Task 1: `grounding.py` — snapshot: normalizar, construir, hashear, renderizar

**Files:**
- Create: `policy/governance/grounding.py`
- Create: `tests/test_governance_grounding.py`

**Interfaces:**
- Consumes: `validator.ValidationContext` (campos `ops: frozenset[str]`, `mutating_capabilities: frozenset[str]`).
- Produces:
  - `normalize_args(args: Mapping[str, object]) -> dict[str, str]`
  - `class SnapshotEntry(pointer: str, predicate: str, args: dict[str, str])` (frozen dataclass)
  - `class Snapshot(entries: tuple[SnapshotEntry, ...], canonical_json: str, sha256: str)` (frozen) con método `lookup(pointer: str) -> SnapshotEntry | None`
  - `class SnapshotError(reason: str)` (frozen)
  - `class GroundingBuildError(RuntimeError)`
  - `build_snapshot(ctx) -> Snapshot` — lanza `GroundingBuildError`
  - `render(snapshot: Snapshot) -> str` — el bloque para el prompt, SIN el hash
  - `SECTION_PREDICATE: dict[str, str] = {"capabilities": "CAPABILITY_AVAILABLE"}`

- [ ] **Step 1: Tests de las propiedades del snapshot (7a–7d del spec, más normalización)**

Crear `tests/test_governance_grounding.py`:

```python
"""
Tests de policy/governance/grounding.py — REFORMAS Fase 2 SP3.

Todo acá es PURO: sin I/O, sin DB, sin red. Los ValidationContext se arman a
mano. Numeración de pruebas = spec §9.1 / §9.1b.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GOVERNANCE = REPO_ROOT / "policy" / "governance"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GOVERNANCE))

import grounding  # noqa: E402
import validator  # noqa: E402
from las_manos.motor_registry.catalog import MotorCatalog  # noqa: E402


def _ctx(ops: set[str], mutating: set[str] = frozenset({"write_file"})) -> validator.ValidationContext:
    return validator.ValidationContext(
        ops=frozenset(ops),
        mutating_capabilities=frozenset(mutating),
        catalog=MotorCatalog({}),
        config_paths_allowlist=frozenset(),
        repo_root=REPO_ROOT,
    )


# Contexto A del spec: write_file es mutante. Con orden por name,
# 'read_file' < 'write_file', así que write_file queda en /capabilities/1.
CTX_A = _ctx({"write_file", "read_file"})


# --- normalize_args ---------------------------------------------------------

def test_normalize_args_str_and_strip_every_value():
    assert grounding.normalize_args({"name": " write_file ", "mode": 1}) == {"name": "write_file", "mode": "1"}


def test_normalize_args_keeps_keys_untouched():
    # Las claves NO se normalizan: ARGS_MISMATCH (paso 2) es quien juzga claves.
    assert grounding.normalize_args({" Name": "x"}) == {" Name": "x"}


# --- build_snapshot: 7a determinismo -----------------------------------------

def test_7a_same_ctx_same_sha256():
    assert grounding.build_snapshot(CTX_A).sha256 == grounding.build_snapshot(CTX_A).sha256


def test_7a_different_ctx_different_sha256():
    ctx_b = _ctx({"read_file"})
    assert grounding.build_snapshot(CTX_A).sha256 != grounding.build_snapshot(ctx_b).sha256


def test_sha256_is_over_the_canonical_json():
    import hashlib
    snap = grounding.build_snapshot(CTX_A)
    assert snap.sha256 == hashlib.sha256(snap.canonical_json.encode("utf-8")).hexdigest()
    # canónico: sort_keys + separators compactos
    assert snap.canonical_json == json.dumps(
        json.loads(snap.canonical_json), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


# --- 7c: orden por name, no por orden de llegada -----------------------------

def test_7c_entries_ordered_by_name_regardless_of_input_order():
    a = grounding.build_snapshot(_ctx({"write_file", "read_file", "list_dir"}))
    b = grounding.build_snapshot(_ctx({"list_dir", "write_file", "read_file"}))
    assert a.sha256 == b.sha256
    assert [e.pointer for e in a.entries] == ["/capabilities/0", "/capabilities/1", "/capabilities/2"]
    assert [e.args["name"] for e in a.entries] == ["list_dir", "read_file", "write_file"]


def test_snapshot_entry_carries_predicate_and_normalized_args():
    snap = grounding.build_snapshot(CTX_A)
    e = snap.lookup("/capabilities/1")
    assert e is not None
    assert e.predicate == "CAPABILITY_AVAILABLE"
    assert e.args == {"name": "write_file", "mode": "mutating"}
    assert snap.lookup("/capabilities/0").args == {"name": "read_file", "mode": "read_only"}


def test_lookup_returns_none_for_unknown_pointer():
    snap = grounding.build_snapshot(CTX_A)
    assert snap.lookup("/capabilities/99") is None
    assert snap.lookup("/facets/0") is None


# --- 7b: render sin hash ------------------------------------------------------

def test_7b_render_does_not_contain_the_hash_nor_any_prefix_of_it():
    snap = grounding.build_snapshot(CTX_A)
    text = grounding.render(snap)
    assert snap.sha256 not in text
    assert snap.sha256[:12] not in text
    assert "sha256" not in text


def test_render_lists_every_entry_with_its_pointer_and_args():
    snap = grounding.build_snapshot(CTX_A)
    text = grounding.render(snap)
    assert "/capabilities/0: name=read_file, mode=read_only" in text
    assert "/capabilities/1: name=write_file, mode=mutating" in text
    assert "evidence_pointer" in text  # la instrucción de cómo citar


# --- 7d: fallo ruidoso (P10) ---------------------------------------------------

def test_7d_build_snapshot_raises_on_broken_ctx_never_returns_empty():
    class Broken:
        @property
        def ops(self):
            raise OSError("config ilegible")
        mutating_capabilities = frozenset()
    with pytest.raises(grounding.GroundingBuildError):
        grounding.build_snapshot(Broken())


def test_empty_ops_is_a_valid_snapshot_not_an_error():
    # Cero capabilities es una OBSERVACIÓN válida (spec: "predicado con
    # resolver y cero entradas = observación válida, no UNGROUNDED").
    snap = grounding.build_snapshot(_ctx(set()))
    assert snap.entries == ()
    assert len(snap.sha256) == 64
```

- [ ] **Step 2: Correr para verificar que falla por módulo inexistente**

Run: `cd /home/fruiz/jax && .venv/bin/python -m pytest tests/test_governance_grounding.py -q 2>&1 | tail -2`
Expected: `ModuleNotFoundError: No module named 'grounding'`.

- [ ] **Step 3: Implementar `grounding.py` (solo lo que estos tests piden)**

Crear `policy/governance/grounding.py`:

```python
"""
policy/governance — Grounding por snapshot inyectado (REFORMAS Fase 2 SP3).

Spec: docs/superpowers/specs/2026-09-02-reformas-fase2-sp3-grounding-design.md

Qué hace: construye, desde el MISMO ValidationContext que usa validator.py,
el conjunto de hechos que el servidor inyecta en el prompt de la Mesa web
(build_snapshot / render), y acredita un claim contra ese snapshot
(accredit) derivando authority y provenance_ref del lado del servidor.

Invariante (spec §3): todo hecho inyectado tiene quién lo re-resuelva. Por
eso el snapshot se genera desde ctx.ops (lo que resuelve
_resolve_capability_available) y no desde una lista curada.

Este módulo es PURO: sin I/O, sin red, testeable en aislamiento. La única
fuente de datos es el ValidationContext que recibe. Si construir el
snapshot falla, LANZA (P10): nunca devuelve vacío por error.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Mapping

# Sección del snapshot -> predicado que acredita. Crece SOLO cuando un
# predicado gana resolver (spec §3): no agregar entradas acá sin resolver
# en validator._RESOLVERS.
SECTION_PREDICATE: dict[str, str] = {"capabilities": "CAPABILITY_AVAILABLE"}

# Formato de evidence_pointer que el modelo puede citar. Estricto a
# propósito: "/capabilities/-1", "capabilities/10", "/capabilities/abc" y
# "" NO matchean -> PROVENANCE_MISMATCH, nunca indexación negativa ni
# excepción (spec §9.1b).
_POINTER_RE = re.compile(r"^/([a-z_]+)/(0|[1-9][0-9]*)$")


class GroundingBuildError(RuntimeError):
    """El snapshot no pudo construirse. Se lanza, no se degrada (P10)."""


@dataclass(frozen=True)
class SnapshotEntry:
    pointer: str
    predicate: str
    args: dict[str, str]


@dataclass(frozen=True)
class Snapshot:
    entries: tuple[SnapshotEntry, ...]
    canonical_json: str
    sha256: str

    def lookup(self, pointer: str) -> SnapshotEntry | None:
        for e in self.entries:
            if e.pointer == pointer:
                return e
        return None


@dataclass(frozen=True)
class SnapshotError:
    """Marca de 'el snapshot falló al construirse' que viaja al validador
    en lugar de un Snapshot. Persistida como sha256='ERROR' (spec §5.4)."""
    reason: str


def normalize_args(args: Mapping[str, object]) -> dict[str, str]:
    """UNA sola normalización para los dos lados (spec §5.3): la usa
    build_snapshot al producir cada entrada y accredit al comparar. Solo
    valores: las claves las juzga ARGS_MISMATCH en validator.validate()."""
    return {k: str(v).strip() for k, v in args.items()}


def _canonical(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_snapshot(ctx) -> Snapshot:
    """Snapshot desde ctx.ops + ctx.mutating_capabilities. Orden por name
    (spec §5.2): mismo contenido => mismo hash y mismos punteros, en
    cualquier orden de archivo."""
    try:
        ops = sorted(ctx.ops)
        mutating = ctx.mutating_capabilities
    except Exception as e:
        raise GroundingBuildError(f"ValidationContext inutilizable: {type(e).__name__}: {e}") from e

    caps = [
        normalize_args({"name": name, "mode": "mutating" if name in mutating else "read_only"})
        for name in ops
    ]
    data = {"capabilities": caps}
    canonical = _canonical(data)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    entries = tuple(
        SnapshotEntry(pointer=f"/capabilities/{i}", predicate=SECTION_PREDICATE["capabilities"], args=c)
        for i, c in enumerate(caps)
    )
    return Snapshot(entries=entries, canonical_json=canonical, sha256=digest)


def render(snapshot: Snapshot) -> str:
    """Bloque para el system prompt. NO incluye el hash (spec §5.1): el
    modelo solo cita la línea; provenance_ref lo escribe el servidor."""
    lines = [
        "HECHOS VERIFICADOS — leídos del sistema por el servidor. "
        "Para afirmar uno, poné su evidence_pointer en el claim.",
        "  capabilities:",
    ]
    for e in snapshot.entries:
        lines.append(f"    {e.pointer}: " + ", ".join(f"{k}={v}" for k, v in e.args.items()))
    return "\n".join(lines)
```

(`accredit` y `Accreditation` se agregan en Task 2.)

- [ ] **Step 4: Correr los tests**

Run: `cd /home/fruiz/jax && .venv/bin/python -m pytest tests/test_governance_grounding.py -q 2>&1 | tail -1`
Expected: `14 passed`.

- [ ] **Step 5: Scanner P10 sobre el archivo nuevo**

Run: `cd /home/fruiz/jax && .venv/bin/python -m pytest policy/tests/test_no_fail_open_except.py -q 2>&1 | tail -1`
Expected: `passed` (no hay `except: pass` en grounding.py; el `except Exception` de `build_snapshot` re-lanza).

- [ ] **Step 6: Commit**

```bash
cd /home/fruiz/jax && git add policy/governance/grounding.py tests/test_governance_grounding.py && git commit -F - <<'EOF'
feat(governance): grounding.py -- snapshot canonico desde el ValidationContext, sin el hash en el render

Orden por name (mismo contenido => mismo hash y mismos punteros), una sola
normalizacion de args para los dos lados, fallo ruidoso al construir (P10).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011AVPMngV4W1np18dUPa542
EOF
```

---

## Task 2: `grounding.accredit()` — la citación se verifica, no se cree

**Files:**
- Modify: `policy/governance/grounding.py` (agregar `Accreditation` y `accredit`)
- Modify: `tests/test_governance_grounding.py` (agregar tests)

**Interfaces:**
- Produces:
  - `class Accreditation(authority: Literal["OBSERVADO","INFERIDO"], provenance_ref: str, evidence_pointer_raw: object | None, outcome: Literal["ACCREDITED","NO_POINTER","MISMATCH","UNAVAILABLE"], detail: str)` (frozen)
  - `accredit(raw_claim: Mapping[str, object], grounding: Snapshot | SnapshotError) -> Accreditation`
- Contrato con Task 3: `validate()` usa `outcome` para los pasos 0, 4 y 5, y `authority` para armar el `Claim`. **`accredit` NO sabe de resolvers**: el paso 3 (`RESOLVER_NOT_IMPLEMENTED`) lo pone `validate()` ANTES de mirar `outcome == "MISMATCH"`.

- [ ] **Step 1: Tests de acreditación (pruebas 2, 3, 4, 5, 6 y 9.1b del spec, en su capa pura)**

Agregar al final de `tests/test_governance_grounding.py`:

```python
# --- accredit: la citación se verifica, no se cree (spec §2.2, §4.1) --------

SNAP_A = grounding.build_snapshot(CTX_A)   # write_file en /capabilities/1


def _raw(predicate="CAPABILITY_AVAILABLE", args=None, pointer="__absent__"):
    claim = {"predicate": predicate, "args": args or {"name": "write_file", "mode": "mutating"}}
    if pointer != "__absent__":
        claim["evidence_pointer"] = pointer
    return claim


def test_5_valid_pointer_exact_args_is_observado_with_server_written_provenance():
    acc = grounding.accredit(_raw(pointer="/capabilities/1"), SNAP_A)
    assert acc.outcome == "ACCREDITED"
    assert acc.authority == "OBSERVADO"
    assert acc.provenance_ref == f"tool_result:sha256:{SNAP_A.sha256}"
    assert acc.evidence_pointer_raw == "/capabilities/1"


def test_3_no_pointer_is_inferido_no_pointer():
    acc = grounding.accredit(_raw(), SNAP_A)
    assert acc.outcome == "NO_POINTER"
    assert acc.authority == "INFERIDO"
    assert acc.evidence_pointer_raw is None


def test_2_pointer_out_of_range_is_mismatch():
    acc = grounding.accredit(_raw(pointer="/capabilities/99"), SNAP_A)
    assert acc.outcome == "MISMATCH"
    assert acc.authority == "INFERIDO"
    assert "99" in acc.detail


def test_4_valid_pointer_but_args_differ_is_mismatch_the_forged_citation():
    # El test que sostiene el diseño (spec §2.2): puntero real, args falsos.
    acc = grounding.accredit(
        _raw(args={"name": "write_file", "mode": "read_only"}, pointer="/capabilities/1"), SNAP_A
    )
    assert acc.outcome == "MISMATCH"
    assert acc.authority == "INFERIDO"
    assert "read_only" in acc.detail and "mutating" in acc.detail


def test_pointer_to_entry_of_another_predicate_is_mismatch():
    # JOB_STATUS citando una línea de capabilities. En validate() esto nunca
    # llega acá (paso 3 antes del 5) -- pero accredit debe ser correcto solo.
    acc = grounding.accredit(_raw(predicate="JOB_STATUS", args={"job_id": "1", "status": "ok"},
                                  pointer="/capabilities/1"), SNAP_A)
    assert acc.outcome == "MISMATCH"


def test_args_are_compared_with_the_same_normalization_that_built_the_snapshot():
    # Espacios y tipos no producen un PROVENANCE_MISMATCH por formato (spec §5.3).
    acc = grounding.accredit(
        _raw(args={"name": " write_file ", "mode": "mutating"}, pointer="/capabilities/1"), SNAP_A
    )
    assert acc.outcome == "ACCREDITED"


def test_6_snapshot_error_is_unavailable_even_with_a_pointer():
    acc = grounding.accredit(_raw(pointer="/capabilities/1"), grounding.SnapshotError("config ilegible"))
    assert acc.outcome == "UNAVAILABLE"
    assert acc.authority == "INFERIDO"
    assert "config ilegible" in acc.detail


@pytest.mark.parametrize("bad", ["", "capabilities/1", "/capabilities/abc", "/capabilities/-1", "/x" * 150])
def test_9_1b_malformed_pointer_is_mismatch_without_exception(bad):
    acc = grounding.accredit(_raw(pointer=bad), SNAP_A)
    assert acc.outcome == "MISMATCH"
    assert acc.authority == "INFERIDO"
    assert acc.evidence_pointer_raw == bad


def test_9_1b_minus_one_never_indexes_from_the_end():
    # Si "-1" se convirtiera a int y se indexara, apuntaría a la ÚLTIMA
    # entrada (write_file) y los args coincidirían -> ACCREDITED. Eso sería
    # fail-open por Python. Debe ser MISMATCH.
    acc = grounding.accredit(_raw(pointer="/capabilities/-1"), SNAP_A)
    assert acc.outcome == "MISMATCH"


def test_non_string_pointer_is_mismatch_not_exception():
    for bad in (7, None, ["/capabilities/1"], {"p": 1}):
        acc = grounding.accredit(_raw(pointer=bad), SNAP_A)
        assert acc.outcome == ("NO_POINTER" if bad is None else "MISMATCH")
```

Nota: `pointer=None` explícito cuenta como ausente (`NO_POINTER`) — un modelo que manda `"evidence_pointer": null` no citó. Está fijado en el último test.

- [ ] **Step 2: Correr para ver el fallo**

Run: `cd /home/fruiz/jax && .venv/bin/python -m pytest tests/test_governance_grounding.py -q 2>&1 | tail -2`
Expected: `AttributeError: module 'grounding' has no attribute 'accredit'` (los 14 anteriores pasan).

- [ ] **Step 3: Implementar `Accreditation` y `accredit`**

Agregar a `policy/governance/grounding.py`, después de `SnapshotError`:

```python
from typing import Literal  # mover arriba con los otros imports


@dataclass(frozen=True)
class Accreditation:
    """Resultado de acreditar un claim contra el snapshot. La AUTORIDAD la
    deriva el servidor acá (spec §2.1, P08): el modelo solo señaló una línea.

    outcome:
      ACCREDITED  -> puntero resuelve y args coinciden: authority=OBSERVADO
      NO_POINTER  -> el claim no trae evidence_pointer: authority=INFERIDO
      MISMATCH    -> puntero malformado / fuera de rango / de otro predicado /
                     args que no coinciden: authority=INFERIDO
      UNAVAILABLE -> el snapshot del turno no existe (SnapshotError)
    Qué veredicto sale de cada uno lo decide validator.validate() en el orden
    normativo del spec §4.1 -- acá no se conoce si el predicado tiene resolver.
    """
    authority: Literal["OBSERVADO", "INFERIDO"]
    provenance_ref: str
    evidence_pointer_raw: object | None
    outcome: Literal["ACCREDITED", "NO_POINTER", "MISMATCH", "UNAVAILABLE"]
    detail: str
```

Y al final del archivo:

```python
def accredit(raw_claim: Mapping[str, object], grounding: Snapshot | SnapshotError) -> Accreditation:
    """Nunca lanza por contenido del claim: un puntero que mata la
    background task es fail-open (spec §9.1b). Todo lo raro es MISMATCH."""
    raw_ptr = raw_claim.get("evidence_pointer")

    if isinstance(grounding, SnapshotError):
        return Accreditation(
            authority="INFERIDO", provenance_ref="none", evidence_pointer_raw=raw_ptr,
            outcome="UNAVAILABLE",
            detail=f"el snapshot de este turno no se construyó: {grounding.reason}",
        )

    if raw_ptr is None:
        return Accreditation(
            authority="INFERIDO", provenance_ref="none", evidence_pointer_raw=None,
            outcome="NO_POINTER", detail="el claim no trae evidence_pointer.",
        )

    def mismatch(why: str) -> Accreditation:
        return Accreditation(
            authority="INFERIDO", provenance_ref="none", evidence_pointer_raw=raw_ptr,
            outcome="MISMATCH", detail=why,
        )

    if not isinstance(raw_ptr, str):
        return mismatch(f"evidence_pointer no es string (es {type(raw_ptr).__name__}).")
    m = _POINTER_RE.match(raw_ptr)
    if m is None:
        shown = raw_ptr if len(raw_ptr) <= 120 else raw_ptr[:120] + "…"
        return mismatch(f"evidence_pointer malformado: {shown!r}.")
    entry = grounding.lookup(raw_ptr)
    if entry is None:
        return mismatch(f"evidence_pointer {raw_ptr!r} no existe en el snapshot del turno.")
    if entry.predicate != raw_claim.get("predicate"):
        return mismatch(
            f"{raw_ptr} es una entrada de {entry.predicate}, el claim es de {raw_claim.get('predicate')!r}."
        )
    args = raw_claim.get("args")
    if not isinstance(args, Mapping):
        return mismatch("args no es un objeto.")
    given = normalize_args(args)
    if given != entry.args:
        return mismatch(f"los args no coinciden con {raw_ptr}: snapshot={entry.args}, claim={given}.")
    return Accreditation(
        authority="OBSERVADO",
        provenance_ref=f"tool_result:sha256:{grounding.sha256}",
        evidence_pointer_raw=raw_ptr,
        outcome="ACCREDITED",
        detail=f"acreditado contra {raw_ptr}.",
    )
```

- [ ] **Step 4: Correr los tests**

Run: `cd /home/fruiz/jax && .venv/bin/python -m pytest tests/test_governance_grounding.py -q 2>&1 | tail -1`
Expected: `29 passed` (14 + 15; el parametrize cuenta 5).

- [ ] **Step 5: Commit**

```bash
cd /home/fruiz/jax && git add policy/governance/grounding.py tests/test_governance_grounding.py && git commit -F - <<'EOF'
feat(governance): accredit() -- OBSERVADO solo si el puntero resuelve Y los args coinciden

La citacion falsa (puntero real, args distintos) es MISMATCH, no INFERIDO.
Punteros malformados, incluido "-1", nunca lanzan ni indexan desde el final.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011AVPMngV4W1np18dUPa542
EOF
```

---

## Task 3: `validator.validate()` — el orden normativo 0–6 y los dos estados nuevos

**Files:**
- Modify: `policy/governance/validator.py:41-52` (Literal) y `:228-265` (`validate`)
- Modify: `tests/test_governance_grounding.py` (pruebas 1, 2, 6, 8 en la capa de veredicto)

**Interfaces:**
- Consumes: `grounding.Accreditation`.
- Produces: `validate(claim, predicates, ctx, accreditation: Accreditation | None = None) -> Verdict`. Sin `accreditation` se comporta EXACTAMENTE como hoy (los 39 tests existentes no cambian).

- [ ] **Step 1: Tests del orden (spec §9.1: 1, 2, 6, 8)**

Agregar al final de `tests/test_governance_grounding.py`:

```python
# --- validate() con acreditación: el orden 0-6 del spec §4.1 ----------------

import claims  # noqa: E402
import loaders  # noqa: E402

PREDICATES = {
    "CAPABILITY_AVAILABLE": loaders.PredicateSpec("CAPABILITY_AVAILABLE", ("name", "mode"), "Registro de capabilities"),
    "JOB_STATUS": loaders.PredicateSpec("JOB_STATUS", ("job_id", "status"), "Scheduler"),
}


def _validate(raw, grounding_result, ctx=CTX_A):
    acc = grounding.accredit(raw, grounding_result)
    claim = claims.Claim(
        predicate=raw["predicate"],
        args=grounding.normalize_args(raw["args"]),
        authority=acc.authority,
        provenance_ref=acc.provenance_ref,
        evidence_pointer=acc.evidence_pointer_raw if isinstance(acc.evidence_pointer_raw, str) else "",
        scope="test",
    )
    return validator.validate(claim, PREDICATES, ctx, accreditation=acc)


def test_1_job_status_with_invented_pointer_is_resolver_not_implemented_not_mismatch():
    v = _validate(_raw(predicate="JOB_STATUS", args={"job_id": "1", "status": "ok"},
                       pointer="/capabilities/1"), SNAP_A)
    assert v.status == "RESOLVER_NOT_IMPLEMENTED"


def test_2_capability_with_invented_pointer_is_provenance_mismatch_not_resolver():
    v = _validate(_raw(pointer="/capabilities/99"), SNAP_A)
    assert v.status == "PROVENANCE_MISMATCH"


def test_3_capability_without_pointer_is_authority_invalid():
    v = _validate(_raw(), SNAP_A)
    assert v.status == "AUTHORITY_INVALID"


def test_4_forged_citation_is_provenance_mismatch():
    v = _validate(_raw(args={"name": "write_file", "mode": "read_only"}, pointer="/capabilities/1"), SNAP_A)
    assert v.status == "PROVENANCE_MISMATCH"


def test_5_accredited_claim_reaches_the_resolver_and_is_valid():
    v = _validate(_raw(pointer="/capabilities/1"), SNAP_A)
    assert v.status == "VALID"


def test_6_snapshot_error_is_grounding_unavailable_before_anything_else():
    v = _validate(_raw(pointer="/capabilities/1"), grounding.SnapshotError("boom"))
    assert v.status == "GROUNDING_UNAVAILABLE"
    # También para un predicado sin resolver y para uno desconocido: paso 0 va primero.
    v2 = _validate(_raw(predicate="JOB_STATUS", args={"job_id": "1", "status": "ok"}), grounding.SnapshotError("boom"))
    assert v2.status == "GROUNDING_UNAVAILABLE"


def test_8_fact_mismatch_exercised_by_breaking_the_branch_by_hand():
    # Acredita contra SNAP_A (de CTX_A) -> OBSERVADO. Resuelve contra ctx B
    # = A sin write_file -> FACT_MISMATCH. Las dos capas (spec §4.3) con
    # datos distintos a propósito, porque en producción hoy no puede
    # dispararse (spec §9.4).
    ctx_b = _ctx({"read_file"})
    raw = _raw(pointer="/capabilities/1")
    acc = grounding.accredit(raw, SNAP_A)
    assert acc.authority == "OBSERVADO"
    v = _validate(raw, SNAP_A, ctx=ctx_b)
    assert v.status == "FACT_MISMATCH"


def test_validate_without_accreditation_keeps_legacy_behaviour():
    # Los 39 tests existentes llaman validate() con 3 args: INFERIDO corta,
    # OBSERVADO llega al resolver. Nada de eso cambia.
    c = claims.Claim(predicate="CAPABILITY_AVAILABLE", args={"name": "write_file", "mode": "mutating"},
                     authority="INFERIDO", provenance_ref="x", evidence_pointer="x", scope="t")
    assert validator.validate(c, PREDICATES, CTX_A).status == "AUTHORITY_INVALID"
    c2 = c.model_copy(update={"authority": "OBSERVADO"})
    assert validator.validate(c2, PREDICATES, CTX_A).status == "VALID"


def test_new_statuses_fit_in_varchar_30():
    for s in ("PROVENANCE_MISMATCH", "GROUNDING_UNAVAILABLE"):
        assert len(s) <= 30
```

- [ ] **Step 2: Correr para ver los fallos**

Run: `cd /home/fruiz/jax && .venv/bin/python -m pytest tests/test_governance_grounding.py -q 2>&1 | tail -3`
Expected: fallos por `TypeError: validate() got an unexpected keyword argument 'accreditation'` y por `Literal` (pydantic rechaza `PROVENANCE_MISMATCH`).

- [ ] **Step 3: Implementar en `validator.py`**

Reemplazar el `Literal` de `Verdict.status` (líneas 41–52) por:

```python
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
        # SP3 (2026-09-02). Ver spec §4.2. NO se agrega UNGROUNDED: con la
        # definición acordada es RESOLVER_NOT_IMPLEMENTED, que ya existía y
        # solo era inalcanzable porque authority cortaba antes.
        "PROVENANCE_MISMATCH",     # citó una línea que no dice eso
        "GROUNDING_UNAVAILABLE",   # el snapshot del turno no se construyó
    ]
    predicate: str
    detail: str
```

Agregar `import grounding  # noqa: E402` junto a `import claims  # noqa: E402`.

Reemplazar `validate` (líneas 228–265) por:

```python
def validate(
    claim: "claims.Claim",
    predicates: dict,
    ctx: ValidationContext,
    accreditation: "grounding.Accreditation | None" = None,
) -> Verdict:
    """Orden NORMATIVO (spec §4.1): define qué falla se le imputa a quién.

      0. snapshot del turno en ERROR      -> GROUNDING_UNAVAILABLE   (sistema)
      1. predicado desconocido            -> UNKNOWN_PREDICATE
      2. claves de args mal               -> ARGS_MISMATCH
      3. SIN resolver para el predicado   -> RESOLVER_NOT_IMPLEMENTED (sistema)
      4. con resolver, sin puntero        -> AUTHORITY_INVALID        (modelo)
      5. puntero que no resuelve / args   -> PROVENANCE_MISMATCH      (modelo)
      6. acreditado                       -> resolver

    3 antes de 5: no se acusa de falsear una cita a quien nunca tuvo dónde
    citarla. 0 antes que todo: la falla del sistema no se imputa al modelo.

    Sin `accreditation` (llamadores anteriores a SP3) el comportamiento es
    el de siempre: authority=INFERIDO corta en 4, cualquier otra llega a 6.
    """
    if accreditation is not None and accreditation.outcome == "UNAVAILABLE":
        return Verdict(
            status="GROUNDING_UNAVAILABLE", predicate=claim.predicate, detail=accreditation.detail
        )

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

    resolver = _RESOLVERS.get(claim.predicate)
    if resolver is None:
        reason = _UNIMPLEMENTED_REASONS.get(
            claim.predicate, f"{claim.predicate}: resolver no implementado."
        )
        return Verdict(
            status="RESOLVER_NOT_IMPLEMENTED", predicate=claim.predicate, detail=reason
        )

    if accreditation is None:
        if claim.authority == "INFERIDO":
            return Verdict(
                status="AUTHORITY_INVALID",
                predicate=claim.predicate,
                detail="authority=INFERIDO prohibido en canal claim (§3.1.4).",
            )
        return resolver(claim, ctx)

    if accreditation.outcome == "NO_POINTER":
        return Verdict(
            status="AUTHORITY_INVALID",
            predicate=claim.predicate,
            detail="authority=INFERIDO prohibido en canal claim (§3.1.4): se ofreció grounding y el claim no cita evidence_pointer.",
        )
    if accreditation.outcome == "MISMATCH":
        return Verdict(
            status="PROVENANCE_MISMATCH", predicate=claim.predicate, detail=accreditation.detail
        )
    return resolver(claim, ctx)
```

**Atención al cambio de orden respecto del código anterior:** antes `AUTHORITY_INVALID` (autoridad) se chequeaba ANTES que `RESOLVER_NOT_IMPLEMENTED`. Ahora es al revés (paso 3 antes del 4). Para los tests existentes no cambia nada observable: los que esperaban `RESOLVER_NOT_IMPLEMENTED` usan `authority="OBSERVADO"`, y los que esperaban `AUTHORITY_INVALID` usan un predicado con resolver. Verificarlo en el Step 4 — si algún test existente cambia de veredicto, ESE test describía un orden que el spec §4.1 reemplaza a propósito, y se actualiza con ese comentario.

- [ ] **Step 4: Correr TODA la gobernanza**

Run: `cd /home/fruiz/jax && .venv/bin/python -m pytest tests/test_governance_claims.py tests/test_governance_loaders.py tests/test_governance_validator.py tests/test_governance_vocab_sweep.py tests/test_governance_renderer.py tests/test_governance_grounding.py -q 2>&1 | tail -1`
Expected: `79 passed` (39 + 29 + 11). **Anotar el número real.**

- [ ] **Step 5: Agregar `tests/test_governance_grounding.py` al job `governance` y subir el piso**

En `.github/workflows/policy.yml` (job `governance`, Task 0): agregar `tests/test_governance_grounding.py` al final de las DOS líneas `python -m pytest ...`, y cambiar `"^39 passed"` y el mensaje `39` por el número del Step 4, con este comentario reemplazando el anterior:

```yaml
        # Medido 2026-09-02: 39 (SP1 + arreglo del test rojo) + N
        # (tests/test_governance_grounding.py, SP3) = <número del Step 4>.
```

- [ ] **Step 6: Scanner P10 y commit**

Run: `cd /home/fruiz/jax && .venv/bin/python -m pytest policy/tests/test_no_fail_open_except.py -q 2>&1 | tail -1`
Expected: `passed`.

```bash
cd /home/fruiz/jax && git add policy/governance/validator.py tests/test_governance_grounding.py .github/workflows/policy.yml && git commit -F - <<'EOF'
feat(governance): validate() con el orden normativo 0-6 -- PROVENANCE_MISMATCH y GROUNDING_UNAVAILABLE

Sin resolver va ANTES que sin puntero: no se acusa de falsear una cita a quien
nunca tuvo donde citarla. Sin `accreditation` el comportamiento es el de
siempre. FACT_MISMATCH ejercitado rompiendo la rama a mano (spec 9.4).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011AVPMngV4W1np18dUPa542
EOF
```

- [ ] **Step 7: Push y PR de `jax`**

```bash
cd /home/fruiz/jax && git push -u origin reformas-fase2-sp3-grounding
```

Luego `gh pr create` con título `feat(governance): SP3 -- grounding por snapshot inyectado (capa pura + CI de gobernanza)` y cuerpo que enlace el spec, liste los 4 commits y diga explícitamente: *"Task 0 pone en CI 38 tests que no corrían y arregla uno rojo en master; el resto es la capa pura de SP3. `jax-platform` depende de este merge."* **No mergear hasta que CI esté verde sobre el headSha real** (`gh api repos/fjruizhn/Jax/commits/<sha>/check-runs`).

---

## Task 4: `jax-platform` — migración y contexto compartido

**Files:**
- Modify: `backend/db/migrations.py:112-146` (CREATE TABLE) y la lista `_COLUMNS` (después de la línea `("axioma_usage", "job_id", ...)`)
- Create: `backend/governance_context.py`
- Modify: `backend/shadow_validation.py:60-104` (importar el contexto compartido)
- Test: `backend/tests/test_shadow_validation_grounding.py` (nuevo; solo el test de columnas por ahora)

**Interfaces:**
- Produces: `governance_context.validation_context() -> tuple[ValidationContext, dict[str, PredicateSpec], dict[str, frozenset[str]]]` (cacheado, `maxsize=1`); columnas `shadow_messages.grounding_snapshot LONGTEXT NULL`, `shadow_messages.grounding_snapshot_sha256 CHAR(64) NULL`, `shadow_claim_verdicts.authority VARCHAR(12) NULL`, `shadow_claim_verdicts.evidence_pointer VARCHAR(100) NULL`.
- `shadow_validation._validation_context` sigue existiendo como nombre (alias del compartido) para que `patch.object(shadow_validation, "_validation_context", ...)` en `tests/test_shadow_validation.py:222` siga funcionando.

- [ ] **Step 0: Rama**

```bash
cd /home/fruiz/jax-platform && git fetch origin && git checkout -b reformas-fase2-sp3-grounding origin/master && git rev-parse --abbrev-ref HEAD
```
Expected: `reformas-fase2-sp3-grounding`.

- [ ] **Step 1: Test de que las columnas existen tras las migraciones**

Crear `backend/tests/test_shadow_validation_grounding.py`:

```python
"""
Shadow validation con grounding (REFORMAS Fase 2 SP3) — spec §9.2.

Corre contra jax_memory_test (fixture `client` levanta la app y corre las
migraciones). Con JAX_CI_NO_DB=1 todo esto se salta por la Regla 1 de
conftest.py.
"""
from __future__ import annotations

import json
import uuid

import pytest

from api.chat import ContractResult


async def _columns(table):
    from db.connection import get_pool
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT COLUMN_NAME, COLUMN_TYPE FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s",
                (table,),
            )
            return {name: ctype for name, ctype in await cur.fetchall()}


def test_migration_adds_the_four_grounding_columns(client):
    sm = client.portal.call(_columns, "shadow_messages")
    assert sm["grounding_snapshot"] == "longtext"
    assert sm["grounding_snapshot_sha256"] == "char(64)"
    cv = client.portal.call(_columns, "shadow_claim_verdicts")
    assert cv["authority"] == "varchar(12)"
    assert cv["evidence_pointer"] == "varchar(100)"
```

- [ ] **Step 2: Correr para verificar que falla**

Run: `cd /home/fruiz/jax-platform/backend && python -m pytest tests/test_shadow_validation_grounding.py -q 2>&1 | tail -2`
Expected: `KeyError: 'grounding_snapshot'`.

- [ ] **Step 3: Migración — las dos vías (instalación nueva y existente)**

En `backend/db/migrations.py`, dentro de `CREATE_SHADOW_MESSAGES`, después de la línea `validated_at TIMESTAMP NULL DEFAULT NULL,` agregar:

```sql
  grounding_snapshot LONGTEXT NULL,
  grounding_snapshot_sha256 CHAR(64) NULL,
```

Dentro de `CREATE_SHADOW_CLAIM_VERDICTS`, después de `args JSON,` agregar:

```sql
  authority VARCHAR(12) NULL,
  evidence_pointer VARCHAR(100) NULL,
```

En la lista `_COLUMNS`, después de la entrada `("axioma_usage", "job_id", ...)`, agregar:

```python
    # SP3 grounding (2026-09-02). shadow_messages: qué vio el modelo y su
    # hash. Tres estados distinguibles a propósito (spec §5.4): NULL = turno
    # anterior a esta migración; 'ERROR' = el snapshot falló al construirse;
    # 64 hex = snapshot real. shadow_claim_verdicts: authority SIEMPRE
    # derivada por el servidor, nunca lo que mandó el modelo (spec §9.1);
    # evidence_pointer tal como se recibió, truncado a 100 (el original va a
    # `detail` si excede).
    ("shadow_messages", "grounding_snapshot",
     "ALTER TABLE shadow_messages ADD COLUMN grounding_snapshot LONGTEXT NULL"),
    ("shadow_messages", "grounding_snapshot_sha256",
     "ALTER TABLE shadow_messages ADD COLUMN grounding_snapshot_sha256 CHAR(64) NULL"),
    ("shadow_claim_verdicts", "authority",
     "ALTER TABLE shadow_claim_verdicts ADD COLUMN authority VARCHAR(12) NULL"),
    ("shadow_claim_verdicts", "evidence_pointer",
     "ALTER TABLE shadow_claim_verdicts ADD COLUMN evidence_pointer VARCHAR(100) NULL"),
```

- [ ] **Step 4: Correr el test**

Run: `cd /home/fruiz/jax-platform/backend && python -m pytest tests/test_shadow_validation_grounding.py -q 2>&1 | tail -1`
Expected: `1 passed`.

- [ ] **Step 5: `governance_context.py` — un solo contexto para `chat.py` y `shadow_validation.py`**

Crear `backend/governance_context.py`:

```python
"""
Contexto de gobernanza compartido (REFORMAS Fase 2 SP3).

Antes vivía como shadow_validation._validation_context(). Se mueve acá
porque chat.py también lo necesita (para construir el snapshot del turno,
spec §5.1) y chat.py NO puede importar shadow_validation a nivel de módulo:
shadow_validation importa `from api.chat import ContractResult` -- ciclo.

Config estática cacheada por proceso, mismo criterio que _load_config() en
chat.py (Lección operativa #6, jax-platform/CLAUDE.md): un cambio real
requiere reiniciar el proceso. Consecuencia declarada en el spec §9.4: el
snapshot y el resolver de CAPABILITY_AVAILABLE leen el MISMO objeto, así que
FACT_MISMATCH no puede dispararse por drift para ese predicado hoy.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

JAX_REPO = Path(os.getenv("JAX_REPO_PATH", os.path.expanduser("~/jax")))
if str(JAX_REPO) not in sys.path:
    sys.path.insert(0, str(JAX_REPO))
if str(JAX_REPO / "policy" / "governance") not in sys.path:
    sys.path.insert(0, str(JAX_REPO / "policy" / "governance"))

import loaders as governance_loaders  # noqa: E402
import validator as governance_validator  # noqa: E402


@lru_cache(maxsize=1)
def validation_context():
    vocabulary = governance_loaders.load_vocabulary()
    ctx = governance_validator.load_validation_context(JAX_REPO, vocabulary.config_paths)
    predicates = governance_loaders.load_predicates()
    return ctx, predicates, vocabulary.term_categories
```

En `backend/shadow_validation.py`: borrar la función `_validation_context` (líneas 93–103, con su decorador) y en su lugar, después de `from db.connection import get_pool  # noqa: E402`, poner:

```python
# El contexto vive en governance_context.py desde SP3 (lo comparte chat.py).
# Se conserva el nombre _validation_context en este módulo a propósito:
# tests/test_shadow_validation.py lo parchea por nombre.
from governance_context import validation_context as _validation_context  # noqa: E402
```

- [ ] **Step 6: La suite existente de shadow validation sigue en verde**

Run: `cd /home/fruiz/jax-platform/backend && python -m pytest tests/test_shadow_validation.py tests/test_shadow_validation_grounding.py -q 2>&1 | tail -1`
Expected: todo `passed` (incluido `test_shadow_validation_leaves_validated_at_null_when_context_load_fails_before_the_insert`, que parchea `_validation_context`).

- [ ] **Step 7: Commit**

```bash
cd /home/fruiz/jax-platform && git add backend/db/migrations.py backend/governance_context.py backend/shadow_validation.py backend/tests/test_shadow_validation_grounding.py && git commit -F - <<'EOF'
feat(shadow): columnas de grounding y contexto de gobernanza compartido -- SP3, parte 1

Cuatro columnas nuevas (snapshot, sha256, authority, evidence_pointer) por
las dos vias (CREATE para instalacion nueva, ALTER idempotente para la
existente). _validation_context() pasa a governance_context.py para que
chat.py lo use sin import circular.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011AVPMngV4W1np18dUPa542
EOF
```

---

## Task 5: `shadow_validation.py` — acreditar antes de validar, persistir todo

**Files:**
- Modify: `backend/shadow_validation.py:154-200` (`run_shadow_validation`), `:106-152` (helpers)
- Modify: `backend/tests/test_shadow_validation_grounding.py`

**Interfaces:**
- Produces: `run_shadow_validation(conv_uuid, shadow_message_id, facet, contract, grounding)` — **quinto argumento obligatorio, sin default** (spec §9.2: es lo que cierra el camino a filas `NULL` nuevas). `grounding: grounding.Snapshot | grounding.SnapshotError`.
- Consumes: `grounding.accredit`, `validator.validate(..., accreditation=)` de `jax` (Tasks 2–3, ya mergeados o en `JAX_REPO_PATH`).

- [ ] **Step 1: Tests §9.2 — cada estado produce su fila**

Agregar a `backend/tests/test_shadow_validation_grounding.py`:

```python
import governance_context  # noqa: E402  (ya está en sys.path por conftest→main)
import grounding as governance_grounding  # noqa: E402


async def _fetch_message_grounding(shadow_message_id):
    from db.connection import get_pool
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT grounding_snapshot, grounding_snapshot_sha256, validated_at "
                "FROM shadow_messages WHERE shadow_message_id = %s",
                (shadow_message_id,),
            )
            return await cur.fetchone()


async def _fetch_verdicts(shadow_message_id):
    from db.connection import get_pool
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT predicate, status, authority, evidence_pointer, detail, args "
                "FROM shadow_claim_verdicts WHERE shadow_message_id = %s ORDER BY id",
                (shadow_message_id,),
            )
            return await cur.fetchall()


def _snapshot():
    ctx, _, _ = governance_context.validation_context()
    return governance_grounding.build_snapshot(ctx)


def _pointer_of(snap, name):
    return next(e.pointer for e in snap.entries if e.args["name"] == name)


def _contract(claims):
    return ContractResult(
        contract_parsed=True, claims=claims, analysis="a", judgment=None,
        degradation_reason=None, raw_text="...",
    )


def _run(client, contract, grounding_result, smid=None):
    from shadow_validation import run_shadow_validation
    smid = smid or str(uuid.uuid4())
    client.portal.call(run_shadow_validation, "conv-sp3", smid, "jekyll", contract, grounding_result)
    return smid


def test_accredited_claim_is_observado_and_valid_with_pointer_persisted(client):
    snap = _snapshot()
    ptr = _pointer_of(snap, "write_file")
    smid = _run(client, _contract([{"predicate": "CAPABILITY_AVAILABLE",
                                    "args": {"name": "write_file", "mode": "mutating"},
                                    "evidence_pointer": ptr}]), snap)
    (predicate, status, authority, pointer, detail, args), = client.portal.call(_fetch_verdicts, smid)
    assert (predicate, status, authority, pointer) == ("CAPABILITY_AVAILABLE", "VALID", "OBSERVADO", ptr)
    snapshot_json, sha, validated_at = client.portal.call(_fetch_message_grounding, smid)
    assert sha == snap.sha256
    assert json.loads(snapshot_json) == json.loads(snap.canonical_json)
    assert validated_at is not None


def test_no_pointer_is_authority_invalid_with_null_pointer(client):
    smid = _run(client, _contract([{"predicate": "CAPABILITY_AVAILABLE",
                                    "args": {"name": "write_file", "mode": "mutating"}}]), _snapshot())
    (_, status, authority, pointer, _, _), = client.portal.call(_fetch_verdicts, smid)
    assert (status, authority, pointer) == ("AUTHORITY_INVALID", "INFERIDO", None)


def test_forged_citation_is_provenance_mismatch(client):
    snap = _snapshot()
    smid = _run(client, _contract([{"predicate": "CAPABILITY_AVAILABLE",
                                    "args": {"name": "write_file", "mode": "read_only"},
                                    "evidence_pointer": _pointer_of(snap, "write_file")}]), snap)
    (_, status, authority, _, _, _), = client.portal.call(_fetch_verdicts, smid)
    assert (status, authority) == ("PROVENANCE_MISMATCH", "INFERIDO")


def test_job_status_with_pointer_is_resolver_not_implemented(client):
    snap = _snapshot()
    smid = _run(client, _contract([{"predicate": "JOB_STATUS",
                                    "args": {"job_id": "1", "status": "ok"},
                                    "evidence_pointer": _pointer_of(snap, "write_file")}]), snap)
    (_, status, authority, _, _, _), = client.portal.call(_fetch_verdicts, smid)
    assert (status, authority) == ("RESOLVER_NOT_IMPLEMENTED", "INFERIDO")


def test_snapshot_error_marks_turn_ERROR_and_claims_grounding_unavailable(client):
    smid = _run(client, _contract([{"predicate": "CAPABILITY_AVAILABLE",
                                    "args": {"name": "write_file", "mode": "mutating"},
                                    "evidence_pointer": "/capabilities/0"}]),
                governance_grounding.SnapshotError("config ilegible"))
    snapshot_json, sha, validated_at = client.portal.call(_fetch_message_grounding, smid)
    assert sha == "ERROR"
    assert json.loads(snapshot_json) == {"error": "config ilegible"}
    assert validated_at is not None
    (_, status, authority, _, detail, _), = client.portal.call(_fetch_verdicts, smid)
    assert (status, authority) == ("GROUNDING_UNAVAILABLE", "INFERIDO")
    assert "config ilegible" in detail


@pytest.mark.parametrize("bad", ["", "capabilities/0", "/capabilities/abc", "/capabilities/-1"])
def test_9_1b_malformed_pointer_completes_the_task(client, bad):
    smid = _run(client, _contract([{"predicate": "CAPABILITY_AVAILABLE",
                                    "args": {"name": "write_file", "mode": "mutating"},
                                    "evidence_pointer": bad}]), _snapshot())
    (_, status, _, pointer, _, _), = client.portal.call(_fetch_verdicts, smid)
    assert status == "PROVENANCE_MISMATCH"
    assert pointer == bad
    _, _, validated_at = client.portal.call(_fetch_message_grounding, smid)
    assert validated_at is not None


def test_9_1b_300_char_pointer_is_truncated_to_100_with_original_in_detail(client):
    long = "/capabilities/" + "9" * 286
    assert len(long) == 300
    smid = _run(client, _contract([{"predicate": "CAPABILITY_AVAILABLE",
                                    "args": {"name": "write_file", "mode": "mutating"},
                                    "evidence_pointer": long}]), _snapshot())
    (_, status, _, pointer, detail, _), = client.portal.call(_fetch_verdicts, smid)
    assert status == "PROVENANCE_MISMATCH"
    assert pointer == long[:100]
    assert long in detail


def test_model_declared_authority_never_enters_the_authority_column(client):
    snap = _snapshot()
    smid = _run(client, _contract([{"predicate": "CAPABILITY_AVAILABLE",
                                    "args": {"name": "write_file", "mode": "mutating"},
                                    "authority": "EJECUTADO"}]), snap)
    (_, status, authority, _, detail, _), = client.portal.call(_fetch_verdicts, smid)
    assert authority == "INFERIDO"          # derivada por el servidor: sin puntero
    assert status == "AUTHORITY_INVALID"
    assert "EJECUTADO" in detail            # lo que mandó el modelo queda en el raw


def test_every_row_written_by_run_shadow_validation_has_non_null_sha256(client):
    # spec §9.2: NULL después de SP3 es solo legado. Con snapshot:
    smid1 = _run(client, _contract([]), _snapshot())
    # y con error:
    smid2 = _run(client, _contract([]), governance_grounding.SnapshotError("x"))
    for smid in (smid1, smid2):
        _, sha, _ = client.portal.call(_fetch_message_grounding, smid)
        assert sha is not None


def test_fifth_argument_is_mandatory(client):
    import inspect
    from shadow_validation import run_shadow_validation
    p = inspect.signature(run_shadow_validation).parameters["grounding"]
    assert p.default is inspect.Parameter.empty
```

- [ ] **Step 2: Correr para ver los fallos**

Run: `cd /home/fruiz/jax-platform/backend && python -m pytest tests/test_shadow_validation_grounding.py -q 2>&1 | tail -3`
Expected: `TypeError: run_shadow_validation() takes 4 positional arguments but 5 were given` y `KeyError: 'grounding'`.

- [ ] **Step 3: Implementar en `shadow_validation.py`**

Agregar a los imports de gobernanza:

```python
import grounding as governance_grounding  # noqa: E402
```

Reemplazar `_insert_shadow_message` (líneas 106–125) por:

```python
def _grounding_columns(grounding_result) -> tuple[str, str]:
    """(grounding_snapshot, grounding_snapshot_sha256). Tres estados a
    propósito (spec §5.4): 'ERROR' con el motivo cuando el snapshot falló;
    64 hex con el JSON canónico cuando existe. NULL no sale de acá nunca:
    NULL = fila anterior a SP3."""
    if isinstance(grounding_result, governance_grounding.SnapshotError):
        return json.dumps({"error": grounding_result.reason}, ensure_ascii=False), "ERROR"
    return grounding_result.canonical_json, grounding_result.sha256


async def _insert_shadow_message(cur, conv_uuid, shadow_message_id, facet, contract, grounding_result):
    # Defensa en profundidad (finding 1 de la revisión final): api/chat.py
    # ya valida facet contra la whitelist de config["personalities"] antes
    # de llegar acá, pero este módulo es importable/invocable por
    # cualquier otro caller de run_shadow_validation — clampear a 30
    # caracteres acá asegura que este INSERT (el primero de la función,
    # ver comentario en run_shadow_validation) nunca falle con
    # "Data too long" por esta columna específicamente, sin importar quién
    # llame. shadow_messages.facet es VARCHAR(30) (db/migrations.py).
    snapshot_json, sha = _grounding_columns(grounding_result)
    await cur.execute(
        "INSERT INTO shadow_messages "
        "(conv_uuid, shadow_message_id, facet, contract_parsed, degradation_reason, "
        "has_claim, has_analysis, has_judgment, grounding_snapshot, grounding_snapshot_sha256) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            conv_uuid, shadow_message_id, facet[:30], contract.contract_parsed,
            contract.degradation_reason,
            bool(contract.claims), bool(contract.analysis), bool(contract.judgment),
            snapshot_json, sha,
        ),
    )
```

Reemplazar `_insert_claim_verdict` (líneas 134–141) por:

```python
_POINTER_COLUMN_WIDTH = 100  # shadow_claim_verdicts.evidence_pointer VARCHAR(100)


async def _insert_claim_verdict(cur, conv_uuid, shadow_message_id, verdict, raw_claim, accreditation):
    # authority: SIEMPRE la derivada por el servidor (spec §9.1). Si el
    # modelo mandó un campo authority, no entra acá: va al detail.
    detail = verdict.detail
    declared = raw_claim.get("authority")
    if declared is not None:
        detail += f" | el modelo declaró authority={declared!r} (ignorado: la autoridad la deriva el servidor)."
    # evidence_pointer: tal como se recibió, truncado al ancho de la columna;
    # si se truncó, el original completo va al detail (spec §9.1b).
    pointer = accreditation.evidence_pointer_raw
    pointer_db = None
    if pointer is not None:
        as_text = pointer if isinstance(pointer, str) else repr(pointer)
        pointer_db = as_text[:_POINTER_COLUMN_WIDTH]
        if len(as_text) > _POINTER_COLUMN_WIDTH:
            detail += f" | evidence_pointer truncado a {_POINTER_COLUMN_WIDTH}; original: {as_text}"
    await cur.execute(
        "INSERT INTO shadow_claim_verdicts "
        "(conv_uuid, shadow_message_id, predicate, status, detail, args, authority, evidence_pointer) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (conv_uuid, shadow_message_id, verdict.predicate, verdict.status, detail,
         json.dumps(raw_claim["args"]), accreditation.authority, pointer_db),
    )
```

Reemplazar la firma y el cuerpo del loop de claims en `run_shadow_validation`:

```python
async def run_shadow_validation(
    conv_uuid: str | None,
    shadow_message_id: str,
    facet: str,
    contract: "ContractResult | None",
    grounding: "governance_grounding.Snapshot | governance_grounding.SnapshotError",
) -> None:
    """`grounding` es OBLIGATORIO y sin default a propósito (spec §9.2): es
    lo que garantiza que ninguna fila nueva de shadow_messages quede con
    grounding_snapshot_sha256 NULL. Un caller que lo omita falla al llamar,
    no produce una fila ambigua."""
    if conv_uuid is None or contract is None:
        return
```

(el resto de la cabecera igual), y donde dice `await _insert_shadow_message(cur, conv_uuid, shadow_message_id, facet, contract)` pasar también `grounding`:

```python
                await _insert_shadow_message(cur, conv_uuid, shadow_message_id, facet, contract, grounding)
```

Y el loop de claims (líneas 186–198) por:

```python
                for raw_claim in contract.claims:
                    # 1) acreditar contra el snapshot del turno (grounding.py,
                    #    puro): de acá salen authority y provenance_ref.
                    accreditation = governance_grounding.accredit(raw_claim, grounding)
                    claim = governance_claims.Claim(
                        predicate=raw_claim["predicate"],
                        args=governance_grounding.normalize_args(raw_claim["args"]),
                        authority=accreditation.authority,
                        provenance_ref=accreditation.provenance_ref,
                        evidence_pointer=(
                            accreditation.evidence_pointer_raw
                            if isinstance(accreditation.evidence_pointer_raw, str) else ""
                        ),
                        scope="mesa_web",
                    )
                    # 2) veredicto en el orden normativo del spec §4.1.
                    verdict = governance_validator.validate(claim, predicates, ctx, accreditation=accreditation)
                    await _insert_claim_verdict(
                        cur, conv_uuid, shadow_message_id, verdict, raw_claim, accreditation
                    )
```

Actualizar el docstring del módulo (líneas 12–16): reemplazar *"authority de todo claim es SIEMPRE "INFERIDO", fijado acá"* por:

```
authority de todo claim la DERIVA EL SERVIDOR acreditando el claim contra el
snapshot que se inyectó en ese turno (grounding.py, SP3, 2026-09-02): OBSERVADO
si citó una línea del snapshot y los args coinciden, INFERIDO en cualquier
otro caso. Nunca lo declara el modelo (P08).
```

- [ ] **Step 4: Arreglar los llamadores existentes en tests**

`tests/test_shadow_validation.py` llama `run_shadow_validation` con 4 argumentos en ~8 tests. Agregar al principio del archivo (después de los imports):

```python
def _grounding():
    import governance_context
    import grounding as governance_grounding
    ctx, _, _ = governance_context.validation_context()
    return governance_grounding.build_snapshot(ctx)
```

y en cada `client.portal.call(run_shadow_validation, <conv>, smid, <facet>, contract)` agregar `, _grounding()` como último argumento. En `test_shadow_validation_claim_produces_authority_invalid_verdict`: el claim no trae puntero → sigue `AUTHORITY_INVALID`; actualizar el comentario *"Resultado esperado de esta ronda..."* por *"Sin evidence_pointer la autoridad es INFERIDO: se ofreció grounding y no citó (spec §4.1 paso 4)."*.

- [ ] **Step 5: Correr las dos suites**

Run: `cd /home/fruiz/jax-platform/backend && python -m pytest tests/test_shadow_validation.py tests/test_shadow_validation_grounding.py -q 2>&1 | tail -1`
Expected: todo `passed`.

- [ ] **Step 6: Scanner P10 y commit**

Run: `cd /home/fruiz/jax-platform && python -m pytest backend/tests/test_no_fail_open_except.py -q 2>&1 | tail -1` (o el comando que use el job `no-fail-open-except` de `.github/workflows/policy.yml` — copiarlo de ahí).
Expected: `passed`.

```bash
cd /home/fruiz/jax-platform && git add backend/shadow_validation.py backend/tests/test_shadow_validation.py backend/tests/test_shadow_validation_grounding.py && git commit -F - <<'EOF'
feat(shadow): acreditar contra el snapshot del turno antes de validar -- SP3, parte 2

Quinto argumento obligatorio. authority siempre derivada por el servidor;
evidence_pointer tal como llego, truncado a 100 con el original en detail.
Cada estado del spec 4.1 provocado a proposito contra la DB.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011AVPMngV4W1np18dUPa542
EOF
```

---

## Task 6: `chat.py` — el contrato admite `evidence_pointer`, y el snapshot viaja

**Files:**
- Modify: `backend/api/chat.py:479-527` (parser), `:565-580` (sufijo), `:855-870` (dispatch), `:956-975` (`_invoke_facet`), `:1040-1110` (endpoint)
- Modify: `backend/tests/test_chat_contract_prompt.py`
- Create: `backend/tests/test_chat_grounding_wiring.py`

**Interfaces:**
- Produces: `_build_grounding() -> Snapshot | SnapshotError` (nunca lanza); `_invoke_facet(..., *, source=..., grounding=None)`; `_invoke_facet_dispatch(..., grounding=None)`; `_parse_contract_response` conserva `evidence_pointer` y `authority` si vienen.
- Consumes: `governance_context.validation_context()`, `grounding.build_snapshot/render/SnapshotError`, `run_shadow_validation(..., grounding)`.

- [ ] **Step 1: Tests del contrato y del parser (regresión de la línea 576)**

Agregar a `backend/tests/test_chat_contract_prompt.py`:

```python
def test_contract_suffix_admits_evidence_pointer_and_no_longer_forbids_it():
    # Regresión (spec §6.3 / §9.3): hasta SP3 el sufijo decía "SOLO esos dos
    # campos, nada más" y "No incluyas ningún otro campo" -- el modelo NO
    # PODÍA citar. Restaurar cualquiera de las dos frases pone rojo.
    assert "SOLO esos dos campos" not in _CONTRACT_PROMPT_SUFFIX
    assert "No incluyas ningún otro campo" not in _CONTRACT_PROMPT_SUFFIX
    assert '"evidence_pointer"' in _CONTRACT_PROMPT_SUFFIX


def test_parser_keeps_evidence_pointer_and_model_declared_authority():
    from api.chat import _parse_contract_response
    raw = ('{"claim": [{"predicate": "CAPABILITY_AVAILABLE", "args": {"name": "x", "mode": "read_only"}, '
           '"evidence_pointer": "/capabilities/3", "authority": "EJECUTADO", "otro": 1}], '
           '"analysis": "a", "judgment": null}')
    r = _parse_contract_response(raw)
    assert r.contract_parsed is True
    assert r.claims == [{
        "predicate": "CAPABILITY_AVAILABLE", "args": {"name": "x", "mode": "read_only"},
        "evidence_pointer": "/capabilities/3", "authority": "EJECUTADO",
    }]  # "otro" se descarta; evidence_pointer y authority se conservan (spec §9.1)


def test_parser_keeps_non_string_evidence_pointer_for_accredit_to_reject():
    # No se degrada el contrato por un puntero raro: eso es PROVENANCE_MISMATCH
    # en shadow validation (spec §9.1b), no un contrato roto.
    from api.chat import _parse_contract_response
    raw = '{"claim": [{"predicate": "P", "args": {}, "evidence_pointer": 7}], "analysis": "a"}'
    r = _parse_contract_response(raw)
    assert r.contract_parsed is True
    assert r.claims[0]["evidence_pointer"] == 7
```

- [ ] **Step 2: Correr para ver los fallos**

Run: `cd /home/fruiz/jax-platform/backend && JAX_CI_NO_DB=1 python -m pytest tests/test_chat_contract_prompt.py -q 2>&1 | tail -3`
Expected: 3 fallos nuevos (las frases prohibidas siguen ahí; el parser descarta los campos).

- [ ] **Step 3: Parser y sufijo**

En `_parse_contract_response`, reemplazar la línea `parsed_claims.append({"predicate": item["predicate"], "args": item["args"]})` por:

```python
        parsed = {"predicate": item["predicate"], "args": item["args"]}
        # SP3: evidence_pointer es lo ÚNICO que el modelo puede aportar a la
        # acreditación (spec §5.1). Se conserva tal cual, sin validar tipo:
        # un puntero raro es PROVENANCE_MISMATCH en shadow validation, no un
        # contrato roto. authority se conserva SOLO para dejar constancia en
        # `detail` de que el modelo intentó declararla -- nunca entra en la
        # columna authority (spec §9.1).
        for passthrough in ("evidence_pointer", "authority"):
            if passthrough in item:
                parsed[passthrough] = item[passthrough]
        parsed_claims.append(parsed)
```

En `_CONTRACT_SUFFIX_TEMPLATE`, reemplazar la línea que empieza con `  Cada claim es {"predicate": "...", "args": {...}} — SOLO esos dos campos, nada más.` por:

```
  Cada claim es {"predicate": "...", "args": {...}, "evidence_pointer": "/<lista>/<n>"}. El evidence_pointer es la línea de HECHOS VERIFICADOS que respalda el claim; si no hay una línea que lo respalde, no lo emitas como claim. Poné [] únicamente si tu respuesta no afirma nada sobre el estado del sistema.
```

y la línea final `No incluyas ningún otro campo. No expliques el formato, solo respondé el JSON.` por:

```
No expliques el formato, solo respondé el JSON.
```

También la línea del ejemplo JSON del principio del template: cambiar `{"claim": [{"predicate": "NOMBRE", "args": {"clave": "valor"}}], ...` por `{"claim": [{"predicate": "NOMBRE", "args": {"clave": "valor"}, "evidence_pointer": "/capabilities/0"}], ...`.

- [ ] **Step 4: Correr**

Run: `cd /home/fruiz/jax-platform/backend && JAX_CI_NO_DB=1 python -m pytest tests/test_chat_contract_prompt.py tests/test_chat_contract_wrapper.py -q 2>&1 | tail -1`
Expected: todo `passed` (los tests de `_parse_contract_response` que comparan `claims == [{"predicate":..., "args":...}]` con claims sin campos extra siguen iguales).

- [ ] **Step 5: Test §9.3 — un objeto, dos consumidores**

Crear `backend/tests/test_chat_grounding_wiring.py`:

```python
"""
Spec §9.3: el camino de PRODUCCIÓN. Un POST /api/chat real produce UN solo
objeto snapshot y se verifica en sus dos consumidores:
  1. render(snapshot) está en el system prompt que salió al proveedor;
  2. ese mismo objeto es el que se encoló para run_shadow_validation.
La persistencia del sha256 de ese objeto la cubre
tests/test_shadow_validation_grounding.py (necesita conv_uuid, que acá es
None por los ids no numéricos -- mismo patrón que test_chat_contract_wrapper).
"""
from __future__ import annotations

import http_client
from unittest.mock import patch

from tests.test_chat_contract_wrapper import _FakeResponse


class _RecordingPostClient:
    def __init__(self):
        self.payloads = []

    async def post(self, url, **kwargs):
        if "/motor/authorize-facet" in url:
            return _FakeResponse({"allowed": True, "reason": "OK"})
        self.payloads.append(kwargs.get("json"))
        return _FakeResponse({
            "choices": [{"message": {"content": '{"claim": [], "analysis": "ok", "judgment": null}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })


def test_same_snapshot_object_reaches_prompt_and_background_task(client):
    import grounding as governance_grounding
    from auth.jwt import create_access_token
    token = create_access_token("test-grounding-user", "test-grounding-tenant", "operator")
    fake = _RecordingPostClient()
    captured = {}

    def spy_add_safe_task(background_tasks, fn, *args):
        captured["args"] = args

    original = http_client._client
    http_client._client = fake
    try:
        with patch("jax_engine.background.add_safe_task", side_effect=spy_add_safe_task):
            resp = client.post("/api/chat", json={"message": "hola", "facet": "jekyll"},
                               headers={"Authorization": f"Bearer {token}"})
    finally:
        http_client._client = original
    assert resp.status_code == 200, resp.text

    # consumidor 2: el background task recibió el snapshot como 5º argumento
    conv_uuid, smid, facet, contract, grounding_obj = captured["args"]
    assert isinstance(grounding_obj, governance_grounding.Snapshot)
    assert len(grounding_obj.sha256) == 64

    # consumidor 1: el system prompt que salió contiene render() de ESE objeto
    assert len(fake.payloads) == 1
    system_prompt = fake.payloads[0]["messages"][0]["content"]
    assert governance_grounding.render(grounding_obj) in system_prompt
    # y el hash NO viajó (spec §5.1)
    assert grounding_obj.sha256 not in system_prompt


def test_snapshot_build_failure_is_marked_not_hidden(client):
    import grounding as governance_grounding
    from auth.jwt import create_access_token
    import api.chat as chat
    token = create_access_token("test-grounding-user-2", "test-grounding-tenant-2", "operator")
    fake = _RecordingPostClient()
    captured = {}

    def spy_add_safe_task(background_tasks, fn, *args):
        captured["args"] = args

    def boom():
        raise governance_grounding.GroundingBuildError("config ilegible")

    original = http_client._client
    http_client._client = fake
    try:
        with patch.object(chat, "_build_snapshot_or_raise", side_effect=boom), \
             patch("jax_engine.background.add_safe_task", side_effect=spy_add_safe_task):
            resp = client.post("/api/chat", json={"message": "hola", "facet": "jekyll"},
                               headers={"Authorization": f"Bearer {token}"})
    finally:
        http_client._client = original
    # el turno responde igual (el grounding es medición, no puede tumbar un chat)
    assert resp.status_code == 200, resp.text
    grounding_obj = captured["args"][4]
    assert isinstance(grounding_obj, governance_grounding.SnapshotError)
    assert "config ilegible" in grounding_obj.reason
    # y el prompt salió SIN bloque de hechos
    assert "HECHOS VERIFICADOS" not in fake.payloads[0]["messages"][0]["content"]
```

- [ ] **Step 6: Correr para ver los fallos**

Run: `cd /home/fruiz/jax-platform/backend && python -m pytest tests/test_chat_grounding_wiring.py -q 2>&1 | tail -3`
Expected: fallos (`captured["args"]` tiene 4 elementos; no existe `_build_snapshot_or_raise`).

- [ ] **Step 7: Cablear `chat.py`**

Después del bloque que define `_CONTRACT_PROMPT_SUFFIX` (línea ~593), agregar:

```python
# --- Grounding por snapshot (REFORMAS Fase 2 SP3, spec §5) -------------------
import grounding as governance_grounding  # noqa: E402  (mismo sys.path que loaders)
from governance_context import validation_context as _governance_context  # noqa: E402


def _build_snapshot_or_raise() -> "governance_grounding.Snapshot":
    """Separado de _build_grounding para poder parchearlo en tests."""
    ctx, _, _ = _governance_context()
    return governance_grounding.build_snapshot(ctx)


def _build_grounding() -> "governance_grounding.Snapshot | governance_grounding.SnapshotError":
    """Nunca lanza. build_snapshot() sí lanza (P10) -- acá se captura, se
    LOGUEA con traceback, y se convierte en la marca SnapshotError que
    viaja al validador y termina como grounding_snapshot_sha256='ERROR'
    (spec §5.4). El turno de chat responde igual: el grounding es medición
    y no puede tumbar un chat, mismo criterio que el encolado de shadow
    validation más abajo. Lo que NO se hace: devolver un snapshot vacío,
    que sería indistinguible de "no hay capabilities"."""
    try:
        return _build_snapshot_or_raise()
    except Exception as e:
        logger.exception("no se pudo construir el snapshot de grounding")
        return governance_grounding.SnapshotError(f"{type(e).__name__}: {e}")
```

En `_invoke_facet_dispatch`: agregar el parámetro `grounding=None` a la firma:

```python
async def _invoke_facet_dispatch(
    facet: str, config: dict, user_id: str, message: str,
    semantic_context: list[dict] | None = None,
    grounding: "governance_grounding.Snapshot | governance_grounding.SnapshotError | None" = None,
) -> tuple[str, UsageInfo | None, str]:
```

y reemplazar la línea `system_prompt = personality.get("system_prompt", "Sos JAX.") + _CONTRACT_PROMPT_SUFFIX` por:

```python
    system_prompt = personality.get("system_prompt", "Sos JAX.") + _CONTRACT_PROMPT_SUFFIX
    # SP3: el bloque de hechos va DESPUÉS del sufijo de contrato, sin el
    # hash (spec §5.1). Con SnapshotError no se anexa nada -- el modelo no
    # tiene con qué citar y el validador lo sabe por la marca. Con None
    # (la sonda de facet_canary) tampoco: la sonda no corre shadow validation.
    if isinstance(grounding, governance_grounding.Snapshot):
        system_prompt += "\n\n" + governance_grounding.render(grounding)
```

En `_invoke_facet`: agregar `grounding=None` a la firma (keyword-only, después de `source`) y pasarlo:

```python
async def _invoke_facet(
    facet: str, config: dict, user_id: str, message: str,
    semantic_context: list[dict] | None = None,
    *, source: str = SOURCE_CHAT,
    grounding: "governance_grounding.Snapshot | governance_grounding.SnapshotError | None" = None,
) -> tuple[str, UsageInfo | None]:
```
```python
        texto, usage, outcome = await _invoke_facet_dispatch(
            facet, config, user_id, message, semantic_context, grounding=grounding)
```

En el endpoint `chat`: justo antes de `try:\n        response_text, usage = await _invoke_facet(...)`, agregar:

```python
    # SP3: UN snapshot por turno, construido acá y pasado a sus dos
    # consumidores (el prompt y el background task) -- spec §9.3.
    grounding = _build_grounding()
```

cambiar la llamada a:

```python
        response_text, usage = await _invoke_facet(
            facet, config, user_id, req.message, semantic_context, grounding=grounding)
```

y la línea de encolado a:

```python
        add_safe_task(background_tasks, run_shadow_validation, conv_uuid, shadow_message_id, facet, contract, grounding)
```

- [ ] **Step 8: Correr todo lo que toca chat**

Run: `cd /home/fruiz/jax-platform/backend && python -m pytest tests/test_chat_grounding_wiring.py tests/test_chat_contract_wrapper.py tests/test_chat_contract_prompt.py tests/test_shadow_validation.py tests/test_shadow_validation_grounding.py tests/test_facet_canary.py -q 2>&1 | tail -1`
Expected: todo `passed`. Si `test_facet_canary` falla por la firma: la sonda parchea `_invoke_facet` entero, no debería — si falla, leer el error antes de tocar.

- [ ] **Step 9: Scanner P10 y commit**

Run: el comando del job `no-fail-open-except` de `.github/workflows/policy.yml` de `jax-platform`.
Expected: `passed` (el `except Exception` de `_build_grounding` no es `pass`: loguea con traceback y devuelve una marca que el validador convierte en veredicto propio).

```bash
cd /home/fruiz/jax-platform && git add backend/api/chat.py backend/tests/test_chat_contract_prompt.py backend/tests/test_chat_grounding_wiring.py && git commit -F - <<'EOF'
feat(chat): el contrato admite evidence_pointer y el snapshot viaja al prompt y al validador -- SP3, parte 3

Un objeto por turno, dos consumidores, ninguno divergente. El sufijo dejaba
de prohibir el campo que el mecanismo necesita. Snapshot que falla al
construirse: marca propia, nunca silencio ni snapshot vacio.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011AVPMngV4W1np18dUPa542
EOF
```

---

## Task 7: Pisos de CI medidos, PR de `jax-platform`, verificación en vivo

**Files:**
- Modify: `.github/workflows/policy.yml` (`PISO_PASSED`, `JAX_CI_MIN_PASSED`)
- Modify (solo si hace falta): `backend/requirements.txt`

**Interfaces:** ninguna nueva.

- [ ] **Step 1: Medir el piso CON DB, con el comando del job**

Run: `cd /home/fruiz/jax-platform/backend && python -m pytest -q -rs 2>&1 | tail -3`
Expected: `N passed, 1 skipped` (skip count sin cambios: 1). Anotar N. Antes de SP3 era **340**.

- [ ] **Step 2: Medir el piso SIN DB**

Run: `cd /home/fruiz/jax-platform/backend && JAX_CI_NO_DB=1 python -m pytest -q -rs 2>&1 | tail -3`
Expected: `M passed, 151 skipped` (skip count sin cambios: 151). Anotar M. Antes era **190**. Los tests nuevos que piden `client` cuentan como skip aquí — si el skip count cambió, hay que explicarlo en el comentario del piso, no ajustarlo en silencio.

- [ ] **Step 3: Escribir los pisos con su medición**

En `.github/workflows/policy.yml`, `PISO_PASSED = 340` → `PISO_PASSED = <N>` con comentario:

```yaml
          # Subido de 340 a <N> (2026-09-02, SP3 grounding): tests de
          # tests/test_shadow_validation_grounding.py, tests/test_chat_grounding_wiring.py
          # y tests/test_chat_contract_prompt.py. Medido con la DB de desarrollo
          # antes/despues (340 -> <N>), skip count sin cambios (1).
```

Y `JAX_CI_MIN_PASSED: "190"` → `"<M>"` con comentario análogo (*"Medido con JAX_CI_NO_DB=1 antes/despues (190 -> <M>), skip count sin cambios (151)"* o el número real con su explicación).

- [ ] **Step 4: Commit y push**

```bash
cd /home/fruiz/jax-platform && git add .github/workflows/policy.yml && git commit -F - <<'EOF'
ci: pisos medidos tras SP3 -- con DB 340 -> N, sin DB 190 -> M

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011AVPMngV4W1np18dUPa542
EOF
git push -u origin reformas-fase2-sp3-grounding
```

- [ ] **Step 5: PR de `jax-platform`**

`gh pr create` con título `feat(shadow): SP3 -- grounding por snapshot inyectado` y cuerpo que: enlace el spec (en `jax`), diga que **depende del PR de `jax` mergeado** (el job clona `master` de `jax`), liste la migración (4 columnas), y reproduzca la tabla de §4.1 del spec. Verificar CI sobre el headSha real con `gh api repos/fjruizhn/jax-platform/commits/<sha>/check-runs`. Si el job con DB falla en la migración: la DB del service container es virgen, así que ejercita la vía CREATE, no la ALTER — las dos están escritas en Task 4.

- [ ] **Step 6: Verificación en vivo (spec §9.5) — DESPUÉS del merge y del reinicio del servicio**

Este paso no se hace hasta que ambos PRs estén mergeados y `jax-platform` reiniciado desde `master` (`sudo systemctl restart jax-platform`; verificar con `git -C /home/fruiz/jax-platform rev-parse --abbrev-ref HEAD` que el checkout está en `master`).

1. Anotar el estado previo: `SELECT COUNT(*) FROM shadow_claim_verdicts; SELECT status, COUNT(*) FROM shadow_claim_verdicts GROUP BY status;`
2. **Sonda explícita**, con token real, `POST /api/chat` a `jax_local` y `jekyll`: *"Emitime un claim CAPABILITY_AVAILABLE sobre write_file citando la línea de HECHOS VERIFICADOS que corresponda."*
3. **Pregunta orgánica**, a las 5 facetas: *"¿Qué capabilities de escritura tiene hoy este sistema?"*
4. Medir: `SELECT sm.facet, cv.predicate, cv.status, cv.authority, cv.evidence_pointer FROM shadow_claim_verdicts cv JOIN shadow_messages sm USING (shadow_message_id) WHERE cv.created_at > <hora de inicio>;` y `SELECT grounding_snapshot_sha256, COUNT(*) FROM shadow_messages WHERE queued_at > <hora> GROUP BY 1;`
5. Escribir el resultado en la memoria de cierre **declarando la contaminación** (qué filas son de sonda) y, si el tráfico orgánico sigue en `[]`, escribirlo como **dato, no fallo** (spec §8).

---

## Self-Review

**Spec coverage**
- §2.2 citación falsa → Task 2 test 4, Task 3 test 4, Task 5 `test_forged_citation_is_provenance_mismatch`. ✓
- §3 snapshot solo desde `ctx.ops` → Task 1 `build_snapshot`. `SECTION_PREDICATE` documenta que crece solo con resolvers. ✓ (Nota: la rama `in_catalog` del resolver queda fuera del snapshot; Task 0 registra en DEUDA.md que hoy está muerta.)
- §4.1 orden 0–6 → Task 3 `validate`. Tests 1, 2, 3, 4, 5, 6, 8. ✓
- §4.2 dos estados, sin `UNGROUNDED`, caben en VARCHAR(30) → Task 3 (Literal + test). ✓
- §4.3 authority y veredicto separados → Task 4 columna `authority`, Task 5 la escribe siempre derivada. ✓
- §5.1 hash fuera del prompt; modelo solo cita → Task 1 `render` (test 7b), Task 6 wiring test. ✓
- §5.2 orden por name → Task 1 test 7c. ✓
- §5.3 una normalización → Task 1 `normalize_args`, usada en `build_snapshot` y `accredit` y en `shadow_validation` al armar el `Claim`. ✓
- §5.4 fallo ruidoso + tres estados → Task 1 test 7d (lanza), Task 6 `_build_grounding` (marca), Task 5 `_grounding_columns` ('ERROR'), Task 4 NULL solo legado. ✓
- §5.5 quinto argumento; propio turno → Task 5 (obligatorio, test de firma). ✓
- §6.2 migración → Task 4, dos vías. ✓
- §6.3 contrato admite `evidence_pointer` → Task 6 (regresión de la 576). ✓
- §7 costo → medido en el spec; el plan no lo re-mide. El `render` del plan es la variante D del spec, carácter por carácter salvo el orden de las entradas (por name). ✓
- §8 → Task 7 Step 6.5. ✓
- §9.1 1–8 → Tasks 1–3. §9.1b → Task 2 (puro) + Task 5 (DB, truncado). §9.2 → Task 5. §9.3 → Task 6. §9.4 → Task 3 test 8 + docstring de `governance_context.py`. §9.5 → Task 7. ✓
- Regresión de la 576 → Task 6 Step 1. ✓

**Placeholder scan:** ningún TBD/TODO. Los `<N>`/`<M>` de Task 7 son valores a medir, con el comando que los produce.

**Type consistency:** `Accreditation.outcome` ∈ {ACCREDITED, NO_POINTER, MISMATCH, UNAVAILABLE} en Task 2 y Task 3; `evidence_pointer_raw: object | None` en Task 2, usado con `isinstance(..., str)` en Tasks 3, 5; `run_shadow_validation(conv_uuid, shadow_message_id, facet, contract, grounding)` en Task 5 y Task 6 (llamada y spy con 5 args); `_build_snapshot_or_raise` definida en Task 6 Step 7 y parcheada en Step 5; `governance_context.validation_context()` en Task 4, importada como `_validation_context` en `shadow_validation` y como `_governance_context` en `chat.py`.
