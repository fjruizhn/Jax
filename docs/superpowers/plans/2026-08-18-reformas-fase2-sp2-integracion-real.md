# REFORMAS-v3 Fase 2, Sub-proyecto 2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Conectar el validador de claims de Sub-proyecto 1 a tráfico real de la Mesa web — contrato de salida `{claim/analysis/judgment}` en las facetas, shadow validation (mide, no bloquea), e instrumentación de distribución.

**Architecture:** Un wrapper en `jax-platform/backend/api/chat.py` pide el contrato JSON y lo parsea sincrónicamente (esto es real, no shadow). Una `BackgroundTask` de FastAPI corre después de responder al usuario: valida cada claim contra `policy/governance/validator.py` (importado directo desde el repo `jax` vía `sys.path`, mismo patrón que ya usa `chat.py` para `MemoryDB`) y barre `analysis`/`judgment` con `vocab_sweep.py`, escribiendo a tres tablas nuevas en `jax_memory`. Un footnote sobrio en `jax-platform-frontend` marca contrato degradado.

**Tech Stack:** Python 3.14 (jax-platform/backend), Python 3.12 (jax), FastAPI + aiomysql + pytest-asyncio, React 19 + Vitest + @testing-library/react (jax-platform-frontend).

**Spec:** `docs/superpowers/specs/2026-08-18-reformas-fase2-sp2-integracion-real-design.md` (repo `jax`) — el plan argumenta desde ahí, léanlo ambos.

## Global Constraints

- Esta rama parte de `master` en `jax-platform`, que está 28 commits detrás de `infra/facetas-bloque-d` (MariaDB 12.3, credenciales, `resolve_facet`/`UsageInfo`). Task 3 usa una señal posicional propia (`_canned_reply` flag, seteado en el punto exacto de cada respuesta enlatada) en vez de `UsageInfo`, que no existe en `master`. Evento esperado, sin ETA confirmada: cuando `infra/facetas-bloque-d` mergee, reconciliar reemplazando el flag por la señal `usage is None` que trae esa rama — no antes, no se aceleró el merge de esa rama para este plan.

- Autoridad de todo claim es siempre `"INFERIDO"`, fijada server-side — nunca autodeclarada por el modelo. Ver spec sección 1a. Consecuencia esperada: `shadow_claim_verdicts` va a mostrar 100% `AUTHORITY_INVALID` esta ronda — no es un bug, no "arreglarlo".
- El modelo solo declara `{"predicate": ..., "args": {...}}` por claim — nunca `authority`/`provenance_ref`/`evidence_pointer`/`scope`.
- `degradation_reason` es siempre texto libre (`TEXT`), nunca un enum — las formas de incumplimiento de contrato no están catalogadas todavía.
- `args` en `shadow_claim_verdicts` es `JSON` nativo de MariaDB, nunca `TEXT`.
- Las tablas usan `conv_uuid` (VARCHAR(36), UUID real de `conversations.conversation_uuid`) + `shadow_message_id` (CHAR(36), generado por `chat.py`) — nunca un `message_id` que no existe en el código real (`_memory.save_message()` es fire-and-forget, no devuelve id).
- `hyde` no participa del wrapper — nunca llega a un LLM en el camino de Mesa web (`chat()` lo intercepta antes). El wrapper aplica a `jax_local`, `jekyll`, `hipatia`, `thot`, `ada`, `kimi`.
- El footnote de frontend usa el mismo estilo hardcoded-a-oscuro que sus vecinos en `Message.jsx` — no hay sistema de temas en esta app, no se inventa uno para esto.
- Todo string visible al usuario nuevo va en `es.js`/`en.js` — cero hardcoding de texto.
- Ningún paso de esta lista modifica `jax/muscles/base.py`, `jax/jacobs/executor.py`, ni el reenrutamiento `_HTTP_FACETS`/`_MOTOR_FACETS` — fuera de alcance, ver spec.

---

## Task 1: Extender `load_vocabulary()`/`sweep()` con categoría

**Files:**
- Modify: `policy/governance/loaders.py`
- Modify: `policy/governance/vocab_sweep.py`
- Modify: `tests/test_governance_loaders.py`
- Modify: `tests/test_governance_vocab_sweep.py`

**Interfaces:**
- Consumes: `ClosedVocabulary` actual (`flattened: frozenset[str]`, `config_paths: frozenset[str]`), `sweep(text: str, vocabulary: frozenset[str]) -> list[str]` actuales.
- Produces: `ClosedVocabulary.term_categories: dict[str, frozenset[str]]` (término → categorías de origen, ej. `"ada"` → `{"facets_las_manos", "facets_jax", "motors"}`). `sweep(text: str, term_categories: dict[str, frozenset[str]]) -> list[tuple[str, frozenset[str]]]` — firma nueva, reemplaza la anterior (Task 5 la consume así).

- [ ] **Step 1: Escribir los tests que fallan — `load_vocabulary()` con categoría**

Editar `tests/test_governance_loaders.py`, agregar después de `test_load_vocabulary_flattens_categories_and_keeps_config_paths_separate`:

```python
def test_load_vocabulary_tracks_term_categories():
    vocab = loaders.load_vocabulary()
    assert "capabilities" in vocab.term_categories["code_swarm"]
    assert "ops" in vocab.term_categories["ssh_exec"]


def test_load_vocabulary_term_in_multiple_categories():
    vocab = loaders.load_vocabulary()
    ada_categories = vocab.term_categories["ada"]
    assert "facets_las_manos" in ada_categories
    assert "facets_jax" in ada_categories
    assert "motors" in ada_categories


def test_load_vocabulary_config_paths_not_in_term_categories():
    vocab = loaders.load_vocabulary()
    assert "policy/" not in vocab.term_categories
```

- [ ] **Step 2: Correr los tests, verificar que fallan**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_loaders.py -k term_categories -v`
Expected: FAIL — `AttributeError: 'ClosedVocabulary' object has no attribute 'term_categories'`

- [ ] **Step 3: Implementar `term_categories` en `loaders.py`**

En `policy/governance/loaders.py`, modificar el dataclass y `load_vocabulary()`:

```python
@dataclass(frozen=True)
class ClosedVocabulary:
    flattened: frozenset[str]
    config_paths: frozenset[str]
    term_categories: dict[str, frozenset[str]]


def load_vocabulary() -> ClosedVocabulary:
    data = yaml.safe_load(VOCABULARY_FILE.read_text(encoding="utf-8"))
    flattened: set[str] = set()
    term_categories: dict[str, set[str]] = {}
    for key, value in data.items():
        if key == "config_paths":
            continue
        if isinstance(value, dict):
            terms = value.keys()
        elif isinstance(value, list):
            terms = value
        else:
            raise RuntimeError(
                f"closed_vocabulary.yaml: la categoría '{key}' tiene un "
                f"valor de tipo {type(value).__name__}, se esperaba dict "
                "o list — el barrido léxico no puede aplanarla sin "
                "silenciar la categoría."
            )
        for term in terms:
            flattened.add(term)
            term_categories.setdefault(term, set()).add(key)
    config_paths = frozenset(data.get("config_paths") or [])
    return ClosedVocabulary(
        flattened=frozenset(flattened),
        config_paths=config_paths,
        term_categories={t: frozenset(cats) for t, cats in term_categories.items()},
    )
```

- [ ] **Step 4: Correr los tests, verificar que pasan**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_loaders.py -v`
Expected: PASS (todos, incluidos los 3 nuevos y los ya existentes — `test_load_vocabulary_includes_list_shaped_categories` y `test_load_vocabulary_raises_for_malformed_category` no deberían necesitar cambios, siguen usando solo `.flattened`)

- [ ] **Step 5: Escribir los tests que fallan — `sweep()` con categoría**

Reemplazar el contenido de `tests/test_governance_vocab_sweep.py`:

```python
"""
Test de policy/governance/vocab_sweep.py — barrido léxico puro, sin I/O.
"""
from __future__ import annotations

import sys
from pathlib import Path

GOVERNANCE = Path(__file__).resolve().parent.parent / "policy" / "governance"
sys.path.insert(0, str(GOVERNANCE))

import vocab_sweep  # noqa: E402

TERM_CATEGORIES = {
    "trae a hyde": frozenset({"commands"}),
    "hyde": frozenset({"facets_las_manos", "facets_jax"}),
    "code_swarm": frozenset({"capabilities"}),
    "ada": frozenset({"facets_las_manos", "facets_jax", "motors"}),
}


def test_sweep_no_matches_returns_empty():
    assert vocab_sweep.sweep("un texto sin nada prohibido", TERM_CATEGORIES) == []


def test_sweep_finds_known_term_with_categories():
    result = vocab_sweep.sweep("invocá a hyde ahora", TERM_CATEGORIES)
    assert result == [("hyde", frozenset({"facets_las_manos", "facets_jax"}))]


def test_sweep_finds_multiple_terms_sorted():
    result = vocab_sweep.sweep("code_swarm y también trae a hyde", TERM_CATEGORIES)
    terms = [t for t, _ in result]
    assert terms == sorted(terms)
    assert ("code_swarm", frozenset({"capabilities"})) in result
    assert ("trae a hyde", frozenset({"commands"})) in result


def test_sweep_term_with_multiple_categories():
    result = vocab_sweep.sweep("dale, trae a ada", TERM_CATEGORIES)
    assert result == [("ada", frozenset({"facets_las_manos", "facets_jax", "motors"}))]


def test_sweep_no_false_positive_on_substring_inside_word():
    tc = {"ada": frozenset({"motors"})}
    assert vocab_sweep.sweep("no hay nada que hacer, cada faceta", tc) == []


def test_sweep_case_insensitive_match():
    tc = {"hyde": frozenset({"facets_jax"})}
    result = vocab_sweep.sweep("invocá a HYDE ahora", tc)
    assert result == [("hyde", frozenset({"facets_jax"}))]
```

- [ ] **Step 6: Correr los tests, verificar que fallan**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_vocab_sweep.py -v`
Expected: FAIL — `TypeError` (el `sweep()` actual espera `frozenset`, recibe `dict`, o compara `list[str]` contra tuplas)

- [ ] **Step 7: Implementar `sweep()` con categoría**

Reemplazar `policy/governance/vocab_sweep.py`:

```python
"""
policy/governance — Barrido léxico contra el vocabulario cerrado
(REFORMAS-v3.md §3.1.5).

Puro: recibe el vocabulario ya cargado y su mapeo de categorías
(loaders.load_vocabulary().term_categories), nunca abre un archivo.
Decidir qué hacer con los términos encontrados es responsabilidad del
llamador — sub-proyecto 2.
"""
from __future__ import annotations

import re


def sweep(
    text: str, term_categories: dict[str, frozenset[str]]
) -> list[tuple[str, frozenset[str]]]:
    if not term_categories:
        return []
    terms = sorted((t for t in term_categories if t), key=len, reverse=True)
    if not terms:
        return []
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(t) for t in terms) + r")(?!\w)",
        re.IGNORECASE,
    )
    found = {m.group(0).lower() for m in pattern.finditer(text)}
    lowered_to_original = {t.lower(): t for t in term_categories if t}
    matched = [lowered_to_original[f] for f in found if f in lowered_to_original]
    return sorted((t, term_categories[t]) for t in matched)
```

- [ ] **Step 8: Correr los tests, verificar que pasan**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_vocab_sweep.py tests/test_governance_loaders.py -v`
Expected: PASS (todos)

- [ ] **Step 9: Correr la suite completa de governance, verificar que nada se rompió**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_claims.py tests/test_governance_loaders.py tests/test_governance_renderer.py tests/test_governance_validator.py tests/test_governance_vocab_sweep.py tests/test_template_hash.py -v`
Expected: PASS (40/40 — 34 previos + 6 nuevos de este task)

- [ ] **Step 10: Commit**

```bash
cd ~/jax
git add policy/governance/loaders.py policy/governance/vocab_sweep.py tests/test_governance_loaders.py tests/test_governance_vocab_sweep.py
git commit -m "policy(governance): load_vocabulary()/sweep() preservan categoría por término (SP2 Task 1)"
```

---

## Task 2: Fix issue #6 — `IsADirectoryError` sin capturar

**Files:**
- Modify: `policy/governance/validator.py`
- Modify: `tests/test_governance_validator.py`

**Interfaces:**
- Consumes: `_resolve_file_exists(claim: claims.Claim, ctx: ValidationContext) -> Verdict` actual, `_path_allowed()`, `_normalize_path()` sin cambios.
- Produces: mismo `_resolve_file_exists`, ahora sin excepción sin capturar cuando `path` coincide con una entrada de directorio de la allowlist real (`policy/` está en `config_paths` de `closed_vocabulary.yaml` hoy).

- [ ] **Step 1: Escribir el test que falla — reproducir el crash real**

Editar `tests/test_governance_validator.py`, agregar en la sección de `FILE_EXISTS` (buscar el bloque de tests de ese predicado, agregar al final):

```python
def test_file_exists_directory_entry_returns_verdict_not_exception(governance_ctx):
    # "policy/" está en config_paths de closed_vocabulary.yaml — un claim
    # con path="policy" coincide con esa entrada de directorio, pasa
    # _path_allowed(), .exists() es True (el directorio existe), y
    # .read_bytes() explota con IsADirectoryError si no se captura.
    claim = claims.Claim(
        predicate="FILE_EXISTS",
        args={"path": "policy", "hash": "0" * 64},
        authority="OBSERVADO",
        provenance_ref="test",
        evidence_pointer="test",
        scope="test",
    )
    verdict = validator.validate(claim, {"FILE_EXISTS": FILE_EXISTS_SPEC}, governance_ctx)
    assert verdict.status == "FACT_MISMATCH"
    assert "directorio" in verdict.detail.lower()
```

Si el archivo no tiene un fixture `governance_ctx`, revisar cómo los tests de `FILE_EXISTS` existentes construyen `ValidationContext` (buscar `load_validation_context` o construcción manual de `ValidationContext(...)` en el archivo) y usar exactamente ese mismo patrón en vez de inventar uno nuevo.

- [ ] **Step 2: Correr el test, verificar que falla con la excepción real**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_validator.py -k directory_entry -v`
Expected: FAIL — `IsADirectoryError: [Errno 21] Is a directory: '.../policy'` (no un `AssertionError` — la excepción se propaga sin capturar, confirmando el bug real antes de arreglarlo)

- [ ] **Step 3: Implementar el fix en `_resolve_file_exists`**

En `policy/governance/validator.py`, modificar `_resolve_file_exists` (agregar el chequeo `is_file()` antes de `read_bytes()`):

```python
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
    if not candidate.is_file():
        return Verdict(
            status="FACT_MISMATCH",
            predicate="FILE_EXISTS",
            detail=f"'{path}' existe pero es un directorio, no un archivo.",
        )

    actual_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        return Verdict(
            status="FACT_MISMATCH",
            predicate="FILE_EXISTS",
            detail=f"'{path}' existe pero su hash no coincide (esperado {expected_hash}).",
        )
    return Verdict(
        status="VALID", predicate="FILE_EXISTS", detail=f"'{path}' existe con hash verificado."
    )
```

- [ ] **Step 4: Correr el test, verificar que pasa**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_validator.py -k directory_entry -v`
Expected: PASS

- [ ] **Step 5: Correr toda la suite de validator, verificar que nada se rompió**

Run: `/home/fruiz/jax/.venv/bin/python -m pytest tests/test_governance_validator.py -v`
Expected: PASS (todos)

- [ ] **Step 6: Commit**

```bash
cd ~/jax
git add policy/governance/validator.py tests/test_governance_validator.py
git commit -m "policy(governance): _resolve_file_exists no explota con IsADirectoryError, devuelve FACT_MISMATCH (fixes #6)"
```

- [ ] **Step 7: Cerrar el issue #6 a mano**

No es un paso de código. Ir a https://github.com/fjruizhn/Jax/issues/6, comentar referenciando el commit del Step 6, y cerrar el issue manualmente (gh CLI se cuelga en hall9000 — ver memoria de proyecto `gh-cli-cuelga-hall9000`).

---

## Task 3: Wrapper de contrato en `chat.py`

**Files:**
- Modify: `/home/fruiz/jax-platform/backend/api/chat.py`
- Test: `/home/fruiz/jax-platform/backend/tests/test_chat_contract_wrapper.py` (nuevo)

**Interfaces:**
- Consumes: `_invoke_facet(facet, config, user_id, message, semantic_context) -> tuple[str, UsageInfo | None]` sin cambios en su firma — se modifica *dónde* se arma `system_prompt`. `usage is None` ya distingue hoy "respuesta enlatada" (facet no disponible, pregunta de identidad de modelo) de "llamada real al LLM" — se reutiliza esa señal, no se agrega una nueva.
- Produces: `_parse_contract_response(raw_text: str) -> ContractResult` (nuevo, puro, testeable sin red). `ChatResponse` gana un campo `contract_degraded: bool = False`. `chat()` gana un `shadow_message_id: str` por respuesta (usado por Task 5, no persistido todavía en este task).

- [ ] **Step 1: Escribir los tests que fallan — `_parse_contract_response`**

Crear `/home/fruiz/jax-platform/backend/tests/test_chat_contract_wrapper.py`:

```python
"""
Wrapper de contrato {claim/analysis/judgment} en chat.py (REFORMAS-v3
Fase 2 Sub-proyecto 2). _parse_contract_response es puro — sin red, sin
DB — se testea con texto crudo simulando lo que devolvería cada
faceta, incluido el truncamiento real de Kimi (488 bytes).
"""
from api.chat import _parse_contract_response


def test_parse_contract_valid_json_with_claims():
    raw = '{"claim": [{"predicate": "CAPABILITY_AVAILABLE", "args": {"name": "code_swarm"}}], "analysis": "revisé el catálogo", "judgment": "está disponible"}'
    result = _parse_contract_response(raw)
    assert result.contract_parsed is True
    assert result.claims == [{"predicate": "CAPABILITY_AVAILABLE", "args": {"name": "code_swarm"}}]
    assert result.analysis == "revisé el catálogo"
    assert result.judgment == "está disponible"
    assert result.degradation_reason is None


def test_parse_contract_valid_json_no_claims():
    raw = '{"claim": [], "analysis": "no hay nada que afirmar acá", "judgment": null}'
    result = _parse_contract_response(raw)
    assert result.contract_parsed is True
    assert result.claims == []
    assert result.judgment is None


def test_parse_contract_strips_markdown_fence():
    raw = '```json\n{"claim": [], "analysis": "ok", "judgment": null}\n```'
    result = _parse_contract_response(raw)
    assert result.contract_parsed is True
    assert result.analysis == "ok"


def test_parse_contract_truncated_json_degrades():
    # Simula el truncamiento real de Kimi a 488 bytes: JSON cortado a
    # mitad de un claim.
    raw = '{"claim": [{"predicate": "CAPABILITY_AVAI'
    result = _parse_contract_response(raw)
    assert result.contract_parsed is False
    assert result.claims == []
    assert result.analysis == raw
    assert result.judgment is None
    assert result.degradation_reason is not None
    assert result.raw_text == raw


def test_parse_contract_not_a_json_object_degrades():
    raw = '["no", "es", "un", "objeto"]'
    result = _parse_contract_response(raw)
    assert result.contract_parsed is False
    assert "objeto" in result.degradation_reason


def test_parse_contract_missing_analysis_key_degrades():
    raw = '{"claim": []}'
    result = _parse_contract_response(raw)
    assert result.contract_parsed is False
    assert "analysis" in result.degradation_reason


def test_parse_contract_malformed_claim_entry_degrades():
    raw = '{"claim": [{"predicate": "X"}], "analysis": "ok"}'
    result = _parse_contract_response(raw)
    assert result.contract_parsed is False


def test_parse_contract_plain_text_not_json_degrades():
    raw = "esto no es json en absoluto, es texto libre normal"
    result = _parse_contract_response(raw)
    assert result.contract_parsed is False
    assert result.analysis == raw
```

- [ ] **Step 2: Correr los tests, verificar que fallan**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/pytest tests/test_chat_contract_wrapper.py -v`
Expected: FAIL — `ImportError: cannot import name '_parse_contract_response' from 'api.chat'`

- [ ] **Step 3: Implementar `ContractResult` y `_parse_contract_response` en `chat.py`**

En `/home/fruiz/jax-platform/backend/api/chat.py`, agregar cerca de `UsageInfo` (después de su definición, antes de `_build_messages`):

```python
import json


class ContractResult(NamedTuple):
    contract_parsed: bool
    claims: list[dict]
    analysis: str
    judgment: str | None
    degradation_reason: str | None
    raw_text: str


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _degraded(raw_text: str, reason: str) -> ContractResult:
    return ContractResult(
        contract_parsed=False, claims=[], analysis=raw_text, judgment=None,
        degradation_reason=reason, raw_text=raw_text,
    )


def _parse_contract_response(raw_text: str) -> ContractResult:
    candidate = _strip_markdown_fence(raw_text)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        return _degraded(raw_text, f"JSON no parsea: {e}")

    if not isinstance(data, dict):
        return _degraded(raw_text, f"JSON parseado no es un objeto (es {type(data).__name__})")

    if "analysis" not in data:
        return _degraded(raw_text, "falta la clave 'analysis' en el JSON")

    raw_claims = data.get("claim", [])
    if not isinstance(raw_claims, list):
        return _degraded(raw_text, f"'claim' no es una lista (es {type(raw_claims).__name__})")

    parsed_claims = []
    for item in raw_claims:
        if not isinstance(item, dict) or "predicate" not in item or "args" not in item:
            return _degraded(raw_text, f"claim mal formado: {item!r}")
        if not isinstance(item["predicate"], str) or not isinstance(item["args"], dict):
            return _degraded(raw_text, f"claim con tipos inválidos: {item!r}")
        parsed_claims.append({"predicate": item["predicate"], "args": item["args"]})

    analysis = data["analysis"]
    if not isinstance(analysis, str):
        return _degraded(raw_text, f"'analysis' no es string (es {type(analysis).__name__})")

    judgment = data.get("judgment")
    if judgment is not None and not isinstance(judgment, str):
        return _degraded(raw_text, f"'judgment' no es string ni null (es {type(judgment).__name__})")

    return ContractResult(
        contract_parsed=True, claims=parsed_claims, analysis=analysis,
        judgment=judgment, degradation_reason=None, raw_text=raw_text,
    )
```

- [ ] **Step 4: Correr los tests, verificar que pasan**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/pytest tests/test_chat_contract_wrapper.py -v`
Expected: PASS (8/8)

- [ ] **Step 5: Escribir el test que falla — el prompt pide el contrato a las 6 facetas, no a hyde**

Agregar a `tests/test_chat_contract_wrapper.py`:

```python
def test_contract_suffix_appended_to_system_prompt():
    from api.chat import _CONTRACT_PROMPT_SUFFIX
    assert "claim" in _CONTRACT_PROMPT_SUFFIX
    assert "analysis" in _CONTRACT_PROMPT_SUFFIX
    assert "judgment" in _CONTRACT_PROMPT_SUFFIX
```

- [ ] **Step 6: Correr, verificar que falla**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/pytest tests/test_chat_contract_wrapper.py -k suffix -v`
Expected: FAIL — `ImportError`

- [ ] **Step 7: Agregar `_CONTRACT_PROMPT_SUFFIX` y conectarlo en `_invoke_facet`**

En `chat.py`, agregar la constante (cerca de `ContractResult`):

```python
_CONTRACT_PROMPT_SUFFIX = """

FORMATO DE RESPUESTA OBLIGATORIO — respondé ÚNICAMENTE con un objeto JSON, sin texto antes ni después, sin fences de markdown:

{"claim": [{"predicate": "NOMBRE", "args": {"clave": "valor"}}], "analysis": "tu razonamiento en texto libre", "judgment": "tu conclusión, o null si no aplica"}

- "claim": lista de afirmaciones verificables (puede ir vacía: []). Cada una es {"predicate": "...", "args": {...}} — SOLO estos dos campos, nada más.
- "analysis": tu análisis en texto libre. Obligatorio, aunque sea corto.
- "judgment": tu conclusión o recomendación, o null si no aplica.

No incluyas ningún otro campo. No expliques el formato, solo respondé el JSON."""
```

Modificar `_invoke_facet` — una sola línea cambia, justo después de leer `system_prompt`:

```python
    personality = config["personalities"].get(facet, config["personalities"]["jax_local"])
    system_prompt = personality.get("system_prompt", "Sos JAX.") + _CONTRACT_PROMPT_SUFFIX
```

(El resto de `_invoke_facet` no cambia — el `ident` de `jax_local` se sigue concatenando después de este `system_prompt` ya extendido, en la rama `if facet == "jax_local"` existente.)

- [ ] **Step 8: Correr, verificar que pasa**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/pytest tests/test_chat_contract_wrapper.py -v`
Expected: PASS (9/9)

- [ ] **Step 9: Escribir el test que falla — `ChatResponse.contract_degraded` y construcción de `response`**

Agregar a `tests/test_chat_contract_wrapper.py`:

```python
def test_build_display_response_valid_contract_no_judgment():
    from api.chat import _build_display_response
    from api.chat import ContractResult
    contract = ContractResult(
        contract_parsed=True, claims=[], analysis="mi análisis",
        judgment=None, degradation_reason=None, raw_text="...",
    )
    text, degraded = _build_display_response(contract)
    assert text == "mi análisis"
    assert degraded is False


def test_build_display_response_valid_contract_with_judgment():
    from api.chat import _build_display_response
    from api.chat import ContractResult
    contract = ContractResult(
        contract_parsed=True, claims=[], analysis="mi análisis",
        judgment="mi conclusión", degradation_reason=None, raw_text="...",
    )
    text, degraded = _build_display_response(contract)
    assert "mi análisis" in text
    assert "mi conclusión" in text
    assert degraded is False


def test_build_display_response_degraded_shows_raw_text():
    from api.chat import _build_display_response
    from api.chat import ContractResult
    contract = ContractResult(
        contract_parsed=False, claims=[], analysis="texto crudo truncado",
        judgment=None, degradation_reason="JSON no parsea", raw_text="texto crudo truncado",
    )
    text, degraded = _build_display_response(contract)
    assert text == "texto crudo truncado"
    assert degraded is True
```

- [ ] **Step 10: Correr, verificar que falla**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/pytest tests/test_chat_contract_wrapper.py -k build_display -v`
Expected: FAIL — `ImportError`

- [ ] **Step 11: Implementar `_build_display_response`**

En `chat.py`, agregar después de `_parse_contract_response`:

```python
def _build_display_response(contract: ContractResult) -> tuple[str, bool]:
    if not contract.contract_parsed:
        return contract.raw_text, True
    if contract.judgment:
        return f"{contract.analysis}\n\n**{contract.judgment}**", False
    return contract.analysis, False
```

- [ ] **Step 12: Correr, verificar que pasa**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/pytest tests/test_chat_contract_wrapper.py -v`
Expected: PASS (12/12)

- [ ] **Step 13: Conectar todo en `ChatResponse` y en el handler `chat()`**

En `chat.py`, modificar `ChatResponse`:

```python
class ChatResponse(BaseModel):
    facet: str
    response: str
    timestamp: str
    contract_degraded: bool = False
```

Agregar `import uuid` al tope del archivo si no está ya importado (`chat.py` ya usa `uuid` indirectamente vía `jax.memory.db` — confirmar con `grep -n "^import uuid" api/chat.py`; si no está, agregarlo junto a los demás imports estándar).

Modificar el handler `chat()` donde llama a `_invoke_facet` y arma la respuesta (reemplazar el bloque que va desde `response_text, usage = await _invoke_facet(...)` hasta el `return ChatResponse(...)` final, manteniendo el manejo de excepciones `httpx.HTTPStatusError`/`Exception` que ya existe sin tocarlo):

```python
    try:
        response_text, usage = await _invoke_facet(facet, config, user_id, req.message, semantic_context)
    except httpx.HTTPStatusError as e:
        detail = f"Error HTTP {e.response.status_code} en {facet}: {e.response.text[:200]}"
        await engine_state.set_facet_status(facet, "error", tenant_id, user_id, detail[:100])
        await engine_state.set_facet_status(facet, "idle", tenant_id, user_id)
        raise HTTPException(status_code=502, detail=detail)
    except Exception as e:
        detail = f"Error en {facet}: {str(e)[:200]}"
        await engine_state.set_facet_status(facet, "error", tenant_id, user_id, str(e)[:100])
        await engine_state.set_facet_status(facet, "idle", tenant_id, user_id)
        raise HTTPException(status_code=502, detail=detail)

    shadow_message_id = str(uuid.uuid4())
    contract = _parse_contract_response(response_text) if usage is not None else None
    if contract is not None:
        display_text, contract_degraded = _build_display_response(contract)
    else:
        display_text, contract_degraded = response_text, False

    await _fire_completed(facet, tenant_id, user_id, display_text)
    if conv_uuid:
        _memory.save_message(conv_uuid, "assistant", display_text, facet=facet)
    _update_history(user_id, req.message, display_text)
    await engine_state.set_facet_status(facet, "idle", tenant_id, user_id)

    return ChatResponse(
        facet=facet, response=display_text, timestamp=timestamp,
        contract_degraded=contract_degraded,
    )
```

**Nota para quien implemente:** el bloque de arriba es una guía de la forma final, no un reemplazo textual exacto — `chat()` ya tiene lógica existente (`_fire_completed`, `_memory.save_message`, `_update_history`, `engine_state.set_facet_status`) en algún orden entre el `try/except` y el `return` actual. Leer el `chat()` real (`api/chat.py`, función completa) antes de editar, y adaptar el bloque de arriba a lo que ya existe ahí — el objetivo es solo insertar el parseo de contrato y el nuevo `return`, no reordenar lo que ya funciona. `shadow_message_id` se calcula acá pero no se usa todavía en este task — lo consume Task 5 (queda como variable local sin persistir por ahora; no fallar si un linter marca "unused variable", eso se resuelve en Task 5).

- [ ] **Step 14: Escribir un test de integración del endpoint con contrato válido**

Agregar a `tests/test_chat_contract_wrapper.py` (usando el mismo patrón `_FakePostClient`/`http_client._client` que `test_chat_usage_capture.py`):

```python
import http_client
from unittest.mock import AsyncMock, patch


class _FakeResponse:
    def __init__(self, json_data):
        self._json_data = json_data

    def json(self):
        return self._json_data

    def raise_for_status(self):
        pass


class _FakePostClient:
    def __init__(self, response):
        self._response = response

    async def post(self, url, **kwargs):
        return self._response


def test_chat_endpoint_marks_contract_degraded_on_truncated_json(client):
    from auth.jwt import create_access_token
    token = create_access_token("test-contract-user", "test-contract-tenant", "operator")
    fake = _FakePostClient(_FakeResponse({
        "choices": [{"message": {"content": '{"claim": [{"predicate": "CAPABILITY_AVAI'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }))
    original = http_client._client
    http_client._client = fake
    try:
        resp = client.post(
            "/api/chat",
            json={"message": "hola", "facet": "jekyll"},
            headers={"Authorization": f"Bearer {token}"},
        )
    finally:
        http_client._client = original
    assert resp.status_code == 200
    body = resp.json()
    assert body["contract_degraded"] is True
    assert body["response"].startswith('{"claim"')
```

**Nota:** si `jekyll` no resuelve a `http_openai_compat` en el `config.toml` real de este entorno, o si `resolve_facet()` requiere un binding activo que no existe en la DB de test, ajustar la faceta usada en el test o consultar `test_chat_usage_capture.py`/`test_facet_model_wiring.py` para el patrón exacto de cómo esos tests garantizan un binding resoluble en `jax_memory_test`.

- [ ] **Step 15: Correr toda la suite de este test file, verificar que pasa**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/pytest tests/test_chat_contract_wrapper.py -v`
Expected: PASS

- [ ] **Step 16: Correr toda la suite de `chat.py` existente, verificar que nada se rompió**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/pytest tests/test_chat_config_cache.py tests/test_chat_conv_uuid_eviction.py tests/test_chat_conversation_eviction.py tests/test_chat_http_pooling.py tests/test_chat_resolved_version_capture.py tests/test_chat_usage_capture.py -v`
Expected: PASS (todos — este task no debería haber cambiado el comportamiento de nada que estos tests cubren)

- [ ] **Step 17: Commit**

```bash
cd ~/jax-platform
git add backend/api/chat.py backend/tests/test_chat_contract_wrapper.py
git commit -m "feat(chat): wrapper de contrato {claim/analysis/judgment}, parseo sincrónico con degradación auditada (SP2 Task 3)"
```

---

## Task 4: Migración de las tres tablas en `jax_memory`

**Files:**
- Modify: `/home/fruiz/jax-platform/backend/db/migrations.py`
- Test: `/home/fruiz/jax-platform/backend/tests/test_shadow_migrations.py` (nuevo)

**Interfaces:**
- Consumes: `_TABLES` list, `_table_exists(cur, table_name)`, `run_migrations()` — patrón existente de `migrations.py`, sin cambios de forma.
- Produces: tablas `shadow_messages`, `shadow_claim_verdicts`, `shadow_vocab_hits` en `jax_memory` (y `jax_memory_test`). Task 5 las escribe.

- [ ] **Step 1: Escribir el test que falla — las tablas no existen todavía**

Crear `/home/fruiz/jax-platform/backend/tests/test_shadow_migrations.py`:

```python
"""Migración de las tablas de shadow validation (REFORMAS-v3 Fase 2
Sub-proyecto 2). Corre contra jax_memory_test (ver conftest.py)."""


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


def test_shadow_messages_table_has_expected_columns(client):
    cols = client.portal.call(_get_columns, "shadow_messages")
    expected = {
        "conv_uuid", "shadow_message_id", "facet", "contract_parsed",
        "degradation_reason", "has_claim", "has_analysis", "has_judgment",
        "queued_at", "validated_at",
    }
    assert expected.issubset(cols)


def test_shadow_claim_verdicts_table_has_expected_columns(client):
    cols = client.portal.call(_get_columns, "shadow_claim_verdicts")
    expected = {"conv_uuid", "shadow_message_id", "predicate", "status", "detail", "args"}
    assert expected.issubset(cols)


def test_shadow_vocab_hits_table_has_expected_columns(client):
    cols = client.portal.call(_get_columns, "shadow_vocab_hits")
    expected = {"conv_uuid", "shadow_message_id", "channel", "term", "category"}
    assert expected.issubset(cols)


async def _column_type(table_name, column_name):
    from db.connection import get_pool
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT DATA_TYPE FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s",
                (table_name, column_name),
            )
            row = await cur.fetchone()
            return row[0] if row else None


def test_degradation_reason_is_text_not_varchar(client):
    dtype = client.portal.call(_column_type, "shadow_messages", "degradation_reason")
    assert dtype == "text"


def test_args_is_json_native_not_text(client):
    dtype = client.portal.call(_column_type, "shadow_claim_verdicts", "args")
    assert dtype == "json"
```

- [ ] **Step 2: Correr los tests, verificar que fallan**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/pytest tests/test_shadow_migrations.py -v`
Expected: FAIL — columnas vacías o tabla inexistente (`expected.issubset(cols)` falla contra `set()`)

- [ ] **Step 3: Agregar las tres tablas a `migrations.py`**

En `/home/fruiz/jax-platform/backend/db/migrations.py`, agregar cerca de las otras `CREATE_*` (después de `CREATE_MODEL_BINDING_PROPOSAL`, antes de `_TABLES = [...]`):

```python
CREATE_SHADOW_MESSAGES = """
CREATE TABLE IF NOT EXISTS shadow_messages (
  id INT AUTO_INCREMENT PRIMARY KEY,
  conv_uuid VARCHAR(36) NOT NULL,
  shadow_message_id CHAR(36) NOT NULL UNIQUE,
  facet VARCHAR(30) NOT NULL,
  contract_parsed BOOLEAN DEFAULT NULL,
  degradation_reason TEXT,
  has_claim BOOLEAN DEFAULT NULL,
  has_analysis BOOLEAN DEFAULT NULL,
  has_judgment BOOLEAN DEFAULT NULL,
  queued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  validated_at TIMESTAMP NULL DEFAULT NULL,
  INDEX idx_shadow_messages_facet (facet),
  INDEX idx_shadow_messages_conv_uuid (conv_uuid)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

CREATE_SHADOW_CLAIM_VERDICTS = """
CREATE TABLE IF NOT EXISTS shadow_claim_verdicts (
  id INT AUTO_INCREMENT PRIMARY KEY,
  conv_uuid VARCHAR(36) NOT NULL,
  shadow_message_id CHAR(36) NOT NULL,
  predicate VARCHAR(50) NOT NULL,
  status VARCHAR(30) NOT NULL,
  detail TEXT,
  args JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_shadow_claims_conv_uuid (conv_uuid),
  INDEX idx_shadow_claims_shadow_message_id (shadow_message_id),
  INDEX idx_shadow_claims_predicate (predicate)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

CREATE_SHADOW_VOCAB_HITS = """
CREATE TABLE IF NOT EXISTS shadow_vocab_hits (
  id INT AUTO_INCREMENT PRIMARY KEY,
  conv_uuid VARCHAR(36) NOT NULL,
  shadow_message_id CHAR(36) NOT NULL,
  channel VARCHAR(20) NOT NULL,
  term VARCHAR(100) NOT NULL,
  category VARCHAR(50) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_shadow_vocab_conv_uuid (conv_uuid),
  INDEX idx_shadow_vocab_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""
```

Y agregar las tres entradas a `_TABLES` (al final de la lista existente):

```python
    ("shadow_messages", CREATE_SHADOW_MESSAGES),
    ("shadow_claim_verdicts", CREATE_SHADOW_CLAIM_VERDICTS),
    ("shadow_vocab_hits", CREATE_SHADOW_VOCAB_HITS),
]
```

(Reemplazar el `]` que cierra `_TABLES` hoy por las tres líneas de arriba seguidas del `]`.)

- [ ] **Step 4: Correr `run_migrations()` contra la DB de test**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/python -c "
import asyncio
from db.migrations import run_migrations
asyncio.run(run_migrations())
"`
Expected: sin errores (crea las tres tablas nuevas en `jax_memory_test`, según `JAX_DB_NAME` de `conftest.py`)

- [ ] **Step 5: Correr los tests, verificar que pasan**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/pytest tests/test_shadow_migrations.py -v`
Expected: PASS (5/5)

- [ ] **Step 6: Confirmar que `run_migrations()` se llama en el startup real de la app**

Run: `grep -n "run_migrations" /home/fruiz/jax-platform/backend/main.py`
Expected: al menos una línea — si `run_migrations()` ya se invoca en el startup de `main.py` (probable, dado que las tablas existentes se crean así hoy), no hay nada más que hacer. Si no aparece, agregar la llamada al bloque de startup de `main.py` siguiendo el mismo patrón que usan las tablas existentes.

- [ ] **Step 7: Commit**

```bash
cd ~/jax-platform
git add backend/db/migrations.py backend/tests/test_shadow_migrations.py
git commit -m "feat(db): tablas shadow_messages/shadow_claim_verdicts/shadow_vocab_hits en jax_memory (SP2 Task 4)"
```

---

## Task 5: `BackgroundTask` de validación shadow

**Files:**
- Create: `/home/fruiz/jax-platform/backend/shadow_validation.py`
- Modify: `/home/fruiz/jax-platform/backend/api/chat.py`
- Test: `/home/fruiz/jax-platform/backend/tests/test_shadow_validation.py` (nuevo)

**Interfaces:**
- Consumes: `ContractResult` de Task 3 (`claims`, `analysis`, `judgment`, `contract_parsed`, `degradation_reason`, `raw_text`). `policy.governance.validator.validate()`, `policy.governance.validator.load_validation_context()`, `policy.governance.claims.Claim`, `policy.governance.loaders.load_predicates()`, `policy.governance.loaders.load_vocabulary()`, `policy.governance.vocab_sweep.sweep()` — todos importados directo desde `~/jax` vía `sys.path`, mismo patrón que `chat.py` ya usa para `jax.memory.db.MemoryDB`. Tablas de Task 4.
- Produces: `run_shadow_validation(conv_uuid: str | None, shadow_message_id: str, facet: str, contract: ContractResult | None) -> None` (async, sin retorno — pensada para `BackgroundTasks.add_task()`). `chat()` la encola.

- [ ] **Step 1: Verificar el import cruzado real antes de escribir código sobre un supuesto**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/python -c "
import sys, os
sys.path.insert(0, os.path.expanduser('~/jax'))
sys.path.insert(0, os.path.expanduser('~/jax/policy/governance'))
import validator, vocab_sweep, loaders, claims
print('OK')
"`
Expected: `OK` — si falla, el problema es el punto de partida de este task, no seguir hasta resolverlo (probablemente falta una dependencia en `jax-platform/backend/.venv`, verificar con `.venv/bin/pip list | grep -i "pyyaml\|pydantic"`).

- [ ] **Step 2: Escribir el test que falla — inserta `shadow_messages` al encolar**

Crear `/home/fruiz/jax-platform/backend/tests/test_shadow_validation.py`:

```python
"""BackgroundTask de shadow validation (REFORMAS-v3 Fase 2 Sub-proyecto 2).
Corre contra jax_memory_test. La validación de claims/vocab importa
policy/governance/ directo desde ~/jax (sys.path) — sin red, sin mocks
de HTTP, es una llamada Python normal."""
import uuid

from api.chat import ContractResult


async def _fetch_shadow_message(shadow_message_id):
    from db.connection import get_pool
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT conv_uuid, facet, contract_parsed, degradation_reason, "
                "has_claim, has_analysis, has_judgment, validated_at "
                "FROM shadow_messages WHERE shadow_message_id = %s",
                (shadow_message_id,),
            )
            return await cur.fetchone()


async def _fetch_claim_verdicts(shadow_message_id):
    from db.connection import get_pool
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT predicate, status, args FROM shadow_claim_verdicts "
                "WHERE shadow_message_id = %s",
                (shadow_message_id,),
            )
            return await cur.fetchall()


async def _fetch_vocab_hits(shadow_message_id):
    from db.connection import get_pool
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT channel, term, category FROM shadow_vocab_hits "
                "WHERE shadow_message_id = %s",
                (shadow_message_id,),
            )
            return await cur.fetchall()


def test_shadow_validation_writes_message_row_and_sets_validated_at(client):
    from shadow_validation import run_shadow_validation
    smid = str(uuid.uuid4())
    contract = ContractResult(
        contract_parsed=True, claims=[], analysis="sin nada verificable",
        judgment=None, degradation_reason=None, raw_text="...",
    )
    client.portal.call(run_shadow_validation, "conv-fake-uuid", smid, "jekyll", contract)

    row = client.portal.call(_fetch_shadow_message, smid)
    assert row is not None
    conv_uuid, facet, contract_parsed, degradation_reason, has_claim, has_analysis, has_judgment, validated_at = row
    assert conv_uuid == "conv-fake-uuid"
    assert facet == "jekyll"
    assert bool(contract_parsed) is True
    assert bool(has_claim) is False
    assert bool(has_analysis) is True
    assert bool(has_judgment) is False
    assert validated_at is not None  # el worker no murió, se completó


def test_shadow_validation_claim_produces_authority_invalid_verdict(client):
    from shadow_validation import run_shadow_validation
    smid = str(uuid.uuid4())
    contract = ContractResult(
        contract_parsed=True,
        claims=[{"predicate": "CAPABILITY_AVAILABLE", "args": {"name": "code_swarm"}}],
        analysis="revisé el catálogo", judgment=None,
        degradation_reason=None, raw_text="...",
    )
    client.portal.call(run_shadow_validation, "conv-fake-uuid-2", smid, "jekyll", contract)

    verdicts = client.portal.call(_fetch_claim_verdicts, smid)
    assert len(verdicts) == 1
    predicate, status, args = verdicts[0]
    assert predicate == "CAPABILITY_AVAILABLE"
    # Resultado esperado de esta ronda (spec, sección "Alcance"): authority
    # siempre INFERIDO, prohibido en canal claim — NO es un bug.
    assert status == "AUTHORITY_INVALID"
    assert args == {"name": "code_swarm"}


def test_shadow_validation_sweeps_analysis_and_judgment_for_vocab_hits(client):
    from shadow_validation import run_shadow_validation
    smid = str(uuid.uuid4())
    contract = ContractResult(
        contract_parsed=True, claims=[],
        analysis="mencioné code_swarm en el análisis",
        judgment="y también trae a hyde en el judgment",
        degradation_reason=None, raw_text="...",
    )
    client.portal.call(run_shadow_validation, "conv-fake-uuid-3", smid, "jax_local", contract)

    hits = client.portal.call(_fetch_vocab_hits, smid)
    channels_terms = {(h[0], h[1]) for h in hits}
    assert ("analysis", "code_swarm") in channels_terms
    assert ("judgment", "trae a hyde") in channels_terms


def test_shadow_validation_navigable_without_messages_row(client):
    # El punto del hallazgo de conv_uuid/shadow_message_id: una fila de
    # shadow es navegable a su conversación aunque `messages` (la tabla
    # real de mensajes, escrita fire-and-forget por _memory.save_message())
    # todavía no tenga el mensaje guardado — no hay FK a `messages`.
    from shadow_validation import run_shadow_validation
    smid = str(uuid.uuid4())
    fake_conv_uuid = str(uuid.uuid4())  # UUID que casi seguro no existe en `conversations`
    contract = ContractResult(
        contract_parsed=True, claims=[], analysis="x", judgment=None,
        degradation_reason=None, raw_text="x",
    )
    client.portal.call(run_shadow_validation, fake_conv_uuid, smid, "jekyll", contract)
    row = client.portal.call(_fetch_shadow_message, smid)
    assert row is not None
    assert row[0] == fake_conv_uuid  # navegable por conv_uuid, sin depender de `messages`


def test_shadow_validation_degraded_message_still_gets_row(client):
    # El caso que el spec marca como el más grave si se pierde: JSON
    # truncado, sin claims recuperables, sin términos de vocabulario.
    from shadow_validation import run_shadow_validation
    smid = str(uuid.uuid4())
    contract = ContractResult(
        contract_parsed=False, claims=[], analysis="texto truncado sin nada reconocible",
        judgment=None, degradation_reason="JSON no parsea", raw_text="texto truncado sin nada reconocible",
    )
    client.portal.call(run_shadow_validation, "conv-fake-uuid-4", smid, "kimi", contract)

    row = client.portal.call(_fetch_shadow_message, smid)
    assert row is not None
    assert bool(row[2]) is False  # contract_parsed
    assert row[3] == "JSON no parsea"  # degradation_reason
    verdicts = client.portal.call(_fetch_claim_verdicts, smid)
    hits = client.portal.call(_fetch_vocab_hits, smid)
    assert verdicts == []
    assert hits == []


def test_shadow_validation_leaves_validated_at_null_when_worker_dies_mid_run(client):
    # El caso que shadow_messages.validated_at existe para hacer visible:
    # si el proceso muere (acá simulado con una excepción real dentro del
    # sweep de vocabulario) DESPUÉS de insertar la fila pero ANTES de
    # completarla, validated_at queda NULL para siempre — esa ausencia ES
    # la métrica de pérdida, sin contador aparte (spec, sección 3).
    import shadow_validation
    from unittest.mock import patch

    smid = str(uuid.uuid4())
    contract = ContractResult(
        contract_parsed=True, claims=[], analysis="texto cualquiera",
        judgment=None, degradation_reason=None, raw_text="...",
    )
    with patch.object(
        shadow_validation.governance_vocab_sweep, "sweep",
        side_effect=RuntimeError("worker murió acá, simulado"),
    ):
        try:
            client.portal.call(
                shadow_validation.run_shadow_validation,
                "conv-fake-uuid-5", smid, "jekyll", contract,
            )
        except RuntimeError:
            pass  # esperado — lo que importa es el estado que quedó en DB

    row = client.portal.call(_fetch_shadow_message, smid)
    assert row is not None  # la fila SÍ se insertó (al encolar, antes del crash)
    assert row[-1] is None  # validated_at — nunca se completó


def test_shadow_validation_skips_when_conv_uuid_is_none(client):
    from shadow_validation import run_shadow_validation
    smid = str(uuid.uuid4())
    contract = ContractResult(
        contract_parsed=True, claims=[], analysis="x", judgment=None,
        degradation_reason=None, raw_text="x",
    )
    client.portal.call(run_shadow_validation, None, smid, "jekyll", contract)
    row = client.portal.call(_fetch_shadow_message, smid)
    assert row is None  # sin conv_uuid no hay a qué mensaje navegar, no se encola
```

- [ ] **Step 3: Correr los tests, verificar que fallan**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/pytest tests/test_shadow_validation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shadow_validation'`

- [ ] **Step 4: Implementar `shadow_validation.py`**

Crear `/home/fruiz/jax-platform/backend/shadow_validation.py`:

```python
"""
Shadow validation (REFORMAS-v3 Fase 2 Sub-proyecto 2) — corre después de
que la Mesa web ya respondió al usuario. Mide, no bloquea: cada claim se
valida contra policy/governance/validator.py, cada bloque de
analysis/judgment se barre contra el vocabulario cerrado.

Importa policy/governance/ directo desde ~/jax (sys.path) — mismo patrón
que api/chat.py ya usa para jax.memory.db.MemoryDB. No hay puente HTTP:
ambos repos viven en el mismo host, y validator.py ya asume ese layout
(sus propios imports insertan REPO_ROOT en sys.path).

authority de todo claim es SIEMPRE "INFERIDO", fijado acá — nunca lo
declara el modelo (ver spec, sección 1a: P08 aplicado a metadata).
Resultado esperado: 100% AUTHORITY_INVALID esta ronda, porque chat.py no
tiene grounding cableado al mecanismo de claims. No es un bug.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

JAX_REPO = Path(os.path.expanduser("~/jax"))
if str(JAX_REPO) not in sys.path:
    sys.path.insert(0, str(JAX_REPO))
if str(JAX_REPO / "policy" / "governance") not in sys.path:
    sys.path.insert(0, str(JAX_REPO / "policy" / "governance"))

import claims as governance_claims  # noqa: E402
import loaders as governance_loaders  # noqa: E402
import validator as governance_validator  # noqa: E402
import vocab_sweep as governance_vocab_sweep  # noqa: E402

from api.chat import ContractResult  # noqa: E402
from db.connection import get_pool  # noqa: E402


@lru_cache(maxsize=1)
def _validation_context():
    # Config estática cacheada por proceso — mismo criterio que
    # _load_config() en chat.py (Lección operativa #6, jax-platform/CLAUDE.md):
    # un cambio real requiere reiniciar el proceso, no releer en cada request.
    vocabulary = governance_loaders.load_vocabulary()
    ctx = governance_validator.load_validation_context(
        JAX_REPO, vocabulary.config_paths
    )
    predicates = governance_loaders.load_predicates()
    return ctx, predicates, vocabulary.term_categories


async def _insert_shadow_message(cur, conv_uuid, shadow_message_id, facet, contract):
    await cur.execute(
        "INSERT INTO shadow_messages "
        "(conv_uuid, shadow_message_id, facet, contract_parsed, degradation_reason, "
        "has_claim, has_analysis, has_judgment) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (
            conv_uuid, shadow_message_id, facet, contract.contract_parsed,
            contract.degradation_reason,
            bool(contract.claims), bool(contract.analysis), bool(contract.judgment),
        ),
    )


async def _mark_validated(cur, shadow_message_id):
    await cur.execute(
        "UPDATE shadow_messages SET validated_at = NOW() WHERE shadow_message_id = %s",
        (shadow_message_id,),
    )


async def _insert_claim_verdict(cur, conv_uuid, shadow_message_id, verdict, args):
    import json
    await cur.execute(
        "INSERT INTO shadow_claim_verdicts "
        "(conv_uuid, shadow_message_id, predicate, status, detail, args) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (conv_uuid, shadow_message_id, verdict.predicate, verdict.status, verdict.detail,
         json.dumps(args)),
    )


async def _insert_vocab_hit(cur, conv_uuid, shadow_message_id, channel, term, category):
    await cur.execute(
        "INSERT INTO shadow_vocab_hits "
        "(conv_uuid, shadow_message_id, channel, term, category) "
        "VALUES (%s, %s, %s, %s, %s)",
        (conv_uuid, shadow_message_id, channel, term, category),
    )


async def run_shadow_validation(
    conv_uuid: str | None,
    shadow_message_id: str,
    facet: str,
    contract: "ContractResult | None",
) -> None:
    if conv_uuid is None or contract is None:
        return

    ctx, predicates, term_categories = _validation_context()
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await _insert_shadow_message(cur, conv_uuid, shadow_message_id, facet, contract)

            for raw_claim in contract.claims:
                claim = governance_claims.Claim(
                    predicate=raw_claim["predicate"],
                    args={k: str(v) for k, v in raw_claim["args"].items()},
                    authority="INFERIDO",
                    provenance_ref=facet,
                    evidence_pointer=f"{conv_uuid}:{shadow_message_id}",
                    scope="mesa_web",
                )
                verdict = governance_validator.validate(claim, predicates, ctx)
                await _insert_claim_verdict(
                    cur, conv_uuid, shadow_message_id, verdict, raw_claim["args"]
                )

            for channel, text in (("analysis", contract.analysis), ("judgment", contract.judgment)):
                if not text:
                    continue
                hits = governance_vocab_sweep.sweep(text, term_categories)
                for term, categories in hits:
                    for category in sorted(categories):
                        await _insert_vocab_hit(
                            cur, conv_uuid, shadow_message_id, channel, term, category
                        )

            await _mark_validated(cur, shadow_message_id)
        await conn.commit()
```

- [ ] **Step 5: Correr los tests, verificar que pasan**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/pytest tests/test_shadow_validation.py -v`
Expected: PASS (7/7) — confirmar en particular que `test_shadow_validation_claim_produces_authority_invalid_verdict` pasa con `AUTHORITY_INVALID` (no `VALID` ni `FACT_MISMATCH`) — si sale distinto, algo en `Claim(authority="INFERIDO", ...)` no está llegando como se espera a `validate()`, revisar antes de seguir.

- [ ] **Step 6: Conectar la `BackgroundTask` en `chat()`**

En `/home/fruiz/jax-platform/backend/api/chat.py`:

Agregar import al tope: `from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks` (reemplazando el import existente de `fastapi` que hoy no incluye `BackgroundTasks`).

Modificar la firma del handler:

```python
@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks, user: AuthUser = Depends(get_current_user)):
```

Y justo antes del `return ChatResponse(...)` del Step 13 de Task 3, agregar:

```python
    from shadow_validation import run_shadow_validation
    background_tasks.add_task(run_shadow_validation, conv_uuid, shadow_message_id, facet, contract)
```

(Import local adentro de la función, no al tope del archivo — evita un ciclo de import: `shadow_validation.py` importa `ContractResult` desde `api.chat`, así que `api.chat` no puede importar `shadow_validation` a nivel de módulo sin crear un ciclo.)

- [ ] **Step 7: Escribir un test de integración end-to-end del endpoint**

Agregar a `tests/test_shadow_validation.py`:

```python
def test_chat_endpoint_enqueues_shadow_validation(client):
    import http_client
    from auth.jwt import create_access_token
    from tests.test_chat_contract_wrapper import _FakePostClient, _FakeResponse

    token = create_access_token("test-shadow-e2e-user", "test-shadow-e2e-tenant", "operator")
    fake = _FakePostClient(_FakeResponse({
        "choices": [{"message": {"content":
            '{"claim": [], "analysis": "no hay nada que afirmar", "judgment": null}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }))
    original = http_client._client
    http_client._client = fake
    try:
        resp = client.post(
            "/api/chat",
            json={"message": "hola shadow", "facet": "jekyll"},
            headers={"Authorization": f"Bearer {token}"},
        )
    finally:
        http_client._client = original
    assert resp.status_code == 200
    assert resp.json()["contract_degraded"] is False
```

**Nota:** este test no verifica directamente que la fila de `shadow_messages` haya sido escrita — `BackgroundTasks` de FastAPI corre después de que `TestClient` recibe la respuesta, y el timing exacto respecto al `client.portal` no está garantizado en este test simple. Verificar la escritura real es responsabilidad de `test_shadow_validation_writes_message_row_and_sets_validated_at` (Step 2), que llama a `run_shadow_validation` directo. Este test solo confirma que el endpoint no rompe al encolar la tarea.

- [ ] **Step 8: Correr toda la suite del task, verificar que pasa**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/pytest tests/test_shadow_validation.py tests/test_chat_contract_wrapper.py -v`
Expected: PASS (todos)

- [ ] **Step 9: Correr toda la suite de backend, verificar que nada se rompió**

Run: `cd /home/fruiz/jax-platform/backend && .venv/bin/pytest tests/ -v`
Expected: PASS (todos — presta atención particular a los tests de `chat.py` ya existentes del Step 16 de Task 3, deberían seguir en verde)

- [ ] **Step 10: Commit**

```bash
cd ~/jax-platform
git add backend/shadow_validation.py backend/api/chat.py backend/tests/test_shadow_validation.py
git commit -m "feat(shadow): BackgroundTask valida claims + barre vocabulario, escribe a las 3 tablas (SP2 Task 5)"
```

---

## Task 6: Footnote en `jax-platform-frontend`

**Files:**
- Modify: `/home/fruiz/jax-platform/frontend/src/components/CenterPanel/Message.jsx`
- Modify: `/home/fruiz/jax-platform/frontend/src/i18n/es.js`
- Modify: `/home/fruiz/jax-platform/frontend/src/i18n/en.js`
- Test: `/home/fruiz/jax-platform/frontend/src/components/CenterPanel/Message.test.jsx` (nuevo)

**Interfaces:**
- Consumes: `message.contract_degraded` (bool, del `ChatResponse` de Task 3 — el store de Zustand que arma el objeto `message` a partir de la respuesta de `/api/chat` necesita propagar este campo; ver Step 5).
- Produces: nota visible al pie del bloque de mensaje cuando `message.contract_degraded === true`.

- [ ] **Step 1: Escribir el test que falla**

Crear `/home/fruiz/jax-platform/frontend/src/components/CenterPanel/Message.test.jsx`:

```jsx
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import Message from './Message'
import { I18nProvider } from '../../i18n/index.jsx'

function renderMessage(message) {
  return render(
    <I18nProvider>
      <Message message={message} />
    </I18nProvider>
  )
}

describe('Message contract degradation footnote', () => {
  it('muestra la nota cuando contract_degraded es true', () => {
    renderMessage({
      facet: 'jekyll',
      content: 'respuesta cruda sin parsear',
      contract_degraded: true,
    })
    expect(screen.getByText(/no cumplió el formato esperado/i)).toBeInTheDocument()
  })

  it('no muestra la nota cuando contract_degraded es false', () => {
    renderMessage({
      facet: 'jekyll',
      content: 'respuesta normal',
      contract_degraded: false,
    })
    expect(screen.queryByText(/no cumplió el formato esperado/i)).not.toBeInTheDocument()
  })

  it('no muestra la nota cuando contract_degraded no está presente (mensajes viejos)', () => {
    renderMessage({ facet: 'jekyll', content: 'mensaje de antes de este cambio' })
    expect(screen.queryByText(/no cumplió el formato esperado/i)).not.toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Correr el test, verificar que falla**

Run: `cd /home/fruiz/jax-platform/frontend && npm run test -- Message.test.jsx`
Expected: FAIL — el texto no existe en el componente todavía

- [ ] **Step 3: Agregar los strings a `es.js`/`en.js`**

En `/home/fruiz/jax-platform/frontend/src/i18n/es.js`, agregar junto a `userLabel` (línea 130):

```js
  contractDegradedNote: 'La respuesta no cumplió el formato esperado.',
```

En `/home/fruiz/jax-platform/frontend/src/i18n/en.js`, agregar junto a `userLabel`:

```js
  contractDegradedNote: 'The response did not meet the expected format.',
```

- [ ] **Step 4: Implementar el footnote en `Message.jsx`**

En `Message.jsx`, agregar el footnote dentro del `div` que envuelve `ReactMarkdown` (después de `<ReactMarkdown>{message.content}</ReactMarkdown>`, antes del cierre de ese `div`):

```jsx
          <ReactMarkdown>{message.content}</ReactMarkdown>
          {message.contract_degraded && (
            <div className="text-xs text-slate-500 mt-2 italic">
              {t.contractDegradedNote}
            </div>
          )}
```

(`text-slate-500` sigue el mismo hardcoded-a-oscuro que el resto del componente — no hay sistema de temas en esta app para respetar, ver Global Constraints.)

- [ ] **Step 5: Correr el test, verificar que sigue fallando — falta propagar el campo desde el store**

Run: `cd /home/fruiz/jax-platform/frontend && npm run test -- Message.test.jsx`
Expected: si el componente ya lee `message.contract_degraded` directo de la prop `message` (como en el test, que construye el objeto a mano), debería PASAR en este punto — los 3 tests son sobre el componente aislado, no sobre el store. Si pasa, seguir al Step 6. Si falla, revisar el JSX del Step 4 contra el test antes de continuar.

- [ ] **Step 6: Verificar (no modificar todavía) cómo el store construye el objeto `message` desde la respuesta de `/api/chat`**

Run: `grep -n "response\.facet\|response\.response\|ChatResponse\|/api/chat" /home/fruiz/jax-platform/frontend/src/store/useJaxStore.js | head -20`

Localizar dónde el store arma el objeto de mensaje del asistente a partir de la respuesta JSON del backend (probablemente algo como `{ facet: data.facet, content: data.response, timestamp: data.timestamp }`). Agregar `contract_degraded: data.contract_degraded ?? false` a ese objeto, seguido del mismo patrón de asignación de propiedades que ya usan `facet`/`content`/`timestamp` ahí. Esta es la única línea que cambia en `useJaxStore.js` — no reestructurar nada más.

- [ ] **Step 7: Escribir un test para el store (si `useJaxStore.js` tiene tests de este flujo)**

Run: `grep -rln "contract_degraded\|/api/chat" /home/fruiz/jax-platform/frontend/src/store/*.test.js`

Si existe un test que ya mockea la respuesta de `/api/chat` para construir un mensaje del store (revisar `useJaxStore.test.js` y los archivos `useJaxStore.*.test.js`), agregar ahí un caso que confirme que `contract_degraded: true` en la respuesta mockeada llega como `contract_degraded: true` en el mensaje que el store agrega a su lista. Seguir exactamente el patrón de mock ya usado en ese archivo (no inventar uno nuevo). Si no existe ningún test de ese flujo, omitir este step — no agregar cobertura de store más allá de lo que ya existe como convención de este proyecto.

- [ ] **Step 8: Correr toda la suite de frontend, verificar que pasa**

Run: `cd /home/fruiz/jax-platform/frontend && npm run test`
Expected: PASS (todos)

- [ ] **Step 9: Build de verificación**

Run: `cd /home/fruiz/jax-platform/frontend && npm run build`
Expected: build sin errores (no se hace deploy en este task — ver Lección operativa #1 de `jax-platform/CLAUDE.md`: el deploy a producción es un paso manual aparte, `rsync` a la VM dev, fuera de alcance de este plan)

- [ ] **Step 10: Commit**

```bash
cd ~/jax-platform
git add frontend/src/components/CenterPanel/Message.jsx frontend/src/components/CenterPanel/Message.test.jsx frontend/src/i18n/es.js frontend/src/i18n/en.js frontend/src/store/useJaxStore.js
git commit -m "feat(frontend): footnote sobrio cuando contract_degraded=true (SP2 Task 6)"
```

---

## Cierre del sub-proyecto

- [ ] Push de las tres ramas correspondientes (`jax`, `jax-platform`) — ver memoria `gh-cli-cuelga-hall9000` para el flujo de PR sin `gh`.
- [ ] Revisión final de rama, como en Sub-proyecto 1 (subagente fresco, whole-branch review) antes de mergear.
- [ ] Confirmar en producción, después de mergear y de un ciclo real de tráfico, que `shadow_messages` está recibiendo filas y que `validated_at` no queda NULL de forma sistemática (si el worker de FastAPI se recicla con frecuencia, `BackgroundTasks` puede perder tareas silenciosamente — ver spec, sección 3, "garantizar que la pérdida se cuente").
- [ ] No "arreglar" el 100% `AUTHORITY_INVALID` en `shadow_claim_verdicts` — es el resultado esperado de esta ronda (ver Global Constraints y spec sección "Alcance"). Si en dos semanas alguien lo lee y quiere cablear grounding, ese es el caso de uso medido para Sub-proyecto 3, no una tarea de mantenimiento de este sub-proyecto.
