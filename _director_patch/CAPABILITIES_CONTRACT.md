# Contrato CRUDO de capabilities — Jacobs ↔ Motor Registry

> Volcado de evidencia para decisión de corrección de raíz. **Sin propuestas.**
> Generado leyendo el árbol vivo de la rama `feat/director-orquesta-waves` (commit `01eacc4`).
> Fuentes: `jacobs/executor.py`, `jacobs/plan.py`, `las_manos/config.toml` (el que carga el Motor Registry).

---

## 1. `_CAPABILITY_MAP` completo (`jacobs/executor.py:377-383`)

Traducción semántica (la que emite el planner) → nombre de capability del catálogo.
Aplica **SOLO** en la ruta del Motor Registry (`_invoke_motor`). Lo que no está en el dict se despacha **crudo** (pass-through) vía `.get(cap, cap)`.

```python
_CAPABILITY_MAP = {
    "analysis":  "pipeline_analysis",
    "research":  "pipeline_analysis",
    "review":    "refactor",
    "code":      "refactor",
    "implement": "code_swarm",
}
```

Mecánica exacta en `_invoke_motor` (`executor.py:388-396`):
```python
capability = _CAPABILITY_MAP.get(step.capability, step.capability)
if step.capability not in _CAPABILITY_MAP:
    logger.warning(
        "Capability '%s' no está en _CAPABILITY_MAP; se despacha cruda al "
        "Motor Registry (debe existir como [capabilities.%s] en config.toml).",
        step.capability, capability,
    )
```

- 5 entradas traducidas. Todo lo demás (p.ej. `generate`, `design`, `validate_consistency`, `reconcile`, `critique`, `reason`) → pasa **crudo** al Registry.

---

## 2. Catálogo completo de capabilities (`las_manos/config.toml`)

Este es el archivo que el Motor Registry carga (`motor_registry/routes.py:36` → `BASE_DIR/config.toml` = `las_manos/config.toml`). **NO** es `config/config.toml`.

### Motores relevantes (estado enabled)
```toml
[motors.kimi]
enabled = true
provider = "kimi"
api_key_env = "KIMI_API_KEY"
model = "kimi-k2.7-code"
sandbox_only = true
default_timeout_seconds = 600

[motors.ada]
enabled = true
provider = "zai"
api_key_env = "ZAI_API_KEY"
model = "glm-5.2"
sandbox_only = true
default_timeout_seconds = 600
```

### Capabilities (bloques TOML crudos)
```toml
[capabilities.code_swarm]
allowed_motors = ["kimi"]
allowed_callers = ["hyde", "ada", "kimi"]
risk_level = "high"
sandbox_only = true
requires_human_gate = true
max_execution_minutes = 30
max_recursion_depth = 1
output_schema = "code_swarm.v1"
fallback_motor = "ada"
fallback_mode = "manual_only"
forbidden_paths = [".env", "secrets/", "private_keys/", "credentials/"]

[capabilities.refactor]
allowed_motors = ["kimi"]
allowed_callers = ["hyde", "ada", "jacobs"]
risk_level = "medium"
sandbox_only = true
requires_human_gate = false
max_execution_minutes = 10
max_recursion_depth = 0
output_schema = "code_patch.v1"

[capabilities.architecture_review]
allowed_motors = ["ada"]
allowed_callers = ["hyde"]
risk_level = "medium"
sandbox_only = true
requires_human_gate = false
max_execution_minutes = 5
max_recursion_depth = 0
output_schema = "architecture_review.v1"

[capabilities.bug_hunt]
allowed_motors = ["kimi"]
allowed_callers = ["hyde", "ada"]
risk_level = "high"
sandbox_only = true
requires_human_gate = true
max_execution_minutes = 15
max_recursion_depth = 0
output_schema = "bug_hunt.v1"

[capabilities.pipeline_analysis]
allowed_motors = ["kimi"]
allowed_callers = ["jacobs", "hyde"]
risk_level = "low"
sandbox_only = true
requires_human_gate = false
max_execution_minutes = 15
max_recursion_depth = 0
output_schema = "analysis.v1"

[capabilities.implementation]
allowed_motors = ["kimi"]
allowed_callers = ["jacobs", "hyde"]
risk_level = "medium"
sandbox_only = true
requires_human_gate = false
max_execution_minutes = 30
max_recursion_depth = 0
output_schema = "code_patch.v1"
forbidden_paths = [".env", "secrets/", "private_keys/", "credentials/"]
```

### Tabla resumen (los 6 del catálogo)

| capability | allowed_motors | allowed_callers | requires_human_gate | output_schema |
|---|---|---|---|---|
| `code_swarm` | kimi | hyde, ada, kimi | **true** | code_swarm.v1 |
| `refactor` | kimi | hyde, ada, **jacobs** | false | code_patch.v1 |
| `architecture_review` | ada | hyde | false | architecture_review.v1 |
| `bug_hunt` | kimi | hyde, ada | **true** | bug_hunt.v1 |
| `pipeline_analysis` | kimi | **jacobs**, hyde | false | analysis.v1 |
| `implementation` | kimi | **jacobs**, hyde | false | code_patch.v1 |

Orden de validación en `policy.py:check()`: (1) capability existe → (2) caller en allowed_callers → (3) human_gate_token si requires_human_gate → (4) recursion_depth → (5) context keys → (6) motor habilitado en allowed_motors → (7) motor sandbox_only. Falla al primer fallo. `jacobs` **nunca** envía `human_gate_token` desde `_invoke_motor` (el payload no lo incluye).

---

## 3. Qué capabilities emite el planner desde un objetivo (`jacobs/plan.py`)

### De dónde salen: del LLM, texto LIBRE. NO hay restricción al catálogo.

`_parse_plan_json` (`plan.py:315-364`) es el único saneo. Valida **facet** contra `VALID_FACETS`, pero **capability se acepta libre** (solo se castea a str y se trunca a 50 chars; default `"reason"`):

```python
        valid = []
        for idx, item in enumerate(data[:max_steps]):
            if not isinstance(item, dict):
                continue
            facet = item.get("facet", "")
            if facet not in VALID_FACETS:
                facet = "jax_local"
            ...
            valid.append({
                "facet": facet,
                "capability": str(item.get("capability", "reason"))[:50],   # <-- LIBRE, sin catálogo
                "prompt": str(item.get("prompt", ""))[:2000],
                "depends_on": depends_on,
            })
```

`VALID_FACETS` (`plan.py`): `{"hipatia","jekyll","thot","ada","kimi","hyde","jax_local"}`. **No existe una lista equivalente para capabilities.**

### Capabilities que el planner ve como ejemplo en los prompts (lo que tiende a emitir)

**`_PLAN_SYSTEM_MODULAR` + `_ada_plan` (ruta formal, cerebro Ada)** — ejemplos sembrados:
- `design` (ada)
- `validate_consistency` (thot) — antepenúltimo
- `reconcile` (ada) — penúltimo
- `assemble` (ada) — último

**`_llm_plan` (ruta trivial, qwen) + ejemplo (`plan.py:292-293`)**:
- `research` (hipatia)
- `analysis` (jekyll)

**`_fallback_plan` (`plan.py:369-381`)** — plan fijo hardcodeado si todo falla:
- `research` (hipatia)
- `analysis` (jekyll)
- `critique` (thot)

### Cruce factual emitido → Motor Registry (solo describe lo que hace el código, sin juicio)

Aplica únicamente si el step va a `kimi` (los demás facets ignoran capability, ver §4).

| capability emitida | en `_CAPABILITY_MAP`? | va al Registry como | existe en catálogo? |
|---|---|---|---|
| `analysis` | sí → | `pipeline_analysis` | sí |
| `research` | sí → | `pipeline_analysis` | sí |
| `review` | sí → | `refactor` | sí |
| `code` | sí → | `refactor` | sí |
| `implement` | sí → | `code_swarm` | sí (gate=true, caller jacobs NO permitido) |
| `design` | no → crudo | `design` | **no** |
| `validate_consistency` | no → crudo | `validate_consistency` | **no** |
| `reconcile` | no → crudo | `reconcile` | **no** |
| `critique` | no → crudo | `critique` | **no** |
| `generate` | no → crudo | `generate` | **no** |
| `reason` (default) | no → crudo | `reason` | **no** |
| `assemble` | — | (no llega al Registry: cortocircuito en `_dispatch_step`) | n/a |

---

## 4. Rutas directas (`_invoke_*`): ¿validan capability?

**No. La ignoran por completo.** Solo `_dispatch_step` rutea, y lo hace por **`step.facet`**. La única vez que `capability` se mira en toda la ruta directa es el cortocircuito `if step.capability == "assemble"`.

Firmas (ninguna recibe `capability`):
```
async def _invoke_hipatia(prompt: str, timeout: int) -> dict
async def _invoke_jekyll(prompt: str, timeout: int) -> dict
async def _invoke_thot(prompt: str, timeout: int) -> dict
async def _invoke_ada(prompt: str, timeout: int) -> dict
async def _invoke_jax_local(prompt: str, timeout: int) -> dict
async def _invoke_motor(step: Step, timeout: int) -> dict   # <-- único que recibe `step` (y por ende capability)
```

`_dispatch_step` completo (`executor.py:498-533`):
```python
async def _dispatch_step(step: Step, pipeline: Pipeline) -> dict:
    """Selecciona el worker correcto según la faceta."""
    # Ensamble mecánico: NO pasa por ningún LLM. Concatena los módulos ya generados.
    if step.capability == "assemble":
        return _assemble_mechanical(step, pipeline)

    ctx_input = _build_context_input(step, pipeline)
    prompt    = _enrich_prompt(ctx_input)
    timeout   = step.timeout_seconds

    if step.facet == "hipatia":
        return await _invoke_hipatia(prompt, timeout)
    if step.facet == "jekyll":
        return await _invoke_jekyll(prompt, timeout)
    if step.facet == "thot":
        return await _invoke_thot(prompt, timeout)
    if step.facet == "ada":
        return await _invoke_ada(prompt, timeout)
    if step.facet == "jax_local":
        return await _invoke_jax_local(prompt, timeout)
    if step.facet in _MOTOR_FACETS:
        return await _invoke_motor(step, timeout)
    if step.facet == "hyde":
        # Llegamos aquí solo si Fernando aprobó vía /approve-step.
        return { ... "result": "[v0.2] Hyde aprobado ...", "approved": True }

    raise ValueError(f"Faceta desconocida: '{step.facet}'")
```

`_MOTOR_FACETS = frozenset({"kimi"})` y `_HTTP_FACETS = frozenset({"hipatia", "jekyll", "thot", "ada"})` (`executor.py:35-38`).

### Resumen factual de ruteo por facet

| facet | ruta | ¿capability validada? | ¿toca Motor Registry / policy? |
|---|---|---|---|
| hipatia | `_invoke_hipatia` | no | no |
| jekyll | `_invoke_jekyll` | no | no |
| thot | `_invoke_thot` | no | no |
| ada | `_invoke_ada` | no | no |
| jax_local | `_invoke_jax_local` | no | no |
| kimi | `_invoke_motor` | **sí** (vía `_CAPABILITY_MAP` + `policy.check`) | **sí** |
| hyde | placeholder v0.2 | no | no |
| (cualquier `capability=="assemble"`) | `_assemble_mechanical` (mecánico, sin LLM) | corto-circuito | no |

---

## Síntesis de los cuatro datos (solo hechos, sin corrección propuesta)

1. Único punto del pipeline donde `capability` se valida contra un catálogo cerrado es el facet **`kimi`** (→ `_invoke_motor` → `policy.check`).
2. El planner emite `capability` **libre** (no la restringe al catálogo); facet sí lo restringe.
3. `_CAPABILITY_MAP` traduce 5 nombres; el resto pasa crudo al Registry.
4. Los facets de API directa (ada, thot, hipatia, jekyll, jax_local) **ignoran** `capability` por completo (excepto el cortocircuito `assemble`).
