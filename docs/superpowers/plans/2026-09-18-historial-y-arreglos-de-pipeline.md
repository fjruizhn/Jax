# Historial de pipelines y arreglos del encadenamiento — plan de implementación

> **Para trabajadores agénticos:** SUB-SKILL REQUERIDA: usá
> `superpowers:subagent-driven-development` (recomendada) o
> `superpowers:executing-plans` para ejecutar este plan tarea por tarea.
> Los pasos usan casilla (`- [ ]`) para seguimiento.

**Goal:** que un pipeline encadene por defecto, se detenga en vez de entregar texto
cortado, termine con una decisión arbitrada, y que su trabajo se pueda ver y releer
después desde la Mesa.

**Architecture:** dos repos. En `jax` (Jacobs) van los arreglos del motor del pipeline:
orden de ejecución, fallo por truncado, paso árbitro, modelo real persistido y aviso por
Telegram. En `jax-platform` (Mesa) va lo que se ve: historial, detalle por paso y aviso
por correo. Cada aviso lo manda el repo que ya tiene ese canal — Telegram existe solo en
`jax`, SMTP existe solo en `jax-platform`. Ninguno de los dos se copia al otro.

**Tech Stack:** Python 3.12 + FastAPI + aiomysql (MariaDB 12.3.3), React 19 + Vite +
Zustand + react-i18next, pytest, vitest.

**Spec:** `docs/superpowers/specs/2026-09-18-historial-y-arreglos-de-pipeline-design.md`
(commit `d575a8a`). El plan argumenta desde el spec: los ejecutores leen los dos.

---

## Global Constraints

Valores exactos, copiados del spec y verificados contra el código el 2026-09-18. Los
requisitos de cada tarea incluyen esta sección implícitamente.

- **i18n:** ningún string visible hardcodeado. `frontend/src/i18n/es.js` y `en.js`,
  claves camelCase. La paridad es/en la fuerza `frontend/src/i18n/paridad.test.js`.
- **Tema:** sin clases `dark:`. Se usa la clase semántica (`bg-superficie`,
  `text-texto-tenue`, `border-borde`) y el valor lo resuelve `data-tema` en `<html>`.
  Tokens en `frontend/src/tema/tokens.css`, lista cerrada en `frontend/src/tema/tokens.js`.
- **Confirmaciones:** nunca `confirm(`/`alert(`/`prompt(`, ni con `window.` ni desnudos.
  `components/Dialogo.jsx`; `ConfirmacionSuma` para lo destructivo.
- **Pisos de CI** (`.github/workflows/policy.yml` de jax-platform), valores actuales:
  vitest **784** (línea 620), backend con DB **2340** (`PISO_PASSED`, línea 1729),
  backend sin DB **1582** (`JAX_CI_MIN_PASSED`, línea 2335). Toda tarea que agregue
  tests sube el piso que corresponda **en el mismo commit**.
- **Base de datos:** CI y producción corren `mariadb:12.3.3` (jax#219 / jax-platform#115).
- **Regla de verificación:** todo test de un arreglo se corre **primero contra el código
  viejo y tiene que salir rojo**. Un control que no falla cuando debería fallar no valida
  nada. Si sale verde contra el código viejo, el test está mal, no el código.
- **Nada de datos inventados:** ningún número de rendimiento se escribe sin haberlo medido.

### Comandos de test

```bash
# jax — sin DB
cd /home/fruiz/jax && PYTHONPATH=.:las_manos python -m pytest -v <archivos>

# jax — con DB (gobernanza)
JAX_DB_HOST=127.0.0.1 JAX_DB_PORT=3306 JAX_DB_USER=root JAX_DB_PASSWORD=ci \
JAX_DB_NAME=jax_memory_test PYTHONPATH=.:las_manos python -m pytest -v <archivos>

# jax-platform — con DB
cd backend && python -m pytest -q -rs

# jax-platform — sin DB
cd backend && JAX_CI_NO_DB=1 python -m pytest -q

# frontend
cd frontend && npx vitest run
```

---

## Estructura de archivos

**jax (Jacobs)**

| Archivo | Responsabilidad | Tarea |
|---|---|---|
| `jacobs/models.py` | `Step` gana `modelo_real` y `finish_reason` | 1, 3 |
| `jacobs/executor.py` | orden de olas, truncado, modelo real, aviso | 1, 2, 3, 6 |
| `jacobs/plan.py` | `depends_on` ausente ≠ vacío; árbitro obligatorio | 2, 4 |
| `jacobs/aviso.py` | **nuevo** — aviso de fin por Telegram | 6 |
| `jacobs/_aviso_test.py` | **nuevo** | 6 |

**jax-platform (Mesa)**

| Archivo | Responsabilidad | Tarea |
|---|---|---|
| `backend/api/pipelines.py` | listado con estado/costo/duración | 7 |
| `backend/jax_engine/state.py` | engancha el correo al fin del pipeline | 8 |
| `backend/aviso_pipeline.py` | **nuevo** — armado y envío del correo | 8 |
| `frontend/src/pages/Historial.jsx` | **nuevo** — listado | 9 |
| `frontend/src/components/historial/DetallePipeline.jsx` | **nuevo** — detalle por paso | 9 |
| `frontend/src/store/useJaxStore.js` | acción `cargarHistorial` | 9 |
| `frontend/src/i18n/es.js` · `en.js` | claves de la sección | 9 |

---

## Task 1: El modelo real queda escrito en el paso

Hoy `Step` no tiene dónde guardar qué modelo corrió. `resolve_facet()` lo sabe en
`executor.py:951` (`f.model`) y lo tira. Por eso el informe de `b8f80733` decía
"Modelo desconocido". Saber que actuó `jax_local` no dice qué modelo fue: el binding
cambia, y un historial que no lo fija miente con el tiempo.

**Files:**
- Modify: `jacobs/models.py` (clase `Step`, líneas 66-84)
- Modify: `jacobs/executor.py:951` (tras `resolve_facet`), `executor.py:604-607` (camino motor)
- Test: `jacobs/_modelo_real_test.py` (nuevo)

**Interfaces:**
- Produces: `Step.modelo_real: str | None` — el `model_id` que realmente despachó ese paso.
  Lo consumen las tareas 7 y 9.

- [ ] **Step 1: Escribir el test que falla**

```python
# jacobs/_modelo_real_test.py
import asyncio
from jacobs.models import Step


def test_step_tiene_donde_guardar_el_modelo_real():
    """Sin este campo, el historial solo puede mostrar la faceta, y la faceta
    no dice que modelo corrio: el binding cambia."""
    paso = Step(step_id="s1", pipeline_id="p1", step_index=0,
                facet="thot", capability="text_generation", input={"prompt": "x"})
    assert hasattr(paso, "modelo_real")
    assert paso.modelo_real is None


def test_el_despacho_escribe_el_modelo_resuelto(monkeypatch):
    """El modelo se conoce en resolve_facet y hoy se tira."""
    from jacobs import executor

    class FacetaFalsa:
        transport = "openai_compat"
        model = "glm-4.6"
        persona = ""
        base_url = "http://x"
        provider_id = 1
        params = {}

    async def resolve_falso(_clave):
        return FacetaFalsa()

    async def invoke_falso(*_a, **_k):
        return {"text": "ok"}

    monkeypatch.setattr(executor, "resolve_facet", resolve_falso)
    monkeypatch.setattr(executor, "_invoke_openai_compat", invoke_falso)

    paso = Step(step_id="s1", pipeline_id="p1", step_index=0,
                facet="thot", capability="text_generation", input={"prompt": "x"})
    pipeline = executor.Pipeline(pipeline_id="p1", name="n", invoked_by="test",
                                 mode="auto", status="running", plan=[paso])
    asyncio.run(executor._dispatch_step(paso, pipeline))
    assert paso.modelo_real == "glm-4.6"
```

- [ ] **Step 2: Correrlo contra el código de hoy y verlo ROJO**

Run: `cd /home/fruiz/jax && PYTHONPATH=.:las_manos python -m pytest -v jacobs/_modelo_real_test.py`
Expected: FAIL — `AttributeError` / `assert None == "glm-4.6"`.

Si sale verde, pará: el test no está midiendo nada.

- [ ] **Step 3: Agregar el campo**

En `jacobs/models.py`, dentro de `class Step`:

```python
    # El model_id que REALMENTE despacho este paso, escrito en el momento del
    # despacho. La faceta no alcanza: facet_binding cambia y un historial que
    # solo guarde la faceta miente con el tiempo.
    modelo_real: str | None = None
```

- [ ] **Step 4: Escribirlo en el despacho**

En `jacobs/executor.py`, justo después de `f = await resolve_facet(step.facet)` (línea 951):

```python
    step.modelo_real = f.model
```

Y en el camino de motor (LAS MANOS), tras recibir el job en `executor.py:604-607`:

```python
        step.modelo_real = job.get("model") or job.get("motor") or step.modelo_real
```

- [ ] **Step 5: Correr los tests y verlos verdes**

Run: `PYTHONPATH=.:las_manos python -m pytest -v jacobs/_modelo_real_test.py`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add jacobs/models.py jacobs/executor.py jacobs/_modelo_real_test.py
git commit -m "feat(jacobs): el paso guarda que modelo lo ejecuto de verdad"
```

---

## Task 2: Encadenar por defecto

`_compute_waves` (`executor.py:986`) mete en la ola 0 a todo paso con `depends_on`
vacío, porque `set() <= satisfied` siempre es verdadero. Eso es lo que hizo que los seis
pasos de `b8f80733` arrancaran en el mismo instante y ninguno leyera a nadie.

La regla nueva: **ausencia de `depends_on` ≠ lista vacía declarada.** Si el plan no dice
nada, se encadena. Si el plan **declara** paralelo, se respeta.

**Hay DOS caminos, no uno**, y el segundo esconde una trampa que este repo ya pisó:

1. **Plan del LLM** → `_parse_plan_json(text: str, max_steps: int, governance: dict|None)`,
   `async` y **staticmethod**, en `plan.py:857`. Recibe **texto crudo**, devuelve
   `list[dict] | None`. Hoy normaliza con `raw_deps = item.get("depends_on", [])` (línea 901).
2. **Pasos explícitos del caller** → `_from_spec` (`plan.py:536`), que hace
   `depends_on=spec.get("depends_on", [])` (línea 573).

La trampa del camino 2: `routes.py` arma los specs con `StepSpec.model_dump()`, que
**siempre incluye la clave aunque el caller nunca la haya tocado**. Es exactamente el
defecto que ya les mordió con `timeout_seconds` y que está documentado en
`jacobs/models.py:200-210`: ahí `int = 300` se cambió a `int | None = None` justo para
poder distinguir "ausente" de "el caller lo pidió". `depends_on` sigue con el defecto
puesto: `list[int] = Field(default_factory=list)` (`models.py:213`).

**Por eso el default de Pydantic tiene que pasar a `None`.** Sin eso, "ausente" y "vacío
declarado" son indistinguibles y el arreglo no puede funcionar.

**Files:**
- Modify: `jacobs/models.py:213` (`StepSpec.depends_on: list[int] | None = None`)
- Modify: `jacobs/plan.py:901` (`_parse_plan_json`), `plan.py:573` (`_from_spec`)
- Test: `jacobs/_encadenado_por_defecto_test.py` (nuevo)

**Interfaces:**
- Consumes: `StepSpec.depends_on: list[int] | None`.
- Produces: tras `PlanBuilder.build()`, ningún `Step` sale con `depends_on is None`:
  o trae lo que el plan declaró, o trae `[N-1]` puesto por el encadenado.

- [ ] **Step 1: Escribir el test que falla**

```python
# jacobs/_encadenado_por_defecto_test.py
import asyncio
import json

from jacobs.models import StepSpec
from jacobs.plan import PlanBuilder

GOBIERNO = {"capabilities": {"text_generation": {}}}


def _parsear(items):
    return asyncio.run(
        PlanBuilder._parse_plan_json(json.dumps(items), max_steps=10, governance=GOBIERNO)
    )


def test_el_plan_del_llm_sin_depends_on_se_encadena():
    """b8f80733: seis pasos con depends_on vacio arrancaron en el mismo
    instante (09:55:25.67) y ninguno leyo al anterior."""
    pasos = _parsear([
        {"facet": "jax_local", "capability": "text_generation", "prompt": "uno"},
        {"facet": "ada", "capability": "text_generation", "prompt": "dos"},
        {"facet": "jekyll", "capability": "text_generation", "prompt": "tres"},
    ])
    assert [p["depends_on"] for p in pasos] == [[], [0], [1]]


def test_el_paralelo_declarado_se_respeta():
    """Ausencia != lista vacia declarada. Un plan que PIDE paralelo lo tiene."""
    pasos = _parsear([
        {"facet": "jax_local", "capability": "text_generation", "prompt": "uno"},
        {"facet": "ada", "capability": "text_generation", "prompt": "dos",
         "depends_on": []},
    ])
    assert pasos[1]["depends_on"] == []


def test_el_default_de_pydantic_no_finge_una_decision_del_caller():
    """models.py:200-210 ya conto esta historia con timeout_seconds:
    model_dump() SIEMPRE incluye la clave, asi que un default de lista vacia
    se lee como 'el caller pidio paralelo' aunque nunca la haya tocado."""
    assert StepSpec(facet="ada", capability="text_generation").depends_on is None


def test_pasos_explicitos_sin_depends_on_se_encadenan():
    constructor = PlanBuilder.__new__(PlanBuilder)
    specs = [StepSpec(facet="ada", capability="text_generation", prompt="uno").model_dump(),
             StepSpec(facet="jekyll", capability="text_generation", prompt="dos").model_dump()]
    pasos = constructor._from_spec("p1", specs, {"text_generation": {}})
    assert [p.depends_on for p in pasos] == [[], [0]]
```

- [ ] **Step 2: Correrlo y verlo ROJO**

Run: `PYTHONPATH=.:las_manos python -m pytest -v jacobs/_encadenado_por_defecto_test.py`
Expected: FAIL en el primero, el tercero y el cuarto. Hoy el primero da `[[], [], []]`.

El segundo tiene que pasar **ya con el código viejo**: es la garantía de que el arreglo no
le rompe el paralelo a quien lo pide de verdad.

- [ ] **Step 3: Sacar el default que finge una decisión**

En `jacobs/models.py:213`:

```python
    # None = el caller no dijo nada; [] = pidio paralelo explicito. Mismo
    # motivo que timeout_seconds arriba: model_dump() SIEMPRE incluye la
    # clave, asi que un default de lista vacia se lee como una decision del
    # caller que nunca existio -- y asi corrio b8f80733, seis pasos a la vez.
    depends_on:      list[int] | None = None
```

- [ ] **Step 4: Encadenar en los dos caminos**

En `_parse_plan_json` (`plan.py:901`), reemplazar `raw_deps = item.get("depends_on", [])` por:

```python
            # Ausencia != vacio declarado. Sin dependencias declaradas, el paso
            # depende del anterior: correr todo junto fue el defecto de b8f80733.
            raw_deps = item.get("depends_on")
            if raw_deps is None:
                depends_on = [idx - 1] if idx > 0 else []
            else:
                depends_on = [
                    int(x) for x in raw_deps
                    if str(x).lstrip("-").isdigit() and 0 <= int(x) < idx
                ]
```

(El filtro de rango se conserva tal cual para el caso declarado.)

En `_from_spec` (`plan.py:573`), reemplazar `depends_on=spec.get("depends_on", [])` por:

```python
            deps_explicitas = spec.get("depends_on")
            depends_on = deps_explicitas if deps_explicitas is not None else (
                [i - 1] if i > 0 else []
            )
```

y pasar `depends_on=depends_on` al `Step(...)`.

- [ ] **Step 5: Correr los tests y verlos verdes**

Run: `PYTHONPATH=.:las_manos python -m pytest -v jacobs/_encadenado_por_defecto_test.py`
Expected: PASS (4).

- [ ] **Step 6: Correr la suite de plan y de director, que es donde esto puede romper**

Run: `PYTHONPATH=.:las_manos python -m pytest -v tests/test_plan_validation.py tests/test_jacobs_director.py`
Expected: PASS. Si algo se rompe, el arreglo es el que está mal — no se toca el test para
que pase.

- [ ] **Step 7: Commit**

```bash
git add jacobs/plan.py jacobs/models.py jacobs/_encadenado_por_defecto_test.py
git commit -m "fix(jacobs): un plan sin depends_on se encadena, no corre todo junto"
```

---

## Task 3: Un paso truncado falla, no entrega

El camino de motor **ya lo hace bien**: `las_manos/motor_registry/worker.py:828` falla el
job si `finish_reason == "length"`. Los transportes HTTP directos ni lo leen: Gemini
(`executor.py:303`), openai-compat (`executor.py:394`), Ollama (`executor.py:455`). Ahí
el texto cortado sigue viaje como resultado bueno y el paso siguiente construye sobre una
frase a medias.

**Files:**
- Modify: `jacobs/executor.py` — `_invoke_gemini`, `_invoke_openai_compat`, `_invoke_ollama`
- Test: `jacobs/_truncado_falla_test.py` (nuevo)

**Interfaces:**
- Produces: excepción `PasoTruncado(Exception)` en `jacobs/executor.py`, con
  `codigo = "paso_truncado"`. La consume el manejo de error de `_run_one_step`, que ya
  deja el pipeline continuable.

- [ ] **Step 1: Escribir el test que falla**

```python
# jacobs/_truncado_falla_test.py
import pytest
from jacobs.executor import _texto_o_truncado, PasoTruncado


def test_openai_compat_truncado_falla():
    """Hoy el texto a medias sigue viaje y el paso siguiente construye sobre
    una frase cortada."""
    data = {"choices": [{"message": {"content": "a medi"}, "finish_reason": "length"}]}
    with pytest.raises(PasoTruncado):
        _texto_o_truncado(data, "openai_compat")


def test_openai_compat_completo_pasa():
    data = {"choices": [{"message": {"content": "entero"}, "finish_reason": "stop"}]}
    assert _texto_o_truncado(data, "openai_compat") == "entero"


def test_ollama_truncado_falla():
    data = {"message": {"content": "a medi"}, "done_reason": "length"}
    with pytest.raises(PasoTruncado):
        _texto_o_truncado(data, "ollama")


def test_gemini_truncado_falla():
    data = {"candidates": [{"finishReason": "MAX_TOKENS",
                            "content": {"parts": [{"text": "a medi"}]}}]}
    with pytest.raises(PasoTruncado):
        _texto_o_truncado(data, "gemini")
```

- [ ] **Step 2: Correrlo y verlo ROJO**

Run: `PYTHONPATH=.:las_manos python -m pytest -v jacobs/_truncado_falla_test.py`
Expected: FAIL con `ImportError` — no existen ni `_texto_o_truncado` ni `PasoTruncado`.

- [ ] **Step 3: Implementar el lector único**

En `jacobs/executor.py`:

```python
class PasoTruncado(Exception):
    """El proveedor corto la salida por tope de longitud. Entregar el texto a
    medias es peor que fallar: el paso siguiente construye sobre una frase
    cortada y nadie se entera. Fallo cerrado, y el pipeline queda continuable."""
    codigo = "paso_truncado"


# Cada proveedor nombra el corte distinto. Un solo lugar que lo sepa.
_CORTE_POR_LONGITUD = {
    "openai_compat": ("length",),
    "ollama": ("length",),
    "gemini": ("MAX_TOKENS",),
}


def _texto_o_truncado(data: dict, transporte: str) -> str:
    if transporte == "openai_compat":
        eleccion = (data.get("choices") or [{}])[0]
        razon = eleccion.get("finish_reason")
        texto = (eleccion.get("message") or {}).get("content", "")
    elif transporte == "ollama":
        razon = data.get("done_reason")
        texto = (data.get("message") or {}).get("content", "")
    elif transporte == "gemini":
        candidato = (data.get("candidates") or [{}])[0]
        razon = candidato.get("finishReason")
        partes = (candidato.get("content") or {}).get("parts") or [{}]
        texto = partes[0].get("text", "")
    else:
        raise ValueError(f"transporte sin lector de truncado: {transporte}")

    if razon in _CORTE_POR_LONGITUD.get(transporte, ()):
        raise PasoTruncado(
            f"{transporte} corto la salida por longitud ({razon}); "
            f"{len(texto)} caracteres entregados"
        )
    return texto
```

- [ ] **Step 4: Usarlo en los tres transportes**

Reemplazar la extracción de texto en `_invoke_openai_compat` (executor.py:395),
`_invoke_ollama` (executor.py:456) y `_invoke_gemini` (executor.py:305 y el retry de 324-335)
por `_texto_o_truncado(data, "<transporte>")`.

- [ ] **Step 5: Correr los tests y la suite de ejecutor**

Run: `PYTHONPATH=.:las_manos python -m pytest -v jacobs/_truncado_falla_test.py tests/test_dispatch_step_reroute.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add jacobs/executor.py jacobs/_truncado_falla_test.py
git commit -m "fix(jacobs): un paso cortado por longitud falla en vez de entregar a medias"
```

---

## Task 4: El paso árbitro, siempre Thot

El informe encontró cinco hojas de ruta que no convergían. Falta quien decida. El último
paso recibe todo y produce **una decisión y un plan donde cada punto cita el paso que lo
sostiene**. Lo que no tiene paso que lo respalde no entra.

Sala limpia: **si el plan pone a Thot también como productor, el plan se rechaza al
crearse.** Ya existe `_check_cleanroom` (`plan.py:291-314`) y `PlanRejected` (`plan.py:273`):
esto es una regla más en ese mismo lugar, no una máquina nueva.

**Files:**
- Modify: `jacobs/plan.py` — `_check_arbitro` nuevo, llamado desde `build()` junto a los otros gates
- Test: `jacobs/_arbitro_test.py` (nuevo)

**Interfaces:**
- Consumes: `PlanRejected` (`plan.py:273`), `Step` con `depends_on` ya encadenado (Task 2).
- Produces: todo plan de 2+ pasos termina en un paso `facet="thot"`,
  `capability="text_generation"`, con `depends_on` = todos los índices anteriores.

- [ ] **Step 1: Escribir el test que falla**

```python
# jacobs/_arbitro_test.py
import pytest
from jacobs.plan import PlanBuilder, PlanRejected


def test_el_plan_termina_en_un_arbitro_thot():
    pasos = PlanBuilder._con_arbitro([
        {"facet": "jax_local", "capability": "text_generation", "prompt": "uno", "depends_on": []},
        {"facet": "ada", "capability": "text_generation", "prompt": "dos", "depends_on": [0]},
    ])
    assert pasos[-1]["facet"] == "thot"
    assert pasos[-1]["depends_on"] == [0, 1]
    assert "cita" in pasos[-1]["prompt"].lower()


def test_thot_productor_rechaza_el_plan():
    """El que produce no arbitra. Es rechazo al crearse, no un aviso."""
    with pytest.raises(PlanRejected):
        PlanBuilder._con_arbitro([
            {"facet": "thot", "capability": "text_generation", "prompt": "uno", "depends_on": []},
        ])


def test_un_plan_de_un_solo_paso_no_gana_arbitro():
    """Con un solo productor no hay nada que arbitrar."""
    pasos = PlanBuilder._con_arbitro([
        {"facet": "jax_local", "capability": "text_generation", "prompt": "uno", "depends_on": []},
    ])
    assert len(pasos) == 1
```

- [ ] **Step 2: Correrlo y verlo ROJO**

Run: `PYTHONPATH=.:las_manos python -m pytest -v jacobs/_arbitro_test.py`
Expected: FAIL — `_con_arbitro` no existe.

- [ ] **Step 3: Implementar**

En `jacobs/plan.py`, dentro de `PlanBuilder`:

```python
    FACETA_ARBITRO = "thot"

    PROMPT_ARBITRO = (
        "Recibiste la salida de todos los pasos anteriores. Produci UNA decision "
        "y UN plan. Cada punto del plan cita el paso que lo sostiene, con el "
        "formato [paso N]. Lo que no tenga un paso que lo respalde NO entra al "
        "plan: decilo como pendiente sin fuente, no como conclusion."
    )

    @staticmethod
    def _con_arbitro(pasos: list[dict]) -> list[dict]:
        """Sala limpia: el que produce no arbitra."""
        if any(p.get("facet") == PlanBuilder.FACETA_ARBITRO for p in pasos):
            raise PlanRejected(
                f"{PlanBuilder.FACETA_ARBITRO} no puede producir y arbitrar el "
                f"mismo plan: el arbitro juzga lo que otros produjeron."
            )
        if len(pasos) < 2:
            return pasos
        return pasos + [{
            "facet": PlanBuilder.FACETA_ARBITRO,
            "capability": "text_generation",
            "prompt": PlanBuilder.PROMPT_ARBITRO,
            "depends_on": list(range(len(pasos))),
        }]
```

Llamarlo en `build()` después de `_parse_plan_json` y **antes** de `_check_facets`
(plan.py:527), para que el árbitro también pase por los gates de gobernanza.

- [ ] **Step 4: Correr los tests y verlos verdes**

Run: `PYTHONPATH=.:las_manos python -m pytest -v jacobs/_arbitro_test.py tests/test_plan_validation.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jacobs/plan.py jacobs/_arbitro_test.py
git commit -m "feat(jacobs): el plan termina en un arbitro Thot que cita sus fuentes"
```

---

## Task 5: Permiso de lectura para los pasos

Los modelos pedían leer y no podían, así que respondían de memoria — de ahí salen las
invenciones. La cadena de autoridad ya existe entera: capability `file_read`
(`plan.py:342`), gate de plan (`plan.py:487-493`), gate de despacho (`executor.py:746`) y
el jail real por tool-call en `las_manos/motor_registry/tool_authority.py:166`, que confina
toda ruta a `WORKSPACE_ROOT` con `resolve_jailed_path`.

Lo que falta es solo la **habilitación**: que `jacobs` esté en `allowed_callers` de
`file_read`. **No se toca el jail ni se agrega escritura.**

**Files:**
- Modify: fila de `capability`/`capability_motor` en la base (migración de jax-platform)
- Test: `tests/test_lectura_de_pasos_db.py` (nuevo, con DB)

- [ ] **Step 1: Escribir el test que falla**

```python
# tests/test_lectura_de_pasos_db.py
import pytest


@pytest.mark.asyncio
async def test_jacobs_puede_pedir_file_read():
    from las_manos.motor_registry.tool_authority import cargar_catalogo
    catalogo = await cargar_catalogo()
    assert "jacobs" in catalogo["file_read"]["allowed_callers"]


@pytest.mark.asyncio
async def test_la_escritura_sigue_cerrada_para_jacobs():
    """Escritura es otra ronda: necesita clon en el jail, rama propia y
    compuerta humana. Habilitarla antes de esos tres contratos es Principio IX."""
    from las_manos.motor_registry.tool_authority import cargar_catalogo
    catalogo = await cargar_catalogo()
    assert "jacobs" not in catalogo["file_write"]["allowed_callers"]
```

- [ ] **Step 2: Correrlo con DB y verlo ROJO en el primero, VERDE en el segundo**

Run: el comando de jax con DB sobre `tests/test_lectura_de_pasos_db.py`.
Expected: el primero FAIL, el segundo PASS. El segundo es la baranda: si también fuera
rojo, alguien ya habría abierto la escritura.

- [ ] **Step 3: Migración que agrega el permiso de lectura**

En `jax-platform/backend/db/migrations.py`, siguiendo el patrón de las migraciones
idempotentes ya presentes (`INSERT IGNORE` en cada arranque, no un `UPDATE` a mano).

- [ ] **Step 4: Verificar el jail con un caso real**

Ejercitar `resolve_jailed_path` con una ruta fuera de `WORKSPACE_ROOT` (`../../etc/passwd`)
y comprobar que la rechaza. Un permiso nuevo sin ejercitar su peor caso es un freno sin
prueba (Principio VII).

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(jacobs): los pasos pueden LEER el workspace; la escritura sigue cerrada"
```

---

## Task 6: Aviso por Telegram al terminar (jax)

`send_telegram_alert(message: str) -> dict` ya existe en `jacobs/reaper.py:82`, no lanza
nunca y degrada a `ok=False` si faltan `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`. Se reusa.

**Límite que queda documentado, no escondido:** `TELEGRAM_CHAT_ID` es **uno solo** para
todo el sistema. Con más de un usuario, el aviso de cualquiera llegaría al mismo chat. Por
eso el mensaje lleva **solo nombre, estado y enlace — nunca contenido del pipeline**. El
día que haya usuarios de verdad, el canal por usuario es otra ronda.

**Files:**
- Create: `jacobs/aviso.py`, `jacobs/_aviso_test.py`
- Modify: `jacobs/executor.py` (al terminar el pipeline)

- [ ] **Step 1: Test que falla**

```python
# jacobs/_aviso_test.py
import asyncio
from jacobs.aviso import mensaje_de_fin


def test_el_mensaje_no_lleva_contenido_del_pipeline():
    """TELEGRAM_CHAT_ID es uno solo: el contenido no sale por ahi."""
    texto = mensaje_de_fin(pipeline_id="abc-123", nombre="ERP",
                           estado="completed", salida="SECRETO DEL CLIENTE")
    assert "SECRETO DEL CLIENTE" not in texto
    assert "abc-123" in texto and "ERP" in texto
```

- [ ] **Step 2: Correrlo y verlo ROJO** (no existe `jacobs/aviso.py`)

- [ ] **Step 3: Implementar `mensaje_de_fin` y el disparo**

El disparo va donde el pipeline pasa a `completed`/`failed` en `_correr_pipeline`
(`executor.py:1166`), envuelto en `try/except` fail-soft: **un aviso que falla no cambia el
estado del pipeline**, pero queda en el log — no se traga en silencio.

- [ ] **Step 4: Tests verdes + commit**

```bash
git commit -m "feat(jacobs): avisa por Telegram cuando un pipeline termina"
```

---

## Task 7: El listado del historial (jax-platform)

`GET /api/pipelines` (`backend/api/pipelines.py:775`) ya lista por dueño con
`WHERE user_id=%s AND tenant_id=%s AND owner_ack_at IS NOT NULL`, orden
`created_at DESC`, tope `LISTA_PIPELINES_MAX`. Le faltan duración y costo, y le sobra el
tope fijo para un historial.

**Files:**
- Modify: `backend/api/pipelines.py:691-695` (SQL), `:775` (handler)
- Test: `backend/tests/test_historial_pipelines.py` (nuevo)

- [ ] **Step 1: Test que falla** — el listado devuelve `duracion_s`, `costo_usd` y pagina.
- [ ] **Step 2: Verlo ROJO.**
- [ ] **Step 3: Implementar.** El costo sale del registro de uso por `trace_id`, no se
      recalcula en el frontend.
- [ ] **Step 4: `EXPLAIN` sobre la consulta REAL.**

```bash
EXPLAIN SELECT ... FROM jacobs_pipelines WHERE user_id=? AND tenant_id=? ... ORDER BY created_at DESC
```

El test tiene que **mirar el SQL y el plan**, no solo el tiempo: un índice que existe no es
un índice que se usa, y una base de tests persistente esconde un índice que ya no se crea.
Sin `filesort` ni `temporary` en el camino caliente.

- [ ] **Step 5: Subir el piso** de backend con DB (2340) en el mismo commit.
- [ ] **Step 6: Commit.**

---

## Task 8: El correo al terminar (jax-platform)

El poller ya detecta la transición en `backend/jax_engine/state.py:233`
(`if updated.status in ("completed", "failed")`), y ahí tiene `tenant_id` y `user_id`.

El correo copia el patrón que ya funciona en recuperación de contraseña
(`backend/api/auth.py:387-525`): `BackgroundTasks` + `asyncio.to_thread`, porque
`smtplib` **es bloqueante y no puede correr en el event loop**.

**Files:**
- Create: `backend/aviso_pipeline.py`, `backend/tests/test_aviso_pipeline.py`
- Modify: `backend/jax_engine/state.py:233`

- [ ] **Step 1: Test que falla** — al pasar a `completed` se encola un envío, y el envío
      **no ocurre dentro del request**.
- [ ] **Step 2: Verlo ROJO.**
- [ ] **Step 3: Implementar**, con `smtp_config.enviar()` en `asyncio.to_thread`.
- [ ] **Step 4:** un correo que falla **no cambia el estado del pipeline** y queda logueado.
- [ ] **Step 5: Subir los pisos** que corresponda. **Commit.**

---

## Task 9: La sección Historial en la Mesa

Hoy el resultado se vuelca al chat como mensajes (`useJaxStore.js:461-523`) y se va hacia
arriba. Por eso Fernando no supo qué hacer cuando terminó.

El detalle **no necesita endpoint nuevo**: `GET /api/pipelines/{id}/results`
(`pipelines.py:870`) ya trae el plan completo con `input.prompt`, `output_ref`, estado y
tiempos, con chequeo de dueño y 404 al que no lo es.

**Files:**
- Create: `frontend/src/pages/Historial.jsx`,
  `frontend/src/components/historial/DetallePipeline.jsx`, sus tests
- Modify: `useJaxStore.js` (acción `cargarHistorial`), `App.jsx` (ruta),
  `i18n/es.js` + `en.js`

- [ ] **Step 1: Tests de vitest que fallan** — el detalle muestra, por paso: faceta,
      **modelo real**, **prompt exacto**, **salida completa**, duración y de qué pasos dependía.
- [ ] **Step 2: Verlos ROJOS.**
- [ ] **Step 3: Implementar**, siguiendo el patrón de `pages/admin/Admin*.jsx`
      (tabla + fetch + filas) y la navegación de `AdminSidebar.jsx` (`NAV_ITEMS`).
- [ ] **Step 4: i18n es/en completo** — `paridad.test.js` lo exige.
- [ ] **Step 5: Tema** — clases semánticas, sin `dark:`. Probar en claro y oscuro.
- [ ] **Step 6: Subir el piso de vitest** (784) en el mismo commit. **Commit.**

---

## Task 10: Medir la carga, y volver a medir U5

Sin número medido no hay GO. Y el p95 de 2,99 ms que quedó registrado para U5 medía a
FastAPI devolviendo un literal, no al servicio: es una VERDAD OPERACIONAL falsa y hay que
reemplazarla.

- [ ] **Step 1: Peor caso, no el feliz** — el pipeline con más pasos y más salida, el
      usuario con más historial, concurrencia real (c=25, como las rondas anteriores).
- [ ] **Step 2: Medir** el listado del historial y el detalle.
- [ ] **Step 3: Verificar que se está midiendo el servicio** y no un literal — el defecto
      exacto que tuvo U5. Confirmar que la respuesta trae datos de verdad.
- [ ] **Step 4: Escribir el número en la Biblioteca** con fecha: peticiones por segundo,
      p95, y con cuántos usuarios simultáneos empieza a degradarse.
- [ ] **Step 5: Commit.**

---

## Verificación de cierre

La ronda no se da por buena con los tests en verde. Se da por buena cuando:

1. Un pipeline nuevo de tres pasos **encadena solo**, y sus `started_at` no se solapan.
2. Aparece en el historial y su detalle muestra prompt exacto, salida completa y **modelo real**.
3. Llegan **el correo y el Telegram** con el enlace que abre ese detalle.
4. El paso árbitro produce una decisión donde **cada punto cita su paso**.
5. Los tests de las tareas 1, 2 y 3 **se vieron rojos contra el código viejo**.
6. Hay un número de carga medido, con fecha, en la Biblioteca.
7. Fernando lo vio en vivo, en claro y en oscuro.

En memoria de Jairo Urbina.
