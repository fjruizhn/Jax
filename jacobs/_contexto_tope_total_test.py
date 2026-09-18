"""IMPORTANTE 2 (revisión final 2026-09-18, tanda historial-y-arreglos-de-pipeline):
el árbitro no tiene presupuesto TOTAL de entrada.

EL DEFECTO. `MAX_DEP_CONTEXT_CHARS` (executor.py) es un tope POR DEPENDENCIA,
y `_build_context_input` lo aplica una vez por cada dep declarada en
`depends_on` sin sumar nunca el TOTAL. El árbitro (Task 4) depende de TODOS
los pasos anteriores -- con 10 pasos de salida larga son hasta 540.000
caracteres armados en un solo prompt; con 20, 1,1 millones. El síntoma real
es un 400 del proveedor en el paso final, con todo el trabajo anterior ya
pagado.

EL ARREGLO. Un tope TOTAL (`MAX_TOTAL_DEP_CONTEXT_CHARS`) sobre la SUMA de
`previous_outputs`. Cuando hay que recortar, el recorte es HONESTO y
VISIBLE: el dep que pierde contenido queda con `truncated=True` y una nota
explícita en su `summary` -- nunca desaparece de la lista en silencio (el
árbitro necesita saber que esa fuente EXISTIÓ aunque no la haya visto
completa; si desapareciera, citaría solo lo que vio sin saber que hubo más).

Corre con:
  cd /home/fruiz/worktrees/jax-historial && PYTHONPATH=.:las_manos python -m pytest -v jacobs/_contexto_tope_total_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import json

from jacobs.executor import MAX_DEP_CONTEXT_CHARS, MAX_TOTAL_DEP_CONTEXT_CHARS, _build_context_input, _enrich_prompt
from jacobs.models import Pipeline, Step


def _pipeline_con_deps_largas(n_deps: int, largo_por_dep: int) -> tuple[Step, Pipeline]:
    """`n_deps` pasos previos, cada uno con una salida de `largo_por_dep`
    caracteres, más un paso árbitro que depende de TODOS ellos (mismo patrón
    que Task 4: depends_on explícito -> modo `full`, contexto COMPLETO por
    dep, no el resumen de 500 chars)."""
    deps_steps = [
        Step(pipeline_id="p1", step_index=j, facet=f"f{j}", capability="research",
             input={}, depends_on=[])
        for j in range(n_deps)
    ]
    step_arbitro = Step(
        pipeline_id="p1", step_index=n_deps, facet="thot", capability="critique",
        input={"prompt": "arbitra"}, depends_on=list(range(n_deps)),
    )
    plan = deps_steps + [step_arbitro]
    context = {"objective": "o"}
    for j in range(n_deps):
        context[f"step_{j}_ref"] = "inline:" + json.dumps({"result": "x" * largo_por_dep})
    pipeline = Pipeline(
        pipeline_id="p1", name="n", invoked_by="plataforma", mode="autonomous",
        plan=plan, context=context, run_epoch=0,
    )
    return step_arbitro, pipeline


def test_el_tope_total_existe_y_es_mayor_que_el_tope_por_dep():
    """El tope agregado tiene que ser un múltiplo chico del tope por-dep, no
    infinito (si no, sigue sin haber techo) ni menor que un solo dep completo
    (rompería incluso UNA dependencia sana)."""
    assert MAX_TOTAL_DEP_CONTEXT_CHARS > MAX_DEP_CONTEXT_CHARS
    assert MAX_TOTAL_DEP_CONTEXT_CHARS < MAX_DEP_CONTEXT_CHARS * 100


def test_la_suma_de_previous_outputs_nunca_supera_el_tope_total():
    """4 deps de 60.000 chars cada una (el tope POR DEP, así que ninguna se
    trunca a nivel individual) suman 240.000 -- por encima de cualquier tope
    total razonable. La suma final tiene que quedar acotada."""
    step_arbitro, pipeline = _pipeline_con_deps_largas(4, MAX_DEP_CONTEXT_CHARS)
    ctx = _build_context_input(step_arbitro, pipeline)
    total = sum(len(p["summary"]) for p in ctx["previous_outputs"])
    assert total <= MAX_TOTAL_DEP_CONTEXT_CHARS, (
        f"el contexto de un solo step sumó {total} caracteres, por encima del "
        f"tope total ({MAX_TOTAL_DEP_CONTEXT_CHARS}) -- exactamente el escenario "
        f"que produce un 400 del proveedor en el último paso"
    )


def test_ninguna_dependencia_desaparece_en_silencio():
    """El recorte por tope TOTAL no puede borrar una dependencia de la lista:
    el árbitro necesita saber que esa fuente EXISTIÓ, aunque no la haya visto
    completa -- si desapareciera, citaría solo lo que vio sin saber que hubo
    más (honesto Y visible, no un truncado silencioso)."""
    step_arbitro, pipeline = _pipeline_con_deps_largas(6, MAX_DEP_CONTEXT_CHARS)
    ctx = _build_context_input(step_arbitro, pipeline)
    assert len(ctx["previous_outputs"]) == 6
    assert [p["step_index"] for p in ctx["previous_outputs"]] == [0, 1, 2, 3, 4, 5]


def test_las_deps_recortadas_por_el_tope_total_quedan_marcadas_truncated():
    """Una dep que individualmente NO excedía MAX_DEP_CONTEXT_CHARS, pero que
    el tope TOTAL obligó a recortar, tiene que quedar con truncated=True --
    el recorte es visible para el consumidor, no silencioso."""
    step_arbitro, pipeline = _pipeline_con_deps_largas(4, MAX_DEP_CONTEXT_CHARS)
    ctx = _build_context_input(step_arbitro, pipeline)
    recortadas = [p for p in ctx["previous_outputs"] if p["truncated"]]
    assert recortadas, "con 4 deps de 60.000 chars (240.000 total) tiene que haber recorte"


def test_el_prompt_final_avisa_del_recorte_por_tope_total():
    """`_enrich_prompt` ya marca [TRUNCADO] por dep individual -- confirma que
    la marca SIGUE apareciendo cuando el recorte lo disparó el tope TOTAL, no
    el tope por-dep (mismo mecanismo, honesto para el consumidor del prompt)."""
    step_arbitro, pipeline = _pipeline_con_deps_largas(4, MAX_DEP_CONTEXT_CHARS)
    ctx = _build_context_input(step_arbitro, pipeline)
    prompt = _enrich_prompt(ctx)
    assert "TRUNCADO" in prompt


def test_con_una_sola_dependencia_grande_no_hay_recorte_de_mas():
    """Caso base: una sola dep, con contenido dentro del tope por-dep. El
    tope TOTAL no puede recortar de más lo que ya entra sin ayuda."""
    step_arbitro, pipeline = _pipeline_con_deps_largas(1, 1_000)
    ctx = _build_context_input(step_arbitro, pipeline)
    assert ctx["previous_outputs"][0]["truncated"] is False
    assert ctx["previous_outputs"][0]["summary"] == "x" * 1_000
