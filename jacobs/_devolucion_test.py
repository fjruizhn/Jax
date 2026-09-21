#!/usr/bin/env python3
"""jacobs/devolucion.py -- el árbitro devuelve (spec 2026-09-18-arbitro-devuelve-design).

**La idea central del spec, verificada acá con mocks:** una devolución NO es
un mecanismo nuevo de reintento. Es `store.continuar_transaccion` -- la MISMA
función que `jacobs/continuar.py` usa para revivir un pipeline aborted/
expired -- invocada con el paso señalado (y todo lo que depende de él)
invalidado y la crítica del árbitro inyectada en su prompt. Estos tests no
tocan la base: mockean `jacobs.store` y `jacobs.prevuelo.prevuelo`, el mismo
patrón que `jacobs/_arbitro_test.py` y `jacobs/_menu_sin_arbitro_test.py` ya
usan para ejercitar lógica de Jacobs sin DB ni red.

Cubre los cinco escenarios de "cómo se sabe que funcionó" (spec §5):
  1. test_devuelve_a_ada_con_la_critica_inyectada_y_afectados_correctos
  2. test_veredicto_sin_estructura_no_dispara_devolucion
  3. test_al_tercer_intento_para_y_avisa_con_las_dos_versiones
  4. test_costo_que_no_cabe_no_ocurre / test_sin_presupuesto_persistido_no_ocurre
  5. (este archivo entero corre rojo contra el código de hoy: `jacobs.devolucion`
     todavía no existe -- ModuleNotFoundError, verificado antes de escribir
     la implementación)

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import ExitStack
from decimal import Decimal
from unittest.mock import AsyncMock, patch

from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus
from jacobs.plan import PlanBuilder
from jacobs.prevuelo_reglas import CostoPaso, Veredicto


# ---------------------------------------------------------------------------
# Un plan con la MISMA forma que el pipeline real e570ac1c-8cae-4423-b397-
# d354f30b4328: 6 productores (0..5) + árbitro (6). El step 4 ("ada") es el
# que propuso sintaxis PostgreSQL sobre un destino MariaDB; el 5 ("jekyll")
# depende de él; el árbitro depende de TODOS.
# ---------------------------------------------------------------------------

def _plan_con_arbitro() -> list[Step]:
    deps = {0: [], 1: [0], 2: [0, 1], 3: [0, 1, 2], 4: [0, 1], 5: [4], 6: [0, 1, 2, 3, 4, 5]}
    facets = {0: "jax_local", 1: "kimi", 2: "hipatia", 3: "jekyll", 4: "ada", 5: "jekyll", 6: "thot"}
    pasos = []
    for i in range(7):
        cap = PlanBuilder.CAPABILITY_ARBITRO if i == 6 else "architecture_review"
        pasos.append(Step(
            step_id=f"s{i}", pipeline_id="p1", step_index=i, facet=facets[i],
            capability=cap, status=StepStatus.completed, depends_on=deps[i],
            input={"prompt": f"prompt original del paso {i}"},
        ))
    return pasos


def _pipeline(**over) -> Pipeline:
    base = dict(
        pipeline_id="p1", name="t", invoked_by="plataforma", mode="supervised",
        status=PipelineStatus.running, plan=_plan_con_arbitro(),
        context={f"step_{i}_ref": f"inline:{{\"success\": true}}" for i in range(7)},
        run_epoch=1, devoluciones=0, costo_max_aceptado_usd=Decimal("5.00"),
        created_at=time.time(), updated_at=time.time(),
    )
    base.update(over)
    return Pipeline(**base)


def _bloque_devolver(paso: int, motivo: str = "usaste BIGSERIAL y TIMESTAMPTZ; el destino es MariaDB") -> str:
    return "inline:" + (
        f'{{"success": true, "facet": "thot", "model": "_test_modelo_thot", '
        f'"result": "## Decisión\\n\\nNo aprobar [paso {paso}]. '
        f'```veredicto\\n{{\\"decision\\": \\"devolver\\", \\"paso\\": {paso}, '
        f'\\"motivo\\": \\"{motivo}\\", \\"cita\\": \\"[paso {paso}]\\"}}\\n```"}}'
    )


def _ok_veredicto(usd: str = "1.00") -> Veredicto:
    costo = CostoPaso(4, "ada", "_test_modelo_ada", 1, 100, 8192, Decimal(usd), "acotado")  # modelo sintético: CostoPaso no lee el catálogo, sólo lo transporta (Principio IV, 2026-09-21)
    return Veredicto(ok=True, violaciones=(), costo_max_usd=Decimal(usd), pasos_costo=(costo,), sondeadas=())


def _correr(coro):
    return asyncio.run(coro)


def _mocks(*, tope=2, transaccion=2, veredicto_costo=None, vigente=(1, PipelineStatus.running),
           gastado=(Decimal("0"), False), pendientes_uso=0, pendientes_uso_entradas=None):
    """Mockea todo lo que `evaluar_y_devolver` toca fuera de sí mismo:
    jacobs.store (event_append/get_tope_devoluciones/continuar_transaccion/
    pipeline_epoca_y_status/costo_gastado_pipeline) y jacobs.prevuelo.prevuelo,
    más el peek de la cola durable de uso. Nunca la base real.

    `vigente`: lo que devuelve `store.pipeline_epoca_y_status` -- el status
    REAL de la fila, no el de `pipeline.status` en memoria (que routes.py
    nunca actualiza a running, ver Ronda de arreglo 1). Default (1, running):
    coincide con el `run_epoch=1` de `_pipeline()`.

    `gastado`: (Decimal, hay_costo_desconocido) que devuelve
    `store.costo_gastado_pipeline` -- Ronda de arreglo 2, §3.4: "lo que
    queda", no el techo completo. Default: nada gastado, nada incierto.

    `pendientes_uso`/`pendientes_uso_entradas`: la cola durable de uso
    (jax.core.cola_uso) -- por default vacía (0 pendientes, sin entradas)."""
    pila = ExitStack()
    m = {}
    m["evento"] = pila.enter_context(patch("jacobs.devolucion.store.event_append", AsyncMock()))
    m["tope"] = pila.enter_context(
        patch("jacobs.devolucion.store.get_tope_devoluciones", AsyncMock(return_value=tope)))
    m["vigente"] = pila.enter_context(
        patch("jacobs.devolucion.store.pipeline_epoca_y_status", AsyncMock(return_value=vigente)))
    m["gastado"] = pila.enter_context(
        patch("jacobs.devolucion.store.costo_gastado_pipeline", AsyncMock(return_value=gastado)))
    m["cola_contar"] = pila.enter_context(
        patch("jacobs.devolucion._uso_contar_pendientes", AsyncMock(return_value=pendientes_uso)))
    m["cola_leer"] = pila.enter_context(
        patch("jacobs.devolucion._uso_leer_pendientes",
              AsyncMock(return_value=pendientes_uso_entradas or [])))
    m["tx"] = pila.enter_context(
        patch("jacobs.devolucion.store.continuar_transaccion", AsyncMock(return_value=transaccion)))
    m["prevuelo"] = pila.enter_context(
        patch("jacobs.devolucion.prevuelo", AsyncMock(return_value=veredicto_costo or _ok_veredicto())))
    return pila, m


# ---------------------------------------------------------------------------
# pasos_afectados: la unidad pura de invalidación transitiva.
# ---------------------------------------------------------------------------

def test_pasos_afectados_incluye_al_dependiente_directo_y_al_arbitro():
    from jacobs.devolucion import pasos_afectados

    plan = _plan_con_arbitro()
    assert pasos_afectados(plan, 4) == {4, 5, 6}


def test_pasos_afectados_de_un_paso_sin_dependientes_es_solo_el_mismo():
    from jacobs.devolucion import pasos_afectados

    plan = _plan_con_arbitro()
    assert pasos_afectados(plan, 3) == {3, 6}  # solo el árbitro depende de 3, nadie más


def test_pasos_afectados_del_paso_0_arrastra_casi_todo():
    from jacobs.devolucion import pasos_afectados

    plan = _plan_con_arbitro()
    # 0 sostiene a 1,2,3,4 (directa o indirectamente) y al árbitro.
    assert pasos_afectados(plan, 0) == {0, 1, 2, 3, 4, 5, 6}


# ---------------------------------------------------------------------------
# 1. El caso real: devuelve a ada, con la crítica viajando en el contexto.
# ---------------------------------------------------------------------------

def test_devuelve_a_ada_con_la_critica_inyectada_y_afectados_correctos():
    from jacobs.devolucion import RESULTADO_DEVUELTO, evaluar_y_devolver

    pipeline = _pipeline(context={
        **{f"step_{i}_ref": "inline:{}" for i in range(6)},
        "step_6_ref": _bloque_devolver(4),
    })
    pila, m = _mocks()
    with pila:
        resultado, payload = _correr(evaluar_y_devolver(pipeline))

    assert resultado == RESULTADO_DEVUELTO
    # La crítica viaja ADENTRO del contexto del paso rehecho (spec §3.2), no
    # solo en un evento que nadie más que un humano lee.
    prompt_devuelto = payload["plan"][4].input["prompt"]
    assert "prompt original del paso 4" in prompt_devuelto
    assert "BIGSERIAL" in prompt_devuelto
    assert "MariaDB" in prompt_devuelto
    assert "[paso 4]" in prompt_devuelto

    # Los pasos 4, 5 y 6 (afectados) se invalidan; el resto NO se toca.
    for i in (4, 5, 6):
        assert payload["plan"][i].status == StepStatus.pending
        assert payload["plan"][i].output_ref is None
    for i in (0, 1, 2, 3):
        assert payload["plan"][i].status == StepStatus.completed
    assert f"step_4_ref" not in payload["context"]
    assert f"step_5_ref" not in payload["context"]
    assert f"step_6_ref" not in payload["context"]
    for i in (0, 1, 2, 3):
        assert f"step_{i}_ref" in payload["context"]

    # Reusa continuar_transaccion (la MISMA maquinaria de Continuar), no un
    # mecanismo nuevo -- con aplicar_cupo=False (el pipeline ya está vivo,
    # no pide un lugar nuevo) e incrementar_devoluciones=True.
    m["tx"].assert_awaited_once()
    _, kwargs = m["tx"].call_args
    assert kwargs["evento_tipo"] == "PIPELINE_DEVUELTO"
    assert kwargs["aplicar_cupo"] is False
    assert kwargs["incrementar_devoluciones"] is True
    assert payload["run_epoch"] == 2


def test_devuelve_con_el_pipeline_construido_como_routes_py_lo_construye():
    """Ronda de arreglo 1 (CRÍTICO). `POST /jacobs/pipeline` (jacobs/routes.py
    ~596-616) arma el `Pipeline` SIN pasar `status=`, así que queda en el
    default de models.py (`pending`) -- y ese es el objeto que
    `background.add_task(run_pipeline, pipeline)` despacha. `_correr_pipeline`
    (jacobs/executor.py) actualiza la FILA a `running` pero nunca reasigna
    `pipeline.status`; el único lugar que lo hacía era continuar.py (vía
    /continue). Si `evaluar_y_devolver` comparara contra `pipeline.status` en
    vez de leer la fila vigente, este test (el camino de CREACIÓN, no de
    /continue) quedaría en RESULTADO_COMPLETAR aunque el veredicto sea
    'devolver' -- exactamente el final del caso real e570ac1c que esta ronda
    existe para cambiar."""
    from jacobs.devolucion import RESULTADO_DEVUELTO, evaluar_y_devolver

    pipeline = _pipeline(
        status=PipelineStatus.pending,  # EXACTO default de models.py, como routes.py lo construye
        context={
            **{f"step_{i}_ref": "inline:{}" for i in range(6)},
            "step_6_ref": _bloque_devolver(4),
        },
    )
    # La FILA real sí está running (la escribió _correr_pipeline) -- eso es
    # lo único que `evaluar_y_devolver` tiene que confiar.
    pila, m = _mocks(vigente=(pipeline.run_epoch, PipelineStatus.running))
    with pila:
        resultado, payload = _correr(evaluar_y_devolver(pipeline))

    assert resultado == RESULTADO_DEVUELTO
    m["vigente"].assert_awaited_once_with(pipeline.pipeline_id)
    # continuar_transaccion tiene que recibir el status VIGENTE (running),
    # nunca el 'pending' del objeto en memoria -- si no, su propio
    # SELECT...FOR UPDATE (store.py) lo compara contra la fila real y aborta.
    args, kwargs = m["tx"].call_args
    assert args[2] == PipelineStatus.running


def test_sin_running_vigente_no_devuelve_ni_escribe_nada():
    """Control (Principio VII): si la fila YA NO está running cuando se
    evalúa el veredicto (alguien la canceló, el kill switch la abortó a
    mitad de la última ola), NO se toma el atajo `aplicar_cupo=False` --
    ese atajo asume 'ya está vivo', y sin confirmarlo sería indistinguible
    de revivir un pipeline en un status que no ocupa cupo."""
    from jacobs.devolucion import RESULTADO_COMPLETAR, evaluar_y_devolver

    pipeline = _pipeline(context={
        **{f"step_{i}_ref": "inline:{}" for i in range(6)},
        "step_6_ref": _bloque_devolver(4),
    })
    pila, m = _mocks(vigente=(pipeline.run_epoch, PipelineStatus.aborted))
    with pila:
        resultado, payload = _correr(evaluar_y_devolver(pipeline))

    assert resultado == RESULTADO_COMPLETAR
    assert payload == {}
    m["tx"].assert_not_awaited()
    m["prevuelo"].assert_not_awaited()  # ni se llega a estimar costo


def test_sin_fila_vigente_no_devuelve():
    """El pipeline desapareció (borrado, id inválido) -- None de
    pipeline_epoca_y_status, mismo camino fail-closed."""
    from jacobs.devolucion import RESULTADO_COMPLETAR, evaluar_y_devolver

    pipeline = _pipeline(context={
        **{f"step_{i}_ref": "inline:{}" for i in range(6)},
        "step_6_ref": _bloque_devolver(4),
    })
    pila, m = _mocks(vigente=None)
    with pila:
        resultado, _ = _correr(evaluar_y_devolver(pipeline))
    assert resultado == RESULTADO_COMPLETAR
    m["tx"].assert_not_awaited()


def test_el_arbitro_no_reescribe_solo_devuelve():
    """§3.5: el árbitro nunca pone la versión buena. El plan devuelto por
    evaluar_y_devolver deja el paso 4 en `pending`, listo para que ADA (su
    faceta original) lo rehaga -- nunca cambia su `facet` a la del árbitro."""
    from jacobs.devolucion import evaluar_y_devolver

    pipeline = _pipeline(context={
        **{f"step_{i}_ref": "inline:{}" for i in range(6)},
        "step_6_ref": _bloque_devolver(4),
    })
    pila, m = _mocks()
    with pila:
        _, payload = _correr(evaluar_y_devolver(pipeline))
    assert payload["plan"][4].facet == "ada"


def test_carrera_perdida_no_reintenta_a_ciegas():
    """continuar_transaccion devuelve None cuando otro pedido cambió el
    pipeline mientras se decidía (época/status ya no coinciden). No hay
    reintento automático -- el que ganó la carrera decide."""
    from jacobs.devolucion import RESULTADO_COMPLETAR, evaluar_y_devolver

    pipeline = _pipeline(context={
        **{f"step_{i}_ref": "inline:{}" for i in range(6)},
        "step_6_ref": _bloque_devolver(4),
    })
    pila, m = _mocks(transaccion=None)
    with pila:
        resultado, _ = _correr(evaluar_y_devolver(pipeline))
    assert resultado == RESULTADO_COMPLETAR


# ---------------------------------------------------------------------------
# 2. Veredicto sin estructura válida -> no dispara devolución, queda registrado.
# ---------------------------------------------------------------------------

def test_veredicto_sin_estructura_no_dispara_devolucion():
    from jacobs.devolucion import RESULTADO_COMPLETAR, evaluar_y_devolver

    pipeline = _pipeline(context={
        **{f"step_{i}_ref": "inline:{}" for i in range(6)},
        "step_6_ref": (
            'inline:{"success": true, "facet": "thot", "result": '
            '"no aprobar [paso 4] [paso 5], sin bloque de veredicto"}'
        ),
    })
    pila, m = _mocks()
    with pila:
        resultado, payload = _correr(evaluar_y_devolver(pipeline))

    assert resultado == RESULTADO_COMPLETAR
    assert payload == {}
    m["tx"].assert_not_awaited()
    m["evento"].assert_awaited_once()
    args, _ = m["evento"].call_args
    assert args[1] == "VEREDICTO_SIN_ESTRUCTURA"


def test_veredicto_aprobar_completa_sin_evento_extra():
    from jacobs.devolucion import RESULTADO_COMPLETAR, evaluar_y_devolver

    pipeline = _pipeline(context={
        **{f"step_{i}_ref": "inline:{}" for i in range(6)},
        "step_6_ref": 'inline:{"success": true, "result": "todo bien ```veredicto\\n{\\"decision\\": \\"aprobar\\"}\\n```"}',
    })
    pila, m = _mocks()
    with pila:
        resultado, _ = _correr(evaluar_y_devolver(pipeline))
    assert resultado == RESULTADO_COMPLETAR
    m["tx"].assert_not_awaited()


def test_plan_de_un_solo_paso_sin_arbitro_completa_sin_tocar_la_base():
    """Un pipeline de 1 paso nunca gana árbitro (PlanBuilder._con_arbitro).
    evaluar_y_devolver no puede asumir que pipeline.plan[-1] es un árbitro:
    tiene que salir ANTES de tocar store/prevuelo."""
    from jacobs.devolucion import RESULTADO_COMPLETAR, evaluar_y_devolver

    solo = Step(step_id="s0", pipeline_id="p1", step_index=0, facet="jax_local",
                capability="text_generation", status=StepStatus.completed,
                input={"prompt": "x"})
    pipeline = _pipeline(plan=[solo], context={"step_0_ref": "inline:{}"})
    pila, m = _mocks()
    with pila:
        resultado, payload = _correr(evaluar_y_devolver(pipeline))
    assert resultado == RESULTADO_COMPLETAR
    assert payload == {}
    m["tope"].assert_not_awaited()
    m["tx"].assert_not_awaited()
    m["prevuelo"].assert_not_awaited()


# ---------------------------------------------------------------------------
# 3. Tope de vueltas: al tercer intento, para y avisa con las dos versiones.
# ---------------------------------------------------------------------------

def test_al_tercer_intento_para_y_avisa_con_las_dos_versiones():
    from jacobs.devolucion import RESULTADO_TOPE, evaluar_y_devolver

    pipeline = _pipeline(
        devoluciones=2,  # ya devolvió dos veces -- tope=2 (default de los mocks)
        context={
            **{f"step_{i}_ref": "inline:{}" for i in range(6)},
            "step_6_ref": _bloque_devolver(4),
        },
    )
    pila, m = _mocks(tope=2)
    with pila:
        resultado, payload = _correr(evaluar_y_devolver(pipeline))

    assert resultado == RESULTADO_TOPE
    assert payload["veredicto"].paso == 4
    m["tx"].assert_not_awaited()  # el árbitro no se concede una vuelta de más
    m["evento"].assert_awaited_once()
    args, kwargs = m["evento"].call_args
    assert args[1] == "DEVOLUCION_TOPE_ALCANZADO"
    payload_evento = args[2]
    assert payload_evento["devoluciones"] == 2
    assert payload_evento["tope"] == 2


def test_el_tope_es_configuracion_no_una_constante():
    """El tope sale de axioma_config (store.get_tope_devoluciones), no de un
    literal en jacobs/devolucion.py -- con tope=0 la PRIMERA devolución ya
    para, aunque el código no haya cambiado."""
    from jacobs.devolucion import RESULTADO_TOPE, evaluar_y_devolver

    pipeline = _pipeline(devoluciones=0, context={
        **{f"step_{i}_ref": "inline:{}" for i in range(6)},
        "step_6_ref": _bloque_devolver(4),
    })
    pila, m = _mocks(tope=0)
    with pila:
        resultado, _ = _correr(evaluar_y_devolver(pipeline))
    assert resultado == RESULTADO_TOPE


# ---------------------------------------------------------------------------
# 4. El dinero: una devolución que no cabe en el tope aceptado no ocurre.
# ---------------------------------------------------------------------------

def test_costo_que_no_cabe_no_ocurre():
    from jacobs.devolucion import RESULTADO_COMPLETAR, evaluar_y_devolver

    pipeline = _pipeline(
        costo_max_aceptado_usd=Decimal("0.10"),
        context={
            **{f"step_{i}_ref": "inline:{}" for i in range(6)},
            "step_6_ref": _bloque_devolver(4),
        },
    )
    caro = _ok_veredicto(usd="5.00")
    pila, m = _mocks(veredicto_costo=caro)
    with pila:
        resultado, payload = _correr(evaluar_y_devolver(pipeline))

    assert resultado == RESULTADO_COMPLETAR
    assert payload == {}
    m["tx"].assert_not_awaited()
    args, _ = m["evento"].call_args
    assert args[1] == "DEVOLUCION_SUPERA_PRESUPUESTO"


def test_sin_presupuesto_persistido_no_ocurre():
    """§3.4: "la devolución cabe dentro de lo que el humano ya aceptó, o no
    ocurre". Sin costo_max_aceptado_usd persistido no hay contra qué medir
    -- fail-closed, nunca se inventa un permiso de gasto."""
    from jacobs.devolucion import RESULTADO_COMPLETAR, evaluar_y_devolver

    pipeline = _pipeline(costo_max_aceptado_usd=None, context={
        **{f"step_{i}_ref": "inline:{}" for i in range(6)},
        "step_6_ref": _bloque_devolver(4),
    })
    pila, m = _mocks()
    with pila:
        resultado, payload = _correr(evaluar_y_devolver(pipeline))

    assert resultado == RESULTADO_COMPLETAR
    m["tx"].assert_not_awaited()
    m["prevuelo"].assert_not_awaited()
    args, _ = m["evento"].call_args
    assert args[1] == "DEVOLUCION_SIN_PRESUPUESTO"


# ---------------------------------------------------------------------------
# Ronda de arreglo 2: el presupuesto es LO QUE QUEDA, no el techo completo.
# Con tope=2, medir cada devolución contra el 100% del tope dejaba gastar
# hasta 3x lo aceptado (corrida original + dos rehechas, cada una "cabe"
# sola). Acá se prueba contra `store.costo_gastado_pipeline` (axioma_usage).
# ---------------------------------------------------------------------------

def test_lo_ya_gastado_se_descuenta_del_tope_no_del_techo_completo():
    """costo_max_aceptado_usd=5.00, ya gastado=4.50 -> queda 0.50. Un
    estimado de 1.00 "cabría" contra el techo completo (5.00) pero NO
    contra lo que queda (0.50): no puede ocurrir."""
    from jacobs.devolucion import RESULTADO_COMPLETAR, evaluar_y_devolver

    pipeline = _pipeline(
        costo_max_aceptado_usd=Decimal("5.00"),
        context={
            **{f"step_{i}_ref": "inline:{}" for i in range(6)},
            "step_6_ref": _bloque_devolver(4),
        },
    )
    estimado_de_1 = _ok_veredicto(usd="1.00")
    pila, m = _mocks(veredicto_costo=estimado_de_1, gastado=(Decimal("4.50"), False))
    with pila:
        resultado, payload = _correr(evaluar_y_devolver(pipeline))

    assert resultado == RESULTADO_COMPLETAR
    assert payload == {}
    m["tx"].assert_not_awaited()
    args, _ = m["evento"].call_args
    assert args[1] == "DEVOLUCION_SUPERA_PRESUPUESTO"
    assert args[2]["restante_usd"] == "0.500000"


def test_lo_que_queda_alcanza_la_devolucion_ocurre():
    """Control positivo del mismo mecanismo: costo_max_aceptado_usd=5.00,
    gastado=1.00 -> queda 4.00, y el estimado (1.00) cabe."""
    from jacobs.devolucion import RESULTADO_DEVUELTO, evaluar_y_devolver

    pipeline = _pipeline(
        costo_max_aceptado_usd=Decimal("5.00"),
        context={
            **{f"step_{i}_ref": "inline:{}" for i in range(6)},
            "step_6_ref": _bloque_devolver(4),
        },
    )
    pila, m = _mocks(gastado=(Decimal("1.00"), False))
    with pila:
        resultado, _ = _correr(evaluar_y_devolver(pipeline))
    assert resultado == RESULTADO_DEVUELTO


def test_costo_desconocido_en_axioma_usage_no_devuelve():
    """Trampa 1 (revisión de Fernando): una fila de ESTE pipeline con
    `cost_usd IS NULL` -- se cobró de verdad, el precio no se pudo resolver.
    Sumar ignorándola daría un 'gastado' más bajo que el real -- fail-closed,
    no se puede confirmar que queda presupuesto."""
    from jacobs.devolucion import RESULTADO_COMPLETAR, evaluar_y_devolver

    pipeline = _pipeline(context={
        **{f"step_{i}_ref": "inline:{}" for i in range(6)},
        "step_6_ref": _bloque_devolver(4),
    })
    pila, m = _mocks(gastado=(Decimal("0.20"), True))  # hay_costo_desconocido=True
    with pila:
        resultado, payload = _correr(evaluar_y_devolver(pipeline))

    assert resultado == RESULTADO_COMPLETAR
    assert payload == {}
    m["tx"].assert_not_awaited()
    m["prevuelo"].assert_not_awaited()
    args, _ = m["evento"].call_args
    assert args[1] == "DEVOLUCION_PRESUPUESTO_INCIERTO"
    assert "cost_usd" in args[2]["motivo_incierto"] or "desconocido" in args[2]["motivo_incierto"]


def test_uso_en_la_cola_durable_sin_drenar_no_devuelve():
    """Trampa 2: uso de ESTE pipeline todavía en el respaldo de
    jax.core.cola_uso, esperando que jax-platform lo drene a axioma_usage --
    mientras tanto tampoco aparece en la suma. Mismo fail-closed."""
    from jacobs.devolucion import RESULTADO_COMPLETAR, evaluar_y_devolver

    pipeline = _pipeline(context={
        **{f"step_{i}_ref": "inline:{}" for i in range(6)},
        "step_6_ref": _bloque_devolver(4),
    })
    pila, m = _mocks(
        pendientes_uso=3,
        pendientes_uso_entradas=[
            {"pipeline_id": "otro-pipeline"},
            {"pipeline_id": pipeline.pipeline_id},  # ESTE pipeline, sin drenar
            {"pipeline_id": "otro-mas"},
        ],
    )
    with pila:
        resultado, payload = _correr(evaluar_y_devolver(pipeline))

    assert resultado == RESULTADO_COMPLETAR
    assert payload == {}
    m["tx"].assert_not_awaited()
    m["prevuelo"].assert_not_awaited()
    args, _ = m["evento"].call_args
    assert args[1] == "DEVOLUCION_PRESUPUESTO_INCIERTO"


def test_cola_durable_con_otros_pipelines_no_bloquea_este():
    """Control (Principio VII): la cola tiene entradas, pero NINGUNA es de
    este pipeline -- no tiene que bloquear."""
    from jacobs.devolucion import RESULTADO_DEVUELTO, evaluar_y_devolver

    pipeline = _pipeline(context={
        **{f"step_{i}_ref": "inline:{}" for i in range(6)},
        "step_6_ref": _bloque_devolver(4),
    })
    pila, m = _mocks(
        pendientes_uso=2,
        pendientes_uso_entradas=[{"pipeline_id": "otro-1"}, {"pipeline_id": "otro-2"}],
    )
    with pila:
        resultado, _ = _correr(evaluar_y_devolver(pipeline))
    assert resultado == RESULTADO_DEVUELTO


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
