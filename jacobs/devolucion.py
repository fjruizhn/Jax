"""Jacobs -- el árbitro devuelve el trabajo mal hecho
(spec `docs/superpowers/specs/2026-09-18-arbitro-devuelve-design.md`).

**La idea central, en una línea** (spec §2): una devolución es un Continuar
apuntado a propósito. `jacobs/continuar.py` ya sabe invalidar pasos, reusar
los buenos, subir de época y pre-volar lo que falta -- este módulo NO
reconstruye nada de eso. `evaluar_y_devolver` decide QUÉ invalidar (el paso
que el árbitro señaló y todo lo que depende de él) y QUÉ inyectar (la
crítica, en el prompt del paso devuelto), y la escritura real es
`store.continuar_transaccion` -- la MISMA función, con `evento_tipo=
"PIPELINE_DEVUELTO"` y `aplicar_cupo=False` (ver el porqué en
`store._sql_pipeline_continuar`: el pipeline que se devuelve ya está vivo,
no está pidiendo un cupo nuevo).

**Cuándo se llama.** Después de que la ÚLTIMA ola de `_correr_pipeline`
(jacobs/executor.py) -- la del árbitro, que depende de todos los demás pasos
y por eso siempre corre sola, al final -- termina, y ANTES de que el
pipeline se marque `completed`. El árbitro es siempre el último step del
plan (`PlanBuilder._con_arbitro`): un plan de menos de 2 pasos no tiene
árbitro, y este módulo lo primero que hace es confirmarlo y salir sin tocar
la base si no lo hay -- ningún pipeline existente (sin árbitro) cambia de
comportamiento.

**Los cinco puntos del spec, y dónde viven acá:**
  §3.1 veredicto accionable, fallo cerrado -> jacobs/veredicto.py
       (parsear_veredicto); None acá dispara el evento
       VEREDICTO_SIN_ESTRUCTURA y NO devuelve.
  §3.2 la crítica viaja en el contexto -> `_inyectar_critica`, en el
       `input["prompt"]` del paso que se rehace (lo que el ejecutor arma
       como CONTEXTO real, ver executor.py::_build_context_input).
  §3.3 tope de vueltas, en configuración -> `store.get_tope_devoluciones()`,
       comparado contra `pipeline.devoluciones` (persistido).
  §3.4 el dinero: cabe en lo aceptado o no ocurre -> `pipeline.
       costo_max_aceptado_usd` (persistido, ver jacobs/store.py y
       jacobs/cupo.py::completar_reserva) contra el estimado de
       `jacobs.prevuelo.prevuelo` para los pasos afectados. Mismo criterio
       que ya usa `jacobs/continuar.py::continuar()` para su propio 409
       costo_supera_lo_aceptado: el estimado de ESTA corrida contra el tope
       COMPLETO, no un saldo neto de gasto acumulado -- Jacobs no lleva ese
       ledger por pipeline hoy, y este módulo no lo inventa (declarado, no
       verificado más allá de lo que continuar() ya hace).
  §3.5 devuelve, no reescribe -> `_inyectar_critica` solo TOCA el prompt del
       paso devuelto; nunca su `facet`, nunca su `output_ref`. Lo rehace la
       MISMA faceta que lo produjo la vez anterior.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import logging

from redaccion import recortar_redactado

from jacobs import store
from jacobs.executor import RefIlegible, _load_ref
from jacobs.models import Pipeline, PipelineStatus, Step, StepStatus
from jacobs.plan import PlanBuilder
from jacobs.prevuelo import prevuelo
from jacobs.prevuelo_reglas import formatear_usd
from jacobs.veredicto import DECISION_DEVOLVER, VeredictoArbitro, parsear_veredicto

logger = logging.getLogger("jacobs.devolucion")

#: Qué hizo evaluar_y_devolver con este veredicto. "completar" cubre TODOS
#: los caminos donde no hay devolución (aprobar, sin estructura, sin
#: presupuesto, costo que no cabe, carrera perdida): el pipeline sigue el
#: camino de siempre (jacobs/executor.py lo marca `completed`).
RESULTADO_COMPLETAR = "completar"
RESULTADO_DEVUELTO = "devuelto"
RESULTADO_TOPE = "tope"

#: Trunco antes de meter la prosa del árbitro en un evento de auditoría --
#: mismo criterio que redaccion.recortar_redactado usa en otros lugares de
#: Jacobs (jacobs/continuar.py), no un número inventado para acá.
_MAX_CHARS_EVENTO = 2000


def pasos_afectados(plan: list[Step], objetivo: int) -> set[int]:
    """`objetivo` y todo lo que depende de él, transitivamente -- SIEMPRE
    incluye al árbitro (depende de todos los pasos productores, así que
    cualquier `objetivo` < índice del árbitro lo arrastra). Es el "y lo que
    dependía de él" del spec §2: devolver el paso 4 sin invalidar el 5 (que
    LEYÓ la salida mala del 4) dejaría al 5 construyendo sobre una entrada
    que va a cambiar."""
    afectados = {objetivo}
    cambiado = True
    while cambiado:
        cambiado = False
        for paso in plan:
            if paso.step_index in afectados:
                continue
            if any(dep in afectados for dep in (paso.depends_on or [])):
                afectados.add(paso.step_index)
                cambiado = True
    return afectados


def _inyectar_critica(paso: Step, veredicto: VeredictoArbitro, intento: int) -> None:
    """§3.2 + §3.5: la crítica viaja EN el prompt del paso devuelto -- nunca
    se toca `paso.facet` (eso sería el árbitro reescribiendo, §3.5)."""
    previo = paso.input.get("prompt", "")
    critica = (
        f"CRÍTICA DEL ÁRBITRO (devolución {intento}): {veredicto.motivo} "
        f"(cita: {veredicto.cita}). Es EXACTAMENTE lo que tenés que corregir "
        f"en esta versión -- no repitas el mismo error."
    )
    paso.input["prompt"] = f"{previo}\n\n{critica}" if previo else critica


def _es_arbitro(pipeline: Pipeline) -> Step | None:
    """El último step del plan, si es el árbitro (PlanBuilder._con_arbitro lo
    agrega SIEMPRE al final, en todo plan de 2+ pasos). None en cualquier
    otro caso -- incluido un plan de 1 solo paso, que nunca lo tiene."""
    if len(pipeline.plan) < 2:
        return None
    ultimo = pipeline.plan[-1]
    if ultimo.capability != PlanBuilder.CAPABILITY_ARBITRO:
        return None
    return ultimo


async def evaluar_y_devolver(pipeline: Pipeline) -> tuple[str, dict]:
    """Se llama DESPUÉS de que la ola del árbitro terminó, ANTES de marcar el
    pipeline `completed` (jacobs/executor.py::_correr_pipeline).

    Devuelve (RESULTADO_*, payload):
      - RESULTADO_COMPLETAR: seguir el camino de siempre. `payload` es {}
        salvo que no haya nada que hacer por costo/estructura (igual {}).
      - RESULTADO_DEVUELTO: `payload` trae "plan"/"context"/"run_epoch"/
        "afectados" -- el llamador actualiza su Pipeline en memoria y vuelve
        a correr (recursión corta: como máximo `tope` veces).
      - RESULTADO_TOPE: se alcanzó el tope de devoluciones; `payload["veredicto"]`
        trae la última decisión del árbitro para que el aviso de fin (ya
        existente, jacobs/aviso.py) la muestre junto a las dos versiones que
        quedan en jacobs_events (PIPELINE_DEVUELTO de cada vuelta + este
        evento final)."""
    arbitro = _es_arbitro(pipeline)
    if arbitro is None:
        return RESULTADO_COMPLETAR, {}

    ref = pipeline.context.get(f"step_{arbitro.step_index}_ref")
    if not ref:
        logger.warning(
            "devolucion: pipeline %s sin ref del árbitro (step %d) -- se completa igual",
            pipeline.pipeline_id, arbitro.step_index,
        )
        return RESULTADO_COMPLETAR, {}
    try:
        datos = _load_ref(ref)
    except RefIlegible as exc:
        logger.warning("devolucion: ref del árbitro ilegible (%s) -- se completa igual", exc)
        return RESULTADO_COMPLETAR, {}
    texto = str(datos.get("result") or "")

    veredicto = parsear_veredicto(texto, len(pipeline.plan))
    if veredicto is None:
        # §3.1: fallo cerrado. Sin estructura válida, NO hay devolución, y
        # queda registrado que el árbitro quiso devolver y no se pudo
        # parsear -- nunca se adivina a qué paso se refería.
        await store.event_append(pipeline.pipeline_id, "VEREDICTO_SIN_ESTRUCTURA", {
            "texto": recortar_redactado(texto, _MAX_CHARS_EVENTO),
        }, step_id=arbitro.step_id)
        return RESULTADO_COMPLETAR, {}
    if veredicto.decision != DECISION_DEVOLVER:
        return RESULTADO_COMPLETAR, {}

    tope = await store.get_tope_devoluciones()
    if pipeline.devoluciones >= tope:
        # §3.3: "a la tercera, el pipeline para y avisa, entregando las dos
        # versiones". Las versiones quedan en jacobs_events (cada
        # PIPELINE_DEVUELTO trae los refs que se invalidaron); este evento
        # es el que dice POR QUÉ se paró acá y con qué crítica pendiente.
        await store.event_append(pipeline.pipeline_id, "DEVOLUCION_TOPE_ALCANZADO", {
            "paso": veredicto.paso, "motivo": veredicto.motivo, "cita": veredicto.cita,
            "devoluciones": pipeline.devoluciones, "tope": tope,
        }, step_id=arbitro.step_id)
        return RESULTADO_TOPE, {"veredicto": veredicto}

    # Ronda de arreglo 1 (revisión 2026-09-18): Principio IX -- `aplicar_cupo=
    # False` le pide a continuar_transaccion que salte la condición del cupo
    # PORQUE el pipeline ya está vivo y no está pidiendo un lugar nuevo. Ese
    # supuesto se EXIGE acá, no se asume: `pipeline.status` (el objeto en
    # memoria) nunca se actualiza a `running` en el camino de creación --
    # routes.py arma el Pipeline con el status por defecto (`pending`,
    # models.py) y despacha ESE objeto; _correr_pipeline solo actualiza la
    # FILA, nunca `pipeline.status`. El único lugar que lo actualizaba era
    # continuar.py (vía /continue). Reenviar ese status viejo como
    # `status_leido` hacía que el SELECT...FOR UPDATE de continuar_transaccion
    # comparara 'running' (la fila real) contra 'pending' (el objeto) y
    # abortara SIEMPRE fuera de /continue -- exactamente el final del caso
    # e570ac1c que esta ronda existe para cambiar. Se lee el status VIGENTE
    # con la misma consulta que el resto del ejecutor usa para "¿sigo siendo
    # mi corrida?" (store.pipeline_epoca_y_status) y se EXIGE 'running' antes
    # de tomar el atajo sin cupo: es lo único que hoy distingue "devolver un
    # pipeline vivo" de "revivir uno en un status que no ocupa cupo".
    vigente = await store.pipeline_epoca_y_status(pipeline.pipeline_id)
    if vigente is None or vigente[1] != PipelineStatus.running:
        logger.warning(
            "devolucion: pipeline %s no está running (vigente=%s) -- se completa sin devolver",
            pipeline.pipeline_id, vigente,
        )
        return RESULTADO_COMPLETAR, {}
    epoca_vigente, status_vigente = vigente

    if pipeline.costo_max_aceptado_usd is None:
        # §3.4, fail-closed: sin tope persistido no hay contra qué medir, y
        # nunca se inventa un permiso de gasto nuevo.
        await store.event_append(pipeline.pipeline_id, "DEVOLUCION_SIN_PRESUPUESTO", {
            "paso": veredicto.paso, "motivo": veredicto.motivo,
        }, step_id=arbitro.step_id)
        return RESULTADO_COMPLETAR, {}

    afectados = sorted(pasos_afectados(pipeline.plan, veredicto.paso))
    estimado = await prevuelo(
        pipeline.plan, pipeline.context, pendientes=set(afectados),
        user_id=pipeline.user_id, tenant_id=pipeline.tenant_id,
    )
    if not estimado.ok or estimado.costo_max_usd > pipeline.costo_max_aceptado_usd:
        await store.event_append(pipeline.pipeline_id, "DEVOLUCION_SUPERA_PRESUPUESTO", {
            "paso": veredicto.paso,
            "costo_max_usd": formatear_usd(estimado.costo_max_usd),
            "costo_max_aceptado_usd": formatear_usd(pipeline.costo_max_aceptado_usd),
            "prevuelo_ok": estimado.ok,
        }, step_id=arbitro.step_id)
        return RESULTADO_COMPLETAR, {}

    # A partir de acá, hay devolución de verdad: copia (nunca el plan/
    # contexto en vivo del ejecutor -- si la escritura pierde la carrera,
    # el pipeline sigue con lo que tenía).
    plan = [paso.model_copy(deep=True) for paso in pipeline.plan]
    contexto = dict(pipeline.context)
    for i in afectados:
        contexto.pop(f"step_{i}_ref", None)
        contexto.pop(f"hyde_approved_{plan[i].step_id}", None)
        plan[i].status = StepStatus.pending
        plan[i].error = None
        plan[i].started_at = None
        plan[i].finished_at = None
        plan[i].output_ref = None
    _inyectar_critica(plan[veredicto.paso], veredicto, pipeline.devoluciones + 1)

    evento_payload = {
        "paso": veredicto.paso, "motivo": veredicto.motivo, "cita": veredicto.cita,
        "afectados": afectados, "devolucion_num": pipeline.devoluciones + 1,
        "run_epoch": epoca_vigente + 1,
    }
    nueva = await store.continuar_transaccion(
        pipeline.pipeline_id, epoca_vigente, status_vigente,
        [plan[i] for i in afectados], plan, contexto, min(afectados),
        evento_payload=evento_payload, evento_tipo="PIPELINE_DEVUELTO",
        aplicar_cupo=False, incrementar_devoluciones=True,
    )
    if nueva is None:
        # Otro pedido cambió el pipeline mientras se decidía (perdió la
        # carrera de época/status) -- NO se reintenta a ciegas (mismo
        # criterio que jacobs/continuar.py::continuar()): el que ganó
        # decide qué pasa. El pipeline sigue el camino de completar.
        logger.warning(
            "devolucion: pipeline %s perdió la carrera de época -- se completa sin devolver",
            pipeline.pipeline_id,
        )
        return RESULTADO_COMPLETAR, {}
    return RESULTADO_DEVUELTO, {
        "plan": plan, "context": contexto, "run_epoch": nueva, "afectados": afectados,
    }
