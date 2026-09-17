"""Jacobs — pre-vuelo antes de gastar (spec 2026-09-17 §4).

Corre obligatoriamente al crear y al continuar (dentro del candado), y por su
propio endpoint (sin escribir). Para cada paso a evaluar:

1. arma el Despacho real: faceta HTTP con su binding, o el motor que
   resolvería MotorPolicy (desvío 1 del plan);
2. mide el prompt con las MISMAS funciones del ejecutor, rellenando cada
   dependencia que todavía no tiene ref con el peor caso (desvío 3);
3. aplica las reglas puras (prevuelo_reglas);
4. sondea en paralelo las claves sin un `ok` fresco, una vez por clave y solo
   si sus pasos no tienen otra violación (desvío 16).

Un error de la base se propaga: sin pre-vuelo no se crea (quien llama responde
503 prevuelo_no_disponible). Nada de esto es fail-open.

`sonda.sondear()` distingue (Task 7, Rulings R13/R16): una caída real de la
base al resolver faceta o credencial (FacetUnavailableError o
CredentialUnavailableError cuya causa no es "fila ausente") se PROPAGA, no se
convierte en `faceta_caida`. El `asyncio.gather` de abajo respeta eso: si
alguna sonda revienta, se cancelan y se esperan las demás ANTES de relanzar,
para no dejar tareas huérfanas (warning "Task exception was never retrieved")
ni perder la excepción original.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

from motor_registry.output_validator import puede_pedir_reintento

from jacobs import prevuelo_catalogo, prevuelo_config, sonda, store
from jacobs.executor import MAX_DEP_CONTEXT_CHARS, _build_context_input, _enrich_prompt
from jacobs.facet_health import salud_de_proveedor
from jacobs.models import MOTOR_FACETS, Pipeline, Step
from jacobs.prevuelo_catalogo import Catalogo, FilaModelo, MotorResuelto
from jacobs.prevuelo_reglas import (
    CostoPaso,
    Despacho,
    Veredicto,
    Violacion,
    armar_veredicto,
    costo_sin_evaluar,
    evaluar_paso,
)

# Largo de un uuid real: el contexto de identidad del Motor Registry lo incluye.
_TASK_ID_DE_MEDIDA = "00000000-0000-0000-0000-000000000000"
# Una dependencia que todavía no corrió: el armado la recorta a
# MAX_DEP_CONTEXT_CHARS (o a 500 si el paso no declara depends_on).
# +1 (fix round 1, revisión de Task 8, 2026-09-17): executor.py marca
# `truncated` con `len(text) > MAX_DEP_CONTEXT_CHARS`, ESTRICTO. Un relleno
# de exactamente el tope no se marca truncado y no lleva la nota
# " [TRUNCADO -- dependencia excede el tope]" que `_enrich_prompt` le agrega
# a una dependencia real que sí lo excede -- el peor caso contaba 40 chars
# menos que un caso real (visto en rojo: 30305 < 30325 contra una
# dependencia real de 2x el tope). "Sobreestima, nunca subestima" (spec
# §4.6) exige cruzar el borde, no tocarlo.
_RELLENO_DE_DEPENDENCIA = "inline:" + json.dumps({"result": "x" * (MAX_DEP_CONTEXT_CHARS + 1)})
_SIN_FILA = FilaModelo(None, None, None, None)
_SEPARADOR_DE_IDENTIDAD = "\n---\n"  # worker.run: identity + "\n---\n" + prompt


def _ahora() -> float:
    return time.time()


def _max_iteraciones() -> int:
    from motor_registry.worker import MAX_TOOL_LOOP_ITERATIONS
    return MAX_TOOL_LOOP_ITERATIONS


def _despacho(step: Step, catalogo: Catalogo, motor: MotorResuelto | None) -> tuple[Despacho | None, str]:
    if step.facet in MOTOR_FACETS:
        if motor is None:
            return None, (
                f"ningún motor habilitado despacha la capability '{step.capability}' "
                f"para la faceta '{step.facet}' (capability_motor / motor.status)"
            )
        fila = catalogo.modelos.get((motor.provider_id, motor.modelo), _SIN_FILA)
        return Despacho(
            clave_salud=motor.clave, via_motor=True, transporte=motor.transporte,
            provider_id=motor.provider_id, base_url=motor.base_url, modelo=motor.modelo,
            max_tokens_param=fila.max_tokens_param, max_output_tokens=fila.max_output_tokens,
            motor_max_tokens=motor.max_tokens, precio_in=fila.precio_in, precio_out=fila.precio_out,
            tiene_herramientas=motor.tiene_herramientas,
            schema_con_reintento=puede_pedir_reintento(motor.output_schema),
            persona=None,
        ), ""
    faceta = catalogo.facetas.get(step.facet)
    if faceta is None:
        return None, f"la faceta '{step.facet}' no tiene fila activa en `facet` con binding primario"
    fila = catalogo.modelos.get((faceta.provider_id, faceta.model_id), _SIN_FILA)
    return Despacho(
        clave_salud=faceta.key, via_motor=False, transporte=faceta.transport,
        provider_id=faceta.provider_id, base_url=faceta.base_url, modelo=faceta.model_id,
        max_tokens_param=fila.max_tokens_param, max_output_tokens=fila.max_output_tokens,
        motor_max_tokens=0, precio_in=fila.precio_in, precio_out=fila.precio_out,
        tiene_herramientas=False, schema_con_reintento=False, persona=faceta.persona,
    ), ""


def _chars_de_entrada(step: Step, pasos: list[Step], contexto: dict, d: Despacho) -> int:
    """SÍNCRONA (lee artifacts de disco): se llama por asyncio.to_thread."""
    peor = dict(contexto)
    for j in range(step.step_index):
        if not peor.get(f"step_{j}_ref"):
            peor[f"step_{j}_ref"] = _RELLENO_DE_DEPENDENCIA
    medida = Pipeline(name="prevuelo", invoked_by="plataforma", mode="autonomous",
                      plan=pasos, context=peor)
    chars = len(_enrich_prompt(_build_context_input(step, medida))) + len(d.persona or "")
    if d.via_motor:
        from motor_registry.identity_context import build_identity_context
        from motor_registry.worker import _REFORMAS_V3_PREDICATES
        chars += len(build_identity_context(
            motor_name=d.clave_salud, capabilities=[step.capability], catalog={},
            predicates=_REFORMAS_V3_PREDICATES, task_id=_TASK_ID_DE_MEDIDA,
        )) + len(_SEPARADOR_DE_IDENTIDAD)
    return chars


@dataclass
class _Vuelo:
    loop: asyncio.AbstractEventLoop
    tarea: asyncio.Future
    esperando: int = 0


# F5 (ola final, 2026-09-17): UN solo vuelo de sonda por clave en el proceso.
# Antes, N pre-vuelos concurrentes con la misma clave sin `ok` fresco lanzaban
# N sondas pagas. No es una caché de resultados (LAS CUATRO #2 no aplica): la
# entrada vive mientras la sonda está EN VUELO y se borra al terminar -- con
# resultado, excepción o cancelación (done callback). El dato fresco para
# después lo guarda facet_health_event. El uso de la sonda compartida queda
# atribuido a la identidad del primer pre-vuelo que la lanzó.
_sondas_en_vuelo: dict[str, _Vuelo] = {}


def _olvidar_vuelo(clave: str, vuelo: _Vuelo) -> None:
    if _sondas_en_vuelo.get(clave) is vuelo:
        del _sondas_en_vuelo[clave]


async def _sondear_una_vez(clave: str, d: Despacho, user_id: str | None,
                           tenant_id: str | None):
    """Se suma al vuelo en curso de `clave` o lanza uno. La sonda corre en su
    propia tarea (shield): cancelar a UNO de los que esperan no la corta para
    los demás; si se cancelan TODOS, se cancela y se espera (no queda tarea
    huérfana). Una excepción de la sonda llega a todos."""
    loop = asyncio.get_running_loop()
    vuelo = _sondas_en_vuelo.get(clave)
    if vuelo is None or vuelo.loop is not loop or vuelo.tarea.done():
        tarea = asyncio.ensure_future(sonda.sondear(clave, d, user_id=user_id, tenant_id=tenant_id))
        vuelo = _Vuelo(loop, tarea)
        _sondas_en_vuelo[clave] = vuelo
        tarea.add_done_callback(lambda _t, c=clave, v=vuelo: _olvidar_vuelo(c, v))
    vuelo.esperando += 1
    try:
        return await asyncio.shield(vuelo.tarea)
    finally:
        vuelo.esperando -= 1
        if vuelo.esperando == 0 and not vuelo.tarea.done():
            # Fuera del registro ANTES de cancelar: un pre-vuelo que llega
            # mientras esta cancelación termina lanza su propia sonda en vez
            # de sumarse a una que ya no va a responder.
            _olvidar_vuelo(clave, vuelo)
            vuelo.tarea.cancel()
            await asyncio.wait({vuelo.tarea})


async def _sondear_todas(a_sondear: dict[str, Despacho], claves: list[str],
                         user_id: str | None, tenant_id: str | None) -> list:
    """`asyncio.gather` sin `return_exceptions`: la primera sonda que revienta
    (Ruling R13/R16 de Task 7 -- DB caída al resolver, nunca `faceta_caida`)
    tiene que salir de acá tal cual, para que el llamador responda 503
    (§8). Pero gather() por sí solo, al fallar, NO cancela ni espera las
    tareas que dejó atrás: quedarían corriendo sueltas y, si también
    revientan, el logger de asyncio se queja de una excepción que nadie
    retiró. Por eso se envuelven en Task explícitas: si el gather corta por
    una excepción, se cancela lo que sigue vivo y se lo espera (con
    `return_exceptions=True`, así ESA espera no puede volver a reventar)
    antes de relanzar la excepción original."""
    tareas = [
        asyncio.ensure_future(_sondear_una_vez(c, a_sondear[c], user_id, tenant_id))
        for c in claves
    ]
    try:
        return await asyncio.gather(*tareas)
    except BaseException:
        for t in tareas:
            t.cancel()
        await asyncio.gather(*tareas, return_exceptions=True)
        raise


async def prevuelo(
    pasos: list[Step],
    contexto: dict,
    *,
    pendientes: set[int] | None = None,
    user_id: str | None = None,
    tenant_id: str | None = None,
) -> Veredicto:
    por_indice = {s.step_index: s for s in pasos}
    indices = sorted(por_indice) if pendientes is None else sorted(pendientes)

    costos: list[CostoPaso] = []
    evaluables: list[Step] = []
    for i in indices:
        corto = costo_sin_evaluar(por_indice[i])
        if corto is None:
            evaluables.append(por_indice[i])
        else:
            costos.append(corto)
    if not evaluables:
        return armar_veredicto([], costos, [])

    ahora = _ahora()
    # UNA conexión del pool del store para todo el catálogo (Task 15b): se
    # devuelve antes de armar prompts y sondear, que no tocan la base.
    async with store.conexion_del_pool() as conexion:
        motores = await prevuelo_catalogo.resolver_motores(
            [s for s in evaluables if s.facet in MOTOR_FACETS], conexion=conexion)
        catalogo = await prevuelo_catalogo.leer_catalogo(
            conexion=conexion,
            facetas={s.facet for s in evaluables if s.facet not in MOTOR_FACETS},
            motores=[m for m in motores.values() if m is not None],
            capabilities={s.capability for s in evaluables},
            ahora=ahora,
        )
    chars_por_token = prevuelo_config.chars_por_token()
    max_iteraciones = _max_iteraciones()

    violaciones: list[Violacion] = []
    despachables: dict[int, Despacho] = {}
    for step in evaluables:
        d, detalle = _despacho(step, catalogo, motores.get(step.step_index))
        chars = 0 if d is None else await asyncio.to_thread(_chars_de_entrada, step, pasos, contexto, d)
        propias, costo = evaluar_paso(
            step.step_index, step.facet, d,
            min_output_tokens=catalogo.min_output_tokens.get(step.capability, 0),
            credencial_activa=d is not None and d.provider_id in catalogo.proveedores_con_credencial,
            chars_entrada=chars, chars_por_token=chars_por_token,
            max_iteraciones=max_iteraciones, detalle_inexistente=detalle,
        )
        violaciones += propias
        costos.append(costo)
        if d is not None and not propias:
            despachables[step.step_index] = d

    a_sondear: dict[str, Despacho] = {}
    for d in despachables.values():
        if salud_de_proveedor(catalogo.salud.get(d.clave_salud), ahora) == "sondear":
            a_sondear.setdefault(d.clave_salud, d)
    claves = sorted(a_sondear)
    resultados = await _sondear_todas(a_sondear, claves, user_id, tenant_id)
    caidas = {c: r for c, r in zip(claves, resultados) if not r.ok}
    for i, d in sorted(despachables.items()):
        r = caidas.get(d.clave_salud)
        if r is not None:
            violaciones.append(Violacion(i, por_indice[i].facet, "faceta_caida",
                                         r.detalle or "la sonda no obtuvo respuesta"))
    return armar_veredicto(violaciones, costos, claves)
