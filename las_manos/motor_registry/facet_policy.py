"""LAS MANOS — Motor Registry: gobernanza de nivel FACET.

check_capability_admission()/check() (policy.py) gobiernan dispatch por
CAPABILITY -- tiene sentido para un step de pipeline con un objetivo
concreto. Mesa web no tiene eso: un turno de chat es texto libre enrutado
a un facet por keyword-matching, sin capability asociada (verificado
leyendo jax-platform/backend/api/chat.py completo -- ver
docs/superpowers/specs/2026-08-27-http-facets-motor-policy-governance-
design.md, seccion 3). check_facet_admission() responde la pregunta que
SI tiene sentido para ese camino: "¿puede este caller hablar con este
facet?" -- nada mas. No toca la tabla `capability`.

Modulo con I/O real (a diferencia de policy.py, "puro, sin I/O") --
consulta facet.allowed_callers directo.

CONEXION (2026-09-17): del pool compartido de Jacobs (`jacobs.store.conexion`),
no una conexion suelta. Este camino corre en CADA `POST /authorize-facet` (uno
por turno de la Mesa web a un facet HTTP): abrir y cerrar un socket por pedido
es exactamente lo que agoto los puertos del host en la carga del frente F. El
guard fail-closed de JAX_DB_HOST/JAX_DB_PORT y el connect_timeout viven en el
store.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import json

from jacobs import store as jacobs_store


async def check_facet_admission(caller: str, facet: str) -> tuple[bool, str]:
    """Fail-closed: facet inexistente, allowed_callers NULL, allowed_callers
    con un JSON que no es lista, o caller ausente de la lista -> (False,
    razon). Nunca deja pasar por duda.

    ALCANCE de `facet.allowed_callers` -- leer antes de editar esa columna:
    gobierna a los callers que NO tienen concepto de capability. Hoy eso es
    exactamente uno: `jax_platform_chat` (Mesa web, jax-platform/backend/
    api/chat.py::_invoke_facet). Jacobs NO pasa por acá: se gobierna por
    `capability.allowed_callers` vía MotorPolicy.check_capability_admission()
    (policy.py), invocado desde jacobs/executor.py::validate_capability().
    Consecuencia práctica: sacar "jacobs" de `facet.allowed_callers` NO
    restringe a Jacobs -- Jacobs seguirá despachando igual. Para cortarle
    el acceso a Jacobs hay que editar `capability.allowed_callers` de las
    capabilities involucradas. El valor "jacobs" está sembrado en la
    columna por simetría descriptiva (refleja el acceso que ya existía de
    hecho), no porque algo lo consulte. Ver DEUDA.md, entrada de
    gobernanza de `_HTTP_FACETS`, para el follow-up candidato (hacer que
    Jacobs también consulte esta función, y que la columna pase a ser el
    gate real de nivel facet para ambos caminos).
    """
    async with jacobs_store.conexion() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT allowed_callers FROM facet WHERE `key`=%s",
                (facet,),
            )
            row = await cur.fetchone()

    if row is None:
        return False, f"facet desconocido: '{facet}'"

    (allowed_callers_raw,) = row
    if allowed_callers_raw is None:
        return False, f"facet '{facet}' no configurado (allowed_callers) -- fail-closed"

    # El CHECK de la DB garantiza json_valid(), NO que el valor sea una
    # lista. Sin este guard, un JSON escalar (p.ej. la cadena
    # "jax_platform_chatX") convertiría el `not in` de abajo en un test de
    # SUBSTRING: caller="jax_platform_chat" pasaría -- un fail-open dentro
    # de la única función cuyo contrato entero es fail-closed. Un dict daría
    # membership contra sus CLAVES, igual de silencioso. Tipo inesperado =
    # rechazo explícito, nunca interpretación creativa.
    allowed_callers = json.loads(allowed_callers_raw)
    if not isinstance(allowed_callers, list):
        return False, (
            f"facet '{facet}': allowed_callers no es una lista JSON "
            f"(tipo {type(allowed_callers).__name__}) -- fail-closed"
        )

    if caller not in allowed_callers:
        return False, f"caller '{caller}' no autorizado para facet '{facet}'. Autorizados: {allowed_callers}"

    return True, f"OK: '{caller}' → facet '{facet}'"
