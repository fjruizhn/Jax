"""Reparación idempotente de la gobernanza de modelos en la base de TEST.

**El problema que esto cierra** (pedido 2026-09-21). `jax_memory_test` -- la
PLANTILLA que `base_de_test.py::_clonar_esquema` copia a cada base de sesión
-- tenía a `ada` apuntando a un modelo `deprecated` sin `max_tokens_param` ni
`max_output_tokens`, y a `thot` sin los dos tampoco.
`jacobs/prevuelo_reglas.py::errores_de_contrato` trata eso como
`sin_contrato_de_salida` y bloquea CUALQUIER pre-vuelo que use esas
facetas, aunque el resto del pipeline esté sano. Medido a mano ejercitando
el árbitro el 2026-09-21 (`jacobs/_ejercicio_manual_arbitro_2026_09_21.py`,
worktree `jax-ejercitar-arbitro`): hubo que parchear 3 columnas de `model` a
mano para que el ejercicio arrancara.

Por separado, `facet_binding.model_ref` (la FK real que usa
`jax/core/facet_resolver.py::cargar_registro` y
`test_contrato_dispatch_db.py`) puede desincronizarse de las columnas de
TEXTO `provider_id`/`model_id` de la misma fila -- medido el 2026-09-21:
`el_juez` y `jax_local` decían (texto) `ollama`/`qwen3-coder:30b` pero su
`model_ref` seguía apuntando a la fila DE PRUEBA `sentinel-dbwins-model:99z`
(deprecated). Eso es justo la contaminación que este módulo también repara.

**Por qué esto y no un UPDATE de una vez.** La plantilla es compartida entre
sesiones concurrentes (ver `base_de_test.py`) y se vuelve a desviar apenas
alguien la toca -- ya pasó: el binding REAL de producción avanzó de
`glm-5.2` a `glm-5.3` sin que nadie actualizara la copia de test. Este
módulo corre en CADA clonado de sesión
(`base_de_test.py::asegurar_base_de_test`, después de `store.init_tables()`),
así que aunque la plantilla se vuelva a ensuciar, la base de CADA sesión
sale reparada igual -- nadie tiene que acordarse de arreglarla a mano otra
vez. Es idempotente a propósito: sobre una fila ya sana no toca nada (el
`WHERE` filtra lo que ya está bien, y `COALESCE` nunca pisa un valor real
por uno de prueba), así que correrlo dos veces da el mismo resultado que
una.

**Qué NO hace.** No inventa nombres de modelo nuevos ni reasigna
`facet_binding.model_id`/`provider_id`: repara el CONTRATO (`status`,
`max_tokens_param`, `max_output_tokens`) de la fila de `model` a la que la
faceta YA apunta, y sincroniza `model_ref` con lo que esa misma fila de
`facet_binding` YA declara en sus columnas de texto -- nunca al revés. El
valor de `max_output_tokens` que siembra (`_TOPE_DE_SALIDA_DE_PRUEBA`) es un
entero de PRUEBA para que `errores_del_contrato()` no lo rechace -- no
pretende ser el tope real de la API del proveedor (eso lo declara el
catálogo real y se sincroniza por otro camino: `resolve_facet`/la sonda de
`jacobs/sonda.py`).

**Qué NO repara, a propósito, y por qué.** `jax_memory_test.model_binding_proposal`
tiene ~3.900 filas acumuladas (medido 2026-09-21) de corridas anteriores a la
sesión-por-defecto (decisión de Fernando, 2026-09-20) -- entre ellas, cientos
de filas que referencian por FK las filas DE PRUEBA `glm-5.2-preview` (id 61)
y `sentinel-dbwins-model:99z` (id 2059), lo que impide BORRARLAS sin tocar esa
tabla también. Purgar `model_binding_proposal` es una decisión de alcance
mayor (¿truncar la plantilla? ¿un TTL?) que este módulo no toma por su
cuenta -- reportado aparte, no resuelto acá.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

#: Los facets HTTP que un pipeline real ejercita hoy con productor + árbitro
#: (`jacobs/_ejercicio_manual_arbitro_2026_09_21.py`: `ada` produce, `thot`
#: arbitra). Es una decisión de ALCANCE del pedido 2026-09-21, no config de
#: la app -- si mañana un tercer facet HTTP entra al camino de un pipeline
#: real, se agrega acá a propósito, nunca se infiere solo de la tabla.
FACETS_CON_CONTRATO_DE_SALIDA = ("ada", "thot")

#: Nombre del parámetro de tope de salida que cada proveedor exige en su API
#: HTTP compatible con OpenAI -- HECHO sobre proveedores reales, documentado
#: en `jax/core/contrato_dispatch.py::_max_tokens_field` (la generación
#: nueva de OpenAI rechaza 'max_tokens' con HTTP 400 y exige
#: 'max_completion_tokens'; el resto de los proveedores acepta 'max_tokens').
#: Vive acá como dato -- igual que la tabla gemela de contrato_dispatch.py --
#: no como config: no es algo que cambie por entorno.
_MAX_TOKENS_PARAM_POR_PROVEEDOR = {"openai": "max_completion_tokens"}
_MAX_TOKENS_PARAM_DEFAULT = "max_tokens"

#: Entero de PRUEBA (ver docstring del módulo): no es un hecho sobre la API
#: real de ningún proveedor. 16384, no un número menor: con 8192 el árbitro
#: (capability 'critique', `capability.min_output_tokens`=13312 medido
#: 2026-09-21) seguía cayendo en `tope_insuficiente` -- una regla DISTINTA de
#: `sin_contrato_de_salida` (la que este módulo existe para cerrar), pero que
#: igual bloqueaba "un pipeline real con ada y thot pasa el pre-vuelo" si no
#: se le daba margen. Mismo valor que ya usaba
#: `jacobs/_ejercicio_manual_arbitro_2026_09_21.py` al parchear esto a mano.
_TOPE_DE_SALIDA_DE_PRUEBA = 16384


async def reparar_contrato_de_salida(cur) -> list[str]:
    """Repara, con el cursor YA ABIERTO `cur`, el contrato de salida
    (`model.status`/`max_tokens_param`/`max_output_tokens`) de los modelos a
    los que apuntan `FACETS_CON_CONTRATO_DE_SALIDA` en la base a la que ese
    cursor está conectado.

    Devuelve la lista de `facet_key` que tuvo que tocar -- vacía si ya
    estaba todo sano (la llamada es entonces un no-op real, no sólo
    semántico: ni una fila se actualiza)."""
    claves = tuple(FACETS_CON_CONTRATO_DE_SALIDA)
    placeholders = ", ".join(["%s"] * len(claves))
    await cur.execute(
        f"SELECT facet_key, provider_id, model_id FROM facet_binding "
        f"WHERE facet_key IN ({placeholders}) AND role='primary'",
        claves,
    )
    filas = await cur.fetchall()
    reparadas: list[str] = []
    for facet_key, provider_id, model_id in filas:
        param = _MAX_TOKENS_PARAM_POR_PROVEEDOR.get(provider_id, _MAX_TOKENS_PARAM_DEFAULT)
        await cur.execute(
            "UPDATE model SET "
            "status = 'available', "
            "max_tokens_param = COALESCE(max_tokens_param, %s), "
            "max_output_tokens = COALESCE(max_output_tokens, %s) "
            "WHERE provider_id = %s AND model_id = %s "
            "AND (status = 'deprecated' "
            "     OR max_tokens_param IS NULL "
            "     OR max_output_tokens IS NULL)",
            (param, _TOPE_DE_SALIDA_DE_PRUEBA, provider_id, model_id),
        )
        if cur.rowcount:
            reparadas.append(facet_key)
    return reparadas


async def reparar_model_ref_desincronizado(cur) -> int:
    """Repara, con el cursor YA ABIERTO `cur`, cualquier `facet_binding` cuyo
    `model_ref` (la FK real que usa el código, `fk_facet_binding_model_ref`)
    quedó apuntando a una fila de `model` DISTINTA de la que sus propias
    columnas de texto (`provider_id`, `model_id`) describen -- el síntoma
    medido en `el_juez`/`jax_local` el 2026-09-21 (columnas de texto
    corregidas a mano en algún momento, `model_ref` nunca las siguió).

    Genérico a propósito (sin lista de facets hardcodeada): la invariante
    "`model_ref` apunta a la fila que las columnas de texto describen" vale
    para CUALQUIER fila de `facet_binding`, no sólo para las dos donde se
    detectó la desincronización. Devuelve cuántas filas reparó."""
    await cur.execute(
        "UPDATE facet_binding fb "
        "JOIN model m ON m.provider_id = fb.provider_id AND m.model_id = fb.model_id "
        "SET fb.model_ref = m.id "
        "WHERE fb.model_ref IS NULL OR fb.model_ref <> m.id"
    )
    return cur.rowcount


async def reparar_gobernanza_de_test(cur) -> dict:
    """Corre las dos reparaciones de este módulo, en orden: primero el
    contrato de salida de `ada`/`thot` (necesita que `model_ref` no importe,
    usa `provider_id`/`model_id` de texto), después la sincronización general
    de `model_ref`. Devuelve un resumen para loguear o para un test."""
    facets_reparados = await reparar_contrato_de_salida(cur)
    filas_de_model_ref = await reparar_model_ref_desincronizado(cur)
    return {
        "contrato_de_salida_reparado": facets_reparados,
        "model_ref_desincronizados_reparados": filas_de_model_ref,
    }
