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

**Ronda 2 (mismo día, corrección de Fernando).** La primera versión de este
módulo reparaba el CONTRATO del modelo al que `ada`/`thot` YA apuntaban
(`glm-5.2`, `gpt-5.5`) sin tocar el nombre -- pasaba el pre-vuelo, pero
mentía sobre qué modelo existe: los dos ya habían sido REEMPLAZADOS en
producción por `glm-5.3` y `gpt-5.6-sol` (textual de Fernando). Revivir un
modelo reemplazado a `available` para que pase el contrato es la misma
familia de defecto que el pedido original vino a cerrar, un nivel más abajo.
Esta versión ESPEJA la faceta completa (proveedor + modelo + contrato)
contra el binding primario VIGENTE de `jax_memory` (producción, **sólo
SELECT** -- ver `_leer_binding_de_produccion`), no un nombre de modelo
hardcodeado: así, cuando producción vuelva a avanzar, la plantilla lo sigue
sola en el próximo clonado, sin que nadie tenga que acordarse de este
archivo. `el_juez`/`jax_local` entran en el mismo espejo (ver
`FACETS_A_ESPEJAR` para el porqué).

Por separado, `facet_binding.model_ref` (la FK real que usa
`jax/core/facet_resolver.py::cargar_registro` y
`test_contrato_dispatch_db.py`) puede desincronizarse de las columnas de
TEXTO `provider_id`/`model_id` de la misma fila -- medido el 2026-09-21:
`el_juez` y `jax_local` decían (texto) `ollama`/`qwen3-coder:30b` pero su
`model_ref` seguía apuntando a la fila DE PRUEBA `sentinel-dbwins-model:99z`
(deprecated). `reparar_model_ref_desincronizado` cierra esa clase de
contaminación para CUALQUIER facet, no sólo las que este módulo espeja.

**Por qué esto y no un UPDATE de una vez.** La plantilla es compartida entre
sesiones concurrentes (ver `base_de_test.py`) y se vuelve a desviar apenas
alguien la toca -- ya pasó DOS veces el mismo día: primero con el binding
real de producción avanzando de `glm-5.2` a `glm-5.3` sin que la copia de
test se enterara, después con esta misma reparación quedándose en el nombre
viejo por no espejar de verdad. Este módulo corre en CADA clonado de sesión
(`base_de_test.py::asegurar_base_de_test`, después de `store.init_tables()`),
así que aunque la plantilla se vuelva a ensuciar -- o producción vuelva a
avanzar --, la base de CADA sesión sale reparada igual. Es idempotente a
propósito: sobre una fila ya sana no toca nada, así que correrlo dos veces
seguidas da el mismo resultado que una.

**Qué SÍ hace con el modelo reemplazado.** Lo vuelve a `deprecated` en la
base de TEST -- y lo DEJA (nunca lo borra): `jax_memory_test.
model_binding_proposal` tiene ~3.900 filas acumuladas que referencian por FK
modelos de prueba viejos (reportado aparte, no resuelto acá), y un `DELETE`
sin auditar esa tabla revienta con `FK RESTRICT` o, peor, la vacía sin que
nadie lo haya decidido.

OJO con lo que `deprecated` afirma acá: para `ada`/`thot`
(`glm-5.2`/`gpt-5.5`), coincide con un hecho que Fernando confirmó a mano
(esos dos modelos fueron reemplazados de verdad). Para el modelo que
`el_juez`/`jax_local` dejan atrás (`qwen3-coder:30b`), `jax_memory` (la
única fuente de verdad de HECHOS de proveedor) sigue diciendo
`status='available'` -- medido 2026-09-21, `SELECT status FROM model WHERE
model_id='qwen3-coder:30b'` en producción da `available`, no `deprecated`.
Este módulo lo demueve IGUAL, pero el significado NO es "el proveedor lo
discontinuó" (eso sería un HECHO inventado, Principio I): es "ningún facet
de `FACETS_A_ESPEJAR` lo usa hoy como binding primario" -- una convención de
gobernanza DE TEST, propia de esta base, no una afirmación sobre el
catálogo real del proveedor.

**Qué NO hace.** No inventa nombres de modelo: todo lo que este módulo
escribe en `model`/`facet_binding` lo COPIA de una fila real de `jax_memory`
leída en el momento (nunca un literal `'glm-5.3'` escrito acá). La única
excepción es el valor sintético de `max_output_tokens` para el modelo
DEMOVIDO cuando hace falta liberar `errores_del_contrato()` en algún test
viejo que lo siga usando -- no aplica hoy (ver `_TOPE_DE_SALIDA_DE_PRUEBA`,
que ronda 2 dejó de usar para `ada`/`thot`: ahora mirroring real trae su
propio `max_output_tokens`).

En memoria de Jairo Urbina.
"""
from __future__ import annotations

#: Los facets que este módulo mantiene ESPEJADOS contra el binding primario
#: vigente de producción -- decisión de alcance 2026-09-21 (Fernando), no
#: algo que se infiera solo de la tabla:
#:   - `ada`/`thot`: el pedido original -- productor + árbitro de un
#:     pipeline real.
#:   - `el_juez`/`jax_local`: MISMA contaminación medida el mismo día
#:     (`facet_binding.model_ref` apuntando a la fila de prueba
#:     `sentinel-dbwins-model:99z` mientras el texto ya decía otra cosa) y
#:     producción YA los movió a `qwen3.6-mesa-131k:latest` -- dejarlos en
#:     `qwen3-coder:30b`/`degraded` sería la MISMA mentira que este módulo
#:     existe para cerrar, un facet más abajo. Verificado 2026-09-21: ningún
#:     test del árbol depende de que la plantilla tenga `qwen3-coder:30b`
#:     para estos dos facets (los que lo nombran son mocks en memoria).
FACETS_A_ESPEJAR = ("ada", "thot", "el_juez", "jax_local")

#: Columnas de `model` que este módulo copia de producción a la base de
#: sesión, en el mismo orden que `_leer_binding_de_produccion`.
_COLUMNAS_MODELO = (
    "is_alias", "context_window", "status", "max_tokens_param",
    "max_output_tokens", "price_input_per_1m_usd", "price_output_per_1m_usd",
)


async def _leer_binding_de_produccion(cur_prod, facet_key: str) -> dict | None:
    """Lee, con un cursor YA ABIERTO contra `jax_memory` -- SÓLO SELECT,
    nunca escribe -- el binding primario vigente de `facet_key` y el
    contrato completo de su modelo. `None` si la faceta no tiene binding
    primario en producción (no debería pasar para `FACETS_A_ESPEJAR`, pero
    no se asume: quien llama decide qué hacer con `None`)."""
    await cur_prod.execute(
        "SELECT fb.provider_id, fb.model_id, m.is_alias, m.context_window, "
        "m.status, m.max_tokens_param, m.max_output_tokens, "
        "m.price_input_per_1m_usd, m.price_output_per_1m_usd "
        "FROM facet_binding fb JOIN model m ON m.id = fb.model_ref "
        "WHERE fb.facet_key=%s AND fb.role='primary'",
        (facet_key,),
    )
    fila = await cur_prod.fetchone()
    if not fila:
        return None
    provider_id, model_id, *resto = fila
    return {"provider_id": provider_id, "model_id": model_id,
            **dict(zip(_COLUMNAS_MODELO, resto))}


async def _upsert_modelo_espejo(cur, vigente: dict) -> int:
    """Crea o actualiza, en la base de SESIÓN (`cur`), la fila de `model`
    que `vigente` describe -- copiada de producción, nunca inventada.
    Devuelve su `id`. `source='manual'`: esta fila no salió de una llamada
    real al proveedor ni de models.dev, salió de espejar `jax_memory` a
    mano en este módulo -- 'provider_api' (lo que dice la fila real en
    producción) mentiría sobre CÓMO se pobló ESTA copia."""
    await cur.execute(
        "SELECT id FROM model WHERE provider_id=%s AND model_id=%s",
        (vigente["provider_id"], vigente["model_id"]),
    )
    fila = await cur.fetchone()
    if fila is None:
        await cur.execute(
            "INSERT INTO model (provider_id, model_id, is_alias, context_window, "
            "status, max_tokens_param, max_output_tokens, price_input_per_1m_usd, "
            "price_output_per_1m_usd, source, source_checked_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'manual', NOW())",
            (vigente["provider_id"], vigente["model_id"], vigente["is_alias"],
             vigente["context_window"], vigente["status"], vigente["max_tokens_param"],
             vigente["max_output_tokens"], vigente["price_input_per_1m_usd"],
             vigente["price_output_per_1m_usd"]),
        )
        await cur.execute(
            "SELECT id FROM model WHERE provider_id=%s AND model_id=%s",
            (vigente["provider_id"], vigente["model_id"]),
        )
        return (await cur.fetchone())[0]

    (model_id_de_la_fila,) = fila
    await cur.execute(
        "UPDATE model SET is_alias=%s, context_window=%s, status=%s, "
        "max_tokens_param=%s, max_output_tokens=%s, price_input_per_1m_usd=%s, "
        "price_output_per_1m_usd=%s WHERE id=%s",
        (vigente["is_alias"], vigente["context_window"], vigente["status"],
         vigente["max_tokens_param"], vigente["max_output_tokens"],
         vigente["price_input_per_1m_usd"], vigente["price_output_per_1m_usd"],
         model_id_de_la_fila),
    )
    return model_id_de_la_fila


async def _binding_actual(cur, facet_key: str) -> tuple[str, str, int | None] | None:
    """(provider_id, model_id, model_ref) del binding primario de
    `facet_key` en la base de SESIÓN, ANTES de espejar -- para saber a qué
    modelo demover a `deprecated` si el espejo lo reemplaza."""
    await cur.execute(
        "SELECT provider_id, model_id, model_ref FROM facet_binding "
        "WHERE facet_key=%s AND role='primary'",
        (facet_key,),
    )
    return await cur.fetchone()


async def espejar_binding_vigente(cur, cur_prod, facet_key: str) -> dict:
    """Deja el binding primario de `facet_key`, en la base de SESIÓN
    (`cur`), igual al vigente de producción (`cur_prod`, sólo SELECT): crea
    o repara la fila de `model` correspondiente, repunta
    `facet_binding.provider_id/model_id/model_ref` a ella, y -- si el
    modelo anterior era otro -- lo demueve a `status='deprecated'` (nunca lo
    borra: puede haber `model_binding_proposal` referenciándolo por FK).

    Devuelve un resumen para loguear o para un test. Es un no-op real
    (ninguna fila tocada) si la base de sesión ya coincide con producción."""
    vigente = await _leer_binding_de_produccion(cur_prod, facet_key)
    if vigente is None:
        return {"facet_key": facet_key, "espejado": False,
                "motivo": "sin binding primario en producción"}

    anterior = await _binding_actual(cur, facet_key)
    ya_vigente = (
        anterior is not None
        and anterior[0] == vigente["provider_id"]
        and anterior[1] == vigente["model_id"]
    )

    model_ref = await _upsert_modelo_espejo(cur, vigente)

    if not ya_vigente:
        await cur.execute(
            "UPDATE facet_binding SET provider_id=%s, model_id=%s, model_ref=%s "
            "WHERE facet_key=%s AND role='primary'",
            (vigente["provider_id"], vigente["model_id"], model_ref, facet_key),
        )
        if anterior is not None and (anterior[0], anterior[1]) != (vigente["provider_id"], vigente["model_id"]):
            await cur.execute(
                "UPDATE model SET status='deprecated' WHERE provider_id=%s AND model_id=%s",
                (anterior[0], anterior[1]),
            )
    elif anterior is not None and anterior[2] != model_ref:
        # Texto ya vigente, pero model_ref (la FK real) seguía apuntando a
        # otra fila -- la MISMA desincronización que
        # reparar_model_ref_desincronizado() cierra en general; se corrige
        # acá también para no depender del orden de llamada.
        await cur.execute(
            "UPDATE facet_binding SET model_ref=%s WHERE facet_key=%s AND role='primary'",
            (model_ref, facet_key),
        )

    return {
        "facet_key": facet_key, "espejado": not ya_vigente,
        "modelo": f"{vigente['provider_id']}/{vigente['model_id']}",
        "anterior": None if anterior is None else f"{anterior[0]}/{anterior[1]}",
    }


async def reparar_model_ref_desincronizado(cur) -> int:
    """Repara, con el cursor YA ABIERTO `cur`, cualquier `facet_binding` cuyo
    `model_ref` (la FK real que usa el código, `fk_facet_binding_model_ref`)
    quedó apuntando a una fila de `model` DISTINTA de la que sus propias
    columnas de texto (`provider_id`, `model_id`) describen -- el síntoma
    medido en `el_juez`/`jax_local` el 2026-09-21 (columnas de texto
    corregidas a mano en algún momento, `model_ref` nunca las siguió).

    Genérico a propósito (sin lista de facets hardcodeada): la invariante
    "`model_ref` apunta a la fila que las columnas de texto describen" vale
    para CUALQUIER fila de `facet_binding`, no sólo para los facets de
    `FACETS_A_ESPEJAR`. Devuelve cuántas filas reparó."""
    await cur.execute(
        "UPDATE facet_binding fb "
        "JOIN model m ON m.provider_id = fb.provider_id AND m.model_id = fb.model_id "
        "SET fb.model_ref = m.id "
        "WHERE fb.model_ref IS NULL OR fb.model_ref <> m.id"
    )
    return cur.rowcount


async def reparar_gobernanza_de_test(cur, cur_prod) -> dict:
    """Corre las reparaciones de este módulo, en orden: primero espeja
    `FACETS_A_ESPEJAR` contra el binding vigente de producción (`cur_prod`,
    SÓLO SELECT), después sincroniza `model_ref` en general. Devuelve un
    resumen para loguear o para un test."""
    espejados = [await espejar_binding_vigente(cur, cur_prod, facet_key)
                 for facet_key in FACETS_A_ESPEJAR]
    filas_de_model_ref = await reparar_model_ref_desincronizado(cur)
    return {
        "espejados": espejados,
        "model_ref_desincronizados_reparados": filas_de_model_ref,
    }
