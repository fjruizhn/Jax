"""Contrato de dispatch del catálogo `model` para los caminos de jax (PR-K,
2026-09-14).

QUÉ RESUELVE. El REPL (`jax/muscles/base.py::HttpMuscle`), el planificador de
Ada y el executor de Jacobs, el Motor Registry (`las_manos/motor_registry/
worker.py`) y los caminos Ollama nativos mandaban `"max_tokens": 131072` fijo,
`motor.max_tokens`, o ningún límite. Ahora el nombre y el tope salen de la fila
de `model` del modelo QUE SE DESPACHA, con la MISMA semántica que la Mesa web.

ESPEJO (ronda 2 de PR-K). El bloque entre las marcas "INICIO/FIN DEL BLOQUE
VERBATIM" es copia EXACTA de jax-platform `backend/contrato_dispatch.py`
(la de PR-L, ed538b5, que se mergea antes que PR-K): `ModelDispatchConfigError`,
`_MAX_TOKENS_PARAM_NAMES`, `_MAX_OUTPUT_TOKENS_TOPE_COLUMNA`, `_max_tokens_field`,
`_max_output_tokens_value`, `faltantes_del_contrato`, `errores_del_contrato` y
`TRANSPORTS_CON_CONTRATO_DE_DISPATCH`. Lo compara `scripts/check_mirror_sync.py`
(familia `contrato_dispatch`). Se decide por TRANSPORTE, como allá. Lo de
abajo del bloque es propio de jax: la lectura de la fila y el fragmento del
body por transporte, incluido Ollama (que la Mesa web no limita).

POR QUÉ UN MÓDULO APARTE Y NO facet_resolver. `facet_resolver.py` está espejado
y la copia de jax no selecciona estas columnas (divergencia DECLARADA allá). Y
varios caminos no pasan por `resolve_facet()`: el REPL arma sus músculos al
arrancar y puede despachar un modelo distinto del asignado (modo pesado); el
Motor Registry resuelve por su catálogo. Se lee la fila por
(provider_id, model_id). Una sola fuente dentro de jax: el REPL lo importa como
`jax.core.contrato_dispatch`, Jacobs y LAS MANOS como `contrato_dispatch`
pelado, por el symlink de `las_manos/`.

SIN CACHÉ NI POOL, MEDIDO (las cuatro del rendimiento). Consulta por clave
única (`uk_provider_model`, EXPLAIN en tests/test_contrato_dispatch_db.py).
Latencia de `_leer_contrato` con conexión nueva por llamada, contra
jax_memory_test en hall9000 el 2026-09-14: N=300, p50=0,21 ms, p95=0,29 ms,
máx 0,74 ms. Al lado de una llamada a un LLM de segundos no justifica caché, y
sin caché el UPDATE que sugiere el error vale en el próximo dispatch, sin
invalidación entre procesos.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

try:
    # Jacobs / LAS MANOS (cwd las_manos/, jax.core no es importable ahí).
    from facet_resolver import _db_conn
except ImportError:
    # REPL (PYTHONPATH=. desde la raíz del repo).
    from jax.core.facet_resolver import _db_conn

# ---- INICIO DEL BLOQUE VERBATIM de jax-platform backend/contrato_dispatch.py (PR-L ed538b5; antes e05c5cc) ----
import logging

logger = logging.getLogger(__name__)

# Transportes cuyo dispatch en la Mesa web lee el contrato de la fila de
# `model`. Hoy uno solo: http_openai_compat manda `{max_tokens_param:
# max_output_tokens}` en el body (_call_openai_compat). Los otros NO lo leen:
# http_gemini (_call_gemini) arma generateContent sin límite de salida desde la
# fila; ollama y subprocess no pasan por el catálogo de límites; motor_registry
# no se despacha en la Mesa web. Si un transporte empieza a leer una columna
# de `model` al despachar, entra en este conjunto en el mismo commit.
TRANSPORTS_CON_CONTRATO_DE_DISPATCH = frozenset({"http_openai_compat"})


class ModelDispatchConfigError(RuntimeError):
    """El catálogo (`model`) no declara un dato que el dispatch NECESITA para
    armar el request. FAIL-CLOSED y RUIDOSO: nunca se asume un valor por
    defecto — un default silencioso es exactamente lo que convierte el
    próximo modelo nuevo en un incidente sin síntoma."""


# Nombres válidos del parámetro de límite de salida. Es el mismo conjunto que
# el ENUM de model.max_tokens_param (db/migrations.py) — se replica acá para
# que un valor imposible en la DB (ej. una migración a mano que se saltó el
# ENUM) no termine armando una clave arbitraria en el JSON que va a la API.
_MAX_TOKENS_PARAM_NAMES = ("max_tokens", "max_completion_tokens")

# Tope superior de max_output_tokens: model.max_output_tokens es INT con signo
# (db/migrations.py). Un valor mayor pasaba el validador y reventaba el UPDATE
# con DataError 1264 (un 500) al declararlo desde el admin (PR-L ronda 2). Mismo
# criterio que el ENUM de arriba: el validador conoce el límite de la columna.
_MAX_OUTPUT_TOKENS_TOPE_COLUMNA = 2**31 - 1

# El límite de salida se manda SIEMPRE explícito: sin él, un modelo de
# razonamiento (reasoning_content compitiendo por el mismo budget que content)
# puede agotarlo y cortar la respuesta antes de escribirla — mismo bug ya
# diagnosticado y corregido en motor_registry/worker.py::_call_kimi (017ba2f,
# 2026-08-10). Lo que dejó de ser universal es el VALOR: acá vivía la constante
# 131072 (la misma que jax/muscles/base.py) hasta que gpt-5.6-terra la rechazó
# con HTTP 400 ("max_tokens is too large: 131072. This model supports at most
# 128000 completion tokens"). Ahora sale de model.max_output_tokens, fila por
# fila. Ver _max_output_tokens_value().


def _max_tokens_field(model: str, max_tokens_param: str | None) -> str:
    """Devuelve el NOMBRE del parámetro de límite de salida que exige la API de
    `model`, tal como lo declara el catálogo (`model.max_tokens_param`).

    Por qué es un dato del catálogo y no una constante: 'max_tokens' fue el
    nombre único durante años, pero OpenAI lo rechaza con HTTP 400
    ("Unsupported parameter: 'max_tokens' is not supported with this model.
    Use 'max_completion_tokens' instead") en su generación nueva — el que tumbó
    a thot/gpt-5.6-terra por 3 días (2026-08-24). Cambiar la constante al
    nombre nuevo arregla la instancia y rompe la clase: deepseek-v4-flash
    (jekyll) y glm-5.3 (ada) siguen exigiendo el viejo. Es una propiedad
    estable POR MODELO, del mismo eje que supports_tool_use /
    supports_structured_output / context_window, y vive en la misma fila.

    NULL falla ruidoso a propósito (decisión del dueño, 2026-08-27): si el
    default fuera el parámetro viejo, el próximo modelo nuevo se rompería igual
    que thot pero en silencio y sin nadie mirando. Preferimos que un modelo sin
    valor falle con un mensaje que un operador pueda ejecutar."""
    if max_tokens_param is None:
        # Sin log acá (2026-09-14, PR-J ronda 1): este validador lo usan
        # también los admins, donde NO se aborta ningún dispatch. El ERROR
        # "dispatch abortado" (con este mensaje completo, que trae el UPDATE:
        # el 502 que ve el usuario trunca a 200 chars) lo escribe el camino de
        # dispatch en api/chat.py::_invoke_facet; el admin escribe su WARNING.
        raise ModelDispatchConfigError(
            f"modelo '{model}': la fila de `model` no declara max_tokens_param, "
            f"así que no se sabe si su API exige 'max_tokens' o "
            f"'max_completion_tokens' y NO se asume ninguno. Sembrala: "
            f"UPDATE model SET max_tokens_param='max_tokens' "  # o 'max_completion_tokens'
            f"WHERE model_id='{model}';  -- agregá AND provider_id='<provider>' "
            f"si ese model_id existe para más de un proveedor. Usá "
            f"'max_completion_tokens' para los modelos que rechazan el viejo "
            f"con HTTP 400 (generación nueva de OpenAI), 'max_tokens' para el resto."
        )
    if max_tokens_param not in _MAX_TOKENS_PARAM_NAMES:
        raise ModelDispatchConfigError(
            f"modelo '{model}': max_tokens_param={max_tokens_param!r} no es un "
            f"nombre de parámetro conocido {_MAX_TOKENS_PARAM_NAMES}. Corregí la "
            f"fila de `model` — no se manda una clave arbitraria a la API."
        )
    return max_tokens_param


def _max_output_tokens_value(model: str, max_output_tokens: int | None) -> int:
    """Devuelve el VALOR del límite de tokens de salida que acepta la API de
    `model`, tal como lo declara el catálogo (`model.max_output_tokens`).

    Par de _max_tokens_field(): aquel resuelve CÓMO se llama el parámetro, éste
    QUÉ VALOR admite. Arreglado el nombre (2026-08-27), la misma API contestó
    HTTP 400 por el valor: "max_tokens is too large: 131072. This model supports
    at most 128000 completion tokens, whereas you provided 131072". La constante
    131072 era universal mientras todos los modelos del camino la aceptaran;
    dejó de serlo, y el tope es una propiedad estable POR MODELO.

    NO se deriva de context_window: aquella es la ventana TOTAL (entrada+salida)
    y ésta el tope de completion. gpt-5.6-terra tiene context_window=1050000
    contra un tope de 128000 — verificado, no supuesto. Derivar uno del otro
    sería inventar el dato.

    NULL falla ruidoso a propósito (decisión del dueño, textual: "prefiero que
    un modelo sin valor falle ruidoso a que asuma"). Un default de 131072
    reproduciría este incidente contra el próximo modelo con tope más bajo; uno
    "conservador" truncaría respuestas de modelos de razonamiento en silencio,
    que es justo el bug que el límite explícito existe para prevenir."""
    if max_output_tokens is None:
        # Sin log acá: ver el comentario gemelo en _max_tokens_field.
        raise ModelDispatchConfigError(
            f"modelo '{model}': la fila de `model` no declara max_output_tokens, "
            f"así que no se sabe cuántos tokens de salida acepta su API y NO se "
            f"asume ninguno. Sembrala: "
            f"UPDATE model SET max_output_tokens=<tope de completion> "
            f"WHERE model_id='{model}';  -- agregá AND provider_id='<provider>' "
            f"si ese model_id existe para más de un proveedor. El tope sale de la "
            f"doc del proveedor o del propio HTTP 400 ('This model supports at "
            f"most N completion tokens'); NO es context_window, que es la ventana "
            f"total entrada+salida y suele ser mucho mayor."
        )
    if not isinstance(max_output_tokens, int) or isinstance(max_output_tokens, bool) or max_output_tokens <= 0:
        # Defensa en profundidad contra un valor imposible en la DB (una
        # migración a mano, un 0 heredado de un backfill): un límite <= 0 haría
        # que la API devuelva vacío o un 400, con un modo de falla que se
        # confunde con un error real del proveedor.
        raise ModelDispatchConfigError(
            f"modelo '{model}': max_output_tokens={max_output_tokens!r} no es un "
            f"entero positivo. Corregí la fila de `model` — no se manda un límite "
            f"inválido a la API."
        )
    if max_output_tokens > _MAX_OUTPUT_TOKENS_TOPE_COLUMNA:
        raise ModelDispatchConfigError(
            f"modelo '{model}': max_output_tokens={max_output_tokens!r} no cabe en "
            f"model.max_output_tokens (INT, máximo {_MAX_OUTPUT_TOKENS_TOPE_COLUMNA}). "
            f"Ningún proveedor documenta un tope así: revisá el valor."
        )
    return max_output_tokens


def faltantes_del_contrato(
    transport: str, model: str, max_tokens_param: str | None, max_output_tokens: int | None,
) -> list[tuple[str, ModelDispatchConfigError]]:
    """Corre los MISMOS validadores que el dispatch sobre una fila de `model`
    y devuelve `(columna_faltante, error)` por cada uno que levantaría (vacío =
    el dispatch la aceptaría). La columna sale de QUÉ validador falló, no de
    parsear su mensaje.

    Solo exige el contrato si `transport` lo lee al despachar
    (TRANSPORTS_CON_CONTRATO_DE_DISPATCH): para ollama/subprocess/http_gemini
    esas columnas no significan nada y bloquear sería inventar un requisito.
    Junta los dos errores en vez de cortar en el primero: quien aprueba ve de
    una vez todo lo que falta sembrar."""
    if transport not in TRANSPORTS_CON_CONTRATO_DE_DISPATCH:
        return []
    return errores_del_contrato(model, max_tokens_param, max_output_tokens)


def errores_del_contrato(
    model: str, max_tokens_param, max_output_tokens,
) -> list[tuple[str, ModelDispatchConfigError]]:
    """Los dos validadores del dispatch sobre un par (param, tope), SIN mirar
    el transporte: `(columna, error)` por cada uno que levantaría. Lo usan
    faltantes_del_contrato (el guard, que primero decide si el transporte lo
    lee) y PUT /api/admin/models/{id}/contrato-dispatch (PR-L), que declara
    el contrato de una fila y tiene que aceptar exactamente lo que el
    dispatch acepta."""
    errores = []
    try:
        _max_tokens_field(model, max_tokens_param)
    except ModelDispatchConfigError as e:
        errores.append(("max_tokens_param", e))
    try:
        _max_output_tokens_value(model, max_output_tokens)
    except ModelDispatchConfigError as e:
        errores.append(("max_output_tokens", e))
    return errores


# ---- FIN DEL BLOQUE VERBATIM ----


# Parámetro propio de cada transporte que NO es http_openai_compat, para que
# nadie les mande `max_tokens`/`max_completion_tokens` creyendo que es
# universal. Fuentes: documentación oficial, leída el 2026-09-14 (el
# controller, y la de Ollama /v1 también en esta ronda):
#   - ollama, /api/chat nativo: `options.num_predict`
#     (https://github.com/ollama/ollama/blob/main/docs/api.md). Lo usan
#     OllamaMuscle (REPL jax_local), jacobs/executor.py::_invoke_ollama y
#     jacobs/plan.py::_llm_plan.
#   - ollama, /v1/chat/completions (OpenAI-compatible): `max_tokens`; NO
#     acepta `max_completion_tokens`
#     (https://docs.ollama.com/api/openai-compatibility). Lo usa el Motor
#     Registry con transport='ollama'.
#   - http_gemini: `generationConfig.maxOutputTokens` en generateContent
#     (https://ai.google.dev/api/generate-content). Hoy no se manda límite,
#     igual que la Mesa web (TRANSPORTS_CON_CONTRATO_DE_DISPATCH arriba).
#   - subprocess (hyde, CLI `claude`) / Anthropic Messages API: `max_tokens`,
#     OBLIGATORIO ("The maximum number of tokens to generate before
#     stopping"; cada modelo tiene su máximo)
#     (https://platform.claude.com/docs/en/api/messages). En jax no hay
#     camino HTTP a Anthropic.
PARAMETRO_PROPIO = {
    "http_gemini": "generationConfig.maxOutputTokens",
    "subprocess": "max_tokens (obligatorio, Messages API de Anthropic)",
}

# Endpoints de Ollama con parámetro distinto (ver arriba).
OLLAMA_API_CHAT = "api_chat"
OLLAMA_API_V1 = "v1"

# Constante y no literal en la llamada: el EXPLAIN de
# tests/test_contrato_dispatch_db.py corre ESTA consulta, no una copia.
_SQL_CONTRATO = (
    "SELECT max_tokens_param, max_output_tokens FROM model "
    "WHERE provider_id = %s AND model_id = %s"
)


async def _leer_contrato(provider_id: str, model_id: str) -> tuple | None:
    """(max_tokens_param, max_output_tokens) de la fila, o None si no existe.
    Por la clave única uk_provider_model (provider_id, model_id)."""
    conn = await _db_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute(_SQL_CONTRATO, (provider_id, model_id))
            return await cur.fetchone()
    finally:
        conn.close()


async def limite_de_salida(
    transport: str, provider_id: str, model_id: str, *, ollama_api: str = OLLAMA_API_CHAT,
) -> dict:
    """Fragmento del body con el límite de salida del modelo que se despacha,
    según el TRANSPORTE, leído de SU fila de `model`:

      - http_openai_compat -> {max_tokens_param: max_output_tokens}; exige los
        dos (faltantes_del_contrato, la regla de la Mesa web).
      - ollama + OLLAMA_API_CHAT -> {"options": {"num_predict": max_output_tokens}}
      - ollama + OLLAMA_API_V1   -> {"max_tokens": max_output_tokens}
        (Ollama no usa max_tokens_param: su nombre lo fija el endpoint).

    Lanza ModelDispatchConfigError (el llamador NO despacha) si el transporte
    no tiene límite que armar acá (dice cuál es su parámetro propio), si la
    fila no existe o si falta o es inválido lo que ese transporte exige (con
    el UPDATE). Un error de la DB se propaga: sin contrato leído no hay
    dispatch."""
    if transport not in TRANSPORTS_CON_CONTRATO_DE_DISPATCH and transport != "ollama":
        propio = PARAMETRO_PROPIO.get(transport, "desconocido")
        raise ModelDispatchConfigError(
            f"transporte '{transport}' (modelo '{model_id}'): este camino no arma "
            f"su límite de salida; su parámetro es {propio}, no max_tokens/"
            f"max_completion_tokens."
        )
    if transport == "ollama" and ollama_api not in (OLLAMA_API_CHAT, OLLAMA_API_V1):
        raise ModelDispatchConfigError(f"endpoint de Ollama desconocido: {ollama_api!r}")
    fila = await _leer_contrato(provider_id, model_id)
    if fila is None:
        # Ronda 2 (I1): no se sugiere "agregalo al catálogo" a secas -- una fila
        # nueva bajo un proveedor que no sirve el modelo habilitaría un dispatch
        # cruzado (el modelo de un proveedor a la URL y credencial de otro).
        raise ModelDispatchConfigError(
            f"modelo '{model_id}' no está en el catálogo `model` con "
            f"provider_id='{provider_id}', así que no hay contrato de dispatch que "
            f"leer y NO se asume ninguno. Si el modelo es de OTRO proveedor, el "
            f"error está en el binding o en el modelo pedido: NO lo agregues bajo "
            f"'{provider_id}' (mandaría el modelo a la URL y credencial de un "
            f"proveedor que no lo sirve). Si de verdad es de '{provider_id}', "
            f"sembrá su fila con max_tokens_param y max_output_tokens."
        )
    max_tokens_param, max_output_tokens = fila
    if transport == "ollama":
        tope = _max_output_tokens_value(model_id, max_output_tokens)
        if ollama_api == OLLAMA_API_CHAT:
            return {"options": {"num_predict": tope}}
        return {"max_tokens": tope}
    errores = faltantes_del_contrato(transport, model_id, max_tokens_param, max_output_tokens)
    if errores:
        raise ModelDispatchConfigError(" | ".join(str(e) for _, e in errores))
    return {max_tokens_param: max_output_tokens}
