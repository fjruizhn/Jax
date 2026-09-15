"""Contrato de dispatch del catálogo `model` para los caminos de jax (PR-K,
2026-09-14).

QUÉ RESUELVE. El REPL (`jax/muscles/base.py::HttpMuscle`, caminos DeepSeek y
OpenAI-compatible) y el planificador de Ada (`jacobs/plan.py::_ada_plan`)
mandaban `"max_tokens": 131072` FIJO. Es el mismo literal que tumbó a thot
(gpt-5.6-terra) en la Mesa web el 2026-08-24: primero por el NOMBRE (exige
`max_completion_tokens`) y después por el VALOR (acepta como mucho 128000).
jax-platform lo cerró leyendo las dos cosas de la fila de `model`
(`backend/contrato_dispatch.py`); acá se aplica la MISMA semántica: nombre de
`model.max_tokens_param`, valor de `model.max_output_tokens`, y NULL o valor
inválido fallan ruidoso con la acción a ejecutar. Nunca se asume un default.

POR QUÉ UN MÓDULO APARTE Y NO facet_resolver. `facet_resolver.py` está
espejado en jax-platform (scripts/check_mirror_sync.py) y la copia de acá NO
selecciona estas columnas: es una divergencia DECLARADA. Además, ninguno de los
dos caminos que se arreglan pasa por `resolve_facet()`: el REPL arma sus
músculos al arrancar con `load_facet_registry()` y puede despachar un modelo
distinto del asignado (MODELO_PESADO). Lo que
hay que leer es la fila del modelo QUE SE DESPACHA, por (provider_id,
model_id). Una sola fuente dentro de jax: el REPL lo importa como
`jax.core.contrato_dispatch` y Jacobs como `contrato_dispatch` pelado, por el
symlink de `las_manos/` (mismo patrón que facet_resolver y grounding_sources).

SIN CACHÉ, A PROPÓSITO (las cuatro del rendimiento, "sin medición previa no hay
caché nuevo"). Es una consulta por clave única (`uk_provider_model`, EXPLAIN en
tests/test_contrato_dispatch_db.py) al lado de una llamada a un LLM que tarda
segundos. Y sin caché, el UPDATE que sugiere el error vale en el turno
siguiente, sin reiniciar nada.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

try:
    # Jacobs / LAS MANOS (cwd las_manos/, jax.core no es importable ahí).
    from facet_resolver import _db_conn
except ImportError:
    # REPL (PYTHONPATH=. desde la raíz del repo).
    from jax.core.facet_resolver import _db_conn


class ModelDispatchConfigError(RuntimeError):
    """El catálogo (`model`) no declara un dato que el dispatch NECESITA para
    armar el request. FAIL-CLOSED y RUIDOSO: nunca se asume un default -- un
    default silencioso es lo que convierte el próximo modelo nuevo en un
    incidente sin síntoma. Mismo nombre y semántica que en jax-platform."""


# Proveedores (provider_id del catálogo) cuyo camino en jax arma un body
# OpenAI-compatible y por eso lee el contrato de la fila. Medido 2026-09-14:
#   - deepseek: HttpMuscle._call_deepseek -> https://api.deepseek.com/chat/completions
#   - openai, moonshot (alias "kimi"), zhipu (alias "zai"): HttpMuscle._call_openai
#   - jacobs/plan.py::_ada_plan y jacobs/executor.py::_invoke_http_openai_compat
#     -> f.base_url + /chat/completions del binding (resolve_facet)
PROVEEDORES_OPENAI_COMPAT = frozenset({"deepseek", "openai", "moonshot", "zhipu"})

# Proveedores que NO son OpenAI-compatibles: tienen su PROPIO parámetro de
# límite de salida, con otro nombre y en otro lugar del body. Declarado para
# que nadie les mande `max_tokens`/`max_completion_tokens` creyendo que es
# universal. Fuentes: documentación oficial, leída por el controller el
# 2026-09-14:
#   - gemini: `generationConfig.maxOutputTokens` en generateContent
#     (https://ai.google.dev/api/generate-content). Hoy
#     HttpMuscle._call_gemini NO manda límite, igual que la Mesa web
#     (jax-platform contrato_dispatch.TRANSPORTS_CON_CONTRATO_DE_DISPATCH).
#   - anthropic: `max_tokens`, OBLIGATORIO en la Messages API ("The maximum
#     number of tokens to generate before stopping"; cada modelo tiene su
#     máximo) (https://platform.claude.com/docs/en/api/messages). En jax no
#     hay camino HTTP a Anthropic: hyde despacha por el CLI `claude`.
#   - ollama nativo: `options.num_predict` en /api/chat
#     (https://github.com/ollama/ollama/blob/main/docs/api.md), el que ya usa
#     jacobs/plan.py::_llm_plan.
PARAMETRO_PROPIO = {
    "gemini": "generationConfig.maxOutputTokens",
    "anthropic": "max_tokens (obligatorio, Messages API)",
    "ollama": "options.num_predict",
}

# Mismo conjunto que el ENUM de model.max_tokens_param (migraciones de
# jax-platform): un valor imposible en la DB no arma una clave arbitraria.
_MAX_TOKENS_PARAM_NAMES = ("max_tokens", "max_completion_tokens")


def _max_tokens_field(model: str, max_tokens_param: str | None) -> str:
    """NOMBRE del parámetro de límite de salida (`model.max_tokens_param`).
    Semántica copiada de jax-platform backend/contrato_dispatch.py: NULL e
    inválido fallan, con el UPDATE a ejecutar."""
    if max_tokens_param is None:
        raise ModelDispatchConfigError(
            f"modelo '{model}': la fila de `model` no declara max_tokens_param, "
            f"así que no se sabe si su API exige 'max_tokens' o "
            f"'max_completion_tokens' y NO se asume ninguno. Sembrala: "
            f"UPDATE model SET max_tokens_param='max_tokens' "
            f"WHERE model_id='{model}';  -- agregá AND provider_id='<provider>' "
            f"si ese model_id existe para más de un proveedor. Usá "
            f"'max_completion_tokens' para los modelos que rechazan el viejo "
            f"con HTTP 400 (generación nueva de OpenAI), 'max_tokens' para el resto."
        )
    if max_tokens_param not in _MAX_TOKENS_PARAM_NAMES:
        raise ModelDispatchConfigError(
            f"modelo '{model}': max_tokens_param={max_tokens_param!r} no es un "
            f"nombre de parámetro conocido {_MAX_TOKENS_PARAM_NAMES}. Corregí la "
            f"fila de `model` -- no se manda una clave arbitraria a la API."
        )
    return max_tokens_param


def _max_output_tokens_value(model: str, max_output_tokens: int | None) -> int:
    """VALOR del límite de salida (`model.max_output_tokens`). No se deriva de
    context_window (ventana total, otro hecho: gpt-5.6-terra 1050000 contra un
    tope de 128000). Semántica copiada de jax-platform."""
    if max_output_tokens is None:
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
        raise ModelDispatchConfigError(
            f"modelo '{model}': max_output_tokens={max_output_tokens!r} no es un "
            f"entero positivo. Corregí la fila de `model` -- no se manda un límite "
            f"inválido a la API."
        )
    return max_output_tokens


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


async def limite_de_salida(provider_id: str, model_id: str) -> dict[str, int]:
    """`{nombre_del_parametro: tope}` para el body OpenAI-compatible del modelo
    que se va a despachar, leído de SU fila de `model`.

    Lanza ModelDispatchConfigError (el llamador NO despacha) si:
      - el proveedor no es OpenAI-compatible (dice cuál es su parámetro propio);
      - la fila no existe;
      - falta o es inválido cualquiera de los dos datos (los junta en un solo
        mensaje: quien siembra ve de una vez todo lo que falta).
    Un error de la DB se propaga tal cual: sin contrato leído no hay dispatch."""
    if provider_id not in PROVEEDORES_OPENAI_COMPAT:
        propio = PARAMETRO_PROPIO.get(provider_id, "desconocido")
        raise ModelDispatchConfigError(
            f"proveedor '{provider_id}' (modelo '{model_id}') no es "
            f"OpenAI-compatible: su límite de salida no es max_tokens/"
            f"max_completion_tokens sino {propio}. No se le arma un body "
            f"OpenAI-compatible."
        )
    fila = await _leer_contrato(provider_id, model_id)
    if fila is None:
        raise ModelDispatchConfigError(
            f"modelo '{model_id}' del proveedor '{provider_id}' no está en el "
            f"catálogo `model`, así que no hay contrato de dispatch que leer y NO "
            f"se asume ninguno. Agregalo al catálogo con max_tokens_param y "
            f"max_output_tokens, o despachá un modelo que sí esté."
        )
    max_tokens_param, max_output_tokens = fila
    errores = []
    campo = tope = None
    try:
        campo = _max_tokens_field(model_id, max_tokens_param)
    except ModelDispatchConfigError as e:
        errores.append(str(e))
    try:
        tope = _max_output_tokens_value(model_id, max_output_tokens)
    except ModelDispatchConfigError as e:
        errores.append(str(e))
    if errores:
        raise ModelDispatchConfigError(" | ".join(errores))
    return {campo: tope}
