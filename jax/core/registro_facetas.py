"""
Registro de facetas del REPL: modelo asignado Y lista permitida desde la DB
(2026-09-14).

Hasta hoy main() pisaba solo `model_default` con el modelo de facet_binding y
dejaba `models_allowed` como estaba en config.toml. La lista nadie la mantenía:
el 2026-09-14 ninguno de los 7 modelos vigentes estaba en la suya y el REPL
fallaba con ModelNotAllowedError en todas las facetas (DEUDA.md). La Mesa web y
Jacobs no tienen lista: autoriza el binding.

Decisión de Fernando: la lista permitida sale del catálogo `model` -- los
modelos del MISMO proveedor que el modelo asignado, en estado available/degraded.
Así pasa el modelo asignado y también el modo pesado del REPL (MODELO_PESADO,
del mismo proveedor que jekyll). El asignado queda siempre permitido aunque el
catálogo lo marque de otro modo: es la autoridad.

El proveedor sale del MODELO ASIGNADO (JOIN por model_ref), no de
facet_binding.provider_id: el endpoint de aprobación cambia model_ref sin tocar
provider_id, así que los dos pueden quedar desalineados (revisión de jax#153).

Si la DB no responde al arrancar, no se toca nada y queda config.toml completo:
modelo y lista salen siempre de la misma fuente.

Módulo propio del REPL a propósito: facet_resolver.py está espejado en tres
lugares (mirror-sync) y esto no le sirve a nadie más. Por eso importa
`_db_conn` en vez de agregarle a facet_resolver una función pública: si ese
nombre cambia en el espejo, el import falla al arrancar y el REPL cae al
fallback del TOML con el aviso de main().

E-21 (2026-09-16): también da url_del_proveedor() a los workers de memoria
(extractor y sintetizador), que no son facetas pero despachan por HttpMuscle
y ya no tienen URL de proveedor por defecto.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

from jax.core.facet_resolver import _db_conn, load_facet_registry
from jax.muscles.base import MuscleInvocationError, _PROVIDER_ID_MAP

# Estados del catálogo que se pueden invocar; deprecated/gone no.
ESTADOS_INVOCABLES = ("available", "degraded")

# PR-K ronda 2 (I1): qué `type` de config.toml sirve a cada transporte del
# catálogo (facet.transport). Un binding cuyo transporte no es el camino que
# arma el TOML no se despacha: se bloquea con el motivo.
TIPO_POR_TRANSPORTE = {
    "http_openai_compat": "http",
    "http_gemini": "http",
    "ollama": "ollama",
    "subprocess": "subprocess",
}

# provider_id del catálogo -> clave de proveedor del HttpMuscle. Derivado de
# base._PROVIDER_ID_MAP (una sola tabla), primera clave de cada proveedor.
CLAVE_HTTP_POR_PROVIDER: dict[str, str] = {}
for _clave, _provider_id in _PROVIDER_ID_MAP.items():
    CLAVE_HTTP_POR_PROVIDER.setdefault(_provider_id, _clave)


async def cargar_registro() -> dict:
    """load_facet_registry() + `models_allowed` por faceta, en UNA consulta
    más. Lanza si la DB no responde: quien llama decide el fallback."""
    registro = await load_facet_registry()
    permitidos: dict[str, list[str]] = {clave: [] for clave in registro}
    if registro:
        conn = await _db_conn()
        try:
            async with conn.cursor() as cur:
                claves = ", ".join(["%s"] * len(registro))
                estados = ", ".join(["%s"] * len(ESTADOS_INVOCABLES))
                # `asignado` = el modelo del binding; `hermano` = cada modelo del
                # mismo proveedor que él. uk_provider_model cubre el segundo JOIN.
                await cur.execute(
                    "SELECT b.facet_key, hermano.model_id "
                    "FROM facet_binding b "
                    "JOIN model asignado ON asignado.id = b.model_ref "
                    "JOIN model hermano ON hermano.provider_id = asignado.provider_id "
                    f"WHERE b.role = 'primary' AND b.facet_key IN ({claves}) "
                    f"AND hermano.status IN ({estados})",
                    (*registro.keys(), *ESTADOS_INVOCABLES),
                )
                for clave, modelo in await cur.fetchall():
                    permitidos.setdefault(clave, []).append(modelo)
                # PR-K ronda 2 (I1): transporte de la faceta y proveedor + URL
                # base del MODELO asignado (no de facet_binding.provider_id, que
                # approve puede dejar desalineado). Todo por claves primarias.
                await cur.execute(
                    "SELECT b.facet_key, f.transport, asignado.provider_id, p.base_url "
                    "FROM facet_binding b "
                    "JOIN facet f ON f.`key` = b.facet_key "
                    "JOIN model asignado ON asignado.id = b.model_ref "
                    "JOIN provider p ON p.id = asignado.provider_id "
                    f"WHERE b.role = 'primary' AND b.facet_key IN ({claves})",
                    tuple(registro.keys()),
                )
                for clave, transporte, provider_id, base_url in await cur.fetchall():
                    registro[clave].update(
                        transport=transporte, provider_modelo=provider_id, base_url_modelo=base_url,
                    )
        finally:
            conn.close()
    # Orden en Python y no en SQL: un ORDER BY acá daba Using temporary +
    # filesort (EXPLAIN, 2026-09-14) por algo que solo sirve para que la lista
    # sea estable entre arranques.
    for clave, info in registro.items():
        info["models_allowed"] = sorted(permitidos.get(clave, []))
    return registro


def aplicar_registro(cfg: dict, registro: dict) -> None:
    """Pone en cfg["personalities"] el modelo asignado y su lista permitida.
    Puro. Una faceta del registro que no está en el TOML no se inventa (no
    tendría system prompt ni tipo). Registro vacío = no se toca nada."""
    for clave, info in registro.items():
        personalidad = cfg["personalities"].get(clave)
        if personalidad is None:
            continue
        permitidos = list(info.get("models_allowed") or [])
        if info["model"] not in permitidos:
            permitidos.insert(0, info["model"])
        personalidad["model_default"] = info["model"]
        personalidad["models_allowed"] = permitidos
        motivo = _camino_del_modelo(clave, personalidad, info)
        if motivo:
            personalidad["dispatch_bloqueado"] = motivo


def _camino_del_modelo(clave: str, personalidad: dict, info: dict) -> str:
    """PR-K ronda 2 (I1): el proveedor, la URL y el camino salen del MODELO
    del binding, no de config.toml. Pone en `personalidad` el proveedor y la
    URL de ese modelo, o devuelve el motivo por el que la faceta NO puede
    despachar (quedará bloqueada con ese error visible). Sin datos de
    transporte en el registro (tests viejos, registro sin la 2da consulta) no
    toca nada."""
    transporte = info.get("transport")
    if transporte is None:
        return ""
    provider_id = info.get("provider_modelo")
    base_url = info.get("base_url_modelo")
    tipo = personalidad.get("type")
    if TIPO_POR_TRANSPORTE.get(transporte) != tipo:
        return (
            f"el binding de '{clave}' usa el transporte '{transporte}' (modelo "
            f"'{info['model']}' de '{provider_id}') y config.toml la arma como "
            f"type='{tipo}': no se despacha por un camino que no es el del modelo. "
            f"Corregí el binding o el type de la faceta en config.toml."
        )
    if tipo == "ollama":
        personalidad["provider_id"] = provider_id
        return ""
    if tipo != "http":
        return ""
    clave_http = CLAVE_HTTP_POR_PROVIDER.get(provider_id)
    if clave_http is None or (transporte == "http_gemini") != (provider_id == "gemini"):
        return (
            f"el modelo '{info['model']}' de '{clave}' es del proveedor "
            f"'{provider_id}' (transporte '{transporte}'), que el REPL no sabe "
            f"despachar por HTTP: no se lo manda a la URL de otro proveedor."
        )
    if not base_url:
        return (
            f"el proveedor '{provider_id}' del modelo de '{clave}' no tiene "
            f"base_url en el catálogo (tabla provider): sin URL no se despacha."
        )
    personalidad["provider"] = clave_http
    personalidad["api_url"] = _url_de_despacho(provider_id, base_url)
    return ""


def _url_de_despacho(provider_id: str, base_url: str) -> str:
    """La URL que HttpMuscle usa de api_url, desde provider.base_url: Gemini
    arma la ruta del modelo sobre la base; los OpenAI-compatibles van a
    /chat/completions. Una sola regla para el REPL y los workers de memoria."""
    if provider_id == "gemini":
        return base_url
    return base_url.rstrip("/") + "/chat/completions"


async def url_del_proveedor(clave_http: str) -> str:
    """E-21 (2026-09-16): api_url de un HttpMuscle que NO es una faceta (el
    extractor y el sintetizador de la memoria), desde provider.base_url del
    catálogo, la misma columna que usa aplicar_registro. Antes esos workers no
    pasaban api_url y despachaban a la URL fija de DeepSeek en base.py.
    Consulta por clave primaria (provider.id). Sin fila o sin base_url:
    MuscleInvocationError y la corrida falla visible; no hay URL de respaldo.
    Si la DB no responde, el error de conexión sube tal cual."""
    provider_id = _PROVIDER_ID_MAP.get(clave_http)
    if provider_id is None:
        raise MuscleInvocationError(
            f"sin URL del proveedor: '{clave_http}' no es un proveedor HTTP conocido."
        )
    conn = await _db_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT base_url FROM provider WHERE id = %s", (provider_id,))
            fila = await cur.fetchone()
    finally:
        conn.close()
    if not fila or not fila[0]:
        raise MuscleInvocationError(
            f"sin URL del proveedor: '{provider_id}' no tiene base_url en el catálogo "
            f"(tabla provider); no se despacha a una URL fija."
        )
    return _url_de_despacho(provider_id, fila[0])
