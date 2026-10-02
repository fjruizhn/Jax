"""
URL de proveedor para los workers de memoria.

E-21 (2026-09-16): `url_del_proveedor()` da a los workers de memoria (extractor y
sintetizador), que no son facetas pero despachan por HttpMuscle, la URL desde
`provider.base_url` del catalogo: ya no tienen URL de proveedor por defecto.

T16 (2026-10-02): se retiro el REPL y con el lo demas de este modulo
(`cargar_registro`, `aplicar_registro`, `_camino_del_modelo`: armaban el modelo y la
lista permitida de las facetas del REPL desde la DB; ningun otro codigo los llamaba).

En memoria de Jairo Urbina.
"""
from __future__ import annotations

from jax.core.facet_resolver import _db_conn
from jax.muscles.base import MuscleInvocationError, _PROVIDER_ID_MAP

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
    catálogo (tabla provider). Antes esos workers no
    pasaban api_url y despachaban a la URL fija de DeepSeek en base.py.
    Consulta por clave primaria (provider.id). Sin fila, sin base_url o con el
    proveedor en status 'deprecated' (deprecated no se invoca, como
    los modelos): MuscleInvocationError y la corrida
    falla visible; no hay URL de respaldo. Si la DB no responde, el error de
    conexión sube tal cual."""
    provider_id = _PROVIDER_ID_MAP.get(clave_http)
    if provider_id is None:
        raise MuscleInvocationError(
            f"sin URL del proveedor: '{clave_http}' no es un proveedor HTTP conocido."
        )
    conn = await _db_conn()
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT base_url, status FROM provider WHERE id = %s", (provider_id,))
            fila = await cur.fetchone()
    finally:
        conn.close()
    if not fila or not fila[0]:
        raise MuscleInvocationError(
            f"sin URL del proveedor: '{provider_id}' no tiene base_url en el catálogo "
            f"(tabla provider); no se despacha a una URL fija."
        )
    if fila[1] != "active":
        raise MuscleInvocationError(
            f"sin URL del proveedor: '{provider_id}' está en status {fila[1]!r} en el catálogo "
            f"(tabla provider); un proveedor que no está active no se despacha."
        )
    return _url_de_despacho(provider_id, fila[0])
