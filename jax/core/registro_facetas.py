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

En memoria de Jairo Urbina.
"""
from __future__ import annotations

from jax.core.facet_resolver import _db_conn, load_facet_registry

# Estados del catálogo que se pueden invocar; deprecated/gone no.
ESTADOS_INVOCABLES = ("available", "degraded")


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
