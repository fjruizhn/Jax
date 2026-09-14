"""
Registro de facetas del REPL: modelo asignado Y lista permitida desde la DB
(2026-09-14).

Hasta hoy main() pisaba solo `model_default` con el modelo de facet_binding y
dejaba `models_allowed` como estaba en config.toml. La lista nadie la mantenía:
el 2026-09-14 ninguno de los 7 modelos vigentes estaba en la suya y el REPL
fallaba con ModelNotAllowedError en todas las facetas (DEUDA.md). La Mesa web y
Jacobs no tienen lista: autoriza el binding.

Decisión de Fernando: la lista permitida sale del catálogo `model` -- los
modelos del MISMO proveedor de la faceta en estado available/degraded. Así pasa
el modelo asignado y también el modo pesado del REPL (MODELO_PESADO, del mismo
proveedor que jekyll). El asignado queda siempre permitido aunque el catálogo lo
marque de otro modo: es la autoridad.

Si la DB no responde al arrancar, no se toca nada y queda config.toml completo:
modelo y lista salen siempre de la misma fuente.

Módulo propio del REPL a propósito: facet_resolver.py está espejado en tres
lugares (mirror-sync) y esto no le sirve a nadie más.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

from jax.core.facet_resolver import _db_conn, load_facet_registry

# Estados del catálogo que se pueden invocar; deprecated/gone no.
ESTADOS_INVOCABLES = ("available", "degraded")


async def cargar_registro() -> dict:
    """load_facet_registry() + `models_allowed` por faceta, en UNA consulta
    más (por los proveedores presentes). Lanza si la DB no responde: quien
    llama decide el fallback."""
    registro = await load_facet_registry()
    proveedores = sorted({i["provider_id"] for i in registro.values() if i.get("provider_id")})
    por_proveedor: dict[str, list[str]] = {p: [] for p in proveedores}
    if proveedores:
        conn = await _db_conn()
        try:
            async with conn.cursor() as cur:
                marcas = ", ".join(["%s"] * len(proveedores))
                estados = ", ".join(["%s"] * len(ESTADOS_INVOCABLES))
                await cur.execute(
                    f"SELECT provider_id, model_id FROM model "
                    f"WHERE provider_id IN ({marcas}) AND status IN ({estados}) "
                    f"ORDER BY provider_id, model_id",
                    (*proveedores, *ESTADOS_INVOCABLES),
                )
                for proveedor, modelo in await cur.fetchall():
                    por_proveedor[proveedor].append(modelo)
        finally:
            conn.close()
    for info in registro.values():
        info["models_allowed"] = list(por_proveedor.get(info.get("provider_id"), []))
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
