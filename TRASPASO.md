# Traspaso — C5 SOLO_ORDENES — 2026-10-06

Estado: implementación en curso en `feat/c5-solo-ordenes`, basada en `685579466c25771a4df5d5cfd2c51bc555a21fa6`. No integrar. No tocar PENDIENTES ni DB de producción.

- Spec y cambios JAX en el worktree `/home/fruiz/wt/jax-c5-solo-ordenes`.
- Focal actual: 462 passed; original/base: 447 recolectadas; delta +15.
- Seed/admin de plataforma aún sin cambios. Inspección indica `/admin/config` es superadmin, audita todas las escrituras y no tiene catálogo de claves; el componente AdminSettings sí requiere un control visible con i18n. `backend/ejecutor/misiones.py::maquinas()` tiene un gate separado que debe aceptar el flag solo órdenes.
- Luego abrir PR JAX antes de crear `/home/fruiz/wt/jxp-c5-solo-ordenes` y hacer PR independiente de plataforma.
- Falta prueba thot real con contenido sintético y tokens bajos; nunca imprimir credenciales.
- Falta Tier 3 independiente del SHA final, re-medir piso al rebasar, y actualizar este archivo con SHA/PR/resultados.
