# Traspaso — C5 SOLO_ORDENES — 2026-10-06

Estado: implementación en curso en `feat/c5-solo-ordenes`, rebasada sobre `origin/master` (`f47820f5`). No integrar. No tocar PENDIENTES ni DB de producción.

- Spec y cambios JAX en el worktree `/home/fruiz/wt/jax-c5-solo-ordenes`.
- Focal actual: 494 tests recolectadas en diez archivos; `origin/master`: 470; delta +24.
- El C3 selección solo órdenes pasa por `JAX_PROXY_CARRIL_C5_SOCKET` y UID/GID dedicados; el proxy mantiene el escritor único y el vigía falla cerrado si falta el canal.
- Seed/admin de plataforma aún sin cambios. Inspección indica `/admin/config` es superadmin, audita todas las escrituras y no tiene catálogo de claves; el componente AdminSettings sí requiere un control visible con i18n. `backend/ejecutor/misiones.py::maquinas()` tiene un gate separado que debe aceptar el flag solo órdenes.
- Luego abrir PR JAX antes de crear `/home/fruiz/wt/jxp-c5-solo-ordenes` y hacer PR independiente de plataforma.
- Una llamada real thot con contenido sintético y 128 tokens falló en el resolvedor antes de HTTP (`FacetUnavailableError`, log interno `RuntimeError`); no repetir.
- Auditoría de `50b5b683`: el BLOCK de idempotencia cerró y encontró MAJOR en canario, corregido filtrando cada familia a Bash proyectable, fail-closed si queda vacía y prueba integrada con cliente real + `MockTransport`. La auditoría de `bda9fcd6` cerró ese MAJOR y halló BLOCK en instrucciones: el modo solo recibía la sección corta y omitía definiciones compartidas de órdenes prohibidas. Se añadieron reglas compartidas de alcance, secretos y acciones destructivas, manteniendo separadas las instrucciones de claims.
- Verificación tras compartir el contrato de órdenes: focal 494 passed; controles de pisos 227 passed; collect-only baseline `origin/master` 470 vs rama 494; diff-check y JSON válidos. Falta commit y auditoría Tier 3 del SHA actual.
- Falta commit de los fixes de revisión, Tier 3 independiente del SHA corregido, abrir PR JAX y luego crear `/home/fruiz/wt/jxp-c5-solo-ordenes` para la PR de plataforma. Actualizar este archivo con los SHA/PR/resultados.
