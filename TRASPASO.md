# Traspaso — C5 SOLO_ORDENES — 2026-10-06

Estado: implementación en PR abierto, rama `feat/c5-solo-ordenes`. PR JAX #363; último SHA conocido antes de los fixes de CI `24f779be`. No integrar. No tocar PENDIENTES ni DB de producción.

- Spec y cambios JAX en el worktree `/home/fruiz/wt/jax-c5-solo-ordenes`.
- Focal actual: 494 tests recolectadas en diez archivos; `origin/master`: 470; delta +24.
- El C3 selección solo órdenes pasa por `JAX_PROXY_CARRIL_C5_SOCKET` y UID/GID dedicados; el proxy mantiene el escritor único y el vigía falla cerrado si falta el canal.
- Plataforma en worktree separado `/home/fruiz/wt/jxp-c5-solo-ordenes`; PR #209 abierto sobre master. SHA `373809aebcd709229b02262b8ced9d4867bb37e8`, auditado escalón 3 APROBADO con cero hallazgos abiertos.
- Una llamada real thot con contenido sintético y 128 tokens falló en el resolvedor antes de HTTP (`FacetUnavailableError`, log interno `RuntimeError`); no repetir.
- Auditoría de `50b5b683`: el BLOCK de idempotencia cerró y encontró MAJOR en canario, corregido filtrando cada familia a Bash proyectable, fail-closed si queda vacía y prueba integrada con cliente real + `MockTransport`. La auditoría de `bda9fcd6` cerró ese MAJOR y halló BLOCK en instrucciones: el modo solo recibía la sección corta y omitía definiciones compartidas de órdenes prohibidas. Se añadieron reglas compartidas de alcance, secretos y acciones destructivas, manteniendo separadas las instrucciones de claims.
- Auditoría JAX inicial: SHA `24f779be` APROBADO, pero la primera CI remota encontró integraciones faltantes: parámetros de `probar_c5`, config nueva ausente en fixtures DB por jax-platform aún no integrado, y marcas explicativas requeridas para tres capturas fail-soft.
- Fixes de CI en curso: pasar modo/flag explícitos al verificador de canarios, sembrar false en el DB descartable de tests, y anotar por qué se ignoran fallos de limpieza/detalle; pruebas focales de llamadas, parser y scanner: 66 passed. Suite DB completa local no ejecutada porque la protección detecta el puerto de producción.
- C5 plataforma: floors frontend 1391; backend con DB 3838 y sin DB 2366. Sin DB medido completo: 2366 passed / 1474 skipped con `JAX_CI_NO_DB=1` y `JAX_WORKSPACE_DIR` temporal. No se ejecutó DB completa local.
- Último paso pendiente: confirmar nuevo SHA de JAX, auditar escalón 3, actualizar PR #363 y esperar sus checks; después confirmar checks del PR #209. Sin merge/despliegue.
