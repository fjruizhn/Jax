# Faro Rule Authority · regresión BaseException de checkpoint

- Objetivo: cerrar el MINOR de Tier 3 de #401 con una regresión para interrupción genérica después de `os.replace`; no cambiar semántica de producción que ya conserva `DURABILITY_UNKNOWN`.
- Base observada: `origin/master` en `4e1a35a7db6861912d3f937b67468d2dee2b324e`. La rama anterior #401 ya está integrada.
- Hecho local: prueba `KeyboardInterrupt` post-replace verifica marcador incierto, retry cerrado durante fallo de directory fsync y recuperación solo al restaurar fsync; floors/docs/historia actualizados. Suite exacta del workflow: `61 passed`; piso JSON verificado.
- Veredicto previo: Tier 3 integral de `4e1a35a7` APROBADO CON CAMBIOS, 0 BLOCK / 0 MAJOR / 1 MINOR. El comentario sobre `4f9006a→7080bc8` solo auditó documentación y no se usa como aprobación del checkpoint.
- Falta: commit y publicar el SHA probado en PR propio; auditoría Tier 3 exacta del artefacto completo y CI exacta; cualquier cambio posterior exige repetir ambos gates. No integrar sin gates y autoridad/ventana aplicable. Producción continúa bloqueada por trusted-root.
- Próximo comando exacto tras commit: `gh pr checks <URL-PR> --watch` para esperar CI; solicitar auditoría adversarial Tier 3 del SHA exacto del PR.
