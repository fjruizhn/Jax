# Traspaso — El Faro 0.6 / R-2

- Fecha: 2026-10-09
- Responsable: Codex, hall9000
- Rama: `codex/faro-r2`
- Actualizado: 2026-10-11
- Base al iniciar: `origin/master@709237f901e8ec8840f6b2b39a1a2ce88682674a`. #379/#392/#399 se integraron después; última base oficial verificada `origin/master@765406f4475ac5babb8ec6af2e6c92b91aa65344`, todavía no reapilada en esta rama.
- Estado: implementación R-2 previa y sus pruebas siguen locales; el diff nuevo corrige la cancelación repetida y hace que el apagado espere las limpiezas. Aún no está commit/push, rebasado, auditado ni integrado.

## Implementación

- `jax/faro/scope.py`: unidad transitoria `faro-jaula-u<uid>.service` del user manager; límites finitos `MemoryMax`, `CPUQuota`, `TasksMax`; estado `LIVE/EMPTY/UNKNOWN`; `--pipe`, `--collect`, `Type=exec`, `KillMode=control-group`, `OOMPolicy=kill`; verifica propiedades efectivas y falla cerrado.
- `jax/faro/control.py`: consulta liveness incluso sin reserva local; `UNKNOWN` retiene el UID; cierre Puerto → SIGKILL de la unidad → espera cgroup vacío → liberar UID solo en `EMPTY`.
- `jax/faro/jaula.py` y `jax/faro/servicio.py`: el lanzador exige el gestor y ejecuta el argv completo dentro de la unidad; `Servicio.control(scope_manager)` falla si falta el proveedor.
- `docs/superpowers/plans/2026-10-02-faro-fase-0.md`: desviación 22 documenta `.service` en lugar de `.scope` porque `faro` no tiene autorización para el system manager ni broker; requiere cuenta `faro` con user manager/linger en 0.10. R-2 queda descrito como implementado, sujeto a auditoría e integración.
- CI incorpora `tests/test_faro_scope.py`; las cuatro pruebas que requieren el user manager llevan `systemd_integration` y se excluyen del runner hospedado, sin convertirlas en skips.
- La configuración de `TasksMax` acepta enteros positivos. Se retiró el sufijo `K/M/G`: el manual local de systemd 259 documenta entero o porcentaje, y `systemd-run --user --property=TasksMax=1K` falló realmente con `Invalid argument`. TDD: la regresión `[4K]` falló contra la versión previa y pasó tras corregir el parser.

## Verificación

- Comando exacto de `faro-fase0` con dependencias instaladas desde los dos requirements con hash: `834 passed, 4 deselected`, cero skips; verificador de piso aprobado con `^834 passed`.
- `tests/test_faro_scope.py`: `102 passed`, incluidas las cuatro integraciones reales con el user manager; comprobó límites efectivos, hijos/nietos, aislamiento entre unidades y lanzamiento real de `bwrap` falso con elevación controlada.
- Job `faro-jaula`: `11 passed`, piso verificado y cero skips. Protección de cableado de archivos de test: `7 passed`.
- El doble de systemd cubre carreras de UID, unidad ocupada ajena, fallos de consulta, límites alterados, cancelación y orden de cierre.
- `git diff --check` limpio. `systemctl --user list-units 'faro-jaula-u*'` vacío después de las pruebas.
- 2026-10-11: la regresión `test_major2_cancelacion_repetida_y_materializacion_tardia_no_deja_unidad_viva` falló antes del cambio con `muertes=[]`; luego pasó con la limpieza registrada que conserva el spawn bajo cancelaciones repetidas. La prueba reproduce materialización >1 s y verifica muerte de la unidad/estado `EMPTY`.
- 2026-10-11: prueba nueva del apagado falla antes porque `ServidorControl.__aexit__` no esperaba `ScopeManager.esperar_limpiezas`; pasa tras añadir esa espera después de cerrar las ejecuciones.
- Con las dependencias exactas de CI (`pip install --require-hashes -r requirements-faro.txt` y `requirements-faro-ci.txt`) en `/tmp/jax-faro-r2-venv`, la selección exacta `faro-fase0` pasó: `844 passed, 5 deselected` (piso vigente 834). Cinco pruebas dirigidas de cancelación/cierre también pasaron. No se ejecutaron aún las cuatro pruebas reales del user manager.
- Revisión Tier 3 de arquitectura actualizada solicitada; no es todavía auditoría final de SHA. GLM/ZCode agotó límite semanal el 2026-10-11; no pudo entregar la revisión concurrente.

## Mapa del Faro y coordinación

- Kimi verificó contra `origin/master@d0c8a68f` que 0.4 ya quedó implementado y fusionado en Jax#352 (`f69f045a`); solo espera el cable productivo, bloqueado por reactivación de B9/Jax#276 y su contrato.
- 0.5 no tiene proxy versionado. S-1 no tiene evidencia técnica archivada por motor; revalidarlo es compuerta antes del proxy. La desviación 14 (red del perfil de motor) sigue pendiente y debe resolverse antes de 0.7.
- 0.5 y 0.6 son paralelos desde 0.2; 0.7 depende de ambos. 0.8, 0.9 (carga/Biblioteca) y 0.10 (instalación) siguen después. No desplegar ni conectar B9 de producción en este tramo.
- c2 lleva la cadena F1.1 y su rama de providers. No tocar sus worktrees, ramas ni marcas. Ya se le informó que 0.4 está hecho y se pidió confirmar propiedad de la línea compartida 0.4/0.5 en `PENDIENTES.md`.

## Siguiente

1. Revisar el diff nuevo y el informe independiente Tier 3; cerrar cualquier BLOCK/MAJOR, incluida la esquina de fallo al consultar la identidad.
2. Revalidar #398 y #400: esperar CI exacta y `post-merge-guard`; después tomar el `origin/master` realmente vigente.
3. Commit pequeño con coautoría Codex, actualizar este handoff y publicar la rama propia.
4. Reapilar sobre el master actual; medir de nuevo piso y repetir exactamente `faro-fase0` más las pruebas de integración disponibles.
5. Borrar `TRASPASO.md` antes de pedir auditoría final y registrar su contenido en la Biblioteca. Ejecutar auditoría Tier 3 independiente sobre el SHA exacto; cualquier cambio exige nueva auditoría y CI.
6. Revalidar ventana, sello `verify-source`, `verify-integration`, merge por PR y `post-merge-guard`. No desplegar R-2: Faro sigue NO-GO de producción hasta completar 0.5/0.7/0.8/0.9/0.10 y la raíz/reglas verificadas.
