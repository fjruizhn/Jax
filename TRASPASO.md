# Traspaso — El Faro 0.6 / R-2

- Fecha de apertura: 2026-10-09
- Responsable: Codex, hall9000
- Rama: `codex/faro-r2`
- Actualizado: 2026-10-12

## Estado vigente — 2026-10-12

Este encabezado reemplaza los estados operativos anteriores de este archivo; las secciones fechadas que siguen se
conservan como historia. R-2 no está completo, integrado ni desplegado.

- Fuente de código oficial observada: `origin/master@4e1a35a7db6861912d3f937b67468d2dee2b324e`. La producción viva
  está en `e9895512`; ninguno de los dos SHA acredita por sí mismo que Faro esté habilitado.
- Jax#402 fue reapilado y publicado por su owner en `aeb3ab10f3c6ffb9070307526f351da8e700b9cf`. Los 33/33 checks y
  el veredicto Tier 3 APROBADO correspondían al head anterior `669558b4`; auditoría nueva y CI exacta siguen
  pendientes sobre `aeb3ab10`, antes de su preflight y guard. No es una integración de R-2 ni autoriza reapilar esta
  rama sin volver a medir sobre el tip que resulte oficial.
- Contrato R-2 v3 vigente: lease durable con `PREPARED → REQUESTED → ACCEPTED → CLIENT_CLOSED`; no se libera por
  cambio de epoch. La aceptación conserva ruta de cgroup y su identidad `st_dev/st_ino`; el release exige cierre del
  emisor, evidencia kernel del mismo inode y namespace vacío. La unidad productiva debe usar `Delegate=no` y la
  jaula no puede migrar su cgroup.
- Rebase local terminado sobre `origin/master@ecfe07f3c2760ed43f53e7d1e858dad8e864e7e1`. Verificación local
  posterior al rebase, Python 3.14: selección CI exacta **955 passed, 6 deselected**, 60.79 s
  (`/tmp/faro-r2-integrated-py314-20261012-rebased.log`); jaula real, **12 passed**
  (`/tmp/faro-r2-faro-jaula-20261012-rebased.log`); integración systemd real, **6 passed, 158 deselected**
  (`/tmp/faro-r2-systemd-integration-20261012-rebased.log`); cableado, **7 passed**
  (`/tmp/faro-r2-wireados-20261012-rebased.log`). Falta asociar esta evidencia al SHA del commit, auditoría Tier 3
  exacta, CI remota, integración y post-merge guard. Los conteos de scope (151) y control (182) son pruebas dirigidas
  previas, no verificación final independiente.
- La auditoría Tier 3 previa señaló **1 BLOCK y 2 MAJOR**; las correcciones están en este WIP y requieren auditoría
  adversarial nueva sobre el SHA exacto. No presentar las pruebas locales ni el contrato como aprobación de integración.
- Block 4 sigue no disponible: `jaxctl authority --json` devolvió `UNAVAILABLE`; el diagnóstico de ledger conserva
  `OperationalError 1142`. No existen la raíz ni checkpoints externos esperados y no hay fuente autorizada de
  restauración. No crear, sustituir ni sintetizar ese material.
- `jaxqwen` está instalado, su servicio está `disabled/inactive` y `dispatch_enabled=false`. Las Voces mediante
  esa ruta son independientes de terminar todas las fases de Faro, pero el ejecutor seguro por nombre todavía no
  existe; no es la primera misión de este tramo.
- Siguen pendientes 0.5 y 0.7–0.10, incluida carga, evidencia de confianza y verificación runtime. Faro no está
  listo para producción.

- Verdad operacional observada el 2026-10-10 20:38 CST por Codex: el nuevo `fetch` sitúa `origin/master` en `4e1a35a7db6861912d3f937b67468d2dee2b324e`. La ventana de Fernando devuelve `abierta: true`. No hay integración ni despliegue de esta rama.
- Auditoría más reciente (Tier 3, `audit_r2_lease`, sobre `HEAD@5d1fac50` + diff fingerprint `953bb0edb9d4e28dfaa539df11017f66c0e6272415d3c79912014b6232d00b8a`): **RECHAZADO, 4 BLOCK / 4 MAJOR**. BLOCK: namespace de unidades por attempt invisible si falta lease; cierre anterior a adquisición del lease y tras terminación natural; cancelación durante cierre del Puerto antes de registrar cleanup; recuperación de lease ausente que bloquea permanentemente toda admisión. MAJOR: falta `--collect`; `argv_matar` aún acepta ausencia de attempt; tareas de limpieza globales y retención de tareas terminadas; el servicio no posee/drena automáticamente los watchers del lanzador. Este resultado sustituye el inventario anterior como estado vigente, sin borrar su historia.
- Última validación previa a esa auditoría: selección exacta del workflow en Python 3.12, **868 passed, 6 deselected**, cero skips; seis integraciones reales de systemd aprobadas. El comparador de piso y `git diff --check` pasaron. Estos resultados no demuestran los escenarios ausentes que identificó el auditor y deben repetirse tras la corrección.
- En curso: arquitecto Sol cierra un contrato único de cierre/recuperación por UID; implementador Terra corrige los dos defectos de argv ya diagnosticados. Un explorador verifica la disponibilidad productiva de Block 4 y el estado versionado de fases 0.5–0.10. No hay peers Claude alcanzables; Kimi sigue mostrando 403 de cuota en su pane.
- Base al iniciar: `origin/master@709237f901e8ec8840f6b2b39a1a2ce88682674a`. La base oficial actual verificada por fetch/API es `origin/master@3d39a313f93142232fa0b8331d06355536200d97`, todavía no reapilada en esta rama.
- Estado (2026-10-10): auditoría Tier 3 del diff actual `5d1fac50 + WIP` rechazada: 4 BLOCK y 1 MAJOR. BLOCKs: (1) cierre ordinario puede liberar lease mientras un start/reaper aún puede materializar; falta drenaje por UID antes de liberar; (2) si el primer drenaje falla, `__aexit__` omite cerrar ejecuciones vivas; (3) recuperación usa `show → list-jobs → show` como barrera causal y además corre antes de comprobar si ya existe el socket; (4) kill por nombre de unidad tiene TOCTOU y no valida identidad exacta inmediatamente antes de matar. MAJOR: errores del waiter/drenaje stderr pueden perderse en `jaula.py` al retirar la tarea terminada. El diff local contiene lease durable, recuperación pre-socket y pruebas, pero todavía permite las carreras indicadas. No publicar ni integrar. Se pidió al auditor una propuesta mínima de contrato/regresiones, pendiente.
- Blocker de producción revalidado: `/etc/jax/authority` y `/var/lib/jax/authority` no existen; `jaxctl authority --json` como `jaxsvc` devuelve `UNAVAILABLE / Block 4 authority source unavailable`. No se encontró una fuente autorizada de raíz/checkpoints; no restaurar ni sintetizar material sin procedencia válida.
- Coordinación remota (2026-10-10): `master@3d39a313`; #398 sigue OPEN (`a9de35d6`) con base vieja `765406f4`, estado CLEAN, checks observados SUCCESS; #375 OPEN (`6ae6986a`) sobre base vieja `61fd79fb`, DIRTY. Son ramas ajenas: no modificarlas ni integrarlas desde este worktree.
- R2 sigue sin commit; la rama no está reapilada. Faltan corregir/auditar los bloqueos, integrar sobre el master vigente por la secuencia autorizada, CI exacta, publicación/PR y guardas. También faltan fases 0.5, 0.7–0.10, evidencia de confianza, load test y verificación del despliegue. No instalar ni desplegar R2 en este tramo.

## Actualización de coordinación 2026-10-10 (Codex)

Esta sección sustituye como estado actual los reportes históricos de cuota, base y contrato anteriores.

- Base verificada: `origin/master@4e1a35a7db6861912d3f937b67468d2dee2b324e`; el push policy `38105335278` terminó SUCCESS. Rebase de la rama propia pendiente hasta terminar ambos writers.
- Contrato técnico R-2 cerrado después de la revisión Sol: stages `PREPARED → REQUESTED → ACCEPTED → CLIENT_CLOSED`. `ACCEPTED` exige observar la Description exacta de la unidad propia. El wait de un cliente que terminó sin aceptación no prueba que el manager rechazó una petición en tránsito; queda `REQUESTED` en cuarentena. Recovery sin handle en REQUESTED/ACCEPTED retiene solo ese UID; una época nueva del manager no libera leases. PREPARED y CLIENT_CLOSED necesitan namespace completo vacío.
- Dos implementadores Terra tienen ownership disjunto: scope/lease/control y tests correspondientes; launcher/servicio y sus tests. El segundo terminó y su selección dirigida pasó 102/102. El primero valida materialización tardía y la carrera entre fin natural y cierre solicitado. No hay auditoría final nueva ni publicación.
- Kimi respondió a la tarea nueva y está cerrando evidencia factual de systemd 259; el 403 anterior no describe su disponibilidad actual. ZCode/GLM fue probado de nuevo con consulta mínima y devolvió 429/1310: Weekly/Monthly Limit Exhausted, reset anunciado `2026-10-14 01:37:49` (zona no indicada). No cambió archivos.
- El master vigente usa Python 3.14 para faro-fase0. La selección final se medirá en ese intérprete después del rebase, conservando el cableado nuevo de scope y sin asumir el piso histórico 868.
- Fuente productiva de B4 sigue UNAVAILABLE; pregunta de procedencia de raíz/checkpoints enviada a Fernando, sin respuesta todavía. No crear llaves/firmas ni convertir un GO operacional en ratificación.
- Próximo comando tras liberar ownership: selección completa scope/control para diagnosticar, commit con este traspaso y publicación de progreso en rama propia; luego rebase, medición exacta y archivado de este archivo antes de auditoría final. Las auditorías históricas rechazadas no autorizan integración.

## Implementación

- `jax/faro/scope.py`: unidad transitoria `faro-jaula-u<uid>.service` del user manager; límites finitos `MemoryMax`, `CPUQuota`, `TasksMax`; estado `LIVE/EMPTY/UNKNOWN`; `--pipe`, `--collect`, `Type=exec`, `KillMode=control-group`, `OOMPolicy=kill`; verifica propiedades efectivas y falla cerrado.
- `jax/faro/control.py`: consulta liveness incluso sin reserva local; `UNKNOWN` retiene el UID; cierre Puerto → SIGKILL de la unidad → espera cgroup vacío → liberar UID solo en `EMPTY`.
- `jax/faro/jaula.py` y `jax/faro/servicio.py`: el lanzador exige el gestor y ejecuta el argv completo dentro de la unidad; `Servicio.control(scope_manager)` falla si falta el proveedor.
- `jax/faro/scope.py`: `LoadState` distingue `ABSENT/OURS/OTHER/UNKNOWN`; `list-jobs` impide tratar `not-found` como ausente mientras haya un start en cola. Cada lanzamiento se registra antes del primer `await`; el cierre cierra admisión, cancela lanzamientos y drena. Errores/esperas vencidas mantienen el reaper y hacen fallar el apagado visiblemente, sin liberar el UID.
- `jax/faro/control.py`: el cierre del servidor cierra admisión antes de cancelar conexiones y elimina el socket en `finally`, incluso si el drenaje falla.
- `docs/superpowers/plans/2026-10-02-faro-fase-0.md`: desviación 22 documenta `.service` en lugar de `.scope` porque `faro` no tiene autorización para el system manager ni broker; requiere cuenta `faro` con user manager/linger en 0.10. El snapshot histórico lo describía como implementado, sujeto a auditoría e integración; el estado vigente está en el encabezado 2026-10-12 y sigue siendo WIP.
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
- Selección exacta del workflow (`-m "not systemd_integration"`) pasó en Python 3.12 tras el lease durable y las regresiones: `863 passed, 6 deselected`, cero skips. `ci/pisos.json` y `docs/ci/pisos.md` suben a `^863 passed`; el comparador pasa. Las seis pruebas reales `systemd_integration` pasaron después, incluido drenaje de salida mayor que una tubería.
- Kimi `k1` está viva pero responde 403 por límite de cinco horas. ZCode/GLM no tiene sesión de modelo activa en `g6`; la pane conserva respuesta previa y límite semanal agotado hasta 2026-10-14. `ListAgents` de Claude no encontró peers alcanzables. Se solicitó reauditoría Tier 3 del diff local; el resultado aún está pendiente.

## Mapa del Faro y coordinación

- Kimi verificó contra `origin/master@d0c8a68f` que 0.4 ya quedó implementado y fusionado en Jax#352 (`f69f045a`); solo espera el cable productivo, bloqueado por reactivación de B9/Jax#276 y su contrato.
- 0.5 no tiene proxy versionado. S-1 no tiene evidencia técnica archivada por motor; revalidarlo es compuerta antes del proxy. La desviación 14 (red del perfil de motor) sigue pendiente y debe resolverse antes de 0.7.
- 0.5 y 0.6 son paralelos desde 0.2; 0.7 depende de ambos. 0.8, 0.9 (carga/Biblioteca) y 0.10 (instalación) siguen después. No desplegar ni conectar B9 de producción en este tramo.
- c2 lleva la cadena F1.1 y su rama de providers. No tocar sus worktrees, ramas ni marcas. Ya se le informó que 0.4 está hecho y se pidió confirmar propiedad de la línea compartida 0.4/0.5 en `PENDIENTES.md`.

## Siguiente

1. Implementar lease durable y recovery pre-socket; conservar la cuarentena en timeout/crash y añadir las regresiones de crash/restart y colisión de UID.
2. #398 y #400 ya muestran sus dos checks pendientes como SUCCESS, pero son PRs de otros owners y aún OPEN; esperar sus merges/guards y usar el master oficial realmente vigente.
3. Commit con coautoría Codex, publicar la rama propia después del veredicto, abrir/actualizar PR y enlazar CI exacta.
4. Reapilar sobre el master final; re-medír piso y repetir el workflow exacto y las pruebas de integración.
5. Antes de auditoría final, registrar este traspaso en la Biblioteca y quitar `TRASPASO.md` del artefacto auditado. Auditoría Tier 3 independiente del SHA exacto; todo cambio obliga auditoría y CI nuevas.
6. Revalidar ventana y sello `verify-source`, usar `verify-integration`, merge por URL exacta y ejecutar `post-merge-guard`. R-2 no habilita producción por sí solo: Faro sigue bloqueado por 0.5/0.7/0.8/0.9/0.10 y raíz/checkpoint confiables aún no demostrados.
