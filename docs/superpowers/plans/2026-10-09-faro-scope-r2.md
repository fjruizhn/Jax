# El Faro 0.6 y R-2: unidad cgroup por ejecución

Fecha: 2026-10-09
Estado vigente al 2026-10-12: implementación local WIP; no integrado ni desplegado
Rama: `codex/faro-r2`
Base: `origin/master@709237f901e8ec8840f6b2b39a1a2ce88682674a`

## Estado

R-2 está en implementación en `codex/faro-r2`; el cambio local agrega lease durable y recovery antes del socket.
La selección CI extendida más reciente corrió en Python 3.14 y obtuvo `904 passed, 6 deselected` en 60.55 s
(`/tmp/faro-r2-integrated-py314-20261012-final.log`); la jaula real obtuvo `12 passed` y las seis integraciones
reales de systemd `6 passed, 158 deselected` en sus logs fechados de `/tmp`. Es evidencia local anterior al rebase,
no una aprobación: la auditoría Tier 3 previa tenía 1 BLOCK y 2 MAJOR, sus correcciones necesitan auditoría nueva;
faltan
commit, reconciliar la base oficial que resulte tras Jax#402 (reapilado por su owner en
`aeb3ab10f3c6ffb9070307526f351da8e700b9cf`, aún pendiente de auditoría/CI exactas), volver a medir el piso,
auditoría adversarial del SHA exacto, CI exacta, integración y guardas. No se instala ni despliega en este trabajo.

Verificado el 2026-10-09 en systemd 259: `tests/test_faro_scope.py` — 102 passed, incluidas cuatro pruebas
reales del user manager; comando exacto de CI `faro-fase0` — 834 passed, 4 deselected, cero skips. El caso
`TasksMax=4K` quedó rechazado: systemd rechaza ese sufijo con `Invalid argument`; la configuración ahora
acepta solo enteros positivos.

Verificación local 2026-10-10 con el lease: selección exacta `faro-fase0` en Python 3.12 —
863 passed, 5 deselected (`systemd_integration`), cero skips. El comparador de piso debe pasar sobre el mismo
log; las cinco integraciones reales del user manager siguen pendientes de repetirse después del cambio.

## Objetivo

Cada ejecución lanzada por Faro debe correr dentro de un cgroup exclusivo con límites explícitos de memoria, CPU y tareas. La retención de su uid debe reflejar la población de ese cgroup, incluso si muere el proceso cliente o se reinicia el servicio. Todo estado que no se pueda verificar mantiene el uid ocupado. Al conectar `LanzadorJaula.lanzar` al canal de `Servicio`, el proveedor de liveness será obligatorio.

La prueba de freno debe matar todos los procesos de la ejecución, incluidos hijos y nietos, sin matar al servicio ni a ejecuciones hermanas.

## Evidencia de partida

- `docs/superpowers/plans/2026-10-02-faro-fase-0.md:141-143`: 0.6 requiere una unidad systemd/cgroup por ejecución con `MemoryMax`, `CPUQuota` y `TasksMax`, y demostrar que el freno mata el árbol.
- `docs/superpowers/plans/2026-10-02-faro-fase-0.md:192-194`: R-2 exige que la retención del uid se base en todos los procesos del cgroup y que el gancho `jaula_viva` no pueda omitirse al cablear el lanzamiento.
- En `origin/master`, `jax/faro/control.py` permite `jaula_viva=None`, consulta el callback solo para uids retenidos en memoria y cierra el Puerto después de decidir si libera el uid.
- En `origin/master`, `jax/faro/jaula.py` conserva objetos `Popen` por uid y su `jaula_viva` solo comprueba el proceso principal.
- `jax/faro/servicio.py` expone `Servicio.control()` sin exigir proveedor de liveness.
- El checkout no contiene `jax/faro/scope.py` ni `tests/test_faro_scope.py`.
- El plan de instalación 0.10 corre el servicio bajo `User=faro`; el host actual aún no tiene esa cuenta ni su user manager. No se hará instalación ni despliegue en este trabajo.
- La autorización versionada de sudoers solo enumera `/usr/bin/bwrap`; no autoriza a `faro` a crear unidades en el system manager (`ops/faro/sudoers-jaula-ejemplo`).
- En el user manager activo de hall9000, probé una `.scope` y una `.service` transitorias con `MemoryMax`, `CPUQuota` y `TasksMax`. Ambos aceptaron los límites y el cgroup registró un hijo vivo. `systemctl --user kill --kill-whom=all --signal=SIGKILL` terminó el árbol y dejó `cgroup.events` vacío. La prueba fue temporal y no instaló servicios persistentes. No equivale a probar la cuenta `faro`, que aún no existe.
- Los manuales locales de systemd 259 documentan que `--pipe` conecta stdin/stdout/stderr del servicio transitorio y que `MemoryMax`, `CPUQuota` y `TasksMax` gobiernan los controladores memory, cpu y pids.

## Decisión técnica

Usar un **servicio transitorio del user manager por uid/ejecución**, con unidad `faro-jaula-u<uid>-<attempt_id>.service`, bajo el slice explícito de Faro. `ServidorControl` permite una ejecución viva por uid; un lease durable exclusivo, bajo `JAX_FARO_SCOPE_STATE_DIR`, reserva el uid antes de pedir el start y conserva la identidad del intento a través de cancelaciones y reinicios. El nombre determinista de la unidad permite consultar systemd, pero no sustituye ese lease ni demuestra por sí mismo que un start pendiente haya terminado.

No usar `systemd-run --user --scope` desde el futuro servicio de sistema `User=faro`: no se probó que el user manager pueda adoptar un PID que nació en `system.slice`, fuera de su subárbol. No autorizar a `faro` a administrar el system manager ni añadir un broker root en esta fase. La unidad `.service` mantiene el contrato de aislamiento y límites por cgroup; documentar la precisión de unidad como desviación del nombre `.scope` del plan 0.6 y someterla a auditoría Tier 3. Si la auditoría demuestra que el tipo `.scope` es requisito literal, detener esa integración y elevar la decisión antes de cambiar la autoridad.

El servicio transitorio usa `Type=exec`, `--pipe`, `--collect`, `KillMode=control-group`, `OOMPolicy=kill` y límites finitos requeridos por configuración. No hay valores por defecto para los límites: sin los tres, el lanzamiento falla cerrado. No se limita cantidad de agentes; `TasksMax` expresa el tope de procesos/hilos como recurso del sistema operativo, conforme a D-4.

**Lease y recovery R-2, contrato v3 (decisión técnica corregida, Codex, 2026-10-10; estado WIP al 2026-10-12).** El lease durable tiene esquema cerrado y etapas `PREPARED → REQUESTED → ACCEPTED → CLIENT_CLOSED`. PREPARED confirma reserva/fsync antes de cualquier llamada externa; REQUESTED confirma fsync antes de crear el cliente. ACCEPTED se persiste únicamente al observar la unidad propia con Description exacta y conserva la ruta de cgroup junto con la identidad kernel `st_dev/st_ino`. Solo desde ACCEPTED puede persistirse CLIENT_CLOSED tras el wait exitoso del cliente exacto. Un cliente que termina sin aceptación conserva REQUESTED: el receptor puede haber recibido una petición aunque no haya respondido. El namespace de unidades/jobs vacío no demuestra por sí solo una barrera causal.

Recovery se ejecuta bajo el lock de instancia antes de abrir el socket. PREPARED se libera únicamente si namespace completo está vacío; CLIENT_CLOSED requiere reconciliación de identidad y prueba de cgroup vacío. REQUESTED o ACCEPTED sin handle conservan cuarentena por UID y no impiden admitir otros UIDs seguros. Archivos corruptos o inseguros bloquean recovery; no se borran. Un cambio de manager/epoch no libera REQUESTED: clientes huérfanos pueden conectar tarde al manager nuevo. La revisión Sol adicional debe cerrar explícitamente la prueba de ausencia de cgroups supervivientes tras un reinicio sin serialización del manager antes de publicar la implementación.

El release exige cierre causal del emisor, evidencia kernel del mismo inode y namespace vacío; una época nueva del
manager nunca la sustituye. La unidad productiva debe fijar `Delegate=no` para que la jaula no migre su cgroup. 0.10
debe instalar `/var/lib/faro/jaulas` persistente, propiedad de `faro:faro` y modo `0700`, configurar
`JAX_FARO_SCOPE_STATE_DIR`, mantener una sola instancia y probar la barrera del supervisor sobre el cgroup completo y
sus clientes. El directorio no puede estar bajo `RuntimeDirectory`. No se crea una API manual/booleana para liberar
leases inciertos.

## Contrato de estado

El estado consultado por uid es trivalente:

- `LIVE`: unidad activa y cgroup poblado.
- `EMPTY`: systemd confirma que no existe unidad activa y el cgroup ya está vacío/eliminado.
- `UNKNOWN`: manager, unidad, ControlGroup, cgroupfs o proceso no permiten concluir.

Solo `EMPTY` libera. `LIVE` y `UNKNOWN` conservan/rechazan el uid. Consultar systemd por la unidad determinista incluso cuando el uid no aparece en los mapas en memoria; así un reinicio del proceso Faro no autoriza reutilizar un uid con cgroup superviviente. La consulta sin lease cubre el namespace completo por UID, incluidos nombres legacy y por intento; una unidad/job huérfana bloquea admisión, sin adoptarla ni matarla.

Orden de cierre:

1. Marcar ejecución cerrándose y conservar uid.
2. Cerrar/revocar el Puerto de esa ejecución.
3. Enviar SIGKILL a todos los procesos de su unidad, sin afectar otras unidades.
4. Esperar confirmación de unidad/cgroup vacío; error o timeout deja uid retenido.
5. Quitar la referencia en memoria y permitir su reutilización solo tras nueva consulta `EMPTY`.

`Servicio.control(scope_manager)` y el constructor de `LanzadorJaula` exigirán el mismo proveedor de unidad. `ServidorControl` no aceptará callback opcional. Las pruebas usarán un doble explícito que falle si se omite o si una consulta resulta desconocida.

## Archivos previstos

- `jax/faro/scope.py`: límites validados; lease durable y recovery antes de admisión; nombre cerrado de unidad; start/wait/kill/query asíncronos sin bloquear el event loop; lectura de `ControlGroup` desde systemd, nunca deducida por concatenación; inspección de `cgroup.events`; estados `LIVE/EMPTY/UNKNOWN`.
- `jax/faro/scope_lease.py`: lease versionado por uid; directorio y archivos con permisos/propiedad exactos; creación exclusiva durable; lectura cerrada ante symlinks, entradas inesperadas o corrupción; liberación por intento exacto.
- `jax/faro/jaula.py`: sustituir liveness basado en `Popen.returncode` por el proveedor de cgroup; ejecutar bwrap dentro de la unidad con `--pipe`.
- `jax/faro/control.py`: callback de liveness obligatorio; revisar cada uid candidato contra systemd, aun sin reserva local; cierre que conserva uid hasta Puerto revocado y cgroup vacío.
- `jax/faro/servicio.py`: `control(scope_manager)` no construye control sin la fuente de liveness.
- `tests/test_faro_scope.py`: propiedades y argumentos cerrados; cgroup y unidad faltantes/desconocidos; árbol con nieto/setsid; límites efectivos; muerte del scope sin afectar hermanos; lease durable/corrupto/colisión; recuperación de unidad propia/ausente/ajena; serialización de recovery; cancelación y fallo de manager.
- `tests/test_faro_control.py`, `tests/test_faro_lanzador.py` y golden de bwrap: adaptar a proveedor explícito; agregar regresiones de reinicio y cierre.
- `docs/superpowers/plans/2026-10-02-faro-fase-0.md`: añadir la desviación de unidad `.service` y la precondición del user manager de `faro` para 0.10.
- `TRASPASO.md`: estado, decisiones y pruebas por commit.

## Secuencia TDD

1. Escribir pruebas de `ScopeManager` para configuración incompleta, argumentos de systemd exactos, liveness triestado y salida fail-closed ante manager/cgroup no disponible.
2. Implementar el gestor hasta que pasen esas pruebas.
3. Escribir prueba real opcional del user manager que arranca bwrap/sleep en unidad transitoria, verifica límites y mata el cgroup completo; omitir solo si el runner CI carece de user manager y registrar esa limitación. La corrida local de hoy ya verificó la mecánica base de systemd 259, no reemplaza la prueba del código.
4. Probar carrera de dos creaciones con el mismo uid: una sola unidad puede arrancar. Un fallo/cancelación de arranque no marca el uid libre hasta una consulta `EMPTY` posterior.
5. Probar cierre tras `control_cerrado`: Puerto revocado antes de matar la unidad; solo la unidad de ese uid muere; `UNKNOWN` conserva uid.
6. Adaptar suite Faro y ejecutar pruebas enfocadas. No reducir pisos existentes; medir/actualizar cualquier piso afectado en la misma rama.
7. Actualizar Biblioteca y el handoff, ejecutar `git diff --check` y verificación i18n/temas/rendimiento aplicable al cambio backend.
8. Pedir auditoría Tier 3 de solo lectura sobre el SHA exacto. No integrar ni desplegar hasta CI y auditoría exacta aprobadas.

## Fuera de alcance y dependencia

- No implementar proxy S-1 en este cambio R-2. La decisión técnica de red 0.5 ya fue cerrada por el coordinador con arquitectura Sol: loopback dentro de `--unshare-net` y relé a proxy UDS por ejecución; su implementación/pruebas son el siguiente cambio.
- No instalar cuenta/user manager, `linger`, slice ni unidades persistentes de `faro`; corresponden al paso operativo 0.10 y requieren comprobación del host de destino.
- No habilitar `agentes.lanzar`, conectores de motores, límites de producción ni despliegue. 0.7/0.8 deben reutilizar obligatoriamente este gestor y el callback.
- La carga final de 0.9 sigue siendo requisito para fijar valores de límites y cualquier GO de producción.

## Criterio de aceptación

El uid permanece ocupado mientras haya cualquier proceso en su unidad/cgroup o el estado sea desconocido; un reinicio de Faro no lo libera por perder memoria; el socket no abre hasta completar recovery durable; el freno mata todos los descendientes de una ejecución sin afectar al servicio ni a otra ejecución; los tres límites se aplican en una unidad verificada antes de darla por lista; ninguna ruta productiva puede construir el canal de control o lanzar una jaula sin proveedor de liveness; CI y auditoría Tier 3 aprueban el SHA exacto.
