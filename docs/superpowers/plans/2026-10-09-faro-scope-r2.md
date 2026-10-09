# El Faro 0.6 y R-2: unidad cgroup por ejecución

Fecha: 2026-10-09  
Estado: plan de implementación  
Rama: `codex/faro-r2`  
Base: `origin/master@709237f901e8ec8840f6b2b39a1a2ce88682674a`

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

Usar un **servicio transitorio del user manager por uid/ejecución**, con unidad `faro-jaula-u<uid>.service`, bajo el slice explícito de Faro. Un uid solo puede tener una ejecución viva según `ServidorControl`; la unidad determinista por uid también hace que systemd arbitre atómicamente dos intentos concurrentes y permite recuperar el estado tras reinicios.

No usar `systemd-run --user --scope` desde el futuro servicio de sistema `User=faro`: no se probó que el user manager pueda adoptar un PID que nació en `system.slice`, fuera de su subárbol. No autorizar a `faro` a administrar el system manager ni añadir un broker root en esta fase. La unidad `.service` mantiene el contrato de aislamiento y límites por cgroup; documentar la precisión de unidad como desviación del nombre `.scope` del plan 0.6 y someterla a auditoría Tier 3. Si la auditoría demuestra que el tipo `.scope` es requisito literal, detener esa integración y elevar la decisión antes de cambiar la autoridad.

El servicio transitorio usa `Type=exec`, `--pipe`, `--collect`, `KillMode=control-group`, `OOMPolicy=kill` y límites finitos requeridos por configuración. No hay valores por defecto para los límites: sin los tres, el lanzamiento falla cerrado. No se limita cantidad de agentes; `TasksMax` expresa el tope de procesos/hilos como recurso del sistema operativo, conforme a D-4.

## Contrato de estado

El estado consultado por uid es trivalente:

- `LIVE`: unidad activa y cgroup poblado.
- `EMPTY`: systemd confirma que no existe unidad activa y el cgroup ya está vacío/eliminado.
- `UNKNOWN`: manager, unidad, ControlGroup, cgroupfs o proceso no permiten concluir.

Solo `EMPTY` libera. `LIVE` y `UNKNOWN` conservan/rechazan el uid. Consultar systemd por la unidad determinista incluso cuando el uid no aparece en los mapas en memoria; así un reinicio del proceso Faro no autoriza reutilizar un uid con cgroup superviviente.

Orden de cierre:

1. Marcar ejecución cerrándose y conservar uid.
2. Cerrar/revocar el Puerto de esa ejecución.
3. Enviar SIGKILL a todos los procesos de su unidad, sin afectar otras unidades.
4. Esperar confirmación de unidad/cgroup vacío; error o timeout deja uid retenido.
5. Quitar la referencia en memoria y permitir su reutilización solo tras nueva consulta `EMPTY`.

`Servicio.control(scope_manager)` y el constructor de `LanzadorJaula` exigirán el mismo proveedor de unidad. `ServidorControl` no aceptará callback opcional. Las pruebas usarán un doble explícito que falle si se omite o si una consulta resulta desconocida.

## Archivos previstos

- `jax/faro/scope.py`: límites validados; nombre cerrado de unidad; start/wait/kill/query asíncronos sin bloquear el event loop; lectura de `ControlGroup` desde systemd, nunca deducida por concatenación; inspección de `cgroup.events`; estados `LIVE/EMPTY/UNKNOWN`.
- `jax/faro/jaula.py`: sustituir liveness basado en `Popen.returncode` por el proveedor de cgroup; ejecutar bwrap dentro de la unidad con `--pipe`.
- `jax/faro/control.py`: callback de liveness obligatorio; revisar cada uid candidato contra systemd, aun sin reserva local; cierre que conserva uid hasta Puerto revocado y cgroup vacío.
- `jax/faro/servicio.py`: `control(scope_manager)` no construye control sin la fuente de liveness.
- `tests/test_faro_scope.py`: propiedades y argumentos cerrados; cgroup y unidad faltantes/desconocidos; árbol con nieto/setsid; límites efectivos; muerte del scope sin afectar hermanos; colisiones/concurrencia por uid; cancellation y fallo de manager.
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

- No elegir la arquitectura de red de 0.5 ni implementar proxy S-1; esa decisión sigue pendiente de Fernando.
- No instalar cuenta/user manager, `linger`, slice ni unidades persistentes de `faro`; corresponden al paso operativo 0.10 y requieren comprobación del host de destino.
- No habilitar `agentes.lanzar`, conectores de motores, límites de producción ni despliegue. 0.7/0.8 deben reutilizar obligatoriamente este gestor y el callback.
- La carga final de 0.9 sigue siendo requisito para fijar valores de límites y cualquier GO de producción.

## Criterio de aceptación

El uid permanece ocupado mientras haya cualquier proceso en su unidad/cgroup o el estado sea desconocido; un reinicio de Faro no lo libera por perder memoria; el freno mata todos los descendientes de una ejecución sin afectar al servicio ni a otra ejecución; los tres límites se aplican en una unidad verificada antes de darla por lista; ninguna ruta productiva puede construir el canal de control o lanzar una jaula sin proveedor de liveness; CI y auditoría Tier 3 aprueban el SHA exacto.
