# El Faro: observación de preparación productiva

Fecha: 2026-10-10, hall9000. Autor: Codex, coordinador. Este archivo es un
snapshot de observaciones y decisiones técnicas; no es autoridad normativa ni
evidencia de un despliegue. Reconsultar las fuentes antes de operar.

## Verdad operacional observada

- Fuente remota: `origin/master@4e1a35a7db6861912d3f937b67468d2dee2b324e`,
  obtenido mediante fetch. El owner del cambio de checkpoint conserva su guard;
  su corrida push `38105335278` terminó `success`, revalidada en GitHub. El pane del owner informó guard SUCCESS sobre ese mismo tip; el recibo no pertenece a esta sesión.
- Fuente runtime: como `jaxsvc`, con la configuración productiva y el Python de
  `/srv/jax-prod/jax/las_manos/.venv`,
  `python3 -m jaxctl authority --json` devolvió código 2,
  `classification=UNAVAILABLE`, `status=UNAVAILABLE` y
  `detail=Block 4 authority source unavailable`.
- Fuente filesystem: no existen `/etc/jax/authority` ni
  `/var/lib/jax/authority`; por tanto faltan los archivos externos previstos en
  `trusted-files.md`. No se encontró todavía una fuente aprobada de restauración.
- Diagnóstico de descubrimiento independiente, pendiente de revalidar antes de
  modificar permisos: `jaxctl` consulta primero el ledger MariaDB; el usuario
  efectivo `jax_user@172.30.5.%` no tiene acceso a `jax_authority` y recibió
  `OperationalError 1142` al consultar `authority_ledger_genesis`.
- Fuente de código en el SHA remoto citado: 0.4 está integrada; no existen todavía
  `jax/faro/proxy.py`, `scope.py`, `agentes.py`, `conectores/` ni
  `scripts/load_test_faro.py`. El entrypoint de Faro verifica y cierra el servicio;
  no sirve un bucle productivo. No hay una unidad Faro instalada.
- R-2/0.6 permanece local en `codex/faro-r2`. La última auditoría del diff fue
  RECHAZADO, 4 BLOCK y 4 MAJOR. Pasar sus suites existentes no cerraba las
  ventanas de cancelación y recuperación detectadas.

## Decisiones técnicas del coordinador

- Para recuperación R-2, un cambio de época de systemd no prueba que haya muerto
  un cliente sobreviviente capaz de enviar un start. PREPARED implica que no se
  entró en el syscall; REQUESTED sin aceptación conserva cuarentena por UID,
  incluso si su cliente terminó. ACCEPTED exige observar la unidad propia y su
  Description exacta; solo entonces el wait exitoso del cliente permite
  CLIENT_CLOSED. ACCEPTED sin handle también conserva cuarentena. El cierre exige
  reconciliar unidad, jobs y cgroup. Se admite a otros UIDs seguros. La futura
  recuperación de REQUESTED requiere una barrera del supervisor realmente
  comprobada en 0.10, nunca un booleano de autorización manual.
- Se acepta la propuesta de arquitectura Sol para la desviación 14 como diseño
  técnico de 0.5: conservar `bwrap --unshare-net`, usar HTTP loopback dentro de la
  jaula y un relé a un socket Unix de capacidad por ejecución. Credenciales reales
  fuera de la jaula, proxy ligado a proveedor/rutas/modelo, autenticación exacta
  por `SO_PEERCRED` y token, cabeceras de entrada y salida permitidas explícitamente,
  sin cookies ni rutas de escape. Es diseño pendiente de implementación y de
  pruebas con binarios reales contra upstreams falsos; no se habilita una capacidad
  externa por documentarlo.

## Dependencias para cerrar producción

1. Terminar R-2, reapilar sobre el tip oficial después de sus guardas, medir el
   piso final, obtener auditoría independiente del SHA, CI exacta e integración.
2. Preparar el acceso mínimo de una cuenta dedicada al ledger autoritativo y
   obtener la fuente verificada de raíz, checkpoints y recibo de bootstrap. El
   runbook `authority-root-recovery.md` prohíbe sintetizar o sobrescribir raíces.
   Falta saber si existe material aprobado o si se trata de primera inicialización.
3. Implementar proxy 0.5 y demostrar S-1 sin gasto con upstreams falsos y los
   artefactos exactos de los motores; después cablear 0.7 y 0.8.
4. Provisionar las credenciales externas aprobadas y el login propio de Faro que
   exija interacción humana; no usar la identidad personal ni firmar por Fernando.
   Las llamadas cubiertas por reglas requieren ratificación Ed25519 del contenido
   exacto, no solo un GO operativo o un merge.
5. Ejecutar carga 0.9, registrar p95, throughput, RSS y degradación de peor caso,
   probar reinicio/crash/barreras del supervisor y preparar paquete y unidad 0.10.
6. Desplegar con la autorización ya dada, respaldo con restauración comprobada,
   verificación runtime y prueba real del freno. Hasta entonces Faro no está listo
   en producción.

Estas dependencias se ejecutan dentro del mandato de Fernando. La falta de
material de identidad o de una firma verificable no se resuelve inventándolos.
