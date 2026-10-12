# El Faro: estado de preparación productiva

Fecha de observación: 2026-10-12, hall9000. Autor: Codex, coordinación R-2.
Este documento registra observaciones fechadas; no es autoridad normativa, recibo de integración ni evidencia de un
despliegue. Reconsultar las fuentes vivas antes de operar.

## Estado observado

- La producción viva estaba en `e9895512`. La fuente oficial observada de desarrollo era
  `origin/master@4e1a35a7db6861912d3f937b67468d2dee2b324e`.
- Jax#402 fue reapilado y publicado por su owner en `aeb3ab10f3c6ffb9070307526f351da8e700b9cf`. Los 33/33 checks y
  la auditoría Tier 3 APROBADO correspondían al head anterior `669558b4`; la auditoría nueva y CI exacta estaban
  pendientes sobre el head reapilado antes de su preflight y guard. Ese estado no acredita la integración de R-2.
- La selección CI extendida local de R-2 en Python 3.14 obtuvo `904 passed, 6 deselected` en 60.55 s; log:
  `/tmp/faro-r2-integrated-py314-20261012-final.log`. La jaula real obtuvo `12 passed` y las seis integraciones
  reales de systemd `6 passed, 158 deselected`, con logs fechados equivalentes en `/tmp`. Sigue sin ser una corrida
  final: faltaban rebase, medición del piso, auditoría del SHA exacto, CI final, merge y post-merge guard.
- R-2 aplica el contrato v3 de cuatro estados (`PREPARED`, `REQUESTED`, `ACCEPTED`, `CLIENT_CLOSED`). La liberación
  exige cierre del emisor, namespace vacío y evidencia kernel del mismo cgroup (`ruta`, `st_dev`, `st_ino`); un cambio
  de epoch del manager no libera un lease. La unidad debe mantener `Delegate=no`.
- La auditoría Tier 3 previa tenía 1 BLOCK y 2 MAJOR. Este WIP incorpora sus correcciones, pero la auditoría nueva
  del SHA exacto continúa pendiente.
- `jaxctl authority --json` seguía en `UNAVAILABLE`; el diagnóstico del ledger registró `OperationalError 1142`.
  No estaban presentes la raíz ni checkpoints externos esperados. No hay fuente autorizada para restaurarlos.
- `jaxqwen` estaba instalado, con servicio `disabled/inactive` y `dispatch_enabled=false`. Las Voces que usen esa
  ruta no requieren terminar cada fase de Faro, aunque el ejecutor seguro por nombre aún no existe.

## Condiciones pendientes para producción Faro

1. Cerrar R-2 con auditoría del SHA exacto, CI, integración y guardas tras la base oficial final.
2. Recuperar o inicializar desde una fuente autorizada la raíz, checkpoints y acceso mínimo de Block 4.
3. Implementar y comprobar 0.5; continuar con 0.7, 0.8, carga 0.9 y operación 0.10.
4. Verificar en runtime el user manager, estado durable, barrera del supervisor, freno de cgroup y carga medida antes
   de declarar el servicio apto para producción.
