# Traspaso — Faro 0.4 `memoria.buscar`

Actualizado: 2026-10-05 · rama `feat/faro-0.4-memoria-buscar`

## Estado

- PR JAX #352, reconciliado con `origin/master` `d136cce5d88b3420a5066b40e7786c158118f29e` mediante merge regular.
- Se conserva la limpieza #353 (`docker rm -fv`) llegada desde master.
- Raíz del rojo `memory-b9-regression` de `aae11a8`: `CursorObservado` del test MariaDB no exponía `fetchone`, aunque el reader lo usa legítimamente para validar la fuente de síntesis. Se añadió delegación y regresión directa; el mínimo sube de 107 a 108.
- Verificación focal: 5 passed. El perfil DB de pruebas no está expuesto en este host y Docker no está disponible a este usuario; el comando exacto de 11 módulos debe ejecutarse por el job aislado de CI, sin alterar host/base/usuario ni conectar B9 de producción.

## Decisiones que no deben cambiar

- B9 no está reactivada en producción; Jax#276 sigue bloqueado. No habilitar conexión productiva sin reactivación y contrato (Principio IX).
- Perfil permitido solo: `jax_test@127.0.0.1:3308/jax_memory_test`; memoria apagada por defecto.
- La identidad del tenant/usuario viene del socket. Todo texto de memoria se trata como no confiable.

## Verificación local más reciente

- `faro-fase0`: 688 passed, 0 skipped.
- `policy/tests`: 343 passed.
- Faro: 688 passed, 0 skipped. Regresiones unitarias nuevas de rollback, síntesis y perfil de pool: 4 passed.
- El test de MariaDB aislada no puede correr localmente: este origen de red no autenticó a `jax_test`; no se cambió el perfil para sortearlo.
- Pisos: Faro 688; mínimo MariaDB 107 casos (recontar contra main al integrar).

## Para quien retome

1. Parte de `git rev-parse HEAD` y este archivo; no reconstruyas el estado desde memoria.
2. Confirmar `git status`, ejecutar la suite exacta `memory-b9-regression` en CI aislada, medir el resultado y subir el piso si supera 108; después archivar este traspaso en `docs/historia/`, borrar `TRASPASO.md`, actualizar PR y pedir auditoría Sol del SHA exacto.
3. No integrar ni desplegar.
