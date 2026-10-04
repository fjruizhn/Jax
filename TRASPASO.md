# Traspaso — Faro 0.4 `memoria.buscar`

Actualizado: 2026-10-04 · rama `feat/faro-0.4-memoria-buscar`

## Estado

- PR JAX #352, base `master` en `142c91f13b200d41e38c214a46c266e56749c7eb`.
- Último commit implementador: `223291f` más cambios locales sin commit para hacer rollback acotado, descartar conexiones colgadas y validar síntesis por existencia de payload sin seleccionar el BLOB.
- Sol re-auditó `223291f`: 0 BLOCK, 2 MAJOR abiertos (rollback colgado durante cancelación y SELECT sin cota del BLOB de fuentes de síntesis). Ambos tienen regresiones locales nuevas; la reauditoría del SHA corregido queda pendiente.
- CI de GitHub debe ejecutarse sobre el nuevo SHA; la integración MariaDB usa el job aislado. Sin conexión ni cableado a B9 de producción.

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
2. Confirmar `git status` y commit de los cambios locales descritos; medir pisos; ejecutar suite/políticas; actualizar el PR y pedir auditoría Sol del SHA exacto.
3. No integrar ni desplegar.
