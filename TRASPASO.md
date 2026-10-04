# Traspaso — Faro 0.4 `memoria.buscar`

Actualizado: 2026-10-04 · rama `feat/faro-0.4-memoria-buscar`

## Estado

- PR JAX #352, base `master` en `142c91f13b200d41e38c214a46c266e56749c7eb`.
- Último commit implementador: endurece el retorno no confiable, pone timeout cancelable, limita payload por SQL y valida el único perfil local de prueba al abrir conexión.
- La auditoría Sol del SHA previo `4f7339c` fue RECHAZADA (1 BLOCK, 3 MAJOR); los cuatro hallazgos tienen regresiones y la reauditoría del head actual está pedida.
- CI del head de código en curso; la prueba MariaDB usa el job aislado. Sin conexión ni cableado a B9 de producción.

## Decisiones que no deben cambiar

- B9 no está reactivada en producción; Jax#276 sigue bloqueado. No habilitar conexión productiva sin reactivación y contrato (Principio IX).
- Perfil permitido solo: `jax_test@127.0.0.1:3308/jax_memory_test`; memoria apagada por defecto.
- La identidad del tenant/usuario viene del socket. Todo texto de memoria se trata como no confiable.

## Verificación local más reciente

- `faro-fase0`: 688 passed, 0 skipped.
- Políticas, inventario y pisos: 210 passed.
- Pruebas del helper de conexión: 2 passed. La lectura MariaDB local no autenticó desde este origen de red; no se cambió el perfil para sortearlo.
- Pisos: Faro 688; mínimo MariaDB 105 casos.

## Para quien retome

1. Parte de `git rev-parse HEAD` y este archivo; no reconstruyas el estado desde memoria.
2. Consulta checks y auditoría de #352 para el SHA exacto. Si hay hallazgos, corrígelos y vuelve a auditar el SHA nuevo.
3. No integrar ni desplegar.
