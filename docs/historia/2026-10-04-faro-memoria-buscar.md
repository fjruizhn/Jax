# El Faro 0.4 — `memoria.buscar`

Historia técnica · Codex · 2026-10-04 (America/Tegucigalpa). Encargo de Fernando por coordinación jax-fe. Fuente de código: rama `feat/faro-0.4-memoria-buscar`, basada en JAX `origin/master` `142c91f13b200d41e38c214a46c266e56749c7eb`.

## Decisión y límite

Fernando confirmó que B9 **no está en producción** y que la reactivación sigue pendiente; Jax#276 depende de ello. Este paso implementa un adaptador de lectura para el Puerto y prueba el reader contra la base de prueba `jax_memory_test` en `127.0.0.1:3308` con `jax_test`. No hay conexión, fallback ni configuración de B9 productiva. La configuración de memoria del servicio por defecto está apagada; si se habilita, solo acepta ese endpoint de prueba. El cable de producción espera que B9 se reactive y reciba su contrato conforme al Principio IX.

## Implementación

- `memoria.buscar` se registra en el servidor MCP por conexión. Pasa por la Guardia existente, que consulta el freno y registra identidad, argumentos saneados y hash del resultado; fallo de bitácora sigue fallando cerrado.
- El esquema MCP acepta `consulta` y `limite`; tenant y usuario se derivan exclusivamente de `Identidad` del socket. Sin identidad devuelve vacío. El B9 reader recibe un `ScopeContext` de usuario/tenant sin project scope.
- El adaptador consulta hasta las 100 revisiones recientes del scope y filtra texto en `payload`; cada resultado devuelve ID de memoria/revisión, contenido y `trust=untrusted_source`. Es búsqueda lexical acotada para la fase 0.4, no autoridad, evidencia de verdad actual ni un índice global.
- Si el adaptador no está conectado o el reader falla, el Puerto devuelve `memoria no disponible`. Las consultas vacías, sobredimensionadas y límites inválidos se rechazan.
- El pool asíncrono es único por servicio y reutilizado. Solo el `MariaDBB9Reader` existente ejecuta su transacción de lectura repetible; no hay mutaciones B9 en el adaptador.

## Verificación

- TDD: la prueba nueva primero falló al colectarse porque `ConfigMemoria` y el adaptador aún no existían.
- Mutaciones verificadas en rojo: scope cambiado de la identidad, eliminación de `trust=untrusted_source`, y memoria activada por defecto.
- Suite exacta `faro-fase0`: 683 passed, 0 skipped (Python 3.14.4 local), incluyendo 11 pruebas nuevas del adaptador/MCP y una prueba de servicio apagado por defecto.
- Reader B9 en MariaDB real de prueba: smoke sin escrituras persistentes, 0 resultados para tenant aleatorio, `jax_memory_test@127.0.0.1:3308`, usuario `jax_test`.
- Prueba DB wireada al job aislado `memory-b9-regression`: usa tablas `TEMPORARY`, lee scope aleatorio y verifica que el reader solo ejecuta `SELECT` / configuración de transacción. 1 passed local.
- CI/policy: prueba de inventario de tests, comparador y migración de pisos: 189 passed. Pisos medidos: `faro-fase0/faro` 671→683 y `memory-b9-regression/casos` 102→103.

## Pendientes

Esperar CI remota y revisión de escalón 3 del SHA exacto del PR. No desplegar ni conectar a B9 de producción desde este cambio. El cable productivo requiere reactivación de B9 y un contrato aprobado.
