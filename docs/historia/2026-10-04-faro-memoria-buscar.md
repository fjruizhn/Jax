# El Faro 0.4 — `memoria.buscar`

Historia técnica · Codex · 2026-10-04 (America/Tegucigalpa). Encargo de Fernando por coordinación jax-fe. Fuente de código: rama `feat/faro-0.4-memoria-buscar`, basada en JAX `origin/master` `142c91f13b200d41e38c214a46c266e56749c7eb`.

## Decisión y límite

Fernando confirmó que B9 **no está en producción** y que la reactivación sigue pendiente; Jax#276 depende de ello. Este paso implementa un adaptador de lectura para el Puerto y prueba el reader contra la base de prueba `jax_memory_test` en `127.0.0.1:3308` con `jax_test`. No hay conexión, fallback ni configuración de B9 productiva. La configuración de memoria del servicio por defecto está apagada; si se habilita, solo acepta ese endpoint de prueba. El cable de producción espera que B9 se reactive y reciba su contrato conforme al Principio IX.

## Implementación

- `memoria.buscar` se registra en el servidor MCP por conexión. Pasa por la Guardia existente, que consulta el freno y registra identidad, argumentos saneados y hash del resultado; fallo de bitácora sigue fallando cerrado.
- El esquema MCP acepta `consulta` y `limite`; tenant y usuario se derivan exclusivamente de `Identidad` del socket. Sin identidad devuelve vacío. El B9 reader recibe un `ScopeContext` de usuario/tenant sin project scope.
- El adaptador consulta hasta las 100 revisiones recientes del scope y filtra texto en `payload`; el contenido devuelto se envuelve con `<untrusted_source>` y sus sentinelas de cierre/tokens de control se neutralizan con el patrón endurecido de LAS MANOS. Es búsqueda lexical acotada para la fase 0.4, no autoridad, evidencia de verdad actual ni un índice global.
- La lectura tiene un plazo configurable `JAX_FARO_MEMORIA_TIMEOUT_S` (default 5 s); al vencer cancela el reader y responde `memoria no disponible`. El reader recibe `LEFT(payload, 8192)` desde MariaDB, informa truncamiento explícito y el adaptador impone el mismo tope antes de envolver. La configuración y el propio helper de conexión revalidan el perfil exacto `jax_test@127.0.0.1:3308/jax_memory_test`.
- Si el adaptador no está conectado o el reader falla, el Puerto devuelve `memoria no disponible`. Las consultas vacías, sobredimensionadas y límites inválidos se rechazan.
- El pool asíncrono es único por servicio y reutilizado. Solo el `MariaDBB9Reader` existente ejecuta su transacción de lectura repetible; no hay mutaciones B9 en el adaptador.

## Verificación

- TDD de la corrección: contra el árbol previo fallaron las nuevas pruebas de neutralización de sentinelas, plazo/cancelación, tamaño del payload, construcción directa de configuración de producción y revalidación del helper antes de abrir conexión; tras la implementación pasaron.
- Auditoría Tier 3 inicial del SHA `4f7339cedca787071f07995e9050d4ceee022db7`: RECHAZADO (1 BLOCK, 3 MAJOR): envoltura hostil incompleta, reader sin timeout, payload sin cota en SQL y allowlist no impuesta en el helper. Las regresiones ahora cubren esos cuatro puntos.
- Reauditoría Tier 3 de `223291faa0888efd7036cf093d0568f3f02a382f`: RECHAZADO (0 BLOCK, 2 MAJOR): cancelación podía quedar esperando un rollback colgado; la validación de fuente de una síntesis seleccionaba el LONGBLOB entero. La corrección acota rollback con `asyncio.wait`, cierra la conexión al vencer y selecciona solo `(p.payload IS NOT NULL)`; pruebas cubren el rollback colgado y fuente presente/ausente sin leer el BLOB.
- La regresión de arranque fail-soft detectó que el `except` opcional no estaba marcado: `policy/tests/test_no_fail_open_except.py`, 21 passed tras anotarlo.
- Suite exacta `faro-fase0`: 688 passed, 0 skipped (Python 3.14.4 local), incluyendo 4 lectores concurrentes con 100 payloads cada uno en el tope por fila. La medición local MariaDB final falló antes de ejecutar el reader porque la cuenta `jax_test` no fue aceptada desde este entorno de red; CI usa un contenedor MariaDB efímero y ejecuta la prueba DB.
- La prueba DB del reader usa tablas `TEMPORARY`, comprueba scope, solo SELECT/SET TRANSACTION, la cota `LEFT(payload, 8192)` y la validación de fuente de síntesis por existencia. Las pruebas unitarias cubren rollback colgado, ausencia de BLOB en validación de síntesis y revalidación del perfil de conexión local.
- Pisos medidos/ajustados: `faro-fase0/faro` 683→688 y mínimo `memory-b9-regression/casos` 103→107, con pruebas para rollback limitado y síntesis que confirma existencia sin leer BLOB; 2 casos previos ya cubrían el límite de perfil del pool.

## Pendientes

Esperar CI remota y nueva revisión de escalón 3 del SHA exacto del PR. No desplegar ni conectar a B9 de producción desde este cambio. El cable productivo requiere reactivación de B9 y un contrato aprobado.
