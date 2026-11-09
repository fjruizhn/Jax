# OP#225 · índice textual de `messages` · 2026-11-09

**Tipo:** HISTORIA de implementación y ensayo. No prueba por sí sola el estado actual de producción.

## Qué se hizo

Se preparó en JAX la migración B9 014 para añadir `FULLTEXT ft_messages_content(content)` a `messages`. Se actualizó el esquema versionado, la provisión de bases de prueba, el preflight y la marcha atrás de los runbooks. La migración fija `lock_wait_timeout=10`, `ALGORITHM=COPY` y `LOCK=SHARED`, y se aplica de forma idempotente. El servicio axioma-memoria solo leerá el índice tras su instalación; no recibe DDL sobre tablas JAX.

## Por qué

En diez preguntas naturales sobre mensajes de la base, el ranking vigente de axioma-memoria dejó los diez objetivos fuera del top 5. Los documentos pueden sumar votos vectoriales y textuales; los mensajes solo tenían la señal vectorial. Un índice textual permite sumar la segunda señal sin imponer cuotas por tipo que degradaron el corpus documental en la prueba (Recall@5 0,95 → 0,90 con reparto igualitario).

## Evidencia y límite

- Código: JAX PR #416, rama `codex/op225-messages-ft-20261109`.
- MariaDB 12.3.3 desechable: la prueba real de 014 se aplicó dos veces, `EXPLAIN` eligió `ft_messages_content`, `MATCH` devolvió el mensaje esperado y la marcha atrás retiró el índice. La suite focal completa dio 50 passed. La provisión fresca informó `PROVISION_OK` y el driver B9 completó 19 casos PASS.
- CI de #416 en el primer SHA: `memory-b9-regression` pasó con MariaDB 12.3.3 real y sin skips. Hay un SHA posterior a esta nota; se repite el control antes de integrar.
- Se copiaron temporalmente 1.677 mensajes de producción a un contenedor local de prueba para medir ranking textual, sin alterar la base real. Esta copia no es un respaldo y se elimina al cerrar el ensayo.
- **No se ha aplicado aún la migración a producción.** La duración de la copia y el costo de inserción en producción siguen sin medirse.

## Lecciones y alternativas

- Un `EXPLAIN` verde no basta: la prueba también ejecuta `SELECT MATCH` y comprueba el objetivo.
- La búsqueda semántica global devolvía mensajes completos, pero dejaba consultas parciales sobre mensajes largos en posiciones aproximadamente 38–67. No sustituye FULLTEXT.
- Dos etapas de RRF subieron hechos, pero bajaron las métricas documentales. Se eligió conservar el RRF plano y sumar señales textuales por fuente.
- JAX PR #349 modifica `messages` en una rama antigua. No hay choque semántico con FULLTEXT; al rebasar ese PR se debe conservar el índice 014 en `jax_memory_schema.sql`.

## Pendientes

1. Auditoría Sol y CI del SHA final de #416; integrar por PR según autoridad vigente.
2. Respaldo de `jax_memory` con restauración probada; medir la DDL en una copia del volumen actual y aplicar 014 en ventana tranquila.
3. Verificar índice `FULLTEXT(content)` con `information_schema` y `EXPLAIN` real; medir escritura y consulta.
4. Activar después el lector SELECT en axioma-memoria, medir ranking de mensajes y carga 1/10/50.

**Decisión técnica:** Sol recomendó índice en JAX y RRF plano; Codex adoptó el plan. **Encargo y autoridad final:** Fernando Ruiz, OP#225 comentario 1561 y “dale”.
