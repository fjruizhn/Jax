# F2-E structured projection — reconciliation handoff

**Fecha:** 2026-10-05
**Tipo:** HISTORIA
**Fuente:** rama `phase2/f2e-structured-projection-jax-20261004`, commits `583f2b6e5e8aad363f02e79342b945bf5e982380` y `cbca8e8670bc974367cbc0be4f1e6fc05e909c47`.

## Objetivo

Mantener la proyección JSON estructurada, aditiva e inmutable de F2-C (`f2-c.structured-projection.1`) y el enlace F2-D de bytes canónicos exactos (`f2-d.structured-bytes.1`). La primera superficie DTO es `jax-platform GET /api/pipelines`: el alcance procede de `AuthUser` y cada claim `PIPELINE_STATUS` se comprueba contra dueño y estado canónicos de Jacobs bajo ese mismo alcance. `/jacobs/pipeline` queda excluido hasta que exista un contrato de identidad delegada firmado.

## Decisiones y límites

- Fernando autorizó la proyección estructurada inmutable, slots de claim del servidor y el enlace F2-D de bytes exactos. Los demás campos del DTO siguen siendo datos tipados no confiables con procedencia por campo; SR-03 sigue activo; versiones API antiguas o desconocidas se rechazan.
- No se tocó producción, Faro, F2-F ni F2-G.
- La cadena que debe conservarse es: productor autenticado → F2-B → F2-C → bytes canónicos F2-D.

## Reconciliación

Se integró `origin/master` `d136cce5d88b3420a5066b40e7786c158118f29e` mediante el merge `583f2b6e5e8aad363f02e79342b945bf5e982380`, sin conflictos. El cambio entrante corrige la limpieza de volúmenes Docker, incorpora su regresión en las dos listas de CI y sube únicamente su piso compartido de 3470 a 3473. Se conservó el paso adicional de F2-E al final de la secuencia congelada del job de gobernanza para que la comprobación de prefijo de `master` se mantenga.

## Evidencia local

En el SHA de reconciliación:

- `tests/test_f2e_structured_projection.py`, `tests/test_output_lifecycle.py` y `tests/test_governed_renderer.py`: **137 passed**.
- Piso exacto F2-E: `tests/test_f2e_structured_projection.py`: **8 passed**.
- Pruebas de cableado y migración de pisos: **234 passed**.
- Regresión entrante de limpieza Docker: **3 passed**.

El módulo `coverage` no estaba instalado localmente; no se produjo porcentaje de cobertura. La evidencia de cobertura pendiente debe obtenerse del entorno CI que instala sus dependencias declaradas.

## Pendiente de continuación

Publicar la rama, esperar CI completa de JAX, congelar el SHA resultante, actualizar el pin de par exacto de Platform y ejecutar sus pisos medidos. Antes de integrar, pedir revisión independiente de escalón 3 sobre el SHA exacto. Comando inicial: `git diff --check && git log --format='%H%n%B' origin/master..HEAD`.

## Remediación Tier 3 MAJOR-1/2

**Fuente:** commits `3f2393fae56564d02123892c68cb9eda13d233ca`,
`dc7e857f1e4b920d97befd9d69838d168311aae0`,
`92a9d706c39c28f40d5975a8d39b7d1a5db2e70e` y
`e60e339`.

MAJOR-1 se cerró con el contrato cerrado de `jax.pipeline-list.json.1`:
las listas activas conservan sus dos claves externas (`pipelines`, `has_more`)
y las descartadas sus tres claves (`pipelines`, `has_more`,
`cursor_siguiente`). Las filas activas contienen exactamente `pipeline_id`,
`name`, `created_at`, `updated_at`, `duracion_s`, `costo_usd`, `causa`; las
descartadas agregan `descartado_at`. Solo F2-C añade `status`. Se rechazan
extras, tipos incorrectos, booleanos donde se espera número, números no
finitos, duplicados, más de 50 filas y cursores descartados inconsistentes.
`causa` admite solamente la forma real de Platform: `tipo` y, de forma
opcional, `paso` entero y `detalle` texto.

MAJOR-2 se cerró con `pipeline_status_snapshots(ids)`: una sola consulta de
cuatro columnas (`pipeline_id`, `user_id`, `tenant_id`, `status`) por PRIMARY
KEY para 1--50 IDs distintos, y snapshots inmutables marcados después de la
lectura. `JacobsPipelineStatusResolver.evidence_many(arguments_seq, scope)`
preserva orden y cardinalidad, emite evidencia individual `UNAVAILABLE` para
ausentes/fallos de lectura, `WRONG_SCOPE` para dueño distinto y conserva los
receipts F2-B individuales. `evidence()` delega a ese camino.

Evidencia local: suites focalizadas F2-E, **58 passed**; piso exacto de
proyección, **14 passed**; migración de pisos, **227 passed**. El piso F2-E
subió de 8 a 14. No había `JAX_DB_*` ni contenedor local de una base aislada,
por lo que no se ejecutó `EXPLAIN`; no se leyó ni modificó configuración,
usuario o base de producción. El test de forma de consulta verifica el único
`SELECT ... WHERE pipeline_id IN (...)`; queda pendiente medir el plan sobre
una base aislada proporcionada antes de consideración de producción.

La CI publicada del SHA `0c925e6` midió **19 passed** en
`tests/test_f2e_runtime_status_remediation.py`; el piso exacto
`governance/f2e-sr` subió de 18 a 19 para incluir la regresión de forma de la
consulta batch. La procedencia es el run `37286787324`, job `111687462660`.
No cambian los demás pisos ni la semántica F2-E-SR.

El `EXPLAIN` se ejecutó en modo lectura contra la base aislada existente
`jax_memory_test` en `127.0.0.1:3308`, para el `SELECT pipeline_id, status,
user_id, tenant_id FROM jacobs_pipelines WHERE pipeline_id IN (...)` de 50
IDs. El plan informó `type=range`, `possible_keys=PRIMARY`, `key=PRIMARY`,
`rows=50`, `Extra=Using where`, sin `Using filesort` ni `Using temporary`.
