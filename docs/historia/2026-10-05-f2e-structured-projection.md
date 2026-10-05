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
