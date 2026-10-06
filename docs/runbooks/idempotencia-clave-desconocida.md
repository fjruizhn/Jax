# Idempotencia de procesamiento: una clave en estado DESCONOCIDO

Contexto: `POST /procesamiento/trabajos` de LAS MANOS guarda cada `Idempotency-Key` en la tabla
`procesamiento_idempotencia` (`las_manos/procesamiento_idempotencia.py`). Una clave `confirmado=1` apunta a un
trabajo que SE CREO. Si el almacen de trabajos (`logs/procesamiento_jobs.jsonl`, `ProcessingJobStore`) ya no
conoce ese `job_id` estando sano, la clave queda **DESCONOCIDA**: LAS MANOS contesta
`503 idempotencia_estado_desconocido` y NUNCA la retoma, porque retomar un trabajo que pudo seguir vivo duplicaria
el OCR.

## Como se ve

- LAS MANOS (log de error): `idempotencia: la clave <hash12> esta confirmada para el trabajo <job_id> pero el
  almacen no lo conoce; estado DESCONOCIDO, no se retoma`.
- jax-platform (log de error del despachador): `LAS MANOS rechazo la clave de idempotencia <hash12> ...
  idempotencia_estado_desconocido; se salta el proyecto <uuid>`. Solo ESE proyecto (y su dueno) queda sin despachar;
  los demas siguen. Los documentos quedan `en_cola`, no `error`.
- Es permanente hasta que la purga por edad borre la clave (`JAX_PROCESAMIENTO_IDEMPOTENCIA_TTL_SEGUNDOS`, 7 dias).

Con `503 procesamiento_no_disponible` es otro caso: el almacen perdio la integridad (disco lleno, ultima linea
truncada). Es global y transitorio: se arregla el almacen (espacio, linea truncada) y el despacho sigue solo.

## Resolverla a mano (necesita ventana o GO: escribe en produccion)

1. Hallar la clave desde el hash corto del log (solo lectura):
   `SELECT id, identidad_servicio, job_id, confirmado, creado_en FROM procesamiento_idempotencia
    WHERE confirmado = 1 AND LEFT(SHA2(clave, 256), 12) = '<hash12>';`
2. Decidir si el trabajo existio de verdad: buscar el `job_id` en `logs/procesamiento_jobs.jsonl` (y en sus
   respaldos) y en `GET /procesamiento/trabajos/<job_id>`. Si el trabajo SIGUE vivo o terminado bien, NO
   borrar la clave: arreglar primero el almacen (restaurar el JSONL) y la clave se resuelve sola.
3. Si el trabajo se perdio de verdad (se sabe que no corre), con respaldo previo de la tabla, borrar SOLO esa fila
   y solo si sigue igual:
   `DELETE FROM procesamiento_idempotencia WHERE id = <id> AND job_id = '<job_id>' AND confirmado = 1;`
   El siguiente ciclo del despachador manda el trozo con la misma clave y LAS MANOS crea un trabajo nuevo.
4. Comprobar en el log del despachador que el proyecto vuelve a despacharse.

Nunca poner `confirmado = 0` a mano: la fila volveria a ser un huerfano que otro reintento retoma, con el riesgo de
duplicar el OCR de un trabajo vivo.
