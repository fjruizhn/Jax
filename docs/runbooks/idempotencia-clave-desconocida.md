# Idempotencia de procesamiento: una clave en estado DESCONOCIDO

Contexto: `POST /procesamiento/trabajos` de LAS MANOS guarda cada `Idempotency-Key` en la tabla
`procesamiento_idempotencia` (`las_manos/procesamiento_idempotencia.py`). Una clave `confirmado=1` apunta a un
trabajo que SE CREO. Si el almacen de trabajos (`las_manos/logs/procesamiento_jobs.jsonl`, `ProcessingJobStore`) ya
no conoce ese `job_id` estando sano, la clave queda **DESCONOCIDA**: LAS MANOS contesta
`503 idempotencia_estado_desconocido` y NUNCA la retoma, porque retomar un trabajo que pudo seguir vivo duplicaria
el OCR.

## Como se ve

- LAS MANOS (log de error): `idempotencia: la clave <hash12> esta confirmada para el trabajo <job_id> pero el
  almacen no lo conoce; estado DESCONOCIDO, no se retoma`.
- jax-platform (log de error + un Telegram por clave): `LAS MANOS no conoce el trabajo de la clave <hash12> (estado
  DESCONOCIDO) ... se salta el trozo del proyecto <uuid>`. Solo ESE trozo queda sin despachar; el resto del proyecto
  y los demas proyectos siguen. Los documentos quedan `en_cola`, no `error`. El log se repite una vez por
  enfriamiento (`proyectos.documentos.freno_incertidumbre_enfriamiento_s`), con las repeticiones omitidas.
- Dura **7 dias como minimo** (`JAX_PROCESAMIENTO_IDEMPOTENCIA_TTL_SEGUNDOS`) y hasta el **siguiente reclamo nuevo de
  cualquier clave**: la purga solo corre tras un reclamo nuevo que gana (como mucho una vez por hora).

Con `503 procesamiento_no_disponible` es otro caso: el almacen perdio la integridad (disco lleno, ultima linea
truncada). Es global y transitorio: se arregla el almacen (espacio, linea truncada) y se reinicia LAS MANOS; el
despacho sigue solo. Esto NO es una clave desconocida.

## Lo que NO se hace

- **No se restaura ni se sobrescribe `procesamiento_jobs.jsonl` para "arreglar" la clave.** El almacen se carga UNA
  vez al importar (`processing_job_store.py`, `_load`): con LAS MANOS en marcha, restaurar el archivo no cambia nada;
  y el almacen solo AGREGA lineas (`job_store.py`, append): pisar el archivo con un respaldo viejo borra los trabajos
  posteriores en el siguiente reinicio. Si hace falta recuperar el JSONL, no es este runbook: escalar a Fernando
  (seria una FUSION de lineas, nunca una copia encima).
- **No se pone `confirmado = 0` a mano**: la fila volveria a ser un huerfano que otro reintento retoma, con riesgo de
  duplicar el OCR de un trabajo vivo.

## Diagnosticar (solo lectura, sin ventana)

1. Hallar la clave desde el hash corto del log:
   `SELECT id, identidad_servicio, job_id, confirmado, creado_en FROM procesamiento_idempotencia
    WHERE confirmado = 1 AND LEFT(SHA2(clave, 256), 12) = '<hash12>';`
2. Buscar el `job_id` en el JSONL: `grep -c '"job_id": "<job_id>"' las_manos/logs/procesamiento_jobs.jsonl` (y en sus
   respaldos).
   - **0 apariciones**: el trabajo no esta en el almacen (se perdio el archivo, o nunca llego a escribirse).
     Va al procedimiento de abajo.
   - **Una o mas apariciones**: el trabajo ESTA en el JSONL pero el almacen no lo muestra: es un trabajo en
     **cuarentena** (`processing_job_store.py`: un evento sin `processing_ownership`, o un archivo con la historia
     rota para ese trabajo). Un trabajo creado por esta ruta siempre lleva dueno, asi que esto indica un JSONL
     editado o corrupto. **No tocar nada: escalar a Fernando** con el `job_id`, el hash y las lineas del JSONL.

## Resolver una clave cuyo trabajo NO esta en el JSONL (necesita ventana o GO: escribe en produccion)

Solo con el diagnostico "0 apariciones". Se hace con LAS MANOS **detenido**, para que ningun trabajo siga corriendo en
memoria mientras se borra la clave:

1. Respaldar: copia del JSONL actual (`cp -p`) y volcado de la tabla (`mariadb-dump <base> procesamiento_idempotencia`).
2. Detener LAS MANOS (con ventana/GO). Volver a comprobar que el `job_id` sigue sin aparecer en el JSONL.
3. Borrar SOLO esa fila y solo si sigue igual:
   `DELETE FROM procesamiento_idempotencia WHERE id = <id> AND job_id = '<job_id>' AND confirmado = 1;`
   (Debe afectar 1 fila. Si afecta 0, algo cambio: parar y escalar.)
4. Iniciar LAS MANOS. El siguiente ciclo del despachador manda el trozo con la misma clave y LAS MANOS crea un trabajo
   nuevo y lo confirma (probado en `las_manos/_procesamiento_idempotencia_db_test.py`,
   `test_el_procedimiento_del_runbook_resuelve_una_clave_desconocida`).
5. Verificar: el log del despachador deja de nombrar la clave y las filas del trozo pasan a `pendiente`.

Si algo de esto no se puede cumplir (no hay ventana, el diagnostico no es claro, el JSONL se toco), el paso correcto es
no borrar nada y escalar a Fernando: una clave desconocida cuesta un trozo detenido, no perdida de datos.
