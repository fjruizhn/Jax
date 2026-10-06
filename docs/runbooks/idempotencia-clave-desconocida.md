# Idempotencia de procesamiento: una clave en estado DESCONOCIDO

Contexto: `POST /procesamiento/trabajos` de LAS MANOS guarda cada `Idempotency-Key` en la tabla
`procesamiento_idempotencia` (`las_manos/procesamiento_idempotencia.py`). Una clave `confirmado=1` apunta a un
trabajo que SE CREO. Si el almacen de trabajos (`las_manos/logs/procesamiento_jobs.jsonl`, `ProcessingJobStore`) ya
no conoce ese `job_id` estando sano, la clave queda **DESCONOCIDA**: LAS MANOS contesta
`503 idempotencia_estado_desconocido` y no la retoma, porque retomar un trabajo que pudo seguir vivo duplicaria el OCR.
Una clave desconocida no tiene un trabajo vivo detras: LAS MANOS es un solo proceso y todo trabajo vivo esta en su
almacen en memoria.

Lo peor que puede pasar con una clave desconocida es un OCR duplicado de un trabajo ya perdido: la purga borra por
edad SIN mirar `confirmado`, y la clave cambia si cambia la composicion del trozo (llegaron filas nuevas al trozo
parcial, o se re-encolo una fila), asi que el reenvio siguiente crea un trabajo nuevo.

## Como se ve

- LAS MANOS (log de error): `idempotencia: la clave <hash12> esta confirmada para el trabajo <job_id> pero el
  almacen no lo conoce; estado DESCONOCIDO, no se retoma`.
- jax-platform (log de error + un Telegram por clave): `LAS MANOS no conoce el trabajo de la clave <hash12> (estado
  DESCONOCIDO) ... se salta el trozo del proyecto <uuid>`. Solo ESE trozo queda sin despachar; el resto del proyecto
  y los demas proyectos siguen. Los documentos quedan `en_cola`, no `error`. El log se repite una vez por
  enfriamiento (`proyectos.documentos.freno_incertidumbre_enfriamiento_s`), con las repeticiones omitidas.
- Dura **7 dias como minimo** (`JAX_PROCESAMIENTO_IDEMPOTENCIA_TTL_SEGUNDOS`) y hasta que corra la purga. La purga corre
  (como mucho una vez por hora y por proceso) tras un reclamo nuevo que gana, tras retomar un reclamo huerfano o un
  trabajo fallido, y tras el primer reclamo despues de reiniciar LAS MANOS (`_ultima_purga` arranca en 0).
- El aviso por Telegram se envia una vez por clave **y por proceso**: tras reiniciar jax-platform se repite una vez por
  cada clave que siga desconocida. Si hay muchas claves desconocidas a la vez, el despachador deja de avisar por clave y
  manda UN aviso agregado («N claves desconocidas, ver el log») con el mismo enfriamiento.

Con `503 procesamiento_no_disponible` es otro caso: el almacen perdio la integridad (disco lleno, ultima linea
truncada). Es global y transitorio: se arregla el almacen (espacio, linea truncada) y se reinicia LAS MANOS; el
despacho sigue solo. Esto NO es una clave desconocida.

## Lo que NO se hace

- **No se restaura ni se sobrescribe `procesamiento_jobs.jsonl` para "arreglar" la clave.** El almacen se carga UNA
  vez al importar (`processing_job_store.py`, `_load`): con LAS MANOS en marcha, restaurar el archivo no cambia nada;
  y el almacen solo AGREGA lineas (`job_store.py`, append): pisar el archivo con un respaldo viejo borra los trabajos
  posteriores en el siguiente reinicio. Si hace falta recuperar el JSONL, no es este runbook: escalar a Fernando
  (seria una FUSION de lineas, nunca una copia encima).
- **No se detiene LAS MANOS.** Al arrancar, `reconciliar_trabajos_huerfanos` falla TODOS los trabajos en vuelo de todos
  los tenants: detenerla para esto cuesta mas que la clave. Una clave desconocida no tiene trabajo vivo, y el DELETE
  de abajo esta condicionado por `(id, job_id, confirmado = 1)`.
- **No se pone `confirmado = 0` a mano**: la fila volveria a ser un huerfano que otro reintento retoma.

## Diagnosticar (solo lectura, sin ventana)

1. Hallar la clave desde el hash corto del log:
   `SELECT id, identidad_servicio, job_id, confirmado, creado_en FROM procesamiento_idempotencia
    WHERE confirmado = 1 AND LEFT(SHA2(clave, 256), 12) = '<hash12>';`
2. Buscar el `job_id` en el JSONL de produccion. Primero comprobar que el archivo EXISTE (un archivo inexistente no es
   «0 apariciones»):
   `test -f /srv/jax-prod/jax/las_manos/logs/procesamiento_jobs.jsonl && grep -c '"job_id": "<job_id>"'
    /srv/jax-prod/jax/las_manos/logs/procesamiento_jobs.jsonl`
   Si el archivo no existe, parar: escalar a Fernando. Ver tambien los respaldos del archivo:
   - **0 apariciones en el JSONL y 0 en los respaldos**: el trabajo no esta en ningun lado. Va al procedimiento de abajo.
   - **0 en el JSONL pero aparece en un respaldo**: el JSONL perdio lineas. **No borrar nada: escalar a Fernando.**
   - **Una o mas apariciones en el JSONL**: el trabajo ESTA en el JSONL pero el almacen no lo muestra: es un trabajo en
     **cuarentena** (`processing_job_store.py`: un evento sin `processing_ownership`, o un archivo con la historia
     rota para ese trabajo). Un trabajo creado por esta ruta siempre lleva dueno, asi que esto indica un JSONL
     editado o corrupto. **No tocar nada: escalar a Fernando** con el `job_id`, el hash y las lineas del JSONL.

## Resolver una clave cuyo trabajo no esta en el JSONL ni en los respaldos (necesita ventana o GO: escribe en produccion)

Solo con el diagnostico «0 y 0». Sin detener nada:

1. Respaldar: copia del JSONL actual (`cp -p`) y volcado de la tabla (`mariadb-dump <base> procesamiento_idempotencia`).
2. Borrar SOLO esa fila y solo si sigue igual:
   `DELETE FROM procesamiento_idempotencia WHERE id = <id> AND job_id = '<job_id>' AND confirmado = 1;`
   Debe afectar 1 fila. Si afecta 0, algo cambio: parar y escalar.
3. El siguiente ciclo del despachador manda el trozo con la misma clave y LAS MANOS crea un trabajo nuevo y lo
   confirma (probado en `las_manos/_procesamiento_idempotencia_db_test.py`,
   `test_el_procedimiento_del_runbook_resuelve_una_clave_desconocida`).
4. Verificar: el log del despachador deja de nombrar la clave y las filas del trozo pasan a `pendiente`.

Si algo de esto no se puede cumplir (no hay ventana, el diagnostico no es claro, el JSONL se toco), el paso correcto es
no borrar nada y escalar a Fernando: una clave desconocida cuesta un trozo detenido, no perdida de datos.
