# Adopción controlada de memoria legacy

El plan y sus snapshots son registros privados de revisión. No otorgan autoridad.
La API B9 revalida administrador, dueño, contenido y destino dentro de la
transacción que registra IMPORT_LEGACY, RE_SCOPE y binding tenant-qualified.

1. Crear y restaurar un respaldo completo en una instancia aislada sin red.
   Guardar su manifiesto 0600 con `backup_path`, `backup_size`, `backup_sha256`,
   `restoration_verified: true`, `restored_at`, `restored_tables` y
   `restored_counts`. Verificar las tablas de memoria, usuarios y procedencia.
2. Con credenciales provistas externamente al proceso y sin cargar `/etc/jax`,
   ejecutar `build_plan(pool, "legacy")` y `private_write(path, plan)`.
   El dry-run no escribe en la base. `summary(plan)` muestra exclusivamente
   conteos y razones. Los snapshots y digests nunca se imprimen.
3. Revisar el archivo privado: dueño viene de `jax_users` activo y su tenant;
   referencias deben coincidir; proyectos necesitan scope y membresía vigentes.
   Superseded, expired, pendientes terminados y filas sin dueño van a cuarentena
   visible en el plan, fuera de B9. Nunca se infiere dueño del contenido.
4. Aplicar `apply_plan(api, plan_path, manifest_path, actor_user_id=admin_id)`.
   El actor es el administrador real, sin impersonar al dueño de la memoria.
   Se exige plan guardado íntegro y respaldo aún idéntico al restaurado.
   Cada importación usa `expected_source_digest` y un binding único para
   reintentos; un cambio concurrente de fuente aborta visiblemente.
5. Verificar los bindings, eventos, proyecciones y lectura del usuario dueño.
   Una repetición del mismo plan debe devolver los mismos IDs sin eventos nuevos.

La verificación legacy no se convierte automáticamente en verificación B9.
La memoria adoptada continúa siendo contexto histórico, no prueba de verdad actual.
No hay borrado ni reversión destructiva automática; ante discrepancias se detiene
la aplicación y se conserva el plan para reconciliación controlada.

## CLI operacional revisable

El principal ejecuta este comando desde el checkout fusionado y aprobado.
Las rutas son archivos privados provistos explícitamente; no se cargan
credenciales al importar el CLI. El deadline total es 300 segundos por defecto
(configurable entre 1 y 900).

```sh
python ops/memory/adopt_legacy.py \
  --env-file /ruta/privada/credenciales.env \
  --plan /ruta/privada/adoption-plan.json \
  --backup-manifest /ruta/privada/manifest.json \
  --actor-user-id ADMIN_ID \
  --expected-database BASE_APROBADA \
  --verify-owner-id OWNER_ID \
  --verify-other-user-id OTHER_ID \
  --timeout-seconds 300
```

El comando exige administrador vigente, tenant real y destinos del plan dentro
de ese tenant. Aplica y repite el mismo plan; cualquier evento nuevo en la
repetición hace fallar la operación. Informa exclusivamente conteos y tipos de
error, sin contenidos, digests ni credenciales.

Los conteos de lectura owner/other son diagnósticos operacionales autorizados
por el administrador y revalidados con el resolver actual. No simulan JWT ni
sesiones de chat, y no prueban que el chat esté desplegado o funcionando. El
límite de lectura es 100; los conteos pueden estar truncados. La comprobación
final del chat requiere un turno real observado por el principal.
