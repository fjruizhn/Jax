# Faro F1.1: durabilidad de reintentos del checkpoint Rule Authority

- **Fecha:** 2026-10-10.
- **Tipo:** HISTORIA / PENDIENTE.
- **Fuente:** veredicto Tier 3 exacto de `a2ed7590198e6f56d122bea898d1e0e7012f2a74`; test de CI `38089606207`, job `114323159699`; código JAX y reproducción local en el worktree propio.
- **Ámbito:** checkpoint externo Rule Authority en `policy/rule_authority/trusted_checkpoint.py`. No cambia el checkpoint de Block 4.

## Hallazgo y causa

Tier 3 encontró que un error de `fsync` del directorio posterior a `os.replace` deja el nuevo nombre visible aunque la persistencia tras crash sea desconocida. `confirmar()` solo releía el archivo y el CAS idempotente de `publicar()` retornaba éxito al ver el mismo head/anterior. Los caminos de recovery e idempotencia del storage dependen de `confirmar()`, así que ese comportamiento podía convertir la visibilidad en un `PERMIT` o consumo exitoso sin restablecer durabilidad.

La regresión reprodujo ambos falsos éxitos con fallo persistente inyectado al `fsync` del directorio. La causa fue la semántica débil de confirmación, no un defecto de cada camino de storage.

## Corrección

`confirmar(head)` ahora devuelve `True` solo después de leer el head exacto, hacer `fsync` del archivo regular de checkpoint y sincronizar su cadena de directorios. Los errores de esas barreras se convierten en `CheckpointInvalido`; un retry queda cerrado mientras la durabilidad siga incierta. El CAS idéntico llama a `confirmar()` antes de retornar éxito, de modo que un retry durable solo procede después de completar la reconciliación.

La prueba `test_reintento_no_confirma_head_visible_hasta_reconciliar_durabilidad` ejercita directamente `confirmar()` y el retry CAS de `publicar()` cuando falla el `fsync` del directorio; también inyecta fallo de `fsync` del archivo. No ejecuta una transacción de recovery de MariaDB: esos caminos consumen el mismo contrato `confirmar()`, que falla cerrado ante una barrera de durabilidad fallida. Al restaurar ambos `fsync`, confirma el head y permite el retry. En TDD, el test falló primero por aceptar el head visible y después volvió a fallar por el retorno prematuro del CAS; pasa tras ambos cambios.

La publicación nueva sincroniza los bytes del archivo temporal antes de `os.replace` y la cadena de directorios después del rename. Como el provider exige que `publicar()` responda solo tras completar esas barreras, se retiraron los `confirmar()` redundantes del camino de éxito nuevo (incluidos bootstrap, evaluación y consumo). El retry del mismo CAS mantiene `confirmar()` para reconciliar un intento anterior cuyo resultado durable fue incierto. Las comprobaciones de recovery y de heads preexistentes también conservan `confirmar()`.

La suite unitaria exacta del job Rule Authority (kernel, store, permit y checkpoint) midió `57 passed`. El piso subió de 56 a 57 en `ci/pisos.json`, y la medición se registró en `docs/ci/pisos.md`. En un benchmark local sintético de 200 publicaciones secuenciales con log creciente en filesystem temporal, `publicar()` midió p50 1.781 ms, p95 3.329 ms y 543.5 ops/s; el control que añade la segunda ronda redundante `confirmar()` dentro del mismo flock midió p50 3.148 ms, p95 5.723 ms y 320.6 ops/s. La pasada y el control usaron un directorio temporal nuevo; ops/s es 200 dividido entre el tiempo total observado. Es una comparación del primitive de checkpoint en este host; no mide concurrencia MariaDB ni latencia end-to-end de evaluate/consume. `git diff --check`, el parser JSON y `piso.py verificar authority-rule-unit/tests` pasan. La CI remota del SHA de corrección y su auditoría Tier 3 exacta siguen pendientes.

## Incidente separado: half-close

El mismo historial de CI de `a2ed7590` contiene un fallo distinto: `test_el_rele_con_half_close_deja_llegar_la_respuesta_antes_de_salir` dio stdout vacío, con `779 passed, 1 failed`; runs cercanos habían medido 780/780. No hay una reproducción local causal de esa intermitencia en este delta. No se cambió el runtime del relay y no se atribuye esta falla al checkpoint.

## Decisiones y pendientes

- **Quién decidió:** Fernando ordenó corregir la semántica raíz y añadir la regresión antes de fijar otro SHA y pedir nueva auditoría. Codex implementó el delta en su rama de seguimiento.
- **Alternativa descartada:** aceptar un head porque aparece al releer el archivo. La lectura no fuerza la entrada renombrada a almacenamiento durable y conserva el falso éxito.
- **Pendiente:** Tier 3 independiente sobre el SHA final y CI exacta; no integrar hasta cerrar esos gates y seguir el procedimiento de integración vigente. No se modificó `.github/workflows/policy.yml`.
