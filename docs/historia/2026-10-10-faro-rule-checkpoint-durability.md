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

La prueba `test_reintento_no_confirma_head_visible_hasta_reconciliar_durabilidad` exige que fallen la confirmación, el recovery equivalente y el retry CAS cuando falla el `fsync` del directorio; también inyecta fallo de `fsync` del archivo. Al restaurar ambos, confirma el head y permite el retry. En TDD, el test falló primero por aceptar el head visible y después volvió a fallar por el retorno prematuro del CAS; pasa tras ambos cambios.

La suite unitaria exacta del job Rule Authority (kernel, store, permit y checkpoint) midió `57 passed`. El piso subió de 56 a 57 en `ci/pisos.json`, y la medición se registró en `docs/ci/pisos.md`. `git diff --check`, el parser JSON y `piso.py verificar authority-rule-unit/tests` pasan. La CI remota del SHA de corrección y su auditoría Tier 3 exacta siguen pendientes.

## Incidente separado: half-close

El mismo historial de CI de `a2ed7590` contiene un fallo distinto: `test_el_rele_con_half_close_deja_llegar_la_respuesta_antes_de_salir` dio stdout vacío, con `779 passed, 1 failed`; runs cercanos habían medido 780/780. No hay una reproducción local causal de esa intermitencia en este delta. No se cambió el runtime del relay y no se atribuye esta falla al checkpoint.

## Decisiones y pendientes

- **Quién decidió:** Fernando ordenó corregir la semántica raíz y añadir la regresión antes de fijar otro SHA y pedir nueva auditoría. Codex implementó el delta en su rama de seguimiento.
- **Alternativa descartada:** aceptar un head porque aparece al releer el archivo. La lectura no fuerza la entrada renombrada a almacenamiento durable y conserva el falso éxito.
- **Pendiente:** Tier 3 independiente sobre el SHA final y CI exacta; no integrar hasta cerrar esos gates y seguir el procedimiento de integración vigente. No se modificó `.github/workflows/policy.yml`.
