# Traspaso · fix/authority-ledger-checkpoint-overlay

## Objetivo

Cerrar dos hallazgos del auditor de #377 (fuera de alcance allí): (1) `append`
escribía el evento y DESPUÉS el checkpoint — si el checkpoint fallaba, la cabeza
quedaba sin anclar y sin reporte; (2) `OVERLAY_ISSUED` hacia un corpus sin
ratificación vigente pasaba el replay. Encargo:
`~/encargos-codex/encargo-glm-g2-ledger-checkpoint-overlay.md`.
Base: `origin/fix/authority-resolution-loader-seal` @ c3d471ed (#379).
Diseño verificado: NINGÚN documento de docs/superpowers/specs menciona OVERLAY
— corre la decisión de Hyde.

## Hecho en esta rama

1. **Checkpoint fail-closed**: el evento no se considera aceptado hasta que su
   checkpoint quedó escrito. Si `checkpoint_store.append` falla,
   `LedgerCheckpointError` (tipado, nuevo) nombra el evento huérfano (id y
   secuencia) y el procedimiento; `verify_authority_ledger` con ancla reporta
   «cabeza sin checkpoint» con la salida documentada;
   `reanchor_authority_checkpoint(store, root, checkpoint)` re-ancla el
   checkpoint al head EXISTENTE tras verificar el stream completo sin ancla.
2. **Overlay exige ratificación vigente**: el replay rechaza OVERLAY_ISSUED
   cuyo `policy_corpus_hash` no tenga ninguna ratificación no revocada en ese
   punto del stream (misma lógica que ACTIVATION). Con el replay previo de
   #377, `append` niega antes de escribir. `audit_003` pasa a esperar el
  rechazo; la tanda MariaDB de 8 eventos se reordenó (overlay antes de
  revocar la ratificación).

3. **Ronda 2 contra el rechazo del escalón 3 (2026-10-07)**: reanchor solo
   procede ante `UnanchoredLedgerHeadError` y exige que el checkpoint vigente
   siga siendo prefijo del stream; append valida el ancla antes de escribir;
   el store rechaza retrocesos/secuencias repetidas y falla cerrado ante una
   última línea parcial. Se reemplazó `pytest.raises(Exception)` por
   `pytest.raises(LedgerIntegrityError)`. Pruebas de regresión para rollback,
   head huérfano, DB retrocedida y línea parcial.

4. **Verificación local de la ronda 2**: 16 pruebas focales pasaron; la suite
   completa `tests/policy/test_authority_ledger*.py`, ejecutada con
   `JAX_AUTHORITY_LEDGER_DOCKER_CMD='sudo -n docker'`, dio **79 passed**.
   `git diff --check` limpio. Se observó que el veredicto previo no cubre la
   carrera entre reanchor y append; queda reportada como riesgo residual.

## Pruebas nuevas (6) y mutantes

- checkpoint: fallo → huérfano tipado → verificador delata → reconciliación
  cierra; reconciliación exige head existente y stream válido (3).
- overlays: corpus jamás ratificado → rechazo pre-escritura; corpus con
  ratificación revocada → rechazo (2). MariaDB: overlay sin ratificación deja
  CERO filas nuevas (1; piso integration 5→6).
- Mutantes muertos: «ignorar el fallo del checkpoint» (2 fallos) y «aceptar
  overlay sin ratificación» (3 fallos). Tabla en la entrega.

## Pisos

- `authority-ledger-mariadb/integration`: 5 → **6** (piso.py verificar OK,
  docs/ci/pisos.md actualizado).
- codec 19, ratificaciones 12, seal 4, golden 7 — verificados sin cambios.
- Identity Foundation Shadow (sin piso en esta línea): 350 → **355**.
- Guardas CI: 237 passed. Vector dorado verde (bytes firmados sin cambio).

## Pendiente

- **Auditoría escalón 3 de `9470e4c1` (2026-10-07): RECHAZADO.** Halló dos BLOCK
  por carrera entre append/reanchor y publicación monotónica del checkpoint; un
  MAJOR por falta de `fsync` del directorio tras `os.replace`; y un MAJOR porque
  el replay estricto de OVERLAY puede invalidar ledgers históricos.
- **Corrección de carreras y durabilidad, en el árbol de trabajo:** lock sidecar
  `flock` mantenido desde antes de leer el ledger hasta confirmar DB, publicar y
  releer checkpoint; fsync del archivo y directorio; validación completa del log.
  Supuesto: writers en un único host con filesystem local.
- **Verificación:** `JAX_AUTHORITY_LEDGER_DOCKER_CMD='sudo -n docker' pytest -q
  tests/policy/test_authority_ledger*.py` → 85 passed; `git diff --check` limpio.
- **UNAVAILABLE:** `jaxctl authority` no pudo consultar el Block 4 operativo en
  hall9000. No está verificado si hay OVERLAY_ISSUED históricos sin ratificación
  vigente. Fernando debe decidir si, de existir, se conservan en cuarentena, se
  rechaza el ledger completo o se conserva su efecto histórico. Pregunta enviada
  el 2026-10-07; esperar respuesta antes de cambiar esa semántica.
- El PR #381 sigue abierto y apilado sobre #379 → #377; integrar en orden.

## Siguiente comando

Tras resolver la semántica de overlays históricos, completar el cambio, actualizar
este traspaso, publicar y solicitar auditoría escalón 3 del SHA exacto; verificar
CI de ese SHA. No integrar hasta que auditoría, CI y preflight correspondan al
mismo SHA.
