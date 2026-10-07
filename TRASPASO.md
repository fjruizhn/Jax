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
- **CI de `24e823b0`:** `no-fail-open-except` detectó dos `except Exception` en
  helpers de threads de pruebas; ambos capturan el error para que el hilo padre
  falle explícitamente. Marcados `fail-closed`; la guarda local quedó 21/21.
- Esta anotación requiere un SHA nuevo y CI nuevo; la auditoría final debe cubrir
  ese SHA exacto, junto con la resolución de overlays históricos.
- **Decisión de Fernando (2026-10-07): cuarentena** para OVERLAY_ISSUED histórico
  sin ratificación vigente. `jaxctl authority` no pudo consultar el Block 4
  operativo en hall9000 (`UNAVAILABLE`); no está verificado si hay eventos así.
- **Implementación de cuarentena, en el árbol de trabajo:** replay preserva los
  eventos firmados, los expone en `quarantined_overlays`, no los aplica aunque
  el corpus se ratifique después y permite revocarlos. La frontera de append
  sigue rechazando nuevas emisiones sin ratificación antes de escribir. Se
  conserva la compatibilidad posicional de `ReconstructedAuthorityState`.
- **Auditoría escalón 3 de `3167689a` (2026-10-07): RECHAZADO.** Encontró un
  BLOCK: append sin checkpoint podía tomar dos snapshots y persistir una nueva
  emisión después de una revocación concurrente. Encontró un MAJOR: error tras
  `os.replace` reportaba incorrectamente un evento huérfano y reanchor no era
  idempotente cuando el checkpoint ya coincidía con DB.
- **Corrección en el árbol de trabajo:** append usa el snapshot `existing_events`
  único para validar, firmar y reproducir; el store DB rechaza el candidato stale.
  Errores tras `os.replace` se tipan como resultado de publicación desconocido;
  reanchor vuelve a sincronizar, relee y verifica idempotentemente un checkpoint
  que ya coincide con el head.
- **Verificación local:** suite `tests/policy/test_authority_ledger*.py` → **90
  passed** con `JAX_AUTHORITY_LEDGER_DOCKER_CMD='sudo -n docker'`; tanda focal
  31 passed; `py_compile` y `git diff --check` limpios.
- **Auditoría escalón 3 de `019a4f23` (2026-10-07): RECHAZADO.** Encontró dos
  BLOCK: se podía escribir sin checkpoint y un replay sin ancla producía el mismo
  sello que autoridad actual; además, DB vacía + checkpoint ausente permitía
  bootstrap implícito. Encontró un MAJOR porque el tipo de error UNKNOWN se
  colapsaba en el mismo `LedgerCheckpointError` que una ausencia conocida.
- **Corrección completa en el árbol de trabajo:** `verify_authority_ledger` exige
  checkpoint; el replay histórico devuelve `HistoricalAuthorityState` distinto,
  no aceptado por consumidores actuales; append siempre verifica/ancla bajo lock.
  Bootstrap es explícito y usa checkpoint cero más recibo exclusivo separado en
  `/etc/jax/authority/trusted-checkpoint-bootstrap-receipt.json`; pérdida de log
  o recibo exige restaurar el par, nunca rebootstrap. Reanchor no crea evidencia
  ausente y puede avanzar desde checkpoint cero o recuperarse idempotentemente.
  UNKNOWN conserva su error público diferenciado.
- Runbooks actualizados: `docs/operations/trusted-files.md` y
  `docs/runbooks/authority-root-recovery.md`.
- **Verificación local final del árbol:** suite ledger + decision + execution →
  **136 passed**; `compileall` y `git diff --check` limpios.
- `jaxctl authority` operativo sigue **UNAVAILABLE**; no está verificado si el
  ledger durable contiene overlays históricos sin ratificación. La decisión de
  cuarentena sigue aplicada determinísticamente en replay.
- CI del SHA `019a4f23` seguía ejecutando `tests-puros` y `faro-jaula` al retomar
  estas correcciones; no corresponde al árbol actual y debe reemplazarse con CI
  del nuevo SHA.
- Falta publicar este árbol y solicitar auditoría escalón 3 del SHA exacto. No
  integrar #381; sigue apilado sobre #379 → #377, y se integra en orden.
- El PR #381 sigue abierto y apilado sobre #379 → #377; integrar en orden.

## Siguiente comando

Publicar los cierres de autoridad/bootstrap, solicitar auditoría escalón 3 del
SHA exacto y verificar CI de ese SHA. No integrar hasta que auditoría, CI y
preflight correspondan al mismo SHA.
