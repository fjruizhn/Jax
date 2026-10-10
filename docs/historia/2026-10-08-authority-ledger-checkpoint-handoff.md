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

## Reanudación 2026-10-07

- PR #381 sigue abierto. El head vigente era `aa8d31e3`; CI falló únicamente en
  `no-fail-open-except`, que exige una marca `fail-soft` en el `except Exception`
  de `test_concurrent_initializers_publish_one_genesis_anchor`. El bloque
  devuelve la excepción al hilo padre para que el propio test clasifique el
  resultado y falle ante cualquier clase inesperada.
- Se agregó esa justificación específica en la línea del `except`. Pendiente:
  revisar el diff, publicar el SHA nuevo, esperar CI completa y obtener auditoría
  escalón 3 de ese mismo SHA antes de cualquier integración.

## Auditoría e1843cef y cierre de BLOCKs · 2026-10-07

- Auditoría escalón 3 de `e1843cefa1aab52d4b34318fc3da0d5fc47f3069`:
  **RECHAZADO**, dos BLOCK. (1) `append()` podía crear desde cero un log con
  cualquier checkpoint positivo, y replay aceptaba primera secuencia 1 sin
  recibo, permitiendo reconstruir la evidencia externa desde la DB. (2) el
  replay degradaba un proveedor que solo tuviera `latest()` y omitía la
  validación del recibo por `hasattr`.
- Corrección en el árbol de trabajo: `TrustedCheckpointStore.append()` exige
  archivo existente; primera fila admite solo secuencia 0 o legacy 1; replay
  actual exige el contrato completo de `TrustedCheckpointStore` y siempre
  valida el recibo, también para log legacy seq=1. Usa una sola lectura completa
  del log como snapshot del head. Los dobles de prueba delegan el contrato
  completo.
- Verificación local tras el cierre: checkpoint **27 passed**; suite
  `JAX_AUTHORITY_LEDGER_DOCKER_CMD='sudo -n docker' pytest -q
  tests/policy/test_authority_ledger*.py` → **100 passed**;
  `python3 -m compileall -q policy/authority_ledger
  tests/policy/test_authority_ledger_checkpoint.py` y `git diff --check` limpios.
- Falta publicar el SHA nuevo, esperar CI y pedir auditoría escalón 3 de ese SHA.

## Cierre del BLOCK de UNKNOWN→VERIFIED · 2026-10-08

- Auditoría escalón 3 de `932f724e`: **RECHAZADO**, un BLOCK reproducido.
  Después de un `os.replace` cuyo fsync fallaba, un replay podía volver a sellar
  el head visible como autoridad actual sin completar durabilidad; lo mismo
  aplicaba al recibo de bootstrap secuencia cero.
- Corrección: el replay actual toma el mismo lock reentrante de publicación y,
  antes de reconstruir autoridad, fsync y vuelve a leer bajo ese lock el archivo
  de checkpoints, su directorio, el recibo de bootstrap y su directorio. Si no
  puede demostrar durabilidad, niega. Una lectura solo sella tras completar
  esa comprobación; writers que ya poseen el lock pueden verificar sin liberar
  la sección crítica.
- `docs/operations/trusted-files.md` ahora documenta el contrato de lock y
  durabilidad para los lectores de autoridad actual.
- Regresiones: append con resultado UNKNOWN seguido de replay; bootstrap con
  resultado UNKNOWN seguido de replay; fsync persistente impide verificar y la
  recuperación de fsync permite una verificación posterior.
- Verificación local: `tests/policy/test_authority_ledger*.py` y
  `tests/policy/test_decision_record.py` → **103 passed**; `compileall` y
  `git diff --check` limpios.
- Pendiente: commit/push de este cierre, CI completo y auditoría escalón 3 del
  SHA nuevo antes de integrar #381.
- Auditoría de `7339bba5` detectó además dos pruebas de decision replay que
  creaban un checkpoint aislado sin recibo. Se corrigieron para reutilizar el
  checkpoint y recibo emparejados del ledger de prueba. Este ajuste cambia el
  SHA; requiere CI y auditoría nuevas.
- Auditoría adicional de `7339bba5` detectó que fsync solo del directorio hoja
  no hace durable la entrada en sus ancestros si el árbol de directorios acaba
  de crearse. Corrección local: creación componente por componente con fsync de
  cada padre; bootstrap, append y replay sincronizan las cadenas de directorios
  hasta la raíz del filesystem. Se añadió regresión de directorios anidados.
  Verificación focal: checkpoint + decision replay → **32 passed**. Falta la
  suite completa, nuevo SHA/CI y auditoría Tier 3 exacta.


## Cierre del handoff · 2026-10-08

- Se cerró el hallazgo Tier 3 del SHA `7339bba5`: directorios confiables creados
  componente a componente y fsync de cada padre; replay fsync de archivo y
  ancestros bajo lock; decision replay reutiliza checkpoint/receipt emparejados.
- La adopción implícita de logs que comienzan en secuencia 1 fue eliminada. La
  documentación exige inventario vivo antes de desplegar; si el Block 4 está
  inaccesible o el formato es legacy, el despliegue queda bloqueado hasta una
  ceremonia de adopción firmada separada. `jaxctl authority` no está instalado
  en esta sesión, así que el estado operativo permanece UNAVAILABLE.
- El código y pruebas locales se verificaron; publicar y auditar el SHA final,
  esperar CI verde y seguir el procedimiento exacto de integración siguen
  pendientes. Este archivo se archiva antes de la auditoría final según el
  runbook de traspaso continuo.

## Cierre de los rechazos previos y reapilado · 2026-10-09

- Fernando autorizó continuar con la recomendación: reparar #381 sobre el
  `master` vigente y auditar un SHA nuevo antes de integrar. El auditor Tier 3
  había rechazado `4d493b98` por dos motivos: podía reabrirse el writer genérico
  de ratificaciones de #379 y una fila intermedia falsificada de checkpoints
  pasaba si el head final coincidía.
- Se reapiló el PR sobre `master@8c850bcec5cd0233db5487343210c0ba96689f97`.
  La resolución conserva `append_ratification_from_candidate` como único
  writer de RATIFICATION_GRANTED, con una captura `plain(view)` única y el lock
  y checkpoint durable; `append_authority_event` y su helper interno rechazan
  ratificaciones suministradas por caller.
- `verify_authority_ledger` ahora recorre el snapshot validado de
  `checkpoint_store.checkpoints()` y compara cada fila de secuencia positiva
  con `events[sequence-1]` (identidad, evento y hash). La regresión de una fila
  intermedia seq=2 falsa con un head posterior válido falló antes del fix
  (`DID NOT RAISE`) y pasó tras el fix.
- La suite MariaDB real midió **10 passed**. Por ser el último en medir, Codex
  elevó el piso de 9 a 10 en `ci/pisos.json`, el comentario de workflow y esta
  documentación; el verificador `piso.py` terminó con rc 0.
- Verificación local: 42 focales de activation/ratification/seal/checkpoint,
  39 de events/ratification/checkpoint, y 10 MariaDB pasaron; `git diff
  --check` limpio. Para Docker se usó `sudo -n -E` porque el usuario no está
  en el grupo `docker`.
- El commit de reapilado `c3701553d5e8ee2c6420fa2f03e4acad35118d9e` se publicó
  en la rama del PR. #381 ahora toma `master` como base. La CI completa y la
  auditoría Tier 3 del SHA final aún deben terminar antes de integrar.
- Decisión conceptual sobre overlays históricos: Fernando había escogido
  cuarentena el 2026-10-07; se mantiene esa semántica. La corrección actual no
  cambia bytes de la política: exige ratificación vigente para emisiones
  nuevas y conserva cuarentena determinista de las históricas.
- Alternativa descartada: mantener la base apilada en la rama #379; esa rama ya
  está integrada y mantenerla como base ocultaba el writer incompatible.

## Rechazo Tier 3 y corrección del writer/piso · 2026-10-09

- Sol auditó el SHA `c8421337e3c380ae4347acb14f6b4e12628b6915` contra
  `master@8c850bcec5cd0233db5487343210c0ba96689f97` y lo rechazó con dos BLOCK:
  el helper privado de ratificación aceptaba un `AuthorityEventIntent` arbitrario,
  y el piso Identity Foundation seguía en 968 pese a nuevas pruebas.
- Corrección: se eliminó `_append_ratification_from_candidate_unlocked`; la
  derivación desde candidate, verificación del ledger, firma, replay, escritura y
  publicación durable del checkpoint viven dentro de
  `append_ratification_from_candidate`, bajo su lock. Se añadió regresión que
  impide reintroducir el helper. No cambia la API pública ni la captura única de
  `plain(view)` para hash y proyección.
- La ejecución de la lista exacta de 75 paths del workflow midió
  **1003 passed, cero skipped** en Python 3.14.4. Piso, workflow y documentación
  se alinearon a 1003. El verificador `piso.py` pasó.
- Las pruebas focales (ratificación, eventos, checkpoints, overlays y sellos del
  loader) pasaron: **64 passed**. `git diff --check` quedó limpio.
- Estado: el nuevo SHA todavía requiere push, auditoría Tier 3 exacta, CI verde
  completa y el procedimiento de integración. El rechazo no autoriza integración
  del SHA anterior.
- Alternativa descartada: conservar el helper privado con validaciones adicionales;
  al aceptar payload de autoridad sigue creando otra frontera de firma. El writer
  permanece dentro del flujo público que deriva el intent del candidate validado.
