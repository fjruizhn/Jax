# Traspaso continuo · Faro F1.1 paso 6

## Objetivo

Completar en esta rama aislada el permiso `RulePermit` de un solo uso y su auditoría
durable, conforme a `docs/superpowers/specs/2026-10-05-faro-f1-1-rule-authority-kernel-design.md`
§9–§15. No integrar ni publicar hasta cerrar los gates y resolver la pregunta humana de
`OVERLAY_ISSUED`.

## Estado verificado

- Rama: `codex/faro-f11-kernel-step6`.
- Base actual: `bf6b05809e1c3505efd7447b33685796ca42e3b6` (master posterior a #386).
- Worktree: `/home/fruiz/jax/.worktrees/codex-faro-f11-kernel-step6`.
- Incremento local: checkpoint externo hash-chained con genesis explícito, CAS por head
  anterior, lock `flock`, escritura atómica, `fsync` de archivo/directorios y relectura.
  `MariaDBRuleDecisionStore` puede usarlo: mantiene el lock externo antes del lock DB,
  coteja el checkpoint con el audit head bloqueado antes de mutar, publica tras commit y
  falla cerrado si el head está atrasado o la publicación no se confirma.
- Modelo `RulePermitDraft` + `RulePermit`: proyección canónica hash-bound, campos cerrados,
  objetos profundamente inmutables; el store solo sella un permiso asociado a una evaluación
  sellada del kernel. El adapter en memoria aplica idempotencia exacta y rechaza reintentos
  con otro permiso.
- Verificación local: las suites de checkpoint, providers, modelos, permisos y store en
  memoria pasan (`303 passed`, checkpoint 9); MariaDB colecciona 17 pruebas. `compileall` y
  `git diff --check` pasan.
- Las pruebas MariaDB no se ejecutaron: el fixture Docker no puede abrir
  `/var/run/docker.sock` por permiso denegado.
- GLM/ZCode revisa read-only el SHA `76c8ecd0` (checkpoint/almacenamiento externo); su
  veredicto final aún está pendiente. Observó que fallo post-commit/post-anclaje queda
  cerrado hasta reconciliación explícita y que adapters futuros deben tomar primero el
  mismo `flock`; falta runbook auditado de reconciliación antes de wiring productivo.
  Kimi CLI no está disponible por límite 403.
- Estado remoto adicional: #385 está en la ascendencia de `master@bf6b0580`; su guard
  `38022192036` falló solo en `tests-puros` al perder MariaDB la conexión durante
  `SELECT VERSION()` (11 passed, 1 error). El mismo head pasó 12/12 en `pull_request`; no
  hay causa raíz probada. Fortalecí localmente el fixture para esperar servidor listo con
  query, reintentar solo errores de inicio transitorios y adjuntar `docker logs` si agota
  tiempo. No ejecutable localmente por falta de permiso al socket Docker; pendiente de CI.
  Las cuatro pruebas unitarias del readiness helper compartido pasan; la prueba MariaDB
  dirigida confirma el bloqueo ambiental antes del arranque del contenedor.
  #387 revierte #385, pero su CI actual falla `pisos-no-bajan`; rama ajena, sin cambios.
- Añadí regresión que forja una cadena hash-válida con un head histórico repetido. Pasa con
  el validador vigente; mutando temporalmente esa guardia, falla como se espera.
- Agregué inyección de fallo de `fsync` antes de `replace` (el head viejo permanece) y de
  `fsync` de directorio después de `replace` (error de publicación incierta; el nuevo archivo
  es visible). Las nueve pruebas de checkpoint y las cinco suites cercanas pasan: 303 total.
- El checkout compartido claude-skills está sincronizado; `claude-skills-sync pull`
  informó una divergencia local de `settings.json` de Claude, sin cambios en PENDIENTES.

### Actualización 2026-10-10 · consumo compartido Block 4 / Rule Authority

- Último commit de código: `abe866d8eeacfcc5478b36441b6d79cb5eccab6d`, local y no publicado;
  después se hizo un commit documental para mantener este handoff. Consultar `git rev-parse
  HEAD` para el tip vivo. El merge-base con
  `origin/master@bf6b05809e1c3505efd7447b33685796ca42e3b6` coincide. Las rondas 5–10 están
  incluidas en el commit de código.
- El fence compartido mantiene orden de flocks Block 4 → Rule Authority → BEGIN y orden
  InnoDB permit → head Block 4 → audit head. El owner entrega el `RulePermit` completo,
  canónico y sellado bajo lock; el store de Rule Authority es dueño de append de consumo,
  hash-chain, commit, publicación y confirmación externa antes de devolver
  `RulePermitConsumption`. Retry usa `permit_id + request_hash` y recupera la fila durable
  aunque el reloj produzca otro timestamp. El resultado post-commit distingue checkpoint
  no confirmado de fallo de cierre tras anchor confirmado.
- El principal de Rule Authority recibe SELECT-only sobre genesis/events/head de Block 4.
  Se añadieron cinco pruebas MariaDB cross-schema para principal/grants, retry, doble
  consumidor, rollback de INSERT, fallo de publicación y cierre post-commit; compilan y
  coleccionan, pero no se ejecutaron contra MariaDB porque Docker falla con
  `permission denied` en `/var/run/docker.sock`.
- Verificación local independiente: 8 pruebas focalizadas de fence/permit pasan; 30 pruebas
  de store/checkpoint/codec pasan; `py_compile`, `git diff --check` y colección de las cinco
  pruebas MariaDB pasan. La Tier 3 exacta de rondas 5–9 encontró y cerró estructuralmente
  los contratos de orden, identidad del checkpoint, token thread/epoch, consumo durable,
  anclaje y retry. La delta-audit de `abe866d8` aprobó con cambios, sin BLOCK/MAJOR, y dejó
  un MINOR documental: el handoff describía estado pre-commit y la corrección de lock como
  pendiente. La aserción está corregida en el commit de código. Se actualizó este handoff
  mediante un commit documental local; su SHA se consulta en Git. La suite MariaDB real no
  se ha ejecutado. No declarar listo para wiring ni integración hasta cerrar esos gates.
- La pregunta directa a Fernando sobre si `OVERLAY_ISSUED` exige ratificación previa no
  revocada del mismo hash y cuarentena permanente de overlays históricos inválidos sigue
  sin respuesta. La implementación de este paso no decide esa semántica.

### Actualización 2026-10-10 · fixes independientes del checkpoint

- Rebase comprobado contra `origin/master@bf6b05809e1c3505efd7447b33685796ca42e3b6`:
  `HEAD` ya descendía de ese SHA; `git rebase origin/master` terminó up-to-date. GitHub
  confirma que #381 se integró como `ec89ef1e431facf420ef3f95e20c089a81f52430` y que
  dicho commit es ancestro de master. Su worktree aparte no fue modificado.
- GLM/ZCode auditó read-only `76c8ecd0` (no el head actual): 1 MAJOR de verificación
  (MariaDB no ejecutada ni CI del branch), 3 MINOR y 4 LOW. La regresión de head histórico
  duplicado ya está en `b7147a3a`; las demás observaciones de API, genesis, temporales,
  permisos de lock, restauración DB/checkpoint y bloqueo entre procesos motivaron este
  diff local. La auditoría no cubre estos cambios posteriores ni `HEAD` actual.
- Diff local añade `AlmacenCheckpointsBloqueable` sin cambiar el protocolo base usado por
  providers en memoria; el lock se crea directamente con modo `0600`; el genesis explícito
  queda descrito correctamente; se limpian solo temporales privados del mismo store bajo
  flock. Añade pruebas de bloqueo entre procesos, modo inicial, limpieza, contratos y de
  rechazo antes de append si checkpoint adelanta a DB restaurada.
- Verificación de raíz: suites de checkpoint/providers/modelos/permiso/store: `308 passed`;
  pruebas de fence/transacción: `8 passed`; MariaDB storage colecciona `18 tests`. Un intento
  de correr el lote mixto confirmó que el fixture Docker falla con permiso denegado en
  `/var/run/docker.sock`; por tanto no cuenta como prueba MariaDB. `py_compile` y
  `git diff --check` pasan. El runbook auditado para reconciliar commit DB sin ancla sigue
  pendiente antes del wiring productivo.
- Pendiente: cerrar diff, commit firmado por Codex, pedir auditoría Tier 3 del SHA exacto,
  publicar solo tras ejecutar `/home/fruiz/claude-skills/bin/ventana estado`, y obtener CI
  MariaDB verde. No hacer merge hasta respuesta de Fernando sobre `OVERLAY_ISSUED`; esa
  decisión no se infiere de la auditoría ni del merge de #381.

### Actualización 2026-10-10 · CI #388 y correcciones de integración MariaDB

- La corrida CI `38032592015` sobre el SHA `49a212b803a89dbb4c8d616316a4340cfe285129`
  terminó `failure`. El job `tests-puros` alcanzó la suite Block 4 MariaDB y reportó
  `28 passed, 5 failed` de 33 casos. Los fallos eran: import ausente de `ThreadPoolExecutor`,
  tres tests de rechazo que capturaban la excepción dentro de `pytest.raises` pero no hacían
  rollback explícito del transaction owner y luego disparaban el guard de salida, y una
  aserción que esperaba `owner.cursor` aunque el contrato deliberadamente no publica cursor.
  Corregí import, cleanup explícito y aserción de ausencia de cursor.
- Medición de runner: 33 casos ejecutados/contados en
  `tests/policy/test_authority_ledger_storage_mariadb.py` +
  `tests/policy/test_authority_ledger_append_validation_mariadb.py` (28 pasaron, 5 fallaron).
  Por eso se actualiza el piso `authority-ledger-mariadb/integration` de 10 a 33, pendiente
  de confirmar 33 `passed` en CI sobre el SHA corregido. El paso Faro Rule Authority MariaDB
  fue omitido por el fallo anterior y aún no tiene conteo de runner; no cambiar su piso 12
  sin la corrida exacta.
- Se agregó el job unitario `faro-rule-authority-unit` con 3 suites y piso 25. Verificación
  local posterior al fix: 56 tests pasaron (escáneres, Bash de workflow, store, permit y
  checkpoint); JSON, `py_compile` y `git diff --check` pasan. La suite MariaDB no se pudo
  ejecutar localmente por permiso denegado a Docker, así que no sustituye la validación CI.
- Fixes de test/piso/handoff están en commits locales de Codex; revisar `git log -3` para los
  SHAs vivos. No se han publicado. El commit de cambios de escáner/job está en `f848f3c2`;
  el siguiente commit contiene los arreglos de los 5 fallos y piso 33. PR #388 continúa draft
  en `49a212b8`; ese SHA no incluye estos cambios.
- Siguiente: revisar y validar `git diff`, ejecutar el verificador de ventana inmediatamente
  antes de publicar el nuevo head de #388, y esperar CI exacta para ambos jobs MariaDB y los
  pisos. Repetir Tier 3 sobre el SHA final. No mergear mientras Fernando no resuelva la
  semántica de `OVERLAY_ISSUED`; tampoco habilitar wiring productivo sin el runbook de
  reconciliación auditado para commit DB exitoso sin ancla publicada.
- CI exacta nueva del head `880896ffc9c79b1abfbf71b16b7dc9199a18b054`: run `38033529303`.
  La auditoría Tier 3 del SHA registró `APROBADO CON CAMBIOS`, BLOCK 0, MAJOR 1 (CI no
  cerrada), MINOR 1 (al job unitario le faltaban dependencias); los tres findings menores
  previos sobre cableado, restore DB y wiring quedan cerrados. Reproduje el entorno mínimo:
  con solo pytest no colecciona por falta de `yaml`; luego de agregar PyYAML faltaba
  `cryptography`; con los pins `PyYAML==6.0.3` y `cryptography==49.0.0`, pasan los 25 tests.
  El workflow local ya fija esas dos dependencias.
- `archivos-de-test-en-ci` también falló en su paso de pisos, no en el scanner de archivos.
  Repro local exacta: `policy/tests/test_pisos_migracion_desde_master.py` exigía solo el job
  previo `pisos-no-bajan`; añadí `faro-rule-authority-unit` a `JOBS_NUEVOS`. Antes del fix,
  el lote fue `226 passed, 1 failed` (227 casos); el piso de 227 sigue correcto. Los logs
  finales del run completo aún deben confirmar ambos fixes y los gates MariaDB.
- CI `38033669969` sobre SHA `3784365fe1ae46ee17858cf29be6b0e2e3255157` confirmó el mismo
  contrato de Unicode en ambos jobs Faro: `faro-fase0` dio 779 passed + 1 failed en Python
  3.12, y `faro-rule-authority-unit` falló en colección con
  `UNSUPPORTED_UNICODE_VERSION: runtime=15.0.0, pinned=16.0.0`. El workflow ya contiene un
  cambio local a Python 3.14 en ambos jobs, coherente con otros jobs JAX que declaran que la
  canonicalización Unicode 16 requiere Python 3.14. Reproducción local con el lock Faro/CI
  hasheado en Python 3.14: la prueba de import pasó (`1 passed`); las 25 pruebas unitarias y
  los 227 tests de migración de pisos también pasan localmente. Aún falta una corrida de CI
  sobre el SHA que incorpora esta alineación.
- Recomendación Tier 3 read-only: mantener intacto el guard de UCD, usar Python 3.14 en
  `faro-fase0` y `faro-rule-authority-unit`; el resto del pipeline puede seguir en 3.12.
  Medición adicional en venv limpio Python 3.14.4 con los requirements Faro hasheados y el
  comando exacto: `780 passed` para fase 0 (incluida la prueba de límite RSS) y `25 passed`
  para Rule Authority; ambos pisos locales se satisfacen. Esos conteos deben confirmarse en
  CI sobre el siguiente head.

### Actualización 2026-10-10 · fallo de CI MariaDB Rule Authority

- La corrida exacta `38033872321` sobre `e993764aa08921f96e02bc8750def3f31629f7b2`
  terminó FAILURE solo en `tests-puros`. Los demás checks, incluido `faro-fase0` (780),
  Rule Authority unitario (25), pisos, permisos-proyectos, canonical-projections y scan,
  terminaron SUCCESS. Block 4 MariaDB confirmó 33 passed.
- La suite MariaDB de Rule Authority produjo `3 failed, 3 passed, 12 errors`: el fixture
  `tests/policy/test_faro_rule_authority_storage_mariadb.py::db` creaba solo el schema de
  Rule Authority, mientras su provisioner otorga SELECT a las tablas de verificación de
  Block 4 (`jax_authority.authority_ledger_genesis/events/head`). La causa fue
  `Table 'jax_authority.authority_ledger_genesis' doesn't exist`, no un fallo transitorio.
- Fix local, aún sin commit ni publicación: el fixture aplica primero
  `policy/authority_ledger/migrations/001_authority_ledger.sql` y después la migración propia
  de Rule Authority, en su misma MariaDB efímera. El intento local dirigido se bloqueó antes
  de iniciar el contenedor por `permission denied` en `/var/run/docker.sock`; CI debe
  confirmar el fix en un SHA nuevo.
- Después del fix, verificación local disponible: Rule Authority unitario `25 passed`, la
  suite MariaDB colecciona sus `18 tests`, `py_compile` y `git diff --check` pasan. No se
  pudo ejecutar la MariaDB real localmente por el permiso del socket; estos resultados no
  sustituyen la nueva CI.
- Tier 3 exacta de `e993764a`: `APROBADO CON CAMBIOS`, BLOCK 0, MAJOR 1 abierto por ese
  fallo de CI, MINOR 0, LOW 3. El veredicto no cubre el fix local. Antes de publicar el SHA
  corregido: ejecutar pruebas disponibles, `git diff --check`, revisar el diff y consultar
  `/home/fruiz/claude-skills/bin/ventana estado` inmediatamente antes del push. Solicitar
  delta-audit y CI exacta nuevas; no integrar mientras la decisión OVERLAY de Fernando siga
  pendiente.

## Límites de autoridad y alcance

- La decisión directa de Fernando sigue pendiente: si `OVERLAY_ISSUED` requiere una
  ratificación previa no revocada del mismo hash y cuarentena permanente para overlays
  históricos inválidos.
- El kernel de este paso debe ignorar overlays y ratificaciones del corpus; solo usa el
  grant individual de Block 4 derivado del ledger verificado.
- No tocar el worktree `/home/fruiz/wt/jax-ledger-checkpoint` ni `codex/faro-r2`.
- No publicar ni integrar hasta resolver la pregunta de OVERLAY, cerrar auditoría/CI y
  revalidar la ventana inmediatamente antes de publicar/integrar. El trabajo aislado y el
  rebase técnico pueden continuar sin decidir la semántica de OVERLAY.

## Siguiente acción exacta

1. Ejecutar las cinco pruebas cross-schema contra MariaDB real con el principal provisionado; no inferir ese resultado de colección o de pruebas puras. Si el runner local sigue sin Docker, ejecutarlas en CI apropiada antes de wiring.
2. Completar el wiring del kernel `evaluate`/`consume`, cobertura de clasificación/STOP/expiración/revocación/cambio de capability/pin y mediciones O(n) p95/RSS/EXPLAIN.
3. Resolver la pregunta de OVERLAY con Fernando. No publicar ni integrar mientras siga pendiente; revalidar `bin/ventana estado` inmediatamente antes de cualquier publicación o integración.

## Archivos tocados

- `policy/rule_authority/trusted_checkpoint.py`
- `policy/rule_authority/permit.py`
- `policy/rule_authority/__init__.py`
- `policy/rule_authority/storage.py`
- `policy/rule_authority/provisioning.py`
- `policy/authority_ledger/storage.py`
- `policy/rule_authority/store.py`
- `policy/rule_authority/migrations/002_enable_atomic_permits.sql`
- `tests/policy/test_faro_rule_permit.py`
- `tests/policy/test_faro_rule_authority_store.py`
- `tests/policy/test_rule_authority_checkpoint.py`
- `tests/policy/test_faro_rule_authority_storage_mariadb.py`
- `tests/policy/test_authority_ledger_storage_mariadb.py`
- `docs/superpowers/plans/2026-10-09-faro-f11-step6-checkpoint.md`
- `TRASPASO.md`
