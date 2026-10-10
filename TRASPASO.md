# Traspaso continuo · Faro F1.1 paso 6

## Estado vivo 2026-10-10 · follow-up tras delta Tier 3

- #389 está integrado en `0f9b7963`; #391 quedó integrado después como `0d6484f2`.
  `git fetch origin master` y GitHub API confirman ese tip oficial. Reapilé localmente #388
  sobre `0d6484f2`; el rebase conservó el handoff de esta tarea cuando el primer commit chocó
  con la eliminación de `TRASPASO.md` en #391. La publicación de #388 sigue pausada por
  instrucción de Fernando: no push, PR update ni integración.
- El piso genérico de `archivos-de-test-en-ci/pisos` quedó en `^233 passed`; el comando
  exacto de los tres módulos dio `233 passed` localmente y coincide con CI #389 run
  `38044350914` sobre `c78080442da81a468768a7af5b2c5b748ff56b37`.
- Re-medí tras reapilar: `authority-rule-unit/tests` = `52 passed` (piso verificado);
  Identity Foundation Shadow = `1044 passed` (piso verificado); `archivos-de-test-en-ci/pisos`
  = `233 passed` (piso verificado). Los comandos unitario e Identity se tomaron del workflow;
  el primero de Identity se adaptó de `python` a `python3` porque este host carece del alias.
  Las suites MariaDB coleccionan 40 (Block 4) y 19 (Rule Authority); no se ejecutaron porque
  este host no puede abrir `/var/run/docker.sock`. CI exacta sigue pendiente para el SHA de
  rebase y los pisos MariaDB.
- El post-merge-guard exacto de #389, run `38045178033`, terminó SUCCESS (30/30 jobs).
  #388 remoto sigue en head `3eac3fff`; sus checks verdes corresponden a ese head viejo y no
  cubren la línea reapilada. El soporte separado de aserción de grants sigue bajo la otra
  sesión; esta rama no incorpora ni publica ese cambio.
- Reconciliación de linaje local de #388 (solo lectura, 2026-10-10): el PR remoto continúa en
  `3eac3fff`, con merge-base `bf6b0580` frente al `HEAD` local. `git range-diff` mapea los 31
  commits remotos: 30 son patch-equivalent; el commit inicial de checkpoint no iguala el
  patch-id completo porque la versión remota crea `TRASPASO.md` y la local actualiza el archivo
  que ya existía. El patch-id restringido a plan/código/tests es idéntico. La
  comparación de árboles completos entre sus SHAs también incluye soporte de pisos integrado
  posteriormente por #389/#391; esos cambios no pertenecen al commit de checkpoint ni se deben
  interpretar como divergencia de su código. La línea local añade después el merge de #389,
  dicho soporte y los commits locales de paso 6. Esta rama local es la continuación que pretende
  continuar #388 sobre master, por instrucción directa de Fernando después del guard. Tip actual
  confirmado por GitHub API y `git fetch`: `master`=`0d6484f2` (merge #391). No se ha actualizado
  el PR remoto por la pausa de publicación.
- Los commits locales `b325613e`, `eeb65d1c` y `511e83f0` no están publicados. Tier 3 GLM exacto de
  `b325613e`: NO APROBADO; BLOCK por carreras de providers contra `consume()` sin tests de
  kernel y BLOCK por falta de CI exacta/MariaDB real; MAJOR por relación de linaje/supersession
  no resuelta frente al head remoto antiguo de #388, y MAJOR porque `limites_de()` permitía
  forma vacía REVERSIBLE. La delta exacta de `eeb65d1c` confirmó ese MAJOR cerrado, y encontró
  dos MINOR nuevos: excepción sin DENY estable cuando `request.rule_hash` acompaña una regla
  vacía, e import sin uso. `511e83f0` los corrigió y añadió pruebas de ambas ramas; su delta
  Tier 3 está pendiente. El piso unitario sube localmente a 46.
- Trabajo local: el kernel valida alcance/límites individuales y construye un candidato
  PERMIT con TTL limitado por configuración, regla y grant; una regla sin quantity, amount,
  frequency ni tope vuelve a fallar cerrada, conforme a la condición aceptada en Tier 3 de
  #371 ronda 3. `consume()` recupera el retry durable por ID y, para un
  consumo nuevo, conserva leases de pin/clasificación/STOP, bloquea permiso antes de Block 4,
  reconstruye la solicitud canónica y revalida grant, regla, scope, expiración, contrato y
  límites antes del append compartido. Argumentos no vacíos y frecuencia/tope acumulativo
  siguen fallando cerrados por falta de matcher/contador durable.
- GLM/ZCode hizo revisión read-only del diff local: no encontró un defecto concreto en el
  orden de locks/recovery inspeccionado, pero marcó como MAJOR la falta de carreras del kernel
  de extremo a extremo exigidas por §13, y como hallazgo histórico la relajación de reglas sin
  límites (ya restaurada). Es una revisión del diff sin commit, no auditoría Tier 3 final.
- La respuesta Tier 3 de `b325613e` también marcó MINOR: rama `NINGUNA` muerta del kernel
  (ya retirada en el worktree posterior), `TEXT` de migration 004 sin CHECK de forma,
  `es_ancestro` bajo locks sin medición de retención, test ausente del invariante flock-
  primero y prueba AST ausente. Evaluar cada punto en la auditoría siguiente.
- Falta probar el flujo completo del kernel contra MariaDB y confirmar en CI los caminos de
  emisión y consumo. Las suites MariaDB no corrieron localmente. El SHA tras rebase necesita
  auditoría Tier 3 y CI exactas antes de cualquier publicación o integración.
- Verificación disponible en el worktree posterior: el conjunto focalizado de kernel/modelos/
  permit/store/providers/checkpoint dio `333 passed`; el job unitario exacto da ahora `52 passed`.
  Se añadieron seis casos con barreras para STOP, pin y clasificación: cambios ganadores antes
  del lease y escritores bloqueados mientras la llamada al append del store está detenida. El
  harness mantiene fake el owner transaccional/persistencia; no prueba escritura durable real ni
  cierra el BLOCK Tier 3 de MariaDB o la carrera integral de revocación/consumo. El delta-audit
  GLM de `d37279dc` encontró tres MINOR (liberación de barreras al fallar, observación antes del
  append y deadline de 5 s); el commit posterior `546c5d95` extiende la pausa hasta la llamada al
  append, y `d144903f` mueve la liberación dentro del pool y amplía el deadline a 10 s.
  El verificador falló con el piso anterior 46; alinear a 52 pasó localmente y CI exacta aún debe
  confirmar el SHA final. Identity Shadow
  dio `1044 passed`; `git diff --check` y `python3 -m json.tool ci/pisos.json` pasan.
- Siguiente acción concreta: solicitar delta-audit Tier 3 del SHA final reapilado; completar
  pruebas E2E de consume/revocación con owner MariaDB real. La publicación requiere que Fernando
  levante la pausa; cualquier integración toca `.github/workflows/policy.yml` reservado a
  Fernando. Tras autorización de publicación, repetir CI exacta y auditoría del SHA que se
  publique, conservar el veredicto y cumplir `post-merge-guard`.

---

## Objetivo

Completar en esta rama aislada el permiso `RulePermit` de un solo uso y su auditoría
durable, conforme a `docs/superpowers/specs/2026-10-05-faro-f1-1-rule-authority-kernel-design.md`
§9–§15. No integrar hasta cerrar los gates y cumplir la autoridad reservada de workflows.

## Estado verificado

- Rama: `codex/faro-f11-kernel-step6`.
- Base actual: `0d6484f2f56059e9fb0d11b04de8990375ee7c53` (master posterior a #391).
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

### Actualización 2026-10-10 · corrida CI del fix de fixture

- CI exacta `38035001043/attempts/1`, SHA `3857b4bc84a315346eedd11a9b32640427e2099e`: la suite Rule
  Authority MariaDB ejecutó `18 passed in 63.07s`. El error por falta de
  `authority_ledger_genesis` quedó resuelto al aplicar la migración Block 4 en el fixture.
  `tests-puros` sigue rojo únicamente en este paso porque el piso aún esperaba 12; elevé el
  piso en `ci/pisos.json` y `docs/ci/pisos.md` a los 18 medidos.
- Ese mismo intento (`38035001043/attempts/1`) reportó en `faro-fase0`:
  `779 passed, 1 failed` en
  `test_el_rele_con_half_close_deja_llegar_la_respuesta_antes_de_salir` (`tests/test_faro_puerto_e2e.py:136`),
  con stdout vacío antes de intentar parsear la respuesta. No hay diagnóstico causal todavía;
  no se cambia el runtime por conjetura. El intento 2 de ese run, relanzado para observar el
  mismo SHA, fue cancelado al publicarse el SHA siguiente; no aportó evidencia adicional. En
  la CI nueva de `2d6a0f41`, fase 0 pasó.
- El delta-audit Tier 3 exacto de `3857b4bc` mantiene un MAJOR abierto mientras CI no cierre.
  El piso actualizado cambia el SHA: correr verificaciones locales disponibles, actualizar
  este handoff, commit Codex, consultar `/home/fruiz/claude-skills/bin/ventana estado`
  inmediatamente antes del push, y pedir delta-audit + CI exactas para el nuevo SHA.
- No integrar hasta que CI y auditoría exactas cierren y Fernando responda la pregunta
  `OVERLAY_ISSUED`; esa respuesta sigue pendiente.

### Cierre de CI y auditoría del piso · 2026-10-10

- CI exacta de `2d6a0f41820c7f45ea4c8c8b85dd510623bb04d8`: policy run `38035792710`
  SUCCESS (31/31 jobs), canonical projections `38035792716` SUCCESS, secret-scan
  `38035792712` SUCCESS. `tests-puros` confirmó los dos pasos MariaDB y sus pisos; `faro-fase0`
  pasó. `gh pr checks 388` mostró todos los checks en pass para ese head.
- Addendum Tier 3 exacto del mismo SHA verificó la coincidencia PR/run/head y cerró el MAJOR:
  `APROBADO CON CAMBIOS`, BLOCK 0, MAJOR 0, MINOR 0, LOW 3. El auditor no encontró hallazgos
  nuevos en el delta de pisos y documentación.
- El valor de Rule Authority se mantiene en 18. La falla previa de half-close en `3857b4bc`
  no se reprodujo en `2d6a0f41`; queda registrada como observación histórica, sin fix de runtime.
- El head de referencia de esta actualización es `2d6a0f41`; este commit documental crea un
  SHA nuevo, que debe recibir su propia CI y delta-audit exacta antes de presentarlo como head
  final listo. La decisión `OVERLAY_ISSUED` continúa pendiente y bloquea merge.

### Seguimiento CI de trazabilidad · 2026-10-10

- CI exacta `38036905805`, SHA `05dfff13715f269ff5583ce99d0b4d61129a5f2f`: 30/31 jobs
  SUCCESS; el único fallo es `faro-fase0`. `tests-puros` terminó SUCCESS, incluidos Block 4
  MariaDB y Rule Authority MariaDB con piso 18; canonical projections `38036905810` y
  secret-scan `38036905796` también SUCCESS.
- El fallo volvió a ser `test_el_rele_con_half_close_deja_llegar_la_respuesta_antes_de_salir`:
  `779 passed, 1 failed`, stdout vacío e `IndexError` al tomar la primera línea. En `2d6a0f41`
  el mismo test pasó; esto todavía no identifica una causa. El test previo ya comprobaba que
  `proc.wait()` devolviera 0, pero no capturaba stderr ni incluía exit code/stdout en el fallo
  por respuesta vacía. El cambio captura esos datos y agrega contexto a las aserciones. No
  cambiar el runtime hasta identificar la causa.
- No fue posible reproducir localmente: el Python del checkout no tiene el módulo `mcp`, así
  que la colección de `tests/test_faro_puerto_e2e.py` falla por dependencia ausente. CI exacta
  debe ser la fuente de resultado para esta prueba.

### Cierre CI de la instrumentación half-close · 2026-10-10

- SHA `bf8bee461a21860db1427199ce0be83d465999c2`: policy run `38037918992` terminó SUCCESS,
  31/31 jobs; canonical projections `38037918990` y secret-scan `38037918993` SUCCESS.
  `tests-puros` pasó, con Block 4 MariaDB y Rule Authority MariaDB en sus pisos; permisos y
  pisos pasaron.
- `faro-fase0` pasó `780 passed in 53.29s`; el test half-close instrumentado PASSED. La
  captura adicional no reveló la causa de los fallos anteriores porque no se reprodujeron.
  Mantener la observación histórica de `3857b4bc` y `05dfff13`; no inferir causa ambiental ni
  modificar runtime sin nueva evidencia.
- La Tier 3 del delta `bf8bee46` recomendó precisar el historial del exit code; la corrección
  está incluida arriba. Pedir auditoría exacta del nuevo SHA y CI exacta tras publicarlo.

### Actualización 2026-10-10 · revisión del contrato del kernel (Tier 3)

- Contexto histórico al iniciar la revisión: #388 apuntaba a `445859bdb90922410983980f2f645755d4fafca0`, base `master@bf6b05809e1c3505efd7447b33685796ca42e3b6`, limpio y con checks verdes. No es el SHA vigente después del commit documental que sigue.
- Tier 3 read-only sobre el SHA exacto `445859bdb90922410983980f2f645755d4fafca0` encontró bloqueos previos al kernel: (a) §10 exige que el segundo consumo falle, pero §11 exige recuperar un retry; falta ID de intento para distinguirlos; (b) la emisión no mantiene bloqueado `authority_ledger_head` desde replay hasta commit, permitiendo que una revocación se interponga; (c) `evaluate()->RuleDecision` no devuelve el `RulePermit` emitido; (d) un proceso reiniciado no puede recuperar el permiso de emisión porque el nuevo `permit_id`/hora genera otro draft; (e) falta binding confiable entre pin y `Path` del repositorio; (f) el permit guarda `capability_id`, pero el catálogo tiene una identidad de contrato independiente; (g) se mezclan los checkpoints Block 4 y Rule Authority, que tienen protocolos/dominios distintos; (h) el guard pierde causas STOP desconocido/reloj retrocedido al envolverlas como proveedor inválido; (i) `RuleLimits` no representa una regla REVERSIBLE sin límites. Referencias: spec §9–§11, `policy/rule_authority/{storage.py,snapshot.py,permit.py,providers.py,models.py}` y `policy/authority_ledger/storage.py`. No implementar sobre estas interfaces hasta cerrar los contratos que cambian comportamiento/autoridad.
- Se consultó a Fernando la semántica retry/single-use: propuesta recomendada `consumption_request_id` durable, mismo ID recupera, ID distinto falla. Esperar su decisión antes de alterar la API o la aceptación del spec. La pregunta independiente sobre `OVERLAY_ISSUED` sigue pendiente y bloquea el merge final, no la publicación independiente autorizada tras verificar ventana.
- Decisión pública de API también consultada: se recomienda `RuleEvaluationOutcome(decision, permit|None)` cerrado, con PERMIT↔permit y negativa↔None, porque el spec declara que `evaluate` devuelve solo `RuleDecision` y ese tipo oculta el único permit resultante. Esperar respuesta antes de cambiar la firma pública.
- GitHub vivo: #385 (head `111ec66505001087ee33618c46db46aa11713eba`) está integrado en `9b2744c33c37fdfeec87bac03ddd00c9b76e8fb5`; #387 propone revertirlo y permanece abierto/UNSTABLE porque `pisos-no-bajan` falla en `38023429466`; resolver ese PR con su owner antes de decidir el destino de #375. #375 sigue abierto/DIRTY con base antigua, pero #385 lo reemplazó: mantenerlo estacionado; si #385 permanece, cerrarlo como supersedido por su owner; solo reconstruir el delta si #387 revierte #385. No tocar ramas ni worktrees ajenos.
- El commit documental `34e91834ba9a2428d26b530fc4ebdcb3f82951e4` publicó la primera versión de esta nota. Su delta-audit exacta: APROBADO CON CAMBIOS, BLOCK 0, MAJOR 0, MINOR 3, tres LOW heredados; halló estado auto-invalidado, acción incondicional incorrecta sobre #375 y ampliación indebida del gate de overlays a publicación. El run `38040434309` terminó SUCCESS en proyecciones; policy `38040434329` fue cancelado y secret-scan `38040434389` terminó SUCCESS. La auditoría exacta de `c479e0b137d4d09c0bc29e3f9f78c9f8ea30e89c` fue APROBADO CON CAMBIOS: 0 BLOCK, 0 MAJOR, 1 MINOR y 3 LOW heredados. El MINOR pide que este handoff nombre `c479e0b1` como SHA vigente y diga que espera CI/delta-audit; no exige un commit documental previo a la auditoría. En la consulta posterior a actualizar la rama, la CI de `c479e0b1` seguía activa: policy `38040662623` y secret-scan `38040662653`; canonical projections `38040662635` pasó. La delta-audit exacta de `8c85ea70dd47bcffdc1789a66dacff727b29c341` fue APROBADO CON CAMBIOS: 0 BLOCK, 0 MAJOR, 2 MINOR y 3 LOW heredados. El primer MINOR pide ejercitar con un `TrustedCheckpointStore` real de Block 4 que el guard y el storage lo rechazan; el segundo pide registrar el SHA/corridas actuales sin volver a llamar “local” a un cambio ya publicado. Para `8c85ea70`, canonical projections `38040889836` terminó SUCCESS; policy `38040889822` y secret-scan `38040889811` seguían IN_PROGRESS en la última consulta. El commit `05ac26943d00d1b970fdbb8bcf750f0a9eedfcde` agregó la regresión de frontera Block 4→Rule Authority. Providers/checkpoint/modelos pasan `299 passed`, con `py_compile` y `git diff --check` limpios. Su auditoría exacta APROBADO CON CAMBIOS reportó 0 BLOCK, 0 MAJOR, 1 MINOR y 3 LOW heredados: la descripción de esta iteración volvía a llamar “diff de trabajo” a código ya publicado y pedía otro SHA. Esta entrada se limita a registrar los hechos de `8c85ea70` y `05ac2694`; los resultados de heads posteriores se consultan en el rollup del PR, sin un commit adicional solo para repetir su propio SHA. La CI exacta de `bc024578` (policy `38041226721`) terminó FAILURE únicamente en el piso de Identity Foundation Shadow: `1042 passed` frente al patrón anterior `1040`; 30/31 jobs pasaron. El aumento de dos corresponde a las dos regresiones de frontera de checkpoints. El piso se midió desde ese log y se alineó a 1042 en `ci/pisos.json`, `.github/workflows/policy.yml` y `docs/ci/pisos.md`. La CI exacta de `ab9dd12214894384d4c24cf21ea6f1aa7d1ec983` confirmó la corrección: policy `38042119923` SUCCESS (31/31, incluido tests-puros), canonical projections `38042119934` SUCCESS y secret-scan `38042119914` SUCCESS. El paso Identity Foundation aceptó el piso 1042. Estos son resultados históricos de `ab9dd122`; los heads posteriores se comprueban en el rollup del PR. La Tier 3 exacta de `5f804860f1314e2b5109646fec0b72add06bb0c1` fue APROBADO CON CAMBIOS: 0 BLOCK, 0 MAJOR, 1 MINOR y 3 LOW heredados. El MINOR identificó que el +2 del piso se atribuyó incorrectamente también al constructor MariaDB: el incremento real viene de `test_el_guard_nombra_el_checkpoint_de_auditoria_sin_confundirlo_con_block4` y `test_el_guard_rechaza_el_checkpoint_real_de_block4`, ambas dentro de la lista Identity Foundation. La aserción MariaDB está en `test_rule_authority_checkpoint.py`, fuera de esa lista, y se añadió a una prueba existente. Para `5f804860`, canonical projections `38042001972` estaba SUCCESS; policy `38042001975` seguía IN_PROGRESS (28 jobs SUCCESS, `faro-fase0`, `permisos-proyectos` y `tests-puros` activos), y secret-scan `38042002002` seguía IN_PROGRESS en la última consulta. La documentación de pisos separa ahora el +2 Identity del rechazo MariaDB.

## Límites de autoridad y alcance

- **Decisión de Fernando, 2026-10-10:** `OVERLAY_ISSUED` requiere ratificación vigente,
  no revocada y del mismo hash; los overlays históricos inválidos quedan en cuarentena
  permanente. Resuelve la pregunta antes pendiente; no autoriza integración por sí sola.
- **Decisión de Fernando, 2026-10-10:** el consumo lleva `consumption_request_id` durable;
  el mismo ID recupera el retry confirmado, un ID distinto se rechaza. `evaluate` devuelve
  `RuleEvaluationOutcome(decision, permit | None)`, con permiso solo en PERMIT.
- El kernel de este paso debe ignorar overlays y ratificaciones del corpus; solo usa el
  grant individual de Block 4 derivado del ledger verificado.
- No tocar el worktree `/home/fruiz/wt/jax-ledger-checkpoint` ni `codex/faro-r2`.
- No integrar hasta cerrar auditoría/CI exactas y el procedimiento de integración. Antes de
  publicar la rama o integrar, ejecutar `/home/fruiz/claude-skills/bin/ventana estado` de
  inmediato; la pregunta de OVERLAY ya está resuelta, pero no sustituye esos gates.

## Siguiente acción exacta

1. Ejecutar las cinco pruebas cross-schema contra MariaDB real con el principal provisionado; no inferir ese resultado de colección o de pruebas puras. Si el runner local sigue sin Docker, ejecutarlas en CI apropiada antes de wiring.
2. Completar el wiring del kernel `evaluate`/`consume`, cobertura de clasificación/STOP/expiración/revocación/cambio de capability/pin y mediciones O(n) p95/RSS/EXPLAIN.
3. Resolver la pregunta de OVERLAY con Fernando antes del merge final. El trabajo/publicación independiente puede continuar con ventana abierta revalidada; antes de cada publicación o integración ejecutar `/home/fruiz/claude-skills/bin/ventana estado`.

## Archivos tocados

- `policy/rule_authority/trusted_checkpoint.py`
- `policy/rule_authority/permit.py`
- `policy/rule_authority/__init__.py`
- `policy/rule_authority/storage.py`
- `policy/rule_authority/provisioning.py`
- `policy/authority_ledger/storage.py`
- `policy/rule_authority/store.py`
- `policy/rule_authority/migrations/002_enable_atomic_permits.sql`
- `policy/rule_authority/migrations/003_durable_consumption_request_id.sql`
- `policy/rule_authority/models.py`
- `tests/policy/test_faro_rule_permit.py`
- `tests/policy/test_faro_rule_authority_store.py`
- `tests/policy/test_rule_authority_checkpoint.py`
- `tests/policy/test_faro_rule_authority_storage_mariadb.py`
- `tests/policy/test_authority_ledger_storage_mariadb.py`
- `docs/superpowers/plans/2026-10-09-faro-f11-step6-checkpoint.md`
- `TRASPASO.md`

### Decisiones confirmadas y contrato de retry · 2026-10-10

- Fernando resolvió las tres preguntas públicas: `OVERLAY_ISSUED` exige ratificación vigente,
  no revocada y del mismo hash; eventos históricos inválidos permanecen en cuarentena
  permanente; `consumption_request_id` durable recupera solo el mismo retry y rechaza otro
  ID; `evaluate` devuelve `RuleEvaluationOutcome(decision, permit | None)`.
- Las menciones anteriores de esta bitácora a la pregunta de OVERLAY como pendiente describen
  el estado previo a la respuesta y no son una condición actual. La respuesta resuelve el
  bloqueo semántico; auditoría, CI y autoridad de integración siguen siendo gates separados.
- El SHA publicado de partida es `3eac3fff32c9472f7d43884c5ad007166a9db3e4`. El delta local
  de trabajo añade el outcome cerrado, el ID UUIDv7 en la proyección canónica de consumo, una
  columna durable única con trigger de validación, y la cobertura de mismo ID/ID distinto.
  El kernel todavía requiere wiring transaccional de evaluate/consume; no declarar el paso
  terminado con este incremento.
- Verificación local del delta: 185 passed en permisos, modelos, store, checkpoint y overlays;
  `py_compile`, parser de las dos sentencias de migración y `git diff --check` pasan. La suite
  MariaDB cross-schema colecciona; su ejecución local falló al arrancar el contenedor Docker.
  En la consulta del SHA remoto de partida, policy y secret-scan seguían en curso; canonical
  projections pasó. Consultar el rollup vigente, no asumir esos resultados como actuales.
- Siguiente: completar y probar el transaction owner compartido Block 4/Rule Authority, el
  kernel y la devolución/recovery del outcome; correr migración e integración MariaDB en CI;
  medir floors afectados, delta-auditar y esperar CI del SHA exacto. No integrar todavía.

### Continuación de coordinación y auditoría arquitectónica · 2026-10-10

- Fernando confirmó las decisiones anteriores. La auditoría Tier 3 read-only de `da66de55`
  rechazó activar el paso 6 mientras sigan abiertos: (1) linealización común de emisión y
  revocación Block 4, (2) recovery durable exacto de emisión/consumo, (3) binding confiable
  entre pin y path del repositorio, y (4) binding de `capability_id` con la identidad del
  contrato. También exige separar checkpoints Block 4/Rule Authority, preservar errores de
  STOP/reloj, verificar permisos legacy sin retry id, y probar ventanas de crash y carreras.
- El soporte autónomo de piso está en el commit local comunicado `4bc47e69`; cambia
  `.github/workflows/policy.yml`, `ci/pisos.json` (de 227 a 232 para el job
  `archivos-de-test-en-ci`) y `docs/ci/pisos.md`. Para evitar colisión, se acordó no publicar
  #388 hasta integrar ese soporte y reconciliar el diff acumulado. La implementación del
  kernel seguirá en esta rama sin editar esos tres archivos; si la medición obliga a cambiar
  un piso, se coordinará antes de tocarlo.
- Se corrigió el lenguaje del plan y del modelo `RuleEvaluationOutcome`: la decisión de
  OVERLAY ya no aparece como gate pendiente y el outcome se define como forma de respuesta,
  no prueba por sí sola de persistencia negativa. Tests unitarios del modelo/permiso:
  163 passed; `py_compile` y `git diff --check` pasaron. La suite MariaDB no se ejecutó.
- Siguiente secuencia: implementar primero el owner/recuperación store, luego los bindings
  de permit y pin, el evaluador puro, y finalmente `evaluate`/`consume` con transacción y
  pruebas de concurrencia/crash. No publicar hasta resolver la coordinación de pisos; no
  integrar sin auditoría/CI exactas y procedimiento reservado para `policy.yml`.
